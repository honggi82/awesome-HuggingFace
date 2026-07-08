## Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis

# arXiv:2402.14797v1[cs.CV]22Feb2024

Willi Menapace1,2,* Aliaksandr Siarohin1 Ivan Skorokhodov1 Ekaterina Deyneka1 Tsai-Shien Chen1,3,* Anil Kag1 Yuwei Fang1 Aleksei Stoliar1 Elisa Ricci2,4 Jian Ren1 Sergey Tulyakov1

Snap Inc.1 University of Trento2 UC Merced3 Fondazione Bruno Kessler4

snap-research.github.io/snapvideo

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

"Dust fills the air as off-road vehicles tear through a vast desert landscape. [...]"

"[...] corgi dog riding a bike in Times Square. It is wearing sunglasses and a beach hat."

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

"[...] a movie set where an otter serves as a film director. [...] raised paws shouting 'Action!' [...] "

"Atop dramatic cliffs, two warriors engage in a sword fight. [...]"

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

"In an empty parking lot, film a professional car drifting exhibition. [...]"

"A cowboy panda riding on the back of a lion, hand-held camera."

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

"[...] a cat donned in a chef's hat expertly kneads dough [...]"

"In the vastness of space, starships engage in a cosmic clash. [...]"

Figure 1. Samples produced by the proposed text-to-video generation method for a selection of prompts. Thanks to joint spatiotemporal video modeling, our generator can synthesize temporally coherent videos with large motion (left) while retaining the semantic control capabilities typical of large-scale text-to-video generators (right). See the Website for additional samples.

### Abstract

Contemporary models for generating images show remarkable quality and versatility. Swayed by these advantages, the research community repurposes them to generate videos. Since video content is highly redundant, we argue that naively bringing advances of image models to the video generation domain reduces motion fidelity, visual quality and impairs scalability. In this work, we build Snap Video, a video-first model that systematically addresses these challenges. To do that, we first extend the EDM framework to take into account spatially and temporally redundant pixels and naturally support video generation. Second, we show that a U-Net—a workhorse behind image generation—scales poorly when generating videos, requiring significant computational overhead. Hence, we propose a new transformer-based architecture that trains 3.31 times faster than U-Nets (and is ∼4.5 faster at inference). This allows us to efficiently train a text-to-video model with billions of

parameters for the first time, reach state-of-the-art results on a number of benchmarks, and generate videos with substantially higher quality, temporal consistency, and motion complexity. The user studies showed that our model was favored by a large margin over the most recent methods.

### 1. Introduction

Creating and sharing visual content is one of the key ways for people to express themselves in the digital world. Accessible to only professionals in the past, the capability to create [30, 40, 43, 69] and edit [6, 37, 42] images with stunning quality and realism was unlocked to everyone by the advent of large text-to-image models and their variations.

Fueled by this progress, large-scale text-to-video models [4, 13, 21, 48, 62] are rapidly advancing too. Current large-scale diffusion-based video generation frame-

∗ Work performed while interning at Snap Inc.

works are strongly rooted into their image counterparts [4, 13]. The availability of consolidated image generation architectures such as U-Nets [41] with publicly-available image-pretrained models [40] made them a logical foundation onto which to build large-scale video generators with the main architectural modifications focusing on the insertion of ad-hoc layers to capture temporal dependencies [4, 13, 21, 48, 62]. Similarly, training is performed under image-based diffusion frameworks with the model being applied both to videos and to a separate set of images to improve the diversity of the results [13, 21, 22, 48].

We argue that such an approach is suboptimal under multiple aspects which we systematically address in this work. First, image and video modalities present intrinsic differences given by the similarity of content in successive video frames [7, 13]. By analogy, image and video compression algorithms are based on vastly different approaches [33]. To address this issue, we rewrite the EDM [25] framework with a focus on high-resolution videos. Differently from past work where videos were treated as a sequence of images, we perform joint video-image training by treating images as high frame-rate videos to avoid modality mismatches introduced by the absence of the temporal dimension within purely image-based training. Second, the widely adopted U-Net [41] architecture is required to fully processes each video frame. This increases computational overhead compared to purely text-to-image models, posing a very practical limit on model scalability. The latter is a critical factor in obtaining high-quality of results [13, 21]. Extending U-Netbased architectures to naturally support spatial and temporal dimensions requires volumetric attention operations, which have prohibitive computational demands. Inability to do so affects the outputs, resulting in dynamic images or motion artifacts being generated instead of videos with coherent and diverse actions.

Following our compression analogy, we propose to leverage repetition between frames and introduce a scalable transformer architecture that treats spatial and temporal dimensions as a single, compressed, 1D latent vector. This highly compressed representation allows us to perform spatio-temporal computation jointly and enables modelling of complex motions. Our architecture is inspired by FIT [8], which we scale to billions of parameters for the first time. Compared to U-Nets, our model features a significant 3.31× reduction in training time and 4.49× reduction in inference time while achieving higher generation quality.

We evaluate Snap Video on the widely-adopted UCF101 [55] and MSR-VTT [65] datasets. Our generator shows state-of-the-art performance across the range of benchmarks with particular regard to the quality of the generated motion. Most interestingly, we performed a number of user studies against the most recent open- and closesource methods and found that according to the participants

of the study our model features photorealism comparable to Gen-2 [11], while being significantly better than Pika [1] and Floor33 [17]. Most excitedly, the preference of user-study participants favoured Snap Video by a large margin when text alignment and motion quality were assessed. Compared to Gen-2 [11] on prompt-video alignment our model was preferred in 81% of cases (80% against Pika [1], 81% against Floor33 [17]), generated most dynamic videos with most amount of motion (96% against Gen2 [11], 89% against Pika [1], 88% against Floor33 [17]) and had the best motion quality (79% against Gen-2 [11], 71% against Pika [1], 79% against Floor33 [17]).

### 2. Related Work

Video Generation Video generation is a challenging and long-studied task. Due to its complexity, a large number of works focus on modeling narrow domains [5, 9, 12, 28, 35, 44, 47, 49, 56, 58, 59, 66, 70, 71] and adopt adversarial training [5, 9, 28, 44, 47, 49, 58, 59, 71] or autoregressive generation techniques [12, 35, 56, 66, 70]. To address the narrow domain limitation, the task of text-to-video generation was proposed [34] and both autoregressive models [23, 34, 61, 63, 64] and GANs [29] emerged.

The recent success of diffusion models in the context of text-to-image generation [3, 40, 43] fostered tremendous progress in the task [2, 4, 13, 16, 17, 21, 22, 32, 48, 62, 67, 72]. ImagenVideo [21] and Make-A-Video [48] propose a deep cascade of temporal and spatial upsamplers to generate videos and jointly train their models on image and video datasets. PYoCo [13] introduces a correlated noise model to capture similarities between video frames. Video LDM [4] adopts a latent diffusion paradigm where a pretrained latent image generator and latent decoder are finetuned to generate temporally coherent videos. AnimateDiff [16] freezes a pre-trained latent image generator and trains only a newly inserted motion modeling module. These works employ U-Nets with separable spatial and temporal computation which poses a limitation on motion modeling capabilities. VideoFactory [62] improves upon this paradigm by proposing a Swapped Spatiotemporal CrossAttention that improves interactions between the spatial and temporal modalities along 3D windows.

Differently from this corpus of works which adapts the U-Net [41] architecture to the video generation task, we show that employing transformer-based FIT [8] architectures results in significant training time savings, scalability improvements, and performance increase thanks to their learnable compressed video representation. In particular, we show that the global joint spatiotemporal modeling strategy enabled by our compressed video representation results in significant improvements in temporal consistency and motion modeling capabilities.

High-Resolution Generation Different approaches have been proposed to enable the generation of high-resolution outputs. Cascaded diffusion models [3, 13, 21, 43, 48] adopt a set of independent diffusion models designed to successively upsample the results of the previous step. Latent diffusion models [2, 4, 17, 40, 72] make use of a pretrained autoencoder to encode the input into a low-dimensional set of latent vectors and learn a diffusion model on this latent representation.

A different family of methods generates high-resolution outputs end-to-end without employing cascades of models or latent diffusion. Simple Diffusion [24] and Chen [7] directly generate high-resolution images by adapting the noise schedule of the diffusion process. f-DM [14] and RDM [57] design a diffusion process that seamlessly transitions between different resolutions. MDM [15] proposes a strategy where a single model is trained to simultaneously denoise inputs at progressively higher resolutions.

In this work, we adopt a two-stage cascaded model out of two considerations: (i) it avoids temporal inconsistencies in the forms of flickering of high-frequency details that may be introduced by latent autoencoders [4], (ii) it increases model capacity with respect to an end-to-end model by creating two specialized models, one for the low resolution focusing on motion modeling and scene structure, and one for the high-resolution, focusing on high-frequency details.

Diffusion Frameworks Diffusion generative models are a set of techniques modeling generation as a pair of processes: a forward process progressively destructing a sample with noise, and a reverse process modeling generation as the progressive denoising of a sample. Different formulations of diffusion models have been proposed in the literature. Denoising Diffusion Probabilistic Models (DDPMs) [20, 50] formulate the forward and backward process as Markov chains. Score-based Generative Models (SGMs) [51, 52] model the score of the probability density function of a series of data distributions perturbed with increasing levels of noise, i.e. the direction of largest increase in the data log probability density function. An avenue of works [53, 54] generalizes DDPMs and SGMs to infinite noise levels through Stochastic Differential Equations (SDEs). In this work, we adopt the SGM framework of EDM [25] which we reformulate for the generation of highresolution videos.

### 3. Method

We propose the generation of high-resolution videos by rewriting the EDM [25] diffusion framework for highdimensional inputs and proposing an efficient transformer architecture based on FITs [8] which we scale to billions of parameters and tens of thousands input patches. Sec. 3.1 provides an introduction to the EDM framework, Sec 3.2

highlights the challenges of applying diffusion frameworks to high dimensional inputs and proposes a revisited EDMbased diffusion framework. Sec. 3.3 proposes a method to reduce the gap between image and video modalities for joint training. Finally, Sec. 3.4 describes our scalable video generation architecture, while Sec. 3.5 and Sec. 3.6 respectively describe the training and inference procedures.

#### 3.1. Introduction to EDM

Diffusion models have achieved remarkable success in image and video generation. Among the proposed frameworks, Karras et al. [25] provide a unified view of common diffusion frameworks and formulate EDM. EDM defines a variance-exploding forward diffusion process p(xσ|x) ∼ N(x,σ2I), where σ ∈ [σmin,σmax] represents the diffusion timestep coinciding with the standard deviation of the applied noise, and xσ represents the data at the current noise level. A denoiser function Dθ is learned to model the reverse process using the denoising objective:

L(Dθ) = Eσ,x,ϵ λ(σ) Dθ(xσ) − x 22 , (1)

where λ is the loss weighting function, x ∼ pdata is a data sample, ϵ is gaussian noise, and σ ∼ ptrain is sampled from a training distribution. Dθ(xσ) is defined as:

Dθ(xσ) = cout(σ)Fθ (cin(σ)xσ) + cskip(σ)xσ, (2)

where Fθ is a neural network, and cout, cskip and cin represent scaling functions. In particular, the denoising objective L(Fθ) can equivalently be expressed in terms of Fθ as:

L(Fθ) = Eσ,x,ϵ w(σ) Fθ(cin(σ)xσ) − cnrm(σ)Ftgt 22 , (3)

where Ftgt represents the training target, cnrm is a normalization factor, and w is a weighting function. These forms, derived in Appx. D, are presented in Tab. 1.

A second order Runge-Kutta sampler is proposed to reverse the diffusion process and produce sample x starting from gaussian noise xσ

max ∼ N(0,σ2maxI).

#### 3.2. EDM for High-Resolution Video Generation

EDM is originally proposed as an image generation framework and its parameters are optimized for 64 × 64px image generation. Alterations in spatial resolution or the introduction of videos with shared content between frames allow the denoising network to trivially recover a noisy frame in the original resolution with higher signal-to-noise-ratio (SNR), which the original framework was designed to see at lower noise levels. To see why, consider a noisy video xσ ∈ RT×s·H×s·W ∼ N(x,σ2I) where T is the number of frames and s is an upsampling factor. We build the corresponding clean and noisy frames at original resolution x˜,x˜σ ∈ R1×H×W by averaging values in each T × s × s block of pixels. As a consequence of averaging, the noise

EDM [25] Our

###### Training and Losses Forw. process xσ x/σin + σϵ x/σin + σϵ Training target Ftgt σx − σ2dataϵ + σ

2 data(σin−1)

###### σinσ x - σx + σ2dataϵ

2 data

Eff. loss weigh. w(σ) 1 (σ2 + σ2data)2/(σ2 + σ

σin )2

Loss weigh. λ(σ) 1/σ2data + 1/σ2 1/σ2data + 1/σ2 Network Parametrization Input scaling cin(σ) 1/

√

√

σ2data/σ2in + σ2 Output scaling cout(σ) σσ

σ2data + σ2 1/

√

2

σ2+σ2data

√ data

−σinσσdata

σ2data+σinσ2 Skip scaling cskip(σ) σ

σ2+σ2data

σinσ2data σinσ2+σ2data Target scaling cnrm(σ) 1/σdata

2 data

σ2+σ2data

√

√

σ2 + σ2data 1/σdata

σ2 + σ2data

Table 1. Definitions of functions in Eq. (1), Eq. (2) and Eq. (3) for the EDM and our proposed diffusion framework as derived in Appx. D and Appx. E, where we highlight the terms induced by the input scaling factor σin. Our framework is equivalent to EDM for σin = 1 but avoids the unstable term σ

2 data(σin−1)

σinσ x induced by σin ̸= 1 in Ftgt. This form highlights that the train target and loss weight match the v-prediction [45] framework for σdata = 1. All other framework parameters are unaltered with respect to EDM.

2

variance is reduced by a factor Ts2, i.e. x˜σ ∼ N(x˜, σ

Ts2 I), thus x˜σ has an increased signal-to-noise-ratio with respect to xσ (see Fig. 2): SNRx˜

. If pixels in each block share similar content, a typical situation in high-resolution videos, then the information in the averaged frame is useful for recovering x and can be exploited at training time by the denoiser function. This creates a traininference mismatch during the initial sampling steps as the average frame does not yet contain a well-formed signal, yet the denoiser is reliant on its presence. Thus, for best performance, any alteration to T or s should instead maintain the same signal-to-noise ratio at the original resolution for which the diffusion framework was designed.

= Ts2SNRx

σ

σ

To restore the optimal SNR at the original resolution, the magnitude of the input signal can be reduced [7] by a corresponding factor σin = s

√

T as illustrated in Fig. 2. Consequently, we redefine the forward process as p(xσ|x) ∼ N(x/σin,σ2I).

We rewrite the EDM framework to introduce the input scaling factor in Appx. E and highlight the changes in Tab. 1. We notice that a naive introduction of the scaling factor would alter the training target Ftgt in a way that makes the objective explode for small noise values (see Appx. D). We thus leverage the training objective expressed in the form of Eq. (3) to rewrite the EDM process in a way that ensures Ftgt remains unchanged, the effective loss weight w(σ) is such that it keeps the loss weight λ(σ) unchanged, cin(σ) and cnrm(σ) normalize the input and training target to have unit variance, and the framework is equivalent to the original EDM formulation for σin = 1 (see Appx. E).

|[Figure 25]|
|---|

|[Figure 26]|
|---|
|[Figure 27]|

|[Figure 28]|
|---|
|[Figure 29]|

|[Figure 30]|
|---|
|[Figure 31]|

Figure 2. Analysis of Signal-to-Noise Ratio (SNR). Top: noise σ is applied to an image. Middle: the same noise σ is applied to a 16-frames-long video x without scaling. A clean image can be easily restored by simply taking average, indicating an increased SNR. Bottom: to maintain the original SNR, we scale down the 16 frames by σin before noise application. Averaging is not able to restore the images, indicating the SNR is maintained as x˜+σϵ.

Finally, we modify the sampler according to the newly defined forward process that requires the signal component in xσ to be scaled by σin. This is achieved by dividing the Dθ(xσ) by σin and multiplying the final denoised sample x0 by σin to restore the signal magnitude.

#### 3.3. Image-Video Modality Matching

Due to the limited amount of captioned video data with respect to images, joint image-video training is widely adopted [13, 21, 22, 48] with the same diffusion process typically applied to both modalities. However, as shown in Sec. 3.2, the presence of T frames in videos calls for a different process with respect to an image with the same resolution. A possibility would be to adopt different input scaling factors for the two modalities. We argue that this solution is undesirable in that it increases the complexity of the framework and image training would not foster the denoising model to learn temporal reasoning, a fundamental capability of a video generator. To sidestep these issues while using a unified diffusion process, we match the image and video modalities by treating images as T frames videos with infinite frame-rate and introduce a variable frame-rate training procedure blending the gap between the image and video modalities.

#### 3.4. Scalable Video Generator

U-Nets [41] have shown success in video generation where they are typically augmented with temporal attention or convolutions for modeling the temporal dimension [4, 13, 21, 22, 48]. However, such an approach requires a full UNet forward pass for each of the T video frames, rapidly becoming prohibitively expensive (see Fig. 3a). These factors pose a practical limit on model scalability—a primary factor in achieving high generation quality [13, 17, 21, 48]—and

|[Figure 32]| |
|---|---|
| | |

|[Figure 33]| |
|---|---|
| | |

|[Figure 34]| |
|---|---|
| | |

|[Figure 35]| |
|---|---|
| | |

Separable Block Learnable Compression

Separable Block

| | |
|---|---|
| | |

Spatial

Spatial

Joint Spatiotemporal Block

Temporal

Temporal

| | |Learnable Decompression| | |
|---|---|---|---|---|
| | | | | |

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

Separable Spatiotemporal Model Our Joint Spatiotemporal Model

(a) Computational Paradigms for Videos

Text Enc.

Joint S.T. Block Compression Decompression

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

Noise Std. Framerate Resolution

XAttn + FF

| |
|---|

| |
|---|

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

|FIT Block<br><br>+FF<br><br>+FF<br><br>+FF<br><br>tTok.Tok.<br><br>tTok.Tok.<br><br>tTok.Tok.<br><br>| |
|---|---|
|SAttn<br><br>SAttn<br><br>SAttn<br><br>LatenLatent<br><br>LatenLatent<br><br>LatenLatent<br><br>| |

SAttn+FF

SAttn+FF

SAttn+FF

LatentTok.LatentTok.

LatentTok.LatentTok.

LatentTok.LatentTok.

LatentTok.LatentTok.

| |
|---|

| |
|---|

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

XAttn + FF XAttn + FF

| | | | | | | | | | | | | | | |a|tc| |h| | |o|k|e|n|s| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Pixels to patches Patches to pixels

(b) Snap Video FIT Architecture

- Figure 3. (a-left) U-Net-based text-to-image architectures are adapted to do video generation by inserting temporal layers applied sequentially with spatial layers, creating separable spatiotemporal blocks. Spatial computation is repeated for each frame independently, limiting scalability. (a-right) Our scalable transformer-based model jointly performs spatial and temporal computation on a learnable compressed video representation for improved motion modeling and scalability. (b) The proposed Snap Video FIT architecture. Given a noisy input video xσ, the model estimates the denoised video xˆσ by recurrent application of FIT blocks. Each block reads information from the patch tokens into a small set of latent tokens on which computation is performed. The results are written to the patch tokens. Conditioning information in the form of text embeddings, noise level σ, frame-rate ν and resolution r is provided through an additional read operation.

similarly limit possibilities for joint spatio-temporal modeling [62]. We argue that treating spatial and temporal modeling in a separable way [4, 13, 21, 48] causes motion artifacts, temporal inconsistencies or generation of dynamic images rather than videos with vivid motion. Video frames, however, contain spatially and temporally redundant content that is amenable to compression [33]. We argue that learning and operating on a compressed video representation and jointly modeling the spatial and temporal dimensions are necessary steps to achieve the scalability and motion-modeling capabilities required for high-quality video generation.

FITs [8] are efficient transformer-based architectures that have recently been proposed for high-resolution image synthesis and video generation. Their main idea, summarized in Fig. 3 is that of learning a compressed representation of their input through a set of learnable latent tokens and of focusing computation on this learnable latent space, allowing input dimensionality to grow with little performance penalty. First, FITs perform patchification of the input and produce a sequence of patch tokens which are later divided into groups. A set of latent tokens is then instantiated and a sequence of computational blocks is applied. Each block first performs a cross attention “read” operation between latent tokens and conditioning signals such as the diffusion timestep, then an additional groupwise “read” cross attention operation between latent and patch tokens of corresponding groups to compress patch information, applies a series of self attention operations to the latent tokens, and performs a groupwise “write” cross attention operation that decompresses information in the latent tokens to update the patch tokens. Finally, the patch tokens are pro-

jected back to the pixel space to form the output. Self conditioning is applied on the set of latent tokens to preserve the compressed video representation computed in previous sampling steps.

While promising, these architectures have not yet been scaled to the billion-parameters size of state-of-the-art UNet-based video generators, nor they have been applied to high-resolution video generation. In the following, we highlight the architectural considerations necessary to achieve these goals. Temporal modeling is a fundamental aspect of a high-quality video generator. FITs produce patch tokens by considering three dimensional patches of size Tp×Hp×Wp spanning both the spatial and temporal dimensions. We find values of Tp > 1 to limit temporal modeling performance, so we consider patches spanning the spatial dimension only. In addition, similarly to patches, FITs group patch tokens into groups spanning both the temporal and spatial dimensions, and perform cross attention operations group by group. We observe that the temporal size of each group should be configured so that each group covers all T video frames for best temporal modeling. Furthermore, videos contain more information with respect to images due to the presence of the temporal dimension, thus we increase the number of latent tokens representing the size of the compressed space in which joint spatiotemporal computation is performed. Finally, FITs make use of local layers which perform self attention operations on patch tokens corresponding to the same group. We find this operation to be computationally expensive for large amounts of patch tokens (147.456 for our largest resolution) and replace it with a feed forward module after each cross attention “read” or “write” operation.

Our model makes use of conditioning information represented by a sequence of conditioning tokens to control the generation process. In addition to the token representing the current σ, to enable text conditioning, we introduce a

- T5-11B [39] text encoder extracting text embeddings from the input text. To support variable video framerates and large differences in resolution and aspect ratios in the training data, we concatenate additional tokens representing the framerate and original resolution of the current input.

To generate high-resolution outputs, we implement a model cascade consisting of a first-stage model producing 36×64px videos and a second-stage upsampling model producing 288 × 512px videos. To improve upsampling quality, we corrupt the second-stage low-resolution inputs with a variable level of noise during training [21, 43] and during inference apply a level of noise to the first-stage outputs obtained by hyperparameter search.

We present detailed model hyperparameters in Appx. A.

#### 3.5. Training

We train Snap Video using the LAMB [68] optimizer with a learning rate of 5e−3, a cosine learning schedule and a total batch size of 2048 videos and 2048 images, achievable thanks to our scalable video generator architecture. We train the first-stage model over 550k steps and finetune the second-stage model on high-resolution videos starting from the first-stage model weights for 370k iterations. Following the observations in Sec 3.2, we pose σin = s

√

T. Considering videos with T = 16 frames and the original 64px resolution for which EDM was designed, we set σin = 4 for the first-stage and σin = 32 for the second-stage model.

We present training details and parameters in Appx. B.

#### 3.6. Inference

We produce video samples from gaussian noise and userprovided conditioning information using the deterministic

- sampler of [25] and our two-stage cascade. We use 256 sampling steps for the first-stage and 40 for the second-stage model, and employ classifier free guidance [19] to improve text-video alignment (see Appx. C.1) unless otherwise specified. We find dynamic thresholding [43] and oscillating guidance [21] to consistently improve sample quality.

### 4. Evaluation

In this section, we perform evaluation of Snap Video against baselines and validate our design choices. Sec. 4.1 introduces the employed datasets, Sec. 4.2 defines the evaluation protocol, Sec. 4.3 shows ablations of our diffusion framework and architectural choices, Sec. 4.4 quantitatively compares our method to state-of-the-art large-scale video generators and Sec. 4.5 performs qualitative evaluation. We complement evaluation by showcasing samples in the Appendix and Website.

FID ↓ FVD ↓ CLIPSIM ↑ Train Thr. ↓ Inf. Thr. ↓

U-Net 85M [10] 8.21 45.94 0.2319 133.2 49.6 U-Net 284M [10] 4.90 23.76 0.2391 230.3 105.1 Snap Video FIT 500M 3.07 27.79 0.2459 69.5 23.4 Snap Video FIT 3.9B 2.51 12.31 0.2579 526.0 130.4

- Table 2. Performance of different architectures and model sizes on our internal dataset in 64×36px resolution. We observe strong performance gains with scaling and note that FITs present better performance with improved speed with respect to U-Nets. Train and inference throughputs in ms/video/GPU.

σdata σin Imgs. as Videos FID ↓ FVD ↓ CLIPSIM ↑

- (i) 0.5 1.0 ✓ 6.58 39.95 0.2370
- (ii) 0.5 4.0 ✓ 4.03 31.00 0.2449 (iv) 1.0 2.0 ✓ 4.45 34.89 0.2428

- (iii) 1.0 1/4.0 ✗ 3.50 24.88 0.2469 Ours 1.0 4.0 ✓ 3.07 27.79 0.2459

- Table 3. Ablation of different diffusion process configurations varying σdata, input scaling σin, and treatment of images as infinite-framerate videos, evaluated on our internal dataset in 64 × 36px resolution.
- 4.1. Datasets

We train our models on an internal dataset consisting of 1.265M images and 238k hours of videos, each with a corresponding text caption. Due to the difficulty in acquiring high-quality captions for videos, we develop a video captioning model that we use to produce synthetic video captions for the portion of videos in the dataset missing such annotation.

We make use of the following datasets for evaluation which are never observed during training: UCF-101 [55] is a video dataset containing 13.320 320 × 240px Youtube videos from 101 action categories.

MSR-VTT [65] is a dataset containing 10.000 320×240px web-crawled videos, each manually annotated with 20 text captions. The test set contains 2.990 videos and 59.800 corresponding captions.

#### 4.2. Evaluation Protocol

To validate the choices operated on the diffusion framework and on model architecture, present method ablations performed in 64 × 36px resolution using the first-stage model only, and compute FID [18], FVD [60] and CLIPSIM [63] metrics against the test set of our internal dataset on 50k generated videos.

To evaluate our method against baselines, we follow the protocols highlighted in [4, 13, 32, 48, 62, 72] for zero-

shot evaluation on the UCF-101 [55] and MSR-VTT [65] datasets. We generate 16 frames videos in 512 × 288px resolution at 24fps for all settings. We evaluate both at the native 512 × 288px resolution with 16:9 aspect ratio and in the 288 × 288px square aspect ratio typically employed on these benchmarks. We note that the evaluation protocols of [4, 13, 32, 48, 62, 72] present different choices regarding the number of generated samples, distribution of class labels, choice of text prompts. We make use of the following evaluation parameters:

Zero-shot UCF-101 [55] We generate 10.000 videos [4, 62] sampling classes with the same distribution as the original dataset. We produce a text prompt for each class label [13] and compute FVD [60] and Inception Score [46]. Zero-shot MSR-VTT [65] We generate a video sample for each of the 59.800 test prompts [13, 48] and compute CLIP-FID [27] and CLIPSIM [63].

To provide a more complete performance assessment and compare against state-of-the-art closed-source methods not reporting results for these benchmarks, we perform a user study evaluating photorealism, video-text-alignment and, most importantly, the quantity and quality of the generated motion, important characteristics of a video generator that may signal the generation of dynamic images, i.e. videos with dim motion, or motion artifacts rather than videos with vivid and high-quality motion.

#### 4.3. Ablations

To evaluate the proposed FIT architecture, we consider the

- U-Net of [10], which we adapt to the video generation setting by interleaving temporal attention operations. We consider two U-Net variants of different capacities and a smaller variant of our FIT to evaluate the scalability of both architectures. We detail the architectures in Appx. A and show results in Tab. 2.

Our 500M parameters FIT trains 3.31× faster than the baseline 284M parameters U-Net, performs inference 4.49× faster and surpasses it in terms of FID and CLIPSIM. In addition, both FITs and U-Nets show strong performance gains with scaling. Our largest FIT scales to 3.9B parameters with only a 1.24× increase in inference time with respect to the 284M U-Net.

To evaluate the choices operated on our diffusion framework, we ablate different configurations of the diffusion process using our 500M FIT architecture. We produce the following variations: (i) the original EDM framework, (ii) our scaled diffusion framework with EDM σdata, (iii) our framework with a reduced value of σin, (iv) our framework with images not treated as infinite-frame-rate videos. Our framework improves over EDM under all metrics (i) and shows benefits in setting σdata = 1, an effect that we attribute to the creation of a training target and loss weight-

FVD ↓ FID ↓ IS ↑

CogVideo [23] (Chinese) 751.3 - 23.55 CogVideo [23] (English) 701.6 - 25.27 MagicVideo [72] 655 - LVDM [17] 641.8 - Video LDM [4] 550.6 - 33.45 VideoFactory [62] 410.0 - Make-A-Video [48] 367.2 - 33.00 PYoCo [13] 355.2 - 47.46

Snap Video (288 × 288 px) 260.1 39.0 38.89 Snap Video (512 × 288 px) 200.2 28.1 38.89

Table 4. Zero-shot evaluation results on UCF101 [55].

CLIP-FID ↓ FVD ↓ CLIPSIM ↑

NUWA [64] (Chinese) 47.68 - 0.2439 CogVideo [23] (Chinese) 24.78 - 0.2614 CogVideo [23] (English) 23.59 - 0.2631 MagicVideo [72] - 998 LVDM [17] - - 0.2381 Latent-Shift [2] 15.23 - 0.2773 Video LDM [4] - - 0.2929 VideoFactory [62] - - 0.3005 Make-A-Video [48] 13.17 - 0.3049 PYoCo [13] 9.73 - -

Snap Video (288 × 288 px) 8.48 110.4 0.2793 Snap Video (512 × 288 px) 9.35 104.0 0.2793

Table 5. Zero-shot evaluation results on MSR-VTT [65].

ing matching the widely used v-prediction formulation of Salimans et al. [45] (see Tab. 1). Using σin < s

√

T (see Sec. 3.2) impairs performance (iii). Finally, treating images as infinite-frame-rate videos consistently improves FID.

#### 4.4. Quantitative Evaluation

We perform comparison of Snap Video against baselines on the UCF101 [55], and MSR-VTT [65] datasets respectively in Tab. 4 and Tab. 5. FID and FVD video quality metrics show improvements over the baselines which we attribute to the employed diffusion framework and joint spatiotemporal modeling performed by our architecture. On UCF101, our method produces the second-best IS of 38.89, demonstrating good video-text alignment. While our method surpasses Make-A-Video [48] on UCF101, we note that it produces a lower CLIPSIM score on MSR-VTT. We attribute this behavior to the use of T5 [39] text embeddings in place of the commonly used CLIP [38] embeddings which were observed [43] to produce higher text-image alignment despite similar CLIPSIM.

To provide a comprehensive evaluation we run a user study to evaluate photorealism, video-text alignment, quantity of motion and quality of motion, important aspects of a video generator. Three publicly-accessible state-of-the-art

|[Figure 56]|[Figure 57]|[Figure 58]|
|---|---|---|

ImagenVideo

|[Figure 59]|[Figure 60]|[Figure 61]|
|---|---|---|

SnapVideo

"A teddy bear running in New York City."

|[Figure 62]|[Figure 63]|[Figure 64]|
|---|---|---|

SnapVideoVideoLDM

|[Figure 65]|[Figure 66]|[Figure 67]|
|---|---|---|

"An astronaut cooking with a pan and fire in the kitchen, high definition, 4k."

|[Figure 68]|[Figure 69]|[Figure 70]|[Figure 71]|[Figure 72]|
|---|---|---|---|---|

SnapVideoPYoCo

|[Figure 73]|[Figure 74]|[Figure 75]|
|---|---|---|

"A huge dinosaur skeleton is walking in a golden wheat field on a bright sunny day."

| | |[Figure 76]| | |
|---|---|---|---|---|

Make-A-Video

|[Figure 77]|[Figure 78]|[Figure 79]|
|---|---|---|

SnapVideo

"A golden retriever eating ice cream on a beautiful tropical beach at sunset, high resolution"

- Figure 4. Qualitative results comparing Snap Video to state-of-the-art video generators on publicly available samples. While baseline methods present motion artifacts (top-left, top-right, bottom-right) or produce dynamic images (bottom-left), our method produces more temporally coherent motion. Best viewed in the Website.

Photorealism Video-Text Align. Mot. Quant. Mot. Qual.

Gen-2 [11] 44.3 81.0 96.0 78.7 PikaLab [1] 61.5 80.3 89.2 70.5 Floor33 [17] 76.3 80.9 88.0 79.1

Table 6. User study on photorealism, video-text alignment, motion quantity and quality against publicly-accessible video generators on 65 dynamic scene prompts. % of votes in favor of our method.

video generators are considered: Gen-2 [11], PikaLabs [1] and Floor33 [17]. We filter a set of 65 prompts from [31] describing scenes with vivid motions, and generate a video for each method with default options. We ask the participants to express preference between paired samples from Snap Video and each baseline, gathering votes from 5 users for each sample. Results are shown in Tab. 6 and video

- samples provided along with the employed prompt list in Appx. C.2 and in the Website. Our method produces results with photorealism comparable to Gen-2, while surpassing PikaLab and Floor33, and outperforms all baselines with respect to video-text alignment. Most importantly, we note that baselines often produce dynamic images, i.e. videos with dim motion, or videos with motion artifacts, a finding we attribute to the challenges in modeling large motion. In contrast, our method, thanks to the joint spatiotemporal modeling approach, produces vivid and high-quality motion as shown by the motion metrics.

#### 4.5. Qualitative Evaluation

In this section, we perform qualitative evaluation of our framework. In Fig. 4, Appx. C.3 and the Website, we

present qualitative results comparing our method to stateof-the-art generators [4, 13, 21, 48] on samples publicly released by the authors. While such prompts might have been selected to highlight strengths of the baselines, our method produces more photorealistic samples aligned to the text descriptions. Most importantly, our samples present vivid and high-quality motion avoiding flickering artifacts that are present in the baselines due to temporal inconsistencies. We accompany qualitative evaluation with a user study performed on the same set of samples in Appx. C.2.

### 5. Conclusions

In this work, we highlight the shortcomings of diffusion processes and architectures commonly used in text-to-video generation, and systematically address them by treating videos as first-class citizens. First, we propose a modification to the EDM [25] diffusion framework for the generation of high-resolution videos and treat images as high frame-rate videos to avoid image-video modality mismatches. Second, we replace U-Nets [41] with efficient transformer-based FITs [8] which we scale to billions of parameters. Thanks to their learnable compressed representation of videos, they significantly improve training times, scalability and performance with particular regards to temporal consistency and motion modeling capabilities due to the joint spatiotemporal modeling on the compressed representation. When evaluated on UCF101 [55] and MSRVTT [65] and in user studies, Snap Video attains state-ofthe-art performance with particular regard to the quality of the modeled motion.

### 6. Acknowledgements

We would like to thank Oleksii Popov, Artem Sinitsyn, Anton Kuzmenko, Vitalii Kravchuk, Vadym Hrebennyk, Grygorii Kozhemiak, Tetiana Shcherbakova, Svitlana Harkusha, Oleksandr Yurchak, Andrii Buniakov, Maryna Marienko, Maksym Garkusha, Brett Krong, Anastasiia Bondarchuk for their help in the realization of video presentations, stories and graphical assets, Colin Eles, Dhritiman Sagar, Vitalii Osykov, Eric Hu for their supporting technical activities, Maryna Diakonova for her assistance with annotation tasks.

### References

- [1] Pika lab discord server. https://www.pika.art/. Accessed: 2023-11-01. 2, 8, 5
- [2] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv, 2023. 2, 3, 7
- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. ArXiv, 2022. 2, 3
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2023. 1, 2, 3, 4, 5, 6, 7, 8, 14

- [5] Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei A Efros, and Tero Karras. Generating long videos of dynamic scenes. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2
- [6] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1
- [7] Ting Chen. On the importance of noise scheduling for diffusion models. arXiv, 2023. 2, 3, 4
- [8] Ting Chen and Lala Li. Fit: Far-reaching interleaved transformers. arXiv, 2023. 2, 3, 5, 8, 1
- [9] Aidan Clark, Jeff Donahue, and Karen Simonyan. Efficient video generation on complex datasets. arXiv, 2019. 2
- [10] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 6, 7
- [11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 8, 5
- [12] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh.

- Long video generation with time-agnostic vqgan and timesensitive transformer. In Proceedings of the European Conference of Computer Vision (ECCV), 2022. 2
- [13] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE International Conference on Computer Vision (ICCV),

2023. 1, 2, 3, 4, 5, 6, 7, 8, 14

- [14] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Miguel Angel Bautista, and Josh Susskind. f-dm: A multi-stage diffusion model via progressive signal transformation. International Conference on Learning Representations (ICLR), 2023. 3
- [15] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Josh Susskind, and Navdeep Jaitly. Matryoshka diffusion models. arXiv, 2023. 3
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv, 2023. 2
- [17] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv, 2023. 2, 3, 4, 7, 8, 5
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 6
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv, 2022. 6, 2
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 3
- [21] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. arXiv, 2022. 1, 2, 3, 4, 5, 6, 8, 14
- [22] Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In ICLR Workshop on Deep Generative Models for Highly Structured Data, 2022. 2, 4, 3
- [23] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv, 2022. 2, 7
- [24] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. Simple diffusion: End-to-end diffusion for high resolution images. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. 3
- [25] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 3, 4, 6, 8
- [26] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv, 2015. 1
- [27] Tuomas Kynk¨a¨anniemi, Tero Karras, Miika Aittala, Timo Aila, and Jaakko Lehtinen. The role of imagenet classes in

- fr´echet inception distance. In International Conference on Learning Representations (ICLR), 2023. 7
- [28] Alex X. Lee, Richard Zhang, Frederik Ebert, P. Abbeel, Chelsea Finn, and S. Levine. Stochastic adversarial video prediction. arXiv, abs/1804.01523, 2018. 2
- [29] Yitong Li, Martin Min, Dinghan Shen, David Carlson, and Lawrence Carin. Video generation from text. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2018. 2
- [30] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. arXiv, 2023. 1
- [31] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv, 2023. 8, 2, 6, 7
- [32] Z. Luo, D. Chen, Y. Zhang, Y. Huang, L. Wang, Y. Shen, D. Zhao, J. Zhou, and T. Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 6, 7
- [33] Siwei Ma, Xinfeng Zhang, Chuanmin Jia, Zhenghui Zhao, Shiqi Wang, and Shanshe Wang. Image and video compression with neural networks: A review. IEEE Transactions on Circuits and Systems for Video Technology, 2019. 2, 5
- [34] Gaurav Mittal, Tanya Marwah, and Vineeth N. Balasubramanian. Sync-draw: Automatic video generation using deep recurrent attentive architectures. In Proceedings of the 25th ACM International Conference on Multimedia, 2017. 2
- [35] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid. CCVS: Context-aware controllable video synthesis. In Advances in Neural Information Processing Systems (NeurIPS),

2021. 2

- [36] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. 2019. 1
- [37] Chenyang QI, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023. 1
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 7
- [39] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a

- unified text-to-text transformer. Journal of Machine Learning Research (JMLR), 2022. 6, 7, 14
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. arXiv, 2021. 1, 2, 3
- [41] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015. 2, 4, 8, 1
- [42] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1
- [43] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic textto-image diffusion models with deep language understanding. arXiv, 2022. 1, 2, 3, 6, 7
- [44] Masaki Saito, Shunta Saito, Masanori Koyama, and Sosuke Kobayashi. Train sparsely, generate densely: Memoryefficient unsupervised training of high-resolution temporal gan, 2020. 2
- [45] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations (ICLR), 2022. 4, 7, 11
- [46] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems (NeurIPS), 2016. 7
- [47] Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Mostgan-v: Video generation with temporal motion styles. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [48] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. arXiv, 2022. 1, 2, 3, 4, 5, 6, 7, 8
- [49] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [50] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (ICML), 2015. 3
- [51] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems (NeurIPS), 2019. 3
- [52] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 3

- [53] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models, 2021. 3
- [54] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021. 3
- [55] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv, 2012. 2, 6, 7, 8, 3, 10
- [56] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. Unsupervised learning of video representations using lstms. In International Conference on Machine Learning (ICML), 2015. 2
- [57] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv, 2023. 3
- [58] Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N Metaxas, and Sergey Tulyakov. A good image generator is what you need for high-resolution video synthesis. In International Conference on Learning Representations (ICLR), 2021. 2
- [59] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [60] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv, 2018. 6, 7
- [61] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and D. Erhan. Phenaki: Variable length video generation from open domain textual description. In International Conference on Learning Representations (ICLR), 2023. 2
- [62] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv, 2023. 1, 2, 5, 6, 7, 14
- [63] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. ArXiv,

2021. 2, 6, 7

- [64] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation. In Proceedings of the European Conference of Computer Vision (ECCV), 2022. 2, 7
- [65] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2, 6, 7, 8

- [66] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv, 2021. 2
- [67] Sheng-Siang Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, Jianlong Fu, Gong Ming, Lijuan Wang, Zicheng Liu, Houqiang Li, and Nan Duan. Nuwa-xl: Diffusion over diffusion for extremely long video generation. In Annual Meeting of the Association for Computational Linguistics, 2023. 2
- [68] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. In International Conference on Learning Representations (ICLR),

2020. 6, 1

- [69] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. 1
- [70] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. Magvit: Masked generative video transformer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [71] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. In International Conference on Learning Representations (ICLR), 2022. 2
- [72] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv, 2023. 2, 3, 6, 7

## Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis Supplementary Material

U-Net 85M U-Net 284M

Resolution 16 × 64 × 40 16 × 64 × 40 Base channels 128 192 Channel multiplier [1, 2, 2, 3] [1, 2, 3, 4] Number of residual blocks 2 2 Attention resolutions [32, 16, 8] [32, 16, 8] Attention heads channels 64 64 Conditioning channels 768 768 Label dropout 0.10 0.10 Dropout 0.10 0.10 Use scale shift norm True True

Table 7. Architectural details of our U-Net baselines.

### A. Architecture details

In this section, we discuss the details of the architectures employed in this work. Tab. 7 details the hyperparameters of the UNet [41], while Tab. 8 shows the hyperparameters of Snap Video FIT. Note that to ensure divisibility by larger powers of 2, for models producing outputs in 64 × 36px resolution we introduce a 4px vertical padding, yielding a model resolution of 64 × 40px. With respect to vanilla FITs [8], our model presents architectural differences that we found beneficial for high-resolution video generation. First, we configure each patch to span over a single frame and each patch group to extend in the temporal dimension to the full extent of the video. We find these modifications to improve temporal modeling. We keep the spatial dimension of the patch to 4 × 4px as we found larger patches to degrade spatial modeling capabilities of the model. These settings produce a large number of 147.456 input patches when processing high-resolution videos. We find the use of local attention layers to be computationally expensive when modeling high-resolution videos, so we replace it with feedforward operations after each cross attention layer. Lastly, our FIT makes use of a large number of latent tokens, a parameter determining the amount of compression applied to the video representation. With respect to typical FIT configurations using no more than 256 latent tokens, we find it beneficial to increase their number to 768 when modeling videos to enable additional temporal information to be stored in the latent tokens.

### B. Training details

In this section, we present training details for the models trained in this work, which we summarize in Tab. 9. We train our model using the LAMB [68] optimizer with a learning rate of 5e−3 and a cosine learning schedule over

FIT 500M FIT 3.9B FIT 3.9B Up.

Input size 16 × 64 × 40 16 × 64 × 40 16 × 512 × 288 Patch size 1 × 4 × 4 1 × 4 × 4 1 × 4 × 4 Group size 16 × 5 × 4 16 × 5 × 4 16 × 18 × 16 Total patch tokens 2.560 2.560 147.456 Cross att. feedforward ✓ ✓ ✓ Patch tokens channels 768 1024 1024 Latent tokens count 512 768 768 Latent tokens channels 1024 3072 3072 FIT blocks count 6 6 6 FIT block global layers 4 4 4 FIT block local layers 0 0 0 Patch att. heads channels 48 64 64 Latent att. heads channels 64 128 128 Self conditioning ✓ ✓ ✓ Self conditioning prob. 0.9 0.9 0.9 Label dropout 0.1 0.1 0.1 Dropout 0.1 0.1 0.1 Positional embeddings learnable learnable learnable Conditioning channels 768 1024 1024

- Table 8. Architectural details of Snap Video FIT models.

U-Nets FIT 3.9B FIT 3.9B Up.

Optimizer Adam [26] LAMB LAMB Learning rate 1e−4 5e−3 5e−3 Learning rate decay cosine cosine cosine Steps 550k 550k 370k Batch size 4096 4096 320 Samples seen 2.25B 2.25B 118M Beta [0.9, 0.999] [0.9, 0.99] [0.9, 0.99] Weight decay 0.01 0.01 0.01 Warmup steps 10.000 10.000 10.000 EMA halflife steps 6.000 6.000 6.000

- Table 9. Training hyperparameters for our experiments.

550k steps with 10k warmup steps. To achieve stable training in our largest model we find it important to introduce the following mechanisms. We set β = [0.9,0.99], a weight decay of 0.01, gradient clipping and a total batch size of 2048 videos and 2048 images. We adopt dropout with probability 0.1 in the feedforward transformer layers following self attention operations. Finally, we use an exponential moving average for the model’s weights with a base decay halflife of 6k steps.

We implement our model in PyTorch [36] and perform all experiments on Nvidia A100 GPUs.

### C. Additional Evaluation Results and Details

In this section, we provide additional evaluation results for our model. In Sec. C.1, we perform ablations on the sampler

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80

70

60

FVD

50

40

30

20

10

0.255 0.260 0.265 0.270 0.275 0.280 CLIPSIM

- Figure 5. FVD and CLIPSIM on our internal dataset in 64 × 36px resolution as a function of the classifier free guidance weight. Points represent weights of [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16].

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

16 32 64 128 256 512 1024 Steps

10

20

30

40

50

60

FVD

- Figure 6. FVD on our internal dataset in 64 × 36px resolution as a function of the sampling steps and sampler type.

parameters. Sec. C.2 presents user study results. In Sec. C.3 and Sec. C.4 we show samples against baselines and additional samples from our method. In Sec. C.5, we describe how to generate longer videos at high framerate. Finally, Sec. C.6 presents evaluation details for UCF101 [55].

#### C.1. Sampler Parameters Ablations

Choices in the sampling parameters can affect the performance of diffusion models significantly. In Fig. 5, we show the impact of classifier free guidance [19] on model’s performance. We produce 10k samples using the first-stage model and report FVD and CLIPSIM at various guidance values, computed on 10k samples with a 40 step deterministic sampler. We find that classifier free guidance can improve both FVD and CLIPSIM, but notice increased sample saturation at high classifier free guidance scales. We adopt dynamic thresholding and oscillating classifier free guidance [43] which we find effective in reducing the phenomenon.

In Fig. 6, we evaluate performance of the model under the same setting, but varying the number of sampling steps. We find that the model produces high-quality samples already at 64 steps and that FVD improves until 256 steps.

Photorealism Video-Text Align. Mot. Quant. Mot. Qual.

Imagen Video [21] 66.9 54.3 49.4 56.3 PYoCo [13] 63.3 64.9 57.1 63.9 Video LDM [4] 61.7 64.4 62.2 65.8 Make-A-Video [48] 80.0 82.2 82.2 75.6

Table 10. User study on photorealism, video-text alignment, motion quantity and quality on publicly available samples from closed-source methods. % of votes in favor of our method.

#### C.2. User Studies

To complement the evaluation, we run a user study on a set of publicly released samples from Make-A-Video [48], PYoCo [13], Video LDM [4] and Imagen Video [21]. For each method, we collect publicly available samples and prompts, generate corresponding samples from Snap Video, and ask participants to express a preference in terms of photorealism, video-text alignment, motion quantity and quality, collecting 5 votes for each sample. Results are shown in Tab. 10 and samples are provided in Fig. 4, Fig. 7 and the Website. Our method shows increased photorealism with respect to the baselines, presents higher video-text alignment with respect to all methods except Imagen Video and consistently surpasses baselines in terms of motion quality (see flickering artifacts and temporally inconsistent backgrounds in scenes with large camera motion), a finding we attribute to our joint spatiotemporal modeling approach.

#### C.3. Qualitative Results Against Baselines

- In Fig. 7 and the Website, we present qualitative results of our method against baselines on samples publicly released by the authors of Make-A-Video [48], PYoCo [13], Video LDM [4] and Imagen Video [21]. Our method produces videos with natural motion and can handle scenes with large motion and camera changes while preserving temporal consistency. On the other hand, we observe that baselines often present flickering artifacts and temporally inconsistent objects in case of large motion.
- In Fig. 8 and the Website, we provide qualitative results

comparing our method to publicly accessible state-of-theart video generators including Gen-2 [11], PikaLab [1] and Floor33 [17]. Our method produces results that are more aligned to the prompts and, differently from the baselines, which often produce dynamic images, our method produces temporally coherent videos with large amounts of motion.

#### C.4. Additional Qualitative Results

Complex Prompts We present in Fig. 9, Fig. 10 and the Website additional samples generated by our method on a set of prompts gathered from ChatGPT, Make-A-Video [48], PYoCo [13], Video LDM [4], Imagen Video [21], and the Evalcrafter [31] benchmark. Our method can synthe-

size a large number of different concepts. Most importantly, it can produce videos with challenging motion including large camera movement, POV videos, and videos of fastmoving objects. Notably, the method maintains temporal consistency and avoids video flickering artifacts. We ascribe these motion modeling capabilities to the joint spatiotemporal modeling performed by our architecture.

Novel Views We qualitatively evaluate the capabilities of the proposed method in producing novel views of objects. To do so, we build prompts using one of the following templates Camera circling around a ⟨object⟩, Camera orbiting around a ⟨object⟩ or Camera moving around a ⟨object⟩, where ⟨object⟩ represents a placeholder for the object to generate. Results are shown in Fig. 11 and the Website. We

- find that the model is capable of generating plausible novel views of objects, suggesting that it possesses an understanding of the 3D geometry of the modeled objects.

Samples Diversity We qualitatively assess the capabilities of the method of generating diverse samples for each prompt. In Fig. 12 and the Website we show three samples produced for a set of prompts and note that the method is capable of producing diverse samples.

UCF 101 In Fig. 13 and the Website, we present qualitative results produced by our mehthod for zero-shot UCF101 [55] evaluation.

- C.5. Hierarchical Video Generation

We train our method to generate videos with a fixed number of frames and variable framerate. We exploit this characteristic and devise a hierarchical generation strategy to increase video duration and framerate by conditioning generation on previously generated frames. In particular, to condition generation on already available frames, we adopt the reconstruction guidance method of Ho et al. [22]. We define a hierarchy of progressively increasing framerates and start by autoregressively generating a video of the desired length at the lowest framerate, at each step using the last generated frame as the conditioning. Subsequently, for each successive framerate in the hierarchy, we autoregressively generate a video of the same length but conditioning the model on all frames that have already been generated at the lower framerates. We show samples in the Website.

- C.6. Zero-Shot UCF101 Evaluation

UFC101 [55] is a dataset of low-resolution Youtube videos. To better match our generated outputs to its distribution, we condition the model to produce videos with a low original resolution through the resolution conditioning mechanism (see Sec. 3.4). Following [13], since UCF101 class labels do not always have sufficiently descriptive names, we produce a prompt for each class which we report in the following: applying eye makeup, applying lipstick, a person

shooting with a bow, baby crawling, gymnast performing on a balance beam, band marching, baseball pitcher throwing baseball, a basketball player shooting basketball, dunking basketball in a basketball match, bench press in a gym, a person riding a bicycle, billiards, a woman using a hair dryer, a kid blowing candles on a cake, body weight squats, a person bowling on bowling alley, boxing punching bag, boxing training on speed bag, swimmer doing breast stroke, brushing teeth, a person doing clean and jerk in a gym, cliff diving, bowling in cricket match, batting in cricket match, cutting in kitchen, diver diving into a swimming pool from a springboard, drumming, two fencers have fencing match indoors, field hockey match, gymnast performing on the floor, group of people playing frisbee on the playground, swimmer doing front crawl, golfer swings and strikes the ball, haircuting, a person hammering a nail, an athlete performing the hammer throw, an athlete doing handstand push up, an athlete doing handstand walking, massagist doing head massage to man, an athlete doing high jump, group of people racing horse, person riding a horse, a woman doing hula hoop, man and woman dancing on the ice, athlete practicing javelin throw, a person juggling with balls, a young person doing jumping jacks, a person skipping with jump rope, a person kayaking in rapid water, knitting, an athlete doing long jump, a person doing lunges exercise in a gym, a group of soldiers marching in a parade, mixing in the kitchen, mopping floor, a person practicing nunchuck, gymnast performing on parallel bars, a person tossing pizza dough, a musician playing the cello in a room, a musician playing the daf drum, a musician playing the indian dhol, a musician playing the flute, a musician playing the guitar, a musician playing the piano, a musician playing the sitar, a musician playing the tabla drum, a musician playing the violin, an athlete jumps over the bar, gymnast performing pommel horse exercise, a person doing pull ups on bar, boxing match, push ups, group of people rafting on fast moving river, rock climbing indoor, a person lifting on a rope in a gym, several people rowing a boat on the river, a man and a woman are salsa dancing, young man shaving beard with razor, an athlete practicing shot put throw, a teenager skateboarding, skier skiing down, jet ski on the water, a person is skydiving in the sky, soccer player juggling football, soccer player doing penalty kick in a soccer match, gymnast performing on still rings, sumo wrestling, surfing, kids swing at the park, a person playing table tennis, a person doing TaiChi, a person playing tennis, an athlete practicing discus throw, trampoline jumping, typing on computer keyboard, a gymnast performing on the uneven bars, people playing volleyball, walking with dog, a person standing doing pushups on the wall, a person writing on the blackboard, a person at a Yo-Yo competition.

|[Figure 80]|[Figure 81]|[Figure 82]|[Figure 83]|[Figure 84]|
|---|---|---|---|---|

|[Figure 85]|[Figure 86]|[Figure 87]|
|---|---|---|

ImagenVideo

SnapVideoPYoCo

|[Figure 88]|[Figure 89]|[Figure 90]|
|---|---|---|

|[Figure 91]|[Figure 92]|[Figure 93]|
|---|---|---|

SnapVideo

"Flying through an intense battle between pirate ships in a stormy ocean." "A cyborg koala dj in front of a turntable, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy, intricate, neon light, soft light, smooth, sharp focus, illustration."

|[Figure 94]|[Figure 95]|[Figure 96]|[Figure 97]|[Figure 98]|
|---|---|---|---|---|

|[Figure 99]|[Figure 100]|[Figure 101]|
|---|---|---|

Make-A-Video

SnapVideoVideoLDM

|[Figure 102]|[Figure 103]|[Figure 104]|
|---|---|---|

|[Figure 105]|[Figure 106]|[Figure 107]|
|---|---|---|

SnapVideo

"A knight riding on a horse through the countryside"

"A horse galloping through van Gogh's Starry Night."

|[Figure 108]|[Figure 109]|[Figure 110]|[Figure 111]|[Figure 112]|
|---|---|---|---|---|

|[Figure 113]|[Figure 114]|[Figure 115]|
|---|---|---|

ImagenVideo

SnapVideoPYoCo

|[Figure 116]|[Figure 117]|[Figure 118]|
|---|---|---|

|[Figure 119]|[Figure 120]|[Figure 121]|
|---|---|---|

SnapVideo

"A panda taking a selfie."

"A very happy fuzzy panda dressed as a chef eating pizza in the New York street food truck."

|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]|[Figure 126]|
|---|---|---|---|---|

|[Figure 127]|[Figure 128]|[Figure 129]|
|---|---|---|

Make-A-Video

SnapVideoVideoLDM

|[Figure 130]|[Figure 131]|[Figure 132]|
|---|---|---|

|[Figure 133]|[Figure 134]|[Figure 135]|
|---|---|---|

SnapVideo

"A dog wearing virtual reality goggles in sunset, 4k, high resolution."

"A blue unicorn flying over a mystical land"

|[Figure 136]|[Figure 137]|[Figure 138]|
|---|---|---|

ImagenVideo

|[Figure 139]|[Figure 140]|[Figure 141]|
|---|---|---|

SnapVideo

"A teddy bear washing dishes."

|[Figure 142]|[Figure 143]|[Figure 144]|
|---|---|---|

SnapVideoVideoLDM

|[Figure 145]|[Figure 146]|[Figure 147]|
|---|---|---|

"Two raccoons are playing drum kit in NYC Times Square, 4k, high resolution."

|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|[Figure 152]|
|---|---|---|---|---|

SnapVideoPYoCo

|[Figure 153]|[Figure 154]|[Figure 155]|
|---|---|---|

"A guineapig painter is painting on the canvas, anthro, very cute kid's film character, concept artwork [...]"

|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|[Figure 160]|
|---|---|---|---|---|

Make-A-Video

|[Figure 161]|[Figure 162]|[Figure 163]|
|---|---|---|

SnapVideo

"A panda playing on a swing set"

- Figure 7. Comparison of Snap Video to publicly released samples from Make-A-Video [48], PYoCo [13], Video LDM [4] and Imagen Video [21]. Our method produces temporally coherent motion while avoiding video flickering. Best viewed on the Website.

|[Figure 164]|[Figure 165]|[Figure 166]|
|---|---|---|

Gen-2Floor33

|[Figure 167]|[Figure 168]|[Figure 169]|
|---|---|---|

SnapVideoPikaLab

|[Figure 170]|[Figure 171]|[Figure 172]|
|---|---|---|

|[Figure 173]|[Figure 174]|[Figure 175]|
|---|---|---|

"A group of teenagers skateboard in an empty parking lot."

|[Figure 176]|[Figure 177]|[Figure 178]|
|---|---|---|

Gen-2Floor33

|[Figure 179]|[Figure 180]|[Figure 181]|
|---|---|---|

SnapVideoPikaLab

|[Figure 182]|[Figure 183]|[Figure 184]|
|---|---|---|

|[Figure 185]|[Figure 186]|[Figure 187]|
|---|---|---|

"A musician strums his guitar, serenading the moonlit night."

|[Figure 188]|[Figure 189]|[Figure 190]|
|---|---|---|

Gen-2Floor33

|[Figure 191]|[Figure 192]|[Figure 193]|
|---|---|---|

SnapVideoPikaLab

|[Figure 194]|[Figure 195]|[Figure 196]|
|---|---|---|

|[Figure 197]|[Figure 198]|[Figure 199]|
|---|---|---|

"A horse gallops through a field, kicking up dust with its hooves."

|[Figure 200]|[Figure 201]|[Figure 202]|
|---|---|---|

Gen-2Floor33

|[Figure 203]|[Figure 204]|[Figure 205]|
|---|---|---|

SnapVideoPikaLab

|[Figure 206]|[Figure 207]|[Figure 208]|
|---|---|---|

|[Figure 209]|[Figure 210]|[Figure 211]|
|---|---|---|

"Teenage girl in red skirt enters a green-lit cave."

|[Figure 212]|[Figure 213]|[Figure 214]|
|---|---|---|

Gen-2Floor33

|[Figure 215]|[Figure 216]|[Figure 217]|
|---|---|---|

SnapVideoPikaLab

|[Figure 218]|[Figure 219]|[Figure 220]|
|---|---|---|

|[Figure 221]|[Figure 222]|[Figure 223]|
|---|---|---|

"A couple shares a milkshake with two straws at a retro diner."

|[Figure 224]|[Figure 225]|[Figure 226]|
|---|---|---|

Gen-2Floor33

|[Figure 227]|[Figure 228]|[Figure 229]|
|---|---|---|

SnapVideoPikaLab

|[Figure 230]|[Figure 231]|[Figure 232]|
|---|---|---|

|[Figure 233]|[Figure 234]|[Figure 235]|
|---|---|---|

"Three travelers wander through the bustling streets of Times Square, taking in the dazzling lights and energy."

- Figure 8. Comparison of Snap Video to the publicly accessible state-of-the-art video generators Gen-2 [11], PikaLab [1] and Floor33 [17]. Rather than producing dynamic images, our method generates videos with large amounts of temporally coherent motion. Best viewed on the Website.

|[Figure 236]|[Figure 237]|[Figure 238]|[Figure 239]|[Figure 240]|[Figure 241]|
|---|---|---|---|---|---|

"A trio of fashionable, beret-clad cats sips coffee at a chic Parisian cafe., time-lapse photography."

"In a chic urban kitchen, a cat donned in a chef's hat expertly kneads dough on a marble countertop. Cinematic close-up shots capture the feline's precise movements. Camera smoothly orbits around the cat, showcasing culinary prowess."

|[Figure 242]|[Figure 243]|[Figure 244]|[Figure 245]|[Figure 246]|[Figure 247]|
|---|---|---|---|---|---|

"A fox dressed in suit dancing in park."

"Large motion, a group of six rabbits dressed in Victorian attire gathers for a garden tea party."

|[Figure 248]|[Figure 249]|[Figure 250]|[Figure 251]|[Figure 252]|[Figure 253]|
|---|---|---|---|---|---|

"Two elephants are playing on the beach and enjoying a delicious beef stroganoff meal. Camera rotates anticlockwise."

"3 sheep enjoying spaghetti together."

|[Figure 254]|[Figure 255]|[Figure 256]|[Figure 257]|[Figure 258]|[Figure 259]|
|---|---|---|---|---|---|

"Within a vibrant market scene, otters expertly juggle an array of colorful fruits, creating a lively spectacle. Cinematically zoom in to showcase their dexterity and the vibrant hues of the fruits, with the 4K resolution highlighting the rich textures."

"In a high-tech control room, otters operate an imaginary spaceship console, embarking on an interstellar adventure. Cinematic lighting effects enhance the futuristic setting, and the camera executes quick cuts to showcase the excitement of their space journey."

|[Figure 260]|[Figure 261]|[Figure 262]|[Figure 263]|[Figure 264]|[Figure 265]|
|---|---|---|---|---|---|

"In a quaint library setting, otters don scholarly attire and engage in a heated debate over miniature books. Cinematic tracking shots following their animated gestures, highlighting the comical intensity of the intellectual otter discourse."

"In a sunlit meadow, otters don tiny bowties and engage in a formal tea party on a miniature table. Cinematic close-ups of their adorable expressions, camera smoothly circling the scene."

|[Figure 266]|[Figure 267]|[Figure 268]|[Figure 269]|[Figure 270]|[Figure 271]|
|---|---|---|---|---|---|

"A blue pickup truck with a rhinoceros in its flatbed."

"A cute golden hamster throwing punches wearing pair of boxing gloves in a boxing ring."

|[Figure 272]|[Figure 273]|[Figure 274]|[Figure 275]|[Figure 276]|[Figure 277]|
|---|---|---|---|---|---|

"A horse gallops through a field, kicking up dust with its hooves."

"Three hamsters run on a wheel, exercising in their cage."

|[Figure 278]|[Figure 279]|[Figure 280]|[Figure 281]|[Figure 282]|[Figure 283]|
|---|---|---|---|---|---|

"A painting of a wise old owl in a tweed jacket puffs on a pipe."

"With the style of Van Gogh, A young couple dances under the moonlight by the lake."

|[Figure 284]|[Figure 285]|[Figure 286]|[Figure 287]|[Figure 288]|[Figure 289]|
|---|---|---|---|---|---|

"An astronaut feeding ducks on a sunny afternoon, reflection from the water."

"An astronaut playing with sparklers for Diwali, photorealistic."

###### Figure 9. Additional samples generated from Snap Video on a collection of prompts gathered from ChatGPT, baseline methods and the Evalcrafter [31] benchmark. Best viewed on the Website.

|[Figure 290]|[Figure 291]|[Figure 292]|[Figure 293]|[Figure 294]|[Figure 295]|
|---|---|---|---|---|---|

"Teddy bear walking down 5th Avenue, front view, beautiful sunset, close up, high definition, 4k." "In Macro len style, a photograph of a knight in shining armor holding a basketball"

|[Figure 296]|[Figure 297]|[Figure 298]|[Figure 299]|[Figure 300]|[Figure 301]|
|---|---|---|---|---|---|

"Amidst the ruins of a post-apocalyptic city, a lone robot scavenger sifts through debris, its sensors scanning for salvageable materials. Cinematic 4K shots capturing the gritty atmosphere, camera tilting and panning to emphasize the desolation."

"Within a futuristic factory, robots assemble intricate machinery. Render detailed close-upsof robotic movements and precision. Use a mix of steady pans and close-ups to convey the complexity and efficiency of the assembly process."

|[Figure 302]|[Figure 303]|[Figure 304]|[Figure 305]|[Figure 306]|[Figure 307]|
|---|---|---|---|---|---|

"Large motion, a flag with a dinosaur on it."

"The brides of Dracula."

|[Figure 308]|[Figure 309]|[Figure 310]|[Figure 311]|[Figure 312]|[Figure 313]|
|---|---|---|---|---|---|

"A couple enjoys a romantic gondola ride in Venice, Italy." "Drove viewpoint, fireworks above the Parthenon."

|[Figure 314]|[Figure 315]|[Figure 316]|[Figure 317]|[Figure 318]|[Figure 319]|
|---|---|---|---|---|---|

"first person perspective, a four-piece band on a stage in front of a small crowd"

"Within a clockmaker's workshop, intricate gears turn with precision. Employ precise rotations and close-ups to reveal the mechanical beauty, emphasizing the fine craftsmanship of the timepiece."

|[Figure 320]|[Figure 321]|[Figure 322]|[Figure 323]|[Figure 324]|[Figure 325]|
|---|---|---|---|---|---|

"In a potter's studio, skilled hands mold clay into a delicate sculpture. Utilize sweeping arcs to highlight the shaping process, emphasizing the intricate details emerging from the artist's touch."

"Noodles falling into a bowl of soup."

|[Figure 326]|[Figure 327]|[Figure 328]|[Figure 329]|[Figure 330]|[Figure 331]|
|---|---|---|---|---|---|

"A man cruises through the city on a motorcycle, feeling the adrenaline rush." "A motorcycle race through the city streets at night."

|[Figure 332]|[Figure 333]|[Figure 334]|[Figure 335]|[Figure 336]|[Figure 337]|
|---|---|---|---|---|---|

"Vintage cars race along a winding coastal highway at sunset. Cinematically capture the gleaming chrome details and classic aesthetics, while the camera smoothly glides alongside, framing the cars against the scenic backdrop."

"In the heart of a city's underground racing scene, modified cars with roaring engines compete in illegal street races. Showcase the intense acceleration, drifts around corners, and the adrenaline-pumping energy of the night races."

|[Figure 338]|[Figure 339]|[Figure 340]|[Figure 341]|[Figure 342]|[Figure 343]|
|---|---|---|---|---|---|

"A man lifts weights at the gym, working on his strength."

"A child in the air while jumping on a trampoline, hand-held camera."

###### Figure 10. Additional samples generated from Snap Video on a collection of prompts gathered from ChatGPT, baseline methods and the Evalcrafter [31] benchmark. Best viewed on the Website.

|[Figure 344]|[Figure 345]|[Figure 346]|[Figure 347]|[Figure 348]|[Figure 349]|
|---|---|---|---|---|---|

"Camera circling around a dog." "Camera circling around a happy squirrel."

|[Figure 350]|[Figure 351]|[Figure 352]|[Figure 353]|[Figure 354]|[Figure 355]|
|---|---|---|---|---|---|

"Camera orbiting around the face of an elderly men with a white beard." "Camera circling around the tower of Pisa."

|[Figure 356]|[Figure 357]|[Figure 358]|[Figure 359]|[Figure 360]|[Figure 361]|
|---|---|---|---|---|---|

"Camera circling around a sports car." "Camera orbiting around a truck."

|[Figure 362]|[Figure 363]|[Figure 364]|[Figure 365]|[Figure 366]|[Figure 367]|
|---|---|---|---|---|---|

"Camera moving around a parked airplane." "Camera moving around a parked helicopter."

|[Figure 368]|[Figure 369]|[Figure 370]|[Figure 371]|[Figure 372]|[Figure 373]|
|---|---|---|---|---|---|

"Camera circling around a steam locomotive." "Camera orbiting around a miniature train model."

|[Figure 374]|[Figure 375]|[Figure 376]|[Figure 377]|[Figure 378]|[Figure 379]|
|---|---|---|---|---|---|

"Camera circling around a chair."

"Camera circling around a chair made of ice."

|[Figure 380]|[Figure 381]|[Figure 382]|[Figure 383]|[Figure 384]|[Figure 385]|
|---|---|---|---|---|---|

"Camera circling around an elegant round table." "Camera circling around a western saloon."

|[Figure 386]|[Figure 387]|[Figure 388]|[Figure 389]|[Figure 390]|[Figure 391]|
|---|---|---|---|---|---|

"Camera circling around a stunning watch." "Camera orbiting around a marble statue masterpiece, black background."

|[Figure 392]|[Figure 393]|[Figure 394]|[Figure 395]|[Figure 396]|[Figure 397]|
|---|---|---|---|---|---|

"Camera circling around a car made of vegetables." "Camera orbiting around a vase with fruit in it."

|[Figure 398]|[Figure 399]|[Figure 400]|[Figure 401]|[Figure 402]|[Figure 403]|
|---|---|---|---|---|---|

"Camera circling around a plate with delicious sushi in it." "Camera circling around a house made of sushi."

###### Figure 11. Videos produced by Snap Video for prompts eliciting circular camera motion around each object. Best viewed on the Website.

|[Figure 404]|[Figure 405]|[Figure 406]|[Figure 407]|[Figure 408]|[Figure 409]|
|---|---|---|---|---|---|

|[Figure 410]|[Figure 411]|[Figure 412]|[Figure 413]|[Figure 414]|[Figure 415]|
|---|---|---|---|---|---|

|[Figure 416]|[Figure 417]|[Figure 418]|[Figure 419]|[Figure 420]|[Figure 421]|
|---|---|---|---|---|---|

"A cat wearing sunglasses and working as a lifeguard at a pool."

"Amidst a lush meadow, otters don aprons and chef hats, skillfully preparing a miniature sushi feast on a lily pad. Capture the detailed sushi-making process with dynamic close-ups, and the camera executing a graceful orbit around the culinary otters to showcase their precision."

|[Figure 422]|[Figure 423]|[Figure 424]|[Figure 425]|[Figure 426]|[Figure 427]|
|---|---|---|---|---|---|

|[Figure 428]|[Figure 429]|[Figure 430]|[Figure 431]|[Figure 432]|[Figure 433]|
|---|---|---|---|---|---|

|[Figure 434]|[Figure 435]|[Figure 436]|[Figure 437]|[Figure 438]|[Figure 439]|
|---|---|---|---|---|---|

"An astronaut cooking with a pan and fire in the kitchen, high definition, 4k." "In macro lens style, a photograph of a knight in shining armor holding a basketball."

|[Figure 440]|[Figure 441]|[Figure 442]|[Figure 443]|[Figure 444]|[Figure 445]|
|---|---|---|---|---|---|

|[Figure 446]|[Figure 447]|[Figure 448]|[Figure 449]|[Figure 450]|[Figure 451]|
|---|---|---|---|---|---|

|[Figure 452]|[Figure 453]|[Figure 454]|[Figure 455]|[Figure 456]|[Figure 457]|
|---|---|---|---|---|---|

"A man with a skull face in flames walking around Piccadilly circus." "A burning volcano.."

|[Figure 458]|[Figure 459]|[Figure 460]|[Figure 461]|[Figure 462]|[Figure 463]|
|---|---|---|---|---|---|

|[Figure 464]|[Figure 465]|[Figure 466]|[Figure 467]|[Figure 468]|[Figure 469]|
|---|---|---|---|---|---|

|[Figure 470]|[Figure 471]|[Figure 472]|[Figure 473]|[Figure 474]|[Figure 475]|
|---|---|---|---|---|---|

"A dirt bike navigates through a dense forest trail. 4K close-up of the bike maneuvering through foliage, camera follows the rider's perspective, creating an immersive experience of the off-road adventure."

"Follow a mountain biker descending a rugged trail. Utilize a mix of drone shots and helmet-mounted cameras to capture the breathtaking landscape,detailed bike maneuvers, and the adrenaline-fueled journey down the mountain in cinematic 4K."

###### Figure 12. Diversity in samples produced by Snap Video for the same prompt. See additional samples on the Website.

|[Figure 476]|[Figure 477]|[Figure 478]|[Figure 479]|[Figure 480]|[Figure 481]|
|---|---|---|---|---|---|

"Young man shaving beard with razor." "Brushing teeth."

|[Figure 482]|[Figure 483]|[Figure 484]|[Figure 485]|[Figure 486]|[Figure 487]|
|---|---|---|---|---|---|

"A person kayaking in rapid water." "Rock climbing indoor."

|[Figure 488]|[Figure 489]|[Figure 490]|[Figure 491]|[Figure 492]|[Figure 493]|
|---|---|---|---|---|---|

"A person doing pull ups on bar." "A person doing lunges exercise in a gym."

|[Figure 494]|[Figure 495]|[Figure 496]|[Figure 497]|[Figure 498]|[Figure 499]|
|---|---|---|---|---|---|

"Two fencers have fencing match indoors." "Drumming."

|[Figure 500]|[Figure 501]|[Figure 502]|[Figure 503]|[Figure 504]|[Figure 505]|
|---|---|---|---|---|---|

"A kid blowing candles on a cake." "A person writing on the blackboard."

###### Figure 13. Samples generated by Snap Video for zero-shot evaluation on UCF101 [55]. Best viewed on the Website.

### D. Derivation of EDM Denoising Objective

We derive the denoising objective L(Fθ) expressed in terms of Fθ for the EDM framework where we modify the forward process introducing the input scaling factor σin:

L(Fθ) = Eσ,x,ϵ w(σ) Fθ(cin(σ)xσ) − cnrm(σ)Ftgt 22 , (4)

We start from the denoising objective L(Dθ) as in the original formulation:

x σin

+ σϵ) − x 22 , (5)

L(Dθ) = Eσ,x,ϵ λ(σ) Dθ(

Where we recall the definition of Dθ:

Dθ(xσ) = cout(σ)Fθ (cin(σ)xσ) + cskip(σ)(xσ). (6)

We also recall the definitions of cin(σ), cout(σ), cskip(σ) , λ(σ):

σ2data σ2 + σ2data

σ2 + σ2data σ2σ2data

σ · σdata σ2 + σ2data

1

, λ(σ) =

, cout(σ) =

, cskip(σ) =

. (7)

cin(σ) =

σ2 + σ2data

Inserting the definition of Dθ and of cin(σ), cout(σ), cskip(σ) into Eq. (4) we obtain:

x σin

x σin

+σϵ) − x 22 (8)

Eσ,x,ϵ λ(σ) cskip(σ)(

+σϵ) + cout(σ)Fθ cin(σ)(

σ2data σ2 + σ2data

σ2 + σ2data σ2σ2data

x σin

σσdata σ2 + σ2data Fθ cin(σ)(

##### x σin

+σϵ) − x 22 (9)

= Eσ,x,ϵ

+σϵ +

σ2 + σ2data σσdata

σdata σ σ2 + σ2data

x σin

x σin

x 22 (10)

= Eσ,x,ϵ 1 ·

+σϵ + Fθ cin(σ)(

+σϵ) −

σx − σ2dataϵ σdata σ2 + σ2data

x σin

x σin

σdata σ σ2 + σ2data

2 2 (11)

= Eσ,x,ϵ 1 · Fθ cin(σ)(

(x −

+σϵ) −

) −

σ2data(σin−1)

σinσ x + σx − σ2dataϵ σdata σ2 + σ2data

x σin

2 2 . (12)

= Eσ,x,ϵ 1 · Fθ cin(σ)(

+σϵ) −

From which follows that:

σ2data(σin − 1) σinσ

1 σdata σ2 + σ2data

Ftgt = σx − σ2dataϵ +

x, cnrm(σ) =

###### , w(σ) = 1. (13)

2 data(σin−1)

Note that the training target has a spurious term σ

σinσ x which approaches infinity as σ approaches 0.

From this formulation we also notice the link between EDM and the v-prediciton framework. First, the training target

consists in a rescaled and negated v-prediction objective with v = σ2dataϵ−σx. Second the loss weight equals to a reweighted 1 + SNR v-prediction framework [45] weighting:

λ(σ) =

σ2 + σ2data σσdata 2

=

1 σ2data

+

1 σ2

1 σ2data

###### + SNR. (14)

=

Thus, when σdata = 1, EDM is equivalent to the v-prediciton formulation. Starting from these observations, we rewrite the framework so that it exhibits a well-formed Ftgt for all values of σ by avoiding the spurious term σ

2 data(σin−1)

##### σinσ x.

- E. Derivation of Our Diffusion Framework We start the derivation of our diffusion framework by imposing that the training target equals the original EDM v-prediction

2 data(σin−1)

objective for all values of σin, without the spurious term σ

σinσ x affecting the original formulation for σin ̸= 1 (see Appx. D).

Ftgt = v = σ2dataϵ − σx. (15) We then derive cnrm(σ) such that cnrm(σ)Ftgt, the function approximated by Fθ, has unit variance:

Varx,ϵ cnrm(σ)Ftgt = 1 (16) Varx,ϵ cnrm(σ) σ2dataϵ − σx = 1 (17)

1 Varx,ϵ σ2dataϵ − σx

cnrm(σ)2 =

(18)

1 σ2data(σ2data + σ2)

cnrm(σ)2 =

(19)

1 σdata (σ2data + σ2)

. (20)

cnrm(σ) =

Following standard normalization practices, we define cin(σ) so that the model input has unit variance:

x σin

Varx,ϵ cin(σ)(

+ σϵ) = 1 (21)

cin(σ)2 =

cin(σ)2 =

cin(σ) =

1 Varx,ϵ σ x

(22)

+ σϵ

in

1

(23)

σ2data

σ2in + σ2

1

. (24)

σ2data

σ2in + σ2

To derive the remaining framework components, we first recall the definition of our forward process:

and note that x can be recovered from xσ and v as:

x σin

+ σϵ, (25)

xσ =

xσ − σσ2

##### v

, (26)

data

x =

##### σin + σσ22

1

data

and consequently

2 data

σ2dataxσ − (σ

σin + σ2)x σ

. (27)

v =

To recover cskip(σ) and cout(σ) we note from the definition of Ftgt = v and the loss expressed in Eq. (4) that as it approaches zero the following holds:

cnrm(σ)Ftgt = Fθ(cin(σ)xσ) (28)

v = Fθ(cin(σ)xσ)

(29)

cnrm(σ)

###### v = σdata σ2 + σ2dataFθ(cin(σ)xσ). (30)

Substituting Eq. (30) into Eq. (26) we obtain:

xσ − σσ2

##### v

(31)

data

Dθ(xσ) = x =

##### σin + σσ22

1

data

σ

σ2datav 1

xσ 1

(32)

−

=

σin + σσ22

σin + σσ22

data

data

σ

σdata σ2 + σ2dataFθ(cin(σ)xσ)

xσ 1

(33)

−

=

σin + σσ22

##### σin + σσ22

1

data

data

= cskip(σ)xσ + cout(σ)Fθ(cin(σ)xσ), (34) from which we recognize:

σ2 + σ2data σ2data + σinσ2

σinσ2data σinσ2 + σ2data

. (35)

, cout(σ) = −σinσσdata

cskip(σ) =

We set the loss weight λ(σ) to the same value as EDM:

1 σ2data

λ(σ) =

1 σ2

. (36)

+

To recover w(σ), inserting the definition of Dθ (Eq. (6)) and of cin(σ), cout(σ), cskip(σ), λ(σ) into Eq. (4) we obtain:

x σin

+σϵ) + cout(σ)Fθ cin(σ)xσ − x 22 (37)

Eσ,x,ϵ λ(σ) cskip(σ)(

σ2 + σ2data σ2data + σinσ2Fθ cin(σ)xσ − x 22 (38)

σinσ2data σinσ2 + σ2data

x σin

= Eσ,x,ϵ λ(σ)

+σϵ) − σinσσdata

(

##### σ

λ(σ) (σ2 + σ

cnrm(σ)Fθ cin(σ)xσ + σ2x − σ2dataσϵ 22 (39)

= Eσ,x,ϵ

2 data

σin )2

##### σ2

λ(σ) cnrm(σ)2 Fθ cin(σ)xσ + cnrm(σ)(σx − σ2dataϵ) 22 (40)

= Eσ,x,ϵ

2 data

(σ2 + σ

σin )2

(σ2 + σ2data)2 (σ2 + σ

Fθ cin(σ)xσ + cnrm(σ)Ftgt 22 (41)

= Eσ,x,ϵ

2 data

σin )2

= Eσ,x,ϵ w(σ) Fθ cin(σ)xσ + cnrm(σ)Ftgt 22 , (42) from which:

(σ2 + σ2data)2 (σ2 + σ

. (43)

w(σ) =

2 data

σin )2

In conclusion, the proposed diffusion framework maintains the training target Ftgt equal to the original EDM training target for all values of the input scaling factor σin, ensuring that Fθ learns the same denoising function while giving control on the quantity of the signal present in the input. In addition, our framework preserves the original loss weight λ(σ), giving control over the input scaling factor without affecting the weight of the loss over the different noise levels.

### F. Discussion

In this section, we describe the limitations of our framework (see Sec. F.1) and discuss societal impact (see Sec. F.2).

#### F.1. Limitations

Snap Video presents limitations which we discuss in this section.

Text Rendering We find our framework to often spell text incorrectly. We attribute this finding to a lack of highquality videos depicting text matched by the exact description of the displayed text. Automated pipelines for OCR can be employed on the training dataset to address this issue.

Object count Similarly, the generator may not render the requested number of entities, especially when the cardinality of the objects is high. The behavior can be explained by the difficulties in learning correct object counts from video data, where descriptions can be noisy, objects can enter and exit the scene and the camera can move widely, changing the cardinality of objects in each frame.

Positional understanding While the generator can place objects in positions requested by the prompts, we find it can not reliably synthesize videos corresponding to prompts entailing complex positional relationships between multiple entities such as “A stack of three cubes: the top one is blue, the bottom one green and the middle one is red”.

Stylization We find that our model can generate stylized content (see Fig. 9), but may present failure cases where the style specification is ignored or the stylized contents only translate in the scene rather than being animated. We attribute this finding to the lack of model training on a filtered set of data presenting high aesthetic scores which we find to contain textual descriptions related to artistic and visual styles with higher probability.

Negation Some challenging prompts such as “A glass of juice next to a plate with no bananas in it” may lead the model to ignore the negation, resulting in the generation of all entities.

Block artifacts Videos may contain content with a very large amount of motion, leading to a greater difficulty in compressing its content to a latent representation of fixed size. In such situations we find that the model may produce patch tokens that do not blend together in a perfect manner, resulting in some visible patches in the videos, akin to video compression artifacts.

Resolution Our two-stage model cascade generates videos in 512 × 288px resolution. We note that generating video content aligning to the given prompt and presenting temporally coherent motion are the most critical problems in video generation, and possible artifacts in these categories are already visible in 512 × 288px resolution, as shown in our comparisons to baselines. We also note that cascaded model stages are independently trained and agnostic to the

employed previous-stage generator. Thus, given an upsampler from 512 × 288px to a higher resolution, any improvement shown in 512×288px resolution with respect to baselines is expected to produce higher resolution results of correspondingly improved quality. We consider the integration of additional cascade stages as an interesting venue for future work.

#### F.2. Societal Impact

Text-to-video generative models are evolving rapidly [4, 13, 21, 62] and hold promise to empower users with new and powerful ways to express their creativity once accessible only to trained experts such as artists and digital content creators. With such improvements comes a greater risk that generated results may be perceived as real with the potential for nefarious individuals to generate harmful or deceiving content. Our model is exposed to a broad range of concepts during training and makes use of a T5 [39] text encoder that was trained on unfiltered internet data, making it necessary to guard it against such possible uses. In addition, the model generates data following its training data distribution which implies that potential biases that may be present in the dataset can be reflected in the model outputs. To avoid misuse, we do not make the model publicly accessible and plan to put in place data cleaning, prompt filtering and output filtering techniques and watermarking as additional safeguards.

