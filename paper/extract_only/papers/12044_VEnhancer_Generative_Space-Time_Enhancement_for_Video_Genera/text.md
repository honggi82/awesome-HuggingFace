# arXiv:2407.07667v1[cs.CV]10Jul2024

## VEnhancer: Generative Space-Time Enhancement for Video Generation

Jingwen He1,2 Tianfan Xue1 Dongyang Liu2 Xinqi Lin2 Peng Gao2 Dahua Lin1 Yu Qiao2 Wanli Ouyang1,2, Ziwei Liu3,

1The Chinese University of Hong Kong, 2Shanghai AI Laboratory, 3S-Lab, Nanyang Technological University https://vchitect.github.io/VEnhancer-project/

### Abstract

We present VEnhancer, a generative space-time enhancement framework that improves the existing text-to-video results by adding more details in spatial domain and synthetic detailed motion in temporal domain. Given a generated low-quality video, our approach can increase its spatial and temporal resolution simultaneously with arbitrary up-sampling space and time scales through a unified video diffusion model. Furthermore, VEnhancer effectively removes generated spatial artifacts and temporal flickering of generated videos. To achieve this, basing on a pretrained video diffusion model, we train a video ControlNet and inject it to the diffusion model as a condition on low frame-rate and low-resolution videos. To effectively train this video ControlNet, we design space-time data augmentation as well as video-aware conditioning. Benefiting from the above designs, VEnhancer yields to be stable during training and shares an elegant end-to-end training manner. Extensive experiments show that VEnhancer surpasses existing state-of-the-art video super-resolution and space-time super-resolution methods in enhancing AIgenerated videos. Moreover, with VEnhancer, exisiting open-source state-of-theart text-to-video method, VideoCrafter-2, reaches the top one in video generation benchmark – VBench.

### 1 Introduction

With the advances of text-to-image (T2I) generation [35, 34, 10, 14] and large-scale text-video paired datasets [1], there has been a surge of progress in the field of text-to-video (T2V) generative models [15, 11, 19, 3, 42, 39, 8, 9, 16, 44]. These developments enable users to generate compelling videos through textual descriptions of the desired content. One common solution [19, 3, 42, 2, 16] to obtain high-quality videos is to adopt cascaded pipelines, which stacks several video diffusion models (DM), including T2V, temporal super-resolution (T-SR) and spatial super-resolution (S-SR) DMs. Such a pipeline significantly reduces computation cost when generating high-resolution and high-frame-rate videos. However, it poses several issues. First, it might be redundant and timeconsuming to enhance videos in spatial and temporal axes separately with different models, as spatial and temporal super-resolution are strongly correlated tasks. Second, the proposed diffusionbased T-SR/S-SR in [3, 19, 2, 42, 31] could only handle fixed interpolation ratio (i.e., predicting 3 frames between two consecutive frames) or fixed upscaling factors (i.e., 4×), thus the flexibility and functionality are limited. Third, training T-SR/S-SR using synthesized video pairs might result in inferior generalization ability as it could only generate low-level details without fundamentally understanding the semantics and structures of video contents. Therefore, such models only improve

Corresponding authors.

Preprint. Under review.

[Figure 1]

[Figure 2]

Videocrafter2 +VEnhancer

A raccoon is playing the electronic guitar

[Figure 3]

[Figure 4]

Clown fish swimming through the coral reef.

- Figure 1: The enhanced screenshots for AI-generated videos (from VideoCrafter-2 [9]). First row: VEnhancer is able to reconstruct the strings of guitar, and regenerate realistic fur of raccoon. Second row: VEnhancer could make the blurry background very sharp as well as enhance the color, which makes the whole picture vivid. Zoom in for best view.

the spatial or temporal resolution with high fidelity, but struggle in modifying (i.e., generative enhancing) the original generated videos, such as eliminating video artifacts and flickering.

Another common approach mainly focuses on removing video artifacts and refining distorted content of generated videos. One example is I2VGEN-XL [50], which follows the idea of SDXL [34] and uses a refinement model to remove visual artifacts and regenerate video content through a noising-denoising process [33]. I2VGEN-XL trains a diffusion model for refinement with largescale high-quality video datasets captioned by short texts, leading to powerful regeneration ability. Although this method has shown incredible performance in improving the stability of generated videos, it cannot increase the spatial and temporal resolution. More importantly, naive noisingdenoising process will change the original video content significantly (i.e., sacrificing fidelity), which cannot always be acceptable in real-world applications.

To conclude, generative video enhancement methods have several limitations. First, cascaded temporal and spatial super-resolution diffusion models requires independent training, but their corresponding training datasets are basically the same (high-quality video datasets). Thus, such design is both sub-optimal and inefficient during the inference. Second, only fixed upscaling factors are supported for both spatial and temporal super-resolution, which limits their practicality. Third, it struggles in obtaining a good balance between quality and fidelity for generative enhancing (i.e., removing artifacts or flickering).

To this end, we propose VEnhancer, a unified generative space-time enhancement framework which supports both spatial and temporal super-resolution with flexible space and time up-sampling scales. Additionally, it also remove visual artifacts and video flickering without severe drop in fidelity. To achieve this, we adopt a pretrained video diffusion model as the fixed generative video prior, supplying the generative ability for video enhancement. To condition the generation on low frame-rate and low-resolution videos, we follow ControlNet [49] and copy the weights of multi-frame encoder and middle block from the generative video prior as the trainable condition network.

Furthermore, to handle different up-sampling scales and reduce artifacts and flickering, we propose space-time data augmentation to construct the training data. In particular, at traininig stage, we randomly sample different downsampling factors (ranging from 1× to 8×), different number skipped frames (ranging from 0 to 7), as well as different noise levels in noise augmentation [21, 19]. To ensure condition network be aware of the associated data augmentation applied to each input video,

https://modelscope.cn/models/iic/Video-to-Video/files

we also propose video-aware conditioning that can realize different conditions across frames. To inject this condition, for key frame, the condition network takes additional input, including the multi-frame condition latents, the embeddings of the associated downscaling factor s and noise level σ by noise augmentation.

Equipped with the above designs, VEnhancer yields stable and efficient end-to-end training. Extensive experiments have demonstrated VEnhancer’s ability in enhancing generated videos (see visual results in Figure 1, 6). Besides, it outperforms state-of-the-art real-world and generative video superresolution methods for spatial super-resolution only. Regarding space-time super-resolution for generated videos, VEnhancer also surpasses state-of-the-art methods as well as two-stage approach (frame interpolation + video super-resolution). Moreover, on the public video generation benchmark, VBench [23], combined with VideoCrafter-2, our approach reaches the first place.

Our contributions can be summarized as below:

- 1. We propose VEnhancer, an efficient approach for generative space-time super-resolution and refinement in a unified diffusion model for the first time. Inspired by ControlNet, we devise a video ControlNet for multi-frame condition injection. Besides, effective data augmentation and conditioning schemes are proposed to assist the end-to-end model training.
- 2. VEnhancer is flexible to adapt to different upsampling factors for either spatial or temporal super-resolution, exceeding the limits of existing diffusion-based spatial or temporal superresolution methods. Besides, it provides flexible control to modify the refinement strength for handling diversified video artifacts.
- 3. VEnhancer surpasses exisitng state-of-the-art video super-resolution methods and space-time super-resolution methods in enhancing generated videos. With VEnhancer, existing text-tovideo method VideoCrafter-2 [9] achieves the top one in VBench [23] in both semantic and quality, outperforming professional video generation products, Gen-2 and Pika .

### 2 Related Work

##### 2.1 Video Generation

Recently, there have been substantial efforts in training large-scale T2V [40, 19, 15, 11, 16, 3, 42, 39] models on large scale datasets. Some works [3, 42, 39] inflate a pre-trained text-to-image (T2I) model by inserting temporal layers and fine-tuning them or all parameters on video data, or adopts a joint image-video training strategy. In order to achieve high-quality video generation, [19, 42, 3] adopts multi-stage pipelines. In particular, cascaded video diffusion models are designed: One T2V base model that is followed by one or more frame interpolation and video super-resolution models. VideoLDM [3], LaVie [42], and Upscale-A-Video [51] all develop the video super-resolution model based on 4× sd (StableDiffusion [35]) upscaler, which has an additional downsampled image for conditioning the generation. One drawback of this base model is losing quite a lot generative ability compared with T2I base models. On the contrary, I2VGEN-XL follows SDXL [34] and uses noising-denoising process [33] to refine the generated artifacts. However, this strategy could improve stability but cannot increase the space-time resolution. VEnhancer is based on a generative video prior, and could address temporal/spatial super-resolution and refinement in a unified model.

##### 2.2 Video Enhancement

Video Super-Resolution. Video Super-Resolution (VSR) is proposed to enhance video quality by upsampling low-resolution (LR) frames into high-resolution (HR) ones. Traditional VSR approaches[4– 6, 24–26, 29, 30, 41, 47] often rely on fixed degradation models to synthesize training data pairs, which leads to a noticeable performance drop in real-world scenarios. To bridge this gap, recent advances[7, 46] in VSR have embraced more diversified degradation models to better simulate real-world low-resolution videos. To achieve photo-realistic reconstruction, Upscale-A-Video[51] integrates diffusion prior to produce detailed textures, upgrading VSR performance into next level. Space-Time Super-Resolution. Space Time Video Super-Resolution (STVSR) aims to simultaneously increase the resolutions of video frames in both spatial and temporal dimensions. Deep-learning

https://runwayml.com/ai-tools/gen-2-text-to-video/ https://pika.art/

based approaches[17, 28, 45, 12] have achieved remarkable results on STVSR. STARNet[17] increases the spatial resolution and frame rate by leveraging the mutual information between space and time. FISR[28] propose a joint framework with a multi-scale temporal loss to upscale the spatialtemporal resolution of videos. [45] proposes a one-stage STVSR framework, which incorporates different sub-modules for LR frame features interpolation, temporal information aggregation and HR reconstruction. VideoINR[12] utilize the continuous video representation to achieve STVSR at arbitrary spatial resolution and frame rate. Although these methods obtain smooth and high-resolution output videos, but they fail in generating realistic texture details.

### 3 Methodology

##### 3.1 Preliminaries: Video Diffusion Models

Our method is built on a video diffusion model [50], which is developed based on one of the latest textto-image diffusion models, Stable Diffusion 2.1 [35]. Given an video x ∈ RF×H×W×3, the encoder E first encodes it into latent representation z = E(x) frame-by-frame, where z ∈ RF×H

′×W′×C. Then, the forward diffusion and reverse denoising are conducted in the latent space. In the forward process, the noise is gradually added to the latent vector z in total T steps. And for each time-step t, the diffusion process is formulated as follows:

zt = αtz + σtϵ, (1)

where ϵ ∈ N(0,I), and αt, σt specify the noise schedule in which the corresponding log signal-tonoise-ratio (log[αt2/σt2]) decreases monotonically with t. And at time-step T, q(zT) = N(0,I). As for backward pass, a diffusion model is used for iteratively denoising under the guidance of the text prompt ctext. By adopting v-prediction parameterization [36], the U-Net denoiser fθ learns to make predictions of vt ≡ αtϵ − σtz. The optimization objective is simply formulated as:

text,ϵ∼N(0,I),t ∥v − fθ(zt,t,ctext)∥22 . (2) At the end, the generated videos are obtained through the VAE decoder: ˆx = D(z).

LLDM = Ez,c

##### 3.2 Architecture Design

The architecture is built upon a video diffusion model. This video diffusion model is able to generate temporal-coherent content and high-quality texture details during iterative denoising. To upsample and refine a low-frame-rate and low-resolution videos in both spatial and temporal dimensions, the visual information should be incorporated into the video diffusion model carefully in order to obtain high-quality results without sacrificing fidelity significantly. High-quality generated videos stem from powerful generative models, while fidelity requires the algorithm to preserve the visual information in the input. Balancing the visual quality and fidelity is always challenging in generative model research. In this work, we follow ControlNet [49] to keep the pretrained video diffusion model untouched for preserving generative capability, but create a trainable copied network for effective condition injection. The architecture is illustrated in Figure 2. We will elaborate on our design carefully in subsequent paragraphs.

The pretrained video diffusion model follows the design of stacking a sequence of interleaved spatial and temporal layers within the 3D-UNet [19, 3] architecture (blue blocks in Figure 2). Specifically, each spatial convolution layer (or attention layer) is followed by a temporal convolution layer (or attention layer). The spatial layers are the same as those in Stable Diffusion 2.1, including ResBlocks [18], self-attention [38] layers, and cross-attention layers. The temporal convolution and attention layers are incorporated with their output layers initialized to zero and finetuned with video datasets. Specifically, the temporal convolution is one-dimensional convolution layer with a kernel size of 3, and the temporal attention is one-dimensional attention layer [39]. In this 3D-UNet, the video features that aligned by temporal layers in encoder will be skipped to the decoder, in which concatenation operation will be performed to combine skipped features with decoder features.

To build the condition network, we make a copy (both the architectures and weights) of the multiframe encoder and the middle block in 3D-UNet (orange blocks in Fig. 2). This condition network

https://huggingface.co/stabilityai/stable-diffusion-2-1

|𝑐𝑡𝑒𝑥𝑡,𝑡,σ,s|
|---|

|𝐼↓𝑠,↑𝑠1|
|---|

|𝑧𝑠,σ1|
|---|

|𝑧𝑠,σ3|
|---|

encoder

+noise σ

|𝐼↓𝑠,↑𝑠3|
|---|
| |
|𝐼↓𝑠,↑𝑠5|

…

VAE

#### VEnhancer

TL

SL

TL

SL

TL SL

TL

SL

|𝑧𝑠,σ5|
|---|

[Figure 5]

ZeroConv

[Figure 6]

…

…

|𝐼↓𝑠,↑𝑠𝑓|
|---|

|𝑧𝑠,σ𝑓|
|---|

ZeroConv

[Figure 7]

ZeroConv

[Figure 8]

key frames (i.e., stride=2)

ZeroConv

Initialize

[Figure 9]

[Figure 10]

full frames

full frames

full frames

[Figure 11]

[Figure 12]

|𝑧𝐻𝑄1|
|---|
| |
|𝑧𝐻𝑄2|
| |
|𝑧𝐻𝑄3|
| |
|𝑧𝐻𝑄4|
| |
|𝑧𝐻𝑄5|

|𝐼𝐻𝑄1|
|---|
| |
|𝐼𝐻𝑄2|
| |
|𝐼𝐻𝑄3|
| |
|𝐼𝐻𝑄4|
| |
|𝐼𝐻𝑄5|

Fixed Trainable

|𝑧𝑡1|
|---|

+

[Figure 13]

+

|𝑧𝑡2|
|---|

+

T steps

+

SL SL

|𝑧𝑡3|
|---|

Spatial Layer

decoder

…

…

VAE

|𝑧𝑡4| |
|---|---|
| | |

TL

TL

SL

SL

TL

SL

TL

SL

TL

SL

SL TL

SL

TL

TL TL Temporal Layer

[Figure 14]

|𝑧𝑡5|
|---|

…

…

…

|𝑧𝐻𝑄𝑓|
|---|

|𝐼𝐻𝑄𝑓|
|---|

|𝑧𝑡𝑓|
|---|

[Figure 15]

- Figure 2: The architecture of VEnhancer. It follows ControlNet [49] and copies the architecures and weights of multi-frame encoder and middle block of a pretrained video diffusion model to build a trainable condition network. This “video ControlNet" accepts low-resolution key frames as well as full frames of noisy latents as inputs. Also, the noise level σ regarding noise augmentation and downscaling factor s serve as additional network conditioning apart from timestep t and prompt ctext.

also takes full frames of noisy latents as input, and outputs multi-scale temporal-coherent video features. These features will be injected into the original 3D-UNet through newly added zero convolutions (yellow blocks in Fig. 2). The output features of the middle block in condition network will be added back to the features of the middle block in 3D-UNet. While for output features of encoder blocks in condition network, their features will be added to the skipped video features in 3D-UNet, which are also produced by encoder blocks. Such architecture design is consistent with the original ControlNet. The main difference is that video ControlNet has temporal convolution or attention layer positioned after spatial convolution or attention layer, and the multi-frame control signals are aligned by temporal layers.

##### 3.3 Space-Time Data Augmentation

In this section, we discuss about how to achieve unified space-time super-resolution with arbitrary up-sampling space and time scales, as well as refinement with varying degrees. To this end, we propose a novel data augmentation strategy for both space and time axes. Details are discussed below.

Time axis. Given a sequence of high-frame-rate and high-resolution video frames I1:f = [I1,I2,...,If] with frame length f, we use a sliding window across time axis to select frames. The frame sliding window size m is randomly sampled from a predefined set, ranging from 1 to 8. This corresponds to time scales from 1× to 8×. Note that 1× time scale requires no frame interpolation, thus the multi-task problem downgrades to video super-resolution. After the frame skipping, we obtain a sequence of key frames I1:f:m = [I1,I1+m,I1+2×m,...,If].

Space axis. Then, we perform spatial downsampling for these obtained key frames. Specifically, the downscaling factor s is ramdomly sampled from [1,8], which represents 1× ∼ 8× space superresolution. When s = 1, there is no need to perform spatial super-resolution. All frames in one sequence are downsampled with the same downscaling factor s. Thus, we arrive at low-frame-rate

and low-resolution video frames: I↓1:sf:m. In practice, we should upsample them back to the original spatial sizes by bilinear interpolation before being passed to the networks, so we obtain I↓1:s,f↑:ms . Note that each space or time scale corresponds to different difficulty level, and thus the sampling is not uniform. Particularly, we set sampling probabilities of scales 4× and 8× based on a ratio of 1 : 2, which is determined by their associated scale values.

Then, we use the encoder part of a pretrained variational autoencoder (VAE) E to project the input sequence to the latent space frame-wisely:

zs1:f:m = [E(I↓1s,↑s),E(I↓1+s,↑ms),E(I↓1+2s,↑s×m),...,E(I↓fs,↑s)]. (3)

Noise augmentation in latent space. At this stage, we conduct noise augmentation to noise the latent condition information in varying degrees in order to achieve controllable refinement. This noise augmentation process is the same as the diffusion process (1) used in the video diffusion model. Specifically, the condition latent sequence is corrupted by:

zs,t1:f′:m = αt′zs1:f:m + σt′ϵ1:f:m, (4)

where αt′, σt′ determine the signal-to-noise-ratio at time-step t′, and t′ ∈ {1,...,T′}. Note that the pretrained video diffusion model adopts 1,000 steps (T = 1000 in Eq. (1)). While the noise augmentation only needs to corrupt the low-level information, T′ is set to 300 empirically. For more intuitive denotation, we use σ instead of t′. Finally, we arrive at zs,σ1:f:m = E(I↓s,↑s)σ1:f:m. The whole process of space-time data augmentation is summarized as follows:

I1:f → I1:f:m → I↓1:sf:m → I↓1:s,f↑:ms → E(I↓s,↑s)1:f:m → E(I↓s,↑s)σ1:f:m. (5)

##### 3.4 Video-Aware Conditioning

Besides data augmentation, the corresponding conditioning mechanism should also be designed in order to boost the model training and avoid averaging performance for different space or time scales and noise augmentation. In practice, the condition latent sequence zs,σ1:f:m, the corresponding downscaling factor s, and augmented noises σ are all considered as for conditioning. Please refer to Fig. 3 for more intuitive demonstration.

###### Video-Aware Conditioning

|𝑠|
|---|

|σ|
|---|

|𝑡|
|---|

+

+

𝑧𝑡1

SL

Spatial Layer

+

|𝑧𝑠,σ1|
|---|

|𝑡|
|---|

|𝑧𝑡2|
|---|

SL

Spatial Layer

|𝑠|
|---|

|σ|
|---|

|𝑡|
|---|

+

+

TemporalLayer

TemporalLayer

|𝑧𝑡3|
|---|

Spatial Layer

SL

+

|𝑧𝑠,σ3| |
|---|---|
| | |

…

|𝑡|
|---|

|𝑧𝑡4|
|---|

SL

Spatial Layer

|𝑠|
|---|

|σ|
|---|

|𝑡|
|---|

+

+

Given the synthesized condition latent sequence zs,σ1:f:m, we use one convolution layer with zeroinitialization Convzero for connecting it to video ControlNet. Specifically, we have:

𝑧𝑡5

Spatial Layer

SL

𝑧𝑠,σ5 +

…

𝑠

…

σ

…

|𝑡|
|---|

+

+

SL

𝑧𝑡𝑓

Spatial Layer

𝑧𝑠,σ𝑓 +

fout1:f = Conv(zt1:f), (6) fout1:f:m = Conv(zt1:f:m) + Convzero(zs,σ1:f:m), (7)

Figure 3: Video-aware conditioning. For frame that has condition image as input (key frame), we add it to the first layer of video ControlNet. Besides, the embeddings of noise level σ and downscaling factor s are added to the existing t embedding, which will be broadcast to all spatial layers.

where Conv is the first convolution layer in video ControlNet, zt1:f and zt1:f:m denote the full frames and key frames of noisy latents at timestep t, respectively. Note that Conv and Convzero share the same hyper-parameter configuration (i.e.,kernel size, padding, et.al.), As it is shown, only key-frame features in video ControlNet will be added with the condition features, while others remain unchanged. This strategy enables progressive condition injection as the weights of Convzero grows from zero starting point.

For conditioning regarding downscaling factor s and noise augmentation σ, we incorporate them to the existing time embedding in video ControlNet. Specifically, for timestep t, sinusoidal encoding [20, 35, 38] is used to provide the model with a positional encoding for time. Then, one MLP (two linear layers with a SiLU [13] activation layer in between) is applied. Specifically, we have:

tpos = Sinusoidal(t), (8) temb = Linear(2)t (SiLU(Linear(1)t (tpos))), (9) t1:embf = Repeat(temb,f), (10)

where t1:embf is obtained by Repeat temb by f times in the frame axis. This time embedding sequence will be broadcast to all ResBlocks in video ControlNet for timestep injection.

Also, we elucidate the conditioning for noise augmentation. As mentioned in Eq.(4), noise augmentation shares the same way as diffusion process, but with much smaller maximum timestep (i.e., T′ = 300). Therefore, we reuse the encoding and mapping for timestep t in diffusion process. After

this, we add a linear layer with zero initialization (denoted as Linearzero). To conclude, we have:

σpos = Sinusoidal(σ), (11) σemb = Linearzero,σ(Linear(2)t (SiLU(Linear(1)t (σpos)))). (12)

To achieve video-aware conditioning, we add σemb only to the key frames. So σemb is repeated k times, where k is the number of key frames. The video-aware controlling is presented as follows:

temb1:f:m = temb1:f:m + σemb1:k , (13) where the addition operation is performed frame-wisely for key frames.

Regarding downscaling factor s, the corresponding encoding, mapping and controlling are similar as above. In particular, we newly introduce one MLP, in which the output layer is zero-initialized. The video-aware conditioning is performed as:

spos = Sinusoidal(s), (14) semb = Linearzero,s(SiLU(Linears(spos))), (15)

- s1:embk = Repeat(semb,k), (16)
- t1:embf:m = temb1:f:m + s1:embk . (17)

With our proposed space-time data augmentation and video-aware conditioning, VEnhancer can be well-trained in an end-to-end manner, and yields great performance for generative enhancement.

### 4 Experiments

Datasets. We collect around 350k high-quality and high-resolution video clips from the Internet to constitute our training set. We train VEnhancer on resolution 720 × 1280 with center cropping, and the target FPS is fixed to 24 by frame skipping. Regarding test dataset, we collect comprehensive generated videos from state-of-the-art text-to-video methods [39, 42, 9, 48]. Practically, we select videos with large motions and diverse contents. This test dataset is denoted as AIGC2023, which is used to evaluate VEnhancer and baselines for video super-resolution and space-time super-resolution tasks. For evaluation on VBench, all generated videos based on the provided prompt suite are considered, resulting in more than 5k videos.

Implementation Details. The batch size is set to 256. AdamW [32] is used as the optimizer, and the learning rate is set to 10−5. During training, we dropout the text prompt with a probability of 10%. The training process lasts about four days with 16 NVIDIA A100 GPUs. During inference, we use 50 DDIM [37] sampling steps and classifier-free guidance (cfg) [22].

Metrics. Regarding evaluation for video super-resolution and space-time super-resolution on AIGC2023 test dataset, we use both image quality assessment (IQA) and video quality assessment (VQA) metrics. As there is no ground-truth available, we can only use non-reference metrics. Specifically, MUSIQ [27] and DOVER [43] are adopted. Moreover, we refer to video generation benchmark, VBench [23], for more comprehensive evaluation. Specifically, we choose Subject Consistency (i.e., whether the subject remains consistent), Motion smoothness (i.e., how smooth the video is), Aesthetic Quality, and Imaging Quality for evaluation. Regarding evaluation for video generation, we consider all 16 evaluation dimensions from VBench.

##### 4.1 Comparison with Video Super-Resolution Methods

For video super-resolution, VEnhancer is compared with the state-of-the-art real-world video superresolution method, RealBasicVSR [7], and the state-of-the-art generative video super-resolution method, LaVie-SR [42] (super-resolution).

- As shown in Table 1, VEnhancer outperforms both generative video super-resolution method (LaVieSR [42]) and real-world video super-resolution method (RealBasicVSR [7]) in all metrics, suggesting its outstanding enhancement ability for videos. Note that LaVie-SR surpasses RealBasicVSR in image/video quality, as LaVie-SR obtains higher scores in MUSIQ, DOVER, Imaging Quality, and Aesthetic Quality. This indicates that LaVie-SR could produce sharper results than RealBasicVSR. It is because that LaVie-SR is based on diffusion models while RealBasicVSR is trained with GAN loss.

- Table 1: Quantitative comparison for video super-resolution (4×) on AIGC2023 test dataset. Red and blue indicate the best and second best performance. The top 3 results are marked as gray .

| |DOVER↑ MUSIQ↑<br><br>|Imaging Quality<br><br>Aesthetic Quality<br><br>Subject Consistency<br><br>Motion Smoothness|
|---|---|---|
|LaVie-SR [42] RealBasicVSR [7] Ours|0.8427 55.8428 0.8252 50.5978 0.8498 56.6113<br><br>|0.5481 0.6692 0.9562 0.9710 0.5401 0.6622 0.9555 0.9729 0.5872 0.6728 0.9624 0.9787<br><br>|

[Figure 16]

[Figure 17]

[Figure 18]

| |
|---|

[Figure 19]

[Figure 20]

[Figure 21]

| |
|---|

| |
|---|

[Figure 22]

Input

[Figure 23]

[Figure 24]

| |
|---|

[Figure 25]

[Figure 26]

[Figure 27]

| |
|---|

| |
|---|

[Figure 28]

[Figure 29]

RealBasicVSR

[Figure 30]

| |
|---|

[Figure 31]

[Figure 32]

[Figure 33]

| |
|---|

| |
|---|

[Figure 34]

[Figure 35]

LaVie-SR

[Figure 36]

| |
|---|

[Figure 37]

[Figure 38]

[Figure 39]

| |
|---|

| |
|---|

Ours

- Figure 4: Visual comparison for video super-resolution (4×) on AIGC2023 test dataset. Prompt: Iron Man flying in the sky. Zoom in for best view.

Besides, RealBasicVSR shows better performance in Motion Smoothness compared with LaVie-SR, since smooth results are more likely to achieve higher score in this metric. Practically, there is a trade off between video smoothness and video quality [23]. Nevertheless, VEnhancer achieves a great balance between these two aspects.

The visual comparison is presented in Fig. 4. The prompt is “Iron man flying in the sky". The input video is already consistent with the prompt, but lacks details on the iron man suit. RealBasicVSR could remove some noises or artifacts of the generated videos as it incorporates complex degradation for model training. However, it fails in generating realistic details but produces over-smoothed results, since its generative ability is limited. On the other hand, the results of LaVie-SR contains more artifacts than input. Without successfully removing artifacts, the generative super-resolution model will enlarge the existing defects. In contrast, VEnhancer could first remove unpleasing artifacts and refine the distorted content (e.g., head region), and then generate faithfuls details (e.g., helmet and armor) that are consistent with the text prompt. Also, the whole video is with high-quality and high-resolution, as well as smooth dynamics.

##### 4.2 Comparison with Space-Time Super-Resolution Methods

For space-time super-resolution task, we compare two state-of-the-art space-time super-resolution methods: VideoINR [12] and Zooming-Slow-Mo [45] (Zoom for short). We also consider LaVie’s two-stage pipeline: LaVie-FI (frame interpolation) + LaVie-SR for more thorough comparison.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

- Table 2: Quantitative comparison for space-time super-resolution (4×) on AIGC2023 test dataset. Red and blue indicate the best and second best performance. The top 3 results are marked as gray .

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

| |DOVER↑ MUSIQ↑<br><br>Imaging Quality<br><br>Aesthetic Quality<br><br>Subject Consistency<br><br>Motion Smoothness|
|---|---|
|LaVie-FI + LaVie-SR [42] VideoINR [12] Zooming Slow-Mo [45] Ours|0.8159 53.2128 0.5299 0.6566 0.9603 0.9857 0.7608 34.1060 0.4778 0.6624 0.9615 0.9933 0.7328 33.8470 0.4925 0.6624 0.9524 0.9908 0.8487 50.3659 0.5648 0.6665 0.9666 0.9898<br><br>|

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Input Key frame

Key frame

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

LaVie-FI+ LaVie-SR

VideoINR

Zoom

Ours

- Figure 5: Visual comparison for space-time super-resolution (4×) on AIGC2023 test dataset. Prompt: A cute raccoon playing guitar in a boat on the ocean. Zoom in for best view.

- As shown in Table 2, we observe that our method surpasses all baselines in DOVER, Imaging Quality, and Aesthetic Quality, showing its superior capability in generating sharp and realistic video content. Besides, it obtains highest score in Subject Consistency, indicating VEnhancer’s ability in maintaining the subject consistent when performing joint frame interpolation and spatial super-resolution. We notice that state-of-the-art space-time super-resolution methods (VideoINR and Zooming Slow-Mo) stand out in Motion Smoothness. As both of them are optimized with reconstruction loss, the produced results are very smooth across frames. At a cost, they perform poorly in metrics regarding quality, such as DOVER, MUSIQ, and Imaging Quality. The two-stage approach (LaVie-FI + LaVie-SR) obtains good scores in DOVER, MUSIQ and Imaging Quality, demonstrating DM-based methods’ advantage in generation. However, its performance in video stability and subject consistency is unsatisfactory due to its inferior capability in video refinement.

The visual comparison is illustrated in Fig. 5. The first and third columns present the low-resolution key frames. Note that the input frames are not consistent especially in the region of guitar strings. The two-stage DM-based approach, LaVie-FI + LaVie-SR [42], can produce very sharp results for all frames (key and predicted ones). However, it generates messy contents which are not semantically aligned with prompt. Moreover, the generated details are changing across time, indicating severe flickering. For reconstruction-based methods (VideoINR [12] and Zoom [45]), the produced results are similar: lacking details and fail in improving the consistency of the original input frames. On the contrary, VEnhancer is not only able to achieve unified space-time super-resolution, but can

also improve the temporal consistency of the generated videos by refinement (i.e., guitar strings and raccoon hands plucking the strings).

##### 4.3 Evaluation on Improving Video Generation

In this section, we evaluate VEnhancer’s ability in improving the existing state-of-the-art T2V method (i.e., VideoCrafter2 [9]). The baselines includes open-source T2V methods: VideoCrafter-2 (VC-2 for short), Show-1 [48], Lavie [42], Open-Sora , as well as professional video generation products, Pika and Gen-2. All dimensions of VBench are considered for evaluation.

The quantitative results are organized in Table 3. In general, VideoCrafter-2 is already the best in overall Semantic compared with other baselines, demonstrating its superiority in generating highfidelity contents to the VBench’s prompt suite. Regarding the overall Quality, it lags behind Pika and Gen-2. But with VEnhancer, VideoCrafter-2 is able to achieve the highest scores in both overall Quality and Semantic. This indicates that VEnhancer can improve the semantic content and video quality at the same time, showing a powerful unified enhancement (i.e., refinement and space-time super-resolution) ability.

Regarding Quality aspect, we note that Pika and Gen-2 could obtain good scores in various dimensions, but perform very poorly in dynamic degree. This dimension assesses whether a video has large motions, and higher score corresponds to larger motion. It is suggested that Gen-2 and Pika obtain good temporal consistency by sacrificing versertile motions. This demonstrates the advantage of adopting a two-stage pipeline: the first T2V model focuses on generating semantic content and motions with good fidelity to the prompts, while the following enhancement model can improve the semantic in low-level and image quality, as well as temporal consistency. We also provide visual comparison in Fig. 6. It is obvious that VideoCrafter-2 yields the best video content but with low-resolution. And VEnhancer significantly improves its quality (with 2k resolution). Gen-2 and Pika could produce results with around 1k resolution, but their generated contents have some apparent flaws (weird robe in Gen-2 and unnatural rabbit’face in Pika). Open-Sora and Lavie both generate videos with square size, leading to incomplete generation for backgrounds.

- Table 3: VBench Evaluation Results per Dimension. This table compares the performance of four open-source video generation models (LaVie [42], Show-1 [48], Open-Sora, VideoCrafter-2 [9]) and two professional video generation products (Pika and Gen-2) across each of the 16 VBench dimensions regarding two aspects (Quality and Semantic). A higher score indicates relatively better performance for a particular dimension. Red and blue indicate the best and second best performance. The top 3 results are marked as gray .

| |Dimensions<br><br>|Show-1 [48] LaVie [42] Open-Sora Pika Gen-2 VC-2 [9] VC-2+VEnhancer|
|---|---|---|
|Quality<br><br>|Subject Consistency Background Consistency Temporal Flickering Motion Smoothness Aesthetic Quality Dynamic Degree Imaging Quality|95.53% 91.41% 92.09% 96.76% 97.61% 96.85% 97.17%<br><br>98.02% 97.47% 97.39% 98.95% 97.61% 98.22% 98.54%<br><br>99.12% 98.30% 98.41% 99.77% 99.56% 98.41% 98.46% 98.24% 96.38% 95.61% 99.51% 99.58% 97.73% 97.75%<br><br><br>57.35% 54.94% 57.76% 63.15% 66.96% 63.13% 65.89% 44.44% 49.72% 48.61% 37.22% 18.89% 42.50% 42.50%<br><br>58.66% 61.90% 61.51% 62.33% 67.42% 67.22% 70.45%<br><br><br>|
|Semantic<br><br>|Object Class Multiple Objects Human Action Color Spatial Relationship Scene Appearance Style Temporal Style Overall Consistency|93.07% 91.82% 74.98% 87.45% 90.92% 92.55% 93.39% 45.47% 33.32% 33.64% 46.69% 55.47% 40.66% 49.83% 95.60% 96.80% 85.00% 88.00% 89.20% 95.00% 95.00% 86.35% 86.39% 78.15% 85.31% 89.49% 92.92% 94.41% 53.50% 34.09% 43.95% 65.65% 66.91% 35.86% 64.88% 47.03% 52.69% 37.33% 44.80% 48.91% 55.29% 51.82% 23.06% 23.56% 21.58% 21.89% 19.34% 25.13% 24.32% 25.28% 25.93% 25.46% 24.44% 24.12% 25.84% 25.17% 27.46% 26.41% 26.18% 25.47% 26.17% 28.23% 27.57%<br><br>|
|Overall|Quality Semantic<br><br>|80.42% 78.78% 78.82% 82.68% 82.46% 82.20% 83.28% 72.98% 70.31% 64.28% 71.26% 73.03% 73.42% 76.73%<br><br>|

https://github.com/hpcaitech/Open-Sora

[Figure 64]

[Figure 65]

VideoCrafter2

+VEnhancer

[Figure 66]

Open-Sora

[Figure 67]

LaVie

A fat rabbit wearing a purple robe walking through a fantasy landscape.

[Figure 68]

[Figure 69]

Gen-2

Show-1

[Figure 70]

Pika

- Figure 6: Visual comparison of screenshots obtained from VideoCrafter-2+VEnhancer and other T2V models. The layout is arranged based on their original resolutions. Zoom in for best view.

##### 4.4 Ablation Studies

Please refer to Appendix A for arbitrary up-sampling scales for space/time super-resolution, as well as different noise levels in noise augmentation.

### 5 Conclusion and Limitation.

In this work, we propose a generative space-time enhancement method, VEnhancer, for video generation. It can achieve spatial super-resolution, temporal super-resolution and video refinement in a unified framework. We base on a pretrained video diffusion model and build a trainable video ControlNet for effective condition injection. Space-time data augmentation and video-aware conditioning are proposed to train video ControlNet in an end-to-end manner. Extensive experiments have demonstrated the superiority over state-of-the-art video super-resolution and space-time superresolution methods in enhancing AI-generated videos. Moreover, VEnhancer is able to improve the results of existing state-of-the-art T2V method, lifting VideoCrater-2’s ranking to top one. However, our work has several limitations. First, as it is based on diffusion models, the inference takes more time than one-step methods, for instance, reconstruction-based space-time super-resolution methods. Second, it may face challenges in handling AI-generated long videos, since the long-term (over 10s) consistency has not been addressed in this work.

### References

- [1] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563– 22575, 2023.
- [4] Jiezhang Cao, Yawei Li, Kai Zhang, and Luc Van Gool. Video super-resolution transformer. arXiv preprint arXiv:2106.06847, 2021.
- [5] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4947–4956, 2021.
- [6] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Basicvsr++: Improving video super-resolution with enhanced propagation and alignment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5972–5981, 2022.
- [7] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in realworld video super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5962–5971, 2022.
- [8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [9] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024.
- [10] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [11] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Delving deep into diffusion transformers for image and video generation. arXiv preprint arXiv:2312.04557, 2023.
- [12] Zeyuan Chen, Yinbo Chen, Jingwen Liu, Xingqian Xu, Vidit Goel, Zhangyang Wang, Humphrey Shi, and Xiaolong Wang. Videoinr: Learning video implicit neural representation for continuous space-time superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2047–2057, 2022.
- [13] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:3–11, 2018.
- [14] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.
- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [16] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662, 2023.
- [17] Muhammad Haris, Greg Shakhnarovich, and Norimichi Ukita. Space-time-aware multi-resolution video enhancement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2859–2868, 2020.

- [18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [19] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [21] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.
- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [23] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023.
- [24] Takashi Isobe, Xu Jia, Shuhang Gu, Songjiang Li, Shengjin Wang, and Qi Tian. Video super-resolution with recurrent structure-detail network. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, pages 645–660. Springer, 2020.
- [25] Takashi Isobe, Songjiang Li, Xu Jia, Shanxin Yuan, Gregory Slabaugh, Chunjing Xu, Ya-Li Li, Shengjin Wang, and Qi Tian. Video super-resolution with temporal group attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8008–8017, 2020.
- [26] Takashi Isobe, Fang Zhu, Xu Jia, and Shengjin Wang. Revisiting temporal modeling for video superresolution. arXiv preprint arXiv:2008.05765, 2020.
- [27] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5148–5157, 2021.
- [28] Soo Ye Kim, Jihyong Oh, and Munchurl Kim. Fisr: Deep joint frame interpolation and super-resolution with a multi-scale temporal loss. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 11278–11286, 2020.
- [29] Jingyun Liang, Jiezhang Cao, Yuchen Fan, Kai Zhang, Rakesh Ranjan, Yawei Li, Radu Timofte, and Luc Van Gool. Vrt: A video restoration transformer. IEEE Transactions on Image Processing, 2024.
- [30] Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V Gool. Recurrent video restoration transformer with guided deformable attention. Advances in Neural Information Processing Systems, 35:378–393, 2022.
- [31] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior, 2024.
- [32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [36] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

- [37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [39] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [40] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024.
- [41] Xintao Wang, Kelvin CK Chan, Ke Yu, Chao Dong, and Chen Change Loy. Edvr: Video restoration with enhanced deformable convolutional networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019.
- [42] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [43] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In International Conference on Computer Vision (ICCV), 2023.
- [44] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [45] Xiaoyu Xiang, Yapeng Tian, Yulun Zhang, Yun Fu, Jan P Allebach, and Chenliang Xu. Zooming slow-mo: Fast and accurate one-stage space-time video super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3370–3379, 2020.
- [46] Liangbin Xie, Xintao Wang, Shuwei Shi, Jinjin Gu, Chao Dong, and Ying Shan. Mitigating artifacts in real-world video super-resolution models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2956–2964, 2023.
- [47] Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T Freeman. Video enhancement with task-oriented flow. International Journal of Computer Vision, 127:1106–1125, 2019.
- [48] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.
- [49] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [50] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [51] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-avideo: Temporal-consistent diffusion model for real-world video super-resolution. arXiv preprint arXiv:2312.06640, 2023.

### A Ablation Studies

- A.1 The Effectiveness of Noise Augmentation.

During training, the noise level regarding noise augmentation is randomly sampled within a predefined range. While during inference, one can change the noise level to achieve refinement with different strengths. In general, higher noise corresponds to stronger refinement and regeneration. We present the visual comparison among different noise levels in Figure 7. The first frame of one AI-generated video is presented in the left. It is of low-resolution and lacks details. Also, the original video has very obvious flickering. If we set σ = 0, VEnhancer will generate unpleasing noises in the background. As there is domain mismatch between the training data and testing data, the enhancement fails in handling unseen and challenging scenarios. Fortunately, we can mitigate this by adding noise in the condition latents for corrupting the noisy and unknown low-level details. As we increase the noise level, the artifacts are gradually vanishing. When σ = 250, the result is noise-clean, and has abundant semantic details.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

input σ = 0 σ = 150 σ = 250

[Figure 75]

| |
|---|

- Figure 7: Visual comparison of setting different noise levels in noise augmentation during testing.

A.2 Arbitrary Up-sampling Scales for Spatial Super-Resolution.

In this section, we show that VEnhancer is able to up-sample videos with arbitrary scales. From Figure 8, we observe that VEnhancer could produce satisfactory results on different scales (2.5×, 3×, 3.5×, 4×, and 4.5×), suggesting its flexibility and generalization in adapting to different tasks. In particular, given one frame of the generated video (312 × 512), VEnhancer could improve the generated details when the up-sampling scale grows up. When s = 2.5 ∼ 3.5, the panda’s hand is less realistic. But it becomes better when s = 4 or s = 4.5. It is also noticed that the panda’ fur is becoming more realistic as s grows.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

2.5 x

Input

3 x

4 x

3.5 x

4.5 x

- Figure 8: Visual results of different up-sampling scales (2.5×, 3×, 3.5×, 4×, and 4.5×) for spatial super-resolution during testing.

##### A.3 Arbitrary Up-sampling Scales for Temporal Super-Resolution.

[Figure 82]

[Figure 83]

Key frame Key frame

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

3 x

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

4 x

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

5 x

- Figure 9: Visual results of different up-sampling scales (3×, 4×, and 5×) for temporal superresolution during testing.

In this part, we show VEnhancer is able to achieve arbitrary up-sampling in time axis. Given two lowresolution key frames, we aim to up-sample them to high-resolution ones, and also interpolate several frames (ranging from 2 to 4) between them. As shown in Figure 9, the results are consistent across frames, showing not flicking or distortions. Besides, the spatial quality has also been significantly improved. As shown in the last row, 5× frame interpolation yields smooth frames with generated

###### contents: the shadow in the right leg is changing, showing a very natural transition. This indicates that diffusion-based frame interpolation has great capability in both motion and content generation.

