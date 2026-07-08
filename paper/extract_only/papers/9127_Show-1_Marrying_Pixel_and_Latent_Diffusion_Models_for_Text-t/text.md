|Noname manuscript No. (will be inserted by the editor)|
|---|

# Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation

### David Junhao Zhang · Jay Zhangjie Wu · Jia-Wei Liu · Rui Zhao · Lingmin Ran · Yuchao Gu · Difei Gao · Mike Zheng Shou

arXiv:2309.15818v3[cs.CV]30May2025

Received: date / Accepted: date

[Figure 1]

[Figure 2]

[Figure 3]

Close up of mystic cat, like a buring phoenix, red and black colors.

A panda besides the waterfall is holding a sign that says “Show 1”.

Toad practicing karate.

- Fig. 1: Given text descriptions, our approach generates highly faithful and photorealistic videos. Click the image to play the video clips. Best viewed with Adobe Acrobat Reader.

Abstract Significant advancements have been achieved in the realm of large-scale pre-trained textto-video Diffusion Models (VDMs). However, previous methods either rely solely on pixel-based VDMs, which come with high computational costs, or on latent-based VDMs, which often struggle with precise text-video alignment. In this paper, we are the first to propose a hybrid model, dubbed as Show-1, which marries pixel-based and latent-based VDMs for text-to-video generation. Our model first uses pixel-based VDMs to produce a low-resolution video of strong text-video correlation. After that, we propose a novel expert translation method that employs the latent-based VDMs to further upsample the low-resolution video to high resolution, which can also remove potential artifacts and corruptions from low-resolution videos.

All authors are affiliated with Show Lab, National University of Singapore. David Junhao Zhang, Jay Zhangjie Wu and Jia-Wei Liu contribute equally. Mike Zheng Shou is the corresponding author.

Compared to latent VDMs, Show-1 can produce high-quality videos of precise text-video alignment; Compared to pixel VDMs, Show-1 is much more efficient (GPU memory usage during inference is 15G vs 72G). Furthermore, our Show-1 model can be readily adapted for motion customization and video stylization applications through simple temporal attention layer finetuning. Our model achieves state-of-the-art performance on standard video generation benchmarks. Code of Show-1 is publicly available and more videos can be found here.

### 1 Introduction

Remarkable progress has been made in developing large-scale pre-trained Text-to-Video Diffusion Models (VDMs), including closed-source ones (e.g., Make-AVideo (Singer et al., 2022), Imagen Video (Ho et al., 2022a), Video LDM (Blattmann et al., 2023a), Gen-

#### 2 (Esser et al., 2023)) and open-sourced ones (e.g.,

(a) Comparisons of different resolutions and spaces for the Keyframes stage

the compacted latent space within a variational autoencoder (VAE), like Video LDM (Blattmann et al., 2023a) and MagicVideo (Zhou et al., 2022).

[Figure 4]

However, both of them have pros and cons. As indicated by (Singer et al., 2022; Ho et al., 2022a), pixel-based VDMs can generate motion accurately aligned with the textual prompt because they start generating video from a very low resolution e.g., 64 × 40 (also demonstrated by Fig. 2). But they typically demand expensive computational costs in terms of time and GPU memory, especially when upscaling the video to the high-resolution. Latent-based VDMs are more resource-efficient because they work in a reduceddimension latent space. But it is challenging for such small latent space (e.g., 8 × 5 for 64 × 40 videos) to cover rich yet necessary visual semantic details as described by the textual prompt. Therefore, as shown in Fig. 2, the generated videos often are not well-aligned with the textual prompts. On the other hand, when directly generating relatively high resolution videos (e.g., 256×160) using latent methods, the alignment between text and video could also be relatively weaker. This occurs because with higher resolution, the latent model tends to concentrate more on spatial appearance, potentially overlooking the text-video alignment, as validated by Fig. 2 and Tab. 4.

CLIP-TextSIM

| |f=0|f=2|f=4|f=8|
|---|---|---|---|---|
|64 x 40| | | | |
|260 x 160| | | | |
|512 x 320| | | | |

Motion Fidelity

(b) Keyframes visual comparisons

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Pixel-based VDM 64 x 40

Latent-based VDM 256 x 160（）

[Figure 11]

[Figure 12]

[Figure 13]

64x40/Pixel f=0 64x40/Latent f=8 256x160/Latent f=8

- Fig. 2: The comparison (a) evaluates the CLIP-Text Similarity Score, highlighting how well the text aligns with video content and the fidelity of motion across various pixel and latent model pairings at different resolutions and compression ratios during the keyframe stage. These keyframe models all utilize identical latent VDM for the final super-resolution phases. The point’s radius signifies the peak memory usage during the whole inference process. For consistency, all models in this study employ the same T5 text encoder and start with pre-trained weights from LAION, followed by additional training on WebVid using uniform steps to maintain fairness. f = 0 indicates the model operating in pixel space, while f = 2,4,8 correspond to different latent compression ratios. The findings reveal that employing a pixel VDM to create low-resolution videos (64x40) at the keyframe stage yields superior outcomes compared to latent VDM across various resolutions and compression ratios. Part (b) presents the visual outcomes of the keyframes.

Prior models have often exclusively used either pixel or latent approaches across all above modules, facing the cons brought by either pixel or latent VDMs. Specifically, pure pixel-based VDMs e.g., MakeA-Video (Singer et al., 2022) are computationally demanding, while latent-based models may compromise text-video alignment and motion fidelity. To solve these problems, integrating the strengths of pixel-based and latent-based Video Diffusion Models (VDMs), while addressing their weaknesses, shows immense potential. Achieving this integration could yield a text-to-video model that not only excels in video-text alignment but also with low computation cost.

Toward this objective, we begin a step-by-step exploration of how to merge pixel and latent VDMs effectively. Referencing Fig. 2, we observe that initiating video generation with low-resolution keyframes using pixel-based VDM leads to improved text-video alignment. Accordingly, we employ a coarse-to-fine generation strategy that starts by creating low-resolution and low-frame-rate keyframes using pixel-based VDM. Then we apply a temporal interpolation module and a super-resolution module to enhance the video in both time and space. In the current step, we leverage the advantages of pixel VDMs, resulting in an improved text-aligned low-resolution video.

VideoCrafter (He et al., 2022), ModelScopeT2V (Wang et al., 2023a). These VDMs can be classified into two types: (1) Pixel-based VDMs that directly denoise pixel values, including Make-A-Video (Singer et al., 2022), Imagen Video (Ho et al., 2022a), PYoCo (Ge et al., 2023), and (2) Latent-based VDMs that manipulate

T T T

…

… …

[Figure 14]

[Figure 15]

poral attention layers of the keyframes UNet on a single video, Show-1 is capable of distilling the video’s motion into these layers. This process allows for motion customization and stylization of the video, as the fixed spatial layers offer a range of appearances based on the text.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Source Ours X-T Slice

SDx4 w/temporal

The key contributions of our paper is summarized as follows:

[Figure 23]

[Figure 24]

T

- – Upon examining pixel and latent VDMs, we discover that: 1) pixel VDMs excel in generating lowresolution videos with more natural motion and superior text-video synchronization compared to latent VDMs; 2) when using the low-resolution video as an initial guide, latent VDMs can effectively function as super-resolution tools by simple expert translation, refining spatial clarity and creating high-quality videos with greater efficiency than pixel VDMs. Meanwhile, with expert translation, the artifacts and corruptions of low resolution videos can be reduced.
- – We are the first to integrate the strengths of both pixel and latent VDMs, resulting into a novel video generation model that can produce high-resolution videos of precise text-video alignment at low computational cost (15G GPU memory during inference).
- – By fine-tuning the temporal attention layer, our Show-1 model can be additionally adapted for motion customization and video stylization applications.

Ours SDx4 w/temporal

- Fig. 3: Final Super-Resolution Comparisons. We contrast our expert translation against typical SDx4 upsampling that includes temporal layers and visualize the X-T slice of the final outcomes. The findings suggest that our approach is capable of managing the possible corruptions found in low-resolution videos, resulting in improved temporal consistency and quality (notably smoother and with reduced noise in the X-T slice) compared to SDx4 with temporal layers.

However, as previously mentioned, continuing to use pixel VDMs as the final super-resolution module for ultimate high-resolution output will result in significant computational costs. Thus, we opt for a latentbased VDM for an efficient final super-resolution module. Typical latent-based VDMs, such as SDx4 (Rombach et al., 2022), usually combine low-resolution video and noise as input for a UNet. Nonetheless, as shown in Fig. 3, there might be some artifacts or corruptions originating from the low-resolution videos. Simply applying typical latent-based VDMs like SDx4 with a temporal extension will not address these issues, leading to subpar final results and poor temporal consistency, as evidenced by the discontinuous and noisy X-T slice in Fig. 3. To overcome this problem, we introduce an expert translation method for latent-based VDMs, which directly uses the encoded noisy low-resolution video as the input for UNet with expert finetuning. We discover that latent-based VDMs with expert translation can effectively convert low-resolution video to high-resolution while preserving the original appearance and accurate text-video alignment. Crucially, it also eliminates the artifacts and corruptions from low resolution videos.

### 2 Related Work

Text-to-image generation. (Reed et al., 2016) stands as one of the initial methods that adapts the unconditional Generative Adversarial Network (GAN) introduced by (Goodfellow et al., 2014) for text-toimage (T2I) generation. Later versions of GANs delve into progressive generation, as seen in (Zhang et al., 2017) and (Hong et al., 2018). Meanwhile, works like (Xu et al., 2018) and (Zhang et al., 2021) seek to improve text-image alignment. Recently, diffusion models have contributed prominently to advancements in text-driven photorealistic and compositional image synthesis (Ramesh et al., 2022; Saharia et al., 2022). For attaining high-resolution imagery, two prevalent strategies emerge. One integrates cascaded super-resolution mechanisms within the RGB domain (Nichol et al., 2021; Ho et al., 2022b; Saharia et al., 2022; Ramesh et al., 2022). In contrast, the other harnesses decoders to delve into latent spaces (Rombach et al., 2022; Gu et al., 2022). Owing to the emergence of robust text-to-

Ultimately, we successfully integrate the benefits of both pixel and latent-based VDMs within a cohesive framework, named Show-1, which achieves state-of-theart performance on popular video generation benchmarks including UCF101 (Soomro et al., 2012), MSRVTT (Xu et al., 2016) and VBench (Huang et al., 2023). Additionally, by exclusively fine-tuning the tem-

a) Key Frame Generation

b) Frame Interpolation

c) Super Resolution

d) Super Resolution

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

…

[Figure 30]

…

…

[Figure 31]

…

[Figure 32]

[Figure 33]

[Figure 34]

…

[Figure 35]

[Figure 36]

[Figure 37]

Decoder

Encoder

[Figure 38]

UNet (interp.)

UNet (base)

UNet (super-res)

UNet (super-res)

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

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

“Toad practicing karate.”

Low-resolution Pixel-based Diffusion

High-resolution Latent-based Diffusion

- Fig. 4: Overview of Show-1. Pixel-based VDMs produce videos of lower resolution with better text-video alignment, while latent-based VDMs upscale these low-resolution videos from pixel-based VDMs to then create high-resolution videos with low computation cost.

image diffusion models, we are able to utilize them as solid initialization of text to video models.

Text-to-video generation. Past research has utilized a range of generative models, including GANs (Vondrick et al., 2016; Saito et al., 2017; Tulyakov et al., 2018; Tian et al., 2021; Shen et al., 2023), autoregressive models (Srivastava et al., 2015; Yan et al., 2021; Le Moing et al., 2021; Ge et al., 2022; Hong et al., 2022; Kondratyuk et al., 2023), and implicit neural representations (Skorokhodov et al., 2021; Yu et al., 2021). Inspired by the notable success of the diffusion model in image synthesis, several recent studies have ventured into applying diffusion models for both conditional and unconditional video synthesis (Voleti et al., 2022; Harvey et al., 2022; Zhou et al., 2022; Wu et al., 2022b; Blattmann et al., 2023b; Khachatryan et al., 2023; H¨ppe et al., 2022; Voleti et al., 2022; Yang et al., 2022; Nikankin et al., 2022; Luo et al., 2023; An et al., 2023; Wang et al., 2023b). Several studies have investigated the hierarchical structure, encompassing separate keyframes, interpolation, and super-resolution modules for high-fidelity video generation. Magicvideo (Zhou et al., 2022), VideoFactory (Wang et al., 2023b), NUWA-XL (Yin et al., 2023), LaVie (Wang et al., 2023c), VideoCrafter (Chen et al., 2023) and Video LDM (Blattmann et al., 2023a) ground their models on latent-based VDMs. On the other hand, PYoCo (Ge et al., 2023), MakeA-Video (Singer et al., 2022), Lumiere Bar-Tal et al. (2024) and Imagen Video (Ho et al., 2022a) anchor their models on pixel-based VDMs. These methods primarily rely on either pixel-based VDM or latentbased VDM. Using only pixel-based VDM results in improved text-video alignment and motion fidelity, but at the expense of significant computational resources. On the other hand, relying solely on latent-based VDM is more efficient, yet it presents challenges in achieving high-quality text-video alignment and motion fidelity.

Unlike these methods, our approach investigates how to effectively combine pixel-based and latent-based VDMs, leveraging the strengths and avoiding the weaknesses of both pixel-based and latent-based VDMs.

3 Show-1 3.1 Preliminaries

Denoising Diffusion Probabilistic Models (DDPMs) (Ho et al., 2020) are generative models that utilize a reverse Markov chain to synthesize data, beginning from a noise distribution and progressively denoising it. This process is driven by optimizing model parameters to align the reverse sequence with the forward noisy sequence. The training objective focuses on minimizing the difference between the actual noise and the noise estimated by the model, formalized as follows:

Ex,ϵ∼N(0,1),t ∥ϵ − ϵθ(xt,t)∥22 . (1)

This expression represents the expected value of the squared L2 norm between the noise ϵ and the noise predicted by the model ϵθ, where ϵ is drawn from a standard Gaussian distribution and xt is the noisy data at timestep t. The model’s parameters θ are trained to minimize this expectation, which corresponds to denoising the data point xt.

### UNet architecture for text-to-image model.

The UNet model is first introduced by (Ronneberger et al., 2015) for biomedical image segmentation. Popular UNet for text-to-image diffusion model usually contains multiple down, middle, and up blocks. Each block consists of a ResNet2D layer, a self-attention layer, and a cross-attention layer. The cross-attention layers play a crucial role in fusing images and texts, allowing textto-image models to generate images that are consistent

with textual descriptions. Text condition c is inserted into cross-attention layer as keys and values. For a textguided diffusion model, with the text embedding c, the objective is given by:

Ex,ϵ∼N(0,1),t,c ∥ϵ − ϵθ(xt,t,c)∥22 . (2)

3.2 Turn Image UNet to Video

We use the spatial weights from a robust text-to-image model. To endow the model with temporal understanding and produce coherent frames, as shown in Fig. 5, we integrate temporal layers within each UNet block. Specifically, after every Resnet2D block, we introduce a temporal convolution layer consisting of four 1D convolutions across the temporal dimension. Additionally, following each spatial self- and cross-attention layer, we implement a temporal attention layer to facilitate dynamic temporal data assimilation. Formally, given a frame-wise input video x ∈ RN×C×H×W, where N is number of frames, C is the number of channels, H and W are the spatial latent dimensions, the spatial selfattention layer operates the input video as a sequence of independent spatial images by transposing the temporal axis into the batch dimension, as illustrated below using einops (Rogozhnikov, 2022) (Here, we include the batch size B to better illustrate the transpose operation. After this, we omit B for simplicity in notation.):

xSA ← rearrange(x,(B N C H W → (B N) (H W) C).

For temporal self-attention layer, the video is reshaped back to temporal dimensions:

xTA ← rearrange(x,(B N C H W → (B H W) N C).

The attention mechanism (Vaswani et al., 2017) implements Attention(Q,K,V ) = Softmax(QK

T

d ) · V , with Q = WQx,K = WKx,V = WV x,

√

where WQ, WK, and WV are learnable matrices that project the inputs to query, key and value, respectively, and d is the output dimension of key and query features. The x is transposed to xSA and xTA for spatial and temporal self-attention respectively. Differently, the crossattention layer receives key and value matrices from the text prompt:

Q = WQxSA,K = WKc,V = WV c,

where c ∈ RN×L×C is the encoded text embedding and L denotes the sequence length of text embedding.

3.3 Pixel-based Keyframe Generation Model

Given a text input, we initially produce a sequence of keyframes using a pixel-based Video UNet at a very low spatial and temporal resolution (Fig. 4, Stage a. This approach results in improved text-to-video alignment. The reason for this enhancement is that we do not require the keyframe modules to prioritize appearance clarity or temporal consistency given that the resolution of video is very low. As a result, the keyframe modules pay more attention to the text guidance. The training objective for the keyframe modules is following Eq. 2.

Why we choose pixel diffusion over latent diffusion here? 1)Latent diffusion employs an encoder to transform the original input x into a latent space. This results in a reduced spatial dimension, for example, H/8,W/8, while concentrating the semantics and appearance into this latent domain. For generating keyframes, our objective is to have a smaller spatial dimension, like 64 × 40. If we opt for latent diffusion, this spatial dimension would shrink further to around 8 × 5, which is not be sufficient to retain ample spatial semantics and appearance within the compacted latent space, resulting in poor text-video alignment as shown in Tab. 4. On the other hand, pixel diffusion operates directly in the pixel domain, keeping the original spatial dimension intact. This ensures that necessary semantics and appearance information are preserved. For the following low resolution stages, we all utilize pixel-based VDMs for the same reason. 2) An alternative is to lower the compression ratio of latent VDMs. Yet, as highlighted in (Rombach et al., 2022), latent diffusion’s main goal is to cut down on computational and memory demands significantly. For instance, stable diffusion compresses a 512×512 image to a 64×64 latent size, achieving 8-fold reduction. However, with a minimal compression ratio, like 2-fold, the efficiency and training costs become comparable to pixel diffusion, as stated in (Rombach et al., 2022). Thus, with a low compression ratio, latent diffusion may be unnecessary, especially since it requires training an extra autoencoder, whereas pixel diffusion does not. 3) Another approach involves using latent-based VDM to generate high-resolution keyframes. However, as indicated in Tab. 4, directly generating high-resolution keyframes leads to poorer text-video alignment and motion quality compared to generating low-resolution keyframes with pixel-based VDM. Furthermore, as the resolution increases (512 × 320 vs 256 × 160 in Tab. 4), both textvideo alignment and motion fidelity deteriorate. These findings suggest that at higher resolutions, the latent model may focus more on spatial appearance, poten-

tially neglecting text-video alignment and motion fidelity.

)*+×0×-%∈ℝ

|Insert<br><br>TemporalA7en8on "(BN)CHW(BHW)NC→<br><br>[Figure 74]<br><br>(BHW)NC(BN)CHW→|
|---|

Insert

"

[Figure 75]

"

[Figure 76]

❄

❄

[Figure 77]

❄

[Figure 78]

TemporalA7en8on

)*+×-×.×/"∈ℝ

)*+×-×.×/"∈ℝ

(BN)(HW)C(BN)CHW→

(BN)CHW(BN)(HW)C→

TemporalConv1D

(BN)(HW)C(BN)CHW→

(BN)CHW(BN)(HW)C→

(BN)CHW(BHW)NC→

(BHW)CN(BN)CHW→

(BHW)NC(BN)CHW→

(BN)CHW(BHW)CN→

CrossA7en8on

- 3.4 Temporal Interpolation Model

SelfA7en8on

ResNet2D

We enhance the temporal resolution of videos with a pixel-based temporal interpolation diffusion model (Fig. 4, Stage b), which iteratively predicts the intermediate frames between the keyframes produced by the previous keyframe model (Sec. 3.3). We employ the masking technique, as highlighted in (Blattmann et al., 2023a), where the target intermediate frames to be interpolated are masked during training process. We inherit the UNet architecture from keyframe model and modify the input channels of the first convolution layer to accommodate the masked key frames as condition via channel-wise concatenation. Specifically, we start from the noisy video frames segment {xit,xjt,xjt+1,xjt+2,xit+1} ∈ R5×C×H×W at timestep t, where x{ti,i+1} are two consecutive key frames and xt{j,j+1,j+2} are three intermediate frames to be interpolated. As depicted in Fig. 6 (Interpolation), we concatenate them with the original key

Fig. 5: UNet block of Show-1. We modify the 2D UNet by inserting temporal convolution and attention layers inside each block. During training, we update the additional temporal layers while keeping spatial layers fixed.

Notably, we reuse the pretrained weights of keyframe model, exluding the last four channels of the first convolution layer, to finetune the interpolation model for fast convergence.

3.5 Super-resolution at Low Spatial Resolution

Upscaling a low-resolution video by 8× presents a significant challenge for a single super-resolution module, given that a video with low resolution, such as one with dimensions of 64×40, lacks sufficient visual detail. To address this, we divide the super-resolution process into two distinct modules. The initial module is tasked with enhancing the spatial quality of the low-resolution video, while the subsequent module is dedicated to generating the final high-resolution output.

- frames x0 ∈ R5×C×H×W and addition binary masks m ∈ R5×1×H×W along the channel dimension as conditioning signals, resulting in an input shape of

5 × (C + C + 1) × H × W. We set x{0j,j+1,j+2} and m{j,j+1,j+2} to 0, indicating the frames to be inter-

polated. Note that x0 and m serve as the conditions. The UNet takes the concatenation with the shape of 5 × (C + C + 1) × H × W as its input. Then the UNet outputs noise with a shape of 5 × C × H × W as the prediction of the noise at timestep t for {xit,xjt,xjt+1,xjt+2,xit+1} ∈ R5×C×H×W. We apply noise conditioning augmentation to conditional key

- frames xi0 and xi0+1 by adding a small amount of random noise. Such augmentation is pivotal in cascaded diffusion models for conditional generation, as observed by (Ho et al., 2022a), and also in text-to-image models as noted by (He et al., 2022). It aids in the simultaneous training of diverse models in a cascade manner and minimizes the vulnerability to domain disparities between the output from previous phase and the training inputs of the following phase. Let the interpolated

In the first low resolution video upsampling module (Fig. 4, Stage c), we introduce a pixel super-resolution approach utilizing the video UNet. The super-resolution model takes as input a low-resolution video x′ ∈ R4N×C×H×W produced by previous stages and outputs a high-resolution video x′′ ∈ R4N×C×4H×4W with a 4× increase in spatial dimension. Similar to the channel-wise conditioning in interpolation model (Sec. 3.4), we concatenate the noisy video frames x′′t ∈ R4N×C×4H×4W at the timestep t with the resized low-resolution video clip x′resized ∈ R4N×C×4H×4W, which is bilinearly upsampled to fit the spatial size of high-resolution video ( Fig. 6, Super Resolution 1). The UNet takes the concatenation [x′′t ,x′resized] with the shape of 4N × (C + C) × 4H × 4W as its input. Then the UNet outputs noise with a shape of 4N × C × 4H × 4W as the prediction of the noise at timestep t for x′′t ∈ R4N×C×4H×4W. In line with the approach Imagen Video (Ho et al., 2022a), we employ gaussian noise augmentation to the upscaled low resolution video condition during its training process, introducing a random signal-to-noise ratio. This aug-

video frames be represented by x′ ∈ R4N×C×H×W(x′t can be regarded as the combination of multiple overlap

segments {xit,xjt,xjt+1,xjt+2,xit+1}. Here we use 4N instead of 4N − 3 for simpler notation). Based on Eq. 2, we can formulate the updated objective as:

Ex′,x0,m,ϵ∼N(0,1),t,c ∥ϵ − ϵθ([x′t,x0,m],t,c)∥22 .

(3)

1) Interpolation

However, injecting the low-resolution input via channel-wise concatenation, as per the first superresolution module and SDx42, is inadequate for addressing these artifacts. This approach results in the poor temporal consistency for high spatial resolution,

|[Figure 79]<br><br>#%&|[Figure 80]<br><br>#%(|[Figure 81]<br><br>#%(#$|[Figure 82]<br><br>#%(#)|[Figure 83]<br><br>#%&#$|
|---|---|---|---|---|
|[Figure 84]<br><br>#'&|0|0|0|[Figure 85]<br><br>#'&#$|
|1|0|0|0|1|

noisy video "! condi-on ""

mask !

- as demonstrated in the X-T slice of Fig. 3. To overcome this issue, we introduce an expert translation for a latent-based Video Diffusion Model (VDM), which proves to be effective in higher resolution stages. This involves two key modifications from the SDx4 approach. Firstly, we implement a noising-denoising process, as outlined by SDEdit (Meng et al., 2021), on the encoded low-resolution videos from earlier stages. These processed videos serve as the input for the UNet, and we do this without appending any extra channels. Specifically, SDEdit utilizes the pretrained diffusion with timesteps ranging from 0 to 1000 but begins inference from a noisy input at an intermediate timestep. Inspired by this, we take low-resolution videos from previous stages, linearly interpolate them to a higher resolution, and add noise
- at an intermediate timestep. Then we apply the diffusion process from this intermediate timestep to 0 with the same prompt, resulting in more detailed outputs than the original linear interpolation.

2) Super Resolution 1

|[Figure 86]|
|---|
|[Figure 87]|

|[Figure 88]|[Figure 89]|[Figure 90]|[Figure 91]|
|---|---|---|---|

noisy video "!## condi-on "$%&'(%#

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

bilinear upsampling

Last frame of previous segment

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

3) Super Resolution 2

|[Figure 101]<br><br>[Figure 102]|[Figure 103]<br><br>[Figure 104]|[Figure 105]<br><br>[Figure 106]|[Figure 107]<br><br>[Figure 108]|[Figure 109]<br><br>[Figure 110]|
|---|---|---|---|---|

bilinear upsampling & add DDPM noise

low-res video

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

- Fig. 6: Illustration of the input for interpolation and super-resolution modules. Interpolation: We concatenate noise with low-FPS frames and a mask that indicates the conditional frames. Super Resolution 1: We resize the low-resolution frames to high-resolution using bilinear upsampling and concatenate them with input noise. We also use the last frame of the previous segment as a condition to enable autoregressive upsampling. Super Resolution 2: We resize the input video to high-resolution, and follow SDEdit (Meng et al., 2021) to add DDPM noise and gradually remove it.

Secondly, as pointed out by (Balaji et al., 2022), various diffusion steps assume distinct roles during the generation process. For instance, the initial diffusion steps, such as from 1000 to 900, primarily concentrate on recovering the overall spatial structure, while subsequent steps delve into finer details. Given our success in securing well-structured low-resolution videos, we suggest adapting the latent VDM to specialize in highresolution detail refinement. More precisely, we train a UNet for only the 0 to 900 timesteps (with 1000 being the maximum) instead of the typical full range of 0 to 1000, directing the model to be an expert emphasizing high-resolution nuances. This strategic adjustment significantly enhances the end video quality, namely expert finetuning. With our first SDEdit modification, we can perform the denoising process from an intermediate timestep, such as 900, for the noisy linearly interpolated video. Therefore, the loss of knowledge from timesteps 1000 to 900 due to expert finetuning is not an issue. Through our empirical observations, we discern that a latent-based VDM with our expert translation can be effectively utilized for enhanced super-resolution with high fidelity and great temporal consistency. This results in the final video, denoted as x′′′ ∈ R4N×C×8H×8W.

mentation can minimize the domain gap between the output from previous interpolation stage and the training inputs of the following stage. During the sampling process, we opt for a consistent signal-to-noise ratio, like 1 or 2. Meanwhile, given that the spatial resolution remains at an upscaled version throughout the diffusion process, it’s challenging to upscale all the interpolated frames in one forward process using a standard GPU with 24GB memory. Consequently, we must divide the frames into four smaller segments and upscale each one individually. However, the continuity between various segments is compromised. To rectify this, as depicted in the Fig. 6, we take the upscaled last frame of one segment to complete the three supplementary channels of the initial frame in the following segment.

- 3.6 Super-resolution at High Spatial Resolution

Sometimes, previous stages may generate videos with artifacts or temporal corruptions as shown in Fig. 3. Therefore, the final super-resolution module (Fig. 4, Stage d) is tasked with managing these artifacts to produce videos of high quality as the end result.

2 https://huggingface.co/stabilityai/stable-diffusion-x4upscaler

Why choose latent-based VDM over pixel-based VDM here? Pixel-based VDMs work directly within the pixel domain, preserving the original spatial dimensions. Handling high-resolution videos this way can be computationally expensive. As shown in Tab. 4, using pixel-based VDM for final super-resolution requires huge GPU memory e.g., 72GB. In contrast, latent-based VDMs compress videos into a latent space (for example, downscaled by a factor of 8), which results in a reduced computational burden. Moreover, although latent-based VDM may result in less precise text-video alignment, it can be re-purposed to translate low-resolution video to high-resolution video, while maintaining the original appearance and the accurate text-video alignment of low-resolution video generated by the pixel base-VDM. Thus, we opt for the latentbased VDMs here.

Another choice is to reduce the parameters of pixel model. For example, Make-A-Video (Singer et al., 2022) reduces its final superresolution model to 0.7B parameters. However, it still requires substantial computational costs because its UNet directly operates on the high output resolution. We replicate Make-A-Video with its original parameters and architecture (Tab. 5) and find that even with 0.7B parameters, it’s still computationally demanding with 52GB memory, while our latent upsampling only needs 15G. Moreover, upsampling synthetic videos also poses challenges, particularly due to the domain gap between training on real data and testing on synthetic outputs. Achieving temporal consistency and high visual quality while minimizing artifacts requires high model complexity, which is impractical with further reducing parameters.

- 3.7 Motion Customization and Video Stylization.

Drawing from recent advancements in Motion Customization (Zhao et al., 2023; Jeong et al., 2023), we have further developed our model to accommodate these applications. In contrast to the Motion Director (Zhao et al., 2023) approach, which requires separate training for spatial and temporal layers tailored to a specific video, our method stands out by focusing finetuning efforts solely on the temporal attention layers of the keyframes’ UNet. This targeted approach to finetuning is designed to be computationally and memoryefficient. Through this process, we are able to encapsulate the motion dynamics of the given video within the temporal attention layers. It’s important to highlight that the later modules for frame interpolation and spatial super-resolution are left unchanged. This approach allows for flexible video editing/ stylization and tailored

### Table 1: Zero-shot text-to-video generation on UCF-101. Ours achieves competitive results in inception score and FVD metrics.

Method IS (↑) FVD (↓) CogVideo (Hong et al., 2022) (English) 25.27 701.59 Make-A-Video (Singer et al., 2022) 33.00 367.23 MagicVideo (Zhou et al., 2022) - 655.00 Video LDM (Blattmann et al., 2023a) 33.45 550.61 VideoFactory (Wang et al., 2023b) - 410.00 Show-1 (ours) resized 35.67 383.46 Show-1 (ours) finetune on square videos 36.02 369.33

adjustments while maintaining the model’s fundamental capability for broad synthesis.

4 Experiments 4.1 Implementation Details

For the generation of pixel-based keyframes, we produce videos of dimensions 8 × 64 × 40 × 3(N × H × W × 3). In our interpolation model, we initialize the weights using the keyframes generation model and produce videos with dimensions of 29 × 64 × 40 × 3. For the first superresolution module, we upsample the video yielding the size 29 × 256 × 160. In the subsequent superresolution module, we modify the latent-based VDM and use our proposed expert translation to generate videos of 29 × 576 × 320.

In terms of training, we employ the public WebVid10M dataset (Bain et al., 2021) as our video training data. Our infrastructure comprised 64 A100 GPUs, each with 40GB, which stands in contrast to the setups used by LaVie (Wang et al., 2023c), ModelScope (Wang et al., 2023a), or (Chen et al., 2023) VideoCrafter. These methods train on large scale internal datasets with more than 128 A100 GPUs, each with 80GB, which require much more data and training resources than ours.

Regarding the ablation studies depicted in Fig. 2, Tab. 4 and Tab. 6, we ensure that the same T5 text encoder (Raffel et al., 2020) is employed across both pixel-based and latent-based VDMs in the keyframes stage. Each model is initialized with the image model weights pre-trained on the LAION (Schuhmann et al., 2022) dataset and has the same number of parameters, maintaining consistency for fair comparisons. Regarding comparisons with the state-of-the-art, our choice of initialization for the pre-trained Text-to-Image model is DeepFloyd3, which serves as the foundation for our model training.

3 https://github.com/deep-floyd/IF

### Table 2: Comparisons with SOTA models on MSR-VTT dataset (Xu et al., 2016).

|Models<br><br>|FID-vid (↓) FVD (↓) CLIPSIM (↑)|
|---|---|
|NUWA¨ (Wu et al., 2022a) CogVideo (Chinese) (Hong et al., 2022) CogVideo (English) (Hong et al., 2022) MagicVideo (Zhou et al., 2022) Video LDM (Blattmann et al., 2023a) Make-A-Video (Singer et al., 2022) ModelScopeT2V (Wang et al., 2023a)<br><br>|47.68 - 0.2439 24.78 - 0.2614 23.59 1294 0.2631<br><br>- 1290 -<br>- - 0.2929<br><br><br>13.17 - 0.3049 11.09 550 0.2930<br><br>|
|Show-1(ours)<br><br>|12.97 536 0.3104<br><br>|

- 4.2 Quantitative Results

UCF-101 Experiment. For our preliminary evaluations, we employ IS and FVD metrics. UCF-101 stands out as a categorized video dataset curated for action recognition tasks. When extracting samples from the text-to-video model, following PYoCo (Ge et al., 2023), we formulate a series of prompts corresponding to each class name, serving as the conditional input. This step becomes essential for class names like jump rope, which aren’t intrinsically descriptive. Following (Singer et al.,

- 2022), we generate totally 10000 video samples to determine the IS metric. For FVD evaluation, we adhere to methodologies presented in prior studies (Le Moing et al., 2021; Tian et al., 2021) and produce 2,048 videos. To ensure a fair comparison with the previous methods (Ge et al., 2022; Singer et al., 2022), which produces square videos since it is directly trained on squared videos, we directly resized our generated videos to square videos. However, this resizing process introduces slight performance degradation to our model. We believe that a more rigorous approach would involve fine-tuning our entire pipeline on square videos to better align with the comparison criteria. Consequently, we present the results for both the resized version and the version fine-tuned on square videos in Table 1

From the data presented in Tab. 1, it’s evident that Show-1’s zero-shot capabilities outperform or are on par with other methods. This underscores Show-1’s superior ability to generalize effectively, even in specialized domains. It’s noteworthy that our keyframes, interpolation, and initial super-resolution models are solely trained on the publicly available WebVid-10M dataset, in contrast to the Make-A-Video models, which are trained on large scale internal text-video data.

on performance metrics including FID-vid (Heusel et al., 2017), FVD (Unterthiner et al., 2018), and CLIPSIM (?). For FID-vid and FVD assessments, we randomly select 2,048 videos from the MSR-VTT testing division. CLIPSIM evaluations utilize all the captions from this test subset, following the approach (Singer et al., 2022). All generated videos consistently uphold a resolution of 256 × 256.

Tab. 2 shows that, Show-1 achieves the second best performance in FID-vid (a score of 12.97) and the best FVD (with a score of 536). This suggests a remarkable visual congruence between our generated videos and the original content. Moreover, our model secures a notable CLIPSIM score of 0.3104, emphasizing the semantic coherence between the generated videos and their corresponding prompts. It is noteworthy that our CLIPSIM score surpasses that of Make-A-Video (Singer

- et al., 2022), despite the latter having the benefit of using additional training data beyond WebVid-10M.

VBench Experiment. VBench (Huang et al., 2024) is a benchmark designed for evaluating video generative models by breaking down video generation quality into well-defined dimensions for precise and objective assessment. A Prompt Suite generates videos across various content types for evaluation, while an evaluation method suite offers automated, objective analyses for each dimension. Incorporating human preference annotation ensures VBench’s evaluations align with human perceptions, promising valuable insights and opensource availability.

As illustrated in Tab. 3, out of 16 different evaluation metrics, our approach leads in 10. Notably, these results are obtained by training our Show-1 model on the publicly accessible WebVideo-10M dataset (Bain

- et al., 2021), marking a significant improvement over VideoCrafter (Chen et al., 2023) and LaVie (Wang

et al., 2023c), which are trained on large-scale, proprietary text-video datasets.

4.3 Qualitative Results

Human evaluation. We gather an evaluation set comprising 256 complex prompts that encompass camera control, natural scenery, food, animals, people, and imaginative content. The survey is conducted on Amazon Mechanical Turk. Following Make-A-Video (Singer

- et al., 2022), we assess video quality, the accuracy of text-video alignment and motion fidelity. In evaluating video quality, we present two videos in a random sequence and inquire from annotators which one possesses superior quality. When considering text-video alignment, we display the accompanying text and prompt

MSR-VTT Experiment. The MSR-VTT dataset (Xu et al., 2016) test subset comprises 2,990 videos, accompanied by 59,794 captions. Every video in this set maintains a uniform resolution of 320 × 240. We carry out our evaluations under a zero-shot setting, given that Show-1 has not been trained on the MSR-VTT collection. In this analysis, Show-1 is compared with state-of-the-art models,

#### Table 3: VBench Evaluation Results per Dimension. This table compares the performance of five video generation models across each of the 16 VBench dimensions. A higher score indicates relatively better performance for a particular dimension.

|Models|Consistency<br><br>Subject|Consistency<br><br>Background<br><br>|Flickering<br><br>Temporal|Smoothness<br><br>Motion<br><br>|Degree<br><br>Dynamic|Quality<br><br>Aesthetic<br><br>|Quality<br><br>Imaging<br><br>|Class<br><br>Object|
|---|---|---|---|---|---|---|---|---|
|LaVie ModelScope VideoCrafter CogVideo<br><br>|91.41% 89.87% 86.24%<br><br>92.19%<br><br><br>|97.47% 95.29% 92.88% 95.42%|98.30% 98.28% 97.60% 97.64%<br><br>|96.38%<br><br>95.79% 91.79%<br><br>96.47%<br><br><br>|49.72% 66.39% 89.72% 42.22%|54.94% 52.06% 44.41% 38.18%<br><br>|61.90% 58.57% 57.22% 41.03%<br><br>|91.82% 82.25% 87.34% 73.40%|
|Show-1|95.53%<br><br>|98.02%|99.12%|98.24%<br><br>|44.44%<br><br>|57.35%<br><br>|59.75%|93.07%|

|Models|Objects<br><br>Multiple<br><br>|Action<br><br>Human|Color<br><br>|Relationship<br><br>Spatial|Scene<br><br>|Style<br><br>Appearance<br><br>|Style<br><br>Temporal|Consistency<br><br>Overall|
|---|---|---|---|---|---|---|---|---|
|LaVie ModelScope VideoCrafter CogVideo<br><br>|33.32% 38.98% 25.93% 18.11%|96.80%<br><br>92.40%<br><br>93.00%<br><br><br>78.20%<br><br>|86.33% 81.72%<br><br>78.84%<br><br>79.57%<br><br><br>|34.09% 33.68% 36.74% 18.24%<br><br>|52.69% 39.26% 43.36% 28.24%|23.56% 23.39%<br><br>21.57%<br><br>22.01%<br>|25.93% 25.37% 25.42% 7.80%<br><br>|26.41% 25.67% 25.21% 7.70%|
|Show-1|45.47%<br><br>|95.60%|86.35%<br><br>|53.5%|47.03%<br><br>|23.06%<br><br>|25.28%|27.46%|

Motion Fidelity Text-Video Alignment Video Quality

80

80

80

70

70

70

60

60

60

50

50

50

40

40

40

30

30

30

20

20

20

10

10

10

0

0

0

ModelScope ZeroScope Lavie VideoCrafter ModelScope ZeroScope Lavie VideoCrafter ModelScope ZeroScope Lavie VideoCrafter

Show-1 Baseline Human Evaluation

- Fig. 7: Human Evaluations for ModelScope (Wang et al., 2023a), ZeroScope, VideoCrafter0.9 (Chen et al.,

- 2023), LaVie (Wang et al., 2023c) and our Show-1 model.

swimming in the sea,” to create a new video based on the prompt ”The planes are flying in the sky.” Other methods struggle to alter the original form of the subjects in the video, resulting in unrealistic transformations like a shark-shaped airplane. On the contrary, Show-1 excels in customizing motion, managing even complex compositional adjustments successfully, such as depicting sharks or airplanes moving accurately in their respective environments.

annotators to determine which video aligns better with the given text, advising them to overlook quality concerns. For motion fidelity, we let annotators determine which video has the most natural notion. As shown in Fig. 7, our method achieves the best human preferences on all evaluation parts.

Specifically, our approach exhibits superior textvideo alignment and motion fidelity compared to the recently open-sourced ModelScope (Wang et al., 2023a), ZeroScope, VideoCrafter (Chen et al., 2023) and LaVie (Wang et al., 2023c). Additionally, as depicted in Fig. 8, our method matches or even surpasses the visual quality of the current state-of-the-art methods, including Imagen Video and Make-A-Video. Furthermore, Show-1 surpasses the commercial products Gen-2 and Pika in terms of text-video alignment, as illustrated in Fig. 9.

As shown in Fig. 12, Show-1 is also capable of delivering impressive results in video stylization and editing that align with the accompanying text. Note that the results are from Jeong et al. (2023).

4.4 Ablation Studies

Decide which stage should use pixel or latent, whether to generate high resolution or low resolution. The initial step involves determining the resolution and the VDM employed for the keyframe stages. As illustrated in Tab. 4, utilizing a pixel-based VDM with a low resolution of 64 × 40 outperforms the corresponding latent model (f = 8) at the same resolu-

Motion Customization and Video Editing/stylization. In Fig. 11, we present visual comparisons of Show-1 with four other methods such as Gen-1 (Esser et al., 2023) and Tune-A-Video (Wu et al., 2022b). The goal is to harness the motion captured in the original video, where ”sharks are

## “A blue tiger in the grass in the sunset, surrounded by butterflies.”

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

ModelScope

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

ZeroScope

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Ours

## “Toad practicing karate.”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

ModelScope

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

ZeroScope

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Ours

“A panda taking a selfie.”

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Imagen Video

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Ours

“A musk ox grazing on beautiful wildflowers.”

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Make-A-Video

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Ours

#### Fig. 8: Qualitative comparisons with existing video generative models. Words in red highlight the misalignment between text and video in other open-source approaches (i.e., ModelScope and ZeroScope), whereas our method maintains proper alignment. Videos from closed-source approaches (i.e., Imagen Video and Make-AVideo) are obtained from their websites.

- Table 4: Comparisons of different combinations of pixel-based and latent-based VDMs on keyframes stages and final super-resolution stage in terms of text-video similarity, memory usage during inference, UCF-101 FVD and human evaluations of motion fidelity. The same T5 text encoder is employed across both pixel-based and latent-based VDMs in the keyframes stage. Each model is initialized with the image model weights pre-trained on the LAION dataset and has the same number of parameters, maintaining consistency for fair comparisons. f indicates the latent compression ratio.

|Stage<br><br>Keyframes<br><br>Stage<br><br>Final Super-Res.|SIM<br><br>CLIP<br><br>Memory<br><br>Max<br><br>FVD↓<br><br>UCF-101<br><br>Alignment<br><br>Text-Video<br><br>Fidelity<br><br>Motion|
|---|---|
|– pixel 64 × 40/ pixel latent|– 72GB – – –<br><br>0.3096 15GB 383 36% 23%|
|64 × 40/ latent f = 8 latent 64 × 40/ latent f = 4 latent 64 × 40/ latent f = 2 latent<br><br>|0.2441 15GB 584 1% 2% 0.2524 15GB 552 1% 5% 0.2742 15GB 465 2% 4%|
|256 × 160/ pixel pixel|0.2784 48GB 462 3% 6%|
|256 × 160/ latent f = 8 latent 256 × 160/ latent f = 4 latent 256 × 160/ latent f = 2 latent<br><br>|0.2874 15GB 416 11% 15% 0.2897 15GB 403 16% 11% 0.2834 26GB 429 8% 9%|
|512 × 320/ latent f = 8 latent 512 × 320/ latent f = 4 latent 512 × 320/ latent f = 2 latent<br><br>|0.2793 15GB 487 7% 10% 0.2879 26GB 426 9% 9% 0.2767 48GB 451 6% 6%|

- Table 5: Comparisons of parameters and speed between Make-A-Video and our method. The numbers are reported in the format of Make-A-Video (Singer et al., 2022) / ours

Stage Prior Keyframes Temp. Interp. Super1 Final Super Total Step 64/- 100/ 75 50/ 75 50/ 50 50/ 40 Para. 1.3B/ - 3.1B/ 1.7B 3.1B/ 1.7B 1.4B/ 0.8B 0.7B/ 1.8B 9.6B/ 6B Time 3s/ – 58s/ 30s 62s/ 60s 70s/ 65s 63s/ 23s 256s/ 178s Memory 7GB/ – 18GB/ 11GB 14GB/ 10GB 52GB/ 14GB 54GB/ 15GB –/– FVD – 569 542 474 383 –

A panda besides the waterfall is holding a sign that says "Show Lab".

### Table 6: Ablation study of our final superresolution module on UCF-101.

[Figure 156]

[Figure 157]

[Figure 158]

Methods FVD(↓) IS(↑) Sdx4 with temporal 459 32.98 Expert translation

Show-1

[Figure 159]

[Figure 160]

[Figure 161]

change input 423 33.83 + expert finetuning 383 35.67

Gen-2

tails as outlined by the text prompt. Additionally, the 64 × 40 pixel VDM also outshines the 256 × 160 latentbased VDM in performance, and when the resolution is increased to 512 × 320, the results diminish, indicating that the latent model may focus more on spatial appearance at higher resolutions, possibly neglecting alignment with the text.Meanwhile, at a resolution of 64×40, the text-video alignment significantly decreases with larger f values. At a resolution of 256 × 160 and 512x320, all values of f(0,2,4,8) result in worse textvideo alignment and efficiency compared to 64 × 40 with f = 0. In conclusion, these findings indicate that starting with very low-resolution keyframes using pixelbased VDM(f = 0) yields the best alignment between video and text, along with motion quality. Given that subsequent stages also work with low-resolution video,

[Figure 162]

[Figure 163]

[Figure 164]

Pika

- Fig. 9: Qualitative comparisons with Gen-2 and Pika. Gen-2 and Pika face challenges in accurately rendering text in videos. Conversely, Show-1 is capable of precise text rendering, indicating superior alignment between text and video.

tion. This suggests the difficulty for a small latent space (e.g., 8 × 5 for videos of 64 × 40 resolution) to capture the comprehensive and necessary visual semantic de-

“A married couple embraces in front of a burning house.”

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

w/o expert finetuning

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

w/ expert finetuning

- Fig. 10: Effect of expert finetuning. With expert finetuning, the visual quality is significantly improved.

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

Source

Gen-1

Tune-A-Video

Show-1

- Fig. 11: Qualitative comparisons for motion customization. The objective is to utilize the movement from the source video to generate a new video following the prompt ”The planes are flying in the sky.” Other methods often fail to modify the subjects’ original shapes in the video, leading to implausible transformations, such as shark-shaped airplanes. In contrast, Show-1 demonstrates superior ability in adapting motion effectively. All results are from Jeong et al. (2023).

source

[Figure 185]

[Figure 186]

[Figure 187]

animation style

[Figure 188]

[Figure 189]

[Figure 190]

Lamborghini in the space

[Figure 191]

[Figure 192]

[Figure 193]

Tank on the snow road

[Figure 194]

[Figure 195]

[Figure 196]

Fig. 12: Visualizations for video stylization and editing results of Show-1. All results are from Jeong et al. (2023).

UNet input and implementing expert fine-tuning, which entails training the latent-based VDMs over timesteps 0-900 out of a maximum of 1000. According to Tab. 6, models enhanced with expert translation yield videos of higher quality compared to the standard SDx4 model equipped with the temporal layers. Furthermore, as depicted in Fig. 10, the visuals demonstrate that our expert fine-tuning approach results in reduced artifacts and captures more complex details.

pixel-based VDM is chosen for these phases as well. However, due to the significantly higher computational cost of pixel models at high resolutions, as shown in Tab. 4, we opt for the latent VDM for our final superresolution module.

Inference Speed. Although hierarchical structures require more inference time compared to single-stage models, their outcomes are significantly superior, as evidenced by advanced generation methods like (Ho et al., 2022a; Singer et al., 2022; Blattmann et al., 2023a). These SOTA methods all employ hierarchical frameworks, including keyframe generation, temporal

Impact of expert translation of latent-based VDM as final super-resolution module. We present ablations with and without the incorporation of expert translation. Detailed in Section 3.6, ”expert translation” involves two key changes: modifying the

interpolation, and superresolution, for video creation. We replicated the Make-A-Video model by precisely matching its parameters and network architecture for inference time and parameters comparisons. As shown in Tab. 5, the results show that our method is faster and more memory efficient than the previous SOTA method, Make-A-Video.

### 5 Conclusion

We introduce Show-1, a novel model that marries the strengths of pixel and latent based VDMS. Our approach employs pixel-based VDMs for initial video generation, ensuring precise text-video alignment and motion portrayal, and then uses latent-based VDMs for super-resolution, transitioning from a lower to a higher resolution efficiently. This combined strategy offers high-quality text-to-video outputs while optimizing computational costs.

### Ackmowledgement

This research is supported by the Ministry of Education, Singapore, under the Academic Research Fund Tier 1 (FY2023). The computational work for this article was partially performed on resources of the National Supercomputing Centre, Singapore.

### References

An J, Zhang S, Yang H, Gupta S, Huang JB, Luo J, Yin X (2023) Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:230408477

Bain M, Nagrani A, Varol G, Zisserman A (2021) Frozen in time: A joint video and image encoder for end-to-end retrieval. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1728–1738

Balaji Y, Nah S, Huang X, Vahdat A, Song J, Zhang Q, Kreis K, Aittala M, Aila T, Laine S, Catanzaro B, et al. (2022) eDiff-I: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:221101324

Bar-Tal O, Chefer H, Tov O, Herrmann C, Paiss R, Zada S, Ephrat A, Hur J, Li Y, Michaeli T, et al. (2024) Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:240112945 Blattmann A, Rombach R, Ling H, Dockhorn T, Kim SW, Fidler S, Kreis K (2023a) Align your latents: High-resolution video synthesis with latent diffusion

models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 22563–22575

Blattmann A, Rombach R, Ling H, Dockhorn T, Kim SW, Fidler S, Kreis K (2023b) Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. In: CVPR

Chen H, Xia M, He Y, Zhang Y, Cun X, Yang S, Xing J, Liu Y, Chen Q, Wang X, et al. (2023) Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:231019512

Esser P, Chiu J, Atighehchian P, Granskog J, Germanidis A (2023) Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:230203011

Ge S, Hayes T, Yang H, Yin X, Pang G, Jacobs D, Huang JB, Parikh D (2022) Long video generation with time-agnostic vqgan and time-sensitive transformer. arXiv preprint arXiv:220403638

Ge S, Nah S, Liu G, Poon T, Tao A, Catanzaro B, Jacobs D, Huang JB, Liu MY, Balaji Y (2023) Preserve your own correlation: A noise prior for video diffusion models. arXiv preprint arXiv:230510474

Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, WardeFarley D, Ozair S, Courville A, Bengio Y (2014) Generative adversarial networks. NIPS

Gu S, Chen D, Bao J, Wen F, Zhang B, Chen D, Yuan L, Guo B (2022) Vector quantized diffusion model for text-to-image synthesis. In: CVPR, pp 10696–10706

Harvey W, Naderiparizi S, Masrani V, Weilbach C, Wood F (2022) Flexible diffusion modeling of long videos. arXiv preprint arXiv:220511495

He Y, Yang T, Zhang Y, Shan Y, Chen Q (2022) Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:221113221

Heusel M, Ramsauer H, Unterthiner T, Nessler B, Hochreiter S (2017) Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30

Ho J, Jain A, Abbeel P (2020) Denoising diffusion probabilistic models. NeurIPS 33:6840–6851

Ho J, Chan W, Saharia C, Whang J, Gao R, Gritsenko A, Kingma DP, Poole B, Norouzi M, Fleet DJ, et al. (2022a) Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:221002303

Ho J, Saharia C, Chan W, Fleet DJ, Norouzi M, Salimans T (2022b) Cascaded diffusion models for high fidelity image generation. JMLR 23:47–1

Hong S, Yang D, Choi J, Lee H (2018) Inferring semantic layout for hierarchical text-to-image synthesis. In: CVPR, pp 7986–7994

Hong W, Ding M, Zheng W, Liu X, Tang J (2022) Cogvideo: Large-scale pretraining for textto-video generation via transformers. arXiv preprint arXiv:220515868

H¨ppe T, Mehrjou A, Bauer S, Nielsen D, Dittadi A

(2022) Diffusion models for video prediction and infilling. arXiv preprint arXiv:220607696

Huang Z, He Y, Yu J, Zhang F, Si C, Jiang Y, Zhang Y, Wu T, Jin Q, Chanpaisit N, et al. (2023) Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:231117982

Huang Z, He Y, Yu J, Zhang F, Si C, Jiang Y, Zhang Y, Wu T, Jin Q, Chanpaisit N, Wang Y, Chen X, Wang L, Lin D, Qiao Y, Liu Z (2024) VBench: Comprehensive Benchmark Suite for Video Generative Models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

Jeong H, Park GY, Ye JC (2023) Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. arXiv preprint arXiv:231200845

Khachatryan L, Movsisyan A, Tadevosyan V, Henschel R, Wang Z, Navasardyan S, Shi H (2023) Text2videozero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:230313439 Kondratyuk D, Yu L, Gu X, Lezama J, Huang J, Hornung R, Adam H, Akbari H, Alon Y, Birodkar V, et al. (2023) Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:231214125

Le Moing G, Ponce J, Schmid C (2021) Ccvs: Contextaware controllable video synthesis. NeurIPS

Luo Z, Chen D, Zhang Y, Huang Y, Wang L, Shen Y, Zhao D, Zhou J, Tan T (2023) VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation. In: CVPR

Meng C, He Y, Song Y, Song J, Wu J, Zhu JY, Ermon S (2021) Sdedit: Guided image synthesis and editing with stochastic differential equations. In: International Conference on Learning Representations

Nichol A, Dhariwal P, Ramesh A, Shyam P, Mishkin P, McGrew B, Sutskever I, Chen M (2021) Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:211210741

Nikankin Y, Haim N, Irani M (2022) Sinfusion: Training diffusion models on a single image or video. arXiv preprint arXiv:221111743

Raffel C, Shazeer N, Roberts A, Lee K, Narang S, Matena M, Zhou Y, Li W, Liu PJ (2020) Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research 21(140):1–67, URL http://jmlr.org/

papers/v21/20-074.html

Ramesh A, Dhariwal P, Nichol A, Chu C, Chen M (2022) Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:220406125

Reed S, Akata Z, Yan X, Logeswaran L, Schiele B, Lee H (2016) Generative adversarial text to image synthesis. In: ICML, PMLR, pp 1060–1069

Rogozhnikov A (2022) Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation. In: International Conference on Learning Representations, URL https://openreview.net/forum?id= oapKSVM2bcj

Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B (2022) High-resolution image synthesis with latent diffusion models. In: CVPR, pp 10684–10695

Ronneberger O, Fischer P, Brox T (2015) U-net: Convolutional networks for biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18 pp 234–241

Saharia C, Chan W, Saxena S, Li L, Whang J, Denton E, Ghasemipour SKS, Ayan BK, Mahdavi SS, Lopes RG, et al. (2022) Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:220511487

Saito M, Matsumoto E, Saito S (2017) Temporal generative adversarial nets with singular value clipping. In: ICCV

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, et al. (2022) Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:221008402

Shen X, Li X, Elhoseiny M (2023) MoStGAN-V: Video Generation With Temporal Motion Styles. In: CVPR

Singer U, Polyak A, Hayes T, Yin X, An J, Zhang S, Hu Q, Yang H, Ashual O, Gafni O, et al. (2022) Makea-video: Text-to-video generation without text-video data. arXiv preprint arXiv:220914792

Skorokhodov I, Tulyakov S, Elhoseiny M (2021) Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. arXiv preprint arXiv:211214683

Soomro K, Zamir AR, Shah M (2012) Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:12120402

Srivastava N, Mansimov E, Salakhudinov R (2015) Unsupervised learning of video representations using lstms. In: ICML

Tian Y, Ren J, Chai M, Olszewski K, Peng X, Metaxas DN, Tulyakov S (2021) A Good Image Generator Is

What You Need for High-Resolution Video Synthesis. In: ICLR

Tulyakov S, Liu MY, Yang X, Kautz J (2018) MoCoGAN: Decomposing Motion and Content for Video Generation. In: CVPR

Unterthiner T, Van Steenkiste S, Kurach K, Marinier R, Michalski M, Gelly S (2018) Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:181201717

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser  L, Polosukhin I (2017) Attention is all you need. Advances in neural information processing systems 30

Voleti V, Jolicoeur-Martineau A, Pal C (2022) Masked conditional video diffusion for prediction, generation, and interpolation. arXiv preprint arXiv:220509853 Vondrick C, Pirsiavash H, Torralba A (2016) Generat-

ing videos with scene dynamics. NIPS

Wang J, Yuan H, Chen D, Zhang Y, Wang X, Zhang S (2023a) Modelscope text-to-video technical report. arXiv preprint arXiv:230806571

Wang W, Yang H, Tuo Z, He H, Zhu J, Fu J, Liu J (2023b) Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:230510874

Wang Y, Chen X, Ma X, Zhou S, Huang Z, Wang Y, Yang C, He Y, Yu J, Yang P, et al. (2023c) Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:230915103 Wu C, Liang J, Ji L, Yang F, Fang Y, Jiang D, Duan N (2022a) Nu¨wa: Visual synthesis pre-training for neural visual world creation. In: ECCV, Springer, pp 720–736

Wu JZ, Ge Y, Wang X, Lei W, Gu Y, Hsu W, Shan Y, Qie X, Shou MZ (2022b) Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:221211565

Xu J, Mei T, Yao T, Rui Y (2016) Msr-vtt: A large video description dataset for bridging video and language. In: CVPR

Xu T, Zhang P, Huang Q, Zhang H, Gan Z, Huang X, He X (2018) Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In: CVPR, pp 1316–1324

Yan W, Zhang Y, Abbeel P, Srinivas A (2021) Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:210410157

Yang R, Srivastava P, Mandt S (2022) Diffusion probabilistic modeling for video generation. arXiv preprint arXiv:220309481

Yin S, Wu C, Yang H, Wang J, Wang X, Ni M, Yang Z, Li L, Liu S, Yang F, et al. (2023) Nuwa-xl: Diffusion over diffusion for extremely long video generation.

arXiv preprint arXiv:230312346 Yu S, Tack J, Mo S, Kim H, Kim J, Ha JW, Shin J

(2021) Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks. In: ICLR Zhang H, Xu T, Li H, Zhang S, Wang X, Huang X, Metaxas DN (2017) Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In: ICCV, pp 5907–5915

Zhang H, Koh JY, Baldridge J, Lee H, Yang Y (2021) Cross-modal contrastive learning for text-to-image generation. In: CVPR, pp 833–842

Zhao R, Gu Y, Wu JZ, Zhang DJ, Liu J, Wu W, Keppo J, Shou MZ (2023) Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:231008465

Zhou D, Wang W, Yan H, Lv W, Zhu Y, Feng J (2022) Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:221111018

