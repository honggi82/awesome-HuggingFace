# arXiv:2410.11795v2[cs.CV]16Oct2024

[Figure 1]

## Efficient Diffusion Models: A Comprehensive Survey from Principles to Practices

Zhiyuan Ma 1∗, Yuzhu Zhang2, Guoli Jia1, Liangliang Zhao1, Yichao Ma2, Mingjie Ma2, Gaofeng Liu3, Kaiyan Zhang1, Jianjun Li2, Bowen Zhou1,4†

1Tsinghua University, 2HUST, 3 SJTU, 4 Shanghai AI Lab

### Abstract

As one of the most popular and sought-after generative models in the recent years, diffusion models have sparked the interests of many researchers and steadily shown excellent advantage in various generative tasks such as image synthesis, video generation, molecule design, 3D scene rendering and multimodal generation, relying on their dense theoretical principles and reliable application practices. The remarkable success of these recent efforts on diffusion models comes largely from progressive design principles and efficient architecture, training, inference, and deployment methodologies. However, there has not been a comprehensive and in-depth review to summarize these principles and practices to help the rapid understanding and application of diffusion models. In this survey, we provide a new efficiency-oriented perspective on these existing efforts, which mainly focuses on the profound principles and efficient practices in architecture designs, model training, fast inference and reliable deployment, to guide further theoretical research, algorithm migration and model application for new scenarios in a reader-friendly way. https://github.com/ponyzym/Efficient-DMs-Survey

### 1 Introduction

Recent years have witnessed the remarkable success of diffusion models (DMs) [1–3], accompanied by a range of visually stunning generative contents emerging. After surpassing GAN on image synthesis [4], DMs have shown a promising algorithm in a wide variety of downstream applications such as image synthesis [5–10], video generation [11–19], audio synthesis [20–22], 3D rendering and generation [23–27] etc., and have emerged as the new state-of-the-art generative models family. Behind these attractive works, the DMs have denser theoretical basis than other generative families such as Variational AutoEncoders (VAEs) and Generative Adversarial Networks (GANs) and a lot of previous efforts have focused on sampling procedure [28–31], conditional guidance [32–35], likelihood maximization [36–39] and generalization ability [40–42] to improve their efficiency and performance for more powerful generative abilities. Standing on the shoulders of these extensive works on the principles and practices of DMs, we have almost seen DMs become a competitive counterpart to LLMs and almost together become the two most brilliant diamonds in the generative AI community today. However, for LLMs, there are already many comprehensive reviews that explain their efforts in efficient architecture design, model training, supervise fine-tuning, preference aligning as well as corresponding applications, but in the field of DMs, existing surveys [43–46] still have a significant limitation in comprehensive and in-depth summarize these previous principles and practices (refer to Figure. 1), for helping rapid understanding and application in future works.

∗Zhiyuan Ma is the project leader. †Corresponding author.

Preprint. Under review.

[Figure 2]

RectifiedFlow

[Figure 3]

SnapFusion

[Figure 4]

[Figure 5]

[Figure 6]

UFOGen

StreamingT2V

[Figure 7]

InstaFlow

SD-3.0

[Figure 8]

[Figure 9]

###### Practices Principles

[Figure 10]

[Figure 11]

MobileDiff

Vidu

[Figure 12]

ADD

SV3D

Make-a-Video

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

CfDG

ImagenVideo

GuideDistill

Latte

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Score-SDE

LoRA

DiTs

Pixart-

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

DDIM

DPM-Solver

CMs

Sora

[Figure 26]

[Figure 27]

[Figure 28]

DDPM

ProgreDistill

SDXL

Lumiere

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

SMLD

DALLE-2

DreamBooth

VideoCrafter

[Figure 34]

[Figure 35]

[Figure 36]

DPMs

LDMs

ControlNet

MagicVideo

2015 - 2021 2022 2023 2024

Figure 1: The timeline of efficient DMs.

Besides, a noteworthy trend is that, driven by the advantages of self-attention and deep scalable architecture, LLMs have acquired powerful language emergence capabilities. However, current DMs still face a scalability dilemma [47], which will play a critical role in supporting large-scale deep generative training and giving rise to emergent abilities [48] similar to LLMs [49]. Representatively, the recent emergence of Sora [50] has pushed the intelligent emergence capabilities of generative models to a climax by treating video models as world simulators. While unfortunately, Sora is still a closed-source system and the mechanism for the intelligence emergence is still not very clear.

In this survey, we aim to present an exhaustive organization of the recent advancements in the rapidly evolving field of efficient DMs to promote the intelligence emergence of generative models, as depicted in Figure.2. We organize the literature in a taxonomy consisting of six primary categories, encompassing various aspects of efficient DMs, including principles, efficient architecture, efficient training and fine-tuning, efficient sampling and inference, deployment, and applications.

- • Principle focuses on the dense theoretical foundation of DMs to explain and reveal the essential reasons for its generative effectiveness by sorting out relevant theories, such as dynamic modeling, score matching, latent projecting, and conditional guidance, to promote the development of new theories and guide various efficient generative practices.
- • Efficient Architecture explores the mainstream backbone networks of the DMs, including: U-Net, DiT, U-ViT, MamBa, etc., and analyzes their design structures to compare their respective advantages and disadvantages, in order to guide the emergence of more powerful new deep scalable architectures.
- • Efficient Training and Fine-tuning sorts out the efficient training, finetuning and preference optimization Methods of DMs such as Low Rank Adaption, Consistency Training, Adversarial Training, Adapter Training, etc., and aims to help researchers and developers make appropriate choices for specific low-resource or personalized training tasks.
- • Efficient Sampling and Inference surveys the most commonly used efficient sampling and inference strategies in diffusion models, covering two categories: learning-free and learning-based methods. By comparing their acceleration performance on various generative tasks, we will provide a theoretical basis for the study of faster sampling methods.
- • Efficient Deployment summarizes the latest solutions for deploying the current DMs on mobile devices and on the web, which will facilitate the operation of the DMs in various cross-platform, low-resource environments and promote the birth of various applications.

Reverse-SDE [51], DPMs [1], VDMs [52], DDPM [2], iDDPM [53], DDIM [3], DDRM [54], PNDM [55], INDM [36], D3PM [56], EDM [57], CDM [58]

Foundational Diffusion Theories and Models(§2.1)

NCSN [59], LSGM [60], Score-SDE [61], SSM [62], ScoreFlow [39], ScoreAppr. [37]

Score-based Matching(§2.2)

Principles(§2)

Latent Modeling(§2.3) LDM [33], LSGM [60], LCM [31]

GLIDE [32], CfDG [63], SDG [64], ADM [4], LDM [33], DALL-E2 [6]

Conditional Guidance(§2.4)

VQVAE [65] VQGAN [66], C-ViViT [67], TATS [68], MAGViT [69], CV-VAE [70], MAGViT-V2 [71]

VAE(§3.1)

LDM [33], SDXL [8], U-ViT [72], DiT [73], FiT [74], SiT [75], DiM [76], ZigMa [77], Dimba[78], Latte [79], SD3.0 [80], Pixart-α[81], CogvideoX [82], Sora [50], Moive Gen [83]

Mainstream Network Architecture(§3)

Backbone(§3.2)

CLIP [84], T5 [85], mCLIP [86], mT5 [87], Lllama [88, 89], ChatGLM3 [90]

Text Encoder(§3.3)

ControlNet Training

- /Fine-tuning(§4.1.1)

ControlNet [9], Controlnet-XS [91], ControlnetXt[92], Controlnet++[93]

Adapter Training

- /Fine-tuning(§4.1.2)

T2I-Adapter [42], IP-Adapter [94], X-Adapter [95], Sur-Adapter [96], SimDA [97], CTRL-Adapter [98]

EfficientDMs

Low Rank Adaption Training/Fine-tuning(§4.1.3)

LoRA [99], LoRA-Composer [100], LCM-LoRA [101], Concept-Sliders [102]

Efficient Training and Fine-tuning(§4)

DDPO [103], HPS [104], DreamTuner [105], ImagenReward [106], Diffusion-DPO [107], RAFT [108], AHF [109]

Preference Optimization(§4.2.1)

Textual Inversion [110], DreamBooth [10], BLIP-Diffusion [111], ELITE [112], Mix-of-show [113], MoA [114], OMG [115]

Personalized Training(§4.2.2)

SDE Solver [116–120, 59, 57], ODE Solver [121–127], Trajectory Optimization [128–130]

Training-Free Methods(§5.1)

Efficient Sampling and Inference(§5)

Distribution Based Distillation [131–133, 30, 31, 134], Trajectory Based Distillation [135–137, 28, 138, 139], Adversarial Based Distillation [140, 141], GAN Objective [142–144], Truncated Diffusion [145, 146]

Training-based Methods(§5.2)

Deployment as a Tool(§6.1) ComfyUI, Automatic1111’s SD WebUI

Efficient Deployment and Usage(§6)

SnapFusion [147], MobileDiffusion [148], DistriFusion [149], PipeFusion [150], AsyncDiff [151]

Deployment as a Service(§6.2)

Figure 2: Organization of efficient diffusion models advancements.

• Application investigates the practical applications of efficient DMs in various domains, emphasizing the balance between generative performance, efficiency and computational cost.

To sum up, this survey delves into these research endeavors, exploring various theories, methods and strategies for making DMs more design-, training- and computation-efficient. We review the

development history of efficient DMs, provide a taxonomy of the strategies for efficient DMs, and comprehensively compare the performance of existing efficient DMs. Through this investigation, we aspire to provide a comprehensive understanding of the current state-of-the-art and efficient generative models. Furthermore, this survey serves as a roadmap, highlighting potential avenues for future research and applications, and fostering a deeper comprehension of the challenges and opportunities that lie ahead in the domain of efficient DMs. In addition to the survey, we have established a GitHub repository where we compile the papers featured in the survey, organizing

- them with the same taxonomy at https://github.com/ponyzym/Efficient-DMs-Survey. We will actively update it and incorporate new research in the future.

### 2 Efficient Diffusion Models: Foundational Principles

The diffusion models [1, 2, 53, 61] are modeled as a family of unsupervised latent variable models inspired by considerations from nonequilibrium thermodynamics [1], which are straightforward to define and efficient to train for generating high-quality samples. We will organize the theoretical contexts of the diffusion models and summarize the core principles below.

#### 2.1 Definition and Theory Preliminaries

Discrete Definition Assuming the data distribution is q(x0), the discrete DMs [1, 2] are defined as a forward data perturbation process q(x1:T|x0) and a learnable reverse denoising process pθ(x0:T), both of them are implemented based on Markov steps for progressive add-noising or denoising,

q(x1:T|x0) :=

T

q(xt|xt−1), pθ(x0:T) := p(xT)

t=1

T

pθ(xt−1|xt). (1)

t=1

Note that the two symmetric processes are carried out in different fashions. The former leverages an artificially noise-adding scheduler to gradually convert x0 into xT, while the latter usually starts from p(xT) = N(xT;0,I) and adopts a score matching model sθ (Sec. 2.2) to gradually estimate the posterior distribution pθ(xt−1|xt) until x0 is predicted. Specifically, they can be described as:

q(xt|xt−1) := N(xt;√αtxt−1,βtI), pθ(xt−1|xt) := N(xt−1;µθ(xt,t),σθ(xt,t)). (2)

Where αt = 1−βt for facilitating computation and expression. The training objective of pθ amounts to minimize the negative log-likelihood of the model,

- pθ(xt−1|xt)

- q(xt|xt−1)

L := −Eq log pθ(x0) ≤ −Eq log p(xT) −

log

t≥1

The above variational bound Lvb can be rewritten into a tractable form,

= Lvb. (3)

Eq DKL(q(xT|x0)||p(xT))

+

LT

DKL(q(xt−1|xt,x0)||pθ(xt−1|xt)) Lt−1

−log pθ(x0|x1) L0

t>1

(4)

Continuous Definition Score-SDE [61] is the first to define continuous-time DMs from the perspective of stochastic differential equations (SDE), which can be simplified as: Let pdata(x) denote the data distribution, the diffusion models start by exerting a perturbation kernel pσ(˜x|x) := N(˜x;x,σ2I) onto pdata(x) for forward process. Then they continues to leverage a reverse ODE (also dubbed as the Probability Flow (PF) ODE by [61]) for inverted denoising, which retain the same marginal probability densities as the forward SDE. The forward and reverse diffusion process in continuous-time form can be expressed as:

dxt dt

- 1

- 2

σ(t)2 · ∇x log pt(xt) (5)

= µ(xt,t) −

dxt = µ(xt,t)dt + σ(t)dwt,

where µ(xt,t) and σ(t) are the drift and diffusion coefficient-terms respectively, and {wt}t∈[0,T] denotes the standard Brownian motion. Moreover, ∇x log pt(xt) denotes the gradient of the loglikelihood of pt(xt), which can be estimated by a score matching network sθ(xt,t).

#### 2.2 Score-based Matching Principle

Score matching is a popular method for estimating unnormalized statistical models, such as energybased and flow-based, and it is also well suited for estimating the gradients ∇x log pt(xt) of aforementioned diffusion models. Given samples x1,x2,··· ,xN ⊂ RD from a data distribution pdata(x), our task is to learn an unnormalized density, p˜m(x;θ), where θ is from the parameter space Θ. The model’s partition function is denoted as Zθ, which is assumed to be existent but intractable. Let p˜m(x;θ) be the normalized density determined by our model, we have:

p˜m(x;θ) Zθ

, Zθ = p ˜m(x;θ)dx. (6)

pm(x;θ) =

#### 2.3 Latent Modeling Principle

The latent space projection is proposed by [33] to compress the input images x0 into a perceptual high-dimensional space to obtain z0 by leveraging a pretrained VQ-VAE model [152]. The VQ-VAE is also adopted by almost all current diffusion models, it consists of an encoder E and a decoder G. The mathematical definition is: Given an input image x ∈ RH×W×3, the VQ-VAE first compress the image x into a latent variable zˆ by encoder E, i.e., zˆ = E(x) and zˆ ∈ Rh×w×d, where h and w respectively denote scaled height and width (scaled factor f = H/h = W/w = 8), and d is the dimensionality of the compressed latent variable. After going through the diffusion step described in Eq. 1 or Eq. 5, the latent variable zˆ is updated and finally reconstructed into xˆ by decoder G,

xˆ = Gπ(LDMF

θ(·)(Eπ(x))), (7)

where LDM(·) represents the latent diffusion models (including Unet-based or Transformer-based Sec. 3.2), θ denotes the parameters of LDM, and π denotes the parameters of the VQVAE that are frozen to train our diffusion models.

#### 2.4 Conditional Guidance Principle

Condition-guided Vision Generation. The core of text-conditional diffusion models is to integrate the semantics of text condition c into noise prediction model ϵθ(zt,t) to generate visual contents conforming to text semantics, i.e., ϵθ(zt,t,c). The classifier-free guidance technique has recently been widely adopted in text-guided image generation as,

ϵ˜θ(zt,t,c,∅) = w · ϵθ(zt,t,c) + (1 − w) · ϵθ(zt,t, ∅) (8)

where w = 7.5 is default linear parameter for weighting the unconditional guidance objective and conditional guidance objective in Stable Diffusion, t is time input, c is text condition, ∅ denotes null text embedding initialized by zero vector and θ is model parameters. Note that all of these parameters will be individually or jointly optimized for controlled image editing in following variants.

Condition-guided Vision Editing. Compared with the condition-guided diffusion models, image editing methods usually own more stringent restrictions, which aim to conduct semantic-guided editing while preserving original pixel characteristics. For ControlNet [9], parameter θ is split into θlocked and θcopy for prior preservation and semantic-guided editing, in which ∅ is trained by a zero convolution layer and condition c is split into text prompt ct and image’s feature map cf. This variant can be formalized as ϵ˜θ

locked,θcopy(zt,t,ct,cf,∅zero) (variant 1). Then, in order to achieve more accurate editing, Prompt-to-Prompt [153] introduces a fixed time hyper-parameter τ to determine when to manipulate the cross-attentive parameters θM

into edited θM∗

, which can be formulated as ϵ˜θ(zt,t,τ,c,c∗) = w · ϵθ(θM∗

t

t

t,t≥τ;zt,t,τ,c), where w can be viewed as a reweight hyper-parameter (variant 2). Afterwards, Null-Text-Inversion [154] optimizes the zero embedding ∅ into time-aware embedding ∅t with pivot supervision from DDIM inversion process, which can be simply denoted as ϵ˜θ(zt,t,c,∅t) (variant 3). Later, to further realize the subject-binding and prior preservation, DreamBooth [10] introduces rare token identifiers “[V]” associated with visual subjects and exploits an additional class-specific prior preservation item for training as ϵ˜θ(zt,t,c,c[V]) = w · ϵθ(zt,t,c[V]) + λ · w

t,t<τ;zt,t,τ,c∗) + (1 − w) · ϵθ(θM

′

· ϵθ(zt,t,c) (variant 4). Moreover, to enable non-grid editing [155, 156], Imagic [155] optimizes text embedding c and leverages an interpolation technique to implement variable guidance, which is controlled by a linear hyper-parameter η as ϵ˜θ(zt,t,c∗) (variant 5), where c∗ = η · ctgt + (1 − η) · copt.

###### Pixel Space Lower-Dimensional Latent Space Condition

[Figure 37]

A dog is swimming in the water

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### Diffusion

Encoder

Text prompt refine

Denosing

Text Encoder

Generative Content

[Figure 47]

Decoder

Frozen

Prompts: ["A dog is swimming

Spatial Downsample

in the water","..."]

|[Figure 48]| |
|---|---|
| | |

|[Figure 49]|
|---|

|DownBlock| |
|---|---|
| | |

UpBlock Residual Block

|Tokenizer|
|---|

Residual Block

Reconstructed Image

|DownBlock| |
|---|---|
| | |

UpBlock

A dog is ... ... ... water

(a) 2D VAE

Text Input：

Spatial & Temporal Downsample

Residual Block

###### ...

| |
|---|

|DownBlock| |
|---|---|
| | |

[Figure 50]

[Figure 51]

[Figure 52]

UpBlock

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

|Text Encoder|
|---|

Residual Block

MidBlock

Reconstructed Video

Prompt Embedding

(b) 3D VAE

U/F-Shape Denosing Network

###### Text Encoder

- Figure 3: A universal pipeline of the diffusion based models for visual content generation. A pretrained VAE (with encoder and decoder structures) compresses the input image or video into a latent space. Diffusion models add noise to the latent features and train a neural network (e.g. U-Net or Transformer) for de-noising. User-input text instructions are refined by a large language model and

- then encoded by a trained text encoder into an embedding space, which is injected into the diffusion model to control content generation.

- 3 Mainstream Network Architectures

As shown in Figure 3, following the Latent diffusion model (LDM) [33], most recent textconditional visual generation models consist of three main modules: a variational auto-encoder (VAE) is trained and served as a latent compressor, which encodes images or videos from a highdimensional pixel space into latent space. The model performs diffusion and denoising in the compressed latent space. A neural network is optimized for learning the probability distribution required for each denoising step. A text encoder that encodes the input text into a text embedding as a condition to control and guide for the generation of the image or video content.

- 3.1 VAE for Latent Space Compression

Diffusion and denoising in high-dimensional RGB pixel space [2, 4–6, 157] results in an expensive training cost and affects the speed of inference. To make the diffusion model accessible while reducing its significant resource consumption, LDM [33] observes that most bits of an image contribute

- to perceptual details and retain semantic and conceptual composition even after aggressive compression. LDM removes pixel-level redundancy by training a VAE that compresses the input image from the pixel space to the latent space. Then diffusion and denoising are performed in latent space, which significantly reduces the cost of training and reasoning for DMs. Figure 4 illustrates the structure of standard VAEs for image/video compression, which include normal variational auto-encoders (VAEs) [167], quantized VAEs such as VQVAE [65] or VQGAN [66] and their variants [168], where the GAN discriminator loss is added to achieve reconstruction quality for higher compression More importantly, the trained VAE is a generalized compression model, and its latent space can be used to train multiple generative models and applied to other downstream tasks. Following the LDM, the posterior image generation approach [169, 158, 8, 170, 73, 171, 74, 75, 81, 159, 161, 80] compresses/decompresses the image in latent space by training a VAE using the encoder and decoder of the VAE. The parameters of the VAE are frozen during diffusion model training and inference. Some

###### Methods Year Organization Backbone VAE Text Encoder # Params ADM [4] 2021 OpenAI

- 554M CDM [157] 2021 Google - DALL-E 2 [6] 2022 OpenAI CLIP 6.5B Imagen [5] 2022 Google T5-XXL 3B LDM [33] 2022 LMU Munich

None

Unet

CLIP ViT-L 400M+55M(VAE)

SD1.5 [33] 2022 LMU Munich CLIP ViT-L 860M SD2.0 [33] 2022 LMU Munich OpenCLIP ViT-H 865M SDXL [8] 2023 Stability AI CLIP ViT-L & OpenCLIP ViT-bigG 2.6B Playground-v2.5 [158] 2024 Playground CLIP -

2D VAE

UViT [72] 2022 Tsinghua University

CLIP ViT-L 501M+84M(VAE) DiT [73] 2022 UC Berkeley CLIP ViT-L 675M+84M(VAE)

PixArt-α [81] 2023 Huawei Noah’s Ark Lab T5-XXL 600M FiT [74] 2024 Shanghai AI Lab CLIP ViT-L SiT [75] 2024 New York University CLIP ViT-L 675M Latte [79] 2024 Shanghai AI Lab T5-XXL 673.68M Hunyuan-DiT [159] 2024 Tencent Hunyuan mCLIP & mT5-XL 1.5B LuminaT2X [160] 2024 Shanghai AI Lab LLama2-7B 7B

2D VAE

Transformer

Kolors [161] 2024 Kuaishou ChatGLM3-6B-Base 2.6B SD3.0 [80] 2024 Stability AI CLIP ViT-L & OpenCLIP ViT-bigG & T5-XXL 8B Flux.1 [162] 2024 BlackForestLabs CLIP ViT-L & OpenCLIP ViT-bigG & T5-XXL 12B

Sora [163] 2024 OpenAI

- -

Open-Sora [164] 2024 Hpcaitech T5-XXL 1.2B Open-Sora-Plan [165] 2024 Peking University T5 & mT5 -

3D VAE

EasyAnimate [166] 2024 Alibaba Group mCLIP & mT5-XL 1.5B CogvideoX [82] 2024 Zhipu AI T5-XXL 2B/5B Moive Gen [83] 2024 Meta TAE MetaCLIP & UL2 & ByT5 30B

Table 1: Comparison of modules and parameters in different diffusion generative Models.

Input Video

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Encoder

ReconstructionLoss

Regularizer

Decoder

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

CausalConv3d SpatialDown/UpBlock SpatialTemporalDown/UpBlock

MidBlock

Conv3d

Causal Convolution in time

Reconstructed Video

- Figure 4: A standard encoder-decoder architecture of 3D Variational Autoencoders (VAEs) are utilized for video compression.

diffusion models [172, 169, 173] are used to generate videos by directly learning pixel distributions. Video contains not only spatial information but also a lot of temporal information, so there are more computational challenges in video generation. In addition, the diffusion video generation models exemplified by Sora [163] use VAE to compress the video and then train and reason in latent space. These video generation models [79, 164–166] are usually derived from Stable Diffusion’s image 2D VAEs, since training 3D from scratch is quite challenging. Time compression is simply achieved

WebVid Panda-70M

Models Compress Ratio

PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ SD2.1 VAE [33] 1×8×8 30.19 0.8379 0.0568 30.40 0.8894 0.0396

SVD VAE [14] 1×8×8 31.15 0.8686 0.0547 31.00 0.9058 0.0379

CV-VAE [70] 4×8×8 30.76 0.8566 0.0803 29.57 0.8795 0.0673 Open-Sora VAE [164] 4×8×8 31.12 0.8569 0.1003 31.06 0.8969 0.0666

Open-Sora-Plan VAE [165] 4×8×8 31.16 0.8694 0.0586 30.49 0.8970 0.0454

Table 2: Comparison of VAE performance in common image and video generation diffusion models.

by uniform frame sampling while ignoring motion information between frames. Table 2 compares the performance parameters of VAEs commonly used in the community Some methods use hybrid 2D-3D VAEs [69, 166, 164, 165, 82, 70, 67] or full 3D VAEs [71]. e.g., MAGViT [69] uses 3D VQGAN with 3D and 2D downsampling layers, and MAGViT-V2 [71] uses a full 3D convolutional encoder with overlapping downsampling. In order to trade off lower memory and computational cost with slightly lower reconstruction quality, the latest video generation model, Moive Gen [83], uses interleaved 2D-1D convolutional encoders in its VAE.

#### 3.2 Denoising Neural Network Backbone

As shown in Figure. 5, the neural networks within the diffusion models mainly serve as residualstyle noise predictors in the de-noising stage [47], which can be categorized into the following mainstream architectures:

latent input z predicted noise

###### CrossAttention Down/Up Block Transformer Down/Up Block

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

y

[Figure 73]

condition

y latent input z t

y latent input z input

t

t

###### time embeds

| | |
|---|---|
| | |

| | |
|---|---|
| | |

timestep

| | |
|---|---|
|Linear & Reshape| |

|Embedding Layer| |
|---|---|
| | |

|GroupNorm|
|---|

|Linear|
|---|

ResnetBlock2D

Norm

SiLU Conv3 3

SiLU expand_dim

DownBlock

UpBlock

Transformer2DModel

Multi-Head Attention

Conv1 1

DownBlock UpBlock

ResnetBlock2D

|GroupNorm|
|---|

Norm

DownBlock UpBlock

SiLU Conv3 3

Transformer2DModel

MLP

ResnetBlock2D

MidBlock

Down / Up-Sample2D

| | |
|---|---|
| | |

U-Shape Denosing Network

[Figure 74]

Diffusion Transformer Block Spatial & Temporal 2D/ 3D Full attention

###### SSM Block

###### predicted noise

| | |
|---|---|
| | |

Temporal

Scale

Linear Projection

| | |
|---|---|
|Linear & Reshape| |

3D Full Attention

Pointwise Feedforward

Spatial

Layer Norm

Scale, Shift

Temporal

Layer Norm

Forward SSM

Backward SSM

3D Full Attention

Activation

Denoising Block

| | |
|---|---|
|Embedding Layer| |

Spatial

Forward Conv1d

Backward Conv1d

Scale

| | |
|---|---|
|Patchify| |

Temporal

Multi-Head Self-Attention

Linear Projection

Linear Projection

3D Full Attention

| | |
|---|---|
|Spatial| |

[Figure 75]

[Figure 76]

[Figure 77]

timestep t

[Figure 78]

[Figure 79]

Scale, Shift

Norm

Layer Norm

MLP

| | |
|---|---|
| | |

condition latent input z

y

input token z

t y

input token z y

t

input token z t y

F-Shape Denosing Network

- Figure 5: The mainstream neural network backbones serving as denoisers in diffusion models, which including U-shaped denoising networks (U-Net based and U-ViT based) and F-shaped denoising networks (DiT-based and SSM-based).

U-Net based Backbone. DDPM [2] as the seminal work that introduces U-Net [174] as the backbone for the diffusion model to predict the probability distribution at each step of the de-noising process. In which the U-Net follows from PixelCNN++ [175] and utilizes an encoder-decoder architecture, where the spatial pixels of the image are downsampled by convolutional operations at each layer of the encoding process while extracting the features. The spatial resolution is progressively restored in the decoder stage, while the feature information extracted by the encoder and decoder is fused via skip connections. U-Net [174] is capable to process the image features at different scales, which helps in the gradual de-noising process. Specifically, Song et al. [116] improved the performance of unconditional image generation tasks by making further changes for U-net in the Sore-based diffusion model. Prafulla et al. [4] improved the U-Net architecture in the

diffusion model by increasing the width and depth of the network, and increasing the number of attention heads, etc., achieved better performance than GAN on the image generation tasks. Other models [157, 32, 6] with U-Net based architectures perform diffusion and denoising directly in highdimensional RGB pixel space, incurring high training costs and affecting the inference speed. Based on the LDM [33], SDXL [8] uses more attention blocks and a larger cross-attention context, thus including more parameters in the U-Net. VDM [176] extends LDM to the video generation task by introducing 3D convolutional layers.

Transformer based Backbone. Transformer has shown the dominance in the fields of Natural Language Processing [177–180], Computer Vision [181] and Multi-modality [182–184] with its scalability and ability to model long-range dependencies of the attention mechanism. This trend is also held in many autoregressive image generation models [185, 186]. However, before U-ViT [72] and DiT [73] were proposed, advanced diffusion models for image generation tasks still adopt a convolutional U-Net architecture. U-ViT [72] introduced the Transformer Block in a U-shaped structure as a backbone for diffusion models, which treats all inputs as tokens and utilizes a long skip connection between the shallow and deep layers. DiT [73] introduced Vision Transformer [181] as a backbone to replace U-Net, and further demonstrates the scalability of Transformer for image generation tasks. Recent works have also demonstrated the superior performance of the diffusion generation model of the DiT architecture for image [81] and video generation tasks [79, 163]. Specifically, PixArtα [81] simplicates computationally intensive class conditional branching in Diffusion Transformer by joining the cross-attention module to inject textual conditions that are encoded through T5 [85]. Latte [79] expands the DiT architecture to the video generation task by extracting spatio-temporal tokens from the input video, and introducing temporal and spatial transformer blocks to model the video distribution in latent spaces, respectively. Futhermore, following DiT, Latte uses AdaLAN for time-step class information injection. Notably, the emergence of Sora [163] demonstrates the substantial scalability of the Transformer architecture for generating high-quality video content. There are a number of recent image [159, 80, 162] and video [164, 165, 82, 166, 83] generation models that have also verified the scalability of transformer in diffusion modeling under large-scale training.

SSM based Backbone. The transformer-based diffusion models suffer from the quadratic complexity of the attention mechanism, making them consume huge computational cost for long sequence generation tasks (e.g., high-resolution image synthesis, video generation, etc.). Advances in statespace modeling (SSM) [187, 188] show a new direction to achieve a trade-off between computational efficiency and model flexibility. Some recent SSM-based approaches [189–191] have been proposed and proven their efficiency on multiple tasks and modalities in modeling long sequence dependencies. Mamba [192] combines SSM architectures and proposes hardware-aware algorithms that enable efficient training and inference. DiM [76] introduces Mamba as a diffusion backbone for high-resolution image generation. Specifically, DiM avoids unidirectional causality between patches by designing the Mamba block to perform the four scanning directions alternately. In consideration of the lack of spatial continuity in mamba scanning schemes, ZigMa [77] allows mamba blocks applicable to 2D images by incorporating continuity-based inductive bias in the images. In addition by performing spatio-temporal decomposition of 3D sequences, which is extended to video generation task.

In addition to the above types of mainstream diffusion model architectures, there are some other diffusion model architectures for image and video generation. Diffusion-RWKV [193] introduces the RWKV [194] architecture as the Backbone for Diffusion models. The RWKV consists of an input layer, a series of stacked residual blocks and an output layer. Each residual block consists of temporal mixing and channel mixing sub-blocks. RWKV improves on the standard RNN architecture by parallelizing computation during training similarly to RNN. It includes enhancements to the linear attention mechanism and designs the receptance weight key value (RWKV) mechanism. DiG [195] introduces a Gated Linear Attention Transformer (GLA) [196] and proposes Diffusion GLA model. DiG achieves high efficiency in terms of training speed and GPU memory for high resolution image generation.

#### 3.3 Text encoder

The text encoder is used to capture the complex semantics within the input text prompts, which is a critical component of the text-conditional visual generation model and directly affects the generated content. Early text-to-image approaches used text encoders which are trained on paired text-images,

SD1.5(CLIP) SDXL(CLIP&OpenCLIP)SD3.0(CLIP&OpenCLIP&T5）OmniDiffusion(Baichuan) Kolors(ChatGLM) Flux.1(CLIP-L&T5)

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

A large brown cow standing on a green grass

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Two people sitting on grass in front of a lake looking at the sky

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

A brown and white dog has is playing with a white ball on the grass

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Two children dressed as pirates

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

一个戴着红色帽子的白发苍苍的老人

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

一个戴着白手套的女人，背后还有一些其他人

- Figure 6: Comparison of generated images from diffusion models with different text encoders. The last two rows are Chinese prompts, which are used to test image generation models with the text encoders that support the multilingual condition (i.e., OmniDiffusion [197], Kolors [161]). For models that do not support the multilingual text condition, the given prompts are translated into the corresponding language to generate images.

and they can be trained from scratch [32, 186] or pre-trained (e.g., CLIP [198]). CLIP uses contrast learning, and is trained to align the embedding representations of the text and images. These text encoders can encode visual and textual semantics after being trained using paired text-images. After tokenizer and embedding, the input text prompt is injected as a condition into the diffusion model generative backbone. As shown in the table 1, some classical text to image diffusion models [6, 33, 72–75, 159, 80, 162] use the text branch of CLIP models for text representation. Typically, the parameters of these text encoders are frozen thus their computational and memory consumption during diffusion model training can be ignored. CLIP series models focus on the global representation of an image by aligning the embedding space of the image and the text, however, it is difficult

to understand the detailed description. Large language models are trained on larger text corpora and have stronger text comprehension and generation capabilities. Imagen [169] compared CLIP with a pre-trained large language model (BERT [199], T5 [85]) as text encoders. In addition, they found that scaling the size of the text encoder can improve the quality of text-to-image generation and that using the T5-XXL encoder achieves better image-text alignment and image fidelity. Some approaches merge both CLIP and T5 encoders to improve the ability of text comprehension. Some image diffusion models [200–202, 159] focus on understanding the multilingual prompt and generating images. HunyuanDiT [159] combines a bilingual CLIP [86] and a multilingual T5 [87] text encoder to improve Chinese comprehension. Some recent image [197, 161] and video [82] generation models use large language models (e.g., Baichuan [203], Llama [88, 89], and ChatGLM [90]) to enhance semantic understanding of complex text. Figure 6 provides a visual comparison that demonstrates how the understanding of complex texts by large language models affects the generation effects of diffusion models.

### 4 Efficient Training and Fine-tuning

The efficient training strategies of diffusion models aim to reduce training time and resource consumption while maintaining performance improvements, making diffusion models more flexible in a wide range of downstream tasks. Here, we mainly lay emphasis on two aspects of efficient training: parameter efficiency and label efficiency. Parameter-efficient methods focus on optimizing the architecture of trainable modules to reduce the number of parameters required for high performance. Meanwhile, label-efficient methods aim to minimize the amount of training data needed, which is especially critical when high-quality labeled datasets are limited or unavailable. In this section, we provide a brief overview of various techniques and approaches that enhance parameter efficiency and label efficiency, and discuss their significance in downstream tasks of diffusion models.

#### 4.1 Parameter-Efficient Methods

Parameter-efficient training methods aim to adapt pre-trained models to new tasks by updating only a small number of parameters, rather than the entire model, thereby prevent overfitting while improving performance. Following the definition in [204], given the pretrained parameters of a diffusion model θ = {w1,w2,...,wn}, the fine-tuning task aims to obtain the parameters θ′ = {w1,w2,...,wm} on a given dataset D. The parameter update is defined as ∆θ = θ − θ′. Compared to full fine-tuning, where |∆θ| = |θ|, efficient training is achieved when |∆θ| ≪ |θ|, where | · | denotes the number of parameters.

[Figure 116]

- Figure 7: A generic training framework for parameter-efficient training approaches in diffusion models. The model leverages frozen base parameters while introducing trainable components through ControlNet, LoRA, and adapter modules with visual, image, and feature inputs progressively encoded to produce the final output.

As shown in Figure 7, parameter-efficient training techniques can be categorized into three types: ControlNet [9], low-rank adaption (LoRA) [99], and adapter [97, 205]. These approaches add and update lightweight modules, enabling efficient adaptation to new tasks. In the following subsections, we will analyze the application advantages of these techniques across various downstream tasks.

#### 4.1.1 ControlNets

Despite the impressive text-to-image capabilities [32, 206, 7, 6, 33, 5], diffusion models often struggle with spatial compositional control [207], particularly in tasks such as depth-to-image and poseto-image. To address these limitations, ControlNet [9] introduces visual features into the multiresolution layers of a pre-trained UNet, thereby enabling more controllable generation. This advancement has spurred further research, resulting in several efficient variants of ControlNet [91– 93]. As illustrated in Figure 8, these improvements focus on two main aspects: reducing the number of parameters in ControlNet while maintaining or improving its performance, and enhancing finergrained control without increasing the number of parameters.

[Figure 117]

Figure 8: An illustration of ControlNet and its extensions, demonstrating its ability to guide image generation using various control signals such as edges, depth, segmentation, and poses.

A branch of works prefer to reduce the parameter count of ControlNet. ControlNet-XS [91] found that with high-frequency and large-bandwidth communication between the control blocks and generative network, the control module requires fewer parameters to achieve better results, speeding up both inference and training. ControlNeXt [92] introduces a lightweight convolutional module to extract control features, replaces zero-convolution with cross normalization to align the parameter distributions with those of main denoising branch and achieve faster and more stable training convergence.

Another line of works enhance ControlNet’s controllability over the generated output while maintaining the same parameter count, ControlNet++ [93] employs a pre-trained discriminative reward model to effectively bridge the gap between conditions and generated images, improving the quality and pixel-level relevance of the output when control signals are reflected in the generated images.

#### 4.1.2 Adapters

Compared to ControlNet [9], which achieves additional spatial control by fine-tuning duplicated encoders, adapter-based methods [94, 42, 97, 98, 205] boast more flexible and lightweight architectures as shown in Figure 9 and reduce the need for extensive data and computational resources as detailed in Table 3. As crucial parts that allow models to perform a variety of downstream tasks, adapters establish the intrinsic connection between the conditional inputs and their corresponding images, making them commonly used for controllable generation [96, 208, 209] and domain adaptation [210].

[Figure 118]

- Figure 9: An overview of the Adapter framework for diffusion models, illustrating various adapters, which facilitate feature mapping and injection processes, allowing the diffusion model to handle diverse input types.

Model Dataset Condition Params Hardware Time Controllable Generation

COCO17 164K images sketch map COCO-Stuff 164K images segmentation map 77M 4 32G V100 3d

T2I-Adapter [42]

LAION-Aesthetics 600K T-I pairs keypoints/color/depth StableSketching [209] Sketchy database 12.5K images abstract sketch - - Uni-ControlNet [208] LAION 10M T-I pairs

global condition 47M

- local condition 412M

Lexica/civitai/ Stable Diffusion Online

SUR-Adapter [96]

57K T-I-T sets simple prompt 20M RTX 3090 5K steps

IP-Adapter [94] LAION & COYO 10M T-I pairs image 1.5M 8 V100 1M steps I2V-Adapter [205] WebVid 10M videos first frame image - - -

MEAD/HDTF/ VoxCeleb2

###### - audio - 8 V100 2.5d Domain Adaptation

FaceChain-ImaginelD [211]

X-Adapter [95] LAION 300K images spatial feature 213M 4 A100 2 epochs Ctrl-Adapter [98]

Panda 200K videos

spatial feature 184M 80G A100 10h LAION POP 300K images

Table 3: Comparison of various adapters and their applications.

Controllable generation Adapters effectively map diverse conditions into meaningful regions within the conditional space of diffusion models. For image generation, T2I-Adapter [42] captures conditional features and maps control feature to internal knowledge of the T2I model, achieving visual control of image generation. StableSketching [209] transforms semantic information from abstract sketch into textual conditional embedding and further constrains control features to pixel-perfect and textually meaningful regions in embedding space. SUR-Adapter [96] effectively navigates simple prompt features toward a more information-dense region within the conditional space, enabling the generation of highly detailed images from simple prompts. IP-Adapter [94] maps image features into a decoupled conditional space, enabling the model to generate images that resemble the input image. In the field of video synthesis, I2V-Adapter [205] aligns each frame of

the video with the semantic information of the image condition, enhancing the overall coherence across frames. FaceChain-ImagineID [211] introduces a textual inversion adapter to convert speech text embeddings into token embeddings. Simultaneously, a spatial conditional adapter maps facial mesh, identity features, and masked adjacent frame features into the conditional space, maintaining audio-visual consistency and spatial coherence throughout the video. In summary, adapters play a crucial role in injecting a wide array of conditions into diffusion models, significantly enhancing the control and quality of generated content in images and videos.

Domain adaptation Adapters in domain adaptation serve to align feature representations, enable task-specific adjustments, and facilitate efficient and effective transfer of knowledge from a source domain to a target domain. X-Adapter [95] establishes a mapping relationship between the spatial features of the base diffusion model and those of the upgraded diffusion model. Ctrl-Adapter [98] integrates the features from a pre-trained image ControlNet into the framework of a target video diffusion model, facilitating multi-conditional control in video generation. Overall, by establishing mappings between different feature spaces, adapters enhance the flexibility of diffusion models across diverse applications.

#### 4.1.3 Low Rank Adaption

Based on the recent observations [212, 213] that over-parameterized models operate in a lowdimensional subspace, LoRA [99] learns parameter offsets using low-rank matrices and assumes that the weight update ∆W during fine-tuning can be represented as a low-rank decomposition of two smaller matrices A ∈ Rd×r and B ∈ Rr×k, such that ∆W = A × B. The fine-tuned weight matrix becomes W = W0 + A × B , where W0 is the original pretrained weight. By restricting A and B to have low-rank r, where r ≪ min(d,k), LoRA reduces the number of trainable parameters and computational overhead during fine-tuning. Instead of freezing diffusion models and inserting new trainable modules to prevent catastrophic forgetting [9, 42], LoRA allows the learned weight update to be merged back into the original model after training, avoiding the need for additional inference time. Therefore, it has been widely applied to various downstream tasks, as shown in the table 4.

[Figure 119]

- Figure 10: An illustration of the mechanism combining LoRA and diffusion models, where LoRA fine-tunes the diffusion model to adapt to various customized tasks such as style, subject and other objective, and exceeds in modular adaptation and conceptual interpolation.

Method Year Base Model Downstream Task Code Image Generation

Image Generation, Depth Map Guided Control-LoRA 2023 ControlNet Image Generation, Canny Edge Guided

[code]

Recolor

Dreamshaper 7 [code] LCM-LoRA [101] 2023 SSD 1B [code]

Fast Image Generation (Text-to-Image, Inpainting, styled-Generation)

[code] Concept Sliders [102] 2023 SD Customized Attribute Editing [code]

SDXL v1.0

LoRA-Composer [100] 2024 SD Multi-Concept Customization [code]

ZipLoRA [214] 2023 SDXL v1.0 Subject & Style Composed Customization [code] Mix-of-Show [113] NeurIPS 2023 SD v1.5 Multi-Concept Customization [code] C-LoRA [215] 2023 SD Continual Concept Customization -

Intrinsic LoRA [216] 2023 SD(v1.1, v1.2, v1.5) Image Normals, Depth, Albedo, Shading Generation -

Smooth Diffusion [217] CVPR 2024 SD Latent Space Interpolation, Image Inversion, Image Editing [code] DiffMorpher [218] CVPR 2024 SD v2.1 Image Morphing [code]

###### Video Generation

AnimateDiff [34] ICLR 2024 SD Personalized Style & Motion Guided, Animation Generation [code] DragVideo [219] 2023 AnimateDiff Sample-Specific, Video Generation [code] MagicStick [220] 2023 SD Scenes-Specific, Video Generation -

3D Synthesis ProlificDreamer [221] NeurIPS 2023 SD Rendered 2D Image Generation, Text & Camera Pose Guided [code] Boosting3D [222] 2023 SD Rendered Image Generation, Text & Camera Pose Guided -

3DFuse [223] 2023 SD Rendered Image Generation, Text & Sparse Depth Map Guided [code] DreamControl [224] CVPR 2024 SD v1.5 Rendered Image Generation, Text & Normal Map Guided [code]

Table 4: The statistics for LoRA methods utilized in recent research

Module adaptation Benefiting from the low-rank property, multiple LoRA parameters, which are fine-tuned on different datasets or downstream tasks, can be directly combined to produce composition capability as shown in Figure 10. LCM-LoRA [101] can generate images in a specific style while supporting fast inference with minimal steps, by linearly combining the style-related LoRA parameter and acceleration LoRA parameter. AnimateDiff [34] trains individual LoRAs to specialize in distinct motion patterns. During inference, these specialized LoRAs can be synergistically combined, enabling the generation of diverse and complex motion effects. LoRA-Composer [100] integrates multiple concept-specific LoRAs into the image generation process, ensuring each concept is accurately rendered in terms of position, size, and distinctive features. These methods enable multiple LoRAs to seamlessly generate different concepts in various regions of an image, or to combine distinct characteristics during the image generation process, fully exploiting the composable nature of LoRAs.

Concept interpolation LoRA approximates the update direction within a compact and structured parameter space through low-rank decomposition. As shown in Figure 10, when performing linear interpolation between LoRA parameters for different concepts, the resulting intermediate parameters smoothly blend the features of the original parameter sets. Concept Sliders [102] subsequently modifies the concept along specific parameter direction by scaling the guidance coefficient in training loss and the hyperparameters of LoRA. DiffMorpher [218] discovers LoRA has the capability to encapsulate image semantic identity, and achieves image morphing by performing linear interpolation on the LoRA parameters adapted to different concepts. Collectively, these advancements demonstrate that LoRA offers significant flexibility and control in the field of image generation and editing, allowing creators to achieve smoother transitions between different concepts.

#### 4.2 Label-Efficient Methods

The scarcity of data can negatively impact the generation quality of diffusion models, leading to the development of two key strategies for efficient adaptation to downstream tasks with minimal labeling. One strategy is preference optimization, which trains annotation models (like reward models) to replace human annotations and uses reinforcement learning to continuously supervise the training of diffusion models to meet human preferences. The other is personalized training, which optimizes the learning process to extract the most salient features from small datasets while preserving the generative capabilities of diffusion models.

Supervised learning

Reference diffusion Model

Preferred

|Prompt：Afoxin traditionalcostume<br><br>issleeping,trees<br><br>𝑧𝑟𝑒𝑓𝑇<br><br>[Figure 120]<br><br>𝑧𝑡𝑢𝑛𝑒𝑇<br><br>[Figure 121]<br><br>Larger scale|
|---|

|[Figure 122]<br><br>Prompt<br><br>𝐼𝑟𝑒𝑓 𝐼𝑡𝑢𝑛𝑒<br><br>[Figure 123]<br><br>Smaller scale|
|---|

𝑧𝑟𝑒𝑓0

|U-Net|
|---|

𝑧𝑟𝑒𝑓𝑇−1

| |
|---|

𝐼𝑟𝑒𝑓0 …𝐼𝑟𝑒𝑓𝑇−1

| |
|---|

| |
|---|

[Figure 124]

[Figure 125]

| |
|---|

| |
|---|

[Figure 126]

[Figure 127]

| |
|---|

U-Net

[Figure 128]

[Figure 129]

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

Prompt：Afoxin traditionalcostume

issleeping,trees

Condition Encoder

Decoder

Denoising for 𝑻 steps

VAE

Reward model

|U-Net|
|---|

| |
|---|

[Figure 130]

[Figure 131]

[Figure 132]

| |
|---|

| |
|---|

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

| |
|---|

| |
|---|

U-Net

| |
|---|

>

𝐼𝑡𝑢𝑛𝑒0 …𝐼𝑡𝑢𝑛𝑒𝑇−1 𝐼𝑟𝑒𝑓

𝐼𝑡𝑢𝑛𝑒

𝑧𝑡𝑢𝑛𝑒0

𝑧𝑡𝑢𝑛𝑒𝑇−1

|For higher reward score|
|---|

Preference optimization

Fine-tuned diffusion Model

Supervised learning Preference optimization

- Figure 11: A generic training framework for preference optimization. The reward model scores the generated images directly without human annotation, saving annotation cost and time.

#### 4.2.1 Preference Optimization

Diffusion models mainly utilize the variational lower bound on the log-likelihood expressed in Equation 2 to approximate the target data distribution. While a decrease in training loss indicates that the model is learning certain patterns, it does not necessarily mean that the generated images meet human aesthetic standards. Therefore, the preference training framework shown in Figure 11 has become a critical approach for aligning models with human expectations. The current process for preference optimization in image generation tasks is generally divided into two steps. First, human aesthetic preferences are formalized into a reward model [104, 109, 106, 225], which reduces the cost and time required for labeling data in the subsequent stages. Secondly, the direct finetuning on preferred outputs [108, 104, 109, 106] or reinforcement learning from human feedback (RLHF) [103, 226, 227, 107] are employed to optimize the diffusion model against the reward model. Figure.12 illustrates the paradigms of each category. These methods avoid the complex computational burden associated with direct supervised training on large datasets labeled with preference tags, making preference optimization an efficient method.

Reward model. It is crucial that the reward model effectively encodes human preferences, as this directly impacts the diffusion model’s ability to correctly learn and reflect individual aesthetics. The general idea of human preference modeling is to maximize the difference that the reward score of a preferred image Iw with prompt condition T is greater than the other outputs Il for any sample from the preference dataset, formulated as follows:

w,Il)∼D[log(σ(R(T,Iw) − R(T,Il)))]. (9)

Lreward = −E(T,I

where σ indicates the activation function and R represents the reward model. HPS [104] fine-tunes the CLIP model using training data that includes text prompts and multiple images (one preferred and the others non-preferred), enabling the model to obtain a human preference score. AHF [109] creates text-image groups with binary feedback datasets to train the CLIP model, applying the mean squared error (MSE) loss for accuracy and the cross-entropy loss to improve generalization to unseen data. ImageReward [106] employs a scoring system where higher rankings yield higher scores to train the BLIP model, utilizing a text-images dataset with ratings and rankings, allowing for a finer distinction in image quality. Pick-a-Pic [225] is a large dataset where each instance includes a prompt, two generated images, and a label indicating preference or tie. It is employed to fine-tune

|Human feedback<br><br>Preference Dataset<br><br>Training Reward Model<br><br>Reward<br><br>model<br><br>Rank outputs<br><br>Sample preference pairs<br><br>|Prompt<br><br>><br><br>[Figure 138]<br><br>[Figure 139]|
|---|
<br><br>Diffusion model<br><br>[Figure 140]<br><br>Prompt: A girl wearing sunglasses sitting on a large cut watermelon.<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>><br><br>><br><br>><br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>reward: 0.8<br><br>reward: 0.3<br><br>|
|---|

|[Figure 151]<br><br>Direct fine-tuning<br><br>Fine-tuned diffusion model<br><br>Reward model<br><br>𝑟<br><br>Maximumreward-weightedlikelihood<br><br>Prompt 𝑧𝑇<br><br>[Figure 152]<br><br>[Figure 153]<br><br>𝐼0<br><br>preferred<br><br>|
|---|

|Direct Preference Optimization<br><br>Reference diffusion model<br><br>Fine-tuned diffusion model<br><br>Human<br><br>Reward or feedback<br><br>model<br><br>ContrastivelearningWhichpreferred<br><br>Prompt 𝑧𝑇<br><br>[Figure 154]<br><br>|𝐼𝑟𝑒𝑓0 …𝐼𝑟𝑒𝑓𝑇−1 𝐼𝑡𝑢𝑛𝑒0 …𝐼𝑡𝑢𝑛𝑒𝑇−1<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]|
|---|
<br><br>|> 𝐼𝑟𝑒𝑓<br><br>𝐼𝑡𝑢𝑛𝑒0 0<br><br>[Figure 163]<br><br>[Figure 164]|
|---|
<br><br>|
|---|

|RLHF<br><br>Fine-tuned diffusion model<br><br>Policygradient<br><br>Reward model<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>𝐼𝑡𝑢𝑛𝑒0 …𝐼𝑡𝑢𝑛𝑒𝑇−1<br><br>𝑟0 …𝑟𝑇−1<br><br>Prompt 𝑧𝑇<br><br>[Figure 169]<br><br>|
|---|

ContrastivelearningWhichpreferred

Policygradient

Maximumreward-weightedlikelihood

>

Preference optimization methods

- Figure 12: The illustrations of preference optimization paradigms. The trained reward model can be used in subsequent various preference optimization methods.

the Pick Score, a reward model based CLIP-H, with the objective of minimizing the KL divergence between the preference label and the softmax normalized scores of the two images.

Direct fine-tuning has achieved remarkable preformance by leveraging a reward model for supervised learning. RAFT [108], in each iteration, uses a reward model to filter the K samples generated by the diffusion model, selects the best-of-K sample for fine-tuning the model, thereby avoiding the overfitting problem when fine-tuning with datasets devoid of preference labels. AHF [109] introduces the negative reward-weighted log-likelihood into the loss function of preference optimization to improve the image-text alignment of the model. ImageReward [106], during the refinement phase of diffusion models utilizes ReFL loss and regularization with pre-training loss to prevent rapid overfitting and stabilize fine-tuning. HPS [104] suggests incorporating a special identifier in the prompts during fine-tuning to distinguish preferred images. During inference, these special identifier serve as negative prompts for classifier-free guidance, effectively preventing the generation of non-preferred images.

Reinforcement learning from human feedback (RLHF) uses policy gradient to optimize humanpreferred policy aimed at maximizing the reward model’s scoring of generated images. DDPO [103] reframes the denoising process of diffusion model as a multi-step Markov decision process and employs importance sampling techniques to optimize it. This algorithm serves as a versatile framework for optimizing any downstream objective, covering aspects such as compressibility, aesthetic quality, and text alignment. DPOK [226] introduces two critical improvements over DPPO. Firstly, it incorporates KL regularization into the loss function, effectively curbing the model’s tendency to overfit to rewards. Secondly, by additionally training a value function, it not only significantly reduces the variance in gradient estimation but also further enhances the performance of the final reward. D3PO [227] overcomes the application obstacles of DPO [228] in diffusion models without the need for pre-trained reward models. It trains through online learning, leveraging real-time preference annotations from experts on two images generated from the same text. Diffusion-DPO [107] directly optimizes the policy that aligns more closely with human preferences during the single-step denoising process, effectively solving the problem of prolonged training time due to the need for multi-step reverse denoising in previous methods.

#### 4.2.2 Personalized Training

The primary challenge of personalized generation based on diffusion model lies in data scarcity, as high-quality training data is often difficult to obtain. To tackle this, personalized training methods [110, 10, 94, 112, 229, 230, 111, 231, 115] tailor the learning process to achieve high perfor-

Pre-TrainedDiffusionModels

SDE Solvers Symbol Explanation

[Figure 170]

🔓

|[Figure 171]<br><br>[Figure 172]<br><br>w/. Diffusion Term Stochastic Process| | |
|---|---|---|
||PredictorCorrector-|
|---|
|Langevin Dynamics||Adaptive Step Schedule|
|---|
|
|PF-ODE Solvers<br><br>| |Optimal Trajectory| |Semi-linear Structure| |Discretization Scheme| |
|---|---|---|---|---|---|---|
<br><br>[Figure 173]<br><br>w/o Diffusion Term Stochastic Process| | |
|Trajectory Optimization| | |

[Figure 174]

🔥 The parameters is updated in training

𝑋 𝑋

[Figure 175]

[Figure 176]

𝑡 ~ 𝑈𝑛𝑖𝑓𝑜𝑟𝑚 ({1, …,𝑇})

Add random noise in forward process

[Figure 177]

[Figure 178]

[Figure 179]

Sampling random noise

[Figure 180]

[Figure 181]

[Figure 182]

###### −

🔓 The parameters is frozen in sampling

[Figure 183]

Down Block

Up Block

𝑋

𝜖 𝑋

[Figure 184]

[Figure 185]

The denoising network for diffusion

gradient decent

t

|…<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]|
|---|

|…<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]|
|---|

|…<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]|
|---|

|…<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]|
|---|

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

…

…

…

…

…

[Figure 207]

Gaussian Distribution ~ 𝒩(0, 1)

Random Noises

|Dynamic Programming|
|---|

|Latent Retrieval|
|---|

GenerateRandomNoises

Sampling Process Generated Images

Figure 13: The illustration of training-free methods.

mance with less data by focusing on relevant and personalized information rather than generalizing across a broad dataset, significantly reducing the need for large amounts of individual data. In this section, we present two mainstream personalized synthesis methods and discuss their contributions to label-efficient approaches.

Fine-tuning-based personalization approaches have focused on fine-tuning pre-trained diffusion models to learn a placeholder token that captures the identity information of reference subjects. For instance, DreamBooth [10] conducts full fine-tuning, Textual Inversion [110] adjusts the embeddings of pseudo-words, and Custom-Diffusion [232] optimizes the key, value mapping matrices within cross-attention layers. Moreover, LoRA [99] introduces a minimal number of trainable parameters and trains individually on a few customized datasets, facilitating the widespread utilization of LoRA for customization. Recent efforts [115, 113, 100, 114] focus on achieving multi-concept customization by combining LoRA weights from different concepts, aiming to improve identity preservation, handle occlusions, and enhance foreground-background harmony.

However, fine-tuning-based approaches often require training for thousands of steps to customize concepts and most of them rely solely on a placeholder token embedding which proves insufficient for effectively decoupling specific concepts from their background layouts. To address this, encoderbased methods [112, 233, 105, 111, 231, 229, 234] utilize additional image encoders to inject the reference image details for subject generation. ELITE [112] and DreamTuner [105] adopt a strategy that progressively extracts visual information of target features, from coarse to fine, enabling more precise and controllable subject-driven image generation. Meanwhile, BLIP Diffusion [111] uses a multimodal encoder (i.e.. Q-former [235]) to filter out background information, focusing on learning the intricate details of the intended concepts.

### 5 Efficient Sampling and Inference

Representative diffusion models often require numerous iterations for de-noising [2], which hinders their practical application [236]. Consequently, researchers devote to the efficient sampling methods [132, 133, 30, 31, 101, 140, 136, 137, 144, 123, 57] that can reduce the number of iterations during the inference stage while maintaining the model’s ability to generate high-quality images. We summarize four types of methods and illustrate them for efficient sampling and inference below.

#### 5.1 Training Free Method

- As illustrated in 2.1, the DMs can be defined as a continuous-time process from the perspectives of SDE and PF-ODE. Many works [116, 121, 57, 123] accelerate sampling process by solving the discretized differential equations.

SDE solver is a numerical method used to approximate the solution of an Stochastic Differential Equation (SDE). It discretizes the continuous-time SDE into multiple time steps, enabling efficient

CIFAR-10

SDE Solver

35 50 100 232 275 500 1000 1160 2000 Score SDE [116] ICLR20 - - - - - - 3.21 - 3.10

CLD [117] ICLR21 - 52.70 - - 3.24 2.41 2.27 - 2.23 DSM-ALS [118] ICLR21 - - - 7.50 - - - 5.60 -

Gotta Go Fast [119] ArXiv21 - 72.29 - - 2.74 - - - NCSN [59] NeurIPS19 - - - - - - 25.32 - EDM [57] NeurIPS22 1.97 - - - - - - - -

CIFAR-10 CelebA LSUN 1 2 4 1 2 4 1 2 4

ODE Solver

DDIM [121] ICLR21 13.68 6.84 4.67 17.33 13.73 9.17 19.95 8.89 6.75 PNDM [122] ICLR22 7.05 4.61 3.68 7.71 5.51 3.34 8.69 9.13 9.89

DPM-Solver [123] NeurIPS22 6.37 4.28 3.90 7.15 4.40 4.23 6.10 3.09 2.53 gDDIM [124] ICLR23 41.7 3.03 2.59 - - - - - -

DEIS [125] ICLR23 4.17 3.33 3.36 6.95 3.41 2.95 - - UniPC [126] NeurIPS23 3.87 - - - - - 3.54 - -

NonUniform [127] CVPR24 3.50 - - - - - - - -

CIFAR-10 ImageNet MS-COCO 5 10 20 5 10 20 20 30 40 GGDM [129] ICLR22 13.77 8.23 4.72 55.14 37.32 20.69 - - -

Trajectory Optimization

ReDi [130] ICML23 - - - - - - 25.50 24.70 25.20

- Table 5: Three types of training-free methods are summarized. We further present the generation performance of these methods in terms of efficiency, i.e., Neural Function Evaluations (NFEs ↓) and quality i.e., Fréchet Inception Distance (FID ↓).

sampling from noise to data. The SDE is fundamental to both the forward and reverse processes in generative modeling. Song et. al. [116] unified previous generative models into a common mathematical framework via SDEs in Eqs. 5. Specifically, the forward and reverse processes in DDPM [2] and SMLD [59] are discretizations of the following SDEs:

forward: dxt = −21β(t)xtdt + β(t)dw, reverse: dxt = −21β(t)[xt − ∇xt

DDPM:

(10)

log pt(x)]dt + β(t)dw¯;

 

2(t)) dt dw,

forward: dxt = d(σ

SMLD:

(11)



2(t)) dt ∇xt

2(t)) dt dw.¯

log pt(x)dt + d(σ

reverse: dxt = −d(σ

By carefully designing the SDE and its discretization scheme, the SDE solver seeks to balance the number of steps and approximation errors, thereby improving both the efficiency and quality of outputs in diffusion models.

Noise-Conditional Score Networks (NCSNs) [59] generate new data points through Langevin dynamics, using score matching to estimate the gradient of the data distribution. NCSNs identified three issues when data lies on a low-dimensional manifold: (1) the score function is undefined in low data-density regions; (2) due to the sparsity of training data, score estimation in low-density regions is inaccurate; and (3) in Langevin dynamics, it is difficult to effectively mix different modes of the distribution. To address these problems, NCSNs introduce multi-level noise to perturb the data and adopt Annealed Langevin Dynamics (ALD), where sampling starts with the score corresponding to the highest noise level, and the noise is gradually reduced until convergence to the original data distribution. Building upon this, Jolicoeur-Martineau et. al. [118] discussed the inconsistencies in noise scaling within ALD and proposed Consistent Annealed Sampling (CAS), a score-based MCMC method that ensures noise levels follow a predefined schedule, providing a more stable alternative to ALD.

Also building on Langevin dynamics, Dockhorn et al. introduced Critically-damped Langevin Diffusion (CLD) [117]. As proved in [116], the score function learnt by the neural network is uniquely determined by the forward process, CLD thus posits that a smoother forward process can lead to faster and more efficient sample generation. Inspired by statistical mechanics, CLD introduces a novel SDE by incorporating a velocity variable vt, enabling diffusion in the joint data-velocity space

Knowledge Distillation Symbol Explanation

ℳ with ODE Sampling

🔥

[Figure 208]

Distribution Trajectory Adversary

[Figure 209]

ℳ

𝑋 𝑋

[Figure 210]

ℳ Teacher model in distillation ℳ Student model in distillation

ℳ

adversarial loss

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

𝑡 ~ 𝑈𝑛𝑖𝑓𝑜𝑟𝑚 ({1, …,𝑇})

|supervision|
|---|

supervision

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

ℳ

ℳ

|straight trajectory|
|---|

[Figure 222]

[Figure 223]

[Figure 224]

Generator in GAN objective models

|𝒢|
|---|

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

🔥

###### −

|supervision|
|---|

supervision

[Figure 230]

[Figure 231]

[Figure 232]

distillation

ℳ

Discriminator in GAN objective models

|𝒟|
|---|

ℳ

ℳ

Down Block

Up Block

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

𝑋

𝜖 𝑋

|GM|
|---|

Generative models e.g., GAN, VAE

[Figure 239]

🔓

t

gradient decent

Optimization Strategy

Early Stop GAN Objective

|…<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]|
|---|

|…<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]|
|---|

|…<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]|
|---|

|…<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]|
|---|

[Figure 252]

[Figure 253]

[Figure 254]

|𝒢|
|---|

[Figure 255]

𝑋

𝑋

DiffusionTruncated

[Figure 256]

CommonDiffusion

𝑥

𝑥

[Figure 257]

[Figure 258]

posterior sampling

[Figure 259]

[Figure 260]

[Figure 261]

|GM|
|---|

𝑋

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

𝑥

𝑥

𝑥

…

…

…

…

…

…

𝑋

𝑋

|𝒟|
|---|

[Figure 266]

Gaussian Distribution ~ 𝒩(0, 1)

Random Noises

𝑋

𝑋

Real Fake

GenerateRandomNoises Sampling Process Generated Images

Figure 14: The illustration of training-based methods.

(xt − vt). In CLD, noise is only injected into vt, thereby avoiding the oscillations of under-damped systems and the slow dynamics of over-damped systems. Additionally, CLD only needs to learn

the gradient of the velocity distribution ∇vt

log pt(vt|xt) given the data, which is arguably simpler than learning the score function of the diffused data directly. This method combines Hamiltonian dynamics with the Ornstein-Uhlenbeck process, efficiently exploring the state space and ensuring convergence, thus enabling more efficient sampling and high-quality data generation.

The predictor-corrector method proposed in [116] solves the reverse-time SDE by alternating a numerical SDE solvers (“predictor”) and a score-based Markov Chain Monte Carlo (“corrector”).

- At each time step, the predictor, such as Euler-Maruyama and stochastic Runge-Kutta methods, approximates the reverse-time SDE, providing an estimate of the sample xt at the next time step t. Then a score-based corrector refines the marginal distribution of xt. The predictor enables fast convergence and the corrector ensures sample diversity and quality. The resulting samples maintain the same time marginals as the solution to the reverse-time SDE, which allows them to closely align with the target distribution during the actual generation process. EDM [57] combines a secondorder deterministic ODE integrator with a Langevin-like “churn” perturbation of alternatively adding and removing noise. This approach improves the corrector from [116], achieving state-of-the-art generation quality at the time.

Another issue of numerical SDE solvers is that they require large number of score network evaluations. Jolicoeur-Martineau et. al. [119] devise an SDE solver with adaptive step sizes to accelerate the generation process. The step size is determined by comparing the outputs of a low-order solver and a high-order solver. At each step of the generation process, the solver generates both low-order sample x′l and high-order sample x′h from the previous sample x′prev. The error between these two samples is then evaluated via:

x′l − x′h δ(x′

, δ(x′l,x′prev) = max(ϵabs,ϵrel max(|x′l|,|x′prev|)), (12)

Eq =

l,x′prev) 2

where ϵabs and ϵrel are absolute and relative tolerance. If x′l and x′h are similar, then x′h is accepted and the step size will be increased.

In a more specific situation, CCDF [120] focuses on efficient sampling in conditional image generation tasks by leveraging the contraction property of the reverse diffusion path. It proposes that the generation process does not need to start from pure Gaussian noise but can significantly reduce sampling steps by starting from an initialization closer to the target. The input image is first perturbed with noise up to t0 (where t0 < T , and this noise addition process is nearly “free”), and then reverse denoising starts from t0 to generate the conditional image. As a result, generating target images needs far fewer steps than T. In super-resolution (SR), inpainting, and MRI reconstruction tasks, the method achieves excellent results with only 10, 20, and 20 reverse diffusion steps, respectively.

PF-ODE solver is one of the most commonly used strategy to accelerate the sampling process [121, 122, 57, 123, 237, 124, 125, 130]. Different from SDE Solver, the sampling process of PF-ODE solvers is deterministic, hence is suitable to serve as the teacher model in the knowledge distillation methods [238, 132, 131].

Denoising Diffusion Implicit Models (DDIM) [121] is a notable faster diffusion sampling scheduler, which supports larger denoising steps via a non-Markovian diffusion processes. Particularly, DDIM is a particular formulation of ODE, whose iteration can be rewrote as:

1 − αt−1 αt−1 −

1 − αt αt

1 αt−1

1 αt

)ϵ(θt)(xt), (13)

xt−1 =

xt + (

After reparameterization, the equation can be transformed to the reverse of ODE. Inspired by the observation that when the training dataset contains one sample, DDIM can exactly solve the corresponding SDEs/ODE, Zhang et.al. extend DDIM to general DMs, i.e., gDDIM [124]. Liu et.al. [122] discover two limitations of DDIM. First, the denoising model and ODE are well-defined only in a limited area. However, the sampling process with larger steps may generates samples away from the well-defined area, hence result in new errors. Second, when the index t −→ 0, the ODE equation tends to infinity in many higher-order numerical methods. The phenomenon leads to additional error for the fine-grained denoising steps. To address these issues, PNDM [122] solve the ODE on certain manifolds, which is consists of gradient and transfer parts. The former finds the gradient in each step, and the latter generates the result at the next step. PNDM makes the sampling trajectory is more consistent with the pre-trained area, hence generating higher-quality images with skipped steps. Further, DPM-Solver [123] and DEIS [125] calculate the exact solutions of the diffusion ODEs by semi-linear structure, hence the solvers support larger steps with less error. Specifically, DPM-Solver finds that diffusion ODEs can be divided two parts, i.e. linear (drift coefficient) and non-linear (diffusion coefficient) functions. Previous methods uniformly deal with these two parts, which causes discretization errors particularly on the linear part. Actually, the part can be analytically computed. For the non-linear part, DPM-Solver [123] simplifies the formulation by introducing log-SNR, which is a strictly decreasing function of t. Next, Taylor expansion is utilized for approximating the non-linear part. Moreover, DEIS [125] utilize high-order polynomial extrapolation to reduce the approximation error, which achieves better sampling quality. Besides, to improve the quality of generated samples with accelerated sampling, UniPC [126] utilizes the output ϵθ(xt,t) at current timestep t to correct the predicted sample. NonUniform [127] accelerates diffusion sampling by exploring discretization scheme for time steps, the orders of different steps can be different in the numerical ODE solvers.

Retrieval based methods retrieve trajectory from pre-computed knowledge base to accelerate the sampling process [130]. Inspired by an crucial common sense that the previous sampling steps determine the layout of the images, and the following steps determine the details [239, 240]. ReDi [130] first proposes a retrieval based learning free acceleration strategy. Specifically, the samples of first few steps are generated, which are utilized as the query in the retrieval process. Sequentially, the top-H keys that have highest similarity with the initialized query are selected. Next, the linearly combined values are utilized as the remaining steps of the sampling process.

#### 5.2 Training Based Method

Knowledge Distillation [242] is one of the most common sampling strategy in learning based method, which distills the knowledge from deterministic ODE (teacher) models to the accelerated sampling (student) models. According to the learning objectives, these sampling strategies can be divided into three groups, i.e., distribution-based, trajectory-based, and GAN-based distillations.

Distribution based distillation strategy accelerate the sampling steps of student models by minimizing the image or latent distributions [131–133, 30, 31, 101, 134]. Luhman et.al. [131] first propose Denoising Student to reduce the iterative denoising steps by knowledge distillation. Specifically, the 100-step DDIM scheduler with pre-trained diffusion model is leveraged as the teacher model Mt, which obtains deterministic x0 from the random xT. Meanwhile, the student model Ms use one step denoising setting to accelerate the sampling process. Next, in order to generate high quality images, the predicted distribution of student model Ms is aligned with the iterative denoised §0 from Mt. The learning objective of Ms is formalized as:

CIFAR-10 ImageNet MS-COCO 1 2 4 1 2 4 1 2 4 Denoising Student [131] ArXiv21 9.36 - - - - - - - Progressive Distillation [132] ICLR22 9.12 4.51 3.00 15.99 7.11 3.84 37.2 26.00 26.40

Distribution Based Distillation

Meng et al. [133] CVPR23 7.34 4.23 3.58 22.74 4.14 2.79 - - CM [30] ICML23 3.55 2.93 - 6.20 4.70 - 7.80 5.22 -

LCM [31] ArXiv23 - - - - - - - - 23.49 DMD [134] CVPR24 2.62 - - 2.62 - - 11.49 - -

CIFAR-10 ImageNet LAION-A MS-COCO 1 2 4 1 4 4 8 4 8

Trajectory Based Distillation

TRACT [135] ArXiv23 3.78 3.32 2.93 7.43 4.97 - - - Rectified Flow [136] ICLR22 2.58 - - - - - - - -

InstaFlow [137] ICLR23 - - - - - 14.32 10.98 13.86 11.40 DSNO [28] ICML23 3.78 - - 7.83 - - - - SFT-PG [138] ICML23 - - - - - - - - -

PeRFlow [139] ArXiv24 - - - - - 8.60 8.52 11.31 14.16 Adversarial Based Distillation

CIFAR-10 ImageNet LAION-A MS-COCO 1 2 4 1 4 4 8 4 8

ADD [140] ArXiv23 - - - - - - 20.60 20.80 20.30 LADD [141] ArXiv24 - - - - - - 19.70 - -

CIFAR-10 CelebA-HQ-256 MS-COCO 1 2 4 1 2 4 1 2 4

GAN Objective

DDGAN [142] ICLR22 14.60 4.08 3.75 - 7.74 - - - SIDDMs [143] NeurIPS23 - - 2.24 - 7.37 - 28.00 - 21.70 UFOGen [144] CVPR24 - - - - - - 22.50 - 22.10

CIFAR-10 ImageNet LSUN-Bedroom 50 100 200 50 100 200 50 100 200

Truncated Diffusion

ES-DDPM [145] ArXiv22 - 5.52 5.02 - 3.75 3.47 - 1.85 1.70 TDPM [146] ArXiv22 2.94 2.88 - 1.77 1.62 - 4.34 3.98 -

- Table 6: Five types of training-based methods are summarized. NFEs and FID are presented to demonstrate the efficiency and quality of sampling methods.

[D(Ms(x0|xT),Mt(x0|xT))], (14)

Ls = x

T

D is the function that measures the distance between distributions, which is implemented by KL divergence. To inherit the learned knowledge, Ms is initialized with the original architecture and weight from the Mt. Compared with SOTA one step models e.g., NVAE [243], BigGAN [244], Denoising Student performs better generation ability on standard datasets. Subsequently, considering the expensive time cost caused by the full number of sampling [131], Progressive Distillation [132] is proposed to iteratively accelerate the sampling process. During each iteration, the student model is trained to predict the noise after 2 DDIM sampling steps, and the optimized student model is utilized as the teacher model in the next iteration. Hence the sampling number is reduced in exponential rate. Moreover, Meng et.al. [133] design a two stage training method to apply the distillation strategy to classifier-free models. In the first stage, following [63], the denoised feature of teacher model is calculated by Mt(zt,c) = (1 + w)Mt(zt,c) − wMt(zt). Then, the learning objective of student model is:

w∼pw,t∼U[0,1] ω(λt)||Ms(zt,c,w) − Mt(zt,c)||22 , (15)

where pw = U[wmin,wmax], ω(λt) is the pre-specified weighting function [52]. After distilling student model to fit classifier-free models, the second stage utilizes the progressively distillation strategy [132] to accelerate the sampling steps. In addition to the above methods, consistency models [30] (CMs) is a milestone in efficient inference and sampling, which proposes a remarkable consistency regularization, i.e.,

##### ,tn+1),fθ−(ˆxϕt

##### ,tn))], (16)

##### L = [λ(tn)d(fθ(xt

n+1

n

where λ(tn) denotes the weighting of n-th step, d(·) measures the distance of two distributions, which can be implemented by L1, L2 and LPIPS functions. Given the distribution x in tn+1-th step, xˆϕt

is acquired by running one discretization step of score based denoising model sϕ. The fθ means the trained denoising network, the parameters θ of the network is updated by θ− in an exponential moving average (EMA) manner. Overall, CMs assums that the distribution at any time step in the PF-ODE trajectory can be directly mapped to the distribution at t0. Sequentially, LCM [31] leverages augmented consistency function to align the diffusers with input text conditions, and further designs skipping-step technique to accelerate the convergence of denoising models. Inspired by previous distribution matching methods [245], DMD [134] finetunes the distilled model to learn the fake distribution of pretrained models, which enforce the generated images of student model is indistinguishable from the original teacher model.

n

Trajectory based distillation strategy accelerates sampling process by improving the trajectory of solving PF-ODE [136, 137, 28, 138, 139]. Rectified Flow [136] proposes to rectify the trajectory from a non-linear path to a straight path, which is formally defined as:

min

v

1

0

[||(X1 − X0) − v(Xt,t)||2]dt, withXt = tX1 + (1 − t)X0, (17)

according to the equation, it can be observed that Xt is the linear interpolation of X0 and X1, which models the shortest path between the samples. To build the one-to-one correspondence between

the samples from two distributions π0 and π1, they design reflow method, which first trains the sampling model using randomly selected X0 and X1. Then, the first stage model is leveraged to provide accurate correspondence for training the second stage model. Sequentially, InstaFlow [137] is proposed to acquire a text conditional rectified flow models. To further accelerate the sampling process, PeRFlow [139] trains a piecewise linear flow by creating K times window, and follows the reflow operation to straightening each trajectory. Similarly, DSNO [28] proposes a parallel decoding method, which is accomplished by Fourier neural operator (FNO) [246]. In addition to the strategy of using gradient descent algorithm based on trajectory for distillation, SFT-PG [138] introduces reinforcement learning into efficient sampling. To this end, the policy gradient is utilized to replace the gradient descent, and minimizing the integral probability metrics (IPM) to achieve better generation quality in few steps.

Adversarial based distillation combines the advantages of GAN and diffusion models [140, 141]. Diffusion models have powerful generation capacity, which are able to generate high-quality images [33, 80] and videos [247, 248]. However, these models suffers from iteratively sampling process [124, 249], hindering their application in real-world scenes. On the contrary, GAN models is able to generate images in single-step formulation, but often fall short of the quality, particularly artifacts [250, 251]. Inspired by these observations, ADD [140] introduces a discriminator model [252] to optimize the accelerated sampling model. The adversarial loss is defined as follows:

##### LDadv(Ms(xt,t),ϕ) = x

0

k

max(0,1 − Dϕ(x0)) + γR1(ϕ)

+ M

s

k

max(0,1 + D(Ms(xt,t))) ,

(18)

where Dϕ is the discriminator, R1 is the R1 gradient penalty [253]. Meanwhile, in order to retaining the high quality generation capacity, a pre-trained diffusion model is utilized as teacher model. Although ADD achieves fast sampling model, its denoising process is limited in the pixel level (RGB space) due to the discriminator. Specifically, LDD utilizes DINOv2 [252] as backbone of the discriminator, which cannot predict in latent space. Moreover, the generated images are fixed to 518×518 pixels. To address the issues, LADD [141] unifies the teacher and discriminator, and input discriminator with latent features. Therefore, LADD is able to produce high-resolution images with smaller storage cost.

GAN objective methods utilize multimodal conditional distribution to replace the rigorous Guassian distribution in the diffusion models, which is called denoising diffusion GAN [142–144]. DDGAN [142] first proposes to train diffusion models with GAN objective, which inherits the fast sampling strength of GAN. The most crucial observation is that only small steps achieves regorious Gaussian distribution, and larger steps result in multimodal (peak) distribution. Therefore, to accelerate the sampling process, the multimodal conditional distribution is utilized to replace the unimodal Gaussian distribution. Please note that adversarial based distillation methods discriminate the generated samples and real images, while denoising diffusion GAN models use denoised latent as ‘real’ samples. However, DDGAN can not be applied to large scale dataset due to the nonscalability of GAN. To this end, SIDDMs [143] adds a loss term to explicitly match the conditional distribution. Sequentially, UFOGen [144] is proposed to achieve the one-step sampling. Xu et.al. thinks the failure of DDGAN and SIDDMs mainly caused by the posterior prediction in the denoising process. In this way, the denoising diffusion GAN is able to directly match the distribution of x0.

Optimization strategy contains methods that design acceleration strategies by introducing prior information during training and inference processes [128, 129, 238, 254, 255, 145, 146]. Watson et.al. introduce an dynamic programming algorithm to find the optimal discrete time schedules, which can be applied to any pre-trained DDPMs [128]. The method is based on the decomposability property of evidence lower bound (ELBO) that the total ELBO is the sum of individual KL terms. Then, they maintain two matrices C,D ∈ R(K+1)×(D+1) to find the sampling path in K steps with minimum ELBO . C[k,t] denotes the minimum ELBO in iteration t with k steps, and D[k,t] records the optimal path of the current step, the state transition equation of the dynamic programming can be formally defined as:

(C[k − 1,s] + L(t,s)), (19)

(C[k − 1,s] + L(t,s)),D[k,t] = arg min

C[k,t] = min

s

s

where L(t,s) is the decomposed ELBO from t to s. However, the metric used in [128] has a mismatch with the quality of generated images, e.g., FID scores. To address this issue, GGDM utilizes Kernel Inception Distance (KID) as perceptual loss to obtain high-fidelity images [129].

Truncated diffusion methods accelerate the sampling process by introducing early stop into training and inference process [145, 256]. The denoising process is started from non-Gaussian distributions, then we can perform only a few denoising steps to generate the high-quality images. Specifically, the non-Gaussian distributions is obtained from existing generative models such as GAN [245, 257] and VAE [167], which is able to approximate the distribution of the data without expensive iteration process.

### 6 Efficient Deployment and Usage

The previous sections explored various efficient diffusion model techniques from a research perspective, focusing on model architecture, training and fine-tuning, sampling and inference optimizations. This section shifts focus to the real-world deployment and application of diffusion models. We divide the deployment and usage scenarios into two main categories: “Efficient Deployment as a Tool” and “as a Service”, as shown in Figure. 15. The former is aimed at users who are already familiar with the fundamental processes of image generation using diffusion models, while the latter requires greater enterprise-level support to provide broader audiences with well-packaged, "one-click" image generation services.

#### 6.1 Efficient Deployment as a Tool

In practical applications, the efficient deployment of diffusion models as tools is crucial for researchers, developers, and other AIGC practitioners. These users require a high degree of flexibility and control over the generation process to adjust and optimize model configurations across various scenarios. This type of deployment offers an environment for deep experimentation and customization, fully leveraging the potential of diffusion models. It is especially suited for tasks that require testing multiple model configurations, adjusting noise parameters, optimizing performance, or integrating custom components. Therefore, tool-based deployment typically emphasizes modular design, scalability, adaptability to diverse needs, and a high level of control.

Advanced Normal Develop

Users Companies

On edge devices

On cloud devices

[Figure 267]

[Figure 268]

Smart phones

PC Dedicated chips

Node-based ComfyUI

High demand High resolution High-end devices

[Figure 269]

[Figure 270]

[Figure 271]

Model compression ......

Tab-style Automatic1111's WebUI

[Figure 272]

[Figure 273]

Elastic resource Curated Parallelism

Less inference steps

...

Deploy as a Tool

Deploy as a Service

Figure 15: Efficient deployment as a tool and as a service.

In implementation, these tools must strike a balance between ease of use and technical depth. Professional users need an interface that is both intuitive and allows for in-depth adjustment of model parameters. Achieving this balance poses significant design challenges, requiring tools that cater to expert needs without overwhelming the user with complexity.

Taking ComfyUI3 as an example, it employs a “node-based workflow interface”, allowing users to visually create and modify complex image generation processes. By connecting different nodes, users can construct each step of the model and flexibly adjust the parameters and hyperparameters of each module. This modular design is particularly well-suited for users who seek to refine and customize the generation process, especially researchers and developers who benefit from being able to track each stage of the workflow from input to output. ComfyUI’s node-based architecture greatly facilitates the integration of custom models and new algorithms. Users can easily introduce new nodes, algorithms, or functional modules to experiment with. This is especially beneficial for developers, as they can flexibly swap components without needing to overhaul the entire system. Researchers, on the other hand, can quickly and conveniently compare the performance of different model components before and after adjustments. However, the flexibility of ComfyUI also makes its learning curve steeper, making it more suitable for users who have a deeper understanding of the overall diffusion model process.

In contrast, Stable Diffusion WebUI4 (commonly referred to as Automatic1111 or WebUI) offers a simple form-like interface. Users can quickly generate images by entering parameters such as prompts, number of steps, CFG scale, and image resolution. This design is particularly well-suited for users who want a fast and straightforward image generation process, especially beginners. Even though the detailed image generation workflow is hidden, WebUI still provides advanced features and customization options to meet the needs of more experienced users. Through its plugin system, users can realize various features, such as “inpainting” and personalized training tools like Textual Inversion and ControlNet. While it lacks flexibility of the node-based ComfyUI, the ease of using plugins makes it ideal for users who want to expand functionality without extensively modifying the model. Automatic1111’s WebUI is more user-friendly and accessible, with its streamlined formbased interface allowing users to input parameters and generate images quickly, making it suitable for those looking for fast results. For users without a strong technical background, it offers a true "plug-and-play" experience.

- 3https://github.com/comfyanonymous/ComfyUI
- 4https://github.com/AUTOMATIC1111/stable-diffusion-webui

These tools offer users extensive control over the generation process, from adjusting the number of diffusion steps to integrating custom plugins or models tailored for specific domains. They not only meet the needs of advanced users involved in research and development but also address the practical requirements of deployment in production environments. When deployed in cloud environments, these tools typically provide scalable infrastructure to accommodate large-scale workflows. For instance, ComfyUI can seamlessly integrate with Amazon EKS, enabling dynamic scaling of GPU instances to meet the demands of large-scale parallel inference in the cloud. Additionally, an active user community contributes numerous resources to these tools, including comprehensive APIs and documentation, encouraging developers to create and share custom plugins. This open ecosystem not only enriches the tools’ functionality but also opens up new possibilities for various applications, spanning from artistic creation to scientific research and industrial design.

#### 6.2 Efficient Deployment as a Service

Efficient Deployment as a Service is aimed at a broader user base, typically requiring neither advanced technical expertise nor local high-end computational resources. Service providers package comprehensive tools to simplify the complex processing of diffusion models into a "one-click" user experience. Their efforts are focused on optimizing the inference process and user interaction for real-world deployment scenarios on mobile and cloud platforms. The goal is to deliver faster, more stable inference services that meet the needs of everyday users, while also addressing cost control and privacy concerns.

In [258], Google optimizes GPU memory I/O to significantly reduce inference latency on mobile devices via two key improvements: enhanced attention modules and Winograd convolution. By using partially fused Softmax to reduce memory access for large intermediate matrices, along with FlashAttention to lower memory bandwidth pressure, the attention mechanism’s efficiency was greatly enhanced. Additionally, Winograd convolution accelerated the 3 × 3 convolution layers, striking a balance between computational efficiency and memory usage. Tests showed that on the Samsung S23 Ultra and iPhone 14 Pro Max, the latency for generating 512px resolution images was reduced by 52.2% and 32.9%, respectively, with inference time dropping to under 12 seconds over 20 steps and memory usage capped at 2,093 MB.

Despite these improvements, latency remains high for interactive mobile applications. SnapFu-

- sion [147] made a breakthrough by reducing inference time to under 2 seconds for text-to-image generation on mobile devices. To achieve this, SnapFusion optimized the UNet by removing redundant computations through an evolving-training framework. To further reduce inference steps, it introduced CFG-aware step distillation, greatly enhancing both efficiency and stability. Tests on the iPhone 14 Pro demonstrated that SnapFusion can generate 512px images in just 2 seconds, and in experiments on the MS-COCO dataset, it achieved superior FID and CLIP scores using only 8 denoising steps, outperforming Stable Diffusion v1.5 with 50 steps. To further optimizes both the architecture of diffusion models for mobile devices, MobileDiffu-
- sion [148] redesigns the UNet by sharing projection matrices, replacing activation functions, and adopting separable convolutions to achieve a lightweight model. The VAE decoder is pruned for width and depth while increasing latent channels, accelerating decoding while maintaining reconstruction quality. For sampling, it introduces UFOGen’s Diffusion-GAN hybrid training method [144], enabling one-step sampling. By leveraging adversarial fine-tuning and distillation techniques, the model generates high-quality images in just one step. On the iPhone 15 Pro, MobileDiffusion generates 512px images in under 0.2 seconds, while also supporting various downstream applications such as controlled generation (e.g., based on text, canny edge or depth map), personalized generation (e.g., Style-LoRA, Object-LoRA), and in-painting.

However, due to the limited computational resources of mobile devices, it is difficult to achieve fast generation of high-quality, high-resolution images. Applications that need to handle largescale tasks while requiring high-speed generation often rely on efficient deployment on cloud-based infrastructure of service providers. Cloud deployment not only leverages more powerful hardware resources to handle complex tasks but also improves the efficiency of concurrent inference through distributed computing and elastic scaling, as shown in Figure.16.

To achieve low-latency, high-resolution image generation without compromising image quality, DistriFusion [149] focuses on parallelism across multiple GPUs. Observing the high similarity between

- (a) Displaced Patch Parallelism: Distribute patches to devices with complete model

- (b) Asynchronous Denoising: Shard model to devices, input complete image (latent)
- (c) Displaced Patch Pipeline Parallelism: Shard model and distribute patches to devices

[Figure 274]

UNet ... UNet

- Slice0 UNet

[Figure 275]

[Figure 276]

Device 0

[Figure 277]

[Figure 278]

- Slice1

Shards image

[Figure 279]

Cross-device communication

...

[Figure 280]

| |Shards model|
|---|---|
| |Shards both|
| | |

Device 1

UNet UNet ...

UNet

Cancat historic activations

[Figure 281]

- Device 0
- Device 1
- Device 2

UNet UNet

UNet

UNet

xt

l

t

t-1

t-2

UNet

Denoise Backbone

UNet

UNet

UNet

UNet

t

t-2

Only use highlight layers

xt-1 xt-2 xt-3

UNet

UNet

UNet

t

t-1

t-2

[Figure 282]

[Figure 283]

DiT DiT DiT DiT ...

t = 0

DiT

[Figure 284]

[Figure 285]

DiT DiT DiT ...

DiT

t = 1

[Figure 286]

...

Shards complete DiT to different devices:

t = 2

DiT DiT

DiT

[Figure 287]

- Device 0
- Device 1

Process of image generation via diffusion models

...

t = 3

DiT

DiT

Figure 16: Efficient cloud-based deployment strategies for diffusion models.

inputs from adjacent diffusion steps, it reuses activations from previous steps to provide global context and inter-block interaction. Based on this, DistriFusion proposes Displaced Patch Parallelism, where the input image is divided into multiple patches and processed in parallel by SD-XL on different GPUs. The global results from the previous step are reused to approximate the context for the current step, while asynchronous communication prepares the global context for the next step, effectively hiding communication latency. In practice, DistriFusion achieves speedups of approximately 2.8×, 4.9×, and 6.1× for generating images at 1024px, 2048px, and 3840px resolutions, respectively, using 8 A100 GPUs, without sacrificing image quality, compared to single A100 GPU processing.

To address the computational and latency challenges of generating high-resolution images with Diffusion Transformers (DiT) across multiple GPUs, PipeFusion [150] also leverages the high similarity between inputs from adjacent steps. However, applying DistriFusion method to DiT can result in inefficient memory usage due to the need for large communication buffers. To overcome this, PipeFusion introduces Displaced Patch Pipeline Parallelism. This method divides the image into patches and distributes transformer layers across different GPUs, using pipeline parallelism for computation and communication. By transmitting only the input activations of the initial layer and the output activations of the final layer via asynchronous point-to-point (P2P) communication between adjacent devices, it significantly reduces data transfer and memory usage. Tested on three GPU clusters using PCIe or NVLink, PipeFusion outperforms other parallelization techniques in terms of end-to-end latency at various resolutions. For instance, in a 4 A100 (PCIe) cluster, PipeFusion achieves latency reductions of 2.01×, 1.48x×, and 1.10× at 1024px, 2048px, and 8192px resolutions, respectively. This is especially significant at 8192px, where other methods often face “Out Of Memory” issues. PipeFusion dramatically lowers the required communication bandwidth, enabling the DiT model to run efficiently on GPUs connected via PCIe, without the need for costly NVLink infrastructure, thus significantly reducing operational costs for service providers.

Unlike patch-based parallel methods, AsyncDiff [151] focuses on asynchronous parallel inference. In traditional diffusion models, denoising steps are performed sequentially, where each step’s input depends on the previous step’s output. AsyncDiff breaks this dependency chain by also leveraging

the high similarity between inputs from adjacent diffusion steps, enabling parallel computation of denoising components. It introduces asynchronous denoising, model parallel strategies, and stride denoising, allowing multiple denoising steps to be processed concurrently in a single parallel round, reducing the number of parallel computation rounds and communication frequency between devices. This approach significantly improves inference speed while maintaining image quality. On four NVIDIA A5000 GPUs, AsyncDiff achieved a 4× speedup on SDv2.1 with only a 0.38 reduction in CLIP score. Additionally, this method is also effective for video diffusion models, significantly reducing latency while maintaining high video quality.

### 7 Applications

In the above analyses, we summarize efficient diffusion models by focusing on five critical components. Next, we conduct a comprehensive review of previous work, showcasing how these models have been applied in various contexts, including image synthesis, image editing, video generation, video editing, 3D synthesis, medical imaging, and bioinformatics engineering, while assessing their strengths and limitations. Based on this foundation, we propose potential development directions aimed at enhancing the efficiency and effectiveness of diffusion models in future applications.

#### 7.1 Image Synthesis

[Figure 288]

[Figure 289]

(a) The number of relevant research works (b) Ratios of different tasks

Figure 17: The number of research papers on Efficient Diffusion Models published between 2022 and 2024

Image synthesis plays an important role in computer vision and has widespread applications in fields such as artistic creation and personalized content generation. The application of diffusion models to image synthesis gains prominence with the emergence of text-to-image diffusion models [32, 6, 5, 33, 206, 286], enabling the generation of high-quality images from natural language descriptions. Subsequently, efficient fine-tuning techniques expand the application of diffusion models to various conditional image generation tasks, including the structures [9, 42] and content [10, 111]. Meanwhile, research into efficient sampling methods further facilitates the practical application of these technologies, driving the broader advancement of image synthesis.

Customized generation is an important research direction in image synthesis, aiming to achieve tailored outputs that meet specific user needs. Dreambooth [10] introduces subject-driven customized generation [287–289, 232, 112, 111, 290, 291], which faithfully preserves the visual content of the themes depicted in the provided samples. In addition, identity customization [231, 292, 293, 230, 294, 229, 295] is achieved through the high-fidelity preservation of facial features. Moreover, some work focuses on visual text generation [296–302] , emphasizing accurate text creation within images, which aids in producing high-quality posters. At the same time, there are also interesting developments in visual storytelling [303–307] applications, which aim to generate a coherent series of images, such as comics, to enhance the efficiency of artistic creation. Finally, in the field of safe image generation, privacy and copyright protection techniques [308–316] have become key research priorities.

#### 7.2 Image Editing

Diffusion models have demonstrated powerful controllable generation capabilities, which are inherently well-suited for editing tasks that require adjustments during the generation process. Among

###### Application Name Organization State Demo Program Weight

FLUX.1 dev Black Forest Labs Open source [demo] [program] [weight] FLUX.1 pro Black Forest Labs Closed Source [demo] - SD3-Ultra Stability AI Closed Source [demo] - Ideogram Ideogram AI Closed Source [demo] - FLUX.1 schnell Black Forest Labs Open source [demo] [program] [weight] Midjourney 6.0 Midjourney Closed Source [demo] - -

Image Synthesis

DALL-E 3 HD [259] OpenAI Closed Source [demo] - -

SD3 Medium [80] Stability AI Open source [demo] [program] [weight]

OutfitAnyone [260] Alibaba Group Closed Source [demo] - M&M VTO [261] Google Research Closed Source - - Diffuse to Choose Amazon Closed Source [demo] - -

Image Editing

DEADiff [262] ByteDance Open source [demo] [program] [weight]

DragDiffusion [263] ByteDance Open source [demo] [program] Sora OpenAI Closed Source [demo] - Gen-3 Alpha Runway Closed Source [demo] - -

Stable Video Diffusion [14] Stability AI Open source [demo] [program] [weight]

Open-Sora HPC-AI Technology Open source [demo] [program] [weight] VideoCrafter [15] Tencent AI Lab Open source [demo] [program] [weight]

Video Generation

Latte [79] Shanghai AI Lab Open source [demo] [program] [weight] MagicVideo-V2 [18] ByteDance Closed Source [demo] - -

NUWA-XL [264] Microsoft Research Asia Closed Source [demo] - -

W.A.L.T [5] Google Research Closed Source [demo] - GenTron [265] Meta AI Closed Source [demo] - -

Text2Video-Zero [266] Picsart AI Resarch Open source [demo] [program] -

ViViD [267] Alibaba Group Open source [demo] [program] [weight] MotionEditor [268] Fudan University Open source [demo] [program] -

FLATTEN [269] Meta AI Open source - [program] [weight]

Video Editing

Dreamix [270] Google Research Closed Source [demo] - ControlVideo [271] Huawei Cloud Open source [demo] [program] -

Rerender_A_Video [272] NTU Open source [demo] [program] RodinHD [273] Microsoft Research Asia Open source [demo] [program] CAT3D [274] Google DeepMind Closed Source [demo] - DreamFusion [23] Google Research Closed Source [demo] - SV3D [27] Stability AI Closed Source [demo] - -

3D Synthesis

DiffPortrait3D [275] ByteDance Open source [demo] [program] [weight]

Inpaint3D [276] Google Research Closed Source [demo] - TextureDreamer [277] Meta AI Closed Source [demo] - -

ViewCrafter [278] Tencent AI Lab Open source [demo] [program] [weight] AlphaFold3 [279] DeepMind Open source [demo] [program] -

Bioinformatics Engineering

DiffDock [280] MIT Open source [demo] [program] RFdiffusion [281] University of Washington Open source [demo] [program] -

DiffAb [282] Helixon Research Open source [demo] [program] DiffMa [283] Sichuan University Open source - [program] [weight] DDM2 [284] Stanford University Closed Source - [program] -

Medical Imaging

ScoreInverseProblems [285] Stanford University Open source - [program] [weight]

BrLP University of Catania Open source [demo] [program] [weight]

Table 7: State-of-the-art models across various applications

these methods, instruction-based editing techniques [317–325, 156, 326] have the broadest applicability and align most closely with human habits. However, they are constrained by the expensive fine-tuning costs to learn the editing instructions. Therefore, some researchers have concentrated on domain-specific editing techniques [327–333] to address this issue. On the other hand, some work focuses on fine-tuning during the inference stage to further enhance editing efficiency. This includes techniques such as text embedding fine-tuning [154, 334, 240, 335], latent variable optimization [263, 336, 337, 328, 338], and fine-tuning of the diffusion model itself [339, 340, 155, 341]. Currently, finetuning-free methods have shown significant potential for efficient editing, attracting increased research attention. To avoid fine-tuning, researchers have closely analyzed the attention layers that interact most frequently with editing control conditions and proposed the classic attention modification methods [153, 342–349]. Subsequently, sampling modification [350–360] and mask guidance [361–366] techniques were introduced to further enhance accuracy.

These techniques have been widely adopted across various editing scenarios. For example, the recently popular virtual try-on technology [261, 367–376, 260] on e-commerce platforms allows users to better visualize how garments will look when worn. Additionally, image style transfer technology [9, 42, 94, 111, 262, 330, 377, 378, 367, 379] allows for the flexible generation of

stylized and customized images, preserving the original content while showcasing a diverse range of visual styles. On the other hand, diffusion model-based methods have also shown outstanding performance in solving low-level vision tasks, such as super-resolution [54, 120, 380–382, 157, 383–392], deblurring [54, 385, 386, 388–390, 392–399], inpainting [120, 389–393, 400–403], and compression artifact removal [404–407]. These can be seen as a broader form of the editing process.

#### 7.3 Video Generation

The essence of video is a sequence of images ordered temporally. Consequently, text-to-video synthesis techniques [11, 18, 12, 35, 408, 169, 172, 409, 410, 266, 411, 412, 265] based on diffusion models greatly benefit from advancements in text-to-image synthesis technology, including shared aspects such as model architecture [33, 12]and training methods [9, 98, 205, 211]. In addition, similar to controllable image generation techniques, video generation has also integrated various control conditions, such as image-guided [15, 34, 413, 414], pose-guided [415–419], motion-guided [415, 420], sound-guided [421–423], depth-guided [424, 425], and multi-modal guided [426–428] approaches. These advancements further enhance controllability and improve the efficiency of custom content creation.

As a dynamic form of images, video emphasizes the controllability of motion [34, 412, 413, 429– 433], making it a crucial research direction in video generation. It allows users to precisely control motion trajectories and dynamic effects, providing greater creative freedom and more accurate visual expression. Meanwhile, character animation [417, 415, 434–436] is a fascinating task that aims to generate character videos from static images using driving signals. Through this process, characters can exhibit natural movements and expressions, resulting in lively and dynamic content. Additionally, world models have become a significant research focus, particularly for the field of autonomous driving [437–441]. These models show great potential for generating high-quality driving videos and designing safe driving strategies by simulating real-world scenarios.Currently, generating longer videos [264, 11, 442, 163, 79] is a highly challenging task, but it holds the potential to create more complex and content-rich visual works.

#### 7.4 Video Editing

Text-guided video editing aims to achieve similar goals to image editing, but with videos as the target for editing. These techniques can be categorized based on their efficiency in achieving editing capabilities. The first category involves training on large-scale video-text datasets to develop generalized editing capabilities, which is the most straightforward approach to developing generalized editing capabilities. The second category, one-shot tuning methods, refines pre-trained models using specific video instances to provide more accurate and contextually relevant video editing, offering a balanced trade-off between effectiveness and efficiency. Finally, training-free methods adapt pretrained models in a zero-shot manner but often face challenges with spatio-temporal distortions. These issues are addressed through techniques such as feature propagation, hierarchical constraints, and attention mechanisms.

One of the fundamental goals of video editing is to maintain temporal consistency between frames, ensuring that the generated video appears smooth and natural. Building on this foundation, virtual try-on for videos represents a significant application that aims to enhance the user’s ability to edit the content and appearance of objects [443, 267, 444–446, 17], allowing for a more realistic experience of different garments or accessories. Concurrently, video action editing has also garnered considerable attention [447, 448, 420, 268], focusing on the flexible manipulation of character or object movements. Recently, research has introduced unified models that integrate these two aspects, aiming to achieve more efficient editing [449–452, 269–271, 269]. This approach not only enhances the flexibility of editing processes but also preserves video coherence, ultimately providing users with a superior editing experience.

#### 7.5 3D Synthesis

- 3D synthesis [23, 25, 221, 453–457, 274, 27, 277, 278] is a technique used to create and combine three-dimensional images or scenes [458–464, 276], typically involving the integration of multiple 3D models, textures, and lighting effects to generate realistic 3D visuals. This technology is widely used in film production, video games, virtual reality, augmented reality, and computer graphics.

Through 3D compositing, users can create highly detailed and dynamic 3D environments, enhancing visual immersion and interactive experiences.

Human motion modeling [465–474] is a crucial component in animating virtual characters to mimic lifelike and dynamic human movements. This area becomes an important focus in various applications, including film production and game development. Building on the foundation of human motion modeling, 3D digital avatars [273, 475–478, 275] take this concept further by creating digital representations of three-dimensional virtual characters. These avatars not only feature realistic appearances and personalized traits but also possess the ability to perform a wide range of actions and behaviors. Moreover, to enhance the realism and interactivity of these avatars, speech-driven gesture [479–482] plays a vital role. By integrating hand, arm, and body movements with speech, this aspect of embodied human communication significantly improves both human-computer interaction and human-human digital communication.

- 7.6 Medical Imaging

Medical imaging is significantly different from traditional imaging due to its complexity and multimodality, involving various techniques such as MRI, CT, and ultrasound. Researchers optimize diffusion models based on the unique characteristics of these imaging modalities, enabling the generation of high-quality medical images that alleviate data scarcity issues and enhance the accuracy of image analysis. One of the most prominent research directions is the generation of missing imaging types, such as translating CT images to MRI [483–485, 283], which improves diagnostic consistency while reducing time and costs. Furthermore, diffusion models directly synthesize high-quality medical images [486–493], thereby effectively alleviating data scarcity issues. By annotating real and high-quality synthetic data, self-supervised learning methods have shown broad potential in tasks such as medical image classification [494–499], segmentation [500, 489, 501, 502], reconstruction [503, 285, 504–509], and denoising [510, 511], thus advancing the development of medical image analysis. Additionally, diffusion models excel in medical anomaly detection [495–499] by generating healthy images and identifying abnormal regions, enhancing diagnostic accuracy and efficiency. These applications indicate that diffusion models hold significant promise and value in the realm of medical imaging.

- 7.7 Bioinformatics Engineering

Diffusion models emerge as highly promising tools in bioinformatics due to their strong capacity for processing high-dimensional data, generating diverse synthetic data, and flexibly adapting to various bioinformatics tasks. Diffusion models provide a more versatile approach to protein design and generation [512–516, 281, 517–519, 279, 282, 520], overcoming the limitations of traditional generative models that can only produce small proteins or specific domains. This capability enables scientists to design proteins with specific functional or structural characteristics, which is crucial for protein engineering and drug discovery. Additionally, in molecular design [521–530], diffusion models effectively model linkers between fragmented molecular components to identify and optimize small molecules that interact with biological targets, such as enzymes or receptors, thereby accelerating the generation and evaluation of potential drug candidates. Lastly, predicting the conformations of ligands bound to proteins is essential for studying protein-ligand interactions [531, 532, 280, 533, 534]. Diffusion models facilitate this by predicting the interaction patterns of small molecules (ligands) with specific binding sites on proteins, helping researchers quickly identify potential drug candidates and optimize their designs to enhance drug efficacy and selectivity. In summary, diffusion models play a critical role in protein design and generation, small molecule and drug design, and modeling protein-ligand interactions, significantly advancing the application of bioinformatics.

- 8 Discussion and Conclusion

- 8.1 Limitations and Future work

Although the theories and applications of diffusion models have emerged rapidly in recent years, there are still many limitations that deserve our attention and solution for efficient diffusion models. We briefly sort out these concerns below, hoping that they can serve as potential directions for future exploration:

- • Despite the excellent performance, the architecture of the current diffusion models still suffer from high computational complexity caused by the attention computing, especially in 3D and video diffusion models, which leads to heavy FLOPs and inference latency. Moreover, current fine-tuned models still struggle with poor task and domain generalization. For instance, ControlNets solely excels at handling vision-conditional controllable generation and Adapters require structural redesign for each new task, which reduce their flexibility. A feasible future direction would involve designing a unified framework that can handle multimodal and multi-granularity control, enabling more efficient and versatile task adaptation without the need for extensive architectural adjustments.
- • By integrating MoE [535, 536] designs into diffusion models, dynamic networks can be created to support a more flexible generation process [537, 538]. By dynamically activating different expert modules according to the input data’s characteristics, the model can allocate computational resources more efficiently, thereby improving its adaptability to diverse and complex tasks.
- • The trade-off between training cost and sampling quality is another critical issue. Up to now, most existing training-free methods still require over 10 sampling NFEs to ensure high-fidelity. Despite training-based methods can accelerate the process, they are still limited by the high training costs, such as time, data, and GPU resources. An effective route is to integrate these above two methods for high-quality and low-consumption model training.
- • In terms of deployment and practical application, current diffusion models still take considerable time, high computational cost and storage requirements, especially in low-resource environments. Moreover, model compression and high-quality generation is still an important balance. Last but not least, communication overhead and memory efficiency in large-scale inference require further optimization. Researchers could focus on more efficient model compression or distillation, improved cross-device communication strategies, and leveraging hardware acceleration to enhance inference speed, stability, and image quality in future works.

In addition to the above mentioned directions, researchers can also investigate speculative decoding [539, 540] by employing a collaborative approach between small and large models to improve the efficiency of diffusion models [541, 542]. In this direction, a small diffusion model rapidly generates preliminary structures or rough sketches, which the larger model subsequently refines and completes them. The division of tasks will reduce the computational load on the larger model, enhancing overall efficiency while preserving generation quality.

#### 8.2 Conclusion

In this study, we conduct an in-depth and comprehensive review and take a deep dive into the realm of efficient DMs literature, providing an all-encompassing view of its central challenges and themes, including foundational theories and principles, as well as their extensive practices. Our goal is to identify and highlight areas that require further research and suggest potential avenues for future studies. This survey aim to provide a comprehensive perspective on the current state of efficient diffusion models, with the hope of inspiring additional research and exciting works. Given the dynamic nature of this field, it’s possible that some recent developments may not be fully covered. To counter this, we will set up a dedicated website that uses crowdsourcing to keep up with the latest advancements. This platform is intended to serve as a continually updated source of information, promoting ongoing growth in the field. Due to space constraints, we can’t cover all technical details in depth but have provided brief overviews of the key contributions in the field. In the future, we plan to continuously update and enhance the information on our website, adding new insights as they come to light.

### References

- [1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pages 6840–6851, 2020.
- [3] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [4] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [5] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.
- [6] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [7] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890– 16902, 2022.
- [8] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2023.
- [9] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836– 3847, 2023.
- [10] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
- [11] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022.
- [12] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.
- [13] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [14] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [15] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [16] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047, 2024.
- [17] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050, 2023.
- [18] Weimin Wang, Jiawei Liu, Zhijie Lin, Jiangqiao Yan, Shuo Chen, Chetwin Low, Tuyen Hoang, Jie Wu, Jun Hao Liew, Hanshu Yan, et al. Magicvideo-v2: Multi-stage high-aesthetic video generation. arXiv preprint arXiv:2401.04468, 2024.

- [19] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024.
- [20] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2020.
- [21] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pages 13916–13932. PMLR, 2023.
- [22] Dongchao Yang, Jianwei Yu, Helin Wang, Wen Wang, Chao Weng, Yuexian Zou, and Dong Yu. Diffsound: Discrete diffusion model for text-to-sound generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
- [23] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2022.
- [24] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In The Twelfth International Conference on Learning Representations, 2023.
- [25] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023.
- [26] Joseph Zhu and Peiye Zhuang. Hifa: High-fidelity text-to-3d with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023.
- [27] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024.
- [28] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In International conference on machine learning, pages 42390–42402, 2023.
- [29] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2021.
- [30] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252, 2023.
- [31] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [32] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [34] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [35] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023.
- [36] Dongjun Kim, Byeonghu Na, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, and Il-chul Moon. Maximum likelihood training of implicit nonlinear diffusion model. Advances in Neural Information Processing Systems, 35:32270–32284, 2022.
- [37] Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang. Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data. In International Conference on Machine Learning, pages 4672–4712. PMLR, 2023.

- [38] Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based diffusion odes by high order denoising score matching. In International Conference on Machine Learning, pages 14429–14460. PMLR, 2022.
- [39] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. Advances in NeurIPS, 34:1415–1428, 2021.
- [40] Yuhan Li, Yishun Dou, Xuanhong Chen, Bingbing Ni, Yilin Sun, Yutian Liu, and Fuzhen Wang. Generalized deep 3d shape prior via part-discretized diffusion process. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16784–16794, 2023.
- [41] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, pages 10696–10706, 2022.
- [42] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4296–4304, 2024.
- [43] Calvin Luo. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970, 2022.
- [44] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
- [45] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Yingxia Shao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. arXiv preprint arXiv:2209.00796, 2022.
- [46] Hanqun Cao, Cheng Tan, Zhangyang Gao, Guangyong Chen, Pheng-Ann Heng, and Stan Z Li. A survey on generative diffusion model. arXiv preprint arXiv:2209.02646, 2022.
- [47] Zhiyuan Ma, Liangliang Zhao, Biqing Qi, and Bowen Zhou. Neural residual diffusion models for deep scalable vision generation. arXiv preprint arXiv:2406.13215, 2024.
- [48] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.
- [49] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [50] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [51] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.
- [52] Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Proceedings of the 35th International Conference on Neural Information Processing Systems, pages 21696–21707, 2021.
- [53] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, pages 8162–8171, 2021.
- [54] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in Neural Information Processing Systems, 35:23593–23606, 2022.
- [55] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2021.
- [56] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in NeurIPS, 34:17981–17993, 2021.
- [57] Tero Karras, Miika Aittala, Samuli Laine, and Timo Aila. Elucidating the design space of diffusionbased generative models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pages 26565–26577, 2022.

- [58] Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise. Advances in Neural Information Processing Systems, 36, 2024.
- [59] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pages 11918–11930, 2019.
- [60] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. Advances in NeurIPS, 34:11287–11302, 2021.
- [61] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020.
- [62] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in Artificial Intelligence, pages 574–584. PMLR, 2020.
- [63] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.
- [64] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. More control for free! image synthesis with semantic diffusion guidance. arXiv preprint arXiv:2112.05744, 2021.
- [65] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [66] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [67] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2022.
- [68] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and time-sensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022.
- [69] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459– 10469, 2023.
- [70] Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, and Ying Shan. Cv-vae: A compatible video vae for latent generative video models. https://arxiv.org/abs/2405.20279, 2024.
- [71] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [72] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.
- [73] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [74] Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai. Fit: Flexible vision transformer for diffusion model. arXiv preprint arXiv:2402.12376, 2024.
- [75] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.

- [76] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024.
- [77] Vincent Tao Hu, Stefan Andreas Baumann, Ming Gui, Olga Grebenkova, Pingchuan Ma, Johannes Fischer, and Bjorn Ommer. Zigma: Zigzag mamba diffusion model. arXiv preprint arXiv:2403.13802, 2024.
- [78] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, and Junshi Huang. Dimba: Transformer-mamba diffusion models. arXiv preprint arXiv:2406.01159, 2024.
- [79] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.
- [80] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for highresolution image synthesis. In International Conference on Machine Learning, 2024.
- [81] Junsong Chen, YU Jincheng, GE Chongjian, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2023.
- [82] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [83] The Movie Gen team@Meta. Movie gen: A cast of media foundation models, October 2024.
- [84] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [85] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [86] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [87] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [88] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [89] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [90] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. arXiv preprint arXiv:2103.10360, 2021.
- [91] Denis Zavadski, Johann-Friedrich Feiden, and Carsten Rother. Controlnet-xs: Designing an efficient and effective architecture for controlling text-to-image diffusion models. arXiv preprint arXiv:2312.06573, 2023.
- [92] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, Ming-Chang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024.
- [93] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback. arXiv preprint arXiv:2404.07987, 2024.

- [94] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [95] Lingmin Ran, Xiaodong Cun, Jia-Wei Liu, Rui Zhao, Song Zijie, Xintao Wang, Jussi Keppo, and Mike Zheng Shou. X-adapter: Adding universal compatibility of plugins for upgraded diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8775– 8784, 2024.
- [96] Shanshan Zhong, Zhongzhan Huang, Weushao Wen, Jinghui Qin, and Liang Lin. Sur-adapter: Enhancing text-to-image pre-trained diffusion models with large language models. In Proceedings of the 31st ACM International Conference on Multimedia, pages 567–578, 2023.
- [97] Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7827–7839, 2024.
- [98] Han Lin, Jaemin Cho, Abhay Zala, and Mohit Bansal. Ctrl-adapter: An efficient and versatile framework for adapting diverse controls to any diffusion model. arXiv preprint arXiv:2404.09967, 2024.
- [99] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.
- [100] Yang Yang, Wen Wang, Liang Peng, Chaotian Song, Yao Chen, Hengjia Li, Xiaolong Yang, Qinglin Lu, Deng Cai, Boxi Wu, et al. Lora-composer: Leveraging low-rank adaptation for multi-concept customization in training-free diffusion models. arXiv preprint arXiv:2403.11627, 2024.
- [101] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.
- [102] Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. arXiv preprint arXiv:2311.12092, 2023.
- [103] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2023.
- [104] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023.
- [105] Miao Hua, Jiawei Liu, Fei Ding, Wei Liu, Jie Wu, and Qian He. Dreamtuner: Single image is enough for subject-driven generation. arXiv preprint arXiv:2312.13691, 2023.
- [106] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.
- [107] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.
- [108] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, SHUM KaShun, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research, 2023.
- [109] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.
- [110] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [111] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024.

- [112] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023.
- [113] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized low-rank adaptation for multiconcept customization of diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [114] Daniil Ostashev, Yuwei Fang, Sergey Tulyakov, Kfir Aberman, et al. Moa: Mixture-of-attention for subject-context disentanglement in personalized image generation. arXiv preprint arXiv:2404.11565, 2024.
- [115] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multi-concept generation in diffusion models. arXiv preprint arXiv:2403.10983, 2024.
- [116] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020.
- [117] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with criticallydamped langevin diffusion. In International Conference on Learning Representations, 2021.
- [118] Alexia Jolicoeur-Martineau, Rémi Piché-Taillefer, Ioannis Mitliagkas, and Remi Tachet des Combes. Adversarial score matching and improved sampling for image generation. In International Conference on Learning Representations, 2021.
- [119] Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021.
- [120] Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12413–12422, 2022.
- [121] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.
- [122] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2022.
- [123] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: a fast ode solver for diffusion probabilistic model sampling in around 10 steps. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pages 5775–5787, 2022.
- [124] Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diffusion implicit models. In International Conference on Learning Representations, 2023.
- [125] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In International Conference on Learning Representations, 2023.
- [126] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: a unified predictor-corrector framework for fast sampling of diffusion models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 49842–49869, 2023.
- [127] Shuchen Xue, Zhaoqiang Liu, Fei Chen, Shifeng Zhang, Tianyang Hu, Enze Xie, and Zhenguo Li. Accelerating diffusion sampling with optimized time steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8292–8301, 2024.
- [128] Daniel Watson, Jonathan Ho, Mohammad Norouzi, and William Chan. Learning to efficiently sample from diffusion probabilistic models. arXiv preprint arXiv:2106.03802, 2021.
- [129] Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In International Conference on Learning Representations, 2022.
- [130] Kexun Zhang, Xianjun Yang, William Yang Wang, and Lei Li. Redi: efficient learning-free diffusion inference via trajectory retrieval. In International Conference on Machine Learning, pages 41770– 41785, 2023.

- [131] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.
- [132] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.
- [133] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023.
- [134] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024.
- [135] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023.
- [136] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2022.
- [137] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for highquality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023.
- [138] Ying Fan and Kangwook Lee. Optimizing ddpm sampling with shortcut fine-tuning. In International Conference on Machine Learning, pages 9623–9639, 2023.
- [139] Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. arXiv preprint arXiv:2405.07510, 2024.
- [140] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.
- [141] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024.
- [142] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion gans. In International Conference on Learning Representations, 2022.
- [143] Yanwu Xu, Mingming Gong, Shaoan Xie, Wei Wei, Matthias Grundmann, Kayhan Batmanghelich, and Tingbo Hou. Semi-implicit denoising diffusion models (siddms). In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 17383–17394, 2023.
- [144] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-toimage generation via diffusion gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8196–8206, 2024.
- [145] Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process. arXiv preprint arXiv:2205.12524, 2022.
- [146] Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated diffusion probabilistic models. arXiv preprint arXiv:2202.09671, 2022.
- [147] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36, 2024.
- [148] Yang Zhao, Yanwu Xu, Zhisheng Xiao, Haolin Jia, and Tingbo Hou. Mobilediffusion: Instant text-toimage generation on mobile devices, 2024.
- [149] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Kai Li, and Song Han. Distrifusion: Distributed parallel inference for high-resolution diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7183–7193, June 2024.
- [150] Jiannan Wang, Jiarui Fang, Aoyu Li, and PengCheng Yang. Pipefusion: Displaced patch pipeline parallelism for inference of diffusion transformer models, 2024.

- [151] Zigeng Chen, Xinyin Ma, Gongfan Fang, Zhenxiong Tan, and Xinchao Wang. Asyncdiff: Parallelizing diffusion models by asynchronous denoising, 2024.
- [152] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021.
- [153] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [154] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
- [155] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [156] Zhiyuan Ma, Guoli Jia, and Bowen Zhou. Adapedit: Spatio-temporal guided adaptive editing algorithm for text-based continuity-sensitive image editing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4154–4161, 2024.
- [157] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR, 23:47–1, 2022.
- [158] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.
- [159] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.
- [160] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.
- [161] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024.
- [162] Black Forest Labs. Flux. https://blackforestlabs.ai, 2024.
- [163] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024.
- [164] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024.
- [165] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, April 2024.
- [166] Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv preprint arXiv:2405.18991, 2024.
- [167] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [168] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022.
- [169] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [170] Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. In International Conference on Machine Learning, pages 1692–1717. PMLR, 2023.

- [171] Zhiyuan Ma, Zhihuan Yu, Jianjun Li, and Bowen Zhou. Lmd: faster image reconstruction with latent masking diffusion. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4145–4153, 2024.
- [172] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.
- [173] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.
- [174] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241, 2015.
- [175] Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517, 2017.
- [176] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [177] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [178] A Radford. Improving language understanding by generative pre-training. 2018.
- [179] Zhiyuan Ma, Jianjun Li, Guohui Li, and Yongjing Cheng. Glaf: global-to-local aggregation and fission network for semantic level fact verification. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1801–1812, 2022.
- [180] Zhiyuan Ma, Jianjun Li, Zezheng Zhang, Guohui Li, and Yongjing Cheng. Intention reasoning network for multi-domain end-to-end task-oriented dialogue. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2273–2285, 2021.
- [181] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.
- [182] Zhiyuan Ma, Jianjun Li, Guohui Li, and Kaiyan Huang. Cmal: A novel cross-modal associative learning framework for vision-language pre-training. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4515–4524, 2022.
- [183] Zhiyuan Ma, Jianjun Li, Guohui Li, and Yongjing Cheng. Unitranser: A unified transformer semantic representation framework for multimodal task-oriented dialog system. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 103–114, 2022.
- [184] Zhiyuan Ma, Zhihuan Yu, Jianjun Li, and Guohui Li. Hybridprompt: bridging language models and human priors in prompt tuning for visual question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 13371–13379, 2023.
- [185] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pages 1691–1703. PMLR, 2020.
- [186] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pages 8821–8831, 2021.
- [187] Albert Gu. Modeling Sequences with Structured State Spaces. Stanford University, 2023.
- [188] Biqing Qi, Yang Luo, Junqi Gao, Pengfei Li, Kai Tian, Zhiyuan Ma, and Bowen Zhou. Exploring adversarial robustness of deep state space models. arXiv preprint arXiv:2406.05532, 2024.
- [189] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. Hippo: Recurrent memory with optimal polynomial projections. Advances in neural information processing systems, 33:1474–1487, 2020.

- [190] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.
- [191] Ankit Gupta, Albert Gu, and Jonathan Berant. Diagonal state spaces are as effective as structured state spaces. Advances in Neural Information Processing Systems, 35:22982–22994, 2022.
- [192] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.
- [193] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, and Junshi Huang. Diffusion-rwkv: Scaling rwkv-like architectures for diffusion models. arXiv preprint arXiv:2404.04478, 2024.
- [194] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.
- [195] Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, and Xinggang Wang. Dig: Scalable and efficient diffusion models with gated linear attention. arXiv preprint arXiv:2405.18428, 2024.
- [196] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. arXiv preprint arXiv:2312.06635, 2023.
- [197] Zhiyu Tan, Mengping Yang, Luozheng Qin, Hao Yang, Ye Qian, Qiang Zhou, Cheng Zhang, and Hao Li. An empirical study and analysis of text-to-image generation using large language model-powered textual representation. arXiv preprint arXiv:2405.12914, 2024.
- [198] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [199] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, pages 4171–4186, 2019.
- [200] Fulong Ye, Guang Liu, Xinya Wu, and Ledell Wu. Altdiffusion: A multilingual text-to-image diffusion model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 6648–6656, 2024.
- [201] Chengyu Wang, Zhongjie Duan, Bingyan Liu, Xinyi Zou, Cen Chen, Kui Jia, and Jun Huang. Paidiffusion: Constructing and serving a family of open chinese diffusion models for text-to-image synthesis on the cloud. arXiv preprint arXiv:2309.05534, 2023.
- [202] Xiaojun Wu, Dixiang Zhang, Ruyi Gan, Junyu Lu, Ziwei Wu, Renliang Sun, Jiaxing Zhang, Pingjian Zhang, and Yan Song. Taiyi-diffusion-xl: Advancing bilingual text-to-image generation with large vision-language model support. arXiv preprint arXiv:2401.14688, 2024.
- [203] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.
- [204] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. arXiv preprint arXiv:2203.06904, 2022.
- [205] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, et al. I2v-adapter: A general image-to-video adapter for diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024.
- [206] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [207] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.
- [208] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

- [209] Subhadeep Koley, Ayan Kumar Bhunia, Deeptanshu Sekhri, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, and Yi-Zhe Song. It’s all about your sketch: Democratising sketch control in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7204–7214, 2024.
- [210] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Efficient parametrization of multi-domain deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8119–8127, 2018.
- [211] Chao Xu, Yang Liu, Jiazheng Xing, Weida Wang, Mingze Sun, Jun Dan, Tianxin Huang, Siyuan Li, Zhi-Qi Cheng, Ying Tai, et al. Facechain-imagineid: Freely crafting high-fidelity diverse talking faces from disentangled audio. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1292–1302, 2024.
- [212] Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv preprint arXiv:2012.13255, 2020.
- [213] Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. arXiv preprint arXiv:1804.08838, 2018.
- [214] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2025.
- [215] James Seale Smith, Yen-Chang Hsu, Lingyu Zhang, Ting Hua, Zsolt Kira, Yilin Shen, and Hongxia Jin. Continual diffusion: Continual customization of text-to-image diffusion with c-lora. arXiv preprint arXiv:2304.06027, 2023.
- [216] Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, and Anand Bhattad. Intrinsic lora: A generalist approach for discovering knowledge in generative models. In Synthetic Data for Computer Vision Workshop@ CVPR 2024.
- [217] Jiayi Guo, Xingqian Xu, Yifan Pu, Zanlin Ni, Chaofei Wang, Manushree Vasu, Shiji Song, Gao Huang, and Humphrey Shi. Smooth diffusion: Crafting smooth latent spaces in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7548–7558, 2024.
- [218] Kaiwen Zhang, Yifan Zhou, Xudong Xu, Bo Dai, and Xingang Pan. Diffmorpher: Unleashing the capability of diffusion models for image morphing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7912–7921, 2024.
- [219] Yufan Deng, Ruida Wang, Yuhao Zhang, Yu-Wing Tai, and Chi-Keung Tang. Dragvideo: Interactive drag-style video editing. arXiv preprint arXiv:2312.02216, 2023.
- [220] Yue Ma, Xiaodong Cun, Yingqing He, Chenyang Qi, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Magicstick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047, 2023.
- [221] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024.
- [222] Kai Yu, Jinlin Liu, Mengyang Feng, Miaomiao Cui, and Xuansong Xie. Boosting3d: High-fidelity image-to-3d by boosting 2d diffusion prior to 3d prior with progressive learning. arXiv preprint arXiv:2311.13617, 2023.
- [223] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Hyeonsu Kim, Jaehoon Ko, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, and Seungryong Kim. Let 2d diffusion model know 3d-consistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937, 2023.
- [224] Tianyu Huang, Yihan Zeng, Zhilu Zhang, Wan Xu, Hang Xu, Songcen Xu, Rynson WH Lau, and Wangmeng Zuo. Dreamcontrol: Control-based text-to-3d generation with 3d self-prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5364–5373, 2024.
- [225] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-apic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:36652–36663, 2023.

- [226] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning textto-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.
- [227] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8941–8951, 2024.
- [228] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.
- [229] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024.
- [230] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-toimage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6527–6536, 2024.
- [231] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identitypreserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.
- [232] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.
- [233] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2(3), 2023.
- [234] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8543–8552, 2024.
- [235] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.
- [236] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.
- [237] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [238] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Genie: higher-order denoising diffusion solvers. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pages 30150–30166, 2022.
- [239] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.
- [240] Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu Chang. Uncovering the disentanglement capability in text-to-image diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1900– 1910, 2023.
- [241] Cristian Buciluˇa, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 535–541, 2006.
- [242] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
- [243] Arash Vahdat and Jan Kautz. Nvae: a deep hierarchical variational autoencoder. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pages 19667–19679, 2020.

- [244] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.
- [245] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [246] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar, et al. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021.
- [247] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In International Conference on Learning Representations, 2024.
- [248] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 9156–9172, 2023.
- [249] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [250] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [251] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clip-guided domain adaptation of image generators. ACM Transactions on Graphics, 41(4):1–13, 2022.
- [252] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [253] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International conference on machine learning, pages 3481–3490, 2018.
- [254] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1972–1981, 2023.
- [255] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: accurate posttraining quantization for diffusion models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 13237–13249, 2023.
- [256] Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated diffusion probabilistic models and diffusion-based adversarial auto-encoders. In The Eleventh International Conference on Learning Representations, 2023.
- [257] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.
- [258] Yu-Hui Chen, Raman Sarokin, Juhyun Lee, Jiuqiang Tang, Chuo-Ling Chang, Andrei Kulik, and Matthias Grundmann. Speed is all you need: On-device acceleration of large diffusion models via gpu-aware optimizations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4651–4655, 2023.
- [259] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [260] Ke Sun, Jian Cao, Qi Wang, Linrui Tian, Xindi Zhang, Lian Zhuo, Bang Zhang, Liefeng Bo, Wenbo Zhou, Weiming Zhang, et al. Outfitanyone: Ultra-high quality virtual try-on for any clothing and any person. arXiv preprint arXiv:2407.16224, 2024.
- [261] Luyang Zhu, Yingwei Li, Nan Liu, Hao Peng, Dawei Yang, and Ira Kemelmacher-Shlizerman. M&m vto: Multi-garment virtual try-on and editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1346–1356, 2024.

- [262] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. Deadiff: An efficient stylization diffusion model with disentangled representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8693–8702, 2024.
- [263] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421, 2023.
- [264] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023.
- [265] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Delving deep into diffusion transformers for image and video generation. arXiv preprint arXiv:2312.04557, 2023.
- [266] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023.
- [267] Zixun Fang, Wei Zhai, Aimin Su, Hongliang Song, Kai Zhu, Mao Wang, Yu Chen, Zhiheng Liu, Yang Cao, and Zheng-Jun Zha. Vivid: Video virtual try-on using diffusion models. arXiv preprint arXiv:2405.11794, 2024.
- [268] Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Motioneditor: Editing video motion via content-aware diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7882–7891, 2024.
- [269] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flow-guided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023.
- [270] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023.
- [271] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023.
- [272] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023.
- [273] Bowen Zhang, Yiji Cheng, Chunyu Wang, Ting Zhang, Jiaolong Yang, Yansong Tang, Feng Zhao, Dong Chen, and Baining Guo. Rodinhd: High-fidelity 3d avatar generation with diffusion models. arXiv preprint arXiv:2407.06938, 2024.
- [274] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024.
- [275] Yuming Gu, Hongyi Xu, You Xie, Guoxian Song, Yichun Shi, Di Chang, Jing Yang, and Linjie Luo. Diffportrait3d: Controllable diffusion for zero-shot portrait view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10456–10465, 2024.
- [276] Kira Prabhu, Jane Wu, Lynn Tsai, Peter Hedman, Dan B Goldman, Ben Poole, and Michael Broxton. Inpaint3d: 3d scene content generation using 2d inpainting diffusion. arXiv preprint arXiv:2312.03869, 2023.
- [277] Yu-Ying Yeh, Jia-Bin Huang, Changil Kim, Lei Xiao, Thu Nguyen-Phuoc, Numair Khan, Cheng Zhang, Manmohan Chandraker, Carl S Marshall, Zhao Dong, et al. Texturedreamer: Image-guided texture synthesis through geometry-aware diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4304–4314, 2024.
- [278] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.

- [279] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A A Kohl, Andrew J Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583–589, 2021.
- [280] Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock: Diffusion steps, twists, and turns for molecular docking. arXiv preprint arXiv:2210.01776, 2022.
- [281] Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. De novo design of protein structure and function with rfdiffusion. Nature, 620(7976):1089–1100, 2023.
- [282] Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, and Jianzhu Ma. Antigen-specific antibody design and optimization with diffusion-based generative models for protein structures. Advances in Neural Information Processing Systems, 35:9754–9767, 2022.
- [283] Zhenbin Wang, Lei Zhang, Lituan Wang, and Zhenwei Zhang. Soft masked mamba diffusion model for ct to mri conversion. arXiv preprint arXiv:2406.15910, 2024.
- [284] Tiange Xiang, Mahmut Yurt, Ali B Syed, Kawin Setsompop, and Akshay Chaudhari. Ddm2: Selfsupervised diffusion mri denoising with generative diffusion models. arXiv preprint arXiv:2302.03018, 2023.
- [285] Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. arXiv preprint arXiv:2111.08005, 2021.
- [286] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. Advances in Neural Information Processing Systems, 36, 2024.
- [287] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems, 36, 2024.
- [288] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069– 8078, 2024.
- [289] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized textto-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024.
- [290] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023.
- [291] Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.
- [292] Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-to-image model on a face. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10, 2023.
- [293] Zhuowei Chen, Shancheng Fang, Wei Liu, Qian He, Mengqi Huang, Yongdong Zhang, and Zhendong Mao. Dreamidentity: Improved editability for efficient face-identity preserved image generation. arXiv preprint arXiv:2307.00300, 2023.
- [294] Yuxuan Yan, Chi Zhang, Rui Wang, Yichao Zhou, Gege Zhang, Pei Cheng, Gang Yu, and Bin Fu. Facestudio: Put your face everywhere in seconds. arXiv preprint arXiv:2312.02663, 2023.
- [295] Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. Portraitbooth: A versatile portrait model for fast identity-preserved personalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27080–27090, 2024.

- [296] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters. Advances in Neural Information Processing Systems, 36, 2024.
- [297] Yuanzhi Zhu, Zhaohai Li, Tianwei Wang, Mengchao He, and Cong Yao. Conditional text image generation with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14235–14245, 2023.
- [298] Jian Ma, Mingjun Zhao, Chen Chen, Ruichen Wang, Di Niu, Haonan Lu, and Xiaodong Lin. Glyphdraw: Learning to draw chinese characters in image synthesis models coherently. arXiv preprint arXiv:2303.17870, 2, 2023.
- [299] Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, and Kai Chen. Glyphcontrol: Glyph conditional control for visual text generation. Advances in Neural Information Processing Systems, 36, 2024.
- [300] Lingjun Zhang, Xinyuan Chen, Yaohui Wang, Yue Lu, and Yu Qiao. Brush your text: Synthesize any scene text on images via diffusion model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 7215–7223, 2024.
- [301] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. arXiv preprint arXiv:2311.03054, 2023.
- [302] Hsiao Yuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and Qing Zhang. Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6018–6026, 2023.
- [303] Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. Synthesizing coherent story with autoregressive latent diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2920–2930, 2024.
- [304] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm-open-ended visual storytelling via latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6190–6200, 2024.
- [305] Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Yingqing He, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, et al. Talecrafter: Interactive story visualization with multiple characters. arXiv preprint arXiv:2305.18247, 2023.
- [306] Hyeonho Jeong, Gihyun Kwon, and Jong Chul Ye. Zero-shot generation of coherent storybook from plain text story using diffusion models. arXiv preprint arXiv:2302.03900, 2023.
- [307] Fei Shen, Hu Ye, Sibo Liu, Jun Zhang, Cong Wang, Xiao Han, and Wei Yang. Boosting consistency in story visualization with rich-contextual conditional diffusion models. arXiv preprint arXiv:2407.02482, 2024.
- [308] Xiao He, Mingrui Zhu, Dongxin Chen, Nannan Wang, and Xinbo Gao. Diff-privacy: Diffusion-based face privacy protection. IEEE Transactions on Circuits and Systems for Video Technology, 2024.
- [309] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2426–2436, 2023.
- [310] Yingqian Cui, Jie Ren, Han Xu, Pengfei He, Hui Liu, Lichao Sun, Yue Xing, and Jiliang Tang. Diffusionshield: A watermark for copyright protection against generative diffusion models. arXiv preprint arXiv:2306.04642, 2023.
- [311] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22691–22702, 2023.
- [312] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In 32nd USENIX Security Symposium (USENIX Security 23), pages 5253–5270, 2023.
- [313] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. Advances in Neural Information Processing Systems, 36:47783–47803, 2023.

- [314] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6048–6058, 2023.
- [315] Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22466–22477, 2023.
- [316] Zhiyuan Ma, Guoli Jia, Biqing Qi, and Bowen Zhou. Safe-sd: Safe and traceable stable diffusion with text prompt trigger for invisible generative watermarking. arXiv preprint arXiv:2407.13188, 2024.
- [317] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.
- [318] Qin Guo and Tianwei Lin. Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6986–6996, 2024.
- [319] Tuhin Chakrabarty, Kanishk Singh, Arkadiy Saakyan, and Smaranda Muresan. Learning to follow object-centric image editing instructions faithfully. arXiv preprint arXiv:2310.19145, 2023.
- [320] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, et al. Instructdiffusion: A generalist modeling interface for vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12709– 12720, 2024.
- [321] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.
- [322] Ahmet Burak Yildirim, Vedat Baday, Erkut Erdem, Aykut Erdem, and Aysegul Dundar. Inst-inpaint: Instructing to remove objects with diffusion models. arXiv preprint arXiv:2304.03246, 2023.
- [323] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9026– 9036, 2024.
- [324] Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike, et al. Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation. Advances in Neural Information Processing Systems, 36, 2024.
- [325] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023.
- [326] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8362–8371, 2024.
- [327] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2426–2435, 2022.
- [328] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. arXiv preprint arXiv:2210.10960, 2022.
- [329] Nisha Huang, Yuxin Zhang, Fan Tang, Chongyang Ma, Haibin Huang, Weiming Dong, and Changsheng Xu. Diffstyler: Controllable dual diffusion for text-driven image stylization. IEEE Transactions on Neural Networks and Learning Systems, 2024.
- [330] Zhizhong Wang, Lei Zhao, and Wei Xing. Stylediffusion: Controllable disentangled style transfer via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7677–7689, 2023.

- [331] Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, and Joyce Chai. Cyclenet: Rethinking cycle consistency in text-guided diffusion for image manipulation. Advances in Neural Information Processing Systems, 36, 2024.
- [332] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10619–10629, 2022.
- [333] Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609–3623, 2022.
- [334] Fei Yang, Shiqi Yang, Muhammad Atif Butt, Joost van de Weijer, et al. Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing. Advances in Neural Information Processing Systems, 36:26291–26303, 2023.
- [335] Wenkai Dong, Song Xue, Xiaoyue Duan, and Shumin Han. Prompt tuning inversion for text-driven image editing using diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7430–7440, 2023.
- [336] Yujun Shi, Chuhui Xue, Jun Hao Liew, Jiachun Pan, Hanshu Yan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8839– 8849, 2024.
- [337] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2328–2337, 2023.
- [338] Hyelin Nam, Gihyun Kwon, Geon Yeong Park, and Jong Chul Ye. Contrastive denoising score for textguided latent diffusion image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9192–9201, 2024.
- [339] Dani Valevski, Matan Kalman, Yossi Matias, and Yaniv Leviathan. Unitune: Text-driven image editing by fine tuning an image generation model on a single image. arXiv preprint arXiv:2210.09477, 2(3):5, 2022.
- [340] Jooyoung Choi, Yunjey Choi, Yunji Kim, Junho Kim, and Sungroh Yoon. Custom-edit: Text-guided image editing with customized diffusion models. arXiv preprint arXiv:2305.15779, 2023.
- [341] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N Metaxas, and Jian Ren. Sine: Single image editing with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6027–6037, 2023.
- [342] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zeroshot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023.
- [343] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22560–22570, 2023.
- [344] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for textdriven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023.
- [345] Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon: Diffusion-based training-free cross-domain image composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2294–2305, 2023.
- [346] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel Cohen-Or. Localizing objectlevel shape variations with text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23051–23061, 2023.
- [347] Hyunsoo Lee, Minsoo Kang, and Bohyung Han. Conditional score guidance for text-driven image-toimage translation. Advances in Neural Information Processing Systems, 36, 2024.
- [348] Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, and Jong Chul Ye. Energy-based cross attention for bayesian context update in text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

- [349] Dong Huk Park, Grace Luo, Clayton Toste, Samaneh Azadi, Xihui Liu, Maka Karalashvili, Anna Rohrbach, and Trevor Darrell. Shape-guided diffusion with inside-outside attention. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4198–4207, 2024.
- [350] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469–12478, 2024.
- [351] Shen Nie, Hanzhong Allan Guo, Cheng Lu, Yuhao Zhou, Chenyu Zheng, and Chongxuan Li. The blessing of randomness: Sde beats ode in general diffusion-based image editing. arXiv preprint arXiv:2311.01410, 2023.
- [352] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinário Passos. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8861–8870, 2024.
- [353] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023.
- [354] Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Anastasis Stathopoulos, Xiaoxiao He, Yuxiao Chen, et al. Proxedit: Improving tuning-free real image editing with proximal guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4291–4301, 2024.
- [355] Jing Zhao, Heliang Zheng, Chaoyue Wang, Long Lan, Wanrong Huang, and Wenjing Yang. Nulltext guidance in diffusion models is secretly a cartoon-style creator. In Proceedings of the 31st ACM International Conference on Multimedia, pages 5143–5152, 2023.
- [356] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22532–22541, 2023.
- [357] Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. Effective real image editing with accelerated iterative diffusion inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15912–15921, 2023.
- [358] Chen Henry Wu and Fernando De la Torre. A latent space of stochastic diffusion models for zero-shot image editing and guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7378–7387, 2023.
- [359] Jaeseok Jeong, Mingi Kwon, and Youngjung Uh. Training-free content injection using h-space in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5151–5161, 2024.
- [360] Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems, 36:16222– 16239, 2023.
- [361] Zihao Yu, Haoyang Li, Fangcheng Fu, Xupeng Miao, and Bin Cui. Fisedit: Accelerating text-to-image editing via cache-enabled sparse diffusion inference. ArXiv, abs/2305.17423, 2023.
- [362] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM transactions on graphics (TOG), 42(4):1–11, 2023.
- [363] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.
- [364] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022.
- [365] Shanglin Li, Bohan Zeng, Yutang Feng, Sicheng Gao, Xiuhui Liu, Jiaming Liu, Lin Li, Xu Tang, Yao Hu, Jianzhuang Liu, et al. Zone: Zero-shot instruction-guided local editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6254–6263, 2024.
- [366] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023.

- [367] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.
- [368] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8580–8589, 2023.
- [369] Alberto Baldrati, Davide Morelli, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Multimodal garment designer: Human-centric latent diffusion models for fashion image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23393–23402, 2023.
- [370] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7599–7607, 2023.
- [371] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8176–8185, 2024.
- [372] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779, 2024.
- [373] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4606–4615, 2023.
- [374] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391, 2023.
- [375] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision, pages 204–219. Springer, 2022.
- [376] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6593–6602, 2024.
- [377] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10146–10156, 2023.
- [378] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983, 2023.
- [379] Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, and Ping Luo. Styleadapter: A single-pass lora-free model for stylized image generation. arXiv preprint arXiv:2309.01770, 2023.
- [380] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE TPAMI, 2022.
- [381] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing, 479:47–59, 2022.
- [382] Shuyao Shang, Zhengyang Shan, Guangxing Liu, LunQian Wang, XingHua Wang, Zekai Zhang, and Jinglin Zhang. Resdiff: Combining cnn and diffusion model for image super-resolution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 8975–8983, 2024.
- [383] Hshmat Sahak, Daniel Watson, Chitwan Saharia, and David Fleet. Denoising diffusion probabilistic models for robust image super-resolution in the wild. arXiv preprint arXiv:2302.07864, 2023.
- [384] Sicheng Gao, Xuhui Liu, Bohan Zeng, Sheng Xu, Yanjing Li, Xiaoyan Luo, Jianzhuang Liu, Xiantong Zhen, and Baochang Zhang. Implicit diffusion models for continuous super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10021–10030, 2023.

- [385] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool. Diffir: Efficient diffusion model for image restoration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13095–13105, 2023.
- [386] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. arXiv preprint arXiv:2303.11435, 2023.
- [387] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021.
- [388] Bahjat Kawar, Gregory Vaksman, and Michael Elad. Snips: Solving noisy inverse problems stochastically. Advances in Neural Information Processing Systems, 34:21757–21769, 2021.
- [389] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.
- [390] Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative diffusion prior for unified image restoration and enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9935–9946, 2023.
- [391] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023.
- [392] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1219–1229, 2023.
- [393] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion nullspace model. arXiv preprint arXiv:2212.00490, 2022.
- [394] Zheng Chen, Yulun Zhang, Ding Liu, Jinjin Gu, Linghe Kong, Xin Yuan, et al. Hierarchical integration diffusion model for realistic image deblurring. Advances in neural information processing systems, 36, 2024.
- [395] Mengwei Ren, Mauricio Delbracio, Hossein Talebi, Guido Gerig, and Peyman Milanfar. Image deblurring with domain generalizable diffusion models. arXiv preprint arXiv:2212.01789, 1, 2022.
- [396] Jay Whang, Mauricio Delbracio, Hossein Talebi, Chitwan Saharia, Alexandros G Dimakis, and Peyman Milanfar. Deblurring via stochastic refinement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16293–16303, 2022.
- [397] Naoki Murata, Koichi Saito, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration. In International conference on machine learning, pages 25501–

25522. PMLR, 2023.

- [398] Yuan Yuan, Siyuan Liu, Jiawei Zhang, Yongbing Zhang, Chao Dong, and Liang Lin. Unsupervised image super-resolution using cycle-in-cycle generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 701–710, 2018.
- [399] Berthy T Feng, Jamie Smith, Michael Rubinstein, Huiwen Chang, Katherine L Bouman, and William T Freeman. Score-based diffusion models as principled priors for inverse imaging. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10520–10531, 2023.
- [400] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH, pages 1–10, 2022.
- [401] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In CVPR, pages 11461–11471, 2022.
- [402] Guanhua Zhang, Jiabao Ji, Yang Zhang, Mo Yu, Tommi S Jaakkola, and Shiyu Chang. Towards coherent image inpainting using denoising diffusion implicit models. 2023.
- [403] Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. arXiv preprint arXiv:2305.04391, 2023.

- [404] Simon Welker, Henry N Chapman, and Timo Gerkmann. Driftrec: Adapting diffusion models to blind image restoration tasks. arXiv preprint arXiv:2211.06757, 2, 2022.
- [405] Ziwei Luo, Fredrik K Gustafsson, Zheng Zhao, Jens Sjölund, and Thomas B Schön. Refusion: Enabling large-size realistic image restoration with latent-space diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1680–1691, 2023.
- [406] Yeying Jin, Wenhan Yang, Wei Ye, Yuan Yuan, and Robby T Tan. Shadowdiffusion: Diffusionbased shadow removal using classifier-driven attention and structure preservation. arXiv preprint arXiv:2211.08089, 2, 2022.
- [407] Lanqing Guo, Chong Wang, Wenhan Yang, Siyu Huang, Yufei Wang, Hanspeter Pfister, and Bihan Wen. Shadowdiffusion: When degradation prior meets diffusion model for shadow removal. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14049–14058, 2023.
- [408] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563– 22575, 2023.
- [409] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [410] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In The Eleventh International Conference on Learning Representations, 2022.
- [411] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023.
- [412] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.
- [413] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024.
- [414] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [415] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. arXiv preprint arXiv:2403.14781, 2024.
- [416] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024.
- [417] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153– 8163, 2024.
- [418] Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for referring human dance generation in real world. arXiv e-prints, pages arXiv–2307, 2023.
- [419] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4117–4125, 2024.
- [420] Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9212–9221, 2024.

- [421] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Luc Van Gool, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024.
- [422] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive-generating expressive portrait videos with audio2video diffusion model under weak conditions. arXiv preprint arXiv:2402.17485, 2024.
- [423] Yujin Jeong, Wonjeong Ryoo, Seunghyun Lee, Dabin Seo, Wonmin Byeon, Sangpil Kim, and Jinkyu Kim. The power of sound (tpos): Audio reactive video generation with stable diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7822–7832, 2023.
- [424] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940, 2023.
- [425] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE Transactions on Visualization and Computer Graphics, 2024.
- [426] David Junhao Zhang, Dongxu Li, Hung Le, Mike Zheng Shou, Caiming Xiong, and Doyen Sahoo. Moonshot: Towards controllable video generation and editing with multimodal conditions. arXiv preprint arXiv:2401.01827, 2024.
- [427] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. Advances in Neural Information Processing Systems, 36, 2024.
- [428] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023.
- [429] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
- [430] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with user-directed camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024.
- [431] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.
- [432] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:2310.08465, 2023.
- [433] Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, Tsung-Yi Lin, and Ming-Hsuan Yang. Motionconditioned diffusion model for controllable video synthesis. arXiv preprint arXiv:2304.14404, 2023.
- [434] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. arXiv preprint arXiv:2303.01418, 2023.
- [435] Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Xiao Ma, Liang Pan, and Ziwei Liu. Insactor: Instructiondriven physics-based characters. Advances in Neural Information Processing Systems, 36, 2024.
- [436] You Xie, Hongyi Xu, Guoxian Song, Chao Wang, Yichun Shi, and Linjie Luo. X-portrait: Expressive portrait animation with hierarchical motion attention. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
- [437] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, and Jiwen Lu. Drivedreamer: Towards realworld-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023.
- [438] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024.

- [439] Yuqing Wen, Yucheng Zhao, Yingfei Liu, Fan Jia, Yanhui Wang, Chong Luo, Chi Zhang, Tiancai Wang, Xiaoyan Sun, and Xiangyu Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6902–6912, 2024.
- [440] Ruiyuan Gao, Kai Chen, Enze Xie, Lanqing Hong, Zhenguo Li, Dit-Yan Yeung, and Qiang Xu. Magicdrive: Street view generation with diverse 3d geometry control. arXiv preprint arXiv:2310.02601, 2023.
- [441] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scene video generation with latent diffusion model. arXiv preprint arXiv:2310.07771, 2023.
- [442] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023.
- [443] Elia Peruzzo, Vidit Goel, Dejia Xu, Xingqian Xu, Yifan Jiang, Zhangyang Wang, Humphrey Shi, and Nicu Sebe. Vase: Object-centric appearance and shape manipulation of real videos. arXiv preprint arXiv:2401.02473, 2024.
- [444] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. Eva: Zero-shot accurate attributes and multiobject video editing. arXiv preprint arXiv:2403.16111, 2024.
- [445] Bojia Zi, Shihao Zhao, Xianbiao Qi, Jianan Wang, Yukai Shi, Qianyu Chen, Bin Liang, Kam-Fai Wong, and Lei Zhang. Cococo: Improving text-guided video inpainting for better consistency, controllability and compatibility. arXiv preprint arXiv:2403.12035, 2024.
- [446] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M Rehg, and Pinar Yanardag. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6507–6516, 2024.
- [447] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. arXiv preprint arXiv:2405.13865, 2024.
- [448] Yi Zuo, Lingling Li, Licheng Jiao, Fang Liu, Xu Liu, Wenping Ma, Shuyuan Yang, and Yuwei Guo. Edit-your-motion: Space-time diffusion decoupling learning for video motion editing. arXiv preprint arXiv:2405.04496, 2024.
- [449] Gihyun Kwon, Jangho Park, and Jong Chul Ye. Unified editing of panorama, 3d scenes, and videos through disentangled self-attention injection. arXiv preprint arXiv:2405.16823, 2024.
- [450] Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. Uniedit: A unified tuning-free framework for video motion and appearance editing. arXiv preprint arXiv:2402.13185, 2024.
- [451] Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng. Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749, 2023.
- [452] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023.
- [453] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023.
- [454] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970–9980, 2024.
- [455] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22246–22256, 2023.
- [456] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it3d: High-fidelity 3d creation from a single image with diffusion prior. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22819–22829, 2023.

- [457] Titas Anciukeviˇcius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J Mitra, and Paul Guerrero. Renderdiffusion: Image diffusion for 3d reconstruction, inpainting and generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12608– 12618, 2023.
- [458] Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, and Sanja Fidler. Neuralfield-ldm: Scene generation with hierarchical latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8496–8506, 2023.
- [459] Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, and SongChun Zhu. Diffusion-based generation, optimization, and planning in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16750–16761, 2023.
- [460] J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20875–20886, 2023.
- [461] Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam. Commonscenes: Generating commonsense 3d indoor scenes with scene graphs. Advances in Neural Information Processing Systems, 36, 2024.
- [462] Jiabao Lei, Jiapeng Tang, and Kui Jia. Rgbd2: Generative scene synthesis via incremental view inpainting using rgbd diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8422–8434, 2023.
- [463] Ryan Po and Gordon Wetzstein. Compositional 3d scene generation using locally conditioned diffusion. In 2024 International Conference on 3D Vision (3DV), pages 651–663. IEEE, 2024.
- [464] Ruiyuan Gao, Kai Chen, Zhihao Li, Lanqing Hong, Zhenguo Li, and Qiang Xu. Magicdrive3d: Controllable 3d generation for any-view rendering in street scenes. arXiv preprint arXiv:2405.14475, 2024.
- [465] Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16010–16021, 2023.
- [466] Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui. Interdiff: Generating 3d human-object interactions with physics-informed diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14928–14940, 2023.
- [467] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36, 2024.
- [468] Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, and Siyu Tang. Guided motion diffusion for controllable human motion synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2151–2162, 2023.
- [469] Han Liang, Wenqian Zhang, Wenxuan Li, Jingyi Yu, and Lan Xu. Intergen: Diffusion-based multihuman motion generation under complex interactions. International Journal of Computer Vision, pages 1–21, 2024.
- [470] Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. arXiv preprint arXiv:2208.15001, 2022.
- [471] German Barquero, Sergio Escalera, and Cristina Palmero. Belfusion: Latent diffusion for behaviordriven human motion prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2317–2327, 2023.
- [472] Yuming Du, Robin Kips, Albert Pumarola, Sebastian Starke, Ali Thabet, and Artsiom Sanakoyeu. Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 481– 490, 2023.
- [473] Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18000–18010, 2023.

- [474] Dong Wei, Huaijiang Sun, Bin Li, Jianfeng Lu, Weiqing Li, Xiaoning Sun, and Shengxiang Hu. Human joint kinematics diffusion-refinement for stochastic motion prediction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 6110–6118, 2023.
- [475] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4563–4573, 2023.
- [476] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and Kwan-Yee K Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 958–968, 2024.
- [477] Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, and KwanYee K Wong. Headsculpt: Crafting 3d head avatars with text. Advances in Neural Information Processing Systems, 36, 2024.
- [478] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. Dreamwaltz: Make a scene with complex 3d animatable avatars. Advances in Neural Information Processing Systems, 36, 2024.
- [479] Fan Zhang, Naye Ji, Fuxing Gao, and Yongping Li. Diffmotion: Speech-driven gesture synthesis using denoising diffusion model. In International Conference on Multimedia Modeling, pages 231–242. Springer, 2023.
- [480] Junming Chen, Yunfei Liu, Jianan Wang, Ailing Zeng, Yu Li, and Qifeng Chen. Diffsheg: A diffusionbased approach for real-time speech-driven holistic 3d expression and gesture generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7352–7361, 2024.
- [481] Lingting Zhu, Xian Liu, Xuanyu Liu, Rui Qian, Ziwei Liu, and Lequan Yu. Taming diffusion models for audio-driven co-speech gesture generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10544–10553, 2023.
- [482] Simon Alexanderson, Rajmund Nagy, Jonas Beskow, and Gustav Eje Henter. Listen, denoise, action! audio-driven motion synthesis with diffusion models. ACM Transactions on Graphics (TOG), 42(4):1– 20, 2023.
- [483] Qing Lyu and Ge Wang. Conversion between ct and mri images using diffusion and score-matching models. arXiv preprint arXiv:2209.12104, 2022.
- [484] Xiangxi Meng, Yuning Gu, Yongsheng Pan, Nizhuan Wang, Peng Xue, Mengkang Lu, Xuming He, Yiqiang Zhan, and Dinggang Shen. A novel unified conditional score-based generative framework for multi-modal medical image completion. arXiv preprint arXiv:2207.03430, 2022.
- [485] Muzaffer Özbey, Onat Dalmaz, Salman UH Dar, Hasan A Bedel, S¸aban Özturk, Alper Güngör, and Tolga Çukur. Unsupervised medical image translation with adversarial diffusion models. IEEE Transactions on Medical Imaging, 2023.
- [486] Puria Azadi Moghadam, Sanne Van Dalen, Karina C Martin, Jochen Lennerz, Stephen Yip, Hossein Farahani, and Ali Bashashati. A morphology focused diffusion probabilistic model for synthesis of histopathology images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2000–2009, 2023.
- [487] Walter HL Pinaya, Petru-Daniel Tudosiu, Jessica Dafflon, Pedro F Da Costa, Virginia Fernandez, Parashkev Nachev, Sebastien Ourselin, and M Jorge Cardoso. Brain imaging generation with latent diffusion models. In MICCAI Workshop on Deep Generative Models, pages 117–126. Springer, 2022.
- [488] Zolnamar Dorjsembe, Sodtavilan Odonchimed, and Furen Xiao. Three-dimensional medical image synthesis with denoising diffusion probabilistic models. In Medical Imaging with Deep Learning, 2022.
- [489] Boah Kim and Jong Chul Ye. Diffusion deformable model for 4d temporal medical image generation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 539–548. Springer, 2022.
- [490] Kai Packhäuser, Lukas Folle, Florian Thamm, and Andreas Maier. Generation of anonymous chest radiographs using latent diffusion models for training thoracic abnormality classification systems. In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pages 1–5. IEEE, 2023.

- [491] Lan Jiang, Ye Mao, Xiangfeng Wang, Xi Chen, and Chao Li. Cola-diff: Conditional latent diffusion model for multi-modal mri synthesis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 398–408. Springer, 2023.
- [492] Dominik JE Waibel, Ernst Röell, Bastian Rieck, Raja Giryes, and Carsten Marr. A diffusion model predicts 3d shapes from 2d microscopy images. In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pages 1–5. IEEE, 2023.
- [493] Lemuel Puglisi, Daniel C Alexander, and Daniele Ravì. Enhancing spatiotemporal disease progression models via latent diffusion and prior knowledge. arXiv preprint arXiv:2405.03328, 2024.
- [494] Yijun Yang, Huazhu Fu, Angelica I Aviles-Rivero, Carola-Bibiane Schönlieb, and Lei Zhu. Diffmic: Dual-guidance diffusion network for medical image classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 95–105. Springer, 2023.
- [495] Julia Wolleb, Florentin Bieder, Robin Sandkühler, and Philippe C Cattin. Diffusion models for medical anomaly detection. In International Conference on Medical image computing and computer-assisted intervention, pages 35–45. Springer, 2022.
- [496] Julian Wyatt, Adam Leach, Sebastian M Schmon, and Chris G Willcocks. Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 650–656, 2022.
- [497] Pedro Sanchez, Antanas Kascenas, Xiao Liu, Alison Q O’Neil, and Sotirios A Tsaftaris. What is healthy? generative counterfactual diffusion for lesion localization. In MICCAI Workshop on Deep Generative Models, pages 34–44. Springer, 2022.
- [498] Julia Wolleb, Robin Sandkühler, Florentin Bieder, and Philippe C Cattin. The swiss army knife for image-to-image translation: Multi-task diffusion models. arXiv preprint arXiv:2204.02641, 2022.
- [499] Walter HL Pinaya, Mark S Graham, Robert Gray, Pedro F Da Costa, Petru-Daniel Tudosiu, Paul Wright, Yee H Mah, Andrew D MacKinnon, James T Teo, Rolf Jager, et al. Fast unsupervised brain anomaly detection and segmentation with diffusion models. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 705–714. Springer, 2022.
- [500] Virginia Fernandez, Walter Hugo Lopez Pinaya, Pedro Borges, Petru-Daniel Tudosiu, Mark S Graham, Tom Vercauteren, and M Jorge Cardoso. Can segmentation models be trained with fully synthetically generated data? In International Workshop on Simulation and Synthesis in Medical Imaging, pages 79–90. Springer, 2022.
- [501] Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, and Vishal M Patel. Ambiguous medical image segmentation using diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11536–11546, 2023.
- [502] Florentin Bieder, Julia Wolleb, Alicia Durrer, Robin Sandkuehler, and Philippe C Cattin. Memoryefficient 3d denoising diffusion models for medical image processing. In Medical Imaging with Deep Learning, 2023.
- [503] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated mri. Medical image analysis, 80:102479, 2022.
- [504] Yutong Xie and Quanzheng Li. Measurement-conditioned denoising diffusion probabilistic model for under-sampled medical image reconstruction. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 655–664. Springer, 2022.
- [505] Cheng Peng, Pengfei Guo, S Kevin Zhou, Vishal M Patel, and Rama Chellappa. Towards performant and reliable undersampled mr reconstruction via diffusion model sampling. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 623–633. Springer, 2022.
- [506] Alper Güngör, Salman UH Dar, S¸aban Öztürk, Yilmaz Korkmaz, Hasan A Bedel, Gokberk Elmas, Muzaffer Ozbey, and Tolga Çukur. Adaptive diffusion priors for accelerated mri reconstruction. Medical image analysis, 88:102872, 2023.
- [507] Guanxiong Luo, Martin Heide, and Martin Uecker. Mri reconstruction via data driven markov chain with joint uncertainty estimation. CoRR, 2022.
- [508] Zhuo-Xu Cui, Chentao Cao, Shaonan Liu, Qingyong Zhu, Jing Cheng, Haifeng Wang, Yanjie Zhu, and Dong Liang. Self-score: Self-supervised learning on score-based models for mri reconstruction. arXiv preprint arXiv:2209.00835, 2022.

- [509] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. Advances in Neural Information Processing Systems, 35:25683–25696, 2022.
- [510] Kuang Gong, Keith Johnson, Georges El Fakhri, Quanzheng Li, and Tinsu Pan. Pet image denoising based on denoising diffusion probabilistic model. European Journal of Nuclear Medicine and Molecular Imaging, 51(2):358–368, 2024.
- [511] Dewei Hu, Yuankai K Tao, and Ipek Oguz. Unsupervised denoising of retinal oct with diffusion probabilistic model. In Medical Imaging 2022: Image Processing, volume 12032, pages 25–34. SPIE, 2022.
- [512] Jin Sub Lee, Jisun Kim, and Philip M Kim. Proteinsgm: Score-based generative modeling for de novo protein design. bioRxiv, pages 2022–07, 2022.
- [513] Kevin E Wu, Kevin K Yang, Rianne van den Berg, Sarah Alamdari, James Y Zou, Alex X Lu, and Ava P Amini. Protein structure generation via folding diffusion. Nature communications, 15(1):1059, 2024.
- [514] Zhangyang Gao, Cheng Tan, and Stan Z Li. Diffsds: A language diffusion model for protein backbone inpainting under geometric conditions and constraints. arXiv preprint arXiv:2301.09642, 2023.
- [515] Yeqing Lin and Mohammed AlQuraishi. Generating novel, designable, and diverse protein structures by equivariantly diffusing oriented residue clouds. arXiv preprint arXiv:2301.12485, 2023.
- [516] Brian L Trippe, Jason Yim, Doug Tischer, David Baker, Tamara Broderick, Regina Barzilay, and Tommi Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem. arXiv preprint arXiv:2206.04119, 2022.
- [517] Jason Yim, Brian L Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay, and Tommi Jaakkola. Se (3) diffusion model with application to protein backbone generation. arXiv preprint arXiv:2302.02277, 2023.
- [518] John B Ingraham, Max Baranov, Zak Costello, Karl W Barber, Wujie Wang, Ahmed Ismail, Vincent Frappier, Dana M Lord, Christopher Ng-Thow-Hing, Erik R Van Vlack, et al. Illuminating protein space with a programmable generative model. Nature, 623(7989):1070–1078, 2023.
- [519] Yangtian Zhang, Zuobai Zhang, Bozitao Zhong, Sanchit Misra, and Jian Tang. Diffpack: A torsional diffusion model for autoregressive protein side-chain packing. Advances in Neural Information Processing Systems, 36, 2024.
- [520] Kaiyan Zhang, Sihang Zeng, Ermo Hua, Ning Ding, Zhang-Ren Chen, Zhiyuan Ma, Haoxin Li, Ganqu Cui, Biqing Qi, Xuekai Zhu, et al. Ultramedical: Building specialized generalists in biomedicine. arXiv preprint arXiv:2406.03949, 2024.
- [521] Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International conference on machine learning, pages 8867–8887. PMLR, 2022.
- [522] Han Huang, Leilei Sun, Bowen Du, and Weifeng Lv. Conditional diffusion based on discrete graph structures for molecular graph generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 4302–4311, 2023.
- [523] Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and Qiang Liu. Diffusion-based molecule generation with informative prior bridges. Advances in Neural Information Processing Systems, 35:36533– 36545, 2022.
- [524] Shitong Luo, Chence Shi, Minkai Xu, and Jian Tang. Predicting molecular conformation via dynamic graph score matching. Advances in Neural Information Processing Systems, 34:19784–19795, 2021.
- [525] Haotian Zhang, Shengming Li, Jintu Zhang, Zhe Wang, Jike Wang, Dejun Jiang, Zhiwen Bian, Yixue Zhang, Yafeng Deng, Jianfei Song, et al. Sdegen: learning to evolve molecular conformations from thermodynamic noise for conformation generation. Chemical Science, 14(6):1557–1568, 2023.
- [526] Fang Wu and Stan Z Li. Diffmd: a geometric diffusion model for molecular dynamics simulations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 5321–5329, 2023.
- [527] Ilia Igashov, Hannes Stärk, Clément Vignac, Arne Schneuing, Victor Garcia Satorras, Pascal Frossard, Max Welling, Michael Bronstein, and Bruno Correia. Equivariant 3d-conditional diffusion model for molecular linker design. Nature Machine Intelligence, pages 1–11, 2024.

- [528] Tomer Weiss, Eduardo Mayo Yanes, Sabyasachi Chakraborty, Luca Cosmo, Alex M Bronstein, and Renana Gershoni-Poranne. Guided diffusion for inverse molecular design. Nature Computational Science, 3(10):873–882, 2023.
- [529] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, pages 1–3, 2024.
- [530] Jiaqi Guan, Xiangxin Zhou, Yuwei Yang, Yu Bao, Jian Peng, Jianzhu Ma, Qiang Liu, Liang Wang, and Quanquan Gu. Decompdiff: diffusion models with decomposed priors for structure-based drug design. arXiv preprint arXiv:2403.07902, 2024.
- [531] Haitao Lin, Yufei Huang, Meng Liu, Xuanjing Li, Shuiwang Ji, and Stan Z Li. Diffbp: Generative diffusion of 3d molecules for target protein binding. arXiv preprint arXiv:2211.11214, 2022.
- [532] Arne Schneuing, Yuanqi Du, Charles Harris, Arian Jamasb, Ilia Igashov, Weitao Du, Tom Blundell, Pietro Lió, Carla Gomes, Max Welling, et al. Structure-based drug design with equivariant diffusion models. arXiv preprint arXiv:2210.13695, 2022.
- [533] Zhuoran Qiao, Weili Nie, Arash Vahdat, Thomas F Miller III, and Anima Anandkumar. Dynamicbackbone protein-ligand structure prediction with multiscale generative diffusion models. arXiv preprint arXiv:2209.15171, 1, 2022.
- [534] Wengong Jin, Siranush Sarkizova, Xun Chen, Nir Hacohen, and Caroline Uhler. Unsupervised proteinligand binding energy prediction via neural euler’s rotation equation. Advances in Neural Information Processing Systems, 36, 2024.
- [535] Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts. arXiv preprint arXiv:2407.06204, 2024.
- [536] Arpita Vats, Rahul Raja, Vinija Jain, and Aman Chadha. The evolution of mixture of experts: A survey from basics to breakthroughs. 2024.
- [537] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Kai Wang, Yibing Song, Gao Huang, Fan Wang, and Yang You. Dynamic diffusion transformer. arXiv preprint arXiv:2410.03456, 2024.
- [538] Alireza Ganjdanesh, Yan Kang, Yuchen Liu, Richard Zhang, Zhe Lin, and Heng Huang. Mixture of efficient diffusion experts through automatic interval and sub-network selection. arXiv preprint arXiv:2409.15557, 2024.
- [539] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.
- [540] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.
- [541] Jacob K Christopher, Brian R Bartoldson, Bhavya Kailkhura, and Ferdinando Fioretto. Speculative diffusion decoding: Accelerating language generation through diffusion. arXiv preprint arXiv:2408.05636, 2024.
- [542] Zizheng Pan, Bohan Zhuang, De-An Huang, Weili Nie, Zhiding Yu, Chaowei Xiao, Jianfei Cai, and Anima Anandkumar. T-stitch: Accelerating sampling in pre-trained diffusion models with trajectory stitching. arXiv preprint arXiv:2402.14167, 2024.

