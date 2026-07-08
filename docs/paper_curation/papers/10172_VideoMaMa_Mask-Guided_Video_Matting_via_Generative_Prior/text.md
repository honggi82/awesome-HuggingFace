## VideoMaMa: Mask-Guided Video Matting via Generative Prior

Sangbeom Lim1† Seoung Wug Oh2 Jiahui Huang2 Heeji Yoon3 Seungryong Kim3 Joon-Young Lee2

# arXiv:2601.14255v1[cs.CV]20Jan2026

1Korea University 2Adobe Research 3KAIST AI https://cvlab-kaist.github.io/VideoMaMa

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

VideoMaMaInput VideoInput Mask

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Model-generated Mask to Matte GT Mask to Matte

Figure 1. We introduce Video Mask-to-Matte Model (VideoMaMa), a diffusion-based model that generates high-quality alpha mattes from input binary segmentation masks obtained either from existing models such as SAM2 [30] or from ground-truth segmentation masks in existing datasets such as SA-V [30]. Examples shown highlights our VideoMaMa’s ability to capture fine-grained details including motion blur, and intricate boundary structures on natural video footage.

### Abstract

Generalizing video matting models to real-world videos remains a significant challenge due to the scarcity of labeled data. To address this, we present Video Mask-to-Matte Model (VideoMaMa) that converts coarse segmentation masks into pixel accurate alpha mattes, by leveraging pretrained video diffusion models. VideoMaMa demonstrates strong zero-shot generalization to real-world footage, even though it is trained solely on synthetic data. Building on this capability, we develop a scalable pseudo-labeling

†Work done during an internship at Adobe Research.

pipeline for large-scale video matting and construct the Matting Anything in Video (MA-V) dataset, which offers high-quality matting annotations for more than 50K realworld videos spanning diverse scenes and motions. To validate the effectiveness of this dataset, we fine-tune the SAM2 model on MA-V to obtain SAM2-Matte, which outperforms the same model trained on existing matting datasets in terms of robustness on in-the-wild videos. These findings emphasize the importance of large-scale pseudo-labeled video matting and showcase how generative priors and accessible segmentation cues can drive scalable progress in video matting research.

### 1. Introduction

Video matting, the task of extracting foreground objects with pixel-level precision from video, serves as a fundamental component in video editing applications, including background replacement [6, 31, 50], visual composition [3, 47], and relighting [2, 31, 42].

Despite its importance, training a video matting model that performs robustly across diverse real-world videos remains challenging for two key reasons. First, high-quality video matting annotations are extremely scarce. Groundtruth mattes are typically captured in controlled environments such as chroma key studios [27, 51] or with specialized camera setups [5], which makes it difficult to scale annotation efforts and to capture diverse objects, scenes, and camera angles. As a result, existing datasets are often limited to human portraits.

Second, most video matting models are trained on synthetic videos, where foregrounds are composited onto arbitrary backgrounds. Such synthetic compositions often introduce unrealistic artifacts in lighting, motion blur, and temporal coherence. These two factors, limited data diversity and the domain gap between synthetic and real videos, significantly hinder the ability of current models to generalize to real-world footage where complex interactions between foregrounds and backgrounds naturally occur.

In this work, we propose a novel bootstrapping strategy that effectively bridges the gap between synthetic and real-world videos, enabling more robust video matting. We introduce Video Mask-to-Matte Model (VideoMaMa), a diffusion-based model that converts binary segmentation masks into continuous alpha mattes (see Figure 1). Built upon pretrained video diffusion models [1], VideoMaMa demonstrates remarkable zero-shot generalization to realworld videos, even though the model is trained exclusively on synthetic data. This strong generalization ability stems from diffusion priors trained on internet-scale image and video data, which can generate high-quality content across diverse domains [1, 9, 41]. To adapt these priors for video matting while preserving their generative capabilities, we introduce a two-stage training strategy that separately optimizes spatial and temporal layers, along with semantic knowledge injection via DINOv3 [38] features.

To evaluate the robustness of the VideoMaMa model, we conduct experiments on video matting guided by binary masks from diverse sources, including those produced by existing video segmentation models. The results show that VideoMaMa consistently generates high-quality video matting outputs regardless of the input mask type, demonstrating strong robustness. These findings suggest that VideoMaMa can be effectively utilized as a pseudo-labeler for constructing large-scale video matting annotations.

In addition, by leveraging VideoMaMa, we propose a simple but scalable pipeline for constructing large-scale

video matting annotations and introduce the Matting Anything in Videos (MA-V) dataset, the first large-scale pseudo video matting dataset built by converting segmentation labels from the SA-V dataset [30]. Unlike existing video matting datasets [8, 14, 27, 48, 51], which rely on synthetic or composited content, MA-V provides high-quality matting annotations for over 50K real-world videos covering diverse scenes, objects, and motion dynamics, as summarized in Table 1. This scalable pipeline demonstrates the potential of leveraging generative priors and segmentation labels, which are considerably easier to obtain than video alpha matte, to efficiently construct large-scale, high-quality video matting datasets.

To validate the effectiveness of large-scale pseudo video matting annotations, we fine-tune the SAM2 [30] model for video matting using MA-V, referred to as SAM2-Matte. Trained without any architectural modifications, the SAM2Matte model achieves substantially more robust matting performance on in-the-wild videos than the same SAM2 model fine-tuned on existing video matting datasets, as well as other existing video matting methods. The results demonstrate the strong potential of large-scale pseudo annotations to drive advancements in video matting research.

Our contributions can be summarized as follows.

- • VideoMaMa: A diffusion-based model that generates realistic video matting annotations from binary masks, enabling scalable matting label creation with easily obtainable segmentation labels.
- • MA-V Dataset: The first large-scale, high-quality video matting dataset built on real captured footage, consisting of over 50K diverse videos.

### 2. Related Work

Video matting. Video matting methods can be categorized into auxiliary-free and auxiliary-guided approaches. Auxiliary-free methods [16, 26, 28] typically focus on portrait matting, restricting their applicability to human-centric scenarios. Trimap-guided approaches [13, 36] require manual trimap annotations, limiting their practicality for zeroshot inference. Recent mask-based guidance methods address these limitations: MaGGIe [14] decouples tracking from matting using binary mask tracks, MatAnyone [48] propagates target-assigned matting through a memoryaugmented architecture, and GVM [8] employs diffusion models for portrait matting. Despite these advances, existing methods remain constrained by domain specificity.

Beyond algorithmic limitations, existing datasets also constrain progress. Representative datasets include VideoMatte240K [27] (484 videos), VideoMatting108 [51] (108 videos), and VM800 [48] (826 videos), predominantly focusing on human subjects in controlled settings [8, 39, 44]. Critically, all rely on composition-based generation where foreground objects are composited onto random back-

grounds, creating artificial scene compositions. No largescale real video matting dataset exists where foreground and background naturally co-occur.

[Figure 29]

ℒreg GT Matte

MLP

DINOv3

|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]| |
|---|---|
| |ℒ|

Semantic injection

|[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Latent Encoder

[Figure 42]

[Figure 43]

mat

[Figure 44]

ℰ Video Frames

|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]|
|---|

Diffusion models for perception tasks. Recent advances have demonstrated that diffusion models [1, 21, 33, 41] encode rich priors about natural scenes, motion dynamics, and temporal coherence. Marigold [15] pioneered this direction by finetuning Stable Diffusion for monocular depth estimation, achieving strong zero shot generalization despite training only on synthetic data. This success has extended to semantic segmentation [43, 52] and video depth estimation [11, 37]. These models exhibit remarkable zero shot generalization across perception tasks [7, 10, 11, 45, 52], maintaining strong performance on real data even when trained exclusively on synthetic data, suggesting they can bridge the synthetic to real domain gap. For matting, SDMatte [12] applies diffusion models to interactive image matting, while GVM [8] focuses on portrait video matting with emphasis on hair details. However, these methods are limited to either single images or specific domains like human portraits, lacking the scale and generality needed for diverse video matting scenarios.

Video Diffusion U-Net

Concat.

Latent Decoder

[Figure 48]

|[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]|
|---|

𝒟 Latent

[Figure 52]

[Figure 53]

[Figure 54]

Prediction

Encoder ℰ Guide Masks

[Figure 55]

[Figure 56]

Spatial layer Temporal layer

Trainable Freeze

Figure 2. Overview of VideoMaMa architecture. RGB frames and guide masks are processed through video diffusion U-Net layers to generate high-quality video mattes. Semantic injection with DINO features is applied during training.

There are several ways to specify the target object in video matting. One approach is to predefine the target semantic class, for example human portraits [8, 22], while another is to condition the model on additional signals such as points, boxes, or masks [12, 14, 49].

We adopt binary segmentation masks as conditioning signals for two critical reasons: (1) masks allow the diffusion model to focus solely on generating fine-grained matting details (e.g., hair strands, motion blur) rather than inferring object boundaries, since shape information is already provided by the mask. This design decouples the challenges of object localization and matte generation, making the pipeline more robust and generalizable and (2) Finally, since binary masks can be obtained through various sources, the applicability of a model can be greatly expanded. In this paper, we demonstrate that VideoMaMa can not only perform video matting in various settings when combined with existing video segmentation models, but also enables the conversion of existing video segmentation datasets into matting datasets.

Pseudo-label generation in segmentation/matting has become a key driver of recent progress. In the Segment Anything Model [20, 30], the authors demonstrated that it is possible to create an iterative self-training loop by building a strong segmentation model and bootstrapping it with its own outputs. ZIM [18] trained a converter model that transforms image binary masks into alpha mattes using existing image matting datasets.

Inspired by the recent successes, we address video matting, which presents distinct challenges. Unlike image matting [22, 24, 25, 34] and video object segmentation, where abundant manually annotated ground-truth data [4, 17, 30, 46] enables training strong labeler models, video matting datasets remain scarce and predominantly synthetic. This limitation hinders model generalization to real-world scenarios. To overcome the challenge, we exploit the generative priors of pre-trained video diffusion models and train a robust pseudo-labeler using small-scale synthetic video matting annotations.

#### 3.2. Architecture Design

We build VideoMaMa on top of Stable Video Diffusion (SVD) [1], a pretrained video diffusion model originally designed for image-to-video generation. By leveraging generative priors, it can generate high-quality video matting annotations that exhibit realistic characteristics such as natural motion blur, proper edge detail, and temporal consistency, effectively transferring the model’s learned understanding of video dynamics to the matting task.

### 3. Video Mask-to-Matte Model

We adapt the SVD architecture for mask-guided video matting by introducing mask-based conditioning in the latent space and employing single-step inference for improved efficiency. In addition, we propose a semantic knowledge injection technique that enhances the model’s understanding of object boundaries and strengthens temporal consistency when tracking complex objects with fine structural details. An overview of the model is shown in Figure 2.

#### 3.1. Problem Formulation

Video matting is mathematically defined by the alpha compositing equation, I = αF + (1 − α)B, where I is the observed image, F is the foreground, B is the background, and α ∈ [0,1] is the alpha matte representing pixel-level opacity. Unlike binary segmentation masks M ∈ {0,1}, alpha mattes capture fine-grained details such as hair strands, motion blur, and intricate regions, which are critical for realistic video compositing.

Latent space formulation. Following modern video diffusion models, we operate in a compressed latent space

to alleviate the computational overhead of processing high resolution videos. We leverage a Variational Autoencoder (VAE) [19] for efficient encoding and decoding. All inputs, including video frames V = {It}Tt=1, binary mask track M = {Mt}Tt=1, and alpha mattes α = {αt}Tt=1, are encoded into the same latent space: zx = E(x),xˆ = D(zx), where E and D denote the encoder and decoder of the VAE, respectively, and x may represent V , M, or α. Since all three modalities share the same spatial dimensions, this unified encoding approach naturally allows them to be processed together in the compressed latent space, reducing memory requirements while preserving spatial correspondence.

[Figure 57]

###### Polygon Degradation

[Figure 58]

[Figure 59]

Original Matte

Weak Strong

[Figure 60]

###### Downsampling Degradation

[Figure 61]

[Figure 62]

Binarization

Polygon Degradation

or

Weak Strong

Downsampling Degradation

Figure 3. Examples of mask augmentation methods. Polygon and Downsampling degradation are applied at weak and strong augmentation levels.

Single-step diffusion formulation. We formulate VideoMaMa as a single-timestep generative model that directly synthesizes high-fidelity alpha matte latents zˆα in a single forward pass, where h and w represent the compressed spatial dimensions and c is the latent channel dimension. Unlike traditional diffusion models that require iterative denoising steps, our model performs direct prediction from noise to clean latents, making it significantly more efficient for large-scale dataset generation.

pling Degradation: We downsample and upsample the mask at the same scale, which effectively removes high-frequency details while preserving the overall shape. These augmentations create a gap between the coarse input mask and the fine-grained target alpha matte, encouraging the model to leverage appearance cues from the RGB video to generate realistic matting details.

The generation process takes as input the frame-wise concatenation of video latents zV , mask latents zM, and Gaussian noise ϵ ∼ N(0,I) along the channel dimension:

Two-stage training. For matting tasks, both training and inference resolution are critical. As matting operates at the pixel level with continuous values, downsampling to lower resolutions can degrade fine-grained matting details. However, training video diffusion models at high resolution is computationally prohibitive. To address this challenge, we propose a two-stage training strategy that decomposes the learning process: first training spatial layers at high resolution on single frames to capture fine details, then training temporal layers at lower resolution on video sequences to learn temporal consistency. In the first stage, we freeze the temporal layers of SVD and train only the spatial layers, enabling the model to focus on capturing pixel-level details at high resolution. While this stage allows the model to learn fine-grained features within individual frames, it lacks temporal consistency across video sequences. In the second stage, we freeze the spatial layers to preserve the learned detail-capturing capability and fine-tune only the temporal layers to learn temporal coherence and video-specific characteristics. This decomposed training strategy enables our model to achieve both high-resolution detail and temporal consistency without requiring full high-resolution video training.

zˆα = FSVD(concat(zV ,zM,ϵ)), (1)

where FSVD represents our adapted SVD model, and concat(·) denotes a concatenation. We modify SVD’s architecture by replacing the original image conditioning input with this concatenated tensor, allowing the model to leverage SVD’s strong temporal modeling capabilities while conditioning on both appearance (from zV ) and shape information (from zM). The final alpha matte video is obtained by decoding the predicted latents through the VAE decoder: αˆ = D(ˆzα).

#### 3.3. Training Recipe

Mask augmentation. While providing binary masks as input conditioning signals simplifies the task for the diffusion model, it also introduces the risk of copy-paste behavior. For objects with simple shapes (e.g., cars or desks) or masks that already contain fine-grained details (e.g., binarized alpha mattes), the model may trivially copy the mask instead of generating realistic matting details by reasoning about the RGB image.

To prevent this, we apply mask augmentation during training to remove fine-grained information from the input masks, forcing the model to infer matting details from the RGB image. Our mask augmentation strategy consists of two operations (see Figure 3): (1) Polygon Degradation: We approximate the mask boundary with a polygon, simplifying the contour and removing fine details. (2) Downsam-

Matting loss function. We adopt v-parameterization [35] for single-step generation, where the model is trained to directly convert noise into clean alpha matte latents. The predicted latent zˆα is decoded to pixel space αˆ = D(ˆzα), and the loss is computed at the pixel level:

V ,zM,ϵ [sim(D(ˆzα),α)], (2)

Lmat = Eα,z

###### Table 1. Dataset statistics for video matting.

###### Dataset Category # Videos Composition Required

SynHairMan [8] Human 200 Yes VideoMatte240K [27] Human 478 Yes V-HIM2K5 [14] Human 443 Yes VideoMatting108 [51] Various 108 Yes CRGNN [44] Various 60 Yes VM800 [48] Human 826 Yes

MA-V (Ours) Various 50,541 No

where sim(·,·) comprises pixel-wise similarity loss terms between the decoded prediction and ground-truth alpha matte. This pixel-level supervision enables direct optimization for visual quality, capturing fine-grained details such as motion blur and intricate boundary structures.

Semantic knowledge injection. While diffusion priors are powerful for generating precise alpha mattes, they struggle with semantic understanding of object boundaries and consistently tracking objects, particularly for complex objects with intricate structures. To address this, we inject semantic knowledge into VideoMaMa by aligning SVD features with robust semantic representations from DINOv3 [38] throughout training.

We first extract DINO features hdino = Fdino(V ) from the video frames V . Then, we extract intermediate features hl from the l-th layer of the diffusion model and project them into the DINO feature space using a learnable MLP pϕ with parameters ϕ. We minimize the alignment loss that maximizes patch-wise cosine similarities:

V ,zM,ϵ,V cos-sim(hdino,pϕ(hl)) , (3)

Lreg = −Eα,z

where n indexes patches and cos-sim(·,·) denotes cosine similarity. This design allows the model to leverage semantic understanding of object categories and structures while maintaining the generative capabilities of the diffusion backbone, enabling more accurate matte generation for challenging cases such as overlapping objects or complex articulated structures.

### 4. Matte Anything in Videos (MA-V) Dataset

The difficulty of obtaining video matting annotations has severely limited existing datasets. As shown in Table 1, prior datasets contain at most hundreds of videos, predominantly focusing on human subjects captured in controlled environments [27, 51] or through manual annotation [22, 23, 29, 40]. While composition-based approaches enable scaling and precise annotations, they create artificial scene compositions that differ fundamentally from natural video footage, limiting model generalization to real world scenarios.

#### 4.1. Dataset Construction

To address these limitations, we introduce Matte Anything in Videos (MA-V), created by applying VideoMaMa to annotate SA-V’s [30] diverse mask annotations. As shown in Table 1, MA-V provides 50,541 videos captured in natural settings, nearly 50× larger than existing real-video datasets. Unlike prior work that focuses predominantly on humans, MA-V encompasses diverse object categories at multiple scales, from small objects to full scenes. The dataset construction process is straightforward yet effective: we leverage VideoMaMa’s ability to generate highquality alpha mattes from binary masks, converting SAV’s segmentation annotations into continuous matting labels while preserving the natural scene context. Critically, MA-V is the first large-scale video matting dataset where both foreground and background naturally co-occur in real captured footage, eliminating the synthetic composition gap that limits previous datasets. Figure 4 illustrates the visual quality and diversity of MA-V compared to SA-V’s binary masks, demonstrating how VideoMaMa generates fine-grained matting details including semi-transparent regions, motion blur, and intricate boundary structures regardless of the target category that are absent in the source segmentation masks.

#### 4.2. Fine-tuning SAM2 with MA-V

We finetune SAM2 [30] on our MA-V dataset, referred to as SAM2-Matte hereafter. SAM2, originally designed for binary segmentation, requires minimal adaptation for matting: we apply a sigmoid function after the mask logits to produce continuous alpha values in the range [0,1]. To assess MA-V quality, we also finetune SAM2 on existing video matting datasets and evaluate their performance on standard matting benchmarks.

### 5. Experiments

#### 5.1. Experimental Details

Implementation details. We train VideoMaMa based on the Stable Video Diffusion [1] (SVD) model using diverse image and video matting datasets. Our two-stage training strategy employs different resolutions optimized for each stage’s objective. In stage 1, we train the spatial layers of SVD on single images at high resolution (1024×1024) to capture fine-grained matting details such as hair strands and intricate boundary structures. We freeze the temporal layers during this stage to focus learning on spatial precision. In stage 2, we freeze the learned spatial layers and train only the temporal layers on video sequences. To balance computational efficiency with temporal modeling, we use 3-frame clips at 704×704 resolution, which enables the model to learn temporal consistency and motion-aware matting characteristics. To inject semantic knowledge, we incorporate

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

RGB SA-V MA-V RGB SA-V MA-V RGB SA-V MA-V

- Figure 4. Qualitative examples from our MA-V dataset. We show RGB frames with our high-quality MA-V annotations and original SA-V [30] masks for comparison. MA-V provides refined alpha mattes for diverse scenarios.

| |[Figure 81]|
|---|---|
| | |

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

| | |
|---|---|
| | |

[Figure 86]

[Figure 87]

[Figure 88]

| | |
|---|---|
| | |

[Figure 89]

| | |
|---|---|
| | |

[Figure 90]

| | |
|---|---|
| | |

[Figure 91]

|[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]|
|---|

[Figure 101]

| | |
|---|---|
| | |

[Figure 102]

| | |
|---|---|
| | |

[Figure 103]

[Figure 104]

[Figure 105]

| | |
|---|---|
| | |

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

| |
|---|

[Figure 118]

[Figure 119]

| |
|---|

[Figure 120]

| |
|---|

[Figure 121]

| |
|---|

[Figure 122]

| |
|---|

[Figure 123]

| |
|---|

[Figure 124]

| |
|---|

[Figure 125]

|[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]|
|---|

[Figure 135]

|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]|
|---|

[Figure 143]

| |
|---|

[Figure 144]

|[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]|
|---|

[Figure 154]

|[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]|
|---|

[Figure 163]

|[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]|
|---|

[Figure 172]

| |
|---|

[Figure 173]

| |
|---|

[Figure 174]

|[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]|
|---|

[Figure 184]

|[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]|
|---|

[Figure 193]

|[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]|
|---|

[Figure 203]

|[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]|
|---|

[Figure 212]

| |
|---|

[Figure 213]

| |
|---|

[Figure 214]

| |
|---|

[Figure 215]

| |
|---|

[Figure 216]

| |
|---|

[Figure 217]

| |
|---|

[Figure 218]

| |
|---|

[Figure 219]

| |
|---|

[Figure 220]

| |
|---|

[Figure 221]

| |
|---|

[Figure 222]

| |
|---|

[Figure 223]

| |
|---|

[Figure 224]

| |
|---|

[Figure 225]

| |
|---|

[Figure 226]

| |
|---|

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

| |
|---|

VideoMaMa (Ours)

SAM2

MatAnyone

MaGGIe

SAM2-Matte (Ours)

All-frame mask-guided

First-frame mask-guided

[Figure 234]

Input Video

[Figure 235]

[Figure 236]

|[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]|
|---|

[Figure 246]

| | |
|---|---|
| | |

[Figure 247]

[Figure 248]

[Figure 249]

|[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]|
|---|

[Figure 258]

|[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]|
|---|

[Figure 268]

|[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]|
|---|

[Figure 277]

[Figure 278]

- Figure 5. Qualitative comparison on in-the-wild videos. We evaluate two settings: (1) all-frame mask-guided video matting where VideoMaMa is compared against MaGGIe [14], and (2) first-frame mask-guided matting where SAM2-Matte is compared against MatAnyone [48]. All methods use SAM2 [30] to generate mask inputs.

features from a frozen DINOv3 [38] encoder, aligning the DINO features with the first upsampling block of the SVD decoder. This integration provides the diffusion model with semantic understanding of object structures while preserving its generative capabilities.

Training details. For both stages, we use a batch size of 64 and a learning rate of 5×10−5 with the AdamW optimizer. The training objective, Lmat, combines L1 loss for pixel-

wise accuracy and Laplacian loss for preserving edge sharpness and boundary details. Each stage is trained for 10,000 iterations until convergence. All experiments are conducted on NVIDIA A100 GPUs with mixed-precision training for efficiency. SAM2-Matte is trained on the combination of existing video matting datasets and MA-V. Details of the existing datasets are provided in the appendix.

- Table 2. All-frame mask-guided video matting comparison on V-HIM60 and YouTubeMatte benchmarks. We compare VideoMaMa (Ours) against mask-guided matting methods: MaGGIe [14] (video mask-guided) and MGM [49] (image mask-guided). We evaluate on two mask types: Synthetically Degraded Masks including downsampling (8×, 32×) and polygon degradation with varying difficulty levels, and Model-Generated Masks from SAM2 [30]. Lower values indicate better performance.

V-HIM60 [14] (Hard) YouTubeMatte [48] (1920 × 1080)

MAD ↓ Gradient ↓ MAD ↓ Gradient ↓ Input Mask Input MGM [49] MaGGle [14] Ours Input MGM [49] MaGGle [14] Ours Input MGM [49] MaGGle [14] Ours Input MGM [49] MaGGle [14] Ours Synthetically Degraded Masks

Downsampl. 8x 2.744 63.8374 2.3790 1.306 11.127 16.5609 5.4150 2.363 1.946 3.4973 1.6420 0.934 8.034 4.0822 4.9733 2.031 Downsampl. 32x 5.132 79.2273 2.5375 1.461 33.664 22.2024 5.9073 2.849 4.203 3.9061 1.8171 1.029 36.540 5.3603 5.6318 2.560 Polygon. (Easy) 3.636 64.5457 2.5808 1.416 16.000 17.3675 5.5572 2.480 3.619 4.1390 2.2250 1.206 19.155 4.8579 5.6773 2.335 Polygon. (Hard) 6.771 75.5501 3.1567 1.640 43.228 20.5516 6.4121 3.024 9.470 4.4549 2.3111 1.404 65.844 5.6132 5.9184 3.237

Model-Generated Masks SAM2 [30] 4.666 66.5221 3.1567 2.435 21.960 18.2103 6.4121 4.180 3.569 4.1136 1.9499 1.737 24.729 5.0448 5.7877 3.635

#### 5.2. Quantitative Evaluations

All-frame mask-guided video matting. We evaluate VideoMaMa following the protocol in [14], where input binary masks for every frame are provided. We use two categories of input masks: (1) Synthetically Degraded Masks through downsampling (8×, 32×) and polygon degradation (easy, hard), mimicking imperfect user annotations, and (2) Model-Generated Masks from SAM2 [30], representing automatic segmentation pipelines. This diverse set evaluates VideoMaMa’s robustness across different mask quality levels.

In Table 2, we compare VideoMaMa against MaGGIe [14] and MGM [49]. For MGM, which is an imagebased method, we perform inference on each frame independently. We report metrics comparing both input masks and generated results against ground-truth alpha mattes. All evaluations use 12-frame sequences.

We adopt standard metrics: MAD for overall accuracy and Gradient error [32] for boundary quality. Results are reported on V-HIM60 Hard [14] and YouTubeMatte 1920×1080 [48].

VideoMaMa consistently outperforms existing maskguided matting models across both synthetically degraded and model-generated masks. Our evaluation across different mask styles demonstrates that VideoMaMa can handle various types of mask inputs and produce high-quality video mattes consistently.

First-frame mask-guided video matting. In Table 3, we evaluate VideoMaMa and SAM2-Matte following a setting proposed in [48] where a binary mask guidance is provided only for the first frame. For VideoMaMa, the binary mask guidance across all frames is generated by propagating the mask from the first frame using SAM2 [30], denoted as SAM2+VideoMaMa. We also report the score of the binary masks produced by SAM2 to provide a baseline for the improvements achieved by VideoMaMa.

We employ multiple evaluation metrics: MAD for overall semantic accuracy, MSE for pixel-level error, MAD-T (trimap-based MAD) which computes MAD only within the uncertain trimap region derived from the ground-truth

alpha map, and Gradient error [32] for assessing boundary detail extraction quality. The trimap-based evaluation is particularly important as it focuses on the challenging boundary regions where precise matting is most critical.

SAM2-Matte trained on the combined dataset surpasses previous state-of-the-art methods, MatAnyone [48], on challenging benchmarks: V-HIM60 Hard [14] and YouTubeMatte [48]. This demonstrates that MA-V provides high-quality training data for video matting models, enabling to extract matting quality and temporal consistency over the video frames. Note that, for VideoMaMa in this experiment, we processed each of the 12-frame segments and merged the results to obtain the complete video matting.

#### 5.3. Qualitative Evaluations

To further evaluate the model’s performance on real-world videos, we present qualitative comparisons in Figure 5. We categorize the methods into two settings: first-frame maskguided and all-frame mask-guided. The initial binary masks are obtained manually using the SAM2 model. Unlike previous matting methods [14, 48], VideoMaMa and SAM2Matte not only perform well on human portraits but also generalize to a wide range of object categories and in-thewild video content. Moreover, the high-quality results from SAM2-Matte demonstrate that MA-V’s large-scale and diverse annotations effectively enhance video matting performance beyond what existing datasets can achieve.

#### 5.4. Ablation Studies

Number of inference frames. We evaluate VideoMaMa’s performance across varying numbers of frames. Although trained on maximum 3 frames, VideoMaMa successfully handles diverse frame counts from single frames to 24 frames at inference time. As shown in Table 4, the model maintains consistent performance across different frame counts, demonstrating strong temporal generalization.

Training recipe. We ablate our two-stage training strategy and semantic knowledge injection on YouTubeMatte [48] with four configurations: (1) image only (stage 1), (2) video only (stage 2), (3) two-stage without semantic knowledge,

- Table 3. First-frame mask-guided video matting comparison on V-HIM60 and YouTubeMatte benchmarks. We evaluate video matting performance across different difficulty levels in V-HIM60. The best and second-best results are highlighted.

Method

V-HIM60 [14] YouTubeMatte [48] Easy Medium Hard 1920 × 1080

MAD MAD-T MSE GRAD MAD MAD-T MSE GRAD MAD MAD-T MSE GRAD MAD MAD-T MSE GRAD

SAM2 [30] 2.7978 91.5246 1.4799 13.3307 4.9618 105.2547 2.9401 25.6735 7.8517 130.0338 6.0075 26.3355 3.8530 103.2181 2.6133 37.4074 MatAnyone [48] 3.5803 92.8273 1.9905 7.0763 7.1955 100.0465 4.7586 11.4541 5.7195 102.4861 3.4620 9.8199 1.9909 51.0939 0.7085 8.9019 SAM2+VideoMaMa 1.3539 40.6830 0.4063 2.8876 2.3132 49.0329 0.9335 4.6944 2.7757 53.4392 1.4428 4.8276 1.3559 35.5064 0.2940 3.5742 SAM2-Matte 1.3446 49.0329 0.2932 3.1245 2.2710 51.8996 0.6669 5.2227 2.6112 58.7796 1.0750 5.0869 1.2695 33.5268 0.2831 3.4640

- Table 4. Ablation study on the number of frame during inference. We report the mask refinement performance in MAD metric on YouTubeMatte dataset [48] with different input masks.

Number of Frames per Inference Mask Input 24 18 12 6 1 Downsampl. 32x

Input 4.2411 4.2455 4.2029 4.1377 4.1336 Refined 1.0237 1.0272 1.0292 1.0418 1.1024

Polygon. (Hard)

Input 9.5559 9.6224 9.4700 8.7498 8.5455 Refined 1.4472 1.4847 1.4042 1.3382 1.5598

SAM2 [30]

Input 3.6023 3.5991 3.5693 3.5476 3.8711 Refined 1.6921 1.7273 1.7370 1.7948 1.8828

- Table 5. Ablation study on training recipe. We evaluate MAD with different training configurations on YouTubeMatte [48]. Scores for input masks are shown below the each input mask type. Percentages in parentheses indicate the relative improvement (green) or degradation (red) compared to the input score.

S1 S2 DINO

Downsampl. 32x Input: 4.2029

Polygon. (Hard) Input: 9.4700

SAM2 Input: 3.5693

✓ ✗ ✗ 3.7620 (10.49%) 4.8306 (49.00%) 4.0496 (-13.46%) ✗ ✓ ✗ 1.2350 (70.62%) 2.2372 (76.38%) 2.3124 (35.21%) ✓ ✓ ✗ 1.2552 (70.13%) 1.9975 (78.91%) 1.9404 (45.64%) ✓ ✓ ✓ 1.0292 (75.51%) 1.4042 (85.17%) 1.7370 (51.34%)

- Table 6. Ablation study for training data on V-HIM60 Hard [14] and DAVIS val [17], including (a) ED, (B) MAV, (c) ED + MA-V. ED denotes that existing dataset.

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

| | |
|---|---|
| | |
| | |

Time

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

RGB MatAnyone SAM2-Matte (ED) SAM2-Matte (MA-V)

Figure 6. Comparison on a real video between MatAnyone [48] and SAM2-Matte variants. Red arrows highlight regions where our method achieves the best results.

datasets only (ED), (b) MA-V only, and (c) both combined. A detailed list of the existing matting datasets used in this experiment will be included in the appendix.

To validate the impact of MA-V on matting task, we report matting quality metrics on the V-HIM60 (Hard) [14]. As shown in Table 6, Configuration (a) shows limited performance due to the small scale and lack of diversity in existing datasets. Remarkably, (b) significantly outperforms (a) and even surpasses state-of-the-art results, demonstrating the effectiveness of MA-V. When combining the existing matting datasets with MA-V in (c), we achieve the best overall matting quality. We hypothesize that the shared synthetic nature of both the existing matting datasets and V-HIM60 contributes to this improved performance.

We evaluate tracking performance on DAVIS [17] using VOS metrics by binarizing matting results. As shown in Table 6, configuration (b) substantially improves performance over (a), demonstrating the effectiveness of MA-V. Interestingly, adding the existing matting datasets in configuration (c) leads to degraded tracking performance compared to using MA-V alone, suggesting that existing synthetic datasets introduce domain biases that reduce tracking robustness on in-the-wild videos.

V-HIM60 (Hard) DAVIS MAD MSE GRAD J&F J F

Method

MatAnyone [48] 4.6655 2.5870 8.8870 79.7 76.5 82.9

- (a) ED 7.5798 5.6824 8.7091 77.0 74.1 80.0
- (b) MA-V 3.1830 1.5603 5.7869 87.9 85.2 90.6
- (c) ED + MA-V 2.6112 1.0750 5.0869 85.9 83.5 88.3

In Figure 6, we compare MatAnyone [48] and SAM2Matte variants. The results show that using MA-V enables robust tracking even along ambiguous boundaries.

and (4) two-stage training with DINO features. As shown in Table 5, having both stage 1 and stage 2 training stages led to the best results. Adding DINO semantic features further improves performance enabling better object understanding and boundary localization. The results validate that both components are necessary and complementary for optimal performance.

### 6. Conclusion

We introduce VideoMaMa, a diffusion-based model that leverages generative priors to perform robust video matting. By exploiting VideoMaMa’s strong generalization capability, we construct MA-V, the first large-scale pseudo video matting dataset built on real-world videos, by converting segmentation masks from SA-V. Through compre-

Impact of MA-V Dataset. We assess the impact of MA-V on matting and tracking performance on first-frame maskguided matting task. We evaluate SAM2-Matte trained under different data configurations by comparing (a) existing

hensive experiments, we demonstrate that both VideoMaMa and models trained on MA-V (e.g., SAM2-Matte) achieve state-of-the-art performance across diverse video matting benchmarks, validating the effectiveness of our data generation approach. We believe that VideoMaMa and MA-V will significantly contribute to advancing video matting research by providing both a VideoMaMa and a large-scale, high-quality dataset that bridges the synthetic-to-real domain gap.

## Appendix

This supplementary material provides additional details on implementation, evaluation metrics, and qualitative results that could not be included in the main paper due to space constraints.

The document is organized as follows:

- • Section A elaborates on the training configurations for SAM2-Matte and details the architectural design of the VideoMaMa feature injection module.
- • Section B provides the formal definition and algorithm for the Trimap-based Mean Absolute Difference (MADT) metric, designed to evaluate transition region accuracy frame-by-frame.
- • Section C presents a comparative analysis between SAM2-Matte and the original binary SAM2, empirically demonstrating the necessity and effectiveness of the MA-V dataset.
- • Section D analyzes the limitations of VideoMaMa, specifically discussing failure cases arising from reliance on semantically inaccurate instance masks.

### A. Additional Implementation Details

VideoMaMa To obtain in-the-wild results from VideoMaMa, we require input masks for all frames in the video sequence. We employ a point prompt on the first frame, which is then propagated throughout the entire video using SAM2’s tracking capability.

For quantitative evaluation on benchmark datasets, we binarize the ground-truth mask from the first frame to serve as the initial prompt for SAM2, subsequently propagating it across all remaining frames to generate the input mask sequence for VideoMaMa.

In the VideoMaMa architecture, we extract semantic features using DINOv2 [38] and inject them into the first upsampling block (Upblock 1) of the Stable Video Diffusion (SVD) [1] backbone. The feature injection is performed through a two-layer MLP projection module that aligns the DINOv3 feature dimensions with the SVD decoder architecture.

SAM2-Matte We implement SAM2-Matte based on the SAM2 [30] Base-Plus architecture, initialized with official

Table 7. Summary of the datasets constituting the “Existing Dataset” used to train SAM2-Matte.

Category Datasets

Image DVM [39], AM-2k [24], DAViD [34], P3M-10k [22], RefMatte [25] Video CRGNN [44], VideoMatte240K [27], DVM [39] Background VideoMatting108 [51], DVM [39]

pretrained weights. The model is trained for 100,000 iterations with a batch size of 4, employing a mixed training strategy that combines both video and image samples. The composition of existing datasets utilized in this training phase is detailed in Table 7. For optimization, we adopt a combination of L1 and Laplacian pyramid losses. The base learning rate is set to 5 × 10−6, while a separate learning rate of 3×10−6 is applied specifically to the image encoder. Training is conducted with a cosine learning rate scheduler.

### B. Evaluation Metric Details

Evaluating matte accuracy in transition regions is critical for video matting, particularly for capturing fine details like hair or motion blur. Standard metrics often dilute these errors by averaging over large background regions. To address this, we evaluate MAD-T, which restricts the Mean Absolute Difference calculation specifically to the unknown region. The complete calculation procedure is detailed in Algorithm 1. As ground-truth trimaps are rarely available for video datasets, we generate pseudo-trimaps dynamically for each frame t. We derive binary foreground and back-

ground masks from the ground-truth alpha αgt(t) and apply morphological erosion using a 10×10 elliptical structuring element. The unknown region U(t) is defined as the set of pixels excluded from both eroded masks. The final MAD-T score is reported as the temporal average of the per-frame error within U(t), scaled by 103.

### C. Comparison with SAM2

SAM2 [30] generates binary masks by computing the dot product between image features and prompt tokens that undergo cross-attention in SAM’s decoder blocks, followed by thresholding at a specified value. By removing the thresholding operation and retaining only the sigmoid activation, the model can simulate alpha matte generation by mapping the mask logits to the continuous range [0,1], which forms the basis of our SAM2-Matte approach. We compare our SAM2-Matte with the original SAM2 in Figure 7 to demonstrate the effectiveness of the MA-V dataset, illustrating that without fine-tuning on MA-V, SAM2 cannot generate robust video matting results.

Algorithm 1 Frame-wise Calculation of MAD-T Require: Pred. αˆt, GT αgt,t, Kernel k = 10 Ensure: MAD-T Score St for frame t

- 1: Initialize structuring element K ← Ellipse(k,k)
- 2: ▷ Define Certain Regions via Erosion
- 3: Mfg ← I(αgt,t = 255) ▷ Binary Foreground
- 4: Mbg ← I(αgt,t = 0) ▷ Binary Background
- 5: Cfg ← Mfg ⊖ K ▷ Erode to ensure certainty
- 6: Cbg ← Mbg ⊖ K ▷ Erode to ensure certainty
- 7: ▷ Isolate Unknown Region (Trimap)
- 8: Ut ← ¬(Cfg ∪ Cbg)
- 9: Nt ← Ut ▷ Pixel count in unknown region
- 10: ▷ Compute Error in Unknown Region
- 11: if Nt > 0 then
- 12: Dt ← |αˆt − αgt,t|
- 13: St ← N1

t (x,y)∈Ut Dt,x,y × 1000

- 14: else
- 15: St ← 0
- 16: end if
- 17: return St

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

| | |
|---|---|
| | |
| | |

Time

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

| | |
|---|---|
| | |
| | |

Time

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

RGB SAM2 SAM2-Matte (ED) SAM2-Matte (MA-V)

Figure 7. Comparison on a real video between SAM2-Matte and SAM2 [30] with sigmoid function on mask logit to simulate alpha matte generation. ED denotes that existing dataset.

### D. Limitation

VideoMaMa While VideoMaMa can generate high-quality video mattes from rough or imperfect masks, it cannot generate mattes for regions where completely wrong mask is provided (e.g. capturing wrong instance). As shown in Figure 8, when the input mask incorrectly captures the wrong instance, VideoMaMa propagates this error to the output. Our method successfully refines masks that lack fine-grained details but struggles when the input mask is fundamentally incorrect, such as when it captures an entirely different object instance. This limitation is inherent to mask-guided approaches, as the model relies on the mask to define the target foreground object.

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Input Video

[Figure 312]

[Figure 313]

[Figure 314]

Input Mask

[Figure 315]

[Figure 316]

[Figure 317]

VideoMaMa

Figure 8. Limitations. As highlighted by the red arrows, the model struggles to refine the matte when the input guidance mask exhibits significant errors from the input mask.

SAM2-Matte Conducting alpha matte generation in highresolution is essential for getting high quality alpha matte. However, the standard SAM2 architecture has a structural limitation for this task due to its mask decoder design. Specifically, SAM2 generates segmentation masks at a resolution of 64 × 64, which is subsequently upsampled to match the input dimensions. This resolution is substantially lower than those employed by other matting methods [18, 48], which typically predict alpha mattes at significantly higher resolutions to preserve fine-grained details such as hair strands and object boundaries. Consequently, directly applying SAM2 for video matting results in the loss of critical high-frequency information necessary for accurate alpha matte estimation. This architectural constraint poses a fundamental challenge for adapting SAM2 to the video matting task, where precise boundary delineation and detail preservation are paramount in getting alpha value of [0,1].

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 5, 9
- [2] Sumit Chaturvedi, Mengwei Ren, Yannick Hold-Geoffroy, Jingyuan Liu, Julie Dorsey, and Zhixin Shu. Synthlight: Portrait relighting with diffusion model by learning to re-render synthetic faces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 369–379, 2025. 2
- [3] Yusuf Dalva, Yijun Li, Qing Liu, Nanxuan Zhao, Jianming Zhang, Zhe Lin, and Pinar Yanardag. Layerfusion: Harmonized multi-layer text-to-image generation with generative priors. arXiv preprint arXiv:2412.04460, 2024. 2
- [4] Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. Mose: A new dataset for video object segmentation in complex scenes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 20224–20234, 2023. 3
- [5] Kenji Enomoto, Scott Cohen, Brian Price, and TJ Rhodes. Polarized color screen matting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 391–399, 2025. 2
- [6] Wenshuo Gao, Xicheng Lan, and Shuai Yang. Anyportal: Zero-shot consistent video background replacement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18990–18999, 2025. 2
- [7] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan De Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 753–

762. IEEE, 2025. 3

- [8] Yongtao Ge, Kangyang Xie, Guangkai Xu, Li Ke, Mingyu Liu, Longtao Huang, Hui Xue, Hao Chen, and Chunhua Shen. Generative video matting. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10,

2025. 2, 3, 5

- [9] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 2
- [10] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024. 3
- [11] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2005–2015, 2025. 3

- [12] Longfei Huang, Yu Liang, Hao Zhang, Jinwei Chen, Wei Dong, Lunde Chen, Wanyu Liu, Bo Li, and Peng-Tao Jiang. Sdmatte: Grafting diffusion models for interactive matting. arXiv preprint arXiv:2508.00443, 2025. 3
- [13] Wei-Lun Huang and Ming-Sui Lee. End-to-end video matting with trimap propagation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14337–14347, 2023. 2
- [14] Chuong Huynh, Seoung Wug Oh, Abhinav Shrivastava, and Joon-Young Lee. Maggie: Masked guided gradual human instance matting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3870– 3879, 2024. 2, 3, 5, 6, 7, 8
- [15] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9492–9502,

2024. 3

- [16] Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, and Rynson WH Lau. Modnet: Real-time trimap-free portrait matting via objective decomposition. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1140– 1147, 2022. 2
- [17] Anna Khoreva, Anna Rohrbach, and Bernt Schiele. Video object segmentation with language referring expressions. In ACCV, 2018. 3, 8
- [18] Beomyoung Kim, Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, and Joonsang Yu. Zim: Zero-shot image matting for anything. arXiv preprint arXiv:2411.00626, 2024. 3, 10
- [19] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [20] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 3
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [22] Jizhizi Li, Sihan Ma, Jing Zhang, and Dacheng Tao. Privacypreserving portrait matting. In Proceedings of the 29th ACM international conference on multimedia, pages 3501–3509,

2021. 3, 5, 9

- [23] Jizhizi Li, Jing Zhang, and Dacheng Tao. Deep automatic natural image matting. arXiv preprint arXiv:2107.07235,

2021. 5

- [24] Jizhizi Li, Jing Zhang, Stephen J Maybank, and Dacheng Tao. Bridging composite and real: towards end-to-end deep image matting. International Journal of Computer Vision, 130(2):246–266, 2022. 3, 9
- [25] Jizhizi Li, Jing Zhang, and Dacheng Tao. Referring image matting. In Proceedings of the IEEE Computer Vision and Pattern Recognition, 2023. 3, 9

- [26] Jiachen Li, Vidit Goel, Marianna Ohanyan, Shant Navasardyan, Yunchao Wei, and Humphrey Shi. Vmformer: End-to-end video matting with transformer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6678–6687, 2024. 2
- [27] Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian L Curless, Steven M Seitz, and Ira KemelmacherShlizerman. Real-time high-resolution background matting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8762–8771, 2021. 2, 5, 9
- [28] Shanchuan Lin, Linjie Yang, Imran Saleemi, and Soumyadip Sengupta. Robust high-resolution video matting with temporal guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 238–247,

2022. 2

- [29] Yu Qiao, Yuhao Liu, Xin Yang, Dongsheng Zhou, Mingliang Xu, Qiang Zhang, and Xiaopeng Wei. Attention-guided hierarchical structure aggregation for image matting. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 5
- [30] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 1, 2, 3, 5, 6, 7, 8, 9, 10
- [31] Mengwei Ren, Wei Xiong, Jae Shin Yoon, Zhixin Shu, Jianming Zhang, HyunJoon Jung, Guido Gerig, and He Zhang. Relightful harmonization: Lighting-aware portrait background replacement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6452–6462, 2024. 2
- [32] Christoph Rhemann, Carsten Rother, Jue Wang, Margrit Gelautz, Pushmeet Kohli, and Pamela Rott. A perceptually motivated online benchmark for image matting. In 2009 IEEE conference on computer vision and pattern recognition, pages 1826–1833. IEEE, 2009. 7
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [34] Fatemeh Saleh, Sadegh Aliakbarian, Charlie Hewitt, Lohit Petikam, Antonio Criminisi, Thomas J Cashman, Tadas Baltruˇsaitis, et al. David: Data-efficient and accurate vision models from synthetic data. arXiv preprint arXiv:2507.15365, 2025. 3, 9
- [35] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 4
- [36] Hongje Seong, Seoung Wug Oh, Brian Price, Euntai Kim, and Joon-Young Lee. One-trimap video matting. In European Conference on Computer Vision, pages 430–448. Springer, 2022. 2
- [37] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from

- video diffusion priors. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22841– 22852, 2025. 3
- [38] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 2, 5, 6, 9
- [39] Yanan Sun, Guanzhi Wang, Qiao Gu, Chi-Keung Tang, and Yu-Wing Tai. Deep video matting via spatio-temporal alignment and aggregation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6975–6984, 2021. 2, 9
- [40] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Ultrahigh resolution image/video matting with spatio-temporal sparsity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14112– 14121, 2023. 5
- [41] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3
- [42] Junying Wang, Jingyuan Liu, Xin Sun, Krishna Kumar Singh, Zhixin Shu, He Zhang, Jimei Yang, Nanxuan Zhao, Tuanfeng Y Wang, Simon S Chen, et al. Comprehensive relighting: Generalizable and consistent monocular human relighting and harmonization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 380–390,

2025. 2

- [43] Jiepeng Wang, Zhaoqing Wang, Hao Pan, Yuan Liu, Dongdong Yu, Changhu Wang, and Wenping Wang. Mmgen: Unified multi-modal image generation and understanding in one go. arXiv preprint arXiv:2503.20644, 2025. 3
- [44] Tiantian Wang, Sifei Liu, Yapeng Tian, Kai Li, and MingHsuan Yang. Video matting via consistency-regularized graph neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4902– 4911, 2021. 2, 5, 9
- [45] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? arXiv preprint arXiv:2403.06090,

2024. 3

- [46] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327, 2018. 3
- [47] Jinrui Yang, Qing Liu, Yijun Li, Soo Ye Kim, Daniil Pakhomov, Mengwei Ren, Jianming Zhang, Zhe Lin, Cihang Xie, and Yuyin Zhou. Generative image layer decomposition with visual effects. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7643–7653, 2025. 2
- [48] Peiqing Yang, Shangchen Zhou, Jixin Zhao, Qingyi Tao, and Chen Change Loy. Matanyone: Stable video matting with consistent memory propagation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7299–7308, 2025. 2, 5, 6, 7, 8, 10

- [49] Qihang Yu, Jianming Zhang, He Zhang, Yilin Wang, Zhe Lin, Ning Xu, Yutong Bai, and Alan Yuille. Mask guided matting via progressive refinement network. arXiv preprint arXiv:2012.06722, 2020. 3, 7
- [50] Jianshu Zeng, Yuxuan Liu, Yutong Feng, Chenxuan Miao, Zixiang Gao, Jiwang Qu, Jianzhang Zhang, Bin Wang, and Kun Yuan. Lumen: Consistent video relighting and harmonious background replacement with video generative models. arXiv preprint arXiv:2508.12945, 2025. 2
- [51] Yunke Zhang, Chi Wang, Miaomiao Cui, Peiran Ren, Xuansong Xie, Xian-Sheng Hua, Hujun Bao, Qixing Huang, and Weiwei Xu. Attention-guided temporally coherent video object matting. In Proceedings of the 29th ACM International Conference on Multimedia, pages 5128–5137, 2021. 2, 5, 9
- [52] Canyu Zhao, Yanlong Sun, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks. arXiv preprint arXiv:2502.17157, 2025. 3

