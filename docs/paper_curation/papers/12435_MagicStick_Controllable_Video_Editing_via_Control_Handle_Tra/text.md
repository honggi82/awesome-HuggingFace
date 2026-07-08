## MagicStick: Controllable Video Editing via Control Handle Transformations

Yue Ma1, Xiaodong Cun2 , Sen Liang3, Jinbo Xing5 Yingqing He1, Chenyang Qi1, Siran Chen4, Qifeng Chen1 1HKUST 2Great Bay University 3USTC 4SIAT@MMLab 5CUHK https://magic-stick-edit.github.io/

# arXiv:2312.03047v2[cs.CV]18Nov2024

A bird standing on a beach next to the ocean water and sand

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Extract

Zoom In

[Figure 10]

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

UserEditing

[Figure 26]

[Figure 27]

Zoom In

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

bird

[Figure 38]

[Figure 39]

+

→

[Figure 40]

[Figure 41]

eagle

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Zoom Out Move

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

+

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

Figure 1. Controllable video editing via modifying control handle. We present a unified framework to modify video properties (e.g., shape, size, location, motion) leveraging the manual keyframe transformations on the extracted internal control signals.

### Abstract

In detail, to keep the appearance, we inflate both the pretrained image diffusion model and ControlNet to the temporal dimension and train low-rank adaptions (LoRA) layers to fit the specific scenes. Then, in editing, we perform an inversion and editing framework. Differently, finetuned ControlNet is introduced in both inversion and generation for attention guidance with the proposed attention remix between the spatial attention maps of inversion and editing. Yet succinct, our method is the first method to show the ability of video property editing from the pre-trained text-to-image model. We present experiments on numerous examples within our unified framework. We also compare with shape-aware text-based editing and handcrafted motion video generation, demonstrating our superior temporal

Text-based video editing has recently attracted considerable interest in changing the style or replacing the objects with a similar structure. Beyond this, we demonstrate that properties such as shape, size, location, motion, etc., can also be edited in videos. Our key insight is that the keyframe’s transformations of the specific internal feature (e.g., edge maps of objects or human pose), can easily propagate to other frames to provide generation guidance. We thus propose MagicStick, a controllable video editing method that edits the video properties by utilizing the transformation on the extracted internal control signals.

†Equal contribution. Corresponding author.

consistency and editing capability than previous works.

Source Frame

Editing condition Attention map Result

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

ZoominZoomoutMoving

### 1. Introduction

Due to the remarkable progress of text-to-image (T2I) generation [28,29,52], video editing has recently achieved significant progress leveraging the generation prior of textto-image models [54]. Previous works have studied various editing effects, such as visual style transfer [3, 21, 50, 68] or modification of the character and object to a similar one [6, 27, 34]. However, many straightforward edits in videos remain out of reach currently.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

It appears feasible to edit images following the mentioned requirements as suggested by Epstein et al. [14], however, it proves challenging when applied to videos. Firstly, the editing needs to be applied to all frames individually due to the distinction among video frames. Secondly, the temporal consistency will be deteriorated if we apply frame-wise editing to the video directly. Finally, taking resizing as an example, if we directly segment the object and then edit, the generated video still relies on the abilities of additional inpainting [1, 2, 65, 66] and harmonization [11]. To handle these problems, our key insight is inspired by ControlNet [75], where the structure can be generated following additional control. Editing the control signal is relatively easier and clearer than directly editing the appearance.

Figure 2. Motivation of the proposed method. We discover that the attention map can be edited through conditional editing. Therefore, we apply conditional injection to the inversion and denoising stages to provide better information guidance.

and motion without the need for large-scale data training. This makes our method more efficient and widely applicable. Besides the mentioned attributes, we can also alter the motion in the input human video, which is also absent in previous T2I-based video editing approaches. To justify the effectiveness of our novel editing framework, we conduct extensive quantitative and qualitative experiments on various videos, demonstrating the superiority of our approach.

As shown in the Fig. 2, the results demonstrate that conditional injection plays a guiding role in the model’s attention module. To this end, we propose a universal video editing framework MagicStick. In addition to common video editing tasks (e.g., editing appearance), MagicStick is also capable of geometry editing, e.g., editing the object shape and position while maintaining its ID. To support geometry editing, we propose a two-stage ‘preserve-andedit’ strategy. Specifically, to preserve the overall appearance, we first train a controllable video generation network following [40, 63] based on the pre-trained Stable Diffusion [54] and ControlNet [75] to gain the ability to generate videos with the specific appearance of the given video. After that, we involve the edited/transformed control signals in both video inversion [44] and generation processes. Finally, we propose a novel attention remixing module, which performs editing by guiding the remixing of attention using fine-tuned ControlNet signals in both the inversion and generation processes. This dual-stage signal guidance significantly enhances the controllability of the editing process, enabling us to successfully modify specific aspects of a video, such as shape, size, and localization, as demonstrated in Fig. 1. Our method only requires adjusting a single video and does not need additional datasets for training. By editing the control signals of the video, we can achieve modifications to properties such as shape, size, location,

Our contributions are summarized as follows:

- • We demonstrate the ability of a series of important video editing aspects (including shape, size, location, and human motion) for the first time by introducing a novel unified controllable video editing framework using pre-trained T2I models.
- • We propose an effective attention remix module utilizing the attention from control signals to faithfully retain the edit-unrelated information from the source video.
- • The experiments show the advantages of the proposed method over the previous similar topics, e.g., shapeaware video editing and handcrafted motion controls.

### 2. Related Work

Video Editing. Editing natural videos is a vital task that has drawn current researchers’ attention in computer vision. Before the advent of diffusion models, many GANbased approaches [7, 16, 19, 22, 26, 39, 41–43, 47, 48, 60] have achieved good performance. The emergency of diffusion models [57] delivers higher quality and more diverse

editing results. Text2live [3] and StableVideo [6] present layer-atlas-based methods and edit the video on a flattened texture map. FateZero [50] and Video-p2p [35], guided by original and target text prompts, perform semantic editing by blending cross-attention activation maps in the denoising U-Net. There are also some approaches [8,15,17,23,37,45] edit the appearance of videos by powerful yet private video diffusion models. However, Most of these methods mainly focus on editing the texture rather than the shape, where the latter is more challenging. They show obvious artifacts even with the optimization of generative priors. In contrast, our framework can achieve the editing of complex properties, including shape, size, and location of objects, while maintaining both appearance and temporal consistency.

Image and Video Generation. Text-to-image generation is a popular topic with extensive research in recent years. Many approaches have been developed based on transformer architectures [13, 24, 25, 29, 31, 53, 67, 71, 72] to achieve textual control for generated content. However, it operates attention during generation and struggles to maintain consistency with input. Moreover, it is designed for image generation and cannot be directly applied to video editing. To address a similar problem in video generation, various works [4,20,28,36,38,40,43,61,76] try to extend image LDM [54] to video domain. These methods then generate continuous content by incorporating additional temporal layers. Tune-A-Video [63] proposes a method specializing in one-shot text-to-video generation tasks. The model has the ability to generate video with similar motion to the source video. However, how to edit real-world video content using this model is still under-exposed. Inspired by the image controllable generation methods [49,69,74,75], our method supports editing real-world video properties by leveraging pre-trained text-to-image models.

### 3. Method

We aim to edit the property changes (e.g., shape, size, location, and motion) in a video through transformations on one specific control signal as shown in Fig. 3. Below, we start with a brief introduction to latent diffusion model (LDM) and inversion in Sec. 3.1. Then, we introduce our video customization method in Sec. 3.2 to keep the appearance. Finally, we present the details of control signal transformation (Sec. 3.3) and editing during inference (Sec. 3.4).

#### 3.1. Preliminary

Latent Diffusion Models (LDMs). Derived from diffusion Models, LDMs [54] reformulate the diffusion and denoising procedures within a latent space. Firstly, an encoder E compresses a pixel space image x to a low-resolution latent z = E(x) , which can be reconstructed from latent feature to image D(z) ≈ x by decoder D. Then, a U-Net [55] εθ with self-attention [58] and cross-attention is trained to estimate

the added noise using the following objective:

0,ε∼N(0,I),t∼ Uniform (1,T) ∥ε − εθ (zt,t,p)∥22 , (1)

min

Ez

θ

where p is the embedding of the text prompt and zt is a noisy sample of z0 at timestep t. After training, we can generate clean image latents z0 from random Gaussian noises zT and text embedding p through step-by-step denoising and then decode the latents into pixel space by D.

DDIM Inversion. During inference, we can use deterministic DDIM sampling to transform a random noise zT to a clean latent z0 across a sequence of timesteps from T to 1:

√1 − αtεθ √αt

zt −

zt−1 = √αt−1

+ 1 − αt−1εθ, (2)

where αt is the parameter for noise scheduling [30, 57]. Thus, DDIM Inversion [12] is proposed to inverse the above progress from a clean latent space z0 to a noisy latent space zT by adding noises:

√1 − αt−1εθ √αt−1

+ √1 − αtεθ, (3)

zˆt = √αt

zˆt−1 −

In this context, z0 can be reconstructed by inverted latent zˆT using DDIM. By using DDIM inversion, we can edit realworld images and videos in a relatively deterministic way.

#### 3.2. Controllable Video Customization

Since the generation process of the generative model is too stochastic to keep the appearance, for our tasks, we first tune the network to satisfy our requirements, which is similar to model customization in text-to-image generation [56]. Differently, we also involve the structure-guided model to add additional correspondence between the control signal and output, which is a trainable ControlNet [75] with additional designs of temporal attention layer [63], trainable LoRA [32] and token embedding for better video customization fine-tuning. We follow the original MSE loss (Eq. 1) for customization training.

In detail, as shown in Fig. 3, the low-rank matrices [32] are injected into pre-trained linear layers within the crossattention modules of the denoising UNet. LoRA employs a low-rank learnable factorization technique to update the attention weight matrix Wq, Wk, Wv:

Wi = Wi

0

+ ∆Wi

0

= Wi

0

+ BiAi, (4)

where i = q,k,v denotes the different part in crossattention. Wi

0 ∈ Rd×k represents the original weights of the pre-trained T2I model [54], B ∈ Rd×r and A ∈ Rr×k represent the low-rank factors, where r is much smaller than original dimensions d and k. Remarkably, this operation does not affect the ability of the pre-trained T2I model [54] to generate and compose concepts because

|[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>original target<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>Propagate<br><br>1st Frame Editing<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>edited conditon| |
|---|---|
| | |

###### A <new1> cup is placed upright on the table.

Text encoder

Structure Guided

DDIM Inversion

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

| | |
|---|---|
| | |
| | |

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

×T steps

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

Inverted Latents

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

| | |
|---|---|
| | |
| | |

[Figure 144]

[Figure 145]

ST Attn

[Figure 146]

[Figure 147]

[Figure 148]

| | |
|---|---|
| | |

×T steps

Cross Attn

DDIM Denoising

Structure Guided

FFN

Structure Guided

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Conv Layers (frozen) Attention ReMix module

With trainable Temporal Layer

[Figure 154]

Trainable token

With trainable customized LoRA

Spatio-Temporal Self-Attention Unet Block (frozen)

edited conditon

- Figure 3. Overview of MagicStick. We propose a controllable video editing method that edits the video properties by utilizing the transformation on the extracted internal control signals. To achieve this, on the left, we tune the given video based on the extracted control signals and the text prompt for video customization. On the right, we first edit the key frame manually and propagate it to other frames as guidance. Then, relying on the proposed attention remix module, we can achieve controllable video editing using structure guidance during inversion and denoising.

LoRA introduces minimal, low-rank updates that preserve the core knowledge and structure of the model. We also train a token embedding to get better customization following [18]. In addition, we adopt inflated ControlNet [75] inspired by Tune-A-Video [63] to include temporal information among the control signal sequences. The original control signal (e.g., sketch or edge, depending on the editing needs in the following sections) of the object is encoded by a structure-guided module, where we convert the original self-attention module in text-to-image diffusion U-Net backbone to spatio-temporal self-attention.

ters in the first edited signal frame. In detail, we detect the bounding box of the objects in the first original control signal frame and edited one. The movement of the bounding box’s center point is calculated as the position parameter, and the changes in length and width are used as the resize parameters. Then, we apply these transformations across all frames, ensuring consistent guidance throughout the video.

#### 3.4. Controllable Video Editing

With guidance and the appearance customization, we can finally edit the video in our framework. Specifically, we conduct video inversion to obtain the intermediate features and then inject these features during the denoising stage, as shown in Fig. 3 (right). Unlike the previous unconditional video inversion and editing pipeline [50], our method guides both processes with the control signals and then performs editing via the proposed attention remix modules.

#### 3.3. Control Signal Transformation

Since naive object editing in a video may deteriorate the surrounding background, we opt to edit the control signal of the first frame to ensure more consistent editing results. Then, the edited control signals are incorporated into UNet after being encoded by ControlNet-based [75] structure guided module. As shown in Fig. 3 (right), our control signal editing involves three steps.

Controllable Inversion and Generation. Most existing works [5, 33, 44] conduct inversion followed by editing to achieve appearance editing. While in our task, we aim to achieve more complex property editing by incorporating the control signals during the inversion and generation. Specifically, the same edited signal frame in two stages is employed for different purposes. During inversion, we use the structure-guided model to encode the edited signal frames and add them into the U-Net as residuals. This operation can easily inject the shape and position information of the edited signals into the self-attention and cross-attention processes of inversion of the source video. As shown in Fig. 4, taking the moving editing operation as an example,

- 1) Extraction. We first segment the interest objects using a text-guided Segment-and-Track-Anything. [9]. Then the annotator (e.g., pose detector, HED-detector, and depth estimator) is utilized to extract corresponding control signals from the clean segmented object sequences.
- 2) Transformation. After getting the intermediate representations of the specific frame, the user can perform single or multiple geometric editing operations on objects in the first signal frame, such as resizing and moving.
- 3) Propagation. We compute the transformation parame-

### 4. Experiments

[Figure 155]

Inversion

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

#### 4.1. Implementation Details

[Figure 174]

Inverted Latents

Latents

| | |
|---|---|
| | |

Self attn

We implement our method based on the pre-trained Stable Diffusion (SD) [54] and ControlNet [75] at 100 iterations. We sample 8 uniform frames at the resolution of 512 × 512 from the input video and fine-tune the models for about 5 minutes on a single 24G NVIDIA RTX 3090Ti GPU. The time cost of training stage is cheap.

Background

Structure 𝑀𝑡 Guidance

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

1-Mt

Cross attn

Edited condition

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

Inverted Latents

Inverted Latents

Self attn

Foreground Fused attention

Denoising

#### 4.2. Applications

Object size modification. Using the pre-trained text-toimage diffusion model [54], our method supports object size modification by manually modifying the specific sketch in the first frame. As shown in the 1st and 2nd rows of Fig. 5, our method achieves consistent editing of the foreground “swan” by scaling down its sketch, while preserving the original background content.

- Figure 4. Attention ReMix module. Taking the moving editing as an example, we utilize edited conditions in both inversion and denoising. During inversion, the edited condition Cedit is employed to edit self-attention and cross attention from input video and get binary mask Mt. While for denoising, Cedit is leveraged to guide the appearance generation in the target region.

Object position editing. One of the applications of our method is to edit the position of objects by adjusting the object’s position in the first frame of the control signal. This task is challenging because the model needs to generate the object at the target area while inpainting the original area simultaneously. As demonstrated in the 3rd and 4th rows of Fig. 5, thanks to the proposed Attention ReMix module, our approach enables moving the “parrot” from the right of the branch to left by shifting the position of the parrot sketch in the 1st signal frame. The background and the object successfully remain consistent across different video frames.

we observe that both the original and target positions are activated by words “cup” after injection in cross-attention. During generation, the edited signal frame is reintroduced into the network to serve as guidance for regenerating specific highlighted regions. In both stages, the injection of the edited signal frame is indispensable to accomplish our editing tasks through their mutual coordination.

Attention Remix Module. Adopting only the guidance of structure in inversion cannot achieve our novel editing tasks, as shown in the 3rd column of the editing results in Fig.7. To achieve our ultimate goal, we propose the Attention ReMix module to modify the attention in both inversion and denoising processes. As shown in Fig. 4, we store the intermediate self-attention maps {ssrct }Tt=1 and crossattention maps {csrct }Tt=1 at each timestep t and the last noisy latent maps zT as:

Human motion editing. Our approach is also capable of editing human motion by replacing the source skeleton signals with the target sequences extracted from other videos. For instance, we can modify man’s motion from “jumping” to “raising hand”, as shown in the 5th and 6th rows in Fig. 5. The result indicates that we can generate new content using target pose sequences while maintaining the human appearance in the source video (e.g., white shoes and black pants).

zT,{csrct }Tt=1,{ssrct }Tt=1 = Inv z0,Cedit , (5)

where Inv donates the DDIM inversion pipeline and Cedit represents edited signal frame. As shown in Fig. 4, during the denoising stage, the activation areas of cross-attention by words “cup” provide significant assistance for binary mask Mt generation, which is obtained by thresholding the cross-attention map {csrct }Tt=1. Then we blend the {ssrct }Tt=1 with Mt and use the structure-guided module to generate the target object. Formally, this process can be represented as

Object appearance editing. Thanks to the rich knowledge learned in the per-trained T2I models [54], we can modify the object appearance through the editing of source text prompts, which is similar to existing mainstream video editing methods. Differently, we can further fulfill the property and appearance editing simultaneously. The 7th and 8th rows in Fig. 5 showcase this powerful and intriguing capability. We replace “bear” with “lion” simply by modifying the corresponding words and enlarging its sketch.

Mt = GetMask(csrct ,τ), sfinalt = G Mt ⊙ ssrct ,Cedit + (1 − Mt) ⊙ ssrct ,

#### 4.3. Comparisons

(6)

We compare our method with other video editing approach, including Shape-aware Edit [34], Tune-AVideo [63], StableVideo [6], FateZero [50] and Flatten [10]. We report the qualitative and quantitative results, respec-

where τ stands for the threshold value, GetMask(·) represents the operation to get mask using a specific word, and G(·) denotes structure guidance.

[Figure 212]

A <new1> swan with a red beak swimming in a river near a wall and bushes.

|[Figure 213]|
|---|

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

A <new1> parrot sitting on a tree branch

|[Figure 238]|
|---|

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

A <new1>man standing on a blue mat, a lake in the background, full body pose

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

A polar bear standing on a frozen lake in the snow

|[Figure 282]|
|---|

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

polar bear → lion

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

- Figure 5. Controllable video editing via modifying the control signals (e.g., sketch map and skeleton). Our framework can achieve consistent editing via the transformation of control signals while keeping its original appearance. We present the editing results of resizing (small), moving (transformation), human pose editing, and shape-aware text-based editing from top to bottom. We can further introduce text-based editing as the bottom sample to change both appearance and shape. Zoom in for the best view.

##### A duck with a red beak swimming in a river near a wall and bushes.

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

Image

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

Motion

[Figure 345]

Input Frames Ours Shape-aware Edit Tune-A-Video&T2I-adapter Framewise& PastingInpainting StableVideo FateZero Flatten

Ours VideoComposer

DragNUWA

- Figure 6. Qualitative comparison. Left: Shape-aware video editing. we resize the swan and replace word “swan” with “duck”. Right: Moving object, the first frame and motion conditions are provided to reconstruct the video. Our results exhibit the best temporal consistency, video fidelity, and editing quality. Zoom in for the best view.

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

W/o SG in inversion

W/o SG in denoising

W/o SG

Input Frame Ours in inversion & denoising

[Figure 356]

[Figure 357]

[Figure 358]

|[Figure 359]<br><br>[Figure 360]<br><br>[Figure 361]<br><br>[Figure 362]<br><br>[Figure 363]|
|---|

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

|[Figure 370]|
|---|

|[Figure 371]|
|---|

[Figure 372]

|[Figure 373]|
|---|

|[Figure 374]|
|---|

|[Figure 375]|
|---|

|[Figure 376]|
|---|

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Signal Frame Editing

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

- Figure 7. Ablation about Attention ReMix module. SG indicates the structure guidance model. The sub-figures located in the upper-left corner of each editing result are the visualization of cross-attention activation maps during denoising, while the mask figures 1 − Mt in the bottom-left corner are generated by the word activation of cross-attention in inversion. If we remove the structure-guided module in inversion or denoising from our model, it fails to complete the moving task (“✓” represents the successful completion of the task, while “×” indicates the failure).

tively. The specific editing task for comparison is simultaneously editing the shape and appearance.

duce video results with natural appearance and better temporal consistency.

Qualitative results. We present various results from different open-source approaches. As shown in the 3rd and 5th columns in Fig 6. The first is Shape-aware editing [34], its appearance is still limited by the optimization, causing blur and unnatural results. And the second one is to segment the objects and direct inpainting [73] and paste directly. However, we find this naive methods struggles with the video harmonization and temporal artifacts caused by inpainting and segmentation. Another relevant method is using TuneA-Video [63] with T2I-Adapter [46]. It accomplishes local editing after overfitting but still exhibits temporal inconsistency. We also compare the method with [6, 10, 50], they are difficult to generate realistic video while modifying the shape. Differently, our proposed methods manage to resize the “swan”,“truck” and edit its appearance to “duck”, “train” simultaneously, while maintaining the temporal coherence. On the other hand, we compare our method on conditional video generation from handcrafted motion signals and a single appearance. As shown in Fig. 6, VideoComposer [62] struggles to generate temporally consistent results, while DragNUWA [70] fails to preserve the visual details of the input image. In contrast, our method can pro-

Quantitative results. We also conduct quantitative comparisons with the following metrics: 1) Tem-Con (Temporal Consistency): Following the previous methods [15,59,64], we evaluate temporal consistency of the generated video frames by calculating the cosine similarity between all pairs of consecutive frame embeddings of CLIP [51] image encoder. 2) Fram-Acc (Frame Accuracy): For object appearance editing, we measure the frame-wise editing accuracy based on the editing text prompt, which is the percentage of frames where the edited frames have a higher CLIP similarity to the target prompt than the source prompt. 3) Four user studies metrics: Following FateZero [50], we assess our approach using four user studies metrics (“Edit”, “Image”, “Temp” and “ID”). They are editing quality, overall framewise image fidelity, temporal consistency of the video, and object appearance consistency, respectively. For a fair comparison, we ask 30 subjects to rank different methods. Each study displays four videos in random order and requests the evaluators to identify the one with superior quality. As demonstrated in Tab. 1, the proposed method achieves the best Tem-Con and Fram-Acc against baselines. As for the user studies, the average ranking of our method earns user

Table 1. Quantitative evaluation on video editing against baselines. The best results are marked in bold.

Method CLIP Metrics↑ User Study↓ Inversion & Editing Tem-Con Fram-Acc Edit Image Temp ID

Tune-A-Video [63] & T2I-adapter [46] 0.891 0.851 2.87 2.99 2.67 3.18 Shape-aware Edit [34] 0.722 0.618 3.56 3.69 3.74 3.53 VideoComposer [62] 0.914 0.846 3.38 2.74 2.94 4.31 Single-frame Inpainting & Pasting 0.920 0.887 3.92 3.97 4.08 1.96 StableVideo [6] 0.922 0.902 2.74 3.97 3.87 2.95 FateZero [50] 0.917 0.884 3.92 2.42 3.16 2.74 Flatten [10] 0.919 0.913 3.92 2.26 2.52 2.87

###### Ours 0.928 0.919 1.27 1.62 1.58 2.04

w/o token embedding tuning

w/o LoRA tuning in cross-attention

Input Video Ours

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

|[Figure 393]|
|---|

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]|
|---|

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

|[Figure 412]|
|---|

|[Figure 413]|
|---|

|[Figure 414]|
|---|

|[Figure 415]|
|---|

- Figure 8. Ablation on Video Customization.. Without LoRA tuning in cross-attention, the generated video can not preserve the object’s content. Similarly, the appearance of the object (e.g., texture and color) experiences a degradation in content consistency when token embedding tuning is absent.

preferences the best in three aspects and the comparable performance in ID metric.

#### 4.4. Ablation Studies

Video Customization via LoRA and token embedding tuning. We present the visual results of our variants, i.e., ours without LoRA in cross-attention or token embedding tuning in the 3rd and 4th column of Fig. 8, respectively. The result in the 3rd column illustrates the significance of crossattention LoRA for preservation of object appearance. Additionally, we notice a deterioration in performance when token embedding remains untuned, which further underscores the significance of this process in content preservation. In contrast, our full method guarantees the preservation of texture and appearance consistency by meticulously fine-tuning both components.

Attention Remix Module. We further investigate the effectiveness of the proposed Attention Remix module. The visual results of the ablation experiments are shown in Fig. 7,

Table 2. Quantitative evaluation about ablation study. The best results are marked in bold.

Method CLIP Metrics↑ User Study↓ Inversion & Editing Tem-Con Fram-Acc Edit Image Temp ID

w/o LoRA tuning in cross-attention 0.891 0.884 3.27 3.34 2.91 3.97 w/o token embedding tuning 0.908 0.915 2.65 3.09 2.63 3.82 w/o spatio-temporal self-attention 0.752 0.869 4.34 3.71 4.09 3.41 w/o temporal layer 0.884 0.893 3.52 3.12 3.92 2.40

Ours 0.931 0.922 1.23 1.74 1.46 1.41

where we remove the structure-guided module to ablate its role during inversion and generation. As presented in the 2rd column, removing the structure-guided module in inversion leads to the failure of moving editing. Since target area mask (bottom left corner) is missing, the self-attention in this region entirely derives from that of the background. The 4th column demonstrates the results of an absence of structure guidance during generation. This variant also fails to achieve the moving editing task even with the target area mask. This is because there is no guidance for the target area during the generation (upper-left corner of the 4th column). Finally, we also show the result produced by the variant by removing the attention remix module in both two stages. This ablated framework severely degrades to be a self-attention reconstruction. As shown in the 5th column, this variant fails to accomplish the moving task and maintain background consistency (green rectangles in the 5th column). In contrast, when we equip the guidance both in inversion and generation, the “cup” can be shifted successfully, further emphasizing our module’s significance. when we remove the Attention ReMix module, the position of cup is not changed, which evident the importance of proposed module. Note that the binary mask Mt, obtained by mapping between word and cross attention, is accurate since the utilization of powerful pretrained T2I model [54].

### 5. Conclusion

In this paper, we propose a new controllable video editing method MagicStick that performs temporally consistent video property editing such as shape, size, location, motion, or their combinations. To the best of our knowledge, our method is the first to demonstrate the capability of editing shape, size and localization of objects in videos using a pre-trained text-to-image model. To achieve this, we make an attempt to study and utilize the transformations on one control signal (e.g., edge maps of objects or human pose) using customized ControlNet. A novel Attention ReMix module is further proposed for more complex video property editing. Our framework leverages pre-trained image diffusion models for video editing, which we believe will contribute to numerous potential video applications.

Acknowledgments. We thank Jiaxi Feng, Yabo Zhang for their helpful comments. This project was supported by the National Key R&D Program of China under grant number 2022ZD0161501.

### References

- [1] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Transactions on Graphics (TOG), 42(4):1–11, 2023. 2
- [2] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 2
- [3] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision, pages 707–723. Springer, 2022. 2, 3
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.

- 4

[6] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050, 2023. 2, 3,

- 5, 7, 8

- [7] Qihua Chen, Yue Ma, Hongfa Wang, Junkun Yuan, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen, and Wei Liu. Follow-your-canvas: Higher-resolution video outpainting with extensive content generation. arXiv preprint arXiv:2409.01055, 2024. 2
- [8] Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real image animation with text-guided motion control. In European Conference on Computer Vision, pages 475–491. Springer, 2025. 3
- [9] Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. Segment and track anything. arXiv preprint arXiv:2305.06558, 2023. 4
- [10] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. Flatten: optical flowguided attention for consistent text-to-video editing. arXiv preprint arXiv:2310.05922, 2023. 5, 7, 8
- [11] Xiaodong Cun and Chi-Man Pun. Improving the harmony of the composite image by spatial-separated attention module. IEEE Transactions on Image Processing, 29:4759– 4771, 2020. 2

- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [13] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022. 3
- [14] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. arXiv preprint arXiv:2306.00986,

2023. 2

- [15] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 3, 7
- [16] Chengyu Fang, Chunming He, Fengyang Xiao, Yulun Zhang, Longxiang Tang, Yuelin Zhang, Kai Li, and Xiu Li. Real-world image dehazing with coherence-based label generator and cooperative unfolding network. arXiv preprint arXiv:2406.07966, 2024. 2
- [17] Kunyu Feng, Yue Ma, Bingyuan Wang, Chenyang Qi, Haozhe Chen, Qifeng Chen, and Zeyu Wang. Dit4edit: Diffusion transformer for image editing. arXiv preprint arXiv:2411.03286, 2024. 3
- [18] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 4
- [19] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clipguided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41(4):1–13, 2022. 2
- [20] Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, and Wei Liu. Inducing high energy-latency of large vision-language models with verbose images. In ICLR, 2024. 3
- [21] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 2
- [22] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [23] Hang Guo, Tao Dai, Zhihao Ouyang, Taolin Zhang, Yaohua Zha, Bin Chen, and Shu-tao Xia. Refir: Grounding large restoration models with retrieval augmentation. arXiv preprint arXiv:2410.05601, 2024. 3
- [24] Chunming He, Kai Li, Yachao Zhang, Longxiang Tang, Yulun Zhang, Zhenhua Guo, and Xiu Li. Camouflaged object detection with feature decomposition and edge reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22046–22055,

2023. 3

- [25] Chunming He, Kai Li, Yachao Zhang, Guoxia Xu, Longxiang Tang, Yulun Zhang, Zhenhua Guo, and Xiu Li. Weaklysupervised concealed object segmentation with sam-based

- pseudo labeling and multi-scale feature grouping. arXiv preprint arXiv:2305.11003, 2023. 3
- [26] Tianyu He, Junliang Guo, Runyi Yu, Yuchi Wang, Jialiang Zhu, Kaikai An, Leyi Li, Xu Tan, Chunyu Wang, Han Hu, et al. Gaia: Zero-shot talking avatar generation. arXiv preprint arXiv:2311.15230, 2023. 2
- [27] Yingqing He, Menghan Xia, Haoxin Chen, Xiaodong Cun, Yuan Gong, Jinbo Xing, Yong Zhang, Xintao Wang, Chao Weng, Ying Shan, et al. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint arXiv:2307.06940, 2023. 2
- [28] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 2, 3
- [29] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 3
- [30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [31] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [32] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [33] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Direct inversion: Boosting diffusion-based editing with 3 lines of code. arXiv preprint arXiv:2304.04269,

2023. 4

- [34] Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, and Jia-Bin Huang. Shape-aware text-driven layered video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14317–14326, 2023. 2, 5, 7, 8
- [35] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023. 3
- [36] Yunfan Liu, Qi Li, Qiyao Deng, and Zhenan Sun. Towards spatially disentangled manipulation of face images with pretrained stylegans. IEEE Transactions on Circuits and Systems for Video Technology, 33(4):1725–1739, 2022. 3
- [37] Zichen Liu, Yue Yu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Wen Wang, Zhiheng Liu, Qifeng Chen, and Yujun Shen. Magicquill: An intelligent interactive image editing system. arXiv preprint arXiv:2411.09703, 2024. 3
- [38] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jinren Zhou, and Tieniu Tan. Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023. 3

- [39] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4117–4125, 2024. 2
- [40] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186, 2023. 2, 3
- [41] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Chenyang Qi, Chengfei Cai, Xiu Li, Zhifeng Li, HeungYeung Shum, Wei Liu, et al. Follow-your-click: Opendomain regional image animation via short prompts. arXiv preprint arXiv:2403.08268, 2024. 2
- [42] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. arXiv preprint arXiv:2406.01900, 2024. 2
- [43] Yue Ma, Yali Wang, Yue Wu, Ziyu Lyu, Siran Chen, Xiu Li, and Yu Qiao. Visual knowledge graph for human action reasoning in videos. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4132–4141, 2022. 2, 3
- [44] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 2, 4
- [45] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 3
- [46] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 7, 8
- [47] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2337–2346,

2019. 2

- [48] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2085–2094,

2021. 2

- [49] Bo Peng, Hongxing Fan, Wei Wang, Jing Dong, and Siwei Lyu. A unified framework for high fidelity face swap and expression reenactment. IEEE Transactions on Circuits and Systems for Video Technology, 32(6):3673–3684, 2021. 3
- [50] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv:2303.09535, 2023. 2, 3, 4, 5, 7, 8
- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 7
- [52] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 2
- [53] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 3
- [54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 5, 8
- [55] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 3
- [56] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3
- [57] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 3
- [58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [59] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023. 7
- [60] Jiangshan Wang, Yue Ma, Jiayi Guo, Yicheng Xiao, Gao Huang, and Xiu Li. Cove: Unleashing the diffusion feature correspondence for consistent video editing. arXiv preprint arXiv:2406.08850, 2024. 2
- [61] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 3
- [62] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 7, 8
- [63] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning

- of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 2, 3, 4, 5, 7, 8
- [64] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. arXiv preprint arXiv:2306.00943, 2023. 7
- [65] Shunxin Xu, Dong Liu, and Zhiwei Xiong. E2i: Generative inpainting from edge to image. IEEE Transactions on Circuits and Systems for Video Technology, 31(4):1308–1322,

2020. 2

- [66] Han Yan, Haijun Zhang, Jianyang Shi, and Jianghong Ma. Texture brush for fashion inspiration transfer: A generative adversarial network with heatmap-guided semantic disentanglement. IEEE Transactions on Circuits and Systems for Video Technology, 2022. 2
- [67] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3
- [68] Shuai Yang, Yifan Zhou, Ziwei Liu, , and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In ACM SIGGRAPH Asia Conference Proceedings, 2023. 2
- [69] Zuopeng Yang, Tianshu Chu, Xin Lin, Erdun Gao, Daqing Liu, Jie Yang, and Chaoyue Wang. Eliminating contextual prior bias for semantic image editing via dual-cycle diffusion. IEEE Transactions on Circuits and Systems for Video Technology, 2023. 3
- [70] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 7
- [71] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 3
- [72] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 3
- [73] Tao Yu, Runseng Feng, Ruoyu Feng, Jinming Liu, Xin Jin, Wenjun Zeng, and Zhibo Chen. Inpaint anything: Segment anything meets image inpainting. arXiv preprint arXiv:2304.06790, 2023. 7
- [74] Wei Yu, Yanping Li, Rui Wang, Wenming Cao, and Wei Xiang. Pcfn: progressive cross-modal fusion network for human pose transfer. IEEE Transactions on Circuits and Systems for Video Technology, 2022. 3
- [75] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2, 3, 4, 5
- [76] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 3

