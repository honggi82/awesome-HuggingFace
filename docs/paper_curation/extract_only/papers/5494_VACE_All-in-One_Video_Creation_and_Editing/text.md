## VACE: All-in-One Video Creation and Editing

Zeyinzi Jiang* Zhen Han∗ Chaojie Mao∗† Jingfeng Zhang Yulin Pan Yu Liu Tongyi Lab, Alibaba Group

[Figure 1]

Reference-to-video generation

[Figure 2]

| |
|---|

[Figure 3]

|[Figure 4]<br><br>[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

# arXiv:2503.07598v2[cs.CV]11Mar2025

[Figure 17]

[Figure 18]

[Figure 19]

###### Video-to-video editing

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

[Figure 26]

[Figure 27]

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

[Figure 38]

[Figure 39]

Pose Transfer (Pose Control) Motion Transfer (Layout Control)

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

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

Structure Transfer (Depth Control) Colorization (Gray Control)

Masked Video-to-video editing

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

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

Subject Recreation (Inpainting)

Subject Removal (Inpainting)

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

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

Canvas Extension (Outpainting)

Temporal Extension (First Clip)

###### Task Composition

[Figure 98]

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

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

Swap Anything (Subject Reference + Inpainting) Animate Anything (Frame Reference + Pose Control)

Figure 1. Comprehensive capabilities of VACE. We present the outstanding generation results based on Wanx2.1 (left) and LTX-Video (right). For each task, the original input image or video (left), the context video (top left corner), and the generated frames are illustrated.

### Abstract

integrate the requirements of various tasks by organizing video task inputs, such as editing, reference, and masking, into a unified interface referred to as the Video Condition Unit (VCU). Furthermore, by utilizing a Context Adapter structure, we inject different task concepts into the model using formalized representations of temporal and spatial dimensions, allowing it to handle arbitrary video synthesis tasks flexibly. Extensive experiments demonstrate that the unified model of VACE achieves performance on par with task-specific models across various subtasks. Simultaneously, it enables diverse applications through versatile task combinations. Project page: https://alivilab.github.io/VACE-Page/.

Diffusion Transformer has demonstrated powerful capability and scalability in generating high-quality images and videos. Further pursuing the unification of generation and editing tasks has yielded significant progress in the domain of image content creation. However, due to the intrinsic demands for consistency across both temporal and spatial dynamics, achieving a unified approach for video synthesis remains challenging. We introduce VACE, which enables users to perform Video tasks within an All-in-one framework for Creation and Editing. These tasks include reference-to-video generation, video-to-video editing, and masked video-to-video editing. Specifically, we effectively

*Equal Contribution. †Project lead.

### 1. Introduction

In recent years, the domain of visual generation tasks has witnessed remarkable advancements, driven in particular by the rapid evolution of diffusion models [24, 25, 48, 53, 54, 56, 57]. Beyond the early foundational pre-trained models for text-to-image [7, 16, 33] or text-to-video [9, 22, 64] generation in the field, there has been a proliferation of downstream tasks and applications, such as repainting [3, 82], editing [4, 42, 68, 70, 75], controllable generation [30, 76], frame reference generation [20, 73], and ID-referenced video synthesis [11, 35, 47, 74]. This array of developments highlights the dynamic and complex nature of the visual generation field. To enhance task flexibility and reduce the overhead associated with deploying multiple models, researchers have begun to focus on constructing unified model architectures [12, 63] (e.g., ACE [23, 41] and OmniGen [71]), aiming to integrate different tasks into a single image model, facilitating the creation of various application workflows while maintaining simplicity in usage. In the field of video, due to the collaborative transformations in both temporal and spatial dimensions, leveraging a unified model can present endless possibilities for video creation. However, leveraging diverse input modalities and ensuring spatiotemporal consistency are still chanlleging for unified video generation and editing.

We propose VACE, an all-in-one model for video creation and editing that performs tasks including referenceto-video generation, video-to-video editing, masked videoto-video editing, and free composition of these tasks, as illustrated in Fig. 1. On one hand, the aggregation of various capabilities reduces the costs of service deployment and user interaction. On the other hand, by combining the capabilities of different tasks within a single model, it addresses challenges faced by existing video generation models such as controllable generation of long videos, multi-condition and reference based generation, and continuous video editing, thereby empowering users with greater creativity. To achieve this, we utilize the current mainstream Diffusion Transformers (DiTs) structure as the foundational video framework and pre-trained text-to-video generation models [22, 64], which provides better basic capabilities and scalability for handling long video sequences. Specifically, VACE takes into account the needs of different tasks during its construction and designs a unified interface, dubbed the Video Condition Unit (VCU), which integrates multiple modalities such as images or videos for editing, references, and masks. Furthermore, to differentiate the visual modality information in editing and reference tasks, we introduce the concept decoupling strategy, enabling the model to understand what aspects need to be retained and what should be modified. Meanwhile, by employing a pluggable Context Adapter structure, concepts from different tasks (e.g., the areas or ranges of editing or reference) are injected into the

model through collaborative spatiotemporal representation, enabling it to possess the capability of adaptive processing for unified tasks.

Due to the lack of existing multi-task benchmarks in video synthesis, we construct a dataset of 480 evaluation samples containing 12 different tasks, while evaluating the performance of the VACE unified model by comparing it with existing specialized models. Experimental results demonstrate that our framework exhibits sufficient competitiveness in both quantitative and qualitative analyses. To the best of our knowledge, we are the first all-in-one model based on the video DiT architecture that concurrently supports such a wide range of tasks. Notably, this innovative framework allows for the compositional expansion of base tasks, enabling the construction of scenarios such as long video re-rendering, which provides a versatile and efficient solution for video synthesis, opening new possibilities for user-side video content creation and editing.

### 2. Related Work

Visual Generation and Editing. With the rapid development of image [2, 7, 16, 18, 58, 59] and video [22, 32, 73, 77] generation models, they are being used to create high-quality visual content and are widely applied in fields such as advertising, film special effects, game development, and animation production [13, 43–45, 55]. Meanwhile, to meet the diverse needs of visual media production and to enhance efficiency and quality, precise generation and editing methods have emerged. Models are required to perform generative creation based on multimodal inputs, such as depth, structure, pose, scene, and characters. According to the purposes of the input conditions, we can categorize them into two types: editing of the input and concept-guided re-creation. A significant portion of the work, such as ControlNet [76], ControlVideo, Composer [26], VideoComposer [68], and SCEdit [30], focuses on single-condition editing and multi-condition composite editing based on temporal and spatial alignment conditions. Additionally, some works that focus on interactive local editing scenarios, such as DragGAN [46] and MagicBrush [75]. Methods that guide generation based on semantic information from the input, such as Cone [38], Cone2 [39], InstantID [67], and PuLID [21], can achieve conceptual understanding of the input and inject it into the model for creative purposes.

Task-unified Visual Generative Model. As the complexity and diversity of user creations increase, relying solely on a single model or a complicated chain of multiple models can no longer provide a convenient and efficient path for implementing creative ideas. In image generation, a unified generation and editing framework has begun to emerge, allowing for more flexible creative approaches. Methods such as UltraEdit [81] and SEED-Data-Edit [19] provide

R2V V2V MV2V

Subject Control

###### Repaint

Extension

T2V

Frame

General

[Figure 122]

Depth Gray Scribble Pose Flow Layout Inpaint Outpaint

Face Object

First / Last Clip

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

###### Composition

Reference Anything Move Anything Animate Anything Swap Anything Expand Anything

Figure 2. Task categories covered by VACE. Four basic tasks can be combined to create a vast number of possibilities.

general-purpose editing datasets, while techniques like InstructPix2Pix [4], MagicBrush [61], and CosXL [60] offer general instruction-based editing features. Additionally, methods like UniControl [50] and UNIC-Adapter [15] have unified controllable generation. Further advancements have led to the development of ACE [23, 41], OmniGen [71], OmniControl [63], and UniReal [12], which expand the scope of tasks by providing flexible controllable generation, local editing, and reference-guided generation. In the video domain, due to the increased difficulty of generation, approaches often manifest as single-task single-model frameworks, offering capabilities for editing or reference generation, as seen in Video-P2P [37], MagicEdit [34], MotionCtrl [69], Magic Mirror [80], and Phantom [35]. VACE aims to fill the gap for a unified model within the video domain, providing possibilities for complex creative scenarios.

### 3. Method

VACE is designed as a multimodal-to-video generation model, where text, image, video, and mask are integrated into a unified conditioning input. To cover as many video generation and editing tasks as possible, we conduct indepth research into existing tasks, then divide them into

##### 4 categories according to their individual requirements of multimodal inputs. Without losing generality, we specifically design a novel multimodal input format for each category under a Video Condition Unit (VCU) paradigm. Finally, we restructure the DiT model for VCU inputs, making it a versatile model for a wide range of video tasks. 3.1. Multimodal Inputs and Video Tasks.

Despite existing video tasks being varying in complex user inputs and ambitious creative goals, we found that most of their inputs can be fully represented in 4 modalities: text, image, video, and mask. Overall, as illustrated in Fig. 2, we group these video tasks into 5 categories based on their requirements of these four multimodal inputs.

- • Text-to-Video Generation (T2V) is a basic video creation task and text is the only input.
- • Reference-to-Video Generation (R2V) requires additional images as reference inputs, making sure that specified contents, such as subjects of faces, animals and other objects, or video frames, appear in the generated video.
- • Video-to-Video Editing (V2V) makes an entire change to a provided video, such as colorization, stylization, controllable generation, etc. We use video control types whose control signals can be represented and stored as RGB videos, including depth, grayscale, pose, scribble, optical flow, and layout; however, the method itself is not limited to these.
- • Masked Video-to-Video Editing (MV2V) makes changes to an input video only within the provided 3D regions of interest (3D ROI), seamlessly blending in with the other unchanged regions, such as inpainting, outpainting, video extension, etc. We use an extra spatiotemporal mask to represent the 3D ROI.
- • Task Composition includes all the compositional possibilities of the 4 types of video tasks above.

#### 3.2. Video Condition Unit

We propose an input paradigm, Video Condition Unit (VCU) to unify diverse input conditions into textual input, frame sequence, and mask sequence. A VCU can be denoted as

###### V = [T;F;M], (1)

where T is a text prompt, while F and M are sequences of context video frames {u1,u2,...,un} and masks {m1,m2,...,mn} respectively. Here, u is in RGB space, normalized to [−1,1] and m is binary, in which “1”s and “0”s symbolize where to edit or not. F and M are aligned both in spatial size h × w and temporal size n. In T2V, no context frame or mask is required. To keep generality, we assign default value 0h×w to each u denoting empty input, and set every m to 1h×w meaning that all these 0-valued

"

❄

"

VCU

Video Tokens Context Tokens

Text Tokens

Noisy Latent

Context Tokens

Concept Decouple

| | | |
|---|---|---|
| | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 134]

[Figure 135]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

A man riding a horse…

[Figure 136]

[Figure 137]

| | | | | | |
|---|---|---|---|---|---|

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Text Tokens

Video Tokens

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

| |
|---|

| |
|---|

| |
|---|

[Figure 154]

DiT Block 1

DiT Block 1

- Context Block 1

[Figure 155]

"

- Context Block 2

[Figure 156]

[Figure 157]

[Figure 158]

Tokenizers

Fc

M

Reactive Inactive Fk Mask

DiT Block 2

DiT Block 2

[Figure 159]

[Figure 160]

Video Text Context

Context Latent Encode

[Figure 161]

[Figure 162]

DiT Block 3

DiT Block 3

Transformer Blocks

…

…

…

[Figure 163]

Context Block M

DiT Block N

DiT Block N

[Figure 164]

[Figure 165]

[Figure 166]

|[Figure 167]<br><br>[Figure 168]<br><br>| |
|---|---|
| | |

[Figure 169]

[Figure 170]

|[Figure 171]|
|---|

[Figure 172]

VAE Decoder

[Figure 173]

Context Embedder

[Figure 174]

(a) Fully Fine-tuning (b) Context Adapter Tuning

- Figure 3. Overview of VACE Framework. Frames and masks are tokenized through Concept Decoupling, Context Latent Encode and Context Embedder. To achieve training with VCU as input, we employ two strategies, (a) Fully Fine-tuning and (b) Context Adapter Tuning. The latter converges faster and supports pluggable features.

- Table 1. The formal representation of frames (Fs) and masks (Ms) under the four basic tasks. Frames and masks are aligned spatially and temporally.

Tasks Frames (Fs) & Masks (Ms) T2V

F = {0h×w} × n M = {1h×w} × n

F = {r1,r2,...,rl} + {0h×w} × n M = {0h×w} × l + {1h×w} × n V2V

R2V

F = {u1,u2,...,un} M = {1h×w} × n MV2V

F = {u1,u2,...,un} M = {m1,m2,...,mn}

pixels are about to be re-generated. For R2V, additional reference frames ri are inserted in front of the default frame sequence, while all-zero masks 0h×w are inserted in front of the mask sequence. These all-zero masks mean that the corresponding frames should be kept unchanged. In V2V, context frame sequence is the input video frames and context mask is a sequence of 1h×w. For MV2V, both context video and mask are required. The formal mathematical representations are shown in Tab. 1.

VCU can also support task composition. For example, the context frames of reference-inpainting task are {r1,r2,...,rl,u1,u2,...,un} and the context masks are

- {0h×w} × l + {m1,m2,...,mn}. In this case, users can modify l objects in the video and regenerate based on the provided reference images. For another example, users only has a scribble image and wants to generate a video begining with the contents described by this scribble image, which is a scribble-based video extension task. The context frames are {u} + {0h×w} × (n − 1) and the context masks are
- {1h×w} × n. In this way, we can achieve multi-condition and reference control generation for long videos.

#### 3.3. Arichitecture

We restructure the DiT model for VACE, as shown in Fig. 3, aiming to support multimodal VCU inputs. Since there is an existing pipeline for text tokenization, we only consider about the tokenization of context frames and masks. After tokenized, the context tokens combined with noisy video tokens and fine-tune the DiT model. Differ from that, we also propose a Context Adapter Tuning strategy, which allows context tokens to pass Context Blocks and added back to the original DiT Blocks.

###### 3.3.1. Context Tokenization

Concept Decoupling. Two different visual concepts of natural video and control signals like depth, pose are encoded in F simultaneously. We believe that explicitly separating these data of different modalities and distributions is essential for model convergence. The concept decoupling is based on masks and yields two frame sequences identical in shape: Fc = F × M and Fk = F × (1 − M), where Fc is called reactive frames contain all the pixels to be changed, while all the pixels to be kept are stored in Fk, named inactive frames. Specifically, the reference images and the unchanged part of V2V and MV2V go to Fk, while control signals and those pixels about to change, such as gray pixels are collected to Fc.

Context Latent Encoding. A typical DiT processes noisy video latents X ∈ Rn

′×h′×w′×d, where n′, h′ and w′ are the temporal and spatial shapes of the latent space. Similar to X, Fc, Fk and M need to be encoded into a highdimensional feature space to ensure the property of significant spatiotemporal correlations. Therefore, we reorganize them together with X into a hierachical and spatiotemporal aligned visual features. Fc, Fk are processed by video VAE and mapped into the same latent space of X, maintaining their spatiotemporal consistency. To aviod any unexpected mishmash of images and videos, reference images are separately encoded by VAE encoder and concatenated back along the temporal dimension, while the correspond-

ing parts need to be removed during decoding. M is directly reshaped and interpolated. After that, Fc, Fk, and M are all mapping into latent spaces and are spatiotemporal aligned with X in the shape of n′ × h′ × w′.

Context Embedder. We extend the embedder layer by concatenating Fc, Fk and M in the channel dimension and tokenizing them into context tokens, which we refer to as the Context Embedder. The corresponding weights to tokenize Fc and Fk is directly copied from the original video embedder, and the weights to tokenize M are initialized by zeros.

- 3.3.2. Fully Fine-Tuning and Context Adapter Tuning

To achieve training with VCU as input, a simple methodology is fully fine-tuning the whole DiT model, as shown in Fig. 3a. Context tokens are added together with noisy tokens X, and all the parameters in DiT and the newly introduced Context Embedder will be updated during training. To aviod fully fine-tuning and achieve faster convergence, as well as to establish a pluggable feature with the foundation model, we also propose another methodology processing the context token in a Res-Tuning [29] manner, as shown in Fig. 3b. Particularly, we select and copy several Transformer Blocks from the original DiT, forming a distributed and cascade-type Context Blocks. The original DiT processes video tokens and text tokens, while the newly added Transformer Blocks processes context tokens and text tokens. The output of each Context Block is inserted back to the DiT blocks as an addictive signal, to assist the main branch in performing generation and editing tasks. In this manner, the parameters of DiT are frozen. Only the Context Embedder and Context Blocks are trainable.

- 4. Datasets

#### 4.1. Data Construction

To obtain an all-in-one model, the diversity and complexity of the required data construction also increase. Existing common text-to-video and image-to-video tasks only require constructing pairs of text and video. However, for the tasks in VACE, the modalities need to be further expanded to include target videos, source videos, local masks, reference and more. To efficiently and rapidly acquire data for various tasks, it is imperative to maintain video quality while also conducting instance-level analysis and understanding of the video data.

To achieve this, we first analyze the video data itself by performing shot slicing and preliminarily filtering out data based on resolution, aesthetic score, and motion amplitude. Next, we label the first frame of the video using RAM [78] and combine it with Grounding DINO [36] for detection, utilizing the localization results to perform secondary filtering on videos with target areas that are either too small or too large. Furthermore, we employ the propagate operation

of SAM2 [52] for video segmentation to obtain instancelevel information across the video. Leveraging the results of video segmentation, we filter instances in the temporal dimension by calculating the effective frame ratio based on the mask area threshold.

In the actual training process, the construction for different tasks also needs to be tailored according to the characteristics of each task: 1) For some controllable video generation tasks, we pre-extract depth [51], scribble [6], pose [5, 72], and optical flow [65] from the filtered videos. For gray and layout tasks, we create data on the fly. 2) For repainting tasks, random instances from the videos can be masked for inpainting, while the inverse of the mask enables the construction of outpainting data. Augmentation of the masks [62] allows for unconditional inpainting. 3) In the case of extension tasks, we extract key frames such as the first frame, last frame, frames from both ends, random frames, and segments from both ends to support a wider variety of extension types. 4) For reference tasks, we can extract several face or object instances from the videos and apply offline or online augmentation operations to create paired data. Notably, we randomly combine all the previously mentioned tasks for training to accommodate a broader range of model application scenarios. Additionally, for all operations involving masks, we perform arbitrary augmentation to satisfy various granular local generation requirements.

#### 4.2. VACE-Benchmark

Significant progress has been made in the field of video generation. However, a scientific and thorough evaluation of the performance of these models remains an urgent issue that needs to be addressed. VBench [27] and VBench++ [28] have established a precise evaluation framework for text-to-video and image-to-video tasks through an extensive assessment suite and dimensional design. Nevertheless, as the video generation ecosystem continues to evolve, more derivative tasks have begun to emerge, such as video reference generation and video editing, for which a comprehensive benchmark is still lacking. To address this gap, we propose VACE-Benchmark to evaluate various downstream tasks related to video in a systematic manner.

Starting from the data sources, we recognize that real videos and generated videos may exhibit different performance characteristics during evaluation. Thus, we collected a total of 240 high-quality videos categorized by their sources, encompassing various data types, including text-to-video, inpainting, outpainting, extension, grayscale, depth, scribble, pose, optical flow, layout, reference face, and reference object tasks, with an average of 20 samples for each task. The input modalities include input videos, masks, and reference, and we also provide the original

- Table 2. Quantitative evaluations on VACE-Benchmark. We compare the automated score metrics of the unified VACE based on LTX-Video and the proprietary model on the dimensions of video quality and video consistency, as well as results of human user studies.

Type Method Video Quality & Video Consistency User Study

VideoQuality

Background

Normalized

Consistency

Consistency

Consistency

Consistency

Smoothness

Temporal

Temporal

Aesthetic

Dynamic

Flickering

Following

Imaging

Average

Subject

Overall

Motion

Prompt

Average

Quality

Quality

Degree

I2V I2VGenXL [77] 55.20% 92.87% 60.00% 63.31% 97.43% 23.78% 89.58% 95.67% 71.54% 2.65 1.60 2.34 2.20 CogVideoX-I2V [73] 57.78% 94.80% 40.00% 68.23% 98.69% 24.38% 93.84% 97.84% 73.66% 3.30 2.28 3.19 2.92 LTX-Video [22] 56.12% 94.57% 35.00% 62.72% 99.27% 24.92% 92.83% 98.41% 72.89% 2.95 2.28 2.28 2.50

VACE (Ours) 57.53% 95.32% 45.00% 68.03% 99.08% 25.13% 93.61% 97.80% 74.38% 3.20 4.00 2.54 3.24 Inpaint ProPainter [82] 44.70% 95.64% 50.00% 61.57% 99.01% 18.48% 92.99% 98.47% 70.15% 2.35 4.00 2.99 3.11

VACE (Ours) 51.30% 96.30% 50.00% 60.39% 99.12% 21.12% 94.59% 98.21% 72.05% 2.40 4.00 2.60 3.00 Outpaint Follow-Your-Canvas [8] 53.30% 95.99% 5.00% 69.53% 98.08% 25.90% 95.38% 97.20% 71.54% 3.05 2.00 1.63 2.23

M3DDM [17] 53.34% 95.87% 30.00% 65.07% 99.22% 25.43% 93.65% 98.85% 73.16% 3.70 3.88 2.28 3.29 VACE (Ours) 57.04% 96.55% 30.00% 69.49% 99.20% 25.36% 94.47% 98.47% 74.25% 3.90 3.92 3.58 3.80

Depth Control-A-Video [10] 50.62% 91.71% 70.00% 67.76% 97.58% 24.48% 88.10% 96.58% 72.35% 2.70 2.28 1.54 2.17 VideoComposer [68] 50.03% 94.18% 70.00% 59.44% 96.23% 24.95% 89.79% 94.38% 70.74% 2.60 2.44 2.17 2.40 ControlVideo [79] 63.30% 95.02% 10.00% 65.13% 96.49% 24.20% 92.29% 95.42% 70.07% 2.55 2.50 1.82 2.29

###### VACE (Ours) 56.72% 96.12% 60.00% 66.41% 98.84% 25.27% 94.09% 97.27% 74.99% 3.10 3.92 2.66 3.23

Pose Text2Video-Zero [31] 57.63% 87.67% 100.00% 70.74% 79.65% 23.94% 84.82% 76.57% 59.69% 2.15 2.00 1.88 2.01 ControlVideo [79] 65.37% 94.56% 25.00% 65.28% 97.32% 25.19% 92.76% 96.82% 72.45% 2.15 1.80 2.03 1.99 Follow-Your-Pose [40] 48.79% 86.80% 100.00% 67.41% 90.12% 26.10% 80.18% 88.02% 66.43% 2.00 2.60 1.58 2.06

VACE (Ours) 60.17% 94.92% 75.00% 64.71% 98.63% 26.44% 94.82% 96.60% 76.13% 2.95 3.96 2.63 3.18 Flow FLATTEN [14] 56.23% 95.80% 70.00% 61.65% 97.86% 26.23% 93.94% 96.17% 74.42% 3.50 2.40 3.19 3.03

###### VACE (Ours) 55.76% 96.07% 75.00% 65.37% 98.98% 25.89% 94.63% 96.93% 75.90% 2.90 3.75 2.60 3.08

R2V Keling1.6 [1] 62.13% 96.04% 85.00% 69.27% 99.38% 27.82% 93.79% 97.79% 78.81% 4.22 4.10 3.80 4.04 Pika2.2 [49] 62.48% 96.79% 65.00% 69.87% 99.37% 26.02% 95.93% 98.90% 77.87% 4.00 3.85 3.87 3.91 Vidu2.0 [66] 64.30% 96.85% 35.00% 67.03% 99.66% 26.53% 96.73% 99.41% 76.47% 3.90 3.85 3.77 3.84

VACE (Ours) 63.25% 98.03% 30.00% 72.29% 99.51% 25.85% 98.54% 99.15% 76.76% 3.47 3.42 3.30 3.40

videos to enable further processing by developers based on the specific characteristics of each task. Regarding the data prompts, we supply the original captions of the videos for quantitative assessment, as well as rewritten prompts tailored to the specific tasks to evaluate the models’ creativity.

### 5. Experiments

#### 5.1. Experimental Setup

Implementation Details. VACE is trained based on Diffusion Transformers for text-to-video generation at various scales. It utilizes LTX-Video-2B [22] for faster generation, while Wan-T2V-14B [64] is used specifically for higher-quality outputs, supporting resolutions of up to 720p. The training employs a phased approach. Initially, we focus on foundational tasks such as inpainting and extension, which are considered modal complementary to the pre-trained text-to-video models. This includes the incorporation of masks and the learning of contextual generation in both spatial and temporal dimensions. Next, from a task expansion perspective, we progressively transition from single input reference frames to multiple input reference frames and from single tasks to composite tasks. Fi-

nally, we fine-tune the model’s quality using higher-quality data and longer sequences. The input for model training accommodates arbitrary resolutions, dynamic durations, and variable frame rates to support diverse input needs of users.

Baselines. Our goal is to achieve the unification of video creation and editing tasks, and currently, there is no comparable all-in-one video generation model available, which leads us to focus our evaluation on comparing our general model with proprietary task-specific models. Moreover, due to the numerous tasks involved and the lack of opensourced methods for many of them, we conduct our comparisons on models that are available either offline or online. Specifically for the tasks, we compare the following: 1) For the I2V task, we examine I2VGenXL [77], CogVideoXI2V [73], and LTX-Video-I2V [22]; 2) In the repainting task, we compare the ProPainter [82] for removal inpainting, while Follow-Your-Canvas [8] and M3DDM [17] are compared for outpainting; 3) For controllable task, we use Control-A-Video [10], VideoComposer [68], and ControlVideo [79] under depth conditions, and compare Text2Video-Zero [31], ControlVideo [79], and FollowYour-Pose [40] under pose conditions, as well as FLAT-

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

“The elegant lady carefully selects bags in the boutique, and she shows the charm of a mature woman in a black slim dress with a pearl necklace. Holding a vintage-inspired brown leather half-moon handbag, she is carefully observing its craftsmanship and texture. The interior of the store…”

Reference Anything

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

| |
|---|
| |
|[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]|
|[Figure 198]<br><br>|
|[Figure 199]|
| |
| |

[Figure 200]

[Figure 201]

“A young boy rises from his chair and walks briskly to the right side of the frame towards the edge of the sun-drenched frame, as if chasing a new adventure. His eyes were bright, and the corners of his mouth were slightly upturned, revealing curiosity and excitement about the unknown…”

Move Anything

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

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

“Anime-style, hot-blooded teenager in bright orange long-sleeved pants sportswear, standing on a surfboard, facing the golden sunshine in the rough sea. The teenager's short yellow hair is flying in the wind, his eyes are firm, and he has a confident smile on the corner of his mouth…”

Animate Anything

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

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

“The video shows an old movie-style scene in retro tones, with a little penguin and a kitten having a joyous bike race. Little penguins and kittens, both dressed in orange-and-red race suits, ride vintage multi-wheeled bikes on a nostalgic dirt road flanked by spectators…”

Swap Anything

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

“In the style of classical oil painting, the background is a river, and in the center of the picture is a mature and elegant woman, wearing a long skirt and sitting on a chair. She took the open red heart-shaped sunglasses from her arms with both hands and put them on...”

Expand Anything

- Figure 4. Visualization results of compositional tasks. VACE creatively enables reference-, move-, animate-, swap-, and expand-anything.

TEN [14] under optical flow conditions; 4) In reference generation, given the absence of open-sourced models, we compare commercial products Keling1.6 [1], Pika2.2 [49], and Vidu2.0 [66].

Evaluation. To comprehensively evaluate the performance of various tasks, we employ the VACE-Benchmark for assessment. Specifically, we divide the evaluation into automatic scoring and a user study for manual assessment. For the automatic scoring, we utilize select metrics from VBench [27] to assess video quality and video consistency, including eight indicators: aesthetic quality, background consistency, dynamic degree, imaging quality, motion smoothness, overall consistency, subject consistency, and temporal flickering. For the manual assessment, we utilize the mean opinion score (MOS) as our evaluation metric, focusing on three aspects: prompt following, temporal consistency, and video quality. In practice, we anonymize the generated data and randomly distribute it to different participants for scoring on a scale from 1 to 5.

#### 5.2. Main Results

Quantitative Evaluation. We compare VACE comprehensive model based on LTX-Video with task proprietary approaches on VACE-Benchmark. For certain tasks, we follow existing methods; for example, although we support generating based on any frame, we conduct comparisons using the first-frame reference approach from current opensource methods to ensure fairness. From Tab. 2, we can seen that for the tasks of I2V, inpainting, outpainting, depth, pose, and optical flow, our method demonstrates better performance than other open-source methods across eight indicators of video quality and video consistency, with normalized average metrics showing superior results. Some competing methods can only generate at a resolution of 256, have very short generation durations, and exhibit instability in temporal coherence, resulting in poorer performance on automatic metric calculations. For the R2V task, there is still a certain gap in metrics compared to commercial models for a small-scale model that aims for fast generation,

[Figure 244]

[Figure 245]

(a) Base structure setting. (b) Hyperparameter settings.

[Figure 246]

[Figure 247]

(c) Context adapter configurations. (d) Concept decouple setting.

- Figure 5. Ablation Studies of the VACE regarding structures, hyperparameters, and module configurations.

while being comparable to the metrics of Vidu 2.0. According to the results of human user studies, our method consistently performs better in evaluation metrics across multiple tasks, aligning well with user preferences.

Qualitative Results. In Fig. 1, we present the results of the VACE single model across various tasks. It is evident that the model achieves a high level of performance in video quality and temporal consistency. Furthermore, in composition tasks shown in Fig. 4, our model showcases impressive abilities, effectively integrating different modalities and tasks to produce results that cannot be generated by existing single or multiple models, thereby demonstrating its strong potential in the fields of video generation and editing. For example, in the “Move Anything” case, by providing a single input image and a movement trajectory, we are able to precisely move the characters in the scene with specified direction while maintaining coherence and narrative consistency.

#### 5.3. Ablation Studies

To better understand the impact of different independent modules on a unified video generation framework, we conducted a series of systematic comparative experiments based on the LTX-Video model to achieve a better model structure and configuration. To accurately assess the different experimental settings, we sample 250 data points for each task as a validation set and calculate the training loss, reflecting the model’s training progress through the mean curve changes of different tasks.

Base Structure. Text-guided image or video generation models only take noise as inference input. When extend

to our unified input paradigm, VCU, we can conduct training using fully fine-tuning or by incorporating additional parameter fine-tuning. Specifically, as shown in Fig. 5a, we compare the concatenation of different inputs along the channel dimension and modify the input dimensions of the patchify projection layer to achieve the loading and fully fine-tuning of the pre-trained model. Additionally, we introduce some extra training parameters in the form of ResTuning [29], which serialize VCU in a bypass branch and inject information into the main branch. The results indicate that both methods yielded similar effects; however, since the additional parameter fine-tuning converge faster, we base our subsequent experiments on this approach. As shown in Fig. 5b, we further conduct hyperparameter experiments based on this structure, focusing on aspects such as weighting schemes, timestamp shifting, and p-zero.

Context Adapter. Since the number of context blocks will significantly effect the model size and inference time consumption, we attempt to find an optimal number and distribution of context blocks. We begin with selecting continuous blocks at the input side and make comparisons between the first 1/4 blocks, 1/2 blocks, and all blocks. Inspired by the Res-Tuning [29] method, we also experiment with evenly distributing the injection blocks instead of selecting a continuous block series. As shown in Fig. 5c, we can see that when using the same number of blocks, the distributed arrangement of blocks outperforms the continuous arrangement in shallow blocks. Furthermore, a greater number of blocks generally yields better results, but due to the limited improvement in effectiveness and the constraints of training resources, we adopt a partially distributed arrangement of blocks.

Concept Decouple. During training, we introduce a Concept Decouple processing module to further disassemble the visual units, clarifying what content the model needs to learn to modify or retain. As shown in Fig. 5d, using this module result in a more significant reduction in loss.

### 6. Conclusion

This paper introduces VACE, an all-in-one video generation and editing framework. It unifies the diverse and complex multimodal inputs required for various video tasks, bridging the gap between specialized models for each individual task. This enables most video AI creation tasks to be completed with a single inference of a single model. While broadly covering various video tasks, VACE also supports flexible and free combinations of these tasks, greatly expanding the application scenarios of video generation models and meeting a wide range of user creative needs. The VACE framework paves the way for the development of unified visual generative models with multimodal inputs and represents a significant milestone in the field of visual generation.

Acknowledgments. We would like to express our sincere appreciation for the contributions of many colleagues for their insightful discussions, valuable suggestions, and constructive feedback, including: Yuwei Wang, Haiming Zhao, Chenwei Xie and Sheng Yao for their data contributions, and Shiwei Zhang, Tao Fang, Xiang Wang for their discussions and suggestions.

### References

- [1] KLING AI. KLING AI, https://klingai.com/,

2025. 6, 7, 13

- [2] Runway AI. Stable Diffusion v1.5 Model Card, https://huggingface.co/runwayml/stablediffusion-v1-5, 2022. 2
- [3] Runway AI. Stable Diffusion Inpainting Model Card, https://huggingface.co/runwayml/stablediffusion-inpainting, 2022. 2
- [4] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. InstructPix2Pix: Learning To Follow Image Editing Instructions. In IEEE Conf. Comput. Vis. Pattern Recog., pages 18392–18402, 2023. 2, 3
- [5] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. IEEE Trans. Pattern Anal. Mach. Intell., 43(1):172–186, 2021. 5
- [6] Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning To Generate Line Drawings That Convey Geometry and Semantics. In IEEE Conf. Comput. Vis. Pattern Recog., pages 7915–7925, 2022. 5
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. arXiv preprint arXiv:2310.00426, 2023. 2
- [8] Qihua Chen, Yue Ma, Hongfa Wang, Junkun Yuan, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen, and Wei Liu. Follow-Your-Canvas: Higher-Resolution Video Outpainting with Extensive Content Generation. In Assoc. Adv. Artif. Intell., 2025. 6, 13
- [9] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, Ting-Che Lin, Shilong Zhang, Fu Li, Chuan Li, Xing Wang, Yanghua Peng, Peize Sun, Ping Luo, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Goku: Flow Based Video Generative Foundation Models. arXiv preprint arXiv:2502.04896, 2025. 2
- [10] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. ControlA-Video: Controllable Text-to-Video Diffusion Models with Motion Prior and Reward Feedback Learning. arXiv preprint arXiv:2305.13840, 2023. 6, 13
- [11] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. AnyDoor: Zero-shot Object-level Image Customization. arXiv preprint arXiv:2307.09481,

2023. 2

- [12] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, and Hengshuang Zhao. UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics. arXiv preprint arXiv:2412.07774,

2024. 2, 3

- [13] Alibaba Cloud. Tongyi Wanxiang, https://tongyi. aliyun.com/wanxiang, 2023. 2
- [14] Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bodo Rosenhahn, Tao Xiang, and Sen He. FLATTEN: Optical FLow-guided ATTENtion for consistent text-to-video editing. In Int. Conf. Learn. Represent., 2024. 6, 7, 13
- [15] Lunhao Duan, Shanshan Zhao, Wenjun Yan, Yinglun Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Mingming Gong, and Gui-Song Xia. UNIC-Adapter: Unified Image-instruction Adapter with Multi-modal Transformer for Image Generation. arXiv preprint arXiv:2412.18928,

2024. 3

- [16] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In Int. Conf. Mach. Learn., 2024. 2
- [17] Fanda Fan, Chaoxu Guo, Litong Gong, Biao Wang, Tiezheng Ge, Yuning Jiang, Chunjie Luo, and Jianfeng Zhan. Hierarchical Masked 3D Diffusion Model for Video Outpainting. In ACM Int. Conf. Multimedia, pages 7890–7900, 2023. 6, 13
- [18] FLUX. FLUX, https://blackforestlabs.ai/,

2024. 2

- [19] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. SEED-Data-Edit Technical Report: A Hybrid Dataset for Instructional Image Editing. arXiv preprint arXiv:2405.04007,

2024. 2

- [20] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, Haibin Huang, and Chongyang Ma. I2VAdapter: A General Image-to-Video Adapter for Diffusion Models. In ACM SIGGRAPH, pages 1–12, 2024. 2
- [21] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. PuLID: Pure and Lightning ID Customization via Contrastive Alignment. In Adv. Neural Inform. Process. Syst., 2024. 2
- [22] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. LTX-Video: Realtime Video Latent Diffusion. arXiv preprint arXiv:2501.00103, 2025. 2, 6, 13
- [23] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer. In Int. Conf. Learn. Represent., 2025. 2, 3

- [24] Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. In Adv. Neural Inform. Process. Syst., 2021. 2
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In Adv. Neural Inform. Process. Syst. Curran Associates, Inc., 2020. 2
- [26] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and Controllable Image Synthesis with Composable Conditions. In Int. Conf. Mach. Learn., 2023. 2
- [27] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive Benchmark Suite for Video Generative Models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 21807– 21818, 2024. 5, 7
- [28] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, YingCong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. arXiv preprint arXiv:2411.13503, 2024. 5
- [29] Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Ao Ma, Yiliang Lv, Yujun Shen, Deli Zhao, and Jingren Zhou. Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone. In Adv. Neural Inform. Process. Syst.,

2023. 5, 8

- [30] Zeyinzi Jiang, Chaojie Mao, Yulin Pan, Zhen Han, and Jingfeng Zhang. SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8995–9004, 2024. 2
- [31] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2Video-Zero: Textto-Image Diffusion Models are Zero-Shot Video Generators. In Int. Conf. Comput. Vis., pages 15954–15964, 2023. 6, 13
- [32] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv preprint arXiv:2412.03603, 2024. 2
- [33] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. HunyuanDiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding. arXiv preprint arXiv:2405.08748, 2024. 2

- [34] Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng. MagicEdit: High-Fidelity and Temporally Coherent Video Editing. arXiv preprint arXiv:2308.14749,

2023. 3

- [35] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Qian He, and Xinglong Wu. Phantom: Subjectconsistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025. 2, 3
- [36] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499, 2023. 5
- [37] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-P2P: Video Editing with Cross-attention Control. In IEEE Conf. Comput. Vis. Pattern Recog., pages 8599– 8608, 2024. 3
- [38] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept Neurons in Diffusion Models for Customized Generation. In Int. Conf. Mach. Learn., 2023. 2
- [39] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable Image Synthesis with Multiple Subjects. In Adv. Neural Inform. Process. Syst., 2023. 2
- [40] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Ying Shan, Xiu Li, and Qifeng Chen. Follow Your Pose: Pose-Guided Text-to-Video Generation using PoseFree Videos. In Assoc. Adv. Artif. Intell., 2024. 6, 13
- [41] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. ACE++: InstructionBased Image Creation and Editing via Context-Aware Content Filling. arXiv preprint arXiv:2501.02487, 2025. 2, 3
- [42] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In Int. Conf. Learn. Represent., 2021. 2
- [43] Midjourney. Midjourney, https://www.midjourney. com, 2023. 2
- [44] MiniMax. Hailuo AI Video, https://hailuoai.com/ video, 2024.
- [45] OpenAI. DALL·E 3, https://openai.com/dall-e3, 2023. 2
- [46] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold. In ACM SIGGRAPH, 2023. 2
- [47] Yulin Pan, Chaojie Mao, Zeyinzi Jiang, Zhen Han, and Jingfeng Zhang. Locate, Assign, Refine: Taming Customized Image Inpainting with Text-Subject Guidance. arXiv preprint arXiv:2403.19534, 2024. 2
- [48] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. In Int. Conf. Comput. Vis., pages 4195– 4305, 2023. 2
- [49] PiKa. PiKa, https://pika.art/, 2025. 6, 7, 13
- [50] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming

- Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, and Ran Xu. UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild. In Adv. Neural Inform. Process. Syst., 2023. 3
- [51] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot CrossDataset Transfer. IEEE Trans. Pattern Anal. Mach. Intell., pages 1623–1637, 2022. 5
- [52] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. SAM 2: Segment Anything in Images and Videos. In Int. Conf. Learn. Represent., 2025. 5
- [53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10684–10695, 2022. 2
- [54] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional Networks for Biomedical Image Segmentation. Med. Image Comput. Computer-Assisted Interv.,

2015. 2

- [55] Runway. Gen-3, https://app.runwayml.com/ video-tools, 2025. 2
- [56] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In Int. Conf. Learn. Represent., 2021. 2
- [57] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In Int. Conf. Learn. Represent., 2021. 2
- [58] StabilityAI. Stable Diffusion v2-1 Model Card, https: / / huggingface . co / stabilityai / stable diffusion-2-1, 2022. 2
- [59] StabilityAI. Stable Diffusion XL Model Card, https: / / huggingface . co / stabilityai / stable diffusion-xl-base-1.0, 2022. 2
- [60] StabilityAI. CosXL Model Card, https : //huggingface.co/stabilityai/cosxl, 2024. 3
- [61] Ya Sheng Sun, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, and Hideki Koike. ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation. In Adv. Neural Inform. Process. Syst., 2023. 3
- [62] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-Robust Large Mask Inpainting With Fourier Convolutions. In IEEE Winter Conf. Appl. Comput. Vis., pages 2149–2159, 2022. 5
- [63] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. OminiControl: Minimal and Universal Control for Diffusion Transformer. arXiv preprint arXiv:2411.15098, 2024. 2, 3
- [64] Wan Team. Wan: Open and advanced large-scale video generative models. 2025. 2, 6, 13

- [65] Zachary Teed and Jia Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In Eur. Conf. Comput. Vis., pages 402–419, 2020. 5
- [66] Vidu. Vidu, https://www.vidu.cn/, 2025. 6, 7, 13
- [67] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. InstantID: Zero-shot Identity-Preserving Generation in Seconds. arXiv preprint arXiv:2401.07519, 2024. 2
- [68] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional Video Synthesis with Motion Controllability. In Adv. Neural Inform. Process. Syst., 2023. 2, 6, 13
- [69] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. In ACM SIGGRAPH, pages 1–11, 2024. 3
- [70] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, Yingya Zhang, and Hongming Shan. DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control. arXiv preprint arXiv:2410.13830, 2024. 2
- [71] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. OmniGen: Unified Image Generation. arXiv preprint arXiv:2409.11340, 2024. 2, 3
- [72] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective Whole-body Pose Estimation with Two-stages Distillation. In Int. Conf. Comput. Vis., pages 4210–4220, 2023. 5
- [73] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In Int. Conf. Learn. Represent.,

2025. 2, 6, 13

- [74] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. IdentityPreserving Text-to-Video Generation by Frequency Decomposition. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 2
- [75] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. MagicBrush: A Manually Annotated Dataset for InstructionGuided Image Editing. In Adv. Neural Inform. Process. Syst.,

2023. 2

- [76] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. In Int. Conf. Comput. Vis., pages 3836–3847, 2023. 2
- [77] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models. arXiv preprint arXiv:2311.04145, 2023. 2, 6, 13
- [78] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo,

- Yaqian Li, Shilong Liu, Yandong Guo, and Lei Zhang. Recognize Anything: A Strong Image Tagging Model. arXiv preprint arXiv:2306.03514, 2023. 5
- [79] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. ControlVideo: Training-free Controllable Text-to-Video Generation. In Int. Conf. Learn. Represent., 2024. 6, 13
- [80] Yuechen Zhang, Yaoyang Liu, Bin Xia, Bohao Peng, Zexin Yan, Eric Lo, and Jiaya Jia. Magic Mirror: ID-Preserved Video Generation in Video Diffusion Transformers. arXiv preprint arXiv:2411.13503, 2025. 3
- [81] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. UltraEdit: Instruction-based Fine-Grained Image Editing at Scale. arXiv preprint arXiv:2407.05282v1,

2024. 2

- [82] Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan, and Chen Change Loy. ProPainter: Improving Propagation and Transformer for Video Inpainting. In Int. Conf. Comput. Vis., pages 10477–10486, 2023. 2, 6, 13

In the supplementary material, we provide more implementation details (Appendix A) including the hyperparameters used in training and inference. Then, we showcase additional comparisons with existing methods and more qualitative results (Appendix B). Furthermore, we discuss the social impacts and limitations (Appendix C).

### A. Implementation Details

- A.1. Hyperparameters

In Tab. 3, we provide an overview of the hyperparameters settings and conduct training based on the foundational textto-video generation models of LTX-Video [22] and WanT2V [64]. The former allows for quick inference with limited resources; in an A100 single-card environment, without a dedicated acceleration strategy, it takes about 24 seconds to sample 40 steps for a video of approximately 5 seconds in duration. This meets the needs of general users for video processing. In contrast, Wan-T2V is a comprehensive performance video generation model that requires relatively more resources for training and inference, but it is capable of producing high-quality visuals and maintaining smooth temporal consistency.

- B. Additional Results

#### B.1. More Visualization

In Fig. 6 and Fig. 7, we present more qualitative results based on Wan-T2V, which include tasks such as outpainting, inpainting, extension, grayscale, depth, scribble, pose, layout, face reference, and object reference.

- B.2. Visualization Comparison

In Fig. 8, we present a visualization of the comparison of the VACE based on LTX-Video-2B [22] with others, including the extension task compared with I2VGenXL [77], CogVideoX [73], and LTX-Video-I2V [22]; the unconditional inpainting task compared with ProPainter [82]; the outpainting task with Follow-Your-Canvas [8] and M3DDM [17]; depth-controlled generation with Control-AVideo [10], VideoComposer [68], and ControlVideo [79]; pose-controlled generation with Text2Video-Zero [31], ControlVideo [79] and Follow-Your-Pose [40]; optical flow-controlled generation with FLATTEN [14]; and the reference task compared with commercially closed-source models Keling 1.6 [1], Pika 2.2 [49], and Vidu 2.0 [66].

- C. Discussion

#### C.1. Limitations

First, the quality of generated content and the overall style are often influenced by the foundation model. This paper verifies this across different model scales: smaller models

are advantageous for rapid video generation, but the quality and coherence of the videos are inevitably challenged; larger parameter models significantly improve the success rate of creative output, but the inference speed slows down and resource consumption increases. Finding a relative balance between the two is also a key focus of our future work.

Secondly, compared to the foundational models for textto-video generation, the current unified models have not been trained on large-scale data and computational power. This results in issues such as the inability to fully maintain identity during reference generation and a lack of complete control over inputs when performing compositional tasks. As discussed in the paper regarding full fine-tuning and additional parameter fine-tuning, when unified tasks begin to apply scaling laws, the results are promising.

In addition, the operational methods for the unified models, compared to image models, present certain challenges due to the inclusion of temporal information and various modalities in their inputs. This aspect creates a threshold for practical usage. Therefore, it is worth exploring how to effectively leverage the capabilities of existing language models or agent models to guide video generation and editing, thereby enhancing productivity.

#### C.2. Societal impacts

From a positive perspective, intelligent video generation and editing can provide creators with a range of innovative tools, helping them to spark new ideas and enhance the artistic and innovative quality of video content. These technologies are gradually being applied across various industries; for example, in the business sector, video generation technology is transforming marketing and advertising strategies. Companies can quickly produce high-quality promotional videos, effectively communicating brand messages and attracting consumers. This ability to increase efficiency not only saves labor costs but also enables businesses to implement more creative marketing strategies, thus enhancing their market competitiveness.

However, with the proliferation of these technologies, certain social challenges have emerged. The convenience of video generation and editing may lead to the spread of misinformation and false content, undermining the public’s trust in information. Additionally, when generating content, the technology may inadvertently reinforce existing biases and stereotypes, negatively impacting societal cultural perceptions. These issues prompt reflections on ethics and responsibility, calling for policymakers, technology developers, and various sectors of society to work together to establish appropriate regulations to ensure the healthy development of these technologies. We must also examine their potential impacts with a cautious attitude, actively exploring ways to balance innovation with social responsibility, so that they can deliver greater benefits to society.

Table 3. Hyperparameter selection for LTX-Video-based and Wan-T2V-based VACE.

#Model

Config

LTX-Video-based Wan-T2V-based

Task 12 tasks + composition task 12 tasks + composition task Batch Size / GPU 1 1/8 Accumulate Step 8 1 Optimizer AdamW AdamW Weight Decay 0.1 0.1 Learning Rate 0.0001 0.00005 Learning Rate Schedule Constant Constant Training Steps 200,000 200,000 Resolution ˜480p ˜720p Shifting Ture True Weighting Scheme uniform uniform Sequence Length 4992 75600 Num Layers 28 40 Context Adapter Res-Tuning Res-Tuning Context Layers [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26 ] [0, 5, 10, 15, 20, 25, 30, 35] Concept Decouple Ture True Pre-trained Model LTX-Video-2b-v0.9 Wan2.1-T2V-14B

Sampler Flow Euler Flow Euler Sample Steps 40 25 Guide Scale 3.0 4.0 Generation speed ˜24s ˜260s (8 gpus)

Device A100×16 A100×128 Training Strategy AMP / DDP / BFloat16 FSDP / Tensor Parallel / BFloat16

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

TASK-Outpainting: A large spaceship is flying through space, with a smaller ship visible in the background. Suddenly, the larger ship explodes in a massive fireball, sending debris flying in all directions. The explosion is intense and bright, with flames and smoke billowing out from the wreckage. The smaller ship remains unharmed and continues to fly away from the scene. …

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

TASK-inpainting: A person is painting on a canvas outdoors, using a palette with various colors of paint. The person is wearing a dark blue jacket and a matching beret, and is seated on a wooden chair. The canvas depicts a landscape with a body of water and mountains in the background. The person is carefully applying green paint to the canvas, adding details to the scene. …

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

TASK-extension: A butterfly with black and orange wings approaches a hanging, brownish seed pod on a branch. The butterfly lands on the seed pod, causing it to sway slightly, then quickly flies away. The background is a blurred green, suggesting a forest or garden setting. The lighting is bright and natural, indicating daytime. The camera remains stationary, …

|[Figure 272]|
|---|

|[Figure 273]|
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

TASK-depth: A pig dressed as a chef is standing in a kitchen, holding a frying pan with a blue flame underneath. The pig is wearing a white chef's hat and apron, and it is stirring shredded yellow food in the pan. The background shows a modern kitchen with stainless steel appliances, a sink, and various kitchen utensils and containers on the counter. The lighting is bright and natural, …

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

TASK-pose: A young woman with curly hair and wearing a white shirt is standing against a yellow background. She is smiling and looking at the camera while holding her sunglasses up to her forehead with her right hand. The woman has dark skin, and her hair is styled in loose curls that fall around her shoulders. She is wearing large, round sunglasses with a gold frame and red lenses. …

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

TASK-gray: A young girl with long, curly hair is lying on a bed of lilac flowers and sheer fabric. She is wearing a pink dress with intricate lace details. The girl reaches up to touch the flowers above her, smiling and looking content. The background is filled with more lilacs and green leaves, creating a dreamy atmosphere. The camera angle is from above, capturing the girl and the surrounding flowers in a soft …

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

TASK-scribble: A person wearing a light blue shirt is gently petting a tabby cat lying on a white table. The cat, adorned with a white garment featuring cartoon characters, appears relaxed and content as the person strokes its head and body. The background is plain and light-colored, keeping the focus on the interaction between the person and the cat. The camera remains stationary, …

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

TASK-layout: An eagle is flying over a calm blue ocean under a clear sky. The eagle, with its brown and white feathers and yellow beak, descends towards the water, its wings spread wide. As it approaches the surface, it dives into the water, creating a splash, and emerges with a fish in its talons. The eagle then takes off again, flying away from the camera with the fish clutched tightly. …

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

TASK-object: A vibrantly colored Chinese lion dance costume stands prominently against a rich red background, exuding traditional cultural significance and festivity. The costume features elaborate details, with yellow fur adorning its edges and a complex pattern of green, red, and gold accents. Adornments such as large, expressive eyes, a wide mouth with a toothy grin, …

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

TASK-face: A man is sitting at a table, playing chess. He is holding a chess piece in his right hand and appears to be contemplating his next move. The man has curly hair and a beard, and he is wearing a black sweater over a white collared shirt. The chessboard is in front of him, with several pieces still on the board. There are also a few bottles on the table. The background is plain and neutral. …

[Figure 321]

Task-Extension: A man in a black long-sleeve shirt is sitting inside a white vehicle, holding a walkie-talkie. He looks out the window with a serious expression. The camera gradually zooms in on his face, emphasizing his focused gaze …

Task-depth: A close-up shot of a white flower with yellow stamen in the center, surrounded by green leaves. The flower is in focus while the background is blurred, creating a bokeh effect. The lighting is natural and bright, …

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Source Video

###### Source Video

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

###### VACE(Ours)

VACE(Ours)

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

LTX-Video

VideoComposer

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

I2VGenXL

Control-A-Video

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

CogVideoX-I2V

ControlVideo

Task-unconditional inpainting: The video shows the streets in autumn, with the ground covered in golden and reddish brown fallen leaves dancing gently in the wind. The colorful trees around shimmer with brilliant colors …

Task-pose: A woman with long brown hair tied in a ponytail, wearing a black long-sleeve crop top and black leggings, is jogging along the edge of a calm body of water. She has white earphones in her ears and a smartwatch on her left wrist …

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Source Video

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

Source Video

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

VACE(Ours)

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

VACE(Ours)

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

ProPainter

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

ControlVideo

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Follow-YourPose

Task-Outpainting: A small monkey with a white and brown fur pattern is sitting in a lush green forest. The monkey looks around curiously, occasionally moving its head and raising its hand to its face. Its eyes are wide and expressive …

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

Source Video

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

Text2VideoZero

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

VACE(Ours)

Task-flow: A group of white geese with orange beaks and feet walk in a line across a lush green field dotted with small white flowers. The geese move steadily from left to right, maintaining their formation as they traverse the grassy terrain …

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Source Video

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

M3DDM

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

VACE(Ours)

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Follow-YourCanvas

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

FLATTEN

[Figure 426]

[Figure 427]

Task-object: A sleek and modern pair of white over-ear headphones is prominently displayed against a orange lighting background. The headphones, with their smooth curves and padded ear cups, seem to float effortlessly, emphasizing their lightweight and comfortable design. The headband arches gracefully, connecting the ear cups that showcase subtle, streamlined detailing along their outer panels.The background is a solid gray that sets a neutral backdrop …

Task-FACE: A man dressed in formal attire stood on the outdoor green grass, smiling. He is wearing a brown tweed jacket over a white dress shirt, neatly buttoned up, and a maroon tie with small white polka dots. His dark hair is neatly combed and styled. The background is a grassland in spring. The camera captures the scene from a straightforward, head-on angle.The lighting is even and bright, typical of studio lighting, which highlights the details of the man's attire …

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

VACE(Ours)

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

VACE(Ours)

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

Kling1.6

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

Kling1.6

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

Vidu2.0

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

Vidu2.0

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

Pika2.2

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

Pika2.2

Figure 8. Qualitative comparisons on various tasks based on LTX-Video-based VACE framework.

