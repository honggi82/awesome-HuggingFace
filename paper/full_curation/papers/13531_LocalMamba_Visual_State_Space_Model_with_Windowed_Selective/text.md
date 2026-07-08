# arXiv:2403.09338v1[cs.CV]14Mar2024

## LocalMamba: Visual State Space Model with Windowed Selective Scan

Tao Huang1, Xiaohuan Pei1, Shan You2, Fei Wang3, Chen Qian2, and Chang Xu1

1 School of Computer Science, Faculty of Engineering, The University of Sydney 2 SenseTime Research 3 University of Science and Technology of China

Abstract. Recent advancements in state space models, notably Mamba, have demonstrated significant progress in modeling long sequences for tasks like language understanding. Yet, their application in vision tasks has not markedly surpassed the performance of traditional Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). This paper posits that the key to enhancing Vision Mamba (ViM) lies in optimizing scan directions for sequence modeling. Traditional ViM approaches, which flatten spatial tokens, overlook the preservation of local 2D dependencies, thereby elongating the distance between adjacent tokens. We introduce a novel local scanning strategy that divides images into distinct windows, effectively capturing local dependencies while maintaining a global perspective. Additionally, acknowledging the varying preferences for scan patterns across different network layers, we propose a dynamic method to independently search for the optimal scan choices for each layer, substantially improving performance. Extensive experiments across both plain and hierarchical models underscore our approach’s superiority in effectively capturing image representations. For example, our model significantly outperforms Vim-Ti by 3.1% on ImageNet with the same 1.5G FLOPs. Code is available at: https: //github.com/hunto/LocalMamba.

Keywords: Generic vision model · Image recognition · State space model

### 1 Introduction

Structured State Space Models (SSMs) have recently gained prominence as a versatile architecture in sequence modeling, heralding a new era of balancing computational efficiency and model versatility [9,12,13,35]. These models synthesize the best attributes of Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs), drawing inspiration from the foundational principles of classical state space models [23]. Characterized by their computational efficiency, SSMs exhibit linear or near-linear scaling complexity with sequence

Correspondence to: Tao Huang <thua7590@uni.sydney.edu.au>, Shan You <youshan@sensetime.com>, Chang Xu <c.xu@sydney.edu.au>

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

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

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

| |
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

[Figure 39]

|[Figure 40]|
|---|

[Figure 41]

|[Figure 42]|
|---|

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

- (a)
- (b)
- (c)

··· ···

[Figure 47]

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

··· ···

| |
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

···

Fig. 1: Illustration of scan methods. (a) and (b): Previous methods Vim [60] and VMamba [32] traverse the entire row or column axis, resulting in significant distances for capturing dependencies between neighboring pixels within the same semantic region (e.g., the left eye in the image). (c) We introduce a novel scan method that partitions tokens into distinct windows, facilitating traversal within each window (window size is 3 × 3 here). This approach enhances the ability to capture local dependencies.

length, making them particularly suited for handling long sequences. Following the success of Mamba [9], a novel variant that incorporates selective scanning (S6), there has been a surge in applying SSMs to a wide range of vision tasks. These applications extend from developing generic foundation models [32,60] to advancing fields in image segmentation [30,34,40,54] and synthesis [14], demonstrating the model’s adaptability and potential in visual domain.

Typically these vision studies need to transform 2D images into 1D sequences for SSMbased processing, and then integrate the original SSM structure of Mamba into their foundational models for specific tasks. Nevertheless, they have only shown modest improvements over traditional CNNs [20,24,39,41,42,52] and Vision Transformers (ViTs) [3,7,33,46]. This modest advancement underscores a significant challenge: the non-causal nature of 2D spatial pattern in images is inherently at odds with the causal processing framework of SSMs. As illustrated in Figure 1, traditional methods that flatten spatial data into 1D tokens disrupt the natural local 2D dependencies, weakening the model’s ability to accurately interpret spatial relationships. Although VMamba [32] introduces a 2D scanning technique to address this by scanning images in both horizontal and vertical directions, it still struggles with maintaining the proximity of originally adjacent tokens within the scanned sequences, which is critical for effective local representation modeling.

85

+0.7

Vim

+local scan

81.0

80.3

80

+2.7

75.8

75

73.1

70

65

60

Tiny Small

ACC (%)

Fig. 2: By extending the original scan with our local scan mechanism, our method significantly improves the ImageNet accuracies of Vim [60] while keeping similar FLOPs.

In this work, we introduce a novel approach to improve local representation within Vision Mamba (ViM) by segmenting the image into multiple distinct local windows. Each window is scanned individually before conducting a traversal

across windows, ensuring that tokens within the same 2D semantic region are processed closely together. This method significantly boosts the model’s capability to capture details among local regions, with the experimental results validated in Figure 2. We design our foundational block by integrating both traditional global scanning directions and our novel local scanning technique, empowering the model to assimilate comprehensive global and nuanced local information. Furthermore, to better aggregate features from these diverse scanning processes, we propose a spatial and channel attention module, SCAttn, engineered to discern and emphasize valuable information while filtering out redundancy.

Acknowledging the distinct impact of scanning directions on feature representation (for instance, a local scan with a window size of 3 excels in capturing smaller objects or details, whereas a window size of 7 is better suited for larger objects), we introduce a direction search method for selecting optimal scanning directions. This variability is especially pronounced across different layers and network depths. Inspired by DARTS [29], we proceed from discrete selection to a continuous domain, represented by a learnable factor, to incorporate multiple scanning directions within a single network. After the training of this network, the most effective scanning directions are determined by identifying those with the highest assigned probabilities.

Our developed models, LocalVim and LocalVMamba, incorporate both plain and hierarchical structures, resulting in notable enhancements over prior methods. Key contributions of this study include:

- 1. We introduce a novel scanning methodology for SSMs that includes localized scanning within distinct windows, significantly enhancing our models’ ability to capture detailed local information in conjunction with global context.
- 2. We develop a method for searching scanning directions across different network layers, enabling us to identify and apply the most effective scanning combinations, thus improving network performance.
- 3. We present two model variants, designed with plain and hierarchical structures. Through extensive experimentation on image classification, object detection, and semantic segmentation tasks, we demonstrate that our models achieve significant improvements over previous works. For example, on semantic segmentation task, with a similar amount of parameters, our LocalVimS outperforms Vim-S by a large margin of 1.5 on mIoU (SS).

### 2 Related Work

#### 2.1 Generic Vision Backbone Design

The last decade has witnessed transformative advancements in computer vision, primarily driven by the evolution of deep neural networks and the emergence of foundational generic models. Initially, Convolutional Neural Networks (CNNs) [17,20,24,39,41,42,50,52] marked a significant milestone in visual model architecture, setting the stage for complex image recognition and analysis tasks.

Among these works, ResNet [20], with a cornerstone residual connection technique, is one of the most popular model that is widely used in broad field of vision tasks; MobileNet [21,41] series lead the design of light-weight models with the utilization of depth-wise convolutions. However, the introduction of the Vision Transformer (ViT) [7] marked a paradigm shift, challenging the supremacy of CNNs in the domain. ViTs revolutionize the approach to image processing by segmenting images into a series of sequential patches and leveraging the selfattention mechanism, a core component of Transformer architectures [48], to extract features. This novel methodology highlighted the untapped potential of Transformers in visual tasks, sparking a surge of research aimed at refining their architecture design [45] and training methodologies [18,31,46,47,53], boosting computational efficiency [3,22,33,49], and extending their application scope [8, 25, 26, 38, 44, 56, 58]. Building on the success of long-sequence modeling with Mamba [9], a variant of State Space Models (SSMs), some innovative models such as Vim [60] and VMamba [32] have been introduced in visual tasks, namely Vision Mamba. These models adapt the Mamba framework to serve as a versatile backbone for vision applications, demonstrating superior efficiency and accuracy over traditional CNNs and ViTs in high-resolution images.

#### 2.2 State Space Models

State Space Models (SSMs) [11, 13, 16, 27, 37], represent a paradigm in architecture designed for sequence-to-sequence transformation, adept at managing long dependency tokens. Despite initial challenges in training, owing to their computational and memory intensity, recent advancements [9–11, 16, 43] have significantly ameliorated these issues, positioning deep SSMs as formidable competitors against CNNs and Transformers. Particularly, S4 [11] introduced an efficient Normal Plus Low-Rank (NPLR) representation, leveraging the Woodbury identity for expedited matrix inversion, thus streamlining the convolution kernel computation. Building on this, Mamba [9] further refined SSMs by incorporating an input-specific parameterization alongside a scalable, hardware-optimized computation approach, achieving unprecedented efficiency and simplicity in processing extensive sequences across languages and genomics.

The advent of S4ND [36] marked the initial foray of SSM blocks into visual tasks, adeptly handling visual data as continuous signals across 1D, 2D, and

##### 3D domains. Subsequently, taking inspiration of the success of Mamba models, Vmamba [32] and Vim [60] expanded into generic vision tasks, addressing the directional sensitivity challenge in SSMs by proposing bi-directional scan and cross-scan mechanisms. Leveraging Mamba’s foundation in generic models, new methodologies have been developed for visual tasks, such as image segmentation [30, 34, 40, 54] and image synthetic [14], showcasing the adaptability and effectiveness of visual Mamba models in addressing complex vision challenges.

### 3 Preliminaries

#### 3.1 State Space Models

Structured State Space Models (SSMs) represent a class of sequence models within deep learning, characterized by their ability to map a one-dimensional sequence x(t) ∈ RL to y(t) ∈ RL via an intermediate latent state h(t) ∈ RN:

h′(t) = Ah(t) + Bx(t), y(t) = Ch(t),

(1)

where the system matrices A ∈ RN×N, B ∈ RN×1, and C ∈ RN×1 govern the dynamics and output mapping, respectively.

Discretization. For practical implementation, the continuous system described by Equation 1 is discretized using a zero-order hold assumption4, effectively converting continuous-time parameters (A, B) to their discrete counterparts (A, B) over a specified sampling timescale ∆ ∈ R> 0:

- A = e∆A

- B = (∆A)−1(e∆A − I) · ∆B.

(2)

This leads to a discretized model formulation as follows:

ht = Aht−1 + Bxt, yt = Cht.

(3)

For computational efficiency, the iterative process delineated in Equation 3 can be expedited through parallel computation, employing a global convolution operation:

y = x ⊛ K

with K = (CB,CAB,...,CAL−1B),

(4)

where ⊛ represents the convolution operation, and K ∈ RL serves as the kernel of the SSM. This approach leverages the convolution to synthesize the outputs across the sequence simultaneously, enhancing computational efficiency and scalability.

#### 3.2 Selective State Space Models

Traditional State Space Models (SSMs), often referred to as S4, have achieved linear time complexity. However, their ability to capture sequence context is inherently constrained by static parameterization. To address this limitation,

4 This assumption holds the value of x constant over a sample interval ∆.

[Figure 67]

| | |
|---|---|
| | |

PatchEmbed

|Scan 1| |
|---|---|
| | |

SSM(S6)Block

|Scan 2| |
|---|---|
| | |

SCAttn

LocalVim Block

+

|Scan 3| |
|---|---|
| | |

× L

|Scan 4| |
|---|---|
| | |

Classiﬁer

(a)

( H, W, C )

global

|)<br><br>)|local|
|---|---|
| | |

AvgPool

FC

( 1, 1, C

( H, W, C/r )

FC

GELU

| | |
|---|---|
| | |

( 1, 1, C/r

C

GELU

| | |
|---|---|
| | |

( H, W, 2C/r )

FC

FC

( 1, 1, C )

( H, W, 1 )

Sigmoid

Sigmoid

channel attn. spatial attn.

( H, W, C )

(b)

- Fig. 3: (a) Structure of the LocalVim model. (b) Illustration of the proposed spatial and channel attention module (SCAttn).

Selective State Space Models (termed Mamba) [9] introduce a dynamic and selective mechanism for the interactions between sequential states. Unlike conventional SSMs that utilize constant transition parameters (A,B), Mamba models employ input-dependent parameters, enabling a richer, sequence-aware parameterization. Specifically, Mamba models calculate the parameters B ∈ RB×L×N, C ∈ RB×L×N, and ∆ ∈ RB×L×D directly from the input sequence x ∈ RB×L×D.

The Mamba models, leveraging selective SSMs, not only achieve linear scalability in sequence length but also deliver competitive performance in language modeling tasks. This success has inspired subsequent applications in vision tasks, with studies proposing the integration of Mamba into foundational vision models. Vim [60], adopts a ViT-like architecture, incorporating bi-directional Mamba blocks in lieu of traditional transformer blocks. VMamba [32] introduces a novel 2D selective scanning technique to scan images in both horizontal and vertical orientations, and constructs a hierarchical model akin to the Swin Transformer [33]. Our research extends these initial explorations, focusing on optimizing the S6 adaptation for vision tasks, where we achieve improved performance outcomes.

### 4 Methodology

This section delineates core components of our LocalMamba, beginning with the local scan mechanism designed to enhance the model’s ability to dig finegrained details from images. Subsequently, we introduce the scan direction search algorithm, an innovative approach that identifies optimal scanning sequences across different layers, thereby ensuring a harmonious integration of global and local visual cues. The final part of this section illustrates the deployment of the LocalMamba framework within both simple plain architecture and complex hierarchical architecture, showcasing its versatility and effectiveness in diverse settings.

#### 4.1 Local Scan for Visual Representations

Our method employs the selective scan mechanism, S6, which has shown exceptional performance in handling 1D causal sequential data. This mechanism processes inputs causally, effectively capturing vital information within the scanned segments, akin to language modeling where understanding the dependencies between sequential words is essential. However, the inherent non-causal nature of 2D spatial data in images poses a significant challenge to this causal processing approach. Traditional strategies that flatten spatial tokens compromise the integrity of local 2D dependencies, thereby diminishing the model’s capacity to effectively discern spatial relationships. For instance, as depicted in Figure 1 (a) and (b), the flattening approach utilized in Vim [60] disrupts these local dependencies, significantly increasing the distance between vertically adjacent tokens and hampering the model’s ability to capture local nuances. While VMamba [32] attempts to address this by scanning images in both horizontal and vertical directions, it still falls short of comprehensively processing the spatial regions in a single scan.

To address this limitation, we introduce a novel approach for scanning images locally. By dividing images into multiple distinct local windows, our method ensures a closer arrangement of relevant local tokens, enhancing the capture of local dependencies. This technique is depicted in Figure 1 (c), contrasting our approach with prior methods that fail to preserve spatial coherence.

While our method excels at capturing local dependencies effectively within each region, it also acknowledges the significance of global context. To this end, we construct our foundational block by integrating the selective scan mechanism across four directions: the original (a) and (c) directions, along with their flipped counterparts facilitating scanning from tail to head (the flipped directions are adopted in both Vim and VMamba for better modeling of non-causual image tokens). This multifaceted approach ensures a comprehensive analysis within each selective scan block, striking a balance between local detail and global perspective.

As illustrated in Figure 3, our block processes each input image feature through four distinct selective scan branches. These branches independently capture relevant information, which is subsequently merged into a unified feature output. To enhance the integration of diverse features and eliminate extraneous information, we introduce a spatial and channel attention module before merging. As shown in Figure 3b, this module adaptively weights the channels and tokens within the features of each branch, comprising two key components: a channel attention branch and a spatial attention branch. The channel attention branch aggregates global representations by averaging the input features across the spatial dimension, subsequently applying a linear transformation to determine channel weights. Conversely, the spatial attention mechanism assesses token-wise significance by augmenting each token’s features with global representations, enabling a nuanced, importance-weighted feature extraction.

Remark. While some ViT variants, such as the Swin Transformer [33], propose the division of images into smaller windows, the local scan in our Local-

H Horizontal scan V Vertical scan 2 2×2 local scan 7 7×7 local scan F: ﬂip

LocalVim-T / LocalVim-S

stride=16

7F

7F

7

VF

VF

VF

7

2F

VF

VF

2

2F

7

2

VF

2F

7

2

7

2F

PatchEmbed

Classiﬁer

7

7

V

V

V

V

VF

VF

V

V

VF

VF

VF

VF

V

2

2

V

2F

2

VF

2F

HF

HF

HF

HF

V

V

HF

HF

V

V

HF

V

HF

V

VF

HF

HF

VF

H

HF

H

H

H

H

HF

HF

H

H

H

H

H

HF

H

H

V

H

H

V

LocalVMamba-T

stride=4 stride=8 stride=16 stride=32

7F

7

7

2

2F

2F

7F

VF

7F

7F

7F

2F

7F

7F

7F

PatchMerging

PatchMerging

PatchMerging

PatchEmbed

Classiﬁer

VF

V

2

VF

VF

2

7

V

7

2

2

VF

7

2F

7

V

HF

VF

V

HF

VF

2F

HF

VF

V

VF

HF

2F

VF

V

H

H

HF

HF

H

HF

H

H

H

HF

V

H

VF

HF

HF

- Fig. 4: Visualization of the searched directions of our models. The visualization of

- LocalVMamba-S is in Section A.2.

Mamba is distinct both in purpose and effect. The windowed self-attention in ViTs primarily addresses the computational efficiency of global self-attention, albeit at the expense of some global attention capabilities. Conversely, our local scan mechanism aims to rearrange token positions to enhance the modelling of local region dependencies in visual Mamba, while the global understanding capability is retained as the entire image is still aggregated and processed by SSM.

#### 4.2 Searching for Adaptive Scan

The efficacy of the Structured State Space Model (SSM) in capturing image representations varies across different scan directions. Achieving optimal performance intuitively suggests employing multiple scans across various directions, similar to our previously discussed 4-branch local selective scan block. However, this approach substantially increases computational demands. To address this, we introduce a strategy to efficiently select the most suitable scan directions for each layer, thereby optimizing performance without incurring excessive computational costs. This method involves searching for the optimal scanning configurations for each layer, ensuring a tailored and efficient representation modeling.

Search space. To tailor the scanning process for each layer, we introduce a diverse set S of 8 candidate scan directions. These include horizontal and vertical scans (both standard and flipped), alongside local scans with window sizes of 2 and 7 (also both standard and flipped). For a consistent computational budget as previous models, we select 4 out of these 8 directions for each layer. This approach results in a substantial search space of (C84)K, with K representing the total number of blocks.

Building upon the principles of DARTS [29], our method applies a differentiable search mechanism for scan directions, employing continuous relaxation to navigate the categorical choices. This approach transforms the discrete selection

- Table 1: Architecture variants. We follow the original structure designs of Vim and VMamba, where Vim uses a plain structure with a patch embedding of stride 16, while VMamba constructs a hierarchical structures with SSM stages on strides 4, 8, 16, and 32.

|Model<br><br>|#Dims|#Blocks<br><br>|Params|FLOPs|
|---|---|---|---|---|
|LocalVim-T LocalVim-S<br><br>|192 384|20 20<br><br>|8M 28M|1.5G 4.8G<br><br>|
|LocalVMamba-T LocalVMamba-S<br><br>|[96, 192, 384, 768] [96, 192, 384, 768]<br><br>|[2, 2, 9, 2] [2, 2, 27, 2]|26M 50M<br><br>|5.7G 11.4G|

process into a continuous domain, allowing for the use of softmax probabilities to represent the selection of scan directions:

y(l) =

s∈S

exp(αs(l)) s′∈S exp(αs(l′))

SSMs(x(l)), (5)

where α(l) denotes a set of learnable parameters for each layer l, reflecting the softmax probabilities over all potential scan directions.

We construct the entire search space as an over-parameterized network, allowing us to simultaneously optimize the network parameters and the architecture variables α, following standard training protocols. Upon completion of the training, we derive the optimal direction options by selecting the four directions with the highest softmax probabilities. We visualize the searched directions of our models in Figure 4. For detailed analysis of the search results, see Section

- 5.5. Scalability of direction search. Our current approach aggregates all scan

directions for selection in training, aptly serving models with a moderate range of options. For instance, a model featuring 20 blocks and 128 directions per block requires 28 GB of GPU memory, indicating scalability limits for extensive choices. To mitigate memory consumption in scenarios with a vast selection array, techniques such as single-path sampling [15,57], binary approximation [1], and partial-channel usage [55] present viable solutions. We leave the investigation of more adaptive direction strategies and advanced search techniques to future endeavors.

#### 4.3 Architecture Variants

To thoroughly assess our methodology’s effectiveness, we introduce architecture variants grounded in both plain [60] and hierarchical [32] structures, named LocalVim and LocalVMamba, respectively. The configurations of these architectures are detailed in Table 1. Specifically, in LocalVim, the standard SSM block is substituted with our LocalVim block, as depicted in Figure 3. Considering the original Vim block comprises two scanning directions (horizontal and flipped

horizontal), and our LocalVim introduces four scanning directions, thereby increasing the computational overhead. To maintain similar computation budgets, we adjust the number of Vim blocks from 24 to 20. For LocalVMamba, which inherently has four scanning directions akin to our model, we directly replace the blocks without changing the structural configurations.

Computational cost analysis. Our LocalMamba block is efficient and effective, with only a marginal increase on computation cost. The scanning mechanism, which involves merely repositioning tokens, incurs no additional computational cost in terms of FLOPs. Furthermore, the SCAttn module, designed for efficient aggregation of varied information across scans, is exceptionally streamlined. It leverages linear layers to reduce the token dimension by a factor of 1/r, thereafter generating attention weights across both spatial and channel dimensions, with r set to 8 for all models. For instance, our LocalVMamba-T model, which replaces the VMamba block with our LocalMamba block, only increases the FLOPs of VMamva-T from 5.6G to 5.7G.

### 5 Experiments

This section outlines our experimental evaluation, starting with the ImageNet classification task, followed by transferring the trained models to various downstream tasks, including object detection and semantic segmentation.

#### 5.1 ImageNet Classification

Training strategies. We train the models on ImageNet-1K dataset [6] and evaluate the performance on ImageNet-1K validation set. Following previous works [32,33,46,60], we train our models for 300 epochs with a base batch size of 1024 and an AdamW optimizer, a cosine annealing learning rate schedule is adopted with initial value 10−3 and 20-epoch warmup. For training data augmentation, we use random cropping, AutoAugment [5] with policy rand-m9-mstd0.5, and random erasing of pixels with a probability of 0.25 on each image, then a MixUp strategy with ratio 0.2 is adopted in each batch. An exponential moving average on model is adopted with decay rate 0.9999.

Scan direction search. For the supernet training, we curtail the number of epochs to 100 while maintaining the other hyper-parameters consistent with standard ImageNet training. The embedding dimension for the supernet in LocalVim variants is set to 128, with search operations conducted identically on LocalVim-T and LocalVim-S due to their uniform layer structure. For LocalVMamba variants, including LocalVMamba-T and LocalVMamba-S, the initial embedding dimension is minimized to 32 to facilitate the search process.

Results. Our results, summarized in Table 2, illustrate significant accuracy enhancements over traditional CNNs and ViT methodologies. Notably, LocalVim-T achieves a 76.2% accuracy rate with 1.5G FLOPs, surpassing the DeiT-Ti, which records a 72.2% accuracy. In hierarchical structures, LocalVMambaT’s 82.7% accuracy outperforms Swin-T by 1.4%. Moreover, compared to our

- Table 2: Comparison of different backbones on ImageNet-1K classification. ∗: Our model without scan direction search.

|Method<br><br>|Image size Params (M) FLOPs (G)|Top-1 ACC (%)|
|---|---|---|
|RegNetY-4G [39] RegNetY-8G [39] RegNetY-16G [39]|2242 21 4.0 2242 39 8.0 2242 84 16.0<br><br>|80.0 81.7 82.9|
|ViT-B/16 [7] ViT-L/16 [7]<br><br>|3842 86 55.4 3842 307 190.7<br><br>|77.9 76.5|
|DeiT-Ti [46] DeiT-S [46] DeiT-B [46]<br><br>|2242 6 1.3 2242 22 4.6 2242 86 17.5|72.2 79.8 81.8<br><br>|
|Swin-T [33] Swin-S [33] Swin-B [33]<br><br>|2242 29 4.5 2242 50 8.7 2242 88 15.4<br><br>|81.3 83.0 83.5|
|Vim-Ti [60] Vim-S [60] LocalVim-T∗ LocalVim-T LocalVim-S∗ LocalVim-S<br><br>|2242 7 1.5 2242 26 5.1<br><br>2242 8 1.5 2242 8 1.5 2242 28 4.8 2242 28 4.8<br><br><br>|73.1 80.3 75.8 76.2 81.0 81.2<br><br>|
|VMamba-T [32] VMamba-S [32] VMamba-B [32] LocalVMamba-T LocalVMamba-S<br><br>|2242 22 5.6<br><br>2242 44 11.2<br><br>2242 75 18 2242 26 5.7 2242 50 11.4<br><br>|82.2 83.5 83.7 82.7 83.7<br><br>|

seminal contributions, Vim and VMamba, our approach registers substantial gains; for instance, LocalVim-T and LocalVMamba-T exceed Vim-Ti and VMambaT by 2.7% and 0.5% in accuracy, respectively. Additionally, to validate the local scan’s effectiveness, we conducted additional experiments on models without the scan direction search, delineated in Section 4.1, marked with ∗ in the table. Incorporating merely our local scans into the original Vim framework, LocalVimT∗ surpasses Vim-Ti by 2.7%, while the complete methodology further elevates accuracy by 0.4%. These findings affirm the pivotal role of scan directions in visual SSMs, evidencing our local scan approach’s capability to enhance local dependency capture effectively.

#### 5.2 Object Detection

Training strategies. We validate our performance on object detection using MSCOCO 2017 dataset [28] and MMDetection library [2]. For LocalVMamba series, we follow previous works [32,33] to train object detection and instance segmentation tasks with Mask-RCNN detector [19]. The training strategies include 1× setting of 12 training epochs and 3× setting with 36 training epochs

Table 3: Object detection and instance segmentation results on COCO val set.

Mask R-CNN 1× schedule

|Backbone<br><br>|Params FLOPs<br><br>|APb APb50 APb75|APm APm50 APm75<br><br>|
|---|---|---|---|
|ResNet-50 Swin-T ConvNeXt-T ViT-Adapter-S VMamba-T LocalVMamba-T<br><br>|44M 260G 48M 267G 48M 262G 48M 403G 42M 286G<br><br>45M 291G<br><br><br>|38.2 58.8 41.4 42.7 65.2 46.8 44.2 66.6 48.3 44.7 65.8 48.3 46.5 68.5 50.7 46.7 68.7 50.8<br><br>|34.7 55.7 37.2<br><br>39.3 62.2 42.2<br><br>40.1 63.3 42.8<br><br><br>39.9 62.5 42.8<br><br>42.1 65.5 45.3<br><br>42.2 65.7 45.5<br><br><br>|
|ResNet-101 Swin-S ConvNeXt-S VMamba-S LocalVMamba-S<br><br>|63M 336G<br><br>69M 354G<br><br>70M 348G<br><br><br>64M 400G 69N 414G<br><br><br>|38.2 58.8 41.4<br><br>44.8 66.6 48.9<br><br>45.4 67.9 50.0 48.2 69.7 52.5 48.4 69.9 52.7<br><br><br>|34.7 55.7 37.2<br><br>40.9 63.2 44.2<br>41.8 65.2 45.1 43.0 66.6 46.4 43.2 66.7 46.5<br><br><br>|

Mask R-CNN 3× MS schedule

|Swin-T ConvNeXt-T ViT-Adapter-S VMamba-T LocalVMamba-T<br><br>|48M 267G 48M 262G 48M 403G 42M 286G 45M 291G<br><br>|46.0 68.1 50.3 46.2 67.9 50.8 48.2 69.7 52.5 48.5 69.9 52.9 48.7 70.1 53.0<br><br>|41.6 65.1 44.9<br>41.7 65.0 44.9<br><br><br>42.8 66.4 45.9<br>43.2 66.8 46.3 43.4 67.0 46.4<br><br><br>|
|---|---|---|---|
|Swin-S ConvNeXt-S VMamba-S LocalVMamba-S<br><br>|69M 354G<br><br>70M 348G<br><br><br>64M 400G 69M 414G<br><br>|48.2 69.8 52.8 47.9 70.0 52.7<br>49.7 70.4 54.2 49.9 70.5 54.4<br><br><br>|43.2 67.0 46.1<br><br>42.9 66.9 46.2<br><br>44.0 67.6 47.3 44.1 67.8 47.4<br><br><br>|

and multi-scale data augmentations. While for LocalVim, we follow Vim [60] to use Cascade Mask R-CNN with ViTDet [26] as the detector.

Results. We summarize our results on LocalVMamba in comparisons to other backbones in Table 3. We can see that, our LocalVMamba outperforms VMamba consistently on all the model variants. And compared to other architectures, CNNs and ViTs, we obtain significant superiority. For example, our

- LocalVMamba-T obtains 46.7 box AP and 42.2 mask AP, improves Swin-T by large margins of 4.0 and 2.9, respectively. For quantitative comparisons to Vim, please refer to supplementary material.

#### 5.3 Semantic Segmentation

Training strategies. Following [32, 33, 60], we train UperNet [51] with our backbones on ADE20K [59] dataset. The models are trained with a total batch size of 16 with 512 × 512 inputs, an AdamW optimizer is adopted with weight decay 0.01. We use a Poly learning rate schedule, which decays 160K iterations with an initial learning rate of 6 × 10−5. Note that Vim did not report the FLOPs and mIoU (MS) and release the code for segmentation, so we implement our LocalVim following the ViT example configuration in MMSegmentation [4].

- Table 4: Results of semantic segmentation on ADE20K using UperNet [51]. We measure the mIoU with single-scale (SS) and multi-scale (MS) testings on the val set. The FLOPs are measured with an input size of 512 × 2048. -: Vim [60] did not report the FLOPs and mIOU (MS). MLN: multi-level neck.

|Backbone<br><br>|Image size Params (M) FLOPs (G)<br><br>|mIoU (SS) mIoU (MS)|
|---|---|---|
|DeiT-Ti Vim-Ti LocalVim-T<br><br>|5122 11 5122 13 5122 36 181<br><br>|39.2 -<br>40.2 43.4 44.4<br><br><br>|
|ResNet-50 DeiT-S + MLN Swin-T Vim-S LocalVim-S VMamba-T LocalVMamba-T<br><br>|5122 67 953<br><br>5122 58 1217<br><br>5122 60 945<br><br>5122 46 5122 58 297 5122 55 964 5122 57 970<br><br>|42.1 42.8<br>43.8 45.1<br>44.4 45.8 44.9 -<br><br><br>46.4 47.5<br><br>47.3 48.3 47.9 49.1<br><br><br>|
|ResNet-101 DeiT-B + MLN Swin-S VMamba-S LocalVMamba-S<br><br>|5122 85 1030 5122 144 2007 5122 81 1039 5122 76 1081 5122 81 1095<br><br>|42.9 44.0 45.5 47.2 47.6 49.5<br><br>49.5 50.5<br><br>50.0 51.0<br><br><br>|

Results. We report the results of both LocalVim and LocalVMamba in Table

- 4. On LocalVim, we achieve significant improvements over the baseline Vim-Ti. For example, with a similar amount of parameters, our LocalVim-S outperforms Vim-S by 1.5 on mIoU (SS). While on LocalVMamba, we achieve significant improvements over the VMamba baseline; e.g., our LocalVMamba-T achieves a remarkable mIoU (MS) of 49.1, surpassing VMamba-T by 0.8. Compared to CNNs and ViTs, our improvements are more obvious. The results demonstrate the efficacy of the global representation of SSMs in dense prediction tasks.
- 5.4 Ablation Study

Effect of local scan. The impact of our local scan technique is assessed, with experiments detailed in Table 5. Substituting Vim-T’s traditional horizontal scan with our local scan yielded a 1% performance boost over the baseline. A combination of scan directions under a constrained FLOP budget in LocalVim-T∗ led to an additional 1.1% accuracy increase. These results underscore the varied impacts of scanning with different window sizes (considering the horizontal scan as a local scan with a window size of 14 × 14) on image recognition, and an amalgamation of these scans enhances performance further.

Effect of SCAttn. In Table 5, the incorporation of SCAttn into the final LocalVim block facilitated an additional improvement of 0.6%, validating the effectiveness of strategically combining various scan directions. This underscores SCAttn’s role in enhancing performance by adaptively merging scan directions.

- Table 5: Ablation study of local scan with LocalVim-T∗ (no scan direction search, 2 × 2 window size) on ImageNet.

|Model|Horizontal scan Local scan SCAttn<br><br>|ACC|
|---|---|---|
|Vim-T Vim-T w/ local scan LocalVim-T∗ w/o SCAttn LocalVim-T∗<br><br>|✓<br><br>✓ ✓ ✓<br><br>✓ ✓ ✓<br><br>|73.1 74.1 75.2 75.8<br><br>|

Effect of scan direction search. Our empirical evaluation, as depicted in Table 2, confirms the significant benefits derived from the scan direction search strategy in the final LocalVim models. These models exhibit marked improvements over versions that merely amalgamate horizontal scans, local scans with a window size of 2×2, and their mirrored counterparts. For instance, LocalVim-T exhibits a 0.4% enhancement over LocalVim-T∗. This performance gain can be attributed to the methodological selection of scan combinations at each layer, offering a diverse set of options to optimize model efficacy.

5.5 Visualization of Searched Scan Directions

Figure 4 presents visualizations of the scanned directions obtained in our models. Observations suggest that within the plain architecture of LocalVim, there is a predilection for employing local scans in both the initial and terminal segments, with intermediate layers favoring global horizontal and vertical scans. Notably, the 2 × 2 local scans tend to concentrate towards the network’s tail, whereas larger 7 × 7 scans are prominent towards the network’s inception. Conversely, the hierarchical structure of LocalVMamba exhibits a greater inclination towards local scans compared to LocalVim, with a preference for 7 × 7 scans over 2 × 2 scans.

- 6 Conclusion

In this paper, we introduce LocalMamba, an innovative approach to visual state space models that significantly enhances the capture of local dependencies within images while maintaining global contextual understanding. Our method leverages windowed selective scanning and scan direction search to significantly improve upon existing models. Extensive experiments across various datasets and tasks have demonstrated the superiority of LocalMamba over traditional CNNs and ViTs, establishing new benchmarks for image classification, object detection, and semantic segmentation. Our findings underscore the importance of scanning mechanisms in visual state space model and open new avenues for research in efficient and effective state space modeling. Future work will explore the scalability of our approach to more complex and diverse visual tasks, as well as the potential integration of more advanced scanning strategies.

### References

- 1. Cai, H., Zhu, L., Han, S.: Proxylessnas: Direct neural architecture search on target task and hardware. arXiv preprint arXiv:1812.00332 (2018) 9
- 2. Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al.: Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155 (2019) 11
- 3. Chu, X., Tian, Z., Wang, Y., Zhang, B., Ren, H., Wei, X., Xia, H., Shen, C.: Twins: Revisiting the design of spatial attention in vision transformers. Advances in Neural Information Processing Systems 34, 9355–9366 (2021) 2, 4
- 4. Contributors, M.: MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/mmsegmentation (2020) 12
- 5. Cubuk, E.D., Zoph, B., Mane, D., Vasudevan, V., Le, Q.V.: Autoaugment: Learning augmentation strategies from data. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 113–123 (2019) 10
- 6. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009) 10
- 7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021), https://openreview. net/forum?id=YicbFdNTTy 2, 4, 11
- 8. Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J., Liu, W.: You only look at one sequence: Rethinking transformer in vision through object detection. Advances in Neural Information Processing Systems 34, 26183–26197 (2021) 4
- 9. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752 (2023) 1, 2, 4, 6, 20
- 10. Gu, A., Goel, K., Gupta, A., Ré, C.: On the parameterization and initialization of diagonal state space models. Advances in Neural Information Processing Systems 35, 35971–35983 (2022) 4
- 11. Gu, A., Goel, K., Ré, C.: Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396 (2021) 4
- 12. Gu, A., Goel, K., Re, C.: Efficiently modeling long sequences with structured state spaces. In: International Conference on Learning Representations (2022), https: //openreview.net/forum?id=uYLFoz1vlAC 1
- 13. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., Ré, C.: Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems 34, 572–585 (2021) 1, 4
- 14. Guo, H., Li, J., Dai, T., Ouyang, Z., Ren, X., Xia, S.T.: Mambair: A simple baseline for image restoration with state-space model. arXiv preprint arXiv:2402.15648

(2024) 2, 4

- 15. Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., Sun, J.: Single path one-shot neural architecture search with uniform sampling. In: Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16. pp. 544–560. Springer (2020) 9
- 16. Gupta, A., Gu, A., Berant, J.: Diagonal state spaces are as effective as structured state spaces. Advances in Neural Information Processing Systems 35, 22982–22994

(2022) 4

- 17. Han, K., Wang, Y., Xu, C., Guo, J., Xu, C., Wu, E., Tian, Q.: Ghostnets on heterogeneous devices via cheap operations. International Journal of Computer Vision 130(4), 1050–1069 (2022) 3
- 18. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022) 4
- 19. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Proceedings of the IEEE international conference on computer vision. pp. 2961–2969 (2017) 11
- 20. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016) 2, 3, 4
- 21. Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017) 4
- 22. Huang, T., Huang, L., You, S., Wang, F., Qian, C., Xu, C.: Lightvit: Towards light-weight convolution-free vision transformers. arXiv preprint arXiv:2207.05557

(2022) 4

- 23. Kalman, R.E.: A new approach to linear filtering and prediction problems (1960) 1
- 24. Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems 25

(2012) 2, 3

- 25. Lee, K., Chang, H., Jiang, L., Zhang, H., Tu, Z., Liu, C.: Vitgan: Training gans with vision transformers. arXiv preprint arXiv:2107.04589 (2021) 4
- 26. Li, Y., Mao, H., Girshick, R., He, K.: Exploring plain vision transformer backbones for object detection. In: European Conference on Computer Vision. pp. 280–296. Springer (2022) 4, 12
- 27. Li, Y., Cai, T., Zhang, Y., Chen, D., Dey, D.: What makes convolutional models great on long sequence modeling? arXiv preprint arXiv:2210.09298 (2022) 4
- 28. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014) 11
- 29. Liu, H., Simonyan, K., Yang, Y.: DARTS: Differentiable architecture search. In: International Conference on Learning Representations (2019), https://openreview. net/forum?id=S1eYHoC5FX 3, 8
- 30. Liu, J., Yang, H., Zhou, H.Y., Xi, Y., Yu, L., Yu, Y., Liang, Y., Shi, G., Zhang, S., Zheng, H., et al.: Swin-umamba: Mamba-based unet with imagenet-based pretraining. arXiv preprint arXiv:2402.03302 (2024) 2, 4
- 31. Liu, J., Liu, B., Zhou, H., Li, H., Liu, Y.: Tokenmix: Rethinking image mixing for data augmentation in vision transformers. In: European Conference on Computer Vision. pp. 455–471. Springer (2022) 4
- 32. Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Liu, Y.: Vmamba: Visual state space model. arXiv preprint arXiv:2401.10166 (2024) 2, 4, 6, 7, 9, 10, 11, 12, 20
- 33. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022

(2021) 2, 4, 6, 7, 10, 11, 12

- 34. Ma, J., Li, F., Wang, B.: U-mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv preprint arXiv:2401.04722 (2024) 2, 4

- 35. Mehta, H., Gupta, A., Cutkosky, A., Neyshabur, B.: Long range language modeling via gated state spaces. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=5MkYIYCbva 1
- 36. Nguyen, E., Goel, K., Gu, A., Downs, G., Shah, P., Dao, T., Baccus, S., Ré, C.: S4nd: Modeling images and videos as multidimensional signals with state spaces. Advances in neural information processing systems 35, 2846–2861 (2022) 4
- 37. Orvieto, A., Smith, S.L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R., De, S.: Resurrecting recurrent neural networks for long sequences. arXiv preprint arXiv:2303.06349 (2023) 4
- 38. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4195–4205

(2023) 4

- 39. Radosavovic, I., Kosaraju, R.P., Girshick, R., He, K., Dollár, P.: Designing network design spaces. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10428–10436 (2020) 2, 3, 11
- 40. Ruan, J., Xiang, S.: Vm-unet: Vision mamba unet for medical image segmentation. arXiv preprint arXiv:2402.02491 (2024) 2, 4
- 41. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4510–4520 (2018) 2, 3, 4
- 42. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014) 2, 3
- 43. Smith, J.T., Warrington, A., Linderman, S.W.: Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933 (2022) 4
- 44. Strudel, R., Garcia, R., Laptev, I., Schmid, C.: Segmenter: Transformer for semantic segmentation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 7262–7272 (2021) 4
- 45. Su, X., You, S., Xie, J., Zheng, M., Wang, F., Qian, C., Zhang, C., Wang, X., Xu, C.: Vitas: Vision transformer architecture search. In: European Conference on Computer Vision. pp. 139–157. Springer Nature Switzerland Cham (2022) 4
- 46. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.: Training data-efficient image transformers & distillation through attention. In: International conference on machine learning. pp. 10347–10357. PMLR (2021) 2, 4, 10, 11
- 47. Touvron, H., Cord, M., Jégou, H.: Deit iii: Revenge of the vit. In: European Conference on Computer Vision. pp. 516–533. Springer (2022) 4
- 48. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 4
- 49. Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 568–578 (2021) 4
- 50. Wang, Y., Xu, C., Xu, C., Xu, C., Tao, D.: Learning versatile filters for efficient convolutional neural networks. Advances in Neural Information Processing Systems 31 (2018) 3
- 51. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: Proceedings of the European conference on computer vision (ECCV). pp. 418–434 (2018) 12, 13
- 52. Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K.: Aggregated residual transformations for deep neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1492–1500 (2017) 2, 3

- 53. Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., Hu, H.: Simmim: A simple framework for masked image modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9653–9663 (2022) 4
- 54. Xing, Z., Ye, T., Yang, Y., Liu, G., Zhu, L.: Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation. arXiv preprint arXiv:2401.13560

(2024) 2, 4

- 55. Xu, Y., Xie, L., Zhang, X., Chen, X., Qi, G.J., Tian, Q., Xiong, H.: Pc-darts: Partial channel connections for memory-efficient architecture search. In: International Conference on Learning Representations (2020), https://openreview.net/ forum?id=BJlS634tPr 9
- 56. Yang, S., Wang, X., Li, Y., Fang, Y., Fang, J., Liu, W., Zhao, X., Shan, Y.: Temporally efficient vision transformer for video instance segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2885–2895 (2022) 4
- 57. You, S., Huang, T., Yang, M., Wang, F., Qian, C., Zhang, C.: Greedynas: Towards fast one-shot nas with greedy supernet. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1999–2008 (2020) 9
- 58. Zhang, B., Gu, S., Zhang, B., Bao, J., Chen, D., Wen, F., Wang, Y., Guo, B.: Styleswin: Transformer-based gan for high-resolution image generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11304–11314 (2022) 4
- 59. Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision 127(3), 302–321 (2019) 12
- 60. Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., Wang, X.: Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417 (2024) 2, 4, 6, 7, 9, 10, 11, 12, 13, 19

### A Appendix

#### A.1 Comparison to Vim on Object Detection

Different from VMamba that uses the Mask R-CNN framework for benchmark, Vim utilizes the neck architecture in ViTDet and train Cascade Mask R-CNN as the detector. We also align with the settings in Vim for fair comparison and evaluation of our LocalVim model.

We summarize the results in Table 6. Our LocalVim-T performs on pair with the Vim-Ti, while has significant superiorities in APb50 and mask AP. For example, our LocalVim-T improves Vim-Ti on APm and APm50 by 0.7 and 2.1, respectively.

Table 6: Object detection and instance segmentation results on COCO val set. Vim [60] did not report the parameters and FLOPs of the models.

|Backbone<br><br>|Params FLOPs<br><br>|APb APb50 APb75|APbs APbm APbl|
|---|---|---|---|
|DeiT-Ti Vim-Ti LocalVim-T<br><br>|- -<br>- -<br><br><br>31M 403G<br><br>|44.4 63.0 47.8<br><br>45.7 63.9 49.6 45.3 66.2 49.1<br><br><br>|26.1 47.4 61.8 26.1 49.0 63.2 26.0 49.5 61.7<br><br>|
|Backbone|Params FLOPs<br><br>|APm APm50 APm75|APms APmm APml<br><br>|
|DeiT-Ti Vim-Ti LocalVim-T<br><br>|- -<br><br>- -<br><br><br>31M 403G<br><br>|38.1 59.9 40.5<br>39.2 60.9 41.7 39.9 63.0 42.5<br><br><br>|18.1 40.5 58.4<br><br>18.2 41.8 60.2 17.7 43.0 60.5<br><br><br>|

H Horizontal scan V Vertical scan 2 2×2 local scan 7 7×7 local scan F: ﬂip

###### LocalVMamba-S

stride=4 stride=8 stride=16

7F

7

7

7

7

7F

7

7

7

7F

VF

2

7

2

7F

VF

7F

PatchMerging

PatchMerging

PatchEmbed

VF

V

VF

2F

2F

2

VF

VF

VF

VF

V

VF

2F

V

VF

V

VF

V

HF

V

V

V

VF

V

V

V

V

HF

V

VF

HF

V

HF

V

H

H

HF

HF

HF

HF

HF

H

H

H

H

H

V

H

HF

H

H

stride=32

7F

7

7

7F

7F

7

2

2F

7

7F

2F

7

7F

7

7F

7F

PatchMerging

Classiﬁer

VF

VF

2F

VF

7

VF

VF

VF

V

7

2

2

VF

2F

7

7

V

HF

VF

V

VF

V

V

V

HF

HF

VF

VF

V

V

VF

V

H

H

HF

HF

V

H

HF

H

H

H

H

H

H

HF

HF

H

###### Fig. 5: Visualization of the searched directions of LocalVMamba-S.

#### A.2 Visualization of Searched Directions on LocalVMamba-S

We visualize the searched directions of LocalVMamba-S in Figure 5. In this model, with 27 layers in stage 3, more 7 × 7 local scans are preferred compared to LocalVMamba-T.

#### A.3 Discussions

Potential negative impact. Investigating the effects of the proposed model requires large consumptions on computation resources, which can potentially raise the environmental concerns.

Limitations. The visual state space models with linear-time complexity to the sequence length, show significant improvements especially on large-resolution downstream tasks compared to previous CNNs and ViTs architectures. Nonetheless, the computational framework of SSMs is inherently more intricate than that of convolution and self-attention mechanisms, complicating the efficient execution of parallel computations. Current deep learning frameworks also exhibit limited capability in accelerating SSM computations as efficiently as they do for more established architectures. On a positive note, ongoing efforts in projects such as VMamba5 [32] are aimed at enhancing the computational efficiency of selective SSM operations. These initiatives have already realized notable advancements in speed, as evidenced by the improvements over the original implementations documented in Mamba [9].

5 Project page: https://github.com/MzeroMiko/VMamba.

