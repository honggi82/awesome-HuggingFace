[Figure 1]

## INF-LLaVA: Dual-perspective Perception for High-Resolution Multimodal Large Language Model

Yiwei Ma Zhibin Wang Xiaoshuai Sun Weihuang Lin Qiang Zhou Jiayi Ji Rongrong Ji yiweima@stu.xmu.edu.cn xssun@xmu.edu.cn

# arXiv:2407.16198v1[cs.CV]23Jul2024

### Abstract

With advancements in data availability and computing resources, Multimodal Large Language Models (MLLMs) have showcased capabilities across various fields. However, the quadratic complexity of the vision encoder in MLLMs constrains the resolution of input images. Most current approaches mitigate this issue by cropping highresolution images into smaller sub-images, which are then processed independently by the vision encoder. Despite capturing sufficient local details, these sub-images lack global context and fail to interact with one another. To address this limitation, we propose a novel MLLM, INFLLaVA, designed for effective high-resolution image perception. INF-LLaVA incorporates two innovative components. First, we introduce a Dual-perspective Cropping Module (DCM), which ensures that each sub-image contains continuous details from a local perspective and comprehensive information from a global perspective. Second, we introduce Dual-perspective Enhancement Module (DEM) to enable the mutual enhancement of global and local features, allowing INF-LLaVA to effectively process high-resolution images by simultaneously capturing detailed local information and comprehensive global context. Extensive ablation studies validate the effectiveness of these components, and experiments on a diverse set of benchmarks demonstrate that INF-LLaVA outperforms existing MLLMs. Code and pretrained model are available at https://github.com/WeihuangLin/ INF-LLaVA.

### 1. Introduction

The field of multimodal large language models (MLLMs) [20, 54, 85] has achieved substantial breakthroughs, driven by monumental advancements in computer vision [25, 30, 31] and natural language processing [1,90,112]. These MLLMs have demonstrated exceptional efficacy across an array of complex tasks, including image captioning [19,65], visual question answering [4,95], and visual dialogue [21, 67]. This substantial progress not

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

ViT Encoder

LLM

[Figure 8]

[Figure 9]

HR Image Sub-images

- (a) Cropping-based high-resolition MLLM
- (b) Dual-Encoder high-resolition MLLM
- (c) Dual-Perspective high-resolition MLLM

[Figure 10]

[Figure 11]

[Figure 12]

ViT Encoder

[Figure 13]

LR Image

Feature LLM Fusion

[Figure 14]

[Figure 15]

Conv Encoder

HR Image

Local Perspective

[Figure 16]

[Figure 17]

DualPerspecti ve

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

ViT Enco der

[Figure 22]

LLM

Global Perspective

Enhance ment

[Figure 23]

[Figure 24]

[Figure 25]

HR Image

Figure 1: Comparison between existing high-resolution MLLMs and INF-LLaVA. LR and HR abbreviate lowresolution and high-resolution, respectively. Zoom in for optimal viewing.

only highlights the transformative potential of MLLMs but also significantly extends the boundaries of understanding, reasoning, and interactive capabilities integral to the development of general artificial intelligence (AGI).

Extensive research has highlighted the critical importance of high-resolution imagery in computer vision, particularly for tasks requiring precise image perception, such as object detection [56, 106] and segmentation [50, 110]. Similarly, enhancing resolution can significantly improve the visual acuity of Multimodal Large Language Models (MLLMs). High-resolution input images inherently provide enriched detail and intricate object relationships, which are essential for mitigating hallucination issues [7,37,103] and enhancing fine-grained perception tasks [33,34]. However, large language models (LLMs) necessitate careful control over the number of image tokens produced by the

1

image encoder, as these tokens significantly affect inference speed and computational cost. Additionally, because the visual encoder in MLLMs is typically a Vision Transformer (ViT) [25], which has a computational complexity that scales quadratically with image resolution, it is crucial to limit the resolution of images fed into the ViT. As discussed, high-resolution image perception poses significant challenges for MLLMs. Thus, achieving a balance between leveraging the advantages of high-resolution inputs and managing the practical limitations of computational resources is essential for the successful deployment of MLLMs.

To address the challenge of efficient high-resolution image perception in MLLMs, existing methods are categorized into two main approaches: cropping-based methods and dual-encoder methods, as depicted in Fig. 1. Given that the ViT encoder [25] is pretrained on low-resolution images and considering its quadratic complexity relationship with image resolution, cropping-based methods [24, 49, 51, 96] partition a high-resolution image into several sub-images. These sub-images are independently processed by the ViT encoder to extract their visual features, as illustrated in Fig. 1(a). However, this independent cropping and encoding approach fails to adequately model the interrelationships between the sub-images. Research [10,62,98,111] underscores that understanding object relationships is essential for comprehensive image interpretation. Recognizing a linear relationship between the complexity of convolutional neural networks (CNNs) [48, 57] and image resolution, some researchers [47,64] have proposed a dual-encoder approach. This method leverages a pretrained ConvNeXt [57] encoder to supplement the ViT encoder for high-resolution image perception, illustrated in Fig. 1(b). However, the dual-encoder method requires additional pretrained convolutional neural networks, necessitating extensive computational resources and large-scale datasets [79,80], often demanding thousands of GPU hours.

In this paper, we introduce INF-LLaVA, a highly effective and efficient framework designed to enhance input image resolution within multimodal large language models (MLLMs), as illustrated in Fig. 1(c). The framework incorporates two innovative designs that significantly improve image resolution handling. Firstly, we propose Dualperspective Cropping Module (DCM), a sophisticated cropping strategy that partitions high-resolution images into sub-images from both local and global perspectives. From the local perspective, sub-images maintain continuous, detailed information, capturing essential details from various regions of the original image. From the global perspective, sub-images aggregate global information, albeit with less detail, as each patch in these sub-images is cropped from the original image following a specific stride pattern. This method approach ensures that DCM surpasses previ-

ous cropping methods by preserving the integrity of both global and local information at the cropping stage. Secondly, we introduce Dual-perspective Enhancement Module (DEM) to facilitate interaction between local and global features. While a straightforward approach would involve cross-attention between these features, the quadratic increase in token number due to high-resolution images often results in out-of-memory issues. To address this, our module applies a more resource-efficient strategy: it concatenates global-perspective sub-image features back into the original image’s shape based on 2D priors. These concatenated global features are then re-cropped into multiple sub-images from a local perspective. Each newly generated sub-image is matched with its corresponding local perspective sub-image, and cross-attention is performed to enrich the global features with enhanced local details. Additionally, symmetric operations are applied to local-perspective sub-images to bolster global information. Building upon DCM and DEM, we propose a new high-resolution MLLM, namely INF-LLaVA. Experimental results overwhelmingly demonstrate that these innovative designs not only enhance the handling of high-resolution images within MLLMs but also significantly optimize computational efficiency, establishing INF-LLaVA as a compelling solution to advance the field.

In summary, our contributions are three-fold:

- • We propose a novel Dual-perspective Cropping Module (DCM), which integrates both global and local perspectives when cropping high-resolution images into sub-images. This enhances the model’s ability to capture detailed and contextual information.
- • We introduce Dual-perspective Enhancement Module (DEM), an effective and efficient module for fusing dual-perspective features, resulting in dual-enhanced features that significantly improve performance.
- • Based on these two novel modules, we develop INFLLaVA, a powerful MLLM that outperforms existing models on multiple benchmarks, demonstrating the effectiveness of our approach.

### 2. Related Work

#### 2.1. Large Language Models (LLMs)

In the early stages of natural language processing (NLP) advancements, models like GPT-2 [77] and BERT [22], pretrained on web-scale text datasets, showcased exceptional representational capabilities. These models achieved monumental success and marked a significant breakthrough in the field of NLP. Building on the effectiveness of the pretraining paradigm, researchers have further enhanced large language models (LLMs) by increasing the amount of pretraining data and scaling up model parameters. Represen-

tative works in this domain include GPT-3 [9], PaLM [18], and OPT [108], which have each set new benchmarks for performance and capability. Recent efforts have pivoted towards improving LLM responses to be more aligned with human preferences by incorporating human instructions and feedback. Notable examples include InstructGPT [75], ChatGPT [73], and GPT-4 [1], which demonstrate strong perceptual and reasoning abilities in human conversations. These models have advanced the state of conversational AI, making interactions more intuitive and human-like. Additionally, the open-source LLaMA series [71, 90, 91] represent a significant contribution to the field. To further enhance the human interaction capabilities of LLaMA, researchers have developed Alpaca [87], Vicuna [17], and MPT [88], which fine-tune the LLaMA model using additional high-quality instruction data. Recognizing the importance of aligning models with human intentions and preferences, some researchers [1, 3, 91] have incorporated Reinforcement Learning from Human Feedback (RLHF) [78,81] into the training process. This approach ensures that models not only respond accurately but also in ways that are aligned with human values and requirements, thereby significantly enhancing the user experience and reliability of AI systems.

#### 2.2.MultimodalLargeLanguageModels(MLLMs)

Multimodal Large Language Models (MLLMs) are designed to extend the capabilities of traditional large language models (LLMs) by incorporating both textual and visual understanding, thereby enhancing their ability to interpret visual information and provide contextually rich responses. MLLMs [60, 113, 115] generally comprise three core components: a vision encoder, a connector, and an LLM. The vision encoder acts as the ”eyes” of the model, enabling it to perceive and analyze visual content. This encoder can utilize various structures, such as Vision Transformer (ViT) [25] or ConvNeXt [57], and can be pretrained using different methodologies, including self-supervised learning [11,74] or supervised learning [76]. Most MLLMs employ CLIP-ViT, which is pre-trained on extensive imagetext pairs, as the vision encoder to extract visual features effectively. The connector in MLLMs is responsible for transforming these visual features into the textual domain, facilitating seamless integration with the LLM. There are three prevalent types of projectors: 1) Crossattention-based methods: Models like Flamingo [2] and CogVLM [93] utilize cross-attention mechanisms to interweave visual and textual tokens within the LLM, effectively merging the two modalities. 2) Query-based methods: Approaches such as Blip-2 [45], Instruct-Blip [20], and Qwen-VL [6] employ learnable queries to extract visual features using transformer-like architectures. These queries are then concatenated with text tokens, and the combined tokens are fed into the LLM. 3) Projection-based methods:

Techniques like LLaVA [52, 54], Mini-GPT4 [115], and DeepSeek-VL [60] leverage a linear layer or a multi-layer perceptron (MLP) to project visual tokens into the textual domain directly, subsequently feeding the mixed tokens into the LLM. The LLM, serving as the ”brain” of the MLLM, interprets and processes the combined text and image information, delivering coherent and contextually appropriate responses. The range of LLMs available for integration is extensive, including models like LLaMA [71,90,91], Qwen [6], DeepSeek [8], and Yi [101]. Through the synergistic combination of these sophisticated components, MLLMs significantly enhance the capabilities of traditional LLMs, enabling them to seamlessly integrate and process multiple modalities.

#### 2.3. High-resolution MLLMs

High-resolution images offer significant advantages for Multimodal Large Language Models (MLLMs) by enabling the capture of detailed object information and complex relationships between objects in images. However, directly inputting high-resolution images into the vision encoder results in prohibitive computational expenses, primarily due to the quadratic complexity associated with the Transformer architecture [92] and the substantial increase in the number of visual tokens. To mitigate this issue, existing highresolution MLLMs can be categorized into two primary types: Cropping-based methods and Dual-Encoder methods, as illustrated in Fig. 1(a) and Fig. 1(b). Croppingbased methods [24,51,53,99] partition an image into multiple non-overlapping patches and feed each patch into the vision encoder separately, thereby obtaining visual features for local regions. To ensure that each patch maintains an aspect ratio close to 1:1, LLaVA-UHD [96] introduces various patching strategies during the crop operation. Furthermore, to individually model each patch’s information, Monkey [49] employs LoRA [35] to fine-tune the vision encoder for each specific patch. Despite their benefits, croppingbased methods can disrupt the global coherence of image information by segmenting a complete image into isolated sub-images. As a result, some researchers have proposed dual-encoder methods to maintain the integrity of global information. Dual-encoder methods leverage an auxiliary visual encoder to enhance high-resolution image understanding without significantly increasing the number of visual tokens. For instance, Vary [94] and Deepseek-VL [60] utilize the Segment Anything Model (SAM) [40] within a highresolution vision encoder to better capture high-resolution information. Meanwhile, MiniGemini [47] and LLaVAHR [64] employ ConvNeXt [57], pretrained on the massive LAION2B dataset [80], to augment the visual features extracted by the Vision Transformer (ViT). However, dualencoder methods necessitate an additional pretrained vision encoder to process high-resolution images. Both SAM, pre-

trained on the SA-1B dataset, and ConvNeXt, pretrained on the LAION-2B dataset, require extensive computational resources, amounting to tens of thousands of GPU hours, which can be cost-prohibitive. In this paper, we introduce INF-LLaVA, a novel framework that addresses these challenges by integrating an innovative Dual-perspective Cropping Module and a new Dual-perspective Enhancement Module to enhance the preservation of high-resolution global information. Our approach ensures not only the efficiency of computational resources but also the comprehensive capture of both local and global image details, thereby advancing the capabilities of high-resolution MLLMs.

### 3. Preliminary

A multimodal large language model (MLLM) is an advanced AI system designed to handle and integrate both visual and textual data effectively. It typically consists of three primary components: an image encoder FI(·), a connector FC(·), and a well-pretrained large language model (LLM) FL(·). The image encoder processes the input image I ∈ RH×W×3, where H and W represent the height and width of the image, respectively, and the three denotes the RGB color channels. This encoder extracts highdimensional visual features from the image. The connector maps these visual features into a format that the LLM can interpret, effectively serving as a bridge between the visual and textual domains. The LLM, pretrained on a vast amount of textual data, processes the integrated visual and textual data to generate coherent and contextually relevant responses.

The input to the MLLM typically includes an image I and a corresponding instruction text Tins ∈ RL, where L is the number of tokens in the instruction. Initially, the image I is processed by the image encoder FI(I) to extract visual features. Concurrently, the instruction Tins is tokenized using the tokenizer FT of the LLM to convert the text into a series of tokens. The extracted visual features are then flattened and projected into visual tokens. The connector FC converts these visual tokens into a format compatible with the LLM. Subsequently, the visual tokens and the textual tokens are concatenated along the spatial dimension and fed into the LLM FL.

The LLM decodes the combined visual and textual tokens to generate a response token-by-token. Mathematically, this decoding process can be formulated as:

p(Rt |I,Tins,R0:t−1) = FL Rt | FC FI(I) ,FT Tins ,FT R0:t−1 .

(1)

Here, p(Rt | I,Tins,R0:t−1) represents the probability distribution of the predicted token Rt at time t, given the image, instruction text, and the previously generated to-

kens. FT(R0:t−1) denotes the tokenized form of the response generated up to token t − 1, and Rt is the t-th token of the generated response.

### 4. Methods

In this section, we commence by providing a comprehensive overview of the proposed INF-LLaVA framework in Sec. 4.1, highlighting its innovative structure and capabilities. Next, we delve into the specifics of the two critical components: Dual-perspective Cropping Module and Dual-perspective Enhancement Module, thoroughly examining their intricate functionalities in Sec. 4.2 and Sec. 4.3, respectively.

#### 4.1. Overview

Fig. 2 depicts the comprehensive pipeline of the proposed INF-LLaVA framework. Due to the pretrained ViT-CLIP encoder’s limitation in processing highresolution images—given its quadratic complexity characteristics—directly feeding high-resolution images into it is computationally prohibitive. To address this challenge, we propose Dual-perspective Cropping Module (DCM) FDCM(·), which partitions high-resolution images into several sub-images from both global and local perspectives, using the resolution defined by the pretrained vision encoder. Mathematically, this operation can be described as:

[I1loc,I2loc,··· ,INloc;I1glo,I2glo,··· ,INglo] = FDCM(I),

(2)

where [I1loc,I2loc,··· ,INloc] and [I1glo,I2glo,··· ,INglo] represent the sub-images from local and global perspectives, re-

spectively. Here, N denotes the number of sub-images from each perspective, and I ∈ RW

h×Hh×3 is the high-resolution input image. Next, each local and global sub-image is separately fed into the pretrained vision encoder:

Floci = FI(Iiloc), (3) Fgloi = FI(Iiglo), (4)

l×hl×d and Fgloi ∈ Rw

where Floci ∈ Rw

l×hl×d are the visual features of the i-th local and global sub-images, respectively. Here, wl × hl denotes the number of visual tokens per sub-image, and d is the channel dimension of the visual features. These local and global sub-image features are then recombined using 2D positional prior information to form high-resolution image features:

Floc = Floc(Floc1 ,Floc2 ,··· ,FlocN ), (5) Fglo = Fglo(Fglo1 ,Fglo2 ,··· ,FgloN ), (6)

where Floc(·) and Fglo(·) are the recombination functions based on local and global positional information. The resulting features Floc ∈ Rw

h×hh×d and Fglo ∈ Rw

h×hh×d

[Figure 26]

[Figure 27]

Local-Perspective Feature

Instruction: How many chairs in the image?

Local Perspective

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

DEM & Pooling

[Figure 39]

[Figure 40]

CLIP-ViT Encoder

LLM

DCM

Global Perspective

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Response: High-resolution Image Eight.

Global-Perspective Feature

(ퟔ   × ퟔ  )

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

###### Local-perspective Cropping Global-perspective Cropping

[Figure 62]

- Figure 2: Overview of the proposed INF-LLaVA framework. To address the limitations of processing high-resolution images directly with the pretrained CLIP-ViT encoder, Dual-perspective Cropping Module (DCM) segments the high-resolution image into sub-images from both local and global perspectives. Each sub-image is then individually passed through the CLIP-ViT encoder to extract distinct visual features. These features are subsequently recombined based 2D positional priors, resulting in a comprehensive set of high-resolution local and global features. Dual-perspective Enhancement Module (DEM) is introduced to facilitate effective interaction between the local and global features. Next, an average pooling layer is applied to reduce the number of visual tokens, enhancing computational efficiency and speeding up both training and inference processes. Finally, the refined visual tokens are concatenated with textual tokens of the instruction and fed into the LLM, which generates responses sequentially, token by token.

represent the high-resolution visual features from local and global perspectives, respectively, where wh×hh is the number of visual tokens in the recombined high-resolution images. To facilitate efficient interaction between the local and global features, the proposed Dual-perspective Enhancement Module (DEM) FDEM(·) is employed. This module ensures a robust exchange of information between local and global features, resulting in dual-enhanced features:

generate the response:

R = FL(FC(Fdual),FT(Tins)), (8)

where R ∈ RL

res represents the response generated by the LLM. Here, Lres is the length of the generated response in tokens.

Fdual = Fpool(FDEM(Floc,Fglo)), (7)

where Fpool(·) is the average pooling function used to reduce the number of visual tokens, thereby accelerating training and inference speeds while minimizing computational overhead. The resulting Fdual ∈ Rw

l×hl×d represents the visual features enhanced from dual perspectives.1 Finally, the connector FC(·) projects the dual-enhanced visual features to obtain visual tokens that align with the textual features. The instruction Tins is tokenized by the tokenizer FT(·), converting it into a sequence of tokens. These visual tokens and textual tokens are then concatenated along the spatial dimension and fed into the pretrained LLM to

1Wh × Hh and Wl × Hl denote the resolutions of the high-resolution and low-resolution images, respectively. Furthermore, wh × hh and wl ×hl represent the resolutions of the high-resolution and low-resolution visual features, respectively.

#### 4.2. Dual-perspective Cropping Module

Dual-perspective Cropping Module (DCM) is designed to effectively partition the high-resolution image I ∈ RW

l×Hl×3, where Wl ×Hl corresponds to the resolution utilized by the vision encoder during pretraining. For instance, in the case of the CLIP-ViT-large-patch14-336 encoder, Wl = Hl = 336. The primary objective of DCM is to perform cropping from both local and global perspectives to capture finegrained details and broader contextual information, respectively.

h×Hh×3 into multiple sub-images Ii ∈ RW

In the sections that follow, we will elaborate on the methodologies employed for cropping high-resolution images from both perspectives, ensuring that the integrity and essential characteristics of the original image are preserved.

##### 4.2.1 Local-perspective Cropping

The local-perspective cropping technique is designed to systematically extract smaller regions from the highresolution image while preserving detailed and continuous visual information. This approach ensures that each subimage retains the high-resolution details necessary for accurate analysis and representation. By strategically segmenting the high-resolution image into localized sub-images, DCM effectively maintains the intricate details of the original image.

h×Hh×3, the first step is to determine the relationship between the dimensions of the high-resolution image (Wh,Hh) and the dimensions expected by the pretrained vision encoder (Wl,Hl). This relationship is quantified as follows:

Given a high-resolution image I ∈ RW

nW =

Wh Wl

, (9)

nH =

Hh Hl

, (10)

where ⌊·⌋ represents the floor function, which rounds down to the nearest integer. nW and nH correspond to the number of local sub-images along the width and height of the highresolution image, respectively.

Thus, the high-resolution image I is divided into nW × nH local sub-images Iiloc, each sized Wl × Hl. Specifically, for each sub-image Iiloc, where i ∈ [0,nH ×nW −1], the following formulas determine the row and column positions:

i nW

row =

, (11)

###### col = i mod nW, (12)

where mod denotes the modulo operation. The bounding box for each local sub-image Iiloc can be described by:

Iiloc = I row · Hl : (row + 1) · Hl, col · Wl : (col + 1) · Wl, : .

(13)

This process ensures that each local sub-image Iiloc contains continuous and detailed visual information from the high-resolution image, thereby preserving the integrity and quality of the image for localized analysis.

By using the local-perspective cropping technique, DCM maintains high-fidelity representation in each sub-image, enabling robust feature extraction and analysis. This approach ensures that finer details are not lost, providing a comprehensive understanding of localized regions without compromising on resolution or detail.

##### 4.2.2 Global-perspective Cropping

Conversely, global-perspective cropping aims to capture broader contextual information from high-resolution images to preserve the spatial relationships between objects. This approach ensures that our model retains an understanding of both micro and macro-level details within the extracted sub-images, facilitating the integration of comprehensive global contextual information.

Given the high resolution Wh × Hh of the input image and the low resolution Wl × Hl used by the pretrained vision encoder, the number of global sub-images nW × nH is computed similarly to the local perspective, as shown in Equ. (14) and (15):

nW =

Wh Wl

, (14)

nH =

Hh Hl

. (15)

For the i-th row and j-th column of sub-images, the pixel indices set Iij is defined as follows:

 

 

- x = j + m · nW,
- y = i + n · nH,

Iijglo =

, (16)

I(x,y)

- 0 ≤ m < W

h

nW ,m ∈ N

- 0 ≤ n < H





nH ,n ∈ N

h

where (x,y) corresponds to the pixel indices within the high-resolution image. N represents the set of natural numbers. Thus, each pixel (u,v) in the Iijglo sub-image can be mapped back to the high-resolution image I as:

Iijglo(u,v) = I (j + u · nW, i + v · nH). (17)

This module effectively partitions the high-resolution image into sub-images that encapsulate the global perspective by interleaving pixels from different regions. Consequently, it enables the model to maintain a coherent global context alongside the detailed local information captured by global-perspective cropping.

#### 4.3. Dual-perspective Enhancement Module

In Sec. 4.2, we meticulously detailed the process of segmenting a high-resolution image into multiple subimages from both local and global perspectives. Building on this foundational step, we now delve into the methods for extracting sub-features from these sub-images and their subsequent integration into high-resolution localperspective and global-perspective features, as discussed in Sec. 4.3.1. Next, we elucidate the sophisticated techniques employed for enhancing these features—specifically, the

[Figure 63]

Global-Perspective Enhancement

Dual-Perspective Fusion

Local Perspective Sub-Images

Local Perspective Sub-Features

Global-Enhanced Feature

Local Perspective Feature

[Figure 64]

[Figure 65]

LocalPerspective Feature

###### Cross Attention

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Linear

CLIP-ViT Encoder

GlobalPerspective Feature

Global Perspective Sub-Images

Global Perspective Sub-Features

[Figure 71]

Concat

Global Perspective Feature

DualEnhanced Feature

[Figure 72]

[Figure 73]

[Figure 74]

Linear

LocalPerspective Feature

Cross Attention

[Figure 75]

[Figure 76]

Local-Enhanced Feature

GlobalPerspective Feature

- Figure 3: Illustration of the integration of local and global perspective sub-features using two-dimensional positional priors. This approach ensures a seamless combination of detailed local information with broader contextual insights, maintaining spatial coherence and enhancing the overall representation of the high-resolution image.

Local-Perspective Enhancement

Figure 4: Illustration of the proposed Dual-perspective Enhancement Module (DEM), highlighting its innovative approach to efficiently integrating and enhancing local and global sub-features for superior image understanding.

global-perspective enhancement in Sec. 4.3.2 and the localperspective enhancement in Sec. 4.3.3. These enhancement techniques are designed to amplify the specific details and contextual richness of their respective features, ensuring a thorough and nuanced capture of both fine-grained and broad-spectrum information. Finally, in Sec. 4.3.4, we detail the fusion process where we meticulously integrate the locally-enhanced global features with the globallyenhanced local features. This fusion results in a dualenhanced feature, which effectively combines detailed local information with comprehensive global insights. This robust integration provides an enriched and holistic representation of the high-resolution image, significantly enhancing the model’s performance in various high-level visual analysis.

h×hh×d. Likewise, the global-perspective

Floc ∈ Rw

sub-features [Fglo1 ,Fglo2 ,··· ,FgloN ] are amalgamated to form the global-perspective feature Fglo ∈ Rw

h×hh×d. This meticulous combination process ensures that the localperspective feature effectively encapsulates fine-grained details, while the global-perspective feature maintains a coherent understanding of the broader contextual information.

##### 4.3.2 Global-Perspective Enhancement After obtaining the local-perspective feature Floc ∈ Rw

h×wl×d and global-perspective feature Fglo ∈ Rw

h×wl×d, the next crucial step involves interacting and fusing these two feature sets to derive more robust and comprehensive features. A straightforward approach would be to leverage cross-attention between Floc and Fglo to enable interaction between local and global information. However, applying cross-attention directly to such high-resolution features can lead to significant out-of-memory issues during training. To circumvent this, we propose a novel dual-perspective enhancement method that interacts with local and global features in a more efficient manner, mitigating these computational challenges.

##### 4.3.1 Sub-features Combination

Upon obtaining the local-perspective sub-images [I1loc,I2loc,··· ,INloc] and global-perspective sub-images [I1glo,I2glo,··· ,INglo] from DCM, we proceed to extract their corresponding features using Equ. (3) and Equ.(4). This extraction yields local-perspective sub-features [Floc1 ,Floc2 ,··· ,FlocN ] and global-perspective sub-features [Fglo1 ,Fglo2 ,··· ,FgloN ]. As illustrated in Fig. 3, the sub-features from both perspectives are then systematically recombined. Specifically, the local-perspective sub-features [Floc1 ,Floc2 ,··· ,FlocN ] are aggregated to construct the comprehensive local-perspective feature

Specifically, for global-perspective enhancement, as illustrated in Fig. 4, we first crop both the local-perspective and global-perspective features from a global perspective.

This operation can be mathematically formulated as follows:

[Gglo1 ,Gglo2 ,··· ,GgloN ] = Cglo(Fglo), (18)

[Lglo1 ,Lglo2 ,··· ,LgloN ] = Cglo(Floc), (19) where Cglo(·) denotes the global-perspective cropping operation. [Gglo1 ,Gglo2 ,··· ,GgloN ] and [Lglo1 ,Lglo2 ,··· ,LgloN ] represent the global and local sub-features, respectively, resulting from this global-perspective cropping.

To infuse local sub-feature information into the global sub-features, we perform a cross-attention operation between corresponding local and global sub-features. This can be formulated as follows:

  ,

  

T

Ggloi · Wq · Lgloi · Wk

Agloi = Softmax

√

d

(20)

V gloi = Agloi · Lgloi · Wv, (21) where Wq,Wk,Wv ∈ Rd×d are learnable embedding matrices. Agloi ∈ Rh

lwl×hlwl are the attention map between local sub-feature and global sub-feature. V gloi ∈ Rw

l×hl×d

are the i-th global-enhanced sub-feature. Finally, the set of enhanced sub-features

[V glo1 ,V glo2 ,··· ,V gloN ] are combined to form the globallyenhanced feature V glo ∈ Rw

h×hh×d. This procedure ensures that the resulting feature representation captures rich, multi-scale information by effectively integrating local details with the global context, thereby enhancing the overall robustness and expressiveness of the model.

##### 4.3.3 Local-Perspective Enhancement

Similarly, Local-Perspective Enhancement aims to refine local features by integrating global contextual information. This process ensures that the local features not only retain fine-grained details but also benefit from the broader context provided by the global features.

First, the local-perspective and global-perspective features are cropped into sub-features through a localperspective cropping operation. This is formulated as follows:

[Gloc1 ,Gloc2 ,··· ,GlocN ] = Cloc(Fglo), (22) [Lloc1 ,Lloc2 ,··· ,LlocN ] = Cloc(Floc), (23)

where Cloc(·) denotes the local-perspective cropping operation. The resulting sub-features, [Gloc1 ,Gloc2 ,··· ,GlocN ] and [Lloc1 ,Lloc2 ,··· ,LlocN ], represent the global and local sub-features respectively, generated from this localperspective cropping.

Next, we employ a cross-attention mechanism to facilitate interaction between local and global sub-features, enhancing the local features with global context. The formulation of this interaction is as follows:

Aloci = Softmax

Lloci · Wq · Gloci · Wk T

√

d

, (24)

V loci = Aloci · Gloci · Wv, (25)

where Wq,Wk,Wv ∈ Rd×d are learnable embedding matrices used for query, key, and value transformations, re-

spectively. The attention map Aloci ∈ Rh

lwl×hlwl captures the relationships between local and global sub-features. The resulting enhanced feature V loci ∈ Rw

l×hl×d is the i-th local-enhanced sub-feature.

Finally, the set of local-enhanced sub-features

[V loc1 ,V loc2 ,··· ,V locN ] are aggregated to form the local-enhanced feature V loc ∈ Rw

h×hh×d. This process ensures that the local features are enriched with complementary global contextual information, providing a more comprehensive and robust representation of the original high-resolution image.

##### 4.3.4 Dual-Perspective Fusion

After obtaining the global-enhanced feature V glo and the local-enhanced feature V loc, the next critical step involves fusing these features to create a comprehensive representation that leverages the strengths of both perspectives. To achieve this, we employ a concatenation-based method that effectively combines the global and local features.

Initially, we utilize two separate embedding layers to reduce the dimensionality of the features. This dimensionality reduction step is essential to ensure that the subsequent concatenation is computationally efficient and to highlight the most salient aspects of each feature set. The embedding operation can be described as follows:

###### V˜ glo = V gloWglo, V˜ loc = V locWloc, (26)

where Wglo,Wloc ∈ Rd×d2 are learnable projection matrices for the global and local features, respectively. These matrices transform the original features into a lowerdimensional space, thereby emphasizing their most critical components.

Next, we concatenate the embedded global and local features along the channel dimension to form the dualenhanced feature. This concatenation ensures that both global contextual information and local detailed information are retained and integrated. The fusion process is formulated as follows:

V dual = [V˜ glo;V˜ loc], (27)

where [·;·] denotes the concatenation operation along the channel dimension. As a result, the dual-enhanced feature V dual ∈ Rw

h×hh×d encapsulates a comprehensive view of the input image, combining the strengths of both perspectives to produce a robust and informative representation.

### 5. Experiments

#### 5.1. Evaluations

To rigorously assess the effectiveness, robustness, and versatility of the proposed INF-LLaVA, we conduct extensive evaluations across a diverse set of vision-language benchmarks. By leveraging a broad spectrum of datasets, we ensure a comprehensive evaluation of the model’s capabilities across various dimensions and contexts. These benchmarks include:

- • ScienceQA-img [63]: ScienceQA is a large-scale multimodal dataset containing 21,208 science questions from elementary and high school curricula. Each question is annotated with lectures and explanations to provide comprehensive contextual understanding, making this dataset ideal for testing the model’s ability to interpret and generate accurate, context-aware responses to science-related queries.
- • OKVQA [66]: This open-ended visual question answering dataset requires the model to utilize external knowledge sources to answer questions accurately. It challenges the model’s proficiency in integrating visual content with external textual information, providing a robust test of the model’s ability to generate responses based on prior knowledge and visual understanding.
- • SEEDBench [43]: SEEDBench is a comprehensive benchmark tailored specifically for evaluating Multimodal Large Language Models (MLLMs). It spans 12 evaluation dimensions, including comprehension of both image and video modalities. This benchmark offers a robust framework for assessing the performance of the model across a diverse array of multimodal tasks, ensuring a holistic evaluation.
- • MMBench [55]: MMBench is a multi-modality benchmark offering a comprehensive evaluation pipeline. It features a curated dataset and introduces the innovative CircularEval strategy using ChatGPT to enhance model prediction assessment. MMBench ensures a holistic evaluation of the model’s multi-modal capabilities by providing a structured and thorough testing environment.

- • MMBench-CN [55]: The Chinese version of MMBench, MMBench-CN, facilitates a direct comparison of Vision-Language Model (VLM) performance in both English and Chinese contexts with verified translations. This benchmark tests the model’s multilingual and cross-cultural adaptability, making it an essential tool for evaluating the robustness of the model in diverse linguistic settings.
- • AI2D [39]: AI2D is a dataset comprising illustrative diagrams aimed at research on diagram understanding and associated question answering. This benchmark challenges the model’s ability to interpret and reason about structured graphical information, testing its proficiency in understanding and generating responses based on diagrammatic data.
- • LLaVA-Bench-in-the-wild [54]: This benchmark evaluates the capabilities of large multimodal models on real-world tasks and domains. Featuring detailed images and curated questions, LLaVA-Bench-in-thewild tests the model’s performance on diverse and challenging datasets regularly encountered in practical applications, ensuring its real-world applicability.
- • MMMU [105]: MMMU is a benchmark designed to evaluate multimodal models on college-level tasks spanning multiple disciplines. It requires advanced reasoning and subject matter expertise, thereby testing the depth of the model’s understanding and its ability to handle complex, cross-disciplinary questions, making it a critical benchmark for assessing advanced cognitive capabilities.

#### 5.2. Implementation Details

The vision encoder used in our implementation is the CLIP-ViT-L/14, which has demonstrated strong performance in visual tasks. The large language model (LLM) employed is LLaMA3-8B, which provides robust language understanding and generation capabilities. The training of INF-LLaVA is meticulously structured into two distinct stages to ensure optimal alignment and fine-tuning of the model components.

In the first stage, the pretraining phase, our primary objective is to align the features extracted by the vision encoder with the word embeddings generated by the LLM. To achieve this, we freeze both the vision encoder and the LLM during the pretraining phase. This allows us to focus on training the DEM and the projector. The model is pretrained using the CC-595k dataset [54], comprising a substantial collection of aligned image-text pairs, for 1 epoch. We utilize the AdamW optimizer [59] with a learning rate of 1 × 10−3 and employ a cosine learning rate schedule to smoothly adjust the learning rate during training. A global

batch size of 256 is used to ensure efficient utilization of computational resources.

In the subsequent stage, the supervised fine-tuning (SFT) phase, our goal is to refine the model’s performance on downstream tasks. During this phase, we freeze the vision encoder and proceed to train the DEM, the projector, and the LLM. The model is fine-tuned using the LLaVA-656K mixture dataset [52], which contains an extensive and diverse set of annotations to support comprehensive learning. We employ a lower learning rate of 2 × 10−5 and a batch size of 128 to carefully fine-tune the model parameters, ensuring precise adjustments without overfitting.

#### 5.3. Quantitative Analysis

As shown in Tab. 1, we conduct a comprehensive comparison of the proposed INF-LLaVA with existing stateof-the-art (SOTA) Multimodal Large Language Models (MLLMs) across 8 widely recognized benchmarks. The results clearly demonstrate the superior performance of INFLLaVA. Specifically, INF-LLaVA achieves an impressive score of 75.71 on the ScienceQA-img benchmark and 70.35 on the MMBench benchmark. These scores represent significant improvements over other leading MLLMs. Notably, INF-LLaVA outperforms QWen-VL-Chat by a considerable margin. It is important to highlight that QWenVL-Chat leverages a substantially larger amount of data, utilizing 5 billion examples for pretraining, 76.8 million examples for multi-task training, and 350,000 in-house examples for supervised fine-tuning. In stark contrast, INFLLaVA achieves its superior results with only 595,000 examples for pretraining and 665,000 examples for finetuning. Additionally, as detailed in Tab. 2, we trained an enhanced version of INF-LLaVA, marked as INF-LLaVA*, using a larger dataset. Our observations indicate that when fed with more data, the performance of INF-LLaVA improves even further across most benchmarks. For instance, on the LLaVA-bench-in-the-wild benchmark, INF-LLaVA* achieves an improvement of over ten points.

#### 5.4. Qualitative Analysis

##### 5.4.1 Comparison of Different Image Resolutions

To thoroughly investigate the impact of different image resolutions on the performance of INF-LLaVA, we conducted a series of experiments using various input image resolutions. As demonstrated in Fig. 5, leveraging high-resolution images enables INF-LLaVA to provide more accurate answers to complex questions, particularly those requiring fine-grained perception. For instance, in the second case of the second row in Fig. 5, the screen displays multiple team names along with their corresponding scores. Identifying the team name associated with a specific score is a challenging task. However, with an input image resolu-

tion of 1008 × 1008 pixels, INF-LLaVA accurately identifies the KDE team as the one that scored 16 points. In contrast, when using a lower resolution of 336 × 336 pixels, INF-LLaVA provides an incorrect response. Moreover, in the second case of the third row, the task involves identifying the contents of a purple box that is closed, making it difficult to ascertain what is inside. With an input resolution of 1008 × 1008 pixels, INF-LLaVA successfully recognizes the ”ringdoll” text on the box lid and accurately infers that the box might contain a ringdoll. Conversely, using a 336 × 336 pixels resolution, INF-LLaVA incorrectly concludes that the box contains no objects, demonstrating a lack of evidence-based reasoning. These observations underscore the importance of high-resolution images in enhancing the model’s ability to perceive and interpret complex visual details, thereby improving its overall accuracy and effectiveness.

##### 5.4.2 Comparison with LLaVA 1.5

In Fig. 6 and Fig. 7, we present a detailed comparison between INF-LLaVA and LLaVA-1.5, both of which utilize the same training dataset. The results clearly demonstrate that INF-LLaVA outperforms LLaVA-1.5 in several critical areas. Firstly, in terms of text recognition capabilities, INFLLaVA exhibits greater accuracy. For instance, as shown in the first case of Fig. 6, when asked, ”What is the tagline with ’Wendell Rodricks’ name?”, INF-LLaVA accurately identifies the tagline, whereas LLaVA-1.5 provides an incorrect response. This highlights INF-LLaVA’s superior ability to discern and interpret textual information within high-resolution images. Secondly, regarding the issue of hallucination, INF-LLaVA demonstrates a clear advantage. When uncertain about an answer, INF-LLaVA candidly indicates its uncertainty, whereas LLaVA-1.5 tends to fabricate responses, leading to hallucination problems. For example, in the last case of Fig. 6, when asked to provide Chinese text, LLaVA-1.5 incorrectly responds with English text, whereas INF-LLaVA acknowledges its inability to answer, thereby avoiding erroneous information. Thirdly, in terms of counting accuracy, INF-LLaVA benefits from the enhanced details provided by high-resolution images, enabling more precise perception and reasoning. For instance, as depicted in the first case of Fig. 7, INF-LLaVA correctly identifies the number of birds in the image, while LLaVA1.5 fails to provide an accurate count. This demonstrates INF-LLaVA’s superior capabilities in tasks requiring detailed visual analysis. These comparisons consistently highlight the superior performance of INF-LLaVA across various complex tasks, emphasizing its advanced capabilities in text recognition, reducing hallucination issues, and enhancing counting accuracy.

- Table 1: Comparison with State-of-the-Art (SOTA) Methods on Vision-Language Benchmarks. Our proposed methods consistently outperform existing SOTA approaches across a variety of benchmarks. The highest score is indicated in bold, and the second highest score is underlined. Note that INF-LLaVA* is trained using a larger dataset, as detailed in Tab. 2.

Model Source ScienceQA-img OKVQA SEEDBench MMB MMB-CN AI2D LLaVA-wild MMMU

Flamingo-80B [2] NeurIPS’22 - 50.60 - - - - - BLIP-2 [45] ICML’23 - 45.90 - - - - - InstructBLIP-7B [20] NeurIPS’23 60.50 - - - - - - InstructBLIP-13B [20] NeurIPS’23 63.10 - - - - - - LLaVA [54] NeurIPS’23 - - 37.00 38.70 36.40 - 62.80 IDEFICS-9B [42] NeurIPS’23 - 38.40 - - - - - GILL [41] NeurIPS’23 - - 52.50 38.20 - - - 28.80 CM3Leon [102] arXiv’23 - 23.80 - - - - OpenFlamingo [5] arXiv’23 - 37.80 - - - - - Shikra [15] arXiv’23 - 47.16 - 58.80 - - - MiniGPT-4 [115] arXiv’23 - 37.50 - - - - - Qwen-VL-Chat [6] arXiv’23 68.20 56.60 65.40 - - 57.7 - MiniGPT-v2 [14] arXiv’23 - 57.80 - - - - - OtterHD-8B [44] arXiv’23 - - - 58.30 - - - ImageBind-LLM [29] arXiv’23 51.40 51.66 - - - - - ChatBridge [114] arXiv’23 - 45.20 - - - - - AnyMAL-13B [72] arXiv’23 52.70 33.10 - - - - - AnyMAL-70B [72] arXiv’23 70.80 42.60 - - - - - Emu-I [86] ICLR’24 - 49.20 - - - - - MGIE [26] ICLR’24 - - 28.80 6.60 - - - 25.60 DreamLLM [23] ICLR’24 - 52.20 - 58.20 - - - mPLUG-Owl2 [100] CVPR’24 68.70 57.70 57.80 - - - - Monkey [49] CVPR’24 69.40 61.30 - - - 57.90 - LLaVA1.5 [52] CVPR’24 66.80 - 66.10 64.30 58.30 - 65.40 36.40 Unified-IO 2 [61] CVPR’24 78.60 50.20 - - - - - Honeybee [12] CVPR’24 - - 64.50 70.10 - - 67.10 OneLLM [28] CVPR’24 63.40 58.90 61.20 60.00 - - - LoRA-Sparse [84] CVPR’24 68.40 - 58.80 - - - 63.40 Pink [97] CVPR’24 - 59.50 - - - - - Prompt Highlighter [109] CVPR’24 - - - 69.50 - - - TIVE [58] arXiv’24 69.20 - 62.20 65.80 57.40 - - GenLLaVA [32] arXiv’24 - - 63.50 65.00 - - - 29.70 AnyGPT [107] arXiv’24 - - 44.50 36.00 - - - 30.60 DeepStack-L-HD [70] arXiv’24 - - - - - - - 35.60 LLaVA-Phi [116] arXiv’24 68.40 - - 59.80 - - - AcFormer [117] arXiv’24 69.40 - - 68.40 - - - DeepStack-L-HD [70] arXiv’24 - - - - - - - 35.60 ConvLLaVA [27] arXiv’24 - - 70.20 68.70 - - - 35.80 LLAVA-HR [64] arXiv’24 65.10 58.90 64.20 - - - - -

INF-LLaVA Ours 75.71 60.67 70.47 70.35 63.23 60.59 67.50 37.00 INF-LLaVA* Ours 77.44 57.04 72.65 74.38 64.57 75.42 78.20 37.20

#### 5.5. Ablation Studies

##### 5.5.1 Diverse Image Resolutions

We conduct a series of experiments to evaluate the impact of different input image resolutions on the performance of INF-LLaVA, as detailed in Tab. 3. Initially, we evaluate the model using single-resolution inputs and observe that higher-resolution images generally yield better results com-

pared to lower-resolution images (e.g., 336 × 336). For instance, on the POPE benchmark, INF-LLaVA achieves scores of 85.86, 87.32, 87.17, and 88.12 for resolutions of 336 × 336, 672 × 672, 1008 × 1008, and 1344 × 1344, respectively. This trend underscores the benefit of highresolution images in capturing finer details, which enhances the model’s performance. However, it is noteworthy that the performance improvement is not strictly linear with in-

[Figure 77]

[Figure 78]

Q：How many gs are mentioned in the poster? INF-LLaVA(10081008×10081008px)：In thethe image, thethe poster mentions 3G. INF-LLaVA(336336×336336px)：In thethe image, thethe poster mentions 186.

Q：Who's name is written on the white booklet?

| |
|---|

INF-LLaVA(10081008×10081008px)：Mira Schendel

| |
|---|

|[Figure 79]|
|---|

INF-LLaVA(336336×336336px)：Mirka schendel

|[Figure 80]<br><br>|
|---|

[Figure 81]

Q：What is hipchat? 1008 1008

Q：What team has 16 points? INF-LLaVA(10081008×10081008px) ： Team with 16 points is KDE team. INF-LLaVA(336336×336336px)： Team with 16 points is POT team.

[Figure 82]

INF-LLaVA(1008×1008px): Hipchat is a communication platform that allows teams to collaborate and share information through various channels, such as chat, video, and file sharing.

and

| |
|---|

| |
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

and

INF-LLaVA(336336×336336px)：HipChat is a man.

[Figure 85]

Q：What is in the box?

[Figure 86]

Q：What number is on the left bottle?

INF-LLaVA(1008×1008px)： The box contains a purple item, which could be a ringdoll in it. The box is open, and the purple item is visible.

1008 1008 The box a purple

| |
|---|

a . The box is purple is .

INF-LLaVA(10081008×10081008px)：13 is on the left bottle.

|[Figure 87]|
|---|

| |
|---|

|[Figure 88]|
|---|

INF-LLaVA(336336×336336px)：41 is on the left bottle.

INF-LLaVA(336336×336336px)：Nothing is in the box

[Figure 89]

| |
|---|

Q：What number is thethe coin on thethe middle right? INF-LLaVA(10081008×10081008px) ： Coin on middle right is number 46. INF-LLaVA(336336×336336px): Coin on middle right is number 45.

Q：What number is written on top right? INF-LLaVA(10081008×10081008px)：Number written on top right is 23. INF-LLaVA(336336×336336px) Number written on top right is 28.

[Figure 90]

| |
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

- Figure 5: Chat comparison using different image resolutions. Certain regions of the input high-resolution images are zoomed in for enhanced visualization.

Infographic VQA [68], SEEDBench [43], MM-Vet [104], OKVQA [66], and LLaVA-in-the-wild [54]. This trend indicates that higher resolutions enable INF-LLaVA to capture more detailed visual information, thereby enhancing performance. However, it is important to note that this upward trend does not uniformly apply across all benchmarks. For instance, on GQA [36], AI2D [39], and TextCaps [83], the optimal resolution is 1008×1008. On the MMMU [105] and DOCVQA [69] benchmarks, the best performance is achieved at a resolution of 672 × 672. This variability suggests that increasing resolution does not necessarily lead to continuous performance improvement for all tasks. In certain cases, higher resolutions may introduce redundancies or distortions due to image enlargement, which can negatively affect model performance.

creasing resolution. The optimal resolution varies across different benchmarks, potentially due to variations in image content and the inherent difficulty levels of the benchmarks. Inspired by previous works [24,49], we explore the synergy of integrating both low-resolution and high-resolution images to capture comprehensive global and local information. When combining low-resolution and high-resolution features, we perform an element-wise addition of the two feature sets. As demonstrated in Tab. 3, this dual-resolution approach enables INF-LLaVA to achieve its best performance on benchmarks such as SEEDBench and MMVet when using 336×336 and 1008×1008 resolutions concurrently. The superior results obtained with this dual-resolution strategy highlight its effectiveness in balancing detailed visualization with global contextual understanding. Consequently, we adopt this resolution setting as the default configuration in subsequent ablation experiments.

5.5.2 Variants of Dual-perspective Enhancement Module

To further investigate the impact of varying image resolutions, we conducted a comprehensive set of experiments testing INF-LLaVA across multiple benchmarks with different image resolutions. As illustrated in Fig. 8, we observe a general trend where performance improves with increased resolution on most benchmarks, such as POPE [46],

As illustrated in Fig. 4, DEM integrates two crucial enhancement operations: Global-Perspective Enhancement and Local-Perspective Enhancement. In this section, we conduct a series of ablation experiments to rigorously evaluate the effectiveness of these two enhancements.

|[Figure 93]|
|---|

[Figure 94]

| |
|---|

User: What is the tagline with 'Wendell Rodricks' name?

[Figure 95]

INF-LLaVA (Ours): The tagline with 'Wendell Rodricks' name is "Wendell Rodricks now at Wills Lifestyle"

[Figure 96]

LLaVA-1.5: The tagline with 'Wendell Rodricks' name is " Wendell rodricks now wendell rodricks womenswear "

|[Figure 97]|
|---|

[Figure 98]

| |
|---|

User: What is the ‘heading’?

[Figure 99]

INF-LLaVA (Ours): The heading of the document is "Request for Change".

[Figure 100]

###### LLaVA-1.5: The heading is "Change of Address".

|[Figure 101]|
|---|

User: Based on the image, in which country does Tina Fey have the highest level of total awareness?

[Figure 102]

| |
|---|

[Figure 103]

INF-LLaVA (Ours): In the image, Tina Fey has the highest level of total awareness in Australia.

| |
|---|

[Figure 104]

LLaVA-1.5: In the image, Tina Fey has the highest level of total awareness in the United States.

|[Figure 105]|
|---|

[Figure 106]

User: What are all the Chinese scene text in the image?

INF-LLaVA (Ours): The image features a book with a painting of a city skyline and Chinese scene text. However, without knowing the specific content of the book, it is not possible to provide a detailed description of the text.

[Figure 107]

[Figure 108]

LLaVA-1.5: The Chinese scene text in the image reads " Chinese cityscape. "

- Figure 6: Chat comparison between INF-LLaVA and LLaVA-1.5. Certain regions of the input high-resolution images are zoomed in for enhanced visualization.

First, we evaluate DEM configured with only the globalperspective enhancement (i.e., DEM w/ global). The results, presented in the first line of Tab. 4, show a significant reduction in performance across all benchmarks when the local-perspective enhancement is excluded. This clear decline in performance underscores the critical role that localperspective enhancement plays in capturing fine-grained details necessary for high model accuracy.

Next, we test DEM with only the local-perspective enhancement (i.e., DEM w/ local). As depicted in the second line of Tab. 4, DEM w/ local yields slightly better performance than DEM w/ global on the POPE, SEEDBench, and MM-Vet benchmarks. Conversely, it performs slightly worse on the LLaVa-Bench-in-the-wild benchmark compared to the DEM w/ global. However, it is important to highlight that both configurations fall short of the complete

Q：What is one of the brands being advertised? INF-LLaVA：One of the brands being advertised is Yamaha. LLava1.5：One of the brands being advertised is YKm

Q：What time is it?

[Figure 109]

[Figure 110]

INF-LLaVA：7:28

| |
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

LLava1.5：10:00

[Figure 113]

Q：How many birds are there?

Q：Where is this mug featuring?

[Figure 114]

| |
|---|

INF-LLaVA：There are six birds in the image.

###### INF-LLaVA：Daytona Beach, Florida.

|[Figure 115]|
|---|

LLava1.5. ：There are five birds in the image..

LLava1.5：Delaware

[Figure 116]

[Figure 117]

Q：What brand of refrigerator is this? The .The

Q：What is the train number?

| |
|---|

| |
|---|

. INF-LLaVA：The image shows two refrigerators.The brand is metalfrio.

|[Figure 118]|
|---|

INF-LLaVA：The train number is 6.

|[Figure 119]|
|---|

LLava1.5. ：The brand of refrigerator is maytag..

LLava1.5. ：The train number is 313..

[Figure 120]

Q：What is the date displayed?

Q：what is thethe year on thethe calender?

[Figure 121]

INF-LLaVA：The date displayed is September 26th

INF-LLaVA：The year on the calender is 2010

| |
|---|

|[Figure 122]|
|---|

LLava1.5：The date displayed is 14th

LLava1.5：The year on the calender is 2014

|[Figure 123]|
|---|

- Figure 7: Chat comparison between INF-LLaVA and LLaVA-1.5. Certain regions of the input high-resolution images are zoomed in for enhanced visualization.

- Table 2: Dataset Details for Training INF-LLaVA*, including the datasets and corresponding Q/A pairs used in the pretraining (Stage 1) and supervised fine-tuning (Stage 2) stages.

combined use of both perspectives in DEM results in superior performance by effectively capturing unique and complementary aspects of the visual data. This integrative approach ensures a balanced and comprehensive understanding, ultimately leading to significantly enhanced model performance across a wide range of benchmarks.

##### Dataset Q/A pairs Total

- Stage1: Pretraining

|LAION-CCSBU [82]|558K<br><br>|1.26M<br><br>|
|---|---|---|
|ALLaVA-4V-Caption [13]|708K| |

- Stage2: Supervised Fine-tuning

##### 5.5.3 Fusion methods in DEM

Fig. 4 illustrates the process of obtaining dual-enhanced features, which necessitate an effective fusion method to combine the local-enhanced feature and global-enhanced feature. In our proposed method, we utilize linear layers to reduce the dimensionality of the features, followed by concatenation to form the dual-enhanced features. This section explores the impact of various fusion methods on performance. Different techniques, including 3 × 3 convolution, element-wise multiplication, weighted addition, max pooling, and addition, were employed to fuse the features. As shown in Tab. 5, the experimental results provide a comprehensive comparison of these methods across several benchmarks. Notably, the embed-and-concat method (LinearConcat) achieves superior performance on most benchmarks. It outperforms other methods, achieving the highest scores on the POPE, SEEDBench, and LLaVA-wild benchmarks. This indicates that by carefully embedding and con-

|LLaVA-Instruct [52]<br><br>|665K|1.3M<br><br>|
|---|---|---|
|ALLaVA-4V-Instruction [13]|692K| |
|ShareGPT4V [16] DocVQA [89] DVQA [38] AI2D [39]<br><br>|25K<br><br>| |

DEM, which includes both enhancement operations. These ablation studies conclusively demonstrate that the

POPE

Infographic VQA val

SEEDBench

NoCaps val

111.2

27.0

- 66

- 67

- 68

- 69

- 70

Max Value

Max Value

88.0

111.0

26.5

Performance

Performance

Performance

Performance

87.5

110.8

26.0

87.0

110.6

86.5

25.5

110.4

Max Value

Max Value

86.0

110.2

25.0

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

Resolution

Resolution

Resolution

Resolution

MM-Vet

OKVQA 2014 val

LLaVA-in-the-wild

GQA

64.0

- 29

- 30

- 31

- 32

- 33

- 34

Max Value

60.6

70.0

63.8

60.4

Performance

Performance

Performance

Performance

67.5

60.2

63.6

65.0

60.0

63.4

62.5

59.8

63.2

60.0

59.6

Max Value

Max Value

Max Value

63.0

59.4

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

Resolution

Resolution

Resolution

Resolution

AI2D

MMMU val

DOCVQA val

HallusionBenchmark

41.5

39.0

60.5

Max Value

30.0

38.5

41.0

Performance

Performance

Performance

Performance

60.0

38.0

29.5

Max Value

40.5

59.5

37.5

29.0

37.0

40.0

59.0

Max Value

Max Value

36.5

28.5

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

Resolution

Resolution

Resolution

Resolution

TextCaps val

SeedBench 2

ScienceQA

ChartQA

59.6

16.2

103.0

Max Value

Max Value

80.6

59.5

102.5

16.0

Performance

Performance

Performance

Performance

59.4

80.4

102.0

15.8

59.3

101.5

80.2

59.2

101.0

80.0

15.6

59.1

100.5

79.8

15.4

Max Value

Max Value

59.0

100.0

79.6

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

336 672 1008 1344 1680

Resolution

Resolution

Resolution

Resolution

- Figure 8: Performance evaluation of INF-LLaVA across various benchmarks and image resolutions. The best performance for each benchmark is highlighted with a red star, illustrating the optimal resolution for INF-LLaVA in different contexts. Unlike the multi-resolution approach shown in the latter half of Tab. 3, this evaluation used a consistent single resolution for each experiment to explore the impact of resolution.

catenating the features, our method effectively integrates the global and local perspectives.

##### 5.5.4 Module Ablation

In this section, we conduct a thorough ablation study to examine the effectiveness of the proposed modules. As detailed in Tab. 6, we first evaluate the performance when using only one type of cropping method. The first two lines show that using global-perspective cropping alone (w/ DCM

global) consistently outperforms local-perspective cropping alone (w/ DCM local). This result suggests that globalperspective cropping, which maintains continuous visual information, is beneficial for model performance. In contrast, local-perspective cropping may inadvertently segment complete objects into several sub-images, complicating the recognition and understanding processes. Next, we investigate the impact of combining both cropping features through element-wise addition of the features (w/ DCM)

- Table 3: Performance on the POPE, SEEDBench, MM-Vet, LLaVA-Bench-in-the-wild benchmarks using different resolutions.

Resolution POPE SEEDBench MM-Vet LLaVA-wild

|[336]<br><br>|85.86 65.99 28.81 58.20|
|---|---|
|[672] [1008] [1344] [1680]|87.32 69.73 32.29 63.80<br><br>87.17 70.03 32.52 64.20<br><br>88.12 69.99 34.13 60.90<br><br><br>87.74 69.77 34.13 71.00<br><br>|
|[336+672] [336+1008] [336+1344] [336+1680]|87.61 69.94 31.74 59.40<br><br>86.94 70.47 34.50 67.50<br><br>87.36 69.04 34.04 61.50<br><br><br>87.31 69.35 33.99 64.50|

- Table 4: Performance comparison of different variants of Dual-perspective Enhancement Module (DEM). DEM w/ global represents the module utilizing only Global-Perspective Enhancement, while DEM w/ local represents the module utilizing only Local-Perspective Enhancement. This comparison elucidates the individual impact of global and local enhancements on model performance.

POPE SEEDBench MM-Vet LLaVA-wild

DEM w/ global 86.15 69.06 31.43 62.90 DEM w/ local 86.24 69.45 31.66 62.70 DEM 86.94 70.47 34.50 67.50

instead of DEM. As evident from the third line of Tab. 6, this approach results in even lower performance than using only one type of cropping. We hypothesize that this degradation in performance arises due to the differing information densities of the tokens produced by each cropping method, making direct element-wise addition a suboptimal fusion strategy. Finally, we consider the complete DEM module, which integrates both cropping methods in a more sophisticated manner. The last row of Tab. 6 demonstrates that DEM achieves the highest performance across all benchmarks. This clearly indicates that the combined use of local- and global-perspective enhancements, when managed effectively, significantly improves model performance. These findings underscore the effectiveness of the proposed DCM and DEM modules and highlight the importance of thoughtful feature integration strategies in enhancing model capabilities.

### 6. Conclusion

In this paper, we proposed INF-LLaVA, a novel MLLM designed for high-resolution image perception and reasoning. INF-LLaVA leverages two innovative modules: Dual-perspective Cropping Module (DCM), which crops high-resolution images into sub-images from both local

and global perspectives, and Dual-perspective Enhancement Module (DEM), which efficiently fuses these features to obtain dual-enhanced features. Extensive experiments demonstrate that these modules significantly enhance INFLLaVA’s ability to understand high-resolution images, resulting in outstanding performance across various benchmarks. This work establishes a new state-of-the-art in vision-language tasks.

### References

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1, 3
- [2] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 3, 11
- [3] Anthropic. Introducing claude, 2023. 3
- [4] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015. 1

- Table 5: Comparison of different feature fusion methods in DEM. The performance is evaluated using various benchmarks. Multiplication denotes element-wise multiplication.

POPE SEEDBench MM-Vet LLaVA-wild

Conv(3x3) 86.64 69.63 25.60 65.70 Multiplication 86.88 69.80 30.69 62.80 Weighted Addition 86.21 69.61 33.53 62.70 Maxpool 85.92 69.66 24.45 63.20 Addition 86.01 70.03 32.02 60.70 Linear-Concat 87.04 70.62 33.44 66.40

- Table 6: Ablation studies of the proposed DEM module. w/ DCM local indicates the use of DEM with only local-perspective cropping, omitting global-perspective cropping. w/ DCM global indicates the use of DEM with only global-perspective cropping, omitting local-perspective cropping. w/ DCM indicates the approach where local-perspective and global-perspective features are combined through element-wise addition.

POPE SEEDBench MM-Vet LLaVA-wild

w/ DCM local 86.12 66.60 27.80 56.30 w/ DCM global 86.15 70.06 32.43 62.90 w/ DCM 85.77 68.89 18.90 60.10 w/ DCM+DEM 87.04 70.62 33.44 66.40

- [5] A. Awadalla, I. Gao, J. Gardner, J. Hessel, Y. Hanafy, W. Zhu, K. Marathe, Y. Bitton, S. Gadre, S. Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023. 11
- [6] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 3, 11
- [7] Z. Bai, P. Wang, T. Xiao, T. He, Z. Han, Z. Zhang, and M. Z. Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024. 1
- [8] X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. 3
- [9] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3
- [10] R. Cadene, H. Ben-Younes, M. Cord, and N. Thome. Murel: Multimodal relational reasoning for visual question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1989–1998,

2019. 2

- [11] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the

- IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 3
- [12] J. Cha, W. Kang, J. Mun, and B. Roh. Honeybee: Localityenhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13817–13827, 2024. 11
- [13] G. H. Chen, S. Chen, R. Zhang, J. Chen, X. Wu, Z. Zhang, Z. Chen, J. Li, X. Wan, and B. Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 14
- [14] J. Chen, D. Zhu, X. Shen, X. Li, Z. Liu, P. Zhang, R. Krishnamoorthi, V. Chandra, Y. Xiong, and M. Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023. 11
- [15] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 11
- [16] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793,

2023. 14

- [17] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. 3
- [18] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton,

- S. Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023. 3
- [19] M. Cornia, M. Stefanini, L. Baraldi, and R. Cucchiara. Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10578–10587, 2020. 1
- [20] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024. 1, 3, 11
- [21] A. Das, S. Kottur, K. Gupta, A. Singh, D. Yadav, J. M. Moura, D. Parikh, and D. Batra. Visual dialog. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 326–335, 2017. 1
- [22] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 2
- [23] R. Dong, C. Han, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J. Sun, H. Zhou, H. Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023. 11
- [24] X. Dong, P. Zhang, Y. Zang, Y. Cao, B. Wang, L. Ouyang, S. Zhang, H. Duan, W. Zhang, Y. Li, et al. Internlmxcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512, 2024. 2, 3, 12
- [25] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 2, 3
- [26] T.-J. Fu, W. Hu, X. Du, W. Y. Wang, Y. Yang, and Z. Gan. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102,

2023. 11

- [27] C. Ge, S. Cheng, Z. Wang, J. Yuan, Y. Gao, J. Song, S. Song, G. Huang, and B. Zheng. Convllava: Hierarchical backbones as visual encoder for large multimodal models. arXiv preprint arXiv:2405.15738, 2024. 11
- [28] J. Han, K. Gong, Y. Zhang, J. Wang, K. Zhang, D. Lin, Y. Qiao, P. Gao, and X. Yue. Onellm: One framework to align all modalities with language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26584–26595, 2024. 11
- [29] J. Han, R. Zhang, W. Shao, P. Gao, P. Xu, H. Xiao, K. Zhang, C. Liu, S. Wen, Z. Guo, et al. Imagebindllm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023. 11
- [30] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, et al. A survey on vision transformer. IEEE transactions on pattern analysis and machine intelligence, 45(1):87–110, 2022. 1
- [31] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 1

- [32] J. Hernandez, R. Villegas, and V. Ordonez. Generative visual instruction tuning. arXiv preprint arXiv:2406.11262,

2024. 11

- [33] A. Hu, Y. Shi, H. Xu, J. Ye, Q. Ye, M. Yan, C. Li, Q. Qian, J. Zhang, and F. Huang. mplug-paperowl: Scientific diagram analysis with the multimodal large language model. arXiv preprint arXiv:2311.18248, 2023. 1
- [34] A. Hu, H. Xu, J. Ye, M. Yan, L. Zhang, B. Zhang, C. Li, J. Zhang, Q. Jin, F. Huang, et al. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895, 2024. 1
- [35] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [36] D. A. Hudson and C. D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709,

2019. 12

- [37] C. Jiang, H. Xu, M. Dong, J. Chen, W. Ye, M. Yan, Q. Ye, J. Zhang, F. Huang, and S. Zhang. Hallucination augmented contrastive learning for multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27036–27046, 2024. 1
- [38] K. Kafle, B. Price, S. Cohen, and C. Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656, 2018. 14
- [39] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016. 9, 12, 14
- [40] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015– 4026, 2023. 3
- [41] J. Y. Koh, D. Fried, and R. R. Salakhutdinov. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36, 2024. 11
- [42] H. Lauren¸con, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2023. 11
- [43] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125,

2023. 9, 12

- [44] B. Li, P. Zhang, J. Yang, Y. Zhang, F. Pu, and Z. Liu. Otterhd: A high-resolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023. 11
- [45] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders

- and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023. 3, 11
- [46] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.R. Wen. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355, 2023. 12
- [47] Y. Li, Y. Zhang, C. Wang, Z. Zhong, Y. Chen, R. Chu, S. Liu, and J. Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024. 2, 3
- [48] Z. Li, F. Liu, W. Yang, S. Peng, and J. Zhou. A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems, 33(12):6999–7019, 2021. 2
- [49] Z. Li, B. Yang, Q. Liu, Z. Ma, S. Zhang, J. Yang, Y. Sun, Y. Liu, and X. Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26763–26773, 2024. 2, 3, 11, 12
- [50] G. Lin, A. Milan, C. Shen, and I. Reid. Refinenet: Multipath refinement networks for high-resolution semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1925–1934,

2017. 1

- [51] Z. Lin, C. Liu, R. Zhang, P. Gao, L. Qiu, H. Xiao, H. Qiu, C. Lin, W. Shao, K. Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575,

2023. 2, 3

- [52] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3, 10, 11, 14
- [53] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3
- [54] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 1, 3, 9, 11, 12
- [55] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 9
- [56] Z. Liu, G. Gao, L. Sun, and Z. Fang. Hrdnet: Highresolution detection network for small objects. In 2021 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2021. 1
- [57] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022. 2, 3
- [58] Z. Liu, K. Zhou, W. X. Zhao, D. Gao, Y. Li, and J.-R. Wen. Less is more: Data value estimation for visual instruction tuning. arXiv preprint arXiv:2403.09559, 2024. 11
- [59] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 9

- [60] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, Y. Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024. 3
- [61] J. Lu, C. Clark, S. Lee, Z. Zhang, S. Khosla, R. Marten, D. Hoiem, and A. Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024. 11
- [62] P. Lu, L. Ji, W. Zhang, N. Duan, M. Zhou, and J. Wang. R-vqa: learning visual relation facts with semantic attention for visual question answering. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1880–1889, 2018. 2
- [63] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022. 9
- [64] G. Luo, Y. Zhou, Y. Zhang, X. Zheng, X. Sun, and R. Ji. Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 2, 3, 11
- [65] Y. Luo, J. Ji, X. Sun, L. Cao, Y. Wu, F. Huang, C.-W. Lin, and R. Ji. Dual-level collaborative transformer for image captioning. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 2286–2293, 2021. 1
- [66] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi. Okvqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 9, 12
- [67] D. Massiceti, N. Siddharth, P. K. Dokania, and P. H. Torr. Flipdial: A generative model for two-way visual dialogue. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6097–6105, 2018. 1
- [68] M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 12
- [69] M. Mathew, D. Karatzas, and C. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 12
- [70] L. Meng, J. Yang, R. Tian, X. Dai, Z. Wu, J. Gao, and Y.-G. Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. arXiv preprint arXiv:2406.04334, 2024. 11
- [71] MetaAI. Introducing meta llama 3: The most capable openly available llm to date. 2024. 3
- [72] S. Moon, A. Madotto, Z. Lin, T. Nagarajan, M. Smith, S. Jain, C.-F. Yeh, P. Murugesan, P. Heidari, Y. Liu, et al. Anymal: An efficient and scalable any-modality augmented language model. arXiv preprint arXiv:2309.16058, 2023. 11

- [73] OpenAI. Introducing chatgpt. 2022. 3
- [74] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. ElNouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3
- [75] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022. 3
- [76] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [77] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei,

I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 2

- [78] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 3
- [79] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294,

2022. 2

- [80] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114,

2021. 2, 3

- [81] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [82] P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556– 2565, 2018. 14
- [83] O. Sidorov, R. Hu, M. Rohrbach, and A. Singh. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer, 2020. 12
- [84] L. Song, Y. Chen, S. Yang, X. Ding, Y. Ge, Y.-C. Chen, and Y. Shan. Low-rank approximation for sparse attention in multi-modal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13763–13773, 2024. 11
- [85] Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024. 1

- [86] Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Zhang, Y. Wang, H. Gao, J. Liu, T. Huang, and X. Wang. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, 2023. 11
- [87] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto. Stanford alpaca: An instruction-following llama model. https: //github.com/tatsu-lab/stanford_alpaca,

2023. 3

- [88] M. N. Team. Introducing mpt-7b: A new standard for opensource, commercially usable llms, 2023. Accessed: 202305-05. 3
- [89] R. Tito, D. Karatzas, and E. Valveny. Document collection visual question answering. In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part II 16, pages 778–792. Springer, 2021. 14
- [90] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 3
- [91] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [92] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [93] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint

- arXiv:2311.03079, 2023. 3

[94] H. Wei, L. Kong, J. Chen, L. Zhao, Z. Ge, J. Yang, J. Sun, C. Han, and X. Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint

- arXiv:2312.06109, 2023. 3

- [95] H. Xu and K. Saenko. Ask, attend and answer: Exploring question-guided spatial attention for visual question answering. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, the Netherlands, October 11–14, 2016, Proceedings, Part VII 14, pages 451–466. Springer,

2016. 1

- [96] R. Xu, Y. Yao, Z. Guo, J. Cui, Z. Ni, C. Ge, T.-S. Chua, Z. Liu, M. Sun, and G. Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024. 2, 3
- [97] S. Xuan, Q. Guo, M. Yang, and S. Zhang. Pink: Unveiling the power of referential comprehension for multi-modal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13838–13848,

2024. 11

- [98] T. Yao, Y. Pan, Y. Li, and T. Mei. Exploring visual relationship for image captioning. In Proceedings of the European conference on computer vision (ECCV), pages 684– 699, 2018. 2

- [99] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, G. Xu, C. Li, J. Tian, Q. Qian, J. Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126,

- 2023. 3

[100] Q. Ye, H. Xu, J. Ye, M. Yan, A. Hu, H. Liu, Q. Qian, J. Zhang, and F. Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13040–13051,

- 2024. 11

- [101] A. Young, B. Chen, C. Li, C. Huang, G. Zhang, G. Zhang, H. Li, J. Zhu, J. Chen, J. Chang, et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024. 3
- [102] L. Yu, B. Shi, R. Pasunuru, B. Muller, O. Golovneva, T. Wang, A. Babu, B. Tang, B. Karrer, S. Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3), 2023. 11
- [103] Q. Yu, J. Li, L. Wei, L. Pang, W. Ye, B. Qin, S. Tang, Q. Tian, and Y. Zhuang. Hallucidoctor: Mitigating hallucinatory toxicity in visual instruction data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12944–12953, 2024. 1
- [104] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 12
- [105] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 9, 12
- [106] Y. Zeng, P. Zhang, J. Zhang, Z. Lin, and H. Lu. Towards high-resolution salient object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7234–7243, 2019. 1
- [107] J. Zhan, J. Dai, J. Ye, Y. Zhou, D. Zhang, Z. Liu, X. Zhang, R. Yuan, G. Zhang, L. Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226, 2024. 11
- [108] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 3
- [109] Y. Zhang, S. Qian, B. Peng, S. Liu, and J. Jia. Prompt highlighter: Interactive control for multi-modal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13215–13224, 2024. 11
- [110] H. Zhao, X. Qi, X. Shen, J. Shi, and J. Jia. Icnet for realtime semantic segmentation on high-resolution images. In Proceedings of the European conference on computer vision (ECCV), pages 405–420, 2018. 1
- [111] L. Zhao, D. Cai, L. Sheng, and D. Xu. 3dvg-transformer: Relation modeling for visual grounding on point clouds. In

- Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2928–2937, 2021. 2
- [112] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223,

2023. 1

- [113] X. Zhao, X. Li, H. Duan, H. Huang, Y. Li, K. Chen, and H. Yang. Mg-llava: Towards multi-granularity visual instruction tuning. arXiv preprint arXiv:2406.17770, 2024. 3
- [114] Z. Zhao, L. Guo, T. Yue, S. Chen, S. Shao, X. Zhu, Z. Yuan, and J. Liu. Chatbridge: Bridging modalities with large language model as a language catalyst. arXiv preprint arXiv:2305.16103, 2023. 11
- [115] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3, 11
- [116] Y. Zhu, M. Zhu, N. Liu, Z. Ou, X. Mou, and J. Tang. Llavaphi: Efficient multi-modal assistant with small language model. arXiv preprint arXiv:2401.02330, 2024. 11
- [117] D. Zong, C. Ding, B. Li, J. Li, K. Zheng, and Q. Zhou. Acformer: An aligned and compact transformer for multimodal sentiment analysis. In Proceedings of the 31st ACM International Conference on Multimedia, pages 833–842,

2023. 11

