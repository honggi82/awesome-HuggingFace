# arXiv:2504.09621v1[cs.CV]13Apr2025

## Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images

Jiuchen Chen Xinyu Yan Qizhi Xu∗ Kaiqi Li Beijing Institute of Technology

{castlechen, yanxinyu, qizhi, kaiqilee}@bit.edu.cn

### Abstract

Global contextual information and local detail features are essential for haze removal tasks. Deep learning models perform well on small, low-resolution images, but they encounter difficulties with large, high-resolution ones due to GPU memory limitations. As a compromise, they often resort to image slicing or downsampling. The former diminishes global information, while the latter discards highfrequency details. To address these challenges, we propose DehazeXL, a haze removal method that effectively balances global context and local feature extraction, enabling end-to-end modeling of large images on mainstream GPU hardware. Additionally, to evaluate the efficiency of global context utilization in haze removal performance, we design a visual attribution method tailored to the characteristics of haze removal tasks. Finally, recognizing the lack of benchmark datasets for haze removal in large images, we have developed an ultra-high-resolution haze removal dataset (8KDehaze) to support model training and testing. It includes 10000 pairs of clear and hazy remote sensing images, each sized at 8192 × 8192 pixels. Extensive experiments demonstrate that DehazeXL can infer images up to 10240 × 10240 pixels with only 21 GB of memory, achieving state-of-the-art results among all evaluated methods. The source code and experimental dataset are available at https://github.com/CastleChen339/ DehazeXL.

### 1. Introduction

Image dehazing is a critical operation in various applications, including surveillance [20, 40], autonomous navigation [22, 34], and remote sensing [46]. Haze significantly degrades image quality by obscuring details and distorting color representation, which impairs the performance of subsequent visual tasks such as object detection [33, 36] and tracking [36]. In order to address this issue, researchers

*corresponding author

###### (a) Downsampling

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Downsample

###### (b) Slicing

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Partition

Merge

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

… … …

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

(c) DehazeXL (ours)

Global Context Fusion

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

GlobalAttentionModule

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Partition

Merge

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

… … …

… … …

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Figure 1. Comparison between different methods for handling large images in haze removal tasks. (a) Downsampling approach, which reduces the image size but loses critical high-frequency details. (b) Image slicing technique, which processes larger inputs but compromises global contextual information and object coherence. (c) The proposed method, which aims to effectively balance global context and local feature extraction to enhance haze removal performance in high-resolution images.

have developed a multitude of approaches that leverage Convolutional Neural Networks (CNNs) [6, 28, 54, 57], Generative Adversarial Networks (GANs) [41, 52, 53], Transformers [14, 32, 37], and Diffusion models [5, 45] to tackle the haze removal problem. These methods have demonstrated exceptional performance in various fields, successfully restoring clarity and improving visual fidelity.

With advancements in image sensor technology, both the resolution and scale of captured images are steadily increasing. However, most existing dehazing methods have been developed and tested on relatively small images, typically

ranging from 256 × 256 to 512 × 512 pixels. Constrained by GPU memory, these methods often employ compromises when processing large inputs, resorting to strategies such as slicing and downsampling [15, 21, 35, 58]. Although image slicing allows the processing of large inputs, it disrupts global contextual information, potentially leading to a loss in object coherence and spatial relationships. On the other hand, downsampling preserves global structure but sacrifices critical high-frequency details that are vital for downstream tasks such as object detection. These limitations highlight the need for innovative solutions that can efficiently balance global context and local details in the haze removal domain, particularly for high-resolution imagery.

In this paper, we propose DehazeXL, an end-to-end haze removal method that effectively integrates global information interaction with local details extraction. As shown in Figure 2, DehazeXL is capable of directly inferring large images without incurring quadratic increases in GPU memory usage. Specifically, the input image is partitioned into equal-sized patches, each encoded into a feature vector by a shared encoder. These feature vectors serve as tokens for the global attention module, facilitating integration of broader contextual information. The globally enhanced features are then passed through a decoder, progressively upsampled to the original patch size, and finally merged to generate the output image.

The key features of DehazeXL are characterized by three aspects: 1) Decoupled Input Dimensions. By partitioning images into fixed-size patches, DehazeXL decouples the encoder-decoder input dimensions from the image size. This approach enables efficient batch processing of image patches while conserving GPU memory, mitigating the risk of memory overflow. Moreover, maintaining a consistent patch size standardizes inputs for both the encoder and decoder, which enhances training stability and convergence. 2) Enhanced Local Feature Representation. A customized global attention module enriches each local feature vector with essential global context, which includes haze distribution, color consistency in clear regions, and brightness levels. This information is vital for accurate scene reconstruction. Without the global information, local feature vectors may lack spatial coherence, potentially leading to artifacts or inconsistencies in the output. 3) Efficient Global Attention Mechanisms. Drawing inspiration from long-context attention mechanisms in large language models, we incorporate locality-sensitive hashing and lowrank decomposition into our global attention module. This design reduces the memory usage and computational demands when processing long contexts, thereby improving the model’s ability to capture extensive contextual dependencies across ultra-high-resolution images.

Compared to existing methods, the most significant advancement of DehazeXL lies in its efficient global model-

###### A100 (80 GB)

GPUMemoryUsage(GB)

RTX A6000 (48 GB)

DehazeXL (ours)

RTX 4090 (24 GB)

RTX

A4000 (16 GB)

RTX 2080 (8 GB)

20.04 GB

Image Size (pixels)

Figure 2. Comparison of GPU memory usage across various models. DehazeXL demonstrates a reduction in memory usage by approximately 65%-80% when processing large images compared to other methods. Notably, when employing FP16 format for inference, DehazeXL can process 10,240 × 10,240 pixel images with only 21 GB of memory.

ing capability for large inputs. To investigate the impact of global information utilization efficiency on dehazing performance, we develop a visual attribution method specifically tailored for haze removal tasks. By analyzing the contribution of each region, we can gain insights into which features are most influential in haze removal, thereby enhancing our understanding of the underlying processes involved. This approach not only facilitates the optimization of model performance but also provides a framework for interpreting results, which is crucial for advancing research in the field.

Additionally, we unexpectedly discovered a notable scarcity of ultra-high-resolution datasets designed for haze removal through extensive literature review. The existing datasets, such as 4KID [59], are limited to a maximum resolution of 3840 × 2160 pixels. To fill this gap, we construct a haze removal dataset (8KDehaze) using aerial images. Un-

like existing haze removal datasets, all images in 8KDehaze have a resolution of 8192 × 8192 pixels, providing a unique resource for training and evaluating dehazing algorithms on ultra-high-resolution data.

In summary, our key contributions are as follows:

- • We propose DehazeXL, an end-to-end haze removal method that seamlessly integrates global information interaction with local feature extraction. This approach allows for efficient processing of large images without significant increases in GPU memory usage.
- • To evaluate the efficiency of global context utilization in haze removal performance, we design a visual attribution method called Dehazing Attribution Map (DAM). This method enables the identification and quantification of how specific regions or features contribute to model performance, supporting optimization and interpretability.
- • We constuct an ultra-high-resolution haze removal dataset (8KDehaze), which comprises images with a resolution of 8192 × 8192 pixels sourced from aerial imagery. This dataset addresses the scarcity of high-resolution resources in haze removal research and includes a diverse range of haze distributions and terrains, facilitating rigorous evaluation and future advancement of dehazing algorithms.

### 2. Related Work

Single Image Dehazing. Single image dehazing has progressed significantly over the past few decades. Traditional methods predominantly relied on atmospheric scattering models, utilizing handcrafted priors such as the Dark Channel Prior [18, 24] and Color Attenuation Prior [61, 62]. However, these methods often struggled in complex scenes due to oversimplified assumptions about scene structure and atmospheric conditions. The advent of large-scale hazy image datasets has catalyzed the rapid development of datadriven methods. Researchers have increasingly turned to deep learning models [8, 31, 58] to overcome the limitations of traditional techniques. Recent methods often incorporate attention mechanisms [25, 42], Multi-scale feature fusion mechanisms [26, 50], and physically grounded models [19, 60] to improve dehazing performance. The integration of deep learning not only enhances feature extraction capabilities but also facilitates the modeling of complex atmospheric phenomena. This transition to data-driven methodologies marks a great advancement in the field, enabling more accurate dehazing results. However, most deep learning-based dehazing methods struggle to infer highresolution images due to GPU memory constraints, limiting their practical use in real-world applications.

Large Image Inference. With advancements in imaging sensor technologies, high-resolution image modeling and inference have emerged as key challenges in computer vision. Techniques for addressing large images typically fall into two categories: multi-scale hierarchical (or cascading)

methods and sliding window strategies. R-CNN [12] and CNN cascades [10] demonstrated the effectiveness of cascading networks for large images, though at the cost of speed. Recently, Gupta et al. [16] designed a visual backbone network for high-level vision tasks involving large images. They sliced the input images to extract local features and then employed a self-attention mechanism to derive global information from these local features. This approach achieved impressive performance in image classification, object detection, and segmentation tasks. In the domain of haze removal, Zheng et al. [59] proposed a model capable of processing 4K images on a single GPU by combining three CNNs for feature extraction, guidance map learning, and feature fusion. Conversely, sliding window methods are widely used in various visual tasks [47], where large images are divided into smaller patches to enable localized processing. However, both approaches have inherent limitations. Multi-scale hierarchical methods suffer from memory usage that scales quadratically with input size, posing serious computational challenges. Sliding window techniques can disrupt spatial coherence in tasks like dehazing, leading to block artifacts at the edges of the windows. Balancing computational efficiency with contextual integrity remains an open research challenge.

Visual Interpretation of Networks As deep neural networks become increasingly prevalent in computer vision, there has been growing interest in understanding the factors that influence their outputs. This process, known as attribution analysis, aims to provide insight into which features contribute most significantly to the network’s decisions. Over recent years, numerous attribution methods [1, 11, 30, 56] have been developed to produce interpretable and intuitive visual explanations. Some works [49, 55] focus on analyzing the internal parameters of the network, tracing how information flows through layers and nodes to attribute predictions. However, this becomes challenging for highly complex models due to the intricate nature of their architectures. To address this, other methods [3, 9] treat the network as a black box, perturbing key features of the input to assess their impact on the output. This perturbation-based approach evaluates the sensitivity of the model to specific input regions or features, offering a more flexible means of interpretation without requiring detailed knowledge of the model’s inner workings. In addition, there are also works on improving model interpretability, such as Local Attribution Map [13] and LIME [29, 48], which provides localized explanations by approximating the complex model’s predictions with simpler, interpretable models in the vicinity of specific instances.

### 3. Methodology

The key contribution of our work is to design an end-toend haze removal model for large images. The architecture

Encoder

##### Encoder

Input Image Patches ……

- Token 1
- Token 2
- Token 3
- Token 4

Token n

Token n-1

…………

- Token 5
- Token 6

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

……

| | | | |
|---|---|---|---|
| | | | |
| | |[Figure 56]<br><br>[Figure 57]| |
| | | | |

[Figure 58]

[Figure 59]

[Figure 60]

Batching

Batching

Concatenate+Rearrange

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

PositionEmbedding

LinearEmbedding

ImagePartition

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

PatchMerging

PatchMerging

PatchMerging

PatchMerging

PatchMerging

PatchMerging

PatchMerging

PatchMerging

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

…………

…………

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

……

……

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Mini-Batch 1 Mini-Batch 2 Mini-Batch n

Mini-Batch 1 Mini-Batch 2 Mini-Batch n

…… …… …… …… ……

……

…… …… …… ……

Encoder Batch Aggregation

Encoder Batch Aggregation

[Figure 100]

[Figure 101]

[Figure 102]

Token n

Token n-

[Figure 103]

[Figure 104]

[Figure 105]

Token n

Token n

Input

###### Asynchronous Encoding/Decoding

###### Bottleneck

###### Bottleneck

×2

×2

×2

×2

×6

×6

×2

×2

Bottleneck

###### Bottleneck

###### Skip Connection

Global Attention Module

[Figure 106]

[Figure 107]

[Figure 108]

RMSNorm Hyper Attention

RMSNorm

LLMTransformerBlock

LLMTransformerBlock

Output

Batching

Batching

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

+

+

Decoder

Decoder

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

SwinTransformerBlock

TransposedConvolution

Rearrange+Divide

ImageStitching

PatchExpanding

PatchExpanding

PatchExpanding

PatchExpanding

PatchExpanding

PatchExpanding

PatchExpanding

PatchExpanding

[Figure 114]

[Figure 115]

[Figure 116]

RMSNorm Forward Feed

RMSNorm

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

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

……

……

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

+

+

……

……

…… …… …… ……

…… …… …… ……

Mini-Batch 1 Mini-Batch 2 Mini-Batch n

Mini-Batch 1 Mini-Batch 2 Mini-Batch n

[Figure 153]

[Figure 154]

[Figure 155]

Batch Aggregation

Batch Aggregation

LLM Transformer Block

LLM Transformer Block

……

……

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

LLM Transformer Block

LLM Transformer Block

……

……

×2

×2

×2

×2

×6

×6

×2

×2

Output Image Patches

Decoder

Decoder

Figure 3. Overall architecture of the proposed model. It begins by partitioning the hazy image into uniform-sized patches, which are then encoded into tokens by the Encoder. The Bottleneck injects global information into each token, enhancing the contextual representation. Subsequently, the Decoder reconstructs the tokens back into image patches, forming the final output image. Notably, to minimize memory consumption, both the Encoder and Decoder employ an asynchronous processing strategy, handling the input in multiple mini-batches sequentially rather than simultaneously. This design optimizes memory efficiency while ensuring effective haze removal.

and details of the proposed DehazeXL are presented in Section 3.1. In addition, we develop a visual attribution method for dehazing tasks called DAM. Section 3.2 presents the principles of this method.

- 3.1. Architecture of DehazeXL As shown in Figure 3, the framework of DehazeXL con-

sists of three primary components: the Encoder, Bottleneck, and Decoder. Initially, the hazy input image is divided into several fixed-size patches. These patches are then input into the Encoder for tokenization. The Bottleneck is designed to inject global information into each token, thereby enhancing their contextual representation. Finally, the Decoder reconstructs the processed tokens into patches, resulting in the final dehazed image.

Encoder. The Encoder can be any visual model backbone capable of extracting local features from each image patch. In our experiments, we employed the Swin Transformer V2 [27] as the Encoder. This choice leverages the Swin Transformer’s ability to capture hierarchical features and its efficient handling of long-range dependencies, which is particularly advantageous for processing complex hazy images. Since the Encoder focuses solely on local features, we adopt a strategy of dividing the patches into multiple minibatches for sequential input to the Encoder, rather than processing all patches simultaneously. While this design may slow down the encoding speed, it effectively decouples the memory usage of the Encoder from the size of the input

image, significantly reducing memory consumption and enabling the processing of large-scale images.

Bottleneck. Within the Encoder, all patches are encoded into smaller feature maps, referred to as tokens. These tokens are then input into the Bottleneck. We constructed an efficient Transformer block to extract global information and inject it into all tokens. We utilized RMSNorm [51] as the normalization layer to save computational time. Additionally, inspired by large language models [43], we implemented Hyper Attention [17], which was confirmed to be effective in natural language processing. This approach aims to enhance inference speed while minimizing memory usage, particularly for long-context inputs. Consequently, all tokens can ”see” each other, facilitating the learning of global information such as haze distributions, color characteristics, and brightness.

Decoder. The Decoder’s function is to reconstruct the tokens into clear, haze-free patches. Similar to the Encoder, we utilized the Swin Transformer V2 [27] as the backbone, substituting the Patch Merging layer with a Patch Expanding layer that employs transposed convolution to iteratively upscale and merge feature maps. Through skip connections, we concatenate the outputs from each layer of the Encoder with the corresponding feature maps in the Decoder, thus enhancing the flow of information and gradients. Consistent with our approach in the Encoder, we adopt a ”divide and conquer” strategy in the Decoder, sequentially process-

ing all tokens instead of concurrently. This strategy allows us to achieve significantly lower memory usage at the cost of slightly increased processing time.

#### 3.2. Dehazing Attribution Map

Inspired by the Integrated Gradients (IG) method [39] and the Local Attribution Map [13], we propose the Dehazing Attribution Map to enhance the interpretability of our model. Let F : Rh×w → Rh×w represent a dehazing network. To quantify the dehazing effect, we utilize a pixel intensity detector, given the significant differences in pixel intensities between hazy and clear images. Specifically, for an input hazy image I ∈ Rh×w, we define the detector as Dxy(I) = i∈[x,x+l],j∈[y,y+l]Iij, where the subscripts i and j denote the spatial coordinates. For clarity, we will omit the subscripts in the subsequent discussion. To conduct attribution analysis for dehazing network, we require a baseline input image I′ which satisfies that F(I′) absent certain features present in F(I). The attribution map D(F(I)) is obtained by computing the pathintegrated gradient along a continuous trajectory transitioning from I′ to I. This smooth path function is denoted as γ(α) : [0,1] → Rh×w, with γ(0) : I′ and γ(1) : I. Therefore, the i-th dimension of the attribution map can be expressed as follows:

1

∂γ(α)i ∂α

∂D(F(γ(α))) ∂γ(α)i ×

dα (1)

DAMF,D(γ)i =

0

As highlighted in [38], the effectiveness of model attribution depends on the choice of an appropriate baseline. For instance, in image classification tasks, a pure black image serves as a suitable baseline since the model is unable to classify it [39]. In this work, we meticulously design baseline inputs specifically tailored for the dehazing network. As stated above, a baseline input must lack certain key features, which are typically determined by the characteristics of the task. In the context of dehazing, clear regions of an image are easy to reconstruct. In contrast, reconstructing hazy regions, particularly those with thick haze, poses substantial challenges. Effectively reconstructing these hazy areas is crucial for achieving superior dehazing results. Therefore, as shown in Figure 4, we utilize the clear image as the baseline input and adopt a linear interpolation function as the path function. In practice, we compute the gradients at uniformly sampled points along the defined path, then approximating the integral as described in Eq (2):

m

∂D(F(γ(mk ))) ∂γ(mk )i ·

(∆γk,m)i m

DAM˜ F,D(γ)i =

(2)

k=1

where m denotes the number of steps used for the integral approximation, and ∆γk,m = γ(mk )−γ(km+1) .Empirically, we find that a step count of 100 is sufficient to approximate the integral effectively.

Hazy image Integration Path Baseline

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

- (a) Interpolated images  ()
- (b) Gradients at interpolation

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Figure 4. Illustration of the baseline image and the path function. The region enclosed by the red box indicates the attribution area.

### 4. Experiments and Analysis

#### 4.1. Dataset

To train and evaluate the proposed network and the comparative methods, we constructed an ultra-high-resolution haze removal dataset (8KDehaze), containing 10,000 images at a resolution of 8192 × 8192 pixels. The clear images in 8KDehaze were sourced from publicly available aerial imagery provided by the United States Geological Survey, while the hazy counterparts were generated using the atmospheric scattering model [18] and the approach proposed by Czerkawski et al.[7]. To the best of our knowledge, 8KDehaze is the first ultra-high-resolution dataset in the field of image dehazing. In addition, to further validate the effectiveness of the proposed method, we conduct extensive training and testing on the synthesis dataset 4KID [58] and the real-world dataset O-HAZE [2]. Table 1 shows the details of datasets used in the experiments.

Table 1. Overview of datasets used in the experiments.

|Dataset| |4KID O-HAZY 8KDehaze|
|---|---|---|
|Quantity| |15606 45 10000<br><br>|
|Image Size| |3840 × 2160<br><br>1286 × 947 to 5436 × 3612<br><br>8192 × 8192<br><br>|
|Source| |Video Frames Commercial Camera Aerial Images|
|Content| |Urban Streets Parks, Suburban<br><br>Urban, Farmland, Mountains, Desert, Coastlines, Rivers|

#### 4.2. Implementation Details

The proposed model was implemented in PyTorch and trained on a single NVIDIA A100 GPU. Input images were randomly cropped to a resolution of 2048 × 2048 pixels, with a batch size of 2 during training. For comparative analysis, we selected a range of recently published state-of-the-art dehazing algorithms, including 4KDehazing [58], Dehamer [14], C2PNet [57], Dehazeformer [37], MB-TaylorFormer [32], ConvIR [6], Mixdehazenet [28], and DEA-Net [4]. Since these approaches could not be

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

| |
|---|

(a) Input (b) 4KDehazing (Slicing) (c) 4KDehazing (Direct) (d) Dehamer (e) C2PNet (f) DehazeFormer-b

-b

7.638 / 0.5301 19.22 / 0.8903 14.72 / 0.8120 21.73 / 0.9008 18.82 / 0.9028 23.30 / 0.9286

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

(g) MB-TaylorFormer (h) ConvIR-b (i) MixDehazeNet-b (j) DEA-Net (k) DehazeXL(ours) (l) GT

- -b -b -

17.61 / 0.8944 19.08 / 0.8931 20.04 / 0.8961 21.47 / 0.8844 33.80 / 0.9914 PSNR / SSIM

- Figure 5. Dehazed results on the 8KDehaze dataset. The patches for comparison are marked with red boxes in the original images. PSNR / SSIM is calculated based on the patches to better reflect the performance difference. The proposed DehazeXL can directly infer images with a resolution of 8192 × 8192 without the need for slicing inference. Compared to other methods, the proposed method effectively eliminates segmentation artifacts and achieves superior visual quality.

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

[Figure 217]

[Figure 218]

(a) Input (b) 4KDehazing (Slicing) (c) 4KDehazing (Direct) (d) Dehamer (e) C2PNet (f) DehazeFormer-b

(g) MB-TaylorFormer (h) ConvIR-b (i) MixDehazeNet-b (j) DEA-Net (k) DehazeXL(ours) (l) GT

9.484 / 0.7500 23.26 / 0.9038 21.65 / 0.8021 16.32 / 0.8703 15.71 / 0.8171 22.26 / 0.9095

13.04 / 0.7988 20.00 / 0.9127 24.65 / 0.9349 12.35 / 0.8343 28.94 / 0.9341 PSNR / SSIM

- Figure 6. Dehazed results on the 4KID [58] dataset. The proposed DehazeXL can effectively utilize global information to guide image restoration in different regions, enhancing the global consistency of the output results.

trained directly on images at the 2048 × 2048 resolution, input image pairs were randomly cropped into patches of size 512 × 512 pixels. The training batch size for these methods was maximized based on available GPU memory. All models were trained using the Adam optimizer [23] with an initial learning rate of 0.001. To facilitate effective training, a cosine annealing schedule was employed to gradually decay the learning rate throughout the training process. Each model was trained for a total of 500 epochs, utilizing the L1 loss function as the objective.

In the testing phase, most comparative methods, including Dehamer [14], C2PNet [57], Dehazeformer [37], MBTaylorFormer [32], ConvIR [6], MixdehazeNet [28], and DEA-Net [4], employed a slicing inference strategy due to their limitations in processing large images. Notably, 4KDehazing [58] is the only comparative method that supports direct inference on large images. Thus, the results for

4KDehazing were obtained using both slicing and direct inference. The proposed DehazeXL directly inferred the input images without employing the slicing strategy.

#### 4.3. Evaluation and Results

Qualitative Evaluation. Figure 5 to 7 present the testing results of the proposed method and comparative algorithms applied to samples from the 8KDehaze, 4KID and O-HAZE datasets. As illustrated in Figure 5, methods that employ the slicing inference strategy exhibit noticeable block artifacts. While 4KDehazing can perform direct inference without the need for slicing, its dehazing performance significantly deteriorates when handling large images. In contrast, the proposed DehazeXL demonstrates superior dehazing capabilities. In Figure 6, all comparative methods exhibit varying degrees of failure, particularly in the sky regions. This is primarily due to the similarity in features be-

|[Figure 219]<br><br>[Figure 220]|
|---|

|[Figure 221]<br><br>[Figure 222]|
|---|

|[Figure 223]<br><br>[Figure 224]|
|---|

|[Figure 225]<br><br>[Figure 226]|
|---|

|[Figure 227]<br><br>[Figure 228]|
|---|

|[Figure 229]<br><br>[Figure 230]|
|---|

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

(a) Input (b) 4KDehazing (Slicing) (c) 4KDehazing (Direct) (d) Dehamer (e) C2PNet (f) DehazeFormer-b

-b

13.02 / 0.5657 17.81 / 0.6570 16.30 / 0.5339 17.57 / 0.7062 18.50 / 0.7141 19.66 / 0.7092

|[Figure 243]<br><br>[Figure 244]|
|---|

|[Figure 245]<br><br>[Figure 246]|
|---|

|[Figure 247]<br><br>[Figure 248]|
|---|

|[Figure 249]<br><br>[Figure 250]|
|---|

|[Figure 251]<br><br>[Figure 252]|
|---|

|[Figure 253]<br><br>[Figure 254]|
|---|

[Figure 255]

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

(g) MB-TaylorFormer (h) ConvIR-b (i) MixDehazeNet-b (j) DEA-Net (k) DehazeXL(ours) (l) GT

- -b -b -

17.95 / 0.6717 18.07 / 0.7060 16.28 / 0.6830 16.66 / 0.6460 20.28 / 0.7325 PSNR / SSIM

- Figure 7. Dehazed results on the O-HAZE [2] dataset. The proposed DehazeXL demonstrates higher color fidelity and restores more details compared with other state-of-the-art methods.

- Table 2. Quantitative evaluations on the 8KDehaze dataset, the 4KID dataset [58], and the O-HAZE dataset [2] in terms of PSNR, SSIM, and average infer time.

|Method<br><br>|Venue&Year<br><br>|8KDehaze| | |4KID [58]| | |O-HAZY [2]| | |
|---|---|---|---|---|---|---|---|---|---|---|
| | |PSNR|SSIM<br><br>|Time(s)<br><br>|PSNR|SSIM|Time(s)<br><br>|PSNR|SSIM<br><br>|Time(s)|
|4KDehazing (Slicing) 4KDehazing (Direct) Dehamer C2PNet DehazeFormer-s DehazeFormer-b MB-TaylorFormer ConvIR-s ConvIR-b MixDehazeNet-s MixDehazeNet-b DEA-Net<br><br>|CVPR2021<br><br>CVPR2021<br>CVPR2022<br>CVPR2023 TIP2023 TIP2023<br><br><br>ICCV2023 TPAMI2024 TPAMI2024 IJCNN2024 IJCNN2024 TIP2024|25.81 20.41 25.92 26.17 26.68 26.83 26.41 25.11 26.93 20.99 23.16 25.89<br><br>|0.9569 0.8664 0.9373 0.9669 0.9729 0.9657 0.9668 0.9599 0.9775 0.8934 0.9284 0.9329|6.682 1.350 6.614 43.269 7.469 15.013 120.540 6.661 8.709 6.563 13.154 7.402<br><br>|19.97 18.68 21.24 18.14 20.83 21.25 18.63 20.66 21.92 21.25 23.22 20.83|0.8624 0.7424 0.8795 0.8299 0.8763 0.8843 0.8497 0.8696 0.888 0.8817 0.9063 0.8834<br><br>|1.04 0.19 1.03 6.76 1.17 2.35 18.83 1.04 1.36 1.03 2.06 1.16|18.73 19.3 19.59 20.29 19.86 20.22 19.57 18.83 19.61 19.09 20.67 20.01<br><br>|0.6726 0.6426 0.7134 0.7113 0.7116 0.7173 0.7104 0.7095 0.7199 0.7165 0.7293 0.6988|1.31 0.27 1.30 8.51 1.47 2.95 23.71 1.31 1.71 1.29 2.59 1.46<br><br>|
|DehazeXL| |32.35|0.9863<br><br>|4.617|26.62|0.9073<br><br>|0.59|21.49<br><br>|0.7348|0.86|

tween the sky and dense haze, which makes it challenging for slice-based inference methods to distinguish between sky regions and those obscured by haze. In contrast, DehazeXL effectively utilizes global information to differentiate the sky from hazy regions, thereby enhancing the global consistency of the output results. Figure 7 further highlights the advantages of DehazeXL in terms of color restoration and overall coherence, demonstrating excellent generalization capability of the proposed method in real hazy scenes. Quantitative Evaluation. Table 2 summarizes the quantitative evaluation results of DehazeXL and the comparative methods on the 8KDehaze, 4KID and O-HAZE datasets, using metrics such as PSNR, SSIM [44], and average inference time. The proposed method achieves the highest

scores for both PSNR and SSIM, indicating its superior dehazing effectiveness. Although 4KDehazing is faster with direct inference, it exhibits weaker performance on larger images and suffers from ghosting and color shifts. In contrast, DehazeXL achieves an excellent balance between dehazing performance and processing time, demonstrating its efficacy in practical applications.

#### 4.4. Ablation Study

We conducted ablation studies to evaluate the impact of different Backbone types and the depth of the Global Attention Module in the Bottleneck of the proposed DehazeXL. These experiments were performed on the 8KDehaze dataset, with the results presented in Table 1. Our find-

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

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

###### AttributionMapsDehazedResults

8.278/0.6453

(1-(1-a)Input

(1-

| |
|---|

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

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

[Figure 308]

[Figure 309]

[Figure 310]

PSNR/SSIM

(1-b)GT

(1- (1- (1- (1- (1- -b (1- - (1- -b (1- -b (1- - (1-

(1-h) MB-TaylorFormer 21.43 / 0.9543

(1-i) ConvIR-b 19.45 / 0.9428

(1-j) MixDehazeNet-b 21.42 / 0.9506

(1-k) DEA-Net 20.29 / 0.8672

(1-l) DehazeXL(ours) 30.39 / 0.9885

(1-d) 4KDehazing (Direct) 14.63 / 0.8680

(1-e) Dehamer 21.63 / 0.8874

(1-c) 4KDehazing (Slicing) 17.20 / 0.8816

(1-f) C2PNet

(1-g) DehazeFormer-b

20.79 / 0.9464

20.40 / 0.9398

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

[Figure 332]

###### AttributionMapsDehazedResults

8.617/0.6821

(2-(2-a)Input

(2-

| |
|---|

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

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

PSNR/SSIM

(2-b)GT

(2- (2- (2- (2- (2- -b (2- - (2- -b (2- -b (2- - (2-

(2-h) MB-TaylorFormer

(2-i) ConvIR-b

(2-j) MixDehazeNet-b

(2-k) DEA-Net

(2-l) DehazeXL(ours)

(2-c) 4KDehazing (Slicing) 17.04 / 0.8655

(2-d) 4KDehazing (Direct) 13.86 / 0.8430

(2-e) Dehamer 22.88 / 0.9438

(2-f) C2PNet

(2-g) DehazeFormer-b

26.46 / 0.9603

21.42 / 0.9470

19.07 / 0.8882

21.20 / 0.9110

31.77 / 0.9813

26.05 / 0.9586

21.15 / 0.9221

- Figure 8. Comparison of the dehazed results and attribution maps of different methods. The red box on (1-a) and (2-a) indicate the regions of interest for attribution. The attribution maps highlight how each pixel influences the dehazing results in the specified region.

ings indicate that larger Backbone sizes and deeper Bottlenecks do indeed lead to improved performance; however, they also result in a significant increase in inference time. Considering the trade-off between inference time and model performance, we selected Swin-T and a depth of 2 as the default choices for the Backbone and Bottleneck, respectively.

- Table 3. Ablation study results for Backbone types and Bottleneck depth in DehazeXL on the 8KDehaze dataset.

|Backbone Type|Bottleneck Depth<br><br>|Metrics| | |
|---|---|---|---|---|
| | |PSNR|SSIM<br><br>|Time(s)|
|Swin-T|1<br><br>2<br><br><br>4|31.61 32.35 32.40<br><br>|0.9719 0.9863 0.9857|4.511 4.617 4.810<br><br>|
|Swin-S<br><br>|1<br>2 4<br>|31.89 32.58 32.76<br><br>|0.9759 0.9871 0.9870<br><br>|5.880 5.967 6.102|
|Swin-B|1<br><br>2<br><br><br>4<br><br>|32.10 32.77 33.06<br><br>|0.9792 0.9879 0.9894|9.066 9.183 9.526<br><br>|
|Swin-L|1<br><br>2<br><br><br>4<br><br>|32.93 32.98 33.30|0.9877 0.9885 0.9911<br><br>|17.37 17.64 18.25|

- 4.5. Attribution Analysis

Figure 8 presents the results of the attribution analysis conducted using the proposed DAM. As illustrated in Figure 8, methods employing the slicing inference strategy are limited to local information in the vicinity of the attribution regions during the image reconstruction process. This restriction can lead to color distortions and artifacts, particularly in areas with complex textures or uneven brightness, thereby adversely affecting global consistency of dehazed results. In contrast, both 4KDehazing and the proposed De-

hazeXL can directly infer high-resolution images without the need for slicing strategies, allowing them to leverage global information to aid in the reconstruction of local areas, thus achieving better global consistency. Furthermore, compared to 4KDehazing, DehazeXL demonstrates a more efficient utilization of local features and global context, resulting in higher quality detail recovery and improved dehazing performance.

Additionally, the attribution maps shown in Figure 8 (1-l) and (2-l) indicate that the model tends to focus on haze-free regions and high-contrast textures during the reconstruction process. This phenomenon suggests that the model prioritizes the use of unambiguous visual cues to enhance the quality of the dehazed output. Compared to methods employing slicing inference strategies, the proposed approach more effectively utilizes the spectral and color information from haze-free regions, thereby underscoring the importance of contextual information in efficient image dehazing.

### 5. Conclusion

In this paper, we propose DehazeXL, an end-to-end haze removal method that effectively integrates global information with local feature, enabling efficient processing of large images while minimizing GPU memory usage. To facilitate a visual interpretation of the factors influencing dehazed results, we design the Dehazing Attribution Map for haze removal tasks. Quantitative and qualitative evaluations demonstrate that the proposed DehazeXL outperforms state-of-the-art haze removal techniques in terms of both accuracy and inference speed across multiple high-resolution datasets. The results of the attribution analysis underscore the critical role of global information in image dehazing tasks. Moreover, our work provides a valuable dataset (8KDehaze) and analytical tool for future research in the field of haze removal.

### References

- [1] Reduan Achtibat, Maximilian Dreyer, Ilona Eisenbraun, Sebastian Bosse, Thomas Wiegand, Wojciech Samek, and Sebastian Lapuschkin. From attribution maps to human-understandable explanations through concept relevance propagation. Nature Machine Intelligence, 5(9):1006– 1019, 2023. 3
- [2] Codruta O Ancuti, Cosmin Ancuti, Radu Timofte, and Christophe De Vleeschouwer. O-haze: a dehazing benchmark with real hazy and haze-free outdoor images. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 754–762, 2018. 5, 7
- [3] Dharanidharan Arumugam and Ravi Kiran. Interpreting denoising autoencoders with complex perturbation approach. Pattern Recognition, 136:109212, 2023. 3
- [4] Zixuan Chen, Zewei He, and Zhe-Ming Lu. Dea-net: Single image dehazing based on detail-enhanced convolution and content-guided attention. IEEE Transactions on Image Processing, 2024. 5, 6
- [5] Longyu Cheng, Xujin Ba, and Yanyun Qu. Dehazediff: When conditional guidance meets diffusion models for image dehazing. In 2024 IEEE International Symposium on Circuits and Systems (ISCAS), pages 1–5. IEEE, 2024. 1
- [6] Yuning Cui, Wenqi Ren, Xiaochun Cao, and Alois Knoll. Revitalizing convolutional network for image restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 5, 6
- [7] Mikolaj Czerkawski, Robert Atkinson, Craig Michie, and Christos Tachtatzis. Satellitecloudgenerator: Controllable cloud and shadow synthesis for multi-spectral optical satellite images. Remote Sensing, 15(17), 2023. 5
- [8] Hang Dong, Jinshan Pan, Lei Xiang, Zhe Hu, Xinyi Zhang, Fei Wang, and Ming-Hsuan Yang. Multi-scale boosted dehazing network with dense feature fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2157–2167, 2020. 3
- [9] Thomas Fel, M´elanie Ducoffe, David Vigouroux, R´emi Cad`ene, Mikael Capelle, Claire Nicod`eme, and Thomas Serre. Don’t lie to me! robust and efficient explainability with verified perturbation analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16153–16163, 2023. 3
- [10] Michael Gadermayr, Ann-Kathrin Dombrowski, Barbara Mara Klinkhammer, Peter Boor, and Dorit Merhof. Cnn cascades for segmenting sparse objects in gigapixel whole slide images. Computerized Medical Imaging and Graphics, 71:40–48, 2019. 3
- [11] Arne Gevaert, Axel-Jan Rousseau, Thijs Becker, Dirk Valkenborg, Tijl De Bie, and Yvan Saeys. Evaluating feature attribution methods in the image domain. Machine Learning, pages 1–46, 2024. 3
- [12] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE con-

- ference on computer vision and pattern recognition, pages 580–587, 2014. 3
- [13] Jinjin Gu and Chao Dong. Interpreting super-resolution networks with local attribution maps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9199–9208, 2021. 3, 5
- [14] Chun-Le Guo, Qixin Yan, Saeed Anwar, Runmin Cong, Wenqi Ren, and Chongyi Li. Image dehazing transformer with transmission-aware 3d position embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5812–5820, 2022. 1, 5, 6
- [15] Ritwik Gupta, Shufan Li, Tyler Zhu, Jitendra Malik, Trevor Darrell, and Karttikeya Mangalam. xt: Nested tokenization for larger context in large images. arXiv preprint

- arXiv:2403.01915, 2024. 2

[16] Ritwik Gupta, Shufan Li, Tyler Zhu, Jitendra Malik, Trevor Darrell, and Karttikeya Mangalam. xt: Nested tokenization for larger context in large images. arXiv preprint

- arXiv:2403.01915, 2024. 3

- [17] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P Woodruff, and Amir Zandieh. Hyperattention: Long-context attention in near-linear time. arXiv preprint arXiv:2310.05869, 2023. 4
- [18] Kaiming He, Jian Sun, and Xiaoou Tang. Single image haze removal using dark channel prior. IEEE transactions on pattern analysis and machine intelligence, 33(12):2341–2353,

2010. 3, 5

- [19] Yufeng He, Cuili Li, and Xu Li. Remote sensing image dehazing using heterogeneous atmospheric light prior. IEEE Access, 11:18805–18820, 2023. 3
- [20] Jehoiada Jackson, Kwame Obour Agyekum, Chiagoziem Ukwuoma, Rutherford Patamia, Zhiguang Qin, et al. Hazy to hazy free: A comprehensive survey of multi-image, singleimage, and cnn-based algorithms for dehazing. Computer Science Review, 54:100669, 2024. 1
- [21] Chen Jin, Ryutaro Tanno, Thomy Mertzanidou, Eleftheria Panagiotaki, and Daniel C Alexander. Learning to downsample for segmentation of ultra-high resolution images. arXiv preprint arXiv:2109.11071, 2021. 2
- [22] Wong Yoke Kim, Yan Chai Hum, Yee Kai Tee, Wun-She Yap, Haman Mokayed, and Khin Wee Lai. A modified single image dehazing method for autonomous driving vision system. Multimedia Tools and Applications, 83(9):25867– 25899, 2024. 1
- [23] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 6
- [24] Sungmin Lee, Seokmin Yun, Ju-Hun Nam, Chee Sun Won, and Seung-Won Jung. A review on dark channel prior based image dehazing algorithms. EURASIP Journal on Image and Video Processing, 2016:1–23, 2016. 3
- [25] Xiaoling Li, Zhen Hua, and Jinjiang Li. Attention-based adaptive feature selection for multi-stage image dehazing. The Visual Computer, 39(2):663–678, 2023. 3
- [26] Yong Liu and Xiaorong Hou. Local multi-scale feature aggregation network for real-time image dehazing. Pattern Recognition, 141:109599, 2023. 3

- [27] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12009–12019, 2022. 4
- [28] LiPing Lu, Qian Xiong, Bingrong Xu, and Duanfeng Chu. Mixdehazenet: Mix structure block for image dehazing network. In 2024 International Joint Conference on Neural Networks (IJCNN), pages 1–10. IEEE, 2024. 1, 5, 6
- [29] Mohammad Nagahisarchoghaei, Mirhossein Mousavi Karimi, Shahram Rahimi, Logan Cummins, and Ghodsieh Ghanbari. Generative local interpretable model-agnostic explanations. In The International FLAIRS Conference Proceedings, 2023. 3
- [30] Ian E Nielsen, Dimah Dera, Ghulam Rasool, Ravi P Ramachandran, and Nidhal Carla Bouaynaya. Robust explainability: A tutorial on gradient-based attribution methods for deep neural networks. IEEE Signal Processing Magazine, 39

(4):73–84, 2022. 3

- [31] Xu Qin, Zhilin Wang, Yuanchao Bai, Xiaodong Xie, and Huizhu Jia. Ffa-net: Feature fusion attention network for single image dehazing. In Proceedings of the AAAI conference on artificial intelligence, pages 11908–11915, 2020. 3
- [32] Yuwei Qiu, Kaihao Zhang, Chenxi Wang, Wenhan Luo, Hongdong Li, and Zhi Jin. Mb-taylorformer: Multi-branch efficient transformer expanded by taylor formula for image dehazing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12802–12813, 2023. 1, 5, 6
- [33] A Radha Rani, Y Anusha, SK Cherishama, and S Vijaya Laxmi. Traffic sign detection and recognition using deep learning-based approach with haze removal for autonomous vehicle navigation. e-Prime-Advances in Electrical Engineering, Electronics and Energy, 7:100442, 2024. 1
- [34] Vani Suthamathi Saravanarajan, Rung-Ching Chen, ChengHsiung Hsieh, and Long-Sheng Chen. Improving semantic segmentation under hazy weather for autonomous vehicles using explainable artificial intelligence and adaptive dehazing approach. IEEE Access, 11:38194–38207, 2023. 1
- [35] Lianlei Shan, Minglong Li, Xiaobin Li, Yang Bai, Ke Lv, Bin Luo, Si-Bao Chen, and Weiqiang Wang. Uhrsnet: A semantic segmentation network specifically for ultra-highresolution images. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 1460–1466. IEEE, 2021. 2
- [36] Monika Sharma, Dileep Kumar Yadav, and SB Goyal. A review on haze removal methods in image and video for object detection and tracking. Computational Intelligence in Robotics and Automation, pages 129–139, 2023. 1
- [37] Yuda Song, Zhuqing He, Hui Qian, and Xin Du. Vision transformers for single image dehazing. IEEE Transactions on Image Processing, 32:1927–1941, 2023. 1, 5, 6
- [38] Pascal Sturmfels, Scott Lundberg, and Su-In Lee. Visualizing the impact of feature attribution baselines. Distill, 5(1): e22, 2020. 5
- [39] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319–3328. PMLR, 2017. 5

- [40] Abishek Suresh, R Bharathi, and Vaidehi Vijayakumar. Enhanced deep dehazing for haze removal in license plates. In 2024 3rd International Conference on Artificial Intelligence For Internet of Things (AIIoT), pages 1–6. IEEE, 2024. 1
- [41] Tewodros Tassew and Nie Xuan. Dc-gan with feature attention for single image dehazing. Signal, Image and Video Processing, 18(3):2167–2182, 2024. 1
- [42] Lihan Tong, Yun Liu, Weijia Li, Liyuan Chen, and Erkang Chen. Haze-aware attention network for single-image dehazing. Applied Sciences, 14(13):5391, 2024. 3
- [43] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 4
- [44] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 7
- [45] Siying Xie, Fuping Li, and Mingye Ju. Frequency-based and physics-guiding diffusion model for single image dehazing. In 2024 43rd Chinese Control Conference (CCC), pages 7262–7267. IEEE, 2024. 1
- [46] Qizhi Xu, Jiuchen Chen, Xinyu Yan, and Wei Li. Mrfnet: An infrared remote sensing image thin cloud removal method with the intra-inter coherent constraint. IEEE Transactions on Geoscience and Remote Sensing, 2024. 1
- [47] Xinyu Yan, Jiuchen Chen, Qizhi Xu, and Wei Li. Mitigating texture bias: A remote sensing super-resolution method focusing on high-frequency texture reconstruction. IEEE Transactions on Geoscience and Remote Sensing, 2025. 3
- [48] Mao Yang, Chuanyu Xu, Yuying Bai, Miaomiao Ma, and Xin Su. Investigating black-box model for wind power forecasting using local interpretable model-agnostic explanations algorithm: Why should a model be trusted? CSEE Journal of Power and Energy Systems, 2023. 3
- [49] Ruo Yang, Binghui Wang, and Mustafa Bilgic. Idgi: A framework to eliminate explanation noise from integrated gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23725– 23734, 2023. 3
- [50] Yan Yang, Haowen Zhang, Xudong Wu, and Xiaozhen Liang. Mstfdn: Multi-scale transformer fusion dehazing network. Applied Intelligence, 53(5):5951–5962, 2023. 3
- [51] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 4
- [52] Shengdong Zhang, Xiaoqin Zhang, Linlin Shen, and En Fan. Gan-based dehazing network with knowledge transferring. Multimedia Tools and Applications, 83(15):45095–45110,

2024. 1

- [53] Xianhong Zhang. Research on remote sensing image de-haze based on gan. Journal of Signal Processing Systems, 94(3): 305–313, 2022. 1
- [54] Yafei Zhang, Shen Zhou, and Huafeng Li. Depth information assisted collaborative mutual promotion network for single

- image dehazing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2846–2855, 2024. 1
- [55] Chenyang Zhao, Kun Wang, Xingyu Zeng, Rui Zhao, and Antoni B Chan. Gradient-based visual explanation for transformer-based clip. In International Conference on Machine Learning, pages 61072–61091. PMLR, 2024. 3
- [56] Xiangwei Zheng, Lifeng Zhang, Chunyan Xu, Xuanchi Chen, and Zhen Cui. An attribution graph-based interpretable method for cnns. Neural Networks, 179:106597,

2024. 3

- [57] Yu Zheng, Jiahui Zhan, Shengfeng He, Junyu Dong, and Yong Du. Curricular contrastive regularization for physics-aware single image dehazing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5785–5794, 2023. 1, 5, 6
- [58] Zhuoran Zheng, Wenqi Ren, Xiaochun Cao, Xiaobin Hu, Tao Wang, Fenglong Song, and Xiuyi Jia. Ultra-highdefinition image dehazing via multi-guided bilateral learning. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16180–16189. IEEE,

2021. 2, 3, 5, 6, 7

- [59] Zhuoran Zheng, Wenqi Ren, Xiaochun Cao, Xiaobin Hu, Tao Wang, Fenglong Song, and Xiuyi Jia. Ultra-highdefinition image dehazing via multi-guided bilateral learning. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16180–16189. IEEE,

2021. 2, 3

- [60] Hao Zhou, Zekai Chen, Yun Liu, Yongpan Sheng, Wenqi Ren, and Hailing Xiong. Physical-priors-guided dehazeformer. Knowledge-Based Systems, 266:110410, 2023. 3
- [61] Qingsong Zhu, Jiaming Mai, and Ling Shao. Single image dehazing using color attenuation prior. In BMVC, pages 1674–1682. Citeseer, 2014. 3
- [62] Qingsong Zhu, Jiaming Mai, and Ling Shao. A fast single image haze removal algorithm using color attenuation prior. IEEE transactions on image processing, 24(11):3522–3533,

2015. 3

