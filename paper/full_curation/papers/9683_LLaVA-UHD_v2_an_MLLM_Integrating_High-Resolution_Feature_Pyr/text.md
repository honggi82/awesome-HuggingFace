# arXiv:2412.13871v2[cs.CV]19Mar2025

## LLaVA-UHD v2: an MLLM Integrating High-Resolution Semantic Pyramid via Hierarchical Window Transformer

Yipeng Zhang1∗ Yifan Liu1∗ Zonghao Guo1† Yidan Zhang4 Xuesong Yang4

Xiaoying Zhang5 Chi Chen1 Jun Song3 Bo Zheng3 Yuan Yao2† Zhiyuan Liu1 Tat-Seng Chua2 Maosong Sun1

1Tsinghua University 2National University of Singapore 3Alibaba Group 4University of Chinese Academy of Sciences 5The Chinese University of Hong Kong yipengzhang97@gmail.com guozonghao96@outlook.com

https://github.com/thunlp/LLaVA-UHD

### Abstract

Vision transformers (ViTs) are widely employed in multimodal large language models (MLLMs) for visual encoding. However, they exhibit inferior performance on tasks regarding fine-grained visual perception. We attribute this to the limitations of ViTs in capturing diverse multi-modal visual levels, such as low-level details. To address this issue, we present LLaVA-UHD v2, an MLLM with advanced perception abilities by introducing a well-designed vision-language projector, the Hierarchical window (Hiwin) transformer. Hiwin transformer enhances MLLM’s ability to capture diverse multi-modal visual granularities, by incorporating our constructed high-resolution semantic pyramid. Specifically, Hiwin transformer comprises two key modules: (i) a visual detail injection module, which progressively injects low-level visual details into high-level language-aligned semantics features, thereby forming an inverse semantic pyramid (ISP), and (ii) a hierarchical window attention module, which leverages cross-scale windows to condense multi-level semantics from the ISP. Extensive experiments show that LLaVA-UHD v2 outperforms compared MLLMs on a wide range of benchmarks. Notably, our design achieves an average boost of 3.7% across 14 benchmarks compared with the baseline method, 9.3% on DocVQA for instance. All the data and code will be publicly available to facilitate future research.

### 1 Introduction

Embedding visual information into large language models (LLMs) [103, 23, 4, 2] has significantly enhanced their ability to handle multimodal tasks, such as visual question answering [9, 42, 36], document analysis [85, 84, 70], and visual interaction [21, 30].

Among these advancements, the CLIP [91] series, built upon the vision transformer (ViT) architecture [27, 91], has emerged as a standard for visual encoding in contemporary multi-modal large language models (MLLMs) [26, 11, 104, 67, 66, 56, 4, 8, 110]. By leveraging contrastive training, CLIP-ViT extracts text-aligned visual features that facilitate seamless integration with LLMs, enabling effective handling of tasks that emphasize text-aligned semantics, e.g., image captioning [90, 61].

∗Equal contribution. †Corresponding authors.

Input Image

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

ViT MLP/Re-sampler LLM

[Figure 6]

- (a) Recent MLLMs

(c) Ours: Hierarchical Window Transformer

- (b) Mixture of Visual Experts for MLLMs

[Figure 7]

[Figure 8]

[Figure 9]

Text Tokens

Input Image

[Figure 10]

[Figure 11]

ViT

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

(CLIP)

[Figure 16]

Customized Projector

###### LLM

[Figure 17]

…

…

[Figure 18]

[Figure 19]

ViT

[Figure 20]

[Figure 21]

(DINO)

Text Tokens

Input Image

Hiwin Transformer

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Vision Detail Injection

Semantic

Integration LLM

ViT

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Inverse Semantic Pyramid

Text Tokens

- Figure 1: Comparison of LLaVA-UHD v2 with other MLLMs. (a) MLLMs typically align ViT features to language space using MLPs [67] or perceiver re-samplers [6, 56], lacking visual granularity. (b) Combining multiple visual encoders is non-universal and computationally intensive. (c) LLaVAUHD v2 employs the Hiwin transformer to build an inverse semantic pyramid and compress it into visual tokens, providing various semantic granularity for language generation.

However, CLIP-ViT-based MLLMs often underperform in tasks requiring extensive low-level visual details, such as visual grounding [114, 18, 116, 113] and optical character recognition [50, 53], hindering the MLLMs’ practical applications. As illustrated in Fig. 1(a), such inferior performance primarily stems from the low resolution and text-aligned nature of CLIP features. To facilitate MLLMs’ fine-grained perception, Yao et al. [109] proposes a Dense Connector [109], which fuse lower layers of CLIP-ViT. Yet, its performance is limited by the low resolution of the features.

To address such limitation, we attempts to incorporate a vision feature pyramid [63, 93, 122] into MLLMs to provide multi-level visual granularity. However, directly employing such pyramid might encounter the following two key challenges: (1) Representation Non-inheritance. The inherent image-text alignment of CLIP-ViT’s feature representations is fundamental to the efficacy of MLLMs. Substituting CLIP-ViT with hierarchical architectures such as Swin [72] to obtain multi-scale representations would disrupt this alignment and consequently compromise the transferability of its pretrained representations. (2) Compression Ineffectiveness. The quadratic computational cost of LLMs w.r.t. the number of visual tokens necessitates effective compression of the feature pyramid. Current projecting methods [97, 64, 77, 102] designed a customized projector to resize multi-scale visual features before token compression, which resulted in a loss in visual details and spatial relations in naive feature resolution, as shown in Fig. 1(b).

To address these issues, we present LLaVA-UHD v2, an advanced MLLM with enhanced perceptual capabilities leveraging the incorporated Hiwin transformer. As shown in Fig. 1(c), Hiwin transformer enables capturing diverse multi-modal granularity by integrating a constructed high-resolution semantic pyramid.

Specifically, the Hiwin transformer consists of two key modules as follows: (i) a visual detail injection module (VDIM) for inheriting the representations. We propose a VDIM to inject low-level details (e.g., edges, textures) from images into text-aligned features from a CLIP-pretrained ViT [91], progressively building an up-sampled level upon the previous level, resulting in an inverse semantic pyramid (ISP). During the training stage, a reconstruction loss between fused and original CLIP features explicitly maintains vision-language alignment while enhancing visual granularity. This strategy can be extended to any ViT for inheriting its powerful multi-modal representations. (ii) a hierarchical window attention module for ensuring effective compression. We propose utilizing a set of hierarchical windows to capture semantics from local regions across different semantic levels. A set of learnable queries is restricted to attending solely to sampled features within their respective windows. This attention mechanism performs effective compression on local dense features at the

[Figure 30]

[Figure 31]

[Figure 32]

|[Figure 33]|
|---|

|[Figure 34]|
|---|

[Figure 35]

HiwinTransformer

|[Figure 36]|Adaptive|
|---|---|
| |Slicing|

[Figure 37]

[Figure 38]

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

Flatten

…

|[Figure 55]|
|---|

|[Figure 56]|
|---|

[Figure 57]

…

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

###### LLM

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

ViT

Image Slices ( )

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

|[Figure 78]|
|---|

Flatten

[Figure 79]

…

[Figure 80]

Ratio-preserving Resizing

Overview

- Figure 2: The overall architecture of proposed LLaVA-UHD v2, consisting of a ViT, our hierarchical window transformer (Hiwin transformer), and an LLM. The Hiwin transformer first injects highfrequency visual details from the image into the high-level semantics of ViT features, forming inverse semantic pyramids (ISP). Then it compresses the ISPs into spatially consistent tokens via cross-scale windows, for a better vision-language alignment. Details about the two procedures are illustrated in
- Figure 3 and 4.

native resolution of each pyramid level, thereby enabling visual tokens to effectively capture both fine-grained visual details and high-level language-aligned semantics.

Extensive experiments demonstrate that LLaVA-UHD v2 dramatically outperforms the compared MLLMs across a wide range of benchmarks. More importantly, LLaVA-UHD v2 surpasses the baseline method (LLaVA-UHD [35]) on 14 popular benchmarks by 3.7% in average, including document-centric visual question answering (e.g., +9.3% on DocVQA), visual grounding (e.g., average +5.7% on RefCOCOs [114]), and high-resolution image perception (e.g., +3.4% on HRBench [105]). Besides, we experimentally reveal that ISP, regardless of construction method (e.g., bilinear interpolation), enhances the visual perception capabilities of MLLMs, offering new insights for future research. In summary, our contributions are three-fold:

- • We present LLaVA-UHD v2, an MLLM with the advanced visual perception ability, through the integration of a high-resolution semantic pyramid representation.
- • We propose the Hiwin transformer, a novel vision-language projector comprises a visual detail injection module and a hierarchical window attention module for capturing diverse multimodal visual granularities.
- • Trained on merely academic-scale data, LLaVA-UHD v2 achieves substantial improvements over the baseline method across 14 benchmarks.

### 2 Related Works

##### 2.1 Feature Pyramid Representation

Image pyramid techniques are fundamental in image processing, facilitating multi-resolution analysis since the era of manual feature design, as seen in SIFT’s scale-space keypoints [14, 75]. In deep learning, CNNs such as ResNet and VGG, inherently extracts hierarchical features across scales [39, 99]. Innovations such as FPN and U-Net enhance semantic hierarchy for tasks like detection and segmentation [63, 93]. Recently, some transformer-based models [72, 117, 122] further advanced feature pyramid construction, capturing more comprehensive visual semantic granularity for visual representations. However, multimodal language models, often using CLIP-based ViTs, underutilize hierarchical features, suggesting a research gap for integrating advanced feature pyramids into these models [117, 91].

##### 2.2 Visual Encoding in MLLMs

CLIP-ViT, favored for its effective alignment of visual features with linguistic semantics through contrastive pre-training, is widely adopted in MLLMs [91, 68, 67, 11, 66, 8, 121, 59]. Emerging research explores alternative visual representations, primarily in three categories: (1) Fusing features from CLIP-based CNNs and ViTs. LLaVA-HR [80] integrates stage-wise CNN features [73] into ViT’s layers, enhancing the fine-grained perception of ViT representations. CogAgent [41] utilizes ViT features to query high-resolution CNN features for detailed information during language decoding.

Mini-Gemini [58] employs a cross-attention-based post-fusion between CNN and ViT features. (2) Fusing features from visual experts trained with different vision pre-training tasks [64, 31, 77, 123, 102, 97, 108]. Candidate experts include DINO-v2 [89] by visual contrastive pre-training, SAM [51] by prompt segmentation pre-training, Pix2Struct [53] by document parsing pre-training, etc. Deepseek-VL [77], SPHINX-X [64, 31] and Eagle [97] down-sample output feature maps and concatenate them along the channel axis. Cambrian-1 [102] initializes embeddings to query local patches from visual experts. (3) Language models for visual encoding: Fuyu [12], Otter-HD [55], and SOLO [22] encode images directly with LLMs, bypassing dedicated visual encoders. However, these approaches, while effective, increase computational demand and hinder a unified image-to-language design.

##### 2.3 Token Projection and Compression

MLPs [68, 67], perceiver resamplers [6, 11, 110] and Q-Formers [56, 25, 119] are basic projectors widely used in modern MLLMs. Recently, various new designs have emerged. (1) Spatial-preserving compression. Qwen2-VL [104] and MiniGPT-v2 [17] employ a simple linear to merge tokens locally (e.g. 2×2). Honey-bee [16] introduces a C-abstractor (i.e. CNN-based block), while Oryx [74] dynamically pools features and utilizes them to query the original ones. (2) Cross-layer feature compression. Token-Packer [57] compresses features across layers in a ViT using cross-attention. MMFuser [15] uses final-layer features as queries to attend to early-layer features. (3) Semanticmerged compression. Chat-UniVi [44] and LLaVA-PruMerge [95] extending the token merging [13] strategy, merge features with similar semantics into a single representation. However, these approaches depend heavily on feature padding, resizing, and reshaping, hindering their ability to compress features with arbitrary resolutions.

### 3 Method

##### 3.1 Overview

The architecture of the proposed LLaVA-UHD v2 is illustrated in Fig. 2. LLaVA-UHD v2 comprises three modules: a visual encoder (ViT), a vision-language projector (Hiwin transformer), and an LLM. We first partition and process the input image with the adaptive slicing strategy from [35], outputting CLIP [91] features with arbitrary size and shape. The resulting features are subsequently passed to the Hiwin transformer for vision-language projection, which is carried out with two stages: (i) constructing an inverse semantic pyramid (ISP) and (ii) integrating the ISP by hierarchical window attention, which will be detailed in Sec.3.2 and Sec.3.3. The core of the Hiwin transformer lies in enhancing each ViT-encoded feature into a high-resolution semantic pyramid encoding, thereby achieving enriched semantic granularity for each slice or image. After the enhancement, visual tokens from different slices are reorganized into a spatially consistent feature map relative to the original image, ensuring clarity in spatial relationships. Followed by concatenating with the overview tokens, the visual tokens provide both high-level language-aligned semantics and high-resolution visual details for language decoding.

##### 3.2 Inverse Semantic Pyramid

Preliminaries. Traditional convolutional neural networks (CNNs) naturally produce a pyramid of hierarchical bottom-up features {Fl ∈ R

p·2l×pW·2l×C}. Note that l is the level index, (H,W) is the image resolution, C is the feature dimension, and p is a down-sampling ratio (e.g., p = 8 in ResNet50 [39]). In these pyramids, higher-resolution (lower-level) feature maps are rich in visual details and lower-resolution (higher-level) ones contain abstract semantic information. However, ViTs, splitting the image into coarse patches, only produce single-scale feature maps (i.e., F0 ∈ R

H

H p ×Wp ×C, where e.g.,p = 14). Lacking such feature pyramids like CNNs hinders their performance on MLLM tasks requiring both fine-grained and high-level visual information. Therefore, how to construct a ViT-based feature pyramid with varying semantic granularity remains a problem.

Visual detail injection module (VDIM). Due to the lack of high-resolution features, up-sampling ViT features becomes the necessary strategy to inversely construct the feature pyramid. Two simple approaches, (1) plain bilinear interpolation and (2) deconvolution network, can be adopted. By doubling

[Figure 81]

###### Visual Detail Injection

Low-level visual detail

 1 2

[Figure 82]

Kernel

Kernel Weights

[Figure 83]

Weights High-level

[Figure 84]

language-aligned semantics

[Figure 85]

ViT

2 Up Conv Up2 Conv

(CLIP)

[Figure 86]

Down-Sampling Multi-level Conv

Down-Sampling Conv

Reconstruction Loss

Only in training phase

- Figure 3: The flowchart illustrates the construction of the Inverse Semantic Pyramid (ISP). As the first level of ISP, F0 is the high-level language-aligned semantic features from CLIP-ViT. Subsequent levels, F1 and F2, are progressively built by injecting high-frequency visual details from the input image into upsampled features from the previous level, via the Visual Detail Injection Module (VDIM). A Multi-level Reconstruction (MLR) loss supervises in each scale, ensuring both textaligned semantic coherence and fine-grained visual fidelity.

p ×Wp·2l×C,l = 0,1,2} is then constructed. While effective, directly up-sampling language-aligned features from CLIP-ViT hardly introduces precise visual details, resulting in suboptimal performance, illustrated in Table. 3. To address this, we design a VDIM to up-sample multi-modal semantic features guided by original image priors.

H·2l

and quadrupling the last-layer feature maps, a ViT-based feature pyramid {Fl ∈ R

Specifically, the objective of VDIM is to learn (l − 1) convolution layers on the image pyramid {Il ∈ R

p ×Wp·2l ×3} to capture high-frequency visual patterns of image texture for guiding the up-sampling process of semantic features, as shown in Fig. 3. For each input image, its (l + 1)-th level features is defined as

H·2l

Fl+1 = Conv Up(Fl);Θl+1(Il+1) , (1)

where Up(·) denotes the up-sampling interpolation and Conv(·) the convolutional operation on feature maps with customized kernel weights Θl+1 learned on image Il+1.

Optimizing VDIM. We propose a multi-level reconstruction (MLR) loss between the higher-level feature maps {F1,F2} and the lowest one F0 as

2

- 1

- 2

∥F0 − Down(Fl;Ωl)∥22, (2)

L =

l=1

where Down(·) is a down-sampling operation with trainable weights Ωl in each level. The proposed MLR loss drives the feature pathway to capture low-level textures as well as to maintain multi-modal semantics during the fusion procedure.

Construction of inverse semantic pyramid (ISP). As shown in Fig. 3, VDIM acts as a progressive feature resolution expanding procedure conditioned on original image priors. During the inference, the resulting multi-level feature maps {F0,F1,F2} form an ISP, which gathers a hierarchical multi-modal semantic representation with corresponding spatial resolutions supporting proper visual granularity.

##### 3.3 Hierarchical Window Attention

The hierarchical nature of ISP necessitates an effective approach for compressing features at varying resolutions while maintaining cross-level spatial alignment.

Hierarchical window generation. Inspired by object detection [92, 122], we utilize the RoIalign [40] to sample key features to keep the spatial locality of cross-level feature maps, in Fig. 4. Specifically, we first uniformly divide feature maps of each level into N × N windows, whose

widths and heights are float-point values (WN , HN ) rather than integers. Windows share the same

Learnable Local Queries

[Figure 87]

ROI-Align

[Figure 88]

[Figure 89]

[Figure 90]

| | | |
|---|---|---|
| |[Figure 91]| |

| | | |
|---|---|---|
|[Figure 92]| | |
| | | |

[Figure 93]

…

Partition

[Figure 94]

| | |
|---|---|
|[Figure 95]| |
| | |

Query

[Figure 96]

- Level 0
- Level 1
- Level 2

Grid Proposals

[Figure 97]

[Figure 98]

[Figure 99]

Attention Block

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Key/Value

Hierarchical Windows Set

[Figure 108]

[Figure 109]

LearnableQueryLocal Cross Attention

[Figure 110]

Level-wise Position Embedding Spatial-wise Position Embedding

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Level 0 Level 1 Level 2

…

… …

[Figure 115]

[Figure 116]

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

(1,1)

(2,3)

(2,3)

(1,1)

(1,1)

(2,3)

- Figure 4: The flowchart of hierarchical window attention. We initialize a set of learnable queries to attend to local regions. Feature maps from the ISP are processed by a set of cross-scale windows, forming hierarchical and local-aware features at different levels. The features are then concatenated along the length axis, to serve as the key and value for the learnable queries. The output is condensed visual tokens rich in diverse and local-aware semantics.

“anchor" point form a set of hierarchical bounding boxes (coordinates of top-left and bottom-right) {Rli,j ∈ R1×4,i,j ∈ 0,1,2...N − 1,l = 0,1,2}, where l is the feature level and (i,j) the 2D index. To mitigate the size distortion of feature maps caused by a difference between the aspect ratio of RoI-aligned feature maps and the image, we define a pooling score to evaluate this difference:

W H − log

rw rh

S(W,H,rw,rh) = − log

, (3)

where (rw,rh) denotes the width and height of pooled features. By maximizing the score S, we select the optimal grid size (rw∗ ,rh∗) from pre-defined proposals {(3,3),(2,3),(3,2),(2,4),(4,2)}. Then, we carry out the RoI-align with the generated windows to sample the key feature maps for the following attention operation.

Cross-scale window querying. To compress the ISP {Fl,l = 0,1,2} of one image or slice I, we initialize a set of queries {Qi,j ∈ R1×C,i,j ∈ 0,1,2...N − 1}, each of which corresponds to a set of hierarchical windows {Rli,j,l = 0,1,2}, in Fig. 4. Regarding each query vector Qi,j, we prepare the key vector Ki,jl ∈ R(r

w·rh∗)×C of l-th level as Ki,jl = RoI(Fl,Rli,j) + ϕl, (4)

∗

where ϕl is a level positional embedding. We then concatenate the Ki,jl in length axis and form the final key vector Ki,j ∈ R(3·r

w·rh∗)×C for each Qi,j. The corresponding value vector Vi,j is obtained in the same way yet without the level positional embedding. Thus, the cross-attention can be performed as

∗

Q∗i,j = CrossAttn(Qi,j + φi,j,Ki,j + ζi,j,Vi,j), (5) where Q∗i,j denotes the updated query, and φ,ζ is the 2D spatial position embedding of query and key vector, respectively. In the end, we concatenate all the Q∗i,j into a feature map P ∈ RN×N×C to represent the visual token of I.

##### 3.4 Spatially-consistent Token Organization

Due to the varying partitions of different images, organizing and conveying the structure of slices to the LLM facilitates a more accurate understanding of the image. Prior studies [35, 64, 110] utilize spatial tokens (e.g., “\n” or “,”) to denote the relative positioning of image slices. While such approaches provide cues regarding the global arrangement of each slice, they overlook the intrinsic spatial relationships among individual visual tokens. For instance, two horizontally adjacent tokens in the image, located in different slices, may become significantly separated in a 1D arrangement.

Leveraging our Hiwin transformer, which preserves the 2D spatial consistency with the original image, we amalgamate these 2D feature maps into a large 2D feature map according to the slicing configuration, in Fig. 2.

### 4 Experiment

In this section, we conduct an empirical evaluation of LLaVA-UHD v2. We begin with a comprehensive outline of the implementation details of our model, followed by a comparative analysis of its performance across widely recognized benchmarks against competitive counterparts. Finally, we provide an in-depth ablation to further elucidate the capabilities and behaviors of LLaVA-UHD v2.

##### 4.1 Implementation Details

Model Setting. We adopt LLaVA-UHD [35] as the baseline method. Specifically, we employ CLIP-ViT-L/14-336 as the visual encoder, Vicuna-7B/13B [23] or Qwen2-7B [104] as the language model, and our proposed Hiwin transformer as the vision-language projector. We set the maximum slice number to 6 to cover a range of aspect ratios and image resolutions. The number of learnable local queries is set as 144 (i.e.,N = 12). Before employing the VDIM in MLLM, we pre-train it with frozen CLIP-ViT on MS-COCO [62] with a global batch of 16 on 8×A100. We leverage Adam optimizer with 1e−3 learning rate for 2000 steps. This is an independent phase for building a task-agnostic representation, and the weights of the ISP are always reused. LLaVA-UHD v2 consists of a two-stage multi-modal training process as outlined below.

- Stage 1: MLLM pre-training. In this stage, the parameters of the visual encoder, pre-trained VDIM, and LLM are frozen. We only fine-tune the parameters within the hierarchical window attention of the HiWin transformer using LLaVA-Pretrain [67] for 1 epoch with a global batch size of 256. We employ the AdamW optimizer and a cosine learning rate scheduler. The learning rate is 1e−3 for Vicuna-7B, 2e−4 for Vicuna-13B, and Qwen2-7B. Note that, in this stage, we only encode the overview image without the slices for efficiency.
- Stage 2: MLLM supervised fine-tuning. In this stage, we fine-tune all parameters except those in the VDIM. The learning rate is set to 2e−5 with a batch size of 128. To manage training costs, we use 825k data for analysis and ablation studies, including LLaVA-mix665k [67] and 160k from Ureader [111]. For comparison with advanced MLLMs, we balance our data distribution and introduce an 858k-mixed dataset, which was detailed in supplementary.

##### 4.2 Experimental Setting

We present the experimental settings, detailing the benchmarks, evaluation metrics, and compared counterparts.

Benchmarks. Extensive benchmarks are used to analyze the effect of our modules. We categorize these benchmarks into the following folds: (1) General VQA benchmarks including MME [28], MMB [69], SEED-Image [54], GQA [42], MMStar [20] and HallusionBench: [34] ; (2) Knowledgebased VQA benchmarks including MMMU-val [115], Science-QA [78], AI2D [48], MathVista [79]; (3) OCR-based VQA benchmarks including ChartQA [84], OCR-Bench [70], TextVQA [100] and DocVQA [85]; (4) Visual spatial understanding benchmarks such as RealWorldQA [1] and RefCOCOs [114]; (5) High-resolution image perception benchmarks like HR-Bench(4K) [105].

Evaluation Protocols. Beyond benchmark evaluations, we report additional metrics for comprehensive analysis: (1) overall volume of training data, (2) maximum supported image resolution for each method, and (3) computation cost of the entire MLLM at maximum resolution.

Counterparts. We compare our model with the advanced MLLM counterparts. (1) General MLLMs like Honey-bee [16], Dense Connector [109], VILA [60] and LLaVA-1.5 [67]. (2) High-resolution MLLMs including Monkey [59], LLaVA-Next [66], PIIP-LLaVA [107], SliME-Llama3-8B [118], DeepseekVL-7B [76] and Token-Packer [57]. (3) Mixture of visual experts such as LLaVA-HR [80], SPHINX-series [64, 31] , MG-LLaVA [120] and Mini-Gemini [8]. (4) OCR-centric MLLMs including UReader [111] and TextMonkey [71].

- Table 1: Main performance on popular benchmarks. #Data denotes the volume of overall data during MLLM pre-training and supervised fine-tuning. “MaxRes." is the maximum accessible resolution of MLLM. “VQAD": DocVQA. “BenchOCR": OCR-Bench. “VQAC": ChartQA. “VQAT": TextVQA. “SQA": Science-QA. “MMMUv": MMMU-val. “Math.": MathVista. “SEEDI": SEED-Image. “MMEP": perception sub-set of MME. “RWQA": RealWorldQA. “BenchHR": HR-Bench.

OCR & Chart Knowledge General

Vision Spatial

High Res. Method LLM #Data MaxRes. #FLOPs. VQAD BenchOCR VQAC VQAT AI2D SQA MMMUv Math. GQA SEEDI MMB MMEP MMStar RWQA BenchHR

mPLUG-Owl2 [112] Llama2-7B 401M 448×448 1.7T - - - 58.2 - 68.7 - 25.5 56.1 57.8 64.5 72.5 34.8 - UReader [111] Llama2-7B 86M 896×1120 20.3T 65.4 - 59.3 57.6 - - - - - - - - - - -

VILA [60] Llama2-7B 51M 336×336 8.2T - - - 64.4 - 68.2 - - 62.3 61.1 68.9 76.7 - - SPHINX-2k [64] Llama2-Accessory 1.01B 762×762 42.2T - - - 61.2 - 70.6 - - 63.1 71.6 65.9 73.6 - - SPHINX-X [31] Llama2-Accessory 15.3M 448×448 21.3T 56.3 - 39.7 58.1 63.0 70.4 - - 56.2 68.8 57.9 63.0 - - LLaVA-HR [80] Vicuna-7B 1.22M 1024×1024 24.3T - - - 67.1 - 65.1 - - 64.2 64.2 - 77.7 - - Honey-bee [16] Vicuna-7B 52.5M 336×336 2.6T - - - - - - 35.3 - - 64.5 70.1 77.2 - - Mini-Gemini [58] Vicuna-7B 3.0M 672×672 54.6T 61.9 47.7 47.4 65.2 68.2 69.6 36.8 - 64.5 66.9 65.8 77.3 - 51.1 50.1

Monkey [59] Vicuna-7B 1.40B 896×1344 28.0T 66.5 51.4 65.1 67.6 62.6 69.4 38.9 33.5 60.7 64.3 59.8 73.6 37 51.6 38.0

LLaVA-1.5 [67] Vicuna-7B 1.22M 336×336 8.0T 21.8 31.8 17.8 45.5 55.5 66.8 37.0 25.5 62.0 65.8 66.5 75.3 33.1 54.8 36.1 LLaVA-Next [66] Vicuna-7B 1.34M 672×672 44.4T 63.6 53.2 54.3 64.9 67.0 70.1 35.8 34.6 64.2 70.2 67.4 76.0 37.6 57.8 47.9 Token-Packer [57] Vicuna-7B 2.7M 1008×1008 13.1T 60.2 45.2 - 68.0 - - 35.4 - - 67.4 74.5 - - -

TextMonkey [71] Vicuna-7B 1.45B 448×448 4.0T 66.7 - 59.9 64.3 - - - - - - - - - - -

LLaVA-1.5 [67] Vicuna-13B 1.22M 336×336 15.1T - - - 61.3 - 71.6 - - 63.3 61.6 67.7 76.5 - - LLaVA-Next [66] Vicuna-13B 1.34M 672×672 67.0T - 53.7 61.4 67.1 - 73.6 36.2 35.3 65.4 71.9 70.0 76.5 40.4 57.6 -

LLaVA-1.5-Qwen [7] Qwen2-7B 1.22M 336×336 8.2T - - - - 64.9 - 40.7 33.6 62.7 69.4 72.0 76.0 - - Dense Connector [109] Llama3-8B 1.22M 384×384 11.6T - - - - - 75.2 40.4 28.6 65.1 - 74.4 - - - LLaVA-LLaMA3 [24] Llama3-8B 1.22M 336×336 8.7T - - - - - 73.3 36.8 - 63.5 - 68.9 - - - -

PIIP-LLaVA [107] Llama3-8B 1.22M 1024×1024 36.0T - - - 67.1 - 68.3 - - 63.9 69.4 67.0 - - - MG-LLaVA [120] Llama3-8B 2.5M 768×768 33.0T - - - 67.3 - 70.8 - - - 69.4 72.1 - - - -

SliME [118] Llama3-8B 2.0M 2016×2016 62.0T - - - 64.8 - - 41.2 - 63.9 - 75.0 - - - DeepseekVL-7B [76] DeepseekLLM-7B - 1024×1024 - - 45.6 - - - - 36.6 36.1 - 70.4 73.2 - 37.1 - -

LLaVA-UHD v2 Vicuna-7B 1.42M 1008×672 17.5T 68.1 53.9 64.5 67.6 70.5 71.3 38.2 34 65.4 70.0 68.2 74.7 40.2 58.2 51.5 LLaVA-UHD v2 Vicuna-13B 1.42M 1008×672 26.4T 68.2 55.6 67.4 70.0 72.4 73.3 37.7 35.2 66.0 71.1 70.3 73.1 42.0 59.6 55.3 LLaVA-UHD v2 Qwen2-7B 1.42M 1008×672 14.3T 72.9 57.7 70.4 70.6 75.5 76.9 43.3 39.1 65.1 73.6 77.1 78.8 49.4 64.6 59.9

- Table 2: Ablation studies of modules in our proposed method. “∆" denotes the overall improvement compared to the baseline. REC reports the average accuracy of RefCOCO/g/+.

OCR & Chart Knowledge General Vision Spatial High Res. Method Average VQAD BenchOCR VQAC VQAT AI2D SQA MMMUv GQA SEEDI MMB MMEP RWQA REC BenchHR

LLaVA-UHD [35] 58.0 56.7 40.9 56.3 62.2 55.4 70.7 37.0 63.8 65.6 64.8 70.0 54.4 68.3 45.6 + VDIM(ISP) 60.0 60.2 50.4 60.4 67.1 57.8 70.5 38.2 64.0 66.7 65.6 71.2 51.9 72.3 43.9 + Hiwin attention 61.5 65.0 51.3 62.5 68.5 58.1 69.2 38.9 64.6 67.4 65.5 73.0 55.5 73.3 48.9 + Token organization 61.7 66.0 50.1 62.8 66.8 59.4 69.8 37.6 64.0 67.4 66.1 73.6 56.9 74.0 49.0

∆ +3.7 +9.3 +9.2 +6.5 +4.6 +4.0 -0.9 +0.6 +0.2 +1.8 +1.3 +3.6 +2.5 +5.7 +3.4

##### 4.3 Main Performance

Table 1 showcases a comparative analysis of our proposed LLaVA-UHD v2 against state-of-the-art MLLMs across 15 widely recognized benchmarks. (1) LLaVA-UHD v2 outperforms current counterparts. Compared with general models (such as LLaVA-1.5, Dense Connector) and highresolution MLLMs (like PIIP-LLaVA, SliME-Llama3-8 and DeepseekVL-7B), LLaVA-UHD v2 demonstrates consistent improvements across various tasks, including general VQA (e.g., 77.1% on MMB and 49.4% on MMStar), ultra-high-resolution image perception (e.g., 59.9% on HR-Bench). Notably, LLaVA-UHD v2 surpasses OCR-centric models like TextMonkey on DocVQA (72.9% vs. 66.5%) and outperforms those with multiple experts (such as MG-LLaVA), achieving superior performance on general tasks like SEED (73.6%). These results underscore the value of rich semantics derived from multi-level multi-modal granularity, enhancing both the understanding and perception abilities of MLLMs. (2) LLaVA-UHD v2 indicates efficiency on data utilization and computation. Compared to LLaVA-Next and Mini-Gemini, both operating at a 672×672 resolution, LLaVA-UHD v2 supports 1.5 times the resolution (i.e., 672×1008) and achieves superior performance with less than 40% of the computational cost. Furthermore, in contrast to Honey-bee and VILA, which utilize 52.5M and 51M data samples respectively, LLaVA-UHD v2 attains comparable or superior performance using only ∼2.8% of the data, demonstrating the data efficiency of our model. As for the training duration, under the same model configuration and data volume, LLaVA-UHD v2 requires ∼27 hours to train on 8×A100 GPUs, while LLaVA-Next needs ∼42 hours, which is well-suited for low-cost exploratory research in the academic community.

##### 4.4 Analytical Study

We conduct analytical experiments on the proposed modules to verify the effect of LLaVA-UHD v2. Without special instructions, we use Vicuna-7B as the base LLM.

- Table 3: Comparison of different methods for semantic pyramid construction. “ConvNext" means we replace the CILP-ViT with CLIP-ConvNext [73] as visual encoder and directly use the feature maps from multiple stages as the final hierarchical feature pyramid.

General Knowledge OCR & Chart High Res. Method Average MMEP GQA AI2D VQAC VQAT VQAD BenchHR

LLaVA-UHD 59.9 70.0 63.8 55.4 56.3 62.2 56.7 45.6 w/ ConvNext 59.7 68.2 62.7 55.6 61.8 63.5 61.8 44.0 w/ DeConv. 61.7 71.2 64.2 57.4 61.8 67.8 63.4 46.3 w/ Bilinear 62.0 72.0 64.5 57.8 62.2 67.6 63.7 46.5 w/ VDIM 63.0 73.0 64.6 58.3 62.5 68.5 65.0 48.9

Main module ablation. In Table 2, by replacing the low-resolution CLIP-ViT features with highestlevel ones of the inverse semantic pyramid (ISP) constructed by VDIM, 2.2% average improvement can be seen, especially, on tasks depending on visual details like OCR-Bench (+9.5%) and RefCOCOs (+4.0%). Employing the Hiwin attention to integrate the ISP further increases 1.3% in average accuracy, especially 5.0% on ultra high-resolution perception (HR-Bench), demonstrating rich visual granularity could facilitate precise language generation. With spatially consistent token organization, visual-spatial understanding observes further improvement, such as 1.3% and 0.6% on RealWorldQA and RefCOCOs, respectively.

ISP demonstrates effectiveness on MLLM tasks compared to traditional feature pyramid. In Table. 3, it is evident that feature pyramids, regardless of their construction method, could enhance performance across various MLLM tasks. Nonetheless, the ISP constructed by VDIM achieves an average performance gain of 1.0% over bilinear interpolation, indicating that the ISP further enhances beneficial visual representations (e.g., high-frequency visual details).

[Figure 127]

[Figure 128]

(a) (b)

- Figure 5: Comparison of performance. (a) Performance of using different projectors on compressing ISP. Hiwin attention exhibits a significant advantage. (b) Performance of our model equipped with different LLMs.

Hierarchical window attention works on compressing multi-scale feature pyramid. As shown in Fig. 5a, Hiwin attention condenses visual tokens by using learnable queries on feature pyramids with arbitrary native resolutions, resulting in superior performance on fine-grained MLLM tasks (e.g., DocVQA and TextVQA). In contrast, directly down-sampling high-resolution features via interpolation before MLP projection [67] loses critical visual details and thereby degrades performance. Moreover, the Perceive Resampler [6, 110] performs notably worse due to its lack of spatial prior constraints on each query, which adversely affects training convergence, as evidenced by its significantly higher pre-training loss compared to Hiwin attention (0.6110 vs. 0.4368).

Hiwin transformer keeps superior when transferring to other LLMs. In Fig. 5b, the proposed Hiwin transformer consistently improves performance across Vicuna-13B, and Qwen2-7B, demonstrating its strong generalization capability to different model scales and architectures. Especially in tasks that require visual details, such as DocVQA and HR-Bench, the improvement is even more pronounced.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

###### LLaVA-Next

LLaVA-Next 25

The specific details about the person planting the tree, such as their name or title, are not provided in the image.

Mini-Gemini

Mini-Gemini The photo of the newspaper shows President Uhuru Kenyatta planting a tree. GPT-4V

[Figure 135]

20

[Figure 136]

###### GPT-4V

The man in blue who has his arm raised is wearing the number 24.

I'm sorry, I can't identify the person planting the tree in the newspaper photo.

LLaVA-UHD v2 (Ours) 3

LLaVA-UHD v2 (Ours) President Kibaki is planting a tree in the photo of the newspaper.

What number is on the man in blue who has his arm raised?

[Figure 137]

Who is planting a tree in the photo of the newspaper?

[Figure 138]

A B

- Figure 6: Qualitative comparison of proposed LLaVA-UHD v2 and advanced MLLMs, including LLaVA-Next, Mini-Gemini, and GPT-4V. Our method outperforms its counterparts, by providing both fine-grained visual information and high-level semantic contexts for the high-resolution complex perception tasks

Input Image & Text Token Semantic Activation on Semantic Activation on Semantic Activation on

|[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]|
|---|

|[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]|
|---|

[Figure 146]

[Figure 147]

|[Figure 148]<br><br>[Figure 149]|
|---|

[Figure 150]

|[Figure 151]<br><br>[Figure 152]|
|---|

[Figure 153]

kid down

Urban

- Figure 7: Activation response of specific textual tokens to different visual feature levels, exhibiting complement to each other. Red circles highlight the obvious difference between levels (Best viewed in color and zoomed-in).

##### 4.5 Visualization Analysis

Case study. In Fig. 6, we visualize the performance of well-known MLLMs on high-resolution, complex perception tasks. This kind of task requires MLLMs to well fuse both visual details and highlevel semantics to accurately identify fine-grained targets (e.g., OCR, colors) during the procedure of complex semantic perception (e.g., semantic relation and visual behavior). It is evident that LLaVA-UHD v2 correctly recognizes the tree planter in the newspaper photo and associates it with the name within the dense image caption (Case A). We also can see that LLaVA-UHD v2 captures the player who raises his hands and reads the “number 3” on his clothes (Case B). In contrast, LLaVA-Next overlooks the name information within dense texts (Case A) and hallucinates on the player number (Case B). Mini-Gemini fails to extract the true name (Case A) and also hallucinates (Case B). Additionally, GPT-4V shows limitations in referencing the information in the newspaper (Case A) and falsely recognizes “number 24” due to wrong fine-grained action perception (Case B).

Semantic activation cross semantic scale. In Fig. 7, we demonstrate the activation responses of specific textual prompts in the language model to the inverse semantic pyramid. As shown, OCR-like textual tokens yield finer-grained and more accurate activations at higher feature levels, facilitating accurate scene text recognition (first row). For object-level semantics, higher feature levels enhance edge detail activations, enabling more precise semantic localization (second row). Collectively, the

semantic pyramid offers a more exhaustive set of visual semantics with rich granularity, effectively supporting nuanced language decoding.

### 5 Conclusion

Our proposed hierarchical window transformer, which is the core of LLaVA-UHD v2, effectively addresses the limitations of conventional ViT-based MLLMs by capturing varying visual granularity essential for precise language generation. The Hiwin transformer adeptly constructs an inverse semantic pyramid for enriched multi-modal representation, which is then condensed into a compact set of visual tokens. This process enhances nuanced visual-linguistic alignment as well as facilitates efficient visual prompting for the LLM. LLaVA-UHD v2 shows substantial gains over the baseline method across a range of MLLM benchmarks, demonstrating its capacity in MLLM tasks that demand both low-level details and high-level semantics. Furthermore, the Hiwin transformer offers versatility, presenting potential adaptability across diverse ViT-based MLLM architectures.

### References

- [1] RealWorldQA. https://huggingface.co/datasets/xai-org/RealworldQA. 2024-09-24.
- [2] Introducing ChatGPT. https://openai.com/blog/chatgpt. 2022.
- [3] LAION-GPT4v dataset. https://huggingface.co/datasets/laion/gpt4v-dataset, 2023.
- [4] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [5] Guillaume Alain. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.
- [6] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.
- [7] Xiang An, Kaicheng Yang, Xiangzi Dai, Ziyong Feng, and Jiankang Deng. Multi-label cluster discrimination for visual representation learning. In European Conference on Computer Vision, pages 428–444. Springer, 2024.
- [8] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 1, 2023.
- [9] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In IEEE ICCV, pages 2425–2433, 2015.
- [10] Rowel Atienza. Vision transformer for fast and efficient scene text recognition, 2021.
- [11] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [12] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, , and Sagnak Tasırlar. Introducing our multimodal models. adept.ai/blog/fuyu-8b. 2023.
- [13] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.
- [14] Peter J. Burt and Edward H. Adelson. The laplacian pyramid as a compact image code. IEEE Trans. Commun., 31(4):532–540, 1983.
- [15] Yue Cao, Yangzhou Liu, Zhe Chen, Guangchen Shi, Wenhai Wang, Danhuai Zhao, and Tong Lu. Mmfuser: Multimodal multi-layer feature fuser for fine-grained vision-language understanding. arXiv preprint arXiv:2410.11829, 2024.

- [16] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In IEEE CVPR, pages 13817–13827, 2024.
- [17] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478, 2023.
- [18] Jierun Chen, Fangyun Wei, Jinjing Zhao, Sizhe Song, Bohuai Wu, Zhuoxuan Peng, S-H Gary Chan, and Hongyang Zhang. Revisiting referring expression comprehension evaluation in the era of large multimodal models. arXiv preprint arXiv:2406.16866, 2024.
- [19] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv:2311.12793, 2023.
- [20] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.
- [21] Peng Chen, Pi Bu, Jun Song, Yuan Gao, and Bo Zheng. Can vlms play action role-playing games? take black myth wukong as a study case. arXiv preprint arXiv:2409.12889, 2024.
- [22] Yangyi Chen, Xingyao Wang, Hao Peng, and Heng Ji. A single transformer for scalable vision-language modeling. arXiv preprint arXiv:2407.06438, 2024.
- [23] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023.
- [24] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/ InternLM/xtuner, 2023.
- [25] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
- [26] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k HD. arXiv preprint arXiv:2404.06512, 2024.
- [27] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.
- [28] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [29] Stephanie Fu, Mark Hamilton, Laura Brandt, Axel Feldman, Zhoutong Zhang, and William T Freeman. Featup: A model-agnostic framework for features at any resolution. arXiv preprint arXiv:2403.10516, 2024.
- [30] Quentin Gallouédec, Edward Beeching, Clément Romac, and Emmanuel Dellandréa. Jack of all trades, master of some, a multi-purpose transformer agent. arXiv preprint arXiv:2402.09844, 2024.
- [31] Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935, 2024.
- [32] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In IEEE CVPR, pages 6904–6913, 2017.
- [33] Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In ICML, pages 369–376, 2006.

- [34] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.
- [35] Zonghao Guo, Ruyi Xu, Yuan Yao, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-resolution images. In ECCV, 2024.
- [36] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. VizWiz grand challenge: Answering visual questions from blind people. In IEEE CVPR, pages 3608–3617, 2018.
- [37] Mark Hamilton, Evan Shelhamer, and William T Freeman. It is likely that your loss should be a likelihood. arXiv preprint arXiv:2007.06059, 2020.
- [38] Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T. Freeman. Unsupervised semantic segmentation by distilling feature correspondences. In ICLR, 2022.
- [39] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE CVPR, pages 770–778, 2016.
- [40] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In IEEE ICCV, pages 2961–2969, 2017.
- [41] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. CogAgent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023.
- [42] Drew A Hudson and Christopher D Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE CVPR, pages 6700–6709, 2019.
- [43] Max Jaderberg, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Synthetic data and artificial neural networks for natural scene text recognition. In NeurIPS, 2014.
- [44] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In IEEE CVPR, pages 13700–13710, 2024.
- [45] Kushal Kafle, Scott Cohen, Brian Price, and Christopher Kanan. DVQA: Understanding data visualizations via question answering. In IEEE CVPR, 2018.
- [46] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, pages 787–798, 2014.
- [47] Aniruddha Kembhavi, Michael Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. arXiv:1603.07396, 2016.
- [48] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, pages 235–251, 2016.
- [49] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv:1609.04836, 2016.
- [50] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocr-free document understanding transformer. In ECCV, pages 498–517, 2022.
- [51] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloé Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment anything. arXiv preprint arXiv:2304.02643, 2023.
- [52] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, pages 32–73, 2017.
- [53] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In ICML, pages 18893–18912, 2023.

- [54] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [55] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. OtterHD: A highresolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023.
- [56] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. ICML, 2023.
- [57] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392, 2024.
- [58] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024.
- [59] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607, 2023.
- [60] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In IEEE CVPR, pages 26689–26699, 2024.
- [61] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755, 2014.
- [62] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755, 2014.
- [63] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In IEEE CVPR, pages 2117–2125, 2017.
- [64] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, and Yu Qiao. SPHINX: the joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.
- [65] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. arXiv:2306.14565, 2023.
- [66] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, ocr, and world knowledge. https://llava-vl.github.io/ blog/2024-01-30-llava-next/. 2024.
- [67] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [68] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024.
- [69] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [70] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023.
- [71] Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473, 2024.
- [72] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In IEEE ICCV, pages 10012–10022, 2021.
- [73] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In IEEE CVPR, pages 11976–11986, 2022.
- [74] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024.

- [75] David G. Lowe. Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vis., 60(2): 91–110, 2004.
- [76] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [77] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [78] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 35:2507–2521, 2022.
- [79] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [80] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024.
- [81] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In IEEE CVPR, pages 11–20, 2016.
- [82] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In IEEE CVPR, 2019.
- [83] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv:2203.10244, 2022.
- [84] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [85] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for VQA on document images. In WACV, pages 2199–2208, 2021.
- [86] Anand Mishra, Karteek Alahari, and CV Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012.
- [87] A. Mishra, K. Alahari, and C. V. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012.
- [88] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual question answering by reading text in images. In ICDAR, pages 947–952, 2019.
- [89] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [90] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In IEEE ICCV, pages 2641–2649, 2015.
- [91] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [92] Shaoqing Ren. Faster r-cnn: Towards real-time object detection with region proposal networks. arXiv preprint arXiv:1506.01497, 2015.
- [93] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241, 2015.
- [94] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In ECCV, pages 146–162, 2022.

- [95] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024.
- [96] ShareGPT. https://sharegpt.com/, 2023.
- [97] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024.
- [98] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In ECCV, pages 742–758, 2020.
- [99] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
- [100] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In IEEE CVPR, pages 8317–8326, 2019.
- [101] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Document collection visual question answering. In ICDAR, 2021.
- [102] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024.
- [103] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [104] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [105] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. arXiv preprint arXiv:2408.15556, 2024.
- [106] Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In IEEE CVPR, pages 10126–10135, 2020.
- [107] Zhaokai Wang, Xizhou Zhu, Xue Yang, Gen Luo, Hao Li, Changyao Tian, Wenhan Dou, Junqi Ge, Lewei Lu, Yu Qiao, et al. Parameter-inverted image pyramid networks for visual perception and multimodal understanding. arXiv preprint arXiv:2501.07783, 2025.
- [108] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint arXiv:2312.06109, 2023.
- [109] Huanjin Yao, Wenhao Wu, Taojiannan Yang, YuXin Song, Mengxi Zhang, Haocheng Feng, Yifan Sun, Zhiheng Li, Wanli Ouyang, and Jingdong Wang. Dense connector for mllms. Advances in Neural Information Processing Systems, 37:33108–33140, 2025.
- [110] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [111] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. UReader: Universal OCR-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023.
- [112] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. arXiv preprint arXiv:2311.04257, 2023.
- [113] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023.

- [114] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In ECCV, pages 69–85, 2016.
- [115] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In IEEE CVPR, pages 9556–9567, 2024.
- [116] Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, and Jinqiao Wang. Griffon: Spelling out all object locations at any granularity with large language models. In ECCV, pages 405–422, 2024.
- [117] Xiaosong Zhang, Yunjie Tian, Lingxi Xie, Wei Huang, Qi Dai, Qixiang Ye, and Qi Tian. Hivit: A simpler and more efficient design of hierarchical vision transformer. In ICLR, 2023.
- [118] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llava-hd: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487, 2024.
- [119] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llava-hd: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487, 2024.
- [120] Xiangyu Zhao, Xiangtai Li, Haodong Duan, Haian Huang, Yining Li, Kai Chen, and Hua Yang. Mg-llava: Towards multi-granularity visual instruction tuning. arXiv preprint arXiv:2406.17770, 2024.
- [121] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [122] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.
- [123] Zhuofan Zong, Bingqi Ma, Dazhong Shen, Guanglu Song, Hao Shao, Dongzhi Jiang, Hongsheng Li, and Yu Liu. Mova: Adapting mixture of vision experts to multimodal context. arXiv preprint arXiv:2404.13046, 2024.

### 1 Appendix

In the supplemental materials, we report the experimental performance of the proposed inverse semantic pyramid (ISP) on general visual tasks, the details of the visual detail injection module (VDIM), and the dataset composition in the supervised fine-tuning phase of MLLM. Furthermore, we additionally analyze the behaviors of LLaVA-UHD v2 through qualitative and quantitative experiments.

### A Implementation Details

##### A.1 Experimental setting on visual tasks

Results on visual tasks. Beyond the VQA task in MLLM, we further evaluate some fundamental visual tasks, including semantic segmentation [62], optical character recognition [86], and finegrained classification [49], to compare the effect of bilinear interpolation and VDIM. In Fig. 8, the VDIM outperforms the bilinear interpolation on OCR (+4.0%), semantic segmentation (+4.7%) and fine-grained classification (+3.8%), which demonstrate more visual details are encoded for precise semantic discrimination. The implementation details for each visual task are provided below. Unless explicitly stated, both naive bilinear interpolation and the VDIM are employed to perform feature up-sampling on low-resolution (i.e., 24×24) feature maps which are extracted from the second-to-last layer of CLIP-ViT [91].

Optical character recognition. We follow the experimental setting of [10], training on MJSynth [43] dataset and evaluating on IIIT5K [87] dataset. Specifically, we first up-sample the low-resolution feature maps to high-resolution ones (i.e., 48×48) by using bilinear interpolation and a pre-trained VDIM. The resulting feature maps are then fed to a sequence encoder and a CTC head [33] to predict the class labels of characters in the images.

We train the entire model with a global batch of 96 on 8×A100s, using Adadelta optimizer with 5e−2 learning rate, and a cosine scheduler for 6800 steps. We use the character-level cross-entropy loss between the predicted labels and the ground truth for supervision. For evaluation, we report the sentence-level recognition accuracy of the test split of IIIT5K [87] dataset in Fig. 8.

Linear probing semantic segmentation. We follow the experimental setting of previous research [29, 5, 38]. To be specific, on the COCOStuff dataset [62], we train a linear projection upon a frozen CLIP-ViT to directly predict the class label of each pixel.

The input of the linear projection is the low-resolution feature maps. We train the linear projection with a global batch of 1024 on 8×A100s, using Adam optimizer with 5e−3 learning rate for 360 steps. We use the pixel-level cross-entropy loss between the predicted labels and the ground truth for supervision. During the linear probing phase, we up-sample the low-resolution feature maps to a high-resolution (i.e., 96×96) one by using bilinear interpolation or pre-trained VDIM and then directly feed them into the linear projection to predict the pixel category. We report the segmentation accuracy on the validation split of COCOStuff in Fig. 8.

Fine-grained classification. We train and assess on CUB-200 [49], a fine-grained bird classification dataset. During the training, the low-resolution feature maps are first up-sampled to high-resolution ones (96×96) with bilinear interpolation and a pre-trained VDIM. Then, a classification head with two linear layers pools the feature maps into a vector for image classification. Note that, the CLIP-ViT is frozen and only the parameters in the classification head are trainable. We set the global batch as 16 on 1×A100, using Adam optimizer with 1e−4 learning rate and a cosine scheduler for 10 epochs. We use the image-level cross-entropy loss between the predicted categories and the ground truth for supervision. We report the image classification accuracy on the CUB-200 validation dataset in Fig. 8.

##### A.2 Visual detail injection module (VDIM)

In this setction, we detail the implementation of the proposed VDIM. Specifically, each pixel value of the up-sampled feature maps is extracted as

Fl+1[x,y] = |U1| (x′,y′)∈U Up(Fl)[x′,y′] × Ddist × Dsim Θl+1(Il+1[x,y]),Θl+1(Il+1[x′,y′]) , (6)

|||Bilinear VDIM<br><br>|
|---|
|
|---|
| |
|77.6|
|73.6<br><br>69.5 70.1<br><br>73.9|
|64.8|
<br><br>55<br><br>60<br><br>65<br><br>70<br><br>75<br><br>80<br><br>OCR Seg Cls<br><br>Accuracy(%)<br><br>Performance on different visual tasks with VDIM and vanilla bilinear optical character recognition, “Seg" the Linear probing semantic|
|---|

Accuracy(%)

- Figure 8: Pe interpolation. “OCR" denotes the segmentation, and “Cls" the fine-grained classification on SUB-200.

- Table 4: Comparison of different choices of feature level on performance and efficiency. ESTVQA [106] is a VQA benchmark focusing on scene text recognition.

Efficiency Performance

Level

Period(h)

Memory(G)

Average

GQA

SQA

REC

CVQA

TVQA

ESTVQA

PMME 0,2 27.7 41.9 63.4 63.9 69.5 71.5 60.5 66.5 40.6 71.0 0,1,2 28.0 41.9 63.7 63.8 70.2 71.8 60.5 66.9 40.8 72.1 0,1,2,3 45.6 53.0 63.8 64.4 69.3 72.6 60.7 66.4 41.6 71.4 0,1,2,3 (w/o MRL) 45.6 52.6 62.4 63.6 69.8 67.1 57.8 66.5 39.9 72.0

From this image, when did television program 'H20 X5 Mop' begin?

LLaVA-Next

The television program "H20 X5 Mop" is scheduled to begin at 9:00 PM on the channel listed as "All Channels.

Mini-Gemini The television program 'H20 X5 Mop' began at 09:30, as indicated by the time listed next to its channel number 820 on the TV guide.

GPT-4V The television program "H20 X5 Mop" began at 09:31.

LLaVA-UHD v2 (Ours) 9:31 From this image, when did the show

'The 3 Mile' perform?

LLaVA-Next The image shows a website for "16th Avenue," which appears to be a music venue or band. The "Upcoming Shows" section lists a performance by "The 3 Mile" on December 29, 2006, at the Imperial Theatre in Saint John, New Brunswick.

Mini-Gemini The show 'The 3 Mile' performed on 01/22/2007, as indicated in the 'upcoming shows' section of the website.

GPT-4V The show at "The 3 Mile" in Saint John, NB, was scheduled for February 3, 2007.

LLaVA-UHD v2 (Ours) 02/03/2007

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

How much is the stoli on sale for?

LLaVA-Next 1.79 Mini-Gemini 2.99 GPT-4V The Stoli is on sale for $23.99. LLaVA-UHD v2 (Ours) 23.99

[Figure 158]

[Figure 159]

|[Figure 160]<br><br>[Figure 161]<br><br>|[Figure 162]|
|---|
|
|---|

|[Figure 163]<br><br>[Figure 164]<br><br>|[Figure 165]|
|---|
|
|---|

|[Figure 166]<br><br>[Figure 167]<br><br>|[Figure 168]|
|---|
|
|---|

A B C D

From the phone, what is the duration of whole workouts?

LLaVA-Next The image shows a smartphone displaying a workout summary with the duration of the whole workout as 1 hour and 45 minutes.

Mini-Gemini The duration of the whole workout, as shown on the smartphone, is 3 hours and 31 minutes.

GPT-4V The duration of the whole workouts shown on the phone is 3 hours, 7 minutes, and 21 seconds.

LLaVA-UHD v2 (Ours) 3:07:21

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

- Figure 9: Qualitative comparison on high-resolution dense perception task which requires the capabilities of fine-grained details perception.

Data Size Response formatting prompts

LLaVA [67] 158K – ShareGPT [96] 40K –

VQAv2 [32] 83K Answer the question using a single word or

phrase.

GQA [42] 72K OKVQA [82] 9K OCRVQA [88] 80K DocVQA [101] 15K ChartQA [83] 20K

A-OKVQA [94] 66K Answer directly with the option’s letter from

the given choices. DVQA [45] 20K – TextCaps [98] 22K Provide a one-sentence caption for the provided

image. ShareGPT4V [19] 55K – AI2D [47] 3K – LAIONGPT4V [3]

11K –

SythDog-EN [50] 40K – LRV-Instruct [65] 30K – RefCOCO 48K [46, 81] Provide a short description for this region. (for Region Caption)

VG [52] 86K Provide the bounding box coordinate of the region this sentence describes. (for Referring Expression Comprehension)

Total 858K

Table 5: Detailed composition of our 858k-mixed dataset.

Table 6: Comparison of different choices of grid sizes on performance and efficiency. “Pyramid" means the feature grids from different levels form a region-level feature pyramid, e.g., [2×3] for level-0, [4×6] for level-1, [8×12] for level-2. “Fix" represents all feature maps are pooled into a 3×3 feature grid. “Selective" represents adaptively selecting a grid size that is closest to the resolution. We measure the training period on 8×A100s, the latency on an A100 with a 1008×672 image, and the GPU memory on 8×A100s with 1 image per GPU in supervised fine-tun ing phase.

Efficiency General Knowledge OCR & Chart Method Period(h) Latency(s) Memory(G) Average MMEP GQA AI2D VQAC VQAT VQAD

Pyramid 62.4 1.26 60.3 62.4 69.0 60.8 57.3 60.7 67.5 58.9 Fix [3×3] 26.9 0.62 41.7 64.6 73.8 63.9 58.8 60.9 66.2 63.8 Selective 27.7 0.54 39.4 65.3 73.0 64.6 58.3 62.5 68.5 65.0

where [x,y] denotes the coordinate index of feature maps, U the region of convolution neighborhood, Ddist is a decay factor based on the distance between [x,y] and [x′,y′], and Dsim is a similarity weight based on the attention of each image pixel using a fully connected layer Θl+1. By iteratively performing Eq. 6, we progressively up-sample the feature maps and finally construct the ISP {F0,F1,F2}.

Up-sampling kernel. Following [37], the kernel weight Θl+1 in Eq.6 of the VDIM relies on two weights Ddist and Dsim. Spatial distance decay Ddist is defined as

∥(x,y) − (x′,y′)∥22 2σdist2

Ddist = exp −

(7)

to represent the Euclidean distance relations between adjacent pixels, where σdist denotes a learnable width. And pixel similarity weight Dsim is determined as

Dsim = softmax (x′,y′)∈U

Θl+1(Il+1[x,y]) · Θl+1(Il+1[x′,y′]) σsim2

, (8)

[Figure 174]

[Figure 175]

|[Figure 176]<br><br>[Figure 177]<br><br>|[Figure 178]|
|---|
|
|---|

|[Figure 179]<br><br>[Figure 180]<br><br>|[Figure 181]|
|---|
|
|---|

LLaVA-Next 1073 Mini-Gemini 123 GPT-4V The number on the very end of the bus is 7012. LLaVA-UHD v2 (Ours) 7012.

LLaVA-Next 22 Mini-Gemini 22 GPT-4V The number on the basketball jersey worn by the man on the left is 2. LLaVA-UHD v2 (Ours) 2

[Figure 182]

What is the number on the very end of the bus?

[Figure 183]

What number is on the basketball jersey worn by the man on the left?

A B C D

[Figure 184]

[Figure 185]

|[Figure 186]<br><br>[Figure 187]<br><br>|[Figure 188]|
|---|
|
|---|

|[Figure 189]<br><br>[Figure 190]<br><br>|[Figure 191]|
|---|
|
|---|

LLaVA-Next 8 hours Mini-Gemini 8 hrs GPT-4V The slow cooker is set to cook on "LOW" for 8 hours. LLaVA-UHD v2 (Ours) 6 hours

LLaVA-Next 12:00 Mini-Gemini 12:41 GPT-4V The time on the phone on the left is 12:28 PM. LLaVA-UHD v2 (Ours) 12:28 pm

[Figure 192]

[Figure 193]

How long is this set to cook?

What time is on the phone on the left?

- Figure 10: Qualitative comparison on high-resolution fine-grained perception task which requires robust fine-grained visual texture perception capabilities.

What is the number above the right tire on the car?

LLaVA-Next 1319-03 Mini-Gemini 1319-03 GPT-4V The number above the right tire on the car is "153". LLaVA-UHD v2 (Ours) 153

What is the number of the player in the middle?

LLaVA-Next 2 Mini-Gemini

- 2

GPT-4V The number of the player in the middle is 3

LLaVA-UHD v2 (Ours)

- 3

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

What number is the coin on the middle right?

LLaVA-Next 36 Mini-Gemini 55 GPT-4V The number of the coin on the middle right is 46. LLaVA-UHD v2 (Ours) 46

What is the lowest ml?

LLaVA-Next 40 Mini-Gemini 1 GPT-4V The lowest measurement visible on the flasks in the image is 20 ml. LLaVA-UHD v2 (Ours) 20

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

|[Figure 202]<br><br>[Figure 203]<br><br>|[Figure 204]|
|---|
|
|---|

|[Figure 205]<br><br>[Figure 206]<br><br>|[Figure 207]|
|---|
|
|---|

|[Figure 208]<br><br>[Figure 209]<br><br>|[Figure 210]|
|---|
|
|---|

|[Figure 211]<br><br>[Figure 212]<br><br>|[Figure 213]|
|---|
|
|---|

A B C D

- Figure 11: Qualitative comparison on high-resolution spatial perception which necessitates the capabilities of high-level spatial contexts.

where σsim is a learnable temperature factor to modulate the distribution of similarity scores and U = 7.

Learnable down-sampler. We detail the implementation of the learnable down-sampler defined in Eq.2 of the main paper. Compared to a simple convolutional layer with a stride of 2, we apply an attention down-sampler following [29] at each feature level.

Specifically, a 1×1 convolution layer is first carried out on the feature maps of (l + 1)-th level to extract a saliency map, followed by combining it with a modified fully-connected layer to normalize

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

- Figure 12: PCA visualization of the up-sampled features by VDIM on nature scene. With VIDM, the high-resolution features could clearly depict object boundaries and text appearance. (Best viewed in color and zoomed in)

the features in the local neighborhood V . We summarize the above operation as a network f(·) with trainable parameters Ωl+1. As a result, the feature pixel at the location of [x,y] on down-sampled feature maps (of l-th level) is formally defined as

#### Down(Fl+1;Ωl+1)[x,y] = softmax f(Fl+1[Vx,y];Ωl+1) · Fl+1[Vx,y], (9)

where Vx,y denotes the local neighborhood in the high-resolution feature maps. Note that, before performing Eq. 9, we experimentally up-sample the Fl+1 to the size of the original image using bilinear interpolation. And we set Vx,y = 14, aligning with the patch size of CLIP-ViT, to simulate the feature extraction of ViT.

##### A.3 Supervised fine-tuning dataset

As illustrated in Table 5, we detail the proposed 858k-mixed dataset in the supervised fine-tuning phase of MLLM.

### B Analysis

##### B.1 Quantitative experiment

Level choice. As shown in Table 4, the introduction of higher level (higher resolution) feature maps results in consistent enhancement of the average performance. However, incorporation of even higher

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

- Figure 13: PCA visualization of the up-sampled features by VDIM on OCR scene. With VDIM, the high-resolution features could clearly depict object boundaries and text appearance. (Best viewed in color and zoomed in)

resolution features, such as level-3 feature maps (i.e., 8× resolution than level-0), yields marginal benefits while substantially increasing the training cost.

When the multi-level reconstruction (MLR) supervision is replaced by the last-level supervision, performance deteriorates. In such a scenario, the VDIM faces substantial challenges in effectively incorporating high-frequency information into CLIP-ViT features.

The choice of RoI-align grid sizes We explore the impact of RoI-align grid size on the efficiency and performance of LLaVA-UHD v2. Specifically, we RoI-align a region-level feature pyramid from the inverse feature pyramid within a window set, ensuring that higher-resolution feature maps retain finer-grained pooling grids, like [102]. However, this approach, rather than improving multi-scale feature integration, significantly degrades performance and inference efficiency, as demonstrated in Table 6. Compared to fixed grid, selecting a proper pooling grid (defined in Eq.3 of main paper) showcases better performance and efficiency, because of a more approximate aspect ratio to the native image.

- B.2 Qualitative experiment

- B.2.1 Enhanced high-resolution features.

We performed a qualitative visualization of our ISP, presented in Fig.12 and Fig.13, Note that all the high-resolution features shown are the highest-level of ISP enhanced from CILP-ViT features. While bilinear interpolation increases the nominal resolution of features, it fails to enhance the fidelity of

image detail representation. Using the FeatUp [29] for feature up-sampling, the resulting features capture finer details but retain a degree of blurriness. In contrast, with the proposed VIDM, the high-resolution features clearly depict object boundary and text appearance, demonstrating accurate and refined representation of visual details.

##### B.2.2 Case studies.

We add more cases to analyze the behavior of our LLaVA-UHD v2. The capabilities are summarized as three aspects as follows.

Dense Perception. In Fig. 9, we visualize the performance of well-known MLLMs on dense perception tasks. Dense perception tasks require models to possess highly robust fine-grained perception capabilities to distinguish object boundaries within a large number of densely packed similar objects, thereby accurately locating the target and its boundaries to identify the target precisely.

It is evident that LLaVA-UHD v2 and GPT-4V accurately identify the beginning time of the television program ‘H20 X5 Mop’ (case A), the performance date of the show ‘The 3 Mile’ (case B), the duration of whole workouts (case C), and the prize of Stoli (case D), indicating highly robust fine-grained perception capabilities provided by our visual pyramid representation. In comparison, other models either fail to precisely locate the target (LLaVA-Next) or cannot distinguish the target from similar adjacent objects, limited in accurately completing dense OCR tasks (Mini-Gemini).

Fine-grained Perception. In Fig. 10, we visualized the performance of well-known MLLM on fine-grained perception tasks. These tasks require models to have robust fine-grained perception capabilities to detect the textures of small or blurry targets, thereby accurately locating and identifying small targets.

Case C indicates that LLaVA-UHD v2 accurately identified the small green light, and the tiny number of duration time associated with green light, demonstrating that the introduction of high-frequency information in hierarchical features can handle small, blurry targets effectively. In contrast, other models can not find the small green light, or fail to accurately perform OCR tasks due to the text being too small or blurry (e.g., GPT-4V, LLaVA-Next, Mini-Gemini).

This capability is further demonstrated in cases A, B and D, where both LLaVA-UHD v2 and GPT-4V accurately identified the tiny number on the basketball jersey (case A), the blurry number on the very end of the bus (case B), and the time on the phone (case D), while LLaVA-Next and Mini-Gemini exhibited limitations.

Spatial Perception. In Fig. 11, we visualized the performance of well-known MLLM on spatial perception tasks. Spatial perception tasks require models to have robust high-semantic perception capabilities to discern the spatial relationships between different objects.

It is evident that LLaVA-UHD v2 and GPT4V perceive the spatial relative positions between different objects, to accurately identify the number above the right tire on the car (case A), the number of the player in the middle (case B), the number of the coin on the middle right (case C), the lowest ml (case D). This accuracy is attributed to our high-resolution visual pyramid representation, which allows the perfect integration of features of varying semantics and spatial information across different levels. In contrast, other models, such as LLaVA-Next and Mini-Gemini, fail to accurately perceive these relative spatial positions.

