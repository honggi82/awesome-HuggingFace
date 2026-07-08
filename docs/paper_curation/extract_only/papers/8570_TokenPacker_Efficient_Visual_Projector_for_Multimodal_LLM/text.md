# arXiv:2407.02392v4[cs.CV]28Aug2024

## TokenPacker: Efficient Visual Projector for Multimodal LLM

Wentong Li1∗, Yuqian Yuan1∗, Jian Liu2, Dongqi Tang2, Song Wang1, Jie Qin3, Jianke Zhu1†, Lei Zhang4 1Zhejiang University 2Ant Group 3NUAA 4The Hong Kong Polytechnical University

### Abstract

The visual projector serves as an essential bridge between the visual encoder and the Large Language Model (LLM) in a Multimodal LLM (MLLM). Typically, MLLMs adopt a simple MLP to preserve all visual contexts via one-to-one transformation. However, the visual tokens are redundant and can be considerably increased when dealing with high-resolution images, impairing the efficiency of MLLMs significantly. Some recent works have introduced resampler or abstractor to reduce the number of resulting visual tokens. Unfortunately, they fail to capture finer details and undermine the visual reasoning capabilities of MLLMs. In this work, we propose a novel visual projector, which adopts a coarse-to-fine scheme to inject the enriched characteristics to generate the condensed visual tokens. In specific, we first interpolate the visual features as a low-resolution point query, providing the overall visual representation as the foundation. Then, we introduce a regionto-point injection module that utilizes high-resolution, multi-level region-based cues as fine-grained reference keys and values, allowing them to be fully absorbed within the corresponding local context region. This step effectively updates the coarse point query, transforming it into an enriched one for the subsequent LLM reasoning. Extensive experiments demonstrate that our approach compresses the visual tokens by 75%∼89%, while achieves comparable or even better performance across diverse benchmarks with significantly higher efficiency. The source codes can be found at https://github.com/CircleRadon/TokenPacker.

### 1 Introduction

With the rapid evolution in Large Language Models (LLM) [63, 49, 3, 50, 51, 1, 22], Multimodal Large Language Models (MLLMs) [35, 33, 34, 4, 64, 9, 15, 45, 62] has witnessed a significant surge in vision-language understanding, reasoning, and interaction capabilities. This is achieved by projecting embeddings from a visual encoder into LLM to enable their visual perception of the world, where visual projector plays a crucial role to bridge the vision and language model.

In the framework of MLLMs, the LLM predominantly drives the entire computation cost, particularly since the visual encoder tends to be substantially smaller compared to the LLM. For instance, the widely used CLIP-ViT-Large [44], which features 0.3 billion parameters, stands in stark contrast to LLMs such as LLaMA [50] or Vicuna [52] with 7/8 billion or 13 billion parameters. Consequently, the efficiency of MLLMs is significantly affected by the number of resulting visual tokens from visual projector. Besides, the visual projector connects the vision and language models by translating visual features into visual tokens in a text embedding space that language model can interpret. Therefore, the quality of these visual tokens directly affects the overall efficacy of MLLM. In this work, we aim

∗Equal contribution. †Corresponding author.

Preprint. Work in progress.

| | | |Ours|Token num|576<br><br>ber|
|---|---|---|---|---|---|
|ML|P| | | |144|
| | | |LDP-v2<br><br>Pixel-<br><br>|shuffle<br><br>|64|
| |C-|Abstractor<br><br>Res|ampler| | |

[Figure 1]

[Figure 2]

[Figure 3]

62.5

62.0

Accuracy

[Figure 4]

[Figure 5]

61.5

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

61.0

60.5

60.0

59.5

###### (a) Linear Projector (b) Resampler (c) TokenPacker (Ours)

5 24 25 26 27 28 29 30

TPS

- Figure 1: (Left) Visual comparisons on typical projectors, including linear MLP [33] and Resampler [4]. Our approach mines multi-level features in a local context region. (Right) Accuracy vs. efficiency (TPS) comparisons with existing methods. Our TokenPacker shows a favorable performance against other counterparts. The accuracy is averaged across six benchmarks (see Table 1).

to investigate an effective visual projector for an MLLM that bridges the vision encoder and LLM with high quality, while making use of the fewer number of tokens possible.

Most of current works adopt either linear projector [35, 33] or resampler [4, 13, 57]. As for linear projector, MLP projection [33] preserves all visual contexts via one-to-one transformation, which retains the detailed information having redundant tokens [6, 46]. More importantly, the number of visual tokens is significantly increased in dealing with high-resolution images or videos. As for another research line, resampler [4] or Q-Former [27] leverage a group of learnable queries to explicitly controls the number of visual tokens and adopt the cross-attention layers to force extracting the most relevant visual cues from visual features. Some recent studies, e.g. Abstractor [7] or LDP [11, 12], utilize convolution layers to encourage local interaction of visual features and generate the compressed tokens. Nonetheless, these methods inevitably lose the finer details information and sacrifice the visual reasoning capabilities of MLLMs. Besides, some methods directly transfer visual features from sequence dimension to channel dimension by a simple pixel shuffle [9] or nearby concatenation operation [15] to reduce the length of sequence. Although having preserved all information, it may destroy the structural characteristics of the visual feature itself.

In this work, we propose a novel visual projector, dubbed TokenPacker, which effectively packs the finer detailed information into compact visual token representations. Our TokenPacker aligns with a coarse-to-fine design, which injects enriched high-resolution characteristics into a coarse low-resolution one to generate the condensed visual tokens. Specifically, we initially interpolate visual feature from the vision encoder as low-resolution point queries, which contain coarse and holistic characteristics of visual cues. Then, we introduce a region-to-point injection module, which makes full use of high-resolution, multi-level CLIP features to provide fine-grained candidate keys and values for reference. During this process, high-resolution visual region details are encouraged to inject into the low-resolution point query to be updated within a local context region. This effectively enhances the coarse query and transforms it into a more enriched one for the subsequent LLM. As an extension, we further present an effective dynamic image slicing scheme to perform efficient high-resolution image understanding with our TokenPacker.

Extensive experiments are conducted across diverse multimodal benchmarks to investigate the efficacy of our approach. Notably, our TokenPacker can effectively reduce 75% (576 vs. 144)∼89% (576 vs. 64) visual tokens in LLaVA-1.5 [33] while achieving comparable or even better performance with significantly higher efficiency. As illustrated in Fig. 1, our method exhibits a more favorable superiority on accuracy and efficiency against other counterparts. Additionally, our approach consistently delivers competitive high-resolution comprehension performance on a variety of multimodal tasks.

### 2 Related Work

#### 2.1 Multimodal Large Language Models (MLLMs)

Large Language Models (LLMs) [50, 51, 63, 3, 49, 1, 22, 1] have attracted considerable attention for their remarkable capabilities across various linguistic tasks, such as question answering and text generation. This wave of interest has paved the way for the development of recent Multimodal Large Language Models (MLLMs) [62], which integrate LLMs with visual encoders to enable an enriched comprehension and understanding of multimodal content. Innovative models like CLIP [44] have

significantly narrowed the gap between language processing and visual tasks, boosting the cross-modal applications. Early efforts such as Flamingo [2] and BLIP-2 [27], have leveraged extensive datasets of image-text pairs to refine cross-modal alignment, substantially enhancing learning efficiency. This enhancement represents a notable advancement in the field of MLLMs, expanding the scope of applications by accommodating both text and imagery. In the recent year, a variety of MLLMs have gained prominence. Notable open-source examples include the LLaVA series [35, 33, 34], MiniGPT4 [64], Qwen-VL [4], CogVLM [53], Shikra [8], InternLM-XComposer [14] and among others [10, 39]. The emergence of proprietary commercial MLLMs marks a pivotal shift in the landscape, as seen with OpenAI’s GPT-4V [43] and Google’s Gemini series [48, 45]. These advancements highlight the diverse and expanding landscape of MLLMs in the field, which has remarkably impacted the landscape of AGI.

#### 2.2 Visual Projector in MLLMs

Visual projector plays a fundamental role to bridge the vision and language model, which aligns visual signals from a visual encoder with the LLM space. Current approaches can be mainly divided into two categories. One is linear projection [35, 33] through MLP. MLP projection can preserve all visual contexts through a one-to-one transformation, which retains the detailed information with redundant tokens [6, 46]. A critical concern with this method is the substantial increment in the number of visual tokens, especially in processing high-resolution images or videos. To tackle this issue, another research line focuses on the reduction of visual tokens to improve the efficiency of MLLMs. Resampler [4] or Q-Former [27] employs learnable queries to explicitly controls the number of visual tokens and force extracting the most relevant visual cues from visual features by cross-attention layers. Building on Resampler, Yu et al. [59] propose a Query Proposal Network (QPN) to generate the initial query and perform the multi-level cross attention. Some recent works, e.g. Abstractor [7] and LDP [11, 12] adopt convolution layers to encourage local interaction of visual features and generate the compressed tokens. However, these methods inevitably omit fine detailed information, thereby compromising visual reasoning abilities of MLLMs. Additionally, some works directly transfer visual features from the length dimension to channel dimension by a simple pixel shuffle [9] or nearby concatenation [15] operation to reduce the number of visual tokens. Although all information are retained, it may destroy intrinsic characteristics of the visual feature itself. Recent research [41] has undertaken an empirical study on commonly-used projectors, concluding that their types have negligible effect. In contrast to these findings, this paper introduces a novel and effective visual projector dubbed TokenPacker.

#### 2.3 High-Resolution Understanding with MLLMs

Most of MLLMs commonly utilize CLIP-ViT [44] as the visual encoder to capture visual information. However, the vision encoder is constrained by low-resolution input, such as 224×224 or 336× 336, which impedes the ability of MLLMs to effectively manage tasks that require finer details, like dense OCR, crowd counting and visual grounding of small objects. To overcome this limitation, a group of methods [54, 19, 60, 28, 39] directly employ the visual encoder, like SAM encoder [24] or ConvNeXt [38], that efficiently supports high-resolution input to capture the finer visual cues. Different from these methods, the patch-cropping strategies are introduced to split a high-resolution image into multiple image patches. The image patches are then processed separately to obtain the visual embeddings of the entire high-resolution image. Some works [32, 30] first resize input image into an accessible size, and adopts the sliding windows to segment images into the uniform patches (e.g. 224×224). While these methods change the raw resolution into a fixed square size, this may result in blurring or distortion of visual content. To alleviate this problem, several studies [56, 34, 16, 9] leverage a similar aspect ratio with input image to resize, instead of adhering to a fixed square ratio.

### 3 Method

In this section, we first revisit the overall framework of a standard MLLM that generates instructionfollowing response for the given multimodal inputs (Section 3.1). Then, we introduce our effective visual projector named TokenPacker, specially designed for bridging visual encoder and LLM, to generate the condensed visual token representations for the subsequent LLM processing (Section 3.2). Finally, we present an dynamic image slicing scheme that supports input images in any aspect ratios

###### Visual tokens TokenPacker

[Figure 11]

Feature Maps

Input text What is the right part of the image?

𝟏 × 𝑀 × 𝐶

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

TokenPacker

Injection Module

Connector

Point to Region Attention

###### Visual tokens Text tokens

point-region pairs

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

…

[Figure 12]

𝟏 × 𝑀 × 𝐶 𝒔𝟐 × 𝑀 × 𝐶

Visual Encoder

[Figure 13]

Flatten Reshape&Flatten

LLM

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | |
|---|---|
| | |

𝑀 × 𝐶 𝑁 × 𝐶

[Figure 14]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

MLP s × downsample

Output text

A little dog sticks out its tongue and

Input Image Interpolate

Multi-level

looks forward happily.

Visual flow

Feature Maps

Language flow

- Figure 2: (Left) Overview of a standard MLLM framework with our TokenPacker as the visual projector. (Right) The architecture of TokenPacker. TokenPacker initially interpolates visual features as a low-resolution point query. Subsequently, high-resolution and multi-level region cues are treated as reference keys and values to inject their finer information to update coarse query via point to region attention in a local context. TokenPacker can generate the compact visual tokens in small quantities, yet encapsulate rich details efficiently.

with a minimal padding content. By integrating TokenPaker, our approach can achieve fine-grained high-resolution image understanding with significant computation efficiency (Section 3.3).

#### 3.1 Revisiting Multimodal Large Language Models (MLLMs)

The aim of MLLMs is to develop a sophisticated model capable of generating producing responses that adhere to given instructions upon multimodal inputs, including visual and textual data. MLLMs are typically composed of three pivotal components: 1) Visual Encoder FI: it converts an input image Iimg ∈ RH×W×3 into a group of distinctive visual embeddings Iv ∈ RN×C. It always leverages the widely-used CLIP-ViT-L/14 as its backbone with a patch size P of 14, and N = HW/P2 denotes the number of visual embeddings. 2) Visual Projector ΓI→T: this component translates visual embedding Iv into the visual token Tv in the textual embedding space T with an appropriate dimension for the subsequent language model. 3) LLM Φ(T

v,Tt): it takes in both visual token Tv and textual token Tt, and produces a coherent response auto-regressively. For a sequence of response with length L, the probability of generating contextually target answers Y = {yi}Li=1 can be calculated by:

L

p(yi|Tv,Tt,<i,Y<i). (1)

p(Y|Tv,Tt) =

i=1

In this typical MLLM framework, the computational and memory demands are predominantly dictated by the LLM Φ(T

v,Tt) with large amount of parameters. It should be emphasized that the computational expenses of LLM Φ(T

v,Tt) generally exhibit a quadratic increase relative to the quantity of its input tokens. This highlights the significant impact that the quantity of input tokens has on the overall efficiency of the framework. The visual projector takes the N visual embeddings Iv and converted them to M visual tokens Tv. Therefore, reducing the number of visual tokens is a pivotal approach to bolster the efficiency of LLM, i.e. M < N.

#### 3.2 TokenPacker: an Efficient Visual Projector

Visual projector plays a vital role in translating the N visual embeddings Iv into M visual tokens Tv before feeding into LLM. As shown in Figure 2, we introduce an effective visual projector, namely TokenPacker, which connects the vision encoder and language model using as small number of tokens as possible. The architecture of our TokenPacker is crafted with a coarse-to-fine framework.

, ,

[Figure 15]

[Figure 16]

| | |[Figure 17]|
|---|---|---|
| | | |

[Figure 18]

[Figure 19]

|[Figure 20]|
|---|

|[Figure 21]|
|---|

Dynamic Image Slicing

(2, 3)

VisualEncoder

336x2

336x2

TokenPacker

\n

336x3 336x3

[Figure 22]

Minimum Padding

[Figure 23]

[Figure 24]

, ,

[Figure 25]

Aspect Ratio-preserving Resize & Padding

\n

336x336

\n

- Figure 3: The pipeline for efficient high-resolution image understanding with our TokenPacker.

Specifically, we initially downsample the visual features Iv ∈ RN×C before the last Transformer layer of CLIP-based vision encoder via bilinear interpolation with a scaling factor s as the low-

resolution visual embeddings Iv′ ∈ RM×C, where M = N/s2. Therefore, the number of visual token M can be controlled by the down-sampling ratio s. The low-resolution Iv′ can be regarded as the coarse representations of original high-resolution visual features, where each pixel of low-resolution Iv′ corresponds to a specific (s×s) sub-region of the high-resolution Iv. Subsequently, we construct the point-region pairs, i.e. each pixel in Iv′ ∈ R1×M×C to sub-region in Iv ∈ Rs

2×M×C, and aim to infuse the detailed information of high-resolution sub-region into each pixel with coarse representation. To accomplish this process, we devise an injection module that effectively performs region-to-point information injection to enhance and update the low-resolution representations.

2×M×C

In particular, we take the low-resolution Iv′ ∈ R1×M×C as point-based queries, and Iv ∈ Rs

- as region-based candidate keys and values for reference. The region-to-point information injection is conducted by a point to region cross-attention operation following a MLP layer to make the low-resolution queries fully absorb the fine-grained keys and values and update to be a compact

and enhanced visual tokens Tv. Furthermore, we leverage multi-level visual features as the more enriched reference keys and values. As evidence in prior work [23], different layers of CLIP encoder display varying biases towards different patterns. The shallow layer features contain detailed lowlevel information, while deep layer features are superior at semantic understanding. The multi-level region-to-point injection process encourages to infuse the plentiful high-resolution information from multiple layers into low-resolution queries, being sufficient to serve as visual tokens. Therefore, our approach is capable of producing superior visual tokens while simultaneously reducing the total number of visual tokens to 1/s2 of the visual embeddings.

#### 3.3 High-Resolution Image Understanding with TokenPacker

To support efficient high-resolution image understanding, we further develop an effective image cropping method with our TokenPacker. Inspired by previous work [56], we focus on an aspect ratio-preserving slicing scheme to avoid the deformation and distortion of visual content that results from resizing operations. Different from the prior approaches [56, 34, 16, 9], we suggest a dynamic image slicing scheme to preserve any aspect-ratio with the minimum padding as possible to ensure the splitted grid is maximally filled with original image content.

Initially, we specify a set of grids G = {(nH,nW) | nH × nW ≤ Ng,nH ∈ N,nW ∈ N} with various partition configurations for input images. Here, nH and nW are the number of rows and columns of the grids, and Ng denotes the maximum allowable number of grids. To obtain an optimal grid configuration for a given image Iimg ∈ RH×W×3, we mainly consider three critical factors: 1) preserving the image’s original aspect ratio to avoid distortion; 2) minimizing the padding proportion so that most of the grids are occupied by original image content; and 3) among the options meeting the first two points, choosing the one whose resolution aligns most closely with the image.

To fulfill the above conditions, we define the padding score Sp and overlap score So as following:

H × W × α2 nH × nW × r2

, (2)

Sp(H,W,r,nH,nW) =

So(H,W,r,nH,nW) = IoU((H,W),(nH × r,nW × r)), (3) where α is the minimum of two ratios, i.e., α = min(αH,αW). Specifically, αH = n

H×r

H and αW = n

W ×r

W . r denotes the size of each grid, we set 336 using CLIP-ViT-L/14 as vision encoder.

Accordingly, the grids suitable for an image can be identified as follows,

nH∗,nW∗ = arg max

Sp(H,W,r,nH,nW) + βSo(H,W,r,nH,nW). (4)

(nH,nW )∈G

As shown in Fig 3, we can obtain image grids of varying sizes with proper configuration for slicing. Then, we resize raw image by the ratio α and pad the remaining part with zero. To preserve the integrity of original image, we also integrate resized original image with aspect ratio preserving to provide a macroscopic overview as in previous works [56, 33]. Following the feature extraction of these image patches, our TokenPacker generates the compact visual tokens for each splitted grid and merge to a sequence of visual tokens according to its original arrangement. Besides, we introduce the comma (‘,’) among each grid, and present a newline (‘\n’) token at the end of each row of the image grids to clarify the 2D structure information of image and avoid the ambiguity in the LLM.

### 4 Experiments

In this section, we first introduce the details of our experimental setup. Then, we benchmark our approach against leading methods across various multimodal testbeds. At the end of this section, the ablation analysis and qualitative results are presented.

#### 4.1 Implementation Details

In this work, we instantiate our approach on the top of LLaVA-1.5 [33]. Specifically, we employed CLIP-ViT-L/14-336px [44] as visual encoder with the default resolution of 336×336, and adopted Vicuna-7/13B model [63] as the LLM. We perform a two-stage training paradigm, consisting of a pre-training phase and an instruction-tuning phase. To ensure efficiency in training, we maintain vision encoders fixed across both stages, while focusing on optimizing our proposed TokenPacker. Concurrently, the optimization of the LLM is exclusively conducted during the instruction-tuning phase. We adjust the down-sampling ratio s ∈ {2,3,4} in TokenPacker to control the quantity of generated visual tokens. In the dynamic slicing scheme for high-resolution image, we set Ng = 9 or Ng = 16 for model training and evaluation to support a range of resolutions, such as 1344×1344, 5376×336, etc. In Eq. 4, we assign β = 0.1 by default. As in [33], we train all models for one epoch by leveraging the AdamW optimizer with a Cosine learning rate schedule. We set the initial learning rates for pre-training phase and instruction tuning phase at 1e−3 and 2e−5, respectively. The models are trained on 8 × NVIDIA A100 GPUs.

#### 4.2 Datasets and Benchmarks

To make a fair comparison, we first conduct the experiments on CC-558K dataset [35] for training our TokenPacker in order to perform the modality alignment at the first stage. The 665K mixture dataset [35] is employed for instruction-tuning at the second stage, following LLaVA-1.5 [33]. To achieve the competitive performance, we then adopt more high-quality training samples as organized in Mini-Gemini [28], around 1.2M for the first stage and 1.5M for the second stage. Furthermore, we conduct an extensive evaluation across a series of widely-used benchmarks to assess multimodal understanding and reasoning capability of our proposed model. The benchmarks employed in our study consist of: 1) General visual question answering benchmarks, like VQAv2 [17], GQA [20], VizWiz [18]; 2) OCR-related benchmarks, like VQAT (TextVQA) [47], OCRBench (OCRB) [37] and DocmentVQA(DocVQA) [40]; 3) Hallucination benchmark like POPE [29]; 4) Comprehensive benchmarks like MMB (MMBench) [36], MM-Vet [58] and MMMU [61].

#### 4.3 Main Results

Normal Resolution. We first examine the effectiveness of our proposed TokenPacker in normal resolution settings with the data as in LLaVA-1.5 [33]. We compare our approach with the previous leading methods, including MobileVLM V2 [12], Shikra [8], IDEFICS [21], Qwen-VL [4], and InstructBLIP [13], LLaVA-PruMerge [46] with fewer visual tokens. Six popular benchmarks are adopted including comprehensive MMBench and MM-Vet, and general VQA-related VizWiz, VQAv2, GQA and hallucination POPE for a thorough performance evaluation. As shown in Table 1, our approach showcases the superior performance on the MMBench, VizWiz and POPE benchmarks, respectively. When juxtaposed with the baseline LLaVA-1.5 model, our proposed TokenPacker as the

- Table 1: Comparison with leading methods on zero-shot benchmarks. Our TokenPacker compresses the visual tokens from 576 to 144 (1/4), 64 (1/9) or 36 (1/16) while still delivering competitive performance in comparison to LLaVA-1.5. The results of our method are highlighted with ■.

Method LLM Res. #Token PT IT MMB MM-Vet VizWiz VQAv2 GQA POPE Avg.

|MobileVLM V2 [12] MLLaMA-2.7B 336 144 1.2M 3.6M Shikra [8] Vicuna-13B 224 256 600k 5.5M IDEFICS-80B [21] LLaMA-65B 224 256 353M 1M Qwen-VL [4] Qwen-7B 448 256 1.4B 50M Qwen-VL-Chat [4] Qwen-7B 448 256 1.4B 50M|57.7 – – – 61.1 84.7 –<br>58.8 – – 77.4 – – – 54.5 – – 60.0 – – – 38.2 – 35.2 78.8 59.3 – – 60.6 – 38.9 78.2 57.5 – –<br><br><br>|
|---|---|
|LLaVA-1.5 [33] Vicuna-7B 336 576 558K 665K LLaVA-TokenPacker Vicuna-7B 336 144 558K 665K<br><br>|64.3 31.1 50.0 78.5 62.0 85.9 62.0<br><br>65.1↑0.8 33.0↑1.9 52.0↑2.0 77.9↓0.6 61.9↓0.1 87.0↑1.1 62.8<br><br><br>|
|LLaVA-1.5 [33] Vicuna-13B 336 576 558K 665K LLaVA-TokenPacker Vicuna-13B 336 144 558K 665K<br><br>|67.7 36.1 53.6 80.0 63.3 85.9 64.4<br><br>68.0↑0.3 34.5↓1.6 55.6↑2.0 78.9↓0.1 62.5↓0.8 87.4↑1.5 64.5<br><br><br>|

Fewer Tokens Setting

|InstructBLIP [13] Vicuna-7B 224 64 129M 1.2M InstructBLIP [13] Vicuna-13B 224 64 129M 1.2M LLaVA-TokenPacker Vicuna-7B 336 64 558K 665K LLaVA-TokenPacker Vicuna-13B 336 64 558K 665K<br><br>|36.0 26.2 – – – – – – 25.6 33.4 – 49.5 78.9 – 64.1 31.7 50.7 77.2 61.1 86.3 61.9 66.2 34.2 52.9 78.1 62.0 87.3 63.5<br><br>|
|---|---|
|LLaVA-PruMerge [46] Vicuna-7B 336 ~32 558K 665K LLaVA-PruMerge [46] Vicuna-13B 336 ~32 558K 665K LLaVA-TokenPacker Vicuna-7B 336 36 558K 665K LLaVA-TokenPacker Vicuna-13B 336 36 558K 665K<br><br>|60.9 – – 72.0 – 86.3 – 62.3 – – 72.8 – 86.2 – 62.8 29.6 50.2 75.0 59.6 86.2 60.6 66.2 34.1 53.9 76.3 60.7 86.5 63.0<br><br>|

- Table 2: Performance comparisons with high-resolution approaches on nine multimodal benchmarks. The token number of our method is the average statistically across all training and test data. †, ♯ and § denote s = 2,3,4 in TokenPacker, respectively. The best results are bold and the second-best results are underlined. * denotes the results obtained through the officially public protocols and checkpoints.

Method LLM #Data Max Res. #Token VQAT OCRB DocVQA MMB MMMU MME VQAv2 VizWiz POPE

|OtterHD [26] Fuyu-8B [5] - 1024×1024 – SPHINX-2k [32] LLaMA-13B 1.0B 762×762 2890 UReader [56] LLaMA-13B 86M 896×1120 – Monkey [30] QWen-7B 1.0B 896×1344 1792 TextHawk [59] InternLM-7B 115M 1344×1344 – LLaVA-UHD [55] Vicuna-13B 1.2M 672×1008 – LLaVA-NeXT [34] Vicuna-7B 1.3M 672×672 2880 LLaVA-NeXT [34] Vicuna-13B 1.3M 672×672 2880 Mini-Genimi-HD [28] Vicuna-7B 2.7M 1536×1536 2880 Mini-Genimi-HD [28] Vicuna-13B 2.7M 1536×1536 2880|– – – 61.2 – – 57.6 – 65.4<br><br>– 514 –<br><br>– – 76.4 67.7 – – 64.9 – –<br><br><br>67.1 – –<br>68.4 456* 65.0* 70.2 501* 70.0*<br><br><br>|58.3 – 1294/– 65.9 – 1471/–<br><br>– – –<br>– – – 74.6 – 1500/– 68.0 – 1535/–<br><br><br>67.4 35.8 1519/332 70.0 36.2 1575/326 65.8 36.8 1546/319<br><br>68.6 37.3 1575/326<br>|– –<br><br>80.7 44.9<br><br>– –<br><br>80.3 61.2<br><br>– –<br><br>81.7 56.1<br><br><br>81.8 57.6<br>82.8 60.5<br><br><br><br><br>80.3* 54.6*<br>81.5* 57.2*<br>|86.0 87.2 – 67.6<br><br>–<br><br>89.1 86.5 86.2<br><br>86.8*<br>87.0*<br>|
|---|---|---|---|---|
|LLaVA-TokenPacker-HD Vicuna-7B 2.7M 1088×1088 ~954† LLaVA-TokenPacker-HD Vicuna-13B 2.7M 1088×1088 ~954† LLaVA-TokenPacker-HD Vicuna-13B 2.7M 1344×1344 ~1393† LLaVA-TokenPacker-HD Vicuna-13B 2.7M 1344×1344 ~619♯ LLaVA-TokenPacker-HD Vicuna-13B 2.7M 1344×1344 ~347§<br><br>|68.0 452 60.2<br><br>69.3 498 63.0<br><br>70.6 521 70.0 68.8 470 63.0 68.4 447 58.0<br><br><br>|67.4 35.4 1489/338 69.5 38.8 1595/356<br><br>68.7 37.4 1574/350<br><br>69.9 38.2 1577/353 68.3 36.9 1577/332<br><br><br>|81.2 54.7<br><br>82.0 59.2 81.7 57.0 81.7 61.0 81.2 58.1<br><br><br>|88.2 88.1 88.0 87.6 88.0<br><br>|

visual projector achieves a reduction of visual tokens by 75% (from 576 to 144), while enhancing performance metrics by +0.8%/+0.3% on MMBench, +2.0% on both measures for VizWiz, and +1.1%/+1.5% on POPE with the Vicuna-7B/13B LLMs, respectively. Although a marginal decline in performance on image question answering benchmarks such as VQAv2 and GQA, our methods still comprehensively brings the average performance gains over LLaVA-1.5 [33], +0.8% and +0.1% respectively using Vicuna-7B and Vicuna-13B models with around 5 times TPS (4.9 vs. 24.9, see

- Table 3 for the details). Additionally, our approach exceeds the previous methods, like Qwen-VLChat [4], InstructBLIP [13] and MobileVLM V2 [12] across most of benchmarks, regardless of their access to more substantial training data.

Furthermore, we compare our method against previous leading approaches with fewer visual tokens. Specially, we set the token number to 64 (11% of 576) and 36 (6% of 576), respectively. It can be seen that our method surpasses these methods across three benchmarks at a large margin. For example, we observe that +3.9% on MMBench, +3.5% on VQAv2 are achieved against recent LLaVAPruMerge [46] with Vicuna-13B. These results affirm the efficacy of our TokenPacker, underscoring its advantageous impact on enhancing visual token representation and overall performance.

High Resolution. We apply our methods including TokenPacker and dynamic image slicing scheme into LLaVA-1.5 (dubbed LLaVA-TokenPacker-HD) to perform the high-resolution image understanding. For model training, the 2.7M data organized in Mini-Gemini [28] are employed. We set Ng = 9 and Ng = 16 to support the maximum input resolution with 1088×1088 and 1344×1344, respectively. The down-sampling ratio s is set to 2,3 or 4 to control the quantity of visual tokens derived

Table 3: Evaluation results on different visual projectors. We adopt token per second (TPS) to evaluate the throughput of LLM during inference, measured by a single NVIDIA A100 GPU.

Projector #Token TPS MMB MM-Vet VQAv2 GQA POPE VizWiz Avg.

|MLP [33]<br><br>|576 4.9|64.3 31.1 78.5 62.0 85.9 50.0|62.0|
|---|---|---|---|
|Average-Pooling Resampler [4] C-Abstractor [7] Pixel-Shuffle [9] LDP-v2 [12] Ours<br><br>|144 27.7<br><br>144 24.8<br><br>144 24.1<br>144 25.2<br><br><br>144 25.1 144 24.9<br><br><br>|64.3 26.7 76.4 60.3 86.4 51.3 63.1 29.2 75.1 58.4 84.7 51.9<br><br>63.1 29.4 74.6 59.2 84.6 49.2<br>64.0 29.7 76.2 60.1 85.9 48.8 66.2 28.7 77.3 61.1 86.1 47.6<br><br>65.1 33.0 77.9 61.8 87.0 52.0<br><br><br>|60.9 60.4 60.0 60.8 61.2 62.8<br><br>|
|Average-Pooling Resampler [4] C-Abstractor [7] Pixel-Shuffle [9] LDP-v2 [12] Ours<br><br>|64 29.2<br><br>64 26.6<br><br>64 26.5<br>64 27.7<br><br><br>64 27.1 64 27.1<br><br><br>|62.4 27.1 72.6 58.8 85.4 48.0<br>63.4 29.2 74.1 57.7 83.4 53.0<br><br><br>62.5 29.0 74.4 59.3 85.0 45.6<br>63.2 28.5 74.6 59.1 85.2 47.4<br><br><br>63.7 30.0 75.3 59.7 85.5 49.3<br><br>64.1 31.7 77.2 61.1 86.3 50.7<br><br><br>|59.1 60.1 59.3 59.7 60.6 61.9<br><br>|

from each image patch. We compare our approach against the existing high-resolution MLLM methods, such as OtterHD [26], SPHINX-2k [32], Monkey [30], document-oriented UReader [56] and TextHawk [59], and the more recent LLaVA-UHD [55], LLaVA-NeXT [34], Mini-Gemini-HD [28]. Table 2 reports the comparison results on nine popular benchmarks, including OCR-related VQAT, OCRB and DocVQA, and comprehensive MMB, MMMU and MME, and general VQA-related VQAv2, VizWiz and POPE benchmarks. It can be seen that our method with Vicuna-13B as LLM achieves the state-of-the-art OCR-realted performance of 70.6% on VQAT and 521 on OCRBench, when the input resolution is set to 1344×1344 with approximately 1393 visual tokens. These promising results can be attributed to the fact that the high-resolution images facilitate MLLM with more visual tokens to precisely recognize intricate fine-grained optical characters or objects. However, for comprehensive benchmarks like MMMU and MME, our approach exhibits the best performance

- at a lower resolutions with 1088×1088. Besides, even with approximately 619 visual tokens, our method obtains the second-best MMMU, MME and VizWiz scores with 38.2%, 1577/353, and 61.0%, respectively. This results demonstrate that MLLM with a reduced number of tokens still deliver robust performance on comprehensive benchmarks and VQA-related tasks. These results demonstrate the pivotal effectiveness of leveraging native high-resolution imagery in diverse multimodal tasks, and highlight the efficacy of our proposed TokenPacker. Figure 4 shows the qualitative comparisons across the representative scenarios.

#### 4.4 Ablation Results

We further dive into in-depth ablation studies to analyze the effectiveness on each component of our approach. All ablation experiments are conducted by employing Vicuna-7B as LLM and the data as in LLaVA-1.5 [33] for model training.

Various Visual Projectors. We first compare our proposed TokenPacker against various previous visual projectors, including direct Average-Pooling, Resampler [4], C-Abstractor [7], Pixel-Shuffle [9] and the recent LDP-v2 [12] on the top of LLaVA-1.5. For the Average-Pooling approach, we directly interpolate the feature map from visual encoder using average pooling, and then use the MLP to generate visual tokens. We replace the original MLP in LLaVA-1.5 with various projectors and keep the same settings to facilitate a fair comparison. To reflect the inference speed, we adopt the token per second (TPS) metric to evaluate the throughput of LLM. Table 3 reports the comparison results. In comparison to the MLP projector, all other visual projectors effectively reduce the number of visual tokens with the significant improvement on inference speed (around 5 vs. 25 TPS). AveragePooling achieves the best inference speed with fewer parameters. Our visual projector achieves +0.8% average performance gain with 144 tokens against the MLP with 576 tokens. Especially on the MM-Vet [58], POPE [29] and VizWiz [18] benchmarks, our method outperforms MLP-based method by +1.9%, +1.1% and +2.0%, respectively. Comparing to other methods, our approach surpasses the previous best method LDP-v2 [12] by +1.6%. In the scenario with 64 tokens, our approach attains a 61.9% average performance, on par with MLP-based method (61.9% vs. 62.0%) and surpasses LDP-v2 [12] by +1.3%. These results demonstrate the effectiveness of TokenPacker compared to previous approaches.

[Figure 26]

[Figure 27]

5.2s

###### LLaVA-TokenPacker (Ours)

The image depicts a group of people gathered in a living room, watching a bowling game on television. There are at least six people in the room, with some sitting on chairs and others standing. The television is located on the right side of the room, capturing the attention of the people. The living room is furnished with a couch and a dining table, which has a cake on it. The cake is placed near the center of the room, and there are several books scattered around the area. A remote control is also visible, likely used to operate the television. A few cups can also be seen, with one close to the cake and another on the left side of the room. A bottle is placed on the left side of the room, possibly containing a beverage for the guests.

[Figure 28]

13.8s

###### LLaVA-1.5

The image depicts a group of people gathered in a living room, enjoying a game of bowling on the Nintendo Wii. There are five people in the room, with some standing and others sitting on a couch. They are all focused on the television, which is displaying the bowling game. The living room is furnished with a dining table and a couch, providing a comfortable space for the group to socialize and play games. There are also several books scattered around the room, indicating that the residents might enjoy reading as well. A few cups can be seen on the table, suggesting that the group may be enjoying some beverages while they play. In the background, there is a refrigerator, and a bowl is placed on the dining table, adding to the cozy atmosphere of the living room.

[Figure 29]

Describe the image in detail.

|[Figure 30]<br><br>| |
|---|
|
|---|

|[Figure 31]<br><br>[Figure 32]|
|---|

[Figure 33]

What’s the title of the page?

[Figure 34]

LLaVA-TokenPacker-HD (Ours) 2.4s

The title of the page is "Providing Value to Customers & Consumers."

[Figure 35]

###### 7.5s

LLaVA-1.5

The title of the page is "Providing Value to Customers."

[Figure 36]

What is the brand name of the ITC personal care product advertised here?

|[Figure 37]<br><br>[Figure 38]|
|---|

[Figure 39]

LLaVA-TokenPacker-HD (Ours) 0.3s Vivel.

[Figure 40]

###### LLaVA-1.5 0.3s Tangel.

- Figure 4: Qualitative comparisons for representative scenarios. Our approach (144 tokens) achieves to handle the content details correctly and facilitates efficient image understanding. Moreover, our high-resolution method is able to capture the finer elements compared to original LLaVA-1.5.

Different Image Slicing Schemes. We then compare our dynamic image slicing scheme with the existing approaches for highresolution image. Here we list two typical methods in previous works. The first one is presented in LLaVA-1.5-HD [33]. It first resizes the original image into a fixed larger resolution (e.g. 672×672 ), then divides the image into smaller image patches. For the sake of brevity, we refer to this approach as “FixedSplit”. The second one is the shape-adaptive cropping scheme presented in UReader [56]. This module also considers to preserve the resolution of the image and the cropping grid fits the aspect ratio of the input image. Nevertheless, there still exists the resize operation in a small scale without considering the quantity of padding. We denote this method as “AdaptiveSplit” for clarity. To facilitate a fair comparison, we re-implement both methods by adopting our TokenPacker as visual projector. As shown in Table 4, our proposed dynamic image slicing scheme outperforms the previous methods on most of benchmarks. In particular, as for OCR-related benchmarks such as VQAT, OCRBench, the ratio-preserving approaches including AdaptiveSplit [56] and our method, surpass FixedSplit by +1.0%/+1.6% and +5/+9, respectively.

Table 4: Experimental results with different image splicing schemes.

|Method<br><br>|Res.|VQAv2 GQA VQAT OCRB|
|---|---|---|
|FixedSplit [33] AdaptiveSplit [56] Ours<br><br>|672 Any Any<br><br>|79.5 63.4 62.4 327<br>79.6 62.8 63.4 332 79.9 63.2 64.0 336<br><br><br>|

Component-wise analysis. Table 5 reports the component-wise experimental results of our method. Firstly, we directly employ the 2× downsampling feature map from visual encoder as low-resolution visual embeddings to feed MLP projector, yielding 144 visual tokens. We set this as the baseline method that achieves 76.4%, 60.3% and 55.3% on VQAv2, GQA and VQAT benchmarks, respectively. We then add (+) our injection module, which infuses high-resolution characteristic into the lowresolution query to be improved. The injection module obtains +1.1%, +1.3% and +1.2% performance gains over the baseline method, respectively. Subsequently, when we change (c) the query from low-

Table 5: Component-wise ablation results.

|Method|#Token|VQAv2 GQA VQAT<br><br>|
|---|---|---|
|Baseline<br><br>+ Injection c Learnable Query<br><br>+ Multi-level Feature<br><br>+ Image Partition<br><br>– Separator Token|144 144 144 144 – –<br><br>|76.4 60.3 55.3<br><br>77.5↑1.1 61.6↑1.3 56.5↑1.2<br><br><br>76.1↓1.4 59.8↓1.8 55.2↓1.3<br><br>77.9↑0.4 61.9↑0.3 57.2↑0.7 79.9↑2.0 63.2↑1.3 64.0↑6.8 76.6↓3.3 61.1↓2.1 58.3↓5.7<br><br><br>|

resolution feature map to a learnable query, the performance decreases with -1.4%, -1.8%, and -1.3%. The results demonstrate that the downsampled low-resolution feature map provides a foundation for absorbing finer high-resolution features. We further employ multi-level visual features as the comprehensive reference keys and values instead of a single-level feature in the injection module. This brings +0.4%, +0.3% and +0.7% improvements, respectively. Finally, we add our image slicing scheme for high-resolution image understanding, and the model obtains +2.0%, +1.3% and +6.8% improvements. When we remove (–) the separator token, i.e. comma (‘,’) and newline (‘\n’), the results show the performance drops with -3.3%, -2.1% and -5.7%. These results demonstrate the vital role to perverse the 2D image structure information in image slicing scheme.

### 5 Conclusion and Limitation

In this work, we proposed a novel visual projector, namely TokenPacker, for MLLM. Our method followed a coarse-to-fine design, which effectively condensed the enriched high-resolution image features to compact visual tokens. As an extension, we further presented an effective dynamic image partition scheme to perform efficient high-resolution image understanding. Extensive experiments have been conducted across diverse benchmarks to verify the effectiveness of our approach. Notably, our TokenPacker can effective reduce 75%∼89% visual tokens in LLaVA-1.5 and maintain comparable or even better performance with significantly higher efficiency.

Limitation. Our TokenPacker offers commendable performance by compressing visual tokens by up to 89%, yet it is not entirely without loss. Specifically, when reduced to 32 (6%) or fewer tokens, a clear decline in performance is evident. We are dedicated to progressing our research to develop more sophisticated visual projectors with very few tokens for efficient visual understanding with MLLM.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv:2308.12966, 2023.
- [5] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023.
- [6] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In ICLR, 2023.
- [7] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In CVPR, 2024.
- [8] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv:2306.15195, 2023.
- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024.

- [11] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv:2312.16886, 2023.
- [12] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.
- [13] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023.
- [14] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.
- [15] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512, 2024.
- [16] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. arXiv preprint arXiv:2404.06512, 2024.
- [17] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, pages 6904–6913, 2017.
- [18] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, 2018.
- [19] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914, 2023.
- [20] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, pages 6700–6709, 2019.
- [21] IDEFICS. Introducing idefics: An open reproduction of state-of-the-art visual language model. https: //huggingface.co/blog/idefics, 2023.
- [22] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv:2401.04088, 2024.
- [23] Dongsheng Jiang, Yuchen Liu, Songlin Liu, Xiaopeng Zhang, Jin Li, Hongkai Xiong, and Qi Tian. From clip to dino: Visual encoders shout in multi-modal large language models. 2023.
- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023.
- [25] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123:32–73, 2017.
- [26] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A highresolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023.
- [27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.
- [28] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024.
- [29] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023.
- [30] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In CVPR, 2024.
- [31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755, 2014.
- [32] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.

- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv:2310.03744, 2023.
- [34] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024.
- [35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.
- [36] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023.
- [37] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023.
- [38] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In CVPR, pages 11976–11986, 2022.
- [39] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [40] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, pages 2200–2209, 2021.
- [41] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.
- [42] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, pages 947–952, 2019.
- [43] OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV_System_Card.pdf, 2023.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [45] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [46] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024.
- [47] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.
- [48] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [49] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/InternLM, 2023.
- [50] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [51] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [52] Vicuna. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https://vicuna. lmsys.org/, 2023.
- [53] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023.
- [54] Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Vary: Scaling up the vision vocabulary for large vision-language models. arXiv preprint arXiv:2312.06109, 2023.
- [55] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024.

- [56] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126, 2023.
- [57] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [58] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024.
- [59] Ya-Qi Yu, Minghui Liao, Jihao Wu, Yongxin Liao, Xiaoyu Zheng, and Wei Zeng. Texthawk: Exploring efficient fine-grained perception of multimodal large language models. arXiv preprint arXiv:2404.09204, 2024.
- [60] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In CVPR, 2024.
- [61] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.
- [62] Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601, 2024.
- [63] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2023.
- [64] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024.

### Supplemental Material

In this part, we further provide additional experimental results and more discussions on our approach. The supplementary material is organized as follows:

- • §A: additional experimental results;
- • §B: broader impacts;
- • §C: asset license and consent.

### A Additional Experimental Results

#### A.1 More Ablation study

- Table A1: Ablation results on various single-level and multi-level features used in TokenPakcer.

|Single-level VQAv2 GQA VQAT|Multi-level VQAv2 GQA VQAT<br><br>|
|---|---|
|23 77.5 61.6 56.5 22 76.3 61.3 56.7 20 75.8 60.8 55.7 16 76.1 61.2 55.5|22-23 77.5 61.4 57.1 20-21-22-23 77.6 61.8 56.9 12-16-22-23 77.9 61.9 57.2<br><br>8-12-22-23 76.8 61.7 56.3|

In our TokenPacker, the injection module employs multi-level visual features as high-resolution reference keys and values to enhance the low-resolution query. To explore its effects and select the suitable combination of multi-level features, we conduct the evaluation experiments. Table A1 reports the comparisons results. One can see that our method adopting single-level feature from the 23rd layer yields superior average performance compared to other single-level methods. In contrast, features from shallower levels exhibit relatively poorer performance. When using the combination of multi-level features from 12th, 16th, 22nd, and 23rd layers, our method achieves the best performance with 77.9%, 61.9% and 57.2% on VQAv2, GQA and VQAT benchmarks, respectively. These results highlight the fact that different layers of CLIP encoder display unique biases towards various patterns ranging from the shallow layer to deep layers. Consequently, an optimal mixture of multi-level features is capable of harnessing a wealth of information, clearly enhancing TokenPacker’s effectiveness.

A.2 Comparisons on Training Times

- Table A2: Training Times Analysis. Eight NVIDIA A100 GPUs are adopted within a same environment. The accuracy is averaged across six benchmarks (refer to Table 1 of main paper).

|Method|Token Pre-training (PT) Instruction-Tuning (IT)|Avg. Acc.<br><br>|
|---|---|---|
|LLaVA-1.5 [33] LLaVA-TokenPacker LLaVA-TokenPacker LLaVA-TokenPacker|576 3.5h 10h 144 1h 7.5h<br><br>64 0.7h 6.5h 36 0.5h 6h|62.0 62.8 61.9 60.6|

To illustrate the efficiency of MLLM through our TokenPacker, we further conduct a thorough analysis on training time. We set our TokenPacker with various token quantities and make a comparison against the original LLaVA-1.5 [33]. Table A2 reports the evaluation results with Vicuna-7B model. As outlined in Table A2, the evaluation results clearly indicate that our approach consistently requires shorter training times with a fewer number of visual tokens in comparison to LLaVA-1.5. Specifically, utilizing 36 visual tokens, our approach achieves pre-training and instruction tuning in only 0.5 hours and 6 hours, respectively. These results verify our method’s superiority in facilitating efficient MLLM advancements.

#### A.3 More Visual Results

To verify the effectiveness of our approach for visual comprehension in practical real-world scenarios, we have put it to the test across diverse tasks involving understanding and reasoning, as illustrated in Figure A1. Leveraging the capabilities of our TokenPacker combined with a dynamic image slicing scheme for high-resolution image, our method adeptly handles various intricate situations,

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

What is SCGP?

What is the full form of AHA?

Up to which year the graph is plotted in 'Distribution of ValueAdded' graph?

| |
|---|

###### DocumentVQA

[Figure 50]

[Figure 51]

[Figure 52]

Shell Coal Gasification Process.

American heart association.

2012.

LLaVA-TokenPacker-HD

LLaVA-TokenPacker-HD

LLaVA-TokenPacker-HD

[Figure 53]

[Figure 54]

[Figure 55]

What is the answer to the bottom left equation?

How many gallons of supreme gasoline can I get with $50?

What is the full form of AHA?

[Figure 56]

[Figure 57]

To determine the number of gallons of supreme gasoline you can get with $50, we need to know the price per gallon. In the image, the price of supreme gasoline is displayed as $3.65 per gallon. If you have $50, you can calculate the number of gallons you can purchase by dividing the amount by the price per gallon: $50 / $3.65 = 13.7 gallons So, with $50, you can get approximately 13.7 gallons of supreme gasoline.

[Figure 58]

###### Math&CountingOCR&Recognition

[Figure 59]

There are six birds in the image.

The answer to the bottom left equation is 12.

LLaVA-TokenPacker-HD

LLaVA-TokenPacker-HD

###### LLaVA-TokenPacker-HD

[Figure 60]

[Figure 61]

What is located to the right of the shampoo?

What should be drawn in the fourth step?

[Figure 62]

What is the price for tomatos?

[Figure 63]

[Figure 64]

[Figure 65]

| |
|---|

In the fourth step of the Multistep Portrait Temporal Directions activity, the child should draw a nose on the face.

A conditioner is located to the right of the shampoo.

The price for tomatoes is 8 euros per kilogram.

LLaVA-TokenPacker-HD

###### LLaVA-TokenPacker-HD

LLaVA-TokenPacker-HD

[Figure 66]

[Figure 67]

Give me a short introduction of the painting.

What earth’s interior structure does number 2 indicate?

The painting is a beautiful and vibrant depiction of a night scene with a large tree in the foreground and a town in the background. The night sky is filled with stars, creating a captivating and serene atmosphere. The painting is reminiscent of Vincent van Gogh's famous "Starry Night" painting, with its use of swirling brushstrokes and vivid colors. The combination of the tree, town, and stars creates a harmonious and visually striking composition that

[Figure 68]

Number 2 on the diagram indicates the Earth's mantle. The mantle is the layer of the Earth's interior that lies between the crust and the core. It is primarily composed of solid rock and is responsible for the Earth's tectonic activity, including the movement of tectonic plates and the formation of volcanoes.

[Figure 69]

###### Knowledge

LLaVA-TokenPacker-HD captures the beauty of the night sky.

LLaVA-TokenPacker-HD

Figure A1: Qualitative results across various visual understanding scenarios with our approach.

including document VQA, Math&Counting, OCR recognition and other tasks that require specialized knowledge.

### B Broader Impacts

This work presents an effective visual projector for efficient MLLM. We have demonstrated its effectiveness over various multimodal benchmarks. On the positive side, our approach has the potential to benefit the efficient MLLM of real-world image or video understanding, which can

clearly reduce the training and inference costs while maintain the competitive performance. On the other side, due to the issue on the robustness of LMMs, some erroneous responses may raise the misinformation or safety issues of human beings. In order to avoid the potentially negative effects, we suggest to adopt a highly stringent security protocol in case that our approach fails to function properly in real-world multimodal applications.

### C Asset License and Consent

We utilize 558K image-caption pairs from the LLaVA-filtered CC3M dataset for pretraining and 665K mixture instruction following data for instruction tuning, which are all publicly and freely available for academic research [33]. The 558K pre-training data is the subset of CC3M with BLIP captions [27], which comply with license of CC-3M (https://ai.google.com/research/ConceptualCaptions/) and license of BLIP(https://github.com/salesforce/BLIP). The 665K mixture insturtion-following data includes publicly available COCO [31], GQA [20], OCR-VQA [42], TextVQA [47] and Visual Genome [25] as the data sources, which is released under the CC BY 4.0. And the GPT-generated multimodal instruction-following data must should abide by the policy (https://openai.com/policies/terms-of-use) of OpenAI. The 2.7M data organized in MiniGenimi [64] is released under the CC BY NC 4.0. We implement all methods with LLaVA (https://https://github.com/haotian-liu/LLaVA) codebase, which are released under the Apache-2.0 license.

