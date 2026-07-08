### Neighboring Autoregressive Modeling for Efficient Visual Generation

Yefei He1,2∗† Yuanyu He1∗ Shaoxuan He1∗ Feng Chen3∗ Hong Zhou1‡ Kaipeng Zhang2‡ Bohan Zhuang1‡

1Zhejiang University, China 2Shanghai AI Laboratory, China 3The University of Adelaide, Australia ∗ Equal contribution † Work done during an internship at Shanghai AI Laboratory ‡ Corresponding authors

# arXiv:2503.10696v1[cs.CV]12Mar2025

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

Figure 1. Generated samples from NAR. Results are shown for 512 × 512 text-guided image generation (1st row), 256 × 256 classconditional image generation (2nd row) and 128 × 128 class-conditional video generation (3rd row).

#### Abstract

able parallel prediction of multiple adjacent tokens in the spatial-temporal space, we introduce a set of dimensionoriented decoding heads, each predicting the next token along a mutually orthogonal dimension. During inference, all tokens adjacent to the decoded tokens are processed in parallel, substantially reducing the model forward steps for generation. Experiments on ImageNet 256 × 256 and UCF101 demonstrate that NAR achieves 2.4× and 8.6× higher throughput respectively, while obtaining superior FID/FVD scores for both image and video generation tasks compared to the PAR-4X approach. When evaluating on text-to-image generation benchmark GenEval, NAR with 0.8B parameters outperforms Chameleon-7B while using merely 0.4% of the training data. Code is available at https://github.com/ThisisBillhe/NAR.

Visual autoregressive models typically adhere to a rasterorder “next-token prediction” paradigm, which overlooks the spatial and temporal locality inherent in visual content. Specifically, visual tokens exhibit significantly stronger correlations with their spatially or temporally adjacent tokens compared to those that are distant. In this paper, we propose Neighboring Autoregressive Modeling (NAR), a novel paradigm that formulates autoregressive visual generation as a progressive outpainting procedure, following a near-to-far “next-neighbor prediction” mechanism. Starting from an initial token, the remaining tokens are decoded in ascending order of their Manhattan distance from the initial token in the spatial-temporal space, progressively expanding the boundary of the decoded region. To en-

#### 1. Introduction

FID

Large language models (LLMs) [34, 35, 49, 50] trained with an autoregressive “next-token prediction” objective have demonstrated unprecedented capabilities in addressing natural language-based tasks. Building on these advancements, many studies [7, 27, 44, 45, 57] have explored autoregressive models for visual generation or even unified multimodal generation. Typically, to adapt the “next-token prediction” paradigm for visual generation, images are tokenized into image tokens and flattened into one-dimensional visual token sequences in the raster order [13, 44, 52]. During inference, models must sequentially generate thousands of tokens to produce a single high-resolution image, resulting in significantly lower efficiency compared to diffusionbased models [32, 33].

DiT-L/2 Diameter

200 300 400 500 Model size / M

PAR-L

AR-L

VAR-d16

NAR-M

- 3

- 3.5
- 4

4.5

5

- 5.5

0 50 100 150 200 250 Throughput (img/s)

Recent efforts to accelerate autoregressive visual generation can be broadly classified into two categories. The first category parallelly generates multiple tokens in an image within one step. For instance, SJD [46] retains a rasterorder visual sequence and predicts multiple consecutive tokens in each forward step. However, prior studies [19, 47] have demonstrated that image tokens exhibit strong correlations with their spatially adjacent counterparts. Consequently, parallel decoding of consecutive tokens leads to insufficient context and notable performance degradation, particularly as the level of parallelism increases. Similarly, MAR [24] and PAR [58] generate multiple tokens in randomly selected positions or block-constrained regions, as shown in Figure 3(b)-(c). These methods require extensive hyperparameter tuning to balance parallelism and spatial coherence. The second category follows a “next-scale prediction” paradigm [25, 47], where token maps are autoregressively generated from coarse to fine scales, as illustrated in Figure 3(d). However, their token sequence is constructed by concatenating visual tokens from multiple scales. This requires specialized multi-scale image tokenizers and increases the overall length compared to single-scale visual sequences. As a result, it leads to higher memory overhead during both training and inference. To establish an efficient visual autoregressive modeling paradigm, we argue that it should satisfy the following criteria: 1) preserve spatial or temporal locality within visual features; 2) support parallel decoding during inference; 3) be compatible with regular single-scale tokenizers and maintain a short visual token sequence. Unfortunately, none have yet achieved all of these objectives simultaneously.

Figure 2. Generation quality and efficiency comparisons between various visual generation methods. Data is collected from ImageNet 256 × 256 dataset over models with parameters around 300M.

we frame autoregressive visual generation as a progressive outpainting process that begins from an initial token and is guided by a near-to-far ’next-neighbor prediction’ mechanism. As illustrated in Figure 3(e), our approach begins by arranging visual tokens in a locality-preserved order. Specifically, the Manhattan distance between visual tokens and the top left token in the spatial domain corresponds to their generation order. This arrangement poses a significant challenge for models based on the “next-token prediction” paradigm, as multiple equidistant tokens can exist in different dimensions, yet only one token can be generated at each step. To address this, we introduce dimension-oriented decoding heads, each responsible for predicting the next token along a distinct, mutually orthogonal dimension in the spatial space. During inference, multiple dimension-oriented decoding heads naturally support parallel decoding. Once a set of visual tokens has been decoded, all adjacent tokens can be generated in the subsequent step, thereby enabling the “next-neighbor prediction” paradigm and significantly improving generation efficiency. This approach can be seamlessly extended to video generation by employing three decoding heads and applying the same strategy, as demonstrated in Figure 3(f). Extensive experiments on image and video generation demonstrate that visual generation with the next-neighbor prediction paradigm can greatly improve the efficiency of visual generation while producing superior results compared to models with the next-token prediction paradigm. For class-conditional image generation on ImageNet 256×256, our method reduces the number of generation steps by 91.8% while lowering the FID by 0.81 over LlamaGen-XL model. For class-conditional video generation on UCF-101, our method achieves a com-

In this paper, we propose neighboring autoregressive modeling (NAR), a novel paradigm for high-fidelity and efficient visual generation. Figure 2 presents an overview of generation efficiency and quality comparisons among NAR and diverse visual generation methods. Drawing inspiration from image outpainting approaches [26, 38], which extend the content of an image beyond its original boundaries,

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

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

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

## …

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

(a) AR (b) MAR (c) PAR

(d) VAR

(e) NAR image generation (f) NAR video generation

[Figure 115]

[Figure 116]

[Figure 117]

- Step 1

- Step 2

- Step 3

0

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

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

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

Ungenerated tokens

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

frame 1

frame 2 frame 3

Figure 3. Comparisons of different autoregressive visual generation paradigm. The proposed NAR paradigm formulates the generation process as an outpainting procedure, progressively expanding the boundary of the decoded token region. This approach effectively preserves locality, as all tokens near the starting point are consistently decoded before the current token.

petitive FVD of 71.1 with a 97.3% generation step reduction compared to the vanilla next-token AR methods. When evaluating text-guided image generation on the GenEval benchmark, NAR-0.8B with merely 6M training data outperforms Chameleon-7B with 1.4B training data, demonstrating the effectiveness of our approach in generating high-resolution, high-aesthetic images.

In summary, our contributions are as follows:

- • We propose neighboring autoregressive modeling (NAR), a new next-neighbor prediction paradigm that formulates autoregressive visual generation as a near-to-far outpainting process.
- • We introduce dimension-oriented decoding heads, each responsible for predicting the next token along distinct, mutually orthogonal dimensions in the spatial-temporal space. This innovation allows for the parallel generation of all adjacent tokens in one step, enhancing the generation efficiency for both images and videos.
- • Extensive experiments on class-conditional image generation, class-conditional video generation, and text-guided image generation demonstrate that our method achieves substantial efficiency gains and superior generation quality compared to the existing autoregressive visual generation methods.

#### 2. Related Work

##### 2.1. Efficient Autoregressive Visual Generation

Typically, autoregressive (AR) visual generation models trained with a next-token prediction objective, necessitate n2 sequential model forward passes to generate an image represented by n × n tokens, resulting in significantly low generation efficiency. This issue is even more pronounced for high-resolution image [27] or video generation [10, 57], hindering its wide deployment in real-world applications.

To address this inefficiency, techniques such as SJD [46] and ZipAR [19] have been proposed as training-free methods to accelerate sampling by predicting multiple consecutive or spatially adjacent tokens in a single step. However, these methods provide only limited speedup, as the pretrained AR models are designed solely to model the distribution of the next token in raster order. Another approach, PAR [58], divides image tokens into spatially distant subsets and employs the next-token prediction paradigm within each subset for parallel generation. Although PAR trains models from scratch, its performance may be suboptimal due to insufficient global context when generating multiple distant regions simultaneously. In contrast to the traditional next-token prediction paradigm, MAR [24] predicts multiple output tokens simultaneously in a random order. VAR [47] iteratively generates a coarse-to-fine multiresolution image token map using a next-scale prediction

[Figure 181]

[Figure 182]

[Figure 183]

paradigm. However, VAR relies on multi-scale image tokenizers and produces longer token sequences, complicating the process and increasing training overhead. In this paper, we propose an efficient neighboring autoregressive generation paradigm that retains the short token sequences and image tokenizers of vanilla next-token AR models while improving image quality and achieving significantly higher generation efficiency.

[Figure 184]

𝑥 ,  𝑥 ,  𝑥 ,  𝑥 , 

| |=| | |
|---|---|---|---|
| | | | |
| |+| | |
| | | | |

###### =

Output Tokens

Horizontal Head Vertical Head

|𝑥|𝑥 , |𝑥 , |
|---|---|---|
|𝑥 , |𝑥 , |[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>𝑥 , <br><br>[Figure 189]<br><br>[Figure 190]|
|𝑥 , |𝑥 , | |

[Figure 191]

- 2.2. Visual Locality in Computer Vision

The concept of visual locality refers to the similarity among adjacent pixels in terms of color, texture, and edges. This locality has been extensively investigated in early computer vision research, such as image compression [3, 53], image denoising [9, 48], and edge detection [4, 42]. Convolutional neural networks (CNNs) [17, 21, 23, 39] leverage this property by restricting interactions to local neighborhoods, which has enabled them to significantly outperform multi-layer perceptron (MLP) models across a wide range of computer vision tasks [18, 29, 36]. In contrast, the standard transformer model [59] utilizes a self-attention mechanism to model interactions between tokens, which inherently lacks a focus on local patterns. This limitation leads to suboptimal performance when such models are directly applied to image data [12]. To address this issue, several techniques have been proposed, including the integration of CNNs with self-attention models [2, 5, 6] and the implementation of self-attention within localized windows [28, 62]. These methods aim to simultaneously capture fine-grained visual locality and long-range dependencies. Despite these advancements, the application of spatial locality in autoregressive image generation remains unexplored. In this paper, we introduce a locality-preserved generation paradigm, which demonstrates superior efficiency and quality in visual generation tasks.

- 3. Method

[Figure 192]

[Figure 193]

𝑥 , 

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Transformer Block

𝑥 , 

Transformer Block × L

[Figure 201]

Token Embedding

[Figure 202]

Input Tokens

[Figure 203]

𝑥 ,  𝑥 ,  𝑥 , 

Figure 4. Illustration of the dimension-oriented decoding heads. The horizontal head and the vertical head are responsible for predicting the next token in the row and column dimensions, respectively. Here, L is the number of Transformer blocks in the backbone network.

ated per step. Although certain methods [19, 46, 58] enable parallel decoding within the NTP framework, their performance significantly degrades when predicting too many tokens simultaneously. As evidenced by the empirical results in Table 4, directly integrating the NAR paradigm with the next-token prediction objective leads to substantial performance degradation. We contend that this issue arises because the original model is trained exclusively to predict the conditional distribution of the next token in raster order, while the distribution of tokens across multiple locations to be predicted can vary substantially.

- 3.1. Neighboring Autoregressive Modeling

To address this, we propose dimension-oriented decoding heads, a simple yet effective approach to model the distinct conditional distributions of parallel decoded tokens. As presented in Figure 4, the extra decoding heads consist of a Transformer block and a fully-connected output layer, which are concatenated to the Transformer backbone to enable parallel decoding along several mutually orthogonal dimensions. For instance, an image can be considered two-dimensional, allowing decoding to proceed along two orthogonal dimensions: rows and columns. Consequently, we apply two dimension-oriented decoding heads for image generation models, with each head modeling the conditional distribution of the next token in one dimension, i.e., the next token in the same row and the token in the same column in the next row. Our method can also be naturally ex-

Motivated by image outpainting methods [26, 38], we propose a novel perspective on autoregressive visual generation. As shown in Figure 3(e)-(f), our approach frames the process as an outpainting procedure initiated from scratch, progressively expanding the boundary of the decoded token region. At each generation step, all adjacent tokens to the decoded tokens are predicted, leading to a neighboring autoregressive modeling (NAR) paradigm. This paradigm effectively preserves locality, as all tokens near the starting point are consistently decoded before the current token.

However, this configuration presents a significant challenge for models trained with the next-token prediction (NTP) objective, as multiple equidistant tokens may exist in different dimensions, yet only one token can be gener-

S1S2 S3 S1 S2 S3

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

S2S3S1

S2S3S1

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | |

| |
|---|
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |
| |

(a) Proximity-aware mask for image generation

(b) Proximity-aware mask for video generation

- Figure 5. Proximity-aware attention masks for the NAR paradigm. “Sn” denotes the n-th generation step. Tokens generated within the same step are represented by the same color. To maintain the autoregressive property, a causal mask is applied between tokens across different generation steps (aligned with Figure 3). Within each step, bidirectional attention is employed among the tokens to enhance consistency during parallel generation.

tended to video generation. Since videos can be regarded as three-dimensional, adding a temporal dimension to images, decoding can be performed along three orthogonal dimensions: times, rows, and columns. Such a decoupled design prevents a single head from predicting multiple mixed conditional distributions, thereby significantly improving the generation performance.

##### 3.2. Implementation Details of NAR

Inference with NAR. By employing a set of decoding heads to predict the next token along multiple dimensions, we can naturally predict all adjacent tokens of a generated token in one forward step. Specifically, the decoding process begins with the initial token positioned at the upper-left corner of the image feature map, aligning with the rasterscan order. In subsequent steps, the tokens generated in the previous step are input into the model to autoregressively generate new tokens, as illustrated in Figure 4. For each input token, its two adjacent tokens along the row and column dimensions are generated using dimension-oriented heads. Notably, starting from the third step, there are overlaps between the two predicted neighboring tokens for input tokens, e.g., x3,1 and x3,2 in Figure 4. In this case, we mix the predictions from different decoding heads for the overlapped tokens, analogous to a model ensemble approach [30]. Empirical results, presented in Table 5, demonstrate that this method consistently enhances generation performance compared to relying on the prediction of a single decoding head.

Formally, given the initial token x0, the set of all tokens

generated at step i can be defined as:

S = {xi | D(xi,x0) = i}, (1)

where D represents the Manhattan distance. With such a generation paradigm, NAR produces a high-resolution image with n×n tokens in 2n−1 steps. For videos represented by t × n × n tokens, NAR completes the generation process in only 2n + t − 2 steps, significantly fewer than the

- tn2 steps required by vanilla next-token AR models. Training with NAR. Training a NAR model shares the same image tokenizer and training pipeline as vanilla nexttoken AR models, requiring only minor modifications to the model architecture and the attention mask. Similar to previous AR image generation methods [44, 47], all decoding heads are trained with cross-entropy loss, predicting the next token along different dimensions. Since the decoding order of tokens is determined by their proximity
- to the initial token, tokens farther from the initial token depend only on those closer to it during decoding. Therefore, a proximity-aware causal attention mask is employed for training NAR models, as shown in Figure 5. Notably, bidirectional attention is applied among tokens equidistant from the initial token, ensuring better consistency during parallel generation.

Compared to the VAR paradigm, the proposed NAR paradigm substantially reduces training overhead. First, VAR necessitates a specialized multi-scale tokenizer, which requires extensive data for training to achieve optimal performance. In contrast, our approach can leverage a diverse range of high performance image tokenizers from the opensource community. Second, the token sequence of VAR is constructed by concatenating visual tokens from multiple scales, which is considerably longer than the single-scale visual token sequence used in NAR. This reduction in sequence length substantially decreases computational complexity during training, as the complexity scales quadratically with the sequence length.

#### 4. Experiments

##### 4.1. Experimental Setup

Model architecture. To validate the effectiveness and scalability of the proposed NAR paradigm, we adopt a decoder-only Transformers architecture, following prior studies [44, 54, 58]. By implementing dimension-oriented decoding heads, our NAR is inherently applicable to both two-dimensional image and three-dimensional video generation, utilizing two and three decoding heads, respectively.

Class-conditional image generation. We evaluate NAR on the widely adopted ImageNet 256×256 dataset. Images are tokenized using an off-the-shelf image tokenizer introduced by [44], with a downsampling factor of 16. All models are trained with 300 epochs with a base learning rate of 10−4

- Table 1. Quantitative evaluation on the ImageNet 256 × 256 benchmark. “Step” denotes the number of model forward passes required to generate an image. The throughput is measured with the maximum batch size supported on a single A100 GPU. Classifier-free guidance is set to 2 for our method. We also report the reconstruction FID (rFID) of visual tokenizers for each method, which serves as an upper bound for generation FID. †: model denoted as M shares the same hidden dimension as the L model but is reduced by 6 layers in depth.

Throughput (img/s)

Tokenizer Type Model Params FID↓ IS↑ Precision↑ Recall↑ Steps

VAR-d16 310M 3.30 274.4 0.84 0.51 10 129.3 VAR-d20 600M 2.57 302.6 0.83 0.56 10 77.6 VAR-d24 1.0B 2.09 312.9 0.82 0.59 10 50.6 VAR-d30 2.0B 1.92 323.1 0.82 0.59 10 29.9

VQVAE-108M (rFID=1.00)

VAR

LlamaGen-B 111M 5.46 193.6 0.84 0.46 256 117.9 LlamaGen-L 343M 3.80 248.3 0.83 0.52 256 47.1

AR

LlamaGen-XL 775M 3.39 227.1 0.81 0.54 256 23.7 LlamaGen-XXL 1.4B 3.09 253.6 0.83 0.53 256 14.1

PAR-B-4X 111M 6.21 204.4 0.86 0.39 67 174.1 PAR-L-4X 343M 4.32 189.4 0.87 0.43 67 93.8

VQVAE-72M (rFID=2.19)

PAR

PAR-XL-4X 775M 3.50 234.4 0.84 0.49 67 53.9 PAR-XXL-4X 1.4B 3.20 288.3 0.86 0.50 67 33.9

NAR-B 130M 4.65 212.3 0.83 0.47 31 419.7 NAR-M† 290M 3.27 257.5 0.82 0.53 31 248.5

NAR

NAR-L 372M 3.06 263.9 0.81 0.53 31 195.4 NAR-XL 816M 2.70 277.5 0.81 0.58 31 98.1

NAR-XXL 1.46B 2.58 293.5 0.82 0.57 31 56.9

and a step learning rate scheduler. The reported Inception Score (IS) and Frechet Inception Distance (FID) results are computed by sampling 50,000 images and evaluating them with ADM’s TensorFlow evaluation suite [11].

Class-conditional video generation. NAR is trained and evaluated on UCF-101 dataset [43]. We utilize a video tokenizer proposed by [54], which encodes a 16 × 128 × 128 video clip into 4×16×16 visual tokens. Models are trained with 3000 epochs with a base learning rate of 10−4 and step learning rate scheduler. Frechet Video Distance (FVD) [51] serves as the main evaluation metric for generation.

Text to image generation. We curate a dataset comprising 4M image-text pairs from the LAION-COCO [1] dataset and 2M open-sourced high-resolution images captioned using large vision-language models [56]. An image tokenizer fine-tuned on LAION-COCO, introduced by [44], is employed with a downsampling factor of 16. Text embeddings are extracted using the pretrained FLAN-T5 model [8], which serves as the conditional input for image generation. Following prior work [44], the training process is divided into two stages. In the first stage, the model is trained on the 4M LAION-COCO subset at a resolution of 256 × 256 for 60 epochs. In the second stage, the model is fine-tuned on the 2M high-quality dataset at a resolution of 512 × 512 for 40 epochs. A cosine-annealing learning rate scheduler is used for both training stages. GenEval [15] is adopted as a fair and fine-grained benchmark for comparison.

##### 4.2. Main Results

###### 4.2.1. Class-conditional Image Generation

In this subsection, we evaluate the performance of NAR models on ImageNet 256 × 256 dataset, as presented in Table 1. For a fair comparison, we adopt the same image tokenizer, model architectures, and training pipelines as LlamaGen [44] and PAR [58]. The employed image tokenizer has merely 72M parameters, which is exclusively trained on the ImageNet dataset and provides a reconstruction FID (rFID) of 2.19. Although the previous parallel decoding method, PAR, improves generation efficiency compared to LlamaGen’s standard next-token prediction paradigm, its FID remains consistently higher than that of LlamaGen at the same model size. In contrast, models employing the NAR paradigm exhibit superior performance and efficiency. For instance, NAR-L, with 372M parameters, achieves a lower FID than LlamaGen-XXL with 1.4B parameters (3.06 vs. 3.09), while reducing the number of model forward steps by 87.8% (31 steps vs. 256 steps) and delivering 13.8× higher throughput (195.4 images/s vs. 14.1 images/s).

On the other hand, the VAR approach [47] employs a larger image tokenizer with 108M parameters, which is trained on the large-scale OpenImages dataset [22]. This tokenizer provides a reconstruction FID (rFID) of 1.00 and offers a higher upper bound for generation performance. Despite this, NAR-M with fewer parameters achieves lower FID than VAR-d16 (3.27 vs. 3.30), while providing 1.92× higher throughput (248.5 images/s vs. 129.3 images/s).

- Table 2. Comparison of class-conditional video generation methods on UCF-101 benchmark. Classifier-free guidance is set to 1.25 for all variants of our method. †: model denoted as LP shares the same hidden dimension as the XL model but is reduced by 6 layers in depth.

Type Method Params FVD↓ Steps Time (s)

VideoFusion [31] N/A 173 - Make-A-Video [40] N/A 81.3 - HPDM-L [41] 725M 66.3 - -

Diffusion

MAGVIT [60] 306M 76 - MAGVIT-v2 [61] 840M 58 - -

Masking

CogVideo [20] 9.4B 626 - TATS [14] 321M 332 - OmniTokenizer [55] 650M 191 5120 336.70

MAGVIT-v2-AR [61] 840M 109 1280 -

AR

LARP-L-Long [54] 343M 102 1280 44.0

PAR-XL-1× [58] 792M 94.1 1280 43.30 PAR-XL-4× [58] 792M 99.5 323 11.27 PAR-XL-16× [58] 792M 103.4 95 3.44

NAR-L 369M 96.2 34 1.09 NAR-LP† 694M 71.1 34 1.30

Ours

Combining NAR with a more advanced image tokenizer will be left as future work.

###### 4.2.2. Class-conditional Video Generation

In this subsection, we evaluate the effectiveness of NAR on class-based video generation using UCF-101 [43]. As shown in Table 2, our NAR surpasses other autoregressive counterparts by significantly reducing generation steps and wall-clock time while achieving lower FVD. Compared to LARP-L-Long [54], which employs the same video tokenizer and has a comparable number of parameters, our NAR-L further enhances generation quality, reducing FVD by 5.8 and generation latency by 97.5%. Furthermore, compared to PAR [58], which is designed for parallel generation, our NAR-XL consistently outperforms it with a 23∼32.3 FVD reduction, all without requiring hyperparameter tuning. Overall, we enhance the scalability of the autoregressive paradigm for image/video generation, making it comparable to diffusion- [31, 40] and maskingbased [60, 61] approaches while using fewer parameters and incurring lower latency.

###### 4.2.3. Text to Image Generation

To verify the effectiveness of NAR in text-guided image generation, we train a text-guided NAR-XL model and evaluate its performance on the GenEval [15] benchmark, as detailed in Table 3. The NAR-XL model, trained on merely 6 million publicly available text-image pairs, outperforms LlamaGen-XL [44] on GenEval benchmark with a notable margin (0.43 vs. 0.32), despite utilizing just 10% of the training data and delivering 4.4× higher throughput. Moreover, the overall score of the NAR model surpasses that of Chameleon [45], an AR visual generation model with 7B parameters trained on 1.4B text-image pairs. Com-

pared to diffusion-based model SDv1.5 [37], NAR achieves comparable performance while using only 0.4% of the training data. These results underscore the capability of the NAR paradigm to generate high-quality images with minimal training data.

Quantitative visualizations demonstrating classconditional image generation, video generation, and text-guided image generation are provided in the supplementary material.

##### 4.3. Deployment Efficiency

In this subsection, we provide a detailed comparison of the efficiency of various autoregressive generation paradigms, evaluating the latency, memory usage, and throughput of models with similar FID scores. As illustrated in Figure 6(a), in terms of generation latency, VAR-d16 exhibits lower latency than both NAR-M and LlamaGen-L when the batch size is smaller than 32. This can be attributed to the memory bottleneck during the decoding process in the vanilla next-token AR and NAR paradigms. However, as the batch size increases, NAR-M achieves significantly lower latency. For instance, with a batch size of 256, NARM reduces latency by 44% compared to VAR-d16 (1.13s vs. 2.02s). Furthermore, as shown in Figure 6(b), NAR-M consistently requires less GPU memory than VAR-d16 for the same batch size, which is due to the shorter sequence length of NAR. Consequently, on the same hardware, NAR paradigm can accommodate a larger batch size, leading to greater throughput, as illustrated in Figure 6(c). For example, on an A100 GPU with 80GB of VRAM, VAR-d16 supports a maximum batch size of 256 during inference, yielding a throughput of 129.3 images per second. In contrast, NAR-M supports a batch size of 512, achieving a throughput of 248.5 images per second, which is 92.1% higher than that of VAR-d16. Compared to the vanilla next-token AR model LlamaGen-L, NAR-M provides 5.2× higher throughput while achieving a superior FID. These results highlight the efficiency advantages of the NAR paradigm in terms of latency, memory usage, and throughput, making it a compelling choice for high-performance and efficient image generation.

##### 4.4. Ablation Study

Effect of dimension-oriented decoding heads. We assess the effectiveness of the proposed dimension-oriented decoding heads by comparing their performance with a baseline method that employs a single head to predict all adjacent tokens in parallel. For a fair comparison, all models are trained with the same pipeline and hyper-parameters. As demonstrated in Table 4, NAR-L without dimensionoriented decoding heads significantly underperforms compared to LlamaGen-L with a next-token prediction objective, leading to a substantially higher FID of 66.31. This

Table 3. Quantitative evaluation on the GenEval benchmark. † denotes the results are reported by [16].

Throughput (img/s) LlamaGen-XL† [44] 0.8B 60M - 0.34 - - 0.32 3.40

Model Params Training Data Counting Two Object Single Object Colors Overall

Chameleon† [45] 7B 1.4B - - - - 0.39 0.09 SDv1.5 [37] 0.9B 2B 0.35 0.38 0.97 0.76 0.43 0.44 NAR-XL 0.8B 6M 0.34 0.37 0.98 0.76 0.43 15.0

60

250

LlamaGen-L (FID=3.80)

LlamaGen-L (FID=3.80)

LlamaGen-L (FID=3.80)

10

Throughput(imgs/s)

VAR-d16 (FID=3.30)

VAR-d16 (FID=3.30)

VAR-d16 (FID=3.30)

200

8

NAR-M (FID=3.27)

Memory(GB)

NAR-M (FID=3.27)

NAR-M (FID=3.27)

40

Latency(s)

150

6

100

4

20

2

50

0

0

0 100 200 300 400 500 Batch Size

0 100 200 300 400 500 Batch Size

0 100 200 300 400 500 Batch Size

- Figure 6. Efficiency comparisons between vanilla AR, VAR and the proposed NAR paradigm for visual generation. With a batch size larger than 64, NAR achieves a lower FID with lower latency, lower memory usage and significantly higher throughput.

performance degradation can be attributed to the single decoding head’s inability to adequately capture the distinct token distributions at different positions, which may vary considerably for spatially distant tokens. In contrast, NARL equipped with the proposed dimension-oriented decoding heads achieves a markedly lower FID of 3.06 while reducing the number of model forward steps by 87.8% (from 256 steps to 31 steps). This improvement highlights the efficacy of the dimension-oriented heads in enhancing generation quality and efficiency.

Table 4. The effect of dimension-oriented decoding heads evaluated on ImageNet 256 × 256 benchmark. Here, “NTP” denotes the next-token prediction paradigm.

Method Dimension-oriented Heads Steps FID↓

LlamaGen-L × 256 3.80 NAR-L × 31 66.31 NAR-L ✓ 31 3.06

Effect of mixed logits sampling. As illustrated in Figure 4, overlaps exist between neighboring tokens predicted by different dimension-oriented decoding heads. During inference, these overlapping tokens can be sampled either from the predictions of a single decoding head or from a combination of predictions across multiple decoding heads. We assess the impact of various decoding head configurations on the class-conditional ImageNet 256×256 benchmark, as presented in Table 5. The results demonstrate that combining predictions from multiple decoding heads significantly enhances generation quality compared to relying on a sin-

gle decoding head. This improvement is evidenced by a substantially lower FID of 3.06 and a higher IS of 263.9.

Table 5. Performance comparison of various decoding head configurations on ImageNet 256 × 256. Here, “Head” denotes the decoding heads to use for the overlapped region. Results are obtained with a classifier-free guidance of 2.0.

Model Head FID↓ IS↑

Horizontal 75.91 22.36 Vertical 25.73 96.08 Mixed 3.06 263.9

NAR-L

#### 5. Conclusion

In this paper, we have proposed Neighboring Autoregressive Modeling (NAR), a new ”next-neighbor prediction” paradigm for efficient and high-quality visual generation. To facilitate the parallel decoding of multiple equidistant tokens, we propose a set of dimension-oriented decoding heads, each responsible for predicting the next token along a mutually orthogonal dimension in the spatial-temporal space. During inference, our approach enables the prediction of all adjacent tokens of a generated token in a single forward step, significantly reducing the number of model forward steps for autoregressive visual generation. Extensive experimental results demonstrate that NAR achieves state-of-the-art generation quality and efficiency trade-off for both image and video generation tasks.

###### Limitations and future work. In this work, we employ a

moderate-size image tokenizer to ensure a fair comparison with existing methods [44, 58]. In the future, we anticipate integrating NAR with more advanced visual tokenizers to further enhance its performance. Moreover, while NAR demonstrates fast and high-quality video generation capabilities, our experiments are currently limited to the classconditional benchmark UCF-101. We anticipate that training NAR on large-scale video datasets will yield even better video generation results.

#### References

- [1] Laion-coco: 600m synthetic captions from laion-2b-en. https://laion.ai/blog/laion-coco/. 6
- [2] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. Attention augmented convolutional networks, 2020. 4
- [3] Thomas Boutell. Png (portable network graphics) specification version 1.0. Technical report, 1997. 4
- [4] John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 1986. 4
- [5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers, 2020. 4
- [6] Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. Pixelsnail: An improved autoregressive generative model, 2017. 4
- [7] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 2

- [8] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024. 6
- [9] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transformdomain collaborative filtering. IEEE Transactions on image processing, 16(8):2080–2095, 2007. 4
- [10] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169,

2024. 3

- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 6
- [12] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 4
- [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2

- [14] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In European Conference on Computer Vision, pages 102–118. Springer, 2022. 7
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 6, 7
- [16] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024. 8
- [17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. arxiv e-prints. arXiv preprint arXiv:1512.03385, 10, 2015. 4
- [18] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017. 4
- [19] Yefei He, Feng Chen, Yuanyu He, Shaoxuan He, Hong Zhou, Kaipeng Zhang, and Bohan Zhuang. Zipar: Accelerating autoregressive image generation through spatial locality. arXiv preprint arXiv:2412.04062, 2024. 2, 3, 4
- [20] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 7
- [21] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. NeurIPS, 2012. 4
- [22] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):1956–1981, 2020. 6
- [23] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541–551, 1989. 4
- [24] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024. 2, 3
- [25] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, and Bhiksha Raj. Controlvar: Exploring controllable visual autoregressive modeling. arXiv preprint arXiv:2406.09750, 2024. 2
- [26] Han Lin, Maurice Pagnucco, and Yang Song. Edge guided progressively generative image outpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 806–815, 2021. 2, 4
- [27] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657, 2024. 2, 3

- [28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows, 2021. 4
- [29] Jonathan Long, Evan Shelhamer, Trevor Darrell, and UC Berkeley. Fully convolutional networks for semantic segmentation. arxiv 2015. arXiv preprint arXiv:1411.4038,

2014. 4

- [30] Jinliang Lu, Ziliang Pang, Min Xiao, Yaochen Zhu, Rui Xia, and Jiajun Zhang. Merge, ensemble, and cooperate! a survey on collaborative strategies in the era of large language models. arXiv preprint arXiv:2407.06089, 2024. 5
- [31] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023. 7
- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2

- [33] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [34] Alec Radford. Improving language understanding by generative pre-training. 2018. 2
- [35] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 2
- [36] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: unified, real-time object detection (2015). arXiv preprint arXiv:1506.02640, 825, 2015. 4
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 7, 8
- [38] Mark Sabini and Gili Rusak. Painting outside the box: Image outpainting with gans. arXiv preprint arXiv:1808.08483,

2018. 2, 4

- [39] Karen Simonyan. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 4
- [40] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 7

- [41] Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, and Sergey Tulyakov. Hierarchical patch diffusion models for high-resolution video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7569–7579, 2024. 7
- [42] I. Sobel and G. Feldman. A 3x3 Isotropic Gradient Operator for Image Processing, pages 271–272. 1973. 4

- [43] K Soomro. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402,

2012. 6, 7

- [44] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2, 5, 6, 7, 8, 9
- [45] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2, 7, 8
- [46] Yao Teng, Han Shi, Xian Liu, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Accelerating autoregressive text-to-image generation with training-free speculative jacobi decoding. arXiv preprint arXiv:2410.01699,

2024. 2, 3, 4

- [47] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. 2024. 2, 3, 5, 6
- [48] Carlo Tomasi and Roberto Manduchi. Bilateral filtering for gray and color images. In Sixth international conference on computer vision (IEEE Cat. No. 98CH36271), pages 839–

846. IEEE, 1998. 4

- [49] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2
- [51] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [52] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 2
- [53] Gregory K Wallace. The jpeg still picture compression standard. Communications of the ACM, 34(4):30–44, 1991. 4
- [54] Hanyu Wang, Saksham Suri, Yixuan Ren, Hao Chen, and Abhinav Shrivastava. Larp: Tokenizing videos with a learned autoregressive generative prior. arXiv preprint arXiv:2410.21264, 2024. 5, 6, 7
- [55] Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. arXiv preprint arXiv:2406.09399, 2024. 7
- [56] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6
- [57] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang,

- Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2, 3
- [58] Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, and Xihui Liu. Parallelized autoregressive visual generation. arXiv preprint arXiv:2412.15119, 2024. 2, 3, 4, 5, 6, 7, 9
- [59] A Waswani, N Shazeer, N Parmar, J Uszkoreit, L Jones, A Gomez, L Kaiser, and I Polosukhin. Attention is all you need. In NIPS, 2017. 4
- [60] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 7
- [61] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 7
- [62] Zizhao Zhang, Han Zhang, Long Zhao, Ting Chen, Sercan O¨ Arik, and Tomas Pfister. Nested hierarchical transformer: Towards accurate, data-efficient and interpretable visual understanding. In AAAI, 2022. 4

#### A. More Visualizations

### Appendix

[Figure 204]

[Figure 205]

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

[Figure 238]

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

Figure A. Video generation samples on UCF-101 dataset. Each row shows sampled frames from a 16-frame, 128 × 128 resolution sequence generated by NAR-XL across various action categories.

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

class id 284, siamese cat class id 973, coral reef

[Figure 288]

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

class id 980, volcano class id 387, lesser panda

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

[Figure 311]

class id 979, valley class id 2, great white shark

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

class id 985, daisy class id 974, geyser

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

class id 933, cheeseburger class id 928, ice cream

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

class id 780, schooner class id 437, beacon

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

class id 90, lorikeet class id 250, Siberian husky

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

class id 562, fountain class id 972, cliff

Prompt: A cozy cabin nestled in a snowy forest with smoke rising from the chimney.

Prompt: A bustling downtown street in Tokyo at night, with neon signs, sidewalks, and skyscrapers.

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Prompt: A bare kitchen has wood cabinets and white appliances.

Prompt: A magical fairy tale castle on a hilltop surrounded by a mystical forest.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

Prompt: A mountain lake at sunrise, with mist rising off, and snow-capped peaks in the background.

Prompt: A large pizza is in a cardboard box.

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

Prompt: a big purple bus parked in a parking spot. Prompt: A snowy scene of trees and a road.

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

LlamaGen (256 steps) NAR (31 steps)

LlamaGen (256 steps) NAR (31 steps)

- Figure D. 256×256 text-guided image generation samples produced by LlamaGen-XL-Stage1 with next-token prediction paradigm and

- NAR-XL-Stage1 with next-neighbor prediction paradigm.

Prompt: a small house on a mountain top. Prompt: a long-island ice tea cocktail.

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Prompt: a thumbnail image of a person skiing. Prompt: a sketch of a skyscraper.

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

Prompt: A cityscape at night with a full moon. Prompt: A wood cabin with a fire pit in front of it.

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

Prompt: A steam locomotive speeding through a desert.

Prompt: A crowd of people watching fireworks by a park.

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

LlamaGen (1024 steps) NAR (63 steps)

LlamaGen (1024 steps) NAR (63 steps)

- Figure E. 512×512 text-guided image generation samples produced by LlamaGen-XL-Stage2 with next-token prediction paradigm and

- NAR-XL-Stage2 with next-neighbor prediction paradigm. The text prompts are sampled from Parti prompts.

