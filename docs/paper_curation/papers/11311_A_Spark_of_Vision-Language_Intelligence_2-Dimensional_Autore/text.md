## A SPARK OF VISION-LANGUAGE INTELLIGENCE: 2-DIMENSIONAL AUTOREGRESSIVE TRANSFORMER FOR EFFICIENT FINEGRAINED IMAGE GENERATION

Liang Chen1, Sinan Tan2, Zefan Cai3, Weichu Xie4, Haozhe Zhao1, Yichi Zhang1 Junyang Lin2, Jinze Bai2, Tianyu Liu2, Baobao Chang1 1Peking University 2Alibaba Group 3University of Wisconsin–Madison 4Beijing Institute of Technology leo.liang.chen@outlook.com

(a) Generation Examples from DnD-Transformer

# arXiv:2410.01912v1[cs.CV]2Oct2024

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|
|---|---|---|---|---|---|
|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|[Figure 13]| |[Figure 14]| |[Figure 15]| |

(b.1) Diffusion (b.2) Autoregressive Image Generation

|[Figure 16]|[Figure 17]|
|---|---|
|[Figure 18]|[Figure 19]|

|[Figure 20]|
|---|

😠？ 😊√

[Figure 21]

[Figure 22]

Figure 1: Generations from DnD-Transformers trained on class-conditional ImageNet256×256 (a.top) and unconditional arXiv images (a.bottom). Unconditional rich-text image generations by trained diffusion (b.1) and autoregressive model (b.2), where autoregressive model has dominating performance, showing a spark of vision-language intelligence after purely training on images.

ABSTRACT

This work tackles the information loss bottleneck of vector-quantization (VQ) autoregressive image generation by introducing a novel model architecture called the 2-Dimensional Autoregression (DnD) Transformer. The DnD-Transformer predicts more codes for an image by introducing a new autoregression direction, model depth, along with the sequence length direction. Compared to traditional 1D autoregression and previous work utilizing similar 2D image decomposition such as RQ-Transformer, the DnD-Transformer is an end-to-end model that can generate higher quality images with the same backbone model size and sequence length, opening a new optimization perspective for autoregressive image generation. Furthermore, our experiments reveal that the DnD-Transformer’s potential extends beyond generating natural images. It can even generate images with rich text and graphical elements in a self-supervised manner, demonstrating an understanding of these combined modalities. This has not been previously demonstrated for popular vision generative models such as diffusion models, showing a spark of vision-language intelligence when trained solely on images. Code, datasets and models are open at https://github.com/chenllliang/DnD-Transformer.

- 1 INTRODUCTION

The field of autoregressive (AR) image generation is experiencing a resurgence of interest, largely driven by groundbreaking advancements in large language models (LLMs), exemplified by the release of ChatGPT (OpenAI, 2022). Because typical AR image generation methods also predict output in a next-token prediction manner, this resemblance has sparked significant efforts in two main areas: 1) transferring advanced, large-scale training techniques and expertise from LLMs to AR image generation models (Bai et al., 2023; Tian et al., 2024; Sun et al., 2024), and 2) developing truly multimodal foundation models capable of both understanding and generating multimodal information within a unified training framework (Lu et al., 2022; 2023; Team, 2024). These developments have the potential to lead to more versatile and powerful multimodal AI systems.

A review of the development history of AR image generation approaches reveals significant efforts focused on finding better sequential decompositions of images and balancing reconstruction fidelity with prediction difficulty. Early models, like PixelCNN (van den Oord et al., 2016), generated images pixel by pixel. This approach was later enhanced by using vector-quantized variational autoencoders (VQVAEs) to compress images and model the prior distribution of discrete tokens in a compact latent space (Van Den Oord et al., 2017). Vector quantization (VQ) paved the way for notable models such as VQGAN (Esser et al., 2021), DALL·E (Ramesh et al., 2021), and MUSE (Chang et al., 2023), and it remains a core technique in recent AR image generation models like VAR (Tian et al., 2024) and LlamaGen (Sun et al., 2024), and multimodal foundation models like LVM (Bai et al., 2023), Unified-IO (Lu et al., 2022; 2023), and Chameleon (Team, 2024).

However, despite advancements in AR image generation, VQ-based autoregressive methods face two persistent criticisms, especially juxtaposed with latent diffusion models (Rombach et al., 2022):

- 1) Information loss inherent in the quantization process. Quantization, specifically in VQVAE, introduces significant information loss. With a typical configuration (N=8192, f=16), the Information Compression Ratio (ICR = log24fN2 , explained in Equation 1) is just 0.21%, drastically lower than the 8.3% of Stable Diffusion’s VAE1, hindering fine-grained detail reconstruction. According to Chameleon (Team, 2024), the authors note that their VQ tokenizer struggles to reconstruct finegrained details like text in images, which we believe is due to the low ICR of their tokenizer.

- 2) Substantially increased computational requirements for producing higher-quality images. According to Equation 1,Increasing ICR by expanding the latent space (N) is logarithmically limited

1The Stable Diffusion VAE (https://huggingface.co/stabilityai/sd-vae-ft-mse) uses a downsampling factor (f) of 8 and 4 channels, with fp32 tensor precision (log N = 4 × 32).

[Figure 23]

|[Figure 24]|
|---|

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

|N|
|---|

28-N

|[Figure 30]|
|---|

17 18 19 20 21 22 23 24 25 26 27

|…|
|---|

28-(N-1)

|[Figure 31]|
|---|

- 28-1
- 28-2

|2|
|---|

|[Figure 32]|
|---|

Denoising Diffusion Process

|1|
|---|

2D Autoregression (DnD, This Work)

|28-N|
|---|

|[Figure 33]|
|---|

O

|Final Transformer Block|
|---|

…

[Figure 34]

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

|+|
|---|

|28-(N-1)|
|---|

|[Figure 35]|
|---|

O

…

|Transformer Block O-(N-1)|
|---|

17 18 19 20 21 22 23 24 25 26 27

…

+

|28-2|
|---|

|[Figure 36]|
|---|

O

- Transformer Block O-1

- Transformer Block O-2

…

+

|28-1|
|---|

|[Figure 37]|
|---|

O

…

|+|
|---|

Add

Input …

|3|
|---|

|27|
|---|

|1|
|---|

|2|
|---|

|25|
|---|

|26|
|---|

1D Autoregression

O Output Head

DnD-Transformer (This Work)

- Figure 2: Illustration of the proposed DnD-Transformer. N denotes the number of depth autoregression. O-i denotes the transformer layer index for the i-th prediction head. Each transformer layer predicts the corresponding depth code, achieving multi-code prediction within one forward pass.

and computationally expensive leading to potential codebook collapse and more embedding parameters, while reducing the downscaling factor (f) significantly increases computational overhead due to a longer token sequence of O(1/f2) and a higher transformer computation complexity of O(1/f4).

We draw inspiration from the Residual Quantization method (Lee et al., 2022b), which provides a new dimension for sequentially decomposing the image for better generation quality. However, the proposed RQ-Transformer employs two separate transformer models. This structure presents difficulties in integrating current LLMs for end-to-end training. In this work, we aim to solve the problem covering the two mentioned concerns: Can we overcome the information loss of VQ-based AR image generation without increasing overall computation budget in an end-to-end manner?

We propose a novel paradigm for AR image generation called 2-Dimensional Autoregression (DnD) and DnD-Transformer, an end-to-end model architecture. DnD Autoregression introduces a new depth dimension along with the original spatial dimension. In the depth dimension, the image patch could be decomposed in any causal coarse-to-fine order, including the residual decomposition (Lee et al., 2022b), Gaussian denoising decomposition (Ho et al., 2020) and etc. With a depth of d and

other configurations unchanged, the ICR of DnD Autoregression becomes d×log24fN2 , more effectively reducing the information loss comparing to increasing the codebook size N.

The remaining problem is how to predict the d times more tokens effectively. We propose the DnDTransformer. As shown in Figure 2, it inserts multiple prediction heads into the backbone transformer decoder model to predict the depth codes and conduct additional autoregressive predictions in each forward process. Different from RQ-Transformer (Lee et al., 2022b), the DnD-Transformer does not require additional modules or increased sequence length, making it applicable to any language model architecture and efficiently generate more fine-grained images.

Our experiments show several interesting results:

- 1. Superior reconstruction of fine-grained image details using residual image decomposition in VQVAEs, disproving VQ’s limitations with text-rich images

- 2. More efficient and lower-entropy decomposition with DnD autoregression compared to 1D methods, evidenced by lower training cross-entropy loss despite predicting more codes
- 3. Significant outperformance of the AR baseline on ImageNet 256x256 generation, achieving up to 1.54 FID and 82.6 IS improvements (XXL model, cfg=2) without increased model size or sequence length, even surpassing larger LlamaGen model trained with longer sequence length
- 4. A spark of vision-language intelligence for the first time, enabling unconditional richtext image generation, outperforming diffusion models like DDPM and Stable Diffusion on dedicated rich-text image datasets, highlighting the distinct advantage of autoregressive models for multimodal modeling.

- 2 2D VISUAL TOKENIZER AND 2D AUTOREGRESSION

- 2.1 UNDERSTAND VQVAE AS COMPRESSION

We introduce the basics of AR generation in Section A in the appendix. We can better understand the reconstruction ability of VQVAE from the lens of compression. Let us assume a VQVAE with downscaling factor f, codebook size N, input image’s size of H × W, then the shape of the quantized code is h × w = (H/f) × (W/f). We assume that the code follows a uniform distribution, so each code has log N bits information. Its information compression ratio (ICR) is as follows.

ICR(N,f) =

(H/f) × (W/f) × log N H × W × 3 × log 256

=

log N 24f2

(1)

A typical configuration (N=8192, f=16) results in 0.21% ICR. This ICR is significantly lower than JPEG’s 5% ICR (Wikipedia). To increase ICR, the 1D AR method could increase N (might face the codebook collapse problem (Mentzer et al., 2023) and the improvement is logarithmically bounded) or decrease f (more effective, but increases the token count quadratically).

- 2.2 IMAGES’ 2D DECOMPOSITION AND QUANTIZATION

As pointed out by Equation 1, the information compression ratio of VQVAE is bounded by the size of the codebook and the downscaling ratio. Residual Quantization (Lee et al., 2022b) proposes a new direction to quantize the image feature with multiple residual codes to reduce the quantization error and improve the quality of the reconstruction. For a feature map having h×w vectors, RQVAE uses h × w × d codes to quantize the feature map, where d is the depth dimension of the code. For each feature vector v, RQ finds d codes (q1,q2,...,qd) by sequentially conducting d times residual decomposition and quantization operation Q(x) as finding the closest entry to x from the codebook:

##### qd = Q(rd−1), rd = rd−1 − qd, r0 = v (2)

Consequently, the sum of the residual codes di=1 qi is expected to approximate more closely the feature vector v, thus reducing the quantization error. We generalize this process as two-dimensional

autoregression (DnD), which extends beyond Markov residual decomposition and can be applied to any decomposition operation, such as the diffusion process (Ho et al., 2020), etc.

DnD Autoregression quantizes a 2D feature map m ∈ Rh·w·c by decomposing it in two directions. First, m is divided into h · w feature vectors. Second, each vector v is decomposed into n codes (q1,...,qn) using a function Dn(v,Q) based on a codebook Q. The resulting quantized map q has shape h·w ·n and is predicted in depth-first-spatial-second order. This decomposition could also be non-Markov, unlike RQVAE. The selection of potentially better decomposition functions is left for future exploration. We still use the residual quantization from Equation 2 as Dn. DnD decomposition increases the ICR d times (Equation 3), more effectively than increasing codebook size. The remaining challenge of predicting d times more codes is addressed by our DnD-Transformer.

(H/f) × (W/f) × log N H × W × 3 × log 256

log N 24f2

= d ×

ICR(N,f,d) = d ×

(3)

|[Figure 38]|
|---|

[Figure 39]

[Figure 40]

[Figure 41]

D=1 (VQ) L2 Loss 0.0088 PSNR 20.56

D=8 (RQ) L2 Loss 0.0027 PSNR 25.68

D=2 (RQ) L2 Loss 0.0059 PSNR 23.27

Ground Truth (512x512)

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

D=1 (VQ) L2 Loss 0.0322 PSNR 14.92

D=8 (RQ) L2 Loss 0.0014 PSNR 28.69

D=2 (RQ) L2 Loss 0.0218 PSNR 16.61

Ground Truth (512x512)

- Figure 3: Performance of our visual tokenizers of different depths. The reconstruction of complex features (i.e., eyes, mouse and text) gains significant improvement as the depth increases.

Text256 Text512 arXiv512 rOCR↑

ImageNet 256×256 rFID↓ L2 Loss↓ Code Usage↑

Depth

Depth

1 0.15 0.73 0.14

- 1 2.98 0.11 100%

- 2 0.93 0.08 100% 4 0.60 0.05 100% 8 0.42 0.04 100%

- 1† 0.00 0.00 0.00

- 2 0.50 0.81 0.49 8 0.80 0.83 0.67

SDXL 0.68 0.05 SD3 0.67 0.04 -

SDXL 0.72 0.83 0.66 SD3 0.82 0.83 0.74

(a) Reconstruction Performance on ImageNet 256×256 Validation Set.

(b) Reconstruction OCR Performance. † indicates zero-shot tokenizer trained on ImageNet.

- Table 1: Ablation studies on the reconstruction performance of visual tokenizers. Our trained tokenizers all have a f = 16 downscaling factor and N = 16384 codebook size.

- 2.3 RECONSTRUCTION PERFORMANCE

We evaluate the reconstruction performance of our trained visual tokenizers with varying maximum codebook depths using the standard ImageNet dataset as the benchmark. All images are resized to 256×256 resolution. We train the different visual tokenizers using the same training objectives as in Lee et al. (2022b), and assess the reconstruction Fr´echet Inception Distance (rFID) on the ImageNet validation set using ADM’s evaluation suite (Dhariwal & Nichol, 2021). The results are presented in Table 1a. For comparison, we include the rFID from the VAE of SDXL (Podell et al., 2023) and Stable-Diffusion 3 (Esser et al., 2024) . Our findings demonstrate that our trained visual tokenizer achieves an rFID lower than 1 with two or more codebook depths, even surpassing the performance of SD3’s continuous VAE with less theoretical information loss. As shown in the example from Figure 3, by increasing code depth, we could reconstruct more fine-grained details in the image.

Code Usage. We further analyze the code usage in each codebook layer, with results shown in Figure 4a. The analysis indicates that usage generally decreases as depth increases. This is due to the diminishing diversity of code usage as the residual decomposition progresses deeper, resulting

Layerwise Code Usage of Different Visual Tokenizers

1.0

Max Depth (d)

- 1

- 2 4

| |
|---|

0.8

CodeUsage(%)

| |
|---|

8

0.6

0.4

0.2

0.0

1 2 3 4 5 6 7 8

Codebook Depth

(a) Layerwise code usage of visual tokenizers.

Code Norm Distribution of Different Depths

- Depth 0

- Depth 1

- Depth 2

- Depth 3

- Depth 4

- Depth 5

- Depth 6

- Depth 7

0.35

0.30

0.25

Density

0.20

0.15

0.10

0.05

0.00

0 25 50 75 100 125 150 175

Norm of Code Embeddings

(b) Code Norm Distribution for Tokeniers

Figure 4: Analysis of visual tokenizers.

in smaller feature norms and more centralized code usage according to Figure 4b. Interestingly, we do not observe signs of codebook collapse with the DnD visual tokenizers, even when using a large codebook size (16384), as mentioned in previous work (Mentzer et al., 2023). While they reported much lower code usage (< 50%), our tokenizer achieves 100% usage across all maximum depths.

- 2.4 VQVAES CAN PERFECTLY RECONSTRUCT RICH-TEXT IMAGES

A prevalent criticism of VQVAE has been its alleged intrinsic information loss problem, particularly its inability to reconstruct images with fine details, such as those containing rich text (Team, 2024). However, we argue that this claim is unfounded. Our findings suggest that VQVAE can indeed achieve perfect reconstruction of detailed images, when provided sufficient data and an increased number of codes used to represent each image. This demonstrates that the perceived limitations of VQVAE can be overcome through appropriate data-centric adjustments and model scaling-up.

rOCR - A New Metric. We proposes rOCR, a novel metric for evaluating rich-text image reconstruction. Unlike rFID/L2 Loss, rOCR measures textual recognizability using the Qwen2-VL72B (Wang et al., 2024a) visual language model for OCR. The metric computes the Rouge-L score between recognized and groundtruth text (or original image OCR if groundtruth is unavailable).

Experiments and Results. Two rich-text image datasets, Text-Image and arXiv-Image (details in Section 4.1), were used to train visual tokenizers. Performance (rOCR scores) was evaluated on both datasets’ 1K test sets, compared against ImageNet-trained tokenizers, SDXL’s (Podell et al., 2023) and Stable-Diffusion-3’s VAE (Esser et al., 2024). Text-Image was also tested at a reduced 256×256 resolution to assess resolution impacts. Table 1a shows the rOCR results, with reconstruction examples in Figures 3 and 11. Results indicate more training data and deeper tokenizers improve text reconstruction. Unlike Team (2024), our discrete visual tokenizers excel in rich-text image reconstruction even compared to continuous VAEs.

- 3 THE DND-TRANSFORMER

Prior section showed DnD visual tokenizers effectively reconstruct fine details like text. However, efficiently predicting the increased number of depth codes (d times more) remains challenging. Existing methods, like RQ-Transformer, use a separate transformer for depth, hindering integration with LLMs. We propose an efficient end-to-end architecture for multi-code prediction.

- 3.1 DND-TRANSFORMER DESIGN

Figure 5 shows DnD-Transformer and its variants: Parallel and Vertical Prediction. Parallel Prediction adds multiple prediction heads for simultaneous multi-depth code prediction, similar to accelerated LLM inference (Cai et al., 2024). However, this ignores the coarse-to-fine nature (Figure 4b) of code distributions, where deeper codes have smaller norms and are more centered. Vertical Prediction addresses this by sequentially predicting codes. Adding autoregression further refines this

[Figure 46]

|28-N|
|---|

O

|Final Transformer Block|
|---|

…

+

|28-(N-1)|
|---|

O

Transformer Block O-(N-1) …

|+| |
|---|---|
| | |

+

|28-2|
|---|

|28|
|---|

…

O

- Transformer Block O-1

- Transformer Block O-2

…

+

|28-1|
|---|

O

…

|+|
|---|

Add

|1|
|---|

|2|
|---|

|3|
|---|

|25|
|---|

|26|
|---|

|27|
|---|

…

O Output Head

DnD Transformer

Figure 5: Different explored multi-token prediction architectures for DnD-Transformer, which are all designed to generate multiple codes with one forward pass.

by conditioning deeper code predictions on previous ones, achieving the best multi-layer code prediction without increasing model parameters or sequence length. Ablation on the structure design is shown in Table 3 from Appendix.

- 3.2 IMPLEMENTATION DETAILS

As shown in the left part of Figure 5, the increment of DnD-Transformer compared to vanilla transformer decoder is the additional output head and embedding add operation. Let’s assume the linearized codemap’s length is L = h × w and code depth is d. During generation, DnD-Transformer conducts L forward process and each forward process generate d codes sequentially. After generating codes for all depths in a forward process, the embeddings of all codes are added up as the next input token. In this way, the model could generate L × d tokens with only L forward passes, improving the generation quality with the same inference cost as standard 1D auto-regression transformer. The only additional hyper-parameter is the layer indexes to predict code of different depths. We adopt the same transformer decoder’s architecture as LLaMA (Touvron et al., 2023) and , please refer to Appendix E for the training details of our DnD-Transformer.

- 4 EXPERIMENTS AND FINDINGS

- 4.1 TASKS AND DATASETS

Class-Conditional Image Generation. We conduct standard conditional image generation task with ImageNet-1k benchmark. Images are resized to 256×256 resolution during training and evaluation. We sample 50k images with classes uniformly distributed, and compute the FID, IS, Precision and Recall aganist the training set data using the ADM evaluation tool Dhariwal & Nichol (2021).

Unconditional Rich-Text Image Generation. We collect two datasets for this task. Dataset examples are shown in Figure 6. Models are trained in a unconditional setting in this task. We aim to explore whether the tested vision generation models could understand and generate the complex logical interrelation among the generated elements such as language.

- 1. Pure Text Images (Text-Image). The dataset is automatically rendered from a portion of English wikipedia (Foundation), consisting of 2.4M images. Each image has a original resolution of 512×512 and a font size of 32pt. We set a maximum of 100 words in each image with a paddling margin of 20pt. We use the PILLOW library to render the image.
- 2. arXiv Images (arXiv-Image) we first download the papers in PDF format from arXiv. org, and render the pages to image of A4 resolution (1260×1782) with PDF2IMAGE tool. We then randomly crop ten 512×512 image from each pages and finally collect 2M images.

We have developed an evaluation pipeline that combines Optical Character Recognition (OCR) and Perplexity Measurement for assessing the quality of generated images, with a focus on the textual information they contain. Initially, we employ the state-of-the-art open-source Vision-Language

|[Figure 47]|[Figure 48]|
|---|---|

Text arXiv

Figure 6: Data examples in of the collected Text-Image and arXiv-Image image datasets.

Model, Qwen2-VL-72B, to extract text from the generated images. Subsequently, we utilize the Qwen2.5-72B model to calculate the perplexity of the generated text, where the LLM is regraded as the evaluator. The resulted score is called PPLocr, we also test the score of groundtruth data from the training images as the performance upper-bound.

- 4.2 MODELS

Visual Tokenizers. We train our visual tokenizer based on RQVAEs (Lee et al., 2022b). We train tokenizers with code depths of {1,2,4,8} and scaling factor f = 16 across different experiments. We choose the checkpoint with best rFID across 150 epochs. Performance comparison of different visual tokenizers is shown in Table 1. We follow Lee et al. (2022b) to train the visual tokenizers. Details of the training of visual tokenizers are listed in Appendix B. Reconstruction performance of the trained visual tokenizers is shown in Table 1.

DnD-Transformer. We train two size of DnD-Transformers across our experiment, namely DnDTransformer-XXL (1.4B) and DnD-Transformer-XXXL (2.5B). Basically, DnD-Transformer inherits the LLaMA (Touvron et al., 2023) architecture. The XXL version strictly align with the LlamaGen-XXL baseline to be fairly compared. Details of the model are shown in Appendix E.

Implemented Baselines for Class-Conditional Image Generation. LlamaGen (Sun et al., 2024) is the major baseline and state-of-the-art model for AR image generation on ImageNet. Our implemented code primarily refers to the same training codebase for fair comparison. LlamaGen could be also viewed as a special version of DnD-Transformer where the decomposition depth equals to 1.

Implemented Baselines for Rich-Text Image Generation. We select multiple diffusion models as the baselines, including DDPM (Ho et al., 2020), Stable Diffusion XL (SDXL) (Podell et al.,

- 2023) and Stable Diffusion v3.0 (SD3) (Esser et al., 2024). For DDPM, we train the model on the dataset from scratch. For SDXL and SD3, we finetune the checkpoints from the official website.

- 4.3 RESULTS OF CLASS-CONDITIONAL IMAGE GENERATION

As demonstrated in Table 2, our DnD-Transformer significantly outperforms the 1D autoregressive baseline LlamenGen across various scales and generation evaluation metrics, including FID and IS. This superior performance is achieved while maintaining the same number of parameters in the backbone model, based on our reported and implemented results. It is noteworthy that our 2.5B model, trained with a sequence length of 256, even outperforms the 3.1B LlamaGen model, which was trained with a much longer image sequence length of 576. This result demonstrates that the DnDTransformer can effectively predict a greater number of tokens within a shorter sequence length, highlighting its significant potential to revolutionize the one-dimensional autoregressive paradigm. We randomly sample some generation results as shown in Figure 1 and compare the generation performance with 1D-AR in Figure 12,13 and 14 from the Appendix. The comparative analysis clearly illustrates the effectiveness of our approach to generate high-quality images.

Type Model #Para. FID↓ IS↑ Precision↑ Recall↑

ADM (Dhariwal & Nichol, 2021) 554M 10.94 101.0 0.69 0.63

CDM (Ho et al., 2022) − 4.88 158.7 − − LDM-4 (Rombach et al., 2022) 400M 3.60 247.7 − − DiT-XL/2 (Peebles & Xie, 2023) 675M 2.27 278.2 0.83 0.57

Diffusion-Reported

VQGAN (Esser et al., 2021) 1.4B 5.20 280.3 − − RQTransformer (Lee et al., 2022a) 3.8B 7.55 134.0 − −

LlamaGen-XXL (cfg=2) (Sun et al., 2024) 1.4B 3.64 296.5 0.86 0.51 LlamaGen-XXL† (384×384, cfg=2) (Sun et al., 2024) 1.4B 2.52 295.4 0.84 0.56 LlamaGen-3B (cfg=2) (Sun et al., 2024) 3.1B 4.21 325.2 0.87 0.49 LlamaGen-3B† (384×384, cfg=2) (Sun et al., 2024) 3.1B 2.81 311.6 0.84 0.54

AR-Reported

LlamaGen-XXL (cfg=4) 1.4B 7.67 345.1 0.89 0.35 LlamaGen-XXL (cfg=2) 1.4B 4.12 266.9 0.83 0.49 DnD-Transformer-XXL (cfg=4) 1.4B 6.55 427.7 0.89 0.42 DnD-Transformer-XXL (cfg=2) 1.4B 2.58 295.6 0.83 0.56 DnD-Transformer-XXL (cfg=1.7) 1.4B 2.78 239.2 0.82 0.56 DnD-Transformer-XXL (cfg=1.5) 1.4B 2.96 232.5 0.80 0.57 DnD-Transformer-XXXL (cfg=4) 2.5B 6.48 413.0 0.89 0.42 DnD-Transformer-XXXL (cfg=2) 2.5B 2.77 319.1 0.85 0.54 DnD-Transformer-XXXL (cfg=1.7) 2.5B 2.21 279.3 0.83 0.58 DnD-Transformer-XXXL (cfg=1.5) 2.5B 2.52 244.2 0.80 0.59

AR-Implemented

- Table 2: Model comparisons on class-conditional ImageNet 256×256 benchmark. The “Reported” results refer to Sun et al. (2024). The “Implemented” results are conducted in this work. † indicates that the model is unorthodoxly trained at 384×384 resolution, which requires 2.25 times longer sequence length compared to our implemented models. “cfg” means the scale of classifierfree guidance. The number of depth autoregression is 2 for DnD-Transformers.

C o m p a r is o n o f F I D s a lo n g T r a in in g o n I m a g e N e t

L la m a G e n - X X L

D n D - T r a n s f o r m e r - X X L

1 0

- 7
- 8
- 9

2 4 6 8 1 0

E p o c h s

(a) Comparison of FIDs along training.

C o m p a r is o n o f P P L a lo n g T r a in in g o n T e x t - I m a g e

T e m p e r a tu r e 0 .1 D D P M

- 1 6 0 0
- 2 0 0 0

- T e m p e r a tu r e 0 .5 G r o u n d T r u th

- T e m p e r a tu r e 1 .0

1 2 0 0

8 0 0

4 0 0

1 8

1 2 3 4 5 6 7 8 9 1 0

E p o c h s

(b) Sampling PPLocr on Text-Image along training.

Figure 7: Curves during training.

- 4.4 RESULTS OF RICH-TEXT IMAGE GENERATION

Generation Results on Text-Image. A DnD-Transformer (depth 1) and a DDPM model were trained on the same text-image dataset. Comparing 250 randomly sampled images from each, the AR model significantly outperformed the diffusion model in generating coherent text (lower OCR perplexity 7b; Generation examples 1, 16, 17, 18 and 19 ). This suggests the AR model’s discrete token reconstruction enables effective autoregressive modeling. We also find that with a lower sampling temperature, the model would generate text images with lower PPL just like LLMs. Conversely, the diffusion model’s simultaneous generation hinders text coherence.

Generation Results on arXiv-Image. An 8-layer visual tokenizer and corresponding DnDTransformer trained on arXiv-Image outperformed diffusion model baselines, generating more valid words and phrases (Figure 8). However, arXiv-Image generation lagged behind Text-Image generation, suggesting joint language and figure modeling is more challenging. More results and baselines are in Figure 15 and 20. While SD3’s VAE reconstructs arXiv images well (Table 1b), its generative performance is inferior to DDPM and AR, suggesting its latent space is less suitable for language modeling comparing to pixel or discrete space.

A Spark of Vision-Language Intelligence. Autoregressive (AR) image generation exhibits a marked advantage over diffusion models in producing text-rich images, as demonstrated by our results. The pixel-level language generation inherent to AR models facilitates this capability. De-

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

SD3

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

DnD

Figure 8: Comparison of Unconditional Rich-Text Image Generation on the more complex arXivImage dataset. SD3 is hard to generate valid words, while DnD-Transformer demonstrates an ability to generate semantically appropriate phrases, as marked in blue. More baselines are in Figure 15.

[Figure 57]

(a) Training Loss for DnD-Transformer trained with different number of prediction heads.

[Figure 58]

(b) Training Loss when trained on different domain datasets.

Figure 9: Analysis of code depths and domains during training DnD-Transformers.

spite limitations imposed by our current training data and model size (preventing direct comparison with large language models), these findings suggest a promising pathway towards vision-language intelligence where language understanding emerges directly from visual perception. Furthermore, our pure image learners display behaviors mirroring language model issues such as repetition and hallucination (Figure 10), implying the potential for integrating pure language modeling into a unified autoregressive framework for joint vision-language image modeling.

- 4.5 TRAINING BECOMES EASIER WHEN PREDICTING MULTIPLE CODES, SAMPLING NOT

Deeper DnD-Transformer codes achieve lower cross-entropy loss during training (Figure 9a), indicating lower entropy image decompositions. However, despite this, increased depth doesn’t improve ImageNet generation fidelity, possibly due to the larger sampling space. Exploring this multi-depth sampling space for better generation is a promising research direction.

- 4.6 AR TRAINING LOSS FOR DIFFERENT DOMAINS ALIGN WITH INNER RANDOMNESS

Training loss for the same DnD-Transformer varies significantly across datasets (Figure 9b), being notably higher for ImageNet than rich-text images. While rich-text image loss nears that of LLMs, ImageNet loss sits between text and natural image datasets. The AR model’s LLM-like training suggests it learns language from visual input alone, implying language’s visual representation has lower entropy than natural images, easing the learning process.

|[Figure 59]|
|---|

|[Figure 60]|
|---|

Repetition Hallucination

- Figure 10: Some cases of the generated text images. We witness similar error pattern (marked in red) to LLMs such as repetition and hallucination in our trained model during sampling.

- 5 RELATED WORK

Image Generation with VQVAE. The vector quantization (VQ) method has been pivotal in the development of generative models (Ramesh et al., 2021; Yu et al., 2022; Chang et al., 2023), which achieve image generation through the prediction of discrete image tokens. Efforts in this area focus on two main directions: the optimization of image tokenization techniques (Esser et al., 2021; Mentzer et al., 2023; Yu et al., 2023; 2024; Weber et al., 2024), and the strategic planning of effective decompositions of image tokens, such as MaskGit (Chang et al., 2022) and VAR (Tian et al.,

- 2024). Meanwhile, alongside the advancement of large language models, there is growing interest in autoregressive image generation, which predicts image tokens sequentially (Tian et al., 2024; Sun et al., 2024). Recent research has also focused on developing multimodal foundation models (Lu et al., 2023; Kondratyuk et al., 2024; Wang et al., 2024b) that integrate both understanding and autoregressive image generation capabilities. They typically convert images or videos into sequences of discretized tokens and train over combined text-image/video token sequences within the AR modeling framework (Lu et al., 2022; Bai et al., 2023; Xie et al., 2024; Team, 2024). However, these models often struggle with inherent information loss during the image quantization and the significantly increased computational demands when generating higher-quality images. The DnDTransformer that adopts the residual 2D decomposition of image features does not require additional modules or increased sequence length for high-quality and fine-grained image generation.

Rich-Text Image Generation. Despite recent significant progress in image generation, the task of rich-text generation within images remains a persistent challenge (Chen et al., 2023b; Ma et al., 2024; OpenAI, 2024). Most advancements have been witnessed in diffusion models (Betker et al., 2023; Saharia et al., 2022b;a), these models either leverage large language models to enhance the character spelling capabilities of generative models (Saharia et al., 2022b; Balaji et al., 2023; Saharia et al., 2022a) or attempt to explicitly control the position and content of the text using additional supervision from different modules (Tuo et al., 2024; Yang et al., 2023; Liu et al., 2024). However, most diffusion-based methods have primarily focused on text rendering Chen et al. (2023a;b); Balaji et al. (2023); Saharia et al. (2022a) in image generation, often limited to generating short words for logos and posters (Yang et al., 2023; Ma et al., 2023; 2024). The full potential of rich-text image generation remains largely unexplored. Our methods, which build on the foundation of DnD Autoregression, show substantial progress in generating rich-text images in an unconditional manner, highlighting the feasibility of conducting joint vision-language modeling tasks using purely images.

- 6 CONCLUSION

This paper investigated the limitations of autoregressive (AR) image generation methods, particularly the information loss and computational burden associated with vector quantization (VQ). We introduced 2-Dimensional Autoregression (DnD) and a novel end-to-end architecture, DnDTransformer, which leverages a depth dimension autoregression alongside the spatial dimension to mitigate these limitations. Our experiments demonstrate that DnD-Transformer achieves significant

improvements in image quality, outperforming strong baselines like LlamaGen without increasing model size or sequence length. Notably, DnD-Transformer showcases emergent vision-language intelligence, generating text-rich images unconditionally, a known weakness of diffusion models. These findings highlight the potential of DnD for efficient and high-quality AR image generation and underscore the promise of this approach for advancing multimodal foundation models.

REFERENCES

Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. arXiv preprint arXiv:2312.00785, 2023.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers, 2023. URL https://arxiv.org/abs/2211.01324.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024. URL https://arxiv.org/abs/2401.10774.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022.

Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser: Diffusion models as text painters, 2023a. URL https://arxiv.org/abs/2305.10855.

Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser-2: Unleashing the power of language models for text rendering, 2023b. URL https://arxiv. org/abs/2311.16465.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024.

Wikimedia Foundation. Wikimedia downloads. URL https://dumps.wikimedia.org. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint

arXiv:2207.12598, 2022. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. The Journal of Machine Learning Research, 23(1):2249–2281, 2022.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation, 2024. URL https://arxiv.org/abs/2312.14125.

Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11523–11532, 2022a.

Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization, 2022b. URL https://arxiv.org/abs/2203. 01941.

Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. Glyphbyt5: A customized text encoder for accurate visual text rendering, 2024. URL https:// arxiv.org/abs/2403.09622.

Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. arXiv preprint arXiv:2206.08916, 2022.

Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. arXiv preprint arXiv:2312.17172, 2023.

Jian Ma, Mingjun Zhao, Chen Chen, Ruichen Wang, Di Niu, Haonan Lu, and Xiaodong Lin. Glyphdraw: Seamlessly rendering text with intricate spatial structures in text-to-image generation, 2023. URL https://arxiv.org/abs/2303.17870.

Jian Ma, Yonglin Deng, Chen Chen, Haonan Lu, and Zhenyu Yang. Glyphdraw2: Automatic generation of complex glyph posters with diffusion models and large language models, 2024. URL https://arxiv.org/abs/2407.02252.

Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantiza-

tion: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023. OpenAI. Chatgpt. https://openai.com/blog/chatgpt, 2022. OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. Accessed:

[01-10-2024. William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. article, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022a. URL https://arxiv.org/abs/ 2205.11487.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022b.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation, 2024. URL https: //arxiv.org/abs/2406.06525.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing, 2024. URL https://arxiv.org/abs/2311. 03054.

Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelcnn decoders, 2016. URL https:// arxiv.org/abs/1606.05328.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution, 2024a. URL https://arxiv.org/abs/2409. 12191.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024b. URL https://arxiv.org/abs/2409.18869.

Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and LiangChieh Chen. Maskbit: Embedding-free image generation via bit tokens, 2024. URL https: //arxiv.org/abs/2409.16211.

Wikipedia. URL https://en.wikipedia.org/wiki/Image_compression.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation, 2024. URL https://arxiv.org/abs/ 2408.12528.

Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, and Kai Chen. Glyphcontrol: Glyph conditional control for visual text generation, 2023. URL https: //arxiv.org/abs/2305.18259.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion– tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation, 2024. URL https://arxiv. org/abs/2406.07550.

- A PRELIMINARY: AUTOREGRESSIVE IMAGE GENERATION

In this section, we introduce the fundamentals of autoregressive image generation. The pipeline is rooted in the Vector Quantized Variational Autoencoder (VQVAE) (Van Den Oord et al., 2017) and the autoregressive Transformer (Vaswani et al., 2017). This approach has been adopted from the early DALLE (Ramesh et al., 2021) to the latest LlamaGen (Sun et al., 2024).

- A.1 STEP1: TRAIN THE VISUAL TOKENIZER AND TOKENIZE THE IMAGES

Images initially exist in the pixel-level RGB color space, which consists of little semantic information and makes it challenging to directly model prior knowledge. For example, an image with a resolution of 256 × 256 comprises 256 × 256 × 3 = 196,608 distinct values, representing the individual red, green, and blue intensities for each pixel. The large sequence length makes it difficult to train in autoregressive manner similar to language models’ technique. Van Den Oord et al. (2017) proposed the Vector Quantized Variational Autoencoder (VQVAE), which significantly alleviates the problem. It downscales and tokenizes the image from the original sparse RGB space into a dense and discrete representational space (codebook) Q by finding the nearest entry. The VQVAE is typically implemented in an encoder-decoder architecture, with its primary training objective being to minimize the image reconstruction loss. You could refer to Van Den Oord et al. (2017) for details in training a standard VQVAE.

- A.2 STEP2: LEARN THE PRIOR DISTRIBUTION OF IMAGE TOKENS

Having tokenized the source images into discrete tokens and trained a visual decoder to map these tokens back to real images, the next crucial step is to learn the prior distribution of the discrete tokens. This distribution enables the sampling process, which is essential for generating new images. AR Image generation generally first linearizes the h × w image tokens q ∈ Q in a raster scan order and formalize 1D sequence (q1,q2,q3,...,qh×w) for the transformer (Vaswani et al., 2017) model to learn.

During training, the training objective is the same as GPT’s next token prediction task (Radford et al., 2018), that the model is required to predict the next image token given the previous tokens

and class or text conditional tokens t h=1×w p(qt | q<t,c). After training, we can generate images by autoregressively sampling h × w tokens from the model. The sampled 1D sequence of image

tokens is then reshaped to 2D code map with height h and width w. This reshaped token map is subsequently fed into the trained VQVAE decoder, which reconstructs the final image from the code representation.

Classifier-Free Guidance As a technique to enhance the visual quality and text-image alignment, classifier-free guidance (Ho & Salimans, 2022) has been adopted across the diffusion models (Rombach et al., 2022; Podell et al., 2023), VQ models (Chang et al., 2023) and autoregressive models (Sun et al., 2024) for image generation. During the training, the model is exposed to data with and without conditioning: the conditioning is randomly discarded from a fraction of the training samples. We have implemented this approach in our model as well. Specifically, during training, we randomly replace the conditional embedding with a learnable unconditional embedding in 10% of the cases. At the inference stage, the logits ℓg are recalculated for each generated token. We form the ℓg by subtracting the unconditional logits ℓu by conditional logits ℓc with the guidance scale t through the following equation:

##### ℓg = ℓu + (ℓc − ℓu) × t (4)

- B TRAINING DETAILS OF VISUAL TOKENIZERS

We follow (Lee et al., 2022b) to train the 2D tokenizers with residual decomposition a combined objective of l2 loss, GAN loss and perceptual loss. Codes from different depth share the same codebook. We train all tokenizers a fixed learning rate of 4e-5, a total batch-size of 256 for 100 epochs and select the one with lowest validation loss as the final tokenizers. We conduct all training on 8×A100 GPUs.

- C RECONSTRUCTION RESULTS OF TEXTS

Figure 11 shows the reconstruction result on arXiv images of different visual tokenizers.

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

No arXiv Data D=1 (VQ)

Add arXiv Data D=1 (VQ)

Add arXiv Data D=8 (RQ)

- Figure 11: Reconstruction Results of Texts. With training data and enough depths of codes, RQ visual tokenizers can well reconstruct the text in the images.

- D ABLATION ON DND-TRANSFORMER’S STRUCTURE

Model Parameters FID IS Precison Recall

- 1D 1.4B 4.12 266.9 0.83 0.49
- 2D Parallel 1.4B 6.32 232.1 0.79 0.44 2D Vertical 1.4B 3.18 289.7 0.83 0.57 DnD-Transformer 1.4B 2.58 295.6 0.83 0.56

Table 3: Ablation of DnD-Transformer Architecture on ImageNet dataset. All models follow the same training setting as in Appendix E.

- E DETAILS OF HYPER-PARAMETERS OF DND-TRANSFORMER

- Table 4 shows the hyper-parameters of our trained models. The XXL model has the same setting as in GPT2 (Radford et al., 2019) and LlamaGen (Sun et al., 2024) for fair comparisons. For DnD-Transformer with multiple prediction heads, the prediction layers’ indexes are set to [39,48] when there are two heads, [39,42,45,48] when there are 4 heads in the ImageNet experiments, [27,30,33,36,39,42,45,48] when there are 8 heads in the arXiv-Image experiments.

All transformer models were trained using settings similar to LlamaGen (Sun et al., 2024): a base learning rate of 10−4 per 256 batch size, the AdamW optimizer with β1 = 0.9, β2 = 0.95, and a weight decay of 0.05, along with gradient clipping at 1.0. A dropout of 0.1 was consistently applied to the input token embedding, attention module, and feed-forward network (FFN) module. Similarly, a dropout of 0.1 was used for the class condition embedding for classifier-free guidance. Training was performed for 300 epochs, and the final checkpoint was used for performance evaluation.

Model Parameters Layers Hidden Size Heads

XXL 1.4B 48 1536 24 XXXL 2.5B 48 2048 32

#### Table 4: Model sizes and architecture configurations

- F GENERATION RESULTS OF DND-TRANSFORMERS

[Figure 64]

- Figure 12: Conditional generation comparisons between LlamaGen-XXL and DnD-TransformerXXL on class “golden retriever” from ImageNet. We random sampled 16 images with cfg=4. DnDTransformer generates images with higher quality than the 1D AR model.

[Figure 65]

###### Figure 13: Conditional generation comparisons between LlamaGen-XXL and DnD-TransformerXXL on class “volcano” from ImageNet. We random sampled 16 images with cfg=4. DnDTransformer generates images with higher quality than the 1D AR model.

[Figure 66]

###### Figure 14: Conditional generation comparisons between LlamaGen-XXL and DnD-TransformerXXL on class “husky” from ImageNet. We random sampled 16 images with cfg=4. DnDTransformer generates images with higher quality than the 1D AR model especially for the more complex eyes of husky.

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

DDPM

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

SDXL

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

SD3

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

### DnD-Transformer

- Figure 15: Comparison of Unconditional Rich-Text Image Generation on the more complex arXivImage dataset. All models are trained on the same dataset. The generated images are all in 256x256 resolution. Diffusion-Family models are hard to generate valid words, while DnD-Transformer demonstrates an ability to generate semantically appropriate phrases, as evidenced by the correct clause ”it should be” observed in the second example.

[Figure 83]

###### Figure 16: Unconditional Generation examples of DDPM on Image-Text.

[Figure 84]

- Figure 17: Unconditional Generation examples of DnD-Transformer on Image-Text with tempera-

- ture=0.1.

[Figure 85]

###### Figure 18: Unconditional Generation examples of DnD-Transformer on Image-Text with temperature=0.5.

[Figure 86]

###### Figure 19: Unconditional Generation examples of DnD-Transformer on Image-Text with temperature=1.0.

[Figure 87]

- Figure 20: Unconditional Generation examples of DnD-Transformer on arXiv data with tempera-

- ture=1.

