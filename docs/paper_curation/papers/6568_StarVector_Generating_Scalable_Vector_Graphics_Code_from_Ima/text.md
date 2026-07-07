arXiv:2312.11556v4[cs.CV]31May2025

[Figure 1]

### StarVector: Generating Scalable Vector Graphics Code from Images and Text

Juan A. Rodriguez1,2,4 Abhay Puri1 Shubham Agarwal1,2 Issam H. Laradji1,5 Pau Rodriguez1 Sai Rajeswar1,2 David Vazquez1 Christopher Pal1,2,3 Marco Pedersoli1,4

1ServiceNow Research 2Mila - Quebec AI Institute 3Canada CIFAR AI Chair 4ÉTS, Montréal, Canada 5UBC, Vancouver, Canada https://starvector.github.io/

Input Images to Vectorize

Image Vectorization Output

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Generated SVG Code

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

##### StarVector

[Figure 15]

[Figure 16]

Multimodal Language Model for SVG Generation

- A
- B

SHUTTLE

SHUTTLE

rasterize

Text Instructions to SVG

Text-Driven SVG Output

An icon of a red cross inside a circle

StarVector performs SVG generation from images or texts. It is trained on the large-scale SVG-Stack Dataset, and it is evaluated on SVG-Bench

[Figure 17]

An green and white emergency exit icon

[Figure 18]

A design of a storm cloud

- Figure 1. StarVector: A foundation model for SVG generation. StarVector’s multimodal architecture allows input from raster images or text instructions. It converts a variety of raster visuals, including icons, logos, and technical diagrams, into vector graphics or generates new SVGs from text. (Left) Inputs: raster images and text. (Right) Outputs: vectorized images (SVG)

###### Abstract

###### 1. Introduction

Scalable Vector Graphics (SVGs) are vital for modern image rendering due to their scalability and versatility. Previous SVG generation methods have focused on curve-based vectorization, lacking semantic understanding, often producing artifacts, and struggling with SVG primitives beyond path curves. To address these issues, we introduce StarVector, a multimodal large language model for SVG generation. It performs image vectorization by understanding image semantics and using SVG primitives for compact, precise outputs. Unlike traditional methods, StarVector works directly in the SVG code space, leveraging visual understanding to apply accurate SVG primitives. To train StarVector, we create SVG-Stack, a diverse dataset of 2M samples that enables generalization across vectorization tasks and precise use of primitives like ellipses, polygons, and text. We address challenges in SVG evaluation, showing that pixelbased metrics like MSE fail to capture the unique qualities of vector graphics. We introduce SVG-Bench, a benchmark across 10 datasets, and 3 tasks: Image-to-SVG, Text-toSVG generation, and diagram generation. Using this setup, StarVector achieves state-of-the-art performance, producing more compact and semantically rich SVGs.

Vector graphics represent an archetypal form of image representation, where visual compositions are constituted by scalable primitive shapes [33, 43, 50, 50]. For modern image rendering, Scalable Vector Graphics (SVGs) [60] have become the standard for representing vector graphics. The SVG format [25] provides a comprehensive set of primitives and styling options. At its core, the path represents basic curves [60]. Combined with primitives like polygon or ellipse, SVGs define complex designs precisely.

The task of image vectorization, i.e., converting pixelbased raster images into SVGs, stands as a fundamental challenge in vector graphics. The main challenge lies in developing methods that generalize across diverse domains, from fonts and logos to complex illustrations and diagrams [7, 8, 69, 70]. Traditional approaches often rely on approximating images through multiple paths [43, 50, 51, 59, 87]. This strategy can be inefficient as shown in Fig.

###### 2 (Right). For instance, a circle shape could be represented as long path or, more precisely and compactly, as a single <circle/> primitive. Similarly, text elements should be vectorized as editable <text/> primitives to retain the original textual content. This balance between curve-based shape approximation and accurately recognizing primitives

###### Input Image StarVector-8B LIVE VTracer

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

SVG Generation

SVG Generation

SVG Generation

###### SVG Generation

###### SVG Generation

SVG Generation

###### SVG Generation

###### SVG Generation

SVG Generation

[Figure 28]

[Figure 29]

[Figure 30]

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

DinoScore: 0.9969 MSE: 0.0024

DinoScore: 0.9748 MSE: 0.0007

DinoScore: 0.9248 MSE: 0.0063

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

SVG Generation

SVG Generation

SVG Generation

###### SVG Generation

###### SVG Generation

SVG Generation

###### SVG Generation

###### SVG Generation

SVG Generation

[Figure 49]

[Figure 50]

[Figure 51]

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

Pixel Difference

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

DinoScore: 0.9692 MSE: 0.0090

DinoScore: 0.9186 MSE: 0.0012

DinoScore: 0.9100 MSE: 0.0026

StarVector-8B <svg width="300" height="300" ... version="1.1"> <rect x="110" y="10" width="160" height="230" fill="pink"

stroke="red"/> <circle cx="160" cy="120" r="120" fill="tan" stroke="green"/> <polygon points="110,20 280,125 175,280 5,180" fill="blue" /> </svg> VTracer <svg version="1.1" ... width="910" height="934"> <path d="M0 0 C300.3 0 600.6 0 910 0 ... Z " fill="#928299"/> <path d="M0 0 C1.484 0.99000001 ..." fill="#A95869"/> <path d="M0 0 C1.320 0.659999999 2.6399 ..." fill="#A95869"/> <path d="M0 0 C4.151643338387942 0 7 ..." fill="#A95869"/> <path d="M0 0 C-0.6599999999999682 0 7 ..." fill="#A95869"/>

[Figure 61]

... </svg>

- Figure 2. (Left) Image Vectorization results using StarVector-8B, LIVE, and VTracer. Each row shows the input image, generated SVGs, and pixel-wise difference maps to highlight accuracy. StarVector-8B better preserves shapes, color gradients, and text, despite minor misplacements. Notably, MSE often misaligns with visual quality, e.g., regarding the ‘planet’ example, StarVector’s MSE (0.009) is higher than LIVE’s (0.0012) and VTracer’s (0.0039), yet StarVector preserves the color gradient. For the ‘diagram’ example, StarVector preserves the text. DinoScore better reflects these details, consistently favoring StarVector. (Right) Curve vs Primitive-based Vectorization. SVG code generated by StarVector and VTracer for the given image. StarVector effectively leverages shape primitives, resulting in a compact vectorization. VTracer decomposes the image into numerous paths, resulting in a more complex result with less semantic clarity.

has been previously unexplored and remains a core challenge in modern vectorization.

Vectorization approaches fall into two main categories: traditional image processing and deep learning (DL)-based methods. Image processing techniques [51, 59, 87] trace vector curves via pixel-level analysis but often generate overly complex representations with artifacts and lack semantic understanding (Figure 2). DL methods [13, 14, 67, 89] improve vector modeling using latent variable models and differentiable rendering [43, 50] but struggle with generalization and underutilize SVG primitives. This limits their effectiveness for complex SVGs like scientific diagrams and hinders their application in modern multimodal tasks such as text-driven SVG generation [24, 65, 72, 96].

Recent advancements in Multimodal Large Language Models (MLLMs) [2, 45] have integrated visual understanding into transformer [84] architectures while demonstrating strong code generation capabilities [1, 42, 49, 55]. Building on these developments, we introduce image vectorization as an inverse rendering and code generation task, leveraging MLLMs to generate SVG code directly from input images. This approach naturally encompasses the full range of SVG primitives, enhancing both semantic understanding and generation capabilities (Table 6).

[Figure 62]

We introduce StarVector, a foundational MLLM for SVG generation. StarVector processes both images and text instructions to produce compilable SVG code, leveraging SVG primitives to accurately represent vector graphics. We build upon the StarCoder works [42, 49] to connect the code generation research with SVG generation. Figure 3 describes the model architecture. It integrates an image encoder that projects images into visual tokens, and a transformer language model for learning the relationships between instructions, visual features, and SVG code sequences, to perform image vectorization (Image-to-SVG) or

text-driven SVG Generation (Text-to-SVG) tasks. StarVector performs primitive-aware vectorization through learned semantic understanding, effectively leveraging SVG primitives without explicit pixel reconstruction objectives. To address the lack of large-scale SVG datasets for training StarVector, we introduce SVG-Stack, with over 2M SVG samples paired with rendered images and text descriptions.

Additionally, we find that conventional metrics like MSE fail to adequately assess vector graphics quality, as demonstrated in Figure 2. Instead, we propose DinoScore, a perceptual similarity metric that better correlates with visual human perception, and introduce SVG-Bench, an evaluation framework spanning 10 datasets and 3 tasks: Image-toSVG, Text-to-SVG, and diagram generation.

###### Contributions

- 1. We introduce StarVector, an MLLM capable of image vectorization and text-driven SVG Generation, uniquely preserving SVG primitives rather than producing multiple curves—a previously unexplored skill.

[Figure 63]

- 2. We create SVG-Stack, a large-scale dataset with 2M samples, supporting Image-to-SVG and Text-to-SVG.

[Figure 64]

- 3. We develop SVG-Bench, an MLLM benchmark with 10 datasets across 3 SVG tasks.

[Figure 65]

- 4. We conduct extensive experiments and evaluations, including human assessments, demonstrating StarVector’s advantages in primitive recognition and highlighting the limitations of pixel-wise metrics for SVG evaluation.

###### 2. Related Work

SVG Generation Methods. Early image vectorization efforts [92] primarily relied on traditional techniques [21, 44, 95] such as segmentation and polynomial curve fitting [43, 51, 59, 87]. With advancements in deep learn-

ing, new approaches have emerged. For instance, SVGVAE [47] employs a class-conditional Variational Autoencoder (VAE) [34] to predict a latent style vector and generate SVGs using an LSTM decoder [30]. DeepSVG [14] introduces a hierarchical VAE architecture using transformers for SVG path representation, while Im2Vec [67] converts pixel images into latent representations that can be decoded into paths with a recurrent neural network. However, these methods are limited to basic primitives like paths, resulting in a performance gap compared to traditional image processing methods. Concurrently, BeyondPixels [99] utilizes LLMs to predict command sequences in synthetic SVGs. These trends highlight the potential of LLM models for improved generalization in SVG generation.

Recent results in image generation using diffusion [29, 73] or autoregressive [23, 64, 96] models and the success of text-to-image generation [6, 65, 72] have inspired the research into text-conditioned SVG generation. VectorFusion [33] leverages a strong text-to-image diffusion model to find the SVG via iterative optimization. Some work has focused on SVG editing from textual inputs [12]. CLIPasso [86] uses a CLIP distance loss to iteratively refine SVG from sketches. These solutions are slow due to their iterative nature. IconShop [94] trains a BERT [20] model for text-conditioned SVG generation of icons, but their method is restricted to using paths.

In contrast, we train an MLLM on the inverse rendering image vectorization task, leveraging visual understanding to produce accurate SVG code with optimal use of primitives.

SVG Benchmarks and Datasets. Previous work on SVG datasets and benchmarks has been limited. Existing Imageto-SVG datasets mainly focus on fonts, icons, and emojis [12–14, 67, 89, 94], offering limited variety for broader SVG types and primitives. Recent text-driven SVG generation efforts [12, 16, 94] leverage large language models [56] for synthetic captions, but limited dataset accessibility hampers reproducibility. Evaluation methods also face challenges. Existing pixel-based metrics like MSE [14, 47, 67] fail to capture the fidelity and structure of SVGs, overlooking aspects like line definition and primitive usage (Figure 2, illustrates how MSE can be misleading). Recent benchmarking efforts [54, 100] focus on caption-based generation or editing but remain limited in scope.

To address these gaps, we introduce SVG-Stack, a largescale dataset for diverse SVG generation tasks from both images and text, and SVG-Bench, a unified benchmark with datasets, tasks, and metrics tailored for SVGs. Together, they provide a solid foundation for improving SVG model training and evaluation.

Multimodal Language Models. Large language models (LLMs) have achieved great success in natural language processing (NLP) [10, 55, 61, 82–84], especially

in code generation tasks [9, 15, 17, 26, 32, 35, 42, 53]. Recent trends in Multimodal Large Language Modeling (MLLM) [2, 45, 56, 88] have allowed infusing image understanding into text-only models, for tasks like visualquestion answering (VQA) [4] or image captioning [39– 41]. Current approaches define an image encoder for computing visual tokens from images, that can be processed by an LLM [2, 45, 56, 88]. Multiple works use a Vision Transformer [22] backbone with pre-trained weights like CLIP [62]. We get inspiration from these architectures for training an MLLM on the tasks of image vectorization and text-driven SVG Generation (also referred to as Image-toSVG and Text-to-SVG).

###### 3. SVG-Stack Dataset

[Figure 66]

To address the lack of large-scale SVG datasets for training foundational SVG models, we introduce SVG-Stack, a dataset containing 2.1 million SVG samples for training, 108k for validation, and 5.7k for testing. Each SVG is paired with its corresponding raster image and descriptive text, making the dataset ideal for multi-modal learning. SVG-Stack is sourced from The Stack [35], a diverse collection of code samples from various software languages. Our selection builds upon the initial filtering and de-duplication processes conducted in [3, 35, 42]. We perform additional filtering to ensure non-duplicate SVG samples.

These samples come from publicly available repositories on GitHub, providing a rich variety of real-world SVGs used across websites, graphic designs, and more. A key advantage of our approach is the inclusion of SVGs with diverse syntactic structures, varying templating approaches, different header formats, framework-specific implementations, and full support for SVG primitives. This diversity significantly enhances model generalization capabilities. This approach contrasts with previous datasets that primarily focused on a narrow subset of SVG types [14, 47, 94]. SVG-Stack represents the first large-scale pretraining dataset for SVG generation, with permissively licensed samples [35], bringing together a broad spectrum of SVGs that closely mirror the diversity seen across the Web. Table 5 and Figures [7, 8, 6] in Appendix 8 provide a comparison of available SVG datasets, showcasing the breadth of test examples found in SVG-Stack.

Data Processing and Curation. Our data processing begins with extracting SVG samples from TheStack [35], followed by a comprehensive deduplication process based on filenames, SVG code, and metadata. We utilized CairoSVG [36] for rasterization, removing samples that produced completely white images. To optimize sequence length and improve visualization, we eliminated comments and XML headers from the SVG code.

###### Synthetic Generation of Text Instructions. To enable

###### a) StarVector Architecture, Training and Inference b) Computing Visual Tokens

[Figure 67]

Input Image Input Text Instruction SVG Code (ground truth)

Input Image (Patch Grid)

<svg xmlns="..." <path style="..."

A folder icon with a white label on it.

d="M 66.132812 0 L 15.234375..."/> </svg>

Linear Vision Transformer

Tokenizer and Embedder Adapter Image Encoder

During training, the model observes sequences of (text, svg) or (image, svg) pairs

Visual Tokens Text Embeddings SVG Embeddings

Adapter

... ... ...

⊕ ||

Linear Swish

Linear

Language Model (LLM)

LayerNorm

During inference, the LLM predicts the SVG code sequentially, conditioned on visual tokens or text instructions

<svg xmlns="..."

...

<path d="M 35.13 37.23..."/> </svg>

Generated SVG Code Raster images

Visual Tokens

- Figure 3. a) StarVector Architecture. Images are projected into visual tokens via an Image Encoder and Adapter, aligned with the Language Model’s hidden space. Text conditioning uses the LLM’s tokenizer and embedder. ⊕ denotes mutually exclusive addition of image or text features to SVG tokens, while ∥ indicates sequence concatenation. During training, the model maps token sequences (visual or textual) to SVG code, and at inference, SVG code is generated sequentially. b) Visual Token Computation. We use a Vision Transformer (ViT) to process image patches into hidden features, which are projected through a non-linear Adapter to form visual tokens.

Text-to-SVG generation, we augment our 2M samples with synthetically generated textual captions describing the rasterized images. We employ open-source image captioning models, specifically BLIP2 [41] and Llava [45], resulting in a comprehensive dataset of 4 million paired textual captions and SVGs. Detailed information about the annotation process is available in Appendix 8.4.

Data Augmentation. We implement several SVG-specific augmentation techniques to enhance model robustness. Rather than storing static-resolution raster images, we perform rasterization during data collation, enabling dynamic SVG transformations without significant training overhead. Our augmentation pipeline includes modifications to image resolution, rotation, translation, scaling, and color properties. We leverage open-source libraries svgpathtools [74] and bs4 [68] for SVG manipulation. Ablation studies demonstrate that these augmentation techniques significantly improve model performance.

- 4. StarVector 4.1. Architecture

puts. This capability emerges as the model learns to predict SVG code directly, while it is trained on a large SVG dataset (SVG-Stack). We frame the task of image vectorization as an inverse rendering and code generation problem, where images are represented as visual tokens that precede the sequence of SVG code tokens. During generation, an image is converted into visual tokens, prompting StarVector to predict the SVG code following a vectorization trigger token. As depicted in Figure 3, our architecture employs a large language model (LLM) fϕ and an image encoder eθ, parameterized by trainable parameters ϕ and θ, respectively.

Visual Tokens. For each input image xv, the Image Encoder provides flattened grid features zv = eθ(xv). A Vision Transformer (ViT) [62] is utilized to define eθ(·). All features from the last transformer layer are used, as we require high visual expressivity. LLM Adapter gφ is devoted to projecting the visual features zv into the dimensionality of the LLM, creating visual tokens hv:

###### hv = gφ(zv),where zv = eθ(xv) (1)

[Figure 68]

As depicted in Figure 3-b, the LLM Adapter gφ performs a non-linear projection of the image embeddings into the LLM embedding space, producing a set of visual token embeddings (or visual tokens). This transformation aligns the image representations with the LLM, effectively bridging the visual and SVG code modalities. The Adapter is

StarVector is a foundational MLLM for SVG generation, trained for Image-to-SVG and Text-to-SVG Generation. It effectively uses image semantics to identify and utilize shape primitives, producing precise and compact SVG out-

composed of a sequence of fully connected (FC) layers with Swish [63] activation function and Layer Normalization [38]. We initialize the adapter parameters φ using the normal distribution. We initialize the image encoder parameters θ using the public weights of CLIP ViT-L/14 [62].

gφ(zv) = LayerNorm(WL · Swish(Wh · zv)) (2)

Language Modeling. For each image xv, we define xt as its corresponding textual caption and xs as the SVG code. For each sample, we have a tuple (xv,xt,xs). Training sequences are constructed by concatenation: (xv,xs) for Image-to-SVG tasks and (xt,xs) for Text-to-SVG tasks. To simplify, we use xc to represent the conditioning sequence, which is either xv or xt, depending on the task.

As depicted in Figure 3, both textual xt and SVG xs sequences are processed by a tokenizer and an embedder, which converts text strings to tokens and tokens to embeddings. This embedding operation has trainable parameters. We model the conditional probability of SVG sequences as:

p(xs | xc) =

L

p(xs,i | xs,<i,xc), (3)

i=1

where L is the length of the SVG sequence xs. This formulation allows us to use a generative objective with nexttoken cross-entropy over the SVG sequence. During inference, only the conditioning sequence xc is given as input, and the SVG code is sampled autoregressively until the ending <svg-end> token is reached.

StarVector Variants. We define two variants of StarVector to explore its scaling behavior, varying in image resolution, LLM parameter count, and context length.

- 1. StarVector-1B is initialized with a CLIP ViT-B/32 [62] image encoder, processing images at a 224×224 resolution to produce 257 visual tokens. The LLM uses StarCoder-1B [42] with a context length of 8192 tokens.
- 2. StarVector-8B employs a SigLip (siglip-so400mpatch14-384) [97] image encoder, processing images at a 384×384 resolution to yield 576 visual tokens. This model utilizes StarCoder2-7B, offering an expanded context length of 16k tokens. With a total of 8B parameters, we investigate how scaling can yield more precise and compact SVGs due to higher image resolution and enhanced LLM capacity.

Inference. During generation, StarVector processes an input image or text, converting it into tokens, and then predicts subsequent tokens auto-regressively to produce SVG code. This code is rasterized with CairoSVG [36], generating an image. A key challenge during generation is ensuring both 1) syntactically valid SVGs and 2) SVGs optimized for compactness and precision. Decoding introduces

stochasticity, and limited context length can result in incomplete SVGs. We find that StarVector is sensible to temperature, length penalty, and logit bias, i.e., adding more weight to certain tokens, like the <svg-end> token which encourages valid SVG outputs (properly closed SVG). We introduce them as inference hyperparameters. To further improve quality, we generate k samples with varied parameters, ranking them based on DinoScore.

###### 5. SVG-Bench: Evaluation Suite for SVG

In response to the limited benchmarks for SVG evaluation and to unify evaluation practices [14, 47, 67], we introduce SVG-Bench. This benchmark assesses SVG methods across Image-to-SVG, Text-to-SVG, and Diagram Generation tasks, encompassing various SVG types from simple graphics like icons and fonts to complex diagrams with multiple primitives. Dataset statistics are presented in Table 5.

###### 5.1. Tasks and Benchmarks

SVG-Bench focuses on the following tasks. For details on dataset curation and visual examples, see Appendix 8.

- 1. Image-to-SVG: This task evaluates converting images to SVGs across varying complexities. We introduce SVG-Fonts, SVG-Emoji, SVG-Icons, and SVG-Stack, increasing in complexity. While prior works used similar datasets [14, 47, 67], access has often been unclear. We provide standardized train, validation, and test splits, along with simplified versions containing only paths for compatibility with certain methods.
- 2. Text-to-SVG: We evaluate the model’s ability to generate SVGs from text instructions. This includes the SVGStack test set, which provides two textual descriptions per image, and the SVG-FIGR dataset, which is sourced from FIGR-8-SVG [16, 94] dataset, enabling the generation of simpler (path-only) icons from text.
- 3. Diagram Generation: We assess the model’s performance in generating diagrams, a specific type of SVG that involves text, rectangles, and arrow primitives. For this, we create the SVG-Diagrams test set by extracting samples from SVG-Stack, including textual captions.

Evaluation Metrics. To compute benchmark scores, we define the following metrics: For image vectorization tasks, we use Mean Squared Error (MSE), Structural Similarity Index (SSIM) [90, 91], and Learned Perceptual Image Patch Similarity (LPIPS) [98]. To address the limitations of pixelbased metrics (see Figure 2), we propose DinoScore [57], which computes L2 distance between DinoV2 features. Token Length (Tokens) measures the size of the SVG samples. We use the StarCoder [42] tokenizer to tokenize SVG code and compute the average length. These metrics are also used for Diagram Generation. For Text-to-SVG, we build

- Table 1. Image Vectorization Results. Image processing methods (denoted by †) excel in pixel-based metrics (SSIM and MSE) while StarVector models lead in semantic-based metrics (DinoScore and LPIPS). StarVector shows better performance in SVG-Stack, SVGFonts, and SVG-Icons but underperforms in SVG-Emoji due to limited training data. We highlight the token lengths of generated SVGs from different models, comparing them to the average token count in test examples (shown in gray below the “Tokens” header). Token counts close to the actual number are marked in green, while the largest counts are highlighted in red. Notably, methods that perform well on MSE tend to utilize a large number of tokens, whereas StarVector shows remarkable compression.

SVG-Stack SVG-Fonts SVG-Icons SVG-Emoji

Method

Dino

↑LPIPS

↓

SSIM

↑

MSE

↓Tokens

2,822 Dino

↑LPIPS

↓

SSIM

↑

MSE

↓Tokens

3,136 Dino

↑LPIPS

↓

SSIM

↑

MSE

↓Tokens

3,305 Dino

↑LPIPS

↓

SSIM

↑

MSE

↓Tokens

5,618

AutoTrace† 0.942 0.063 0.930 0.009 59.1k 0.954 0.025 0.968 0.006 30.8k 0.946 0.053 0.937 0.014 56.7k 0.975 0.077 0.902 0.011 94.0k Potrace† 0.898 0.139 0.856 0.036 7.5k 0.967 0.009 0.988 0.002 4.2k 0.972 0.023 0.973 0.004 12.0k 0.882 0.267 0.780 0.067 9.7k VTracer† 0.954 0.062 0.883 0.010 9.7k 0.964 0.027 0.888 0.009 4.5k 0.940 0.062 0.914 0.017 20.0k 0.981 0.074 0.894 0.008 15.7k Im2Vec 0.692 0.291 0.765 0.181 4.3k 0.733 0.140 0.837 0.135 4.3k 0.754 0.150 0.889 0.055 4.3k 0.732 0.465 0.774 0.126 3.8k LIVE 0.934 0.059 0.953 0.003 18.3k 0.956 0.013 0.977 0.001 18.3k 0.959 0.035 0.973 0.004 18.2k 0.969 0.060 0.958 0.002 18.3k DiffVG 0.810 0.156 0.856 0.019 19.7k 0.821 0.051 0.959 0.007 19.7k 0.952 0.056 0.956 0.015 19.8k 0.814 0.242 0.776 0.034 19.7k GPT-4-V 0.852 0.317 0.711 0.195 443 0.842 0.198 0.749 0.197 279 0.848 0.238 0.755 0.144 524 0.850 0.344 0.712 0.170 672

StarVector-1B 0.926 0.149 0.840 0.078 3.7k 0.978 0.022 0.961 0.022 2.4k 0.975 0.040 0.931 0.026 3.5k 0.929 0.217 0.820 0.063 4.8k StarVector-8B 0.966 0.058 0.947 0.026 5.3k 0.982 0.030 0.946 0.029 3.0k 0.984 0.035 0.975 0.012 2.8k 0.943 0.193 0.829 0.052 6.7k

SVG-Stacksim SVG-Fontssim SVG-Iconssim SVG-Emojisim AutoTrace† 0.945 0.063 0.922 0.018 74.1k 0.928 0.125 0.886 0.050 1.5k 0.915 0.111 0.901 0.044 1.3k 0.963 0.090 0.874 0.029 134.8k Potrace† 0.970 0.022 0.968 0.006 12.2k 0.991 0.012 0.983 0.003 7.7k 0.983 0.025 0.976 0.004 10.4k 0.992 0.037 0.951 0.008 26.7k VTracer† 0.935 0.061 0.914 0.020 16.0k 0.946 0.040 0.939 0.013 12.7k 0.945 0.043 0.946 0.012 11.9k 0.948 0.063 0.911 0.021 16.2k Im2Vec 0.725 0.186 0.892 0.046 4.3k 0.857 0.184 0.833 0.096 284 0.860 0.207 0.792 0.129 453 0.695 0.179 0.898 0.045 3.7k LIVE 0.963 0.039 0.974 0.005 18.3k 0.975 0.016 0.991 0.001 18.3k 0.961 0.030 0.978 0.003 18.2k 0.958 0.075 0.934 0.014 18.2k DeepSVG 0.907 0.192 0.835 0.071 1.5k 0.928 0.125 0.886 0.050 1.5k 0.915 0.111 0.901 0.044 1.3k 0.822 0.209 0.841 0.074 1.8k GPT-4 V 0.874 0.226 0.768 0.137 329 0.946 0.040 0.939 0.013 12.7k 0.945 0.043 0.946 0.012 11.9k 0.852 0.212 0.802 0.105 424

StarVector-1B 0.954 0.089 0.870 0.053 2.9k - - - - - - - - - - 0.977 0.073 0.897 0.043 3.0k StarVector-8B 0.977 0.074 0.888 0.045 2.1k 0.993 0.012 0.970 0.009 1.3k 0.990 0.024 0.947 0.017 2.7k 0.903 0.163 0.791 0.091 3.2k

- Table 2. Image Vectorization on SVG-Diagrams. StarVector outperforms LIVE in DinoScore, LPIPS, and SSIM, while LIVE ranks best in MSE. However, visual results (Fig. 2) confirm that StarVector is the only effective method for SVG generation, underscoring the misalignment of MSE. Additionally, it remains competitive in terms of token length.

Table 3. Usage of Paths and Inference Time. We ablate the use of the path primitive across models. LIVE and DiffVG allow setting the number of paths, while VTracer, Autotrace, and StarVector dynamically determine them. More paths generally improve performance. LIVE achieves the best pixel metrics, but StarVector excels in DinoScore. We also report average inference time per sample, noting that LIVE is significantly slower. Results are averaged across SVG-Bench datasets.

SVG-Diagrams Method Dino ↑ LPIPS ↓ SSIM ↑ MSE ↓ Tokens ↓

Method # Paths Dino ↑ LPIPS ↓ SSIM ↑ MSE ↓ Time (s) ↓

LIVE 5 0.898 0.137 0.881 0.013 190 16 0.930 0.064 0.937 0.006 290 32 0.937 0.057 0.944 0.004 650 60 0.939 0.053 0.947 0.003 1,412

Autotrace† 0.874 0.114 0.883 0.013 90.6k Potrace† 0.875 0.153 0.862 0.026 22.6k VTracer† 0.882 0.116 0.877 0.011 15.8k LIVE 0.870 0.121 0.859 0.010 18.3k DiffVG 0.822 0.170 0.859 0.019 19.8k

DiffVG 15 0.781 0.205 0.819 0.066 21

60 0.844 0.135 0.881 0.018 31 120 0.895 0.107 0.907 0.013 45

StarVector-1B 0.943 0.107 0.862 0.032 9.5k StarVector-8B 0.959 0.093 0.890 0.027 -

Vtracer 18 0.942 0.067 0.892 0.011 0.09 Potrace - 0.937 0.109 0.897 0.024 10 AutoTrace 3k 0.951 0.065 0.924 0.010 1

StarVector-1B 8 0.952 0.107 0.883 0.044 41 StarVector-8B 10 0.963 0.085 0.911 0.031 74

on text-to-image literature [62, 66, 73, 81] and prior Textto-SVG methods [12, 94], using FID [81], FID-CLIP [94], and CLIP Score [62] to measure image-text alignment.

###### 6. Experiments and Results

We train StarVector (1B and 8B versions) on the inverse rendering vectorization task using SVG-Stack dataset. We

then fine-tune on the other datasets mentioned in Section 5, as well as for Text-to-SVG task. We evaluate StarVector and other methods on SVG-Bench, focusing on quantitative and qualitative performance, SVG primitive use, and compact-

Table 4. Results on Text-to-SVG: We report FID, FID-CLIP (FID-C), and CLIP Score (CLIP) on SVG-Stack and SVG-FIGR. StarVector models outperform all previous baselines in all metrics. We observe improvement when scaling StarVector from 1B to 8B. Results for DeepSVG+GAN [14, 28] and Bert on SVG-FIGR are extracted from [94], while StarVector is trained on the same data and splits. Missing scores are due to limited model access.

SVG-FIGR SVG-Stack Method FID ↓ FID-C ↓ CLIP ↑ FID ↓ FID-C ↓ CLIP ↑

DeepSVG+GAN - 12.011 21.783 - - Bert - 35.104 22.035 - - IconShop - 4.657 25.746 - - GPT-4 32.953 19.026 26.088 37.381 9.664 26.228 CodeLlama 29.002 22.536 26.227 34.777 11.152 25.532

StarVector-1B 15.263 3.834 26.342 28.374 6.482 29.372 StarVector-8B 10.067 1.308 27.366 25.828 4.645 31.307

ness. The following sections present the experimental setup and results. Ablation studies on the architecture, data augmentation, and generation can be found in Appendix 10.1.

Baselines. Baselines. For our comparisons, we consider the following model baselines: For Image-to-SVG, we evaluate top image processing algorithms such as Potrace [59], Vtracer [87], and Autotrace [51]. We also report on deep learning methods including DeepSVG (<5M parameters) [14], Im2Vec (<5M) [67], and MLLMs like GPT-4 Vision (>100B) [56]. For Text-to-SVG, we consider methods like IconShop (>1B), DeepSVG+GAN (<5M), and BERT (>1B), as outlined in [94]. Additionally, we evaluate LLMs such as CodeLlama-70b [82] and GPT-4 (>100B) [55]. For more details, see Appendix 9, and Table 6 summarizes the SVG capabilities of all methods.

Training and Inference. We train StarVector-1B with a batch size of 128 and StarVector-8B with 512, using a learning rate of 1e-5 and the AdamW optimizer. StarVector-1B took 7 days on 8 A100 GPUs, while StarVector-8B took 10 days on 64 H100 GPUs, both completing 2 epochs. Full training details are in Appendix 11.2. During inference, we generate k = 5 SVG outputs with temperatures ranging from 0 to 1, selecting the one with the highest DinoScore. We set a logit bias of 10 for the <svg-end> token and apply top-p nucleus sampling (0.9) with a length penalty of -0.5, using beam search with a size of 1. We utilize vLLM [37] as the backend to accelerate the generation process.

Experimental Setup. StarVector models are initialized as described in Section 4 and all their weights are unfrozen for the Image-to-SVG task using SVG-Stack. We then finetune these models on the train sets of SVG-Emoji, SVGFonts, and SVG-Icons, including their simplified versions. For the Text-to-SVG task, the image encoder is disregarded. Only the LLM is trained on SVG-Stack and FIRG-SVG.

We perform a comprehensive evaluation on SVG-Bench, analyzing the performance of baseline models mentioned above, alongside the StarVector models. We reproduce all baselines by training them on the benchmark’s training sets, within the limits of their availability.

###### 6.1. Main Results

Image Vectorization. Table 1 presents vectorization scores of models across 8 benchmarks. StarVector outperforms all other models in terms of DinoScore, achieving the highest score on six out of the eight benchmarks, thereby establishing its dominance in this metric. Results for the other metrics—LPIPS, SSIM, and MSE—are more varied. However, LIVE demonstrates superior performance on the SSIM and MSE metrics across all datasets. It is important to note that models that perform well on MSE tend to generate larger SVG files, as indicated by the number of tokens in the SVG code. This creates overly complex vectors with visible artifacts (see Fig. 4). Specifically, LIVE’s SVG outputs average around 18k tokens, while VTracer varies between 4.5k and 20k tokens. In contrast, StarVector averages approximately 3k tokens, closely matching the ground truth token count. This efficiency is primarily attributed to StarVector’s effective use of SVG primitives, as illustrated in Figure 2 (right).

Figure 4 presents qualitative results of models, on SVGStack and SVG-Diagrams. In terms of visual quality, LIVE, VTracer, and AutoTrace produce artifacts, especially when dealing with small details. Potrace offers more sharp results, but it is monochromatic. StarVector-8B produces superior results on shape preservation and definition. More results and qualitative samples can be found in Appendix 10.

Why is pixel-based MSE not well-suited? Our results reveal significant limitations of pixel-based metrics (MSE, SSIM, LPIPS) for SVG quality assessment. While StarVector shows worse MSE scores compared to other methods in Table 2, visual inspection of the results in Figures 4 and 14 (especially on diagrams) demonstrates StarVector’s superior quality. This discrepancy is particularly evident in the ‘planet’ example (Figure 2), where StarVector preserves color gradients and line definition, yet receives poor MSE scores due to minor pixel misalignments. Human evaluation (Fig. 5) shows a preference for StarVector’s outputs over other models, contradicting these metrics.

Pixel-based metrics have limitations due to (a) the prevalence of constant background colors (allowing even empty SVGs to score reasonably), (b) sensitivity to small spatial misalignments, and (c) inability to capture non-smooth artifacts at corners. DinoScore proves more reliable and aligns consistently with visual quality, scoring StarVector higher on well-formed samples while penalizing poorly formatted ones, thanks to robust self-supervised training [57].

###### Input Image StarVector-8B LIVE VTracer PoTrace AutoTrace

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

[Figure 98]

DinoScore: 0.9990 MSE: 0.0014

DinoScore: 0.9326 MSE: 0.0008

DinoScore: 0.9593 MSE: 0.0072

DinoScore: 0.7909 MSE: 0.1709

DinoScore: 0.9628 MSE: 0.0069

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

DinoScore: 0.9663 MSE: 0.0116

DinoScore: 0.8512 MSE: 0.0013

DinoScore: 0.9454 MSE: 0.0047

DinoScore: 0.7648 MSE: 0.1857

DinoScore: 0.9619 MSE: 0.0048

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

DinoScore: 0.9825 MSE: 0.0153

DinoScore: 0.9523 MSE: 0.0015

DinoScore: 0.9674 MSE: 0.0061

DinoScore: 0.8882 MSE: 0.0122

DinoScore: 0.9872 MSE: 0.0038

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

[Figure 174]

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

DinoScore: 0.9986 MSE: 0.0084

DinoScore: 0.8559 MSE: 0.0088

DinoScore: 0.9199 MSE: 0.0141

DinoScore: 0.9427 MSE: 0.0536

DinoScore: 0.9043 MSE: 0.0182

[Figure 189]

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

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

DinoScore: 0.9946 MSE: 0.0155

DinoScore: 0.9836 MSE: 0.0026

DinoScore: 0.8662 MSE: 0.0113

DinoScore: 0.9678 MSE: 0.0031

DinoScore: 0.8057 MSE: 0.0131

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

DinoScore: 0.9849 MSE: 0.0384

DinoScore: 0.9321 MSE: 0.0167

DinoScore: 0.8833 MSE: 0.0199

DinoScore: 0.9787 MSE: 0.0115

DinoScore: 0.9180 MSE: 0.0253

Figure 4. Image vectorization results on SVG-Stack (first 3 rows) and SVG-Diagrams (last 3 rows) test sets.

Diagram Generation. Table 2 shows results on SVGDiagrams, and Figures 4 and 14 provide visual examples. Results highlight that StarVector is the only method capable of performing diagram generation, as it uniquely applies the required primitives like rectangles, arrows, and text, whereas other methods produce blobs and curves that attempt to replicate structure and color. Metrics such as DinoScore, LPIPS, capture this advantage, while MSE and SSIM remain poorly aligned. Human evaluations further confirmed the preference for StarVector’s outputs.

SVG Primitive Usage. StarVector produces more compact SVGs by optimally using SVG primitives. This innovation combines visual semantic understanding and shape composition with direct SVG code generation, enabling decomposition into basic primitives. As shown in Figure 2 (right), StarVector efficiently represents shapes using primitives, while VTracer relies on a large collection of paths. See Appendix 10.4 for more examples (Figure 10) and SVG tag distribution analysis (Figure 30).

Human Evaluation. We conducted human evaluations comparing StarVector-8B with baseline results, involving participants from diverse backgrounds screened for conflicts of interest. Results in Figure 5 show a strong preference for StarVector-8B in all settings, particularly in the SVG-Diagrams tasks, highlighting a disconnect between pixel-based metrics (MSE, SSIM) and visual perception of SVG. While baselines prioritize pixel-perfect reconstruction, humans prefer StarVector’s sharp, well-defined shapes and effective use of primitives (Figure 30). Spearman correlations between model metrics (MSE and DINO) and human evaluation show weak correlations for MSE (0.0596

###### SVG-Stack

StarVector-8B vs LIVE-32P

- 53.8%

63.4%

- 54.9%

14.2%

13.9%

17.1%

Win

Tie

Both Good

StarVector-8B vs AutoTrace

11.1%

10.7%

14.8%

Lose

StarVector-8B vs Vtracer

12.9%

12.4%

20.8%

SVG-Diagrams

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |71.1%| | | | |18.| |9%| | | | |7.8%|
| | | | | | | | | | | | | | |
| |73.5%| | | | | |15|.9%| |10.6%| | | |
| | | | | | | | | | | | | | |
| |58.6%| | | |30.3%| | | | | | |8.5%| |
| | | | | | | | | | | | | | |

StarVector-8B vs LIVE-32P

StarVector-8B vs AutoTrace

StarVector-8B vs Vtracer

0 20 40 60 80 100

Percentage (%)

Figure 5. Human evaluation. StarVector-8B was evaluated against top-performing baselines: LIVE, AutoTrace, and Vtracer. Results consistently showed a strong human preference for SVG outputs generated by StarVector-8B over the baselines. In total, 1,948 evaluations were collected from 30 unique participants.

and -0.1002), indicating it is not a strong predictor. In contrast, DinoScore exhibits stronger correlations, with values of -0.6193 and 0.6214, and a strong correlation of 0.7577 between DINO differences and human evaluation differences, highlighting DINO as a more reliable metric.

Text-to-SVG. Table 4 shows StarVector outperforms baselines on SVG-FIGR and SVG-Stack. While qualitative results show potential, semantic accuracy often suffers due to data quality (see Appendix 10.2).

###### 7. Conclusions

We introduced StarVector, an MLLM that excels in Imageto-SVG and Text-to-SVG generation, delivering superior visual quality, precise line and shape rendering, and optimal SVG primitive usage compared to baselines. To train StarVector, we created SVG-Stack, a diverse dataset that enables generalization across SVG types and primitives. Additionally, we developed SVG-Bench, a unified benchmark with tasks, datasets, and targeted metrics. Our study shows that traditional metrics inadequately capture SVG quality, leading us to propose more effective alternatives.

Limitations and Future Work. StarVector is constrained by its 16k token context, which is inadequate for complex SVGs. It relies primarily on code prediction with minimal visual feedback. Additionally, generation speed is limited by the LLM. Future work may investigate integrating pixel-based signals to overcome these challenges.

Acknowledgements. We sincerely thank Ghazwa Darwiche, Christian Hudon, Fanny Rancourt, and Marie-Ève Marchand for their invaluable administrative and technical support. This work was supported by the Natural Sciences and Engineering Research Council of Canada and Mitacs. Chris Pal acknowledges the Canada CIFAR AI Chair. We also appreciate the human verifiers for their contributions to assessment and data verification.

###### References

- [1] The claude 3 model family: Opus, sonnet, haiku. 2
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35: 23716–23736, 2022. 2, 3, 28
- [3] Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988, 2023. 3
- [4] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015. 3
- [5] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 40

- [6] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 3
- [7] Jonas Belouadi, Anne Lauscher, and Steffen Eger. Automatikz: Text-guided synthesis of scientific vector graphics with tikz. arXiv preprint arXiv:2310.00367, 2023. 1, 42
- [8] Jonas Belouadi, Simone Paolo Ponzetto, and Steffen Eger. Detikzify: Synthesizing graphics programs for scientific figures and sketches with tikz. arXiv preprint arXiv:2405.15306, 2024. 1
- [9] Ekaba Bisong and Ekaba Bisong. Google bigquery. Building Machine Learning and Deep Learning Models on Google Cloud Platform: A Comprehensive Guide for Beginners, pages 485–517, 2019. 3
- [10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3, 18
- [11] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-

4. arXiv preprint arXiv:2303.12712, 2023. 18, 19

- [12] Mu Cai, Zeyi Huang, Yuheng Li, Haohan Wang, and Yong Jae Lee. Leveraging large language models for scalable vector graphics-driven image understanding. arXiv preprint arXiv:2306.06094, 2023. 3, 6, 40
- [13] Defu Cao, Zhaowen Wang, Jose Echevarria, and Yan Liu. Svgformer: Representation learning for continuous vector graphics using transformers. In Proceedings of the

- IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10093–10102, 2023. 2, 13
- [14] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems, 33:16351–16361, 2020. 2, 3, 5, 7, 13, 14, 15, 18, 29
- [15] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 3
- [16] Louis Clouâtre and Marc Demers. Figr: Few-shot image generation with reptile. ArXiv, abs/1901.02199, 2019. 3, 5
- [17] Arghavan Moradi Dakhel, Vahid Majdinasab, Amin Nikanjam, Foutse Khomh, Michel C Desmarais, and Zhen Ming Jack Jiang. Github copilot ai pair programmer: Asset or liability? Journal of Systems and Software, 203:111734,

2023. 3

- [18] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022. 41
- [19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 40
- [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 3
- [21] James Richard Diebel. Bayesian Image Vectorization: the probabilistic inversion of vector image rasterization. Stanford University, 2008. 2
- [22] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3, 40
- [23] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 3, 26, 28, 39, 40
- [24] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning,

2024. 2

- [25] Jon Ferraiolo, Fujisawa Jun, and Dean Jackson. Scalable vector graphics (SVG) 1.0 specification. iuniverse Bloomington, 2000. 1
- [26] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset

- of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020. 3
- [27] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249–256. JMLR Workshop and Conference Proceedings, 2010. 40
- [28] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 7
- [29] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [30] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997. 3
- [31] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019. 27, 41
- [32] Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436, 2019. 3
- [33] Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1911–1920, 2023. 1, 3
- [34] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [35] Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, et al. The stack: 3 tb of permissively licensed source code. arXiv preprint arXiv:2211.15533, 2022. 3, 14
- [36] Kozea. Cairosvg. https://cairosvg.org/, 2023. 3, 5
- [37] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023. 7
- [38] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. ArXiv e-prints, pages arXiv–1607,

2016. 5

- [39] Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, et al. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. arXiv preprint arXiv:2205.12005, 2022. 3
- [40] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.

- [41] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 3, 4, 14
- [42] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023. 2, 3, 5, 39, 41
- [43] Tzu-Mao Li, Michal Lukáˇc, Michaël Gharbi, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG), 39(6):1–15, 2020. 1, 2, 17, 18, 29
- [44] Zicheng Liao, Hugues Hoppe, David Forsyth, and Yizhou Yu. A subdivision-based representation for vector image editing. IEEE transactions on visualization and computer graphics, 18(11):1858–1867, 2012. 2
- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 2, 3, 4, 14, 28
- [46] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976– 11986, 2022. 26, 28, 39, 40
- [47] Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7930–7939, 2019. 3, 5, 13, 14
- [48] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 41
- [49] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024. 2, 39
- [50] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. Towards layerwise image vectorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16314–16323, 2022. 1, 2, 15, 17, 40
- [51] Martin Weber. Autotrace. https://github.com/ autotrace/autotrace, 2024. 1, 2, 7, 16, 17
- [52] Kenton Murray and David Chiang. Correcting length bias in neural machine translation. arXiv preprint arXiv:1808.10006, 2018. 41
- [53] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474, 2022. 3, 39
- [54] Kunato Nishina and Yusuke Matsui. Svgeditbench: A benchmark dataset for quantitative assessment of llm’s svg editing capabilities. arXiv preprint arXiv:2404.13710,

2024. 3

- [55] OpenAI. Gpt-4 technical report, 2023. 2, 3, 7, 19

- [56] OpenAI. GPT-4V(ision) System Card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. Accessed: 2023-11-05. 3, 7, 16, 18, 29

- [57] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 5, 7
- [58] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In NeurIPS-W, 2017. 41
- [59] Peter Selinger. Potrace. https://github.com/ tatarize/potrace, 2024. 1, 2, 7, 16, 17
- [60] Antoine Quint. Scalable vector graphics. IEEE MultiMedia, 10(3):99–102, 2003. 1
- [61] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. openAI, 2018. 3
- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 4, 5, 6, 26, 28, 39, 40
- [63] Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017. 5
- [64] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 3
- [65] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation, 2021. 2, 3
- [66] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 6
- [67] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. Im2vec: Synthesizing vector graphics without vector supervision. arXiv preprint arXiv:2102.02798,

2021. 2, 3, 5, 7, 15, 17, 29

- [68] Leonard Richardson. Beautifulsoup. https://www. crummy.com/software/BeautifulSoup/, 2023. 4
- [69] Juan A Rodriguez, David Vazquez, Issam Laradji, Marco Pedersoli, and Pau Rodriguez. Figgen: Text to scientific figure generation. arXiv preprint arXiv:2306.00800, 2023. 1
- [70] Juan A Rodriguez, David Vazquez, Issam Laradji, Marco Pedersoli, and Pau Rodriguez. Ocr-vqgan: Taming textwithin-image generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3689–3698, 2023. 1

- [71] Juan A Rodriguez, Nicholas Botzer, David Vazquez, Christopher Pal, Marco Pedersoli, and Issam Laradji. Intentgpt: Few-shot intent discovery with large language models. arXiv preprint arXiv:2411.10670, 2024. 19
- [72] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 2, 3
- [73] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3, 6, 26
- [74] Andy S. svgpathtools. https://github.com/ mathandy/svgpathtools, 2023. 4
- [75] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 39, 40
- [76] Louis Shao, Stephan Gouws, Denny Britz, Anna Goldie, Brian Strope, and Ray Kurzweil. Generating high-quality and informative conversation responses with sequence-tosequence models. arXiv preprint arXiv:1701.03185, 2017. 41
- [77] Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019. 41
- [78] Benjamin Spector and Chris Re. Accelerating llm inference with staged speculative decoding. arXiv preprint arXiv:2308.04623, 2023. 41
- [79] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014. 40
- [80] Chengjun Tang, Kun Zhang, Chunfang Xing, Yong Ding, and Zengmin Xu. Perlin noise improve adversarial robustness. arXiv preprint arXiv:2112.13408, 2021. 14
- [81] Lucas Theis, Aäron van den Oord, and Matthias Bethge. A note on the evaluation of generative models. arXiv preprint arXiv:1511.01844, 2015. 6
- [82] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3, 7, 19
- [83] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 19
- [84] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2, 3
- [85] Ashwin K Vijayakumar, Michael Cogswell, Ramprasath R Selvaraju, Qing Sun, Stefan Lee, David Crandall, and

- Dhruv Batra. Diverse beam search: Decoding diverse solutions from neural sequence models. arXiv preprint arXiv:1610.02424, 2016. 41
- [86] Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics (TOG), 41(4):1–11, 2022. 3
- [87] Vision Cortex. VTracer. https : / / www . visioncortex.org/vtracer- docs, 2023. 1, 2, 7, 16, 17
- [88] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 3
- [89] Yizhi Wang and Zhouhui Lian. Deepvecfont: Synthesizing high-quality vector fonts via dual-modality learning. ACM Transactions on Graphics (TOG), 40(6):1–15, 2021. 2, 3
- [90] Zhou Wang and Alan C Bovik. Mean squared error: Love it or leave it? a new look at signal fidelity measures. IEEE signal processing magazine, 26(1):98–117, 2009. 5
- [91] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5
- [92] Wikipedia. Comparison of raster-to-vector conversion software — Wikipedia, the free encyclopedia. http://en.wikipedia.org/w/index.php? title = Comparison % 20of % 20raster - to vector % 20conversion % 20software & oldid = 1185354750, 2024. [Online; accessed 07-March-2024]. 2
- [93] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019. 41
- [94] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. Iconshop: Text-based vector icon synthesis with autoregressive transformers. arXiv preprint arXiv:2304.14400, 2023. 3, 5, 6, 7, 14, 19, 38
- [95] Tian Xia, Binbin Liao, and Yizhou Yu. Patch-based image vectorization with automatic curvilinear feature alignment. ACM Transactions on Graphics (TOG), 28(5):1–10, 2009. 2
- [96] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022. 2, 3
- [97] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 5, 40
- [98] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of

- deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [99] Tong Zhang, Haoyang Liu, Peiyan Zhang, Yuxuan Cheng, and Haohan Wang. Beyond pixels: Exploring humanreadable svg generation for simple images with vision language models, 2023. 3
- [100] Bocheng Zou, Mu Cai, Jianrui Zhang, and Yong Jae Lee. Vgbench: Evaluating large language models on vector graphics understanding and generation. arXiv preprint arXiv:2407.10972, 2024. 3

[Figure 249]

### StarVector: Generating Scalable Vector Graphics Code from Images and Text

#### Supplementary Material

In the following sections, we provide additional details on the datasets used in this paper, present further experiments, and describe our baselines in detail. We also discuss the StarVector architecture, its training process, and the method for sampling SVG code from the model. Additionally, we provide more insights into SVG-Bench, including the proposed datasets and the different baselines within the evaluation setup. Finally, qualitative results are presented to showcase the strengths and limitations of our foundational model.

###### 8. SVG Datasets in SVG-Bench

Here we describe available SVG datasets in the recent literature. We extend our description of the datasets used for training and evaluating StarVector and other baselines. Earlier SVG datasets proposed in the literature (mainly datasets of emojis and fonts) were not easily accessible due to broken URLs and no direct entry point. Therefore, we provide them as part of SVGBench for easy reproducibility. We introduce splits for train, validation, and testing. The train set is used to optimize the parameter weights of the network. The validation is used for tuning sampling hyperparameters, and the test is used for evaluation. Our model can handle up to 8k context tokens. Therefore, our datasets only consider examples with up to 8,192 tokens. See Table 5 for a complete description of the datasets. See Figures 6, 7, 8 for ground truth examples of the test sets of SVG-Bench.

###### 8.1. Datasets with Simplified SVGs.

We create simplified versions of our four main datasets, i.e. emojis, icons, fonts, and SVG-Stack. This is done because DeepSVG [14] requires a simplification of the SVG in its input. The simplification consists of eliminating complex primitives and using only vector paths. Also, color and shapes are abstracted only to use simple line strokes.

###### 8.2. Creating the SVG-Fonts Dataset

To construct the SVG-Font dataset, we replicate the procedure described in SVG-VAE [47]1, which provides a list of public URLs containing open font packages. We download these packages, excluding any with broken URLs. The TTF files are then converted to SFD format, and we further use InkScape2 to convert them into SVG code. Samples from the test set are shown in Figure 7 (bottom-left). These samples contain only path elements and represent a

- 1https://github.com/magenta/magenta/tree/main/

magenta/models/svg_vae

- 2https://inkscape.org/

SVG-Diagrams

|| | |
|---|---|
|Server_Response<br><br>| |
|MegaDatabase_Money query MegaDatabase_Empty balance| |
| | |
<br><br>|Server_Request<br><br>| |
|---|---|
|List<Server_Response> query| |
| | |
| | |
|
|---|

|Printer Vendor<br><br>Samsung Elect...<br><br>(6.8%) Brother Indus...<br><br>(13.6%)<br><br>Canon (18.2%)<br><br>Seiko Epson (4.5%)<br><br>Others (20.5%)<br><br>Hewlett-Packard (36.4%)|
|---|

_mmheap::has_gparent

_mmheap::bubble_up_max

_mmheap::gparent

_mmheap::bubble_up_min

_mmheap::bubble_up

_mmheap::has_parent

_mmheap::min_level

_mmheap::parent

|H3C CH3 H3C CH3<br><br>CH3<br><br>H2N<br><br>H2N<br><br>N<br><br>N<br><br>N<br><br>HO<br><br>OH<br><br>OH<br><br>OH<br><br>O<br><br>OH<br><br>H<br><br>H<br><br>H H<br><br>H|
|---|

|parse_float<br><br>|GPGGASentenceParser ::parse|
|---|
<br><br>|GPRMCSentenceParser ::parse|
|---|
<br><br>|PSTI030SentenceParser ::parse|
|---|
<br><br>|PSTI032SentenceParser ::parse|
|---|
|
|---|

|Space Used<br><br>101-250 GB (9.2%) 21-50 GB (11.8%)<br><br>1-20 GB (58.0%)<br><br>51-100 GB (6.7%)<br><br>251-500 GB (5.0%)<br><br>Others (9.2%)|
|---|

|[Figure 250]<br><br>«U» ssas.model_json - U<br><br>databasename : (nvarchar(128))<br><br>[Figure 251]<br><br>«V» ssas.model_json_10 - V<br><br>[Figure 252]<br><br>«V» ssas.model_json_20 - V<br><br>[Figure 253]<br><br>«V» ssas.model_json_32_relationships - V<br><br>databasename : (nvarchar(128)) relationships_name : (nvarchar(500))<br><br>The diagram is interactive and contains links.|
|---|

|Accordion title<br><br>Accordion content<br><br>[Figure 254]<br><br>Collapsed accordion<br><br>[Figure 255]<br><br>Collapsed accordion<br><br>Open Sans Semibold 16px<br><br>Open Sans Regular 16px<br><br>This is the first accordion Additional information Open Sans Light Italic 16px<br><br>[Figure 256]|
|---|

|06 months<br><br>2012 2014 2016 2018<br><br>70.0%<br><br>72.5%<br><br>75.0%<br><br>77.5%<br><br>80.0%<br><br>82.5%<br><br>Fully vaccinated six month olds|
|---|

|wph<br><br>impbid<br><br>wff ph<br><br>wps<br><br>|- ( ph -> ( ps <-> ch ) )<br><br>wff ps<br><br>wch<br><br>wff ch<br><br>impcon4bid.1<br><br>|- ( ph -> ( ps -> ch ) )<br><br>con4d<br><br>wph<br><br>wff ph<br><br>wps<br><br>|- ( ph -> ( ch -> ps ) )<br><br>wff ps<br><br>wch<br><br>wff ch<br><br>impcon4bid.2<br><br>|- ( ph -> ( -. ps -> -. ch ) )|
|---|

|biosample<br><br>organismal entity<br><br>is_a<br><br>thing with taxon<br><br>uses id name category related to node property iri full name description systematic synonym has phenotype in taxon<br><br>|
|---|

Figure 6. SVG Diagrams examples. These are ground truth SVG examples from the test set. They are presented as SVG, showing the challenge of understanding intricate structures and small texts, with images of variate aspect ratios.

narrow range of images, each consisting of a single character per image, making them an ideal test case for SVG generation [13, 14, 47].

###### 8.3. Creating SVG-Diagrams

We introduce a novel SVG dataset, SVG-Diagrams, which focuses exclusively on diagrams, graphs, workflows, and other designs characterized by discrete elements such as boxes, arrows, and text. To construct this dataset, we filtered all SVGs containing the text element. Table 5 provides detailed statistics about the dataset, and Figure 6 illustrates test samples from SVG-Diagrams.

This new benchmark for diagram generation is highly relevant, as it addresses a use case that cannot be tackled by traditional image-processing models or methods limited to the path primitive. Only approaches capable of leveraging SVG code can fully exploit the use of primitives to generate such structured designs effectively.

###### 8.4. Generating Synthetic Captions on SVG-Stack

To generate textual instructions for vector images, we process SVG-Stack images using visual captioning models.

- Table 5. Summary of datasets. We offer a summary of statistics about the datasets used in our training and evaluation experiments. This datasets are included in SVG-Bench. The subscript sim stands for the simplified version of the dataset, as required by some baselines.

Dataset Train Val Test Source Token Length SVG Primitives Annotation SVG-Stack 2,1M 108k 5,7k

1,822 ± 1,808 All Caption SVG-Stacksim 601k 30,k 1,5k 2k ± 918 Vector path Caption SVG-Diagrams - - 472 3,486 ± 1,918 All Caption SVG-Fonts 1,8M 91,5k 4,8k

TheStack [35]

2,121 ± 1,868 Vector path Font type SVG-Fontssim 1,4M 71,7k 3,7k 1,722 ± 723 Vector path Font type SVG-Emoji 8,7k 667 668

Glypazzn [47]

2,551 ± 1,805 All Class SVG-Emojisim 580 57 96 2,448 ± 1,026 Vector Path Class SVG-Icons 80,4k 6,2k 2,4k

OpenMoji, NotoEmoji, TweMoji

2,449 ± 1,543 Vector path SVG-Iconssim 80,435 2,836 1,277 2,005 ± 824 Vector path SVG-FIGR ‡ 270k 27k 3k IconShop [94] 5,342 ± 2,345 Vector path Class, Caption

DeepSVG [14]

This approach provides a textual description for each image, enabling us to fine-tune our model to follow textual instructions. For this task, we leverage off-the-shelf AI captioners, specifically BLIP2 [41] and Llava [45].

Through prompt engineering, we guide these models in performing the captioning task with reasonable quality. The prompt we used is shared in Prompt 8.4. After automatically captioning all SVG samples in SVG-Stack, we compute the CLIP Score for the text-image pairs generated by the two models—producing two captions per image. Using a CLIP Score threshold of 30, we filter out text captions that fall below this threshold.

Prompt 1. Utilized with BLIP2 and Llava for SVG Captioning: You are a helpful assistant. Your task is to caption the input images with a concise and clear description that represents what are the contents of the image.

###### 8.5. Data Augmentation for SVG

We introduce several data augmentation operations on SVGs that aim to perform minor modifications to the SVG code and rasterize it to get a new sample while training. We include rotation, color, and curve noise. We evaluate this setting on datasets with fewer samples, namely SVGEmoji and SVG-Icons, as the other two datasets are large enough to do not overfit. Results are shown in Table 8. Both datasets display improvements using these augmentations. We see a substantial uplift for SVG-Emoji, which has limited training data.

We introduce several augmentation operations to SVGs to apply slight changes that help our model learn to generate more precise results — for instance, being able to capture

exact colors from the image and encode them in hexadecimal code to insert it in the fill attribute of the SVG element. Applying rotations or adding noise to the curve’s control points helps the model learn to precisely capture the position of the edges or thickness of the stroke.

We perform random rotations in an angle range. We perform color changes by first parsing the element’s color using the fill attribute and adding slight white Gaussian noise to the RGB values. We propose curve noise by injecting a small Perlin [80] noise into the control points in Bézier curves. We also experimented with adding Gaussian noise, which resulted in much less natural results. We apply this noise by uniformly sampling a scalar from the interval between 0.01 and 0.05 and use it to scale the noise.

We apply these augmentations directly on the SVG code, which involves parsing the XML code and accessing the attributes and arguments of the primitives defined. We use the libraries BeautifulSoup3 and SvgPathTools4. Some primitives are simplified using our augmentations.

###### 9. SVG Methods and Baselines

Here, we describe the previous methods and baselines used to compare StarVector’s performance in Image-toSVG and Text-to-SVG generation tasks. We consider previous deep learning-based methods and image-processing methods. We evaluate the baselines with publicly available code in our proposed setup.

###### 9.1. Image-to-SVG Baselines

We reproduce all previous approaches on our proposed SVG-Bench benchmark, as the available results stem from an unclear version of the fonts, emojis, and icons datasets. For theImage-to-SVG task, we consider several baseline

- 3https://www.crummy.com/software/BeautifulSoup/

bs4/doc/

- 4https://github.com/mathandy/svgpathtools

SVG-Stack

SVG-Emoji

## CM YK

RAM Size

16.01-24.0 GB (15.7%)

4.01-8.0 GB (16.9%)

64.01-256.0 GB (6.7%)

Others (7.9%)

8.01-16.0 GB (18.0%)

32.01-64.0 GB (34.8%)

SVG-Fonts

SVG-Icons

- Figure 7. Datasets in SVG-Bench. Ground truth test examples from the test sets of SVG-Stack, SVG-Emoji, SVG-Fonts, and SVGIcons. We show SVG images.

methods across deep learning and image processing approaches.

In the deep learning methods category, we start with

DeepSVG[14], Im2Vec[67], and LIVE[50], using the official implementations with the hyperparameters proposed by the authors, and applying their pre- and post-processing

SVG-Stack-Simple SVG-Emoji-Simple

SVG-Fonts-simple

SVG-Icons-Simple

- Figure 8. Simplified Datasets in SVG Bench. Ground truth test examples from the simplified test sets of SVG-Stack, SVG-Emoji, SVG-Fonts, and SVG-Icons. We show SVG images.

code as required. Additionally, we incorporate the recent GPT-4 Vision[56], which is capable of processing images

- as input and generating corresponding SVG code as output. For the image processing-based methods, which do not

rely on data-driven learning, we consider VTracer[87], Autotracer[51], and Potrace [59], running them on the test sets of SVG-Bench.

- Table 6. Summary of SVG Methods. We compare SVG generation methods based on Image Processing, Latent Variable, Differentiable Rendering, and Multimodal LLM, evaluating their performance in SVG Generalization and SVG Generation Tasks. The SVG Generalization column shows whether a model generates diverse SVG types (e.g., icons, logos, complex shapes) with ✓ or specializes in a subtype (e.g., emojis, fonts) with ✗. The SVG Primitive Coverage column indicates access to all SVG primitives. The table also evaluates Image Vectorization, Text to SVG, and Diagram Generation, using ✓ for support and ✗ for limitations. ∗DeepSVG requires modifications for image input.

SVG Coverage and Generalization SVG Generation Tasks Method Type Model Input Train

SVG Generalization

SVG Primitive Coverage

Image Vectorization

Text to SVG Diagram

Generation Image Processing

Supervision

Vtracer Image Image ✓ ✗ ✓ ✗ ✗ Autotrace Image Image ✓ ✗ ✓ ✗ ✗ Potrace Image Image ✓ ✗ ✓ ✗ ✗

Im2Vec Image Image ✗ ✗ ✓ ✗ ✗ DeepSVG SVG Vector ✗ ✗ ✓∗ ✗ ✗ SVGFormer SVG Vector ✗ ✗ ✓ ✗ ✗

Latent Variable

DiffVG Image Image ✓ ✗ ✓ ✗ ✗ LIVE Image Image ✓ ✗ ✓ ✗ ✗ SAMVG Image, Text Image ✓ ✗ ✓ ✗ ✗ SVGDreamer Image, Text Image ✓ ✗ ✗ ✓ ✗

Diff. Rendering

GPT-4 V Image, Text SVG ✓ ✓ ✗ ✓ ✓ CodeLlama Image SVG ✓ ✓ ✗ ✓ ✓ IconShop Text SVG ✗ ✗ ✗ ✓ ✗ StarVector Image, Text SVG ✓ ✓ ✓ ✓ ✓

Multimodal LLM

Autotrace5 [51] is a tool designed for converting images to vector graphics, similar to Potrace. It supports various input formats and can output to several vector formats. Autotrace’s key feature is its ability to transform pixelated images into smooth, scalable vectors, making it ideal for upgrading images for various applications without losing detail or clarity. Our experiments leverage the Python bindings6 implementation of AutoTrace.

Potrace7 [59] is a utility designed to convert images into refined, scalable vector graphics. It accepts input in various bitmap formats and outputs to a selection of vector formats. This functionality is particularly valuable for generating SVG of scanned imagery, such as logos and handwritten documents. We employ a Python library8, which acts as a wrapper around the original C implementation of Potrace.

VTracer9 [87] is an image processing algorithm to convert images to SVGs. This 3-step pipeline algorithm relies on the hierarchical clustering of images, which are traced into vectors. First, pixels are converted into paths and then simplified into polygons. In the last step, polygons are smoothened and approximated with a Bezier curve fitter. We use the Python library10 for experiments, a wrapper over the Rust implementation. Similar to Im2Vec, we scale down

- 5https://potrace.sourceforge.net/
- 6https://github.com/lemonyte/pyautotrace
- 7https://potrace.sourceforge.net/
- 8https://github.com/tatarize/potrace
- 9https://github.com/visioncortex/vtracer
- 10https://github.com/etjones/vtracer_py

all the images to 128X128 resolution. We use all the default values for the image processing engine, which generates a multi-colored SVG.

Im2Vec [67] uses an end-to-end VAE, trained using only image supervision to produce vector graphics. The input rasterized image is encoded to a ‘global’ latent vector and passed to an RNN to produce latent code for each path. The path decoder decodes these codes into Bezier paths to generate the output SVG. We used the publicly available code11 to report the results.

We scaled all the images to 128 × 128 resolution to be compatible with the Im2Vec model. We used a learning rate of 5×10−4 and a batch size of 8. We implemented a custom post-processing operation for converting the vector parameters obtained during Im2Vec inference to obtain compilable SVG code.

LIVE, (Layer-wise Image Vectorization) [50] is a method for progressively generating SVGs that closely fit a given raster image by recursively adding and optimizing closed vector paths. Using a differentiable renderer (based on DiffVG [43]), LIVE enables direct optimization of paths under raster image supervision while controlling shape complexity by adjusting the number of path segments. It introduces component-wise path initialization, identifying key visual components to ensure efficient topology extraction and minimize redundant shapes. LIVE achieves high-quality reconstructions with fewer paths, reducing SVG file size com-

11https://github.com/preddy5/Im2Vec

pared to other approaches. Nevertheless, its test time optimization approach makes it time-consuming during generation. We utilized their official open-source implementation 12 with the proposed hyperparameters. This method requires to define a constant number of paths; the more paths defined, the more accurate. We have performed an ablation on the number of paths (see Table 3) and found that paths=32 is an optimal value that brings good visual results. However, it takes more than 10 minutes to generate a single SVG, which makes it slow for a professional use case.

DiffVG [43] is a landmark in vector graphics research, pioneering deep learning-based methods with the first differentiable vector graphics rasterization pipeline. By leveraging a combination of anti-aliasing techniques and gradientbased optimization, DiffVG ensures differentiability. Unlike methods relying on non-differentiable curve-to-mesh conversions, DiffVG employs a forward-backward rasterization process, where the forward pass generates antialiased images and the backward pass computes gradients with respect to vector graphic parameters. Using the official implementation13 and proposed hyperparameters, we ablate the number of paths, finding paths=60 to be optimal. DiffVG balances versatility and performance, achieving approximately 30 seconds per generation while excelling in differentiable rendering tasks.

DeepSVG [14] was introduced as a hierarchical path-based VAE encoder-decoder transformer architecture. Here, input paths are encoded separately using a path encoder and aggregated using a second encoder to produce a latent vector. The decoder uses this latent vector to output the path representations, which provide actual draw commands and arguments. We used the open-source code14 to reproduce the results on different datasets. However, since the DeepSVG framework only allows simplified SVGs, we report results on the ‘simplified’ test sets (see Table 1).

This model can only handle simplified SVGs composed of simple line strokes and splines (see examples in Figure 9). Further, it can only process SVGs with eight groups (i.e., groups of shapes or parent nodes) and vector paths of

- at most 30 commands. To reproduce the DeepSVG baseline, we use the original hyperparameters, including a learning rate of 1e − 3 and a number of epochs of 50. We use a batch size of 200, except for the smaller emoji dataset, where we experiment with a batch size of 50.

- 12https://github.com/ma-xu/LIVE
- 13https://github.com/BachiLi/diffvg
- 14https://github.com/alexandre01/deepsvg

GPT-4 Vision. We use GPT-4V [56] by inserting an image and zero-shot prompting to generate SVG code. Here, we show how one can use prompt engineering [10, 11, 56] to condition the model to generate executable SVG code representing the given image. Prompt 9.2 was used for this endeavor. We use the OpenAI library15.

[Figure 257]

Figure 9. Image-to-SVG results on simplified SVG-Icons and SVG-Fonts test set.

[Figure 258]

Figure 10. Image-to-SVG results on SVG-Icons test set.

[Figure 259]

Figure 11. Image-to-SVG results on SVG-Stack test set. We show cherry-picked failure examples of StarVector.

15https://platform.openai.com/docs/libraries

[Figure 260]

- Figure 12. Image-to-SVG results on SVG-Emoji test set. We show cherry-picked failure examples of StarVector.

###### 9.2. Text-to-SVG Generation Baselines

For the Text-conditioned SVG generation task, we select baselines based on works that contain reproducible methodologies in public datasets or public code repositories. We reproduce baseline models from their official repositories, respecting the proposed hyperparameters.

Prompt 3. Used on GPT4 and CodeLlama for Text-to-SVG Generation: You are a helpful assistant assisting researchers in generating SVG code from textual descriptions. You will be provided with details to guide your SVG creation. Your task is to write SVG code that accurately represents the given textual information to the fullest extent possible. You are committed to solving the task of SVG generation for a robust system, so always strive to produce the best SVG code you can. Feel free to use multiple paths and any necessary shapes, colors, or lines to generate compilable SVG code within a maximum of 9000 tokens. The goal is to ensure the resulting SVG, when rasterized, best represents the described content. Respond only with the SVG code, enclosed in triple quotes, that directly corresponds to the provided textual description. Avoid adding any explanation or commentary.

CodeLlama, [82, 83] has shown great success in general coding benchmarks. To the best of our knowledge, CodeLlama has seen SVGs during training. Hence, it is reasonable to consider it a strong baseline for text-conditioned SVG generation. We use Anyscale endpoints16 to generate CodeLlama results.

GPT 4, is a closed source LLM that shows state-of-the-art results in many NLP sceneraios [11, 55, 71]. We evaluate GPT-4’s 0-shot ability in generating SVGs when prompted with text inputs. We use OpenAI API17 to generate results for GPT-4 in the 0-shot setting. Prompt 9.2 was used for the Text-to-SVG task.

Prompt 2. Used on GPT4-V VLM for Imageto-SVG Translation: You are a helpful assistant. Your task is to help researchers write SVG code to reconstruct the provided image as accurately as possible. You should also provide a caption for the image. You are dedicated to solving the task of Imageto-SVG conversion for a robust system. Therefore, you must always respond with the best SVG code you can create. Feel free to use multiple paths to generate a compliant SVG code within a maximum of 8000 tokens. You should present the SVG code that best reconstructs the input image enclosed in triple quotes.

- 16https://app.endpoints.anyscale.com/
- 17https://platform.openai.com/docs/guides/gpt

IconShop IconShop [94] uses a transformer-based architecture to encode path commands and learn to model SVG path sequences autorregressively. It has shown excellent results in simplified icon scenarios and provides a good solution to Text-to-SVG generation by extending the FIGRSVG dataset with captions. We have access to their dataset and original splits and have trained our model on that data using a pre-trained checkpoint (trained on SVG-Stack). We have extracted the results from IconShop and included them here to compare our method.

###### 10. Additional Experiments and Results

###### 10.0.1. Image-to-SVG Results

We show additional Image-to-SVG results from StarVector. Figures [13 - 18] show substantial qualitative samples generated by StarVector on all the proposed datasets. All results are computed in the test sets. We can observe the weaknesses and strengths of our model. Simplified datasets (Figures 16) are near-perfectly converted to SVG. In the case of icons, in Figure 18, sometimes the model runs out of SVG code tokens, and the image is incomplete. Results on SVGEmoji 17 show impressive performance in estimating the shape’s color and semantics. However, it lacks fine-grained and accurate positioning of objects, i.e., in some examples, the model loses track of the coherent position and form of shapes. These problems result from insufficient emoji samples, i.e., less than 10,000 training examples. This problem can be alleviated by scaling up the current model in the number of parameters (currently 1.4 billion), training data for pre-training, and computing resources.

Ground truth Generated

Ground truth Generated

Ground truth Generated

Display Manager

Display Manager

SDDM (42.9%)

SDDM (42.9%)

LightDM (3.6%)

LightDM (3.7%)

GDM (2.4%)

GDM (2.4%)

Unknown (51.2%)

Unknown (51.2%)

Ground truth Generated

Ground truth Generated

Ground truth Generated

Ground truth Generated

Ground truth Generated

| |
|---|

| |
|---|

- Figure 13. Image-to-SVG Results on SVG-Stack. We present vectorizations of StarVector-1B on the t set SVG-Stack. Left is input raster image, right is the SVG image (in SVG format).

|test|
|---|

|of|
|---|

###### Input Image StarVector-8B LIVE VTracer PoTrace AutoTrace

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

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

DinoScore: 0.9871 MSE: 0.0075

DinoScore: 0.8325 MSE: 0.0019

DinoScore: 0.8750 MSE: 0.0026

DinoScore: 0.7806 MSE: 0.0062

DinoScore: 0.8243 MSE: 0.0034

MSE: 0.0026

MSE: 0.0026

MSE: 0.0026

MSE: 0.0026

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

DinoScore: 0.9941 MSE: 0.0052

DinoScore: 0.7835 MSE: 0.0026

DinoScore: 0.8687 MSE: 0.0033

DinoScore: 0.9006 MSE: 0.0100

DinoScore: 0.7814 MSE: 0.0040

MSE: 0.0026

MSE: 0.0026

MSE: 0.0026

MSE: 0.0026

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

DinoScore: 0.9988 MSE: 0.0071

DinoScore: 0.8420 MSE: 0.0069

DinoScore: 0.9195 MSE: 0.0119

DinoScore: 0.9014 MSE: 0.0446

DinoScore: 0.9060 MSE: 0.0159

MSE: 0.0446 MSE: 0.0159

MSE: 0.0446 MSE: 0.0159

MSE: 0.0446 MSE: 0.0159

MSE: 0.0446 MSE: 0.0159

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

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

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

DinoScore: 0.9960 MSE: 0.0150

DinoScore: 0.8670 MSE: 0.0060

DinoScore: 0.8624 MSE: 0.0066

DinoScore: 0.8014 MSE: 0.0094

DinoScore: 0.7757 MSE: 0.0082

MSE: 0.0082

MSE: 0.0082

MSE: 0.0082

MSE: 0.0082

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

DinoScore: 0.9887 MSE: 0.0532

DinoScore: 0.8513 MSE: 0.0159

DinoScore: 0.7943 MSE: 0.0196

DinoScore: 0.8560 MSE: 0.0842

DinoScore: 0.7922 MSE: 0.0309

MSE: 0.0159

MSE: 0.0159

MSE: 0.0159

MSE: 0.0159

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

DinoScore: 0.9974 MSE: 0.0210

DinoScore: 0.7979 MSE: 0.0083

DinoScore: 0.8366 MSE: 0.0104

DinoScore: 0.8128 MSE: 0.0147

DinoScore: 0.8001 MSE: 0.0116

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

DinoScore: 0.9891 MSE: 0.0093

DinoScore: 0.8584 MSE: 0.0035

DinoScore: 0.8542 MSE: 0.0049

DinoScore: 0.8462 MSE: 0.0056

DinoScore: 0.8987 MSE: 0.0058

MSE: 0.0049

MSE: 0.0049

MSE: 0.0049

MSE: 0.0049

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

DinoScore: 0.9690 MSE: 0.0316

DinoScore: 0.8433 MSE: 0.0166

DinoScore: 0.8202 MSE: 0.0164

DinoScore: 0.7061 MSE: 0.0243

DinoScore: 0.8119 MSE: 0.0209

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

- Figure 14. We compare the results from StarVector-8B with those from the most powerful baselines. Notably, StarVector is the only method capable of producing acceptable results that preserve both structural integrity and textual content by utilizing a variety of SVG primitives. In contrast, other methods tend to generate blobs and curves that merely attempt to fit the structure and color of the original image. We present two metric scores for each sample: DinoScore and MSE. MSE consistently yields higher scores for other methods, as they focus on fitting vectors to the image as accurately as possible. While StarVector may not achieve perfect reconstruction, its results are preferred for their semantic fidelity. This highlights the limitations of MSE and the importance of DinoScore in capturing these aspects.

DinoScore: 0.9980 MSE: 0.0085

DinoScore: 0.9050 MSE: 0.0049

DinoScore: 0.9099 MSE: 0.0082

DinoScore: 0.9012 MSE: 0.1465

DinoScore: 0.9400 MSE: 0.0048

MSE: 0.0049 MSE: 0.0082

MSE: 0.0049 MSE: 0.0082

MSE: 0.0049 MSE: 0.0082

MSE: 0.0049 MSE: 0.0082

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

DinoScore: 0.9662 MSE: 0.0446

DinoScore: 0.8076 MSE: 0.0216

DinoScore: 0.8401 MSE: 0.0222

DinoScore: 0.9015 MSE: 0.0277

DinoScore: 0.8374 MSE: 0.0293

MSE: 0.0446

MSE: 0.0446

MSE: 0.0446

MSE: 0.0446

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

DinoScore: 0.9986 MSE: 0.0055

DinoScore: 0.9493 MSE: 0.0022

DinoScore: 0.8036 MSE: 0.0043

DinoScore: 0.8336 MSE: 0.0102

DinoScore: 0.8010 MSE: 0.0061

DinoScore: 0.8929 MSE: 0.0994

DinoScore: 0.9115 MSE: 0.0490

DinoScore: 0.8773 MSE: 0.0899

DinoScore: 0.9831 MSE: 0.0692

DinoScore: 0.9308 MSE: 0.0686

[Figure 621]

###### Figure 15. Image-to-SVG results on SVG-Fonts test set. Results are remarkably good, obtaining perfect font reconstructions. Intricate details are preserved. This is because the dataset is very large, above 1M samples. This shows that if having access to a large dataset, StarVector can learn high-quality SVG generation.

[Figure 622]

###### Figure 16. Image-to-SVG results on SVG-Fonts simplified test set.

[Figure 623]

###### Figure 17. Image-to-SVG results on SVG-Emoji test set. Results are mostly wrong in this benchmark, due to the small training dataset of approximately 8k examples.

[Figure 624]

###### Figure 18. Image-to-SVG results on SVG-Icons test set.

Comparing parameter count of models . The number of learnable parameters in deep learning based models often correlates with performance. Pre-LLM models like Im2Vec and DeepSVG use significantly fewer parameters (up to 5M) compared to StarVector and GPT-based models, which operate in the billions. While pre-LLM models can produce accurate results, they lack the generalization ability of LLM-based approaches. Comparing StarVector with GPT models reveals that high-fidelity SVG generation is achievable with just 1B parameters, whereas GPT4V lacks specific training for this task. Future models will likely incorporate SVG data in training, but current results already demonstrate that LLM-based approaches offer superior generalization and scalability for SVG generation, at the cost of utilizing more parameters.

Context Length Limitation. The model’s architecture imposes a clear limitation on context length, which significantly impacts training and testing data pipelines, as well as the skills the model can learn. Our experiments show that the model scales effectively with increasing context lengths, from 8k to 16k, indicating that this limitation must be addressed with techniques for handling longer contexts—an area LLMs are expected to improve. For fair comparisons, we restricted our benchmark tests to a context length of 8k and evaluated all baselines within this setting. However, the benchmarks also provide versions with longer context lengths to assess future models, as increased length generally correlates with more complex SVGs. We did not observe substantial differences in scores between the 4k and 10k token settings, primarily because the data in our benchmark can typically be represented using an average of 3k tokens.

Limitations on Complex SVG Structures. StarVector encounters challenges with complex SVG structures, intricate shapes, and detailed illustrations primarily due to limitations in its architecture. Currently, the model’s image encoder handles images by simply padding and resizing them to fixed dimensions of 224 or 384 pixels. This approach may not adequately capture the nuances of complex diagrams. A potential improvement would be to implement a dynamic image processing system akin to those found in newer Vision-Language Models (VLMs), which could enhance the model’s ability to interpret and generate intricate SVGs more effectively. Additionally, improving data cleaning processes is crucial, as the model sometimes produces hallucinated information due to noise in the input data, such as URLs or base64-encoded images. Addressing these issues through architectural enhancements and more robust data preprocessing could significantly improve StarVector’s performance on complex SVG tasks.

Generalization to Non-Standard SVGs. StarVector’s ability to generalize to non-standard SVGs—those not represented in its main training distribution—poses a significant challenge. While the model performs well on common styles and primitives encountered during training, it struggles with more unique or unconventional SVGs. This is primarily due to the model’s training data, which tends to focus on widely used shapes and designs. As a result, StarVector may exhibit a bias towards these common styles, leading to suboptimal performance when faced with SVGs that feature unusual structures or less frequent elements.

To assess StarVector’s generalization capabilities, we evaluated its performance on various datasets that include non-standard SVGs. The results indicate that while the model can produce reasonable outputs for some nonstandard examples, it often falls short in accurately capturing the intricacies of less familiar styles. This limitation suggests that the model’s training set lacks sufficient diversity to encompass the full range of potential SVG designs.

To address these concerns, future work should focus on expanding the training dataset to include a wider variety of SVG styles and structures. Incorporating data from niche applications and artistic domains could enhance the model’s ability to generate SVGs across a broader spectrum of design elements. Additionally, techniques such as domain adaptation and transfer learning could be explored to improve generalization to non-standard SVGs, allowing StarVector to adapt more effectively to unfamiliar inputs.

###### 10.1. Ablation Studies

We performed ablations on the image encoder type, the data augmentation pipeline, inference techniques, and generation parameters. Most of our ablations were performed on the StarVector-1B model for faster iteration, and we empirically find they work well on the larger StarVector-8B.

Image Encoder Ablation. The choice of image encoder for the problem of Image-to-SVG is highly impactful, as it determines how well visual information from raster images can be preserved in a representation suitable for precise reconstruction in the SVG space. We ablated the visual encoders by replacing them with VQGAN [23], ConvNext [46], and CLIP ViT-B/32 [62], in our StarVector-1B proposed architecture. This setup evaluates three commonly used approaches in visual representation learning [23, 62, 73]. In our experiments, CLIP consistently outperformed across all metrics for various datasets (see Table 9). Figures [19–22 further illustrate how VQGAN and ConvNext tend to lose local details during generation, even while maintaining semantic relevance.

###### SVG-Fonts SVG-Emojis SVG-Icons SVG-Stack

Sampling technique LPIPS ↓ SSIM ↑ MSE ↓ LPIPS SSIM MSE LPIPS SSIM MSE LPIPS SSIM MSE Greedy 0.019 0.969 0.013 0.251 0.731 0.071 0.059 0.912 0.028 0.157 0.797 0.067 + Beam Search (B=5) 0.018 0.970 0.012 0.250 0.732 0.070 0.058 0.913 0.027 0.156 0.798 0.066

- Nucleus Sampling (T=0.5) 0.013 0.976 0.008 0.202 0.778 0.051 0.043 0.923 0.022 0.153 0.785 0.072

- Nucleus Sampling (T=1.0) 0.015 0.975 0.009 0.244 0.742 0.067 0.053 0.917 0.025 0.161 0.786 0.069

###### + Beam-Search (B=5) 0.034 0.948 0.027 0.244 0.742 0.068 0.065 0.913 0.027 0.195 0.766 0.089 + Beam-Search (B=10) 0.040 0.943 0.031 0.251 0.742 0.072 0.071 0.910 0.028 0.175 0.762 0.079

Table 7. Ablation study on sampling strategies. We experimented using greedy decoding and added a beam search with B=5. We test nucleus sampling [31] using top p=0.9, with temperatures T=0.5 and T=1.0. The two final rows describe beam search with nucleus sampling at T=1.0. See huggingface.com/blog/how-to-generate for reference on these sampling techniques.

Pre-training on SVG-Stack. Pre-training on the SVGStack is highly beneficial for the downstream datasets with small data. Table 8 shows the uplift on all the metrics for different datasets. Qualitatively, we can also see that pretraining helps the model to identify the nuanced details from the images. For the case of SVG-Emoji, pre-training is a vital requirement, as it overfits without it due to limited data. Figure 17 shows that the model relies on colors and shapes to generate the SVG.

Ablation on Generation Hyperparameters. We explore the impact of different generation hyperparameters on the StarVector-1B model. After an initial exploration to empirically determine the most relevant hyperparameters, we focus our ablation on these. We find that temperature and the number of beams in beam search significantly affect performance. The model is evaluated across various configurations (see Table 7 and Figure 23). Our results show that a beam search size of 5 achieves the best outcomes, albeit with increased memory usage and runtime. Similarly, nucleus sampling with a top-p of 0.9 and a temperature of 0.5 delivers the best overall performance.

###### 10.2. Text-to-SVG Results

Figures [25 - 28] show additional qualitative results of StarVector when performing the task of text-conditioned SVG generation, performed on SVG-Stack and FIGR-SVG test sets. Our samples show reasonable effectiveness at this task, consistently grasping features like colors, shapes, and semantic concepts. However, sometimes some details required in the prompt are lost, e.g., an exact number of circles, shapes inside other shapes, or the direction of arrows. In some cases, some vector graphics shapes lose coherence, which we attribute to our model’s current scale in terms of model parameters and context length. We suspect that these mistakes are due to the limited quality of the textual descriptions, sometimes lacking precision and grounding on the

[Figure 625]

- Figure 19. Ablation of Image Encoders Image vectorization results using different visual encoders on SVG-emoji test set. CLIP is the image encoder that delivers the best results, whereas VQGAN and ConvNet often miss relevant semantics of the image. No SVG-Stack Pretrain (CLIP) refers to an ablation where we use CLIP out of the box, without unfreezing its weights.

[Figure 626]

- Figure 20. Ablation of Image Encoders Image vectorization results using different visual encoders on SVG-Stack test set. CLIP offers the best results. VQ-GAN and ConvNet often miss relevant semantics of the image.

SVG images. See Figure 24 for successful cases of Text-toSVG generation on SVG-Stack. Figure 29 highlights some failure modes of StarVector-8B. These figures illustrate the impact of different generation temperatures. We rank the outputs generated at different temperatures based on their CLIP Score in relation to the text instruction.

Nevertheless, the StarVector approach of using LLMs for SVG code generation is the only method among baselines that allows us to create diverse vector graphics unre-

[Figure 627]

- Figure 21. Ablation of Image Encoders Image vectorization results using different visual encoders on SVG-Icons test set. CLIP brings the best visual results. VQ-GAN and ConvNext are not able to capture correctly the details for correct vectorization. No SVGStack Pretrain (CLIP) refers to an ablation where we use CLIP out of the box, without unfreezing its weights. Notably, better results are obtained when training the CLIP image encoder on SVGStack.

[Figure 628]

- Figure 22. Ablation of Image Encoders Image vectorization results using different visual encoders on SVG-Fonts test set. As in the other datasets tested, CLIP brings the best visual results, as others are not able to provide perfect vector reconstruction when intricate details are present. No SVG-Stack Pretrain (CLIP) refers to an ablation where we use CLIP out of the box, without unfreezing its weights. Notably, better results are obtained when training the CLIP image encoder on SVG-Stack.

strainedly, paving the way for more challenging and intricate designs.

###### 10.3. Results on SVG-Diagrams

Figure 2 presents the results of StarVector-8B, along with comparisons to LIVE, VTracer, Potrace, and AutoTrace. StarVector-8B is the only approach that produces plausible results, as it effectively leverages appropriate SVG primitives. DinoScore aligns well with this visual assessment, accurately reflecting the quality of StarVector-8B’s outputs.

In contrast, MSE consistently favors other baselines despite their limitations. This is because MSE prioritizes exact pixel matching, favoring models designed to fit curves and colors to the input image. However, these baselines fail to preserve the semantics of the original diagrams, resulting in outputs where the meaning and structure are completely lost.

- Table 8. Results of SVG Data Augmentation. We ablate both our data augmentation pipeline and the use of a pretraining stage with SVG-Stack. These experiments are conducted on smaller datasets that are more susceptible to overfitting, using the StarVector-1B model. Vanilla refers to the StarVector model trained directly on the given dataset without SVG-Stack pretraining. Next, we introduce our data augmentation pipeline. Finally, we initialize training from an SVG-Stack pretrained checkpoint and fine-tune on the given dataset. The “+” symbol indicates that the methods from the previous rows are also included.

SVG-Emojis SVG-Icons Method LPIPS ↓ SSIM ↑ MSE ↓ LPIPS↓ SSIM↑ MSE↓

StarVector (vanilla) 0.355 0.683 0.108 0.104 0.845 0.047 + Data Augmentation 0.329 0.706 0.097 0.057 0.905 0.029 + SVG-Stack Pretrain 0.225 0.748 0.061 0.057 0.894 0.031

- Table 9. Ablation of Image Encoders. We ablate different image encoders with StarVector-1B, namely CLIP ViT-B/32 [62], VQGAN [23], and ConvNext [46]. We experiment with training experiments on SVG-Fonts and SVG-Emojis datasets. CLIP gives the best results on all reconstruction metrics.

SVG-Fonts SVG-Emojis Encoder LPIPS ↓ SSIM ↑ MSE ↓ LPIPS ↓ SSIM ↑ MSE ↓

CLIP 0.026 0.955 0.021 0.202 0.778 0.051 VQGAN 0.092 0.854 0.072 0.345 0.688 0.099 ConvNext 0.085 0.854 0.073 0.311 0.708 0.088

- 10.4. Analysis of SVG Primitives

This section examines how StarVector leverages SVG primitives to produce more compact and semantically accurate SVGs. In contrast to prior models constrained to using only path primitives, StarVector effectively utilizes the entire range of SVG primitives, including parametrically defined shapes, gradients, and text elements.

This enhanced capability stems from its ability to operate directly within the SVG code space, facilitated by its multimodal, transformer-based architecture [2, 45], which integrates visual and textual inputs. StarVector generates SVG code that closely resembles the input raster image while maintaining semantic awareness, enabling the use of symmetry, parametric shapes, and text. Prior methods, limited to first-order path primitives, lack this semantic understanding, resulting in less compact and less expressive SVG representations.

Qualitative Analysis of SVG Primitives Table 10 presents tests conducted on StarVector-8B and VTracer using simple designs composed of basic shapes such as circles, rectangles, and triangles, with variations in color, transparency, and levels of overlap. StarVector-8B demonstrates the ability to precisely identify the primitives that

[Figure 629]

- Figure 23. Ablation study on sampling temperature. We tested the performance impact of StarVector-1B when changing the sampling temperature. Results are computed for SVG-Emoji and SVG-Icons validation sets.

make up each design, producing visually accurate results while maintaining compact, concise, and interpretable SVG representations.

SVG Tag Distribution Here, we show how StarVector can generate complex SVGs using the full syntax of the SVG language, in contrast with most of the literature methods, which are restricted to using only the path command. Figure 30 displays the distribution of SVG tags in the SVGs generated by StarVector along with the distribution in the SVG-Stack dataset, showcasing the strength of our method in using SVG tags and syntax in a similar way to the original in-the-wild dataset. We have computed the exact statistics on previous methods and found that they cannot come close to StarVector in this metric, as they are limited to using paths and basic primitives. The effective usage of the large array of SVG tags and syntax makes our method the first SVG model to support these complexities.

###### 10.5. Human Evaluation

We conducted a human evaluation to compare the outputs of StarVector-8B, our best model, with those of the most powerful baselines. Participants were selected from diverse backgrounds and carefully screened for conflicts of interest, with none of the key authors involved. The evaluation was performed through a web interface (shown in Figure 31) that provided anonymized outputs and randomized sample presentations to prevent pattern recognition or bias.

The results, presented in Figure 5, demonstrate a strong preference for StarVector-8B across all settings, especially in SVG-Diagrams tasks. This highlights a disconnect between pixel-based metrics (e.g., MSE, SSIM) and human visual perception of SVGs. While baseline models often prioritize pixel-perfect reconstruction, human evaluators preferred StarVector’s sharp, well-defined shapes and its effective use of primitives (Figure 30).

Spearman correlation analyses between model metrics and human evaluations further emphasize this gap. MSE shows weak correlations (0.0596 and -0.1002), indicating

its inadequacy as a predictor of human preferences. In contrast, DinoScore exhibits significantly stronger correlations, with values of -0.6193 and 0.6214. Moreover, a robust correlation of 0.7577 between differences in DinoScore and human evaluation scores highlights DinoScore as a more reliable metric for assessing SVG quality in alignment with human judgment.

###### 10.6. Comparing StarVector with Baselines

Here, we discuss the results of each baseline individually and compare them to our proposed approach.

- 1. DeepSVG [14] is an elegant approach to learning a latent variable model for SVG. It proves effective at learning the task for the simplified datasets (Figure 9). It can accurately represent corners and edges. However, it only works in simplified datasets. This limitation restricts it from being a suitable solution in real applications.
- 2. Im2Vec [67] proposes a training procedure that does not require having SVG ground truth. It uses only pixelbased reconstruction loss with the input image, finding the optimal SVG parameters using a differentiable rasterizer like DiffVG [43]. This framework is appealing, as it aims to be used in images without SVG supervision. However, it requires hundreds of epochs with a reduced dataset to overfit the model to those examples, only working on modeling training examples, as seen in [67]. Im2Vec results on the datasets presented SVGBench are quite poor as it has bad generalization. Therefore, qualitative samples are not presented.
- 3. GPT-4 Vision [56] excels at capturing the semantics of images and generating captions that accurately describe them. The SVG generated from this description

Text Instruction Ground Truth Temp = 0.0 Temp = 0.125 Temp = 0.25 Temp = 0.375 Temp = 0.5

Generate an SVG of x icon in black circle on white background

Generate an SVG of a green check mark in a flower

Design an SVG of pencil icon vector

Draw an SVG picture of a black and white icon of a paper document

Design an SVG of a colorful square logo with a blue background.

Draw an SVG image of four square black and white icons.

Generate an SVG of a folder with a fan on it.

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

Draw an SVG image of a blue and white train icon.

Create an SVG of a large orange bell, positioned in the center of the image.

Create an SVG for a logo of a company with a gold triangle

- Figure 24. Text-to-SVG Results. We show successful Text-to-SVG results using StarVector-8B. We sample 5 different temperatures as an ablation, showing the sensitivity of this parameter during generation. Results are presented in SVG (not raster images)

A black and white image of a cross

A circle with dots on it in black and

white Tv icon vector

A file icon with a white square on it

The japanese flag is shown in a white background

Chinese font design for the word 'love'

A black and white icon of a pen and paper.

A purple and white striped logo

[Figure 630]

[Figure 631]

Folder icon with a person icon

A black circle with a play button in the center.

A white arrow on an orange background.

A fork and a leaf icon on a white background

Memory Size

16384 (22.2%)

2048 (16.7%)

32768 (5.6%)

Memory size pie chart

4096 (22.2%)

8192 (33.3%)

A white cross on a white background.

A knife is shown on a white background

A square shape is shown in the image.

An eye icon with a triangle shape

A yellow star is displayed on a white background.

An i symbol in a black circle

A gray circle with a question mark in the center.

A pink light bulb icon.

A green smiley face with a smiley face

Sun icon in a circle on a white background

An open book icon on a white background

A black and white microphone icon.

A black and white square with a black border.

A black and white headset icon.

A pink pen on a white background vector

A blue and white sign that says "continue setup".

A black and white triangle shape with two lines

| |
|---|

A blue square with a black line in the middle.

A black star on a white background

A black and white map marker symbol.

A graph of the number of people in a class

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

A green button with a gray arrow pointing to it

A black and white icon of a no entry sign

A green and black circle with a white background

A black and white image of a game controller

A red square with a white letter A in the center.

A black and white icon of a trophy cup

# JS Ablackand

Standard js logo

white logo of a curved shape

A blue and white paper clip icon.

A gray square with four squares on it

The facebook logo, with a white letter f Apple pay logo

A black and white icon of an envelope.

A video player icon is displayed in a square.

An eye icon in a circle with a black dot

A black and white illustration of a cell phone

The flag of france and the flag of france

A black and white icon of a lightning bolt

A yellow sign with the number 10 on it.

A black and white image of a japanese symbol

Chinese font for the word love

A white and gray checkered pattern

The icon for the instagram app

A black circle is placed on a white background.

A cross symbol with four lines

A black and white sun symbol.

The s logo in black and white

A blue and white icon of a camera.

A speech bubble with an x symbol on it

A yellow star with a black outline.

Ethernet Model

82579LM Gigab...

(5.6%) Ethernet

Others (27.8%)

A pie chart with the words ethernet model

Conn... (5.6%) I211 Gigabit...

(5.6%) RTL810xE PCI

... (11.1%)

RTL8111/8168/... (44.4%)

The facebook logo is shown on a white background

A black and white arrow pointing to the right

An envelope icon on a white background

A red heart with a lightning bolt through it.

The facebook logo in black and white

A black cross on a white background.

A white square with a black border

A person avatar icon on a white background

A black and white airplane icon.

A heart with a cross in it

A black heart shape on a white background.

A blue button with the word stata on it

The flag of oman is shown in red and green

A black cross on a white background

A blue bar is displayed on a white background.

A black and white logo with two arrows

A black square is placed on a white background.

A black circle with a white square in the middle.

A clock icon is displayed in a white background.

###### Figure 25. Text-to-SVG Generation results using StarVector-1B on SVG-Stack test set (i).

Chinese character for the word 'fortune'

A black circle with a white outline.

A blue square with white text on it

A yellow square with a light bulb icon

Twitter logo with a blue bird

A purple car icon on a white background

A blue triangle shape on a white background.

A black and white arrow pointing to the right.

A crosshair icon on a white background

Four square black and white icons

A black and white fidgetr with four circles

Youtube video player icon

An orange arrow pointing up in a folder

A green arrow pointing to the right

A black and white image of a file icon

Glasses icon for a website

A yellow emoticion with a black eye

A yellow arrow pointing upwards.

Square icon vector

A red emoticion with an angry face

The letter v is shown in black and white

A pink speech bubble on a white

background Whatsapp logo

A black and white icon of a sun

A black and white image of a large X.

A yellow triangle with a black outline.

A black and white cloud with a rain icon

Three stacked white boxes with green lines

Twitter logo in a circle

A yellow and black shield with a rooster on it.

A black and white logo with a white eye

A pixel style image of a brown box

The v logo with a blue arrow

A black and white icon of a document

A black and white logo for JPG.

Ethereum logo with a blue background

A black star with a white outline.

A grey and white image of a lock and key.

A white X is displayed on a pink background.

Headphones icon vector

A black and white icon of a radio wave.

A black and white light bulb icon.

| |
|---|

A purple circle with an

envelope icon Pixel heart png

A blue cloud is displayed on a white background.

A black and white coffee cup icon.

A black circle with a smiling face on it.

A black and white YouTube icon.

A letter a logo with a red and black letter

A purple circle with a down arrow in the center.

A black and white crescent on a white background

A black and white logo of a drop

A black and white pizza icon.

A phone icon in a heart shape

The s logo in a black and white circle

A black circle with a white circle in the middle.

A blue arrow pointing up on a white background

A black and white shopping cart icon.

The letter c in black and white

A black and white magnifying glass icon.

A black and white cloud icon.

A phone icon with a plus sign

A brown and white battery icon.

A purple square icon with two circles on it

A blue circle with a white V in the center.

Cross symbol clip art

A black and white image of a hamburger.

A black and white film reel icon

Credit card icon vector

A black and white icon of a refrigerator.

A dollar sign icon on a white background

A purple and gray circle with a crescent moon

A blank white card with a green border

Sun icon clip art

A blue square with an x in the middle

Google plus logo

A black and white smiley face icon

Youtube logo with a red play button

The c logo in a green square

Youtube logo with a play button

A black and white clock with a blue hour hand.

A green circle with an arrow pointing upwards.

A black and white logo of a letter G.

A yellow emoticion with glasses on it

Clipboard icon with a square and a square

A hand holding a bag with the letter r

Chinese character for cross

The html5 logo with the word html5

A black and white sign with the letter e

Sun icon vector

A black and white square with a white border

A white and purple logo of a letter C.

A black and white key symbol.

A purple circle with a knife icon

A red triangle with a white background

A black and white image of a question mark.

A black and white icon of a trash can.

A diagram of the heartbeat

api IStream device_stream<IStream>

A red power button with an arrow pointing up

A blue circle with a white center

A black and white mouse cursor icon

A black and white icon of a x

###### Figure 26. Text-to-SVG Generation results using StarVector-1B on SVG-Stack test set(ii).

I love Halloween skul l-and-bones emoji.

A necktie is a formal neckwear for a uniform.

A square is a quadrilateral shape in geometry.

The abstract square is distorted by a

third. Map pin.

The lorry is a delivery truck for transport.

The rupee is currency.

Weight dumbbell.

Hand-drawn sad emoticon upset and angry.

Player interface for video play.

A frozen sweet treat, popsicle ice-cream.

Wifi is online through w-lan.

There is a signpost with a direction arrow.

A necktie is an accessory for clothes.

Search with a magnifier to explore and view.

The smiley emoji represents happy emotions.

I like the social like with a thumbs-up.

The map pin marks the location.

The Holy Cross is a Christian symbol.

Move in the direction of the sharp turn arrow.

Remove groceries from shopping basket.

Cleaning laundry with a washing machine.

The soup spoon is tableware.

A square and a cube.

A good idea is thinking about a light bulb.

The up arrow directs the move.

Unlock your smartphone screen

notification. Mustache.

The arrow sign shows the three-way direction.

Pinpoint your music interests on the map.

Wifi-router is an electronic device.

The gear wheel's configuration is set by the cog.

A square is a shape.

The milk bottle is a dairy beverage.

Create new cloud server plus.

The sun is a solar emoji emoticon.

Bell notification.

The chef wore a toque in the kitchen.

A bar chart is a type of chart.

Man has a mustache.

Upload bag shopping arrow.

The pin marks the location with a pointer.

The user's avatar is a map pin on their

profile. Right arrow.

A tank-top is a type of shirt or undershirt.

The arrow points down to the heart with love.

A hexagon is a shape with 6 sides.

The idea invented electricity's light bulb.

Search using a magnifying

glass. Wifi.

Protect the privacy of a private document/file.

Letter or email message.

Data charted on a graph shows market statistics.

I drive my backpack with flash storage.

Search and find with a magnifying glass.

Chat using a speech bubble.

A medal, award, badge, prize, or star.

The unhappy face emoji.

Aircraft, missile, rocket, spacecraft.

Web hosting on the cloud server.

Usb flash drives store data in memory.

Move the arrow to expand and drag to drop.

The arrow indicates the up direction

where. Candle.

Admin shields star.

The anchor stopped the heavy navy ship.

A rhombus is a symmetrical di amond-shaped figure.

I wore a necktie.

Check the checklist for a checkmark.

The audio cord connects to the microphone.

Search with magnifying glass tool and

zoom. A beer mug. Apple is a fruit.

Search with magnifier.

A trapezoid is a shape.

One dice is used in the

game of luck. Favorite star.

Find location with GPS map pin.

Follow the arrow for direction.

Money is currency or cash, such as the euro.

A rupee coin is currency used for payment.

The pound bulb is an idea for a light bulb.

The bus is a vehicle for transportation.

The cassette tape contains music sound on tape.

Navigate using the down arrow.

Turn left at the junction.

Figure 27. Text-to-SVG generation on FIGR-SVG test set (i).

Download data from cloud database with arrow.

Christian faith has a cross in the church.

I need a coffee-to-go in a paper cup.

The arrow indicates the direction.

Left arrow points backward.

A people group network.

Women have a female gender symbol.

Spruce, pine, and Christmas trees are all plants.

Download the signpost arrow direction.

Flat brush used for painting flat illustration.

Shop for a cart while shopping.

Map-pin marks a location on a map.

Sweep with a broom to clean the floor.

The map pin shows the location with GPS.

The pill bottle contained drugs for health.

Snowflakes fall in winter weather.

The bowl is a food container in the kitchen set.

The erlenmeyer flask is a chemistry equipment.

The tool has options and settings with cogs.

Bug is an insect that can have a virus or error.

The atom has a proton link in physics science. Boat or ship.

The modern thick arrow points right.

Male restroom sign.

The pointer points to the arrow right. Tie. Service bell. Anchor.

Wifi is wireless connectivity to a network.

The pharmacy uses a mortar and pestle.

The ace of clubs is a poker card.

Move in the direction of the right arrow

square. Map-pin.

Minimal trophy.

Watch TV on monitor.

Media player plays movie, music, and video.

Boxer-briefs are a type of underwear.

Arrow points up at junction for navigation.

The up arrow indicates the top direction.

Connect to the Wi-Fi. Crown.

Buy a home online using a laptop.

Record sound with a microphone for voice audio.

Withdraw cash from ATM for euro money.

The MKV file extension is a type of video file.

My favorite documents are starred on my computer.

A smartwatch displays the time.

Search for glass with mag nifying-glass view.

Drawers in wardrobe.

Place a map pin.

Snowflakes have symmetry.

The receiver follows the direction of the arrow.

The mpg file is a document format.

Buy cart or basket from market for shopping.

Download the pointer arrow down using technology.

Lightning has electricity and high voltage.

Water in a plastic bottle is a beverage drink.

Calculate math using a calculator for accounting.

Christmas ball is a decoration.

Music sound files contain musical notes.

Find and focus by zooming with a magnifier.

Up arrow indicates direction. Cylinder.

Target in crosshair.

A spoon is silverware.

Find location using GPS search and pins.

Secure euro payment with money lock.

Wifi is a network for internet connection.

Erlenmeyer flask and beaker.

The wise owl has evil eyes.

Chat or message when talking in a conversation.

I rate a half star. Euro.

Play videos on YouTube using the video player.

Online video on desktop PC.

The pitcher holds a beverage.

Ice cream is a dessert food popped in a popsicle.

Christianity revolves around Jesus and the cross.

Online shopping trolley/cart finance.

The ball is a Christmas tree decoration.

Camera takes photos with lens for multimedia.

Bus is a public transport vehicle.

Wi-fi signal is wireless. Cocktail.

Sort tiles in a grid.

The magnifying glass can indicate an enlargement.

A file is a document, paper, or sheet with pages.

The thermometer measures temperature in Celsius.

A QR code is a type of general barcode.

Picture camera photography.

###### Figure 28. Text-to-SVG generation on FIGR-SVG test set (ii).

The piano has keys that produce sound.

Search using the magnifying glass icon.

The arrow points in a direction.

The dot-smiley angry emoji face is a sticker.

A map pin marks the location.

Buy cart for ecommerce shopping.

Search and find tools magnify and zoom.

A Wi-Fi network signal.

The left arrow is a back arrow symbol.

Write with a pen or pencil, edit as needed.

Text Instruction Ground Truth Temp = 0.0 Temp = 0.125 Temp = 0.25 Temp = 0.375 Temp = 0.5

Make an SVG of a black arrow pointing upwards.

Generate an SVG of a black and white image of a box with a square shape.

Design an SVG of a black cat in tears.

[Figure 632]

Make an SVG of HC logo in a white circle

Design an SVG of the tesla logo

Design an SVG of a black and white image of three dots

Draw an SVG image of a green recycling symbol, three arrows pointing inward.

Make an SVG of a black and white icon of a computer program called "GIT".

||git|
|---|
|
|---|

Generate an SVG of a blue and gray square with a blue arrow pointing to it.

Design an SVG of a woman wearing a yellow shirt and blue jeans.

- Figure 29. Text-to-SVG Results. We show failure Text-to-SVG results using StarVector-8B (cherry picked examples that show limitations). We sample 5 different temperatures as an ablation, showing the sensitivity of this parameter during generation. Results are presented in SVG (not raster images)

Table 10. Usage of SVG Primitives. Image vectorization results of StarVector and VTracer applied to images containing basic shapes, such as circles, rectangles, and polygons, with varied colors and transparencies. The leftmost column shows the input images prompted for vectorization, and other columns show the output SVG code, with the SVG primitives in red color. StarVector accurately identifies and generates SVG code for each primitive, preserving their distinct characteristics. In contrast, VTracer relies on the path primitive, resulting in SVG code that captures the input image in terms of pixels, with less fidelity to individual shapes. Due to the length of VTracer’s SVG output, only the initial lines are shown. VTracer serves as a baseline model, representative of other baselines, which are omitted for space but exhibit similar behavior, primarily using path without shape recognition.

###### Test Example StarVector VTracer

[Figure 633]

<svg width="150" height="150" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"> <rect x="35" y="0" width="114" height="132" style="fill:rgb(255,50,50);stroke-width:1;stroke:rgb(0,0,0)"/> <polygon points="56.25,49.5 112.5,147.75 0,147.75" style="fill:rgb(200,0,200);stroke-width:1;stroke:rgb(0,0,0)"/> </svg>

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="530" height="460">

<path d="M0 0 C174.9 0 349.8 0 530 0 C530 151.8 530 303.6 530 460 C355.1 460 180.2 460 0 460 C0 308.2 0 156.39999999999998 0 0 Z " fill="#FE3232" transform="translate(0,0)"/>

<path d="M0 0 C4.06429570258797 3.6330418563035494 6.775718865918918

7.966461775462079 9.71484375 12.50390625 C10.273576812744153 13.357300872802739 10.832309875488278 14.210695495605478..." fill="#A95869" transform="translate(343,60)"/>

<path d="M0 0 C40.260000000000005 0 80.52000000000001 0 122 0 C123.16960912291928 66.0224184213402 123.16960912291928 66.0224184213402 123.416015625 93.876953125 C123.47276957931317..." fill="#A95869" transform="translate(343,60)"/>

... </svg>

[Figure 634]

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="1196" height="1140">

<path d="M0 0 C394.68 0 789.36 0 1196 0 C1196 376.20000000000005 1196 752.4000000000001 1196 1140 C801.3199999999999 1140 406.64 1140 0 1140 C0 763.8 0 387.5999999999999 0 0 Z " fill="#FEFEFE" transform="translate(0,0)"/>

<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"

width="64px" height="64px" viewBox="0 0 64 64" enable-background="new 0 0 64 64" xml:space="preserve">

<path d="M0 0 C1.8671875 0.24609375 1.8671875 0.24609375 4 1 C5.1328125 2.722656250000071 5.1328125 2.722656250000071 6.125 5.0625 C8.342476699064719 9.859072675207557 10.955923466443437 14.196311705833779 13.8125 18.625 C17.369173047863796 ..." fill="#A95869" transform="translate(343,60)"/>

<ellipse fill="none" stroke="#CD2A2A" stroke-width="2.5" stroke-miterlimit="10" cx="32.333" cy="32.083" rx="29.333" ry="31.25"/>

<polygon fill="#CD2A2A" stroke="#CD2A2A" stroke-width="0.25" stroke-miterlimit="10" points="32.333,3.833 55.167,46.333 9.5,46.333 "/>

<path d="M0 0 C1.3488281421188617 2.6976562842377234 0.8033826481469646 4.151643338387942 0 7 C-0.6599999999999682 7.329999999999984

<line fill="none" stroke="#000000" stroke-width="0.25"

-1.3199999999999932 7.660000000000025 -2 8 C-2.6555511917654258 10.527332350510733 -2.6555511917654258 10.527332350510733 ..." fill="#A95869" transform="translate(343,60)"/>

stroke-miterlimit="10" x1="0" y1="0.167"x2="64"y2="0.167"/> </svg>

... </svg>

[Figure 635]

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="910" height="934">

<path d="M0 0 C300.3 0 600.6 0 910 0 C910 308.22 910 616.44 910 934 C609.7 934 309.4 934 0 934 C0 625.78 0 317.55999999999995 0 0 Z " fill="#928299" transform="translate(0,0)"/>

<svg width="300" height="300" xmlns="http://www.w3.org/2000/svg" version="1.1"> <rect x="110" y="10" width="160" height="230" fill="pink"

<path d="M0 0 C1.4849999999999994 0.9900000000000091 1.4849999999999994 0.9900000000000091 3 2 C3 2.659999999999968 3 3.32000000000005 3 4 C3.7296093749999955 4.103124999999977 ..." fill="#A95869" transform="translate(343,60)"/>

stroke="red"/> <circle cx="160" cy="120" r="120" fill="tan" stroke="green"/> <polygon points="110,20 280,125 175,280 5,180" fill="blue" fill-opacity="0.25" stroke="blue" stroke-width="2" /> </svg>

<path d="M0 0 C1.32000000000005 0.6599999999999966 2.6399999999999864 1.3200000000000003 4 2 C3.009999999999991 2 2.019999999999982 2 1 2 C1 2.6599999999999966 1 3.3200000000000003" fill="#A95869" transform="translate(343,60)"/>

... </svg>

[Figure 636]

<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"> <circle cx="50.5" cy="39.5" r="40" fill="#60f0ff" /> <ellipse cx="50" cy="39" rx="50" ry="20" fill="#2810ff" /> <polygon points="10,100 50,42 90,100" fill="green" />

</svg>

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="510" height="520">

<path d="M0 0 C168.3 0 336.6 0 510 0 C510 171.6 510 343.2 510 520 C341.7 520 173.39999999999998 520 0 520 C0 348.4 0 176.8 0 0 Z ..." fill="#2E17FE" transform="translate(0,0)"/>

- <path d="M0 0 C0.6599999999999682 0 1.3199999999999932 0 2 0 C3.1463165283203125 1.3810577392578693 3.1463165283203125 1.3810577392578693 4.428955078125 3.356689453125 C4.916970977783194 " fill="#2E17FE" transform="translate(0,0)"/>

- <path d="M0 0 C1.5393356364577242 2.844848138263643 2.426629556693727 5.489600992013067 3.1875 8.625 C8.714349606195654 27.37779319117132 28.666518702768336 42.668449172855844 45 52 C77.0038243523029 ..." fill="#2E17FE" transform="translate(0,0)"/>

<path d="M0 0 C0.3300000000000409 0 0.6599999999999682 0 1 0 C1 101.31 1 202.62 1 307 C-16.159999999999968 307 -33.31999999999999 307 -51 307 C-69.10137393182612 285.06539394143425 -69.10137393182612 285.06539394143425 -73.55859375 275.48046875 ..." fill="#2E17FE" transform="translate(0,0)"/>

... </svg>

is valid and effectively incorporates semantic concepts along with the accurate colors of the input image into the SVG code (see Figures [9 - 12]). However, it falls

short in terms of reconstruction fidelity, as GPT-4V was not specifically trained for reconstruction tasks, making these results predictable.

description (0.02%)

paragraph (0.03%)

requires (0.02%)

tabList (0.03%)

textRect (0.03%)

Outer: Ground Truth Middle: StarVector Inner: Baseline (LIVE)

textBlock (0.03%)

path (30.87%)

License (0.03%)

feMorphology (0.03%)

subject (0.03%)

g (19.20%)

text (5.93%)

date(0.03%)

Bag(0.03%)

pattern(0.03%)

rect(5.64%)

font(0.04%)

license(0.04%)

title(4.31%)

userDefs(0.04%)

polygon(4.07%)

creator(0.04%)

foreignObject (0.02%)

feMerge(0.05%)

feBlend (0.02%)

switch (0.02%)

feColorMatrix (0.05%)

path (34.07%)

stop(3.80%)

br(0.05%)

feOffset (0.03%)

feGaussianBlur(0.05%)

path-effect(0.06%)

div (0.03%)

g(27.19%)

perspective(0.06%)

namedview(0.05%)

line(3.28%)

use(16.94%)

p(0.06%)

Work(0.06%)

type(0.06%)

ud(0.07%)

text(2.81%)

circle(2.66%)

format(0.06%)

grid(0.07%)

guide(0.06%)

rect(2.79%)

guide(0.08%)

defs(2.27%)

RDF(0.06%)

defs (1.62%)

Agent(0.08%)

polygon(2.05%)

filter(0.07%)

ellipse(2.10%)

permits(0.09%)

mask(0.07%)

path (96.76%)

feMergeNode(0.10%)

title(1.97%)

desc(0.09%)

linearGradient(1.71%)

g (1.62%)

symbol(0.10%)

metadata(0.09%)

stop(1.94%)

li(0.10%)

radialGradient(0.11%)

use(1.46%)

symbol(1.76%)

image(0.10%)

polyline(0.13%)

animate(0.28%)

style(1.02%)

feComposite(0.11%)

line(1.40%)

tspan(0.29%)

feFlood(0.11%)

circle(1.17%) defs (1.01%)

tspan(0.88%)

a(0.42%)

marker(0.18%)

style (0.48%)

ellipse (0.93%) clipPath (0.81%)

linearGradient (0.63%)

a(0.80%)

mask(0.21%)

div(0.68%)

feBlend(0.21%)

feOffset(0.21%)

metadata(0.63%)

radialGradient(0.26%)

animate(0.60%)

foreignObject(0.28%)

clipPath(0.55%) RDF(0.48%)

switch(0.28%)

feColorMatrix (0.29%)

Work (0.48%) format (0.47%)

feGaussianBlur (0.33%)

filter (0.37%)

type (0.47%) desc (0.47%)

polyline (0.37%)

namedview (0.42%)

- Figure 30. Distribution of SVG Primitives and SVG Tags. We show the frequency of SVG tags that appear in the ground truth of SVGStack dataset (outer ring), compared to the frequency of tags generated by StarVector (middle ring), and compared to our most performant baseline (LIVE). For visualization purposes, we apply a logarithmic scaling over the counts and show the base percentage in parentheses. StarVector generates SVGs that contain tags with a similar distribution to the ground truth. The baseline is limited to paths or basic primitives.

- 4. LIVE achieves the best results in terms of pixel-based metrics like MSE, LPIPS, and SSIM (see Tables [1, 2, 3]). Using 32 paths, it effectively represents a wide range of images, making it highly versatile for vectorization tasks, including natural images, which StarVector cannot

handle due to its specialized training. However, LIVE has notable limitations. It relies on a slow test-time optimization process (approximately 10 minutes per sample, using 32 paths, Table 3) to refine SVG outputs for good MSE scores, often introducing unwanted visual artifacts.

Additionally, its exclusive use of path primitives results in significantly larger SVG files (19k tokens), as shown in Table1. In contrast, StarVector produces compact (3k tokens) and professional-grade SVGs by leveraging a variety of SVG primitives beyond paths, achieving higher precision and efficiency. Notably, we find that this method, and the broader family of differentiable techniques it belongs to, is unsuitable for generating images that require specific primitives, such as those in diagrams. Its performance on the SVG-Diagrams benchmark is poor, as illustrated in Figure 14.

cannot recognize or encode it semantically. Despite these limitations, these methods excel in generation speed, making them highly efficient for many tasks. As shown in Table 3, VTracer and AutoTrace can generate SVGs in under a second, while Potrace typically takes around 10 seconds. In comparison, StarVector requires over a minute, and other baselines can take anywhere from 10 to 20 minutes. While StarVector’s semantic richness and compact outputs make it better suited for certain applications, these image processing methods demonstrate a clear advantage in scenarios where speed is critical.

- 5. DiffVG offers comparable results as the other baselines in terms of reconstruction metrics, and it can produce suitable SVG image vectorization with 32 paths (same as LIVE), as seen in Tables [1, 2, 3]. This tables also show that DiffVG produces large SVGs as seen in the number of tokens (Tokens column), approaximately 19k tokens, similar to LIVE. This means that files are extremely large compared to the ones of StarVector. As mentioned before, this is due to StarVector leveraging understanding and SVG primitieves. Nevertheless, this method is substantially faster than LIVE, requiring 30 seconds per sample.
- 6. Image Processing Methods: VTracer, Potrace, Autotrace Previous image processing methods for Image-toSVG are powerful, excelling at fitting vector images to raster inputs with near-zero MSE while reliably capturing the shapes and colors of the original image. However, we identify several shared limitations across these methods: (1) lack of SVG file compression, as they often generate excessively long paths (see the Tokens column in Tables 1 and 2); (2) susceptibility to visual artifacts, especially with challenging patterns; and

(3) poor performance in vectorizing diagrams, as illustrated in Figure 14. On the SVG-Diagrams benchmark, VTracer, Potrace, and AutoTrace struggle with producing high-quality results. These methods perform best on images that can be segmented into distinct regions by color or texture but fail with complex patterns, such as small, closely spaced shapes or fine details. For example, in Figures 9 and 4, small polygons and intricate text are inadequately vectorized, with details often lost. All three methods are restricted to path primitives, limiting their ability to support features like optical character recognition or rendering text with the <text> tag. Although Potrace sometimes better preserves text compared to others, it still

- 7. IconShop achieves remarkable results on the SVGFIGR dataset, as demonstrated in Table 4 and Figures of their original work [94]. However, it is not designed to handle SVG-Stack due to its restriction to modeling only the path primitive. StarVector outperforms IconShop, as shown in Tables 4, across metrics such as FID, FID CLIP, and CLIP Score. Qualitative examples further highlight StarVector’s superior performance in Text-toSVG generation within the FIGR-SVG dataset.
- 8. LLMs for Code Generation. Methods utilizing LLMs to directly generate SVG code present appealing advantages. In our evaluation, we assessed GPT-4, GPT-4V, CodeLlama, and our proposed StarVector approach. By leveraging the code space, these models can utilize various SVG primitives based on their understanding of raster images, including path shapes and higher-order primitives like circles and text. This capability enables applications in new domains, such as diagram generation, as demonstrated by the results in Figure 14. However, most LLMs have not been specifically trained for the Image-to-SVG task, which limits their performance. In contrast, StarVector outperforms other LLMs due to its dedicated architecture and tailored training methodology, excelling in both image understanding and SVG generation. Upon reviewing the complete set of results, we conclude that the StarVector approach is the only deep learningbased Image-to-SVG model capable of achieving results comparable to those of VTracer, Potrace, and AutoTrace. Furthermore, StarVector paves the way for novel research avenues in vector graphics generation, enabling applications such as diagram generation, Text-to-SVG generation, and potentially enhanced editing and understanding of vector images.

###### 10.7. Human Evaluation

We conducted a human evaluation to compare the outputs of StarVector-8B, our best model, with those of the most powerful baselines. Participants were selected from diverse backgrounds and carefully screened for conflicts of interest, with none of the key authors involved. The evaluation was performed through a web interface (shown in Figure 31) that provided anonymized outputs and randomized sample presentations to prevent pattern recognition or bias. The results, presented in Figure 5, demonstrate a strong preference for StarVector-8B across all settings, especially in SVG-Diagrams tasks. This highlights a disconnect between pixel-based metrics (e.g., MSE, SSIM) and human visual perception of SVGs. While baseline models often prioritize pixel-perfect reconstruction, human evaluators preferred StarVector’s sharp, welldefined shapes and its effective use of primitives (Figure 30). Spearman correlation analyses between model metrics and human evaluations further emphasize this gap. MSE shows weak correlations (0.0596 and -0.1002), indicating its inadequacy as a predictor of human preferences. In contrast, DinoScore exhibits significantly stronger correlations, with values of -0.6193 and 0.6214. Moreover, a robust correlation of 0.7577 between differences in DinoScore and human evaluation scores highlights DinoScore as a more reliable metric for assessing SVG quality in alignment with human judgment.

###### 11. StarVector Method

Here, we provide details on the StarVector architecture, training recipe, and generation process.

###### 11.1. Architecture

###### 11.1.1. Large Language Model

We consider several aspects when choosing the LLM to handle the SVG code generation. First, we require an LLM that can handle large token contexts during training, as SVG code samples are typically of long lengths (i.e., between 1,000-4,000 tokens for the most common SVG datasets but growing arbitrarily for much more complex vector graphics). Second, we need fast decoding during the generation of these large contexts. Finally, we would benefit from models that have been extensively pre-trained on general coding tasks to avoid early training costs.

Some prior works offer open-source models that fit these requirements. We explored the open-source families of models CodeGen [53], StarCoder [42] and StarCoder2 [49].

We empirically find the StarCoder family to be the most suitable choice for our requirements. StarCoder offers a pre-trained model with 1B parameters

(starcoderbase-1b) and a context length of 8k tokens, making it ideal for smaller-scale experiments while maintaining efficient generation speeds. The model employs Multi-Query Attention and a Fill-in-the-Middle objective, trained on 1 trillion tokens, with a context window of 8192 tokens. Its compact size ensures compatibility with GPUs, facilitating data parallelism during training—a crucial benefit when fine-tuning all network parameters, including the memory-intensive image encoder.

The second generation, StarCoder2, extends the context length to 16,384 tokens, presenting an exciting avenue for exploring training on SVGs with longer context requirements. For this, we leverage the 7B parameter version (starcoder2-7b), which incorporates a sliding window attention mechanism of 4,096 tokens and was trained on over 3.5 trillion tokens of code using the same Fill-in-theMiddle objective [49]. This enhanced context capacity and training scale make it a promising candidate for scaling SVG-based experiments.

These two types of LLMs define the backbones of our two StarVector variants. StarVector-1B is based on the starcoderbase-1b architecture and weights, while StarVector-8B is based on the starcoder2-7b architecture and weights.

###### 11.1.2. Image Encoder

Our image encoding pipeline computes the images’ feature representations using a backbone image encoder and aligns them to the LLM via the Adapter module (see Figure 3). State-of-the-art image encoders are typically focused on natural images. However, our data contains images of logotypes, icons, fonts, or emojis, which usually contain no background (which we set to white) and mostly constant colors.

Note that the image encoder is used exclusively for the Image-to-SVG task and is not employed during the Textto-SVG task. For Image-to-SVG, images must be projected into a representation with the same dimensionality as that of the LLM. We train the model to enable the LLM to ingest these representations and generate SVG code sequentially.

To choose the best encoder, we draw inspiration from the success of pre-trained encoder backbones in downstream computer vision tasks such as classification [62], retrieval, and generation [23], including both convolutional and transformer-based models. Specifically, we experiment with CLIP ViT-B/32 [62], ConvNext [46] (pre-trained on LAION-2B [75]), and VQGAN [23], which we pre-train on an image reconstruction task using raster images from SVG-Stack. For CLIP, we have Lv = 257 embeddings, including the CLS token. For VQGAN, we use the prequantization layers and flatten them to obtain Lv = 196 embeddings. For ConvNext, we flatten the last activation map to get Lv = 49 embeddings.

We explore several image encoders based on different

[Figure 637]

Figure 31. The web interface used during the human evaluation.

paradigms. VQGAN [23] is based on learning to project images to discrete tokens. First, we fine-tune an Imagenet [19]-pretrained VQGAN on the SVG-Stack dataset with the VQ-adversarial reconstruction task. We find that using the features before the quantization yields better results. ConvNext [46] is a convolutional backbone, which we extract features before pooling. We start from a LAION2B [75]-pretrained checkpoint. Finally, ViT CLIP [62] is based on the Visual Transformer (ViT) [22] and is well prepared for autoregressive tasks. We extract all output representations. We use a LAION-2B pre-trained model. During the training of StarVector, all the parameters of the image encoders are updated. We find that the best choice is using CLIP. We consider that the gains in performance come from CLIP using more visual tokens (257) than the other image encoders.

The adapter first projects the features from the original dimensionality Dv to a dimensionality Dv × 2, followed by a Swish non-linear activation function and a linear projection to the LLM dimensionality Dl. Finally, we apply a layer normalization [5]. We initialize the adapter parameters with Glorot [27]. Dropout [79] of 0.1 is applied at the beginning. These hyperparameters were found using a random search on SVG-Fonts.

Our results show that image resolution is essential to capture fine-grained details like texts or high-frequency patterns. As seen in the SVG-Diagrams dataset in Figure 6), diagrams and figures are part of the SVG-Stack dataset and present challenging horizontal or vertical aspect ratios. When images have these aspect ratios, we make the image fit in the 224×224 resolution, losing much detail, especially

for the OCR capabilities of reading rendered texts and accurately displaying them.

Additional results comparing image encoders can be found in Figures 19 and 22. These results show the boost in precision obtained when using CLIP. VGQAN and ConvNext often fail to capture the image’s shape and the path’s trajectory. We note that ConvNext performs better than VQGAN. These differences are also due to the differences in the number of parameters. The CLIP ViT-L/14 model that we use consists of 290M parameters, VQGAN consists of 29M, and ConvNext consists of 179M parameters.

Generating SVGs from natural images is out of the scope of this project. However, future work should focus on adapting this approach to natural images, drawing from [50] and [12] to create a dataset of natural images and SVG pairs.

The selected image encoder architectures for StarVector include two variants: one with fewer parameters and reduced image resolution, based on the CLIP ViT-B/32 model, which processes images at 224x224 pixels and is utilized in StarVector-1B. The second variant, SigLip (siglip-so400m302 patch14-384), has a larger number of parameters and processes images at a higher resolution of 384x384 pixels, and is employed in StarVector8B. Given the positive results from the ViT architecture, we chose the SigLip variant due to its demonstrated effectiveness [97] and the enhanced resolution it provides.

###### 11.2. Training

For training the StarVector model, we define the task of Image-to-SVG as an inverse rendering problem that converts a raster image (represented with visual tokens) into a

sequence of SVG code. This can be viewed as a sequenceto-sequence problem that models the translation between the image and SVG code domains. As detailed in Section 11, we utilize a CLIP ViT-B/32 for StarVector-1B and SigLip for StarVector-8B as image encoders, along with a non-linear adapter to generate a sequence of visual tokens.

The training process consists of two stages. In the first stage, the Image-to-SVG training phase, we construct sequences of visual tokens (produced by the image encoder and adapter) and SVG tokens, separated by a trigger token, <svg-start>. We train the LLM to learn these sequences on a large SVG-Stack dataset using a basic language modeling loss that calculates the cross-entropy loss in predicting the next token in a sequence based on the previous tokens.

This task enables the model to learn the concept of drawing with SVG vectors that resemble the input image. Importantly, this training can occur without supervision in the image domain (i.e., without pixel loss), relying solely on categorical cross-entropy loss for the LLM vocabulary introduced by the next-token prediction task.

In the second stage, we fine-tune the checkpoint from the first stage, which has learned SVG syntax through the Image-to-SVG task, on the Text-to-SVG task. During this phase, the image encoder is disregarded, as it becomes a Text-to-text task where the text instructions and SVG codes can be tokenized and processed directly by the LLM.

Training Details. We trained StarVector-1B on 1 node of 8 A100 80GB GPUs using Accelerate with DeepSpeed stage 2 and StarVector-8B on 8 nodes of 8 H100 80GB GPUs with Fully Shared Data Parallel (FSDP). For Image-to-SVG, we used total batch sizes of 128 and 512 for StarVector-

- 1B and -8B, respectively, a learning rate of 1e-5, and the AdamW optimizer. To optimize memory and computation, we employed bf16 precision, FlashAttention2, and gradient checkpointing. StarVector-1B took 7 days to train, while StarVector-8B took 10 days, with both models completing
- 2 epochs.

We use HuggingFace Transformers [93] and PyTorch [58] for the implementation. We use a batch size of 2. Images are processed with a resolution of 224x224, as defined by the pre-trained CLIP image encoder, and process a maximum of 8192 tokens, considering the 257 visual tokens and the rest for the SVG tokens. We use gradient batch accumulation of 8 and train on a data parallel setup with 4 A100 80GB GPUs, having an effective batch size of 64. The learning rate is set to 5 × 10−4 for training, using AdamW optimizer [48] for approximately five days of training on the SVG-Stack dataset.

###### 11.3. Generation

Here, we describe how to sample SVG code from our model. As a decoder-only LLM [42], StarVector first computes the key-value (KV) cache using the visual tokens from the image and then produces the initial set of output logits. This stage is often quick because the model can process the entire visual token sequence simultaneously [78]. The selected token from the output logits is then input back into the model, which generates logits for the subsequent token. This process is iteratively repeated until the model produces the desired quantity of tokens. Our approach uses architectural improvements for fast decoding, such as FlashAttention [18] and Multi-Query Attention [77]. We leverage vLLM to improve inference speed.

We perform a grid search on SVG-Emoji and SVG-Icons validation sets to select the correct sampling temperature. The choice of temperature does not strongly impact the results. However, a 1-point increase in performance is observed on CD for SVG-Emoji using temperatures close to 1.0.

We also present an ablation study of StarVector-1B popular decoding techniques [31, 52, 76, 85]. Specifically, we experiment with greedy decoding, beam search, and nucleus sampling with top-p. Results are shown in Table 7. The use of nucleus sampling with top-p=0.9 and temperature T=0.5 (no beam search) shows to be the best option. The beam search improves the greedy decoding baseline but does not work well when combined with nucleus sampling, increasing the inference time. In sum, we recommend nucleus sampling [31] with top p=0.9 and temperature between 0.5 and 0.9 for the best performance.

Are the SVGs valid and compilable? A common issue when generating SVGs with our approach is that the maximum token length of the LLM might not be sufficient to complete the SVG code, leading to compilation errors. We find that 85% of the generated SVG fit within the context length and compile successfully. The remaining incomplete samples are post-processed with cairosvg to produce a complete and compilable SVG. However, in some cases, parts of the image may be lost during this process. With this technique, 100% of the generated SVGs are valid and compilable.

Improving SVG Quality Through Sampling. The generation process is stochastic, meaning the outputs may sometimes take an incorrect path, leading to failed generations or repetitive patterns. To address this, we propose a simple baseline approach: generate k SVG outputs with varying sampling parameters (e.g., by adjusting the temperature),

then compare the outputs with the ground truth using a visual metric (we propose DinoScore) to select the most accurate result. In an ablation study conducted on SVG-Stack using StarVector-8B with k = 1 and k = 5, we observe a boost in DinoScore by 0.12. Empirically, after sampling 100 test samples, we find that 32% of the SVGs are more accurate when using k = 5, though this increases the generation time by a factor of k. The use of vLLM helps mitigate the slower sampling process, as it operates much faster. For further improvement in code generation, previous work has used MCTS techniques [7], which leverage visual feedback more effectively to guide the generation and enhance the stochastic sampling process.

