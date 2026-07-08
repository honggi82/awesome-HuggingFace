# arXiv:2504.01934v2[cs.CV]3Apr2025

[Figure 1]

## ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement

Runhui Huang2∗ Chunwei Wang1∗ Junwei Yang1 Guansong Lu1 Yunlong Yuan1 Jianhua Han1 Lu Hou1 Wei Zhang1 Lanqing Hong1 Hengshuang Zhao2† Hang Xu1‡ 1Huawei Noah’s Ark Lab, 2The University of Hong Kong

###### Image Understanding

Image Generation

[Figure 2]

###### Poster

Chart

[Figure 3]

[Figure 4]

640×1920

[Figure 5]

[Figure 6]

[Figure 7]

Question: What was the value of the commercial property market in 2016? Answer: 883

[Figure 8]

512 ×512

Math

1024×1024

[Figure 9]

1152×768

Question: As shown in the figure, AB is the diameter of ⊙O, CD is the chord of ⊙O, ∠ADC

[Figure 10]

= 26.0, then the degree of ∠CAB is () Choices: (A) 26° (B) 74° (C) 64° (D) 54°

Answer: Solution: Since AB is the diameter of circle O, we have angle ACB = 90 degrees. Also, angle B = angle ADC = 26 degrees. Therefore, angle CAB = 90 degrees - angle B = 90 degrees 26 degrees = 64 degrees. Hence, the answer is C.\nAnswer:C

1152×768

[Figure 11]

General

[Figure 12]

Question: How many hot air balloons are in the image? A. 1 B. 3 C. 2 D. 4 Answer: B

768×1152 1536×768

1192×640

[Figure 13]

[Figure 14]

Question: How many objects are in the image? A. 1 B. 2 C. 3 D. 4 Answer: A

Question: What are the other types of Tennis field surface rather than Clay and Hard? Answer: grass, carpet

512×2048

###### Image Editing

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Color Modification: turn the kid‘s shirt to blue

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Add a birdhouse hanging from the branches

Replace the chandelier with industrial lights

Delete the tennis racket from the man’s hand

Transfer to the style of comic book

Change it into winter

Object Removal: Remove the fruit bowl on the table

Original image

ILLUME ILLUME+

Figure 1: ILLUME+ can understand and generate images at any resolution. Compared to our previous work, ILLUME [63], it demonstrates improved texture preservation in image editing tasks.

∗Equal contribution, †Corresponding author, ‡ Project leader.

Preprint.

### Abstract

We present ILLUME+, an enhanced version of the previous ILLUME model, which leverages dual visual tokenization and a diffusion decoder to improve both deep semantic understanding and high-fidelity image generation. Existing unified models have struggled to simultaneously handle the three fundamental capabilities expected of a unified model: understanding, generation, and editing. Models like Chameleon and EMU3 utilize VQGAN for image discretization. Due to the lack of deep semantic interaction, they lag behind specialist models like LLaVA in visual understanding tasks. LaViT and ILLUME employ semantic encoders for tokenization, but they struggle with image editing due to poor texture preservation. Meanwhile, Janus series decouples the input and output image representation, limiting their abilities to seamlessly handle interleaved image-text understanding and generation. In contrast, ILLUME+ introduces a unified dual visual tokenizer, DualViTok, which preserves both fine-grained textures and text-aligned semantics while enabling a coarse-to-fine image representation strategy for multimodal understanding and generation. Additionally, we employ a diffusion model as the image detokenizer to enhance generation quality and enable efficient super-resolution. ILLUME+ follows a continuous-input, discrete-output scheme within the unified Multimodal Large Language Model (MLLM) and adopts a progressive training procedure that supports dynamic resolution across the vision tokenizer, MLLM, and diffusion decoder. This design allows for flexible and efficient context-aware image editing and generation across diverse tasks. ILLUME+ (3B) exhibits competitive performance against existing unified MLLMs and specialized models across multimodal understanding generation, and editing benchmarks. Its support for flexible high-resolution images enhances visual understanding tasks and enables detailed image synthesis up to 1024×1024 resolution. With its strong performance, ILLUME+ provides a scalable foundation for future multimodal model applications. Project Page: https://illume-unified-mllm.github.io/. Code and models will be publicly available soon.

### 1 Introduction

“What I cannot create, I do not understand.” ——Richard Feynman

Recent advancements in Large Language Models (LLMs) have significantly enhanced their capability to handle multimodal tasks, particularly by integrating visual inputs into language models. Efforts such as the LLaVA series and the QwenVL series [38, 37, 1, 64] have demonstrated remarkable visual comprehension performance. Meanwhile, the development of text-to-image generation models, such as diffusion-based approaches [53, 50, 51, 54] and more recent autoregressive approaches [52, 16, 74], has made substantial strides in generating high-fidelity images. These developments have driven the push towards creating unified Multimodal Large Language Models (MLLMs) that seamlessly integrate both visual understanding and generation capabilities. A unified model holds promise not only for advancing task coordination and generalization but also for contributing to the exploration of artificial general intelligence (AGI). By merging understanding and generation capabilities within a single framework, unified models can genuinely grasp the deep relationships between visual and textual information, enabling more intelligent and flexible interactions and task execution in complex real-world scenarios.

To build such unified models in an autoregressive framework, existing approaches have explored several distinct paradigms. As illustrated in Fig. 2 (a), the earliest models, e.g., Chameleon [59] and Emu3 [65], utilized VQGAN [18] to discretize images, enabling a shared vocabulary for text and vision within an autoregressive (AR) framework. However, even with large-scale training, they lag behind models like LLaVA [38] in visual understanding tasks. To address this issue, works like LaViT [28] and ILLUME [63] (Fig. 2 (b)) learn the semantic codebook from the pretrained text-aligned semantic encoders [78, 84], and employ diffusion models to reconstruct images based on semantic tokens. This improves image-text alignment in MLLM pretraining and achieves strong performance in both understanding and generation tasks. However, the lack of texture preservation in vision tokenizers limits their capability in fine-grained image editing. To mitigate this, another

Pixel Detokenizer

Diffusion Decoder

Detokenizer

Pixel Detokenizer

Unified Detokenizer or Diffusion Decoder

Separate Head

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

| |
|---|

MLLM

MLLM

MLLM

MLLM

MLLM

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

Text Tokenizer

Text Tokenizer

Text Tokenizer

Text Tokenizer

Semantic Encoder

Semantic Encoder

Pixel Tokenizer

Pixel Encoder

Text Tokenizer

Semantic Encoder

RQVAE Tokenizer

Image Text

Image Text (c) e.g., ViLA-U

Image Text (d) Janus, Janus-Pro

Image Text (b) e.g., LaViT, ILLUME

Image Text (e) ILLUME+ (Ours)

(a) e.g., Chameleon, EMU3

(a) (b) (c) (d) (e)

MJHQ30K

GenEval

[Figure 31]

GenAI-bench

###### Key Design of Unified Models

###### Advantages

POPE

√

√

√

√ √ √ √ √

Faster image-text alignment

Pre-text-align vision encoder

×

EMU-edit

Texture preservation

Support fine-grained image editing

√

√ × √

√

×

MMMU

No quantization loss for input

Higher performance on understanding

√

√

×

Unified image representation

Support multi-turn and interleaved generation

√

√ √

ChartQA

×

Unified text and image head Unified cross-modal modeling, scalable

√

× √

MMB

Preferable Abilities Perform well on understanding tasks

DocVQA

√ √

√ √ ×

√ √ √ √

× √

× √

SEED

Perform well on generation tasks

InfoVQA VQA-text

AI2D

× ×

× ×

×

Perform well on editing tasks Support any resolution image input & output

×

×

- Figure 2: Characteristics comparison among existing unified models. Existing methods explore distinct paradigms to balance visual understanding, generation, and editing capabilities. Early approaches using VQGAN discretization struggle in understanding and context-aware generation tasks due to limited semantic alignment. Later frameworks incorporate semantic encoders, achieving better alignment but compromising texture preservation essential for fine-grained editing. ILLUME+ deep-integrates image understanding, generation, and editing into a single, unified architecture, enabling more intelligent and flexible interactions and task execution.

line of work (Fig. 2 (c)) integrates pre-aligned vision encoders with image-text contrastive learning during VQGAN training. Due to the usage of RQVAE to balance the pixel reconstruction and imagetext alignment, it requires a separate head for image generation, increasing design complexity and potentially posing a bottleneck when scaling up the models. Finally, methods in Fig. 2 (d) decouple image understanding and generation tasks by employing the semantic encoder and VQGAN tokenizer independently. While effective, this structure fails to support interleaved image-text tasks and multiturn dialogues, as it necessitates manual specification of whether a task involves understanding or generation, contradicting the flexibility expected of a unified foundation model.

By reviewing the current research, we suggest that a strong unified foundation model should demonstrate three core competencies: visual understanding (accurate image interpretation), generation (high-quality image synthesis), and editing (instruction-following modification while maintain consistency of other parts). These are fundamental capabilities that guarantee the model has the potential to benefit from scaling in both model capacity and task diversity. Then it raises a question: How can we construct such a unified foundation model? We summarize the following key design principles:

Pre-text-align vision encoder. Previous studies [45, 62] have shown that the pre-text-aligned vision encoder, i.e., CLIP-like models, significantly benefit to visual understanding capability. Additionally, in contrast to VQGAN-based models (Fig. 2 (a)) that purely supervised by image reconstruction, these semantic encoders facilitate image-text alignment as demonstrated in ILLUME [63]. Thus, incorporating a pre-text-aligned semantic encoder is essential for a unified foundation model.

Image texture preservation. The quality of image reconstruction of vision tokenizers is essential for handling editing tasks, which determines the up-bound of unified models to maintain consistency of unchanged regions in images. Therefore, not only semantic information but also texture preservation is required as a crucial consideration in vision tokenizer design choices.

No information loss for image input. While the vision tokenizer is the key to enabling unified autoregressive image-text generation, it inevitably introduces information loss during the quantization process. To this end, using continuous features before the quantizer of vision tokenizer as visual input for LLM serves as a more suitable choice to guarantee fine-grained multimodal understanding capability.

Unified image input and output representation. The decoupled mechanism in Janus series [67, 11], i.e., semantic representation for visual input while pixel representation for visual output, inevitably hinders the model’s ability to accurately interpret and further modify its own visual outputs in multiround steps. Hence, a unified representation for visual input and output is necessary to further support image-text interleaved generation, multi-turn dialogues and chain-of-thought reasoning.

Unified text and image head. In a single autoregressive framework, a unified output head for both image and text is preferable, as it not only simplifies infrastructure design but also enhances crossmodal interactions. In contrast, the requirement of a separate image head in Fig. 2 (c) introduces challenges in modality switching during generation. For example, as we need special line separator tokens interleaved with visual tokens to represent an image in different resolutions, how can the model seamlessly transition between text and image heads during inference? To avoid such complexity, a unified head offers a more effective and elegant solution.

Building on the above analysis, we introduce ILLUME+, an enhanced version of the ILLUME model that encompasses all the key designs mentioned above but also exhibits all the preferable abilities, listed in Fig. 2’s Table). ILLUME+ supports flexible any-resolution visual input and output and excels in multimodal understanding, generation, and editing tasks, as demonstrated in Fig. 1. Its key features are outlined below:

Dual vision tokenizer for semantic and texture preservation. We introduce the DualViTok, a dual-branch vision tokenizer designed to capture both deep semantics and fine-grained textures. The semantic branch utilizes a pre-trained text-aligned vision encoder for semantic feature extraction, supervised by feature reconstruction loss. In parallel, the pixel branch integrates quantized features from both the semantic encoder and a CNN-based pixel encoder to enhance pixel-level reconstruction. To improve robustness against incorrect token predictions in autoregressive generation, we introduce noise injection during training by randomly perturbing visual tokens. Despite its simplicity, DualViTok is specifically designed for unified models, ensuring both semantic and texture preservation while maintaining robust token decoding.

Unified MLLM with unified coarse-to-fine image representation. Unlike the Janus series [67, 11], which decouples visual input and output representations, we adopt a coarse-to-fine strategy, first generating semantic tokens followed by pixel tokens. This sequential arrangement enables LLMs to utilize a unified LM head with a simple vocabulary expansion while leveraging semantic visual tokens as a bridge to enhance alignment between text and visual textures. Additionally, to prevent information loss at the input stage, we employ a continuous-input, discrete-output scheme following ILLUME [63], using pre-quantized continuous features as inputs while generating discrete tokens for image synthesis.

Diffusion decoder for enhanced generation quality and efficient super-resolution. We incorporate a diffusion model as an optional choice for image generation, offering two key benefits: (i) Higher generation quality. Diffusion models refine details and reduce artifacts, surpassing direct token decoding from a vision tokenizer in both fidelity and robustness. (ii) Efficient super-resolution. They upscale images during decoding, mitigating the token explosion issue in autoregressive highresolution generation.

Progressive training procedure for flexible resolution visual input and output. We employ a progressive training procedure for all the above three modules, gradually increasing resolution from fixed low to flexible high, to ensure training stability and final performance. Additionally, during MLLM training, we incrementally increase tasks diversity and complexity, with carefully designed data distribution for each stage.

Based on these design choices, ILLUME+ with only a 3B LLM excels among existing unified MLLMs and exhibits competitive performance against specialized models across multimodal understanding, generation, and editing benchmarks. Its support for high-resolution images enhances both visual understanding and detailed image synthesis, surpassing existing unified models in document-oriented tasks and enabling generation up to 1024×1024 resolution. Additionally, ILLUME+ improves texture preservation over ILLUME in image editing as illustrated in Fig. 1. ILLUME+’s robust performance across these diverse benchmarks highlights its effectiveness in unifying understanding, generation, and editing, providing a scalable solution for future multimodal applications.

### 2 Related Work

MLLM for image understanding. Recent advancements in Large Language Models (LLMs) have led to the development of Multimodal Large Language Models (MLLMs) for image understanding. Early models like LLaVA [39] and MiniGPT-4 [82] used vision adapters to align visual features with LLMs, showing strong performance in visual perception tasks. Later models such as QwenVL series [1, 64], and InternVL series [13, 12] improved upon this with higher-quality datasets and better training strategies, but still focus primarily on visual understanding. However, despite the strong understanding capabilities of these models, they primarily focus on visual perception and comprehension. This highlights the need for more comprehensive solutions that unify both understanding and generation, allowing models to learn deeper relationships between visual and textual information, enabling more intelligent and flexible interactions and task execution in complex real-world scenarios.

Image generation model. Generative adversarial networks (GANs [23]) is the pioneering method for image generation in the era of deep learning. However, it suffers from the problem of unstable training process and mode collapse. In recent years, diffusion-based methods [26, 53, 50, 54, 2] have shown excellent image generation capabilities. These models learn to predict Gaussian noise in a forward diffusion process, and then generate high-quality images through an inverse denoising process. Among them, the latent diffusion model [53, 51, 17] addresses computational challenges by creating images from low-dimensional latent representations. Another line of research turns to explore the image generation in the autoregressive model [52, 16, 74, 61], which converts images into discrete tokens using VQGAN-like vision tokenizers [18, 30] and generates images by predicting the tokens autoregressively. In this paper, we unify the image generation and image editing into one MLLM with image understanding tasks under the autoregressive manner, and further adopt a diffusion model to further improve the reconstruct quality from the predicted tokens.

Unified multimodal understanding and generation. Early efforts to unify visual understanding and generation using LLMs include models like Emu [58] and X-VILA [73], which adopt unified autoregressive approaches to predict multimodal elements. However, the non-unified optimization of different modalities limits feature integration, and additional components like diffusion decoders reduce efficiency. Models such as LWM [40], Chameleon [59], and VILA-U [68] use VQ tokenizers to convert images into vision tokens, enabling a unified training framework for text and image generation. Despite these advancements, challenges remain in integrating understanding and generation. Janus series [67, 11] decouples visual encoding for understanding and generation, which may suffer from misaligned representations due to separate branches for understanding and generation. These limitations highlight the need for better solutions that allowing flexible and efficient context-aware image understanding and generation across various tasks.

### 3 Method

- Figure 3 provides an overview of our proposed framework, ILLUME+, which comprises a dual vision tokenizer, a MLLM, and a diffusion decoder. Our architecture’s core design principle is the unified dual visual tokenization mechanism that captures both deep semantic information and fine-grained texture details, ensuring a comprehensive image representation for visual understanding, generation, and editing tasks. The following section elaborates on the architectural details, training procedures, and data composition.

#### 3.1 Dual Vision Tokenizer

Although vision tokenizers have been studied for years, their design for unified models that simultaneously support understanding, generation, and editing remains an open challenge. To mitigate this, we propose the Dual Vision Tokenizer (DualViTok) tailored for unified models with semantic and texture preservation and robust decoding.

Semantic and texture information preservation. As shown in the Fig. 3 (a), DualViTok incorporates a dual-branch design to learn both deep semantics and fine-grained textures. The semantic branch leverages the pre-trained text-aligned vision encoder, QwenViT [64], to extract high-level semantic features, which are then quantized into discrete tokens and reconstructed using a lightweight decoder. The semantic reconstruction is optimized with a cosine similarity loss between the reconstructed

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Text Detokenizer

Output Text

Diffusion Decoder

Reconstructed Semantic Features

Semantic

Semantic

QuantizerQuantizer

Encoder

Decoder

The image contains one wolf.

(768×1152)

(1024×1024)

… …

…

| | |
|---|---|
| | |

[Figure 36]

[Figure 37]

[Figure 38]

Discrete Tokens

Discrete Tokens Discrete Tokens

(512×2048)

Output Image

| | | |
|---|---|---|
| | | |

(512 ×512)

(384 ×576)

###### MLLM

[Figure 39]

[Figure 40]

[Figure 41]

(256 ×1024)

Encoder

Decoder

Input Image

…

… …

Pixel

Pixel

Input Text

C

(512 ×512)

(384 ×576)

[Figure 42]

Continuous Features Discrete Tokens

Continuous Features

How many wolves in the image? Editing the image: Add two gray wolves standing in the foreground snow.

(256 ×1024)

[Figure 43]

[Figure 44]

Reconstructed Image

[Figure 45]

[Figure 46]

Adapter Adapter

(a) Dual Vision Tokenizer

(512 ×512)

(384 ×576)

[Figure 47]

[Figure 48]

[Figure 49]

Text Tokenizer

Semantic Encoder

Pixel Encoder

[Figure 50]

(256 ×1024)

Up Sampling

Input Image

[Figure 51]

[Figure 52]

(c) MLLM Architecture

Quantized Features Denoising

C

U-Net

(1024 ×1024)

(768 ×1152)

384 576 SOI SOS EOL EOL EOS SOT EOL EOL EOT EOI

27 9 14 31 2 17 26 6

[Figure 53]

[Figure 54]

height/width indicator start/end-of-image start/end-of-semantic Start/end-of-pixel end-of-line

(512 ×2048)

Reconstructed Image Gaussian Noise

(d) Coarse-to-fine Unified Image Representation

(b) Diffusion Decoder

- Figure 3: Architecture of ILLUME+. (a) The dual vision tokenizer preserves both semantic and texture information. (b) The diffusion refiner decodes discrete tokens into high-quality images. (c) The unified MLLM enables deep semantic interactions and context-aware image generation. (d) We introduce an unambiguous image representation of discrete tokens in a chain-of-thought pattern (semantic tokens first, followed by pixel tokens), resulting in improved generation performance.

features and semantic features from the vision encoder. Meanwhile, the pixel branch follows a MoVQGAN-based architecture [81]. After quantization, the pixel and semantic tokens are concatenated along the channel dimension for image decoding. Following standard VQGAN procedures [18], the pixel branch is trained with L1 loss, perceptual loss, and GAN loss. More specifically, for the semantic branch, we adopt a 28× downsampling rate commonly used in state-of-the-art understanding models [32, 64] and employ a 16× rate for the pixel branch to preserve fine-grained textures. To further minimize the information loss induced by tokenization, we incorporate space-to-channel and channel-to-space transformations inspired by DC-AC [7] in the downsampling and upsampling process. Additionally, to ensure reconstruction fidelity, we employ larger codebook sizes compared with previous works [18], 32,768 for the semantic branch and 98,304 for the pixel branch, with SimVQ [83] as quantization method to maintain high codebook utilization rate. Please refer to ablation studies in Sec. 4 for more discussions about detailed design choices of our vision tokenizer.

Robust decoding for incorrect token predictions. LLMs may occasionally predict incorrect visual tokens, and the tokenizer decoder should be resilient to such errors to minimize artifacts in the generated images. To improve robustness, we introduce noise injection during tokenizer training: each sample has an α=10% probability of being perturbed, with β=10% of its tokens randomly replaced, helping the decoder better handle erroneous token predictions.

#### 3.2 Unified Multimodal Large Language Model

As shown in Fig. 3 (c), ILLUME+ inherits the architecture of existing VLMs [37, 38] by extending LLMs with an additional vision vocabulary to generate discrete vision tokens, employing a continuousinput, discrete-output scheme for image processing. To avoid information loss caused by tokenizer quantization, we employ both the semantic encoder and pixel encoder within dual vision tokenizers to extract features from input images, which are then aligned with the LLM’s input space via two separate vision adaptors. For visual generation, images are converted into discrete tokens, enabling a unified modeling of visual and text tokens within the LLM. By using the same next-token prediction loss, both modalities are seamlessly integrated into a shared prediction head.

During inference, we apply the classifier-free guidance (CFG) following [70, 63] for text-to-image generation and image editing tasks, where the unconditioned setting serves as masked text descriptions and editing instructions, respectively.

Coarse-to-fine unified image representation. As illustrated in Fig. 3 (d), we adopt a unified representation for images with a coarse-to-fine sequence arrangement, i.e., first generating semantic tokens followed by pixel tokens. Since semantic representations align more naturally with text, generating semantic tokens first allows the model to determine content before refining details based on semantic information, thus enhancing alignment between text and visual textures. Specifically,

###### Dual-Tokenizer Training Diffusion Decoder Refinement MLLM Stage 1 Visual Embedding Initialization

MLLM Stage 3 Supervised Fine-tuning

MLLM Stage 2 Unified Image-Text Alignment

###### MLLM

MLLM

###### MLLM

[Figure 55]

[Figure 56]

[Figure 57]

Semantic Decoder

Pixel Decoder

Diffusion Decoder

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

LM Head (Vision) LM Head (Text)

LM Head (Vision) LM Head (Text)

LM Head (Vision) LM Head (Text)

[Figure 64]

[Figure 65]

[Figure 66]

Transformers

Transformers

Transformers

###### C

C

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Vision Embedding Word Embedding

Vision Embedding Word Embedding

Word Embedding

[Figure 73]

[Figure 74]

Quantizer Quantizer

Quantizer Quantizer

Vision Embedding

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

Adapter Text Tokenizer

Adapter Text Tokenizer

Adapter Text Tokenizer

Adapter

Adapter

Adapter

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Semantic Encoder

Semantic Encoder

Pixel Encoder

Pixel Encoder

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Pixel Encoder

Pixel Encoder

Semantic Encoder

Semantic Encoder

Pixel Encoder

Semantic Encoder

- Figure 4: Illustration of our progressive training pipeline. We first pre-train the dual-tokenizer system by reconstruction of the semantic and pixel information. We then fine-tune the diffusion model as a high-quality image decoder. The MLLM training consists of three main stages that gradually increase task resolution and complexity.

we use <start-of-image/semantic/pixel> and <end-of-image/semantic/pixel> markers to indicate the boundaries of the entire image, semantic representation, and texture representation, respectively. Additionally, <end-of-line> tokens are inserted at the end of each row to distinguish different resolutions, while the height and width indicators at the sequence’s start provide explicit resolution information during image generation.

#### 3.3 Diffusion Decoder

We introduce an additional diffusion model to decode image from predicted discrete tokens for enhanced generation quality and efficient super-resolution. Specifically, based on the SDXL model [51], our diffusion decoder replaces text encoders with zero embeddings in the cross-attention layers. Semantic and pixel tokens from dual vision tokenizer are mapped to feature representations using the learned codebooks and then injected into the UNet model by concatenation with the noisy image latent. Note that the model performs super-resolution, doubling the image size (i.e., 256×256 to 512×512), to mitigate the token explosion issue in autoregressive high resolution generation. During training, similar to our dual vision tokenizer, random perturbations are applied to 50% of the samples, replacing 10% of the tokens to enhance robustness against noisy input tokens. To prioritize semantic tokens, we use a smaller masking probability (10%) for semantic features and larger (50%) for texture.

#### 3.4 Training Procedure and Data Composition

- Figure 4 illustrates the whole training procedure of our proposed ILLUME+, where we adopt a progressive training paradigm designed to support flexible resolutions and leverage the capabilities of each component. An overview of data distribution at each stage is illustrated in Fig. 5.

Dual vision tokenizer training. As illustrated in Fig. 4, all components of our proposed DualViTok except for the pretrained semantic encoder are trainable. To ensure stable training across arbitrary resolutions, we progressively scale the input resolution: starting from a fixed 256×256, then 512×512, and finally allowing flexible resolutions up to 512×512. For training efficiency, we employ a bucketresolution strategy, batching samples with similar resolutions. The training corpus consists of 63M samples of various types with the data composition shown in Fig. 5.

Diffusion decoder training. We incorporate an additional diffusion model to decode image for enhanced generation quality and efficient super-resolution. In this stage, the two encoders and codebooks from DualViTok are frozen, while the pixel decoder is replaced by a diffusion-based decoder to reconstruct and and upscales images by a factor of 2. To support flexible resolutions, we predefine 11 commonly used aspect ratios: {1:1, 3:4, 4:3, 2:3, 3:2, 1:2, 2:1, 1:3, 3:1, 1:4, 4:1}. Each image is matched to the closest predefined aspect ratio and cropped accordingly. Note that images that cropped more than 20% of the original content is removed during training to preserve image integrity. For high-resolution support and efficient training, we adopt a two-stage process: the first stage handles images with a total pixel count near 5122, while the second stage scales up to approximately 10242. The training data consists of a 10M-image subset from our tokenizer dataset.

MLLM training. Following ILLUME, the training procedure consists of three stages as below, where we progressively unfreeze more parameters and increase the complexity and variety of tasks.

|23.9%<br><br>1%<br><br>7.8%<br><br>13.3%<br><br>13.7% 20.6%<br><br>19.6%|
|---|

10.8% 8.5%

Stage Dataset Tokenizer & Diffusion COYO [4], EMOVA, in-house aesthetics data

1.1%

9.7%

7.8%

14.9%

10.3% 2.6%

22.5%

14.0%

17.5%

25%

COYO, Wukong [24], EMOVA-Pretrain [9], LLAVA-SFT [39], in-house aesthetics data, UltraEdit [80], SEED-Edit [21], AnyEdit [75] Magpie [71], OpenOrca [35], SCP-116K [44], OpenHermes [60], OPC-SFT-Stage1 [27]

75%

- Stage 1 (5M)
- Stage 2-2 (19M)

MLLM Pre-training

64.6%

Stage 2-1 (46M)

55.5%

60.4%

EMOVA-SFT, Pixmo [15], M4-Instruct [37] COYO, in-house aesthetics data, OmniEdit [66], AnyEdit [75], UltraEdit [80], Instruct-Pix2Pix [3], Magpie [71]

###### Dual-Tokenizer(63M) & Diffusion (10M)

MLLM Pre-training (70M)

###### MLLM SFT (9.7M)

MLLM SFT

Text-to-Text Image-to-Text (General) Text-to-Image Image Reconstruction Image-to-Text (OCR, Doc, Chart) Image-to-Text (Math & Science)

Natural image Portrait Aesthetic image Math/Science

Image Editing

Document/Chart/Screen

Multi-Image

- Figure 5: Summary of the data mixture in each stage. Our training data gradually covers a wide range of tasks and various image resoluton.

MLLM Settings

DualViTok Diffusion Decoder

Stage 1 Stage 2-1 Stage 2-2 SFT

Vis. Adapter 1e-3 Vis. Adapter 5e-5 Vis. Adapter 2e-5

Learning Rate 1e-4 2e-5

Vis. Embed. & Head 2e-4 LLM 5e-5 LLM 2e-5, ViT 2e-6 Batch Size 256 128 1024 1024 512 256 Training Steps 270k/50k/78k 265k 5k 98k 40k 40k Image Res. Mode Fix/Fix/Anyres Multi-ratio Fix Fix Fix Anyres Image Main/Max Res. 256/512/512 512/1024 256 256 512 1024

Table 1: Training hyperpparameters of experiments.

- Stage 1: Visual Embedding Initialization. The primary goal of this stage is to initialize a good visual representation. We optimize vision-related components, namely the adapter, the vision embedding, and the LM head of the vision part, on image reconstruction and image captioning tasks. In this stage, we fix the image resolution as 256 × 256.
- Stage 2: Unified Image-Text Alignment. This stage focuses on image-text alignment to learn on multimodal data. We unfreeze the LLM and vision adaptor, with training data covering a variety of tasks, including text data, image caption data for both natural images and documents, text-to-image generation, and image editing data. This process contains two sub-stages: the first sub-stage fixed image resolution as 256 × 256 while the second one uses fixed image resolution as 512 × 512.
- Stage 3: Supervised Fine-tuning. After pretraining, we train the whole model with task-specific data to handle various multimodal understanding, generation, and editing tasks. In this stage, we leverage flexible resolution training. Specifically, for understanding tasks, it processes images with their naive resolutions, while for image generation and editing tasks, each input image is matched to a predefined aspect ratio and resize-and-crop accordingly with a total pixel count near 5122. Note that benefit to the diffusion decoder, our final model enables generation up to 1024 × 1024 resolution with multiple aspect ratios.
- 4 Experiments

#### 4.1 Implementation Details

In our experiments, we use Qwen2.5 [72] as the base LLM. In our DualViTok, the semantic encoder uses a pretrained QwenVIT [64]. The semantic decoder consists of 4 attention blocks with 2D-RoPE. The pixel encoder and decoder follow MoVQGAN-based architecture [81] with the basic channel of 128 and 384, respectively. We use a codebook size of 32,768 for the semantic branch and 98,304 for the pixel branch. Codebook dimensions are 32 for both semantic and pixel codebooks. We apply AdamW optimizer [43] without weight decay and constant learning rate for DualViTok, diffusion decoder and MLLM. The specific training hyperparameters of three parts are summarized in Table 1. The training process of DualViTok and the diffusion decoder took around 3+3 days on a cluster of 256 Ascend NPUs. Then 3B MLLM model took about 13 days to finish the 3-stage training.

General Doc Method LLM.

POPE MMBench SEED MME-P MM-Vet MMMU AI2D VQA-text ChartQA DocVQA InfoVQA OCRBench Understanding Only

InstructBLIP [14] Vicuna-7B - 36.0 53.4 - 26.2 30.6 33.8 50.1 12.5 13.9 - 276 Qwen-VL-Chat [1] Qwen-7B - 60.6 58.2 1487.5 - 35.9 45.9 61.5 66.3 62.6 - 488 LLaVA-1.5 [37] Vicuna-7B 85.9 64.3 58.6 1510.7 31.1 35.4 54.8 58.2 18.2 28.1 25.8 318 ShareGPT4V [10] Vicuna-7B - 68.8 69.7 1567.4 37.6 37.2 58 60.4 21.3 - - 371 LLaVA-NeXT [38] Vicuna-7B 86.5 67.4 64.7 1519 43.9 35.1 66.6 64.9 54.8 74.4 37.1 532 Emu3-Chat [65] 8B from scratch 85.2 58.5 68.2 - 37.2 31.6 70.0 64.7 68.6 76.3 43.8 687

Unify Understanding and Generation

Unified-IO 2 [46] 6.8B from scratch 87.7 - 61.8 - - - - - - - - Chameleon [59] 7B from scratch - - - - 8.3 22.4 - - - - - LWM [40] LLaMA-2-7B 75.2 - - - 9.6 - - 18.8 - - - Show-o [70] Phi-1.5B 73.8 - - 948.4 - 25.1 - - - - - VILA-U (256) [68] LLaMA-2-7B 83.9 - 56.3 1336.2 27.7 - - 48.3 - - - VILA-U (384) [68] LLaMA-2-7B 85.8 - 59 1401.8 33.5 - - 60.8 - - - Janus [67] DeepSeek-LLM-1.3B 87.0 69.4 63.7 1338.0 34.3 30.5 - - - - - Janus-Pro-1B [67] DeepSeek-LLM-1.3B 86.2 75.5 68.3 1444.0 39.8 36.3 - - - - - Janus-Pro-7B [67] DeepSeek-LLM-7B 87.4 79.2 72.1 1567.1 50.0 41.0 - - - - - -

ILLUME Vicuna-7B 88.5 75.1 72.9 1445.3 37.0 38.2 71.4 72.1 66.7 76.0 45.5 669 ILLUME+ Qwen-2.5-3B 87.6 80.8 73.3 1414.0 40.3 44.3 74.2 69.9 69.9 80.8 44.1 672

- Table 2: Quantitative results on visual understanding benchmarks. Our performance is close to and even outperforms both understanding only and unified models. The performance with top-1 and top-2 value are denoted in bold and underline respectively.

MJHQ30k GenAI-bench GenEval Method Params. Type

FID Basic Advanced Overall Single Obj Two Obj. Counting Colors Position Color Attri. Generation Only

SDv1.5 [53] 0.9B Diffusion - - - 0.43 0.97 0.38 0.35 0.76 0.04 0.06 PixArt-α [8] 0.6B Diffusion 6.14 - - 0.48 0.98 0.5 0.44 0.8 0.08 0.07 SDXL [51] 2.6B Diffusion 9.55 0.83 0.63 0.55 0.98 0.74 0.39 0.85 0.15 0.23 Emu3-Gen [65] 8B Autoregressive - - - 0.54 0.98 0.71 0.34 0.81 0.17 0.21

Unify Understanding and Generation Chameleon [59] 7B Autoregressive - - - 0.39 - - - - - LWM [40] 7B Autoregressive 17.77 0.63 0.53 0.47 0.93 0.41 0.46 0.79 0.09 0.15 Show-o [70] 1.5B Autoregressive 15.18 0.70 0.60 0.53 0.95 0.52 0.49 0.82 0.11 0.28 VILA-U(256) [68] 7B Autoregressive 12.81 0.76 0.64 − − − − − − − VILA-U(384) [68] 7B Autoregressive 7.69 0.73 0.61 − − − − − − − Janus [67] 1.3B Autoregressive 10.1 − − 0.61 0.97 0.68 0.3 0.84 0.46 0.42 Janus-Pro-1B [11] 1.3B Autoregressive - − − 0.73 0.98 0.82 0.51 0.89 0.65 0.56 Janus-Pro-7B [11] 7B Autoregressive - − − 0.80 0.99 0.89 0.59 0.90 0.79 0.66 ILLUME [63] 7B Autoregressive 7.76 0.75 0.60 0.61 0.99 0.86 0.45 0.71 0.39 0.28 ILLUME+ 3B Autoregressive 6.00 0.72 0.71 0.72 0.99 0.88 0.62 0.84 0.42 0.53

- Table 3: Quantitative results on text-to-image generation benchmarks. ILLUME+ achieves comparable results with specialist models and unified MLLMs. The performance with top-1 and top-2 value are denoted in bold and underline respectively.

#### 4.2 Compare to State-of-the-Art

Multimodal understanding. To evaluate the multimodal understanding capabilities, we conduct evaluation on two types of widely-used benchmarks: (1) General, including POPE [34], MMBench [41], SEED [31], MME-P [20], MM-Vet [76], MMMU [77] and AI2D [29]; (2) Document-oriented, including VQA-text [56], ChartQA [47], DocVQA [49], InfoVQA [48] and OCRBench [42]. As shown in our results, despite using only a 3B model, ILLUME+ achieves competitive performance compared to state-of-the-art unified models, including Janus-Pro-7B and ILLUME-7B, on general benchmarks. Notably, our 3B model demonstrates exceptional performance on document-related tasks, a challenge for most existing unified models. This highlights the effectiveness of our dual-encoder design in preserving strong understanding capabilities within a unified model. More visualizations on understanding tasks are illustrated in Fig. 6.

Multimodal image generation. To evaluate the multimodal visual generation capability, we use the MJHQ-30K [33], GenAI-bench [36] and GenEval [22] benchmarks in Table 3. For MJHQ-30K, we adopt the Fréchet Inception Distance (FID [25]) metric on 30K generated images compared to

###### Method Res. ratio # Scales Dim Size rFID↓ PSNR↑ SSIM↑

VQGAN [18] 256 16 1 256 16384 4.99 20.00 0.629 MaskGIT [6] 256 16 1 256 1024 2.28 − − LLamaGEN [57] 256 16 1 32 16384 2.19 20.79 0.675 VILA-U [68] 256 14.2 4 256 16384 1.80 - VILA-U [68] 384 14.2 16 256 16384 1.25 - -

DualViTok 256 16 2 32 131072 1.37 22.53 0.741 DualViTok 384 16 2 32 131072 0.69 23.62 0.769 DualViTok 512 16 2 32 131072 0.45 24.83 0.803

- Table 4: Comparisons with other visual tokenizers. The evaluations are on ImageNet 50k validation set under different image resolution.

Emu Edit Method Type Tasks

DINO CLIP-I CLIP-T CLIP-DIR

InstructPix2Pix [3] Diffusion Edit only 0.762 0.834 0.219 0.078 MagicBrush [79] Diffusion Edit only 0.776 0.838 0.222 0.09 OmniGen [69] Diffusion Edit only 0.804 0.836 0.233 Emu Edit [55] Diffusion Edit only 0.819 0.859 0.231 0.109

PUMA [19] AR Edit only 0.785 0.846 0.270 ILLUME AR Und, Gen, Edit 0.791 0.879 0.260 ILLUME+ AR Und, Gen, Edit 0.826 0.872 0.275 0.101

Table 5: Quantitative results on image editing benchmarks. The performance with top-1 and top-2 value are denoted in bold and underline.

Continuous Semantic Pixel Image Reconstruction Image Gen. Image Understanding Input Encoder Encoder rFID↓ PSNR↑ SSIM↑ gFID↓ POPE MMB SEED MME-P MM-Vet MMMU VQA-text

- ✗ ✓ ✗ 5.48 15.69 0.487 - 81.6 46.9 54.1 1171.9 15.1 38.0 42.0
- ✗ ✗ ✓ 2.08 21.86 0.720 28.24 66.0 30.7 41.4 820.7 13.9 36.1 40.6
- ✗ ✓ ✓ 1.83 21.68 0.714 26.70 82.1 50.4 56.0 1203.7 17.7 39.7 43.3

✓ ✓ ✓ 1.83 21.68 0.714 26.70 85.3 70.9 66.6 1491.6 34.0 42.4 56.2

- Table 6: Ablation study of visual tokenizer on image reconstruction and image generation. The rFID, PSNR and SSIM are evaluated on ImageNet 50k validation set. The gFID is evaluated on MJHQ30k. The setting of the main experiments are marked in gray.

30K high-quality images, measuring the generation quality and diversity. GenAI-bench [36] and GenEval [22] are challenging text-to-image generation benchmarks designed to reflect the consistency between text descriptions and generated images. We compare ILLUME+ with previous state-ofthe-art multimodal generation-only and unified models. With our dual vision tokenizer, ILLUME achieves a 6.00 FID score on the MJHQ30K benchmark, achieving state-of-the-art performance across both generation-only and unified models. This highlights the superior generation quality and diversity enabled by our diffusion-based approach. Additionally, ILLUME+ achieves competitive results on the GenAI-bench and GenEval benchmarks and attains the highest accuracy (0.72) in advanced categories on GenAI-bench, demonstrating its ability to understand and generate images from complex text descriptions. Figure 7 shows more results of ILLUME+ on generating flexible resolution images.

Multimodal image editing. To assess the multimodal image editing capability of our method, we evaluate it on the Emu Edit [55] benchmark and report the CLIP-I, CLIP-T, CLIP-DIR and DINO [5] scores. The CLIP-I and DINO scores measure the model’s ability to preserve elements from the source image, while the CLIP-T and CLIP-DIR score measures the consistency between the output image and the target caption. As illustrated in Table 5, our model demonstrates strong performance in image editing tasks, surpassing specialized models, particularly in the CLIP-T metric. This indicates that the unified model’s superior understanding enhances its ability to interpret editing instructions, resulting in more precise modifications. Furthermore, our dual-codebook design, which accounts for texture information, improves consistency with the original image as shown in Fig. 8.

Image reconstruction of vision tokenizer. Table 7 compares various state-of-the-art visual tokenizers on the ImageNet 50k validation set across different image resolutions using the rFID, PSNR, and SSIM. At a resolution of 256 × 256, our DualViTok achieves state-of-the-art performance, exhibiting the best performance among the compared methods. Notably, DualViTok demonstrates the capability to handle multiple resolutions within a single model. For instance, when comparing performance at a higher resolution of 384x384, DualViTok significantly outperforms VILA-U [68] at the same resolution, with a substantial improvement of 0.56 in rFID. This highlights DualViTok ’s advantage in achieving superior reconstruction quality across varying input sizes with a single, versatile model, showcasing its efficiency and flexibility compared to fixed-resolution approaches like the specific VILA-U [68] instance presented for 384x384.

#### 4.3 Ablation Studies

In our ablation studies, we train our DualViTok on the ImageNet training set for 20 epochs at an image resolution of 256 × 256. Unless specified otherwise, both the semantic codebook and the pixel codebook have a vocabulary size of 32768, respectively. For image generation, we train the

[Figure 96]

[Figure 97]

[Figure 98]

Q: Determine the vertical displacements at A of the pinconnected structure in Figure P9.31. Given: E = 200 GPa, AAB = 1000 mm2 , and AAC = AAD = 500 mm2.

- A. 4.32 mm
- B. 4.12 mm
- C. 4.22 mm A: B

Q: In which two years did J. G. Farrell win the Booker prize? A: 1970, 1973

[Figure 99]

[Figure 100]

Q: A finite automaton (FA) is an abstract machine that can be used to represent certain forms of computation. The above FA identifies which of the following regular expressions?

- A. for
- B. [a-z][a-z0-9]*
- C. ([1-9][0-9]*)|0 A: C

[Figure 101]

Q: Which state in the north-eastern region of the U.S. has greater than 6% of year-over-year home price gain New York, Vermont, New Hampshire, Maine? A: New Hampshire

Q: Calculate the steady-state voltage, $v_c(t)$, for the network shown in Figure.

- A. $\sqrt(2) cos[(1/3)t]$
- B. $\sqrt(3) cos[(2/3)t]$
- C. $\sqrt(2) cos[(2/3)t]$ A: C

[Figure 102]

[Figure 103]

Q: How many matches were played in the 2015 ICC Cricket World Cup? A: 49

[Figure 104]

Q: What would happen if there was a sudden decrease in grass?

- A. sheep would thrive
- B. you can't predict it
- C. crickets would die A: C

Q: What is the second benefit of deep reading? A: Social perception

Q: How many windows are in the room? A: 2

Figure 6: More visualizations on understanding tasks.

MLLM with DualViTok on a dataset of 10M high-fidelity images. For image understanding, we train the MLLM on LLaVA’s pretraining and SFT datasets. The performance of image reconstruction is assessed on ImageNet val (50k), and the gFID is assessed on MJHQ30k.

Dual tokenization vs. single tokenization. We compare our DualViTok with the single tokenization method, i.e., semantic autoencoder and pixel autoencoder. Specifically, the semantic autoencoder is trained on both the image and semantic reconstruction task. The pixel autoencoder is only trained on image reconstruction. As shown in Table 6, our dual tokenization, DualViTok, which fuses

[Figure 105]

[Figure 106]

[Figure 107]

768×1152

640×1920

[Figure 108]

[Figure 109]

[Figure 110]

1152×768

768×1152

[Figure 111]

[Figure 112]

[Figure 113]

768×1024

[Figure 114]

1536×768 1024×1024

1192×640 1192×640

[Figure 115]

[Figure 116]

[Figure 117]

1024×1024 1024×1024

[Figure 118]

[Figure 119]

[Figure 120]

1152×768

1536×768

[Figure 121]

[Figure 122]

768×1152

[Figure 123]

1152×768 1152×768

768×1152

1152×768

[Figure 124]

1152×768

512×2048

Figure 7: More visualizations on generation tasks.

the semantic and pixel branches, outperforms single tokenization methods in image reconstruction. Moreover, combining the semantic and pixel codebooks via the coarse-to-fine generation enhances the generation performance, i.e., a 1.54 rFID improvement. According to the image understanding abilities, the semantic encoder outperforms the pixel encoder, and our DualViTok boosts performance across all benchmarks.

Continuous input vs. discrete input. The last two rows of Table 6 compare the performance differences between continuous visual input and discrete visual input. Obviously, continuous input

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Add a Roman statue next to the vase

Add a baby bear on the rock

Delete the picture hanging on the wall

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Change image to look like Pop Art

Change the apple stem to green Change the background to snowy mountain

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Add a brown dog jumping up to catch the frisbee

Show me this in the style of comic book art

Change all the white parts of the truck to green

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Remove the glass Apply a stereo pixel art to the image Show me the image in ghibli style

Figure 8: More visualizations on editing tasks.

Encoder Decoder Up/down Codebook Channel Channel Block Size Noise rFID↓ PSNR↑ SSIM↑

Noise Type α β rFID↓ PSNR↑ SSIM↑

- - - 1.83 21.68 0.714 Random 10 100 1.76 21.85 0.722 Random 10 50 1.87 21.60 0.713 Random 10 10 1.81 21.98 0.722 Random 50 10 1.90 21.47 0.710 Random 100 10 2.06 21.65 0.707

256 256 Origin 32k/32k None 1.83 21.68 0.714 128 256 Origin 32k/32k None 1.67 21.95 0.721 128 384 Origin 32k/32k None 1.54 21.98 0.723 128 384 DC Block 32k/32k None 1.41 21.76 0.716 128 384 DC Block 32k/98k None 1.44 22.29 0.736 128 384 DC Block 32k/98k Random 1.33 22.21 0.731

- Table 7: Ablation study of DualViTok’s components. The first value of the Codebook Size is semantic codebook size and the second is the pixel codebook size. The setting of the main experiments are marked in gray.

Quantizer Codebook Dim rFID↓ PSNR↑ SSIM↑ Codebook Utilization

VQ 32 2.24 20.81 0.666 6.59% SimVQ 32 1.83 21.68 0.714 100% SimVQ 16 1.82 21.89 0.721 100% SimVQ 8 1.84 21.95 0.715 100%

- Table 8: Ablation study of visual tokenizer about quantization types and codebook dimensions. The setting of the main experiments are marked in gray.

Zero 10 100 1.88 21.57 0.717 Zero 10 50 1.73 21.97 0.725 Zero 10 10 1.90 21.97 0.722 Zero 50 10 1.75 21.88 0.722 Zero 100 10 1.79 21.75 0.716

Table 9: Ablation of noise on visual tokenizer. The noise has α% probability to perturbe the current sample, with β% of its tokens randomly replaced. The setting of the main experiments are marked in gray.

leads to better performance on all benchmarks, proving the importance of continuous input in achieving superior image understanding ability.

Ablation of the components of DualViTok. Table 7 demonstrates the components to improve our DualViTok’s baseline from 1.83 to 1.33 on rFID. First, we find the smaller encoder and larger decoder can improve the reconstruction performance from 1.83 to 1.54. Then the application of the DC block [7] brings 0.13 improvements. Scaling the pixel codebook size from 32k to 98k can further improve the performance on PSNR and SSIM while the import of random noise in the pixel codebook can improve the rFID to 1.33.

Ablation of the random noise in DualViTok. Table 9 presents an ablation study on the impact of random noise and zero noise of the visual tokenizer as well as the effects of the α and β as described in Sec. 3.1. The results indicate that both random and zero noise can achieve better reconstruction performance compared to the baseline model. However, because random noise more accurately reflects the erroneous tokens predicted by LLMs it was chosen for the main experiments.

Effect of the quantization method. We compare vanilla VQ [18] with SimVQ [83] on Table 8. Results show that SimVQ can achieves better reconstruction performance on ImageNet and maintains high codebook utilization rate. Besides, we compare different codebook dimension and find similar performance between dimension of 32, 16 and 8.

### 5 Conclusion

In this paper, we present ILLUME+, an enhanced version of the ILLUME model, which advances the integration of visual understanding, generation, and editing in a unified multimodal large language model. ILLUME+ proposes the dual visual tokenizor, DualViTok, to preserve semantic and texture in images and utilizes a diffusion decoder to enhance image generation and achieve super-resolution. By leveraging unified coarse-to-fine image representation and a progressive training procedure for dynamic visual resolution, ILLUME+ with only 3B parameters enables to process flexible resolution visual inputs and outputs, and demonstrates good performance across various benchmarks in multimodal understanding, generation, and editing tasks.

There are several promising directions for future work, including scaling to larger model sizes (7B+) for enhanced task generalization and developing more advanced image-text interleaved pretraining techniques. Further improvements can be made by constructing more complex multimodal datasets and exploring post-training strategies for unified models. These advancements will help unlock the full potential of unified multimodal models in real-world applications, supporting more sophisticated tasks and enabling broader generalization across domains.

### References

- [1] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [2] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [3] T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392– 18402, 2023.
- [4] M. Byeon, B. Park, H. Kim, S. Lee, W. Baek, and S. Kim. Coyo-700m: Image-text pair dataset. https: //github.com/kakaobrain/coyo-dataset, 2022.
- [5] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [6] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315– 11325, 2022.
- [7] J. Chen, H. Cai, J. Chen, E. Xie, S. Yang, H. Tang, M. Li, Y. Lu, and S. Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.
- [8] J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426,

- 2023.

[9] K. Chen, Y. Gou, R. Huang, Z. Liu, D. Tan, J. Xu, C. Wang, Y. Zhu, Y. Zeng, K. Yang, et al. Emova: Empowering language models to see, hear and speak with vivid emotions. arXiv preprint arXiv:2409.18042,

- 2024.

- [10] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.
- [11] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [12] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [13] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
- [14] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.
- [15] M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.
- [16] M. Ding, Z. Yang, W. Hong, W. Zheng, C. Zhou, D. Yin, J. Lin, X. Zou, Z. Shao, H. Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021.
- [17] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, D. Podell, T. Dockhorn, Z. English, K. Lacey, A. Goodwin, Y. Marek, and R. Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024.
- [18] P. Esser, R. Rombach, and B. Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [19] R. Fang, C. Duan, K. Wang, H. Li, H. Tian, X. Zeng, R. Zhao, J. Dai, H. Li, and X. Liu. Puma: Empowering unified mllm with multi-granular visual generation. arXiv preprint arXiv:2410.13861, 2024.
- [20] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun, Y. Wu, and R. Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024.
- [21] Y. Ge, S. Zhao, J. Zhu, Y. Ge, K. Yi, L. Song, C. Li, X. Ding, and Y. Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

- [22] D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [23] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [24] J. Gu, X. Meng, G. Lu, L. Hou, N. Minzhe, X. Liang, L. Yao, R. Huang, W. Zhang, X. Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Advances in Neural Information Processing Systems, 35:26418–26431, 2022.
- [25] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [26] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.
- [27] S. Huang, T. Cheng, J. K. Liu, J. Hao, L. Song, Y. Xu, J. Yang, J. H. Liu, C. Zhang, L. Chai, R. Yuan, Z. Zhang, J. Fu, Q. Liu, G. Zhang, Z. Wang, Y. Qi, Y. Xu, and W. Chu. Opencoder: The open cookbook for top-tier code large language models. 2024.
- [28] Y. Jin, K. Xu, K. Xu, L. Chen, C. Liao, J. Tan, Y. Mu, et al. Unified language-vision pretraining in llm with dynamic discrete visual tokenization. In International Conference on Learning Representations, 2024.
- [29] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [30] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022.
- [31] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [32] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, Y. Li, Z. Liu, and C. Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [33] D. Li, A. Kamko, E. Akhgari, A. Sabet, L. Xu, and S. Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.
- [34] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [35] W. Lian, B. Goodson, E. Pentland, A. Cook, C. Vong, and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https://huggingface.co/datasets/Open-Orca/ OpenOrca, 2023.
- [36] Z. Lin, D. Pathak, B. Li, J. Li, X. Xia, G. Neubig, P. Zhang, and D. Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2025.
- [37] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.
- [38] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024.
- [39] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [40] H. Liu, W. Yan, M. Zaharia, and P. Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024.
- [41] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025.
- [42] Y. Liu, Z. Li, B. Yang, C. Li, X. Yin, C.-l. Liu, L. Jin, and X. Bai. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023.
- [43] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations.
- [44] D. Lu, X. Tan, R. Xu, T. Yao, C. Qu, W. Chu, Y. Xu, and Y. Qi. Scp-116k: A high-quality problem-solution dataset and a generalized pipeline for automated extraction in the higher education science domain. arXiv preprint arXiv:2501.15587, 2025.

- [45] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, Y. Sun, et al. Deepseek-vl: Towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.
- [46] J. Lu, C. Clark, S. Lee, Z. Zhang, S. Khosla, R. Marten, D. Hoiem, and A. Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024.
- [47] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [48] M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022.
- [49] M. Mathew, D. Karatzas, and C. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [50] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [51] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [52] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.
- [53] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [54] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.
- [55] S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.
- [56] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.
- [57] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [58] Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Zhang, Y. Wang, H. Gao, J. Liu, T. Huang, and X. Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.
- [59] C. Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [60] Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023.
- [61] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.
- [62] P. Tong, E. Brown, P. Wu, S. Woo, A. J. V. IYER, S. C. Akula, S. Yang, J. Yang, M. Middepogu, Z. Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.
- [63] C. Wang, G. Lu, J. Yang, R. Huang, J. Han, L. Hou, W. Zhang, and H. Xu. Illume: Illuminating your llms to see, draw, and self-enhance. arXiv preprint arXiv:2412.06673, 2024.
- [64] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [65] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [66] C. Wei, Z. Xiong, W. Ren, X. Du, G. Zhang, and W. Chen. Omniedit: Building image editing generalist models through specialist supervision. In The Thirteenth International Conference on Learning Representations, 2024.
- [67] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.

- [68] Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.
- [69] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, S. Wang, T. Huang, and Z. Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.
- [70] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [71] Z. Xu, F. Jiang, L. Niu, Y. Deng, R. Poovendran, Y. Choi, and B. Y. Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. arXiv preprint arXiv:2406.08464, 2024.
- [72] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [73] H. Ye, D.-A. Huang, Y. Lu, Z. Yu, W. Ping, A. Tao, J. Kautz, S. Han, D. Xu, P. Molchanov, et al. X-vila: Cross-modality alignment for large language model. arXiv preprint arXiv:2405.19335, 2024.
- [74] J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [75] Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024.
- [76] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [77] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [78] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [79] K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36, 2024.
- [80] H. Zhao, X. S. Ma, L. Chen, S. Si, R. Wu, K. An, P. Yu, M. Zhang, Q. Li, and B. Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.
- [81] C. Zheng, T.-L. Vuong, J. Cai, and D. Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022.
- [82] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [83] Y. Zhu, B. Li, Y. Xin, and L. Xu. Addressing representation collapse in vector quantized models with one linear layer. arXiv preprint arXiv:2411.02038, 2024.
- [84] Y. Zhu, Y. Zhou, C. Wang, Y. Cao, J. Han, L. Hou, and H. Xu. Unit: Unifying image and text recognition in one vision encoder. arXiv preprint arXiv:2409.04095, 2024.

