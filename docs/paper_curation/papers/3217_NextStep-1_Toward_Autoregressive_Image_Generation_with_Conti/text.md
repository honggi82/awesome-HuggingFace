# arXiv:2508.10711v2[cs.CV]18Aug2025

[Figure 1]

## NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale

NextStep-Team, StepFun

Homepage: https://stepfun.ai/research/en/nextstep1 Github: https://github.com/stepfun-ai/NextStep-1 Huggingface: NextStep-1 Collections

### Abstract

Prevailing autoregressive (AR) models for text-to-image generation either rely on heavy, computationally-intensive diffusion models to process continuous image tokens, or employ vector quantization (VQ) to obtain discrete tokens with quantization loss. In this paper, we push the autoregressive paradigm forward with NextStep-1, a 14B autoregressive model paired with a 157M flow matching head, training on discrete text tokens and continuous image tokens with next-token prediction objectives. NextStep-1 achieves state-of-the-art performance for autoregressive models in text-to-image generation tasks, exhibiting strong capabilities in highfidelity image synthesis. Furthermore, our method shows strong performance in image editing, highlighting the power and versatility of our unified approach. To facilitate open research, we will release our code and models to the community.

#### 1. Introduction

The remarkable success of autoregressive models in large language models (Brown et al., 2020; OpenAI, 2025a; Radford et al., 2018, 2019) has motivated their extension to text-to-image generation. By unifying multimodal inputs into a single sequence, autoregressive image generation models (Chen et al., 2025b; Fan et al., 2024; Sun et al., 2023, 2024b,c; Wang et al., 2024b; Yu et al., 2022) offer a scalable and flexible approach to text-to-image generation that naturally accommodates diverse conditioning signals.

However, most existing autoregressive approaches for text-to-image generation (Chen et al., 2025b; Dong et al., 2024; Sun et al., 2024a,b; Tong et al., 2024; Wang et al., 2024b) either rely on heavy diffusion models or adopt vector quantization (VQ) (Eslami et al., 2021; Yu et al., 2023; Zheng et al., 2022) to tokenize images into discrete visual tokens, which encounter limitations including exposure bias (Han et al., 2025) and suboptimal image tokenization (Li et al., 2024c). While recent efforts with continuous latent representations (Fan et al., 2024; Li et al., 2024c; Sun et al., 2024c; Tschannen et al., 2024, 2025) have shown promise, a significant performance gap persists between autoregressive models and state-of-the-art diffusion methods (Esser et al., 2024; Labs, 2024; Podell et al., 2024), particularly in image quality and consistency.

In this paper, we introduce NextStep-1, a simple yet effective autoregressive model built on the next-token prediction paradigm that achieves state-of-the-art performance in text-toimage generation tasks. Comprehensive evaluations confirm its competitive performance across a suite of challenging benchmarks. Specifically, NextStep-1 demonstrates exceptional compositional and linguistic understanding, achieving 0.54 on WISE (Niu et al., 2025), 0.67 on the advanced prompts of GenAI-Bench (Lin et al., 2024), 85.28 on DPG-Bench (Hu et al., 2024), and 0.417 on the English prompts of OneIG-Bench (Chang et al., 2025). These results demonstrate its capabilities across diverse scenarios, from short and long prompts to tasks

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

###### ImageGenerationImageEditing

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

Add Object Material Change Background Change Motion Change Tone Transfer Style Change

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

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

PortraitBeautify

ComplexEdit

ColorChange

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Playingwithabutterfly

Jumpingtotheleftside

###### Free-formManipulation

Sleeping

Jumping

Atnight

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Holdingacoffeecup

Atbusstation

Atclassroom

Atbooksea

Atpark

###### Figure 1 | Overview of NextStep-1 in high-fidelity image generation, diverse image editing, and complex free-form manipulation.

[Figure 53]

[Figure 54]

Mean Square Error

……

|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]|
|---|

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

|[Figure 65]<br><br>dog|
|---|

|[Figure 66]<br><br>on|
|---|

|[Figure 67]<br><br>rock|
|---|

|[Figure 68]<br><br>[eoi]|
|---|

[Figure 69]

[Figure 70]

[Figure 71]

Next Token Prediction

𝑡

Velocity Prediction

[Figure 72]

[Figure 73]

LM Head Flow Matching Head

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

…… ……

Target Patch

[Figure 86]

[Figure 87]

Flow Matching Head

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Target Flow

[Figure 92]

Causal Transformer

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

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

…… ……

[Figure 109]

1 − 𝑡

Noise Sample

Hidden State

[Figure 110]

Text Tokenizer Image Tokenizer

[Figure 111]

Patch-Wise Flow Matching

[Figure 112]

……

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

|[Figure 122]<br><br>[eoi]|
|---|

|[Figure 123]<br><br>[bos]|
|---|

|[Figure 124]<br><br>dog|
|---|

|[Figure 125]<br><br>rock|
|---|

|[Figure 126]<br><br>on|
|---|

|[Figure 127]<br><br>[boi]|
|---|

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

- Figure 2 | Overview of NextStep-1 Framework. NextStep-1 employs a causal transformer to process tokenized text and image tokens. During training, Flow Matching Head predicts the continuous flow

from a noise sample to the next target image patch, conditioned on the output hidden state. At inference, this allows for generating images by iteratively guiding noise to create the next patch.

requiring world knowledge. Beyond generation, the versatility of NextStep-1 is validated by its strong performance on instruction-based image editing, NextStep-1-Edit, achieving competitive scores of 6.58 for English prompts on GEdit-Bench (Liu et al., 2025) and 3.71 on ImgEditBench (Ye et al., 2025). We showcase the qualitatve performance in Fig. 1.

NextStep-1 is a 14-billion-parameter autoregressive model composed of a Transformer backbone, a standard language modeling head for discrete text tokens, a lightweight flow matching head for continuous image tokens, and an image tokenizer. The flow matching head is a 157million-parameter, MLP-based model trained with a flow matching objective, following the approach of (Li et al., 2024c). In autoregressive modeling, high-dimensional latent spaces are critical for achieving high image quality but often induce training instability and divergence. Our image tokenizer addresses this trade-off by enhancing the robustness of continuous image tokens and promoting a well-dispersed, normalized latent space, thereby ensuring stable convergence even at higher dimensionalities (e.g., 16 channels). Empirical results confirm that this design is essential for stable and effective training with 16-channel latents.

#### 2. Framework

###### 2.1. Unified Multi-model Generation with Continuous Visual Tokens

NextStep-1 extends the well-established autoregressive language modeling paradigm to image generation through a simple and intuitive architecture, as illustrated in Fig. 2. To unify multimodal inputs into a single sequence, the images will be tokenized to continuous image tokens by the image tokenizer and combined with discrete text tokens. Supposing 𝑥 = {𝑥0, 𝑥1,..., 𝑥𝑛} is the multimodel token sequence, where 𝑥𝑖 is either a discrete text token or a continuous visual token, the autoregressive objective under the unified sequence is formalized as:

𝑛

𝑝(𝑥𝑖 | 𝑥<𝑖). (1)

𝑝(𝑥) =

𝑖=1

The unified multi-modal generation task proceeds by sampling the next token 𝑥𝑖 from the conditional distribution 𝑝(𝑥𝑖 | 𝑥<𝑖) modeled by a network. Discrete text tokens are sampled via a language modeling head, while continuous image tokens are sampled by a flow-matching head.

Our training objective consists of two distinct losses: a standard cross-entropy loss for discrete text tokens, and a flow matching loss (Lipman et al., 2023b) for continuous image tokens. Specifically, the flow matching loss is the mean squared error between the predicted and target velocity vectors that map a noised patch to its corresponding clean patch. The model is trained end-to-end by optimizing a weighted sum of these two losses:

Ltotal = 𝜆textLtext + 𝜆visualLvisual (2)

where Ltext and Lvisual denote the loss for text and image tokens respectively, which are balanced by the hyperparameters 𝜆text and 𝜆visual.

###### 2.2. Model Architecture

Image Tokenizer. Our image tokenizer is fine-tuned from flux VAE (Labs, 2024) with only reconstruction and perceptual losses. The tokenizer first encodes an image into 16-channel latents 𝑧, applying an 8× spatial downsampling factor. To stabilize and normalize the latent space, we apply channel-wise normalization, standardizing each channel to zero mean and unit variance. Furthermore, to enhance the robustness of the image tokenizer and encourage a more uniform latent distribution, we introduce a stochastic perturbation to the normalized latents. This technique is adapted from 𝜎-VAE (Sun et al., 2024c), where it was employed to prevent variance collapse.

𝑧˜ = Normlization(𝑧) + 𝛼 · 𝜀, where 𝛼 ∼ U[0, 𝛾] and 𝜀 ∼ N(0, 𝐼) (3)

where 𝜀 is standard Gaussian noise, and its magnitude is scaled by a random factor 𝛼 sampled uniformly from [0, 𝛾]. The 𝛾 is a hyperparameter controlling the maximum noise intensity.

The latents from the image tokenizer are pixel-shuffled into a more compact sequence. This is achieved by applying a space-to-depth transformation with a 2×2 kernel, which flattens 2×2 spatial latents into the channel dimension. For example, this converts the latents of a 256×256 image into a 16×16 grid of 64-channel tokens. This grid is then flattened into a 1D sequence of 256 tokens to serve as input for the following Causal Transformer.

Causal Transformer. We initialize our model from the decoder-only Qwen2.5-14B (Yang et al.,

- 2024), leveraging its strong language understanding and reasoning capabilities for text-to-image generation. We organize the multimodal input sequence in the following format:

{text} <image_area>h*w <boi> {image} <eoi>...

where {text} denotes discrete text tokens, and {image} represents continuous image tokens. <boi> and <eoi> are special tokens marking the beginning-of-image and end-of-image. <image_area>h*w represents the metadata about the spatial dimensions of the 2D image tokens.

Then the output hidden states from LLM are passed to two lightweight heads for modalityspecific loss:

- • Language Modeling Head. We compute Cross-Entropy loss for hidden states of texts.
- • Patch-wise Flow Matching Head. Following (Li et al., 2024c), we use each patch-wise image hidden states as condition, denoise target patch at timesteps t, and compute the patch-wise flow-matching loss (Lipman et al., 2023a) with a 157M, 12-layer, and 1536 hidden-dimensions MLP.

For positional information, we use the standard 1D RoPE (Su et al., 2024). Despite the availability of more complex 2D or multimodal RoPE alternatives (Bai et al., 2025; Wang et al.,

- 2024a), we found that the simple 1D formulation remains highly effective for mixed text-image sequences, and thus retain it for simplicity and efficiency.

3. Data

To comprehensively equip our model with broad and versatile capabilities, we construct a diverse training corpus composed of four primary data categories: a text-only corpus, imagetext pair data, image-to-image data, and interleaved data. Each category is curated to serve a distinct role in fostering different aspects of the model’s generative abilities.

3.1. Text-only Corpus

To preserve the extensive language capabilities inherent in the large language model (LLM), we incorporate 400B text-only tokens sampled from Step-3 (Wang et al., 2025a) during training.

- 3.2. Image-Text Pair Data

Data consisting of image-text pairs forms the foundation of the model’s text-to-image generation capabilities. We developed a comprehensive pipeline to curate a high-quality, large-scale dataset from a diverse set of initial sources.

- 1. Data Sourcing: We collected a large-scale dataset from diverse sources, including web data, multi-task VQA data, and text-rich documents.
- 2. Quality-Based Filtering: We then applied a rigorous filtering process, evaluating each image on aesthetic quality, watermark presence, clarity, OCR detection, and text-image semantic alignment.
- 3. Re-captioning: After deduplicating the filtered images, we used the Step-1o-turbo 1 to generate rich and detailed captions for each image in both English and Chinese.

This multi-stage pipeline yields a final dataset of 550M high-quality image-text pairs, providing a foundation for training a model with both strong aesthetic sense and broad world knowledge.

- 3.3. Instruction-Guided Image-to-Image Data

To enable a wide range of practical applications, we curated a high-quality dataset for instructionguided image-to-image tasks, such as visual perception (Kirillov et al., 2023), controllable image generation (Zhang et al., 2023b), image restoration (Labs, 2025), general image editing (Peng et al., 2024), and more.

For visual perception and controllable image generation tasks, we synthesized 1M samples by applying the annotator of ControlNet (Zhang et al., 2023b) to a part of our high-quality image-text pair data. For image restoration and general image editing, we collected 3.5M samples, comprising data from GPT-Image-Edit (Wang et al., 2025d), Step1X-Edit (Liu et al.,

- 2025), and a proprietary in-house dataset. Following Step1X-Edit (Liu et al., 2025), all editing data were subjected to a rigorous VLM-based filtering pipeline that assessed both image-pair quality, rationality, consistency, and instruction alignment, resulting in about 1M high-quality instruction-guided image-to-image data for training.

1https://platform.stepfun.com/docs/llm/vision

[Figure 132]

###### Face detection & character binding

Step I: Step II: Step III:

|[Figure 133]|
|---|

|[Figure 134]|
|---|

[Figure 135]

Web data collection

|[Figure 136]|
|---|

|[Figure 137]|
|---|

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Char id: Frame id Bbox array

uniform sampling

ArcFace

[Figure 144]

Coarse filtering

Get face embeddings within each scene

Matching faces based on cosine similarity

Saving meta infos for each character

[Figure 145]

Frame extraction & MLLM captioning

[Figure 146]

Face detection & character binding

[Figure 147]

Step I: Step II: Step III:

[Figure 148]

Step 1o

Char id: Frame id Bbox array

[Figure 149]

Step 1o

|[Figure 150]|
|---|

[Figure 151]

“The camera shifts to the outdoors, where the man is talking to a woman”

| | | | |
|---|---|---|---|
| | | | |

[Figure 152]

Checklist:

Frame extraction & MLLM captioning

- • Object Consistency
- • Attribute Consistency
- • Relation Consistency
- • ……

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

Frame extraction using meta infos

Figure 3 | Data processing of character-centric data.

###### 3.4. Interleaved data

Interleaved data seamlessly integrates text and images, offering rich and nuanced sequential associations between modalities. Specifically, our knowledge-rich interleaved dataset is primarily composed of four distinct categories: general video-interleaved data, tutorials, character-centric scenes, and multi-view data.

To endow our model with extensive world knowledge, we first constructed a large-scale, 80Msample video-interleaved dataset. This was achieved through a meticulous curation pipeline, inspired by Step-Video (Ma et al., 2025a), which encompasses frame extraction, deduplication, and captioning. Furthermore, following the methodology of mmtextbook (Zhang et al., 2025), we collected and processed tutorial videos by leveraging ASR and OCR tools. This component specifically targets text-rich real-world scenes, enhancing the model’s textual understanding and generation in context. A key contribution, detailed in Fig. 3, is our character-centric dataset, NextStep-Video-Interleave-5M. For this dataset, we extracted video frames centered around specific characters and generated rich, storytelling-style captions akin to (Oliveira and de Matos,

- 2025), thereby significantly improving the model’s capacity for multi-turn interaction. Finally, to bolster geometric reasoning, we curated multiview data from two open-source datasets, MV-ImageNet-v2 (Han et al., 2024) and Objaverse-XL (Deitke et al., 2023), which enhances the model’s ability to maintain multiview consistency.

#### 4. Training Recipe

###### 4.1. Training Image Tokenizer

Our image tokenizer is initialized from the Flux.1-dev VAE (Labs, 2024), selected for its strong reconstruction performance. We fine-tune this model on the image-text dataset detailed in Sec. 3.2 to adapt it to our specific data distribution. For optimization, we employ the AdamW optimizer (Loshchilov and Hutter, 2019) with (𝛽1 = 0.9, 𝛽2 = 0.95,𝜀 = 1×10−8) for its convergence stability. The model is trained for 50K steps with a total batch size of 512, using a constant learning rate of 1 × 10−5 preceded by a linear warm-up of 1000 steps.

Table 1 | Training recipe of NextStep-1.

Pre-Training Post-Training

Stage1 Stage2 Annealing SFT DPO Hyperparameters

Learning Rate (Min, Max) 1 × 10−4 1 × 10−5 (0, 1 × 10−5) (0, 1 × 10−5) 2 × 10−6 LR Scheduler Constant Constant Cosine Cosine Constant Weight Decay 0.1 0.1 0.1 0.1 0.1 Loss Weight (CE : MSE) (0.01 : 1) (0.01 : 1) (0.01 : 1) (0.01 : 1) Training Steps 200K 100K 20K 10K 300 Warm-up Steps 5K 5K 0 500 200 Sequence Length per Rank 16K 16K 16K 8K Image Area (Min, Max) 256×256 (256×256, 512×512) (256×256, 512×512) (256×256, 512×512) (256×256, 512×512) Image Tokens (Min, Max) 256 (256, 1024) (256, 1024) (256, 1024) (256, 1024) Training Tokens 1.23T 0.61T 40B 5B -

###### Data Ratio

Text-only Corpus 0.2 0.2 0.2 0 Image-Text Pair Data 0.6 0.6 0.6 0.9 Image-to-Image Data 0.0 0.0 0.1 0.1 Interleaved Data 0.2 0.2 0.1 0 -

###### 4.2. Pre-Training

The specific hyperparameters and data ratios for our pre-training are detailed in Tab. 1. Specifically, the pre-training follows a three-stage curriculum designed to progressively refine the model’s capabilities. Throughout these stages, all model parameters are trained end-to-end except for the pre-trained image tokenizer.

- Stage1. In this initial stage, the model learns a foundational understanding of image structure and composition. For computational efficiency, all images are resized and randomly cropped to a fixed 256×256 resolution. The training curriculum is composed of a diverse data mixture: 20% text-only corpora, 60% image-text pairs, and 20% interleaved data. This stage consumed approximately 1.23T tokens.
- Stage2. We implement a dynamic resolution strategy to train the model on a range of higher resolutions, targeting 256×256 and 512×512 base areas. This strategy utilizes different aspect ratio buckets for computational efficiency. In this stage, we enrich the data mixture with more text-rich and video-interleaved data, leveraging the model’s enhanced capacity to process fine details at these resolutions.

Annealing. In the final stage of pre-training, we perform an annealing phase to sharpen the model’s capabilities on a highly curated dataset. This is achieved by training the model for one epoch on a high-quality subset of 20M samples, which were selected from Sec. 3.2 by applying stricter filtering thresholds for aesthetic score, image clarity, semantic similarity, watermark, and so on. This annealing step significantly improves the model’s final output, enhancing overall image structure, composition, texture, and aesthetic appeal.

###### 4.3. Post-Training

Following pre-training on a broad corpus to establish a generalist model, post-training serves to align the model’s output with human preferences and downstream tasks. We achieve this alignment via a two-stage process: Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO) (Rafailov et al., 2023). The hyperparameters for each stage are in Tab. 1.

Supervised Fine-Tuning (SFT). The SFT stage enhances the model’s instruction-following capabilities and aligns its outputs with human preferences. The SFT dataset, comprising a total of 5M samples, is organized into three components: 1) a corpus of human-selected image-text pairs with high semantic consistency and visual appeal, augmented by images from other generative models to improve the model’s handling of complex and imaginative prompts through distillation; 2) Chain-of-Thought (CoT) data (Deng et al., 2025; Wei et al., 2022), improving text-to-image generation by incorporating a language-based reasoning step before the final image is created; 3) high-quality instruction-guided image-to-image data from Sec. 3.3 to strengthen the model’s image editing capabilities.

Direct Policy Optimization (DPO). To align our model with human preferences, we employ Direct Policy Optimization (DPO) (Rafailov et al., 2024), a method inspired by Diffusion-DPO (Wallace et al., 2024). To this end, we construct two distinct types of preference datasets from a curated set of approximately 20,000 diverse prompts.

- 1. Standard DPO Dataset: For each prompt 𝑐, we directly use the SFT model to generate 16 candidate images. These images is then scored by ImageReward (Xu et al., 2023) to form a preference pair (𝑦𝑤, 𝑦𝑙), where the winning image 𝑦𝑤 is randomly sampled from the top 4 candidates, while the losing image 𝑦𝑙 is randomly sampled from the remaining 12.
- 2. Self-CoT DPO Dataset: To enhance the model’s reasoning capabilities, we introduce an explicit reasoning step. For each prompt 𝑐, we first prompt our model to generate a detailed textual CoT, which is then extended to the original prompt. Using this CoT-enhanced prompt, we follow the identical pipeline as above to form a preference pair (𝑦𝑤, 𝑦𝑙).

#### 5. Model Performance

###### 5.1. Performance of Text-to-Image Generation

We comprehensively evaluate the text-to-image (T2I) generation performance of NextStep-1 on several representative benchmarks, each targeting different aspects of image generation, including visual-textual alignment and world knowledge.

Image–Text Alignment. As shown in Table 2, we assess NextStep-1’s prompt-following ability across three key benchmarks. On GenEval (Ghosh et al., 2023), NextStep-1 scores 0.63 (0.73 with Self-CoT), demonstrating robust counting, grounding, and spatial alignment. Its strong compositional abilities are further validated on GenAI-Bench (Li et al., 2024a), where it achieves 0.88 on basic prompts and 0.67 on advanced prompts (0.9 and 0.74 with Self-CoT). These results demonstrate NextStep-1 as a great autoregressive image generation model, with performance competitive with some diffusion models such as Stable Diffusion 3.5 Large (Stability-AI, 2024) and BAGEL (Deng et al., 2025). Finally, when evaluated on DPG-Bench (Hu et al., 2024) for longcontext, multi-object scenes, NextStep-1 achieves 85.28, confirming its reliable compositional fidelity under complex prompts.

To perform a fine-grained analysis, we evaluated our model on OneIG-Bench (Chang et al., 2025) with English prompts. This benchmark assesses performance across areas such as alignment, text rendering, reasoning and stylistic control. As shown in Tab. 3, NextStep-1 achieves an overall score of 0.417. This result significantly outperforms its autoregressive peers, such as Emu3 (Wang et al., 2024b) (0.311) and Janus-Pro (Chen et al., 2025b) (0.267).

World Knowledge. To evaluate NextStep-1’s ability to integrate world knowledge into image generation, we use the WISE benchmark (Niu et al., 2025), which emphasizes factual grounding

Table 2 | Comparison of image-text alignment on GenEval (Ghosh et al., 2023), GenAI-Bench (Lin et al., 2024), and DPG-Bench (Hu et al., 2024). * result is with rewriting. † result is with Self-CoT.

GenAI-Bench↑ DPG-Bench↑ Basic Advanced

Method GenEval↑

Proprietary

DALL-E 3 (Betker et al., 2023) 0.67 0.90 0.70 83.50 Seedream 3.0 (Gao et al., 2025) 0.84 - - 88.27 GPT4o (OpenAI, 2025b) 0.84 - - 85.15

###### Diffusion

Stable Diffusion 1.5 (Rombach et al., 2022) 0.43 - - Stable Diffusion XL (Podell et al., 2024) 0.55 0.83 0.63 74.65 Stable Diffusion 3 Medium (Esser et al., 2024) 0.74 0.88 0.65 84.08 Stable Diffusion 3.5 Large (Esser et al., 2024) 0.71 0.88 0.66 83.38 PixArt-Alpha (Chen et al., 2024b) 0.48 - - 71.11 Flux.1-dev (Labs, 2024) 0.66 0.86 0.65 83.79 Transfusion (Zhou et al., 2025) 0.63 - - CogView4 (Z.ai, 2025) 0.73 - - 85.13 Lumina-Image 2.0 (Qin et al., 2025) 0.73 - - 87.20 HiDream-I1-Full (Cai et al., 2025) 0.83 0.91 0.66 85.89 Mogao (Liao et al., 2025) 0.89 - 0.68 84.33 BAGEL (Deng et al., 2025) 0.82/0.88† 0.89/0.86† 0.69/0.75† 85.07 Show-o2-7B (Xie et al., 2025b) 0.76 - - 86.14 OmniGen2 (Wu et al., 2025b) 0.80/0.86* - - 83.57 Qwen-Image (Wu et al., 2025a) 0.87 - - 88.32

###### AutoRegressive

SEED-X (Ge et al., 2024) 0.49 0.86 0.70 Show-o (Xie et al., 2024) 0.53 0.70 0.60 VILA-U (Wu et al., 2024) - 0.76 0.64 Emu3 (Wang et al., 2024b) 0.54/0.65* 0.78 0.60 80.60 SimpleAR (Wang et al., 2025c) 0.63 - - 81.97 Fluid (Fan et al., 2024) 0.69 - - Infinity (Han et al., 2025) 0.79 - - 86.60 Janus-Pro-7B (Chen et al., 2025b) 0.80 0.86 0.66 84.19 Token-Shuffle (Ma et al., 2025b) 0.62 0.78 0.67 NextStep-1 0.63/0.73† 0.88/0.90† 0.67/0.74† 85.28

and semantic understanding. As shown in Table 4, NextStep-1 achieves the best performance among autoregressive models with an overall score of 0.54 (0.67 with Self-CoT), also exceeding most diffusion models. Notably, under the prompt rewrite protocol, its score increases to 0.79 (0.83 with Self-CoT). Collectively, these results demonstrate NextStep-1’s robust knowledgeaware semantic alignment and cross-domain reasoning capabilities.

###### 5.2. Performance of Image Editing

Quantitative Results on Editing Benchmarks. We developed NextStep-1-Edit by finetuning NextStep-1 on 1M high-quality edit-only data in Sec. 3.3, demonstrates competitive performance against advanced diffusion-based models. As shown in Tab. 5, NextStep-1-Edit achieves scores of 6.58 on GEdit-Bench-EN (Liu et al., 2025) and 3.71 on ImgEdit-Bench (Ye et al., 2025), indicating its strong practical editing capabilities.

Table 3 | Comparison on OneIG-Bench (Chang et al., 2025) in English prompts.

Method Alignment Text Reasoning Style Diversity Overall↑ Proprietary

- Imagen3 (Baldridge et al., 2024) 0.843 0.343 0.313 0.359 0.188 0.409 Recraft V3 (team, 2024) 0.810 0.795 0.323 0.378 0.205 0.502 Kolors 2.0 (team, 2025) 0.820 0.427 0.262 0.360 0.300 0.434 Seedream 3.0 (Gao et al., 2025) 0.818 0.865 0.275 0.413 0.277 0.530

- Imagen4 (deepmind Imagen4 team, 2025) 0.857 0.805 0.338 0.377 0.199 0.515 GPT-4o (OpenAI, 2025b) 0.851 0.857 0.345 0.462 0.151 0.533 Diffusion

Stable Diffusion 1.5 (Rombach et al., 2022) 0.565 0.010 0.207 0.383 0.429 0.319 Stable Diffusion XL (Podell et al., 2024) 0.688 0.029 0.237 0.332 0.296 0.316 Stable Diffusion 3.5 Large (Stability-AI, 2024) 0.809 0.629 0.294 0.353 0.225 0.462 Flux.1-dev (Labs, 2024) 0.786 0.523 0.253 0.368 0.238 0.434 CogView4 (Z.ai, 2025) 0.786 0.641 0.246 0.353 0.205 0.446 SANA-1.5 1.6B (PAG) (Xie et al., 2025a) 0.762 0.054 0.209 0.387 0.222 0.327 SANA-1.5 4.8B (PAG) (Xie et al., 2025a) 0.765 0.069 0.217 0.401 0.216 0.334 Lumina-Image 2.0 (Qin et al., 2025) 0.819 0.106 0.270 0.354 0.216 0.353 HiDream-I1-Full (Cai et al., 2025) 0.829 0.707 0.317 0.347 0.186 0.477 BLIP3-o (Chen et al., 2025a) 0.711 0.013 0.223 0.361 0.229 0.307 BAGEL (Deng et al., 2025) 0.769 0.244 0.173 0.367 0.251 0.361 Show-o2-1.5B (Xie et al., 2025b) 0.798 0.002 0.219 0.317 0.186 0.304 Show-o2-7B (Xie et al., 2025b) 0.817 0.002 0.226 0.317 0.177 0.308 OmniGen2 (Wu et al., 2025b) 0.804 0.680 0.271 0.377 0.242 0.475 Qwen-Image (Wu et al., 2025a) 0.882 0.891 0.306 0.418 0.197 0.539

###### AutoRegressive

Emu3 (Wang et al., 2024b) 0.737 0.010 0.193 0.361 0.251 0.311 Janus-Pro (Chen et al., 2025b) 0.553 0.001 0.139 0.276 0.365 0.267 NextStep-1 0.826 0.507 0.224 0.332 0.199 0.417

Table 4 | Comparison of world knowledge reasoning on WISE (Niu et al., 2025). † result is with Self-CoT.

Model Cultural Time Space Biology Physics Chemistry Overall↑ Overall (Rewrite)↑ Proprietary GPT-4o (OpenAI, 2025b) 0.81 0.71 0.89 0.83 0.79 0.74 0.80 Diffusion

Stable Diffusion 1.5 (Rombach et al., 2022) 0.34 0.35 0.32 0.28 0.29 0.21 0.32 0.50 Stable Diffusion XL (Podell et al., 2024) 0.43 0.48 0.47 0.44 0.45 0.27 0.43 0.65 Stable Diffusion 3.5 Large (Stability-AI, 2024) 0.44 0.50 0.58 0.44 0.52 0.31 0.46 0.72 PixArt-Alpha (Chen et al., 2024b) 0.45 0.50 0.48 0.49 0.56 0.34 0.47 0.63 Playground v2.5 (Li et al., 2024b) 0.49 0.58 0.55 0.43 0.48 0.33 0.49 0.71 Flux.1-dev (Labs, 2024) 0.48 0.58 0.62 0.42 0.51 0.35 0.50 0.73 MetaQuery-XL (Pan et al., 2025) 0.56 0.55 0.62 0.49 0.63 0.41 0.55 BAGEL (Deng et al., 2025) 0.44/0.76† 0.55/0.69† 0.68/0.75† 0.44/0.65† 0.60/0.75† 0.39/0.58† 0.52/0.70† 0.71/0.77† Qwen-Image (Wu et al., 2025a) 0.62 0.63 0.77 0.57 0.75 0.40 0.62 -

###### AutoRegressive

Show-o-512 (Xie et al., 2024) 0.28 0.40 0.48 0.30 0.46 0.30 0.35 0.64 VILA-U (Wu et al., 2024) 0.26 0.33 0.37 0.35 0.39 0.23 0.31 Emu3 (Wang et al., 2024b) 0.34 0.45 0.48 0.41 0.45 0.27 0.39 0.63 Janus-Pro-7B (Chen et al., 2025b) 0.30 0.37 0.49 0.36 0.42 0.26 0.35 0.71 NextStep-1 0.51/0.70† 0.54/0.65† 0.61/0.69† 0.52/0.63† 0.63/0.73† 0.48/0.52† 0.54/0.67† 0.79/0.83†

#### 6. Discussions

###### 6.1. What Governs Image Generation: the AR Transformer or the FM Head?

- A key architectural distinction of our framework is its direct, autoregressive modeling of continuous image tokens using a flow matching objective. Prevailing autoregressive models

Table 5 | Comparison of image editing performance on GEdit-Bench (Full Set) (Liu et al., 2025) and ImgEdit-Bench (Ye et al., 2025). G_SC, G_PQ, and G_O refer to the metrics evaluated by GPT-4.1 (OpenAI, 2025a). Performance is evaluated based on the NextStep-1-Edit with 1:1 aspect ratio.

GEdit-Bench-EN (Full Set)↑ GEdit-Bench-CN (Full Set)↑ ImgEdit-Bench↑ G_SC G_PQ G_O G_SC G_PQ G_O

Model

Proprietary

Gemini 2.0 (Gemini2, 2025) 6.87 7.44 6.51 5.26 7.60 5.14 Doubao (Shi et al., 2024) 7.22 7.89 6.98 7.17 7.79 6.84 GPT-4o (OpenAI, 2025b) 7.74 8.13 7.49 7.52 8.02 7.30 4.20 Flux.1-Kontext-pro (Labs et al., 2025) 7.02 7.60 6.56 1.11 7.36 1.23 -

###### Open-source

Instruct-Pix2Pix (Brooks et al., 2023) 3.30 6.19 3.22 - - - 1.88 MagicBrush (Zhang et al., 2023a) 4.52 6.37 4.19 - - - 1.83 AnyEdit (Yu et al., 2024a) 3.05 5.88 2.85 - - - 2.45 OmniGen (Xiao et al., 2024) 5.88 5.87 5.01 - - - 2.96 OmniGen2 (Wu et al., 2025b) 7.16 6.77 6.41 - - - 3.44

- Step1X-Edit v1.0 (Liu et al., 2025) 7.13 7.00 6.44 7.30 7.14 6.66 3.06
- Step1X-Edit v1.1 (Liu et al., 2025) 7.66 7.35 6.97 7.65 7.40 6.98 BAGEL (Deng et al., 2025) 7.36 6.83 6.52 7.34 6.85 6.50 3.42 Flux.1-Kontext-dev (Labs et al., 2025) - - 6.26 - - - 3.71 GPT-Image-Edit (Wang et al., 2025d) - - 7.24 - - - 3.80 NextStep-1 7.15 7.01 6.58 6.88 7.02 6.40 3.71

for image generation (Chen et al., 2025a; Dong et al., 2024; Sun et al., 2023, 2024b; Zhou et al., 2025) typically rely on heavy diffusion models for an entire image: an autoregressive model first produces a semantic embedding, which is then used to condition a diffusion model that generates an entire image in a single denoising process. In contrast, our model autoregressively generates the image patch-by-patch (Chen et al., 2024a; Fan et al., 2025; Kou et al., 2024), modeling the distribution of each patch with a significantly more lightweight flow matching model. We argue that this establishes our framework under the pure autoregressive paradigm with next-token prediction (NTP) modeling, rather than a diffusion model merely orchestrated by a Transformer.

A key finding from our experiments is the model’s surprising insensitivity to the size of its flow-matching head. We ablated this on three heads with different sizes (small, base, and large). For each experiment, we re-initialized and trained only the head for 10k steps. Despite the significant variation in model size, all three heads produced remarkably similar results (Tab. 7, Fig. 4). This insensitivity to the head’s size strongly suggests that the transformer backbone performs the core generative modeling of the conditional distribution 𝑝(𝑥𝑖 | 𝑥<𝑖). The flowmatching head, akin to the LM head in language models, primarily acts as a lightweight sampler that translates the transformer’s contextual prediction into a continuous token. Consequently, the essential generative logic resides within the transformer’s autoregressive NTP process.

###### 6.2. Tokenizer is the Key to Image Generation

Mitigating Instability under Strong Classifier-Free Guidance. A known failure mode in VAE-based autoregressive models is the emergence of visual artifacts, such as gray patches, particularly under strong classifier-free guidance scales (Fan et al., 2024). While prior work hypothesized this instability stemmed from discontinuities in 1D positional embeddings, our analysis reveals that the root cause lies in the amplification of token-level distributional shifts under high guidance scales.

At inference time, CFG is introduced to enhance conditional fidelity. The guided prediction

###### FM Head Small FM Head Base FM Head Large

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

- Figure 4 | Images generated under different flow-matching heads.

- Table 6 | Configurations for different flow-matching heads.

Layers Hidden Size # Parameters FM Head Small 6 1024 40M

FM Head Base 12 1536 157M FM Head Large 24 2048 528M

- Table 7 | Quantitative results for different flow-matching head configurations. All variants are finetuned from the

baseline with a newly initialized head.

GenEval GenAI-Bench DPG-Bench Baseline 0.59 0.77 85.15

w/ FM Head Small 0.55 0.76 83.46

w/ FM Head Base 0.55 0.75 84.68 w/ FM Head Large 0.56 0.77 85.50

𝑣˜ is computed via an interpolation: 𝑣˜(𝑥|𝑦) = (1 − 𝑤) · 𝑣𝜃(𝑥|∅) + 𝑤 · 𝑣𝜃(𝑥|𝑦) (4)

where 𝑣𝜃(𝑥|∅) and 𝑣𝜃(𝑥|𝑦) are the unconditional and conditional predictions, and 𝑤 is guidance scale. In diffusion models, inference with a high guidance scale is stable because latent variables are typically normalized, ensuring that conditional and unconditional predictions maintain a consistent scale. However, in token-level autoregressive models, global normalization of the entire latent tensor does not enforce per-token statistical consistency. Consequently, small discrepancies between conditional and unconditional predictions are magnified by a large guidance scale, leading to a significant drift in the statistics of generated tokens over the sequence.

We empirically demonstrate this phenomenon in Fig. 5. At a moderate guidance scale of 1.5, the per-token mean and variance remain stable throughout the generation process. In contrast, at a high guidance scale of 3.0, both statistics diverge significantly for later tokens, a distributional shift that corresponds directly to the appearance of visual artifacts. Our tokenizer design, which incorporates channel-wise normalization (see Equation (3)), directly addresses this issue by enforcing per-token statistical stability. This simple but critical design choice mitigates the instability, enabling the use of strong guidance without degrading image quality.

- A Regularized Latent Space is Critical for Generation A key finding of our work is a counterintuitive inverse correlation between the generation loss and the final synthesis quality of the autoregressive model. Specifically, applying higher noise intensity (𝛾 in Equation (3)) during tokenizer training increases generation loss but paradoxically improves the quality of the generated images. For instance, NextStep-1 uses a tokenizer trained at 𝛾 = 0.5, which incurred the highest generation loss yet produced the highest-fidelity images. Conversely, tokenizers trained for low generation loss caused the autoregressive model to yield outputs resembling pure noise.

We attribute this phenomenon to noise regularization, cultivating a well-conditioned latent space. This process enhances two key properties: the tokenizer decoder’s robustness to latent perturbations (Fig. 6) and a more dispersed latent distribution (Fig. 7), a property prior work has also found beneficial for generation (Sun et al., 2024c; Yang et al., 2025; Yao et al., 2025). While it remains unclear whether robustness or dispersion plays a critical role, these results underscore

|[Figure 167]|
|---|

CFG=1.5

|[Figure 168]|
|---|

CFG=3.0

###### w/o channel-wise input latent normalization w/ channel-wise input latent normalization

|[Figure 169]<br><br>CFG=1.5<br><br>[Figure 170]|
|---|

|[Figure 171]<br><br>CFG=3.0<br><br>[Figure 172]|
|---|

|[Figure 173]<br><br>CFG=7.0<br><br>[Figure 174]|
|---|

|[Figure 175]<br><br>CFG=15.0<br><br>[Figure 176]|
|---|

- Figure 5 | Evolution of per-token mean and variance over sampling steps under two CFG settings. At CFG = 1.5, the mean and variance stay close to 0 and 1, respectively, indicating stability. At CFG = 3.0,

they drift significantly, causing image quality degradation. With normalization, the distributions of output latents remain stable across all CFG settings.

Table 8 | Comparison of reconstruction performance on ImageNet-1K 256×256 (Deng et al., 2009).

Tokenizer Latent Shape PSNR ↑ SSIM ↑ Discrete Tokenizer

SBER-MoVQGAN (270M) (Zheng et al., 2022) 32x32 27.04 0.74 LlamaGen (Sun et al., 2024a) 32x32 24.44 0.77 VAR (Tian et al., 2024) 680 22.12 0.62 TiTok-S-128 (Yu et al., 2024b) 128 17.52 0.44 Sefltok (Wang et al., 2025b) 1024 26.30 0.81

###### Continuous Tokenizer

Stable Diffusion 1.5 (Rombach et al., 2022) 32x32x4 25.18 0.73 Stable Diffusion XL (Podell et al., 2024) 32x32x4 26.22 0.77 Stable Diffusion 3 Medium (Esser et al., 2024) 32x32x16 30.00 0.88 Flux.1-dev (Labs, 2024) 32x32x16 31.64 0.91 NextStep-1 32x32x16 30.60 0.89

the practical benefits of noise-based regularization and highlight promising directions for future analysis.

Reconstruction Quality is the Upper Bound of Generation Quality. The reconstruction fidelity of the image tokenizer fundamentally determines the upper bound for the quality of the final generated image, particularly for fine details and textures. This principle has been validated in numerous recent studies (Dai et al., 2023; Esser et al., 2024; Labs, 2024), leading to a trend

Original

Stable Diffusion XL Stable Diffusion 3 Medium Flux.1-dev NextStep

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

noisestd=0.2noisestd=0.5

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

- Figure 6 | Impact of Noise Perturbation on Image Tokenizer Performance. The top panel displays quantitative metrics (rFID↓, PSNR↑, and SSIM↑) versus noise intensity. The bottom panel presents

qualitative reconstruction examples at noise standard deviations of 0.2 and 0.5.

in the diffusion paradigm of building generative models on top of VAEs with exceptional reconstruction performance (e.g., PSNR > 30). In contrast, VQ-based autoregressive models have historically struggled to surpass this threshold, as shown in Tab. 8. While a trade-off between reconstruction and generation quality is often debated (Yao et al., 2025), our work successfully applies autoregressive models to high-fidelity continuous VAEs, bridging this gap.

###### 6.3. Limitations and Challenges

Artifacts. While NextStep-1 successfully demonstrates that autoregressive models can operate on high-dimensional continuous latent spaces, achieving generation quality comparable to diffusion models, this approach also introduces unique stability challenges. We observed the emergence of several distinct generative artifacts when transitioning from a VAE with a lowerdimensional latent space (e.g., spatial downsample factor is 8 and number of latent channels is 4) to one with a higher-dimensional space (e.g., spatial downsample factor is 8 and number of latent channels is 16). While the former configuration produced stable outputs, the latter occasionally exhibited failure modes, as illustrated in Fig. 8.

While the underlying causes remain an open question, we identify several plausible contributing factors: (1) Local noise or block-shaped artifacts emerging in the later stages of generation may arise from numerical instabilities; (2) Global noise across the image may reflect under-convergence, implying that additional training could mitigate the issue; and (3) Subtle grid-like artifacts could reveal limitations of the 1D positional encoding in capturing 2D spatial

Latent Distribution of Flux.1-dev VAE

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

Latent Distribution of NextStep-1 VAE w/o Noise

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Latent Distribution of NextStep-1 VAE

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

- Figure 7 | Latent distributions in 16 channels for three VAE variants: Flux.1-dev, NextStep-1 w/o noise, and NextStep-1. Blue bars show empirical histograms; red lines indicate the standard normal distribution.

NextStep-1 VAE aligns best with the normal distribution, reflecting a dispersed latent distribution.

relationships.

Inference Latency of Sequential Decoding. A theoretical analysis of per-token latency on an H100 GPU (983 TFLOPS, 3.36 TB/s bandwidth) with a batch size of 1, as detailed in Tab. 9, decomposes the contributions of individual components. The results show that the dominant bottleneck lies in the serial decoding of the LLM, while the multi-step sampling in the flowmatching head also constitutes a substantial portion of the per-token generation cost. These observations suggest two promising directions for accelerating inference. First, the efficiency of the flow matching head could be improved by reducing its parameter count, applying distillation to achieve few-step generation (Meng et al., 2023), or using more advanced few-step samplers (Lu et al., 2022, 2025). Second, the autoregressive backbone could be accelerated by adapting recent advances from the LLM field, such as speculative decoding (Leviathan et al., 2023) or multi-token prediction (Gloeckle et al., 2024), to the domain of image token generation.

Challenges in High-Resolution Training. Our framework faces two primary challenges in scaling to high-resolution image generation, particularly when compared to diffusion models,

BottomsolidcolorBottomnoiseNoisyimageGrid

|[Figure 187]|[Figure 188]|[Figure 189]|[Figure 190]|[Figure 191]|
|---|---|---|---|---|
|[Figure 192]|[Figure 193]|[Figure 194]|[Figure 195]|[Figure 196]|
|[Figure 197]|[Figure 198]|[Figure 199]|[Figure 200]|[Figure 201]|
|[Figure 202]|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|

Figure 8 | Failure cases for high-dimensional continuous tokens.

which benefit from well-established techniques (Chen et al., 2024b; Esser et al., 2024) in this domain. First, the strictly sequential nature of autoregressive generation requires substantially more training steps to converge at higher resolutions. In contrast, diffusion models refine the entire image in parallel at each iteration, enabling more direct exploitation of 2D spatial inductive biases. Second, techniques recently developed for high-resolution diffusion models, such as timestep shift, are difficult to adapt to our setting. This limitation arises because the Flow Matching Head acts primarily as a lightweight sampler, while the transformer backbone performs the core generative modeling; thus, modifications to the sampling process have only a marginal impact on the final output. Designing high-resolution generation strategies specifically for patch-wise autoregressive models remains an important direction for future research.

Challenges in SFT. SFT in our autoregressive framework poses unique challenges compared to diffusion models. We observe that fine-tuning on small, high-quality datasets exhibits unstable dynamics. In contrast to diffusion models, which can often adapt to a target distribution and maintain stable and general image generation with only a few thousand samples, our SFT process yields substantial improvements only when trained on datasets at the million-sample scale. With smaller datasets, the model remains in a precarious equilibrium; it either improves marginally with negligible impact or abruptly overfits to the target distribution. Consequently, identifying an intermediate checkpoint that achieves alignment with the target distribution while preserving general generative capability remains a significant challenge.

Table 9 | Inference latency breakdown at 983 TFLOP/s compute and 3.36 TB/s memory bandwidth.

Last-token Latency (ms) Accumulated Latency (s) LLM Decoder LM Head FM Head Total w/o FM Head

Sequence Length

256 7.20 0.40 3.40 2.82 1.95 1024 7.23 0.40 3.40 11.31 7.83 4096 7.39 0.40 3.40 45.77 31.86

#### Contributors and Acknowledgments

We designate researchers as those who are involved in the development of NextStep-1, while contributors refer to those who provide support in areas such as data, systems, platforms, early version work, or part-time contributions. ★ indicates core executors, and † indicates the project leader. Authors are listed alphabetically by first name.

Researchers: Chunrui Han★, Guopeng Li★, Jingwei Wu★, Quan Sun★†, Yan Cai★, Yuang Peng★, Zheng Ge★†, Deyu Zhou, Haomiao Tang, Hongyu Zhou, Kenkun Liu

Contributors: Ailin Huang, Bin Wang, Changxin Miao, Deshan Sun, En Yu, Fukun Yin, Gang Yu, Hao Nie, Haoran Lv, Hanpeng Hu, Jia Wang, Jian Zhou, Jianjian Sun, Kaijun Tan, Kang An, Kangheng Lin, Liang Zhao, Mei Chen, Peng Xing, Rui Wang, Shiyu Liu, Shutao Xia, Tianhao You, Wei Ji, Xianfang Zeng, Xin Han, Xuelin Zhang, Yana Wei, Yanming Xu, Yimin Jiang, Yingming Wang, Yu Zhou, Yucheng Han, Ziyang Meng

Sponsors: Binxing Jiao, Daxin Jiang, Xiangyu Zhang, Yibo Zhu Acknowledgments: We would like to sincerely thank Tianhong Li and Yonglong Tian for their insightful discussions.

#### References

- S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong,

- Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie,
- Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

J. Baldridge, J. Bauer, M. Bhutani, N. Brichtova, A. Bunner, L. Castrejon, K. Chan, Y. Chen,

- S. Dieleman, Y. Du, et al. Imagen 3. arXiv preprint arXiv:2408.07009, 2024.

J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al. Improving image generation with better captions. OpenAI blog, 2023.

- T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

- T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems (NeurIPS), 2020.

- Q. Cai, J. Chen, Y. Chen, Y. Li, F. Long, Y. Pan, Z. Qiu, Y. Zhang, F. Gao, P. Xu, et al. Hidream-i1:

- A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

J. Chang, Y. Fang, P. Xing, S. Wu, W. Cheng, R. Wang, X. Zeng, G. Yu, and H.-B. Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977, 2025.

- B. Chen, D. Martí Monsó, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in neural information processing systems (NeurIPS), 2024a.

J. Chen, C. Ge, E. Xie, Y. Wu, L. Yao, X. Ren, Z. Wang, P. Luo, H. Lu, and Z. Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision (ECCV), 2024b.

J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025a.

- X. Chen, C. Wu, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, and P. Luo. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.

- X. Dai, J. Hou, C.-Y. Ma, S. Tsai, J. Wang, R. Wang, P. Zhang, S. Vandenhende, X. Wang, A. Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.

G. deepmind Imagen4 team. Imagen4, 2025. URL https://storage.googleapis.com/dee pmind-media/Model-Cards/Imagen-4-Model-Card.pdf.

M. Deitke, R. Liu, M. Wallingford, H. Ngo, O. Michel, A. Kusupati, A. Fan, C. Laforte, V. Voleti, S. Y. Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems (NeurIPS), 2023.

- C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2009.

- R. Dong, C. Han, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J. Sun, H. Zhou, H. Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. In International Conference on Learning Representations (ICLR), 2024.

- S. M. A. Eslami, S. Liu, A. v. d. Oord, O. Vinyals, M. J. Wainwright, and I. Sutskever. Taming transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), 2021.

- P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer,

- F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), 2024.

L. Fan, T. Li, S. Qin, Y. Li, C. Sun, M. Rubinstein, D. Sun, K. He, and Y. Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.

L. Fan, L. Tang, S. Qin, T. Li, X. Yang, S. Qiao, A. Steiner, C. Sun, Y. Li, T. Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025.

Y. Gao, L. Gong, Q. Guo, X. Hou, Z. Lai, F. Li, L. Li, X. Lian, C. Liao, L. Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

Y. Ge, S. Zhao, J. Zhu, Y. Ge, K. Yi, L. Song, C. Li, X. Ding, and Y. Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arxiv:2404.14396, 2024.

- G. Gemini2. Experiment with gemini 2.0 flash native image generation, 2025. URL https: //developers.googleblog.com/en/experiment-with-gemini-20-flash-nativ e-image-generation.

- D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In Advances in neural information processing systems (NeurIPS), 2023.

- F. Gloeckle, B. Y. Idrissi, B. Rozière, D. Lopez-Paz, and G. Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

J. Han, J. Liu, Y. Jiang, B. Yan, Y. Zhang, Z. Yuan, B. Peng, and X. Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025.

- X. Han, Y. Wu, L. Shi, H. Liu, H. Liao, L. Qiu, W. Yuan, X. Gu, Z. Dong, and S. Cui. Mvimgnet2. 0: A larger-scale dataset of multi-view images. arXiv preprint arXiv:2412.01430, 2024.

- X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. B. Girshick. Segment Anything. In IEEE International Conference on Computer Vision (ICCV), 2023.

- S. Kou, J. Jin, Z. Liu, C. Liu, Y. Ma, J. Jia, Q. Chen, P. Jiang, and Z. Deng. Orthus: Autoregressive interleaved image-text generation with modality-specific heads. arXiv preprint arXiv:2412.00127, 2024.

B. F. Labs. Flux, 2024. URL https://github.com/black-forest-labs/flux. B. F. Labs. Flux.1-fill-dev, 2025. URL https://huggingface.co/black-forest-labs/FL

UX.1-Fill-dev.

B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

- B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, X. Xia, P. Zhang, G. Neubig, and D. Ramanan. Evaluating and improving compositional text-to-visual generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024a.

D. Li, A. Kamko, E. Akhgari, A. Sabet, L. Xu, and S. Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024b.

T. Li, Y. Tian, H. Li, M. Deng, and K. He. Autoregressive image generation without vector quantization. In Advances in neural information processing systems (NeurIPS), 2024c.

- C. Liao, L. Liu, X. Wang, Z. Luo, X. Zhang, W. Zhao, J. Wu, L. Li, Z. Tian, and W. Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

- Z. Lin, D. Pathak, B. Li, J. Li, X. Xia, G. Neubig, P. Zhang, and D. Ramanan. Evaluating text-tovisual generation with image-to-text generation. arXiv preprint arXiv:2404.01291, 2024.

- Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In International Conference on Machine Learning (ICLR), 2023a.

Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2023b.

S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. Step1xedit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2019.

- C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022.

- C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, pages 1–22, 2025.

- G. Ma, H. Huang, K. Yan, L. Chen, N. Duan, S. Yin, C. Wan, R. Ming, X. Song, X. Chen, Y. Zhou, D. Sun, D. Zhou, J. Zhou, K. Tan, K. An, M. Chen, W. Ji, Q. Wu, W. Sun, X. Han, Y. Wei, Z. Ge, A. Li, B. Wang, B. Huang, B. Wang, B. Li, C. Miao, C. Xu, C. Wu, C. Yu, D. Shi, D. Hu, E. Liu,

- G. Yu, G. Yang, G. Huang, G. Yan, H. Feng, H. Nie, H. Jia, H. Hu, H. Chen, H. Yan, H. Wang,
- H. Guo, H. Xiong, H. Xiong, J. Gong, J. Wu, J. Wu, J. Wu, J. Yang, J. Liu, J. Li, J. Zhang, J. Guo,

- J. Lin, K. Li, L. Liu, L. Xia, L. Zhao, L. Tan, L. Huang, L. Shi, M. Li, M. Li, M. Cheng, N. Wang,

- Q. Chen, Q. He, Q. Liang, Q. Sun, R. Sun, R. Wang, S. Pang, S. Yang, S. Liu, S. Liu, S. Gao, T. Cao, T. Wang, W. Ming, W. He, X. Zhao, X. Zhang, X. Zeng, X. Liu, X. Yang, Y. Dai, Y. Yu,

- Y. Li, Y. Deng, Y. Wang, Y. Wang, Y. Lu, Y. Chen, Y. Luo, Y. Luo, Y. Yin, Y. Feng, Y. Yang, Z. Tang,
- Z. Zhang, Z. Yang, B. Jiao, J. Chen, J. Li, S. Zhou, X. Zhang, X. Zhang, Y. Zhu, H.-Y. Shum, and D. Jiang. Step-video-t2v technical report: The practice, challenges, and future of video foundation model, 2025a. URL https://arxiv.org/abs/2502.10248.

- X. Ma, P. Sun, H. Ma, H. Tang, C.-Y. Ma, J. Wang, K. Li, X. Dai, Y. Shi, X. Ju, et al. Tokenshuffle: Towards high-resolution image generation with autoregressive models. arXiv preprint arXiv:2504.17789, 2025b.

C. Meng, R. Rombach, R. Gao, D. Kingma, S. Ermon, J. Ho, and T. Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14297–14306, 2023.

- Y. Niu, M. Ning, M. Zheng, B. Lin, P. Jin, J. Liao, K. Ning, B. Zhu, and L. Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

- D. A. Oliveira and D. M. de Matos. Storyreasoning dataset: Using chain-of-thought for scene understanding and grounded story generation. arXiv preprint arXiv:2505.10292, 2025.

OpenAI. Introducing gpt-4.1 in the api. OpenAI Blog, 2025a. URL https://openai.com/i ndex/gpt-4-1.

OpenAI. Introducing 4o image generation, 2025b. URL https://openai.com/index/intro ducing-4o-image-generation.

- X. Pan, S. N. Shukla, A. Singh, Z. Zhao, S. K. Mishra, J. Wang, Z. Xu, J. Chen, K. Li, F. JuefeiXu, J. Hou, and S. Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

- Y. Peng, Y. Cui, H. Tang, Z. Qi, R. Dong, J. Bai, C. Han, Z. Ge, X. Zhang, and S.-T. Xia. Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855, 2024.

- D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR), 2024.

- Q. Qin, L. Zhuo, Y. Xin, R. Du, Z. Li, B. Fu, Y. Lu, J. Yuan, X. Li, D. Liu, et al. Lumina-image 2.0:

- A unified and efficient image generative framework. arXiv preprint arXiv:2503.21758, 2025.

- A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving language understanding by generative pre-training. San Francisco, CA, USA, 2018.

- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019.

- R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems (NeurIPS), 2023.

- R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2024.

- R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Y. Shi, P. Wang, and W. Huang. Seededit: Align image re-generation to image editing. arXiv preprint arXiv:2411.06686, 2024.

##### Stability-AI. stable-diffusion-3.5-large, 2024. URL https://github.com/Stability-AI/sd 3.5.

- J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

- P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024a.

- Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Zhang, Y. Wang, H. Gao, J. Liu, T. Huang, and X. Wang. Emu: Generative pretraining in multimodality. In International Conference on Learning Representations (ICLR), 2023.

- Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024b.

Y. Sun, H. Bao, W. Wang, Z. Peng, L. Dong, S. Huang, J. Wang, and F. Wei. Multimodal latent

language modeling with next-token diffusion. arXiv preprint arXiv:2412.08635, 2024c. K. K. team. Kolors2.0, 2025. URL https://app.klingai.com/cn.

- R. team. Recraft v3, 2024. URL https://www.recraft.ai/blog/recraft-introduces-a

-revolutionary-ai-model-that-thinks-in-design-language.

K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems (NeurIPS), 2024.

- S. Tong, D. Fan, J. Zhu, Y. Xiong, X. Chen, K. Sinha, M. Rabbat, Y. LeCun, S. Xie, and Z. Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.

M. Tschannen, A. S. Pinto, and A. Kolesnikov. Jetformer: An autoregressive generative model of raw images and text. arXiv preprint arXiv:2411.19722, 2024.

M. Tschannen, C. Eastwood, and F. Mentzer. Givt: Generative infinite-vocabulary transformers. In European Conference on Computer Vision (ECCV), 2025.

- B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik. Diffusion model alignment using direct preference optimization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- B. Wang, B. Wang, C. Wan, G. Huang, H. Hu, H. Jia, H. Nie, M. Li, N. Chen, S. Chen, et al. Step-3 is large yet affordable: Model-system co-design for cost-effective decoding. arXiv preprint arXiv:2507.19427, 2025a.

B. Wang, Z. Yue, F. Zhang, S. Chen, L. Bi, J. Zhang, X. Song, K. Y. Chan, J. Pan, W. Wu, et al. Selftok: Discrete visual tokens of autoregression, by diffusion, and for reasoning. arXiv preprint arXiv:2505.07538, 2025b.

J. Wang, Z. Tian, X. Wang, X. Zhang, W. Huang, Z. Wu, and Y.-G. Jiang. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455, 2025c.

- P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

- X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arxiv:2409.18869, 2024b.

- Y. Wang, S. Yang, B. Zhao, L. Zhang, Q. Liu, Y. Zhou, and C. Xie. Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025d.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems (NeurIPS), 2022.

C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

- C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

- S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

- E. Xie, J. Chen, Y. Zhao, J. Yu, L. Zhu, C. Wu, Y. Lin, Z. Zhang, M. Li, J. Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025a.

J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arxiv:2408.12528, 2024.

J. Xie, Z. Yang, and M. Z. Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025b.

J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems (NeurIPS), 2023.

A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

J. Yang, T. Li, L. Fan, Y. Tian, and Y. Wang. Latent denoising makes good visual tokenizers. arXiv preprint arXiv:2507.15856, 2025.

J. Yao, B. Yang, and X. Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025.

Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

- J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan,

- B. Hutchinson, W. Han, Z. Parekh, X. Li, H. Zhang, J. Baldridge, and Y. Wu. Scaling autoregressive models for content-rich text-to-image generation. In TMLR, 2022.

L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, V. Birodkar, A. Gupta, X. Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024a.

Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.-C. Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems (NeurIPS), 2024b.

T. Z.ai. Cogview4, 2025. URL https://github.com/THUDM/CogView4. K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su. Magicbrush: A manually annotated dataset

for instruction-guided image editing. In Advances in neural information processing systems (NeurIPS), 2023a.

L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), 2023b.

W. Zhang, H. Zhang, X. Li, J. Sun, Y. Shen, W. Lu, D. Zhao, Y. Zhuang, and L. Bing. 2.5 years in class: A multimodal textbook for vision-language pretraining. arXiv preprint arXiv:2501.00958, 2025.

- C. Zheng, T.-L. Vuong, J. Cai, and D. Phung. Movq: Modulating quantized vectors for highfidelity image generation. Advances in Neural Information Processing Systems (NeurIPS), 2022.

###### C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. International Conference on Learning Representations (ICLR), 2025.

