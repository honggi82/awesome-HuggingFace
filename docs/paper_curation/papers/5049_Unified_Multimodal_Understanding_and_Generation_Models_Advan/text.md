1

# Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities

Shanshan Zhao*, Xinjie Zhang*, Jintao Guo*, Jiakui Hu, Lunhao Duan, Minghao Fu, Yong Xien Chng, Guo-Hua Wang, Qing-Guo Chen†, Zhao Xu, Weihua Luo, Kaifu Zhang

## arXiv:2505.02567v6[cs.CV]26Jan2026

Abstract—Recent years have seen remarkable progress in both multimodal understanding models and image generation models. Despite their respective successes, these two domains have evolved independently, leading to distinct architectural paradigms: While autoregressive-based architectures have dominated multimodal understanding, diffusion-based models have become the cornerstone of image generation. Recently, there has been growing interest in developing unified frameworks that integrate these tasks. The emergence of GPT-4o’s new capabilities exemplifies this trend, highlighting the potential for unification. However, the architectural differences between the two domains pose significant challenges. To provide a clear overview of current efforts toward unification, we present a comprehensive survey aimed at guiding future research. First, we introduce the foundational concepts and recent advancements in multimodal understanding and text-to-image generation models. Next, we review existing unified models, categorizing them into three main architectural paradigms: diffusion-based, autoregressive-based, and hybrid approaches that fuse autoregressive and diffusion mechanisms. For each category, we analyze the structural designs and innovations introduced by related works. Additionally, we compile datasets and benchmarks tailored for unified models, offering resources for future exploration. Finally, we discuss the key challenges facing this nascent field, including tokenization strategy, cross-modal attention, and data. As this area is still in its early stages, we anticipate rapid advancements and will regularly update this survey. Our goal is to inspire further research and provide a valuable reference for the community. The references associated with this survey are available on https://github.com/AIDC-AI/Awesome-UnifiedMultimodal-Models

Index Terms—Unified multimodal models, Multimodal understanding, Image generation, Autoregressive model, Diffusion model

✦

### 1 INTRODUCTION

In recent years, the rapid advancement of large language models (LLMs), such as LLaMa [1], [2], PanGu [3], [4], Qwen [5], [6], and GPT [7], has revolutionized artificial intelligence. These models have scaled up in both size and capability, enabling breakthroughs across diverse applications. Alongside this progress, LLMs have been extended into multimodal domains, giving rise to powerful multimodal understanding models like LLaVa [8], Qwen-VL [9], [10], InternVL [11], Ovis [12], and GPT4 [13]. These models have expanded their capabilities beyond simple image captioning to performing complex reasoning tasks based on user instructions. On the other hand, image generation technology has also experienced rapid development, with models like SD series [14], [15] and FLUX [16] now capable of producing high-quality images that adhere closely to user prompts.

- • Xinjie Zhang is with Alibaba Group and Hong Kong University of Science and Technology.
- • Jintao Guo and Minghao Fu are with Alibaba Group and Nanjing University.
- • Jiakui Hu is with Alibaba Group and Peking University.
- • Yong Xien Chng is with Alibaba Group and Tsinghua University.
- • Shanshan Zhao, Lunhao Duan, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, and Kaifu Zhang are with Alibaba Group.
- • Shanshan Zhao, Xinjie Zhang, and Jintao Guo contributed equally to this work.
- • Project leader: Qing-Guo Chen.

The predominant architectural paradigm for LLMs and multimodal understanding models is autoregressive generation [17], which relies on decoder-only structures and nexttoken prediction for sequential text generation. In contrast, the field of text-to-image generation has evolved along a different trajectory. Initially dominated by Generative Adversarial Networks (GANs) [18], image generation has since transitioned to diffusion-based models [19], which leverage architectures like UNet [14] and DiT [20], [21] alongside advanced text encoders such as CLIP [22] and T5 [23]. Despite some explorations into using LLM-inspired architectures for image generation [24], [25], [26], diffusionbased approaches remain the state-of-the-art in terms of performance currently.

While autoregressive models lag behind diffusion-based methods in image generation quality, their structural consistency with LLMs makes them particularly appealing for developing unified multimodal systems. A unified model capable of both understanding and generating multimodal content holds immense potential: it could generate images based on complex instructions, reason about visual data, and visualize multimodal analyses through generated outputs. The unveiling of GPT-4o’s enhanced capabilities [27] in March 2025 has further highlighted this potential, sparking widespread interest in unification.

However, designing such a unified framework presents significant challenges. It requires integrating the strengths

[Figure 1]

|Publicly Available/Unavailable|
|---|

[Figure 2]

[Figure 3]

TBAC-UniImage

Ovis-U1

[Figure 4]

Tar

[Figure 5]

GPT-4o Gemini 2.0 Flash

[Figure 6]

[Figure 7]

[Figure 8]

UniPic

[Figure 9]

Ming-Omni

UniWorld-V1

[Figure 10]

M2-omni

UniFluid

[Figure 11]

[Figure 12]

[Figure 13]

Bifrost-1

Pisces

Show-o2

[Figure 14]

[Figure 15]

UGen OmniMamba

|Model: Supports modal input/ output beyond image and text.<br><br>|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

OmniGen2 Unicode2

X-Omni

[Figure 19]

[Figure 20]

UniTok Harmon

[Figure 21]

[Figure 22]

[Figure 23]

Qwen-Image

[Figure 24]

UniFork

UniLIP

[Figure 25]

Janus-Pro VARGPT

[Figure 26]

[Figure 27]

SemHiTok UniDisc

###### 6-8

4-5

###### 9-12

1-3

[Figure 28]

[Figure 29]

MIO

Emu3

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

2025

Selftok

[Figure 34]

MonoFormer ILLUME+

[Figure 35]

Show-o

Tuna

###### EMU3.5

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

VARGPT-1.1

Mogao

[Figure 41]

VILA-U

Ming-UniVision

Anole

10-12

MammothModa2

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

OpenUni

[Figure 46]

Ming-Lite-Uni

Transfusion

QLIP

[Figure 47]

[Figure 48]

LightFusion

UniModel

Unified-IO 2

[Figure 49]

FUDOKI

7-9

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

HBridge

###### PUMA MMAR

Next-GPT

UniPic 2.0

Emu2

[Figure 56]

MetaQueries

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Janus Janus-flow

[Figure 61]

[Figure 62]

VL-GPT

1-6

SEED-LLAMA

Ming-Flash-Omni

EMMA

[Figure 63]

[Figure 64]

Nexus-Gen

[Figure 65]

Orthus MUSE-VL

[Figure 66]

[Figure 67]

[Figure 68]

DreamLLM LaVIT Emu

LongCat-Flash-Omni

[Figure 69]

[Figure 70]

[Figure 71]

TokLIP

Qwen3-Omni

[Figure 72]

2024

Dual Diffusion LMFusion MetaMorph

[Figure 73]

[Figure 74]

SEED

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

LaViDa-O

[Figure 79]

BLIP3-o

Liquid

OneCat

[Figure 80]

[Figure 81]

Chameleon

7-12

SEED-X

[Figure 82]

[Figure 83]

[Figure 84]

Tokenflow ILLUME

BAGEL

[Figure 85]

2023

[Figure 86]

Mini-Gemini LWM

[Figure 87]

[Figure 88]

[Figure 89]

SynerGen-VL

DualToken

MMaDA

[Figure 90]

[Figure 91]

MM-Interleaved

AnyGPT

[Figure 92]

[Figure 93]

[Figure 94]

UniToken

Muddit

OmniFlow

[Figure 95]

[Figure 96]

Video-LaVIT X-VILA

[Figure 97]

[Figure 98]

MindOmni

Spider

- Fig. 1. Timeline of Publicly Available and Unavailable Unified Multimodal Models. The models are categorized by their release years, from 2023 to 2025. Models underlined in the diagram represent any-to-any multimodal models, capable of handling inputs or outputs beyond text and image, such as audio, video, and speech. The timeline highlights the rapid growth in this field.

of autoregressive models for reasoning and text generation with the robustness of diffusion-based models for highquality image synthesis. Key questions remain unresolved, including how to tokenize images effectively for autoregressive generation. Some approaches [28], [29], [30] employ VAE [31] or VQ-GAN [32] commonly used in diffusionbased pipelines, or relevant variants, while others [33], [34], [35] utilize semantic encoders like EVA-CLIP [36] and OpenAI-CLIP [22]. Additionally, while discrete tokens are standard for text in autoregressive models, continuous representations may be more suitable for image tokens, as suggested by emerging research [25]. Beyond tokenization, hybrid architectures [37], [38], [39] that combine parallel diffusion strategies with sequential autoregressive generation offer another promising approach aside from naive autoregressive architecture. Thus, both image tokenization techniques and architectural designs remain in their nascent stages for unified multimodal models.

modal models. These resources span multimodal understanding, text-to-image generation, image editing, and other relevant tasks, providing a foundation for future exploration. Finally, we discuss the key challenges facing this nascent field, including efficient tokenization strategy, data construction, model evaluation, etc. Tackling these challenges will be crucial for advancing the capabilities and scalability of unified multimodal models.

In the community, there exist excellent surveys on large language models [40], [41], multimodal understanding [42], [43], [44], and image generation [45], [46], while our work focuses specifically on the integration of understanding and generation tasks. Readers are encouraged to consult these complementary surveys for a broader perspective on related topics. We aim to inspire further research in this rapidly evolving field and provide a valuable reference for the community. Materials including relevant references, datasets, and benchmarks associated with this survey are available on GitHub and will be regularly updated to reflect ongoing advancements.

To provide a comprehensive overview of the current state of unified multimodal models (as illustrated in Fig. 1), thereby benefiting future research endeavors, we present this survey. We begin by introducing the foundational concepts and recent advancements in both multimodal understanding and image generation, covering both autoregressive and diffusion-based paradigms. Next, we review existing unified models, categorizing them into three main architectural paradigms: diffusion-based, autoregressive-based, and hybrid approaches that fuse autoregressive and diffusion mechanisms. Within the autoregressive and hybrid categories, we further classify models based on their image tokenization strategies, reflecting the diversity of approaches in this area.

2 PRELIMINARY

2.1 Multimodal Understanding Model

Multimodal understanding models refer to LLM-based architectures capable of receiving, reasoning over, and generating outputs from multimodal inputs [47]. These models extend the generative and reasoning capabilities of LLMs beyond textual data, enabling rich semantic understanding across diverse information modalities [42], [48]. Most efforts of existing methods focus on vision-language understanding (VLU), which integrates both visual (e.g., images and videos) and textual inputs to support a more comprehensive understanding of spatial relationships, objects, scenes, and

Beyond architecture, we assemble datasets and benchmarks tailored for training and evaluating unified multi-

###### Dual-encoder architecture

###### Next-Pixel Prediction

Autoregressive Modeling

###### Connector types

Single pixel as a unit

Text

[Figure 99]

Text Embeddings

Flatten Pixels

Text Encoder

[Figure 100]

… Pixel sequence

𝑥 𝑥 𝑥 𝑥 𝑥

…

Text

MLP

Contrastive Learning

[Figure 101]

Decoder

Autoregressive Models

Image

###### Next-Token Prediction

Vision Encoder

[Figure 102]

Single token as a unit

| | | | |
|---|---|---|---|
| | | | |
| | |[Figure 103]| |
| | | | |

Encoder

K

Q-Former

Flatten

Image Embeddings

…

…

𝑥 𝑥 𝑥 𝑥

[s]

V

Token sequence

###### LLM-based architecture

Discrete Vector Quantization

Text

Next-Multiple-Tokens Prediction

Learnable Queries

Text Embeddings

[Figure 104]

[Figure 105]

Multiple tokens as a unit

Decoder

Encoder

| | | | |
|---|---|---|---|
| |[Figure 106]| | |
| | | | |
| | | | |

Encoder

Text

… Group

Spatial groups

Scale tokens

Frequency Spectrum

[Figure 107]

LLM

[Figure 108]

Image Embeddings Image

LLM

[Figure 109]

…

[Figure 110]

MH-Attn

[Figure 111]

Vision

[Figure 112]

Q K V

Q K V

Encoder Connector

Fig. 4. Illustration of core components in autoregressive models, including the autoregression sequence modeling and discrete vector quantization. Exiting autoregressive models can be roughly divided into three types: Next-Pixel Prediction flattens the image into a pixel sequence, Next-Token Prediction converts the image into a token sequence via a visual tokenizer, and Next-Multiple-Tokens Prediction outputs multiple tokens in an autoregressive step.

- Fig. 2. Architecture of multimodal understanding models, containing multimodal encoders, a connector, and a LLM. The multimodal encoders transform images, audio, or videos into features, which are processed by the connector as the input of LLM. The architectures of theconnector can be broadly categorized by three types: projection-based, query-based, and fusion-based connectors.

𝑥 … 𝑥 -  𝑞(𝑥 |𝑥 ) 𝑥 … 𝑥

𝑝 (𝑥 |𝑥 )

[Figure 113]

[Figure 114]

[Figure 115]

Forward process: Reverse process:

|Pixel Space 𝑧<br><br>𝑧 𝑧<br><br>𝑥<br><br>𝑥<br><br>Identity<br><br>Style Sketch<br><br>Conditions<br><br>Features<br><br>𝜏 …<br><br>ℰ Forward process<br><br>Latent Space<br><br>Q K V<br><br>Q K V<br><br>Denoising<br><br>𝑧 𝑧<br><br>×(𝑇 − 1)<br><br>𝒟<br><br>Text-to-image Diffusion Model|
|---|

- Fig. 3. Illustration of diffusion-based text-to-image generation models, where various conditions beyond text are introduced to steer the outcomes. The image generation is formulated as a pair of Markov chains: a forward process that gradually corrupts input data by adding Gaussian noise, and a reverse process that learns a parameterized distribution to iteratively denoise back to the input data.

of the mode. With the emergence of powerful LLMs, VLU models have progressively shifted toward decoder-only architectures that incorporate frozen or minimally fine-tuned LLM backbones. These methods primarily transform image embeddings through a connector with different structures, as illustrated in Fig. 2. Specifically, MiniGPT-4 [57] utilized a single learnable layer to project CLIP-derived image embeddings into the token space of Vicuna [58]. BLIP-2 [53] introduced a querying transformer, to bridge a frozen visual encoder with a frozen LLM (e.g., Flan-T5 [59] or Vicuna [58]), enabling efficient vision-language alignment with significantly fewer trainable parameters. Flamingo [60] employed gated cross-attention layers to connect a pretrained vision encoder with a frozen Chinchilla [61] decoder.

Recent advances in VLU highlight a shift toward general multimodal understanding. GPT-4V [62] extends the GPT4 framework [13] to analyze image inputs provided by the user, demonstrating strong capabilities in visual reasoning, captioning, and multimodal dialogue, despite its proprietary nature. Gemini [63], built upon a decoder-only architecture, supports image, video, and audio modalities, with its Ultra variant setting new benchmarks in multimodal reasoning tasks. The Qwen series exemplifies scalable multimodal design: Qwen-VL [5] incorporates visual receptors and grounding modules, while Qwen2-VL [9] adds dynamic resolution handling and M-RoPE for robust processing of varied inputs. LLaVA-1.5 [64] and LLaVANext [65] use CLIP-based vision encoders and Vicuna-style LLMs for competitive performance in VQA and instructionfollowing tasks. The InternVL series [11], [66], [67] explore a unified multimodal pre-training strategy, which simultaneously learns from both text and visual data to enhance performance across various visual-linguistic tasks. Ovis [12] introduces a structural embedding alignment mechanism through a learnable visual embedding lookup table, thus producing visual embeddings that structurally mirror textual tokens. Recently, some models have explored scalable and unified architectures for multimodal processing. DeepSeek-VL2 [68] employs a Mixture-of-Experts (MoE) architecture to enhance cross-modal reasoning. Overall, these

abstract concepts [49], [50], [51]. A typical architecture of multimodal understanding models is illustrated in Fig. 2. These models operate within a hybrid input space, where textual data are represented discretely, while visual signals are encoded as continuous representations [52]. Similar to traditional LLMs, their outputs are generated as discrete tokens derived from internal representations, using classification-based language modeling and task-specific decoding strategies [8], [53].

Early VLU models primarily focused on aligning visual and textual modalities using dual-encoder architectures, wherein images and text are first encoded separately and then jointly reasoned over via aligned latent representations, including CLIP [22], ViLBERT [54], VisualBERT [55], and UNITER [56]. Although these pioneering models established key principles for multimodal reasoning, they depended heavily on region-based visual preprocessing and separate encoders, limiting the scalability and generality

models mark a clear progression toward instruction-tuned and token-centric frameworks capable of addressing diverse multimodal tasks in a unified and scalable manner.

#### 2.2 Text-to-Image Model

Diffusion models. Diffusion models (DM) formulate generation as a pair of Markov chains: a forward process that gradually corrupts data x0 by adding Gaussian noise over T timesteps to produce xT, and a reverse process that learns a parameterized distribution to iteratively denoise back to the data manifold [19], [69], [70]. Formally, as shown in Fig. 3 in the forward process, given the date distribution x0 ∼ q(x0), at each step t, the data xt is noised:

q(x1:T|x0) :=

T

q(xt|xt−1), (1)

t=1

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI), (2)

where βt is the variance hyperparameters of the noise. During the reverse process, the model progressively denoises the data to approximate the reverse of the Markov chain. The reverse transition pθ(xt−1|xt) is parameterized as:

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),Σθ(xt,t)), (3) where the network parameterizes the mean µθ(xt,t) and variance Σθ(xt,t). The network takes the noised data xt and time step t as inputs, and outputs the parameters of the normal distribution for noise prediction. The noise vector is initiated by sampling xT ∼ p(xT), and then successively sample from the learned transition kernels xt−1 ∼ pθ(xt−1|xt) until t = 1. The training objective is to minimize a Variational Lower-Bound of the Negative LogLikelihood: L = Eq(x0,x1:T) ∥ϵθ(xt,t) − ϵ∗(xt,t)∥2 , where ϵθ(xt,t) is the model’s prediction of the noise at timestep t, and ϵ∗(xt,t) is the true noise added at that timestep.

Early diffusion models utilized a U-Net architecture to approximate the score function [19]. The U-Net design, based on a Wide ResNet, integrates residual connections and self-attention blocks to preserve gradient flow and recover fine-grained image details. These methods could be roughly divided into pixel-level methods and latent-featurelevel methods. The pixel-level methods directly operate the diffusion process in the pixel space, including GLIDE [71] that introduced “classifier-free guidance” and Imagen [72] that employ the pretrained large language model, i.e., T5XXL [23], as text encoder. However, these methods suffer expensive raining and inference computation costs, leading to the development of Latent Diffusion Models (LDMs) [14] that operate in the latent space of a pre-trained variational autoencoder. LDMs achieve computational efficiency while preserving high-generation quality, thus inspiring various diffusion-based generative models, including VQ-Diffusion [73], SD 2.0 [74], SD XL [75], and UPainting [76].

Advancements in transformer architectures have led to the adoption of transformer-based models in diffusion processes. The pioneering Diffusion Transformers (DiT) [20] transforms input images into a sequence of patches and feeds them through a series of transformer blocks. DiT takes additional conditional information such as the diffusion timestep t and a conditioning signal c as inputs.

The success of DiT inspired many advanced generative methods, including REPA [77] that injects self-supervised visual representations into diffusion training to strengthen large-scale performance, SD 3.0 [15] use two separate sets of weights to model text and image modality, and others [78], [79], [80]. For text encoders, these methods primarily use utilized contrastive learning to align image and text modalities in a shared latent space, which jointly trained separate image and text encoders on large-scale image-caption pairs [22], [53], [81]. Specifically, GLIDE [71] explores both CLIP guidance and classifier-free guidance, demonstrating that CLIP-conditioned diffusion outperforms earlier GAN baselines and supports powerful text-driven editing. SD [14] employs a frozen CLIP-ViT-L/14 encoder to condition its latent diffusion denoiser, achieving high-quality samples with efficient computation. SD 3.0 [15] utilizes CLIP ViTL/14, OpenCLIP bigG/14, and T5-v1.1 XXL to transform text into embeddings for generation guidance.

Recent advancements in diffusion models have incorporated LLMs to enhance text-to-image diffusion generation [82], [83], which significantly improves the text-image alignment as well as the quality of generated images. RPG [83] leverages the vision-language prior of multimodal LLMs to reason out complementary spatial layouts from text prompts, and manipulates the object compositions for diffusion models in both text-guided image generation and editing process. However, these methods require different model architectures, training strategies, and parameter configurations for specific tasks, which presents challenges in managing these models. A more scalable solution is to adopt a unified generation model capable of handling a variety of data generation tasks [84], [85], [86], [87]. OmniGen [84] achieves text-to-image generation capabilities and supports various downstream tasks, such as image editing, subject-driven generation, and visual-conditional generation. UniReal [85] treats image-level tasks as discontinuous video generation, treating varying numbers of input and output images as frames, enabling seamless support for tasks such as image generation, editing, customization, and composition. GenArtist [86] provides a unified image generation and editing system, coordinated by a multimodal large language model (MLLM) agent. UniVG [87] treats multi-modal inputs as unified conditions with a single set of weights to enable various downstream applications. As research in this domain advances, it is expected that increasingly unified models will emerge, capable of addressing a broader spectrum of image generation and editing tasks.

Autoregressive models. Autoregressive (AR) models define the joint distribution of a sequence by factorizing it into a product of conditional probabilities, whereby each element is predicted in turn based on all previously generated elements. This paradigm, originally devised for language modeling, has been successfully adapted to vision by mapping an image to a 1D sequence of discrete tokens (pixels, patches, or latent codes). Formally, given a sequence x = (x1,x2,...,xN), the model is trained to generate each element by conditioning all preceding elements:

p(x) =

N

p(xi|x1,x2,...,xi−1;θ). (4)

i=1

where θ is the model parameters. The training objective is to minimize the negative log-likelihood(NLL) loss:

N

L(θ) = −

log p(xi|x1,x2,...,xi−1;θ). (5)

i=1

As shown in Fig. 4, existing methods are divided into three types based on sequence representation strategies: pixelbased, token-based, and multiple-tokens-based models.

- 1) Pixel-based models. PixelRNN [88] was the pioneering

method for next-pixel prediction. It transforms a 2D image into a 1D sequence of pixels and employs LSTM layers to sequentially generate each pixel based on previously generated values. While effective in modeling spatial dependencies, it suffers from high computational costs. PixelCNN [89] introduces dilated convolutions to more efficiently capture long-range pixel dependencies, while PixelCNN++ [90] leverages a discretized logistic mixture likelihood and architectural refinements to enhance image quality and efficiency. Some advanced works [91] have also proposed parallelization methods to reduce computational overhead and enable faster generation, particularly for high-resolution images.

- 2) Token-based models. Inspired by natural language

processing paradigms, token-based AR models convert images into compact sequences of discrete tokens, greatly reducing sequence length and enabling high-resolution synthesis. This process begins with vector quantization (VQ): an encoder-decoder trained with reconstruction and commitment losses learns a compact codebook of latent indices, after which a decoder-only transformer models the conditional distribution over those tokens [92]. Typical VQ models include VQ-VAE-2 [93], VQGAN [32], ViT-VQGAN [94], and others [95], [96], [97] Many works have been investigated to enhance the decoder-only transformer models. LlamaGen [24] applies the VQGAN tokenizer to LLaMA backbones [1], [2], achieving comparable performance with DiTs and showing that generation quality improves with the increase of parameters. In parallel, data-efficient variants like DeLVM [98] achieve comparable fidelity with substantially less data, and models such as AiM [26], ZigMa [99], and DiM [100] integrate linear or gated attention layers from Mamba [101] to deliver faster inference and superior performance. To enrich contextual modeling, stochastic and hybrid decoding strategies have been proposed. Methods like SAIM [102], RandAR [103], and RAR [104] randomly permute patch predictions to overcome rigid raster biases, while SAR [105] generalizes causal learning to arbitrary orders and skip intervals. Hybrid frameworks further blend paradigms: RAL [106] uses adversarial policy gradients to mitigate exposure bias, ImageBART [107] interleaves hierarchical diffusion updates with AR decoding, and DisCo-Diff [108] augments diffusion decoders with discrete latent for best-in-class FID.

- 3) Multiple-tokens-based methods. To improve genera-

tion efficiency, recent AR models have shifted from generating individual tokens to predicting multiple tokens as a group, achieving significant speedups without quality loss. Next Patch Prediction (NPP) [109] aggregates image tokens into patch-level tokens with high information density, thus significantly reducing sequence length. Similarly, Next Block Prediction (NBP) [110] extends grouping to

large spatial blocks, such as rows or entire frames. Neighboring AR (NAR) [111] proposes to predict outward using a localized “next-neighbor” mechanism, and Parallel Autoregression (PAR) [112] partitions tokens into disjoint subsets for concurrent decoding. MAR [25] abandons discrete tokenization and fixed ordering in favor of continuous representations trained with a diffusion loss. Beyond spatial grouping, VAR [113] introduced a coarse-to-fine nextscale paradigm, which inspired various advanced methods, including FlowAR [114], M-VAR [115], FastVAR [116], and FlexVAR [117]. Some frequency-based methods decompose generation spectrally: FAR [118] and NFIG [119] synthesize low-frequency structures before refining high-frequency details. xAR [120] abstractly unifies autoregressive units, including patches, cells, scales, or entire images, under a single framework. These multiple-token methods demonstrate the importance of defining appropriate autoregressive units for balancing fidelity, efficiency, and scalability in modern image generation.

Control mechanisms have also been integrated into autoregressive decoders for more precise editing. ControlAR [121] introduces spatial constraints such as edge maps and depth cues during decoding, allowing fine-grained control over token-level edits. ControlVAR [122] further advances this concept by implementing scale-aware conditioning on image-level features, enhancing coherence and editability. CAR [123] elaborates on a similar concept, focusing on advanced control mechanisms in autoregressive models to enhance the detail and adaptability of visual outputs. For complex scenarios involving multiple objects or temporally coherent sequences, Many-to-Many Diffusion (M2M) [124] adapts the autoregressive framework for multi-frame generation, ensuring semantic and temporal consistency across images. MSGNet [125] combines VQ-VAE with autoregressive modeling to preserve spatial-semantic alignment across multiple entities in a scene. In the medical domain, MVG [126] extends autoregressive image-to-image generation to tasks such as segmentation, synthesis, and denoising by conditioning on paired prompt-image inputs. These textto-image generation AR methods provide the basics of the model architecture and visual modeling methods, effectively advancing research on unified multimodal models for understanding and generation.

### 3 UNIFIED MULTIMODAL MODELS FOR UNDERSTANDING AND GENERATION

Unified multimodal models aim to build a single architecture capable of both understanding and generating data across multiple modalities. These models are designed to process diverse forms of input (e.g., text, image, video, audio) and produce outputs in one or more modalities in a unified manner. A typical unified multimodal framework can be abstracted into three core components: modalityspecific encoders that project different input modalities into a representation space; a modality-fusion backbone that integrates information from multiple modalities and enables cross-modal reasoning; and modality-specific decoders that generate output in the desired modality (e.g., text generation or image synthesis).

TABLE 1 Overview of Unified Multimodal Understanding and Generation Models. This table categorizes models based on their backbone, encoder-decoder architecture, and the specific diffusion or autoregressive models used. It includes information on model, encoder, decoder and the mask used in image generation. The release dates of these models are also provided, highlighting the evolution of multimodal architectures over time.

Architecture

Model Type

Date Backbone Und. Enc. Gen. Enc. Gen. Dec. Mask

Diffusion Model

|Dual Diffusion [127] UniDisc [128] MMaDA [129] FUDOKI [130] Muddit [131] Lavida-O [132] UniModel [133]|a a a a a a a|D-DiT DiT LLaDA DeepSeek-LLM Meissonic (MM-DiT) LaViDa MMDiT in Qwen-Image|SD-VAE MAGVIT-v2 MAGVIT-v2<br><br>SigLIP VQGAN VQGAN SigLIP VQ-Encoder Wan-2.1-VAE|SD-VAE MAGVIT-v2 MAGVIT-v2 VQGAN VQGAN VQ-Encoder Wan-2.1-VAE<br><br>|Bidirect. Bidirect. Bidirect. Bidirect. Bidirect. Bidirect. Bidirect.|2024-12<br>2025-03 2025-05 2025-05 2025-05 2025-09 2025-11<br>|
|---|---|---|---|---|---|---|
| | | | | | | |

Autoregressive Model

LWM [29] b-1 LLaMa-2 VQGAN VQGAN Causal 2024-02 Chameleon [30] b-1 LLaMa-2 VQ-IMG VQ-IMG Causal 2024-05

ANOLE [134] b-1 LLaMa-2 VQ-IMG VQ-IMG Causal 2024-07

Emu3 [135] b-1 LLaMA-2 SBER-MoVQGAN SBER-MoVQGAN Causal 2024-09 MMAR [136] b-1 Qwen2 SD-VAE + EmbeddingViT Diffusion MLP Bidirect. 2024-10 Orthus [137] b-1 Chameleon VQ-IMG+Vision embed. Diffusion MLP Causal 2024-11

SynerGen-VL [138] b-1 InterLM2 SBER-MoVQGAN SBER-MoVQGAN Causal 2024-12 Liquid [139] b-1 GEMMA VQGAN VQGAN Causal 2024-12 UGen [140] b-1 TinyLlama SBER-MoVQGAN SBER-MoVQGAN Causal 2025-03

Harmon [141] b-1 Qwen2.5 MAR MAR Bidirect. 2025-03 TokLIP [142] b-1 Qwen2.5 VQGAN+SigLIP VQGAN Causal 2025-05 Selftok [143] b-1 LLaMA3.1 SD3-VAE+MMDiT SD3 Causal 2025-05 OneCat [144] b-1 Qwen2.5 Convolution + MLP VAE in Infinity VAE in Infinity Causal 2025-09 Emu3.5 [135] b-1 Modified Qwen3 SBER-MoVQGAN SBER-MoVQGAN Causal 2025-10

Emu [145] b-2 LLaMA EVA-CLIP SD Causal 2023-07

LaVIT [146] b-2 LLaMA EVA-CLIP SD-1.5 Causal 2023-09 DreamLLM [34] b-2 LLaMA OpenAI-CLIP SD-2.1 Causal 2023-09

Emu2 [33] b-2 LLaMA EVA-CLIP SDXL Causal 2023-12

VL-GPT [35] b-2 LLaMA OpenAI-CLIP IP-Adapter Causal 2023-12 MM-Interleaved [147] b-2 Vicuna OpenAI-CLIP SD-v2.1 Causal 2024-01

Mini-Gemini [148] b-2 Gemma&Vicuna OpenAI-CLIP+ConvNext SDXL Causal 2024-03 VILA-U [149] b-2 LLaMA-2 SigLIP+RQ RQ-VAE Causal 2024-09

PUMA [150] b-2 LLaMA-3 OpenAI-CLIP SDXL Bidirect. 2024-10 MetaMorph [151] b-2 LLaMA SigLIP SD-1.5 Causal 2024-12

ILLUME [152] b-2 Vicuna UNIT SDXL Causal 2024-12 UniTok [153] b-2 LLaMa-2 ViTamin ViTamin Causal 2025-02 QLIP [154] b-2 LlaMa-3 QLIP-ViT+BSQ BSQ-AE Causal 2025-02 DualToken [155] b-2 Qwen2.5 SigLIP RQVAE Causal 2025-03

UniFork [156] b-2 Qwen2.5 SigLIP+RQ RQ-VAE Causal 2025-06 UniCode2 [157] b-2 Qwen2.5 SigLIP+RQ FLUX.1-dev / SD-1.5 Causal 2025-06 UniWorld [158] b-2 Qwen2.5-VL SigLIP2 DiT Bidrect. 2025-06

Pisces [159] b-2 LLaMA-3.1 SigLIP EVA-CLIP Diffusion Causal 2025-06 Tar [160] b-2 Qwen2.5 SigLIP2+VQ VQGAN / SANA Causal 2025-06

OmniGen2 [161] b-2 Qwen2.5-VL SigLIP OmniGen Causal 2025-06 Ovis-U1 [162] b-2 Ovis AimV2 MMDiT Causal 2025-06 X-Omni [163] b-2 Qwen2.5-VL QwenViT Siglip FLUX Causal 2025-07

Qwen-Image [164] b-2 Qwen2.5-VL QwenViT MMDiT Causal 2025-08 Bifrost-1 [165] b-2 Qwen2.5-VL QwenViT ViT FLUX Causal 2025-08 Ming-UniVision [166] b-2 Ming-UniVision MingTok MingTok Causal 2025-10 MammothModa2 [167] b-2 Qwen3-VL-8B QwenViT MammothTok Single-stream DiT Causal 2025-11

SEED [168] b-3 OPT SEED Tokenizer Learnable Query SD Causal 2023-07 SEED-LLaMA [169] b-3 LLaMa-2 &Vicuna SEED Tokenizer Learnable Query unCLIP-SD Causal 2023-10

SEED-X [170] b-3 LLaMa-2 SEED Tokenizer Learnable Query SDXL Causal 2024-04 MetaQueries [171] b-3 LLaVA&Qwen2.5-VL SigLIP Learnable Query Sana Causal 2025-04

Nexus-Gen [172] b-3 Qwen2.5-VL QwenViT Learnable Query FLUX Causal 2025-04 Ming-Lite-Uni [173] b-3 M2-omni NaViT Learnable Query Sana Causal 2025-05

BLIP3-o [174] b-3 Qwen2.5-VL OpenAI-CLIP Learnable Query Lumina-Next Causal 2025-05 OpenUni [175] b-3 InternVL3 InternViT Learnable Query Sana Causal 2025-05

UniLIP [176] b-3 InternVL3 InternViT Learnable Query Sana Causal 2025-07 TBAC-UniImage [177] b-3 Qwen2.5-VL QwenViT Learnable Query Sana Causal 2025-08 UniPic 2.0 [178] b-3 Qwen2.5-VL QwenViT Learnable Query SD3.5-Medium Causal 2025-09

Janus [179] b-4 DeepSeek-LLM SigLIP VQGAN VQGAN Casual 2024-10 Janus-Pro [180] b-4 DeepSeek-LLM SigLIP VQGAN VQGAN Casual 2025-01

OmniMamba [181] b-4 Mamba-2 DINO-v2+SigLIP VQGAN VQGAN Causal 2025-03 Unifluid [182] b-4 Gemma-2 SigLIP SD-VAE Diffusion MLP Causal 2025-03 MindOmni [183] b-4 Qwen2.5-VL QwenViT VAE OmniGen Causal 2025-06

Skywork UniPic [184] b-4 Qwen2.5 SigLIP2 SDXL-VAE SDXL-VAE Causal 2025-08 MUSE-VL [185] b-5 Qwen-2.5&Yi-1.5 SigLIP VQGAN VQGAN Causal 2024-11 Tokenflow [186] b-5 Vicuna&Qwen-2.5 OpenAI-CLIP MSVQ MSVQ Causal 2024-12

VARGPT [187] b-5 Vicuna-1.5 OpenAI-CLIP MSVQ VAR-d30 Causal 2025-01 SemHiTok [188] b-5 Qwen2.5 SigLIP ViT ViT Causal 2025-03

VARGPT-1.1 [189] b-5 Qwen2 SigLIP MSVQ Infinity Causal 2025-04 ILLUME+ [190] b-5 Qwen2.5 QwenViT MoVQGAN SDXL Causal 2025-04 UniToken [191] b-5 Chameleon SigLIP VQ-IMG VQGAN Causal 2025-04

Show-o2 [192] b-5 Qwen2.5 Wan-3DVAE + SigLIP Wan-3DVAE Wan-3DVAE Causal 2025-06 Fused Autoregressive and Diffusion Model

|Transfusion [38] Show-o [39] MonoFormer [37] LMFusion [193] TUNA [194]|c-1 c-1 c-1 c-1 c-1|LLaMA-2 LLaVA-v1.5-Phi TinyLLaMA LLaMA Qwen-2.5<br><br>|SD-VAE MAGVIT-v2 SD-VAE SD-VAE+UNet down. VAE+Siglip2|SD-VAE MAGVIT-v2 SD-VAE SD-VAE+UNet up. VAE|Bidirect. Bidirect. Bidirect. Bidirect. Bidirect.|2024-08<br><br>2024-08<br>2024-09<br><br><br>2024-12<br>2025-12<br>|
|---|---|---|---|---|---|---|
|Janus-flow [195] Mogao [196] BAGEL [197] LightFusion [198] HBridge [199] EMMA [200]|c-2 c-2 c-2 c-2 c-2 c-2|DeepSeek-LLM Qwen2.5 Qwen2.5 QWen2.5-VL +Wan2.2-TI2V QWen + OmniGen2 Qwen3-34B|SigLIP SDXL-VAE SigLIP+SDXL-VAE SDXL-VAE SigLIP FLUX-VAE<br><br>QWen2.5-VL Wan2.2-TI2V<br><br>QwenViT SigLip SigLIP DCAE|SDXL-VAE SDXL-VAE FLUX-VAE DCAE OmniGen DCAE|Causal Bidirect. Bidirect. Bidirect. Bidirect. Bidirect.|2024-11<br>2025-05 2025-05 2025-11<br><br><br>2025-11<br>2025-12<br>|

Pixel Encoding

Pixel Encoding Semantic Encoding

Learnable Query Encoding

Text Tokenizer

Pixel Encoder

Text Tokenizer

Pixel Encoder

Text Tokenizer

Semantic Encoder

Semantic Encoder

Text Tokenizer

Learnable Query

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

Diffusion

MLLM (AR)

MLLM (AR)

MLLM (AR)

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

Pixel/Diffusion Decoder

Pixel/Diffusion Decoder

Pixel Decoder

Diffusion Decoder

(a) (b-1)

(b-3) Hybrid Encoding

(b-2)

Pixel Encoding Hybrid Encoding

Text Tokenizer

Semantic Encoder

Pixel Encoder Text Tokenizer

Semantic Encoder

Pixel Encoder

Text Tokenizer

Pixel Encoder

Text Tokenizer

Semantic Encoder

Pixel Encoder

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

MLLM (AR)

MLLM (AR)

MLLM (AR+Diffusion)

MLLM (AR+Diffusion)

| | | | |
|---|---|---|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

Pixel/Diffusion Decoder

Pixel/Diffusion Decoder

Pixel Decoder

Pixel Decoder

(b-4) (b-5)

(c-1) (c-2)

Fig. 5. Classification of Unified Multimodal Understanding and Generation Models. The models are divided into three main categories based on their backbone architecture: Diffusion, MLLM (AR), and MLLM (AR + Diffusion). Each category is further subdivided according to the encoding strategy employed, including Pixel Encoding, Semantic Encoding, Learnable Query Encoding, and Hybrid Encoding. We illustrate the architectural variations within these categories and their corresponding encoder-decoder configurations.

TABLE 2 Overview of Any-to-Any Multimodal Models Supporting Modal Input/Output Beyond Image and Text. This table categorizes models that support a variety of input and output modalities, including audio, music, image, video, and text. It includes information on the model’s backbone architecture, modality encoders and decoders, the type of attention mask used in vision generation, and the model release dates. These models exemplify the shift toward broader multimodal interactions in recent years.

|Model|Architecture Backbone Modality Enc. Modality Dec. Mask<br><br>| | |Date|
|---|---|---|---|---|
| | | | | |
|Next-GPT [201] Unified-IO 2 [202] Video-LaVIT [203] AnyGPT [204] X-VILA [205] MIO [206]<br><br>Spider [207] OmniFlow [208] M2-omni [209]|Vicuna T5 LLaVA-1.5 LLaMA-2 Vicuna Yi-Base<br><br>LLaMA-2 MMDiT<br>LLaMA-3<br>|ImageBind AudioLDM+SD-1.5+Zeroscope-v2 Audio Spectrogram Transformer+Vision ViT Audio ViT-VQGAN + Vision VQGAN LaVIT+Motion VQ-VAE SVD img2vid-xt Encodec+SEED Tokenizer+SpeechTokenizer Encodec+SD+SoundStorm<br><br>ImageBind AudioLDM+SD-1.5+Zeroscope-v2 SpeechTokenizer+SEED-Tokenizer SpeechTokenizer+SEED Tokenizer<br><br>ImageBind<br><br>AudioLDM+SD-1.5+Zeroscope-v2 +Grounding DINO+SAM<br><br>HiFiGen+SD-VAE+Flan-T5 HiFiGen+SD-VAE+TinyLlama paraformer-zh+NaViT CosyVoice-vocoder+SD-3|Causal Causal Causal Causal Causal Causal<br><br>Causal Bidirect. Casual|2023-09<br><br>2023-12<br>2024-02<br><br><br>2024-02 2024-05<br><br><br>2024-09<br><br>2024-11<br>2024-12<br><br><br>2025-02<br>|

In this section, we primarily focus on unified multimodal models that support vision-language understanding and generation, i.e., models that take both image and text as input and produce either text or image as output. As shown in Fig. 5, existing unified models can be broadly categorized into three main types: diffusion models, autoregressive models, and fused AR + diffusion models. For autoregressive models, we further classify them based on their modality encoding methods into four subcategories: pixel-based encoding, semantic-based encoding, learnable query-based encoding, and hybrid encoding. Each of these encoding strategies represents different ways of handling visual and textual data, leading to varying levels of integration and flexibility in the multimodal representations. Fused AR + diffusion models are divided into two subcategories based on modality encoding: pixel-based encoding and hy-

brid encoding. These models combine aspects of both autoregressive and diffusion techniques, offering a promising approach to more unified and efficient multimodal generation.

In the following sections, we will delve deeper into each category: Section 3.1 explores diffusion-based models, discussing their unique advantages in terms of generating high-quality images and text from noisy representations. Section 3.2 focuses on autoregressive-based models, detailing how different encoding methods impact their performance in vision-language tasks. Section 3.3 covers fused AR + diffusion models, examining how the combination of these two paradigms can enhance multimodal generation capabilities. Finally, we extend our discussion to any-toany multimodal models, which generalize this framework beyond vision and language to support a broader range of

modalities such as audio, video, and speech, with the aim of building universal, general-purpose generative models.

#### 3.1 Diffusion Models

Diffusion models have achieved remarkable success in the field of image generation owing to several key advantages. First, they provide superior sample quality compared to generative adversarial networks (GANs), offering better mode coverage and mitigating common issues such as mode collapse and training instability [210]. Second, the training objective—predicting the added noise from slightly perturbed data—is a simple supervised learning task that avoids adversarial dynamics. Third, diffusion models are highly flexible, allowing the incorporation of various conditioning signals during sampling, such as classifier guidance [210] and classifier-free guidance [211], which enhances controllability and generation fidelity. Furthermore, improvements in noise schedules [212] and accelerated sampling techniques [213], [214] have significantly reduced the computational burden, making diffusion models increasingly efficient and scalable.

Leveraging these strengths, researchers have extended diffusion models beyond unimodal tasks toward multimodal generation, aiming to support both text and image outputs within a unified framework. As shown in Fig. 5 (a), in multimodal diffusion models, the denoising process is conditioned not only on timestep and noise but also on multimodal contexts, such as textual descriptions, images, or joint embeddings. This extension enables synchronized generation across different modalities and allows for rich semantic alignment between generated outputs.

A representative example is Dual Diffusion [127], which introduces a dual-branch diffusion process for joint text and image generation. Specifically, given a text-image pair, Dual Diffusion first encodes the text using a pretrained T5 encoder [23] with softmax probability modeling to obtain discrete text representations, and encodes the image using the VAE encoder from Stable Diffusion [14] to obtain continuous image latents. Both text and image latents are independently noised through separate forward diffusion processes, resulting in noisy latent variables at each timestep. During the reverse process, the model jointly denoises the text and image latents using two modality-specific denoisers: a Transformer-based text denoiser and a UNet-based image denoiser. Crucially, at each timestep, the denoisers incorporate cross-modal conditioning, where the text latent attends to the image latent and vice versa, enabling semantic alignment between the modalities throughout the denoising trajectory. After denoising, the text latent is decoded into natural language via a T5 decoder, and the image latent is decoded into a high-fidelity image via the VAE decoder. Training is supervised by two distinct loss terms: the image branch minimizes a standard noise prediction loss, while the text branch minimizes a contrastive log-loss. By coupling the two diffusion chains and introducing explicit crossmodal interactions, Dual Diffusion enables coherent and controllable multimodal generation from pure noise.

Unlike Dual Diffusion [127], which combines discrete text diffusion with continuous image diffusion via Stable Diffusion [14], UniDisc [128] employs a fully discrete diffusion framework to train a Diffusion Transformer [215] from

scratch. It tokenizes text using the LLaMA2 tokenizer [2] and converts images into discrete tokens with the MAGVITv2 encoder [216], allowing unification of both modalities in a discrete token space. These tokens undergo a discrete forward diffusion process, where structured noise is added simultaneously across modalities. In the reverse process, UniDisc progressively denoises the tokens to generate coherent sequences. The LLaMA2 and MAGVIT-v2 decoders then transform these sequences into high-quality text and images. By adopting a fully discrete approach, UniDisc enables simultaneous refinement of text and image tokens, enhancing inference efficiency and supporting versatile crossmodal conditioning.

In contrast to earlier discrete diffusion-based methods, FUDOKI [130] introduces a novel generative approach based on a discrete flow matching [217]. Under this framework, FUDOKI models a direct path between noise and data distributions by employing a kinetic-optimal, metricinduced probability trajectory. This design enables a continuous self-correction mechanism, which provides a clear advantage over the simple masking strategies used in earlier models. FUDOKI’s model architecture is based on Janus1.5B [179]. However, it introduces essential modifications to support unified vision-language discrete flow modeling. One key change is the replacement of the standard causal mask with a full attention mask. This allows every token to attend to all others, thereby enhancing global contextual understanding. Although this modification removes the explicit causal structure, the model still supports next-token prediction by shifting its output logits by one position. Another important distinction is in the way FUDOKI handles time or corruption levels. Instead of relying on explicit timestep embeddings, as required in diffusion models, FUDOKI infers the corruption state directly from the input data. Following Janus-1.5B, FUDOKI decouples the processing paths for understanding and generation. A SigLIP encoder [218] is employed to capture high-level semantic features for image understanding, while a VQGAN-based tokenizer from LlamaGen [24] encodes the image into a sequence of low-level discrete tokens for image generation. At the output stage, the feature embeddings generated by the Janus1.5B backbone are passed through modality-specific output heads to produce the final text and image outputs.

In a similar vein, Muddit [131] introduces a unified model for bidirectional generation using a purely discrete diffusion framework to handle text and images. Its architecture features a single Multimodal Diffusion Transformer (MM-DiT) with an architectural design similar to that of FLUX [219]. To leverage a strong image prior, the MM-DiT generator is initialized from Meissonic [220], a model extensively trained for high-resolution synthesis. Both modalities are quantized into a shared discrete space, where a pre-trained VQ-VAE [32] encodes images into codebook indices and a CLIP model [22] provides text token embeddings. During its unified training, Muddit employs a cosine scheduling strategy to mask tokens, and the single MM-DiT generator is trained to predict the clean tokens conditioned on the other modality. For output, a lightweight linear head decodes text tokens, while the VQ-VAE decoder reconstructs the image, allowing a single set of parameters to handle both text and image generation.

Building upon this foundation, MMaDA [129] scales up the diffusion paradigm toward a unified multimodal foundation model. It adopts LLaDA-8B-Instruct [221] as the language backbone and uses a MAGVIT-v2 [222] image tokenizer to convert images into discrete semantic tokens. This unified token space enables seamless multimodal conditioning during generation. To improve alignment across modalities, MMaDA introduces a mixed chain-of-thought (CoT) fine-tuning strategy, which unifies reasoning formats between text and vision tasks. This alignment facilitates cold-start reinforcement learning, allowing effective post-training from the outset. Furthermore, MMaDA incorporates a novel UniGRPO method, a unified policygradient-based RL algorithm designed for diffusion models. UniGRPO enables post-training optimization across both reasoning and generation tasks by leveraging diversified reward signals, such as factual correctness, visual-textual alignment, and user preferences. This design ensures the model consistently improves across a broad range of capabilities, rather than overfitting to a narrow task-specific reward. Built on LaViDa [223], Lavida-O [132] bridges the gap between unified multi-modal diffusion models and state-of-the-art AR models by implementing a resourceefficient Elastic Mixture-of-Transformers (Elastic-MoT) architecture. To overcome the scarcity of effective masked generative training, it employs progressive upscaling and token compression for robust model scaling. Furthermore, it distinguishes itself through planning and self-reflection mechanisms, which allow the model to use its understanding capability to actively refine and improve generation outputs.

Building upon the insight from DeepSeek-OCR [224] that text can be processed as a visual signal, UniModel [133] unifies these tasks by rendering text into images and employing a diffusion model for both understanding and generation. While this represents a novel strategy, significant challenges remain, including text rendering fidelity, multilingual generation capabilities, and long-context modeling.

Despite these innovative approaches, significant challenges and limitations persist in the landscape of unified discrete diffusion models. A primary concern is inference efficiency. Although models like Mercury [225] and Gemini Diffusion [226] demonstrate potential for high-speed parallel token generation, most open-source discrete diffusion models still lag behind the practical inference speeds of their autoregressive counterparts. This discrepancy is primarily due to a lack of support for key-value cache and the degradation in output quality that occurs when decoding multiple tokens in parallel. The effectiveness of diffusion models is also hindered by training difficulties. Unlike autoregressive training, where every token provides a learning signal, discrete diffusion training offers only sparse supervision, as the loss is computed on a randomly selected subset of masked tokens, leading to inefficient use of the training corpus and high variance. Moreover, these models exhibit a length bias and struggle to generalize across different output lengths because they lack a built-in stopping mechanism like the end-of-sequence token found in autoregressive models. Additional development is also needed in architecture and supporting infrastructure. Architecturally, many existing models reuse designs originally created for autoregressive

systems, an approach chosen for engineering simplicity that is not always suited to the diffusion process, which aims to capture joint data distributions in a way that is fundamentally different from the sequential nature of autoregressive models. On the infrastructure side, support for discrete diffusion models remains limited. Compared to the mature frameworks available for autoregressive models, they lack well-developed pipelines and robust open-source options. This gap hinders fair comparisons, slows research, and complicates real-world deployment. Addressing these interconnected challenges in inference, training, architecture, and infrastructure is essential to advance the capabilities and practical use of unified discrete diffusion models.

#### 3.2 Auto-Regressive Models

One major direction in unified multimodal understanding and generation models adopts autoregressive (AR) architectures, where both vision and language tokens are typically serialized and modeled sequentially. In these models, a backbone Transformer, typically adapted from large language models (LLMs) such as LLaMA family [1], [2], [227], Vicuna [58], Gemma series [228], [229], [230], and Qwen series [5], [6], [9], [10], serves as the unified modality-fusion module to autoregressively predict multimodal outputs.

To integrate visual information into the AR framework, as shown in Fig. 5, existing methods propose different strategies for image tokenization during modality encoding. These approaches can be broadly categorized into four types: pixel-based, semantic-based, learnable query-based, hybrid-based encoding methods.

1) Pixel-based Encoding. As shown in Fig. 5 (b-1), pixelbased encoding typically refers to the representation of images as continuous or discrete tokens obtained from pretrained autoencoders supervised purely by image reconstruction, such as VQGAN-like models [32], [231], [232], [233]. These encoders compress the high-dimensional pixel space into a compact latent space, where each spatial patch corresponds to an image token. In unified multimodal autoregressive models, image tokens serialized from such encoders are processed analogously to text tokens, allowing both modalities to be modeled within a single sequence.

Recent works have adopted and enhanced pixel-based tokenization with various encoder designs. LWM [29] employs a VQGAN tokenizer [32] to encode images into discrete latent codes without requiring semantic supervision. It proposes a multimodal world modeling framework, wherein visual and textual tokens are serialized together for unified autoregressive modeling. By learning world dynamics purely through reconstruction-based visual tokens and textual descriptions, LWM demonstrates that largescale multimodal generation is feasible without specialized semantic tokenization. Both Chameleon [30] and ANOLE [134] adopt VQ-IMG [233], an improved VQ-VAE variant designed for content-rich image generation. Compared to standard VQGAN tokenizers, VQ-IMG features a deeper encoder with larger receptive fields and incorporates residual prediction to better preserve complex visual details. This enhancement enables Chameleon and ANOLE to serialize image content more faithfully, thereby supporting high-quality multimodal generation. Moreover, these models facilitate

interleaved generation, allowing text and image tokens to be generated alternately within a unified autoregressive framework. Emu3 [135], SynerGen-VL [138] and UGen [140] employs SBER-MoVQGAN [231], [232], a multi-scale VQGAN variant that encodes images into latent representations capturing both global structure and fine-grained details. By leveraging multi-scale tokenization, these models improve the expressiveness of visual representations for autoregressive modeling while maintaining efficient training throughput. EMU3.5 [234] develops Discrete Diffusion Adaptation to accelerate inference via bidirectional parallel prediction. Similar with LWM [29], Liquid [139] utilizes a VQGANstyle tokenizer and uncovers a novel insight that visual understanding and generation can mutually benefit when unified under a single autoregressive objective and shared visual token representation. Moreover, MMAR [136], Orthus [137], Harmon [141] introduce the frameworks that utilize continuous-valued image tokens extracted by their corresponding encoders, avoiding the information loss associated with discretization. They also decouple the diffusion process from the AR backbone by employing lightweight diffusion heads atop each auto-regressed image patch embedding. This design ensures that the backbone’s hidden representations are not confined to the final denoising step, facilitating better image understanding. TokLIP [142] integrates a low-level discrete VQGAN tokenizer with a ViT-based token encoder SigLIP [218] to capture high-level continuous semantics, which not only empowers visual tokens with high-level semantic understanding but also enhances lowlevel generative capacity. Selftok [143] introduces a novel discrete visual self-consistency tokenizer, achieving a favorable trade-off between high-quality reconstruction and compression rate while enabling optimal policy improvement for effective visual reinforcement learning. OneCat [144] employs a patch embedding layer to map raw images into continuous tokens for understanding and editing, alongside a pre-trained multi-scale VAE [235] for reconstruction. Notably, it adopts a hybrid generation strategy: standard nexttoken prediction for text and next-scale prediction [235] for image synthesis.

Across most of these models, causal attention masks are applied during both pretraining and generation phases, ensuring that each token only attends to preceding tokens in the sequence. They are trained using a next-token prediction loss, where both image and text tokens are predicted autoregressively, thus unifying the training objective across modalities. Notably, in pixel-based encoding approaches, the decoder used to reconstruct images from latent tokens typically follows the paired decoder structure originally proposed in VQGAN-like models. These decoders are lightweight convolutional architectures specifically optimized to map discrete latent grids back to the pixel space, focusing primarily on accurate low-level reconstruction rather than high-level semantic reasoning. Moreover, since some methods, like MMAR [136], Orthus [137] and Harmon [141], tokenize the image into continuous latents, they adopt the lightweight diffusion MLP as their decoder to map continuous latents back to the pixel space.

Despite their effectiveness, pixel-based encoding methods face several inherent limitations: First, since the visual tokens are optimized purely for pixel-level reconstruction,

they often lack high-level semantic abstraction, making cross-modal alignment between text and image representations more challenging. Second, pixel-based tokenization tends to produce dense token grids, significantly increasing sequence lengths compared to text-only models, especially for high-resolution images. This leads to substantial computational and memory overhead during autoregressive training and inference, limiting scalability. Third, because the underlying visual encoders are trained with reconstructioncentric objectives, the resulting visual tokens may retain modality-specific biases, such as excessive sensitivity to textures and low-level patterns, which are not necessarily optimal for semantic understanding or fine-grained crossmodal reasoning.

2) Semantic Encoding. To overcome the semantic limitations inherent in pixel-based encoders, a growing body of work adopts semantic encoding, where image inputs are processed using pretrained text-aligned vision encoders such as OpenAI-CLIP [22], SigLIP [218], EVA-CLIP [36], or more recent unified tokenizers like UNIT [236], as shown in Fig. 5 (b-2). Some of these models leverage the multimodal features encoded by the multimodal autoregressive model as conditions for a diffusion model, enabling image generation with retained multimodal understanding capabilities, like OmniGen2 [161] that utilizes Qwen2.5-VL [10] as the multimodal model and enhanced OmniGen [237] as the image diffusion model, Ovis-U1 [162] extending the multimodal model Ovis [12] into a unified model by incorporating a custom-designed diffusion transformer while Qwen-Image [164] similarly building upon Qwen2.5-VL [10] by integrating a diffusion transformer. UniVideo [238] and Omni-Video [239] extend this paradigm to the video modality. However, most of these models are trained on large-scale image-text pairs with contrastive or regression-based objectives, producing visual embeddings that align closely with language features in a shared semantic space. Such representations enable more effective cross-modal alignment and are particularly beneficial for multimodal understanding and generation.

Several representative models leverage different semantic encoders and architectural designs to support unified multimodal tasks. Emu [145], Emu2 [33], and LaViT [146] all employ EVA-CLIP [36] as their vision encoder. Notably, Emu [145] introduces the initial architecture combining a frozen EVA-CLIP encoder, a large language model, and a diffusion decoder to unify VQA, image captioning, and image generation. Emu2 [33] builds upon Emu [145] by proposing a simplified and scalable modeling framework for unified multimodal pretraining. It scales the MLLM model up to 37B parameters, significantly enhancing both understanding and generation capabilities. Bifrost-1 [165] employs two semantic encoders, ViT for generation and the one employed in the used MLLM (QWen2.5-VL) for understanding. The predicted CLIP latents are used to bridge the MLLM and diffusion model. LaViT [146] introduces a dynamic visual tokenization mechanism built on top of EVACLIP. It employs a selector and merger module to adaptively select visual tokens from image embeddings based on content complexity. This process dynamically determines the length of the visual token sequence per image. The dynamic tokenization significantly reduces redundant information

while preserving important visual cues, improving training efficiency and generation quality in tasks such as captioning, visual question answering, and image generation. DreamLLM [34], VL-GPT [35], and MM-Interleaved [147], and PUMA [150] utilize OpenAI-CLIP encoder [22]. DreamLLM [34] introduces a lightweight linear projection to align CLIP embeddings with language tokens, while VL-GPT [35] employs a powerful casual transformer after OpenAI-CLIP vision encoder to effectively retain both semantic information and pixel details of the original image. Both MM-Interleaved [147] and PUMA [150] extract multi-granular image features via a CLIP tokenizer with simple ViT-Adapter or pooling operation to provide fine-grained feature fusion, thus supporting rich multimodal generation. Mini-Gemini [148] introduces a visual token enhancement mechanism that requires dual semantic encoders. Specifically, it leverages a CLIP-pretrained ViT encoder [22] to obtain global visual tokens, while a LAION-pretrained ConvNeXt encoder provides dense local visual information. A cross-attention module is then employed to refine the global visual tokens by incorporating detailed visual cues from the dense encoder. These enhanced global tokens are subsequently combined with text tokens and processed by an LLM for joint vision-language understanding and generation. This design effectively bridges the semantic abstraction of CLIP features with the pixel-level precision of dense encoders. MetaMorph [151] employs SigLIP [218] to extract visual embeddings and introduces modality-specific adapters within a pretrained language model. These adapters are inserted throughout multiple transformer layers, allowing for deeper vision-language interaction compared to shallow projection approaches. ILLUME [152] adopt UNIT [236] as its vision encoder to provide a unified representation that balances semantic alignment and pixel-level fidelity. Unlike CLIP-like encoders that focus purely on contrastive objectives, UNIT [236] is jointly trained with both image reconstruction and contrastive alignment losses, producing tokens suitable for both vision-language understanding and image synthesis. Built on the powerful UNIT tokenizer, ILLUME effectively generates image tokens that retain both semantic and pixellevel information, which achieves better performance in multiple understanding and generation tasks, including captioning, VQA, text-to-image, and interleaved generation. Similarly, VILA-U [149] and UniTok [153] mimic UNIT [236] to introduce image-text contrastive learning to obtain a novel text-aligned vision tokenizer that balances semantic alignment and pixel-level fidelity. QLIP [154] addresses the potential conflict between reconstruction and text-image alignment tasks by implementing binary-spherical quantization. Tar [160] initiates the visual codebook by leveraging the vocabulary of LLMs and incorporates scale-adaptive pooling and decoding methodologies. This approach enables the model to adjust the length of the tokenizer according to the requirement: employing coarse-grained tokenizers for efficient generation and fine-grained tokenizers for comprehensive understanding. In generation tasks, Tar utilizes diffusion techniques to enhance the visual generation outcomes of AR models. UniFork [156] capitalizes on the text-aligned vision features of VILA-U. However, differentiating itself from the fully-shared parameters of understanding and generation MLLM, UniFork shares the parameters

solely with these tasks at the shallow layer. At the deeper layer, these tasks are managed by distinct networks. This architecture successfully mediates the equilibrium between shared learning and task-specific specialization. UniCode2 [157] employs a cascaded codebook. In line with the method outlined in [240], it utilizes a substantial codebook derived from clustered SigLIP feature as the frozen foundational codebook, while introducing supplementary learnable codebooks to refine semantics specific to particular tasks. This separation enhances utilization and fosters robust learning. Recent work DualToken [155] uses shallow-layer features of SigLIP for reconstruction and deep-layer features of SigLIP for semantic learning, thereby obtaining the texture and semantic visual features simultaneously. As a result, DualToken [155] achieves superior performance in both reconstruction and semantic tasks while demonstrating remarkable effectiveness in downstream MLLM understanding and generation tasks. X-Omni [163] utilizes SigLIP-VQ as a visual encoder and employs reinforcement learning to mitigate the cumulative error associated with autoregressive inference and to reduce the information loss inherent in discrete encoding. This methodology substantially enhances the generation quality of discrete autoregressive models, facilitating a seamless integration of image and language generation. Ming-UniVision [166] introduces MingTok, a tokenizer comprising three modules: a low-level encoder, a semantic decoder, and a pixel decoder. The low-level encoder projects the input image into a compact latent space, which is subsequently processed by the semantic decoder for understanding or editing tasks. For generation, predicted tokens within this latent space are reconstructed into raw images via the pixel decoder. MammothModa2 [167] introduces MammothTok, a unified visual tokenizer built upon AIMv2 for generation, while utilizing QwenViT for understanding. Notably, it leverages multi-layer features from the understanding component to condition the generation process. Recently, RAE [241] demonstrated that semantic encoders can also be well adapted to diffusion models, and thus using semantic encoders as visual tokenizers for unified models might become the future direction. VQRAE [242] has demonstrated that discretized SigLIP features are effective for both understanding and generation tasks. However, the potential for a fully unified framework based on these features remains underexplored..

Across most of these models, causal attention masks are applied during MLLM training, and next-token prediction loss is used to optimize both text and vision token generation. For image generation, most of these models typically employ diffusion-based decoders, such as SD family [14], [243], IP-adapter [244], FLUX [16], and Lumina-Next [245], which are trained independently from the MLLM. During inference, the MLLM produces semantic-level visual tokens, which are then passed to the diffusion decoder for final image synthesis. This design choice—pairing semantic encoders with diffusion decoders—is motivated by the fact that semantic embeddings encode high-level conceptual information but lack the spatial density and lowlevel granularity required for direct pixel reconstruction. Diffusion models, with their iterative denoising mechanisms, are particularly well-suited for this setting: they are capable of progressively refining semantic representations

into high-resolution, photorealistic images, even when the input tokens are sparse or abstract. In contrast, although few approaches (i.e., VILA-U [149], UniTok [153], and VQRAE [242]) adopt pixel-based decoders, their generated image quality is less competitive than the diffusion decoders. Thus, diffusion decoders provide a more robust and expressive decoding pathway for semantically compressed visual tokens, significantly improving text-image alignment, global coherence, and visual fidelity. UniWorld [158] and Pisces [159] have endeavored to develop and expand such a solution. UniWorld directly utilizes the output features of pre-trained MLLM for visual comprehension as a high-level conditional signal, while employing SigLIP as low-level conditional signal to deliver comprehensive semantic visual control for DiT. Pisces employs EVA-CLIP as a condition for visual generation tasks and leverages diffusion to further enhance the model’s visual generation output. For various tasks, Pisces introduces tailored visual vector lengths and employs distinct MLPs to encode conditions. This approach increases the flexibility of model design while mitigating the inference cost compared to a single encoder configuration.

Despite these advantages, semantic encoding also comes with several limitations. First, due to the abstraction of low-level cues, the resulting visual tokens are less controllable at the pixel level, making it difficult to perform fine-grained image editing, local inpainting, or structurepreserving transformation. Second, semantic encoders often provide only global or mid-level representations, which can be insufficient for tasks requiring spatial correspondence (e.g., referring expression segmentation or pose-accurate synthesis). Lastly, since the semantic encoder and diffusion decoder are typically trained separately, the lack of endto-end optimization can lead to mismatch between MLLM outputs and decoder expectations, occasionally causing semantic drift or generation artifacts.

3) Learnable Query Encoding. Learnable query encoding has emerged as an effective strategy for producing adaptive and task-relevant image representations. As shown in Fig. 5 (b-3), instead of relying purely on fixed visual tokenizers or dense image patches, this approach introduces a set of learnable query tokens that dynamically extract informative content from image features. These query tokens act as content-aware probes that interact with visual encoders to generate compact and semantically aligned embeddings, well-suited for multimodal understanding and generation.

Current implementations of learnable query encoding can be broadly divided into two representative paradigms. The first is represented by SEED [168], which proposes a seed tokenizer that learns causal visual embeddings. Specifically, an input image is first encoded into dense token features via a BLIP-2 ViT encoder [53]. These features are then concatenated with a set of learnable query tokens and processed by a causal Q-Former to produce causal visual embeddings. This design is trained using both image-text contrastive learning and image reconstruction supervision, allowing the learned embeddings to simultaneously retain low-level visual detail and capture high-level semantic alignment with text. Building on this foundation, SEEDLLAMA [169] and SEED-X [170] enhance the model’s capacity by replacing the OPT backbone [246] with a stronger LLaMA2 model [2] and upgrading the decoder to UnCLIP-

SD [14] or SDXL [243], leading to improved performance in both understanding and generation tasks. The second approach, introduced by MetaQueries [171], provides a simplified version of learnable query encoding. Here, image features are extracted via a frozen SigLIP encoder [218], which are then concatenated with learnable query tokens and directly passed through a frozen vision-language backbone such as LLaVA [227] or Qwen2.5-VL [10]. The output causal embeddings are used as conditioning inputs for a diffusion-based image decoder, enabling high-quality image generation. Because the backbone is kept frozen, the visionlanguage understanding capabilities remain consistent with the underlying pretrained models, offering a lightweight yet effective solution for multimodal generation. OpenUni [175] refines the architecture of MetaQueries by utilizing solely learnable queries and a lightweight connector between a MLLM and a diffusion model, facilitating cohesive multimodal understanding and generation. OpenUni demonstrates that the connetor between the MLLM visual understanding component and the diffusion-based visual generation component can be minimal in complexity, exemplified by a configuration comprising merely six Transformer layers. Nexus-Gen [172] and Ming-Lite-Uni [173] follow the MetaQueries paradigm, but with notable advancements to further enhance multimodal generation. NexusGen [172] introduces a more powerful diffusion decoder, FLUX-1.dev, which significantly improves the generation quality. This approach allows the model to better capture the intricate details and high-fidelity features necessary for complex image generation tasks. On the other hand, MingLite-Uni [173] takes a different route by introducing a highly capable MLLM model, M2-omini [209], for enhanced visionlanguage interaction. This model performs advanced visionlanguage conditioning to generate the conditioned image embeddings, ensuring a more semantically aligned representation. In addition, Ming-Lite-Uni fine-tunes its diffusion model by incorporating multi-scale learnable tokens, which facilitate improved semantic alignment across various visual scales. The multi-scale representation alignment mechanism enhances the model’s ability to generate detailed and contextually rich images from textual prompts, addressing challenges such as resolution mismatches and semantic inconsistencies. This innovative approach makes Ming-LiteUni a powerful tool for multimodal understanding and generation, pushing the boundaries of current methods in both flexibility and performance. UniLIP [176] incrementally incorporates reconstruction ability into CLIP through the self-distillation, then employs a learnable query along with the hidden state of the last layer of the MLLM as combined conditions. This framework is demonstrated to optimize the abundant information for visual editing. To exploit the hierarchical representations within the MLLM’s intermediate layers, TBAC-UniImage [177] applies learnable queries at multiple layers instead of the last layer. UniPic 2.0 [178] integrates SD3.5-Medium with Qwen2.5VL-7B via the MetaQuery strategy [171] and proposes a Progressive Dual-Task Reinforcement strategy for strengthening both tasks. To sum up, these learnable query-based designs share a common strength: they provide adaptive, compact, and semantically enriched representations that support both efficient image understanding and high-quality generation. By focusing on

task-driven token extraction, such models offer a flexible and extensible alternative to traditional visual tokenizers, especially in unified multimodal frameworks.

Despite its flexibility and promising results, learnable query encoding also comes with several limitations that may restrict its broader applicability. First, one key challenge is the increased computational overhead introduced by the learnable query tokens. As the number of query tokens grows, the model’s memory consumption and computational complexity can significantly rise, especially when scaling up to large datasets or more intricate multimodal tasks. Furthermore, the use of a fixed encoder (as seen in approaches like MetaQueries) can hinder the model’s flexibility when confronted with novel or complex visual inputs that diverge from the pretrained data distributions. Second, in methods like SEED [168] and MetaQueries [171], the reliance on frozen or pretrained backbones can limit the adaptability of visual features to downstream tasks. While freezing reduces training cost and preserves pre-learned knowledge, it also restricts the capacity of the model to dynamically align image features with the evolving query semantics, especially in more diverse or compositional settings. Finally, while learnable queries effectively capture task-relevant content, they may not always handle diverse visual content uniformly. For instance, complex scenes with multiple objects, fine-grained details, or ambiguous visual cues might not be as well-represented by a relatively small number of learnable queries. This limitation is particularly evident when the model must generate highly detailed outputs, as the fixed or small query set may fail to capture the richness and variability of the visual input in certain contexts.

4) Hybrid Encoding. To address the inherent limitations of using a single modality of visual representation, hybrid encoding strategies have been introduced in unified multimodal models. Pixel-based encoding methods (e.g., VQVAE or VQGAN) excel at preserving fine-grained visual details but often lack semantic alignment with text. In contrast, semantic-based encoders (e.g., SigLIP or CLIP variants) produce abstract representations that are semantically rich yet less effective at retaining low-level image fidelity. Hybrid encoding aims to combine the strengths of both approaches by incorporating both pixel-level and semanticlevel features into a unified representation. Depending on how pixel and semantic tokens are integrated, hybrid encoding methods can be broadly categorized into two types: pseudo hybrid encoding and joint hybrid encoding.

Pseudo Hybrid Encoding. Representative works in this category include Janus [179], Janus-Pro [180], OmniMamba [181], Unifluid [182], and MindOmni [183]. As shown in Fig. 5 (b-4), these models adopt dual encoders—typically a semantic encoder (e.g., SigLIP) and a pixel encoder (e.g., VQGAN or VAE)—but use them in a task-specific manner. During training, the semantic encoder branch is enabled for vision-language understanding tasks, while the pixel encoder branch is activated for image generation tasks. Although the dual encoders are trained concurrently with combined understanding and generation datasets, the pixel encoder is not utilized during inference in understanding tasks and the semantic encoder is disabled for text-to-image generation. However, for image editing, Unifluid [182] uses

the semantic encoder to encode the source image while MindOmni [183] utilizes both VAE and semantic encoder to encode the source image. The rationale behind this design choice is that mixed training with both types of data can enhance performance across understanding and generation tasks. Skywork UniPic [184] employs SigLIP2 as the encoder for understanding tasks and MAR [25] as the encoder for generative tasks. However, since only one encoder is active at any given time, these models do not fully harness the advantages of hybrid encoding. Specifically, they miss the opportunity to employ semantic grounding in generation tasks and fail to utilize high-fidelity visual details in comprehension tasks. Consequently, these models typically engage pixel decoders to reconstruct images from latent codes.

Joint Hybrid Encoding. As shown in Fig. 5 (b-5), joint hybrid encoding methods integrate both semantic and pixel tokens into a single unified input for the language model or decoder, enabling simultaneous utilization of both representations. These models differ in their fusion strategies. MUSE-VL [185] and UniToken [191] concatenates the features from SigLIP and VQGAN along the channel dimension before passing them into the LLM. Tokenflow [186] incorporate dual encoders and codebooks with a shared mapping, enabling the joint optimization of high-level semantics and low-level pixel details. VARGPT [187], VARGPT-1.1 [189], and ILLUME+ [190] concatenate the semantic and pixel tokens along the sequence dimension, maintaining both token types in the LLM’s input. SemHiTok [188] introduces the Semantic Guided Hierarchical Codebook (SGHC), which perfectly inherits the semantic information of the semantic codebook while incorporating texture information to achieve pixel reconstruction. It is significant to observe that, contrary to other methods that directly employ distinct network branches for image processing, Show-o2 [192] utilizes separate network branches for the processing of latent features generated by 3DVAE [247], and uses the spatial-temporal fusion module to aggregate the outputs of different branches. This approach enables Show-o2 to capture both low-level and high-level visual information. However, such an operation might result in the loss of subtle semantic elements, owing to Show-o2’s use of 3D VAE for lossy compression of images or videos, potentially causing suboptimal handling of visual semantic details. By integrating both semantic and detailed visual information, joint hybrid encoding enables more robust and expressive modeling capabilities for multimodal understanding and generation. These models support pixel decoders (e.g., VQGAN, Infinity [235], VAR-D30 [113]) as well as diffusion-based decoders (e.g., SDXL [243]), allowing them to generate images with improved semantic alignment and visual realism.

While hybrid encoding offers a promising direction by integrating the complementary strengths of pixel-level and semantic-level representations, it still faces several limitations. Many pseudo hybrid methods do not leverage both encoders simultaneously at inference time, thereby underutilizing the potential synergy between fine-grained visual details and high-level semantics. Even in joint hybrid approaches, the fusion of heterogeneous token types can introduce modality imbalance or redundancy, which may hinder downstream performance if not carefully managed. Additionally, the dual-encoder architecture substantially in-

creases computational and memory overhead, posing challenges for scalability, especially in high-resolution or longsequence scenarios. Aligning pixel and semantic tokens also remains a non-trivial problem, as implicit mismatches can lead to incoherent representations or conflicting learning signals. Finally, current hybrid encoding techniques often assumes implicit alignment between the pixel and semantic tokens. However, in practice, such alignment is non-trivial. Misalignment between visual details and semantic abstraction can lead to conflicting supervision signals or incoherent representations, especially in data-scarce or noisy training settings.

#### 3.3 Fused Autoregressive and Diffusion Models

Fused autoregressive (AR) and diffusion modeling has recently emerged as a powerful framework for unified visionlanguage generation. In this paradigm, text tokens are generated autoregressively, preserving the compositional reasoning strengths of large language models, while image tokens are generated through a multi-step denoising process, following the diffusion modeling principle. This hybrid strategy allows image generation to proceed in a non-sequential manner, resulting in improved visual quality and global consistency.

Representative models such as Transfusion [38], Showo [39], MonoFormer [37], and LMFusion [193], follow this approach. During generation, noise is added to latent visual representations and removed iteratively, with the process conditioned on previously generated text or full crossmodal context. Although this design increases inference cost due to multiple sampling steps, it achieves an effective trade-off between symbolic control and visual fidelity, making it well-suited for high-quality vision-language generation tasks. Existing fused AR + diffusion models typically adopt one of two image tokenization strategies: pixel-based encoding and hybrid encoding.

1) Pixel-based Encoding: As shown in Fig. 5 (c-1), pixelbased encoding transforms images into either discrete tokens or continuous latent vectors, which are then used

- as targets in a diffusion-based denoising process conditioned on autoregressively generated text tokens. Among recent works, Transfusion [38], MonoFormer [37], and LMFusion [193] all adopt continuous latent representations extracted via SD-VAE. These models share a common training objective that combines autoregressive loss for language modeling and diffusion loss for image reconstruction, and utilize bidirectional attention to enable spatial coherence. Despite this shared framework, each model introduces distinct architectural innovations: Transfusion [38] proposes a unified transformer backbone with modality-specific layers to jointly handle discrete and continuous inputs; MonoFormer [37] introduces a compact architecture with shared blocks and task-dependent attention masking to balance AR and diffusion tasks; and LMFusion [193] enables frozen LLMs to perform high-quality image generation through a lightweight visual injection module, preserving language capabilities while training only the vision branch. In contrast, Show-o [39] employs a discrete pixel-based tokenizer based on MAGVIT-v2 [222], generating symbolic image tokens compatible with transformer-style decoding. It sup-

ports both AR-based text token generation and diffusionbased image synthesis, supervised through a combination of autoregressive and diffusion losses. TUNA [194] achieves unified visual representations via a VAE encoder followed with a semantic encoder. Collectively, these models demonstrate the effectiveness of pixel-based encoding in balancing semantic controllability from language models and highresolution visual fidelity from diffusion processes.

Despite their effectiveness, pixel-based encoding approaches in fused AR and diffusion frameworks also face several limitations. First, models that rely on continuous latent spaces (e.g., via SD-VAE) introduce significant computational overhead during training and inference, due to the iterative nature of diffusion sampling and the need for high-dimensional feature processing. This can become especially burdensome when scaling to high-resolution image generation or multi-turn vision-language interactions. Second, alignment between textual and visual modalities remains challenging. While bidirectional attention mechanisms enable cross-modal fusion, the latent space representations—particularly those learned through unsupervised reconstruction objectives in SD-VAE—may not always be optimally aligned with semantically meaningful language tokens, potentially leading to weaker fine-grained controllability or less interpretable generation. Finally, discrete tokenization schemes, as used in Show-o, inherit issues from VQ-based models such as codebook collapse and limited capacity to represent subtle visual nuances. These symbolic tokens, while compatible with transformer-style modeling, may constrain visual diversity and reduce reconstruction fidelity compared to continuous latent methods.

2) Hybrid Encoding: As shown in Fig. 5 (c-2), hybrid encoding fuses both semantic features (e.g., from CLIP or ViT encoders) and pixel-level latents (e.g., from SD-VAE), providing a more expressive image representation. This approach allows models to leverage high-level semantic abstraction while maintaining detailed visual information. Specifically, Janus-flow [195], Mogao [196] and BAGEL [197] adopt a dual-encoder architecture and presents a minimalist architecture that harmonizes AR language models with rectified flow. They decouples the understanding and generation encoders, using SigLIP or the concatenation of SigLIP and SDXL-VAE as the vision encoder for multimodal understanding and SDXL-VAE or FLUX-VAE for image generation. LightFusion [198] reduces the training difficulty of Bagel by initializing independent understanding and generation models. Omni-View [248] further achieves this unification in multi-view image scenarios by adopting a generative paradigm inspired by MVAR [249]. EMMA [200] concatenates the understanding tokens and generation tokens along the channel dimension rather than the token-wise concatenation [197], enabling the model to handle understanding and generation tasks simultaneously. Meanwhile, EMMA replaces the understanding encoder SigLIP2 with a mixtureof-experts architecture to better handle diverse types of input images. Unlike Bagel, which connects generation and understanding across all layers, HBridge [199] is built upon a pre-trained LLM paired with a diffusion-based generative expert, connecting them only via a selective mid-layer bridge. However, the pseudo hybrid encoding design limits the model’s ability to simultaneously leverage both seman-

tic and pixel-level features during generation, as only the pixel encoder is active in the image synthesis process. This decoupling, while beneficial for modularity and training efficiency, prevents the model from fully exploiting semantic cues during image decoding, potentially weakening finegrained alignment and multimodal compositionality in generative tasks.

Despite their advancements, hybrid encoding methods face several challenges. The integration of dual-encoder architectures and the combination of autoregressive and diffusion processes increase the model’s overall complexity. This can result in higher computational costs and longer training times, making them less efficient compared to simpler models. Furthermore, ensuring effective alignment between semantic and pixel-level features requires careful architectural design and optimization. This alignment process can be difficult to achieve and fine-tune, limiting the model’s ability to fully utilize both modalities in a balanced way. Additionally, balancing the objectives of vision-language understanding and image generation within a unified model often leads to trade-offs, where improvements in one task may come

- at the expense of the other. These limitations underscore the need for more efficient hybrid designs that can better leverage the strengths of both visual and semantic features while reducing computational overhead and maintaining high performance across tasks.

#### 3.4 Any-to-Any Multimodal Models

While early unified multimodal models primarily focused on text-image pairs, recent research has expanded toward any-to-any multimodal modeling. This ambitious approach seeks to create models that can process and generate across a diverse set of modalities, including audio, video, speech, music, and beyond. These models aim to unify modalityspecific encoders and decoders within a single architecture, enabling tasks such as text-to-audio, video-to-text, speechto-music, or even image-to-video generation. This section reviews representative works in this emerging field, highlighting their design principles, modularity, and current limitations.

Most any-to-any models follow a modular design, where each modality is paired with a specialized encoder and decoder, while a shared backbone facilitates cross-modal representation learning and sequence modeling. For example, OmniFlow [208] integrates HiFiGen [250] for audio and music generation, SD-VAE [14] for image processing, and uses a DiT-like diffusion model (MMDiT) [15] as the backbone. This modular design allows the model to efficiently combine different modalities for complex generation tasks.

Some models rely on shared embedding spaces to unify different modalities at the feature level. For instance, Spider [207], X-VILA [205], and Next-GPT [201] leverage ImageBind—a contrastively trained model that maps six modalities (text, image, video, audio, depth, and thermal) into a single embedding space. This unified representation enables flexible conditioning and generation via modality-specific decoders, such as Stable Diffusion [14], Zeroscope, or LLMbased text decoders [1]. While this approach is elegant in theory, its generative capacity is often constrained by the quality of the decoder and the granularity of the shared embedding.

Other models, such as AnyGPT [204] and Unified-IO 2 [202], extend the sequence-to-sequence paradigm to handle multiple modalities. AnyGPT [204] utilizes EnCodec [251] for audio tokenization, SpeechTokenizer [252] for speech, and trains a unified Transformer with modality-specific prefixes. Unified-IO 2 [202], on the other hand, adopts a more structured encoder-decoder design that includes visual, audio, and language modalities, supporting tasks like AST-to-text, speech-to-image, or video captioning within a single model.

A recent and notable addition to the any-to-any unified multimodal models is M2-omni [209], which introduces a highly versatile architecture capable of processing and generating a wide variety of modalities, including text, image, video, and audio. M2-omini takes a step forward by incorporating multiple modality-specific tokenizers and decoders, each carefully designed to handle the unique characteristics of different data types. Specifically, it utilizes NaViT [253] to encode videos and images of arbitrary resolution, and combines a pre-trained SD-3 [243] as the image decoder. For audio, M2-omini introduces paraformer-zh [254] to extract audio tokens, and feeds the predicted discrete audio tokens into the pretrained CosyVoice [255] flow matching and vocoder model to generate audio streams. This integration ensures that M2-omini can effectively generate high-quality images, and audio streams from various inputs, making it a truly multi-modal powerhouse. Ming-Omni [256] adheres to the integrated MoE architecture, wherein modality-specific routing is facilitated through dedicated mechanisms tailored for each token, thereby enabling customized routing distributions. To address the multi-scale phenomenon inherent in visual generation [113], Ming-Omni employs multi-scale learnable queries, directed by an alignment strategy, to iteratively generate images progressing from coarse to fine detail. Furthermore, Ming-Omni integrates the audio modality and implements a dual-stage training strategy to mitigate the mutual influence between audio comprehension and generation tasks. The initial stage emphasizes comprehension capabilities, while the subsequent stage concentrates on enhancing generation quality. BLIP3-o [174] also employs learnable queries to bridge multimodal understanding and generation. However, it utilizes two diffusion models: one for learning CLIP embeddings and the other for using CLIP as a condition to generate images. It reveals that flow matching loss is more effective than MSE loss, enabling more diverse image sampling and yielding better image quality. By using Ling-Flash-2.0 [257] as the main backbone, Ming-Falsh-Omni [258] achieves a favorable trade-off between performance and efficiency. Addressing the specific demands of real-time interaction, Qwen3-Omni [259] proposes a Thinker-Talker MoE architecture, which decouples deep reasoning from streaming speech generation to minimize latency without sacrificing intelligence. Furthermore, LongCat-Flash-Omni [260] extends these capabilities into long-context scenarios (up to 128k tokens), utilizing a Shortcut-connected MoE and curriculum learning strategies to effectively model long-range dependencies in video and audio sequences.

Despite promising progress, current any-to-any models still face several challenges. One key issue is modality imbalance, where text and image modalities are often dominant,

while others like audio, video, and music are underrepresented. This limits the diversity of tasks these models can handle. Another challenge is scalability, as supporting a wide range of modalities increases model complexity, leading to higher inference latency and greater resource requirements. Additionally, ensuring semantic consistency across modalities remains a non-trivial task, with models often struggling to maintain grounded and aligned outputs. These challenges represent ongoing areas of research in the development of any-to-any multimodal models.

Nevertheless, these models represent a crucial step toward developing universal foundation models that can understand and generate across the full spectrum of human sensory input and communication. As data, architectures, and training paradigms evolve, future any-to-any models are expected to become more compositional, efficient, and capable of truly universal cross-modal generation.

### 4 DATASETS ON UNIFIED MODELS

Large-scale, high-quality, and diverse training data form the bedrock for building powerful unified multimodal understanding and generation models. These models typically require pre-training on vast amounts of image-text pairs to learn cross-modal correlations and representations. It is important to note that before being trained on largescale multi-modal data, these models are ofter initialized with parameters derived from training on a large-scale natural language corpus, such as Common Crawl 1, RedPajama [318], WebText [319], etc. Since this survey primarily focuses on multimodal models, the discussion in this section will exclude text-only data. Based on the primary use and modality characteristics, common pre-training multimodal datasets can be broadly categorized as follows: Multimodal Understanding datasets, Text-to-Image Generation datasets, Image Editing datasets, Interleaved Image-Text datasets, and other datasets for image generation conditioned on both text and image inputs. This section will elaborate on representative datasets listed in Tab. 3 within each category, focusing on those released from 2020 onwards.

#### 4.1 Multimodal Understanding Datasets

These datasets are primarily used to train the cross-modal understanding capabilities of models, enabling tasks such as image captioning, visual question answering (VQA), imagetext retrieval, and visual grounding. They typically consist of large collections of images paired with corresponding textual descriptions.

- • RedCaps [261]: This dataset comprises 12 million image-text pairs sourced from Reddit. It is particularly specialized in capturing everyday items and moments (like pets, hobbies, food, leisure, etc.) frequently shared by users on social media platforms.
- • Wukong [262]: The Wukong dataset is a large-scale Chinese multimodal pre-training dataset containing 100 million Chinese image-text pairs filtered from the web. Its creation addressed the lack of large-scale, highquality Chinese multimodal pre-training data, significantly contributing to the development of multimodal models targeting Chinese scenarios.

1. https://commoncrawl.org

TABLE 3 Overview of common datasets used for pre-training unified multimodal understanding and generation models. This table categorizes datasets by primary application (Multimodal Understanding, Text-to-Image Generation, Image Editing, Interleaved Image-Text, and Other conditional generation tasks), detailing the approximate sample size and release date for each dataset.

Dataset Samples Date Multimodal Understanding

RedCaps [261] 12M 2021-11 Wukong [262] 100M 2022-02

LAION [263] 5.9B 2022-03 COYO [264] 747M 2022-08

Laion-COCO [265] 600M 2022-09 DataComp [266] 1.4B 2023-04 GRIT [267] 20M 2023-06

CapsFusion-120M [268] 120M 2023-10 ShareGPT4V [269] 100K 2023-11 ALLaVA-4V [227] 1.4M 2024-02

Cambrian-10M(7M) [270] 10M 2024-06 LLaVA-OneVision [271] 4.8M 2024-08 Infinity-MM [272] 40M 2024-10 Honey-Data-15M [273] 15M 2025-10

Text-to-Image

CC-12M [274] 12M 2021-02 LAION-Aesthetics [263] 120M 2022-08

SAM [275] 11M 2023-04 Mario-10M [276] 10M 2023-05

RenderedText [277] 12M 2023-06

JourneyDB [278] 4M 2023-07 AnyWord-3M [279] 3M 2023-11

CosmicMan-HQ 1.0 [280] 6M 2024-04

DOCCI [281] 15K 2024-04 PixelProse [282] 16M 2024-06

DenseFusion [283] 1M 2024-07

Megalith [284] 10M 2024-07 text-to-image-2M [285] 2M 2024-09

PD12M [286] 12M 2024-10 SFHQ-T2I [287] 122K 2024-10

EliGen TrainSet [288] 500k 2025-01 TextAtlas5M [289] 5M 2025-02 BLIP-3o 60k [174] 60K 2025-05

- ShareGPT-4o-Image [290] 45K 2025-06 Poster100K [291] 100K 2025-06

Text-Render-2M [291] 2M 2025-06 Echo-4o-Image [292] 106K 2025-08 FLUX-Reason-6M [293] 6M 2025-09

Image Editing

InstructP2P [294] 313K 2022-11 Magicbrush [295] 10K 2023-06

HIVE [296] 1.1M 2023-07 HQ-Edit [297] 197K 2024-04

SEED-Data-Edit [170] 3.7M 2024-05 EditWorld [298] 8.6K 2024-06

UltraEdit [299] 4M 2024-07 PromptFix [300] 1M 2024-09 OmniEdit [301] 1.2M 2024-11

AnyEdit [302] 2.5M 2024-11 RefEdit [303] 18K 2025-04 Imgedit [304] 1.2M 2025-05

ByteMorph-6M [305] 6.4M 2025-05

- ShareGPT-4o-Image [290] 46K 2025-06

GPT-Image-Edit-1.5M [306] 1.5M 2025-07

X2Edit [307] 3.7M 2025-08 Pico-Banana-400K [308] 400K 2025-10

Interleaved Image-Text

Multimodal C4 [309] 101.2M 2023-04 OBELICS [310] 141M 2023-06

CoMM [311] 227K 2024-06 OmniCorpus [312] 8B 2024-10

Other Text+Image-to-Image

LAION-Face [313] 50M 2021-12 MultiGen-20M [314] 20M 2023-05

Subjects200K [315] 200K 2024-11 X2I-subject-driven [84] 2.5M 2024-12

SynCD [316] 95K 2025-02

Graph200K [317] 200K 2025-03 MetaQuery Instruct 2.4M [171] 2.4M 2025-06

Echo-4o-Image [292] 73K 2025-08

##### • LAION [263]: The LAION (Large-scale Artificial Intelligence Open Network) project provides one of the

largest publicly available image-text pair datasets. For instance, LAION-5B contains nearly 6 billion image-text pairs crawled from the web. This data is filtered using CLIP models to ensure a degree of relevance between images and texts. Due to its immense scale and diversity, the LAION dataset has become fundamental for pre-training many large multimodal models. Its subset, Laion-COCO [265], contains 600 million samples with high-quality captions and aims to provide a large-scale dataset stylistically closer to MS COCO [320].

- • COYO [264]: COYO is another large-scale image-text pair dataset, comprising approximately 747 million samples. Similar to LAION, it is sourced from web crawls and undergoes filtering processes. It offers the community an alternative large-scale pre-training resource to LAIONl.
- • DataComp [266]: DataComp, contains 1.4 billion samples derived from Common Crawl using carefully designed filtering strategies (CLIP score and Image-based filtering), intended to provide higher quality image-text pairs than raw crawled data.
- • ShareGPT4V [269]: This dataset provides approximately 100K high-quality image-text conversational data points. It is specifically designed and used to enhance the instruction-following and dialogue capabilities of large multimodal models, making them better conversational agents.
- • ALLaVA [227]: This dataset, comprising 1.4 million samples, is synthetically generated to facilitate the training of resource-friendly Lite Vision-Language Models (LVLMs). The generation pipeline leverages strong proprietary models (like GPT-4V) in a multistage process: first, images are selected from sources like LAION and Vision-FLAN; then, fine-grained, detailed captions are generated for these images; finally, complex reasoning visual question-answering pairs are created, emphasizing detailed answers that include evidence and chain-of-thought, to support robust visual instruction fine-tuning.
- • CapsFusion-120M [268]: It is a large-scale collection of 120M image-text pairs selected from LaionCOCO [265]. The caption is acquired by integrating the captions in Laion-COCO with CapsFusionLLaMA [268].
- • Cambrian-10M(7M) [270]: Cambrian-10M is a largescale dataset designed for multimodal instruction tuning, sourced from a diverse array of data with an unbalanced distribution across categories. To enhance the quality of the dataset, data filtering based on a refined data ratio is applied, which results in the creation of Cambrian-7M.
- • LLaVA-OneVision [271]: This visual instruction tuning collection features two main parts: a Single-Image dataset of 3.2 million diverse, categorized samples (QA, OCR, math, etc.), and the OneVision dataset with 1.6 million mixed-modal samples (including video, multiimage, and selected single-image data).
- • Infinity-MM [271]: Infinity-MM is a comprehensive multimodal training dataset with over 40 million samples, created by extensively collecting and categorizing existing open-source datasets alongside newly gen-

erated data. This collection includes image captions, general visual instructions, higher-quality selective instructions, and a significant portion of data generated by GPT-4 or synthesized using a custom VLM-based pipeline to ensure alignment and diversity. All data undergoes rigorous processing and filtering for quality and consistency.

• Other Datasets: Additional understanding datasets developed recently include GRIT (Grid-based Representation for Image-Text) [267] (20M samples emphasizing fine-grained image region-text phrase alignment). Furthermore, while SAM Dataset [275] does not initially consist of image-text pairs, the collection of 11 million high-resolution images with detailed segmentation masks offers valuable spatial and semantic information. It can enhance the fine-grained understanding capabilities of multimodal models, like comprehending object locations, boundaries, or performing region-specific operations. In addition, data for text-to-image models can also be used for multimodal understanding task.

#### 4.2 Text-to-Image Datasets

These datasets are mainly used for training models that generate images corresponding to textual descriptions. They typically consist of image-text pairs, often with a higher emphasis on the aesthetic quality of the images, the richness of the content, or specific stylistic attributes.

- • CC-12M (Conceptual Captions 12M) [274]: CC-12M contains about 12 million image-text pairs extracted and filtered from web Alt-text. Compared to raw webcrawled data, its textual descriptions are generally more concise and descriptive, making it widely used for training text-to-image models.
- • LAION-Aesthetics [263]: This is a subset of the LAION dataset, filtered using an aesthetic scoring model to select approximately 120 million images (and their texts) deemed to have higher “aesthetic value”.
- • Text Rendering Datasets: Several datasets have been developed to specifically address the challenges of accurately and legibly rendering text within generated images. Mario-10M [276], with 10 million samples, was used to train the TextDiffuser model [276], providing data designed to improve text placement and legibility. The RenderedText dataset [277] offers 12 million highresolution synthetic images of handwritten text, generated with diverse visual attributes, serving as a rich resource for handwritten text understanding and generation. AnyWord-3M [279], containing 3 million samples, is crucial for training models like AnyText [279] and also focuses on enhancing the quality of generated text. Lastly, TextAtlas5M [289] targets dense text generation, incorporating a diverse mix of interleaved documents, synthetic data, and real-world images with longer captions and human annotations to tackle complex textrich image scenarios.
- • JourneyDB [278]: JourneyDB consists of 4 million highquality image-prompt pairs generated by the Midjourney platform 2. As Midjourney is known for generating

2. www.midjourney.com

creative and artistic images, this dataset provides valuable resources for training models to learn complex, detailed, and artistically styled text-to-image mappings.

- • CosmicMan-HQ 1.0 [280]: It comprises 6 million highquality real-world human images with an average resolution of 1488 × 1255 pixels. This dataset is distinguished by its precise text annotations, derived from 115 million attributes varying in granularity. It can be used to improve the capability of generating human images.
- • DOCCI [281]: DOCCI provides 15k uniquely curated images, each with long, human-annotated English descriptions (average 136 words) designed to be highly detailed and to differentiate between similar images. The dataset’s focus on fine-grained descriptions and contrastive image sets makes it a valuable resource for training and evaluating both image-to-text and text-toimage models, particularly for their ability to handle nuanced details and complex compositions.
- • PixelProse [282]: PixelProse extracted from DataComp [266], CC-12M [274], and RedCaps [261], contains richly annotated images with corresponding textual descriptions. This dataset provides valuable metadata such as watermark presence and aesthetic scores which can be used for filtering to get expected images.
- • Megalith [284]: Megalith is a dataset consisting of approximately 10 million links to Flickr images categorized as “photo” with licenses ensuring no copyright restrictions. The captions made by the community using models like ShareCaptioner [269], Florence2 [321], and InternVL2 [11], [66] are available publicly.
- • PD12M [286]: PD12M consists of 12.4 million highquality public domain and CC0-licensed images paired with synthetic captions generated using Florence-2large [321]. It is designed for training text-to-image models, offering a substantial collection while minimizing copyright concerns.
- • Synthesized Datasets: Specialized datasets for text-toimage synthesis are increasingly created using existing generative models. The text-to-image-2M dataset [285] provides 2 million enhanced text-image pairs for finetuning, curated using advanced T2I and captioning models. SFHQ-T2I [287] offers 122K diverse, highresolution synthetic face images generated by multiple T2I models, ensuring variance and privacy. For entity control, the EliGen TrainSet [288] uses images from a baseline model (FLUX.1-dev) and MLLM-generated prompts for stylistic consistency and detailed annotation. Similarly, BLIP-3o 60k [174] provides 60,000 instruction tuning samples distilled from GPT-4o, covering various categories for diverse training. ShareGPT4o-Image [290] contributes 45K text-to-image pairs, where prompts are generated through both a structured attribute-first approach and an image-first approach, with corresponding images synthesized by GPT-4o’s image generation capabilities to distill its advanced skills. To specifically address blind spots in real-world data, Echo-4o-Image [292] provides over 100K samples targeting surreal fantasy scenarios and complex, longtail instructions to enhance model imagination and alignment.

• Other Datasets: SAM dataset [275] (approx. 11 M highresolution images) and DenseFusion [283] (1M samples) are other potential data sources for text-to-image generation model training. Note that, the multimodal understanding datasets can be utilized for synthesizing text-to-image generation data via aesthetics score filtering, NSFW filtering, resolution filtering, watermark filtering, recaption, etc., which is not introduced here.

#### 4.3 Image Editing Datasets

With advancing model capabilities, instruction-based image editing has become an important research direction. Datasets in this category typically contain triplets of (source image, editing instruction, target image). These datasets are utilized to train models to alter input images according to textual commands, thereby enhancing both the comprehension and generation capabilities of unified models.

- • InstructPix2Pix [294]: This dataset was generated using an innovative synthetic approach: first, a large language model (like GPT-3) generates an editing instruction and a caption for the target image; then, a text-to-image model (like Stable Diffusion) generates the “before” and “after” images based on the original and target captions. This method automatically created about 313K (instruction, input image, output image) training samples.
- • MagicBrush [295]: MagicBrush is a high-quality, manually annotated dataset for instruction-based image editing. It contains approximately 10K samples covering various realistic and fine-grained editing operations (like object addition/removal/replacement, attribute modification, style transfer) and provides masks for the edited regions. Its manual annotation leads to more natural and diverse instructions.
- • HIVE [296]: The HIVE framework introduces human feedback into instructional visual editing, providing a 1.1M training dataset (generated similarly to InstructPix2Pix using GPT-3 and Prompt-to-Prompt, plus cycle consistency augmentation) and a 3.6K reward dataset where humans rank model outputs.
- • EditWorld [298]: EditWorld introduces the task of “world-instructed image editing,” focusing on realistic world dynamics. Its dataset is curated through two branches: one uses GPT-3.5 for world instructions and T2I models for complex input-output image generation, and the other extracts paired frames from videos with vision-language models generating corresponding instructions for dynamic transformations.
- • PromptFix [300]: PromptFix constructs a large-scale instruction-following dataset ( 1.01M triplets) focusing on a comprehensive range of image processing tasks, particularly low-level tasks (e.g., inpainting, dehazing, super-resolution, colorization).
- • HQ-Edit [297], SEED-Data-Edit [170], UltraEdit [299], OmniEdit [301], AnyEdit [302]: These represent more recent, larger-scale image editing datasets. For instance, SEED-Data-Edit contains 3.7M samples, UltraEdit has 4M samples, AnyEdit provides 2.5M samples, OmniEdit includes 1.2M samples, and HQ-Edit contains 197K samples. They often combine automated gener-

ation with human filtering/annotation, aiming to provide larger-scale, higher-quality, and more diverse editing instructions and image pairs to train more robust instruction-following editing models.

- • RefEdit [303]: This synthetic dataset specifically targets instruction-based editing challenges involving referring expressions in complex scenes. It’s generated using GPT-4o for text components (prompts, instructions, referring expressions), FLUX for initial images, Grounded SAM for precise mask generation from expressions, and specialized models for controlled edits like object removal or modification.
- • ImgEdit [304]: ImgEdit is a large-scale (1.2M edit pairs) dataset designed for high-quality single-turn and multiturn image editing. Its multi-stage generation pipeline filters LAION-Aesthetics, uses vision-language models and detection/segmentation models for grounding and instruction generation (including spatial cues and multi-turn dialogues), employs state-of-the-art generative models (FLUX, SDXL with plugins) for taskspecific inpainting, and uses GPT-4o for final quality filtering.
- • ByteMorph-6M [305]: ByteMorph-6M is a large-scale dataset with over 6 million image editing pairs specifically designed for instruction-guided editing involving non-rigid motions (e.g., camera viewpoint shifts, object deformations, human articulations). It is constructed by first using a Vision-Language Model to generate “Motion Captions” from instruction templates for an initial frame; then, an image-to-video model generates a dynamic video based on this motion caption; finally, frames are sampled from these videos, and an LLM generates precise editing instructions describing the transformation between neighboring frame pairs, which form the source-target editing data.
- • ShareGPT-4o-Image (Editing) [290]: Complementing its text-to-image data, ShareGPT-4o-Image also includes 46K instruction-guided image editing triplets. These samples are generated by first selecting a source image (either from its text-to-image collection or real photos), then sampling an editing task from a predefined taxonomy (e.g., object manipulation, style transfer), having an LLM synthesize a natural language instruction for that task and image, and finally using GPT-4o’s image generation capabilities to produce the edited image.
- • GPT-Image-Edit-1.5M [306]: GPT-IMAGE-EDIT-1.5M is a large-scale image editing dataset containing over 1.5 million high-quality instruction-guided image editing triplets. It is constructed by leveraging the powerful capabilities of GPT-4o to systematically unify and refine three existing datasets: OmniEdit, HQ-Edit, and UltraEdit. The core methodology involves regenerating output images to enhance visual quality and instruction alignment, as well as selectively rewriting prompts to improve semantic clarity. This process results in a highfidelity corpus designed to bridge the gap between proprietary and open-source instruction-guided image editing models.
- • X2Edit [307]: X2Edit is a large-scale and comprehensive image editing dataset with 3.7 million samples, designed to be balanced across 14 diverse editing tasks.

It is constructed through an automated pipeline that first uses a VLM to generate task-aware instructions, which are then executed by industry-leading and expert generative models to produce the edited images. To overcome the quality and balance issues in existing open-source resources, all generated pairs undergo a final, rigorous filtering stage based on multiple scoring mechanisms to ensure high fidelity and accuracy.

#### 4.4 Interleaved Image-Text Datasets

Beyond datasets consisting of paired images and captions, another important category comprises interleaved imagetext data. These datasets contain documents or sequences where text and images naturally follow one another, mirroring content found on webpages or in documents. Training models on this interleaved data enhances their capability to comprehend and generate multimodal content, an essential goal for unified models.

- • Multimodal C4 (MMC4) [309]: MMC4 augments the large-scale text-only C4 [23] corpus by algorithmically interleaving images into the text documents sourced from Common Crawl. This public dataset, containing over 101 million documents and 571 million images, was created to provide the necessary interleaved pretraining data for models designed to process mixed sequences of images and text.
- • OBELICS [310]: OBELICS is an open, web-scale dataset comprising 141 million multimodal web documents extracted from Common Crawl, featuring 353 million images interleaved with 115 billion text tokens. The dataset focuses on capturing the full document structure rather than isolated image-text pairs, aiming to improve model performance on various benchmarks.
- • CoMM [311]: CoMM is a high-quality, curated dataset focused specifically on the coherence and consistency of interleaved image-text sequences, containing approximately 227K samples. It addresses limitations in narrative flow and visual consistency observed in larger datasets by sourcing content primarily from instructional and visual storytelling websites (like WikiHow) and applying a multi-perspective filtering strategy. CoMM aims to enhance MLLMs’ ability to generate logically structured and visually consistent multimodal content and introduces new benchmark tasks specifically designed to evaluate these capabilities.
- • OmniCorpus [312]: OmniCorpus is a very large-scale (10 billion-level) image-text interleaved dataset, containing 8.6 billion images and 1,696 billion text tokens from 2.2 billion documents. It was created using an efficient data engine that extracts and filters content from diverse sources, including English and non-English websites as well as video platforms (extracting keyframes and transcribing audio). The dataset incorporates human-feedback filtering to enhance data quality, aiming to provide a solid foundation for MLLM research.

#### 4.5 Other Text+Image-to-Image Datasets

Beyond the previously mentioned categories, to further enhance a unified model’s capabilities—such as generating

images based on provided subject images, or utilizing control signals (e.g., depth maps, canny maps) —we introduce relevant datasets in this section.

- • LAION-Face [313]: The datasets discussed above emphasize general subject-driven generation, whereas IDpreserving image generation represents a specialized subset of this category. Utilizing LAION-Face, which includes 50 million image-text pairs, recent advancements such as InstantID [322] have succeeded in generating images while maintaining character identity.
- • MultiGen-20M [314]: This dataset comprises 20 million samples designed to train models capable of unified image generation conditioned on multiple control signals (e.g., text descriptions, edge maps, depth maps, segmentation masks, sketches), such as UniControl [314]. It integrates data from various sources and converts them into a unified format, enabling models to learn multitask, multi-conditional image generation. The dataset can be structured as triples, such as “depth map, instruction with prompt, target image” (The “instruction with prompt” might be phrased as: “Generate an impressive scene following the depth map.”), to effectively train unified models.
- • Subjects200K [315]: Containing 200K samples, Subjects200K focuses on subject-driven image generation, crucial for personalized content creation. This dataset was generated synthetically through a multistage pipeline: initially, an LLM (ChatGPT-4o) creates structured descriptions involving object categories and scenes; subsequently, an image synthesis model (FLUX [16]) generates diverse yet consistent paired images based on these descriptions; finally, the LLM performs quality assessment on the generated pairs to ensure subject consistency, proper composition, and high resolution.
- • SynCD [316]: SynCD (Synthetic Customization Dataset) provides approximately 95K sets of images specifically designed for text+image-to-image customization tasks, addressing the lack of public datasets containing multiple images of the same object under diverse conditions. It is synthesized by leveraging existing text-to-image models and 3D asset datasets (like Objaverse [323]) to generate multiple consistent views of an object with varied lighting, backgrounds, and poses, incorporating techniques like shared attention and depth guidance.
- • X2I-subject-driven [84]: The X2I-subject-driven dataset facilitates subject-driven image generation through two components. The GRIT-Entity dataset is derived from the GRIT [267] dataset by automatically detecting and segmenting objects from images, followed by an optional MS-Diffusion [324] repainting step to improve quality and diversity. To encourage more robust generative capabilities beyond simple copy-paste patterns, the higher-quality Web-Images dataset was constructed by identifying notable individuals through automated text analysis and large language model filtering, scraping their web images, performing automated visual verification to ensure subject accuracy, and then captioning the selected images.
- • Graph200K [317]: Graph200K is a graph-structured

dataset built upon Subjects200K, where each image is augmented with 49 types of annotations spanning five meta-tasks: conditional generation (e.g., canny edges, depth maps, segmentation), IP preservation, style transfer (semantic-variant and -invariant), image editing (background-variant and -invariant using VLMs and inpainting models), and restoration (via online degradations). This structure aims to increase task density and inter-relation, enabling models to learn shared and transferable knowledge for universal image generation by formulating tasks as paths within this graph.

• Echo-4o-Image (Multi-Reference) [292]: This component of the dataset addresses the scarcity of structured, multi-input generation tasks in natural image collections. It provides 73K synthetic samples for “Multi-toone” generation, which are explicitly designed with diverse instructions and rich reference information. This controlled synthesis offers a more targeted and varied training source for multi-reference image composition compared to alternatives like sampling from video frames.

Subject-driven generation, involving both single and multiple subjects, is a crucial image generation capability that is increasingly attracting attention within the community. It is also anticipated to be a significant feature inherent in unified models. However, obtaining such specialized data from public datasets is challenging, leading to the frequent use of data synthesis methods, exemplified by datasets like Subjects200K and SynCD. These datasets illustrate the growing reliance on synthetic data to address the shortage of publicly available training examples needed for tasks like subject-driven generation and customization.

To create large-scale datasets, various pipelines [85], [324], [325], [326] have been developed to programmatically generate suitable training data, typically utilizing readily accessible image or video sources. Below, we provide a brief overview of these pipelines for reference.

- • Data synthesis from images: These pipelines often start with single images, using models like BLIP-2 [53] or Kosmos2 [267] for initial captioning (including grounding captions with bounding boxes), followed by object detection (e.g., Grounding DINO [327]) and segmentation (e.g., SAM [275]) to extract subject masks and region captions. These pipelines can generate data for single subject customization and multiple subjects customization.
- • Data synthesis from videos: Data constructed from images often cause the copy-paste issue in model learning. The pipeline of synthesizing data from videos can alleviate this issue by extracting subjects from different frames with video segmentation models (e.g., SAM2 [328]). In addition, this pipeline can also enable the generation of training data for image editing task. [85].

Robust unified multimodal models rely critically on large-scale, high-quality, and diverse training datasets developed recently, encompassing image-text pairs, interleaved image-text documents, and task-specific formats. While massive web-scale paired data (like LAION, COYO) and interleaved document corpora (like MMC4, OBELICS)

provide broad semantic coverage and contextual understanding for pre-training, significant efforts focus on enhancing data quality and tailoring resources for specific attributes or advanced capabilities. Specialized datasets are increasingly crucial for improving instruction-based editing, accurate text rendering, coherent multimodal generation, and complex conditional control. Furthermore, recognizing the scarcity of high-quality public data for tasks like instruction-based image editing and subject customization, the development and utilization of data synthesis pipelines have become essential, enabling the creation of targeted datasets needed to train these highly specific model functionalities. Ultimately, the continuous evolution, growing scale, targeted specialization, and innovative synthesis of these varied data resources are the fundamental drivers enabling the increasingly sophisticated understanding and generation capabilities of unified multimodal models.

### 5 BENCHMARKS

Modern large-scale unified multimodal models should not only align visual and linguistic information at the pixel level but also perform complex reasoning, support coherent multi-turn dialogue and integrate external knowledge. Simultaneously, these models are expected to produce highfidelity visual outputs that faithfully adhere to textual prompts while providing users with fine-grained control over stylistic and compositional elements. In this section we systematically summarize the related evaluation benchmarks. Please refer to Tab. 4 for statistical summary.

#### 5.1 Evaluation on Understanding

Perception. Modern vision-language large models must accurately connect visual inputs with linguistic descriptions through grounding, recognition and retrieval. Early image–text retrieval and captioning benchmarks such as Flickr30k [392], MS COCO Captions [393] evaluate whether models can retrieve relevant captions and localize textual phrases to image regions. Visual question answering benchmarks like VQA [329], VQA v2 [330], VisDial [335] and TextVQA [337] further require models to interpret complex scenes and answer free-form queries about objects, attributes and relationships. Domain-specific challenges such as ChartQA [336] assess understanding of structured charts and graphs, while VSR [8] probes spatial relation reasoning in real-world images.

To unify the evaluation, large-scale meta-benchmark suites test both low-level perception and expert reasoning. MMBench [343] supplies 3K bilingual multiple-choice questions spanning grounding, recognition and retrieval, enabling cross-lingual comparison. MMMU [344] adds about 11.5K college-level multimodal problems across six disciplines to probe domain knowledge and logical deduction. HaluEval [339] diagnoses hallucination recognition on a diverse set of model-generated and annotated statements. MM-Vet [345] covers recognition, OCR, spatial reasoning, maths and open question answering, and its v2 [346] further evaluates interleaved image–text sequences. SEEDBench [348] designs a pipeline for generating multiplechoice questions that target specific evaluation dimensions

and finally offers 19K multi-choice items over 12 dimensions. LLaVa-Bench [341] provides COCO [320] and inthe-wild image sets with dense queries for generalization checks. LAMM [340] supplies instruction-tuning examples covering 2D and 3D modalities for agent development. Open-VQA [349] formulates hierarchical follow-up questions to refine coarse VQA answers. OwlEval [342] offers human-rated open-ended visual questions assessing relevance and informativeness. MMStar [347] curates carefully balanced challenge samples spanning six core skills and 18 axes for high-precision evaluation.

Reasoning. Building on perception-level evaluation, reasoning benchmarks probe progressively richer cognitive skills. CLEVR [331] systematically varies object attributes and spatial relations, forcing models to execute multi-hop programs that test counting, comparison and relational logic. Moving to natural images, GQA [332] leverages dense scene graphs to generate compositional questions whose functional programs are used to test consistency, grounding and plausibility.

Commonsense extensions such as OK-VQA [333] and its larger successor A-OKVQA [338] select questions whose answers lie outside the image, requiring retrieval or inference over world knowledge bases. VCR [334] further demands that a model not only choose the correct answer but also justify it by selecting a coherent rationale, thereby coupling recognition with explanation and testing multistep commonsense chains.

Domain-specific reasoning datasets extend this progression beyond everyday scenes. ChartQA [336] introduces questions that intertwine visual perception with quantitative reasoning over bar, line and pie charts, integrating data extraction, logical comparison and arithmetic calculation. MathVista [350] broadens the scope to mathematical problem solving in visually grounded contexts and combines fine-grained visual understanding with symbolic manipulation across diversified examples. These benchmarks form a layered spectrum that spans structured logical inference, open-domain commonsense, visual explanation and numerically intensive tasks, offering a comprehensive stress-test for multimodal reasoning systems.

Moreover, General-Bench [351], an ultra-large benchmark comprising over 700 tasks and 325,800 instances across varied modalities and capabilities, provides a synergydriven evaluation suite for multimodal generalist models.

#### 5.2 Evaluation on Image Generation

Text-to-Image Generation. Early automated metrics such as FID [394] and CLIPScore [22] established the foundation for evaluating image quality. More recent benchmarks, however, emphasize compositional reasoning, prompt alignment, and real-world applicability. PaintSkills [353], DrawBench [72], and PartiPrompts [352] evaluate core compositional capabilities. GenEval [356] evaluates six finegrained tasks, including single-object generation, object cooccurrence, counting, color control, relative positioning, and attribute binding by comparing outputs from pretrained detectors against ground-truth annotations.

Expanding on this, GenAI-Bench [361] presents 1.6K meticulously crafted human prompts that cover relational,

TABLE 4 Statistical summary of current evaluations and benchmarks for unified large-scale generative models. This table categorizes benchmarks into Understanding, Image Generation, and Interleaved Generation, detailing the size, description, input/output types, and publication venues for each.

Benchmark Size Description In.out Type Venue Understanding

VQA [329] 10M QAs Open-domain Visual QA Image + Question → Answer ICCV2015 VQAv2 [330] 1M QAs Open-domain Visual QA Image + Question → Answer CVPR2017 CLEVR [331] 853K QAs Compositional Visual QA Image + Question → Answer CVPR2017

GQA [332] 22M QAs Compositional Visual QA Image + Question → Answer CVPR2019 OK-VQA [333] 14K QAs Knowledge-based VQA Image + Question → Answer CVPR2019

VCR [334] 290K QAs Commonsense Visual QA Img. + Q. → Answer + Rationale CVPR2019

VisDial [335] 1.2M Dialogs Multi-turn Visual Dialog Image + Dialog → Answer CVPR2019 ChartQA [336] 32.7K QAs Data Visualization QA Image + Question → Answer ACL2020 TextVQA [337] 45K QAs Scene Text Visual QA Image + Question → Answer CVPR2020

A-OKVQA [338] 25K QAs Expanded Commonsense VQA Image + Question → Answer ECCV2022 HaluEval [339] 35K Samples Hallucination Detection Model output → Yes / No EMNLP2023 VSR [8] 3K QAs Spatial Reasoning Image + Question → True / False TACL2023

LAMM [340] 62K QAs Instruction Benchmarking Features + Instruction → Output NeurIPS2023 LLaVa-Bench [341] 150 QAs Instruction Benchmarking Image + Question → Answer NeurIPS2023 OwlEval [342] 82 Qs Visual-related Eval Image + Instruction → Answer Arxiv2023

MMBench [343] 3K QAs Fine-grained Multi-modal Eval Image + Question → Answer ECCV2024 MMMU [344] 11.5K QAs Expert-level Understanding Image + Question → Answer CVPR2024 MM-Vet [345] 218 Samples VL Capability Eval Image + Question → Answer ICML2024

MM-Vet v2 [346] 218+ Samples VL Sequence Understanding Image + Sequences → Answer Arxiv2024

MMStar [347] 1.5K QAs Vision Indispensable Eval Image + Question → Answer NeurIPS2024 SEED-Bench [348] 19K QAs Comprehensive Evaluation Image/Video + MCQ → Answer CVPR2024

Open-VQA [349] Varied VQA Evaluation Image + Q/A → QA Chain ICLR2024 MathVista [350] 6K QAs Math Reasoning Image + Text → Math Output ICLR2024

General-Bench [351] >700 tasks Ultra Large-scale Eval Varied by Task Arxiv2025 Image Generation

DrawBench [72] 200 Prompts Comprehensive Eval Text Prompt → Image NeurIPS2022

PartiPrompts [352] 1600 Prompts Comprehensive Eval Text Prompt → Image TMLR2022 PaintSkills [353] ∼7K Scenes Compositional Eval Text Prompt → Image ICCV2023 HRS-Bench [354] 960 Prompts Multi-skill Eval Text Prompt → Image ICCV2023

TIFA [355] 4081 Prompts QA-based Eval Text Prompt → Image ICCV2023 GenEval [356] 1000 Prompts Object-focused Eval Text Prompt → Image NeurIPS2023

T2I-CompBench [357] 6000 Prompts Compositional Eval Text Prompt → Image NeurIPS2023

HEIM [358] ∼1620 Prompts Comprehensive Eval Text Prompt → Image NeurIPS2023 Commonsense-T2I [359] 500 Prompts Commonsense-driven Eval Text Prompt → Image COLM2024

DSG-1k [360] 1060 Prompts Compositional Eval Text Prompt → Image ICLR2024

GenAI-Bench [361] 1600 Prompts Compositional Eval Text Prompt → Image CVPR2024 ConceptMix [362] 2100 Prompts Compositional Eval Text Prompt → Image NeurIPS2024 DPG-Bench [363] 1065 prompts Attribute Eval Text Prompt → Image Arxiv2024

T2I-CompBench++ [364] 6000+ Prompts Compositional Eval Text Prompt → Image TPAMI2025

MMIG-Bench [365] 4850 Prompts Comprehensive Eval Text Prompt → Image Arxiv2025 OneIG-Bench [366] ∼2k Prompts Comprehensive Eval Text Prompt → Image Arxiv2025

WISE [367] 1k Prompts World Knowledge Eval Text Prompt → Image Arxiv2025 CVTG-2K [368] 2k Prompts Multi-region Visual Text Eval Text Prompt → Image Arxiv2025

WorldGenBench [369] 1072 Prompts World Knowledge Eval Text Prompt → Image Arxiv2025

EditBench [370] 240 Edits Mask-guided Editing Img. + Ins. + [Mask] → Image CVPR2023 MagicBrush [295] 1053 Edits Real-image Editing Image + Instruction → Image NeurIPS2023

EditVal [371] 648 Edits Attribute-focused Eval Image + Instruction → Image Arxiv2023

Emu-Edit [372] 3055 Edits Multi-task Editing Image + Instruction → Image CVPR2024 Reason-Edit [373] 219 Edits Complex Instruction Editing Image + Instruction → Image CVPR2024

I2EBench [374] 2240 Edits Multi-dimensional Eval Image + Instruction → Image NeurIPS2024

HumanEdit [375] 5.7K Edits Human-rewarded Editing Img. + Ins. + [Mask] → Image Arxiv2024 HQ-Edit [297] ∼200K Edits High-resolution Editing Image + Instruction → Image ICLR2025 AnyEdit [302] 1250 Edits Comprehensive Eval Image + Instruction → Image CVPR2025 IE-Bench [376] 301 Edits Human-aligned Perceptual Eval Image + Instruction → Image Arxiv2025

GEdit-Bench [377] 606 Edits Real-world-grounded Editing Image + Instruction → Image Arxiv2025 CompBench [378] 3K Edits Complex Instruction Editing Image + Instruction → Image Arxiv2025

GIE-Bench [379] 1080 Edits Content-preserving Eval Image + Instruction → Image Arxiv2025

EditInspector [380] 783 Edits Comprehensive Eval Image + Instruction → Image Arxiv2025 ComplexBench-Edit [381] <1K List of Edits Chain-dependent Editing Eval Image + Instruction → Image Arxiv2025

ByteMorph-Bench [305] 613 Edits Non-rigid Editing Eval Image + Instruction → Image Arxiv2025 RefEdit-Bench [303] 200 Edits Expression-driven Editing Eval Image + Instruction → Image Arxiv2025 ImgEdit-Bench [304] 200 Edits Expression-driven Editing Eval Image + Instruction → Image Arxiv2025

KRIS-Bench [382] 1267 Edits Cognitive Reasoning Eval Image + Instruction → Image Arxiv2025 Interleaved / Compositional Generation

InterleavedBench [383] 815 Samples Human-curated Interleaving Text + Images → Text + Images EMNLP2024 OpenLEAF [384] 30 Queries Open-domain Interleaving Query → Text + Images MM2024 ISG [385] 1150 Samples Scene-driven Interleaving Graph + Text → Text + Images ICLR2025

MMIE [386] 20K Queries Knowledge-intensive Interleaving History + Query → Response ICLR2025 OpenING [387] 5.4K Samples Open-domain Interleaving Query → Text + Images CVPR2025 UniBench [388] 81 fine-grained tags Unified Compositional Eval Prompt → Images + Answer Arxiv2025

Other Types

MultiGen-20M [314] Varied Controllable Generation Featues + Instruction → Image NeurIPS2023 Dreambench [389] 30 objects Subject-Driven Generation Ref Img. + Instruction → Image CVPR2023 Dreambench++ [390] 150 imgs Personalized Generation Ref Img. + Instruction → Image ICLR2025

VTBench [391] Varied Visual Tokenizer Eval Image → Reconstructed Image Arxiv2025

logical, and attribute-based categories. Its evaluation framework combines human preference judgments with automated alignment scores to provide a comprehensive assessment. In addition, HRS-Bench [354] evaluates 13 distinct skills that are grouped into five major categories: accuracy, robustness, generalization, fairness, and bias, thereby ensuring scalable and reliable performance measurement. Moreover, DPG-Bench [363] focuses on dense prompts that describe multiple objects, with each object characterized by a variety of attributes and relationships.

The T2I-CompBench [357] and its successor T2ICompBench++ [364] specifically target compositional generalization, testing the generation of novel attribute and relation combinations using detector-based scoring. VISOR [395] proposes an automatic method for evaluating the spatial understanding capabilities of generative models. Complementing these, Commonsense-T2I [359] challenges models to depict everyday concepts that require commonsense grounding.

To support large-scale concept diversity, EvalMuse40K [396] provides 40K crowdsourced prompts focusing on nuanced concept representation, and HEIM [358] identifies 12 aspects, including text-image alignment, image quality, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality and efficiency. Considering practical needs, FlashEval [397] shrinks the large-scale evaluation set into diverse smaller ones through iterative search to accelerate the benchmark testing. MEMOBench [398] introduces a comprehensive benchmark for evaluating the emotional understanding and expression capabilities of T2I models and MLLMs. ConceptMix [362] evaluates text-to-image models’ compositional generation ability by sampling k-tuples of visual concepts to construct prompts and automatically verifying concept presence in the resulting images using a strong visual language model. TIFA [355] offers a fine-grained benchmark for evaluating text-to-image faithfulness via visual question answering generated from prompts.

To enrich dependencies in question generation for VQAbased evaluation of image–prompt alignment, DSG-1k [360] refines its questions using a multi-level semantic graph. MMIG-Bench [365] introduced a multi-dimensional assessment framework that rigorously examines text-to-image generation models. OneIG-Bench [366] introduces a comprehensive fine-grained evaluation framework for text-toimage models across more dimensions. [367], [369] evaluate text-to-image models’ world knowledge understanding, which emphasizes semantic consistency, realism, and aesthetics. CVTG-2K [368] evaluates visual-text generation on complex multi-region layouts, diverse text attributes, and fine-grained positioning.

Image Editing. Benchmarks for instruction-guided image editing have grown in scale and scope. MagicBrush [295] is a large-scale, manually annotated dataset for instructionguided real image editing that covers diverse scenarios: single-turn, multi-turn, mask-provided, and mask-free editing. HQ-Edit [297] contains approximately 200K highresolution edits with computed alignment and coherence scores, allowing quantitatively assessing the quality of image edit pairs using GPT-4V.

Building on this, I2EBench [374] consolidates over 2K

images and 4K multi-step instructions across 16 editing dimensions. EditVal [371] offers a standardized benchmark with fine-grained edit annotations and an automated evaluation pipeline aligned with human judgment. EmuEdit [372] covers seven editing tasks including background changes, object-level edits, and style modifications, with paired instructions and I/O descriptions. Reason-Edit [373] is a diagnostic benchmark targeting causal and counterfactual reasoning, emphasizing object relations, attribute dependencies, and multi-step inference.

Offering masked input–reference pairs across varied objects, attributes, and scenes, EditBench [370] delivers a diagnostic benchmark for text-guided image inpainting that enables precise evaluation of editing quality. HumanEdit [375] includes 5,751 high-resolution images and open-form instructions spanning six edit types, with annotated masks and multi-stage human feedback. IE-Bench [376] provides a human-aligned benchmark for evaluating text-driven image editing quality with diverse edits and perceptual scores.

More recent benchmarks include GEdit-Bench [377] which features 606 real-world instruction–image pairs, CompBench [378] that decomposes edits into location, appearance, dynamics and object dimensions via large-scale MLLM–and–human collaboration, and GIE-Bench [379] which uses multiple-choice VQA and object-aware masking on over 1,000 examples to evaluate editing accuracy and content preservation. Following this trend, benchmarks like [302], [304], [380], [381] also undertake comprehensive evaluation of text-guided image editing, assessing vision consistency, artifact detection, instruction adherence, visual quality, and detail preservation.

Other benchmarks include ByteMorph-Bench [305] which tackles non-rigid image manipulation, RefEditBench [303] which evaluates referring-expression–based edits in complex multi-entity scenes, and KRIS-Bench [382] which offers a cognitively grounded suite assessing factual, conceptual and procedural reasoning.

Other Types of Image Generation. Beyond text-to-image generation and editing, additional benchmarks target conditional and personalized synthesis. MultiGen-20M [314] provides over 20 million image–prompt–condition triplets from LAION-Aesthetics-V2 [399], supporting automated evaluation across diverse visual conditions. DreamBench [389] benchmarks personalized generation using 30 reference objects with curated prompts and human fidelity ratings. DreamBench++ [390] scales this to 150 subjects and 1,350 prompts, using advanced vision–language models for human-aligned scoring of concept preservation, composition, and style. Together, these datasets span large-scale automated and fine-grained human-centric evaluation of conditional generation.

VTBench [391] provides a systematic benchmark for evaluating visual tokenizers in autoregressive image generation across image reconstruction, detail preservation, and text preservation.

#### 5.3 Evaluation on Interleaved Generation

Interleaved evaluation benchmarks challenge models to seamlessly alternate between text and image modalities across multiple turns, reflecting realistic dialogue and sto-

rytelling scenarios. InterleavedBench [383] is a representative benchmark carefully curated for the evaluation of interleaved textand-image generation, featuring a rich array of tasks to cover diverse real-world use cases and evaluating models on text quality, perceptual fidelity, multimodal coherence and helpfulness. Building on this, ISG [385] introduces scene-graph annotations and a four-tier evaluation (holistic, structural, block-level and image-specific) over 1K samples in eight scenarios and 21 subtasks, enabling finegrained assessment of interleaved text–image outputs.

Other benchmarks emphasize open-domain instruction and end-to-end interleaving. OpenING [387] assembles 5K human-annotated instances across 56 real-world tasks (e.g. travel guides, design ideation) with IntJudge to test open-ended multimodal generation methods on arbitrary instruction-driven interleaved generation. In contrast, OpenLEAF [384] gathers 30 open-domain queries with each written and reviewed by annotators to probe foundational interleaved text–image generation, measuring entity and style consistency via LMM evaluators plus human validation. Finally, MMIE [386] proposes a unified interleaved suite by sampling from 12 fields and 102 subfields, offering a mix of multiple-choice and open-ended question formats to evaluate models in a diverse manner. In a more recent work, UniBench [388] was introduced as a comprehensive compositional benchmark for evaluating unified models, offering 81 fine-grained tags to ensure high diversity.

#### 5.4 Evaluation on Unification

The above evaluation paradigms mainly assess understanding and generation capabilities in isolation, which is insufficient to determine whether unified models can achieve a complementary interaction between understanding and generation tasks. For example, unified models may leverage their understanding ability to enhance generation, or use generative simulation to support more complete understanding.

To fill this critical gap, RealUnify [400] builds the evaluation on unification around two core aspects: (1) Understanding-Enhanced Generation (UEG), which requires reasoning (e.g., commonsense, logic) to guide image generation; and (2) Generation-Enhanced Understanding (GEU), which requires mental simulation or reconstruction (e.g., of transformed or disordered visual inputs) to solve reasoning tasks. RealUnify contains 1,000 carefully humanannotated instances, covering 10 categories and 32 subtasks. This benchmark combines direct end-to-end evaluation with diagnostic step-by-step evaluation, decomposing tasks into separate understanding and generation stages, enabling us to precisely determine whether performance bottlenecks arise from deficiencies in core capabilities or from failures to effectively integrate these capabilities.

### 6 CHALLENGES AND OPPORTUNITIES ON UNIFIED MODELS

Currently, at its rudimentary stage, unified multimodal models face several significant challenges that should be addressed to achieve robust and scalable understanding and generation capabilities. First, the high dimensionality

of visual and textual data leads to extremely long token sequences. Efficient tokenization and compression strategies are essential to reduce memory and computation costs while preserving representational fidelity. Second, crossmodal attention becomes a performance bottleneck as image resolution and context length increase. Scalable alternatives such as sparse or hierarchical attention mechanisms may potentially mitigate this issue. Third, pretraining datasets often include noisy or biased image–text pairs, particularly for complex image compositions and interleaved image-text data. Reliable data filtering, debiasing, and synthesizing are crucial to ensure fairness and robustness. Fourth, evaluation protocols are typically designed for single tasks in isolation. There is a growing need for comprehensive benchmarks that assess both understanding and generation in an integrated manner, especially for sophisticated tasks such as image editing and interleaved image-text generation. Apart from the issues and challenges on architecture, data, and evaluation, applying chain-of-thought (CoT) reasoning and reinforcement learning (RL) techniques into unified MLLM models to improve both interpretability and performance [401] is also worth exploring. CoT can guide the model to generate intermediate reasoning steps, which are particularly beneficial for complex visual-question answering or image-conditioned generation. Meanwhile, RL can be used to optimize long-horizon objectives such as factual consistency, user satisfaction, or task success rate beyond tokenlevel likelihoods. Moreover, exploring the demographic and social biases of existing unified MLLM models [402] is an important topic to ensure responsible deployment. As these models become increasingly capable across diverse modalities and tasks, unintentional amplification of cultural stereotypes, gender bias, or geographic imbalances embedded in pretraining data may result in harmful outputs. Future work should investigate effective fairness-aware training pipelines. Finally, enabling personalized knowledge-driven generation within unified MLLMs [403] is an emerging and important direction. Personalized models aim to incorporate user-provided concepts—such as specific objects, characters, or styles—into the model’s understanding and generation capabilities. However, current approaches often treat understanding and generation separately, using distinct concept embeddings for each task. This separation limits the model’s ability to generalize to compositional prompts that require implicit knowledge, such as generating ”<bo> wearing its hat” without explicitly describing the hat. Unifying personalized understanding and generation under a shared modeling framework would allow better semantic grounding and contextual generalization.

To the best of our knowledge, most of current unified multimodal models primarily emphasize image understanding and text-to-image generation, while capabilities such as image editing are only attained through postfinetuning. Moreover, advanced functionalities like spatially controlled image generation, subject(s)-driven image generation, and interleaved image-text generation remain largely unexplored in the unified framework. Consequently, we believe there are abundant opportunities to advance the field by addressing key areas such as architectural design, training efficiency, dataset curation, evaluation methodologies, fairness, and reasoning to achieve unified multimodal

models.

### 7 CONCLUSION

We have presented a comprehensive view on unified multimodal models that integrate vision–language understanding and image generation within a single framework. Initially, we provide a concise overview of the foundational knowledge and recent advancements in both multimodal understanding and text-to-image generation models. Subsequently, we systematically survey unified multimodal models by categorizing them into three main paradigms: diffusion-based, autoregressive-based, and hybrid-based approaches. For each category, we introduce related works and further subdivide them into distinct subcategories to help readers better grasp the landscape of this field. Additionally, we curate relevant datasets and benchmarks to facilitate practical implementation and evaluation. Finally, we discuss the key challenges and opportunities in this domain, emphasizing that the study of unified multimodal models is still in its infancy. We hope that our survey will serve as a valuable resource to advance research and innovation in the development of unified multimodal models.

### REFERENCES

- [1] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.
- [2] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.
- [3] W. Zeng, X. Ren, T. Su, H. Wang, Y. Liao, Z. Wang, X. Jiang, Z. Yang, K. Wang, X. Zhang et al., “Pangu-alpha: Large-scale autoregressive pretrained chinese language models with autoparallel computation,” arXiv preprint arXiv:2104.12369, 2021.
- [4] X. Ren, P. Zhou, X. Meng, X. Huang, Y. Wang, W. Wang, P. Li, X. Zhang, A. Podolskiy, G. Arshinov et al., “Pangu-sigma: Towards trillion parameter language model with sparse heterogeneous computing,” arXiv preprint arXiv:2303.10845, 2023.
- [5] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond,” arXiv preprint arXiv:2409.12191, 2024.
- [6] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei et al., “Qwen2. 5 technical report,” arXiv preprint arXiv:2412.15115, 2024.
- [7] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.
- [8] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34892–34916, 2023.
- [9] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.
- [10] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.
- [11] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24185–24198.
- [12] S. Lu, Y. Li, Q.-G. Chen, Z. Xu, W. Luo, K. Zhang, and H.-J. Ye, “Ovis: Structural embedding alignment for multimodal large language model,” arXiv preprint arXiv:2405.20797, 2024.

- [13] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [14] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.
- [15] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller,¨ H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024.
- [16] B. F. Labs, “Flux,” https://github.com/black-forest-labs/flux, 2024.
- [17] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al., “Improving language understanding by generative pre-training,” 2018.
- [18] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020.
- [19] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
- [20] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.
- [21] J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu et al., “Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis,” arXiv preprint arXiv:2310.00426, 2023.
- [22] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.
- [23] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.
- [24] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan, “Autoregressive model beats diffusion: Llama for scalable image generation,” arXiv preprint arXiv:2406.06525, 2024.
- [25] T. Li, Y. Tian, H. Li, M. Deng, and K. He, “Autoregressive image generation without vector quantization,” Advances in Neural Information Processing Systems, vol. 37, pp. 56424–56445, 2024.
- [26] H. Li, J. Yang, K. Wang, X. Qiu, Y. Chou, X. Li, and G. Li, “Scalable autoregressive image generation with mamba,” arXiv preprint arXiv:2408.12245, 2024.
- [27] OpenAI, “Introducing 4o image generation,”

2025. [Online]. Available: https://openai.com/index/ introducing-4o-image-generation/

- [28] L. Fan, L. Tang, S. Qin, T. Li, X. Yang, S. Qiao, A. Steiner, C. Sun, Y. Li, T. Zhu et al., “Unified autoregressive visual generation and understanding with continuous tokens,” arXiv preprint arXiv:2503.13436, 2025.
- [29] H. Liu, W. Yan, M. Zaharia, and P. Abbeel, “World model on million-length video and language with blockwise ringattention,” arXiv preprint arXiv:2402.08268, 2024.
- [30] C. Team, “Chameleon: Mixed-modal early-fusion foundation models,” arXiv preprint arXiv:2405.09818, 2024.
- [31] D. P. Kingma, M. Welling et al., “Auto-encoding variational bayes,” 2013.
- [32] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12873–12883.
- [33] Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang, “Generative multimodal models are incontext learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14398–14409.
- [34] R. Dong, C. Han, Y. Peng, Z. Qi, Z. Ge, J. Yang, L. Zhao, J. Sun, H. Zhou, H. Wei et al., “Dreamllm: Synergistic multimodal comprehension and creation,” arXiv preprint arXiv:2309.11499, 2023.
- [35] J. Zhu, X. Ding, Y. Ge, Y. Ge, S. Zhao, H. Zhao, X. Wang, and Y. Shan, “Vl-gpt: A generative pre-trained transformer for vision

- and language understanding and generation,” arXiv preprint arXiv:2312.09251, 2023.
- [36] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “Eva-clip: Improved training techniques for clip at scale,” arXiv preprint arXiv:2303.15389, 2023.
- [37] C. Zhao, Y. Song, W. Wang, H. Feng, E. Ding, Y. Sun, X. Xiao, and J. Wang, “Monoformer: One transformer for both diffusion and autoregression,” arXiv preprint arXiv:2409.16280, 2024.
- [38] C. Zhou, L. Yu, A. Babu, K. Tirumala, M. Yasunaga, L. Shamis, J. Kahn, X. Ma, L. Zettlemoyer, and O. Levy, “Transfusion: Predict the next token and diffuse images with one multi-modal model,” arXiv preprint arXiv:2408.11039, 2024.
- [39] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou, “Show-o: One single transformer to unify multimodal understanding and generation,” arXiv preprint arXiv:2408.12528, 2024.
- [40] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., “A survey of large language models,” arXiv preprint arXiv:2303.18223, vol. 1, no. 2, 2023.
- [41] Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang et al., “A survey on evaluation of large language models,” ACM transactions on intelligent systems and technology, vol. 15, no. 3, pp. 1–45, 2024.
- [42] Z. Liang, Y. Xu, Y. Hong, P. Shang, Q. Wang, Q. Fu, and K. Liu, “A survey of multimodel large language models,” in Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering, 2024, pp. 405–409.
- [43] C. Cui, Y. Ma, X. Cao, W. Ye, Y. Zhou, K. Liang, J. Chen, J. Lu, Z. Yang, K.-D. Liao et al., “A survey on multimodal large language models for autonomous driving,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 958–979.
- [44] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,” National Science Review, vol. 11, no. 12, 2024.
- [45] F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, “Diffusion models in vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10850–10869, 2023.
- [46] F. Bie, Y. Yang, Z. Zhou, A. Ghanem, M. Zhang, Z. Yao, X. Wu, C. Holmes, P. Golnari, D. A. Clifton et al., “Renaissance: A survey into ai text-to-image generation in the era of large model,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [47] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,” National Science Review, vol. 11, no. 12, p. nwae403, 11 2024.
- [48] K. Carolan, L. Fennelly, and A. F. Smeaton, “A review of multi-modal large language and vision models,” arXiv preprint

- arXiv:2404.01322, 2024.

[49] F. Bordes, R. Y. Pang, A. Ajay, A. C. Li, A. Bardes, S. Petryk, O. Manas,˜ Z. Lin, A. Mahmoud, B. Jayaraman et al., “An introduction to vision-language modeling,” arXiv preprint

- arXiv:2405.17247, 2024.

- [50] I. Hartsock and G. Rasool, “Vision-language models for medical report generation and visual question answering: A review,” Frontiers in Artificial Intelligence, vol. 7, p. 1430984, 2024.
- [51] Z. Li, X. Wu, H. Du, F. Liu, H. Nghiem, and G. Shi, “A survey of state of the art large vision language models: Alignment, benchmark, evaluations and challenges.”
- [52] Z. Li, J. Zhang, D. Wang, Y. Wang, X. Huang, and Z. Wei, “Continuous or discrete, that is the question: A survey on large multi-modal models from the perspective of input-output space extension,” 2024.
- [53] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19730–19742.
- [54] J. Lu, D. Batra, D. Parikh, and S. Lee, “Vilbert: Pretraining taskagnostic visiolinguistic representations for vision-and-language tasks,” Advances in neural information processing systems, vol. 32, 2019.
- [55] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang, “Visualbert: A simple and performant baseline for vision and language,” arXiv preprint arXiv:1908.03557, 2019.
- [56] Y.-C. Chen, L. Li, L. Yu, A. El Kholy, F. Ahmed, Z. Gan, Y. Cheng, and J. Liu, “Uniter: Universal image-text representation learning,” in European conference on computer vision. Springer, 2020, pp. 104–120.

- [57] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.
- [58] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez et al., “Vicuna: An opensource chatbot impressing gpt-4 with 90%* chatgpt quality,” See https://vicuna. lmsys. org (accessed 14 April 2023), vol. 2, no. 3, p. 6, 2023.
- [59] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma et al., “Scaling instructionfinetuned language models,” Journal of Machine Learning Research, vol. 25, no. 70, pp. 1–53, 2024.
- [60] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23716–23736, 2022.
- [61] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark et al., “Training compute-optimal large language models,” arXiv preprint arXiv:2203.15556, 2022.
- [62] OpenAI, “Gpt-4v(ision) system card,” URL: https://openai.com/index/gpt-4v-system-card/, 2023.
- [63] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.
- [64] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26296–26306.
- [65] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llavanext: Improved reasoning, ocr, and world knowledge,” 2024.
- [66] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,” arXiv preprint arXiv:2404.16821, 2024.
- [67] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.
- [68] Z. Wu, X. Chen, Z. Pan, X. Liu, W. Liu, D. Dai, H. Gao, Y. Ma, C. Wu, B. Wang et al., “Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding,” arXiv preprint arXiv:2412.10302, 2024.
- [69] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International conference on machine learning. pmlr, 2015, pp. 2256–2265.
- [70] P. Cao, F. Zhou, Q. Song, and L. Yang, “Controllable generation with text-to-image diffusion models: A survey,” arXiv preprint arXiv:2403.04279, 2024.
- [71] A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” in International Conference on Machine Learning. PMLR, 2022, pp. 16784–16804.
- [72] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in neural information processing systems, vol. 35, pp. 36479–36494, 2022.
- [73] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, and B. Guo, “Vector quantized diffusion model for text-to-image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10696–10706.
- [74] StableAI, “Stable diffusion 2.0 release,” URL: https://stability.ai/news/stable-diffusion-v2-release, 2023.
- [75] ——, “Stable diffusion v2.1,” URL: https://stablediffusionxl.com/, 2023.
- [76] W. Li, X. Xu, X. Xiao, J. Liu, H. Yang, G. Li, Z. Wang, Z. Feng, Q. She, Y. Lyu et al., “Upainting: Unified text-to-image diffusion generation with cross-modal guidance,” arXiv preprint arXiv:2210.16031, 2022.
- [77] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think,” arXiv preprint arXiv:2410.06940, 2024.

- [78] Y. Lee, Y.-J. Lee, and S. J. Hwang, “Dit-pruner: Pruning diffusion transformer models for text-to-image synthesis using human preference scores,” in European Conference on Computer Vision (ECCV) 2024, 2024, pp. 1–9.
- [79] H. Li, Y. Zou, Y. Wang, O. Majumder, Y. Xie, R. Manmatha, A. Swaminathan, Z. Tu, S. Ermon, and S. Soatto, “On the scalability of diffusion-based text-to-image generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9400–9409.
- [80] J. Chen, C. Ge, E. Xie, Y. Wu, L. Yao, X. Ren, Z. Wang, P. Luo, H. Lu, and Z. Li, “Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation,” in European Conference on Computer Vision. Springer, 2024, pp. 74–91.
- [81] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and visionlanguage representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.
- [82] X. Zhang, L. Yang, Y. Cai, Z. Yu, K.-N. Wang, Y. Tian, M. Xu, Y. Tang, Y. Yang, B. Cui et al., “Realcompo: Balancing realism and compositionality improves text-to-image diffusion models,” Advances in Neural Information Processing Systems, vol. 37, pp. 96963–96992, 2024.
- [83] K. Wang, D. Tang, W. Zhao, K. Schurholt,¨ Z. Wang, and Y. You, “Recurrent diffusion for large-scale parameter generation,” arXiv preprint arXiv:2501.11587, 2025.
- [84] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu, “Omnigen: Unified image generation,” arXiv preprint arXiv:2409.11340, 2024.
- [85] X. Chen, Z. Zhang, H. Zhang, Y. Zhou, S. Y. Kim, Q. Liu, Y. Li, J. Zhang, N. Zhao, Y. Wang et al., “Unireal: Universal image generation and editing via learning real-world dynamics,” arXiv preprint arXiv:2412.07774, 2024.
- [86] Z. Wang, A. Li, Z. Li, and X. Liu, “Genartist: Multimodal llm as an agent for unified image generation and editing,” Advances in Neural Information Processing Systems, vol. 37, pp. 128374–128395, 2024.
- [87] T.-J. Fu, Y. Qian, C. Chen, W. Hu, Z. Gan, and Y. Yang, “Univg: A generalist diffusion model for unified image generation and editing,” arXiv preprint arXiv:2503.12652, 2025.
- [88] A. Van Den Oord, N. Kalchbrenner, and K. Kavukcuoglu, “Pixel recurrent neural networks,” in International conference on machine learning. PMLR, 2016, pp. 1747–1756.
- [89] A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves et al., “Conditional image generation with pixelcnn decoders,” Advances in neural information processing systems, vol. 29, 2016.
- [90] T. Salimans, A. Karpathy, X. Chen, and D. P. Kingma, “Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications,” arXiv preprint arXiv:1701.05517, 2017.
- [91] S. Reed, A. Oord, N. Kalchbrenner, S. G. Colmenarejo, Z. Wang, Y. Chen, D. Belov, and N. Freitas, “Parallel multiscale autoregressive density estimation,” in International conference on machine learning. PMLR, 2017, pp. 2912–2921.
- [92] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.
- [93] A. Razavi, A. Van den Oord, and O. Vinyals, “Generating diverse high-fidelity images with vq-vae-2,” Advances in neural information processing systems, vol. 32, 2019.
- [94] J. Yu, X. Li, J. Y. Koh, H. Zhang, R. Pang, J. Qin, A. Ku, Y. Xu, J. Baldridge, and Y. Wu, “Vector-quantized image modeling with improved vqgan,” arXiv preprint arXiv:2110.04627, 2021.
- [95] S. Cao, Y. Yin, L. Huang, Y. Liu, X. Zhao, D. Zhao, and K. Huang, “Efficient-vqgan: Towards high-resolution image generation with efficient vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7368–7377.
- [96] Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.-C. Chen, “An image is worth 32 tokens for reconstruction and generation,” Advances in Neural Information Processing Systems, vol. 37, pp. 128940–128966, 2024.
- [97] L. Zhu, F. Wei, Y. Lu, and D. Chen, “Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%,” arXiv preprint arXiv:2406.11837, 2024.
- [98] J. Guo, Z. Hao, C. Wang, Y. Tang, H. Wu, H. Hu, K. Han, and

- C. Xu, “Data-efficient large vision models through sequential autoregression,” arXiv preprint arXiv:2402.04841, 2024.
- [99] V. T. Hu, S. A. Baumann, M. Gui, O. Grebenkova, P. Ma, J. Fischer, and B. Ommer, “Zigma: Zigzag mamba diffusion model,” arXiv e-prints, pp. arXiv–2403, 2024.
- [100] Y. Teng, Y. Wu, H. Shi, X. Ning, G. Dai, Y. Wang, Z. Li, and X. Liu, “Dim: Diffusion mamba for efficient high-resolution image synthesis,” arXiv preprint arXiv:2405.14224, 2024.
- [101] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.
- [102] Y. Qi, F. Yang, Y. Zhu, Y. Liu, L. Wu, R. Zhao, and W. Li, “Exploring stochastic autoregressive image modeling for visual representation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 2, 2023, pp. 2074–2081.
- [103] Z. Pang, T. Zhang, F. Luan, Y. Man, H. Tan, K. Zhang, W. T. Freeman, and Y.-X. Wang, “Randar: Decoder-only autoregressive visual generation in random orders,” arXiv preprint arXiv:2412.01827, 2024.
- [104] Q. Yu, J. He, X. Deng, X. Shen, and L.-C. Chen, “Randomized autoregressive visual generation,” arXiv preprint arXiv:2411.00776, 2024.
- [105] W. Liu, L. Zhuo, Y. Xin, S. Xia, P. Gao, and X. Yue, “Customize your visual autoregressive recipe with set autoregressive modeling,” arXiv preprint arXiv:2410.10511, 2024.
- [106] K. E. Ak, N. Xu, Z. Lin, and Y. Wang, “Incorporating reinforced adversarial learning in autoregressive image generation,” in European conference on computer vision. Springer, 2020, pp. 18–34.
- [107] P. Esser, R. Rombach, A. Blattmann, and B. Ommer, “Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis,” Advances in neural information processing systems, vol. 34, pp. 3518–3532, 2021.
- [108] Y. Xu, G. Corso, T. Jaakkola, A. Vahdat, and K. Kreis, “Disco-diff: Enhancing continuous diffusion models with discrete latents,” arXiv preprint arXiv:2407.03300, 2024.
- [109] Y. Pang, P. Jin, S. Yang, B. Lin, B. Zhu, Z. Tang, L. Chen, F. E. Tay, S.-N. Lim, H. Yang et al., “Next patch prediction for autoregressive visual generation,” arXiv preprint arXiv:2412.15321, 2024.
- [110] S. Ren, S. Ma, X. Sun, and F. Wei, “Next block prediction: Video generation via semi-auto-regressive modeling,” arXiv preprint arXiv:2502.07737, 2025.
- [111] Y. He, Y. He, S. He, F. Chen, H. Zhou, K. Zhang, and B. Zhuang, “Neighboring autoregressive modeling for efficient visual generation,” arXiv preprint arXiv:2503.10696, 2025.
- [112] Y. Wang, S. Ren, Z. Lin, Y. Han, H. Guo, Z. Yang, D. Zou, J. Feng, and X. Liu, “Parallelized autoregressive visual generation,” arXiv preprint arXiv:2412.15119, 2024.
- [113] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual autoregressive modeling: Scalable image generation via nextscale prediction,” Advances in neural information processing systems, vol. 37, pp. 84839–84865, 2024.
- [114] S. Ren, Q. Yu, J. He, X. Shen, A. Yuille, and L.-C. Chen, “Flowar: Scale-wise autoregressive image generation meets flow matching,” arXiv preprint arXiv:2412.15205, 2024.
- [115] S. Ren, Y. Yu, N. Ruiz, F. Wang, A. Yuille, and C. Xie, “M-var: Decoupled scale-wise autoregressive modeling for high-quality image generation,” arXiv preprint arXiv:2411.10433, 2024.
- [116] H. Guo, Y. Li, T. Zhang, J. Wang, T. Dai, S.-T. Xia, and L. Benini, “Fastvar: Linear visual autoregressive modeling via cached token pruning,” arXiv preprint arXiv:2503.23367, 2025.
- [117] S. Jiao, G. Zhang, Y. Qian, J. Huang, Y. Zhao, H. Shi, L. Ma, Y. Wei, and Z. Jie, “Flexvar: Flexible visual autoregressive modeling without residual prediction,” arXiv preprint arXiv:2502.20313, 2025.
- [118] H. Yu, H. Luo, H. Yuan, Y. Rong, and F. Zhao, “Frequency autoregressive image generation with continuous tokens,” arXiv preprint arXiv:2503.05305, 2025.
- [119] Z. Huang, X. Qiu, Y. Ma, Y. Zhou, C. Zhang, and X. Li, “Nfig: Autoregressive image generation with next-frequency prediction,” arXiv preprint arXiv:2503.07076, 2025.
- [120] S. Ren, Q. Yu, J. He, X. Shen, A. Yuille, and L.-C. Chen, “Beyond next-token: Next-x prediction for autoregressive visual generation,” arXiv preprint arXiv:2502.20388, 2025.
- [121] Z. Li, T. Cheng, S. Chen, P. Sun, H. Shen, L. Ran, X. Chen, W. Liu, and X. Wang, “Controlar: Controllable image generation with autoregressive models,” arXiv preprint arXiv:2410.02705, 2024.

- [122] X. Li, K. Qiu, H. Chen, J. Kuen, Z. Lin, R. Singh, and B. Raj, “Controlvar: Exploring controllable visual autoregressive modeling,” arXiv preprint arXiv:2406.09750, 2024.
- [123] Z. Yao, J. Li, Y. Zhou, Y. Liu, X. Jiang, C. Wang, F. Zheng, Y. Zou, and L. Li, “Car: Controllable autoregressive modeling for visual generation,” arXiv preprint arXiv:2410.04671, 2024.
- [124] Y. Shen, Y. Zhang, S. Zhai, L. Huang, J. M. Susskind, and J. Gu, “Many-to-many image generation with auto-regressive diffusion models,” arXiv preprint arXiv:2404.03109, 2024.
- [125] B. Cardenas, D. Arya, and D. K. Gupta, “Generating annotated high-fidelity images containing multiple coherent objects,” in 2021 IEEE International Conference on Image Processing (ICIP). IEEE, 2021, pp. 834–838.
- [126] S. Ren, X. Huang, X. Li, J. Xiao, J. Mei, Z. Wang, A. Yuille, and Y. Zhou, “Medical vision generalist: Unifying medical imaging tasks in context,” arXiv preprint arXiv:2406.05565, 2024.
- [127] Z. Li, H. Li, Y. Shi, A. B. Farimani, Y. Kluger, L. Yang, and P. Wang, “Dual diffusion for unified image generation and understanding,” arXiv preprint arXiv:2501.00289, 2024.
- [128] A. Swerdlow, M. Prabhudesai, S. Gandhi, D. Pathak, and K. Fragkiadaki, “Unified multimodal discrete diffusion,” arXiv:2503.20853, 2025.
- [129] L. Yang, Y. Tian, B. Li, X. Zhang, K. Shen, Y. Tong, and M. Wang, “Mmada: Multimodal large diffusion language models,” arXiv preprint arXiv:2505.15809, 2025.
- [130] J. Wang, Y. Lai, A. Li, S. Zhang, J. Sun, N. Kang, C. Wu, Z. Li, and P. Luo, “Fudoki: Discrete flow-based unified understanding and generation via kinetic-optimal velocities,” arXiv:2505.20147, 2025.
- [131] Q. Shi, J. Bai, Z. Zhao, W. Chai, K. Yu, J. Wu, S. Song, Y. Tong, X. Li, X. Li et al., “Muddit: Liberating generation beyond text-toimage with a unified discrete diffusion model,” arXiv:2505.23606, 2025.
- [132] S. Li, J. Gu, K. Liu, Z. Lin, Z. Wei, A. Grover, and J. Kuen, “Lavidao: Elastic large masked diffusion models for unified multimodal understanding and generation,” arXiv preprint arXiv:2509.19244, 2025.
- [133] C. Zhang, J. Wang, Y. Wang, Y. Liang, X. Yang, Z. Li, H. Huang, and X. Li, “Unimodel: A visual-only framework for unified multimodal understanding and generation,” arXiv preprint arXiv:2511.16917, 2025.
- [134] E. Chern, J. Su, Y. Ma, and P. Liu, “Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation,” arXiv preprint arXiv:2407.06135, 2024.
- [135] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu et al., “Emu3: Next-token prediction is all you need,” arXiv preprint arXiv:2409.18869, 2024.
- [136] J. Yang, D. Yin, Y. Zhou, F. Rao, W. Zhai, Y. Cao, and Z.-J. Zha, “Mmar: Towards lossless multi-modal auto-regressive probabilistic modeling,” arXiv preprint arXiv:2410.10798, 2024.
- [137] S. Kou, J. Jin, C. Liu, Y. Ma, J. Jia, Q. Chen, P. Jiang, and Z. Deng, “Orthus: Autoregressive interleaved image-text generation with modality-specific heads,” arXiv preprint arXiv:2412.00127, 2024.
- [138] H. Li, C. Tian, J. Shao, X. Zhu, Z. Wang, J. Zhu, W. Dou, X. Wang, H. Li, L. Lu et al., “Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding,” arXiv preprint arXiv:2412.09604, 2024.
- [139] J. Wu, Y. Jiang, C. Ma, Y. Liu, H. Zhao, Z. Yuan, S. Bai, and X. Bai, “Liquid: Language models are scalable multi-modal generators,” arXiv preprint arXiv:2412.04332, 2024.
- [140] H. Tang, H. Liu, and X. Xiao, “Ugen: Unified autoregressive multimodal model with progressive vocabulary learning,” arXiv preprint arXiv:2503.21193, 2025.
- [141] S. Wu, W. Zhang, L. Xu, S. Jin, Z. Wu, Q. Tao, W. Liu, W. Li, and C. C. Loy, “Harmonizing visual representations for unified multimodal understanding and generation,” arXiv preprint arXiv:2503.21979, 2025.
- [142] H. Lin, T. Wang, Y. Ge, Y. Ge, Z. Lu, Y. Wei, Q. Zhang, Z. Sun, and Y. Shan, “Toklip: Marry visual tokens to clip for multimodal comprehension and generation,” arXiv preprint arXiv:2505.05422, 2025.
- [143] B. Wang, Z. Yue, F. Zhang, S. Chen, L. Bi, J. Zhang, X. Song, K. Y. Chan, J. Pan, W. Wu et al., “Discrete visual tokens of autoregression, by diffusion, and for reasoning,” arXiv preprint arXiv:2505.07538, 2025.
- [144] H. Li, X. Peng, Y. Wang, Z. Peng, X. Chen, R. Weng, J. Wang, X. Cai, W. Dai, and H. Xiong, “Onecat: Decoder-only auto-

- regressive model for unified understanding and generation,” arXiv preprint arXiv:2509.03498, 2025.
- [145] Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Zhang, Y. Wang, H. Gao, J. Liu, T. Huang, and X. Wang, “Emu: Generative pretraining in multimodality,” in The Twelfth International Conference on Learning Representations, 2024.
- [146] Y. Jin, K. Xu, L. Chen, C. Liao, J. Tan, Q. Huang, B. Chen, C. Lei, A. Liu, C. Song et al., “Unified language-vision pretraining in llm with dynamic discrete visual tokenization,” arXiv preprint arXiv:2309.04669, 2023.
- [147] C. Tian, X. Zhu, Y. Xiong, W. Wang, Z. Chen, W. Wang, Y. Chen, L. Lu, T. Lu, J. Zhou et al., “Mm-interleaved: Interleaved imagetext generative modeling via multi-modal feature synchronizer,” arXiv preprint arXiv:2401.10208, 2024.
- [148] Y. Li, Y. Zhang, C. Wang, Z. Zhong, Y. Chen, R. Chu, S. Liu, and J. Jia, “Mini-gemini: Mining the potential of multi-modality vision language models,” arXiv preprint arXiv:2403.18814, 2024.
- [149] Y. Wu, Z. Zhang, J. Chen, H. Tang, D. Li, Y. Fang, L. Zhu, E. Xie, H. Yin, L. Yi et al., “Vila-u: a unified foundation model integrating visual understanding and generation,” arXiv preprint

- arXiv:2409.04429, 2024.

[150] R. Fang, C. Duan, K. Wang, H. Li, H. Tian, X. Zeng, R. Zhao, J. Dai, H. Li, and X. Liu, “Puma: Empowering unified mllm with multi-granular visual generation,” arXiv preprint

- arXiv:2410.13861, 2024.

- [151] S. Tong, D. Fan, J. Zhu, Y. Xiong, X. Chen, K. Sinha, M. Rabbat, Y. LeCun, S. Xie, and Z. Liu, “Metamorph: Multimodal understanding and generation via instruction tuning,” arXiv preprint arXiv:2412.14164, 2024.
- [152] C. Wang, G. Lu, J. Yang, R. Huang, J. Han, L. Hou, W. Zhang, and H. Xu, “Illume: Illuminating your llms to see, draw, and selfenhance,” arXiv preprint arXiv:2412.06673, 2024.
- [153] C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi, “Unitok: A unified tokenizer for visual generation and understanding,” arXiv preprint arXiv:2502.20321, 2025.
- [154] Y. Zhao, F. Xue, S. Reed, L. Fan, Y. Zhu, J. Kautz, Z. Yu, P. Kr¨ahenbuhl,¨ and D.-A. Huang, “Qlip: Text-aligned visual tokenization unifies auto-regressive multimodal understanding and generation,” arXiv preprint arXiv:2502.05178, 2025.
- [155] W. Song, Y. Wang, Z. Song, Y. Li, H. Sun, W. Chen, Z. Zhou, J. Xu, J. Wang, and K. Yu, “Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies,” arXiv preprint arXiv:2503.14324, 2025.
- [156] T. Li, Q. Lu, L. Zhao, H. Li, X. Zhu, Y. Qiao, J. Zhang, and W. Shao, “Unifork: Exploring modality alignment for unified multimodal understanding and generation,” arXiv preprint arXiv:2506.17202, 2025.
- [157] Y. chieh Chan, H. Zhong, Y. Li, and Z. Yang, “Unicode²: Cascaded large-scale codebooks for unified multimodal understanding and generation,” arXiv preprint arXiv:2506.20214, 2025.
- [158] B. Lin, Z. Li, X. Cheng, Y. Niu, Y. Ye, X. He, S. Yuan, W. Yu, S. Wang, Y. Ge et al., “Uniworld: High-resolution semantic encoders for unified visual understanding and generation,” arXiv preprint arXiv:2506.03147, 2025.
- [159] Z. Xu, J. Chen, Z. Lin, X. Pan, L. Huang, T. Zhou, M. Khabsa, Q. Wang, D. Jin, M. Yasunaga et al., “Pisces: An auto-regressive foundation model for image understanding and generation,” arXiv preprint arXiv:2506.10395, 2025.
- [160] J. Han, H. Chen, Y. Zhao, H. Wang, Q. Zhao, Z. Yang, H. He, X. Yue, and L. Jiang, “Vision as a dialect: Unifying visual understanding and generation via text-aligned representations,” arXiv preprint arXiv:2506.18898, 2025.
- [161] C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou et al., “Omnigen2: Exploration to advanced multimodal generation,” arXiv preprint arXiv:2506.18871, 2025.
- [162] G.-H. Wang, S. Zhao, X. Zhang, L. Cao, P. Zhan, L. Duan, S. Lu, M. Fu, X. Chen, J. Zhao et al., “Ovis-u1 technical report,” arXiv preprint arXiv:2506.23044, 2025.
- [163] Z. Geng, Y. Wang, Y. Ma, C. Li, Y. Rao, S. Gu, Z. Zhong, Q. Lu, H. Hu, X. Zhang et al., “X-omni: Reinforcement learning makes discrete autoregressive image generative models great again,” arXiv preprint arXiv:2507.22058, 2025.
- [164] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.

- [165] H. Lin, J. Cho, A. Zadeh, C. Li, and M. Bansal, “Bifrost-1: Bridging multimodal llms and diffusion models with patch-level clip latents,” arXiv preprint arXiv:2508.05954, 2025.
- [166] Z. Huang, D. Zheng, C. Zou, R. Liu, X. Wang, K. Ji, W. Chai, J. Sun, L. Wang, Y. Lv et al., “Ming-univision: Joint image understanding and generation with a unified continuous tokenizer,” arXiv preprint arXiv:2510.06590, 2025.
- [167] T. Shen, X. Wan, T. Chen, R. Zhang, J. Pan, D. Lu, F. Lei, Z. Lu, Y. Yang, C. Cheng et al., “Mammothmoda2: A unified ar-diffusion framework for multimodal understanding and generation,” arXiv preprint arXiv:2511.18262, 2025.
- [168] Y. Ge, Y. Ge, Z. Zeng, X. Wang, and Y. Shan, “Planting a seed of vision in large language model,” arXiv preprint arXiv:2307.08041, 2023.
- [169] Y. Ge, S. Zhao, Z. Zeng, Y. Ge, C. Li, X. Wang, and Y. Shan, “Making llama see and draw with seed tokenizer,” arXiv preprint arXiv:2310.01218, 2023.
- [170] Y. Ge, S. Zhao, C. Li, Y. Ge, and Y. Shan, “Seed-data-edit technical report: A hybrid dataset for instructional image editing,” arXiv preprint arXiv:2405.04007, 2024.
- [171] X. Pan, S. N. Shukla, A. Singh, Z. Zhao, S. K. Mishra, J. Wang, Z. Xu, J. Chen, K. Li, F. Juefei-Xu et al., “Transfer between modalities with metaqueries,” arXiv preprint arXiv:2504.06256, 2025.
- [172] H. Zhang, Z. Duan, X. Wang, Y. Chen, Y. Zhao, and Y. Zhang, “Nexus-gen: A unified model for image understanding, generation, and editing,” arXiv preprint arXiv:2504.21356, 2025.
- [173] A. G. Inclusion AI, “Ming-lite-uni: Advancements in unified architecture for natural multimodal interaction,” arXiv preprint, 2025.
- [174] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese et al., “Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset,” arXiv preprint arXiv:2505.09568, 2025.
- [175] S. Wu, Z. Wu, Z. Gong, Q. Tao, S. Jin, Q. Li, W. Li, and C. C. Loy, “Openuni: A simple baseline for unified multimodal understanding and generation,” arXiv preprint arXiv:2505.23661, 2025.
- [176] H. Tang, C. Xie, X. Bao, T. Weng, P. Li, Y. Zheng, and L. Wang, “Unilip: Adapting clip for unified multimodal understanding, generation and editing,” arXiv preprint arXiv:2507.23278, 2025.
- [177] J. Xu, Y. Yin, and X. Chen, “Tbac-uniimage: Unified understanding and generation by ladder-side diffusion tuning,” arXiv preprint arXiv:2508.08098, 2025.
- [178] H. Wei, B. Xu, H. Liu, C. Wu, J. Liu, Y. Peng, P. Wang, Z. Liu, J. He, Y. Xietian et al., “Skywork unipic 2.0: Building kontext model with online rl for unified multimodal model,” arXiv preprint arXiv:2509.04548, 2025.
- [179] C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan et al., “Janus: Decoupling visual encoding for unified multimodal understanding and generation,” arXiv preprint arXiv:2410.13848, 2024.
- [180] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, “Janus-pro: Unified multimodal understanding and generation with data and model scaling,” arXiv preprint arXiv:2501.17811, 2025.
- [181] J. Zou, B. Liao, Q. Zhang, W. Liu, and X. Wang, “Omnimamba: Efficient and unified multimodal understanding and generation via state space models,” arXiv preprint arXiv:2503.08686, 2025.
- [182] L. Fan, T. Li, S. Qin, Y. Li, C. Sun, M. Rubinstein, D. Sun, K. He, and Y. Tian, “Fluid: Scaling autoregressive text-toimage generative models with continuous tokens,” arXiv preprint arXiv:2410.13863, 2024.
- [183] Y. Xiao, L. Song, Y. Chen, Y. Luo, Y. Chen, Y. Gan, W. Huang, X. Li, X. Qi, and Y. Shan, “Mindomni: Unleashing reasoning generation in vision language models with rgpo,” arXiv preprint arXiv:2505.13031, 2025.
- [184] P. Wang, Y. Peng, Y. Gan, L. Hu, T. Xie, X. Wang, Y. Wei, C. Tang, B. Zhu, C. Li et al., “Skywork unipic: Unified autoregressive modeling for visual understanding and generation,” arXiv preprint arXiv:2508.03320, 2025.
- [185] R. Xie, C. Du, P. Song, and C. Liu, “Muse-vl: Modeling unified vlm through semantic discrete encoding,” arXiv preprint

- arXiv:2411.17762, 2024.

[186] L. Qu, H. Zhang, Y. Liu, X. Wang, Y. Jiang, Y. Gao, H. Ye, D. K. Du, Z. Yuan, and X. Wu, “Tokenflow: Unified image tokenizer for multimodal understanding and generation,” arXiv preprint

- arXiv:2412.03069, 2024.

- [187] X. Zhuang, Y. Xie, Y. Deng, L. Liang, J. Ru, Y. Yin, and Y. Zou, “Vargpt: Unified understanding and generation in a visual autoregressive multimodal large language model,” arXiv preprint arXiv:2501.12327, 2025.
- [188] Z. Chen, C. Wang, X. Chen, H. Xu, R. Huang, J. Zhou, J. Han, H. Xu, and X. Liang, “Semhitok: A unified image tokenizer via semantic-guided hierarchical codebook for multimodal understanding and generation,” arXiv preprint arXiv:2503.06764, 2025.
- [189] X. Zhuang, Y. Xie, Y. Deng, D. Yang, L. Liang, J. Ru, Y. Yin, and Y. Zou, “Vargpt-v1. 1: Improve visual autoregressive large unified model via iterative instruction tuning and reinforcement learning,” arXiv preprint arXiv:2504.02949, 2025.
- [190] R. Huang, C. Wang, J. Yang, G. Lu, Y. Yuan, J. Han, L. Hou, W. Zhang, L. Hong, H. Zhao et al., “ILLUME+: Illuminating unified mllm with dual visual tokenization and diffusion refinement,” arXiv preprint arXiv:2504.01934, 2025.
- [191] Y. Jiao, H. Qiu, Z. Jie, S. Chen, J. Chen, L. Ma, and Y.G. Jiang, “Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding,” arXiv preprint arXiv:2504.04423, 2025.
- [192] J. Xie, Z. Yang, and M. Z. Shou, “Show-o2: Improved native unified multimodal models,” arXiv preprint arXiv:2506.15564, 2025.
- [193] W. Shi, X. Han, C. Zhou, W. Liang, X. V. Lin, L. Zettlemoyer, and L. Yu, “Llamafusion: Adapting pretrained language models for multimodal generation,” arXiv preprint arXiv:2412.15188, 2024.
- [194] Z. Liu, W. Ren, H. Liu, Z. Zhou, S. Chen, H. Qiu, X. Huang, Z. An, F. Yang, A. Patel et al., “Tuna: Taming unified visual representations for native unified multimodal models,” arXiv preprint arXiv:2512.02014, 2025.
- [195] Y. Ma, X. Liu, X. Chen, W. Liu, C. Wu, Z. Wu, Z. Pan, Z. Xie, H. Zhang, L. Zhao et al., “Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation,” arXiv preprint arXiv:2411.07975, 2024.
- [196] C. Liao, L. Liu, X. Wang, Z. Luo, X. Zhang, W. Zhao, J. Wu, L. Li, Z. Tian, and W. Huang, “Mogao: An omni foundation model for interleaved multi-modal generation,” arXiv preprint arXiv:2505.05472, 2025.
- [197] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, G. Shi, and H. Fan, “Emerging properties in unified multimodal pretraining,” arXiv preprint arXiv:2505.14683, 2025.
- [198] Z. Wang, Z. Chen, C. Gou, F. Li, C. Deng, D. Zhu, K. Li, W. Yu, H. Tu, H. Fan et al., “Lightfusion: A light-weighted, double fusion framework for unified multimodal understanding and generation,” arXiv preprint arXiv:2510.22946, 2025.
- [199] X. Wang, Z. Zhang, H. Zhang, Z. Lin, Y. Zhou, Q. Liu, S. Zhang, Y. Li, S. Liu, H. Zheng et al., “Hbridge: H-shape bridging of heterogeneous experts for unified multimodal understanding and generation,” arXiv preprint arXiv:2511.20520, 2025.
- [200] X. He, L. Wei, J. Ouyang, L. Xie, and Q. Tian, “Emma: Efficient multimodal understanding, generation, and editing with a unified architecture,” arXiv preprint arXiv:2512.04810, 2025.
- [201] S. Wu, H. Fei, L. Qu, W. Ji, and T.-S. Chua, “Next-gpt: Any-to-any multimodal llm,” in Forty-first International Conference on Machine Learning, 2024.
- [202] J. Lu, C. Clark, S. Lee, Z. Zhang, S. Khosla, R. Marten, D. Hoiem, and A. Kembhavi, “Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26439–26455.
- [203] Y. Jin, Z. Sun, K. Xu, L. Chen, H. Jiang, Q. Huang, C. Song, Y. Liu, D. Zhang, Y. Song et al., “Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization,” arXiv preprint arXiv:2402.03161, 2024.
- [204] J. Zhan, J. Dai, J. Ye, Y. Zhou, D. Zhang, Z. Liu, X. Zhang, R. Yuan, G. Zhang, L. Li et al., “Anygpt: Unified multimodal llm with discrete sequence modeling,” arXiv preprint arXiv:2402.12226, 2024.
- [205] H. Ye, D.-A. Huang, Y. Lu, Z. Yu, W. Ping, A. Tao, J. Kautz, S. Han, D. Xu, P. Molchanov et al., “X-vila: Cross-modality alignment for large language model,” arXiv preprint arXiv:2405.19335, 2024.
- [206] Z. Wang, K. Zhu, C. Xu, W. Zhou, J. Liu, Y. Zhang, J. Wang, N. Shi, S. Li, Y. Li et al., “Mio: A foundation model on multimodal tokens,” arXiv preprint arXiv:2409.17692, 2024.
- [207] J. Lai, J. Zhang, J. Liu, J. Li, X. Lu, and S. Guo, “Spider: Any-tomany multimodal llm,” arXiv preprint arXiv:2411.09439, 2024.

- [208] S. Li, K. Kallidromitis, A. Gokul, Z. Liao, Y. Kato, K. Kozuka, and A. Grover, “Omniflow: Any-to-any generation with multi-modal rectified flows,” arXiv preprint arXiv:2412.01169, 2024.
- [209] Q. Guo, K. Song, Z. Feng, Z. Ma, Q. Zhang, S. Gao, X. Yu, Y. Sun, T.-W. Chang, J. Chen et al., “M2-omni: Advancing omni-mllm for comprehensive modality support with competitive performance,” arXiv preprint arXiv:2502.18778, 2025.
- [210] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.
- [211] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” in NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.
- [212] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in International conference on machine learning. PMLR, 2021, pp. 8162–8171.
- [213] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021.
- [214] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps,” Advances in Neural Information Processing Systems, vol. 35, pp. 5775–5787, 2022.
- [215] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in ICCV, 2023.
- [216] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, V. Birodkar, A. Gupta, X. Gu et al., “Language model beats diffusion–tokenizer is key to visual generation,” in ICLR, 2024.
- [217] I. Gat, T. Remez, N. Shaul, F. Kreuk, R. T. Chen, G. Synnaeve, Y. Adi, and Y. Lipman, “Discrete flow matching,” in NeurIPS, 2024.
- [218] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11975–11986.
- [219] B. F. Labs, “Flux: Inference repository,” https://github.com/ black-forest-labs/flux, 2024, accessed: 2025-04-25.
- [220] J. Bai, T. Ye, W. Chow, E. Song, Q.-G. Chen, X. Li, Z. Dong, L. Zhu, and S. Yan, “Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis,” in ICLR, 2025.
- [221] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. ZHOU, Y. Lin, J.-R. Wen, and C. Li, “Large language diffusion models,” in ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2025.
- [222] L. Yu, Y. Cheng, K. Sohn, J. Lezama, H. Zhang, H. Chang, A. G. Hauptmann, M.-H. Yang, Y. Hao, I. Essa et al., “Magvit: Masked generative video transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10459–10469.
- [223] S. Li, K. Kallidromitis, H. Bansal, A. Gokul, Y. Kato, K. Kozuka, J. Kuen, Z. Lin, K.-W. Chang, and A. Grover, “Lavida: A large diffusion language model for multimodal understanding,” arXiv preprint arXiv:2505.16839, 2025.
- [224] H. Wei, Y. Sun, and Y. Li, “Deepseek-ocr: Contexts optical compression,” arXiv preprint arXiv:2510.18234, 2025.
- [225] I. Labs, S. Khanna, S. Kharbanda, S. Li, H. Varma, E. Wang, S. Birnbaum, Z. Luo, Y. Miraoui, A. Palrecha, S. Ermon, A. Grover, and V. Kuleshov, “Mercury: Ultra-fast language models based on diffusion,” arXiv:2506.17298, 2025.
- [226] Google Deepmind, “Gemini diffusion,” https://deepmind. google/models/gemini-diffusion, 2025, accessed: 2025-06-25.
- [227] G. H. Chen, S. Chen, R. Zhang, J. Chen, X. Wu, Z. Zhang, Z. Chen, J. Li, X. Wan, and B. Wang, “Allava: Harnessing gpt4vsynthesized data for a lite vision-language model,” 2024.
- [228] G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivi`ere, M. S. Kale, J. Love et al., “Gemma: Open models based on gemini research and technology,” arXiv preprint arXiv:2403.08295, 2024.
- [229] G. Team, M. Riviere, S. Pathak, P. G. Sessa, C. Hardin, S. Bhupatiraju, L. Hussenot, T. Mesnard, B. Shahriari, A. Ram´e et al., “Gemma 2: Improving open language models at a practical size,” arXiv preprint arXiv:2408.00118, 2024.
- [230] G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ram´e, M. Rivi`ere et al., “Gemma 3 technical report,” arXiv preprint arXiv:2503.19786, 2025.

- [231] C. Zheng, T.-L. Vuong, J. Cai, and D. Phung, “Movq: Modulating quantized vectors for high-fidelity image generation,” Advances in Neural Information Processing Systems, vol. 35, pp. 23412–23425, 2022.
- [232] A. Razzhigaev, A. Shakhmatov, A. Maltseva, V. Arkhipkin,

I. Pavlov, I. Ryabov, A. Kuts, A. Panchenko, A. Kuznetsov, and D. Dimitrov, “Kandinsky: An improved text-to-image synthesis with image prior and latent diffusion,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2023, pp. 286–295.

- [233] O. Gafni, A. Polyak, O. Ashual, S. Sheynin, D. Parikh, and Y. Taigman, “Make-a-scene: Scene-based text-to-image generation with human priors,” in European Conference on Computer Vision. Springer, 2022, pp. 89–106.
- [234] Y. Cui, H. Chen, H. Deng, X. Huang, X. Li, J. Liu, Y. Liu, Z. Luo, J. Wang, W. Wang et al., “Emu3. 5: Native multimodal models are world learners,” arXiv preprint arXiv:2510.26583, 2025.
- [235] J. Han, J. Liu, Y. Jiang, B. Yan, Y. Zhang, Z. Yuan, B. Peng, and X. Liu, “Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis,” arXiv preprint arXiv:2412.04431, 2024.
- [236] Y. Zhu, Z. Yanpeng, C. Wang, Y. Cao, J. Han, L. Hou, and H. Xu, “Unit: Unifying image and text recognition in one vision encoder,” Advances in Neural Information Processing Systems, vol. 37, pp. 122185–122205, 2024.
- [237] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu, “Omnigen: Unified image generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 13294–13304.
- [238] C. Wei, Q. Liu, Z. Ye, Q. Wang, X. Wang, P. Wan, K. Gai, and W. Chen, “Univideo: Unified understanding, generation, and editing for videos,” arXiv preprint arXiv:2510.08377, 2025.
- [239] Z. Tan, H. Yang, L. Qin, J. Gong, M. Yang, and H. Li, “Omnivideo: Democratizing unified video understanding and generation,” arXiv preprint arXiv:2507.06119, 2025.
- [240] L. Zhu, F. Wei, Y. Lu, and D. Chen, “Scaling the codebook size of vq-gan to 100,000 with a utilization rate of 99%,” in The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024.
- [241] B. Zheng, N. Ma, S. Tong, and S. Xie, “Diffusion transformers with representation autoencoders,” arXiv preprint arXiv:2510.11690, 2025.
- [242] S. Du, J. Guo, B. Li, S. Cui, Z. Xu, Y. Luo, Y. Wei, K. Gai, X. Wang, K. Wu et al., “Vqrae: Representation quantization autoencoders for multimodal understanding, generation and reconstruction,” arXiv preprint arXiv:2511.23386, 2025.
- [243] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller,¨ J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952, 2023.
- [244] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, “Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models,” arXiv preprint arXiv:2308.06721, 2023.
- [245] L. Zhuo, R. Du, X. Han, Y. Li, D. Liu, R. Huang, W. Liu et al., “Lumina-next: Making lumina-t2x stronger and faster with nextdit,” arXiv preprint arXiv:2406.18583, 2024.
- [246] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin et al., “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022.
- [247] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al., “Wan: Open and advanced largescale video generative models,” arXiv preprint arXiv:2503.20314, 2025.
- [248] J. Hu, S. Zhao, Q.-G. Chen, X. Qiu, J. Liu, Z. Xu, W. Luo, K. Zhang, and Y. Lu, “Omni-view: Unlocking how generation facilitates understanding in unified 3d model based on multiview images,” arXiv preprint arXiv:2511.07222, 2025.
- [249] J. Hu, Y. Yang, J. Liu, J. Wu, C. Zhao, and Y. Lu, “Autoregressively generating multi-view consistent images,” arXiv preprint arXiv:2506.18527, 2025.
- [250] J. Kong, J. Kim, and J. Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” Advances in neural information processing systems, vol. 33, pp. 17022–17033, 2020.
- [251] A. D´efossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural audio compression,” arXiv preprint arXiv:2210.13438, 2022.

- [252] X. Zhang, D. Zhang, S. Li, Y. Zhou, and X. Qiu, “Speechtokenizer: Unified speech tokenizer for speech large language models,” arXiv preprint arXiv:2308.16692, 2023.
- [253] M. Dehghani, B. Mustafa, J. Djolonga, J. Heek, M. Minderer, M. Caron, A. Steiner, J. Puigcerver, R. Geirhos, I. M. Alabdulmohsin et al., “Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution,” Advances in Neural Information Processing Systems, vol. 36, pp. 2252–2274, 2023.
- [254] Z. Gao, S. Zhang, I. McLoughlin, and Z. Yan, “Paraformer: Fast and accurate parallel transformer for non-autoregressive end-toend speech recognition,” in INTERSPEECH, 2022.
- [255] Z. Du, Q. Chen, S. Zhang, K. Hu, H. Lu, Y. Yang, H. Hu, S. Zheng, Y. Gu, Z. Ma et al., “Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens,” arXiv preprint arXiv:2407.05407, 2024.
- [256] I. AI, B. Gong, C. Zou, C. Zheng, C. Zhou, C. Yan, C. Jin, C. Shen, D. Zheng, F. Wang et al., “Ming-omni: A unified multimodal model for perception and generation,” arXiv preprint arXiv:2506.09344, 2025.
- [257] InclusionAI, “Ling-flash-2.0,” https://huggingface.co/ inclusionAI/Ling-flash-2.0, 2025, accessed: 2025-09-17.
- [258] I. AI, B. Ma, C. Zou, C. Yan, C. Jin, C. Shen, C. Lian, D. Zheng, F. Wang, F. Xu et al., “Ming-flash-omni: A sparse, unified architecture for multimodal perception and generation,” arXiv preprint arXiv:2510.24821, 2025.
- [259] J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, Y. Lv, Y. Wang, D. Guo, H. Wang, L. Ma, P. Zhang, X. Zhang, H. Hao, Z. Guo, B. Yang, B. Zhang, Z. Ma, X. Wei, S. Bai, K. Chen, X. Liu, P. Wang, M. Yang, D. Liu, X. Ren, B. Zheng, R. Men, F. Zhou, B. Yu, J. Yang, L. Yu, J. Zhou, and J. Lin, “Qwen3omni technical report,” arXiv preprint arXiv:2509.17765, 2025.
- [260] M. L. Team, B. Wang, B. Xiao, B. Zhang, B. Rong, B. Chen, C. Wan, C. Zhang, C. Huang, C. Chen et al., “Longcat-flash-omni technical report,” arXiv preprint arXiv:2511.00279, 2025.
- [261] K. Desai, G. Kaul, Z. Aysola, and J. Johnson, “Redcaps: Webcurated image-text data created by the people, for the people,” arXiv preprint arXiv:2111.11431, 2021.
- [262] J. Gu, X. Meng, G. Lu, L. Hou, N. Minzhe, X. Liang, L. Yao, R. Huang, W. Zhang, X. Jiang et al., “Wukong: A 100 million largescale chinese cross-modal pre-training benchmark,” Advances in Neural Information Processing Systems, vol. 35, pp. 26418–26431, 2022.
- [263] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25278–25294, 2022.
- [264] M. Byeon, B. Park, H. Kim, S. Lee, W. Baek, and S. Kim, “Coyo700m: Image-text pair dataset,” 2022.
- [265] C. Schuhmann, A. K¨opf, R. Vencu, T. Coombes, and R. Beaumont, “Laion coco: 600m synthetic captions from laion2b-en,” URL https://laion. ai/blog/laion-coco, vol. 5, 2022.
- [266] S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang et al., “Datacomp: In search of the next generation of multimodal datasets,” Advances in Neural Information Processing Systems, vol. 36, pp. 27092–27112, 2023.
- [267] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei, “Kosmos-2: Grounding multimodal large language models to the world,” arXiv preprint arXiv:2306.14824, 2023.
- [268] Q. Yu, Q. Sun, X. Zhang, Y. Cui, F. Zhang, Y. Cao, X. Wang, and J. Liu, “Capsfusion: Rethinking image-text data at scale,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14022–14032.
- [269] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin, “Sharegpt4v: Improving large multi-modal models with better captions,” in European Conference on Computer Vision. Springer, 2024, pp. 370–387.
- [270] P. Tong, E. Brown, P. Wu, S. Woo, A. J. V. IYER, S. C. Akula, S. Yang, J. Yang, M. Middepogu, Z. Wang et al., “Cambrian-1: A fully open, vision-centric exploration of multimodal llms,” Advances in Neural Information Processing Systems, vol. 37, pp. 87310–87356, 2024.
- [271] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

- [272] S. Gu, J. Zhang, S. Zhou, K. Yu, Z. Xing, L. Wang, Z. Cao, J. Jia, Z. Zhang, Y. Wang et al., “Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data,” arXiv preprint arXiv:2410.18558, 2024.
- [273] Y. Zhang, B. Ni, X.-S. Chen, H.-R. Zhang, Y. Rao, H. Peng, Q. Lu, H. Hu, M.-H. Guo, and S.-M. Hu, “Bee: A high-quality corpus and full-stack suite to unlock advanced fully open mllms,” 2025. [Online]. Available: https://arxiv.org/abs/2510.13795
- [274] S. Changpinyo, P. Sharma, N. Ding, and R. Soricut, “Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 3558–3568.
- [275] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.
- [276] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “Textdiffuser: Diffusion models as text painters,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [277] S. AI and LAION, “Renderedtext,” 2023. [Online]. Available: https://huggingface.co/datasets/wendlerc/RenderedText
- [278] K. Sun, J. Pan, Y. Ge, H. Li, H. Duan, X. Wu, R. Zhang, A. Zhou, Z. Qin, Y. Wang et al., “Journeydb: A benchmark for generative image understanding,” Advances in neural information processing systems, vol. 36, pp. 49659–49678, 2023.
- [279] Y. Tuo, W. Xiang, J.-Y. He, Y. Geng, and X. Xie, “Anytext: Multilingual visual text generation and editing,” 2023.
- [280] S. Li, J. Fu, K. Liu, W. Wang, K.-Y. Lin, and W. Wu, “Cosmicman: A text-to-image foundation model for humans,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6955–6965.
- [281] Y. Onoe, S. Rane, Z. Berger, Y. Bitton, J. Cho, R. Garg, A. Ku, Z. Parekh, J. Pont-Tuset, G. Tanzer et al., “Docci: Descriptions of connected and contrasting images,” in European Conference on Computer Vision. Springer, 2024, pp. 291–309.
- [282] V. Singla, K. Yue, S. Paul, R. Shirkavand, M. Jayawardhana, A. Ganjdanesh, H. Huang, A. Bhatele, G. Somepalli, and T. Goldstein, “From pixels to prose: A large dataset of dense image captions,” arXiv preprint arXiv:2406.10328, 2024.
- [283] X. Li, F. Zhang, H. Diao, Y. Wang, X. Wang, and L. DUAN, “Densefusion-1m: Merging vision experts for comprehensive multimodal perception,” Advances in Neural Information Processing Systems, vol. 37, pp. 18535–18556, 2024.
- [284] O. Boer Bohan, “Megalith-10m,” https://huggingface.co/ datasets/madebyollin/megalith-10m, June 2024, accessed: 202410-07. [Online]. Available: https://huggingface.co/datasets/ madebyollin/megalith-10m
- [285] jackyhate, “text-to-image-2m: A high-quality, diverse textto-image training dataset,” 2024. [Online]. Available: https: //huggingface.co/datasets/jackyhate/text-to-image-2M
- [286] J. Meyer, N. Padgett, C. Miller, and L. Exline, “Public domain 12m: A highly aesthetic image-text dataset with novel governance mechanisms,” arXiv preprint arXiv:2410.23144, 2024.
- [287] D. Beniaguev, “Synthetic faces high quality - text 2 image (sfhq-t2i) dataset,” 2024. [Online]. Available: https://github. com/SelfishGene/SFHQ-T2I-dataset
- [288] H. Zhang, Z. Duan, X. Wang, Y. Chen, and Y. Zhang, “Eligen: Entity-level controlled image generation with regional attention,” arXiv preprint arXiv:2501.01097, 2025.
- [289] A. J. Wang, D. Mao, J. Zhang, W. Han, Z. Dong, L. Li, Y. Lin, Z. Yang, L. Qin, F. Zhang et al., “Textatlas5m: A large-scale dataset for dense text image generation,” arXiv preprint arXiv:2502.07870, 2025.
- [290] J. Chen, Z. Cai, P. Chen, S. Chen, K. Ji, X. Wang, Y. Yang, and B. Wang, “Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation,” 2025. [Online]. Available: https://arxiv.org/abs/2506.18095
- [291] S. Chen, J. Lai, J. Gao, T. Ye, H. Chen, H. Shi, S. Shao, Y. Lin, S. Fei, Z. Xing, Y. Jin, J. Luo, X. Wei, and L. Zhu, “Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework,” arXiv preprint arXiv:2506.10741, 2025.
- [292] J. Ye, D. Jiang, Z. Wang, L. Zhu, Z. Hu, Z. Huang, J. He, Z. Yan, J. Yu, H. Li et al., “Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation,” arXiv preprint arXiv:2508.09987, 2025.
- [293] R. Fang, A. Yu, C. Duan, L. Huang, S. Bai, Y. Cai, K. Wang, S. Liu, X. Liu, and H. Li, “Flux-reason-6m & prism-bench: A

- million-scale text-to-image reasoning dataset and comprehensive benchmark,” arXiv preprint arXiv:2509.09680, 2025.
- [294] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18392–18402.
- [295] K. Zhang, L. Mo, W. Chen, H. Sun, and Y. Su, “Magicbrush: A manually annotated dataset for instruction-guided image editing,” Advances in Neural Information Processing Systems, vol. 36, pp. 31428–31449, 2023.
- [296] S. Zhang, X. Yang, Y. Feng, C. Qin, C.-C. Chen, N. Yu, Z. Chen, H. Wang, S. Savarese, S. Ermon, C. Xiong, and R. Xu, “Hive: Harnessing human feedback for instructional visual editing,” arXiv preprint arXiv:2303.09618, 2023.
- [297] M. Hui, S. Yang, B. Zhao, Y. Shi, H. Wang, P. Wang, Y. Zhou, and C. Xie, “Hq-edit: A high-quality dataset for instruction-based

- image editing,” arXiv preprint arXiv:2404.09990, 2024.

[298] L. Yang, B. Zeng, J. Liu, H. Li, M. Xu, W. Zhang, and S. Yan, “Editworld: Simulating world dynamics for instruction-following

- image editing,” arXiv preprint arXiv:2405.14785, 2024.

- [299] H. Zhao, X. S. Ma, L. Chen, S. Si, R. Wu, K. An, P. Yu, M. Zhang, Q. Li, and B. Chang, “Ultraedit: Instruction-based fine-grained image editing at scale,” Advances in Neural Information Processing Systems, vol. 37, pp. 3058–3093, 2024.
- [300] Z. Zeng, H. Hua, J. Fu, J. Luo et al., “Promptfix: You prompt and we fix the photo,” Advances in Neural Information Processing Systems, vol. 37, pp. 40000–40031, 2024.
- [301] C. Wei, Z. Xiong, W. Ren, X. Du, G. Zhang, and W. Chen, “Omniedit: Building image editing generalist models through specialist supervision,” in The Thirteenth International Conference on Learning Representations, 2024.
- [302] Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang, “AnyEdit: Mastering unified high-quality image editing for any idea,” arXiv preprint arXiv:2411.15738, 2024.
- [303] B. Pathiraja, M. Patel, S. Singh, Y. Yang, and C. Baral, “RefEdit: A benchmark and method for improving instruction-based image editing model for referring expression,” arXiv preprint arXiv:2506.03448, 2025.
- [304] Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan, “Imgedit: A unified image editing dataset and benchmark,” arXiv preprint arXiv:2505.20275, 2025.
- [305] D. Chang, M. Cao, Y. Shi, B. Liu, S. Cai, S. Zhou, W. Huang, G. Wetzstein, M. Soleymani, and P. Wang, “ByteMorph: Benchmarking instruction-guided image editing with non-rigid motions,” arXiv preprint arXiv:2506.03107, 2025.
- [306] Y. Wang, S. Yang, B. Zhao, L. Zhang, Q. Liu, Y. Zhou, and C. Xie, “Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset,” arXiv preprint arXiv:2507.21033, 2025.
- [307] J. Ma, X. Zhu, Z. Pan, Q. Peng, X. Guo, C. Chen, and H. Lu, “X2edit: Revisiting arbitrary-instruction image editing through self-constructed data and task-aware representation learning,” arXiv preprint arXiv:2508.07607, 2025.
- [308] Y. Qian, E. Bocek-Rivele, L. Song, J. Tong, Y. Yang, J. Lu, W. Hu, and Z. Gan, “Pico-banana-400k: A large-scale dataset for text-guided image editing,” 2025. [Online]. Available: https://arxiv.org/abs/2510.19808
- [309] W. Zhu, J. Hessel, A. Awadalla, S. Y. Gadre, J. Dodge, A. Fang, Y. Yu, L. Schmidt, W. Y. Wang, and Y. Choi, “Multimodal c4: An open, billion-scale corpus of images interleaved with text,” Advances in Neural Information Processing Systems, vol. 36, pp. 8958–8974, 2023.
- [310] H. Lauren¸con, L. Saulnier, L. Tronchon, S. Bekman, A. Singh, A. Lozhkov, T. Wang, S. Karamcheti, A. Rush, D. Kiela et al., “Obelics: An open web-scale filtered dataset of interleaved image-text documents,” Advances in Neural Information Processing Systems, vol. 36, pp. 71683–71702, 2023.
- [311] W. Chen, L. Li, Y. Yang, B. Wen, F. Yang, T. Gao, Y. Wu, and L. Chen, “Comm: A coherent interleaved image-text dataset for multimodal understanding and generation,” arXiv preprint arXiv:2406.10462, 2024.
- [312] Q. Li, Z. Chen, W. Wang, W. Wang, S. Ye, Z. Jin et al., “Omnicorpus: A unified multimodal corpus of 10 billion-level images interleaved with text,” in The Thirteenth International Conference on Learning Representations, 2025.
- [313] Y. Zheng, H. Yang, T. Zhang, J. Bao, D. Chen, Y. Huang, L. Yuan, D. Chen, M. Zeng, and F. Wen, “General facial representation

- learning in a visual-linguistic manner,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18697–18709.
- [314] C. Qin, S. Zhang, N. Yu, Y. Feng, X. Yang, Y. Zhou, H. Wang, J. C. Niebles, C. Xiong, S. Savarese et al., “Unicontrol: A unified diffusion model for controllable visual generation in the wild,” arXiv preprint arXiv:2305.11147, 2023.
- [315] Z. Tan, S. Liu, X. Yang, Q. Xue, and X. Wang, “Ominicontrol: Minimal and universal control for diffusion transformer,” arXiv preprint arXiv:2411.15098, 2024.
- [316] N. Kumari, X. Yin, J.-Y. Zhu, I. Misra, and S. Azadi, “Generating multi-image synthetic data for text-to-image customization,” ArXiv, 2025.
- [317] Z.-Y. Li, R. Du, J. Yan, L. Zhuo, Z. Li, P. Gao, Z. Ma, and M.-M. Cheng, “Visualcloze : A universal image generation framework via visual in-context learning,” arXiv preprint arxiv:, 2025.
- [318] M. Weber, D. Y. Fu, Q. Anthony, Y. Oren, S. Adams, A. Alexandrov, X. Lyu, H. Nguyen, X. Yao, V. Adams, B. Athiwaratkun, R. Chalamala, K. Chen, M. Ryabinin, T. Dao, P. Liang, C. R´e,

I. Rish, and C. Zhang, “Redpajama: an open dataset for training large language models,” NeurIPS Datasets and Benchmarks Track, 2024.

- [319] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.
- [320] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13. Springer, 2014, pp. 740–755.
- [321] B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu, and L. Yuan, “Florence-2: Advancing a unified representation for a variety of vision tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4818–4829.
- [322] Q. Wang, X. Bai, H. Wang, Z. Qin, A. Chen, H. Li, X. Tang, and Y. Hu, “Instantid: Zero-shot identity-preserving generation in seconds,” arXiv preprint arXiv:2401.07519, 2024.
- [323] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A universe of annotated 3d objects,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 13142–13153.
- [324] X. Wang, S. Fu, Q. Huang, W. He, and H. Jiang, “Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance,” arXiv preprint arXiv:2406.07209, 2024.
- [325] X. Pan, L. Dong, S. Huang, Z. Peng, W. Chen, and F. Wei, “Kosmos-g: Generating images in context with multimodal large language models,” arXiv preprint arXiv:2310.02992, 2023.
- [326] Z. Huang, S. Zhuang, C. Fu, B. Yang, Y. Zhang, C. Sun, Z. Zhang, Y. Wang, C. Li, and Z.-J. Zha, “Wegen: A unified model for interactive multimodal generation as we chat,” arXiv preprint arXiv:2503.01115, 2025.
- [327] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European Conference on Computer Vision. Springer, 2024, pp. 38–55.
- [328] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Doll´ar, and C. Feichtenhofer, “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024. [Online]. Available: https://arxiv.org/abs/2408.00714
- [329] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “Vqa: Visual question answering,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2425–2433.
- [330] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, “Making the v in vqa matter: Elevating the role of image understanding in visual question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6904–6913.
- [331] J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick, and R. Girshick, “CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

- [332] D. A. Hudson and C. D. Manning, “GQA: A new dataset for realworld visual reasoning and compositional question answering,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- [333] K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi, “OKVQA: A visual question answering benchmark requiring external knowledge,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- [334] R. Zellers, Y. Bisk, A. Farhadi, and Y. Choi, “From recognition to cognition: Visual commonsense reasoning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- [335] A. Das, S. Kottur, K. Gupta, A. Singh, D. Yadav, J. M. Moura, D. Parikh, and D. Batra, “Visual dialog,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 326– 335.
- [336] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque, “Chartqa: A benchmark for question answering about charts with visual and logical reasoning,” arXiv preprint arXiv:2203.10244, 2022.
- [337] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.
- [338] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R. Mottaghi, “A-OKVQA: A benchmark for visual question answering using world knowledge,” in Proceedings of the European Conference on Computer Vision, 2022.
- [339] J. Li, X. Cheng, W. X. Zhao, J.-Y. Nie, and J.-R. Wen, “Halueval: A large-scale hallucination evaluation benchmark for large language models,” arXiv preprint arXiv:2305.11747, 2023.
- [340] Z. Yin, J. Wang, J. Cao, Z. Shi, D. Liu, M. Li, X. Huang, Z. Wang, L. Sheng, L. Bai et al., “Lamm: Language-assisted multimodal instruction-tuning dataset, framework, and benchmark,” Advances in Neural Information Processing Systems, vol. 36, pp. 26650–26685, 2023.
- [341] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34892–34916, 2023.
- [342] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi et al., “mplug-owl: Modularization empowers large language models with multimodality,” arXiv preprint arXiv:2304.14178, 2023.
- [343] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in European conference on computer vision. Springer, 2024, pp. 216–233.
- [344] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun et al., “Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9556–9567.
- [345] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” arXiv preprint arXiv:2308.02490, 2023.
- [346] W. Yu, Z. Yang, L. Ren, L. Li, J. Wang, K. Lin, C.-C. Lin, Z. Liu, L. Wang, and X. Wang, “Mm-vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities,” arXiv preprint arXiv:2408.00765, 2024.
- [347] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin et al., “Are we on the right way for evaluating large vision-language models?” arXiv preprint arXiv:2403.20330, 2024.
- [348] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan, “Seedbench: Benchmarking multimodal llms with generative comprehension,” arXiv preprint arXiv:2307.16125, 2023.
- [349] S. Ging, M. A. Bravo, and T. Brox, “Open-ended vqa benchmarking of vision-language models by exploiting classification datasets and their semantic hierarchy,” arXiv preprint arXiv:2402.07270, 2024.
- [350] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” in Proceedings of the International Conference on Learning Representations, 2024.
- [351] H. Fei, Y. Zhou, J. Li et al., “On path to multimodal generalist: General-Level and General-Bench,” arXiv:2505.04620, 2025.
- [352] J. Yu, Y. Xu, J. Y. Koh, T. Luong, G. Baid, Z. Wang, V. Vasudevan, A. Ku, Y. Yang, B. K. Ayan et al., “Scaling autoregressive mod-

- els for content-rich text-to-image generation,” arXiv:2206.10789, vol. 2, no. 3, p. 5, 2022.
- [353] J. Cho, A. Zala, and M. Bansal, “Dall-Eval: Probing the reasoning skills and social biases of text-to-image generation models,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 3043–3054.
- [354] E. M. Bakr, P. Sun, X. Shen, F. F. Khan, L. E. Li, and M. Elhoseiny, “HRS-Bench: Holistic, reliable and scalable benchmark for textto-image models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20041–20053.
- [355] Y. Hu, B. Liu, J. Kasai, Y. Wang, M. Ostendorf, R. Krishna, and N. A. Smith, “TIFA: Accurate and interpretable text-to-image faithfulness evaluation with question answering,” arXiv:2303.11897, 2023.
- [356] D. Ghosh, H. Hajishirzi, and L. Schmidt, “Geneval: An objectfocused framework for evaluating text-to-image alignment,” Advances in Neural Information Processing Systems, vol. 36, pp. 52132– 52152, 2023.
- [357] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2I-Compbench: A comprehensive benchmark for open-world compositional textto-image generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 78723–78747, 2023.
- [358] T. Lee, M. Yasunaga, C. Meng, Y. Mai, J. S. Park, A. Gupta, Y. Zhang, D. Narayanan, H. Teufel, M. Bellagente et al., “Holistic evaluation of text-to-image models,” Advances in Neural Information Processing Systems, vol. 36, pp. 69981–70011, 2023.
- [359] X. Fu, M. He, Y. Lu, W. Y. Wang, and D. Roth, “CommonsenseT2I challenge: Can text-to-image generation models understand commonsense?” arXiv preprint arXiv:2406.07546, 2024.
- [360] J. Cho, Y. Hu, J. Baldridge, R. Garg, P. Anderson, R. Krishna, M. Bansal, J. Pont-Tuset, and S. Wang, “Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation,” in International Conference on Learning Representations, 2024.
- [361] B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, T. Ling, X. Xia, P. Zhang, G. Neubig et al., “GenAI-Bench: Evaluating and improving compositional text-to-visual generation,” arXiv preprint arXiv:2406.13743, 2024.
- [362] X. Wu, D. Yu, Y. Huang, O. Russakovsky, and S. Arora, “CONCEPTMIX: A compositional image generation benchmark with controllable difficulty,” arXiv:2408.14339, 2024.
- [363] X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu, “EllA: Equip diffusion models with LLM for enhanced semantic alignment,” arXiv:2403.05135, 2024.
- [364] K. Huang, C. Duan, K. Sun, E. Xie, Z. Li, and X. Liu, “T2ICompBench++: An enhanced and comprehensive benchmark for compositional text-to-image generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [365] H. Hua, Z. Zeng, Y. Song, Y. Tang, L. He, D. Aliaga, W. Xiong, and J. Luo, “MMIG-Bench: Towards comprehensive and explainable evaluation of multi-modal image generation models,” arXiv preprint arXiv:2505.19415, 2025.
- [366] J. Chang, Y. Fang, P. Xing, S. Wu, W. Cheng, R. Wang, X. Zeng, G. Yu, and H.-B. Chen, “OneIG-Bench: Omni-dimensional nuanced evaluation for image generation,” 2025. [Online]. Available: https://arxiv.org/abs/2506.07977
- [367] Y. Niu, M. Ning, M. Zheng, W. Jin, B. Lin, P. Jin, J. Liao, K. Ning, C. Feng, B. Zhu, and L. Yuan, “WISE: A world knowledgeinformed semantic evaluation for text-to-image generation,” arXiv preprint arXiv:2503.07265, 2025.
- [368] N. Du, Z. Chen, Z. Chen, S. Gao, X. Chen, Z. Jiang, J. Yang, and Y. Tai, “TextCrafter: Accurately rendering multiple texts in complex visual scenes,” arXiv preprint arXiv:2503.23461, 2025.
- [369] D. Zhang, C. Jiang, R. Xu, B. Chen, Z. Jin, Y. Lu, J. Zhang, L. Yong, J. Luo, and S. Luo, “WorldGenBench: A world-knowledgeintegrated benchmark for reasoning-driven text-to-image generation,” arXiv preprint arXiv:2505.01490, 2025.
- [370] S. Wang, C. Saharia, C. Montgomery, J. Pont-Tuset, S. Noy, S. Pellegrini, Y. Onoe, S. Laszlo, D. J. Fleet, R. Soricut et al., “Imagen editor and EditBench: Advancing and evaluating textguided image inpainting,” in IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18359–18369.
- [371] S. Basu, M. Saberi, S. Bhardwaj, A. M. Chegini, D. Massiceti, M. Sanjabi, S. X. Hu, and S. Feizi, “Editval: Benchmarking diffusion based text-guided image editing methods,” arXiv preprint arXiv:2310.02426, 2023.

- [372] S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman, “Emu edit: Precise image editing via recognition and generation tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8871–8879.
- [373] Y. Huang, L. Xie, X. Wang, Z. Yuan, X. Cun, Y. Ge, J. Zhou, C. Dong, R. Huang, R. Zhang et al., “SmartEdit: Exploring complex instruction-based image editing with multimodal large language models,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8362–8371.
- [374] Y. Ma, J. Ji, K. Ye, W. Lin, Z. Wang, Y. Zheng, Q. Zhou, X. Sun, and R. Ji, “I2EBench: A comprehensive benchmark for instructionbased image editing,” arXiv preprint arXiv:2408.14180, 2024.
- [375] J. Bai, W. Chow, L. Yang, X. Li, J. Li, H. Zhang, and S. Yan, “Humanedit: A high-quality human-rewarded dataset for instruction-based image editing,” arXiv preprint arXiv:2412.04280, 2024.
- [376] S. Sun, B. Qu, X. Liang, S. Fan, and W. Gao, “IE-Bench: Advancing the measurement of text-driven image editing for human perception alignment,” arXiv:2501.09927, 2025.
- [377] S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han et al., “Step1X-Edit: A practical framework for general image editing,” arXiv preprint arXiv:2504.17761, 2025.
- [378] B. Jia, W. Huang, Y. Tang et al., “CompBench: Benchmarking complex instruction-guided image editing,” arXiv:2505.12200, 2025.
- [379] Y. Qian, J. Lu, T.-J. Fu, X. Wang, C. Chen, Y. Yang, W. Hu, and Z. Gan, “GIE-Bench: Towards grounded evaluation for textguided image editing,” arXiv:2505.11493, 2025.
- [380] R. Yosef, M. Yanuka, Y. Bitton, and D. Lischinski, “EditInspector: A benchmark for evaluation of text-guided image edits,” arXiv preprint arXiv:2506.09988, 2025.
- [381] C. Wang, Y. Zhou, Q. Wang, Z. Wang, and K. Zhang, “Complexbench-edit: Benchmarking complex instruction-driven image editing via compositional dependencies,” arXiv preprint arXiv:2506.12830, 2025.
- [382] Y. Wu, Z. Li, X. Hu, X. Ye, X. Zeng, G. Yu, W. Zhu, B. Schiele, M.H. Yang, and X. Yang, “KRIS-Bench: Benchmarking next-level intelligent image editing models,” arXiv preprint arXiv:2505.16707, 2025.
- [383] M. Liu, Z. Xu, Z. Lin, T. Ashby, J. Rimchala, J. Zhang, and L. Huang, “Holistic evaluation for interleaved text-and-image generation,” arXiv preprint arXiv:2406.14643, 2024.
- [384] J. An, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, L. Wang, and J. Luo, “Openleaf: Open-domain interleaved image-text generation and evaluation,” arXiv preprint arXiv:2310.07749, 2023.
- [385] D. Chen, R. Chen, S. Pu, Z. Liu, Y. Wu, C. Chen, B. Liu, Y. Huang, Y. Wan, P. Zhou et al., “Interleaved scene graph for interleaved text-and-image generation assessment,” arXiv preprint arXiv:2411.17188, 2024.
- [386] P. Xia, S. Han, S. Qiu, Y. Zhou, Z. Wang, W. Zheng, Z. Chen, C. Cui, M. Ding, L. Li et al., “MMIE: Massive multimodal interleaved comprehension benchmark for large vision-language models,” arXiv preprint arXiv:2410.10139, 2024.
- [387] P. Zhou, X. Peng, J. Song, C. Li, Z. Xu, Y. Yang, Z. Guo, H. Zhang, Y. Lin, Y. He et al., “Gate opening: A comprehensive benchmark for judging open-ended interleaved image-text generation,” arXiv preprint arXiv:2411.18499, 2024.
- [388] Y. Li, H. Wang, Q. Zhang, B. Xiao, C. Hu, H. Wang, and X. Li, “UniEval: Unified holistic evaluation for unified multimodal understanding and generation,” arXiv:2505.10483, 2025.
- [389] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22500–22510.
- [390] Y. Peng, Y. Cui, H. Tang, Z. Qi, R. Dong, J. Bai, C. Han, Z. Ge, X. Zhang, and S.-T. Xia, “Dreambench++: A human-aligned benchmark for personalized image generation,” arXiv preprint arXiv:2406.16855, 2024.
- [391] H. Lin, T. Geng, Z. Xu, and W. Zhao, “VTBench: Evaluating visual tokenizers for autoregressive image generation,” arXiv:2502.01634, 2025.
- [392] B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting regionto-phrase correspondences for richer image-to-sentence models,” International Journal of Computer Vision, vol. 123, no. 1, p. 74–93, 2017.

- [393] X. Chen, H. Fang, T.-Y. Lin, R. Vedantam, S. Gupta, P. Doll´ar, and C. L. Zitnick, “Microsoft COCO Captions: Data collection and evaluation server,” arXiv preprint arXiv:1504.00325, 2015.
- [394] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.
- [395] T. Gokhale, H. Palangi, B. Nushi, V. Vineet, E. Horvitz, E. Kamar, C. Baral, and Y. Yang, “Benchmarking spatial relationships in text-to-image generation,” arXiv preprint arXiv:2212.10015, 2022.
- [396] S. Han, H. Fan, J. Fu, L. Li, T. Li, J. Cui, Y. Wang, Y. Tai, J. Sun, C. Guo et al., “EvalMuse-40K: A reliable and fine-grained benchmark with comprehensive human annotations for text-to-image generation model evaluation,” arXiv preprint arXiv:2412.18150, 2024.
- [397] L. Zhao, T. Zhao, Z. Lin, X. Ning, G. Dai, H. Yang, and Y. Wang, “Flasheval: Towards fast and accurate evaluation of text-to-image diffusion generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16122–16131.
- [398] Y. Zhou, Z. Zhang, J. Cao, J. Jia, Y. Jiang, F. Wen, X. Liu, X. Min, and G. Zhai, “MEMO-Bench: A multiple benchmark for textto-image and multimodal large language models on human emotion analysis,” arXiv preprint arXiv:2411.11235, 2024.
- [399] C. Schuhmann and R. Beaumont, “Laion-aesthetics v2,” https: //laion.ai/blog/laion-aesthetics/, 2022, subset of LAION-5B filtered by the LAION-Aesthetics Predictor V2 to images with predicted aesthetic scores of 6.5 or higher.
- [400] Y. Shi, Y. Dong, Y. Ding, Y. Wang, X. Zhu, S. Zhou, W. Liu, H. Tian, R. Wang, H. Wang et al., “Realunify: Do unified models truly benefit from unification? a comprehensive benchmark,” arXiv preprint arXiv:2509.24897, 2025.
- [401] D. Jiang, Z. Guo, R. Zhang, Z. Zong, H. Li, L. Zhuo, S. Yan, P.A. Heng, and H. Li, “T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot,” arXiv preprint arXiv:2505.00703, 2025.
- [402] M. Liu, H. Chen, J. Wang, L. Wang, B. R. Ramakrishnan, and W. Zhang, “On fairness of unified multimodal large language model for image generation,” arXiv preprint arXiv:2502.03429, 2025.
- [403] R. An, S. Yang, R. Zhang, Z. Shen, M. Lu, G. Dai, L. Hao, Z. Guo, S. Yan, Y. Luo, B. Zou, C. Yang, and W. Zhang, “Unictokens: Boosting personalized understanding and generation via unified concept tokens,” arXiv preprint arXiv:2505.14671, 2025.

