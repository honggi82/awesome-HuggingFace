## Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining

Dongyang Liu*12 Shitian Zhao*2 Le Zhuo*12 Weifeng Lin*1 Yi Xin2 Xinyue Li2 Qi Qin2 Yu Qiao2 Hongsheng Li¶1 Peng Gao¶♦2

# arXiv:2408.02657v3[cs.CV]24Apr2025

### Abstract

We present Lumina-mGPT, a family of multimodal autoregressive models capable of various vision and language tasks, particularly excelling in generating flexible photorealistic images from text descriptions. By initializing from multimodal Generative PreTraining (mGPT), we demonstrate that decoder-only Autoregressive (AR) model can achieve image generation performance comparable to modern diffusion model with high efficiency through Flexible Progressive Supervised Finetuning (FP-SFT). Equipped with our proposed Unambiguous image Representation (UniRep), Lumina-mGPT can flexibly generate highquality images of varying aspect ratios. Building on the strong image generation capabilities, we further explore Ominiponent Supervised Finetuning (Omni-SFT), an initial attempt to elevate Lumina-mGPT into a unified multi-modal generalist. The resulting model demonstrates versatile multimodal capabilities, including visual generation tasks like text-to-image/multiview generation and controllable generation, visual recognition tasks like segmentation and depth estimation, and vision-language tasks like multi-turn visual question answering, showing the rosy potential of the technical direction. Codes and checkpoints are available at https://github.

com/Alpha-VLLM/Lumina-mGPT.

### 1. Introduction

Seminal models, including DALL-E 3 (Betker et al., 2023), Stable Diffusion 3 (Esser et al., 2024), and SoRA (Brooks et al., 2024), have demonstrated superior performance in photorealistic image and video generation using diffusionbased generative modeling over continuous latent image features. In contrast, autoregressive (AR) generative models,

*Equal contribution 1The Chinese University of Hong Kong 2Shanghai AI Laboratory. Correspondence to: Hongsheng Li <hsli@ee.cuhk.edu.hk>, Peng Gao <gaopeng@pjlab.org.cn>.

which rely on “next-token prediction”, have revolutionized text generation with groundbreaking reasoning abilities, as exemplified by models like GPT-4 (Achiam et al., 2023) and Deepseek-v3 (Team et al., 2023).

However, AR-based generative modeling with vectorquantized image features remains a relatively less mature topic. While previous autoregressive efforts, such as DALLE (Ramesh et al., 2021), CogView (Ding et al., 2021), Parti (Yu et al., 2022), OFA (Wang et al., 2022), UnifiedIO (Lu et al., 2022; 2024), LlamaGen (Sun et al., 2024), and Chameleon (Team, 2024a), have made important explorations, each of these methods suffers from one or more of the following limitations: (1) their image generation capabilities have been either unsatisfactory or constrained to academic benchmarks such as ImageNet (Deng et al., 2009); (2) they rely on complex architectures, such as encoderdecoder frameworks, which impede scalability and limit generalization to other tasks; (3) they are restricted in decoding resolution and flexibility, often producing small images with fixed resolutions; and (4) they lack task extensibility, making them unsuitable for broader scenarios that require integration with tasks like dense labeling or controllable image generation. A more detailed discussion of these challenges is provided in Appendix D.

To address the aforementioned challenges, we present Lumina-mGPT, a decoder-only transformer initiated with effective multimodal Generative PreTraining (mGPT) and then supervised-finetuned over flexible, high-quality, highresolution discrete image tokens in a progressive manner. This framework illuminates flexible high-resolution photorealistic image generation and can be easily extended to solve various downstream tasks in a unified manner. We provide a detailed comparison of the architecture design and model capabilities of existing multimodal autoregressive approaches in Table 1. The key features of Lumina-mGPT are outlined below:

➀ Simple Decoder-Only Architecture: We adopt a simple decoder-only architecture, which offers a significant advantage over more complex designs such as encoder-decoder architectures. Decoder-only models provide an elegant and extensible framework that unifies various understanding and

Generate an image of [WxH] according to the following prompt: [image caption prompt]

Text-to-Image

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Multi-turn

A1: Focus on time management, use productivity tools,

Q1: How can I improve my productivity?

A3: Work for 25 minutes, take a 5-minute break.

Q2: Time management tips, please.

A2: Prioritize tasks and set clear goals.

Q3: What’s the Pomodoro Technique?

Dialog

and stay motivated.

Q1: Generate an

A1: A2: The porcelain teapot with its intricate

Q2: What if I

Multi-turnVisual

image of 768x768 according to the following prompt: “A piece of porcelain is placed on the table.”

use the tool shown in this image to hit the porcelain?

floral designs and golden trim. However, the tool depicted—a metallic hammer with grip—is designed for striking purposes. If used to hit the teapot, the hammer's weighty metal head would likely cause the teapot to shatter or chip due to its fragile nature, especially if struck with any force applied from above as a hammer would.

[Figure 5]

Dialog

[Figure 6]

[Figure 7]

Generate an image according to the provided image, and according to the following caption:

An ice cream with pink color. A butterfly is flying in the forest. A birthday cake.

Controllable

Generation

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Pose Estimation Depth Estimation Segmentation

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Recognition

[Figure 18]

[Figure 19]

Visual

ImageEditing

Q1: Generate an image of 512x512 according to the following prompt: “A cup of coffee is placed on the table.”

Q1: Remove the cup in the background.

Q3:Add a spoon

[Figure 20]

[Figure 21]

[Figure 22]

Multi-turn

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Q1: Generate 16 object views of 384x384 according to the following prompt: “A shoe with a checkerboard pattern.”

Multi-view

Image

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

- Figure 1. Unified in the next-token prediction framework, Lumina-mGPT can perform a wide range of multi-modal tasks. See Figure 6 to Figure 11 in the Appendix for more demonstrations.

Table 1. Design choices and capabilities of multimodal autoregressive approaches. Lumina-mGPT is the only model capable of both flexible photorealistic image generation and multimodal task unification, due to its decoder-only transformer design and multimodal generative pretraining.

Model Multimodal Photorealistic Flexible Image Task Architecture Pretraining Image Generation Aspect Ratio Extensibility

Method

DALL-E (Ramesh et al., 2021) Decoder-only × × × × Cogview (Ding et al., 2021) Decoder-only × × × × Parti (Yu et al., 2022) Encoder-Decoder × × × × LlamaGen (Sun et al., 2024) Encoder-Decoder × × × × OFA (Wang et al., 2022) Encoder-Decoder ✓ × × ✓ Unified-IO (Lu et al., 2022) Encoder-Decoder × × × ✓ Unified-IO 2 (Lu et al., 2024) Encoder-Decoder ✓ × × ✓ CM3Leon (Yu et al., 2023) Decoder-only ✓ × × ✓ Chameleon (Team, 2024a) Decoder-only ✓ × × × Lumina-mGPT Decoder-only ✓ ✓ ✓ ✓

generation tasks across different modalities. This makes them a compelling choice for achieving true unification. Furthermore, by leveraging the same architecture as the rapidly evolving text-only LLMs, Lumina-mGPT can take advantage of well-established theories and infrastructure in the LLM community, including scaling properties (Chowdhery et al., 2022; Brown et al., 2020) and advanced techniques for optimizing training and inference (Dao et al., 2022; Dao, 2024; Kwon et al., 2023).

➁ Flexible High-quality Image Generation: Despite the aforementioned strengths of decoder-only architecture, the image generation capabilities of such models remain limited, creating a gap between the potential and the reality of this architecture. We thus propose Flexible Progressive Supervised Finetuning (FP-SFT) to fully fulfill the potential of high-quality text-to-image generation. This approach starts with low-resolution discrete tokens and progressively transitions to high-resolution discrete tokens. Combined with Unambiguous image Representation (Uni-Rep), this weakto-strong SFT strategy effectively grants the model with the ability to generate high-quality photorealistic images with flexible aspect ratios.

➂ Omnipotent Task Unification: The high-quality image generation capabilities achieved during the FP-SFT stage provide the prerequisites for further exploring the fundamental strength, namely task unification, of the decoderonly architecture. We thus propose Omnipotent Supervised Finetuning (Omni-SFT), a preliminary attempt to create a multi-modal omnipotent generalist. Omni-SFT treats various tasks–such as multi-turn dialog, visual multi-turn understanding, dense labeling, text-to-image generation, text-tomultiview generation, image editing, and spatial-conditional image generation–as a unified discrete modeling task, allowing the model to achieve omnipotence via a unified natural language interface.

➃ Effective Multimodal Generative Pretraining: Instead

of random initialization, Lumina-mGPT leverages an effective multimodal Generative PretTraining (mGPT). This representation is derived from a multimodal transformer trained at scale using a straightforward "next-token prediction" loss. The rich knowledge encapsulated in this pretraining significantly accelerates the learning process for text-to-image generation and the downstream vision-language tasks. Given the constraints in training resources, we adopt the mGPT representation directly from the pretrained Chameleon 7B and 30B models released by Meta (Team, 2024a).

The summarized contributions are as follows: (1) We are the first, particularly in the open-source domain, to demonstrate that a decoder-only AR model can achieve image generation performance on par with modern diffusion models. Besides, with mGPT initialization, this capability can be achieved with remarkable efficiency, requiring only 32 A100 GPUs over 7 days to train a 7B model. (2) We introduce UniRep, a novel image representation that enables decoder-only AR models to generate images with varying aspect ratios, offering greater flexibility in image generation. (3) Leveraging the strong image generation capabilities achieved, we propose Omni-SFT, a pioneering exploration aimed at elevating the model into a unified generalist. (4) We release the full-pipeline code implementation to facilitate further advancements from the research community.

### 2. Methodolgy

Lumina-mGPT is a decoder-only transformer initialized with multimodal Generative PreTraining (mGPT) and finetuned over high-quality multimodal tokens derived from various tasks. Based on the robust mGPT representation and our proposed supervised finetuning strategies with unambiguous image representation, Lumina-mGPT achieves superior performance in photorealistic image generation and omnipotent task unification with high flexibility in image resolution and aspect ratio.

###### VQ-VAE

Codebook

- 0

- 1 … N

[Figure 39]

[Figure 40]

|19|
|---|
|0|
|13|

25 4 37 13 32 34 7

Encoder

Decoder

Quantize

|8|
|---|
|31|

VQ

VQ

<Input Image> (Resolution: 512x384)

<Output Image>

###### Resolution-Aware Prompt Unambiguous Image Representation

Generate an image of 512x384 according to the following prompt: Two dogs are wearing matching brown cowboy hats against a blue background.

<start-of-image> <end-of-image>

height/width indicator

<end-of-line>

| | | | | | | | |
|---|---|---|---|---|---|---|---|

|25|
|---|

|8|
|---|

|7|31|
|---|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

SOI 384 EOL EOL EOL EOI

512 19

4 37 0 13 32

13 34

- Figure 2. Illustration of Resolution-Aware Prompt (bottom left) and Unambiguous Image Representation (bottom right). These designs are used in all supervised finetuning stages to eliminate the ambiguity in image representation, enabling flexible resolution image modeling.

#### 2.1. Revisiting mGPT with Chameleon

prediction objective. Additionally, Chameleon applies zloss (Chowdhery et al., 2022) to stabilize the training.

mGPT represents the family of models utilizing a decoderonly transformer architecture and pretrained on extensive multimodal token sequences. We use the recent open-source model, Chameleon (Team, 2024a), as an example to illustrate the design choices and implementing details of mGPT.

Limitations of Chameleon Although Chameleon demonstrates potential for joint image and text understanding within one decoder-only transformer, its image generation ability remains inferior to state-of-the-art diffusion-based frameworks (Esser et al., 2024; Chen et al., 2024; Li et al., 2024c; Team, 2024b; Zhuo et al., 2024) in both quality and resolution flexibility. Moreover, it is worth noting that the image generation ability is even absent in the open-source version of Chameleon. Additionally, the capabilities of Chameleon are confined to vision-language and text-only tasks, excluding a broader range of vision-centric tasks. These include classic visual recognition tasks such as segmentation and depth prediction, as well as creative visual generation tasks like controllable generation and image editing. Lumina-mGPT is built upon Chameleon to unlock its full potential for flexible photorealistic image generation and to become a versatile vision generalist.

Multimodal Tokenization To unify text and images into a multimodal token sequence, it is essential first to tokenize both text and image into discrete space. The choice of image tokenizer is especially crucial as it determines the upper limit of generation quality. Chameleon trains a byte pair encoding (BPE) tokenizer for text; for images, it adopts the quantization-based tokenization method following prior works (Esser et al., 2021; Yu et al., 2022; Ramesh et al.,

- 2021), converting continuous image patches into discrete tokens from a fixed codebook while reducing spatial dimensions. The quantized image tokens are then flattened into 1D sequence and concatenated with text tokens to form a multimodal token sequence for unified modeling.

2.2. Lumina-mGPT 2.2.1. EFFECTIVE INITLIZATION

Decoder-Only Transformer In contrast to the encoderdecoder architecture adopted by Unified-IO and Parti, the decoder-only architecture leads to a simpler and more extensible approach for multimodal generative modeling. By transforming text and image inputs into a unified sequence of discrete tokens, the understanding and generation within and across modalities can be elegantly unified, forming a rosy direction towards general intelligence. During training, mGPT models the conditional probability p(xt|x1,...,xt−1) of multimodal sequences using the standard next-token

Large-scale pre-training and scalable model architecture have been widely verified as the golden path to advanced intelligence. As mGPTs like Chameleon are pretrained on large-scale datasets and have developed effective and generalizable representations, they are well-suited to serve as the starting point for flexible photorealistic image generation and beyond. In this work, we initialize from Chameleon with parameters ranging from 7B to 30B, using just 10M

high-quality image-text data points.

- 2.2.2. SUPERVISED FINETUNING FOR LUMINA-MGPT

Unambiguous Image Representation Existing methods represent images as 1D flattened sequences of 2D discrete image codes. While adequate for fixed resolutions, this approach becomes ambiguous when supporting variable resolutions, as with Lumina-mGPT. For instance, images with resolutions of 512 × 512, 256 × 1024, and 1024 × 256 can all be encoded into the same number of tokens, making it impossible to infer the original shape without examining the token contents. This ambiguity poses significant challenges for both image perception and generation.

To address this problem, we propose Unambiguous image Representation (Uni-Rep), which augments the image representations by adding extra height/width indicator tokens immediately after the <start-of-image> token and inserting <end-of-line> tokens after image tokens belonging to the same row. As shown in Figure 2, this modification ensures that the original shape of the images can be accurately parsed from the 1D representation without additional context or delving into the contents of the image tokens. This enhancement provides the foundation for Lumina-mGPT’s ability to perform image-related tasks at any resolution and aspect ratio.

Discussions: While either the height/width indicators or the <end-of-line> tokens can independently achieve disambiguation, we use both simultaneously due to their distinct benefits. When generating images, the height/width indicators, generated before any image tokens, pre-determine the shape of the image, aiding Lumina-mGPT in composing the image contents. The <end-of-line> tokens, on the other hand, serve as anchors, offering the 1D token sequence with additional explicit spatial information. Section 3.3 shows empirical analysis over these tokens.

Flexible Progressive Supervised Finetuning (FP-SFT) The FP-SFT process equips the pretrained mGPT with the capability to generate high-resolution images with flexible aspect ratios in a progressive manner. The process is divided into three stages, where the product of width and height approximates 5122, 7682, and 10242, respectively. In each stage, a set of candidate resolutions with similar areas but different height-width ratios are prepared, and each image is matched to the most suitable resolution. In the low-resolution stage, shorter sequence lengths and the resulting high training throughput allow the model to quickly traverse a large amount of data, learning the general composition of images and a broad spectrum of visual concepts. Conversely, in the high-resolution stage, the model is expected to focus on learning high-frequency fine-grained details unique to high-resolution images. Benefiting from the

strong foundation built during the high-throughput pretraining and low-resolution finetuning stages, the low-throughput high-resolution finetuning stage is data-efficient, thereby enhancing the overall efficiency of the FP-SFT process.

A meticulously curated dataset of high-resolution photorealistic image-text pairs is used for FP-SFT. Moreover, the pure-text data from OpenHermess (Teknium, 2023) and the image-to-text data from Mini-Gemini (Li et al., 2024b) are also incorporated during training to prevent catastrophic forgetting. To provide users with a natural way to specify the desired resolution, we develop a resolution-aware prompt (Figure 2). For each image and its corresponding description, the prompt is structured as follows:

Generate an image of {width}x{height} according to the following prompt: \n

{description}

Omnipotent Supervised Finetuning (Omni-SFT) While flexible photorealistic image generation is the primary target of Lumina-mGPT, we find that the resulting model after FP-SFT can be efficiently transferred to a wide spectrum of image understanding and generation tasks. We thus present Omni-SFT, a preliminary exploration toward boosting Lumina-mGPT to a visual generalist. Training tasks and data for Omni-SFT consists of the following:

- 1. Single- and multi-turn language-guided image-editing with data from MagicBrush (Zhang et al., 2024) and SEED (Ge et al., 2024) (only involving the real-world and multi-turn subsets).
- 2. Dense prediction tasks, including surface norm estimation from NYUv2 (Silberman et al., 2012) and ScanNet (Dai et al., 2017), depth estimation from Kitti v2 (Cabon et al.,

2020) and Sintel (Butler et al., 2012), pose estimation from MSCOCO (Lin et al., 2014), semantic segmentation data annotated with OneFormer (Jain et al., 2023) on image from Laion (Schuhmann et al., 2022), and grounding data from RefCOCO (Kazemzadeh et al., 2014).

- 3. In-house spatial-conditional image generation following ControlNet (Zhang et al., 2023), with conditions including surface norm, depth, pose, and segmentation.
- 4. Text-conditional multiview generation using an internal dataset consisting of 100k high-quality samples with rendered 3842 images from 16 viewpoints uniformly distributed in azimuth angles.
- 5. A small fraction of data sampled from those used in the previous FP-SFT process, including both text modeling and text-to-image generation to maintain its learned capabilities.

We tokenize all text and images into discrete tokens and formulate these tasks as a unified next-token prediction objective. Notably, we also incorporate tasks multiview genera-

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

2022) with a weight of 1 × 10−5 for both the 7B and 30B models. Additionally, dropout with a probability of 0.05 was applied specifically to the 7B model. Initially, we underestimated the significance of z-loss, as it is rarely mentioned in the literature on training multimodal large language models. However, our experiments revealed that omitting this term led to a surge in the magnitude of logits, resulting in diverging loss curves. Incorporating z-loss proved to be pivotal in enhancing training stability. Furthermore, we observed that applying z-loss significantly reduced the magnitude of logits, which in turn lowered the optimal temperature for inference-time image generation.

Parti

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

LlamaGen

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

### 3. Experiments

#### 3.1. Photorealistic Text-to-Image Generation

We first demonstrate the fundamental text-to-image generation capabilities achieved after FP-SFT. As shown in Figure 6, Lumina-mGPT can generate photorealistic images in a variety of resolutions, achieving the first native 1K autoregressive generation without cascaded models (Yu et al., 2022; Saharia et al., 2022; Chang et al., 2023; Pernias et al., 2024). Though finetuned on limited computational resources and text-image pairs, the results exhibit strong semantic coherence with intricate visual details

Lumina-mGPT

- Figure 3. Qualitative comparison on text-to-image generation. Prompts: (1) A raccoon wearing formal clothes, wearing a tophap and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of Rembrandt. (2) A teddy bear wearing a motorcycle helmet and cape is standing in front of Loch Awe with Kilchurn Castle behind him, dslr photo. (3) A tornado made of sharks crashing into a skyscraper. Painting in the style of Hokusai.

(4) A photo of an Athenian vase with a painting of pandas playing basketball in the style of Egyptian hieroglyphics.

Qualitative Comparison Compared with LlamaGen (Sun et al., 2024) and Parti (Yu et al., 2022), Lumina-mGPT achieves better visual quality and aesthetics (Fig. 3). Besides, LlamaGen and Parti do not support end-to-end generation at high resolutions (e.g., 1k) or at arbitrary aspect ratios, whereas it is supported by Lumina-mGPT. Specifically, LlamaGen only supports a fixed resolution of 512 × 512, while Parti generates 1024 × 1024 images using an additional super-resolution upsampler. Beyond AR-based approaches, we also provide a side-by-side comparison with diffusion-based counterparts by training on the same dataset, as detailed in Appendix G.

tion, which requires generating a sequence of image frames,

- as a preliminary for video generation. As demonstrated in Section 3.2, after Omni-SFT, Lumina-mGPT exhibits a general capability for completing a wide range of tasks other than text-to-image generation, indicating the potential for building a multimodal generalist along this direction.

Training Setup Though multiple tasks are involved in the SFT process, a unified next-token-prediction loss is used for all of the tasks. As Lumina-mGPT is designed as a chat model, all data are organized into single or multi-turn dialogs, with the loss applied only to the response parts. For all experiments, the AdamW (Loshchilov & Hutter, 2017) optimizer with weight decay = 0.1 and betas = (0.9,0.95) is used, and the learning rate is set to 2e-5. Inspired by the classifier-free guidance in diffusion models (Ho & Salimans,

Quantitative Comparison We evaluate the performance of Lumina-mGPT on widely used text-to-image benchmarks T2I-CompBench (Huang et al., 2023), GenEval (Ghosh et al., 2023), and DPG-Bench (Hu et al., 2024). The results are presented in Tab. 2.

- 2022), we randomly drop the context by a probability of 10% during training, as detailed in Appendix F. To accommodate the large model volume, PyTorch FSDP (Zhao et al.,
- 2023) is employed with gradient checkpointing. To increase training throughput, all data are pre-tokenized before training and are clustered according to the number of tokens, ensuring that each global batch is composed of data with similar lengths. To stabilize training, we applied z-loss (Chowdhery et al.,

Notably, Lumina-mGPT shows significant improvement over Chameleon (Team, 2024a) and surpasses LuminaNext (Zhuo et al., 2024), a state-of-the-art diffusion transformer trained on the same text-to-image dataset as LuminamGPT. Moreover, Lumina-mGPT consistently outperforms SDv1.5 and SDv2.1 (Rombach et al., 2022), becoming the first autoregressive model with competitive generation quality to SDXL (Podell et al., 2023). Nevertheless, a clear gap remains towards SOTA diffusion models like SD3 (Esser

Table 2. Quantitative comparison on text-to-image benchmarks.

GenEval ↑ DPG ↑ T2I-CompBench ↑

Methods

Two Obj. Counting Color Attri. Overall Entity Relation Attribute Overall Color Shape Texture

Diffusion Models SDv1.5 (Rombach et al., 2022) - - - 0.40 74.23 73.49 75.39 63.18 0.3730 0.3646 0.4219 Lumina-Next (Zhuo et al., 2024) 0.49 0.38 0.15 0.46 83.78 89.78 82.67 75.66 0.5088 0.3386 0.4239 SDv2.1 (Rombach et al., 2022) 0.51 0.44 0.50 0.47 - - - 68.09 0.5694 0.4495 0.4982 SDXL (Podell et al., 2023) 0.74 0.39 0.23 0.55 82.43 86.76 80.91 74.65 0.6369 0.5408 0.5637 SD3-medium (Esser et al., 2024) 0.74 0.63 0.36 0.62 91.01 80.70 88.83 84.08 - - DALL-E3 (Betker et al., 2023) 0.87 0.47 0.45 0.67 89.61 90.58 88.39 83.50 0.8110 0.6750 0.8070

###### AutoRegressive Models

LlamaGen (Sun et al., 2024) 0.34 0.21 0.04 0.32 - - - 65.16 - - Chameleon (Team, 2024a) - - - 0.39 - - - - - - Lumina-mGPT (Ours) 0.77 0.27 0.32 0.56 86.60 91.29 84.61 79.68 0.6371 0.4727 0.6034

Table 3. Evaluation on image understanding benchmarks. Method MMBench MME-p MME-r Chameleon 19.80 153.10 49.60 Lumina-mGPT 32.20 976.85 290.36 Method SEEDBench-I MMMU-val POPE Chameleon 30.50 22.40 19.40 Lumina-mGPT 50.93 27.11 70.43

et al., 2024) and DALL-E3 (Betker et al., 2023), showing the room for further progress along this direction.

On the Effectiveness of FP-SFT To further validate the effectiveness of FP-SFT, we visualize the images generated

- at different finetuning stages in Figure 4. With increasing image resolution, we observe a progressive decrease in visual artifacts introduced by VQ-VAE and the emergence of diverse fine-grained visual details. From these illustrations, we can conclude that our FP-SFT can unleash the potential of generating high-quality images from mGPT in a progressive manner.

Decoding Configuration Matters We observe the significant difference between the optimal hyperparameters for text and image generation, which inspires us to introduce a status-aware control mechanism switching settings based on the type of content being generated. We then explore how different inference configurations, such as temperature, top-k, and classifier-free guidance scale, affect the quality of generated images. For example, lower temperatures and top-k values often result in over-smoothed images, while higher values enhance detail but may introduce artifacts. Due to space limits, we defer the details to Appendix F.

#### 3.2. Omnipotent Task Unification with Lumina-mGPT

Beyond text-to-image generation, Lumina-mGPT also possesses the capability of text-only dialog and image under-

standing. After Omni-SFT, the spectrum of capabilities is further extended, covering multi-turn image editing, dense labeling, spatial-conditional image synthesis, and multiview generation. Such unification embodies the inherent benefits and potential of the decoder-only autoregressive architecture for unified modeling. Notably, the high-quality image generation ability achieved by Lumina-mGPT serves as the bedrock for such unification, without which the relevant explorations would be impossible.

To intuitively illustrate these capabilities, we provide visualizations in Figure 7 to 11 in the Appendix.

First, Lumina-mGPT works effectively as a normal LLM to handle general text-only tasks (Figure 7), such as math problem solving, coding, and commonsense reasoning, thanks to the extensive pertaining in Chameleon and our involvement of text-only data to mitigate catastrophic forgetting.

Shown in Figure 8, Lumina-mGPT can also handle various visual understanding tasks including captioning, visual question answering, and general multi-turn multi-image dialog. Akin to image generation, flexible image aspect ratios are also supported in image understanding. Quantitative results on visual understanding benchmarks (Liu et al., 2023; Yue et al., 2024; Li et al., 2023; 2024a; Fu et al., 2023) are presented in Table 3, illustrating that Lumina-mGPT significantly outperforms the original Chameleon model.

As a visual generalist, Lumina-mGPT incorporates classic visual recognition tasks. Using natural language as a unified interface, Lumina-mGPT can perform multiple high-level computer vision tasks including image segmentation, pose estimation, depth estimation, surface normal estimation, and referring object detection. See Figure 9 and 10 for examples.

Lumina-mGPT also supports image generation with versatile spatial conditions, such as depth maps, segmentation maps, normal maps, and human poses, as demonstrated in Figure 10 and 11. Beyond single-image generation, Lumina-

|[Figure 61]<br><br>[Figure 62]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|[Figure 63]|[Figure 64]<br><br>[Figure 65]<br><br>| | | |
|---|---|---|
| | | |
<br><br>| | | |
|---|---|---|
| | | |
<br><br>|[Figure 66]|[Figure 67]<br><br>[Figure 68]<br><br>| | | |
|---|---|---|
| | | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|[Figure 69]<br><br>|
|---|---|---|---|---|---|
| |[Figure 70]| |[Figure 71]| |[Figure 72]|
| |[Figure 73]| |[Figure 74]| |[Figure 75]|
| |[Figure 76]| |[Figure 77]| |[Figure 78]|

512x512 768x768 1024x1024

- Figure 4. Samples with zoom-ins generated by Lumina-mGPT in different resolution finetuning stages. The visual details continuously improve along with the progressively increasing resolution.

[Figure 79]

|[Figure 80]<br><br>EOL<br><br>|
|---|

|[Figure 81]<br><br>SOI<br><br>|
|---|

| |
|---|

| |
|---|

Postion of 1D tokens

Attention

prompt tokens image tokens

- Figure 5. Visualization of averaged attention logits from the last image token. Added indicator tokens are allocated with a large proportion of attention scores.

that the indicator tokens, such as <start-of-image> and <end-of-line>, are assigned exceptionally high attention scores. This suggests that these indicator tokens play a pivotal role in the image-generation process. For instance, the high attention allocated to the <start-of-image> token implies that much of the semantic information from text tokens is encapsulated within it. Similarly, the high attention assigned to the <end-of-line> tokens indicates their effectiveness in introducing 2-D positional information to the 1-D RoPE positional embeddings.

mGPT can produce a sequence of consistent images from multiview, as shown in Figure 11. This task marks a preliminary step toward video generation by rendering the multiview images into a coherent video.

Given the above examples, though preliminary, they showcase that Lumina-mGPT can effectively follow diverse instructions, highlighting its promising potential as a unification of various challenging tasks in one framework.

#### 3.3. Attention Visualization

To gain deeper insights into the sampling behavior of Lumina-mGPT, we visualize the average attention logits of the final image token during text-to-image generation, as illustrated in Figure 5. The results demonstrate that attention scores diminish for more distant tokens, indicating that the tokens attend more to local tokens over distant ones. This behavior is consistent with the local correlation nature of image data and also aligns with the long-term decay property inherent in the RoPE mechanism. Additionally, we observe

### 4. Conclusion

In this work, we present Lumina-mGPT, a pioneering decoder-only multimodal autoregressive model that delivers flexible, high-quality photorealistic image generation from text prompts while unifying a wide range of vision and language tasks. Leveraging our novel Unambiguous Image Representation (Uni-Rep) alongside the Flexible Progressive Supervised Finetuning (FP-SFT) and Omnipotent Supervised Finetuning (Omni-SFT) strategies, Lumina-mGPT achieves performance on par with modern diffusion models and surpasses existing autoregressive models like LlamaGen and Parti in visual quality and detail. Our extensive experiments demonstrate Lumina-mGPT’s versatility as a multimodal generalist, effectively handling tasks from textto-image synthesis to visual question answering. By releasing all code and checkpoints, we aim to foster further advancements and collaborations within the research community, paving the way toward more unified and powerful multimodal generative models.

### References

H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omernick, M., Dai, A. M., Pillai, T. S., Pellat, M., Lewkowycz, A., Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., and Fiedel, N. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3): 8, 2023.

Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., and Nießner, M. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33: 1877–1901, 2020.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Butler, D. J., Wulff, J., Stanley, G. B., and Black, M. J. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pp. 611–625. Springer, 2012.

Ding, M., Yang, Z., Hong, W., Zheng, W., Zhou, C., Yin, D., Lin, J., Zou, X., Shao, Z., Yang, H., et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34: 19822–19835, 2021.

Cabon, Y., Murray, N., and Humenberger, M. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020.

Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M.-H., Murphy, K. P., Freeman, W. T., Rubinstein, M., et al. Muse: Text-to-image generation via masked generative transformers. In International Conference on Machine Learning, pp. 4055–4075. PMLR, 2023.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., and Li, Z. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif, E., Du, N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson, K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim,

Ge, Y., Zhao, S., Li, C., Ge, Y., and Shan, Y. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, volume 36, pp. 52132–52152, 2023.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. Ella: Equip diffusion models with llm for enhanced semantic alignment, 2024.

Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv:2307.06350, 2023.

Jain, J., Li, J., Chiu, M. T., Hassani, A., Orlov, N., and Shi, H. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2989–2998, 2023.

Kazemzadeh, S., Ordonez, V., Matten, M., and Berg, T. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 787–798, 2014.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13299–13308, 2024a.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. arXiv preprint arXiv:2305.10355, 2023.

- Li, Y., Zhang, Y., Wang, C., Zhong, Z., Chen, Y., Chu, R., Liu, S., and Jia, J. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024b.
- Li, Z., Zhang, J., Lin, Q., Xiong, J., Long, Y., Deng, X., Zhang, Y., Liu, X., Huang, M., Xiao, Z., et al. Hunyuandit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024c.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pp. 740– 755. Springer, 2014.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Lu, J., Clark, C., Zellers, R., Mottaghi, R., and Kembhavi, A. Unified-io: A unified model for vision, language, and multi-modal tasks. In The Eleventh International Conference on Learning Representations, 2022.

Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., and Kembhavi, A. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26439–26455, 2024.

Pernias, P., Rampas, D., Richter, M. L., Pal, C., and Aubreville, M. Würstchen: An efficient architecture for largescale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=gU58d5QeGv.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494, 2022.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35: 25278–25294, 2022.

Silberman, N., Hoiem, D., Kohli, P., and Fergus, R. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pp. 746–760. Springer, 2012.

Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024a.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Team, K. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024b.

Teknium. Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants, 2023. URL https://huggingface.co/datasets/ teknium/OpenHermes-2.5.

Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J., Zhou, C., Zhou, J., and Yang, H. Ofa: Unifying architectures, tasks, and modalities through a simple sequenceto-sequence learning framework. In International conference on machine learning, pp. 23318–23340. PMLR, 2022.

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022.

Yu, L., Shi, B., Pasunuru, R., Muller, B., Golovneva, O., Wang, T., Babu, A., Tang, B., Karrer, B., Sheynin, S., et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv preprint arXiv:2309.02591, 2(3), 2023.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhang, K., Mo, L., Chen, W., Sun, H., and Su, Y. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36, 2024.

Zhang, L., Rao, A., and Agrawala, M. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Zhuo, L., Du, R., Xiao, H., Li, Y., Liu, D., Huang, R., Liu, W., Zhao, L., Wang, F.-Y., Ma, Z., et al. Luminanext: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024.

### A. Additional Qualitative Results

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

1024x1024 1024x1024

640x1536

[Figure 86]

[Figure 87]

[Figure 88]

640x1536 768x768 768x768

1024x1024

[Figure 89]

[Figure 90]

1280x800 1024x1024

[Figure 91]

[Figure 92]

[Figure 93]

1024x1024

1024x1024 1024x1024

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Dense Labeling

[Figure 98]

[Figure 99]

1024x1024 1024x1024

Controllable Generation

- Figure 6. Selected photorealistic images of various resolutions generated by Lumina-mGPT. At the bottom, we include two examples to demonstrate the omnipotent task unification in Lumina-mGPT.

###### 1. Multi-Turn Dialog

I'm planning a trip to Paris.

That sounds exciting! Paris is a beautiful city with a rich history and culture. Have you considered checking the weather, flight deals, and any specific activities you're interested in while

[Figure 100]

planning your trip?

No, I have not. Can you do some planning for me?

Of course! Here's a general outline of some things you can include in your itinerary when planning your trip to Paris:

- 1. Visit iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. These are all must-see attractions and offer a glimpse into Paris's architectural prowess and artistic heritage.
- 2. Take a romantic walk along the Seine River, which is a picturesque and romantic way to explore the city's central area. Stroll along the riverside, enjoying the scenery and the gentle sound of the water flowing nearby.
- 3. Explore Montmartre, one of Paris' most famous neighborhoods. With its narrow streets, colorful houses, and numerous art galleries, Montmartre is a must-visit destination for art and culture enthusiasts.
- 4. Take a food tour, as Paris is known for its delectable cuisine. Many of the city's restaurants offer a mix of traditional and modern dishes, and you can try traditional Parisian snacks and drinks at cafes and bakeries.
- 5. Visit the Musée de l'Orangerie, which is home to the famous Monet painting "Water Lilies." This museum is located in a beautiful, park-like setting with plenty of outdoor seating options, allowing you to enjoy the art in a more relaxed and enjoyable atmosphere.
- 6. Experience a traditional Parisian night out, with its lively bars, vibrant nightclubs, and numerous music and entertainment venues. Paris is known for its late-night scene and its bustling club culture.
- 7. Take a guided tour of the Palace of Versailles, which is located just outside the city and about 20 miles east of Paris. This majestic estate is an excellent day trip from the city and offers a chance to explore the opulent and extravagant lifestyles of the French aristocracy. Remember to research flight deals, local transportation options, and activities that interest you before embarking on your journey to Paris. Good luck and have a great trip!

[Figure 101]

###### 2. Math

Which is bigger? 9.9 or 9.11

[Figure 102]

The bigger value between 9.9 and 9.11 is 9.9.

A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost? Let's think step by step.

- Step 1: Identify the given information We know that the total cost of the bat and the ball combined is $1.10. We also know that the bat costs $1.00 more than the ball.
- Step 2: Create an equation Let's denote the cost of the ball as 'x' (since it's the unknown variable). Then, the cost of the bat is 'x + $1.00' (since it's $1.00 more than the ball).
- Step 3: Use the given information to form an equation Using the given information, we can create an equation: (x + $1.00) + x = $1.10
- Step 4: Simplify and solve for the unknown variable 'x' Combining the terms with 'x', we get: 2x + $1.00 = $1.10 2x + $1.00 - $1.00 = $1.10 - $1.00 2x = $0.10
- Step 5: Solve for the unknown variable 'x' To find the cost of the ball (x), we need to divide both sides of the equation by 2: x = $0.10 / 2 x = $0.05
- Step 6: State the conclusion The ball costs $0.05, which is 5 cents.

[Figure 103]

###### 3. Coding

Write a python script, implementing a sorting algorithm.

Here is a simple sorting algorithm implemented in Python: def sort_list(list):

sorted_list = list for i in range(len(sorted_list) - 1):

min_index = i for j in range(i + 1, len(sorted_list)):

if sorted_list[min_index] > sorted_list[j]: min_index = j

[Figure 104]

sorted_list[i], sorted_list[min_index] = sorted_list[min_index], sorted_list[i] return sorted_list

list = [3,6,8,1,7] print(sort_list(list))

The sort_list function implements the selection sort algorithm. This is one of the simplest sorting algorithms, but it can be quite slow and inefficient for larger datasets. It works by finding the minimum element in an unsorted part of the list and swapping it with the first unsorted element. This process continues until the entire list is sorted.

###### 4. Flexible Photorealistic Image Generation

“Generate an image of 1280x800 according to the following prompt: A serene alpine meadow in spring: The meadow is a carpet of vibrant green, dotted with colorful wildflowers in full bloom. Snow-capped mountains provide a stunning backdrop, their peaks glittering in the bright sunlight. A clear, babbling brook winds through the meadow, adding to the scene's tranquility”.

“<|image|>”

[Figure 105]

[Figure 106]

“Generate an image of 640x1536 according to the following prompt: A serene alpine meadow in spring:A rocket launching into space, captured at the peak of its ascent: The rocket, sleek and powerful, is shown against a backdrop of a twilight sky. Flames and smoke trail behind it as it pierces through the atmosphere. The intense light from the engines illuminates the scene, creating a breathtaking contrast with the darkening sky.”

“<|image|>”

[Figure 107]

[Figure 108]

###### 5. Image Caption

“Describe this image in detail.<|image|>”

[Figure 109]

“The image captures a moment on a city street that appears to be in a Southeast Asian country, likely Vietnam, based on the script on the signs, which is in Devanagari script used for several languages in India, but also used in Southeast Asia, particularly in Vietnam. The presence of English text on the signs indicates that this is a region frequented by tourists or by locals familiar with Western languages. The scene is bustling with a diverse array of individuals and goods, indicating a market or marketplace. People are seen engaged in various activities, such as walking and handling items for sale, which contributes to the impression of a busy urban environment. The image's composition shows a snapshot of daily life, illustrating the interaction between people and their surroundings in this urban setting.”

[Figure 110]

- 6. Visual Question Answering
- 7. Multi-Turn Multi-Image Dialog

“What might happen in this image in the next second?<|image|>”

[Figure 111]

“In the next second, the photographer is likely to continue taking photos of this moment. The crocodile and its open mouth are positioned to create an appealing composition for a shot. Given the open mouth and visible teeth, the photographer might capture a more dramatic and intense image by positioning themselves and the crocodile closer to the camera, aiming to include more of the crocodile's head in the frame. The next photograph could potentially feature the crocodile with its mouth open wide to mimic the one in the first image, creating a sense of continuity and action in the scene.”

[Figure 112]

“<image|>”

“Generate an image of 768x768 according to the following prompt: A piece of porcelain is placed on the table.”

[Figure 113]

[Figure 114]

“What if I use the tool shown in this image to hit the porcelain?<|image|>”

[Figure 115]

“Based on the image, the porcelain teapot with its intricate floral designs and golden trim is shown in a stationary position. However, the tool depicted—a red, metallic hammer with a black rubber grip—is designed for striking purposes. If used to hit the teapot, the hammer's weighty metal head would likely cause the teapot to shatter or chip due to its fragile nature, especially if struck with any force applied from above as a hammer would. “

[Figure 116]

8. Image Editing

“<image|>”

[Figure 117]

[Figure 118]

“Generate an image of 512x512 according to the following prompt: A cup of coffee is placed on the table.”

“<image|>”

[Figure 119]

[Figure 120]

“Remove the cup in the background.”

“<image|>”

[Figure 121]

[Figure 122]

“Add a spoon.”

- 9. Image Segmentation

12. Surface Normal Estimation

- 10. Pose Estimation
- 11. Depth Estimation

“Semantic segmentation. <|image|>”

[Figure 123]

“<|image|>”

[Figure 124]

[Figure 125]

“Human pose estimation. <|image|>”

[Figure 126]

“<|image|>”

[Figure 127]

[Figure 128]

Human pose estimation.

“Depth estimation. <|image|>”

[Figure 129]

“<|image|>”

[Figure 130]

[Figure 131]

“Surface normal estimation. <|image|>”

[Figure 132]

“<|image|>”

[Figure 133]

[Figure 134]

13. Referring Object Detection

“Detect: The red apple. <|image|>”

“<|image|>”

[Figure 135]

[Figure 136]

[Figure 137]

“Detect: The green apple. <|image|>”

“<|image|>”

[Figure 138]

[Figure 139]

[Figure 140]

- 14. Depth Conditioned Image Generation

“Generate an image according to the provided image, and according to the following caption: An ice cream with pink color. <|image|>”

“<|image|>”

[Figure 141]

[Figure 142]

[Figure 143]

Human pose estimation.

“Generate an image according to the provided image, and according to the following caption: An ice cream with vanilla flavor. <|image|>”

“<|image|>”

[Figure 144]

[Figure 145]

[Figure 146]

“Generate an image according to the provided image, and according to the following caption: An ice cream with golden textured background. <|image|>”

“<|image|>”

[Figure 147]

[Figure 148]

[Figure 149]

- 16. Surface Normal Conditioned Image Generation
- 17. Pose conditioned Image Generation

[Figure 150]

[Figure 151]

[Figure 152]

15. Segmentation Conditioned Image Generation

“Generate an image according to the provided image, and according to the following caption: A butterfly is flying in the forest. <|image|>”

[Figure 153]

“Generate an image according to the provided image, and according to the following caption: A birthday cake. <|image|>”

“Generate an image according to the provided image, and according to the following caption: A woman is holding a bag. <|image|>”

“<|image|>”

[Figure 154]

“<|image|>”

“<|image|>”

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

- 18. Multi-view Image Generation

“Generate 16 object views of 384x384 according to the following prompt: A red samurai helmet with a black face guard and green horns on the top, the surface is decorated with yellow and purple patterns, 3d asset”

“ <|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|> <|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|> ”

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

Figure 11. Lumina-mGPT as a generalist for various multimodal tasks.

### B. Related Work

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

| |
|---|

| |
|---|

OriginalImageLumina-mGPTVQ-VAESD-XLVAE

Multimodal Large Language Models Recent large language models (LLMs) (Chowdhery et al., 2022; Touvron

[Figure 182]

[Figure 183]

[Figure 184]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- et al., 2023; Brown et al., 2020; Achiam et al., 2023; Team et al., 2023) demonstrate strong instruction-following and reasoning capabilities, coupled with extensive world knowledge. To extend LLMs’ expertise from the text domain to multimodal domains such as images and videos, prior works (Liu et al., 2023; Lin et al., 2023a; Maaz et al., 2023; Lin et al., 2023b) have aligned pretrained encoders from various modalities with LLMs by curating multimodal instruction tuning datasets. Although these multimodal large language models (MLLMs) have exhibited powerful visual understanding capabilities, their multimodal functionality is primarily limited to perceiving the visual world, exemplified by tasks such as visual question answering and image captioning, rather than generating visual outputs. Another line of research (Koh et al., 2024; Dong et al., 2024; Sun et al.,

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

| |
|---|

| |
|---|

[Figure 191]

[Figure 192]

[Figure 193]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

| |
|---|

| |
|---|

[Figure 200]

[Figure 201]

[Figure 202]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

| |
|---|

| |
|---|

[Figure 209]

[Figure 210]

[Figure 211]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 12. Reconstruction quality of different methods. LuminamGPT means first encode using VQ-VAE encoder, then send the latent to Lumina-mGPT using image editing system prompt with the instruction "no edit", and finally decode the newly generated latents using VQ-VAE decoder.

- 2023; Wu et al.) has proposed augmenting MLLMs with the ability to generate images, videos, and audio from text instructions. These approaches introduce additional visual tokens for generation and align these generative tokens as conditional information with a pretrained generator, such as Stable Diffusion (Esser et al., 2024; Podell et al., 2023) for text-to-image generation. Consequently, the generation capabilities heavily rely on the external expert generator rather than MLLMs themselves, resulting in inconsistent and inferior generation results. To combine the strength of both approaches, our model aims to learn both understanding and generation of images using an MLLM with native multimodal capabilities, drawing inspiration from Chameleon (Team, 2024a), a mixed-modal early-fusion foundation model.

Text-to-Image Generation The task of text-to-image generation seeks to synthesize photorealistic and diverse images based on textual descriptions. Nowadays, diffusion models, whether in pixel space (Saharia et al., 2022) or in latent space (Podell et al., 2023; Esser et al., 2024; Pernias et al., 2024), have become the de-facto approaches due to their superior performance, particularly in producing extremely high-aesthetic images. Among these models, the recent trend of scaling diffusion transformers (DiTs) (Chen et al.,

- 2024b;a; Gao et al., 2024; Esser et al., 2024; Li et al., 2024; Team, 2024b) suggests a unified architecture for both text and image modeling. However, existing DiTs still leverage separate language models, such as CLIP (Radford et al.,

less attention in recent years. Early works (Ramesh et al., 2021; Ding et al., 2021) proposed a two-stage generation approach: first, training a VQ-VAE (Van Den Oord et al., 2017; Esser et al., 2021) for image tokenization and detokenization, and then using an autoregressive transformer to model discrete image token sequences, akin to language modeling. Parti (Yu et al., 2022) scaled up the autoregressive transformer to 20 billion parameters, demonstrating promising high-fidelity image generation results. LlamaGen (Sun et al., 2024) further improved the image tokenizer and integrated advanced techniques in LLMs, bridging the performance gap with diffusion counterparts. Unlike Parti and LlamaGen, Lumina-mGPT proposes multimodal generative pertaining on unified text-image sequences, followed by supervised finetuning on high-quality text-to-image pairs, achieving flexible high-aesthetic image generation with autoregressive models.

### C. Discussions on Reconstruction Quality

VQ-VAEs (Van Den Oord et al., 2017; Razavi et al., 2019; Esser et al., 2021) compress images at the cost of information loss, which introduces quality degradation at reconstruction, especially for high-frequency details such as edges, hair, and text. As Generative models such as Lumina-mGPT only has access to the VQ-VAE latents during training and cannot access original images, intuitively the VQ-VAE reconstruction quality should somehow build an upper bound for such models’ image generation quality.

- 2021) or T5 (Raffel et al., 2020), as text encoders. This modality gap between text and image representations not only leads to inaccurate generation but also hinders the development of a unified multimodal foundational generative model. Compared to the dominance of diffusion models, the progress of autoregressive image generation has received

However, we observe an interesting and counter-intuitive

phenomenon. Given two data flows:

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

ControllableGenerationImageGenerationDenseLabeling

- 1. Image −−−−−−−−→VQVAEEncoder latent−−−−−−−−→VQVAEDecoder Recon1
- 2. Image −−−−−−−−→VQVAEEncoder latent−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−→Lumina-mGPTusingeditingsystempromptwithinstruction"noedit" latent2−−−−−−−−→VQVAEDecoder Recon2

Depth Detection

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

We surprisingly find that the quality of Recon2 sometimes surpasses that of Recon1, and we show such cases in Fig. 12. For reference, the reconstruction results using SDXL VAE (Podell et al., 2023) are also presented. Note that while Lumina-mGPT has been trained on the image editing task, it is not trained with the "no edit" instruction. This intriguing observation may suggest some meaningful insights. For example, it may possibly indicate that the latents encoded by VQ-VAE encoder could contain noises of certain patterns that can be learned and even corrected. We leave the further exploration of this phenomenon for future work.

Surface Normal Human Pose

[Figure 220]

[Figure 221]

[Figure 222]

Figure 13. Failure cases of current Lumina-mGPT. Due to inadequate training and limited data size, Lumina-mGPT sometimes struggles to understand input conditions and produce visual artifacts.

### D. Limitations of Existing Approaches

Randomly-Initialized Transformer While transfer learning has revolutionized key fields such as visual recognition (He et al., 2016; Radford et al., 2021; Lu et al., 2019) and language generation (Raffel et al., 2020; Radford et al., 2018; 2019; Brown et al.,

- 2020), popular autoregressive image generation approaches such as DALL-E, Parti, and LlamaGen all adopt a randomly-initialized causal transformer, which fails to utilize pretrained transferable representation and large-scale datasets. As a result, AR-based approaches often lead to poor image generation quality and slow convergence without leveraging proper large-scale pretraining.

Verbose Encoder-Decoder Architeture DALL-E and CogView initially propose using a decoder-only transformer for image generation with discrete representation, where a single transformer acts as both a text encoder and an image token decoder. However, subsequent approaches, such as Parti and LlamaGen, adopt a verbose encoder-decoder architecture that injects frozen T5 text features (Raffel et al., 2020) using cross-attention or prefix-filling approaches, motivated by the findings of Imagen (Saharia et al., 2022). Compared to the trend in scaling LLMs (Touvron et al., 2023; Bai et al., 2023; Bi et al., 2024), such encoder-decoder architecture is cumbersome due to the decoupling of text encoding and image token modeling. This design significantly complicates the autoregressive-decoding framework, limits the scalability of image generation, and hinders the generalization to additional modalities and tasks.

Limited Decoding Resolution and Flexibility Natural images exsit in various resolutions and aspect ratios. Advanced diffusion models (Chen et al., 2024a;b; Esser et al., 2024; Team, 2024b; Li et al., 2024; Zhuo et al., 2024; Lu et al., 2024b) can successfully generate diverse photorealistic images at arbitrary resolution with skewed ratios. In contrast, current AR-based approaches (Yu et al., 2022; Team, 2024a; Sun et al., 2024) rely on central-cropping a low-resolution 512 × 512 image and transforming the cropped lowresolution image into a fixed-length sequence of discrete tokens using a pretrained Vector-Quantized Variational Autoencoder (VQVAE) (Van Den Oord et al., 2017; Razavi et al., 2019; Esser et al.,

- 2021). This approach simplifies autoregressive training but at the cost of deteriorated image quality and generation flexibility.

ever, previous AR-based image generation approaches (Ramesh et al., 2021; Ding et al., 2021; Yu et al., 2022; Sun et al., 2024) have been limited to text-to-image generation without exploring the unification with other tasks, such as dense labeling and controllable image generation. This lack of task extensibility significantly constrains the applicability of AR-based image generation models to a broader range of scenarios.

### E. Limitations of Lumina-mGPT

Failure Cases Despite Lumina-mGPT can generate photorealistic images, it sometimes produces images with noticeable visual artifacts. For example, Lumina-mGPT may generate people and animals with unreasonable limbs, as shown in the first row of Figure 13. Besides, compared to SoTA text-to-image generation approaches including SD3 (Esser et al., 2024), Kolors (Team, 2024b), and HunyuanDiT (Li et al., 2024), all pretrained over 1B image-text pairs, Lumina-mGPT’s prompt-following ability is inferior due to the limited training resources and data size, which are many times smaller than these SoTA methods. Regarding dense labeling and controllable generation, Lumina-mGPT currently showcases preliminary results with a limited training budget. Hence, the second row in Figure 13 provides such an example where Lumina-mGPT produces inaccurate predictions or semantically inconsistent images, failing to understand the given image conditions. Therefore, we expect by scaling data size with more computational resources, Lumina-mGPT can effectively address the above failure cases such as inadequate instruction-following ability and visual artifacts.

Generation Speed Autoregressive models require numerous network evaluations during inference due to the nature of nexttoken prediction, similar to the iterative denoising process in diffusion models. This becomes worse when generating high-resolution images, which often require minutes to generate a full sequence of image tokens, significantly slower than current diffusion models with advanced samplers. However, there have been plenty of techniques to optimize the inference speed designed for autoregres-

Poor Task Extensability Autoregressive modeling excels at unified generative modeling for various tasks and modalities using discrete tokens (Wang et al., 2022; Lu et al., 2022; 2024a). How-

T=0.7 T=0.8 T=0.9 T=1.0

CFG

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Top-k=2000,CFG=4.0T=1.0,CFG=4.0T=1.0,Top-k=2000

- 1
- 2

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Top-k=500

Top-k=50 Top-k=2000 Top-k=8192

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

CFG=1.0 CFG=2.0 CFG=4.0 CFG=8.0

8

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Figure 15. Samples generated by Lumina-mGPT using different Top-k and Temperature.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Different Decoding Hyperparameters for Image and Text The sampling strategy of autoregressive models involves numerous hyperparameters that significantly influence the sampling results. We find that the optimal decoding hyperparameters differ greatly between text decoding and discrete image code decoding. For example, the top-k=5 setting performs well in generating text. However, when generating images, the value of top-k should be much larger (e.g. 2000) to avoid repetitive and meaningless patterns. Therefore, we implement a status-aware control mechanism for inference. Specifically, a set of default hyperparameters is used for text decoding; once a <start-of-image> token is generated, the hyperparameters switch to those optimized for image generation. After the <end-of-image> token is generated, the parameters revert to the initial settings.

20

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

100

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

400

Figure 14. Samples generated by Lumina-mGPT using different CFG; T=1.0, Top-k=2000.

Classfier-Free Guidance Classifier-Free Guidance (CFG) (Ho & Salimans, 2022) is originally proposed to enhance the quality and text alignment of generated samples in text-to-image diffusion models. We incorporate this technique into autoregressive models during inference. When generating an image token, the CFG-processed logits lcfg are formulated as lcfg = l + s(l − l′), where l represents the original logits conditioned on the complete context; l′ represents the context-independent logits, which are conditioned solely on the tokens following the <start-of-image> token of the currently generating image, and are independent of any prior context; s denotes the guidance scale of Classifier-Free Guidance. To make CFG work, during training, the context before <start-of-image> is randomly dropped by a probability of 10%. In practice, KV cache can be used for accelerating the computation of both l and l′. As shown in Fig. 14, similar to the trend of diffusion models, increasing CFG initially raises the quality and stability of generation, but increasing it further would make the quality deteriorate.

sive models, such as vLLM (Kwon et al., 2023) and FlashAttention (Dao et al., 2022; Dao, 2024). We believe that by integrating these approaches in the future, Lumina-mGPT can achieve a remarkable speed up during inference.

VQ-VAE Reconstruction Quality VQ-VAE is employed as the image tokenizer to convert continuous images into discrete token representations. Meanwhile, it also introduces information bottlenecks by compressing the spatial dimensions of images. As a result, the reconstruction quality of VQ-VAE largely determines the upper limit of generation quality. We discover that the VQVAE proposed in Chameleon sometimes struggles to reconstruct high-frequency details, especially when text and human faces are present in images. Incorporating further improvements on VQVAE, such as FSQ (Mentzer et al., 2023), may also enhance the generation quality of Lumina-mGPT.

### F. Inference Configuration of Lumina-mGPT

In autoregressive models, various configuration parameters during Lumina-mGPT’s decoding stage significantly affect sample quality (Holtzman et al., 2020; Radford et al., 2018; 2019). Hyperparameters such as temperature (T), top-k, and classifier-free guidance scale (CFG) have not been extensively investigated in the visual domain. In this section, we explore how these hyperparameters influence the generated image in terms of quality, texture, and style.

Influence of Temperature To evaluate the effect of these decoding parameters, we first set a standard decoding configuration: T=1.0, top-k=2000, CFG=4.0, which serves as a good-to-use setting. From this baseline, we gradually shift T from 0.7 to 1.0 to generate corresponding images at different temperatures. As shown in Figure 15, it is evident that when setting the temperature low, visual details diminish and objects tend to be over-smoothed. Conversely, when setting the temperature high, the generated images contain rich visual content but are prone to contain more artifacts.

Seed: 1 Seed: 2 Seed: 3 Seed: 4

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Lumina-mGPTLumina-Next-SFTLumina-AiR-3Lumina-AiR-2

Lumina-mGPTLumina-Next-SFT

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

Lumina-Next-SFTLumina-mGPT

- Figure 17. Text rendering comparison between Lumina-Next-SFT and Lumina-mGPT. From left to right, the correct texts to be rendered on the image are: “Bloom”, “Hope”, “Brew”, and “Joy”.

Lumina-mGPTLumina-Next-SFT

[Figure 279]

is wearing .

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

prompt1 prompt2 prompt3 prompt4

prompt1: is wearing . Prompt2: is sleeping on .

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

- prompt3: 日落时分的海滩，阳光透过云层在海面上撒下一片金色的光芒，海浪轻轻拍打着沙滩，远处 的帆船在余晖中若隐若现。
- prompt4: 森林中的清晨，阳光穿过茂密的树冠，形成斑驳的光影，地上覆盖着厚厚的落叶，几只小鹿 在林间悠闲地觅食。

- Figure 18. Emoji and multilingual instruction understanding ability comparison between Lumina-Next-SFT and Lumina-mGPT. Lumina-mGPT struggle to understand emojis and multilingual prompts.

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

Figure 16. Comparison between Lumina-Next-SFT and LuminamGPT using different random seeds. Images generated by LuminamGPT exhibit comparable aesthetics with greater diversity.

Influence of Top-k Based on the standard decoding setting, we vary the top-k value, from 50 to 8192, where 8192 is equal to the size of the VQ-VAE codebook used usedin Lumina-mGPT. The results, visualized in Figure 15, indicate a similar trend with increasing temperature. When top-k is low, the image content and texture are relatively simple, exhibiting the over-smoothed problem as well. When top-k is set high, the image detail and texture are diverse, making it more aesthetically appealing, while increasing the potential of artifacts.

### G. Comparison with Diffusion-based Appraoches

For a long period of time, diffusion models have dominated the field of text-to-image generation compared to autoregressive models. Although LlamaGen claims to beat diffusion models, their results are limited to the ImageNet benchmark and there has been no direct comparison between these two architectures so far. In this section, we aim to provide a detailed comparison of autoregressive and diffusion-based methods trained on these same text-image datasets, focusing on image quality, diversity, text-rendering, and multilingual capabilities. Specifically, we adopt Lumina-mGPT and Lumina-Next-SFT (Zhuo et al., 2024) as representatives of autoregressive and diffusion-based methods, respectively. A direct visual comparison between Lumina-Next-SFT and Lumina-mGPT reveals both the similarities and differences between autoregressive and diffusion-based generative modeling approaches.

On the Similarity between Diffusion- and AR-based Generation Given the same set of text prompts, both diffusion- and AR-based approaches generate photorealistic images with similar aesthetic style and fine-grained details, illustrated in Figure 16. This reveals the fact that both architectures can achieve satisfac-

tory text-to-image generation performance when provided with the same training data, training budget, and comparable model sizes. The AR-based methods display remarkable visual aesthetics on par with their diffusion counterparts, challenging the notion that diffusion models are more effective and promising architecture in generative modeling. This finding also aligns with the platonic representation hypothesis (Huh et al., 2024) that neural networks are converging to learn a shared representation space despite being trained with different architectures and objectives. Therefore, this hypothesis highlights the importance of collecting more highquality data and optimizing training infrastructure as directions for data and model scaling, to boost the overall model performance that is agnostic to any specific architecture.

On the Differences between Diffusion- and AR-based Generation As shown in Figure 16, Lumina-mGPT exhibits more diversity using different random seeds, while Lumina-Next-SFT generates similar images with identical layouts and textures. This can be partly attributed to the use of high temperature and top-k values in Lumina-mGPT. However, excessive diversity also causes our model to be less stable and more prone to producing visual artifacts, which is discussed in Section E.

We also compare the text rendering and multilingual understanding capabilities between Lumina-mGPT and Lumina-Next-SFT. As illustrated in Figure 17, Lumina-mGPT exhibits significantly better text synthesizing results, while Lumina-Next-SFT struggles to generate any complete character. We argue that this underscores the importance of mGPT, where the model learns a seamless multimodal representation between text and images using massive interleaved data during the pertaining stage. However, when it comes to multilingual understanding, Lumina-mGPT performs worse than Lumina-Next-SFT in terms of emoji and Chinese prompts shown in Figure 18. The reason is that although Lumina-mGPT learns better text-image alignment, the lack of multilingual text corpus used in pertaining limits its performance. In contrast, the text encoder used in Lumina-Next-SFT showcases significantly stronger multilingual capabilities than Chameleon. Hence, we hope that by comprehensively enhancing the capabilities of the base mGPT model, such as adding more multilingual data, Lumina-mGPT can benefit in all downstream tasks.

In addition to text-to-image generation, Lumina-mGPT supports various vision and language tasks within a unified framework. However, the design of diffusion models limits their compatibility and performance across multiple modalities and tasks. They often require specific architecture designs and additional training for each unseen task (Ke et al., 2024; Xu et al., 2023). In contrast, Lumina-mGPT treats input from all modalities as multimodal token sequences and leverages natural language as the interface to unify diverse tasks with next-token prediction.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Bi, X., Chen, D., Chen, G., Chen, S., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fu, Z., et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., and Li, Z. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024a.

Chen, J., YU, J., GE, C., Yao, L., Xie, E., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id= eAKmQPe3m1.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif,

E., Du, N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson, K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim, H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omernick, M., Dai, A. M., Pillai, T. S., Pellat, M., Lewkowycz, A., Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., and Fiedel, N. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Ding, M., Yang, Z., Hong, W., Zheng, W., Zhou, C., Yin, D., Lin, J., Zou, X., Shao, Z., Yang, H., et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021.

Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., Kong, X., Zhang, X., Ma, K., and Yi, L. DreamLLM: Synergistic multimodal comprehension and creation. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=y01KGvd9Bw.

Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Gao, P., Zhuo, L., Lin, Z., Liu, C., Chen, J., Du, R., Xie, E., Luo, X., Qiu, L., Zhang, Y., et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.

He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 630–645. Springer, 2016.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=rygGQyrFvH.

Huh, M., Cheung, B., Wang, T., and Isola, P. The platonic representation hypothesis. In International Conference on Machine Learning, 2024.

Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R. C., and Schindler, K. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9492–9502, 2024.

Koh, J. Y., Fried, D., and Salakhutdinov, R. R. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36, 2024.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Li, Z., Zhang, J., Lin, Q., Xiong, J., Long, Y., Deng, X., Zhang, Y., Liu, X., Huang, M., Xiao, Z., et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.

Lin, B., Zhu, B., Ye, Y., Ning, M., Jin, P., and Yuan, L. Videollava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023a.

Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023b.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.

Lu, J., Batra, D., Parikh, D., and Lee, S. Vilbert: Pretraining taskagnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.

Lu, J., Clark, C., Zellers, R., Mottaghi, R., and Kembhavi, A. Unified-io: A unified model for vision, language, and multimodal tasks. In The Eleventh International Conference on Learning Representations, 2022.

Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., and Kembhavi, A. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26439–26455, 2024a.

Lu, Z., Wang, Z., Huang, D., Wu, C., Liu, X., Ouyang, W., and BAI, L. Fit: Flexible vision transformer for diffusion model. In Forty-first International Conference on Machine Learning, 2024b. URL https://openreview.net/forum?id= jZVen2JguY.

Maaz, M., Rasheed, H., Khan, S., and Khan, F. S. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.

Mentzer, F., Minnen, D., Agustsson, E., and Tschannen, M. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023.

Pernias, P., Rampas, D., Richter, M. L., Pal, C., and Aubreville, M. Würstchen: An efficient architecture for large-scale textto-image diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=gU58d5QeGv.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. Improving language understanding by generative pre-training. 2018.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever,

I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-to-image generation. In International conference on machine learning, pp. 8821– 8831. Pmlr, 2021.

Razavi, A., Van den Oord, A., and Vinyals, O. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

- Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., and Wang, X. Emu: Generative pretraining in multimodality. In The Twelfth International Conference on Learning Representations, 2023.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024a.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Team, K. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024b.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J., Zhou, C., Zhou, J., and Yang, H. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International conference on machine learning, pp. 23318–23340. PMLR, 2022.

Wu, S., Fei, H., Qu, L., Ji, W., and Chua, T.-S. Next-gpt: Any-toany multimodal llm. In Forty-first International Conference on Machine Learning.

Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., and De Mello, S. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2955–2966, 2023.

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022.

Zhuo, L., Du, R., Xiao, H., Li, Y., Liu, D., Huang, R., Liu, W., Zhao, L., Wang, F.-Y., Ma, Z., et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024.

