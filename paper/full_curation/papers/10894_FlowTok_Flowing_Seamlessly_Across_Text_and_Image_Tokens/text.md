# arXiv:2503.10772v3[cs.CV]26Nov2025

## FlowTok: Flowing Seamlessly Across Text and Image Tokens

Ju He1 Qihang Yu1 Qihao Liu2 Liang-Chieh Chen1 1 ByteDance Seed 2 Johns Hopkins University https://tacju.github.io/projects/flowtok.html

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

Figure 1. Text-to-Image Generation Results by FlowTok. FlowTok projects both text and images into a unified, compact 1D latent space, enabling direct flow matching between 1D tokens and facilitating the efficient generation of diverse, high-fidelity images.

### Abstract

Bridging different modalities lies at the heart of crossmodality generation. While conventional approaches treat the text modality as a conditioning signal that gradually guides the denoising process from Gaussian noise to the target image modality, we explore a much simpler paradigm—directly evolving between text and image modalities through flow matching. This requires projecting both modalities into a shared latent space, which poses a significant challenge due to their inherently different representations: text is highly semantic and encoded as 1D tokens, whereas images are spatially redundant and represented as

2D latent embeddings. To address this, we introduce FlowTok, a minimal framework that seamlessly flows across text and images by encoding images into a compact 1D token representation. Compared to prior methods, this design reduces the latent space size by 3.3× at an image resolution of 256, eliminating the need for complex conditioning mechanisms or noise scheduling. Moreover, FlowTok naturally extends to image-to-text generation under the same formulation. With its streamlined architecture centered around compact 1D tokens, FlowTok is highly memory-efficient, requires significantly fewer training resources, and achieves much faster sampling speeds—all while delivering performance comparable to state-of-the-art models. Code is available at https://github.com/TACJu/FlowTok.

|CLIP Text Embedding legendText Token Image Token<br><br>|
|---|

Conditioning Signal

“a cutesy plushy cat”

Text

N x C

[Figure 16]

[Figure 17]

Flow Matching

Noise

Image

H x W x D

H x W x D

#### Text as Conditions Direct Flow between Modalities

Text-to-Image Generation

[Figure 18]

“a cutesy

Text plushycat” Flow Matching Image

N x D N x D

Image-to-Text Generation

Figure 2. Text as Conditions vs. Direct Flow between Modalities. Top: Conventional text-to-image generation relies on the diffusion process, where text serves as a conditioning signal to guide the denoising process. Bottom: The proposed FlowTok enables direct flow between text and image modalities by projecting both into a shared, compact 1D latent space, facilitating seamless generation of both.

### 1. Introduction

Bridging different modalities is essential for comprehending the diverse forms of data that represent our world, encompassing both understanding and generation. In multimodal understanding, extensive research has focused on designing architectures that project different modalities into a shared latent space [4, 18, 22, 32, 36, 37, 45, 46, 55, 65, 85]. These approaches have significantly advanced cross-modal representation learning and real-world understanding by leveraging a common latent space between modalities.

In contrast, multimodal generation (e.g., text-to-image generation) follows a different paradigm, primarily relying on the diffusion process [35, 52, 61, 75, 76], where the source modality (e.g., text) serves as a conditioning signal to guide the denoising process. Various conditioning mechanisms have been explored, including concatenation [9], cross-attention [14, 70], conditioning embeddings [61], and hybrid strategies [24, 39]. While effective, these approaches introduce substantial complexity, requiring intricate conditioning mechanisms and noise scheduling. This naturally raises an important question: Can we unify multimodal understanding and generation by enabling direct transitions within a shared latent space?

To address this question, we revisit flow matching [5, 49, 54]—a modern generative framework that learns a direct path from noise to data, enabling faster convergence and accelerated sampling, leading to state-of-the-art multimodal generation results [24, 41]. Unlike diffusion models, flow matching is not constrained to using noise as the

source distribution; instead, it only requires the source and target distributions to share the same shape. Pioneering works [5, 50, 73, 82, 96] have demonstrated its effectiveness in learning direct mappings within the same modality (e.g., image-to-image generation). Meanwhile, CrossFlow [51] extends flow matching to cross-modal learning by mapping text into a 2D latent space to match the shape of image embeddings, paving the way for new possibilities. However, while this approach simplifies the overall pipeline, it still operates on 2D latent representations. As a result, the additional computational overhead introduced by the text variational autoencoder [40] in CrossFlow makes it slower than modern text-to-image diffusion models like SD1.5 and SD2.1 [70], ultimately contradicting its original goal of efficiency.

To this end, we introduce FlowTok, a minimal framework that enables seamless Flowing of Tokens across text and image—the two most prevalent modalities (Fig. 2). At the core of FlowTok, both text and images are encoded into compact 1D latent tokens within a unified space, enabling direct flow matching between them. On the text side, FlowTok employs a pre-trained text encoder [65] to extract initial 1D text embeddings. Since these embeddings typically reside in a higher-dimensional space than image latents, FlowTok introduces a lightweight text projector to map text embeddings into a low-dimensional variational latent space. On the image side, FlowTok builds on recent advancements in image tokenization [39, 93] to encode images into compact 1D latent tokens. Specifically, we enhance TA-TiTok [39] by integrating RoPE [78] and SwiGLU FFN [72], improving positional information handling and reconstruction quality.

To enable direct flow matching, we align the number of image latent tokens K in TA-TiTok to match the text encoder’s output sequence length (K = 77 for CLIP text encoder).

By integrating these simple yet effective designs across text and image modalities, FlowTok represents both in the same 1D low-dimensional space with shape 77 × 16 (77 tokens, each with 16 dimensions). This compact representation is 3.3× smaller than typical 2D flow matching shapes [77] of 32 × 32 × 4 for image resolutions of 256. This alignment enables fast, direct flow matching and seamless evolution between the two modalities.

Unlike standard flow matching models [49, 54, 57], FlowTok eliminates the need for intricate conditioning mechanisms, offering a fully self-attention-based generative model. This allows for direct flow across modalities without additional complexity. Unlike CrossFlow [51], which converts text into 2D embeddings, FlowTok retains the 1D structure of text embeddings, avoiding the need for flattening and transformation into 2D. This simplifies the framework while eliminating reliance on heavy parametric contrastive losses for semantic preservation.

As a result, FlowTok offers a streamlined and resourceefficient training process. Its largest variant, FlowTok-H (1.1B), supports a batch size of 8K on 8 A100 GPUs without requiring gradient checkpointing or gradient accumulation. In contrast, recent text-to-image models of similar scale typically require 32 to 64 A100 GPUs to train with a batch size of only 2K [14, 70]. Moreover, FlowTok converges significantly faster, as shown in Fig. 3a, with FlowTok-H completing training in just 26.1 8-A100 days—far less than SD 2.1 [70], which requires 1041.6 8-A100 GPU days.

Inference is also highly efficient, with FlowTok achieving over 10× the throughput of Show-o [87] and CrossFlow [51], as shown in Fig. 3b. This dramatically reduces computational costs, making text-to-image research far more accessible. To ensure full reproducibility, we train FlowTok exclusively on publicly available datasets, avoiding reliance on high-quality proprietary data. Remarkably, despite its minimalist design and reduced data requirements, FlowTok achieves state-of-the-art text-to-image performance.

Beyond text-to-image generation, FlowTok seamlessly extends to image-to-text generation, maintaining strong performance under the same minimalist framework. We believe our work establishes a strong foundation for future research in generalized cross-modality generation.

### 2. Related Work

Flow Matching. Flow matching [5, 49, 54] models generative processes by constructing a transport map between two distributions via an ordinary differential equation (ODE). It has recently gained traction as the foundation for stateof-the-art text-to-image and text-to-video synthesis models [16, 24, 63, 67, 69], offering faster training and sampling

| | | | | |
|---|---|---|---|---|
| | | |SD-2.1| |
|lowT|ok-XL| | | |
|Flo|wTok-H|CrossFlow|SD-1.|5|
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
|ross|Flow| |Flo|wTok-XL| |
| | | |FlowTo|k-H| |
|ho|w-o| | | | |
| | | | | | |

15.0

15.0

12.0

12.0

COCOFID-30K

COCOFID-30K

F

C

10.0

10.0

9.0

9.0

8.0

8.0

30

20 50 100 1000

1 5 10 20

Training Costs (8-A100 days)

Inference Speed (imgs / second)

(a) FID vs. Training Costs.

(b) FID vs. Inference Speed.

Figure 3. COCO Results. FlowTok presents comparable performance to previous methods on COCO while significantly reducing training resource requirements (Fig. 3a) and achieving much faster sampling speed (Fig. 3b). This efficiency stems from its minimalist design centered around 1D tokens, which facilitates direct transformation between text and image modalities, leading to superior performance with enhanced computational efficiency. We note that the compared CrossFlow [51] uses high-quality proprietary data.

compared to conventional diffusion methods [35, 52, 53, 61]. Several works further optimize flow trajectories by minimizing curvature [43, 64, 81]. Despite its theoretical flexibility in handling arbitrary distributions, recent approaches primarily evolve noise into target distributions, often relying on complex control signal conditioning, which complicates the pipeline and overlooks the potential of directly transforming control signals into target distributions. In contrast, only a few works [5, 50, 54, 73, 82, 96] explore direct transport within the same modality (e.g., image-to-image [26, 54, 96]), leaving cross-modal transport (e.g., text-to-image) underexplored. In this work, we introduce FlowTok, a minimal yet effective framework that enables seamless flow across text and image modalities using 1D tokens. Unlike CrossFlow [51], which follows a similar paradigm but relies on 2D latent representations and incurs additional computational costs due to the text variational encoder, FlowTok operates within a unified, compact 1D token space. This design achieves a 3.3× compression rate in latent size, significantly reducing training costs and accelerating the sampling process, all while maintaining state-of-the-art performance.

Text-to-Image Generation. Text-to-image generation has advanced rapidly in recent years, driven by various generative paradigms, including diffusion models [14, 62, 70, 71], flow matching models [16, 24, 74, 88, 94], sequence models [25, 28, 68, 91, 92], and masked generative models [7, 11, 39, 86]. While early works in each category establish the foundation for their respective approaches, subsequent advancements across different model types have primarily emerged from three key areas: careful data collection and advanced image recaptioning for improved data quality [10, 20, 39], architectural and conditioning improvements for faster convergence and better text-image align-

ment [14, 16, 24, 25], and micro-conditioning for finer control over generated samples [7, 39, 62]. By contrast, this work introduces a minimalist framework FlowTok that directly maps text tokens to image tokens, eliminating the need for noise scheduling and complex conditioning mechanisms. This streamlined design enhances both efficiency and simplicity while maintaining competitive performance.

### 3. Preliminary

- 1D Visual Tokenization [39, 93] deviates from traditional
- 2D grid-based latent tokenization by adopting a compact 1D representation, eliminating the need to preserve the 2D spatial structure. This work focuses on continuous 1D visual tokens for flow matching. During tokenization, given an input image I ∈ RH×W×3, the image is downscaled by a fac-

H f ×Wf ×D. These patches are concatenated with a set of latent tokens L ∈ RK×Dto form a sequence that is passed through a Vision Transformer (ViT) [21] encoder, Enc, to generate embeddings. Only the embeddings corresponding to the latent tokens are retained, forming a compact 1D latent representation. This representation is modeled as a Gaussian distribution with KL divergence regularization, resulting in a compact 1D VAE representation, ZI ∈ RK×D. In the de-tokenization phase, text guidance is applied by incorporating text embeddings generated by a pre-trained text encoder [65]. These text embeddings are projected through a linear layer to align with the channel dimensions of ViT decoder, resulting in T ∈ RN×D, where N is the number of context tokens predefined by the text encoder. The text embedding T is then concatenated with the latent tokens ZI and a set of mask tokens M ∈ R

tor of f, resulting in patches P ∈ R

H f ×Wf ×D. The combined sequence is passed through the decoder Dec, yielding the reconstructed imageˆI. Formally, with ⊕ denoting concatenation, the tokenization and de-tokenization can be represented as:

ZI = Enc(P ⊕ L), ˆI = Dec(ZI ⊕ T ⊕ M).

Flow matching [49, 54] is a framework that learns a continuous transformation between a source distribution and a target distribution. The source distribution is not necessarily required to be Gaussian noise, though we use Gaussian noise as a concrete example below.

During training, given a sample X from the target distribution, a sampled time step t ∈ [0,1], and a noise sample N ∼ N(0,I) from the source distribution. An intermediate representation Xt is obtained by:

Xt = (1 − t) · X + t · N.

The flow matching model is trained to estimate the velocity field Vt, which describes the direction from the source

to the target distribution. Taking the derivative of Xt with respect to t, we have:

dXt dt

= N − X,

Vt =

where Vt indicates the direction from the source to the target distribution such that the induced flow accurately transports the source distribution to the target distribution.

Notably, while the source distribution is typically modeled as Gaussian noise in generative frameworks [24], the flow matching formulation generalizes to arbitrary source distributions, provided that the source and target distributions share the same shape. In FlowTok, we directly define a unified latent space for image and text modalities, treating them as both source and target distributions. This design enables seamless generation across different modalities.

### 4. Method

In this section, we focus on text-to-image generation as the primary task to illustrate FlowTok. We first detail how images and text are projected into a unified, compact latent space as 1D tokens while preserving semantic information (Sec. 4.1). Next, we introduce FlowTok as a general framework for seamless flow between text and image tokens and discuss its extension to image-to-text generation under the same formulation (Sec. 4.2).

##### 4.1. Unifying Latent Space of Image and Text

The structural discrepancy between text and images presents a significant challenge in unifying them within the same latent space for flow matching. Text is inherently semantic, encoded as a 1D latent sequence with high-dimensional channels to preserve meaning, whereas images contain spatially redundant information and are typically represented as 2D feature maps with lower channel dimensions to retain spatial priors. To bridge this gap, we propose encoding images into compact 1D tokens by leveraging recent advancements in image tokenization. This formulation helps preserve the 1D structure of text embeddings, requiring only their projection into a more compressed set of tokens while ensuring that semantic information is retained. Below, we detail how both images and text are encoded.

Encoding Images into Compact Tokens. We build upon the core idea of TA-TiTok [39] with several enhancements to improve our image tokenizer. Specifically, we replace the original learnable 1D positional embedding with RoPE [78] to enhance TA-TiTok performance. Additionally, we substitute the MLP blocks in the Vision Transformer (ViT) [21] with SwiGLU FFN [72], which helps learn a more effective latent space [15, 90]. To align with the number of context tokens N of the text encoder, we set the number of latent tokens K in TA-TiTok accordingly (K = N = 77 for

|Tinit|
|---|

|ZI|
|---|

|ZT|
|---|

###### x L

[Figure 19]

DiTBlock

DiTBlock

DiTBlock

CLIP Text Encoder

Image VAE

“dog meme king” Text Projector

Decoder

- N x C N x D N x D
- N x D N x D

Text-to-Image Generation

|CLIP Text Embedding legendText Token Image Token<br><br>|
|---|

Image-to-Text Generation

|ZI|
|---|

|ZT|
|---|

x L

[Figure 20]

DiTBlock

DiTBlock

DiTBlock

Image VAE Encoder Text Decoder

CLIP Text “dog meme king”

Tokenizer

Figure 4. Overview of FlowTok. FlowTok is a minimal framework that facilitates seamless flow between 1D text tokens and image tokens for both text-to-image and image-to-text generation. Top: For text-to-image generation, the input text is encoded by the CLIP text encoder into Tinit ∈ RN×C, projected into a low-dimensional latent space as text tokens ZT ∈ RN×D, then transformed into image tokens ZI ∈ RN×D of the same shape through flow matching and decoded by a 1D Image VAE Decoder to generate the final image. Bottom: For image-to-text generation, an input image is encoded by a 1D Image VAE Encoder into ZI, mapped to ZT through flow matching and decoded into text via a text decoder. Unlike conventional approaches that rely on 2D noise and image latents (e.g., 32 × 32 × 4 for 256-resolution images) with text as conditions, our direct 1D transformation (i.e., 77 × 16) achieves a 3.3× compression rate, significantly reducing memory costs, accelerating training, and enabling faster inference.

CLIP [65]). As a result, the encoder of TA-TiTok encodes each image into a compact 1D token sequence ZI ∈ RK×D. Transforming Texts into Compact Tokens. We use a pretrained text encoder [65] to extract the initial text embedding Tinit ∈ RN×C, where C denotes the number of channels. Notably, C is typically much larger than the image latent size D, as it carries richer semantic information. Since our goal is to directly flow the text embedding into the image latent space, we need to ensure both embeddings have the same shape. While we already align K with N through careful tuning of the image tokenizer, ensuring the image is encoded to match the length of the text tokens. The remaining challenge lies in aligning the number of channels (i.e., C and D), which we resolve using a text projector. Since only the number of channels in Tinit needs adjustment while preserving its 1D shape, we employ a few simple Transformer blocks as the projector. To introduce variability in image generation from the same text, we model the projected text latents ZT ∈ RN×D as a Gaussian distribution by applying KL divergence regularization Lkld.

A crucial aspect of text-to-image generation is ensuring that the generated image accurately reflects the input text description. Since reducing the channel dimensions of text embeddings via a learnable projector may result in semantic information loss, we introduce an auxiliary text alignment loss Lalign to preserve semantic consistency. Specifically, we employ a lightweight MLP to project Tinit into a new space TP ∈ RN×D for alignment. We then flatten and normalize both TP and ZT along the channel dimension

and compute a contrastive loss between them, inspired by CLIP [65]. Concretely, we calculate the scaled pairwise cosine similarities using a learnable temperature parameter τ, followed by a symmetric cross-entropy loss:

logitsTZ = exp(τ) × (TP × ZTT), logitsZT = exp(τ) × (ZT × TTP), Lalign = (CE(logitsTZ,labels) + CE(logitsZT,labels))/2,

where T denotes the transpose operation, CE represents the cross-entropy loss, and labels are assigned based on their batch indices, ensuring that each text token is explicitly trained to align with its corresponding CLIP text embedding within the same batch. We also explore alternative approaches to preserving semantic information, such as aligning with the average-pooled text embedding or using a cosine similarity loss with a margin. However, we find that the CLIP-style loss achieves the best performance. More details are provided in Sec. 5.3.

Through the aforementioned designs, FlowTok efficiently tokenizes text into the same low-dimensional latent space while preserving semantic information. This alignment with the tokenized image latent space establishes a foundation for direct flow between compressed text tokens ZT and image tokens ZI. Notably, when using CLIP as the text encoder, FlowTok effectively reduces the latent size compared to traditional 2D flow matching methods. At an image resolution of 256, the latent size is reduced from 32×32×4 to 77×16, achieving a 3.3× compression. This reduction significantly lowers memory requirements and accelerates training, en-

|model|depth width mlp heads #params<br><br>|
|---|---|
|FlowTok-B FlowTok-XL FlowTok-H<br><br>|12 768 3072 12 153M 28 1152 4608 16 698M 36 1280 5120 20 1.1B|

- Table 1. Architecture Configuration of FlowTok. Following prior work, we scale up DiT blocks across three configurations.

hancing the framework’s efficiency and scalability.

##### 4.2. FlowTok: A General Framework for Seamless Flow Across Text and Image Tokens

Text-to-Image Generation. As shown in Fig. 4 (top), with both image and text mapped into the same latent space, FlowTok leverages vanilla flow matching [49] by stacking DiT blocks [61]. Notably, source modality (i.e., text) is directly treated as the source distribution for flow matching, removing the need for concatenation or cross-attention within the DiT blocks. This design choice further simplifies the overall framework and streamlines text-to-image generation. Combined with the compact 1D tokens introduced in Sec. 4.1, FlowTok achieves high memory efficiency, supporting a batch size of 8K on 8 A100 GPUs. Additionally, it enables fast sampling, running over 10× faster than modern textto-image diffusion models [70], significantly lowering the computational barrier for training large-scale text-to-image generative models.

Image-to-Text Generation. FlowTok can also be seamlessly extended to image-to-text generation using compact 1D image and text tokens under the same formulation, as shown in Fig. 4 (bottom). Specifically, the image tokens ZI flow to text tokens ZT, where a trained text decoder takes ZT as input and outputs tokenizer indices, which can then be decoded back into the corresponding caption.

### 5. Experimental Results

In this section, we first provide the implementation details of FlowTok (Sec. 5.1), followed by the main results on text-toimage and image-to-text generation (Sec. 5.2). Finally, we present the ablation studies to better understand the design choices of FlowTok in text-to-image generation (Sec. 5.3).

##### 5.1. Implementation Details

Image Tokenizer. We build our image tokenizer upon the official TA-TiTok [39] codebase with minimal modifications. The encoder uses ViT-B [21], while the decoder uses ViT-L, both operating with a patch size of f = 16. To align with the output sequence length of CLIP’s text encoder, we set the number of 1D latent tokens K to 77 and the token dimension to 16. Additionally, we enhance the tokenizer with RoPE [78] and SwiGLU FFN [72]. Notably, our enhanced tokenizer achieves a FID of 1.02 in a zero-shot evaluation

on the ImageNet validation set, matching the performance of the original TA-TiTok with 128 tokens.

Text Projector. We train a text projector for text-to-image generation, which transforms CLIP text embeddings into a latent representation ZT of shape 77 × 16, aligning with the image latent space ZI encoded by our image tokenizer. The text projector consists of six Transformer [83] blocks, each comprising a multi-head self-attention mechanism and a multi-layer perceptron (MLP), both enhanced with skip connections [33] to ensure stable training.

Text Decoder. We train a text decoder for image-to-text generation, composed of six Transformer [83] blocks, similar to the text projector. The decoder takes the text latent representation ZT as input and outputs the corresponding CLIP text tokenizer indices, which can be further converted into text using the CLIP text tokenizer.

FlowTok. We adopt DiT [61] blocks as the fundamental building units of our FlowTok to model token interactions. Specifically, we follow the DiT architecture to implement FlowTok-B for efficient ablation studies and FlowTok-XL for enhanced performance. To further push the performance, we scale up the depth, width, and number of attention heads, constructing FlowTok-H with 1.1B parameters. The detailed model configurations are provided in Tab. 1.

Dataset. We employ open-source datasets [39] to facilitate the reproducibility of FlowTok’s simple framework. Specifically, our image tokenizer is trained on DataComp-1B [27], and our text tokenizer is trained on COCO [48]. For text-toimage generation, inspired by recent works [13, 14, 19, 95], we adopt a two-stage training strategy: pre-training and finetuning. The pre-training stage leverages a combination of DataComp-1B [27], CC12M [12], and LAION-aesthetic [1], while the fine-tuning stage incorporates additional highquality datasets, including LAION-art [2], LAION-pop [3], JourneyDB [79], and DALLE3-1M [23]. For image-to-text generation, we follow the Karpathy split [38] of COCO [17] to divide the training and validation sets. Detailed dataset information is provided in the Appendix.

Training. The training objectives of FlowTok primarily focus on predicting velocity in flow matching, denoted as Lfm. For text-to-image generation, we introduce two additional losses: KL-divergence loss (Lkld) to enforce a Gaussian distribution on text tokens and text alignment loss (Lalign) to preserve semantic information as discussed in Sec. 4.1. Formally, the overall training objective is:

L = Lfm + γ1 ∗ Lkld + γ2 ∗ Lalign,

where γ1 and γ2 control the weighting of losses. By default, we set γ1 to 1×10−4 and γ2 to 1 for text-to-image generation, while both are set to 0 for image-to-text generation.

Evaluation. We follow standard evaluation practices to report relevant metrics for both text-to-image and imageto-text generation. Specifically, for text-to-image genera-

method params open-data T↓ I↑ COCO FID-30K↓ MJHQ-30K FID↓ text as conditions GLIDE [58] 5.0B ✗ - - 12.24 Dalle·2 [66] 6.5B ✗ - - 10.39 LlamaGen [80] 775M ✗ - - - 25.59 PixArt-α [14] 630M ✗ 94.1 7.9 7.32 9.85 SDXL [62] 2.6B ✗ - - - 8.76 LDM [70] 1.4B ✓ - - 12.63 -

- Stable-Diffusion-1.5 [70] 860M ✓ 781.2 - 9.62 -

- Stable-Diffusion-2.1 [70] 860M ✓ 1041.6 - 13.45 26.96 Show-o [87] 1.3B ✓ - 1.0 9.24 14.99

text as source distributions CrossFlow [51] 950M ✗ 78.8 1.1 9.63 FlowTok-XL 698M ✓ 20.4 22.7 10.06 7.68 FlowTok-H 1.1B ✓ 26.1 18.2 9.67 7.15

- Table 2. Zero-Shot Text-to-Image Generation Results on COCO and MJHQ-30K. We compare FlowTok with state-of-the-art methods, categorized into two approaches: (1) text as conditions, where text tokens are used as conditions to guide the generation process, and (2) text as source distributions, where the model directly learns the alignment between text and image distributions. “open-data”: Models are trained exclusively with publicly available datasets. “T”: Model training cost, measured in 8 A100 days using float16 precision. “I”: Model inference throughput, measured at 256px resolution in samples per second on a single A100 with batch size 64 using float16 precision.

method B@4↑ M↑ R↑ C↑ S↑ direct flow from image to text distributions CrossFlow [51] 36.4 27.8 57.1 116.2 20.4 FlowTok-XL 37.1 27.8 57.6 117.0 20.5 other methods MNIC [29] 30.9 27.5 55.6 108.1 21.0 MIR [42] 32.5 27.2 - 109.5 20.6 NAIC-CMAL [31] 35.3 27.3 56.9 115.5 20.8 SATIC [97] 32.9 27.0 - 111.0 20.5 SCD-Net [56] 37.3 28.1 58.0 118.0 21.6

- Table 3. Image-to-Text Generation Results on COCO. FlowTok achieves performance comparable to state-of-the-art methods on image-to-text generation, evaluated on the COCO Karpathy split. For a fair comparison, we restrict our evaluation to nonautoregressive methods trained without CIDEr optimization.

into two groups: text as conditions, where text serves as a guiding signal for image generation, and text as source distributions, where text is directly modeled as a distribution in the generative process. As observed, FlowTok achieves comparable performance to prior methods in both categories on COCO FID-30K. Specifically, compared to CrossFlow [51], which also uses text as the source distribution, FlowTok-H attains a FID-30K of 9.67—roughly on par with CrossFlow. When further evaluating FlowTok on MJHQ-30K to assess the aesthetic quality of generated images, we find that, despite being trained solely on publicly available datasets without access to high-quality proprietary data, FlowTok-XL already surpasses other state-of-the-art models, demonstrating its ability to generate diverse, high-quality images. Furthermore, FlowTok-H further improves the FID score to 7.15, underscoring its superior image generation capabilities.

Beyond performance, FlowTok requires significantly fewer training resources compared to existing state-of-theart models. Specifically, FlowTok-XL completes training in just 20.4 8-A100 days, while FlowTok-H increases the budget slightly to 26.1 8-A100 days. In contrast, the most efficient text-as-condition model, PixArt-α [14], still demands 94.1 8-A100 days. Compared to CrossFlow [51], which also treats text as source distributions and requires 78.8 8-A100 days, FlowTok is much more efficient.

tion, we report FID-30K on the COCO [48], and FID on MJHQ-30K [44]. For image-to-text generation, we report BLEU-4 [60], METEOR [8], ROUGE [47], CIDEr [84], and SPICE [6] on the COCO Karpathy Split [38]. To incorporate classifier-free guidance (CFG) [34] within FlowTok, we follow CrossFlow[51] and utilize a CFG indicator. Unless otherwise stated, we find that using only 20 steps for sampling is sufficient due to the small 1D latent shape of FlowTok. This significantly speeds up the inference process, enabling faster generation without compromising performance.

Additionally, FlowTok demonstrates significantly faster inference speeds. At 256px resolution, FlowTok-XL generates 22.7 images per second, while FlowTok-H achieves 18.2 images per second. In contrast, PixArt-α runs at 7.9 images per second, and Show-o at just 1.0 images per second. More notably, within the text as source distributions category, FlowTok achieves a 20× speedup in sampling time

##### 5.2. Main Results

Text-to-Image Generation. We report zero-shot textto-image generation results on COCO [48] and MJHQ30K [44] in Tab. 2. The compared methods are categorized

|target<br><br>|COCO FID-30K↓|
|---|---|
|Ave Pool MLP<br><br>|36.02 29.14<br><br>|

(a) Text Alignment Target

|loss type|COCO FID-30K↓<br><br>|
|---|---|
|Cosine Contrastive|31.80 29.14<br><br>|

(b) Text Alignment Loss Function

|γ2|COCO FID-30K↓|
|---|---|
|1.0 2.0<br><br>|29.14 30.59|

(c) Text Alignment Loss Weight

- Table 4. Ablation Studies on Text Alignment Loss. We conduct comprehensive ablation studies on three key aspects of the text alignment loss Lalign: the alignment target (Tab. 4a), the choice of loss function (Tab. 4b), and the loss weight γ2 (Tab. 4c), aiming to identify the most effective strategy for preserving semantic information within FlowTok during text-to-image generation. For efficient verification, we report FID-30K on COCO using FlowTok-B, without applying the CFG indicator.

compared to CrossFlow, which runs at only 1.1 images per second. This efficiency stems from FlowTok ’s streamlined framework and its effective use of 1D tokens, significantly reducing computational overhead.

Image-to-Text Generation. We evaluate image-to-text generation on COCO [48] using the Karpathy split [38], with results summarized in Tab. 3. To ensure a fair comparison, we categorize methods into two groups: direct flow from image to text distributions, which represents a new paradigm leveraging flow matching for direct image-to-text transformation, and other methods, considering only those trained without CIDEr optimization. Within the direct flow category, FlowTok-XL consistently outperforms its counterpart, CrossFlow [51], across most metrics. Specifically, FlowTokXL achieves a BLEU-4 (B@4) score of 37.1, surpassing CrossFlow by 0.7, and a CIDEr score of 117.0, exceeding CrossFlow by 0.8. Moreover, compared to state-of-the-art methods from other paradigms, FlowTok-XL demonstrates competitive performance, highlighting direct flow matching as a promising approach for image-to-text generation. Notably, FlowTok performs image-to-text generation under the same formulation using compact 1D tokens, theoretically requiring fewer training resources and enabling faster sampling compared to paradigms that operate on 2D latents, as adopted by CrossFlow. However, a direct quantitative comparison is not possible, as CrossFlow has not released the corresponding checkpoint for evaluation.

- 5.3. Ablation Studies

learn the alignment target (row 2), as inspired by prior works [30, 59, 94]. We attribute this performance gap to the fact that adjacent channels in the CLIP text embedding are not necessarily correlated, and simple average pooling discards too much semantic information. In contrast, a learnable MLP mitigates this information loss, making it a more effective choice for defining the text alignment target.

Text Alignment Loss Function. Next, we examine the text alignment loss function in Tab. 4b. Besides the contrastive loss adopted from [65], we explore using a cosine similarity loss, similar to [89]. Specifically, we compute the cosine similarity between the text tokens and the alignment target, applying a penalty to pairs with similarity below a threshold. Our experiments show that while both loss functions are effective, the contrastive loss achieves better performance.

Text Alignment Loss Weight. Finally, we investigate the impact of the text alignment loss weight, γ2 in Tab. 4c. Our results indicate that setting γ2 to 1.0, equal to the weight of the flow matching loss, is sufficient to preserve semantic information while maintaining high-quality image generation. Increasing γ2 further can cause the text alignment loss to dominate the overall objective during early training stages, potentially hindering final performance.

### 6. Conclusion

In this paper, we introduce FlowTok, a minimal yet powerful framework that enables seamless direct flow between 1D text and image tokens. Through carefully designed key modules and loss functions, FlowTok projects both modalities into a unified 1D latent space while preserving semantic information, enabling both text-to-image and image-to-text generation under the same formulation. This design makes FlowTok highly memory-efficient, supporting an 8K batch size on just 8 A100 GPUs during training. Additionally, its simplicity accelerates convergence—within approximately 20 days on 8 A100 GPUs, FlowTok achieves performance comparable to state-of-the-art models that require significantly longer training times. The streamlined design also enables over 10× faster sampling than modern text-to-image generative models. By releasing our code, we aim to further advance research in text-image cross-modal generation.

We conduct ablation studies on text-to-image generation using FlowTok-B and evaluate them on COCO for efficiency. Our ablations focus on the design of the text alignment loss, as it plays a critical role in preserving semantic information. Specifically, we investigate three key aspects: the text alignment target (Tab. 4a), the choice of loss function (Tab. 4b), and the loss weight (Tab. 4c). Details are provided below.

Text Alignment Target. We first investigate the choice of alignment target for the projected text tokens ZT in Tab. 4a. A straightforward baseline is to directly apply average pooling (row 1) to the original CLIP text embedding Tinit along the channel dimension, reducing the dimensionality from 768 to 16 to match ZT. However, this approach performs significantly worse compared to using a simple MLP to

### References

- [1] LAION2B-en-aesthetic. https://huggingface.co/ datasets/laion/laion2B-en-aesthetic, . 6
- [2] LAION-art. https://huggingface.co/datasets/ laion/laion-art, . 6
- [3] LAION-pop. https : / / huggingface . co / datasets/laion/laion-pop, . 6
- [4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022. 2
- [5] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022. 2, 3
- [6] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In ECCV, 2016. 7
- [7] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng YAN. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. In ICLR, 2025. 3, 4
- [8] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005. 7
- [9] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 2
- [10] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science, 2(3):8, 2023. 3
- [11] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, José Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Textto-image generation via masked generative transformers. In ICML, 2023. 3
- [12] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR,

2021. 6

- [13] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-Σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In ECCV, 2024. 6
- [14] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR,

2024. 2, 3, 4, 6, 7

- [15] Jieneng Chen, Qihang Yu, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Vitamin: Designing scalable vision mod-

els in the vision-language era. In CVPR, 2024. 4

- [16] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, et al. Goku: Flow based video generative foundation models. arXiv preprint arXiv:2502.04896, 2025. 3, 4
- [17] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 6
- [18] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In ICML, 2021. 2
- [19] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, Matthew Yu, Abhishek Kadian, Filip Radenovic, Dhruv Mahajan, Kunpeng Li, Yue Zhao, Vladan Petrovic, Mitesh Kumar Singh, Simran Motwani, Yi Wen, Yiwen Song, Roshan Sumbaly, Vignesh Ramanathan, Zijian He, Peter Vajda, and Devi Parikh. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023. 6
- [20] Xueqing Deng, Qihang Yu, Ali Athar, Chenglin Yang, Linjie Yang, Xiaojie Jin, Xiaohui Shen, and Liang-Chieh Chen. Coconut-pancap: Joint panoptic segmentation and grounded captions for fine-grained understanding and generation. arXiv preprint arXiv:2502.02589, 2025. 3
- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 4, 6
- [22] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 2
- [23] Ben Egan, Alex Redden, XWAVE, and SilentAntagonist. Dalle3 1 Million+ High Quality Captions. https://huggingface.co/datasets/ ProGamerGov/synthetic-dataset-1m-dalle3high-quality-captions, 2024. 6
- [24] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML,

2024. 2, 3, 4

- [25] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024. 3, 4
- [26] Johannes S Fischer, Ming Gui, Pingchuan Ma, Nick Stracke, Stefan A Baumann, and Björn Ommer. Boosting latent diffusion with flow matching. arXiv preprint arXiv:2312.07360,

2023. 3

- [27] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. NeurIPS, 2023. 6
- [28] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scene-based text-to-image generation with human priors. In ECCV, 2022. 3
- [29] Junlong Gao, Xi Meng, Shiqi Wang, Xia Li, Shanshe Wang, Siwei Ma, and Wen Gao. Masked non-autoregressive image captioning. arXiv preprint arXiv:1906.00717, 2019. 7
- [30] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 2020. 8
- [31] Longteng Guo, Jing Liu, Xinxin Zhu, Xingjian He, Jie Jiang, and Hanqing Lu. Non-autoregressive image captioning with counterfactuals-critical multi-agent learning. arXiv preprint arXiv:2005.04690, 2020. 7
- [32] Ju He, Qihang Yu, Inkyu Shin, Xueqing Deng, Alan Yuille, Xiaohui Shen, and Liang-Chieh Chen. A simple video segmenter by tracking objects along axial trajectories. Transactions on Machine Learning Research, 2024. 2
- [33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 6
- [34] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 7
- [35] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2, 3
- [36] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In ICML, 2021. 2
- [37] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 2
- [38] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In CVPR, 2015. 6, 7, 8
- [39] Dongwon Kim, Ju He, Qihang Yu, Chenglin Yang, Xiaohui Shen, Suha Kwak, and Liang-Chieh Chen. Democratizing text-to-image masked generative models with compact text-aware one-dimensional tokens. arXiv preprint arXiv:2501.07730, 2025. 2, 3, 4, 6
- [40] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 2
- [41] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2
- [42] Jason Lee, Elman Mansimov, and Kyunghyun Cho. Deterministic non-autoregressive neural sequence modeling by iterative refinement. arXiv preprint arXiv:1802.06901, 2018. 7
- [43] Sangyun Lee, Beomsu Kim, and Jong Chul Ye. Minimizing trajectory curvature of ode-based generative models. In ICML,

2023. 3

- [44] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 7
- [45] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In ICML, 2022. 2
- [46] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019. 2
- [47] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 2004. 7
- [48] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 6, 7, 8
- [49] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 2, 3, 4, 6
- [50] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima Anandkumar. I2sb: Image-to-image schrödinger bridge. arXiv preprint arXiv:2302.05872, 2023. 2, 3
- [51] Qihao Liu, Xi Yin, Alan Yuille, Andrew Brown, and Mannat Singh. Flowing from words to pixels: A framework for crossmodality evolution. arXiv preprint arXiv:2412.15213, 2024. 2, 3, 7, 8
- [52] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. NeurIPS, 2024. 2, 3
- [53] Qihao Liu, Ju He, Qihang Yu, Liang-Chieh Chen, and Alan Yuille. Revision: High-quality, low-cost video generation with explicit 3d physics modeling for complex motion and interaction. arXiv preprint arXiv:2504.21855, 2025. 3
- [54] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 2, 3, 4
- [55] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. NeurIPS, 2019. 2
- [56] Jianjie Luo, Yehao Li, Yingwei Pan, Ting Yao, Jianlin Feng, Hongyang Chao, and Tao Mei. Semantic-conditional diffusion networks for image captioning. In CVPR, 2023. 7
- [57] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024. 3
- [58] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 7
- [59] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2:

- Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 8
- [60] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 7
- [61] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2, 3, 6
- [62] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3, 4, 7
- [63] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. 3
- [64] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky TQ Chen. Multisample flow matching: Straightening flows with minibatch couplings. arXiv preprint arXiv:2304.14772, 2023. 3
- [65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 4, 5, 8
- [66] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2): 3, 2022. 7
- [67] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Flowar: Scale-wise autoregressive image generation meets flow matching. arXiv preprint arXiv:2412.15205, 2024. 3
- [68] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025. 3

- [69] Sucheng Ren, Qihang Yu, Ju He, Alan Yuille, and LiangChieh Chen. Grouping first, attending smartly: Trainingfree acceleration for diffusion transformers. arXiv preprint arXiv:2505.14687, 2025. 3
- [70] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 6, 7
- [71] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022. 3
- [72] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 2, 4, 6
- [73] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger bridge matching. NeurIPS, 2023. 2, 3
- [74] Inkyu Shin, Chenglin Yang, and Liang-Chieh Chen. Deeply supervised flow-based generative models. arXiv preprint arXiv:2503.14494, 2025. 3
- [75] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [76] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 2019. 2
- [77] stabilityai, 2023. 3
- [78] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864,

2021. 2, 4, 6

- [79] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. NeurIPS, 2023. 6
- [80] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 7
- [81] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482, 2023. 3
- [82] Alexander Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio. Simulation-free schr\" odinger bridges via score and flow matching. arXiv preprint arXiv:2307.03672, 2023. 2, 3
- [83] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. 6
- [84] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015. 7

- [85] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021. 2
- [86] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit: Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024. 3
- [87] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3, 7
- [88] Chenglin Yang, Celong Liu, Xueqing Deng, Dongwon Kim, Xing Mei, Xiaohui Shen, and Liang-Chieh Chen. 1.58-bit flux. arXiv preprint arXiv:2412.18653, 2024. 3
- [89] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 8
- [90] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025. 4
- [91] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. TMLR,

2022. 3

- [92] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and LiangChieh Chen. Randomized autoregressive visual generation. arXiv preprint arXiv:2411.00776, 2024. 3
- [93] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. NeurIPS, 2024. 2, 4
- [94] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025. 3, 8
- [95] Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogview3: Finer and faster text-to-image generation via relay diffusion. arXiv preprint arXiv: 2403.05121, 2024. 6
- [96] Linqi Zhou, Aaron Lou, Samar Khanna, and Stefano Ermon. Denoising diffusion bridge models. arXiv preprint arXiv:2309.16948, 2023. 2, 3
- [97] Yuanen Zhou, Yong Zhang, Zhenzhen Hu, and Meng Wang. Semi-autoregressive transformer for image captioning. In ICCV, 2021. 7

