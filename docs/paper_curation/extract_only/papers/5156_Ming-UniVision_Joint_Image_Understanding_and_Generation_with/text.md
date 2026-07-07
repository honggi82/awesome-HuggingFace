# arXiv:2510.06590v1[cs.CV]8Oct2025

## Ming-UniVision: Joint Image Understanding and Generation with a Unified Continuous Tokenizer

Inclusion AI, Ant Group∗

∗See Contributions section (Sec. 6) for full author list.

Visual tokenization remains a core challenge in unifying visual understanding and generation within the autoregressive paradigm. Existing methods typically employ tokenizers in discrete latent spaces to align with the tokens from large language models, where the quantization errors can limit semantic expressiveness and degrade the capability of vision-language understanding. To address this, we introduce MingTok, a new family of visual tokenizers with a continuous latent space, for unified autoregressive generation and understanding. While understanding tasks favor discriminative highdimensional features, generation tasks prefer compact low-level codes. Thus, to reconcile these competing demands, MingTok adopts a three-stage sequential architecture involving low-level encoding, semantic expansion, and visual reconstruction. Built on top of it, Ming-UniVision eliminates the need for task-specific visual representations, and unifies diverse vision-language tasks under a single autoregrsssive prediction paradigm. By formulating both understanding and generation as next-token prediction in a shared continuous space, it seamlessly supports multi-round, in-context tasks such as iterative understanding, generation and editing. Empirically, we find that using a unified continuous visual representation reconciles the competing requirements on the tokenizers by the understanding and generation tasks, thereby leading to state-of-the-art level performance across both domains. We hope our findings will facilitate unified visual tokenization in the continuous domain. Inference code and model weights are released to benefit community.

Date: Oct 7, 2025 : https://github.com/inclusionAI/Ming-UniVision : https://huggingface.co/inclusionAI

[Figure 1]

: https://www.modelscope.cn/organization/inclusionAI

[Figure 2]

### 1 Introduction

Autoregression has emerged as a powerful and general technique to model diverse data modalities, achieving remarkable success in language (Radford et al., 2019; Dubey et al., 2024), audio (Chu et al., 2023; Ding et al., 2025), and omni-modal sequences (Hurst et al., 2024; AI et al., 2025a). By formulating both visual understanding and image generation as sequential prediction problems, autoregressive models have achieved competitive results in vision-language understanding (Bai et al., 2025; Guo et al., 2025) and generation (Li et al., 2024; Tian et al., 2024). This has motivated recent efforts to unify vision-language understanding and generation within a single autoregressive framework (AI et al., 2025b; Pan et al., 2025; Deng et al., 2025), to fully leverage in-context learning, compositional reasoning, and task generalization abilities inherent in large language models.

In terms of input and output, visual understanding and generation are inverse tasks in some sense, where one where one maps pixels to semantic concepts and the other synthesizes pixels from textual

Image Generation

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

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Image Edit

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

Add some hair Make him serious Make her younger Remove the umbrella Make her hair brown Make him laugh

Multi-round in-context editing

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Furtherenhanceimageclarity

Enhanceimageclarity

Enhanceimageclarity

Colorizetheimage

[Figure 64]

50 47 43

VAE

Generation

Pixel Decoder

>3.5x faster

Understanding

Semantic Feature 1024d

Image Latent 16d

GenEvalScore

MingTok

Semantic Decoder

Uniﬁed MLLM

Uniﬁed MLLM

30

Image Latent 32d

MingTok-L/14-224 (stage 2) MingTok-L/14-224 (stage 1)

Generation

Low-level Encoder

>6x faster

22

Semantic Feature 1024d

20 17

SD-VAE-f16 (stage 2) SD-VAE-f16 (stage 1)

Semantic Encoder

Understanding

[Figure 65]

[Figure 66]

9

SigLIP / QwenViT …

50100 200 300 400 500

600 700

Training steps

(a) Existing uniﬁed models w/ continuous latents (b) Uniﬁed model with MingTok (c) Convergence speedup with MingTok

- Figure 1 Conceptual comparison and qualitative examples of MingTok. (a) Existing models using continuous latent spaces for unified visual understanding and generation uses two sets of representations for visual contents. (b) MingTok employes a unified tokenizer for generating semantic and low-level image representatinos. (c) Compared with SD-VAE (Rombach et al., 2022), MingTok achieves over 3.5 times acceleration for text-to-image generation.

descriptions. Most of current multi-modal models handle them as two separate tasks with totally distinct approaches (Wu et al., 2025a; Pan et al., 2025; Deng et al., 2025), as they typically use distinct representation spaces for understanding and generation. This discrepancy stems from the fundamentally different and competing requirements for their underlying representations, where understanding tasks favor high-dimentional semantic features (Radford et al., 2021; Zhai et al.,

- 2023), while generation demands compact and low-dimentional structured latent codes that preserve fine-grained visual details (Esser et al., 2021; Rombach et al., 2022). As a result, most existing frameworks resort to asymmetric designs (Deng et al., 2025; Fan et al., 2025; Chen et al., 2025d), using distinct tokenization schemes for the two tasks, which introduces optimization difficulty and architectural complexity to the unified system.

Recent efforts have sought to bridge this gap through shared visual tokenizers (Qu et al., 2025; Wu et al., 2024a; Jiao et al., 2025). However, these approaches often rely on discrete latent representations, which introduces quantization error and limits representation capacity. This degradation not only constrains the fidelity of generated images, but also impairs semantic expressiveness, leading to suboptimal performance on understanding tasks (Ma et al., 2025).

To address these limitations, we present MingTok, a visual tokenizer with a continuous latent space that enables unified autoregressive vision-language understanding and generation. The core of MingTok is a three-stage sequential architecture designed to reconcile the competing representational demands of understanding and generation: (i) A low-level encoder maps input images into a compact, continuous latent representation optimized for efficient autoregressive generation; (ii) A semantic decoder progressively expands this compact latent sequence into high-dimensional semantic features through autoregressive refinement, producing expressive high-dimensional representations suitable for vision-language reasoning; (iii) A pixel decoder reconstructs the original image from the semantic features, ensuring high-fidelity image reconstruction. The entire model is end-to-end optimized with a masked image modeling objective that imposes supervision over both intermediate representations and reconstructed pixels, thereby simultaneously enhancing semantic richness and shaping the compact latent space to be more conducive to autoregressive generation.

On top of MingTok, we build a unified multi-modal model, Ming-UniVision, that adopts a single autoregressive architecture for joint text and image modeling. Both understanding and generation operate on a shared high-level semantic space as input, while generating compact continuous latents in a token-by-token manner, resulting in significantly reduced architectural complexity compared to existing dual-encoder designs, as illustrated in Fig. 1. This unified paradigm enables strong performance across both vision-language understanding and text-to-image generation, demonstrating that a single autoregressive framework can support diverse capabilities through universal visual representations.

Crucially, by processing both modalities within a shared next-token prediction paradigm and maintaining unified representations within each modality, our model establishes a coherent interface for multimodal sequence modeling. This enables seamless integration: any prefix sequence, whether composed of text tokens, visual latents, or mixed modalities, can be directly consumed to condition downstream generation or reasoning in either modality. As a result, the same architecture supports not only text-to-image and image-to-text tasks, but also unlocks complex multi-round interactions, including iterative super-resolution, region-aware editing, and sequential refinement, e.g., upsample → colorize (Fig. 7) and segment → edit (Fig. 8), all within a single and coherent framework. These results suggest that unified modeling through persistent latent representations opens up new possibilities for interactive, human-in-the-loop vision systems, where generation, editing, and

H/32 x W/32+1

Latent Semantics

Full Attention Causal Attention

Full Attention

[Figure 67]

[Figure 68]

W

Uni-directional

Uni-directional

Bi-directional

Bi-directional

Bi-directional

Bi-directional

Projectionout

Projectionin

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

SABlock

SABlock

SABlock

SABlock

SABlock

SABlock

xN

xM

xL

…

…

…

H

Semantic Decoder

Low-level Encoder

Pixel Decoder

MLP MLP

[Figure 69]

[Figure 70]

Image Patches

Mask tokens

Masked Feature Prediction

Masked Feature Prediction

Pretrained Model

Pretrained Model

Latent tokens (d=32)

Cls token

Semantic tokens

Pretrained features

- Figure 2 The model architecture and the training objectives of MingTok. MingTok performs image compression, semantic decoding and image reconstruction sequentially through low-level encoder, semantic decoder, and pixel decoder. During training, both the image latent and the semantic features are supervised by pre-trained visual encoders with masked feature prediction, while the pixel decoder is trained by masked and unmasked image reconstruction.

understanding are no longer isolated pipelines, but interwoven steps in a continuous visual dialogue. We believe this shift toward dynamic, context-aware multimodal interaction will enable more flexible, intuitive, and cognitively aligned models in the future.

Our contributions are as follows:

- • We propose MingTok, a continuous visual tokenizer that unifies generation and understanding through a three-stage architecture, eliminating quantization error while supporting both compact latents and rich semantics.
- • Built on MingTok, we introduce Ming-UniVision, a unified autoregressive framework where diverse vision-language tasks are cast as next-token prediction, enabling seamless integration of perception, generation, and editing without task-specific representations.
- • We demonstrate efficient multi-round in-context editing, enabled by unified latent representations that eliminate the need for repeated encoding through distinct tokenizers. By drastically reducing the visual token count—requiring up to 66% fewer input tokens compared to prior unified architectures—Ming-UniVision enables faster iteration and lower memory overhead in interactive generation workflows.

### 2 Unified Visual Tokenization without Vector Quantization

##### 2.1 Tokenizer Model Architecture

A unified visual tokenizer must produce compact and high-dimensional patch representations, trained for both understanding and generation. As shown in Fig. 2, MingTok employs a three-stage sequential architecture composed of a low-level encoder, a semantic decoder, and a pixel decoder, all primarily implemented using vision transformer blocks with different masking strategies.

Low-level encoder transforms raw image pixels into compact latent embeddings for autoregressive generation. The compression ratio of the image latents is entirely determined by the patch embedding layer. The number of tokens are kept the same throughout the low-level encoder and the semantic decoder. A linear projection layer equipped with a channel-averaging shortcut serves as the output

###### Component Resolution Patch size Head dim Num heads Embed dim Depth Out dim Attention

Low-level Encoder 512 32 64 12 768 12 32 Full Semantic decoder 512 32 64 16 1024 24 1024 Causal Pixel decoder 512 16 64 16 1024 24 3 Full

- Table 1 The architectural hyperparameters of MingTok.

layer, compressing the latent channel dimension to a generation-friendly size, typically 16 or 32. Within the low-level encoder, full attention is employed to holistically model spatial dependencies and structural patterns in the input images.

Semantic decoder is connected after the low-level encoder, expanding the compact latent embeddings into rich semantic features for visual understanding. A linear projection layer with a channel-repeating shortcut serves as the input layer, expanding the latent channel dimension for semantic understanding. In order to support per-token autoregressive generation, the semantic decoder adopts causal attention.

Pixel decoder reconstructs the original images from the high-dimentional semantic features generated by the semantic decoder, serving as the final rendering stage in both reconstruction and autoregressive generation. To recover the fine visual details, we apply a pixel unshuffle layer (Shi et al., 2016) before the transformer blocks in the pixel decoder to increase the number of visual tokens and reduce the effective patch size. We find that this leads to significantly better texture fidelity and edge sharpness in the output images. Within the transformer layers, full self-attention is used to capture potential long-range dependencies across the entire feature map.

The architectural hyperparameters are shown in Table. 1.

##### 2.2 Training

The training of MingTok is guided by three key design principles: (i) the generative latent space should be both structured and compact to enable fast convergence and efficient autoregressive modeling; (ii) the semantic space is required to support strong scalability and generality in vision-language understanding tasks; (iii) the model should be able to achieve high-fidelity image reconstruction to ensure the quality of the generated visual contents. Therefore, we adopt a multi-task learning framework built upon the masked image modeling paradigm (Wei et al., 2022; Xie et al., 2022; Fang et al., 2024; He et al., 2022), with three complementary objectives targeting different components of the architecture.

Structured latent space regularization. As shown in prior work on variational autoencoders in visual generative models (Yao et al., 2025; Chen et al., 2025b; Skorokhodov et al., 2025; Chen et al., 2024a, 2025a), the structure and compactness of the latent space critically influence generation efficiency and downstream model convergence. To encourage a highly structured and compressed representation, we use a spatial compression ratio of 32. The low-level encoder and latent space are trained via masked feature prediction. The token sequences are randomly masked at the input side of the low-level encoder, and the masked tokens at the output side is used to predict the corresponding features from a pre-trained vision foundation model (e.g., DINOv2 (Oquab et al., 2023)) at the same spatial locations. This objective regularizes the latent space with rich semantic and structural priors required for autoregressive visual generation, and our empirical verification demonstrate the effectiveness of such a training objective.

Scalable semantic representation learning. To build a semantic decoder that supports strong

generalization and scaling in multimodal reasoning, we apply the same masked feature prediction paradigm—known for its superior scalability over plain feature distillation (Fang et al., 2023). Specifically, the compact latent sequence from the low-level encoder with partial token masked is further passed through the semantic decoder, which is autoregressively expanded to high-dimensional semantic feature sequence. The expanded semantic features at the masked locations are supervised by the feature representations of the visual backbones aligned with text semantics during pre-training (e.g., CLIP (Radford et al., 2021)). This ensures that the semantic space develops expressive, scalable visual representations suitable for complex vision-language tasks.

Pixel Reconstruction Objective. To enhance the robustness and fidelity of the pixel decoder, we train it under both masked and unmasked conditions, where the representations of both observed patches and masked patches are received by the pixel decoder. The decoder then learns to reconstruct the full image. This dual-setting supervision forces the decoder to recover fine-grained details even when some latents are missing or noisy—mimicking the autoregressive generation process where tokens are generated sequentially. As a result, the decoder becomes more robust and capable of producing high-quality reconstructions.

Since the pixel decoder reconstructs image from semantic features produced by the semantic decoder, it also requires the semantic decoder to preserve sufficient low-level details, which in turn enhances the fine-grained visual perception capability of the semantic features. In our early experiments, we found adding reconstruction objective to masked feature prediction can notably improve the downstream understanding performance when combined with multi-modal LLMs.

### 3 Unified Image Understanding and Generation

##### 3.1 Multimodal Model Architecture

The overall architecture of Ming-UniVision, a multi-modal model for image understanding, generation, and in-context multi-round editing is shown in Fig. 3. Our unified model achieves seamless integration of vision and language through two key unifications enabled by MingTok:

Unified Visual Input Representation. For both understanding and generation tasks, the language model consistently receives high-level semantic features produced by the semantic decoder. In image understanding, the representation is derived from real images. The input is first encoded into compact continuous latents via the low-level encoder, then passed through the semantic decoder to produce rich, text-aligned visual embeddings. Since the entire image is available upfront, all semantic tokens are computed in parallel. In autoregressive image generation, instead of encoding an observed image, the vision head of the language model generates compact latents one token at a time. Each generated latent token is immediately expanded into its corresponding semantic feature by the semantic decoder, which is then fed into the language model as the contextual input for next token prediction. This ensures a unified interface for multimodal interaction, regardless of whether the visual content is perceived or synthesized.

Unified Next-Token Prediction. On the output side, both modalities are generated autoregressively under a shared sequence modeling paradigm. Textual tokens are predicted using the standard language model head, preserving full compatibility with pre-trained LLMs. For visual content, a per-token vision head is attached to the language model to predict compact continuous latents one patch at a time, enabling seamless interleaving of text and image generation within the same autoregressive framework.

###### Understanding

###### Generation

###### Multi-Round Editing

Text Head Per-token Vision Head

Per-token Vision Head

Large Language Model

BOI

EOI

BOI EOI

BOI EOI

Text

Text

Semantic Decoder Low-level Encoder

Semantic Decoder

Semantic Decoder

MingTok

MingTok

###### MingTok

Input Image

Generated Image

Edited Image

- Figure 3 The architecture of Ming-UniVision. Owing to the autoregressive semantic decoding capability of MingTok, both image understanding (image-to-text generation) and image synthesis (text-to-image generation) can be formulated consistently with the same next-token prediction paradigm and unified input representation space. This allows our unified multimodal model to support multi-round in-context tasks, seamlessly switch from understanding to generation/editing task, and vice versa.

This vision head is inspired by recent works on unified multi-modal modeling (Li et al., 2024; Fan et al., 2025), but incorporates two key structural improvements. First, we replace the diffusion-based denoising head with a rectified flow (Liu et al., 2022) prediction objective, which allows for faster convergence and fewer inference steps. Second, we adopt a SwiGLU-based feed-forward network (FFN) (Shazeer, 2020) in place of the standard MLP block, which we find empirically improves latent prediction accuracy and final image quality under the same parameter budget.

Together, the unified input representation and next-token prediction enable a single model to universally handle understanding, generation, and editing, simplifying the architecture for multiround in-context image understanding, generation, and manipulation, which we elaborate in Sec. 3.2.

##### 3.2 Multi-round In-context Image Understanding, Generation and Manipulation

While recent unified vision-language models have made significant advances in joint understanding and generation (Deng et al., 2025; Chen et al., 2025d; Fan et al., 2025), they still follow a fragmented generate-then-understand pipeline. Whether through dual branches (Deng et al., 2025; Shi et al.,

- 2024) or separate tokenizers (Fan et al., 2025), these approaches decouple the generation and perception processes into distinct spaces, leading to repeated encoding and decoding cycles, imposing significant overhead on iterative in-context refinement. There are several major obstacles preventing such architectures from supporting efficient and scalable multi-round in-context image generation:

DiT is structurally incapable of multi-round in-context editing. Diffusion Transformers (DiTs), such as FLUX.1, are architecturally designed to generate a fixed number of images per forward pass. During training, the diffusion transformers are configured to generate images based on a pre-determined number of reference images, resulting in a static input-output structure. This rigidity limits their ability to dynamically extend generation sequences or flexibly interleave image editing steps within a single context, rendering them ill-suited for adaptive, multi-round tasks.

Hybrid AR–Diffusion is burdened with dual-branch overhead. Hybrid AR-Diffusion designs (Deng et al., 2025; Shi et al., 2024) integrate autoregressive and per-image diffusion to enable in-context multi-round generation and editing. While the dual-branch architecture supports

multi-round generation, it introduces significant computational and implementation overhead:

- • Training computation overhead: Hybrid models maintain multiple distinct representations per image, i.e., semantic features for understanding, noised latents for denoising, and clean latents for conditioning future steps. This substantially increases the effective token sequence length during training, leading to higher memory consumption and longer training times.
- • Training complexity: Unconventional attention masking schemes are required to manage cross-feature space and cross-round dependencies:

- – Noisy tokens from prior generation rounds are masked out in subsequent generation steps, ensuring that only the clean latents are observed in future image generation processes.
- – Different masking strategies are applied to across distinct feature spaces: Causal attention over semantic features and full attention over image latents to support global denoising.

- • Inference inefficiency: Multi-round generation requires frequent conversions between heterogeneous spaces: latent space (generation) → pixel space (full decoding via VAE) → feature space (semantic encoding via understanding encoder). After each round of generation, a full decode-encode cycle is needed, increasing both latency and computational overhead.

Consequently, while AR-Diffusion frameworks supports multi-round editing, they suffer from increased architectural complexity, reduced training stability, and inefficient inference, posing practical challenges for scalable deployment.

##### Unified AR is limited by separated tokenization. Architectures such as UniFluid (Fan et al.,

- 2025) adopt a single-branch autoregressive loop to unify understanding, generation and editing within a shared sequence modeling framework. Compared to AR-Diffusion hybrid models, their unified architectures simplify training and inference by relying on a single next-token prediction objective, eliminating the need for complex masking schemes.

However, they still rely on distinct representations for understanding and gneration, which necessitates frequent conversions between domains during multi-round editing. Moreover, during training, both semantic and generative token sequences are processed in parallel, effectively doubling the input length and increasing memory and computational overhead. As a result, despite architectural simplification, unified autoregressive models still inherit key inefficiencies from hybrid approaches, particularly in terms of latency and scalability in iterative editing scenarios.

In contrast, Ming-UniVision unifies understanding and generation within a single continuous token space (see Section 2). This is made possible by the unified input representation enabled by MingTok, which allows the high-dimensional features from the semantic decoder to be reused as conditional inputs for generation or editing, without costly pixel-space detour. As illustrated in Fig. 4, this design supports efficient in-context interaction, enabling reversible edits, faithful reconstruction, and iterative refinement, all while preserving the full context in the latent space.

This design enables a seamless, in-place iterative workflow during inference: after generating an image, its semantic feature representation St remains in the latent space. For the next step t + 1, St is concatenated with a new textual instruction (e.g., “add a hat”) and fed back into the model to produce an updated semantic feature St+1. Because the entire process is executed purely in the latent space—bypassing costly pixel-space re-encoding—it avoids cumulative quality degradation, preserves visual fidelity, and supports low-latency, multi-round editing and generation that can fluidly interleave image understanding and free-form content creation.

Semantic visual tokens Visual latent tokens Noised visual latent tokens

#### …

Hybrid AR-diffusion

BOI

BOI

EOI

EOI

Und Encoder Edit instruction 1

VAE Encoder VAE Encoder Edit instruction 2

VAE Encoder VAE Encoder Und Encoder

Image

Image Image Image Image

Image

#### …

Uniﬁed AR

BOI

EOI

BOI

EOI

VAE Encoder Edit instruction 1 Und Encoder

VAE Encoder Edit instruction 2

Und Encoder

Image

Image Image

Image

#### …

Ming-UniVision

BOI BOI

EOI

EOI

50% less visual tokens

Edit instruction 2

Edit instruction 1

Semantic Decoder

Semantic Decoder

66% less visual tokens

Low-level Encoder

Low-level Encoder

MingTok

MingTok

Image

Image

- Figure 4 Comparison of input token structures across different unified model architectures. Ming-UniVision reduces the number of input visual tokens by 66% compared to hybrid AR-diffusion models (Shi et al., 2024; Deng et al., 2025) and by 50% compared to existing unified autoregressive models (Fan et al., 2025), thanks to the unified representation enabled by MingTok.

##### 3.3 Training

The training of our unified multi-modal model is divided into two main phases: pre-training on large-scale web data to establish foundational vision-language capabilities, followed by supervised fine-tuning (SFT) on curated datasets to enhance instruction-following behavior and support complex multimodal interactions. This staged strategy ensures stable optimization while enabling progressive acquisition of understanding, generation, and editing skills.

- 3.3.1 Pre-training Pre-training consists of two stages designed to align representations before joint modeling.

- Stage 1: MLP and rectified flow head warm-up. In this initialization phase, we focus on training the MLP between MingTok and the LLM, and the per-token vision head for latent prediction. The MingTok and the LLM backbone remain fixed during this stage. We use a mix of image-text pairs, with approximately 30% dedicated to understanding tasks and 70% to autoregressive generation tasks. This warms up both the vision-to-language and the language-to-vision pathways.
- Stage 2: Joint image understanding and generation pre-training. In this stage, we aim to build robost single-turn vision-language capabilities using large-scale image-text data. Since the prediction of per-token rectified flow head is limited to the current token, it relies on the language model to model sequential relationships among visual tokens. Hence, in this stage, we unlock the language model, allowing it to capture inter-token structure during autoregressive generation.

To enhance fine-grained visual perception without destabilizing the pre-trained latent space, we introduce mixed resolution training and selectively unlock only the semantic decoder of MingTok, keeping the low-level encoder fixed. During understanding tasks, images are resized to 1024×1024, and the semantic decoder learns to produce high-fidelity, detail-rich embeddings taht align with text semantics. For text-to-image generation, the inputs remain at 512×512 in consideration of the computational efficiency and compatibility with the pre-trained compact latent space.

This setup enables the model to perceive fine details during understanding while retaining stable, fast generation—critical for downstream editing and in-context interaction. The training data consists of approximately 25% image-text understanding pairs, 70% text-to-image generation samples, and 5% general NLP tasks.

- Table 2 Quantitative evaluations on MMBench (Liu et al., 2024a), MMStar (Chen et al., 2024b), MMMU (Yue et al., 2024), MathVista (Lu et al., 2023), HallusionBench (Guan et al., 2024), AI2D (Kembhavi et al., 2016), MM-Vet (Yu et al., 2023), OCRBench (Liu et al., 2024b), and MME (Fu et al., 2023).

Model MMB↑ MMS↑ MMMU↑ MathV↑ Hall↑ AI2D↑ MM-Vet↑ OCRBench↑ MME↑

Understanding Only Emu3-Chat (Wang et al., 2024) 58.5 - 31.6 - - - 37.2 687 Qwen2.5-VL-3B (Bai et al., 2025) 79.1 55.9 53.1 62.3 46.3 81.6 - 797 2157 Qwen2.5-VL-7B (Bai et al., 2025) 83.5 63.9 58.6 68.2 52.9 83.9 67.1 864 2347 InternVL2.5-4B (Chen et al., 2024c) 81.1 58.3 52.3 60.5 46.3 81.4 60.6 828 2338 InternVL2.5-8B (Chen et al., 2024c) 84.6 62.8 56.0 64.4 50.1 84.5 62.8 822 2344 DeepSeek-VL2 (Wu et al., 2024b) 79.6 61.3 51.1 62.8 - 81.4 - 811 2253

Unified model, Separate representation Janus-Pro-7B (Chen et al., 2025d) 79.2 - 41.0 - - - 50.0 - LMFusion (Shi et al., 2024) - - 41.7 - - - - - 1603 MetaQuery-L (Pan et al., 2025) 78.6 - 53.1 - - - 63.2 - Show-o2-7B (Xie et al., 2025) 79.3 56.6 48.9 - - 78.6 - - BLIP3-o 4B (Chen et al., 2025c) 78.6 - 46.6 - - - 60.1 - 2161 BAGEL (Deng et al., 2025) 85.0 - 55.3 73.1 - - 67.2 - 2388

Unified model, Unified representation

VILA-U (Wu et al., 2024a) - - - - - - 33.5 - 1402 TokenFlow-XL (Qu et al., 2025) 76.8 - 43.2 - - - 48.2 - 1922 UniTok (Ma et al., 2025) - - - - - - 33.9 - 1448 Harmon-1.5B (Wu et al., 2025c) 65.5 - 38.9 - - - - - 1476 TokLIP (Lin et al., 2025b) 67.6 - 43.1 - - - 29.8 - -

Ming-UniVision-16B-A3B (Ours) 78.5 63.7 40.3 66.6 47.8 82.8 64.2 724 2023

##### 3.3.2 Supervised Finetuning

After pre-training, we perform supervised fine-tuning on high-quality datasets and custom instructionfollowing corpora to enhance instruction adherence, generation fidelity, and support for complex multi-modal interactions. This phase has two stages that progressively introduce task complexity.

- Stage 1: Image understanding and generation. This stage focuses on aligning the model with human intent in standard vision-language tasks. We freeze MingTok and unlock the remaining parts,

- as we observe no performance gain when the semantic decoder is unlocked during this stage. Mixed resolution training is further employed in this stage. The data distribution includes approximately 30% for understanding tasks, 10% for NLP tasks, and 60% for text-to-image generation.

- Stage 2: Image understanding, generation, and in-context image manipulation. To enable multi-round context-aware instructions such as iterative editing and refinement, we introduce a final fine-tuning stage focused on image generation and in-context manipulation. We constructed the instruction chains for the model to learn in-context image manipulation (see details in Sec. 5.2). The training strategy follows the same strategy as stage 1, with data composition significantly shifted, 15% understanding, 5% NLP, 35% standard text-to-image generation, and 55% single or multi-round editing tasks.

### 4 Evaluations

Ming-UniVision is evaluated on a wide range of image understanding and generation benchmarks. For the comparison with existing approaches, we employ Ling-lite language model featuring 2.8 billion activated parameters.

###### Table 3 Evaluation of text-to-image generation ability on GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024). † denotes performance obtained by rewritten prompts.

###### Method Single Obj.↑ Two Obj.↑ Counting↑ Colors↑ Position↑ Color Attri.↑ Overall↑ DPG-Bench↑

Generation Only

LlamaGen (Sun et al., 2024) 0.71 0.34 0.21 0.58 0.07 0.04 0.32 PixArt-α (Chen et al., 2023) 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDv2.1 (Rombach et al., 2022) 0.98 0.51 0.44 0.85 0.07 0.17 0.50 -

- DALL-E 2 (Ramesh et al., 2022) 0.94 0.66 0.49 0.77 0.10 0.19 0.52 Emu3-Gen (Wang et al., 2024) 0.98 0.71 0.34 0.81 0.17 0.21 0.54 80.60 SDXL (Podell et al., 2023) 0.98 0.74 0.39 0.85 0.15 0.23 0.55 74.65
- DALL-E 3 (Betker et al., 2023) 0.96 0.87 0.47 0.83 0.43 0.45 0.67 83.50 SD3-Medium (Esser et al., 2024) 0.99 0.94 0.72 0.89 0.33 0.60 0.74 84.08

Unified model, Separate representation Show-o (Xie et al., 2024) 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Ming-Lite-Uni (AI et al., 2025b) 0.99 0.76 0.53 0.87 0.26 0.30 0.62 Janus-Pro-1B (Chen et al., 2025d) 0.98 0.82 0.51 0.89 0.65 0.56 0.73 82.63 Janus-Pro-7B (Chen et al., 2025d) 0.99 0.89 0.59 0.90 0.79 0.66 0.80 84.19 Show-o2-7B (Xie et al., 2025) 1.00 0.87 0.58 0.92 0.52 0.62 0.76 86.14 MetaQuery-L† (Pan et al., 2025) - - - - - - 0.78 81.10 Blip3-o 4B (Chen et al., 2025c) - - - - - - 0.81 79.36 BAGEL (Deng et al., 2025) 0.99 0.94 0.81 0.88 0.64 0.63 0.82 -

Unified model, Unified representation Harmon-1.5B (Wu et al., 2025c) 0.99 0.86 0.66 0.85 0.74 0.48 0.79 TokenFlow-XL (Qu et al., 2025) 0.95 0.60 0.41 0.81 0.16 0.24 0.55 73.38 Ming-UniVision-16B-A3B (Ours) 1.00 0.93 0.59 0.93 0.92 0.70 0.85 82.12

##### 4.1 Multi-modal Understanding

The understanding capability of our model is summarized in Table 2, where we compare with existing models of comparable parameter scale. Compared to both dedicated vision-language understanding models and unified architectures that employ separate representations for perception and generation, our model achieves comparable overall performance.

Notably, Ming-UniVision shows competitive results on MMStar (Chen et al., 2024b), HallusionBench (Guan et al., 2024), AI2D (Kembhavi et al., 2016), and MM-Vet (Yu et al., 2023), which evaluate semantic reasoning and hallucination detection. This indicates that the shared semantic representation learned by MingTok is sufficiently expressive for general-purpose vision-language understanding. However, we observe a performance gap on OCRBench (Liu et al., 2024b) and MMMU (Yue et al., 2024), where fine-grained recognition is critical. This suggests limitations in preserving character-level details, which is likely due to the compressed nature of the latent space used during autoregressive generation as well as the causal architecture of the semantic decoder. We leave the optimization for detail-preserving latents and semantic representations for future work.

##### 4.2 Visual Generation

The visual generation capabilities of Ming-UniVision are benchmarked in Table 3, where we evaluate on GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024). Compared with existing models, our model achieves state-of-the-art performance on the overall GenEval benchmark.

Notably, it particularly excels in attribute control and spatial reasoning, outperforming all other models in the Position (0.92), Colors (0.93), and Color Attribute (0.70) sub-tasks. The significant lead in the position-related tasks underscores our model’s superior compositional control. This strong performance, combined with faster training convergence inherent to our unified architecture, highlights the effectiveness of the shared semantic space in guiding image synthesis.

We attribute these improvements to the joint perception–generation representation, which facilitates

###### Table 4 Evaluation on the image reconstruction ability on the validation set of ImageNet (Deng et al., 2009).

###### Table 5 Image editing performance on GEdit-Bench (EN) (Liu et al., 2025). Metrics are evaluated by GPT-4.1.

Tokenizer Res. # Tokens rFID↓ PSNR ↑ SSIM ↑ LPIPS ↓ Specialized tokenizers SD-VAE (Esser et al., 2024) 256 1024 1.06 28.62 0.86 GigaTok (Xiong et al., 2025) 256 256 0.51 21.32 0.69 0.21 VA-VAE (Yao et al., 2025) 256 256 0.26 28.59 0.80 0.09 DC-AE (Chen et al., 2024a) 512 64 0.22 26.15 0.71 0.08 MAE-Tok (Chen et al., 2025b) 512 128 0.62 - - TexTok (Zha et al., 2025) 512 256 0.73 24.45 0.66 0.19 Unified tokenizers

UniTok (Ma et al., 2025) 256 256 0.38 - - TokenFlow (Qu et al., 2025) 384 729 0.63 22.77 0.73 -

MingTok 512 256 0.54 30.77 0.62 0.14 MingTok † 512 256 0.38 31.09 0.64 0.12

† denotes using semantic decoder after joint pre-training.

Model G_SC G_PQ G_O Specialized image editing model Instruct-P2P (Brooks et al., 2023) 3.58 5.49 3.68 MagicBrush (Zhang et al., 2023) 4.68 5.66 4.52 AnyEdit (Yu et al., 2025) 3.18 5.82 3.21 UniWorld-V1 (Lin et al., 2025a) 4.93 7.43 4.85 OmniGen (Xiao et al., 2025) 5.96 5.89 5.06 OmniGen2 (Wu et al., 2025b) 7.16 6.77 6.41 Step1X-Edit (Liu et al., 2025) 7.09 6.76 6.70 Unified model BAGEL* (Deng et al., 2025) 7.36 6.83 6.52 Ours (single-round) 6.04 6.86 5.54 Ours (multi-round) 6.60 6.25 5.78

* indicates model pretrained on large-scale interleaved data.

both semantic grounding and efficient optimization. Further enhancements in fine-grained detail generation remain an avenue for future work.

##### 4.3 Visual Editing

To evaluate the image editing performance of our model, we use GEdit-Bench-EN (Liu et al., 2025), a benchmark featuring real-world user instructions across 11 diverse categories. Performance is measured using three metrics: Semantic Consistency (SC), Perceptual Quality (PQ), and Overall Score (O), all on a 0–10 scale. Since our model does not rely on large-scale interleaved pre-training, we find consistent resolution between understanding and generation stages to be critical for effective editing. Therefore, we report results using a base model trained without mixed-resolution strategy. As shown in Table 5, our model achieves competitive single-round editing quality (G_PQ) compared to existing methods, while demonstrating strong multi-round success rates (G_SC). Although the overall score lags behind prior work, we attribute this gap primarily to two factors: the absence of large-scale multimodal sequence pre-training, and the high per-token detail density in our continuous tokenizer—both of which limit current fidelity under complex instructions. We discuss these limitations in depth in Sec. 5.4 and plan to address them in future work.

##### 4.4 Image Reconstruction

We compare the reconstruction performance of our model with existing tokenizers in Table 4. MingTok operates at a 32× compression ratio, encoding 512×512 images into a compact representation of 256 continuous latent tokens. Under this high compressionm MingTok achieves an rFID of 0.54 and a PSNR of 30.77 db, indicating strong structural alignment and high pixel fidelity. After the semantic decoder is jointly trained during the pre-training process of the unified multi-modal model, the reconstruction quality is further improved, with LPIPS decreasing to 0.12 and rFID dropping to 0.38. This suggests that end-to-end optimization within the unified framework enhances the semantic decoder’s ability to retain fine textures and global semantics.

### 5 Analysis

In this section, we analyze some critical designs in our Ming-UniVision. Here, we switch to smaller data scale during pre-training compared to previous experiments, and a smaller dense language model, i.e., Qwen-2.5-3B (Team, 2024), for efficient ablation analysis.

Table 6 Ablation study on visual representation designs for unified multi-modal models. Und_Tok and Gen_Tok denote the tokenizers used for understanding and generation, respectively. When MingTok serves as both the understanding and generation tokenizer, the model achieves optimal performance on both understanding and generation tasks, demonstrating the advantage of a unified, shared visual space.

Und_Tok Gen_Tok MMB MMStar MMMU Mathvista AI2D OCRBench Average ↑ GenEval ↑

CLIP VAE 66.41 51.03 40.00 48.0 69.98 36.1 51.92 0.3591 CLIP MingTok 67.18 52.13 38.67 49.5 72.25 35.5 52.54 0.4600 MingTok VAE 69.42 51.57 40.89 47.9 71.99 30.4 52.03 0.3950 MingTok MingTok 69.93 51.89 40.44 52.3 75.23 31.6 53.57 0.4654

##### 5.1 Unified Representation for Competing Tasks

In this section, we explore the influence of unified visual representation on the understanding and generation capability of unified multi-modal models. For a comprehensive comparison, three tokenizers are introduced into our experiments: CLIP for image understanding, VAE for image generation, and the proposed MingTok, which can be used either for understanding or generation. By combining these tokenizers, we get four kinds of unified multi-modal models for image understanding and generation, to be specific: 1) CLIP as und_tok (short for understanding tokenizer) and VAE as gen_tok (short for generation tokenizer), 2) CLIP as und_tok and MingTok as gen_tok,

- 3) MingTok as und_tok and VAE as gen_tok, and 4) MingTok as both und_tok and gen_tok. It’s noteworthy that the former three have separate understanding and generation representation spaces while the last one has unified representation space.

Unified representation matters for understanding. As can be seen in Table. 6, the last row MingTok as both und_tok and gen_tok performs the best on ‘Average’, which implies that pretraining in a unified representation space has better image understanding performance than that in two separate spaces. To look further, within each und_tok group, the performance is inferior when VAE is used as gen_tok. The reason is speculated to be that during joint training, the MLLM has to expend considerable effort to align the understanding and generation representation spaces. From this perspective, since VAE’s features focus more on details and have little semantic information, while MingTok’s features themselves contain sufficient semantic information, so it is easier for MingTok than VAE to align with und_tok’s understanding representations.

Unified representation matters for generation. First of all, MingTok demonstrates its ability to image generation task in Table. 6, as we can see that no matter what und_tok is, MingTok as gen_tok always shows a significant ‘GenEval’ improvement compared to its VAE counterpart. We hypothesize that this advantage may come from the fact that MingTok’ features, not only contain the detail information used for image reconstruction but also contain sufficient semantic information, which can probably accelerate the convergence of image generation. So in this way, MingTok can serve as an alternative to existing image generation tokenizers. Besides, the best generation performance is also obtained when MingTok acts as both und_tok and gen_tok, which indicates that pretraining in a unified representation space is more effective for image generation tasks than that in different representation spaces.

To further explore the effect of MingTok as an image generation tokenizer, we study the training process of the generation tasks, as illustrated in Figure. 5, with two additional settings: pure generation with MingTok (denoted as MingTok(G)), and pure generation with VAE (denoted as VAE(G)). From these curves, it’s easy to draw the following conclusions: 1) Generation-only models achieve superior performance than joint trained models with both understanding and generation

###### Generation Performance Comparison of Stage 1 Pre-training

###### Generation Performance Comparison of Stage 2 Pre-training

[Figure 71]

[Figure 72]

60

40

50

30

GenEvalScore

GenEvalScore

40

20

30

VAE (G)

VAE (G)

MingTok (G)

MingTok (G)

10

MingTok (G & U)

MingTok (U) / VAE (G)

MingTok (G & U)

MingTok (U) / VAE (G)

20

CLIP (U) / MingTok (G)

CLIP (U) / MingTok (G)

CLIP (U) / VAE(G)

CLIP (U) / VAE(G)

50 100 150 200 250 300

0 100 200 300 400 500 600 700

Training iterations (K steps)

Training iterations (K steps)

- Figure 5 Generation performance comparison during pre-training across different understanding (U) and generation (G) tokenizer combinations. Using MingTok as the generation representation (MingTok (G)) achieves the best performance in generation-only training, significantly outperforming VAE-based representations (VAE (G)). When MingTok is used for both roles (MingTok (G & U), unified setting), the performance gap between pure generation and unified training narrows notably, demonstrating the benefit of universal visual representations for joint vision-language modeling.

capabilities; 2) MingTok outperforms VAE as an image generation tokenizer; 3) Joint training in a unified representation space minimizes performance degradation in image generation tasks.

##### 5.2 Multi-round Understanding, Generation, and Editing

As discussed in Sec. 3.2, most existing unified architectures either do not support explicit multi-round training, or—when extended to such scenarios—must maintain tokens from multiple heterogeneous feature spaces (separately optimised for understanding and for generation) in memory simultaneously. This heterogeneity not only complicates the attention mechanism during multi-round training, but also makes the sequential editing process more cumbersome to optimise.

To examine how task formulation impacts multi-round performance, we begin with two foundational comparison experiments:

- • Recon + Edit (Baseline): A standard single-round setting in which the model reconstructs the original image and then performs a single edit.
- • Add Seg-as-Edit (Proposed): Extends the baseline by adding a reconstruction + segmentation editing task. Specifically, a portion of the training samples are modified to require reconstruction followed by a segmentation-as-editing operation, encouraging the model to learn fine-grained boundary localisation and semantic consistency within its latent space.

Analysis. Across categories, Seg-as-Edit improves semantic consistency in 9/11 tasks, with the largest gains in motion_change (+0.82 G_SC) and background_change (+0.52 G_SC). The average G_SC rises +0.41, and G_O rises +0.33, indicating better preservation of target semantics and overall output quality. Perceptual quality remains on par in most categories, reflecting that structural regularisation via segmentation strengthens consistency without sacrificing visual fidelity. Qualitative examples in Fig. 6 further show that styles and local details are better maintained over successive edits.

Table 7 Multi-round editing performance: Baseline vs. Seg-as-Edit.

Recon+Edit Add Seg-as-Edit G_SC G_PQ G_O G_SC G_PQ G_O

Category

background_change 6.931 6.345 6.177 7.448 6.138 6.563 color_alter 8.118 6.588 6.069 8.118 5.912 6.595 material_alter 5.536 5.714 4.989 5.214 5.929 4.651 motion_change 2.818 6.818 2.741 3.636 6.727 3.459 ps_human 3.756 8.268 4.047 4.171 8.146 4.429 style_change 7.167 5.042 5.788 7.271 5.292 6.032 subject-add 6.026 7.684 5.674 6.605 7.289 6.208 subject-remove 8.690 7.262 7.542 8.857 7.238 7.747 subject-replace 6.891 5.913 5.987 7.609 5.957 6.418 text_change 3.938 5.210 3.637 4.469 5.185 4.238 tone_transfer 7.800 7.040 7.005 7.520 6.760 6.894

Average 6.034 6.535 5.423 6.447 6.416 5.749

Such targeted improvements in multi-round robustness form a natural bridge to practical, real-world workflows, where editing often proceeds through long, dependent sequences of transformations.

Qualitative Analysis of Practical Workflows. Beyond controlled ablations, the strengths of our unified architecture are most evident in complex sequential scenarios typical of creative practice the very situations where prior art suffers from feature-space fragmentation and attention complexity. In popular single-round editors such as Qwen-Image, each edit is treated independently (stateless editing), making it difficult to maintain identity consistency: facial features or clothing can drift noticeably between iterations. In contrast, our method preserves a coherent latent representation across steps, ensuring consistency even in extended editing chains.

- • Old Photograph Restoration: Multi-stage processes like super-resolution followed by colourisation often accumulate errors. Our model transitions seamlessly from resolution enhancement to plausible colouring within a stable latent space (Fig. 7), preserving fine detail and scene coherence.
- • Iterative Super-Resolution: In traditional methods, repeated super-resolution magnifies artifacts. MingTok maintains structure and introduces plausible details across iterations (low-res → super-res → super-res), avoiding quality degradation cascades.
- • High-Fidelity Subject Matting: Leveraging segmentation-as-editing to create a high-quality alpha mask prior to background removal minimises halo artifacts and retains delicate structures like individual hair strands, as in Fig. 7.

These practical cases highlight the core benefit of MingTok’s stateful design: by unifying understanding and generation in a single feature space, it converts fragmented, multi-step edits into a coherent creative flow, enabling more powerful and reliable interactive tools.

##### 5.3 Visualized CoT for Visual Reasoning and Image Editing

Building on the multi-image generation capabilities of MingTok described in Sec. 3.2, we introduce Visualized Chain-of-Thought (Visualized CoT), which enables explicit visual reasoning by generating intermediate visualizations prior to image editing. This novel paradigm first highlights the regions of the reference image that require modification according to the editing instructions.

[Figure 73]

[Figure 74]

Input Reconstruct the image Change the dress to red

Input Reconstruct the image Change the dress to red

| |
|---|

[Figure 75]

[Figure 76]

[Figure 77]

Input Reconstruct the image Remove smiles Input Reconstruct the image Remove smiles

[Figure 78]

[Figure 79]

Input Reconstruct the image Change the dress to white

Input Reconstruct the image Change the dress to white

Baseline Add Seg Data

- Figure 6 Qualitative comparison of multi-step editing across the three training strategies. Column 1 (Baseline): Struggles with sequential edits, altering the shirt’s style during an incomplete color change and leaving artifacts after smile removal. Column 2 (Add Seg-as-Edit): Benefiting from the segmentation task, our main model improves color fidelity and preserves the shirt’s style; smile removal is cleaner.

Table 8 Quantitative comparison between single-step editing and two-step generative Visualized CoT approach.

GEdit-Bench-EN-Sub GEdit-Bench-EN-Full G_SC G_PQ G_O G_SC G_PQ G_O

Method

Single-step 6.055 6.890 5.582 6.042 6.855 5.535 Visualized CoT 6.543 6.278 5.791 6.490 6.259 5.668

Subsequently, the edited image is generated guided by these visual cues.

Unlike Generation Chain-of-Thought (GoT) (Fang et al., 2025), which conducts reasoning in natural language, Visualized CoT performs reasoning visually by directly generating an image in which the regions to be edited are highlighted with overlayed colors. Moreover, GoT requires translating its reasoning outputs into an editing mask, which is then encoded to condition the image editing process. In contrast, our method directly leverages the visualized context to guide the editing. This end-to-end visual reasoning and generation framework enables seamless integration of understanding and editing, enhancing both transparency and efficiency in the image editing workflow.

To construct the training data, we follow UniWorld-V1 (Lin et al., 2025a) to obtain the editing regions by calculating the differences between edited images and their reference images. The resulting editing region masks are then overlaid onto the reference images as intermediate outputs for visual reasoning. The edited images are then used as the final desired outputs, forming a twostep Visualized CoT image editing paradigm. The unified feature space for visual generation and understanding enables this multi-image generation framework to be trained end-to-end.

To evaluate the effectiveness of this new paradigm, we conduct quantitative evaluations on the GEdit-Bench (Liu et al., 2025). We compare our method against a standard single-step pipeline, which serves as a baseline by directly generating the edited image from the reference image and

###### Input QwenEdit-2509 Input QwenEdit-2509

Ming-UniVision-R1 Ming-UniVision-R2 Ming-UniVision-R1 Ming-UniVision-R2

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

(a) Old photograph restoration (b) Iterative Super-Resolution

[Figure 96]

[Figure 97]

Input QwenEdit-2509 Input QwenEdit-2509

Ming-UniVision-R1 Ming-UniVision-R2 Ming-UniVision-R1 Ming-UniVision-R2

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

(c) High-Fidelity Subject Matting

- Figure 7 Qualitative illustration of multi-step visual editing workflows enabled by our unified model. (a) An old photograph is restored through a sequential process: first upscaled to higher resolution, then colorized. (b) Further iterative super-resolution can be applied to progressively enhance image quality, demonstrating the model’s capability for context-preserving, in-context refinement. (c) High-fidelity subject matting is achieved in two steps: segmentation to highlight the subject, then background removal.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Input Get editing region Erase the knife Input Get editing region Remove the peanuts Input Get editing region Make the girl smile

Input Get editing region Replace with jade Input Get editing region Add a golden glass ball Input Get editing region Add a hot air balloon

- Figure 8 Illustration of the Visualized CoT. Under this paradigm, the model first performs visual reasoning to generate an image in which the regions requiring editing are highlighted on the reference image. Subsequently, the model completes the image editing according to these visual cues.

textual instructions. As show in Table 8, our Visualized CoT approach outperforms the single-step baseline, most notably in semantic consistency, with a gain of +0.5 points. This improvement is attributed to the intermediate visual reasoning result, which introduces a strong spatial prior and reduces editing ambiguity. Figure 8 illustrates the visual reasoning process of Visualized CoT, where the model accurately identifies the regions requiring editing.

##### 5.4 Discussions

This work represents an early step toward unified vision-language modeling in continuous latent spaces. Our goal in releasing this version is not to claim state-of-the-art performance across all tasks, but to highlight the potential of using a single, shared continuous representation for both visual understanding and autoregressive generation. While MingTok and Ming-UniVision demonstrate promising capabilities in joint perception and synthesis, it still has its limitations, particularly in fine-grained editing and understanding. We plan to address these in subsequent versions.

Limitations on current editing performance. While our model supports flexible in-context image editing through autoregressive sequence modeling, its performance on quantitative editing

benchmarks has room for improvement. We identify two key factors that currently limit editing success rate and quality.

First, the model lacks large-scale interleaved-pretraining, i.e., pre-training on sequences that alternate between text and image tokens across diverse editing scenarios. Such data could help the model learn generalizable edit patterns before fine-tuning. Without it, the model relies heavily on SFT to acquire editing behavior, which may not generalize well beyond seen prompts. This limitation is particularly pronounced under mixed-resolution training, where generation and editing operate

- at lower resolutions and thus cannot leverage the understanding capability learned during highresolution understanding training. Moving forward, achieving optimal joint performance in a unified resolution setting will be a key focus of our next-stage development.

Second, due to the high compression ratio of MingTok designed for generation efficiency, each latent token encodes a large amount of visual detail. This high information density makes fine-grained editing challenging, as small changes in tokens can lead to significant and often uncontrollable changes in pixel space. In future work, we plan to explore higher-resolution tokenizeation or lower compression ratios to reduce per-token information load, thereby imprving the precision and quality in both generation and editing.

Challenges in multi-round and freeform interleaved interaction. While MingTok supports basic in-context editing, it still falls short in more advanced interaction patterns. In multi-round editing, we observe that the model struggles to generalize to editing sequences longer than those seen during training. More fundamentally, the model yet still struggles with freeform interleaved understanding and generation (or editing), such as arbitrarily ordered sequences like "describe, generate, compare, revise, regenerate, etc.". The current training paradigm focused on structured, unidirectional flows, does not sufficiently prepare the model for flexible, dynamic task switching. We leave the exploration of length-generalizable editing strategies and training on diverse, real-world interaction trajectories for future work.

Mutual enhancement between generation and understanding. Finally, we emphasize that a unified visual representation is not merely an architectural choice, but a key enabler for mutual enhancement between generation and understanding. By sharing the same representation space across tasks, MingTok allows knowledge learned in generation—such as fine-grained texture synthesis and compositional reasoning—to benefit perception, while visual understanding provides grounded, coherent priors for more controllable and faithful generation. We observe early evidence of this synergy: using a shared representation reduces the performance gap between pure generation and unified training, and alleviates task competitions that typically arise when pathways diverge. We hope this perspective inspires the research community to further explore unified modeling of generation and understanding, moving toward more integrated and synergistic multimodal systems.

### 6 Contributors

Ziyuan Huang, DanDan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, Taozhi Huang, Jiajia Liu, Qingpei Guo, Ming Yang, Jingdong Chen, Jun Zhou

### References

Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, et al. Ming-omni: A unified multimodal model for perception and generation. arXiv preprint arXiv:2506.09344, 2025a.

Inclusion AI, Biao Gong, Cheng Zou, Dandan Zheng, Hu Yu, Jingdong Chen, Jianxin Sun, Junbo Zhao, Jun Zhou, Kaixiang Ji, et al. Ming-lite-uni: Advancements in unified architecture for natural multimodal interaction. arXiv preprint arXiv:2505.02471, 2025b.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

Cong Chen, Ziyuan Huang, Cheng Zou, Muzhi Zhu, Kaixiang Ji, Jiajia Liu, Jingdong Chen, Hao Chen, and Chunhua Shen. Hieratok: Multi-scale visual tokenizer improves image reconstruction and generation. arXiv preprint arXiv:2509.23736, 2025a.

Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. In Forty-second International Conference on Machine Learning, 2025b.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025c.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37: 27056–27087, 2024b.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025d.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024c.

Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv preprint arXiv:2311.07919, 2023.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al. Kimi-audio technical report. arXiv preprint arXiv:2504.18425, 2025.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. https://arxiv.org/abs/2403.03206.

Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025.

Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, et al. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639, 2025.

Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19358–19369, 2023.

Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 149:105171, 2024.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Yang Jiao, Haibo Qiu, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3600–3610, 2025.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025a.

Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422, 2025b.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024a.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12): 220102, 2024b.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent

diffusion models. pages 10684–10695, 2022. Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting

pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024.

Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1874–1883, 2016.

Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. arXiv preprint arXiv:2502.14831, 2025.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion:

Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. Qwen Team. Qwen2.5: A party of foundation models, September 2024. https://qwenlm.github.io/blog/qwen2.5/. Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via

next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14668–14678, 2022.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025c.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024a.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024b.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9653–9663, 2022.

Tianwei Xiong, Jun Hao Liew, Zilong Huang, Jiashi Feng, and Xihui Liu. Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. arXiv preprint arXiv:2504.08736, 2025.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

Kaiwen Zha, Lijun Yu, Alireza Fathi, David A Ross, Cordelia Schmid, Dina Katabi, and Xiuye Gu. Language-guided image tokenization for generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15713–15722, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

