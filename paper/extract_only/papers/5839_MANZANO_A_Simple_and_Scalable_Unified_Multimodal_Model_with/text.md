## arXiv:2509.16197v1[cs.CV]19Sep2025

[Figure 1]

# MANZANO: A Simple and Scalable Unified Multimodal Model with a Hybrid Vision Tokenizer

Yanghao Li◦†, Rui Qian◦, Bowen Pan◦, Haotian Zhang◦, Haoshuo Huang◦, Bowen Zhang◦‡, Jialing Tong⋆, Haoxuan You⋆, Xianzhi Du⋆, Zhe Gan⋆, Hyunjik Kim⋆, Chao Jia⋆, Zhenbang Wang⋆, Yinfei Yang⋆, Mingfei Gao, Zi-Yi Dou, Wenze Hu, Chang Gao, Dongxu Li, Philipp Dufter, Zirui Wang, Guoli Yin, Zhengdong Zhang, Chen Chen, Yang Zhao, Ruoming Pang‡, Zhifeng Chen

Apple

◦First authors ⋆Core authors †Project lead ‡Work done at Apple

###### Abstract

Unified multimodal Large Language Models (LLMs) that can both understand and generate visual content hold immense potential. However, existing open-source models often suffer from a performance trade-off between these capabilities. We present Manzano, a simple and scalable unified framework that substantially reduces this tension by coupling a hybrid image tokenizer with a well-curated training recipe. A single shared vision encoder feeds two lightweight adapters that produce continuous embeddings for image-to-text understanding and discrete tokens for text-to-image generation within a common semantic space. A unified autoregressive LLM predicts high-level semantics in the form of text and image tokens, with an auxiliary diffusion decoder subsequently translating the image tokens into pixels. The architecture, together with a unified training recipe over understanding and generation data, enables scalable joint learning of both capabilities. Manzano achieves state-of-the-art results among unified models, and is competitive with specialist models, particularly on text-rich evaluation. Our studies show minimal task conflicts and consistent gains from scaling model size, validating our design choice of a hybrid tokenizer.

### 1 Introduction

Unified multimodal models [12, 18, 66, 92, 104], which integrate both understanding and generation capabilities, have become increasingly prominent within the research community. The appeal of this paradigm stems from the discovery that integrating these domains unlocks emergent capabilities [18, 66] in generation, such as complex world reasoning, multimodal instruction following, and iterative visual editing. Yet, in practice, adding generation often degrades understanding. Existing unified models [12, 18, 23, 44] consistently lag far behind their understanding-only counterparts [3, 76, 102], especially on text-rich benchmarks [58, 60].

A key reason for this gap is the conflicting nature of visual tokenization. Auto-regressive generation usually prefers discrete image tokens [5, 56, 95] while understanding typically benefits from continuous embeddings. Many models adopt a dual-tokenizer strategy [12, 23, 85, 91], using a semantic encoder for rich, continuous features while a separate quantized tokenizer like VQ-VAE [86] handles generation. However, this forces the language model to process two different image token types, one from high-level semantic space versus one from low-level spatial space, creating a significant task conflict. While some solutions like Mixtureof-Transformers (MoT) [18, 44] can mitigate this by dedicating separate pathways for each task, they are parameter-inefficient and are often incompatible with modern Mixture-of-Experts (MoE) [24, 41] architectures. An alternative line of work bypasses this conflict by freezing a pre-trained multimodal LLM and connecting it to a diffusion decoder [67, 92, 93]. While this preserves the understanding capability, it decouples generation, losing potential mutual benefits and limiting potential gains for generation from scaling the multimodal LLM.

The bird is flying below the elephant.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

A photo of a corgi with a sign that says 'I am not a real corgi'.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

A hand-drawn blueprint for a time machine, with the caption 'Time Traveling Device'.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Manzano Bagel Janus Pro Nano Banana

#### GPT-4o

- Figure 1 Qualitative text-to-image generation on challenging prompts. Manzano handles counterintuitive, physicsdefying prompts (e.g., ‘The bird is flying below the elephant’) comparably to GPT-4o [35] and Nano Banana [17].

[Figure 17]

Mathvista

MMMU

MMBench

DPG

WISE GenEval

InfoVQA

ChartQA

TextVQA

DocVQA

- Figure 2 Quantitative comparisons on popular understanding and generation benchmarks. Manzano 3B and 30B models achieve superior or competitive performance compared to other SOTA unified multimodal LLMs.

To overcome the above challenges, we propose Manzano, a simple unified model that harmonizes the representations for understanding and generation. Manzanoemploys a unified shared visual encoder with two lightweight and specialized adapters: a continuous adapter for understanding tasks and a discrete adapter for generation. Because two adaptors originate from the same encoder, it yields hybrid representations from a homogeneous source, significantly mitigating task conflict in the LLM. We first pre-train the hybrid tokenizer with a small LLM decoder to pre-align the image features with the LLM feature space. Then the autoregressive multimodal LLMs are jointly trained on a mixture of pure text, image understanding, and image generation data. Finally, we leverage a diffusion image decoder [7, 68] to render pixels by taking the generated image tokens as conditioning.

We train the unified multimodal LLM with a joint recipe to learn image understanding and generation simultaneously. This training consists of three stages: a pre-training stage on a large-scale corpus of text-only, interleaved image-text, image-to-text (IT), and text-to-image (TI) data; a continued pre-training stage on higher-quality IT and TI data; and a supervised fine-tuning (SFT) stage on curated text, IT, and TI instruction data to enhance instruction following capability and improve both understanding and generation tasks.

We demonstrate that Manzano achieves state-of-the-art performance on both understanding and generation tasks. As shown in Fig. 1 and 2, our 3B model, despite its smaller LLM size, achieves competitive generation performance compared to other unified multimodal LLMs. Simultaneously, it delivers significantly better understanding performance, especially on text-rich benchmarks that demand precise perceptual capabilities. Our ablations on the training recipes also indicate minimal cross-task conflict under joint training (Fig. 5). These findings suggest that the architecture and the training recipe effectively mitigate the conflict between understanding and generation, even in a compact model.

Facilitated by the simplicity of the architecture and the joint training recipe, we further investigate the scaling behavior of Manzano. Our scaling studies in Sec. 5.3 show substantial improvements across both understanding and generation benchmarks when scaling the LLM decoder (from 300M to 30B). In addition, enlarging the diffusion decoder also leads to significant gains in image structural integrity, as validated by large-scale human evaluations.

### 2 Related Work

###### 2.1 MLLMs for Image Understanding

Recent advances in Multimodal Large Language Models (MLLMs) have led to a widely adopted architectural pattern that links a vision encoder with a language model through a trainable interface. Typical vision encoders include CLIP [71], SigLIP [101], and the recent InternViT [14]. Early works experimented with elaborate connector designs—for example, Flamingo [2] incorporates gated cross-attention layers to inject image features into the LLM, while BLIP-2 [43] introduces the Q-Former to better align visual and textual representations. A notable departure from these complex strategies is LLaVA [48, 49], which demonstrates that a lightweight Multi-Layer Perceptron (MLP) projection can effectively serve as the connector. This simplicity has since become the blueprint for many follow-up systems, such as the MM1 series [61, 102], the InternVL family [14, 87, 106], and the Qwen-VL models [3, 81], which further improve performance by scaling up both data and backbone models. However, these MLLMs are primarily designed for understanding tasks and lack the capability to generate high-quality images, which limits their applicability in tasks that require bidirectional visual–text reasoning and creation. Despite being limited to understanding tasks, MLLMs still provide valuable strengths—their training recipes and scaling strategies are much more mature than those of current unified multimodal models, which we discuss next.

###### 2.2 Unified Multimodal Models

The integration of image understanding and generation within a single, unified multimodal LLM is becoming prominent. GPT-4o [66] demonstrates embedding image generation capabilities directly into an autoregressive LLM, which unlocks emergent abilities, such as stronger instruction following, improved text rendering, multi-turn visual editing, and sophisticated world knowledge reasoning. Existing unified models can be broadly categorized into three architectural paradigms. First, the unified autoregressive (AR) approach

[5, 9, 10, 12, 23, 27, 30, 31, 56, 66, 82, 85, 94, 95] converts images into sequences of discrete or continuous tokens, enabling LLM to jointly model both image and text sequences in an autoregressive manner. Second, the decoupled LLM-diffusion approach [67, 92, 93] employs a largely frozen LLM for semantic understanding and contextual reasoning, while delegating image synthesis to a separate diffusion decoder. In this design, the LLM itself does not possess native image generation capability. Third, the hybrid AR-diffusion approach [18, 44, 104] integrates both paradigms within a single transformer, using autoregressive decoding for text and an embedded diffusion process for images. Our model is most closely aligned with the first, autoregressive paradigm. However, instead of employing separate tokenizers [12, 18, 23, 93] for understanding and generation, we introduce a unified semantic tokenizer to produce both continuous features for understanding tasks and quantized features for generation tasks. This hybrid tokenizer strategy substantially mitigates the task conflict that commonly arises. Moreover, while our LLM backbone follows the autoregressive design, we augment it with a diffusion decoder for image synthesis, enabling high-fidelity generation guided by the semantic representations generated by the LLM.

###### 2.3 Diffusion Models for Image Generation

Diffusion-based generative models [19, 33, 79] have become one of the most prominent approaches for high-fidelity image synthesis. These models gradually refine Gaussian noise into realistic images through a learned denoising process. Latent diffusion methods [69, 74] enhance computational efficiency by conducting generation in the latent space of a pre-trained variational autoencoder (VAE) [37], reducing memory and compute requirements while preserving visual quality. More recently, flow matching approaches [51, 57, 83] have been introduced to connect source and target distributions via simplified continuous trajectories, leading to further gains in synthesis performance [22]. In parallel, architectural advances such as Diffusion Transformers (DiTs) [7, 8, 22, 38, 68] have demonstrated strong scalability and quality improvements, echoing the success of transformer-based designs in natural language processing.

Building on these developments, our diffusion decoder integrates the strengths of the field by employing a DiT in the latent domain with a conditional flow matching objective. Unlike conventional text-to-image diffusion models [72, 75] conditioned on semantic embeddings from pre-trained text encoders such as CLIP [71], our method leverages visual token embeddings generated by LLM as conditioning signals.

### 3 Model

Manzano is a multimodal large language model (MLLM) that unifies understanding and generation tasks using the auto-regressive (AR) approach. The architecture comprises three components: (i) a hybrid vision tokenizer that produces both continuous and discrete visual representations; (ii) an LLM decoder that accepts text tokens and/or continuous image embeddings and auto-regressively predicts the next discrete image or text tokens from a joint vocabulary; and (iii) an image decoder that renders image pixels from predicted image tokens (see Figure 3 for the framework).

###### 3.1 Design Choices

Unified hybrid representation. The hybrid image tokenizer encodes images into continuous tokens for understanding (I2T), and discrete tokens for generation (T2I), while sharing the same visual encoder.

- • Continuous for I2T. Manzano utilizes continuous embeddings for I2T tasks, a strategy widely adopted in popular visual understanding models [76, 81], which has proven superior performance, especially on text-rich tasks that require more visual details (e.g., DocVQA, ChartQA, and InfoVQA). Our ablation (Table 1) also shows discrete tokens underperform on understanding tasks, which reflects the weak understanding results reported for some pure-discrete unified models [5, 89].
- • Discrete for T2I. Representing images as discrete code indices lets the LLM use the same AR next-token learning strategy as text, simplifying the generation pipeline and scaling behavior.
- • Shared unified semantic space. Both branches originate from the same encoder backbone; thus, continuous and discrete tokens inhabit a common semantic space, which reduces potential task conflict.

###### Hybrid Image Tokenizer Training

###### Inference and Architecture

Understanding task

|[Figure 18]|
|---|

|[Figure 19]|
|---|

[Figure 20]

[Figure 21]

"<img_start> <img_end> A vibrant apple tree laden with red apples."

Image Decoder

“EL MANZANO”

Vision Encoder

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

Unified LLM Decoder

Continuous Adapter

Discrete Adapter

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

Hybrid Image

random sample

Tokenizer "A friendly cartoon tree with a smiling face on its trunk, bearing two red apples. Please generate an image.”

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

|[Figure 22]|
|---|

"What does the sign on the tree say "

300M LM Decoder

Generation task

| |
|---|

| |
|---|

| |
|---|

text loss

Text token Continuous image token Discrete image token

| |
|---|

| |
|---|

| |
|---|

- Figure 3 Our hybrid tokenizer workflow. (Left): The tokenizer produces two distinct but homogeneous feature streams through separate adapters. During training, one adapter output is randomly sampled and passed to a small LLM decoder for alignment. (Right): Once the tokenizer is trained, the right panel illustrates how these two feature types are applied to understanding and generation tasks.

The LLM decoder focuses on regressing high-level semantics (text and image tokens), while the diffusion decoder is responsible for rendering high-fidelity details in pixel space. Many existing unified models rely on separate tokenizers for understanding and generation [12, 18] — for instance, using a CLIP tokenizer for understanding tasks and a VAE tokenizer for generation. Although this strategy preserves more image spatial details, it exacerbates the task conflict within the subsequent LLM. Some studies [9, 10] find that a dedicated generation tokenizer is not as compatible with LLM as the semantic tokenizer. Thus, our hybrid unified image tokenizer employs a single image encoder for both understanding and generation tasks.

Simplicity and scalability. Our design keeps the training losses standard and components cleanly decoupled, which simplifies unification and scaling for the unified MLLM.

- • Unified AR objective. Our unified LLM decoder uses a single AR objective for text-only, I2T, and T2I tasks without additional auxiliary losses or per-task heads.
- • Decoupled components. The clear split between semantic prediction (LLM decoder) and detail generation (image decoder) supports independent scaling of the base LLM and the image decoder.
- • Practical scaling. Our approach readily leverages mature, scalable training pipelines from LLM/MLLM and diffusion decoders. By contrast, prior works (e.g., Transfusion [104] and Bagel [18]) explored incorporating auto-regressive text prediction and a diffusion image generation process for image generation in one LLM, but leave large-scale scaling under-explored. Our decoupled design facilitates scaling the LLM decoder to 30B and diffusion decoder to 3B, yielding promising scaling behavior (Sec. 5.3).

###### 3.2 Architecture

Hybrid Image Tokenizer. Our tokenizer comprises three components: (i) a standard vision transformer (ViT) [20] as the vision backbone; (ii) a continuous adapter, which first applies a 3 × 3 Spatial-to-Channel (STC) layer to reduce the number of spatial tokens by a factor of 9 (e.g., from 42×42×1024 to 14×14×9216) and then uses an MLP to project each feature into the LLM feature dimension (e.g., 2048); and (iii) a discrete adapter, which also starts with the STC compression step but further quantizes the features using finite scalar quantization (FSQ) [62] — chosen for its simplicity and scalability to large codebooks (64K in our experiments) — before applying an MLP projection into the LLM feature dimension.

###### Unified LLM Training

###### Image Decoder Training

|[Figure 23]|
|---|

"A friendly cartoon tree with a smiling face on its trunk, bearing two red apples.

|[Figure 24]|
|---|

"<img_start> <img_end> A vibrant apple tree laden with red apples."

|[Figure 25]|
|---|

<img_start> <img_end> "

Vision Encoder

Vision Encoder

Discrete Adapter

text tokenization

Continuous Adapter

Discrete Adapter

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

Image Decoder

Unified LLM Decoder

[Figure 26]

[Figure 27]

diﬀusion loss

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

text loss image loss

- Figure 4 Training overview. (Left): Unified LLM training with hybrid tokens, the continuous adapter produces embeddings used for the text loss, while the discrete adapter generates hard tokens serving as targets for the image loss. (Right): With vision encoder and adapters fixed, an image decoder is trained to reconstruct images using a diffusion loss.

Unified LLM. We connect our hybrid image tokenizer to a standard text LLM decoder for unified training on a mixture of datasets containing text, understanding, and generation data. For the language backbone, we leverage pre-trained LLMs [29, 105].

Image Decoder. We train an image decoder on top of a pre-trained hybrid image tokenizer to reconstruct images in pixel space from discrete image tokens. Given an input image, the hybrid tokenizer first encodes it into a latent representation, which serves as the conditioning input for a flow-matching pipeline [47] that transports Gaussian noise into realistic images. For the decoder backbone, we adopt the DiT-Air architecture [7], which employs a layer-wise parameter-sharing strategy that reduces the size of the standard MMDiT model [22] by approximately 66% while maintaining comparable performance. We provide three decoder configurations with the parameter size of 0.9B, 1.75B, and 3.52B, supporting a range of output canvas resolutions from 256 to 2048 pixels.

Inference Pipeline. The inference pipeline for both understanding and generation tasks is shown in Fig. 3 (right). For understanding tasks, Manzano uses the hybrid image tokenizer to extract continuous features. These features, along with text features, are then fed into the unified LLM decoder to predict the final answer. For generation tasks, Manzano takes a text input and predicts a sequence of image tokens. The image decoder then renders these tokens into image pixels.

### 4 Training

###### 4.1 Data

Our training data mixture includes text-only, image understanding, and generation data, divided into pretraining, continued pre-training, and supervised fine-tuning (SFT) stages. We leverage high-quality text-only data [105] for both pre-training and SFT to maintain the language modeling capability of Manzano model.

###### 4.1.1 Pre-training & Continued Pre-training

Understanding. We use two types of image understanding data: captioning (paired images and text descriptions), and interleaved image-text data. For captioning, we use a combination of sources with 2.3B image-text pairs, including CC3M [77], CC12M [6], COYO [4], VeCap [39], and in-house licensed data. This

data undergoes a filtering and re-captioning process to ensure high quality. For interleaved data, we use 1.7B documents from [40] and web-crawled interleaved data, similar to MM1 [61] and MM1.5 [102].

In the continued pre-training stage, we further train on 24M high-quality capability-oriented data, including documentation, charts, multilingual OCR, knowledge & reasoning, high-quality synthetic captions data, all with image splitting [25, 46, 102] enabled.

Generation. The image generation pre-training data consists of 1B in-house text-to-image pairs. Following [7], we generate synthetic captions using different captioner models. For the continued pre-training stage, we select a high-quality subset of licensed images and re-caption them with a more powerful MLLM, generating descriptions of lengths varying from 20 to 128 tokens.

###### 4.1.2 Supervised Fine-tuning

Understanding. Following MM1.5 [102], our final understanding SFT recipe comprises 75% image–text data and 25% text-only data. The image–text portion is further composed of approximately 30% general knowledge data, 20% document and chart understanding data, and 25% vision chain-of-thought (CoT) and in-house generated reasoning data.

Generation. Our text-to-image SFT data includes a curated blend of real and synthetic data. We begin with real-world text-image pairs from the DreamO dataset [63]. However, we observed that training solely on this dataset, while sufficient for standard diffusion-based generators, caused our unified auto-regressive model to overfit. To mitigate this, we expand our training data with synthetic examples. First, we incorporated 90K text-image pairs from established datasets, including DALLE3-1M [21], BLIP-3o [9], and ShareGPT-

- 4o [16]. Second, to achieve a larger scale, we generated an additional 4M pairs by feeding prompts from JourneyDB [80] into an open-source standalone diffusion model, Flux.1-schnell [38].

###### 4.2 Training Recipes

###### 4.2.1 Hybrid tokenizer training

The hybrid image tokenizer aims to produce two types of tokens: continuous for understanding and discrete for generation, which are pre-aligned with the multimodal LLM semantic space.

We first pre-train the vision encoder (ViT) using CLIP [70]. Then we attach a pretrained small LLM decoder (300M) to the shared vision encoder through two parallel continuous and discrete adapters (See Fig. 3-Left). For each training sample, we randomly select one adapter and feed the corresponding embeddings to the LLM decoder, which is trained with next-token prediction. We unfreeze all parameters and train the model on a variety of understanding data domains, including general knowledge, reasoning, and text-rich tasks.

This process enhances the tokenizer’s understanding capability, encompassing both high-level semantic understanding and fine-grained spatial details. Meanwhile, the branches are also being aligned to the same space. We follow the pre-training, continued pre-training and SFT stages using the understanding and text-only data described in Sec. 4.1.

After training, we discard the small LLM decoder and retain the resulting hybrid image tokenizer, which is then used as a vision input module for the unified LLM and image decoder.

###### 4.2.2 Unified LLM Training

As shown in Fig. 4-Left, we freeze the parameters of both the vision encoder and the discrete adapter to maintain a fixed vocabulary of image tokens during training. We extend the LLM embedding table with 64K Image tokens following the same codebook size of FSQ layer in the tokenizer.

For image understanding, the image tokenizer extracts the continuous features from the input image and feeds them directly into the LLM with standard next-token loss on text targets. For image generation, the tokenizer uses its discrete adapter to convert input images into a sequence of discrete image token IDs that are mapped to image tokens via the extended LLM embedding table. The LLM then computes a cross-entropy

Understanding Tasks Generation Tasks General Knowledge Text-Rich GenEval DPG WISE

Tokenizer Paradigm

Pure-Discrete 63.3 62.2 62.3 77 80.9 35 Dual-Encoder 63.8 63.6 72.0 65 66.3 17

Hybrid Tokenizer 64.9 66.5 73.3 77 79.9 35

Table 1 Tokenizer strategy ablation. Tokenizers are evaluated with a 1B unified LLM model. The full list of evaluation tasks for understanding can be referred to Sec. 5.4.1. The hybrid tokenizer outperforms the other two tokenizer paradigms.

loss on these image tokens only. To balance the training of understanding and generation tasks, we set the weight ratio of text loss to image loss at 1:0.5.

We train the unified LLM in three stages. Pre-training and continued pre-training use a 40/40/20 mix of image understanding, image generation and text-only data as described in Sec. 4.1.1. We train our model with

- 1.6T tokens (0.8T tokens for the 30B model) during the pre-training and an additional 83B tokens during the continued pre-training. Similarly, SFT stage uses curated instruction data with a 41/45/14 mix ratio for understanding, generation, and text using datasets in Sec. 4.1.2.

###### 4.2.3 Image Decoder Training

Our image decoder is trained following a progressive resolution growing paradigm [7, 22]. We first pre-train the decoder at a resolution of 256x256 for 400K steps. Subsequently, the model is fine-tuned progressively on higher resolutions of 512, 1024, and 2048, with each stage trained for a shorter schedule of 100K steps. For each stage, only images with short sides larger than the target resolution were used for training.

### 5 Experiments

- 5.1 Evaluation We evaluate our models on image understanding and generation capabilities on popular benchmarks. Understanding. We adopt the following three categories of benchmarks for multimodal understanding.

- • General VQA: SeedBench [42], RealWorldQA [103], and MMBench [52].
- • Knowledge & Reasoning: AI2D [36], ScienceQA [55], MMMU [100], and MathVista [54].
- • Text-rich Document & Chart Understanding: ChartQA [58], TextVQA [78], DocVQA [60], InfoVQA [59], and OCRBench [53].

Generation. We use both automated and human evaluations.

- • Automated Evaluation: The automated benchmarks include GenEval [28] and DPGBench [34] for prompt following generation, and WISE [64] for World Knowledge-Informed generation.
- • Human Evaluation: We curate a comprehensive evaluation set comprising 800 challenging prompts, subsampled from established academic benchmarks [90, 98] and from widely used community evaluation platforms. The generated outputs are assessed by in-house human raters on three dimensions: structural integrity, instruction following, and aesthetic quality. For each dimension, raters assign one of three grades: major issues, minor issues, or no issues, and are quantized to scores afterwards. To mitigate bias, entity information is masked, and the sample order is randomized. Each sample is independently rated by three raters, and the final scores are obtained by averaging across raters to reduce variability.

- 5.2 Understanding-Generation Interplay

In this section, we study the task conflict along two axes: (i) tokenizer strategy (pure-discrete vs. dual-encoder vs. our hybrid); (ii) task mixing (unified vs. single-task). For simplicity, we skip the continued pre-training stage in the unified LLM training for these ablations.

Understanding Tasks

###### Generation Task

Understanding Task

###### Generation Task

90

95

85

90

|-0.8 -0.3<br><br>-0.8|
|---|
| |
| |
| |
| |

|+0.0 -1.6|
|---|
| |
|+0.0|
| |
| |

|-1.5|
|---|
|+0.4 -2.3|
| |
| |
| |

|+2.0<br><br>+0.2|
|---|
| |
|-1.0|
| |
| |

72

76

68

72

54

57

51

54

36

38

34

36

18

19

17

18

0

0

0

0

General Knowledge Text-rich

GenEval DPG WISE

General Knowledge Text-rich

GenEval DPG WISE

Understanding-only Unified model

Generation-only Unified model

Understanding-only Unified model

Generation-only Unified model

(a) 300M Model

(b) 3B Model

- Figure 5 Unified vs. Single-task study. Our unified model exhibits a slight regression compared with the understandingonly model on understanding task; however, this effect becomes negligible at the 3B scale, where the gap is less than 1.0. For generation, the unified model shows a decline on only one benchmark compared with the generation-only model. Tokenizer Strategy. We construct two baselines to compare our unified hybrid tokenizer strategy:

- • Pure-discrete. Prior works [5, 89, 95] train a quantized semantic vision tokenizer using various quantization techniques [62, 86] and then use an LLM to predict the next text and image tokens. To mimic these methods in our setting, we replace the understanding inputs for LLM with discrete features from our hybrid tokenizer, so the LLM uses the same discrete tokens for both understanding and generation. To isolate the effect of quantization on understanding, we use the same weights for the vision encoder and the discrete adapter from our hybrid tokenizer.
- • Dual-encoder. Another popular models [12, 18] uses a dual-encoder strategy to preserve detailed features by a semantic encoder for understanding and a VAE-style encoder for generation, effectively mitigating the degradation of understanding. We reproduce this baseline by replacing the discrete tokens from our hybrid tokenizer with those generated by an internal reproduction of MagViT-2 [99], an autoencoder-style tokenizer. This MagViT-2 tokenizer uses FSQ [62] with a 64K codebook and a spatial compression ratio of 8. For generation tasks, we resize images to 128x128 pixels instead of the original 256x256. This reduced the number of tokens per image to 256, which we found improved the model’s instruction-following capabilities on benchmarks.

- Table 1 shows the results on both image understanding and generation tasks. Our hybrid tokenizer paradigm shows the least task conflict and outperforms both pure-discrete and dual-encoder baselines on all tasks. Specifically, the pure-discrete baseline leads to a significant drop in understanding performance–especially on text-rich benchmarks, due to information loss from quantization. While the dual-encoder baseline mitigates some of this degradation, it still consistently underperforms our hybrid tokenizer on all understanding tasks– especially on knowledge benchmarks, which rely heavily on the LLM’s reasoning abilities. This suggests that the conflict between heterogeneous visual tokens resides within the LLM.

Unified vs. Single-task. To quantify the task conflict in our hybrid tokenizer paradigm, we compare our unified model with baselines trained exclusively for understanding or generation. For the understanding-only baseline, we remove all text-to-image data from both the pre-training and SFT stages. We reduce the training steps to ensure it is exposed to the same number of text and image understanding tokens as our unified model. Similarly, for the generation-only baseline, we remove the understanding data and keep only the text-only and text-to-image data, while also reducing the training steps. We conduct this ablation study with a 300M and a 3B LLM decoder. The results, plotted in Fig. 5a and 5b, show that the unified LLM trained with our hybrid tokenizer performs on par with the dedicated, single-task models on nearly all tasks, even at a compact size like 300M. This demonstrates that our unified hybrid tokenizer paradigm successfully unifies visual perception and generation without a performance trade-off.

###### 5.3 Model Scaling Behavior

Facilitated by the decoupled design of LLM Decoder and Image Decoder, we explore the model scaling behavior along two dimensions: LLM Decoder and Image Decoder. Similar to Sec. 5.2, we skip the continued pre-train stage in the unified LLM training for the scaling experiments.

[Figure 28]

[Figure 29]

(a) LLM Decoder Scaling (b) Image Decoder Scaling

- Figure 6 Model scaling behavior of Manzano. (a) Scaling the LLM decoder yields monotonic improvements across both understanding and generation benchmarks. (b) Scaling the image decoder enhances structural integrity while maintaining stable quantitative benchmarks. A drop in aesthetic quality is observed, which we leave for more in-depth study in future work.

Scaling LLM Decoder. We vary only the LLM Decoder size (300M, 1B, 3B, and 30B) while keeping the image decoder (0.9B), data mixtures, and training hyperparameters fixed1. Fig. 6a shows monotonic gains across all understanding (General / Knowledge / Text-Rich) and generation (GenEval / DPG / WISE) metrics as the LLM decoder scales. Compared to 300M, our 3B Manzano model improves significantly by +14.2 (General), +18.8 (Knowledge), +10.9 (Text-rich), +11.0 (GenEval), +1.48 (DPG), +12.0 (WISE). Further scaling to 30B yields smaller but consistent gains over 3B. Fig. 7 shows the qualitative examples for image generation. We can see that the generation capabilities, including instruction-following, text-rendering, and overall image quality, are improved consistently across different LLM scales. The results support the simple yet effective design for Manzano: LLM decoder captures high-level semantics, and scaling it benefits both understanding and generation.

Scaling Image Decoder. We evaluate the performance of image decoders of varying sizes built on top of a 3B LLM decoder. Figure 6b shows that, in human evaluations, structural integrity improves substantially (+9.9), while instruction following performance remains unchanged. A slight decrease is observed in aesthetic quality. For automatic generation benchmarks, performance on GenEval and DPGEval remains nearly identical, whereas WISE exhibits a modest improvement (+2.0).

Takeaways. Scaling the unified LLM backbone consistently improves both understanding and generation, with substantial gains on text-rich understanding tasks and on WISE for generation. Scaling the image decoder also enhances image quality, without negatively affecting understanding. We observed that performance on the GenEval and DPG benchmarks becomes saturated when the model becomes larger. This saturation motivates a re-examination of how emergent capabilities of unified models could be assessed, as existing benchmarks may capture only a limited portion of overall capability and can be boosted through targeted data tuning [50]. Meanwhile, we observe substantial improvements on world-knowledge generation tasks, and we hope these findings pave the way for new directions in future community research.

###### 5.4 Comparisons with Unified and Specialist models

In this section, we evaluate our Manzano model on various benchmarks for both image understanding and text-to-image generation. To comprehensively assess our model’s capabilities, we compare its performance against SOTA unified and specialist models (i.e., understanding-only and standalone generation models).

###### 5.4.1 Image Understanding

As mentioned in Sec. 5.1, we evaluate our model’s understanding capabilities from three perspectives: Knowledge & Reasoning, General Visual Question Answering, and Text-rich Document & Chart Understanding. The results, shown in Table 2, compare our model against other understanding-only models of a similar size.

1We pre-train 30B LLM Decoder on roughly half the tokens compared to other model sizes due to compute limits.

###### Prompt 300M 1B 3B 30B

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Scholarly elephant reading a newspaper with the headline 'elephants take over the world'.

This image is in black and white. There is a rusty looking brick building on the right of the image. There is a sign with two circles on it in front of a rail. The rail seems to be leaning to the side. In the distance there is a church.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Bananas arranged on a picnic table to form the message 'That s bananas!'

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

An artist in a lost city painting surreal, floating landscapes, surrounded by cool-toned mists.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

A vintage postage stamp showing a painting of the Golden Gate Bridge and the text 'California'.

- Figure 7 Qualitative generation results when scaling LLM decoder size. The generated image quality improves as the LLM decoder size increases. For example, in rows 1, 3, and 5, there is a clear trend toward better text rendering and creativity. In row 2, the scene configuration improves significantly with each increase in the LLM decoder’s scale. The 300M model generates an image with only the brick building and the church that are mentioned in the prompt, but as the model grows to 1B and 3B, it begins to include the sign with two circles. Furthermore, the 30B model generates an image that accurately depicts and integrates all the concepts mentioned in the prompt.

Despite being a unified model, our model achieves state-of-the-art performance on many understanding benchmarks, particularly on text-rich tasks.

Knowledge & Reasoning. At the 3B scale, our model outperforms all unified models within the 7B scale and achieves performance on par with or better than the best specialist models at the 3B size. At the 30B scale, our model ranks first on the ScienceQA, MMMU, and MathVista benchmarks and third on the AI2D benchmark, outperforming all other unified and specialist models in these categories. Notably, our model surpasses the proprietary models listed in the final three rows on ScienceQA and is competitive with the current state-of-the-art model on the AI2D benchmark.

General Visual Question Answering. For general visual question answering, our model generally outperforms other unified models, despite its smaller size. It also achieves competitive results with state-of-the-art specialist models at both scales.

Text-rich Document and Chart Understanding. On text-rich and chart understanding tasks, our model

General Benchmarks Knowledge Benchmarks Text-rich Benchmarks SEEDI RealWorldQA

Model

AI2D (test)

OCRBench (test) 3B Specialist Model

SQA (test)

MMMU (val)

MathV (testmini)

ChartQA (test)

TextVQA (val)

DocVQA (test)

InfoVQA (test)

MMBench (dev-en)

MiniCPM-V 2.0-3B [97] 67.1 55.8 69.6 62.9 80.7 38.2 38.7 59.8 74.1 71.9 37.6 60.5 VILA1.5-3B [45] 67.9 – 61.7 – 69.0 33.3 – – – – – – Gemini Nano-2 [26] – – – 51.0 – 32.6 30.6 51.9 65.9 74.3 54.5 – Bunny-4B [32] 72.5 – – – 78.3 41.4 – – – – – – BLIP-3-4B [96] 72.2 60.5 – – 88.3 41.1 39.6 – 71.0 – – – Phi-3-Vision-4B [1] 71.8 59.4 78.6 76.7 90.8 40.4 44.5 81.4 70.1 83.3 49.0 63.7 MM1.5-3B [102] 72.4 56.9 72.4 65.7 85.8 37.1 44.4 74.2 76.5 87.7 58.5 65.7 InternVL2.5-2B [13] – 60.1 77.2 74.9 – 43.6 51.3 79.2 74.3 88.7 60.9 80.4 InternVL2.5-4B [13] – 64.3 78.7 81.4 – 52.3 60.5 84.0 76.8 91.6 72.1 82.8 InternVL3.5-2B [88] – 62.0 – 78.7 – 59.0 71.8 80.7 76.5 89.4 70.8 83.6 InternVL3.5-4B [88] – 66.3 – 82.6 – 66.6 77.1 86.0 77.9 92.4 78.0 82.2 Qwen2.5VL-3B [3] – 65.4 76.4 81.6 – 53.1 62.3 84.0 79.3 93.9 77.1 79.7

30B Specialist Model

LLaVA-NeXT-34B [48] 75.9 – – – 81.8 51.1 46.5 – 69.5 – – – Cambrian-34B [84] 75.3 67.8 – 79.7 85.6 49.7 53.2 75.6 76.7 75.5 – 60.0 MM1-30B [61] 72.1 59.4 – 73.3 81.0 44.7 39.4 76.9 73.5 75.8 47.3 60.6 MM1.5-30B [102] 75.0 69.0 – 77.2 91.9 47.4 55.6 83.6 79.2 91.4 67.3 65.8 InternVL2.5-26B [13] – 74.5 – 86.4 – 51.8 67.7 87.2 82.4 94.0 79.8 85.2

Unified Model

Janus-Pro-1.5B [12] 68.3 – 75.5 67.0† – 36.3 36.5† 23.4† 41.9† 35.5† 14.1† 48.7† Harmon-1.5B [94] 67.1 – 65.5 – – 38.9 – – – – – – Blip-3o-4B [9] 73.8 60.4 78.6 – – 46.6 – – 78.0 – – – Emu3-8B [89] 68.2 57.4 58.5 70.0 – 31.6 47.6 – 64.7 76.3 – 68.7 Janus-Pro-7B [12] 72.1 – 79.2 68.1† – 41.0 42.5 25.8† 45.6† 40.8† 21.3† 59.0† X-Omni-7B [27] 74.1 62.6† 74.8 76.8† – 47.2† 54.1† 81.5† 77.4† 88.6 46.9† 70.4† Bagel-14B [18] 78.5† 72.8† 85.0† 89.2† – 55.3 73.1 78.5† 80.0† 88.1† 51.0† 73.3† Manzano-3B 74.3 65.1 78.1 82.2 92.9 51.4 69.8 88.2 80.1 93.5 75.0 85.7 Manzano-30B 76.0 70.1 83.4 86.0 96.2 57.8 73.3 89.0 84.4 94.3 81.9 86.3

GPT-4o [35] 77.1 75.4 – 84.6 90.7 69.2 61.3 85.7 – 92.8 – 73.6

- Gemini-1.5-Pro [73] – 64.1 82.8 79.1 85.7 60.6 57.7 87.2 78.7 93.1 81.0 75.4

- Gemini-2.5-Pro [15] – 78 86.3 89.5 88.4 81.7 82.7 83.3 76.8 94.0 84.3 86.2

###### Table 2 Comparison on general, knowledge, and text-rich benchmarks. GPT-4o, Gemini-1.5-Pro, and Gemini-

- 2.5-Pro numbers are from OpenVLM Leaderboard. † represents that the results were reproduced independently in our experiments and might differ from those reported in previous studies. Manzano demonstrate competitive understanding capabilities, especially on text-rich benchmarks.

achieves the best performance on four out of five benchmarks (ChartQA, TextVQA, DocVQA, and OCRBench) when compared to all other unified, specialist, and proprietary models. For the InfoVQA task, our model significantly outperforms its unified counterparts and achieves the best results among specialist models.

###### 5.4.2 Image Generation

We present the quantitative results for our model’s image generation capabilities, evaluating them on two benchmarks: GenEval [28] and WISE [64]. While both benchmarks assess how well models follow text instructions, WISE additionally evaluates semantic grounding through world-knowledge-informed attributes.

As shown in Table 3, our model achieves SOTA results among unified MLLMs on both GenEval and WISE. The 3B model can already perform competitively with or better than much larger unified models, and scaling to 30B further improves generation quality – most notably yielding a large gain on WISE, while maintaining strong GenEval performance. This confirms that our unified architecture and training recipe support strong instruction-following generation.

###### 5.4.3 Comparison with Unified Models

In addition to specialist models, we also compare against recent unified models such as Janus-Pro [12], XOmni [27], and Bagel [18], which aim to handle both understanding and generation within a single framework. Our Manzano model substantially outperforms these unified baselines across almost all understanding benchmarks. At a similar scale, our 3B model exceeds X-Omni and BAGEL on DocVQA, OCRBench, and SEEDBench while maintaining competitive performance on MathVista and ChartQA. Our 30B model further extends this lead, consistently surpassing all existing unified models across knowledge, general

An oil painting of a poppy field in the impressionist style.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

The snake wears a striped top and wears a dress.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

A dog is to the right of the cat.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Painting of a mammoth in black and white, in the style of ancient cave paintings.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

A gothic painting of a mountain landscape in acrylic on canvas.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Lunar base under a starry sky.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

##### Manzano Janus Pro Bagel GPT-4o Nano Banana

- Figure 8 Qualitative comparison with SOTA unified models. We compare our Manzano-30B model to the SOTA models through side-by-side comparison. The images generated by our model demonstrate strong capabilities in instruction following, aesthetics, and creativity, often with a photo-realistic quality.

GenEval Benchmark WISE Benchmark Single Two Counting Colors Position Color Attr. Overall Cultural Time Space Biology Physics Chemistry Overall Dedicated T2I Model

Model

SDXL-3.5B [69] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 0.43 0.48 0.47 0.44 0.45 0.27 0.43 DALL-E 3 [65] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 – – – – – – – SD3-Medium-2B [22] 0.99 0.94 0.72 0.89 0.33 0.60 0.74 0.43 0.50 0.52 0.41 0.53 0.33 0.45 PixArt-Alpha-0.6B [11] – – – – – – – 0.45 0.50 0.48 0.49 0.56 0.34 0.47 FLUX.1-dev-12B [38] 0.98 0.93 0.75 0.93 0.68 0.65 0.82 0.48 0.58 0.62 0.42 0.51 0.35 0.50

LLM & Diffusion Conjunction

MetaQuery-XL-7B [67] – – – – – – 0.80† 0.56 0.55 0.62 0.49 0.63 0.41 0.55 OmniGen2-7B [93] 1.00 0.95 0.64 0.88 0.55 0.76 0.80 – – – – – – – Qwen-Image-27B [92] 0.99 0.92 0.89 0.88 0.76 0.77 0.87 0.67 0.67 0.80 0.62 0.79 0.41 0.67

Unified Multimodal LLM

Harmon-1.5B [94] 0.99 0.86 0.66 0.85 0.74 0.48 0.76 0.38 0.48 0.52 0.37 0.44 0.29 0.41 Janus-Pro-7B [12] 0.99 0.89 0.59 0.90 0.79 0.66 0.80† 0.30 0.37 0.49 0.36 0.42 0.26 0.35 Bagel-14B-A7B [18] 0.99 0.94 0.81 0.88 0.64 0.63 0.82 0.44 0.55 0.68 0.44 0.60 0.39 0.52 X-Omni-7B [27] 0.98 0.95 0.75 0.91 0.71 0.68 0.83† – – – – – – – Manzano-3B 0.98 0.91 0.82 0.71 0.78 0.71 0.85 0.42 0.51 0.59 0.45 0.51 0.32 0.46 Manzano-30B 1.00 0.91 0.83 0.87 0.84 0.65 0.85 0.58 0.50 0.65 0.50 0.55 0.32 0.54

GPT-4o [35] 0.99 0.92 0.85 0.92 0.75 0.61 0.84 0.81 0.71 0.89 0.83 0.79 0.74 0.80

- Table 3 Comparison on GenEval and WISE benchmarks. † represents the evaluation that involves LLM rewriting. Manzano achieves competitive performance compared with other unified models.

VQA, and text-rich domains. This demonstrates that unification does not have to come at the cost of understanding capability. With careful architectural and training design, our model matches or surpasses the best specialist models while providing strong generative capability. We provide more qualitative comparison to the state-of-the-art unified models in Fig. 8.

###### 5.5 Capability Extension to Image Editing

Image editing represents both a crucial application and a natural extension of text-to-image generation. Despite Manzano demonstrating strong multimodal modeling capabilities, particularly on text-rich understanding benchmarks, achieving pixel-level precision in fine-grained image editing remains challenging. Similarly, recent work within the decoupled LLM–diffusion paradigm [93] reports difficulties when relying solely on the LLM for precise editing, since the LLM lacks native mechanisms for direct pixel-level control.

We adopt an approach similar to [93] by providing the reference image simultaneously to both the LLM and the diffusion decoder. In this formulation, the LLM is responsible for diverse instruction following and maintaining semantic coherence, while the diffusion decoder enforces precise pixel-level control. By jointly conditioning on the reference image, Manzano enables accurate semantic instruction following while preserving fine-grained visual consistency. In Fig. 9, Manzano demonstrates versatile editing capabilities, including instruction-guided editing, style transfer, inpainting, outpainting, and depth estimation.

### 6 Conclusion

We introduced Manzano, an MLLM that combines visual understanding and image generation through a hybrid image tokenizer and a unified autoregressive backbone. The LLM predicts high-level semantics in the form of text and image tokens, while a lightweight diffusion-based image decoder renders final pixels from the generated image tokens. Coupled with a streamlined three-stage training recipe, this architecture delivers: (i) state-of-the-art on understanding tasks, (ii) substantial gains on generation among unified models, and (iii) minimal task interference as validated by interplay and scaling ablations. Beyond generation, Manzano naturally supports image editing by conditioning both the LLM and image decoder on a reference image, enabling instruction-following with pixel-level control.

Looking forward, we believe the same “hybrid tokenizer + unified AR backbone + image decoder” recipe may provide even stronger unification benefits, and we are eager to explore conversational editing, reasoning, and unification with more capabilities and more modalities in our future works. Collectively, our findings suggest that unification need not sacrifice accuracy for creativity — with clean objectives and better visual representations, a simple and scalable model can achieve both, and achieve them well.

[Figure 80]

- Figure 9 Editing capabilities of Manzano. (a) instruction-guided editing, (b) style transfer across diverse visual domains, and (c) extended editing tasks including inpainting, outpainting, and depth-estimation. Manzano achieves pixel-level controls across these five editing tasks.

### Acknowledgement

We’d like to thank Aleksei Timofeev, Brian Feng, Chen Zhang, David Haldimann, Forrest Huang, Jeff Lai, Jimmy Hu, Juan Lao Tebar, Junting Pan, Keen You, Kavya Nerella, Ke Ye, Manjot Bilkhu, Marcin Eichner, Nina Wenzel, Peter (Zhe) Fu, Peter Grasch, Qibin Chen, Shiyu Li, Tom Gunter, Vasileios Saveris, Wentao Wu, Xiujun Li, Yihao Qian, Yiran Fei, Zizhen Wang for their contributions to the Manzano project.

### References

- [1] Abdin, M., Jacobs, S. A., Awan, A. A., Aneja, J., Awadallah, A., Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H., et al. (2024). Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.
- [2] Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. (2022). Flamingo: a visual language model for few-shot learning. NeurIPS.
- [3] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. (2025). Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923.
- [4] Byeon, M., Park, B., Kim, H., Lee, S., Baek, W., and Kim, S. (2022). Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset.
- [5] Chameleon, T. (2024). Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818.
- [6] Changpinyo, S., Sharma, P., Ding, N., and Soricut, R. (2021). Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568.
- [7] Chen, C., Qian, R., Hu, W., Fu, T.-J., Tong, J., Wang, X., Li, L., Zhang, B., Schwing, A., Liu, W., et al. (2025a). Dit-air: Revisiting the efficiency of diffusion model architecture design in text to image generation. arXiv preprint arXiv:2503.10618.
- [8] Chen, J., Jincheng, Y., Chongjian, G., Yao, L., Xie, E., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. (2024a). Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations.
- [9] Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., Xue, L., Xiong, C.,

- and Xu, R. (2025b). Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset.

[10] Chen, J., Xu, Z., Pan, X., Yang, S., Qin, C., Yan, A., Zhou, H., Chen, Z., Zhou, T., Savarese, S., Xue, L., Xiong, C.,

- and Xu, R. (2025c). Blip3o-next: A next-generation multimodal foundation model.

- [11] Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. (2023). Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis.
- [12] Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. (2025d). Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811.
- [13] Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al. (2024b). Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271.
- [14] Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. (2024c). Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR.
- [15] Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. (2025). Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.
- [16] Cui, E., He, Y., Ma, Z., Chen, Z., Tian, H., Wang, W., Li, K., Wang, Y., Wang, W., Zhu, X., Lu, L., Lu, T., Wang, Y., Wang, L., Qiao, Y., and Dai, J. (2024). Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o.
- [17] DeepMind, G. (2025). Create and edit images with gemini. Google DeepMind. https://deepmind.google/ models/gemini/image/.
- [18] Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. (2025). Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683.
- [19] Dhariwal, P. and Nichol, A. (2021). Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794.
- [20] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

- [21] Egan, B., Redden, A., XWAVE, and SilentAntagonist (2024). Dalle3 1 Million+ High Quality Captions.
- [22] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.

(2024). Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

- [23] Fan, L., Tang, L., Qin, S., Li, T., Yang, X., Qiao, S., Steiner, A., Sun, C., Li, Y., Zhu, T., et al. (2025). Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436.
- [24] Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39.
- [25] Gao, P., Zhang, R., Liu, C., Qiu, L., Huang, S., Lin, W., Zhao, S., Geng, S., Lin, Z., Jin, P., et al. (2024). Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935.
- [26] Gemini, T., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. (2023). Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.
- [27] Geng, Z., Wang, Y., Ma, Y., Li, C., Rao, Y., Gu, S., Zhong, Z., Lu, Q., Hu, H., Zhang, X., et al. (2025). Xomni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058.
- [28] Ghosh, D., Hajishirzi, H., and Schmidt, L. (2023). Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152.
- [29] Gunter, T., Wang, Z., Wang, C., Pang, R., Narayanan, A., Zhang, A., Zhang, B., Chen, C., Chiu, C.-C., Qiu, D., et al.

(2024). Apple intelligence foundation language models. arXiv preprint arXiv:2407.21075.

- [30] Han, J., Chen, H., Zhao, Y., Wang, H., Zhao, Q., Yang, Z., He, H., Yue, X., and Jiang, L. (2025a). Vision as a dialect: Unifying visual understanding and generation via text-aligned representations. arXiv preprint arXiv:2506.18898.
- [31] Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., and Liu, X. (2025b). Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15733–15744.
- [32] He, M., Liu, Y., Wu, B., Yuan, J., Wang, Y., Huang, T., and Zhao, B. (2024). Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530.
- [33] Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851.
- [34] Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. (2024). Ella: Equip diffusion models with llm for enhanced semantic alignment. CoRR.
- [35] Islam, R. and Moushi, O. M. (2024). Gpt-4o: The cutting-edge advancement in multimodal llm. Authorea Preprints.
- [36] Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., and Farhadi, A. (2016). A diagram is worth a dozen images. In ECCV.
- [37] Kingma, D. P. and Welling, M. (2022). Auto-encoding variational bayes.
- [38] Labs, B. F. (2024). Flux. https://github.com/black-forest-labs/flux.
- [39] Lai, Z., Zhang, H., Zhang, B., Wu, W., Bai, H., Timofeev, A., Du, X., Gan, Z., Shan, J., Chuah, C.-N., et al. (2024). Veclip: Improving clip training via visual-enriched captions. In European Conference on Computer Vision, pages 111–127. Springer.
- [40] Laurençon, H., Saulnier, L., Tronchon, L., Bekman, S., Singh, A., Lozhkov, A., Wang, T., Karamcheti, S., Rush, A., Kiela, D., et al. (2024). Obelics: An open web-scale filtered dataset of interleaved image-text documents. NeurIPS.
- [41] Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. (2021). {GS}hard: Scaling giant models with conditional computation and automatic sharding. In ICLR.
- [42] Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y. (2023a). Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125.
- [43] Li, J., Li, D., Savarese, S., and Hoi, S. (2023b). Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML.

- [44] Liang, W., Yu, L., Luo, L., Iyer, S., Dong, N., Zhou, C., Ghosh, G., Lewis, M., Yih, W.-t., Zettlemoyer, L., et al.

(2024). Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996.

- [45] Lin, J., Yin, H., Ping, W., Molchanov, P., Shoeybi, M., and Han, S. (2024). Vila: On pre-training for visual language models. In CVPR.
- [46] Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., et al. (2023). Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575.
- [47] Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. (2022). Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.
- [48] Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. (2024a). Llava-next: Improved reasoning, ocr, and world knowledge.
- [49] Liu, H., Li, C., Wu, Q., and Lee, Y. J. (2023). Visual instruction tuning. arXiv preprint arXiv:2304.08485.
- [50] Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. (2025). Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470.
- [51] Liu, X., Gong, C., and Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003.
- [52] Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. (2024b). Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.
- [53] Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.-C., Liu, C.-L., Jin, L., and Bai, X. (2024c). Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102.
- [54] Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. (2023). Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.
- [55] Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. (2022). Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS.
- [56] Ma, C., Jiang, Y., Wu, J., Yang, J., Yu, X., Yuan, Z., Peng, B., and Qi, X. (2025). Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321.
- [57] Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., Vanden-Eijnden, E., and Xie, S. (2024). Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer.
- [58] Masry, A., Long, D. X., Tan, J. Q., Joty, S., and Hoque, E. (2022). Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244.
- [59] Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., and Jawahar, C. (2022). Infographicvqa. In WACV.
- [60] Mathew, M., Karatzas, D., and Jawahar, C. (2021). Docvqa: A dataset for vqa on document images. In WACV.
- [61] McKinzie, B., Gan, Z., Fauconnier, J.-P., Dodge, S., Zhang, B., Dufter, P., Shah, D., Du, X., Peng, F., Weers, F., et al.

(2024). Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611.

- [62] Mentzer, F., Minnen, D., Agustsson, E., and Tschannen, M. (2023). Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505.
- [63] Mou, C., Wu, Y., Wu, W., Guo, Z., Zhang, P., Cheng, Y., Luo, Y., Ding, F., Zhang, S., Li, X., et al. (2025). Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915.
- [64] Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Ning, K., Feng, C., Zhu, B., and Yuan, L. (2025). Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265.
- [65] OpenAI (2024). Dall·e 3. https://openai.com/index/dall-e-3/.
- [66] OpenAI (2025). Addendum to gpt-4o system card: 4o image generation. Accessed: April 2, 2025.
- [67] Pan, X., Shukla, S. N., Singh, A., Zhao, Z., Mishra, S. K., Wang, J., Xu, Z., Chen, J., Li, K., Juefei-Xu, F., et al. (2025). Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256.

- [68] Peebles, W. and Xie, S. (2023). Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205.
- [69] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. (2023). Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952.
- [70] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. (2021a). Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.
- [71] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. (2021b). Learning transferable visual models from natural language supervision.
- [72] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. (2022). Hierarchical text-conditional image generation with clip latents.
- [73] Reid, M., Savinov, N., Teplyashin, D., Lepikhin, D., Lillicrap, T., Alayrac, J.-b., Soricut, R., Lazaridou, A., Firat, O., Schrittwieser, J., et al. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.
- [74] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.
- [75] Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S. K. S., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. (2022). Photorealistic text-to-image diffusion models with deep language understanding.
- [76] Seed, T. (2025). Seed1.5-vl technical report. arXiv preprint arXiv:2505.07062.
- [77] Sharma, P., Ding, N., Goodman, S., and Soricut, R. (2018). Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565.
- [78] Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. (2019). Towards vqa models that can read. In CVPR.
- [79] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2020). Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456.
- [80] Sun, K., Pan, J., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., et al. (2023). Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems, 36:49659–49678.
- [81] team, Q. (2024). Qwen2-vl.
- [82] Tian, K., Jiang, Y., Yuan, Z., Peng, B., and Wang, L. (2024). Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865.
- [83] Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., and Bengio, Y. (2023). Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482.
- [84] Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, A., Fergus, R., LeCun, Y., and Xie, S. (2024a). Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860.
- [85] Tong, S., Fan, D., Zhu, J., Xiong, Y., Chen, X., Sinha, K., Rabbat, M., LeCun, Y., Xie, S., and Liu, Z. (2024b). Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164.
- [86] Van Den Oord, A., Vinyals, O., et al. (2017). Neural discrete representation learning. Advances in neural information processing systems, 30.
- [87] Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al. (2025a). Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.
- [88] Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al. (2025b). Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.
- [89] Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. (2024). Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.

- [90] Wiles, O., Zhang, C., Albuquerque, I., Kajic,´ I., Wang, S., Bugliarello, E., Onoe, Y., Papalampidi, P., Ktena, I., Knutsen, C., et al. (2025). Revisiting text-to-image evaluation with gecko: On metrics, prompts, and human ratings. ICLR.
- [91] Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al. (2025a). Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977.
- [92] Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., and Liu, Z. (2025b). Qwen-image technical report.
- [93] Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al. (2025c). Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871.
- [94] Wu, S., Zhang, W., Xu, L., Jin, S., Wu, Z., Tao, Q., Liu, W., Li, W., and Loy, C. C. (2025d). Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979.
- [95] Wu, Y., Zhang, Z., Chen, J., Tang, H., Li, D., Fang, Y., Zhu, L., Xie, E., Yin, H., Yi, L., et al. (2024). Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429.
- [96] Xue, L., Shu, M., Awadalla, A., Wang, J., Yan, A., Purushwalkam, S., Zhou, H., Prabhu, V., Dai, Y., Ryoo, M. S., et al.

(2024). xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872.

- [97] Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al. (2024). Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.
- [98] Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. (2022). Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789.
- [99] Yu, L., Lezama, J., Gundavarapu, N. B., Versari, L., Sohn, K., Minnen, D., Cheng, Y., Birodkar, V., Gupta, A., Gu, X., et al. (2023). Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737.
- [100] Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. (2023). Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502.
- [101] Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. (2023). Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986.
- [102] Zhang, H., Gao, M., Gan, Z., Dufter, P., Wenzel, N., Huang, F., Shah, D., Du, X., Zhang, B., Li, Y., et al. (2024a). Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning. arXiv preprint arXiv:2409.20566.
- [103] Zhang, Y.-F., Zhang, H., Tian, H., Fu, C., Zhang, S., Wu, J., Li, F., Wang, K., Wen, Q., Zhang, Z., et al. (2024b). Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257.
- [104] Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. (2024). Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039.
- [105] Zhou, H., Hornberger, E., Guo, P., Zhou, X., Wang, S., Wang, X., He, Y., Chang, X., Rauch, R., D’hauwe, L., et al.

(2025). Apple intelligence foundation language models: Tech report 2025. arXiv preprint arXiv:2507.13575.

- [106] Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al. (2025). Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

