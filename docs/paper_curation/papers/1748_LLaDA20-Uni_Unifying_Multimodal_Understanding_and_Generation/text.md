# arXiv:2604.20796v1[cs.CV]22Apr2026

[Figure 1]

[Figure 2]

[Figure 3]

## LLaDA2.0-Uni: Unifying Multimodal Understanding and Generation with Diffusion Large Language Model

Tiwei Bie, Haoxing Chen, Tieyuan Chen, Zhenglin Cheng, Long Cui, Kai Gan, Zhicheng Huang Zhenzhong Lan†, Haoquan Li, Jianguo Li†, Tao Lin†, Qi Qin, Hongjun Wang, Xiaomei Wang Haoyuan Wu, Yi Xin, Junbo Zhao AGI Research Center, Inclusion AI

##### Abstract

We present LLaDA2.0-Uni, a unified discrete diffusion large language model (dLLM) that supports multimodal understanding and generation within a natively integrated framework. Its architecture combines a fully semantic discrete tokenizer, a MoE-based dLLM backbone, and a diffusion decoder. By discretizing continuous visual inputs via SigLIP-VQ, the model enables block-level masked diffusion for both text and vision inputs within the backbone, while the decoder reconstructs visual tokens into high-fidelity images. Inference efficiency is enhanced beyond parallel decoding through prefix-aware optimizations in the backbone and few-step distillation in the decoder. Supported by carefully curated large-scale data and a tailored multi-stage training pipeline, LLaDA2.0-Uni matches specialized VLMs in multimodal understanding while delivering strong performance in image generation and editing. Its native support for interleaved generation and reasoning establishes a promising and scalable paradigm for next-generation unified foundation models.

[Figure 4]

[Figure 5]

GitHub https://github.com/inclusionAI/LLaDA2.0-Uni HuggingFace Model https://huggingface.co/inclusionAI/LLaDA2.0-Uni

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### LLaDA2.0-Uni LLaDA-O Lumina-DiMOO LLaDA-V InternVL-U BAGEL

Figure 1: Benchmark Performance of LLaDA2.0-Uni.

Authors are listed in alphabetical order based on last name. † indicates tech-leaders.

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

###### Figure 2: Showcases of LLaDA2.0-Uni in High-Fidelity Image Generation.

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

|[Figure 39]|
|---|

|[Figure 40]|[Figure 41]|[Figure 42]|[Figure 43]|[Figure 44]|
|---|---|---|---|---|

Multiple Reference Editing

In a modern living room, a young man in a black t-shirt and a backward black cap sits on a beige couch, looking down with a slight smile. Beside him, a young girl with blonde hair in a ponytail, wearing a light blue shirt, looks up at him with a curious expression.

Moving forward, the man turns his head back to face the camera, his smile wide and cheerful, while the girl beside him beams with a bright, toothy grin, her eyes sparkling with joy.

Next, the man turns his head to look directly at the camera with a warm, engaging smile, while the girl beside him shifts her gaze slightly, her expression becoming more neutral and observant.

[Figure 45]

[Figure 46]

[Figure 47]

Let's consider option A, moving the king to e3. This move moves the king backwards, potentially making it too slow to catch the advancing black pawn.

[Figure 48]

Finally, let's consider option D, moving the king to d5. This move moves the king towards the center, potentially placing it on a square too far to stop the promotion. Answer: B.

Next, let's consider option C, moving the king to e5. This move moves the king further into the center, potentially allowing the black pawn to promote safely.

Now, let's consider option B, moving the king to c4. This move moves the king directly towards the threat, effectively attacking the pawn and securing a draw.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Q: What is the best move for White to play?

A: Ke3 B: Kc4 C: Ke5 D: Kd5

###### Figure 3: Showcases of LLaDA2.0-Uni in Single/Multi-Reference Editing, Interleaved Generation and Reasoning. Note: For editing tasks, the original image is positioned on the left or top.

##### 1 Introduction

Large language models (LLMs) have evolved beyond text to handle a wide variety of multimodal tasks (Cui et al., 2025b; AI et al., 2025; Wu et al., 2025a; Gu et al., 2025; Liu et al., 2026). Understanding and generation represent the two primary categories of multimodal tasks. By modeling both visual understanding and generation as token sequence prediction, LLMs have achieved highly competitive results in both areas. Traditionally, these tasks are handled by separate specialized models, such as Qwen-VL (Qwen Team, 2025a;b) or InternVL (Chen et al., 2024d;c; Wang et al., 2025b) for understanding, and Flux (Black-forest-labs,

- 2024) or Z-Image (Cai et al., 2025) for generation. However, a unified model that handles both within a single framework offers several key benefits: it promotes mutual enhancement between understanding and generation, improves deployment efficiency, and unlocks advanced capabilities like interleaved generation and reasoning, ultimately bringing us closer to artificial general intelligence (AGI).

Current unified multimodal models predominantly build upon autoregressive (AR) architectures. Janus (Wu et al., 2025b) and Lumina-mGPT (Liu et al., 2026) tokenizes images into discrete sequences and unifies both modalities under next-token prediction, while OmniGen2 (Wu et al., 2025c), Hunyuan Image 3.0 (Cao et al.,

- 2025b), and BAGEL (Deng et al., 2025) adopt a hybrid paradigm combining text autoregression with image diffusion. While these AR-based approaches have shown promise, masked diffusion models (Lou et al., 2023; Sahoo et al., 2024; Xin et al., 2025c) offer an alternative paradigm with inherent advantages in parallel decoding and bidirectional context modeling. A unified masked diffusion framework further simplifies training through a single objective, avoiding the delicate balance between AR and diffusion losses. However, existing unified masked diffusion models, such as MMaDA (Yang et al., 2025) and Lumina-DiMOO (Xin et al., 2025a), still lag behind state-of-the-art AR-based unified architectures in both task coverage and benchmark performance. This gap fundamentally stems from their architecture and modeling designs: 1) their reconstructive VQ tokenizers lack semantic information, causing poor understanding performance; 2) excessive image compression by VQ tokenizers compromises generation quality; 3) their fully bidirectional modeling has been shown to be unreliable for text. Furthermore, they commonly assume fixed output lengths for understanding tasks, limiting their applicability in open-ended scenarios.

To overcome these limitations, we propose LLaDA2.0-Uni, a unified dLLM-based Mixture-of-Experts (MoE) model for seamless multimodal understanding and generation. At its core, LLaDA2.0-Uni utilizes LLaDA2.0 (Bie et al., 2025) (a 16B dLLM MoE architecture) as its backbone. A key architectural innovation is the introduction of the SigLIP-VQ tokenizer, which converts continuous visual inputs into fully discrete semantic tokens. Unlike previous reconstruction-based tokenizers that struggle with multimodal understanding, this purely semantic representation preserves crucial details and effectively supports complex visual reasoning. Consequently, this design maintains the unified discrete modeling format, allowing both text and images to be optimized under a shared block-level masked diffusion objective. For image generation, LLaDA2.0-Uni employs a dedicated Diffusion Decoder to process the discrete tokens generated by the dLLM backbone. Optimized through distillation, this decoder synthesizes high-fidelity images in just 8 inference steps, achieving an excellent balance between speed and quality.

LLaDA2.0-Uni achieves top-tier performance across both understanding and generation benchmarks, as shown in Figure 1. In multimodal understanding, LLaDA2.0-Uni demonstrates competitive visual question answering and document reasoning capabilities compared with specialized VLMs such as Qwen2.5-VL (Qwen Team, 2025a) Regarding image generation, LLaDA2.0-Uni produces high-quality images and enables highly flexible image editing. Beyond these general tasks, the unified discrete representation empowers LLaDA2.0Uni to support interleaved generation and reasoning. This flexibility establishes LLaDA2.0-Uni as a powerful and efficient paradigm for the next generation of unified foundation models.

The key contributions of LLaDA2.0-Uni can be summarized as follows:

- • Novel Unified Architecture. LLaDA2.0-Uni integrates a fully semantic tokenizer, a 16B MoE dLLM backbone, and a diffusion decoder. This architecture unifies text and image modeling through a shared block-wise mask prediction objective.
- • Interleaved Generation and Reasoning. Beyond its strong performance in both understanding and generation, LLaDA2.0-Uni inherently supports interleaved generation and reasoning, marking a significant step toward exploring how generation and understanding can reinforce each other.
- • Efficient Inference. Building on the advantages of parallel decoding, LLaDA2.0-Uni further accelerates inference by optimizing the decoding process in the dLLM backbone and applying few-step distillation to the decoder, achieving an effective balance between speed and performance.

[Figure 54]

[Figure 55]

Text Token Image Token Special Token Mask Token

Answer: Based on the chart, the South region (green) recorded the highest rainfall in any single month. This peak occurred in Month 5, reaching nearly 200 mm, which is the highest individual point across all regions and months shown.

MultimodalDecoder Text De-Tokenizer Diffusion Decoder

|<w>|
|---|

|<h>|
|---|

|<w>|
|---|

|<h>|
|---|

|<w>|
|---|

|<w>|
|---|

|<h>|
|---|

|<h>|
|---|

[Figure 56]

LLaDA2.0-mini

###### (Diffusion LLM & Modality-Agnostic MoE)

|<w>|
|---|

|<h>|
|---|

|<w>|
|---|

|<h>|
|---|

|<w>|
|---|

|<w>|
|---|

|<h>|
|---|

|<h>|
|---|

Multimodal Tokenizer

Text Tokenizer SigLIP-VQ

Image Editing Multimodal Understanding

Image Generation Text Prompt: A small anthropomorphic teapot stands beside a huge teacup, saying, “I think I need a little more tea,” in a Pixar-style animation.

[Figure 57]

[Figure 58]

Instruction: Change the text on the blackboard to “LLaDA Coffee”.

Question: Which region had the highest rainfall in any single month?

Multimodal Input

Height: 768

Height: 576

Width: 992

Width: 1024

- Figure 4: Architecture Overview of LLaDA2.0-Uni. The framework integrates a SigLIP-VQ tokenizer and a language model decoder to process multimodal inputs, including text, image, and video.

• Strong Benchmark Performance. LLaDA2.0-Uni achieves strong performance across visual understanding, generation, and editing benchmarks, performing on par with state-of-the-art unified models.

##### 2 Model Design

###### 2.1 Motivation and Design Principles

We aim to extend the dLLM architecture into a unified model for multimodal understanding and generation, leveraging its advantages in parallel decoding and bidirectional context modeling. Prior approaches to this goal exhibit notable limitations. MMaDA (Yang et al., 2025) and Lumina-DiMOO (Xin et al., 2025a) rely on reconstructed visual tokens from VQ-VAE, which degrades understanding performance and yields sub-optimal visual quality. LLaDA-o (You et al., 2026) and BAGEL (Deng et al., 2025) adopt decoupled visual modules (ViT for understanding, VAE for generation), introducing a modeling gap and divergent optimization objectives within the same model. To overcome these limitations, we design LLaDA2.0-Uni around a key principle: use fully discrete semantic tokens for both understanding and generation. This unified representation eliminates the need for heterogeneous encoders and enables end-to-end training under a single mask prediction objective, as illustrated in Figure 4.

###### 2.2 Architecture

LLaDA2.0-Uni consists of three core components: (1) a SigLIP-VQ tokenizer that converts images into discrete semantic tokens, (2) a 16B MoE diffusion language model that processes both text and visual tokens under a unified mask prediction objective, and (3) a diffusion decoder that reconstructs semantic tokens into high-fidelity images. This design enables end-to-end training and inference for both understanding and generation tasks within a single coherent framework.

###### 2.2.1 Semantic Discrete Tokenizer

The tokenizer adopts a SigLIP-VQ architecture building upon X-Omni (Geng et al., 2025) to convert continuous images into discrete tokens. Unlike standard VQ-VAEs (Esser et al., 2021; Wang et al., 2024b) that rely on pixel-level reconstruction, SigLIP-VQ is trained directly on understanding tasks, thereby preserving rich semantic information. Consequently, it demonstrates a clear advantage over reconstruction-based VQ-VAEs in multimodal understanding tasks. Specifically, the tokenizer utilizes a pre-trained SigLIP2-g ViT (Tschannen et al., 2025) as the visual feature extractor and supports dynamic resolution processing. Following the ViT encoder, a vector quantizer aligns the visual representations with a pre-trained large language model, featuring a codebook with a vocabulary size of 16,384 and a dimensionality of 2,048. While SigLIP-VQ excels in semantic extraction, it lacks a native mechanism to reconstruct images from these discrete tokens. We address this by designing a custom diffusion decoder, detailed in Section 2.2.3.

###### 2.2.2 Diffusion Large Language Model

MoE Backbone for Multi-Modal Capacity. A modality-agnostic Mixture-of-Experts (MoE) architecture enables language backbones to serve as universal multi-task learners, dynamically allocating capacity across modalities without the need for modality-specific designs. We adopt LLaDA-2.0-mini (Bie et al., 2025) as our dLLM backbone, an MoE architecture with 16B total parameters. To integrate visual information, we expand the original dLLM vocabulary by appending tokens from the SigLIP-VQ codebook, along with a set of custom special tokens for image generation and understanding. In the input embedding layer, we retain the pre-trained language embeddings while randomly initializing the new visual token embeddings. Similarly, the final prediction head is expanded to accommodate the enlarged vocabulary, with the language-specific portion initialized from pre-trained weights to preserve linguistic proficiency.

Block-wise Attention for Training Stability. For dLLMs, full bidirectional attention is theoretically ideal for parallel sampling. However, prior studies (Nie et al., 2025; Yang et al., 2025; Xin et al., 2025a) show that unconstrained full attention often degrades performance. We adopt a block-wise attention scheme (Arriola et al., 2025) to balance quality and efficiency. This design is particularly important for SigLIP-VQ tokens: since they are semantically aligned with Qwen2.5, they inherit an autoregressive bias that would be disrupted by pure full-attention. By constraining attention within predefined blocks and selectively enabling it across blocks, we maintain parallel decoding speed while achieving strong performance in both language and visual tasks.

Positional Embedding & Arbitrary Resolution. Rotary Position Embedding (RoPE) (Su et al., 2021) is a standard choice in LLMs due to its flexibility and scalability. While many recent unified models adopt 2D RoPE for images (Cao et al., 2025b; Deng et al., 2025), we keep the original 1D RoPE structure for simplicity. To represent 2D spatial information, we add special <height> and <width> tokens (e.g., <imgsize 512>) before the flattened 1D visual sequence. Previous studies (Liu et al., 2026; Xin et al., 2025b; Geng et al., 2025) confirm that this simple approach is highly effective. These size tokens also enable the model to handle arbitrary image resolutions without architectural changes.

###### 2.2.3 Diffusion Decoder

Semantic VQ requires a specialized decoder to map features from the semantic space back to the image space, unlike traditional reconstruction-based VQ that can directly use a pixel decoder. We introduce a diffusion model built upon Z-Image-Base (Cai et al., 2025), a 6B pre-trained text-to-image model. Once the dLLM generates image tokens, they serve as the conditioning signal, replacing conventional text prompts. This differs from existing methods like NextFlow (Zhang et al., 2026b) and X-Omni (Geng et al., 2025), which redundantly combine text prompts with visual tokens. Beyond basic decoding, our diffusion model performs

- 2× super-resolution, using upsampled semantic tokens as the sole conditioning input. To address the computational cost of 50-step sampling with CFG, we employ model distillation to achieve 8-step CFG-free inference (Section 4.4). Together, the SigLIP-VQ tokenizer, dLLM backbone, and diffusion decoder form a unified pipeline where understanding and generation share the same discrete token representation.

###### 2.3 Training-free Inference Acceleration

Block-wise discrete diffusion language models require B × T forward passes to generate B blocks with T denoising steps each. Uniform KV cache eviction and fixed-schedule step reduction degrade quality in multimodal settings due to heterogeneous per-token difficulty and differing information density across modalities. We propose SPRINT (Sparse Prefix Retention with Inference-time Non-uniform Token Unmasking), a training-free framework that reduces cost along two orthogonal axes. Sparse Prefix Retention prunes the prefix KV cache in a modality-aware manner to lower per-step cost. Non-uniform Token Unmasking replaces

the fixed denoising schedule with confidence-adaptive unmasking to reduce the step count. Together the two components achieve up to 1.6× speedup with negligible quality loss (Section 5.5.1).

Sparse Prefix Retention. Each denoising step attends to the full prefix, whose quadratic attention cost dominates as the generated sequence lengthens. SPRINT constructs a pruned prefix KV cache once per block, so that all subsequent steps attend to a much shorter effective sequence.

The first step of each block performs a full forward pass to obtain logits and a complete KV cache. Each prefix position i is then scored by a composite importance measure that blends the key-norm importance I¯i, reflecting how strongly a position influences the attention distribution, with the token confidence ci, capturing the model’s prediction certainty at that position:

si = α · I¯i + (1 − α) · ci , (1)

where I¯i = ∥ki∥2 L 1 ∑Lj=1 ∥kj∥2 is the mean-normalized key norm, ci = maxv pθ(v | xt) is the top-1 softmax confidence, and α = 0.5.

The pruning is modality-aware: we maintain separate keep ratios rtext and rimg rather than a single uniform ratio, because image tokens exhibit high spatial redundancy and tolerate aggressive pruning, whereas text tokens carrying instructions or reasoning chains do not. Within each modality, the top-⌊r · n⌋ positions by si are retained and the rest are evicted. We explore two settings: selective image pruning with rtext = 1.0,rimg = 0.8, and full prefix retention with rtext = rimg = 1.0. The global keep ratio is r = 0.5 in both cases.

Non-uniform Token Unmasking. The standard denoising schedule unmasks a fixed ⌈m/T⌉ tokens per regardless of prediction certainty, wasting computation on confident predictions and under-allocating it to uncertain ones. SPRINT replaces this schedule with a confidence-adaptive strategy. For all m masked positions the model computes per-position confidence cn = pθ(xˆ0n | xt) and accepts every position whose confidence exceeds a threshold τ in a single step.

A = n ∈ [m] : cn > τ . (2)

A minimum of ⌈m/(T − t)⌉ acceptances is enforced at each step to guarantee termination. We examine τ ∈ {0.93,0.95}.

##### 3 Data Preparation

###### 3.1 Multimodal Understanding

Pretrain Data Source & Processing. In the pre-training stage, the model learns to perceive images through text supervision. We collect extensive image-captioning data from open-source datasets (An et al., 2025; Zhang et al., 2026a), supplemented by specialized categories:

- • OCR Data. We develop a coarse-to-fine pipeline to produce millions of samples. By combining PaddleOCR (Cui et al., 2025a) pseudo-labels with refinements from Qwen3-VL, we achieve high-quality document understanding data without manual annotation.
- • Grounding & Counting Data. Using Objects365 (Shao et al., 2019) and RefCOCO (Yu et al., 2016; Kuznetsova et al., 2020; Kazemzadeh et al., 2014), we refine spatial data through detection confidence filtering and Qwen3-VL-235B-A22B (Qwen Team, 2025b) verification. Coordinates are normalized to [0, 1000] for stability, and counting data is automatically derived from verified bounding boxes.
- • World Knowledge & Reasoning. We curate data across three domains: general world knowledge, logical reasoning, and mathematics.
- • Text Data. High-quality text-only data is sourced from Ling2.0 (Ling Team, 2025) and LLaDA2.0 (Bie et al., 2025), covering general knowledge, code, and mathematics.

SFT Data Source & Processing. Our SFT dataset contains approximately 60 million samples with a 1:5 ratio of text-only to multimodal data. This collection covers single/multi-turn dialogues and single/multi-image scenarios across various tasks, including General VQA, Chart/Table QA, mathematical reasoning, etc. We implement a two-stage filtering pipeline for quality control: (1) Query Filtering: Qwen3-VL audits the input space, pruning vague or low-information instructions while rewriting ambiguous queries to enhance semantic clarity. (2) Response Filtering: Rule-based heuristics rectify structural artifacts, and GPT-OSS (Agarwal et al., 2025) filters semantic biases while ensuring alignment with ground-truth references.

- 3.2 Image Generation

Data Source. We collect over 200 million web images with their original text descriptions. To improve performance on challenging generation tasks, we specifically increased the proportion of images featuring human body and rendered text. Since image generation requires higher visual quality than understanding, all data undergoes rigorous filtering.

Filtering Pipeline. We apply a three-stage cleaning process: (1) Metadata filtering removes low-resolution images (less than 512 pixels on the shortest side) and highly compressed images ((Height×Width)/Filesize < 0.15). (2) Aesthetics filtering discards images with ArtiMuse (Cao et al., 2025a) scores below 60. (3) Quality filtering eliminates images with DeQA-Score (You et al., 2025b) under 4.0. After filtering, 140 million high-quality images are retained.

Image Captioning. We generate captions using Qwen3-VL-235B-22B. To retain real-world knowledge, the VLM evaluates the original web text: if informative, it incorporates this information into the caption (e.g., using “Corgi Dog” instead of a generic “dog”), producing richer and more accurate descriptions.

- 3.3 Image Editing

Data Source. Our image editing data combines open-source datasets and synthesized pairs. We incorporate X2Edit (Ma et al., 2026), OmniEdit (Wei et al., 2024), Nano-consistent-150k (Ye et al., 2025a), Pico-Banana (Qian et al., 2025), UniWorld (Lin et al., 2025), StructVisuals (Zhuo et al., 2025), UnicEdit (Ye et al., 2025b), and CrispEdit (Chow et al., 2025). We also synthesize high-fidelity editing pairs by processing images from our generation dataset through an automated pipeline, further expanding data diversity while ensuring consistency between generation and editing tasks.

Instruction Refinement. We use Qwen3-VL-235B-22B for quality control. First, we filter out “failed” samples where editing produces no observable change or introduces visual artifacts. Second, for high-quality transformations with inaccurate or vague instructions, the VLM rewrites instructions based on actual visual changes. This ensures both visual integrity and precise instruction-image alignment.

- 3.4 Interleaved Data

Data Source & Filtering. We construct interleaved image-text data from the Koala36M (Wang et al., 2025a) video corpus through strict filtering: (1) Duration filtering discards clips longer than 30 seconds or shorter than 10 seconds to minimize fragmentation errors. (2) Quality filtering retains the top 50% by aesthetic score (> 4.0) and clarity (> 0.7). (3) Motion filtering requires a motion score greater than 4 to avoid degenerate solutions where the model generates static images. This pipeline removes approximately 75% of raw data, yielding 6M refined clips free from blur, static scenes, and low-aesthetic content. We sample frames every 5 seconds, producing interleaved sequences of 2–6 frames.

Interleaved Captioning. We use Qwen3-VL-235B-A22B to generate detailed descriptions of actions and scene changes from frame sequences. Additionally, we generate user instructions tailored to these sequences, providing high-quality instruction-following data for SFT.

- 3.5 Reasoning-Augmented Data

To equip LLaDA2.0-Uni with reasoning capabilities, we incorporate a dedicated dataset comprising two components: reasoning-based image generation and interleaved reasoning. We source this data from Flux-

- 6M (Fang et al., 2025), Zebra-CoT (Li et al., 2026), and Weave (Chow et al., 2024), totaling approximately 8M samples for SFT. This data enables chain-of-thought reasoning before image generation and multi-step reasoning across interleaved image-text sequences.

##### 4 Model Training

###### 4.1 Training Recipe

Our training pipeline consists of three stages that progressively enhance model capabilities: foundational cross-modal alignment, multi-task pre-training, and supervised fine-tuning. Table 1 summarizes the data composition, token scale, and training configurations for each stage.

- Stage 0: Vision–Language Alignment. The primary objective of stage 0 (S0) is to align visual and linguistic representations within the dLLM backbone. We use high-quality image–caption pairs and visual knowledge datasets, supplemented with pure text data to preserve language capabilities. During training, a random masking strategy is applied to a subset of text and image tokens: for generation tasks, only image tokens are

Table 1: Overview of the Training Stages. This includes details on data composition, token scale, sequence length, and trainable components.

Stages&Objective S0: Vision-Language Alignment S1: Multi-task Pre-training S2: Supervised Fine-Tuning

Image Caption, Text, OCR, Grounding, Counting, Video Data, Multimodal VQA

High-quality Multimodal VQA High-quality Text QA Interleaved Reasoning

Understanding Data

Image Caption, Text

High-quality Image Generation Image Generation with CoT High-quality Image Editing High-quality Interleaved Generation Interleaved Reasoning

Text-to-image Image Editing Interleaved Generation

Generation Data

Text-to-image

Gen. Resolution 256 → 512 512 512 (diffusion decoder → 1024)

Under. Max Edge 800 800 800 Training Tokens 100B 210B 80B Sequence length 8192 8192 8192 → 16384

masked; for understanding tasks, only text tokens are masked. To handle long visual token sequences, we adopt a progressive arbitrary resolution scheme: generation starts at 256 × 256 (∼256 tokens) and transitions to 512 × 512 (∼1024 tokens), while understanding consistently uses 800 × 800 with arbitrary resolution (∼2048 tokens).

- Stage 1: Multi-task Pre-training. The model is trained on diverse multimodal data to develop comprehensive understanding and generation capabilities. Visual understanding data includes image–text interleaved data, OCR, and visual counting/grounding tasks. Generation data includes image editing, subject-driven generation, controllable generation, style transfer using reference images, and multi-view generation tasks. This stage strengthens cross-modal connections and enables the model to handle increasingly complex tasks.
- Stage 2: Supervised Fine-tuning. The Supervised Fine-Tuning (SFT) process is conducted in two stages: an initial phase at 8k context length for fundamental instruction-following capabilities, followed by expansion to 16k context for complex visual reasoning and generation.

###### 4.2 Pre-Training Optimization

We adopt the Block Diffusion Language Model (BDLM) (Arriola et al., 2025) training objective, which extends standard discrete diffusion by operating on block-level masked regions rather than individual tokens. This design enables parallel decoding while maintaining coherent context within each block, making it well-suited for the variable-length sequences common in multimodal tasks.

BDLM Loss. The training loss under the BDLM paradigm is defined as:

LB

K

α′t 1 − αt

1[xti,k = [MASK]] log pθ(x0,i k|x0,<k,xt,k) , (3)

#### ∑

#### ∑

LBDLM(θ) = −Et,x0,xt

i=1

k=1

where the expectation is over timestep t, the clean sequence x0, and its corrupted version xt (tokens masked with probability 1 − αt). The indicator 1[·] ensures predictions are made only for masked tokens, and −α′t/(1 − αt) is the diffusion-derived time weight. We define: K = Ltotal/LB as the number of blocks, LB as the block size, xti,k as the i-th token in block k, x0,<k as the preceding clean blocks, and xt,k as the noisy version of the current block.

Load Balancing Strategy. In MoE models, imbalanced expert utilization can lead to routing collapse. We adopt an auxiliary-loss-free load balancing mechanism (Liu et al., 2024a) that promotes differentiated expert specialization while encouraging uniform workload distribution. To improve numerical stability, we scale routing gate outputs by a factor of 2.5, stabilizing their root-mean-square (RMS) magnitude. The auxiliaryloss-free bias is updated according to (Su, 2025):

(Fi − Qi)

, (4)

bi = bi + u ×

1 n ∑nj=1 (Fj − Qj)2

where F = E(f) denotes the current expert load distribution induced by the bias b, and Q = [n1, n1, . . . , n1]

represents the ideal uniform distribution over n experts. This RMSNorm-style normalization smooths bias updates, leading to stable load balancing throughout training.

###### 4.3 Supervised Fine-Tuning Optimization

During SFT, we adopt the same load balancing strategy while introducing complementary masking and a mask token reweighting loss to handle variable-length sequences.

Mask Token Reweighting Loss. We adapt the BDLM objective to be conditional on an input prompt c:

LSFT(θ) = −Et,(c,x0),xt

LB

K

α′t 1 − αt

1[xti,k = [MASK]] log pθ(x0,i k|c,x0,<k,xt,k) . (5)

∑

∑

i=1

k=1

A key challenge in SFT is that sample lengths vary significantly—by up to two orders of magnitude. Naive token-averaged loss causes gradients to be dominated by long sequences, while sample-level averaging encourages brevity. We therefore propose a re-weighting mechanism to balance these extremes:

∑j βjLSFT(j) ∑j βj

1

, where βj =

. (6)

LMTRS =

∑kK=1 ∑iL=B1 1[xti,,(kj) = [MASK]]

The scaling factor βj is the inverse square root of the number of masked tokens in sample j, equilibrating gradient contributions across diverse response lengths.

Complementary Masking. Complementary masking (Li et al., 2025b) is a strategy to enhance data efficiency for dLLMs by constructing two antithetical training instances from a single sequence x0: a primary noised sequence xt and a complementary sequence xt′ using the inverse mask. This design ensures that every token position appears uncorrupted exactly once per pair, thereby doubling effective information utilization and eliminating token-level sampling bias. We adapt this strategy in our framework.

###### 4.4 Diffusion Decoder Training

Training Paradigm. We optimize the diffusion decoder via the standard flow matching objective (Lipman et al., 2022). The overall training trajectory is decoupled into a preliminary warm-up followed by a progressive two-stage fine-tuning scheme. The flow matching loss is formulated as:

LFM(θ) = Ex0,x1,z,t ∥vθ,t(xt,z) − vt∥22 , (7)

where z represents the conditioned semantic visual tokens, vθ,t denotes the velocity field predicted by the network at timestep t, and vt is the target velocity. The overall training process is divided into three stages:

- • Stage 1: Warm-up. We freeze the semantic processor and update only the remaining modules to establish cross-modal alignment while preserving pre-trained priors.
- • Stage 2: Multi-domain Generalization. Following the warm-up, we unfreeze all parameters and fine-tune on diverse domains for robust generalization.
- • Stage 3: High-fidelity Refinement. In the final stage, we refine on high-quality data to elevate aesthetic fidelity and fine-grained visual details.

Few-step Generation. To accelerate visual generation, we adopt a lightweight consistency-based distillation framework (Sun et al., 2026) for the diffusion decoder. This method requires only an auxiliary projection layer—a final additional layer added to the decoder backbone—which will be discarded at inference time. The distillation objective combines flow matching with a consistency term:

LDistill(θ) = Ex0,z,t ∥vθ,t − vt∥22 + ∥uθ,t − vt + t ·

duθ−,t

dt ∥22 ,where uθ−,t = stop grad(uθ,t), (8)

where vt is the target velocity, vθ,t,uθ,t are the dual outputs of the diffusion decoder. The time derivative duθ−,t/dt is a Jacobian-vector product (JVP) (Lu & Song, 2024) output of the diffusion decoder, where the JVP calculation is approximated using the second-order difference technique proposed in UCGM (Sun et al., 2025). This strategy enables 8-step CFG-free inference while maintaining high image quality.

Fixed Max Length

TrainingBatch

Interleaved Editing MMU MMU Text

…

T2I T2I MMU Interleaved Editing

- Figure 5: Data Packing Strategy for Efficient Training. Multiple shorter samples are concatenated into fixed-length sequences, minimizing padding tokens and improving GPU utilization.

###### 4.5 Infrastructure for Training Efficiency

Image Tokens Pre-extraction. LLaDA2.0-Uni employs a Vector Quantized (VQ) tokenizer to transform images into discrete visual tokens, incurring substantial computational cost during training. We adopt an offline pre-extraction strategy: prior to training, the entire dataset is processed through the frozen tokenizer, with token indices stored on disk. During training, the data loader retrieves pre-extracted tokens directly, eliminating repeated encoder passes and significantly accelerating the pipeline.

Load Balancing via Data Packing. In LLaDA2.0-Uni, sequence lengths vary significantly across multimodal tasks (e.g., short text-only tasks versus long image-generation tasks). Traditional batching strategy requires extensive padding to match the longest sequence, wasting computation on padding tokens. We address this with an offline data packing strategy that consolidates multiple shorter samples into fixed-length sequences (Figure 5), significantly increasing effective token throughput for both pre-training and post-training.

Distributed Framework. We employ dFactory (InclusionAI, 2025) as the primary training engine for both pre-training and post-training phases—a high-efficiency framework specifically optimized for Diffusion Large Language Models. Built upon the VeOmni (Ma et al., 2025) distributed training ecosystem, dFactory enables the flexible deployment of sophisticated parallelization strategies.

##### 5 Experiments

- 5.1 Multimodal Understanding

- 5.1.1 Evaluation Settings

Benchmarks. We evaluate LLaDA2.0-Uni across 21 multimodal understanding benchmarks, focusing on three core capabilities: general VQA, reasoning, and OCR/document understanding:

- • General Tasks: MMStar (Chen et al., 2024b), MMBench (Liu et al., 2024b), MME (Fu et al., 2023), HallusionBench (Guan et al., 2024), RealWorldQA (OpenAI, 2024) and SimpleVQA (Cheng et al., 2025).
- • Reasoning Tasks: MMMU (Yue et al., 2024), and MMMU-Pro (Yue et al., 2025), MathVista (Lu et al., 2023), We-Math (Qiao et al., 2025), MathVision (Wang et al., 2024a), and MathVerse (Zhang et al., 2024).
- • OCR&Chart Tasks: ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), InfoVQA (Mathew et al., 2022), CharXiv (Wang et al., 2024c), OCRBench (Liu et al., 2024c), and AI2D (Kembhavi et al., 2016).
- • Other Multimodal Tasks: CountBench (Paiss et al., 2023), VLRewardBench (Li et al., 2025a), and V∗ (Wu & Xie, 2023).

Baselines. To evaluate the multimodal performance of LLaDA2.0-Uni, we compare it against an extensive set of baselines. We first compare LLaDA2.0-Uni with leading specialized VLMs, including Qwen2.5-VL7B (Qwen Team, 2025a) (AR-based) and LLaDA-V (You et al., 2025a) (diffusion-based). Furthermore, we evaluate LLaDA2.0-Uni against state-of-the-art unified models categorized into: 1) AR-based models, such as BAGEL (Deng et al., 2025) and InternVL-U (Tian et al., 2026); and 2) diffusion-based models, such as Lumina-DiMOO (Xin et al., 2025a) and LLaDA-o (You et al., 2026).

- 5.1.2 Multimodal Understanding Performance

As shown in Table 2, LLaDA2.0-Uni demonstrates strong and comprehensive multimodal understanding capabilities. Compared to existing diffusion-based unified models like Lumina-DiMOO and LLaDA-o, LLaDA2.0-Uni achieves significant improvements across all major categories, particularly in general VQA tasks (e.g., MMStar: 64.1 vs. 58.0) and complex reasoning tasks (e.g., MMMU: 50.1 vs. 44.9). Furthermore,

Table 2: The Overall Comparison of LLaDA2.0-Uni and Existing Specialist VLMs and Unified Models.

Specialist VLMs Unified Models Qwen2.5-

LLaDA2.0Uni General Tasks

LuminaDiMOO LLaDA-o

VL-7B LLaDA-V BAGEL InternVL-U

MMStar 63.9 60.1 67.0 54.7 61.0 58.0 64.1 MMBenchEN 83.5 82.9 85.0 75.3 84.5 71.1 81.5 MMBenchCN 83.4 70.1 82.4 73.6 71.8 69.9 81.2 MME-C 62.4 49.1 66.7 27.9 35.2 52.7 58.7 HallusionBench 51.9 39.2 52.5 44.8 32.9 47.4 50.2 RealWorldQA 68.5 63.2 73.9 56.4 52.4 60.8 66.7 SimpleVQA 47.9 26.0 41.9 20.7 12.1 29.2 44.0 Reasoning Tasks

MMMUval 51.3 48.6 55.3 54.7 58.6 44.9 50.1 MMMUProstandard 38.3 35.2 37.1 20.8 20.6 28.3 34.0 MathVistamini 68.2 59.7 73.1 55.8 10.3 66.1 68.1 MathVisionmini 22.4 21.2 24.1 22.1 13.1 15.7 26.7 WeMath 33.3 24.6 45.8 18.3 6.2 29.3 29.3

###### OCR & Chart Tasks

CharXiv(DQ) 73.9 47.0 70.6 53.3 27.8 69.8 68.4 ChartQA 84.1 78.3 74.3 76.6 8.3 87.9 80.1 OCRBench 84.2 63.2 73.3 83.9 7.6 74.6 75.7 DocVQA 94.9 83.9 94.3 85.4 7.2 91.5 89.5 AI2Dw mask 82.6 77.8 88.9 76.3 43.2 79.3 82.0 InfoVQA 80.3 66.3 60.7 68.3 6.2 54.7 70.1 Other Tasks

CountBench 84.9 75.1 93.2 62.2 48.4 91.7 86.0 VL-RewardBench 45.2 46.0 28.9 46.4 51.7 42.4 47.8 V∗ 80.1 41.8 67.5 51.3 52.7 57.6 61.8

LLaDA2.0-Uni also delivers consistently high performance in challenging OCR and document understanding scenarios, where baselines like Lumina-DiMOO struggle significantly. Most impressively, LLaDA2.0-Uni performs on par with state-of-the-art specialized VLMs such as Qwen2.5-VL-7B, even slightly outperforming it on specific metrics like MMStar (64.1 vs. 63.9) and CountBench (86.0 vs. 84.9). Overall, these results confirm that LLaDA2.0-Uni closes the gap between unified diffusion architectures and top-tier specialized VLMs.

- 5.2 Text-to-Image Generation

- 5.2.1 Evaluation Settings

Benchmarks. We conduct a comprehensive evaluation using a suite of established public benchmarks, including GenEval (Ghosh et al., 2023), DPG-Bench (Hu et al., 2024), One-IG Bench Chang et al. (2025), and UniGenBench Wang et al. (2025c) for general generative capabilities, as well as CVTG-2K (Du et al., 2025) for text rendering proficiency. To further evaluate reasoning-informed image generation, we also test LLaDA2.0-Uni on the WISE-Bench (Niu et al., 2025).

Baselines. To comprehensively assess the text-to-image generation capabilities of LLaDA2.0-Uni, we benchmark our model against a diverse spectrum of strong baselines. We first compare LLaDA2.0-Uni with leading specialized generation models (Gen. Only), including diffusion models like FLUX.1 [Dev] (Blackforest-labs, 2024), Lumina-Image 2.0 (Qin et al., 2025), Seedream 3.0 (Gao et al., 2025), Qwen-Image (Wu et al., 2025a), LongCat-Image (Team et al., 2025), and Z-Image (Cai et al., 2025), as well as AR-based models like Emu3 (Wang et al., 2024b) and Lumina-mGPT 2.0 (Xin et al., 2025b). Furthermore, we evaluate LLaDA2.0Uni against state-of-the-art unified models categorized into: 1) AR-based and hybrid (AR + Diff.) models,

Table 3: Comparison of Text-to-Image Generation Ability on GenEval Benchmark.

Single Object

Two Object

Attribute Binding

Type Model Arch.

Counting Colors Position

Overall↑

FLUX.1 [Dev] Diff. 0.98 0.81 0.74 0.79 0.22 0.45 0.66 Emu3-Gen AR 0.98 0.71 0.34 0.81 0.17 0.21 0.54

Gen.Only

Lumina-mGPT 2.0 AR 0.99 0.87 0.44 0.85 0.44 0.54 0.69 Seedream 3.0 Diff. 0.99 0.96 0.91 0.93 0.47 0.80 0.84 Qwen-Image Diff. 0.99 0.92 0.89 0.88 0.76 0.77 0.87

LongCat-Image Diff. 0.99 0.98 0.86 0.86 0.75 0.73 0.87 Z-Image-Turbo Diff. 1.00 0.95 0.77 0.89 0.65 0.68 0.82

Janus-Pro AR 0.99 0.89 0.59 0.90 0.79 0.66 0.80

BAGEL AR + Diff. 0.99 0.94 0.81 0.88 0.64 0.63 0.82 OmniGen2 AR + Diff. 1.00 0.95 0.64 0.88 0.55 0.76 0.80

HunyuanImage-3.0 AR + Diff. 1.00 0.92 0.48 0.82 0.42 0.63 0.72

Unified

NextFlow AR + Diff. 0.98 0.92 0.73 0.90 0.77 0.69 0.83 InternVL-U AR + Diff. 0.99 0.94 0.74 0.91 0.77 0.74 0.85

MMaDA D-Diff. 0.99 0.76 0.61 0.84 0.20 0.37 0.63 Lumina-DiMOO D-Diff. 1.00 0.94 0.85 0.89 0.85 0.76 0.88

LLaDA-o D-Diff. + Diff. 0.99 0.98 0.73 0.96 0.69 0.83 0.86 LLaDA2.0-Uni D-Diff. + Diff. 1.00 0.98 0.73 0.92 0.90 0.84 0.89

Table 4: Comparison of Text-to-Image Generation Ability on DPG Benchmark.

Type Model Arch. Global Entity Attribute Relation Other Overall↑

FLUX.1 [Dev] Diff. 74.35 90.00 88.96 90.87 88.33 83.84 Emu3-Gen AR 85.21 86.68 86.84 90.22 83.15 80.60

Gen.Only

Lumina-Image 2.0 Diff. 86.63 91.97 90.20 94.85 84.80 87.20 Seedream 3.0 Diff. 94.31 92.65 91.36 92.78 88.24 88.27 Qwen-Image Diff. 91.32 91.56 92.02 94.31 92.73 88.32

LongCat-Image Diff. 89.10 92.54 92.00 93.28 87.50 86.80 Z-Image-Turbo Diff. 91.29 89.59 90.14 92.16 88.68 84.86

Janus-Pro AR 86.90 88.90 89.40 89.32 89.48 84.19

BAGEL AR + Diff. 88.94 90.37 91.29 90.82 88.67 85.07 OmniGen2 AR + Diff. 88.81 88.83 90.18 89.37 90.27 83.57

HunyuanImage-3.0 AR + Diff. 92.12 92.53 89.13 92.13 91.92 86.10

Unified

NextFlow AR + Diff. 92.40 90.05 90.51 92.72 91.14 86.00 InternVL-U AR + Diff. 90.39 90.78 90.68 90.29 88.77 85.18

MMaDA D-Diff. 77.81 78.48 81.74 84.79 63.20 69.97 Lumina-DiMOO D-Diff. 81.46 92.08 88.98 94.31 82.00 86.04

LLaDA-o D-Diff. + Diff. 92.91 93.30 90.40 91.75 92.79 87.04 LLaDA2.0-Uni D-Diff. + Diff. 91.14 93.55 91.98 92.17 93.18 87.76

including Janus-Pro (Wu et al., 2025b), BAGEL (Deng et al., 2025), and OmniGen2 (Wu et al., 2025c), Hunyuan Image 3.0 (Cao et al., 2025b), NextFlow (Zhang et al., 2026b), and InternVL-U (Tian et al., 2026); and 2) discrete diffusion (D-Diff.) and hybrid diffusion (D-Diff. + Diff.) models, including MMaDA (Yang et al., 2025), Lumina-DiMOO (Xin et al., 2025a) and LLaDA-o (You et al., 2026).

###### 5.2.2 General Image Generation Performance

GenEval. Table 3 presents a comparison of model performance on the GenEval benchmark, which is designed to evaluate object-centric T2I generation using compositional prompts with diverse object attributes. LLaDA2.0-Uni demonstrates strong compositional capabilities, achieving an overall score of 0.89. This performance is highly competitive, significantly outperforming all unified models and bridging the gap with top-tier generation-only models. In particular, LLaDA2.0-Uni shows a clear advantage in spatial arrangement, securing the highest Position score (0.90) across all evaluated models.

Table 5: Comparison of Text-to-Image Generation Ability on OneIG-EN Benchmark.

Type Model Arch. Alignment Text Reasoning Style Diversity Overall↑

FLUX.1 [Dev] Diff. 0.786 0.523 0.253 0.368 0.238 0.434

Gen.Only

Lumina-Image 2.0 Diff. 0.819 0.106 0.270 0.354 0.216 0.353 Seedream 3.0 Diff. 0.818 0.865 0.275 0.413 0.277 0.530 Qwen-Image Diff. 0.882 0.891 0.306 0.418 0.197 0.539

Z-Image-Turbo Diff. 0.840 0.994 0.298 0.368 0.139 0.528

Janus-Pro AR 0.553 0.001 0.139 0.276 0.365 0.267

BAGEL AR + Diff. 0.769 0.244 0.173 0.367 0.251 0.361 OmniGen2 AR + Diff. 0.804 0.680 0.271 0.377 0.242 0.475 InternVL-U AR + Diff. 0.820 0.740 0.270 0.400 0.250 0.500

Unified

Lumina-DiMOO Diff. 0.820 0.550 0.280 0.400 0.230 0.460 LLaDA2.0-Uni D-Diff. + Diff. 0.882 0.661 0.323 0.400 0.259 0.505

Table 6: Comparison of Text-to-Image Generation Ability on UniGenBench.

Type Model Arch. Style World Attr. Action Relat. Logic Gram. Comp. Layout Text Overall↑

FLUX.1 [dev] Diff. 83.90 88.92 67.84 62.17 67.26 30.91 60.96 47.04 71.83 32.18 61.30

Gen.Only

Emu3-Gen AR 86.80 77.06 51.39 40.11 49.75 19.32 52.94 36.86 44.78 1.15 46.02 Seedream 3.0 Diff. 98.19 94.90 84.62 83.14 80.18 51.83 60.30 72.32 88.74 69.86 78.41 Qwen-Image Diff. 94.70 94.15 87.93 82.60 80.08 51.59 60.96 72.94 86.57 72.13 78.36

Z-Image Diff. 96.80 94.46 82.48 78.90 80.20 49.08 68.98 76.80 84.89 68.39 78.10

Janus-Pro AR 90.80 86.71 67.74 64.26 68.40 37.05 64.44 62.11 72.01 2.59 61.61

BAGEL AR + Diff. 90.20 85.60 67.74 61.98 70.69 30.23 66.44 58.12 76.49 7.76 61.53 OmniGen2 AR + Diff. 91.90 86.39 72.12 62.83 68.27 32.50 59.89 56.31 71.64 29.02 63.09

Unified

MMaDA D-Diff. 82.40 56.65 48.39 37.83 50.25 17.95 55.75 32.35 30.22 1.15 41.35 Lumina-DiMOO D-Diff. 89.70 90.03 81.62 71.12 78.43 45.45 70.45 73.32 82.84 25.57 71.12

LLaDA2.0-Uni D-Diff. + Diff. 95.30 93.67 91.77 85.65 86.42 63.99 72.19 85.82 90.30 31.23 79.63

Table 7: Comparison of Text Rendering Ability on CVTG-2K Benchmark.

Word Accuracy ↑ NED ↑ CLIPScore ↑ 2 regions 3 regions 4 regions 5 regions average

Type Model Arch.

FLUX.1 [Dev] Diff. 0.608 0.553 0.466 0.431 0.496 0.687 0.740 Seedream 3.0 Diff. 0.628 0.596 0.604 0.561 0.592 0.853 0.782 Qwen-Image Diff. 0.837 0.836 0.831 0.816 0.829 0.912 0.802

Gen.Only

LongCat-Image Diff. 0.912 0.873 0.855 0.831 0.865 0.936 0.785 Z-Image-Turbo Diff. 0.887 0.866 0.862 0.834 0.858 0.928 0.804

BAGEL AR + Diff. 0.498 0.391 0.332 0.291 0.356 0.657 0.779 InternVL-U AR + Diff. 0.729 0.660 0.618 0.549 0.623 0.804 0.816

Unified

Lumina-DiMOO D-Diff. 0.723 0.646 0.571 0.505 0.590 0.805 0.831 LLaDA2.0-Uni D-Diff. + Diff. 0.788 0.776 0.763 0.746 0.765 0.911 0.818

DPG-Bench. Table 4 reports the text-to-image generation results on the DPG benchmark. Notably, LLaDA2.0-

Uni achieves the state-of-the-art overall score of 87.76 among unified models, outperforming strong baselines such as LLaDA-o (87.04) and HunyuanImage-3.0 (86.10). Specifically, our model secures the highest scores on the Entity (93.55) and Other (94.04) sub-metrics. Furthermore, LLaDA2.0-Uni delivers highly competitive performance even against specialized generation-only models, surpassing Z-Image-Turbo (84.86) and demonstrating robust text-image alignment capabilities.

One-IG Bench. As shown in Table 5, LLaDA2.0-Uni achieves a highly competitive overall score of 0.505 on OneIG-EN. Notably, it outperforms all other unified models in Alignment (0.882) and Reasoning (0.323), reaching levels comparable to top dedicated generation models like Qwen-Image. However, LLaDA2.0-Uni

Table 8: Comparison of Reasoning-Informed Image Generation Ability on WISE Benchmark.

Type Model Arch. Cultural Time Space Biology Physics Chem. Overall↑

SD3-Medium Diff. 0.43 0.50 0.52 0.41 0.53 0.33 0.45 FLUX.1 [Dev] Diff. 0.48 0.58 0.62 0.42 0.51 0.35 0.50

Gen.Only

Emu3-Gen AR 0.34 0.45 0.48 0.41 0.45 0.27 0.39 Qwen-Image Diff. 0.62 0.63 0.77 0.57 0.75 0.40 0.62

LongCat-Image Diff. 0.66 0.61 0.72 0.66 0.72 0.49 0.65

Janus-Pro AR 0.30 0.37 0.49 0.36 0.42 0.26 0.35

BAGEL AR + Diff. 0.44 0.55 0.68 0.44 0.60 0.39 0.52 NextFlow AR + Diff. 0.62 0.60 0.70 0.54 0.58 0.38 0.59

Unified

InternVL-U AR + Diff. 0.37 0.51 0.68 0.39 0.62 0.39 0.46 Lumina-DiMOO D-Diff. 0.35 0.43 0.59 0.31 0.49 0.34 0.40

LLaDA2.0-Uni D-Diff. + Diff. 0.54 0.77 0.82 0.79 0.87 0.60 0.68 + w/ thinking D-Diff. + Diff. 0.73 0.79 0.86 0.81 0.88 0.74 0.78

Table 9: Comparison of Instruction-based Image Editing Ability on ImgEdit Benchmark.

Type Model Add Adjust Extract Replace Remove Back. Style Hybrid Action Overall ↑

FLUX.1 Kontext 4.25 4.15 2.35 4.56 3.57 4.26 4.57 3.68 4.63 4.00

Gen.Only

Step1X-Edit 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 Qwen-Image-Edit 4.32 4.36 4.04 4.64 4.52 4.37 4.84 3.39 4.71 4.35

Z-Image-Edit 4.40 4.14 4.30 4.57 4.13 4.14 4.85 3.63 4.50 4.30

BAGEL 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 OmniGen2 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 InternVL-U 4.13 3.40 2.27 4.13 3.39 3.84 4.77 3.03 4.05 3.67

Unified

Lumina-DiMOO 3.41 2.38 1.90 3.26 2.21 2.11 4.19 2.26 3.17 2.77 LLaDA2.0-Uni 3.76 4.16 2.40 4.04 3.82 4.07 4.60 3.97 4.42 3.92

falls short of leading models in generating dense text, indicating an area for future improvement.

UniGenBench. LLaDA2.0-Uni sets a new standard for unified models on UniGenBench (EN), achieving a top overall score of 79.63 (Table 6). It performs consistently well across all ten dimensions, showing a clear advantage in Logic (63.99) and Layout (90.30), where it even surpasses many specialized generation models. These results demonstrate that LLaDA2.0-Uni effectively closes the performance gap between general-purpose unified models and top-tier specialized models.

###### 5.2.3 Text-Centric Image Generation Performance

CVTG-2k. The CVTG-2K benchmark evaluates the capability of the model for text rendering across multiple regions. As reported in Table 7, LLaDA2.0-Uni leads the unified models with an overall score of 0.765. A key observation is its exceptional stability in multi-region text generation. While baselines like BAGEL, Lumina-DiMOO, and InternVL-U show a sharp performance drop when the number of regions increases, our model experiences a much slower decline.

###### 5.2.4 Reasoning-Informed Image Generation Performance

WISE-Bench. The benefits of our large-scale multimodal pre-training are evident on the WISE benchmark (Table 8), which evaluates complex semantic understanding and world knowledge in image generation. LLaDA2.0-Uni achieves a strong overall score of 0.68, ranking first among all unified models and performing on par with generation-only models like LongCat-Image. Notably, incorporating a reasoning mode yields an additional 10% improvement. These results highlight the strong capability of LLaDA2.0-Uni in reasoninginformed generation.

- 5.3 Image Editing

- 5.3.1 Evaluation Settings

Benchmarks. We evaluate LLaDA2.0-Uni on two general instruction-based image editing benchmarks: ImgEdit-Bench (Ye et al., 2025c) and GEdit-Bench (Liu et al., 2025). Additionally, we provide qualitative

Table 10: Comparison of Instruction-based Image Editing Ability on GEdit Benchmark. Abbreviations: Semantic Consistency (G SC), Perceptual Quality (G PQ), and Overall Score (G O).

GEdit-Bench-EN↑ GEdit-Bench-CN↑ G SC G PQ G O G SC G PQ G O

Type Model Arch.

FLUX.1 Kontext Diff. 6.52 7.38 6.00 - - -

Gen.Only

Step1X-Edit Diff. 7.66 7.35 6.97 7.20 6.87 6.86 Qwen-Image-Edit Diff. 8.00 7.86 7.56 7.82 7.79 7.52

LongCat-Image-Edit Diff. 8.18 8.00 7.64 8.08 7.99 7.60 Z-Image-Edit Diff. 8.11 7.72 7.57 8.03 7.80 7.54

BAGEL AR + Diff. 7.36 6.83 6.52 7.34 6.85 6.50

OmniGen2 AR + Diff. 7.16 6.77 6.41 - - InternVL-U AR + Diff. - - 6.66 - - -

Unified

Lumina-DiMOO D-Diff. - - 3.91 - - LLaDA2.0-Uni D-Diff. + Diff. 6.68 7.52 6.61 6.63 7.67 6.66

Table 11: Comparison of Multi-Reference Image Editing Ability on MICo-Bench.

Model Arch. Object Person HOI De&Re Overall↑ Qwen-Image-Edit Diff. 52.4 21.1 35.0 37.4 35.9

BAGEL AR + Diff. 39.0 28.5 25.3 44.5 34.4 OmniGen2 AR + Diff. 46.3 22.9 32.2 36.8 33.8

Lumina-DiMOO D-Diff. 38.4 12.1 24.7 21.3 23.3 LLaDA2.0-Uni D-Diff. + Diff. 51.0 32.8 46.0 54.4 47.1

comparisons against leading models on MICo-Bench (Wei et al., 2025), a challenging multi-reference image editing benchmark.

Baselines. We evaluate LLaDA2.0-Uni against specialized editing models (FLUX.1 Kontext (Labs et al., 2025), Step1X-Edit (Liu et al., 2025), Qwen-Image-Edit (Wu et al., 2025a), Z-Image-Edit (Cai et al., 2025)) and unified models (BAGEL (Deng et al., 2025), OmniGen2 (Wu et al., 2025c), InternVL-U (Tian et al., 2026), Lumina-DiMOO (Xin et al., 2025a)). For the multi-reference editing benchmark, we compare with BAGEL, Qwen-Image-Edit, and OmniGen2, as they are the only baselines that natively support multiple image inputs.

- 5.3.2 General Image Editing Performance

ImgEdit. Table 9 presents the instruction-based image editing performance on the ImgEdit benchmark. Among unified models, LLaDA2.0-Uni achieves the best Overall score of 3.92, ranking first and significantly outperforming peers like OmniGen2 (3.44) and InternVL-U (3.67). Notably, LLaDA2.0-Uni excels in the Adjust and Hybrid tasks, securing the highest scores within the unified category. These results underscore its robust capability to comprehend and execute intricate editing instructions.

GEdit-Bench. To further assess complex image editing capabilities, we evaluate our model on the GEdit benchmark. As detailed in Table 10, LLaDA2.0-Uni achieves solid overall scores across both English (6.61) and Chinese (6.66) evaluations. A key highlight is its strong performance in the Perceptual Quality category, proving that the model can execute edits without sacrificing the visual quality of the original image.

- 5.3.3 Multi-Reference Image Editing Performance

MICo-Bench. To evaluate the multi-reference image composition capabilities of LLaDA2.0-Uni, we conduct experiments on the MICo-Bench. As shown in Table 11, LLaDA2.0-Uni achieves the best overall performance, setting a new state-of-the-art on this benchmark with a score of 47.1. It significantly outperforms strong baselines such as OmniGen2 (33.8) and Qwen-Image (35.9). Notably, Lumina-DiMOO, which shares the same dLLM architecture, struggles significantly on this task, yielding an overall score of only 23.3. This stark contrast clearly validates the effectiveness of our architecture and data pipeline.

- 5.4 Interleaved

- 5.4.1 Interleaved Generation

Benchmark Construction. Existing models like Bagel and Nextflow support interleaved generation, yet a standard benchmark remains absent. Datasets like ISG-BENCH (Chen et al., 2024a) and OpenING (Zhou et al.,

[Figure 59]

|[Figure 60]<br><br>A skateboarder launching off a concrete planter, gliding across a spacious urban plaza, and moving toward the center of the frame with a serene cityscape in the background.| |
|---|---|
| | |
|[Figure 61]<br><br>.<br><br>A skateboarder in a black hoodie and dark pants is captured mid-air, launching off a large concrete... <image 1> Next, the skateboarder has landed and is now gliding… <image 2> Moving forward, the skateboarder continues his ride across the expansive plaza… <image 3><br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]| |

|Please help me generate the cooking process for Cola Chicken Wings.<br><br>[Figure 65]|
|---|
|[Figure 66]<br><br>Step 1: Pan-fry the wings until golden and slightly crispy on both sides.<image 1> Step 2: Add cola and simmer. <image 2> Step 3: Thicken the sauce and top with sesame. <image 3><br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]|

Recipe /Cooking

Character

TravelGuide

Cartoon

Telling

Explanation

[Figure 70]

Story

### InterGen Bench Evaluation Metrics

[Figure 71]

sDaily

Product Manual

|A hydraulic press machine is gradually pressing into a placed can. What will happen next?<br><br>[Figure 72]<br><br>[Figure 73]|Scenarios|
|---|---|
|[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>The hydraulic press machine gradually presses the can out of shape, bending and crumpling its thin metal walls. <image1> The hydraulic press machine flattens the can thoroughly into a thin metal sheet. <image2>| |
| | |

Text Coherence

ScoringDimensions

[Figure 77]

Event Forecasting

Text-image Alignment

Action Anticipation

Plot

Movie

[Figure 78]

ID Consistency

Movement Trajectory

Figure 6: Overview of InterGen Benchmark.

###### User Prompt LLaDA2.0-Uni Interleaved Generation Output

A muscular man with a beard, wearing a black tank top and a black beanie, sits on a beige couch. He looks directly at the camera with a neutral expression, his right hand resting on his thigh. Behind him, a large window reveals a cityscape with tall buildings, and a framed picture hangs on the wall to his left.

Next, the man shifts his gaze slightly to his right, his expression becoming more animated as he appears to be speaking. His right hand remains on his thigh, while his left hand is now raised in a subtle, open-palmed gesture, suggesting he is emphasizing a point in his conversation.

Then, he turns his head back to face the camera, his mouth slightly open as if midsentence. His right hand is now lowered, resting on his thigh, while his left hand is no longer visible in the frame. The background remains unchanged, with the city view and framed picture still prominent.

Simultaneously, he turns his head slightly to his left, his eyes looking off-camera as if addressing someone or something offscreen. His right hand is still on his thigh, and his left hand is now lowered, resting near his thigh, indicating a momentary pause in his speech.

Please generate a sequence of 6 images showing a muscular bearded man in a black tank top and beanie sitting on a couch, engaging with the camera through a series of expressive gestures and facial reactions in a bright modern apartment setting.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Then, place the steak in the pan carefully and cook it for a few minutes on each side until the surface becomes nicely browned. If needed, adjust the cooking time depending on how well-done you want the steak.

First, take the steak out of the refrigerator and season both sides with

Next, heat the pan over medium-high heat and add a small amount of oil. Wait until the pan is hot enough so the steak can sear properly.

Finally, remove the steak from the pan and let it rest for a few minutes on a rack

salt and black pepper. Prepare a little oil, a frying pan, and a pair of tongs.

or plate, loosely covered with foil. This Please teach me how to helps keep the meat juicy before serving. cook a steak in a frying pan step by step.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Figure 7: Qualitative Results on Interleaved Generation Task.

2025) contain interleaved cases but specialized tasks (e.g., 3D scene transformation) that most models cannot process. To address this, we propose the InterGen benchmark. As shown in Figure 6, it is structured into three main categories and various subcategories to comprehensively cover practical interleaved applications. InterGen comprises 150 samples and utilizes advanced VLMs (Gemini-3 and Qwen3-VL) as a judge to assess performance across three dimensions: text coherence, text-image alignment, and ID consistency.

InterGen Benchmark. We primarily compare our model with Emu3.5 (Cui et al., 2025c), as other models capable of interleaved generation (such as NextFlow (Zhang et al., 2026b) and Mogao (Liao et al., 2025)) are not yet open-source. As shown in Table 12, LLaDA2.0-Uni generally outperforms Emu3.5 on the InterGen benchmark. Specifically, it achieves higher scores in the Story Telling and Time Series Forecasting tasks, while demonstrating comparable performance in Explanation. More visualizations of the interleaved generation are shown in Figure 7.

###### 5.4.2 Interleaved Reasoning

Unified models have already achieved impressive results in standard image understanding and generation tasks. Looking forward, exploring interleaved reasoning emerges as a critical bottleneck to overcome. We

Table 12: Comparison of Interleaved Generation Ability on InterGen Benchmark.

Story Telling Explanation Event Forecasting Gemini↑ Qwen3-VL↑ Gemini↑ Qwen3-VL↑ Gemini↑ Qwen3-VL↑ Emu3.5 AR + Diff. 6.28 6.83 6.19 6.48 5.08 5.75

Model Arch.

LLaDA2.0-Uni D-Diff. + Diff. 6.42 7.02 6.22 6.35 5.19 5.94

###### Question LLaDA2.0-Uni Interleaved Reasoning Output

|[Figure 87]|
|---|

Step 1: The problem presents a classic Atwood machine setup. To solve it, I need to apply Newton's second law to each mass individually. The key insight is that both masses will share the same magnitude of acceleration, but in opposite directions. The tension in the rope will be uniform for both masses.

Step 3: The free-body diagram shows the forces acting on each mass: the gravitational force (weight) pulling them down and the tension force pulling up. To apply Newton's second law, I need to sum these forces along the axis of motion. This requires choosing a coordinate system. Let's set the positive y-axis as upward.

Step 4: Now, I will apply Newton‘s second law to each mass separately. For mass m1 , the equation is T - m1g = m1a. For mass m2, the equation is m2g - T = m2a . By solving these equations: The acceleration magnitude of the system is a ≈ 5.98 m/s², where m1 accelerates upwards and m2 accelerates downwards.

Step 2: First, I will determine the direction of motion. Since 6.2 kg is

heavier than m1 = 1.5 kg, m2 will accelerate downwards, and m1 will accelerate upwards. I will visualize this with a free-body diagram to clarify the forces acting on each mass.

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

Question: Consider the system depicted in the diagram, where two masses, with

Final Answer:

m1 = 1.5 kg and m2 = 6.2 kg, are connected by a cord that passes over a frictionless, massless pulley. Calculate the acceleration of the masses and the tension within the connecting cord.

The tension in the rope is T ≈ 23.67 N.

[Figure 91]

- Step 1: The problem asks for the best move for White. I need to evaluate the given options and determine which one is the most effective.
- Step 2: Let's consider option A, Qd2. This move develops the queen to a central square, potentially connecting the

- Step 3: After Qd2, the queen is optimally placed on d2, effectively completing development and preparing White for the middlegame.
- Step 4: Let's consider option B, Qc2. This move develops the queen to the cfile, potentially placing it on a less active

- Step 5: After Qc2, the queen is on c2, but it fails to connect the rooks and leaves the d-file unsupported.
- Step 6: Let's consider option C, Re1. This move develops the rook to the open file, potentially committing it too early

- Step 7: After Re1, the rook is active, but the queen remains stuck on d1, delaying the crucial connection of White's major pieces.
- Step 8: Let's consider option D, g4. This move pushes a pawn in front of the castled king, potentially exposing it to devastating counterattacks.

before the queen is developed.

Final Answer: A.

rooks for a fully coordinated position.

square behind the c4 pawn.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Question: What is the best move for White to play?

A: Qd2. B: Qc2. C: Re1. D: g4.

Figure 8: Qualitative Results on Interleaved Reasoning Task.

believe this is an essential step toward building a strong synergy between visual generation and understanding. In LLaDA 2.0-Uni, we conduct a preliminary exploration into interleaved reasoning capability. As illustrated in Figure 8, through interleaved reasoning, our model successfully deduces logical strategies in in chess games and provides step-by-step solutions to physics problems. These promising results give us great confidence to further expand interleaved reasoning capabilities in future research.

- 5.5 Ablation Study

- 5.5.1 Analysis of SPRINT Acceleration

- Table 13 evaluates SPRINT on nine multimodal benchmarks, covering both understanding (AI2D, OCRBench, MathVista, ChartQA, DocVQA, MMMU, MMStar) and generation (GenEval, DPG). SPRINT shifts the average score from 76.3 to 75.7 (−0.6) while accelerating generation from 24.3 to 39.8 TPS (1.6×). Several benchmarks show score decreases; the most notable are OCRBench (−2.3) and DPG (−1.5). OCRBench demands precise character-level prediction, where the lower threshold τ = 0.93 may accept tokens before sufficient refinement. The speedup is largest on benchmarks with longer outputs, where the per-step savings from prefix pruning compound across many denoising iterations: DocVQA reaches 3.5× (8.0 → 27.6), and ChartQA and AI2D both reach 2.2×. SPRINT improves MMMU by +2.4 and ChartQA by +0.9. The non-uniform unmasking schedule concentrates refinement on uncertain positions, effectively increasing the denoising budget for difficult tokens without additional forward passes. Furthermore, we are integrating LLaDA2.0-Uni with

Table 13: Performance and TPS comparison with and without SPRINT. (w/o SGLang)

Method Metric AI2D OCRB MathVista ChartQA DocVQA MMMU MMStar GenEval DPG Avg. ∆

Score 82.0 75.7 68.1 80.1 89.5 50.1 64.1 89.0 87.76 76.3 – TPS 19.5 21.2 55.0 28.7 8.0 49.4 31.7 2.8 2.7 24.3 –

LLaDA2.0-Uni

Score 80.9 73.4 67.2 81.0 89.0 52.5 63.0 87.8 86.27 75.7 −0.6 TPS 42.9 36.0 75.0 62.3 27.6 52.2 49.2 5.1 7.8 39.8 ×1.6

+ SPRINT

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

DiffusionDecoder

(50Steps) DistilledDecoder

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

(8Steps)

Figure 9: Visual Comparison of the Decoder and the Distilled Version.

SGLang (Zheng et al., 2024) to accelerate inference, and the implementation will be made publicly available soon.

###### 5.5.2 Analysis of Diffusion Decoder

- Table 14 shows the performance and speed comparison between the Diffusion Decoder (50 steps) and Diffusion Decoder Turbo (8 steps) across five benchmarks. Speed is measured on a single GPU at 1024 × 1024 resolution with a batch size of 1, using BF16 precision.

Table 14: Performance and Speed Comparison between Diffusion Decoder and Diffusion Decoder Turbo.

Method Speed (s / img) GenEval DPG UniGenBench OneIG-EN WISE Diffusion Decoder (50 steps) 32.95 0.89 87.76 79.63 0.505 0.68

Diffusion Decoder Turbo (8 steps) 2.90 0.87 87.24 79.76 0.500 0.68

Through few-step acceleration, the Diffusion Decoder Turbo achieves an 11.4× speedup (from 32.95s/img to 2.90s/img) while maintaining competitive performance across all benchmarks. Specifically, it retains a GenEval score of 0.87 (vs. 0.89 for the base) and a DPG score of 87.24 (vs. 87.76). Similarly, on the UniGenBench, OneIG-EN, and WISE benchmarks, Diffusion Decoder Turbo achieves performance on par with the original decoder. Moreover, as shown in Figure 9, the visual quality remains virtually indistinguishable.

##### 6 Conclusion and Future Directions

In this work, we introduce LLaDA2.0-Uni, a unified framework that enables both multimodal understanding and generation within a single diffusion large language model. Built on the LLaDA 2.0 backbone, our approach uses a SigLIP-VQ tokenizer to map visual inputs into semantically rich discrete tokens, allowing text and images to be modeled in a shared space. Extensive experiments show that LLaDA2.0-Uni achieves strong performance across various benchmarks, including multimodal understanding, image generation, and editing. Furthermore, the model naturally supports interleaved generation and chain-of-thought reasoning, demonstrating the flexibility and practicality of our unified architecture.

Despite these advancements, several areas remain for future improvement:

- • Enhancing Visual Detail. While the SigLIP-VQ tokenizer provides rich semantic information, it struggles to preserve fine-grained image details. Future work could focus on better reconstruction techniques to benefit detail-sensitive tasks like image editing.
- • Scaling Interleaved Capabilities. To fully unlock the model’s potential for complex interleaved generation and reasoning, further scaling of training data and model capacity is required.
- • Reinforcement Learning. Although we have begun exploring RL for unified dLLMs, optimizing its performance remains a challenge. We plan to further refine the RL framework in future versions and then release it to the community.

##### References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Inclusion AI, Bowen Ma, Cheng Zou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Chenyu Lian, Dandan Zheng, Fudong Wang, Furong Xu, et al. Ming-flash-omni: A sparse, unified architecture for multimodal perception and generation. arXiv preprint arXiv:2510.24821, 2025.

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, and Jiankang Deng. Llava-onevision-1.5: Fully open framework for democratized multimodal training. In arXiv, 2025.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. Proceedings of the International Conference on Learning Representations (ICLR), 2025.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

Black-forest-labs. FLUX.1. 2024. Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang,

Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Shuo Cao, Nan Ma, Jiayang Li, Xiaohui Li, Lihao Shao, Kaiwen Zhu, Yu Zhou, Yuandong Pu, Jiarui Wu, Jiaquan Wang, Bo Qu, Wenhai Wang, Yu Qiao, Dajuin Yao, and Yihao Liu. Artimuse: Fine-grained image aesthetics assessment with joint scoring and expert-level understanding. arXiv preprint arXiv:2507.14533, 2025a.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025b.

Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arxiv:2506.07977, 2025.

Dongping Chen, Ruoxi Chen, Shu Pu, Zhaoyi Liu, Yanru Wu, Caixi Chen, Benlin Liu, Yue Huang, Yao Wan, Pan Zhou, et al. Interleaved scene graphs for interleaved text-and-image generation assessment. arXiv preprint arXiv:2411.17188, 2024a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems (NeurIPS), 2024b.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024c.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024d.

Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for multimodal large language models. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025.

Wei Chow, Jiachun Pan, Yongyuan Liang, Mingze Zhou, Xue Song, Liyu Jia, Saining Zhang, Siliang Tang, Juncheng Li, Fengda Zhang, et al. Weave: A benchmark for evaluating multimodal editing models. arXiv

- preprint arXiv:2511.15738, 2024.

Wei Chow, Linfeng Li, Lingdong Kong, Zefeng Li, Qi Xu, Hang Song, Tian Ye, Xian Wang, Jinbin Bai, Shilin Xu, et al. Editmgt: Unleashing potentials of masked generative transformers in image editing. arXiv

- preprint arXiv:2512.11715, 2025.

Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025a.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025b.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, et al. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025c.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Nikai Du, Zhennan Chen, Zhizhou Chen, Shan Gao, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461, 2025.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, et al. Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833, 2025.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm

for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. InclusionAI. dfactory: Easy and efficient dllm fine-tuning. 2025. Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in

photographs of natural scenes. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A

diagram is worth a dozen images. In Proceedings of the European Conference on Computer Vision (ECCV), 2016. Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali,

Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image

classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision (IJCV), 2020.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Muller,¨ Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Ang Li, Charles Wang, Deqing Fu, Kaiyu Yue, Zikui Cai, Wang Bill Zhu, Ollie Liu, Peng Guo, Willie Neiswanger, Furong Huang, Tom Goldstein, and Micah Goldblum. Zebra-cot: A dataset for interleaved vision-language reasoning. In Proceedings of the International Conference on Learning Representations (ICLR), 2026.

Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, et al. Vl-rewardbench: A challenging benchmark for vision-language generative reward models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025a.

Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025b.

Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

Ling Team. Every activation boosted: Scaling general reasoner to 1 trillion open language foundation. arXiv preprint arXiv:2510.22115, 2025.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Dongyang Liu, Yi Xin, Shitian Zhao, Le Zhuo, Weifeng Lin, Xinyue Li, Qi Qin, Guangtao Zhai, Xiaohong Liu, Hongsheng Li, et al. Lumina-mgpt: Flexible photorealistic autoregressive text-to-image generation. International Journal of Computer Vision (IJCV), 2026.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In Proceedings of the European Conference on Computer Vision (ECCV), 2024b.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences (SCIS), 2024c.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Jian Ma, Xujie Zhu, Zihao Pan, Qirong Peng, Xu Guo, Chen Chen, and Haonan Lu. X2edit: Revisiting arbitrary-instruction image editing through self-constructed data and task-aware representation learning. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2026.

Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, et al. Veomni: Scaling any modality model training with model-centric distributed recipe zoo. arXiv preprint arXiv:2508.02317, 2025.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2022.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2021.

Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2022.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Chaoran Feng, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI. Grok 1.5v: A new era in ai understanding. 2024. Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip

to count to ten. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023.

Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv:2510.19808, 2025.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Xinyue Li, Dongyang Liu, Xiangyang Zhu, et al. Lumina-image 2.0: A unified and efficient image generative framework. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2025.

Qwen Team. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025a. Qwen Team. Qwen3-vl: Sharper vision, deeper thought, broader action. Qwen Blog. Accessed, pp. 10–04,

2025b.

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

Jianlin Su. Moe travels 3. 2025. Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary

position embedding. arXiv preprint arXiv:2104.09864, 2021.

Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025. Peng Sun, Xinyi Shang, Tao Lin, and Zhiqiang Shen. Duality models: An embarrassingly simple one-step

generation paradigm. arXiv preprint arXiv:2602.17682, 2026.

Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.

Changyao Tian, Danni Yang, Guanzhou Chen, Erfei Cui, Zhaokai Wang, Yuchen Duan, Penghao Yin, Sitao Chen, Ganlin Yang, Mingxin Liu, et al. Internvl-u: Democratizing unified multimodal models for understanding, reasoning, generation and editing. arXiv preprint arXiv:2603.09877, 2026.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems (NeurIPS), 2024a.

Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025a.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Yibin Wang, Zhimin Li, Yuhang Zang, Jiazi Bu, Yujie Zhou, Yi Xin, Junjun He, Chunyu Wang, Qinglin Lu, Cheng Jin, et al. Unigenbench++: A unified semantic evaluation benchmark for text-to-image generation. arXiv preprint arXiv:2510.18701, 2025c.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems (NeurIPS), 2024c.

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In Proceedings of the International Conference on Learning Representations (ICLR), 2024.

Xinyu Wei, Kangrui Cen, Hongyang Wei, Zhen Guo, Bairui Li, Zeqing Wang, Jinrui Zhang, and Lei Zhang. Mico-150k: A comprehensive dataset advancing multi-image composition. arXiv preprint arXiv:2512.07348, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025b.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025c.

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025a.

Yi Xin, Juncheng Yan, Qi Qin, Zhen Li, Dongyang Liu, Shicheng Li, Victor Shea-Jay Huang, Yupeng Zhou, Renrui Zhang, Le Zhuo, et al. Lumina-mgpt 2.0: Stand-alone autoregressive image modeling. arXiv preprint arXiv:2507.17801, 2025b.

Yi Xin, Le Zhuo, Qi Qin, Siqi Luo, Yuewen Cao, Bin Fu, Yangfan He, Hongsheng Li, Guangtao Zhai, Xiaohong Liu, et al. Resurrect mask autoregressive modeling for efficient and scalable image generation. arXiv preprint arXiv:2507.13032, 2025c.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025.

Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025a.

Keming Ye, Zhipeng Huang, Canmiao Fu, Qingyang Liu, Jiani Cai, Zheqi Lv, Chen Li, Jing Lyu, Zhou Zhao, and Shengyu Zhang. Unicedit-10m: A dataset and benchmark breaking the scale-quality barrier via unified verification for reasoning-enriched edits. arXiv preprint arXiv:2512.02790, 2025b.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025c.

Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025a.

Zebin You, Xiaolu Zhang, Jun Zhou, Chongxuan Li, and Ji-Rong Wen. Llada-o: An effective and lengthadaptive omni diffusion model. arXiv preprint arXiv:2603.01068, 2026.

Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models to regress accurate image quality scores using score distribution. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025b.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

Boqiang Zhang, Lei Ke, Ruihan Yang, Qi Gao, Tianyuan Qu, Rossell Chen, Dong Yu, and Leoweiliang. Penguin-vl: Exploring the efficiency limits of vlm with llm-based vision encoders. arXiv preprint arXiv:2603.06569, 2026a.

Huichao Zhang, Liao Qu, Yiheng Liu, Hang Chen, Yangyang Song, Yongsheng Dong, Shikun Sun, Xian Li, Xu Wang, Yi Jiang, et al. Nextflow: Unified sequential modeling activates multimodal understanding and generation. arXiv preprint arXiv:2601.02204, 2026b.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Pengfei Zhou, Xiaopeng Peng, Jiajun Song, Chuanhao Li, et al. Opening: A comprehensive benchmark for judging open-ended interleaved image-text generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Le Zhuo, Songhao Han, Yuandong Pu, Boxiang Qiu, Sayak Paul, Yue Liao, Yihao Liu, Jie Shao, Xi Chen, Si Liu, et al. Factuality matters: When image generation and editing meet structured visuals. arXiv preprint arXiv:2510.05091, 2025.

