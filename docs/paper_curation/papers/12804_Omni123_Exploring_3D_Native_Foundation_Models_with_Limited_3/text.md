arXiv:2604.02289v1[cs.CV]2Apr2026

[Figure 1]

## Exploring 3D Native Foundation Models with Limited 3D Data by Unifying Text to 2D and 3D Generation

Chongjie Ye1,2 , Cheng Cao3 , Chuanyu Pan3 , Yiming Hao2 , Yihao Zhi2 , Yuanming Hu3 , Xiaoguang Han2,1,*

1FNii-Shenzhen 2SSE, CUHK(SZ) 3Meshy AI *Corresponding author: hanxiaoguang@cuhk.edu.cn

###### Abstract

Recent multimodal large language models have achieved remarkable success in unified text and image understanding and generation, yet extending such native capability to 3D remains an open challenge. The core bottleneck is data: compared to the near-infinite 2D imagery on the web, high-quality 3D assets are orders of magnitude scarcer, leaving 3D synthesis severely under-constrained. Current methods often circumvent this limitation through indirect pipelines that edit in 2D image space and lift results into 3D via iterative optimization, sacrificing geometric consistency and the straightforwardness of native generation. We present Omni123, a 3D native foundation model that addresses limited 3D data by unifying text-to-2D and text-to-3D generation within a single autoregressive framework. Our key insight is that cross-modal generative consistency between images and 3D can act as an implicit structural constraint: by representing text, images, and 3D geometry as discrete tokens in a shared sequence space, the model can leverage abundant 2D observations as a rich geometric prior to fortify 3D representations. To realize this, we introduce an interleaved X-to-X training paradigm that coordinates diverse cross-modal tasks over heterogeneous paired datasets without requiring fully aligned text–image–3D triplets. By traversing semantic–visual–geometric cycles (e.g., text → image → 3D → image) within single autoregressive sequences, the model learns representations that simultaneously satisfy high-level semantic intent, appearance fidelity, and multi-view geometric consistency, while mitigating harmful interference between appearance and geometry objectives. Extensive experiments demonstrate that Omni123 significantly improves geometric consistency and semantic alignment in text-guided 3D generation and editing, validating that unifying 2D and 3D generative processes provides an effective and scalable pathway toward multimodal 3D world models.

[Figure 2]

Figure 1 Comparison of Text-to-3D paradigms

#### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 2.1 Image Generation & Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.2 3D Generation & Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.3 Multimodal Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Data Pre-Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3.1 Data Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Text–Image Pairs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.3 Image–3D Pairs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.4 Text–3D Pairs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.5 Interleaved SFT Triplets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Model Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 4.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.2 3D Shape Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.3 Image Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.4 Model Pre-training (PT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.5 Continued Training (CT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.6 Supervised Fine-Tuning (SFT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 5 Align to Instruction-Based 3D Editing . . . . . . . . . . . . . . . . . . . . . . . . 16
- 6 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 6.1 Text-to-3D Shape Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 6.2 Instruction-Based 3D Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 6.3 Image Tokenizer Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

2

[Figure 3]

- Figure 2 Omni123 enables native 3D generation and editing within a unified multimodal framework trained with limited 3D data. Top: Text-to-joint-2D-and-3D generation results across diverse prompts. Each example shows the generated 2D image (inset) alongside the normal map of the corresponding 3D output, demonstrating high geometric fidelity and strong semantic alignment with the input text. Bottom: Native 3D editing of 3D models through sequential and branching text instructions (e.g., “+wearing a kimono,” “+riding a skateboard”), illustrating the model’s ability to perform iterative and diverse 3D modifications directly in 3D space. More qualitative results are shown in Figures 8 and 9.

#### 1 Introduction

Recent breakthroughs in generative AI have catalyzed a shift toward unified multimodal models capable of jointly reasoning across linguistic and visual domains, with models like GPT-4o[53] and Nano-Banana [23] enabling native capabilities for high-fidelity synthesis and instructionguided editing. As the community pushes toward more immersive digital environments, the natural evolution is the quest for a "3D Nano-Banana": a unified framework capable of native 3D generation and editing, which is essential for advancing applications in embodied AI and autonomous virtual world synthesis. However, achieving such native capability in 3D remains a persistent challenge.

Unlike 2D AIGC, which leverages near-infinite web-scale data, 3D generation faces acute data scarcity. This disparity leaves 3D synthesis inherently under-constrained, especially under sparse supervision from text or single views. To compensate, current methods often treat 2D images as an intermediate proxy—performing edits in the image space and lifting them into 3D through iterative optimization. While effective, these indirect approaches frequently compromise geometric consistency and lack the straightforwardness of 2D workflows. This suggests that 3D learning requires stronger structural constraints than sparse signals can provide. Crucially, as

###### 2D visual data implicitly encodes deep 3D structural information, including object shape [44], appearance [74], and spatial relationships [3], it prompts a pivotal hypothesis: Can we leverage the abundance of 2D observations to fortify 3D generation via a unified multimodal framework?

A common strategy involves aggregating multiple cross-modal tasks. However, it does not inherently guarantee effective knowledge transfer. This stems from the fact that different generative tasks are rooted in distinct statistical priors: the text-to-image task focuses on modeling appearance distributions, whereas the text-to-3D task requires consistent geometric reasoning. Without meticulous coordination, joint training may introduce both beneficial transfer and harmful task interference, where competing gradients may ultimately degrade the quality of the learned 3D representations.

To address this challenge, we introduce an interleaved X-to-X training paradigm. We posit that cross-modal generative consistency between images and 3D can act as an implicit structural constraint for learning robust 3D representations. Different modalities offer complementary observations of the same underlying object: text describes high-level semantic intent, images capture rich appearance cues and partial geometric hints, while

###### 3D representations provide explicit spatial structure and multi-view consistency. Consequently, we propose that cross-modal generative consistency between images and 3D can serve as a potent implicit constraint. Specifically, by forcing a model to traverse a "semantic-visual-geometric" cycle (e.g., text → image → 3D → image), the resulting representations could simultaneously satisfy high-level intent, appearance fidelity, and multi-view consistency.

We investigate this hypothesis through a unified autoregressive formulation that treats text, image, and 3D geometry as discrete tokens within a shared sequence space. To circumvent the reliance on fully paired text–image–3D triplets, we leverage heterogeneous paired datasets. The model is trained on a diverse set of conditional generation tasks where any modality can serve

- as input, and the target modality is predicted autoregressively. This formulation maximizes the utility of available multimodal data and ensures that the model does not merely learn isolated tasks, but rather develops a unified latent representation across the semantic, visual, and geometric domains.

In summary, we incorporate the following key technical advancements:

- • We introduce a framework that tokenizes text, images, and 3D geometry within a shared sequence space. This enables native 3D reasoning and editing, eliminating the reliance on intermediate 2D proxies and iterative image-space optimization.
- • We propose an interleaved training scheme that coordinates diverse generative tasks across

- heterogeneous datasets. This approach mitigates task interference between appearance and geometry priors, facilitating beneficial cross-modal knowledge transfer from 2D to 3D.
- • We leverage generative consistency across "semantic-visual-geometric" cycles as an implicit structural constraint. This allows abundant 2D observations to serve as a rich geometric prior, effectively fortifying 3D representations against acute data scarcity. Through extensive experiments, we study how different multimodal objectives interact

during training and how cross-modal generative processes influence the quality of learned 3D representations. Our results indicate that interleaved cross-modal generation significantly improves geometric consistency and semantic alignment in text-guided 3D generation and editing tasks. These findings suggest that cross-modal generative consistency provides an effective pathway toward scalable multimodal 3D world models.

#### 2 Related Work

- 2.1 Image Generation & Editing

Generative adversarial network (GAN) is the first representative image generation [5, 21, 35, 36] and editing [45, 54] method in the deep learning era. To improve the distribution coverage, subsequent research also explored likelihood-based generative models. Diffusion models [8, 17, 25, 55, 62] and Flow Matching [19, 38, 46] later became the dominant paradigm by formulating image synthesis as the reverse denoising process from noise to data. Beyond their strong generation quality, diffusion models provide a more expressive latent space and a natural interface for conditioning, which has driven the rapid development of image editing methods, including controllable generation[69, 94], inversion-based training-free editing[12, 20], and optimizationbased editing[33, 61]. More recently, emerging large-scale visual generation models[7, 39, 77] have increasingly explored leveraging stronger multi-modal reasoning and instruction-following capabilities of VLMs to unify image editing, generation, and understanding.

- 2.2 3D Generation & Editing
- 3D generation has evolved along two main paradigms. [47, 58] relies on pretrained image/video diffusion models and distills their priors into 3D representations through score distillation. While effective, this indirect 2D supervision often leads to slow optimization, inconsistent 3D appearance, and artifacts. In contrast, recent native 3D generative models [28, 79, 80] learns 3D distribution directly, providing a more efficient and scalable route toward high-quality 3D synthesis and already seeing broad adoption in commercial applications[31, 41, 52, 72]. Building on these strong foundation models, many recent works [27, 29, 90] have further explored editing tasks. Nevertheless, direct end-to-end text-3D synthesis [26, 56, 59] still lags far behind due to the large modality gap. To address this challenge, we propose a text-image-3D-image interleaved training paradigm that uses images as a bridge to alleviate both the modality gap and the conflict between generation and reconstruction, achieving state-of-the-art text-3D performance.

- 2.3 Multimodal Generation

Numerous previous works have explored extending LLMs toward visual generation and understanding via mapping learned visual tokens into the text feature space of a pretrained diffusion [11, 37, 68]. Later studies further explore an end-to-end unified architecture for generation and understanding across modalities [9, 15, 48, 75, 84]. More recently, GPT-4o [53] and Nano-Banana [23] have further accelerated this progress. For the 3D modality, ShapeLLM-Omni [89] is among the first to unify 3D generation and understanding within a single autoregressive framework. However, it purely suffers from quantizationinduced information loss, which limits generation quality. Lucid3D [10] further extends to 3D editing, while Cube3D[4] improves the 3D tokenizer with self-supervised objectives, leading to higher fidelity and further extending to 3D scene generation. AR3D-R1[70] introduces the first

###### 3D RL system to improve 3D consistency and alignment with human preferences. However, due to the scarcity and expense of 3D data, the scalability and generalization of previous works have been constrained. To alleviate this limitation, we unify 2D and 3D generation so that 3D tasks can benefit from rich and general visual knowledge from large-scale 2D data. This enables high-quality 3D creation with only limited text-image-3D pairs, while also yielding emergent generation and editing capabilities.

#### 3 Data Pre-Processing

In this section, we describe the composition of our multi-modal dataset, the cleaning and filtering pipelines applied to each modality, the data synthesis pipeline used to scale high-quality SFT data, and the captioning strategies used to bridge modalities.

##### 3.1 Data Overview

Our training corpus spans three modalities—text, images, and 3D assets—and is organized into the following paired and triplet subsets. Table 1 summarizes the complete data composition across pre-training and supervised fine-tuning stages.

Table 1 Summary of training data composition. Token counts per sample are computed using 77 text tokens, 1,024 image tokens per view, and 1,024 3D mesh tokens. The table is divided into pairwise pre-training data (upper) and continued training & supervised fine-tuning data (lower). Total tokens shown are the dataset size; tokens seen during training are higher due to multiple epochs (see Table 6).

###### Stage Data Configuration Items Tokens/Sample Total Tokens Token %

Text + Image 63.7M 1,101 70.1B 14.8% Image + 3D 120M 2,048 245.8B 51.8% Text + 3D 9M 1,101 9.9B 2.1%

Pre-Training

Subtotal 192.7M — 325.8B 68.6%

Text + Single-View Image + 3D 2M 2,125 4.3B 0.9% Text + 6 Posed Images + 3D 20M 7,245 144.9B 30.5%

CT & SFT

Subtotal 22M — 149.2B 31.4% Total 214.7M — 475.0B 100%

The pre-training stage uses 194.7M pairwise samples spanning three cross-modal directions (Text↔Image, Image↔3D, Text↔3D), totaling 328.0B tokens in the dataset itself. The interleaved SFT stage introduces two chain-of-generation tasks: (i) a single-view pipeline (Text→Image→3D) with 2M triplets, and (ii) a multi-view pipeline (Text→6 Posed Images→3D) with 20M samples, where six views are rendered from known camera poses. Despite comprising only 9.2% of all items, the multi-view SFT data accounts for 30.4% of all tokens due to the much longer per-sample sequence length (7,245 tokens). Note that the total tokens seen during training (reported in Table 6) is higher (∼1.16T for pre-training) due to multiple epochs over the data. Table 2 details the tokenization configuration and sequence lengths for each modality.

##### 3.2 Text–Image Pairs

The text–image pre-training corpus contains 63.7M open-domain images in total, comprising proprietary data mixed with synthetic data by Z-Image [7].

A natural question is whether the 120M rendered images from our Image–3D corpus (Section 3.3) should also be included, given their scale. We deliberately choose not to do so: these renderings exhibit uniform lighting, synthetic material appearance, and the absence of natural backgrounds, diverging substantially from in-the-wild photographs. Mixing the two domains in the text–image objective would create a distribution conflict that degrades image generation quality. We therefore confine rendered imagery to the Image→3D and Text→3D objectives, where the rendering distribution is consistent with the target task.

##### 3.3 Image–3D Pairs

The image–3D pairs constitute the largest subset (120M) and are critical for learning the reconstruction mapping between 2D observations and 3D geometry. Following established

Table 2 Per-modality tokenization configuration.

Modality Tokens Tokenizer Text 77 CLIP [60] text encoder + Qwen3-0.6B [87] for en-

hanced semantic understanding Image 1,024 / view Our image tokenizer, a 1D discrete visual tokenizer with semantic regularization 3D Mesh 1,024 Cube3D [4], a Perceiver-based VQ-VAE with optimaltransport vector quantization

practices in recent large-scale 3D data engineering [30, 42, 43, 93], we process this corpus through a three-stage pipeline:

- (1) Rendering and format conversion: 3D assets from heterogeneous sources are converted into

meshes, aligned to a canonical orientation, standardized with PBR textures, saved as GLB files, and rendered using Meshy’s in-house renderer with HDRI environments and lighting conditions sampled from a pool of 2,000 setups.

- (2) Asset filtering: we exclude assets with poor geometric quality (low polygon counts,

simplistic topology), poor texture quality (UV defects, low resolution), noisy photogrammetry scans, and large thin-walled structures—the latter causing SDF sign discontinuities that destabilize training and multi-view inconsistencies where thin geometry becomes invisible under certain viewpoints.

(3) Post-processing: filtered meshes undergo watertight conversion, narrow-band SDF sampling, and point cloud sampling.

##### 3.4 Text–3D Pairs

Existing captions for 3D objects [50] either suffer from poor alignment with the geometry they describe or lack sufficient detail, which limits high-quality text-to-3D generation. We therefore design a multi-granularity captioning pipeline utilizing visual chain-of-thought reasoning, to produce multiple precise and detailed text descriptions focusing on different aspects of a 3D model. When applicable, metadata is also supplied to the model for factual world knowledge grounding.

The pipeline is run on each of the 5M filtered 3D assets, consisting of three stages:

(1) Visual Chain of Thought Analysis: we feed multi-view rendered images of each asset into a Visual-Language-Model (VLM) such as Gemini-3 [22] to generate highly detailed perview descriptions that are used as anchors for the model to perform orientation and spatial correspondence analysis. After the model has identified spatial correspondence between images and therefore oriented the object, the VLM is asked to analyze the model in terms of its appearance, geometry, potential function, and origin in the real world.

- (2) Captioning and Categorization: The VLM takes the analysis and produces a paragraph-

long detailed caption that goes over all aspects of the 3D model. Categories are then assigned by the VLM, producing rough semantic classification.

- (3) Human captions mimicking: We prepared 10K human-labeled short captions distributed

over all categories, and 4 random examples are selected per-asset based on its evaluated category to perform few-shot caption augmentation. Human captions leveraged in this way greatly improve the lexical and syntactical diversity of our captions.

All three stages are completed within one continuous multi-turn conversation, preserving all context to the model — including the images. Reasoning is enabled, and the reasoning framework is enforced through schema-guided generation.

Through this multi-granularity captioning process, each of the 5M filtered 3D assets is augmented with multiple caption variants. We then perform an alignment filtering step, removing pairs where the caption poorly matches the 3D geometry based on VLM-evaluated consistency scores. This yields a total of 9M Text–3D training pairs used in pre-training (Section 4.4).

##### 3.5 Interleaved SFT Triplets

While the Objaverse corpus [13, 14] provides a substantial volume of image–3D training pairs, the majority of its assets exhibit simple structures and plain surface details, limiting the generation capabilities of models trained exclusively on this data. Given the prohibitive cost of manually creating high-quality 3D assets, we construct a data synthesis pipeline that performs Text→Image→3D generation, producing a dataset of synthesized 3D assets with considerably more complex structures and richer geometric detail.

- Step 1: Prompt Curation. We initiate the synthesis process from text prompts rather than

image prompts, as text affords more direct control over semantic diversity and ensures variety in the resulting geometries. We first source approximately 14M high-quality raw prompts from DiffusionDB [76]. A Gemini-3-Flash [22] classifier is applied to filter out complex multi-object scenes, retaining only single-object descriptions amenable to 3D asset generation. Rule-based filtering further eliminates stylistic modifiers (e.g., “cinematic”, “4k wallpaper”), followed by a structural standardization step to enforce consistent prompt formatting. After all filtering stages, approximately 2M prompts are retained.

- Step 2: Text-to-Image Generation. We employ Z-Image-Turbo[7] for high-throughput text-

to-image generation. The text prompt conditioning is augmented with viewpoint and lighting specifications to produce object-centric images with controlled camera angles, which is important for stable downstream 3D reconstruction.

- Step 3: Background Removal. To obtain clean object masks suitable for image-to-3D

generation, we apply BiRefNet [96] for high-resolution foreground segmentation. We then subsequently use VLM to evaluates each masked image for segmentation artifacts (e.g., residual background, truncated object parts) and filters out samples with poor mask quality. It also simultaneously re-captions the masked object to produce an aligned text description, ensuring cross-modal consistency between the final caption and the visual content.

- Step 4: Image-to-3D Synthesis. We adopt Meshy 4 to synthesize 3D meshes from the masked

images. The generated meshes undergo the same watertight processing and point cloud sampling pipeline described in Section 3.3.

- Step 5: Quality Filtering. A rigorous data cleaning process combines automated geometric

assessment with MLLM-based evaluation. Specifically, we render multi-view images of each synthesized mesh and evaluate three quality axes: (i) geometric consistency—whether the object is structurally sound with preserved symmetry and no warped geometry; (ii) surface texture clarity—whether micro-details (e.g., wood grain, metal finish, fabric weave) are sharp rather than flat or muddy; and (iii) silhouette connectivity—whether the object has a clear, recognizable silhouette without heavy occlusion or detached floaters. For each axis, the MLLM assigns a binary “high”/“low” rating, and only samples rated “high” on all three axes are retained in the final dataset.

Multi-Granularity Re-Captioning. Each retained asset is re-captioned at three levels of specificity by an MLLM: a long caption (40–70 words) providing specification-level detail including materials, textures, colors, and geometric complexity; a medium caption (10–25 words) covering subject, primary material, and basic geometry; and a short caption (1–5 words) capturing only the core subject identity. These multi-granularity captions serve both the SFT triplet task (Text→Image→3D) and the multi-view SFT task (Text→6 Posed Images→3D).

#### 4 Model Training

##### 4.1 Architecture

A key obstacle to unified multi-modal generation is the heterogeneity of representations across modalities. Text is naturally a sequence of discrete symbols; images are typically encoded as

- 2D spatial grids of continuous features [1, 62]; and 3D shapes are represented as volumetric lattices [89], unordered point sets, or continuous latent vectors [92]. These representations are incompatible with autoregressive language modeling in two fundamental ways. First, continuous latent spaces—such as the latent vectors produced by 3DShape2VecSet [92]—lack the discrete vocabulary required for next-token prediction; Second, even when discretized, 2D or 3D spatial tokenizers produce token counts that scale with spatial resolution, yielding intractable context lengths, and their modality-specific positional encodings and attention patterns fragment the model and hinder cross-modal knowledge transfer.

Building on recent advances in vector-quantized tokenization [4, 85] that convert continuous representations into discrete codebook indices amenable to next-token prediction, and 1D tokenization that compresses each modality into a short, structurally uniform sequence, Omni123 adopts a self-developed representative image tokenizer (Section 4.3) for images and Cube3D [4] geometry tokenizer for 3D meshes. This combination yields compact 1D discrete token sequences for both modalities, enabling seamless concatenation without modality-specific positional encoding. Both modalities are then processed by a shared autoregressive Transformer backbone that maximizes parameter sharing and enforces cross-modal coherence through shared self-attention.

As illustrated in Figure 3, the architecture consists of modality-specific tokenizers, a unified autoregressive Transformer backbone, and modality-specific output heads. For text conditioning, we employ two complementary encoders: CLIP [60] provides Lclip = 77 vision-language-aligned embeddings, while Qwen3-0.6B [87] provides Lqwen embeddings that capture richer linguistic detail; the two are concatenated and injected via cross-attention. Images are tokenized into Li = 1,024 1D tokens via our image tokenizer; and 3D geometry are encoded into Lm = 1,024 1D geometry tokens via Cube3D [4] shape tokenizer (detailed in Sections 4.2 and 4.3). Image and 3D shape tokens are concatenated at the sequence level and processed by the backbone autoregressively.

The backbone follows the dual-stream autoregressive architecture of Cube3D [4]. A conditioning stream carrying text embeddings and a generation stream carrying image and 3D shape tokens run in parallel through the dual-stream layers. At each dual-stream layer, the two streams are independently projected into queries, keys, and values; these are then concatenated along the sequence dimension and processed by a single joint attention operation under a causal mask. The attention output is split back along the sequence boundary and used to update both streams through separate feed-forward sub-layers. In the final dual-stream layer, the conditioning stream contributes only keys and values—its query projection and post-attention update are omitted, so the information flow becomes unidirectional from conditioning to generation. The subsequent single-stream layers retain only causal self-attention over the generation tokens, without further text interaction. Within the generation stream, image and 3D shape tokens are concatenated into a single flat sequence and processed by fully shared self-attention weights—no modality-specific branches or attention masks are introduced. This means every attention layer implicitly performs cross-modal fusion, and all generation-side parameters are shared across modalities, enabling visual priors learned from the large-scale Text→Image data to directly benefit the data-scarcer Text→3D task. Each transformer layer employs SwiGLU [63] as the feed-forward network with a

- 4× expansion ratio, following the design of Cube3D [? ]. Detailed architecture hyperparameters are listed in Table 3.

Table 3 Model architecture configuration.

Hyperparameter Value Dual-stream (joint-attention) layers 24 Single-stream (self-attention) layers 6 Hidden dimension d 1,536 Attention heads 12 Per-head dimension 128 FFN expansion ratio 4× (SwiGLU) Text encoder (vision-language) CLIP (Lclip = 77) Text encoder (linguistic) Qwen3-0.6B Image tokens Li 1,024 3D shape tokens Lm 1,024 3D codebook embedding dim 16 RoPE θ 1,000 Layer norm type LayerNorm (ϵ = 10−6, no affine) QK-Norm RMSNorm Token dropout pdrop 0.1

##### 4.2 3D Shape Tokenizer

A central requirement of our unified architecture is a 3D tokenizer that produces discrete tokens capable of faithfully capturing a wide range of geometric properties—smooth surfaces, sharp edges, high-frequency details—while being natively compatible with autoregressive sequence models. We adopt Cube3D [4], a shape tokenizer built on the 3DShape2VecSet [92] framework, which converts continuous shape representations into discrete tokens suitable for mixed-modal foundation models [71].

Encoder-Decoder Architecture. The tokenizer follows an encoder-decoder design. The encoder takes as input a point cloud sampled from the mesh surface and embeds it using a phase-modulated positional encoding, which improves the Perceiver-based transformer’s ability to disambiguate spatially distinct points in cross-attention layers. A Perceiver-based transformer [32] compresses the point cloud into a set of Lm = 1,024 continuous latent vectors, which are subsequently discretized through optimal-transport vector quantization [4] to produce discrete 3D shape tokens. The decoder reconstructs an implicit occupancy field from these discrete tokens, from which a triangle mesh can be extracted via Marching Cubes.

##### 4.3 Image Tokenizer

Image tokenizers [64, 73] play a crucial role in compressing visual data into compact discrete latent tokens, facilitating scalable and efficient training of autoregressive models for visual generation via next-token prediction. We build our image tokenizer upon GiGaTok [85], whose 1D token representation is more compatible with our 3D shape tokenizer, while also better handling the inherent redundancy in images [91]. Recent studies [98] trace the sub-optimal performance of discrete tokenizers back to a core limitation: the disjoint optimization of the codebook. This disjointness inherently degrades reconstruction fidelity compared to continuous VAEs [62, 88], ultimately acting as a critical bottleneck that hinders pure autoregressive models from matching the state-of-the-art generation quality of diffusion models or hybrid approaches. To alleviate this issue, we propose a two-stage training strategy (Figure 4): First, we train a continuous VAE to learn semantically rich visual representations and high-fidelity reconstruction. Then we insert the 1D GiGaTok [85] into the pretrained VAE and train it only to reconstruct the continuous features, simplifying the vector quantization to a pure 1D compact token extraction task. This

###### Input and Tokenization Unified Autoregressive Transformer

6x Single Stream Blocks

푐푙  

# ·

Text CLIP

Encoder

Causal Self-Attn

[Figure 4]

Text Tokens

A modern chair, with a sample and continuing prompt...

Image Head

Image Decoder

Qwen3 Encoder

24x Dual Stream Blocks

Image

Causal Self-Attention

푞푤  

[Figure 5]

[Figure 6]

[Figure 7]

v q

1D Image Tokens

k

Image Tokenizer

[Figure 8]

FFN FFN

= 1024

3D Shape Head

3D Shape Decoder

Causal Self-Attention

3D Shape

[Figure 9]

qkv proj

qkv proj

1D Geometry Tokens

3D Shape Tokenizer

Cross Entropy Loss

Generation Tokens

= 1024

- Figure 3 Overview of the Omni123 architecture. Text is encoded by dual text encoders (CLIP [60] and Qwen3-0.6B [87]) and fed into a conditioning stream, while images and 3D shapes are tokenized into 1D discrete tokens and concatenated into a unified generation stream. The unified autoregressive transformer backbone uses 24 dual-stream blocks to jointly process the conditioning and generation tokens under causal attention, followed by 6 single-stream layers operating only on generation tokens, and finally with modality-specific linear heads decoding token logits over the 2D and 3D codebooks.

design yields a high-performance tokenizer that is scalable and equipped with rich semantic representations.

Architectural Details. Inspired by recent successful Representation Autoencoder[18, 65, 95], we retrain DINO-Tok [34] with Dinov3-L [66] encoder and an 832-dimensional latent space as aforementioned VAE on ImageNet [16] with 256 × 256 and 512 × 512 resolution. In the second stage, we freeze the DINO-Tok VAE and only train the 1D Q-Former on a 60M-image dataset and 512 × 512 resolution with 1024 query tokens and a 64K-entry codebook of dimension 64. To the best of our knowledge, our tokenizer is the first 1D visual tokenizer trained at this scale.

##### 4.4 Model Pre-training (PT)

The central challenge motivating our pre-training design is the acute scarcity of 3D data relative to 2D: our corpus contains 63.7M text–image pairs but only 9M text–3D pairs. As argued in Section 1, 2D observations implicitly encode rich 3D structural information—shape, appearance, and spatial relationships—suggesting that large-scale 2D supervision can serve as a powerful geometric prior for 3D learning. However, naïvely mixing appearance-focused and geometryfocused objectives risks task interference rather than beneficial transfer.

Our pre-training strategy addresses this tension through the X-to-X paradigm: all crossmodal generation tasks—text-to-image, text-to-3D, image-to-3D, and 3D-to-image—are unified as autoregressive next-token prediction over a shared discrete vocabulary. By training a single model to traverse multiple generation paths across the “semantic–visual–geometric” cycle, the framework enforces cross-modal generative consistency as an implicit structural constraint. Concretely, the large-scale Text→Image subset strengthens the shared text conditioning pathway that Text→3D relies on despite its limited data, while the 3D→Image task—as the inverse of Image→3D—forces the model to maintain explicit 3D-to-2D geometric correspondences, which in turn benefits forward Image→3D reconstruction.

Task Formulation. During pre-training, we train on four core generation tasks (Table 4). Each task takes one or two source modalities as conditioning and autoregressively generates tokens in the target modality. Text, when present, is injected via the cross-attention pathway; non-text source tokens are placed as a non-causal prefix in the self-attention sequence. The model is trained to predict only the target tokens that follow the prefix. Note that text conditioning

[Figure 10]

###### Stage I: Continuous VAE

Projection

Texture

###### Input image Recon Image

Continuous Latent

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

VIT Decoder

Semantic

DINOv3-L

Reconstruction Loss

Image Reconstruction Loss

###### Stage II: 1D Tokenizer

Query Tokens

[Figure 16]

Reconstructed Continuous Latent

Continuous Latent

[Figure 17]

Codebook

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

VAE Encoder

VAE Decoder

Q-Former

Q-Former

Input image

Feature Reconstruction Loss Recon Image

- Figure 4 Two-stage image tokenizer training strategy. Stage 1 trains a continuous VAE (DINO-Tok) to learn high-fidelity visual representations. Stage 2 freezes the VAE and trains a 1D Q-Former to reconstruct the continuous features, reducing vector quantization to a compact 1D token extraction task.

- Table 4 Pre-training task configurations. Text conditioning is optionally dropped for Classifier-Free Guidance. The loss is computed only over the target column.

###### Task Cond. Stream Prefix Target

Text → Image Caption — Image tokens Text → 3D Caption — 3D shape tokens (Text +) Image → 3D (Caption) Image tokens 3D shape tokens (Text +) 3D → Image (Caption) 3D shape tokens Image tokens

is randomly dropped during training to enable Classifier-Free Guidance (discussed in Training Objective below), so the model must also learn to generate from the non-text prefix alone. Training Objective. All four tasks share a single autoregressive cross-entropy loss over the target token sequence:

Ltgt

log pθ xj | x<j, c (1)

L = −

j=1

where xj is the j-th target token, x<j denotes preceding tokens including any source-modality prefix, and c is the text conditioning (empty when dropped). A uniform loss weight is applied across all tasks. To enable Classifier-Free Guidance [24] at inference time, we randomly drop the text conditioning with probability pdrop = 0.1 during training [67], replacing it with a learned null embedding. For the two prefix-conditioned tasks (Image→3D and 3D→Image), this means the model is exposed to three conditioning regimes: text + prefix, prefix only, and unconditional, allowing flexible guidance at inference.

Data Mixture. Pre-training draws from three data pools of varying scale: Text–Image pairs (63.7M), Image–3D pairs (120M), and Text–3D pairs (9 M). Each pool serves one or more tasks: Text–Image supports Text→Image; Image–3D supports both Image→3D and 3D→Image (with the task direction randomly selected per sample); and Text–3D supports Text→3D. Naïvely sampling in proportion to dataset size would severely under-represent the smallest yet critical Text–3D subset. We adopt a temperature-based weighted sampling strategy. The probability of

Table 5 Pre-training data mixture. Each pool may serve multiple tasks as described in the text.

###### Data Pool Size Tasks Served Priority Effective Pi

Text–Image 63.7M Text→Image 1.0 0.30 Image–3D 120M Image→3D, 3D→Image 1.5 0.53 Text–3D 9M Text→3D 3.0 0.17

Table 6 Training recipe across all stages.

Pre-training Post-training Hyperparameter Stage 1 Stage 2 CT SFT Peak learning rate 5e-4 1e-4 1e-5 1e-6 LR scheduler Cosine Optimizer AdamW (β1=0.9, β2=0.999) Weight decay 0.1 Gradient norm clip 5.0 Warm-up steps 5,000 1,000 1,000 2,000 Training steps 400,000 250,000 120,000 115,000 Effective batch size 2,048 1,024 1,024 256 Image resolution 2562 5122 5122 5122 GPUs 256×H100 Tokens seen ∼0.76T ∼0.40T ∼0.20T ∼0.15T

drawing from subset i is:

(Sizei)T × Priorityi

Pi =

(2)

n j=1 (Sizej)T × Priorityj

where Sizei is the number of samples in pool i, Priorityi is a manually assigned importance weight reflecting task value, and T ∈ (0,1] is a temperature controlling the degree of rebalancing—lower T flattens the size distribution, giving smaller but high-priority pools greater representation.

- Table 5 lists the configuration. Training Schedule. Pre-training proceeds in two stages (Table 6). Stage 1 trains at 256×256 resolution for 400,000 steps with a peak learning rate of 5e-4, allowing the model to learn cross-modal alignment across all four tasks. Stage 2 increases the image resolution to 512 × 512 and continues for 250,000 steps with a reduced learning rate of 1e-4, refining visual fidelity while preserving the learned cross-modal representations. Both stages use AdamW [49] (β1 = 0.9, β2 = 0.999, weight decay = 0.1) with a cosine learning rate schedule. Gradient norms are clipped

- at 5.0. The effective batch size is 2,048 (Stage 1) and 1,024 (Stage 2) across 256×H100 GPUs. In total, the model sees approximately 1.16T tokens during pre-training.

##### 4.5 Continued Training (CT)

Pre-training establishes cross-modal alignment between text, image, and 3D representations, yet the model lacks an explicit notion of viewpoint: it can generate images and 3D shapes, but cannot reason about how the same object appears from different camera poses. This is a critical gap—multi-view consistency is the bridge that connects 2D appearance to 3D geometry, and without it, the “semantic–visual–geometric” cycle introduced in Section 1 remains incomplete. Continued training (CT) closes this gap by introducing view tokens (Figure 5): a set of N learnable embeddings, each bound to a fixed canonical viewpoint (front, back, left, right, top, bottom). By prepending a view token to the target image sequence, the model learns to associate

Task Definition

Target Image Tokens

[Figure 27]

[Figure 28]

Front Back Left Right Top Bottom

- • Text → lmage
- • Image → 3D
- • Text → 3D
- • 3D → lmage Pre-training Tasks
- • Image → Posed Image
- • 3D → Posed Image
- • Text → Posed lmage

###### Front

[Figure 29]

Self-Attention Stream

View-Conditioned Tasks

View Token Mechanism

Continued Training (CT)

[Figure 30]

[Figure 31]

푉푖  target

View

[Figure 32]

… Omni123

… Target View Sequences

Target View Sequences

- Figure 5 Continued training introduces view-conditioned generation via learnable view tokens. Top left: N=6 view tokens corresponding to canonical viewpoints (front, back, left, right, top, bottom) are mapped from 3D camera extrinsics. Top right: A view token v is prepended to the target image tokens and fed into the self-attention stream, enabling viewpoint-controllable generation. Bottom: In the Posed Image

→ Posed Image task, the source image is accompanied by its own view token vsrc, providing explicit source–target viewpoint correspondence for novel view synthesis.

each embedding with a specific camera pose, enabling viewpoint-controllable generation that is geometrically grounded in the shared 3D representation learned during pre-training.

Task Formulation. CT retains all four pre-training tasks and introduces three new viewconditioned tasks (Figure 5): Image → Posed Image, 3D → Posed Image, and Text → Posed Image. The new tasks share a common mechanism: a learnable view token v encoding the target camera extrinsics are prepended to the output image tokens in the generation stream. When the source is also an image, it is accompanied by its own view token vsrc in the prefix, providing the model with explicit source–target viewpoint correspondence and enabling novel view synthesis. This design yields progressively richer 3D-aware capabilities—from single-image novel view synthesis (Image → Posed Image) to explicit 3D-controllable rendering (3D → Posed Image) to text-driven viewpoint-specific generation (Text → Posed Image).

Data. The view-conditioned tasks are constructed from the existing Image–3D pairs. For each 3D asset, we render images at N = 6 fixed canonical viewpoints (front, back, left, right, top, bottom), yielding posed image–3D tuples. During training, a task and a target viewpoint are randomly sampled per example. The pre-training data pools remain active to prevent catastrophic forgetting of the base generation capabilities.

Training Setup. CT uses the same training objective (Eq. 1) and optimizer configuration as pre-training. Only the N view token embeddings are newly initialized; all other parameters are warm-started from the pre-trained checkpoint. The learning rate is reduced to 1 × 10−5 with 1,000 warm-up steps. The model trains for 120,000 steps with an effective batch size of 1,024. The temperature-based data mixing strategy (Eq. 2) is extended to include the new view-conditioned pool, with priority weights adjusted to emphasize 3D-aware generation while retaining base capabilities.

##### 4.6 Supervised Fine-Tuning (SFT)

Pre-training and Continued Training equip the model with cross-modal alignment and viewpoint awareness, but each task is still trained on pairwise data independently: there is no guarantee that a 3D mesh produced by Image→3D renders back to the conditioning image, or that an image generated by Text→Image yields a plausible 3D reconstruction. The SFT stage closes this loop by introducing interleaved multi-modal sequences that chain image generation, 3D reconstruction,

- Table 7 SFT task configurations. All tasks use text conditioning via the conditioning stream. vk denotes view tokens. Img×N denotes N posed images generated sequentially.

Task Sequence Text → Img → 3D [Img1, 3D] Text → 3D → Img [3D, v1, Img1]

Img×N → 3D [v1, Img1, ..., vN, ImgN, 3D] 3D → Img×N [3D, v1, Img1, ..., vN, ImgN] Text → Img×N → 3D → Img×M [v1, Img1, ..., vN, ImgN, 3D, vN+1, ImgN+1, ...]

and novel view rendering into a single autoregressive sequence. Because each subsequent modality is conditioned on all previously generated tokens, the training signal naturally enforces crossmodal consistency: a 3D mesh must be faithful to the images that precede it, and novel views rendered from that mesh must be coherent with the original generation.

Task Formulation. We define five SFT tasks (Table 7) covering all practical generation pipelines. Each task is formulated as a single autoregressive sequence over interleaved modality tokens. Text conditioning remains in the conditioning stream with the same CFG dropout (pdrop = 0.1). View tokens vk specify camera poses for posed image generation.

#### 5 Align to Instruction-Based 3D Editing

Earlier sections have demonstrated that Omni123’s unified autoregressive architecture excels not only at 3D generation but also at cross-modal understanding, owing to its inherited textconditioned capabilities and the rich 3D–2D correspondences learned during pre-training and continued training. Consequently, we can adapt the model into an instruction-based 3D editing system, further revealing its potential to benefit users in practical content-creation workflows.

Task Formulation. We formulate instruction-based 3D editing as a conditional generation task as follows. Let Msrc denote the source mesh tokenized into Lm = 1,024 discrete tokens s via Cube3D, and let cedit be a natural-language edit instruction (e.g., “add wings to the car”, “change the material to gold”). Following the prefix-conditioned paradigm established in pre-training (Section 4.4), the source 3D shape tokens are placed as a non-causal prefix in the self-attention stream, while the edit instruction is injected via cross-attention. The model autoregressively generates the target token sequence s′ = (s′1,...,s′L

), which is decoded into the edited mesh Mtgt via the Cube3D decoder and Marching Cubes. Text conditioning is randomly dropped with probability pdrop = 0.1 during training to enable Classifier-Free Guidance at inference, so the model is exposed to three conditioning regimes: edit instruction + source prefix, source prefix only, and unconditional, mirroring the CFG setup of the prefix-conditioned pre-training tasks.

m

Data. A key bottleneck for instruction-based 3D editing has been the lack of large-scale, high-quality paired data. Unlike 2D image editing, where paired datasets can be constructed at scale using foundation models [6], 3D editing data must additionally ensure cross-view consistency, structural fidelity in unedited regions, and semantic coherence of the edited parts. We adopt 3DEditVerse [78], the largest paired 3D editing benchmark to date, comprising 116,309 highquality training pairs and 1,500 curated test pairs. Each pair consists of a source mesh, an edited mesh, and a natural-language edit instruction describing the modification. 3DEditVerse is constructed through two complementary automated pipelines: (i) a pose-driven geometric pipeline that leverages publicly available 3D characters and animation sequences, exploiting the fact that different poses of the same character naturally form valid “before–after” edit pairs for articulation and structural variations; and (ii) a foundation-model-guided appearance pipeline

that orchestrates multiple foundation models to generate edited assets with modified textures, colors, or stylistic attributes while preserving the underlying geometry. Both pipelines ensure edit locality, multi-view consistency, and semantic alignment between the instruction and the applied modification. Each training pair is tokenized into source tokens s, target tokens s′, and text embeddings via the same encoders used throughout Omni123.

Training Objective. The editing task shares the same autoregressive cross-entropy loss as all earlier training stages (Eq. 1), applied over the target token sequence:

Lm

Ledit = −

log pθ s′k | s′<k, s, cedit (3)

k=1

where s′k is the k-th target 3D token, s′<k denotes all preceding target tokens, s is the source prefix, and cedit is the text conditioning (replaced with a learned null embedding when dropped). The loss is computed only over the target tokens; the source prefix is excluded from the loss computation. A uniform loss weight is applied across all target positions.

Training Setup. All parameters are initialized from the SFT checkpoint (Section 4.6). We fine-tune the full model using AdamW [49] (β1 = 0.9, β2 = 0.999, weight decay = 0.1) with a peak learning rate of 1×10−5 and a cosine schedule with 1,000 warm-up steps. Training proceeds for 10,000 steps with an effective batch size of 128 across 32×H100 GPUs. Gradient norms are clipped at 5.0, consistent with the pre-training configuration.

[Figure 33]

Input: a gummy bear driving a convertible

Input: a dragon cat hybrid

[Figure 34]

[Figure 35]

[Figure 36]

Sana +Hunyuan3D 2.1

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Sana 1.5 +Hunyuan3D 2.1

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Show-o +Hunyuan3D 2.1

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

LLama-Gen +Hunyuan3D 2.1

[Figure 49]

[Figure 50]

ShapeLLM-

### omni NA

### NA

[Figure 51]

[Figure 52]

### NA

### NA

Cube3D

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Ours

- Figure 6 Qualitative comparison of text-to-3D shape generation across two compositional prompts. Rows 1–4: Two-stage Text→Image→3D methods, each showing the generated image (left) and reconstructed mesh as rendered image and normal map (right). Rows 5–6: Native text-to-3D baselines, which generate only meshes without intermediate images (marked “NA”). Row 7: Omni123 (Ours), which natively generates both a rendered image and a 3D mesh within a single autoregressive sequence.

#### 6 Experiments

We evaluate Omni123 on text-to-3D shape generation (Section 6.1), instruction-based 3D editing (Section 6.2), and image tokenizer reconstruction quality (Section 6.3). Unless otherwise stated, all Omni123 results use greedy decoding with Classifier-Free Guidance scale ω = 5.0.

##### 6.1 Text-to-3D Shape Generation

As a cornerstone of 3D content generation, structurally consistent and high-quality shape generation provides the essential geometric scaffolding required for subsequent pipelines, including texturing, rendering, and physics simulation. Here, we evaluate the text-to-shape generation performance of Omni123, focusing on geometric fidelity and semantic alignment with input prompts.

Metrics. To evaluate semantic alignment, we uniformly sample 8,192 surface points from each generated mesh and compute point-text similarities using two established multi-modal models: ULIP [86](reporting ULIP-T) and Uni3d [97](reporting Uni3D-T). We evaluate on 100 diverse text prompts encompassing various object categories and styles, compositional descriptions, and fine-grained attributes.

Baselines. To rigorously isolate the advantages of our interleaved X-to-X training paradigm, we benchmark against two dominant methodologies. (i) Cascaded text-to-imageto-3D pipelines: We pair leading image generators of comparable parameter scales—Sana-1.5 [82] (4.8B), Sana [81] (1.6B), Show-o [83] (1.3B), and LlamaGen [67] (0.775B)—with Hunyuan3D2.1 [28]. These two-stage baselines represent indirect workflows that frequently compromise geometric consistency. (ii) Native text-to-3D models: ShapeLLM-Omni [89], a 7B autoregressive model that extends a pre-trained VLM (Qwen2.5-VL [2]) with 3D generation capabilities, and Cube3D [4], a 1.6B autoregressive model trained natively for text-to-3D generation. Both bypass the two-stage handoff but are constrained by 3D data scarcity and limited cross-modal transfer.

Quantitative Comparison. Table 8 demonstrates that two-stage pipelines consistently underperform native methods in semantic-geometric alignment, primarily bottlenecked simultaneously by the inherent prior gap between the image generator and the 3D lifter and inevitable error accumulation across the cascaded pipeline. Conversely, VLM-based native models avoid this handoff but are constrained by 3D data scarcity, ineffective knowledge transfer, and sub-optimal modality fusion. By leveraging abundant 2D observations as a unified geometric prior, both Omni123 variants surpass these baselines by a clear margin. Notably, our 2B model achieves superior alignment compared to the 7B ShapeLLM-Omni [89], demonstrating that interleaved cross-modal training is a fundamentally more parameter-efficient, scalable route than blindly scaling isolated unimodal generators upon a pretrained VLM backbone.

- Qualitative Comparison. Figure 6 visually supports our quantitative findings on complex compositional prompts. Two-stage pipelines produce compelling intermediate images, but their image-to-3D handoff frequently collapses into artifact-ridden meshes—geometric flattening, disjointed limbs, and loss of fine-grained structure are common failure modes. Native 3D baselines bypass this fragile lifting step, yet their reliance on scarce 3D data limits compositional generalization: complex prompts often result in entangled features and semantically unfaithful geometry. Omni123 avoids both failure modes by generating a high-fidelity 2D image and a structurally sound mesh within a single autoregressive sequence, successfully articulating fine-grained anatomical features and precise multi-object spatial relationships.

##### 6.2 Instruction-Based 3D Editing

As the most direct and natural interface for 3D creation, language instruction-driven 3D editing is a pivotal capability for broad 3D applications. We evaluate Omni123 on Edit3D-Bench [51], utilizing its curated source-target mesh pairs for geometric addition and removal operations. Our

- Table 8 Quantitative comparison of text-to-3D shape generation. We evaluate shape–text alignment using ULIP-T [86] and Uni3D-T [97]. The two-stage methods pair various image generators with Hunyuan3D2.1 [28] as the image-to-3D backbone. Best overall is bolded, second-best is underlined.

Method Params ULIP-T ↑ Uni3D-T ↑ Two-stage: Text →Image → 3D (3D backbone: Hunyuan3D 2.1)

Sana-1.5 [82] 4.8B 0.1516 0.1672 Sana [81] 1.6B 0.1448 0.1589 Show-o [83] 1.3B 0.1447 0.1581 LlamaGen [67] 0.775B 0.0946 0.1067

Native: Text →3D

ShapeLLM-Omni [89] 7B 0.1649 0.2666 Cube3D [4] 1.6B 0.1726 0.2768 Ours 2.2B 0.1832 0.2855

Table 9 Quantitative comparison of image tokenizers on ImageNet-1K [16] 512 × 512 test set.

Method FID ↓ PSNR ↑ SSIM ↑ Continuous Cosmos-CI16 × 16 [1] 0.356 28.60 0.7826 Flux.1 [38] 0.017 37.69 0.963 SDXL [55] 0.188 30.35 0.835 SVG-T2I [65] 1.994 20.60 0.587 Ours VAE 0.017 39.15 0.966 Discrete

Cosmos-DI16 × 16 [1] 1.215 23.76 0.645 LlamaGen [67] 0.839 23.96 0.664 Show-o [83] 1.183 24.20 0.680 Ours VAE + 1D Tokenizer 0.432 24.61 0.759

goal is to perform semantically faithful and structurally articulated modifications to a source mesh closely following language instructions.

Metrics. To quantify editing fidelity, we report Chamfer Distance (CD) and F1 score (F1) between the generated and ground-truth edited meshes. Each test sample requires the model to resolve natural-language directives into precise, localized geometric transformations.

Baselines. We compare against four recent 3D editing methods: ShapeLLM-Omni [89], Steer3D [51], 3DEditFormer [78], and Tailor3D [57]. This selection strategically spans both native autoregressive editors and optimization-based pipelines. While ShapeLLM-Omni features a native text-guided variant, its publicly available weights only support image-based inputs. To ensure maximum fairness and isolate each model’s intrinsic 3D editing and instruction-following capabilities, we intentionally employ the powerful Flux-Kontext [40] to provide high-quality 2D edited image input.

- Quantitative Comparison. Table 10 highlights our superior geometric fidelity. Omni123 achieves the lowest Chamfer Distance (CD) across all tasks, indicating significantly tighter structural alignment with the ground-truth targets. While Steer3D leads in F1 score, we note that Edit3D-Bench is constructed using the same data pipeline as Steer3D’s training set,

Table 10 Comparison on add and remove editing tasks in Edit3D-Bench [51].

Add Remove Avg CD ↓ F1 ↑ CD ↓ F1 ↑ CD ↓ F1 ↑ Ours 0.0736 0.1743 0.0632 0.2259 0.0684 0.2001 ShapeLLM-Omni [89] 0.2546 0.0877 0.2237 0.1166 0.2392 0.1022 Steer3D [51] 0.1404 0.2414 0.0976 0.3044 0.1190 0.2729 3DEditFormer [78] 0.1676 0.1955 0.1342 0.1836 0.1509 0.1896 Tailor3D [57] 0.1661 0.1217 0.1755 0.1352 0.1708 0.1285

Method

Table 11 Quantitative comparison of image tokenizers on ImageNet-1K [16] 512 × 512 test set.

Method FID ↓ PSNR ↑ SSIM ↑ Continuous Cosmos-CI16 × 16 [1] 0.356 28.60 0.7826 Flux.1 [38] 0.017 37.69 0.963 SDXL [55] 0.188 30.35 0.835 SVG-T2I [65] 1.994 20.60 0.587 Ours VAE 0.017 39.15 0.966 Discrete

Cosmos-DI16 × 16 [1] 1.215 23.76 0.645 LlamaGen [67] 0.839 23.96 0.664 Show-o [83] 1.183 24.20 0.680 Ours VAE + 1D Tokenizer 0.432 24.61 0.759

giving it a distributional advantage on this benchmark. Despite this, our consistent lead in CD demonstrates more precise execution of global geometric transformations and topological changes. These results suggest that the cross-modal representations learned through interleaved training translate effectively to editing tasks, enabling faithful instruction following even with relatively limited fine-tuning data (116K pairs).

##### 6.3 Image Tokenizer Evaluation

As the representational foundation of our unified framework, a high-fidelity image tokenizer dictates essential structural and textural constraints and mediates the cross-modal information flow. We evaluate our two-stage tokenizer (Section 4.3) on the ImageNet-1K [16] 512 × 512 test set, benchmarking both the continuous VAE (Stage 1) and the full discrete 1D pipeline.

Metrics. We evaluate the tokenizer’s performance across three hierarchical dimensions: FID to measure the distributional alignment between reconstructed and ground-truth image manifolds, ensuring results are perceptually natural, realistic, and statistically sound; PSNR to quantify pixel-level fidelity; and SSIM to assess structural similarity and geometric integrity.

- Quantitative Comparison. Table 11 highlights our superior reconstruction fidelity. In the continuous regime, our VAE surpasses the renowned Flux.1 in fidelity and structural consistency using only ImageNet training set. This underscores the exceptionally powerful prior inherited from the pretrained representation encoder (DINOv3 [66]). Similarly, our discrete pipeline significantly outperforms recent state-of-the-art tokenizers by a large margin across all metrics. These results validate our two-stage strategy as an effective means of minimizing

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

- (a)
- (b)
- (c)
- (d)

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

GT Ours Cosmos-DI16x16 Show-O LLamaGen

- Figure 7 Qualitative comparison of discrete image tokenizers on web images of 512 × 512 resolution.Our tokenizer demonstrates superior reconstruction quality in (a) object-centric images, (b) scene-level details, (c) human details, and (d) text rendering.

quantization degradation while ensuring a robust representational space.

- Qualitative Comparison. Figure 7 illustrates the outstanding reconstruction quality achieved by our image tokenizer. While competing methods often suffer from significant blurriness and semantic distortion, Omni123 demonstrates a clear superiority in preserving high-frequency details and structural integrity. Notably, our model achieves highly faithful text rendering and logo reconstruction(Rows a, c, and d), which are particularly challenging for discrete tokenizers. Moreover, in complex scenes (Row b) and human portraits (Row c), our tokenizer preserves sharp geometric boundaries and identity-consistent facial features. These qualitative results are well aligned with the quantitative comparisons, further confirming that our 1D image tokenizer effectively retains the geometric and semantic priors necessary to provide high-fidelity guidance for joint 3D synthesis.

#### 7 Conclusion

We presented Omni123, a unified multimodal generative model that addresses 3D data scarcity by unifying text-to-2D and text-to-3D generation within a single autoregressive framework. Our three-stage training recipe—cross-modal pre-training over heterogeneous pairs, continued training with learnable view tokens for viewpoint awareness, and Text–Image–3D interleaved fine-tuning that closes the “semantic–visual–geometric” loop—enables abundant 2D observations to serve as an implicit structural constraint for 3D learning. Experiments on text-guided 3D generation and instruction-based 3D editing to validate that this unified approach significantly improves geometric fidelity and semantic alignment, offering a scalable pathway toward multimodal 3D world models. Current limitations include the fixed mesh resolution of the 3D tokenizer and reliance on canonical viewpoints; future work will explore adaptive-resolution tokenization, scene-level generation, and material and physics modeling.

Acknowledgments. We thank Xiaohe Ma, Bing Yu, Jianfeng Xiang for insightful discussions and valuable suggestions. We thank the Meshy AI team for their contributions to infrastructure, data curation, and evaluation.

[Figure 77]

###### Figure 8 Additional qualitative showcase I. More examples of joint image-and-3D generation by Omni123.

[Figure 78]

###### Figure 9 Additional qualitative showcase II. More examples of joint image-and-3D generation by Omni123.

#### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.
- [3] Sangwon Baik, Hyeonwoo Kim, and Hanbyul Joo. Learning 3d object spatial relationships from pre-trained 2d diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8418–8428, 2025.
- [4] Kiran Bhat, Nishchaie Khanna, Karun Channa, Tinghui Zhou, Yiheng Zhu, Xiaoxia Sun, Charles Shang, Anirudh Sudarshan, Maurice Chu, Daiqing Li, et al. Cube: A roblox view of 3d intelligence. arXiv preprint arXiv:2503.15475, 2025.
- [5] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [7] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. PixArt-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [9] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [10] Yongwei Chen, Tianyi Wei, Zhaoyang Lyu, Shangchen Zhou, Xudong XU, Xingang Pan, et al. Lucid-3d: A lightweight and compatible framework for unified 3d understanding and generation. ICLR, 2025.
- [11] Michael Cruz, Christopher Lee, Saurabh Aggarwal, and Robert Martin. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. 2023.
- [12] Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxspace: Disentangled semantic editing in rectified flow transformers. arXiv preprint arXiv:2412.09611, 2024.
- [13] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36: 35799–35813, 2023.

- [14] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023.
- [15] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining, 2025. URL https://arxiv. org/abs/2505.14683, 2(6).
- [16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [17] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [18] Sinan Du, Jiahao Guo, Bo Li, Shuhao Cui, Zhengzhuo Xu, Yifu Luo, Yongxian Wei, Kun Gai, Xinggang Wang, Kai Wu, et al. Vqrae: Representation quantization autoencoders for multimodal understanding, generation and reconstruction. arXiv preprint arXiv:2511.23386, 2025.
- [19] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [20] Xiang Fan, Anand Bhattad, and Ranjay Krishna. Videoshop: Localized semantic video editing with noise-extrapolated diffusion inversion. In European conference on computer vision, pages 232–250. Springer, 2024.
- [21] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [22] Google. A new era of intelligence with gemini 3. https://blog.google/ products-and-platforms/products/gemini/gemini-3/, 2025. Official Google blog post, accessed 2026-03-31.
- [23] Google. Nano banana 2 – gemini ai image generator & photo editor, 2026. URL https: //gemini.google/overview/image-generation/.
- [24] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [26] Hanzhe Hu, Tianwei Yin, Fujun Luan, Yiwei Hu, Hao Tan, Zexiang Xu, Sai Bi, Shubham Tulsiani, and Kai Zhang. Turbo3d: Ultra-fast text-to-3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23668–23678, 2025.
- [27] Shimin Hu, Yuanyi Wei, Fei Zha, Yudong Guo, and Juyong Zhang. Easy3e: Feed-forward 3d asset editing via rectified voxel flow. arXiv preprint arXiv:2602.21499, 2026.

- [28] Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, et al. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025.
- [29] Team Hunyuan3D, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Penghao Wang, et al. Hunyuan3d-omni: A unified framework for controllable generation of 3d assets. arXiv preprint arXiv:2509.21245, 2025.
- [30] Team Hunyuan3D, Bowen Zhang, Chunchao Guo, Dongyuan Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jiaao Yu, Jiachen Xu, Jingwei Huang, et al. Hy3d-bench: Generation of 3d assets. arXiv preprint arXiv:2602.03907, 2026.
- [31] Hyper3D. Rodin – ai 3d model generator, 2026. URL https://hyper3d.ai/rodin/page/1.
- [32] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021.
- [33] Hyeonho Jeong, Jinho Chang, Geon Yeong Park, and Jong Chul Ye. Dreammotion: Spacetime self-similar score distillation for zero-shot video editing. In European Conference on Computer Vision, pages 358–376. Springer, 2024.
- [34] Mingkai Jia, Mingxiao Li, Liaoyuan Fan, Tianxing Shi, Jiaxin Guo, Zeming Li, Xiaoyang Guo, Xiao-Xiao Long, Qian Zhang, Ping Tan, et al. Dino-tok: Adapting dino for visual tokenizers. arXiv preprint arXiv:2511.20565, 2025.
- [35] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10124–10134, 2023.
- [36] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.
- [37] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36:21487–21506, 2023.
- [38] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [39] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.
- [40] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.
- [41] Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, et al. Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint arXiv:2506.16504, 2025.
- [42] Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, et al. Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747, 2025.

- [43] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [44] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [45] Pengyang Ling, Lin Chen, Pan Zhang, Huaian Chen, Yi Jin, and Jinjin Zheng. Freedrag: Feature dragging for reliable point-based image editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6860–6870, 2024.
- [46] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [47] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36:22226–22246, 2023.
- [48] Zhiheng Liu, Weiming Ren, Haozhe Liu, Zijian Zhou, Shoufa Chen, Haonan Qiu, Xiaoke Huang, Zhaochong An, Fanny Yang, Aditya Patel, et al. Tuna: Taming unified visual representations for native unified multimodal models. arXiv preprint arXiv:2512.02014, 2025.
- [49] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [50] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. Advances in Neural Information Processing Systems, 36:75307–75337, 2023.
- [51] Ziqi Ma, Hongqiao Chen, Yisong Yue, and Georgia Gkioxari. Feedforward 3d editing via text-steerable image-to-3d. arXiv preprint arXiv:2512.13678, 2025.
- [52] Meshy. Meshy ai - the #1 ai 3d model generator, 2026. URL https://www.meshy.ai/.
- [53] OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/ introducing-4o-image-generation/.
- [54] Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 conference proceedings, pages 1–11, 2023.
- [55] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [56] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [57] Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. arXiv preprint arXiv:2407.06191, 2024.

- [58] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023.
- [59] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normaldepth diffusion model for detail richness in text-to-3d. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9914–9925, 2024.
- [60] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [61] Yufan Ren, Zicong Jiang, Tong Zhang, Søren Forchhammer, and Sabine Süsstrunk. Fds: Frequency-aware denoising score for text-guided latent diffusion image editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2651–2660, 2025.
- [62] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [63] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [64] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization.(2025). URL https://arxiv. org/abs/2412, 2692, 2025.
- [65] Minglei Shi, Haolin Wang, Borui Zhang, Wenzhao Zheng, Bohan Zeng, Ziyang Yuan, Xiaoshi Wu, Yuanxing Zhang, Huan Yang, Xintao Wang, et al. Svg-t2i: Scaling up text-to-image latent diffusion model without variational autoencoder. arXiv preprint arXiv:2512.11749, 2025.
- [66] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.
- [67] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [68] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.
- [69] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14940–14950, 2025.
- [70] Yiwen Tang, Zoey Guo, Kaixin Zhu, Ray Zhang, Qizhi Chen, Dongzhi Jiang, Junli Liu, Bohan Zeng, Haoming Song, Delin Qu, et al. Are we ready for rl in text-to-3d generation? a progressive investigation. arXiv preprint arXiv:2512.10949, 2025.

- [71] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [72] Tripo. Tripo | easy 2d to 3d: Free ai 3d models from images or texts, 2026. URL https://www.tripo3d.ai/.
- [73] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [74] Haoyuan Wang, Zhenwei Wang, Xiaoxiao Long, Cheng Lin, Gerhard Hancke, and Rynson WH Lau. Mage: Single image to material-aware 3d via the multi-view g-buffer estimation model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10985– 10995, 2025.
- [75] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [76] Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pages 893–911, 2023.
- [77] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [78] Ruihao Xia, Yang Tang, and Pan Zhou. Towards scalable and consistent 3d editing. arXiv preprint arXiv:2510.02994, 2025.
- [79] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. arXiv preprint arXiv:2512.14692, 2025.
- [80] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21469–21480, 2025.
- [81] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformer, 2024. URL https://arxiv. org/abs/2410.10629, 4.
- [82] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.
- [83] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [84] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

- [85] Tianwei Xiong, Jun Hao Liew, Zilong Huang, Jiashi Feng, and Xihui Liu. Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18770–18780, 2025.
- [86] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1179–1189, 2023.
- [87] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [88] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.
- [89] Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. Shapellm-omni: A native multimodal llm for 3d generation and understanding. arXiv preprint arXiv:2506.01853, 2025.
- [90] Junliang Ye, Shenghao Xie, Ruowen Zhao, Zhengyi Wang, Hongyu Yan, Wenqiang Zu, Lei Ma, and Jun Zhu. Nano3d: A training-free approach for efficient 3d editing without masks. arXiv preprint arXiv:2510.15019, 2025.
- [91] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37:128940–128966, 2024.
- [92] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics (TOG), 42(4):1–16, 2023.
- [93] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024.
- [94] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [95] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [96] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. arXiv preprint arXiv:2401.03407, 2024.
- [97] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. arXiv preprint arXiv:2310.06773, 2023.
- [98] Yongxin Zhu, Bocheng Li, Yifei Xin, Zhihua Xia, and Linli Xu. Addressing representation collapse in vector quantized models with one linear layer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22968–22977, 2025.

