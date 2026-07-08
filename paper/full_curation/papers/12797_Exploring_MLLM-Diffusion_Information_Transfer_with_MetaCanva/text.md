# arXiv:2512.11464v1[cs.CV]12Dec2025

## Exploring MLLM-Diffusion Information Transfer with MetaCanvas

Han Lin1,2,∗, Xichen Pan1,3, Ziqi Huang1,4,∗, Ji Hou1, Jialiang Wang1, Weifeng Chen1,∗, Zecheng He1,∗, Felix Juefei-Xu1, Junzhe Sun1, Zhipeng Fan1, Ali Thabet1, Mohit Bansal2, Chu Wang1

1Meta Superintelligence Labs, 2UNC Chapel Hill, 3New York University, 4Nanyang Technological University

∗Work done at Meta

Multimodal learning has rapidly advanced visual understanding, largely via multimodal large language models (MLLMs) that use powerful LLMs as cognitive cores. In visual generation, however, these powerful core models are typically reduced to global text encoders for diffusion models, leaving most of their reasoning and planning ability unused. This creates a gap: current multimodal LLMs can parse complex layouts, attributes, and knowledge-intensive scenes, yet struggle to generate images or videos with equally precise and structured control. We propose MetaCanvas, a lightweight framework that lets MLLMs reason and plan directly in spatial and spatiotemporal latent spaces and interface tightly with diffusion generators. We empirically implement MetaCanvas on three different diffusion backbones and evaluate it across six tasks, including text-to-image generation, text/image-to-video generation, image/video editing, and in-context video generation, each requiring precise layouts, robust attribute binding, and reasoning-intensive control. MetaCanvas consistently outperforms global-conditioning baselines, suggesting that treating MLLMs as latent-space planners is a promising direction for narrowing the gap between multimodal understanding and generation.

Date: December 15, 2025 Correspondence: hanlincs@cs.unc.edu, wangchu@meta.com Website: https://metacanvas.github.io

1 Introduction

Multimodal learning has primarily focused on multimodal understanding (Liu et al., 2023; Tong et al., 2024a) and multimodal generation (Rombach et al., 2021; Guo et al., 2023). Recent progress in multimodal understanding typically leverages Large Language Models (LLMs) (Touvron et al., 2023) as cognitive cores for building multimodal LLMs. These models achieve strong performance in visual question answering (Antol et al., 2015) and visual reasoning (Johnson et al., 2017), enabled by visual instruction tuning (Liu et al., 2023) that aligns them with perception encoders (Radford et al., 2021; Zhai et al., 2023). Using LLMs for multimodal understanding is highly effective because LLMs already excel at reasoning and question answering, and emerging evidence shows that they even acquire visual priors from language pretraining (Han et al., 2025).

However, in multimodal generation, LLMs are still largely treated as conditional encoders that map text prompts into representations for visual generative models (Saharia et al., 2022), leaving most of their core capabilities underutilized. As a result, these models often struggle to produce visual content with accurate object positions and attributes (Ghosh et al., 2023), or to handle cases requiring world-knowledge reasoning (Niu et al., 2025). In contrast, multimodal LLMs can easily interpret such structured visual content. This mismatch between the strengths of multimodal understanding and the limitations of multimodal generation has sparked interest in studying how these two abilities might complement each other (Tong et al., 2024b), and how to more effectively transfer the rich capabilities of MLLMs into multimodal generative models.

Recent advances typically study this through the LLM+Diffusion paradigm, pairing pretrained (M)LLMs with external diffusion models that act as visual decoders (Pan et al., 2025; Tong et al., 2024b; Liu et al., 2025; Lin et al., 2025b; Wu et al., 2025b; Lin et al., 2025a; Koh et al., 2023; Ge et al., 2024; Chen et al., 2025a; Pan et al., 2024; Yin et al., 2025). In these two-component systems, the (M)LLMs first processes all contextual signals

that guide generation, and their outputs are then injected into a separate diffusion generator via attention. Specifically, MetaQuery (Pan et al., 2025) provides early evidence that learnable query embeddings from frozen LLMs can serve as effective conditions, with improved performance in transferring LLM knowledge and in-context learning capabilities to reasoning-augmented image generation.

The optimal interface between (M)LLMs and diffusion models remains unclear. Existing approaches typically use the output of (M)LLMs as a global condition for visual generation (Pan et al., 2025; Lin et al., 2025a; Wu et al., 2025b). Although this provides useful guidance to the diffusion decoder, it constrains the ability of (M)LLMs to deliver fine-grained spatial or temporal controls throughout the diffusion process. In practice, the control required for visual generation is inherently structured and region-specific (Zhang et al., 2023b). Global conditions alone struggles to compel (M)LLMs to explicitly specify where objects should appear, how spatial relations should be grounded, or how temporal dependencies should evolve. In this paper, we wish to investigate whether (M)LLMs can instead reason and plan directly within spatial and spatiotemporal latent spaces, and whether these capabilities would benefit multimodal generation. To do so, we explore an alternative approach: enforcing explicit, patch-by-patch control over the generation process using a learnable latent canvas whose layout is planned by the (M)LLM.

Concretely, we present MetaCanvas, a lightweight architecture that connects pretrained MLLMs with diffusionbased visual generators (see Figure 1 for an overview). The key idea is to have the MLLM produce implicit visual sketches of the desired output, serving as spatial or spatiotemporal priors for guiding the diffusion process. Specifically, MetaCanvas appends a set of learnable, multidimensional canvas tokens to the MLLM’s multimodal input sequence and processes them using multimodal RoPE (Bai et al., 2025b). The MLLM outputs embeddings for these canvas tokens, which are then passed into a lightweight connector module (see Figure 2 and Section 3.4 for design details). This connector comprises two Transformer blocks that align the canvas tokens and inject them into the diffusion model’s noisy latents patch by patch. To stabilize training, we follow the zero-initialization strategy from (Zhang et al., 2023b), ensuring the injected signals initially leave the diffusion model’s inputs unchanged.

We demonstrate the effectiveness of MetaCanvas through extensive experiments. First, in Section 5.1, we conduct an exploratory small-scale text-to-image (T2I) generation experiment. We visually validate that canvas tokens extracted from the MLLM can serve as reasonable visual planners to guide image synthesis in DiT (see Figure 5), and quantitatively observe that MetaCanvas converges faster and consistently outperforms other design variants on the GenEval (Ghosh et al., 2023) benchmark. Motivated by this positive signal, in Section 5.2, we extend MetaCanvas to image editing tasks and scale up the architectures to Qwen2.5VL-7B (Bai et al., 2025b) paired with FLUX.1-Kontext-Dev (Batifol et al., 2025), and observe consistent gains in both training efficiency and editing benchmark performance (see Table 2 and Table 1). To evaluate the generalizability of MetaCanvas across tasks, we extend it to video generation and editing in Section 5.3. By adopting a multi-dimensional canvas design (Section 3.3), we integrate Qwen2.5-VL-7B with Wan2.25B (Wan et al., 2025) through a multi-stage training procedure. This design unifies diverse tasks, including text-to-video (T2V), image-to-video (I2V), and video editing (V2V), and naturally enables new capabilities such as reference-guided video generation, allowing flexible multimodal inputs (e.g., text combined with reference images or videos) while fully preserving the MLLM’s core multimodal understanding. Notably, while unlocking multiple tasks, MetaCanvas preserves the original video generation capabilities (see Table 4) and achieves substantial improvements in V2V prompt accuracy and overall video quality compared to the latest open-source methods (Team, 2025; Wu et al., 2025c; Bai et al., 2025a), attaining the highest overall evaluation scores in both VLM-based and human assessments (see Table 5). Finally, we also demonstrate that MetaCanvas generalizes to in-context video generation tasks, achieving competitive or superior performance compared to previous methods (see Table 7).

In short, our contribution can be summarized as follows:

- • Latent canvas connector for MLLM-to-diffusion. MetaCanvas uses learnable canvas tokens as implicit visual sketches and a lightweight connector to inject them patch-wise into diffusion latents, providing a simple and effective interface.
- • Spatial–temporal planning in latent space. MetaCanvas’s multidimensional canvas queries enable the MLLM to plan object layouts and temporal evolution directly in latent space, improving structural fidelity and providing fine-grained editing control.

[Figure 1]

Diffusion Transformer

[Figure 2]

[Figure 3]

MLP Connector

Canvas Connector

[Figure 4]

[Figure 5]

VAE Encoder

VAE Encoder

Pre-trained MLLM

LoRA

w/ MRoPE

|[Figure 6]<br><br>[Figure 7]|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Replace the words in to ‘MetaCanvas’ Make it Children's Book Illustration

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

Multimodal Instructions

Input Image/Video Target Image/Video

Learnable 2D/3D Canvas Tokens

- Figure 1 Overview of the MetaCanvas framework. MetaCanvas tokenizes the text and encodes it using the MLLM’s text embedder, while user-provided images and videos are encoded using both the MLLM’s visual encoder and the VAE encoder. The text embeddings produced by the MLLM are passed through a lightweight MLP connector and used as conditioning for the DiT. In addition, we append a set of learnable multidimensional canvas tokens to the MLLM input, which are processed using multimodal RoPE (Bai et al., 2025b). The resulting canvas embeddings are then fused with the noisy latents through a lightweight transformer-based connector with two blocks. Connector details are illustrated in Figure 2. Green tokens represent media context tokens, blue tokens represent text context tokens, and purple tokens represent the canvas tokens.

• Performance and generalizability. MetaCanvas generalizes across six diverse image and video tasks and three diffusion backbones (i.e., MMDiT and cross-attention). It surpasses baseline without canvas tokens, and matches state-of-the-art performance while preserving the MLLM’s multimodal understanding.

#### 2 Related Work

Text as the interface to generators. The most direct way to connect LLMs with visual generators is to use the LLM’s text output as the interface: the model expands a prompt into detailed captions, layouts, or scripts that condition a diffusion model (Li et al., 2023; Yang et al., 2023; Feng et al., 2023; Cho et al., 2023; Lin

- et al., 2024; Zala et al., 2024; Lian et al., 2024, 2023; Huang et al., 2025; Qu et al., 2023). However, text-only signals provide limited guidance, they cannot reliably encode dense layouts, precise attribute bindings, or long-range temporal structure required for complex, compositional scenes. A more expressive alternative is to condition generation using dense embeddings from MLLMs.

Embeddings from (M)LLMs as the interface. To increase the bandwidth between (M)LLMs and diffusion backbones, recent methods directly feed dense embeddings from (M)LLMs into the generator. The (M)LLM processes text, images, or videos and either autoregressively predicts visual representations (Sun et al., 2023; Tong et al., 2024b) or encodes query tokens into continuous embeddings (Ge et al., 2024; Dong et al., 2023; Pan et al., 2025; Chen et al., 2025a). This interface is more expressive than pure text and largely preserves the (M)LLM’s reasoning ability, but it still compresses geometry, layout, and motion into a single 1D sequence. Because these embeddings are used only as a global, sequence-level condition, the diffusion model lacks explicit spatial or temporal handles for generation. This motivates introducing a structured spatial or spatiotemporal latent canvas into which the (M)LLM can directly write patch-wise plans for the diffusion generator to follow.

#### 3 MetaCanvas

We introduce MetaCanvas, a novel unified multimodal learning framework that effectively bridges pretrained MLLMs and diffusion models with learnable multi-dimensional canvas tokens and a lightweight connector, which improves information transfer between MLLMs and diffusion models with efficient training. We first introduce preliminaries about multimodal conditioning via MLLMs in Section 3.1, then discuss the overall framework in Section 3.2, followed by canvas tokens design in Section 3.3 and connector design in Section 3.4.

DiT

[Figure 22]

Linear Proj (Zero Init.)

Linear Proj (Zero Init.)

Timestep

FFN

Mix-FFN

###### Vanilla Transformer Block

AdaLN

Self-Attn

Linear-Attn DiT Block

[Figure 23]

Patchify

MLLM

w/ MRoPE

[Figure 24]

[Figure 25]

Learnable Canvas Tokens

Noisy Latents

- Figure 2 MetaCanvas connector design details. The connector comprises a vanilla Transformer block and a Diffusion Transformer (DiT) block. The vanilla Transformer block transforms the learnable canvas tokens to align them with the DiT latent space. The second DiT block adopts a ControlNet-style design, where the transformed canvas tokens and the noisy latents are first combined and then passed through a DiT block with Adaptive LayerNorm (Perez et al., 2018). We adopt Linear-Attn and Mix-FFN design from (Xie et al., 2024a) to reduce memory usage. The outputs of both blocks are followed by a zero-initialized linear projection layer, ensuring that at the beginning of training, the learnable canvas tokens have no influence on the DiT’s latent inputs.

- 3.1 Preliminaries

Multimodal conditioning via MLLM. In bridging methods that connect MLLMs and diffusion models, the text conditioning tokens ctext in the latent diffusion model are replaced by the output tokens from the MLLM. Specifically, the input modalities including text, images and videos are first encoded by the matching tokenizer, then passed through the MLLM backbone. The resulting text (and visual) tokens are used as conditions to guide the diffusion model. In MetaCanvas, we further append a set of learnable canvas tokens to the MLLM’s multimodal input sequence and process them jointly. The output embeddings of these canvas tokens are used as structural visual-semantic priors and are added to the noisy latents zt to provide additional and direct diffusion canvas guidance during generation.

- 3.2 Overall Framework for MetaCanvas

We illustrate the overall MetaCanvas framework in Figure 1. Given user instructions with text and visual inputs, MetaCanvas tokenizes the text and encodes it with the MLLM’s text embedder, while images or videos are encoded by the MLLM’s visual encoder and also by the diffusion model’s VAE encoder for both semantic and low-level details.

Context token conditioning. Embeddings of the multimodal input tokens produced by the MLLM (i.e., context tokens for generation) are passed through a lightweight two-layer MLP connector and given to the DiT via cross-attention or self-attention, depending on the DiT’s architecture, following standard practice in (Pan

- et al., 2025; Tong et al., 2024b; Liu et al., 2025; Lin et al., 2025b,a; Ge et al., 2024; Chen et al., 2025a). Further discussion is provided in Appendix D.1.

Canvas token conditioning. We append a set of learnable multidimensional canvas tokens to the MLLM input, which are processed using multimodal RoPE (Bai et al., 2025b). The resulting canvas embeddings are then fused additively with the noisy latents through a lightweight Canvas Connector (see Section 3.4), ensuring strong alignment between the MLLM-drafted canvas and the actual generation latents.

Training and inference. The main learnable components of MetaCanvas include the MLP connector, the canvas connector, and the DiT (LoRA-finetuned in Section 5.1, finetuned only on the visual branch in Section 5.2, or fully finetuned in Section 5.3). During training, the MLLM is kept frozen, optionally with an additional LoRA to enhance its visual generation capacity (ablations in Appendix B.5). During inference, canvas tokens are appended to the MLLM only after the end-of-sentence <EoS> token. If a LoRA is enabled, it remains

Mask

Reference Latents Condition Latents Target Latents MetaCanvas Keyframe Latents

Patchify

C

DiT

[Figure 26]

- Figure 3 MetaCanvas keyframes and reference/condition frames injection strategy for video tasks. We modify the input layer of Wan2.2-5B (Wan et al., 2025) to concatenate reference and condition latents with noisy latents along the channel dimension. The resulting tensor is then passed through the patch embedding layer and combined with MetaCanvas keyframes after interpolation. Light purple tokens represent interpolated keyframe canvas. Note that we do not apply MetaCanvas keyframe latents to reference frames for video tasks.

inactive until <EoS> and is activated only after the canvas tokens are appended. This design preserves the MLLM’s understanding and reasoning capabilities while enabling strong generative and steering performance through lightweight, low-cost trainable components.

- 3.3 Multi-dimensional Canvas Tokens

Keyframe canvas design. For image generation and editing tasks, MetaCanvas employs 2D canvas tokens whose spatial arrangement adapts to the image resolution. For videos, instead of using a dense 3D canvas covering all latent frames, we introduce learnable sparse keyframe canvas tokens that capture representative temporal information and are then linearly interpolated to the full latent-frame space before being incorporated into the DiT latents. This strategy preserves temporal coherence while keeping the token interface compact and efficient for both training and inference. Additional details regarding the image and video canvas design are provided in Appendix D.3.

Input layer adaptation for video tasks. As illustrated in Figure 3, and following (Cheng et al., 2025), we modify the input layer of Wan2.2-5B (Wan et al., 2025) to concatenate the reference and condition latents with the noisy latents along the channel dimension. These latents are then passed through the patch embedding layer and combined with the MetaCanvas keyframe tokens after interpolation (detailed below).

- 3.4 MetaCanvas Connector Design

The connector for learnable canvas tokens is designed with the following components. Ablation studies on the effectiveness of these components are shown in Table 3.

- • Vanilla Transformer block. We first pass the learnable canvas tokens, being 2D canvas for image case or 3D keyframe canvas for video case, through a single Transformer block to transfer and align their features with the DiT latent space. Note that for video tasks (Section 3.3), we will additionally perform linear interpolation of the output tokens from this block to match the noisy latents shape of the DiT.
- • DiT block. Next, we add the canvas tokens to the DiT noisy latents after the patchify layer. For video tasks, as shown in Figure 3, we only add canvas tokens to the noisy latent frames while avoiding any modification of the reference frames. The fused canvas tokens are then processed using an efficient linear-attention DiT block in SANA (Xie et al., 2024a) and ultimately added back to the original DiT latent space. We adopt AdaLN (Perez et al., 2018) in the connector design to dynamically modulate the influence of the canvas tokens across different timesteps.
- • Zero-initialized linear projections. After each Transformer block, we apply a zero-initialized linear projection, inspired by (Zhang et al., 2023b), to ensure that at the start of training the canvas branch produces a null residual and does not modify the DiT latent inputs.
- • Patchify then fuse. We find that fusing canvas tokens with the noisy latents after patchification yields better performance, as it avoids projecting high-dimensional canvas tokens into the lower-dimensional VAE space, which would otherwise result in information loss.

#### 4 Experiment Setup

- 4.1 Training Setup

Exploratory experiment on T2I generation task. To quickly test MetaCanvas and assess the connector design, we begin with exploratory experiments using Qwen2.5-VL-3B (Bai et al., 2025b) and SANA-1.6B (Xie et al., 2024a). We retain SANA’s original T5 text encoder for text conditioning and add additional token conditions using different methods from Qwen2.5-VL-3B to evaluate information transfer from the MLLM to the DiT. For canvas tokens, we adopt the transformer-based connector described in Section 3.4. For MetaQuery-style 1D query tokens, we reuse the text-conditioning interface and concatenate the MLLM’s query tokens with the T5 text tokens as a joint condition to the DiT. We keep the MLLM frozen, and train the connectors, canvas/query tokens, as well as the DiT from scratch. We use BLIP3o-60k (Chen et al., 2025a) as the training data. Training hyperparameters are listed in Appendix B.

Image editing task. As FLUX.1-Dev (Labs, 2024) and FLUX.1-Kontext-Dev (Batifol et al., 2025) share the same T5 text encoder, we follow GPT-Image-Edit (Wang et al., 2025) and initialize from the MLP connector in UniWorld-V1 (Lin et al., 2025a), which was trained to bridge Qwen2.5-VL-7B (Bai et al., 2025b) and FLUX.1 [Dev] (Labs, 2024). We start the training directly by unfreezing the diffusion model’s visual branch, together with the additional learnable canvas tokens and the transformer-based connector. To increase MLLM’s model capacity, we apply LoRA with rank of 64 to the MLLM backbone. The model is trained on O(1M) image-editing samples. See Appendix C for training details.

Video generation and editing tasks. Wan2.2-5B (Wan et al., 2025) uses T5 text encoder by default. To better support multimodal inputs, we replace T5 with a MLLM (Qwen2.5-VL-7B (Bai et al., 2025b)) rather than fusing MLLM and T5. Hybrid MLLM+T5 features can introduce conflicting optimization signals (Lin et al., 2025a), and early reliance on T5 often drives training into poor local minima, whereas a single MLLM pathway yields more stable convergence and better alignment. We employ a three-stage training strategy (see Appendix H for details):

- • Stage 1: Connector alignment. In the first stage, we use O(40M) image-text pairs mainly focusing on image-only training, with the goal of aligning the image semantics between the two models. As we train on relatively low resolution images, we freeze the whole MLLM and diffusion model, and only train the connector.
- • Stage 2: High-resolution finetuning. In the second stage, we further incorporate O(3M) video-text pairs, and set the video:image data ratio as 4:1, with standard resolutions and video length of 121 frames, to let the model learn motions. We unfreeze the cross-attention layers in diffusion models in this stage for better alignment.
- • Stage 3: Multitask training. In the third stage, we train on diverse image and video tasks, unfreeze all parameters in diffusion models, and add LoRA on MLLM as well as trainable parameters in MetaCanvas connectors. We train the model jointly on diverse image and video tasks. Task-specific dataset details are illustrated in the Appendix.

- 4.2 Evaluation Setup We evaluate MetaCanvas on a wide variety of benchmarks:

- • ExploratoryexperomentsonT2Igeneration: Evaluate attributes (count, color, spatial, etc.) on GenEval (Ghosh

- et al., 2023).

- • Image editing: Following prior works (Lin et al., 2025a; Wu et al., 2025a; Deng et al., 2025; Wu et al.,

- 2025b), we report results on GEdit (Liu et al., 2025) and ImgEdit (Ye et al., 2025), using GPT-4o (OpenAI, 2024b) as the evaluator.

- • Video generation: For T2V and I2V generation tasks, we use VBench (Huang et al., 2024a,b) as the evaluation benchmark.
- • Video editing: For video editing task, due to the scarcity of open-source video editing evaluation benchmarks with high resolution (720p) and sufficient duration (121 frames at 24 FPS), we curate a balanced evaluation set consisting of 300 video prompts. Dataset details are illustrated in Appendix F.

70

60

GenEval()

50

40

Text + MetaCanvas

30

Text-Only (SANA Default)

20

Text + MetaQuery

Text + MetaCanvas

MetaQuery

MetaCanvas w/o Text

10

2k 4k 6k 8k 10k 12k 14k 16k 18k 20k

2k 4k 6k 8k 10k 12k 14k 16k 18k 20k

Training Steps

Training Steps

- Figure 4 Left: Comparison of MetaCanvas with MetaQuery (Pan et al., 2025) and text conditioning. Right: Comparison of MetaCanvas with and without additional text conditioning.

Generated

Images Canvas

Features

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

Text Conditioning: VLM ✅ | DiT ❌

A photo of a dog

A photo of an orange toaster

A photo of a broccoli and a vase

a photo of a purple suitcase and an orange pizza

A photo of a clock below a tv

A photo of four giraffes

A photo of four donuts

- Figure 5 Visualization of canvas features (1st row) and generated images (2nd row) using only canvas tokens without extra text conditioning in DiT. We train SANA (Xie et al., 2024a) from scratch using only canvas tokens from Qwen2.5-VL (Wang

- et al., 2024a) as the conditioning input, with no text signals provided to the DiT. Following (Tumanyan et al., 2023), we apply PCA to the features produced by the MetaCanvas connector. Canvas tokens output from MLLM can serve as reasonable visual planning sketches to effectively guide the final image synthesis in the DiT.

• In-context video generation: We evaluate reference image-to-video generation, introducing OmniContextVideo by extending the image-generation benchmark OmniContext (Wu et al., 2025b) to video, covering both single-ID and multi-ID scenarios across objects, characters, and scenes. Dataset details are illustrated in Appendix G.

5 Results and Discussion

We empirically validate MetaCanvas across multiple tasks and diffusion model backbones. Specifically, we begin with exploratory T2I generation experiments as proof of concept (Section 5.1), then implement the connector design on image editing task (Section 5.2), and finally scale up to video generation and editing tasks (Section 5.3).

- 5.1 Exploratory Experiments on T2I Generation

We aim to validate two questions here: Q1: Does MetaCanvas really help guide the generation process of diffusion models? Q2: What connector design is most effective?

Comparison with other design choices. To answer Q1, in Figure 4 (left), we compare MetaCanvas with (1) the default SANA baseline (T5 text conditioning), (2) an architecture equivalent to MetaQuery (Pan

- et al., 2025) that uses 256 learnable 1D query tokens produced by Qwen-2.5-VL while reusing the same text-conditioning interface, and (3) a variant that concatenates T5 text embeddings with the 256 MetaQuery tokens for additional context. As shown, combining text as global guidance with MetaCanvas as a visual

- Table 1 Quantitative comparison with models on ImgEdit (Ye et al., 2025) benchmark. Full table is shown in Table 12.

|Method|Add Adjust Extract Replace Remove Background Style Hybrid Action Overall↑<br><br>|
|---|---|
|Step1X-Edit (Liu et al., 2025) BAGEL (Deng et al., 2025) OmniGen2 (Wu et al., 2025b) GPT-Image-1 [High] (OpenAI, 2025) Qwen-Image-Edit-2509 (Wu et al., 2025a)|3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06<br><br>3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20<br><br>3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44<br><br><br>4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20<br><br><br>4.32 4.36 4.04 4.64 4.52 4.37 4.84 3.39 4.71 4.35|
|FLUX.1 Kontext [Dev] (Batifol et al., 2025) FLUX.1 Kontext [Dev] + MetaCanvas<br><br>|3.76 3.45 2.15 3.98 2.94 3.78 4.38 2.96 4.26 3.52<br><br>4.20 3.50 2.11 4.41 3.72 3.89 4.83 3.61 4.49 3.86 0.44↑ 0.05↑ 0.04↓ 0.43↑ 0.78↑ 0.11↑ 0.45↑ 0.65↑ 0.23↑ 0.34↑<br><br><br>|

- Table 2 Quantitative comparison results on GEdit-EN-full (Liu et al., 2025) benchmark. Best numbers are bolded, and the second best are underlined. Full table is shown in Table 11.

|Model<br><br>|GEdit-Bench-EN↑|
|---|---|
| |G_SC G_PQ G_O|
|UniWorld-V1 (Lin et al., 2025a) Step1X-Edit (v1.1) (Liu et al., 2025) BAGEL (Deng et al., 2025) OmniGen2 (Wu et al., 2025b) GPT-Image-Edit (Wang et al., 2025) Qwen-Image-Edit-2509 (Wu et al., 2025a)<br><br>|4.93 7.43 4.85 7.66 7.35 6.97<br><br>7.36 6.83 6.52<br><br>7.16 6.77 6.41<br>8.00 7.86 7.56<br><br><br>8.80 7.74 7.98<br><br><br>|
|FLUX.1-Kontext-Dev (Batifol et al., 2025) FLUX.1-Kontext-Dev + MetaCanvas<br><br>|6.52 7.38 6.00 8.24 7.68 7.67 1.72↑ 0.50↑ 1.67↑<br><br>|

###### prior yields consistent gains and has the fastest GenEval convergence among all variants. In Figure 4 (right), we further evaluate a no-text variant. Even without any text conditioning, adding 2D learnable canvas tokens on top of the noisy latents in DiT provides meaningful structural guidance, demonstrating effective information transfer from the MLLM to the DiT via MetaCanvas. Visualization of this no-text variant on GenEval examples is shown in Figure 5.

Connector design. We address Q2 with an ablation study on the MetaCanvas connector design in Table 3. Visualizations of these architectural variants are provided in the appendix. We find that conditioning on the timestep enables dynamic control over the influence of canvas tokens on the noisy latents, while the proposed DiT block and accompanying transformer blocks effectively transform and fuse canvas-token information with the latents. Moreover, avoiding early projection of canvas tokens into the low-dimensional VAE space yields additional gains.

Table 3 Ablation study on MetaCanvas connector design.

| |GenEval↑<br><br>|
|---|---|
|MetaCanvas (ours) Remove Timestep Condition in DiT Block Remove DiT Block Remove Vanilla Transformer Block Add Canvas Tokens Before Patchification<br><br>|68.02 (0.00↓) 67.42 (0.60↓) 66.39 (1.03↓) 66.19 (0.20↓) 65.34 (0.85↓)|
|Baseline (Default SANA Architecture)<br><br>|64.09 (0.00↓)|

- 5.2 Results on Image Editing Task

We evaluate the fine-tuned image-editing model FLUX.1-Kontext [Dev] (Batifol et al., 2025) augmented with MetaCanvas against competing methods on ImgEdit-Bench (see Table 1) and GEdit-Bench (see Table 2). Equipping FLUX.1-Kontext [Dev] with MetaCanvas yields consistent improvements on both benchmarks.

- Figure 6 further contrasts the vanilla model with its MetaCanvas-augmented counterpart under the same training setup, showing steady gains throughout training. Notably, these benefits come from adding only lightweight connector modules, incurring minimal parameter and computational overhead.

7.55

0.30

TrainingLoss()

| |
|---|

GEdit-Bench()

| |
|---|

| |
|---|

| |
|---|

| |
|---|

7.10

0.28

0.26

6.65

0.24

FLUX.1-Kontext[Dev] + MetaCanvas

FLUX.1-Kontext[Dev]

6.20

0k 10k 20k 30k 40k

0k 10k 20k 30k 40k

Training Steps

Training Steps

- Figure 6 Comparison of training loss and GEdit-Bench (Liu et al., 2025) scores for the baseline method without canvas tokens and MetaCanvas. Both models are fine-tuned on the same training dataset.

- Table 4 Quantitative comparison results on VBench-I2V (Huang et al., 2024b). Best numbers are bolded, and the second best are underlined. Full table is shown in the Appendix.

I2V Score↑ Quality Score↑ Overall↑

CogVideoX-5B (Kong et al., 2024) 94.79 78.61 86.70 HunyuanVideo (Kong et al., 2024) 95.10 78.54 86.82

- Wan2.1-14B (Wan et al., 2025) 92.90 80.82 86.86
- Wan2.2-5B (Wan et al., 2025) 95.69 78.26 86.98 Wan2.2-5B + MetaCanvas 97.50 76.76 87.13

5.3 Results on Video Generation and Editing Tasks

Video generation. In Table 4, we compare videos generated by MetaCanvas after three training stages (see Section 4.1) with open-source models, including CogVideoX-5B (Yang et al., 2024), HunyuanVideo (Kong

- et al., 2024), Wan2.1-14B (Wan et al., 2025), and the baseline model Wan2.2-5B (Wan et al., 2025). Our method achieves comparable performance while being additionally equipped with strong video editing capabilities.

- Table 5 Quantitative comparison on video editing task. The best and second best numbers are bolded and underlined.

VBench VLM Eval (GPT-4o) Human Preference Rate Quality↑ Sementics↑ Quality↑ Overall↑ Edit Acc.↑ Consistency↑ Overall↑

Model DiT Backbone

Methods w/ Text Encoder InsViE (Wu et al., 2025c) CogVideoX-2B 79.19 3.85 4.48 3.60 - - Lucy-Edit-Dev (Team, 2025) Wan2.2-5B 78.07 5.57 7.19 5.44 - - Lucy-Edit-Dev v1.1 (Team, 2025) Wan2.2-5B 77.70 5.68 6.59 5.43 9.5% 42.7% 26.10% Ditto (Bai et al., 2025a) Wan2.1-14B 80.64 4.90 7.54 5.48 18.4% 7.7% 13.04% Method w/ MLLM as Multimodal Encoder w/o Canvas Tokens (Baseline) Wan2.2-5B 81.39 6.61 7.25 6.68 - - w/ Canvas Tokens (MetaCanvas) Wan2.2-5B 80.27 7.91 7.50 7.56 72.1% 49.6% 60.84%

Video editing. We compare MetaCanvas with recent state-of-the-art models, including InsViE (Wu et al.,

- 2025c), Ditto (Bai et al., 2025a), and Lucy-Edit-Dev (Team, 2025), as well as a control setup of our method that excludes canvas tokens. As shown in Table 5, MetaCanvas achieves comparable video quality scores, as measured by VBench (Huang et al., 2024a,b) and GPT-4o (Hurst et al., 2024), while outperforming all baselines in editing accuracy (i.e., semantics) by a large margin. In addition, we conduct human evaluations comparing Lucy-Edit-DeV v1.1, Ditto, and MetaCanvas, and report the win rates for editing accuracy, spatio-temporal consistency, and overall user preference (see Appendix for details). MetaCanvas achieves the highest preference rate across all evaluation dimensions. Furthermore, the controlled variant without canvas tokens attains competitive or better performance relative to other baselines, demonstrating the effectiveness of replacing the text encoder with a MLLM-based multimodal condition encoder. We provide visualization of these methods together with closed-source model Gen4-Aleph (RunwayML, 2025) in Figure 7. MetaCanvas demonstrates strong prompt understanding and effective grounding of the editing regions, with better spatial alignment and preservation of foreground and background details.

Ablation study on the number of canvas keyframes. Our default design uses three canvas keyframes for video tasks. As an ablation, we compare different numbers of canvas keyframes in Table 6. We find that using three keyframes achieves better performance than using more keyframes, highlighting the simplicity and efficiency

Replace the blue leaf-patterned wall with a soft, textured olive-green wallpaper featuring subtle white floral accents.

Change the projected city background to a rain-soaked, neon-lit street scene in Tokyo with glowing blue signage that matches the room's ambient lighting.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

MetaCanvasGen4-AlephDittoLucy-EditInputVideo

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

- Figure 7 Visualization on video editing task. MetaCanvas demonstrates strong prompt understanding and effective grounding of the editing regions, achieving better spatial alignment and improved preservation of both foreground and background details.

- Table6 Comparison between models with and without Canvas tokens for video editing task, and ablations of MetaCanvas with different number of canvas keyframes. Best and second best numbers are bolded and underlined.

w/o Canvas Canvas w/ Different # Keyframes

Tokens 1 3 6 11 31

Spatial Alignment Flow Err.↓ 6.07 4.92 4.91 5.48 5.25 5.59 PSNR↑ 12.36 13.62 13.65 13.04 13.08 12.54 Video Quality Evaluation VBench↑ 81.39 79.75 80.28 79.92 79.91 80.79 VLM Eval (GPT-4o)

Sementics↑ 6.61 8.01 7.91 7.46 7.60 7.35 Quality↑ 7.25 7.85 7.50 7.25 7.45 7.26 Overall↑ 6.68 7.82 7.56 7.20 7.36 7.12 Train Time / Step (Seconds)

20.75 21.40 21.46 21.77 22.08 23.12

of our design. Although using a single keyframe (i.e., 2D canvas) slightly outperforms three keyframes in MLLM-based evaluation, we empirically observe noticeable temporal flickering in the first few frames, resulting in a lower VBench video quality score (79.75 vs. 80.28). We hypothesize that this is due to the VAE design in Wan2.2-5B, which we describe in more detail in Appendix D.3. Compared with the controlled variant without canvas tokens, adding three canvas keyframes increases the clock time per training step by only 3.1% (from 20.75 to 21.40 seconds per step).

In-context video generation. In Table 7, we compare MetaCanvas with Wan-VACE (Jiang et al., 2025)1.3B/14B on video generation from reference images. MetaCanvas achieves competitive performance with these baselines, particularly on human-object interaction tasks (i.e., character + object under multiple ID categories). Visualization results for MetaCanvas and Wan-VACE-14B are shown in Figure 8 and in the Appendix. We describe our training data curation pipeline in the Appendix, and acknowledge that there remains room for further improvement in our data creation strategy.

[Figure 88]

MetaCanvasReferenceImagesVACE14B

He gently cradles the bird as it perches delicately on his outstretched fingers, its feathers ruffling slightly in the warm breeze of a tranquil savanna. The golden grass sways softly around them under a vast sky painted with soft pastel hues of dawn, while his focused gaze meets the bird’s bright eyes, creating a quiet moment of connection.

+TextInput

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

###### Figure 8 Visualization of in-context video generation. MetaCanvas accurately generates the video by composing reference images in accordance with the text prompt.

Table 7 Quantitative comparison on OmniContext-Video benchmark for in-context video generation from reference images.

Single ID Multiple IDs Scene

Model

Char. Obj. Char. Obj. Char. + Obj. Char. Obj. Char. + Obj. Average↑

Wan-VACE-1.3B (Jiang et al., 2025) 6.44 4.88 4.36 5.66 4.36 3.37 4.88 4.92 4.85 Wan-VACE-14B (Jiang et al., 2025) 6.71 5.79 3.16 5.10 4.64 3.85 4.00 4.64 4.86 Wan2.2-5B (Wan et al., 2025) + MetaCanvas 6.16 6.34 3.77 5.69 6.35 4.60 5.43 4.75 5.40

- 6 Conclusion

In this paper, we present MetaCanvas, an elegant framework that bridges MLLMs and diffusion models through a novel connector design. By introducing learnable multi-dimensional canvas tokens as spatiotemporal priors that fuses multimodal input conditions, MetaCanvas effectively plans and guides media synthesis in a principled manner. Across three backbones and six diverse tasks (image/video generation, editing, and in-context generation with multimodal cues), our experiments show that MetaCanvas delivers strong performance while maintaining training efficiency relative to prior architectures.

References

Meta AI. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation, April 2025. https: //ai.meta.com/blog/llama-4-multimodal-intelligence/. Accessed: 2025-05-12.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015.

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506, 2025.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset, 2025a. https://arxiv.org/abs/2505.09568.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.

Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025.

Jaemin Cho, Abhay Zala, and Mohit Bansal. Visual programming for step-by-step text-to-image generation and evaluation. Advances in Neural Information Processing Systems, 36:6048–6069, 2023.

Google DeepMind. Gemini 2.5 pro, March 2025. https://cloud.google.com/vertex-ai/generative-ai/docs/models/ gemini/2-5-pro. Large language model.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. In The Twelfth International Conference on Learning Representations, 2023.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. arXiv preprint arXiv:2305.15393, 2023.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. In The Twelfth International Conference on Learning Representations, 2023.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Junlin Han, Shengbang Tong, David Fan, Yufan Ren, Koustuv Sinha, Philip Torr, and Filippos Kokkinos. Learning to see before seeing: Demystifying llm visual priors from language pre-training. arXiv preprint arXiv:2509.26625, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024a.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024b.

Ziqi Huang, Ning Yu, Gordon Chen, Haonan Qiu, Paul Debevec, and Ziwei Liu. Vchain: Chain-of-visual-thought for reasoning in video generation. arXiv preprint arXiv:2510.05094, 2025.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit:

A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024. Aaron Hurst et al. Gpt-4o system card. https://arxiv.org/abs/2410.21276, 2024. arXiv:2410.21276 [cs.CL]. Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and

editing. arXiv preprint arXiv:2503.07598, 2025.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. NeurIPS,

- 2023.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603,

- 2024.

Black Forest Labs. Flux.1-dev. https://huggingface.co/black-forest-labs/FLUX.1-dev, 2024. Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee.

Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22511–22521, 2023.

Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. arXiv preprint arXiv:2305.13655, 2023.

Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. In The Twelfth International Conference on Learning Representations, 2024.

Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025a.

Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. In First Conference on Language Modeling, 2024.

Han Lin, Jaemin Cho, Amir Zadeh, Chuan Li, and Mohit Bansal. Bifrost-1: Bridging multimodal llms and diffusion models with patch-level clip latents. arXiv preprint arXiv:2508.05954, 2025b.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2024.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai yu, Liang Zhao, Yisong Wang, Jiaying Liu, and Chong Ruan. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation, 2024.

Chong Mou, Qichao Sun, Yanze Wu, Pengze Zhang, Xinghui Li, Fulong Ye, Songtao Zhao, and Qian He. Instructx: Towards unified visual editing with mllm guidance. arXiv preprint arXiv:2510.08485, 2025.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI. Gpt-4.1. https://platform.openai.com, 2024a. Large language model. OpenAI. Gpt-4o. https://openai.com/, 2024b. Large language model. OpenAI. Introducing our latest image generation model in the api, April 2025. https://openai.com/index/

image-generation-api/. Accessed: 2025-10-30.

Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. In The Twelfth International Conference on Learning Representations, 2024.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Film: visual reasoning with a general conditioning layer. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, pages 3942–3951, 2018.

Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, and Tat-Seng Chua. Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 643–654, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. arxiv 2022. arXiv preprint arXiv:2112.10752, 2021.

RunwayML. Runwayml gen4-aleph, 2025. https://replicate.com/runwayml/gen4-aleph. Video-to-video editing model. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed

Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022. https://arxiv.org/abs/2205.11487.

Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation, 2025. https://arxiv.org/abs/2412.15188.

Yuxin Song, Wenkai Dong, Shizun Wang, Qi Zhang, Song Xue, Tao Yuan, Hu Yang, Haocheng Feng, Hang Zhou, Xinyan Xiao, et al. Query-kontext: An unified multimodal model for image generation and editing. arXiv preprint arXiv:2509.26641, 2025.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

doi: 10.48550/arXiv.2405.09818. https://github.com/facebookresearch/chameleon. DecartAI Team. Lucy edit: Open-weight text-guided video editing. 2025.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024a.

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024b.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-toimage translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1921–1930, 2023.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025.

Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024.

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16692–16701, 2025c.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. Sana: Efficient high-resolution image synthesis with linear diffusion transformer, 2024a. https://arxiv.org/abs/2410.10629.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024b.

Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, et al. Reco: Region-controlled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14246–14255, 2023.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

Aoxiong Yin, Xu Tan, Kai Shen, Yichong Leng, Xinyu Zhou, Juncheng Li, and Siliang Tang. The best of both worlds: Integrating language models and diffusion models for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15604–15615, 2025.

Abhay Zala, Han Lin, Jaemin Cho, and Mohit Bansal. Diagrammergpt: Generating open-domain, open-platform diagrams via llm planning. In First Conference on Language Modeling, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023a.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE International Conference on Computer Vision (ICCV), 2023b.

Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025.

Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024. https://arxiv.org/abs/2408.11039.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. https://arxiv.org/abs/2504.10479.

## Appendix

- A Background 18

- A.1 Extended Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.2 Extended Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- B Exploratory T2I Generation 19

- B.1 Architecture Details of Other Design Choices . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.2 Ablation for Canvas Connector Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.3 Experiment Setup Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.4 Additional Comparison with Query-Based Architecture . . . . . . . . . . . . . . . . . . . . . . 21
- B.5 Additional Ablation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- C Image Editing Task 23
- D MetaCanvas Architecture Design for Video Tasks 24

- D.1 Context and Canvas Connectors Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.2 Interface Design for Canvas Keyframe and Reference/Condition Frames Injection . . . . . . . 25
- D.3 2D canvas vs. 3D canvas . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- E Text/Image-to-Video Generation Tasks 26
- F Video Editing Task 26

- F.1 Training dataset details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- F.2 Evaluation benchmark details. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- G In-Context Video Generation Task 27

- G.1 Training Dataset Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- G.2 Evaluation Benchmark Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- H Video Tasks Training Details 28
- I Visualizations 28
- J Simplified Code Implementation 30
- K Limitations and Future Works 30

#### A Background

- A.1 Extended Related Works

Nativemodelsforvisualgenerationandunderstanding. There has been extensive efforts in recent models to unify visual and text generation and understanding capabilities into a single model. The visual generation capabilites in these models are trained with either the same autoregressive objectives as text, such as Chameleon (Team, 2024), VILA-U (Wu et al., 2024), TokenFlow (Geyer et al., 2023), EMU3 (Wang et al., 2024b), JanusPro (Chen

- et al., 2025b), or through diffusion denoising or flow-matching objective, such as Transfusion (Zhou et al.,

- 2024), BAGEL (Deng et al., 2025), JanusFlow (Ma et al., 2024), Show-o (Xie et al., 2024b), LlamaFusion (Shi et al., 2025), Mogao (Liao et al., 2025). On the positive side, these approaches are designed to maximize modality and task fusion through end-to-end (E2E) training, which usually initialize the backbone from a (M)LLM, or a dual-branch architecture that models text and vision. With potential higher upper bound due to E2E model designs, such methods require significant resources to learn high-quality visual generation from scratch. Unlike native approaches, MetaCanvas bridges the merits of existing MLLMs and diffusion generators without extensive retraining.

Bridging (M)LLMs with diffusion models. Another approach is to bridge a pretrained (M)LLM with a strong pretrained diffusion model, preserving the strengths of the individual components. The simplest variant is text only guidance: the LLM expands or structures the prompt (e.g., detailed captions, layouts), which then conditions the diffusion model (Li et al., 2023; Yang et al., 2023; Feng et al., 2023; Cho et al., 2023; Lin et al.,

- 2024; Zala et al., 2024; Lian et al., 2024). While efficient, pure text is often too sparse to convey the dense visual cues required for complex scenes with high fidelity. A more expressive alternative is to directly use the output token embeddings from the MLLM (Pan et al., 2025; Tong et al., 2024b; Liu et al., 2025; Lin et al.,
- 2025b; Wu et al., 2025b; Lin et al., 2025a; Koh et al., 2023; Ge et al., 2024; Chen et al., 2025a; Pan et al., 2024; Yin et al., 2025; Song et al., 2025; Mou et al., 2025). In this setting, the MLLM provides the final hidden states of the multimodal inputs, or a sequence of learned query embeddings summarizing the multimodal conditioning; these are injected into the diffusion model via various conditioning injection methods, including the self or cross attention (aka text conditioning method), token concatenation, or lightweight adapters like control nets. Representative examples include GILL (Koh et al., 2023), which maps LLM features into the diffusion model text encoder space, and MetaMorph (Tong et al., 2024b), which trains an MLLM to autoregressively produce image condition tokens later consumed by diffusion. MetaQueries (Pan et al., 2025) and BLIP3-o (Chen et al., 2025a) learn a fixed set of query vectors and a transformer based connector that abstract MLLM semantics to condition the generator. Although this paradigm is generally more compute efficient than training a native unified architecture and tends to preserve the MLLM’s reasoning ability, encoding structural visual priors into a single-dimensional (1D) token sequence is non-trivial. Models often require large connector blocks and substantial data to recover precise positional and temporal information. Consequently, 1D tokens can be insufficient to model composition, object placement, geometry and motion, motivating interfaces with explicit multi-dimensional grounding. Bifrost-1 (Lin et al., 2025b) addresses this challenge by leveraging patch-level latents that are natively aligned with the MLLM visual encoder as 2D visual priors (rather than compressing the information into a 1D token sequence) to guide the diffusion model. This design outperforms architectures that rely on non-MLLM-aligned visual features (e.g., VAE, SigLIP) and significantly improves training efficiency. However, fine-tuning the visual generation branch, implemented as a trainable copy initialized from the original MLLM parameters (e.g., Qwen2.5-VL-7B), becomes increasingly challenging as the MLLM’s parameter size grows, indicating that there remains room for architectural improvement.

Summary. In contrast to previous work, we propose a novel design of learnable multidimensional canvas tokens that are trained jointly with a small connector and injected directly into the diffusion latent space, creating a compact and extensible framework that excels in explicitly composing spatial temporal signatures for diffusion generation. Our method is capable to generalize from images (2D) to videos (3D) seamlessly, while also maintaining efficiency in the volume of canvas tokens in the interface. In summary, MetaCanvas preserves the strengths of each backbone, namely understanding for the MLLMs and generation for diffusion models, lowers the training cost, and scales naturally across image and video tasks, effectively serving as a SOTA-level lightweight bridge from MLLMs to diffusion models.

EmptyText T5

DiT

TextInput T5

DiT

##### DiT

##### T5

Text Input

[Figure 96]

[Figure 97]

Connector Noise

Connector Noise

Noise

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Patchify

Patchify

Patchify

MLLM

MLLM

w/ MRoPE

w/ MRoPE

[Figure 104]

[Figure 105]

[Figure 106]

Text Input

Text Input

Canvas Tokens

Input Image

Canvas Tokens

Input Image

(1) Text + MetaCanvas

(2) MetaCanvas w/o Text

(3) SANA Default

Token Concat

DiT

DiT

C

Connector Noise

Connector Noise

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Patchify

Patchify

MLLM

MLLM

[Figure 111]

[Figure 112]

Text Input

Text Input

Query Tokens Input Image

Query Tokens Input Image

(5) Text + MetaQuery

(4) MetaQuery

- Figure 9 Visualizations of the architectures for the comparison variants in Figure 4.

- A.2 Extended Preliminaries

Latent Diffusion Models with Flow-Matching. Given an RGB visual input x ∈ RF×3×H×W, with F = 1 for images and F > 1 for videos, we first encode it with a spatial (or spatiotemporal) variational autoencoder (VAE) (Kingma and Welling, 2013) into d-dimensional latents z = E(x) ∈ RF

′×d×H′×W′, where F′ ≤ F, H′ < H, and W′ < W due to downsampling (Bai et al., 2025b; Zhu et al., 2025; AI, 2025; DeepMind, 2025). We then adopt a flow-matching objective (Lipman et al., 2024). Specifically, let ϵ ∼ N(0,I) and t ∼ U(0,1). We define a straight-line path (rectified flow) in latent space zt = (1−t)z+tϵ, and the corresponding target velocity field ut = dz

dt = ϵ−z. A latent diffusion model (LDM) Fθ(zt,t,ctext,cvision) is trained to match this field given

t

conditioning text ctext and/or images/videos cvision, via LFM = Ez,ϵ,t Fθ(zt,t,ctext,cvision) − ϵ − z 22. At inference, we solve the ODE dz

dt = Fθ(zt,t,ctext,cvision) from t = 1 to t = 0 with multiple denoising steps to obtain zˆ, and decode with the VAE decoder xˆ = D(zˆ). In MetaCanvas, we train with the same flow matching objective LFM without extra auxiliary losses, and we use the same denoising strategy during inference.

t

#### B Exploratory T2I Generation

- B.1 Architecture Details of Other Design Choices

Here, we provide a more detailed illustration of the experiments in Figure 4 of Section 5.1. Specifically, in this exploratory experiment, we aim to demonstrate that:

- • Canvas tokens can effectively transfer information from the MLLM into DiT.
- • The transferred information provides additional gains beyond the information contained in text.
- • Combining canvas information with text leads to faster training convergence compared with other information injection strategies, including MetaQuery (Pan et al., 2025).

T5

DiT

DiT

T5

Text Input

Text Input

[Figure 113]

[Figure 114]

Timestep

Linear Proj

Linear Proj

Linear Proj

Linear Proj

AdaLN

Vanilla Transformer Block

Vanilla Transformer Block

DiT Block

DiT Block

Noise Noise

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Patchify

Patchify

MLLM

MLLM

w/ MRoPE

w/ MRoPE

[Figure 121]

[Figure 122]

Canvas Tokens

Input Image

Canvas Tokens

Input Image

Ablation (1): Remove Timestep Condition in DiT Block

Default MetaCanvas Connector Design

T5

DiT

DiT

DiT

T5

T5

Text Input

Text Input

Text Input

[Figure 123]

[Figure 124]

Patchify

Linear Proj

Linear Proj

Chanel Concat

Vanilla Transformer Block

Noise Noise Noise

Linear Proj

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Patchify

Patchify

MLLM

MLLM

MLLM

w/ MRoPE

w/ MRoPE

w/ MRoPE

[Figure 131]

[Figure 132]

[Figure 133]

Canvas Tokens

Input Image

Canvas Tokens

Input Image

Canvas Tokens

Input Image

Ablation (2): Remove DiT Block Ablation (3): Remove Vanilla Transformer Block

Ablation (4): Add Canvas Tokens Before Patchification

- Figure 10 Ablation study on MetaCanvas connector architecture design. Corresponding quantitative results are shown in Figure 2. Note that our focus here is to demonstrate the effectiveness of canvas tokens; therefore, we keep the T5 text encoder in SANA instead of replacing it with text embeddings from MLLMs. We later replace the text encoder with the output multimodal embeddings from the MLLM in the image editing and video tasks in Section 5.2 and Section 5.3.

The quantitative results for these variants are shown in Figure 4, and their architectures are visualized in Figure 9:

- • (1) Text + MetaCanvas: We use the canvas embeddings output from the MLLM, together with the text embeddings from the original T5 text encoder in SANA as conditioning. Our goal is not to create a unified conditioning strategy by replacing the original T5 text encoder with an MLLM; rather, we only add the canvas tokens output from the MLLM to the latents before inputting them into DiT.
- • (2) MetaCanvas w/o Text: We provide empty text (i.e., Text Input = “) to the T5 text encoder. This allows us to control for the same DiT architecture without removing any components from its cross-attention blocks.
- • (3) SANA Default: This is the default architecture used in SANA, where only text information is used for conditional generation.
- • (4) MetaQuery: The 1D query embeddings output from the MLLM are provided as input, using the same cross-attention interface of DiT.
- • (5) Text + MetaQuery: We concatenate the text embeddings from the T5 text encoder with the learnable query tokens from the MLLM and input them into DiT to provide richer information.

A photo of a purple computer keyboard and a red chair

A photo of a blue pizza and a yellow baseball glove

A photo of a skateboard above a person

A photo of four giraffes

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

### MetaCanvasBLIP-3o

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

- Figure 11 Comparison between MetaCanvas with query-based architecture (i.e., BLIP-3o (Chen et al., 2025a)) on GenEval (Ghosh et al., 2023) prompts.

- B.2 Ablation for Canvas Connector Design

- In Figure 10, we provide a more detailed visualization of the ablation study for canvas connector architecture design variants, as mentioned in Table 3 of Section 3.4. The effectiveness of each component is quantitatively demonstrated in Table 3, so we skip redundant illustration here for simplicity.

B.3 Experiment Setup Details

- In Table 9, we give the hyper-parameter and model details of the experiments in Figure 4 and Table 3. Specifically, in both experiments, we keep the parameters of the MLLM fully frozen without adding any LoRA, and we train the DiT from scratch.

Additionally, as the generated images are of 512 × 512 resolution and the compression rate is 32, the DiT latents have a shape of 16 × 16. Therefore, we use canvas tokens of the same shape, resulting in a total of 256 canvas tokens. The training is conducted on a single A100 node and takes around one day to complete for approximately 20k steps.

B.4 Additional Comparison with Query-Based Architecture

In our main paper, we compare MetaCanvas with MetaQuery by training the DiT (i.e., SANA) from scratch. Here, we provide additional experiments that compare MetaCanvas with this line of query-based architectures. Since MetaQuery is not open-sourced, we compare our method with BLIP-3o (Chen et al., 2025a), another model that uses query tokens generated from an MLLM as conditioning for the DiT. For a fair comparison, our model is trained solely on the BLIP3o-60k dataset, which is a subset of the training data used in BLIP-3o. In this experiment, we apply LoRA fine-tuning to the DiT with a rank of 32.

- In Figure 11, we compare MetaCanvas with BLIP-3o on several challenging GenEval prompts, including attribute combinations with uncommon colors (e.g., “A photo of a blue pizza and a yellow baseball glove”), unusual spatial relationships (e.g., “A photo of a skateboard above a person”), and object counts greater than three (e.g., “A photo of four giraffes”). As shown in the figure, MetaCanvas consistently generates images that correctly capture these attributes, spatial relationships, and counts, whereas BLIP-3o fails on these prompts by mixing colors between objects, struggling to produce counterfactual spatial relationships such as a person below a skateboard, and generating an incorrect number of giraffes. In Table 8, we compare our model with models that use a DiT-only architecture for generation-only tasks, as

###### Table 8 Quantitative results on GenEval benchmarks. Here, we LoRA finetune SANA instead of training it from scratch. For a fair comparison, MetaCanvas is trained solely on the BLIP3o-60k dataset, which is a subset of the training data used in BLIP-3o.

Model GenEval

DiT w/o MLLM

FLUX1-Dev (Labs, 2024) 0.82 SANA (Xie et al., 2024a) 0.66 Query-Based Architecture

MetaQuery-XL (Pan et al., 2025) 0.80 BLIP3o-8B (Chen et al., 2025a) 0.84 MetaCanvas (Ours) 0.86

well as models with a query-based architecture, including MetaQuery and BLIP-3o. Note that BLIP3o-8B achieves better performance than MetaQuery-XL, while our model uses the same MLLM and DiT (i.e., Qwen2.5-VL-7B and SANA1.0-1.6B) as MetaQuery, and is trained on BLIP3o-60k, which is a subset of the training data used in BLIP-3o. Under this controlled setting, our method still outperforms both query-based models. These quantitative results, together with the visualizations above, demonstrate the effectiveness of our canvas design.

###### Table 9 Hyper-parameter and training details of the experiments in Figure 4 and Table 3.

Learning Rate 1e-4 Warmup Learning Rate 1e-5 LR Scheduler Constant w/ WarmUp Weight Decay 0.0 Gradient Clip 1.0 Optimizer AdamW Warm-Up Steps 300 Training Steps 20k Image Batch Size / GPU 40 # A100 GPUs 8 MLLM Training Type Frozen DiT Training Type From Scratch # Canvas Tokens 16×16=256

###### Table 10 Ablation study on the effectiveness of multimodal-RoPE for encoding learnable canvas tokens.

GenEval

MetaCanvas w/o MRoPE 0.85 MetaCanvas (Default) 0.86

- B.5 Additional Ablation Results

MultimodalRoPEinMLLM. Here, we discuss the use of multimodal RoPE in the MLLM to process the learnable canvas tokens. Specifically, for image and video understanding tasks in Qwen2.5-VL (Bai et al., 2025b), images and videos are encoded using multimodal RoPE (i.e., MRoPE), which provides improved spatial–temporal encoding of visual information. Therefore, we adopt the same strategy to encode our learnable canvas tokens.

- In Table 10, we show that MetaCanvas without multimodal RoPE achieves slightly worse results on GenEval compared with our default setting, although both variants outperform previous models, including MetaQuery and BLIP-3o. Since applying multimodal RoPE is essentially free that requires only a preprocessing step for position IDs and incurring no additional computational cost, we adopt it as our default configuration.

MLLM LoRA. In addition, we observe that adding a LoRA with rank of 32 to the MLLM effectively increases its capacity and helps the canvas tokens extract information more accurately from the MLLM, which further improves the GenEval score from 0.86 to 0.87.

Cross-Attn

MMDiT

DiT

MLP Connector

MLP Connector

MLLM

MLLM

(a) Architecture with MMDiT (FLUX.1 Kontext)

(b) Architecture with Cross-Attention (SANA, WAN2.2)

- Figure 12 MetaCanvas implemented with MMDiT and cross-attention-based architectures.

FLUX.1-Kontext-Dev

VAE Encoder

Replace the words in to ‘MetaCanvas’

Qwen2.5-VL-7B

Multimodal Instructions

MLP Connector

[Figure 142]

VAE Encoder

Canvas Connector

w/ MRoPE

LoRA

[Figure 143]

Learnable 2D Canvas Tokens

[Figure 144]

[Figure 145]

|[Figure 146]|
|---|

[Figure 147]

[Figure 148]

Target Image Input Image

Text Branch Vision Branch

[Figure 149]

[Figure 150]

[Figure 151]

Noise

- Figure 13 MetaCanvas implementation architecture details in Section 5.2. We adopt FLUX.1-Kontext-Dev (Batifol et al.,

- 2025) as the MMDiT and Qwen2.5-VL-7B (Bai et al., 2025b) as the MLLM. The trainable components include the vision branch, the LoRA in the MLLM, as well as the canvas tokens and the corresponding lightweight canvas connector.

#### C Image Editing Task

- Table 11 Quantitative comparison results on GEdit-EN-full (Liu et al., 2025) benchmark.

|Method|BG Color Mat. Motion Portrait Style Add Remove Replace Text Tone Avg↑|
|---|---|
|Instruct-Pix2Pix (Brooks et al., 2023) MagicBrush (Zhang et al., 2023a) OmniGen (Xiao et al., 2025) Step1X-Edit (Liu et al., 2025) Step1X-Edit (v1.1) (Liu et al., 2025) BAGEL (Deng et al., 2025) OmniGen2 (Wu et al., 2025b) GPT-Image-Edit (Wang et al., 2025) Qwen-Image-Edit-2509 (Wu et al., 2025a)<br><br>|3.94 5.40 3.52 1.27 2.62 4.39 3.07 1.50 3.48 1.13 5.10 3.22<br><br>6.17 5.41 4.75 1.55 2.90 4.10 5.53 4.13 5.10 1.33 5.07 4.19 5.23 5.93 5.44 3.12 3.17 4.88 6.33 6.35 5.34 4.31 4.96 5.01<br>7.03 6.26 6.46 3.66 5.23 7.24 7.17 6.42 7.39 7.40 6.62 6.44 7.45 7.38 6.95 4.73 4.70 7.11 8.20 7.59 7.80 7.91 6.85 6.97 7.44 6.99 6.26 5.09 4.82 6.04 7.94 7.37 7.31 7.16 6.17 6.60<br><br><br>- - - - - - - - - - - 6.42<br><br>7.80 7.54 7.12 7.75 7.09 6.74 8.04 7.95 7.17 5.45 6.95 7.24<br>8.25 8.10 7.51 8.60 6.92 6.85 8.56 8.59 8.37 8.18 7.85 7.98<br>|
|FLUX.1-Kontext-Dev (Batifol et al., 2025) FLUX.1-Kontext-Dev + MetaCanvas<br><br>|7.06 7.03 5.52 5.62 4.68 5.55 6.95 6.76 6.13 6.10 7.48 6.26 7.71 7.88 7.27 7.97 7.82 6.93 8.44 8.15 8.02 6.11 8.06 7.67 0.65↑ 0.85↑ 1.75↑ 2.35↑ 3.14↑ 1.38↑ 1.49↑ 1.39↑ 1.89↑ 0.01↑ 0.58↑ 1.41↑<br><br>|

As FLUX.1-Dev (Labs, 2024) and FLUX.1-Kontext-Dev (Batifol et al., 2025) share the same T5 text encoder, we follow GPT-Image-Edit (Wang et al., 2025) and initialize from the MLP connector in UniWorld-V1 (Lin et al., 2025a), which was trained to bridge Qwen2.5-VL-7B (Bai et al., 2025b) and FLUX.1-Dev (Labs,

- 2024). Note that SANA uses cross-attention as its conditioning interface, whereas FLUX.1-Dev concatenates text and image tokens as input to the DiT. This architectural difference leads to differences in how text tokens are injected when replacing the original text encoder of the DiT with output embeddings from the MLLM. A high-level illustration comparing the MMDiT and cross-attention–based architectures is provided

- Table 12 Quantitative comparison results on ImgEdit (Ye et al., 2025) benchmark.

|Method<br><br>|Add Adjust Extract Replace Remove Background Style Hybrid Action Overall↑|
|---|---|
|MagicBrush (Zhang et al., 2023a) Instruct-Pix2Pix (Brooks et al., 2023) UltraEdit (Zhao et al., 2024) OmniGen (Xiao et al., 2025) ICEdit (Zhang et al., 2025) Step1X-Edit (Liu et al., 2025) BAGEL (Deng et al., 2025) UniWorld-V1 (Lin et al., 2025a) OmniGen2 (Wu et al., 2025b) GPT-Image-Edit (Wang et al., 2025) FLUX.1 Kontext [Pro] (Batifol et al., 2025) GPT-Image-1 [High] (OpenAI, 2025) Qwen-Image-Edit (Wu et al., 2025a) Qwen-Image-Edit-2509 (Wu et al., 2025a)|2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.90<br><br>2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88<br><br>3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70<br><br>3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96<br><br>3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05<br><br>3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06<br><br>3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20<br><br>3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26<br><br>3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44<br><br>4.07 3.79 2.04 4.13 3.89 3.90 4.84 3.04 4.52 3.80<br><br><br><br><br>4.25 4.15 2.35 4.56 3.57 4.26 4.57 3.68 4.63 4.00<br><br><br>4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20<br><br><br>4.38 4.16 3.43 4.66 4.14 4.38 4.81 3.82 4.69 4.27<br><br><br>4.32 4.36 4.04 4.64 4.52 4.37 4.84 3.39 4.71 4.35<br><br><br><br><br>|
|FLUX.1 Kontext [Dev] (Batifol et al., 2025) FLUX.1 Kontext [Dev] + MetaCanvas<br><br>|3.76 3.45 2.15 3.98 2.94 3.78 4.38 2.96 4.26 3.52<br><br>4.20 3.50 2.11 4.41 3.72 3.89 4.83 3.61 4.49 3.86 0.44↑ 0.05↑ 0.04↓ 0.43↑ 0.78↑ 0.11↑ 0.45↑ 0.65↑ 0.23↑ 0.34↑<br><br><br>|

in Figure 12. A more detailed version that specifically bridges FLUX.1-Kontext-Dev with Qwen2.5-VL-7B is shown in Figure 13.

We start the training directly by unfreezing the diffusion model’s visual branch, together with the additional learnable canvas tokens and the transformer-based connector. To increase MLLM’s model capacity, we apply LoRA with rank of 64 to the MLLM backbone. We illustrate the training hyper-parameters in Table 13. The model is trained on O(1M) image-editing samples from OmniEdit (Wei et al., 2024) and HQEdit (Hui et al.,

- 2024). Table 12 and Table 11 contain extended evaluation results of Table 1 and Table 2.

- Table 13 Hyperparameters and training details of the image editing experiments in Table 1 and Table 2.

Training Hyperparameters Learning Rate 1e-6 LR Scheduler Constant

- Adam β1 0.9
- Adam β2 0.99 Adam ϵ 1e-8 Weight Decay 0.0 Gradient Clip 1.0 Optimizer AdamW Warm-Up Steps 0 Training Steps 90k Image Batch Size / GPU 1 # A100 GPUs 64 Model Parameters MLLM Training Type LoRA MLLM LoRA Rank 64 MLLM LoRA Dropout 0.05 DiT Training Type Train Vision Branch # Canvas Tokens 32×32=1024

#### D MetaCanvas Architecture Design for Video Tasks

- D.1 Context and Canvas Connectors Design

Context connector design. For the MLP context connector in video tasks, we adopt a similar strategy as in the image editing tasks by training a 2-layer MLP, with input channels equal to the embedding dimension of the MLLM and output channels equal to the embedding dimension of the DiT. We use an expansion ratio of 4 in the middle layer of the MLP to increase its expressiveness. A dropout ratio of 0.1 is applied to the parameters in this MLP connector.

- Channel 1

- Channel 2

- Channel 3

- Channel 4

- Channel 5

- Channel 6

- Channel 7

- Channel 8

- 0

- 1

Value

1

2

3

1 3 5 7 9 11 13 15 17 19 21

Latent Frame Number

- Figure 14 Visualization of the first 8 channels of Wan2.2 VAE after encoding a static video of 81 frames (21 latent frames).

- Table 14 Quantitative comparison results on VBench-I2V (Huang et al., 2024b) for I2V generation.

|Method|I2V I2V Camera Subject Background Motion Dynamic Aesthetic Imaging<br><br>Subject BG. Motion Consistency Consistency Smooth. Degree Quality Quality Overall↑<br><br>|
|---|---|
|Wan-5B Wan-5B + MetaCanvas<br><br>|97.79 98.92 42.54 94.31 96.14 96.60 58.53 61.95 71.24 86.98 97.89 98.81 80.60 94.24 93.66 97.32 54.30 60.97 70.38 87.13<br><br>|

Canvas connector design. For the canvas connector, we use the same architecture as in the image generation and editing tasks without any modifications. Specifically, the canvas connector consists of a single vanilla transformer block and a single DiT block. The vanilla transformer block is designed to more effectively transform the canvas embeddings output by the MLLM into the feature space of the DiT, while the DiT block further fuses the canvas embeddings with the DiT latents and dynamically determines the influence of the canvas tokens on the latents through AdaLN. As the self-attention module in the DiT operates spatially and temporally across all frames, the canvas connector here can be applied only on a per-frame basis rather than across all frames, which reduces GPU memory consumption.

- D.2 Interface Design for Canvas Keyframe and Reference/Condition Frames Injection

One key design aspect is the interface for injecting canvas keyframes into the DiT while seamlessly enabling diverse video tasks, including text-to-video generation, image-to-video generation, video editing, and referenceto-video generation. Our goal is to design an interface that is compatible with all of these tasks and can be easily extended to additional video tasks. We partially follow the interface of Wan-Animate (Cheng et al., 2025) to support conditioning on both reference frames and source video frames. In addition, to effectively inject canvas embeddings into the latents, we use the design shown in Figure 3 to combine the canvas embeddings with the latents after patchification.

- D.3 2D canvas vs. 3D canvas

Quantitative comparison. A natural question is whether we should keep the same 2D canvas for video tasks, as in image tasks, or instead use a 3D canvas to encode both spatial and temporal information. As shown quantitatively in Table 6, using a 2D canvas achieves comparable spatial alignment scores (measured by optical flow error and PSNR) and also attains the best evaluation score with GPT-4o. Although the overall evaluation scores remain reasonably high, we empirically observe that using a 2D canvas introduces noticeable temporal flickering in the first few frames, whereas applying a 3D canvas with three keyframes effectively alleviates this issue.

Hypothesis and explanations. Upon further investigation, we found that this discrepancy likely stems from the VAE used in Wan2.2. Specifically, the Wan2.2 VAE supports both image and video encoding: the first frame is encoded as a single latent frame, while every subsequent four frames are collapsed into one latent frame. Additionally, the VAE employs a moving-average mechanism to better preserve temporal information across latent frames. To illustrate this behavior, we encode a static 81-frame video using the Wan2.2 VAE,

resulting in 21 latent frames. Figure 14 visualizes the first eight channels of these latents. We observe clear value inconsistencies across the initial latent frames, while the later latent frames become stable. This explains why a 2D canvas, whose embeddings are added identically to all latent frames, produces temporal flickering in the early frames. In contrast, a 3D canvas with 3 keyframes adapts to the temporal structure of the latent sequence and effectively mitigates this issue. We acknowledge that the current design may still be suboptimal and leave the exploration of improved designs for future work.

#### E Text/Image-to-Video Generation Tasks

Task-specific model details. For the image-to-video generation task, we follow the setting in Wan2.2-5B by replacing the first latent frame in the noisy latents with the clean latent of the first input image. During both training and inference, the first latent frame is assigned a timestep of zero, while the remaining latent frames receive noise according to their corresponding timesteps. This ensures that the generated video better preserves the information from the input image.

Training dataset details. In stage 1 (a), we use approximately O(40M) in-house images for training. In later stages, we further incorporate approximately O(8M) in-house videos for training.

#### F Video Editing Task

- F.1 Training dataset details.

As there’s no natual source-edited video pairs, we utilize the following strategy to construct a total of 300k video editing samples. Specifically, our training data comes from two sources:

- • Generate from videos. Given a raw video, we use a MLLM (i.e., Qwen2.5-VL) to generate a video editing instruction that performs one of the following operations: local object addition, replacement, deletion, background change, or global style transfer. We then use Qwen-Image-Edit (Wu et al., 2025a) to generate the edited first frame. Additionally, we employ depth ControlNet to extract depth maps from the raw video. Finally, the edited video is generated from the edited first frame and the corresponding depth maps.
- • Generate from images. Starting from paired image editing data, we use Wan2.2-5B (Wan et al., 2025) to animate the raw image into a video and extract depth frames via depth ControlNet. The edited video is then generated from the edited first frame and the extracted depth maps.

- F.2 Evaluation benchmark details.

Evaluation dataset curation strategy. Due to the scarcity of open-source video editing evaluation benchmarks with high resolution (at least 720p) and sufficient duration (at least 121 frames at 24 FPS), we curate a balanced evaluation set consisting of 300 video prompts. This set covers a diverse range of video editing tasks, including local object editing (addition, removal, or replacement), background change, and style transfer. Specifically, we first randomly sample 3k videos from our high-quality in-house video data, which are not used during training to avoid potential data leakage. For each video, we prompt GPT-4o (OpenAI, 2024b) to generate an editing instruction corresponding to one of the target categories, ensuring that the instructions balance creativity with natural suitability to the video content. Next, GPT-4o is used to filter out low-quality video-editing prompt pairs and to maintain a balanced distribution of prompts across categories. We also perform a manual verification step to ensure the instructions are meaningful and suitable to the videos. Additionally, half of the videos contain humans, which helps align the evaluation set with user needs for human-centered video generation.

Evaluation strategy. To assess the edited videos, we employ the following evaluation methods jointly:

• VBench (Huang et al., 2024b) score. We use the quality scores from VBench to evaluate the overall video quality, which is orthogonal to how well the edited video follows the editing instructions. The quality score is computed as the average of subject consistency, background consistency, motion smoothness, aesthetic quality, dynamic degree, and imaging quality. We follow the VBench codebase to calculate these metrics.

- Table 15 Quantitative comparison results on our OmniContext-Video benchmark for in-context video generation from reference images. This table is an extended version of Table 7 that further includes detailed prompt-following and subject-consistency scores for completeness, in addition to the overall score. Compared with MetaCanvas, VACE14B (Jiang et al., 2025) tends to over-copy the reference images into the final generated video, making it harder to follow the user instructions.

Prompt-Following Single ID Multiple IDs Scene

Char. Obj. Char. Obj. Char. + Obj. Char. Obj. Char. + Obj. Average↑

Wan-VACE-14B 5.54 4.74 2.76 3.96 5.14 3.58 3.48 4.28 4.10 Wan2.2-5B + MetaCanvas 5.58 5.78 4.08 5.44 6.60 4.60 5.46 4.74 5.28

Subject-Consistency Single ID Multiple IDs Scene

Char. Obj. Char. Obj. Char. + Obj. Char. Obj. Char. + Obj. Average↑

Wan-VACE-14B 8.86 8.96 4.52 7.02 6.66 4.54 4.90 5.26 6.34 Wan2.2-5B + MetaCanvas 7.46 7.80 4.06 6.46 6.68 5.04 5.64 4.90 6.01

- • MLLM evaluation. We adopt a strategy similar to the GEdit (Liu et al., 2025) benchmark for video editing. Specifically, GPT-40 (Hurst et al., 2024) is used to evaluate semantic and quality scores, following the same prompt templates as GEdit but replacing “input image” with “input frames”. To improve API efficiency, video frames are sampled at 1 FPS. The overall score is calculated in the same manner as in the original GEdit benchmark.
- • Human evaluation. We conduct human evaluation on videos generated by MetaCanvas and the two strongest baseline methods, Lucy-Edit-Dev-v1.1 (Team, 2025) and Ditto (Bai et al., 2025a). A total of 50 human evaluators participate, with videos randomly shuffled to ensure blind scoring. Evaluators are provided with the input video, editing prompt, and the three edited videos, and are asked to identify the videos with the best and worst editing accuracy, as well as the best and worst consistency. Editing accuracy measures prompt-following ability, similar to semantic evaluation in the MLLM method, while consistency measures whether the edited video is spatially and temporally aligned with the input video while maintaining subject and background consistency. The winning rates for each method are reported in Table 5.

#### G In-Context Video Generation Task

- G.1 Training Dataset Details

Below, we provide a detailed illustration of our data curation strategy. Given the first frame of a video, we first use Qwen2.5-VL (Bai et al., 2025b) to generate editing prompts that extract the dominant objects and characters, as well as the background scene, from the image. We then provide the image along with these extraction prompts to Qwen-Image-Edit (Wu et al., 2025a) to generate images containing the extracted characters, objects, and scenes as reference images.

Additionally, we observed that when Qwen-Image-Edit extracts objects or characters, it often uses a white background, whereas the input reference images usually contain diverse backgrounds. To address this, we use Qwen2.5-VL to imagine suitable and natural backgrounds for the character and object reference images, and then use Qwen-Image-Edit to generate new edited images with these backgrounds. Using this process, we create a total of 70k in-context video data samples for training.

- G.2 Evaluation Benchmark Details

As in-context video generation from reference images is a relatively new task and no well-established opensource evaluation benchmark currently exists, we adapt OmniContext (Wu et al., 2025b), an in-context image generation benchmark, into a video generation benchmark. We named this new benchmark as OmniContextVideo. Specifically, we reuse the same editing prompts and reference images from OmniContext without any modifications, but the objective is now to generate videos instead of images based on these reference images and user text instructions. This design allows us to directly leverage the original OmniContext evaluation protocol, by simply replacing the prompt instruction to evaluate the generated “video” instead of the generated “image.” For API efficiency, we sample the generated video at 1 FPS before providing it to GPT-4.1 (OpenAI, 2024a) for evaluation. Consistent with the original benchmark, we evaluate both prompt-following and subject-consistency. Table 7 in the main paper reports the overall score averaged across

Stage 1. Connector Alignment Stage 2. High-Resolution Finetuning

[Figure 152]

[Figure 153]

[Figure 154]

Wan2.2-5B

Wan2.2-5B

Cross-Attn

[Figure 155]

[Figure 156]

MLP Connector

MLP Connector

[Figure 157]

[Figure 158]

Qwen2.5-VL-7B VAE Encoder

Qwen2.5-VL-7B VAE Encoder

Text

Text Image/Video

Image/Video

Stage 3. Multi-Task Training

[Figure 159]

Wan2.2-5B

[Figure 160]

[Figure 161]

MLP Connector

Canvas Connector

[Figure 162]

[Figure 163]

VAE Encoder

Qwen2.5-VL-7B VAE Encoder

LoRA

Learnable 3D Canvas Tokens

Multimodal Instructions

Input Image/Video Target Image/Video

- Figure 15 MetaCanvas implementation architecture details in Section 5.3. We adopt Wan2.2-5B (Wan et al., 2025) as the DiT and Qwen2.5-VL-7B (Bai et al., 2025b) as the MLLM. We illustrate the trainable components in each training stage respectively.

these two dimensions, and we provide an expanded version including individual scores for each dimension in Table 15 for completeness.

#### H Video Tasks Training Details

As detailed in Section 4.1, we adopt a three-stage training strategy for video tasks. To help readers better understand the trainable components in each stage, we visualize the training frameworks in Figure 15. Additionally, we provide a comprehensive overview of the training hyperparameters, model parameters, and data sampling ratios in Table 16 to ensure reproducibility.

We note that the first two stages are designed solely to align the MLLM (i.e., Qwen2.5-VL-7B (Bai et al.,

- 2025b)) with the video generation model (i.e., Wan2.2-5B (Wan et al., 2025)), without involving any canvas tokens or canvas connector training. These stages are trained on basic tasks, including text-to-image generation, image reconstruction, and text-to-video generation. We allocate a relatively large compute budget to these stages, as sufficient training is crucial to properly align the MLLM with the diffusion model. Insufficient training in these stages would make it difficult for the model to capture fine-grained prompt information in stage 3 due to diluted learning on the the above basic tasks.

The third stage focuses on multi-task training, incorporating our model-specific canvas components. Compared with the first two stages, stage 3 requires relatively fewer training steps, completing in only 10k steps. We also observe that multi-task training provides mutual benefits across tasks. For example, in Figure 18, MetaCanvas demonstrates improved understanding of styles, an ability likely learned from the image and video style editing data.

#### I Visualizations

In this section, we first illustrate how we plot the canvas map, and then provide additional visualization examples for different tasks.

- Table 16 Training hyperparameters on Wan2.2-TI2V-5B with MetaCanvas across different stages.

Training Stages Stage 1 Stage 2 Stage 3

Hyperparameters

Connector Alignment High Resolution Finetuning Multi-task Training

Training Hyperparameters Learning Rate 1e-3 2e-5 1e-5 LR Scheduler Constant Constant Constant Weight Decay 0.0 0.0 0.0 Gradient Clip 1.0 1.0 1.0 Optimizer AdamW AdamW AdamW Warm-Up Steps 100 100 100 Training Steps 100k 15k 10k Image/Video Resolution 480×832 704×1280 704×1280 Video # Frames - 121 121 Video FPS - 24 24 Image Batch Size / GPU 32 32 16 Video Batch Size / GPU - 3 1 # A100 GPUs 128 128 128 Model Parameters Trainable Modules MLP Connector Connector + Cross-Attn All Params MLLM Tokens Per Image 144 144 144 MLLM LoRA Rank - - 64 # Canvas Tokens - - 11×20×3=660 Data Sampling Ratio Text-to-Image 80% 16% 5% Image Reconstruction 20% 4% Image Editing - - 15% Text-to-Video - 80% 15% Image-to-Video - - 15% Video Editing - - 25% Reference-image-to-Video - - 25%

Canvas map visualization. Following (Tumanyan et al., 2023), we apply PCA to the features produced by the MetaCanvas connector. The features are extracted at approximately the middle of the diffusion timesteps (e.g., t = 499 for the text-to-image generation task in Figure 5, and t = 522 for the video tasks). For each extracted feature, we apply PCA and visualize the top three principal components. Note that the colors on the PCA map only represent salient subjects or backgrounds, and the color assigned to a given subject or background may vary across different canvas keyframes. As a supplement to Figure 5, we provide additional visualizations of the canvas maps for text-to-video generation and image-to-video generation tasks in Figure 16, and visualizations for video tasks in Figure 17.

Visualizations for T2V generation task. In Figure 18, we present visualizations for the text-to-video generation task, specifically on the “appearance style” category in VBench (Huang et al., 2024a). As our model is jointly trained on diverse tasks, including video global style editing, we observe that such joint training enables MetaCanvas to better understand stylistic variations in prompts.

Visualizations for I2V generation task. In Figure 19, we show visualizations for the image-to-video generation task, focusing on the “camera motion” category in VBench (Huang et al., 2024b). Compared with the base Wan2.2-5B model, MetaCanvas more effectively interprets instructions involving camera motions. In Table 14, we further provide the expanded results corresponding to Table 4 in the main paper.

Visualizations for video editing task. In Figures 20 to 22, we show visualizations for the (local) video editing task. Compared with other methods (Bai et al., 2025a; Team, 2025), MetaCanvas achieves more precise grounding of the target objects. In Figure 23, we show visualizations for the video background editing task. Compared with other methods (Bai et al., 2025a; Team, 2025), MetaCanvas achieves more precise instruction following and grounding of the target background. In Figure 24, we show visualizations for the video global style editing task. MetaCanvas achieves results comparable to Ditto (Bai et al., 2025a), whereas Lucy-Edit-Dec-v1.1 (Team,

- 2025) fails on this task.

Visualizations for in-context video generation task. In Figures 25 to 27, we present visualizations for the in-context video generation task using reference images. We observe that the comparison baseline (i.e., VACE-14B (Jiang et al., 2025)) suffers from a “copy–paste” effect, where the background of the reference image is replicated in the generated video. This makes it difficult for the model to follow the instructions and to naturally compose multiple reference images into the correct background. In addition, both VACE-14B and MetaCanvas produce imperfect videos when the number of reference images increases to three. We hypothesize that an improved data creation strategy with more and higher-quality training data, and assigns a higher proportion of examples to multi-reference video generation, could help alleviate this issue.

- J Simplified Code Implementation

Our codebase builds upon the publicly available BLIP3o codebase (Chen et al., 2025a). Detailed training procedures and model hyperparameters are provided in Section 4.1 and Table 9. We will release our code upon acceptance. For better reproducibility, we also provide a simplified PyTorch implementation of the DiT block in the canvas connector in Algorithm 1.

- K Limitations and Future Works

In this work, we primarily focus on investigating the effectiveness of information transfer between MLLMs and diffusion Transformers via MetaCanvas. Our approach follows prior work that bridges MLLMs with diffusion models through lightweight connector interfaces. One potential limitation of the current setup is that visual information (e.g., images and videos) is provided to both the MLLM and the diffusion models, following previous works (Wei et al., 2025; Wu et al., 2025a; Lin et al., 2025a; Wu et al., 2025b; Liu et al., 2025) to maximize performance. It would be interesting to explore whether a more elegant framework could be designed that passes all visual information solely to the MLLM, allowing DiT to directly render images and videos without repeated conditioning on visual inputs.

Additionally, we evaluated the effectiveness of MetaCanvas across diverse tasks using text, image, video, or combinations thereof as inputs. However, we note that the quality of our curated training data is not optimal, and the scale of the data is limited for some of the tasks. For instance, we observed that the success rate for in-context video generation from three or more reference images is not high. Expanding the task-specific dataset could further improve performance.

Algorithm 1 Simplified PyTorch Implementation for the DiT Block in Canvas Connector import torch from torch import nn import torch.nn.functional as F from einops import rearrange from diffusers.models.normalization import AdaLayerNormSingle from diffusers.models.transformers.sana_transformer import SanaTransformerBlock

class CanvasConnectorDiTBlock(nn.Module):

def __init__(self, in_channels=48, embed_dim=3072, num_layers = 1): super().__init__() self.embed_dim = embed_dim self.time_embed = AdaLayerNormSingle(embed_dim) self.patch_embed = nn.Conv3d(in_channels, embed_dim,

kernel_size=(1, 2, 2), stride=(1, 2, 2)

) self.out_proj = zero_module(nn.Linear(embed_dim, embed_dim)) self.canvas_dit_blocks = nn.ModuleList(

[

SanaTransformerBlock( dim=embed_dim, num_attention_heads=embed_dim//32, attention_head_dim=32, num_cross_attention_heads=None, cross_attention_head_dim=None, cross_attention_dim=None, attention_bias=False,

) for _ in range(num_layers)

] )

def forward(self, latent, interpolated_canvas_keyframes, timestep): latent_embed = self.patch_embed(latent) b, t, _, h, w = latent_embed.shape

# add after patchification hidden_states = latent_embed + interpolated_canvas_keyframes hidden_states = rearrange(hidden_states, "b c t h w -> (b t) (h w) c")

timestep = timestep if timestep.ndim == 1 else timestep[:,-1] timestep, _ = self.time_embed(timestep).repeat(t, 1)

# fused through the DiT block conditioned on the timestep for index_block, block in enumerate(self.canvas_dit_blocks):

hidden_states = block(

hidden_states, None, None, None, timestep, h, w, None, None )

fused = self.out_proj(hidden_states) # linear projection before output fused = rearrange(fused, "(b t) (h w) c -> b c t h h w", b=b, h=h, w=w) latent_embed = latent_embed + fused # latent_embed is passed to the DiT return latent_embed

Text-to-Video Generation

Prompt A zebra bending down to drink water from a river

[Figure 164]

[Figure 165]

[Figure 166]

Canvas Map

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Edited Video

Prompt A person playing guitar

[Figure 172]

[Figure 173]

[Figure 174]

Canvas Map

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Edited Video

Prompt A cute happy Corgi playing in park, sunset, pixel art

[Figure 180]

[Figure 181]

[Figure 182]

Canvas Map

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Edited Video

Image-to-Video Generation

Prompt A light house on the edge of the water, camera zooms in

[Figure 188]

[Figure 189]

[Figure 190]

First Frame & Canvas Map

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

###### Edited Video

Prompt A young man is covered in colored powder

[Figure 196]

[Figure 197]

[Figure 198]

First Frame & Canvas Map

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Edited Video

- Figure 16 Visualization of the PCA feature maps of the 3 canvas keyframes for the text-to-video generation and image-to-video generation tasks. Note that for the image-to-video generation task, because the first frame is assigned t = 0 during inference with Wan2.2-5B, we omit the visualization for the first canvas keyframe. Interpretation of the canvas map visualization is discussed in Appendix I.

Prompt Change the girl's dress to a flowing white dress with subtle embroidery along the hem.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Source Video

[Figure 209]

[Figure 210]

[Figure 211]

Canvas Map

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Edited Video

Change the palm trees in the background to vibrant autumn-colored maple trees with falling red and gold leaves.

Prompt

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Source Video

[Figure 222]

[Figure 223]

[Figure 224]

Canvas Map

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Edited Video

Change the background light to a cool, ethereal blue with soft diffused rays instead of the current yellowish-white, making the light streaks appear as shimmering cobalt threads against the dark backdrop.

Prompt

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

Source Video

[Figure 235]

[Figure 236]

[Figure 237]

Canvas Map

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Edited Video

Prompt Change the background to a sunlit alpine road with snow-capped mountains and a blue sky.

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Source Video

[Figure 248]

[Figure 249]

[Figure 250]

Canvas Map

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Edited Video

- Figure 17 Visualization of the PCA feature maps of the 3 canvas keyframes for the video editing task. Interpretation of the canvas map visualization is discussed in Appendix I.

Prompt A panda drinking coffee in a cafe in Paris by Hokusai, in the style of Ukiyo

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Wan2.2-5B

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Wan2.2-5B+ MetaCanvas

Prompt A panda drinking coffee in a cafe in Paris, animated style

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Wan2.2-5B

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Wan2.2-5B+ MetaCanvas

Prompt A panda drinking coffee in a cafe in Paris, black and white

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Wan2.2-5B

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Wan2.2-5B+ MetaCanvas

Prompt A panda drinking coffee in a cafe in Paris, in cyberpunk style

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Wan2.2-5B

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Wan2.2-5B+ MetaCanvas

Prompt A panda drinking coffee in a cafe in Paris, Van Gogh style

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Wan2.2-5B

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Wan2.2-5B+ MetaCanvas

- Figure 18 Visualization on text-to-video generation task. The input prompts are sourced from VBench (Huang et al.,

- 2024a). Multi-task training in Stage 3 enables MetaCanvas to better understand stylistic variations in prompts.

###### Prompt Tower bridge in london, camera zooms in

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Wan2.2-5B

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Wan2.2-5B+ MetaCanvas

Prompt Two people in a canoe on a lake with mountains in the background, camera zooms out

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Wan2.2-5B

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Wan2.2-5B+ MetaCanvas

Prompt The parthenon in acropolis, greece, camera tilts up

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Wan2.2-5B

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Wan2.2-5B+ MetaCanvas

Prompt An asian city street at night with people and bicycles, camera tilts down

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Wan2.2-5B

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

Wan2.2-5B+ MetaCanvas

Prompt The golden gate bridge in san franscisco is lit up by the setting sun, camera pans right

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Wan2.2-5B

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Wan2.2-5B+ MetaCanvas

- Figure19 Visualizationonimage-to-videogenerationtask. The input images and prompts are sourced from VBench (Huang et al., 2024b). Compared with the base model, MetaCanvas better understands instructions involving camera motions.

Replace the laptop screen content with a flowing digital watercolor painting in soft teal and gold hues, matching the room's natural lighting.

Prompt

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

Source Video

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Ditto

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Lucy-Edit-Dev v1.1

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

MetaCanvas

Add in the background a rustic wooden cabin featuring a glowing window and a snow-laden roof.

Prompt

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Source Video

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

Ditto

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

Lucy-Edit-Dev v1.1

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

MetaCanvas

Replace the green leaf pattern on the blanket with a delicate pattern of small white daisies on a soft light blue background.

Prompt

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

Source Video

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Ditto

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Lucy-Edit-Dev v1.1

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

MetaCanvas

- Figure 20 Visualization for the video local editing task (part 1/3). MetaCanvas achieves more precise grounding of the target

Replace the pink coffee cup with a vintage porcelain cup featuring a delicate blue floral pattern, maintaining the same placement and lighting on the green table.

Prompt

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

Source Video

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

Ditto

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Lucy-Edit-Dev v1.1

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

MetaCanvas

Replace the blurred colorful planets with high-definition, realistic celestial bodies showing visible cloud formations and textured surfaces, matching the boy's space-themed play.

Prompt

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Source Video

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

Ditto

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

Lucy-Edit-Dev v1.1

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

MetaCanvas

Change the copper mug’s surface to a frosty, icy finish with delicate ice crystals forming on its exterior, reflecting the dim snowy light.

Prompt

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

Source Video

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Ditto

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

Lucy-Edit-Dev v1.1

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

MetaCanvas

- Figure 21 Visualization for the video local editing task (part 2/3). MetaCanvas achieves more precise grounding of the target

Replace the guitar's body with a deep, glossy cherry wood finish that mirrors the soft light on the left side of the frame.

Prompt

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

Source Video

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

Ditto

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Lucy-Edit-Dev v1.1

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

MetaCanvas

Change the background to a smooth, pale green ceramic plate with subtle floral patterns, enhancing the sushi's traditional presentation.

Prompt

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

Source Video

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

Ditto

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Lucy-Edit-Dev v1.1

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

MetaCanvas

Replace the pineapple-shaped inflatable ring with a smooth, glossy ring that reflects the ocean's blue hues.

Prompt

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

Source Video

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

Ditto

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

Lucy-Edit-Dev v1.1

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

MetaCanvas

- Figure 22 Visualization for thevideo local editingtask (part 3/3). MetaCanvas achieves more precise grounding of the target

Change the background to a tranquil, misty alpine meadow with wildflowers and distant snow-capped peaks, maintaining the same warm, natural lighting and camera perspective.

Prompt

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

Source Video

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

Ditto

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

Lucy-Edit-Dev v1.1

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

MetaCanvas

Change the wooden surface to a smooth, light gray stone surface with subtle marbling patterns.

Prompt

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

Source Video

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

Ditto

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

Lucy-Edit-Dev v1.1

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

MetaCanvas

Shift the lighting from cool purple-blue tones to warm golden-hour sunlight, casting honeygold highlights through the woman's curls while maintaining the slow-motion movement and cinematic depth.

Prompt

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

Source Video

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

Ditto

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

Lucy-Edit-Dev v1.1

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

MetaCanvas

- Figure 23 Visualization for the video background editing task. MetaCanvas achieves more precise instruction following and grounding of the target background.

Prompt Make it Children's Book Illustration.

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

Source Video

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

Ditto

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

Lucy-Edit-Dev v1.1

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

MetaCanvas

Convert the video into a pixel art style by reducing the resolution and creating a mosaic effect with square-shaped pixels and simplified colors for a classic pixelated appearance.

Prompt

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

Source Video

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

Ditto

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

Lucy-Edit-Dev v1.1

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

MetaCanvas

Prompt Convert the snowy mountain landscape into a watercolor painting.

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

Source Video

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

Ditto

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

Lucy-Edit-Dev v1.1

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

MetaCanvas

- Figure 24 Visualization for the video global style editing task. MetaCanvas achieves results comparable to Ditto (Bai et al.,

- 2025a), whereas Lucy-Edit-Dev-v1.1 (Team, 2025) fails on this task.

[Figure 656]

Place the anime character with long, flowing light blue hair in a serene garden at sunset, powerfully lifting weights. Sweat glistens on her determined face as she strains against the heavy barbell.

Ref. Image(s) + Prompt

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

VACE-14B

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

MetaCanvas

[Figure 667]

Show the man in a suit embracing his partner in a lavender field, both smiling and holding bouquets.

Ref. Image(s) + Prompt

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

VACE-14B

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

MetaCanvas

[Figure 678]

Position a slice of tiramisu elegantly on a marble altar in the center of the grand cathedral, surrounded by flickering candlelight that enhances its rich colors.

Ref. Image(s) + Prompt

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

VACE-14B

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

MetaCanvas

[Figure 689]

Perch a dark blue starling with white spots on the armrest of a vintage-style armchair, its yellow beak glinting in the warm light filtering through the window.

Ref. Image(s) + Prompt

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

VACE-14B

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

MetaCanvas

- Figure 25 Visualization for the in-context video generation task with single character or object. MetaCanvas demonstrates improved prompt understanding and more accurately places the specified character or object in the appropriate scene, whereas VACE (Jiang et al., 2025) incorrectly adheres too closely to the background of the reference image.

Set a vibrant peacock perched gracefully on a rustic

[Figure 700]

[Figure 701]

wooden table in an outdoor café setting, surrounded by potted plants and a charming stone wall, while placing a sleek silver car parked nearby, reflecting the warm sunlight

Ref. Image(s) + Prompt

filtering through the glass doors.

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

VACE-14B

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

MetaCanvas

[Figure 712]

[Figure 713]

Ref. Image(s) + Prompt

I wish the person and the man would stare at each other.

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

VACE-14B

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

MetaCanvas

[Figure 724]

[Figure 725]

She reaches out to touch the deer's head, with the deer looking calmly at her in a serene forest setting.

Ref. Image(s) + Prompt

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

VACE-14B

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

MetaCanvas

Please make the lady from the first image crouch slightly in a warmly lit living room as she playfully pats the Shiba Inu from photo 2. Her hand is mid-motion, just above the dog’s head. The dog stands on a patterned rug, glancing up with an excited yet puzzled look. A floor lamp casts soft shadows in the cozy, woodpaneled room.

[Figure 736]

[Figure 737]

Ref. Image(s) + Prompt

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

VACE-14B

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

MetaCanvas

- Figure 26 Visualization for the in-context video generation task with multiple characters and/or objects. MetaCanvas demonstrates an improved ability to naturally compose the reference images into the appropriate scene, whereas VACE (Jiang et al., 2025) either adheres too closely to the background of the reference images or fails to compose both reference images correctly.

[Figure 748]

[Figure 749]

Place the red panda on the edge of the bed, curling up against the decorative pillows as it gazes out the window, soaking in the warm sunlight.

Ref. Image(s) + Prompt

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

VACE-14B

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

MetaCanvas

[Figure 760]

[Figure 761]

Ref. Image(s) + Prompt

Place the fluffy dog on the wet pavement in front of the modern museum.

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

VACE-14B

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

MetaCanvas

[Figure 772]

[Figure 773]

Ref. Image(s) + Prompt

He plays guitar in the place.

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

VACE-14B

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

MetaCanvas

[Figure 784]

[Figure 785]

[Figure 786]

The muscular character from the first image enjoys the large burger while walking in the grand interior of the cathedral from the second image.

Ref. Image(s) + Prompt

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

VACE-14B

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

MetaCanvas

- Figure 27 Visualization for the in-context video generation task with character and/or object in a scene. MetaCanvas demonstrates improved prompt understanding and naturally composes the reference images into the appropriate scene. However, both methods produce imperfect videos when the number of reference images increases to three.

