## Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device

Abdelrahman Shaker1,∗,† Ahmed Heakl1,† Jaseel Muhammad1 Ritesh Thawkar1 Omkar Thawakar1 Senmao Li1 Hisham Cholakkal1 Ian Reid1 Eric P. Xing1,2 Salman Khan1 Fahad Shahbaz Khan1,3

1Mohamed bin Zayed University of Artificial Intelligence 2Carnegie Mellon University 3Link¨oping University

# arXiv:2602.20161v2[cs.CV]24Feb2026

Codebase: github.com/Amshaker/Mobile-O Project Page: amshaker.github.io/Mobile-O Models: https://huggingface.co/collections/Amshaker/mobile-o-models Datasets: https://huggingface.co/collections/Amshaker/mobile-o-datasets

[Figure 1]

Text-to-Image Generation

[Figure 2]

~3.0 secs ~1.4 GB

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Tropical rainforest scene, scarlet macaw perched …..

Breathtaking mountain landscape, golden hour …..

Underwater coral reef with individual coral polyps ……

A mystical desert scene at twilight, towering red ….

Visual Understanding

~0.4 sec ~1.1 GB

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### Prompt:

Response: Based on the visual look and common ingredients .... likely components are: 1. Penne Pasta: tube-like with ridges .... 2. Tomato Sauce: red coating, tomato-based .... 3. Cheese: shredded white topping, Parmesan .... 4. Herbs: green sprinkles, basil or parsley ..

Give me the list of ingredients used in this dish

[Figure 13]

~4.0 secs ~1.8 GB

[Figure 14]

[Figure 15]

Visual Prompt Understanding

Prompt: Convert this drawing to a photorealistic image

[Figure 16]

[Figure 17]

Detailed Caption: The image is a black and white coloring page featuring a

detailed illustration of a rabbit sitting in a forest … The text in the image identiﬁes the following elements: 1. Rainy golden hour … 2. Green Trees: This is labeled with an arrow pointing to … 3. White: This is labeled with an arrow pointing to the rabbit … The background of the image is ﬁlled with detailed illustrations of plants, trees, and foliage, creating a lush ……

Mobile-O Generation

Mobile-O Understanding

[Figure 18]

[Figure 19]

Speed and Memory shown on iPhone Speed and Memory shown on MacBook

Figure 1. Comparison of our approach with existing unified models. Left: Qualitative comparison illustrating Mobile-O’s capabilities in text-to-image generation, visual understanding, and visual prompt understanding. Right: Quantitative comparison with Show-O, Janus, and JanusFlow, demonstrating that Mobile-O achieves a superior trade-off. Our Mobile-O outperforms Show-O by 5.0% on GenEval and runs significantly faster on iPhone.

*Corresponding author: abdelrahman.youssief@mbzuai.ac.ae †Equal contributions

### Abstract

Unified multimodal models can both understand and generate visual content within a single architecture. Existing models, however, remain data-hungry and too heavy for deployment on edge devices. We present Mobile-O, a compact vision-language-diffusion model that brings unified multimodal intelligence to a mobile device. Its core module, the Mobile Conditioning Projector (MCP), fuses vision–language features with a diffusion generator using depthwise-separable convolutions and layerwise alignment. This design enables efficient cross-modal conditioning with minimal computational cost. Trained on only a few million samples and post-trained in a novel quadruplet format (generation prompt, image, question, answer), Mobile-O jointly enhances both visual understanding and generation capabilities. Despite its efficiency, Mobile-O attains competitive or superior performance compared to other unified models, achieving 74% on GenEval and outperforming Show-O and JanusFlow by 5% and 11%, while running 6× and 11× faster, respectively. For visual understanding, Mobile-O surpasses them by 15.3% and 5.1% averaged across seven benchmarks. Running in only ∼3s per 512×512 image on an iPhone, Mobile-O establishes the first practical framework for real-time unified multimodal understanding and generation on edge devices. We hope Mobile-O will ease future research in real-time unified multimodal intelligence running entirely on-device with no cloud dependency. Our code, models, datasets, and mobile application are publicly available.

### 1. Introduction

Unified multimodal models capable of both understanding and generating visual content have recently gained popularity in vision. Inspired by the success of large language models (LLMs), recent works extend their reasoning and generative capabilities to vision-language tasks, where the unified multimodal models can caption images, answer visual questions, and generate visuals within a single framework [4, 10, 43, 44]. Earlier unified approaches [14, 35] explore a single transformer design that can perform both multimodal understanding and generation, when trained jointly on text and image tokens. Subsequent works [49] incorporate diffusion-based generation directly into unified architectures. Recent methods [4, 10] further explore unified model training on large-scale interleaved multimodal data, achieving improved performance.

Despite these advances, existing unified multimodal models face two critical challenges that limit their practical deployment on consumer devices. First, most existing unified models employ computational and memorydemanding visual encoders and denoising modules. For

instance, BLIP-3o [4] requires a 2.6B-parameter UNet for denoising and 3B vision-language model (VLM), in addition to 1.5B for diffusion transformer (DiT), resulting in 7.1B total parameters. While few recent works [22] explore computational efficiency in unified multimodal models, they still remain unsuitable for real-time deployment on edge devices (see Fig. 1). Second, effective cross-modal alignment within unified models often depends on massive pre-training datasets, typically 50M–1B samples [4, 10], making pre-training expensive and time-consuming. These observations motivate us to explore a key question: Can we build a unified multimodal model that is effective for both tasks (understanding and generation), while being efficient for deployment on consumer devices like mobile phones?

In this work, we present Mobile-O, a compact, efficient unified multimodal model that can run directly on a mobile device with low memory overhead and real-time latency, as shown in Fig. 1. Unlike prior approaches that require extensive pre-training, our Mobile-O achieves strong understanding and generation performance with only a few million pre-training samples and carefully curated unified post-training data. At the core of our approach is the Mobile Conditioning Projector, a mobile-optimized connector that fuses the final hidden states of the VLM with the conditioning space of the diffusion model. Furthermore, we address a key limitation in existing training paradigms. Prior unified models either mix disjoint task-specific datasets [37, 43] or adopt sequential training that isolates understanding and generation tasks [4, 26]. In contrast, we propose a unified multimodal post-training stage that leverages a compact unified dataset where each sample simultaneously supports both tasks through a quadruplet (generation prompt, image, question, answer) representation for improved cross-modal alignment. Finally, we demonstrate real-time deployment of our MobileO on edge devices, including iPhone, NVIDIA Jetson Nano, and MacBook. The model achieves ∼3 seconds per 512 × 512 image generation on an iPhone device, setting a new benchmark for on-device unified multimodal generation. In summary, our key contributions are:

- • We introduce Mobile-O, an efficient unified vision–language–diffusion model that achieves state-ofthe-art multimodal understanding and image generation performance, while enabling real-time inference on a mobile device (see Fig. 1).
- • To build Mobile-O, we first design a solid baseline mobile unified architecture, which is further enhanced with two contributions. First, we introduce the Mobile Conditioning Projector (MCP), a lightweight cross-modal fusion module that effectively bridges visual understanding and diffusion-based generation using depthwiseseparable convolutions and layerwise alignment. Second, we propose a unified multimodal post-training scheme

that leverages a quadruplet data representation (generation prompt, image, question, answer) with a unified dataset of 105k samples, enabling joint optimization of multimodal understanding and generation tasks.

• Our Mobile-O, with only 1.6B total parameters, achieves 74% on GenEval, outperforming Show-O and JanusFlow by 5% and 11%, respectively, while being up to 11× faster. For multimodal image understanding, it surpasses them by 15.3% and 5.1%, respectively, on average across seven widely used benchmarks (see Fig. 1).

### 2. Related Work

##### Multimodal Understanding & Generation:

Earlier unified multimodal models [21, 34, 43] unify both understanding and generation tasks with a single transformer. Hybrid designs, such as Janus [41], BLIP3-o [4], and JanusFlow [22] integrate diffusion decoders for better text-to-image generation, while Emu3 [40] shows that autoregression can suffice for text-to-image generation.

While achieving promising results, the aforementioned unified models either rely on heavy UNet-style [4] or computationally heavy architectures [41, 43] (e.g., CLIP-ViT image encoder). Moreover, most existing unified models depend on disjoint supervision across understanding and generation, thereby improving one task while freezing the other [4, 26]. In contrast, we present a unified mobileoptimized architecture that utilizes a unified multimodal post-training stage where the performance of both tasks is simultaneously improved through a multi-task objective.

Efficient Multimodal Understanding Models: Recent advances in efficient vision-language modeling have focused primarily on optimizing visual encoding strategies [23, 30, 39]. FastVLM [39] addresses the computational bottleneck of processing high-resolution images by introducing FastViTHD, a hybrid vision encoder with competitive visual understanding performance. Similarly, SmolVLM [23] shows that careful architectural optimizations and aggressive tokenization enable compact models to achieve competitive performance, while consuming less GPU memory. While these approaches focus at efficient multimodal understanding, our work advances this research line of efficient multimodal intelligence by introducing a unified framework that couples a compact vision-language understanding model with lightweight diffusion through novel conditioning projector to perform both multimodal understanding and image generation tasks in a single architecture.

Efficient Text-to-Image Generation Models: Recent works [6, 42] have explored efficient text-to-image (T2I) generation. SANA [42] introduces high-resolution image generation through deep compression autoencoders and linear attention mechanisms. However, they use heavy text encoders (i.e, Gemma-2B [28]). SnapGen [6] proposes systematic architecture optimization and cross-architecture

distillation, generating images efficiently with multiple steps on resource-constrained devices. Both approaches are designed for T2I generation and lack multimodal understanding capabilities like FastVLM [39]. In contrast, our work strives to design a unified mobile-optimized approach that can effectively perform both multimodal understanding and generation tasks within a single framework.

Data Efficiency and Training Stages in Unified Models: Training unified multimodal models typically requires extensive datasets. BAGEL [9] studies emerging properties in unified multimodal pre-training, revealing fundamental insights about data requirements. Existing unified approaches generally follow two training strategies: (i) Joint Training: Methods like Metamorph [37] and Show-o [43] perform multitask learning by mixing data for image understanding and image generation. While joint training allows the two tasks to potentially benefit from each other [17, 19], its effectiveness strongly depends on the total data size and the ratio between understanding and generation samples. Current unified training datasets often consist of disjoint subsets for each task [17], e.g., LLaVA-665K for understanding and BLIP3o-60K for generation, which limits the model’s ability to learn fully aligned cross-task understanding. (ii) Sequential Training: Other unified works [26, 41] adopt a two-stage approach: first training the VLM, then freezing the backbone and training only the generation module. For instance, BLIP3-o [4] uses a pre-trained VLM and freezes it in all stages. This strategy preserves understanding capability, while dedicating to enhance generation performance. However, it does not exploit potential cross-task interactions during training to improve both tasks.

To address these limitations, we introduce a unified post-training stage with 105k samples, where each sample simultaneously supports both understanding and generation. Each training sample is formatted as (generation prompt, image, question, answer), enabling the model to learn aligned understanding and generation capabilities during post-training. This unified format allows us to effectively leverage cross-modal transfer while avoiding the task imbalance and inter-task interference.

### 3. Method

Motivation: To motivate our approach, we distinguish two desirable characteristics to be considered when designing an efficient unified multimodal model for edge deployment. • Efficient Understanding and Generation Connection:

Generally, standard unified models employ a connection module that contains MLP layers to connect understanding and generation components. In addition, the connection module leverages a set of learnable queries that act as a bridge between multimodal LMM and diffusion, enabling improved generation performance. However, such a connection design achieves sub-optimal performance

Visual Understanding Image Generation

|| |
|---|
<br><br>| |
|---|
<br><br>| | |
|---|---|
| | |
<br><br>|Linear LN+HS|
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
| |
|---|---|
|| |
|---|
<br><br>| |
| | |
|| |
|---|
| |
<br><br>| |
|---|
<br><br>Average Pool<br><br>Hard Swish<br><br>Hard Swish<br><br>Efficient Channel Attention<br><br>Depthwise Separable Conv1D<br><br>CompressRefineBlock<br><br>Linear LN+HS<br><br>O<br><br>x<br><br>+<br><br>x<br><br>x<br><br>x<br><br>Mobile Conditionting Projector<br><br>LayerN<br><br>LayerN-1<br>LayerN-2<br>LayerN-3<br><br><br>W3<br><br>W1<br>W2<br><br><br>W4<br><br><br>Linear (4C C)<br><br>Linear (C 4C)|
|---|

|[Figure 20]|
|---|

|There are 6 arches in the image.| |
|---|---|
| | |

###### I2T Loss DiT

Answer

Mobile Conditionting Projector

|VAE Decoder|
|---|

Und. Tokens

| | |
|---|---|
| | |
| | |
| | |
| | |

LayerN

- LayerN-1

Layer1

- LayerN-2
- LayerN-3

[Figure 21]

|[Figure 22]|
|---|

Autoregressive Language Model

CompressutputRefineBlock

T2I Loss

Embedding Layer

Image Encoder

| |
|---|

[Figure 23]

How many arches are visible on the bridge in the image?

A photo of Prague?s Charles Bridge and Prague Castle at sunset, golden hour lighting.

Gen. Tokens

Question Generation Prompt

Image

Figure 2. Overview of Mobile-O. Left: The proposed framework consists of an efficient image encoder with a compact autoregressive language model for visual understanding. For image generation, a lightweight linear diffusion transformer (DiT) is employed alongside a simple yet effective VAE-based encoder–decoder. Right: Our novel Mobile Conditioning Projector (MCP) bridges the understanding and generation tasks by directly conditioning the diffusion model on weighted hidden states from the VLM without the need for intermediate query tokens. The projector leverages layer-wise feature fusion, depthwise separable convolutions, and efficient channel attention to produce high-fidelity conditioning signals with minimal cost, enabling seamless deployment on edge devices.

Encoder

VAE

when using substantially less pre-training data (around 5× less than BLIP3o [4]). Therefore, an efficient yet effective connection design is desired to achieve superior performance when constructing a data-efficient mobile unified framework.

• Unified Post-training for Symbiotic Learning: As discussed earlier, most existing unified models either employ joint training [17, 43] or utilize sequential training [4, 41] for understanding and generation. However, joint training typically relies on a careful balancing of disjoint understanding and generation data samples, whereas sequential training only aims to improve one task (e.g., generation) while freezing the other (e.g., understanding). To address this, a unified post-training approach is desired based on a multi-task objective using a joint set of understanding and generation data samples to simultaneously improve both understanding and generation tasks.

#### 3.1. Baseline Mobile Unified Framework

Since existing mobile-optimized models are designed to either perform multimodal visual understanding or image generation, we first aim at building a solid baseline mobile unified architecture capable of handling both tasks. Motivated by recent unified models such as BLIP-3o [4], which build generation capabilities directly on top of existing understanding models (e.g., Qwen2-VL), we adopt a similar yet mobile-optimized design strategy. To establish a strong mobile unified baseline, we consider efficient pre-

trained vision-language model (VLM) backbones and diffusion decoders in configurations reflecting prior unified models. Specifically, as our baseline, we employ FastVLM [39] for multimodal understanding and integrate it with a DiTstyle diffusion decoder [42] for multimodal generation.

Let fθ denote the vision-language encoder-decoder (FastVLM [39]) and gϕ the diffusion image decoder (SANA-0.6B [42]). Given a text prompt p and an optional image x (for understanding), the VLM produces layerwise hidden states {H(1),...,H(L)}, where H(ℓ) ∈ RN×d

vlm for token length N and hidden size dvlm. The diffusion model gϕ is a DiT-style decoder with cross-attention blocks accepting encoder features of dimension dcond. Following recent unified models [4, 43–45], gϕ remains fully learnable, but we avoid introducing extra textual tokens beyond those produced by fθ. Unlike SANA-0.6B [42], which uses the Gemma-2B [28] model as a text encoder to process generation prompts, we employ the same LLM used for the understanding model to handle the generation prompts, resulting in a more parameter-efficient design.

Our goal is to jointly learn θ and ϕ so the model can (i) perform visual understanding tasks (e.g., question answering) and (ii) generate images from prompts, all within a mobile-optimized architecture. Next, we discuss how to further improve the performance of the baseline mobile unified framework through an efficient yet effective projector design and a unified post-training approach with a multitask objective to improve understanding and generation.

#### 3.2. Mobile Conditioning Projector (MCP)

Unified frameworks usually insert learnable query tokens between the VLM and the image decoder [4, 26, 45]. While this approach is effective for large models, it requires massive pre-training data for effective alignment. To this end, we design an efficient yet effective conditioning projection (MCP) layer that directly connects VLM hidden states to the diffusion decoder, as shown in Fig. 2. The MCP maps the VLM’s final-layer features (or a fusion of the last K layers) to diffusion-compatible conditioning sequences with minimal parameters and FLOPs.

Layerwise Fusion. Let S = {L−K+1,...,L} denote the last K VLM layers. We compute a temperature-scaled softmax weighting αℓ = exp(w

ℓ/τ)

j∈S exp(wj/τ), and form a fused representation,

αℓ H(ℓ) ∈ RN×d

. (1)

vlm

Hfuse =

ℓ∈S

where the weights {wℓ} are learned; τ is cosine-annealed during the training.

Compression and Refinement. We project Hfuse to a compact space and refine it using depthwise-separable 1D convolutions and lightweight channel attention:

H˜ = LN HfuseWc , Wc ∈ Rd

vlm×dh, (2) H˜ ← SeqRefine H ˜ , (3)

where SeqRefine applies a depthwise-separable Conv1D followed by pointwise mixing and a tiny MLP-based channel attention. Operating along sequence length N (not spatial grids) avoids expensive 2D convolutions and retains token-level alignment with language stream.

Output Projection. The diffusion cross-attention expects dcond-dimensional keys and values. We compute

E = LN(HW˜ o), Wo∈Rd

h×dcond, E∈RN×d

. (4)

cond

All cross-attention layers in gϕ use the same sequence E as encoder features, analogous to CLIP-conditioning in latent diffusion, but learned end-to-end with the VLM. Compared to query-token approaches [4, 26, 45], the proposed MCP introduces no extra token budget and reduces parameter count and requires less pre-training data.

Complexity. For hidden size dh and kernel k, the refinement block costs O(k dh) (depthwise) + O(d2h) (pointwise) per token, substantially cheaper than full 2D convolution or attention over new query tokens.

#### 3.3. Training Scheme

We propose a three-stage training scheme for our Mobile-O that progressively enhances multimodal understanding and generation capabilities. The three stages are: cross-modal alignment, supervised fine-tuning and unified multimodal

[Figure 24]

[Figure 25]

Mobile Conditioning Projector

I2T Loss

DiT

[Figure 26]

Autoregressive Language Model

T2I Loss

[Figure 27]

[Figure 28]

[Figure 29]

Embedding Layer Image Encoder

VAE Encoder

Answer Generation Prompt Image

Question

[Figure 30]

|[Figure 31]<br><br>LoRa-training| | |
|---|---|---|
| |Unified Components| |

[Figure 32]

Frozen Components Learnable Components

Understanding Components

|Generation Components|
|---|

Figure 3. Overview of the proposed unified multimodal posttraining pipeline. We jointly optimize multimodal understanding and generation through a multi-task objective using a quadruplet format (generation prompt, image, question, answer). Both I2T and T2I losses are computed simultaneously, enabling aligned cross-modal learning where each training sample supports both multimodal understanding and generation.

post-training. During the first two stages, the visual encoders and LLM backbone are frozen to learn better multimodal generation. The focus of our design is the introduction of a novel unified multimodal post-training stage (stage 3), where both multimodal understanding and generation are improved using a joint set of data samples via a multi-task objective (see Fig. 3).

- Stage 1: Cross-Modal Alignment. Here, the primary objective is to establish robust connections between visual and linguistic representations within a unified embedding space. We adopt a parameter-efficient approach by freezing the visual encoders and LLM backbone, and update only the DiT and MCP. In this stage, we conduct pre-training on JourneyDB [32], which provides high-quality 4 million text–image pairs covering diverse visual concepts, and 5 million pairs from BLIP3o-Short-Caption [4], a curated subset emphasizing compositional understanding.
- Stage 2: Supervised Fine-tuning. Following initial alignment, we perform targeted fine-tuning on ∼105K curated prompt-image pairs (60K from BLIP3o [4], 45K from ShareGPT-4o-Image [5]) to address specific weaknesses observed after pre-training [4]. Due to our compact pretraining corpus (only 20% of BLIP-3o’s data during stage 1), the model initially struggled with complex human gestures, common objects and landmarks. This stage specifically targets these underrepresented domains while maintaining the same frozen/trainable component configuration as in the previous stage.
- Stage 3: Unified Multimodal Post-Training. This stage aims to improve both multimodal understanding and generation. To this end, we construct training samples as quadruplets S = {p,ximg,q,a}, where p denotes the gen-

- Table 1. Comparison with recent multimodal understanding models. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. Total Params represent the sum of visual encoder, language model, and diffusion/unet components (when applicable). Compared to unified models with similar size (≤ 2B), our Mobile-O-0.5B achieves superior overall performance with a score of 61.9 averaged over seven datasets. Further, Mobile-O-0.5B also outperforms its understanding-only counterpart (FastVLM) by 1.6% in average performance.

Type Model # Total Params MMMU↑ TextVQA↑ MMVet↑ SEED↑ ChartQA↑ POPE↑ GQA↑ Average↑

Und. Only > 1B LLaVA-Phi [50] 3.1B - 48.6 28.9 - - 85.0 - LLaVA-v1.5-Phi-1.5 [50] 1.6B 30.7 - - - - 84.1 56.5 MobileVLM [7] 1.7B - 41.5 - - - 84.5 56.1 MobileVLM-V2 [8] 1.7B - 52.1 - - - 84.3 59.3 LLaVa-OV [16] 1.6B 31.4 - 29.1 65.5 61.4 - - -

Und. Only ≤ 1B Smol-VLM-0.5B [23] 0.6B 33.7 60.2 - - 62.8 - - -

FastVLM-0.5B [39] 0.6B 33.3 68.0 37.5 69.3 71.6 81.1 62.7 60.5 Und. and Gen. > 2B EMU3-8B [40] 9.0B 31.6 64.7 37.2 68.2 68.6 85.2 60.3 59.4

BLIP3o-4B [4] 7.1B 46.6 78.0 60.1 73.8 - - - -

Und. and Gen. ≤ 2B Janus [41] 2.1B 30.5 50.2 34.3 63.7 53.0 87.0 59.1 54.0 Show-o [43] 1.5B 25.1 - - - - 73.8 48.7 Show-o-Clip-ViT [43] 1.6B 27.4 41.2 20.9 51.6 44.7 84.5 57.5 46.8 JanusFlow [22] 2.1B 29.3 55.5 30.9 70.5 64.6 88.0 60.3 57.0 Mobile-O-0.5B (Ours) 1.6B 34.6 67.8 38.1 69.4 75.2 86.4 62.9 62.1

eration prompt, ximg represents the image, and (q,a) form question-answer pairs (see Fig. 3). Since no existing dataset supports such a quadruplet format, we construct the data as follows:

- 1. Prompt GPT-4o [25] to generate highly detailed compositionally-aware caption for each image.
- 2. Synthesize diverse question-answer sets probing different aspects of visual understanding. This yields a unified dataset with bi-directional multi-

modal learning within a single framework, where both understanding with image-to-text (I2T) and generation with text-to-image (T2I) tasks share the same embedding layer and autoregressive language model, as shown in Fig. 3.

- 3.4. Training Objectives

Our unified training optimizes a weighted combination of multimodal understanding and generation objectives:

Lunified = λlangLlang + λdiffLdiff (5)

Image-to-Text (I2T) Loss. For multimodal understanding, we employ standard cross-entropy loss on the autoregressive language model’s output tokens:

|a|

Llang = −

t=1

log P(at|ximg,q,a<t) (6)

where, the model predicts answer tokens a conditioned on the image encoding and question q.

Text-to-Image (T2I) Loss. For multimodal image generation, we employ a flow-matching objective [4, 44] instead of standard noise prediction. Given a clean latent x from the VAE encoder and noise ϵ ∼ N(0,I), we sample a noise

level σ ∈ [0,1] and form:

xσ = (1 − σ)x + σϵ, v⋆(xσ;σ) = ϵ − x (7)

The DiT model predicts a velocity field vϕ(xσ,σ,cp) conditioned on MCP features cp derived from the generation prompt p (see Eq. 4). The loss minimizes the weighted mean-squared error:

Ldiff = Ex,p,ϵ,σ w(σ)∥vϕ(xσ,σ,cp) − (ϵ − x)∥22 (8)

where, w(σ) is a scale-dependent weighting function. This formulation directly learns the probability flow ODE, yielding faster and more stable training compared to standard diffusion objectives.

### 4. Experiments

#### 4.1. Implementation Details

We use FastVLM-0.5B [39] as the image understanding model, which extends FastViT [38] as the vision encoder and Qwen2-0.5B [36] as the language backbone. For image generation, we adopt the SANA-600M-512 [42] diffusion model as the visual generator. Both understanding and generation branches are connected through the proposed Mobile Conditioning Projector, implemented as a lightweight linear layers with depthwise-separable convolutions for efficient cross-modal alignment. All images used for understanding tasks are resized to 1024 × 1024 resolution using bicubic interpolation, while generation tasks operate at 512 × 512. All experiments are conducted on a single node equipped with 8 NVIDIA A100 GPUs, requiring approximately 3 days for 50k pre-training steps (roughly 3 epochs). The subsequent SFT and unified multimodal post-training

- Table 2. Evaluation of text-to-image generation performance on the GenEval benchmark. “Und.” and “Gen.” denote “understanding” and “generation”, respectively. Total Params represent the sum of the visual encoder, language model, and diffusion/unet components (when applicable). Compared to unified models with similar size (≤ 2B), our Mobile-O-0.5B achieves superior overall score of 0.74 and outperforms Show-o-Clip-ViT [43] by 5.0%.

Type Method # Total Params Single Obj. Two Obj. Counting Colors Position Color Attri. Overall↑

Gen. Only > 1B LlamaGen [33] 3.8B 0.71 0.34 0.21 0.58 0.07 0.04 0.32 LDM [29] 1.5B 0.92 0.29 0.23 0.70 0.02 0.05 0.37 PixArt-α [3] 4.9B 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDXL [27] 3.0B 0.98 0.74 0.39 0.85 0.15 0.23 0.55 SDv2.1 [29] 1.9B 0.98 0.51 0.44 0.85 0.07 0.17 0.50 SANA-0.6B [42] 2.6B 0.99 0.77 0.62 0.88 0.21 0.47 0.66

Gen. Only ≤ 1B SDv1.5 [29] 1.0B 0.97 0.38 0.35 0.76 0.04 0.06 0.43 SnapGen [6] 0.4B 1.00 0.84 0.60 0.88 0.18 0.45 0.66

Und. and Gen. > 2B SEED-X† [11] 16.0B 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Chameleon [35] 34.0B - - - - - - 0.39 LWM [20] 7.2B 0.93 0.41 0.46 0.79 0.09 0.15 0.47 BLIP3o-4B [4] 7.3B - - - - - - 0.81

Und. and Gen. ≤ 2B Janus [41] 2.1B 0.97 0.68 0.30 0.84 0.46 0.42 0.61 Show-o [43] 1.5B 0.98 0.80 0.66 0.84 0.31 0.50 0.68 Show-o-Clip-ViT [43] 1.6B 0.98 0.85 0.67 0.81 0.28 0.55 0.69 JanusFlow [22] 2.1B 0.97 0.59 0.45 0.83 0.53 0.42 0.63 Mobile-O-0.5B (Ours) 1.6B 0.98 0.87 0.57 0.86 0.68 0.49 0.74

Prompt Image Janus JanusFlow Show-O Mobile-O (Ours) Answer

Janus JanusFlow Show-O Mobile-O (Ours)

Question

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Area of Effect

KwaZulu-Natal ..... populated South Africa, ... female-headed percentage ...

What % of households in KwaZulu-Natal female-headed?

Healing tree, realism, welcoming, marketable, ...

47.6% 48.8% 47.9%

47.9%

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Which brand registered sales worth ?263 mil. in 2019?

A white flamingo in jungle, cartoon style, lush vegetation, ...

Moschino Moschino Alberta Ferretti. Adidas Moschino

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

The book is " The New York Stories" by H.R.R. Lowenthal, published by Houghton Mifflin Harcourt, Inc., in the " From the Pest Zone" .......

The book is titled " From the Pest Zone: The New York Stories" and is authored by H. P. Lovecraft. ..... short stories from Lovecraft's works .......

The book is a paperback edition of the book " The Book of the Night" by Robert Louis Stevenson.

Book Title: "From the Pest Zone: The New York Stories" by H. P.

The book is titled " From the Pest Zone: The New York Stories" .. The publisher is Houghton Mifflin, and the edition is the 1st.

Man Monkey with headphone, singer, cyberpunk ,lens 85mm, ...

Summarize the back cover text of the book.

Lovecraft. The back of the book describes five stories of the author in his time livining in New York City

Figure 4. Qualitative comparison of text-to-image generation (left) and visual understanding (right) across unified multimodal models. Each column shows Janus, JanusFlow, Show-O, and Mobile-O (ours) for the same prompts/questions. Mobile-O yields more consistent, detailed, and semantically faithful images with high fidelity and style diversity for image generation. For visual understanding, it delivers more accurate and contextually coherent responses. Additional results are presented in suppl. material. Best viewed zoomed in.

stages run for 20 epochs and 7 epochs, taking 15 hours and 5 hours, respectively. Detailed hyperparameter configurations for each stage are provided in the suppl. material.

#### 4.2. Quantitative Comparison

Multimodal Visual Understanding: We evaluate MobileO-0.5B on a diverse suite of understanding benchmarks. General multimodal understanding and reasoning are evaluated on MMMU [48], MM-Vet [47], and SEED [15]. For OCR and text-based VQA, we employ TextVQA [31] and ChartQA [24]. Text hallucination robustness is examined on POPE [18], while scene understanding is assessed on GQA [13]. Tab. 1 shows the comparison with

understanding-only models having <1B and >1B and unified models with <2B and >2B, on seven benchmarks. Here, the total number of parameters reflects all components, and not only the LLM. Mobile-O-0.5B offers distinct merits over models in its scale range (≤2B), such as Janus [41], JanusFlow [22], and Show-O [43]. Compared to JanusFlow [22], our Mobile-O-0.5B obtains an absolute gain of 4.9% averaged over seven benchmarks with less total parameters (JanusFlow: 2.1B vs. Ours: 1.6B). It is worth mentioning that our Mobile-O-0.5B obtains an absolute gain of 1.6% over FastVLM [39], highlighting the effectiveness of our unified multimodal post-training, where both understanding and generation tasks are improved via a multi-task

objective using joint training samples as quadruplets.

Text-to-Image Generation: We evaluate our model on the widely-used GenEval [12] benchmark. We follow strictly to raw prompts for GenEval. As shown in Tab. 2, we evaluate Mobile-O-0.5B with generation-only models having different sizes (> 1B and ≤ 1B) and unified models ( > 2B and ≤ 2B). Here, total number of parameters reflects all components. Compared to unified models with similar size (≤ 2B), our Mobile-O-0.5B achieves best overall results with score of 0.74, outperforming Show-o [43] by 5.0%.

Text-and-Image-to-Image Generation: Beyond text-toimage generation and visual understanding, the Mobile-O framework naturally supports image editing, taking both a source image and a textual instruction as input and producing an edited image as output. This capability emerges from the MCP design, which bridges the understanding and generation pathways through a shared multimodal representation. Because MCP captures low-level visual details from the input image, it is well-suited for editing tasks that require preserving the global scene structure while applying localized modifications.

To enable image editing, we fine-tune Mobile-O on a small subset of 46k editing samples from ShareGPT4V [5]. During editing, the source image is encoded through the vision encoder and projected via MCP, while the textual editing instruction is processed by the language model. The generation backbone then produces the edited image conditioned on both the visual and textual representations. No architectural modifications are required—the same MCP, language model, and generation backbone used for text-toimage generation and visual understanding are reused for editing. We evaluate Mobile-O-0.5B on the ImageEdit [46] benchmark, which measures both edit fidelity and scene preservation. Mobile-O-0.5B achieves an overall score of 2.5 on ImageEdit, despite being fine-tuned on only 46k editing samples. We note that Mobile-O-0.5B’s editing capability is achieved with minimal dedicated training data compared to specialized editing models such as BLIP3-

- o [4] and Emu-Edit [40], which are trained on significantly larger editing-specific datasets. With dedicated fine-tuning
- on larger-scale editing data, we expect both the edit fidelity and global scene preservation to further improve.

#### 4.3. Qualitative Comparison

Fig. 4 illustrates the generation and understanding capabilities of Mobile-O-0.5B with other unified models ≤ 2B parameters. Compared to Janus, JanusFlow, and Show-O, Mobile-O-0.5B produces images with sharper details, more coherent layouts, and more consistent illumination. It maintains higher visual fidelity in complex scenes, such as tree leaves or strands of a monkey’s hair. Janus and JanusFlow show counting errors in the second row of Fig. 4, consistent with their lower counting scores in Tab. 2. These

Edit Prompt Input Image Generated Image

[Figure 48]

[Figure 49]

Change the dog's fur color to a soft brown.

[Figure 50]

[Figure 51]

Transfer the image into a classic impasto oilpainting style.

[Figure 52]

[Figure 53]

Add a person wearing a red winter coat and black snow pants …

Figure 5. Qualitative image editing results of Mobile-O-0.5B. Given a source image and a textual editing instruction, MobileO-0.5B produces the edited output. The model is fine-tuned on only 46k editing samples from ShareGPT4V [5].

counting issues sometimes yield higher diversity but reduce text–image alignment. For understanding, MobileO-0.5B correctly answers samples from ChartQA [24] and TextVQA [31], and in the last row accurately summarizes a book cover, mentioning both title and author. Complete output comparison is provided in suppl. material. In Fig. 5, Mobile-O-0.5B successfully performs a range of basic editing operations, including adding an object, attribute modification, and style transfer.

#### 4.4. Ablation Study

Generality of Mobile-O: A natural question is whether the Mobile-O framework, specifically the Multi-modal Connector Projector (MCP), unified post-training data format, and training recipe, generalizes beyond the specific backbone choices presented in the main paper. To address this, we construct Mobile-O-1.5B by replacing the original components with larger counterparts: FastVLM-1.5B [39] as the vision-language understanding backbone and SANA1.5B [42] as the image generation backbone, yielding a unified model with approximately 3.5B parameters. The MCP dimensions are adjusted accordingly to match the hidden sizes of the larger backbones, while the overall architecture and training procedure remain unchanged. We evaluate understanding performance across seven established benchmarks: MMMU [48], TextVQA [31], SEED-Bench [15],

- Table 3. Mobile-O-1.5B: Scaling to FastVLM-1.5B and SANA1.5B components. Understanding performance is averaged over seven benchmarks (MMMU, TextVQA, SEED-Bench, ChartQA, POPE, GQA, MM-Vet). Generation quality is measured by GenEval overall score. The proposed post-training stage consistently improves both capabilities.

Model Und. Acc. (%) Gen. Acc. (%)

FastVLM-1.5B [39] 64.8% – SANA-1.5B [42] – 66% Mobile-O-1.5B (SFT) 64.8% 75% Mobile-O-1.5B (Post-train) 66.2% 78%

ChartQA [24], POPE [18], GQA [13], and MM-Vet [47]. For generation quality, we report the GenEval [12] overall score. Results are summarized in Table 3.

Mobile-O-1.5B after supervised fine-tuning preserves the full understanding capability of the standalone FastVLM-1.5B (64.8% average across the seven benchmarks) while simultaneously gaining strong generation ability (75% GenEval), which the original FastVLM entirely lacks. After the post-training stage, both capabilities improve further: understanding increases to 66.2% (+1.4% absolute over SFT) and generation reaches 78% (+3% absolute over SFT). Notably, the post-trained Mobile-O-3B also surpasses the standalone SANA-1.5B generation backbone (78% vs. 66%), demonstrating that the unified training and post-training recipe not only preserves but enhances the individual component capabilities. These results confirm that the Mobile-O framework is architecture-agnostic: the MCP design, unified data format, and post-training recipe transfer effectively to larger backbones, consistently improving both understanding and generation.

We analyze the contributions of the proposed MCP design and the effectiveness of our post-training data strategy. On the MCP Design. Tab. 4 shows how different MCP configurations influence cross-modal alignment and generation quality. Notably, all experiments in this table are conducted without pre-training. Using a simple MLP connector between the VLM and diffusion decoder achieves 68.5% on GenEval but requires over 3.2M trainable parameters. Replacing it with our single-layer MCP with a compression module reduces parameter count by nearly half, while maintaining comparable performance of 68.4%. Extending to the last four layers with uniform fusion further improves alignment to 69.6%. Introducing learnable weights across layers enables the model to dynamically attend to informative representations, boosting accuracy to 70.0%. Finally, adding the lightweight refinement block leads to best results of 70.4% with only 2.4M parameters.

On the Effect of Unified Post-Training. Tab. 5 evaluates our efficient post-training phase designed to enhance both understanding and generation tasks. We compare stan-

- Table 4. Ablation on the Mobile Conditioning Projector (MCP). We study the effect of layer fusion, learnable weighting, and the refinement block.

Proj. # Layers Fusion Compress CA Acc. (%) Params (M)

MLP – – – – 68.5 3.3 MCP 1 Uniform ✓ – 68.4 1.7 MCP 4 Uniform ✓ – 69.6 1.7 MCP 4 Learnable ✓ – 70.0 1.7 MCP 4 Learnable ✓ ✓ 70.4 2.4

- Table 5. Effect of Unified Post-Training. Our post-training data improves both understanding and generation alignment when using joint quadruplets.

Method Und. Acc. (%) Gen. Acc. (%)

SFT 60.5 73.3 SFT + Post-Train (image-text pairs) 60.6 73.4 SFT + Post-Train (quadruplets) 62.1 74.2

dard SFT against two post-training variants. Adding posttraining with generation-only triplets slightly improves results across benchmarks, showing better consistency in generative alignment. When generation and understanding triplets are used jointly, we observe measurable improvements, increasing average accuracy on seven image understanding tasks from 60.5% to 62.1% and GenEval by 1%. These results demonstrate that multi-objective post-training is a straightforward yet effective approach to enhance crossmodal coherence without need for large-scale pre-training.

#### 4.5. Edge Deployment

To assess the practicality on consumer devices, we evaluate recent unified methods below 2B parameters on three representative edge platforms: MacBook M2 Pro, NVIDIA Jetson Orin Nano, and iPhone 17 Pro. Tab. 6 reports inference times for visual understanding (vision encoder + text token forward time, TTFT) and total latency for image generation with 20 denoising steps. Mobile-O-0.5B demonstrates notable efficiency gains over prior unified models. On the MacBook M2 Pro, it is 2–8× faster than Janus and Show-O for understanding and 11–46× faster for image generation. On Jetson Orin Nano, Mobile-O-0.5B generates images in only 4 s, vs. 22–52 s for other methods. On iPhone 17 Pro, Mobile-O-0.5B achieves vision encoder latency of 102 ms, TTFT of 248 ms, and image generation in 3.0 s, highlighting its suitability for real-world deployment.

For mobile deployment, Mobile-O-0.5B components are converted using MLX [2] and CoreML [1]. The language model runs in MLX Swift with 8-bit weights on GPU for efficient token decoding, while the vision encoder, DiT backbone, VAE decoder, and MCP are exported to Core ML in float32, keeping the total memory footprint below 2GB.

- Table 6. Image understanding and generation performance comparison on MacBook M2 Pro, Jetson Orin Nano, and iPhone for Mobile-O-0.5B. Vision Enc. and TTFT denote understanding latency, while Latency indicates image generation latency.

Model Vision Enc. (ms) TTFT (ms) Latency (s) MacBook M2 Pro

Janus 783 ± 244 289 ± 19 201 ± 15614 JanusFlow 1909 ± 466 935 ± 152 24 ± 0.8 Show-O 699 ± 107 797 ± 5 47 ± 0.2 Mobile-O-0.5B (Ours) 56 ± 7 187 ± 31 4 ± 0.5

###### Jetson Orin Nano

Janus 745 ± 19 749 ± 19 44 ± 0.8 JanusFlow 741 ± 27 745 ± 27 22 ± 0.1 Show-O 403 ± 4 720 ± 14 52 ± 4

Mobile-O-0.5B (Ours) 88 ± 7 488 ± 9 4 ± 0.6 iPhone 17 Pro Mobile-O-0.5B (Ours) 102 ± 4 248 ± 10 3 ± 0.5

### 5. Conclusion

We introduce a unified vision–language–diffusion model, Mobile-O, with a new quadruplets format for unified posttraining and mobile conditioning projector to achieve highquality image understanding and text-to-image generation on edge devices. Experiments on MacBook M2 Pro, Jetson Orin Nano, and iPhone device show that Mobile-O outperforms recent unified models in both latency and memory efficiency, while preserving visual fidelity and semantic accuracy. Mobile-O-0.5B maintains a memory footprint below 2GB on iPhone within ∼3 seconds, making it practical for real-time on-device deployment.

### 6. Acknowledgment

The computations were enabled by resources provided by NAISS at Alvis partially funded by Swedish Research Council through grant agreement no. 2022-06725, LUMI hosted by CSC (Finland) and LUMI consortium, and by Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the NSC.

### References

- [1] Core ml: Integrate machine learning models into your app. https://developer.apple.com/ documentation/coreml, . Accessed: 2025-11-13.
- [2] Mlx: Machine learning for apple silicon. https:// opensource.apple.com/projects/mlx, . Accessed: 2025-11-13.
- [3] Junsong Chen et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [4] Jiuhai Chen et al. Blip3-o: A family of fully open unified multimodal models–architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.

- [5] Junying Chen et al. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025.
- [6] Jierun Chen et al. Snapgen: Taming high-resolution text-toimage models for mobile devices with efficient architectures and training. In CVPR, 2025.
- [7] Xiangxiang Chu et al. Mobilevlm: A fast, strong and open vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023.
- [8] Xiangxiang Chu et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.
- [9] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [10] Chaorui Deng et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [11] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.
- [12] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In NIPS, 2023.
- [13] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.
- [14] Sungwoong Kim, Daejin Jo, Donghoon Lee, and Jongmin Kim. Magvlt: Masked generative vision-and-language transformer. In CVPR, 2023.
- [15] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [16] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2025.
- [17] Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. Onecat: Decoder-only auto-regressive model for unified understanding and generation. arXiv preprint arXiv:2509.03498, 2025.
- [18] Yixuan Li, Dongxu Li, Wenguan Li, and Yi Yang. Evaluating object hallucination in large vision-language models. In EMNLP, 2023.
- [19] Qian Liang, Yujia Wu, Kuncheng Li, Jiwei Wei, Shiyuan He, Jinyu Guo, and Ning Xie. Mm-r1: Unleashing the power of unified multimodal large language models for personalized image generation. arXiv preprint arXiv:2508.11433, 2025.
- [20] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268, 2024.
- [21] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha

- Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In CVPR, 2024.
- [22] Yiyang Ma et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In CVPR, 2025.
- [23] Andr´es Marafioti, Orr Zohar, Miquel Farr´e, Elie Bakouch, Pedro Manuel Cuenca Jim´enez, Cyril Zakka, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, et al. Smolvlm: Redefining small and efficient multimodal models. In Second Conference on Language Modeling, 2025.
- [24] Ahmed Masry, Xuan Long Do, Jianshuo Qi Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022.
- [25] OpenAI et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [26] Xichen Pan et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.
- [27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024.
- [28] Morgane Riviere et al. Gemma 2: Improving open language models at a practical size, 2024.
- [29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [30] Abdelrahman Shaker, Muhammad Maaz, Chenhui Gou, Hamid Rezatofighi, Salman Khan, and Fahad Shahbaz Khan. Mobile-videogpt: Fast and accurate video understanding language model. arxiv, 2025.
- [31] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019.
- [32] Keqiang Sun et al. Journeydb: A benchmark for generative image understanding. In NIPS, 2023.
- [33] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [34] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024.
- [35] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [36] Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.
- [37] Shengbang Tong, David Fan, Jiachen Li, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. In ICCV, 2025.

- [38] Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Fastvit: A fast hybrid vision transformer using structural reparameterization. In ICCV, 2023.
- [39] Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, et al. Fastvlm: Efficient vision encoding for vision language models. In CVPR, 2025.
- [40] Xinlong Wang et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [41] Chengyue Wu et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.
- [42] Enze Xie et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformer. arXiv preprint arXiv:2410.10629, 2024.
- [43] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In ICLR, 2025.
- [44] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.
- [45] Junzhe Xu, Yuyang Yin, and Xi Chen. Tbac-uniimage: Unified understanding and generation by ladder-side diffusion tuning. arXiv preprint arXiv:2508.08098, 2025.
- [46] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. In NeurIPS, 2025.
- [47] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [48] Xiang Yue et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.
- [49] Chunting Zhou et al. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.
- [50] Yichen Zhu, Minjie Zhu, Ning Liu, Zhiyuan Xu, and Yaxin Peng. Llava-phi: Efficient multi-modal assistant with small language model. In Proceedings of the 1st International Workshop on Efficient Multimedia Computing under Limited, 2024.

## Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device Supplementary Material

[Figure 54]

[Figure 55]

- Table 7. Ablation on the number of layers for the Mobile Conditioning Projector (MCP). We systematically vary the number of VLM layers aggregated by MCP to condition the diffusion process. All configurations use the final design of MCP: learnable fusion, compression , and channel attention (CA).

Proj. # Layers Fusion Compress CA Accuracy (%)

- MCP 1 - ✓ ✓ 68.7
- MCP 2 Learnable ✓ ✓ 69.8 MCP 4 Learnable ✓ ✓ 70.4 MCP 8 Learnable ✓ ✓ 70.2

### 7. Mobile Conditioning Projector Depth

(a) Text-to-Image generation (b) Image-to-Text generation

The Mobile Conditioning Projector (MCP) aggregates features from multiple VLM layers to provide rich semantic conditioning for the diffusion model. Tab. 7 investigates how the number of aggregated layers affects text-to-image generation quality on GenEval. Using a single layer yields

Figure 6. Mobile-O running natively on iPhone 17 Pro. We demonstrate real-world deployment of Mobile-O’s unified capabilities on consumer hardware. (a) Text-to-image generation: Given a detailed prompt describing a Bengal tiger. (b) Image-to-text generation: Mobile-O provides detailed visual descriptions, analyzing composition and subject positioning

- 68.7% accuracy, suggesting that features from one depth level provide insufficient semantic diversity for accurately capturing complex compositional prompts. Aggregating 2 layers with learnable fusion improves performance to
- 69.8%, demonstrating the value of combining features from different network depths. The best performance (70.4%) is achieved with 4 layers, striking an optimal balance between semantic richness and computational efficiency. Interestingly, further increasing to 8 layers slightly degrades performance to 70.2%, indicating that excessive aggregation may introduce redundant or conflicting information that complicates the conditioning process. With four layers, it suggests that mid-depth VLM features capture the most relevant semantic abstractions for guiding compositional image generation, while avoiding the diminishing returns and increased computational cost associated with deeper aggregation.

out cloud dependency, ensuring user privacy and enabling offline functionality—critical requirements for real-world mobile applications. This deployment validates our architectural optimizations for the design choices, including the Mobile Conditioning Projector, proving that an efficient yet effective unified model can maintain high-quality unified capabilities with less than 2GB of memory.

### 9. More Implementation Details

All experiments are conducted on a single node with 8 NVIDIA A100 GPUs (80GB VRAM). We employ DeepSpeed ZeRO-3 during Stage 1 to efficiently handle the 9M training samples and large model parameters, then switch to ZeRO-1 for the last two stages, where smaller dataset sizes allow for reduced communication overhead. Mixedprecision training with BF16 throughout due to better numerical stability with transformer architectures. TF32 is enabled for matrix multiplications to leverage Ampere architecture acceleration. Images for understanding tasks undergo bicubic interpolation to 1024×1024, while generation tasks use 512×512.

### 8. On-Device Mobile Deployment

- Fig. 6 demonstrates Mobile-O running natively on an iPhone 17 Pro, validating the practical feasibility of deploying unified models on consumer mobile devices. The implementation showcases both core capabilities within a chat-based interface: text-to-image generation produces a detailed Bengal tiger image from a complex compositional prompt in 3 seconds, while image-to-text generation provides rich visual descriptions analyzing scene composition, subject positioning, depth perception, and atmospheric qualities in 0.3 seconds for text token forward time. The chat-based interface enables seamless switching between understanding and generation tasks within a single unified model, showcasing practical mobile AI applications with-

We use LoRa with reduced rank (r=16) and α=32 to prevent overfitting during unified training on the smaller 105K quadruplet dataset while still allowing fine-grained adaptation. All LoRA modules use a dropout of 0.1 for regularization. All stages use cosine annealing with minimum learning rate thresholds: Stage 1: LR decays from 2e-4 to

Stage 1: Cross-Modal Alignment Stage 2: Supervised Fine-tuning Stage 3: Unified Post-Training Data Source JourneyDB (4M) + BLIP3o-60K + BLIP3o-60K +

BLIP3o-Short (5M) ShareGPT-4o (45K) ShareGPT-4o (45K) Total Samples 9M 105K 105K Format Prompt-image pairs Prompt-image pairs Quadruplet {p,ximg,q,a}

Learning Rate 2e-4 2e-4 1e-4 Batch Size 512 384 128 LR Schedule cosine w/ min LR cosine w/ min LR cosine w/ min LR Min LR 2e-6 1e-6 1e-6 LR Warmup Ratio 0.02 0.05 0.05 Optimizer AdamW (β2=0.95) AdamW (β2=0.95) AdamW (β2=0.95) Weight Decay 0.01 0.01 0.01 Epochs 5 20 7 Training Time ∼3 days ∼15 hours ∼5 hours Trainable Modules DiT + MCP DiT + MCP DiT + MCP + LLM + VE Frozen Modules VE + LLM + VAE VE + LLM + VAE VAE

- Table 8. Three-stage training setup for Mobile-O. Stage 1 establishes cross-modal alignment using large-scale image-text pairs. Stage 2 performs targeted fine-tuning to address weaknesses in complex gestures, common objects, and landmarks. Stage 3 introduces unified

multimodal post-training with quadruplet samples {p, ximg, q, a} for joint understanding and generation. All experiments were conducted on 8×A100 GPUs.

2e-6 over 50K steps with 2% warmup (1,000 steps), allowing aggressive initial learning while maintaining stability in later training. Stage 2: LR decays from 2e-4 to 1e-6 with 5% warmup, providing more gradual adaptation for the targeted fine-tuning phase. Stage 3: Reduced initial LR of 1e-4 (min: 1e-6) with 5% warmup accommodates the unified training paradigm’s increased complexity.

[Figure 56]

Prompt: Summarize the back cover text of the book.

The book is titled " From the Pest Zone: The New York Stories" and is edited by H.P. Lovecraft and S.T. Joshi. The back cover text describes the book as a collection of stories that explore the New York City environment and its inhabitants, including the supernatural and the paranormal. The text also mentions that the stories are part of a series and that the author, H.P. Lovecraft, is a prominent figure in the horror genre. The publisher is Houghton Mifflin, and the edition is the 1st.

Janus

### 10. More Image-to-Text Qualitative Results

The book is " The New York Stories" by H.R.R. Lowenthal, edited by S. T. Joshi and David F. Schultz, published by Houghton Mifflin Harcourt, Inc., in the "From the Pest Zone" series, with a barcode and ISBN number of 0-610-24510-X.

JanusFlow

- Fig. 7 evaluates the models’ ability to perform dense text understanding and information extraction from real-world imagery. The task requires reading small, low-contrast text from a book’s back cover and summarizing its bibliographic information—a challenging scenario combining OCR, reading comprehension, and structured information extraction. Mobile-O accurately identifies the book as ”From the Pest Zone: The New York Stories” authored by H.P. Lovecraft, correctly extracts the editors’ names (S.T. Joshi and David E. Schultz), identifies specific story titles mentioned in the synopsis, and even captures the price 15.00 USD. In contrast, competing models exhibit significant hallucinations and misidentify the book title, authors, and fail to display the price. These results validate MobileO’s robust text understanding capabilities even in challenging real-world conditions with dense text, complex layouts, and varying contrast levels.

Show-O

The book is a paperback edition of the book " The Book of the Night" by Robert Louis Stevenson.

The book is titled " From the Pest Zone: The New York Stories" and is authored by H. P. Lovecraft. The back cover text provides a brief synopsis of the book's content, mentioning that it is a collection of short stories from Lovecraft's works, including " The Shunned House," " The Horror in the Valley," and " The New York Stories." The back cover also includes a description of the book's themes and the author's background, highlighting his connection to New York City and his influence on the horror genre. The publisher is H. P. Lovecraft. The price is listed as $15.00.

Mobile-O (ours)

Figure 7. Qualitative comparison on dense text understanding and information extraction. We evaluate Mobile-O against other models on a challenging OCR and comprehension task requiring the model to read, parse, and summarize the back cover text of a book. Green text indicates correctly extracted information, while red indicates hallucinations or errors. Mobile-O demonstrates superior performance in accurately extracting key bibliographic details, including the correct title, author, editors, and price information from the densely-packed text on the book cover.

comparison spans multiple cognitive domains including scientific reasoning (organic chemistry reaction analysis), optical character recognition with challenging perspectives and lighting conditions (theater signage reading), fine-grained

Fig. 9 presents a comprehensive qualitative evaluation of visual understanding capabilities across unified visionlanguage models on diverse question-answering tasks. The

mountain landscape, Mobile-O captures more realistic geological textures and natural color grading, while SANA exhibits somewhat exaggerated saturation in the foreground flowers. The portrait comparison reveals Mobile-O’s superior handling of skin tones and facial features with more natural lighting and realistic depth of field. Mobile-O achieves these results while simultaneously supporting visual understanding tasks within the model.

Prompt SANA-0.6B Mobile-O (Ours)

[Figure 57]

[Figure 58]

A vibrant tropical rainforest scene with a scarlet macaw perched on a moss-covered branch, brilliant red and blue feathers vivid against emerald foliage, cascading waterfall visible through the dense canopy in soft focus

[Figure 59]

[Figure 60]

A breathtaking mountain landscape at golden hour, featuring snow-capped peaks reflecting pink and orange light, a crystal-clear alpine lake in the foreground mirroring the sky, wildflower meadows in purple and yellow dotting the green valleys

### 12. More Text-to-Image Qualitative Results

- Fig. 10 presents a comprehensive qualitative comparison between Mobile-O and recent unified models across diverse and challenging prompts. The comparison includes Janus [41], JanusFlow [22], and Show-O [43], evaluating generation quality on prompts ranging from fantastical scenes (underwater cities, fire-breathing dragons) to photorealistic scenarios (bio-luminescent bays, space nebulae, portrait photography). Mobile-O demonstrates competitive visual quality while maintaining significantly lower computational requirements suitable for mobile deployment. Notably, Mobile-O excels at rendering fine details and maintaining prompt adherence across complex compositional scenarios, such as the intricate architectural details in the underwater city scene and the nuanced lighting in the portrait photography example. While competing models occasionally produce visually striking results, MobileO achieves a favorable balance between generation quality, prompt fidelity, and computational efficiency. The nebula scene particularly highlights Mobile-O’s ability to capture subtle color gradations and spatial depth, while the elderly woman portrait demonstrates proficient handling of photorealistic skin textures and natural lighting.
- Fig. 11 showcases Mobile-O’s text-to-image generation

[Figure 61]

[Figure 62]

A hyper-realistic portrait of a 30-year-old Middle Eastern man, soft natural lighting, DSLR depth of field, subtle beard, cinematic tone.

- Figure 8. Qualitative comparison with SANA-0.6B on text-toimage generation. We compare Mobile-O (1.6B total parameters) against SANA-0.6B (2.6B total parameters), our generation baseline, on challenging prompts requiring photorealistic rendering, complex lighting, and fine-grained details. Mobile-O demonstrates competitive or superior visual quality across diverse scenarios, including wildlife photography, landscape composition, and portrait rendering. Best viewed zoomed in.

object recognition requiring specific domain knowledge (retro gaming console and software identification), text extraction from stylized fonts (comic book titles), and cultural artifact classification (ancient civilization identification) from MMMU [48], ChartQA [24], and TextVQA [31]. These results validate that Mobile-O’s mobile-optimized architecture preserves robust visual understanding capabilities, demonstrating that aggressive model compression need not compromise the ability to accurately interpret and reason about diverse visual information.

capabilities across diverse categories, including photorealistic portraits, macro nature photography, food imagery, and creative scenes with complex lighting effects. The model demonstrates proficiency in rendering fine details (facial features, textures), managing challenging optical effects (bokeh, volumetric lighting, caustics), and maintaining color accuracy across varied subjects. These results validate Mobile-O’s versatility in generating high-quality imagery across different styles and compositional complexities while operating within mobile computational constraints. The prompts used in Fig. 11 are provided in Tab. 9.

### 11. Comparison with Generation-Only Baseline

- Fig. 8 compares Mobile-O against SANA-0.6B, the generation component that serves as our baseline architecture. Despite Mobile-O having 1.6B total parameters compared to SANA-0.6B’s 2.6B parameters (38% reduction), Mobile-O achieves competitive or superior generation quality across diverse prompts. In the rainforest scene, Mobile-O produces sharper feather details and a more natural background compared to SANA’s slightly oversaturated rendering. For the

### 13. Limitations

Mobile-O currently reuses the same lightweight LLM from the unified VLM as its text encoder, rather than employing a dedicated standalone language model optimized solely for textual understanding. This design choice significantly reduces memory footprint and allows on-device deployment, but it may limit the expressiveness and depth of text repre-

Question Image Janus JanusFlow Show-O Mobile-O (Ours) Answer

Which statement is incorrect ?

[Figure 63]

- A. Product is aromatic
- B. Product has high dipole moment

The statement " The force is stronger than the force" is incorrect.

The statement that is incorrect is that the reaction is a double bond.

B C

C

C. Product has less resonance energy

D. Product is soluble in polar solvent

[Figure 64]

What is the name of the theatre?

Dominion Dominion Oscar Wilde. Dominion Dominion

[Figure 65]

The setup shows a classic SEGA console, likely a SNES, with a game being played on the screen.

The retro gaming a Sega console connected. The game being played is "OutRun," a racing game that was popular....

The retro gaming setup in the image suggests the gaming era is from the late .... 1980s or 1990s.

The game shown is "OutRun," ... from 1980s ... console visible is a Sega Genesis, ... "SEGA" logo on ..... console.

The gaming setup shown is a SEGA console. The game played is OutRun

Describe this retro gaming setup. What game is being played?

[Figure 66]

The Amazing Adventures of Captain America: The First Team

What is the title of comic strip?

Tell It to the Marines.

Tell it to the MARINES.

Tell it to the Marines.

Tell it to the Marines.

[Figure 67]

Which culture is the image from?

- A: Sumerian,
- B: Mycenaen,
- C: Etruscan,
- D: Minoan

Greek Egypt India C C

- Figure 9. Qualitative comparison of image-to-text across unified models below 2B. Mobile-O is compared against Janus [41], JanusFlow [22], and Show-O [43] on diverse visual question answering tasks, including scientific reasoning, OCR, object recognition, and cultural knowledge. Green indicates correct answers, red indicates errors. Mobile-O demonstrates competitive visual understanding, despite its mobile-optimized architecture, correctly answering complex questions that require fine-grained visual analysis and domain knowledge.

sentations compared to approaches that use larger text-only models. For instance, SANA [42] adopts Gemma-2B-it [28] as a dedicated text encoder, benefiting from a more powerful linguistic backbone that can yield better alignment.

However, integrating such a model into Mobile-O is currently impractical for on-device deployment. A 2Bparameter model in FP16 requires approximately 4.0 GB just for the weights alone, excluding memory for activations, attention caches, and runtime overhead, which typ-

ically increases total memory requirements by several additional GBs. This exceeds the memory constraints of most mobile and resource-limited edge devices, where efficiency and low latency are core deployment objectives.

Prompt Janus JanusFlow Show-O Mobile-O (Ours)

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

A mesmerizing underwater city at night, futuristic buildings adorned with neon lights, marine creatures illuminated by artificial lights, a sense of wonder and exploration, Illustration, digital art with vibrant colors.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Create a hyperrealistic image of a tranquil bay in Mexico on a clear night, with water brimming with plankton, mediumsized waves, and a starry sky. The perspective should be from the beach looking out to sea, with the bay surrounded by lush jungle and palm trees visible only on the right and left sides of the frame. The water should be crystalclear, allowing the viewer to see the mesmerizing glow of the plankton in the water. The starry sky above should enhance the sense of serenity and calm.

a close-up of a fire-spitting dragon, bright flames illuminating the scene, well-lit, cinematic shot

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

A close-up of a fire-spitting dragon, bright flames illuminating the scene, well-lit, cinematic shot

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Nebula in deep space captured by telescope, billowing clouds of ionized hydrogen gas glowing in vibrant magenta and deep purple, wisps of cyan and blue oxygen emissions, distant stars scattered throughout creating depth, dark nebulae silhouettes, astronomical photography, ultra high resolution, realistic space photography

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Close-up portrait of an elderly woman with deep wrinkles and wise eyes, silver hair in a bun, soft window lighting illuminating her face, gentle smile, warm skin tones, highly detailed skin texture, shallow depth of field, photorealistic, professional portrait photography

Figure 10. Qualitative comparison of text-to-image generation across unified models below 2B. Mobile-O is compared against Janus [41], JanusFlow [22], and Show-O [43] on challenging prompts spanning fantasy, photorealism, and scientific visualization. Despite its mobile-optimized architecture, Mobile-O maintains competitive visual quality and prompt adherence. Best viewed zoomed in.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

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

###### Figure 11. Additional Text-to-Image generation examples of Mobile-O. Best viewed zoomed in.

Table 9. Text-to-image generation prompts used for visualization.

Detailed Prompts for Image Generation in Fig. 11 — (i,j) denotes the image at row i, column j

- Row 1 Portrait Photography

- (1,1) Young woman with freckles, green eyes with detailed iris and catchlight, natural skin texture, flowing red hair catching golden hour sunlight, warm vibrant lighting, gentle expression, shallow depth of field
- (1,2) Elderly Asian man with silver hair and peaceful expression, kind brown eyes, colorful traditional clothing, bright natural window lighting, detailed skin texture
- (1,3) Young man with curly hair and beard, bright turquoise shirt, outdoor lighting
- (1,4) Child with bright blue eyes and blonde hair, curious expression, rosy cheeks, individual eyelashes visible, natural freckles, bright daylight from side creating dimension, colorful playground in blurred background

- Row 2 Flowers & Garden Scenes

- (2,1) Sunflower field with detailed centers, yellow petals with texture, bees visiting flowers, blue sky with white clouds, warm summer sunlight, cheerful atmosphere, depth of field
- (2,2) Single red tulip with dewdrops on petals, water droplets reflecting light, delicate petal veins, green stem, bright spring morning light, shallow depth of field
- (2,3) Cherry blossom branch in full bloom, pink petals with detailed stamens, some petals floating in air, bright blue sky background, spring sunshine creating glow, intricate branch structure
- (2,4) Cluster of wildflowers, purple lupines and orange California poppies with detailed petal texture, yellow daisy centers, white Queen Anne’s lace intricate patterns, bright midday sun, morning dew

- Row 3 Food & Culinary Compositions

- (3,1) Rainbow cake slice showing colorful layers, white frosting, sprinkles, bright studio lighting
- (3,2) Basket with red apples on wooden kitchen table, bright natural window lighting
- (3,3) Colorful layered smoothie in clear glass, vibrant pink strawberry, purple acai, yellow mango, green spinach layers, fresh fruit garnish, straw, bright natural lighting
- (3,4) Ice cream sundae with rainbow sprinkles, individual sprinkle shapes and colors sharp, melting ice cream texture, whipped cream peaks, glossy cherry, colorful syrup drizzle, bright studio lighting

- Row 4 Colorful Objects & Items

- (4,1) Colorful hot air balloons in mid-air, red, yellow, and blue, wicker baskets, golden morning sunlight, blue sky, countryside below
- (4,2) Colorful kite with rainbow geometric patterns, ribbon tails, bright blue sky, sunlight
- (4,3) Ornate stained glass window with intricate patterns in reds, blues, yellows, greens, bright sunlight streaming through, colorful light on floor
- (4,4) Soap bubble bursting in mid-air, water droplets spraying outward, iridescent rainbow colors on fragmenting bubble surface, dynamic motion, dramatic lighting

- Row 5 Natural Landscapes & Outdoor Scenes

- (5,1) Golden retriever dog sitting on tropical beach, turquoise water, white sand, bright blue sky, sunny day
- (5,2) Autumn forest path with vibrant orange, red, and yellow fall foliage, sun rays piercing through misty air, leaves falling mid-flight, dramatic golden light beams, warm glowing atmosphere
- (5,3) Orange and white clownfish among pink sea anemone tentacles with texture, vibrant purple and yellow corals with polyp detail, blue tang fish nearby, bright sunlight filtering through water creating god rays, bubbles rising
- (5,4) Landscape after rain with vibrant double rainbow arching across sky, green rolling hills with visible grass texture, wildflowers, small farmhouse, sunlight breaking through dramatic clouds, puddle in foreground reflecting rainbow, wet grass sparkling

