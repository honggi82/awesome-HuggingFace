[Figure 1]

2026-1-19

## VInO: A Unified Visual Generator with Interleaved OmniModal Context

[Figure 2]

###### Junyi Chen1,†, Tong He1, Zhoujie Fu3, Pengfei Wan2, Kun Gai2 and Weicai Ye2,

1Shanghai Jiao Tong University, 2Kling Team, Kuaishou Technology, 3Nanyang Technology University

An Akita dog wearing a tiny yellow raincoat, standing in gentle drizzle.

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

# arXiv:2601.02358v2[cs.CV]16Jan2026

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

|[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

We present VINO, a unified visual generator that performs image and video generation and editing within a single framework. Instead of relying on task-specific models or independent modules for each modality, VINO uses a shared diffusion backbone that conditions on text, images and videos, enabling a broad range of visual creation and editing tasks under one model. Specifically, VINO couples a vision–language model (VLM) with a Multimodal Diffusion Transformer (MMDiT), where multimodal inputs are encoded as interleaved conditioning tokens, and then used to guide the diffusion process. This design supports multi-reference grounding, long-form instruction following, and coherent identity preservation across static and dynamic content, while avoiding modality-specific architectural components. To train such a unified system, we introduce a multi-stage training pipeline that progressively expands a video generation base model into a unified, multi-task generator capable of both image and video input and output. Across diverse generation and editing benchmarks, VINO demonstrates strong visual quality, faithful instruction following, improved reference and attribute preservation, and more controllable multi-identity edits. Our results highlight a practical path toward scalable unified visual generation, and the promise of interleaved, in-context computation as a foundation for general-purpose visual creation.

##### Project Page: https://sotamak1r.github.io/VINO-web/ Github Code: https://github.com/SOTAMak1r/VINO-code/

[Figure 44]

[Figure 45]

Corresponding author. † Work done during an internship at Kling Team, Kuaishou Technology.

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

[Figure 59]

[Figure 60]

[Figure 61]

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

[Figure 77]

[Figure 78]

[Figure 79]

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

###### Figure 1 | Showcase of VINO in image generation and image editing.

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

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

###### Figure 2 | Showcase of VINO in video generation and video editing.

### 1. Introduction

Recent diffusion models [5, 33, 48, 52, 70] achieve high-fidelity image and video synthesis, and multimodal assistants [2, 9, 49, 89] show that unified perception across text and vision is feasible. Yet, visual generation pipelines in practice remain fragmented: text-to-image [25, 43], text-to-video [1, 13, 76], and visual editing models [38, 51, 77] are developed and deployed separately, while multimodal LLMs offer unified perception but still rely on external diffusion backbones or decoders for high-resolution visual generation [16, 61]. This motivates the goal of building a unified visual generation framework.

Building such a unified generator, however, poses two fundamental challenges. First, generation tasks often rely on rich, descriptive captions [11], whereas editing tasks use short, imperative instructions that modify localized regions or attributes. Second, when text, images, videos, and other guidance signals are provided simultaneously, current models lack mechanisms to reliably disentangle and prioritize multimodal signals, resulting in semantic conflicts or inconsistent conditioning effects. These challenges suggest that a unified generator should process multimodal cues jointly and reason about their interactions.

To address these limitations, we propose VINO, a unified visual generator that performs interleaved generation and editing across both images and videos. VINO couples a vision–language model (VLM) with a Multimodal Diffusion Transformer (MMDiT), where the VLM processes all control signals (text, reference images and video) and exposes them as unified conditioning tokens, while the MMDiT operates as a token-based diffusion backbone over latent images and videos. This formulation enables a single diffusion backbone to accommodate multimodal control sources without task-specific modules.

Beyond unifying control signals, we introduce two architectural components: first, a key component is the use of learnable query tokens at the VLM input. Inspired by recent work [61], these learnable tokens provide a flexible interface between high-level instructions and low-level diffusion features and are jointly optimized with the generator. We integrate the learnable tokens processed by the VLM together with tokens from all modalities into the MMDiT. Empirically, this formulation yields smoother optimization, improved stability, and stronger generation and editing quality. Second, to preserve detailed reference information, we not only use the VLM-extracted features as conditioning signals, but also injects the corresponding VAE latents into the MMDiT [23, 45, 82]. Crucially, for each reference image or video, we reuse the same paired start–end tokens that wrap its VLM tokens to additionally wrap its VAE latent tokens. In this way, both the semantic (VLM) and latent (VAE) representations of the same reference are associated with the same boundary markers, enabling the MMDiT to consistently recognize and group features originating from the same visual source.

While the architecture provides unified conditioning, directly training such a model remains challenging due to heterogeneous instruction formats. First, we utilize image and video data with long captions to align the output space of the VLM with the condition space of MMDiT. To adapt the model to short captions and prompt the learnable tokens to acquire the ability to refine short captions, we introduce a mixed training phase that incorporates both long and short captions. After the model has adapted to both long and short captions, we proceed to train its multi-task capabilities.

We evaluate VINO across diverse image and video generation and editing tasks, including textto-image, text-to-video, instruction-based image editing, reference-driven video generation, and instruction-based video editing. Empirically, VINO delivers strong visual quality, improved identity preservation and faithful instruction following across both images and videos, providing a more scalable path toward unified visual generation than existing task-specific diffusion pipelines.

In summary, this paper makes the following contributions:

- 1. We present VINO, a unified diffusion-based architecture that handles both image and video generation and editing within a single framework by coupling a VLM with a MMDiT.
- 2. We incorporate learnable query tokens into the VLM’s input and provide an empirical study demonstrating that they improve multimodal conditioning, stabilize optimization.
- 3. We propose a token-boundary mechanism that reuses special VLM tokens within the MMDiT’s VAE latent stream to maintain consistent grounding across semantic and latent representations of reference images and videos. This reduces identity swapping and attribute leakage in complex, multimodal reference scenes.

[Figure 140]

- 4. We introduce a progressive training strategy that upgrades a pretrained video model into a multi-task visual generator while preserving its original visual generation strength.

[Figure 141]

[Figure 142]

[Figure 143]

Denoise result

MMDiT Blocks

st ed st ed

st ed st ed

Noise

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

###### VAE Encoder VAE Encoder

###### VLM

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Replace the sunglasses worn by the person in the video with those with strange shapes as shown in the image 1

Learnable tokens

Image 1 System prompt + Instructions Reference image/video

Target image/video

- Figure 3 | Overview of the VINO pipeline. Our unified framework conditions generation on an interleaved omnimodal context that jointly encodes system prompts, prompts/instructions, reference images/videos, and learnable tokens. A frozen VLM processes textual instructions together with visual references, producing multimodal embeddings that are augmented with learnable tokens (purple) and separated by special tokens (vision start token and vision end token ). These interleaved multimodal representations are fed into the MMDiT blocks, which also receive VAE latents from the reference images or video. The MMDiT model performs denoising conditioned on the full multimodal context, enabling VINO to execute image and video generation as well as instruction-based editing within a single unified architecture.

[Figure 159]

[Figure 160]

### 2. Methods

In this section, we present our unified framework for multi-modal image and video generation/editing. Our objective is to design a system that accepts heterogeneous control signals—textual instructions, reference images or videos, and learnable tokens—and uses them to guide a diffusion-based visual generator. Following the high-level model pipeline (Figure. 3), we structure this section around three central components. We first describe how multi-modal conditions are processed by the vision–language model (VLM) to obtain coherent feature representations in Section 2.1. We then explain how these encoded conditions are injected into the Multi-Modal Diffusion Transformer (MMDiT) without causing ambiguity or incorrect cross-modal grounding in Section 2.2. Finally, we detail the training strategy that makes the entire architecture a unified, multi-task visual generator capable of supporting a wide spectrum of editing and generation tasks in Section 2.3.

- 2.1. Multi-Modal Conditions To handle diverse forms of input, we employ a frozen VLM model as the front-end encoder for all language and visual conditions. As illustrated in the Figure 4, the system prompt varies with the presence and number of input modalities. When no visual modality is provided, the user supplies only a text input, which serves as the sole condition for text-to-image or text-to-video generation. When visual inputs are present, they are first sorted by type (image, then video) and placed at the beginning of the prompt, each assigned a unique identifier such as Image 1 or Video 1. The user can then reference these identifiers in the text input to specify different visual conditions, enabling complex multimodal control. In addition, we append a set of learnable tokens at the end of the prompt to

System Prompt for Tasks Conditioned on Text

<|im_start|>system You are a helpful assistant aimed at generating an output video caption. <|im_end|> <|im_start|>user <|user_text|> <|im_end|> <|im_start|>assistant

System Prompt for Tasks Conditioned on Visual Modalities and Instruction

<|im_start|>system You are a helpful assistant aimed at generating an output video caption. <|im_end|> <|im_start|>user Image 1: <|user_image_1|> · · · Image n: <|user_image_n|> Video 1: <|user_video|> <|user_text|> <|im_end|> <|im_start|>assistant

- Figure 4 | System prompt for different condition task, where <|user_text|>, <|user_image|> and <|user_video|> denote the user-provided input conditions across different modalities. For brevity, we omit the <|vision_start|> and <|vision_end|> tokens for the visual modalities.

[Figure 161]

st ed st ed

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

3D RoPE t-axis

0 1 2 3 4 5 6 7 8 9 10

- Figure 5 | 3D RoPE strategy for the VAE branch in VINO. We apply a unified 3D RoPE schedule along the temporal axis to interleave different visual modalities in the MMDiT VAE branch. Each modality—single reference images, multi-frame reference videos, and noisy target latents—is placed on a shared RoPE timeline, separated by special tokens, which are projected from the VLM output. This structured RoPE layout enables the model to distinguish heterogeneous visual sources.

extract cross-modal features into a shared space. These tokens are also handled using causal masking rather than granting full bidirectional attention [61]. Finally, we use the penultimate layer hidden states of the VLM as the encoded conditioning, apply a two-layer Multi-Layer Perceptron (MLP) for feature projection, and then feed them into the subsequent MMDiT.

- 2.2. Interleaved OmniModal Context Although the VLM provides robust high-level multimodal semantics, it significantly compresses visual information, resulting in a lack of fine-grained spatial details and texture fidelity. Consequently, it cannot adequately handle tasks requiring precise structural control, such as local editing. To compensate for this information bottleneck, we complement the VLM embeddings with VAE-encoded latents of all visual modalities. As illustrated in the Figure 5, these VAE latents are arranged following the same ordering used in the VLM, and place the noised image/video latents at the end. However, simply concatenating image and video latents introduces ambiguity. To uniquely distinguish different visual conditions and to align each VAE latent with its corresponding VLM feature, we reuse the VLM’s <|vision_start|> and <|vision_end|> embeddings. After projecting these embeddings through an MLP to match the MMDiT input dimension, they are used to mark the boundaries of each visual latent block. This explicit boundary marking serves as a strong positional cue, allowing the attention mechanism to correctly effectively distinguish and interpret distinct visual conditioning inputs within

- the sequence.
- 2.3. Training the Unified Multi-Task Visual Generator To build a unified visual generator that supports multimodal conditioning, we start from a text-to-video diffusion model, as it already provides strong priors for temporal dynamics. To replace the original text encoder, we first align the output space of VLM with the model’s native text encoder. In this initial stage, only a two-layer MLP connector is trained to map between the two embedding spaces. Modern text-to-video models typically rely on long, well-structured textual prompts, whereas editing tasks often involve short instructions, creating a distributional gap. To bridge this gap, we adopt a progressive training strategy that gradually shifts the input-condition distribution. Specifically, we treat short prompts as an intermediate form between long prompts and concise editing instructions. In the second stage, we train the model with a mixture of long and short prompts to ensure robustness across both forms, and we begin updating the MMDiT parameters during this stage. Once the model has adapted to short-prompt inputs, we enter the final stage, where full multi-task mixed training is performed. The data mixture ratio for each stage is illustrated in the Figure 6. This allows the model to smoothly transition from structured text-to-video conditioning to instruction-based multimodal generation and editing.

long/short t2i 4%

image selection 10%

image selection 1%

image reconstrution 10%

multi-ID video gen 19%

long prompt t2v 15%

short prompt t2i 15%

long/short t2v 5%

short prompt t2v 15%

text to video 45%

#### stage1 stage2 stage3

instruction i2v 5%

text to image 40%

image edit 46%

long prompt t2i 15%

instruction video edit 13%

long prompt i2v 15%

short prompt i2v 15%

image-ref video edit 7%

image to video 5%

- Figure 6 | Training data distribution across the stages. This progressive strategy gradually transforms the base model from a pure text-to-video generator into a capable multi-task visual generator.

### 3. Experiments

In this section, we present experimental results demonstrating VINO’s performance across a variety of visual generation and editing tasks. We compare against several advanced baselines, including both open-source and close-source systems. Notably, most competing methods remain proprietary and do not offer a unified generator comparable to VINO. Additional visualizations are available on our project webpage.

- 3.1. Implement Details Datasets Composition. As illustrated in Figure 6, our training pipeline allows for a progressive capability acquisition, organized into multiple stages where each targets a distinct subset of tasks. To support this staged paradigm, we curate a comprehensive mixture of datasets, combining large-scale open-source image/video collections [7, 8, 16, 19, 39, 57, 60, 63, 68, 84, 86, 93] and high-quality distillation data from open-source models [18, 21, 38, 73, 91], ensuring broad coverage across visual generation and visual editing. To handle this diverse data efficiently, we implement a dynamic resolution bucketing strategy. Instead of resizing all inputs to a fixed square resolution, we dynamically rescale images and videos based on their original dimensions. Specifically, within each training stage, we ensure that all samples have similar effective areas (i.e., total pixel count) while preserving their original aspect ratios. This strategy maintains a balanced computational load across GPUs at every

- Table 1 | Training hyperparameters and trainable modules for the three-stage progressive training pipeline.

###### Stage 1 Stage 2 Stage3

Learning rate 1 × 10−4 4 × 10−5 2 × 10−5 LR scheduler Constant Constant Constant Weight decay 0.01 0.01 0.01 Gradient norm clip 1.0 1.0 1.0 Training steps 20k 4k 16k Training samples O(100)𝑘 O(100)𝑘 O(100)𝑘 AdamW betas (0.9, 0.99) (0.9, 0.95) (0.9, 0.95) Use ema no yes yes Ema decay - 0.9999 0.9999 Frame area 3202 6402 6402 Timestep shift 3 5 5

connectors learnable tokens

connectors learnable tokens MMDiT

connectors learnable tokens MMDiT

Trainable modules

training step, while simultaneously enabling the model to learn and generalize across a range of spatial resolutions.

Base Models. Regarding the architecture, we integrate strong pre-trained priors to accelerate convergence. We adopt Qwen3VL-4B-Instruction [9], one of the most recent and advanced vision–language models, as our multimodal encoder to provide strong understanding capabilities. For the visual generator, we initialize our model with HunyuanVideo [41], an advanced text-to-video backbone. As shown in Table 2, compared to other open-source baselines, HunyuanVideo demonstrates better prompt adherence, producing visually coherent outputs even with concise or minimal textual descriptions, making it an ideal foundation for our instruction-driven generation tasks.

Training details. Table 1 provides a comprehensive overview of the hyperparameters employed across our three-stage training pipeline. This progressive curriculum is engineered to incrementally expand the model’s capabilities while ensuring stability and efficient optimization. For all stages, we train using DeepSpeed ZeRO-2 [64], which reduces memory consumption and enables training large modules efficiently. We further apply gradient checkpointing to the MMDiT backbone to reduce activation memory during video-heavy tasks. To ensure that each GPU can sustain a local batch size of 1, we dynamically adjust the number of video frames and number of reference images depending on the task. Since different tasks have different computational costs, we enforce strict task synchronization across all GPUs at each step, preventing workload imbalance and ensuring optimal training throughput.

- Table 2 | Geneval [30] results comparing VINO with image and video models. Compared with its base model [41], VINO maintains strong text-to-image ability using only a small proportion of T2I samples in Stage 3. † refer to the methods using LLM rewriter [23].

Type Name Single obj. Two obj. Counting Colors Position Color attr. Overall ↑

ImageModels

SDv2.1 [65] 0.98 0.51 0.44 0.85 0.07 0.17 0.50 SDxl [62] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 Sana [81] 0.99 0.77 0.62 0.88 0.21 0.47 0.66 Emu3-Gen [75] 0.99 0.81 0.42 0.80 0.49 0.45 0.66 DALLE 3 [11] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 FLUX.1-dev [43] 0.99 0.81 0.79 0.74 0.20 0.47 0.67 SD3 [25] 0.99 0.94 0.72 0.89 0.33 0.60 0.74 Playgroundv3 [80] 0.99 0.95 0.72 0.82 0.50 0.54 0.76

VideoModels

Wan2.1-14B [73] 0.88 0.55 0.51 0.71 0.16 0.25 0.51 HunyuanVideo [41] 0.95 0.77 0.34 0.77 0.43 0.44 0.61 HunyuanVideo† [41] 0.96 0.88 0.43 0.84 0.66 0.47 0.71 VINO 0.95 0.72 0.33 0.80 0.21 0.49 0.59 VINO† 0.97 0.88 0.52 0.88 0.65 0.62 0.75

- Table 3 | VBench [35] results across closed-source and open-source video models. VINO retains strong text-to-video generation capability, achieving performance comparable to its base model [41]. Moreover, VINO—which uses a stronger VLM as the conditioning encoder—shows clear improvements in semantic score. These results demonstrate that our unified framework not only preserves the generative strengths of the base model but also benefits from enhanced multimodal conditioning through stronger VLMs. † refer to the methods using LLM rewriter [34].

Quality score

Semantic score

Aesthetic quality

Dynamic degree

Object class

Overall consistency

Type Name Total ↑

Sora [13] 84.28 85.51 79.35 63.46 79.91 93.93 26.26 Veo3 [76] 85.06 85.70 82.49 63.81 72.43 93.89 27.88 Kling1.6 [1] 83.40 85.00 76.99 64.81 62.22 93.34 26.04 Jimeng [14] 81.97 83.29 76.69 68.80 38.43 89.62 27.10 Gen-3 [67] 82.32 84.11 75.17 63.34 60.14 87.81 26.69

Closesource

StepVideo [58] 81.83 84.46 71.28 61.23 53.06 80.56 27.12 CogVideoX-5B [83] 81.91 83.05 77.33 61.88 69.51 85.07 27.65 Wan2.1-14B [73] 83.69 85.59 76.11 66.07 65.46 86.28 25.91 HunyuanVideo [41] 83.24 85.09 75.82 60.36 70.83 86.10 26.44 VINO 82.80 84.00 78.01 65.60 58.33 91.10 26.83 VINO† 83.17 83.69 81.08 68.11 55.56 91.17 27.00

Opensource

- 3.2. Visual Generation We initiate our quantitative evaluation by assessing foundational Text-to-Image (T2I) and Textto-Video (T2V) capabilities using Geneval [30] and VBench [35]. These benchmarks assess core visual generation abilities including object composition, color fidelity, semantic alignment, visual quality, and temporal dynamics. A primary concern in instruction tuning is catastrophic forgetting, where new tasks degrade original performance. However, despite Stage 3 allocating only a minor fraction of the data budget to standard T2I/T2V samples, VINO retains performance metrics highly comparable to the HunyuanVideo backbone. This empirical evidence confirms that our training strategy effectively mitigates model degradation, preserving the strong pre-trained generative priors while accommodating multimodal customization tasks.

We subsequently evaluate VINO on OpenS2V [86], a benchmark specifically designed for subjectdriven, reference-based video generation—a capability widely absent in standard T2V models. Here, VINO demonstrates clear advantages over its base model. As shown in Table 4, the model demonstrates well-balanced performance, indicating that VINO effectively internalizes the customized, referencealigned generation abilities introduced during later training stages.

Collectively, these results verify that VINO successfully safeguards the robust open-domain capabilities of HunyuanVideo, while substantially expanding the system toward multimodal, reference-guided, and instruction-driven customization. This validates the effectiveness of our unified framework in handling hybrid conditioning without architectural conflict.

- Table 4 | OpenS2V [86] open-domain results on subject-to-video generation. The benchmark evaluates multi-reference video generation across diverse categories, including humans, objects, and face identity consistency. VINO even surpasses several strong closed-source systems.

Type Name Total ↑ Aesthetics

Motion Smoothness

Motion Amplitude

FaceSim GmeScore NexusScore NaturalScore

Close

source

Pika2.1 [3] 51.88 46.88 87.06 24.71 30.38 69.19 45.40 63.32 Vidu2.0 [4] 51.95 41.48 90.45 13.52 35.11 67.57 43.37 65.88 Kling1.6 [1] 56.23 44.59 86.93 41.60 40.10 66.20 45.89 74.59

Opensource

SkyReels-A2 [26] 52.25 39.41 87.93 25.60 45.95 64.54 43.75 60.32 MAGREF [24] 52.51 45.02 93.17 21.81 30.83 70.47 43.04 66.90 Phantom-14B [50] 56.77 46.39 96.31 33.42 51.46 70.65 37.43 69.35 VACE-14B [38] 57.55 47.21 94.97 15.02 55.09 67.27 44.08 67.04 VINO 57.85 45.92 94.73 12.30 52.00 69.69 42.67 71.99

- Table 5 | Image editing results on the ImgEdit [85] benchmark. Despite receiving editing supervision only in stage 3, VINO acquires image-editing capability extremely quickly: after just 1k training steps in stage 3, the model already surpasses most open-source baselines. With full stage 3 training, VINO further improves, demonstrating the effectiveness of our progressive training strategy.

Type Name Average ↑ Adjust Remove Replace Add Compose Action

Gemini2.5 [22] 4.30 4.48 4.39 4.24 4.30 3.88 4.61 GPT4o [37] 4.30 4.52 4.09 4.45 4.36 4.10 4.83 Seedream4 [69] 4.46 4.52 4.47 4.52 4.44 4.29 4.78

source

Close

Bagel [23] 3.20 3.31 2.62 3.30 3.56 2.38 4.17 UniWorld-V1 [47] 3.26 3.64 3.24 3.47 3.82 2.96 2.74 OmniGen2 [78] 3.44 3.06 3.20 3.74 3.57 2.52 4.68 Step1x-EditV1.1 [51] 4.01 4.17 3.73 4.11 4.26 3.97 3.84 Flux-Kontext-Dev [43] 4.09 4.28 3.85 4.22 4.09 3.48 4.47 VINO (only 1𝑘 step) 3.82 3.75 3.03 3.71 4.12 3.23 4.40 VINO 4.18 4.25 4.37 4.00 4.18 4.36 4.51

Opensource

- 3.3. Visual Editing We assess the visual editing capabilities of VINO across both spatial (image) and spatio-temporal (video) domains.

Image editing. For image editing, we adopt two comprehensive benchmarks: ImgEdit [85] and GEdit [51], which jointly cover a broad range of editing skills including adjustment, removal, replacement, addition, composition and style change. Since our base model lacks text-rendering ability, we exclude the text-change subtask in GEdit. Even though HunyuanVideo has no inherent editing capability, our progressive training strategy successfully equips VINO with strong instruction-following behavior. Notably, when we evaluate the model after only 1𝑘 training steps in Stage 3, where editing instructions are introduced for the first time, VINO already surpasses many open-source models on ImgEdit—demonstrating the rapid adaptation enabled by our unified architecture and highlighting the inherent advantages of video models when applied to motion-aware image editing tasks [28]. This rapid improvement demonstrates the effectiveness of our unified architecture and training strategy in enabling fast adaptation and efficient skill transfer from generation tasks to editing tasks.

Video editing. To thoroughly evaluate the video editing capabilities of VINO, we conduct a quantitative comparison on the OpenVE-Bench [31], a comprehensive benchmark designed for assessing video

- Table 6 | Results on the GEdit [51] benchmark. GEdit measures semantic consistency (SC), perceptual quality (PQ), and an overall score (O) under GPT-based evaluation (denoted by G_). Since our base model [41] lacks text-rendering capability, we exclude the text-change subtask from evaluation.

Type Name Average ↑ Subject replace Style change

G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O

Gemini2.5 [22] 7.48 8.30 7.17 9.17 7.54 8.12 6.65 8.05 6.75 GPT4o [37] 8.06 7.80 7.48 8.68 7.68 8.06 8.45 6.81 7.34 Seedream4 [69] 8.33 8.00 7.72 9.20 7.93 8.45 8.90 6.86 7.69

source

Close

UniWorld-V1 [47] 5.04 7.56 4.98 5.77 7.03 5.52 5.50 6.97 5.33 OmniGen2 [78] 6.79 6.68 6.18 7.68 6.32 6.71 7.75 5.75 6.47 Flux-Kontext-Dev [43] 7.23 7.28 6.53 8.10 7.03 7.01 6.80 6.03 5.91 Bagel [23] 7.52 6.69 6.54 8.62 6.53 7.31 7.93 4.92 6.04 Step1x-EditV1.1 [51] 7.60 7.29 6.87 8.73 7.25 7.80 8.48 6.20 7.11 VINO 7.26 7.71 6.88 8.22 7.03 7.30 8.05 6.83 7.29

Opensource

editing performance. We compare VINO against a wide range of open-source baselines, including VACE-14B [38], OmniVideo [71], InsViE [79], Lucy-Edit [72], ICVE [46], Ditto [8], and OpenVE-Edit [31]. Following the OpenVE-Bench protocol, we employ two advanced Multimodal LLMs, Gemini

- 2.5 Pro [22] and Qwen3VL [9], as judges to evaluate the editing quality. As shown in Table 7 and Table 8, VINO significantly outperforms all competing methods. The consensus across both MLLM evaluators further confirms that VINO achieves precise adherence to editing instructions, consistently surpassing the baselines.

To visually assess the editing capabilities, we compare VINO with Lucy-Edit [72] and Ditto [8]. Adopting the Ditto evaluation protocol, we utilize their curated mini video editing test set1 and generate diverse editing instructions using GPT-5. As shown in Figure 7, the comparison highlights that VINO achieves more robust instruction comprehension and higher generative quality than the competing methods.

Together, these results demonstrate that through a unified modeling design and a progressive training pipeline, VINO achieves advanced visual editing capability across both images and videos, offering strong generalization, precise instruction following, and high-quality multimodal editing under a single integrated framework.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

InputvideoDittoOursLucy-Edit

Remove the green novelty glasses from the person on the right. Replace the plain background with a softly blurred indoor café setting. Transform the scene into a gritty vintage war-era film style.

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

###### Figure 7 | Qualitative comparison of video editing results between VINO , Ditto [8] and LucyEdit [72]. Given the same input video and editing instructions, VINO delivers markedly stronger instruction following and better visual quality.

1https://huggingface.co/datasets/QingyanBai/Ditto-1M/tree/main/mini_test_videos

###### Table 7 | Quantitative Comparison on OpenVE-Bench [31] with Gemini 2.5 pro.

Name Overall ↑ Global Style Background Change Local Change Local Remove Local Add Subtitle Edit Creative Edit Camera Edit

VACE-14B [38] 1.57 1.49 1.55 2.07 1.46 1.26 1.48 1.47 1.62 OmniVideo [71] 1.31 1.11 1.18 1.14 1.14 1.36 1.00 2.26 1.00 InsViE [79] 1.53 2.20 1.06 1.48 1.36 1.17 2.18 2.02 1.09 Lucy-Edit [72] 2.15 2.27 1.57 3.20 1.75 2.30 1.61 2.86 1.61 ICVE [46] 2.07 2.22 1.62 2.57 2.51 1.97 2.09 2.41 1.11 Ditto [8] 1.98 4.01 1.68 2.03 1.53 1.41 2.81 1.23 1.32 OpenVE-Edit [31] 2.49 3.16 2.36 2.98 1.85 2.15 2.91 2.31 2.02

- VINO 3.18 4.34 2.54 3.73 3.22 2.77 2.61 3.29 2.81

Table 8 | Quantitative Comparison on OpenVE-Bench [31] with Qwen3VL.

Name Overall ↑ Global Style Background Change Local Change Local Remove Local Add Subtitle Edit Creative Edit Camera Edit

VACE-14B [38] 3.01 3.46 2.81 2.47 3.99 1.76 4.41 2.17 3.09 OmniVideo [71] 3.66 3.41 4.11 3.75 4.52 2.80 4.95 1.13 3.62 InsViE [79] 3.25 3.63 2.68 2.82 3.56 2.25 4.77 3.36 3.61 Lucy-Edit [72] 3.77 3.64 3.25 3.93 3.95 3.92 4.23 4.19 3.54 ICVE [46] 3.76 3.87 3.51 3.87 4.50 3.77 4.68 3.54 2.84 Ditto [8] 3.44 4.48 3.52 2.89 3.53 2.48 3.69 4.14 3.33 OpenVE-Edit [31] 3.89 4.24 4.10 3.80 3.50 3.41 3.98 3.71 3.25

- VINO 4.34 4.78 4.46 4.41 4.45 4.43 3.39 4.60 4.08

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

GradnormLoss

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

- Figure 8 | Ablation on learnable tokens. Training curves comparing models with (purple) and without (blue) learnable tokens across tasks (T2V, T2I, I2V, and image reconstruction). Introducing learnable tokens substantially stabilizes optimization, yielding lower gradient variance, reduced gradient norms, and smoother, healthier loss trajectories. This demonstrates that learnable tokens provide more reliable multimodal conditioning and facilitate more stable convergence during unified visual generation training.

- 3.4. Ablation Study We conduct comprehensive ablation studies to isolate and validate the contributions of three pivotal components in our framework: (1) the integration of learnable tokens for robust multimodal alignment,

(2) the impact of image classifier-free guidance (Image CFG) on inference dynamics, and (3) the special token inserted between VAE latents to disentangle heterogeneous latent sequences.

Learnable tokens. As shown in Figure 8, introducing learnable tokens leads to a substantially more stable training process, characterized by smoother and more well-behaved optimization curves across all tasks (T2V, T2I, I2V, and reconstruction). In contrast, removing these tokens results in noticeably noisier gradients and less stable convergence behavior. Beyond stability, learnable tokens also enhance multimodal conditioning fidelity and semantic precision. As illustrated in Figure 9, the model equipped with learnable tokens performs more accurate object removal and replacement, demonstrating that these tokens are essential for effectively incorporating multimodal conditions, directly translating into better performance in complex generation and editing tasks.

[Figure 200]

[Figure 201]

[Figure 202]

Remove the elderly man wearing glasses.

[Figure 203]

[Figure 204]

[Figure 205]

Replace the fruits in the colander with tomatoes.

Input image Instruction Ours w/o Learnable tokens

- Figure 9 | Ablation on learnable tokens for image editing. Visual comparison of editing results with and without learnable tokens under multimodal conditioning. When provided with both the input image and an instruction, the model equipped with learnable tokens produces edits that more faithfully follow the instruction while preserving scene structure and appearance. In contrast, the model without learnable tokens often misinterprets the editing intent or generates semantically inconsistent content.

The guidance of the reference images. We investigate the role of Image CFG as a control knob for the fidelity-motion trade-off in reference-guided generation. Increasing the Image CFG scale significantly reinforces visual adherence, ensuring that the generated results strictly preserves the identity and content of the reference image (Figure 10). However, we observe a diminishing return: excessively high guidance scales constrain the generative process, suppressing the model’s inherent motion priors and causing the output to degenerate into static, frozen sequences. Consequently, a moderate image CFG is essential to strike the optimal balance, maintaining high identity fidelity while allowing for natural temporal dynamics.

Special Token Between Latents. Finally, we analyze the use of a special token to disentangle VAE latents from different modalities and different lengths. Simply concatenating VAE latents from static images and dynamic videos introduces ambiguity, leading the attention mechanism to erroneously interpret the static reference latents as part of the temporal video sequence. This misalignment manifests as severe artifacts, particularly structural distortions in the initial frames, as shown in the bottom row of Figure 11. Furthermore, this explicit separation ensures generalization robustness, allowing the model to handle variable-length inputs during inference without overfitting to specific training sequence patterns.

Together, these ablations confirm that these components are not redundant but foundational: they collectively ensure optimization stability, precise controllability, and effective modality disentanglement in multimodal generation.

### 4. Related Works

Recent advancements in visual generation have coalesced into a dynamic landscape characterized by three pivotal trends. First, Diffusion Probabilistic Models have emerged as the de facto paradigm for high-fidelity synthesis and editing, providing the fundamental generative backbone with versatile conditioning interfaces (Section 4.1). Concurrently, mirroring the unification seen in Large Language Models (LLMs), there is a paradigm shift towards Unified and Omni-Visual Generators, where disparate tasks and modalities are consolidated into single, versatile frameworks (Section 4.2). Complementing these structural evolutions, the maturation of Vision–Language Models (VLMs) has revolutionized generative control, enabling systems to leverage rich semantic understanding for precise guidance and

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

input images image cfg = 1 image cfg = 1.5 image cfg = 2

- Figure 10 | Qualitative comparison under different image-CFG scales. Increasing the image-CFG weight strengthens adherence to the visual identity provided by the reference images, yielding video frames that more closely match the subject’s appearance and attributes. However, excessively large image-CFG values over-constrain the generative process, suppressing motion diversity and leading to noticeably reduced temporal dynamics.

evaluation (Section 4.3). Our work is situated at the intersection of these developments, leveraging their collective strengths to enable robust multimodal conditional generation.

- 4.1. Diffusion-Based Generation and Editing Diffusion models have enabled high-fidelity image [25, 43, 66, 81] and video [17, 41, 73, 83] synthesis from natural language prompts and become the de facto backbone for a wide range of downstream editing methods. Instruction-based editing methods [12, 28] learn to follow free-form text instructions to modify an input image by training on synthetic paired data. Other lines of work focus on adding explicit spatial control signals: ControlNet [90] augments a frozen text-to-image model with trainable conditional branches, enabling precise geometric or structural control while preserving the base model’s generation quality. Subsequent research explores training-free or inversion-based editing, using techniques like null-text inversion [59], iterative noising [29], or latent token manipulation [15] to reconstruct an input image in the diffusion latent space and then apply text-guided edits while maintaining identity and background.

- 4.2. Unified and Omni Visual Generation Inspired by how large language models unify diverse NLP tasks, recent work has started to pursue “unified” visual generators. Some works [40, 47, 78] propose diffusion models for unified image generation that support text-to-image, reference-guided generation, image editing, and other visualconditional tasks within a single framework, largely through a carefully designed conditioning space and multi-task training. On the video side, unified video frameworks [56] aim to jointly support video understanding, generation, and instruction-based editing, typically by sharing a video encoder/decoder while exposing different heads or objectives for various tasks. There are also works [38] on unified in-context video editing that attempt to handle multiple editing conditions in a single model, often by conditioning on example clips or prompts and relying on a generalized editing prior. A recent survey on unified models for image understanding and generation further emphasizes the trend toward “all-in-one” models, but also points out that most current systems remain limited either to images or to a narrow space of video tasks.

- 4.3. Vision–Language Models for Editing and Generative Control As VLMs have rapidly advanced in perception, reasoning, and instruction following, recent work increasingly leverages them as controllers, teachers, or evaluators for diffusion-based editing. Datacentric approaches such as HQ-Edit [36] use GPT-4V [2] and DALL·E 3 [11] to synthesize high-quality editing pairs and automatically score alignment and coherence, effectively turning VLMs into both dataset generators and supervisory signals. Model-centric methods [42, 92] integrate VLMs directly

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

inputvideow/specialtokenw/ospecialtoken

input image

[Figure 219]

[Figure 220]

[Figure 221]

Instruction: Replace the man on the right side of the video with the female superhero in the image

[Figure 222]

[Figure 223]

[Figure 224]

First frame Last frame

- Figure 11 | Ablation on the special token for separating VAE latents. Without special token (bottom row), the model incorrectly entangles the temporal structure of the input video with static image latents, causing noticeable artifacts—most prominently distorted structures in the first generated frame.

into the editing pipeline to provide more precise and localized edits or use as a training reward. On the evaluation side, VLM-based metrics [36, 51] provide instruction-aware assessments by detecting what changed and whether the modifications adhere to user intent. Collectively, these works demonstrate a growing trend in which VLMs play a central role in instruction decomposition, semantic localization, supervision, and evaluation for controllable generative editing.

### 5. Conclusion

We presented VINO, a unified visual generator capable of performing both image and video generation and editing under a single framework. By carefully designing the model components and a conditioning pipeline that accepts interleaved omnimodal context, VINO can seamlessly integrate heterogeneous inputs and handle a broad spectrum of visual tasks. Extensive comparisons demonstrate the effectiveness and strong performance of our approach. Moreover, our progressive training strategy enables the model to retain the generative strengths of its base video backbone while acquiring robust multitask capabilities, ultimately yielding a coherent and unified visual generator. VINO offers a flexible, scalable foundation for many-to-many visual generation and paves the way toward more general-purpose multimodal generative systems.

Limitations and future works. Despite its versatility, VINO inherits several limitations.

- • Our base model lacks text-rendering capability, which places VINO at a disadvantage on benchmarks that explicitly evaluate text editing or text generation.
- • Existing instruction-based editing datasets are generally of lower quality compared with largescale generation datasets. These editing samples often contain limited motion and simpler visual structures, which may bias the target distribution. Consequently, after incorporating instruction-edit tasks, the model may exhibit slightly reduced visual fidelity or motion richness compared with the original base generator. These limitations highlight the need for higherquality multimodal editing datasets and more balanced training strategies in future work.
- • Within MMDiT the complexity of full attention grows quadratically. Therefore, supplying a reference video together with a large number of reference images, the inference latency increases substantially. This suggests that exploring more efficient visual generation backbones is a promising direction.

- • The modalities currently supported in VINO are ultimately constrained by the VLM. Although certain modalities can be converted into others (e.g., transforming audio into text, or using video to represent motion), our training pipeline only considers the most general modalities—namely text, image, and video. Exploring more powerful and more comprehensive VLMs constitutes a promising direction for future research.

### Acknowledgements

This work was done during Junyi Chen’s internship at Kling Team, Kuaishou Technology. We acknowledge Google DeepMind for the paper template [76] inspiration.

### References

- [1] K. AI. Kling ai, 2025. https://klingai.kuaishou.com/.
- [2] O. AI. Gpt4v, 2023. https://openai.com/index/gpt-4v-system-card/.
- [3] P. AI. Pika ai, 2024. https://pika-art.net/pika-2-1/.
- [4] V. AI. Vidu ai, 2024. https://www.vidu.cn/.
- [5] M. S. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.
- [6] X. An, Y. Xie, K. Yang, W. Zhang, X. Zhao, Z. Cheng, Y. Wang, S. Xu, C. Chen, C. Wu, H. Tan, C. Li, J. Yang, J. Yu, X. Wang, B. Qin, Y. Wang, Z. Yan, Z. Feng, Z. Liu, B. Li, and J. Deng. Llava-onevision-1.5: Fully open framework for democratized multimodal training. In arXiv, 2025.
- [7] J. Bai, M. Xia, X. Fu, X. Wang, L. Mu, J. Cao, Z. Liu, H. Hu, X. Bai, P. Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025.
- [8] Q. Bai, Q. Wang, H. Ouyang, Y. Yu, H. Wang, W. Wang, K. L. Cheng, S. Ma, Y. Zeng, Z. Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025.
- [9] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [10] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [11] J. Betker, G. Goh, L. Jing, T. Brooks, J. Wang, L. Li, L. Ouyang, J. Zhuang, J. Lee, Y. Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [12] T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [13] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.
- [14] ByteDance. Bytedance, 2025. https://jimeng.jianying.com/.
- [15] M. Cao, X. Wang, Z. Qi, Y. Shan, X. Qie, and Y. Zheng. Masactrl: Tuning-free mutual selfattention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22560–22570, 2023.
- [16] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [17] J. Chen, Y. Zhao, J. Yu, R. Chu, J. Chen, S. Yang, X. Wang, Y. Pan, D. Zhou, H. Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.

- [18] J. Chen, H. Zhu, X. He, Y. Wang, J. Zhou, W. Chang, Y. Zhou, Z. Li, Z. Fu, J. Pang, et al. Deepverse: 4d autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025.
- [19] T.-S. Chen, A. Siarohin, W. Menapace, E. Deyneka, H.-w. Chao, B. E. Jeon, Y. Fang, H.-Y. Lee, J. Ren, M.-H. Yang, and S. Tulyakov. Panda-70m: Captioning 70m videos with multiple crossmodality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [20] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [21] G. Cheng, X. Gao, L. Hu, S. Hu, M. Huang, C. Ji, J. Li, D. Meng, J. Qi, P. Qiao, et al. Wananimate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025.
- [22] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [23] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [24] Y. Deng, X. Guo, Y. Yin, J. Z. Fang, Y. Yang, Y. Wang, S. Yuan, A. Wang, B. Liu, H. Huang, et al. Magref: Masked guidance for any-reference video generation. arXiv preprint arXiv:2505.23742, 2025.
- [25] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [26] Z. Fei, D. Li, D. Qiu, J. Wang, Y. Dou, R. Wang, J. Xu, M. Fan, G. Chen, Y. Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025.
- [27] C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025.
- [28] Z. Fu, X. Zeng, J. Lan, X. Liao, C. Chen, J. Chen, J. Wei, W. Cheng, S. Liu, Y. Chen, et al. imontage: Unified, versatile, highly dynamic many-to-many image generation. arXiv preprint arXiv:2511.20635, 2025.
- [29] D. Garibi, O. Patashnik, A. Voynov, H. Averbuch-Elor, and D. Cohen-Or. Renoise: Real image inversion through iterative noising. In European Conference on Computer Vision, pages 395–413. Springer, 2024.
- [30] D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [31] H. He, J. Wang, J. Zhang, Z. Xue, X. Bu, Q. Yang, S. Wen, and L. Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.
- [32] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [33] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [34] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [35] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [36] M. Hui, S. Yang, B. Zhao, Y. Shi, H. Wang, P. Wang, Y. Zhou, and C. Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.
- [37] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [38] Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.
- [39] X. Ju, Y. Gao, Z. Zhang, Z. Yuan, X. Wang, A. Zeng, Y. Xiong, Q. Xu, and Y. Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024.
- [40] X. Ju, W. Ye, Q. Liu, Q. Wang, X. Wang, P. Wan, D. Zhang, K. Gai, and Q. Xu. Fulldit: Multi-task video generative foundation model with full attention. arXiv preprint arXiv:2503.19907, 2025.
- [41] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [42] N. Kumari, S.-Y. Wang, N. Zhao, Y. Nitzan, Y. Li, K. K. Singh, R. Zhang, E. Shechtman, J.-Y. Zhu, and X. Huang. Learning an image editing model without image editing pairs. arXiv preprint arXiv:2510.14978, 2025.
- [43] B. F. Labs. Black forest labs, 2024. https://github.com/black-forest-labs/flux.
- [44] K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo, L. Wang, and Y. Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark, 2024.
- [45] C. Liao, L. Liu, X. Wang, Z. Luo, X. Zhang, W. Zhao, J. Wu, L. Li, Z. Tian, and W. Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.
- [46] X. Liao, X. Zeng, Z. Song, Z. Fu, G. Yu, and G. Lin. In-context learning with unpaired clips for instruction-based video editing. arXiv preprint arXiv:2510.14648, 2025.
- [47] B. Lin, Z. Li, X. Cheng, Y. Niu, Y. Ye, X. He, S. Yuan, W. Yu, S. Wang, Y. Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.
- [48] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [49] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [50] L. Liu, T. Ma, B. Li, Z. Chen, J. Liu, G. Li, S. Zhou, Q. He, and X. Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.

- [51] S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [52] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [53] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [54] Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X.-C. Yin, C.-L. Liu, L. Jin, and X. Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), Dec. 2024. ISSN 1869-1919. doi: 10.1007/s11432-024-4235-6. URL http: //dx.doi.org/10.1007/s11432-024-4235-6.
- [55] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.
- [56] J. Luo, J. Lin, Z. Zhang, B. Wu, M. Fang, L. Chen, and H. Tang. Univid: The open-source unified video model. arXiv preprint arXiv:2509.24200, 2025.
- [57] Y. Luo, J. Bai, X. Shi, M. Xia, X. Wang, P. Wan, D. Zhang, K. Gai, and T. Xue. Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140, 2025.
- [58] G. Ma, H. Huang, K. Yan, L. Chen, N. Duan, S. Yin, C. Wan, R. Ming, X. Song, X. Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.
- [59] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023.
- [60] K. Nan, R. Xie, P. Zhou, T. Fan, Z. Yang, Z. Chen, X. Li, J. Yang, and Y. Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.
- [61] X. Pan, S. N. Shukla, A. Singh, Z. Zhao, S. K. Mishra, J. Wang, Z. Xu, J. Chen, K. Li, F. Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.
- [62] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [63] Y. Qian, E. Bocek-Rivele, L. Song, J. Tong, Y. Yang, J. Lu, W. Hu, and Z. Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv:2510.19808, 2025.
- [64] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.
- [65] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [66] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [67] Runway. Runway, 2024. https://runwayml.com/research/introducing-gen-3-alpha.
- [68] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35: 25278–25294, 2022.
- [69] T. Seedream, Y. Chen, Y. Gao, L. Gong, M. Guo, Q. Guo, Z. Guo, X. Hou, W. Huang, Y. Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.
- [70] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [71] Z. Tan, H. Yang, L. Qin, J. Gong, M. Yang, and H. Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025.
- [72] D. Team. Lucy edit: Open-weight text-guided video editing. 2025. URL https://d2drjpui nn46lb.cloudfront.net/Lucy_Edit__High_Fidelity_Text_Guided_Video_Editi ng.pdf.
- [73] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [74] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [75] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [76] T. Wiedemer, Y. Li, P. Vicol, S. S. Gu, N. Matarese, K. Swersky, B. Kim, P. Jaini, and R. Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.
- [77] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [78] C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [79] Y. Wu, L. Chen, R. Li, S. Wang, C. Xie, and L. Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. arXiv preprint arXiv:2503.20287, 2025.
- [80] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.
- [81] E. Xie, J. Chen, J. Chen, H. Cai, H. Tang, Y. Lin, Z. Zhang, M. Li, L. Zhu, Y. Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.
- [82] J. Xie, Z. Yang, and M. Z. Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.
- [83] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [84] J. Ye, D. Jiang, Z. Wang, L. Zhu, Z. Hu, Z. Huang, J. He, Z. Yan, J. Yu, H. Li, et al. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025.
- [85] Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.
- [86] S. Yuan, X. He, Y. Deng, Y. Ye, J. Huang, B. Lin, J. Luo, and L. Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.
- [87] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, C. Wei, B. Yu, R. Yuan, R. Sun, M. Yin, B. Zheng, Z. Yang, Y. Liu, W. Huang, H. Sun, Y. Su, and W. Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.
- [88] B. Zhang, K. Li, Z. Cheng, Z. Hu, Y. Yuan, G. Chen, S. Leng, Y. Jiang, H. Zhang, X. Li, P. Jin, W. Zhang, F. Wang, L. Bing, and D. Zhao. Videollama 3: Frontier multimodal foundation models for image and video understanding. 2025. URL https://arxiv.org/abs/2501.13106.
- [89] H. Zhang, X. Li, and L. Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.
- [90] L. Zhang, A. Rao, and M. Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [91] L. Zhang, A. Rao, and M. Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= u1cQYxRI1H.
- [92] J. Zhou, J. Li, Z. Xu, H. Li, Y. Cheng, F.-T. Hong, Q. Lin, Q. Lu, and X. Liang. Fireedit: Fine-grained instruction-based image editing via region-aware vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13093–13103, 2025.
- [93] Y. Zhou, Y. Wang, J. Zhou, W. Chang, H. Guo, Z. Li, K. Ma, X. Li, Y. Wang, H. Zhu, et al. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025.

### Appendix

### A. Base Models

Qwen3-VL Qwen3-VL [9] is an advanced vision–language model belonging to the Qwen3 series. It extends large-language-model (LLM) capabilities into vision by integrating image and video encoders, enabling reasoning over diverse visual inputs and natural language instructions. This structure allows the model to build cross-modal attention without relying on handcrafted fusion modules, providing strong representational alignment across text and vision modalities. Given these properties, Qwen3-VL serves as our visual–instruction foundation: it provides cross-modal alignment, efficient visual tokenization, and empirically strong visual understanding and instruction-following capabilities, which are all essential for many-to-many reference-guided generation in our system.

HunyuanVideo HunyuanVideo [41] is a generative video model based on the Multimodal Diffusion Transformer (MMDiT) architecture [25], which leverages a token-based sequence framework to effectively model video generation. The MMDiT excels at learning multimodal conditioning, using full attention to directly integrate and condition both visual and multimodal inputs, enhancing the model’s ability to generate coherent and contextually aligned visual content. These characteristics closely match the requirements of many-to-many generation and editing in our unified setting. Consequently, HunyuanVideo offers a suitable video base model, enabling our approach to inherit advanced motion dynamics, coherent cross-modal grounding, and more controllable visual conditioning within a single diffusion framework.

### B. More Details

Training details. For each task, we define a separate dataloader. Before each training step, a task is selected probabilistically, and this selection is broadcasted from rank 0 to synchronize the tasks across all GPUs. This approach prevents idle times on any specific GPU by ensuring that all GPUs process the same task in each step. We use the default vision preprocessing configuration from Qwen3-VL for all visual inputs, but we limit the total number of video frames fed into Qwen3-VL to a maximum of 8 frames, with the default frame rate set to 1 FPS. During training, we randomly drop conditions [12, 32], with the drop probability for different modalities being the same but independent, set to 0.1 for each. Additionally, to maintain load balance, when the number of input images for a task becomes too large, we ensure that all GPUs either drop the condition simultaneously or not at all, preventing a significant discrepancy in the final token sequence lengths.

Proxy tasks. To enhance the model’s performance, we introduce two proxy tasks during the training process. These tasks are designed to help the model better learn and understand visual representations and relationships. In the stage 1, we employ an image reconstruction proxy task. Specifically, we feed an image into the Vision-Language Model (VLM) while bypassing the Variational Autoencoder (VAE). The model is trained to reconstruct the input image. This task helps the MMDiT model learn and align the visual features extracted by the VLM, strengthening the model’s ability to handle the correspondence between high-level semantics and low-level details. In the stage 2, we introduce an image selection proxy task. When multiple reference images are provided, we can refer to each image by its ID in the instruction. To help the model learn the relationship between image IDs and their corresponding visual content, this task involves randomly sampling 𝑘 images and prompting the model to reconstruct the 𝑖-th image. This allows the model to learn the correspondence between the image ID and the image itself, further improving its ability to handle dynamic image inputs and their relationships.

Inference details. Our inference method is based on [12, 32]. For text-conditioned generation tasks, we use a classifier-free guidance (cfg) value 𝑤𝑡𝑒𝑥𝑡 of 7, and for multimodal condition-controlled generation tasks, we set 𝑤𝑡𝑒𝑥𝑡 = 5 and 𝑤𝑖𝑚𝑎𝑔𝑒 = 1.5. HunyuanVideo natively supports cfg as an input parameter without the need for multiple inferences, but this requires additional post-training, which we did not adopt.

### C. Quantitative Results

- C.1. Visual Understanding In Table 9, we present the quantitative results related to the understanding capabilities of our model. While our approach primarily focuses on generation, it inherently incorporates components for understanding tasks. leveraging the strengths of our base VLM model, Although we did not train Qwen3-VL directly, we utilize it to evaluate various understanding-related metrics, which serves as an integral part of VINO. We report on several of these metrics to provide a comprehensive view of the model’s performance in visual understanding.

Table 9 | Quantitative results on visual understanding benchmarks. We report comparison of visual understanding performance on standard benchmarks. Params denotes the number of model parameters involved when performing visual understanding tasks. Results are reported for both understanding-only models and unified understanding–generation models.

Type Name Params MMMU [87] MMBench-EN [53] VideoMME𝑤/𝑜 𝑠𝑢𝑏 [27] MVBench [44] OCRBench [54] MathVista𝑚𝑖𝑛𝑖 [55]

Und.only

VideoLLaMA3 [88] 7B 48.8 - 66.2 69.7 828 67.1 InternVL3.5 [74] 4B 66.6 - 65.4 71.2 822 77.1 Qwen2.5-VL [10] 7B 58.6 83.5 65.1 69.6 864 68.2 LLaVA-OneVision1.5 [6] 4B 52.7 84.2 - - 800 67.9

Und.andGen.

show-o2 [82] 7B 48.9 79.3 57.4 55.8 - Janus-Pro [20] 7B 41.0 79.2 - - - UniWorld-V1 [47] 7B 58.6 83.5 - - - OmniGen2 [78] 3B 53.1 79.1 - - - Bagel [23] 7B 55.3 85.0 - - - 73.1 MetaQuery-XL [61] 7B 58.6 83.5 65.1 69.6 864 68.2 VINO 4B 67.4 83.9 69.3 68.9 881 73.7

- C.2. Visual Generation and Editing In this section, we present detailed quantitative results of VINO , as summarized in the table 10, 11 and 12.

Table 10 | Quantitative results on Vbench [35]. † refer to the methods using LLM rewriter [34].

Smoothness

Background

relationship

Appearance

consistency

consistency

consistency

Temporal

Temporal

flickering

Semantic

Aesthetic

Dynamic

Multiple

Quality

Human

Subject

Overall

Motion

objects

quality

quality

Spatial

degree

Object

Image

action

Scene

Color

score

score

class

style

style

Name Total ↑

VINO 82.80 84.00 78.01 95.99 26.83 26.29 24.73 46.74 67.85 90.77 94.00 69.04 91.09 64.43 65.58 58.33 98.73 99.31 97.65 VINO† 83.17 83.69 81.08 95.87 26.99 24.73 24.96 51.00 75.85 90.22 97.80 83.16 91.17 62.23 68.11 55.56 98.72 99.04 97.69

Table 11 | Quantitative results on ImgEdit [85].

Average adjust style background extract remove replace add compose action

4.18 4.25 4.44 3.99 3.56 4.37 4.00 4.18 4.36 4.51

### D. Qualitative results

Stage 3 training process. As shown in Figure 12, in Stage 3, the training process benefits greatly from the alignment achieved in the previous two stages, providing a strong initialization. At the start of training, the model begins by learning the correspondence between visuals and instructions. As training continues, it gradually improves and starts to capture more detailed features. Over time, the model becomes better at handling more complex details, showing clear progress as the training steps increase.

The effect of learnable tokens. In Figure 13, we demonstrate the role of the learnable token in the final model. When we choose longer captions, the model allocates more attention to the text features that are related to high-level semantics, while the learnable token learns to represent more low-level

###### Table 12 | Quantitative results on GEdit [51].

Average ↑ background change color material motion change ps human style change subject add subject remove subject replace tone transfer

G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O G_SC G_PQ G_O 7.26 7.71 6.88 7.90 7.48 7.30 8.40 7.53 7.51 7.55 7.03 7.10 4.83 8.58 4.91 4.41 8.51 4.90 8.05 6.83 7.29 8.17 7.82 7.73 7.98 8.26 7.63 8.22 7.03 7.30 7.05 8.00 7.09

Input image Step 0 Step 10k

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Make the clothing fabric from premium linen.

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Change the background to the sea.

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Can you add penguin eyes to this image?

- Figure 12 | Training progression from Stage 3, demonstrated through image editing tasks. The model benefits from the strong initialization achieved in earlier stages, gradually improving its ability to align visuals with instructions. As training progresses, the model refines its understanding, capturing finer details and showing clear progress in handling different tasks.

details. This aligns with our initial design: the learnable token acts as a more flexible interface for the VLM, optimized together with MMDiT, allowing the model to more effectively learn knowledge at different levels. However, when shorter captions are selected, the role of the learnable token becomes more pronounced.

[Figure 246]

[Figure 247]

The image showcases a character with a striking appearance. The character has a mohawk-style hairstyle with a vibrant pink hue. The side of their head is shaved, adorned with a metallic band that has a unique symbol on it. Their face is painted with bold red streaks, and they have piercings on their ears. The character is draped in a shiny, red cloak or garment. The background is a dilapidated, post-apocalyptic setting with rusted metal, broken windows, and a hazy atmosphere.

[Figure 248]

[Figure 249]

The image showcases a futuristic, mechanically intricate spaceship resembling a whale. The spaceship is predominantly white with blue accents, and it appears to be in flight, soaring above the clouds. The craft is equipped with various antennas, beams, and intricate designs, suggesting advanced technology. The backdrop is a clear blue sky with fluffy white clouds, and the overall ambiance of the image is serene yet awe-inspiring.

[Figure 250]

[Figure 251]

In a futuristic setting, a person oversees a large, transparent dome housing an immense creature, while advanced technology fills the surroundings.

[Figure 252]

[Figure 253]

An anthropomorphic plant figure relaxes on a couch in a festively decorated room with a fireplace, tree, falling snow, and floral arrangements.

Full model Remove learnable tokens

- Figure 13 | The effect of learnable tokens in the final model. Use the same initial gaussian noise.

