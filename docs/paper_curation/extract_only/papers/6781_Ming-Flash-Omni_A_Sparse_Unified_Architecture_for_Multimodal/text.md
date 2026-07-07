## arXiv:2510.24821v3[cs.CV]26Mar2026

### Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation

Inclusion AI, Ant Group∗

∗See Contributions section (Sec. 6) for full author list.

We propose Ming-Flash-Omni, an upgraded version of Ming-Omni, built upon a sparser Mixture-ofExperts (MoE) variant of Ling-Flash-2.0 with 100 billion total parameters, of which only 6.1 billion are active per token. This architecture enables highly efficient scaling (dramatically improving computational efficiency while significantly expanding model capacity) and empowers stronger unified multimodal intelligence across vision, speech, and language, representing a key step toward Artificial General Intelligence (AGI). Compared to its predecessor, the upgraded version exhibits substantial improvements across multimodal understanding and generation. Notably, it achieves strong performance on vision-language understanding benchmarks, with overall scores on par with Gemini 2.5 Pro, and enables seamless switching among multimodal tasks in multi-turn interactions. In speech, it achieves strong performance in contextual and dialect-aware ASR while enabling joint, continuous-generation of speech, sound, and music. In vision, it introduces generative semantic segmentation that achieves competitive standalone performance and enhances spatial control and editing consistency, alongside marked improvements in identity preservation, and high-fidelity inimage text rendering. Together, these capabilities demonstrate that a single unified model can serve as a practical foundation for general-purpose multimodal intelligence.

[Figure 1]

Date: Mar 26, 2026 Project Homepage: https://github.com/inclusionAI/Ming

#### 1 Introduction

In everyday life, humans naturally integrate visual and auditory cues to express ideas through speech or writing, while also forming vivid mental images from descriptions or concepts. This ability to visualize enhances creativity, problem-solving, and communication, serving as a core aspect of human intelligence and interaction. The ultimate goal of Artificial General Intelligence (AGI) is to replicate this human-like multimodal intelligence, evolving from a mere tool into a powerful agent that augments and liberates human productivity.

Driven by advances in Large Language Models (LLMs) and extensive training on large-scale multimodal datasets, Multi-modal Large Language Models (MLLMs) have demonstrated remarkable perceptual capabilities in both vision (Chen et al., 2024c; Bai et al., 2025b; KimiTeam et al., 2025; Xu et al., 2025c) and audio (Ding et al., 2025; Xu et al., 2025a,c), as well as generative capabilities in these two modalities (Huang et al., 2025a; Ding et al., 2025; OpenAI, 2025; Tong et al., 2024; Huang et al., 2025c; Pan et al., 2025; Xu et al., 2025c). Nevertheless, effectively integrating comprehension and generation across multiple modalities into a unified model remains challenging. While humans naturally learn by combining multiple modalities, leveraging their complementary strengths and interactions to enhance overall learning efficiency, building a unified Omni-MLLM is hindered by representational disparities and modality imbalances.

[Figure 2]

- Figure 1 Ming-Flash-Omni generally demonstrates highly competitive performance across various domains, including vision-text understanding, controllable image generation, speech recognition, and speech synthesis. Specifically, in image generation, MingFlash-Omni introduces a novel generative segmentation paradigm to achieve fine-grained spatial and semantic control over the generated images. Moreover, Ming-Flash-Omni significantly enhances Context-Aware Speech Recognition (ContextASR) and Chinese dialect recognition, thereby broadening its applicability in real-world scenarios.

In this paper, we introduce Ming-Flash-Omni, which builds upon the Ming-Omni architecture with a redesigned foundation and targeted enhancements across multimodal understanding and generation. At its core, Ming-Flash-Omni adopts Ling-Flash-2.0 lin (2025) (a scaled-up, highly sparse Mixture-of-Experts architecture) where an increased sparsity ratio enables substantial model capacity while maintaining bounded inference latency, striking a favorable trade-off between performance and efficiency.

On the understanding side, the model introduces three key advances. First, Ming-Flash-Omni upgrades the positional encoding to time-interleaved VideoRoPE Wei et al. (2025), which interleaves timestamps with video frames to align each visual token with its precise temporal position, better capturing video temporal dynamics. Second, Ming-Flash-Omni focus on improving the contextaware ASR capability itself, enhancing the model’s ability to leverage surrounding linguistic context during speech recognition and thereby achieving more accurate transcription in context-dependent scenarios. Third, the model supports multi-turn, cross-task understanding, enabling seamless switching between heterogeneous comprehension tasks (e.g., visual QA, audio captioning, document analysis) within a single dialogue session.

On the generation side, Ming-Flash-Omni introduces three key advancements: 1) beyond zeroshot voice cloning, it enables joint, single-channel generation of speech, sound, and music with fine-grained vocal control, replacing discrete acoustic tokens with continuous representations to circumvent quantization artifacts and produce more natural, expressive TTS outputs; 2) the model supports generative semantic segmentation, enabling pixel-level semantic content generation conditioned on multimodal inputs; and 3) it enables fine-grained controllable image generation with improved identity preservation, high-fidelity in-image text rendering, and robustness enhanced through vision-specific reinforcement learning.

These architectural innovations empower Ming-Flash-Omni to deliver exceptional cross-modal performance in both comprehension and generation tasks. Specifically, in image perception, MingFlash-Omni attains performance comparable to that of Gemini 2.5 Pro (Team, 2025); in image generation, it supports generative semantic segmentation and fine-grained controllable synthesis

with strong identity preservation and high-fidelity in-image text rendering; and in speech, it achieves state-of-the-art end-to-end understanding and generation capabilities.

The remainder of this paper is organized as follows. Section 2 presents the detailed architecture of Ming-Flash-Omni. Sections 3 describes the pretraining and post-training datasets. Section 4 reports the evaluation results and compare Ming-Flash-Omni with recent multimodal models. Sections 5 is conclusion.

##### 2 Ming-Flash-Omni

As illustrated in Figure 2, Ming-Flash-Omni retains the unified two-stage pipeline of MingOmni AI et al. (2025), where perception supports multimodal understanding and generation targets speech and image synthesis, while markedly advancing long-context modeling, reasoning, and controllable generation. At the core is Ling-Flash-2.0 lin (2025), a sparse MoE LM (100B; 6.1B per token) with a dual balancing scheme that stabilizes training and improves efficiency. On the perception side, Ming-Flash-Omni employs time-interleaved VideoRoPE, an enhanced variant of VideoRoPE Wei et al. (2025) that interleaves timestamps with video frames to achieve finegrained temporal alignment, explicitly associating each visual token with its precise temporal position; it also integrates context-aware ASR for more reliable speech understanding and a think mode for deeper multi-step reasoning. On the generation side, we replace discrete speech tokens with continuous acoustic latents, avoiding quantization loss and improving fidelity; for images, we upgrade to a synergistic training paradigm that enables generative segmentation-as-editing, facilitating fine-grained and controllable generation. Overall, Ming-Flash-Omni advances the unified model with stable expert routing and scalable long-context modeling, yielding more reliable multimodal understanding and controllable generation.

###### 2.1 Unified Understanding Across Modalities

A core challenge in unified multimodal models is the effective fusion of image understanding and generation. While our Ming-Flash-Omni injects hierarchical semantics via multi-scale query tokens, its language pathway remains frozen during training to prevent interference from the generative objective. Although this ensures stability, it creates a critical bottleneck: a misalignment between understanding and generation objectives. Consequently, even with injected semantics, finegrained visual knowledge (such as object attributes and spatial relationships) cannot be efficiently transferred to high-precision generation and editing, limiting final quality and controllability.

###### 2.2 Unified Speech Understanding and Generation

Ming-Flash-Omni unifies speech understanding and generation within a single architecture. For generation, we replace discrete speech tokens with continuous acoustic latents, avoiding quantization loss and improving fidelity. Specifically, we utilize a fixed, pre-trained audio head based on Qwen2.5 (0.5B parameters), which takes LLM-generated text tokens and downsampled VAE latents as input and autoregressively predicts the conditioning signals for the flow-matching head—following the paradigm of Jia et al. (2025).

Beyond zero-shot voice cloning, Ming-Flash-Omni enables joint, single-channel generation of speech, sound effects, and music, alongside fine-grained control over vocal attributes. This capability is underpinned by a novel variational autoencoder VAE-based audio tokenizer. To address two key challenges, cumulative error in autoregressive (AR) modeling caused by the inherently long

This image/video captures SpongeBob’s carefree spirit.

[Figure 3]

!

Perception Training

[Figure 4]

###### Vision Head

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Generation Training

[Figure 9]

DiT Blocks

Text Token Vision Token

Audio Token

Audio Detokenizer

Image Decoder

DiT Blocks

Pad Token

Word token

Semantic token Ref token

Noise token

Shared Experts

[Figure 10]

[Figure 11]

Routing Experts

Connector Refiner Byt5

Refiner

Audio Head

Vision Head

Ref VAE latent

||T-Router|
|---|
<br><br>… … …<br><br>|V-Router|
|---|
<br><br>|A-Router|
|---|
<br><br>1 2 3 … 256<br><br>1<br><br>|Bias|
|---|
<br><br>|Bias|
|---|
<br><br>|Bias|
|---|
|
|---|

###### MoE FFN

| | |
|---|---|
|RMSNorm| |

###### Lingflash

x N

|Linear Linear Linear<br><br>RMSNorm RMSNorm<br><br>| | |
|---|---|
|Attention Interface| |
<br><br>| | |
|---|---|
|Linear| |
<br><br>Q K<br><br>VideoRope VideoRope|
|---|

Attention Layer

| | |
|---|---|
|RMSNorm| |

|<0.0s> <0.5s><br><br>Interleaved video tokens|
|---|

[Figure 12]

Multi-Scale Learnable Tokens

Generate a vivid bird perched…

Please answer the ..

[Figure 13]

Audio Encoder

Vision Encoder

###### Ming-Omni

! !

[Figure 14]

[Figure 15]

- Figure 2 The overall framework of Ming-Flash-Omni. This version features a sparser LLM based on Ling-flash-2.0 MoE architecture, and integrates VideoRoPE to enhance temporal modeling. Speech generation now uses continuous features instead of discrete tokens, and image generation has been upgraded with support for segmentation.

sequence lengths of raw audio, and the mismatch in sampling rates between high-fidelity music (typically 44.1 kHz) and speech, we introduce a low-frame-rate tokenizer operating at 12.5 Hz. This design substantially compresses the token sequence length, thereby mitigating error propagation during AR generation. Moreover, we unify all training data to a consistent 44.1 kHz sampling rate and integrate a built-in super-resolution module, ensuring seamless compatibility with inputs originally sampled at lower rates.

Regarding controllability, in zero-shot scenarios, reference audio often contains extraneous attributes beyond timbre, such as emotional prosody or speaking style, that can interfere with instruction-guided generation. Conventional approaches typically address this by employing specialized components like gradient reversal layers to disentangle timbre from other acoustic factors; however, such methods introduce considerable architectural and training complexity. To address this, we designed a multistage training strategy: the first stage focuses on learning fundamental text-to-audio mappings, while the second stage concurrently injects voiceprint features extracted from reference audio alongside textual instructions. Given the robust pronunciation foundation established in the first stage, the model rapidly learns to map voiceprints to timbre and text instructions to style attributes, ultimately achieving disentanglement of timbre and control instructions without requiring any additional

modules. On the Instruct-TTS-Eval-zh benchmark (Huang et al., 2025b), Ming-Flash-Omni matches the performance of Qwen3-TTS (Hu et al., 2026). Moreover, the model incorporates a robust text normalization pipeline capable of accurately rendering complex symbolic expressions, including mathematical equations and chemical formulas.

###### 2.3 Unified Image Understanding and Generation

A core challenge in unified multimodal models is the effective fusion of image understanding and generation. While our Ming-Omni injects hierarchical semantics via multi-scale query tokens, its language pathway remains frozen during training to prevent interference from the generative objective. Although this ensures stability, it creates a critical bottleneck: a misalignment between understanding and generation objectives. Consequently, even with injected semantics, fine-grained visual knowledge (such as object attributes and spatial relationships) cannot be efficiently transferred to high-precision generation and editing, limiting final quality and controllability.

We address this through three synergistic pillars: 1) a single-stream diffusion transformer that unifies cross-modal representation; 2) a generative segmentation pretraining task that tightly couples perception and generation; and 3) a robust multi-reward reinforcement learning framework that refines controllability without overfitting.

Single-Stream Diffusion Transformers To overcome semantic fragmentation in dual-stream architectures (e.g., MMDiT), we adopt a native single-stream diffusion transformer (Cai et al., 2025). In dual-stream architectures, the noisy latent and reference latent are processed with independent self-attention, which exacerbates copy-and-paste artifacts from the reference image. In single-stream architectures, text, images, and other modalities are mapped into a unified token space, with full cross-modal attention in a single stream—breaking modality boundaries. Unlike the “multi-image collage” artifacts of prior methods, our approach achieves global semantic coherence in lighting, proportions, and layout, drastically reducing visible “copy-paste” effects.

Generative Segmentation as an Editing Task Building on this unified representation, we bridge the understanding–generation gap by reframing image segmentation as a generative editing task. Instead of producing abstract binary masks (e.g., “segment the banana”), the model performs semanticspreserving edit (e.g., “color the banana purple”). This reformulation tightly couples understanding and generation: accurate editing requires precise perception of object boundaries, making comprehension a prerequisite for generation. The edit quality thus provides direct supervision for visual understanding, unifying their optimization objectives. As a result, the model acquires fine-grained spatio-semantic control, which crucially addresses compositionality challenges in text-to-image generation.

Robust vision generation reinforcement learning To further refine alignment and controllability, we adopt a post-training RL framework that avoids SDE solvers and accelerates training by optimizing over a subset of diffusion steps (Zheng et al., 2025). To mitigate reward hacking, we initialize with a generative segmentation task and replace the KL divergence constraint with offline data regularization. We then perform sequential RL training on image generation and editing, using multi-dimensional rewards (realism, instruction adherence, aesthetic quality, and task-specific scores) to prevent overfitting to any single metric.

Empowering Advanced Controllable Capabilities The synergy of these components unlocks a suite of advanced, high-fidelity controllable functions:

Identity (ID) Preservation To preserve the identity of the reference subject, we concatenate the VAEencoded features of the reference image with the noisy latent representation of the diffusion model along the token dimension and enhance identity consistency through self-attention mechanisms. This approach is further complemented by learnable query tokens that encode global semantic information from the reference image. Together, these two complementary information streams ensure structural and appearance consistency in unedited regions while simultaneously improving the fidelity and accuracy of the edited content. For highly dynamic motion scenes (such as those commonly encountered in travel photography), we introduce motion-aware action label representations. By employing a balanced sampling strategy across these action labels during training, our model learns to effectively generate and edit motion-related regions. This strategy significantly mitigates the occurrence of unnatural poses and enhances the overall realism of the synthesized results.

High-Fidelity Text Rendering. By integrating a specialized Glyph-byT5 text encoder, our model leverages its learned pixel-level control to accurately place text, ensuring seamless contextual integration and high-quality results. Specifically, we feed the text to be rendered into a ByT5 encoder to obtain glyph features, which are then used together with learnable query tokens as conditioning inputs to the diffusion model. The advantage of this approach is faster training convergence, while the downside is that, during inference, we must first extract the text to be rendered from the user’s prompt in advance.

###### 2.4 Overall Training Procedure

The training procedure of Ming-Flash-Omni retains a two-stage pipeline: perception and generation. The perception stage of Ming-Flash-Omni comprises three sequential phases: progressive pre-training, two-stage instruction tuning, and reinforcement learning. Pre-training spans three stages: two at 8K context length to build robust multimodal foundations, followed by a 24K stage to capture long-range cross-modal dependencies. Instruction tuning then proceeds in two phases: Stage I uses 10K sequences over text, images, and audio for basic instruction following; Stage II extends the context to 64K, integrating video and multi-turn dialogue to enable temporally coherent multimodal interaction. During the reinforcement learning stage, we adopt Group Relative Policy Optimization (GRPO) Shao et al. (2024) to train the model. To accommodate different task types, we construct a domain-specific reward system that integrates rule-based rewards, model-based rewards, and checklist-based general rewards, enabling the unified reinforcement learning of heterogeneous tasks within a single training stage. To further improve the model’s continuous interaction capability across arbitrary modalities, we introduce multi-turn modality-switching conversation optimization over image, audio, video, and text.

After perception, we freeze the perception MLLM and optimize only the image generator, while leveraging the pre-trained audio generator from Canxiang et al. (2025). For image generation, the training procedure contains three sequential stages. In the first stage, we pre-train a diffusion-based image generator using a flow matching objective, while keeping the perception MLLM frozen. The generator is equipped with multi-scale learnable queries to capture hierarchical visual semantics from textual inputs. In the second stage, we extend the model to support image editing by conditioning the denoising process on reference images: the VAE-encoded representations of input images are concatenated with the noisy latent to enforce structural and semantic consistency with the original content. Additionally, input word-level captions are encoded via ByteT5 embeddings to enrich

textual conditioning. In the third stage, we conduct RL training sequentially on image generation and image editing, integrating multi-dimensional objectives, including image realism, instruction adherence, aesthetic quality, and task-specific rewards.

###### 2.5 Infrastructure

Compared to large language models (LLMs), the training of multimodal foundation models is characterized by distinctive challenges rooted in data heterogeneity and model heterogeneity. First, data heterogeneity arises from the need to dynamically switch between diverse input modalities (text, images, audio, and video) during training. These modalities exhibit significant differences in tensor shape, most notably in the form of dynamic batch sizes and variable-length sequences. This variability complicates the design of a unified parallel computation layout. As a result, computational workloads become unevenly distributed across processing ranks, leading to load imbalance. Moreover, the frequent allocation and deallocation of GPU memory buffers for inputs of varying shapes induce severe memory fragmentation, substantially degrading training efficiency and hardware utilization. Second, in contrast to large language models (LLMs), which are predominantly based on homogeneous, decoder-only Transformer architectures, multimodal foundation models typically employ modality-specific encoders at the input stage, introducing model heterogeneity. Despite their modest size, these encoders are adversely affected by parallelization strategies that introduce substantial pipeline bubbles (i.e., idle computation cycles). This inefficiency becomes a major bottleneck for training throughput.

To address these challenges, Ming-Flash-Omni is trained on an enhanced version of the MegatronLM Shoeybi et al. (2019) framework with three key extensions tailored for multimodal workloads:

- • Sequence Packing For Data Heterogeneity: To support dynamic input shapes, we adopt sequence packing, which packs multiple variable-length samples into fixed-length batches. In addition, to ensure data consistency across PP stages, we design a communication protocol that overlaps communication with computation. Together, these methods improve memory utilization and computational density.
- • Flexible Parallelization for Model Heterogeneity: To address workload imbalance caused by architectural asymmetry between encoder and decoder, we extend Megatron-LM to support flexible parallelization, enabling independent DP/PP/TP configurations for each component. Furthermore, we formulate the pipeline dependency as a linear programming problem to automatically derive optimal stage layouts for varying sequence lengths. These synergistic optimizations yield end-to-end training speedups of 58.1%–69.7% over uniform manual setups across multiple model scales.

Additionally, we observe that state-of-the-art memory-intensive operators (especially during backward passes) exhibit low memory bandwidth utilization, achieving only a fraction of the theoretical hardware peak. To address this inefficiency, we design custom CUDA kernels (e.g., for LayerNorm and RMSNorm) that raise bandwidth utilization to approximately 80% of the hardware limit. Our optimized kernels reduce forward and backward latency by up to 25.3% compared to Transformer Engine and deliver a 5× speedup over native PyTorch.

Collectively, these optimizations enable a Model FLOPs Utilization (MFU) of 28%, representing more than a 4x improvement in training throughput over the baseline Megatron-LM implementation.

#### 3 Data Construction

We have collected a large and diverse set of training data to enable models to process and understand information from multiple modalities, including text, images, audio and videos. The majority of this data comes from Ming-Omni (AI et al., 2025). In addition, we develop several data processing pipelines to ensure data quality, diversity and deduplication. Establishing an effective multimodal data strategy is essential for the joint multi-modal training, as it facilitates seamless alignment of knowledge across diverse modalities. We categorize the training data based on the core modalities they are designed to enhance, including image, audio, video, and text. The detailed sources and construction methods for each type of data are elaborated in this section.

###### 3.1 Image Data

Image data serves as the cornerstone of our multi-modal corpus. Following Ming-Omni (AI et al., 2025), we integrate both image-understanding and image-generation datasets to enable the MLLM to acquire unified perception and generation capabilities. Additionally, we further design novel pipelines to synthesize high-quality datasets across diverse dimensions to improve model’s capabilities and user interaction quality.

###### 3.1.1 Image Understanding Data

Knowledge Data: To enhance Ming-Flash-Omni’s expert-level comprehension and perception in knowledge-intensive tasks, we construct a large-scale, high-fidelity encyclopedia data ecosystem and directly integrate it into Ming-Flash-Omni’s training pipeline. This integration enables MingFlash-Omni to perform expert-level reasoning, such as accurately identifying rare or endangered species using Latin binomial nomenclature. The corpus encompasses a broad range of knowledge domains, with representative coverage across three thematic categories: biological (Animals, Plants), cultural (Celebrities, Anime Characters, Landmarks), and daily-life (Dishes, Antiques, Calligraphy and Paintings). Data are sourced from academic databases, institutional websites, digital museums, and specialized professional platforms.

We design two complementary data pipelines to generate synergistic forms of knowledge-intensive training data, both directly used to train Ming-Flash-Omni for multimodal alignment and reasoning:

- • Expert-Level Entity Recognition Data: Using a “breadth-to-depth” strategy, we harvest canonical entities (e.g., species binomials) and retrieve semantically relevant images via search engines. A progressive filter (combining CLIP consistency, MLLM-based verification, and manual refinement) ensures high precision, enabling Ming-Flash-Omni to recognize finegrained expert concepts.
- • Encyclopedia-Based Knowledge QA: We automatically align visual entities with structured knowledge by extracting 〈image, entity, knowledge〉 triplets, filtering them via a multi-VLM consensus system for cross-modal consistency, and converting validated triplets into 〈image, question, answer〉 VQA pairs using an LLM. This yields a broad-coverage dataset that significantly strengthens Ming-Flash-Omni’s knowledge-grounded reasoning.

STEM Reasoning Data: In the reasoning training of Ming-Flash-Omni, we enrich chain-ofthought (CoT) data to strengthen the model’s capacity for complex, multi-step STEM reasoning. The curated reasoning dataset is centered on two core themes: multidisciplinary reasoning and mathematical reasoning. To ensure high quality and structural coherence, we develop a systematic

pipeline for CoT generation and filtering, comprising four key stages: (1) QA extraction: We extract raw question–answer pairs from diverse sources; (2) Difficulty-aware filtering: We discard samples deemed too simple for Ming-Flash-Omni based on preliminary model evaluation; (3) CoT expansion: We leverage state-of-the-art multimodal reasoning models (Team et al., 2025; Bai et al., 2025a; Gemini et al., 2023) to generate detailed, step-by-step reasoning traces, forming an initial CoT pool; (4) Quality validation: We rigorously assess the logical consistency and factual correctness of the synthesized CoT responses and filter out low-quality or erroneous examples. This pipeline yields 1.6 million multimodal long CoT samples, with individual sequences reaching up to 8K tokens. Empirical results show that this data significantly enhances Ming-Flash-Omni’s performance on challenging STEM reasoning benchmarks.

OCR Data: Text recognition and document understanding capabilities are crucial for MLLM. We construct a large-scale heterogeneous training dataset with millions of samples, consisting of three data sources: open-source data, expert-collaborative pseudo-labeled data, and human-annotated enhancement data. The expert-collaborative pseudo-labeled data is generated by diagnosing model weaknesses, and using expert models to label targeted data. In addition, to enhance the model’s capability in text–visual analysis and logical reasoning, we incorporate the Chain-of-Thought (CoT) paradigm into the training data. We incorporate the open-source ChartQA-PoT dataset to enhance the model’s numerical reasoning ability on charts and pioneeringly use executable Python code as the intermediate reasoning representation.

Safety Data: We treat safety as an intrinsic capability of intelligent systems, aligning model behavior with ethical standards while preserving naturalness and helpfulness. This is achieved through a context-aware synthetic pipeline that expands unimodal safety seeds into a multimodal corpus at nearly 30× scale, structured along two axes: (1) illegal content and conduct (e.g., drugs, violence, pornography) and (2) socially sensitive topics governed by policy norms (e.g., politics, financial ethics). For query construction, we use multimodal large language models (MLLMs) to extract context-specific keywords from seed examples to guide image retrieval, then pair the retrieved images with synthesized prompts encoding harmful intents, adversarial metaphors, or high-risk inquiries. For response generation, we adopt a dual-strategy framework: queries involving strictly prohibited content elicit explicit refusals accompanied by corrective guidance, while safe but sensitive inquiries receive informative, boundary-respecting responses that maintain helpfulness without compromising safety.

GRPO Data: During the reinforcement learning (RL) stage, we train the model on multi-task data spanning a broad spectrum of general-purpose application scenarios, including knowledge-based question answering, mathematical and logical reasoning, multimodal instruction following, and OCR. The data is sourced from diverse tasks in the instruction tuning phase (e.g., conversational QA and multimodal understanding) and further augmented with user-experience-oriented examples (emphasizing fluency and practicality) as well as safety-critical cases (covering harmful content avoidance, privacy preservation, and compliance constraints). To ensure quality and diversity, we construct the RL dataset using task-type proportion control, difficulty-level stratification, and rigorous data cleaning and resampling strategies, yielding a curriculum specifically designed to strengthen the model’s general-purpose capabilities.

###### 3.1.2 Image Generation Data

Image generation data enhances MLLM capabilities beyond traditional image understanding tasks. Building on the image generation data used in Ming-Omni (AI et al., 2025), we further incorporate

segmentation, text rendering, and portrait preservation data to improve user experience. We extract image–text embeddings, apply k-means clustering, and use the resulting cluster labels to construct a semantically balanced training set via cluster-aware sampling.

Image segmentation data: To improve the model’s generative segmentation capability, we construct two types of data: (1) We use the open-source referring segmentation datasets RefCOCO/+/g (Kazemzadeh et al., 2014; Mao et al., 2016) to construct image editing data. The original image serves as the reference, and binary masks highlight target regions with specified colors to create edited images. (2) For semantic and panoptic segmentation, samples are built from COCOPanoptic (Lin et al., 2014; Kirillov et al., 2019), where each class or instance is assigned a unique color via a predefined colormap to generate edited images.

Portrait preservation data: The portrait preservation data consist of two data sources: (1) ID Photo Dataset: We collected and constructed 200k paired lifestyle-ID photos. And filter the data using four criteria, e.g., face similarity, face size and confidence, face angle and manual review. (2) Landmark Check-In Portrait Dataset: We collect 20K high-quality portraits from 225 landmarks.The original images serve as edited images, while using advanced segmentation model like SAM2 (Ravi et al., 2024) to segments the foreground person and places them onto 1,000 manually collected daily background scenes as pre-edited images. And then use LLM generate diverse prompts for each landmark.

Text generation data: We build a Chinese-English text generation dataset across three difficulty levels: (1) Monotonic background text rendering: Text is rendered directly by setting background color, font type, size, color, and position. (2) Text rendering on existing images: texts are rendered on suitable smooth regions obtained by Felzenszwalb algorithm. (3) Text-image integrated rendering: Using the SOTA LLMs to generate text rendering prompts, the advanced generation models (e.g., Qwen-image (Wu et al., 2025) and Nano Banana) are used for image generation, followed by OCR for consistency checks, resulting in a high-quality dataset.

###### 3.2 Audio Data

For audio data, we mainly use the data from Ming-Omni (AI et al., 2025). In addition, we incorporate the following three datasets to further enhance the model’s audio understanding and generation capabilities.

Context ASR Data: Current ASR systems face challenges in recognizing homophones or phonetically similar words when the context is limited, pronunciations are unclear, or accents are present. ContextASR addresses these issues by leveraging the preceding context. We propose to synthesize a large-scale dataset using LLMs to endow models with ContextASR capabilities. We extract named entities and construct context passages using LLMs based on existing ASR text, producing 3 million Chinese and English samples in the format <audio, text, context, entity_list>. During training, we further filter and sample the data to reduce keyword density, remove keywords that are absent from the text, and generate negative samples to enhance the model’s discriminative ability.

TTS Data: The diversity of TTS data is essential for fully leveraging the pretrained language model’s capabilities in audio generation. In addition to the open-source data used in Ming-Omni, we develop a data generation pipeline to create large-scale TTS data. Specifically, (1) we crawl extensive audio data using keywords expanded from handcrafted seeds through domain-specific lexical variations, (2) apply VAD (Gao et al., 2023) to segment well-conditioned short clips, and (3) iteratively train an audio labeler—initially on high-quality data, then using its predictions to

annotate the corpus with fine-grained labels for dimensions such as age, personality, style, speed, pitch, profession, emotion, and dialect. We curated a large-scale dataset of high-quality audio clips from diverse domains. These clips are annotated with a rich set of multi-dimensional labels, which in turn equips Ming-Flash-Omni with strong capabilities for controllable synthesis.

###### 3.3 Video Data

We enhance the base video corpus from Ming-Omni (AI et al., 2025) with two core data strategies to improve reasoning quality and interactive realism.

High-Value Video Selection: We discard simple heuristics (e.g., resolution) and instead use a VLM with Chain-of-Thought reasoning to score videos across presentation, content, and reasoning difficulty. Clips strong in any dimension are retained, preserving challenging edge cases while filtering out bland, high-resolution filler.

Realistic Multi-Turn Interaction Synthesis: We generate temporally grounded dialogues for long videos (up to 30 min, 256k tokens) by combining hierarchical event extraction with a “look-back” mechanism that enforces cross-turn consistency. To model real user behavior, each dialogue is conditioned on an LLM-simulated cognitive state (e.g., curiosity or confusion), yielding intent-rich, multi-turn interactions beyond standard benchmarks.

###### 3.4 Text Data

For text data, we utilize corpus from Ling (LingTeam et al., 2025), M2-Omni (Guo et al., 2025), and Ming-Omni (AI et al., 2025) to preserve and further enhance the model’s language proficiency.

#### 4 Evaluation

In this section, we present the evaluation details and quantitative examples of Ming-Flash-Omni on both public and in-house benchmarks.

###### 4.1 Public Benchmarks

The details of the public benchmarks are provided in Appendix A. As shown in Table 1∼11, our holistic assessment covers more than 50 rigorously curated public benchmarks across the following seven distinct multi-modal dimensions: Image → Text (Understanding), Text → Image (Generation), Image → Image (Editing), Image → Image (Segmentation), Audio → Text (Understanding), Text → Audio (Generation), and Video → Text (Understanding).

###### 4.2 In-house Benchmarks

In addition to public benchmarks, we also establish three in-house benchmarks to comprehensively evaluate multiple capabilities of MLLMs, including:

Wiki Knowledge. To thoroughly evaluate the factual knowledge capabilities of our models, we have developed an in-house benchmark called Wiki Knowledge, constructed from Wikipedia. This benchmark encompasses both broad general knowledge and specialized domain categories, including famous landmarks, notable figures, and distinctive dishes, enabling a comprehensive assessment of MLLMs’ understanding and reasoning over diverse factual information.

Video Streaming Multi-turn Benchmark. The evaluation of video streaming multi-turn dialogue capabilities requires quantifying not only the model’s understand capability but also assessing its

interactive experience, including proactivity and naturalness. Previous streaming dialogue datasets, such as StreamBench (Lin et al., 2024) and OvO-Bench (Niu et al., 2025), have primarily focused on the understanding aspect while lacking a thorough evaluation of the interactive experience. To address this gap, we introduce StreamingMultiturnBench. To construct StreamingMultiturnBench, we manually selected 380 videos, carefully ensuring coverage of multiple key domains including life recording, education, TV shows, video games, and documentaries. Then we use SOTA closed-source modelfor machine annotation. Subsequently, a team of 10 human annotators revise and doublecheck the dialogue content to ensure it aligns with human conversational preferences. This process yielded 2,200 video question-answer pairs. During evaluation, we use advanced closed-source model, e.g. GPT-4o (OpenAI, 2025), to compare the model’s output against the human-annotated answers, scoring it on a scale of 1 to 5 across the five dimensions: accuracy, completeness, relevance, naturalness, and proactivity. The final score is the average for each dimension. To align our metrics with other video benchmarks, we linearly scale the results to a 100-point scale. We commit to open-sourcing and publicly maintaining this benchmark to ensure reproducibility.

Multi-Dialect and Multi-Domain Audio Understanding Benchmark. To extend audio understanding benchmarks into multi-dialect and multi-domain settings, we constructed two specialized datasets. The multi-dialect dataset was created from 15 regions, while the multi-domain one was curated from six domains. All samples were manually verified for quality by trained annotators. The final datasets comprise 51,986 multi-dialect samples and 10,397 multi-domain samples, with the latter distributed across: Noisy (8,145), Chat (443), Government (462), Health (450), Knowledge (421), and Local Services (476).

- Table 1 Performance of Ming-Flash-Omni on Vision-to-Text Benchmarks compared to leading models. * denotes our own evaluation using the official benchmark prompts.

|Type Benchmark<br><br>|Ming-Flash Omni|Qwen3-Omni 30B-A3B<br><br>LongCat-Flash-Omni 560B-A27B<br><br>Gemini2.5 Pro<br><br>|GLM-4.6-V 106B-A12B|
|---|---|---|---|
|General<br><br>MMStar AI2D HallusionBench MMVet MMBench-ENtest<br><br>|74.9 89.3 66.1 85.6 87.0<br><br>|68.5 70.9 73.6 85.2 - 89.5 59.7 - 64.1 73.9* 69.0 83.3* 84.7* 87.5 90.8*<br><br>|75.9* 88.8* 63.2* 79.8* 88.8*<br><br>|
|Knowledge Wiki Knowledge(In-house)|64.6<br><br>|36.5 - 48.7|35.3|
|STEM/Reasoning<br><br>MMMUval MathVistamini MathVerseVision MathVision LogicVista|77.1 83.9 75.5 65.6 67.6<br><br>|69.1 70.7 80.9 75.9 77.9 77.7 49.6 - 73.1 56.3 - 66.0 48.3 - 68.7<br><br>|76.0 85.2 75.4 63.5 62.1<br><br>|
|OCR<br><br>ChartQA DocVQA OCRBench|87.0 94.7 891<br><br>|87.5 87.6 83.3<br><br>95.0 91.8 860 849 866<br><br>|91.1 95.6 865<br><br>|
|Grounding RefCOCO-avg<br><br>|88.5<br><br>|87.4 - 87.5<br><br>|85.9|
|Multi-image<br><br>MMTBench_val_mi MuirBench LLaVA-Interleave<br><br>|70.1 61.4 63.1|69.9* - 69.7*<br><br>61.9 - 61.6* - -<br><br>|70.6* 77.1 63.5*<br><br>|
|Video<br><br>MVBench VideoMME w/o sub LongVideoBench MLVU PerceptionTest Charades_STA TOMATO<br><br>|77.5 73.4 65.4 77.7 78.5 59.3 40.4<br><br>|71.2* 75.2 74.2* 70.5 76.2 84.3 59.4* 69.3 67.1*<br><br>75.2 - 81.2*<br><br>65.5* - 78.4 28.9* - -<br><br>28.2* - 46.9*<br><br>|74.9<br><br>74.8 66.5*<br>75.1* 80.6* 27.1* 39.4*<br><br><br>|

- Table 2 Performance of Ming-Flash-Omni on Text-to-Image Generation Benchmarks compared to leading models. “Gen.” denotes models for pure image generation, while “Uni.” denotes models capable of both image understanding and generation. Note that the global best performance is highlighted by an underline, and the local best result in “Gen.” or “Uni.” is marked with bold.

GenEval

Type Model

DPG-Bench 1-Obj. 2-Obj. Count Colors Posit. Color. AVG

SDv2.1 0.98 0.51 0.44 0.85 0.07 0.17 0.50 68.09 Emu3-Gen 0.98 0.71 0.34 0.81 0.17 0.21 0.54 80.60 SDXL 0.98 0.74 0.39 0.85 0.15 0.23 0.55 74.65 DALL-E 3 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD3-Medium 0.99 0.94 0.72 0.89 0.33 0.60 0.74 84.08

Gen.

SEED-X 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o 0.95 0.52 0.49 0.82 0.11 0.28 0.53 TokenFlow-XL 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Janus 0.97 0.68 0.30 0.84 0.46 0.42 0.61 79.68 JanusFlow 0.97 0.59 0.45 0.83 0.53 0.42 0.63 80.09 JanusPro-7B 0.99 0.89 0.59 0.90 0.79 0.66 0.80 84.19 UniWorld-V1 0.98 0.93 0.81 0.89 0.74 0.71 0.84 81.38 OmniGen2 0.99 0.96 0.74 0.98 0.71 0.75 0.86 83.57 BAGEL 0.99 0.94 0.81 0.88 0.64 0.63 0.82 Z-Image 1.00 0.94 0.78 0.93 0.62 0.77 0.84 88.14 Qwen-Image 0.99 0.92 0.89 0.88 0.76 0.77 0.87 88.32 Qwen-Image-RL 1.00 0.95 0.93 0.92 0.87 0.83 0.91 Ming-Flash-Omni 0.99 0.98 0.92 0.94 0.96 0.89 0.94 86.98

Uni.

###### 4.3 Quantitative Results

We conduct comprehensive evaluations of Ming-Flash-Omni against state-of-the-art MLLMs on over 50 different multimodal benchmarks, as illustrated in Table 1∼11. Extensive experiments demonstrate that Ming-Flash-Omni achieves comparable performance with leading MLLMs.

Vision → Text (Understanding) As shown in Table 1, Ming-Flash-Omni demonstrates strong and balanced performance across multimodal tasks, standing out among open omni-modal models. It achieves top scores on HallusionBench (66.1) and MMVet (85.6), reflecting robust visual reasoning and low hallucination. On MMStar, it scores 74.9, slightly below GLM-4.6-V (75.9) but ahead of Qwen3-Omni (68.5) and Gemini 2.5 Pro (73.6). Its dominant result on the in-house Wiki Knowledge benchmark (64.6 vs. ≤ 48.7 for others) highlights superior factual grounding. In STEM reasoning, Ming-Flash-Omni attains the best overall score on MathVerseVision (75.5), where success depends on interpreting visual mathematical content, and leads open models on MMMU (77.1). On OCR tasks, it sets a new high on OCRBench (891), though it is slightly behind GLM-4.6-V on ChartQA and DocVQA. In multi-image understanding, it outperforms Qwen3-Omni on MMT-Bench (70.1 vs. 69.9) and LLaVA-Interleave (63.1 vs. 61.6), but lags on MuirBench.

In video understanding, Ming-Flash-Omni achieves state-of-the-art results on MVBench (77.5), surpassing all other open models, and performs competitively on moment localization, dynamic reasoning, and long-form video tasks—trailing only closed-source or larger-scale systems like Gemini 2.5 Pro. Overall, it offers state-of-the-art open performance with particular strengths in visually grounded reasoning, factual alignment, and document understanding.

Text → Image (Generation). As shown in Table 2, our experimental results demonstrate that the generation quality of Ming-Flash-Omni is on par with state-of-the-art diffusion models. Notably, on the Geneval benchmark, our model surpasses all previous methods, demonstrating exceptional

- Table 3 Performance of Ming-Flash-Omni on Image-to-Image Editing Benchmarks compared to leading models. All metrics are evaluated by GPT-4.1. “Edit.” denotes models specifically trained for image editing, while “Generalist.” denotes models that serve as general-purpose models capable of image understanding, generation, and editing. The global best performance is marked with bold.

Type Model

GEdit-Bench-EN (Full set)↑ GEdit-Bench-CN (Full set)↑

G_SC G_PQ G_O G_SC G_PQ G_O

Edit.

Instruct-Pix2Pix 3.58 5.49 3.68 - - AnyEdit 3.18 5.82 3.21 - - MagicBrush 4.68 5.66 4.52 - - Step1X-Edit 7.09 6.76 6.70 7.20 6.87 6.86 Qwen-Image-Edit 8.00 7.86 7.56 7.82 7.79 7.52 Z-Image-Edit 8.11 7.72 7.57 8.03 7.80 7.54

Generalist.

UniWorld-v1 4.93 7.43 4.85 - - OmniGen 5.96 5.89 5.06 - - OmniGen2 7.16 6.77 6.41 - - BAGEL 7.36 6.83 6.52 7.34 6.85 6.50 Ming-Flash-Omni 8.11 7.87 7.64 8.02 7.95 7.62

- Table 4 Performance of Ming-Flash-Omni on Image-to-Mask Segmentation Benchmarks compared to leading models. Model types are denoted as: Vision. for vision-only models, SAM. for models equipped with an additional SAM-like segmentation head, and Uni. for unified MLLMs capable of both understanding and generation. Results with “*” are obtained by evaluating on 500 images sampled from each dataset via the official API.

Type Model RefCOCO (val)↑ RefCOCO+ (val)↑ RefCOCOg (val)↑

Vision.

VLT 67.5 56.3 55.0 CRIS 70.5 62.3 59.9 LAVT 72.7 62.1 61.2 PolyFormer-B 74.8 67.6 67.8

SAM.

LISA-7B 74.1 62.4 66.4 PixelLM-7B 73.0 66.3 69.3 OMG-LLAVA 75.6 65.6 70.7

Uni.

Nano-banana* 15.7 13.9 14.9 Qwen-Image-Edit* 30.3 28.8 34.0 Ming-Flash-Omni 72.1 65.2 65.4

- Table 5 Performance of Ming-Flash-Omni on PUBLIC Text-to-Speech Benchmarks compared to leading MLLMs.

|Type<br><br>Benchmark (Seed-TTS-Eval)|Ming-Flash Ming-Lite Qwen3 Seed F5 CosyVoice3 Qwen2.5<br><br>Omni Omni Omni TTS TTS Omni|
|---|---|
|Chinese<br><br>Zh-wer ↓ Zh-sim ↑<br><br>|0.87 1.69 1.07 1.11 1.56 1.16 1.70 0.72 0.68 - 0.80 0.74 0.78 0.75|
|English<br><br>En-wer ↓ En-sim ↑|2.19 4.31 1.39 2.24 1.83 2.02 2.72 0.61 0.51 - 0.76 0.65 0.718 0.63<br><br>|

Table 6 Performance of Ming-Flash-Omni on Voice Control for Sichuanese Dialect Generation benchmarks.

Model Performance WSC-TTS-Eval-easy WSC-TTS-Eval-hard

CER(%)↓ | SIM ↑ | ACC(%) ↑ CER(%)↓ | SIM ↑ | ACC(%) ↑

Cosyvoice3 3.17 | 0.696 | 68.06 4.07 | 0.723 | 80.90 Step-Audio-TTS 10.83 | 0.676 | — 12.52 | 0.545 | Qwen-TTS 4.13 | — | — 7.35 | — | CosyVoice2-WSC 4.28 | 0.727 | — 8.78 | 0.625 | CosyVoice2-WSC-SFT 4.08 | 0.788 | — 7.22 | 0.679 | Ming-Flash-Omni 2.25 | 0.695 | 82.08 3.18 | 0.717 | 84.42

Table 7 Performance of Ming-Flash-Omni on on ZipVoice-Dia(zh) test sets. Arrows indicate the desired direction (↓ = lower is better, ↑ = higher is better). Best values per column are in bold.

ZipVoice-Dia(zh) CER-zh ↓ cpSIM ↑ UTMOS ↑

Model

ZipVoice-Dia 3.39 0.553 2.24 MoonCast (Ju et al., 2025) 27.43 0.441 1.76 MOSS-TTSD (Zhao et al., 2025) 8.62 0.421 1.70 VibeVoice-1.5B (Peng et al., 2025) 12.87 0.455 1.74 FireRedTTS2 (Xie et al., 2025b) 3.34 0.512 1.90 SoulX-Podcast (Xie et al., 2025a) 2.2 0.599 2.09 Ming-Flash-Omni 2.12 0.457 2.25

Table 8 Performance of Ming-Flash-Omni on Context ASR benchmarks.

Model Performance Avg Speech-English Dialogue-English Speech-Mandarin Dialogue-Mandarin

WER | NE-WER | NE-FNR WER | NE-WER | NE-FNR WER | NE-WER | NE-FNR WER | NE-WER | NE-FNR

Gemini2.5-Pro 6.3 6.65 | 12.82 | 8.86 3.59 | 8.18 | 3.53 5.37 | 8.35 | 4.09 4.03 | 8.96 | 1.18 Kimi-Audio 9.24 2.90 | 6.68 | 8.01 4.67 | 13.50 | 11.31 1.95 | 11.13 | 15.28 2.90 | 15.91 | 16.68 Baichuan-Omni-1.5 8.17 8.16 | 7.69 | 6.53 9.91 | 14.40 | 5.54 2.98 | 8.39 | 4.71 5.00 | 16.83 | 7.84 Qwen3-ASR 4.23 2.95 | 5.00 | 3.93 2.91 | 8.19 | 4.38 0.92 | 5.92 | 1.17 1.36 | 10.18 | 3.83 Qwen3-Omni-30B-A3B-Instruct 4.34 1.13 | 1.66 | 0.28 4.41 | 21.26 | 1.15 0.80 | 5.41 | 0.36 1.59 | 13.35 | 0.69 Ming-Flash-Omni 3.30 2.91 | 2.53 | 2.13 3.50 | 7.60 | 2.08 1.13 | 5.41 | 1.16 1.77 | 8.30 | 1.04

controllability. This advantage is particularly pronounced in the "Position" and "Color." subcategories. On the DPG-Bench benchmark, Ming-Flash-Omni achieves an overall score of 86.98, closely approaching the state-of-the-art performance of 88.32 achieved by Qwen-Image, while substantially outperforming pure diffusion models such as SD3-Medium (84.08).

Image → Image (Editing). As shown in Table 3, Ming-Flash-Omni demonstrates impressive image editing performance, surpassing all other models. Specifically, Ming-Flash-Omni supports editing instructions in Chinese, achieving performance comparable to that with English instructions. Compared to Z-Image-Edit, which is specifically trained for editing tasks, Ming-Flash-Omni achieves comparable semantic consistency while delivering superior perceptual quality and higher overall scores under joint training for image generation and editing.

Image → Image (Segmentation). As shown in Table 4, Ming-Flash-Omni is capable of performing segmentation tasks, achieving performance comparable to that of specialized models designed explicitly for this purpose. Compared to other unified MLLMs, Ming-Flash-Omni demonstrates a significant advantage in segmentation. For instance, Qwen-Image-Edit often struggles to accurately localize the target object, while Nano-banana frequently misinterprets user intent during inference. In contrast, Ming-Flash-Omni exhibits superior robustness and a more accurate understanding of spatial and semantic instructions.

Audio → Text (Understanding). As shown in Table 8, Ming-Flash-Omni achieves the best overall average (Avg) on ContextASR-Bench, indicating robust context utilization. On several individual subsets/metrics, Qwen3-ASR(Shi et al., 2026) and Qwen3-Omni-30B-A3B-Instruct(Xu et al., 2025b) perform better, suggesting complementary strengths. It also exhibits highly competitive performance across various ASR benchmarks, with notable strengths in dialect recognition (Table 9). In audio question answering, Ming-Flash-Omni surpasses all open-source audio-centric and other Omni models, with the exception of Qwen3-Omni-Flash-Instruct(Xu et al., 2025b) (Table 10). Taken

###### Table 9 Performance of Ming-Flash-Omni on PUBLIC and IN-HOUSE Audio Understanding Benchmarks.

|Type Benchmark|Ming-Flash Qwen3 Qwen2 Kimi<br><br>Omni Omni Audio Audio<br><br>|
|---|---|
|Aishell1 ↓ Aishell2-test-android ↓ PUBLIC Aishell2-test-ios ↓ Chinese Cv15-zh ↓<br><br>Benchmarks Fleurs-zh ↓<br><br>Wenetspeech-testmeeting ↓ Wenetspeech-testnet ↓ SpeechIO ↓ Average (Chinese) ↓<br><br>|1.11 1.04 1.53 0.60<br><br>2.41 2.64 2.92 2.64 2.44 2.55 2.92 2.56 5.03 4.31 6.90 7.21 2.82 2.20 7.50 2.69 5.83 5.89 7.16 6.28 5.02 4.69 8.42 5.37<br><br><br>2.49 2.18 3.01 2.23<br><br>3.39 3.19 5.05 3.70<br><br><br>|
|Librispeech-test-clean ↓ Librispeech-test-other ↓<br><br>PUBLIC Multilingual-librispeech ↓<br><br>English Cv15-en ↓ Benchmarks Fleurs-en ↓<br><br>Voxpopuli-v1.0-en ↓ Average (English) ↓<br><br>|1.17 1.22 1.60 1.28<br><br>2.20 2.48 3.60 2.42<br><br>3.75 3.67 5.40 5.88 5.99 6.05 8.60 10.31<br><br><br>2.99 2.72 6.90 4.44 5.76 6.02 6.84 7.97<br><br>3.64 3.69 5.49 5.38<br><br><br>|
|Hunan ↓ Minnan ↓ Guangyue ↓ Chuanyu ↓ Shanghai ↓ Anhui ↓<br><br>IN-HOUSE Dongbei ↓<br><br>Dialect Henan ↓ Benchmarks Hubei ↓<br><br>Jiangsu ↓ Kejiahua ↓ Shaanxi ↓ Shandong ↓ Tianjin ↓ Yunnan ↓ Noisy ↓ IN-HOUSE Chat ↓<br><br>Domain Government ↓ Benchmarks Health ↓<br><br>Knowledge ↓ Local-live ↓<br><br>Average (All IN-HOUSE) ↓<br><br>|7.63 20.82 25.88 31.93 13.29 24.60 123.78 80.28<br><br>4.21 27.43 7.59 41.49 3.80 5.54 7.77 6.69<br><br>10.19 30.96 31.73 60.64<br><br>5.49 5.70 5.72 8.61<br><br>5.14 3.92 4.87 4.40 7.47 8.94 12.31 14.40<br><br>6.14 14.55 16.39 20.07 10.46 13.19 12.80 17.25 16.87 27.88 22.33 29.78<br><br><br>6.48 5.31 6.32 6.09 10.32 17.04 15.00 18.66 13.33 19.70 21.78 34.57<br><br>10.44 19.78 21.57 32.79<br><br>11.15 9.25 12.46 24.40<br><br><br>2.96 2.09 4.29 2.96<br><br>1.57 1.55 2.70 2.03<br><br>3.15 2.22 4.18 2.38 3.17 11.52 3.33 1.98<br><br>2.07 1.37 2.34 2.05<br><br><br><br><br>7.40 13.02 17.39 21.12<br><br><br>|

###### Table 10 Performance of Ming-Flash-Omni on PUBLIC Audio Question-Answering Benchmarks.

Open-ended QA Knowledge Multi-Choice QA Instruction Safety

Models Mean

AlpacaEval CommonEval SD-QA MMSU OpenBookQA IFEval AdvBench Step-Audio-chat 57.10 79.80 59.80 46.84 31.87 29.19 65.77 86.73

Qwen2-Audio-chat 54.70 73.80 68.00 35.35 35.43 49.01 22.57 98.85 Baichuan-Audio 62.50 80.00 67.80 49.64 48.80 63.30 41.32 86.73 GLM-4-Voice 57.20 81.20 69.60 43.31 40.11 52.97 24.91 88.08

Kimi-Audio 76.90 89.20 79.40 63.12 62.17 83.52 61.10 100.00 Megrez-3B-Omni 46.20 70.00 59.00 25.95 27.03 28.35 25.71 87.69

DiVA 55.70 73.40 70.80 57.05 25.76 25.49 39.15 98.27 Qwen2.5-Omni 74.10 89.80 78.60 55.71 61.32 81.10 52.87 99.42

Qwen3-Omni-Flash-Instruct 85.40 95.40 91.00 76.80 68.40 91.40 75.20 99.40 Baichuan-Omni-1.5 71.10 90.00 81.00 43.40 57.25 74.51 54.54 97.31

MiniCPM-o 71.70 88.40 83.00 50.72 54.78 78.02 49.25 97.69 Ming-Flash-Omni 82.95 95.58 87.10 71.30 72.09 85.50 69.87 99.23

- Table 11 Performance of Ming-Flash-Omni on StreamingMultiturnBench compared to leading omni-MLLMs.

|Model<br><br>|Accuracy Completeness Relevance Naturalness Proactivity Average|
|---|---|
|Ming-Lite-Omni Qwen2.5-Omni Qwen3-Omni Gemini-2.5-Pro<br><br>|44.63 40.58 69.28 95.65 28.78 55.78 54.03 53.68 78.00 98.28 46.25 66.05 60.91 58.03 83.65 99.31 42.51 68.88 68.03 69.07 86.85 98.99 52.02 74.99<br><br>|
|Ming-Flash-Omni|71.23 65.07 90.07 99.36 53.85 75.92<br><br>|

together, these findings demonstrate the robust and versatile audio understanding capabilities of Ming-Flash-Omni.

Text → Audio (Generation). As shown in Table 5, leveraging advancements in speech representation and model architecture, Ming-Flash-Omni achieves SOTA performance among open-source models of a comparable size on the test-zh subset of the SEED-TTS-Eval benchmark(Anastassiou

- et al., 2024b). For controllable tasks, Ming-Flash-Omni achieves an accuracy (ACC) of 83.25% on the Sichuanese dialect, outperforming CosyVoice3 (as shown in Table 6). On the podcast task, Ming-Flash-Omni sets the new state-of-the-art (SOTA) among open-source models for CER and UTMOS on the ZipVoice-Dia(zh) benchmark (see Table 7).

Video + Audio → Text (Video Streaming Conversation). As shown in Table 11, benefiting from the introduction of high-quality and diverse streaming video multiturn data, Ming-Flash-Omni has achieved significant improvements in all dimensions compared to Ming-Lite-Omni. MingFlash-Omni also outperforms Qwen3-Omni (Xu et al., 2025b) and Gemini 2.5-Pro (Team, 2025) in the dimensions of accuracy, completeness, and relevance, providing better experience in streaming video conversation scenarios.

###### 4.4 Visualization Results

In this section, we present several qualitative examples to illustrate the multifaceted capabilities of Ming-Flash-Omni. As shown in the figure, Ming-Flash-Omni exhibits strong visual understanding across multiple dimensions. In world knowledge grounding, it accurately identifies landmarks, animal species, and famous artworks from images—demonstrating its ability to integrate visual perception with rich encyclopedic knowledge. In mathematical reasoning, it solves complex problems ranging from Chinese college entrance exam (Gaokao) questions to International Mathematical Olympiad (IMO)-level challenges, providing clear, step-by-step derivations that reflect deep logical understanding. For multi-image understanding, it generates creative and coherent narratives grounded in multiple input images, showcasing its capacity for cross-image reasoning and contextual synthesis. Turning to speech recognition, Ming-Flash-Omni achieves strong performance on contextual ASR tasks. By leveraging contextual information, it effectively resolves many challenging cases where conventional ASR systems tend to fail—such as ambiguous homophones, domainspecific terminology, or noisy conversational speech. Moreover, this version also supports multiple Chinese dialects, significantly broadening its applicability in real-world multilingual and regional speech scenarios. Notably, Ming-Flash-Omni also supports multi-turn multimodal dialogues with seamless switching between modalities and tasks, enabling fluid interactions that combine vision, speech, and text in a unified conversational framework. Lastly, we visualize the capabilities of Text/Image →Image generation tasks in Figure 4 and Figure 5, covering a wide range of applications including image generation, image editing, image segmentation, multi-image editing, ID photo generation, ID photo editing, and background replacement. As can be seen, Ming-Flash-Omni not only supports a broader set of generative capabilities but also achieves higher output quality and

[Figure 16]

###### Figure 3 Qualitative results of Ming-Flash-Omni across diverse understanding tasks, including world knowledge grounding, multi-image reasoning, mathematical problem solving, contextual and dialect-aware ASR, and multi-turn multimodal dialogue.

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

###### Figure 4 Visualization results of Ming-Flash-Omni on Text/Image → Image tasks, including image generation task, image editing task, and image segmentation task.

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

###### Figure 5 Visualization results of Ming-Flash-Omni on Image → Image tasks, including ID photo generation, ID photo editing, background replacement, and multi-image editing.

greater controllability compared to previous versions.

#### 5 Conclusion

In this paper, we present Ming-Flash-Omni, built upon Ling-Flash-2.0 with 100 billion parameters, where only 6.1B parameters are activated per token. Ming-Flash-Omni demonstrates advanced multimodal perception and generation capabilities with improved computational efficiency while scaling model capacity. It achieves SOTA performance across a broad spectrum of tasks, including multi-image and video processing, image generation, generative segmentation, Contextual Automatic Speech Recognition (ContextASR), and multi-dialect recognition, outperforming omni models of comparable scale. We believe the open-sourcing of our models and code will facilitate the development of AGI by advancing multimodal intelligence research and enabling broader real-world applications.

- 6 Contributors Authors are listed alphabetically by the first name.

Ant Inclusion AI Baihui Li Bowen Ma Cheng Zou ChengKun Du Canxiang Yan Chunxiang Jin Chunjie Shen Chenyu Lian Chengxiang Fan Dandan Zheng Fudong Wang Furong Xu Guangming Yao Haohao Liu Han Peng Jun Zhou Junluan Xia Jingdong Chen Jianing Li Jianxin Sun Jianjiang Zhu Jianping Jiang Jinpeng Ou Jun Peng Jin Peng

Kaixiang Ji Li Tang Libin Wang Lixiang Ru Longhua Tan Lu Ma Lan Wang Mochen Bai Minghong Cai Mingxue Yang Ning Gao Qingpei Guo Qinglong Zhang Qiang Xu Qin Zhao Rui Liu Ruijie Xiong Ruobing Zheng Sirui Gao Shaoxiong Lin Tao Zhang Tianqi Li Tinghao Liu Tongli Wang Taoye Huang Weilong Chai

Xiaomei Wang Xiaolong Wang Xiaojian Liu Xiao Lu Xiaoyu Li Xingning Dong Xuzheng Yu Xuezhi Wang Yi Yuan Yuting Gao Yuting Xiao Yunxiao Sun Yipeng Chen Yifan Mao Yifei Wu Yingying Zhang Yongjie Lyu YuQian Li Ziping Ma Zhiqiang Fang Zhihao Qiu Ziyuan Huang Zizheng Yang Zhengyu He

#### References

Ling-flash-2.0, 2025. https://huggingface.co/inclusionAI/Ling-flash-2.0. Accessed: 2025-09-17. Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng,

Fudong Wang, et al. Ming-omni: A unified multimodal model for perception and generation. arXiv preprint arXiv:2506.09344, 2025.

Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, Mingqing Gong, Peisong Huang, Qingqing Huang, Zhiying Huang, Yuanyuan Huo, Dongya Jia, Chumin Li, Feiya Li, Hui Li, Jiaxin Li, Xiaoyang Li, Xingxing Li, Lin Liu, Shouda Liu, Sichao Liu, Xudong Liu, Yuchen Liu, Zhengxi Liu, Lu Lu, Junjie Pan, Xin Wang, Yuping Wang, Yuxuan Wang, Zhen Wei, Jian Wu, Chao Yao, Yifeng Yang, Yuanhao Yi, Junteng Zhang, Qidi Zhang, Shuo Zhang, Wenjie Zhang, Yang Zhang, Zilin Zhao, Dejian Zhong, and Xiaobin Zhuang. Seed-tts: A family of high-quality versatile speech generation models. CoRR, abs/2406.02430, 2024a.

Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024b.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. AISHELL-1: an open-source mandarin speech corpus and a speech recognition baseline. In 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment, O-COCOSDA, pages 1–5. IEEE, 2017.

Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Canxiang, Chunxiang Yan, Dawei Jin, Haibing Huang, Han Yu, Hui Peng, Jie Zhan, Jing Gao, Jingdong Peng, Jun Chen, Kaimeng Zhou, Ming Ren, Mingxue Yang, Qiang Yang, Qin Xu, Ruijie Zhao, Shaoxiong Xiong, Xuezhi Lin, Yi Wang, Yifei Yuan, Yongjie Wu, Zhengyu Lyu, He, Zhihao , Zhiqiang Qiu, Ziyuan Fang, and Huang. Ming-uniaudio: Speech llm for joint understanding, generation and editing with unified representation. arXiv preprint arXiv:https://arxiv.org/submit/6926109/view, 2025.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024a.

Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T. Tan, and Haizhou Li. Voicebench: Benchmarking llm-based voice assistants, 2024b. https://arxiv.org/abs/2410.17196.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024c.

Yuhang Dai, Ziyu Zhang, Shuai Wang, Longhao Li, Zhao Guo, Tianlun Zuo, Shuiyuan Wang, Hongfei Xue, Chengyou Wang, Qing Wang, et al. Wenetspeech-chuan: A large-scale sichuanese corpus with rich annotation for dialectal speech processing. arXiv preprint arXiv:2509.18004, 2025.

Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al. Kimi-audio technical report. arXiv preprint arXiv:2504.18425, 2025.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024.

Zhifu Gao, Zerui Li, Jiaming Wang, Haoneng Luo, Xian Shi, Mengzhe Chen, Yabin Li, Lingyun Zuo, Zhihao Du, Zhangyu Xiao, and Shiliang Zhang. Funasr: A fundamental end-to-end speech recognition toolkit, 2023. https://arxiv.org/abs/2305.11013.

Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, June 2024.

Qingpei Guo, Kaiyou Song, Zipeng Feng, Ziping Ma, Qinglong Zhang, Sirui Gao, Xuzheng Yu, Yunxiao Sun, Tai-Wei Chang, Jingdong Chen, et al. M2-omni: Advancing omni-mllm for comprehensive modality support with competitive performance. arXiv preprint arXiv:2502.18778, 2025.

Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, et al. Qwen3-tts technical report. arXiv preprint arXiv:2601.15621, 2026.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, et al. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946, 2025a.

Kexin Huang, Qian Tu, Liwei Fan, Chenchen Yang, Dong Zhang, Shimin Li, Zhaoye Fei, Qinyuan Cheng, and Xipeng Qiu. Instructttseval: Benchmarking complex natural-language instruction following in text-to-speech systems. arXiv preprint arXiv:2506.16381, 2025b.

Ziyuan Huang, DanDan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, Taozhi Huang, Jiajia Liu, Qingpei Guo, Ming Yang, Jingdong Chen, and Jun Zhou. Ming-univision: Joint image understanding and generation with a unified continuous tokenizer. arXiv preprint arXiv:2510.06590, 2025c.

Dongya Jia, Zhuo Chen, Jiawei Chen, Chenpeng Du, Jian Wu, Jian Cong, Xiaobin Zhuang, Chumin Li, Zhen Wei, Yuping Wang, et al. Ditar: Diffusion transformer autoregressive modeling for speech generation. arXiv preprint arXiv:2502.03930, 2025.

Zeqian Ju, Dongchao Yang, Jianwei Yu, Kai Shen, Yichong Leng, Zhengtao Wang, Xu Tan, Xinyu Zhou, Tao Qin, and Xiangyang Li. Mooncast: High-quality zero-shot podcast generation. arXiv preprint arXiv:2503.14345, 2025.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1086. https://aclanthology.org/D14-1086/.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images, 2016.

KimiTeam, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.

Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollár. Panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9404–9413, 2019.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024a.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024b.

Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

LingTeam, Binwei Zeng, Chao Huang, Chao Zhang, Changxin Tian, Cong Chen, Dingnan Jin, Feng Yu, Feng Zhu, Feng Yuan, et al. Every flop counts: Scaling a 300b mixture-of-experts ling llm without premium gpus. arXiv preprint arXiv:2503.05139, 2025.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024a.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), December 2024b. ISSN 1869-1919. doi: 10.1007/s11432-024-4235-6. http://dx.doi.org/10.1007/s11432-024-4235-6.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 11–20,

2016. doi: 10.1109/CVPR.2016.9.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.177. https://aclanthology.org/2022.findings-acl.177/.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 2199–2208, 2021. doi: 10.1109/WACV48630.2021.00225.

Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18902–18913, 2025.

OpenAI. Introducing 4o image generation. https://openai.com/index/introducing-4o-image-generation/, 2025. Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen,

Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, et al. Vibevoice technical report. arXiv preprint arXiv:2508.19205, 2025.

Viorica P˘atr˘aucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In Advances in Neural Information Processing Systems, 2023. https://openreview.net/forum?id=HYEGXFnPoq.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models, 2025. https://arxiv.org/abs/2410.23266.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. Qwen3-asr technical report, 2026. https://arxiv.org/abs/2601.21337.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Young Chol Song. Temporal grounding of activities using multimodal large language models. arXiv preprint arXiv:2407.06157, 2024.

Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. https://arxiv.org/abs/2507.01006.

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.

Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021.

Fei Wang, Xingyu Fu, James Y.Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, Tianyi Yan Lorena, Jacky Wwenjie Mo, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024a.

He Wang, Linhan Ma, Dake Guo, Xiong Wang, Lei Xie, Jin Xu, and Junyang Lin. Contextasr-bench: A massive contextual speech recognition benchmark, 2025. https://arxiv.org/abs/2507.05727.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal

mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024b. Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang,

et al. Videorope: What makes for good video rotary position embedding? arXiv preprint arXiv:2502.05173, 2025. Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Hanke Xie, Haopeng Lin, Wenxiao Cao, Dake Guo, Wenjie Tian, Jun Wu, Hanlin Wen, Ruixuan Shang, Hongmei Liu, Zhiqi Jiang, et al. Soulx-podcast: Towards realistic long-form podcasts with dialectal and paralinguistic diversity. arXiv preprint arXiv:2510.23541, 2025a.

Kun Xie, Feiyu Shen, Junjie Li, Fenglong Xie, Xu Tang, and Yao Hu. Fireredtts-2: Towards long conversational speech generation for podcast and chatbot. arXiv preprint arXiv:2509.02020, 2025b.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025a.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report, 2025b. https://arxiv.org/abs/2509.17765.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025c.

Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006, 2024.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

Albert Zeyer, André Merboldt, Wilfried Michel, Ralf Schlüter, and Hermann Ney. Librispeech transducer model with internal language model prior correction. In Hynek Hermansky, Honza Cernocký, Lukás Burget, Lori Lamel, Odette Scharenborg, and Petr Motlícek, editors, Annual Conference of the International Speech Communication Association, Interspeech, pages 2052–2056. ISCA, 2021.

Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, Di Wu, and Zhendong Peng. WENETSPEECH: A 10000+ hours multi-domain mandarin corpus for speech recognition. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, pages 6182–6186. IEEE, 2022.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

Xingjian Zhao, Zhe Xu, Qinyuan Cheng, Zhaoye Fei, Luozhijie Jin, Yang Wang, Hanfu Chen, Yaozhou Jiang, Qinghui Gao, Ke Chen, et al. Moss-speech: Towards true speech-to-speech models without text guidance. arXiv preprint arXiv:2510.00499, 2025.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.

Han Zhu, Wei Kang, Liyong Guo, Zengwei Yao, Fangjun Kuang, Weiji Zhuang, Zhaoqing Li, Zhifeng Han, Dong Zhang, Xin Zhang, et al. Zipvoice-dialog: Non-autoregressive spoken dialogue generation with flow matching. arXiv preprint arXiv:2507.09318, 2025.

# Appendix

- A Public Benchmarks

Image → Text (Understanding). Our evaluation of the image-to-text understanding capabilities primarily encompasses the following six tasks: 1) general image understanding capabilities evaluated on MMStar (Chen et al., 2024a), AI2D (Kembhavi et al., 2016), HallusionBench (Guan et al., 2024), MMBench (Liu et al., 2024a), MMVet (Yu et al., 2023). 2) STEM and reasoning capabilities are evaluated on MMMUval (Yue et al., 2024), MathVistamini (Lu et al., 2024), MathVerseVision (Zhang et al., 2024), MathVision(Wang et al., 2024b), LogicVista (Xiao et al., 2024). 3) OCR capabilities evaluated on ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021) and OCRBench (Liu et al., 2024b). 4) multi-image capabilities evaluated on MMTBench (Ying et al., 2024), MuirBench (Wang

- et al., 2024a), and LLaVA-interleave Bench (Li et al., 2024a).

Video → Text (Understanding). Our evaluation of the video-to-text understanding capabilities contains the following four benchmarks: MVBench (Li et al., 2024b), VideoMME (Fu et al., 2024), LongVideoBench (Wu et al., 2024), MLVU (Zhou et al., 2024), PerceptionTest (P˘atr˘aucean et al.,

- 2023), CharadesSTA (Song, 2024) and TOMATO (Shangguan et al., 2025).

Text → Image (Generation). We incorporate text-to-image generation capabilities to enable our MLLM with unified perception-generation abilities, which are evaluated on GenEval (Ghosh et al.,

- 2024) and DPG-Bench (Hu et al., 2024).

Image → Image (Editing). Our evaluation of image-to-image editing capabilities is conducted on the GEdit-Bench benchmark (Liu et al., 2025).

Image → Image (Segmentation). We evaluate the segmentation capability of our MLLM on the standard referring expression segmentation (RES) benchmarks RefCOCO/+ (Kazemzadeh et al., 2014) and RefCOCOg (Mao et al., 2016).

Audio → Text (Understanding). Our evaluation of the audio-to-text understanding capabilities mainly includes the following three tasks: 1) Fundamental audio understanding capabilities evaluated on a broad range of public benchmarks, including public Chinese benchmarks like Aishell1 (Bu et al., 2017) and Wenetspeech (Zhang et al., 2022), and public English benchmarks like Librispeech (Zeyer et al., 2021) and Voxpopuli (Wang et al., 2021). And 2) audio question-answering capabilities evaluated on various benchmarks across five specific tasks, such as AlpacaEval and CommonEval from VoiceBench (Chen et al., 2024b) for open-ended QA tasks, and SD-QA for knowledge-based QA tasks. Finally 3) evaluates the model’s ability to utilize context on ContextASR-Bench(Wang

- et al., 2025).

Text → Audio (Generation). We incorporate text-to-audio generation capabilities to enable our MLLM with unified audio perception-generation abilities, which are evaluated on Seed-TTSEval (Anastassiou et al., 2024a). For the controllable tasks, we use Wenetspeech-chuan(WSC) (Dai et al., 2025) to evaluate the performance of dialect generation. For the podcast tasks, we utilize ZipVoice (Zhu et al., 2025) for evaluation.

- B The prompt used in data generation This section presents the prompt used for data generation in Sec. 3.

Prompt for portrait preservation data

|Based on the person ’ s age and gender in the image , generate a detailed description of a r e a l i s t i c l i f e scene photo . Requirements :<br><br>1. Must include : clothing , location , action , styl e .<br><br>2. S t r i c t l y within 50 words .<br><br>3. Change the scene ( real scenes : indoor , outdoor , r e s i d e n t i a l areas , homes , companies , kitchens , parks , etc . ) and character costumes to ensure that even similar images have d i f f e r e n t descriptions .<br><br>4. Clothing must match age and gender , vary descriptions .<br><br>5. Scenario d e t a i l s must include at l e a s t 3 r e a l i s t i c elements , be vivid and l i f e l i k e . Output only the description in English , no additional text . Example : A young woman with medium- length dark hair tied back in a neat ponytail stands outdoors near a park bench , dressed in a crisp white button up s h i r t under a f i t t e d black blazer . She wears a subtle headband , her brow knitted as she runs her f i n g e r s through her hair , her l i p s s l i g h t l y downturned and eyes g l i s t e n i n g with unshed tears . The scene captures her from<br><br><br>a distant angle , with trees and a blurred pathway in the soft - focus background , suggesting contemplation or d i s t r e s s amidst nature .<br><br>|
|---|

Prompt for text generation data

|Describe according to the following requirements :<br><br>1. Randomly s e l e c t a theme from {theme} , and generate text content in Chinese , English , and numbers , with 3 -5 characters .<br><br>2. Use your imagination ; the theme i s not fixed . Keep the description under 100 characters .<br><br>3. Generate a suitable image description based on the text content .<br><br>4. Refer to the { text style } and choose appropriate font , color , and layout .<br><br><br>Output format : Description , Text : " content " , Font style , Font color , and Image layout<br><br>Examples : In the image , there i s a gym scene , Text : "Power 3.0 " , font sty le i s broken art font with cracks and shattering e f f e c t s , font color i s metallic gray with<br><br>dark red gradient , text layout i s centered - right , background includes dumbbells , treadmills , and r e f l e c t i v e glass walls .<br><br>|
|---|

###### Prompt for video multi-turn conversation data

|# Role: Expert Video Analyst You are an expert video analyst . Your task i s to analyze the provided video<br><br>and output a structured JSON object containing your analysis . You must adhere s t r i c t l y to the format and rules described below .<br><br># Instructions : Analyze the video and generate a s i n g l e JSON object with the following keys<br><br>. Your response should ONLY be the JSON object , enclosed in a s i n g l e markdown code block ( ‘ ‘ ‘ json . . . ‘ ‘ ‘) . Do not include any other text , explanations , or introductory phrases .<br><br># JSON Fields Definition :<br><br>1. ‘" category " ‘: ( String ) Select ONLY ONE category from the following l i s t that best describes the main subject of the video .<br><br>I f the video category i s not among the l i s t e d below , output "Others" .<br><br>* ** List of 34 Categories **: [ "Others" , "LifeRecord - TravelLog" , "LifeRecord - DailyLife " , "LifeRecord HouseTour" , "LifeRecord - Reaction" , "LifeRecord - AnimalPet" , "LifeRecord - Cooking" , "LifeRecord - Fashion" , "LifeRecord - Workout" , "Education Lecture " , "Education - Finance" , "Education - Multilingual " , "Education Handifraft " , "Education - Science " , "Education - Art" , "Education OnlineTutorial " , "TVShow- TVSeries" , "TVShow-News" , "TVShow- TalkShow" , "TVShow- Celebration " , "TVShow- CommentaryProgram" , "Competition Football " , "Competition - Athletics " , "Competition - Basketball " , " Competition - Snooker" , "Competition - Boxing" , "Competition - Car" , " VideoGames - Sandbox" , "VideoGames - OpenWorld" , "Documentary - Nature" , " Documentary - Science " , "Documentary - Culture" , "Documentary - Kids" , " Movie -Comedy" , "Movie - Adventure" ]<br><br>2. ‘" caption " ‘: ( String ) A concise but descriptive summary of the video ’ s content in English , not exceeding 30 words . Describe what i s happening , who<br><br>i s involved , and the main subject .<br><br>3. ‘"content_score " ‘: ( Integer ) An integer from 1 to 10. This score evaluates the video ’ s potential as a prompt for a synthetic AI - user conversation .<br><br>* ** Score 1 -3 (Low) **: The video content i s sparse , repetitive , or features a s i n g l e person ’ s monologue with l i t t l e interaction or environmental d e t a i l . It ’ s d i f f i c u l t to ask questions or start a conversation based on i t<br><br>.<br><br>* ** Score 4 -7 (Medium) **: The video has some i n t e r e s t i n g elements but may lack depth or variety . It can support a few questions but may not lead to an extended , rich conversation .<br><br>* ** Score 8 -10 ( High ) **: The video i s rich in content , detail , and action . It shows a process , an interaction , or a complex scene that naturally i n v i t e s questions and allows for a deep , extended conversation ( e . g . , a detailed cooking tutorial , a complex assembly process , a travel vlog with multiple a c t i v i t i e s ) .<br><br><br>4. ‘" fov " ‘: ( Integer ) Field of View .<br><br><br>* **1**: The video i s shot from a f i r s t - person perspective (FPV) , where the camera acts as the viewer ’ s eyes .<br><br>* **0**: The video i s shot from a third - person or s t a t i c perspective .<br><br><br>|
|---|

|5. ‘"task_complexity" ‘: ( Integer ) An integer from 1 to 10 , representing the complexity of the primary task shown in the video . I f no s p e c i f i c task i s shown , rate the complexity of the main a c t i v i t y .<br><br>* ** Score 1 -3 (Low) **: Simple , everyday actions that require minimal s k i l l ( e . g . , opening a bottle , pouring water , petting a cat ) .<br><br>* ** Score 4 -7 (Medium) **: Tasks that require some s k i l l , knowledge , or multiple steps ( e . g . , cooking a simple dish , assembling IKEA furniture , basic makeup application ) .<br><br>* ** Score 8 -10 ( High ) **: Highly complex , specialized , or p r o f e s s i o n a l tasks that require s i g n i f i c a n t expertise , precision , or e f f o r t ( e . g . , performing surgery , building a car engine , p r o f e s s i o n a l programming , playing a<br><br><br>complex musical piece ) . # Required Output Format:<br><br>‘ ‘ ‘ json {<br><br>" category " : " . . . " , " caption " : " . . . " , " content_score " : . . . , " fov " : . . . , "task_complexity" : . . .<br><br>} ‘ ‘ ‘<br><br>|
|---|

