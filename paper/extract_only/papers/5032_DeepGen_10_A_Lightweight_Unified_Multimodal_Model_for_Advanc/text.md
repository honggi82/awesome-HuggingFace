# arXiv:2602.12205v2[cs.CV]13Feb2026

[Figure 1]

[Figure 2]

## DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing

Dianyi Wang1,2*†, Ruihang Li1,3* Feng Han1,2*, Chaofan Ma4*, Wei Song1,5,6*, Siyuan Wang8*, Yibin Wang1,2*, Yi Xin1,7, Hongjian Liu3, Zhixiong Zhang1,4, Shengyuan Ding1,2, Tianhang Wang1,5, Zhenglin Cheng1,5,6, Tao Lin6, Cheng Jin2,

Kaicheng Yu6, Jingjing Chen2, Wenjie Wang3, Zhongyu Wei1,2, Jiaqi Wang1†

1Shanghai Innovation Institute, 2Fudan University, 3University of Science and Technology of China, 4Shanghai Jiao Tong University, 5Zhejiang University, 6Westlake University, 7Nanjing University, 8University of Southern California

*Equal Contribution, †Project Leaders

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

General Generation

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Text Rendering

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Reasoning Generation

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

General Editing

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Reasoning Editing

- Figure 1 Overview of DeepGen 1.0’s visual generation and editing abilities, including reasoning-intensive scenarios.

### Abstract

Current unified multimodal models for image generation and editing typically rely on massive parameter scales (e.g., >10B), entailing prohibitive training costs and deployment footprints. In this work, we present DeepGen 1.0, a lightweight 5B unified model that achieves comprehensive capabilities competitive with or surpassing much larger counterparts. To overcome the limitations of compact models in semantic understanding and fine-grained control, we introduce Stacked Channel Bridging (SCB), a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable “think tokens” to provide the generative backbone with structured, reasoning-rich guidance. We further design a data-centric training strategy spanning three progressive stages: (1) Alignment Pre-training on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) Joint Supervised Fine-tuning on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) Reinforcement Learning with MR-GRPO, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts. Despite being trained on only ∼50M samples, DeepGen 1.0 achieves leading performance across diverse benchmarks, surpassing the 80B HunyuanImage by 28% on WISE and the 27B Qwen-Image-Edit by 37% on UniREditBench. By open-sourcing our training code, weights, and datasets, we provide an efficient, high-performance alternative to democratize unified multimodal research.

GitHub: https://github.com/DeepGenTeam/DeepGen HuggingFace: https://huggingface.co/DeepGenTeam/DeepGen-1.0 Datasets: https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0

### 1 Introduction

Advancing image generation and editing to handle increasingly complex instructions requires models that go beyond mere pixel synthesis to possess deep semantic understanding. To meet this demand, a promising paradigm has emerged that integrates the comprehensive capabilities of vision-language models (VLMs) with the generative power of diffusion models, aiming to achieve semantically accurate generation and precise editing. Closed-source systems such as GPT-Image-1 [1] and Nano Banana [2] have validated this potential. In the open-source domain, a recent wave of models, including BAGEL [3], HunyuanImage 3.0 [4], Qwen-Image [5], and LongCat-Image [6], has actively explored this direction to elevate generative performance through unified understanding. These advancements underscore the transformative impact of unified models in redefining the boundaries of visual generation.

Despite this rapid progress, current high-performing unified models remain prohibitively expensive. Models such as Qwen-Image (27B), HunyuanImage 3.0 (80B), BAGEL (14B), and Emu3.5 (34B) all demand billions of training samples and massive computational resources. Many further require separate generation and editing models, doubling the total parameter count, e.g., pushing deployment footprints to a total of 54B for Qwen-Image & Qwen-Image-Edit and 26B for LongCat-Image & LongCat-Image-Edit. While the need for lightweight alternatives is clear, existing small-scale unified models [7, 8, 9] have consistently underperformed across diverse tasks, thereby reinforcing a common perception: compact models lack the capacity for comprehensive multimodal generation and editing. Interestingly, a closer examination of recent benchmarks challenges this view: performance does not scale monotonically with model size. For example, as shown in Fig. 2, Lumina-DiMOO (8B) achieves a generation score of 86.04 on DPG-Bench, surpassing the larger BAGEL (14B, 85.10). Similar patterns are observed across other benchmarks and evaluation dimensions (Table 1, 2, 3, 4, and 5). This indicates that, for unified multimodal models, larger scale alone does not necessarily guarantee stronger performance.

Motivated by this observation, we argue that a lightweight model, when empowered by synergistic architecture design and data-centric training strategies, can achieve comprehensive capabilities competitive with or even surpassing

Model Performance (Overall Score)

85

| | | | |[Figure 51]<br><br>DeepGe|n 1.0 (SFT)| |
|---|---|---|---|---|---|---|
| | |[Figure 52]<br><br>G|PT-Image-1| |[Figure 53]<br><br>DeepGen|1.0 (RL)|
| | | | | | | |
| | | | | | | |
| | |[Figure 54]<br><br>N|ano Banana| |[Figure 55]|Qwen-Image|
| | |[Figure 56]|BAGEL| | | |
| | | |[Figure 57]<br><br>Lumin|a-DiMOO|[Figure 58]<br><br>S|eedream 4.0|
| |[Figure 59]<br><br>OmniGen2| | | | | |
| | | | | | | |
| | | | | | | |

80

ImageEditing(UniREditBench)

75

70

65

60

55

50

45

40

83 84 85 86 87 88 89

Image Generation (DPG-Bench)

- Figure 2 Model performance comparison on image generation and editing benchmarks. Bubble size is proportional to model parameter count. Dashed outer rings indicate models with unreported parameter counts. Higher scores correspond to better performance.

much larger counterparts. To substantiate this, we present DeepGen 1.0, a compact framework with a total of 5B parameters (3B VLM and 2B DiT) that integrates general generation, reasoning generation, text rendering, general editing, and reasoning editing within a single model. Despite its compact size, DeepGen 1.0 achieves results competitive with or exceeding models 3× to 16× its size, as highlighted in Fig. 2. For instance, in general instruction following DPG-Bench, DeepGen 1.0 attains 87.90, eclipsing massive baselines like HunyuanImage 3.0 (86.10). Moving to reasoning-intensive tasks, it achieves 0.73 on WISE, outperforming the 80B HunyuanImage 3.0 (0.57) by a remarkable 28% margin. Furthermore, on the editing front, it dominates the UniREditBench with 77.5, surpassing the dedicated 27B Qwen-Image-Edit (56.5) by over 37%. Across the board, DeepGen 1.0 demonstrates that intelligent design can triumph over raw scale. Remarkably, the entire training requires only ∼50M samples across a simple three-stage pipeline, compared to 1.2B samples for LongCat-Image and 5B for HunyuanImage 3.0.

To support these comprehensive capabilities within a compact 5B budget, we introduce a specialized architecture that maximizes VLM-DiT synergy. DeepGen 1.0 employs a 3B VLM [10] as the understanding and reasoning backbone and a 2B DiT [11] as the generative backbone. To align these two modules, we propose Stacked Channel Bridging (SCB). SCB first extracts hidden states from six uniformly distributed VLM layers (spanning low, mid, and high levels) to capture hierarchical features from visual and text inputs. To further enhance reasoning, we inject learnable “think tokens” that act as an implicit chain of thoughts. These multi-source features are then channel-wise concatenated and fused via a lightweight connector into a dense multimodal conditional sequence. Unlike prior methods that rely on the final VLM layer [5, 12] or use average pooling [13] that blurs fine-grained details, this design fully preserves both fine-grained visual details and high-level semantics, while providing the DiT with structured, reasoning-rich guidance.

To fully unlock the potential of of DeepGen 1.0’s compact architecture, we design a data-centric training strategy tailored for tight VLM-DiT integration in the low-parameter regime. This strategy emphasizes simplicity and data efficiency across three progressively stages. First, in Alignment Pre-training, we optimize only the connector and learnable think tokens to align VLM representations with the DiT’s latent space, utilizing large-scale image-text pairs and editing triplets. Second, during Joint Supervised Fine-tuning (SFT), we unfreeze the DiT and apply LoRA to the VLM for end-to-end optimization. We curate a high-quality data mixture by integrating general generation and editing data, reasoning-based generation and editing data, and text-rendering data to foster omni-capabilities while preserving the VLM’s inherent knowledge. Finally,

[Figure 60]

- Figure 3 Overview of DeepGen 1.0 architecture. DeepGen 1.0 adopts a unified VLM-DiT paradigm with a dual-branch visual encoding strategy: a ViT encoder captures high-level semantics for the VLM, while a VAE encoder extracts compressed latent features for the DiT. Multimodal conditions derived from the VLM, together with reference-image VAE latents, are concatenated with the target image’s noise tokens to form a single DiT input sequence, enabling self-attention over both conditioning and generation signals. Stacked channel bridging (SCB) performs deep feature fusion between the VLM and DiT to strengthen generation and editing, while DiT positional encodings explicitly distinguish reference tokens from target tokens. Icons shown at the right of each block indicate whether the corresponding module is frozen or trainable during the Pre-Training, SFT, and RL stages, respectively.

we employ Reinforcement Learning (RL) to further align the model with human preferences. We adopt our novel MR-GRPO, with mixture of rewards and supervision signals, enhancing it with decoupled advantage normalization [14] to better preserve multi-reward granularity. To prevent capability degradation during RL, we introduce an auxiliary supervised diffusion loss, ensuring the model retains the broad capabilities acquired during the joint supervised fine-tuning stage.

Our contributions are summarized as follows:

- • We present DeepGen 1.0, a compact 5B unified model that integrates general generation, reasoning, text rendering, and editing within a single framework. Despite its small size, it achieves performance competitive with or surpassing models up to 16× larger (e.g., 80B), demonstrating that massive scaling is not the sole path to high-performance multimodal generation.
- • We propose Stacked Channel Bridging (SCB), a lightweight alignment module that fuses multi-layer VLM features via channel concatenation and a shallow connector. Augmented with learnable think tokens, SCB enables deep semantic transfer from the VLM to the DiT while preserving fine-grained visual details, offering a superior alternative to standard final-layer or average-pooling approaches.
- • We design a data-centric training strategy spanning three progressive stages: (1) alignment pre-training on large-scale pairs and triplets, (2) joint SFT on a high-quality mixture of generation, reasoning, editing, and text rendering tasks, and (3) we propose MR-GRPO for RL alignment with auxiliary supervision and mixture of rewards , enabling stable preference optimization without capability degradation.
- • We conduct comprehensive evaluations across diverse benchmarks, demonstrating leading performance among open-source models in reasoning-based generation and editing, while maintaining competitive general generation quality.

[Figure 61]

- Figure 4 Overview of our training data for broad omni-capabilities and comprehensive evaluation across benchmarks.

• We publicly release the DeepGen 1.0 framework, including model weights, training and evaluation code, and key data components. By providing an efficient and high-performance alternative to resourceintensive large models, we aim to democratize unified multimodal research and empower broader community exploration.

### 2 Model Architecture

DeepGen 1.0 follows a VLM-DiT architecture as shown in Fig 3, where the VLM offers strong multimodal understanding with well cross-modal alignment and rich world knowledge to capture complex multimodal priors from both textual and visual inputs. The DiT serves as a high-fidelity generation decoder guided by multimodal conditional inputs extracted from the VLM. We utilize Qwen-2.5-VL (3B) [10] as our pretrained VLM and SD3.5-Medium (2B) as our DiT, initialized from [11] with joint generation and editing capability. Feature alignment is achieved via a streamlined connector module, which instantiates a SigLIP visual encoder [15] followed by six transformer layers [16]. This compact design maintains a total model size of approximately 5B parameters, striking an optimal balance between performance and computational efficiency.

Stacked Channel Bridging (SCB) Prior unified multimodal models [5, 6, 12, 17] typically take the final-layer (or penultimate-layer) hidden states of a VLM, transform them through a connector, and use them as multimodal conditional input to the DiT. This design has two key limitations. First, the final VLM layers are heavily biased toward high-level semantic abstraction, often discarding fine-grained visual details that are critical for DiT modeling [18]. Second, relying on a single layer makes the conditional signal vulnerable to layer-specific representation biases, which can hinder stable alignment and effective fusion between the VLM and DiT. An alternative line of work [3, 19, 20] performs deep fusion by introducing shared attention between the VLM and DiT at every layer. However, this approach substantially increases parameter scale and optimization complexity, making efficient and reliable training challenging. Subsequent works [13] aggregate hidden states from multiple VLM layers using average pooling.

To more effectively and efficiently aggregate features from multiple VLM layers while preserving fine-grained information and enhancing reasoning, we propose the Stacked Channel Bridging (SCB) framework. SCB operates through three integrated steps:

- - Think Token Injection. While standard VLM representations provide rich interleaved multimodal signals [7, 21], explicit reasoning tokens can further act as implicit Chains of Thought (CoT). To strengthen the model’s reasoning capability, we first inject a fixed set of learnable “think tokens” into the VLM input sequence. These tokens interact with textual and visual inputs across all layers via self-attention, progressively summarizing hidden representations and effectively extracting knowledge encoded in the VLM.

- Table 1 Comparison of different models across general image generation and editing benchmarks. Top-1/2/3 results within each column excluding closed-source models are marked with gold, silver, and bronze icons.

General T2I Generation General Editing Model Params

GenEval↑ DPGBench↑ UniGenBench↑ ImgEdit↑ GEdit-EN↑

Closed-source Models

Nano Banana – 0.75 85.23 87.45 4.35 7.54 GPT-Image-1 – 0.84 85.20 92.77 4.20 7.53 Seedream 4.0 – 0.84 88.25 87.30 4.18 7.68

FLUX.1 Kontext [Pro] – – – 75.84 4.00 6.56 Open-source Models

Janus-Pro 7B 0.80 84.20 61.61 – – Show-o2 7B 0.76 86.14 62.73 – – BLIP3-o 7B + 1.4B 0.84 81.60 59.87 – –

MetaQuery-XL 7B+ 1.6B 0.80 82.05 – – –

OmniGen2 3B + 4B 0.80 83.57 63.09 3.43 6.41 UniWorld v1 7B + 12B 0.80 81.38 63.11 3.26 4.85

BAGEL 14B 0.82 85.10 61.53 3.20 6.52 FLUX.1 [Dev] 12B 0.82 83.84 69.88 – –

[Figure 62]

X-Omni 7B + 12B 0.83 87.65 53.77 – – Lumina-DiMOO 8B 0.88 86.04 71.12 – –

[Figure 63]

[Figure 64]

Mammoth2 8B + 3B + 2B 0.87 87.20 – 4.06 6.60

[Figure 65]

LongCat-Image 7B + 6B 0.87 86.80 – – – LongCat-Image-Edit 7B + 6B – – – 4.50 7.60 Hunyuan-Image 3.0 80B 0.72 86.10 – – –

[Figure 66]

[Figure 67]

Z-Image-Turbo 4B + 6B 0.84 85.15 71.40 – – Qwen-Image 7B + 20B 0.87 88.32 78.81 – – Qwen-Image-Edit [2509] 7B + 20B – – – 4.35 7.54 GLM-Image 9B + 7B – 84.78 – – –

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

DeepGen 1.0 (SFT) 3B + 2B 0.86 87.05 74.18 4.09 7.12 DeepGen 1.0 (RL) 3B + 2B 0.87 87.90 75.74 4.14 7.17

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- - Layer Selection. With the think tokens injected, we select multiple VLM hidden states to fuse, balancing performance and computational efficiency. Instead of relying on a single layer, and following [22] which suggests that sparsely and uniformly distributed layers within VLMs provide effective representations for visual information, we select six hidden states sampled uniformly across the low-, mid-, and high-level layers. This ensures the capture of varying-granularity visual features and semantics, alongside the reasoning information embedded in the think token positions.
- - Feature Fusion. Finally, we integrate the selected multi-layer hidden states, which now encode both

multimodal features and think token representations. Given a set of selected VLM hidden states [𝑥1, . . . , 𝑥𝑛] ∈ ℝ𝐿×𝑑 where 𝑛 denotes the number of selected layers and 𝐿 is the sequence length (including think tokens), we first stack them along the channel dimension. This concatenated feature tensor in dimension 𝑑′ is then projected to match the DiT input width using a lightweight two-layer MLP. The aligned features are then fed into a Transformer-encoder-based connector to deeply fuse information across layers, producing the final robust conditional input 𝑐 ∈ ℝ𝐿×𝑑

DiT:

𝑐 = Encoder(MLP(Concatch(𝑥1, . . . , 𝑥𝑛))). (1)

### 3 Training

#### 3.1 Stage 1: Alignment Pre-Training

In the initial stage, we focus on establishing alignment between the VLM and the DiT. To achieve this, we train only the connector and 128 learnable think tokens while keeping all other model parameters frozen.

- Table 2 Evaluation of reasoning-based text-to-image generation involving world knowledge on the WISE [23] benchmark. "*" denotes generation with textual CoT reasoning.

Model Params Cultural Time Space Biology Physics Chemistry Overall↑

Closed-source Models

GPT-Image-1 – 0.81 0.71 0.89 0.83 0.79 0.74 0.80 Seedream 4.0 – 0.78 0.73 0.85 0.79 0.84 0.67 0.78

Open-source Models

Janus-Pro 7B 0.30 0.37 0.49 0.36 0.42 0.26 0.35 FLUX.1 [Dev] 12B 0.48 0.58 0.62 0.42 0.51 0.35 0.50

MetaQuery-XL 7B+ 1.6B 0.56 0.55 0.62 0.49 0.63 0.41 0.55

BLIP3-o 7B + 1.4B – – – – – – 0.62 UniWorld-V1 7B + 12B 0.53 0.55 0.73 0.45 0.59 0.41 0.55

OmniGen2 3B + 4B 0.42 0.52 0.64 0.43 0.50 0.34 0.47 BAGEL* 14B 0.76 0.69 0.75 0.65 0.75 0.58 0.70

[Figure 80]

NextFlow-RL 7B + 18B 0.63 0.63 0.77 0.58 0.67 0.39 0.62

STAR 7B 0.61 0.67 0.61 0.74 0.69 0.66 0.66 Hunyuan-Image 3.0 80B 0.58 0.57 0.70 0.56 0.63 0.31 0.57

Qwen-Image 7B + 20B 0.62 0.63 0.77 0.57 0.75 0.40 0.62 LongCat-Image 7B + 6B 0.66 0.61 0.72 0.66 0.72 0.49 0.65

[Figure 81]

DeepGen 1.0 (SFT) 3B + 2B 0.70 0.71 0.82 0.62 0.79 0.65 0.72 DeepGen 1.0 (RL) 3B + 2B 0.72 0.81 0.70 0.67 0.82 0.66 0.73

[Figure 82]

This phase utilizes general text-to-image generation and image editing tasks. Specifically, the model is trained for 200,000 iterations with the data details listed in Table 8. All images are generated at a fixed resolution of 512 × 512. We utilize a learning rate of 1 × 10−4 with 20,000 warm-up steps. For a complete list of hyperparameters, please refer to Table 9 in Appendix A.

#### 3.2 Stage 2: Joint Supervised Fine-Tuning

In the second stage, we unfreeze the entire model and conduct a joint VLM-DiT training, aiming to strengthen instruction-following capability and image synthesis quality with improved visual fidelity, semantic alignment, and knowledge-aware reasoning. To mitigate potential degradation of the VLM’s multimodal comprehension during joint optimization, we apply LoRA [24] for efficient fine-tuning of the VLM. We train the model on a diverse and high-quality mixture of tasks designed to foster omni abilities, including general text-to-image generation and editing, reasoning-based generation and editing, and text rendering.

We perform supervised fine-tuning for 400,000 iterations on the multi-task dataset detailed in table 8. Images are trained at a fixed resolution of 512×512 while preserving the original aspect ratio via dynamic resizing. The model is optimized with a learning rate of 5 × 10−5 with 20,000 warm-up steps. Detailed LoRA configurations and hyperparameters are provided in Table 9 of Appendix A.

DeepGen 1.0 follows a VLM-DiT architecture as shown in Fig 3, where the VLM offers strong multimodal understanding with well cross-modal alignment and rich world knowledge to capture complex multimodal priors from both textual and visual inputs. The DiT serves as a high-fidelity generation decoder gudided by multimodal conditional inputs extracted from the VLM. We utilize Qwen-2.5-VL (3B) [10] as our pretrained VLM and SD3.5-Medium (2B) as our DiT, initialized from [11] with joint generation–editing capability. Feature alignment is achieved via a streamlined connector module, which instantiates a SigLIP visual encoder [15] followed by six transformer layers [16]. This compact design maintains a total model size of approximately 5B parameters, striking an optimal balance between performance and computational efficiency.

#### 3.3 Stage 3: Reinforcement Learning

To further improve generation quality and alignment with human preferences, we apply reinforcement learning after supervised fine-tuning. We propose the MR-GRPO framework, a variant of Pref-GRPO [27],

- Table 3 Evaluation of reasoning-based text-to-image generation with the philosophical framework on the T2ICoREBench [25] benchmark through Qwen3-VL-32B-Thinking [26]. "*" denotes generation with textual CoT reasoning.

Model Params R-LR R-BR R-HR R-PR R-GR R-AR R-CR R-RR Overall↑

Closed-source Models

Nano Banana – 65.4 59.7 57.2 88.3 83.5 84.1 67.5 58.7 70.5 GPT-Image-1 – 61.6 52.0 58.1 89.9 76.7 82.4 67.7 47.5 67.0 Seedream 4.0 – 79.2 51.4 52.9 89.1 88.6 80.1 70.8 42.8 69.4

Open-source Models Janus-Pro 7B 27.2 15.9 28.0 25.4 7.3 30.8 8.8 4.6 18.5

FLUX.1 [Dev] 12B 26.3 18.0 25.9 66.8 38.0 59.7 35.7 18.1 36.1 Show-o2 7B 30.2 21.3 29.4 59.7 40.4 54.7 32.8 13.1 35.2 BLIP3-o 7B + 1.4B 18.4 16.0 19.0 44.6 45.0 51.1 36.8 12.3 30.4

OmniGen2 3B + 4B 26.8 19.2 32.9 64.1 37.5 56.5 37.9 13.6 36.1

BAGEL* 14B 28.6 22.2 24.8 66.2 55.8 59.5 42.6 29.3 41.1 Hunyuan-Image 3.0 80B 41.6 27.4 42.3 76.3 52.7 52.2 55.1 20.6 46.0

[Figure 83]

Qwen-Image 7B + 20B 42.2 29.5 40.0 78.6 47.9 55.2 59.0 18.4 46.3 Z-Image-Turbo 4B + 6B 37.8 24.8 37.8 75.6 46.0 59.4 49.6 18.6 43.7 LongCat-Image 7B + 6B 41.7 32.2 38.4 78.3 72.6 66.3 55.8 32.6 52.2 DeepGen 1.0 (SFT) 3B + 2B 38.8 28.7 40.2 79.1 51.5 65.7 42.0 19.8 45.7 DeepGen 1.0 (RL) 3B + 2B 38.5 29.0 41.2 79.5 51.9 66.9 45.6 19.6 46.5

[Figure 84]

[Figure 85]

which extends Group Relative Policy Optimization (GRPO) [28] to flow matching models by performing on-policy stochastic sampling and evaluating each generated image with a mixture of pointwise and pairwise reward models. We further introduce a novel auxiliary supervised diffusion loss that complements KL regularization to mitigate capability degradation during prolonged RL training. In addition, we validate and adopt two concurrent improvements into our pipeline: (1) a noise-preserving stochastic sampling strategy [29] that produces cleaner samples and more accurate reward signals, and (2) a decoupled advantage normalization scheme [14] that better preserves multi-reward signal granularity.

Concretely, given a text condition ℎ, the flow model samples a group of 𝐺 images {𝑥0𝑖 }𝐺𝑖=1 and the corresponding denoising trajectories {𝑥𝑇𝑖 , 𝑥𝑇𝑖 −1, . . . , 𝑥0𝑖 }𝐺𝑖=1. For multi-reward optimization with reward functions {𝑅𝑘}𝐾𝑘=1, we normalize each reward independently within each group before aggregation, following [14]:

𝑅𝑘(𝑥0𝑖 , ℎ) − mean({𝑅𝑘(𝑥0𝑗 , ℎ)}𝐺𝑗=1) std({𝑅𝑘(𝑥0𝑗 , ℎ)}𝐺𝑗=1)

, (2)

𝐴𝑖𝑘 =

and obtain the final advantage 𝐴ˆ𝑖 via weighted aggregation 𝑘 𝑤𝑘𝐴𝑖𝑘 followed by batch-wise normalization across the training batch. The training objective is:

ℒGRPO(𝜃) = 𝔼ℎ∼𝒟

𝑇−1

𝐺

1 𝑇

1 𝐺

𝑡=0

𝑖=1

min 𝑟𝑡𝑖(𝜃) 𝐴ˆ𝑖, clip(𝑟𝑡𝑖(𝜃), 1−𝜖, 1+𝜖) 𝐴ˆ𝑖 − 𝛽 𝐷KL(𝜋𝜃∥𝜋ref) , (3)

where 𝑟𝑡𝑖(𝜃) = 𝑝𝜃(𝑥𝑡𝑖−Δ𝑡|𝑥𝑡𝑖, ℎ) 𝑝𝜃old(𝑥𝑡𝑖−Δ𝑡|𝑥𝑡𝑖, ℎ) is the per-step importance ratio. We use 3 complementary reward functions to jointly optimize visual quality, text rendering accuracy, and semantic alignment; details

on the reward design, stochastic sampler, and training configuration are deferred to Appendix B. The KL-divergence regularization is computed in velocity space:

𝐷KL(𝜋𝜃∥𝜋ref) = ∥ˆ𝑣𝜃(𝑥𝑡, 𝑡) − 𝑣ˆref(𝑥𝑡, 𝑡)∥2. (4)

While the KL penalty constrains the policy from drifting too far from the reference model, we observe that it alone is insufficient to prevent capability degradation as RL training scales beyond ∼1000 steps:

0.34

0.756

| |Overall<br><br>Text<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

UniGenBenchOverall

UniGenBenchText

0.754

0.32

0.752

0.30

0.750

0.28

0.748

0.26

0 500 1000 1500

Training Steps

- Figure 5 UniGenBench evaluation curves during RL training over 1,500 steps. The left axis shows the overall score and the right axis shows the text generation sub-score. Both metrics improve steadily throughout training, with the overall score rising from ∼0.747 to ∼0.756 and the text score increasing from ∼0.25 to ∼0.34, demonstrating that RL simultaneously enhances text rendering fidelity and general generation quality.

the model exhibits a notable performance drop on tasks requiring complex instruction comprehension, such as reasoning-based generation. We attribute this to the complementary nature of the two forms of regularization: KL divergence acts as process-level guidance, constraining the denoising trajectory to stay close to the reference policy at each step, whereas the supervised loss provides outcome-level guidance, directly anchoring the final generation quality to the SFT distribution. Process-level constraints alone, without outcome-level anchoring, leave the model susceptible to gradual drift during prolonged training. To this end, we introduce an auxiliary supervised diffusion loss ℒSFT computed on our high-quality SFT dataset, which continuously anchors the model to its supervised fine-tuning distribution. The overall training objective is:

ℒtotal = (1 − 𝜆)ℒGRPO + 𝜆 ℒSFT, (5)

where ℒSFT is the standard flow matching loss and 𝜆 is a small mixing coefficient. This formulation allows the model to optimize for reward signals via GRPO while retaining the generation capabilities acquired during supervised fine-tuning.

### 4 Data

The overall composition of our training data is illustrated in Fig. 4. It combines real-world, synthetic, and carefully curated open-source datasets, covering a broad spectrum of tasks including general generation and editing, reasoning-based generation and editing, text rendering, and application-oriented scenarios.

General Generation Our pre-training corpus is sourced from several publicly available image–text pair datasets, including text-to-image-2M [30], LAION-Aesthetic-6M [31], Megalith-10M [32], RedCaps-5M [33], and CC-12M [34]. For high-quality instruction fine-tuning, we curate a mixture of open instruction-following datasets, including BLIP-3o (60k samples) [7], ShareGPT-4o-Image (45k samples) [35], Echo-4o-Image (100k samples) [36], and OpenGPT4o-Image (40k samples) [37]. These are combined with 10M in-house real samples spanning both long- and short-form prompts (ratio 3:1). In addition, we synthesize approximately 50k high-clarity photorealistic images paired with fine-grained prompts using Nano Banana, further enriching detailed image generation covering both Chinese and English.

General Editing For general image editing, we collect image-instruction-image triplets from a variety of open-source datasets, including NHR-Edit [38] (720k samples), GPT-Image-Edit (1.5M samples) [39], ShareGPT-4o-Image-Edit set (50k samples) [35], OpenGPT4o-Image-Edit set (40k samples) [37], Nano-bananaconsist (150k samples) [40], Pico-Banana (250k samples) [41], X2I2 [12](1.6M samples) and Uniworld-Edit set [17](1.2M samples) together with 1.1M in-house editing samples covering both Chinese and English.

Reasoning-based Generation and Editing We utilize reasoning generation and editing datasets (150k and 100k samples, respectively) from UniReason [42], covering five major knowledge domains: cultural commonsense, natural science, spatial, temporal and logical reasoning.

Table 4 Evaluation of reasoning-based editing involving world knowledge on the RISE [43] and UniREditBench [44]. "*" denotes generation with textual CoT reasoning.

|RISE<br><br>| |UniREditBench| |
|---|---|---|---|
|Temporal Causal Spatial Logical|Overall↑|Real World Game World<br><br>|Overall↑|

Model Params

Closed-source Models

Nano Banana – 25.9 47.8 37.0 18.8 32.8 75.2 60.4 68.3 GPT-Image-1 – 34.1 32.2 37.0 10.6 28.9 81.0 62.1 73.4 Seedream 4.0 – 12.9 12.2 11.0 7.1 10.8 66.2 45.4 55.8

FLUX-Kontext-Pro – – – – – – 45.0 46.5 45.8 Open-source Models

FLUX.1-Kontext [Dev] 12B 2.3 5.5 13.0 1.2 5.8 – – –

OmniGen2 3B + 4B – – – – – 53.7 33.1 43.4 Lumina-DiMOO 8B – – – – – 51.4 45.6 48.5

BAGEL* 14B 5.9 17.8 21.0 1.2 11.9 56.8 45.1 51.0 Qwen-Image edit [2509] 7B + 20B 4.7 10.0 17.0 2.4 8.9 71.0 41.9 56.5

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

DeepGen 1.0 (SFT) 3B + 2B 15.3 18.9 14.0 4.7 13.3 74.3 80.7 77.5 DeepGen 1.0 (RL) 3B + 2B 12.9 14.4 13.0 2.4 10.8 73.2 78.2 75.7

[Figure 90]

[Figure 91]

Text Rendering and Application-oriented Scenarios To strengthen text rendering, we curate captions from document- and infographic-centric multimodal QA datasets [45]. Gemini 2.5 Pro [46] is used to stochastically compose diverse rendering attributes, e.g., font styles, layouts, and color schemes, and combine them with an open-source prompt set tailored for text rendering from [47]. Corresponding images are synthesized using Qwen-Image, resulting in 500k text-rendering samples. We further extend the corpus to application-oriented scenarios such as Chinese poetry generation and poster design, contributing an extra 60k samples.

The detailed dataset usage in each stage is provided in Table 8 of Appendix A.

### 5 Experiments

#### 5.1 Evaluation Setup

General Generation We assess general text-to-image generation using GenEval [48] to measure fundamental semantic alignment, and DPG-Bench [49] to assess long-prompt instruction following. In addition, we adopt UniGenBench [27] for a comprehensive and fine-grained evaluation of general generation capability, covering ten major categories (e.g., attribute binding, style control, and text rendering).

Reasoning Generation We evaluate world-knowledge reasoning-based generation on WISE [23], which contains 1,000 prompts spanning cultural knowledge, natural science, and spatial–temporal understanding. In addition, we adopt the T2I-CoREBench reasoning set [25], which covers eight reasoning categories—Logical (R-LR), Behavioral (R-BR), Hypothetical (R-HR), Procedural (R-PR), Generalization (R-GR), Analogical (R-AR), Commonsense (R-CR), and Reconstructive (R-RR)—to assess reasoning generation under a structured, philosophy-inspired taxonomy.

General Editing We evaluate general image editing on ImgEdit [50] and GEdit-EN [51]. These benchmarks assess core editing competencies, including instruction following, editing consistency and output quality.

Reasoning Editing We evaluate world-knowledge reasoning-based image editing using UniREditBench [44] with 2,700 meticulously curated samples covering both real- and game-world scenarios, and RISE [52] with 327 samples across temporal, causal, spatial, and logical dimensions.

Text Rendering We evaluate text rendering performance on CVTG-2K [53], which focuses on English text generation across diverse real-world scenarios, including street scenes, advertisements, and memes.

Table 5 Evaluation of text rendering on the CVTG-2K [53].

Model Params Word Accuracy↑ NED↑ CLIPScore↑

Closed-source Models

Nano Banana Pro – 0.7788 0.8754 0.7372 GPT-Image-1 – 0.8569 0.9478 0.7982 Seedream 4.0 – 0.8451 0.9224 0.7975

Open-source Models FLUX.1 [dev] 12B 0.4965 0.6879 0.7401

[Figure 92]

[Figure 93]

Z-Image-Turbo 4B + 6B 0.8585 0.9281 0.8048 Hunyuan-Image 3.0 80B 0.7650 0.8765 0.8121

[Figure 94]

Qwen-Image 7B + 20B 0.8288 0.9116 0.8017 LongCat-Image 7B + 6B 0.8658 0.9361 0.7859

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

GLM-Image 9B + 7B 0.9116 0.9557 0.7877 DeepGen 1.0 (SFT) 3B + 2B 0.6605 0.8426 0.8227

[Figure 99]

[Figure 100]

DeepGen 1.0 (RL) 3B + 2B 0.7533 0.8936 0.8278

#### 5.2 Model Performance

We compare DeepGen 1.0 against a broad set of strong baselines, covering both closed-source and open-source models. Closed-source systems include GPT-Image-1 [1], the Nano Banana family (i.e., Gemini-2.5-FlashImage [2]), Seedream 4.0 [54], and FLUX.1 Kontext [Pro] [55]. Open-source baselines span advanced generation-only models such as FLUX.1 [Dev] [55] and Z-Image-Turbo [56], as well as state-of-the-art unified multimodal models supporting both multimodal understanding and image synthesis. These include autoregressive unified models (e.g., Janus-Pro [57]) and discrete diffusion-based approaches (e.g., Lumina-DiMOO [58]).

Most unified models follow the VLM–DiT paradigm, connecting VLMs with diffusion transformers via explicit connectors. Representative examples include BLIP-3o [7] and MetaQuery-XL [21], which use a fixed set of learnable tokens to convey multimodal conditions to the DiT, as well as UniWorld-V1 [17], OmniGen2 [12], the Qwen-Image series [5], and LongCat-Image [6], which condition the DiT on single-layer VLM hidden states. In contrast, deep-fusion methods tightly couple VLMs and DiTs through shared attention within a unified backbone, as exemplified by Hunyuan-Image-3.0 [4], BAGEL [3], and Show-o2 [59].

We further include models that autoregressively predicts discrete image tokens as conditions for subsequent DiT refinement, such as X-Omni [60], GLM-Image [61], NextFlow-RL [62], STAR [63], and Mammoth2 [13]. Notably, our DeepGen 1.0 remains highly lightweight, with only approximately 5B parameters, whereas most competing unified multimodal models operate at 7B parameters or more.

##### 5.2.1 Performance of General Generation and Editing

As shown in Table 1, DeepGen 1.0 achieves a strong performance–efficiency trade-off. With only 5B parameters (3B+2B), it consistently matches or surpasses substantially larger unified multimodal baselines across a wide range of general generation and editing benchmarks, ranking among the top three in all evaluated settings. Notably, DeepGen 1.0 unifies high-quality generation and editing within a single model, rather than relying on separate specialized models.

General Generation On GenEval [48], DeepGen 1.0 achieves 0.87, matching leading models such as Qwen-

- Image [5] and LongCat-Image [6] while using significantly fewer parameters and no external LLM-based prompt rewriting. On DPGBench [49], it scores 87.90, ranking second and demonstrating strong long-horizon instruction following ability. On the more comprehensive UniGenBench, DeepGen 1.0 achieves 75.74, again ranking second and outperforming many larger open-source baselines, including LongCat-Image [6], Z-Image-Turbo [56], and Hunyuan-Image 3.0 [4]. Despite using approximately 4× fewer parameters, it approaches open-source state-of-the-art performance. Overall, these results demonstrate DeepGen 1.0’s robust semantic alignment, strong long-horizon instruction following for long prompts, and comprehensive fine-grained generation capabilities.

Table 6 Ablation study of DeepGen 1.0 architecture.

GenEval DPGBench GEdit-EN WISE RISE DeepGen 1.0 Settings 0.86 87.05 7.12 0.72 13.3

w/o SCB 0.86 85.55 6.75 0.70 12.6 w/o Think Tokens 0.87 86.35 7.02 0.68 11.7 w/o Activate VLM 0.85 86.74 6.93 0.71 12.9

General Editing On ImgEdit [50] and GEdit-EN [51], DeepGen 1.0 remains highly competitive, ranking third under RL. It outperforms strong unified baselines such as Mammoth2, BAGEL, and OmniGen2, while approaching the performance of larger, edit-specialized models (e.g., Qwen-Image-Edit and LongCat-ImageEdit). Across both generation and editing, RL consistently yields further performance gains. As the RL curve on UniGenBench visualized in Fig 5, RL simultaneously enhances the model’s general capabilities and text rendering performance.

##### 5.2.2 Performance of Reasoning-based Generation and Editing

While maintaining strong general capabilities, DeepGen 1.0 exhibits advanced reasoning performance under a compact 5B (3B+2B) parameter budget across both reasoning-based generation and editing benchmarks. Results for world-knowledge reasoning-based generation on WISE [23], T2I-CoREBench [25], and worldknowledge-grounded editing on RISE [52] and UniREditBench [44] are shown in Table 2, 3, and 4, respectively.

Reasoning-based Generation On WISE, DeepGen 1.0 achieves the best performance (0.73) among open-source models, outperforming strong baselines such as BAGEL [3] (relying on explicit CoT for reasoning), LongCat-

- Image [6], and STAR [63], while further narrowing the gap to closed-source systems (e.g., GPT-Image-1 [1] and Seedream 4.0 [54]). Improvements are consistent across diverse knowledge domains including cultural, temporal, spatial, and natural scientific reasoning, demonstrating DeepGen 1.0’s effective use of world knowledge during generation. On T2I-CoREBench, DeepGen 1.0 attains 46.5, ranking among the top opensource models and matching or slightly surpassing substantially larger baselines such as Qwen-Image [5], Hunyuan-Image 3.0 [4], and Z-Image-Turbo [56]. This indicates broad coverage across diverse reasoning types, including logical, procedural, analogical, commonsense, and reconstructive reasoning.

Reasoning-based Editing DeepGen 1.0 also demonstrates strong reasoning-based editing capability. On RISE, it achieves a leading overall score 13.3 (ranked 1st) with SFT and remaining competitive under RL. On UniREditBench, it achieves 77.5 (SFT) and 75.7 (RL), significantly outperforming other open-source baselines and even exceeding the closed-source GPT-Image-1 overall. These results highlight DeepGen 1.0’s robust world-knowledge-grounded editing across both real-world and game-world scenarios [64].

##### 5.2.3 Performance of Text Rendering

As shown in Table 5, DeepGen 1.0 exhibits strong text-rendering performance with only 5B parameters. RL training substantially improves Word Accuracy from 0.6605 to 0.7533, significantly enhancing character-level correctness and legibility. Meanwhile, DeepGen 1.0 preserves the highest CLIPScore (0.8278) among opensource models, indicating that improved textual fidelity does not compromise overall semantic alignment. These results validate that our RL stage effectively enhances precise text synthesis while maintaining strong instruction-level consistency.

#### 5.3 Ablation Study

##### 5.3.1 Architecture Design

We conduct ablation studies to quantify the contribution of key architectural components in DeepGen 1.0, by respectively implementing without applying: (1) stacked channel bridging, (2) think tokens, and (3) VLM activation. Results across benchmarks are shown in Table 6.

Effect of SCB. Removing Stacked Channel Bridging (w/o SCB) consistently degrades performance across all benchmarks: DPGBench drops from 87.05 to 85.55, GEdit from 7.12 to 6.75, WISE from 0.72 to 0.70, and RISE from 13.3 to 12.6. This verifies that SCB effectively aggregates multiple-layer VLM features and mitigates

0.756

Baseline

w/o SFT Loss

w/o Reward-wise Norm

0.754

w/o KL

UniGenBench(Overall)

0.752

0.750

0.748

0.746

0.744

0 200 400 600 800 1000

Training Steps

0.34

Baseline

w/o SFT Loss

w/o Reward-wise Norm

UniGenBench(TextGeneration)

w/o KL

0.32

0.30

0.28

0.26

0.24

0 200 400 600 800 1000

Training Steps

- Figure 6 Evaluation curves during training for ablation variants on UniGenBench. (a) Overall score showing the importance of auxiliary SFT loss for training stability. Without it, performance degrades after ∼300 steps and falls well below the starting point. (b) Text generation score demonstrating that all methods improve text rendering, but removing the SFT loss results in slower and less stable progress.

information loss compared to single-layer conditioning, thereby providing higher-quality multimodal signals to the DiT for both generation and editing.

Effect of Think Tokens. Removing the learnable think tokens (w/o Think Tokens) leads to the most pronounced regression on reasoning-intensive benchmarks: WISE decreases from 0.72 to 0.68 and RISE from 13.3 to 11.7. This suggests that think tokens serve as an implicit reasoning buffer that distills knowledge from VLM representations, strengthening world-knowledge-driven generation and editing beyond what hidden-state conditioning alone.

Effect of Activating the VLM. Disabling VLM activation (w/o Activate VLM) also harms performance (e.g., GenEval 0.85, GEdit 6.93, WISE 0.71, RISE 12.9), indicating that modest VLM fine-tuning improves alignment with the DiT and downstream tasks, yielding more robust generation, editing, and reasoning.

##### 5.3.2 RL Settings

To validate the contribution of each setting in our MR-GRPO framework, we conduct ablation studies by removing: (1) the auxiliary SFT loss, (2) the KL divergence regularization, and (3) the reward-wise advantage normalization. All variants are trained for 1,000 steps under identical configurations and evaluated on UniGenBench.

Effect of Auxiliary SFT Loss. The auxiliary SFT loss is critical for maintaining generation quality during extended RL training. As shown in Figure 6(a), removing this loss leads to performance degradation after approximately 300 steps, eventually dropping well below the initial checkpoint by the end of training. Figure 6(b) further shows that text rendering improvement is also slower and more erratic without the SFT loss, lagging behind the baseline throughout most of training. This indicates that KL regularization alone is insufficient to anchor the model to its supervised fine-tuning distribution, and the SFT loss provides essential positive guidance that prevents capability drift and stabilizes learning across all objectives.

Effect of KL Regularization. Removing KL regularization leads to a lower UniGenBench overall score (75.07 vs. 75.69) and a noticeable drop on DPGBench (87.32 vs. 87.75), as shown in Table 7. Figure 6(a) further reveals that the w/o KL variant lags behind the baseline throughout training, indicating that unconstrained policy updates can lead to forgetting of capabilities acquired during supervised fine-tuning. The combination of KL regularization and auxiliary SFT loss provides complementary constraints: KL penalizes divergence from the reference policy, while SFT loss provides positive guidance toward high-quality generation.

Effect of Reward-wise Normalization. Normalizing advantages independently for each reward before aggregation stabilizes multi-reward optimization. As shown in Figure 6(a), replacing reward-wise normalization with joint normalization across all rewards yields comparable performance in the early stages but leads to a growing gap after approximately 600 steps, with the final performance falling notably short of the baseline.

- Table 7 Ablation study of RL training settings. All variants are trained for 1,000 steps and evaluated on generation (GenEval, DPGBench & UniGenBench) and editing (GEdit-EN). We individually remove the auxiliary SFT loss, velocity KL regularization and reward-wise advantage normalization from the full configuration.

GenEval DPGBench GEdit-EN UniGenBench (Text) UniGenBench (Overall) DeepGen 1.0 (RL) 0.87 87.75 7.05 35.06 75.69 w/o Auxiliary SFT Loss 0.87 87.40 (-0.35) 6.99 (-0.06) 33.33 (-1.73) 74.33 (-1.36)

w/o Velocity KL 0.87 87.32 (-0.43) 7.02 (-0.03) 32.47 (-2.59) 75.07 (-0.62) w/o Reward-wise Norm 0.86 (-0.01) 87.73 (-0.02) 7.02 (-0.03) 32.18 (-2.88) 75.27 (-0.42)

- Table 7 further shows a significant drop in text generation score (32.18 vs. 35.06), suggesting that high-variance rewards can dominate the policy updates and impede progress on specific objectives when normalization is not applied per reward.

### 6 Conclusion

In this work, we present DeepGen 1.0, a lightweight yet powerful unified multimodal model that seamlessly integrates image generation and editing within a compact 5B parameter framework. By synergizing a deep VLM-DiT alignment architecture with a progressive, data-centric training strategy, we demonstrate that comprehensive omni-capabilities, spanning generation, reasoning, and editing, can be achieved without relying on massive parameter scaling or excessive computational resources. Extensive evaluations highlight that DeepGen 1.0 not only outperforms existing open-source models of similar size but also rivals substantially larger systems (e.g., 80B parameters), particularly in reasoning-intensive and instruction-following tasks.

Beyond technical contributions, DeepGen 1.0 offers broader implications for sustainable AI. By decoupling high-quality generation from massive computational resources, it paves the way for accessible research on consumer-grade hardware. By open-sourcing DeepGen 1.0, we hope it serves as a foundational step toward democratizing unified multimodal intelligence and inspiring new efficient architectures.

### References

- [1] OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/introducing-4o-image-generation/. Accessed: 2025.
- [2] Google. Introducing Gemini 2.5 Flash Image, our state-of-the-art image model. https://developers. googleblog.com/introducing-gemini-2-5-flash-image/, August 2025.
- [3] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv: 2505.14683, 2025.

- [4] Siyu Cao, Hangting Chen, Peng Chen, Yĳi Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

- [5] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv: 2508.02324, 2025.

- [6] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.

- [7] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv: 2505.09568, 2025.

- [8] Jinheng Xie, Weĳia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhĳie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [9] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751, 2025.

- [10] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shĳie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv: 2502.13923, 2025.

- [11] Hongyang Wei, Baixin Xu, Hongbo Liu, Size Wu, Jie Liu, Yi Peng, Peiyu Wang, Zexiang Liu, Jingwen He, Yidan Xietian, et al. Skywork unipic 2.0: Building kontext model with online rl for unified multimodal model. arXiv preprint arXiv:2509.04548, 2025.

- [12] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [13] Tao Shen, Xin Wan, Taicai Chen, Rui Zhang, Junwen Pan, Dawei Lu, Fanding Lei, Zhilin Lu, Yunfei Yang, Chen Cheng, et al. Mammothmoda2: A unified ar-diffusion framework for multimodal understanding and generation. arXiv preprint arXiv:2511.18262, 2025.

- [14] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, YuChiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026.

- [15] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

- [16] Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025.

- [17] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. corr, abs/2506.03147, 2025. doi: 10. 48550. arXiv preprint ARXIV.2506.03147.

- [18] Kevin Li, Manuel Brack, Sudeep Katakol, Hareesh Ravi, and Ajinkya Kale. Unifusion: Vision-language model as unified encoder in image generation. arXiv preprint arXiv:2510.12789, 2025.

- [19] Zeyu Wang, Zilong Chen, Chenhui Gou, Feng Li, Chaorui Deng, Deyao Zhu, Kunchang Li, Weihao Yu, Haoqin Tu, Haoqi Fan, et al. Lightfusion: A light-weighted, double fusion framework for unified multimodal understanding and generation. arXiv preprint arXiv:2510.22946, 2025.

- [20] Weĳia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Lmfusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024.

- [21] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

- [22] Siyuan Wang, Dianyi Wang, Chengxing Zhou, Zejun Li, Zhihao Fan, Xuan-Jing Huang, and Zhongyu Wei. Activating distributed visual region within llms for efficient and effective vision-language training and inference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 30715–30727, 2025.

- [23] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv: 2503.07265, 2025.

- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

- [25] Ouxiang Li, Yuan Wang, Xinting Hu, Huĳuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, Xiaojuan Qi, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv: 2509.03516, 2025.

- [26] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv: 2511.21631, 2025.

- [27] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.

- [28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [29] Feng Wang and Zihao Yu. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

- [30] Jacky He and contributors. text-to-image-2M: A high-quality, diverse text–image training dataset. https:// huggingface.co/datasets/jackyhate/text-to-image-2M, 2024. Hugging Face dataset.
- [31] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

- [32] Ollin Matsubara and AI Draw Things. Team. megalith-10m: A dataset of 10 million public-domain photographs.
- [33] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431, 2021.

- [34] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3558–3568, 2021.

- [35] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv: 2506.18095, 2025.

- [36] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025.

- [37] Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, et al. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv:2509.24900, 2025.

- [38] Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, and Aleksandr Gordeev. Nohumansrequired: Autonomous high-quality image editing triplet mining. arXiv preprint arXiv:2507.14119, 2025.

- [39] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025.

- [40] Nano-banana-150k. https://github.com/yejy53/Nano-banana-150k, 2024. GitHub repository.
- [41] Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv:2510.19808, 2025.

- [42] Dianyi Wang, Chaofan Ma, Feng Han, Size Wu, Wei Song, Yibin Wang, Zhixiong Zhang, Tianhang Wang, Siyuan Wang, Zhongyu Wei, et al. Unireason 1.0: A unified reasoning framework for world knowledge aligned image generation and editing. arXiv preprint arXiv:2602.02437, 2026.

- [43] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, Hua Yang, Xue Yang, and Haodong Duan. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv: 2504.02826, 2025.

- [44] Feng Han, Yibin Wang, Chenglin Li, Zheming Liang, Dianyi Wang, Yang Jiao, Zhipeng Wei, Chao Gong, Cheng Jin, Jingjing Chen, et al. Unireditbench: A unified reasoning-based image editing benchmark. arXiv preprint arXiv:2511.01295, 2025.

- [45] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

- [46] Google. Gemini 2.5 pro. https://deepmind.google/models/gemini/pro/, 2025.
- [47] Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025.

- [48] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

- [49] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

- [50] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

- [51] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- [52] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, Ming-Hsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv: 2505.16707, 2025.

- [53] Nikai Du, Zhennan Chen, Shan Gao, Zhizhou Chen, Xi Chen, Zhengkai Jiang, Jian Yang, and Ying Tai. Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461, 2025.

- [54] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

- [55] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [56] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shĳie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

- [57] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [58] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025.

- [59] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

- [60] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025.

- [61] Z.ai Team. Glm-image: Auto-regressive for dense-knowledge and high-fidelity image generation, jan 2026. URL https://z.ai/blog/glm-image.
- [62] Huichao Zhang, Liao Qu, Yiheng Liu, Hang Chen, Yangyang Song, Yongsheng Dong, Shikun Sun, Xian Li, Xu Wang, Yi Jiang, et al. Nextflow: Unified sequential modeling activates multimodal understanding and generation. arXiv preprint arXiv:2601.02204, 2026.

- [63] UNIFIED MULTIMODAL LEARNING. Star: Stacked autoregressive scheme for unified multimodal learning.
- [64] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Zhiheng Xi, Changhao Jiang, Zhangyue Yin, Yining Zheng, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, and Xuanjing Huang. Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning. arXiv preprint arXiv: 2505.13886, 2025.

- [65] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

- [66] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

- [67] Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Unified multimodal chain-of-thought reward model through reinforcement fine-tuning. arXiv preprint arXiv:2505.03318, 2025.

- [68] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.

- [69] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

## Appendix

### A Pre-Training & SFT Details

Table 8 and 9 provide the details of dataset usage and hyperparameter configurations at each stage, respectively.

Table 8 The data details used in Pre-Training and Supervised Fine-Tuning stages. "†" denotes covering both Chinese and English prompts.

Stage Task Data source Size Pre-Training

|General Generation|text-to-image-2M [30], LAION-Aesthetic-6M [31], Megalith-10M [32], RedCaps-5M [33], CC-12M [34]<br><br>|35M|
|---|---|---|
|General Editing<br><br>|NHR-Edit [38], GPT-Image-Edit [39], ShareGPT-4o-Image-Edit [35], OpenGPT4o-Image-Edit [37], Nano-banana-consist [40], Pico-banana [41], X2I2 [12], UniWorld-Edit set [17], in-house editing data†<br><br>|6.6M|

|General Generation<br><br>|BLIP-3o [7], ShareGPT-4o-Image [35], Echo-4o-Image [36], OpenGPT4o-Image [37], Self-Banana-50K, in-house generation data†|11M|
|---|---|---|
|General Editing|NHR-Edit [38], GPT-Image-Edit [39], ShareGPT-4o-Image-Edit [35], OpenGPT4o-Image-Edit [37], Nano-banana-consist [40], Pico-banana [41], X2I2 [12], UniWorld-Edit set [17], in-house editing data†<br><br>|6.6M|
|Reasoning Generation<br><br>|UniReason-T2I set [42]|150K|
|Reasoning Editing|UniReason-Edit set [42]|100K|
|Text Rendering<br><br>|General text rendering, poster design†, Chinese poem|560K|

Supervised Fine-Tuning

Table 9 Detailed Hyperparameters and Configurations of the Pre-Training and Supervised Fine-Tuning.

Hyperparameters Stage-I (Pre-Training) Stage-II (Supervised Fine-Tuning)

Learning Rate 1.0 × 10−4 5.0 × 10−5 LR Scheduler Cosine Cosine Weight Decay 0.05 0.05 Gradient Norm Clip 1.0 1.0 Optimizer AdamW AdamW warmup ratio 0.01 0.01 Batch Size 512 768 Training GPUs 64×H200 64×H200

|Gen. Resolution Arbitrary Resolution<br><br>|512 x|512<br><br>✓|
|---|---|---|
|Trainable Param LoRA Rank LoRA 𝛼 LoRA Dropout|SCB connector -<br><br>|SCB connector, DiT, LoRA in VLM 64 128 0.05<br><br>|

### B Reinforcement Learning Details

Noise-Preserving Stochastic Sampling. When sampling trajectories, the deterministic flow-matching ODE 𝑑𝑥𝑡 = 𝑣ˆ𝜃(𝑥𝑡, 𝑡) 𝑑𝑡 is unsuitable for the exploration required by reinforcement learning. Prior works [65, 66] convert it into a stochastic differential equation (SDE) to introduce randomness. However, the standard Flow-SDE formulation injects noise that exceeds the scheduler’s expected noise level at each timestep, degrading sample quality and producing inaccurate reward signals. We instead adopt a noise-preserving

stochastic sampling strategy [29] that ensures the noise level remains consistent with the flow matching scheduler at every timestep:

𝑥𝑡−Δ𝑡 = 1 − (𝑡−Δ𝑡) 𝑥 ˆ0 + (𝑡−Δ𝑡)cos 𝜂𝜋2 𝑥 ˆ1 + (𝑡−Δ𝑡)sin 𝜂𝜋2 𝜖, (6)

where 𝑥ˆ0 = 𝑥𝑡 − 𝑡 𝑣ˆ𝜃 and 𝑥ˆ1 = 𝑥𝑡 + (1−𝑡) 𝑣ˆ𝜃 are the predicted clean sample and noise respectively, 𝜖 ∼ 𝒩(0, 𝐼) is freshly sampled Gaussian noise, and 𝜂 ∈ [0, 1] controls the stochasticity strength. The log-probability for computing importance ratios is simplified as [29]:

log 𝑝𝜃(𝑥𝑡−Δ𝑡 | 𝑥𝑡) = −∥𝑥𝑡−Δ𝑡 − 𝜇𝜃(𝑥𝑡, 𝑡)∥2, (7)

where 𝜇𝜃(𝑥𝑡, 𝑡) = (1−(𝑡−Δ𝑡)) 𝑥ˆ0 + (𝑡−Δ𝑡)cos 𝜂𝜋2 𝑥 ˆ1 is the deterministic component of the sampling step. This formulation removes the variance normalization term present in the standard log-probability, avoiding

numerical instability at small noise levels.

Reward Functions. We employ three reward functions to provide complementary training signals. (1) A VLM-based pairwise preference reward [27] from our Unified-Reward-Think [67] that evaluates image-text alignment and visual quality by comparing all generated images within each group and computing per-sample win rates as reward scores. (2) An OCR reward [68] that measures text rendering accuracy by detecting rendered text in the generated image and comparing it against the target text specified in the prompt. (3) A CLIP similarity score [69] that captures overall semantic consistency between the generated image and the text condition. Each prompt category is assigned a different reward composition: text-rendering prompts are weighted toward the OCR reward, while general text-to-image prompts prioritize the preference reward. The detailed reward weights are provided in Table 11.

Training Details. The RL training prompts are drawn from two categories: general text-to-image prompts and text-rendering prompts. The auxiliary SFT data is sampled from an independent curated corpus of high-quality image-text pairs covering both general generation and text rendering. Dataset details are provided below. We train with a group size of 𝐺 = 8, generating images at 512 × 512 resolution using 50 denoising steps. The model is optimized with a learning rate of 2 × 10−6 for 1,500 steps. The complete set of hyperparameters is listed in Table 10.

Hyperparameters. Table 10 summarizes the full set of hyperparameters used for RL training.

Table 10 Hyperparameters for reinforcement learning training.

Hyperparameter Value Group size 𝐺 8 Image resolution 512 × 512 Denoising steps 50 SDE stochasticity 𝜂 1.0 Timestep fraction 0.6 Learning rate 2 × 10−6 Total training steps 1,500 KL coefficient 𝛽 5 × 10−7 Clip range 𝜖 1 × 10−4 SFT auxiliary coefficient 𝜆 1 × 10−4 SFT auxiliary frequency Every step Global batch size 256 DeepSpeed stage ZeRO-2 Precision BF16

Reward Weights. Table 11 shows the per-category reward weight configuration. Text-rendering prompts are weighted toward the OCR reward to directly optimize text accuracy, while general text-to-image prompts rely primarily on the VLM-based preference reward for holistic quality assessment.

Table 11 Reward weight configuration by prompt category.

Prompt Category Preference CLIP Sim OCR Text rendering 0.2 0.1 0.7 General T2I 0.7 0.3 –

RL Training Prompts. The RL training prompts consist of two categories with proportional sampling. Text-rendering prompts (sample weight 3.0×) are drawn from UniGenBench text data, Qwen-Image text rendering captions, and curated text rendering prompts. General text-to-image prompts (sample weight 1.0×) are sourced from UniGenBench general data, BLIP3-o captions, ShareGPT-4o image descriptions, and CoREBench prompts.

Auxiliary SFT Data. The auxiliary supervised data for computing ℒSFT is drawn from an independent corpus of high-quality image-text pairs. This corpus includes general text-to-image pairs (from BLIP3-o, ShareGPT-4o, Echo-4o, OpenGPT-4o, GenEval, and Self-Banana-50K collections) with sample weight 1.0×, and text rendering pairs with sample weight 3.0× to match the emphasis on text rendering in the RL prompts.

