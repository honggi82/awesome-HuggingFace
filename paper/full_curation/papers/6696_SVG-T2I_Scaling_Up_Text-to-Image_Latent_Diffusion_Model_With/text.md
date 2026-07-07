# arXiv:2512.11749v1[cs.CV]12Dec2025

## SVG-T2I: SCALING UP TEXT-TO-IMAGE LATENT DIFFUSION MODEL WITHOUT VARIATIONAL AUTOENCODER

Minglei Shi1∗ Haolin Wang1∗ Borui Zhang1 Wenzhao Zheng1 Bohan Zeng2 Ziyang Yuan2† Xiaoshi Wu2 Yuanxing Zhang2 Huan Yang2 Xintao Wang2 Pengfei Wan2 Kun Gai2 Jie Zhou1 Jiwen Lu1†

1Department of Automation, Tsinghua University 2Kling Team, Kuaishou Technology Code Repository: https://github.com/KlingTeam/SVG-T2I Model Weights: https://huggingface.co/KlingTeam/SVG-T2I

[Figure 1]

[Figure 2]

ABSTRACT

Visual generation grounded in Visual Foundation Model (VFM) representations offers a highly promising unified pathway for integrating visual understanding, perception, and generation. Despite this potential, training large-scale text-to-image diffusion models entirely within the VFM representation space remains largely unexplored. To bridge this gap, we scale the SVG (Self-supervised representations for Visual Generation) framework, proposing SVG-T2I to support high-quality text-to-image synthesis directly in the VFM feature domain. By leveraging a standard text-to-image diffusion pipeline, SVG-T2I achieves competitive performance, reaching 0.75 on GenEval and 85.78 on DPG-Bench. This performance validates the intrinsic representational power of VFMs for generative tasks. We fully opensource the project, including the autoencoder and generation model, together with their training, inference, evaluation pipelines, and pre-trained weights, to facilitate further research in representation-driven visual generation.

1 INTRODUCTION

Generative modeling, particularly text-to-image synthesis (Rombach et al., 2021; Esser et al., 2024; Labs, 2024), has advanced rapidly in recent years, with diffusion models (Rombach et al., 2021; Ho et al., 2020; Song et al., 2021b; Lipman et al., 2023; Esser et al., 2024) becoming the prevailing solution. To accelerate VAE (Kingma & Welling, 2022)-based latent diffusion models (LDM) (Rombach et al., 2021), existing works (Yu et al., 2025; Leng et al., 2025; Yao et al., 2025; Kouzelis et al., 2025b; Wu et al., 2025) incorporate pretrained Visual Foundation Model (VFM) features through feature alignment or joint generation.

SVG (Shi et al., 2025) and RAE (Zheng et al., 2025) take a further step by training diffusion models directly in the high-dimensional VFM feature space, achieving improved generation quality as well as higher efficiency during both training and inference. In addition to performance gains, this paradigm opens the possibility of unifying the encoder infrastructure across tasks, leveraging a single encoder in place of traditional choices such as SigLIP (Zhai et al., 2023; Tschannen et al., 2025) for understanding, VAE (Kingma & Welling, 2022) for generation, and VGGT (Wang et al., 2025) for geometry perception and reasoning.

However, two key challenges remain unresolved:

- 1. Can a unified feature space support visual reconstruction, perception, high-fidelity generation, and semantic understanding without compromising performance on any of these tasks?
- 2. Are VFM-derived representations inherently compatible with large-scale, high-resolution text-toimage diffusion training, which is essential for real-world applications?

∗Equal contribution. †Corresponding author

[Figure 3]

#### Figure 1: Showcases of SVG-T2I with diverse text prompts. Examples are generated at 720×1280 (left), 1080×1080 (middle), and 1440×720 (right) resolution. All corresponding prompts are listed in Appendix B.

Addressing the first challenge is essential for realizing a native, unified general vision model. A principled representation space would eliminate the need for fragmented task-specific encoders, enabling a single architecture to jointly support low-level perceptual fidelity, accurate reconstruction, and high-level semantic understanding and generation. Encouraging progress highlights the feasibility of this goal. Large-scale self-supervised learning (Kouzelis et al., 2025c) has demonstrated that expanding model capacity and training data yields strong perceptual robustness together with competitive performance in understanding tasks. Moreover, frameworks such as SVG (Shi et al., 2025), UniFlow (Yue et al., 2025), and UniLiP (Tang et al., 2025) show that VFM representations can be lightly adapted to support precise reconstruction and synthesis while retaining their discriminative semantics. These advances collectively suggest that a unified representation space capable of aligning perception, reconstruction, understanding, and generation is an achievable next step for the field.

Regarding the second challenge, existing approaches (Shi et al., 2025; Zheng et al., 2025; Yue et al., 2025) have already demonstrated strong performance on class-conditioned ImageNet (Deng et al., 2009) generation. However, large-scale and systematic investigations on text-to-image generation remain largely absent. Given that ImageNet has a limited scale and category diversity, and most vision encoders are already pre-trained on it, such results cannot validate the generalization ability of these methods. As a result, there remain open questions regarding their feasibility and performance potential in realistic text-to-image scenarios.

In this paper, we mainly focus on the second challenge and conduct the first large-scale study on training a text-to-image diffusion model directly within the VFM feature space.

We summarize the primary contributions of this paper as follows:

- • We provide the first large-scale validation of the text-to-image latent diffusion model on the VFM feature space.
- • We open-source the complete training and inference pipeline of the SVG-T2I model, together with pre-trained weights of multiple sizes, to facilitate future research.

2 RELATED WORKS

Visual Generative Models. Visual generative models aim to approximate the underlying data distribution of real-world visual content and synthesize new samples drawn from it. Over their development, several major paradigms have emerged. Generative adversarial networks (GANs) (Goodfellow et al., 2014; Arjovsky et al., 2017; Gulrajani et al., 2017; Karras et al., 2019; Zhu et al., 2017; Radford et al., 2016; Karras et al., 2018; Sauer et al., 2022) generate high-fidelity images via adversarial training but often suffer unstable training, model collapse, and poor interpretability. Autoregressive models (Salimans et al., 2017; Vaswani et al., 2018; Chen et al., 2020; Sun et al., 2024; Tian et al., 2024) frame generation as a sequential prediction task, achieving strong distribution modeling and stability through likelihood maximization, yet they incur high inference latency due to their token-by-token nature. Masked generative models (Chang et al., 2022; Zheng et al., 2024; Yu et al., 2024; Li et al.,

- 2024a) predict missing tokens given visible context, analogous to masked language models in NLP, also achieving strong performance. Recently, Diffusion models (Ho et al., 2020; Nichol & Dhariwal, 2021; Song et al., 2021a;b) have been established as the leading paradigm by learning to invert a progressive noising process. They offer robust optimization objectives, superior mode coverage. An enhanced approach, the latent diffusion model (LDM) (Rombach et al., 2021; Peebles & Xie, 2022; Ma et al., 2024; Liu et al., 2022), incorporates a VAE (Kingma & Welling, 2022) into the diffusion pipeline to operate within a lower-dimensional latent space, thereby reducing computational cost while preserving generation quality. However, the latent space of VAEs lacks coherent semantic structure, making it suboptimal for downstream understanding and perception tasks.

Representation for Generative Modeling. Recent works have increasingly focused on improving the representation of generative modeling. One direction enhances the VAE latent space through structural regularization (Burgess et al., 2018; Xu & Durrett, 2018; Kouzelis et al., 2025a; Skorokhodov et al.,

- 2025) or diffusion-guided reconstruction (Preechakul et al., 2022; Pandey et al., 2021; Vahdat et al., 2021). Another line aligns generative latents with external semantic embeddings (Yu et al., 2025; Leng et al., 2025; Yao et al., 2025; Li et al., 2023) or incorporates discriminative knowledge in generation (Li et al., 2023; Wu et al., 2025; Kouzelis et al., 2025b). More recently, SVG (Shi et al., 2025) and RAE (Zheng et al., 2025) demonstrate strong performance by training diffusion

[Figure 4]

- Figure 2: Architecture of the proposed SVG-T2I. (Left) Single-Stream DiT architecture. (Right) We optionally augment the DINO encoder with a Residual Encoder to achieve high-quality reconstruction and preserve transferability.

models directly in the VFM feature space. Concurrent works such as UniLiP (Tang et al., 2025) and UniFlow (Yue et al., 2025) demonstrate that self-distilled VFMs, when coupled with a pixel decoder and trained under MAR (Li et al., 2024a) or MetaQueries-style paradigms (Pan et al., 2025), can effectively support both high-quality reconstruction and generation.

- 3 METHODOLOGY

- 3.1 PRELIMINARIES

Diffusion models. Diffusion Models (Ho et al., 2020; Rombach et al., 2021; Song et al., 2021b) have been the dominant generative modeling for continuous feature space, which can transform the Gaussian distribution to the data distribution through iterative inference. In this paper, we adopt the widely used flow matching method (Liu et al., 2022; Lipman et al., 2023; Esser et al., 2024) for training SVG-T2I. Flow matching constructs a velocity field that interpolates between a Gaussian distribution and the data distribution:

xt = (1 − t)x0 + tϵ, t ∈ [0,1], ϵ ∼ N(0,I), (1) The flow matching objective is then formulated as

LFM = Ex

0∼p0(x),ϵ∼p1(x)[λ(t)∥vθ(xt,t) − (ϵ − x0)∥]. (2) Sampling from a flow-based model can be achieved by solving the probability flow ODE.

- 3.2 SELF-SUPERVISED REPRESENTATIONS FOR VISUAL GENERATION

SVG (Shi et al., 2025) demonstrated the feasibility of achieving high-quality image reconstruction and class-to-image generation within high-dimensional VFM feature spaces. Building on this foundation, SVG-T2I extends the approach to large-scale text-to-image training, enabling effective generation directly in the VFM feature domain. The overall architecture of SVG-T2I is shown in Figure 2.

SVG-T2I Autoencoder. Inheriting the architectural design from SVG (Shi et al., 2025) and RAE (Zheng et al., 2025), we release two autoencoder configurations to facilitate community research. The first, autoencoder-P (Pure), utilizes frozen DINOv3 features directly. The second, autoencoder-R (Residual), retains the residual branch design from SVG as an optional choice. This residual module, built on a Vision Transformer (Dosovitskiy et al., 2021), is designed to compensate for high-frequency details and color cast artifacts when higher fidelity is required. Both variants utilize the same decoder design to map features back to pixel space.

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

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Ground Truth 1024 Recon. 1024 Feature 512 Recon. 512 Feature 256 Recon. 256 Feature 896 Feature 448 Feature 224 Feature

- Figure 3: Visualization of SVG reconstruction. For each reconstructed image, we also present the PCA visualization of its DINOv3 features, and DINOv2 features are shown in the right three columns. The numbers indicate the resolution of the input images.

SVG-T2I DiT. We use the Unified Next-DiT (Qin et al., 2025) architecture as our backbone, which treats text and image tokens as a joint sequence, enabling natural cross-modal interactions and allowing seamless task expansion. The Unified Next-DiT architecture is a scalable single-stream variant similar to that used in the state-of-the-art open-source VAE-based text-to-image model ZImage (Team et al., 2025). We adopt this single-stream design to achieve greater parameter efficiency and to jointly process text and DINO features. We train the backbone directly on high-dimensional VFM(DINOv3) feature space, using the flow matching objective defined in Equation (2). In our framework, we use the DINOv3-ViT-S/16+ encoder, which maps an H × W × 3 image to a (H/16) × (W/16) × 384 feature representation.

SVG-T2I Training Pipeline. Training proceeds in two stages. In the first stage, we train autoencoderP, autoencoder-R separately from scratch. Specifically, autoencoder-R is optimized with both reconstruction losses and distribution-matching strategy on its residual branch and decoder following (Shi et al., 2025). In the second stage, we train SVG-T2I DiT equipped with autoencoder-P, following a progressive schedule (see training details).

- 3.3 SCALING SVG TO HIGHER RESOLUTION

SVG (Shi et al., 2025) and RAE (Zheng et al., 2025) primarily focused on learning generative diffusion models within VFM representation spaces under low-resolution settings. In this work, we extend this line of inquiry by examining the behavior and effectiveness of SVG for high-resolution generation.

We observe distinct resolution-dependent behaviors when reconstructing images from DINOv3 features, as illustrated in Figure 3. While reconstructions from low-resolution inputs suffer from degradation in fine structures, high-resolution inputs yield substantially more detailed and faithful results. This indicates that DINOv3 representations inherently preserve detailed visual cues effectively at higher resolutions. Crucially, this capability suggests that the DINOv3 encoder alone is relatively sufficient for high-resolution reconstruction, obviating the need for an auxiliary residual encoder. Furthermore, relying exclusively on VFM representations offers a more generalized and reusable paradigm compared to hybrid architectures. Motivated by both the representation sufficiency and the desire for a streamlined, versatile framework, we configure the residual encoder in the original SVG Autoencoder as optional, omitting it during high-resolution reconstruction or generation.

1024 512 256 896 448 224 1024 512 256

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

0.90

0.88

1.00

0.60

0.70

0.99

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

0.96

0.90

1.00

0.68

0.79

0.96

- Figure 4: Comparison of VAE and DINO features. We visualize the PCA projections of features extracted by DINOv3 (Left), DINOv2 (Middle), and the VAE (Right). Cosine similarity across different resolutions is computed by downsampling higher-resolution features to match the spatial size of the lower ones. Overall, VAE features exhibit limited semantic structure, yet demonstrate stronger scale invariance compared to DINO features.

Table 1: Overview of the datasets used across different training stages. Cap. Sample Ratio denotes the probability of sampling short, middle, and long captions during training for each dataset, which controls the caption-length distribution seen by the model.

Data Type Number Description Caption Length Cap. Sample Ratio Reconstruction

- A 1.2M ImageNet General Data - -
- B 3M High-quality Realistic Data - -

Generation

- C 60M High-quality General Data Short, Middle, Long (0.10, 0.35, 0.55)
- D 15M High-quality Realistic Data Short, Middle, Long (0.10, 0.35, 0.55)
- E 1M High-aesthetic Data Short, Middle, Long (0.00, 0.00, 1.00)

- 4 EXPERIMENTS

In this section, we describe the training recipe of SVG-T2I, then validate the feasibility and effectiveness of the proposed SVG-T2I through extensive experiments.

- 4.1 MODEL TRAINING

### Training Details of SVG-T2I Autoencoder.

The autoencoder is trained with a progressive strategy. We first pre-train the model on ImageNet (Data A) for 40 epochs at a fixed resolution of 256×256. Then, during the multi-resolution fine-tuning stage, we continue training using native-resolution images from a 3M-sample dataset (Data B). In this phase, the model is trained at an anchor resolution of 512×512 for 10M seen images, followed by 1024×1024 for an additional 6M seen images.

### Training Details of SVG-T2I DiT.

We adopt the Unified Next-DiT architecture from Lumina-Image-2.0 (Qin et al., 2025) as the backbone of our diffusion transformer. For text conditioning, we utilize the Gemma2-2B large language model, which possesses strong multilingual capabilities, to extract rich textual embeddings. We set the maximum text token length to 256 to balance long-caption modeling capability and training efficiency at first three stages. In high quality data tuning state, the maximum text token length is set to 512. Each image in Data (C, D, E) is annotated with bilingual captions (Chinese and English) in three lengths: short, middle, and long. During training, we adopt a mixed sampling strategy that selects both the caption language and its length. The sampling probabilities for short, middle, and long captions are provided in Table 1, and the language sampling ratio is fixed to 0.2 for Chinese and 0.8 for English.

We train SVG-T2I which is equipped with autoencoder-P using a multi-stage progressive training strategy. In the first two stage, the model is trained at low resolution and middle resolution on 60M

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

##### STAGE1STAGE3STAGE4STAGE2

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

- Figure 5: Comparison of SVG-T2I four-stage results. The generated images become progressively more detailed and aesthetically refined as the stages advance.

samples (Data C) to establish robust text–image alignment and capture low-frequency structures. In the third stage, we transfer the learned knowledge to higher resolutions, enabling the model to refine fine-grained visual details using 15M samples (Data D). In the final stage, SVG-T2I is fine-tuned on 1M high-quality aesthetic samples (Data E) to further enhance its ability to synthesize realistic and visually appealing outputs. As shown in Figure Figure 5, the visual quality improves steadily across stages.

- 4.2 MAIN RESULTS

Evaluation. We evaluated SVG-T2I through both quantitative and qualitative metrics. We report performance on GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024) to evaluate allaround capabilities of SVG-T2I following their official protocol. All images used for evaluation were generated at a high resolution of 1024 × 1024. Our SVG-T2I model successfully scales the VFM representation paradigm for large-scale T2I generation, achieving competitive performance across these two benchmarks. On GenEval Ghosh et al. (2023) (Table 5), our final model, SVG-T2I, attains an overall score of 0.74, matching the performance of models like SD3-Medium (Esser et al., 2024) and significantly surpassing SDXL (Podell et al., 2023) and DALL-E 2 (Betker et al., 2023). Furthermore, on DPG-Bench (Table 6), SVG-T2I achieves an overall score of 85.78, placing it statistically comparable to top VAE-based diffusion models such as FLUX.1 (Labs, 2024) and HiDream-I1-Full (Cai et al., 2025).

- Table 2: Model configuration of the proposed SVG-T2I DiT architecture.

Model Params Patch Size Dimension Head KV Heads Layers Pos. Emb. SVG-T2I 2.6B 1 2304 24 8 26 M-RoPE

- Table 3: Training configuration of SVG-T2I Decoder across different stages.

Stage Anchor Resolution #Images Training Steps (K) Batch Size #Seen Samples

Fixed Low Res. 2562 1.2M 94K 512 48M Multi Middle Res. 5122 3M 78K 128 10M Multi High Res. 10242 3M 190K 32 6M

- 4.3 ANALYSIS

Limitations of current VFM features. Existing self-supervised learning methods produce representations that capture both high-level semantic context and fine-grained visual detail, offering

Table 4: Training configuration of SVG-T2I DiT across different stages.

Stage Anchor Resolution #Images Training Steps (K) Batch Size #Seen Samples

Multi Low Res. 2562 60M 91K 1536 140M Multi Middle Res. 5122 60M 90K 768 70M Multi High Res. 10242 15M 44K 768 34M HQ Tuning. 10242 1M 40K 768 30M

Table 5: Evaluation of text-to-image generation ability on GenEval (Ghosh et al., 2023) benchmark. † refer to the methods using LLM rewriter.

Model Single Obj. Two Obj. Counting Colors Position Color Attri. Overall↑

Discrete Generation (Autoregressive)

Chameleon (Team, 2025) - - - - - - 0.39 Emu3-Gen† Wang et al. (2024) 0.99 0.81 0.42 0.80 0.49 0.45 0.66 Show-o Xie et al. (2025) 0.98 0.80 0.66 0.84 0.31 0.50 0.68 Janus-Pro-7B (Chen et al., 2025) 0.99 0.89 0.59 0.90 0.79 0.66 0.80

VAE-based Generation (Diffusion)

PixArt-α (Chen et al., 2023a) 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SDv2.1 (Rombach et al., 2021) 0.98 0.51 0.44 0.85 0.07 0.17 0.50

- DALL-E 2 (Betker et al., 2023) 0.94 0.66 0.49 0.77 0.10 0.19 0.52 SDXL (Podell et al., 2023) 0.98 0.74 0.39 0.85 0.15 0.23 0.55
- DALL-E 3 (Betker et al., 2023) 0.96 0.87 0.47 0.83 0.43 0.45 0.67 Lumina-Image-2.0† (Qin et al., 2025) - 0.87 0.67 - - 0.62 0.73 SD3-Medium (Esser et al., 2024) 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1-dev† (Labs, 2024) 0.98 0.93 0.75 0.93 0.68 0.65 0.82

Representation Generation (Diffusion)

SVG-T2I† 0.94 0.89 0.49 0.89 0.69 0.62 0.75

a strong basis for downstream reconstruction and generation. In principle, these representations are largely self-sufficient. However, this self-sufficiency is critically challenged when the training paradigm involves multiple input resolutions. As represented in Figure 4, VAE features exhibit nearly resolution-invariant behavior. Their cross-resolution consine similarity is close to 1.0, whereas DINOv3 and DINOv2 features vary more substantially. This observation indicates that VFM-derived features undergo non-negligible shifts across scales.

When a VFM encoder uses a fixed patch or receptive-field size (e.g., 16×16) across inputs of different absolute resolution, the semantic granularity and effective compression ratio of each patch vary systematically with scale: a patch on a low-resolution image aggregates a much larger portion of the scene, producing strongly compressed, detail-poor features; the identical patch size on a highresolution image captures finer, predominantly local texture and structural detail. Because VFM encoders are typically optimized to produce semantically discriminative tokens rather than to preserve uniform local detail, they are particularly sensitive to this scale-dependent shift in the semantic/texture balance. By contrast, reconstruction-oriented encoders (e.g., VAEs) do not explicitly account for the semantic content present in each encoded region; instead, they primarily aim to capture sufficient local information for pixel-level reconstruction, leading to a more uniform and resolution-stable allocation of representation capacity.

Accordingly, for semantic visual encoders used for diffusion modeling, maintaining stable crossresolution behavior emerges as an important optimization goal. The training pipeline may need to incorporate mechanisms that encourage consistent feature geometry and help preserve the fidelity of fine-grained details across scales.

Limitations of SVG-T2I. While SVG-T2I demonstrates strong generation capability across diverse scenarios, several limitations remain. As shown in Figure 6, the model occasionally struggles to produce highly detailed human faces, particularly in regions requiring fine-grained spatial consistency, such as eyes, eyebrows. Similarly, the generation of anatomically accurate fingers continues to be challenging, a common failure mode in generative models, often resulting in distorted shapes or incorrect topology when pose complexity increases. SVG-T2I also exhibits limited reliability in text rendering. These shortcomings largely stem from the insufficient coverage of such fine-grained

Table 6: Evaluation of text-to-image generation ability on DPG (Hu et al., 2024) benchmark.

Model Global Entity Attribute Relation Other Overall ↑ Discrete Generation (Autoregressive)

Janus (Wu et al., 2024) 82.33 87.38 87.70 85.46 86.41 79.68 Emu3-Gen (Wang et al., 2024) 85.21 86.68 86.84 90.22 83.15 80.60 Janus-Pro-1B (Chen et al., 2025) 87.58 88.63 88.17 88.98 88.30 82.63 Janus-Pro-7B (Chen et al., 2025) 86.90 88.90 89.40 89.32 89.48 84.19

VAE-based Generation (Diffusion)

SD1.5 (Rombach et al., 2021) 74.63 74.23 75.39 73.49 67.81 63.18 PixArt-α (Chen et al., 2023b) 74.97 79.32 78.60 82.57 76.96 71.11 Lumina-Next (Zhuo et al., 2024) 82.82 88.65 86.44 80.53 81.82 74.63 SDXL (Podell et al., 2023) 83.27 82.43 80.91 86.76 80.41 74.65 Hunyuan-DiT (Li et al., 2024b) 84.59 80.59 88.01 74.36 86.41 78.87 PixArt-Σ (Chen et al., 2024) 86.89 82.89 88.94 86.59 87.68 80.54 DALL-E 3 (Betker et al., 2023) 90.97 89.61 88.39 90.58 89.83 83.50 FLUX.1 [Dev] (Labs, 2024) 74.35 90.00 88.96 90.87 88.33 83.84 SD3 Medium (Esser et al., 2024) 87.90 91.01 88.83 80.70 88.68 84.08 HiDream-I1-Full (Cai et al., 2025) 76.44 90.22 89.48 93.74 91.83 85.89 Lumina-Image 2.0 (Qin et al., 2025) - 91.97 90.20 94.85 - 87.20

Representation Generation (Diffusion)

SVG-T2I 88.50 91.00 91.86 92.21 91.86 85.78

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

- Figure 6: Failure cases of SVG-T2I. These examples illustrate that SVG-T2I may struggle with generating highly detailed human faces, accurate finger structures, and reliable text rendering, which typically require more specialized data and larger training compute.

cases in the training corpus, as well as the substantial computational demand required to model high-frequency patterns and precise geometric relationships. Addressing these limitations will likely require more specialized datasets and additional training compute.

- 5 CONCLUSION

In this work, we successfully extended the original SVG framework to large-scale, high-resolution text-to-image synthesis, culminating in the SVG-T2I model. Our work validates the feasibility of training a high-quality T2I model from scratch based on VFM representations, achieving generative metrics comparable to modern advanced methods and demonstrating the potential of the VFM semantic space as an effective latent manifold for high-resolution synthesis. To foster further research and ensure reproducibility, we have fully open-sourced the training, inference, and evaluation code, along with the model weights, hoping to benefit the academic community. However, in the course of this research, we also identified a critical challenge: existing VFM encoders (such as

DINOv2 (Oquab et al., 2023) and DINOv3 (Sim´eoni et al., 2025)) produce representations with poor internal consistency when encoding the same image at different input resolutions. This resolutiondependent feature instability directly compromises the T2I model’s ability to generalize across various sizes and maintain generation quality, underscoring the necessity for future research to focus on scale-invariance. Overall, we believe that the strategic use and refinement of a powerful VFM latent space, as demonstrated in this work, present a highly promising avenue toward achieving a truly unified representation for diverse visual tasks.

REFERENCES

Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein gan, 2017.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang‡, Linjie Li‡, Long Ouyang†, Juntang Zhuang†, Joyce Lee†, Yufei Guo†, Wesam Manassra†, Prafulla Dhariwal†, Casey Chu†, and Yunxin Jiao†. Improving image generation with better captions, 2023. Preprint. Available at OpenAI Papers.

Christopher P. Burgess, I. Higgins, Arka Pal, L. Matthey, Nicholas Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in beta-vae. ArXiv, abs/1804.03599, 2018.

Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In CVPR, June 2022.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023a.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023b.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.

Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, and Ilya Sutskever. Generative pretraining from pixels. 2020.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment, 2023.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), NeurIPS, volume 27. Curran Associates, Inc., 2014.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), NeurIPS, volume 33, pp. 6840–6851. Curran Associates, Inc., 2020.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment, 2024.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In ICLR, 2018.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative

adversarial networks, 2019. Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. EQ-VAE:

Equivariance regularized latent space for improved generative image modeling. In ICML, 2025a.

Theodoros Kouzelis, Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Boosting generative image modeling via joint image-feature synthesis. In NeurIPS, 2025b.

Theodoros Kouzelis, Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Ko-

modakis. Boosting generative image modeling via joint image-feature synthesis. 2025c. Black Forest Labs. Flux: A powerful tool for text generation, 2024. Accessed: 2024-09-26. Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng.

Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.

Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. arXiv:2312.03701, 2023.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization, 2024a.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding, 2024b.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers, 2024.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, pp. 8162–8171. PMLR, 2021.

Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Kushagra Pandey, Avideep Mukherjee, Piyush Rai, and Abhishek Kumar. VAEs meet diffusion models: Efficient and high-fidelity generation. In NeurIPS, 2021.

William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.

Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In CVPR, 2022.

Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Manyuan Zhang, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Conghui He, Bo Zhang, Xiaohong Liu, Hongsheng Li, Yu Qiao, Chang Xu, and Peng Gao. Lumina-image 2.0: A unified and efficient image generative framework, 2025.

Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks, 2016.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models, 2021.

Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. Pixelcnn++: A pixelcnn implementation with discretized logistic mixture likelihood and other modifications. In ICLR, 2017.

Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pp. 1–10, 2022.

Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder, 2025.

Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025.

Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. In ICML, 2025.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. ICLR, 2021a.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021b.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2025.

Image Team, Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, Zhen Li, Zhong-Yu Li, David Liu, Dongyang Liu, Junhan Shi, Qilong Wu, Feng Yu, Chi Zhang, Shifeng Zhang, and Shilin Zhou. Z-image: An efficient image generation foundation model with single-stream diffusion transformer, 2025.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. 2024.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier H´enaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features, 2025.

Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In NeurIPS, 2021.

Ashish Vaswani, Niki Parmar, Jakob Uszkoreit, Noam Shazeer, and Lukasz Kaiser. Image transformer, 2018.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer, 2025.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need, 2024.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. Janus: Decoupling visual encoding for unified multimodal understanding and generation, 2024.

Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation, 2025.

Jiacheng Xu and Greg Durrett. Spherical latent spaces for stable variational autoencoders. In EMNLP, 2018.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In CVPR, 2025.

Lijun Yu, Jos´e Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion – tokenizer is key to visual generation, 2024.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025.

Zhengrong Yue, Haiyu Zhang, Xiangyu Zeng, Boyu Chen, Chenting Wang, Shaobin Zhuang, Lu Dong, KunPeng Du, Yi Wang, Limin Wang, and Yali Wang. Uniflow: A unified pixel flow tokenizer for visual understanding and generation, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training, 2023.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders, 2025.

Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. In TMLR, 2024.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networkss. In ICCV, 2017.

Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, Xu Luo, Zehan Wang, Kaipeng Zhang, Xiangyang Zhu, Si Liu, Xiangyu Yue, Dingning Liu, Wanli Ouyang, Ziwei Liu, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-next: Making lumina-t2x stronger and faster with next-dit, 2024.

Table 7: Hyperparameter setup of SVG-T2I Autoencoder.

Autoencoder-P Autoencoder-R

Encoder Base model DINOv3-s16p DINOv3-s16p Downsample Ratio 16 × 16 16 × 16 Latent dim. 384 392 Residual branch / ViT-S-RoPE Training mode Frozen Frozen Params 29M 51M

Decoder Channels dim. [512, 256, 256, 128, 128] Out channels 3 3 Z channels 384 392 Params 43M 43M

Optimization Optimizer Adam Adam

- lr 1e-4 1e-4 (β1,β2) (0.5, 0.9) (0.5, 0.9)

Table 8: Hyperparameter setup of SVG-T2I DiT.

SVG-T2I-L Lumina-Image-2.0

Architecture Downsample Ratio 16 × 16 8 × 8 Latent dim. 384 16 Num. layers 26 26 Hidden dim. 2304 2304 Num. heads 24 24 Params 2.6B 2.6B Base-encoder Autoencoder-P Flux-VAE

Optimization Optimizer AdamW AdamW

- lr 2e-4 2e-4 (β1,β2) (0.9, 0.95) (0.9, 0.95)

Interpolants αt 1 − t 1 − t σt t t Training objective v-prediction v-prediction Sampler Euler Euler

- A MORE QUALITATIVE RESULTS

We provide additional qualitative results of SVG-T2I with 1080 × 1080 resolution. These results further demonstrate the diversity and visual quality of the proposed approach.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Realistic, high-resolution landscape photography. Long exposure technique used to capture the movement of clouds and the clarity of stars. The image emphasizes natural textures and atmospheric effects., A large, conical sandstone formation rises from a barren, undulating desert landscape under a night sky with streaking clouds and visible stars., The subject is a prominent, conical sandstone butte with a rugged, weathered surface and layered striations in shades of tan, brown, and reddish hues. The formation has a sharp peak at the top and a slightly eroded base, with visible rock outcrops and ridges. The surface texture is rough, with shallow gullies and scattered small rocks. The butte occupies the central portion of the image, with its base extending toward the foreground., The scene is an outdoor desert landscape at night. The foreground consists of dry, cracked earth and low, rounded mounds in muted brown and tan tones, with sparse patches of small shrubs. The background features additional eroded hills and mesas stretching into the distance, with a dark horizon line. The sky above is deep blue-black, scattered with numerous small, bright stars. Streaks of white and gray clouds move horizontally across the sky, creating a

The subject is a young East Asian woman with fair skin and a slender, proportionate build. Her long, wavy, ash-blonde hair falls over her shoulders and down her chest. She wears a decorative headpiece adorned with silver, gold, and blue gemstones and metallic leaf motifs. Her gown is a light lavender color with sheer long sleeves and intricate floral embroidery in silver and white, featuring a fitted bodice and a voluminous tulle skirt. Her posture is upright, with both hands gently placed on her lap, fingers relaxed. Her body faces slightly to the left of the frame, and her head is oriented forward., The scene is an indoor setting with a soft, dreamy atmosphere. The background consists of white curtains with a subtle lace pattern, creating a delicate and airy backdrop. Green leafy vines are draped along the upper left and right edges of the background, adding a natural element. The foreground is clear, showing the subject's gown and hands in detail. The background is softly blurred, with the greenery and curtain textures providing gentle contrast., Centered composition; the subject occupies the central portion of the frame and approximately 70% of the image. Medium close-up shot, straight-on angle at eye level. The camera is positioned at a medium distance, using a standard focal length lens with a wide aperture, resulting in a shallow depth of field that keeps the subject in sharp focus while softly blurring the background., Realistic, high-resolution photography with a soft, romantic, and slightly dreamy aesthetic. The image emphasizes delicate textures and fine details, suitable for fashion or portrait editorial use., A young woman with long wavy hair sits indoors, wearing a light lavender gown and an ornate jeweled headpiece, with her hands resting on her lap., Realistic, Soft, diffused lighting with a cool to neutral color temperature. The light source appears to be natural or simulated daylight, coming from the front and slightly above, evenly illuminating the subject and minimizing harsh shadows. The lighting creates a gentle, ethereal effect on the gown and hair.

Realistic, high-resolution studio photography with a dramatic and commercial fitness aesthetic., A muscular adult man stands shirtless in a dark indoor setting, facing forward with his arms relaxed at his sides., The subject is an adult Caucasian man with a highly muscular and defined physique. He has short, dark brown hair. His skin tone is light with a warm undertone, and his body is covered in prominent muscle definition, especially in the chest, shoulders, arms, and abdomen. He is wearing loose, dark athletic shorts with a drawstring. His posture is upright, with both arms relaxed and slightly bent at the elbows, hands resting near his thighs. His body faces directly forward, and his head is oriented straight ahead., The scene is an indoor studio with a dark, nearly black background. The foreground is occupied entirely by the subject, with the background fading into darkness. The lighting creates a subtle gradient from the subject outward, with the edges of the subject softly blending into the background. The overall atmosphere is dramatic and focused., Realistic, Artificial lighting, cool to neutral color temperature. The main light source is positioned behind and slightly above the subject, creating a rim light effect that highlights the contours of the muscles. The lighting is hard, producing strong shadows and accentuating the subject’s physique. The background remains in deep shadow, emphasizing the subject., Centered composition; the subject occupies the majority of the frame, approximately 70% of the image area. Medium close-up shot, straight-on angle at chest height. Standard focal length lens, wide aperture, shallow depth of field, with the subject in sharp focus and the background blurred.

Realis c, high-resolu on food photography with a focus on texture and color. The image emphasizes freshness and appe zing detail, suitable for commercial or editorial use., A rectangular raspberry cake topped with fresh mint leaves sits on a piece of parchment paper, with three ripe raspberries placed in front of it on a textured surface., The subject is a rectangular cake with a golden-brown crust and a topping of ﬁnely chopped red raspberries. The cake is garnished with a sprig of fresh mint with bright green, textured leaves. In front of the cake, three whole raspberries are arranged, each with a round, bumpy surface and a vibrant red color. The cake has a moist, slightly crumbly texture, and the raspberries appear fresh and plump., The scene is set indoors on a gray, stone-textured surface. The foreground features the three raspberries and the lower edge of the cake, both in sharp focus. The background is a so ly blurred, light beige wall, crea ng a neutral backdrop that contrasts with the vivid colors of the cake and fruit. The parchment paper under the cake is visible at the base, adding a rus c touch., Realis c, Natural light, so  and diﬀused, coming from the le  side of the image. The ligh ng is warm, highligh ng the textures of the cake and raspberries and crea ng gentle shadows on the right side of the objects., The composi on is centered, with the cake occupying the right side of the frame and the raspberries in the lower le  foreground. The shot is a close-up, taken from a slightly elevated front angle. The camera is posi oned at a low height, close to the surface. A standard lens with a wide aperture is used, resul ng in a shallow depth of ﬁeld that keeps the cake and raspberries sharp while so ly blurring the background.

The style is digital abstract art, characterized by smooth gradients, so  edges, and a painterly, semi-realis c rendering of forms. The image quality is high, with a polished, ad-quality ﬁnish. The technique emphasizes ﬂuidity and organic movement, reminiscent of modern digital pain ng or genera ve art. The overall eﬀect is contemporary and visually immersive, suitable for use as a background, wallpaper, or conceptual illustra on., Abstract digital artwork featuring ﬂowing, wave-like forms in a surreal landscape, with vibrant gradients and so  transi ons., Non-realis c, The ligh ng is so  and diﬀused, with a mixed color temperature that combines warm orange and cool teal tones. The light appears to emanate from mul ple direc ons, gently illumina ng the curves and edges of the forms. Subtle gradients and highlights create a luminous, glowing eﬀect, while the shadows are so  and blended, contribu ng to the overall dreamy atmosphere., The composi on is horizontally oriented and symmetrical, with the main wave-like forms centrally placed and extending across the width of the image. The forms overlap and intertwine, crea ng a layered eﬀect. The perspec ve is ﬂat, with a shallow depth of ﬁeld implied by the so  transi ons between colors and shapes. The image appears to be created from a straight-on, eye-level viewpoint.

sense of motion. The foreground is in sharp focus, while the distant background is slightly blurred., Realistic, Natural light, cool color temperature. Soft, diffused moonlight illuminates the landscape from above and slightly to the left, casting

gentle shadows and highlighting the textures of the sandstone. The sky is darker, with subtle gradients and visible starlight., Centered composition; the sandstone butte is positioned centrally and occupies about 60% of the frame. Wide landscape shot, straight-on angle at eye-level height. Wide-angle lens, small aperture, deep depth of field. The horizon line is placed in the lower third of the frame, emphasizing the sky and clouds.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Realistic, high-resolution landscape photography, A clear mountain stream flows through a rocky alpine valley, with snow patches and rugged mountains rising on both sides under a cloudy sky., The subject is a shallow, clear mountain stream with smooth, rounded rocks of various sizes visible beneath the water. The rocks are mostly gray and brown, with some patches of green moss. The water is transparent, revealing the stones beneath. The stream flows gently from the foreground toward the background, with the rocks becoming smaller and more distant as they recede., The scene is an outdoor alpine valley during daytime in late spring or early summer. The foreground features the stream with visible stones and patches of green moss. The midground consists of grassy and rocky terrain on both sides of the stream, with snow patches lingering on the slopes. The background shows steep, rugged mountain slopes with exposed rock and more snow, partially covered by clouds. The sky is mostly cloudy with some blue patches visible in the upper right. The foreground is in sharp focus, while the background mountains are slightly softened by atmospheric haze., Realistic, Natural daylight, soft and diffused due to cloud cover. The light is cool in tone, with even illumination across the scene. The lighting is primarily top-down, with subtle highlights on the rocks and water surface., The composition uses a leading line created by the stream, drawing the viewer's eye from the bottom center foreground toward the distant mountains. Wide-angle shot, low camera height near the water surface, straight-on angle. The stream occupies the lower half of the frame, with the mountains and sky filling the upper half. Deep depth of field, wide focal length, small aperture.

Realistic, photography, documentary style, An elderly Indian man with a long white beard and turban stands outdoors, wearing a red jacket and a white scarf draped around his neck., The subject is an elderly Indian man with a tall, slender build and medium brown skin tone. He has a long, full white beard that extends below his chin. His hair is covered by a large, neatly wrapped ochre-yellow turban. He is dressed in a bright red jacket with a visible black zipper and a white scarf with orange accents draped around his neck. His posture is upright, with shoulders relaxed and arms at his sides. His body is oriented slightly to the left of the frame, and his head is facing forward., The scene is outdoors during daytime. The background consists of a wire mesh fence with a grid pattern, stretching horizontally across the image. Behind the fence, there is dense green foliage with various shades of green and some yellowish leaves, indicating a natural setting. The background is softly blurred, while the subject is in sharp focus. The foreground is clear, showing the subject's upper body and clothing., Realistic, Natural daylight, soft and diffused, with even illumination across the subject. The light source appears to be ambient, likely from an overcast sky or shaded area, resulting in minimal shadows and a neutral color temperature., Centered composition; the subject occupies the central portion of the frame and fills about 60% of the image. Medium close-up shot, straight-on angle at eye level. Standard focal length, moderate aperture, shallow depth of field with the background blurred.

Realis c, photography, high-resolu on outdoor pet portrait, A blue-gray Weimaraner dog sits on a grassy path in a wooded outdoor area, looking directly at the camera., The subject is an adult Weimaraner dog with a lean, muscular build and short, smooth blue-gray fur. The dog has long, ﬂoppy ears, a broad head, and light blue eyes. Its face is wedge-shaped with a black nose and dark lips. The dog wears a bright red collar around its neck. It is si ng upright with its front legs straight and hind legs bent, tail res ng on the ground. The dog's body faces forward, and its gaze is directed straight at the camera, with an alert and calm expression., The scene is outdoors in a wooded park or forest during day me. The foreground consists of a grassy path with patches of dry grass and sca ered fallen leaves, in clear focus. The background features tall, leaﬂess trees with dark trunks and branches, and a blurred view of more trees and a hint of a road or path in the distance. The background colors are muted browns, grays, and greens, with a so  blur that contrasts with the sharpness of the dog in the foreground., Realis c, Natural daylight, so  and diﬀused, likely from an overcast sky. The light is even, with gentle shadows under the dog and so  highlights on the fur. The overall color temperature is cool., Centered composi on; the dog is posi oned in the middle of the frame and occupies about 60% of the image height. Full-body shot, eye-level angle, straight-on view. Medium focal length lens, wide aperture, shallow depth of ﬁeld with the subject in sharp focus and the background blurred.

The subject is a group of dried chickpeas (Cicer arietinum), a legume species. The chickpeas are small, round, and beige in color, with a slightly rough, matte texture. The pile is dense in the center and gradually thins out toward the edges. A few chickpeas are separated from the main cluster and scattered around the perimeter. The chickpeas are uncooked and have a uniform appearance., Soft, natural light with a neutral color temperature. The light source appears to come from above and slightly to the left, creating gentle shadows around the chickpeas and highlighting their texture. The lighting is even and diffused., Realistic, A large pile of dried chickpeas is spread out on a wooden surface, with some individual chickpeas scattered around the main cluster., Centered composition; the pile of chickpeas occupies the central area and covers most of the frame, approximately 80% of the image. The shot is a top-down view, closeup, with the camera positioned directly above the subject. Standard focal length, small aperture, deep depth of field, all elements in focus., Realistic, highresolution photography, The scene is set indoors on a wooden tabletop with a visible wood grain pattern in brown tones. The foreground consists of the main pile of chickpeas, which is in sharp focus. The background is the wooden surface extending beyond the pile, with the wood grain continuing and becoming slightly less distinct toward the edges. The overall color palette is dominated by beige from the chickpeas and brown from the wood.

写实⻛格，摄影，画质细腻，⾊彩饱和度⾼，细节丰富，整体呈现健康⽣ 活主题。, ⼀碗⾊彩丰富的⽔果沙拉置于麻布上，周围摆放着⾹蕉、猕猴桃、 苹果、葡萄和草莓等新鲜⽔果，画⾯充满⾃然与健康的⽣活⽓息。, 主体是 ⼀碗混合⽔果沙拉，包含切⽚⾹蕉、草莓、猕猴桃、葡萄、芒果和苹果等 多种⽔果，⾊彩鲜艳，切块⼤⼩均匀，⽔果表⾯新鲜有光泽，整体排列饱 满，盛放在浅⾊陶瓷碗中，碗⼝圆润，碗⾝光滑。, 场景为室内，主体位于 画⾯中央偏下⽅，前景为麻布和⽊质桌⾯，细节清晰可⻅，左下⽅有两颗 红⾊草莓和⼀把浅⾊⽊勺，右上⽅有⼀颗红⻩渐变⾊苹果，背景为⼀串紫 红⾊葡萄和⼀个完整猕猴桃，左上⽅有三根⻩⾊⾹蕉，背景物体略微虚化， 整体⾊调温暖，氛围⾃然清新。, ⾃然光，暖光，软光，顺光，光线从画⾯ 左上⽅照射，整体为⾼调光，画⾯明亮，⽔果表⾯有柔和⾼光反射。

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Low-angle, wide shot with the camera positioned close to the ground, capturing the flowers from a slightly upward perspective. The composition is horizontal, with the field of flowers occupying the lower two-thirds of the frame and the mountains and sky in the upper third. Wide-angle lens, small aperture, deep depth of field in the foreground, with gradual blur toward the background., Realistic, photography, high-resolution landscape style, Realistic, The scene is outdoors in a natural setting during daytime. The foreground is filled with cosmos flowers and green foliage, all in sharp focus. The background features a range of blue-green mountains with a soft, misty appearance, and a cloudy sky with patches of blue. Power lines and utility poles are visible on the left side, slightly blurred. The overall color palette includes vibrant orange and yellow in the foreground, deep green in the foliage, and cool blue and gray tones in the mountains and sky., Natural daylight, soft and diffused due to cloud cover. The light is even, with minimal shadows, creating a gentle and balanced illumination across the flowers and landscape., The subject is a large group of Cosmos sulphureus flowers in full bloom. The flowers have bright orange and yellow petals with yellow centers, and slender green stems. The petals are thin and slightly ruffled, with some flowers facing upward and others angled sideways. The green leaves are feathery and elongated, growing densely among the stems. The flowers vary in height, with some closer to the camera appearing larger and more detailed, while others in the background are smaller and slightly blurred., A dense field of blooming orange and yellow cosmos flowers stretches across the foreground, with green stems and leaves, set against a backdrop of distant bluegreen mountains and a cloudy sky.

Realis c, high-resolu on landscape photography, A vast expanse of eroded, mul colored badlands stretches across the landscape, with rugged ridges and valleys under a partly cloudy sky., The subject is a series of sedimentary badlands forma ons composed of layered, undula ng hills and ridges. The terrain displays a range of colors, including yellow, tan, ochre, and brown, with some darker streaks and patches. The surface is rough and textured, with sharp ridges and deep gullies. The forma ons occupy the majority of the frame, with the highest ridges in the foreground and midground, and the terrain gradually receding into the distance., The scene is an outdoor desert landscape during day me. In the foreground, the terrain is sharply detailed, showing ridges and slopes with yellow and brown hues. The midground features more pronounced valleys and ridges, with some areas displaying darker brown and reddish tones. The background consists of a range of rugged, rocky mountains in shades of purple, blue, and brown, which are slightly so ened by atmospheric haze. The sky above is mostly covered with light gray and white clouds, with some blue visible near the horizon. The overall atmosphere is arid and expansive., Realis c, Natural daylight, so  and diﬀused due to cloud cover. The light is even, with minimal harsh shadows, and the color temperature is neutral to slightly warm. The ligh ng highlights the textures and colors of the terrain., The composi on is horizontal and panoramic, with the badlands occupying the lower two-thirds of the frame and the mountains and sky in the upper third. The camera is posi oned at a high vantage point, looking slightly downward across the landscape. Wide-angle lens, deep depth of ﬁeld, with sharp focus on the foreground and gradual so ening toward the background.

The subject is a king-sized bed with a cream-colored, tu ed headboard and a plush, light gray and white pa erned comforter. The bed is made with mul ple pillows, including two large white pillows, two smaller white pillows, and two decora ve pillows with a gray and white abstract pa ern. At the foot of the bed is a rectangular upholstered bench with light gray fabric and white legs. The bed is posi oned centrally in the room, with the bench in the foreground., Realis c, high-resolu on interior photography, commercial style., A modern bedroom features a large bed with a tu ed headboard, surrounded by elegant furniture and so  ligh ng., The scene is an indoor modern bedroom during day me. The foreground includes a light gray area rug with subtle pa erns, covering a sec on of dark brown wooden ﬂooring. To the le  of the bed is a gray upholstered armchair with dark wooden legs and a matching side table holding a small bouquet of white ﬂowers in a glass vase. A tall ﬂoor lamp with a white shade stands beside the armchair. The background features a taupe wall with a framed architectural print in sepia tones, and a sec on of the wall behind the bed is decorated with beige panels featuring geometric pa erns and integrated ligh ng. On either side of the bed are dark wood nightstands, each with a lamp and small decora ve items. Sheer white curtains hang in the background, par ally covering a window. The foreground is in sharp focus, while the background is slightly so er., Realis c, Ar ﬁcial ligh ng dominates, with warm, so  light from the ﬂoor lamp on the le  and table lamps on both nightstands. Addi onal ambient light comes from the ceiling and wall ligh ng behind the headboard. The overall ligh ng is warm and evenly distributed, crea ng a cozy and invi ng atmosphere., The composi on is centered, with the bed occupying the central and right por ons of the frame and the bench in the lower foreground. Wide shot, straight-on angle at eye level. The camera is posi oned at a low height, close to the ﬂoor, emphasizing the bed and bench. Wide-angle lens, small aperture, deep depth of ﬁeld.

Realistic, high-resolution landscape photography. The image emphasizes natural textures and vibrant colors, with a focus on clarity and detail in the water and rocks., A turquoise river flows rapidly over rocks, with a dense green forest lining the riverbank in the background., The subject is a natural river with clear, turquoise water. The water is in motion, creating white, foamy currents and swirling patterns. Several large, irregularly shaped gray rocks with patches of yellow-green moss are partially submerged in the river, with one prominent rock near the center of the image. Smaller rocks and stones are visible along the riverbank in the foreground, some covered with moss. The water appears to be moving swiftly, with visible turbulence and foam around the rocks., The scene is outdoors during daytime, set in a forested river valley. The foreground consists of a rocky riverbank with variously sized gray stones and boulders, some covered in moss, all in sharp focus. The midground features the main river, with swirling turquoise water and visible white foam. The background is densely packed with tall, green coniferous and broadleaf trees, forming a continuous forest wall along the riverbank. The background trees are slightly blurred, while the foreground rocks and water are clear. The overall color palette includes turquoise, gray, green, and white., Realistic, Natural daylight, soft and diffused, with a cool color temperature. The light source appears to be from above and slightly to the left, illuminating the water and rocks evenly. The lighting creates gentle highlights on the water surface and rocks., Horizontal composition with a wide-angle perspective. The river occupies the lower two-thirds of the frame, with rocks in the foreground and the forested riverbank in the upper third. The main rock is positioned slightly left of center. Wide shot, eye-level angle. Deep depth of field in the foreground and midground, with slight background blur. Likely shot with a DSLR camera and wide-angle lens.

Realistic, high-resolution wildlife photography, sharp detail on the subject with a smooth, blurred background., A woodpecker perches on a mossy branch in a natural outdoor setting, facing left with its body upright and gaze directed forward., The subject is an adult woodpecker, likely a male Red-bellied Woodpecker. It has a medium-sized, slender body with a proportionate build. The bird's head features a bright red crown and nape, with a pale gray face and a black beak. Its eyes are dark and round, with a focused gaze directed slightly to the left. The upperparts are black with white spots, while the underparts are pale beige with a hint of orange on the lower belly. The wings are folded against the body, showing black and white patterning. The tail is long and pointed, with black and white markings. The bird's feet grip the branch firmly, with zygodactyl toes visible. The posture is upright, with the body and head oriented to the left, and the tail pointing downward., The scene is outdoors in a forest or woodland area during daytime. The bird is perched on a horizontal branch covered with greenish-gray moss and lichen, which occupies the lower part of the image and is in sharp focus. The background consists of blurred green and yellow foliage, creating a soft, natural bokeh effect. The foreground includes the branch and some out-of-focus moss. The overall color palette is dominated by earthy greens, browns, and muted yellows., Realistic, Natural daylight, soft and diffused, with even illumination across the bird and branch. The light source appears to be ambient, possibly filtered through foliage, resulting in gentle shadows and a neutral color temperature., Centered composition; the woodpecker occupies the central vertical axis and about 60% of the frame. Medium close-up shot, side view, with the camera at eye level to the bird. Telephoto lens, wide aperture, shallow depth of field, isolating the subject from the background.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

写实⻛格，野⽣动物摄影，画质⾼清，细节丰富，⾊彩鲜明，采⽤⼤光圈 突出主体，背景虚化，强调⾃然⽣态美感。, 画⾯采⽤⾃然光，主光为暖⾊ 调阳光，光线柔和，为软光效果，光照⽅向为顺光，基调为⾼调光，主体 明亮，⽔⾯和苔藓反射出⾼光，整体光影层次丰富。

现代简约⻛格，写实摄影，画质⾼清，⾊彩⾃然，材质表现细腻，空间布 局简洁，强调舒适与功能性。, 画⾯整体采⽤⾃然光，主光源为窗外⽇光， ⾊温偏冷，光线柔和，为软光效果。光照⽅向为顺光，基调为⾼调光，画 ⾯明亮，局部有轻微阴影，增强空间层次感。

数字插画⻛格，融合古埃及艺术元素，采⽤平⾯化、装饰性强的绘画技 法，⾊彩饱和度⾼，线条清晰，细节以⼏何和象形图案为主，整体⻛格庄 重神秘，具有强烈的装饰性和象征性。, 画⾯整体为⾼调光，⾊彩明亮，采 ⽤均匀的软光，光源为虚拟环境光，⾊温中性，光影过渡平滑，突出装饰 性与平⾯感。

写实⻛格，摄影艺术，画质细腻，⾊彩还原真实，适合⼴告或美⻝摄影场 景。, 画⾯整体为冷暖对⽐，主光为⼈⼯光，软光，暖光，顺光照射，基调 为⾼调光，主体明亮，背景较暗，葡萄表⾯有⾼光反射。

写实，摄影，画⾯细节丰富，⾊彩⾃然，整体⻛格清新明快，具有⽇本春 季⾃然⻛光摄影的典型特征。, ⾃然光，软光，暖光，顺光，光线来⾃画⾯ 正上⽅，画⾯整体为⾼调光，明亮通透，樱花和富⼠⼭均被均匀照亮。

Figure 7: Visualization of SVG-T2I at 1080×1080 resolution. These examples highlight the model’s ability to produce high-quality, text-aligned visual content with fine-grained details and strong spatial consistency.

- B PROMPTS USED IN FIGURE 1

Here we summarize the prompts/instructions used in Figure 1, which can be directly input into the corresponding model to reproduce our generation results.

- Column #1:

- • Case #1: Realistic, high-resolution photography with a commercial still-life style. The image emphasizes clarity, texture, and color accuracy, suitable for food and beverage advertising or editorial use., Two clear wine glasses filled with white wine are placed on a rustic wooden table, surrounded by green grapes, bread, wine corks, and a bottle of white wine in the background., The main subjects are two transparent wine glasses, each filled with pale yellow-white wine. The glasses are made of clear glass with long stems and round bowls. The wine is clear and slightly golden, with visible reflections on the glass surfaces. The glasses are positioned close together, with the left glass slightly in front of the right. Both glasses are upright and stable on the table. In the foreground, there are clusters of green grapes with smooth, round shapes and translucent skin. Several loose grapes are scattered near the base of the glasses. To the right of the glasses, there is a piece of rustic brown bread with a rough texture and visible crust. In the background, a wire basket contains multiple cylindrical beige wine corks, and a glass bottle of white wine is partially visible., The scene is set indoors on a rustic wooden table with a natural, weathered texture and light brown color. The foreground features green grapes in sharp focus, with some grapes and corks scattered around the base of the glasses. The midground contains the wine glasses, bread, and a wire basket of corks. The background is slightly blurred and includes a glass bottle of white wine and a wooden wall with a warm, neutral tone. The overall color palette consists of natural wood, green, beige, and pale yellow tones, creating a cozy and inviting atmosphere., Realistic, The lighting is natural and soft, with a warm color temperature. The main light source comes from the left side, casting gentle highlights on the glass surfaces and creating soft reflections. The lighting is even, with subtle shadows under the glasses and grapes, contributing to a bright and inviting tone., The composition is centered, with the two wine glasses occupying the central area of the frame and taking up about 50% of the image. The shot is a close-up, taken at a slightly high angle, focusing on the glasses and surrounding elements. The camera is positioned at table height, angled slightly downward. A standard lens with a wide aperture is used, resulting in a shallow depth of field that keeps the foreground and main subjects sharp while softly blurring the background.
- • Case #2: The subject is a young East Asian woman with fair skin and a slender, proportionate build. Her long, wavy, ash-blonde hair falls over her shoulders and down her chest. She wears a decorative headpiece adorned with silver, gold, and blue gemstones and metallic leaf motifs. Her gown is a light lavender color with sheer long sleeves and intricate floral embroidery in silver and white, featuring a fitted bodice and a voluminous tulle skirt. Her posture is upright, with both hands gently placed on her lap, fingers relaxed. Her body faces slightly to the left of the frame, and her head is oriented forward., The scene is an indoor setting with a soft, dreamy atmosphere. The background consists of white curtains with a subtle lace pattern, creating a delicate and airy backdrop. Green leafy vines are draped along the upper left and right edges of the background, adding a natural element. The foreground is clear, showing the subject’s gown and hands in detail. The background is softly blurred, with the greenery and curtain textures providing gentle contrast., Centered composition; the subject occupies the central portion of the frame and approximately 70% of the image. Medium close-up shot, straight-on angle at eye level. The camera is positioned at a medium distance, using a standard focal length lens with a wide aperture, resulting in a shallow depth of field that keeps the subject in sharp focus while softly blurring the background., Realistic, high-resolution photography with a soft, romantic, and slightly dreamy aesthetic. The image emphasizes delicate textures and fine details, suitable for fashion or portrait editorial use., A young woman with long wavy hair sits indoors, wearing a light lavender gown and an ornate jeweled headpiece, with her hands resting on her lap., Realistic, Soft, diffused lighting with a cool to neutral color temperature. The light source appears to be natural or simulated daylight, coming from the front and slightly above, evenly illuminating the subject and

- minimizing harsh shadows. The lighting creates a gentle, ethereal effect on the gown and hair.
- • Case #3: Realistic, high-resolution landscape photography, A snow-covered mountain range rises behind a calm lake, with the peaks and surrounding forest reflected clearly in the water under a bright blue sky., The subject is a group of rugged mountain peaks covered in white snow, with sharp ridges and rocky outcrops. The mountains are flanked by steep, forested slopes with dark green coniferous trees. The lower slopes and the area around the lake are dusted with snow. The reflection of the mountains and trees is visible in the still, pale greenish water of the lake, forming a near-symmetrical image., The scene is an outdoor alpine landscape during daytime in winter. The foreground consists of the shallow, clear lake water with a sandy and slightly muddy bottom, reflecting the mountains and sky. The middle ground features the lake’s edge lined with snow and a dense band of dark green coniferous trees. The background is dominated by the snow-capped mountain range, with rocky faces and patches of snow, set against a cloudless, vivid blue sky. The foreground water is clear and detailed, while the background mountains are sharp and well-defined., Realistic, Natural sunlight, cool color temperature, hard light. The light source is from above and slightly to the left, illuminating the snow and casting subtle shadows on the mountain slopes. The lighting is even and high-key, enhancing the clarity of the snow and the reflection in the water., Symmetrical composition with the mountain range centered in the frame and its reflection forming a vertical axis. Wide landscape shot, straight-on angle at eye level. The mountains occupy the upper half of the frame, while the lake and its reflection fill the lower half. Wide-angle lens, small aperture, deep depth of field.
- • Case #4: photorealistic, The image features three cupcakes adorned with colorful sprinkles and encased in checkered wrappers. The cupcakes are positioned on a pink surface with scattered sprinkles around them. In the background, there is a small green leaf, possibly a mint leaf, adding a hint of freshness to the composition., The foreground of the image showcases three cupcakes. Each cupcake is generously topped with a colorful assortment of sprinkles that include shades of pink, white, red, green, and blue. The cupcakes are wrapped in a checkered brown and black paper, providing a neat and structured appearance. The upper surface of the cupcakes is golden brown, indicating they are well-baked, with a soft, crumbly texture visible. They are compactly arranged, filling the bottom half of the image., The background of the image consists of a smooth pink surface, which enhances the vibrant colors of the sprinkles. Among the scattered sprinkles, there is a small green leaf, likely a mint leaf, offering a contrast to the otherwise pink and colorful visuals. The setting suggests a modern and bright environment, focused on accentuating the cupcakes., solid color, close-up, standard lens, natural light, central composition, frontal view
- • Case #5: realistic photography, frontal view, blurred, central composition, A vibrant red rose is prominently featured in the foreground, with droplets of water on its petals. The background shows green foliage and bushes, likely indicating a garden setting. The image captures the delicacy and color of the rose against a blurred backdrop of greenery., natural light, close-up, macro lens.
- • Case #6: realistic, high-resolution photography, A Bengal cat with green eyes lies on a brown surface, facing the camera with its head slightly tilted., The subject is an adult Bengal cat. It has a muscular build and short, dense fur with a golden-brown base and dark brown rosette and stripe patterns. The cat’s face is broad with a pinkish-brown nose, white muzzle, and long white whiskers. Its ears are upright and pointed, with pinkish inner fur. The cat’s eyes are large, round, and bright green, with vertical slit pupils. Its front legs are tucked under its body, and its head is slightly tilted to the left. The cat’s gaze is directed straight at the camera, and its expression appears calm and alert., The scene is indoors. The foreground consists of a brown fabric surface, likely a couch or cushion, which is in sharp focus. The background is a smooth, dark brown wall, softly blurred. The overall color palette is warm, dominated by shades of brown and gold., Realistic, Natural light, soft and diffused, coming from the front left. The lighting is warm, evenly illuminating the cat’s face and fur, with gentle

shadows under the chin and on the right side of the face., Centered composition; the cat’s face occupies the central area of the frame, filling about 70% of the image. Medium close-up shot, eye-level angle. The camera is positioned close to the subject, using a standard or short telephoto lens. Shallow depth of field, with the cat in sharp focus and the background blurred.

- Column #2:

- • Case #1: Realistic, high-resolution photography with a classic portrait style. The image emphasizes texture and detail in clothing and accessories, with a focus on elegance and sophistication., A woman with curly brown hair is dressed in a fur-collared coat and a black hat with ornate embroidery, standing against a dark, softly lit background., The subject is a woman, likely a young adult, with light skin and voluminous, curly brown hair that frames her head and falls around her ears. She wears a black hat adorned with intricate gold and silver embroidery featuring floral and leaf patterns. Her ears are visible, and she wears dangling, multi-stone earrings. She is dressed in a coat with a thick, grayish fur collar that frames her neck and shoulders. Her posture is upright, with her head and body facing directly forward., The scene is indoors with a dark, softly blurred background. The background features deep brown and reddish hues, creating a warm and subdued atmosphere. The foreground is occupied by the subject’s upper body and clothing, which are in sharp focus. The background is out of focus, providing a sense of depth and isolating the subject., Realistic, Soft, warm lighting, likely artificial, coming from the front and slightly above the subject. The light gently illuminates the subject’s hair, hat, and fur collar, creating subtle highlights and soft shadows. The background remains darker, enhancing the subject’s prominence., Centered composition; the subject occupies the central portion of the frame and fills most of the vertical space. Medium close-up shot, straight-on angle at eye level. The camera is positioned close to the subject, using a standard or short telephoto lens. Shallow depth of field, with the subject in sharp focus and the background blurred.
- • Case #2: Realistic style, photography, finely detailed, high color saturation, rich in details, overall conveying a healthy lifestyle theme. A bowl of colorful fruit salad placed on a burlap cloth, surrounded by fresh fruits such as bananas, kiwis, apples, grapes, and strawberries, the scene full of natural and healthy life vibes. The main subject is a bowl of mixed fruit salad, including sliced bananas, strawberries, kiwis, grapes, mangoes, and apples, with vivid colors, evenly cut pieces, fresh and glossy fruit surfaces, arranged generously, served in a light-colored ceramic bowl with a rounded rim and smooth body. The scene is indoors, with the main subject positioned slightly lower in the center of the frame; the foreground features burlap and a wooden tabletop with clear details. In the lower-left corner, there are two red strawberries and a light wooden spoon; in the upper-right corner, a red-yellow gradient apple; the background contains a bunch of reddish-purple grapes and a whole kiwi, with three yellow bananas in the upper-left corner. Background objects are slightly blurred, the overall color tone is warm, and the atmosphere is natural and fresh. Natural warm soft light, front lighting from the upper-left side, overall high-key lighting, bright scene, with soft highlights reflecting off the fruit surfaces.
- • Case #3: A vibrant watercolor botanical illustration featuring red and yellow tulip flowers, depicted in a close-up pattern style with abstract blue and green backgrounds., The image is a high-quality, semi-realistic botanical illustration created in watercolor. The style combines realistic floral forms with expressive, abstract background elements. The technique features wet-on-wet blending, visible brushstrokes, and splatter effects, characteristic of contemporary botanical art. The overall effect is lively, decorative, and artistic, suitable for use in textiles, wallpapers, or stationery., The main subjects are several tulip flowers, shown in various stages of bloom. The tulips have elongated, slightly curved petals with a mix of red, yellow, and orange hues, blending softly at the edges. The petals are rendered with visible watercolor gradients and subtle textural variations, giving a sense of translucency and freshness. The green stems and leaves are painted with loose, expressive brushstrokes, adding to the lively aesthetic. The flowers are depicted in a natural, upright orientation, with

- some petals overlapping and others angled differently, creating a dynamic arrangement., The lighting is implied through the use of watercolor techniques, with soft, diffused highlights and gentle shading that suggest natural daylight. The color temperature is warm, with the red and yellow petals contrasting against the cooler blue and green background. The overall tone is high key, with bright, luminous colors and minimal shadow, enhancing the fresh and uplifting mood., The environment is an abstract, artistic background composed of overlapping washes of blue, green, and yellow watercolor, with splashes and drips that evoke a sense of spontaneity and movement. The background is mostly clear, with the floral elements standing out against the colorful, painterly backdrop. The overall atmosphere is bright, cheerful, and energetic, reminiscent of a spring or early summer setting., Non-realistic, The composition is a close-up, pattern-like arrangement with a repeating motif. The tulip flowers are distributed diagonally and vertically across the frame, filling most of the image space. The perspective is flat and frontal, with the flowers overlapping and intersecting, creating a sense of depth through layering. The image uses a centered and balanced composition, with the flowers occupying the majority of the visual field. The depth of field is shallow, with all elements rendered in sharp focus due to the illustrative nature.
- • Case #4:

The scene is indoors with a dark, neutral background. The foreground features soft, out-offocus shapes in the lower left and right corners, possibly fabric or shadows, which frame the subject. The background is uniformly dark, providing contrast to the subject’s lighter clothing and hair. The overall atmosphere is subdued and formal., Centered composition; the subject occupies the central portion of the frame, filling approximately 60% of the image. Medium close-up shot, straight-on angle at eye level. The camera is positioned close to the subject. Shallow depth of field, with the foreground and background softly blurred, focusing attention on the subject’s upper body and hair., Realistic, black-and-white photography with a formal portrait style. The image has a high-resolution, professional finish, emphasizing texture and contrast., Soft, diffused lighting with a cool tone. The light source appears to come from the front and slightly above, illuminating the subject’s hair and shirt evenly. The lighting creates gentle shadows and a smooth gradient across the background., Realistic, The subject is a young Caucasian man with fair skin and a proportionate build. His hair is short, dark, and neatly combed to the side, with a smooth texture. He is wearing a white collared shirt and a light-colored tie, suggesting formal attire. His posture is upright, with shoulders squared and body facing forward. The subject’s head is oriented straight ahead., A young Caucasian man with neatly styled hair is pictured in a formal setting, wearing a collared shirt and tie, with his upper body visible against a dark background.

- Column #3:

- • Case #1:

Realistic, high-resolution wildlife photography, An adult Bengal tiger lies on the ground in a grassy outdoor setting, facing the camera with its body partially visible and its gaze directed forward., The subject is an adult Bengal tiger. It has a large, muscular build with orange fur and prominent black stripes running along its body and face. The tiger’s face is broad with a pink nose, white fur around the mouth and chin, and white whiskers. Its ears are rounded with black backs and white inner fur. The tiger’s eyes are yellow-green, and it is gazing directly at the camera. Its body is reclined on the ground, with the front legs extended forward and the head held upright. The tiger’s expression is calm and alert., The scene is outdoors during daytime. The foreground contains a pile of pale beige straw or dried grass, which is in sharp focus. Behind the tiger, there is a patch of green grass and a horizontal log with a rough brown texture. The background consists of more green grass and blurred beige ground, creating a natural habitat atmosphere. The foreground is clear, while the background is softly blurred., Natural daylight, soft and diffused, with even illumination across the tiger’s face and body. The light source appears to be from above and slightly to the front, producing gentle shadows and highlighting the tiger’s fur texture., Centered composition; the tiger’s head and upper body occupy the central area of the frame, filling about 60% of the image. Medium close-up shot, eye-level angle. The camera is positioned at the tiger’s eye height, straight-on. Medium focal length, moderate aperture, shallow depth of field., Realistic.

- • Case #2: The scene is outdoors in a garden or natural setting during daytime. The foreground features the sharply focused cosmos flowers and a few green stems and buds. The background is blurred, displaying additional purple cosmos flowers and green foliage, with circular bokeh highlights in shades of purple, green, and yellow. The overall color palette includes purples, greens, and warm yellow tones, creating a vibrant and lively atmosphere., Realistic, photography, high-resolution macro style with a focus on natural detail and color. The image emphasizes clarity and vibrancy, typical of botanical or nature photography., Realistic, Several blooming purple cosmos flowers are clustered together outdoors, with sunlight illuminating their petals and casting soft shadows., The subjects are cosmos flowers (Cosmos bipinnatus) in full bloom. The flowers have broad, delicate petals in a soft purple hue with subtle gradients and slightly ruffled edges. The centers are bright yellow-orange, surrounded by a ring of small, dark-tipped stamens. The petals are thin and semi-translucent, catching the light. The flowers are at various angles, with the central flower facing forward and slightly upward, while others are angled to the left or right. The stems are slender and green, with some unopened buds visible below the flowers., Centered composition with the main flower group occupying the middle of the frame. Medium close-up shot, straight-on angle at flower height. The main flowers fill about 60% of the frame. Shallow depth of field achieved with a wide aperture, rendering the background and some foreground elements softly blurred. Likely taken with a DSLR camera and a macro or standard lens., Natural sunlight, warm color temperature, soft light quality. The light source comes from the upper left, creating gentle highlights on the petals and subtle shadows on the lower right sides of the flowers. The lighting emphasizes the translucency and texture of the petals.
- • Case #3: The image features a close-up of two glass cups filled with Dalgona coffee. The foreground shows a cup with whipped coffee foam on top of a milk layer. The background includes a blurred cup of the same drink and a stone-like surface with scattered coffee beans., realistic, blurred, close-up, natural light, frontal view, standard lens, central composition.

