# arXiv:2410.08261v4[cs.CV]13Mar2025

MEISSONIC: REVITALIZING MASKED GENERATIVE TRANSFORMERS FOR EFFICIENT HIGH-RESOLUTION TEXT-TO-IMAGE SYNTHESIS

Jinbin Bai1,2∗, Tian Ye3∗, Wei Chow5, Enxin Song5,

Xiangtai Li2, Zhen Dong6, Lei Zhu3,4†, Shuicheng Yan2,1† Model: https://huggingface.co/MeissonFlow/Meissonic Code: https://github.com/viiika/Meissonic

ABSTRACT

We present Meissonic, which elevates non-autoregressive masked image modeling (MIM) text-to-image to a level comparable with state-of-the-art diffusion models like SDXL. By incorporating a comprehensive suite of architectural innovations, advanced positional encoding strategies, and optimized sampling conditions, Meissonic substantially improves MIM’s performance and efficiency. Additionally, we leverage high-quality training data, integrate micro-conditions informed by human preference scores, and employ feature compression layers to further enhance image fidelity and resolution. Our model not only matches but often exceeds the performance of existing models like SDXL in generating highquality, high-resolution images. Extensive experiments validate Meissonic’s capabilities, demonstrating its potential as a new standard in text-to-image synthesis. We release a model checkpoint capable of producing 1024 × 1024 resolution images.

1 INTRODUCTION

Diffusion models, such as Stable Diffusion (Rombach et al., 2022a; Podell et al., 2023; Desync, 2024; Art, 2023), have rapidly advanced to become the dominant paradigm in visual generation by replacing Generative Adversarial Network (GAN). Recent developments like LlamaGen (Sun et al., 2024) have ventured into autoregressive image generation using discrete image tokens derived from VQVAE (Yu et al., 2022a). Despite progress, the substantial number of image tokens compared to text tokens makes autoregressive generation inefficient. For example, tokenizing one 1024 × 1024 image using a 16× downsampled VQVAE yields 4096 tokens, where a sequential generation process is prohibitively slow.

Masked generative transformers, a class of generative models, have achieved significant results in the fields of image generation, Specifically, MaskGIT (Chang et al., 2022) introduced a more efficient, non-autoregressive alternative, where all image tokens are predicted simultaneously in a parallel, iterative refinement process. Then, MUSE (Chang et al., 2023) extended this technique to higher resolutions, achieving 512 × 512 resolution T2I generation. These non-autoregressive methods offer around 99% reduction in decoding steps compared to autoregressive methods. However, despite their efficiency, non-autoregressive transformers remain limited in performance compared to advancing diffusion or autoregressive models, particularly in high-quality, high-resolution text-toimage synthesis.

In this work, we address these challenges and introduce two key innovations to make masked image modeling (MIM) competitive with advanced diffusion models:

∗Equal contribution. : jinbin.bai@u.nus.edu †Corresponding authors. 1National University of Singapore 2Skywork AI 3HKUST(GZ) 4HKUST 5ZJU 6UC Berkeley

[Figure 1]

[Figure 2]

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

#### Figure 1: Images produced by Meissonic exhibit exceptional image quality. More samples can be found in Appendix N. Notably, Meissonic can effortlessly produce images with solid-color backgrounds without requiring any additional modifications.

Enhanced Transformer Architecture: Previous MIM methods (Chang et al., 2023; 2022) predominantly utilized naive transformer architectures, potentially limiting their capabilities. We discovered that a combination of multi-modal and single-modal transformer layers can significantly boost MIM training efficiency and performance. Language and vision representations are inherently different. The multi-modal transformer can effectively capture cross-modal interactions, extracting information from unpooled text representations and effectively bridging the gap between these distinct modalities. This allows the model to harness useful signals from noisy data. Additionally, subsequent single-modal transformer layers refine the visual representation, improving performance and training stability. Empirically, a 1 : 2 ratio between these two types of transformer layers yields optimal performance.

Advanced Positional Encoding & Masking Rate as Sampling Condition: We incorporate Rotary Position Embedding (RoPE) (Su et al., 2024) for encoding positional information in queries and keys, which helps maintain detail in high-resolution images. RoPE effectively addresses the issue of context disassociation in transformers as the number of tokens increases. Traditional absolute positional encoding methods lead to distortions and loss of detail at 512 × 512 resolutions, whereas RoPE significantly mitigates these issues. Additionally, we introduce the masking rate as a dynamic sampling condition throughout the generation process. Previous MIM methods (Chang et al., 2023;

- 2022) have overlooked this aspect, resulting in suboptimal image details. This issue arises because the number of tokens predicted by the MIM model changes dramatically throughout the sampling loop. With the masking rate condition, the model can ascertain the current stage of the sampling period by leveraging conditional information from the masking rate. Note that merely relying on attention masks is insufficient to bridge this gap. We achieve effective conditional encoding by discretizing the continuous masking rate into 1000 levels. This approach enables the model to adapt to different stages of the sampling process, significantly improving image detail and overall quality.

Beyond these architectural improvements, to achieve comparable performance with SDXL for highresolution generation, we adopt effects in three additional aspects:

High-Quality Training Data: The quality of training data is crucial. While LAION (Schuhmann

- et al., 2022) offers a diverse visual dataset, its captions can be subpar (Chen et al., 2024). We curated a high-quality internal dataset with accurate captions, which, combined with our training strategy, significantly improved the generative capabilities of the base model.

Micro-Conditioning: We identified that incorporating original image resolution, crop coordinates, and human preference score (Wu et al., 2023) as micro-conditions greatly enhances model stability during high-resolution aesthetic training.

Feature Compression Layers: To efficiently generate high-resolution images, we integrated feature compression layers, maintaining computational efficiency even at 1024 × 1024 resolution.

Our contributions culminate in Meissonic, a next-generation T2I model based on masked discrete image token modeling. Unlike larger diffusion models such as SDXL (Podell et al., 2024) and DeepFloyd-XL (Liu et al., 2024a), Meissonic, with just 1B parameters, offers comparable or superior 1024×1024 high-resolution, aesthetically pleasing images while being able to run on consumergrade GPUs with only 8GB VRAM without the need for any additional model optimizations. Moreover, Meissonic effortlessly generates images with solid-color backgrounds, a feature that usually demands model fine-tuning or noise offset adjustments in diffusion models.

Advancement of Meissonic represents a significant stride towards high-resolution, efficient, and accessible T2I MIM models. We evaluate Meissonic using various qualitative and quantitative metrics, including HPS, MPS, GenEval benchmarks, and GPT4o assessments, demonstrating its superior performance and efficiency.

2 METHOD

- 2.1 MOTIVATION

Recent breakthroughs in text-to-image synthesis have been largely propelled by diffusion models, such as Stable Diffusion XL, which have set de facto standards for image quality, detail, and conceptual fidelity.

R C C

| | | | |
|---|---|---|---|
| | | | |

y

Text Enc

The Canadian astronaut lands on the moon.

c

Cross Entropy Loss

Single-Modal Transformer Block

Multi-Modal Transformer Block

Multi-Modal Transformer Block

Single-Modal Transformer Block

[Figure 19]

[Figure 20]

.Decomp

VQ Dec

VQ Enc

Comp.

x

Mask

Multi-modal Transformer For MIM

× # steps

- Figure 2: The architecture of Meissonic. During the image generation process, discrete tokens are created randomly according to a predefined schedule. Meissonic then applies masking and performs predictions over several steps to reconstruct all tokens and decode the resulting image. In the case of image editing, the original image is converted into discrete tokens, which are masked according to a specified masking strategy. After a series of processing steps, the masked tokens are reconstructed and utilized to decode the target image. Text prompt and other conditions are incorporated to control the synthesis process. R represents the masking rate condition, and C represents the micro conditions. Comp. and Decomp. denotes feature compression layers and feaure decompression layers, respectively. More details about Multi-modal Transformer Block can be found in Appendix I.

Another approach, non-autoregressive Masked Image Modeling (MIM) techniques, exemplified by MaskGIT and MUSE, has shown potential for efficient image generation to replace slow autoregressive techniques like Llamagen. Yet, despite their promise, MIM approaches face two critical limitations:

- (a) Resolution Constraint. Current MIM methods are limited to generating images at a maximum resolution of 512 × 512 pixels. This limitation hinders their broader adoption and advancement, particularly as the text-to-image synthesis community increasingly adopts 1024 × 1024 resolution as the standard.
- (b) Performance Gap. Existing MIM techniques have not yet achieved the level of performance exhibited by leading diffusion models like SDXL. They notably underperform in key areas such

- as image quality, intricate detailing, and conceptual representation, which are critical for practical applications.

These challenges necessitate the exploration of new approaches. Our objective is to empower MIM to efficiently generate high-resolution images (e.g., 1024×1024), while narrowing the gap with toptier diffusion models, and ensuring computational efficiency suitable for consumer-grade hardware.

Through our work, Meissonic, we aim to push the boundaries of MIM methods and bring them to the forefront of text-to-image synthesis.

- 2.2 MODEL ARCHITECTURE

The Meissonic model is architected to facilitate efficient high-performance text-to-image synthesis through an integrated framework comprising a CLIP text encoder (Radford et al., 2021), a vectorquantized (VQ) image encoder and decoder (Esser et al., 2021a), and a multi-modal Transformer backbone. Figure 2 illustrates the overall structure of the model.

Vector-quantized Image Encoder and Decoder. We employ a VQ-VAE model (Esser et al., 2021a) to convert raw image pixels into discrete semantic tokens. This model comprises an encoder, a decoder, and a quantization layer that maps input images into sequences of discrete tokens using a

learned codebook. For an image of size H×W, the encoded token size is Hf ×Wf , where f represents the downsampling ratio. In our implementation, we utilize a downsampling ratio of f = 16 and a codebook size of 8192, allowing a 1024 × 1024 image to be encoded into a sequence of 64 × 64 discrete tokens.

Flexible and Efficient Text Encoder. Instead of using large language model encoders, such as T5XXL1 (Raffel et al., 2020) or LLaMa (Touvron et al., 2023), which are prevalent in previous works (Chen et al., 2024; Esser et al., 2024), we utilize a single text encoder from the state-of-the-art CLIP model with a latent dimension of 1024, and fine-tune for optimal T2I performance. While this decision may limit the model’s capacity to fully comprehend lengthy text prompts, our observations indicate that excluding large-scale text encoders like T5 does not diminish visual quality. Moreover, this approach significantly reduces GPU memory requirements and computational cost. Notably, offline extraction of T5 features would entail approximately 11 times more processing time and 6 times more storage than employing the CLIP text encoder, underscoring the efficiency of our design.

Multi-modal Transformer Backbone for Masked Image Modeling. Our transformer architecture builds upon the Multi-modal Transformer framework (Sauer et al., 2024), incorporating sampling parameters r to encode sampling parameters and Rotary Position Embeddings (RoPE) (Su et al., 2024) for spatial information encoding. We introduce feature compression layers to efficiently handle high-resolution generation with numerous discrete tokens. These layers compress embedding features from 64×64 to 32×32 before processing through the transformer, and followed by feature decompression layers to 64×64, thereby alleviating computational burdens. To enhance training stability and mitigate the NaN Loss issue, we follow the training strategy from LLaMa (Touvron et al.,

- 2023), implementing gradient clipping and checkpoint reloading during distributed training and integrating QK-Norm layers into the architecture. We elaborate on the designs of our transformer in the subsequent section.

Diverse Micro Conditions. To augment generation performance, we incorporate additional conditions such as original image resolution, crop coordinates, and human preference score (Wu et al.,

- 2023). These conditions are transformed into sinusoidal embeddings and concatenated as additional channels to the final pooled hidden states of the text encoder.

Masking Strategy. Following the approach established in Chang et al. (2023), we employ a variable masking ratio with cosine scheduling. Specifically, we randomly sample a masking ratio r ∈ [0,1] from a truncated arccos distribution characterized by the following density function:

2 π

- 1

- 2

(1 − r2)−

p(r) =

In contrast to autoregressive models that learn conditional distributions P(xi | x<i) for fixed token orders, our approach utilizes random masking with variable ratios to enable the model to learn

P(xi | xΛ) for arbitrary subsets of tokens Λ. This flexibility is pivotal for our parallel sampling strategy and facilitates various zero-shot image editing capabilities, which will be demonstrated in Section 3.

- 2.3 MULTI-MODAL TRANSFORMER FOR MASKED IMAGE MODELING

Meissonic employs the Multi-modal Transformer as its foundational architecture and innovatively customizes the modules to address the distinctive challenges inherent in high-resolution masked image modeling. We introduce several specialized designs for MIM as follows:

- • Rotary Position Embeddings. RoPE (Su et al., 2024) has demonstrated exceptional performance within in LLMs (Su et al., 2024; Touvron et al., 2023; Ding et al., 2024; Bai et al., 2023). Some studies (Lu et al., 2024; Lin et al., 2023; Zhuo et al., 2024) have attempted to extend 1D RoPE (Su et al., 2024) to 2D or 3D for image diffusion models. Our findings reveal that, due to the high-quality image tokenizer used for converting images into discrete tokens, the original 1D RoPE yields promising results. This 1D RoPE facilitates a seamless transition from the 256 × 256 stage to the 512 × 512 stage, simultaneously enhancing the generative performance of the model.
- • Deeper Model with Single-modal Transformer. Although the Multi-modal Transformer block demonstrated commendable performance, our experiments reveal that reducing the number of multi-modal blocks to a single-modal block configuration offers a more stable and computationally efficient approach for training T2I models. Therefore, we opt to employ Multi-modal Transformer blocks in the initial stages of the network, transitioning to

1Many works indicate that the T5 text encoder is the key factor in obtaining the ability to synthesize words, we still show the ability to synthesize letters in Figure 8. We leave this a future improvement.

exclusively Single-modal Transformer blocks in the latter half. Our findings suggest an optimal block ratio of about 1:2.

- • Micro Conditions with Human Preference Score. Our experiments reveal that incorporating three micro-conditions is pivotal for achieving a stable and reliable High-resolution MIM Model: original image resolution, crop coordinates, and human preference score. The original image resolution effectively aids the model in implicitly filtering out low-quality data and learning the properties of high-quality, high-resolution data, while crop coordinates enhance training stability, likely due to improved consistency between image conditions and semantic conditions during cropped patch coordination. In the final stage, we leverage the Human Preference Score (Wu et al., 2023) to effectively enhance image quality, using signals provided by the Human Preference Model to guide the model’s outputs in mimicking and approximating human preferences.
- • Feature Compression Layers. Existing multi-stage approaches, such as MUSE (Chang et al., 2023) and DeepFloyd-XL (DeepFloyd, 2023), employ cascading multiple subnetworks to achieve higher-resolution image generation. We argue that such multi-stage training introduces unnecessary complexity and hampers the generation of high-fidelity, highresolution images. Instead, we advocate integrating streamlined feature compression layers during the fine-tuning stage to facilitate efficient high-resolution generation process learning. This approach functions akin to a lightweight high-resolution adapter (Guo et al., 2024), a module extensively explored and integrated within Stable Diffusion. By incorporating 2D convolution-based feature compression layers into the transformer backbone, we compress the feature maps prior to the transformer layers and subsequently decompress them after the transformer layers, effectively addressing the challenges of efficiency and resolution transition.

- 2.4 TRAINING DETAILS

Meissonic is constructed using a CLIP-ViT-H-142 text encoder (Ilharco et al., 2021), a pre-trained VQ image encoder and decoder (Patil et al.,

Table 1: Comparison of training data and time for various models. Model

8×A100 GPU Daysa W¨urstchen (Pernias et al., 2024) 1.0 1420 128.1

Params (B)

Training Images (M)

- SD-1.5 (Rombach et al., 2022b) 0.9 4800 781.2

- SD-2.1 (Rombach et al., 2022b) 0.9 3900 1041.6 Imagen (Saharia et al., 2022) 3.0 860 891.5 Dall-E 2 (Ramesh et al., 2022) 6.5 650 5208.3 GigaGAN (Kang et al., 2023) 0.9 980 597.8 SDXL (Podell et al., 2024) 2.6 unknown unknown Meissonic 1.0 210 19b

- 2024), and a customized Transformer-based (Esser et al., 2024) backbone. We employ classifier-free guidance (CFG) (Ho & Salimans, 2022) and cross-entropy loss to train Meissonic. Training occurs across three resolution stages, leveraging both public datasets and our curated data. First, we train Meissonic-256 with a batch size of 2,048 for 100,000 steps. Second, we continue training Meissonic-512 with a batch size of 512 for an additional 100,000 steps. Third, we continue training Meissonic with a batch size of 256 for 42,000 steps with a resolution of 1024 × 1024. The performance results of Meissonic-512 and Meissonic are reported in Table 2. All experiments are carried out with a fixed learning rate of 1×10−4 except Stage 4. Further details are elaborated in Section 2.5. All inferences in this paper are performed with CFG = 9 and 48 steps. We present performance comparisons with different numbers of inference steps and Classifier Free Guidance (CFG) in Appendix E.

- a Data collected from Sehwag et al. (2024).
- b FP16 Tensor Core of A100 is 312 TFLOPS and H100 is 989 TFLOPS. GPU hours are adjusted from 48 H100 days based on this rate.

It’s crucial to highlight the resource efficiency of our training process. Our training is considerably more resource-efficient compared to Stable Diffusion (Podell et al., 2023). Meissonic is trained in approximately 48 H100 GPU days, demonstrating that a production-ready image synthesis foundation model can be developed with considerably reduced computational costs. Additional details on this comparison can be found in Table 1.

2We utilize “laion/CLIP-ViT-H-14-laion2B-s32B-b79K” from OpenCLIP as our initial weights.

Table 2: HPS v2.0 benchmark. Scores are collected from https://github.com/tgxs002/ HPSv2. We highlight the best.

HPS v2.0 Model Animation Concept-art Painting Photo Averaged

GLIDE (Nichol et al., 2022) 23.34 23.08 23.27 24.50 23.55 LAFITE (Zhou et al., 2022) 24.63 24.38 24.43 25.81 24.81 VQ-Diffusion (Gu et al., 2022) 24.97 24.70 25.01 25.71 25.10 Latent Diffusion (Rombach et al., 2022b) 25.73 25.15 25.25 26.97 25.78 DALL·E mini 26.10 25.56 25.56 26.12 25.83 VQGAN + CLIP (Esser et al., 2021b) 26.44 26.53 26.47 26.12 26.39 CogView2 (Ding et al., 2022) 26.50 26.59 26.33 26.44 26.47 Versatile Diffusion (Xu et al., 2023) 26.59 26.28 26.43 27.05 26.59 DALL·E 2 (Ramesh et al., 2022) 27.34 26.54 26.68 27.24 26.95 Stable Diffusion v1.4 (Rombach et al., 2022a) 27.26 26.61 26.66 27.27 26.95 Stable Diffusion v2.0 (Rombach et al., 2022a) 27.48 26.89 26.86 27.46 27.17 Epic Diffusion 27.57 26.96 27.03 27.49 27.26 DeepFloyd-XL (DeepFloyd, 2023) 27.64 26.83 26.86 27.75 27.27 Openjourney 27.85 27.18 27.25 27.53 27.45 MajicMix Realistic 27.88 27.19 27.22 27.64 27.48 ChilloutMix 27.92 27.29 27.32 27.61 27.54 Deliberate (Desync, 2024) 28.13 27.46 27.45 27.62 27.67

- SDXL Base 0.9 (Podell et al., 2024) 28.42 27.63 27.60 27.29 27.73 Realistic Vision (SG 161222, 2024) 28.22 27.53 27.56 27.75 27.77

- SDXL Refiner 0.9 (Podell et al., 2024) 28.45 27.66 27.67 27.46 27.80 Dreamlike Photoreal 2.0 (Art, 2023) 28.24 27.60 27.59 27.99 27.86

SDXL Base 1.0 (Podell et al., 2024) 28.88 27.88 27.92 28.31 28.25

- SDXL Refiner 1.0 (Podell et al., 2024) 28.93 27.89 27.90 28.38 28.27 Meissonic-512 28.90 28.15 28.22 28.04 28.33 Meissonic 29.57 28.58 28.72 28.45 28.83

- 2.5 PROGRESSIVE AND EFFICIENT TRAINING STAGE DECOMPOSITION

Our approach systematically decomposes the training process into four carefully designed stages, allowing us to progressively build and refine the model’s generative capabilities. These stages, combined with precise enhancements to specific components, contribute to continual improvements in synthesis quality. Given that SDXL has not disclosed details regarding its training data, our experience is particularly valuable for guiding the community in constructing SDXL-level text-toimage models. We present images generated by Meissonic at each of the four training stages in Appendix K to support our claims.

- Stage 1: Understanding Fundamental Concepts from Extensive Data. Previous studies (Chen et al., 2024; Yu et al., 2024) indicate that raw captions from LAION are insufficient for training textto-image models, often requiring the caption refinement provided by MLLMs such as LLaVA (Liu et al., 2024b). However, this solution is computationally demanding and time-intensive. While some studies (Chen et al., 2024; Sehwag et al., 2024) utilize the extensively annotated SA-10M (Kirillov

et al., 2023) dataset, our findings reveal that SA-10M does not comprehensively cover fundamental concepts, particularly regarding human faces. Thus, we adopt a balanced strategy that leverages the original high-quality LAION data for foundational concepts learning in the initial training phase, utilizing a reduced resolution to enhance efficiency. Specifically, we carefully curated the deduplicated LAION-2B dataset by filtering out images with aesthetic scores below 4.5, watermark probabilities exceeding 50%, and other criteria outlined in Kolors (2024). This meticulous selection resulted in approximately 200 million images, which were employed for training at a resolution of 256 × 256 in this initial stage.

- Stage 2: Aligning Text and Images with Long Prompts. In the first stage, our approach does not rely on high-quality image-text paired data. Therefore, in the second stage, we focus on improving the model’s capability to interpret long, descriptive prompts. We filtered the initial LAION set more rigorously, retaining only images with aesthetic scores above 8, and other criteria outlined in Kolors

(2024). Additionally, we incorporate 1.2 million synthetic image-text pairs with refined captions exceeding 50 words, primarily derived from publicly available high-quality synthetic datasets, com-

[Figure 21]

[Figure 22]

A graphic poster depicting the fiery end of the world with detailed botanical illustrations and artistic influences.

[Figure 23]

[Figure 24]

A Pokemon that resembles a phone booth is gaining popularity on Artstation and Unreal Engine.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Low poly John Travolta in Golden Eye 64.

SD 1.5 SD 2.1 DeepFloyd-XL Deliberate SDXL 1.0 Meissonic

Figure 3: Qualitative Comparisons with SD 1.5, SD 2.1, DeepFloyd-XL, Deliberate, and SDXL.

plemented by additional high-quality images from our internal 6 million dataset. This aggregation results in around 10 million image-text pairs. Notably, we maintain the model architecture while increasing the training resolution to 512 × 512, enabling the model to capture more intricate image details. We observed a significant boost in the model’s ability to capture abstract concepts and respond accurately to complex prompts, including diverse styles and fantasy characters.

### Stage 3: Mastering Feature Compression for Higher-resolution Generation. High-resolution

generation remains an unexplored area within MIM (Chang et al., 2023; 2022; Patil et al., 2024). Unlike methods such as MUSE (Chang et al., 2023) or DeepFloyd-XL (DeepFloyd, 2023), which rely on external super-resolution (SR) modules, we demonstrate that efficient 1024×1024 generation is feasible through feature compression for MIM. By introducing feature compression layers, we achieve a seamless transition from 512×512 to 1024×1024 generation with minimal computational cost. In this stage, we further refine the dataset by filtering based on resolution and aesthetic score, selecting approximately 100K high-quality, high-resolution image-text pairs from the LAION subset utilized in Stage 2. This, combined with the remaining high-quality data, results in approximately 6 million samples for training at 1024 resolution.

#### Stage 4: Refining High-Resolution Aesthetic Image Generation. In the final stage, we fine-tune the model using a small learning rate, without freezing the text encoder, and incorporate human preference score as a micro condition. This can significantly enhance the model’s performance in high-resolution image generation. This targeted adjustment significantly enhances the model’s performance in generating high-resolution images, while also improving diversity. The training data remains the same as in Stage 3.

- 3 RESULTS

- 3.1 QUANTATIVE COMPARISON

Classic evaluation metrics for image generation models, such as FID and CLIP Score, have limited relevance to visual aesthetics, as highlighted by Podell et al. (2024); Chen et al. (2024); Kolors (2024); Sehwag et al. (2024). Therefore, we report our model’s performances using Human Pref-

Table 3: GenEval benchmark. We highlight the best result.

Objects

Model Overall

Counting Colors Position Color Attribution Single Two

DALL-E mini 0.23 0.73 0.11 0.12 0.37 0.02 0.01 SD v1.5 0.43 0.97 0.38 0.35 0.76 0.04 0.06 SD v2.1 0.50 0.98 0.51 0.44 0.85 0.07 0.17 DALL-E 2 0.52 0.94 0.66 0.49 0.77 0.10 0.19 SD XL 0.55 0.98 0.74 0.39 0.85 0.15 0.23

Meissonic 0.54 0.99 0.66 0.42 0.86 0.10 0.22

50

Meissonic-256 (fp16) Meissonic-512 (fp16) Meissonic-1024 (fp16)

GPUmemorycost(GB)

40

SDXL base (fp16)

SDXL base+refiner (fp16)

30

24.0 GB

20

12.0 GB 8.0 GB 4.0 GB

10

1 2 3 4 5 6 Inference Batch Size

3.0

40

| | |SDXL Ba Meissoni Meissoni<br><br>|se (step=1) c-256 (step c-512 (step|=1) =1)| | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | |Meissoni SDXL Ba Meissoni|c-1024 (ste se (step=50 c-256 (step|p=1) )<br><br>=50)| | | | | | | |
| | | | | | | | | | | | |
| | |Meissoni Meissoni<br><br>|c-512 (step c-1024 (ste|=50) p=50)| | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

InferenceTime(s)-Step50

InferenceTime(s)-Step1

2.5

30

2.0

1.5

20

1.0

10

0.5

0.0

0

1 2 3 4 5 6 7 8 Inference Batch Size

Table 4: GPU Memory Cost for for Different Models and Batch Sizes.

Table 5: Inference Time Comparison for Different Models and Batch Sizes.

erence Score v2 (HPSv2) (Wu et al., 2023), GenEval (Ghosh et al., 2024), and Multi-Dimensional Human Preference Score (MPS)3 (Zhang et al., 2024b), as illustrated in Table 2,3,6.

In our pursuit of making Meissonic accessible to the broader community, we optimized our model to 1 billion parameters, ensuring that it runs efficiently on 8GB VRAM, making inference and fine-tuning both convenient. Figure 4 provides a comparative analysis of GPU memory consumption4 across different inference batch sizes against SDXL. Additionally, Figure 5 details the inference time per step5. Furthermore, Figure 5 illustrates Meissonic’s proficiency in generating text-driven style art image.

Table 6: MPS scores on RealUser-800 Prompts. We highlight the best result.

Model MPS VQ-Diffusion (Gu et al., 2022) 9.70 Latent Diffusion (Rombach et al., 2022b) 10.56 DALL·E mini (Dayma et al., 2021) 11.32 VQGAN + CLIP (Esser et al., 2021b) 11.50 CogView2 (Ding et al., 2022) 12.39 Versatile Diffusion (Xu et al., 2023) 12.61 Stable Diffusion v1.4 (Rombach et al., 2022a) 13.89 Stable Diffusion v2.0 (Rombach et al., 2022a) 14.39 DeepFloyd-XL (DeepFloyd, 2023) 15.22

- SDXL Base 0.9 (Podell et al., 2024) 16.37

- SDXL Refiner 0.9 (Podell et al., 2024) 16.64

SDXL Base 1.0 (Podell et al., 2024) 16.46

- SDXL Refiner 1.0 (Podell et al., 2024) 16.56 Meissonic 17.34

We also present qualitative comparisons of image quality and text-image alignment in Figure 3, with additional comparisons provided in the Appendix M, performance comparisons for complex prompts versus simple prompts in Appendx D, performance comparisons with different numbers of inference steps and Classifier Free Guidance (CFG) in Appendix E, more comparisons with SDXL for image generation ability in Appendix G, additional images generated by Meissonic

- at diverse resolutions. These images can be found in Appendix O.

- 3Given that the KolorsPrompts benchmark was unavailable, we curated a diverse prompt dataset consisting

of 800 real user-generated prompts spanning various concepts and themes for the MPS evaluation.

- 4GPU memory usage was gauged using torch.cuda.memory reserved(). While this method might

yield higher values, all models are measured under identical settings to maintain fairness.

5Inference time is assessed using an A100 GPU with fp16 models. Notably, the reported times contributions from the VAE and text encoder, meaning that multi-step inferences do not scale linearly.

| |Evaluation Metrics<br><br>Image Quality<br><br>Color and Lighting<br><br>Composition<br><br>Detailed Representation<br><br>Creativity<br><br>Overall Impression| | | | |
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

100

90

80

Winrate[%]

70

60

50

40

30

20

SD v1.4 SD v2.0 DeepFloyd-XL SDXL

Models

- Figure 4: GPT4o Preference Evaluation of Meissonic against current open Text-to-image Models.

To complement these analyses, we conduct human evaluation by K-Sort Arena (Li et al., 2024) with internal checkpoint, we also conduct GPT-4o to evaluate the performance between Meissonic and other models in Figure 4.

All Figures and Tables demonstrate that Meissonic achieves competitive performance in human performance and text alignment compared to DALL-E 2 and SDXL, as well as showcasing its efficiency.

[Figure 30]

[Figure 31]

[Figure 32]

(a) SD 1.5 (b) SD 2.1

(c) SDXL (d) Messionic

[Figure 33]

- Figure 5: Evaluating the ability to generate diverse styles. The enlarged samples of (d) Meissonic are provided in Appendix L. Prompt: A garden full of [Y] illustrated in [X] style.

#### 3.2 ZERO-SHOT IMAGE-TO-IMAGE EDITING

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

A stylish dog wearing sunglasses

A woman grasp a cloth in her left hand

There is a fashionable girl standing among the green grass

A woman wearing a white suspender skirt is sitting

The steak is served on the plate and there is a fork next to it

A woman with short hair wore a silver gas mask

Figure 6: Examples of image editing with mask on internal Image Editing Dataset

For image editing tasks, we benchmark Meissonic against state-of-the-art models using the EMU-Edit dataset (Sheynin

Model CLIP-I↑ CLIP-T↑ DINO↑

InstructPix2Pix (Brooks et al., 2023) 0.834 0.219 0.762 MagicBrush (Zhang et al., 2024a) 0.838 0.222 0.776 PnP (Tumanyan et al., 2023) 0.521 0.089 0.153 Null-Text Inv. (Mokady et al., 2023) 0.761 0.236 0.678 EMU-Edit (Sheynin et al., 2024) 0.859 0.231 0.819 Meissonic 0.871 0.266 0.760

- et al., 2024), which includes seven different operations: background alteration, comprehensive image changes, style alteration, object removal, object addition, localized modifications, and color/texture alterations. We present results in Table 7. Additionally, examples from HumanEdit (Bai et al., 2024), including mask-guided editing in Figure 6 and mask-free editing in Figure 7, further showcase Meissonic’s versatility. Remarkably, Meissonic achieved this performance without any training or fine-tuning on image editing-specific data or instruction dataset. More comparisons for zero-shot image editing ability can be found in Appendix F.

Table 7: Results on the EMU-Edit (Sheynin et al., 2024) test set. We highlight the best result.

- 4 CONCLUSION AND IMPACT

In this work, we have significantly advanced masked image modeling (MIM) for text-to-image (T2I) synthesis by introducing several key innovations: a transformer architecture blends multi-modal and single-modal layers, advanced positional encoding strategies, and an adaptive masking rate as the sampling condition. These innovations, coupled with high-quality curated training data, progressive and efficient training stage decomposition, micro-conditions, and feature compression layers, have culminated in Meissonic, a 1B parameter model that outperforms larger diffusion models in highresolution, aesthetically pleasing image generation while remaining accessible on consumer-grade GPUs. Our evaluations demonstrate Meissonic’s superior performance and efficiency, marking a significant step towards accessible and efficient high-resolution non-autoregressive MIM T2I models.

Broader Impact. Recently, offline text-to-image applications on mobile devices have emerged, such as Pixel Studio from Google Pixel 9 and Image Playground from Apple iPhone 16. These

Input Output

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

InpaintingMask-freeEditingOutpainting

A great mosque A hot air balloon Statue of Liberty

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Fall mountains Rocket launch site Volcano

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Dog Small tiger A robot cat

- Figure 7: Examples of image inpainting, outpainting, and mask-free image editing on HumanEdit

innovations reflect a growing trend toward enhancing user experience and privacy. As a pioneering resource-efficient foundation model, Meissonic represents a significant advancement in this field, delivering state-of-the-art image synthesis capabilities with a strong emphasis on user privacy and offline functionality. This development not only empowers users with creative tools but also ensures the security of sensitive data, marking a notable leap forward in mobile imaging technology.

- 5 ACKNOWLEDGEMENTS

This work was supported in part by NUS Start-up Grant A-0010106-00-00, the Guangdong Science and Technology Department (No. 2024ZDZX2004), the Nansha Key Area Science and Technology Project (No. 2023ZD003) and the InnoHK funding launched by Innovation and Technology Commission, Hong Kong SAR.

We would like to express our gratitude to all those who contributed their time, expertise, and insights during the development of Meissonic. Listed in no particular order: Jingjing Ren, Sixiang Chen from HKUST(GZ), Wenhao Chai from University of Washington, Donghao Zhou from CUHK, and other anonymous friends. We are profoundly grateful for their commitment and the unique perspectives they brought to this project.

- A MODEL NAME ORIGIN

The name “Meissonic” is derived from a combination of the renowned French painter Ernest Meissonier and the term “sonic”. Ernest Meissonier is celebrated for his meticulous attention to detail and his ability to capture dynamic moments in art. The addition of “sonic” evokes a sense of speed and modernity, highlighting the model’s capabilities in efficient image synthesis and transformation.

- B RELATED WORK

Diffusion-based Image Generation. Diffusion models have achieved remarkable advances in image generation, with notable contributions like Stable Diffusion (Rombach et al., 2022b), and the more recent SDXL (Podell et al., 2024), often driven by large-scale datasets. These models move beyond pixel-level operations by working within compressed latent spaces, forming what we now recognize as latent diffusion models (Luo et al., 2023; Podell et al., 2024; Wu et al., 2024a; Shi et al., 2024; Zhou et al., 2024; Yi et al., 2024; Wu et al., 2024b). SDXL represents a significant leap in this domain, introducing micro-conditions and multi-aspect training to gain greater control over image generation, which has inspired a wide range of derivative models in the community, such as Deliberate (Desync, 2024) and RealVisXL (SG 161222, 2024).

The integration of transformer architectures has also become more prevalent, with models like DiT (Peebles & Xie, 2023) and U-ViT (Bao et al., 2023) demonstrating the potential of diffusion transformers in this field. SD3 (Esser et al., 2024), which combines diffusion transformers with flow matching at an impressive scale of 8B parameters, underscores the scalability and potential of the multimodal transformer-based diffusion backbone. Despite these advances, diffusion models still face challenges, particularly their reliance on acceleration techniques (Sauer et al., 2023; Luo et al., 2023; Yin et al., 2024) to speed up inference, making them cumbersome for real-time applications. Additionally, the quantization of diffusion transformers has proven less straightforward than with large language models (Li et al., 2023). The research community continues to explore better paradigms for image generation. Addressing these limitations, our work aims to contribute an efficient, high-quality alternative in the form of Meissonic.

Token-based Image Generation. Token-based autoregressive transformers (Lee et al., 2022; Chen et al., 2018; Yu et al., 2022b), first validated by VQ-GAN (Esser et al., 2021b), have shown considerable promise for image generation. However, these methods are inherently computationally demanding, requiring the prediction of hundreds to thousands of tokens to form a single image. As a pioneering work, MaskGIT (Chang et al., 2022) challenged this paradigm by introducing a masked image modeling (MIM) approach, achieving competitive fidelity and diversity in class-conditional image generation. Building on this, MUSE (Chang et al., 2023) extended MIM to text-to-image synthesis, scaling up to 3B parameters and achieving remarkable performance.

MUSE demonstrates the viability of non-autoregressive token-based models, but it encountered limitations in generating high-resolution images, capping at 512×512, and lagging behind SDXL (Podell et al., 2023) in terms of fidelity and text-image alignment. Meissonic advances the performance of token-based models beyond what latent diffusion methods have achieved, effectively pushing the envelope in terms of both quality and resolution in the text-to-image synthesis landscape with the MIM method.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- Figure 8: Zero-shot generation of stylized letters. Meissonic can synthesize individual letters to form the word “MEISSONIC”. Prompt: A post featuring a [COLOR] ‘[LETTER]’ painted on top.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Figure 9: Memes generated by Meissonic.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Figure 10: Cartoon Stickers generated by Meissonic.

- C APPLICATIONS

We present the letter synthesis capability of Meissonic in Figure 8. We present the combination capability of complex concepts of Meissonic in Figure 1. We present meme generation in Figure 9. We present cartoon sticker generation in Figure 10.

- D PERFORMANCE COMPARISONS FOR COMPLEX VERSUS SIMPLE PROMPTS We present performance comparisons for complex prompts versus simple prompts in Figure 11.
- E PERFORMANCE COMPARISONS WITH DIFFERENT NUMBERS OF INFERENCE STEPS AND CLASSIFIER FREE GUIDANCE (CFG)

We present performance comparisons with different numbers of inference steps and Classifier Free Guidance (CFG) in Figure 12,13,14,15,16,17.

[Figure 81]

[Figure 82]

[Figure 83]

A white table with a vase of flowers and a cup of coffee on top of it, accompanied by a plate of buttery croissants, a folded linen napkin, and a faint ray of sunlight streaming through a nearby window in a cozy dining room.

A white table with a vase of flowers and a cup of coffee on top of it.

Table flowers.

[Figure 84]

[Figure 85]

[Figure 86]

A busy train station with people hurrying along the platforms, some carrying luggage, while a sleek modern train is arriving, its headlights cutting through the slight morning haze, under a vast glass roof with beams of sunlight streaming in.

A busy train station with people hurrying along the platforms.

Train station.

[Figure 87]

[Figure 88]

[Figure 89]

A cozy wooden cabin covered in a blanket of snow, with smoke rising from its chimney, surrounded by tall pine trees, as soft snowflakes fall from the gray sky, and a warm yellow glow from the windows invites you in.

Snow cabin.

A cozy wooden cabin covered in a blanket of snow.

[Figure 90]

[Figure 91]

[Figure 92]

A vibrant city at night with skyscrapers illuminated by neon lights, busy streets filled with cars and people, and a towering billboard flashing colorful advertisements, while a clear night sky reveals the faint twinkle of distant stars.

Night city.

A vibrant city at night with skyscrapers illuminated by neon lights.

Figure 11: Performance Comparisons for Complex versus Simple Prompts

CFG: 1 CFG: 2 CFG: 3 CFG: 4 CFG: 5 CFG: 6 CFG: 7 CFG: 8 CFG: 9 CFG: 10

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

Steps:1

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

Steps:3Steps:5Steps:7Steps:9Steps:11Steps:12Steps:13Steps:15Steps:20Steps:30Steps:40Steps:48Steps:50Steps:60

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

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

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

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

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

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

- Figure 12: Performance Comparisons with Different Numbers of Inference Steps and Classifier Free Guidance (CFG). Prompt: A statue of a man with a crown on his head.

CFG: 1 CFG: 2 CFG: 3 CFG: 4 CFG: 5 CFG: 6 CFG: 7 CFG: 8 CFG: 9 CFG: 10

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Steps:1

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Steps:3Steps:5Steps:7Steps:9Steps:11Steps:12Steps:13Steps:15Steps:20Steps:30Steps:40Steps:48Steps:50Steps:60

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

- Figure 13: Performance Comparisons with Different Numbers of Inference Steps and Classifier Free Guidance (CFG). Prompt: Studio photo portrait of Lain Iwakura from Serial Experiments Lain wearing floral garlands over her traditional dress.

CFG: 1 CFG: 2 CFG: 3 CFG: 4 CFG: 5 CFG: 6 CFG: 7 CFG: 8 CFG: 9 CFG: 10

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Steps:1

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

Steps:3Steps:5Steps:7Steps:9Steps:11Steps:12Steps:13Steps:15Steps:20Steps:30Steps:40Steps:48Steps:50Steps:60

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

- Figure 14: Performance Comparisons with Different Numbers of Inference Steps and Classifier Free Guidance (CFG). Prompt: A girl gazes at a city from a mountain at night in a colored manga illustration by Diego Facio.

CFG: 1 CFG: 2 CFG: 3 CFG: 4 CFG: 5 CFG: 6 CFG: 7 CFG: 8 CFG: 9 CFG: 10

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

Steps:1

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

Steps:3Steps:5Steps:7Steps:9Steps:11Steps:12Steps:13Steps:15Steps:20Steps:30Steps:40Steps:48Steps:50Steps:60

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

- Figure 15: Performance Comparisons with Different Numbers of Inference Steps and Classifier Free Guidance (CFG). Prompt: A tranquil lake surrounded by snow-capped mountains under a clear sky.

CFG: 1 CFG: 2 CFG: 3 CFG: 4 CFG: 5 CFG: 6 CFG: 7 CFG: 8 CFG: 9 CFG: 10

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

Steps:1

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

Steps:3Steps:5Steps:7Steps:9Steps:11Steps:12Steps:13Steps:15Steps:20Steps:30Steps:40Steps:48Steps:50Steps:60

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

- Figure 16: Performance Comparisons with Different Numbers of Inference Steps and Classifier Free Guidance (CFG). Prompt: A futuristic cityscape with hovering vehicles and towering structures.

CFG: 1 CFG: 2 CFG: 3 CFG: 4 CFG: 5 CFG: 6 CFG: 7 CFG: 8 CFG: 9 CFG: 10

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

Steps:1

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

Steps:3Steps:5Steps:7Steps:9Steps:11Steps:12Steps:13Steps:15Steps:20Steps:30Steps:40Steps:48Steps:50Steps:60

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

- Figure 17: Performance Comparisons with Different Numbers of Inference Steps and Classifier Free Guidance (CFG). Prompt: A massive starship docked in a glowing nebula.

## F MORE COMPARISONS FOR ZERO-SHOT IMAGE EDITING ABILITY

To ensure fair evaluations of zero-shot capabilities with SD1.5 and SDXL, we utilize Null-Text Inversion (Mokady et al., 2023) for zero-shot editing with our method, taking into account that other methods have been extensively trained on editing datasets. The configurations used for Null-Text Inversion, along with any undocumented parameters, align with those provided in the official code repository. The primary parameters are outlined as follows:

- • cross replace steps.default = 0.8

- • self replace steps = 0.5

- • blend words = None

- • equilizer params = None

For consistency, we use the recommended 512 × 512 resolution for editing and ran tests using torch.float32, which is the official setting for Null-Text Inversion.On A6000 GPUs (48 GB), the execution of MagicBrush (Zhang et al., 2024a) takes approximately 36 hours for SD1.5 and 60 hours for SDXL. The runtime for Emu-Edit is significantly longer. Given the extensive computation, we randomly sample 500 examples per benchmark for testing.

We present more comparisons for zero-shot image editing ability on EMU-Edit in Table 8.

| |CLIP-I↑ CLIP-T↑ DINO↑ L1↓ CLIPdir↑|
|---|---|
|SD 1.5 + Null-Text Inv. SDXL + Null-Text Inv. Meissonic-512 (Ours)<br><br>|0.780 0.240 0.637 0.159 0.096 0.787 0.238 0.653 0.146 0.085 0.791 0.244 0.689 0.128 0.102|

- Table 8: EMU-Edit Results

We present more comparisons for zero-shot image editing ability on MagicBrush in Table 9.

| |CLIP-I↑ CLIP-T↑ DINO↑ L1↓ CLIPdir↑|
|---|---|
|SD 1.5 + Null-Text Inv. SDXL + Null-Text Inv. Meissonic-512 (Ours)<br><br>|0.824 0.228 0.647 0.121 0.106 0.840 0.241 0.665 0.122 0.111 0.835 0.248 0.689 0.115 0.120|

- Table 9: MagicBrush Results

Our findings indicate that due to the inherent characteristics of MIM, Meissonic exhibits faster zeroshot editing capabilities. Performances are evaluated with batch size = 1 and inference step = 50 (compared to Null-Text Inv., which requires 500 backpropagation steps). Tests are conducted on an A6000 GPU with 48 GB VRAM.

Besides, we present inference time comparision in Table 10.

| |SD 1.5 + Null-Text Inv. SDXL + Null-Text Inv. Meissonic-512 (Ours)|
|---|---|
|Time (s/10 pairs) GPU (GB)<br><br>|1040 + 100 1850 + 120 108 13.4 26.8 5.9|

Table 10: Inference Time Comparison

These results demonstrate the substantial potential for reduced processing time with Meissonic. We also present qualitative comparisons on zero-shot image editing ability in Figure 18.

## G MORE COMPARISONS WITH SDXL FOR IMAGE GENERATION ABILITY We present more comparisons with SDXL for image generation ability in Figure 19,20,21.

SD 1.5 SDXL Meissonic-512

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

Edit Prompt: A wooden bear

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

Edit Prompt: Mountains with a river flowing between them

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

Edit Prompt: A cat wearing hat

Figure 18: Qualitative comparisons on zero-shot image editing ability.

Meissonic SDXL

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

- Figure 19: Qualitative comparisons with SDXL for image generation ability. Prompt: A breathtaking photo of a serene mountain lake at sunrise, crystal-clear water reflecting the surrounding snow-capped peaks, with a soft mist floating above the surface.

- H ABLATION STUDY

Detailed roadmap to build Meissonic. We present ablation studies during training Meissonic-512 in Table. 22. The HPS v2.1 (Wu et al., 2023) scores are calculated for verifying the effectiveness of

Meissonic SDXL

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

- Figure 20: Qualitative comparisons with SDXL for image generation ability. Prompt: A professional studio photograph of a fresh bouquet of wildflowers in a glass vase, water droplets visible on the petals and leaves, placed on a clean white background.

[Figure 1041]

[Figure 1042]

Meissonic SDXL

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

- Figure 21: Qualitative comparisons with SDXL for image generation ability. Prompt: A sharp photo of a modern skyscraper during blue hour, its glass facade reflecting the city lights and the deep indigo sky in the background.

each compoment. Our ablations are based on training stage 2, ensuring consistency with the training dataset scale, model scale, and other training configurations.

| |26.36<br><br>27.85<br><br>28.19 28.14<br><br>28.74<br><br>30.17 29.32<br><br>30.77 30.94| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Transformer baseline add micro_conds finetune text enc

finetune vqvae more inference steps

add RoPE reduce training data

7

add masking_rate conds Meissonic-512

4

26 28 30 32

Figure 22: HPS v2.1 Score on internal 1000 prompts

𝑦 Linear

𝑦 Linear

𝑄 𝐾 𝑉

ModulationModulation

ModulationModulation LayerNormLayerNorm

NormNorm

LayerNormLayerNorm

RoPERoPE

Linear

LinearLinear

⊙

𝑐

Softmax Attention

⊙

𝑄 𝐾 𝑉

Linear

x

⊙

𝑦 Linear

𝑦 Linear

Figure 23: Multi-modal Transformer For Meissonic.

- I MULTIMODAL TRANSFORMER BLOCK FOR MEISSONIC

We present a detailed structure of our Multi-modal Transformer Block for Meissonic in Figure 23. Specifically, x denotes image embedding inputs, c denotes text embedding inputs, and y denotes conditions inputs.

[Figure 1059]

Figure 24: Word cloud image of our RealUser-800 prompts benchmark.

- J WORD CLOUD OF OUR REALUSER800 BENCHMARK

We present a word cloud image that illustrates the diverse concepts, styles, and themes encompassed within our RealUser-800 prompts benchmark in Figure 24.

- K IMAGES GENERATED DURING DIFFERENT TRAINING STAGES

We present images generated using the same prompt across Meissonic’s four training stages in Figure 25.

- L ENLARGED EXAMPLES FROM GENERATING DIVERSE STYLES We present enlarged samples from Figure 5 (d) Meissonic in Figure 26.
- M MORE EXAMPLES OF QUALITATIVE COMPARISONS We present more examples of qualitative comparisons in Figure 27.
- N MORE IMAGES PRODUCED BY MEISSONIC

We present additional images generated by Meissonic using CC3M (Sharma et al., 2018) items, with detailed captions provided by VILA-1.5 (Lin et al., 2023) and Morph (Pan et al., 2024). These images can be found in Figure 28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46.

We present additional images generated by Meissonic using HPS (Wu et al., 2023) benchmark prompts. These images can be found in Figure 47,48,49,50,51,52.

- O MORE IMAGES PRODUCED BY MEISSONIC AT DIVERSE RESOLUTIONS

We present additional images generated by Meissonic at diverse resolutions. These images can be found in Figure 53,54.

Stage 1 Stage 2 Stage 3 Stage 4

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

An image of a Pikachu wearing a birthday hat and playing guitar

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

A black and white line drawing of a rhinoceros head.

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

A leopard is sitting on a tree branch in a forest with its front paws resting on the trunk.

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

A sculpture of a Greek woman head with a headband and a head of hair.

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

A white sports car is driving down a desert road.

- Figure 25: Images generated using the same prompt across Meissonic’s four training stages. The resolutions for stages 1 and 2 are 2562 and 5122, respectively, while stages 3 and 4 are 10242. For clarity and comparison, all images are displayed in a consistent layout.

[Figure 1080]

UnicornsPandasButterfliesReindeers

Sketch Origami Animated Aquarelle Pixel Art Cyberpunk Papercut

- Figure 26: Enlarged Examples from generating diverse styles with Meissonic. Prompt: A garden full of [Y] illustrated in [X] style.

[Figure 1081]

[Figure 1082]

Spiderman as Wolverine with detailed muscular features and a full face, trending on multiple art platforms,

created with hyperdetailed Unreal Engine, and optimized for high resolution viewing.

[Figure 1083]

[Figure 1084]

A digital painting of a Pokémon named Faerow in a concept art style.

[Figure 1085]

[Figure 1086]

The image features Breton monks resembling Rasputin from The Lorax, with cinematic lighting and a shallow

depth of field.

[Figure 1087]

[Figure 1088]

The image depicts a God smashing mirrors, while a detailed unicom-dragon is present in the scene.

[Figure 1089]

[Figure 1090]

Architecture render with pleasing aesthetics.

[Figure 1091]

[Figure 1092]

Exploded view diagram of a xenomorph.

[Figure 1093]

[Figure 1094]

A samurai in space. SD 1.5 SD 2.1 DeepFloyd-XL Deliberate SDXL 1.0 Meissonic

- Figure 27: Qualitative Comparisons with SD 1.5, SD 2.1, DeepFloyd-XL, Deliberate, and SDXL.

[Figure 1095]

[Figure 1096]

The wizard chants a spell over the apple

Pumpkin head wearing black wizard hat

[Figure 1097]

[Figure 1098]

A bedroom with a canopy bed and a wooden floor

Two women in black dresses with feathers on their heads.

[Figure 1099]

[Figure 1100]

A sled sits in a field with a sunset in the background.

A blue and white drawing of a sea dragon.

Figure 28: High Quality Samples Produced by Meissonic.

[Figure 1101]

[Figure 1102]

A body of water with a cliff in the background

A collection of statues of Asian men and women.

[Figure 1103]

[Figure 1104]

A table with a parrot on it and a map on it.

A man with blonde hair and glasses is looking at the camera.

[Figure 1105]

[Figure 1106]

A beautiful sunset with a reflection of the Marina Bay Sands hotel.

Two snowmen are standing next to a snowman with a blank sign.

Figure 29: High Quality Samples Produced by Meissonic.

[Figure 1107]

[Figure 1108]

Two people walking in the snow with a sled.

A tiger is swimming in a body of water.

[Figure 1109]

[Figure 1110]

A man with a crown and a blue robe is holding a glass.

A seal is sitting in the snow with its mouth open.

[Figure 1111]

[Figure 1112]

A small bird is perched on a wooden post.

An old man with a blue turban and a blue shirt is standing in front of a wooden wall.

Figure 30: High Quality Samples Produced by Meissonic.

[Figure 1113]

[Figure 1114]

A small cute white dog is sitting on a bed.

A woman is laying on a couch and smiling.

[Figure 1115]

[Figure 1116]

A cloudy sky over a body of water. A fat man is holding a large black and white dog in a black-white figure style.

[Figure 1117]

[Figure 1118]

A white car with a silver rim and a headlight.

A young girl is holding a bouquet of flowers.

Figure 31: High Quality Samples Produced by Meissonic.

[Figure 1119]

[Figure 1120]

A man in a black leather suit sits in a red chair.

A bed with a red and white quilt on it.

[Figure 1121]

[Figure 1122]

A woman in a white dress is looking at her phone.

A dog with a blue collar is looking at the camera.

[Figure 1123]

[Figure 1124]

A statue of a man in front of a building.

A seal is wearing a Santa hat and is on a snowy hill with the words Happy New Year written below it.

Figure 32: High Quality Samples Produced by Meissonic.

[Figure 1125]

[Figure 1126]

A plush toy of a girl with red eyes and a pink shirt.

A woman with a flower crown on her head.

[Figure 1127]

[Figure 1128]

A metal sculpture of a deer with antlers.

A doll wearing a blue and white dress and a tan shawl.

[Figure 1129]

[Figure 1130]

A penguin walks in the snow with a red hat on.

A man in a red jersey holding a basketball.

Figure 33: High Quality Samples Produced by Meissonic.

[Figure 1131]

[Figure 1132]

A cat is looking at a butterfly A woman in a white wedding dress stands in a courtyard.

[Figure 1133]

[Figure 1134]

Two firefighters standing in front of a smoky background.

A model walks down a runway in a black dress.

[Figure 1135]

[Figure 1136]

A man wearing an orange hat and scarf is screaming

A black and white drawing of a dog's head in a circle.

Figure 34: High Quality Samples Produced by Meissonic.

[Figure 1137]

[Figure 1138]

A statue of a woman surrounded by flowers

A chess board with a row of chess pieces.

[Figure 1139]

[Figure 1140]

A woman in a pink shirt is sitting on a bed.

A bronze statue of an owl with its wings spread.

[Figure 1141]

[Figure 1142]

A group of women in red uniforms pose for a picture.

A gold mask with a gold strap is on a black surface.

Figure 35: High Quality Samples Produced by Meissonic.

[Figure 1143]

[Figure 1144]

A white goat with horns is standing in the snow.

A map of Africa with a blue background.

[Figure 1145]

[Figure 1146]

A woman stands on a dock in the fog. A woman is standing next to a picture of another woman.

[Figure 1147]

[Figure 1148]

A man wearing a virtual reality headset.

A white table with a vase of flowers and a cup of coffee on top of it.

Figure 36: High Quality Samples Produced by Meissonic.

[Figure 1149]

[Figure 1150]

A white and blue coffee mug with a picture of a man on it.

A statue of a man with a crown on his head.

[Figure 1151]

[Figure 1152]

Four bottles of maple syrup in different colors.

A woman in a black wetsuit sits on a bench gazing at the sea on the beach.

[Figure 1153]

[Figure 1154]

A soccer player in a blue and white uniform runs with the ball.

A man in a yellow wet suit is holding a big black dog in the water.

Figure 37: High Quality Samples Produced by Meissonic.

[Figure 1155]

[Figure 1156]

A pillow with a picture of a man on it. An Indian woman is wearing a white saree and standing in front of a pink wall.

[Figure 1157]

[Figure 1158]

A woman holding a baby.

A large ship is in the water with a foggy background.

[Figure 1159]

[Figure 1160]

An ancient Egyptian carved stone wall with three figures and hieroglyphics.

A snowy owl is sitting in the snow.

Figure 38: High Quality Samples Produced by Meissonic.

[Figure 1161]

[Figure 1162]

A woman is sitting on a boat and looking at a boat in the water.

A woman drinking from a cup with a blurry background.

[Figure 1163]

[Figure 1164]

A large body of water with a rock in the middle and mountains in the background.

A puffin is sitting on a rock and looking off into the distance.

[Figure 1165]

[Figure 1166]

A lynx is standing in the snow. Two actors are posing for a picture with one wearing a black and white face paint.

Figure 39: High Quality Samples Produced by Meissonic.

[Figure 1167]

[Figure 1168]

A statue of Jesus Christ is holding a feather in his hand in a purple style.

A dog is laying on the floor.

[Figure 1169]

[Figure 1170]

A narrow stone pathway is enveloped by lush greenery and a veil of mist.

A black boat is tied to a dock on a calm lake.

[Figure 1171]

[Figure 1172]

A white and black motorcycle with a headlight on it.

A woman with short red hair is looking off into the distance.

Figure 40: High Quality Samples Produced by Meissonic.

[Figure 1173]

[Figure 1174]

A statue of a lion stands in front of a building.

A bathroom with a modern design and a classic design.

[Figure 1175]

[Figure 1176]

Benjamin Franklin appears among a pile of US dollars

A frozen river with ice on the surface.

[Figure 1177]

[Figure 1178]

Pope Francis is talking to black priests.

Cherry blossoms bloom under the Eiffel Tower.

Figure 41: High Quality Samples Produced by Meissonic.

[Figure 1179]

[Figure 1180]

A young girl is holding a bowling ball. A pair of bride and groom figurines are positioned atop a white, twotiered cake.

[Figure 1181]

[Figure 1182]

A woman in a gold dress poses for a photo.

A ship is sailing in the ocean with mountains in the background.

[Figure 1183]

[Figure 1184]

A woman wearing a headband hat and a white dress is walking down a runway.

A squirrel is holding a gift bag with mouse open in the snow.

Figure 42: High Quality Samples Produced by Meissonic.

[Figure 1185]

[Figure 1186]

A man with long hair and a beard stands in a room, with a portrait of himself positioned behind him.

A gorilla is looking at the camera with a serious expression.

[Figure 1187]

[Figure 1188]

A man with a hoodie on is looking at the camera.

A sunset over a body of water with a tree in a small island.

[Figure 1189]

[Figure 1190]

The collage consists of photos featuring the bride and groom. The bride occupies half of the collage. The groom appears in two photos, one in a white suit and the other in a black suit.

A black and white photo of a cross in a field.

Figure 43: High Quality Samples Produced by Meissonic.

[Figure 1191]

[Figure 1192]

A woman with short hair and earrings is smiling.

A man dressed in Viking attire is seated among the crowd.

[Figure 1193]

[Figure 1194]

Two gloden statues of lions standing in a field.

A dog is sitting in the snow in front of a mountain.

[Figure 1195]

[Figure 1196]

A race car is driving on a track. A guitar is sitting on a wooden floor in front of a purple wall.

Figure 44: High Quality Samples Produced by Meissonic.

[Figure 1197]

[Figure 1198]

A man with a shaved head and a tattoo on his back.

A deer is drawn in a geometric style.

[Figure 1199]

[Figure 1200]

A baby is sitting on a white blanket holding a white rose.

A woman is standing on a staircase, back to the camera with three chains hanging from the ceiling.

[Figure 1201]

[Figure 1202]

A woman wearing a crown and a necklace is smiling.

A surreal mental landscape, in which elements of nature and a house emerge from the back of a woman's head.

Figure 45: High Quality Samples Produced by Meissonic.

[Figure 1203]

[Figure 1204]

A forest with trees and fog. A cloudy sky over a green field.

[Figure 1205]

[Figure 1206]

An image depicting a minimalist design featuring a pool situated in front of a white building with palm trees.

A man wearing a large blue and gold feathered headdress.

[Figure 1207]

[Figure 1208]

A lion's head is shown in a grayscale image.

Three origami dogs, one of which is purple, while the others are yellow.

Figure 46: High Quality Samples Produced by Meissonic.

[Figure 1209]

[Figure 1210]

A blind monk wearing an orange robe stares out the window of a spaceship in a dramatic lighting as depicted in a matte painting.

A racoon wearing a suit smoking a cigar in the style of James Gurney.

[Figure 1211]

[Figure 1212]

Classical romantic painting of Hatsune Miku with blue hair.

An astronaut floats amidst planets against a cosmic backdrop in a highly detailed, refreshing digital painting by James Jean.

[Figure 1213]

[Figure 1214]

Close-up hyperrealistic oil painting portrait of a nun fashion model looking up against a black background, with classicism and 80s sci-fi Japanese book art influences.

A digital painting of a hairless, inside-out cat with intricate details and a horror theme.

Figure 47: High Quality Samples Produced by Meissonic.

[Figure 1215]

[Figure 1216]

A raccoon in formal attire, carrying a bag and cane, depicted in a Rembrandt-style oil painting.

A cinematic fashion portrait of a Hindu goddess standing in a beautiful garden.

[Figure 1217]

[Figure 1218]

A Landrover crosses a forest path in the rain in a highly-detailed digital painting by artists Greg Rutkowski and Artgerm.

A portrait of Mario and Luigi from Mario Bros with a detailed face and a city background, painted by Bouguereau.

[Figure 1219]

[Figure 1220]

Image of Albert Einstein created by Park Jun Seong.

A painting depicting a wuxia character standing on a roof under a moonlit night.

Figure 48: High Quality Samples Produced by Meissonic.

[Figure 1221]

[Figure 1222]

Steve Buscemi portrays the Joker. Theintricateimageanddepictshyperstormtroopersdetailed design,incharacterizeda hyper realisticbystyle,ambientwithand volumetric lighting, reminiscent of Star Wars concept art by George Lucas and Ralph McQuarrie, with a style similar to GTA V.

[Figure 1223]

[Figure 1224]

Image depicting a person's face composed entirely of fruits and vegetables.

A space man sat on a beach chair on the moon, pixel art.

[Figure 1225]

[Figure 1226]

A cyberpunk-style Batman in a dark city, depicted in an extremely detailed piece of artwork by Chris Labrooy.

The image is a trippy cheeseburger with warm colors, depicted in highly detailed illustration and rendered in octane, created by the award winning studio 4.

Figure 49: High Quality Samples Produced by Meissonic.

[Figure 1227]

[Figure 1228]

A Salem black cat girl in anime style with a simple background.

A cute anthropomorphic fox knight wearing a cape and crown in pale blue armor.

[Figure 1229]

[Figure 1230]

Blond-haired girl depicted in anime style.

A cute anime-style female cat girl with large eyes is pictured underwater with a simple background.

[Figure 1231]

[Figure 1232]

A girl peers over the edge of a mountain at a giant city in the dark of night, depicted in a manga illustration by Kentaro Miura and Hiromu Arakawa.

Illustration of an anime maid with a pretty face and eyes, shown in a fullbody upper shot.

Figure 50: High Quality Samples Produced by Meissonic.

[Figure 1233]

[Figure 1234]

A full-body shot of an anime maid with rich detail, featuring a pretty face and eyes.

A minimalist tattoo inspired by the Studio Ghibli films

[Figure 1235]

[Figure 1236]

Asuna from Sword Art Online. The Little Prince talking to the fox in an animation shot by Tim Burton's art.

[Figure 1237]

[Figure 1238]

Anime portrait of an Asian schoolgirl with her pet sugar glider.

Luke Skywalker with Muppets.

Figure 51: High Quality Samples Produced by Meissonic.

[Figure 1239]

[Figure 1240]

Medium shot black and white manga pencil drawing with a highly detailed face of Alita by Yukito Kishiro.

Studio photo portrait of Lain Iwakura from Serial Experiments Lain wearing floral garlands over her traditional dress.

[Figure 1241]

[Figure 1242]

Frontal portrait of anime girl with pink hair wearing white t-shirt and smiling.

Anime oil painting of Rem from Re Zero.

[Figure 1243]

[Figure 1244]

Anime-style fighter pilot in cockpit engaged in a night air battle with explosions.

A girl gazes at a city from a mountain at night in a colored manga illustration by Diego Facio.

Figure 52: High Quality Samples Produced by Meissonic.

[Figure 1245]

[Figure 1246]

960 x 1280, A dense jungle with sunlight filtering through the canopy.

[Figure 1247]

1280 x 960, A large body of water with a rock in the middle and mountains in the background.

[Figure 1248]

960 x 1280, A massive starship docked in a glowing nebula.

[Figure 1249]

1280 x 1024, A white table with a vase of flowers and a cup of coffee on top of it.

960 x 1280, A mystical temple hidden deep within a cloud-covered mountain.

Figure 53: More Images Produced by Meissonic at Diverse Resolutions.

[Figure 1250]

[Figure 1251]

960 x 1280, A quiet meadow bathed in soft morning dew.

[Figure 1252]

1280 x 960, A frozen lake surrounded by snow-covered trees under a pale winter sun.

1024 x 1280, A stormy sea with crashing waves and lightning illuminating the clouds.

[Figure 1253]

1024 x 2048, A frozen lake surrounded by snow-covered trees under a pale winter sun.

Figure 54: More Images Produced by Meissonic at Diverse Resolutions.

REFERENCES

Dreamlike Art. Dreamlike photoreal 2.0. https://huggingface.co/dreamlike-art/ dreamlike-photoreal-2.0, 2023.

Jinbin Bai, Wei Chow, Ling Yang, Xiangtai Li, Juncheng Li, Hanwang Zhang, and Shuicheng Yan. Humanedit: A high-quality human-rewarded dataset for instruction-based image editing. arXiv preprint arXiv:2412.04280, 2024.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22669–22679, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11315–11325, 2022.

Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. Pixelsnail: An improved autoregressive generative model. In International conference on machine learning (ICML), pp. 864–872. PMLR, 2018.

Boris Dayma, Suraj Patil, Pedro Cuenca, Khalid Saifullah, Tanishq Abraham, Phuc Le Khac, Luke Melas, and Ritobrata Ghosh. Dall· e mini. https://huggingface.co/spaces/ dallemini/dalle-mini, 2021.

IF DeepFloyd. Deepfloyd if, 2023. https://huggingface.co/DeepFloyd, 2023. Desync. Perfect deliberate. https://civitai.com/models/24350/

perfectdeliberate, 2024.

Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems (NeurIPS), 35:16890–16902, 2022.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. Longrope: Extending llm context window beyond 2 million tokens. arXiv preprint arXiv:2402.13753, 2024.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12873–12883, 2021a.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 12873–12883, 2021b.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems (NeurIPS), 36, 2024.

Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 10696–10706, 2022.

Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation. arXiv preprint arXiv:2402.10491, 2024.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021.

Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10124–10134, 2023.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, 2023.

Kolors. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024.

Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11523–11532, 2022.

Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17535–17545, 2023.

Zhikai Li, Xuewen Liu, Dongrong Fu, Jianquan Li, Qingyi Gu, Kurt Keutzer, and Zhen Dong. K-sort arena: Efficient and reliable benchmarking for generative models via k-wise human preferences. arXiv preprint arXiv:2408.14468, 2024.

Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2023.

Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024b.

Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai. Fit: Flexible vision transformer for diffusion model. arXiv preprint arXiv:2402.12376, 2024.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6038–6047, 2023.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning (ICML), pp. 16784–16804. PMLR, 2022.

Kaihang Pan, Siliang Tang, Juncheng Li, Zhaoyu Fan, Wei Chow, Shuicheng Yan, Tat-Seng Chua, Yueting Zhuang, and Hanwang Zhang. Auto-encoding morph-tokens for multimodal llm. arXiv preprint arXiv:2405.01926, 2024.

Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction, 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville. W¨urstchen: An efficient architecture for large-scale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022a.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 10684–10695, 2022b.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems (NeurIPS), 35:36479–36494, 2022.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022.

Vikash Sehwag, Xianghao Kong, Jingtao Li, Michael Spranger, and Lingjuan Lyu. Stretching each dollar: Diffusion training from scratch on a micro-budget. arXiv preprint arXiv:2407.15811, 2024.

SG 161222. Realvisxl v5.0. https://civitai.com/models/139562/ realvisxl-v50, 2024.

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of ACL, 2018.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871– 8879, 2024.

Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, Xiangtai Li, and MingHusan Yang. Relationbooth: Towards relation-aware customized object generation. arXiv preprint arXiv:2410.23280, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1921–1930, 2023.

Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. Advances in Neural Information Processing Systems (NeurIPS), 2024a.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023.

Zike Wu, Pan Zhou, Xuanyu Yi, Xiaoding Yuan, and Hanwang Zhang. Consistent3d: Towards consistent high-fidelity text-to-3d generation with deterministic sampling prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9892–9902, 2024b.

Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi. Versatile diffusion: Text, images and variations all in one diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision (CVPR), pp. 7754–7765, 2023.

Xuanyu Yi, Zike Wu, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, and Hanwang Zhang. Diffusion timestep curriculum for one image to 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9948–9958, 2024.

Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024.

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In International Conference on Learning Representations (ICLR), 2022a.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for contentrich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022b.

Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Yue Cao, Xinlong Wang, and Jingjing Liu. Capsfusion: Rethinking image-text data at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14022–14032, 2024.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36, 2024a.

Sixian Zhang, Bohan Wang, Junqiang Wu, Yan Li, Tingting Gao, Di Zhang, and Zhongyuan Wang. Learning multi-dimensional human preference for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8018–8027, 2024b.

Donghao Zhou, Jiancheng Huang, Jinbin Bai, Jiaze Wang, Hao Chen, Guangyong Chen, Xiaowei Hu, and Pheng-Ann Heng. Magictailor: Component-controllable personalization in text-to-image diffusion models. arXiv preprint arXiv:2410.13370, 2024.

Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. Towards language-free training for text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 17907–17917, 2022.

Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024.

