[Figure 1]

[Figure 2]

[Figure 3]

# arXiv:2502.04896v2[cs.CV]10Feb2025

## Goku: Flow Based Video Generative Foundation Models

Shoufa Chen1∗ Chongjian Ge1∗ Yuqi Zhang2 Yida Zhang2 Fengda Zhu2 Hao Yang2 Hongxiang Hao2 Hui Wu2 Zhichao Lai2 Yifei Hu2 Ting-Che Lin2 Shilong Zhang1 Fu Li2 Chuan Li2 Xing Wang2 Yanghua Peng2 Peize Sun1 Ping Luo1 Yi Jiang2 Zehuan Yuan2 Bingyue Peng2 Xiaobing Liu2

1The University of Hong Kong 2Bytedance Inc ∗ Equal Contribution

https://saiyan-world.github.io/goku/

### Abstract

This paper introduces Goku, a state-of-the-art family of joint image-and-video generation models leveraging rectified flow Transformers to achieve industry-leading performance. We detail the foundational elements enabling high-quality visual generation, including the data curation pipeline, model architecture design, flow formulation, and advanced infrastructure for efficient and robust large-scale training. The Goku models demonstrate superior performance in both qualitative and quantitative evaluations, setting new benchmarks across major tasks. Specifically, Goku achieves 0.76 on GenEval and 83.65 on DPG-Bench for text-to-image generation, and 84.85 on VBench for text-to-video tasks. We believe that this work provides valuable insights and practical advancements for the research community in developing joint image-and-video generation models.

#### 1. Introduction

Video generation has garnered significant attention owing to its transformative potential across a wide range of applications, such media content creation (Polyak et al., 2024), advertising (Zhang et al., 2024; Bacher et al., 2021), video games (Yang et al., 2024b; Valevski et al., 2024; Quevedo et al., 2024), and world model simulators (Ha and Schmidhuber, 2018; Brooks et al., 2024; Agarwal et al., 2025). Benefiting from advanced generative algorithms (Goodfellow et al., 2014; Ho et al., 2020; Liu et al., 2023; Lipman et al., 2023), scalable model architectures (Vaswani et al., 2017; Peebles and Xie, 2023), vast amounts of internet-sourced data (Chen et al., 2024b; Nan et al., 2024; Ju et al., 2024), and ongoing expansion of computing capabilities (Corporation, 2022, 2023, 2024), remarkable advancements have been achieved in the field of video generation (Ho et al., 2022b,a; Singer et al., 2023; Blattmann et al., 2023b; Brooks et al., 2024; Kuaishou, 2024; Yang et al., 2024c; Jin et al., 2024; Polyak et al., 2024; Kong et al., 2024; Ji et al., 2024).

In this work, we present Goku, a family of rectified flow (Lipman et al., 2023; Liu et al., 2023) transformer models designed for joint image and video generation, establishing a pathway toward industry-grade performance. This report centers on four key components: data curation, model architecture design, flow formulation, and training infrastructure optimization—each rigorously refined to meet the demands of high-quality, large-scale video generation.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

A glass transparent emoji cartoon hand making the peace sign gesture, with fingers straight up and down

A white bearded man's face emerges from a cloud of white butterflies, background is white

An extremely happy American Cocker Spaniel.

[Figure 8]

[Figure 9]

[Figure 10]

A native Warrior shaman Bengal Cat with a black and white leopard pattern, blue eyes, short fur, and portrait pose, colorful feathers and colorful ornaments, a regal oil-style portrait of the queen of native Kitty shaman white Cat with wings and headdress. Nordic is kind and motherly, it has black eye makeup and her hair is in messy.

An ancient artifact rests on a pedestal, the word “GOKU” etched onto its surface, glowing as if holding a hidden power within.

An enchanted forest with a waterfall cascading over rocks, the word “GOKU” formed by glowing moss along the stone surface, lighting up the misty surroundings.

Goku Black, in Super Saiyan Rose form, stands in a destroyed cityscape. The word "SAIYAN" is etched into the ground with dark energy.

###### (a) Text-to-Image Samples

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A giant panda sitting comfortably at a table, eating a hotpot meal.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

A flock of paper airplanes flutters through a dense jungle, weaving around trees as if they were migrating birds.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

An individual standing in a kitchen, wearing an apron, and holding a frying pan positioned above a burner.

(b) Text-to-Video Samples

- Figure 1 | Generated samples from Goku. Key components are highlighted in RED.

First, we present a comprehensive data processing pipeline designed to construct largescale, high-quality image and video-text datasets. The pipeline integrates multiple advanced techniques, including video and image filtering based on aesthetic scores, OCR-driven content analysis, and subjective evaluations, to ensure exceptional visual and contextual quality. Furthermore, we employ multimodal large language models (MLLMs) (Yuan et al., 2025) to generate dense and contextually aligned captions, which are subsequently refined using an additional large language model (LLM) (Yang et al., 2024a) to enhance their accuracy, fluency, and descriptive richness. As a result, we have curated a robust training dataset comprising approximately 36M video-text pairs and 160M image-text pairs, which are proven sufficient for training industry-level generative models.

Secondly, we take a pioneering step by applying rectified flow formulation (Lipman et al.,

- 2023) for joint image and video generation, implemented through the Goku model family, which comprises Transformer architectures with 2B and 8B parameters. At its core, the Goku

framework employs a 3D joint image-video variational autoencoder (VAE) to compress image and video inputs into a shared latent space, facilitating unified representation. This shared latent space is coupled with a full-attention (Vaswani et al., 2017) mechanism, enabling seamless joint training of image and video. This architecture delivers high-quality, coherent outputs across both images and videos, establishing a unified framework for visual generation tasks.

Furthermore, to support the training of Goku at scale, we have developed a robust infrastructure tailored for large-scale model training. Our approach incorporates advanced parallelism strategies (Jacobs et al., 2023; Zhao et al., 2023) to manage memory efficiently during long-context training. Additionally, we employ ByteCheckpoint (Wan et al., 2024) for high-performance checkpointing and integrate fault-tolerant mechanisms from MegaScale (Jiang et al., 2024) to ensure stability and scalability across large GPU clusters. These optimizations enable Goku to handle the computational and data challenges of generative modeling with exceptional efficiency and reliability.

We evaluate Goku on both text-to-image and text-to-video benchmarks to highlight its competitive advantages. For text-to-image generation, Goku-T2I demonstrates strong performance across multiple benchmarks, including T2I-CompBench (Huang et al., 2023), GenEval (Ghosh et al., 2024), and DPG-Bench (Hu et al., 2024), excelling in both visual quality and text-image alignment. In text-to-video benchmarks, Goku-T2V achieves state-of-the-art performance on the UCF-101 (Soomro et al., 2012) zero-shot generation task. Additionally, Goku-T2V attains an impressive score of 84.85 on VBench (Huang et al., 2024), securing the top position on the leaderboard (as of 2025-01-25) and surpassing several leading commercial text-to-video models. Qualitative results, illustrated in Figure 1, further demonstrate the superior quality of the generated media samples. These findings underscore Goku’s effectiveness in multi-modal generation and its potential as a high-performing solution for both research and commercial applications.

#### 2. Goku: Generative Flow Models for Visual Creation

In this section, we present three core components of Goku, the image-video joint VAE (Yang et al., 2024c), the Goku Transformer architecture, and the rectified flow formulation. These components are designed to work synergistically, forming a cohesive and scalable framework for joint image and video generation. During training, each raw video input 𝑥 ∈ R𝑇×𝐻×𝑊×3 (with images treated as a special case where 𝑇 = 1 ) is encoded from the pixel space to a latent space using a 3D image-video joint VAE (Section 2.1). The encoded latents are then organized into mini-batches containing both video and image representations, facilitating the learning of a unified cross-modal representation. Subsequently, the rectified flow formulation (Section 2.3) is applied to these latents, leveraging a series of Transformer blocks (Section 2.2) to model complex temporal and spatial dependencies effectively.

###### 2.1. Image-Video Joint VAE

Earlier research (He et al., 2022; Rombach et al., 2022; Esser et al., 2021) demonstrates that diffusion and flow-based models can significantly improve efficiency and performance by modeling in latent space through a Variational Auto-Encoder (VAE) (Esser et al., 2021; Kingma, 2013). Inspired by Sora (Brooks et al., 2024), the open-source community has introduced 3D-VAE to explore spatio-temporal compression within latent spaces for video generation tasks (Lab and etc., 2024; Zheng et al., 2024; Yang et al., 2024c). To extend the advantages of latent space modeling across multiple media formats, including images and videos, we adopt a jointly trained Image-Video VAE (Yang et al., 2024c) that handles both image and video data within a

Model Layer Model Dimension FFN Dimension Attention Heads

- Goku-1B 28 1152 4608 16
- Goku-2B 28 1792 7168 28 Goku-8B 40 3072 12288 48

- Table 1 | Architecture configurations for Goku Models. Goku-1B model is only used for pilot experiments in Section 2.3

unified framework. Specifically, for videos, we apply a compression stride of 8 × 8 × 4 across height, width, and temporal dimensions, respectively, while for images, the compression stride is set to 8 × 8 in spatial dimensions.

###### 2.2. Transformer Architectures

The design of the Goku Transformer block builds upon GenTron (Chen et al., 2024a), an extension of the class-conditioned diffusion transformer (Peebles and Xie, 2023) for text-toimage/video tasks. It includes a self-attention module for capturing inter-token correlations, a cross-attention layer to integrate textual conditional embeddings (extracted via the Flan-T5 language model (Chung et al., 2024)), a feed-forward network (FFN) for feature projection, and a layer-wise adaLN-Zero block that incorporates timestep information to guide feature transformations. Additionally, we introduce several recent design enhancements to improve model performance and training stability, as detailed below.

Plain Full Attention. In Transformer-based video generative models, previous approaches (Chen

et al., 2024a; Wu et al., 2023; Singer et al., 2023; Blattmann et al., 2023b) typically combine temporal attention with spatial attention to extend text-to-image generation to video. While this method reduces computational cost, it is sub-optimal for modeling complex temporal motions, as highlighted in prior work (Yang et al., 2024c; Polyak et al., 2024). In Goku, we adopt full attention to model multi-modal tokens (image and video) within a unified network. Given the large number of video tokens remaining after VAE processing—particularly for high-framerate, long-duration videos—we leverage FlashAttention (Shah et al., 2024; Dao, 2024) and sequence parallelism (Li et al., 2021) to optimize both GPU memory usage and computational efficiency.

Patch n’ Pack. To enable joint training on images and videos of varying aspect ratios and lengths, we follow the approach from NaViT (Dehghani et al., 2024), packing both modalities into a single minibatch along the sequence dimension. This method allows flexible mixing of training instances with different sequence lengths into a single batch, eliminating the need for data buckets (Podell et al., 2023).

3D RoPE Position Embedding. Rotary Position Embedding (RoPE) (Su et al., 2024) has demonstrated effectiveness in LLMs by enabling greater sequence length flexibility and reducing inter-token dependencies as relative distances increase. During joint training, we apply 3D RoPE embeddings to image and video tokens. In our joint training framework, we extend 3D RoPE embeddings to image and video tokens, leveraging their extrapolation capability to accommodate varying resolutions. This adaptability makes RoPE particularly suited for

handling diverse resolutions and video lengths. Furthermore, our empirical analysis revealed that RoPE converges faster than sinusoidal positional embeddings during transitions across different training stages

Q-K Normalization. Training large-scale Transformers can occasionally result in loss spikes, which may lead to model corruption, manifesting as severe artifacts or even pure noise in generated images or videos. To mitigate this issue, we incorporate query-key normalization (Dehghani et al., 2023) to stabilize the training process. Specifically, we apply RMSNorm (Zhang and Sennrich, 2019) to each query-key feature prior to attention computation, ensuring smoother and more reliable training dynamics.

The overall Transformer model is constructed by stacking a sequence of blocks as described above. To address varying computational demands and performance requirements, we design three model variants, summarized in Table 1. The Goku-1B model serves as a lightweight option for pilot experiments. The Goku-2B variant consists of 28 layers, each with a model dimension of 1792 and 28 attention heads, providing a balance between computational efficiency and expressive capacity. In contrast, the larger Goku-8B variant features 40 layers, a model dimension of 3072, and 48 attention heads, delivering superior modeling capacity aimed at achieving high generation quality.

###### 2.3. Flow-based Training

Our flow-based formulation is rooted in the rectified flow (RF) algorithm (Albergo and VandenEijnden, 2023; Lipman et al., 2023; Liu et al., 2023), where a sample is progressively transformed from a prior distribution, such as a standard normal distribution, to the target data distribution. This transformation is achieved by defining the forward process as a series of linear interpolations between the prior and target distributions. Specifically, given a real data sample x1 from the target distribution and a noise sample x0 ∼ N(0,1) from the prior distribution, a training example is constructed through linear interpolation:

x𝑡 = 𝑡 · x1 + (1 − 𝑡) · x0, (1)

where 𝑡 ∈ [0,1] represents the interpolation coefficient. The model is trained to predict the velocity, defined as the time derivative of x𝑡, v𝑡 = 𝑑𝑑𝑡x𝑡 , which guides the transformation of intermediate samples x𝑡 towards the real data x1 during inference. By establishing a direct, linear interpolation between data and noise, RF simplifies the modeling process, providing improved theoretical properties, conceptual clarity, and faster convergence across data distributions.

Goku takes a pioneering step by adopting a flow-based formulation for joint image-andvideo generation. We conduct a pilot experiment to validate the rapid convergence of flow-based training by performing class-conditional generation with Goku-1B a model specifically designed for these proof-of-concept experiments, on ImageNet-1K (256 × 256) (Deng et al., 2009). The model is configured with 28 layers, an attention dimension of 1152, and 16 attention heads. To evaluate performance, we compare key metrics, such as FID-50K and Inception Score (IS), for models trained using the denoising diffusion probabilistic model (DDPM) (Ho et al., 2020) and rectified flow. As shown in Table 2, RF demonstrates faster convergence than DDPM. For instance, Goku-1B (RF) achieves a lower FID-50K after 400k training steps compared to Goku-1B (DDPM), which requires 1000k steps to reach a similar level of performance.

Loss Steps FID ↓ sFID ↓ IS ↑ Precision ↑ Recall ↑

DDPM 200k 3.0795 4.3498 226.4783 0.8387 0.5317 DDPM 400k 2.5231 4.3821 265.0612 0.8399 0.5591 DDPM 1000k 2.2568 4.4887 286.5601 0.8319 0.5849

Rectified Flow 200k 2.7472 4.6416 232.3090 0.8239 0.5590 Rectified Flow 400k 2.1572 4.5022 261.1203 0.8210 0.5871

- Table 2 | Proof-of-concept experiments of class-conditional generation on ImageNet 256×256. Rectified flow achieves faster convergency compared to DDPM.

- 2.4. Training Details

Multi-stage Training. Directly optimizing joint image-and-video training poses significant challenges, as the network must simultaneously learn spatial semantics critical for images and temporal motion dynamics essential for videos. To tackle this complexity, we introduce a decomposed, multi-stage training strategy that progressively enhances the model’s capabilities, ensuring effective and robust learning across both modalities.

- • Stage-1: Text-Semantic Pairing. In the initial stage, we focus on establishing a solid understanding of text-to-image relationships by pretraining Goku on text-to-image tasks. This step is critical for grounding the model in basic semantic comprehension, enabling it to learn to associate textual prompts with high-level visual semantics. Through this process, the model develops a reliable capacity for representing visual concepts essential for both image and video generation, such as object attributes, spatial configurations, and contextual coherence.
- • Stage-2: Image-and-video joint learning. Building on the foundational capabilities of text-tosemantic pairing, we extend Goku to joint learning across both image and video data. This stage leverages the unified framework of Goku, which employs a global attention mechanism adaptable to both images and videos. Besides, acquiring a substantial volume of high-quality video data is generally more resource-intensive compared to obtaining a similar amount of high-quality image data. To address this disparity, our framework integrates images and videos into unified token sequences during training, enabling the rich information inherent in high-quality images to enhance the generation of video frames (Chen et al., 2024a). By curating a carefully balanced dataset of images and videos, Goku not only gains the capability to generate both high-quality images and videos but also enhances the visual quality of videos by leveraging the rich information from high-quality image data.
- • Stage-3: Modality-specific finetuning. In the final stage, we fine-tune Goku for each specific modality to further enhance its output quality. For text-to-image generation, we implement image-centric adjustments aimed at producing more visually compelling images. For textto-video generation, we focus on adjustments that improve temporal smoothness, motion continuity, and stability across frames, resulting in realistic and fluid video outputs.

Cascaded Resolution Training. In the second stage of joint training, we adopt a cascade resolution strategy to optimize the learning process. Initially, training is conducted on low-resolution image and video data (288 × 512), enabling the model to efficiently focus on fundamental textsemantic-motion relationships at reduced computational costs. Once these core interactions are well-established, the resolution of the training data is progressively increased, transitioning from 480 × 864 to 720 × 1280. This stepwise resolution enhancement allows Goku to refine its

understanding of intricate details and improve overall image fidelity, ultimately leading to superior generation quality for both images and videos.

###### 2.5. Image-to-Video

To extend Goku for adapting an image as an additional condition for video generation, we employ a widely used strategy by using the first frame of each clip as the reference image (Girdhar et al., 2023; Blattmann et al., 2023a; Yang et al., 2024c). The corresponding image tokens are broadcasted and concatenated with the paired noised video tokens along the channel dimension. To fully leverage the pretrained knowledge during fine-tuning, we introduce a single MLP layer for channel alignment, while preserving the rest of the model architecture identical to Goku-T2V.

#### 3. Infrastructure Optimization

To achieve scalable and efficient training of Goku, we first adopt advanced parallelism strategies (Section 3.1), to handle the challenges of long-context, large-scale models. To further optimize memory usage and balance computation with communication, we implement finegrained Activation Checkpointing (Section 3.2). Additionally, we integrate robust fault tolerance mechanisms from MegaScale, enabling automated fault detection and recovery with minimal disruption (Section 3.3). Finally, ByteCheckpoint is utilized to ensure efficient and scalable saving and loading of training states, supporting flexibility across diverse hardware configurations (Section 3.4). The details of these optimizations are introduced below.

###### 3.1. Model Parallelism Strategies

The substantial model size and the exceptionally long sequence length (exceeding 220K tokens for the longest sequence) necessitate the adoption of multiple parallelism strategies to ensure efficient training. Specifically, we employ 3D parallelism to achieve scalability across three axes: input sequences, data, and model parameters.

Sequence-Parallelism (SP) (Korthikanti et al., 2023; Li et al., 2021; Jacobs et al., 2023) slices the input across the sequence dimension for independent layers (e.g., LayerNorm) to eliminate redundant computations, reduce memory usage, and support padding for non-conforming input. We adopt Ulysses (Jacobs et al., 2023) as our implementation, which shards samples across the sequence parallel group from the start of the training loop. During attention computation, it uses all-to-all communication to distribute query, key, and value shards, allowing each worker to process the full sequence but only a subset of attention heads. After parallel computation of attention heads, another all-to-all communication aggregates the results, recombining all heads and the sharded sequence dimension.

Fully Sharded Data Parallelism (FSDP) (Zhao et al., 2023) partitions all parameters, gradients and optimizer states across the data parallel ranks. Instead of all-reduce in Distributed Data Parallelism, FSDP performs all-gather for parameters and reduce-scatter for gradients, enabling overlap with forward and backward computations to potentially reduce communication overhead. In our case, we adopt the HYBRID_SHARD strategy, which combines FULL_SHARD within a shard group and parameter replication across such groups, which effectively implements data parallelism (DP). This approach minimizes communication costs by limiting all-gather and reduce-scatter operations.

###### 3.2. Activation Checkpointing

While the parallelism methods discussed in Section 3.1 provide significant memory savings and enable large-scaling training with long sequences, they inevitably introduce communication overhead among ranks, which can lead to suboptimal overall performance. To address this issue and better balance the computation and communication by maximizing their overlap in the profiling trace, we designed a fine-grained Activation Checkpointing (AC) (Chen et al., 2016) strategy. Specifically, we implemented selective activation checkpointing to minimize the number of layers requiring activation storage while maximizing GPU utilization.

###### 3.3. Cluster Fault Tolerance

Scaling Goku training to large-scale GPU clusters inevitably introduces fault scenarios, which can reduce training efficiency. The likelihood of encountering failures increases with the number of nodes, as larger systems have a higher probability of at least one node failing. These disruptions can extend training time and increase costs. To enhance stability and efficiency at scale, we adopted fault tolerance techniques from MegaScale (Jiang et al., 2024), including self-check diagnostics, multi-level monitoring, and fast restart/recovery mechanisms. These strategies effectively mitigate the impact of interruptions, enabling Goku to maintain robust performance in large-scale generative modeling tasks.

###### 3.4. Saving and Loading Training Stages

Checkpointing training states—such as model parameters, exponential moving average (EMA) parameters, optimizer states, and random states—is crucial for training large-scale models, particularly given the increased likelihood of cluster faults. Reloading checkpointed states ensures reproducibility, which is essential for model reliability and debugging potential issues, including those caused by unintentional errors or malicious attacks.

To support scalable large-scale training, we adopt ByteCheckpoint (Wan et al., 2024) as our checkpointing solution. It not only enables parallel saving and loading of partitioned checkpoints with high I/O efficiency but also supports resharding distributed checkpoints. This flexibility allows seamless switching between different training scales, accommodating varying numbers of ranks and diverse storage backends. In our setup, checkpointing an 8B model across over thousands of GPUs blocks training for less than 4 seconds, which is negligible compared to the overall forward and backward computation time per iteration.

#### 4. Data Curation Pipeline

We unblock the data volume that is utilized for industry-grade video/image generation models. Our data curation pipeline, illustrated in Figure 2, consists of five main stages: (1) image and video collection, (2) video extraction and clipping, (3) image and video filtering, (4) captioning, and (5) data distribution balancing. We describe the details of data curation procedure below.

###### 4.1. Data Overview

We collet raw image and video data from a variety of sources, including publicly available academic datasets, internet resources, and proprietary datasets obtained through partnerships with collaborating organizations. After rigorous filtering, the final training dataset for Goku consists of approximately 160M image-text pairs and 36M video-text pairs, encompassing both

[Figure 27]

FFmpeg Error

|Raw Video|
|---|

Video Tag

Aesthetic Score OCR

Read Video Info

long-tail

DINOv2 Similarity

Motion Score

Background

Extract Clips

[Figure 28]

|Raw Image|
|---|

Caption

NSFW Video Filtering

Keyframe per Second

uniform

|Collection|Extraction|Filtering|Captioning|Balancing|
|---|---|---|---|---|

- Figure 2 | The data curation pipeline in Goku. Given a large volume of video/image data collected from Internet, we generate high-quality video/image-text pairs through a series of data filtering, captioning and balancing steps.

publicly available datasets and internally curated proprietary datasets. The detailed composition of these resources is outlined as follows:

- • Text-to-Image Data. Our text-to-image training dataset includes 100M public samples from LAION (Schuhmann et al., 2022) and 60M high-quality, internal samples. We use public data for pre-training and internal data for fine-tuning.
- • Text-to-Video Data. Our T2V training dataset includes 11M public clips and 25M in-house clips. The former include Panda-70M (Chen et al., 2024b), InternVid (Wang et al., 2023b), OpenVid-1M (Nan et al., 2024), and Pexels (Lab and etc., 2024). Rather than directly using these datasets, we apply a data curation pipeline to keep high-quality samples.

###### 4.2. Data Processing and Filtering

To construct a high-quality video dataset, we implement a comprehensive processing pipeline comprising several key stages. Raw videos are first preprocessed and standardized to address inconsistencies in encoding formats, durations, and frame rates. Next, a two-stage video clipping method segments videos into meaningful and diverse clips of consistent length. Additional filtering processes are applied, including visual aesthetic filtering to retain photorealistic and visually rich clips, OCR filtering to exclude videos with excessive text, and motion filtering to ensure balanced motion dynamics. In addition, the multi-level training data is segmented based on resolution and corresponding filtering thresholds for DINO similarity, aesthetic score, OCR text coverage, and motion score, as summarized in Table 4. We provide the details of each processing step as follows.

Table 3 presents the key parameters and their corresponding thresholds used for video quality assessment. Each parameter is essential in ensuring the generation and evaluation of high-quality videos. The Duration parameter specifies that raw video lengths should be at least 4 seconds to capture meaningful temporal dynamics. The Resolution criterion ensures that the minimum dimension (either height or width) of the video is no less than 480 pixels, maintaining adequate visual clarity. The Bitrate, which determines the amount of data processed per second during playback, requires a minimum of 500 kbps to ensure sufficient quality, clarity, and manageable file size. Videos with low bitrate typically correspond to content with low complexity, such as static videos or those featuring pure color backgrounds. Finally, the Frame Rate enforces a standard of at least 24 frames per second (film standard) or 23.976 frames

Parameter Description Threshold Duration Raw video length ≥ 4 seconds Resolution Width and height of the video 𝑚𝑖𝑛{ height, width} ≥ 480

Amount of data processed per second during playback, which impacts the video’s quality, clarity, and file size

Bitrate

≥ 500 kbps

Frame Rate Frames displayed per second ≥ 24 FPS (Film Standard) / 23.976 FPS (NTSC Standard)

- Table 3 | Summary of video quality parameters and their thresholds for preprocessing. The table outlines the criteria used to filter and standardize raw videos based on essential attributes, ensuring uniformity and compatibility in the dataset.

per second (NTSC standard) to guarantee smooth motion and prevent visual artifacts. These thresholds collectively establish a baseline for evaluating and generating high-quality video content.

- • Preprocessing and Standardization of Raw Videos. Videos collected from the internet often require extensive preprocessing to address variations in encoding formats, durations, and frame rates. Initially, we perform a primary filtering step based on fundamental video attributes such as duration, resolution, bitrate. The specific filtering criteria and corresponding thresholds are detailed in Table 3. This initial filtering step is computationally efficient compared to more advanced, model-based filtering approaches, such as aesthetic (Schuhmann

- et al., 2022) evaluation models. Following this stage, the raw videos are standardized to a consistent coding format, H.264 (Wiegand et al., 2003), ensuring uniformity across the dataset and facilitating subsequent processing stages.

- • Video Clips Extraction. We employ a two-stage video clipping method for this stage. First, we use PySceneDetect (Castellano, 2024) for shot boundary detection, resulting coarse-grained video clips from raw videos. Next, we further refine the video clips by sampling one frame per second, generating DINOv2 (Oquab et al., 2023) features and calculating cosine similarity between adjacent frames. When similarity falls below a set threshold, we mark a shot change and further divide the clip. Specifically, as shown in Table 4, for video resolutions around 480 × 864, we segmented the video clips where the DINO similarity between adjacent frames exceeds 0.85. For resolutions greater than 720 × 1280, the threshold is set at 0.9. Besides, to standardize length, we limit clips to a maximum of 10 seconds. Furthermore, we consider the similarity between different clips derived from the same source video to ensure diversity and maintain quality. Specifically, we compute the perceptual hashing (Contributors, 2013) values of keyframes from each clip and compare them. If two clips have similar hash values, indicating significant overlap, we retain the clip with a higher aesthetic score. This ensures that the final dataset includes diverse and high-quality video clips.
- • Visual Aesthetic Filtering. To assess the visual quality of the videos, we utilize aesthetic models (Schuhmann et al., 2022) to evaluate the keyframes. The aesthetic scores of the keyframes are averaged to obtain an overall aesthetic score for each video. For videos with resolutions around 480 × 864, those with an aesthetic score below 4.3 are discarded, while for resolutions exceeding 720 × 1280, the threshold is raised to 4.5. This filtering process ensures that the selected clips are photorealistic, visually rich, and of high aesthetic quality.

Stage Amount Resolution DINO-Sim. Aesthetic OCR Motion

480p 36M ≥ 480×864 ≥0.85 ≥ 4.3 <= 0.02 0.3 ≤ score ≤ 20.0 720p 24M ≥ 720×1280 ≥0.90 ≥ 4.5 <= 0.01 0.5 ≤ score ≤ 15.0

1080p 7M ≥ 1080×1920 ≥0.90 ≥ 4.5 <= 0.01 0.5 ≤ score ≤ 8.0

- Table 4 | Overview of multi-stage training data.This table summarizes the thresholds for each filtering criterion, including resolution, DINO similarity, aesthetic score, OCR text coverage, motion score, and the corresponding data quantities.

- • OCR Filtering. To exclude videos with excessive text, we employ an internal OCR model to detect text within the keyframes. The OCR model identifies text regions, and we calculate the text coverage ratio by dividing the area of the largest bounding box detected by the total area of the keyframe. Videos with a text coverage ratio exceeding predefined thresholds are discarded. Specifically, for videos with resolutions around 480 × 864, the threshold is set at 0.02, while for resolutions exceeding 720 × 1280, the threshold is reduced to 0.01. This process effectively filters out videos with excessive text content.
- • Motion Filtering. Unlike images, videos require additional filtering based on motion characteristics. To achieve this, we utilize RAFT (Teed and Deng, 2020) to compute the mean optical flow of video clips, which is then used to derive a motion score. For videos with resolutions around 480 × 864, clips with motion scores below 0.3 (indicating low motion) or above 20.0 (indicating excessive motion) are excluded. For resolutions exceeding 720 × 1280, the thresholds are adjusted to 0.5 and 15.0, respectively. Furthermore, to enhance motion control, the motion score is appended to each caption.

###### 4.3. Captioning

Detailed captions are essential for enabling the model to generate text-aligned images/videos precisely. For images, we use InternVL2.0 (Chen et al., 2024c) to generate dense captions for each sample. To caption video clips, we start with InternVL2.0 (Chen et al., 2024c) for keyframe captions, followed by Tarsier2 (Yuan et al., 2025) for video-wide captions. Note that the Tarsier2 model can inherently describe camera motion types (e.g., zoom in, pan right) in videos, eliminating the need for a separate prediction model and simplifying the overall pipeline compared to previous work such as (Polyak et al., 2024). Next, we utilize Qwen2 (Yang et al., 2024a) to merge the keyframe and video captions. Besides, we also empirically found that adding the motion score (calculated by RAFT (Teed and Deng, 2020)) to the captions improves motion control for video generation. This approach enables users to specify different motion scores in prompts to guide the model in generating videos with varied motion dynamics.

###### 4.4. Training Data Balancing

The model’s performance are significantly influenced by the data distribution, especially for video data. To balance the video training data, we first use an internal video classification model to generate semantic tags for the videos. We then adjust the data distribution based on these semantic tags to ensure a balanced representation across categories.

• Data Semantic Distribution. The video classification model assigns a semantic tag to each video based on four evenly sampled keyframes. The model categorizes videos into 9 primary

[Figure 29]

(a) Semantic distribution of video clips.

[Figure 30]

[Figure 31]

Sub-category from Human half-selfie

Sub-category from Scenery

forest

natural

rivers

multi

snow

human full-selfie

grass

sky

(b) The balanced semantic distribution of subcategories.

- Figure 3 | Training data distributions. The balanced semantic distribution of primary categories and subcategories are shown in (a) and (b), respectively.

classes (e.g., human, scenery, animals, food) and 86 subcategories (e.g., half-selfie, kid, dinner, wedding). Figure 3a presents the semantic distribution across our filtered training clips, with humans, scenery, food, urban life, and animals as the predominant categories.

• Data Balancing. The quality of the generated videos is closely tied to the semantic distribution of the training data. Videos involving humans pose greater modeling challenges due to the extensive diversity in appearances, whereas animals and landscapes exhibit more visual consistency and are relatively easier to model. To address this disparity, we implement a data-balancing strategy that emphasizes human-related content while ensuring equitable representation across subcategories within each primary category. Overrepresented subcategories are selectively down-sampled, whereas underrepresented ones are augmented through artificial data generation and oversampling techniques. Balanced data distribution is shown in Figure 3b.

GenEval T2I-CompBench DPG-Bench

Method

Text Enc. Overall Color Shape Texture Average SDv1.5 (Rombach et al., 2022) CLIP ViT-L/14 0.43 0.3730 0.3646 0.4219 63.18

- DALL-E 2 (Ramesh et al., 2022) CLIP ViT-H/16 0.52 0.5750 0.5464 0.6374 SDv2.1 (Rombach et al., 2022) CLIP ViT-H/14 0.50 0.5694 0.4495 0.4982 SDX (Podell et al., 2023) CLIP ViT-bigG 0.55 0.6369 0.5408 0.5637 74.65 PixArt-𝛼 (Chen et al., 2023) Flan-T5-XXL 0.48 0.6886 0.5582 0.7044 71.11

- DALL-E 3 (Betker et al., 2023) Flan-T5-XXL 0.67† 0.8110† 0.6750† 0.8070† 83.50† GenTron (Chen et al., 2024a) CLIP T5XXL - 0.7674 0.5700 0.7150 SD3 (Esser et al., 2024) Flan-T5-XXL 0.74 - - - Show-o (Xie et al., 2024) Phi-1.5 0.53 - - - Transfusion (Zhou et al., 2024) - 0.63 - - - Chameleon (Lu et al., 2024) - 0.39 - - - LlamaGen (Sun et al., 2024) FLAN-T5 XL 0.32 - - - Emu 3 (Wang et al., 2024b) - 0.66† 0.7913† 0.5846† 0.7422† 80.60

Goku-T2I (2B)

0.70 0.7521 0.4832 0.6691

83.65 Goku-T2I (2B)† 0.76† 0.7561† 0.5759† 0.7071†

FLAN-T5 XL

- Table 5 | Comparison with state-of-the-art models on image generation benchmarks. We evaluate on GenEval (Ghosh et al., 2024); T2I-CompBench (Huang et al., 2023) and DPGBench (Hu et al., 2024). Following (Wang et al., 2024b), we use † to indicate the result with prompt rewriting.

#### 5. Experiments

###### 5.1. Text-to-Image Results

we conduct a comprehensive quantitative evaluation of Goku-T2I on widely recognized image generation benchmarks, including GenEval (Ghosh et al., 2024), T2I-CompBench (Huang

- et al., 2023), and DPG-Bench (Hu et al., 2024). Details of these benchmarks could be found in Appendix Appendix A. The results are summarized in Table 5.

Performance on GenEval. To assess text-image alignment comprehensively, we employ the GenEval benchmark, which evaluates the correspondence between textual descriptions and visual content. Since Goku-T2I is primarily trained on dense generative captions, it exhibits a natural advantage when handling detailed prompts. To further explore this, we expand the original short prompts in GenEval with ChatGPT-4o, preserving their semantics while enhancing descriptive detail. As shown in Table 5, Goku-T2I achieves strong performance with the original short prompts, surpassing most state-of-the-art models. With the rewritten prompts, Goku-T2I attains the highest score (0.76), demonstrating its exceptional capability in aligning detailed textual descriptions with generated images.

Performance on T2I-CompBench. We further evaluate the alignment between generated images and textual conditions using the T2I-CompBench benchmark, which focuses on various object attributes such as color, shape, and texture. As illustrated in Table 5, Goku-T2I consistently

|Method|Resolution FVD (↓) IS (↑ )<br><br>|
|---|---|
|CogVideo (Chinese) (Hong et al., 2022) CogVideo (English) (Hong et al., 2022) Make-A-Video (Singer et al., 2023) VideoLDM (Blattmann et al., 2023b) LVDM (He et al., 2022) MagicVideo (Zhou et al., 2022) PixelDance (Zeng et al., 2024) PYOCO (Ge et al., 2023) Emu-Video (Girdhar et al., 2023) SVD (Blattmann et al., 2023a)<br><br>|480×480 751.34 23.55 480×480 701.59 25.27 256×256 367.23 33.00<br><br>- 550.61 33.45 256×256 372.00 -<br><br>- 655.00 -<br>- 242.82 42.10<br>- 355.19 47.76<br><br><br>256×256 317.10 42.7 240×360 242.02 -|
|Goku-2B (ours) Goku-2B (ours) Goku-2B (ours)<br><br>|256×256 246.17 45.77 ± 1.10 240×360 254.47 46.64 ± 1.08 128×128 217.24 42.30 ± 1.03|

- Table 6 | Zero-shot text-to-video performance on UCF-101. We generate videos of different resolutions, including 256×256, 240×360, 128×128, for comprehensive comparisons.

outperforms several strong baselines, including PixArt-𝛼 (Chen et al., 2023), SDXL (Podell et al.,

- 2023), and DALL-E 2 (Mishkin et al., 2022). Notably, the inclusion of prompt rewriting leads to improved performance across all attributes, further highlighting Goku-T2I’s robustness in text-image alignment.

Performance on DPG-Bench. While the aforementioned benchmarks primarily evaluate textimage alignment with short prompts, DPG-Bench is designed to test model performance on dense prompt following. This challenging benchmark includes 1,000 detailed prompts, providing a rigorous test of a model’s ability to generate visually accurate outputs for complex textual inputs. As shown in the last column of Table 5, Goku-T2I achieves the highest performance with an average score of 83.65, surpassing PixArt-𝛼 (Chen et al., 2023) (71.11), DALL-E 3 (Betker et al., 2023) (83.50), and EMU3 (Wang et al., 2024b) (80.60). These results highlight Goku-T2I’s superior ability to handle dense prompts and maintain high fidelity in text-image alignment.

###### 5.2. Text-to-Video Results

Performance on UCF-101. We conduct experiments on UCF-101 (Soomro et al., 2012) using zero-shot text-to-video setting. As UCF-101 only has class labels, we utilize an video-language model, Tarsier-34B (Wang et al., 2024a), to generate detailed captions for all UCF-101 videos. These captions are then used to synthesize videos with Goku. Finally, we generated 13,320 videos at different resolutions with Goku-2B model for evaluation, including 256×256, 240×360 and 128×128. Following standard practice (Skorokhodov et al., 2022), we use the I3D model, pre-trained on Kinetics-400 (Carreira and Zisserman, 2017), as the feature extractor. Based on the extracted features, we calculated Fréchet Video Distance (FVD) (Unterthiner et al., 2018) to evaluate the fidelity of the generated videos. The results in Table 6 demonstrate that Goku consistently generates videos with lower FVD and higher IS. For instance, at a resolution of 128×128, the FVD of videos generated by Goku is 217.24, achieving state-of-the-art performance and highlighting significant advantages over other methods.

Human

Dynamic Multiple Appear. Quality Semantic

Models

Scene

Overall Action Degree Objects Style Score Score

AnimateDiff-V2 92.60 50.19 40.83 36.88 22.42 82.90 69.75 80.27 VideoCrafter-2.0 95.00 55.29 42.50 40.66 25.13 82.20 73.42 80.44 OpenSora V1.2 85.80 42.47 47.22 58.41 23.89 80.71 73.30 79.23 Show-1 95.60 47.03 44.44 45.47 23.06 80.42 72.98 78.93 Gen-3 96.40 54.57 60.14 53.64 24.31 84.11 75.17 82.32 Pika-1.0 86.20 49.83 47.50 43.08 22.26 82.92 71.77 80.69 CogVideoX-5B 99.40 53.20 70.97 62.11 24.91 82.75 77.04 81.61 Kling 93.40 50.86 46.94 68.05 19.62 83.39 75.68 81.85 Mira 63.80 16.34 60.33 12.52 21.89 78.78 44.21 71.87 CausVid 99.80 56.58 92.69 72.15 24.27 85.65 78.75 84.27 Luma 96.40 58.98 44.26 82.63 24.66 83.47 84.17 83.61 HunyuanVideo 94.40 53.88 70.83 68.55 19.80 85.09 75.82 83.24

Goku (ours) 97.60 57.08 76.11 79.48 23.08 85.60 81.87 84.85

- Table 7 | Comparison with leading T2V models on VBench. Goku achieves state-of-the-art overall performance. Detailed results across all 16 evaluation dimensions are provided in Table 8 in the Appendix.

Performance on VBench. As presented in Table 7, we evaluate Goku-T2V against state-ofthe-art models on VBench (Huang et al., 2024), a comprehensive benchmark designed to assess video generation quality across 16 dimensions. Goku-T2V achieves state-of-the-art overall performance on VBench, showcasing its ability to generate high-quality videos across diverse attributes and scenarios.

Among the key metrics, Goku-T2V demonstrates notable strength in human action representation, dynamic degree, and multiple object generation, reflecting its capacity for handling complex and diverse video content. Additionally, it achieves competitive results in appearance style, quality score, and semantic alignment, highlighting its balanced performance across multiple aspects.

For detailed results on all 16 evaluation dimensions, we refer readers to Table 8 in the Appendix. This comprehensive analysis underscores Goku-T2V’s superiority in video generation compared to prior approaches.

###### 5.3. Image-to-Video

We finetune Goku-I2V from the T2V initialization with approximate 4.5M text-image-video triplets, sourced from diverse domains to ensure robust generalization. Despite the relatively small number of fine-tuning steps (10k), our model demonstrates remarkable efficiency in animating reference image while maintaining strong alignment with the accompanying text. As illustrated in Figure 4, the generated videos exhibit high visual quality and temporal coherence, effectively capturing the semantic nuances described in the text.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

A lion running towards the left side of the scene, with flames engulfing its body. As it runs, the lion gradually transforms into a mass of flames…

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

A woman in workout gear is lifting weights at a gym, her biceps flexing with each lift, sweat visible on her forehead, with a closeup on her determined expression…

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

A man surfing on a wave, with the camera following his movement and focusing on his face. He is smiling and giving a thumbs-up to the camera, …

- Figure 4 | Samples of Goku-I2V. Reference images are presented in the leftmost columns. We omitted redundant information from the long prompts, displaying only the key details in each one. Key words are highlighted in RED.

- 5.4. Image and Video Qualitative Visualizations

For intuitive comparisons, we conduct qualitative assessments and present sampled results in Figure 6. The evaluation includes open-source models, such as CogVideoX (Yang et al., 2024c) and Open-Sora-Plan (Zheng et al., 2024), alongside closed-source commercial products, including DreamMachine (Luma, 2024), Pika (pika, 2024), Vidu (Bao et al., 2024), and Kling (Kuaishou, 2024). The results reveal that some commercial models struggle to generate critical video elements when handling complex prompts. For instance, models like Pika, DreamMachine, and Vidu (rows 3–5) fail to render the skimming drone over water. While certain models succeed in generating the target drone, they often produce distorted subjects (rows 1–2) or static frames lacking motion consistency (row 6). In contrast, Goku-T2V (8B) demonstrates superior performance by accurately incorporating all details from the prompt, creating a coherent visual output with smooth motion. Additional comparisons are provided in the appendix for a more comprehensive evaluation. Furthermore, more video examples are available at the goku homepage.

- 5.5. Ablation Studies

Model Scaling. We compared Goku-T2V models with 2B and 8B parameters. Results in

- Figure 5a indicate that model scaling helps mitigate the generation of distorted object structures, such as the arm in Figure 5a (row 1) and the wheel in Figure 5a (row 2). This aligns with findings observed in large multi-modality models.

Joint Training. We further examine the impact of joint image-and-video training. Starting from the same pretrained Goku-T2I (8B) weights, we fine-tuned Goku-T2V (8B) on 480p videos for an equal number of training steps, with and without joint image-and-video training. As shown in Figure 5b, Goku-T2V without joint training tends to generate low-quality video frames, while the model with joint training more consistently produces photorealistic frames.

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

GOKU-T2V(2B) GOKU-T2V(8B)

- (a) Model Scaling

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

GOKU-T2V w/o Joint Training GOKU-T2V w/ Joint Training

- (b) Joint Training

- Figure 5 | Ablation Studies of Model Scaling and Joint Training. Fig. (a) shows the comparison between Goku-T2V(2B) and Goku-T2V(8B). Fig. (b) shows the comparison between whether joint training is adopted or not.

#### 6. Conclusion

In this work, we presented Goku, a novel model for joint image-and-video generation for industry-standard performance. Through an advanced data curation process and a robust model architecture, Goku delivers high-quality outputs by ensuring both fine-grained data selection and effective integration of image and video modalities. Key components, such as the image-video joint VAE and the application of rectified flow, facilitate seamless token interaction across modalities, establishing a shared latent space that enhances model adaptability and attention across tokens. Empirical results highlight Goku’s superiority in commercial-grade visual generation quality.

###### Acknowledgements

We sincerely appreciate the support of our collaborators at ByteDance who contributed to this work. Xibin Wu, Chongxi Wang, Yina Tang, Fangzhou Ai, Yi Ren, Wei Wang, Chen Chen, Colin Young, Bobo Zeng, Ge Bai, Yi Fu, Ruoyu Guo, Prasanna Raghav, Weiguo Feng, Xugang Ye, Adithya Sampath, Aaron Shen, Da Tang, Yuan Fang, Qijun Gan, Chen Zhang, Zhenhui Ye, Pan Xie, Houmin Wei, Gaohong Liu, Zherui Liu, Chenyuan Wang, Yun Zhang, Kaihua Jiang, Zhuo Jiang, Yang Bai, Weiqiang Lou, Hongkai Li, Xi Yang, Shuguang Wang, Junru Zheng, Zuquan Song, Zixian Du, Jingzhe Tang, Yongqiang Zhang, Mingji Han, Heng Zhang, Li Han, Sophie Xie, Shuo Li, Xinzhi Yao, Peng Li, Lianke Qin, Dongyang Wang, Yang Cheng, Chundian Liu, Wenhao Hao, Haibin Lin, Xin Liu

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

GOKU(8B)ViduKling(1.5)DreamMachineOpen-soraPlan(v1.3)PikaCogVideoX1.5(5B)

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

Prompt：Gliding through a crystal-clear coral reef, the drone skims just above the vibrant marine life below. Brightly colored corals, schools of fish, and rays of sunlight penetrating the water’s surface all contribute to the serene yet fast-paced journey. The scene showcases the beauty of the underwater world, as the drone swiftly maneuvers through coral arches and narrow underwater channels.

- Figure 6 | Qualitative comparisons with state-of-the-art (SoTA) video generation models. This figure showcases comparisons with leading models, including (Yang et al., 2024c), Open-Sora Plan (Lab and etc., 2024), Pika (pika, 2024), DreamMachine (Luma, 2024), Vidu (Bao et al., 2024), and Kling v1.5 (Kuaishou, 2024).

#### Appendix A. Benchmark Configurations

T2I-Compbench (Huang et al., 2023) We evaluate the alignment between the generated images and text conditions using T2I-Compbench, a comprehensive benchmark for assessing compositional text-to-image generation capabilities. Specifically, we report scores for color binding, shape binding, and texture binding. To evaluate these results, we employ the Disentangled BLIP-VQA model. For each attribute, we generate 10 images per prompt, with a total of 300 prompts in each category.

GenEval (Ghosh et al., 2024) GenEval is an object-focused framework designed to evaluate compositional image properties, such as object co-occurrence, position, count, and color. For evaluation, we generate a total of 2,212 images across 553 prompts. The final score is reported as the average across tasks.

DPG-Bench (Hu et al., 2024) Compared to the aforementioned benchmarks, DPGBench offers longer prompts with more detailed information, making it effective for evaluating compositional generation in text-to-image models. For this evaluation, we generate a total of 4,260 images across 1,065 prompts, with the final score reported as the average across tasks.

VBench (Huang et al., 2024) VBench is a benchmark suite for evaluating video generative models. It provides a structured Evaluation Dimension Suite that breaks down “video generation quality" into precise dimensions for detailed assessment. Each dimension and content category includes a carefully crafted Prompt Suite and samples Generated Videos from various models.

#### Appendix B. More Visualization Examples

###### Appendix B.1. Goku-T2I Samples Visualization

We present more generated image samples with their text prompts in Figure 7. The prompts are randomly selected from the Internet 1. Goku-T2I achieves strong performance in both visual quality and text-image alignment. It can interpret visual elements and their interactions from complex natural language descriptions. Notably, in Figure 8, Goku-T2I exhibits impressive abilities on generating images with rich details, for example, the clear textures of leaves and berries.

###### Appendix B.2. Goku-T2V Samples Visualization

In Figure 9 we show more examples generated by Goku-T2V, in both landscape (e.g., rows one through five) and portrait mode (e.g., the last row). Goku-T2V is capable of generating highmotion videos (e.g., skiing) and realistic scenes (e.g., forests). All videos are configured with a duration of 4 seconds, a frame rate of 24 FPS, and a resolution of 720p. For visualization, we uniformly sample five frames in temporal sequence.

###### Appendix B.3. Goku-T2V Comparisons with Prior Arts

Additional comparisons with state-of-the-art text-to-video generation models are presented in Figure 10 and Figure 11. These results demonstrate the strong performance of Goku when evaluated against both open-source models (Yang et al., 2024c; Zheng et al., 2024) and commercial products (pika, 2024; Kuaishou, 2024; Bao et al., 2024; Luma, 2024). For instance, in Figure 11, Goku successfully generates smooth motion and accurately incorporates the specified low-angle shot. In contrast, other models, such as CogVideoX (Yang et al., 2024c), Vidu (Bao et al., 2024), and Kling (Kuaishou, 2024), often produce incorrect objects or improper camera views.

###### Appendix B.4. Goku-I2V Samples Visualization

We present additional visualization of generated samples from Goku-I2V in Figure 12, which further validate the effectiveness and versatility of our approach. As shown in the figure, GokuI2V demonstrates an impressive ability to synthesize coherent and visually compelling videos from diverse reference images, maintaining consistency in motion and scene semantics.

For instance, in the first row, the model successfully captures the dynamic and high-energy nature of water boxing, generating fluid and natural movements of splashes synchronized with the subject’s motions. In the second row, the sequence of a child riding a bike through a park illustrates the model’s proficiency in creating smooth and realistic forward motion while preserving environmental consistency. Finally, the third row showcases the model’s ability to handle creative and imaginative scenarios, as seen in the detailed depiction of pirate ships battling atop a swirling coffee cup. The photorealistic rendering and accurate motion trajectories underscore the model’s robustness in both realism and creativity.

These examples highlight Goku-I2V’s capacity to generalize across a wide range of inputs, reinforcing its potential for applications in video generation tasks requiring high fidelity and adaptability.

1https://promptlibrary.org/

overallconsistency

AnimateDiff-V280.2782.9069.7595.3097.6898.7597.7640.8367.1670.1090.9036.8892.6087.4734.6050.1922.4226.0327.04

OpenSoraV1.279.2380.7173.3094.4597.9099.4798.2047.2256.1860.9483.3758.4185.8087.4967.5142.4723.8924.5527.07

Show-178.9380.4272.9895.5398.0299.1298.2444.4457.3558.6693.0745.4795.6086.3553.5047.0323.0625.2827.46

Gen-382.3284.1175.1797.1096.6298.6199.2360.1463.3466.8287.8153.6496.4080.9065.0954.5724.3124.7126.69

CogVideoX-5B81.6182.7577.0496.2396.5298.6696.9270.9761.9862.9085.2362.1199.4082.8166.3553.2024.9125.3827.59

Mira71.8778.7844.2196.2396.9298.2997.5460.3342.5160.1652.0612.5263.8042.2427.8316.3421.8918.7718.72

HunyuanVideo83.2485.0975.8297.3797.7699.4498.9970.8360.3667.5686.1068.5594.4091.6068.6853.8819.8023.8926.44

Kling81.8583.3975.6897.6099.3099.4046.9461.2165.6287.2468.0593.4089.9073.0350.8619.6224.1726.4298.33

Pika-1.080.6982.9271.7796.9497.3647.5062.0461.8788.7243.0886.2090.5761.0349.8322.2624.2225.9499.7499.50

Luma83.6183.4797.3397.4398.6499.3544.2665.5166.5596.4092.3383.6724.6628.1384.1794.9582.6358.9826.29

VideoCrafter-2.080.4482.2073.4296.8598.4197.7342.5063.1367.2292.5540.6695.0035.8655.2925.8498.2292.9225.1328.23

Goku85.6081.8795.5596.6797.7198.5076.1194.4079.4897.6083.8157.0823.0825.6427.3584.8567.2271.2985.72

CausVid84.2778.7597.5397.1996.2498.0564.1568.8892.9972.1580.1764.6556.5824.2725.3327.5185.6592.6999.80

temporalstyle

appearancestyle

scene

spatialrelationship

color

humanaction

multipleobjects

objectclass

imagingquality

aestheticquality

dynamicdegree

motionsmoothness

temporalflickering

backgroundconsistency

subjectconsistency

SemanticScore

QualityScore

TotalScore

Method

comparewithGen-3(Runway,2023),Vchitect-2.0(Team,2024),VEnhancer(Heetal.,2024),Kling(Kuaishou,2024),LaVie-2(Wangetal.,

Table8WeevaluateonVBench(Huangetal.,2024)andComparisonwithstate-of-the-artmodelsonvideogenerationbenchmarks.|

2023a),CogVideoX(Yangetal.,2024c),Emu3(Wangetal.,2024b).

[Figure 102]

[Figure 103]

An embroidered sweater with an anatomical illustration of the human torso and chest, the skin is open to reveal the internal anatomy.

[Figure 104]

A portrait featuring a 26-year-old Chinese male model in a six-grid layout. He has a sleek, naturally layered Korean hairstyle with subtly drooping bangs. Each panel shows him wearing modern

Prototype flying fox made from blown glass, Lino Tagliapietra style Muranese glassmaking, intricate details.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Close up shot of hand of a woman touching oats in oat farm. Shot from behind.

3D illustration of the chip with text "AI" floating above it, with a blue color scheme.

A simple design in black on a white background. The word "VINTAGE" is at the bottom.

3d cube woman underwater, iridescent water, dreamlike

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Great Dane Dog sitting on a toilet bowl in wide bathroom, reading a large double page spread newspaper, sit like human. The background is in a white room.

Full body shot of balenciaga fashion model and parrot hybrid with a human body and the head of the parrot. He is walking through a podium like a model.

Full body photo of a screaming cauliflower monster roaring towards the viewer, very detailed textures. The background is clean and blue.

Create realistic playing cards on fire. The playing cards are presented with 4A. The fire is red and intense. The background is black.

###### Figure 7 | Qualitative samples of Goku-T2I. Key words are highlighted in RED.

[Figure 113]

[Figure 114]

[Figure 115]

| |
|---|

[Figure 116]

[Figure 117]

| |
|---|

Prompt：Raspberry in the form of women walk along the path of a fairy tale forest. She carries a jug of water with her. Her head is made of one big raspberry on which she has big and beautiful eyes, as well as nose and mouth. The skin of the face has a raspberry color. She has very beautiful hair which consists of raspberry, leaves and thin stems. Her arms and legs are made entirely of intertwined stems. She also wears a skirt with raspberry leaves and small raspberries and she looks very delicate and feminine.

- Figure 8 | Qualitative samples of Goku-T2I. Key words are highlighted in RED. For clarity, we zoom in on specific regions to enhance visualization.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

At an aquarium, a diver in a yellow wetsuit is feeding tropical fish in a large tank.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Zooming through a dense, lush rainforest at incredible speed, weaving between colossal trees, with rays of sunlight breaking through the canopy and exotic birds scattering in the distance.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

A snowboarder carves down a steep slope, their board cutting swiftly through the snow.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

A boxer dances around the ring, fists raised and jabbing rapidly at their opponent.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

A kung fu master swiftly maneuvers through a series of rapid punches and palm strikes, their arms blurring with speed.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

In a cozy living room with a roaring fireplace and plush furniture, a dog with a shiny coat sits contentedly on a soft rug.

###### Figure 9 | Qualitative samples of Goku-T2V. Key words are highlighted in RED.

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

GOKU(8B)ViduKling(1.5)DreamMachineOpen-soraPlan(v1.3)PikaCogVideoX1.5(5B)

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

Prompt：An astronaut runs across the surface of the moon, with a low-angle shot showcasing the vast lunar landscape. The movements are smooth and light.

###### Figure 10 | Qualitative comparisons of Goku-T2V with SOTA video generation methods. Key words are highlighted in RED.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

GOKU(8B)ViduKling(1.5)DreamMachineOpen-soraPlan(v1.3)PikaCogVideoX1.5(5B)

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

Prompt：A man surfing on a wave, with the camera following his movement and focusing on his face. He is smiling and giving a thumbs-up to the camera, conveying a sense of enjoyment and excitement. The ocean waves are vibrant and dynamic around him, with sunlight glistening on the water. The background features a clear blue sky, enhancing the lively atmosphere of the scene as he rides the waves with confidence and enthusiasm.

###### Figure 11 | Qualitative comparisons of Goku-T2V with SOTA video generation methods. Key words are highlighted in RED.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

A person performing dynamic and fast-paced water boxing, demonstrating quick, fluid arm movements while splashing water…

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

A kid rides a bike in the park, pedaling fast and moving towards the camera…

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

A highly detailed, photorealistic close-up image of two pirate ships engaged in an intense battle, their sails billowing as they maneuver through the dark, swirling surface of a coffee cup.

###### Figure 12 | Qualitative samples of Goku-I2V. Key words are highlighted in RED.

#### References

Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. (2025). Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575.

Albergo, M. S. and Vanden-Eijnden, E. (2023). Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations.

Bacher, I., Javidnia, H., Dev, S., Agrahari, R., Hossari, M., Nicholson, M., Conran, C., Tang, J., Song, P., Corrigan, D., et al. (2021). An advert creation system for 3d product placements. In Machine Learning and Knowledge Discovery in Databases: Applied Data Science Track: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part IV, pages 224–239. Springer.

Bao, F., Xiang, C., Yue, G., He, G., Zhu, H., Zheng, K., Zhao, M., Liu, S., Wang, Y., and Zhu, J.

(2024). Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233.

Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. (2023). Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8.

Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al. (2023a). Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127.

Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S. W., Fidler, S., and Kreis, K. (2023b). Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. (2024). Video generation models as world simulators.

Carreira, J. and Zisserman, A. (2017). Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308.

Castellano, B. (2024). PySceneDetect. Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al. (2023).

Pixart-alphaalpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426.

- Chen, S., Xu, M., Ren, J., Cong, Y., He, S., Xie, Y., Sinha, A., Luo, P., Xiang, T., and Perez-Rua, J.-M. (2024a). Gentron: Diffusion transformers for image and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6441–6451.
- Chen, T., Xu, B., Zhang, C., and Guestrin, C. (2016). Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174.

Chen, T.-S., Siarohin, A., Menapace, W., Deyneka, E., Chao, H.-w., Jeon, B. E., Fang, Y., Lee, H.-Y., Ren, J., Yang, M.-H., et al. (2024b). Panda-70m: Captioning 70m videos with multiple

cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331.

Chen, Z., Wang, W., Tian, H., Ye, S., Gao, Z., Cui, E., Tong, W., Hu, K., Luo, J., Ma, Z., et al. (2024c). How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. (2024). Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53.

Contributors, I. H. (2013). Image hash.

- Corporation, N. (2022). Nvidia h100 tensor core gpu architecture.
- Corporation, N. (2023). Nvidia announces dgx gh200 ai supercomputer.
- Corporation, N. (2024). Nvidia h200 nvl pcie gpu accelerates ai and hpc applications.

Dao, T. (2024). FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR).

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., et al. (2023). Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning, pages 7480–7512. PMLR.

Dehghani, M., Mustafa, B., Djolonga, J., Heek, J., Minderer, M., Caron, M., Steiner, A., Puigcerver, J., Geirhos, R., Alabdulmohsin, I. M., et al. (2024). Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In CVPR, pages 248–255.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. (2024). Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning.

Esser, P., Rombach, R., and Ommer, B. (2021). Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883.

Ge, S., Nah, S., Liu, G., Poon, T., Tao, A., Catanzaro, B., Jacobs, D., Huang, J.-B., Liu, M.-Y., and Balaji, Y. (2023). Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941.

Ghosh, D., Hajishirzi, H., and Schmidt, L. (2024). Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36.

Girdhar, R., Singh, M., Brown, A., Duval, Q., Azadi, S., Rambhatla, S. S., Shah, A., Yin, X., Parikh, D., and Misra, I. (2023). Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.

Ha, D. and Schmidhuber, J. (2018). World models. arXiv preprint arXiv:1803.10122. He, J., Xue, T., Liu, D., Lin, X., Gao, P., Lin, D., Qiao, Y., Ouyang, W., and Liu, Z. (2024). Venhancer: Generative space-time enhancement for video generation. arXiv preprint arXiv:2407.07667. He, Y., Yang, T., Zhang, Y., Shan, Y., and Chen, Q. (2022). Latent video diffusion models for

high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2(3):4.

Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., et al. (2022a). Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303.

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851.

Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D. J. (2022b). Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646.

Hong, W., Ding, M., Zheng, W., Liu, X., and Tang, J. (2022). Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. (2024). Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135.

Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. (2023). T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.

(2024). Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818.

Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song, S. L., Rajbhandari, S., and He, Y. (2023). Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509.

Ji, Y., Zhang, J., Wu, J., Zhang, S., Chen, S., GE, C., Sun, P., Chen, W., Shao, W., Xiao, X., et al.

(2024). Prompt-a-video: Prompt your video diffusion model via preference-aligned llm. arXiv preprint arXiv:2412.15156.

Jiang, Z., Lin, H., Zhong, Y., Huang, Q., Chen, Y., Zhang, Z., Peng, Y., Li, X., Xie, C., Nong, S., et al. (2024). Megascale: Scaling large language model training to more than 10,000 gpus. arXiv preprint arXiv:2402.15627.

Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., and Lin, Z. (2024). Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954.

Ju, X., Gao, Y., Zhang, Z., Yuan, Z., Wang, X., Zeng, A., Xiong, Y., Xu, Q., and Shan, Y. (2024). Miradata: A large-scale video dataset with long durations and structured captions. arXiv preprint arXiv:2407.06358.

Kingma, D. P. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.

(2024). Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

Korthikanti, V. A., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., and Catanzaro, B.

(2023). Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5:341–353.

Kuaishou (2024). Kling ai. https://klingai.com/. Lab, P.-Y. and etc., T. A. (2024). Open-sora-plan. Li, S., Xue, F., Baranwal, C., Li, Y., and You, Y. (2021). Sequence parallelism: Long sequence

training from system perspective. arXiv preprint arXiv:2105.13120. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations.

Liu, X., Gong, C., and qiang liu (2023). Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations.

Lu, P., Peng, B., Cheng, H., Galley, M., Chang, K.-W., Wu, Y. N., Zhu, S.-C., and Gao, J. (2024). Chameleon: Plug-and-play compositional reasoning with large language models. Advances in Neural Information Processing Systems, 36.

Luma (2024). Luma ai. https://lumalabs.ai/dream-machine. Mishkin, P., Ahmad, L., Brundage, M., Krueger, G., and Sastry, G. (2022). Dall· e 2 preview-risks

and limitations. Noudettu, 28(2022):3.

Nan, K., Xie, R., Zhou, P., Fan, T., Yang, Z., Chen, Z., Li, X., Yang, J., and Tai, Y. (2024). Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. (2023). Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Peebles, W. and Xie, S. (2023). Scalable diffusion models with transformers. In Proceedings of the

IEEE/CVF International Conference on Computer Vision, pages 4195–4205. pika (2024). Pika ai. https://pika.art/try. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach,

R. (2023). Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952.

Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., et al. (2024). Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720.

Quevedo, J., McIntyre, Q., Campbell, S., and Wachen, R. (2024). Oasis: A universe in a transformer.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Runway (2023). Gen-2: Generate novel videos with text, images or video clips. https: //runwayml.com/research/gen-2/.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. (2022). Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. (2024). Flashattention3: Fast and accurate attention with asynchrony and low-precision. arXiv preprint arXiv:2407.08608.

Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., and Taigman, Y. (2023). Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations.

Skorokhodov, I., Tulyakov, S., and Elhoseiny, M. (2022). Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3626–3636.

Soomro, K., Zamir, A. R., and Shah, M. (2012). Ucf101: A dataset of 101 human actions classes from videos in the wild. Technical report, Center for Research in Computer Vision, Orlando, FL 32816, USA. CRCV-TR-12-01.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. (2024). Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525.

Team, V. (2024). Vchitect-2.0: Parallel transformer for scaling up video diffusion models.

##### https://github.com/Vchitect/Vchitect-2.0.

Teed, Z. and Deng, J. (2020). Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer.

Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., and Gelly, S. (2018). Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717.

Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. (2024). Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

Wan, B., Han, M., Sheng, Y., Lai, Z., Zhang, M., Zhang, J., Peng, Y., Lin, H., Liu, X., and Wu, C.

(2024). Bytecheckpoint: A unified checkpointing system for llm development. arXiv preprint arXiv:2407.20143.

Wang, J., Yuan, L., Zhang, Y., and Sun, H. (2024a). Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634.

- Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., Zhao, Y., Ao, Y., Min, X., Li, T., Wu, B., Zhao, B., Zhang, B., Wang, L., Liu, G., He, Z., Yang, X., Liu, J., Lin, Y., Huang, T., and Wang, Z. (2024b). Emu3: Next-token prediction is all you need.
- Wang, Y., Chen, X., Ma, X., Zhou, S., Huang, Z., Wang, Y., Yang, C., He, Y., Yu, J., Yang, P., et al. (2023a). Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103.

Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al. (2023b). Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942.

Wiegand, T., Sullivan, G. J., Bjontegaard, G., and Luthra, A. (2003). Overview of the h. 264/avc video coding standard. IEEE Transactions on circuits and systems for video technology, 13(7):560– 576.

Wu, J. Z., Ge, Y., Wang, X., Lei, S. W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., and Shou, M. Z.

(2023). Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. (2024). Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528.

Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., et al. (2024a). Qwen2 technical report. arXiv preprint arXiv:2407.10671.

Yang, M., Li, J., Fang, Z., Chen, S., Yu, Y., Fu, Q., Yang, W., and Ye, D. (2024b). Playable game generation. arXiv preprint arXiv:2412.00887.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. (2024c). Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

Yuan, L., Wang, J., Sun, H., Zhang, Y., and Lin, Y. (2025). Tarsier2: Advancing large visionlanguage models from detailed video description to comprehensive video understanding. arXiv preprint arXiv:2501.07888.

Zeng, Y., Wei, G., Zheng, J., Zou, J., Wei, Y., Zhang, Y., and Li, H. (2024). Make pixels dance: High-dynamic video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8850–8860.

Zhang, B. and Sennrich, R. (2019). Root mean square layer normalization. Advances in Neural Information Processing Systems, 32.

Zhang, J., Chen, J., Wang, C., Yu, Z., Qi, T., Liu, C., and Wu, D. (2024). Virbo: Multimodal multilingual avatar video generation in digital marketing. arXiv preprint arXiv:2403.11700.

Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., Desmaison, A., Balioglu, C., Damania, P., Nguyen, B., Chauhan, G., Hao, Y., Mathews, A., and Li, S. (2023). Pytorch fsdp: Experiences on scaling fully sharded data parallel. Proc. VLDB Endow., 16(12):3848–3860.

Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., and You, Y. (2024). Open-sora: Democratizing efficient video production for all.

- Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. (2024). Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039.
- Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., and Feng, J. (2022). Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018.

