[Figure 1]

## FireRed Edit

#### FireRed-Image-Edit-1.0 Technical Report

Super Intelligence Team, Xiaohongshu Inc.

##### Abstract

We present FireRed-Image-Edit, a diffusion transformer for instruction-based image editing that achieves state-of-the-art performance through systematic optimization of data curation, training methodology, and evaluation design. We construct a 1.6B-sample training corpus, comprising 900M text-to-image and 700M image editing pairs from diverse sources. After rigorous cleaning, stratification, auto-labeling, and two-stage filtering, we retain over 100M high-quality samples balanced between generation and editing, ensuring strong semantic coverage and instruction alignment. Our multi-stage training pipeline progressively builds editing capability via pre-training, supervised fine-tuning, and reinforcement learning. To improve data efficiency, we introduce a Multi-Condition Aware Bucket Sampler for variable-resolution batching and Stochastic Instruction Alignment with dynamic prompt re-indexing. To stabilize optimization and enhance controllability, we propose Asymmetric Gradient Optimization for DPO, DiffusionNFT with layout-aware OCR rewards for text editing, and a differentiable Consistency Loss for identity preservation. We further establish REDEdit-Bench, a comprehensive benchmark spanning 15 editing categories, including newly introduced beautification and low-level enhancement tasks. Extensive experiments on REDEdit-Bench and public benchmarks (ImgEdit and GEdit) demonstrate competitive or superior performance against both open-source and proprietary systems. We release code, models, and the benchmark suite to support future research.

# arXiv:2602.13344v2[cs.CV]14Jun2026

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

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

GitHub https://github.com/FireRedTeam/FireRed-Image-Edit

HuggingFace Model https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0

HuggingFace Demo Online Demo (HuggingFace)

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

Figure 1 | This figure benchmarks generative image models across four human evaluation dimensions (alignment, consistency, realism, aesthetics) and three editing tasks (Imgedit, Gedit, RedEdit).

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

###### Figure 2 | Showcase of FireRed-Image-Edit in general image editing.

###### Contents

- 1 Introduction 4
- 2 Data 4

- 2.1 Data Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.2 Data Pre-Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 2.2.1 Deduplication . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.2.2 Photometric & Statistical Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.2.3 Content Validity & Artifact Removal . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.2.4 Perceptual Quality & Aesthetics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.2.5 AIGC Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.3 Data Production Engine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.4 Long-tail Data Supplementation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 2.5 Captioning Engine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 2.5.1 Structured Captioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 2.5.2 Instruction Captioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 2.6 Data Post-Filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 3 Model Training 11

- 3.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.2 Training Efficiency Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 3.3 Pre-training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 3.4 Continued Pre-training (CT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 3.5 Supervised Fine-Tuning (SFT) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 3.6 Reinforcement Learning with Human Feedback (RLHF) . . . . . . . . . . . . . . . . . . . . . . . . 15

- 3.6.1 Direct Preference Optimization (DPO) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 3.6.2 Diffusion NFT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 3.7 Consistency Loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 3.8 Training Strategy. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 4 REDEdit-Bench 18

- 4.1 Benchmark Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 4.2 Evaluation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 5 Performance and Evaluation 18

- 5.1 Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 5.1.1 Evaluation Dimensions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 5.1.2 Results Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 5.2 Quantitative and Qualitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 5.2.1 General Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 5.2.2 Text-Centric Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 5.2.3 Creative Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 5.2.4 Try-on Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 6 Conclusion 29
- 7 Authors 30 References 31

3

###### 1. Introduction

The domain of text-to-image (T2I) synthesis has undergone a paradigm shift in recent years, progressing rapidly from elementary texture generation to the creation of photorealistic imagery with intricate semantic alignment [26, 7, 13, 38, 29, 20, 1]. However, this surge in performance has erected substantial barriers to entry. The current ecosystem is increasingly polarized: on one hand, proprietary commercial heavyweights—such as Nano Banana Pro [10] and Seedream 4.0 [29]—operate as opaque "black boxes," delivering superior fidelity but offering zero interpretability or reproducibility. On the other hand, while the open-source community strives for democratization, the prevailing trend has been to relentlessly scale model size—ballooning to tens of billions of parameters (e.g., Qwen-Image [38] at 20B, FLUX.2 [14] at 32B, and Step-1X-Edit [20] at 19B). This trajectory imposes unsustainable computational burdens for both training and deployment. Consequently, distilling synthetic data from proprietary models has become a popular workaround for resource-limited academic fields [22, 17]. Despite these efforts, the community continues to fixate on the sheer scaling of model parameters, often at the expense of efficiency in data processing and model training. Critical gaps remain in two key areas: the efficient construction of high-quality datasets for image editing, and the development of a scientifically rigorous, standardized benchmark capable of evaluating these models across universal metrics.

In this paper, we introduce FireRed-Image-Edit, a holistic, end-to-end framework that rigorously optimizes every facet of the model’s lifecycle-spanning data curation, architecture design, training efficiency, and inference optimization-enabling state-of-the-art performance with massive high-quality data.

To fully exploit this architecture, we scale our training regime to approximately 100 million diverse image-text pairs spanning text-to-image generation, single and multi-image-to-image synthesis, and complex instructionbased editing. Supporting this massive data scale demands systematic training efficiency innovations. We introduce a Multi-Condition Aware Bucket Sampler that explicitly accounts for variable input image counts across different editing tasks, optimizing aspect ratio batching to minimize padding-induced computational waste while preserving spatial layout integrity. Complementing this, our Stochastic Instruction Alignment mechanism employs random dropout and permutation of reference images during data collation, with dynamic re-indexing of text prompts to maintain semantic correspondence—forcing the model to decouple spatial orderings from content and thereby enhancing generalization in complex multi-reference scenarios. At the system level, we integrate pre-extracted text feature caching, Fully Sharded Data Parallel (FSDP) for distributed state management, gradient checkpointing, and mixed-precision training to maximize throughput without sacrificing stability.

Our training strategy further refines convergence through three coordinated mechanisms. Distributed Stratified Timestep Sampling decomposes the diffusion horizon into sub-intervals assigned across GPU ranks, with synchronized rotation preventing rank-specific overfitting and ensuring global uniformity in noise coverage. Logit-Normal Loss Weighting concentrates gradient contributions on the semantically critical intermediate diffusion timesteps while suppressing negligible extremes. Finally, Model Weight Averaging via Exponential Moving Average synthesizes parameters across the optimization trajectory, neutralizing transient fluctuations for enhanced robustness against real-world distribution shifts.

Furthermore, we recognize that conventional metrics often fail to capture the nuance required for productionready applications, leading to a disconnect between academic scores and actual user experience. To bridge this gap, we establish a comprehensive, multi-dimensional benchmark designed to rigorously evaluate model performance closer to practical deployment standards. This evaluation framework encompasses a wide spectrum of editing tasks, ranging from subtle textural adjustments to complex structural alterations, and assesses the model across multiple axes—including instruction alignment, background preservation, and photographic fidelity. By shifting the evaluation focus from theoretical metrics to applied utility, our benchmark provides a granular analysis of current limitations and demonstrates our model’s superior adaptability in facilitating genuine, unrestricted creative workflows.

###### 2. Data

The quality of training data is fundamental to generative models and largely sets their achievable performance. To this end, we collected 1.6 billion samples in total, comprising 900 million text-to-image pairs and 700 million image editing pairs. The editing data is drawn from diverse sources, including open-source datasets (e.g., OmniEdit [37], UnicEdit-10M [47]), our data production engine, video sequences, and the internet, while the text-to-image samples are incorporated to preserve generative priors and ensure training stability. Through rigorous cleaning, fine-grained stratification, and comprehensive labeling, and with a two-stage filtering pipeline (pre-filter and post-filter), we retain 100M+ high-quality samples for training, evenly split between text-to-image and image editing data, ensuring broad semantic coverage and high data fidelity.

###### 2.1. Data Distribution

Our dataset is carefully curated to support the development of a robust image editing model. The dataset is divided into two main categories: Text-to-Image (T2I) and Image-to-Image (I2I). The data is meticulously balanced across various subcategories to ensure diversity and comprehensive coverage of real-world scenarios, as shown in Fig. 3. After thorough filtering and selection, the dataset is ultimately approximately balanced in a 1:1 ratio between T2I and I2I data. This balanced composition is intentional, as the inclusion of T2I data is crucial for building a strong foundation in text semantic understanding, which enables the model to achieve better generalization on a wide variety of image editing tasks.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

16.32%)

Objects

Activities(2.94%)

Sports(3.75%)

Cityscape (8.75%)

[Figure 94]

[Figure 95]

(

Portrait (11.93%)

Landscape (7.34%)

Posters (3.80%)

Animals (7.33%)

[Figure 96]

UI (4.40%)

[Figure 97]

[Figure 98]

Indoor (6.29%)

Cartoon (4.87%)

People

Nature

19.7%)

(57.7%)

(

Plants (4.30%)

Design(22.6%)

Arts (8.26%)

Text2Image (100%)

Food (3.78%)

Tone Transformation (4.69%)

[Figure 99]

[Figure 100]

Image2Image (100%)

[Figure 101]

(24.2%Stylistic)

Portrait Edit (11.45%)

Transfer (5.30%)

Structural

Style

49.6%)

(Semantic 26.2%)

Alteration (5.97%)

(

Modiﬁcation

Text

Color

(11.32%) Modiﬁcation

Change

(6.51%)

[Figure 102]

[Figure 103]

Material

[Figure 104]

[Figure 105]

Replacement

Motion

Subject Add & Removal(8.67%)

Background

(5.73%)

(10.02%) Change

(8.95%)

View

Change

Subject

(8.14%)

(5.22%)

Change

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Hybrid

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

- Figure 3 | Overview of Data Distrubution. Our training dataset achieves an approximate 1:1 ratio between Text-to-Image (T2I) and Image-to-Image (I2I) tasks after data cleaning. The T2I component is divided into three main categories: Nature (general-purpose generation), People (human-centric generation), and Design (artistic styles, text rendering, and complex layouts). The I2I component is also divided into three categories: Semantic Editing (content-based modifications), Stylistic Editing (aesthetic adjustments), and Structural Editing (spatial arrangement and composition). Our data collection strategy ensures a balance of diversity and quality throughout the training process, providing comprehensive coverage and precise annotations to foster robust model training.

Text-to-Image (T2I) The T2I data is incorporated into the training process to preserve generative priors and ensure training stability, allowing the model to produce coherent and realistic images from textual prompts. The dataset is divided into several key categories, each selected to cover a wide range of contexts, styles, and subject matter. The major categories in the T2I dataset are as follows:

- • Nature (57.7%): The largest category, covering diverse subcategories such as Objects, Landscapes, Cityscapes, Plants, Animals, Food, and Indoor scenes. This category is essential for generating realistic and varied natural environments.
- • Design (22.6%): Focuses on structured and artistic visual content, including Posters, User Interfaces, Cartoon and other forms of art. It plays a key role in training the model to understand complex layouts, artistic styles, and detailed textual prompts.

!

#

%

"

$

### &

- • People (19.7%): Includes images related to human figures, such as Portraits, Sports, and Activities. This category is crucial for training the model to generate accurate and diverse human-related imagery, enhancing its ability to handle human-centered tasks.

Image-to-Image (I2I) The I2I component of our dataset is crutial for enhancing or modifying existing images based on specific instructions. The tasks are broadly categorized into three main types: Semantic Editing, Stylistic Editing, and Structural Editing. These categories are defined by the nature of the modifications made to the image—whether it’s altering the content, enhancing the visual style, or adjusting the structure and composition of the scene. Below, we describe each category along with its associated subcategories:

- • Semantic Editing (26.2%): This category involves modifying the content of an image, such as changing the objects or background. It includes tasks like adding, removing, or replacing objects, and altering the background. These tasks are important for enabling the model to edit the primary elements of the scene, allowing for precise content manipulation while maintaining the overall coherence of the image.
- • Stylistic Editing (24.2%): Stylistic editing focuses on enhancing the visual style and aesthetic characteristics of an image without altering its core content. Tasks in this category include color alteration, style transfer, tone transformation, and material modification. These tasks are crucial for training the model to change the image’s appearance, such as modifying the atmosphere, applying artistic styles, or improving the visual aesthetics while preserving the subject matter.
- • Structural Editing (49.6%): Structural editing covers changes to the spatial arrangement and positioning of elements within the image. It includes fine-grained, control-intensive scenarios such as view change, motion change, and portrait change. Beyond these, more complex cases like text modification and higherlevel hybrid change are also included, requiring coordinated spatial and visual reasoning and enabling the model to perform more advanced and controllable scene transformations.

###### 2.2. Data Pre-Filtering 2.2.1. Deduplication

To mitigate redundancy from diverse sources and ensure high-quality training data, we implement a multi-level hierarchical deduplication pipeline, which consists of three levels with increasing strictness:

Level 1: Coarse Global Deduplication At the first level, we perform a large-scale near-duplicate retrieval and clustering across all images using a scalable image deduplication system [25]. Images are embedded into a shared feature space, where we identify and group visually similar samples, leaving only representative

Deduplication Photometric & Statistical

[Figure 122]

[Figure 123]

Brightness

[Figure 124]

Global Level

[Figure 125]

Saturation

[Figure 126]

RGB Entropy

Pair Level

[Figure 127]

Texture Complexity

[Figure 128]

Sharpness

Multi-Metric Level

###### Data Filtering

###### Perceptual Quality & Aesthetics

Content Validity & Artifact Removal

[Figure 129]

LAION-Aesthetics

Watermark Text Overlay

Mosaic

###### Over Smooth

Blind-IQA

Collage

Homogenized Lighting

QR codes

Rotation

Barcodes

AIGC Detection

- Figure 4 | Overview of Data Filtering. Our multi-stage data filtering pipeline includes deduplication, photometric and statistical filtering, artifact removal, perceptual quality assessment, and AIGC detection. These steps ensure the dataset is free from redundancy, visual noise, artifacts, and synthetic content, maintaining high-quality samples for image editing model training.

samples that contribute distinct visual content to the dataset. This stage primarily addresses exact duplicates and broad similarities across the image pool.

- Level 2: Pair-Level Deduplication The second level focuses on image-to-image (I2I) editing data, where redundancy can also manifest in the form of duplicate source–target pairs. In this stage, we apply a coarse pair-level deduplication, where CLIP [27] embeddings and an internal image embedding model are used to compute similarity scores between source and target images. Pairs with high semantic similarity are discarded to remove trivial identity mappings or identical edit results. This helps in reducing template-based redundancy that can negatively affect model generalization.
- Level 3: Fine-Grained Multi-Metric Deduplication At the final level, we apply a strict multi-metric deduplication strategy to ensure the retention of only genuinely distinct and meaningful edit samples. Here, we combine Peak Signal-to-Noise Ratio (PSNR) [9], Structural Similarity Index Measure (SSIM) [36], Image Similarity Challenge 2021 metric (ISC21) [6], and CLIP similarity to perform a fine-grained comparison. This multi-metric approach allows us to eliminate subtle near-duplicates, templates with minimal variation, and images that exhibit visually insignificant changes. By applying this stringent filtering, we ensure that the dataset retains only high-quality, diverse, and non-repetitive edit samples.

###### 2.2.2. Photometric & Statistical Filtering

To reduce visual noise and stabilize training, we perform photometric and statistical filtering using descriptors like Brightness, Saturation, RGB Entropy, Texture Complexity, and Sharpness. Brightness and saturation are used to detect under- or over-exposed images and those with abnormal color responses due to extreme lighting or non-natural rendering. RGB Entropy filters out images with large uniform regions or flat backgrounds, while Texture Complexity removes samples with irregular high-frequency patterns, often caused by compression artifacts or non-semantic textures. Sharpness is used to identify images with motion blur, defocus, or other degradations that obscure fine details. We estimate the empirical distribution of each descriptor and discard samples outside the statistically plausible range, ensuring the dataset retains stable photometric properties and richer structural content for training.

###### 2.2.3. Content Validity & Artifact Removal

To ensure valid content and prevent misleading supervision signals, we use specialized detectors to remove images with undesirable artifacts. This includes filtering images with watermarks, visible text overlays, mosaics, collages, privacy masks, QR codes, barcodes, as these can introduce non-semantic patterns or distortions that impair model performance. We also apply rotation correction based on metadata and geometric cues to ensure consistent spatial alignment, reducing unnecessary geometric variance. This filtering improves the validity of visual supervision and provides a cleaner foundation for accurate image editing.

###### 2.2.4. Perceptual Quality & Aesthetics

We use deep learning evaluators to assess image quality and aesthetics. IQA metrics [24, 23] detect issues like blur, noise, and exposure imbalance, while aesthetics predictors [28] evaluate composition and appeal. Low-quality or poorly rated samples are filtered out, ensuring better visual coherence for image editing training.

###### 2.2.5. AIGC Detection

We use internal AIGC detection to identify and filter images generated by generative models. These samples may bypass standard quality checks but exhibit biases like smooth textures, uniform lighting, and suppressed details. Removing AIGC content preserves authentic image statistics and prevents synthetic priors from degrading model generalization.

###### 2.3. Data Production Engine

While real-world data offers a natural distribution, it often lacks the density required to support specialized editing tasks. To address this limitation, we build a data production pipeline that generates paired editing data through three forward construction strategies, as illustrated in Fig. 5, enabling dense coverage and controllable synthesis across the editing task space.

Synthesis of Expert Models via Instructional Control We adopt image editing models [51, 39, 32, 16] that require only the input image and an instruction as the default experts, enabling fast and broad data generation. Building on this foundation, we develop an instruction-driven expert synthesis pipeline to stably map diverse

###### (1) Synthesis of Expert Models via Instructional Control

Option1

[Figure 130]

[Figure 131]

Content Discovery Task Inventories Auxiliary Metadata

| |Option2|
|---|---|
| | |
| |Option3|

[Figure 132]

[Figure 133]

[Figure 134]

Instructions Lexicons

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Expert Models

Original Full Instruction

VLM

[Figure 139]

###### (2) Synthesis of Expert Models via Structured Control

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

| | |
|---|---|
| | |

Iterable

SAM3

Mask

Structural Info

[Figure 144]

[Figure 145]

[Figure 146]

Original

Original

DWPose Keypoint Preset Config

Templates

(3) Model-free Template-based Synthesis

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

（ ）

[Figure 151]

[Figure 152]

Algorithmic Filters

Layout Template

3D Template

Original

Edited

- Figure 5 | Overview of Data Engine. The data production engine generates paired image editing samples through three forward construction strategies. (1) Instructional Control synthesizes expert models using instruction templates and edit-target lexicons grounded by VLM discovery and auxiliary metadata. (2) Structured Control leverages structural priors such as masks and pose keypoints extracted from perception modules to guide expert models with precise control signals. (3) Model-free Template-based Synthesis includes approaches such as predefined 3D templates, layout templates, and algorithmic filters to enable controllable and deterministic generation. The pipeline is designed to be iterable, supporting complex multi-step edits.

editing tasks to the capability boundaries of different expert models. Concretely, we formalize each expert’s scope into deterministic instruction templates and build and maintain: an Expert-instruction library and an Edit-target lexicon. The Expert-instruction library summarizes model constraints and triggering conditions into canonical primitives and structured slots, improving instruction stability and performance. The Edit-target lexicon defines what to edit by combining VLM Discovery to interpret the image and propose grounded edit targets, Task Inventories as a task-aligned library of editable objects/attributes for long-tail coverage, and auxiliary metadata (e.g., OCR, coordinates) as computable anchors for precise grounding. To further improve editing quality, we refine expert models with SFT or LoRA on curated data.

Synthesis of Expert Models via Structured Control We further introduce expert models [51, 35, 43, 45, 18, 55, 12, 34, 11] that require structured control, where editing is driven by non-text, structured signals to reduce language ambiguity and improve generation controllability. Specifically, we extract structural priors from auxiliary perception modules such as SAM [3] and DWpose [46], including segmentation masks and pose keypoints, and combine them with task parameters in Preset Configuration Templates to derive task-specific structured control info. The structured info is fed into expert models to precisely modulate the editing process. This structured-control approach is particularly effective for spatially sensitive tasks such as precise object removal, facial expression and body pose transfer, and appearance redirection.

Model-free Template-based Synthesis To complement generative approaches, we adopt a model-free template-based synthesis pipeline that avoids model inference, instead utilizing a rich library of predefined algorithmic templates to produce editing pairs with precise edits and without artifacts. The template set includes, but is not limited to: 3D Parametric Templates that leverage graphics engines and skeletal rigs to synthesize pixel-consistent facial expressions, body re-posing, and other parametric transformations; Structured Layout Templates that define spatial anchors for text, logos, UI elements, and similar graphic components to strictly enforce typographic and layout constraints; and Algorithmic Filter Templates that cover deterministic image

signal processing operations such as sharpening, color redirection, and other low-level visual enhancements. Beyond these, we incorporate additional task-specific templates to support a wider range of deterministic edits.

Beyond the three forward construction strategies, we further enhance data fidelity and complexity through two mechanisms: Task Inversion and Task Splitting. Task Inversion leverages the reversibility of certain edits by swapping source and target images, thereby reducing instruction direction bias. Task Splitting addresses high-difficulty requests by decomposing multi-operation edits into sequential atomic steps, applying our iterable pipeline that performs one modification at a time, thereby reducing quality degradation and preserving structural consistency.

###### 2.4. Long-tail Data Supplementation.

To deal with the long-tail problem where some editing tasks lack enough samples, we use a straightforward "check and fill" strategy.

We first evaluate the coverage of our existing dataset by indexing all training instructions into a vector database. When the model exhibits poor performance in a specific editing domain, we query this index to quantify the available training samples. If the sample density for a particular instruction is insufficient, we identify it as a "data gap" that requires targeted supplementation.

Once a gap is identified, we leverage our robust image retrieval framework to acquire base imagery from a massive candidate pool. This framework employs a specialized deep feature encoder to map raw imagery into a high-dimensional latent space, generating dense, highly discriminative embeddings. Supported by a large-scale vector indexing engine, the system utilizes seed images or semantic queries to retrieve samples exhibiting high fidelity in spatial layout, texture, color distribution, and semantic coherence via metric learning. Using both text-to-image and image-to-image retrieval, we can precisely locate images that match the missing task’s requirements. These retrieved images are then processed through our data production engine to generate new, high-quality training pairs, effectively rectifying the weak points in our data distribution.

###### 2.5. Captioning Engine

To provide high-quality textual supervision for both Text-to-Image (T2I) generation and Image-to-Image (I2I) editing, we developed a comprehensive Captioning Engine. This engine operates progressively: from static single-image understanding to dynamic cross-image differential reasoning, and finally to user-centric instruction refinement.

###### 2.5.1. Structured Captioning

For both T2I training and the visual grounding of I2I pairs, simple alt-texts are insufficient. We employ a VLM to generate Structured Captions (see Annotation Prompt 7) that provide a comprehensive breakdown of the image. These captions explicitly detail the main subject, background elements, lighting conditions, artistic style, and camera angles. This "structured" supervision ensures the model understands what is in the image before learning how to change it.

[Figure 153]

[Figure 154]

Viewpoint Color Tone Left-right Orientation

让⼥孩站在右侧，⽬光向左侧远⽅。 镜头聚焦在⼥孩上半⾝，特写⾯部， 确保烟花棒从右⼿转移到左⼿。背景 设置为温暖⽇落⾊调。

帮我把图变为⼀个好看的特写

请将图像中的⼥孩调整为⾯朝左侧， ⽬光望向海岸，保持略微侧脸的姿 势，仿佛她正在凝视着海平线。镜头 聚焦于⼥孩的上半⾝，进⾏特写，拉 近视⾓以突显她的⾯部表情和细节， 同时确保她⼿中的仙⼥棒从右⼿转移 到左⼿。背景⾊调变为温暖的⽇落氛 围，强化橙黄和蓝⾊渐变的天空，突 出⽇落光辉。微调海⾯的波光，轻微 模糊背景使整个画⾯更加宁静与……

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Random VLMs VLM

把⼥孩摆放在图像右侧，⽬光向远⽅ 望去。把镜头聚焦在⼥孩上半⾝，特 写她的⾯容，仙⼥棒从右⼿换到左 ⼿，并且背景展现柔和的⽇落光辉。

VLM

可以帮我修出氛围感特写吗？

Detailed Instruction Concise Instruction User-Like Instruction

Figure 6 | An example of Instruction Captioning. Detailed Instruction Captioning generates precise descriptions of visual changes, Concise Instruction Refinement simplifies these instructions while retaining key details, and User-Like Instruction Refinement rephrases them in a more natural, conversational tone.

###### 2.5.2. Instruction Captioning

To provide high-quality textual supervision for I2I (Image-to-Image) editing tasks, we categorize our instruction generation into three distinct types: Detailed Instruction Captioning, Concise Instruction Refinement, and

User-Like Instruction Refinement.

Detailed Instruction Captioning Leveraging a specially fine-tuned VLM [33], this stage focuses on generating foundational ground truth benchmarks for visual transitions between image pairs. The objective is to produce descriptions that are accurate, complete, and unambiguously reflective of the source-to-target changes. The model is specifically trained to conduct high-fidelity differential reasoning, consistently modeling subject attributes, spatial relationships, and semantic shifts. Special emphasis is placed on high-sensitivity dimensions, such as left-right orientation, viewpoint variations, and fine-grained attributes, which demand superior recognition and expressive capabilities from the model.

Concise Instruction Refinement This stage concurrently addresses both instruction brevity and linguistic diversity, integrating them into a unified refinement pipeline. By utilizing random VLM models from a suite of diverse VLM models [8, 31] and extensive lexical libraries, we implement varying levels of simplification while strictly preserving the original editing intent. To prevent the model from overfitting to the rigid templates of the base Caption VLM, we dynamically sample diverse syntactic structures during the rewriting process, ranging from inversions and subjectless imperatives to informal phrasings and various other linguistic patterns. This approach facilitates intent purification by stripping redundant visual modifiers and abstracting concrete concepts, ensuring the model masters core editing logic rather than specific object-based patterns.

User-Like Instruction Refinement This stage refines the concise outputs into natural, everyday language to better reflect how people actually speak. The goal is to bridge the gap between technical commands and human conversation by injecting persona-driven tones and help-seeking markers. This simulates real-world scenarios where a user might request help using phrases such as "Can you help me fix this?" or "Could you save this image by changing something?". By incorporating these warm and interactive phrases at varying ratios, we significantly enhance the model’s ability to interpret informal, intuitive, and help-seeking user intents.

By mixing these rewritten concise instructions with the original detailed descriptions for training, we constructed a multi-scale data distribution covering the spectrum from minimal colloquialisms to fine-grained specifications. This mixed strategy allows the Fine-grained Visual Reasoning capabilities learned from long texts to be effectively generalized to short text scenarios. Consequently, the model can leverage its deep visual knowledge to perform accurate inference and execution, even when facing ambiguous or concise user instructions.

###### 2.6. Data Post-Filtering

To ensure high fidelity and precise semantic alignment, we implement a post-filtering pipeline for automated data quality assessment and curation. This stage serves as an automated quality gate to refine data including both natural real-world pairs and synthesized pairs. Specifically, we train a specialized multimodal evaluation model based on Qwen3-VL-8B [33] by integrating LLM-driven hard negative mining with expert-level human annotation. This model precisely quantifies edited images across multiple dimensions. Subsequently, we leverage this model to perform automated filtering on a dataset of more than a hundred million raw samples. By eliminating instances exhibiting semantic misalignment or poor perceptual quality, we significantly enhance the overall distributional quality of the training dataset.

Hard Negative Mining and Multi-Dimensional Annotation High-quality negative samples are critical for enhancing the discriminative capability of evaluation models during data refinement. We initially sample 50,000 triplets of "Source Image – Editing Instruction – Target Image" from the massive raw dataset to serve as positive samples. Subsequently, we employ an LLM to apply semantic perturbations and stochastic rewriting to the original instructions. This process generates deviate instructions to construct 50,000 negative samples, resulting in a balanced dataset of 100,000 instances. To establish a gold-standard benchmark, we organize a group of human experts for double-blind annotation. The evaluation criteria focus on two key metrics: instruction alignment (assessing whether the edit accurately executes the textual command) and perceptual quality (checking clarity, artifacts, and aesthetic appeal). This rigorous annotation scheme ensures a balanced semantic distribution, providing a robust foundation for the precision of the subsequent evaluation model.

Training of the Vision-Language Evaluation Model To achieve an automated assessment mechanism that aligns closely with human aesthetic and logical judgments, we utilize Qwen3-VL-8B as the backbone for fine-tuning. Through supervised fine-tuning (SFT) on the aforementioned 100,000 high-quality annotated samples, the model acquires the capability of parsing complex editing instructions and perceiving subtle visual discrepancies. Experimental results demonstrate that the evaluation model exhibits exceptional robustness in multi-dimensional scoring tasks. Its metrics show a strong correlation with the evaluations of human experts. Beyond identifying subtle semantic deviations, the model is highly sensitive to structural distortions and textural artifacts in generated images, serving as a reliable automated proxy to replace costly manual evaluation.

Large-Scale Data Curation and Quality Assurance Using the trained evaluation model, we conduct a comprehensive quality assessment of the raw training data at a scale of over 100 million samples. We establish strict filtering thresholds to perform joint screening based on two dimensions: alignment score and quality score. Specifically, the model automatically identifies and discards samples that fail to follow the instructions, exhibit semantic shifts, or possess low visual fidelity (e.g., blurring, artifacts, or logical inconsistencies). This systematic data purification process significantly refines the distribution of the training set. It ensures that the generative model focuses on high-quality, high-fidelity editing pairs during subsequent large-scale training, thereby fundamentally improving the performance of the final model.

###### 3. Model Training

global Roi1 Roi2

key={task}-{img1}-{img2}..{imgn}

[Figure 159]

Cond Roi

[Figure 160]

[Figure 161]

[Figure 162]

| |
|---|

Roi2

Input

GT

|1:1|
|---|

Image Encoder

ℒ

Task

1:1 9:16

| |
|---|

Pred Roi

[Figure 163]

[Figure 164]

[Figure 165]

Roi1

Bucket 1

MSE loss

Input

[Figure 166]

Consistency Loss

GT

𝜀 𝜀̂

|4:3|
|---|

Task

1:1

Bucket 2

…

MMDiT Block

Input

GT

|3:4|
|---|

Task

1:1 1:1 1:1

[Figure 167]

timestep Noise 𝜀

Bucket n

Bucket Sampler

###### VAE Encoder Qwen VL

Prompt :

[Figure 168]

- Roi1 Caption

| |
|---|

- Roi2

[Figure 169]

[Figure 170]

[Figure 171]

Fig1

[Figure 172]

| |
|---|

|Roi1|
|---|

[Figure 173]

Instruction or Instructive/ Dense

A man in Fig1 and a woman in Fig2 were standing on the road, holding a cat in Fig3 between them.

drop

Roi2

Fig2

Editing Target Input Image(s)

[Figure 174]

Prompt :

[Figure 175]

[Figure 176]

Fig3 Fig2

… Fig1 … … … Fig2 …. …Fig3....

Collate Shuffle & Drop

drop

Select the same bucket for all processes

Collate Shuffle & Drop

Bucket Sampler

- Figure 7 | Architecture overview. The data pipeline begins with a Bucket Sampler that organizes input sequences based on task categories and aspect ratios to handle variable resolutions efficiently. This is followed by a Collate Shuffle & Drop mechanism that augments text prompts by randomly permuting or omitting figure identifiers to enhance robustness. The core model employs MMDiT Blocks to process multimodal features, where visual inputs are encoded by a VAE Encoder and multimodal conditions (reference images and textual instructions) are processed by Qwen VL. To ensure high-fidelity generation, the training process incorporates an consistency similarity loss; this mechanism extracts regions of interest (RoIs) from both predicted and ground truth images, passing them through a shared image encoder to minimize identity discrepancy.

###### 3.1. Architecture

Scalability and instructional fidelity are the cornerstone objectives guiding the design of our proposed model. Built upon an open-source multimodal text-to-image foundation [38], our architecture inherits a profound understanding of vision-language nuances, which we further extend to the generative and editing domains. To bridge the gap from multimodal perception to high-fidelity synthesis, we scale our training regime using a curated dataset of approximately one billion image-text pairs. This extensive corpus encompasses a diverse array of tasks, including text-to-image, single/multi-image-to-image generation, and complex instruction-based editing. Such a massive data scale enables the model to reconcile abstract editing commands with pixel-level semantic transformations, ensuring robust generalization across diverse domains.

Following the paradigm of unified sequence modeling, we employ a Double-Stream Multi-Modal Diffusion Transformer (MM-DiT) as the core generative engine. In this configuration, text embeddings derived from the [38] backbone, latent tokens from a high-fidelity VAE, and—in editing scenarios—reference image features are concatenated into a unified input stream. This setup facilitates dense bidirectional interactions between modalities, maximizing the model’s ability to preserve structural integrity while executing precise modifications.

To manage the high-dimensional sequences, we utilize 3D Unified RoPE, wherein reference and target image tokens share spatial coordinates but are differentiated by a temporal interval. This coordinate alignment, coupled with distinct time-conditioning for clean reference and noisy target latents, empowers our model to achieve stateof-the-art (SOTA) performance across multiple competitive benchmarks, demonstrating exceptional stability and stylistic consistency in complex image manipulations.

###### 3.2. Training Efficiency Optimization

Multi-Condition Aware Aspect Ratio Batching. To facilitate efficient distributed training across tasks ranging from Text-to-Image (T2I) generation to complex multi-image referenced editing, we designed a Multi-Condition Aware Bucket Sampler. Unlike traditional samplers that rely solely on single-image resolutions, our approach explicitly accounts for the variable number of input images 𝑁 inherent in different editing instructions. To minimize synchronization overhead and ensure balanced computational loads across GPUs, we define a bucket B𝑟,𝑛 characterized by a target aspect ratio 𝑟 and the count of input images 𝑛. For a given batch, the total visual sequence length 𝐿𝑣𝑖𝑠 is strictly constrained to ensure alignment with the hardware’s computational capacity, formulated as 𝐿𝑣𝑖𝑠 = 𝑛𝑖=1⌈𝐻

𝑖·𝑊𝑖 𝑝2 ⌉ ≈ 𝐶, where 𝑝 denotes the patch size and 𝐶 represents a constant token capacity

per device. This strategy effectively reduces GPU idle time caused by uneven token lengths.

To preserve the structural integrity of diverse spatial layouts (e.g., 1:1, 9:16, 3:4) and ensure tensor consistency within a distributed batch, we minimize the cropping transformation T applied to a set of images {𝐼1, . . . , 𝐼𝑛}. We define the optimal bucket dimensions (ℎ,𝑤) by minimizing the aggregate cropping area, ensuring that the information loss is mathematically minimized:

∑︁𝑛

arg min

|(𝐻𝑖 · 𝑊𝑖) − (ℎ · 𝑤)| (1)

(ℎ,𝑤)∈S

𝑖=1

where S represents the set of predefined resolution buckets. This optimization ensures that all images within a synchronized batch across distributed nodes share identical tensor dimensions, significantly reducing paddinginduced computational waste while maintaining high-fidelity aspect ratio consistency for various input scales.

Stochastic Instruction Alignment. To enhance the model’s robustness against input permutations and prevent over-fitting to specific reference patterns, we implement a Collate Shuffle & Drop mechanism. During the data collation phase, reference images are subjected to a random dropout probability and a stochastic permutation of their input order. Crucially, to maintain semantic alignment between visual tokens and linguistic references, the text prompt is dynamically updated to reflect these spatial changes. For instance, if an image originally tagged as “Fig 1” is permuted to the position of “Fig 2” due to shuffling, the corresponding tokens in the instruction prompt are re-indexed to ensure the text correctly references the shuffled image sequence. This unified augmentation strategy forces the model to decouple specific spatial orderings from semantic content, fostering superior generalization in complex, multi-reference editing scenarios without relying on rigid input sequences.

System-Level Efficiency and Stability. To address the substantial memory demands of the large-scale DiT architecture, we implemented a series of system-level optimizations to maximize training throughput. A key strategy involved decoupling the VLM encoding process by pre-computing embeddings and storing them as offline tensors. This eliminates redundant online computation and reduces per-iteration FLOPs, allowing the system’s compute power to be dedicated entirely to the generative transformer.

Building upon the memory headroom gained from offline feature extraction, we managed distributed model states using Fully Sharded Data Parallel (FSDP). By sharding optimizer states and gradients, we transformed the memory complexity of these overheads from 𝑂(𝑀) to 𝑂(𝑀/𝑁), where 𝑁 is the number of GPUs. To further minimize the peak memory footprint, we integrated gradient checkpointing and utilized BF16 mixed-precision training.

Crucially, the cumulative reduction in VRAM overhead allowed us to transition from a global FSDP approach to the more communication-efficient Hybrid Sharded Data Parallel (HSDP) scheme. While standard FSDP shards model states across the entire cluster—often leading to communication bottlenecks over inter-node links—HSDP constrains sharding to within individual nodes. This allowed us to fully exploit high-speed intra-node interconnects (e.g., NVLink) for scale-up efficiency, while utilizing traditional data parallelism across nodes. By reducing the volume of inter-node synchronization, we harmonized our scale-out strategy with the available network bandwidth, resulting in a marked increase in overall training throughput.

DiffusionNFT

Continued Pre-training

Supervised Fine-tuning

Pre-training

Reinforcement Learning DPO

- Figure 8 | As shown in the figure, our training pipeline comprises progressive stages. We begin with Pretraining and Continued Pre-training to establish robust semantic and aesthetic foundations. The model then undergoes Supervised Fine-tuning to align with editing instructions, followed by Reinforcement Learning (DPO), culminating in the final DiffusionNFT model with enhanced generation quality and adherence.

###### 3.3. Pre-training

During the foundational pre-training phase, our primary objective is to establish a comprehensive visual vocabulary and a robust understanding of world knowledge. Unlike subsequent fine-tuning stages that prioritize aesthetic quality, this phase emphasizes data scale and semantic diversity. We adopt a holistic approach comprising three key pillars:

Roi2

roi1 roi2

| |
|---|

Input

GT

Scalable Framework with Arbitrary-Resolution Adaptation. To accommodate the diverse nature of realworld imagery, we implement a highly efficient and flexible training framework. Rather than adhering to a fixed, square resolution—which often necessitates aggressive cropping or resizing that corrupts image composition—we employ a dynamic bucket sampling strategy. This mechanism automatically groups images of similar aspect ratios into predefined buckets, allowing the model to train on arbitrary resolutions and aspect ratios (VAR) natively. This approach not only preserves the semantic integrity of the visual data but also optimizes computational efficiency by minimizing padding. Furthermore, our framework is architected to seamlessly alternate between text-to-image and single-image input modes. This flexibility enables the model to leverage both paired image-text data for semantic alignment and unpaired image data for purely visual feature learning, maximizing the utilization of available compute resources.

|9:1 6|
|---|

| |
|---|
| |
| |
| |

|1:1|
|---|

Task

3:4 1:1

!"#$!"#

Image Encoder

| |
|---|

Loss + Loss

Bucket 1

id lpips

Roi1

| |
|---|
| |
| |
| |

%&!"#

Input

GT

|1:1|
|---|

|4:3|
|---|

|16:9|
|---|

…

Task

ID similarity Loss

output token

Bucket 2

Input

MMDiT Block

GT

|3:4|
|---|

|3:4|
|---|

|1:1|
|---|

|1:1|
|---|

Task

… … … …

…

Bucket n

Bucket Samplerkey={task}-{img1}-{img2}..{imgn}

Vae Encoder Qwen VL

Prompt :

Fig1

[Figure 187]

[Figure 189]

or Dense

[Figure 190]

Intruction or Dense Caption

A man in Fig1 and a woman in Fig2 were standing on the road, holding a cat in Fig3 between them.

Roi1

drop

Roi2

Fig2

Massive-Scale Data Integration with Hybrid Conditioning. Our data strategy prioritizes the breadth of information, leveraging a massive corpus of data collected from the open internet. Acknowledging the inherent long-tail distribution and variable quality of web data, we adopt an inclusive filtering protocol. Instead of enforcing strict aesthetic curation or artificial class balancing—which might discard rare but semantically valuable concepts—we integrate all "effective" data samples. This allows the model to learn a vast, albeit noisy, representation of the visual world. To address the noise in web texts, we utilize a hybrid captioning scheme where the model is co-trained on: (1) Original Metadata, preserving raw, noisy descriptions to capture rich, unstructured world knowledge and diverse linguistic styles; and (2) Structured Synthetic Captions, generated to provide precise, tag-heavy descriptions that enhance the model’s prompt-following capabilities and object recognition. This mixture ensures the model acquires generalized knowledge while maintaining the ability to respond to structured prompts.

[Figure 191]

Prompt :

[Figure 192]

[Figure 193]

Fig3 Fig2

Collate Shuffle & Drop

… Fig1 … … … Fig2 …. …Fig3....

drop

Select the same bucket for all processes

Bucket Sampler

Collate Shuffle & Drop

Curriculum Learning via Progressive Timestep Sampling. Given the high variance in data quality and the massive scale of the dataset, training stability is paramount. We introduce a progressive timestep curriculum to govern the noise diffusion process. In the initial phase, sampling is biased towards high-noise timesteps. Since high noise levels correspond to the low-frequency components of an image, this strategy forces the model to prioritize learning global semantic structures, layouts, and large-scale visual patterns, which are generally robust even in lower-quality data. As training converges, we gradually shift the distribution towards uniform sampling across all timesteps. This transition allows the model to refine its generation of high-frequency details and textures. This curriculum effectively mitigates the risk of divergence when training on noisy internet data and ensures a smooth transfer from learning coarse concepts to mastering fine-grained details.

###### 3.4. Continued Pre-training (CT)

Unified Task Sampling and Resolution Adaptability To enhance the model’s versatility across different generation paradigms, the Continued Pre-training (CT) phase adopts a unified sampling strategy that balances inputs across Text-to-Image, single-image, and multi-image tasks. This joint training regime leverages the computational budget to solidify the model’s ability to handle diverse input modalities simultaneously. Furthermore,

- Table 1 | Training Stages and Hyperparameters

Stage Attribute Pretrain CT SFT DPO NFT Dataset

Internet Collection 100M 5M 50K 10K 10K Synthetic data 5M 10M 50K 50K 10K

Original Cap. 55% 20% 10% 10% 10% Structured Cap. 40% 60% 45% 30% 30% Instructive Cap. 5% 20% 45% 60% 60%

Captioning

Resolution(pixel) 384-512 512-1024 1024 1024 1024 Training Steps 300K 65K 5K 5K 500 Warm-up steps 0 0 500 500 0

Train Recipe

Optimizer AdamW (𝛽1 = 0.9, 𝛽2 = 0.999) Gradient clip 1 Weight decay 0.01

we significantly broaden the scope of our resolution management by implementing a more extensive bucket sampling protocol. We explicitly train on a comprehensive spectrum of aspect ratios—including 2:1, 16:9, 3:2,

- 4:3, 1:1, 3:4, 2:3, 9:16, and 1:2. This wide-ranging aspect ratio coverage prevents overfitting to standard formats and ensures the model maintains compositional integrity and visual fidelity regardless of the canvas dimensions requested by the user.

Synthetic Augmentation and Dense Semantic Alignment The composition of the training data during this stage is meticulously engineered to bridge the gap between quantity and quality. We maintain a curated stream of high-quality data sampled from the Internet, supplemented by a significant proportion of synthetic data designed to cover underrepresented domains. A critical innovation in this phase is the integration of dense caption training. By utilizing comprehensive, highly detailed descriptions for training samples, we compel the model to align visual features with complex and specific textual cues. This approach is particularly effective for learning long-tail vocabulary, ensuring that the model develops a robust understanding of rare objects, nuanced textures, and intricate scenarios that are often glossed over by sparse captions.

Cluster-Based Distribution Balancing To address the inherent imbalances found in large-scale datasets, we enforce a rigorous class-balancing protocol throughout the CT stage. Rather than relying on raw data frequency, we categorize the entire dataset into distinct semantic clusters and apply a uniform sampling strategy across these groups. This method ensures that every category—from common objects to niche artistic styles—receives sufficient exposure during training. By guaranteeing that each semantic cluster is fully and equally trained, we mitigate the risk of mode collapse or overfitting to dominant classes, resulting in a model that exhibits consistent high-quality generation capabilities across a universally diverse range of subjects.

###### 3.5. Supervised Fine-Tuning (SFT)

High-Fidelity Data Alignment and Balanced Distribution. To transition the model from a broad, noisy pretraining distribution to a focused, high-quality sub-manifold, we construct a hierarchically organized dataset dominated by high-resolution imagery (1024×1024) that has undergone meticulous human filtering. Unlike the loose supervision in pre-training, our SFT phase utilizes instruction-following captions and structured prompts to enforce precise alignment between textual descriptions and visual outputs. This rigorous supervision acts as an anchor, forcing the model to discard low-quality modes (e.g., artifacts or inconsistent rendering) and converge towards superior textural quality. Furthermore, to address the risk of catastrophic forgetting regarding long-tail concepts, we enforce strict consistency across different semantic clusters. By employing a dynamic resampling strategy that ensures strict balance among data categories, we preserve the semantic diversity of the pre-trained model while elevating its aesthetic fidelity.

Refined Optimization with Model Weight Averaging. During the optimization process, we adopt a significantly smaller learning rate compared to the pre-training phase. This conservative update strategy is designed to refine high-frequency details and optimize photorealistic attributes without disrupting the global structural understanding established earlier. To further enhance the model’s generation stability and aesthetic robustness, we incorporate an Exponential Moving Average (EMA) strategy, also referred to as Model Weight Averaging. By maintaining a moving average of model parameters throughout the training trajectory, we effectively smooth out the optimization landscape. This fusion process helps synthesize capabilities from different training steps, mitigating specific biases inherent in individual checkpoints and ensuring the final model achieves a balanced

performance with greater generalization capabilities.

- 3.6. Reinforcement Learning with Human Feedback (RLHF)

- 3.6.1. Direct Preference Optimization (DPO)

We optimize the model using a DPO framework enhanced with Positive Sample Reinforcement (PSR) to ensure training stability. The alignment relies on a two-stage data synthesis strategy, combining Mix-Policy construction for robust instruction following with specific Visual-Enhancement sampling for aesthetic fidelity.

Positive Sample Reinforcement (PSR) Strategy While standard DPO provides a robust framework, our experiments reveal a "double degradation" phenomenon. Specifically, we observe that the Win Diff and Lose Diff terms often exhibit a synchronous upward trend during training. This indicates that the optimization trajectory deviates from the high-quality data manifold; the model effectively degrades its generation capability for chosen samples (𝑥𝑤) while attempting to distance itself from rejected ones (𝑥𝑙).

This instability suggests that in continuous high-dimensional spaces, simply penalizing negative samples is insufficient to guide the policy toward the desired distribution. To address this, we propose an Asymmetric Gradient Optimization strategy that anchors alignment on Positive Sample Reinforcement (PSR). We introduce a weighting coefficient 𝜔 and an SFT regularization term 𝜆 to explicitly prioritize the reconstruction of high-quality distributions:

LOurs = −E(𝑐,𝑥

𝑤,𝑥𝑙)∼D log𝜎 𝛽[(L𝑙𝜃 − L𝑙ref)

Lose Diff

−𝜔 · (L𝑤𝜃 − L𝑤ref)

Win Diff

] − 𝜆L𝑤𝜃 (2)

By setting 𝜔 > 1, we amplify the gradient contribution from the Win Diff term, ensuring that the optimization is primarily driven by the high-fidelity modes (𝑥𝑤) rather than being misguided by the unconstrained avoidance of negative samples.

Automated Data Synthesis and Mix-Policy Strategy To bypass the costs of human annotation, we implement a fully automated pipeline that employs a Multi-Stage Instruction Evolution strategy, using VLMs to synthesize instructions ranging from atomic tasks to complex real-world requests. To avoid the "capability ceiling" inherent in self-sampling, we adopt a Mix-Policy data construction method. Instead of relying solely on the SFT model, we source high-quality positive samples () from diverse expert branches and aesthetic-optimized models, contrasting them with VLM-filtered negative samples () from the base model. This approach creates a steeper learning signal, breaking the self-reinforcing loop of self-generation while specifically enhancing visual fidelity and texture quality.

- 3.6.2. Diffusion NFT

In this stage, we conduct online reinforcement learning via DiffusionNFT [54], optimizing a composite reward derived from the Fine-grained VLM and Layout-Aware OCR models. This process is further enhanced by a semi-hard sample mining strategy to ensure data efficiency.

Diffusion Negative-aware FineTuning We employ DiffusionNFT to perform online reinforcement learning directly on the forward process. Distinct from DPO which requires paired data, DiffusionNFT utilizes the optimality probability 𝑟 ∈ [0,1] of each online sampled image. The objective minimizes the weighted flow matching error via implicitly parameterized positive and negative policies:

###### +(1 − 𝑟) ∥𝑣𝜃−(𝑥𝑡,𝑡) − 𝑣∥2

LNFT = E𝑡,𝑥0∼𝜋old 𝑟 ∥𝑣𝜃+(𝑥𝑡,𝑡) − 𝑣∥2

Negative Match

Positive Match

(3)

𝜃 = (1 − 𝛽)𝑣old + 𝛽𝑣𝜃 and 𝑣−

where 𝑣 denotes the target velocity, and the implicit policies are defined as 𝑣+

𝜃 = (1 + 𝛽)𝑣old − 𝛽𝑣𝜃. Theoretically, this formulation defines a contrastive improvement direction that pushes the policy towards high-reward regions while explicitly penalizing low-reward trajectories.

Fine-grained Logit-Weighted Ensembling Reward Leveraging the specialized reward models detailed in Sec. 2.6, we derive dense reinforcement signals across critical dimensions. These models incorporate Chain-ofThought (CoT) reasoning prior to scoring, ensuring alignment with nuanced human judgment. To circumvent the optimization instability inherent in sparse and discrete integer rewards, we adopt a continuous soft-scoring

mechanism during the VLM forward pass. Specifically, immediately following the generated CoT sequence, we extract the logits corresponding to numeric rating tokens to compute a Softmax-normalized probabilityweighted expectation rather than relying on hard argmax predictions. By averaging these soft scores across an ensemble of 𝐾 stochastic inference passes, we effectively smooth the reward landscape, guaranteeing a continuous and stable reinforcement signal 𝑅 for robust optimization:

###### ∑︁𝐾

∑︁

exp(𝑧𝑣) 𝑣′∈V exp(𝑧𝑣′)

1

𝑅(x,y) =

𝑣 · 𝑃(𝑣 | x,y,c𝑘), with 𝑃(𝑣 | ·) =

, (4)

𝐾

𝑘=1

𝑣∈V

where c𝑘 denotes the CoT rationale generated in the 𝑘-th pass, V represents the set of numeric tokens (e.g., {1, . . . ,5}), and 𝑣 is the scalar value associated with the token.

Layout-Aware OCR-based Reward For text-editing tasks, we introduce a layout-aware OCR-based reward that more accurately reflects both textual fidelity and spatial consistency. Conventional OCR-based rewards depend solely on the edit distance between the recognized text and the target string. Although straightforward, this ignores typography and layout, allowing models to hack the reward by generating oversized characters that OCR systems recognize more easily but blend poorly with the image and disrupt layout consistency. To overcome this limitation, we incorporates layout consistency alongside text correctness. Rather than treating OCR outputs as a flat string, we decompose both predicted and reference OCR outputs into character-level elements with positions and scales, and assess whether each character appears in a plausible location and size. This formulation penalizes missing, misaligned, or abnormally scaled characters while retaining the semantic alignment signal. A lightweight gating mechanism further ensures that layout terms activate only when the text content is reasonably correct. The total reward is formulated as:

###### ∑︁

𝑑(𝑠pred, 𝑠tgt) max(|𝑠tgt|,1)

|𝑠pred| |𝑠tgt|

𝑒−𝑑𝑖𝑒−Δ𝑠𝑖

RLA-OCR = 𝑤text 1 −

(5)

+𝑤layout 𝐺𝑎𝑡𝑒(Rtext)

𝑖

Rtext

Rlayout

where 𝑠pred and 𝑠tgt are the OCR-predicted and target text strings, and 𝑑𝑖 and Δ𝑠𝑖 denote the center-distance and overscaling penalty for the 𝑖-th matched character. This design mitigates reward hacking and greatly reduces text collapse, resulting in more stable glyphs and more natural typography in text-editing tasks.

Semi-Hard Sample Mining for Data Efficiency We empirically observe that naive random sampling yields negligible gains, as the model inefficiently allocates capacity to samples already within its comfort zone. To mitigate this inefficiency, we introduce a hardness-aware curriculum that utilizes the preceding DPO checkpoint for offline hardness estimation. By generating a diverse set of candidate edits for each instruction, we estimate the underlying reward distributions. We specifically target “semi-hard” instances, defined by a distinct performance profile: competent yet high-variance. These samples typically exhibit a satisfactory mean reward—indicating that the model grasps the semantic alignment—yet suffer from significant instability in their lower-bound scores. Unlike trivial samples that offer limited information gain, or completely hard samples that often precipitate optimization collapse, these semi-hard instances maximize gradient utility by targeting the model’s capability boundary. By focusing on samples that are conceptually feasible but lack robustness, this strategy maximizes the information gain per training step, thereby significantly enhancing both the consistency and fidelity of the editing process.

###### 3.7. Consistency Loss

We observe that maintaining identity in general-purpose image editing remains a significant challenge when relying solely on standard mean squared reconstruction losses, particularly in human-centric applications. While these objectives are effective for ensuring low-level pixel fidelity, they often fail to capture the high-level semantic nuances required for identity consistency, leading to the undesirable phenomenon of identity drift. Given the high practical value and growing demand for human-centric image editing, there is an urgent need for mechanisms that can robustly preserve identity throughout the editing process. Therefore, we integrate a robust identity consistency constraint into our training framework.

The denoising process in rectified flow follows a distinct coarse-to-fine progression. We observe that the early stage (high-noise regime) is pivotal for identity formulation, where global semantic structures are established, whereas the later stage focuses on the refinement of fine-grained textures. We argue that as the model enters the low-noise regime, the noisy input image already carries a stable identity structure. Since the identity is effectively “locked” within the input at this point, further identity-driven optimization is not only less

likely to alter established facial features but may also introduce unintended visual artifacts by competing with the model’s ability to synthesize fine-grained details. Driven by this insight, we propose a dynamic weight scheduling for the identity loss L𝑖𝑑. Specifically, the weight 𝜆𝑖𝑑 is defined as a function of the noise level 𝜎:

𝜂 · 𝜎2, if 𝜎 < 0.9 0, otherwise

(6)

𝜆𝑖𝑑(𝜎) =

where 𝜂 is a scaling hyperparameter. This quadratic decay ensures the constraint gradually tapers off as the model transitions from semantic anchoring to pixel-level refinement. The total training objective is then formulated as:

L𝑡𝑜𝑡𝑎𝑙 = L𝑚𝑠𝑒 + 𝜆𝑖𝑑(𝜎) · L𝑖𝑑. (7)

The implementation of the identity loss L𝑖𝑑 begins by defining a differentiable spatial transformation T designed to extract and normalize the Region of Interest (ROI) from the image. For human-centric editing, this transformation is specifically derived from facial landmarks detected in the ground-truth image 𝑥𝑔𝑡, which aligns the face into a canonical representation. During training, we obtain the one-step denoised estimate 𝑥ˆ0 from the predicted velocity 𝑣𝑡 as 𝑥ˆ0 = 𝑥𝑡 − 𝜎𝑡 · 𝑣𝑡. Although 𝑥ˆ0 may be blurry or contain artifacts at higher noise levels, pre-trained semantic backbones—most notably face recognition models—exhibit significant robustness to such degradations. This resilience ensures that L𝑖𝑑 can provide relative stable semantic guidance and effectively optimize the subject’s identity even when the underlying image reconstruction is coarse. To maintain strict spatial correspondence and ensure training stability, we apply the identical transformation T (parameterized by 𝑥𝑔𝑡) to the predicted 𝑥ˆ0, ensuring the loss is computed on perfectly aligned regions.

This framework naturally extends to multi-subject scenarios. For an image containing 𝑁 individuals, we detect and align each face independently to obtain a set of transformations {T𝑖}𝑖𝑁=1. The total identity loss is formulated as the average cosine distance across all detected subjects:

###### ∑︁𝑁

𝜙(T𝑖(𝑥ˆ0)) · 𝜙(T𝑖(𝑥𝑔𝑡)) ∥𝜙(T𝑖(𝑥ˆ0))∥2 · ∥𝜙(T𝑖(𝑥𝑔𝑡))∥2

1

1 −

, (8)

L𝑖𝑑 =

𝑁

𝑖=1

where 𝜙(·) denotes the pre-trained face recognition backbone. By aggregating the loss across all subjects, our model effectively anchors the unique identity of each individual during the critical structural formation window while preserving textual integrity. The alignment is extensible and could potentially be adapted to the preservation of other universal objects by employing appropriate semantic encoders.

###### 3.8. Training Strategy.

To enhance training stability and maximize data efficiency, our framework incorporates a rigorous optimization strategy designed to ensure robust model generalization. This approach coordinates global timestep distribution through a distributed stratified sampling mechanism and refines parameter convergence via a post-training weight averaging protocol.

Distributed Stratified Timestep Sampling. In standard distributed training, independent random sampling of diffusion timesteps across GPUs often leads to statistical clusters, preventing the global batch from uniformly covering the noise spectrum. To address this, we implement a Discrete Stratified Sampling strategy. The total diffusion timestep horizon 𝑇 is decomposed into 𝐾 disjoint equidistant sub-intervals, where 𝐾 corresponds to the distributed world size (or group size). At each training iteration, every computational rank is assigned a specific sub-interval from which to sample 𝑡 uniformly. To prevent inductive bias—where specific ranks overfit to fixed noise levels—we employ a synchronized interval rotation mechanism. A global permutation seed is broadcast periodically to dynamically reassign noise intervals among ranks. This ensures that the aggregated global batch approximates a perfect uniform distribution over 𝑡 ∈ [0,𝑇] at every step, significantly stabilizing the convergence of the diffuion objective.

Logit-Normal Loss Weighting. While the sampling probability is globally uniform, the contribution of each timestep to the gradient is modulated by a Logit-Normal weighting scheme. The loss weight is determined based on the noise level, following a distribution centered at the meaningful middle trajectory of the diffusion process. This allows the optimizer to suppress contributions from negligible high-noise or low-noise extremes while concentrating capacity on the critical intermediate phase where semantic structure and texture are primarily established.

Model Weight Averaging. To synthesize the strengths of the model across the optimization trajectory, we adopt a Exponential Moving Average (EMA) protocol. Rather than relying on a single final checkpoint, which may be sensitive to the stochasticity of the last few batches, we compute the arithmetic mean of model parameters 𝜃

over the converging iterations. This ensemble-in-weights approach smoothens the loss landscape, effectively neutralizing transient fluctuations and enhancing the model’s robustness against distribution shifts in real-world editing scenarios.

###### 4. REDEdit-Bench

Background Change

To better evaluate instruction-following fidelity and preservation quality, we introduce REDEdit-Bench, a benchmark containing 1,673 bilingual (Chinese–English) edit pairs spanning 15 structured editing categories. As shown in Fig. 9, the benchmark provides a balanced distribution across all task categories. As summarized in Table 2, REDEdit-Bench demonstrates clear advantages over widely used open-source edit benchmarks in both scale and evaluation design.

View Change

Text Modiﬁcation

Color Alteration

(3.4%)

(6.0%)

(7.8%)

(6.0%)

SubjectReplacement

ComposeEdit

(6.2%)

(9.0%)

Portrait

SubjectExtract

(9.0%)

(6.1%)

REDEdit Bench

LowlevelEdit

SubjectRemoval

(3.0%)

MotionChange(5.0%)

(9.0%)

- 4.1. Benchmark Construction

We collect over 3,000 real-world images from the internet and legitimate procurement channels, covering natural landscapes, architecture, objects, animals, and portraits. Professional technical staff first select each image and write corresponding editing instructions, ultimately constructing 1,673 bilingual (Chinese–English) edit pairs. These pairs are further reviewed and verified by multiple experts to ensure the diversity of the editing instructions and the legality of the image sources.

Table 2 | Comparison of key attributes of open-source image editing benchmarks. REDEdit-Bench achieves the largest scale and is the only benchmark covering all listed attributes.

Benchmarks Size Real Image Human Filtering Sub-tasks Bilingual Task-Specific Evaluation Prompts

MagicBrush [50] 1,053 ✔ ✔ 7 ✘ ✘ AnyEdit [30] 1,250 ✔ ✘ 25 ✘ ✘ ImgEdit [48] 811 ✔ ✔ 14 ✘ ✔ GEdit-Bench [21] 606 ✔ ✔ 11 ✔ ✘

REDEdit-Bench (Ours) 1,673 ✔ ✔ 15 ✔ ✔

- 4.2. Evaluation Pipeline

Based on REDEdit-Bench, we conduct a comprehensive evaluation of mainstream image editing models, with a primary focus on Prompt Compliance, Visual Naturalness, and Physical & Detail Coherence. With Gemini

- 3 Flash employed as the automated evaluator, our model achieves SOTA performance among open-source models.

Subject Adjustment

(4.8%)Beauty

Style Transfer

Subject Addition (8.8%)

(9.9%)

(6.0%)

Figure 9 | Task category distribution of REDEdit-Bench.

For text editing tasks, we introduce two novel metrics: OCR and VLM Judge. The OCR metric quantifies character generation precision by comparing OCR results against the ground truth to calculate the Levenshtein distance, completion rate, and word accuracy, producing a weighted normalized score. To address OCR’s limitation of focusing solely on spelling, the VLM Judge utilizes Multimodal Large Language Models to evaluate visual quality. It assesses metrics including SuccessEdit, OverEdit, Style, and Consistency (background fusion). Together, these metrics construct a comprehensive evaluation balancing character accuracy with visual fidelity.

###### 5. Performance and Evaluation

We evaluate our method through both human assessment and standardized benchmark testing to provide a comprehensive analysis of its effectiveness. As detailed in Section 5.1, we first conduct a structured human evaluation where annotators assess model outputs across carefully defined dimensions. Section 5.2 then presents quantitative and qualitative results on benchmark datasets, covering four representative editing categories: general editing, text-centric editing, creative editing, and virtual try-on editing. This dual evaluation framework enables a thorough validation from both perceptual and objective perspectives.

- 5.1. Human Evaluation

|65.7%|34.3%|
|---|---|

|57.4%|42.6%|
|---|---|

|44.3%|55.7.%<br><br>[Figure 194]|
|---|---|

|55.3%|44.7%<br><br>[Figure 195]|
|---|---|

|56.1%|43.9%|
|---|---|

|61.1%|38.9%|
|---|---|

|51.2%|48.8%<br><br>[Figure 196]|
|---|---|

|55.0%|45.0%<br><br>[Figure 197]|
|---|---|

Prompt Following Consistency Preservation

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

|[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>LongCat Nano-Banana-Pro Qwen-Image-Edit-2511 FireRed-Image-Edit Seedream4.5|
|---|

- Figure 10 | The figure compares FireRed-Image-Edit against four other models across two metrics. FireRedImage-Edit achieves the highest consistency scores and leads in text-to-image alignment against most competitors, except for Nano-Pro which slightly outperforms it in that category.

To comprehensively assess the perceptual quality and practical effectiveness of our image editing foundation model, we conduct a multi-model blind human evaluation against strong open-source and commercial baselines. Instead of pairwise comparison, multiple model outputs for the same input are presented simultaneously in randomized order without revealing model identities. Annotators independently evaluate each result to ensure fairness and eliminate positional or brand bias. This evaluation protocol is designed to jointly measure the correctness of instruction execution and the perceptual quality of the edited results from multiple complementary perspectives.

- 5.1.1. Evaluation Dimensions We evaluate each editing result along two carefully designed dimensions:

Prompt Following. This dimension measures how accurately and completely the model understands and executes the user-provided prompt. Annotators assess whether the intended edit type is correctly identified, all key constraints are satisfied, and the final output faithfully reflects the user’s editing intent without missing or extraneous modifications.

Consistency Preservation. This metric focuses on the protection of non-edited regions. Ideally, only regions explicitly specified by the prompt should be modified, while all other content—such as facial identity, background, object structure, texture, and global layout—remains unchanged. This dimension is particularly critical for localized and iterative editing scenarios.

- 5.1.2. Results Analysis

As illustrated in Fig. 10, our model achieves leading performance across Prompt Following and Consistency. FireRed-Image-Edit significantly outperforms open-source baselines such as LongCat and Qwen-Image-Edit2511, and remains competitive with commercial systems, indicating strong instruction understanding and execution capability.

For Consistency Preservation, our model obtains the highest score among all compared methods, demonstrating superior ability to modify only the intended regions while preserving non-edited content. This highlights its robustness in controlled and precise editing scenarios.

- 5.2. Quantitative and Qualitative Results

- 5.2.1. General Editing

ImgEdit. We evaluate our model on the ImgEdit benchmark, which measures instruction completion and visual quality. Table 3 shows that FireRed-Image-Edit achieves state-of-the-art performance, surpassing both open-source and closed-source models. This result highlights the model’s proficiency in following common editing commands.

GEdit. We assess our model’s capabilities on the GEdit benchmark, a comprehensive evaluation suite designed to test both instruction adherence and output visual fidelity. As illustrated in Table 4, FireRed-Image-Edit

- Table 3 | Results on ImgEdit-Bench (Category-wise and Overall Performance). Best results are shown in bold and second-best results are underlined.

Model Add Adjust Extract Replace Remove Background Style Hybrid Action Overall↑

Nano-Banana [10] 4.62 4.41 3.68 4.34 4.39 4.40 4.18 3.72 4.83 4.29 Seedream4.0 [29] 4.33 4.38 3.89 4.65 4.57 4.35 4.22 3.71 4.61 4.30 Seedream4.5 [29] 4.57 4.65 2.97 4.66 4.46 4.37 4.92 3.71 4.56 4.32 Nano-Banana-Pro [10] 4.44 4.62 3.42 4.60 4.63 4.32 4.97 3.64 4.69 4.37

Instruct-Pix2Pix [2] 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 MagicBrush [50] 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.90 AnyEdit [49] 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit [53] 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 OmniGen [44] 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 ICEdit [52] 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 BAGEL [5] 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 UniWorld-V1 [19] 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 OmniGen2 [40] 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 Dreamomini2 [42] 3.93 3.09 2.11 3.95 3.64 3.75 4.38 2.90 4.04 3.53

- FLUX.1 Kontext [Dev] [15] 3.99 3.88 2.19 4.27 3.13 3.98 4.51 3.23 4.18 3.71 Step1X-Edit-v1.2 [20] 3.91 4.04 2.68 4.48 4.26 3.90 4.82 3.23 4.22 3.95 Qwen-Image-Edit-2509 [38] 4.34 4.27 3.42 4.73 4.36 4.37 4.91 3.56 4.80 4.31

- FLUX.2 [Dev] [14] 4.50 4.18 3.83 4.65 4.65 4.31 4.88 3.46 4.70 4.35 Emu3.5 [4] 4.61 4.32 3.96 4.84 4.58 4.35 4.79 3.69 4.57 4.41 ChronoEdit [41] 4.48 4.39 3.49 4.66 4.67 4.57 4.91 3.82 4.83 4.42 LongCat-Image-Edit [32] 4.44 4.53 3.83 4.80 4.60 4.33 4.92 3.75 4.82 4.45 Qwen-Image-Edit-2511 [38] 4.54 4.57 4.13 4.70 4.46 4.36 4.89 4.16 4.81 4.51 FireRed-Image-Edit 4.55 4.66 4.34 4.75 4.58 4.45 4.97 4.07 4.71 4.56

- Table 4 | Results on GEdit (Category-wise and Overall Performance). Best results are shown in bold and second-best results are underlined.

GEdit-Bench-EN GEdit-Bench-CN G_SC↑ G_PQ↑ G_O↑ G_SC↑ G_PQ↑ G_O↑

Model

Nano-Banana [10] 7.396 8.454 7.291 7.540 8.424 7.399 Seedream4.0 [29] 8.143 8.124 7.701 8.159 8.074 7.692 Nano-Banana-Pro [10] 8.102 8.344 7.738 8.135 8.306 7.799 Seedream4.5 [29] 8.268 8.167 7.820 8.254 8.167 7.800

FLUX.2 [Dev] [14] 7.835 8.064 7.413 7.697 8.046 7.278 Qwen-Image-Edit-2509 [38] 7.974 7.714 7.480 7.988 7.679 7.467 Step1X-Edit-v1.2 [20] 7.974 7.714 7.480 7.988 7.679 7.467 Longcat-Image-Edit [32] 8.128 8.177 7.748 8.141 8.117 7.731 Qwen-Image-Edit-2511 [38] 8.297 8.202 7.877 8.252 8.134 7.819 FireRed-Image-Edit 8.363 8.245 7.943 8.287 8.227 7.887

outperforms existing open-source models. This superior performance underscores the model’s exceptional command comprehension and precise execution of diverse editing tasks.

REDEdit-Bench. REDEdit-Bench is our self-built benchmark to evaluate real-world editing usage, with stronger emphasis on complex composite instructions and preservation constraints. It is constructed in Section 4. Table 5- 6 summarizes the main results. FireRed-Image-Edit ranks highest among open-source models, demonstrating competitive instruction-based editing performance.

Visualization We show general-purpose editing cases that correspond to frequently encountered and welldefined user requests. These examples include object insertion and replacement (add, replace), attribute and appearance modification (color), hybrid edit tasks (compose), as well as image restoration tasks (denoted as low-level). Such tasks require precise alignment between the instruction and the results, while maintaining global consistency and avoiding unnecessary changes to unrelated regions. The presented Figs. 11–13 demonstrate that FireRed-Image-Edit can reliably perform these edits with accurate localization, physically plausible interactions, and stable visual quality, making it suitable for a wide range of everyday image editing applications.

- Table 5 | Results on REDEdit-Bench-CN (General dimensions). Best results are shown in bold and second-best results are underlined.

Model Overall Add Adjust BG Beauty Color Compose Extract Portrait Low-level Motion Remove Replace Stylize Text Viewpoint

Nano-Banana [10] 4.13 4.66 4.26 4.63 4.37 4.13 3.94 3.17 4.83 4.05 4.75 4.07 4.74 3.63 3.69 3.09 Seedream4.0 [29] 4.15 4.55 4.11 4.61 3.83 4.14 4.16 2.48 4.77 4.17 4.68 4.02 4.53 4.94 3.94 3.29 Seedream4.5 [29] 4.18 4.58 4.09 4.57 3.97 4.12 4.05 2.56 4.80 3.99 4.78 4.12 4.53 4.94 4.07 3.53 Nano-Banana-Pro [10] 4.48 4.66 4.41 4.58 4.35 4.58 4.36 3.42 4.86 4.46 4.91 4.54 4.79 4.85 4.69 3.75

Qwen-Image-Edit-2509 [38] 4.00 4.45 4.04 4.48 3.36 4.20 3.92 2.64 4.16 3.52 4.66 4.27 4.66 4.81 3.53 3.32 FLUX.2 [Dev] [14] 4.05 4.31 3.88 4.57 3.80 3.91 3.85 2.47 4.50 4.43 4.68 3.50 4.47 4.95 3.53 3.88 Longcat-Image-Edit [32] 4.12 4.34 4.25 4.54 3.72 4.12 3.92 2.48 4.49 4.31 4.67 4.27 4.61 4.94 3.83 3.30 Qwen-Image-Edit-2511 [38] 4.18 4.50 4.23 4.52 3.61 4.09 4.00 3.22 4.31 4.19 4.66 4.41 4.68 4.83 4.08 3.51 FireRed-Image-Edit 4.33 4.57 4.37 4.64 3.69 4.45 4.29 3.49 4.50 4.56 4.65 4.47 4.81 4.93 4.49 3.14

- Table 6 | Results on REDEdit-Bench-EN (General dimensions). Best results are shown in bold and second-best results are underlined.

Model Overall Add Adjust BG Beauty Color Compose Extract Portrait Low-level Motion Remove Replace Stylize Text Viewpoint

Nano-Banana [10] 4.15 4.65 4.23 4.60 4.37 4.08 3.98 3.39 4.72 4.03 4.63 4.07 4.68 3.68 3.87 3.23 Seedream4.0 [29] 4.18 4.59 4.12 4.63 3.89 4.10 4.14 2.28 4.77 4.12 4.73 4.23 4.56 4.98 4.21 3.42 Seedream4.5 [29] 4.20 4.66 4.08 4.64 4.12 4.07 4.10 2.23 4.74 4.28 4.75 4.24 4.58 4.97 4.20 3.44 Nano-Banana-Pro [10] 4.42 4.72 4.40 4.64 4.37 4.43 4.32 3.25 4.82 4.36 4.85 4.52 4.75 4.90 4.54 3.51

Qwen-Image-Edit-2509 [38] 3.99 4.47 4.06 4.49 3.13 3.98 3.85 2.91 4.30 3.71 4.58 4.40 4.67 4.77 3.77 2.85 FLUX.2 [Dev] [14] 4.07 4.37 3.96 4.47 3.72 3.86 3.87 2.36 4.44 4.45 4.67 4.02 4.48 4.87 3.80 3.84 LongCat-Image-Edit [32] 4.12 4.38 4.04 4.49 3.89 4.10 3.93 2.98 4.47 4.27 4.69 4.24 4.51 4.86 3.83 3.25 Qwen-Image-Edit-2511 [38] 4.23 4.55 4.17 4.56 3.49 4.07 4.07 3.54 4.42 4.52 4.72 4.42 4.65 4.85 4.06 3.38 FireRed-Image-Edit 4.26 4.41 4.33 4.60 3.55 4.47 4.25 3.49 4.50 4.44 4.65 4.46 4.70 4.94 4.44 2.78

###### 5.2.2. Text-Centric Editing

Text-Centric Evaluation. To further evaluate text editing capabilities, we apply our proposed OCR and VLM Judge pipeline to the text subset of REDEdit-Bench, assessing performance across five specific dimensions. The evaluation examines the accuracy of edited content at the character and word level, the preservation of original font style and visual attributes, the overall visual consistency between the edited text and the surrounding image, and whether unnecessary modifications are introduced beyond the given instruction. This setting enables a more fine-grained analysis of text-centric behavior. As shown in Table 7, FireRed-Image-Edit achieves leading results among open-source models, demonstrating strong capability in accurate text modification while maintaining style fidelity and global coherence.

Visualization. We additionally present text-centric editing results in Figs. 16–17 focusing on complex text generation and poster-style visual design. The showcased examples include long-form text synthesis with consistent typography and layout, precise text replacement and correction while preserving font style and visual appearance, as well as poster and cover editing scenarios where text content must be harmoniously integrated with the underlying imagery. These cases require accurate instruction following at the character and word level, together with global layout awareness and aesthetic consistency across text, graphics, and background. The results demonstrate that FireRed-Image-Edit can reliably handle both content correctness and visual coherence in text-heavy editing scenarios, making it suitable for practical applications such as poster design, cover editing, and creative visual communication.

###### 5.2.3. Creative Editing

Creative editing differs from image editing tasks in that it often requires deeper instruction understanding, implicit scene reasoning, and global compositional restructuring. The challenge lies in synthesizing content that follows the text but remains visually consistent with the original image.

Visualization. Fig. 15 demonstrates the creative editing capabilities of FireRed-Image-Edit . The displayed results range from structural abstraction (e.g., design sketches, cross-sectional views) and concept-driven synthesis (e.g., sculptural styles, foldable designs) to imaginative edits that defy physical constraints (e.g., floating objects). While non-exhaustive, these examples demonstrate that FireRed-Image-Edit can effectively translate high-level creative intents into coherent visual outcomes.

###### 5.2.4. Try-on Editing

We further evaluate FireRed-Image-Edit under structured virtual try-on settings, where the model is tasked with transfer garments from a reference image onto a target person while simultaneously following detailed styling instructions.

- Table 7 | Results on REDEdit-Bench (Text dimension).

Model OCR SuccessEdit OverEdit Style Consistency

Nano-Banana [10] 0.958 7.93 9.53 9.30 8.04 Seedream4 [29] 0.969 8.59 9.26 9.19 9.15 Seedream4.5 [29] 0.975 8.61 9.37 9.00 8.93 Nano-Banana-Pro [10] 0.984 9.54 9.63 9.68 9.53

FLUX.2 [Dev] [14] 0.950 7.93 8.90 8.83 8.33 Qwen-Image-Edit-2509 [38] 0.957 8.35 9.26 9.15 8.15 Qwen-Image-Edit-2511 [38] 0.969 9.42 9.36 9.31 9.27 LongCat-Image-Edit [32] 0.976 8.60 8.96 8.67 8.50 FireRed-Image-Edit 0.983 9.57 9.53 9.49 9.51

Visualization. As illustrated in Fig. 14, each case involves combining the outfit from one image with a target model under additional constraints on garment fit (e.g., loose or tight silhouette), length, color specification, and accessory composition (e.g., handbags, jewelry, footwear). Such scenarios require accurate garment preservation, natural deformation under body pose, and consistent integration with newly introduced items, while maintaining identity and overall visual realism. Compared with existing editing baselines, FireRed-ImageEdit produces more coherent clothing geometry, cleaner boundary transitions, and better alignment between textual styling instructions and visual outcomes. These results indicate that the model can handle compositional multi-constraint editing in a stable and controllable manner, making it suitable for practical fashion and styling applications.

Adding someone to the bed who is using a laptop

Input Image Qwen-Image-Edit

Flux2

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Nano-Banana Pro Ours

Seedream4.0

Wear a delicate silver bracelet adorned with small snowflake pendants on your lower wrist

Input Image Qwen-Image-Edit Flux2

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Seedream4.0 Nano-Banana Pro Ours

- Figure 11 | Demonstration of Object Addition, where the model seamlessly inserts new elements into the scene, maintaining consistency with the original context. Our model outperforms competitors by strictly adhering to instructions and ensuring the added objects blend naturally with the scene.

Turn the chocolate sauce drizzled on the waffles and plate into a yellow mayonnaise

Input Image Qwen-Image-Edit

Flux2

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Nano-Banana Pro Ours

Seedream4.0

Replace the white sauce with ketchup and expand the image outwards Input Image Qwen-Image-Edit Flux2

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Seedream4.0 Nano-Banana Pro Ours

- Figure 12 | Examples of Object Modification, demonstrating precise control over object attributes. Our model achieves accurate editing without disrupting the surrounding textures, whereas competitors often over-edit the subject or lose detail in non-target areas.

Please help me restore this old photo and improve its clarity

Input Image Qwen-Image-Edit

Flux2

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Seedream4.0

Nano-Banana Pro Ours

Repair this photo, remove blur and noise, and improve overall clarity Input Image Qwen-Image-Edit Flux2

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Seedream4.0 Nano-Banana Pro Ours

- Figure 13 | Cases of Low-Level Editing tasks, demonstrating the model’s capability in image restoration and enhancement. Our model significantly outperforms baselines in detail recovery and visual quality, effectively reconstructing high-fidelity details from heavily degraded inputs.

Input Image Qwen-Image-Edit Seedream4.0

[Figure 255]

[Figure 256]

[Figure 257]

让模特穿上图1的上衣叠穿 套装，长度及腰，搭配任意 的下身穿搭，配件与包保持 不变，并且姿势保持不变

[Figure 258]

[Figure 259]

[Figure 260]

Dress the model in the layered top set from Image 1, with a length that falls to the waist, paired with any bottoms. Keep the accessories and bag unchanged, and keep the same pose.

Nano-Banana-Pro Ours

Input Image

Qwen-Image-Edit Seedream4.0

[Figure 261]

[Figure 262]

[Figure 263]

让模特穿上图1的连衣裙， 下摆长度过臀，款式纹理与 图1保持一致，原有的配件 保持不变，模特的姿势保持 不变

[Figure 264]

[Figure 265]

Dress the model in the onepiece from Image 1, with a hem length that falls below the hips. Match the same style and texture as in Image 1. Keep the original accessories unchanged, and keep the model’s pose unchanged.

[Figure 266]

Nano-Banana-Pro Ours

- Figure 14 | Visualization of the Virtual Try-on task, where our model transfers reference garments with superior fidelity compared to competitors, accurately rendering text-specified styling attributes (e.g., fit, accessories) while preserving details that baselines often distort or overlook.

[Figure 267]

###### Figure 15 | This visualization demonstrates the power of our FireRed-Image-Edit in a variety of creative scenarios. The results clearly demonstrate the model’s ability to interpret complex instructions and generate high-quality, creative edited images.

⽤参考图的玩偶作为画⾯主⻆，⾐服上⾯印着“FireRed-Image-Edit”字样，站在童话感花园草地中，周围有精致⼩花和柔和建筑背景， 整体⻛格温暖梦幻，超清细节，商业级摄影质感。 ⼩红薯正对镜头，⾃信可爱地站⽴，⾝后是⼀块⿊板，⽤⽩⾊粉笔清晰写着：

Input Prompt

“FireRed-Image-Edit 三⼤绝活： ⽂字艺术家：中英⽂字体排版专业稳定，视觉⻛格统⼀ 时光修复师：⽼照⽚修复细节丰富，呈现⾃ 然真实 造型设计师：智能换装精准⾃然，多⻛格服饰⼀键切换”，⽂字为⽩⾊粉笔⼿写体。 画⾯光线柔和⾃然光，浅景深，背景轻 微虚化，⾊彩明亮饱满，⾼清8K，真实摄影⻛格，细节锐利，⽆噪点，⽆畸变。

[Figure 268]

[Figure 269]

[Figure 270]

Input image Ours Nano-Banana Pro

[Figure 271]

[Figure 272]

[Figure 273]

Qwen-Image-Edit Flux2 Seedream4.0

Input Prompt 在书本封⾯“Python”的下⽅，添加⼀⾏英⽂⽂字“2ndEdition”。

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Input image

Ours

Nano-Banana Pro

Qwen-Image-Edit

Flux2

Seedream4.0

Input Prompt 将图中“吸烟区”的标志牌上的⽂字“吸烟区”改为“⽆烟区”，并将吸烟标志改为禁⽌吸烟标志。

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

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Input image Ours Nano-Banana Pro Qwen-Image-Edit Flux2 Seedream4.0

Input Prompt 将海报上右下⻆的⽂字“programme”修改为“programongoing”，保持字体和⻛格⼀致。

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

|[Figure 322]|
|---|

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Input image Ours Nano-Banana Pro Qwen-Image-Edit Flux2 Seedream4.0

- Figure 16 | Text-centric showcases. Our model demonstrates superior legibility and style consistency for text editing. While competitors often struggle with blurred characters or layout distortion, our method achieves precise text rendering that preserves the original perspective and material texture.

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

- Figure 17 | Text-centric showcases.

###### 6. Conclusion

We present FireRed-Image-Edit, a diffusion transformer for instruction-based image editing, with contributions spanning data engineering, training methodology, and evaluation.

In data engineering, we curate approximately one billion multi-source image–text pairs, covering text-to-image generation, multi-image synthesis, and diverse editing tasks. Through an end-to-end pipeline integrating rigorous cleaning, fine-grained stratification, auto-labeling, and a two-stage filtering mechanism, we distill over 100 million high-quality samples evenly balanced between generation and editing. This large-scale, structured curation substantially improves semantic alignment and data reliability.

In training methodology, we develop an efficiency-oriented framework featuring a Multi-Condition Aware Bucket Sampler to reduce padding overhead and Stochastic Instruction Alignment to enhance robustness. A multi-stage training pipeline (pre-training, supervised fine-tuning, reinforcement learning) incorporates Asymmetric Gradient Optimization for DPO to stabilize optimization. Additional components—including Multi-Image DPO, Diffusion NFT for text editing, Consistency Loss, Distributed Stratified Timestep Sampling,

Logit-Normal loss, and EMA—jointly improve consistency, controllability, and convergence stability.

In evaluation, we introduce REDEdit-Bench, a benchmark spanning 15 structured editing categories with 1,673 bilingual samples, including newly designed portrait beautification and low-resolution enhancement tasks. Extensive experiments on REDEdit-Bench, ImgEdit, and GEdit demonstrate state-of-the-art performance among open-source models and competitive results against proprietary systems, validating that carefully engineered efficiency and system-level optimization can rival brute-force scaling.

###### 7. Authors

- • Core Contributors (listed alphabetically): Changhao Qiao, Chao Hui, Chen Li, Cunzheng Wang, Dejia Song, Jiale Zhang, Jing Li, Qiang Xiang, Runqi Wang, Shuang Sun, Wei Zhu, Xu Tang, Yao Hu, Yibo Chen, Yuhao Huang, Yuxuan Duan, Zhiyi Chen, Ziyuan Guo
- • Contributors(listed alphabetically): Haohua Chen, Haolu Liu, Honghao Cai, Qing Yu, Shurui Shi, Shuyang Lin, Sijie Xu, Tianshuo Yuan, Tianze Zhou, Wenxin Yu, Xiangyuan Wang, Xudong Zhou, Xuecan Wang, Yahui Wang, Yandong Guan, Yanqin Chen, Yilian Zhong, Ying Li, Yunfan Liu, Yunhao Bai, Yushun Fang, Zeming Liu, Zhangyu Lai, Zhiqiang Wu

###### References

- [1] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [3] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.
- [4] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners, 2025.
- [5] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [6] Matthijs Douze, Giorgos Tolias, Ed Pizzi, Zoë Papakipos, Lowik Chanussot, Filip Radenovic, Tomas Jenicek, Maxim Maximov, Laura Leal-Taixé, Ismail Elezi, et al. The 2021 image similarity dataset and challenge. arXiv preprint arXiv:2106.09672, 2021.
- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the International Conference on Machine Learning (ICML), 2024.
- [8] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.
- [9] Rafael C. Gonzalez and Richard E. Woods. Digital Image Processing. Pearson, 4th edition, 2018.
- [10] Google. Nano banana pro. https://storage.googleapis.com/deepmind-media/Model-Cards/G emini-3-Pro-Image-Model-Card.pdf, 2025.
- [11] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168, 2024.
- [12] Tianyao He, Runqi Wang, Yang Chen, Dejia Song, Nemo Chen, Xu Tang, and Yao Hu. Flux-sculptor: Text-driven rich-attribute portrait editing through decomposed spatial flow control. arXiv preprint arXiv:2507.03979, 2025.
- [13] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2023.
- [14] Black Forest Labs. FLUX.2: State-of-the-Art Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.
- [15] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [16] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025.
- [17] Zhangyu Lai, Yilin Lu, Xinyang Li, Jianghang Lin, Yansong Qu, Liujuan Cao, Ming Li, and Rongrong Ji. Anomalypainter: Vision-language-diffusion synergy for zero-shot realistic and diverse industrial anomaly synthesis. arXiv preprint arXiv:2503.07253, 2025.
- [18] Fangjian Liao, Xingxing Zou, and Waikeung Wong. Appearance and pose-guided human generation: A survey. ACM Computing Surveys, 56(5):1–35, 2024.

- [19] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.
- [20] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [21] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1xedit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [22] Yilin Lu, Jianghang Lin, Linhuang Xie, Kai Zhao, Yansong Qu, Shengchuan Zhang, Liujuan Cao, and Rongrong Ji. Generate aligned anomaly: Region-guided few-shot anomaly image-mask pair synthesis for industrial inspection. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 11259–11268, 2025.
- [23] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing, 21(12):4695–4708, 2012.
- [24] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.
- [25] Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, and Matthijs Douze. A self-supervised descriptor for image copy detection. Proc. CVPR, 2022.
- [26] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [28] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [29] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.
- [30] Ying Shen, Yizhe Zhang, Shuangfei Zhai, Lifu Huang, Joshua M. Susskind, and Jiatao Gu. Many-to-many image generation with auto-regressive diffusion models, 2024.
- [31] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
- [32] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.
- [33] Qwen3-VL Team. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [34] Cunzheng Wang, Ziyuan Guo, Yuxuan Duan, Huaxia Li, Nemo Chen, Xu Tang, and Yao Hu. Target-driven distillation: Consistency distillation with target timestep selection and decoupled guidance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7619–7627, 2025.
- [35] Runqi Wang, Yang Chen, Sijie Xu, Tianyao He, Wei Zhu, Dejia Song, Nemo Chen, Xu Tang, and Yao Hu. Dynamicface: High-quality and consistent face swapping for image and video using composable 3d facial priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13438–13447, 2025.
- [36] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

- [37] Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In The Thirteenth International Conference on Learning Representations, 2024.
- [38] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [39] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025.
- [40] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [41] Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, and Huan Ling. Chronoedit: Towards temporal reasoning for image editing and world simulation. arXiv preprint arXiv:2510.04290, 2025.
- [42] Bin Xia, Bohao Peng, Yuechen Zhang, Junjia Huang, Jiyang Liu, Jingyao Li, Haoru Tan, Sitong Wu, Chengyao Wang, Yitong Wang, et al. Dreamomni2: Multimodal instruction-based editing and generation. arXiv preprint arXiv:2510.06679, 2025.
- [43] Qiang Xiang, Shuang Sun, Binglei Li, Dejia Song, Huaxia Li, Nemo Chen, Xu Tang, Yao Hu, and Junping Zhang. Instanceassemble: Layout-aware image generation via instance assembling attention. arXiv preprint arXiv:2509.16691, 2025.
- [44] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025.
- [45] Sijie Xu, Runqi Wang, Wei Zhu, Dejia Song, Nemo Chen, Xu Tang, and Yao Hu. Single trajectory distillation for accelerating image and video style transfer. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9463–9471, 2025.
- [46] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4210–4220, 2023.
- [47] Keming Ye, Zhipeng Huang, Canmiao Fu, Qingyang Liu, Jiani Cai, Zheqi Lv, Chen Li, Jing Lyu, Zhou Zhao, and Shengyu Zhang. Unicedit-10m: A dataset and benchmark breaking the scale-quality barrier via unified verification for reasoning-enriched edits. arXiv preprint arXiv:2512.02790, 2025.
- [48] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.
- [49] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025.
- [50] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.
- [51] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 3836–3847, 2023.
- [52] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large-scale diffusion transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2504.20690.

- [53] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.
- [54] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.
- [55] Zhengguang Zhou, Jing Li, Huaxia Li, Nemo Chen, and Xu Tang. Storymaker: Towards holistic consistent characters in text-to-image generation. arXiv preprint arXiv:2409.12576, 2024.

###### Annotation Prompt

# Image Annotator You are a professional image annotator. Please complete the following tasks based on the input image.

- ## Step 1: Image Caption

- 1. Use natural, descriptive language without structured formats.
- 2. Include details: object attributes, spatial relationships, and environment.
- 3. Transcribe visible text exactly as shown, enclosed in quotation marks.
- 4. Be specific and accurate, avoiding vague descriptions.

- ## Step 2: Image Analysis

- 1. Image Type: Classify by source (e.g., Photograph, Screenshot, Illustration).
- 2. Visual Style: Identify artistic style (e.g., Photorealistic, Anime, Watercolor).
- 3. Watermarks: List any detected watermarks, or "None".
- 4. Anomalies: Note distracting elements like QR codes, mosaics, or artifacts.

## Output Format ‘‘‘json {

"caption": "Descriptive text with ’OCR text’ quoted.", "image_type": "Photograph", "style": "Photorealistic", "anomalies": ["QR code"] or [], "subjects": { "main": [

{

"appearance": "Age, gender, hair, build", "clothing": "Type, color, accessories", "action": "Pose or activity", "emotion": "Expression and mood"

}

], "other": []

}, "environment": {

"setting": "Background elements, weather, season", "lighting": "Light source and quality", "palette": "Dominant colors", "shot": "Camera angle and framing"

}, "quality": {

"resolution": "Sharp or blurry", "watermarks": "Yes or No"

}

} ‘‘‘

