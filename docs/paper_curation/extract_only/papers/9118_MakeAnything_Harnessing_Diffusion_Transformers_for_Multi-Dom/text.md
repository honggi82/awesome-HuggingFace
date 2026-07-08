# arXiv:2502.01572v2[cs.CV]5Feb2025

## MakeAnything: Harnessing Diffusion Transformers for Multi-Domain Procedural Sequence Generation

#### Yiren Song1 Cheng Liu1 Mike Zheng Shou1

🙂 🤖 🙂 🤖

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Carve a wooden chair toy.

Build a robot’s foot using ZBrush.

[Figure 7]

[Figure 8]

Give me a Chinese pain ng of a swan.

Give me a fabric toy of a cupcake.

🙂 🤖

[Figure 9]

[Figure 10]

[Figure 11]

Paint a girl’s portrait with golden hair.

[Figure 12]

How to make meatball spaghe ?

[Figure 13]

[Figure 14]

How to build this LEGO ship?

Figure 1: We introduce MakeAnything, a tool that realistically and logically generates step-by-step procedural tutorial for activities such as painting, crafting, and cooking, based on text descriptions or conditioned images.

### Abstract

ties and task-specific performance by freezing encoder parameters while adaptively tuning decoder layers. Additionally, our ReCraft model enables image-to-process generation through spatiotemporal consistency constraints, allowing static images to be decomposed into plausible creation sequences. Extensive experiments demonstrate that MakeAnything surpasses existing methods, setting new performance benchmarks for procedural generation tasks. Code is released at https://github.com/showlab/MakeAnything

A hallmark of human intelligence is the ability to create complex artifacts through structured multi-step processes. Generating procedural tutorials with AI is a longstanding but challenging goal, facing three key obstacles: (1) scarcity of multi-task procedural datasets, (2) maintaining logical continuity and visual consistency between steps, and (3) generalizing across multiple domains. To address these challenges, we propose a multi-domain dataset covering 21 tasks with over 24,000 procedural sequences. Building upon this foundation, we introduce MakeAnything, a framework based on the diffusion transformer (DIT), which leverages fine-tuning to activate the in-context capabilities of DIT for generating consistent procedural sequences. We introduce asymmetric low-rank adaptation (LoRA) for image generation, which balances generalization capabili-

### 1. Introduction

A defining characteristic of human intelligence—and a key differentiator from other species—is the capacity to create complex artifacts through structured step-by-step processes. In computer vision, generating such procedural sequences for tasks like painting, crafting, product design, and culinary arts remains a significant challenge. The core difficulty lies in producing multi-step sequences that maintain logical continuity and visual consistency, requiring models to both capture intricate visual features and understand causal relationships between steps. This challenge becomes par-

1Show Lab, National University of Singapore, Singapore. Correspondence to: Mike Zheng Shou <mike.zheng.shou@gmail.com>.

ticularly pronounced when handling diverse domains and styles without compromising generation quality—a problem space that remains underexplored.

Existing research primarily focuses on decomposing painting processes, with early methods employing reinforcement learning/optimization algorithms through stroke-based rendering to approximate target images. Subsequent works like ProcessPainter (Song et al., 2024a) and PaintsUndo (Team, 2024) utilize temporal models on synthetic datasets, while Inverse Painting (Chen et al., 2024)redicts the order of human painting, generating the painting process by region. However, these approaches remain limited to single-task scenarios and exhibit poor cross-domain generalization. Furthermore, ProcessPainter’s Animatediff-based framework constrains modifications to minor motion adjustments, making it unsuitable for categories requiring structural transformations (e.g., recipes or crafts). Although Diffusion Transformer (DIT) (Peebles & Xie, 2023)-based video generation models can produce long sequences, their effectiveness is hindered by distribution shifts in training data when generating complex procedural workflows.

We posit that replicating human creative intelligence requires both high-quality multi-task procedural data and advanced methodology design.. To this end, we curate a comprehensive multi-domain dataset spanning 21 categories (including painting, crafts, SVG design, LEGO assembly, and cooking) with over 24,000 procedurally annotated sequences—the largest such collection for step-by-step creation tasks. Methodologically, we propose MakeAnything, a novel framework that harnesses the in-context capabilities of Diffusion Transformers (DIT) through LoRA fine-tuning to generate high-quality instructional sequences.

Addressing the challenge of severe data scarcity (some categories have as few as 50 data entries.) and imbalanced distributions, we employ an asymmetric low-rank adaptation (LoRA)(Zhu et al., 2024; Hu et al., 2022) strategy for image generation. This approach combines a pretrained encoder on large-scale data with a task-specific fine-tuned decoder, achieving an optimal balance between generalization and domain-specific performance.

To address practical needs for reverse-engineering creation processes, we develop the ReCraft Model—an efficient controllable generation method that decomposes static images into step-by-step procedural sequences. Building upon the pretrained Flux model with minimal architectural modifications, ReCraft introduces an image-conditioning mechanism where clean latent tokens from the target image (encoded via VAE) guide the denoising of noisy intermediate frames through multi-modal attention. Remarkably, this lightweight adaptation enables efficient training with limited data—ReCraft achieves robust performance with just hundreds or even dozens of process sequences per task.

During inference, the model recursively predicts preceding frames over concatenated latent representations, effectively reconstructing the creation history from static artworks.

In summary, our contributions are as follows:

- 1. Unified Procedural Generation Framework: We introduce MakeAnything, the first DIT-based architecture enabling cross-domain procedural sequence synthesis, supporting both text-to-process and image-toprocess generation paradigms.
- 2. Technical Innovations: We employ an asymmetric LoRA architecture for cross-domain generalization and the ReCraft Model for image-conditioned process reconstruction with limited training data.
- 3. Dataset Contribution: We propose a multi-domain procedural dataset (21 categories, 24K+ sequences) with hierarchical annotations, significantly advancing research in procedural understanding and generation.

### 2. Related Work

#### 2.1. Diffusion Models

Diffusion probability models (Song et al., 2020; Ho et al., 2020) are advanced generative models that restore original data from pure Gaussian noise by learning the distribution of noisy data at various levels of noise. Their powerful capability to adapt to complex data distributions has led diffusion models to achieve remarkable success across several domains including image synthesis (Rombach et al.,

- 2022; Peebles & Xie, 2023), image editing (Brooks et al.,
- 2023; Hertz et al., 2022; Zhang et al., 2024c;d), and video gneration (Guo et al., 2023; Blattmann et al., 2023; Song et al., 2024a), evaluation (Song et al., 2024b). Stable Diffusion (Rombach et al., 2022) (SD), a notable example, utilizes a U-Net architecture and extensively trains on largescale text-image datasets to iteratively generate images with impressive text-to-image capabilities. The Diffusion Transformer (DiT) model (Peebles & Xie, 2023), employed in architectures like FLUX.1 (AI, 2024), Stable Diffusion 3 (Esser et al., 2024), and PixArt (pixart), uses a transformer as the denoising network to iteratively refine noisy image tokens. Customized generation methods enable flexible customization of concepts and styles by fine-tuning U-Net (Ruiz et al., 2023) or certain parameters (Hu et al., 2022; Kumari et al., 2023), alongside trainable tokens. Trainingfree customization methods (Ye et al., 2023; Zhang et al.,
- 2024a;b) leverage pre-trained CLIP (Radford et al., 2021) encoders to extract image features for efficient customized generation.

#### 2.2. Controllable Generation in Diffusion Models

Controllable generation has been extensively studied in the context of diffusion models. Text-to-image models (Ho

###### Legends

###### A en on Map

Task3 …

[Figure 15]

Key

[Figure 16]

[Figure 17]

[Figure 18]

🔥 ❄

[Figure 19]

Text Tokens Noised Latent Tokens Image Condi on Tokens

Learnable Frozen

Task2

Text Latent Condi on

[Figure 20]

[Figure 21]

LatentCondi onText

[Figure 22]

DiT Block

DiT Block

[Figure 23]

…

…

Query

Task1

[Figure 24]

Feed Forward

Feed Forward

Self-A en on

Self-A en on

[Figure 25]

Domain-speciﬁc Metric B ❄

… QKVProj.

…

[Figure 26]

🔥 🔥 🔥 …

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Prompt

DiT Block

DiT Block

Recra  Model

[Figure 31]

|Shared A 🔥<br><br>| |
|---|---|
| | |

[Figure 32]

🔥

QKV Proj.❄

[Figure 33]

Text Encoder

VAE

VAE

[Figure 34]

[Figure 35]

| | |
|---|---|
| | |

###### (a) Asymmetric LoRA Training (b) ReCra  Model Training

Figure 2: The MakeAnything framework comprises two core components: (1) an Asymmetric LoRA module that generates diverse creation processes from text prompts through asymmetric LoRA, and (2) the ReCraft Model, which constructs an image-conditioned base model by merging pretrained LoRA weights with the Flux foundation model, enabling process prediction via injected visual tokens.

et al., 2020; Song et al., 2020) have established a foundation for conditional generation, while various approaches have been developed to incorporate additional control signals such as images. Notable methods include ControlNet (Zhang & Agrawala, 2023), enabling spatially aligned control in diffusion models, and T2I-Adapter (Mou et al., 2023), which improves efficiency with lightweight adapters. UniControl (Zhao et al., 2023) uses Mixture-of-Experts (MoE) to unify different spatial conditions, further reducing model size. However, these methods rely on spatially adding condition features to the denoising network’s hidden states, inherently limiting their effectiveness for spatially non-aligned tasks like subject-driven generation. IPAdapter (Ye et al., 2023) addresses this by introducing crossattention through an additional encoder. Based on the DiT architecture, OminiControl (Tan et al., 2024) proposes a unified solution that is applicable to both spatially aligned and non-aligned tasks by concatenating condition tokens with noise tokens.

niques like Stylized Neural Painting (Kotovenko et al., 2021) have advanced stroke optimization, which can be integrated with neural style transfer. The field of vector graphic generation employs similar techniques (Frans et al., 2022; Song et al., 2023; Song, 2022; Song & Zhang, 2022). However, these methods differ greatly from human creative processes due to variations in artists’ styles and subjects. Inverse Painting (Chen et al., 2024) achieves realistic painting process simulation by predicting the painting order and implementing image segmentation. ProcessPainter (Song et al., 2024a) and Paints Undo (Team, 2024) method fine-tunes diffusion models using data from artists’ painting processes to learn their true distributions, enabling the generation of painting processes in multiple styles.

### 3. Method

In this section, we begin by exploring the preliminaries on diffusion transformer as detailed in Section 3.1. Next, Section 3.2 introduce the overall architecture of the MakeAnything method. In Section 3.3, we present asymmetric LoRA for Procedual learning. Section 3.4 introduces the ReCraft Model, an effective image condition model that generates procedural sequences highly consistent with reference images. Finally, we introduce the new dataset we proposed in Section 3.5.

#### 2.3. Procedural Sequences Generation

Generating the creation process of paintings or handicrafts is something that has always been desired but is difficult to achieve. The problem of teaching machines ”how to paint” has been thoroughly explored within stroke-based rendering (SBR), focusing on recreating non-photorealistic imagery through strategic placement and selection of elements like paint strokes (Hertzmann, 2003). Early SBR methods included greedy searches or required user input (Haeberli, 1990; Litwinowicz, 1997), while recent advancements have utilized RNNs and RL to sequentially generate strokes (Ha & Eck, 2017; Zhou et al., 2018; Xie et al., 2013). Adversarial training has also been introduced as an effective way to produce non-deterministic sequences (Nakano, 2019). Tech-

#### 3.1. Preliminary

The Diffusion Transformer (DiT) model, uses a transformer as the denoising network to iteratively refine noisy image tokens. A DiT model processes two types of tokens: noisy image tokens z ∈ RN×d and text condition tokens cT ∈ RM×d, where d is the embedding dimension, and N and M are the number of image and text tokens. Throughout

the network, these tokens maintain consistent shapes as they pass through multiple transformer blocks.

In FLUX.1, each DiT block consists of layer normalization followed by Multi-Modal Attention (MMA) (Pan et al., 2020), which incorporates Rotary Position Embedding (RoPE) (Su et al., 2024) to encode spatial information. For image tokens z, RoPE applies rotation matrices based on the token’s position (i,j) in the 2D grid:

zi,j → z = zi,j · R(i,j), (1)

where R(i,j) is the rotation matrix at position (i,j). Text tokens cT undergo the same transformation with their positions set to (0,0).

The multi-modal attention mechanism then projects the position-encoded tokens into query Q, key K, and value V representations. It enables the computation of attention between all tokens:

MMA([z;cT]) = softmax

QK⊤ √

d

V, (2)

where [z;cT] denotes the concatenation of image and text tokens. This formulation enables bidirectional attention.

#### 3.2. Overall Architecture

As shown in Fig. 2, the training of MakeAnything is divided into two stages: First, we train on the MakeAnything dataset using the asymmetric LoRA method, enabling the generation of creative tutorials from text descriptions. Then, the LoRA from this first phase is merged with the Flux base model to form the base model for training the ReCraft Model. In the second stage, image condition tokens are concatenated with noised latent tokens, introducing an image-conditioned mechanism into the denoising process. This setup is further fine-tuned using LoRA to complete the training of the ReCraft Model.

#### 3.3. Asymmetric LoRA for Procedural Learning

Serpentine Sequence Layout. The core of MakeAnything involves arranging different frames of a sequence into a grid and using the in-context capabilities and attention mechanism of DiT to achieve consistent Sequence generation. Tokens within the DiT’s attention mechanism tend to focus on spatially adjacent tokens, a tendency that stems from the strong correlations between neighboring image pixels captured during the pre-training of the diffusion model (Wan et al., 2024). To enhance the model’s learning effectiveness for grid sequences, we propose the Serpentine dataset construction method. As shown in Fig. 3, we arrange sequences of 9 frames and 4 frames in a serpentine pattern to ensure that temporally adjacent frames are also spatially adjacent (either horizontally or vertically adjacent).

Asymmetric LoRA. Another challenge is that training a single LoRA on all data leads to difficulties in learning diverse knowledge, while training LoRA on a single type of sequence data results in overfitting due to the limited quantity of process data for each category. Inspired by HydraLoRA (Zhu et al., 2024), we introduce an asymmetric LoRA design for the first time in image generation. This design combines shared knowledge and specialized functionalities by jointly training a shared central matrix A and multiple task-specific matrices B, significantly improving multi-task performance.

Each layer of LoRA consists of an A matrix and a B matrix, where A captures general knowledge, and B adapts to specific tasks. The asymmetric LoRA architecture can be formulated as:

W = W0 + ∆W = W0 +

N

ωi · BiA, (3)

i=1

where Bi ∈ Rd×r and the shared matrix A ∈ Rr×k. This structure effectively balances generalization and taskspecific adaptation, enhancing the model’s performance across diverse tasks.

Inference stage, the domain-specific matrix B and the domain-agnostic matrix A are used in combination, balancing generalization capabilities with performance on specific tasks. Our method can also be combined with the stylized LoRA from the Civitai website (which is not trained on procedural sequences), to enhance performance in unseen domains.

Conditional Flow Matching Loss. The conditional flow matching loss function is following SD3 (Esser et al., 2024), which is defined as follows:

t(z|ϵ),p(ϵ) ∥vΘ(z,t,cT) − ut(z|ϵ)∥2 (4)

LCFM = Et,p

Where vΘ(z,t,cT) represents the velocity field parameterized by the neural network’s weights, t is timestep, cI and cT are image condition tokens extracted from source image Isrc and text tokens. ut(z|ϵ) is the conditional vector field generated by the model to map the probabilistic path between the noise and true data distributions, and E denotes the expectation, involving integration or summation over time t, conditional z, and noise ϵ.

#### 3.4. ReCraft Model

In practical applications, users not only want to generate creation processes from text but also wish to upload an image and predict the creation process of the existing artwork or handicraft in the picture. For this, we implemented the ReCraft model, which allows users to upload images and generates a sequence of steps highly consistent with the uploaded image.

[Figure 36]

1⃣ 2⃣ 3⃣

[Figure 37]

[Figure 38]

[Figure 39]

- 6⃣ 5⃣ 4⃣

[Figure 40]

[Figure 41]

[Figure 42]

- 7⃣ 8⃣ 9⃣

[Figure 43]

[Figure 44]

[Figure 45]

9 frame layout

Pain ng Sand Art

Sketch Illustra on

Portrait Icon landscape illustra on

LEGO

Transformer

Cook

Clay toys

Pencil sketch Ink pain ng Fabric toys

1⃣ 2⃣

[Figure 46]

[Figure 47]

4⃣ 3⃣

[Figure 48]

[Figure 49]

4 frame layout

Oil pain ng Wood Sculpture Clay Sculpture Zbrush Modeling Jade Carving Emoji

Line draw

- Figure 3: Examples from the MakeAnything Dataset, which consists of 21 tasks with over 24,000 procedural sequences.

A major challenge in training the ReCraft model is the limited number of datasets available for each task, which is insufficient to train a controllable plugin like Controlnet or IP-Adapter from scratch. To address this, we innovatively designed the ReCraft model by reusing the pretrained Flux model and making minimal modifications to extend it into an image-conditioned generation model. Specifically, during training, we input the final frame into a VAE to obtain latent image condition tokens, which are then concatenated with noised latent tokens, using the attention mechanism to provide conditional information for denoising other frames. Notably, the noise addition and removal process are only performed on other frames, while the image condition tokens are clean. During inference, we predict the previous eight frames from the tail frame, revealing step-by-step how the object in the reference image was formed.

based on the final frame. This predicts how the object in the reference image was created step by step.

#### 3.5. MakeAnything Dataset

As shown in Fig. 3, we have collected a multi-task dataset that encompasses tutorials for 21 tasks. We assembled a professional data collection and annotation team that gathered and processed various tutorials from the internet and also collaborated with artists to customize high-quality painting process data. The datasets vary in size from 50 to 10,000 entries, totaling over 24,000. The first ten tasks have data in 9 frames, while the rest have 4 frames, arranged into 3x3 and 2x2 grids for training purposes. We used GPT-4o to label all datasets and describe each frame.

### 4. Experiment 4.1. Experimental Setting

In ReCraft model, multi-modal attention mechanisms are used to provide conditional information for the denoising of other frames.

Setup. We implemented MakeAnything based on the pretrained Flux 1.0 dev. We replaced the Adam optimizer with the CAME optimizer, and experiments showed that this setup achieved better generation quality. During the training phases of Asymmetric LoRA and the ReCraft model, the resolution was set to 1024, LoRA rank was 64, learning rate was 1e-4, and batch size was 2. Asymmetric LoRA and the ReCraft model were trained for 40,000 steps and 15,000 steps, respectively.

QK⊤ √

MMA([z;cI;cT]) = softmax

V, (5)

d

where [z;cI;cT] denotes the concatenation of image and text tokens. This formulation enables bidirectional attention.

The conditional flow matching loss with image condition can be defined as follows:

t(z|ϵ),p(ϵ) ∥vΘ(z,t,cI,cT) − ut(z|ϵ)∥2

LCFM = Et,p

Baselines. In the Text-to-Sequence task, we compare our approach with state-of-the-art baseline methods, namely ProcessPainter (Song et al., 2024a), Flux 1.0 (AI, 2024), and the commercial API Ideogram (Ideogram, 2023). We categorize the test prompts into two types: painting and

(6) Where vΘ(z,t,cI,cT) represents the velocity field parameterized by the neural network’s weights.

During inference, Recraft Model predict previous 8 frames

[Figure 50]

Step-by-step clay making process, <image-1> Roll a pink clay base for the cupcake. <image-2> Shape the cupcake liner with ridges.…

Chinese painting process, <image-1> outlines the basic form of a cherry blossom branch. <image-2> …

+ Ink pain ng LoRA

+ Clay toys LoRA

Pottery making process, <image-1> Kneading and conditioning the clay for hand-building. <image-2> Forming the clay into a basic teapot shape…

Sketch painting process,

- <image-1> Sketching the basic form of a historical figure.
- <image-2> …

+ Sketch LoRA

+ Clay LoRA

Step-by-step wool fabric making process, <image-1> A grey crocheted owl base is formed. <image-2>…

Oil painting process, <image-1> a basic sketch of a vibrant autumn forest with colorful leaves outlined. <image-2> …

+ Oil pain ng LoRA

+ Illustra on LoRA

Step-by-step carving process, <image-1> initial outline of an eagle soaring, sketched on the material . <image-2>…

Landscape create process, <image-1> a pale violet dusk sky, then subtle silhouettes of leafless trees. <image-2> …

+ Fabric toys LoRA

+ Landscape Ill. LoRA

Sand art creation process, <image-1> The outline of a cozy cottage begins to form, capturing the basic shape of the house and chimney…

+ Sand Art LoRA

Transformers from vehicles to robots, <image-1> A red convertible sports car <image2>The open-top roof folds inward…

+ Transformer LoRA

- (a) Text-to-Sequence genera on results
- (b) Image-to-Sequence genera on results

[Figure 51]

Input image

Wood carving process of pineapple, <image-1> a basic sketch of the pineapple on wood…

Wood carving process of a cat toy, <image-1> a basic sketch of a cat toy on wood…

Input image

Step-by-step ketch process, a cartoon character drawing, <image-1>basic geometric shapes, outlining the character's head, <image-2> …

Input image

LEGO Brickheadz of Santa Claus, step-by-step tutorial, <image-1> the base of the character being formed,<image-2> …

Input image

[Figure 52]

Ice Sculpture LoRA

Relief sculpture LoRA

+ Pain ng LoRA

+ Wood Sculpture LoRA

Ice sculpture process of a house, <image-1> a basic shape of a house made of ice …

Relief sculpture process of a flower, <image-1> a basic shape of flower on white wall …

Watercolor LoRA

Paper Quilling LoRA

+ Pain ng LoRA

+ Pain ng LoRA

Painting process of a house in watercolor style, <image-1> a basic shape of a house…

Paper quilling process of flower, <image-1> a basic sketch of flower made of paper…

(c) Generaliza on results on the unseen domain tasks

- Figure 4: Generation results of MakeAnything. From top: Text-to-Sequence outputs conditioned on textual prompts; Image-to-Sequence reconstructions via ReCraft Model; Unseen Domain generalization combining procedural LoRA (blue) with stylistic LoRA (red).

###### (1) Image-to-Tutorial methods (2) Text-to-Tutorial methods (others) (3) Text-to-Tutorial methods (pain ng)

[Figure 53]

OursProcessPainterIdeogramFlux

InversionPain ngPaintsUndo

FluxIdeogram

Input image

Ours

Ours

Painting process of a locomotive… Wool fabric making process, a white crocheted polar bear… Painting process, ink painting, lotus…

Figure 5: Compare with baselines on different tasks.

#### 4.2. Experimental Results

others, because some baselines only support painting. In the Image-to-Sequence task, our baselines are Inverse Painting (Chen et al., 2024) and PaintsUndo (Team, 2024), which are capable of predicting the creation process of a painting.

- Fig. 4(a) showcases the results of generating process sequences from textual descriptions. Benefiting from highquality datasets, a robust pre-trained model, and an innovative method design, MakeAnything consistently produces high-quality and logically coherent process sequences. Table 1 presents the quantitative evaluation results of MakeAnything across 21 tasks, including scores from GPT and human assessments, with 20 sequences generated per task, with 20 sequences generated per task.
- Fig. 4(b) highlights the model’s ability to generate process sequences conditioned on input images. The results indicate a high degree of alignment between the generated sequences and the original image content. This showcases the model’s capacity to interpret complex visual inputs and reconstruct logically consistent creation processes, enabling its application in diverse fields such as reverse engineering and educational tutorials.
- Fig. 4(c) shows the results of MakeAnything in unseen domains. We collected various LoRAs from the Civitai (Civitai, 2025) website, including watercolor, relief, ice sculpture, and paper quilling art, and combined them with our procedural LoRA. It is evident that MakeAnything demonstrates quite impressive generalization capabilities, despite not having been trained on these creative processes.

Evaluation Metrics. A good procedural sequence needs to be coherent, logical, and useful; however, evaluating procedural sequence generation and its rationality lacks precedents. We employ the CLIP Score to assess the text-image alignment of the generated results. Additionally, we evaluate the coherence and usability of the generated results using GPT-4o and human evaluations. Specifically, we meticulously design the input prompts for GPT-4o and scoring rules to align with human preferences. In the comparative experiments, we concatenate the results from various baselines with our results, input them into GPT-4o all at once, and have it select the best results across different evaluation dimensions.

Table 1: Combined Evaluation of Procedural Sequence Generation Results Across Different Tasks. Abbreviations: G = GPT score, H = Human evaluation, C = CLIP score.

Task Alignment (G | H | C) Coherence (G | H) Usability (G | H)

Painting 4.50 | 4.27 | 34.24 4.80 | 3.98 4.60 | 4.13 Sketch 4.10 | 3.97 | 29.35 4.70 | 4.11 4.10 | 4.13 Sand Art 4.20 | 4.30 | 31.82 4.70 | 4.12 4.30 | 4.18 Portrait 4.25 | 4.28 | 33.84 5.00 | 4.28 4.05 | 4.33 Icon 3.45 | 4.33 | 31.46 3.50 | 4.17 3.15 | 4.25 Landscape Ill. 4.55 | 4.28 | 32.25 4.85 | 3.95 4.50 | 4.12 Illustration 3.12 | 4.17 | 31.68 3.40 | 4.07 2.45 | 4.07 LEGO 4.60 | 4.32 | 34.40 4.90 | 4.15 4.75 | 4.00 Transformer 4.75 | 4.30 | 33.03 4.90 | 4.23 4.75 | 4.15 Cook 3.20 | 4.21 | 34.41 4.25 | 4.03 3.65 | 3.90 Clay Toys 4.30 | 4.17 | 35.25 4.50 | 4.30 4.20 | 4.30 Pencil Sketch 3.85 | 4.33 | 34.44 4.50 | 4.20 3.80 | 4.25 Chinese Painting 4.80 | 4.37 | 33.46 4.90 | 4.22 4.70 | 4.33 Fabric Toys 4.35 | 4.30 | 32.83 4.60 | 4.08 4.40 | 4.30 Oil Painting 4.90 | 4.30 | 37.30 4.95 | 4.17 4.85 | 4.20 Wood Sculpture 4.65 | 4.32 | 33.83 4.85 | 4.23 4.65 | 4.08 Clay Sculpture 4.30 | 4.17 | 35.25 4.50 | 4.30 4.20 | 4.30 Brush Modeling 4.20 | 4.33 | 32.27 4.15 | 4.03 4.05 | 4.25 Jade Carving 4.90 | 4.28 | 32.93 4.85 | 4.12 4.75 | 4.00 Line Draw 4.10 | 4.20 | 30.76 4.20 | 3.97 3.90 | 4.08 Emoji 3.75 | 4.25 | 34.20 3.60 | 4.17 3.80 | 4.18

#### 4.3. Comparation and Evaluation

This section consolidates the comparative evaluations of our method against baseline approaches on 50 sequence groups. Fig. 5(a) and (b), demonstrate that MakeAnything produces higher quality procedural sequence with superior logic and coherence, unlike the baseline methods which struggle with consistency. Fig. 5(c) compares the ReCraft model to a baseline, highlighting our method’s training on diverse real data, resulting in varied and authentic creative processes. Quantitative results in Fig. 7 confirm MakeAnything’s superiority in Text-Image Alignment, Coherence, and Usability across all tested metrics.

[Figure 54]

Base model

w/o Asymmetric LoRA

Full

Painting of a goldfish, beginning with a simple color blocking… Sketch, Drawing a character on a motorcycle, starting from basic shapes and lines…

##### Figure 6: Ablation study results.

Text-to-Pain ng Sequence (GPT)

Text-to-Pain ng Sequence (Human)

method achieves both procedural rationality and text-image alignment by leveraging knowledge from large-scale pretraining. Quantitative results across more tasks (Table 2) further validate these findings.

0.6

0.7

0.525

0.45

0.35

0.3

0.175

0.15

0

0

Alignment Coherence Usability

Alignment Coherence Usability

Ours Ideogram Flux ProcessPainter

Table 2: Ablation Study Results Using GPT Evaluation and CLIP Score.

Text-to-Other Sequence (Human)

Text-to-Other Sequence (GPT)

0.6

0.8

0.45

0.6

0.3

0.4

0.15

0.2

Model Task Alignment(G | C) Coherence Usability

0

0

Alignment Coherence Usability

Alignment Coherence Usability

Portrait 3.75| 29.78 3.45 3.35

Wood 3.25| 35.29 2.95 2.65 Fabric toys 3.55| 32.95 4.00 3.85

Base Model

Ours Ideogram Flux

Image-to-Pain ng Sequence (GPT)

Image-to-Pain ng Sequence (Human)

Portrait 4.25| 31.08 4.50 4.15 Wood Sculpture 3.55| 31.05 4.35 3.75 Fabric toys 3.75| 30.72 3.15 3.20

0.8

0.8

w/o Asymmetric LoRA

0.6

0.6

0.4

0.4

Portrait 4.55| 32.95 4.75 4.25 Wood Sculpture 4.25| 33.89 3.80 4.05 Fabric toys 4.40| 32.01 4.25 4.35

0.2

0.2

Full

0

0

Consistency Coherence Usability

Consistency Coherence Usability

Ours Inverse Paints PaintsUndo

- Figure 7: Comparison results on three tasks, evaluated by GPT and humans respectively.

#### 4.4. User Study

To comprehensively evaluate MakeAnything’s effectiveness, we conducted a user study comparing our method against baselines. Participants rated sequences across four metrics: Alignment (text-image similarity), Coherence (logical step progression), Usability (practical value), and Consistency (consistency between image condition). As shown in Fig. 7, MakeAnything demonstrates comprehensive superiority across all metrics.

#### 4.5. Ablation Study

In this section, we conducted ablation experiments on asymmetric LoRA, and Fig. 6 compares the results of portrait and sketch tutorial generation task. The former was trained on 50 portrait painting sequences, while the latter was trained on 300 cartoon character sketch sequences. While the base model produces coherent text but fails in step-by-step synthesis, standard LoRA exhibits severe overfitting on small datasets with imbalanced class distributions—yielding plausible steps but compromised text-image alignment. Our

### 5. Limitations and Future Work

The current grid-based composition strategy in MakeAnything introduces two inherent limitations: constrained output resolution (max 1024×1024) and fixed frame count (up to 9 steps). We plan to address these limitations in future work, enabling arbitrary-length sequence generation with high-fidelity outputs.

### 6. Conclusion

We introduced MakeAnything, a novel framework for generating high-quality process sequences using the DiT model with LoRA fine-tuning. By leveraging multi-domain procedural dataset and adopting an asymmetric LoRA design, our approach effectively balances generalization and taskspecific performance. Additionally, the image-conditioned plugin enables controllable and interpretable sequence generation. Extensive experiments demonstrated the superiority of our method across diverse tasks, establishing a new benchmark in this field. Our contributions pave the way for further exploration of step-by-step process generation, opening up exciting possibilities in computer vision and related applications.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Hertzmann, A. A survey of stroke-based rendering. Institute of Electrical and Electronics Engineers, 2003.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

### References

AI, F. Flux.1 ai, 2024. URL https://flux1ai.com/. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D.,

Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Chen, B., Wang, Y., Curless, B., Kemelmacher-Shlizerman, I., and Seitz, S. M. Inverse painting: Reconstructing the painting process. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

Civitai. Civitai website. http://www.civitai.com,

2025. Accessed: 2025-01-29.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Frans, K., Soros, L., and Witkowski, O. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. Advances in Neural Information Processing Systems, 35:5207–5218, 2022.

Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., and Dai, B. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

Ha, D. and Eck, D. A neural representation of sketch drawings. arXiv preprint arXiv:1704.03477, 2017.

Haeberli, P. Paint by numbers: Abstract image representations. In Proceedings of the 17th annual conference on Computer graphics and interactive techniques, pp. 207–214, 1990.

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-Or, D. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=nZeVKeeFYf9.

Ideogram. Ideogram ai, 2023. URL https:// ideogram.ai. Accessed: 2025-02-04.

Kotovenko, D., Wright, M., Heimbrecht, A., and Ommer, B. Rethinking style transfer: From pixels to parameterized brushstrokes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12196–12205, 2021.

Kumari, N., Zhang, B., Zhang, R., Shechtman, E., and Zhu, J.-Y. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1931– 1941, 2023.

Litwinowicz, P. Processing images and video for an impressionist effect. In Proceedings of the 24th annual conference on Computer graphics and interactive techniques, pp. 407–414, 1997.

Mou, C., Wang, X., Xie, L., Zhang, J., Qi, Z., Shan, Y., and Qie, X. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.

Nakano, R. Neural painters: A learned differentiable constraint for generating brushstroke paintings. arXiv preprint arXiv:1904.08410, 2019.

Pan, Z., Luo, Z., Yang, J., and Li, H. Multi-modal attention for speech emotion recognition. arXiv preprint arXiv:2009.04107, 2020.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine

Learning Research, pp. 8748–8763. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/radford21a.html.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510, 2023.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Song, Y. Cliptexture: Text-driven texture synthesis. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 5468–5476, 2022.

Song, Y. and Zhang, Y. Clipfont: Text guided vector wordart generation. In BMVC, pp. 543, 2022.

Song, Y., Shao, X., Chen, K., Zhang, W., Jing, Z., and Li, M. Clipvg: Text-guided image manipulation using differentiable vector graphics. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 2312–2320, 2023.

Song, Y., Huang, S., Yao, C., Ye, X., Ci, H., Liu, J., Zhang, Y., and Shou, M. Z. Processpainter: Learn painting process from sequence data. arXiv preprint arXiv:2406.06062, 2024a.

Song, Y., Liu, X., and Shou, M. Z. Diffsim: Taming diffusion models for evaluating visual similarity. arXiv preprint arXiv:2412.14580, 2024b.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Tan, Z., Liu, S., Yang, X., Xue, Q., and Wang, X. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024.

Team, P.-U. Paints-undo github page, 2024. Wan, C., Luo, X., Cai, Z., Song, Y., Zhao, Y., Bai, Y., He,

Y., and Gong, Y. Grid: Visual layout generation. arXiv preprint arXiv:2412.10718, 2024.

Xie, N., Hachiya, H., and Sugiyama, M. Artist agent: A reinforcement learning approach to automatic stroke generation in oriental ink painting. IEICE TRANSACTIONS on Information and Systems, 96(5):1134–1144, 2013.

Ye, H., Zhang, J., Liu, S., Han, X., and Yang, W. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Zhang, L. and Agrawala, M. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.

Zhang, Y., Song, Y., Liu, J., Wang, R., Yu, J., Tang, H., Li, H., Tang, X., Hu, Y., Pan, H., et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8069–8078, 2024a.

Zhang, Y., Song, Y., Yu, J., Pan, H., and Jing, Z. Fast personalized text to image synthesis with attention injection. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6195–6199. IEEE, 2024b.

Zhang, Y., Wei, L., Zhang, Q., Song, Y., Liu, J., Li, H., Tang, X., Hu, Y., and Zhao, H. Stable-makeup: When real-world makeup transfer meets diffusion model. arXiv preprint arXiv:2403.07764, 2024c.

Zhang, Y., Zhang, Q., Song, Y., and Liu, J. Stable-hair: Realworld hair transfer via diffusion model. arXiv preprint arXiv:2407.14078, 2024d.

Zhao, S., Chen, D., Chen, Y.-C., Bao, J., Hao, S., Yuan, L., and Wong, K.-Y. K. Uni-controlnet: All-in-one control to text-to-image diffusion models. arXiv preprint arXiv:2305.16322, 2023.

Zhou, T., Fang, C., Wang, Z., Yang, J., Kim, B., Chen, Z., Brandt, J., and Terzopoulos, D. Learning to sketch with deep q networks and demonstrated strokes. arXiv preprint arXiv:1810.05977, 2018.

Zhu, J., Greenewald, K., Nadjahi, K., Borde, H. S. d. O., Gabrielsson, R. B., Choshen, L., Ghassemi, M., Yurochkin, M., and Solomon, J. Asymmetry in lowrank adapters of foundation models. arXiv preprint arXiv:2402.16842, 2024.

### A. Implementation details of the GPT4-o evaluation.

In the GPT-4-o evaluation process, we tailor distinct evaluation metrics for different tasks, ensuring both direct scoring and selective ranking are covered to suit the task’s nature.

- A.1. Direct Scoring Evaluation (for Procedural Sequence Generation and Ablation Studies) The assistant evaluates a sequence of images depicting a procedural process with criteria such as:

- • Accuracy: Measures content alignment with the provided prompt, scored from 1 (not accurate) to 5 (completely accurate).
- • Coherence: Assesses logical flow from 1 (disjointed) to 5 (seamless progression).
- • Usability: Rates helpfulness for understanding the procedure from 1 (not helpful) to 5 (highly helpful).

Scores are output in JSON format, for example: {

"Accuracy": 4, "Coherence": 5, "Usability": 4

}

- A.2. Selective Ranking Evaluation (for User Study Comparisons) This evaluation compares multiple images from different models, ranking them by:

- • Accuracy: Which image best represents the prompt?
- • Coherence: Which image shows the clearest, most logical process?
- • Usability: Which image offers the most helpful visual guidance?

Rankings are provided from 1 (best) to 4 and outputted in JSON format, e.g., {

"Accuracy": 1, "Coherence": 2, "Usability": 3

}

Example of Task Prompt and Evaluation: Prompt: ”This image shows the process of creating a handmade sculpture.” Images: [Upload images of models 1, 2, 3, and 4] Evaluation: The assistant ranks the models for Accuracy, Coherence, and Usability in JSON format. This evaluation merges qualitative and quantitative assessments to determine the effectiveness of the images generated by GPT-4-o models.

### B. More results

Fig 8-11 show more generation results of MakeAnything. Table 3-6 display the raw data from GPT evaluations and human assessments.

[Figure 55]

##### Figure 8: More generation results. From top to bottom, they are portrait, Sand Art, landscape illustration, painting, LEGO, transformer, and cook respectively.

[Figure 56]

##### Figure 9: More generation results. From top to bottom, they are oil painting and line draw.

[Figure 57]

##### Figure 10: More generation results. From top to bottom, they are ink painting and clay sculpture.

[Figure 58]

##### Figure 11: More generation results. From top to bottom, they are wood sculpure, Zbrush, and fabric toys.

- Table 3: Compare with Text-to-Sequence methods (GPT)

Category Methods Alignment Coherence Usability

Painting

Processpainter 0.24 0.26 0.22

Ideogram 0.32 0.14 0.26 Flux 0.02 0.04 0.00 Ours 0.42 0.56 0.52

Others

Ideogram 0.36 0.30 0.32 Flux 0.28 0.28 0.30 Ours 0.36 0.42 0.38

- Table 4: Compare with Image-to-Sequence methods (GPT)

Category Methods Consistency Coherence Usability

Painting

Inverse Paints 0.02 0.00 0.02 PaintsUndo 0.18 0.30 0.24 Ours 0.80 0.70 0.74

- Table 5: Compare with Text-to-Sequence methods (Human)

Category Methods Alignment Coherence Usability

Painting

Processpainter 0.06 0.10 0.14

Ideogram 0.06 0.06 0.10 Flux 0.21 0.15 0.13 Ours 0.67 0.69 0.63

Others

Ideogram 0.19 0.19 0.17 Flux 0.11 0.13 0.12 Ours 0.70 0.68 0.71

- Table 6: Compare with Image-to-Sequence methods (Human)

Category Methods Consistency Coherence Usability

Inverse Paints 0.27 0.31 0.33 PaintsUndo 0.18 0.08 0.06 Ours 0.55 0.61 0.61

Painting

