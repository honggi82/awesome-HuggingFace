## IC-Effect: Precise and Efficient Video Effects Editing via In-Context Learning

Yuanhang Li1, Yiren Song2, Junzhe Bai1, Xinran Liang1, Hu Yang3, Libiao Jin1, Qi Mao1 1 School of Information and Communication Engineering, Communication University of China 2 Show Lab, National University of Singapore 3 Baidu Inc., Beijing, China https://cuc-mipg.github.io/IC-Effect/

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

Source Video

Source Video

# arXiv:2512.15635v1[cs.CV]17Dec2025

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

Target Video

Target Video

Target Video

Add a flame-colored lightning effect to the edge of the stone cross.

Add a growth and bounce effect to the building.

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

Target Video

Target Video

Add a blue flame burning effect to the stone cross.

Add a dynamic line shuttle effect to the building.

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

Source Video

Source Video

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

Target Video

Target Video

Add an anime clone effect to the right of the woman wearing yellow.

Add a "Fa Xiang Tian Di" effect to the upper right of the woman.

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

Target Video

Target Video

Add a particle spread effect to the woman wearing yellow.

Add a particle aggregation effect for woman.

Figure 1. Video VFX Editing Results of IC-Effect. Our IC-Effect enables precise video VFX editing aligned with textual instructions while preserving the complete spatiotemporal information of the source video. The complete video is available in our supplementary materials.

### Abstract

paired VFX editing dataset spanning 15 high-quality visual styles. Extensive experiments show that IC-Effect delivers high-quality, controllable, and temporally consistent VFX editing, opening new possibilities for video creation.

We propose IC-Effect, an instruction-guided, DiT-based framework for few-shot video VFX editing that synthesizes complex effects (e.g., flames, particles and cartoon characters) while strictly preserving spatial and temporal consistency. Video VFX editing is highly challenging because injected effects must blend seamlessly with the background, the background must remain entirely unchanged, and effect patterns must be learned efficiently from limited paired data. However, existing video editing models fail to satisfy these requirements. IC-Effect leverages the source video as clean contextual conditions, exploiting the contextual learning capability of DiT models to achieve precise background preservation and natural effect injection. A two-stage training strategy, consisting of general editing adaptation followed by effect-specific learning via EffectLoRA, ensures strong instruction following and robust effect modeling. To further improve efficiency, we introduce spatiotemporal sparse tokenization, enabling high fidelity with substantially reduced computation. We also release a

### 1. Introduction

Visual Effects (VFX) aim to create videos or edit existing ones by incorporating visually compelling elements such as flames, cartoon characters, or particle effects. As a core technology in filmmaking, gaming, and virtual reality, VFX enrich visual storytelling, highlighting key elements, and creating immersive experiences. However, traditional video VFX workflows rely on complex animation design, computer-generated imagery (CGI), and professional postproduction compositing. These processes incur high production costs, long turnaround times, and extensive manual intervention, which hinder personalized or real-time applications. Recent advances in text-to-video (T2V) generation [27, 52, 57] open new possibilities for automated VFX creation [33, 38]. However, video VFX editing, which automatically adds or modifies effects in an existing video,

Corresponding Author: qimao@cuc.edu.cn

remains largely unexplored.

As a unique and higher-level video editing task, video VFX editing fundamentally differs from video VFX generation [33, 38]. Its core objective is to seamlessly and realistically integrate visual effects into a source video while strictly preserving the spatial structure and temporal coherence of the original content. Although recent video editing models [5, 12, 14, 25, 50, 56] achieve significant progress across various editing tasks, they still struggle to meet the stringent requirements of video VFX editing. Existing methods [12, 25, 50, 56] support global style changes or local content modification, but they often tolerate certain degrees of background or appearance changes, making it difficult to ensure pixel-level consistency with the source video. This limitation is unacceptable in video VFX editing, where the background must remain entirely unchanged. Mask-based editing methods [5, 14] can preserve the unmasked regions, but they rely on pixel-accurate masks, which fundamentally conflicts with the goal of automated video VFX editing. Furthermore, unlike general video editing approaches that improve performance by leveraging large-scale data, producing high-quality paired VFX data is challenging, limiting the scalability of model training. Effective video VFX editing must learn the unique patterns of effect injection from these high-quality paired samples to achieve physically consistent integration of effects into real scenes. These challenges make automated video VFX editing a largely unresolved problem.

To address above challenges, we propose IC-Effect, an instruction-guided, few-shot video VFX editing framework based on DiT-based T2V models. IE-Effect leverages the contextual learning capability [10, 21, 23, 62] of the DiT [40] model, treating the source video as a clean contextual condition to provide the model with distortion-free information. This enables the model to inject visual effects naturally while strictly preserving the original background content. The framework adopts a two-stage training strategy: a pretrained DiT-based T2V model is first adapted into a universal video editor, then applying Effect-LoRA to extract effect-specific patterns a small set of paired VFX data for style customization while preserving the base model. To improve efficiency, we introduce spatiotemporal sparse tokenization with position correction to preserve key features, reduce computation, and maintain accurate alignment between conditional tokens and generated frames for precise video VFX editing.

To mitigate data scarcity, we construct a dedicated dataset for video VFX editing that includes 15 representative effect types—such as flames, anime clones, light particles, and bouncing. Each sample contains a triplet annotation: the source video, the edited video with the target effect, and the corresponding textual description with spatiotemporal annotations. All samples are carefully aligned

in viewpoint, content, and motion to ensure reliable supervision for both training and evaluation. To our knowledge, this is the first large-scale paired benchmark specifically designed for video VFX editing, filling a crucial gap in data resources and providing a reproducible platform for future research.

In summary, our contributions are as follows:

- • We propose IC-Effect, the first instruction-guided video VFX editing framework built on DiT, achieving realistic effects while preserving background and temporal consistency.
- • We leverage DiT’s contextual learning with source video as clean condition, introducing spatiotemporal sparse tokenization and position correction for efficient and precise video VFX editing.
- • We construct the first high-quality paired video VFX dataset and demonstrating IC-Effect’s effectiveness and superiority through extensive qualitative and quantitative experiments.

### 2. Related Work

#### 2.1. Diffusion-Based Video Generation and Editing

The success of diffusion models in image generation [18, 39, 44, 60, 61, 63] drives their application to Video tasks [8, 17, 27, 30, 52, 53, 55, 57]. Early approaches [17, 55] extended image diffusion models by integrating temporal modules to produce short clips, while subsequent works [8, 53] design specialized spatiotemporal architectures to enhance motion consistency. However, the U-Net backbone struggles with long-term dynamics. This limitation has been largely overcome by DiT-based models like Sora [6], CogVideoX [57], HunyuanVideo [27], and Wan [52], which treat videos as unified spatiotemporal token sequences and employ 3D full attention to capture complex dependencies, significantly enhancing generation quality, smoothness, and semantic fidelity.

Concurrently, diffusion-based video editing models [15, 25, 28, 32, 45–47, 50, 58] are also widely explored. Initial strategies [15, 28, 32] inject source frames during inference but often suffer from flickering or artifacts. More stable alternatives [16, 20, 34–37] encode structural priors—such as depth or optical flow—via auxiliary modules. Recent DiTbased image and video editing models [22, 25, 48, 50, 58] enable diverse editing by fusing source frames, reference conditions, and latent tokens at the token level. Yet, they fall short in video effects editing due to the abstract nature of visual effects and scarce paired training data. To bridge this gap, we propose a novel framework for text-guided video effects editing that synthesizes dynamic, semantically coherent effects while maintaining original content integrity and motion consistency.

###### Legends Prompt

| |
|---|

| |
|---|

###### Causal Attention

𝒁𝒁𝒕𝒕−𝟏𝟏

[Figure 61]

Learnable

Temporally Sparse Tokens 𝒁𝒁𝑺𝑺↓

Add a dynamic light effect on the minaret.

| |
|---|

Key

Query

[Figure 62]

Spatially Sparse Tokens 𝒁𝒁𝑰𝑰 Frozen

| |𝒁𝒁𝑻𝑻| | | |𝒁𝒁𝑺𝑺↓| |𝒁𝒁𝑰𝑰|
|---|---|---|---|---|---|---|---|
|𝒁𝒁𝑻𝑻| | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
|𝒁𝒁𝑺𝑺↓| | | | | | | |
| | | | | | | | |
|𝒁𝒁𝑰𝑰| | | | | | | |

| |
|---|

Noisy Latent Tokens 𝒁𝒁𝑻𝑻

DiT Block

Source Video 𝑽𝑽𝑺𝑺 Text Encoder

[Figure 63]

Target video 𝑽𝑽𝑻𝑻

DiT Block

[Figure 64]

𝑽𝑽𝑰𝑰

[Figure 65]

Feed Forward

[Figure 66]

𝑽𝑽𝑺𝑺↓

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

QKV Projection LoRA

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

CA

QKV Projection LoRA

[Figure 75]

[Figure 76]

3D VAE

[Figure 77]

SA

Add Noise

| | |
|---|---|
|| |
|---|
| |

|| |
|---|
<br><br>|
|---|

| |
|---|

𝒁𝒁𝒕𝒕

- Figure 2. The overall architecture and training paradigm of IC-Effect. Given a source video VS, IC-Effect first tokenizes it into spatiotemporal sparse tokens ZS↓ and ZI. These tokens are concatenated with noisy target tokens ZT along the token dimension to form a unified sequence, which is fed into a DiT module equipped with causal attention. At the output, ZS↓ and ZI are discarded, and only the target tokens ZT are decoded by a VAE to produce the edited video. During training, we first fine-tune the model with high-rank LoRA to acquire general video editing and instruction-following capabilities, and then further fine-tune it with low-rank LoRA on a small set of paired visual effects data to accurately capture the stylistic characteristics of diverse effects.

#### 2.2. Visual Effect Generation

multimodal conditions like depth maps and camera trajectories. However, since DiT’s computational cost grows quadratically with token count, directly concatenating long conditional token incurs substantial overhead, limiting the efficiency of high-resolution or long-duration video generation. In contrast, our method introduces spatiotemporally sparse condition tokens, fully exploiting DiT’s contextual understanding with minimal extra computation.

VFX creation aims to generate or edit images and videos to present hand-drawn, cartoon, or other creative artistic styles. In recent years, the development of artificial intelligence drives the partial automation of the VFX production pipeline [4, 59], significantly improving content creation efficiency. In the image domain, PhotoDoodle [23] first introduces a text-guided automatic image VFX editing method that adds semantically consistent visual elements to static images according to textual prompts. In the video domain, works such as OmniEffects [38] and VFX-Creator [33] achieve text-guided video VFX generation. However, these methods mainly focus on synthesizing stylized video clips from scratch rather than performing precise and controllable dynamic VFX editing on existing videos. This paper aims to fill this gap by enabling creative visual style fusion based on textual descriptions, while preserving the structural integrity and motion coherence of the original video.

### 3. Method

In this section, we introduce the overall architecture of IC-Effect in Section 3.1. Next, we detail its key innovations: in-context conditioning for video VFX editing (Sec-

- tion 3.2), Effect-LoRA for efficient effect learning (Sec-
- tion 3.3), and spatiotemporal sparse tokenization for reduced computational cost (Section 3.4). Finally, we describe the proposed special-effects dataset in Section B. 3.1. Overall Architecture

#### 2.3. Vision In-context Learning

The overall architecture of IC-Effect illustrated in Fig. 2. We employ a two-stage training strategy:

In-context learning (ICL) enables models to rapidly adapt to new tasks with only a few examples and is widely adopted in large language models [7]. Inspired by this, recent work explore the conditioning capability of DiTs, revealing their strong potential across various vision tasks [10, 21, 23, 25, 62]. Existing methods [10, 21, 23, 62] typically introduce additional condition tokens, concatenate them with latent noise sequences, and jointly model them through 3D full attention to implicitly learn condition-content correlations. Recently, this framework is extended to the more challenging video generation domain [9, 25, 26, 50], achieving remarkable progress. For example, FullDiT [26] achieves highly controllable text-to-video generation by integrating

Pretraining Video-Editor. We first pretrain a video editor on a large-scale video editing dataset. This stage enables the model to accurately understand and respond to the text instruction, achieves controllable editing behavior, and establish a strong foundation for subsequent effect generation. Video Effect Fine-Tuning. After pretraining, we introduce an Effect-LoRA module using Low-Rank Adaptation (LoRA) [19] and fine-tune it on a small set of paired video effect data. This stage focuses on learning the visual elements and stylistic characteristics of specific video effects, enabling precise modeling across different effect types. The proposed design supports efficient and lightweight cus-

tomization, meeting the needs of diverse and personalized video effect generation.

#### 3.2. In-Context Conditioning for Video Editing

Given a source video VS and its corresponding text instruction TE, we aim to perform video VFX editing that aligns with the textual description while preserving the source video’s spatial layout and temporal coherence. Preserving the spatial layout and temporal coherence of the source video is essential for achieving faithful and artifactfree video VFX editing. To achieve this, we leverage the strong contextual modeling capability of the DiT-based T2V model and reformulate video VFX editing as a conditional generation problem, while minimizing modifications to the pretrained DiT architecture.

Specifically, we encode the source and edited video into latent representations ZS and ZT through a 3D VAE. Both representations share the same 3D rotary positional embeddings, which help the model capture stable relative spatiotemporal relationships during contextual modeling. ZS and ZT are patchified and concatenated into a unified token sequence, which is processed by the self-attention mechanism within the DiT blocks,

MMA([ZT;ZS]) = softmax

QK⊤ √

d

V, (1)

where ZT denotes the noisy latent tokens and ZS represents clean conditional tokens. By using clean conditional tokens, the method preserves both the spatial structure and temporal motion information of the source video, providing faithful source information to ZT and preventing degradation in video quality during iterative denoising. Additionally, the attention mechanism allows the model to copy from the clean tokens or generate new content according to instructions, ensuring the edited video retains the source background information. During training, we apply the flow matching loss [13] only to the latent tokens to guide the model in learning high-quality and structurally consistent video VFX editing.

Causal Attention. Within the bidirectional attention of DiT, interactions between latent noise and conditional tokens can cause clean conditional tokens to be fused with noisy representations, thereby degrading the quality of the generated results. To prevent this, we introduce a causal attention mechanism. In this mechanism, noise tokens attend to both themselves and clean conditional tokens, while conditional tokens attend only to themselves, avoiding any interaction with noise tokens. We realize this causal attention using a specifically designed attention mask:

Mi,j = −∞, if i ∈/ ZT and j ∈ ZT, 0, otherwise.

(2)

This causal attention effectively isolates clean conditional tokens from latent noise, preserving their fidelity and ensuring high-quality generation.

#### 3.3. Effect-LoRA

LoRA [19] enables efficient fine-tuning of large-scale pretrained models by introducing trainable low-rank matrices while keeping the original weights frozen. Given a pretrained weight matrix W0 ∈ Rm×n, LoRA introduces two low-rank matrices A ∈ Rm×r (r ≪ m) and B ∈ Rr×n (r ≪ n), where r ≪ min(m,n), to model the parameter update:

W = W0 + AB. (3)

This formulation significantly reduces the number of trainable parameters while maintaining performance comparable to full-model fine-tuning.

To effectively learn the editing pattern of a specific video effect from limited paired VFX data, we propose EffectLoRA. Effect-LoRA fine-tunes only a small number of trainable parameters, enabling efficient learning of video effect editing patterns while significantly reducing the risk of overfitting. In our IC-Effect framework, the Video-Editor is trained on a large-scale paired dataset with a high-rank LoRA, which equips the model with strong instructionfollowing capability. In contrast, Effect-LoRA adopts a low-rank LoRA specifically designed to capture the editing style of a single video effect. By guiding the behavior of the Video-Editor, Effect-LoRA enables the model to generate results that exhibit the desired effect style. When a new source video VS and a corresponding text instruction are provied, the model produces a target video VT that exhibits the visual effects described by the instruction.

#### 3.4. Spatiotemporal Sparse Tokenization

The T2V model built upon the DiT [40] architecture mainly relies on attention mechanisms, whose computational complexity is proportional to the square of the token length. Consequently, directly using the source video with the same resolution as the edited output as the contextual condition results in excessively long input tokens, thereby significantly increasing computational cost. To overcome this limitation, we introduce a spatiotemporal sparse tokenization (STST) strategy that converts the source video VS into a set of spatiotemporally sparse tokens, effectively reducing the number of conditional tokens processed by the T2V model and improving inference efficiency.

Given a source video VS, we do not directly convert it into latent tokens. Instead, we encode a downsampled version of the video along with its first frame into latent representations ZS↓ and ZI using a 3D VAE. These representations are then patchified into temporally sparse tokens and spatially sparse tokens, respectively. Afterwards, ZS↓, ZI

and ZT are concatenated along the token dimension and fed into the DiT-based text-to-video (T2V) model. Formally, the transformation in Eq.(1) is expressed as follows:

MMA ZT;ZS↓;ZI = softmax

QK⊤ √

d

V. (4)

To efficiently capture the spatiotemporal information from the source video, we sparsify it along both temporal and spatial dimensions. The temporally sparse tokens provide essential motion information, while the spatially sparse tokens supply complementary fine-grained spatial details. This spatiotemporal sparse tokenization achieves a balance between computational efficiency and visual fidelity, preserving the complete spatiotemporal characteristics of the source video. It provides reliable guidance for editing without incurring the high computational cost associated with tokenizing the full-resolution video.

Furthermore, Eq.(2) is reformulated as follows:

 

0, if i ∈ ZT, 0, if (i,j) ∈ ZS↓ or (i,j) ∈ ZI, −∞, otherwise.

(5)

Mi,j =



Spatiotemporal Position Correction. Although spatiotemporal sparse tokens significantly improve computational efficiency, they introduce a critical issue: spatiotemporal misalignment between the sparse conditional tokens and the target generation space. This misalignment often leads to structural inconsistencies between the edited video and the source video, thereby degrading overall model performance. To address this problem, inspired by OmniControl2 [49], we propose a spatiotemporal position correction technique.

For temporal sparse tokens ZS↓, we establish explicit correspondences between the temporal sparse tokens and the target regions to prevent spatial misalignment:

(n · i,n · j). (6)

PZ↓

= PZ

T

S

For spatially sparse tokens ZI, we use the positional encoding of the first frame of the noisy latent tokens ZT to avoid temporal misalignment. The spatiotemporal position correction technique is crucial, as it ensures that the spatiotemporally sparse tokens provide appropriate temporal and spatial information for video editing.

#### 3.5. Video Effects Dataset

We construct the first VideoVFX dataset for text-guided video effect editing, covering 15 high-quality effect categories with over 20 curated videos. The main categories include dynamic particle dispersion, particle dissipation and reassembly, line traversal, subject cartoon mirroring, flame

combustion, graffiti, and architectural bouncing. Each sample contains a source video VS—from real-world scenes to portraits—and a target video VT with artist-designed effects exhibiting precise spatiotemporal coherence. Effects include local stylization, animated overlays, motionenhanced transitions, new dynamic elements, and structural modifications. For each pair, we provide VS, VT, and a text instruction, enabling supervised learning for instructionfollowing video editing models.

### 4. Experiment

#### 4.1. Experimental Setup

Implementation Details. In the pretraining stage of VideoEditor, we initialize the DiT architecture with the parameters of Wan 2.2-A14B-T2V [52] and train it on a selfconstructed video editing dataset. All videos are resized to 224 × 416 with 81 frames. The training uses four A800 GPUs with a batch size of 2, a learning rate of 1×10−4, and a LoRA rank of 96 for 50,000 optimization steps. The resulting LoRA weights are merged into the base T2V model to form Video-Editor, which serves as the backbone for subsequent tasks. In the Effect-LoRA training stage, we further fine-tune Video-Editor on the paired video VFX editing dataset using two A800 HPUs for 1,000 steps, with a LoRA rank of 32, a batch size of 2, and a learning rate of 1 × 10−4. During inference, the model generates edited videos at a resolution of 480 × 832 with 81 frames.

Baseline Methods. To evaluate the performance of our method, we compare it with several open-source approaches, including InsV2V [12], InsViE [56], VACE [25], and Lucy Edit [50]. For a fair comparison, we conduct experiments under two scenarios: common video editing and video VFX editing. In the Video VFX editing scenario, we fine-tune the attention layers of the above models using the same VFX dataset. Finally, we compare all the trained LoRA-based models with our IC-Effect framework.

Benchmarks. We collect a total of 80 video samples from the DAVIS [41] dataset and the Internet to evaluate the performance of the proposed Video-Editor. For the customized video effect editing task, we further construct a new benchmark dataset, entirely sourced from the Internet, consisting of 50 high-quality videos that cover both single-subject and multi-subject scenes. The subjects include humans, animals, buildings, and vehicles. This benchmark aims to comprehensively assess the model’s capability in customized video effect editing across diverse and realistic scenarios.

Evalution Metrics. To evaluate the effectiveness of the proposed method, we adopt the following automatic evaluation metrics that analyze performance from multiple perspectives: 1)Video Quality. We employ CLIP [42] Image Similarity (CLIP-I) to measure temporal consistency by computing the cosine similarity between consecutive

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

Source VFX Video Data

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

Add a purple lightning effect to the edge of the hot air balloon. Add a flame-colored lightning effect to the edge of the green car’s rear window.

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

Source VFX Video Data

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

Add a dynamic line shuttle effect around the lighthouse. Add a dynamic line shuttle effect on the white church.

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

Source VFX Video Data

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

Add a particle spread effect that drifts towards the upper right for the dog. Add a particle spread effect for women wearing dresses.

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

Source VFX Video Data

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

Add a graffiti effect from bottom left to top right on the background wall. Add a graffiti effect from top left to bottom right on the wall behind the woman.

- Figure 3. Video VFX Editing Results of IC-Effect. IC-Effect accurately edits the source video following textual instructions, applying the visual effect styles present in the VFX data. The complete video is available in our supplementary materials. frames of the edited videos using the CLIP image encoder.

- 2)Semantic Alignment. To assess the semantic consistency between the edited results and the text prompts, we use both the CLIP [42] encoder and the ViCLIP [54] encoder to calculate frame-level and video-level similarities between the edited videos and the corresponding text prompts.
- 3)Overall Quality. We adopt multiple sub-metrics from the VBench [24] toolkit, including Smoothness, Dynamic Degree, and Aesthetic Quality, to comprehensively evaluate the visual fidelity and naturalness of the edited videos. To further evaluate the structural preservation and effectiveness of the edited results, We further employ GPT-4o [1] to evaluate edited results along two dimensions: structural preservation, measuring spatial–temporal consistency with the source video, and effect accuracy, assessing alignment with the intended target (reference effect or textual prompt).

- 4.2. Video VFX Editing Results

the spatiotemporal consistency of non-effect regions, avoiding color drift and structural distortion. Notably, IC-Effect maintains stable performance and a high success rate under different settings, exhibiting strong generalization and controllability, and achieves consistent editing results in production environments without selective sampling.

#### 4.3. Comparison with Baselines

Qualitative Comparison. Fig. 4 present qualitative comparisons of the proposed method on general video editing and customized video effect editing tasks. As shown in Fig. 4, compared with state-of-the-art video editing methods, Video-Editor demonstrates stronger instructionfollowing capability, accurately performing the modifications described in the textual prompts while effectively preserving the structural and content consistency of non-edited regions. This advantage mainly stems from the use of our carefully constructed high-quality training dataset and the task-oriented model design. In the customized video VFX editing task (Fig. 4), our method significantly outperforms the baselines. The generated videos exhibit higher visual quality, with added effects that align closely with the original ones in terms of style and motion dynamics, faithfully adhering to the textual instructions while minimizing content drift and unintended modifications to non-target areas. These results demonstrate that our method achieves a wellbalanced trade-off between visual coherence and precise, controllable editing, substantially enhancing the reliability and naturalness of complex video effect editing.

Fig. 3 presents the results of IC-Effect in video VFX editing. IC-Effect accurately injects visual effects into the source video according to the given instruction while maintaining consistency between the generated effect styles and those in the training data, demonstrating excellent instructionfollowing ability. This performance benefits from training on high-quality paired data, which enables the model to learn robust effect generation and background preservation capabilities. When fine-tuned with Effect-LoRA on a limited dataset of paired effect data, IC-Effect continues to generate the specified effects stably and strictly preserves

- Table 1. Quantitative Comparison of Common Video Editing and Video VFX Editing. The best and second-best values are highlighted in blue and green , respectively.

Video Quality Semantic Alignment Overall Quality GPT Score Method/Metrics

CLIP-I (↑) CLIP-T (↑) ViCLIP-T (↑) Smoothness (↑) Dynamic Degree (↑) Aesthetic Quality (↑) Structural Preservation (↑) Effect Accuracy (↑)

InsV2V [12] 0.9750 22.9317 21.7245 0.9798 0.6147 0.4723 3.9006 2.4452 InsViE [56] 0.9686 23.0331 20.5973 0.9695 0.5856 0.4877 3.6468 2.3687 VACE [25] 0.9698 23.4918 21.4685 0.9803 0.4820 0.5190 3.8562 2.6218

###### Common

Lucy Edit [50] 0.9721 22.8130 21.1405 0.9811 0.5737 0.5071 4.0529 3.1218

Ours 0.9795 25.6774 24.3290 0.9815 0.6374 0.5261 4.3824 4.2652

InsV2V [12] 0.9767 25.6705 23.8159 0.9882 0.2019 0.5612 3.9326 1.9807 InsViE [56] 0.9770 25.6530 23.7950 0.9858 0.3258 0.5158 4.1012 2.8988 VACE [25] 0.9782 25.5283 24.5594 0.9897 0.2788 0.5401 4.2105 3.6988

###### VFX

Lucy Edit [50] 0.9784 27.0618 26.2143 0.9894 0.3653 0.5546 4.3461 3.9423

Ours 0.9786 27.2321 26.6312 0.9911 0.3771 0.5823 4.7947 4.5614

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

Source Video Instruction

Source Video

VFX Data

Instruction

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

Video VFX Editing

Common Video Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

Add a particle spread effect that floats to the upper right for the man in the video.

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

Delete the person wearing a white T-shirt and gray jacket.

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

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

Source Video

Source Video

VFX Data

Instruction Instruction

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Video VFX Editing

Common Video Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

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

Add a "Fa Xiang Tian Di" effect to the upper right of the female figure on the right.

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

Make it Cyberpunk style.

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

- Figure 4. Qualitative Comparison of Video Editing. IC-Effect demonstrates strong performance in instruction following, spatiotemporal consistency, and editing quality. The complete video is available in our supplementary materials.

Quantitative Comparison. As shown in Table 1, our method outperforms all baseline approaches across all evaluation metrics on the general video editing task, achieving the best overall objective performance. This demonstrates that Video-Editor possesses stronger comprehensive capabilities in terms of temporal consistency, semantic alignment, and overall visual quality. For the video effect editing task, our method also achieves the best metrics. Notably, GPT-4o significantly outperforms baseline methods in both structure preservation and effect accuracy. The significant improvements in these metrics further validate the superiority of the proposed method in editing precision and semantic consistency, highlighting its enhanced controlla-

bility and generation quality in complex, customized video editing scenarios.

#### 4.4. User Study

To further validate the effectiveness of the proposed method, we conduct a user study through an online questionnaire to evaluate user preferences in two scenarios: conventional video editing and customized video effect editing. Each participant is presented with a text instruction, a source video, and two edited results respectively generated by our method and a baseline. The two results are shown in a randomized order to prevent participants from inferring their sources and to minimize subjective bias. The study

- Table 2. Quantitative Comparison of Ablation Studies. The best and second-best values are highlighted in blue and green , respectively.

Video Quality Semantic Alignment Overall Quality GPT Score Inference overhead Method/Metrics

CLIP-I (↑) CLIP-T (↑) ViCLIP-T (↑) Smoothness (↑) Dynamic Degree (↑) Aesthetic Quality (↑) Structural preservation (↑) Effect accuracy (↑) Time GPU memory W/o STST 0.9792 27.5514 26.3071 0.9885 0.3806 0.5940 4.8375 4.5975 5880s 74.71GB

w/o ZI 0.9367 20.9034 21.4638 0.9657 0.8163 0.3805 4.0123 3.6140 2617s 63.74GB w/o Pretrain 0.9224 26.7977 25.8940 0.9886 0.3762 0.5395 4.2471 3.9134 2790s 64.07GB w/o Effect-LoRA 0.9749 25.0413 23.4875 0.9802 0.3709 0.5395 4.1428 3.9423 2790s 64.07GB Ours 0.9786 27.2321 26.6312 0.9911 0.3771 0.5823 4.7947 4.5614 2790s 64.07GB

Structual Fidelity Effect Consistency Overall Preference

Structual Fidelity Effect Consistency Overall Preference

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

100

100

SourceVideow/opretrainw/oEffectLoRAFull

PreferenceRate(%)

PreferenceRate(%)

75

75

98.22

97.47

97.41

97.16

95.98

94.81

94.09

100

100

93.53

92.68

91.93

91.57

91.38

90.64

50

50

88.37

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

84.92

82.32

82.16

90

83.5

76.25

82

w/oSTSTw/o𝑍𝑍𝐼𝐼

60.18

25

25

0

0

vs InsV2V vs InsViE vs VACE vs LucyEdit

vs InsV2V vs InsViE vs VACE vs LucyEdit

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

(a) User Study Results in Common Video Editing Tasks

(a) User Study Results in Video VFX Editing Tasks

- Figure 5. Human Preference Study. Our method is significantly more preferred by users compared with the comparative methods.

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

follows an A/B testing protocol, where participants evaluate the results from three aspects: (1) instruction following, (2) consistency between the edited video and the source video (i.e., structual fidelity) and (3) Overall preference. A total of 20 participants are recruited for the study. As shown in Fig. 5, the statistical results demonstrate that, compared with baseline methods, our approach achieves higher user preference in both instruction adherence and source-video fidelity, verifying its effectiveness and advantage in realworld applications.

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Add a particle aggregation effect to the white cake.

Figure 6. Ablation Study of the Proposed Approach.

complete method efficiently generates video effects that are highly consistent with the text prompts while faithfully preserving the original video content. As illustrated in Table 2, the performance of the full method across all metrics further confirms these observations.

#### 4.5. Ablation Studies

To evaluate the effectiveness of the proposed strategies and modules, we conduct detailed ablation experiments, including the spatiotemporal sparse tokenization strategy, Video-Editor pretraining, and Effect-LoRA. As shown in the Fig. 6, our method maintains editing quality comparable to the fully tokenized model, while various metrics in Table 2 also approach those of the fully tokenized model, and the computational cost is significantly reduced. Without using a high-quality first frame as the conditional input, relying solely on temporally sparse tokens fails to provide sufficient spatial details, resulting in noticeable blotchy artifacts and blurring, which is also reflected by the lowest metrics in Table 2. In contrast, our spatiotemporal sparse tokenization strategy introduces minimal additional computation while substantially improving visual fidelity and spatiotemporal consistency, as also confirmed by the qualitative results in Table 2. Furthermore, skipping the pretrained Video-Editor module and directly training Effect-LoRA causes the generated video to apply particle dispersion effects to all subjects, instead of following the textual instructions to apply

### 5. Conclusion

In this work, we propose IC-Effect, an instruction-guided few-shot video effect editing framework that learns unique visual styles from a minimal number of paired effect samples. By leveraging the intrinsic contextual learning of the DiT architecture, our method performs video VFX editing that strictly follows textual instructions while preserving background consistency. The two-stage training strategy ensures both instruction adherence and the learning of specific effect styles. Additionally, our spatiotemporal sparse tokenization mechanism preserves full spatiotemporal information from the source video while significantly reducing computational cost. We also introduce a new dataset with 15 types of video effects, providing a valuable benchmark for future research. Extensive experiments show that IC-Effect consistently outperforms existing methods in both general video editing and video VFX editing, demonstrating superior effect fidelity and background consistency.

- them only to the white cake. Conversely, using only the pretrained Video-Editor without Effect-LoRA fails to produce customized VFX that align with user intent. In contrast, the

### References

- [1] Gpt-4o. Accessed May 13, 2024 [Online] https:// openai.com/index/hello-gpt-4o/. 6, 1, 2
- [2] Pexels. https://www.pexels.com/, 2025. 2
- [3] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 2

- [4] Alla Belova. Google doodles as multimodal storytelling. Cognition, communication, discourse, (23):13–29, 2021. 3
- [5] Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Anylength video inpainting and editing with plug-and-play context control. In SIGGRAPH, pages 1–12, 2025. 2
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 2

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, and et al. Askell, Amanda. Language models are few-shot learners. In NeurIPS, pages 1877–1901, 2020. 3
- [8] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, pages 7310–7320, 2024. 2
- [9] Lan Chen, Yuchao Gu, and Qi Mao. Univid: Unifying vision tasks with pre-trained video generation models. arXiv preprint arXiv:2509.21760, 2025. 3
- [10] Lan Chen, Qi Mao, Yuchao Gu, and Mike Zheng Shou. Edit transfer: Learning image editing via vision in-context relations. arXiv preprint arXiv:2503.13327, 2025. 2, 3
- [11] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In CVPR, pages 22831–22840, 2025. 2
- [12] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. In ICLR, 2024. 2, 5, 7, 1
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 4
- [14] Chenjian Gao, Lihe Ding, Xin Cai, Zhanpeng Huang, Zibin Wang, and Tianfan Xue. Lora-edit: Controllable first-frameguided video editing via mask-aware lora fine-tuning. arXiv preprint arXiv:2506.10082, 2025. 2
- [15] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 2
- [16] Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang,

- Mike Zheng Shou, and Kevin Tang. Videoswap: Customized video subject swapping with interactive semantic point correspondence. In CVPR, pages 7621–7630, 2024. 2
- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2
- [19] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 3, 4
- [20] Zhihao Hu and Dong Xu. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073,

2023. 2

- [21] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 2, 3
- [22] Shijie Huang, Yiren Song, Yuxuan Zhang, Hailong Guo, Xueyin Wang, and Jiaming Liu. Arteditor: Learning customized instructional image editor from few-shot examples. In ICCV, pages 17651–17662, 2025. 2
- [23] Shijie Huang, Yiren Song, Yuxuan Zhang, Hailong Guo, Xueyin Wang, Mike Zheng Shou, and Jiaming Liu. Photodoodle: Learning artistic image editing from few-shot pairwise data. arXiv preprint arXiv:2502.14397, 2025. 2, 3
- [24] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024. 6
- [25] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2, 3, 5, 7, 1
- [26] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Video generative foundation models with multimodal control via full attention. In ICCV, pages 15737– 15747, 2025. 3
- [27] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 2
- [28] Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. TMLR, 2024. 2
- [29] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Diffueraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025. 2
- [30] Yuanhang Li, Qi Mao, Lan Chen, Zhen Fang, Lei Tian, Xinyan Xiao, Libiao Jin, and Hua Wu. Starvid: Enhanc-

- ing semantic alignment in video diffusion models via spatial and syntactic guided attention refocusing. arXiv preprint arXiv:2409.15259, 2024. 2
- [31] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, pages 38–55. Springer, 2024. 2
- [32] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. In CVPR, pages 8599–8608, 2024. 2
- [33] Xinyu Liu, Ailing Zeng, Wei Xue, Harry Yang, Wenhan Luo, Qifeng Liu, and Yike Guo. Vfx creator: Animated visual effect generation with controllable diffusion transformer. arXiv preprint arXiv:2502.05979, 2025. 1, 2, 3
- [34] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In AAAI, pages 4117–4125, 2024. 2
- [35] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024.
- [36] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Leqi Shen, Chenyang Qi, Jixuan Ying, Chengfei Cai, Zhifeng Li, Heung-Yeung Shum, et al. Follow-your-click: Open-domain regional image animation via motion prompts. In AAAI, pages 6018–6026, 2025.
- [37] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. arXiv preprint arXiv:2506.05207, 2025. 2
- [38] Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025. 1, 2, 3
- [39] Qi Mao, Lan Chen, Yuchao Gu, Zhen Fang, and Mike Zheng Shou. Mag-edit: Localized image editing in complex scenarios via mask-based attention-adjusted guidance. In ACM MM, pages 6842–6850, 2024. 2
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, pages 4195–4205, 2023. 2, 4
- [41] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 5
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 5, 6, 1
- [43] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, H Khedr, R R¨adle, C Rolland, L Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2

- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 2
- [45] Yiren Song, Shijie Huang, Chen Yao, Xiaojun Ye, Hai Ci, Jiaming Liu, Yuxuan Zhang, and Mike Zheng Shou. Processpainter: Learn painting process from sequence data. arXiv preprint arXiv:2406.06062, 2024. 2
- [46] Yiren Song, Danze Chen, and Mike Zheng Shou. Layertracer: Cognitive-aligned layered svg synthesis via diffusion transformer. arXiv preprint arXiv:2502.01105, 2025.
- [47] Yiren Song, Cheng Liu, and Mike Zheng Shou. Makeanything: Harnessing diffusion transformers for multidomain procedural sequence generation. arXiv preprint arXiv:2502.01572, 2025. 2
- [48] Yiren Song, Cheng Liu, and Mike Zheng Shou. Omniconsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445, 2025. 2
- [49] Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280,

2025. 5

- [50] DecartAI Team. Lucy edit: Open-weight text-guided video editing. 2025. 2, 3, 5, 7, 1
- [51] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, pages 402–419. Springer, 2020. 1
- [52] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 5
- [53] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [54] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In ICLR, 2024. 6, 1
- [55] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, pages 7623–7633, 2023. 2
- [56] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In ICCV, pages 16692–16701, 2025. 2, 5, 7, 1
- [57] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan.Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In ICLR, 2025. 1, 2
- [58] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and

- Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025. 2
- [59] Emilie Yu, Kevin Blackburn-Matzen, Cuong Nguyen, Oliver Wang, Rubaiat Habib Kazi, and Adrien Bousseau. Videodoodles: Hand-drawn animations on videos with scene-aware canvases. ACM Transactions on Graphics (TOG), 42(4):1–12, 2023. 3
- [60] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In CVPR, pages 8069–8078,

2024. 2

- [61] Yuxuan Zhang, Lifu Wei, Qing Zhang, Yiren Song, Jiaming Liu, Huaxia Li, Xu Tang, Yao Hu, and Haibo Zhao. Stablemakeup: When real-world makeup transfer meets diffusion model. arXiv preprint arXiv:2403.07764, 2024. 2
- [62] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. In ICCV, pages 19513–19524,

2025. 2, 3

- [63] Yuxuan Zhang, Qing Zhang, Yiren Song, Jichao Zhang, Hao Tang, and Jiaming Liu. Stable-hair: Real-world hair transfer via diffusion model. In AAAI, pages 10348–10356, 2025. 2

## IC-Effect: Precise and Efficient Video Effects Editing via In-Context Learning Supplementary Material

In this supplementary material, we provide additional implementation details, extended ablation studies, more qualitative results, and a discussion of limitations, as summarized below:

- • In Section A, we present additional implementation details of IC-Effect, the baselines, the quantitative metrics, and the user study setup.
- • In Section B, we describe the datasets used for VideoEditor pretraining as well as the paired VFX editing dataset.
- • In Section C and Section D, we demonstrate that ICEffect supports flexible instruction control and multieffect editing.
- • In Section E, we conduct ablation studies on causal attention and positional correction.
- • In Section F, we discuss the limitations of the proposed method and outline future directions.
- • In Section G, we provide additional video VFX editing results of IC-Effect and present extended qualitative comparisons with baseline methods on both general video editing and video VFX editing tasks.

### A. Implementation Details

#### A.1. Training Details

During the pre-training stage of VideoEditor, we initialize the DiT architecture using the Wan 2.2-A14B-T2V weights from the Wan model. All training videos are resized to a spatial resolution of 224 × 416 with 81 frames. We train a rank-96 LoRA for 50k iterations using the AdamW optimizer with a batch size of 2, a learning rate of 1×10−4, and a weight decay of 1 × 10−2, on four A800 GPUs. During the Effect-LoRA training stage, we initialize the DiT architecture with the merged weights from VideoEditor. The video resolution and frame count remain 224 × 416 and 81 frames, respectively. We train a rank-32 LoRA for 1000 iterations using AdamW with a batch size of 2, a learning rate of 1 × 10−4, and a weight decay of 1 × 10−2, on two A800 GPUs.

#### A.2. Inference Details

During inference, we generate videos at a spatial resolution of 480×832. For source videos with ≥ 81 frames, we select the first 81 frames; for videos with < 81 frames, we select the first 4n + 1 frames as the source. We set the classifierfree guidance scale to 5.0 and perform 50 denoising steps.

https : / / huggingface . co / Wan - AI / Wan2 . 2 - I2V -

- A14B-Diffusers

#### A.3. Implementation Details of Baselines

For common video editing comparisons, we use the official codes and released weights of InsV2V [12], InsViE [56], VACE [25], and Lucy Edit [50]. For video VFX editing comparisons, we train the competing methods on the same paired VFX dataset using their officially provided training code or our faithful re-implementations.

#### A.4. Evaluation Details

Automatic Evaluation. We use the CLIP Vit-L/14 [42] model to calculate CLIP Image Similarity and CLIP Text Alignment, assessing the quality of the edited videos in terms of both temporal consistency and alignment with the textual prompts. We further evaluate the semantic consistency between the edited videos and their corresponding textual prompts by computing video-level semantic similarity using ViClip [54]. In addition, we adopt multiple submetrics from VBench, including motion smoothness, dynamic degree, and aesthetic quality, to comprehensively assess the visual fidelity and naturalness of the edited videos. Following the official VBench settings, we compute motion smoothness using a frame interpolation model, evaluate dynamic degree using RAFT [51], and assess aesthetic quality using the LAION aesthetic predictor.

GPT-4o Evaluation. To evaluate the structural preservation and editing effectiveness of the generated results, we further employ the state-of-the-art vision–language model (VLM) GPT-4o [1] as an automatic evaluator. For general video editing, we uniformly sample five frames from both the source video and the edited video. For each sample, the VLM receives the source frames, the edited frames, and the textual instruction, and it is required to score the editing result on two dimensions—structural preservation and editing effectiveness—using a 1 − 5 rating scale. For video VFX editing, we additionally provide the reference VFX video so that the VLM can assess editing effectiveness with respect to the visual effect demonstrated in the reference. Fig. 10

https://github.com/amazon- science/instruct-

video-to-video https://github.com/langmanbusi/InsViE https://github.com/ali-vilab/VACE https://github.com/DecartAI/lucy-edit-comfyui https://huggingface.co/openai/clip-vit-large-

patch14 https://huggingface.co/OpenGVLab/ViCLIP https://huggingface.co/lalala125/AMT/tree/main https : / / dl . dropboxusercontent . com / s /

4j4z58wuv8o0mfz/models.zip

https : / / github . com / LAION - AI / aesthetic predictor

and Fig. 11 present the prompt templates used to guide the VLM during automatic evaluation.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

SourceResult2Result1SourceResult2Result1

User Study. For common video editing comparisons, each questionnaire presents participants with the source video, the edited videos produced by the competing methods, our edited video, and the corresponding editing instruction. The edited videos from the competing methods and ours are randomly ordered to prevent participants from inferring their origins. We ask participants to answer questions along the following three dimensions:

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Add a red lightning effect to the edge of the stone cross.

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Add a purple lightning effect to the edge of the stone cross.

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

- • Instruction Following: Which edited video better follows the textual instruction?
- • Structual Fidelity: Which edited video preserves the non-edited regions more faithfully and aligns better with the source video?
- • Overall Preference: Which edited result do you prefer subjectively?

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Add a red light particle line shuttle effect from back to front on roads.

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Add a yellow light particle line shuttle effect from front to back on roads.

Figure 7. Flexible Instruction-Based Control of Video VFX Editing Results.

For video VFX editing comparisons, we additionally present the corresponding reference VFX video and ask participants to answer questions along the following three dimensions:

ploy GPT-4o [1] to generate editing instructions for each source video, and then synthesize the edited video using VACE [25]. We filter the generated videos using a VLM to remove blurry or invalid results.

- • Effect Consistency: Which edited video better follows the textual instruction and more closely matches the visual effect in the reference VFX video?
- • Structual Fidelity: Which edited video preserves the non-edited regions more faithfully and aligns better with the source video?
- • Overall Preference: Which edited result do you prefer subjectively?

For style transfer data, GPT-4o [1] first generates the textual editing descriptions. We then edit the first frame using FLUX-kontext [3], estimate depth maps of the source video using a video depth estimator [11], and feed the edited first frame, depth maps, and textual prompt into VACE [25] to obtain the final stylized video.

In total, we collect approximately 50,000 high-quality paired video editing samples. All videos are resized to a spatial resolution of 480 × 832.

### B. Datasets

#### B.1. Datasets of Pretrained Video-Editor

#### B.2. Video VFX Datasets

To pre-train VideoEditor and equip it with strong instruction-following capability, we construct a highquality paired video editing dataset. Considering the requirements of video VFX editing, the constructed dataset covers the following editing tasks: addition, removal, replacement, attribute modification, and style transfer. The source videos primarily come from high-resolution videos on Pexels [2].

Since existing techniques cannot automatically produce high-quality paired video VFX editing data, we construct the first paired dataset consisting of source videos and their corresponding VFX videos. The dataset includes 15 distinct visual effects, covering: animated character insertion (with two different characters), anime duplication, rotating rings, graffiti strokes, lightning outlines, mystical aura, firework explosion, flame burning, architectural growth and bounce, line traversal, line shaping, particle dispersion, particle aggregation, and light-particle traversal. For each pair, we provide a {source video, target VFX video, editing instruction} triplet. All videos are standardized to a resolution of 480 × 832 and a duration of 5 seconds at 16 FPS.

For addition and removal data, we use GPT-4o [1] to analyze each video and identify the target object category. We

- then obtain the corresponding object masks using Grounding DINO [31] and SAM [43]. With the masks, we remove the target object using the object erasing model DiffuEraser [29]. The original video and the object-removed video form a pair for both addition and removal tasks, and we use GPT-4o [1] to generate the corresponding “remove” and “add” editing instructions. To ensure both video quality and instruction correctness, we further refine the data using a VLM and manual verification.

### C. Instruction Control for Video VFX Editing

Our IC-Effect framework supports flexible control over the editing results through instruction manipulation. By simply adjusting the textual prompt, the model flexibly controls the attributes and directions of the generated visual effects. As

For replacement and attribute modification data, we em-

###### Table 3. Quantitative Ablation of Positional Encoding Correction and Causal Attention. The best values are highlighted in blue .

Video Quality Semantic Alignment Overall Quality GPT Score Method/Metrics

CLIP-I (↑) CLIP-T (↑) ViCLIP-T (↑) Smoothness (↑) Dynamic Degree (↑) Aesthetic Quality (↑) Structural preservation (↑) Effect accuracy (↑) w/o PEC 0.9735 26.8748 25.7468 0.3715 0.9903 0.5324 4.4615 4.3908

w/o C-Attn 0.9783 27.0595 26.4503 0.3750 0.9887 0.5444 4.5583 4.4866 Ours 0.9786 27.2321 26.6312 0.9911 0.3771 0.5823 4.7947 4.5614

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

SourceResultSourceResult

w/oC-Attnw/oPECSourceFull

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

Add a graffiti effect from left to right on the sky. Add a particle aggregation effect to the woman by the seaside.

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

Add a purple flame burning special effect on the sofa. Add a particle spread effect to the man on the sofa.

Add a flame burning effect above the head of a woman wearing a dress.

Figure 8. Video Multi VFX Editing of IC-Effect.

Figure 9. Ablation Study on Positional Encoding Correction and Causal Attention.

shown in Fig. 7, minor modifications to the prompt already produce clear and controllable variations in the edited effects.

high-quality paired data is extremely challenging. In the future, we plan to train a feature extractor to directly extract target effects from reference VFX videos and use these extracted features to edit the source video.

### D. Video Multi VFX Editing

To verify that IC-Effect supports multi-effect editing, we perform mixed training on all effect categories based on Video-Editor. As shown in Fig. 8, IC-Effect accurately follows the textual instructions to inject multiple effects into the video without cross-effect leakage. It applies each effect precisely to the instruction-specified target while consistently preserving the non-edited regions. This capability mainly stems from the carefully designed architecture and the construction of high-quality training data, which together endow IC-Effect with strong instruction-following ability and enable precise source-video editing.

### G. Additional Results

#### G.1. Additional Results of Video VFX Editing

Fig. 12 and Fig. 13 present additional video VFX editing results produced by IC-Effect. Across diverse effect categories, our IC-Effect accurately follows the textual prompts and generates visually coherent and well-integrated edits.

#### G.2. Additional Qualitative Comparison of Video VFX Editing

Fig. 14 - Fig. 16 present additional quantitative comparisons with baseline methods on video VFX editing. Compared with the baselines, IC-Effect consistently follows the textual prompts to edit the source video while preserving its spatial structure and temporal coherence.

### E. Additional Ablation Studies

In this section, we conduct ablation studies on the positional encoding correction (PEC) and the causal attention mechanism (C-Attn). As shown in Fig. 9, removing positional encoding correction introduces artifacts and blurring, while altering information from the source video. Without the causal attention mechanism, the originally clear spatiotemporally sparse tokens are contaminated by latent noise, resulting in severe artifacts in the edited output. Furthermore, the quantitative results in Table 3 corroborate these observations. In contrast, the complete model demonstrates higher editing accuracy and stronger visual consistency.

#### G.3. Additional Qualitative Comparison of Common Video Editing

Fig. 17 - Fig. 21 present qualitative comparisons between IC-Effect and existing methods across diverse common video editing tasks, including addition, removal, attribute modification, replacement, and style transfer. Across these common editing scenarios, IC-Effect demonstrates superior performance: it accurately follows textual instructions, produces precise edits, and preserves the structural integrity of non-edited regions.

### F. Limitation and Future Work

A limitation of IC-Effect is its reliance on high-quality paired video VFX editing data. However, producing such

You are a meticulous video editing quality evaluator. Your task is to provide a detailed assessment of a video edit by comparing the original image with the edited image based on a given text prompt. Editing Prompt:"{prompt}" Instructions: Analyze the five provided video frames and evaluate the implementation of the "editing prompts." The frame structure is as follows: top is the source video frame, and bottom is the edited video frame, separated by spaces. You will evaluate the edit across two distinct criteria. For each criterion, provide a score from 1 (worst) to 5 (best). Your evaluation should focus on two key aspects:

- 1. Prompt Following (Score: 1-5) Question: Does the edit accurately and completely fulfill the instructions in the "Editing Prompt"? Scoring Guide:

- - 5: perfectly follows; all required changes are present; no unrelated changes.
- - 4: mostly follows; minor missing details or minor unrelated changes.
- - 3: partially follows; noticeable missing parts or some unrelated changes.
- - 2: poorly follows; major missing parts or many unrelated changes.
- - 1: does not follow; requested edits absent or mostly wrong.

- 2. Background Consistency (Score: 1-5) Question: Have the areas that should not have been edited remained unchanged between the "Before" and "After" images? Scoring Guide:

- - 5: unedited parts are preserved very well with minimal unintended changes.
- - 4: mostly preserved, minor unintended changes.
- - 3: partly preserved, noticeable unintended changes.
- - 2: poorly preserved, major unintended changes.
- - 1: not preserved; extensive unintended changes.

Note: No further explanation is needed, just reply with a scores. Structure the output in JSON format with:

- - Prompt Following: [Your score, 1-5]
- - Background Consistency: [Your score, 1-5]

###### Figure 10. Prompt Template for Common Video Editing Evaluation.

You are a meticulous video editing quality evaluator. Your task is to provide a detailed assessment of the video edit by comparing a reference special effects video, the original image, and the edited image based on given text prompts. Editing Prompt:"{prompt}" Instructions: Analyze the five provided video frames and evaluate the implementation of the "editing prompts." The frame structure is as follows: top is the reference special effects video frame, middle is the source video frame, and bottom is the edited video frame, separated by spaces. You will evaluate the edit across two distinct criteria. For each criterion, provide a score from 1 (worst) to 5 (best). Your evaluation should focus on two key aspects:

- 1. Effects Compliance (Score: 1–5) Question: Does the edited result accurately reflect both (a) the semantic intent of the editing prompt and (b) the visual style, intensity, and spatial

placement of the effect shown in the reference frame? Scoring Guide:

- - 5: Perfect alignment—effect matches the prompt exactly and visually replicates the reference with fidelity.
- - 4: Strong alignment—minor deviations in timing, intensity, or placement; still clearly consistent with both prompt and reference.
- - 3: Moderate alignment—core idea present, but noticeable discrepancies (e.g., wrong object affected, weaker effect, or partial mismatch with reference).
- - 2: Weak alignment—major elements missing or misapplied; effect only loosely related to prompt or reference.
- - 1: No alignment—effect absent, completely incorrect, or contradicts the prompt/reference.

- 2. Background Consistency (Score: 1–5) Question: Are non-target areas (i.e., regions not specified in the prompt) preserved identically between the source and edited frames? Scoring Guide:

- - 5: Perfect preservation—no unintended changes outside the edited region.
- - 4: Near-perfect—tiny, localized changes (e.g., minor color bleed) but background intact.
- - 3: Partial preservation—some unintended alterations (e.g., lighting shift, texture change) in background.
- - 2: Poor preservation—large background areas altered unintentionally.
- - 1: No preservation—background heavily modified or degraded.

Note: No further explanation is needed, just reply with a scores. Structure the output in JSON format with:

- -Effects Compliance: [Your score, 1-5]
- -Background Consistency: [Your score, 1-5]

###### Figure 11. Prompt Template for Video VFX Editing Evaluation.

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

Source VFX Video Data

[Figure 402]

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

[Figure 413]

[Figure 414]

[Figure 415]

Add a growth and bounce effect to the stone cross. Add a growth and bounce effect to the central building of the screen.

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

Source VFX Video Data

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

Add a white jet skateboard style line effect on the skateboard. Add a blue sun style line effect on the pink swimming circle.

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

Source VFX Video Data

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

Add a cartoon bunny effect between two women. Add a cartoon bunny effect on the left side of the yoga mat.

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

Source VFX Video Data

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

Add an anime clone effect to the right of the woman. Add an anime clone effect to the right of the woman.

###### Figure 12. Additional Video VFX Editing Results of IC-Effect.

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

Source VFX Video Data

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

Add a particle aggregation effect to hot air balloons in the sky. Add a particle aggregation effect to the man on the sofa.

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

Source VFX Video Data

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

Add a black reflective rotating effect above the cat’s head. Add a blue reflective rotating effect above the pizza.

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

Source VFX Video Data

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

Add a purple flame burning effect on the sofa. Add a flame burning effect to the dog.

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

Source VFX Video Data

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

Add a blue light particle line shuttle effect from back to front on the small path. Add a yellow light particle line shuttle effect from left to right on the wire.

###### Figure 13. Additional Video VFX Editing Results of IC-Effect.

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

Source Video Instruction

Source Video

VFX Data

Instruction

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

Video VFX Editing

Video VFX Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

Add a blue light particle line shuttle effect from back to front on the shore.

Add a dynamic line shuttle effect around the Chinese style wooden tower.

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

Source Video

Source Video

VFX Data

Instruction Instruction

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

Video VFX Editing

Video VFX Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

Add a graffiti effect from top left to bottom right on the wall behind the woman.

Add a flame burning effect on the tower in the center of the screen

[Figure 700]

[Figure 701]

[Figure 702]

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

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

###### Figure 14. Additional Qualitative Comparison for Video VFX Editing.

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

Source Video Instruction

Source Video

VFX Data

Instruction

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

Video VFX Editing

Video VFX Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

Add a dynamic line shuttle effect on the white church.

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

Add a blue flame burning effect on the stone cross.

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

Source Video

Source Video

VFX Data

Instruction Instruction

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

Video VFX Editing

Video VFX Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

Add a yellow light particle line shuttle effect from back to front on the median strip of the road.

Add a red heartshaped line effect on the white cake.

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

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

[Figure 853]

[Figure 854]

[Figure 855]

###### Figure 15. Additional Qualitative Comparison for Video VFX Editing.

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

Source Video Instruction

Source Video

VFX Data

Instruction

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

Video VFX Editing

Video VFX Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

Add a flamecolored lightning effect to the edge of the green car’s rear window.

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

Add a blue umbrella style line effect on the stone cross.

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

Source Video

Source Video

VFX Data

Instruction Instruction

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

Video VFX Editing

Video VFX Editing

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

Add a particle dispersal effect that drifts upwards to the right for women wearing hats by the seaside.

Add a "Fa Xiang Tian Di" effect above the man wearing black clothes in the middle of the photo.

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

###### Figure 16. Additional Qualitative Comparison for Video VFX Editing.

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

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

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

Add a hat to the head of the person wearing blue pajamas. Add a man with an umbrella.

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

Source

Source

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

Add a man in a black suit to the woman’s left. Add a cyclist riding along the road on the left side of the SUV.

###### Figure 17. Additional Qualitative Comparison for Common Video Editing.

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

Modify the pink shawl to a light turquoise shade. Change the color of the leftmost person’s pajamas from blue to red.

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

[Figure 1180]

[Figure 1181]

Source

Source

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

Modify the car color to metallic black with a glossy finish. Change the color of the man’s blue shirt to red.

###### Figure 18. Additional Qualitative Comparison for Common Video Editing.

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

Remove the woman on the left wearing a light blue tracksuit. Delete the brown bear walking on the rocks.

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

Source

Source

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

[Figure 1311]

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1312]

[Figure 1313]

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

[Figure 1344]

[Figure 1345]

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

[Figure 1350]

[Figure 1351]

Delete the elephant. Remove the large backpack from the man’s back.

###### Figure 19. Additional Qualitative Comparison for Common Video Editing.

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1372]

[Figure 1373]

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

[Figure 1382]

[Figure 1383]

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

Transform the video into the Pixar style. Transform it into a paper-cutting art style.

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

[Figure 1416]

[Figure 1417]

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

Source

Source

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1432]

[Figure 1433]

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

[Figure 1458]

[Figure 1459]

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

Make it Moe style anime. Draw in a pencil sketch style.

###### Figure 20. Additional Qualitative Comparison for Common Video Editing.

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

[Figure 1498]

[Figure 1499]

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

Replace the husky on the carpet with a blue shorthair cat. Replace the retro blue and white car with a black Jeep.

[Figure 1532]

[Figure 1533]

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

Source

Source

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

[Figure 1550]

[Figure 1551]

InsV2VInsViEVACELucyEditOurs

InsV2VInsViEVACELucyEditOurs

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

[Figure 1558]

[Figure 1559]

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

Replace the SUV with a vintage red convertible car. Replace the human with golden robot.

###### Figure 21. Additional Qualitative Comparison for Common Video Editing.

