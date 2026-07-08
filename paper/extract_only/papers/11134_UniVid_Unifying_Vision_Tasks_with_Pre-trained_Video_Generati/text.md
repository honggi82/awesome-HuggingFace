### UniVid: Unifying Vision Tasks with Pre-trained Video Generation Models

Lan Chen1 Yuchao Gu2 Qi Mao1,

1MIPG, Communication University of China 2Show Lab, National University of Singapore

# arXiv:2509.21760v1[cs.CV]26Sep2025

##### Abstract

Large language models, trained on extensive corpora, successfully unify diverse linguistic tasks within a single generative framework. Inspired by this, recent works like Large Vision Model (LVM) extend this paradigm to vision by organizing tasks into sequential visual sentences, where visual prompts serve as the context to guide outputs. However, such modeling requires task-specific pre-training across modalities and sources, which is costly and limits scalability to unseen tasks. Given that pre-trained video generation models inherently capture temporal sequence dependencies, we explore a more unified and scalable alternative: can a pre-trained video generation model adapt to diverse image and video tasks? To answer this, we propose UniVid, a framework that fine-tunes a video diffusion transformer to handle various vision tasks without task-specific modifications. Tasks are represented as visual sentences, where the context sequence defines both the task and the expected output modality. We evaluate the generalization of UniVid from two perspectives: (1) cross-modal inference with contexts composed of both images and videos, extending beyond LVM’s uni-modal setting; (2) cross-source tasks from natural to annotated data, without multi-source pretraining. Despite being trained solely on natural video data, UniVid generalizes well in both settings. Notably, understanding and generation tasks can easily switch by simply reversing the visual sentence order in this paradigm. These findings highlight the potential of pre-trained video generation models to serve as a scalable and unified foundation for vision modeling. Our code will be released at https://github.com/CUC-MIPG/UniVid.

##### 1. Introduction

Large language models (LLMs) such as GPT [26] and DeepSeek-R1 [6] have garnered significant attention for their ability to tackle a broad spectrum of language tasks within a unified framework. This success motivates the pursuit of developing similar unified models for various vision tasks. A representative effort in this direction is the Large

Vision Model (LVM) [1], which seeks to bridge the gap by representing diverse vision tasks—including both natural data (images, videos) and annotated data (e.g., segmentation maps)—as visual sentences in pixel space. LVM [1] processes these heterogeneous inputs using a unified sequential architecture, analogous to how language models handle natural language sequences.

Within this framework, sequential task-specific data—including images, videos, or annotations—compose the visual prompt (or visual context) that guides output generation, as illustrated in Fig. 1(a). However, realizing such unified visual modeling in practice remains challenging: LVM [1] still requires pre-training on separate, task- and modality-specific datasets (e.g., for generation vs. understanding, and for images vs. videos). This fragmented data curation process is highly labor-intensive and fundamentally constrains scalability to new tasks. These limitations motivate us to explore a more unified and scalable approach for vision modeling.

Unlike the complex data curation required for sequential modeling in LVM [1], pre-trained video generation models benefit from the inherent sequential structure of video data, allowing them to naturally capture temporal dependencies without the need for extensive task- and modality-specific annotations. Inspired by LLaVA [13], which adapts a single generative backbone pre-trained on large-scale corpora to a variety of downstream tasks via supervised fine-tuning (SFT), we formulate the central hypothesis of this work: Could a single large video-generation model, pre-trained once for synthesis, serve as a universal visual backbone that can be efficiently adapted to a broad range of vision tasks through SFT?

To investigate this hypothesis, we propose UniVid, a unified paradigm that fine-tunes a pre-trained video diffusion transformer (DiT) to address generation and pixel-level understanding vision tasks without any task-specific architectural modifications. Within this framework, both image and video tasks are naturally organized as visual sentences that align with the temporal dimension of video data. As illustrated in the right block of Fig. 1(a), each training sample is structured as A → A′ → B → B′, where the context (A,A′,B) defines the vision task and specifies the desired

UniVid

LVM[1]

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

#### …

[Figure 9]

… …

## ＋

Pre-training: Natural Videos

A A' B B'

[Figure 10]

| |
|---|
|[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]|
| |

| |
|---|
| |
|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
| |

[Figure 17]

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]|
|---|
| |

[Figure 21]

[Figure 22]

- Ⅰ
- Ⅱ
- Ⅲ
- Ⅳ

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

|[Figure 29]|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|
|---|---|---|---|---|---|

[Figure 35]

…

[Figure 36]

[Figure 37]

[Figure 38]

| |
|---|
| |
|[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]|
| |

| |
|---|
|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]|
| |
| |

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

|[Figure 50]| |[Figure 51]|[Figure 52]| | | |[Figure 53]<br><br>[Figure 54]| | | |[Figure 55]<br><br>[Figure 56]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

…

| |
|---|
| |
|[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]|
| |

| |
|---|
|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
| |
| |

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

###### Pre-training: Images,

[Figure 68]

[Figure 69]

Vision Tasks

Vision Tasks

Supervised Fine-tuning: Images, Videos and Annotations

[Figure 70]

Videos and Annotations

(a) Differences of training strategy between the LVM[1] and our UniVid

LVM[1]

|[Figure 71]<br><br>[1]| |
|---|---|

ImageVideo

Output

Visual Context

|[Figure 72]| |
|---|---|

###### InferenceStage

Visual Context Output

###### UniVid

A A' B B' A A' B B'

|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]|
|---|

|[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]|
|---|

|[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]|
|---|

|[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

[Figure 89]

[Figure 90]

[Figure 91]

Ⅰ Ⅱ

[Figure 92]

[Figure 93]

[Figure 94]

|[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]|
|---|

|[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]|
|---|

|[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]|
|---|

|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

Ⅲ Ⅳ

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Context Output Context Output

(b) Differences in supported vision tasks at inference between LVM[1] and our UniVid

Figure 1. LVM [1] vs. UniVid. (a) LVM [1] requires large-scale, modality- and source-specific paired data for pre-training to support diverse vision tasks. In contrast, UniVid explores whether a pre-trained video generation model can be efficiently adapted to a broad range of vision tasks via lightweight SFT with minimal paired data. (b) At inference, LVM [1] is limited to uni-modal visual contexts, whereas UniVid enables a unified framework that accommodates both cross-modal and cross-source vision tasks. Stacked blocks represent videos; a single block represents an image.

output modality. We evaluate the generalization capacity of UniVid from two perspectives. First, while LVM [1] is limited to uni-modal contexts (i.e., either images or videos alone) at inference, we examine whether UniVid can accommodate cross-modal contexts—where the output modality is inferred from mixed image-video inputs. Second, we investigate UniVid’s capability for cross-source tasks, such as depth estimation from natural videos to annotated data, even without the multi-source pre-training required by LVM [1]. As shown in Fig. 1(b), despite being pre-trained solely on continuous natural video data, our model adapts effectively to both cross-modal and cross-source tasks through SFT. Importantly, under this unified paradigm, the distinction between generation and understanding tasks is reduced to the ordering of elements within the visual sentence. These findings highlight the potential of pre-trained video genera-

tion models as a unified pre-training backbone for generalpurpose visual modeling.

Our contributions are summarized as follows:

- • To the best of our knowledge, we are the first to explore unified vision modeling using a pre-trained video generation model, eliminating the need for task- and modality-specific pre-training data.
- • We propose UniVid, a unified framework that leverages lightweight SFT to efficiently adapt a pre-trained video DiT to a broad range of image and video tasks, without any task-specific architectural modifications.
- • Extensive experiments show that UniVid effectively generalizes to both cross-modal and cross-source scenarios, highlighting its potential as a unified and scalable foundation for general-purpose vision modeling.

Example Query

Encoder

Context

Encoder

Wan

Wan

𝐴 𝐴′ 𝐵 𝐵′

𝐴 𝐴′ 𝐵

###### A A′ B B′

- I Video Video Video Video
- II Image Image Image Image
- III Image Image Video Video
- IV Image Video Image Video Table 1. Four Types of Visual Contexts.

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Self-Attention

Self-Attention

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

QKV Projection

QKV

Projection

##### 2. Related Works

LoRA

LoRA

| | |
|---|---|
| | |

| | |
|---|---|
| | |

BlockDiT

BlockDiT

Large Vision Models. Recent advancements in universal vision frameworks predominantly follow two paradigms: image-resembling generation [2, 30, 31] and sequential modeling [1, 24]. Image-resembling generation methods [2, 30, 31] reformulate diverse vision tasks as image inpainting problems, enabling models to make predictions through generating masked regions. Sequential modeling [1, 24] approaches draw inspiration from LLMs, treating visual data as sequences of discrete tokens and optimizing models via next-token prediction. However, these approaches heavily rely on large-scale annotated data to construct task-specific training samples, which is resourceintensive and hinders scalability. In contrast, we demonstrate that a model trained solely on continuous video data can be effectively adapted to a broad range of vision tasks.

Cross-Attention

Cross-Attention

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

QKV

QKV Projection

Projection

LoRA

LoRA

Feed Forward

Feed Forward

DiT Block

DiT Block

(a) Training Stage (b) Inference Stage

[Figure 125]

[Figure 126]

Frozen Learnable

Image or Video Condition Tokens Noisy Tokens

| |
|---|

| |
|---|

| |
|---|

Random Noise

Figure 2. The framework of UniVid.

Vision In-Context Learning. In-context learning, where models perform tasks prompted by examples, has been extensively studied in LLMs [26]. Inspired by this success, many efforts [1, 2, 4, 8, 17, 24, 25, 30, 31, 35] have extended the paradigm to vision, demonstrating its potential across a wide range of vision tasks. Early methods [1, 2, 24, 30] primarily rely on sequential models trained on large-scale annotated input-output pairs to perform vision tasks. More recently, in-context generation capability has been demonstrated in DiT-based text-to-image (T2I) models [4, 8, 17, 25, 35] for controllable image generation and manipulation. Similarly, the DiT-based video model captures temporal dependencies through full attention across frames, which we leverage to adapt the pre-trained model to a wide range of vision tasks. Video Generation Models. Recent progress in video generation has been driven by both autoregressive [7, 11, 14, 27, 32, 33] and diffusion-based [3, 10, 12, 15, 16, 20, 28, 29, 34] architectures. Autoregressive models [7, 11, 14, 27, 32, 33] compress video frames into discrete tokens and employ transformers to generate video sequences token by token. Early diffusion-based video models [3, 10, 29] extend UNet [22] architectures originally designed for T2I generation, lacking temporal consistency. Recent hybrid architectures [12, 15, 16, 20, 28, 34] adopt DiT framework with advanced attention mechanism, yielding improved generation quality and temporal coherence. In this work, we take the Wan model [28] as a strong foundation to assess the effectiveness of video generative pre-training for downstream vision tasks.

##### 3. Methodology

In this section, we first review existing vision sequential models in Section 3.1 and define the problems we explore in Section 3.2. Based on the discussion, we introduce our method, experimental setup, and results analysis in Section 3.3.

###### 3.1.Preliminaries: LimitationsofSequentialVision Models

Recent advances in sequential vision modeling, exemplified by the LVM [1], aim to unify diverse vision tasks under a single framework. However, LVM [1] faces two significant limitations:

- (a) Reliance on Annotated Pairs for Pre-training. As shown in Fig. 1(a), LVM [1] requires large-scale, task- and modality-specific annotated pairs for pre-training in order to support a wide range of downstream tasks. This data curation process is labor-intensive and fundamentally restricts scalability. In this work, we question whether annotated pairs are truly necessary at the pre-training stage.
- (b) Limited Generalization Beyond Single-Modality Contexts. As shown in Fig. 1(b), despite being pre-trained on heterogeneous data, LVM [1] is limited to tasks within imageonly or video-only contexts at inference, leaving cross-modal scenarios largely underexplored. Furthermore, LVM [1] is restricted to video extrapolation and fails to generalize to cross-source video tasks, such as semantic segmentation, despite relevant annotations in pre-training.

Image (A) Image (A') Video (B) Video (B')

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Image (A) Video (A') Image (B) Video (B')

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

- (a) Cross-modal Contexts

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Video (A) Video (A') Video (B) Video (B')

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

- (b) Cross-source Tasks

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

Video (A) Video (A') Video (B) Video (B')

[Figure 185]

Understanding: Natural Video → Salient Object Mask

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Generation: Salient Object Mask → Natural Video

(c)Unifying Understanding and Generation Tasks

- Figure 3. Main observations. The top colored row serves as a legend indicating the modality and role of each clip shown below. The following figure follows the same format. (a) The model infers the correct output modality from cross-modal contexts. (b) Despite being pre-trained solely on natural video data, it generalizes to cross-source understanding tasks. (c) Under the UniVid framework, understanding and generation tasks are unified and can be converted by reordering the visual sentence.

These limitations motivate us to seek an alternative unified vision modeling paradigm that reduces data annotation burdens and extends generalization across modalities and sources.

sources, spanning both pixel-level understanding and generation tasks. Our central question is: Can a video generation model pre-trained solely on natural videos adapt to diverse vision tasks within a unified visual sentence paradigm, even when the contexts span modalities and data sources?

###### 3.2. Problem Formulation: Unified Visual Sentence Paradigm

3.3. Video Generative Pre-training with SFT for Unified Vision Modeling

To address these challenges, we revisit the video generation model, treating it as a naturally pre-trained visual sequential learner. We reformulate a broad set of vision tasks using a unified visual sentence paradigm: given an example inputoutput pair A → A′, the model is tasked to predict B′ for a query B, with (A,A′,B) collectively defining the task context and output modality. We focus on evaluating the adaptation capacity of a pre-trained video generation model from two key perspectives:(a) Cross-modal generalization: the ability to handle tasks prompted by mixed-modal contexts (e.g., combining images and videos), as illustrated in Table 1 III and IV. (b) Cross-source generalization: the ability to perform vision tasks across heterogeneous data

We start from a video DiT model i.e., Wan [28], pre-trained exclusively on continuous natural video data, as our unified backbone. To enable adaptation to diverse vision tasks, we employ Low-Rank Adaptation (LoRA) modules for efficient SFT.

Data Structure. We leverage the temporal dimension of video by representing each input-output pair as a sequence of video clips concatenated along the time axis. As shown in Fig. 2(a), each training sample is structured as a visual sentence V = [A,A′,B,B′]. A and A′ comprise an example pair that demonstrates a reference vision task A → A′ (e.g., a source video and its scribble map). B is the query in-

Video (A) Video (A') Video (B) Video (B') Image (A) Image (A') Video (B) Video (B')

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

[Figure 253]

[Figure 254]

[Figure 255]

Image (A) Video (B) Image (A') Video (B')

[Figure 256]

[Figure 257]

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

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

- Figure 4. Performance across diverse vision tasks and context formats. We show results for scribble map transfer, motion transfer, and salient object tracking under various visual contexts. Each task is fine-tuned independently within each context configuration, demonstrating that the pre-trained video generation model adapts well across all applicable settings listed in Table 1. With a fixed example pair, outputs change with the query, reflecting context-based inference.

put, and B′ is the expected output, which should undergo the same transformation as A → A′. Each element can be either an image or a video segment, allowing unified processing of diverse contexts.

(6) by reversing them into conditional generation settings to further establish its generalization.

Results and Analysis. Based on our experimental results (see Fig. 3), we summarize the following three key observations regarding the model’s adaptability to diverse vision tasks:

Fine-Tuning and Inference. During training, V is first embedded into latent tokens. We treat the tokens corresponding to (A,A′,B) as the context C, which remains clean during training. Noise is added only to the target clip B′, producing noisy latent tokens zt, as shown in Fig. 2(a). The complete token sequence is then processed by the DiT blocks using 3D full attention [28], enabling cross-clip interactions throughout the temporal dimension. We insert LoRA modules into both Cross-Attention (CA) and Self-Attention (SA) layers for efficient adaptation. At inference, the model generates B′ conditioned on (A,A′,B), with clean context tokens guiding the generation process as illustrated in Fig. 2(b).

- Observation 1: Robust cross-modal adaptation. Al-

though pre-trained solely on continuous video data, the finetuned model effectively interprets manually composed visual sentences and flexibly handles various task contexts, including those spanning multiple modalities (see Fig. 3(a)).

- Observation 2: Effective cross-source generalization.

Despite only being exposed to natural video data during pre-training, the model adapts successfully to cross-source tasks such as predicting depth maps from natural video data (see Fig. 3(b)).

Observation 3: Unified formulation of understanding and generation tasks. As illustrated in Fig. 3(c), the model can perform both understanding and generation tasks by simply reordering the elements within a visual sentence. This demonstrates that, under the unified paradigm, these tasks are seamlessly interchangeable through sentence organization.

Experimental Protocol. To systematically assess adaptation and generalization, we collect paired data across six representative tasks, including generation tasks (1) scribble map transfer, (2) Van Gogh style transfer and (3) camera movement transfer and understanding tasks (4) depth map prediction, (5) semantic segmentation prediction, and (6) salient object tracking. Each task consists of only 20 training samples. To validate the question posed in Section 3.2, we conduct straightforward fine-tuning experiments: First, for each vision task, we fine-tune the video generation model on its applicable contexts listed in Table 1 separately. Additionally, we construct generation variants of tasks (4) and

##### 4. Experiments

In this section, we first present the implementation details in Section 4.1. In Section 4.2, we assess how well UniVid generalizes to a range of vision tasks under varied visual contexts. Next, we explore the results of mixed fine-tuning

Video (A) Video (A') Video (B) Video (B')

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

Understanding: Natural Video → Salient Object Mask

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

Generation: Salient Object Mask → Natural Video

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

Understanding: Natural Video → Depth Map

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

Generation: Depth Map → Natural Video

- Figure 5. Unified understanding and generation tasks. Our proposed UniVid allows flexible switching between understanding and generation tasks by simply reordering visual sentences.

strategies in Section 4.3 and the effects of shot number in Section 4.4. Quantitative comparisons with LVM [1] are provided in Section 4.5 to validate the effectiveness of our approach. Finally, we discuss the remaining limitations and future work in Section 4.6.

contexts I, II, and III. As shown in Fig. 4, UniVid demonstrates strong adaptability across modalities and data sources. Please refer to appendix for remaining results.

Unified Understanding and Generation. We reverse the visual sentence structure from (A → A′ → B → B′) to (A′ → A → B′ → B), converting understanding tasks into generation tasks. As illustrated in Fig. 5, the fine-tuned model generates coherent videos under both sequence orderings, highlighting the unified formulation of both understanding and generation.

###### 4.1. Implementation Details.

We employ Wan2.1-T2V-1.3B as the backbone video generation model for our experiments. The number of frames in each clip of (A,A′,B,B′) is set to 1 for image modality and 17 for video modality. We use the Wan Encoder [28] to embed the four clips separately. During fine-tuning, we set the LoRA rank to 16, the learning rate to 1 × 10−4, and the batch size to 1. Please refer to appendix for additional details.

Context-Conditioned Inference. As shown in Fig. 4, given a fixed example pair (A,A′), the output B′ changes based on query B, confirming the model’s context-conditioned reasoning capability.

###### 4.3. Mixed Fine-tuning Strategy

###### 4.2. Generalization across Tasks and Contexts

Building on the results showing that the video generation model performs well when each vision task is fine-tuned with a single visual context, as demonstrated in Fig. 3 and Fig. 4, we further explore its adaptability under joint training regimes. We consider two configurations: (1) train-

Task-level Generalization. We evaluate UniVid on the six tasks under diverse context configurations (Table 1). Specifically, the camera motion transfer task is evaluated under contexts I and IV, while the other tasks are trained under

Video (A) Video (A') Video (B) Video (B')

Video (A) Video (A') Video (B) Video (B')

ScribbleMap VanGogh

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

Image (A) Image (A') Image (B) Image (B')

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

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

DepthMapStyle Semantic

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

Image (A) Image (A')

Video (B')

Video (B)

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

[Figure 416]

[Figure 417]

Video (B)

Segmentation Salient

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

Depth Map Prediction

[Figure 428]

[Figure 429]

Video (A) Video (A')

Video (B) Video (B')

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

ObjectTrack Camera

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

Image (A) Video (A') Image (B)

Video (B')

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

Movement

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Camera Movement Transfer

(a) Performance with training across all vision contexts. (b) Performance under training across all tasks and contexts.

- Figure 6. Results of mixed fine-tuning strategy. (a) When fine-tuned on visual sequences spanning all context types, the model can dynamically infer the output modality at inference time based on the context (A, A′, B). (b) We co-train the model on all vision tasks, each covering all applicable contexts. Under this mixed setup, the model consistently performs well across all tasks and contexts, demonstrating strong generalization. Additional results are shown in the appendix.

ing each task with a mixture of visual contexts, and (2) jointly training all tasks, where each task is exposed to all applicable contexts.

to 10. Remaining tasks are measured by pixel-space RMSE. As shown in Table 2, the co-training strategy consistently achieves better performance than separate fine-tuning across all tasks.

Per-task Mixed Context Fine-tuning. For the camera motion transfer task, the training data is composed of a mixture of contexts I and IV, while the other tasks are trained under contexts I, II, and III. As shown in Fig. 6(a), when trained on mixed contexts, the model can automatically adapt to the specific context (A,A′,B) and produce consistent results in correct modality, demonstrating strong generalization within the unified training paradigm. Please refer to appendix for the results of other tasks.

###### 4.4. Impact of Shot Number

We investigate the effect of shot number by fine-tuning three separate models using 4-shot, 6-shot, and 8-shot configurations, respectively. Each model is then evaluated under all three shot settings. Specifically, the 4-shot, 6-shot, and 8-shot settings are defined as follows:

- • 4 shots: A → A′ → B → B′
- • 6 shots: A → A′ → B → B′ → C → C′
- • 8 shots: A → A′ → B → B′ → C → C′ → D → D′ In all cases, the final clip (e.g., B′, C′, or D′) is the target

Joint Multi-task Fine-tuning. We co-train the model on all six vision tasks, with each task provided 20 training examples covering all applicable contexts. Fig. 6 presents results for the six vision tasks under Context I, with the remaining three contexts provided in appendix. These results demonstrate that even with limited fine-tuning data mixed from multiple vision tasks and contexts, the model generalizes robustly across diverse vision tasks and contexts, validating its potential as a unified foundation for visual modeling. We further report quantitative results of joint multi-task finetuning, employing CLIP-T for style transfer task. For the camera movement task, we extract the first, middle, and last frames and ask GPT-4o to rate the quality on a scale from 0

to be generated. We evaluate on a classic understanding task (depth estimation) and a generation task (style transfer) on uni-image Context II, reporting results in Table 3. For depth estimation, performance tends to degrade when the number of test shots exceeds that used in fine-tuning, as the model never saw depth maps during pre-training and relies solely on fine-tuning supervision. In contrast, style transfer remains stable across shot counts since similar visual data were seen during pre-training. While longer contexts yield better results, they also increase inference time. For consistent analysis, we adopt a four-shot setting in this paper.

Training Strategy Separate Co-training Style Transfer (CLIP-T ↑) 18.33 24.01

Task

Camera Movement (GPT-4o ↑) 6.33 6.73

Scribble Map (RMSE ↓) 61.03 51.94 Depth Estimation (RMSE ↓) 74.16 2.55

Semantic Segmentation (RMSE ↓) 126.12 123.03 Salient Object Track (RMSE ↓) 33.35 30.59

###### Table 2. Quantitative comparisons between different training strategies.

Depth Estimation (RMSE ↓)

Style Transfer (CLIP-T ↑)

FT shots

Test shots

Time (s)

4 13.37 61.03 21.28 6 19.36 74.25 21.00 8 25.47 86.58 20.93

4

4 13.37 44.85 21.16 6 19.36 54.51 21.00 8 25.47 47.84 21.01

6

4 13.37 49.5 21.65 6 19.36 52.07 21.57 8 25.47 55.16 21.51

8

Table 3. Effects of the shot number.

###### 4.5. Quantitative Comparisons

To assess the effectiveness of UniVid, we compare it with LVM [1] across five tasks:

- • Van Gogh style transfer is evaluated with CLIP-T (alignment with “Van Gogh style”) and CLIP-I (consistency with reference image A′).
- • Edge map prediction on the BIPED [19] dataset, is evaluated by fixed contour threshold (ODS), per-image best threshold (OIS), and average precision (AP).
- • Semantic segmentation is conducted on the ADE20K dataset [36], measured by mean Intersection over Union (mIoU) and pixel accuracy (pAcc).
- • Depth estimation is evaluated on the NYU-v2 dataset [23], reporting the percentage of pixels within various δ thresholds, absolute relative error (AbsRel), squared relative error (SqRel), root mean square logarithmic error (RMSELog), and scale-invariant logarithmic error (SILog).
- • Surface normal estimation is also evaluated on the NYU-v2 dataset [23], assessed by Mean Angular errors (Mean) and Median Angular Errors (Med), along with accuracy under thresholds of 5◦, 11.25◦, and 30◦.

All tasks are trained on small subsets of the standard training sets (such as 65 of 175K in NYU-v2 [23]) and evaluated on the full test splits, whereas LVM [1] uses the full training split. As shown in Table 4, Table 5 and Table 6, despite being

Semantic Segmentation

Style Transfer

Edge Map Prediction

Method

CLIP-T↑ CLIP-I↓ ODS↑ OIS↑ AP↑ mIoU↑ pACC↑

LVM 16.24 0.712 0.656 0.678 0.630 1.423 23.30 Ours 19.76 0.670 0.873 0.877 0.871 8.712 53.13

###### Table 4. Comparison across style transfer, edge map prediction, and semantic segmentation.

Method δ1↑ δ3↑ δ3↑ AbsRel↓ SqRel↓ RMSElog↓ SILog↓

LVM 0.15 0.31 0.48 0.53 0.18 1.15 72.91 Ours 0.43 0.76 0.91 0.27 0.28 0.42 30.74

###### Table 5. Depth estimation performance.

Method Mean ↓ Med ↓ 5◦↑ 11.25◦↑ 30◦↑

LVM 30.76 13.73 24.70% 45.10% 65.98% Ours 29.84 13.52 25.22% 45.57% 67.53%

Table 6. Surface normal estimation performance.

trained on limited data, our method consistently outperforms LVM [1], demonstrating strong generalization capability to both visual generation and understanding tasks.

###### 4.6. Limitations and Future Work

While our study validates a promising approach to unifying vision tasks using a video generation model, certain limitations remain. The context length of Wan model [28] is limited to 81 frames per sequence, restricting the duration of each visual clip. Additionally, due to the inherent randomness of generative processes, label consistency across instance types in the segmentation task cannot be guaranteed. In the future, we plan to explore long-context video generation [5] architectures and mitigate ambiguity in understanding tasks such as segmentation.

##### 5. Conclusions

In this work, we explore a new direction for building a unified visual backbone by reusing a pre-trained video generation model for various vision tasks through lightweight supervised fine-tuning. Unlike prior approaches that rely on large-scale, task-specific data to train a visual sequential model, we simply fine-tune a pre-trained video generation model using minimal supervised data. The visual data are organized into visual sentences, where the context defines both the vision task and the expected output modality. Despite being pre-trained solely on natural, continuous videos without annotations, the fine-tuned model generalizes well to cross-modal and cross-source contexts. Under this unified paradigm, understanding and generation tasks are differentiated only by the order of visual sentences and can be seamlessly interchanged. Moreover, results show that with minimal supervision, the model can jointly adapt to diverse vision tasks and context types through single fine-tuning.

##### References

- [1] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan L Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22861– 22872, 2024. 1, 2, 3, 6, 8, 11
- [2] Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, and Alexei Efros. Visual prompting via image inpainting. Advances in Neural Information Processing Systems, 35: 25005–25017, 2022. 3
- [3] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310–7320,

2024. 3

- [4] Lan Chen, Qi Mao, Yuchao Gu, and Mike Zheng Shou. Edit transfer: Learning image editing via vision in-context relations. arXiv preprint arXiv:2503.13327, 2025. 3
- [5] Yuchao Gu, weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 8
- [6] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [7] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868,

2022. 3

- [8] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 3
- [9] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 11
- [10] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023. 3
- [11] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 3
- [12] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [13] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1

- [14] Haozhe Liu, Shikun Liu, Zijian Zhou, Mengmeng Xu, Yanping Xie, Xiao Han, Juan C Pérez, Ding Liu, Kumara Kahatapitiya, Menglin Jia, et al. Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280, 2024. 3
- [15] Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding. Vdt: General-purpose video diffusion transformers via mask modeling. arXiv preprint arXiv:2305.13311, 2023. 3
- [16] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 3
- [17] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025. 3
- [18] Jiaxu Miao, Yunchao Wei, Yu Wu, Chen Liang, Guangrui Li, and Yi Yang. Vspw: A large-scale dataset for video scene parsing in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4133–4143, 2021. 11
- [19] Xavier Soria Poma, Edgar Riba, and Angel Sappa. Dense extreme inception network: Towards a robust cnn model for edge detection. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1923– 1932, 2020. 8
- [20] Di Qiu, Zhengcong Fei, Rui Wang, Jialin Bai, Changqian Yu, Mingyuan Fan, Guibin Chen, and Xiang Wen. Skyreels-a1: Expressive portrait animation in video diffusion transformers. arXiv preprint arXiv:2502.10841, 2025. 3
- [21] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 11
- [22] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 3
- [23] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012. 8
- [24] Zeyi Sun, Ziyang Chu, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. X-prompt: Towards universal in-context image generation in auto-regressive vision language foundation models. arXiv preprint arXiv:2412.01824, 2024. 3
- [25] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024. 3
- [26] OpenAI Team. Language models are few-shot learners. Advances in neural information processing systems, 33:1877– 1901, 2020. 1, 3

- [27] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 3
- [28] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3, 4, 5, 6, 8, 11
- [29] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [30] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023. 3
- [31] Zhaoqing Wang, Xiaobo Xia, Runnan Chen, Dongdong Yu, Changhu Wang, Mingming Gong, and Tongliang Liu. Lavin-dit: Large vision diffusion transformer. arXiv preprint arXiv:2411.11505, 2024. 3
- [32] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [33] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3
- [34] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3
- [35] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027,

2025. 3

- [36] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3):302–321, 2019. 8
- [37] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Señorita-2m: A high-quality instructionbased dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025. 11

##### A. Implementation Details

Data Collection. We collect paired data across six representative tasks, including generation tasks (1) scribble map transfer, (2) Van Gogh style transfer and (3) camera movement transfer and perception tasks (4) depth map prediction, (5) semantic segmentation prediction, and (6) salient object tracking. For task (1)(4)(6), we collect the source clips (A,B) from the Señorita-2M dataset [37], and the annotated clips (A′,B′) are obtained using preprocessing tools from VACE [9] code repository1. For task (2)(3), we use the source videos from Señorita-2M dataset [37] and edit them using TokenFlow [21] and computer software CapCut2, respectively. For task (5), we source data from VSPW dataset [18].

Fine-tuning Details. We employ Wan2.1-T2V-1.3B3 as the backbone video generation model for our experiments. The model is trained for 20~40 epochs for each vision tasks, with each epoch consisting of 200 iterations. For co-training across all vision tasks and contexts, we train the model for 20 epochs, with 1,200 iterations per epoch. Training on two A800 GPUs, each epoch takes about 12 minutes for 200 iterations and roughly one hour for 1200 iterations.

Experimental Details. In fine-tuning with mixed vision contexts, we randomly choose the vision contexts of each training sample. For tasks (1)(2)(4)(5)(6), we sample vision contexts I and II each with probability p = 0.3, context III with p = 0.4. For task (3), since the transformation pertains to the temporal dimension, sampling is limited to contexts I and IV, each with probability p = 0.5. In the ablation study investigating the impact of text, we utilize prompts at multiple levels of granularity, including detailed, rough, and null texts. The prompt template is illustrated in Fig. 7. For all other experiments, we consistently use the detailed text prompt.

##### B. Comparison between LVM and video generation model

As summarized in Table 7, the training data required by LVM [1] is complex to construct, while the video generation model Wan is pretrained only on raw images and videos. Although Wan uses more total training tokens than LVM, it achieves higher visual quality by employing an 8× downsampling encoder, compared to LVM’s 16× downsampling. Additionally, Wan has fewer parameters than the released version of LVM, resulting in lower computational costs.

##### C. Additional Experimental Results

We provide additional results related to four experiments presented in the main paper. Specifically, we show the results

- 1https://github.com/ali-vilab/VACE
- 2https://www.capcut.com/
- 3https://github.com/Wan-Video/Wan2.1

Text Prompts

Detailed Text: “[clip1] is the original source video, and [clip2] is its corresponding segmentation map. [clip3] is another, different source video. In [clip4], the segmentation map transformation applied from [clip1] to [clip2] is similarly applied to [clip3].”

Rough Text: “[clip1] and [clip2] form an editing pair. Apply the same transformation observed from [clip1] to [clip2] to [clip3], and generate [clip4].”

###### Null Text: “”

Figure 7. Text Prompts at multiple levels of granularity.

LVM Wan Dataset Composition

Raw images and videos

- 1. Single images
- 2. Image sequences
- 3. Images with annotations
- 4. Image sequences with annotations

Dataset Scale

420B tokens O(1)T

tokens [28] Downsample Ratio

###### 16X 8X

Parameters 7B (released version) 1.3B

Table 7. Comparison between LVM [1] and the video generation model Wan [28].

of each vision task across all contexts in Section C.2, the performance of each task under mixed-context fine-tuning in Section C.3, the impact of text prompts in Section C.5, and the results of co-training with all tasks and contexts in Section C.4.

###### C.1. Conditional Generation Tasks

As shown in Fig. 8, the video generation model is capable of performing conditional generation tasks based on depth maps or masked videos.

###### C.2. Performance Across Different Contexts

We present the performance of each vision task across different contexts in Fig. 10. The results demonstrate that the fine-tuned video generation model effectively handles not only image and video tasks, but also cross-modal and crossdata-source tasks.

###### C.3. Performance under Mixed-Context FineTuning

As shown in Fig. 11, Fig. 12 and Fig. 13 , when fine-tuned on mixed vision contexts, the model can automatically adjust

Video (A) Video (A') Video (B) Video (B')

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

- Figure 8. Results of conditional generation tasks.

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

Image (A) Image (A') Video (B) Video (B')

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

[Figure 563]

[Figure 564]

[Figure 565]

- (a) With Detailed Text
- (b) With Rough Text
- (c) With Null Text

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

- Figure 9. Impact of texts under contexts III.

###### C.5. Impact of Texts Across Different Vision Contexts

to each vision context, demonstrating strong generalization ability.

As shown in Fig. 14, Fig. 15, Fig. 9, and Fig. 16, the model effectively learns the relationships among the four clips across different vision contexts without explicit textual guidance, demonstrating strong in-context learning capabilities in the temporal dimension.

###### C.4.Co-trainingwithAllVisionTasksandContexts

As shown in Fig. 17, when co-trained on all vision tasks under mixed contexts, the model achieves consistent performance across tasks and contexts, demonstrating robust generalization ability with limited supervision.

Video (A) Video (A') Video (B) Video (B')

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

Image (A) Image (A') Image (B) Image (B') Image (A) Image (A') Image (B) Image (B')

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

Image (A) Image (A') Video (B) Video (B')

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

[Figure 703]

[Figure 704]

[Figure 705]

Image (A) Video (A') Image (B) Video (B')

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

- Figure 10. Additional results of various vision tasks across context types.

Video (A) Video (A') Video (B) Video (B')

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

- (a) Separate Training
- (b) Mixed Training

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

- Figure 11. Performance under separate and mixed training for context I.

Image (A) Image (A') Image (B) Image (B')

Image (A) Image (A') Image (B) Image (B')

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

(b) Mixed Training

(a) Separate Training

- Figure 12. Performance under separate and mixed training for context II.

Image (A) Image (A') Video (B) Video (B')

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

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

- (a) Separate Training

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

- (b) Mixed Training

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

- Figure 13. Performance under separate and mixed training for context III.

Video (A) Video (A') Video (B) Video (B')

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

- (a) With Detailed Text

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

- (b) With Rough Text

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

- (c) With Null Text

- Figure 14. Impact of texts under contexts I.

Image (A) Image (A') Image (B) Image (B') Image (A) Image (A') Image (B) Image (B')

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

- (a) With Detailed Text
- (b) With Rough Text
- (c) With Null Text

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

- Figure 15. Impact of texts under contexts II.

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

Image (A) Video (A') Image (B) Video (B')

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

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

- (a) With Detailed Text
- (b) With Rough Text
- (c) With Null Text

- Figure 16. Impact of texts under contexts IV.

Video (A) Video (A') Video (B) Video (B')

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

Image (A) Image (A') Image (B) Image (B') Image (A) Image (A') Image (B) Image (B')

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

Image (A) Image (A') Video (B) Video (B')

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

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

Image (A) Video (A') Image (B) Video (B')

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

Figure 17. Co-training with all vision tasks and contexts.

