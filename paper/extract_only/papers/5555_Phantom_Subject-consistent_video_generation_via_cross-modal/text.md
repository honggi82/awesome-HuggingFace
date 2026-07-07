## Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment

Lijie Liu∗ Tianxiang Ma∗ Bingchuan Li∗† Zhuowei Chen∗ Jiawei Liu Gen Li Siyu Zhou Qian He Xinglong Wu

Intelligent Creation Team, ByteDance https://phantom-video.github.io/Phantom/

# arXiv:2502.11079v2[cs.CV]10Apr2025

### Abstract

Text-to-video

The continuous development of foundational models for video generation is evolving into various applications, with subject-consistent video generation still in the exploratory stage. We refer to this as Subject-to-Video, which extracts subject elements from reference images and generates subject-consistent videos following textual instructions. We believe that the essence of subject-to-video lies in balancing the dual-modal prompts of text and image, thereby deeply and simultaneously aligning both text and visual content. To this end, we propose Phantom, a unified video generation framework for both single- and multi-subject references. Building on existing text-to-video and image-to-video architectures, we redesign the joint text-image injection model and drive it to learn cross-modal alignment via text-imagevideo triplet data. The proposed method achieves highfidelity subject-consistent video generation while addressing issues of image content leakage and multi-subject confusion. Evaluation results indicate that our method outperforms other state-of-the-art closed-source commercial solutions. In particular, we emphasize subject consistency in human generation, covering existing ID-preserving video generation while offering enhanced advantages.

Prompt following

Text-modal

Alignment

Balanced

Subject-to-video

Visionconsistency

Image-modal

Alignment

Image-to-video

Figure 1. Relationship in cross-modal video generation tasks.

Sora [27, 35, 40, 59, 63], which have demonstrated promising visual effects and are paving the way for a new era in Artificial Intelligence Generated Content.

Currently, foundational video generation models focus mainly on two major tasks: text-to-video[35] and imageto-video [2]. Text-to-video (T2V) leverages language models to understand input text instructions and generate visual content describing the expected characters, movements, and backgrounds. While it allows for creative and imaginative content combinations, it often struggles with generating consistently predictable results due to inherent randomness. On the other hand, image-to-video (I2V) typically provides the first frame of an image along with optional text descriptions to transform a static image into a dynamic video. Although it is more controllable, the content richness is often limited by the strict “copy-paste” [4, 40] nature of the first frame. We term the process of subject-consistent video generation as subject-to-video (S2V) [4, 23, 34], which involves capturing the subject from an image and flexibly generating a video based on text prompts, while combining the diversity and controllability of joint image and text inputs. As shown in Figure 1, its essence lies in balancing the dual-

### 1. Introduction

The rise of diffusion models [18, 38] is rapidly reshaping the field of generative modeling at an astonishing pace. Among them, the advancements in video generation brought by diffusion models are particularly remarkable. In the visual domain, video generation requires to pay more attention to the continuity and consistency of multiple frames compared to image generation, which poses additional challenges. Inspired by the scaling laws of large language models [36, 51, 58], the focus of video generation has shifted towards investigating foundational large models, similar to

∗Equal Contributions †Project Lead

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

- Figure 2. Subject-consistent video generation examples using our method, with reference images and corresponding generated video frames (text prompts omitted). The last three rows show multiple reference subjects.

modal prompts of text and image, requiring the model to simultaneously align text instructions and image contents.

However, the research on subject consistency in video generation tasks still lags behind image generation scenarios. As text-to-image (T2I) foundation models [11, 28] have matured, subject-to-image (S2I) has evolved from parameter optimization methods [20, 44] to adapter-based training approaches [21, 60], to unified image editing approaches

- [5, 15, 57], achieving impressive results (refer to Sec 2.2). The most straightforward way to implement S2V is to combine S2I with I2V, but there are two main limitations. First, S2I has greater difficulty in learning subject consistency compared to S2V, as the S2V training data naturally include multi-view dynamic variations, allowing for better understanding of the subject. Second, transitioning from S2I to I2V can lead to information loss. For instance, when generating a back-to-front view motion, the subject’s ID information may be lost because the first frame lacks it, which hinders I2V from maintaining ID consistency (see supple-

mentary material). Therefore, subject-consistent generation requires a specialized video model for unified processing.

Specifically, the subject-consistent video generation task aims to deeply and simultaneously align the content described in the text and images. To achieve this, we propose a data pipeline for the S2V task, producing training data in the form of text-image-video triplets. Two key issues must be addressed. First, prevent the leakage of image content into the generated video. Some methods [4, 16, 23, 29, 61] sample key frames from a video as image conditions to reconstruct the video, but this allows the model to copy-paste image content, reducing text responsiveness. While some approaches enhance data through transformations [4], they fail to address the rigid properties and overall lighting of the image. We focus on constructing cross-video multisubject pairs to ensure that subjects exhibit non-rigid deformations and color variations while maintaining content matching. Second, address the issue of confusion arising from multi-subject generation. Specifically, when similar

[Figure 19]

|[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>…<br><br>Clip-1<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>…<br><br>Clip-2<br>|
|---|

caption Detection Matching

Filter

LLM VLM CLIP

Drop

video source

Scene Cut

|[Figure 28]<br><br>[Figure 29]<br><br>…<br><br>[Figure 30]<br><br>[Figure 31]<br><br>…|
|---|

|[Figure 32]<br><br>[Figure 33]<br><br>…<br><br>[Figure 34]<br><br>[Figure 35]<br><br>…|
|---|

|Clip-1: A woman in a pink shirt, a girl in a white and purple shirt<br>Clip-2: a woman in a pink shirt and purple leggings, a young girl in a light purple shirt<br>|
|---|

|Low aesthetics Low quality Low motion strength Video subtitles<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]|
|---|

sample

…

[Figure 40]

alignment

[Figure 41]

[Figure 42]

[Figure 43]

Subject detection Cross-pair data

A long video

Video clips

Subject term extraction

- Figure 3. Data processing pipeline for cross-modal video generation. The process involves filtering, adding captions, detection, and matching stages to extract subjects from video clips and align them with the text prompts, ensuring consistent video generation.

subject-to-video (S2V) task and elucidate its relationship with text-to-video (T2V) and image-to-video (I2V), as in Fig. 1; (2) Phantom offers a feasible path for the S2V task, focusing on high-quality alignment of both textual and visual content.

subjects trigger identical textual descriptions, it can lead to content ambiguity. To resolve this, we emphasized distinct descriptions of the subjects’ appearances in the video. The appearances of multiple subjects should be distinguishable and precisely match the contents of the sampled reference images. Furthermore, we build a rephraser that rephrases user’s input text prompts to include detailed description of the image content.

Technology. (1) A new data pipeline constructs crossmodal aligned triplet data, effectively addressing the issues of information leakage (copy-paste) and content confusion (multiple subjects); (2) Phantom offers a unified framework for generating videos from both single and multiple subject references, utilizing dynamic injection scheme of various conditions at its core.

Our model design is based on two primary considerations: (1) How to simply and effectively extend a video foundation model to support S2V capabilities; (2) How to achieve a unified framework for single- and multi-subject consistency generation. Thus, we redesigned the imagetext joint injection model based on pre-trained T2V and I2V models [30] to ensure effective cross-modal learning. Specifically, our method is built on the MMDiT [11] architecture. Referring to [54], full self-attention is replaced with window attention to reduce computational costs. VAE [10] and CLIP [62] are used to encode the reference images, and the encoded results are fed into the video and text branches of MMDiT, respectively. The VAE latent provides low-level detail information, while CLIP offers high-level semantic information. Additionally, we introduced a dynamic information injection strategy during attention calculation, allowing the insertion of one or more reference images without affecting the window size and position encoding [48], achieving a unified model architecture for singleand multi-subject consistent video generation.

Significance. (1) Phantom demonstrates superior generation quality, bridging the gap between academic research and proprietary commercialization; (2) the unified consistency generation paradigm covers subtasks such as ID generation and demonstrates significant advantages, indicating that Phantom-like solutions have broad prospects in scenarios such as the film industry or advertising production.

### 2. Related Work 2.1. Video foundation model

The diffusion algorithm has spurred the rise of video foundation model research, significantly impacting content creation and intelligent interaction. Early latent diffusion models (LDM) [42] typically utilized U-Net [43] architectures, such as the open-source Stable Diffusion 1.5 [42]. Temporal modules were later added to these models, evolving them into video generation models like Make-A-Video [47], SVD [2], and Animatediff [13]. The DiT [38] architecture, guided by scaling laws, has led to the development of more vision foundation models. Among these, Stable Diffusion

In addition, for the S2V task, we constructed evaluation datasets for portrait IDs, single subjects, and multiple subjects, and developed corresponding evaluation metrics. Since the performance of some open-source reproducible projects [4, 16, 23, 29, 61] has not yet matched that of closed-source commercial solutions [26, 34, 39, 40, 53], our focus is on comparing with commercial methods. Overall, our proposed Phantom has the following contributions:

##### 3 introduced MMDiT [11], a dual-stream DiT architecture, which has been adopted in open-source video generation projects such as CogvideoX [59], HunyuanVideo [27] and SeedVR [54].

Concepts. (1) We are the first to clearly define the

Video Input

+ ⊕

3D VAE Encoder

Vision Branch

Noise

Reference Images

Video Output

Vision Encoder

3D VAE Decoder

Window Self Attention

𝑡

Text Prompt

Text Branch

⊕

LLM

[Figure 44]

[Figure 45]

Triplet Data Input Head

MMDiT Block ×𝑵

- Figure 4. Overview of the Phantom architecture. Triplet data is encoded into latent space at the input head, and after combination, it is processed through modified MMDiT blocks to learn the alignment of different modalities.

#### 2.2. Subject-consistency image generation

MovieWeaver [29], and VideoAlchemist [4] have demonstrated capabilities in generating consistent multi-subject videos in general scenarios. However, there are currently no open-source methods for the S2V task, and commercial software’s S2V capabilities [26, 34, 39, 40, 53] remain state-of-the-art. Therefore, comparing the performance with commercial closed-source solutions is crucial for evaluating the superiority of the proposed method.

In recent years, significant progress has been made in subject-consistent generation for image tasks. Optimization-based methods [12, 19, 20, 44, 45] training bind image content with special identifiers for text-toimage generation. A notable work in the training and inference paradigm is IP-Adapter [60], which freezes the base model weights while only training additional adapters to achieve subject-consistent generation. This approach is also widely used in tasks requiring facial ID consistency [6, 14, 55]. However, these solutions often rely on CLIP [7] or DINO [37] for extracting image semantics, leading to a trade-off between low-level detail reconstruction and flexible text response. Recent advancements have unified image generation and editing tasks [5, 15, 33, 57], enabling various types of editing tasks within a single model, including subject-consistent generation. Compared to adapter-based approaches, this method deeply learns image-text alignment, fully leveraging foundation models and resolving degradation issues from multiple adapters.

### 3. Phantom

This section introduces the specific implementation of Phantom. The first subsection describes how to construct cross-modal alignment training data, emphasizing the creation of cross-pair text-image-video triplets to address the ”copy-paste” issue. The second subsection presents the design and considerations of the Phantom architecture, focusing on how single and multiple subject features are dynamically injected into the framework. The third subsection introduces some key training settings and inference techniques to ensure the efficient implementation of S2V capabilities.

#### 2.3. Subject-consistency video generation

#### 3.1. Data Pipeline

From recent research developments, the advancement of video generation capabilities and algorithmic innovations tends to lag behind image tasks. Similar to image consistency techniques, Kling [26] has released an optimizationbased video generation method for facial ID consistency, which requires uploading multiple videos of the same person for optimization, resulting in significant computational costs. Adapter-based approaches have also been attempted for video ID consistency tasks, such as IDAnimator [16] and ConsisID [61]. However, these works have been validated on small datasets (around 10k), which limits their ability to fully align facial information with text descriptions. Recent works like ConceptMaster [23],

To achieve subject-to-video (S2V) generation, we constructed a triplet data structure of text-image-video for cross-modal learning (Figure 3), ensuring that videos are paired with both images and text. First, we sampled long videos from Panda70M [3] and in-house sources. These videos were cut into single-scene segments using AutoShot [64] and PySceneDetect [50], and any clips with low quality, aesthetics, or motion levels were filtered out. Next, we used Gemini [49] to generate captions for the filtered video clips, focusing on describing the subjects’ appearance, behavior, and the scene. Further, the LLM [36] is utilized to analyze the caption and extract the subject words with ap-

|𝑸𝒗𝒊𝒅 𝑲𝒗𝒊𝒅 𝑽𝒗𝒊𝒅|
|---|

image embedder [46] of the improved CLIP architecture to score and pair subjects detected across different videos. Pairs with scores that are excessively high (indicating a likelihood of copy-pasting) or too low (indicating different subjects) are eliminated.

|…<br><br>Window 1 Window 2 Window N<br><br>0 1 … 8 9 10 11 12 𝑀-2 𝑀-1|
|---|

|𝑸𝒓𝒆𝒇_𝒗 𝑲𝒓𝒆𝒇_𝒗 𝑽𝒓𝒆𝒇_𝒗|
|---|

After constructing the cross-paired data pipeline, further segmentation is required based on application scenarios. These primary elements include people, animals, objects, backgrounds, and more. Additionally, interactions between multiple elements can further categorize scenarios, such as multi-person interactions, human-pet interactions, and human-object interactions. By segmenting the data sources according to these application scenarios, we can quantitatively supplement missing data types. For example, virtual try-on applications require specific collections of model images and garment layouts. Ultimately, we obtained cross-pair data on the order of one million, among which the data containing human subjects accounted for the largest proportion. In addition, we also added a portion of paired image data to increase diversity. The data sources are Subject200k [5] and OmniGen [57].

|…<br><br>single subject<br><br>| | | | |
|---|---|---|---|
<br><br>0 7 8 9 10<br><br>…<br><br>| | | | |
|---|---|---|---|
<br><br>0 7 8<br><br>| | | | |
|---|---|---|---|
<br><br>multiple subjects<br><br>𝑀-1 𝑣<br><br>𝑣 𝑣 𝑣 𝑣 𝑣 𝑣<br><br>𝑣 𝑀-2 𝑀<br><br>𝑀-2 𝑀-1<br><br>… 𝑣<br><br>…|
|---|

Dynamic injection

𝑣

or

| | |
|---|---|

𝑣 𝑣

…

+rope3d

|𝑸𝒕𝒆𝒙𝒕 𝑲𝒕𝒆𝒙𝒕 𝑽𝒕𝒆𝒙𝒕|
|---|

|𝑸𝒗𝒊𝒅 𝑲𝒗𝒊𝒅 𝑽𝒗𝒊𝒅|
|---|

|𝑸𝒓𝒆𝒇_𝒄 𝑲𝒓𝒆𝒇_𝒄 𝑽𝒓𝒆𝒇_𝒄|
|---|

Window n

Dynamic injection

… …

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

…

𝒇𝒕𝒆𝒙𝒕 𝒄𝟏 𝒄𝟐

𝒇𝒗𝒊𝒅𝒏

#### 3.2. Framework

⊕

The Phantom architecture, shown in Figure 4, consists of an untrained input head and a trained MMDiT module. The input head includes a 3D VAE [59] encoder and an LLM [58] inherited from the video foundation model [30, 54], which encode the input video and text, respectively. The vision encoder, critically, comprises both a Variational Autoencoder (VAE) [10] and CLIP [41, 62]. The image features Fref v concatenated with the video latents Fvid reuse the 3D VAE to maintain consistency in the visual branch input. Meanwhile, the image CLIP features Fref c concatenated with the text features Ftext provide high-level semantic information, compensating for the low-level features from the VAE. Feature merging involves dimensional alignment, as detailed below,

𝑙

… …

𝑐

| | | |
|---|---|---|

𝒇𝒗𝒊𝒅𝒏

𝒇𝒕𝒆𝒙𝒕 𝒇𝒓𝒆𝒇_𝒄

Window Self Attention

Output Feature

- Figure 5. Dynamic injection strategy and attention calculation for single or multiple reference subjects in each MMDiT block.

Fl

T = Fl

1+l2,c

text ⊕ Fl

1,c

2,c

ref c, (1)

pearance descriptions, which are used as prompts for the VLM [1] to obtain the subject detection boxes of the reference frames. At this point, the descriptions of the subjects in the captions can be exactly aligned with the subject elements detected in the reference images.

FVt+n,h,w,c = Fvidt,h,w,c ⊕ Frefn,h,w,cv , (2) where ⊕ denotes concatenation. The concatenated features FT and FV are fed into the visual and text branches of MMDiT, and the model only separates the injected features during the calculation of attention.

Although the reference images and text are aligned, the reference images are taken from specific frames within the videos. These image-video pairs are termed ”in-pair” data. Some existing methods [4, 23] use in-pair data to train S2V models, ensuring subject consistency between images and videos. However, high visual similarity might cause the model to disregard text prompts, resulting in generated videos that simply copy-paste the input images. To address this issue, we undertake an additional effort to further establish pairings between cross-video clips. We employ the

Specifically, the MMDiT block is based on [30, 54] and improved for reference image input, primarily modifying the Attention [52] block, as shown in Figure 5. First, the Qvid, Kvid, Vvid features calculated from Fvid are divided into windows of size 9. Then, the Qref v, Kref v, Vref v features calculated from Fref v are dynamically concatenated to the end of each window, while the in-situ features are sequentially shifted to the start of the next window. This approach maintains the window structure while ensuring inter-

[Figure 46]

[Figure 47]

Figure 6. Video quality evaluation (left) and user study results for multi-subject consistency (right).

action between video and subject features within each window, as well as adaptive input for single- or multi-subject. Meanwhile, the Qtext, Ktext, Vtext features calculated from Ftext and the Qref c, Kref c, Vref c features calculated from Fref c are dynamically concatenated. After collecting all reference information, self-attention is calculated within each window. Then, the dynamically injected reference image features (including ref v and ref c) and the text features within each window are extracted from the output features and averaged. This process ensures that the dimensions of the input and output features within the current block remain consistent, thereby facilitating subsequent block computations.

#### 3.3. Training and inference

Training setup. We employ rectified flow (RF) [31, 32] to construct the training objective and adjust the noise distribution sampling [11]. RF aims to learn an appropriate flow field, enabling the model to efficiently and high-quality generate meaningful data samples from noise. In the forward process of training, noise is added to clean data x0 to generate xt = (1 − t) · x0 + t · ϵ, where ϵ is Gaussian noise with the distribution N(0,I) and t is a randomly sampled step scaled to a value between 0 and 1 based on the total steps (T=1000). The model predicts velocity vt to regress velocity ut = dxt/dt, and vt is represented by,

vt = Gθ(xt,t,FT,FV ). (3) Thus, the RF training loss is given by,

Lmse = ∥vt − ut∥2. (4)

Notably, vt includes additional (n)-dimensional features at the tail (refer to Eq.2), which does not participate in the loss calculation. The model training is conducted in two phases:

the first phase trains for 50k iterations at 256p/480p resolution, and the second phase incorporates mixed 720p data, training for an additional 20k iterations to enhance higher resolution generation capabilities. Additionally, since one of the training objectives of VAE is pixel-level reconstruction, the CLIP features can be overshadowed when trained together with VAE features. Therefore, we set a relatively high dropout rate (0.7) for VAE during training to achieve balance. The total computational resources consumed approximately 30,000 GPU-hours on A100.

Inference settings. Phantom inference can accept 1 to 4 reference images and generate corresponding videos by describing the reference subjects using a given text prompt. Note that generating with more reference subjects may lead to unstable results. To align with the training data, the text prompt used in inference must first be adjusted by a rephraser to ensure it accurately describes the appearance and behavior of each reference subject, avoiding confusion between similar subjects (see supplementary materials). The Euler method is used for sampling over 50 steps, and the classifier-free guidance [17] separates the image and text conditions. The denoised output at each step is given by,

xt−1 =x⊘t−1 + ω1(xIt−1 − x⊘t−1)

+ ω2(xTIt−1 − xIt−1),

(5)

where x⊘t−1 is the unconditional denoising output, xIt−1 is the image-conditioned denoising output, and xTIt−1 is the joint text-image conditioned denoising output. The weights ω1 and ω2 are set to 3 and 7.5, respectively.

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

Kling1.6Vidu2.0Pika2.1Ours

Kling1.6Vidu2.0Pika2.1Ours

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

This laptop is hovering in the air and spinning.

The little white dog is bouncing and running from a distance, with the living room in the background, blinking its eyes and smiling at the camera.

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

Kling1.6Vidu2.0Pika2.1Ours

Vidu2.

0 Kling1.6Pika2.1Ours

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

A woman strolls in the park wearing a dress, with a large area of beautiful flowers in the background.

A man and a dog are playing frisbee in the park.

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

OursHailuoConsisID

HailuoOursConsisID

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

A man in a black suit, paired with a white shirt and a dark tie, is walking along a busy city street. He is carrying a steaming cup of coﬀee in one hand. The street is lined with tall buildings and bustling with people.

A little girl in a green and yellow dinosaur costume is bouncing up and down on the sofa. The living room was bright and the sunlight entered through nearby windows. A warm light emanates from the scene. The walls are painted light blue.

Figure 7. Comparison results showing, from top to bottom, single subject, multi-subject, and facial ID-consistent video generation, with four uniformly sampled frames displayed in each case.

### 4. Experiments

#### 4.1. Evaluation materials

Phantom can be fine-tuned from any video generation base model [30, 54]. The T2V and I2V pre-training stages are excluded from this evaluation. We focus on assessing the subject consistency generation capability, with additional independent evaluations for face ID-based video generation. Due to the lack of an established benchmark for subject-tovideo, we constructed a specific test set and defined evaluation metrics accordingly.

We collected 50 reference images from different scenarios, covering humans, animals, products, environments, and clothing. Each reference image is paired with 3 different text prompts. To ensure confidence in each case, each textimage pair is generated with three random seeds, resulting in a total of 450 videos. For scenarios with multiple ref-

erence images, we mixed the aforementioned reference images and rewrote the text prompts to obtain a test set of 50 groups. Additionally, considering the unique value of portrait scenarios, we collected an additional 50 portrait reference images, including both celebrities and ordinary individuals, for independent evaluation of ID consistency.

For the S2V task, the existing available state-of-theart (SOTA) methods are closed-source commercial tools. Therefore, we evaluated and compared the latest capabilities of Vidu [53], Pika [39], and Kling [26]. For the ID-preserving video generation task, the commercial tool Hailuo [34] demonstrated impressive results. We also evaluated an excellent open-source algorithm ConsisID [61].

#### 4.2. Quantitative results

We classify the S2V evaluation metrics into three major categories: video quality, text-video consistency, and subject-

Identity Consistency Prompt Following FaceSim-Arc ↑ FaceSim-Cur ↑ FaceSim-glink ↑ ViCLIP-T ↑

Methods

ConsisID 0.538 0.417 0.470 21.76 Hailuo-ID 0.542 0.504 0.557 23.31 Phantom-ID 0.581 0.529 0.590 24.12

- Table 1. Comparison of different methods based on identity consistency and prompt following

Methods

Subject Consistency Prompt Following CLIP-I ↑ DINO-I ↑ CLIP-I-Seg ↑ DINO-I-Seg ↑ ViCLIP-T ↑

Vidu2.0 0.706 0.511 0.724 0.544 22.78 Pika2.1 0.697 0.498 0.712 0.534 23.05 Kling1.6 0.732 0.554 0.715 0.569 21.62 Phantom-IP 0.714 0.523 0.731 0.538 23.41

- Table 2. Comparison of different methods based on single subject consistency and prompt following. Boldface indicates the highest scores in each column, and underline indicates the second-highest scores.

video consistency. First, the visualization of video quality is shown in the radar chart on the left side of Figure 6. We selected six metrics provided by VBench [24] for testing and supplemented them with four inner model scores such as structure breakdown score. For text-video consistency, we used ViCLIP [56] to directly calculate the cosine similarity score between the text and the video. For single subject consistency, we uniformly sampled 10 frames from each video and calculated the CLIP [7] and DINO [37] feature Direction Scores with the reference image. Additionally, we used grounded-sam to segment the subject part of the video and calculate the CLIP and DINO scores (excluding scene graphs). For ID consistency, we used three facial recognition models to measure similarity [8, 22].

The video quality evaluation results, shown on the left side of Figure 6, indicate that Phantom performs slightly worse [24], while excelling in other metrics. As shown in Table 1 and 2, Phantom leads in overall metrics for subject consistency (Identity Consistency) and prompt following. For multi-subject video generation, due to high error rates in automated subject detection and matching, we conducted a user study. We surveyed 20 users, who rated the methods on a scale of 1 to 3 (1: unusable, 2: usable, 3: satisfactory). The evaluation results, displayed in the bar chart on the right side of Figure 6, show that Phantom ’s multi-subject performance is comparable to commercial solutions, with some advantages in subject consistency.

#### 4.3. Qualitative results

We present the comparison results of several typical cases in Figure 7. Each generated video is displayed with four evenly sampled frames, including the first and last frames. The first two rows of Figure 7 respectively show the results of generating single- and multiple subject consistency. It can be seen that Vidu [53] and Phantom exhibit balanced

Subject Consistency Prompt Following Video Quality CLIP-I ↑ DINO-I ↑ ViCLIP-T ↑ Aes score ↑ Clarity score ↑

Methods

w/o CLIP 0.693 0.519 23.63 62.03 71.40 w/o VAE 0.512 0.302 22.79 48.82 70.76 w/ All 0.714 0.523 23.40 64.32 71.72

Table 3. The ablation experiment results of VAE and CLIP.

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Reference w/o VAE w/o CLIP w/ All

Figure 8. Qualitatively display the ablation of VAE and CLIP.

performance in subject consistency, visual effect, and text response. Pika [39] performs poorly in subject consistency. Kling [26] has a notable issue: some cases exhibit characteristics analogous to I2V approaches. For instance, the first frame of character videos almost matches the input reference image, leading to low success rates in virtual tryon scenarios. Additionally, the laptop case shows that the compared methods tend to cause deformations in rigid body movements. The last row of Figure 7 shows the results of video generation for facial ID preservation. The opensource method ConsisID [61] tends to exhibit motion blur, and has weak text response. Hailuo [34] excels in visual aesthetics, but there is some loss in facial similarity. Our results are balanced across all dimensions, with particular advantage in ID consistency. More qualitative analyses are presented in supplementary materials.

#### 4.4. Ablation study

Selection of visual encoder. Due to differences in training methods, CLIP aligns image-text pairs and tends to extract semantic information, while VAE aims for lossless reconstruction and focuses on detailed information. As shown in Figure 8, faces generated using only CLIP features are smoother and more refined but show decreased similarity. In contrast, faces generated using VAE features are sharper but may amplify undesirable details, making them contain more artifacts. In general object scenarios, CLIP is inadequate at reproducing details like text and patterns, thus primarily serving to supplement VAE’s high-level information. Quantitative results in Table 3 show that combining VAE and CLIP features is more advantageous. Additional ablation studies are given in supplementary materials.

### 5. Conclusion

We propose Phantom, a method for subject-consistent video generation that achieves cross-modal alignment through text-image-video triplet learning. By redesigning the joint text-image injection mechanism and leveraging dynamic feature integration, Phantom demonstrates competitive performance in unified single/multi-subject generation and facial ID preservation tasks, outperforming commercial solutions in quantitative evaluations.

Acknowledgments We would like to express our gratitude to the Bytedance-Seed team for their support. Special thanks to Haoyuan Guo, Zhibei Ma, Sen Wang and Lu Jiang for their assistance with the model and data. In addition, we are also very grateful to Siying Chen, Qingyang Li, and Wei Han for their help with the evaluation.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5, 13
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 3
- [3] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 4
- [4] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. arXiv preprint arXiv:2501.06187, 2025. 1, 2, 3, 4, 5
- [5] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024. 2, 4, 5
- [6] Zhuowei Chen, Shancheng Fang, Wei Liu, Qian He, Mengqi Huang, and Zhendong Mao. Dreamidentity: Enhanced editability for efficient face-identity preserved image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1281–1289, 2024. 4
- [7] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scal-

- ing laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2829, 2023. 4, 8
- [8] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019. 8, 13
- [9] dreamina. Generate image and video capabilities. https: //dreamina.capcut.com/, 2024. 12
- [10] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 3, 5
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 6
- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 4
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [14] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024. 4, 12
- [15] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. Ace: Allround creator and editor following instructions via diffusion transformer. arXiv preprint arXiv:2410.00086, 2024. 2, 4
- [16] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024. 2, 3, 4
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 4
- [20] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 2, 4
- [21] Mengqi Huang, Zhendong Mao, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom: Narrowing real text

- word for real-time open-domain text-to-image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7476–7485, 2024. 2, 12
- [22] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2020. 8
- [23] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025. 1, 2, 3, 4, 5
- [24] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024. 8
- [25] Tanuj Jain, Christopher Lennan, Zubin John, and Dat Tran. Imagededup. https://github.com/idealo/ imagededup, 2019. 13
- [26] keling. Image to video elements feature. https: //klingai.com/image-to-video/multi-id/ new/, 2024. 3, 4, 7, 8, 14
- [27] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 3
- [28] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2
- [29] Feng Liang, Haoyu Ma, Zecheng He, Tingbo Hou, Ji Hou, Kunpeng Li, Xiaoliang Dai, Felix Juefei-Xu, Samaneh Azadi, Animesh Sinha, et al. Movie weaver: Tuning-free multi-concept video personalization with anchored prompts. arXiv preprint arXiv:2502.07802, 2025. 2, 3, 4
- [30] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316,

2025. 3, 5, 7

- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 6
- [32] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 6
- [33] Pengqi Lu. Qwen2vl-flux: Unifying image and text guidance for controllable image generation, 2024. 4
- [34] MiniMax. Hailuo s2v-01. https://www.minimaxi. com/en/news/s2v-01-release/, 2024. 1, 3, 4, 7, 8
- [35] OpenAI. Sora. https://openai.com/, 2023. Accessed: February 10, 20245. 1
- [36] OpenAI. Chatgpt (gpt-4 version). https://chat. openai.com/, 2024. 1, 4, 13

- [37] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4, 8
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 3

- [39] Pika. Pikascenes. https : / / pika . art / ingredients/, 2024. 3, 4, 7, 8, 14
- [40] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 1, 3, 4

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 3
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 4
- [45] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2024. 4
- [46] Shihao Shao and Qinghua Cui. 1st place solution in google universal images embedding. arXiv preprint arXiv:2210.08473, 2022. 5
- [47] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 3

- [48] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [49] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,

- Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 4
- [50] PySceneDetect Development Team. Pyscenedetect: An open-source video scene detection tool. https://www. scenedetect.com/, 2024. 4
- [51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1
- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 5
- [53] Vidu. Reference to video. https://www.vidu.com/,

2024. 3, 4, 7, 8, 14

- [54] Jianyi Wang, Zhijie Lin, Meng Wei, Yang Zhao, Ceyuan Yang, Chen Change Loy, and Lu Jiang. Seedvr: Seeding infinity in diffusion transformer towards generic video restoration. arXiv preprint arXiv:2501.01320, 2025. 3, 5, 7
- [55] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 4
- [56] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 8
- [57] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 2, 4, 5
- [58] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 1, 5
- [59] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 3, 5
- [60] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 4

- [61] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024. 2, 3, 4, 7, 8
- [62] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 3, 5

- [63] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 1
- [64] Wentao Zhu, Yufang Huang, Xiufeng Xie, Wenxian Liu, Jincan Deng, Debing Zhang, Zhangyang Wang, and Ji Liu. Autoshot: A short video dataset and state-of-the-art shot boundary detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2238– 2247, 2023. 4

### Supplementary Materials

#### 1. S2I+I2V vs S2V

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Reference image

Subject-to-image Image-to-video

[Figure 156]

[Figure 157]

[Figure 158]

Subject-to-video

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Reference image

Subject-to-image Image-to-video

[Figure 163]

[Figure 164]

[Figure 165]

Subject-to-video

Figure 9. Comparison of subject-to-image-to-video [9] and subject-to-video (ours).

As mentioned in the main text, combining subject-toimage (S2I) and image-to-video (I2V) can achieve similar effects to subject-to-video (S2V), but there are some difficult limitations. Firstly, existing methods [9, 14, 21] for generating subject-consistent images or ID-consistent images still exhibit noticeable artificial artifacts, and there is significant room for improvement in the dimension of subject consistency. Equally important, I2V cannot ensure consistency of the subject during motion. As illustrated in Figure 9, when inputting a reference portrait, S2I first generates a reference image for the initial frame of I2V. If the initial frame includes a back view or occlusions, I2V may ”imagine” a false ID during the process of removing the occlusion, leading to a failure in maintaining consistency.

#### 2. Copy-paste problem

In the field of video generation, the copy-paste issue is particularly prominent, manifesting as the leakage of image content into the generated video. Some methods sample keyframes from a video and use them as image conditions to reconstruct the video. However, this approach allows the model to employ shortcut learning strategies, simplifying the content understanding process. Figure 10 shows examples of the copy-paste issue, sampling from the initial, middle, and final frames: In the first row, the girl’s expression

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Reference image

A girl is wearing a school uniform. with a solemn look on her face, she lowers her head.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Reference image

This man is angrily blew a building.

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Reference image The man is standing on the podium, waving his fist and shouting passionately. He is full of

fervor with an exaggerated expression.

Figure 10. Intuitive cases of copy-paste problems. The red font in the text prompt does not function as intended.

remains unchanged, ignoring the text prompt. In the second row, the cartoon character’s movements remain stiff and identical to the reference. The third row illustrates a common case where the generated video is too similar to I2V, diminishing the effectiveness of scene-related text and reducing content diversity. To address this, we focus on constructing cross-video multi-subject pairings, ensuring subjects match in content while allowing for non-rigid deformations and changes in color distribution, thereby avoiding the copy-paste problem.

#### 3. Ablation study supplement

Multi-subject confusion issue. When multiple reference subjects are input simultaneously, appearance confusion may occur. Our solution aligns text descriptions with video subjects during training, ensuring distinct descriptions for each subject. During inference, a rephraser adjusts the input text prompts to align with the training data format. For example, in the first row of Figure 12, the original prompt ”A family of three is having a meal at the table” caused confusion. The rephrased prompt ”a woman in black, a young girl in white, and an elderly man in a suit eating together at the table” resolved this issue. In the second row of Figure 12, the original prompt ”a girl in casual clothes walking by the beach” failed to match the reference. The rephrased prompt ”a girl in a white T-shirt and jeans walking by the beach”

w/o text-image alignment w/ text-image alignment Success rate 65% 95%

Table 4. Success rate of multi-subject generation with and without text-image alignment.

[Figure 178]

[Figure 179]

Figure 11. Distribution of object frequencies and class.

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

Multi-subjects Confused cases Successful cases

- Figure 12. Examples of multi-subject confusion: On the left are the multi-subject reference images, while the right columns present the cases of confusion and the successful cases after improvement.

successfully matched the reference. Quantitative analysis, shown in Table 4, indicates a significant increase in the success rate of subject-consistent generation with this method. Aligning image and text is crucial for multi-subject generation tasks. This approach, which requires no additional complex data structures or model designs, significantly optimizes the multi-subject confusion problem.

4. Data pipeline for face ID

|[Figure 190]<br><br>[Figure 191]<br><br>…<br><br>[Figure 192]<br><br>…<br><br>[Figure 193]<br><br>|
|---|

caption

video source

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

…

Detection Matching

Face detection

ArcFace

sample

|[Figure 199]<br><br>…<br><br>[Figure 200]<br><br>…<br><br>[Figure 201]<br><br>[Figure 202]|
|---|

A long video Subject detection Cross-pair data

Scene Cut Filter

Dedup

- Figure 13. Facial data processing pipeline for constructing ID cross-pair

To enhance facial ID consistency, we developed an additional data pipeline for processing facial data. As shown in Figure 13, the facial data pipeline reuses the scene segmentation, video filtering, and annotation steps from the general subject pipeline. During the detection stage, we use an internal facial detection tool to identify each face in the video reference frames and calibrate it with the VLM [1] results from the captions using IOU (Intersection Over Union). In the matching stage, we calculate facial similarity using Arcface [8] features and add a deduplication operator [25] to further calibrate the recognition results.

#### 5. Data distribution

Distribution of video object quantities. We sample three frames at [0.05, 0.5, 0.95] of the video timeline and perform object detection on these frames. We filter out objects that meet the following criteria: (1) objects that are small in size or occupy a small proportion of the frame; (2) objects with a high degree of overlap with other objects; and (3) incomplete objects judged by the VLM [1]. The final distribution of the number of objects per video is shown in the table on the left side of Figure 11.

Distribution of video object types. We use LLM [36] to classify the noun fields in all captions into the following categories: human, animal, clothes, product, landmark, IP character, and others. The distribution is shown in the accompanying Figure 11, with human, clothes, and product categories accounting for the majority.

#### 6. Model architecture

The architecture of the Phantom model is shown in Figure 14, which supplements the missing details in the main text. As illustrated, it integrates the VAE and CLIP encoders to process reference images, while the text encoder handles captions. The encoded features are combined with added noise and processed through multiple MMDiT blocks, resulting in the final output. This design ensures a bal-

first man’s clothes and the second man’s face, and includes a person unrelated to the reference images. Pika misses one person, and Kling also lacks one person and shows the same issue as Vidu. The final case demonstrates that Vidu and Pika appear more realistic, indicating that their text responsiveness is stronger than their subject consistency.

Reference images

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Caption

Padding

⊕

#### 8. Limitations and future work

|CLIP Encoder|
|---|

|VAE Encoder|
|---|

Text Encoder

Limitations. While Phantom demonstrates strong performance in subject-consistent video generation, several challenges persist. First, handling uncommon subjects (e.g., rare animals or niche objects) remains difficult due to biases in training data coverage. Second, complex multi-subject interactions (e.g., overlapping movements or fine-grained spatial relationships) often lead to partial confusion or inconsistent relative subject sizes. Third, generating videos that strictly adhere to intricate text responses (e.g., precise spatial layouts or nuanced temporal dynamics) is limited by the current cross-modal alignment mechanism. These issues stem from three core factors: (1) gaps in dataset diversity, particularly for non-human-centric scenarios; (2) the inherent rigidity of the reference image injection strategy, which struggles to disentangle entangled features from multiple subjects; and (3) biases inherited from pre-trained base models and visual encoders, such as CLIP’s semantic oversimplification and VAE’s over-referenced details.

⊕

Add noise x

y +

c

MMDiT- Block 1

Timestep embedding

MMDiT- Block 2

...

MMDiT- Block n

Output

- Figure 14. The supplementary diagram of the Phantom framework.

Future work. Addressing these limitations will require multi-faceted innovations, and we propose the following directions:

- • Enhanced Cross-Modal Alignment: Develop adaptive injection mechanisms that dynamically prioritize text or image conditions based on task requirements, reducing content leakage and improving text responsiveness.
- • Spatiotemporal Disentanglement: Integrate spatial-aware attention modules and physics-inspired motion priors to better model multi-subject interactions and enforce consistent relative scales.
- • Bias-Aware Training: Mitigate dataset and model biases through adversarial debiasing techniques and synthetic data augmentation for underrepresented subjects.
- • Granular Control: Explore auxiliary control signals (e.g., depth maps, segmentation masks) to complement text prompts, enabling precise alignment with complex instructions.
- • Foundation Model Adaptation: Fine-tune pre-trained encoders on domain-specific data (e.g., medical imaging, animation) to broaden Phantom’s applicability while preserving generalization.

ance between detailed reconstruction and high-level information alignment while also guaranteeing a unified training paradigm for single and multiple subject inputs.

#### 7. Qualitative analysis

Qualitative comparison results of single-subject consistency generation are shown in Figure 15. Firstly, Vidu [53] performs well in both image consistency and text following for the first two cases but fails in the third shoe case with two different seeds. The effectiveness of Pika [39] is evident, as the first two cases show significant disadvantages in maintaining subject consistency, tending towards a cartoonish appearance. The major issue with Kling [26] is that most cases resemble the I2V mode, where the initial frame directly replicates the reference image (as indicated by the red box in Figure 15), followed by subject motion generated based on text, thereby limiting the effectiveness of textual descriptions.

Figure 16 displays some qualitative comparisons of multi-subject consistency generation. Firstly, Kling still reflects the I2V pattern, appearing unnatural transitions in the first few frames of the video. Additionally, in the second example with three reference images of persons, confusion issues are evident in all methods except ours. Vidu shows the

By advancing these areas, Phantom could evolve into a versatile tool for industrial applications such as virtual tryons, interactive storytelling, and educational content creation, ultimately narrowing the gap between academic research and real-world demands.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Vidu2.0Pika2.1Keling1.6Ours

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

The sky is drizzling, and a woman is walking in a retro alley, reaching out to catch the rain. The raindrops are clearly visible.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Vidu2.0Vidu2.0Pika2.1Pika2.1Keling1.6Keling1.6OursOurs

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

The rabbit bounces on the soft mattress and falls from the bed to the wooden floor.

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

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

###### The man puts on the sneakers and starts jogging along the riverbank.

- Figure 15. Comparative results of single reference subject-to-video generation.

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Vidu2.0Pika2.1Keling1.6Ours

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

On the stage, they both shook hands and hugged.

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Vidu2.0Pika2.1Keling1.6Ours

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

| |
|---|

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

They were fiercely discussing the solution to a math problem in front of the blackboard, pointing and pointing at the blackboard.

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Vidu2.0Pika2.1Keling1.6Ours

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

| |
|---|

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

A character walking on the moon.

- Figure 16. Comparative results of multi-reference subject-to-video generation.

