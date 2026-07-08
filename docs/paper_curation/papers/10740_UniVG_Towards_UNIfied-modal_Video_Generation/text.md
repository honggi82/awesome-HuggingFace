# arXiv:2401.09084v1[cs.CV]17Jan2024

## UNIVG: TOWARDS UNIFIED-MODAL VIDEO GENERATION

Ludan Ruan, Lei Tian, Chuanwei Huang, Xu Zhang, Xinyan Xiao Baidu Inc. Beijing, China {ruanludan, tianlei09}@baidu.com huangcw21@gmail.com {zhangxu44, xiaoxinyan}@baidu.com

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

A cat is eating carrot.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

A cat is eating carrot.

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

A girl is floating with fish around.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

A girl is floating with fish around.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Figure 1: UniVG is a unified video generation framework that supports various video generation tasks, such as Text-toVideo, Image-to-Video, and Text&Image-to-Video. Here displays two sets of examples. Row 1: Input text to generate semantically consistent videos; Row 2: Input image to produce pixel-aligned videos; Row 3: Combine the semantic of input text and image to create semantically aligned videos. All videos are shown on https://univg-baidu.github.

io.

#### ABSTRACT

Diffusion based video generation has received extensive attention and achieved considerable success within both the academic and industrial communities. However, current efforts are mainly concentrated on single-objective or single-task video generation, such as generation driven by text, by image, or by a combination of text and image. This cannot fully meet the needs of real-world application scenarios, as users are likely to input images and text conditions in a flexible manner, either individually or in combination. To address this, we propose a Unified-modal Video Genearation system that is capable of handling multiple video generation tasks across text and image modalities. To this end, we revisit the various video generation tasks within our system from the perspective of generative

freedom, and classify them into high-freedom and low-freedom video generation categories. For high-freedom video generation, we employ Multi-condition Cross Attention to generate videos that align with the semantics of the input images or text. For low-freedom video generation, we introduce Biased Gaussian Noise to replace the pure random Gaussian Noise, which helps to better preserve the content of the input conditions. Our method achieves the lowest Fréchet Video Distance (FVD) on the public academic benchmark MSR-VTT, surpasses the current open-source methods in human evaluations, and is on par with the current close-source method Gen2. For more samples, visit https://univg-baidu.github.io.

#### 1 Introduction

In recent years, diffusion-based generative models [1, 2, 3] have significant progress in image generation [4, 5, 6, 7, 8, 9] with applications rapidly expanding to video generation [10, 11, 12, 13, 14]. The majority of video generation models employ textual descriptions as conditional inputs [15, 16, 17, 18, 10, 11, 19]. However, recent studies have begun to explore the use of image conditions to improve the detail of generated videos [20] or for pixel-level controlling [21, 22, 13, 12, 23, 24]. Additionally, to enhance the temporal smoothness and spatial resolution of the generated videos, current approaches often incorporate modules for frame interpolation and super-resolution [10, 11]. However, existing works focus exclusively on single-objective or single-task video generation, where the input is limited to text [10, 11, 9, 18], an image [13], or a combination of text and image [12, 24]. This single-objective or single-task pipeline lacks the necessary flexibility to satisfy all user needs. In practice, users may not have the requisite text or image conditions for input, rendering the model unusable. Alternatively, the introduction of conflicting text-image pairs may lead to the generation of static videos or videos with abrupt transitions (similar conclusion is proposed in [24]).

In essence, all models used in video generation are conditional generative models that accept one or more conditions to produce a corresponding video. These conditions can be text, images, low-resolution videos, even control signals. In order to construct a versatile video generation system capable of handling multiple video generation tasks, we revisit existing methods and categorize the relevant methods based on generative freedom rather than the task itself. The concept of generative freedom that we propose corresponds to the range of solution space for video generation models given certain conditions. In this paper, we categorize various video generation tasks as either high-freedom or low-freedom video generation. Specifically, high-freedom video generation is characterized by input conditions, i.e., text and image, that are weakly constrained at the semantic level, so that the generative model in this scenario has a larger solution space, providing a higher degree of freedom. Conversely, low-freedom video generation typically involves strongly constrained conditions at the low-level information (i.e., pixel), such as in image animation and video super-resolution. These constraints limit the solution space available to the generative model, resulting in a lower degree of freedom.

In order to better match the characteristics of various video generation tasks, different strategies with varying degrees of generative freedom should be taken for video generation. For high-freedom video generation, the standard diffusion Generation Paradigm is appropriate and has been extensively utilized in existing research some refs should be provided @ludan. Specifically, during training stage, the diffusion model learns the added noise in the forward processing, and predicts the target distribution by reversing from a purely random Gaussian distribution during inference stage. Classifier guidance [4] and classifier free guidance [25] are employed to align the predicted distribution with the one specified by the input conditions. For low-freedom video generation, the Editing Paradigm is more suitable. Taking image editing [26] as a case in point, a prevalent practice involves adding noise to the original image up to a certain level and then using text as the editing signal to steer the distribution toward the intended outcome. This approach, compared to generation from scratch, offers better retention of the original input’s content. Video super-resolution has utilized a similar technique to that of image editing [23]. However, the Editing Paradigm has a limitation in the form of a discrepancy between training stage and inference one. Specifically, the model is trained solely to approximate the target distribution without learning the transition from the conditional distribution to the target distribution. This discrepancy results in a trade-off-related issue, i.e., the less noise that is introduced, the weaker the model’s ability to edit, whereas the more noise that is added, the less capable the model is of preserving the input. In extreme cases, when the noise level approaches that of a completely random Gaussian distribution, editing paradigm becomes analogous to generation one, significantly diminishing the model’s capability to preserve the content of the original input. How to reconcile the training and inference stages of editing models to balance their editing capabilities while preserving the input is also a problem that needs to be addressed but has been overlooked in previous work.

In this paper, we propose a unified system Unified-modal Video Generation (i.e.UniVG), designed to support flexible video generation conditioned on the arbitrary combination of image and text. To achieve this, we categorize all models within the system into two groups: high-freedom video generation and low-freedom video generation. For high-freedom video generation, we present a base model that is capable of the requirements of handling arbitrary combinations

of text and image conditions. We accomplish this by enhancing the original cross-attention module of the UNet architecture with a multi-condition cross-attention module. With regard to low-freedom video generation, we propose two corresponding models that are individually tailored for image animation and video super-resolution task. These models utilize the editing paradigm, as opposed to the generation paradigm. To reconcile the differences between the training process based on generation paradigm and the inference process based on editing one, in this paper, we predict Biased Gaussian Noise (shorted as BGN) that is directed towards the target distribution, instead of standard Gaussian noise, by refining the objective function during training stage.

The proposed UniVG system comprises a Base model, an Image Animation model and a Super Resolution model. The Base model is capable of handling arbitrary combinations of text and image conditions and outputs a video sequences of 24 × 320 × 576 that are semantically aligned with the input conditions at 8 frames per second (fps). The Image Animation model that fine-tuned from the Base model with the additional condition of image concatenation, generates video frames of 24 × 320 × 576 that are pixel-aligned with the input image. The Super Resolution model enhances the resolution of each frame to 720 × 1280 pixels. Compared to previous works, Our UniVG demonstrates better tasks adaptability for video generation, i.e., handling various video generation tasks within an unified system, but also significantly improvements on the generation details and frame consistency. Experiments have proven the effectiveness of our method. On objective metrics, our method significantly surpasses other existing methods, and in manual evaluations, our approach is on par with Gen2 and exceeds the other methods.

Our contributions can be summarized as follows:

- 1. We propose UniVG, the first video generation system that is capable of handling multiple video generation tasks, such as semantically aligned text/image-to-video generation, image animation.
- 2. We introduce Biased Gaussian Noise and confirm its effectiveness for low-freedom video generation tasks, such as image animation and super-resolution.
- 3. Experiments demonstrate that our method surpasses existing text/image-to-video generation methods in terms of objective metrics and is on par with Gen2 in subjective evaluations.

#### 2 Related Work

##### 2.1 Text-to-Video Generation

Early works on Text-to-Video generation utilized GANs [27, 28, 29], VQ-VAEs [30, 31], auto-regressive models [30, 18], or transformer structure [32], but were limited by low resolution and suboptimal visual quality. Following the success of diffusion models in image generation [4, 5, 6, 7], audio generation [33, 34, 35], and other domains [36, 37, 38], VideoDiffusion [39] marked the first application of diffusion models in video generation. Subsequently, Make-AVideo [10] and ImagenVideo [11] expanded video generation into the open domain by extending the 2D U-Net from text-to-image generation to 3D U-Nets. Until then, researchers had been studying video modeling in the pixel space, which requires massive GPU memory consumption and high training costs. To address this issue, many researchers shifted their focus to conducting the diffusion process in the latent space instead of pixel space [8, 15, 40, 16], and to improving the sampling efficiency by learning-free sampling [2, 41, 42, 43] or learning-based sampling [44, 45]. Additionally, some work has concentrated on reducing the training cost to that of a single video [46] or to no training cost at all [47].

##### 2.2 Image-to-Video Generation

Generating video directly from text is a challenging task with high complexity. A natural thought is to use images as an intermediate bridge. Similar to Text-to-Video generation, early works on video prediction used non-diffusion methods [48, 49, 50], which were often limited in low resolutions or specific domains. With the significant advancements in diffusion-based methods in Text-to-Video tasks, I2VGen-XL [23] is, to our knowledge, the first to utilize diffusion for open-domain Image-to-Video generation. It replaces the textual CLIP features with image CLIP features within the text-to-video framework, achieving video generation semantically aligned with the input image. Similarly, SVD [13] also fine-tunes from a text-to-video model to an image-to-video model but further concatenates the image’s VAE features as a stronger controlling signal. Concurrently, videogen [21], VideoCrafter1 [20], EMU Video [12] and Make Pixels Dance [24] remain their objective of text-to-video generation, but they introduce Text-to-Image synthesis as an intermediate step. The generated images are incorporated into the video generation framework either through concatenation or by CLIP features.

As can be inferred from the above, although text-to-video generation and image-to-video generation serve different applications, they share many similarities in their technical approaches. Therefore, this paper explores whether a

(a) Pipeline of UNIMO-VG

[Figure 35]

[Figure 36]

[Figure 37]

A dog is swimming

[Figure 38]

[Figure 39]

CrossAttn CrossAttn

TextImage

Linear

[Figure 40]

[Figure 41]

###### CLIP

…

[Figure 42]

[Figure 43]

𝐹

……

CLIP

ℱ

……

### ℱ

[Figure 44]

[Figure 45]

𝐹

[Figure 46]

𝐹

[Figure 47]

[Figure 48]

Spatial Layer Temporal Layer Multi-condition Cross Attention

[Figure 49]

[Figure 50]

ℱ

[Figure 51]

𝐹

……

……

(b) Multi-condition Cross Attention

concatenate

Figure 2: Overview of the proposed UniVG system. (a) displays the whole pipeline of UniVG, which includes the Base Model FB, the Animation model FA, and the Super Resolution model FSR. (b) illustrates the Multi-condition Cross Attention involved in FB and FA.

single framework can unify these two objectives. The primary distinction of our UniVG from earlier works is that we differentiate various models included in video generation from the perspective of generative freedom rather than task.

#### 3 Method

This section presents our proposed Unified-modal Video Generation (i.e. UniVG) for flexibly conditional video generation. Before diving into specific designs, we first briefly recap the preliminary knowledge of diffusion models in Sec 3.1. We then illustrate the overview of the whole system UniVG in Sec 3.2, the Multi-condition Cross Attention (i.e. MCA) used for high-freedom generation in Sec 3.3, and the Biased Guassian Noise (i.e. BGN) used for low-free generation in Sec 3.4.

##### 3.1 Preliminaries

Diffusion Models [1] are a class of generative models that are trained to generate the samples by iteratively denoising from Gaussian noise. During training, timestep t(0 < t ≤ N) determined noise is added at the original input x to get noisy input xt = √αtx0 + √1 − αtϵ (α refers to noise schedule and ϵ refers to the noise that sampled from standard Gaussian distribution N(0,I)), the model is trained to predict the added noise by either ϵ-prediction [1] or v-prediction [45]. During inference, samples are generated from pure noise xN ∼ N(0,I) by iteratively denoising. Furthermore, Conditional Diffusion Models [4, 25] introduce extra conditional signals to bias the predicted distribution by xt = pθ(xt+1) + wc(pθ(xt+1,c) − pθ(xt+1)), where θ defines the diffusion model, c defines input condition, and wc defines guidance scale of control intensity. Another mainstream adopted diffusion models are Latent Diffusion Models (LDM) [8], which consists of a Variational AutoEncoder (VAE) [51] and a latent diffusion model that denoising in latent hidden space. This approach reduces the complexity of fitting distributions at high resolution. In this paper, each single model of UniVG is a Conditional Latent Diffusion Model. That is, the video V consists of F RGB frames is first compressed into latent space X ∈ RF×C×H×W with an image auto encoder, then input into UNet with one or multiple conditions (text condition T, image condition I, and low resolution video V lr).

##### 3.2 UniVG

As illustrated in Figure 2-(a), our entire UniVG consists of three models: (1) A Base model FB accepts any combination of text and image conditions for high-freedom video generation. (2) An Image Animation FA model accepts text-image pairs to generated video aligned with input image in pixel level, and (3) a Super-resolution model FSR for improving spatial resolution. Each model is a latent diffusion model with 3D UNet architecture composed of Spatial Layers, Temporal Layers, and Cross Attention Layers. Following previous works [10, 13], the Spatial Layer consists of 2D Convolution layer and spatial transformers, while the Temporal Layer consists of 1D temporal Convolution layer and temporal transformers. The cross attention module is used to process semantic control signals, such as text and image feature.

(1) For the Base Model FB, we employ an image encoder that matches the text encoder of CLIP [52] inspired by VideoCrafter1 [20]. To fully utilize the global semantics and local details of input image and text, we utilize all

(a) Original Forward & Backward Process (b) Biased Forward & Backward Process

𝑡

𝑞(𝑣 |𝑣 ,𝑡)

𝑞(𝑣 |𝑣 ,𝑡)

𝑞(𝑣 |𝑣 ,𝑣 ,𝑡)

𝑡

𝑞(𝑣 |𝑣 ,𝑡)

Forward process Ideal Backward process Standard Gaussian Distribution Condition Distribution Target Distribution

Actual Backward process

Figure 3: The forward & backward diffusion process with Random Gaussian Noise and Biased Gaussian Noise.

KI visual tokens FI = {fiI}K

i=0 and all KT text tokens FT = {fiT}K

i=0 from the last layer of CLIP ViT. To enable the model with the ability of processing more than one semantic features, we extend the original Cross Attention to Multi-condition Cross Attention and introduce its mechanism in Sec 3.3. (2) In order to further generate videos that aligned with the input image at the pixel level, we train the Image Animation model FA by finetuning FB and concatenating the hidden space feature of the first frame as an additional condition. Because of the additional condition, the corresponding channel dimension of the initial convolution layer’s kernel changes from C to 2C. We initialize the extra parameters to zero to preserve the performance of the original model. Using either FB or FA, we can obtain video frames of 24 × 320 × 576. (3) To upscale the clarity of the generated videos, we further finetune a Super-Resolution model FSR from FB. Since super-resolution tasks have no image condition, the multi-condition cross attention module reverts to a regular cross-attention module that only accepts the text condition. During training, FSR accepts videos of low resolution V lr, which are obtained by destroying high-definition videos through RamdomBlur, RandomResize, JPEG Compression and so on. As we classify the tasks corresponding to FA, and FSR as low-freedom generation, we present the Biased forward and backward processes from conditional distribution to target distribution by adjusting the standard Gaussian Noise to Biased Gaussian Noise (BGN that is introduced in Sec 3.4).

I

T

##### 3.3 Multi-condition Cross Attention

Since our base model FB and Image Animation model FA accept text and image CLIP features, we use Multi-condition Cross Attention instead of the standard Cross Attention. This module’s architecture mainly follows VideoCrafter [20],

which computes Fout by:

QinKT⊺

QinKI⊺

Fout = Softmax

d · VT + Softmax

√

√

d · VI Qin = WQ · Fin, KT = WK

T · FT, VT = WV

T · FT, KI = WK

I · FI, VI = WV

I · FI

where dk is the dimensionality of the key/query vectors and Qin is shared between FI and FT. The weights WK

and WV

I

, respectively. Unlike VideoCrafter1 that treats image as an additional input enhancement, we regard the image as an equally significant control signal along with the text. This is achieved by applying a certain proportion of image dropout throughout the training process. By extension, MCA can accommodate more than two conditions by increasing the number of cross-attention units, without the necessity for retraining (e.g. stronger text features). This flexibility greatly reduces the cost of extending the model’s training to handle new conditions.

are initialized from WK

and WV

I

T

T

##### 3.4 Biased Gaussian Noise

Our proposed Biased Gaussian Noise is used to transfer condition distribution to target distribution for low-freedom video generation. As illustrated in Figure 3-(a), the standard forward diffusion process transitions from the target distribution vT to the standard Gaussian distribution ϵ via vtT = √αtvT + √1 − αtϵ. However, typically in the backward process, these are the only two distributions involved. This can result in suboptimal editing outcomes when the samples are introduced from a condition distribution vC during inference. To account for the condition distribution in both forward and backward processes, we segment the original diffusion into three parts, as illustrated in Figure 3-(b). For timestep between 0 to tm, vt is calculated by the target sample with q(vt|vT,t) = √αtv0T +√1 − αtϵ(0 ≤ t < tm) that followed the original forward process. For timestep between tn to N, vt is calculated by the condition sample with q(vt|vC,t) = √αtvC + √1 − αtϵ(tn ≤ t < N). The core problem is how to design q(vt|vC,vT,t) that can smoothly transition from vt

. To preserve the original diffusion schedule, we introduce a variable for the noise ϵ, denoted as ϵ′. Assume that for timesteps between tm and tn, we have q(vt|vC,vT,t) = √αtvT + √1 − αtϵ′, which meets the

to vt

m

n

v0T + √1 − αt

v0C + √1 − αt

= √αt

= √αt

conditions vt

ϵ. Thus, the corresponding ϵ′ should satisfy the following formulas at timestep tm and tn.

ϵ and vt

m

m

m

n

n

n

√αt √1 − αnt

ϵ′t

= ϵ, ϵ′t

× vC − vT

= ϵ +

m

n

n

In theory, there are an infinite number of solutions to ϵ′. In this paper, we simply define ϵ′ as a linear transformation following

√αt √1 − αt ×

t − tm tn − tm × vC − vT , (tm ≤ t < tn)

ϵ′t = ϵ +

The ϵ′ is sampled from a Biased Gaussian distribution, with its mean value shifted by a weighted combination of vC and vT. This bias is crucial to bridging the diffusion process from the condition distribution to the target distribution. Alternative solutions for ϵ′ will be explored in our future work.

#### 4 Experiments

##### 4.1 Implementation Details

Dataset Our training datasets include publicly available academic datasets such as WebVid-10M [53] and LAIONCOCO [54], along with self-collected data. WebVid-10M is a large and diverse text-video dataset containing approximately 10 million open-domain videos with a resolution of 336 × 596 pixels. LAION-COCO is a substantial text-image dataset comprising 600 million high-quality images, filtered from LAION-2B and scored using the Aesthetic and Semantic Estimate (ASE). To further enhance the quality of the generated videos and to address the issue of watermarks present in WebVid-10M, we continue training on our own curated datasets of videos and images, which contain high-quality visual content. We prepare the self-collected videos by first proportionally compressing them to 720p resolution along their shorter edge and then segmenting them into 10-second clips. This process yielded 5 million high-quality text-video pairs. Additionally, our self-curated image dataset includes 1.3 million high-quality text-image pairs, with a focus on artistic styles.

Training Our FB is trained with an image:video:video frame ratio of 1:1:1, where the training video frames were sampled with equal probability from 8 to 24 frames. We set the text dropout to 0.5 and the image dropout to 0.1. In addition, we utilize offset noise [55] with a strength of 0.1 and zero terminal Signal-to-Noise Ratio (SNR) [12]. Offset noise has been proven helpful to be helpful in generating extremely dark or bright images. Zero terminal-SNR has been shown to be beneficial for generating high-quality and high-resolution visual content by adding noise to pure Gaussian noise following a rescaled schedule. Both techniques have proven useful in our experiments. Subsequently, we continue finetuning FB to obtain FA and FSR, using Biased Gaussian Noise (BGN) on our self-curated video dataset only. For FA, we set the text dropout to 0.1 and the image dropout to 0.1, the BGN is experimentally set during timesteps tm = 600 to tn = 990 since the earlier steps determine the content [26]. For FSR, the text dropout is set to 0.1, and the BGN is applied during timesteps tm = 0 to tn = 700 since the later steps deciding the details [26]. We incorporate ϵ−prediction [1] for FB and FA, v−prediction for FSR. The learning rate of all models is fixed at 1 × 10−5. We use DPM Solver [56] for accelerating sampling: 50 steps for FB and FA, and 7 steps for FSR since we set initial weight to 0.7.

Evaluation We use both objective metrics and human evaluation as the assessment criteria for UniVG. In terms of objective metrics, we follow the previous work [21, 24] to use the test set of MSR-VTT [57] as the standard benchmark. This test set comprises 2,990 test videos, each corresponding to 20 prompts, totaling 59,800 prompts. For efficiency in our ablation study, we randomly selected one prompt for each test video, ultimately obtaining 2,990 prompts as the evaluation set. We calculate the CLIPSIM [30] between the generated videos and the corresponding text, and FVD [58] between the generated videos and the original videos as comparative metrics. Since some studies [12] have pointed out that objective metrics may not always align with human perception, we primarily employ human evaluation. Specifically, we adopt the categorization of video generation metrics from EMU video [12], which includes Visual Quality (including Visual Quality consists of pixel sharpness and recognizable objects/scenes), Motion Quality (including frame consistency, motion smoothness and amount of motion), Text Faithfulness (Includes textspatial alignment and text-temporal alignment). Since UniVG supports conditional generation for any combination of text and image, we further introduce Image Faithfulness (Includes text-spatial alignment and text-temporal alignment) to measure the alignment performance of generated videos with given images. Evaluators also provide their Overall Likeness of the two videos being compared, serving as a complement to the aforementioned sub-indicators. The prompts used for human evaluation were collected from the webpages of previous work [10, 11, 21, 24, 16], totaling

131 in number. To simplify the annotation process, annotators need only select G (our method is better), S (equally good), or B (other methods are better) for each indicator. To ensure fairness, the videos being compared are randomized during the annotation process. Our six annotators provide a total of 6 × 131 (prompts) ×10 (sub-metrics) = 7,860 evaluation results.

##### 4.2 Comparison with SOTA

- Table 1: Zero-shot performance comparison on MSR-VTT. T refers that the input condition contains text and I refers to image. UniVG-HG refers to high-freedom generation within our UniVG, UniVG-LG refers to low-freedom generation within our UniVG. Best in bold

|Method<br><br>|Input CLIPSIM↑ FVD↓|
|---|---|
|CogVideo(En) [18] MagicVideo [59] LVDM [19] Video-LDM [15] InternVid [60] Modelscope [17] Make-a-Video [10] Latent-Shift [61] VideoFactory [9]|T 0.2631 1294 T - 1290 T 0.2381 742<br><br>T 0.2929 T 0.2951 T 0.2939 550<br><br>T 0.3049 T 0.2773 T 0.3005 -<br><br>|
|PixelDance [24] Videogen [21]|T+I 0.3125 381<br><br>T+I 0.3127 -<br><br>|
|UniVG-HG UniVG-HG UniVG-LG<br><br>|T 0.3014 336 T+I 0.3136 331 T+I 0.3140 291|

88.99

68.40

43.34

52.98

38.36

7.24

16.79

25.18

21.37

20.55

3.77

14.81

31.48

25.65

41.09

I2VGen VideoCrafter1 SVD Pika beta Gen2

Ours Preferred No Preference Other Preferred

Figure 4: Percentage(%) of Overall Preference of UniVGLG generated videos compared with other SOTA methods.

Automatic Metrics Due to the previous work involving both plain text-to-video and image-to-video generations, we adopt aligned settings to conduct a fair comparison with them. For text-to-video generation, we use only text as the input condition to generate videos(FB+ FSR). For image-to-video generation, we start by creating images from prompts using SDXL1.0+refiner [62] and then proceed with both high-free generation (FB+ FSR) and low-free generation (FA+ FSR) using UniVG for the combination of text and images. Since the i3d model [63] used for testing FVD can only accept 16 frames, we random sample 16 frames from our generated 24 frame and the test videos in MSR-VTT. The results are shown in Table 1. Whether utilizing only text as the input condition or using both text and image together, our method generates videos that outperform those created by other methods under the same settings. Even when using only text as the condition, the videos generated by our method surpass in the FVD metric those generated by others that use both text and image. This demonstrates the capability of UniVG to concurrently process text and image conditions and generate high-quality videos with both good visual content and text alignment.

- Table 2: The winning rate (%) of UniVG-LG compared to other methods in human evaluations across 10 subdimensions(The abbreviations include VQ: Visual Quality, MQ: Motion Quality, TF: Text Faithfulness, IF:Image Faithfulness, OL: Overall Likeness, PS: Pixel Sharpness, RO/S: Recognizable Objects/Scenes, FC: Frame Consistency, Motion Smoothness, AM: Amount of Motion, TSA: Text-Spatial Alignment, TTA: Text-Temporal Alignment, ISA: Image-Spatial Alignment, ITA:Image-Temporal Alignment)

|Method<br><br>|resolution<br><br>|VQ PS RO/S<br><br>|MQ MS FC AM<br><br>|TF TSA TTA|IF ISA ITA<br><br>|OL|
|---|---|---|---|---|---|---|
| | | | | | | |
|I2VGen-XL [23] VideoCrafter1 [20] SVD [13] Pika beta [64] Gen2 [65]<br><br>|32 × 720 × 1280 16 × 576 × 1024 25 × 576 × 1024 72 × 576 × 1024 96 × 1536 × 2816<br><br>|98.79 72.85 73.74 11.45 28.11 4.41 55.11 2.44 -34.86 -2.19|87.63 63.20 -11.61 80.61 20.92 -12.52 79.06 12.59 -41.43 16.34 9.62 8.09<br><br>-3.72 1.75 -14.64<br><br>|26.24 27.30<br><br>-3.66 -3.05<br>-0.44 -4.39 3.76 6.26<br>-1.09 4.04<br>|97.13 73.76 92.82 54.35<br><br>-14.79 -1.76<br><br>0.92 2.14<br><br>-14.54 3.17<br><br><br>|85.22 53.59 11.86 27.33 -2.73|

Human Evaluation 1 Due to the fact that automatic metrics are not able to fully reflect an individual’s subjective perception of video quality, we further conduct human evaluations. Since many projects are close sourced, in this paper, we chose to compare with accessible works, including open-source works I2VGen-XL [23], VideoCrafter1 [20],

1Done in December 18th. The compared I2VGen is the version released in September.

[Figure 52]

[Figure 53]

###### Input Image:

###### Input Image:

Input 𝒘 : 𝑤 = 9.5, 𝑤 = 5.5

Input text: A cat is eating carrots under the sea.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

𝑤 = 9.5 𝑤 = 0.0

A cat is drinking beer

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

𝑤 = 9.5 𝑤 = 5.5

A cat is yawning

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

𝑤 = 0.0 𝑤 = 5.5

A cat is reading a book at the snow mountain

(a) Video Generation with different 𝑤 (b) Video Generation with different text prompts

Figure 6: The generation cases of FB with different classifier free guidance scale of text wT and wI and different text prompts.

SVD [13], and closed-source works Pika beta [64] and Gen2 [65] that we can obtain the results from website or discord. All of these are recent works and represent the current best level in text/image-to-video generation. For a fair comparison, except for SVD and Pika beta which only support image input, all other works were kept consistent in terms of text and image inputs (The images are generated from text prompt by SDXL1.0 and refiner). The comparison results are shown in Figure 4 and Table 2. Figure 4 shows a comparison of Overall Likeness between videos generated by our model (FA + FSR) and those produced by other methods. We find that the videos generated by our method outperform open-source Text/Image-to-video models and the closed-source method Pika beta, and are on par with the closed-source method Gen2. Table 2 records the winning rates for other sub-metrics. The formula for calculating the winning rate from GSB is (G − B)/(G + S + B). The number>0 indicates our method is better, and the number<0 indicates the other method is better. We found that the prominent advantage of our method lies in its FC, which is due to our adoption of an editing paradigm for low-freedom video generation, benefiting FA in producing more stable videos. Additionally, our generated videos exhibit superior PS compared to videos of similar resolution (except for gen2 that generates videos of much larger resolution). This is because we employ BGN, ensuring consistency between training and inference by directly predicting high-resolution videos from low-resolution ones. One significant drawback of our generated videos is the AM, due to our current lack of filtering for static videos in the training data. Addressing this will be part of our future work.

##### 4.3 Ablation Studies

636.03

text-to-video

659.22

image-to-video

600

text+image-to-video

521.28

FVD

504.04

468.92 477.55

500

443.81

| |
|---|

| |
|---|

| |
|---|

| |
|---|

400

| |
|---|

160k 200k 240k 270k 280k

iter

Figure 5: FVD Scores on MSR-VTT during the Training Process of FB.

Table 3: FVD scores on MSR-VTT of FA and FSR that w/ or w/o BGN

|model|BGN<br><br>|FVD↓|
|---|---|---|
|FA FA|w/o BGN w/ BGN<br><br>|393.53 369.27|
|FSR FSR|w/o BGN w/ BGN<br><br>|648.68 491.32|

Training Process of Base Model As our base model FB emphasizes the conditional video generation with arbitrary combinations of text and images, a core question is whether the base model can maintain capabilities in text-to-video, image-to-video, and text/image-to-video generation simultaneously. Therefore, we take the checkpoints from the training process of FB and test their performance in text-to-video, image-to-video, and text&image-to-video generation with FVD. The results are shown in Figure 5, where the overall trends of three curves are downward, indicating that

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Input low resolution

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

w/o BGN, Init step = 700

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

w/o BGN, Init step = 900

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

w/ BGN, Init step = 700

Figure 7: The generation cases of FSR w/o or w/ BGN.

the training process enhances the base model’s ability to generate videos from text or images. This proves that for high-freedom video generation, multi-condition video generation can be integrated into one single model.

Biased Gaussian Noise To demonstrate that Biased Gaussian Noise (BGN) better suits low-freedom video generation tasks, we conducted ablation studies on the Animation Model FA and the Video Super Resolution model FSR. The results, shown in Table 3, indicate that BGN enhances video quality in both Image Animation and Super Resolution, as evidenced by lower FVDs. It proves more beneficial for Super Resolution, a task with less freedom than Image Animation. Figure 7 visualizes FSR’s performance with and without BGN. The first row shows the original, lowresolution input video. Rows 2 and 3 depict the outputs from FSR without BGN, processed from the upscaled low-resolution input and subjected to 700 and 900 denoising steps, respectively. The fourth row presents the output from FSR using BGN at timestep tm = 700 to tn = 0, illustrating how a low-resolution video upscaled to high-resolution can be denoised effectively after 700 steps. Each row’s far right offers a magnified view to better showcase the detail in the model-generated content. Our observations indicate that absent BGN, a smaller initial noise step count results in less clarity (second row), while a larger count produces a clear yet inconsistent output due to noise overpowering the original content (third row). With BGN, the model directly predicts high-resolution videos from low-resolution inputs, achieving clarity and preserving original features (fourth row). We also acknowledge that BGN’s application can extend to other low-freedom video generation tasks, such as frame interpolation and video editing, which we aim to explore in future work.

Text&Image Conditions Since our system is capable of generating videos that align both image and text flexibly, we explore the videos generated under different inference weights for these two conditions. Given text prompt T and image condition I, the inference formula we use is Vout = FB(∅) + wT(FB(T) − FB(∅)) + wI(FB(I) − FB(∅)). We adjust the classifier free guidance scale of text wT and image wI, the generating videos are shown in Figure 6-(a), we find that adjusting the wT and wI can bias the generated video towards the text or image conditions. Figure 6-a shows that in row 1, wI = 0, FB generates a video that is almost unrelated to the input image, while in row 3, wT = 0, FB produces a video that is almost unrelated to the text. By adjusting both wT and wI to appropriate values, the second row’s generated video retains the characteristics of the input image and is also aligned with the textual semantics. Based on this feature, our FB can achieve different video generation with the same input image combined with different text prompts, as shown in Figure 6-(b). We have also explored whether FA possesses similar properties. However, due to the concatenated image features having much more stronger constraints than text, the generated videos mainly rely on image semantics. Nevertheless, inputting consistent text helps to enhance the dynamic effects of the generated videos.

#### 5 Conclusion

In this paper, we propose the UniVG system, designed for multi-task conditional video generation that leverages both text and images. We propose a novel categorization of models within our system based on generative freedom, distinguishing between high-freedom and low-freedom video generation. The high-freedom component of UniVG features a base model capable of modulating the influence of text and images to produce videos under varying semantic conditions. For low-freedom video generation, UniVG includes an Image Animation model and a Super Resolution model, which generate videos closely pixel-aligned with the input. In low-freedom generation, we propose Biased Gaussian Noise to replace the standard random Gaussian noise, facilitating a more direct connection between the conditional and the target distributions. Our experiments show that our system outperforms existing methods in objective assessments and matches Gen2 in subjective evaluations.

#### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [2] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.
- [3] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In NeurIPS, 2020.
- [4] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021.
- [5] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022.
- [6] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [7] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.
- [8] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [9] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023.
- [10] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023.
- [11] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [12] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.
- [13] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [15] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.
- [16] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In CVPR, 2023.
- [17] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-tovideo technical report. arXiv preprint arXiv:2308.06571, 2023.

- [18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2022.
- [19] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022.
- [20] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [21] Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, and Jingdong Wang. Videogen: A reference-guided latent diffusion approach for high definition text-to-video generation. arXiv preprint arXiv:2309.00398, 2023.
- [22] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023.
- [23] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.
- [24] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: High-dynamic video generation. arXiv preprint arXiv:2311.10982, 2023.
- [25] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. More control for free! image synthesis with semantic diffusion guidance. In WACV, 2023.
- [26] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022.
- [27] Yitong Li, Martin Renqiang Min, Dinghan Shen, David E. Carlson, and Lawrence Carin. Video generation from text. In AAAI, 2017.
- [28] Yingwei Pan, Zhaofan Qiu, Ting Yao, Houqiang Li, and Tao Mei. To create what you tell: Generating videos from captions. In ACM MM, 2017.
- [29] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. In ICLR, 2022.
- [30] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.
- [31] Gaurav Mittal, Tanya Marwah, and Vineeth N. Balasubramanian. Sync-draw: Automatic video generation using deep recurrent attentive architectures. In ACM MM, 2017.
- [32] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic VQGAN and time-sensitive transformer. In ECCV, 2022.
- [33] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In ICLR, 2021.
- [34] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In ICLR, 2021.
- [35] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail A. Kudinov. Grad-tts: A diffusion probabilistic model for text-to-speech. In ICML.
- [36] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M. Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. Nerfdiff: Single-image view synthesis with nerf-guided distillation from 3d-aware diffusion. In ICML, 2023.
- [37] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In CVPR, 2021.
- [38] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In CVPR, 2023.
- [39] Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In NeurIPS, 2022.
- [40] Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion models. In NeurIPS, 2022.

- [41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.
- [42] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In ICLR, 2022.
- [43] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In ICLR, 2023.
- [44] Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In ICLR, 2023.
- [45] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.
- [46] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022.
- [47] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.
- [48] Ruslan Rakhimov, Denis Volkhonskiy, Alexey Artemov, Denis Zorin, and Evgeny Burnaev. Latent video transformer. In Giovanni Maria Farinella, Petia Radeva, José Braz, and Kadi Bouatouch, editors, VISIGRAPP, 2021.
- [49] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using VQ-VAE and transformers. arXiv preprint arXiv:2104.10157, 2021.
- [50] Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, and Sergey Tulyakov. A good image generator is what you need for high-resolution video synthesis. In ICLR, 2021.
- [51] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.
- [52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 2021.
- [53] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021.
- [54] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [55] Nicholas Guttenberg. Diffusion with offset noise, 1 2023.
- [56] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022.
- [57] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016.
- [58] Christoph Feichtenhofer, Haoqi Fan, Bo Xiong, Ross B. Girshick, and Kaiming He. A large-scale study on unsupervised spatiotemporal representation learning. In CVPR, 2021.
- [59] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.
- [60] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.
- [61] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023.
- [62] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [63] João Carreira and Andrew Zisserman. Quo vadis, action recognition? A new model and the kinetics dataset. In CVPR, 2017.
- [64] Pika labs. Accessed December 18, 2023. [Online]. Available: https://www.pika.art/.
- [65] Gen-2. Accessed December 18, 2023. [Online]. Available: https://research.runwayml.com/gen2.

