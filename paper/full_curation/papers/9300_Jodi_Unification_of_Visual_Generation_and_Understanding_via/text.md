# arXiv:2505.19084v1[cs.CV]25May2025

## Jodi: Unification of Visual Generation and Understanding via Joint Modeling

Yifeng Xu1,2, Zhenliang He1, Meina Kan1,2, Shiguang Shan1,2, Xilin Chen1,2 1State Key Lab of AI Safety, Institute of Computing Technology, CAS, China 2University of Chinese Academy of Sciences, China yifeng.xu@vipl.ict.ac.cn, {hezhenliang,kanmeina,sgshan,xlchen}@ict.ac.cn

(a) Joint Generation: 𝒑(𝐱, 𝐲𝟏, 𝐲𝟐, ⋯) (b) Controllable Generation: 𝒑(𝐱 | 𝐲𝟏, 𝐲𝟐, ⋯)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

“ Three dark, bumpy avocados ”

“ A rusted humanoid robot in a rainy alley ”

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(c) Image Perception: 𝒑( 𝐲𝟏, 𝐲𝟐, ⋯| 𝐱)

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Figure 1: Our Jodi framework is capable of performing (a) joint generation, (b) controllable generation, and (c) image perception in a unified diffusion model. More visual results can be found in the appendix.

#### Abstract

Visual generation and understanding are two deeply interconnected aspects of human intelligence, yet they have been traditionally treated as separate tasks in machine learning. In this paper, we propose Jodi, a diffusion framework that unifies visual generation and understanding by jointly modeling the image domain and multiple label domains. Specifically, Jodi is built upon a linear diffusion transformer along with a role switch mechanism, which enables it to perform three particular types of tasks: (1) joint generation, where the model simultaneously generates images and multiple labels; (2) controllable generation, where images are generated conditioned on any combination of labels; and (3) image perception, where multiple labels can be predicted at once from a given image. Furthermore, we present the Joint-1.6M dataset, which contains 200,000 high-quality images collected from public sources, automatic labels for 7 visual domains, and LLM-generated captions. Extensive experiments demonstrate that Jodi excels in both generation and understanding tasks and exhibits strong extensibility to a wider range of visual domains. Code is available at https://github.com/VIPL-GENUN/Jodi.

#### 1 Introduction

Visual generation [47, 32, 43, 23, 89, 26, 82, 83, 40, 84, 64, 65, 2] and understanding [55, 52, 81, 86, 38, 31, 11, 76, 37, 48] have long been regarded as two separate research fields, each addressed by specialized models. However, from the perspective of human cognition [24, 51, 12, 27], a profound understanding of a visual scene/object is fundamental to its creation; conversely, the process of creating that scene/object can further enhance and refine our understanding of it. In other words, generation and understanding are two sides of the same coin and deeply interdependent. Therefore, exploring the unification of visual generation and understanding within a single foundation model, analogous to the human brain, might be a promising avenue toward human-level artificial intelligence.

Theoretically, generation and understanding can be associated through the joint distribution. Let x denote the image domain and y denote the label domain, generation tasks are typically formulated as learning p(x) for unconditional generation and p(x|y) for conditional generation, whereas understanding tasks are commonly represented as p(y |x). It is a theoretical fact that, once we have the joint distribution p(x,y), we can derive any of the corresponding marginal distributions p(x) and p(y), as well as the conditional distributions p(x|y) and p(y |x)1. This implies that the joint distribution inherently encodes the interdependence between generation tasks and understanding tasks. Inspired, an intriguing idea arises: Is it possible to achieve the unification of visual generation and understanding by jointly modeling the image domain and the label domain?

In this paper, we propose Jodi (Joint Diffusion), a diffusion model that jointly learns the distributions over the image domain x and multiple label domains y1,y2,..., including depth, normal, albedo, edge, line art, segmentation, and human skeleton. During the training process, each domain has the chance to serve as one of three roles: as a generation target, as a condition input, or to be ignored. As a result, our unified model simultaneously learns three types of probability distributions, including:

- 1) p(x,y1,y2,···), joint generation, where the model simultaneously generates both the image and the corresponding labels of different domains; 2) p(x|y1,y2,···), controllable generation, where the images are generated conditioned on any combination of the label domains; 3) p(y1,y2,··· |x), image perception, where the model accepts an input image and predicts multiple labels at once. In a word, the proposed model is capable of performing both image generation and understanding, as shown in Figure 1.

To effectively capture the correspondence and model the consistency among different visual domains, we employ the powerful attention mechanism [91, 71]. However, as the number of domains increases, the computational burden of full attention grows quadratically in terms of both time and space, making the training inefficient or even infeasible. To address this issue, we adopt the linear diffusion transformer [44, 99] and design a masked variant to accommodate our role switch mechanism, which achieves linear time and space complexities relative to the number of domains. To further enhance the inter-domain consistency, we introduce domain-invariant positional embeddings to provide an explicit cue for the spatial alignment between visual domains. As a result, our framework is capable of modeling as many as 8 visual domains simultaneously with high consistency.

Our contributions are summarized below:

- 1. Inspired by the theoretical fact that the joint distribution intrinsically connects generation and understanding, we propose to jointly model the image domain and multiple label domains, achieving a unification of visual generation and understanding. As a result, our framework is capable of joint generation, controllable generation, and image perception in a unified diffusion model. We believe that the proposed paradigm is a positive and promising attempt toward unified and general artificial intelligence.
- 2. Our model effectively and efficiently captures complex inter-domain relationships through the linear attention, and achieves high consistency across different domains by using the proposed domain-invariant positional embeddings.
- 3. Comprehensive experiments demonstrate the superiority of our model in various image generation and understanding tasks, despite using significantly less data and computational resources. Our model also supports novel applications not supported by previous unified models, such as joint generation of images and labels, multi-conditional generation, and performing multiple understanding tasks simultaneously. 1p(x) = p(x, y) dy, p(x | y) = p(x, y) / p(y), p(y | x) = p(x, y) / p(x)

#### 2 Related Work

Diffusion Models for Image Generation Diffusion models [82, 83, 40, 84, 64, 65, 2] have made remarkable progress in image generation, with large-scale text-to-image (T2I) models [75, 78, 5, 57, 14, 25, 7, 99] excelling in generating both photorealistic and imaginative scenes. To enhance the controllability, conditional diffusion models [60, 111, 68, 87, 113] introduce spatial conditions to enable more fine-grained control over the generated images. Subsequent methods [115, 73, 104] further improve the efficiency by unifying different types of conditions within a single model. Moreover, several studies incorporate reference images [77, 107, 53, 34, 79, 16] or face identities [61, 94, 97, 19, 33] as controlling conditions, broadening the application scope of diffusion models.

Diffusion Models for Image Understanding In addition to generation tasks, diffusion models have also exhibited superior performance in image understanding tasks, such as geometry estimation [46, 29, 56, 106, 109, 102, 36], segmentation [103, 116, 72, 119], and edge detection [108]. These methods either use pretrained diffusion models as feature extractors or reformulate the prediction objectives with diffusion frameworks. Furthermore, a recent work called Diception [114] unifies a wide range of image understanding tasks into a single diffusion model, demonstrating the capability of diffusion models in complicated image understanding.

Diffusion Models for General Purposes Recent efforts [63, 98, 54, 17, 28] have developed generalist diffusion models to handle various tasks of both image generation and understanding within a single model. Typically, these methods achieve general capabilities by training diffusion models on large-scale datasets that span diverse visual tasks. However, these methods do not investigate the relationships among different tasks, and each task requires a separate inference process. In contrast, our work emphasizes and models the correspondence and consistency among various visual domains (tasks), enabling novel applications unattainable with previous generalist methods, such as joint generation of image and multiple labels for data synthesis, multi-conditional generation, and simultaneously performing multiple understanding tasks.

Concurrent to our work, MMGen [92] also explores the joint modeling of multiple visual domains. However, this approach is limited to only 4 domains (image, depth, normal, and segmentation) and is trained only on ImageNet [22] scale with 256 image resolution. In contrast, our method is built upon a text-to-image foundation model [99] and incorporates as many as 8 visual domains with image resolutions of approximately 1024×1024 pixels, making it significantly more versatile in real-world applications.

Multi-modal Generation and Understanding In the context of multi-modal learning, previous works have explored the unification of vision and language by jointly modeling images and texts with autoregressive [88, 95, 96, 18], diffusion [6, 62], or hybrid frameworks [118, 100, 18]. These methods are capable of various cross-modality tasks, such as image-text mixed generation, text-to-image generation, and visual question-answering. In contrast to these methods that focus on unifying vision and language modalities, our work concentrates on the unification of pure visual domains within a single diffusion framework.

#### 3 Method

Overview In this section, we present the details of our Jodi framework, which unifies visual generation and understanding within a single diffusion model by jointly modeling the image domain and multiple label domains. As shown in Figure 2, our Jodi mainly consists of four parts: a Deep Compression Autoencoder (DC-AE) [15], a role assignment mechanism, a Switch module, and a linear diffusion transformer backbone [99]. Specifically, all of the image domain and the label domains are first compressed into a set of tokens by DC-AE with a downsampling factor of 32. Then, each domain is randomly assigned one of three roles: as a generation target, as a condition input, or to be ignored. Depending on the roles, the Switch module further processes the tokens in one of the following ways: adding noise, preserving their values, or setting them to zero. Subsequently, tokens from all domains are concatenated and fed into the linear diffusion transformer, which facilitates interactions across these domains and predicts the velocity field as in Rectified Flow [65]. Please refer to the appendix for more details on the framework architecture.

𝐯

𝐯

𝐯

𝐯

Timestep

###### Text Linear Diffusion Transformer

Domain

+ noise identity zero

[ G ] [ C ] [ X ] Role = G / C / X

𝐱

𝐲

𝐲

𝐲

Role

Switch

Switch 𝐲

Switch

Switch

G / C / X

𝐱

𝐲

𝐲

DC-AE

DC-AE

DC-AE

DC-AE

G: generate C: condition X: ignore

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Switch

Figure 2: Overview of our Jodi framework. For the sake of clarity, only four domains are illustrated.

###### 3.1 Joint Modeling

Role Assignment Let y0 = x denote the tokens of image domain and y1,y2,...,yM denote the tokens of M distinct label domains. At each training iteration, each domain is randomly assigned one of three roles: 1) [G], which means the model will learn to generate this domain; 2) [C], which means the model will use this domain as a condition; 3) [X], which means this domain will be ignored. In this manner, our model learns a class of probability distributions as follows:

p {ym |rolem=[G]} {ym |rolem=[C]} . (1)

Since each domain can be an outcome, a condition, or be ignored in Eq. (1), our model learns diverse distributions, including three most typical ones: 1) p(x,y1,y2,···), joint generation, where the model simultaneously generates both the image and the corresponding labels of different domains; 2) p(x|y1,y2,···), controllable generation, where the images are generated conditioned on any combination of the label domains; 3) p(y1,y2,···|x), image perception, where the model accepts an input image and predicts multiple labels at once. In a word, our method unifies various distributions related to both image generation and understanding within a single model.

Switch Module Depending on the roles assigned, the Switch module processes the tokens in different ways, as shown on the right of Figure 2. Specifically, at diffusion time step t, the [G] tokens are linearly interpolated with noise ϵm ∼ N(0,I) as in Rectified Flow [65], the [C] tokens remain unchanged, and the [X] tokens are set to zero. Let y0m = ym, this process is formulated as follows:

 

(1 − t)y0m + tϵm if rolem = [G] y0m if rolem = [C] 0 if rolem = [X]

ytm =

(2)



Objective Function Given the processed tokens in Eq. (2), we optimize our model by flow matching [64, 65]. Specifically, our model learns to predict the velocity field of [G] tokens conditioned on [C] tokens, with the following objective function:

vθm(yt0,··· ,ytM,t) − (ϵm − y0m) 2 , (3)

L = Et∼U(0,1),ϵ0:M∼N(0,I),y00:M∼D

m: rolem=[G]

where vθ(·) is the velocity predictor with a linear transformer architecture, introduced in Section 3.2.

###### 3.2 Model Architecture

Linear Diffusion Transformer We employ the attention mechanism [91] to model the interaction among different visual domains and predict the velocity field in Equation (3). However, as the number of domains increases, we need to carefully consider the computational complexity. Suppose we have M visual domains in total, each domain contains N tokens, and each token is D-dimensional. In this setting, the full attention mechanism exhibits a time complexity of O(M2N2D + MND2) and a space complexity of O(M2N2 + MND), both scaling quadratically with respect to the number of

domains M. In consequence, training our model with a full attention diffusion transformer [25, 7] is computationally inefficient or even infeasible. To solve this problem, we choose Sana [99] as our backbone, which adopts linear transformer [44] for efficient text-to-image generation. Using linear transformer, the time complexity is reduced to O(MND2) and the space complexity to O(MND), both of which are linear with respect to M. As a result, our model can efficiently handle as many as 8 visual domains. An empirical comparison on the computational cost can be found in the appendix.

When a domain is assigned the role [X], i.e., to be ignored, the corresponding tokens should not participate in the attention computation. To this end, we design a masked version of linear attention. For a single attention head, let Qi,Ki,Vi ∈ R1×d denote the query, key, and value of the ith token, and mi ∈ {0,1} indicate whether to ignore this token, the masked linear attention is designed as:

ReLU(Qi) MNj=1 ReLU(mjKj)TVj ReLU(Qi) MNj=1 ReLU(mjKj)T

, i = 1,2,...,MN. (4)

Oi =

When mj = 0 in Eq. (4), the jth token vanishes from both the denominator and numerator, which means it is excluded from the attention computation.

Domain-invariant Positional Embeddings A notable feature of our backbone Sana [99] is that it does not use explicit positional embeddings (NoPE) [35, 45]. However, in our multi-domain scenario, there is a strong spatial correspondence between the visual domains. Therefore, it is necessary to explicitly indicate the spatial positions to facilitate precise spatial alignment across different domains. To this end, we add domain-invariant sinusoidal positional embeddings to the tokens of each visual domain, where the same positions in different visual domains share identical positional embeddings, providing an explicit cue for the spatial alignment. Besides, we also introduce domain embeddings and role embeddings to help the model distinguish the domains and the roles of the tokens.

###### 3.3 Data Construction

To support joint modeling of multiple visual domains, we require a large-scale dataset containing high-quality images and corresponding labels of various domains. We construct the dataset from two kinds of sources: 1) images with predicted labels and 2) images with ground-truth labels.

First, we collect images with high quality and diversity from several publicly available sources, including Subjects200K [87], Aesthetic-4K [110], and Pexels [70, 30]. All of these images have resolutions over 1024×1024, which is advantageous for training a high-resolution generative model. As these datasets lack labels, we use state-of-the-art predictors to automatically annotate the data with labels corresponding to 7 specific domains. Specifically, we employ Informative Drawings [13] to generate line arts, PiDiNet [85] to extract edge maps, Depth Anything V2 [105] and Lotus [36] to estimate depth maps, Lotus [36] to estimate normal maps, RGB2X [109] to estimate albedos, Oneformer [42] to predict segmentation colormaps, and Openpose [8] to predict human skeletons. In this manner, we construct a dataset containing 200,000 images with corresponding 7×200,000 predicted labels. We name this dataset Joint-1.6M, and it will be made publicly available.

However, the predicted labels may lack sufficient accuracy, especially for in-the-wild images. To this end, we also employ datasets with ground-truth labels. Specifically, we use BSDS500 [4] for edge maps, Hypersim [74] for depth, normal, and albedo maps, and ADE20K [117] for semantic segmentation maps. These datasets encompass a total of 90,000 images.

Furthermore, we use BLIP2-OPT-2.7b [58] and Qwen2-VL-7b-Instruct [93] to generate captions. The former tends to provide a concise description of the main subject in the image, while the latter tends to give a long paragraph that describes the subject, background, and the overall atmosphere in detail. During the training process, one of these two captions is randomly selected for each image.

Table 1: Comparison of training costs among unified models. We use much less data and resources.

Method Base Model Parameters Dataset Size Training GPU OmniGen [98] Phi-3 [1] 3.8B 100M 104×A800 PixWizard [63] Lumina-Next-T2I [120] 2B 30M OneDiffusion [54] (from scratch) 2.8B 75M TPU v3-256, 64×H100 Jodi (ours) Sana [99] 1.6B 290K 8×A6000

#### 4 Experiment

###### 4.1 Setup

Training Details We adopt Sana [99] as our base model. We train our model using the CAME-8bit optimizer [67] for 130K steps, with a learning rate of 4 × 10−5, a batch size of 32, and BF16 mixed-precision, which takes around 535 hours on 8 RTX A6000. Since our dataset contains images with various aspect ratios, we use a ratio bucketing strategy [69] during training to prevent important contents from being cropped. This also allows users to generate images with a wide range of aspect ratios during inference. It is worth noting that we use significantly less data and computational resources than the other unified models, as shown in Table 1.

Sampling Details We employ Flow-DPM-Solver [99], a variant of DPM-Solver++ [66] adapted for rectified flow. The classifier-free guidance [41] scale is set to 4.5. For joint generation and controllable generation, we use 20 sampling steps. For image perception, we use 10 sampling steps since increasing the steps leads to little performance gain.

###### 4.2 Visual Generation and Understanding

Joint Generation In Figure 3, we illustrate the capability of our Jodi to simultaneously generate high-quality images of various aspect ratios along with corresponding labels, including depth, normal, albedo, edge, lineart, segmentation, and openpose. The generated images and the generated labels are semantically consistent and spatially aligned, credited to the linear attention and domain-invariant positional embeddings. Please refer to the appendix for more results.

Controllable Generation To demonstrate Jodi’s performance in controllable generation, we first generate images using existing labels as input conditions and evaluate their fidelity using FID scores [39]. Then, to evaluate the faithfulness of the generated images to the input conditions, we re-extract the conditions from the generated images and compare them to the input conditions using LPIPS [112]. As shown in Table 2 and Figure 4, our Jodi achieves superior performance for all conditions compared to existing unified models as well as generation-only specialist models.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

A close-up photograph of a titmouse perched on a branch.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Portrait photo of a beautiful elf from The Lord of the Rings.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

A man standing at the top of a mountain.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Figure 3: Joint generation of images and labels across a wide range of aspect ratios. Table 2: Quantitative comparison of controllable generation.

Depth Normal Edge Lineart Openpose

Method

LPIPS↓ FID↓ LPIPS↓ FID↓ LPIPS↓ FID↓ LPIPS↓ FID↓ LPIPS↓ FID↓

ControlNet [111] 0.29 19.5 0.35 28.0 0.23 18.9 0.33 15.9 0.11 32.0 UniControl [73] 0.29 18.8 0.35 22.5 0.31 39.1 - - 0.11 26.8 EasyControl [113] 0.27 19.5 - - 0.31 20.0 - - 0.12 33.9

OmniGen [98] 0.31 20.4 0.33 24.9 0.25 23.3 0.35 102.7 0.22 33.5 PixWizard [63] 0.23 14.4 0.27 16.7 0.29 22.9 0.22 14.6 0.16 31.7 OneDiffusion [54] 0.24 15.9 0.41 21.6 0.26 40.5 0.40 37.2 - Jodi (ours) 0.23 13.6 0.27 13.6 0.20 13.7 0.20 11.3 0.15 23.8

* First block: specialist models, second block: unified models.

Input Jodi (ours) OmniGen PixWizard OneDiffusion

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

A train is on the tracks.

Depth

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

A grasshopper is sitting on a piece of wood.

Normal

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

A beachside resort with palm trees and a sunset.

Edge

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

A red bird sits on a branch in the snow.

Lineart

Figure 4: Visual comparison of controllable generation.

Image Perception We assess the visual understanding ability of our Jodi on four image perception tasks: depth estimation, normal estimation, albedo estimation, and edge detection. For depth estimation, we evaluate our model on NYUv2 [80], ScanNet [20], and DIODE [90] datasets using absolute mean relative error. For normal estimation, we evaluate our model on NYUv2 [80], ScanNet [20], and iBims [49] datasets using mean angular error. For albedo estimation, we evaluate our model on the Hypersim [74] test set using PSNR and LPIPS [112]. For edge detection, we evaluate our model on the BSDS500 [4] test set, using F-scores at Optimal Dataset Scale (ODS) and Optimal Image Scale (OIS) as evaluation metrics. Besides, given the stochastic nature of diffusion models, we also report the ensemble performance by sampling five times for each input image and averaging the results. As shown in Table 3, Table 4, Table 5, Table 6, and Figure 5, our method consistently achieves superior or comparable results to the other unified models and specialist models.

Table 3: Quantitative comparison of depth estimation with absolute mean relative error ↓.

Method NYUv2 ScanNet DIODE

Marigold [46] 5.5 6.4 30.8 GeoWizard [29] 5.6 6.4 33.5 Lotus-D [36] 5.1 5.5 22.8

OmniGen [98] 9.2 10.1 30.6 PixWizard [63] 7.0 7.9 25.4 OneDiffusion [54] 8.9 9.7 25.2 Jodi (ours) 10.1 12.1 25.9 Jodi (ours, ensemble) 8.3 9.9 25.8

* First block: specialist models, second block: unified models.

Table 5: Quantitative comparison of albedo estimation on the Hypersim [74] test set.

Method PSNR↑ LPIPS↓

Ordinal Shading [9] 15.6 0.34 Kocsis et al. [50] 11.3 0.49 Careaga and Aksoy [10] 15.7 0.36 RGB2X [109] 20.6 0.18

Jodi (ours) 15.5 0.31 Jodi (ours, ensemble) 16.5 0.33

* First block: specialist models, second block: unified models.

Table 4: Quantitative comparison of surface normal estimation with mean angular error ↓.

Method NYUv2 ScanNet iBims

GeoWizard [29] 18.9 17.4 19.3 GenPercept [102] 18.2 17.7 18.2 StableNormal [106] 18.6 17.1 18.2 Lotus-D [36] 16.2 14.7 17.1

OmniGen [98] 28.9 28.9 31.3 PixWizard [63] 23.5 26.6 22.5 Jodi (ours) 21.1 24.3 20.1 Jodi (ours, ensemble) 18.6 20.3 18.2

* First block: specialist models, second block: unified models.

Table 6: Quantitative comparison of edge detection on the BSDS500 [4] test set.

Method ODS↑ IDS↑ HED [101] 0.788 0.808 PiDiNet [85] 0.807 0.823 OmniGen [98] 0.767 0.781 PixWizard [63] 0.605 0.633 OneDiffusion [54] 0.682 0.691 Jodi (ours) 0.826 0.851

* First block: specialist models, second block: unified models.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

NormalEstimationAlbedoEstimationDepthEstimationEdgeDetection

Input Jodi (ours) OmniGen PixWizard OneDiffusion

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Input Jodi (ours) OmniGen PixWizard Lotus

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Input Jodi (ours) Kocsis et al. Careaga and Aksoy RGB2X

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Input Jodi (ours) OmniGen PixWizard OneDiffusion

Figure 5: Visual comparison of image perception tasks on in-the-wild images.

###### 4.3 Analysis

Effect of Domain-invariant Positional Embeddings As described in Section 3.2, we introduce domain-invariant positional embeddings to encourage the spatial alignment across different visual domains. To validate the effect, we compare our models trained for 10K steps with and without positional embeddings, by observing whether the joint generated images and labels are spatially aligned. As shown in Figure 6, our model aligns the image domain and label domains significantly better with positional embeddings, whereas obvious misalignment is observed without positional embeddings.

Attention Map Visualization To further investigate how the tokens from different visual domains align and interact with each other, we pick two query tokens from the image domain and visualize the corresponding attention maps in Figure 7. As can be seen, most domains show strong activation at the same spatial location as the query token, demonstrating a good alignment between these domains. Interestingly, attention maps of different domains also reveal their own unique structural patterns. For example, the segmentation domain exhibits strong activation along semantic boundaries, and the openpose domain focuses more on the human figure.

Joint Consistency In Figure 8, we illustrate the consistency of our unified model across joint generation, controllable generation, and image perception tasks. We first perform joint generation based on the input prompt to produce samples covering all of the image and label domains. According to each generated label, we then apply controllable generation to generate new images that comply with these labels. Besides, we perform image perception on the image generated in the first step to detect all its labels. As can be observed, three types of tasks produce visually consistent results.

Extension to New Domains Our well-trained Jodi model can be readily extended to one or more new domains by appending the corresponding tokens to the existing ones. Figure 9 presents the joint generation results after fine-tuning the model on the doodle sketch domain [3] as well as simultaneous fine-tuning on the pixel, irradiance, and canny domains.

Image Image Image

Image & Depth Image & Edge

Image & Depth Image & Edge Image & Depth Image & Edge

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

w/opos.emb.w/pos.emb.

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

A hot air balloon in the shape of a heart.

A parrot standing on a branch.

A close up photo of a green seedling breaks out of the ground.

###### Figure 6: Effect of positional embeddings. Generated labels are overlaid on images for a better view.

Image Depth Normal Albedo Edge Lineart Segmentation Openpose

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Query 1

- Q1
- Q2

Query 2

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Figure 7: Visualization of attention map.

|[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]|
|---|

Joint Generation Controllable Generation Image Perception

|[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]|
|---|

A young woman with red hair and a red lipstick.

|[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]|
|---|

Figure 8: Jodi shows consistency among joint generation, controllable generation, image perception.

Image Doodle Sketch Image Pixel Irradiance Canny

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

A blue Porsche parked in a forest.

A house made out of sushi.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

A small cactus in the Sahara desert.

A cute cat.

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

A rocky lighthouse on a stormy sea.

A fallen maple leaf on wooden surface.

Figure 9: Joint generation results of our model extended to new domains.

#### 5 Conclusion and Limitations

Motivated by the interdependence between generation and understanding inherent in the joint distribution, we propose Jodi, a diffusion framework that jointly models the image domain and multiple label domains, unifying the visual generation and understanding. We design a role switch mechanism that allows the model to simultaneously learn joint generation, controllable generation, and image perception. Furthermore, to facilitate the interaction and alignment between tokens from different visual domains, we introduce masked linear attention and domain-invariant positional embeddings. As a result, our Jodi is capable of both generation and understanding tasks across the image domain and 7 distinct label domains. We also introduce the Joint-1.6M dataset, which will be publicly released to advance this research area.

While Jodi achieves impressive performance, it still comes with certain limitations. First, due to the limited size of our training dataset, the generated images may exhibit structural distortions, especially for human figures. Second, we simply represent each domain in RGB space. As a consequence, our model is currently limited to handling 12 clustered classes for the segmentation domain (see the appendix for details), as increasing the number of classes makes the RGB representations of the segments too similar to be reliably distinguished. Similarly, the RGB space is also not the ideal choice for the openpose domain, where the keypoints are better represented by coordinates. These problems may be resolved by incorporating more data and designing specific encoders and decoders for each visual domain, which we leave for future work.

It is important to note that, as with all generative models, Jodi may inherit biases present in the training dataset and could be misused to generate malicious or unintended content. Users should remain vigilant and comply with the usage policies to mitigate these risks.

#### References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.

- [2] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.

- [3] Ellie Arar, Yarden Frenkel, Daniel Cohen-Or, Ariel Shamir, and Yael Vinker. Swiftsketch: A diffusion model for image-to-vector sketch generation. arXiv preprint arXiv:2502.08642, 2025.

- [4] Pablo Arbelaez, Michael Maire, Charless Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. IEEE transactions on pattern analysis and machine intelligence, 33(5):898–916, 2010.

- [5] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

- [6] Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. In International Conference on Machine Learning, pages 1692–1717. PMLR, 2023.

- [7] BlackForestLab. Flux.1. https://blackforestlabs.io/flux-1/.
- [8] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.

- [9] Chris Careaga and Ya˘gız Aksoy. Intrinsic image decomposition via ordinal shading. ACM Transactions on Graphics, 43(1):1–24, 2023.

- [10] Chris Careaga and Ya˘gız Aksoy. Colorful diffuse intrinsic image decomposition in the wild. ACM Transactions on Graphics (TOG), 43(6):1–12, 2024.

- [11] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, pages 213–229, 2020.

- [12] Rebecca Chamberlain, Jennifer E Drake, Aaron Kozbelt, Rachel Hickman, Joseph Siev, and Johan Wagemans. Artists as experts in visual cognition: An update. Psychology of Aesthetics, Creativity, and the Arts, 13(1):58, 2019.

- [13] Caroline Chan, Frédo Durand, and Phillip Isola. Learning to generate line drawings that convey geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7915–7925, 2022.

- [14] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.

- [15] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.

- [16] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6593–6602, 2024.

- [17] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024.

- [18] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [19] Siying Cui, Jia Guo, Xiang An, Jiankang Deng, Yongle Zhao, Xinyu Wei, and Ziyong Feng. Idadapter: Learning mixed features for tuning-free personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 950–959, 2024.

- [20] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

- [21] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344– 16359, 2022.

- [22] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

- [23] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

- [24] Melissa Ellamil, Charles Dobson, Mark Beeman, and Kalina Christoff. Evaluative and generative modes of thought during the creative process. Neuroimage, 59(2):1783–1794, 2012.

- [25] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.

- [26] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

- [27] Myra A Fernandes, Jeffrey D Wammes, and Melissa E Meade. The surprisingly powerful influence of drawing on memory. Current Directions in Psychological Science, 27(5):302–308, 2018.

- [28] Tsu-Jui Fu, Yusu Qian, Chen Chen, Wenze Hu, Zhe Gan, and Yinfei Yang. Univg: A generalist diffusion model for unified image generation and editing. arXiv preprint arXiv:2503.12652, 2025.

- [29] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision, pages 241–258. Springer, 2024.

- [30] gaunernst. pexels-portrait. https://huggingface.co/datasets/gaunernst/pexels-portrait.
- [31] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587, 2014.

- [32] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

- [33] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. Advances in neural information processing systems, 37:36777– 36804, 2024.

- [34] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023.

- [35] Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. Transformer language models without positional encodings still learn positional information. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1382–1390, 2022.

- [36] Jing He, Haodong LI, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. In The Thirteenth International Conference on Learning Representations, 2025.

- [37] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.

- [38] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

- [39] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

- [40] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [41] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [42] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2989–2998, 2023.

- [43] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.

- [44] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020.

- [45] Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. Advances in Neural Information Processing Systems, 36:24892–24928, 2023.

- [46] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492–9502, 2024.

- [47] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

- [48] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

- [49] Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Korner. Evaluation of cnn-based singleimage depth estimation methods. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018.

- [50] Peter Kocsis, Vincent Sitzmann, and Matthias Nießner. Intrinsic image diffusion for indoor singleview material estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5198–5208, 2024.

- [51] Aaron Kozbelt. Artists as experts in visual cognition. Visual cognition, 8(6):705–723, 2001.

- [52] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

- [53] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023.

- [54] Duong H Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all. arXiv preprint arXiv:2411.16318, 2024.

- [55] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

- [56] Hsin-Ying Lee, Hung-Yu Tseng, and Ming-Hsuan Yang. Exploiting diffusion prior for generalizable dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7861–7871, 2024.

- [57] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.

- [58] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

- [59] Kunchang Li, Yali Wang, Gao Peng, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatial-temporal representation learning. In International Conference on Learning Representations, 2022.

- [60] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023.

- [61] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8640–8650, 2024.

- [62] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. arXiv preprint arXiv:2501.00289, 2024.

- [63] Weifeng Lin, Xinyu Wei, Renrui Zhang, Le Zhuo, Shitian Zhao, Siyuan Huang, Junlin Xie, Peng Gao, and Hongsheng Li. Pixwizard: Versatile image-to-image visual assistant with open-language instructions. In The Thirteenth International Conference on Learning Representations, 2025.

- [64] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

- [65] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

- [66] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.

- [67] Yang Luo, Xiaozhe Ren, Zangwei Zheng, Zhuo Jiang, Xin Jiang, and Yang You. Came: Confidenceguided adaptive memory efficient optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4442–4453, 2023.

- [68] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4296–4304, 2024.

- [69] NovelAI. Novelai improvements on stable diffusion, 2022.
- [70] opendiffusionai. pexels-photos-janpf. https://huggingface.co/datasets/opendiffusionai/ pexels-photos-janpf.
- [71] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

- [72] Koutilya Pnvr, Bharat Singh, Pallabi Ghosh, Behjat Siddiquie, and David Jacobs. Ld-znet: A latent diffusion approach for text-based image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4157–4168, 2023.

- [73] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. Advances in Neural Information Processing Systems, 36, 2024.

- [74] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021.

- [75] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [76] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.

- [77] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.

- [78] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

- [79] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2024.

- [80] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012.

- [81] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

- [82] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.

- [83] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

- [84] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

- [85] Zhuo Su, Wenzhe Liu, Zitong Yu, Dewen Hu, Qing Liao, Qi Tian, Matti Pietikäinen, and Li Liu. Pixel difference networks for efficient edge detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5117–5127, 2021.

- [86] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015.

- [87] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.

- [88] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- [89] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pages 1747–1756. PMLR, 2016.

- [90] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.

- [91] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

- [92] Jiepeng Wang, Zhaoqing Wang, Hao Pan, Yuan Liu, Dongdong Yu, Changhu Wang, and Wenping Wang. Mmgen: Unified multi-modal image generation and understanding in one go. arXiv preprint arXiv:2503.20644, 2025.

- [93] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [94] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identitypreserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.

- [95] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [96] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, and Yao Lu. VILA-u: a unified foundation model integrating visual understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025.

- [97] Yi Wu, Ziqiang Li, Heliang Zheng, Chaoyue Wang, and Bin Li. Infinite-id: Identity-preserved personalization via id-semantics decoupling paradigm. In European Conference on Computer Vision, pages 279–296. Springer, 2024.

- [98] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

- [99] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. SANA: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In The Thirteenth International Conference on Learning Representations, 2025.

- [100] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025.

- [101] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pages 1395–1403, 2015.

- [102] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? In The Thirteenth International Conference on Learning Representations, 2025.

- [103] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Openvocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2955–2966, 2023.

- [104] Yifeng Xu, Zhenliang He, Shiguang Shan, and Xilin Chen. CtrloRA: An extensible and efficient framework for controllable image generation. In The Thirteenth International Conference on Learning Representations, 2025.

- [105] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.

- [106] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics (TOG), 43(6):1–18, 2024.

- [107] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

- [108] Yunfan Ye, Kai Xu, Yuhang Huang, Renjiao Yi, and Zhiping Cai. Diffusionedge: Diffusion probabilistic model for crisp edge detection. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 6675–6683, 2024.

- [109] Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick Hold-Geoffroy, Yiwei Hu, Fujun Luan, LingQi Yan, and Miloš Hašan. Rgbx: Image decomposition and synthesis using material- and lighting-aware diffusion models. In ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH ’24, New York, NY, USA,

2024. Association for Computing Machinery.

- [110] Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

- [111] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836– 3847, 2023.

- [112] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

- [113] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027, 2025.

- [114] Canyu Zhao, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks. arXiv preprint arXiv:2502.17157, 2025.

- [115] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and KwanYee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

- [116] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5729–5739, 2023.

- [117] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.

- [118] Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In The Thirteenth International Conference on Learning Representations, 2025.

- [119] Muzhi Zhu, Yang Liu, Zekai Luo, Chenchen Jing, Hao Chen, Guangkai Xu, Xinlong Wang, and Chunhua Shen. Unleashing the potential of the diffusion model in few-shot semantic segmentation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

- [120] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, Xu Luo, Zehan Wang, Kaipeng Zhang, Lirui Zhao, Si Liu, Xiangyu Yue, Wanli Ouyang, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-next : Making lumina-t2x stronger and faster with next-dit. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

### Appendix

#### A Detailed Architecture

Figure 10 shows the detailed architecture of our Jodi. To incorporate the label domains into our backbone model Sana [99], we add a new patch embedding layer and a new final layer for each label domain. The patch embedding layers are responsible for projecting the encoded tokens to match the input dimension of the backbone model, and the final layers project them back to match the input dimension of the decoder. The patch embedding layers of the label domains are initialized from the pretrained Sana weights of the image domain, while the new final layers are randomly initialized. We find that this initialization strategy leads to the best convergence.

The backbone is a stack of linear transformer blocks, where each block is composed of AdaLN-Zero layers [71], a linear attention layer [44], a cross attention layer [75], and a mix FFN layer [99]. The scale, shift, and gate parameters of the AdaLN-Zero layers are obtained via an MLP that takes the role embeddings, domain embeddings, and timestep embeddings as input; therefore, these parameters are tailored for each domain, helping the model distinguish the roles and domains.

𝐯 𝐯 𝐯 𝐯

Final Layer

Final Layer

Final Layer

Final Layer

Gate

Gate

Gate

Gate

Mix FFN

Scale, Shift

Scale, Shift

Scale, Shift

Scale, Shift

Layer Normalization

Cross Attention

Gate

Gate

Gate

Gate

Linear Attention

Scale, Shift

Scale, Shift

Scale, Shift

Scale, Shift

Layer Normalization

PatchEmbed

PatchEmbed

PatchEmbed

PatchEmbed

𝐱

𝐲 𝐲

𝐲

Switch

Switch

Switch

Switch

𝐱

𝐲

𝐲

𝐲

MLP

Embed Embed Embed

Text Encoder

DC-AE

DC-AE

DC-AE

DC-AE

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

- A person holding a large, vibrant red

Role Timestep Domain

umbrella with intricate patterns, …

###### Figure 10: Detailed architecture of Jodi. For the sake of clarity, only four domains are illustrated.

#### B Relationships among Text, Image, and Labels

For joint generation and controllable generation, we can always assume a proper input text describing the content of the image. However, for image perception tasks, the labels are generally determined by the given image alone, regardless of the text description. In the context of graphical models, the labels and the text are conditionally independent given the image, as illustrated by the probabilistic graph in Figure 11. Accordingly, we set the text input empty for image perception tasks during both training and inference.

Text Image Labels

Figure 11: The probabilistic graph of text, image, and labels.

#### C Computational Cost

In Section 3.2 of the main paper, we analyze the theoretical computational complexity of using linear attention versus full attention. In Figure 12, we present the actual VRAM usage, training time, and inference latency when using vanilla full attention [91], flash attention [21], and linear attention [44] in practice. As the number of domains increases, the VRAM usage of vanilla full attention quickly exceeds the memory limits of an RTX A6000 GPU, making our training infeasible. Although flash attention reduces memory usage, its training time is over twice as long as that of linear attention when handling 8 domains, resulting in lower efficiency.

vanilla full attention flash attention linear attention

RTX A6000 (48 GiB)

25 InferenceLatency(s/sample)

48

25

TrainingTime(s/iteration)

20

20

45

VRAM(GiB)

2.3x

15

15

10

10

40

5

5

35

0

0

2 3 4 5 6 7 8

2 3 4 5 6 7 8

2 3 4 5 6 7 8

# domains

# domains

# domains

Figure 12: Comparison of actual computational cost among three types of attention.

#### D Notes on Segmentation Domain

As discussed in the limitation part in Section 5 of the main paper, we represent each visual domain in RGB space, which is not suitable for the semantic segmentation domain. Specifically, the segmentation dataset ADE20K [117] contains as many as 150 semantic classes, where some of the classes are assigned similar or even the same colors in RGB space, causing confusion for the model. To mitigate this problem, we group the 150 classes into 12 manually defined superclasses as shown in Table 7, where the RGB values assigned to different superclasses are set to be as far apart as possible. However, this is apparently not the optimal solution because it decreases the number of distinguishable classes. In the future, we plan to extend our model beyond the RGB space to better accommodate special domains like the segmentation domain.

Table 7: 12 superclasses and the corresponding RGB colors.

Superclass Person Animal Plant Water Mountain Sky Building Vehicle Wall Road Furniture Others

Color

FF7FFF FF7F00 7FFF00 007FFF 00FF7F 7FFFFF FF007F 7F00FF FFFF7F 7F0000 007F00 00007F

#### E Additional Quantitative Results

Semantic Segmentation In Table 8, we report the Intersection-over-Union (IoU) for each superclass except “Others”, as well as their mean IoU (mIoU). We also report the ensemble performance by sampling five times for each input image and performing majority voting. For the methods trained on the original 150 classes of ADE20K [117], we map their predictions to our 12 superclasses before computing IoUs. It is worth noting that such a comparison is somewhat unfair to the other methods, because these methods are trained to predict 150 classes, which is a more challenging task than predicting our 12 superclasses.

Table 8: Quantitative comparison on semantic segmentation (12 classes) on ADE20K [117] test set.

IoU per class

Method

mIoU Person Animal Plant Water Mountain Sky Building Vehicle Wall Road Furniture

Uniformer [59] 78.0 62.9 75.8 64.9 61.7 93.2 87.1 76.2 87.9 74.6 81.5 76.7 Oneformer [42] 87.3 65.4 81.0 88.4 69.7 95.2 90.8 86.2 90.0 82.7 86.0 83.9

PixWizard [63] 47.1 0.0 53.0 25.4 14.4 85.1 50.3 29.7 66.1 38.9 24.5 39.5 Jodi (ours) 74.0 14.7 55.7 50.7 37.9 90.9 67.0 52.5 72.4 61.0 56.2 57.5 Jodi (ours, ensemble) 79.5 1.9 65.6 60.9 38.9 92.4 78.0 66.4 79.4 67.5 65.4 63.3

- * First block: specialist models, second block: unified models.

Full Results of Depth and Normal Estimation We present the full evaluation results of depth estimation in Table 9 and normal estimation in Table 10, where additional methods and metrics are included for a comprehensive comparison. The detailed description of these metrics can be found in Appendix A.2 of the Lotus paper [36].

Table 9: Quantitative comparison on depth estimation.

Method

NYUv2 [80] ScanNet [20] DIODE [90] AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑

Marigold [46]§ 5.5 96.4 99.1 6.4 95.2 98.8 30.8 77.3 88.7 GeoWizard [29]§ 5.6 96.3 99.1 6.4 95.0 98.4 33.5 72.3 86.5 GenPercept [102]§ 5.6 96.0 99.2 6.2 96.1 99.1 35.7 75.6 86.6 Lotus-D [36]§ 5.1 97.2 99.2 5.5 96.5 99.0 22.8 73.8 86.2

OmniGen [98]† 9.2 91.8 98.6 10.1 90.0 98.2 30.6 71.0 85.8 PixWizard [63]† 7.0 95.0 99.1 7.9 93.7 98.8 25.4 72.1 85.0 OneDiffusion [54]† 8.9 92.0 98.2 9.7 90.7 98.0 25.2 72.2 85.3 Jodi (ours) 10.1 89.6 97.9 12.1 84.7 96.4 25.9 69.0 84.1 Jodi (ours, w/ ensemble) 8.3 92.7 98.8 9.9 89.4 97.8 25.8 71.0 84.9

- * First block: specialist models, second block: unified models.
- * § sourced from Lotus [36], † evaluated by ourselves following the Lotus protocol.

Table 10: Quantitative comparison on normal estimation.

Method

NYUv2 [80] ScanNet [20] iBims [49] mean↓ 11.25°↑ 30°↑ mean↓ 11.25°↑ 30°↑ mean↓ 11.25°↑ 30°↑

GeoWizard [29]§ 18.9 50.7 81.5 17.4 53.8 83.5 19.3 63.0 80.3 GenPercept [102]§ 18.2 56.3 81.4 17.7 58.3 82.7 18.2 64.0 82.0 StableNormal [106]§ 18.6 53.5 81.7 17.1 57.4 84.1 18.2 65.0 82.4 Lotus-D [36]§ 16.2 59.8 83.9 14.7 64.0 86.1 17.1 66.4 83.0

OmniGen [98]† 28.9 18.1 64.5 28.9 17.7 64.7 31.3 18.3 63.1 PixWizard [63]† 23.5 33.9 72.6 26.6 25.5 65.3 22.5 40.1 78.3 Jodi (ours) 21.1 47.7 77.7 24.3 41.3 73.9 20.1 60.0 79.6 Jodi (ours, w/ ensemble) 18.6 50.5 80.4 20.3 46.2 78.0 18.2 61.8 81.0

- * First block: specialist models, second block: unified models.
- * § sourced from Lotus [36], † evaluated by ourselves following the Lotus protocol.

Multi-conditional Controllable Generation In Table 11, we compare our performance of singleconditional and multi-conditional controllable generation. Specifically, we evaluate controllable generation conditioned individually on each of “Depth”, “Normal”, “Edge”, and “Lineart”, as well as conditioned on all of them together. Since multiple conditions provide more information than a single condition, it is natural that the former presents better controllability.

Table 11: Comparison between single and multi-conditional controllable generation.

Depth Normal Edge Lineart LPIPS↓ FID↓ LPIPS↓ FID↓ LPIPS↓ FID↓ LPIPS↓ FID↓

Method

Jodi (single) 0.23 13.6 0.27 13.6 0.20 13.7 0.20 11.3 Jodi (multi) 0.22 10.2 0.22 10.2 0.16 10.2 0.20 10.2

Multi-label Image Perception One of the notable features of our Jodi is that it can simultaneously predict multiple types of labels for a given image. In Table 12, we compare the performance of predicting all types of labels at once to predicting one label at a time. As can be seen, the performance of multi-label prediction is slightly inferior to that of single-label prediction, which we attribute to the absence of ground-truth labels for learning multi-label prediction (we use predicted labels as surrogates). Despite slightly lower performance, predicting all labels at once significantly saves inference time compared to predicting them one by one. For example, performing multi-label prediction 5 times still takes no more inference time than predicting 5 labels individually. Therefore, we can ensemble these 5 repeats of multi-label prediction to achieve better performance, which outperforms single-label prediction in most cases.

Table 12: Comparison between single and multi-label image perception.

Depth Normal Albedo Edge Seg.

(NYUv2 [80]) (NYUv2 [80]) (Hypersim [74]) (BSDS500 [4]) (ADE20K [117]) AbsRel↓ δ1↑ δ2↑ mean↓ 11.25°↑ 30°↑ PSNR↑ LPIPS↓ ODS↑ IDS↑ mIoU↑

Method

Jodi (single) 10.1 89.6 97.9 21.1 47.7 77.7 15.5 0.31 0.826 0.851 57.5 Jodi (multi) 11.8 85.9 97.0 22.1 44.5 76.1 13.9 0.44 0.765 0.782 57.1 Jodi (multi, ensemble) 9.6 90.4 98.3 19.6 46.9 79.0 15.1 0.43 - - 62.2

#### F Additional Visual Results

In this part, we provide additional visual results of our Jodi, including Figure 13 for joint consistency, Figure 14 for joint generation, Figure 15, Figure 16, and Figure 17 for controllable generation, and Figure 18 and Figure 19 for image perception.

##### G Licenses and Sources Licenses and sources of datasets and models used in our paper are listed in Table 13 and Table 14.

Table 13: Datasets used in this paper.

Dataset License Source

Aesthetic-4K [110] MIT HuggingFace Pexels-photos [70] Pexels HuggingFace Pexels-portrait [30] Pexels HuggingFace Subjects200K [87] Apache-2.0 HuggingFace ADE20K [117] BSD-3-Clause Official Website BSDS500 [4] - Official Website Hypersim [74] CC BY-SA 3.0 GitHub

Table 14: Models used in this paper.

Model License Source

Sana-1600M-1024px-BF16 [99] NVIDIA GitHub BLIP2-OPT-2.7b [58] MIT GitHub Qwen2-VL-7b-Instruct [93] Apache-2.0 GitHub Depth Anything V2 [105] CC BY-NC 4.0 GitHub Informative Drawings [13] MIT GitHub Lotus [36] Apache-2.0 GitHub Oneformer [42] MIT GitHub Openpose [8] Openpose GitHub PiDiNet [85] PiDiNet GitHub RGB2X [109] Adobe GitHub

Joint Generation Controllable Generation Image Perception

|[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]|
|---|

A beautiful sunset over the mountains with flowers growing on the ground.

|[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]|
|---|

|[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]|
|---|

|[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]|
|---|

Two prairie dogs are standing next to each other.

|[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]|
|---|

|[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]|
|---|

|[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]|
|---|

A woman with gold and silver paint on her face.

|[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]|
|---|

|[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]|
|---|

|[Figure 244]<br><br>[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]|
|---|

A castle on a cliff.

|[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]|
|---|

|[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]|
|---|

Figure 13: Jodi shows consistency among joint generation, controllable generation, image perception.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

A pirate ship charges forward into a raging storm of thunder and lightning.

A wizard with a long staff, looking out over a cliff edge.

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

An alley beneath a starry sky.

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

A watercolor of a fox in a snowy field.

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

A girl with silver hair wearing a flowing red cloak, facing the wind.

A lone tree on a hill under a starry night sky.

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

A close-up of a model wearing bold sunglasses and red lipstick.

Figure 14: Additional visual results of joint generation.

Input Jodi (ours) OmniGen PixWizard OneDiffusion

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

A beautiful sunset over the mountains with flowers growing on the ground.

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

A sewing machine on a table in a room.

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

A large cactus garden.

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

A modern apartment with a living room, kitchen and dining area.

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

A man standing in front of a waterfall.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

The colosseum in rome, Italy.

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

Beautiful woman with red sweater and earrings.

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

A person riding a dirt bike on a dirt track.

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

A man with a beard and a headband standing in front of a foggy mountain.

Figure 15: Additional visual comparisons of controllable generation.

Input Output 1 Output 2 Output 3

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

A blue butterfly on a purple flower.

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

A dog jumping into the water.

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

A close up of a flower.

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

A glass cup filled with a cup of milk and a stack of cookies.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

A plate of tomatoes, mozzarella and basil on a table.

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

A group of hot air balloons are flying in the sky.

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

A black car parked in front of a building.

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

A bunch of colorful flowers.

- Figure 16: Additional visual results of controllable generationusing depth, normal, or albedo as input.

Input Output 1

Output 2 Output 3

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

A long pier with a lighthouse at the end.

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

A tall brick building with windows and a clock.

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

An airport sign is lit up.

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

A loaf of bread on a plate with a knife.

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

A goat with standing on a hill.

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

A moose with large antlers is standing in a field.

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

A bustling urban scene with the Burj Khalifa.

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

A person standing on top of a mountain.

- Figure 17: Additionalvisualresultsofcontrollablegenerationusingedge,lineart,segmentationasinput.

Input Jodi (ours) OmniGen PixWizard OneDiffusion

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

Input Jodi (ours) OmniGen PixWizard Lotus

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

Input Jodi (ours) Kocsis et al. Careaga and Aksoy RGB2X

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

Input Jodi (ours) OmniGen PixWizard OneDiffusion

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

Figure 18: Additional visual comparisons of single-label perception on in-the-wild images.

Input Depth Normal Albedo Edge Lineart Segmentation

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

Input Depth Normal Albedo Edge Lineart Segmentation Openpose

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

Figure 19: Additional visual results of multi-label perception on in-the-wild images.

