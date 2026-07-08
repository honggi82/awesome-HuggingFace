## VideoCrafter1: Open Diffusion Models for High-Quality Video Generation

Haoxin Chen1* Menghan Xia1* Yingqing He1,2* Yong Zhang1* Xiaodong Cun1* Shaoshu Yang1,3,4 Jinbo Xing1,5 Yaofang Liu1,6 Qifeng Chen2 Xintao Wang1† Chao Weng1 Ying Shan1

# arXiv:2310.19512v1[cs.CV]30Oct2023

1 Tencent AI Lab 2 Hong Kong University of Science and Technology 3 Center for Research on Intelligent Perception and Computing, CASIA 4 School of Artificial Intelligence, UCAS 5 The Chinese University of Hong Kong 6 City University of Hong Kong

Project page: https://ailab-cvc.github.io/videocrafter GitHub: https://github.com/AILab-CVC/VideoCrafter

###### VideoCrafter1

[Figure 1]

[Figure 2]

[Figure 3]

Prompt:

a robot holding a basketball in outer space with expressive face, cyberpunk, digital art

T2V

Image:

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

#### I2V

Prompt:

a car running fast on the road

Figure 1. We have open-sourced two diffusion models for video generation in VideoCrafter1. The Text-to-Video (T2V) model takes a text prompt as input and generates a video accordingly. On the other hand, the Image-to-Video (I2V) model accepts either an image, a text prompt, or both as input for video generation.

### Abstract

can generate realistic and cinematic-quality videos with a resolution of 1024 × 576, outperforming other open-source T2V models in terms of quality. The I2V model is designed to produce videos that strictly adhere to the content of the provided reference image, preserving its content, structure, and style. This model is the first open-source I2V foundation model capable of transforming a given image into a video clip while maintaining content preservation constraints. We believe that these open-source video generation models will contribute significantly to the technological advancements within the community.

Video generation has increasingly gained interest in both academia and industry. Although commercial tools can generate plausible videos, there is a limited number of open-source models available for researchers and engineers. In this work, we introduce two diffusion models for high-quality video generation, namely text-to-video (T2V) and image-to-video (I2V) models. T2V models synthesize a video based on a given text input, while I2V models incorporate an additional image input. Our proposed T2V model

* Equal contribution. † Corresponding author.

### 1. Introduction

With the rapid development of generative models, particularly diffusion models [26, 47], numerous breakthroughs have been achieved in fields such as image generation [11, 12, 16, 19, 39, 41, 43, 57] and video generation [10, 24, 27, 45, 50], as well as in recognition and detection tasks [13, 37]. The most well-known open-source text-toimage (T2I) generative model is Stable Diffusion (SD) [42], which produces plausible results. Subsequently, its enhanced version, SDXL [39], was released, offering improved concept composition and image quality. Another notable T2I open-source model is IF [2], a cascaded model that operates on pixels rather than latent features. Regarding text-to-video (T2V) models, Make-A-Video [45] and Imagen Video [27] are cascaded models, while most other works, such as LVDM [24], Magic Video [63], ModelScope [50], and Align your Latents [10], are SD-based models. These models extend the SD framework to videos by incorporating temporal layers to ensure temporal consistency among frames. The spatial parameters are inherited from the pretrained SD UNet.

The flourishing of successful T2I models and advancements in downstream tasks can be largely attributed to the open-source environment within the community. SD serves as a critical foundation, as it is trained on a vast collection of text-image pairs using immense computing power. The cost associated with this is often prohibitive for most academic research groups. In contrast, in the field of T2V, Make-AVideo [45] and Imagen Video [27] demonstrate promising video results, but neither of them are open-sourced. Several startups, such as Gen-2 [1], Pika Labs [6], and Moonvalley [5], can generate high-quality videos, but their models remain inaccessible to researchers for further exploration.

Currently, several open-source T2V models exist, i.e., ModelScope [50], Hotshot-XL [4], AnimateDiff [23], and Zeroscope V2 XL [7]. The released ModelScope model can only generate videos with a resolution of 256 × 256, and the image quality is unsatisfactory. Zeroscope V2 XL improves its visual quality by tuning it on a small set of videos, but flickers and visible noise persist in its generated videos. Hotshot-XL aims to extend SDXL into a video model and produce a gif with 8 frames and a resolution of 512 × 512. AnimateDiff proposes to combine the temporal module with the spatial module of a LORA SD model. Since the temporal module is trained on Webvid-10M, the results of the original T2V model of AnimateDiff are poor. The combination with a high-quality LORA model can generate high-quality videos. However, the scope is restricted by the LORA model in terms of style and concept composition ability. There is still a lack of an open-source generic T2V foundation model capable of generating highresolution and high-quality videos.

Recently, Pika Labs and Gen-2 released their image-

to-video (I2V) models, aiming to animate a given image with a prompt while preserving its content and structure. Such a technique is still in its early stages, as the generated motion is limited, and there are usually visible artifacts. The only open-source generic I2V foundation model, I2VGen-XL [15], is released in ModelScope. This model uses image embedding to replace text embedding for tuning a pretrained T2V model. However, it does not satisfy the content-preserving constraints. The generated videos match the semantic meaning in the given image but do not strictly follow the reference content and structure. Hence, there is an urgent need for a good I2V model in the open-source community.

In this work, we introduce two diffusion models for highquality video generation: one for text-to-video (T2V) generation and the other for image-to-video (I2V) generation. The T2V model builds upon SD 2.1 by incorporating temporal attention layers into the SD UNet to capture temporal consistency. We employ a joint image and video training strategy to prevent concept forgetting. The training dataset comprises LAION COCO 600M [3], Webvid10M [8], and a 10M high-resolution collected video dataset. The T2V model can generate videos with a resolution of 1024 × 576 and a duration of 2 seconds. The I2V model, on the other hand, is based on a T2V model and accepts both text and image inputs. The image embedding is extracted using CLIP [14] and injected into the SD UNet through cross attention [54], similar to the injection of text embeddings. The I2V model is trained on LAION COCO 600M and Webvid10M. By releasing these models, we aim to make a significant contribution to the open-source community, enabling researchers and practitioners to build upon our work and further advance the field of video generation.

Our contributions can be summarized as follows:

- • We introduce a text-to-video model capable of generating high-quality videos with a resolution of 1024 × 576 and cinematic quality. The model is trained on 20 million videos and 600 million images.
- • We present an image-to-video model, the first opensource generic I2V model that can strictly preserve the content and structure of the input reference image while animating it into a video. This model allows for both image and text inputs.

### 2. Related Works

Diffusion models (DMs) [26, 46, 48] have recently shown unprecedented capability in content generation field, especially in text-to-image (T2I) generation [9, 21, 25, 38, 39, 41–43, 62]. Following the success of T2I DMs, Video Diffusion Models (VDMs) are proposed to model the spatial and temporal distribution of videos under the condition of text prompts (T2V). The first VDM [28] utilizes a space-time factorized U-Net to model low-resolution

Prompt: a girl with long curly blonde hair and sunglasses, camera pan from left to right

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

VideoCrafter1PikaLabsZeroscope-XLGen-2I2VGen-XL

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

Prompt: a dog wearing vr goggles on a boat

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

VideoCrafter1PikaLabsZeroscope-XLGen-2I2VGen-XL

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

Figure 2. Visual comparison with Gen-2, Pika Labs, I2VGen-XL, and Zeroscope-XL.

videos in pixel space, which is trained jointly on image and video data. To generate high-definition videos, ImagenVideo [27] introduces an effective cascaded paradigm of DMs with v-prediction parameterization method. To promote the computational efficiency, subsequent studies [10, 24, 50, 52, 63] mainly take the way of transferring T2I knowledge to T2V generation [33, 45] and learning DMs in latent space. Most recently, Zhang et al. [59] highlight the issue of heavy computational cost in pixel-based VDM [20] and poor text-video alignment in latent-based VDM, proposing a hybrid-pixel-latent VDM framework to address these issues.

Although T2V models can generate high-quality videos, they only accept text prompts as semantic guidance, which can be verbose and may not accurately reflect users’ intentions. Similar to adding controls in T2I models [31, 36, 44, 55, 60], introducing conditional controls in T2V DMs has increasingly attracted researchers’ attention. Gen-1 [18] and Make-Your-Video [53] integrate structure control into VDMs by concatenating frame-wise depth map with input noise sequences for video editing, while other control conditions, including pose [34, 61] and canny edge [29, 61] are also investigated. However, visual conditions in VDMs, such as RGB images, remain under-explored. Most recently, image condition is examined in Seer [22] and VideoComposer [51] for text-image-to-video synthesis, by keeping first-frame latent clean and concatenating image embedding with noise in channel dimension, respectively. Nevertheless, they either focus on the curated domain, i.e., indoor objects [22], or fail to generate temporally coherent frames and realistic motions [51] due to insufficient semantic understanding of the input image. Although DragNUWA [56] further introduce trajectory control into image-to-video generation, which can only mitigate the unrealistic-motion issue to some extent. Moreover, some recent close-sourced text-to-video diffusion models [30, 35, 45] or auto-regressive models [49, 58] successfully demonstrate their extension applicability to image-to-video synthesis. However, their results rarely adhere to the input image condition and suffers from the unrealistic temporal variation issue. While we build our model upon textconditioned VDMs to leverage their rich dynamic prior for animating open-domain images, incorporating tailored designs for better conformity and semantic understanding of the input image.

### 3. Methodology

###### 3.1. VideoCrafter1: Text-to-Video Model

Structure Overview. The VideoCrafter T2V model is a Latent Video Diffusion Model (LVDM) [24] consisting of two key components: a video VAE and a video latent diffusion model, as illustrated in Fig. 3. The Video VAE is

responsible for reducing the sample dimension, allowing the subsequent diffusion model to be more compact and efficient. First, the video data x0 is fed into the VAE encoder E to project it into the video latent z0, which exhibits a lower data dimension with a compressed video representation. Then, the video latent can be projected back into the reconstructed video x′0 via the VAE decoder D. We adopt the pretrained VAE from the Stable Diffusion model to serve as the video VAE and project each frame individually without extracting temporal information. After obtaining the video latent z0, the diffusion process is performed on z0 via:

T

q(zt|zt−1), (1)

q(z1:T|z0) :=

t=1

q(zt|zt−1) := N(zt; 1 − βtzt−1,βtI), (2)

where T is the number of diffusion timesteps, and βt is the noise level at timestep t. Thus, we can obtain a series of noisy video latents zt at arbitrary timesteps t.

To perform the denoising process, a denoiser U-Net is learned to estimate the noise in the input noisy latent, which will be discussed in the next section. After the progressive denoising process, the latent sample transitions from noisy to clean, and it can finally be decoded by the VAE decoder into a generated video in the pixel space.

Denoising 3D U-Net. As illustrated in Fig.3, the denoising U-Net is a 3D U-Net architecture consisting of a stack of basic spatial-temporal blocks with skip connections. Each block comprises convolutional layers, spatial transformers (ST), and temporal transformers (TT), where

ST = Projin ◦ (Attnself ◦ Attncross ◦ MLP) ◦ Projout,

- (3)

TT = Projin ◦ (Attntemp ◦ Attntemp ◦ MLP) ◦ Projout.

- (4)

The controlling signals of the denoiser include semantic control, such as the text prompt, and motion speed control, such as the video fps. We inject the semantic control via the cross-attention:

Attention(Q,K,V) = softmax

###### QKT √

d · V,where

- (5)

Q = WQ(i) · φi(zt),K = WK(i) · ϕ(y),V = WV(i) · ϕ(y).

- (6)

i

ϵ represents spatially flattened tokens of video latent, ϕ denotes the Clip text encoder, and y is the

φi(zt) ∈ RN×d

Denoising

Conv layers

𝒙𝟎

###### t

Spatial Transformer

fps

ℰ

Temporal Transformer

𝐗 𝐗 𝐗

𝒛𝟎

Diffusion

Denoising U‐Net

[Figure 48]

[Figure 49]

[Figure 50]

Text/Image Encoder

𝒛𝒕 𝒛𝒕−𝟏

𝒟

[Figure 51]

[Figure 52]

[Figure 53]

⋯

Embedding Layer

[Figure 54]

[Figure 55]

[Figure 56]

Text prompt

𝒙𝟎′ Image prompt

Element-wise addition

Spatial convolution

Temporal convolution

Spatial attention

Temporal attention

- Figure 3. The framework of the video diffusion model in VideoCrafter1. We train the video UNet in the latent space of the auto-encoder. FPS is taken as a condition to control the motion speed of the generated video. For the T2V model, only the text prompt is fed into the spatial transformer via cross-attention, while for the I2V model, both the text and image prompts are taken as the inputs.

Projection Net

Text prompt

Image

prompt

Text Encoder

|𝐅𝑖𝑛|
|---|

|𝐅𝑜𝑢𝑡|
|---|

Cross Attention

| | |
|---|---|
| | |

Image Encoder

Cross Attention

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

- Figure 4. The diagram of image conditional branch. The U-Net backbone features Fin are processed with the text and image embeddings via a dual cross-attention layer, the output of which are fused as Fout.

Text prompt: “a girl is reading books in her bedroom”

[Figure 57]

[Figure 58]

[Figure 59]

(a) (b) (c)

Figure 5. Image-conditioned text-to-video generation comparison. (a) Conditional image input. (b) Generation with the global semantic token conditioned. (c) Generation with the full patch visual tokens conditioned. The used text prompt is ”a beautiful girl with colorful hair”.

aim to integrate an additional conditional input, i.e., image prompt, into the video diffusion model, which is expected to synthesize dynamic visual content based on the provided image. For text-to-video diffusion models, the conditional text embedding space plays a crucial role in determining the visual content of the final output videos. To supply the video model with image information in a compatible manner, it is essential to project the image into a text-aligned embedding space. We propose learning such an embedding with rich details to enhance visual fidelity. Figure 4 illustrates the diagram of equipping the diffusion model with an image conditional branch.

input text prompt. Motion speed control with fps is incorporated through an FPS embedder, which shares the same structure as the timestep embedder. Specifically, the FPS or timestep is projected into an embedding vector using sinusoidal embedding. This vector is then fed into a two-layer MLP to map the sinusoidal embedding to a learned embedding. Subsequently, the timestep embedding and FPS embedding are fused via elementwise addition. The fused embedding is finally added to the convolutional features to modulate the intermediate features.

###### 3.2. VideoCrafter1: Image-to-Video Model

Text prompts offer highly flexible control for content generation, but they primarily focus on semantic-level specifications rather than detailed appearance. In the I2V model, we

Text-Aligned Rich Image Embedding. Since the text embedding is constructed using the pretrained CLIP [40] text encoder, we employ its image encoder counterpart to extract the image features from the input image. Although

Visual Text-Video Motion Temporal Quality Alignment Quality Consistency

I2VGen-XL† 55.23 47.22 59.41 59.31 ZeroScope 56.37 46.18 54.26 61.19 PikaLab∗ 63.52 54.11 57.74 69.35 Gen2∗ 67.35 52.30 62.53 69.71 VideoCrafter23.04 46.88 41.56 56.24* 55.78 VideoCrafter23.08 59.53 51.29 51.97 56.36 VideoCrafter23.10 61.64 66.76 56.06 60.36

Table 1. Human-preference aligned results from four different aspects, with the rank of each aspect in the brackets. * indicated these models are not open-sourced.

the global semantic token fcls from the CLIP image encoder is well-aligned with image captions, it primarily represents visual contents at a semantic level, while being less capable of capturing details. Inspired by existing visual conditioning works [44, 55], we utilize the full patch visual tokens Fvis = {fi}Ki=0 from the last layer of the CLIP image ViT [17], which are believed to encompass much richer information about the image.

To promote alignment with the text embedding, we utilize a learnable projection network P to transform Fvis into the target image embedding Fimg = P(Fvis), enabling the video model backbone to process the image feature efficiently. The text embedding Ftext and image embedding Fimg are then used to compute the U-Net intermediate features Fin via dual cross-attention layers:

QK⊤img

QK⊤text

Fout = Softmax(

)Vtext+Softmax(

√

√

)Vimg,

d

d

(7) where Q = FinWq, Ktext = FtextWk, Vtext = FtextWv, and Kimg = FimgWk′ , Vimg = FimgWv′ accordingly. Note that we use the same query for image crossattention as for text cross-attention. Thus, only two parameter matrices Wk′ , Wv′ are newly added for each crossattention layer. Figure 5 compares the visual fidelity of the generated videos conditioned on the global semantic token and our adopted rich visual tokens, respectively.

### 4. Experiments

###### 4.1. Implementation Details

Datasets. We employ an image and video joint training strategy for model training. The image dataset used is LAION COCO [3], a large text-image dataset consisting of 600 million generated high-quality captions for publicly available web images. For video datasets, we utilize the publicly available WebVid-10M [8], a large-scale dataset of short videos with textual descriptions sourced from stock footage sites, offering diverse and rich content. Additionally, we compile a large-scale high-quality video dataset containing 10 million videos with resolutions greater than 1280 × 720 for the training of T2V models.

[Figure 60]

Figure 6. The raw ratings from our user studies.

Training Scheme. To train the T2V model, we employ the training strategy used in Stable Diffusion, i.e., training from low resolution to high resolution. We first train the video model extended from the image model at a resolution of 256 × 256 for 80K iterations with a batch size of 256. Next, we resume from the 256 × 256 model and finetune it with videos for 136K iterations at a resolution of 512×320. The batch size is 128. Finally, we finetune the model for 45K iterations at a resolution of 1024 × 576. The batch size is 64. For the I2V model, we initially train the mapping from the image embedding to the embedding space used for the cross attention. Subsequently, we fix the mappings of both text and image embeddings and finetune the video model for improved alignment.

Evaluation Metrics. We employ comprehensive metrics to assess video quality and the alignment between text and video using EvalCrafter [32], a benchmark for evaluating video generation models. EvalCrafter conducts comparisons among our model, Gen-2, Pika Labs, and ModelScope, considering both quantitative metrics and user studies. We present the main results in Table 1 and Figure 6. Our T2V model achieves the best visual quality and video quality among open-source models. Please refer to EvalCrafter for further details. For qualitative evaluation, we provide several visual examples in Figures 2, 7, and 9 for illustration.

Relations to Floor33. We deploy the two open-source models on a Discord channel named Floor33, to allow users to explore the capability of the models online by just typing the prompt. We add an optional function, prompt exten-

Prompt: Macro len style, A tiny mouse in a dainty dress holds aparasol to shield from the sun.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

VideoCrafter1PikaLabsZeroscope-XLGen-2I2VGen-XL

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

Prompt: The old man the boat. in watercolor style

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

VideoCrafter1PikaLabsZeroscope-XLGen-2I2VGen-XL

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

Figure 7. Visual comparison with Gen-2, Pika Labs, I2VGen-XL, and Zeroscope-XL.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

##### 23.04 23.08 23.10

Figure 8. The visual comparisons of the visual quality between different VideoCrafter text-to-video versions. The prompts are “In Marvel movie style, supercute siamese cat as sushi chef”, ”A wise tortoise in a tweed hat and spectacles reads a newspaper, Howard Hodgkin style” and “hand-held camera, a politician giving a speech at a podium”, respectively. The comparison video will be released on our Github.

sion, to enrich the information in the user’s prompt. The discord channel can be accessed at https://discord. gg/rrayYqZ4tf.

###### 4.2. Performance Evaluation

Text-to-Video Results. We compare our T2V model with commercial models such as Gen-2 and Pika Labs, as well as the open-source model I2VGen-XL. Since I2VGen-XL is an image-to-video model, we first generate an image using SDXL and then employ I2VGen-XL to create a video. The results are displayed in Fig. 2 and 7. As shown in Table 1, our model outperforms open-source T2V models in terms of visual quality and text-alignment. Our model encourages large object movements during training, resulting in more significant motion in the generated videos compared to other models. However, larger motions can sometimes introduce errors in temporal consistency. Fig. 2 and 7 also demonstrate our superiority in visual quality and concept composition compared to open-source models. The image quality of Zeroscope is subpar, as it sometimes fails to generate content or produces artifacts like repetitive grids.

Gen-2 and Pika Labs consistently generate videos with high aesthetic scores, and noise is suppressed in their re-

sults. Nevertheless, Gen-2 occasionally struggles with concept composition, as seen in the two examples in Fig. 7, and its results are overly smooth. Pika Labs exhibits the best text-alignment performance but does not always generate the correct style, such as in the second example in Fig. 7.

We also compare the differences between VideoCrafter versions to verify our efforts. As shown in Fig. 8 and Table. 1, our method has a great process this year in both visual quality, text-video alignment, temporal consistency, and motion quality. We also find that our latest version (23.10) has achieved the same quality as Pika Lab [6], which demonstrates the benefits of our training and datasets.

Image-to-Video Results. We evaluate our method against existing state-of-the-art image-to-video approaches, including two open-source conditional video diffusion models and two commercial product demos. VideoComposer [51] is a recently released model for compositional video generation that supports text-image-to-video synthesis. I2VGenXL [15] is an open-source image-to-video generation project. Pika [6] and Gen-2 [1] are well-known text-tovideo generation products developed by commercial com-

[Figure 110]

[Figure 111]

OursI2VGen‐XLImageinputVideoComposerPikaGEN2

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

Figure 9. Visual comparisons with image-to-video approaches: VideoComposer, I2VGen-XL, Pika, Gen-2 and our I2V model.

### 5. Conclusion and Future Work

panies, which also support image-to-video applications.

We introduce two diffusion models for video generation. One is a text-to-video generation model capable of producing high-quality, high-resolution, cinematic-quality videos with a resolution of 1024 × 576. It offers the best quality among open-source T2V models. The other is an imageto-video generation model, which is the first open-source generic I2V foundation model that can preserve the content and structure of the given reference image.

The visual comparison results are illustrated in Figure 9. We observe that Pika, Gen-2, and our approach achieve relatively better visual fidelity to the conditional image than VideoComposer and I2V-XL. Although the first frame is almost identical to the input image, VideoComposer suffers from severe temporal inconsistency, where the subsequent frames transform into entirely different appearances. I2VXL exhibits good temporal consistency and motion magnitude, but the appearance deviates significantly from the conditional image. Pika achieves the best visual fidelity and temporal consistency; however, it generally presents very subtle motion magnitude. In contrast, Gen-2 can generate satisfying motion magnitude and visual fidelity, but its performance is not stable, i.e., it sometimes suffers from temporal drifting problems (as in the car case). Our I2V model demonstrates better performance in these cases, with good temporal consistency and motion magnitude, as well as acceptable visual fidelity. However, our I2V model still has several limitations such as the successful rate, unsatisfactory facial artifacts, etc, requiring further efforts for improvement.

The existing open-source models merely represent the starting point. Improvements in duration, resolution, and motion quality remain crucial for future developments. Specifically, the current duration of the two models is limited to 2 seconds; extending this to a longer duration would be more beneficial. This can be accomplished by training with additional frames and developing a frame-interpolation model. As for resolution, employing a spatial upscaling module or collaborating with ScaleCrafter [25] presents a promising strategy. Moreover, improvements in motion and visual quality can be achieved by utilizing higher-quality data.

### References

- [1] Gen-2. Accessed October 22, 2023 [Online] https:// research.runwayml.com/gen2.
- [2] If. Accessed October 22, 2023 [Online] https:// github.com/deep-floyd/IF.
- [3] Laion-coco. Accessed October 22, 2023 [Online] https: //laion.ai/blog/laion-coco/.
- [4] Hotshot-xl. Accessed October 22, 2023 [Online] https: //github.com/hotshotco/Hotshot-XL.
- [5] Moonvalley. Accessed October 22, 2023 [Online] https: //moonvalley.ai/.
- [6] Pika labs. Accessed October 22, 2023 [Online] https: //www.pika.art/.
- [7] Zeroscope-xl. Accessed October 22, 2023 [Online] https: //huggingface.co/cerspense/zeroscope_v2_ XL.
- [8] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021.
- [9] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [10] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.
- [11] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.
- [12] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [13] Shoufa Chen, Peize Sun, Yibing Song, and Ping Luo. Diffusiondet: Diffusion model for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19830–19843, 2023.
- [14] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2829, 2023.
- [15] I2VGen-XL contributors. I2vgen-xl. ModelScope. Accessed October 15, 2023 [Online] https://modelscope.cn/ models/damo/Image-to-Video/summary.
- [16] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.

- [17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020.
- [18] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In ICCV, 2023.
- [19] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In European Conference on Computer Vision, pages 89–106. Springer, 2022.
- [20] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In ICCV, 2023.
- [21] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022.
- [22] Xianfan Gu, Chuan Wen, Jiaming Song, and Yang Gao. Seer: Language instructed video prediction with latent diffusion models. arXiv preprint arXiv:2303.14897, 2023.
- [23] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [24] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022.
- [25] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. arXiv preprint arXiv:2310.07702, 2023.
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [27] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [28] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022.
- [29] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.
- [30] Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, and Jingdong Wang. Videogen: A reference-guided latent diffusion approach for high definition text-to-video generation. arXiv preprint arXiv:2309.00398, 2023.

- [31] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023.
- [32] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models, 2023.
- [33] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In CVPR, 2023.
- [34] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186, 2023.
- [35] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023.
- [36] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.
- [37] Cindy M Nguyen, Eric R Chan, Alexander W Bergman, and Gordon Wetzstein. Diffusion in the dark: A diffusion model for low-light text recognition. arXiv preprint arXiv:2303.04291, 2023.
- [38] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. 2022.
- [39] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. 2021.
- [41] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [43] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022.
- [44] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023.

- [45] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023.
- [46] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. 2015.
- [47] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [48] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.
- [49] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. In ICLR, 2023.
- [50] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.
- [51] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023.
- [52] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023.
- [53] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. arXiv preprint arXiv:2306.00943, 2023.
- [54] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023.
- [55] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [56] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023.
- [57] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [58] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In CVPR, 2023.

- [59] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation, 2023.
- [60] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.
- [61] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023.
- [62] Yuechen Zhang, Jinbo Xing, Eric Lo, and Jiaya Jia. Realworld image variation by aligning diffusion inversion chain. arXiv preprint arXiv:2305.18729, 2023.
- [63] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.

