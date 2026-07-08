### MoonShot: Towards Controllable Video Generation and Editing with Multimodal Conditions

David Junhao Zhang♦♢, Dongxu Li♦, Hung Le♦, Mike Zheng Shou♢*, Caiming Xiong♦, Doyen Sahoo♦ ♦Salesforce Research ♢Show Lab, National University of Singapore https://showlab.github.io/Moonshot/

# arXiv:2401.01827v1[cs.CV]3Jan2024

Given Image Given Image

A robot is walking A robot is landing to fly

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

Image Animation

Zero-Shot Subject Customized Video Generation

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

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

A can running with heavy snow

Wearing a suit

Directly Use Image ControlNet Video Editing

Figure 1. Our foundational video diffusion model, underpinned by multimodal conditioning, effectively facilitates image animation, the creation of customized videos, and precise control over geometric structures through the image ControlNet. Additionally, it enables video editing guided by text and image inputs, producing better results than recent state-of-the-art methods in these applications.

#### Abstract

generation, image animation and video editing, unveiling its potential to serve as a fundamental architecture for controllable video generation. Models will be made public on https://github.com/salesforce/LAVIS.

Most existing video diffusion models (VDMs) are limited to mere text conditions. Thereby, they are usually lacking in control over visual appearance and geometry structure of the generated videos. This work presents MoonShot, a new video generation model that conditions simultaneously on multimodal inputs of image and text. The model builts upon a core module, called multimodal video block (MVB), which consists of conventional spatialtemporal layers for representing video features, and a decoupled crossattention layer to address image and text inputs for appearance conditioning. In addition, we carefully design the model architecture such that it can optionally integrate with pre-trained image ControlNet modules for geometry visual conditions, without needing of extra training overhead as opposed to prior methods. Experiments show that with versatile multimodal conditioning mechanisms, MoonShot demonstrates significant improvement on visual quality and temporal consistency compared to existing models. In addition, the model can be easily repurposed for a variety of generative applications, such as personalized video

#### 1. Introduction

Recently, text-to-video diffusion models (VDMs) [5, 12, 15, 21, 24, 50, 60, 78, 82, 83] have developed significantly, allowing creation of high-quality visual appealing videos. However, most existing VDMs are limited to mere text conditional control, which is not always sufficient to precisely describe visual content. Specifically, these methods are usually lacking in control over the visual appearance and geometry structure of the generated videos, rendering video generation largely reliant on chance or randomness.

It is well acknowledged that text prompts are not sufficient to describe precisely the appearance of generations, as illustrated in Fig. 2. To address this issue, in the context of text-to-image generation, efforts are made to achieve personalized generation [9, 31, 34, 46, 75] by fine-tuning diffusion models on input images. Similarly for video generation, AnimateDiff relies on customized model weights to inject conditional visual content, either via LoRA [27] or

∗Corresponding Author

spatial layers

[Figure 29]

Text prompt

Text prompt Image prompt

resnet2d self-attn text-attn image-attn temporal layers

What does the car look like?

|A car running in the desert| |
|---|---|
| | |

|A car running in the desert| |
|---|---|
| | |

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

insert temp-conv temp-attn frozen/train

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

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

…

###### …

[Figure 50]

(a) Video with text only (b) Video with text and image

(a) original (b) composed (c) decomposed (d) MVB

[Figure 51]

[Figure 52]

[Figure 53]

visual quality image ControlNet

Figure 2. A single text prompt (a) lacks the precision and detail for accurate subject description. However, a picture (b) conveys much more, worth a thousand words. By combining a picture with text, we can produce videos that more closely match user requirements.

[Figure 54]

[Figure 55]

[Figure 56]

Figure 3. Different designs of spatial-temporal modules include: (a) the original spatial module from U-Net, (b) a temporal module added within the spatial module, which hinders the image controlnet, and (c) a temporal module appended after the spatial module, allowing for image control network functionality but failing to produce high-quality videos with text-only conditioning. In contrast, our MVB block, conditioned on both image and text, enables the image controlnet and can generate high-quality videos.

DreamBooth tuning [46]. Nonetheless, such an approach incurs repetitive and tedious fine-tuning for each individual visual conditional inputs, hindering it from efficiently scaling to wider applications. This inefficiency, as hinted by the prior work [34], stems from the fact that most pretrained text-to-video models are not able to condition both on images and text inputs. To overcome this issue, we introduce a decoupled multimodal cross-attention module to simultaneously condition the generation on both image and text inputs, thus facilitating to better control the visual appearance, while minimizing required fine-tuning efforts and unlocking zero-shot subject customized video generation.

distribution, thereby making direct integration with image ControlNet not feasible. In contrast, AnimateDiff [19] discards temporal convolution layers and only inserts temporal attention layers after spatial layers, as shown in Fig. 3(c). Such design preserves spatial feature distribution thus facilitating immediate reuse of image ControlNet. However, it depends exclusively on text conditions, which do not offer sufficient visual cues. Consequently, temporal modules have to learn extra spatial information for compensation, which diminishes the focus on temporal consistency, causing increased flickering and a decline in video quality.

In terms of geometric structure control, despite methods such as ControlNet [79] and T2I-Adapter [38] are developed to leverage depth, edge maps as visual conditions for image generation, analogous strategies for video synthesis remain indeterminate. Among the few existing attempts, VideoComposer [63] adds video-specific ControlNet modules to video diffusion models (VDMs) and subsequently re-trains the added modules from scratch, incurring substantial extra training overhead. In contrast, alternative methods [10, 81] reuse pre-trained ControlNet modules for images. However, they require adapting text-to-image models for video generation via frame propagation [16, 74] or cross-frame attention [29, 68], resulting in subpar temporal consistency compared to those based on VDMs.

To address these issues, we introduce MoonShot, a video generation model that consumes both image and text conditional inputs. The foundation of the model is a new backbone module for video generation, called multimodal video block (MVB). Specifically, each MVB highlights three main design considerations:

- • a conventional spatial-temporal module for video generation, which in order consists of a spatial convolution layer, a self-attention layer and a temporal attention layer that aggregates spatial features. Such a design allows reuse of pre-trained weights from text-to-image generation models without altering its spatial feature distribution, thus subsuming its generation quality.
- • a decoupled multimodal cross-attention layer that conditions the generation on both text and image inputs. These two conditions complement each other to guide the generation. In addition, image input offers reference visual cues, allowing temporal modules to focus on video consistency. This improves overall generation quality and

Building upon the aforementioned observations, our objective is to explore a model architecture that combines the strengths of both realms. Namely, we expect the model to be a high quality VDM that produces consistent video frames, while also able to directly leverage pre-trained image ControlNet to condition on geometry visual inputs. In this regard, we observe that as shown in Fig. 3(b), prior work [13, 60, 68, 83] typically insert temporal modules, such as temporal convolutions layers, in between of spatial modules, usually before self-attention and after spatial convolution layers. This design modifies the spatial feature

frame coherence, as evidenced experimentally;

• optionally, since spatial feature distribution is preserved, pre-trained image ControlNet modules can be immediately integrated to control the geometric structure of the generation, without needing of extra training overhead.

As a result, our model generates highly consistent videos based on multimodal inputs, and can further utilize geometry inputs, such as depth and edge maps, to control the compositional layout of the generation. Moreover, thanks to its generic architecture with versatile conditioning mechanisms, we show that MoonShot can be easily repurposed for a variety of generative applications, such as image animation and video editing. Qualitative and quantitative results show that MoonShot obtains superior performance on personalized video generation, image animation and video editing. When provided with a video frame as the image condition, MoonShot demonstrates competitive or better results with state-of-the-art foundation VDMs, validating the effectiveness of the model.

#### 2. Related Work

Text to Video Generation. Previous research in this field has leveraged a diversity of generative models, such as GANs [47, 49, 54, 55, 59], Autoregressive models [14, 25, 33, 53, 72], and implicit neural representations [51, 77]. Driven by the diffusion model’s significant achievements in image synthesis, a number of recent studies have explored the application of diffusion models in both conditional and unconditional video synthesis [1, 6, 20, 26, 28, 36, 40, 57, 57, 62, 67, 68, 73, 83]. Most existing approaches focus on conditioning a single modality. For instance, Imagen Video [24] and Make-A-Video are conditioned exclusively on text, while I2VGen-XL [80] relies solely on images.

In contrast, our method supports multimodal conditions with both image and text, enabling more precise control.

Video Editing and ControlNet. Beginning with Tune-AVideo [68], numerous works [7, 8, 16, 18, 29, 44, 61, 74] adapt the Text to Image (T2I) Model [45] for video editing. These methods introduce additional mechanisms like cross-frame attention, frame propagation, or token flow to maintain temporal consistency. While Dreammix [37] uses the VDM for video editing, it necessitates source video finetuning. In contrast, our method directly employs VDM for video editing, achieving temporally consistent videos without the need for fine-tuning or complex designs.

ControlNet [79] and T2I-Adapter [38] utilize structural signals for image generation. While ControlVideo [81] applies this approach with a T2I model for video generation, resulting in limited temporal consistency. Control-AVideo [10] integrates a temporal layer into both the base VDM and ControlNet, training them jointly. Gen-1 [12] concatenates structural signals with noise as the UNet input. However, without these signals, the base VDMs in Control-

A-Video and Gen-1 will fail. Videocomposer [63] incurs high training costs by adding a video controlnet after VDM training. Our method stands out as our VDM independently produces high-quality videos and directlu uses image ControlNets without extra training.

Generation Model Customization. Customizing large pre-trained foundation models [11, 17, 32, 46, 52, 65] enhances user-specific preferences while retaining robust generative capabilities. Dreambooth [46] leads the way in this field, though it requires extensive fine-tuning. IP adapter [76] and BLIP-Diffusion [34] achieve zero-shot customization using an additional image cross-attention layers and text-aligned image embeddings. Animatediff [19] pioneers subject customization in videos but needs additional Dreambooth training for the new subject.

Our method differs from IP-adapter in image domains by introducing a decoupled image-text cross-attention layer in the video block. We explore its effectiveness in facilitating high-quality and smooth video creation and further apply it to video tasks involving image animation and video editing. Unlike Animatediff, our approach enables zero-shot subject customization directly in videos.

Image Animation. This task aims to generate subsequent frames given an initial image. Existing methods [2, 30, 30, 35, 35, 39, 41, 70] have focused on specific domains like human actions and nature scenes, while recent advancements like I2VGen-XL [80], DynamicCrafter [69], and Videocomposer [63] target open-domain image animation with video diffusion models. However, they often struggle to maintain the appearance of the conditioning image.

In contrast, our approach, using masked condition and decoupled image-text attention, effectively aligns the first frame of the animated video with the provided image, ensuring a more accurate retention of its original identity. Our work is concurrent with DynamicCrafter [69] (Oct.,2023) and we make comparisons in Fig. 7.

#### 3. Model Architecture and Adaptations

Text-to-video latent diffusion models generate videos by denoising a sequence of Gaussian noises with the guidance of text prompts. The denoising network θ is usually a UNet-like model, optimized by a noise prediction loss:

0,y,ϵ∼N(0,I),t∼U(0,T) ∥ϵ − ϵθ(zt,t,y))∥22 , (1)

L = Ez

where z0 is the latent code of training videos from VAE encoder, y is the text prompt, ϵ is the Gaussian noise added to the latent code, t is the time step and ϵθ is the noise prediction by the model. In the following, we first introduce the MoonShot model architecture including its core component, multimodal video block. Then, we showcase the capabilities of the model by describing methods to repurpose the model for various video generation tasks, such as

Image Encoder

$#

#$#

Video

Image Feature

[Figure 57]

%#

#"#

|[Figure 58]|
|---|

[Figure 59]

Image Cross-Attn

|[Figure 60]|
|---|

|!(%<br><br>[Figure 61]|
|---|

|!%%|
|---|

DownBlock

ℰ

Latent Feature

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|!%&|
|---|

!(&

[Figure 64]

UpBlock

W! "

[Figure 65]

[Figure 66]

## …

…

## …

…

## …

|!('<br><br>[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|!%'|
|---|

[Figure 70]

Noise

Latents

Prediction

W"

%

Text Feature

Text Encoder

|A corgi running on the grass|
|---|

W$

$

UNet

Text Cross-Attn

Figure 4. The overall workflow and the structure of our decoupled multimodal cross-attention layers. In the training phase, we use the initial frame of the video as the image condition. For inference, the model accepts any image along with accompanying text.

Image animation noise

the ability of text-to-image generation, but also becomes incompatible with established techniques developed for textto-image models, such as ControlNet, making direct integration of these techniques infeasible.

|[Figure 71]<br><br>z"!|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]<br><br>z#!|
|---|---|---|---|---|
|z""<br><br>[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]<br><br>z""|
|1|0|0|0|0|

condition condition

Differently, we observe that the addition of temporal attention layers after the cross-attention layer does not significantly modify the spatial feature distribution, while contributing effectively for temporal feature aggregation. By freezing the spatial layers during training, we can reuse ControlNet to condition the generation on geometry visual inputs by broadcasting it along the temporal axis, as shown in Fig. 3(c). In particular, we use space-time attention similar to [4], where each patch attends to those at the same spatial location and across frames.

mask

Figure 5. Masked condition for image animation.

geometry controlled video generation, image animation and video editing.

##### 3.1. Multimodal Video Block

Our model architecture builds upon multimodal video blocks (MVB). There are three primary objectives underlying the design of this key module. First, we aim for the model to consistently produce video frames of high quality. Second, it is desired for immediate integration of pretrained image ControlNet. In this way, we can facilitate the use of geometric images for controlling the compositional layout without extra training. Third, the model is expected to accommodate multimodal text and image inputs for better visual appearance conditioning. To this end, each MVB consists of two groups of layers, spatialtemporal U-Net layers and decoupled multimodal cross-attention layers as detailed in order below.

Decoupled Multimodal Cross-attention Layers. Most existing video generation models use a cross-attention module to condition the generation on texts. Given fy the embedding of text prompts, diffusion models condition on it to enhance the U-Net features fx via cross-attention layers, where the query Q is obtained from U-Net features fx, key K and value V are from the text embedding fy. The crossattention operation is then defined as:

Q = WQ · fx; K = WK · fy; V = WV · fy; CrossAttention(Q,K,V) = softmax(QK

T

d ) · V,

√

(2) where Q ∈ RBN×H×W×C, K,V ∈ RBN×L×C, with B the batch size, N the number of frames, H the height, W the width and C the number of channels, L the number of text tokens, d the hidden size. Note that text embeddings are duplicated for video frames.

Spatialtemporal U-Net Layers. Typical U-Net in textto-image models consists in order of a spatial convolution layer (ResNet2D), a self-attention layer, and a crossattention layer that conditions the generation on texts. Prior work [5, 50, 60] adapts these models for video generation by introducing additional temporal convolution and attention layers. As shown in Fig. 3(b), the temporal convolution layer is usually inserted before each self-attention layer and after the spatial convolution layer. While this architecture may enhance temporal consistency, it alters the distribution of the spatial feature baked in the pre-trained text-to-image generation models. As a result, the model not only loses

There are two issues with this design. First, relying solely on text prompts usually proves inadequate in accurately describing highly customized visual concepts for the desired generation. Second, specifically for video generation, the absence of visual conditioning mechanism places excessive burden on the temporal attention layers. They

Given Image Text-Image to Video A bear astronaut

must simultaneously ensure the consistency across frames and also preserve high-quality spatial features, often resulting in compromises on both fronts with flickered lowquality videos.

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Wear a hat on the beach

To address these issues, we introduce decoupled multimodal cross-attention, where one extra key and value transformation are optimized for image conditions, denoted as KI,V I ∈ RBN×L×C. The attention is formulated as:

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Depth Map

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Image ControlNet

CrossAttention(Q,K,V) + CrossAttention(Q,KI,VI).

(3) This approach enables the model to effectively manage both the image and text conditions. Additionally, conditioning on visual cues allows the subsequent temporal modules to focus more on maintaining temporal consistency, resulting in smoother and higher-quality video outputs. With the additional image condition, the training loss is thereby reformulated as:

Wearing a glass

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Figure 6. Subject customized video generation results with the option of utilizing an image controlnet network or not.

0,y,y′,ϵ∼N(0,I),t∼U(0,T) ∥ϵ − ϵθ(zt,t,y,y′))∥22 ,

L = Ez

condition and incorporates visually appealing elements as described in the text, resulting in a smooth edited video.

(4) where y′ is the image condition.

Geometry Controlled Generation. Since our model preserves the spatial features of the pre-trained text-to-image models, we can directly integrate the image ControlNet for geometry conditioning. To achieve this, we attach pretrained ControlNet modules to the model. Then condition features for each frame is added to the corresponding feature maps via residues. Due to the careful design of spatialtemporal U-Net layers, we observe satisfactory geometry control effect without needing of video-specific fine-tuning.

##### 3.2. Adapting for Video Generation Applications

Masked Condition for Image Animation. With the additional conditioning on image input, our model is a natural fit for the task of image animation, which aims to transform an input image into a short video clip with consistent content. To enhance the content consistency, we adapt the mask-conditioning mechanism introduced by [5] for image animation. In particular, we use the first frame as an additional input condition for the U-Net. Apart from the original four latent channels, as illustrated in Fig. 5, we add five more channels to the U-Net’s input. Among them, four channels represent the replicated first frame latent z00, and one binary channel is used to denote the masked frames. This approach encourages that the identity of the subject in the animated video remains identical to that in the conditioning image. We observe that incorporating an extra image cross attention layer is essential for image animation. It helps significantly to prevent sudden changes in appearance and reduce temporal flickering, which are seen commonly in models driven merely by text conditions.

#### 4. Experiments

##### 4.1. Implementation Details

Our spatial weights are initialized using SDXL [42], which are fixed throughout the training process. Initially, following IP-Adpter [76], we train the image cross-attention layers using the LAION [48] dataset at a resolution of 512 × 320. Subsequently, we keep the spatial weights unchanged and proceed to train only the temporal attention layers. This training step utilizes the WebVid10M [3] dataset, each clip sampled 16 frames at a 512×320 resolution, with conditioning on video captions as text conditions and the first frame as image conditions. Further refinement is carried out on a set of 1000 videos from the InternVideo [64] collection, aimed at removing watermarks. We use 16 A100 40G GPUs for training. More details can be found in supplementary material.

Video Editing with Video Diffusion Models. DreamMix [37] shows that VDM can be repurposed for video editing, which however requires extensive fine-tuning. Our model is for general-purpose video generation, yet can be used for video editing without needing of fine-tuning. Specifically, for a selected encoded source video z0, we add Gaussian noise using the DDPM [23] forward process. Next, we employ diffusion directly with VDM, conditioned on both text and image. This process effectively replaces the subject in the original video with that from the image

##### 4.2. Human Evaluations

We follow Make-a-Video [50] to perform human evaluations on Amazon Mechanical Turk. In the realm of video editing tasks (Tab. 3), in line with FateZero [44] and

|Touch the helmet|
|---|

|Big waves|
|---|

|Robot is walking|
|---|

Given Image

Given Image First Frame

First Frame

Videocomposer

Videocomposer

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

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

I2VGen-XL I2VGen-XL

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

DynamiCrafter (VideoCrafter)

DynamiCrafter (VideoCrafter)

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

Ours

Ours

- Figure 7. Image animation. Comparing our method with I2VGEN-XL [80], DynamiCrafter [69], and VideoComposer [63], it stands out in its ability to control the animated video’s first frame to match the provided image exactly, maintaining the identity of the given image more precisely and animating the image according to text prompts. In contrast, I2VGen-XL and DynamicCrafter show relatively weaker identity preservation in their animated videos, and their response to text prompts is less effective.

DINO (First) DINO (Avg) CLIP-T (Avg) GT 0.781 0.644 – I2VGen-XL [80] 0.624 0.573 0.232 VideoComposer [63] 0.751 0.285 0.269 Ours 0.765 0.614 0.284

Model DINO CLIP-I CLIP-T

Non-Customized T2V 0.283 0.594 0.296 I2VGen-XL [80] 0.542 0.737 0.218 AnimateDiff [19]

0.582 0.784 0.243 300 finetune steps

Ours (zero-shot) 0.556 0.763 0.292 Ours (80 finetune steps) 0.624 0.802 0.292

Table 2. Image animation results. Our model shows better visual and textual alignment than competing methods.

Table 1. Subject customized video generation performance on Dreambooth dataset [46]. Higher metrics are better.

performance. If fine-tuned with as few as 80 steps, our approach further surpasses AnimateDiff by a significant margin, demonstrating the effectiveness of our model.

Render-A-Video [74], we direct annotators to identify the most superior outcomes from five methods, judging by three standards: 1) the precision of prompt-to-edited-frame alignment, 2) temporal coherence of the video, and 3) the overall quality. In addition, for ControlNet evaluations (see Tab. 5), we request annotators to judge whether the created video adheres to the control signal. Furthermore, regarding the text-to-video ablation studies (Tab. 6), we invite annotators to evaluate the overall video quality, the accuracy of textvideo alignment and motion fidelity.

Qualitative Results. As shown in Fig. 6, our model produces customized videos that align with both the subject of image condition and the text condition. Additionally, the image ControlNet can be directly integrated to realize control over geometric structures.

##### 4.4. Image Animation

Quantitative Results. To assess image animation capabilities, we select 128 video-text pairs from the Webvid evaluation set, covering diverse themes. We use DINO (First) to measure the similarity between the first frame of the animated video and the conditioning image, DINO (Average) for the average similarity across all video frames compared to the conditioning image, and CLIP-T for the overall alignment between text prompts and the animated video across frames. As shown in Tab. 2, our method outperforms others in all metrics, demonstrating superior identity preservation, temporal consistence and text alignment.

##### 4.3. Subject Customized Generation

Quantitative Results. To evaluate our method for subjectcustomized video generation, we perform experiments on the DreamBooth [46] dataset, which includes 30 subjects, each with 4-7 text prompts. We utilize DINO and CLIP-I scores to assess subject alignment, and CLIP-T for videotext alignment, calculating average scores for all frames. As shown in Tab. 1, our method achieves strong zero-shot customization, surpassing non-customized text-to-video (T2V) models by a large margin. Different from AnimateDiff, which requires repetitive re-training for new subjects, our method utilizes pre-trained decoupled multimodal attention layers, achieving zero-shot customization with compared

Qualitative Results. We compare our results qualitatively with I2VGEN-XL [80], DynamiCrafter [69], and VideoComposer [63] as shown in Fig. 7, it’s seen that the identity or appearance in animated videos from I2VGEN-XL and DynamiCrafter is different from the original image.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

|Models<br><br>|FID-vid (↓) FVD (↓) CLIP-T (↑)|
|---|---|
|NUWA¨ [67] CogVideo (Chinese) [25] CogVideo (English) [25] MagicVideo [83] Video LDM [5] Make-A-Video [50] ModelScopeT2V [60]<br><br>|47.68 - 0.2439 24.78 - 0.2614 23.59 1294 0.2631<br><br>- 1290 -<br>- - 0.2929<br><br><br>13.17 - 0.3049 11.09 550 0.2930|
|Ours|10.98 542 0.3068|

FateZero + Tune a Video

A blue Lamborghini running on the road, Van Gogh style.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Table 4. Quantitative comparisons on MSR-VTT [71].

Render a Video

A blue Lamborghini running on the road, Van Gogh style.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Pixel-MSE, the averaged mean-squared pixel error between aligned consecutive frames. Our method excels in temporal consistency and ranks second in frame editing accuracy. And our method achieves better human preference on all evaluation metrics (Sec. 4.2). In fact, prior methods typically utilize image models with frame propagation or crossframe attention mechanisms, which tend to yield worse temporal consistency compared to our approach. This demonstrates the clear advantage of using foundation VDMs for video editing, compared to those relying on image models.

[Figure 159]

Ours A car running on the road, Van Gogh style. +

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

- Figure 8. Visual comparisons with SOTA video editing methods.

Qualitative Results. As shown in Fig. 8, FateZero reconstructs the input sports car frame well but the result is not aligned with the prompt. Render-A-Video struggles to swap the jeep to a sports car, facing challenges with shape changes. Conversely, our method adeptly replaces the jeep with a sports car as specified in the conditioning image while also adhering to the text prompts.

Metric FateZ [44] Pix2V [7] T2V-Z [29] Render-A-V [74] Ours Fram-Acc (↑) 0.534 0.978 0.943 0.959% 0.976 Tem-Con (↑) 0.953 0.942 0.963 0.965 0.986 Pixel-MSE (↓) 0.092 0.256 0.091 0.073 0.064 User-Balance 4.4% 6.2% 7.4% 21.4% 60.6% User-Temporal 3.6% 2.0% 3.8% 18.2% 72.4% User-Overall 3.1% 3.1% 7.0% 24.6% 62.2%

Table 3. Quantitative comparisons and user preference rates for video editing task.

- 4.6. Text to Video Generation Since the spatial layers are frozen during training, we first generate an image according to the text prompt. The image is later combined with text for multimodal conditioned generation. We use MSR-VTT dataset [71] to evaluate quality of zero-shot generation, whose test set contains 2,990 videos of 320x240 resolution, accompanied by 59,794 captions in total. We compare our model with state-of-the-art methods as shown in Tab. 4. Our method achieves the best results across FID-vid [22], FVD [56], and CLIP-T [66], demonstrating better visual quality and text alignment.
- 4.7. Ablation Studies

While VideoComposer replicates the conditioning image as the first frame, subsequent frames show abrupt changes in appearance, as indicated by its high DINO (First) and low DINO (Average) scores in Tab. 2. This issue may stem from its limited capacity to extract visual cues. Our method, in contrast, utilizes multimodal cross-attention layers and condition masks, and excels by promoting the similarity between the first frame of the animated video and the conditioning image, maintaining more effectively appearance, and enabling animation in line with text prompts.

##### 4.5. Video Editing

Spatial temporal modules designs. We investigate the design of temporal modules that allows direct integration of the image ControlNet. As indicated in Tab 5, the design which inserts the temporal convolution within spatial modules (Fig.3b) alters the original spatial features, rendering the image ControlNet ineffective, even with fixed spatial (fs) weights during video training. Conversely, placing the temporal attention after all spatial modules in each block and fixing spatial weight during video training renders the image ControlNet feasible.

Quantitative Results. We compare with four video editing methods: FateZero [44], Pix2Video [7], Text2VideoZero [29], and Render-A-Video [74]. Notably, RenderA-Video and Text2Video-Zero employ customized models that incorporate ControlNet. In contrast, our approach utilizes the base VDM model without integrating ControlNet. Following the FateZero and Pix2Video, we utilize 72 videos from Davis [43] and various in-the-wild sources. We report three metrics in Tab. 3: Fram-Acc, a CLIP-based measure of frame-wise editing accuracy; Tmp-Con, assessing the cosine similarity between consecutive frames using CLIP; and

Impact of image condition on video consistency and quality. In Tab. 6, we explore the impact of mutlimodal

appearance flickering and suddenly change

two consecutive frames

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

only text cross-attn

An astronaut walking on the moon

two consecutive frames

appearance and smooth motion

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

decoupled

multimodal cross-attn

- Figure 9. Comparison between text-only and multimodal conditioned VDMs. We compare MoonShot with AnimateDiff-XL [19], which uses only text conditions. Despite trained on high-quality internal data, AnimateDiff-XL suffers appearance flickering and abrupt changes. In contrast, our multimodal-conditioned model trained on a public dataset shows improved temporal consistency and visual quality.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

only text cross-attn

A rabbit is turning around

decoupled

multimodal cross-attn

- Figure 10. Impact of multimodal conditions for image animation.

composed composed-fs decomposed-fs MVB rate 0% 13% 97% 97%

Table 5. Image controlent successful rates with different spatial temporal designs in VDMs.

FVD(↓) T-V alignment Motion-Fidelty Quality Text Only

Ours w/o image condition 602 8% 4% 0% Animatediff-XL 589 40% 12% 30%

MVB Ours 542 52% 84% 70%

Table 6. Impact of multi-modal condition for the VDM.

DINO (First) DINO (Avg) CLIP-T (Avg)

text only 0.264 0.262 0.285 + masked condition 0.760 0.296 0.210 + image condtion 0.638 0.562 0.282 + both 0.765 0.614 0.284

condition on video generation. We freeze the spatial layers and only train the temporal ones. Animatediff-XL also employs fixed SDXL spatial weights conditioned only by text and is included in this ablation. However, since it’s trained on internal high-quality data, we’ve also trained a model without image conditions on the same data used for our final model, to ensure a fair ablation. Text-Video (T-V) Alignment and Motion Fidelity are agreement scores from human evaluation (Sec 4.2). We find that using only text condition leads to weaker temporal consistency, motion fidelity, and visual quality compared to multimodal conditions. Fig. 9 also shows that with only text condition, the resulting video experiences significant temporal flickering and sudden appearance changes. These outcomes validate that when training temporal modules only, the additional image cross attention provides effective visual signals, thereby allowing the temporal module to focus on video consistency, leading to reduced flickering and improved video quality.

Table 7. Impact of conditions masks and decoupled cross attention for image animation. Higher metrics are better.

to produces the first animation frame that matches the conditioning image, yet cannot guarantee the temporal consistency, as suggested by a high DINO (First) score and low DINO (Avg) score. Adding the image condition improves temporal consistence and subject identify, as evidenced by the high DINO (Avg) score and confirmed in Fig. 10. With both masked condition and image condition, our model produces the first frame that highly preserves the conditioning image, as well as visually consistent animations.

#### 5. Conclusion

We present MoonShot, a new video generation model that conditions on both image and text inputs using the Multimodal Video Block (MVB). Our model stands out in producing high-quality videos with controllable visual ap-

Impact of image condition and masked condition on image animation. Tab. 7 shows that masked condition helps

pearance. In the meantime, our model is able to harvest the pre-trained image ControlNet to control over geometry without extra training overhead. The model offers a generic architecture and versatile conditioning mechanisms, which allows it to be easily adapted for various video generation tasks, such as image animation, video editing and subject-customized video generation, with superior generation quality than previous methods, revealing its great potential to serve as a foundation model for video generation research and application.

#### 6. Ethic

Our model operates with both image and text inputs, and similarly to the Blip-Diffusion [34], we aim to ensure a strong relevance of the generation samples to the supplied image. This image can originate from two sources: either directly uploaded by human users or generated by a textto-image model like DALL-E 3 or Stable Diffusion XL. When the image is produced by a text-to-image model, the bias, fairness, and toxicity levels of our model are inherent and comparable to those of the text-to-image model. Conversely, when the image is provided by the user, these aspects are mostly governed by the user, not our model. To mitigate harmful content that may result from inappropriate image/text inputs, we plan to implement an NSFW (Not Safe For Work) detector, which is open-sourced in diffusers library [58] and can be directly integrated into our model. This detector will strengthen the model governance layer to control the input contents, either from human users or from the text-to-image model, thereby reducing harmful components. In addition to the above measures, we are conducting red-teaming experiments to understand and minimize harmful generation contents as a result of red-teaming attacks. Note that by default, our model is designed to generate nontoxic elements as long as the provided inputs do not elicit harmful or biased content. Regardless, we are committed to eliminating any toxicity or biases in our model and we will not open source or make our model publicly available until the safety measures are properly in place.

#### References

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 3
- [2] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. arXiv preprint arXiv:1710.11252, 2017. 3
- [3] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,

2021. 5

- [4] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning (ICML), 2021. 4
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 1, 4, 5, 7
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 3
- [7] Duygu Ceylan, Chun-Hao Huang, and Niloy J. Mitra. Pix2video: Video editing using image diffusion. arXiv:2303.12688, 2023. 3, 7
- [8] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. arXiv preprint arXiv:2308.09592, 2023. 3
- [9] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023. 1
- [10] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models,

2023. 2, 3

- [11] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 3
- [12] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023. 1, 3
- [13] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XV, pages 89–106. Springer, 2022. 2
- [14] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. arXiv preprint arXiv:2204.03638,

2022. 3

- [15] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. arXiv preprint arXiv:2305.10474, 2023. 1
- [16] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arxiv:2307.10373, 2023. 2, 3
- [17] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning

- Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 3
- [18] Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, and Kevin Tang. Videoswap: Customized video subject swapping with interactive semantic point correspondence. arXiv preprint arXiv:2312.02087, 2023. 3
- [19] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3, 6, 8
- [20] William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos. arXiv preprint arXiv:2205.11495, 2022. 3
- [21] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 1
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 5
- [24] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1, 3
- [25] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3, 7
- [26] Tobias H¨oppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, and Andrea Dittadi. Diffusion models for video prediction and infilling. arXiv preprint arXiv:2206.07696, 2022. 3
- [27] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 1
- [28] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 3
- [29] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2, 3, 7
- [30] Yunji Kim, Seonghyeon Nam, In Cho, and Seon Joo Kim. Unsupervised keypoint learning for guiding classconditional video prediction. Advances in neural information processing systems, 32, 2019. 3

- [31] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 1
- [32] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3
- [33] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid. Ccvs: Context-aware controllable video synthesis. NeurIPS,

2021. 3

- [34] Dongxu Li, Junnan Li, and Steven CH Hoi. Blipdiffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint arXiv:2305.14720, 2023. 1, 2, 3, 9
- [35] Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, and Ming-Hsuan Yang. Flow-grounded spatial-temporal video prediction from still images. In Proceedings of the European Conference on Computer Vision (ECCV), pages 600– 615, 2018. 3
- [36] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In CVPR, 2023. 3
- [37] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 3, 5
- [38] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2, 3
- [39] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18444–18455, 2023. 3
- [40] Yaniv Nikankin, Niv Haim, and Michal Irani. Sinfusion: Training diffusion models on a single image or video. arXiv preprint arXiv:2211.11743, 2022. 3
- [41] Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan, and Xiaogang Wang. Video generation from single semantic label map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3733–3742, 2019. 3
- [42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 5
- [43] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 7
- [44] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fus-

- ing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023. 3, 5, 7
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 3
- [46] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. 2022. 1, 2, 3, 6
- [47] Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In ICCV, 2017. 3
- [48] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Theo Coombes, Cade Gordon, Aarush Katta, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: laion-5b: A new era of open large-scale multi-modal datasets. https: //laion.ai/laion-5b-a-new-era-of-openlarge-scale-multi-modal-datasets/, 2022. 5
- [49] Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Mostgan-v: Video generation with temporal motion styles. In CVPR, 2023. 3
- [50] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 1, 4, 5, 7

- [51] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. arXiv preprint arXiv:2112.14683, 2021. 3
- [52] James Seale Smith, Yen-Chang Hsu, Lingyu Zhang, Ting Hua, Zsolt Kira, Yilin Shen, and Hongxia Jin. Continual diffusion: Continual customization of text-to-image diffusion with c-lora. arXiv preprint arXiv:2304.06027, 2023. 3
- [53] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. Unsupervised learning of video representations using lstms. In ICML, 2015. 3
- [54] Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, and Sergey Tulyakov. A good image generator is what you need for high-resolution video synthesis. In ICLR, 2021. 3
- [55] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In CVPR, 2018. 3
- [56] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [57] Vikram Voleti, Alexia Jolicoeur-Martineau, and Christopher Pal. Masked conditional video diffusion for prediction, generation, and interpolation. arXiv preprint arXiv:2205.09853,

2022. 3

- [58] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 9

- [59] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. NIPS, 2016. 3
- [60] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1, 2, 4, 7
- [61] Wen Wang, kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 3
- [62] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023. 3
- [63] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 2, 3, 6
- [64] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 5
- [65] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 3
- [66] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 7
- [67] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation. In ECCV, pages 720–736. Springer, 2022. 3, 7
- [68] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022. 2, 3
- [69] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors.

2023. 3, 6

- [70] Wei Xiong, Wenhan Luo, Lin Ma, Wei Liu, and Jiebo Luo. Learning to generate time-lapse videos using multi-stage dynamic generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2364–2373, 2018. 3
- [71] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016. 7
- [72] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3

- [73] Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. Diffusion probabilistic modeling for video generation. arXiv preprint arXiv:2203.09481, 2022. 3
- [74] Shuai Yang, Yifan Zhou, Ziwei Liu, , and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In ACM SIGGRAPH Asia Conference Proceedings, 2023. 2, 3, 6, 7
- [75] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 1
- [76] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 3, 5
- [77] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. In ICLR, 2021. 3
- [78] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 1
- [79] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2, 3
- [80] Shiwei* Zhang, Jiayu* Wang, Yingya* Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qing, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. 2023. 3, 6
- [81] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 2, 3
- [82] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-tovideo diffusion models. arXiv preprint arXiv:2310.08465,

2023. 1

- [83] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 1, 2, 3, 7

