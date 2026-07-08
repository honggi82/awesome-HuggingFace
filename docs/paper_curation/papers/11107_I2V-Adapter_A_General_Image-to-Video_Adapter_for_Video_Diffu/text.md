# arXiv:2312.16693v4[cs.CV]26Jun2024

### I2V-Adapter: A General Image-to-Video Adapter for Diffusion Models

Xun Guo1∗† Mingwu Zheng2∗ Liang Hou2 Yuan Gao2 Yufan Deng2 Pengfei Wan2 Di Zhang2 Yufan Liu3 Weiming Hu3 Zhengjun Zha1 Haibin Huang2‡ Chongyang Ma2 1University of Science and Technology of China 2Kuaishou Technology 3Institute of Automation, Chinese Academy of Sciences

#### ABSTRACT

Text-guided image-to-video (I2V) generation aims to generate a coherent video that preserves the identity of the input image and semantically aligns with the input prompt. Existing methods typically augment pretrained text-to-video (T2V) models by either concatenating the image with noised video frames channel-wise before being fed into the model or injecting the image embedding produced by pretrained image encoders in cross-attention modules. However, the former approach often necessitates altering the fundamental weights of pretrained T2V models, thus restricting the model’s compatibility within the open-source communities and disrupting the model’s prior knowledge. Meanwhile, the latter typically fails to preserve the identity of the input image. We present I2V-Adapter to overcome such limitations. I2V-Adapter adeptly propagates the unnoised input image to subsequent noised frames through a cross-frame attention mechanism, maintaining the identity of the input image without any changes to the pretrained T2V model. Notably, I2V-Adapter only introduces a few trainable parameters, significantly alleviating the training cost and also ensures compatibility with existing community-driven personalized models and control tools. Moreover, we propose a novel Frame Similarity Prior to balance the motion amplitude and the stability of generated videos through two adjustable control coefficients. Our experimental results demonstrate that I2V-Adapter is capable of producing high-quality videos. This performance, coupled with its agility and adaptability, represents a substantial advancement in the field of I2V, particularly for personalized and controllable applications.

#### CCS CONCEPTS

• Computing methodologies → Motion processing; Computer vision; Neural networks.

#### KEYWORDS

Image-to-video generation; Diffusion models.

#### 1 INTRODUCTION

As an essential medium, video has risen to prominence on the Internet, offering users a rich and dynamic visual experience. The technology behind video generation has therefore garnered widespread attention due to its capacity to create innovative video content and its immense potential in the field of generative artificial intelligence. Recently, with the remarkable success of text-to-image (T2I) diffusion models such as Stable Diffusion [Rombach et al. 2022]

∗ Joint first authors. † Work done as an intern at Kuaishou Technology. ‡ Corresponding author: jackiehuanghaibin@gmail.com

and SDXL [Podell et al. 2023], there have been significant advancements in text-to-video (T2V) diffusion models [Blattmann et al. 2023; Chen et al. 2023c; Girdhar et al. 2023; Guo et al. 2024; Ho et al. 2022; Mullan et al. 2023; Wang et al. 2023b,a]. These models integrate temporal modules, such as temporal convolutions or temporal self-attention, onto T2I diffusion models to capture the relations between video frames. Typical approaches like AnimateDiff [Guo et al. 2024] decouples spatial modeling and temporal modeling to learn a motion prior from video data in a plug-and-play manner. By solely training the newly introduced temporal modules while maintaining the pretrained weights of T2I models, these approaches have achieved significant improvements in the field of T2V.

While T2V models have shown significant progress, crafting effective text prompt to generate the desired content remains a challenge, often necessitating complex prompt engineering [Witteveen and Andrews 2022]. This intricacy highlights the advantages of an alternative methodology, i.e., image-to-video (I2V) generation, which takes an image together with a text prompt as the input. This kind of method is often more intuitive and convenient, since the input image provides a direct and tangible visual reference for the video generation. However, translating specific images into coherent frame sequences in high fidelity necessitate not only generative capabilities but also fine-grained understanding of input images.

To tackle this unique challenge, many existing I2V models consider how to effectively inject the input image into the diffusion process during training, building upon the pretrained T2V models. One common approach involves channel-wise concatenation of the input image with the original noised inputs of the diffusion model’s U-Net [Blattmann et al. 2023; Chen et al. 2023a; Girdhar et al. 2023; Zeng et al. 2023]. Such a modification inevitably necessitates alterations to the original model’s structure and weights as the input space undergoes significant changes. Typically, this approach requires intensive training on a voluminous video-text dataset to mitigate this discrepancy. However, such an approach exacerbates the complexity and instability of the training process which could result in catastrophic forgetting and significantly hampers the performance and generalizability. Moreover, this modification limits their reusability with personalized T2I models [Hu et al. 2022; Ruiz et al. 2023] as well as their compatibility with control tools like ControlNet [Zhang et al. 2023a]. Notably, these models and tools, developed by a vast community of AI artists and enthusiasts, have been instrumental in numerous applications. Another common approach explored is the additional cross attention mechanism that relates the outputs of a pretrained image encoder (e.g., CLIP [Radford et al. 2021] and DINO [Oquab et al. 2023]) with the diffusion

Input Image Generated Video

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

w/ControlNetw/PersonalizedT2II2V-Adapter

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

- Figure 1: Our lightweight I2V-Adapter achieves high-quality and versatile image-to-video generation, without tuning any pretrained spatial or temporal modules. First row: I2V results of I2V-Adapter. Second row: results of I2V-Adapter combined with a personalized T2I model (CuteYukiMix). Third row: results of I2V-Adapter combined with ControlNet [Zhang et al. 2023a].

model [Chen et al. 2023c,a; Zhang et al. 2023b]. This strategy leverages the ability of cross attention mechanism to capture high-level dependencies and correlations. However, the outputs of these encoders often fall short in preserving the identity and fine details of the input image, leading to results that are less than satisfactory.

In this work, we present I2V-Adapter, a lightweight and plugand-play adapter designed to overcome aforementioned limitations for image-to-video generation. I2V-Adapter capitalizes on the prior knowledge and capabilities of the pretrained T2V model by feeding the unnoised input image in parallel with the subsequent noised frames to the model. Our approach propagates the input image’s identity information to the subsequent frames through a crossframe attention mechanism without altering any parameters of the spatial or motion modules (Stable Diffusion and AnimateDiff in our implementation). I2V-Adapter is integrated with the pretrained model via a trainable copy of query projector and a trainable zeroinitialized output projector, ensuring that the model’s initialization remains unaffected by the newly incorporated modules. Notably, I2V-Adapter introduces only a few trainable parameters, significantly alleviating the training cost while ensuring the compatibility with the open-source communities. Moreover, we propose a novel Frame Similarity Prior to balance the motion magnitude and stability in generated videos.

To summarize, our contributions are as follows:

- • We presentI2V-Adapter, anovellightweight adapter that achieves general image-to-video generation without training any preexisting spatial or motion modules.
- • We propose the Frame Similarity Prior, a novel concept that leverages the inherent similarity between video frames for imageto-video generation. This approach is designed to balance the stability with motion magnitude in generated videos, thereby enhancing the model’s controllability and the diversity of generated videos.
- • Our experiments demonstrate that our model can achieve superior results in image-to-video generation. Notably, our model requires significantly fewer trainable parameters (down to a minimum of 24M), as low as 1% of mainstream models.
- • We show that our I2V-Adapter exhibits versatility and compatibility, supporting a variety of personalized T2I models and controllable tools, thus enabling creative and custom applications.

#### 2 RELATED WORK

Visual content generation, notably influenced by the advancements in diffusion models [Ho et al. 2020; Song et al. 2020a,b], is emerging as a popular research area. This section provides an overview of closely related work from three perspectives: text-to-video (T2V)

generation, image-to-video (I2V) generation, and adapter techniques.

Text-to-video generation. T2V generation extends beyond the capabilities of T2I by producing a sequence of continuous frames, where diffusion models also play a pivotal role. Unlike T2I, T2V demands the synthesis of temporally coherent and visually consistent video content from textual descriptions, posing unique challenges in maintaining continuity and realism over time.

In the realm of T2V, early methods such as LVDM [He et al.

- 2022] and ModelScope [Wang et al. 2023b] have pioneered the integration of temporal modules with the 2D U-Net [Ronneberger et al. 2015] architecture. These approaches often involve redesign and comprehensive training of volumetric U-Net models, incorporating temporal aspects to cater to the dynamic nature of video content. This methodology signifies a critical step in handling the temporal complexity inherent in video generation. Another notable category of methods includes AnimateDiff [Guo et al. 2024] and Hotshot-XL [Mullan et al. 2023]. These models incorporate trainable temporal layers while retaining the foundational structure of existing T2I frameworks. This strategy leverages the established efficacy of T2I models, extending their capabilities to handle the additional dimension of time, thereby facilitating the generation of seamless video sequences. Additionally, approaches like Imagen Video [Ho et al. 2022] and LaVie [Wang et al. 2023a] adopt a cascaded structure with temporal and spatial super-resolution modules. This kind of architecture addresses both the spatial and temporal intricacies of video content, ensuring higher resolution and temporally coherent video generation.

To achieve multimodal compatibility and enhanced controllability within a unified framework, VideoComposer [Wang et al.

- 2023c] emerges as a significant contribution. This model integrates various modalities, enabling more controllable video generation, a crucial factor in applications demanding high fidelity and specific contextual alignment. SparseCtrl [Guo et al. 2023] employs an additional encoder on top of the base T2V model which injects temporally sparse conditions (e.g., sketch, depth, RGB image) into the diffusion process, enabling the base model to be used for various applications, including sketch-to-video, depth-guided generation, and image animation.

Image-to-video generation. I2V generation is characterized by its requirement to maintain the identity of the input image in the generated video. This poses a challenge in balancing the preservation of the image’s identity with the dynamic nature of video.

Mask-based approaches like MCVD [Voleti et al. 2022] and SEINE

- [Chen et al. 2023b] utilize video prediction techniques to achieve I2V generation. These methods are pivotal in maintaining the continuity of the input content across the generated video frames, ensuring a seamless transition from the static to the dynamic. Other innovations such as I2VGen-XL [Zhang et al. 2023b] and VideoCrafter1
- [Chen et al. 2023c] take a different approach by employing the CLIP [Radford et al. 2021] image encoder. They construct image embeddings as conditions, enabling a more nuanced interpretation of the initial image and its subsequent transformation into video format. Another set of techniques, including PixelDance [Zeng et al. 2023], Stable Video Diffusion [Blattmann et al. 2023], and VideoGen [Li et al. 2023], operate in the VAE [Kingma and Welling

2013] latent space. They concatenate image data within this space, effectively incorporating the input image into the video generation framework. This method has shown promise in maintaining the fidelity of the input image while enabling dynamic video creation. LAMP [Wu et al. 2023a], in particular, employs a firstframe-attention mechanism. This method focuses on transferring the information of the first frame to subsequent frames, ensuring that the essence of the initial image is reflected throughout the video sequence. While this method has shown promising results, its application is limited to fixed motion patterns in a few-shot setting. This limitation restricts its ability to generalize across a wide range of scenarios. AnimateAnyone [Hu et al. 2023] and MagicAnimate [Xu et al. 2023] both focus on creating human animations using pose sequences. AnimateAnyone utilizes a ReferenceNet for image encoding and a Spatial-Attention for identity consistency. It also utilizes a Pose Guider module to aid animation production. MagicAnimate employs a similar mechanism with an Appearance Encoder and ControlNet [Zhang et al. 2023a] for image and pose encoding. The design of both systems resembles a dual-tower U-Net structure.

In our work, we diverge from these singular approaches and introduce a method characterized by its plug-and-play nature. Our approach is designed to be compatible with various tools within the Stable Diffusion [Rombach et al. 2022] community, such as DreamBooth [Ruiz et al. 2023], LoRA [Hu et al. 2022], and ControlNet [Zhang et al. 2023a]. This compatibility enhances the flexibility and applicability of our method. Furthermore, our approach is designed for ease of training, making it accessible for a broader range of applications and users.

Adapter. In the field of natural language processing, adapters [Houlsby et al. 2019] have become a popular technique for efficient transfer and fine-tuning of pretrained large language models, such as BERT [Devlin et al. 2019] and GPT [Radford et al. 2018]. Adapters function by inserting small, task-specific modules between layers of pretrained models, thus enabling the learning of new task features without altering the original pretrained weights. The advantage of this approach is that it allows the model to retain foundational knowledge while rapidly adapting to specific tasks without the need for costly retraining of the entire network.

In the realm of diffusion models, a similar concept is being applied. ControlNet [Zhang et al. 2023a] and T2I-Adapter [Mou et al. 2023] are among the first to propose the use of adapters to integrate more control signals for diffusion models. IP-Adapter [Ye et al. 2023] specifically proposes a decoupled cross-attention mechanism for image conditioning, enabling the model to generate images that resemble the input image.

In our work, we introduce I2V-Adapter, a universal adapter that achieves efficient transfer from T2V to I2V with minimal overhead in terms of parameter number and works in a plug-and-play manner.

#### 3 METHOD

Given a reference image and a text prompt, our goal is to generate a video sequence starting with the provided image. This task is particularly challenging as it requires ensuring consistency with

the first frame, compatibility with the prompt, and preserving the coherence of the video sequence throughout.

#### 3.1 Preliminaries

Latent diffusion models. Diffusion models [Dhariwal and Nichol 2021; Ho et al. 2020; Song et al. 2020a,b] are a class of generative models that involves a fixed forward diffusion process and a learned reverse denoising diffusion process. Given a data x0 ∼ 𝑝data(x), the forward process gradually adds Gaussian noise 𝝐 ∼ N(0, 1) to the data to obtain a noised data, as shown as follows:

x𝑡 = 𝛼𝑡x0 + 𝜎𝑡𝝐, (1)

where 𝛼𝑡,𝜎𝑡 ∈ [0, 1] are two predefined functions of random time step 𝑡 ∼ U(1, . . .,𝑇), satisfying the variance preserving principle,

i.e., 𝛼𝑡2 + 𝜎𝑡2 = 1.

Diffusion models typically train a neural network 𝝐𝜃 (·) to learn the added noise, which is denoised during sampling to gradually generate new data. For any condition c (e.g., text, image, and audio), the training objective function is defined as

Ex0,c,𝑡,𝝐 ∥𝝐 − 𝝐𝜃 (x𝑡, c,𝑡)∥22 . (2)

min

𝜃

Latent diffusion models (LDM) such as Stable Diffusion [Rombach et al. 2022] integrate the diffusion process within the lowdimensionallatentspaceofavariational autoencoder (VAE) [Kingma and Welling 2013] instead of the high-dimensional pixel space for fast and efficient training and inference, which also yields comparable text-to-image generation performance. The denoising network of diffusion models typically adopts a U-Net based architecture [Ronneberger et al. 2015], where each block contains residual convolution layers [He et al. 2016], self-attention blocks [Vaswani et al. 2017], and cross-attention blocks [Vaswani et al. 2017].

Self-attention in LDM. The self-attention mechanism models the relationships between feature patches that generate the image. Selfattention has been demonstrated to control the layout, object shapes [Hertz et al. 2022; Tumanyan et al. 2023], and fine-grained texture details [Cao et al. 2023] of generated images. Self-attention can be defined by:

###### Q𝑖(K𝑖)𝑇 √

V𝑖, (3)

Attention(Q𝑖, K𝑖, V𝑖) = softmax

𝑑

where Q𝑖 = X𝑖W𝑄 represents the query from the input feature map X𝑖 through the query weight matrix W𝑄, K𝑖 = X𝑖W𝐾 represents the key from the input feature map X𝑖 through the key weight matrix W𝐾, and V𝑖 = X𝑖W𝑉 represents the value from the input feature map X𝑖 through the value weight matrix W𝑉 .

Cross-attention in LDM. To incorporate textual conditional information into networks for text-to-image tasks, cross-attention has been introduced in LDM to model the dependency between image feature patches and textual tokens. In the cross-attention mecha-

nism, the query Q𝑖 = X𝑖W𝑄cross stems from the image feature X𝑖, while the key Ktext = XtextW𝐾cross and the value Vtext = XtextW𝑉cross are derived from text embeddings Xtext of textual tokens encoded via the CLIP text encoder [Radford et al. 2021]. This attention mechanism enables different textual tokens to govern the appearance of

objects at different positions within the image. Cross-attention can be defined by the following equation:

Attention(Q𝑖, Ktext, Vtext) = softmax

Q𝑖K𝑇text

√

𝑑

Vtext. (4)

#### 3.2 I2V-Adapter

Temporal modeling with Stable Diffusion. The challenge in video generation lies in handling high-dimensional data to ensure coherence along the temporal dimension. Most existing video generation models incorporate motion modules on top of Stable Diffusion [Rombach et al. 2022] to model the temporal relationships in videos. In this paper, we are interested in AnimateDiff [Guo et al. 2024], a model initially developed for personalized T2V generation. It employs transformer blocks with positional encoding to capture the temporal relationships of the feature map along the dimension of frames. Due to this frame-wise attention mechanism, the motion module operates independently from the spatial modules, enabling it to generalize across multiple varieties of Stable Diffusion.

Inspired by this unique feature, we believe that the pretrained motion module can be seen as a universal motion representation, capable of being applied to more general video generation tasks, e.g., I2V generation, without any finetuning. Consequently, in this paper, we only modify spatial features and keep the pretrained AnimateDiff temporal weights fixed.

Adapter for attention layers. Unlike text-to-video (T2V) generation, image-to-video (I2V) generation not only requires adherence to the high-level semantic information of a textual prompt but also needs to maintain the content information of the reference image, specifically its fine-grained, low-level features. To this end, a common approach is to concatenate the reference image with the original noised inputs along the channel dimension before being fed into the U-Net [Ronneberger et al. 2015], thereby introducing reference image information at the input stage. Unfortunately, concatenation of additional input channels often necessitates retraining the entire U-Net network. This may disrupt the prior knowledge learned by the original Stable Diffusion model and also limits the flexibility to incorporate controllable or personalized models (e.g., ControlNet [Zhang et al. 2023a] and DreamBooth [Ruiz et al. 2023]) developed by the open-source community.

Unlike these approaches, we apply the first-frame-conditioned pipeline proposed by LAMP [Wu et al. 2023a], in which the first frame is always preserved as unnoised during both the training and inference stages, as detailed in Fig. 2. This pipeline keeps the space of the U-Net’s input and output unchanged, thereby reducing the trainable parameters while protecting the pretrained model weights.

However, this condition is weak since it is challenging for the motion module to ensure interaction with the first frame. To enhance the condition, most existing methods propose using a pretrained image encoder, such as CLIP [Radford et al. 2021] or DINO [Oquab et al. 2023], to extract features and then feed them into the crossattention module of U-Net. We follow this paradigm and integrate a pretrained Content-Adapter [Ye et al. 2023] into the cross-attention module. However, despite the improvement, these image encoders

Input Image

Input Prompt

⋯ ⋯

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

⋯

Text Encoder

Image Encoder

[Figure 28]

[Figure 29]

[Figure 30]

Spatial Block

I2V-

Self Attention

[Figure 31]

[Figure 32]

⋮

Adapter Temporal Block

[Figure 33]

⋯

+

[Figure 34]

Spatial Block

ContentAdapter

Cross Attention

[Figure 35]

Temporal Block

[Figure 36]

[Figure 37]

+ ⋯

[Figure 38]

Spatial Block

Spatial Block

Temporal Block

1

[Figure 39]

Spatial Block

′

⋮

Q K V

[Figure 40]

Temporal Block

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Spatial Block

Multi-Head Attention

[Figure 45]

Temporal Block

′

⋯ ⋯

O

[Figure 46]

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

I2V-Adapter

Output Video

- Figure 2: The overall architecture of our method. During both the training and inference procedures, we set the first frame as the input image and keep it noise-free. To capture fine-grained information, we add an additional branch alongside the self-attention block, i.e., I2V-Adapter. For each subsequent frame, the attention keys and values are shared with the first frame and remain unchanged. Only the query weights and zero-initialized output weights are trainable. This lightweight architecture allows us to infuse the input image into the diffusion process effectively.

usually only capture high-level semantic information and struggle to preserve lower-level features of the reference image.

In this paper, we propose a novel and effective approach to fuse low-level information into the first-frame-conditioned pipeline. Our key observation is that the self-attention plays a critical role in controlling the layout, object shapes, as well as fine-grained spatial details during the generation process [Hertz et al. 2022; Tumanyan et al. 2023]. Therefore, we add a cross-frame attention layer alongside the original self-attention layer to directly query the rich context information of the first frame encoded by the pretrained U-Net.

Specifically, for each frame 𝑖 ∈ {1, ...,𝑙}, where 𝑙 indicates the video length, we add a new cross-frame attention layer to integrate the input image features into the original self-attention process. Given the key and value features of the first frame K1 and V1, the

Reverse SDE

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

## …

Frame N

Pure Noise

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|add noise|
|---|---|
| | |

|[Figure 66]|[Figure 67]|
|---|---|
| | |

Frame 1 (input image)

Downgraded Frame 1

Reverse SDE with frame similarity prior

Figure 3: An illustration about Frame Similarity Prior. Instead of solving the reverse SDE from pure noise (the top row), we start the denoising process from corrupted input image (the bottom row). The frame similarity prior allows the model to generate videos with fewer steps and stabilizes the final result.

output of the new cross-frame attention is computed as follows:

Attention((Q𝑖)′, K1, V1) = softmax (Q𝑖)′(K1)𝑇

V1, (5)

√

𝑑

where (Q𝑖)′ = X𝑖W𝑄′ , and W𝑄′ represents a newly trainable query weight matrix. The parameters of W𝑄′ are initialized from the original W𝑄 in the corresponding layer. We then add the output to the original self-attention output, using the output projection matrices

- W𝑂 and W𝑂′ :
- X𝑖out=Attention(Q𝑖, K𝑖, V𝑖)W𝑂 + Attention((Q𝑖)′, K1, V1)W𝑂′ .

(6)

From a macroscopic perspective, the additional term on the right side borrows rich information from the first frame by querying its keys and using its values. Furthermore, the original self-attention module remains unchanged, ensuring that the T2I capability is not compromised. Besides, inspired by ControlNet [Zhang et al.

2023a], we zero-initialize the output projection layer W𝑂′ to ensure that the model starts as if no modifications have been made. In

total, our lightweight I2V-Adapter comprises only two trainable matrices: W𝑄′ and W𝑂′ , while leaving the pretrained weights of Stable Diffusion and the motion module completely unchanged, thereby ensuring that their original functionality remains optimal.

#### 3.3 Frame Similarity Prior

It is noteworthy that I2V generation is an ill-posed problem since the model is required to generate a full video from just a single reference image. Despite the tremendous data prior we leverage from pretrained spatial and temporal modules, the model still occasionally tends to produce unstable and distorted results.

To tackle this problem, we propose an additional frame similarity prior. Our key assumption is that, on a relatively low Gaussian noise level, the marginal distributions of the noised first frame and the noised subsequent frames are adequately close. At a high level, we assume that in most short video clips, all frames are structurally similar and become indistinguishable after being corrupted with a certain amount of Gaussian noise, see Fig. 3. Specifically, inspired

by SDEdit [Meng et al. 2022], to inject this prior into the diffusion process, instead of solving SDE [Song et al. 2020b] from pure noise x𝑇 , we start from an intermediate time 𝑡0 ∈ (0, 1] and sample each frame 𝑖 as x𝑖𝑡0 ∼N(𝛼𝑡0D(x10);𝜎𝑡20I).

To strike a balance between the realism and the strength of motion effects, we define the degradation operation D as a combination of masked Gaussian blur. Specifically, the operation is given by:

D(x) = M𝑝 ◦ x + (1 − M𝑝) ◦ GaussianBlur(x), (7)

where M𝑝 is a randomly generated binary mask with a probability 𝑝 dictating the presence of the original image, and (1−M𝑝) represents the complementary mask applied to the Gaussian blurred image. M and x share the same dimensions. The parameter 𝑝 controls the contribution of the original image. A smaller 𝑝 results in more information being retained from the input image, leading to a more static output video. Conversely, as𝑝 increases, the dynamic strength of the output is amplified, producing more motion.

Experimentally, we have found that this prior is critical to obtaining stable results, especially for relatively weak base T2I models such as Stable Diffusion V1.5 [Rombach et al. 2022].

- 4 EXPERIMENTS

- 4.1 Experimental Setup

Datasets. We employ the WebVid-10M [Bain et al. 2021] dataset for training our models. This dataset encompasses roughly 10 million video clips in a resolution of 336 × 596, each averaging 18 seconds. Each video clip is coupled with a text description of the video content. A notable complication with WebVid-10M is the inclusion of watermarks, which inevitably appear in the generated videos. In order to generate videos without watermarks and at higher resolutions, we supplement our training data with additional self-collected video clips that are free of watermarks in a resolution of 720 × 1280. Finally, we employ the MJHQ-30K [PlaygroundAI 2023] dataset as the benchmark for evaluation. This dataset encompasses 10 categories, each containing 3,000 samples, with each sample corresponding to an image and its text prompt.

Implementation details. We implement I2V-Adapter using two versions of AnimateDiff [Guo et al. 2024] based on Stable Diffusion V1.5 (SD1.5) [Rombach et al. 2022] and SDXL [Podell et al. 2023], respectively. In both versions, we freeze the parameters of the pretrained models and only fine-tune the newly added modules. Specifically, for the SD1.5 version of AnimateDiff, we start by training on the WebVid-10M [Bain et al. 2021] dataset for one epoch, where input sequences of 16 frames are center-cropped to a resolution of 256 × 256. Subsequently, in the second phase, we train for another one epoch on a self-collected dataset, cropping in the same manner to a resolution of 512 × 512. During both phases, we use the AdamW optimizer [Loshchilov and Hutter 2017] with a fixed learning rate of 1e-4. Considering our training strategy does not predict for the first frame (the input image), we calculate the loss only for the subsequent 15 frames. Due to the necessity of higher resolution inputs for training the SDXL version, the implementation of this version is carried out exclusively on the self-collected

dataset in the resolution of 512 × 512, with all other settings remaining unchanged. Most results in our paper are based on the SDXL version.

Evaluation metrics. We use four metrics to evaluate the results quantitatively, i.e., DoverVQA [Wu et al. 2023b], CLIPTemp [Liu et al. 2023], FlowScore [Liu et al. 2023], and WarpingError [Liu et al. 2023]. DoverVQA indicates the video technical score, CLIPTemp reflects the consistency between the generated video and the reference image by calculating the average pairwise cosine similarity between input image and each frame. FlowScore measures the magnitude of motion within the video. WarpingError assesses the accuracy of the video’s motion. FlowScore and WarpingError are obtained based on the optical flow computed via RAFT [Teed and Deng 2020]. Additionally, we conduct a user study involving 10 participants. We evaluate all models using 20 test cases and ask participants to rank the results according to various factors, including consistency with reference image, visual quality, and motion range. We report the results using Average User Ranking (AUR) as the user-preference metric. These metrics, reflecting model performance from various perspectives, enable a more comprehensive and impartial evaluation.

#### 4.2 Qualitative Results

For qualitative evaluations, we conduct comparisons in an opendomain scenario with four existing I2V models, which encompass three community open-sourced models SVD [Blattmann et al. 2023], I2VGen-XL [Zhang et al. 2023b] and DynamicCrafter [Xing et al. 2023] and two commercial models Gen-2 [RunwayAI 2023] and Pika [Pika 2023]. The results in Figure 8 illustrate that our model generates coherent and natural videos while ensuring consistency between the input image and subsequent frames in terms of image identity. Moreover, our results demonstrate superior responsiveness to the input text prompt. Figure 7 presents additional results achieved by our method. Lastly, the results from the user study (the last column of Table 1) also demonstrate that our method surpasses the comparative solutions.

#### 4.3 Quantitative Results

For quantitative analysis, we randomly sample 50 image-text pairs from MJHQ-30K dataset [PlaygroundAI 2023] to conduct evaluations. Table 1 shows that our model achieves the highest aesthetic score, surpassing all competing models in terms of consistency with the reference image. Additionally, our generated videos exhibit the largest range of motion, and a relatively low warping error, indicating our model’s ability to produce more dynamic videos while maintaining precise motion accuracy.

#### 4.4 Ablation Study

Model design. To validate the effectiveness of the key components in our method, we use the proposed full model (the last row of Figure 4) as a baseline and design three alternative experimental configurations for comparison: (1) removing the I2V-Adapter module; (2) removing the Content-Adapter module; and (3) removing both of them. The corresponding qualitative and quantitative results are shown in Figure 4 and Table 3 respectively. The results reveal that the absence of the I2V-Adapter leads to significant loss of

##### Table 1: Quantitative comparison and user study results on the MJHQ-30K Dataset. In each column, the best number is highlighted in bold, while the second best one is underlined.

Reusable to custom models

Compatible with controllable tools

Multimodal inputs

Trainable parameters

Method

DoverVQA ↑ CLIPTemp ↑ FlowScore ↑ WarpingError ↓ User Study ↓

Gen-2 [RunwayAI 2023] N/A N/A N/A 0.0800 0.9252 0.2490 12.5976 2.50

Pika [Pika 2023] N/A N/A N/A 0.0691 0.9021 0.1316 9.4587 4.45 DynamiCrafter [Xing et al. 2023] 1.10B 0.0848 0.9275 0.4185 14.3825 6.45

I2VGen-XL [Zhang et al. 2023b] 3.40B 0.0459 0.6904 0.6111 12.6566 6.55

SVD [Blattmann et al. 2023] 1.52B 0.0847 0.9365 0.6883 20.8362 4.35 I2V-Adapter (SD1.5) 24M 0.0873 0.9379 0.7069 9.8967 2.25 I2V-Adapter (SDXL) 204M 0.0887 0.9389 0.7124 11.8388 1.45

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Fullmodelw/oI2V-Adapterw/oContent-Adapterw/oContent-Adapter

w/oI2V-Adapter

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

##### Figure 4: Qualitative ablation study results on model design.

detail, as illustrated in the third row of Figure 4, where subsequent frames exhibit considerable changes in colors and finer details, underscoring the module’s role in enhancing identity consistency. Conversely, in the absence of the Content-Adapter, the underlying information of subsequent frames remains largely consistent with the first frame, yet these frames tend to manifest instability, indicating a lack of global information guidance and control. Finally, the outcomes where both modules are omitted show poor performance, both in terms of detail preservation and global stability.

Frame Similarity Prior. To verify the effectiveness of our Frame Similarity Prior (Section 3.3), we conduct experiments by adjusting the parameters𝑡0 and𝑝 (see Table 2). We observe that increasing the 𝑝 incrementally leads to a higher motion amplitude, as evidenced by the ascending values of the FlowScore metric. Simultaneously, the WarpingError metric exhibits a decreasing trend, indicating an enhancement in the stability of the generated video. Similarly, a gradual increase of the parameter 𝑡0 also yields a comparable effect.

Input Image Generated video with control sequence

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

##### Figure 5: Our results with ControlNet [Zhang et al. 2023a]. The pose control is shown as an inset at the bottom left for each frame.

Input Image Generated Video

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

w/BluePencil-XLw/AnimeArtXL

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

##### Figure 6: Our results with personalized T2I models [Ruiz et al. 2023].

These results indicate that our approach can effectively balance motion amplitude with the stability of the generated videos.

#### 4.5 Additional Applications

By keeping the parameters of the T2I models frozen, I2V-Adapter functions as a versatile plug-and-play module, allowing us to seamlessly integrate plugins from the open-source community. This section showcases two such applications. First, given an input image

##### Table 2: Ablation study on parameters 𝑡0 and 𝑝 of Frame Similarity Prior.

𝑡0 𝑝 DoverVQA ↑ CLIPTemp ↑ FlowScore ↑ WarpingError ↓ 1.0 1.0 0.0900 0.9393 0.8395 11.7421

- 0.9 1.0 0.0933 0.9445 0.4168 10.0771

- 0.8 1.0 0.0940 0.9454 0.2712 9.1360

- 0.7 1.0 0.0929 0.9485 0.1912 8.3508

- 0.6 1.0 0.0925 0.9449 0.1428 7.6332

- 1.0 0.9 0.0904 0.9379 0.9895 12.0888

- 1.0 0.7 0.0907 0.9389 0.9064 11.8403

- 1.0 0.5 0.0894 0.9391 0.8046 11.6175

- 1.0 0.3 0.0902 0.9404 0.7477 11.4993 1.0 0.1 0.0901 0.9415 0.6918 11.3017

##### Table 3: Quantitative results of ablation study on model design.

Model Design DoverVQA ↑ CLIPTemp ↑ FlowScore ↑ WarpingError ↓ w/o I2V-Adapter and Content-Adapter 0.0264 0.7013 1.5671 24.5928

w/o I2V-Adapter 0.0428 0.8841 0.2684 13.7764 w/o Content-Adapter 0.0813 0.9158 1.3533 19.5557

Full model 0.0887 0.9389 0.7124 11.8388

and a sequence of poses, our model, in combination with ControlNet [Zhang et al. 2023a], generates a corresponding video (Figure 5). Notably, our method adapts well to non-square resolutions different from the training resolution, underscoring its adaptability and generalization performance. Second, with a reference image featuring a personalized style, our model can create a style-consistent video based on personalized T2I models (e.g., DreamBooth [Ruiz et al. 2023], LoRA [Hu et al. 2022]), as shown in Figure 6.

#### 5 CONCLUSION AND FUTURE WORK

In this paper, we introduce I2V-Adapter, a lightweight and efficient solution for text-guided image-to-video generation. Our method retains the prior knowledge of pretrained models. Our method propagates the unnoised input image to subsequent frames through a cross-frame attention mechanism, thereby ensuring the output subsequent frames are consistent with the input image. Furthermore, we introduce a novel Frame Similarity Prior to balance stability and motion magnitude of generated videos.

Our method has demonstrated effectiveness in open-domain image-to-video generation, producing high-quality videos with a reduced number of trainable parameters. In addition, our method retains the prior knowledge of pretrained models and the decoupled design ensures compatibility with existing community-driven personalized models and controlling tools, facilitating the creation of personalized and controllable videos without further training.

Constrainedbypretrainedbasemodels and available high-quality video data, our method can only generate 16-frame, 512x512 videos, which restricts the creation of longer or higher-resolution content. In future work, we plan to incorporate frame interpolation and spatial super-resolution modules to generate videos of longer duration and higher resolution.

#### ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (No. 62192785, No. 62372451, No. 62372082, No. U1936204, No. 62272125, No. 62306312, No. 62036011, No. 62192782 and No. 61721004, No. U2033210), the Project of Beijing Science and technology Committee (Project No. Z231100005923046), the Beijing Natural Science Foundation No. L223003, the Major Projects of Guangdong Education Department for Foundation Research and Applied Research (Grant: 2017KZDXM081, 2018KZDXM066), Guangdong Provincial University Innovation Team Project (Project No. 2020KCXTD045).

#### REFERENCES

Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. 2021. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 1728–1738.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 22560–22570.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. 2023c. VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. arXiv:2310.19512 [cs.CV]

Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. 2023a. LivePhoto: Real Image Animation with Text-guided Motion Control. arXiv preprint arXiv:2312.02928 (2023).

Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2023b. Seine: Short-to-long video diffusion model for generative transition and prediction. arXiv preprint arXiv:2310.20700 (2023).

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186. https://doi. org/10.18653/v1/N19-1423

Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. In Advances in neural information processing systems. 8780–8794.

Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. 2023. Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. arXiv preprint arXiv:2311.10709 (2023).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933 (2023).

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In The Twelfth International Conference on Learning Representations.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221 (2022).

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. 2022. Prompt-to-Prompt Image Editing with Cross-Attention Control. In International Conference on Learning Representations (ICLR).

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Advances in neural information processing systems. 6840–6851.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.

Parameter-efficient transfer learning for NLP. In International Conference on Machine Learning. PMLR, 2790–2799.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR). OpenReview.net.

Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. 2023. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117 (2023).

Diederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013).

Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, and Jingdong Wang. 2023. Videogen: A reference-guided latent diffusion approach for high definition text-to-video generation. arXiv preprint arXiv:2309.00398 (2023).

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. 2023. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440 (2023).

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. 2023. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453 (2023).

John Mullan, Duncan Crawbuck, and Aakash Sastry. 2023. Hotshot-XL. https://github. com/hotshotco/hotshot-xl.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).

Pika. 2023. Pika. https://pika.art/. PlaygroundAI. 2023. MJHQ-30K Dataset. https://huggingface.co/datasets/

playgroundai/MJHQ-30K.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. CoRR abs/2307.01952 (2023).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 8748–8763.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. Technical Report. OpenAI. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-Resolution Image Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, (CVPR). IEEE, 10674–10685.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023. IEEE, 22500–22510.

RunwayAI. 2023. Gen-2: The Next Step Forward for Generative AI. https://research. runwayml.com/gen2. Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020a. Denoising Diffusion Implicit Models. In International Conference on Learning Representations (ICLR).

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020b. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).

Zachary Teed and Jia Deng. 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In ECCV (2) (Lecture Notes in Computer Science, Vol. 12347). Springer, 402–419.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2023. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1921–1930.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In

Advances in neural information processing systems.

Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. 2022. MCVD-masked conditional video diffusion for prediction, generation, and interpolation. In Advances in Neural Information Processing Systems. 23371–23385.

Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023b. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571 (2023).

Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2023c. VideoComposer: Compositional Video Synthesis with Motion Controllability. In Advances in Neural Information Processing Systems. 7594–7611.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. 2023a. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103 (2023).

Sam Witteveen and Martin Andrews. 2022. Investigating prompt engineering in diffusion models. arXiv preprint arXiv:2211.15462 (2022).

Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. 2023b. Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives. In International Conference on Computer Vision (ICCV).

Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, and Xiangyu Zhang. 2023a. Lamp: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769 (2023).

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. 2023. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190 (2023).

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2023. Magicanimate: Temporally consistent human image animation using diffusion model. arXiv preprint arXiv:2311.16498 (2023).

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. 2023. Make Pixels Dance: High-Dynamic Video Generation. arXiv preprint arXiv:2311.10982 (2023).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023a. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qing, Xiang Wang, Deli Zhao, and Jingren Zhou. 2023b. I2VGen-XL: High-Quality Imageto-Video Synthesis via Cascaded Diffusion Models. arXiv preprint arXiv:2311.04145 (2023).

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

The great wall of China at sunset, in the style of skyblue and beige, 32k uhd, colorful street murals.

Winter night. Wide shot of a small log cabin on fire. A very old man on his hands and knees on the ground.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

The cutest dog ever seen, outside playing in the mountains of France. Hyper cute, cuddly pet photography kittens cuteness overload.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Man hair short and dark on a ship and throws an anchor water a harbor sunshine island in the background life journey be romantic.

A girl is charging forward in the snow, portraying the martial arts in Chinese mythological stories, creating a tense and dynamic atmosphere.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Astronaut emerging colorful water and reaching for the stars, bright galaxy above the water, spaceman, dramatic lighting.

Modern digital street fighter versus screen, with elon musk as kazuya and jack as an ai robot going head to head.

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Pouring an orbital mechanics coffee. Huge triplepatty burger over a city, in the style of beeple.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Gigant warm is destroying a city in real time live tucker reaction. Gallons of water are pouring down on Hailee Steinfelds head, soaking wet, full body.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

2 characters, obligatorily 1 elephant and 1 giraffe, together side by side, making a cake, happy, exaggerated facial expression.

A cute Dutch pig wearing a future space suit,the background is the universe, tide play blind box, Pixar, pastel color, Busts, Natural light.

##### Figure 7: More videos of diverse categories generated by I2V-Adapter, with the text prompt at the bottom and the leftmost image as input.

Input Image Input Prompt

Input Image Input Prompt

[Figure 164]

[Figure 165]

Mario in a 3D Minecraft world, blocky world, Minecraft world, minecraft trees, sunset in the horizon, colourful lighting, centered, front view, red hat, white gloves, blue pants, moustache, ultrarealistic, unreal engine 5, HDR, Ray tracing, cinematic, depth of field, sharp focus, natural light, concept art, super resolution, cartoon style, pixar style, flat coloring.

Long exposure photograph of a ranch at night, featuring a big red barn and a large house amidst a dramatic thunderstorm, capturing the lightning illuminating the sky and the surrounding landscape. Shot with a Sony A7R IV, 1635mm f2.8 lens, ISO 100, 30s shutter speed, and a sturdy tripod to ensure sharpness in the captured lightning bolts.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

PikaDynamicCrafterI2VGen-XLI2V-AdapterGen-2

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

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

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

SVD

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

- Figure 8: We compare our results with those obtained by SVD [Blattmann et al. 2023], Gen-2 [RunwayAI 2023], Pika [Pika 2023], and DynamiCrafter [Xing et al. 2023], which are state-of-the-art I2V approaches. Results show that our model not only generates coherent and natural videos but also ensures consistency between the input image and subsequent frames. Moreover, our results demonstrate superior responsiveness to the input text prompt.

