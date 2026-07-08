## WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens

Xiaofeng Wang*1 Zheng Zhu*1 Guan Huang*1,2 Boyuan Wang1 Xinze Chen1 Jiwen Lu2 1GigaAI 2Tsinghua University

Project Page: https://world-dreamer.github.io

# arXiv:2401.09985v1[cs.CV]18Jan2024

Image to Video

[Figure 1]

[Figure 2]

[Figure 3]

WorldDreamer

Text to Video

“A polar bear is walking on ice.”

Video Inpainting

[Figure 4]

“A monkey is reading a book in the desert.”

Video Stylization

[Figure 5]

“Autumn style, a sailboat glides across the lake.”

Action to Video

[Figure 6]

[Figure 7]

[Figure 8]

Figure 1. WorldDreamer demonstrates a comprehensive understanding of visual dynamics in the general world. It excels in image-to-video synthesis, text-to-video generation, video inpainting, video stylization and even action-to-video generation.

### Abstract

language models, WorldDreamer frames world modeling as an unsupervised visual sequence modeling challenge. This is achieved by mapping visual inputs to discrete tokens and predicting the masked ones. During this process, we incorporate multi-modal prompts to facilitate interaction within the world model. Our experiments show that WorldDreamer excels in generating videos across different scenarios, including natural scenes and driving environments. WorldDreamer showcases versatility in executing tasks such as text-to-video conversion, image-tovideo synthesis, and video editing. These results underscore WorldDreamer’s effectiveness in capturing dynamic elements within diverse general world environments.

World models play a crucial role in understanding and predicting the dynamics of the world, which is essential for video generation. However, existing world models are confined to specific scenarios such as gaming or driving, limiting their ability to capture the complexity of general world dynamic environments. Therefore, we introduce WorldDreamer, a pioneering world model to foster a comprehensive comprehension of general world physics and motions, which significantly enhances the capabilities of video generation. Drawing inspiration from the success of large

*These authors contributed equally to this work. Corresponding author: Zheng Zhu, zhengzhu@ieee.org

### 1. Introduction

The next significant leap in artificial intelligence is expected to come from systems that possess a profound understanding of the dynamic visual world. At the core of this advancement are world models, crucial for comprehending and predicting the dynamic nature of our world. World models hold great promise for learning motion and physics in the general world, which is essential for video generation.

The early exploration of world models [19] primarily focus on gaming scenarios, which proposes a generative neural network model capable of learning compressed representations of spatial and temporal dynamics within game environments. Subsequent research in the Dreamer series [21–23] further validated the efficacy of world models across diverse gaming scenarios. Considering its structured nature and paramount importance, autonomous driving has become a forefront domain for the practical application of world models. Various approaches [31,32,49,50] are introduced to explore the efficacy of world models in autonomous driving scenarios. Furthermore, DayDreamer [52] has extended the application of world models to encompass real-world robotic environments, However, current world models are predominantly confined to gaming, robotics, and autonomous driving, lacking the capability to capture the motion and physics of the general world. Besides, relevant research in world models mainly relies on Recurrent Neural Networks (RNNs) [20–24,35,42,52] and diffusion-based methods [32,49,50] to model visual dynamics. While these approaches have yielded some success in video generation, they encounter challenges in effectively capturing the motion and physics in general world scenes.

In this paper, we introduce WorldDreamer, which pioneers the construction of general world models for video generation. Drawing inspiration from the successes of large language models (LLMs) [5, 12, 37, 38], we predict the masked visual tokens to effectively model the intricate dynamics of motion and physics embedded in visual signals. Specifically, WorldDreamer involves encoding images into discrete tokens using VQGAN [15]. We then randomly mask a portion of these tokens and utilize the unmasked tokens to predict the masked ones, a process integral to capturing the underlying motion and physics in visual data. WorldDreamer is constructed on the Transformer architecture [46]. Regarding the spatial-temporal priority inherent in video signals, we propose the Spatial Temporal Patchwise Transformer (STPT), which enables attention to focus on localized patches within a temporal-spatial window, facilitating the learning of visual signal dynamics and accelerating the convergence of the training process. Additionally, WorldDreamer integrates language and action signals through cross-attention, to construct multi-modal prompts for interaction within world model. Notably, compared to diffusion-based methods, WorldDreamer capitalizes on the

reuse of LLM infrastructure and benefits from optimizations developed over years for LLMs, including model scaling learning recipes. Besides, WorldDreamer exhibits a remarkable speed advantage, parallel decoding videos with just a few iterations, which is ∼3× faster than diffusionbased methods [3,10,48]. Therefore, WorldDreamer holds great promise for constructing a general world model from visual signals.

The main contributions of this paper can be summarized as follows: (1) We introduce WorldDreamer, the first general world model for video generation, which learns general world motion and physics. (2) We propose the Spatial Temporal Patchwise Transformer (STPT), which enhances the focus of attention on localized patches within a temporalspatial window. This facilitates easier learning of visual signal dynamics and expedites the training process. (3) We conduct extensive experiments to verify that WorldDreamer excels in generating videos across different scenarios, including natural scenes and driving environments. WorldDreamer showcases versatility in executing tasks such as text-to-video conversion, image-to-video synthesis, video editing, and action-to-video generation (see Fig. 1).

### 2. Related Work

#### 2.1. Video Generation

Currently, state-of-the-art video generation models are primarily classified into two categories: Transformer-based methods and diffusion-based methods.

Transformer-based methods. The Transformer-based video generation methods are derived from the general family of LLMs [5, 12, 37, 38]. Typically, these methods employ autoregressive prediction of the next token or parallel decoding of masked tokens to generate videos. Drawing inspiration from image generation techniques [11,13,40,54], VideoGPT [53] integrates VQVAE [45] with Transformerbased token prediction, enabling it to autoregressively predict visual tokens for video generation. Furthermore, GAIA-1 [31] integrates various modalities, including text descriptions, images, and driving actions, resulting in the generation of autonomous driving scenario videos. Unlike these autoregressive methods, some Transformer-based approaches [29, 47], draw inspiration from [8, 9, 14, 55], accelerating video generation through parallel decoding. In addition to these methods, VideoPoet [33] adopts video tokenizer [56] and generates exceptionally high-quality videos based on parallel decoding. The incorporation of Transformer models into video language models showcases their formidable zero-shot capability in handling various tasks during pretraining. Therefore, employing Transformerbased mask image models as the foundation for general world models emerges as a promising avenue.

##### Diffusion based methods. Compared to Transformer-

Image to Video

[Figure 9]

Image

[Figure 10]

[Figure 11]

Spatial Temporal Patchwise Transformer

[Figure 12]

× Maksed Visual Embeddings

Video stylization

Video

Mask

[Figure 13]

Visual Tokenizer

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Predicted Visual Tokens

Text to Video

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Text: ‘A lizard is reading a book in the desert.’ ‘A monkey is reading a book in the desert.’

Visual Decoder

Text Embeddings

LLM Encoder

Video Inpainting

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Action Embeddings

[Figure 34]

[Figure 35]

[Figure 36]

Action to Video

Action: Velocity/steering

MLP

[Figure 37]

[Figure 38]

[Figure 39]

Figure 2. Overall framework of WorldDreamer. WorldDreamer first converts images and videos into visual tokens, followed by a tokenmasking operation. Text and action inputs are encoded separately into embeddings, acting as multimodal prompts. Subsequently, STPT predicts the masked visual tokens, which are processed by visual decoders to enable video generation and editing in various scenarios.

immense potential for acquiring insights into motion and physics on a global scale. Initially, the exploration of world model [19] focuses primarily on gaming scenarios, presenting a generative neural network model capable of learning condensed representations of spatial and temporal dynamics within game environments. Subsequent research within the Dreamer series [21–23] affirmed the effectiveness of world models across a diverse array of gaming scenarios. Given its structured nature and critical significance, the domain of autonomous driving has emerged as a forefront application area for world models. Numerous approaches [31,32,49,50] have been introduced to assess the efficacy of world models in autonomous driving scenarios. Additionally, DayDreamer [52] has expanded the scope of world models to encompass real-world robotic environments. However, it is noteworthy that current world models primarily operate within the realms of gaming, robotics, and autonomous driving, lacking the capability to comprehensively capture the motion and physics of the general world.

based models, there has been extensive research employing diffusion-based models for video generation. VideoLDM [4] introduces a temporal dimension to the latent space of the 2D diffusion model and fine-tuned it using videos, effectively transforming the image generator into a video generator and enabling high-resolution video synthesis. Similarly, LVDM [26] explores lightweight video diffusion models, making use of a low-dimensional 3D latent space. Make-AVideo [43] also employs a pre-trained text-to-image model, eliminating the need for large-scale video training. Moreover, in the Imagen Video [27], a cascading video diffusion model is built upon the pretrained 2D diffusion model [27]. DiffT [25] and W.A.L.T [18] improve the video generation by utilizing a Transformer-based Diffusion network. Recently, Emu Video [17] and PixelDance [57] propose a two-step factorization approach for text-to-video generation, wherein the process is initially decomposed into textto-image conversion, followed by image-to-video synthesis. This methodology capitalizes on the effectiveness of contemporary text-to-image models, strategically directing the focus of the video diffusion model training toward the learning of motion dynamics. However, diffusion-based methods have difficulty integrating multiple modalities within a single model. Furthermore, these diffusion-based approaches struggle to produce results that accurately capture dynamics and motion.

### 3. WorldDreamer

#### 3.1. Overall Framework

The overall framework of WorldDreamer is depicted in Fig. 2. The initial phase involves encoding visual signals (i.e., images and videos) into discrete tokens using a visual tokenizer. These tokens undergo a carefully devised masking strategy before being processed by STPT. Meanwhile, textual and action signals are separately encoded into embeddings, which serve as multimodal prompts. STPT en-

#### 2.2. World Models

World models play a pivotal role in comprehending and predicting the dynamic nature of our environment, holding

gages in the pivotal task of predicting the masked visual tokens, which are then decoded by visual decoders, facilitating video generation and editing in multiple contexts.

× Spatial Temporal Patchwise Self-attention

3D Conv. Layers

Visual Embeddings

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

To train WorldDreamer, we construct triplets of VisualText-Action data, where the training supervision solely involves predicting masked visual tokens without any additional supervision signals. WorldDreamer also supports training without text or action data, which not only reduces the difficulty of data collection but also enables WorldDreamer to learn unconditional or single-condition video generation. At inference time, WorldDreamer can accomplish various video generation and video editing tasks: (1) For image-to-video, only a single image input is needed, considering the remaining frames as masked. WorldDreamer can also predict the future frames based on both single image condition and text condition. (2) For video stylization, a video segment can be input, with a random masking of certain pixels. WorldDreamer can alter the video style, such as creating an autumn-themed effect, based on both the input language. (3) For text-to-video, providing language input allows WorldDreamer to predict the corresponding video, assuming that all visual tokens are masked. (4) For video inpainting, a video segment can be input, with a manually masked region of interest. WorldDreamer can fill in the masked portion based on the input language and unmasked visual signals. (5) For action-tovideo, inputting the initial frame of a driving scene along with future driving commands allows WorldDreamer to predict future frames.

Spatial-wise Cross-attention

Multimodal Embeddings

Figure 3. Overall architecture of STPT. STPT first utilizes 3D convolution to aggregate visual embeddings. Then these embeddings are partitioned into several patches for spatial-temporal patchwise self-attention.In the following, spatial-wise cross attention is applied to facilitate feature interaction between visual embeddings and multimodal embeddings.

text or action embedding can be empty, enabling unconditional learning.

During training, and the optimizing objective is to predict the masked visual token conditioned on unmasked tokens and multimodal prompts:

LWorldDreamer = −log p(TˆV| TV,EM), (2)

where TˆV are masked visual tokens, and TV are unmasked visual tokens.

The subsequent subsections elaborate on the model architecture and the masking strategy.

STPT STPT leverages the foundation of U-ViT [30] while strategically enhancing its architecture to better capture the intricacies of spatial-temporal dynamics in video data. Specifically, STPT confines the attention mechanism within spatial-temporal patches. Additionally, to seamlessly incorporate multimodal information, spatial-wise cross-attention is employed to integrate multimodal embeddings. For input tokens TˆV, STPT transforms them into visual embeddings EV ∈ RN×h×w×C

#### 3.2. Model Architecture

Preliminery WorldDreamer utilizes VQGAN [16] to tokenize visual signals:

V = Fv(I), (1)

V by referencing a learnable codebook. The size of this codebook is set to 8193, exceeding the codebook size of VQGAN by 1, which enables compatibility with masked tokens. In each layer of STPT, as illustrated in Fig. 3, the visual embeddings are first processed through a 3D convolutional network. Then these embeddings are spatially partitioned into several patches EP ∈ RN×h/s×w/s×C

where I ∈ RN×H×W×3 are N frames of visual inputs. VQGAN Fv downsamples the resolution by 16×, which produces visual tokens TV ∈ RN×h×w (h = H4 ,w = W4 ). The VQGAN has a vocabulary size of 8192 and is trained with billions of images [41]. For text inputs, we employ the pretrained T5 [39] to map them into high-dimensional embeddings ET ∈ RK×C

T, where K is the sequence length and CT is the embedding channel. To be compatible with the feature learning in STPT, the text embeddings are repeated for N frames, and the embedding channel is mapped into CV. Furthermore, Multi Layer Perception (MLP) is utilized to encode action inputs, which generates action embeddings EA ∈ RN×C

V, where we empirically set patch stride s as 2. Subsequently, each patch embeddings are flattened for spatial-temporal patchwise self-attention:

EP = Gs−1(Fst(Gs(EP))), (3)

where Gs is the flatten operation that maps the embedding dimension to RNhw/s

V. The text embeddings and action embeddings are concatenated, producing the multimodal prompt embeddings EM ∈ RN×(K+1)×C

2×CV, and Gs−1 is the reverse operation. Fs is the standard self-attention. These patches are

V. Note that either

Diffusion Model: Denoising Process

Autoregressive Model: Autoregressively Predicting Next Token

WorldDreamer: Parallel Predicting Masked Tokens

Figure 4. Comparison between the inference schedule of diffusion-based methods, autoregressive methods and WorldDreamer. Diffusionbased methods usually require ∼30 steps to reduce noise, and autoregressive methods need ∼200 steps to iteratively predict the next token. In contrast, WorldDreamer parallel predicts masked tokens, achieving video generation in about 10 steps.

then concatenated and reshaped back to their original dimensions. In the following, the spatial-wise cross attention is applied, which facilitates feature interaction between visual embeddings and multimodal embeddings:

masking rate based on cosine scheduling. Specifically, we sample a random mask rate r ∈ [0,1] in each iteration, and

−1

2 tokens are masked in each frame. Note that we employ the same token mask across different frames. This decision is grounded in the similarity of visual signals between adjacent frames. Using different token masks could potentially lead to information leakage during the learning process. In comparison to an autoregressive mask scheduler, the dynamic mask schedule employed in our approach is crucial for parallel sampling at inference time, which enables the prediction of multiple output tokens in a single forward pass. This strategy capitalizes on the assumption of a Markovian property, where many tokens become conditionally independent given other tokens [8]. The inference process also follows a cosine mask schedule, selecting a fixed fraction of the highest-confidence masked tokens for prediction at each step. Subsequently, these tokens are unmasked for the remaining steps, effectively reducing the set of masked tokens. As shown in Fig. 4, diffusionbased methods usually require ∼30 steps to reduce noise, and autoregressive methods need ∼200 steps to iteratively predict the next token. In contrast, WorldDreamer, parallel predicts masked tokens in about 10 steps, presenting a 3× ∼ 20× acceleration compared to diffusion-based or autoregressive methods.

totally 2hwπ (1 − r2)

EV = Fc(Ev,EM), (4)

where Fc is the cross-attention operation that regards the frame number as batch size. After being processed through L layers of STPT, the feature dimensionality of EV is mapped to the codebook size of VQGAN. This enables the utilization of softmax to calculate the probability of each token, facilitating the prediction of masked visual tokens. Finally, cross-entropy loss is employed to optimize the proposed STPT:

Lce(TˆV,PSTPT( TV,EM)), (5)

where PSTPT( TV,EM) are visual token probabilities predicted by the STPT.

Notably, the proposed STPT can be trained jointly with videos and images. For image inputs, we simply replace the attention weight of Fs as a diagonal matrix [18]. Simultaneous training on both video and image datasets offers a substantial augmentation of the training samples, enabling more efficient utilization of extensive image datasets. Besides, the joint training strategy has significantly enhanced WorldDreamer’s capability to comprehend temporal and spatial aspects within visual signals.

### 4. Experiment

#### 4.1. Datasets

#### 3.3. Mask Strategy

We employ a diverse set of images and videos to train WorldDreamer, enhancing its understanding of visual dynamics. The specific data utilized in this training includes:

Mask strategy is crucial for training WorldDreamer, following [8], we train WorldDreamer utilizing a dynamic

Deduplicated LAION-2B [34] The original LAION dataset [41] presented challenges such as data duplication and discrepancies between textual descriptions and accompanying images. We follow [36] to address these issues. Specifically, we opted to utilize the deduplicated LAION2B dataset [34] for training WorldDreamer. This refined dataset excludes images with a watermark probability exceeding 50% or an NSFW probability surpassing 45%. The deduplicated LAION dataset was made available by [41], following the methodology introduced in [51].

WebVid-10M [1] WebVid-10M comprises approximately 10 million short videos, each lasting an average of 18 seconds and primarily presented in the resolution of 336 × 596. Each video is paired with associated text correlated with the visual content. A challenge posed by WebVid-10M is the presence of watermarks on all videos, resulting in the watermark being visible in all generated video content. Therefore, we opted to further refine WorldDreamer leveraging high-quality self-collected video-text pairs.

Self-collected video-text pairs We obtain publicly available video data from the internet and apply the procedure detailed in [3] to preprocess the obtained videos. Specifically, we use PySceneDetect [7] to detect the moments of scene switching and obtain video clips of a single continuous scene. Then, we filtered out clips with slow motion by calculating optical flow. Consequently, 500K high-quality video clips are obtained for training. For video caption, we extract the 10th, 50th, and 90th percentile frames of the video as keyframes. These key frames are processed by Gemini [44] to generate captions for each keyframe. Additionally, Gemini is instructed to aggregate these individual image captions into an overall caption for the entire video. Regarding that highly descriptive captions enhance the training of generative models [2], we prompt Gemini to generate captions with as much detail as possible. The detailed captions allow WorldDreamer to learn more finegrained text-visual correspondence.

NuScenes [6] NuScenes is a popular dataset for autonomous driving, which comprises a total of 700 training videos and 150 validation videos. Each video includes approximately 20 seconds at a frame rate of 12Hz. WorldDreamer utilizes the front-view videos in the training set, with a frame interval of 6 frames. In total, there are approximately 28K driving scene videos for training. For video caption, we prompt Gemini to generate a detailed description of each frame, including weather, time of the day, road structure, and important traffic elements. Then Gemini is instructed to aggregate these image captions into an overall caption for each video. Furthermore, we extract the yaw angle and velocity of the ego-car as the action metadata.

#### 4.2. Implementation Details

Train details WorldDreamer is first trained on a combination of WebVid and LAION datasets. For WebVid videos, we extract 16 frames as a training sample. For the LAION dataset, 16 independent images are selected as a training sample. Each sample is resized and cropped to an input resolution of 256 × 256. WorldDreamer is trained over 2M iterations with a batch size of 64. The training process involves the optimization with AdamW and a learning rate of 5 × 10−5, weight decay 0.01. To enhance the training and extend the data scope, WorldDreamer is further finetuned on self-collected datasets and nuScenes data, where all (1B) parameters of STPT can be trained. During the finetuning stage, the input resolution is 192×320, and each sample has 24 frames. WorldDreamer is finetuned over 20K iterations with a batch size of 32, and the learning rate is 1 × 10−5.

Inference details At inference time, Classifier-Free Guidance (CFG) [28] is utilized to enhance the generation quality. Specifically, we randomly eliminate multimodal embeddings for 10% of training samples. During inference, we calculate a conditional logit c and an unconditional logit u for each masked token. The final logits g are then derived by adjusting away from the unconditional logits by a factor of β, referred to as the guidance scale:

g = (1 + β)c − βu. (6)

For the predicted visual tokens, we employ the pretrained VQGAN decoder to directly output the video. Notably, WorldDreamer can generate a video consisting of 24 frames at a resolution of 192 × 320, which takes only 3 seconds on a single A800.

#### 4.3. Visualizations

We have conducted comprehensive visual experiments to demonstrate that WorldDreamer has acquired a profound understanding of the general visual dynamics of the general world. Through detailed visualizations and results, we present compelling evidence showcasing Worlddreamer’s ability to achieve video generation and video editing across diverse scenarios.

Image to Video WorldDreamer excels in high-fidelity image-to-video generation across various scenarios. As illustrated in Fig. 5, based on the initial image input, Worlddreamer has the capability to generate high-quality, cinematic landscape videos. The resulting videos exhibit seamless frame-to-frame motion, akin to the smooth camera movements seen in real films. Moreover, these videos adhere meticulously to the constraints imposed by the original image, ensuring a remarkable consistency in frame composition. It generates subsequent frames adhering to the constraints of the initial image, ensuring remarkable frame consistency.

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

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Figure 5. WorldDreamer excels in producing high-fidelity image-to-video generation across various scenarios.

“A brown bear is surfing in the rainbow.”

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

“A train passes through the valley, anime-style.”

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

“Evening streets of Japan, zoom in.”

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

“The waves crash against the beach.”

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Figure 6. WorldDreamer demonstrates proficiency in generating videos from text in diverse stylistic paradigms.

Text to Video Fig. 6 demonstrates WorldDreamer’s remarkable proficiency in generating videos from text across various stylistic paradigms. The produced videos seam-

lessly align with the input language, where the language serves as a powerful control mechanism for shaping the content, style, and camera motion of the videos. This high-

“Jellyfish swim in the sea.” “A Bear swim in the sea.”

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

“A lizard is reading a book in the desert.” “A monkey is reading a book in the desert.”

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Figure 7. WorldDreamer possesses an exceptional ability to achieve high-quality video inpainting.

“A sailboat glides across the lake.”

“Autumn style, a sailboat glides across the lake.”

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

“A small elephant is crossing the forest.”

“Autumn style, a small elephant is crossing the forest.”

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Figure 8. WorldDreamer excels in delivering high-quality video stylization capabilities.

lights WorldDreamer’s effectiveness in translating textual descriptions into visually faithful video content.

tions (e.g., controlling the car to make a left-turn or a rightturn).

Video Inpainting As depicted in Fig. 7, WorldDreamer exhibits an exceptional ability for high-quality video inpainting. By providing a mask outlining the specific area of interest and a text prompt specifying desired modifications, WorldDreamer intricately alters the original video, yielding remarkably realistic results in the inpainting process.

### 5. Conclusion

In conclusion, WorldDreamer marks a notable advancement in world modeling for video generation. Unlike traditional models constrained to specific scenarios, WorldDreamer capture the complexity of general world dynamic environments. WorldDreamer frames world modeling as a visual token prediction challenge, fostering a comprehensive comprehension of general world physics and motions, which significantly enhances the capabilities of video generation. In experiments, WorldDreamer shows exceptional performance across scenarios like natural scenes and driving environments, showcasing its adaptability in tasks such as text-to-video conversion, image-to-video synthesis, and video editing.

Video Stylization Fig. 8 shows that WorldDreamer excels in delivering high-quality video stylization. By supplying a randomly generated visual token mask and a style prompt indicating desired modifications, WorldDreamer convincingly transforms the original video, achieving a genuinely realistic outcome in the stylization process.

Action to Video WorldDreamer shows the ability to generate videos based on actions in the context of autonomous driving. As shown in Fig. 9, given identical initial frames and different driving actions, WorldDreamer can produce distinct future videos corresponding to different driving ac-

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Turn Left

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Turn Right

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Turn Left

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Turn Right

[Figure 129]

Figure 9. WorldDreamer excels in realizing the ability to generate videos based on actions in the context of autonomous driving.

### References

- Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023. 2, 5
- [9] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 2
- [10] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [11] Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, David Luan, and Ilya Sutskever. Generative pretraining from pixels. 2020. 2
- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 2
- [13] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. NeurIPS, 2021. 2
- [14] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. NIPS, 2022. 2
- [15] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 2

- [16] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 4

- [17] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021. 6
- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023. 6
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 6
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 3
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020. 2
- [6] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020. 6
- [7] Brandon Castellano. Pyscenedetect. Github repository,

2020. 6

- [8] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse:

- Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 3
- [18] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662, 2023. 3, 5
- [19] David Ha and J¨urgen Schmidhuber. Recurrent world models facilitate policy evolution. NeurIPS, 2018. 2, 3
- [20] Danijar Hafner, Kuang-Huei Lee, Ian Fischer, and Pieter Abbeel. Deep hierarchical planning from pixels. NeurIPS,

2022. 2

- [21] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019. 2, 3
- [22] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193, 2020. 2, 3
- [23] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023. 2, 3
- [24] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023. 2
- [25] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. arXiv preprint arXiv:2312.02139, 2023. 3
- [26] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 3
- [27] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [28] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [29] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2
- [30] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. arXiv preprint arXiv:2301.11093, 2023. 4
- [31] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 2, 3
- [32] Fan Jia, Weixin Mao, Yingfei Liu, Yucheng Zhao, Yuqing Wen, Chi Zhang, Xiangyu Zhang, and Tiancai Wang. Adriver-i: A general world model for autonomous driving. arXiv preprint arXiv:2311.13549, 2023. 2, 3
- [33] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan

- Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 2
- [34] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M Rush, Douwe Kiela, et al. Obelisc: An open web-scale filtered dataset of interleaved image-text documents. arXiv preprint arXiv:2306.16527,

2023. 6

- [35] Jessy Lin, Yuqing Du, Olivia Watkins, Danijar Hafner, Pieter Abbeel, Dan Klein, and Anca Dragan. Learning to model the world with language. arXiv preprint arXiv:2308.01399,

2023. 2

- [36] Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. arXiv preprint arXiv:2401.01808, 2024. 6
- [37] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. OpenAI, 2018. 2
- [38] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI, 2019. 2
- [39] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 2020. 4
- [40] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 2
- [41] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NIPS, 2022. 4, 6
- [42] Younggyo Seo, Danijar Hafner, Hao Liu, Fangchen Liu, Stephen James, Kimin Lee, and Pieter Abbeel. Masked world models for visual control. In CoRL, 2023. 2
- [43] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 3

- [44] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 6
- [45] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NeurIPS, 2017. 2
- [46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NIPS, 2017. 2
- [47] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. ICLR, 2023. 2

- [48] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [49] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, and Jiwen Lu. Drivedreamer: Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023. 2, 3
- [50] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. arXiv preprint arXiv:2311.17918, 2023. 2, 3
- [51] Ryan Webster, Julien Rabin, Loic Simon, and Frederic Jurie. On the de-duplication of laion-2b. arXiv preprint arXiv:2303.12733, 2023. 6
- [52] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In CoRL, 2023. 2, 3
- [53] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2
- [54] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 2
- [55] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In CVPR, 2023. 2
- [56] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 2
- [57] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: High-dynamic video generation. arXiv preprint arXiv:2311.10982, 2023. 3

