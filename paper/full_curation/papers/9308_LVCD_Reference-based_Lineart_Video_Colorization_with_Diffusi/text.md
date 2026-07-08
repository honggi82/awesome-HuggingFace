# arXiv:2409.12960v1[cs.CV]19Sep2024

## LVCD: Reference-based Lineart Video Colorization with Diffusion Models

ZHITONG HUANG, City University of Hong Kong, China MOHAN ZHANG, WeChat, Tencent Inc., China JING LIAO∗, City University of Hong Kong, China

Referenceframes

Fig. 1. Given a reference image and a sequence of lineart, our method aims to colorize the sketches to produce long temporal-consistent animation videos. The upper-left number indicates the frame index. Input frames: 1𝑠𝑡 row is from movie Princess Mononoke, 2𝑛𝑑 and 4𝑡ℎ rows are from movie Big Fish & Begonia, 3𝑟𝑑 row is from movie Mr. Miao.

We propose the first video diffusion framework for reference-based lineart video colorization. Unlike previous works that rely solely on image generative models to colorize lineart frame by frame, our approach leverages a large-scale pretrained video diffusion model to generate colorized animation videos. This approach leads to more temporally consistent results and is better equipped to handle large motions. Firstly, we introduce Sketch-guided ControlNet which provides additional control to finetune an image-to-video diffusion model for controllable video synthesis, enabling the generation of animation videos conditioned on lineart. We then propose Reference Attention to facilitate the transfer of colors from the reference frame to other frames containing fast and expansive motions. Finally, we present a novel scheme for sequential sampling, incorporating the Overlapped Blending Module and Prev-Reference Attention, to extend the video diffusion model beyond its original fixed-length limitation for long video colorization. Both qualitative and quantitative results demonstrate that our method significantly outperforms state-of-the-art techniques in terms of frame and video quality, as well as temporal consistency. Moreover, our method is capable of generating high-quality, long temporal-consistent animation videos with large motions, which is not achievable in previous works. Our code and model are available at https://luckyhzt.github.io/lvcd.

CCS Concepts: • Computing methodologies → Animation; Neural networks.

Additional Key Words and Phrases: Lineart video colorization, Diffusion Models, Animation

∗Corresponding author.

Authors’ addresses: Zhitong Huang, luckyhzt@gmail.com, City University of Hong Kong, Hong Kong SAR, China; Mohan Zhang, zeromhzhang@tencent.com, WeChat, Tencent Inc., Shenzhen, China; Jing Liao, jingliao@cityu.edu.hk, City University of Hong Kong, Hong Kong SAR, China.

1 INTRODUCTION

Animation, as a prominent form of media and entertainment, continues to captivate audiences worldwide. However, the production of animation remains a labor-intensive and time-consuming endeavor. Traditionally, artists manually sketch and colorize keyframes, leaving the in-between frames to be completed in accordance with the style and color palette of the keyframes. This filling process, while essential for preserving consistency and coherence in the animation, can be highly repetitive and tedious for the artists involved. Therefore, an automated method for colorizing the in-between frames is crucial for boosting efficiency and productivity in the animation industry.

While previous works have attempted lineart video colorization, their frameworks typically rely on image generative models that process one frame at a time, focusing solely on the coherence between neighboring frames. To maintain temporal consistency in the generated video, most of these works employ the previously generated frame as the new reference frame to produce subsequent frames. Unfortunately, this propagation can lead to significant error accumulation and artifacts, even within 10 consecutive samplings (an example is provided in 2𝑛𝑑 and 3𝑟𝑑 rows of Fig. 4). Another drawback of previous approaches is they all use frameworks based on Generative Adversarial Networks [Goodfellow et al. 2014] (GANs), which have limited generative ability compared to more recent architectures such as transformers [Vaswani et al. 2017] and diffusion models [Song et al. 2021]

Recently, generative models have garnered significant attention from researchers. Among these, diffusion models [Song et al. 2021] have emerged as particularly powerful tools for generating visually appealing content, including images and videos [Guo et al. 2024; He et al. 2022; Kang et al. 2024]. Leveraging the strengths of diffusion

models, we propose the first video diffusion framework for lineart video colorization based on reference frame. Following extensive evaluation, we select Stable Video Diffusion [Blattmann et al. 2023] (SVD) as the base model for our framework. This decision is driven by two key factors: a) the large amount of training data empowers SVD with robust generative capabilities to synthesize high-fidelity videos; b) the temporal video model facilitates the generation of temporally consistent videos, surpassing image generative models.

While SVD provides a robust foundation for our proposed task, we must address three significant challenges to adapt it accordingly: 1). Additional control of lineart sketch: The original SVD, as an image-tovideo model, only supports the conditioning from a reference image. Introducing additional control of the lineart sketch is essential to the model’s adaptation. 2). Adaptation to expansive motions: While SVD is limited to produce videos with subtle motions, animation clips often feature larger motions. Thus, modifications to the original SVD are necessary to accommodate expansive motions in animations. 3). Extension to long video: The original SVD is restricted to generating videos of fixed length, which does not meet the requirements for animation colorization. Therefore, extending the original SVD to generate long, temporally consistent videos is imperative.

We tackle the aforementioned challenges by introducing the first reference-based lineart video colorization framework with a diffusion model. This framework is adept at generating high-quality, large-motion, and temporally consistent animation videos guided by lineart sketches. To achieve this, we first extend the ControlNet [Zhang et al. 2023] to a video version, termed Sketch-guided ControlNet, incorporating additional lineart sketch control as a crucial guide for the animation’s layout and structure. Secondly, we introduce Reference Attention to replace the original spatial attention layers in SVD, facilitating long-range spatial matching between the first reference frame and consecutive generated frames. This modification enhances the model’s ability to colorize frames with large motions relative to the reference frame. Lastly, in contrast to the original SVD, which is trained to generate videos of fixed length (e.g., 14 frames) and thus inadequate for generating long videos with temporal consistency, we introduce a novel scheme for sequential sampling, including Overlapped Blending Module and Prev-Reference Attention, to enable the colorization of long animation.

The extensive experiments demonstrate the effectiveness of our method in generating long, temporally consistent animations with large motions of high fidelity, given a reference frame and a sequence of lineart. In summary, our contributions are three-fold:

- • We propose the first video diffusion framework for referencebased lineart animation colorization, harnessing the capabilities of a pretrained video diffusion model to generate long, temporally consistent animations of high quality.
- • We introduce Reference Attention for SVD, enhancing the model’s ability to generate animations with swift motions.
- • We design a novel sequential sampling mechanism, including the Overlapped Blending Module and Prev-Reference Attention, to extend the model to produce long animations with long-term temporal consistency.

2 RELATED WORKS

Lineart Image Colorization. Unlike natural image colorization, where the grayscale image retains an illuminance channel for colors, lineart exclusively contains structural information. This characteristic affords greater freedom and flexibility in lineart colorization. Traditional optimization-based methods for lineart colorization [Qu et al. 2006; Sýkora et al. 2009] typically involve users manually brushing desired colors onto specific regions. The emergence of deep neural networks has propelled lineart colorization into a new realm, where various techniques have been employed, including color hint points [Zhang et al. 2018], color scribbles [Ci et al. 2018], text tags [Kim et al. 2019], and even natural language [Zou et al. 2019]. Another significant approach is reference-based lineart colorization, wherein users only need to provide a reference cartoon image to colorize the lineart with similar colors and styles. The crux of this method lies in establishing correspondence between the lineart and the reference image. Sato et al. [Sato et al. 2014] proposed a matching technique based on segmented graph structures, while Chen et al. [Chen et al. 2022] introduced an active learning framework for matching, and Yu et al. [Cao et al. 2022] employed an attention-based network. The first diffusion-based framework for anime face lineart colorization, based on reference images, was proposed in [Cao et al. 2024]. Despite the advancements in lineart image colorization, these frameworks have limitations in producing videos without accounting for temporal coherence between colorized frames.

Reference-based Lineart Video Colorization. Compared to lineart image colorization, which involves various types of user interactions and conditions, reference frame is the most significant and reasonable approach for lineart video colorization, which can be directly reused for subsequent frames without human intervention. Several methods extend reference-based colorization to lineart videos. Zhang et al. [Zhang et al. 2021] trained a reference-based lineart image colorization framework using frame pairs from animation video clips. However, this method only preserves consistency with the reference frame, lacking temporal coherence between the generated frames. In [Thasarathan et al. 2019], previously colorized frames are used as new reference frames for subsequent frame generation to maintain short-term temporal coherence. Nevertheless, this may lead to error accumulation as the video length increases. Wang et al. [Wang et al. 2023] addressed this by using both the first reference frame and previously generated frames as references, potentially mitigating error accumulation. Yu et al. [Yu et al. 2024] proposed propagating colors from the reference frame to subsequent frames based on estimated optical flow, but refinement is still required afterward, leading to certain degree of error accumulation. In [Thasarathan and Ebrahimi 2019], a user-guided framework preserved temporal coherence by requiring users to provide dense and consistent color hint points. However, this method contradicts the initial motivation of automatic lineart video colorization due to its significant demand for user interaction and proficiency. All aforementioned works rely on image generative models, limiting the model’s ability to produce long, temporally consistent animations. Our framework, based on the video diffusion model, addresses this

limitation by generating long, temporally consistent animations through sequential sampling.

Video Interpolation. Unlike reference-based lineart video colorization, where only the first frame is provided, video interpolation frameworks aim to predict in-between frames given both the first and last frames. Building on a pretrained text-to-video diffusion model, SparseCtrl [Guo et al. 2023] introduced additional image conditions to perform video interpolation. SEINE [Chen et al. 2024] proposed a diffusion-based framework for video transitions using masked frame conditions, allowing for differences in content and style between the first and last frames. The concept of video interpolation is further adapted to cartoon animations, where a temporal constraint network is incorporated in [Shi et al. 2020]. Li et al. [Li

- et al. 2022] developed a framework for colorizing in-between lineart sketches based on optical flow. AnimeInterp [Siyao et al. 2021] interpolates a single middle frame through warping with predicted flows from segment-guided matching. To improve the perpetual quality of animation interpolation, EISAI [Chen and Zwicker 2022] proposed a forward-warping interpolation architecture to prevent line destruction and ghosting artifacts. More recently, ToonCrafter [Xing et al. 2024] introduced a sketch-guided video diffusion model for animation interpolation. Despite these advancements, the task of animation interpolation remains limited to generating a fixed number of intermediate frames, requiring users to provide extensive reference frames for colorizing long sequences of lineart sketches.

Video Diffusion Models. As advancements in diffusion-based image synthesis [Dhariwal and Nichol 2021; Kang et al. 2024; Kumari et al. 2023] continue, numerous diffusion-based frameworks for video synthesis have emerged. Some approaches involve training a video diffusion model with temporal layers from scratch [He et al. 2022; Zhang et al. 2024], while a more common method is to add temporal layers to pretrained image diffusion models and finetune them for video synthesis [Guo et al. 2024; Ma et al. 2024; Yin et al. 2023]. SVD [Blattmann et al. 2023] is one such framework, initiated from an image latent diffusion model [Rombach et al. 2022], and adapted to video synthesis with additional temporal layers including 3D convolution and temporal attention. In [Chen et al. 2023], structural controls, e.g., canny, depth, and HED, are further incorporated to guide the video generation by introducing ControlNet [Zhang et al. 2023] architecture to video diffusion models. Despite the existence of video diffusion models for generating both realistic videos [Ma et al. 2024] and animations [Guo et al. 2024], none performs lineart video colorization using a reference frame. Building upon SVD, a large-scale pretrained video diffusion model, we introduce the first diffusion-based framework tailored for this specific task.

3 PRELIMINARY

We introduce the preliminary of Stable Video Diffusion [Blattmann

- et al. 2023] (SVD), the backbone image-to-video model for our work, including the diffusion framework (EDM [Karras et al. 2022]) and Euler-step sampling utilized in SVD.

Stable Video Diffusion. We adopt SVD as our backbone model, composed of a Variational Autoencoder [Kingma and Welling 2014] (VAE) and a U-Net [Ronneberger et al. 2015], illustrated in Fig. 2. The VAE encoder maps input video frames into a lower-dimensional

latent space, and the VAE decoder reconstructs the latents back into frames, with temporal layers introduced to mitigate minor flickering artifacts. Then, the U-Net, initialized from the text-to-image Stable Diffusion [Rombach et al. 2022] (SD), is finetuned to denoise fixedlength sequences of latents (i.e., 14 frames), incorporating additional temporal attention and 3D convolutional layers. Originally, SVD was trained for the image-to-video task, where the first frame is provided to guide the denoising process.

EDM. In SVD, the denoising network is trained with EDM, a continuous-time diffusion framework. The denoiser 𝐷𝜃 is trained with denoising score-matching (DSM) loss, given by:

E(𝑥0,𝑐)∼𝑝𝑑𝑎𝑡𝑎(𝑥0,𝑐),(𝜎,𝑛)∼𝑝(𝜎,𝑛) 𝜆𝜎 ∥𝐷𝜃 (𝑥0 + 𝑛;𝜎,𝑐) − 𝑥0∥22 (1) where 𝑝(𝜎,𝑛) is the distribution of noise level 𝜎 and normal noise 𝑛, 𝜆𝜎 is a loss weighting function for different noise levels, and 𝑐

represents the conditioning signal (e.g., conditional frame in SVD). The denoiser 𝐷𝜃 receives the clean image from the outputs of the UNet 𝑈𝜃:

𝐷𝜃 (𝑥;𝜎,𝑐) = 𝑐𝑠𝑘𝑖𝑝(𝜎) ·𝑥 +𝑐𝑜𝑢𝑡 (𝜎) ·𝑈𝜃 𝑐𝑖𝑛(𝜎) ·𝑥;𝑐𝑛𝑜𝑖𝑠𝑒(𝜎),𝑐 (2)

where 𝑐𝑠𝑘𝑖𝑝(𝜎), 𝑐𝑜𝑢𝑡 (𝜎), 𝑐𝑖𝑛(𝜎) and 𝑐𝑛𝑜𝑖𝑠𝑒(𝜎) are EDM preconditioning parameters dependent on noise level. We adhere to the same training scheme used in SVD to finetune the model.

Euler-step Sampling. During sampling, we employ Euler-step to gradually obtain the clean image 𝑥0 from Gaussian noise 𝑥𝑇 as in SVD:

𝜎𝑡 − 𝜎𝑡−1 𝜎𝑡

𝜎𝑡−1 𝜎𝑡

𝐷𝜃 (𝑥𝑡;𝜎𝑡,𝑐) (3)

𝑥𝑡−1 =

𝑥𝑡 +

where 𝜎𝑡 represents the discretized noise level for each timestep 𝑡 ∈ [0,𝑇].

4 METHODOLOGY

We aim to design a video diffusion framework for reference-based lineart video colorization, capable of producing temporally consistent long sequences of animations with large motions. First, in Sec. 4.1, we discuss the model architecture, including the sketchguided ControlNet and Reference Attention, which enable the model to generate animations with fast and expansive movements guided by lineart sketches. After modifying the model architecture, we finetune it using animation videos to perform our task. During inference, we extend the original SVD to produce long, temporally consistent animations through sequential sampling, incorporating the Overlapped Blending Module and Prev-Reference Attention, as discussed in Sec. 4.2.

4.1 Model Architecture

An overview of our framework is shown in Fig. 2. Our objective is to colorize a sequence of lineart sketches 𝑆 = 𝑠[1:𝑁 ] = {𝑠1, ...,𝑠𝑁 } given a reference image 𝐼0, to produce a sequence of video frames 𝐼[1:𝑁 ] (currently we assume the video length equals to the input length 𝑁 of the model) with a diffusion-based framework:

𝑋0 = 𝐷𝜃 (𝑋𝑇,𝑆,𝐶) (4)

where 𝐷𝜃 is the denoiser, 𝑋𝑇 = 𝑥𝑇[1:𝑁 ] = {𝑥𝑇1, ...,𝑥𝑇𝑁 } is the initial noised latents sampled from Gaussian distribution, 𝐶 = [𝑥0] × 𝑁

| | |
|---|---|
| | |

Generatedvideoframes

- C2 SpatialConv2d

- C3 TempConv3d

#### SVD

Reflatent

Referencepath

Referencepath

| | |
|---|---|
| | |

### ...

C2 SA CA TC

Noisedlatent

SA Self-Attention

VAE Decoder

Referenceframe

CA Cross-Attention

...

C2 C3 SA CA TS TC

Noisedlatents

TS TempSelf-Attn

Videopath

...

VAE Encoder

...

Videopath

TC TempCross-Attn

Repeat

...

Outputlatents

Reflatents

Element-wiseadd

SpatialReferenceAttention

Concatenate

1 2 ... N

Q

Skipconnection

ControlNet

Finetune

...

Zero-Conv

Zero-Conv

1 2 ... N

KV

0 0 0

Frozen

Sketches

Fig. 2. Model architecture of sketch-guided ControlNet and Reference Attention. All frames are from Big Fish & Begonia.

is the conditional input of repeated reference latents which are concatenated to noised latents forming the 𝑖𝑡ℎ input as [𝑥0,𝑥𝑇𝑖 ], where 𝑖 = 1, ..., 𝑁. Here, 𝑥0 = E(𝐼0) is the encoded reference frame. For simplicity, we omit the Euler-step sampling and directly obtain the denoised video latents 𝑋0, which are then decoded into video frames 𝐼[1:𝑁 ] = D(𝑥0[1:𝑁 ]).

Sketch-guided ControlNet. Besides the reference image, another key condition is the lineart sketch, which is not supported in the original SVD. As shown in Fig. 2, we adapt the design from ControlNet [Zhang et al. 2023] to incorporate the sketch as an additional condition. Firstly, we duplicate the original UNet’s encoder, cloning all the layers, including temporal attention and 3D convolutional layers, along with their weights. Secondly, we introduce several zero-initialized convolutional layers to encode the lineart sketches and concatenate them to the input of the cloned encoder. Finally, the outputs of each layer of the ControlNet are added to the skip connections of the original U-Net decoder. During training, all the layers in ControlNet are finetuned to generate video sequences conditioned on both the reference image and the lineart sketches.

Reference Attention. Animation videos often feature fast and large motions due to their low frame rate and virtual nature, necessitating the model to perform long-range spatial matching between the reference frame and the frames to be colorized. However, the original video path of SVD struggles to meet these requirements for two main reasons. Firstly, the encoded reference latents are concatenated with the noised latents along the channel dimension to

form the video input [𝑥0,𝑥𝑇𝑖 ],𝑖 = 1, ..., 𝑁, as shown in Fig. 2. This spatial alignment of the two inputs prevents effective long-range

matching between them. Secondly, the temporal layers in the video path which interact between the 𝑁 inputs are one-dimensional, considering only the correlation between frames at the same spatial position, thereby neglecting the relationship between frames at different positions. These two factors limit the model’s ability to propagate color information effectively from the reference frame to distant frames with large motions.

As shown in Fig. 2, to enable long-range spatial matching, we propose to calculate global attention between reference latent and noised frame latent. We concatenate the encoded reference latent

with a noised latent to form the reference input [𝑥0,𝑥𝑇0] and feed it to a new reference path, where all temporal layers (temporal

Conv3d and temporal self-attention layers) are skipped. We then insert the intermediate features of the reference path to interact with the features of the video path within the spatial self-attention layers using Reference Attention:

Attn(𝑄𝑖,𝐾𝑖,𝑉𝑖) → Attn(𝑄𝑖, 𝐾𝑖,𝐾0′ , 𝑉𝑖,𝑉0′ ) (5)

where [., .] represents spatial concatenation, 𝐾0′ and 𝑉0′ are the key and value mappings for intermediate features of the reference input

[𝑥0,𝑥𝑇0] fed into the reference path, and 𝑖 is the index of the 𝑁 inputs going through the video path. As shown in Fig. 2-Spatial

Reference Attention, each frame performs attention not only with itself but also queries information from the reference input (labeled as orange blocks). Since the SVD model is trained to reconstruct the ground-truth latent 𝑥𝑖 for each input [𝑥0,𝑥𝑡𝑖] using the Eqn. (1), the input [𝑥0,𝑥𝑡0] to the reference path has also learned to reconstruct 𝑥0, thereby ensuring that the intermediate features of the reference path contain complete information about the reference latent 𝑥0. This modified Reference Attention calculates the global attention between the reference frame and the frames, enabling effective propagation of content to distant positions when large motions are engaged.

Finally, we use the loss in Eqn. (1) to finetune the modified network with sketch-guided ControlNet and Reference Attention. We update all the layers in ControlNet as well as the spatial and temporal self-attention layers in UNet, as illustrated in Fig. 2.

4.2 Sequential Sampling for Long Animation

With sketch-guided ControlNet and Reference Attention, our model can already generate high-quality animations of a fixed length 𝑁. For longer video sequences, we divide the video into segments and generate each segment sequentially. To avoid accumulated errors,

SpatialReferenceAttention

A

B

SpatialReferenceAttention

Reference input

Blended

Q 1

Q N-3

N-1 N N+1 N+2 ...

2 3

N

N-2

N-2 N-1

N-3

2N-4

...

Blending

N-2 N-1

2N-7

N

1 2 3

N-3 N-3

###### N-2 N-2

###### N-1 N-1

N

N-2

N-1

N-3

Latent input

KV

KV

x 1 0 Attention amplify

...

...

0 0 0

0

N

0

0 0 0

0

0

x 1 0

x 1 0

x 1 0

x 1 0

1stSegment:NormalSamplingwithRef-Attn

2ndSegment:SamplingwithOlapBlend&Prev-RefAttn

Fig. 3. Sequential sampling with Overlapped Blending and Prev-Reference Attention.

we consistently use the first reference frame 𝐼0 for samplings of all segments. However, this sampling scheme only ensures temporal consistency within the same segment. To introduce additional dependencies between segments, we utilize a certain number of overlapped frames. Specifically, we divide the video sequence 𝐼[1:𝐿] of length 𝐿 into segments indexed from 1 to 𝑁𝐿−−𝑜𝑜 , where 𝑜 is the number of overlapped frames. For instance, the 𝑛𝑡ℎ segment is 𝑉𝑛 = 𝐼[(𝑛−1)(𝑁−𝑜)+1:(𝑛−1)(𝑁−𝑜)+𝑁 ]. Based on the scheme of sequential sampling, we then introduce Overlapped Blending Module and Prev-Reference Attention to preserve the long-term temporal coherence across consecutive segments.

Overlapped Blending Module. After splitting the video into overlapped segments, the beginning 𝑜 frames of the segment 𝑉𝑛, are already generated in the preceding segment. To incorporate this information into subsequent segments, we use two methods. Firstly, we extend spatial blending [Avrahami et al. 2022], originally used for image inpainting, to temporal blending. As shown in Fig. 3, for the overlapped frames (highlighted in pink), we replace the intermediate denoising results of segment 𝑉𝑛 with those from the previous segment 𝑉𝑛−1 for all denoising timesteps 𝑡:

𝐷𝜃 (𝑋𝑡𝑛,𝑆𝑛,𝐶𝑛,𝑅) ← 𝐷𝜃 (𝑋𝑡𝑛−1,𝑆𝑛−1,𝐶𝑛−1,𝑅) (6)

Secondly, we further insert the contents of the previously generated frames via Reference Attention. As shown in Fig. 3-B, we first feed all previously generated results of the overlapped frames (in orange blocks) as reference input to the reference path as mentioned in Sec. 4.1, which captures complete information from these inputs. Then, the spatial Reference Attention of the overlapped frames is defined as:

Attn(𝑄𝑖, 𝐾𝑖, log𝛼 · 𝐾𝑖′ , 𝑉𝑖,𝑉𝑖′ ), 𝑖 ≤ 𝑜 (7)

where𝐾𝑖′ and𝑉𝑖′ are the mapped key and value of the previous result 𝑥𝑖 fed through the reference path with input [𝑥𝑖,𝑥𝑇𝑖 ]. The term log𝛼 is an amplifying factor that amplifies the attention weights of previous results by 𝛼 = 10.0 times.

Prev-Reference Attention. To effectively propagate the contents of the blended overlapped frames to distant frames, we further propose enhancing temporal propagation within the spatial selfattention layers. As shown in Fig. 3-B, we shift the original selfattention of non-overlapped frames to the left by three frames:

Attn(𝑄𝑖, 𝐾𝑖−3,𝐾0′ , 𝑉𝑖−3,𝑉0′ ), 𝑖 > 𝑜 (8) By enabling the non-overlapped frames to query information from the overlapped frames, whose contents are restored to previously generated results through the Overlapped Blending Module, we effectively preserve the content consistency across consecutive segments.

By incorporatingOverlappedBlending Module and Prev-Reference Attention, we guarantee long-term temporal coherence for extended long video synthesis.

5 IMPLEMENTATION DETAILS

Dataset. We train our model with six animation movies directed by Hayao Miyazaki, comprising My Neighbor Totoro, The Secret World of Arriey, Whisper of the Heart, Ponyo, Kiki’s Delivery Service, and The Wind Rises. For each movie, we first quantize the RGB values into 1,000 colors, then apply a threshold of 30.0 on the bin-wise RMSE between the color histograms of consecutive frames to detect scene switches. We further exclude video clips with lengths smaller than 15 frames (the minimum required input length) and larger than 200 frames (which typically indicates an undetected scene switch). The final filtered training set consists of 6,472 video clips with an average length of 58 frames. We then utilize the method in [Chan et al. 2022] to extract the lineart sketch for each frame.

Training. During training, we input 𝑁 +1 frames per batch, where 𝑁 = 14 corresponds to the length of the original SVD input sequence, with an additional frame for Reference Attention. For each video clip with frames {𝐼1, ...,𝐼𝐿} of length 𝐿 in the training dataset, we split it into 𝐿 − 𝑁 − 1 training sets indexed from 1 to 𝐿 − 𝑁, where the 𝑘𝑡ℎ set consists of a sequence of consecutive frames {𝐼𝑘+1, ...,𝐼𝑘+𝑁 } and a group of candidate reference frames {𝐼1, ...,𝐼𝑘}. For each training step, a reference frame is randomly selected from the candidate set to predict the 𝑁 consecutive frames, allowing for variable and random distances between the reference and predicted frames, which adapts to inference scenarios.

We finetune the network using the loss function in Eqn. (1), updating all layers in ControlNet and the spatial and temporal selfattention layers in UNet. During training, the noise level 𝜎 is sampled from log𝜎 = N(1.0, 1.6). Due to limited computing resources, we train at a resolution of 576×320, as the original SVD is pre-trained with a resolution of 576 × 320 and then finetuned to 1024 × 576. We use a batch size of 32 with a learning rate of 5.0 × 10−5, and train for 18,000 steps over 80 hours on four NVIDIA RTX A6000 GPUs.

Sampling. During sampling, we utilize 𝑜 = 4 overlapped frames. For the noise scale, we set 𝜎𝑚𝑖𝑛 = 0.002 and 𝜎𝑚𝑎𝑥 = 700, and then linearly decrease 𝜎𝑡1/𝜌 from 𝜎𝑚𝑎𝑥1/𝜌 to 𝜎𝑚𝑖𝑛1/𝜌 as 𝑡 decreases from𝑇 to

- 1, where 𝜌 = 7. The denoising is performed for 𝑇 = 25 steps using the Euler step specified in Eqn. (3). We apply Overlapped Blending and Prev-Reference Attention for all denoising steps to enhance temporal consistency. The average inference time is approximately
- 2 seconds per frame on a single NVIDIA RTX A6000 GPU.

6 EXPERIMENTAL RESULTS

- 6.1 Experimental Setup

Test Dataset. We choose four movies directed by Hayao Miyazaki, namely Howl’s Moving Castle, Porco Rosso, Princess Mononoke, and Spirited Away, for our test set, labeled as Similar Testset which exhibit similar artistic styles but different content from the training dataset. Additionally, to assess the generalization ability of our model across diverse animation styles and contents, we select three movies, including Big Fish & Begonia, Mr. Miao, and LuoXiaoHei, produced by other directors, denoted as General Testset. We segment the videos into clips and extract the lineart sketches using the same methods as employed for the training dataset. For each testset, we evenly select 1,000 video clips for evaluation, with an average length of 59 frames. We measure the average movements of the two testsets with optical flow in resolution 256 × 256, where static positions are excluded. We found that 55% of the clips have average motions larger than 5 pixels and 28% of them exceed 10 pixels.

Evaluation Metrics. We evaluate the quality of the colorized animations across four aspects: 1). Frame & Video Quality: We use FID [Heusel et al. 2017] and FVD [Unterthiner et al. 2019] to evaluate the frame and video quality of the generated videos, respectively. 2). Frame Similarity: Since the animation is generated conditioned on the lineart sketches and the first reference frame, both extracted from the original animation, we measure the similarity between the generated frames and the original animation frames using PSNR, LPIPS, and SSIM [Wang et al. 2004]. 3). Sketch Alignment: To evaluate whether the generated frames align with the structure of the input lineart sketches, we extract the sketches of the generated frames and calculate the Euclidean Distance Map [Danielsson 1980] (ED Map), which measures the distances from each pixel to its nearest sketch. Subsequently, we compute the Euclidean Distance Map Difference (EDMD) in terms of RMSE, which indicates the average pixel shifts compared to the input sketches. 4). Temporal Consistency: We define the Temporal Consistency (TC) as:

TC =

∥𝐼𝑔𝑡→𝑡+1 − 𝐼𝑔𝑡+1∥2 / ∥𝐼𝑔𝑡+1∥2 ∥𝐼𝑡→𝑡+1 − 𝐼𝑡+1∥2 / ∥𝐼𝑡+1∥2

(9)

where 𝐼𝑔𝑡 is the 𝑡𝑡ℎ frame in the generated video, 𝐼𝑡 denotes the original frame, and 𝐼𝑡→𝑡+1 represents the warped frame 𝑡 + 1 from frame 𝑡. Here, optical flow predicted from the original animation with RAFT [Teed and Deng 2020] is utilized for both warpings in the original and generated frames. For all the metrics, we resize the frames to 256 × 256 and normalize the pixel value to [0.0, 1.0] for calculation.

- 6.2 Comparisons on Reference-based Works

We compare our proposed method with two existing reference-based lineart video colorization frameworks: ACOF [Yu et al. 2024] (an optical-flow-based method) and TCVC [Thasarathan et al. 2019] (an image-to-image framework), both of which are GAN-based image model. Since there is no widely accepted benchmark dataset for lineart video colorization, to ensure a fair comparison, we utilize our dataset to generate frame pairs for training ACOF and TCVC using their official code. For both methods, we evaluate two versions:

the original version, Prev Sample, updates the reference frame to the previously generated frame, while the modified version, First Sample, consistently uses the first frame as the reference. Given the absence of diffusion-based frameworks for our task, we employ an image ControlNet [Zhang et al. 2023] with AnythingV3 [AnythingV3 2023], finetuned from Stable Image Diffusion cartoon images and lineart controls. Additionally, we utilize Reference-only [Mikubill 2023] during sampling to provide reference frames as guidance. We further select an animation interpolation work EISAI [Chen and Zwicker 2022] and a diffusion-based video interpolation work SEINE [Chen et al. 2024] for comparison. Since both methods interpolate between colorized reference keyframes at a regular interval, we first apply the ControlNet + Reference-only method to colorize keyframes for every 13 frames before applying EISAI and SEINE to interpolate the remaining frames.

Qualitative Comparison. In Fig. 4, we present colorized frames from three animation clips. In the 1𝑠𝑡 example, both ACOF and TCVC exhibit severe accumulated artifacts when using the original version of Prev Sample. In the 2𝑛𝑑 and 3𝑟𝑑 examples, even with First Sample, both methods struggle to generate frames with significant movements compared to the reference frame, resulting in a ghosting effect at the moving positions. Hence, both methods are prone to certain artifacts, with either Prev or First Sample. This suggests that previous CNN-based methods struggle to effectively colorize long sequences of lineart sketches with substantial motions. Regarding the image ControlNet plus Reference-only methods, they can generate frames successfully only when the movement range is limited. The Reference-only technique, applied solely during inference and not trained with the model, may incorrectly interpret the correspondence from the reference frame when the object displacement is too large, as seen in the red hair of the girl in the 1𝑠𝑡 example and the distorted elder in Frame 60 and 70 of the 3𝑟𝑑 example. The experiment highlights the limitations of the image diffusion model in fulfilling our task. For interpolation methods, EISAI exhibits ghosting effects, while SEINE generates significant noise which blurs the frames (zoom-in on the figure is suggested to note the artifacts). This suggests that neither method can be adapted to our task, even provided with keyframes generated by the CNet + Refonly method.

Our method, leveraging a video diffusion model with long-range spatial matching through Reference Attention, is capable of producing long temporal-consistent animations featuring large motions. For instance, in all the examples, when the sprite and the characters change positions, our method accurately locates the correct correspondence and effectively colorizes them. Moreover, with the sequential sampling incorporating Overlapped Blending and PrevReference Attention, our method can preserve long-term temporal consistency. As evidenced by the tail of the sprite in the 2𝑛𝑑 example and the generated head (not in the reference frame) in the 1𝑠𝑡 example, a similar color is preserved throughout the entire animation. Overall, our method successfully achieves the task of colorizing long sequences of linearts, a feat unattainable by previous works, including both CNN-based and diffusion-based frameworks.

Quantitative Comparison. In this section, we compare our method with others in terms of frame and video quality, frame similarity, sketch alignment with ground-truth animations, and temporal

##### Fig. 4. Qualitative comparison with five methods: ACOF [Yu et al. 2024], TCVC [Thasarathan et al. 2019], CNet+Refonly [Zhang et al. 2023], EISAI [Chen and Zwicker 2022], and SEINE [Chen et al. 2024]. It is recommended to zoom in on the figure to observe the differences. Input frames: 1𝑠𝑡 and 2𝑛𝑑 examples are from Big Fish & Begonia, 3𝑟𝑑 example is from Mr. Miao.

Table 1. Quantitative comparison with ACOF [Yu et al. 2024], TCVC [Thasarathan et al. 2019], ControlNet [Zhang et al. 2023] with Reference-only, and video interpolation methods EISAI [Chen and Zwicker 2022] and SEINE [Chen et al. 2024] .

Similar Testset General Testset FID↓ FVD↓ LPIPS↓ PSNR↑ SSIM↑ EDMD↓ TC↓ FID↓ FVD↓ LPIPS↓ PSNR↑ SSIM↑ EDMD↓ TC↓

ACOF (prev) 77.9654 690.6636 0.3671 10.8886 0.2634 13.2373 2.6077 91.6352 588.8971 0.3772 10.7118 0.2788 15.2450 3.0174 ACOF (first) 25.2291 280.1266 0.1310 19.6044 0.8013 7.8377 1.2265 23.4283 224.1050 0.1097 21.2396 0.8208 8.6646 1.2601

TCVC (prev) 77.6475 779.2147 0.3792 11.7784 0.4752 11.3185 1.8821 91.7752 711.6013 0.4191 11.0152 0.4570 13.5151 2.0850 TCVC (first) 23.4118 217.4509 0.1419 17.9502 0.7587 6.4081 1.3628 26.6516 224.9740 0.1531 17.7315 0.7473 7.3664 1.4165

CNet + Refonly 16.9340 170.2000 0.1114 18.7949 0.7844 7.1903 1.4581 17.1442 135.8851 0.0879 20.6059 0.8043 8.1612 1.3272

EISAI + CNet 18.9262 227.6656 0.1413 18.1603 0.7314 8.2527 1.2952 19.0419 190.6165 0.1099 20.0647 0.7810 9.5414 1.2306 SEINE + CNet 23.0797 836.4528 0.1655 17.2425 0.6504 8.6777 2.3644 30.5665 820.9603 0.1464 18.2319 0.6940 10.6964 2.2858

Ours 8.8423 40.2711 0.0560 24.5489 0.8790 4.3386 1.0784 8.8038 32.6929 0.0399 26.7859 0.9182 6.4648 1.1025

Table 2. Ablation study on model architecture and sequential sampling.

Similar Testset FID↓ FVD↓ LPIPS↓ EDMD↓ TC↓

Full method 8.8423 40.2711 0.0560 4.3386 1.0784

− Ref Attn 9.6793 41.2857 0.0694 5.0137 1.1563 − Sample Schemes 8.6321 40.5472 0.0523 4.4863 1.1479

Prev Sample 11.1873 42.2713 0.0925 5.4924 1.2475

Table 3. Ablation study on number of overlapped frames. The last column indicates the inference time ratio compared with no overlapping (𝑜 = 0).

Similar Testset FID↓ FVD↓ LPIPS↓ EDMD↓ TC↓ Inf. Time↓ 𝑜 = 2 8.7594 50.8983 0.0558 4.3143 1.0949 1.17

𝑜 = 4 (ours) 8.8423 40.2711 0.0560 4.3386 1.0784 1.40 𝑜 = 6 8.8582 45.2426 0.0565 4.2694 1.0737 1.75 𝑜 = 8 8.8227 44.3100 0.0567 4.3265 1.0785 2.33

𝑜 = 10 8.8683 40.8429 0.0578 4.3795 1.0745 3.50

|11.0%<br><br>0.8%<br>1.7%<br>2.2%<br><br><br>1.1%<br><br>1.0%<br><br>22.7%<br><br>3.0%<br>4.1%<br><br><br>7.0%<br><br>3.0%<br><br>3.5%<br><br>24.6%<br><br>2.0%<br><br>2.3%<br><br>6.2%<br><br>2.1%<br><br><br>1.7%<br><br>0 0.1 0.2 0.3 0.4 0.5 0.6 0.7<br><br>Ours<br><br>SEINE<br><br>EISAI<br><br>CNet<br><br>TCVC<br><br>ACOF CG & CV Art & Design Others<br><br>Preferred rate|
|---|

- Fig. 5. Results of user study. Our method has a preferred rate of 58.3% (62.0%, 52.4%, and 63.2% for user group CG & CV, Art & Design, and others).

consistency, as specified in Sec. 6.1. As demonstrated in Tab. 1, our method significantly outperforms other approaches in all aspects, particularly excelling in video quality (FVD) and temporal consistency (TC). These results signify our method’s ability to generate long, temporally consistent animations of superior quality compared to previous works. Furthermore, our method demonstrates the capacity to generalize to animations with diverse styles from the training dataset, yielding comparable quantitative results. Overall, we introduce new metrics (EDMD and TC) to effectively evaluate the sketch alignment and temporal consistency of generated animations, setting a new standard for reference-based lineart video colorization.

- 6.3 User Study

We conducted a user study to further evaluate our method’s performance. From a pool of 30 animations (15 from Similar Testset and 15 from General Testset), participants were presented with groundtruth animations and lineart sketches for reference, followed by animations generated by ACOF [Yu et al. 2024], TCVC [Thasarathan et al. 2019], CNet+Refonly [Zhang et al. 2023], EISAI [Chen and

Zwicker 2022], SEINE [Chen et al. 2024], and our method, in random orders. Each user is required to answer 10 randomly selected questions from the pool by choosing the best animation considering all of the three aspects: 1) similarity to the original, 2) alignment with lineart, and 3) overall quality. Among 113 participants, 20 have worked or majored in areas related to Computer Graphics and Computer Vision (CG & CV), 49 are involved in Art & Design, and 44 are from other fields. As shown in Fig. 5, our method garnered the highest preference rate of 58.3%, with user groups from CG & CV, Art & Design, and other fields showing preference rates of 62.0%, 52.4%, and 63.2%, respectively.

6.4 Ablation Study

Ablation on Model Architecture. To investigate the method’s effect, we conducted an ablation study by removing the Reference Attention layers and re-training the network with the same hyperparameters. Quantitative results in Tab. 2 demonstrate a degradation across all metrics, indicating the model’s diminished capacity to handle animations with large motions. Visually, as shown in Fig. 6, the absence of Reference Attention leads to inconsistent and incorrect coloring of areas with significant motion, such as the black collar in the right example, as well as inconsistent colors of the deer in the left example. These findings underscore the crucial role of Reference Attention in enhancing the model’s ability to generate high-quality animations with large motions.

Ablation on Sequential Sampling Schemes. In Tab. 2, we compare our method with two variants. Firstly, we remove all the sampling schemes mentioned in Sec. 4.2 and sample the animations using only the first reference frame. For Prev Sample, we further switch the reference to the generated frame from the previous segment. Quantitatively, removing the sampling schemes leads to a

Fig. 6. Ablation study on model architecture and sampling scheme. All input frames are from Big Fish & Begonia.

Fig. 7. Ablation study on sampling schemes including Overlapped Blending and Prev-Reference Attention. Input frame is from Big Fish & Begonia.

Anime2Sketch SketchKeras HED Ours

Original

Fig. 8. Impact of different lineart extraction methods. Input frame is from Big Fish & Begonia.

reduction in temporal consistency (higher TC), while other metrics stay similar. For Prev Sample, all metrics degrade due to accumulated errors when using the previously generated frame as reference.

In the qualitative results depicted in Fig. 6, our sampling schemes demonstrate superior ability to preserve temporal consistency. For instance, the colors of the deer (left example) and collar (right example) remain consistent with our full method, while inconsistencies arise in the results without sampling schemes. Moreover, as the little plate in the left example (zoomed in the green frames) gradually reveals more area, our full method maintains a consistent red color akin to the reference image. Compared to Prev Sample, our sampling schemes effectively mitigate accumulated artifacts. For instance, yellow region appears on the blue sky in the left example, and the wall in the right example turns red with Prev Sample. In summary, our sampling schemes can enhance long-term temporal consistency while simultaneously addressing the issue of accumulated artifacts.

In Fig. 7, we analyze the effects of the two schemes for sequential sampling, i.e., Overlapped Blending and Prev-Reference Attention.

We show the intermediate denoised outputs for 𝑡 = 25 → 0, where frame 14 of the 1𝑠𝑡 segment overlaps with frame 14 in the 2𝑛𝑑 segment. For the results without both modules, we note that the content (i.e. the red sleeve) of frame 14 cannot propagate from the 1𝑠𝑡 to the 2𝑛𝑑 segment, leading to inconsistent brown sleeve in the newly sampled frame 14. Upon integrating Overlapped Blending, the red sleeve in frame 14 in the 1𝑠𝑡 segment can be inherited by the 2𝑛𝑑 segment. Finally, with the incorporation of Prev-Reference Attention, the red color of the sleeve successfully propagates to frame 24, resulting in a temporally consistent animation.

Ablation on Number of Overlapped Frames. As discussed in Sec. 4.2, we divide the video sequences into segments with 𝑜 overlapped frames for sequential sampling. In Tab. 3, we analyze different numbers of overlapped frames. We observe that increasing 𝑜 from 2 to 4 significantly improves video quality (FVD) and temporal consistency (TC), while other image quality metrics remain almost unchanged. However, further increasing the number of overlapped frames does not yield significant improvements and even causes a decline in FVD for 𝑜 = 6 and 𝑜 = 8, while also slowing down inference speed. Therefore, to balance the inference quality and speed, a setting of 𝑜 = 4 overlapped frames is the optimal choice.

6.5 Impact of Different Lineart Extraction Methods

To assess the impact of different lineart extraction methods, we apply our model, which was trained using linearts extracted from [Chan et al. 2022], to colorize linearts generated by various extraction methods, including Anime2Sketch [Xiang et al. 2021], SketchKeras

Fig. 9. Colorized results with hand-drawn sketches. Input frames: 1𝑠𝑡 row is from Mr. Miao, 2𝑛𝑑 row is from Big Fish & Begonia.

Fig. 10. Limitations. 1𝑠𝑡 row: artifacts due to VAE. 2𝑛𝑑 row: artifacts due to coarseness in lineart sketches. 3𝑟𝑑 row: artifacts due to partial new object. Input frames: 1𝑠𝑡 and 2𝑛𝑑 rows are from Big Fish & Begonia, 3𝑟𝑑 row is from Mr. Miao.

[lllyasviel 2017], HED [Xie and Tu 2015] (using combined features from all layers), and the method employed in our training dataset. As illustrated in Fig. 8, our model is capable of producing results of similar quality for linearts extracted by Anime2Sketch and SketchKeras, despite the differences in style and detail from the linearts used in our training, demonstrating the generalizability of our method. However, when applied to linearts that are excessively thick, as in HED, our model tends to generate animations with thick and blurred lines. This issue could be solved by augmenting the training data with linearts of different thicknesses.

- 6.6 Application to Hand-drawn Linearts

To verify the practical applicability of our method, we engaged students specializing in painting to create hand-drawn lineart sketches using graphic tablets. Then, we utilized our method to colorize these hand-drawn linearts. As shown in Fig. 9, our model, originally trained with automatically generated sketches, seamlessly accommodates the hand-drawn lineart sketches.

- 6.7 Limitations

Despite our method’s effectiveness, it has two limitations. Firstly, our method may suffer artifacts in tiny details, due to reconstruction loss of VAE and coarseness in the input sketches. As shown in Fig. 10, the details of the girl’s face are lost in 1𝑠𝑡 row due to

reconstruction loss, and the elder’s face is blurred due to the coarseness of the sketches in 2𝑛𝑑 row. Finetuning the original VAE to suit cartoon image domains and the resolution used in our framework and employing data augmentation on the training sketches could mitigate this issue. Another limitation is the potential for inaccurate colorization of partially visible new objects. As in the 3𝑟𝑑 row, when only a portion of a new character enters the scene, the body is incorrectly colorized as the color of nearby petals. The colorization only becomes accurate when the character’s full body is visible. Modifying our video clipping algorithm to include more scene change cases that involve more new objects during training, may help our model handle such scenarios.

7 CONCLUSIONS

In summary, we introduce the first video diffusion framework for reference-based lineart video colorization, addressing the limitations of previous methods. Our approach can produce long temporalconsistent animations of superior quality, by leveraging a pretrained video diffusion model. To adapt the pretrained SVD to our task, we introduce sketch-guided ControlNet for controllable video generation, and Reference Attention, enabling the model to handle expansive motions. Furthermore, our novel sequential sampling, including Overlapped Blending and Prev-Reference Attention, extends the model’s capability to generate long animations while preserving temporal consistency. Our experiments validate the efficacy of our method, showcasing its ability to produce high-quality animations with large motions, which is not achieved by previous works.

As our framework is generic, it can be applied to other modalities, e.g., edge, depth, and normal map. In future works, we may extend our method to generate realistic videos guided by other modalities or even multi-modality. The performance of realistic video generation could be further improved by using large-scale realistic video datasets and leveraging the fact that SVD is also pretrained on similar videos.

ACKNOWLEDGMENTS

We thank the anonymous reviewers for helping us to improve this paper. We also thank Bi An Tian (Beijing) Culture Co., Ltd., and Gu Dong Animation Studio for approving us to use their animation content. The work described in this paper was fully supported by a GRF grant from the Research Grants Council (RGC) of the Hong Kong Special Administrative Region, China [Project No. CityU 11216122].

REFERENCES

AnythingV3. 2023. Anything V3.0. https://huggingface.co/swl-models/anything-v3.0. Omri Avrahami, Dani Lischinski, and Ohad Fried. 2022. Blended Diffusion for Text-

Driven Editing of Natural Images. In Proc. CVPR. 18208–18218.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. 2023. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv:2311.15127 [cs.CV]

Yu Cao, Xiangqiao Meng, P. Y. Mok, Tong-Yee Lee, Xueting Liu, and Ping Li. 2024. AnimeDiffusion: Anime Diffusion Colorization. IEEE Transactions on Visualization and Computer Graphics (2024), 1–14. https://doi.org/10.1109/TVCG.2024.3357568

Yu Cao, Hao Tian, and P. Y. Mok. 2022. Attention-Aware Anime Line Drawing Colorization. 2023 IEEE International Conference on Multimedia and Expo (ICME) (2022), 1637–1642.

Caroline Chan, Fredo Durand, and Phillip Isola. 2022. Learning to generate line drawings that convey geometry and semantics. In CVPR. Shuhong Chen and Matthias Zwicker. 2022. Improving the Perceptual Quality of 2D Animation Interpolation. In Proc. ECCV.

Shu-Yu Chen, Jia-Qi Zhang, Lin Gao, Yue He, Shihong Xia, Min Shi, and Fang-Lue Zhang. 2022. Active Colorization for Cartoon Line Drawings. IEEE Transactions on Visualization and Computer Graphics 28, 2 (2022), 1198–1208. https://doi.org/10. 1109/TVCG.2020.3009949

Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. 2023. Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models. arXiv:2305.13840 [cs.CV]

Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction. In The Twelfth International Conference on Learning Representations.

Yuanzheng Ci, Xinzhu Ma, Zhihui Wang, Haojie Li, and Zhongxuan Luo. 2018. UserGuided Deep Anime Line Art Colorization with Conditional Adversarial Networks. In Proceedings of the 26th ACM international conference on Multimedia. https: //doi.org/10.1145/3240508.3240661

Per-Erik Danielsson. 1980. Euclidean distance mapping. Computer Graphics and Image Processing 14, 3 (1980), 227–248. https://doi.org/10.1016/0146-664X(80)90054-4 Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion Models Beat GANs on Image

Synthesis. In Advances in Neural Information Processing Systems, Vol. 34. 8780–8794.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, Vol. 27.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai.

2023. SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models. arXiv:2311.16933 [cs.CV] https://arxiv.org/abs/2311.16933

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. International Conference on Learning Representations (2024).

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent Video Diffusion Models for High-Fidelity Long Video Generation. arXiv:2211.13221 [cs.CV]

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems, Vol. 30.

Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. 2024. Distilling Diffusion Models into Conditional GANs. arXiv preprint arXiv:2405.05967 (2024).

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. 2022. Elucidating the Design Space of Diffusion-Based Generative Models. In Proc. NeurIPS.

H. Kim, H. Jhoo, E. Park, and S. Yoo. 2019. Tag2Pix: Line Art Colorization Using Text Tag With SECat and Changing Loss. In ICCV 2019. IEEE Computer Society, Los Alamitos, CA, USA, 9055–9064. https://doi.org/10.1109/ICCV.2019.00915

Diederik P. Kingma and Max Welling. 2014. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. 2023. Multi-Concept Customization of Text-to-Image Diffusion. In Proc. CVPR. Xiaoyu Li, Bo Zhang, Jing Liao, and Pedro V. Sander. 2022. Deep Sketch-Guided Cartoon Video Inbetweening. IEEE Transactions on Visualization and Computer Graphics 28, 8 (aug 2022), 2938–2952. https://doi.org/10.1109/TVCG.2021.3049419

lllyasviel. 2017. SketchKeras: An u-net to take the sketch from a painting. https: //github.com/lllyasviel/sketchKeras

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. 2024. Latte: Latent Diffusion Transformer for Video Generation. arXiv preprint arXiv:2401.03048 (2024).

Mikubill. 2023. ControlNet for Stable Diffusion WebUI. https://github.com/Mikubill/sdwebui-controlnet.

Yingge Qu, Tien-Tsin Wong, and Pheng-Ann Heng. 2006. Manga colorization. ACM Trans. Graph. 25, 3 (jul 2006), 1214–1220. https://doi.org/10.1145/1141911.1142017 Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-Resolution Image Synthesis With Latent Diffusion Models. In Proc. CVPR. 10684–10695.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. Cham, 234–241.

Kazuhiro Sato, Yusuke Matsui, Toshihiko Yamasaki, and Kiyoharu Aizawa. 2014. Reference-based manga colorization by graph correspondence using quadratic programming. In SIGGRAPH Asia 2014 Technical Briefs. Association for Computing Machinery, Article 15, 4 pages. https://doi.org/10.1145/2669024.2669037

Min Shi, Jia-Qi Zhang, Shu-Yu Chen, Lin Gao, Yu-Kun Lai, and Fang-Lue Zhang. 2020. Deep Line Art Video Colorization with a Few References. arXiv:2003.10685 [cs.CV]

Li Siyao, Shiyu Zhao, Weijiang Yu, Wenxiu Sun, Dimitris Metaxas, Chen Change Loy, and Ziwei Liu. 2021. Deep Animation Video Interpolation in the Wild. In CVPR 2021. 6583–6591. https://doi.org/10.1109/CVPR46437.2021.00652

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising Diffusion Implicit Models. In International Conference on Learning Representations.

Daniel Sýkora, John Dingliana, and Steven Collins. 2009. LazyBrush: Flexible Painting Tool for Hand-drawn Cartoons. Comput. Graph. Forum 28 (04 2009), 599–608. https://doi.org/10.1111/j.1467-8659.2009.01400.x

Zachary Teed and Jia Deng. 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In Proc. ECCV 2020. 402–419. https://doi.org/10.1007/978-3-030-585365_24

Harrish Thasarathan and Mehran Ebrahimi. 2019. Artist-Guided Semiautomatic Animation Colorization. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW). 3157–3160. https://doi.org/10.1109/ICCVW.2019.00388

Harrish Thasarathan, Kamyar Nazeri, and Mehran Ebrahimi. 2019. Automatic Temporally Coherent Video Colorization. In 2019 16th Conference on Computer and Robot Vision (CRV). 189–194. https://doi.org/10.1109/CRV.2019.00033

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. 2019. FVD: A new Metric for Video Generation. In ICLR 2019 Workshop DeepGenStruct.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, Vol. 30.

Ning Wang, Muyao Niu, Zhi Dou, Zhihui Wang, Zhiyong Wang, Zhaoyan Ming, Bin Liu, and Haojie Li. 2023. Coloring anime line art videos with transformation region enhancement network. Pattern Recognition 141 (2023), 109562. https://doi.org/10. 1016/j.patcog.2023.109562

Zhou Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. 2004. Image Quality Assessment: From Error Visibility to Structural Similarity. Trans. Img. Proc. 13, 4 (apr 2004), 600–612. https://doi.org/10.1109/TIP.2003.819861

Xiaoyu Xiang, Ding Liu, Xiao Yang, Yiheng Zhu, and Xiaohui Shen. 2021. Anime2Sketch: A Sketch Extractor for Anime Arts with Deep Networks. https://github.com/ Mukosame/Anime2Sketch

Saining Xie and Zhuowen Tu. 2015. Holistically-Nested Edge Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2024. ToonCrafter: Generative Cartoon Interpolation. arXiv:2405.17933 [cs.CV] https://arxiv.org/abs/2405.17933

Sheng-Siang Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, Jianlong Fu, Gong Ming, Lijuan Wang, Zicheng Liu, Houqiang Li, and Nan Duan. 2023. NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation. In Annual Meeting of the Association for Computational Linguistics.

Yifeng Yu, Jiangbo Qian, Chong Wang, Yihong Dong, and Baisong Liu. 2024. Animation line art colorization based on the optical flow method. Computer Animation and Virtual Worlds 35 (02 2024). https://doi.org/10.1002/cav.2229

Lvmin Zhang, Chengze Li, Tien-Tsin Wong, Yi Ji, and Chunping Liu. 2018. Two-stage sketch colorization. ACM Trans. Graph. 37, 6, Article 261 (dec 2018), 14 pages. https://doi.org/10.1145/3272127.3275090

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. In Proc. ICCV. 3836–3847.

Qian Zhang, Bo Wang, Wei Wen, Hai Li, and Junhui Liu. 2021. Line Art Correlation Matching Feature Transfer Network for Automatic Animation Colorization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 3872–3881.

Zhongwei Zhang, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Ting Yao, Yang Cao, and Tao Mei. 2024. TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Video Diffusion Models. arXiv:2403.17005 [cs.CV]

Changqing Zou, Haoran Mo, Chengying Gao, Ruofei Du, and Hongbo Fu. 2019. Language-based colorization of scene sketches. ACM Trans. Graph. 38, 6, Article 233 (nov 2019), 16 pages. https://doi.org/10.1145/3355089.3356561

