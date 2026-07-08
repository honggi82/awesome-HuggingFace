# arXiv:2401.01256v2[cs.CV]16Sep2024

## VideoStudio: Generating Consistent-Content and Multi-Scene Videos

##### Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei

HiDream.ai Inc. {longfuchen, qiuzhaofan, tiyao, tmei}@hidream.ai

Abstract. The recent innovations and breakthroughs in diffusion models have significantly expanded the possibilities of generating high-quality videos for the given prompts. Most existing works tackle the single-scene scenario with only one video event occurring in a single background. Extending to generate multi-scene videos nevertheless is not trivial and necessitates to nicely manage the logic in between while preserving the consistent visual appearance of key content across video scenes. In this paper, we propose a novel framework, namely VideoStudio, for consistentcontent and multi-scene video generation. Technically, VideoStudio leverages Large Language Models (LLM) to convert the input prompt into comprehensive multi-scene script that benefits from the logical knowledge learnt by LLM. The script for each scene includes a prompt describing the event, the foreground/background entities, as well as camera movement. VideoStudio identifies the common entities throughout the script and asks LLM to detail each entity. The resultant entity description is then fed into a text-to-image model to generate a reference image for each entity. Finally, VideoStudio outputs a multi-scene video by generating each scene video via a diffusion process that takes the reference images, the descriptive prompt of the event and camera movement into account. The diffusion model incorporates the reference images as the condition and alignment to strengthen the content consistency of multi-scene videos. Extensive experiments demonstrate that VideoStudio outperforms the SOTA video generation models in terms of visual quality, content consistency, and user preference. Source code is available at https://github.com/FuchenUSTC/VideoStudio.

### 1 Introduction

Diffusion Probabilistic Models (DPM) have demonstrated high capability in generating high-quality images [7,15,16,35,36,44,48,50,68]. DPM approaches image generation as a multi-step sampling process, involving the use of a denoiser network to progressively transform a Gaussian noise map into an output image. Compared to 2D images, videos have an additional time dimension, which introduces more challenges when extending DPM to video domain. One typical way is to leverage pre-trained text-to-image models to produce video frames [20,40,59]

###### Input prompt: A young man with blue hair is making cake Output video:

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]|
|---|

- Scene-1: The young man measures out ingredients
- Scene-2: The young man pours the batter into a pan
- Scene-3: The young man stirs the batter in the pan

|[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]|
|---|

|[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]|
|---|

|[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]|
|---|

Scene-4: The young man puts the cake on the table

|[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]|
|---|

Scene-5: The young man makes a phone call to invite his friends

|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]|
|---|

Scene-6: The young man is in the outside of his house to wait his friends

- Fig. 1: An illustration of prompt and multi-scene video generation by VideoStudio.

or utilize a 3D denoiser network learnt on video data to generate a sequence of frames in an end-to-end manner [3,11,12,14,34,47]. Despite having impressive results in the realm of text-to-video generation, most existing works focus on only single-scene videos, featuring one event in a single background. The generation of multi-scene video is still a problem not yet fully explored in the literature.

The difficulty of multi-scene video generation generally originates from two aspects: 1) how to arrange and establish different events in a logical and realistic way for a multi-scene video? 2) how to guarantee the consistency of common entities, e.g., foreground objects or persons, throughout the video? For instance, given an input prompt of “a young man is making cake,” a multi-scene video is usually to present the step-by-step procedure of making a cake, including measuring out the ingredients, pouring the ingredients into a pan, cooking the cake, etc. This necessitates a comprehensive understanding and refinement of the prompt. As such, we propose to mitigate the first issue through capitalizing on Large Language Models (LLM) to rewrite the input prompt into multi-scene video script. LLM inherently abstracts quantities of text data on the Web about the input prompt to produce the script, which describes and decomposes the video logically into multiple scenes. To alleviate the second issue, we exploit the common entities to generate reference images as the additional condition to produce each scene video. The reference images, as the link across scenes, effectively align the content consistency within a multi-scene video.

To consolidate the idea, we present a new framework dubbed as VideoStu-

dio for consistent-content and multi-scene video generation. Technically, VideoStu-

dio first transforms the input prompt into a thorough multi-scene video script by using LLM. The script for each scene consists of the descriptive prompt of the event in the scene, a list of foreground objects or persons, the background, and camera movement. VideoStudio then identifies common entities that appear across multiple scenes and requests LLM to enrich each entity. The resultant entity description is fed into a pre-trained Stable Diffusion [44] model to produce a reference image for each entity. Finally, VideoStudio outputs a multi-scene video via involving two diffusion models, i.e., VideoStudio-Img and VideoStudio-

Vid. VideoStudio-Img is dedicated to incorporating the descriptive prompt of the event and the reference images of entities in each scene as the condition to generate a scene-reference image. VideoStudio-Vid takes the scene-reference image plus temporal dynamics of the action depicted in the descriptive prompt of the event and camera movement in the script as the inputs and produces a video clip for each scene.

The main contribution of this work is the proposal of VideoStudio for generating consistent-content and multi-scene videos. The solution also leads to the elegant views of how to use LLM to properly arrange content of multi-scene videos and how to generate visually consistent entities across scenes, which are problems seldom investigated in literature. Extensive experiments conducted on public benchmarks demonstrate that VideoStudio outperforms SOTA video generation models in terms of visual quality, content consistency and user preference.

### 2 Related Work

Image generation is a fundamental challenge of computer vision and has evolved rapidly in the past decade. Recent advances in Diffusion Probabilistic Models (DPM) have led to remarkable improvements in generating high-fidelity images [3,7,15,16,32,33,35–37,43,44,48–50,68]. DPM is a category of generative models that utilizes a sequential sampling process to convert random Gaussian noise into high-quality images. For example, GLIDE [37] and DALL-E 2 [43] exploit the sampling process in the pixel space, conditioned on the text prompt using classifier-free guidance [16]. Nevertheless, training a powerful denoising network remains challenging due to high computational cost and memory demand associated with sampling at the pixel level. To mitigate this problem, Latent Diffusion Models (LDM) [44] employ sampling in the latent feature space that is established by a pre-trained autoencoder, leading to the improvements on computation efficiency and image quality. Furthermore, the application of DPM is further enhanced by incorporating advanced sampling strategies [32, 33, 49] and additional control signals [35,68].

Video generation is a natural extension of image generation in video domain. The early approaches, e.g., ImagenVideo [14] and Make-A-Video [47], train video diffusion models in the pixel space, resulting in high computational complexity. Following LDM in image domain, several works [3,6,11,34,70] propose to exploit the sampling process in the latent feature space for video generation. These works extend the 2D UNet with the transformer layers [23,62,63] to

##### 3D UNet by injecting temporal self-attentions [28,29] and/or temporal convolutions [30,31]. For instance, Video LDM [3] and AnimateDiff [11] focus on training the injected temporal layers while freezing the spatial layers to preserve the ability of the pre-trained image diffusion model. VideoFusion [34] decomposes the 3D noise into a 2D base noise shared across frames and a 3D residual noise, enhancing the correlation between frames. However, the generated videos usually have a limited time duration, typically around 16 frames. Consequently, some recent researches emerge to generate long videos by an extrapolation strategy or

(1) Multi-scene video script generation (2) Entity reference image generation

[Figure 159]

Input Prompt

Text-to-Image Model （T2I Model)

|“a young man with blue hair is making cake”|
|---|

Entity Descriptions

Foreground Reference images Background Reference images

Query

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

|“You need to<br><br>envision a multiscene video ... ”|
|---|

Large Language Model (LLM)

Multi-Scene Video Script

- Scene 1: “The young man measures out ingredients”; Objects: young man; Background: kitchen; Camera: moving left fast

- Scene 2: “The young man pours the batter into a pan”; Objects: young man; Background: kitchen; Camera: moving left slowly

“young man” “cake” “kitchen” “dining room”

(3) Video scene generation

Action: putting the cake on the table; Camera: static

x N

###### …

[Figure 165]

[Figure 166]

[Figure 167]

VideoStudio-Img

Scene N-1: “The young man puts the cake on the table”; Objects: young man, cake; Background: kitchen; Camera: static

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Scene N-1: “The young man

###### Scene N: “The young man makes a phone call to invite his friends”;

puts the cake on the table”

Objects: young man, phone; Background: dining room; Camera: static

[Figure 172]

Entities:

[Figure 173]

VideoStudio-Vid

Query

[Figure 174]

[Figure 175]

“young man”

Video Scene

Scene-Reference Image

|“You need to<br><br>describe the entities<br><br>in detail ... ”|
|---|

“cake” “dining room”

VideoStudio-Img VideoStudio-Vid

Foreground Reference

Entity Descriptions

Scene-Reference Image

Visual Encoder

Text Encoder

theThecakeyoungonmanthe tableputs Embedder action

Visual Encoder

Action

young man: “The photo depicts a young man with blue hair standing in a kitchen, wearing an apron and holding a mixer ...”

Background Reference

|CameraMoving| |
|---|---|
| | |

cake: “The cake is a masterful work of art, with intricate designs and patterns that are both visually appealing and visually stunning ... ”

[Figure 176]

[Figure 177]

CameraMoving

[Figure 178]

[Figure 179]

[Figure 180]

3D Conv

3D Conv

2D Conv

2D Conv

kitchen: “This photo of a kitchen features a clean and modern aesthetic, with white cabinets and countertops ... ”

3D Attn

3D Attn

2D

2D

…

…

Attn

Attn

x T x T

dining room: “The photo captures a modern luxurious dining room

2D Noise

3D Denoising UNet

2D Denoising UNet

with elegant furniture and a breathtaking view of the city skyline.”

3D Noise

- Fig. 2: An overview of our VideoStudio framework for consistent-content and multiscene video generation. VideoStudio consists of three main stages: (1) multi-scene video script generation, (2) entity reference image generation, and (3) video scene generation. In the first stage, LLM is utilized to convert the input prompt into a comprehensive multi-scene script. The script for each scene includes the descriptive prompt of the event in the scene, a list of foreground objects or persons, the background, and camera movement. We then request LLM to detail the common foreground/background entities across scenes. These entity descriptions are fed into a text-to-image (T2I) model to produce reference images in the second stage. Finally, in the third stage, VideoStudioImg exploits the descriptive prompt of the event and the reference images of entities in each scene as the condition to generate a scene-reference image. VideoStudio-Vid takes the scene-reference image plus temporal dynamics of the action depicted in the descriptive prompt of the event and camera movement in the script as the inputs and produces a video clip for each scene.

hierarchical architecture [12,25,52,53,66]. In addition, video editing techniques utilize the input video as a condition and generate a video by modifying the style or key object of the input video [9,10,12,18,39,40,46,54,57,59,65].

In short, our work in this paper focuses on consistent-content and multi-scene video generation. The most related work is [26], which aligns the appearance of entities across scenes through the bounding boxes provided by LLM. Ours is different in the way that we explicitly determine the appearance of entities by generating reference images, which serve as a link across scenes and effectively enhance the content consistency within a multi-scene video.

- 3 VideoStudio

This section presents the proposed VideoStudio framework for consistent-content and multi-scene video generation. Figure 2 illustrates an overview of VideoStudio framework, consisting of three main stages: (1) multi-scene video script generation (Sec. 3.1), (2) entity reference image generation (Sec. 3.2), and (3) video scene generation (Sec. 3.3).

#### 3.1 Multi-Scene Video Script Generation

As depicted in Figure 2(1), VideoStudio utilizes LLM to convert the input prompt into a comprehensive multi-scene script. In view of its high deployment flexibility and inference efficiency, we use the open-source ChatGLM3-6B model [8,67]. The LLM is requested by a pre-defined query, “You need to envision a multi-scene video and describe each scene ...”, to treat the input prompt as the theme, logically decompose the video into multiple scenes and generate a script for each scene in the following format:

- [Scene 1: prompt, foreground, background, camera move];
- [Scene 2: prompt, foreground, background, camera move];

(1)

... [Scene N: prompt, foreground, background, camera move].

Here N denotes the number of video scenes, which is determined by the LLM. For each scene, the descriptive prompt of the event in the scene, a list of foreground objects or persons, the background, and camera movement are provided. The camera movement is restricted to a close-set of directions {static, left, right, up, down, forward, backward} and speeds {slow, medium, fast}.

Next, VideoStudio identifies the common entities, which include foreground objects or persons and background locations. To achieve this, we ask the LLM to assign the common object, person, or background the same name across scenes when generating the video script. Therefore, we strictly match the name of entities and discover the entities that appear in multiple scenes. To further improve the quality of the video script, we employ the capability of the LLM for multiround dialogue. Specifically, we start the dialogue by asking the LLM to specify the key aspects with respect to the entity, such as “What are the aspects that should be considered when describing a photo of a young man in detail?” In the next round of dialogue, we request the LLM to describe the entity from the viewpoints of the given aspects. Moreover, the original prompt is also taken as the input to the LLM to ensure that the essential characteristics, e.g., “blue hair” of the young man, are emphasized in entity description generation.

Please note that the GPT-4 [38] can also be used for script generation, but it incurs an additional 0.12 USD for the GPT-4 API call per query. In VideoStudio, we leverage the open-source ChatGLM3-6B and perform the inference on our devices to circumvent the need for API call. Nevertheless, the scale of ChatGLM36B is much smaller, resulting in unstable outcomes that may deviate from the specified script format. To alleviate this issue, we have empirically abstracted the following principles to enhance the stability of open-source LLM:

- • Before the dialogue starts, we provide comprehensive instructions to the LLM, delineating the additional requirements, specifying the script format, and offering the examples of the expected outputs.
- • For each query, we manually select five in-context examples as the historical context for multi-round dialogue. These examples are very carefully designed to ensure a diverse range of scenes, key objects, and background, and serve to emphasize the required script format for LLM.

- • After each round of dialogue, we verify the output format. If the results are seemingly inappropriate, we re-run the entire script generation stage. Such strategy is simple to implement without requiring any additional expenses.

We will provide the full version of our instructions, examples, and queries in the supplementary materials.

#### 3.2 Entity Reference Image Generation

In the second stage of VideoStudio, we unify the visual appearance of common entities by explicitly generating a reference image for each entity. The reference images act as the link to cohere the content across scenes. We achieve this by first feeding the entity description into a pre-trained Stable Diffusion model for text-to-image generation. Then, we employ the U2-Net [41] model for salient object detection, and segment the foreground and background areas in each resultant image. By utilizing the segmentation masks, we can further remove the background pixels from the foreground reference image and vice versa, in order to prevent the interference between the foreground and background visual contents in the reference images.

#### 3.3 Video Scene Generation

VideoStudio produces a multi-scene video by generating each scene via the diffusion models by taking the reference images, the descriptive prompt of the event and camera movement into account. This stage involves two primary components: the VideoStudio-Img, which utilizes the descriptive prompt of the event and the reference images of entities in each scene as the condition to generate a scene-reference image, and the VideoStudio-Vid, which employs the scene-reference image plus temporal dynamics of the action depicted in the descriptive prompt of the event and camera movement in the script as the inputs and produces a video clip for each scene.

The VideoStudio-Img component aims to generate a scene-reference image conditioning on the event prompt and entity reference images for each scene. To accomplish this, we remold the Stable Diffusion architecture by replacing the original attention module with a novel attention module that can handle three contexts: the text prompt, foreground reference image, and background reference image. As depicted in Figure 3a, we utilize text and visual encoder of a pre-trained CLIP model to extract the sequential text feature yt ∈ RL

t×Ct and local image features yf ∈ RL

b×Cb for the prompt, foreground reference image, and background reference image, respectively. Here, L and C denote the length and the channels of the feature sequence. For the case of multiple foregrounds in one scene, we concatenate the features from all foreground reference images along the length dimension. Given the input feature x, the

f×Cf and yb ∈ RL

Scene

Foreground

Background

Scene-Reference

Action Category

prompt

Reference Image

Reference Image

Image

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

CLIP-Text CLIP-Image CLIP-Image

Action-Embed

CLIP-Image

[Figure 186]

Frozen Tuned

[Figure 187]

[Figure 188]

[Figure 189]

CA CA CA

CA

[Figure 190]

[Figure 191]

+

+

[Figure 192]

Spatial SA

[Figure 193]

SA

[Figure 194]

Frozen

[Figure 195]

Temporal SA

+

+

[Figure 196]

Tuned

(a)

(b)

Fig. 3: Diagram illustrations of (a) attention module in the VideoStudio-Img which takes the scene prompt and foreground/background reference images as the inputs and (b) attention module in the VideoStudio-Vid conditioning on the scene-reference image and the described action category.

outputs z of the attention are computed as

- y = CA1(x, yt) + CA2(x, yf) + CA3(x, yb),
- z = x + SA(y),

(2)

where CA1 and SA are the cross-attention and self-attention modules, respectively, in the original Stable Diffusion architecture. We add two additional crossattention modules, CA2 and CA3, which leverage the guidance provided by entity reference images. Moreover, we propose to optimize the parameters of CA2 and CA3 while freezing the other parts of the network.

The VideoStudio-Vid is a video diffusion model that employs the scenereference image, the action described in the prompt of the event, and camera movement in the script as the inputs. Particularly, we start by extending the Stable Diffusion model to a spatio-temporal form and replacing the original attention module with a new one that is conditioned on the scene-reference image and action category, as shown in Figure 3b. Taking 400 action categories in Kinetics [4] as an action vocabulary, an indicator vector ya ∈ [0,1]400 is built to infer if each action in the vocabulary exists in the scene prompt and subsequently converted into feature space using a linear embedding f. For the scene-reference image, we use the visual encoder of CLIP to extract the image feature ys ∈ RL

s×Cs, which is then fed into the cross-attention operation. The original self-attention is decomposed into a spatial self-attention (Spatial SA) and a temporal self-attention (Temporal SA), which operate self-attention solely on spatial and temporal dimension, respectively, to reduce computations. Hence, given the input feature x, the attention module is formulated as

- y = CA(x, ys) + f(ya),
- z = x + Temporal SA(Spatial SA(y)).

(3)

Moreover, we further inject several temporal convolutions behind each spatial convolution into the Stable Diffusion architecture, to better capture temporal dependencies in image-to-video generation.

To reflect the camera movement stated by the script in the generated video, we uniquely modify the frames in the intermediate step of sampling process by warping the neighboring frames based on the camera moving direction and speed. We execute this adjustment after the first Tm DDIM sampling steps, followed by continuing the sampling process. Such modification ensures that the resultant video clip maintains the same camera movement as we warp the intermediate frames. In general, setting a small Tm for early modification may not effectively control the camera movement, while a late modification may affect the visual quality of the output videos. In practice, we observe that Tm=5 provides a good trade-off. We will detail the formulation of the modification process and the ablation study of the step Tm in our supplementary materials.

### 4 Experiments

#### 4.1 Datasets

Our VideoStudio framework is trained on three large-scale datasets: LAION2B [45], WebVid-10M [1] and HD-VG-130M [56]. The LAION-5B is one of the largest text-image dataset consisting of around 5 billion text-image pairs. To train VideoStudio-Img, We utilize a subset, namely LAION-2B, which focuses on the text prompts in English. The WebVid-10M and HD-VG-130M are the large-scale single-scene video datasets, containing approximately 10M and 130M text-video pairs, respectively. VideoStudio-Vid is trained on the combination of WebVid-10M and a randomly chosen 20M subset from HD-VG-130M.

To evaluate video generation, we select the text prompts from three video datasets, i.e., MSR-VTT [61], ActivityNet Captions [22] and Coref-SV [26]. The first one provides the single-scene prompts, while the remaining two datasets comprise multi-scene prompts. The MSR-VTT consists of 10K web video clips, each annotated with approximate 20 natural sentences. We utilize the text annotation of validation videos to serve as single-scene prompts in our evaluation. The ActivityNet Captions dataset is a multi-event video dataset designed for dense-captioning tasks. Following [26], we randomly sample 165 videos from the validation set and exploit the event captions as the multi-scene prompts. The Coref-SV is a multi-scene description dataset, which was constructed by replacing the subject of multi-scene paragraphs in Pororo-SV dataset [21,24]. Coref-SV samples 10 episodes from the Pororo-SV dataset and replaces the subject with 10 real-world entities, resulting in 100 multi-scene prompts.

#### 4.2 Evaluation Metrics

For the video generation task, we adopt five evaluation metrics. To assess the visual quality of the generated videos, we utilize the average of the per-frame Fréchet Inception Distance (FID) [13] and the clip-level Fréchet Video Distance (FVD) [51], both of which are commonly used metrics. We also employ the CLIPSIM [58] metric to evaluate the alignment between the generated frames

and the input prompt. To verify the content consistency, we calculate frame consistency (Frame Consis.) by determining the CLIP-similarity between consecutive frames, serving as an intra-scene consistency measure. Additionally, we employ the Grounding-DINO detector [27] to detect common objects across scenes and then calculate the CLIP-similarity between the common objects appeared in different scenes, achieving cross-scene consistency (Scene Consis.).

#### 4.3 Implementation Details

We implement the proposed VideoStudio using the Diffusers codebase on the platform of PyTorch.

Training stage of VideoStudio-Img. VideoStudio-Img is originated from the Stable Diffusion v2.1 model by incorporating two additional cross-attention modules. These modules are initialized from scratch and trained on the textimage pairs from LAION-2B dataset, while other parts of the network are frozen. For each image, we randomly sample a 512×512 patch cropped from the original image, and utilize the U2-Net model to segment the foreground area of each patch. The isolated foreground and background areas serve as the foreground and background reference images, respectively, to guide the generation of the input patch. We set each minibatch as 512 patches that are processed on 64 A100 GPUs in parallel. The parameters of the model are optimized by AdamW optimizer with a fixed learning rate of 1 × 10−4 for 20K iterations.

Training stage of VideoStudio-Vid. VideoStudio-Vid is developed based on the Stable Diffusion XL architecture by inserting temporal attentions and temporal convolutions. The training is carried out on the WebVid-10M and HDVG-130M datasets. For each video, we randomly sample a 16-frame clip with the resolution of 320×512 and an FPS of 8. The middle frame of the clip is utilized as the scene-reference image. Each minibatch consists of 128 video clips implemented on 64 A100 GPUs in parallel. We utilize the AdamW optimizer with a fixed learning rate of 3 × 10−6 for 480K iterations.

#### 4.4 Experimental Analysis of VideoStudio

Evaluation on VideoStudio-Img. We first verify the efficacy of VideoStudioImg in aligning with the input entity reference images. To this end, we take the prompts from MSR-VTT validation set. The input foreground and background reference images are produced by using LLM and Stable Diffusion model. We validate the generated images on the measure of foreground similarity (FG-SIM) and background similarity (BG-SIM), which are the CLIP-similarity values with the foreground and background reference images, respectively. Table 1 lists the performance comparisons of IP-Adapter [64] and different VideoStudio-Img variants by leveraging different input references. Specifically, the use of foreground/background reference image as guidance leads to higher FG-SIM/BGSIM values comparing to IP-Adapter or not leveraging reference images. Though both of IP-Adapter and VideoStudio-Img exploit additional cross-attention to maintain visual contents in image diffusion, our VideoStudio-Img is devised for

- Table 1: Performance comparisons of IPAdapter [64] and VideoStudio-Img variants with different input references on the MSR-VTT validation set.

Input References

FG-SIM BG-SIM CLIPSIM FG Ref. BG Ref.

w/o Ref. 0.5162 0.4131 0.3001

IP-Adapter [64] ✓ 0.7116 0.4035 0.2910

✓ 0.5128 0.5059 0.2954

VideoStudio-Img ✓ 0.7919 0.4393 0.2982

✓ 0.5362 0.5742 0.3002

✓ ✓ 0.8102 0.5861 0.3023

Fig. 4: Examples of the foreground and background reference images and the generated scene-reference image by the VideoStudio-Img variants.

Foreground

Reference Image ReferenceBackgroundImage w/o Ref. w/ FG Ref. w/ BG Ref. VideoStudio-Img

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

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

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

[Figure 530]

[Figure 531]

[Figure 532]

- Table 2: Performance comparisons for single-scene video generation with real frame as scene-reference image on WebVid-10M.

Approach FVD (↓) Frame Consis. (↑)

RF+VideoCrafter [5] 293.3 97.9 RF+I2VGen-XL [69] 254.9 97.6 RF+VideoComposer [57] 231.0 95.9 RF+DynamiCrafter [60] 176.8 97.5 RF+SVD [2] 153.0 98.7

RF+VideoStudio-Vid− 157.3 98.5 RF+VideoStudio-Vid 116.5 98.8

a more complex scenario to specify foreground objects and background. There are two major differences: 1) We pre-segment the foreground/background of the reference images to avoid the visual content interference; 2) IP-Adapter extracts global image features from CLIP, while ours utilizes local image tokens from CLIP to improve spatial discrimination in local regions. As indicated by the results, emphasizing the feature learning of local region on the more clean (masked) foreground/background reference image does benefit the visual alignment. Furthermore, the combination of both reference images achieves the highest FG-SIM of 0.8102 and BG-SIM of 0.5861. Figure 4 showcases four generated images by different VideoStudio-Img variants with various reference images. The results demonstrate the advantage of VideoStudio-Img to align with the visual contents in the entity reference images.

Evaluation on VideoStudio-Vid. Next, we assess the visual quality of the single-scene videos generated by VideoStudio-Vid. We exploit the real frame from the WebVid-10M validation set as the scene-reference image irrespective of the generation quality, and produce a video using the corresponding text prompt, which is referred to as RF+VideoStudio-Vid. We compare our proposal with five image-to-video diffusion models and one variant of VideoStudio-Vid, i.e., RF+VideoStudio-Vid−, which disables the action guidance in VideoStudioVid. Table 2 presents the performance comparisons for single-scene video gen-

- Table 3: Performance comparisons for single-scene video generation on MSR-VTT validation set. RF indicates whether to utilize the real frame as the reference.

Approach RF FID (↓) FVD (↓)

CogVideo [17] 23.6 MagicVideo [71] - 998 Make-A-Video [47] 13.2 VideoComposer [57] - 580 VideoDirectorGPT [26] 12.2 550 ModelScopeT2V [55] 11.1 550 SD+VideoStudio-Vid 11.9 381

RF+VideoCrafter [5] ✓ 45.0 339 RF+I2VGen-XL [69] ✓ 37.4 264 RF+VideoComposer [57] ✓ 31.3 208 RF+DynamiCrafter [60] ✓ 26.1 196 RF+SVD [2] ✓ 15.3 172 RF+VideoStudio-Vid ✓ 10.8 133

Input MultiScene Prompt

ModelScopeT2V VideoDirectorGPT VideoStudio (Ours)

|[Figure 533]<br><br>[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]<br><br>[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]<br><br>[Figure 540]|
|---|

|[Figure 541]<br><br>[Figure 542]<br><br>[Figure 543]<br><br>[Figure 544]<br><br>[Figure 545]<br><br>[Figure 546]<br><br>[Figure 547]<br><br>[Figure 548]<br><br>[Figure 549]<br><br>[Figure 550]<br><br>[Figure 551]<br><br>[Figure 552]<br><br>[Figure 553]<br><br>[Figure 554]|
|---|

|[Figure 555]<br><br>[Figure 556]<br><br>[Figure 557]<br><br>[Figure 558]<br><br>[Figure 559]<br><br>[Figure 560]<br><br>[Figure 561]<br><br>[Figure 562]<br><br>[Figure 563]<br><br>[Figure 564]<br><br>[Figure 565]<br><br>[Figure 566]<br><br>[Figure 567]<br><br>[Figure 568]<br><br>[Figure 569]<br><br>[Figure 570]<br><br>[Figure 571]<br><br>[Figure 572]<br><br>[Figure 573]<br><br>[Figure 574]<br><br>[Figure 575]<br><br>[Figure 576]<br><br>[Figure 577]<br><br>[Figure 578]|
|---|

- Scene-1: A mouse is holding a book

and makes a happy face

- Scene-2: A mouse looks happy and

talks

- Scene-3: A mouse is pulling petals off the flower

- Scene-4: A mouse is ripping a petal from the flower

|[Figure 579]<br><br>[Figure 580]<br><br>[Figure 581]<br><br>[Figure 582]<br><br>[Figure 583]<br><br>[Figure 584]<br><br>[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]<br><br>[Figure 588]<br><br>[Figure 589]<br><br>[Figure 590]<br><br>[Figure 591]<br><br>[Figure 592]<br><br>[Figure 593]<br><br>[Figure 594]<br><br>[Figure 595]<br><br>[Figure 596]<br><br>[Figure 597]<br><br>[Figure 598]<br><br>[Figure 599]<br><br>[Figure 600]<br><br>[Figure 601]<br><br>[Figure 602]|
|---|

|[Figure 603]<br><br>[Figure 604]<br><br>[Figure 605]<br><br>[Figure 606]<br><br>[Figure 607]<br><br>[Figure 608]<br><br>[Figure 609]<br><br>[Figure 610]|
|---|

|[Figure 611]<br><br>[Figure 612]<br><br>[Figure 613]<br><br>[Figure 614]<br><br>[Figure 615]<br><br>[Figure 616]<br><br>[Figure 617]<br><br>[Figure 618]<br><br>[Figure 619]<br><br>[Figure 620]<br><br>[Figure 621]<br><br>[Figure 622]<br><br>[Figure 623]<br><br>[Figure 624]|
|---|

|[Figure 625]<br><br>[Figure 626]<br><br>[Figure 627]<br><br>[Figure 628]<br><br>[Figure 629]<br><br>[Figure 630]<br><br>[Figure 631]<br><br>[Figure 632]|
|---|

|[Figure 633]<br><br>[Figure 634]<br><br>[Figure 635]<br><br>[Figure 636]<br><br>[Figure 637]<br><br>[Figure 638]<br><br>[Figure 639]<br><br>[Figure 640]<br><br>[Figure 641]<br><br>[Figure 642]<br><br>[Figure 643]<br><br>[Figure 644]<br><br>[Figure 645]<br><br>[Figure 646]|
|---|

|[Figure 647]<br><br>[Figure 648]<br><br>[Figure 649]<br><br>[Figure 650]<br><br>[Figure 651]<br><br>[Figure 652]<br><br>[Figure 653]<br><br>[Figure 654]<br><br>[Figure 655]<br><br>[Figure 656]<br><br>[Figure 657]<br><br>[Figure 658]<br><br>[Figure 659]<br><br>[Figure 660]<br><br>[Figure 661]<br><br>[Figure 662]<br><br>[Figure 663]<br><br>[Figure 664]<br><br>[Figure 665]<br><br>[Figure 666]<br><br>[Figure 667]<br><br>[Figure 668]<br><br>[Figure 669]<br><br>[Figure 670]|
|---|

|[Figure 671]<br><br>[Figure 672]<br><br>[Figure 673]<br><br>[Figure 674]<br><br>[Figure 675]<br><br>[Figure 676]<br><br>[Figure 677]<br><br>[Figure 678]|
|---|

|[Figure 679]<br><br>[Figure 680]<br><br>[Figure 681]<br><br>[Figure 682]<br><br>[Figure 683]<br><br>[Figure 684]<br><br>[Figure 685]<br><br>[Figure 686]<br><br>[Figure 687]<br><br>[Figure 688]<br><br>[Figure 689]<br><br>[Figure 690]<br><br>[Figure 691]<br><br>[Figure 692]|
|---|

|[Figure 693]<br><br>[Figure 694]<br><br>[Figure 695]<br><br>[Figure 696]<br><br>[Figure 697]<br><br>[Figure 698]<br><br>[Figure 699]<br><br>[Figure 700]<br><br>[Figure 701]<br><br>[Figure 702]<br><br>[Figure 703]<br><br>[Figure 704]<br><br>[Figure 705]<br><br>[Figure 706]<br><br>[Figure 707]<br><br>[Figure 708]<br><br>[Figure 709]<br><br>[Figure 710]<br><br>[Figure 711]<br><br>[Figure 712]<br><br>[Figure 713]<br><br>[Figure 714]<br><br>[Figure 715]<br><br>[Figure 716]|
|---|

- Fig. 5: Examples of generated multi-scene videos by ModelScopeT2V [55], VideoDirectorGPT [26] and our VideoStudio utilizing a multi-scene prompt from the Coref-SV dataset. For each video, only the first four scenes are given. The results of VideoDirectorGPT are provided in the project webpage and thus with bounding box annotation.

eration on the WebVid-10M dataset. With the same scene-reference images, VideoStudio-Vid− outperforms most image-to-video approaches and obtains comparable FVD performance with the strong baseline SVD. The competitive result is attributed to the deep network architecture and large-scale training set. The performance is further enhanced to 116.5 FVD and 98.8 frame consistency by RF+VideoStudio-Vid, verifying the superiority of involving action category guidance to improve visual quality and intra-scene consistency.

Similar performance trends are observed on MSR-VTT dataset, as summarized in Table 3. The methods in this table are grouped into two categories: the methods with or without real frame (RF) as reference. To compare with the generation models without RF, we develop a two-step solution that first generates the scene-reference image by Stable Diffusion, and then converts the image into a video clip by VideoStudio-Vid, which is denoted as SD+VideoStudio-Vid. Specifically, VideoStudio-Vid attains the best FVD on both settings with and without a real frame as reference. SD+VideoStudio-Vid is slightly inferior to ModelScopeT2V in FID. We speculate that this may be the result of not optimizing Stable Diffusion on video frames, resulting in poorer frame quality against ModelScopeT2V. Nevertheless, SD+VideoStudio-Vid ap-

- Table 4: Performance comparisons for multi-scene video generation on ActivityNet Captions dataset.

Approach FID (↓) FVD (↓) Scene Consis. (↑)

ModelScopeT2V [55] 18.1 980 46.0 VideoDirectorGPT [26] 16.5 805 64.8

VideoStudio w/o Ref. 17.3 624 50.8 VideoStudio 13.2 395 75.1

Foreground Reference Image

Background Reference Image

Output video Foreground Reference Image

|[Figure 717]<br><br>[Figure 718]<br><br>[Figure 719]<br><br>[Figure 720]<br><br>[Figure 721]<br><br>[Figure 722]<br><br>[Figure 723]<br><br>[Figure 724]<br><br>[Figure 725]<br><br>[Figure 726]<br><br>[Figure 727]|
|---|

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

[Figure 842]

[Figure 843]

Scene-1: The cute Shiba lies in the room

|[Figure 844]<br><br>[Figure 845]<br><br>[Figure 846]<br><br>[Figure 847]<br><br>[Figure 848]<br><br>[Figure 849]<br><br>[Figure 850]<br><br>[Figure 851]<br><br>[Figure 852]<br><br>[Figure 853]<br><br>[Figure 854]|
|---|

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

[Figure 901]

[Figure 902]

[Figure 903]

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

Scene-2: The cute Shiba with smile sits in the car and goes to a place

|[Figure 970]<br><br>[Figure 971]<br><br>[Figure 972]<br><br>[Figure 973]<br><br>[Figure 974]<br><br>[Figure 975]<br><br>[Figure 976]<br><br>[Figure 977]<br><br>[Figure 978]<br><br>[Figure 979]<br><br>[Figure 980]|
|---|

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

- Scene-3: The cute Shiba plays in flowers
- Scene-4: The cute Shiba rests next to a tree

|[Figure 1096]<br><br>[Figure 1097]<br><br>[Figure 1098]<br><br>[Figure 1099]<br><br>[Figure 1100]<br><br>[Figure 1101]<br><br>[Figure 1102]<br><br>[Figure 1103]<br><br>[Figure 1104]<br><br>[Figure 1105]<br><br>[Figure 1106]|
|---|

[Figure 1107]

[Figure 1108]

[Figure 1109]

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

Background Reference Image

Output video

|[Figure 1222]<br><br>[Figure 1223]<br><br>[Figure 1224]<br><br>[Figure 1225]<br><br>[Figure 1226]<br><br>[Figure 1227]<br><br>[Figure 1228]<br><br>[Figure 1229]<br><br>[Figure 1230]<br><br>[Figure 1231]<br><br>[Figure 1232]<br><br>[Figure 1233]<br><br>[Figure 1234]<br><br>[Figure 1235]<br><br>[Figure 1236]<br><br>[Figure 1237]<br><br>[Figure 1238]<br><br>[Figure 1239]<br><br>[Figure 1240]<br><br>[Figure 1241]<br><br>[Figure 1242]|
|---|

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

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

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

Scene-1: The cute cat lies in the room

|[Figure 1298]<br><br>[Figure 1299]<br><br>[Figure 1300]<br><br>[Figure 1301]<br><br>[Figure 1302]<br><br>[Figure 1303]<br><br>[Figure 1304]<br><br>[Figure 1305]<br><br>[Figure 1306]<br><br>[Figure 1307]<br><br>[Figure 1308]<br><br>[Figure 1309]<br><br>[Figure 1310]<br><br>[Figure 1311]<br><br>[Figure 1312]<br><br>[Figure 1313]<br><br>[Figure 1314]<br><br>[Figure 1315]<br><br>[Figure 1316]<br><br>[Figure 1317]<br><br>[Figure 1318]|
|---|

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

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

Scene-2: The cute cat with smile sits in the car and goes to a place

|[Figure 1373]<br><br>[Figure 1374]<br><br>[Figure 1375]<br><br>[Figure 1376]<br><br>[Figure 1377]<br><br>[Figure 1378]<br><br>[Figure 1379]<br><br>[Figure 1380]<br><br>[Figure 1381]<br><br>[Figure 1382]<br><br>[Figure 1383]<br><br>[Figure 1384]<br><br>[Figure 1385]<br><br>[Figure 1386]<br><br>[Figure 1387]<br><br>[Figure 1388]<br><br>[Figure 1389]<br><br>[Figure 1390]<br><br>[Figure 1391]<br><br>[Figure 1392]<br><br>[Figure 1393]|
|---|

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

- Scene-3: The cute cat plays in flowers
- Scene-4: The cute cat rests next to a tree

|[Figure 1448]<br><br>[Figure 1449]<br><br>[Figure 1450]<br><br>[Figure 1451]<br><br>[Figure 1452]<br><br>[Figure 1453]<br><br>[Figure 1454]<br><br>[Figure 1455]<br><br>[Figure 1456]<br><br>[Figure 1457]<br><br>[Figure 1458]<br><br>[Figure 1459]<br><br>[Figure 1460]<br><br>[Figure 1461]<br><br>[Figure 1462]<br><br>[Figure 1463]<br><br>[Figure 1464]<br><br>[Figure 1465]<br><br>[Figure 1466]<br><br>[Figure 1467]<br><br>[Figure 1468]|
|---|

[Figure 1469]

[Figure 1470]

[Figure 1471]

[Figure 1472]

[Figure 1473]

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

[Figure 1490]

[Figure 1491]

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

- Fig. 6: Two examples of generated multi-scene videos by our VideoStudio using the real images as entity reference images.

parently surpasses ModelScopeT2V in FVD, validating the video-level quality by VideoStudio-Vid.

To evaluate the effectiveness of the action category condition for motion generation, we additionally implement an ablation study on the recent released VBench [19] benchmark. We measure the action score in VBench to assess whether human subjects can accurately execute the specific action mentioned in the text prompts. By using the action category as the condition in video diffusion, the action score of VideoStudio-Vid is improved from 90.3% to 96.5%, indicating the efficacy of action category condition to emphasize motion patterns.

#### 4.5 Evaluations on Multi-Scene Video Generation

We validate VideoStudio for multi-scene video generation on ActivityNet Captions and Coref-SV datasets. Both of the datasets consist of multi-scene prompts, which necessitate the LLM to write the video script based on the given prompt of each scene. We compare with three approaches: ModelScopeT2V, VideoDirectorGPT and VideoStudio w/o Ref. by disabling the reference images in VideoStudio. Table 4 details the performance comparisons on ActivityNet Captions. As indicated by the results in the table, VideoStudio exhibits superior visual quality and better cross-scene consistency. Specifically, VideoStudio surpasses VideoStudio w/o Ref. by 24.3 scene consistency, which essentially verifies the effectiveness of incorporating entity reference images. Moreover, VideoStudio leads to 10.3 and 29.1 improvements in scene consistency over VideoDirectorGPT and Mod-

###### Table 5: Performance comparisons for multi-scene video generation on Coref-SV.

###### Approach CLIPSIM (↑) Scene Consis. (↑)

ModelScopeT2V [55] 0.3021 37.9 VideoDirectorGPT [26] - 42.8

VideoStudio w/o Ref. 0.3103 40.9 VideoStudio 0.3304 77.3

Input Prompt: A man is cooking spaghetti Input Prompt: An old woman is walking around her house in the snow Input Prompt: A black hair boy is reading a book related to the monster

|[Figure 1523]<br><br>[Figure 1524]<br><br>[Figure 1525]<br><br>[Figure 1526]<br><br>[Figure 1527]<br><br>[Figure 1528]<br><br>[Figure 1529]<br><br>[Figure 1530]<br><br>[Figure 1531]<br><br>[Figure 1532]<br><br>[Figure 1533]<br><br>[Figure 1534]<br><br>[Figure 1535]<br><br>[Figure 1536]<br><br>[Figure 1537]<br><br>[Figure 1538]<br><br>[Figure 1539]<br><br>[Figure 1540]<br><br>[Figure 1541]<br><br>[Figure 1542]|
|---|

|[Figure 1543]<br><br>[Figure 1544]<br><br>[Figure 1545]<br><br>[Figure 1546]<br><br>[Figure 1547]<br><br>[Figure 1548]<br><br>[Figure 1549]<br><br>[Figure 1550]<br><br>[Figure 1551]<br><br>[Figure 1552]<br><br>[Figure 1553]|
|---|

|[Figure 1554]<br><br>[Figure 1555]<br><br>[Figure 1556]<br><br>[Figure 1557]<br><br>[Figure 1558]<br><br>[Figure 1559]<br><br>[Figure 1560]<br><br>[Figure 1561]<br><br>[Figure 1562]<br><br>[Figure 1563]<br><br>[Figure 1564]<br><br>[Figure 1565]<br><br>[Figure 1566]<br><br>[Figure 1567]<br><br>[Figure 1568]<br><br>[Figure 1569]<br><br>[Figure 1570]|
|---|

Scene-1: An old woman opens her window and gives a big yawn. She is out of her house which is in the middle of a forest. The whole land is covered with snow

Scene-1: A pot of water boils on a stove

Scene-1: A black hair boy is in the library

|[Figure 1571]<br><br>[Figure 1572]<br><br>[Figure 1573]<br><br>[Figure 1574]<br><br>[Figure 1575]<br><br>[Figure 1576]<br><br>[Figure 1577]<br><br>[Figure 1578]<br><br>[Figure 1579]<br><br>[Figure 1580]<br><br>[Figure 1581]<br><br>[Figure 1582]<br><br>[Figure 1583]<br><br>[Figure 1584]<br><br>[Figure 1585]<br><br>[Figure 1586]<br><br>[Figure 1587]<br><br>[Figure 1588]<br><br>[Figure 1589]<br><br>[Figure 1590]|
|---|

|[Figure 1591]<br><br>[Figure 1592]<br><br>[Figure 1593]<br><br>[Figure 1594]<br><br>[Figure 1595]<br><br>[Figure 1596]<br><br>[Figure 1597]<br><br>[Figure 1598]<br><br>[Figure 1599]<br><br>[Figure 1600]<br><br>[Figure 1601]|
|---|

|[Figure 1602]<br><br>[Figure 1603]<br><br>[Figure 1604]<br><br>[Figure 1605]<br><br>[Figure 1606]<br><br>[Figure 1607]<br><br>[Figure 1608]<br><br>[Figure 1609]<br><br>[Figure 1610]<br><br>[Figure 1611]<br><br>[Figure 1612]<br><br>[Figure 1613]<br><br>[Figure 1614]<br><br>[Figure 1615]<br><br>[Figure 1616]<br><br>[Figure 1617]<br><br>[Figure 1618]|
|---|

Scene-2: The old woman is introducing herself to the audiences with a nice smile. The old woman is in her cabin

Scene-2: A man shows us a package of spaghetti then adds it to the boiling water

Scene-2: The black hair boy is starting to read a book about monster

|[Figure 1619]<br><br>[Figure 1620]<br><br>[Figure 1621]<br><br>[Figure 1622]<br><br>[Figure 1623]<br><br>[Figure 1624]<br><br>[Figure 1625]<br><br>[Figure 1626]<br><br>[Figure 1627]<br><br>[Figure 1628]<br><br>[Figure 1629]<br><br>[Figure 1630]<br><br>[Figure 1631]<br><br>[Figure 1632]<br><br>[Figure 1633]<br><br>[Figure 1634]<br><br>[Figure 1635]<br><br>[Figure 1636]<br><br>[Figure 1637]<br><br>[Figure 1638]|
|---|

|[Figure 1639]<br><br>[Figure 1640]<br><br>[Figure 1641]<br><br>[Figure 1642]<br><br>[Figure 1643]<br><br>[Figure 1644]<br><br>[Figure 1645]<br><br>[Figure 1646]<br><br>[Figure 1647]<br><br>[Figure 1648]<br><br>[Figure 1649]|
|---|

|[Figure 1650]<br><br>[Figure 1651]<br><br>[Figure 1652]<br><br>[Figure 1653]<br><br>[Figure 1654]<br><br>[Figure 1655]<br><br>[Figure 1656]<br><br>[Figure 1657]<br><br>[Figure 1658]<br><br>[Figure 1659]<br><br>[Figure 1660]<br><br>[Figure 1661]<br><br>[Figure 1662]<br><br>[Figure 1663]|
|---|

Scene-3: The old woman is walking around her house. The whole land is covered with snow

Scene-3: In the story, at night, a big scary monster has appeared in village

Scene-3: The man stirs the spaghetti in the pot

|[Figure 1664]<br><br>[Figure 1665]<br><br>[Figure 1666]<br><br>[Figure 1667]<br><br>[Figure 1668]<br><br>[Figure 1669]<br><br>[Figure 1670]<br><br>[Figure 1671]<br><br>[Figure 1672]<br><br>[Figure 1673]<br><br>[Figure 1674]<br><br>[Figure 1675]<br><br>[Figure 1676]<br><br>[Figure 1677]<br><br>[Figure 1678]<br><br>[Figure 1679]<br><br>[Figure 1680]<br><br>[Figure 1681]<br><br>[Figure 1682]<br><br>[Figure 1683]|
|---|

|[Figure 1684]<br><br>[Figure 1685]<br><br>[Figure 1686]<br><br>[Figure 1687]<br><br>[Figure 1688]<br><br>[Figure 1689]<br><br>[Figure 1690]<br><br>[Figure 1691]<br><br>[Figure 1692]<br><br>[Figure 1693]<br><br>[Figure 1694]|
|---|

|[Figure 1695]<br><br>[Figure 1696]<br><br>[Figure 1697]<br><br>[Figure 1698]<br><br>[Figure 1699]<br><br>[Figure 1700]<br><br>[Figure 1701]<br><br>[Figure 1702]<br><br>[Figure 1703]<br><br>[Figure 1704]<br><br>[Figure 1705]<br><br>[Figure 1706]<br><br>[Figure 1707]<br><br>[Figure 1708]|
|---|

Scene-4: The man drains the spaghetti and rinses it before putting it into a bowl

Scene-4: The old woman is making a snowball. The woods are covered with snow

Scene-4: The black hair boy is still reading the book at the story's night background

- Fig. 7: Examples of generated multi-scene videos by VideoStudio on MSR-VTT. For each video, only the first four scenes are given.

elScopeT2V, respectively. Similar results are also observed on Coref-SV dataset, as summarized in Table 5. Note that as Coref-SV only offers prompts without the corresponding videos, FID and FVD cannot be measured for this case. As shown in the table, VideoStudio again achieves the highest cross-scene consistency of 77.3, making an absolute improvement of 39.4 and 34.5 over ModelScopeT2V and VideoDirectorGPT. Figure 5 showcases an example of generated four-scene videos by different approaches on Coref-SV, manifesting the ability of VideoStudio on generating visually similar entities (e.g., mouse/garden) across scenes. Figure 6 further shows two examples of multi-scene video generation by VideoStudio using the real images as entity reference images, which demonstrates the potential of VideoStudio in customizing the generated objects or environments.

#### 4.6 Human Evaluation

Multi-Scene Video Quality. In this section, we conduct a human study to evaluate the entire process of generating multi-scene video from a single prompt. We compare our VideoStudio with four approaches: ModelScopeT2V w/o LLM and VideoStudio w/o Ref. w/o LLM to generate five scenes by duplicating the input prompt, ModelScopeT2V w/ LLM and VideoStudio w/o Ref. to utilize LLM to provide video script as described in Sec. 3.1 while generate each scene individually. We invite 12 evaluators and randomly select 100 prompts from MSR-VTT validation set for human evaluation. We show all the evaluators the five videos generated by each approach plus the given prompt and ask them to rank the five videos from 1 to 5 (good to bad) with respect to the three criteria: visual quality (VQ), logical coherence (LC) and content

###### Table 6: The user study on three criteria: visual quality (VQ), logical coherence (LC) and content consistency (CC).

Approach VQ (↓) LC (↓) CC (↓)

ModelScopeT2V w/o LLM 4.5 4.7 3.9 ModelScopeT2V w/ LLM 4.5 3.8 4.2

VideoStudio w/o Ref. w/o LLM 2.0 3.0 2.3 VideoStudio w/o Ref. 2.4 2.3 3.4 VideoStudio 1.6 1.2 1.2

###### Table 7: User preferences on script/videos by using different LLMs in VideoStudio.

ChatGLM3-6B [8] GPT-4 [38] Tie

Video Script 25% 37% 38% Multi-Scene Video 20% 21% 59%

consistency (CC). For each approach, we average the ranking on each criterion of all the generated videos. As indicated by the results in Table 6, the study proves the impact of LLM in generating video script and entity reference images to improve logical coherence and content consistency, respectively. Figure 7 illustrates the examples of the generated multi-scene videos by our VideoStudio.

Different LLMs. To further investigate the effectiveness of different LLMs for multi-scene video generation, we carried out an ablation study on a variant of VideoStudio with GPT-4 [38] in Table 7. Evaluators vote on the preferring video text script by using ChatGLM3-6B and GPT-4, and the corresponding multi-scene videos generated by VideoStudio. “Tie” refers to a close preference. The results indicate that the video script generated by GPT-4 is of higher quality than ChatGLM3-6B. This is not surprising given the significantly larger parameters of GPT-4 (∼1T v.s. 6B). Nevertheless, the voting on multi-scene videos is comparable, showing that the use of an open-source LLM does not affect video quality much. Our exploitation of open-source LLM leads to an elegant view of how responses of light-weight LLM could be improved for video script generation.

### 5 Conclusions

We have presented a new VideoStudio framework for consistent-content and multi-scene video generation. VideoStudio involves LLM to benefit from the logical knowledge learnt behind and rewrite the input prompt into a multi-scene video script. Then, VideoStudio identifies common entities throughout the script and generates a reference image for each entity, which serves as the link across scenes to ensure the appearance consistency. To produce a multi-scene video, VideoStudio devises two diffusion models of VideoStudio-Img and VideoStudioVid. VideoStudio-Img creates a scene-reference image for each scene based on the corresponding event prompt and entity reference images. VideoStudio-Vid converts the scene-reference image into a video clip conditioning on the specific action and camera movement. Extensive evaluations on four video benchmarks demonstrate the superior visual quality and content consistency by VideoStudio over SOTA models.

### References

- 1. Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval. In: ICCV (2021)
- 2. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V., Rombach, R.: Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127 (2023)
- 3. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. In: CVPR (2023)
- 4. Carreira, J., Zisserman, A.: Quo Vadis, Action Recognition? A New Model and The Kinetics Dataset. In: CVPR (2017)
- 5. Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., Weng, C., Shan, Y.: VideoCrafter1: Open Diffusion Models for HighQuality Video Generation. arXiv preprint arXiv:2310.19512 (2023)
- 6. Chen, Z., Long, F., Qiu, Z., Yao, T., Zhou, W., Luo, J., Mei, T.: Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video SuperResolution. In: CVPR (2024)
- 7. Dhariwal, P., Nichol, A.: Diffusion Models Beat GANs on Image Synthesis. In: NeurIPS (2021)
- 8. Du, Z., Qian, Y., Liu, X., Ding, M., Qiu, J., Yang, Z., Tang, J.: GLM: General Language Model Pretraining with Autoregressive Blank Infilling. In: ACL (2022)
- 9. Esser, P., Chiu, J., Atighehchian, P., Granskog, J., Germanidis, A.: Structure and Content-guided Video Synthesis with Diffusion Models. In: ICCV (2023)
- 10. Geyer, M., Bar-Tal, O., Bagon, S., Dekel, T.: TokenFlow: Consistent Diffusion Features for Consistent Video Editing. arXiv preprint arXiv:2307.10373 (2023)
- 11. Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., Dai, B.: AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. arXiv preprint arXiv:2307.04725 (2023)
- 12. He, Y., Yang, T., Zhang, Y., Shan, Y., Chen, Q.: Latent Video Diffusion Models for High-Fidelity Long Video Generation. arXiv preprint arXiv:2211.13221 (2022)
- 13. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In: NIPS (2017)
- 14. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., Salimans, T.: Imagen Video: High Definition Video Generation with Diffusion Models. arXiv preprint arXiv:2210.02303 (2022)
- 15. Ho, J., Jain, A., Abbeel, P.: Denoising Diffusion Probabilistic Models. In: NeurIPS

(2020)

- 16. Ho, J., Salimans, T.: Classifier-Free Diffusion Guidance. arXiv preprint arXiv:2207.12598 (2022)
- 17. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: CogVideo: Large-Scale Pretraining for Text-to-Video Generation via Transformers. In: ICLR (2023)
- 18. Hu, Z., Xu, D.: VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet. arXiv preprint arXiv:2307.14073 (2023)
- 19. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive Benchmark Suite for Video Generative Models. arXiv preprint arXiv:2311.17982 (2023)

- 20. Khachatryan, L., Movsisyan, A., Tadevosyan, V., Henschel, R., Wang, Z., Navasardyan, S., Shi, H.: Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators. In: ICCV (2023)
- 21. Kim, K.M., Heo, M.O., Choi, S.H., Zhang, B.T.: DeepStory: Video Story QA by Deep Embedded Memory Networks. In: IJCAI (2017)
- 22. Krishna, R., Hata, K., Ren, F., Fei-Fei, L., Niebles, J.C.: Dense-Captioning Events in Videos. In: ICCV (2017)
- 23. Li, Y., Yao, T., Pan, Y., Mei, T.: Contextual Transformer Networks for Visual Recognition. IEEE Trans. on PAMI (2022)
- 24. Li, Y., Gan, Z., Shen, Y., Liu, J., Cheng, Y., Wu, Y., Carin, L., Carlson, D., Gao, J.: StoryGAN: A Sequential Conditional GAN for Story Visualization. In: CVPR

(2019)

- 25. Liang, J., Wu, C., Hu, X., Gan, Z., Wang, J., Wang, L., Liu, Z., Fang, Y., Duan, N.: NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis. In: NeurIPS (2022)
- 26. Lin, H., Zala, A., Cho, J., Bansal, M.: VideoDirectorGPT: Consistent Multi-Scene Video Generation via LLM-Guided Planning. arXiv preprint arXiv:2309.15091

(2023)

- 27. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding DINO: Marrying Dino with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499 (2023)
- 28. Long, F., Qiu, Z., Pan, Y., Yao, T., Luo, J., Mei, T.: Stand-Alone Inter-Frame Attention in Video Models. In: CVPR (2022)
- 29. Long, F., Qiu, Z., Pan, Y., Yao, T., Ngo, C.W., Mei, T.: Dynamic Temporal Filtering in Video Models. In: ECCV (2022)
- 30. Long, F., Yao, T., Qiu, Z., Tian, X., Luo, J., Mei, T.: Gaussian Temporal Awareness Networks for Action Localization. In: CVPR (2019)
- 31. Long, F., Yao, T., Qiu, Z., Tian, X., Luo, J., Mei, T.: Bi-calibration Networks for Weakly-Supervised Video Representation Learning. IJCV (2023)
- 32. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. In: NeurIPS (2022)
- 33. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. arXiv preprint arXiv:2211.01095 (2023)
- 34. Luo, Z., Chen, D., Zhang, Y., Huang, Y., Wang, L., Shen, Y., Zhao, D., Zhou, J., Tan, T.: VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation. In: CVPR (2023)
- 35. Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y., Qie, X.: T2IAdapter: Learning Adapters to Dig out More Controllable Ability for Text-toImage Diffusion Models. arXiv preprint arXiv:2302.08453 (2023)
- 36. Nichol, A., Dhariwal, P.: Improved Denoising Diffusion Probabilistic Models. In: ICML (2021)
- 37. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. In: ICML (2022)
- 38. OpenAI: GPT-4 Technical Report (2023)
- 39. Ouyang, H., Wang, Q., Xiao, Y., Bai, Q., Zhang, J., Zheng, K., Zhou, X., Chen, Q., Shen, Y.: CoDeF: Content Deformation Fields for Temporally Consistent Video Processing. arXiv preprint arXiv:2308.07926 (2023)
- 40. Qi, C., Cun, X., Zhang, Y., Lei, C., Wang, X., Shan, Y., Chen, Q.: FateZero: Fusing Attentions for Zero-shot Text-based Video Editing. In: ICCV (2023)

- 41. Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O., Jagersand, M.: U2Net: Going Deeper with Nested U-Structure for Salient Object Detection. Pattern Recognition (2020)
- 42. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning Transferable Visual Models from Natural Language Supervision. In: ICML (2021)
- 43. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv preprint arXiv:2204.06125 (2022)
- 44. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-Resolution Image Synthesis with Latent Diffusion Models. In: CVPR (2022)
- 45. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models. In: NeurIPS

(2022)

- 46. Shin, C., Kim, H., Lee, C.H., gil Lee, S., Yoon, S.: Edit-A-Video: Single Video Editing with Object-Aware Consistency. arXiv preprint arXiv:2303.07945 (2023)
- 47. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., Taigman, Y.: Make-a-video: Textto-Video Generation without Text-Video Data. arXiv preprint arXiv:2209.14792

(2022)

- 48. Sohl-Dickstein, J., Weiss, E.A., Maheswaranathan, N., Ganguli, S.: Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In: ICML (2015)
- 49. Song, J., Meng, C., Ermon, S.: Denoising Diffusion Implicit Models. In: ICLR

(2021)

- 50. Song, Y., Ermon, S.: Generative Modeling by Estimating Gradients of the Data Distribution. In: NeurIPS (2019)
- 51. Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: FVD: A New Metric for Video Generation. In: ICLR Workshop (2019)
- 52. Villegas, R., Babaeizadeh, M., Kindermans, P.J., Moraldo, H., Zhang, H., Saffar, M.T., Castro, S., Kunze, J., Erhan, D.: Phenaki: Variable Length Video Generation from Open Domain Textual Description. In: ICLR (2023)
- 53. Voleti, V., Jolicoeur-Martineau, A., Pal, C.: MCVD-Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation. In: NeurIPS (2022)
- 54. Wang, F.Y., Chen, W., Song, G., Ye, H.J., Liu, Y., Li, H.: Gen-L-Video: MultiText to Long Video Generation via Temporal Co-Denoising. arXiv preprint arXiv:2305.18264 (2023)
- 55. Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., Zhang, S.: ModelScope Textto-Video Technical Report. arXiv preprint arXiv:2308.06571 (2023)
- 56. Wang, W., Yang, H., Tuo, Z., He, H., Zhu, J., Fu, J., Liu, J.: VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation. arXiv preprint arXiv:2305.10874 (2023)
- 57. Wang, X., Yuan, H., Zhang, S., Chen, D., Wang, J., Zhang, Y., Shen, Y., Zhao, D., Zhou, J.: VideoComposer: Compositional Video Synthesis with Motion Controllability. In: NeurIPS (2023)
- 58. Wu, C., Huang, L., Zhang, Q., Li, B., Ji, L., Yang, F., Sapiro, G., Duan, N.: GODIVA: Generating Open-Domain Videos from Natural Descriptions. arXiv preprint arXiv:2104.14806 (2021)
- 59. Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. In: ICCV (2023)

- 60. Xing, J., Xia, M., Zhang, Y., Chen, H., Yu, W., Liu, H., Wang, X., Wong, T.T., Shan, Y.: DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors. arXiv preprint arXiv:2310.12190 (2023)
- 61. Xu, J., Mei, T., Yao, T., Rui, Y.: MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In: CVPR (2016)
- 62. Yao, T., Li, Y., Pan, Y., Wang, Y., Zhang, X.P., Mei, T.: Dual Vision Transformer. IEEE Trans. on PAMI (2023)
- 63. Yao, T., Pan, Y., Li, Y., Ngo, C.W., Mei, T.: Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning. In: ECCV (2022)
- 64. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv preprint arXiv:2308.06721 (2023)
- 65. Yin, S., Wu, C., Liang, J., Shi, J., Li, H., Ming, G., Duan, N.: Dragnuwa: Finegrained Control in Video Generation by Integrating Text, Image, and Trajectory. arXiv preprint arXiv:2308.08089 (2023)
- 66. Yin, S., Wu, C., Yang, H., Wang, J., Wang, X., Ni, M., Yang, Z., Li, L., Liu, S., Yang, F., Fu, J., Ming, G., Wang, L., Liu, Z., Li, H., Duan, N.: NUWAXL: Diffusion over Diffusion for eXtremely Long Video Generation. arXiv preprint arXiv:2303.12346 (2023)
- 67. Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., et al.: GLM-130B: An Open Bilingual Pre-Trained Model. arXiv preprint arXiv:2210.02414 (2022)
- 68. Zhang, L., Rao, A., Agrawala, M.: Adding Conditional Control to Text-to-Image Diffusion Models. In: ICCV (2023)
- 69. Zhang, S., Wang, J., Zhang, Y., Zhao, K., Yuan, H., Qin, Z., Wang, X., Zhao, D., Zhou, J.: I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models. arXiv preprint arXiv:2311.04145 (2023)
- 70. Zhang, Z., Long, F., Pan, Y., Qiu, Z., Yao, T., Cao, Y., Mei, T.: TRIP: Temporal Residual Learning with Image Noise Prior for Image-to-Video Diffusion Models. In: CVPR (2024)
- 71. Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., Feng, J.: Magicvideo: Efficient Video Generation with Latent Diffusion Models. arXiv preprint arXiv:2211.11018

(2022)

## VideoStudio: Generating Consistent-Content and Multi-Scene Videos — ECCV 2024 Supplementary Material

##### Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei

HiDream.ai Inc. {longfuchen, qiuzhaofan, tiyao, tmei}@hidream.ai

The supplementary material contains: 1) the instructions of LLM; 2) the implementation details of VideoStudio-Img; 3) the implementation details of VideoStudio-Vid; 4) performance contribution of VideoStudio; 5) more video examples generated by VideoStudio; 6) a video demo for VideoStudio.

### 1 Instructions of LLM

The LLM instructions, output examples and in-context examples for video script and entity description generation are given in Figure 2 and Figure 3, respectively. The multi-round dialogue for entity description generation is shown in Figure 4.

### 2 Implementation details of VideoStudio-Img

VideoStudio-Img is constructed on Stable Diffusion v2.1 model by incorporating the two additional cross-attention modules. Table 1 details the structures of VideoStudio-Img. We utilize the CLIP ViT-H/14 [42] as the text and visual encoder to extract text features from text prompt, and local image features from foreground/background reference image, respectively. The sequence length Lt of the text features is 77 while the length Lf/Lb of foreground/background image features is set as 256. The cross attention dimension Ct and Cf/Cb are set as the default number in each block of the original diffusion model.

### 3 Implementation details of VideoStudio-Vid

We build the 3D UNet of VideoStudio-Vid by inserting temporal transformer and temporal convolution layers into 2D UNet of SD-XL. Table 2 details the hyperparameters and structures of VideoStudio-Vid. We employ the CLIP ViT-H/14 as the visual encoder to extract image features from scene-reference images. To enhance the visual alignment between the scene-reference image and synthesized video, we concatenate the latent code of the scene-reference image with the noisy video latent code along temporal dimension as the input of 3D UNet.

Action condition. In the stage of model training, the VideoMAE [?] finetuned on Kinetics-400 [4] is leveraged as the action classifier to measure the action probability (i.e., indicator vector) ya of input videos. A linear embedding

- Table 1: Detailed hyper-parameters and structures of VideoStudio-Img. Hyper-parameter Value Hyper-parameter Value Base structure SD v2.1 Spatial transformer blocks [1, 1, 1, 1] Latent shape 4 × 64 × 64 Image embed sequence 256 Channels 320 Text CLIP CLIP ViT-H/14 Layers per block 2 Parameterization ϵ Channel multiplier [1, 2, 4, 4] Diffusion steps 1000 Attention resolutions [64, 32, 16] Noise schedule Scaled Linear Head channels [5, 10, 20] β1 0.00085 Number of heads 64 βT 0.0120 CA embed dim 1024 Sampler DDIM CA resolutions [64, 32, 16] Inference steps 50 Autoencoders AutoKL GPU Type A100-80G Image CLIP CLIP ViT-H/14 GPU Number 64 Learning rate 1 × 10−4 Train steps 20K Total batch size 512 # of UNet params 915M

- Table 2: Detailed hyper-parameters and structures of VideoStudio-Vid.

Hyper-parameter Value Hyper-parameter Value

Base structure SD-XL Spatial transformer blocks [0, 2, 10] Latent shape 4 × 16 × 40 × 64 Temporal transformer blocks [0, 2, 10] Channels 320 Temporal SA head number 64 Layers per block 2 Diffusion steps 1000 Channel multiplier [1, 2, 4] Noise schedule Scaled Linear Attention resolutions [32, 16] β1 0.00085

Head channels [10, 20] βT 0.0120 Number of heads 64 Sampler DDIM

CA embed dim 1280 Inference steps 70 CA resolutions [32, 16] η 1.0 Autoencoders AutoKL Guidance scale 12.0 Image CLIP CLIP ViT-H/14 GPU Type A100-80G Parameterization ϵ GPU Number 64 Learning rate 3 × 10−6 Train steps 480K Total batch size 128 # of 3D-UNet params 4.7B

50

MSE on Optical Flow

40

MSE

30

20

10

0 2 4 6 8 10

Trade-off Parameter Tm

Fig. 1: The impact of trade-off parameter Tm for camera movement.

is then learnt on the probability ya and further treated as a condition to adjust the video diffusion as demonstrated in the main paper. In inference stage, we use the spaCy library to extract all action phrases from the input text prompt. Next, the text features of the action phrases are obtained by using CLIP model, which are further exploited for cosine similarity computation with action vocabulary of Kinetics-400. For each action phrase, we choose the action category with the max cosine similarity score. If the cosine similarity is lower than 0.2, the action category will be dropped. After collecting all action categories and corresponding cosine similarity, we construct the action indicator vector ya ∈ [0,1]400 by assigning the normalized cosine similarity into the corresponding category index.

Camera movement. We control the camera movement of each scene video

during the inference process of VideoStudio-Vid. Specifically, at inference timestep

t, the noisy video xt = αtx0 + σtϵˆt is decomposed into the clean video x0 with an estimated noise ϵˆt = ϵθ (xt,t) with fixed scheduling weights αt and σt. The noisy video xt is transformed into a video xt−1 with reduced noise:

xt−1 = sampling(xt, ϵˆt, t), (1)

where xT represents the pure noise ϵT. sampling is the DDIM [49] update strategy. After the first Tm steps, we execute an adjustment to the noisy video x(T

m−1)

to maintain the camera movement indicated by the video script as

xˆ0 = (x(Tm−1) − σ(Tm−1)ϵˆTm)/α(Tm−1), x0 = 0.5 × xˆ0 + 0.5 × warp(xˆ0, flow), x(Tm−1) = α(Tm−1)x0 + σ(Tm−1)ϵTˆm,

(2)

where warp(xˆ0,flow) is to warp the frames in xˆ0 based on the optical flow of required camera movement, and x(T

m−1) is the modified noisy video. Such modification ensures that the resultant video clip maintains the same camera movement as we warp the intermediate frames. To analysis the impact of hyperparameter Tm, we conduct the experiments on the MSR-VTT dataset and calculate the mean squared error (MSE) between the optical flow of generated and

###### Table 3: Performance comparison on ActivityNet Captions dataset.

Approach Ref Training Data Architecture FID (↓) FVD (↓) Scene Consis. (↑)

ModelScopeT2V [55] LAION + WebVid-10M SD-2.1 18.1 980 46.0 VideoDirectorGPT [26] LAION + WebVid-10M + GLIGEN SD-2.1 16.5 805 64.8

LAION + WebVid-10M SD-XL 17.7 789 49.1 LAION + WebVid-10M + HD-VG SD-XL 17.3 624 50.8 ✓ LAION + WebVid-10M + HD-VG SD-XL 13.2 395 75.1

VideoStudio

target videos with different Tm, as shown in Figure 1. In general, setting a small Tm for early modification may not effectively control the camera movement, while a late modification may affect the visual quality of the generated videos. As indicated by the figure, Tm=5 provides a good trade-off empirically.

### 4 Performance Contribution of VideoStudio

To ablate the performance contribution of VideoStudio more transparent from different perspectives (e.g., with or without reference image, training data and architecture), we report the performances of different VideoStudio variants on ActivityNet Captions dataset in Table 3. The first variant is trained on LAION and WebVid-10M, which is the same as in ModelScopeT2V. The improvement over ModelScopeT2V in video quality (FID & FVD) is due to the deeper Stable Diffusion (SD) backbone and the two-stage (T2I+I2V) framework. The second variant further utilizes HD-VG dataset for model training and leads to slightly better video quality. The full version of VideoStudio takes entity reference images into consideration and improves both video quality and cross-scene consistency.

### 5 More Video Examples

Here, we present more examples of multi-scene videos generated by VideoStudio on MSR-VTT in Figure 5 and Figure 6 with single foreground reference image and multiple foreground reference images, respectively. For each example, the input prompt, camera movement, foreground/background reference images and generated multi-scene video are given. Figure 7 further showcases three examples of generated multi-scene videos by VideoStudio using the real images as the entity reference images including foreground and background reference images.

### 6 Video Demo

We have provided a video demo (VideoStudio.mp4) to illustrate the generated videos by VideoStudio in various scenarios. For more details, please refer to the offline project page (VideoStudio.html).

Instructions: Given a single text prompt, you need to envision a multi-scene video by generating a sequence of prompts to describe the text prompt. For each scene, you also need to generate the set of entities needed and describe the background scene where the video should occur. Related scenes should maintain similar entities and background scenes. Before you write each description, you must follow these instructions:

- 1. Each scene prompt must contain only a single motion or action.
- 2. Each scene prompt must include all relevant objects and describe the environment scene.
- 3. Make sure each scene prompt must be easy described by a shot video clip.
- 4. Gives 5-10 scenes in total.

Use the following format: [scene_1: [prompt, entities, background]; scene_2: [prompt, entities, background]; … scene_n: [prompt, entities, background]]

###### The example of output:

- [scene_1: [A woman walks along a village road; entities: woman; background: a picturesque coastal village with quaint houses and a view of the sea]]
- [scene_2: [The woman's hair gently blows in the sea breeze; entities: woman; background: a picturesque coastal village with quaint houses and a view of the sea]]

... In-context examples: User: A happy elderly couple is riding a tandem bicycle LLM:

- [scene_1: [The elderly couple mounts the tandem bicycle; entities: elderly couple, tandem bicycle; background: a scenic park with lush greenery and a winding path]]
- [scene_2: [They pedal together, their smiles beaming with joy; entities: elderly couple, tandem bicycle; background: a scenic park with lush greenery and a winding path]] …

- Fig. 2: The LLM instructions, output examples and in-context examples for generating scene prompts and common entities.

Instructions: Given descriptions of several video clips, you need to give the camera movement of each clip. You must follow these instructions:

- 1. The camera movement composes of direction (including static, moving left, moving right, moving up, moving down, zoom in and zoom out) and speed (including slow, medium and fast).
- 2. The camera movement should be diverse.

Use the following format:

[scene_1: [camera movement]; scene_2: [camera movement]; ... scene_n: [camera movement]] The example of output:

- [scene_1: [moving left, fast]]
- [scene_2: [zoom in, medium]]

... In-context examples: User:

- [scene_1: The father kicks the soccer ball towards the son.]
- [scene_2: The son receives the ball and dribbles towards the water.] … LLM:

- [scene_1: [static, slow]]
- [scene_2: [moving right, medium]] …

- Fig. 3: The LLM instructions, output examples and in-context examples for generating camera movements.

Multi-round dialogue for entity description generation

User: Give some aspects that should be considered when describing a photo of {entity name} in detail. LLM:

###### ... User:

As a professional photographer, give more aspects that should be considered when describing a photo of {entity name} in detail, e.g., theme,

composition, focal length and depth of field, details and texture, technology and post-processing, rendering technology, camera brand and model used, film type and characteristics, location and characteristics of light sources, reference to the master's work, etc. LLM: … User: Considering the above mentioned aspects, given you a sentence of video: "{input prompt}", give a description (single paragraph without

segmentation) for a photo of {entity name} in this video in detail. You must follow these instructions:

- 1. The description provided should be concise and detailed.
- 2. Prohibition of artistic appreciation and personal emotions.
- 3. While retaining the author's meaning, clearly supplement all aspects just mentioned.
- 4. It is prohibited to include vague descriptions such as "may" and "may".

###### Fig. 4: The multi-round dialogue of LLM to achieve detailed entity description.

Camera Movement

Entity Reference Image

Output Video

Input Prompt

|[Figure 1709]<br><br>[Figure 1710]<br><br>[Figure 1711]<br><br>[Figure 1712]<br><br>[Figure 1713]<br><br>[Figure 1714]<br><br>[Figure 1715]<br><br>[Figure 1716]<br><br>[Figure 1717]<br><br>[Figure 1718]<br><br>[Figure 1719]<br><br>[Figure 1720]<br><br>[Figure 1721]<br><br>[Figure 1722]|
|---|

[Figure 1723]

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

[Figure 1740]

slow forward

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

[Figure 1756]

[Figure 1757]

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

Scene-1: The video begins with a shot of a Spanish-speaking woman

[Figure 1766]

[Figure 1767]

Foreground: The photo depicts a woman sitting on a bench in a lush garden, surrounded by vibrant flora and fauna

|[Figure 1768]<br><br>[Figure 1769]<br><br>[Figure 1770]<br><br>[Figure 1771]<br><br>[Figure 1772]<br><br>[Figure 1773]<br><br>[Figure 1774]<br><br>[Figure 1775]<br><br>[Figure 1776]<br><br>[Figure 1777]<br><br>[Figure 1778]<br><br>[Figure 1779]<br><br>[Figure 1780]<br><br>[Figure 1781]<br><br>[Figure 1782]<br><br>[Figure 1783]|
|---|

medium right

Spanish language music video

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

[Figure 1790]

[Figure 1791]

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

Scene-2: A close-up of the woman's face, her eyes closed in concentration as she dances

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

[Figure 1808]

[Figure 1809]

[Figure 1810]

|[Figure 1811]<br><br>[Figure 1812]<br><br>[Figure 1813]<br><br>[Figure 1814]<br><br>[Figure 1815]<br><br>[Figure 1816]<br><br>[Figure 1817]<br><br>[Figure 1818]<br><br>[Figure 1819]<br><br>[Figure 1820]<br><br>[Figure 1821]<br><br>[Figure 1822]<br><br>[Figure 1823]<br><br>[Figure 1824]|
|---|

[Figure 1825]

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

slow backward

[Figure 1842]

Background: This photo captures the vibrant culture of Spain, showcasing a bustling street scene in the city of Barcelona

Scene-3: The video ends with a shot of the woman walking away

|[Figure 1843]<br><br>[Figure 1844]<br><br>[Figure 1845]<br><br>[Figure 1846]<br><br>[Figure 1847]<br><br>[Figure 1848]<br><br>[Figure 1849]<br><br>[Figure 1850]<br><br>[Figure 1851]<br><br>[Figure 1852]<br><br>[Figure 1853]<br><br>[Figure 1854]|
|---|

[Figure 1855]

[Figure 1856]

[Figure 1857]

[Figure 1858]

[Figure 1859]

[Figure 1860]

[Figure 1861]

[Figure 1862]

[Figure 1863]

[Figure 1864]

[Figure 1865]

[Figure 1866]

[Figure 1867]

[Figure 1868]

[Figure 1869]

[Figure 1870]

[Figure 1871]

[Figure 1872]

slow right

[Figure 1873]

[Figure 1874]

[Figure 1875]

[Figure 1876]

[Figure 1877]

[Figure 1878]

[Figure 1879]

[Figure 1880]

[Figure 1881]

[Figure 1882]

[Figure 1883]

[Figure 1884]

[Figure 1885]

[Figure 1886]

[Figure 1887]

[Figure 1888]

[Figure 1889]

[Figure 1890]

[Figure 1891]

[Figure 1892]

[Figure 1893]

[Figure 1894]

[Figure 1895]

[Figure 1896]

[Figure 1897]

- Scene-1: The baby girl reaches out to touch a stuffed animal
- Scene-2: The baby girl is tired and sleeping in the cradle

[Figure 1898]

[Figure 1899]

Foreground: The photo

|[Figure 1900]<br><br>[Figure 1901]<br><br>[Figure 1902]<br><br>[Figure 1903]<br><br>[Figure 1904]<br><br>[Figure 1905]<br><br>[Figure 1906]<br><br>[Figure 1907]<br><br>[Figure 1908]<br><br>[Figure 1909]<br><br>[Figure 1910]<br><br>[Figure 1911]<br><br>[Figure 1912]<br><br>[Figure 1913]|
|---|

depicts a baby girl sitting in a cradle, her tiny body supported by soft, white blankets

A baby girl is sitting in the cradle

static

[Figure 1914]

[Figure 1915]

[Figure 1916]

[Figure 1917]

[Figure 1918]

[Figure 1919]

[Figure 1920]

[Figure 1921]

[Figure 1922]

[Figure 1923]

[Figure 1924]

[Figure 1925]

[Figure 1926]

[Figure 1927]

[Figure 1928]

[Figure 1929]

[Figure 1930]

[Figure 1931]

[Figure 1932]

[Figure 1933]

[Figure 1934]

[Figure 1935]

[Figure 1936]

[Figure 1937]

[Figure 1938]

[Figure 1939]

[Figure 1940]

|[Figure 1941]<br><br>[Figure 1942]<br><br>[Figure 1943]<br><br>[Figure 1944]<br><br>[Figure 1945]<br><br>[Figure 1946]<br><br>[Figure 1947]<br><br>[Figure 1948]<br><br>[Figure 1949]<br><br>[Figure 1950]<br><br>[Figure 1951]<br><br>[Figure 1952]<br><br>[Figure 1953]<br><br>[Figure 1954]|
|---|

[Figure 1955]

[Figure 1956]

[Figure 1957]

[Figure 1958]

[Figure 1959]

[Figure 1960]

[Figure 1961]

[Figure 1962]

[Figure 1963]

[Figure 1964]

[Figure 1965]

[Figure 1966]

[Figure 1967]

[Figure 1968]

[Figure 1969]

[Figure 1970]

[Figure 1971]

[Figure 1972]

static

Background: This cozy bedroom photo features a soft bed with inviting, pastel-colored bedding and pillows

Scene-3: The mother gently picks up the baby girl and holds her close

|[Figure 1973]<br><br>[Figure 1974]<br><br>[Figure 1975]<br><br>[Figure 1976]<br><br>[Figure 1977]<br><br>[Figure 1978]<br><br>[Figure 1979]<br><br>[Figure 1980]<br><br>[Figure 1981]<br><br>[Figure 1982]<br><br>[Figure 1983]<br><br>[Figure 1984]<br><br>[Figure 1985]<br><br>[Figure 1986]<br><br>[Figure 1987]<br><br>[Figure 1988]|
|---|

[Figure 1989]

[Figure 1990]

[Figure 1991]

[Figure 1992]

[Figure 1993]

[Figure 1994]

[Figure 1995]

[Figure 1996]

[Figure 1997]

[Figure 1998]

[Figure 1999]

[Figure 2000]

[Figure 2001]

[Figure 2002]

[Figure 2003]

[Figure 2004]

[Figure 2005]

[Figure 2006]

slow right

[Figure 2007]

[Figure 2008]

[Figure 2009]

[Figure 2010]

[Figure 2011]

[Figure 2012]

[Figure 2013]

[Figure 2014]

[Figure 2015]

[Figure 2016]

[Figure 2017]

[Figure 2018]

[Figure 2019]

[Figure 2020]

[Figure 2021]

[Figure 2022]

[Figure 2023]

[Figure 2024]

[Figure 2025]

[Figure 2026]

[Figure 2027]

[Figure 2028]

[Figure 2029]

[Figure 2030]

[Figure 2031]

[Figure 2032]

- Scene-1: A person with red shirt mixes ingredients in a bowl
- Scene-2: The person with red shirt adds toppings to the dessert
- Scene-3: The person with red shirt places the dessert in the fridge

[Figure 2033]

Foreground: In the photo, a person with a red shirt is seen preparing dessert in the kitchen. The subject is the central focus of the image

|[Figure 2034]<br><br>[Figure 2035]<br><br>[Figure 2036]<br><br>[Figure 2037]<br><br>[Figure 2038]<br><br>[Figure 2039]<br><br>[Figure 2040]<br><br>[Figure 2041]<br><br>[Figure 2042]<br><br>[Figure 2043]<br><br>[Figure 2044]<br><br>[Figure 2045]|
|---|

A person with red clothes is preparing dessert in the kitchen

static

[Figure 2046]

[Figure 2047]

[Figure 2048]

[Figure 2049]

[Figure 2050]

[Figure 2051]

[Figure 2052]

[Figure 2053]

[Figure 2054]

[Figure 2055]

[Figure 2056]

[Figure 2057]

[Figure 2058]

[Figure 2059]

[Figure 2060]

[Figure 2061]

[Figure 2062]

[Figure 2063]

[Figure 2064]

[Figure 2065]

[Figure 2066]

[Figure 2067]

[Figure 2068]

[Figure 2069]

[Figure 2070]

|[Figure 2071]<br><br>[Figure 2072]<br><br>[Figure 2073]<br><br>[Figure 2074]<br><br>[Figure 2075]<br><br>[Figure 2076]<br><br>[Figure 2077]<br><br>[Figure 2078]<br><br>[Figure 2079]<br><br>[Figure 2080]<br><br>[Figure 2081]<br><br>[Figure 2082]<br><br>[Figure 2083]<br><br>[Figure 2084]|
|---|

[Figure 2085]

[Figure 2086]

[Figure 2087]

[Figure 2088]

[Figure 2089]

[Figure 2090]

[Figure 2091]

[Figure 2092]

[Figure 2093]

[Figure 2094]

[Figure 2095]

[Figure 2096]

[Figure 2097]

[Figure 2098]

[Figure 2099]

[Figure 2100]

[Figure 2101]

medium right

[Figure 2102]

[Figure 2103]

[Figure 2104]

Background: This cozy home kitchen photo perfectly encapsulates the warmth and coziness of a nurturing space

- Fig. 5: Three examples of generated multi-scene videos by VideoStudio on MSR-VTT with single foreground reference image.

Entity Reference Image

Camera Movement

Input Prompt

[Figure 2105]

[Figure 2106]

[Figure 2107]

[Figure 2108]

[Figure 2109]

[Figure 2110]

[Figure 2111]

[Figure 2112]

[Figure 2113]

[Figure 2114]

[Figure 2115]

[Figure 2116]

[Figure 2117]

[Figure 2118]

[Figure 2119]

slow left

[Figure 2120]

[Figure 2121]

[Figure 2122]

[Figure 2123]

[Figure 2124]

[Figure 2125]

[Figure 2126]

[Figure 2127]

[Figure 2128]

[Figure 2129]

[Figure 2130]

[Figure 2131]

[Figure 2132]

[Figure 2133]

Foreground: The photo

depicts a man who appears to possess graphic quality

[Figure 2134]

[Figure 2135]

[Figure 2136]

[Figure 2137]

[Figure 2138]

[Figure 2139]

[Figure 2140]

[Figure 2141]

[Figure 2142]

[Figure 2143]

[Figure 2144]

[Figure 2145]

[Figure 2146]

[Figure 2147]

[Figure 2148]

[Figure 2149]

[Figure 2150]

medium left

[Figure 2151]

[Figure 2152]

[Figure 2153]

[Figure 2154]

[Figure 2155]

[Figure 2156]

[Figure 2157]

[Figure 2158]

[Figure 2159]

[Figure 2160]

[Figure 2161]

[Figure 2162]

Foreground: The photo, taken by a professional photographer, depicts a dog walking on a leash by a river of water

A man is playing with a dog in the park after getting off work

[Figure 2163]

[Figure 2164]

[Figure 2165]

[Figure 2166]

[Figure 2167]

[Figure 2168]

[Figure 2169]

[Figure 2170]

[Figure 2171]

[Figure 2172]

[Figure 2173]

[Figure 2174]

[Figure 2175]

[Figure 2176]

[Figure 2177]

slow left

[Figure 2178]

[Figure 2179]

[Figure 2180]

[Figure 2181]

[Figure 2182]

[Figure 2183]

[Figure 2184]

[Figure 2185]

[Figure 2186]

[Figure 2187]

[Figure 2188]

[Figure 2189]

[Figure 2190]

[Figure 2191]

Background: The photo depicts a neutral-toned room with a desk and a chair, a sense of professionalism

[Figure 2192]

[Figure 2193]

[Figure 2194]

[Figure 2195]

[Figure 2196]

[Figure 2197]

[Figure 2198]

[Figure 2199]

[Figure 2200]

[Figure 2201]

[Figure 2202]

[Figure 2203]

[Figure 2204]

[Figure 2205]

[Figure 2206]

slow forward

[Figure 2207]

[Figure 2208]

[Figure 2209]

[Figure 2210]

[Figure 2211]

[Figure 2212]

[Figure 2213]

[Figure 2214]

[Figure 2215]

[Figure 2216]

[Figure 2217]

[Figure 2218]

[Figure 2219]

[Figure 2220]

Background: The photo depicts a serene and picturesque public park, nestled in the heart of the city

Output Video

|[Figure 2221]<br><br>[Figure 2222]<br><br>[Figure 2223]<br><br>[Figure 2224]<br><br>[Figure 2225]<br><br>[Figure 2226]<br><br>[Figure 2227]<br><br>[Figure 2228]<br><br>[Figure 2229]<br><br>[Figure 2230]<br><br>[Figure 2231]<br><br>[Figure 2232]<br><br>[Figure 2233]<br><br>[Figure 2234]<br><br>[Figure 2235]<br><br>[Figure 2236]<br><br>[Figure 2237]<br><br>[Figure 2238]<br><br>[Figure 2239]<br><br>[Figure 2240]<br><br>[Figure 2241]<br><br>[Figure 2242]<br><br>[Figure 2243]<br><br>[Figure 2244]|
|---|

- Scene-1: The man is writing on the paper in the office room
- Scene-2: The man is making a phone call in the office room

|[Figure 2245]<br><br>[Figure 2246]<br><br>[Figure 2247]<br><br>[Figure 2248]<br><br>[Figure 2249]<br><br>[Figure 2250]<br><br>[Figure 2251]<br><br>[Figure 2252]<br><br>[Figure 2253]<br><br>[Figure 2254]<br><br>[Figure 2255]<br><br>[Figure 2256]<br><br>[Figure 2257]<br><br>[Figure 2258]<br><br>[Figure 2259]<br><br>[Figure 2260]<br><br>[Figure 2261]<br><br>[Figure 2262]<br><br>[Figure 2263]<br><br>[Figure 2264]<br><br>[Figure 2265]<br><br>[Figure 2266]<br><br>[Figure 2267]<br><br>[Figure 2268]|
|---|

|[Figure 2269]<br><br>[Figure 2270]<br><br>[Figure 2271]<br><br>[Figure 2272]<br><br>[Figure 2273]<br><br>[Figure 2274]<br><br>[Figure 2275]<br><br>[Figure 2276]<br><br>[Figure 2277]<br><br>[Figure 2278]<br><br>[Figure 2279]<br><br>[Figure 2280]<br><br>[Figure 2281]<br><br>[Figure 2282]<br><br>[Figure 2283]<br><br>[Figure 2284]<br><br>[Figure 2285]<br><br>[Figure 2286]<br><br>[Figure 2287]<br><br>[Figure 2288]<br><br>[Figure 2289]<br><br>[Figure 2290]<br><br>[Figure 2291]<br><br>[Figure 2292]|
|---|

Scene-3: The man is playing with the dog in the park

|[Figure 2293]<br><br>[Figure 2294]<br><br>[Figure 2295]<br><br>[Figure 2296]<br><br>[Figure 2297]<br><br>[Figure 2298]<br><br>[Figure 2299]<br><br>[Figure 2300]<br><br>[Figure 2301]<br><br>[Figure 2302]<br><br>[Figure 2303]<br><br>[Figure 2304]<br><br>[Figure 2305]<br><br>[Figure 2306]<br><br>[Figure 2307]<br><br>[Figure 2308]<br><br>[Figure 2309]<br><br>[Figure 2310]<br><br>[Figure 2311]<br><br>[Figure 2312]<br><br>[Figure 2313]<br><br>[Figure 2314]<br><br>[Figure 2315]<br><br>[Figure 2316]|
|---|

Scene-4: The dog is sitting down under a tree in the park

###### Fig. 6: One example of generated multi-scene videos by VideoStudio on MSR-VTT with multiple foreground reference images.

Foreground Reference Image

Background Reference Image

Output Video

|[Figure 2317]<br><br>[Figure 2318]<br><br>[Figure 2319]<br><br>[Figure 2320]<br><br>[Figure 2321]<br><br>[Figure 2322]<br><br>[Figure 2323]<br><br>[Figure 2324]<br><br>[Figure 2325]<br><br>[Figure 2326]<br><br>[Figure 2327]<br><br>[Figure 2328]<br><br>[Figure 2329]<br><br>[Figure 2330]|
|---|

[Figure 2331]

[Figure 2332]

[Figure 2333]

[Figure 2334]

[Figure 2335]

[Figure 2336]

[Figure 2337]

[Figure 2338]

[Figure 2339]

[Figure 2340]

[Figure 2341]

[Figure 2342]

[Figure 2343]

[Figure 2344]

[Figure 2345]

[Figure 2346]

[Figure 2347]

[Figure 2348]

[Figure 2349]

[Figure 2350]

[Figure 2351]

[Figure 2352]

[Figure 2353]

[Figure 2354]

[Figure 2355]

[Figure 2356]

[Figure 2357]

[Figure 2358]

[Figure 2359]

[Figure 2360]

[Figure 2361]

[Figure 2362]

[Figure 2363]

[Figure 2364]

[Figure 2365]

[Figure 2366]

[Figure 2367]

[Figure 2368]

[Figure 2369]

[Figure 2370]

Scene-1: The cat lies in the room

|[Figure 2371]<br><br>[Figure 2372]<br><br>[Figure 2373]<br><br>[Figure 2374]<br><br>[Figure 2375]<br><br>[Figure 2376]<br><br>[Figure 2377]<br><br>[Figure 2378]<br><br>[Figure 2379]<br><br>[Figure 2380]<br><br>[Figure 2381]<br><br>[Figure 2382]<br><br>[Figure 2383]|
|---|

[Figure 2384]

[Figure 2385]

[Figure 2386]

[Figure 2387]

[Figure 2388]

[Figure 2389]

[Figure 2390]

[Figure 2391]

[Figure 2392]

[Figure 2393]

[Figure 2394]

[Figure 2395]

[Figure 2396]

[Figure 2397]

[Figure 2398]

[Figure 2399]

[Figure 2400]

[Figure 2401]

[Figure 2402]

[Figure 2403]

[Figure 2404]

[Figure 2405]

[Figure 2406]

[Figure 2407]

[Figure 2408]

[Figure 2409]

[Figure 2410]

[Figure 2411]

[Figure 2412]

[Figure 2413]

[Figure 2414]

[Figure 2415]

[Figure 2416]

[Figure 2417]

[Figure 2418]

[Figure 2419]

[Figure 2420]

[Figure 2421]

[Figure 2422]

[Figure 2423]

- Scene-2: The cat lies in the driving car
- Scene-3: The cat plays in the flowers

|[Figure 2424]<br><br>[Figure 2425]<br><br>[Figure 2426]<br><br>[Figure 2427]<br><br>[Figure 2428]<br><br>[Figure 2429]<br><br>[Figure 2430]<br><br>[Figure 2431]<br><br>[Figure 2432]<br><br>[Figure 2433]<br><br>[Figure 2434]<br><br>[Figure 2435]<br><br>[Figure 2436]<br><br>[Figure 2437]|
|---|

[Figure 2438]

[Figure 2439]

[Figure 2440]

[Figure 2441]

[Figure 2442]

[Figure 2443]

[Figure 2444]

[Figure 2445]

[Figure 2446]

[Figure 2447]

[Figure 2448]

[Figure 2449]

[Figure 2450]

[Figure 2451]

[Figure 2452]

[Figure 2453]

[Figure 2454]

[Figure 2455]

[Figure 2456]

[Figure 2457]

[Figure 2458]

[Figure 2459]

[Figure 2460]

[Figure 2461]

[Figure 2462]

[Figure 2463]

[Figure 2464]

[Figure 2465]

[Figure 2466]

[Figure 2467]

[Figure 2468]

[Figure 2469]

[Figure 2470]

[Figure 2471]

[Figure 2472]

[Figure 2473]

[Figure 2474]

[Figure 2475]

[Figure 2476]

[Figure 2477]

|[Figure 2478]<br><br>[Figure 2479]<br><br>[Figure 2480]<br><br>[Figure 2481]<br><br>[Figure 2482]<br><br>[Figure 2483]<br><br>[Figure 2484]<br><br>[Figure 2485]<br><br>[Figure 2486]<br><br>[Figure 2487]<br><br>[Figure 2488]<br><br>[Figure 2489]<br><br>[Figure 2490]<br><br>[Figure 2491]|
|---|

[Figure 2492]

[Figure 2493]

[Figure 2494]

[Figure 2495]

[Figure 2496]

[Figure 2497]

[Figure 2498]

[Figure 2499]

[Figure 2500]

[Figure 2501]

[Figure 2502]

[Figure 2503]

[Figure 2504]

[Figure 2505]

[Figure 2506]

[Figure 2507]

- Scene-1: The parrot stands in the bedroom
- Scene-2: The parrot stands in the forest

|[Figure 2508]<br><br>[Figure 2509]<br><br>[Figure 2510]<br><br>[Figure 2511]<br><br>[Figure 2512]<br><br>[Figure 2513]<br><br>[Figure 2514]<br><br>[Figure 2515]<br><br>[Figure 2516]<br><br>[Figure 2517]<br><br>[Figure 2518]<br><br>[Figure 2519]<br><br>[Figure 2520]<br><br>[Figure 2521]|
|---|

[Figure 2522]

[Figure 2523]

[Figure 2524]

[Figure 2525]

[Figure 2526]

[Figure 2527]

[Figure 2528]

[Figure 2529]

[Figure 2530]

[Figure 2531]

[Figure 2532]

[Figure 2533]

[Figure 2534]

[Figure 2535]

[Figure 2536]

[Figure 2537]

[Figure 2538]

[Figure 2539]

[Figure 2540]

[Figure 2541]

[Figure 2542]

[Figure 2543]

[Figure 2544]

[Figure 2545]

[Figure 2546]

[Figure 2547]

[Figure 2548]

[Figure 2549]

[Figure 2550]

[Figure 2551]

[Figure 2552]

[Figure 2553]

[Figure 2554]

[Figure 2555]

[Figure 2556]

[Figure 2557]

[Figure 2558]

[Figure 2559]

[Figure 2560]

[Figure 2561]

|[Figure 2562]<br><br>[Figure 2563]<br><br>[Figure 2564]<br><br>[Figure 2565]<br><br>[Figure 2566]<br><br>[Figure 2567]<br><br>[Figure 2568]<br><br>[Figure 2569]<br><br>[Figure 2570]<br><br>[Figure 2571]<br><br>[Figure 2572]<br><br>[Figure 2573]<br><br>[Figure 2574]<br><br>[Figure 2575]|
|---|

[Figure 2576]

[Figure 2577]

[Figure 2578]

[Figure 2579]

[Figure 2580]

[Figure 2581]

[Figure 2582]

[Figure 2583]

[Figure 2584]

[Figure 2585]

[Figure 2586]

Scene-3: The parrot stands in front of the river

|[Figure 2587]<br><br>[Figure 2588]<br><br>[Figure 2589]<br><br>[Figure 2590]<br><br>[Figure 2591]<br><br>[Figure 2592]<br><br>[Figure 2593]<br><br>[Figure 2594]<br><br>[Figure 2595]<br><br>[Figure 2596]<br><br>[Figure 2597]<br><br>[Figure 2598]<br><br>[Figure 2599]<br><br>[Figure 2600]|
|---|

[Figure 2601]

[Figure 2602]

[Figure 2603]

[Figure 2604]

[Figure 2605]

[Figure 2606]

[Figure 2607]

[Figure 2608]

[Figure 2609]

[Figure 2610]

[Figure 2611]

[Figure 2612]

[Figure 2613]

[Figure 2614]

[Figure 2615]

[Figure 2616]

[Figure 2617]

[Figure 2618]

[Figure 2619]

[Figure 2620]

[Figure 2621]

[Figure 2622]

[Figure 2623]

[Figure 2624]

[Figure 2625]

[Figure 2626]

[Figure 2627]

[Figure 2628]

[Figure 2629]

[Figure 2630]

[Figure 2631]

[Figure 2632]

[Figure 2633]

[Figure 2634]

[Figure 2635]

[Figure 2636]

[Figure 2637]

[Figure 2638]

[Figure 2639]

[Figure 2640]

Scene-1: The motorcyclist stays in the town

|[Figure 2641]<br><br>[Figure 2642]<br><br>[Figure 2643]<br><br>[Figure 2644]<br><br>[Figure 2645]<br><br>[Figure 2646]<br><br>[Figure 2647]<br><br>[Figure 2648]<br><br>[Figure 2649]<br><br>[Figure 2650]|
|---|

[Figure 2651]

[Figure 2652]

[Figure 2653]

[Figure 2654]

[Figure 2655]

[Figure 2656]

[Figure 2657]

[Figure 2658]

[Figure 2659]

[Figure 2660]

[Figure 2661]

[Figure 2662]

[Figure 2663]

[Figure 2664]

[Figure 2665]

[Figure 2666]

[Figure 2667]

[Figure 2668]

[Figure 2669]

[Figure 2670]

[Figure 2671]

[Figure 2672]

[Figure 2673]

[Figure 2674]

[Figure 2675]

[Figure 2676]

[Figure 2677]

[Figure 2678]

[Figure 2679]

[Figure 2680]

[Figure 2681]

[Figure 2682]

[Figure 2683]

[Figure 2684]

[Figure 2685]

[Figure 2686]

[Figure 2687]

[Figure 2688]

[Figure 2689]

[Figure 2690]

Scene-2: The motorcyclist is riding on the road under the sunset

|[Figure 2691]<br><br>[Figure 2692]<br><br>[Figure 2693]<br><br>[Figure 2694]<br><br>[Figure 2695]<br><br>[Figure 2696]<br><br>[Figure 2697]<br><br>[Figure 2698]<br><br>[Figure 2699]<br><br>[Figure 2700]<br><br>[Figure 2701]<br><br>[Figure 2702]<br><br>[Figure 2703]<br><br>[Figure 2704]|
|---|

[Figure 2705]

[Figure 2706]

[Figure 2707]

[Figure 2708]

[Figure 2709]

[Figure 2710]

[Figure 2711]

[Figure 2712]

[Figure 2713]

[Figure 2714]

[Figure 2715]

[Figure 2716]

[Figure 2717]

[Figure 2718]

[Figure 2719]

[Figure 2720]

[Figure 2721]

[Figure 2722]

[Figure 2723]

[Figure 2724]

[Figure 2725]

[Figure 2726]

[Figure 2727]

[Figure 2728]

[Figure 2729]

[Figure 2730]

[Figure 2731]

[Figure 2732]

[Figure 2733]

[Figure 2734]

[Figure 2735]

[Figure 2736]

[Figure 2737]

[Figure 2738]

[Figure 2739]

[Figure 2740]

[Figure 2741]

[Figure 2742]

[Figure 2743]

[Figure 2744]

Scene-3: The motorcyclist is ridding on the moon

- Fig. 7: Three example of generated multi-scene videos by our VideoStudio using the real images as entity reference images.

