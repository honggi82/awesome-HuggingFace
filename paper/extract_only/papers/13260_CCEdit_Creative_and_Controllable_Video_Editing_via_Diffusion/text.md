## CCEdit: Creative and Controllable Video Editing via Diffusion Models

Ruoyu Feng1,2*, Wenming Weng1,2, Yanhui Wang1,2, Yuhui Yuan2, Jianmin Bao2, Chong Luo2†, Zhibo Chen1†, Baining Guo2 1University of Science and Technology of China 2Microsoft Research Asia https://ruoyufeng.github.io/CCEdit.github.io/

# arXiv:2309.16496v3[cs.CV]7Apr2024

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]<br><br>[Figure 7]|
|---|

|[Figure 8]<br><br>[Figure 9]|
|---|

|[Figure 10]<br><br>[Figure 11]|
|---|

|[Figure 12]<br><br>[Figure 13]|
|---|

|[Figure 14]<br><br>[Figure 15]|
|---|

[Figure 16]

Intention: Transform this video into the Van Gogh Starry Night style.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|[Figure 22]<br><br>[Figure 23]|
|---|

|[Figure 24]<br><br>[Figure 25]|
|---|

|[Figure 26]<br><br>[Figure 27]|
|---|

|[Figure 28]<br><br>[Figure 29]|
|---|

|[Figure 30]<br><br>[Figure 31]|
|---|

[Figure 32]

Intention: Transform this video into the mechanical style.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

|[Figure 38]<br><br>[Figure 39]|
|---|

|[Figure 40]<br><br>[Figure 41]|
|---|

|[Figure 42]<br><br>[Figure 43]|
|---|

|[Figure 44]<br><br>[Figure 45]|
|---|

|[Figure 46]<br><br>[Figure 47]|
|---|

[Figure 48]

Intention: Transform this video into the cyberpunk style.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

|[Figure 54]<br><br>[Figure 55]|
|---|

|[Figure 56]<br><br>[Figure 57]|
|---|

|[Figure 58]<br><br>[Figure 59]|
|---|

|[Figure 60]<br><br>[Figure 61]|
|---|

|[Figure 62]<br><br>[Figure 63]|
|---|

[Figure 64]

Intention: Make the foreground puppy chubby and adorable.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

|[Figure 70]<br><br>[Figure 71]|
|---|

|[Figure 72]<br><br>[Figure 73]|
|---|

|[Figure 74]<br><br>[Figure 75]|
|---|

|[Figure 76]<br><br>[Figure 77]|
|---|

|[Figure 78]<br><br>[Figure 79]|
|---|

[Figure 80]

Intention: Replace the background with snowy winter scene.

Input Video Edited Video

Figure 1. Built upon diffusion models, CCEdit provides users with a powerful and flexible set of video editing capabilities, including style transfer (row 1-3), foreground modifications (row 4), and background replacement (row 5).

### Abstract

In this paper, we present CCEdit, a versatile generative video editing framework based on diffusion models. Our approach employs a novel trident network structure that separates structure and appearance control, ensuring precise and creative editing capabilities. Utilizing the foundational ControlNet architecture, we maintain the structural

∗ This work is done when Ruoyu Feng is an intern with MSRA. † Corresponding author.

integrity of the video during editing. The incorporation of an additional appearance branch enables users to exert fine-grained control over the edited key frame. These two side branches seamlessly integrate into the main branch, which is constructed upon existing text-to-image (T2I) generation models, through learnable temporal layers. The versatility of our framework is demonstrated through a diverse range of choices in both structure representations and personalized T2I models, as well as the option to provide the edited key frame. To facilitate comprehensive evaluation, we introduce the BalanceCC benchmark dataset, com-

prising 100 videos and 4 target prompts for each video. Our extensive user studies compare CCEdit with eight state-ofthe-art video editing methods. The outcomes demonstrate CCEdit’s substantial superiority over all other methods.

### 1. Introduction

In recent years, the domain of visual content creation and editing has undergone a profound transformation, driven by the emergence of diffusion-based generative models [11, 19, 50]. A large body of prior research has demonstrated the exceptional capabilities of diffusion models in generating diverse and high-quality images [40, 42, 45] and videos [5, 20, 48], conditioned by text prompts. These advancements have naturally paved the way for innovations in generative video editing [7, 25, 35, 37, 54, 57, 58, 62].

Generative video editing, despite its rapid advancement, continues to face a series of significant challenges. These challenges include accommodating diverse editing requests, achieving fine-grained control over the editing process, and harnessing the creative potential of generative models. Diverse editing requirements include tasks such as stylistic alterations, foreground replacements, and background modifications. Generative models, while powerful and creative, may not always align perfectly with the editor’s intentions or artistic vision, resulting in a lack of precise control. In response to these challenges, this paper introduces CCEdit, a versatile generative video editing framework meticulously designed to strike a harmonious balance between controllability and creativity while accommodating a wide range of editing requirements.

CCEdit achieves its goal by effectively decoupling structure and appearance control in a unified trident network. This network comprises three essential components: the main text-to-video generation branch and two accompanying side branches dedicated to structure and appearance manipulation. The main branch leverages a pre-trained textto-image (T2I) diffusion model [42], which is transformed into a text-to-video (T2V) model through the insertion of temporal modules. The structure branch, implemented as ControlNet [59], is responsible for digesting the structural information extracted from each frame of the input video and seamlessly infusing it into the main branch. Simultaneously, the appearance branch introduces an innovative mechanism for precise appearance control, when an edited reference frame is available. The structure and appearance branches are effectively integrated into the central branch through learnable temporal layers. These layers serve not only as a cohesive link, aggregating information from side

1 CCEdit is currently a research project, and there are no immediate intentions to integrate it into a product or extend public accessibility. Any future research endeavor will adhere to Microsoft’s AI principles.

branches, but also as a crucial element ensuring temporal consistency across the generated video frames.

In highlighting the versatility of our framework, we provide a wide range of control choices for both structure and appearance manipulation. For structure control, users can choose from various types of structural information, including line drawings [8], PiDi boundaries [51], and depth maps [41], all of which can serve as input to the structure branch. On the appearance control front, the main branch already provides an inherent mechanism, allowing control through text prompts. Additionally, personalized T2I models from the Stable Diffusion community, such as DreamBooth and LoRA [21, 44], can be integrated as plugins into CCEdit, offering greater flexibility and creativity. More importantly, the appearance branch can accommodate the referenced key frame, facilitating fine-grained appearance control. Notably, all these control options are seamlessly integrated within the same framework, yielding editing outcomes that demonstrate both temporal coherence and precision. This not only underscores the versatility of our solution but also ensures ease of adoption, making it a compelling choice for AI-assisted video editing.

To address the challenges inherent in evaluating generative video editing methods, we introduce the BalanceCC benchmark dataset. Comprising 100 diverse videos and 4 target prompts for each video, this dataset includes detailed scene descriptions and attributes related to video category, scene complexity, motion, among others. These descriptions are generated with the assistance of the cutting-edge GPT-4V(ision) model [1, 32–34] and then refined by human annotators. Through extensive experimental evaluations on this dataset, we not only confirm the outstanding functionality and editing capabilities of CCEdit, but also underscore the comprehensiveness of the benchmark dataset. We firmly believe that BalanceCC stands as a robust and all-encompassing evaluation platform for the dynamic field of generative video editing.

### 2. Related Work

#### 2.1. Diffusion-based Image and Video Generation

Diffusion models (DM) [11, 19, 50] have demonstrated exceptional capabilities in the field of image synthesis. These models indeed help by learning to approximate a data distribution through the iterative denoising of a diffused input. What makes DMs truly practical is the incorporation of text prompt as condition to control the output image during the generative process [31, 39, 42, 45]. Apart from the proliferation of advanced techniques in the field of image synthesis, DMs have also excelled in video generation [5, 20, 31, 48]. This is achieved by integrating modulated spatial-temporal modules, enabling the synthesis of highquality videos while maintaining temporal consistency.

[Figure 81]

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

Appearance Control:

Key Frame Editing Structure Extraction

- • Text prompt, e.g. “A bear is walking, anime style.”
- • Personalized models
- • Edited key frame:

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Structure Information

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

[Figure 118]

[Figure 119]

###### …

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

###### Text prompt

LoRA: Moxin

Base Model: ToonYou LoRA: Pixel Art

|~𝒩(𝟎,𝑰)|
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

###### …

Style transfer

Foreground replacement

Background modification

###### Structure Control:

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Learnable appearance encoder Learnable temporal layers

[Figure 158]

…

Frozen T2I model

Line Drawing Depth

Pidi Boundary

Frozen structure ControlNet

[Figure 159]

[Figure 160]

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

P: A tiger is walking. <S: PiDi Boundary> <B: SD v1.5> <w/ Reference>

P: A bear is walking, in winter. <S: Depth> <B: SD v1.5> <w/ reference>

A bear is walking, anime style. <S: Line drawings> <B: Toonyou>

P: A bear is walking, pixel art style. <S: Depth> <B: Counterfeit> <L: Pixel Art> <w/ Reference>

P: A bear is walking, Chinese Ink style. <S: PiDi Boundary> <B: Counterfeit> <L: Moxin>

Figure 2. Illustration of our overall framework. Structure and appearance information in the target video are modulated independently, and seamlessly integrated into the main branch. Structure control is conducted via the pre-trained ControlNet [59]. Appearance control is achieved precisely by the edited key frame. Details regarding the autoencoder and iterative denoising process are omitted for simplicity. “P”, “S”, “B”, “L” indicate prompt, structure, base model, and LoRA, respectively.

#### 2.2. Video Editing with Diffusion Models

Recent studies leverage the inherent generative priors of DMs for image editing [3, 10, 16, 27, 36, 52]. The same idea is also applied in the field of video editing. Unlike image editing, video editing involves not only the manipulation of appearance-based attributes but also requires the meticulous preservation of temporal coherence throughout frames. A lapse in maintaining this temporal coherence can result in visual artifacts, such as flickering and degradation.

Some generative video editing methods [6, 14, 22, 37, 53, 58, 60] strive to achieve training-free temporal consistency. They accomplish this by transitioning from spatial self-attention mechanisms within T2I diffusion models to temporal-aware cross-frame attention techniques. Some other methods [26, 47, 55, 62] perform per-video finetuning. They focus on optimizing the parameters of pretrained T2I models according to the input video, aiming to achieve temporal coherence within the target video. However, this optimization for each input video can be timeconsuming and inadequate tuning of the temporal modules might lead to suboptimal temporal coherence. Recent studies [15, 24, 57] have introduced trainable temporal layers to construct T2V generative models. These models are trained

on extensive text-video paired datasets, and they are used in both video generation and editing tasks [12, 29].

Unlike previous work, this study does not seek a simple fix to existing T2I models for video editing, nor does it attempt to train a full-fledged T2V model. Instead, we introduce a unique network architecture tailored for video editing. Our approach involves dataset-level fine-tuning, circumvents the expenses associated with per-video tuning during inference time, and prioritizing the effective training of temporal layers to achieve robust model performance.

### 3. Approach

#### 3.1. Preliminary

Diffusion models [19] are probabilistic generative models that approximate a data distribution p(x) by gradually denoising a normally distributed variable. Specifically, DMs aim to learn the reverse dynamics of a predetermined Markov chain with a fixed length of T. The forward Markov chain can be conceptualized as a procedure of injecting noise into a pristine image. Empirically, DMs can be interpreted as an equally weighted sequence of denoising autoencoders ϵθ(xt,t) where t = 1,...,T. These autoencoders are trained to predict a denoised variant of the noisy

input xt. The corresponding objective can be simplified to

0,t,ϵ∼N(0,I)[∥ϵ − ϵθ(xt,t)∥22]. (1) Latent diffusion models (LDMs) are trained in the learned latent representation space. The bridge between this latent space and the original pixel-level domain is established via a perceptual compression model. The perceptual compression model is composed of an encoder E and a decoder D, where z = E(x) and x ≈ D(E(x)). Then the optimization objective in Eq. (1) is modified as

Ex

0,t,ϵ∼N(0,I)[∥ϵ − ϵθ(zt,t)∥22]. (2)

Ez

#### 3.2. The CCEdit Framework

The primary objective of our work is to empower creative control in video editing. Although creativity naturally emerges in generative models, achieving controllability is a more complex endeavor. To address this challenge, CCEdit strategically decouples the management of structure and appearance within a unified trident network. In Fig. 2, we provide an illustrative overview of the framework’s architecture, which comprises three vital components.

The main branch. The main branch of our model fundamentally operates as a text-to-video generation network. It is built upon the well-established text-to-image model, Stable Diffusion [42]. We transform this model into a text-tovideo variant by incorporating temporal layers into spatial layers of both the encoder and decoder. This entails the addition of a one-dimensional temporal layer with the same type as its previous spatial layer, i.e., convolution blocks and attention blocks. Besides, we also use the skip connection and zero-initialized projection out layer of each newly added temporal layer for stable and progressive updating, which has been proven to be effective [15, 48, 59]. The zero-initialized projection out layer is instantiated as a linear layer. Formally, let F(·;Θs) be the 2D spatial block, F(·;Θt) be the 1D temporal block, and Z(·;Θz) be the zero-initialized projection out layer, where Θs, Θt, and Θz represent corresponding network parameters. The complete process of one pseudo-3D block that maps the input feature u to the output feature v is written as

v = F(u;Θs) + Z(F(F(u;Θs);Θt);Θz), (3)

where u and v are both 3D feature maps, i.e., u ∈ Rl×h×w×c with {l,h,w,c} as the number of frames, height, width, and the number of channels, respectively.

Moreover, we draw inspiration from AnimateDiff [15] and VideoLDM [5], which advocates the shared utilization of temporal layers among personalized T2I models such as DreamBooth [44] and LoRA [21]. The key aspect of it is training the temporal layers while keeping the spatial weights frozen. We follow this schedule to inherit the T2I model’s compatibility and visual generation capability.

The structure branch. The introduction of the structure branch is motivated by the common need in video editing tasks to preserve frame structure for non-edited or styletransferred segments. Striking a delicate balance between maintaining faithful frame structure and allowing the generative model ample creative freedom poses a significant challenge. The structure branch is implemented with the pretrained ControlNet [59]. To accommodate varying levels of structure control, we use various types of structure representation, including line drawings [8], PiDi boundaries [51], and depth maps [41], ensuring adaptability to control structure at different degrees.

Specifically, the structure representation from all frames is extracted individually and injected into the main branch. Each frame undergoes preprocessing to derive a structure representation, and the weights of the ControlNet are held in a frozen state during training, emphasizing the preservation of learned structural features. Formally, let F(·;Φc) denote the ControlNet that maps structure information into features, and Z(·;Φz1) and Z(·;Φz2) denote the two instances of zero convolutions in [59]. Then the process of adding structure control to the 3D-aware feature v is

vs = v + Z(F(zt + Z(cs;Φz1);Φc);Φz2), (4)

where zt denotes the noisy input in latent space, cs denotes the structure condition of the video sequence, and vs denotes the feature aware of structure information.

The appearance branch. In addition to using text prompts and incorporating personalized models for appearance control, we introduce a novel design—the appearance branch. This architectural innovation introduces a pioneering approach for fine-grained appearance control, allowing for the integration of an edited frame as a detailed reference in the context of video editing. Since the editing of key frame can be accomplished through precise user edits or by using advanced off-the-shelf image editing algorithms, the introduction of appearance branch provides our framework with greater creativity and controllability. Specifically, a key frame is initially assigned to the latent variable by the encoder E. Subsequently, a neural network with similar architecture to the main branch’s encoder extracts multi-scale features. The extracted features are incorporated into the main branch. Through this design, the appearance information from the edited key frame propagates to all frames via the temporal modules, effectively achieving the desired creative control in the output video. Formally, suppose F(·;Ψ) is the encoder that maps the pixel-wise appearance of the key frame into features, Z(·;Ψz) denotes the zero convolution projection out layer, vj indicates the feature of the j-th frame, and cja is the key frame. Then the process of adding appearance control to the features is as follows

###### vaj = vj + Z(F(E(cja);Ψ);Ψz), (5)

where vaj is the j-th feature, aware of the edited appearance. Training. Before training, we initialize the spatial weights of the main branch with pre-trained T2I models. Temporal weights are randomly initialized while the projection out layers are zero-initialized. We instantiate the model in the structure branch by pre-trained ControlNets [59]. As for the appearance branch, we copy the encoder of pre-trained T2I model and remove text cross-attention layers. During training, given the latent variables z0 = E(x0) of an input video clip x0. Diffusion algorithms progressively add noise to it and produce the noisy input zt. Given conditions of time step t, text prompt ct, structure information cs, and appearance information cja of the key frame, the overall optimization objective is

0,t,ct,cs,cja,ϵ∼N(0,I)[∥ϵ − ϵθ(zt,t,ct,cs,cja)∥22], (6) where ϵθ indicates the whole network to predict the noise added to the noisy input zt. We freeze the spatial weights in the main branch and the weights in the structure branch. Concurrently, we update the parameters of the newly incorporated temporal layers in the main branch, as well as the weights in the appearance branch. By default, the appearance branch takes the center frame of the video clip as input. Inference with anchor prior. We find that, in some challenging cases, the edited video may exhibit large areas of flickering. This is often caused by inconsistent structural representations extracted by image-level pre-processing modules. Therefore, we propose a simple yet efficient strategy to improve the stability and quality of the result by modifying the start noise. Specifically, consider the individual noise sequence [ϵ1ind,...,ϵlind] and the edited center frame cja, where l and j indicate the frame numbers and the index of the edited key frame, respectively. The start noise ϵi for each frame is modified as

Ez

ϵi = ϵiind + αE(cja), (7) where α is the hyperparameter that controls the strength of prior, and E(cja) is the latent of the edited key frame. We call this strategy anchor prior, which is tailored for our pipeline of editing videos with an reference key frame. We empirically found that α = 0.03 works well in most cases. The intuition behind it lies in that the video frames are usually similar to each other. The operation of adding noise to diffusion models tends to rapidly destroy high-frequency information while slowly degrading low-frequency information. Therefore, the anchor prior can be seen as providing a bit of low-frequency information to all frames while ensuring that the distribution remains almost unchanged (achieved by small α), thus becoming better starting points.

#### 3.3. Editing for Long Videos

Video editing tools face a challenge in maintaining a consistent look and feel across clips that span tens of seconds,

equivalent to hundreds of frames. The inherent limitation of generative models, processing only a dozen frames per inference due to memory constraints, introduces variability in results, even with a fixed random seed. CCEdit addresses this challenge with its fine-grained appearance control, enabling the editing of long videos into a cohesive look and feel through extension and interpolation modes.

In essence, let L + 1 represent the frames CCEdit processes in one run. For videos exceeding L + 1 frames, we select one key frame for every L frames. In the initial run, the first L+1 key frames undergo editing. Subsequent runs, in extension mode, treat the last edited frame from the previous run as the first frame. The edited result serves as a reference for the appearance branch. This process iterates until all key frames are processed. Transitioning to the interpolation mode, two adjacent frames become the first and last frames of an inference run to edit the L − 1 intermediate frames, and both edited frames serve as references for the appearance branch. This continues until all frames are edited. This meticulous process ensures consistent editing results throughout the entire video.

### 4. BalanceCC Benchmark

#### 4.1. Overview

While generative video editing has gained considerable attention as a growing research field, the absence of a standardized benchmark for assessing the efficacy of different approaches poses a potential hindrance to the technical progression of the field. Despite the recent introduction of TGVE 2023 [56] as an evaluation benchmark, it is crucial to note that the videos within this benchmark present challenges such as severe camera shake, overly complex scenes, blur, and low frame rates. In response to this, we introduce BalanceCC, a benchmark that contains 100 videos with varied attributes, designed to offer a comprehensive platform for evaluating video editing, focusing on both controllability and creativity.

#### 4.2. Benchmark Establishment

We curated a collection of 100 open-license videos suitable for legal, non-stigmatizing modifications. These videos range from 2 to 20 seconds in duration, each with a frame rate of about 30 fps. Besides, we utilize GPT-4V(ision) [1, 32–34] as an assistant to establish this benchmark. For each video, GPT-4V(ision) provides a description and assigns a complexity score to the scene using the center frame as a reference, with ratings from 1 (Simple) to 3 (Complex). Additionally, we manually annotate each video for camera movement, object movement, and categorical content, with motion rated on a scale from 1 (Stationary) to 3 (Quick), and categories that include humans, animals, objects, and landscapes. Following this, GPT-4V(ision) is tasked to craft tar-

[Figure 199]

Figure 3. Illustration of the statistics on BalanceCC.

get prompts for video editing, encompassing style, object, and background alterations, along with compound changes. This process, while akin to TGVE 2023 [56], we additionally introduce a “Fantasy Level” to indicate the imaginative and creative degree of the target prompt. These measures are intended to assist researchers in appraising the applicability of various methods to source videos and in gauging their potential. See supplementary for details on the prompting pipeline, specific instructions, principles of labeling, and illustrative examples.

- 4.3. Statistics

The overall distribution of BalanceCC is illustrated in Fig. 3. For the data of original videos, the distribution across categories tends towards uniformity, yet the “Human” category is slightly more prevalent than others. This was a deliberate choice, as editing human subjects is more practically significant and, due to the complexity of human and facial structures, editing in the “Human” category presents more challenges. Regarding “Scene Complexity” and “Object Motion”, videos with moderate and slow levels are slightly more common. In terms of “Camera Motion”, videos of lower levels predominate (Stationary: 54%, Slow: 38%). Finally, regarding the “Fantasy Level” distribution in target prompts, there is a relatively balanced allocation, with a marginal inclination towards videos categorized at a moderate level.

We hope that the aforementioned categorization of the benchmark will better assist researchers and users in understanding the strengths and weaknesses of a method, thus enabling targeted improvements and fostering rapid development in the field.

- 5. Experiments

- 5.1. Implementation Details

Stable Diffusion-v1.5 is used as the base T2I model in the main branch. We use the pre-trained ControlNet [59] for the structure information guidance. The training dataset com-

|[Figure 200]<br><br>[Figure 201]|
|---|

|[Figure 202]<br><br>[Figure 203]|
|---|

|[Figure 204]<br><br>[Figure 205]|
|---|

|[Figure 206]<br><br>[Figure 207]|
|---|

Input Video

“A mechanical tiger is walking.”

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

Line Drawing

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

PiDi Boundary

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

Scribble

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

Depth

###### Figure 4. Results under different structural guidance.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

|[Figure 260]<br><br>[Figure 261]|
|---|

|[Figure 262]<br><br>[Figure 263]|
|---|

|[Figure 264]<br><br>[Figure 265]|
|---|

|[Figure 266]<br><br>[Figure 267]|
|---|

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

<ReV Animated, Moxin> “A boat is sailing, Chinese traditional ink style.”

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

|[Figure 276]<br><br>[Figure 277]|
|---|

|[Figure 278]<br><br>[Figure 279]|
|---|

|[Figure 280]<br><br>[Figure 281]|
|---|

|[Figure 282]<br><br>[Figure 283]|
|---|

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

< MeinaMix, Pixel Art Style> “A bear is walking, pixel art style.”

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

|[Figure 292]<br><br>[Figure 293]|
|---|

|[Figure 294]<br><br>[Figure 295]|
|---|

|[Figure 296]<br><br>[Figure 297]|
|---|

|[Figure 298]<br><br>[Figure 299]|
|---|

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

<A-ZovyaRPGArtistTools>

“An astronaut rides a motorbike, planets in back.”

Figure 5. Results of video style translation. ⟨·⟩ indicate the personalized T2I model we used.

bines WebVid-10M [4] and a self-collected private dataset. We trained the temporal consistency modules and appearance ControlNet towards various types of structural information, including line drawings [8], PiDi boundaries [51], depth maps detected by Midas [41], and human scribbles. Depth maps are used by default. The control scales are set as 1. For the temporal interpolation model, we train it exclusively on depth maps, employing a smaller control scale of 0.5. This approach is adopted because its requirement for structural information is comparatively less than that of other models. During the training process, we first resize the shorter side to 384 pixels, followed by a random crop to obtain video clips with a size of 384 × 576. 17 frames at 4 fps are sampled from each video. The batch size is 32 and the learning rate is 3e − 5. We train each model for 100K iterations. During inference, we employ the DDIM [49] sampler with 30 steps, classifier-free guidance [18] of magnitude 9.

Input Video

Key Frame

Edited Video

[Figure 304]

[Figure 305]

[Figure 306]

|[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]|
|---|

|[Figure 311]<br><br>[Figure 312]|
|---|

|[Figure 313]<br><br>[Figure 314]|
|---|

|[Figure 315]<br><br>[Figure 316]|
|---|

[Figure 317]

[Figure 318]

[Figure 319]

|[Figure 320]<br><br>[Figure 321]|
|---|

<Hellomecha, Building Block World>

“A building block style car in the parking lot.”

[Figure 322]

[Figure 323]

[Figure 324]

|[Figure 325]<br><br>[Figure 326]|
|---|

|[Figure 327]<br><br>[Figure 328]|
|---|

|[Figure 329]<br><br>[Figure 330]|
|---|

|[Figure 331]<br><br>[Figure 332]| |
|---|---|
| | |

[Figure 333]

[Figure 334]

[Figure 335]

|[Figure 336]<br><br>[Figure 337]|
|---|

<Counterfeit>

“A man hikes on the moon, anime style.”

[Figure 338]

[Figure 339]

[Figure 340]

|[Figure 341]<br><br>[Figure 342]| |
|---|---|
| | |

|[Figure 343]<br><br>[Figure 344]|
|---|

|[Figure 345]<br><br>[Figure 346]|
|---|

|[Figure 347]<br><br>[Figure 348]|
|---|

[Figure 349]

[Figure 350]

[Figure 351]

|[Figure 352]<br><br>[Figure 353]|
|---|

<ReV Animated> “A paladin in armor rides on motorcycle, on fire.”

Figure 6. Video editing results with customized center frame as reference. The first row corresponds to customizing foreground, the second row corresponds to customizing background, and the third row is taking given reference image to affect the entire picture. ⟨·⟩ indicate the personalized T2I model we used.

#### 5.2. Applications

Controllable and creative style transfer. In CCEdit, the controllability and creativity of video style transfer are manifested in various dimensions. Two basic aspects include the diversity of structural information and the availability of off-the-shelf personalized models [9, 13]. The former enables users to customize the granularity and type of structural information retained from the original video, as depicted in Fig. 4. The latter allows users to edit the video into their desired domain, as shown in Fig. 5.

Video editing with precise appearance control. Sometimes, users require stronger control over the content they want to generate. For example, they may want to change only the foreground, alter just the background, or edit the texture content of a video in a specific way. Therefore, CCEdit focuses more on precise appearance control by initially modifying the key frame with image editing techniques and then using it as a reference for the entire video. As depicted in Fig. 6, we first edit the center frames of the videos by Stable Diffusion Web UI [2], followed by utilizing these edited center frames as guides for the video editing process. Thanks to end-to-end network training, our method coherently propagates edits from the key frame throughout the entire video.

Long video editing. A seamless and visually appealing video typically necessitates a higher frame count and increased frame rate, elements that have been inadequately addressed by many contemporary video editing methodologies. CCEdit effectively resolves this through its hierarchical design for key frames editing, combined with iterative extension and a tailored temporal interpolation mechanism. This approach enables the editing of videos comprising up

|[Figure 354]<br><br>[Figure 355]|
|---|

|[Figure 356]<br><br>[Figure 357]|
|---|

|[Figure 358]<br><br>[Figure 359]|
|---|

|[Figure 360]<br><br>[Figure 361]|
|---|

|[Figure 362]<br><br>[Figure 363]|
|---|

Original

Video

“City at night, in winter.”

|[Figure 364]<br><br>[Figure 365]|
|---|

|[Figure 366]<br><br>[Figure 367]|
|---|

|[Figure 368]<br><br>[Figure 369]|
|---|

|[Figure 370]<br><br>[Figure 371]|
|---|

|[Figure 372]<br><br>[Figure 373]|
|---|

Edited

Video

Frame: 1 Frame: 61 Frame: 121

Frame: 181 Frame: 241

- Figure 7. Illustration of long video editing. CCEdit achieves good consistency across over 240 frames. Zoom in for best view.

|[Figure 374]<br><br>[Figure 375]|
|---|

|[Figure 376]<br><br>[Figure 377]|
|---|

|[Figure 378]<br><br>[Figure 379]<br><br>|[Figure 380]|
|---|
|
|---|

|[Figure 381]<br><br>[Figure 382]<br><br>|[Figure 383]|
|---|
|
|---|

|[Figure 384]<br><br>[Figure 385]|
|---|

|[Figure 386]<br><br>[Figure 387]|
|---|

|[Figure 388]<br><br>[Figure 389]|
|---|

|[Figure 390]<br><br>[Figure 391]|
|---|

|[Figure 392]<br><br>[Figure 393]|
|---|

|[Figure 394]<br><br>[Figure 395]|
|---|

|[Figure 396]<br><br>[Figure 397]|
|---|

|[Figure 398]<br><br>[Figure 399]|
|---|

|[Figure 400]<br><br>[Figure 401]|
|---|

|[Figure 402]<br><br>[Figure 403]|
|---|

|[Figure 404]<br><br>[Figure 405]<br><br>|[Figure 406]|
|---|
|
|---|

|[Figure 407]<br><br>[Figure 408]<br><br>|[Figure 409]|
|---|
|
|---|

Input Video

Pix2Video

TokenFlow

CCEdit

“A person riding a horse over an obstacle, Van Gogh style.”

- Figure 8. Qualitative comparison results. Red boxes reveals TokenFlow’s inadequate local detail preservation, in contrast to our method’s detailed, coherent output. Zoom in for best view.

to hundreds of frames with 24 fps (frames per second). An example is shown in Fig. 7.

#### 5.3. State-of-the-Art Comparisons

Datasets. We employ a smaller segment of our proposed benchmark, designated as mini-BalanceCC. This subset encompasses 50 videos, each randomly selected from the original BalanceCC dataset, ensuring a representative distribution similar to that of the original collection.

Compared methods. To conduct an exhaustive comparison, we have selected eight representative video editing methodologies: Tune-A-Video [55], vid2vid-zero [53], Text2Video-zero [22], FateZero [37], Pix2Video [6], ControlVideo [60], Rerender A Video [58], and TokenFlow [14]. Method details are omitted for brevity, and can be found in supplementary. Regarding our approach, we employ depth maps as structure control. For the appearance control, we adopt the off-the-shelf method of PnPDiffusion [52] with the same hyper-parameters to automat-

Method Edit Aes. Tem. Ove. Win Tie Lose Tune-A-Video [55] 3.24 3.01 2.72 2.77 16.4 6.9 76.7 vid2vid-zero [53] 3.00 2.38 2.11 2.35 10.6 4.6 84.8 Text2Video-Zero [22] 2.07 1.43 1.41 1.48 16.5 1.3 86.2 FateZero [37] 2.47 3.16 3.30 2.79 16.6 3.6 79.8 Pix2Video [6] 3.68 2.97 2.80 2.97 29.9 5.2 64.9 ControlVideo [60] 3.01 2.71 2.60 2.66 13.8 5.6 80.6 Rerender A Video [58] 2.40 2.69 2.82 2.50 11.1 0.0 88.9 TokenFlow [14] 3.78 3.61 3.79 3.58 32.4 14.7 52.9 CCEdit (Ours) 4.06 4.00 3.74 3.87 - - -

Table 1. Left: Mean opinion scores (MOS) over different aspects of the generated video, including editing accuracy (Edit), aesthetics (Aes.), temporal consistency (Tem.), and overall impression (Ove.). Scores range from 1 to 5. Right: Win, Tie, and Lose percentage in side-by-side comparisons with CCEdit.

ically edit the center frame of each video clip. To ensure fairness in comparison, Stable Diffusion-v1.5 is used as the base model for all methods.

Evaluation metrics. In our preliminary study, we observed that automatic metrics, such as CLIP-Score [17] to assess text alignment and frame consistency, do not fully align with human preferences [29, 56, 61]. We focused on collecting human preferences for a comprehensive user study, comparing our method against recent state-of-the-art techniques based on mean opinion score (MOS) and direct comparisons. We gathered 1,119 scoring results from 33 volunteers, each reflecting all indicators for an edited video. For automatic metric results, refer to the supplementary.

Results. As illustrated in Tab. 1, CCEdit excels in both editing accuracy and aesthetic quality, and is just slightly inferior to TokenFlow in temporal smoothness. For overall impression, our approach achieved a MOS of 3.87 on a scale from 1 to 5. Among the eight reference methods, TokenFlow performed closest to ours, with an overall MOS of 3.58. The remaining seven methods scored between 1.5 to 3.0 on the MOS scale. As for direct comparisons, our method outperforms all eight reference schemes significantly. While TokenFlow remains the closest competitor, our CCEdit prevails in 52.9% of test cases against it, trails in 32.4%, and ties in 14.7% of cases.

Furthermore, Fig. 8 presents the qualitative results of the top three finalists (CCEdit, TokenFlow [14], and Pix2Video [6]). It shows that Pix2Video struggles to keep temporal coherence, while TokenFlow demonstrates noticeable blurring. In contrast, our method can accurately achieve the editing objective while maintaining the temporal coherence as well as the structure of the input video.

#### 5.4. Ablation Study

Appearance control. Fig. 9 illustrates the importance of taking the edited key frame as a reference in certain scenarios. Initially, translating video scenes into “cyberpunk”

|[Figure 410]<br><br>[Figure 411]|
|---|

|[Figure 412]<br><br>[Figure 413]|
|---|

|[Figure 414]<br><br>[Figure 415]|
|---|

|[Figure 416]<br><br>[Figure 417]|
|---|

Input

Video

“City at night, in cyberpunk style, with neon lights.”

|[Figure 418]<br><br>[Figure 419]|
|---|

|[Figure 420]<br><br>[Figure 421]|
|---|

|[Figure 422]<br><br>[Figure 423]|
|---|

|[Figure 424]<br><br>[Figure 425]|
|---|

w/o edited

key frame

|[Figure 426]<br><br>[Figure 427]|
|---|

|[Figure 428]<br><br>[Figure 429]|
|---|

|[Figure 430]<br><br>[Figure 431]|
|---|

|[Figure 432]<br><br>[Figure 433]|
|---|

w/ edited key frame

- Figure 9. Ablation study on appearance control. In some challenging cases, appearance control is crucial to achieving the expected results.

|[Figure 434]<br><br>[Figure 435]|
|---|

|[Figure 436]<br><br>[Figure 437]<br><br>| |
|---|
|
|---|

|[Figure 438]<br><br>[Figure 439]<br><br>| |
|---|
|
|---|

|[Figure 440]<br><br>[Figure 441]|
|---|

|[Figure 442]<br><br>[Figure 443]|
|---|

|[Figure 444]<br><br>[Figure 445]|
|---|

|[Figure 446]<br><br>[Figure 447]|
|---|

|[Figure 448]<br><br>[Figure 449]|
|---|

|[Figure 450]<br><br>[Figure 451]|
|---|

|[Figure 452]<br><br>[Figure 453]|
|---|

|[Figure 454]<br><br>[Figure 455]|
|---|

|[Figure 456]<br><br>[Figure 457]|
|---|

w/o anchor prior

w/ anchor

prior

“A man wanders in the field, with the Milky Way in the sky.”

Input Video

- Figure 10. Ablation study on anchor prior. Our proposed anchor prior helps a lot in stabilizing the appearance across frames. The red boxes demonstrate the localized flickering in the frames.

style (1st row) solely through prompt adjustments appears challenging, as this word is unfamiliar to the pre-trained T2I model weights and the temporal consistency modules. Providing a customized center frame allows the network to smoothly extend its appearance to adjacent frames, creating a cohesive video. Besides, we replicated the user study pipeline from Sec. 5.3 to evaluate the effectiveness of appearance control. The model without appearance control received a mean opinion score (MOS) of 2.88, significantly lower than the 3.87 scored by the process of editing one key frame first and then propagating to surrounding frames.

Anchor prior. Fig. 10 demonstrates the ablation study for our anchor prior. It reveals that the absence of the anchor prior may lead to regional flickering in the video sequence, while its presence effectively mitigates this issue.

### 6. Limitation and Future Works

In our approach, structural control is exerted by explicitly extracting the structural representation from the source video and sustaining it via the structure branch. However, it may encounter challenges when tasked with substantial structural alterations-exemplified by the conversion of a “cute rabbit” into a “majestic tiger.” Addressing these complexities will be a primary objective of our future work.

### 7. Conclusion

This paper presents an innovative trident network architecture specifically designed for generative video editing. This unified framework enables precise and controllable video editing while broadening creative possibilities. To address the challenges in evaluating generative video editing approaches, we introduce the meticulously curated BalanceCC benchmark dataset. Our aim is to pave the way for researchers in the generative video editing domain and equip practitioners with indispensable tools for their creative workflows.

### References

- [1] Chatgpt can now see, hear, and speak. https://openai. com/blog/chatgpt-can-now-see-hear-andspeak, 2023. 2, 5, 12
- [2] AUTOMATIC1111. Stable Diffusion Web UI, 2022. 7
- [3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022. 3
- [4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,

2021. 6

- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2, 4, 17
- [6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23206–23217, 2023. 3, 7, 8, 14, 15, 16
- [7] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23040–23050, 2023. 2
- [8] Caroline Chan, Fr´edo Durand, and Phillip Isola. Learning to generate line drawings that convey geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7915–7925, 2022. 2, 4, 6
- [9] Civitai. Civitai. https://civitai.com/, 2022. 7, 14
- [10] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations, 2022. 3
- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2

- [12] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 3
- [13] Hugging Face. Hugging face. https://huggingface. co/, 2022. 7
- [14] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 3, 7, 8, 14, 15, 16, 17
- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3, 4, 14
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2022. 3
- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, 2021. 8, 15
- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 6
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [20] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [21] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Lowrank adaptation of large language models. In International Conference on Learning Representations, 2021. 2, 4, 14
- [22] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 3, 7, 8, 14, 15, 16
- [23] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023. 15
- [24] Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng. Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749,

2023. 3

- [25] Jia-Wei Liu, Yan-Pei Cao, Jay Zhangjie Wu, Weijia Mao, Yuchao Gu, Rui Zhao, Jussi Keppo, Ying Shan, and Mike Zheng Shou. Dynvideo-e: Harnessing dynamic nerf for large-scale motion-and view-change human-centric video editing. arXiv preprint arXiv:2310.10624, 2023. 2

- [26] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023. 3
- [27] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2021. 3
- [28] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 14
- [29] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 3, 8, 15
- [30] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 17
- [31] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pages 16784–16804. PMLR, 2022. 2
- [32] OpenAI. Gpt-4v(ision) system card. 2023. 2, 5, 12
- [33] OpenAI. Gpt-4v(ision) technical work and authors. https: //cdn.openai.com/contributions/gpt-4v. pdf, 2023.
- [34] OpenAI. Gpt-4 technical report, 2023. 2, 5, 12
- [35] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023. 2
- [36] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 3
- [37] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023. 2, 3, 7, 8, 14, 15, 16
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 15
- [39] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 2

- [40] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [41] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 2, 4, 6
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4, 13
- [43] Denis Zavadski Carsten Rother. Controlnet-xs. 2023. 17
- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 4, 14
- [45] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [46] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 13
- [47] Chaehun Shin, Heeseung Kim, Che Hyun Lee, Sang-gil Lee, and Sungroh Yoon. Edit-a-video: Single video editing with object-aware consistency. arXiv preprint arXiv:2303.07945,

2023. 3

- [48] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, 2022. 2, 4
- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020. 6, 14
- [50] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020. 2
- [51] Zhuo Su, Wenzhe Liu, Zitong Yu, Dewen Hu, Qing Liao, Qi Tian, Matti Pietik¨ainen, and Li Liu. Pixel difference networks for efficient edge detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5117–5127, 2021. 2, 4, 6
- [52] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven

- image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 3, 7, 16
- [53] Wen Wang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 3, 7, 8, 14, 15, 16
- [54] Yuanzhi Wang, Yong Li, Xin Liu, Anbo Dai, Antoni Chan, and Zhen Cui. Edit temporal-consistent videos with image diffusion model. arXiv preprint arXiv:2308.09091, 2023. 2
- [55] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 3, 7, 8, 14, 15, 16
- [56] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, et al. Cvpr 2023 text guided video editing competition. arXiv preprint arXiv:2310.16003, 2023. 5, 6, 8, 15
- [57] Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. arXiv preprint arXiv:2308.09710, 2023. 2, 3
- [58] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954, 2023. 2, 3, 7, 8, 14, 15, 16
- [59] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023. 2, 3, 4, 5, 6, 12, 14
- [60] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 3, 7, 8, 14, 15, 16
- [61] Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, and Luoqi Liu. Towards consistent video editing with text-to-image diffusion models. arXiv preprint arXiv:2305.17431, 2023. 8, 15
- [62] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098,

2023. 2, 3

### A. Details of the Trident Network

The detailed architecture of our proposed trident network is illustrated in Fig. 11. Specifically, in the appearance branch, the edited key frame cja is encoded by the VAE encoder E. Then it’s fed into the encoder of appearance branch. Subsequently, the features extracted from each layer are fed into zero convolutions and the output are added to the corresponding features in the encoder side of main branch. On the right side, i.e., the structure branch, structure information cs of original video clip is encoded by the zero convolution and fed into structure branch encoder. Similar to the appearance branch, features extracted are fed into zero convolutions. Differently, the output are added to the corresponding features in the decoder side of main branch. The structure branch is instantiated by ControlNet [59]. Note that in the original paper of ControlNet, it consists a tiny network to encode the pixel-wise structure representation. Here we omit it for simplicity. Ultimately, the appearance information within the key frame is propagated to all frames through the temporal modules and the inherited structure information will ensure the structural fidelity, achieving the stable and controllable editing.

It is important to highlight that, we don’t use a trainfrom-scratch tiny encoder to encode the condition as ControlNet [59] does in the appearance branch. Instead, we use the VAE encoder E to map the pixel-wise appearance into latent variable, which is in the same representation space as latent variable z0. The intuition behind is its inherent capacity to act as a natural bridge, mapping pixel-wise appearance into the latent space which is exactly the U-Net works in. Consequently, we are able to seamlessly copy the weights from the main branch encoder to initialize appearance branch, thereby accelerating and stabilizing the convergence process.

### B. BalanceCC Benchmark

Our objective is to develop a benchmark dataset specifically designed for tasks involving controllable and creative video editing. Therefore, we collected 100 open-license videos of different categories, including Animal, Human, Object, and Landscape. In addition, for each source video, we provided a text description and graded Camera Motion, Object Motion, and Scene Complexity on a scale from 1 to 3. For each video, there are four types of edits along with corresponding target prompts and Fantasy Levels (also ranging from 1 to 3), namely Style Change, Object Change, Background Change, and Compound Change. Our aim in doing so is to better compare the strengths and weaknesses of different methods and their areas of expertise, as well as to assist researchers in advancing their techniques. In this section, we provide details about how to prompt GPT-4V(ision) [1, 32– 34] to assistant us to establish our proposed BalanceCC

Key Frame Input

Structure Condition

| |𝐜𝑎𝑗| |
|---|---|---|
| | | |

| |𝐱0| |
|---|---|---|
| | | |

𝐜𝑠

VAE Encoder

VAE Encoder

| | | |
|---|---|---|
| |𝐳0| |
| | | |

Zero Conv

Add Noise

ℰ(𝐜𝑎𝑗)

𝐳𝑡

+

|3 ×|
|---|

|× 3|
|---|

|× 3|
|---|

| | |
|---|---|
|× 3| |
|3<br><br>3| |
|× 3| |
| | |

Appearance Branch Encoder

Main Branch Encoder

Structure Branch Encoder

3 ×

× 3

| | |
|---|---|
|×| |
|| | |
|---|---|
|× 3| |
| | |
|× 3| |
| | |
| |
|×| |
| | |

3 ×

× 3

| | |
|---|---|
| |3 ×|
| | |
| |3 ×|
| | |

| | |
|---|---|
|× 3| |
| | |
|× 3| |
| | |

× 3

Middle Block Middle Block

Middle Block

Zero Conv

Zero Conv

Zero Conv

Zero Conv

Main Branch Decoder

3 ×

Zero Conv Zero Conv

Zero Conv Zero Conv

3 ×

|3 ×|
|---|

|× 3|
|---|

Zero Conv

Zero Conv

Output

Add on key frame Skip connection

Add on all frames

Figure 11. Illustration of our proposed trident network. Left: Appearance branch. Middle: Main branch. Right: Structure branch. Text prompts and time embedding are incorporated are omitted for simplicity.

benchmark and some illustrative examples. The BlanceCC benchmark will be public soon.

#### B.1. Prompting Pipeline and Instructions

GPT-4V(ision) [1, 32–34] is a multi-modal model that possesses powerful capabilities in visual understanding, language comprehension, conversational skills, and a vast repository of knowledge. Consequently, we aim to leverage these dual capabilities to help us establish the BalanceCC benchmark. The process is akin to seeking advice from a wise person with extensive knowledge and excellent vision. Specifically, we first inform GPT-4V(ision) of our intention to create a benchmark dataset dedicated to video editing, explaining our requirements including scene complexity, original prompts, target prompts, editing types, and the corresponding fantasy levels. Then, we send the center frame of each video clip to GPT-4V(ision), allowing it to output the content we need in the specified format. In our initial attempts, we observed that GPT-4V(ision) still experienced some hallucinations, overly detailed descriptions and expansions, and instances of forgetfulness during interactions. Consequently, we made repeated and emphasized adjustments in our prompt. Additionally, we found that merely describing our needs was insufficient to achieve the desired results. Our solution was to provide corresponding examples as references, which significantly improved the quality of the content provided by GPT-4V(ision). The final prompt we used is as follows,

Now I’m trying to build a benchmark for video editing. I need you to assist me in doing that. I will provide the center frame of each video for you. About the image, I hope you provide the following information to me:

- 1. Classify this video into one of “Human,

Animal, Object, Landscape”.

- 2. Describe this image, be brief, concise, and

precise. Don’t use too many adjectives.

- 3. Try to generate four text prompts of dif-

ferent types to edit this video. Be creative and imaginative. Offer me the corresponding “Editing Type”, “Target Prompt”, and “Fantasy Level” of each prompt. The “Editing Type” should be one of “Style Change, Object Change, Background Change, and Compound Change.” About “Style Change”, some examples are “old movies”, “impressionist style”, “Van Gogh style”, “neon lights style”, “cyberpunk style”, “sepia-toned photo”, “grayscale”, “claymation style”, “origami style”, “oil painting style”. About “Object Change”, just change the object into other ones, like “dog to cat”, “cat to tiger”, “human to bear”, “human to teddy bear”, and even some specific identities like “Ironman”. About “Background Change”, just change the background, here are some examples, “in the Mars”, “in the moon”, “in the forest”, “in the ocean”, “in the castle”. You can pick one of the examples I provided, and I hope you can also consider other ones that you think are interesting or suit this video. About “Compound Change”, just combine what mentioned above. Please remember, be creative and imaginative, and don’t be too outrageous. Besides, all targets including “Style Change, Object Change, Background Change, and Compound Change” should be provided for one video. The form of “Target Prompt” should be just like a description of an video, don’t say something like “Transform the background into moon.” Here is an example, the original prompt is “A majestic black swan gracefully floats on calm waters, with its reflection visible.”, the “Target Prompt” can be “An elegant flamingo swan gracefully floats on calm waters, with its reflection visible, set against a backdrop of a mystical enchanted forest.”. As for the “Fantasy Level” for each “Target Prompt”, it indicates the degree of imagination. For example, if you change the cat to a tiger or change the background from autumn to winter, it can be seen as a relatively low degree of imagination. Transforming a cat into pixel tiger or tiger made of origami is relative high degree of imagination. Here is also 1-3 in total 3 levels. And similar to the description, be brief, concise, and precise.

- 4. Is the scene complex or not? Rank it from

1 to 3, corresponding to simple, moderate, and complex.

- B.2. Human Refinement

Upon receiving initial outcomes from GPT-4V(ision), we engaged in a manual refinement and augmentation process. This primarily entailed the verification and rectification of existing annotations, along with the inclusion of additional details regarding the magnitude of camera and object motion within the video sequences. Specifically, our rule to define levels of different attributes is as follows:

Camera Motion: 1 corresponds to stationary, indicating minimal scene change and camera movement. 2 corresponds to slow movement, where the camera moves steadily and slowly. 3 corresponds to scenarios with intense camera shake and rapid movement.

Object Motion: 1 corresponds to stationary, where the target is almost motionless or has very minimal movement. 2 corresponds to slow movement, where the target follows a slow, simple, and regular trajectory (such as uniform linear motion). 3 corresponds to targets engaging in fast and complex movements (such as dancing and boxing).

Scene Complexity: 1 corresponds to scenes with a single target and a clean background. 2 corresponds to scenes with a few targets where both the targets and the background are not complex. 3 corresponds to scenes with multiple foreground targets, complex backgrounds, and intricate depth relationships.

Fantasy Level: 1 corresponds to simple target or background replacements and style transfers, such as transforming a dog into a cat or shifting to a Van Gogh painting style. 2 corresponds to more creative target and background replacements and style transfers, like replacing the background with a Martian landscape or turning an airplane into a dragon. 3 corresponds to complex and creative editing objectives combined together, with the Fantasy Level for Compound Change generally being 3.

- B.3. Illustrative Examples Four illustrative examples are shown in Fig. 15.

### C. Experiments

#### C.1. Personalized T2I Models

As mentioned in the main text, our method can integrate off-the-shelf personalized models as plugins, enabling the generation of domain-specific results. In this section, we briefly introduce the principles and specific implementations of personalized models.

Stable Diffusion [42] is trained on a huge dataset that encompasses a broad spectrum of domains [46]. Although the Stable Diffusion model is highly versatile and capable of generating a wide array of images, it occasionally falls short in specific details, particularly when it comes to generating human faces and hands, where subtle variations

Model Name Type

Counterfeit T2I Base Model ToonYou T2I Base Model rev Animated T2I Base Model HelloMecha T2I Base Model hellonijicute25d T2I Base Model

A-Zovya Photoreal LoRA kMechAnimal LoRA Pixel Art Style LoRA

fat animal LoRA Building Block World LoRA

MoXin LoRA mechanical dog LoRA

Table 2. Personalized models utilized in this paper, all sourced from CivitAI [9].

can markedly influence the overall perception. Additionally, it often struggles to precisely meet users’ expectations for specific content, styles, and attributes. Therefore, personalized T2I models are designed to address these challenges. Two respective methods are DreamBooth [44] and LoRA [21]. The former uses a unique string as an indicator to represent the corresponding domain or concept during training. Once trained, this indicator can be employed to transfer the expectations to the fine-tuned T2I model. DreamBooth faces challenges due to the extensive weight parameters, making communication less convenient. To use much less parameters and inherent the generalization of the base model, LoRA fine-tunes the model by preserving all original parameters and introducing the weight residuals ∆W to update the weights W. This process is formulated

- as W′ = W + α∆W, where α is the hyperparameter that controls the significance of the added ∆W. Typically, the parameters of ∆W are significantly fewer than those of W. Finally, two additional methods for creating robust personalized T2I base models are fine-tuning the entire model directly on the self-collected datasets and blending parameters from various models. Personalized T2I models play a crucial role in today’s AI content generation. They empower both beginners and seasoned artists, as well as enthusiasts, to swiftly and autonomously produce stunning images and create new models. A significant objectives of our framework is to ensure compatibility with personalized T2I models, allowing creators to freely combine and perform highly creative edits on videos using models from the community.

In this paper, we collect several personalized T2I base models and LoRA weights from CivitAI [9] and explored different combinations, which are illustrated in Table 2. Similar to previous work [15], we employ the “trigger words” to activate these personalized models. α of all LoRA models is set as 0.9.

- C.2. More Visualizations Fig. 17 shows several visualized results of CCEdit.
- C.3. Comprehensive Comparison C.3.1 Compared Methods

We compared our methods with eight state-of-the-art generative video editing methods: Tune-A-Video [55], vid2vid-zero [53], Text2Video-zero [22], FateZero [37], Pix2Video [6], ControlVideo [60], Rerender A Video [58], and TokenFlow [14]. The brief descriptions of these methods are as follows:

Tune-A-Video [55] propose the sparse attention mechanism to maintain the temporal coherence and optimize the network parameters through training on the source video. DDIM inversion [49] is utilized to preserve the structure of input video.

Vid2vid-zero [53] utilizes off-the-shelf image diffusion models and employs the null-text inversion module [28] for text-to-video alignment. Additionally, it incorporates a cross-frame modeling module to ensure temporal consistency and a spatial regularization module to maintain fidelity to the original video.

Text2Video-zero [22] introduces a method to enhance the latent codes of generated frames with motion dynamics, ensuring global scene and temporal consistency in the background. Additionally, it reprograms frame-level self-attention through cross-frame attention, focusing each frame on the first one to maintain the context, appearance, and identity of the foreground object.

FateZero [37] proposes to capture intermediate attention maps during inversion process, enhancing structural and motion information retention, and employs a novel spatialtemporal attention mechanism in the denoising UNet for improved frame consistency.

Pix2Video [6] involves two steps to conduct generative video editing: initially, using a structure-guided (e.g., depth) image diffusion model to edit an anchor frame based on text prompts, followed by a key step of progressively propagating these edits to subsequent frames. This is done via self-attention feature injection, adapting the core denoising phase of the diffusion model. Adjustments are then made to the latent code of each frame before continuing the process.

ControlVideo [60] leverages ControlNet [59] to ensure the structural consistency from input video clips. In addition, it introduces full cross-frame interaction in selfattention modules for appearance coherence, an interleavedframe smoother to reduce flickering through frame interpolation.

Rerender A Video [58] propose to tackle the task of video editing by two parts: key frame translation and full video translation. Initially, it employs an adapted diffusion model

to generate key frames, applying hierarchical cross-frame constraints to ensure coherence in shapes, textures, and colors. Subsequently, the framework extends these key frames to other frames using temporal-aware patch matching and frame blending techniques.

TokenFlow [14] propose the idea that the edited features convey the same inter-frame correspondences and redundancy as the original video features. Therefore, it propagates diffusion features based on inter-frame correspondences inherent in the model to ensure consistency in the diffusion feature space.

During the evaluation, all the videos consist of 17 frames

- at 6fps. We select depth maps as the structural representation. Additionally, to ensure fairness, the base model for all methods is Stable Diffusion v1.5.

##### C.3.2 Qualitative Results

The qualitative results for two videos are presented in Fig. 18 and Fig. 19. It can be observed that Tune-A-Video achieves effective editing that aligns well with the specified prompts, but falls short in maintaining temporal consistency and tends to produce overly contrasted images, possibly due to overfitting to the source video and excessively high default classifier-free guidance settings. Vid2vid-zero, Text2Video-Zero, and Pix2Video also struggle with insufficient temporal coherence. While FateZero exhibits better temporal coherence, its editing accuracy is not optimal. ControlVideo, despite its reasonable editing accuracy and temporal coherence, lacks a natural feel in its edited videos due to its global attention mechanism and interleaved-frame smoother technique. Rerender A Video demonstrates a limitation in executing precise edits, potentially due to an excessive dependence on detailed structural control mechanisms (line drawing and Canny edge of ControlNet). Such mechanisms restrict the method predominantly to minor stylistic alterations. TokenFlow achieves stable results in both temporal coherence and editing accuracy, yet it still encounters blurring issues in scenes with significant object motion or rapid camera movements (see the horse legs in Fig. 19). At last, our approach demonstrates a notable capacity for sustaining temporal consistency, coupled with achieving exceptional accuracy in editing.

##### C.3.3 Quantitative Results

Automatic Metrics. Our evaluation metrics include two aspects of both automatic ones and user study results. Automatic metrics are mainly conducted through the trained CLIP [17, 23, 38] model, similar to previous methods [6, 37, 55, 60]. Specifically, “Tem-Con” evaluates the temporal consistency of edited frames by calculating the similarity between successive frame pairs. Meanwhile, “Tex-Ali”

Method Tem-Con ↑ Tex-Ali ↑ Pick ↑ Tune-A-Video [55] 0.937 0.284 0.206 vid2vid-zero [53] 0.933 0.284 0.209 Text2Video-Zero [22] 0.949 0.262 0.203 FateZero [37] 0.942 0.245 0.205 Pix2Video [6] 0.939 0.285 0.208 ControlVideo [60] 0.950 0.285 0.210 Rerender A Video [58] 0.928 0.247 0.201 TokenFlow [14] 0.949 0.270 0.210 CCEdit (Ours) 0.936 0.281 0.213

Table 3. State-of-the-art comparison of automatic metrics. “Tem-Con” represents temporal consistency, “Text-Ali” indicates textural alignment, and “Pick” represents to the PickScore [23].

quantifies frame-wise editing accuracy, represented as the cosine similarity between edited frames and target prompts. Additionally, the PickScore [23] is incorporated to predict the aesthetic quality and user preference of the edited videos. Regarding the user study, we designed an interface and invited 33 volunteers to score the videos and pickup the winners, receiving a total of 1119 ratings. Each rating corresponds to various aspects of a single video. Specifically, the aspects to be rated include: “Editing Accuracy”, representing whether the edited video accurately achieves the intended meaning of the target prompt; “Aesthetics”, denoting the visual appeal of the edited video; “Temporal Consistency”, indicating whether the video maintains coherence over time; and “Overall Impression”, which reflects the subjective overall rating of the video. The interface is illustrated in Fig. 16.

Results of Automatic Metrics. The results are illustrated in Tab. 3. Although our method ranked second in temporal consistency and first in text alignment in the table of user study presented in the main text, it did not particularly stand out in terms of corresponding objective metrics. This observation has been noted in many previous works [29, 56, 61], further emphasizing the significance of more advanced objective automatic metrics for the development of this field. Finally, our method achieved the best performance in the CLIP-based scoring function, PickScore, an indicator of human preference, demonstrating its superior alignment with human subjective perceptions.

##### C.3.4 Runtime Analysis

Tab. 4 presents the runtime of various methods, detailing the time spent on pre-processing, inference, and the total duration, respectively. Pre-processing includes tasks of finetuning on the source video, performing inversion operations, caching attention maps, key frame editing, and others. The inference time represents the duration of the sampling process, along with all the associated operations. Overall, the time consumed by our method is not lengthy compared

Method Pre-Processing Inference Total

Tune-A-Video [55] 545 22 567 vid2vid-zero [53] 148 230 378 Text2Video-Zero [22] 0 28 28 FateZero [37] 199 42 241 Pix2Video [6] 0 188 188 ControlVideo [60] 0 56 56 Rerender A Video [58] 76 96 172 TokenFlow [14] 182 27 209

CCEdit (Ours) 134 46 170

Table 4. Runtime comparison (seconds).

to other video editing techniques. It is worth noted that in our method, the time spent on key frame editing using PnP [52]) during pre-processing constitutes the majority of the total time, while the actual sampling time is relatively brief. It’s attributed to the absence of any inversion and attention map operations. The only additional computational overhead arises from the extra network parameters introduced during the network forward process. In practical applications, one can opt for more advanced and lightweight image editing methods or manually make fine adjustments, thereby achieving the desired trade-off. This further demonstrates the practicality and flexibility of our approach.

#### C.4. Study on Control Scales

Structure Branch. Sometimes, the appearance of the edited key frame may structurally differ from the corresponding structure representation of the original video. Since the features of the structure branch are injected into the main branch through summation, the intensity of structure information infusion can be adjusted by modifying the coefficients (named control scale) applied to the features during this summation process. In such cases, reducing the control scale of the structure branch could help. This adjustment lessens its structural constraints on the results, allowing for a greater reliance on the information provided by the appearance branch and adherence to the coherence adjustments made by the temporal layers. The visualized results are shown in Fig. 12. It can be observed that in the edited key frame, the astronaut’s silhouette appears markedly larger than that of the original person, a consequence of the voluminous spacesuit. When the structure control is relatively high (0.6∼1.0), the editing results show that the center frame remain consistent with the edited frame, while the structure of other frames is overly constrained by the structure representation. At a control scale of 0, the loss of structure information leads to the astronaut being unable to move correctly. However, with a moderate control scale (0.2∼0.4), a better trade-off is achieved in terms of appearance, structure, and motion. Note that in comparisons with other methods, to ensure fairness, our

|[Figure 458]<br><br>[Figure 459]|
|---|

|[Figure 460]<br><br>[Figure 461]|
|---|

|[Figure 462]<br><br>[Figure 463]|
|---|

|[Figure 464]<br><br>[Figure 465]|
|---|

|[Figure 466]<br><br>[Figure 467]|
|---|

Input

Video

|[Figure 468]<br><br>[Figure 469]|
|---|

|[Figure 470]<br><br>[Figure 471]|
|---|

|[Figure 472]<br><br>[Figure 473]|
|---|

|[Figure 474]<br><br>[Figure 475]|
|---|

|[Figure 476]<br><br>[Figure 477]|
|---|

Depth

Maps

|[Figure 478]<br><br>[Figure 479]|
|---|

Edited

Key Frame

|[Figure 480]<br><br>[Figure 481]|
|---|

|[Figure 482]<br><br>[Figure 483]|
|---|

|[Figure 484]<br><br>[Figure 485]|
|---|

|[Figure 486]<br><br>[Figure 487]|
|---|

|[Figure 488]<br><br>[Figure 489]|
|---|

Scale=0.0

|[Figure 490]<br><br>[Figure 491]|
|---|

|[Figure 492]<br><br>[Figure 493]|
|---|

|[Figure 494]<br><br>[Figure 495]|
|---|

|[Figure 496]<br><br>[Figure 497]|
|---|

|[Figure 498]<br><br>[Figure 499]|
|---|

Scale=0.2

|[Figure 500]<br><br>[Figure 501]|
|---|

|[Figure 502]<br><br>[Figure 503]|
|---|

|[Figure 504]<br><br>[Figure 505]|
|---|

|[Figure 506]<br><br>[Figure 507]|
|---|

|[Figure 508]<br><br>[Figure 509]|
|---|

Scale=0.4

|[Figure 510]<br><br>[Figure 511]|
|---|

|[Figure 512]<br><br>[Figure 513]|
|---|

|[Figure 514]<br><br>[Figure 515]|
|---|

|[Figure 516]<br><br>[Figure 517]|
|---|

|[Figure 518]<br><br>[Figure 519]|
|---|

Scale=0.6

|[Figure 520]<br><br>[Figure 521]|
|---|

|[Figure 522]<br><br>[Figure 523]|
|---|

|[Figure 524]<br><br>[Figure 525]|
|---|

|[Figure 526]<br><br>[Figure 527]|
|---|

|[Figure 528]<br><br>[Figure 529]|
|---|

- Scale=0.8
- Scale=1.0

|[Figure 530]<br><br>[Figure 531]|
|---|

|[Figure 532]<br><br>[Figure 533]|
|---|

|[Figure 534]<br><br>[Figure 535]|
|---|

|[Figure 536]<br><br>[Figure 537]|
|---|

|[Figure 538]<br><br>[Figure 539]|
|---|

Figure 12. Results at different scales of structure branch. The target prompt is “An astronaut with a jetpack floats above a Martian landscape, with red rocky terrains and tall, alien-like mountains in the backdrop.”

method consistently employed a control scale of 1.

Appearance Branch. Since the features of the Appearance Branch are also injected into the main branch through summation, the intensity of appearance information infusion can similarly be adjusted by tuning the summation coefficients of the appearance branch. The results are shown in Fig. 13. At a lower control scale (0∼0.2), the influence of appearance information is minimal, barely impacting the edited video. When the control scale is moderate (0.4∼0.6), appearance information begins to play a role. However, possibly due to conflicts with the priors of the main branch, this results in a somewhat dull and dark color tone in the visuals. Conversely, at a higher control scale (0.8∼1.0), appearance information exerts a decisive control over the overall appearance of the edited video.

#### C.5. Study on Text Prompt

Another point worth exploring is whether text prompts are still necessary after introducing appearance control. To address this, we conducted a visual experiment. As shown in Fig. 14, providing a normal text prompt leads to correct

|[Figure 540]<br><br>[Figure 541]|
|---|

|[Figure 542]<br><br>[Figure 543]|
|---|

|[Figure 544]<br><br>[Figure 545]|
|---|

|[Figure 546]<br><br>[Figure 547]|
|---|

|[Figure 548]<br><br>[Figure 549]|
|---|

Input

Video

|[Figure 550]<br><br>[Figure 551]|
|---|

|[Figure 552]<br><br>[Figure 553]|
|---|

|[Figure 554]<br><br>[Figure 555]|
|---|

|[Figure 556]<br><br>[Figure 557]|
|---|

|[Figure 558]<br><br>[Figure 559]|
|---|

Depth

Maps

|[Figure 560]<br><br>[Figure 561]|
|---|

Edited

Key Frame

|[Figure 562]<br><br>[Figure 563]|
|---|

|[Figure 564]<br><br>[Figure 565]|
|---|

|[Figure 566]<br><br>[Figure 567]|
|---|

|[Figure 568]<br><br>[Figure 569]|
|---|

|[Figure 570]<br><br>[Figure 571]|
|---|

Scale=0.0

|[Figure 572]<br><br>[Figure 573]|
|---|

|[Figure 574]<br><br>[Figure 575]|
|---|

|[Figure 576]<br><br>[Figure 577]|
|---|

|[Figure 578]<br><br>[Figure 579]|
|---|

|[Figure 580]<br><br>[Figure 581]|
|---|

Scale=0.2

|[Figure 582]<br><br>[Figure 583]|
|---|

|[Figure 584]<br><br>[Figure 585]|
|---|

|[Figure 586]<br><br>[Figure 587]|
|---|

|[Figure 588]<br><br>[Figure 589]|
|---|

|[Figure 590]<br><br>[Figure 591]|
|---|

Scale=0.4

|[Figure 592]<br><br>[Figure 593]|
|---|

|[Figure 594]<br><br>[Figure 595]|
|---|

|[Figure 596]<br><br>[Figure 597]|
|---|

|[Figure 598]<br><br>[Figure 599]|
|---|

|[Figure 600]<br><br>[Figure 601]|
|---|

Scale=0.6

|[Figure 602]<br><br>[Figure 603]|
|---|

|[Figure 604]<br><br>[Figure 605]|
|---|

|[Figure 606]<br><br>[Figure 607]|
|---|

|[Figure 608]<br><br>[Figure 609]|
|---|

|[Figure 610]<br><br>[Figure 611]|
|---|

- Scale=0.8
- Scale=1.0

|[Figure 612]<br><br>[Figure 613]|
|---|

|[Figure 614]<br><br>[Figure 615]|
|---|

|[Figure 616]<br><br>[Figure 617]|
|---|

|[Figure 618]<br><br>[Figure 619]|
|---|

|[Figure 620]<br><br>[Figure 621]|
|---|

Figure 13. Results at different scales of appearance branch. The target prompt is “An astronaut with a jetpack floats above a Martian landscape, with red rocky terrains and tall, alien-like mountains in the backdrop.”

results, whereas the absence of any text prompt results in significant distortions in the generated output. When given a text prompt that contradicts the appearance information, only the center frame retains the appearance information, while the other frames are controlled by the text prompt. Consequently, the conclusion is that text prompts are still necessary within this framework. We believe this may be due to the weights of the main branch and the structure branch being frozen during the training process. As a result, the entire editing process seems to involve the appearance branch exerting more detailed control over the image after the text prompt has already provided a coarse guide.

### D. Limitation and Future Works D.1. Structural Deviation

As described in the main text, a primary challenge that needs addressing in our video editing approach is the structural deviation (also the major issue mentioned in TokenFlow [14]) between the input and target videos. This deviation could stem from semantic changes inherent to the tar-

|[Figure 622]<br><br>[Figure 623]|
|---|

|[Figure 624]<br><br>[Figure 625]|
|---|

|[Figure 626]<br><br>[Figure 627]|
|---|

|[Figure 628]<br><br>[Figure 629]|
|---|

|[Figure 630]<br><br>[Figure 631]|
|---|

Input Video

|[Figure 632]<br><br>[Figure 633]|
|---|

Edited Key Frame

|[Figure 634]<br><br>[Figure 635]|
|---|

|[Figure 636]<br><br>[Figure 637]|
|---|

|[Figure 638]<br><br>[Figure 639]|
|---|

|[Figure 640]<br><br>[Figure 641]|
|---|

|[Figure 642]<br><br>[Figure 643]|
|---|

Normal Text Prompt

|[Figure 644]<br><br>[Figure 645]|
|---|

|[Figure 646]<br><br>[Figure 647]|
|---|

|[Figure 648]<br><br>[Figure 649]|
|---|

|[Figure 650]<br><br>[Figure 651]|
|---|

|[Figure 652]<br><br>[Figure 653]|
|---|

No Text Prompt

|[Figure 654]<br><br>[Figure 655]|
|---|

|[Figure 656]<br><br>[Figure 657]|
|---|

|[Figure 658]<br><br>[Figure 659]|
|---|

|[Figure 660]<br><br>[Figure 661]|
|---|

|[Figure 662]<br><br>[Figure 663]|
|---|

Contradicted Text Prompt

Figure 14. Illustration of results with different text prompts. The normal prompt is “A bear is walking”. The contradicted text prompt is “A tiger is walking”. The “ToonYou” personalized T2I model is used.

get or from alterations in the target’s behavior. For instance, transitioning from a “cute rabbit” to a “fierce tiger” is challenging due to their fundamentally different physiological structures. Most existing methods struggle to overcome this hurdle and often only manage to modify their textural appearance. In our approach, adjusting the scale coefficient of structure branch and employing coarser-grained structure representations (like the skeleton) may alleviate this issue to some extent, but we believe it doesn’t fundamentally solve the problem. Achieving changes in the target’s behavior, such as transforming a “running bear” into a “dancing bear”, is even more challenging. This complexity arises primarily because most contemporary generative video editing methods employ Text-to-Image (T2I) models at the image level. These models, devoid of prior knowledge concerning actions, encounter difficulties in editing motion.

We posit that a promising approach could be to integrate a pre-trained T2V (text-to-video) model, cleverly utilizing its priors to tackle these challenges.

#### D.2. Heavy Appearance and Structure Branch

In CCEdit, the appearance and structure branch utilize two heavy encoder to extract features, consisting significant amount of parameters. This may be unnecessary and could lead to issues such as increased GPU memory consumption and longer editing times during use. In the future, we plan to explore the adoption of more lightweight networks [30, 43] to address these concerns.

#### D.3. Flickering Problem.

We observed flickering in videos with higher frame rates or after frame interpolation, especially noticeable in highfrequency fine texture details. This is primarily attributed to our video editing operations being performed in the latent domain encoded by the 2D autoencoder. Introducing additional temporal layers in the autoencoder [5] is an promising way to solve this problem.

[Figure 664]

[Figure 665]

- <1> Object Change: “A majestic flamingo swimming in a pond with lush greenery in the background.”
- <2> Background Change: “A black swan swimming in a crystal and clear lake surrounded by snowcapped mountains.”

|[Figure 666]<br><br>[Figure 667]|
|---|

|[Figure 668]<br><br>[Figure 669]|
|---|

|[Figure 670]<br><br>[Figure 671]|
|---|

|[Figure 672]<br><br>[Figure 673]|
|---|

|[Figure 674]<br><br>[Figure 675]|
|---|

|[Figure 676]<br><br>[Figure 677]|
|---|

|[Figure 678]<br><br>[Figure 679]|
|---|

|[Figure 680]<br><br>[Figure 681]|
|---|

|[Figure 682]<br><br>[Figure 683]|
|---|

Video Type: Animal

Original Prompt: “A black swan swimming in a pond with lush greenery in the background.”

- Camera Motion: 2

- Object Motion: 2

- Scene Complexity: 2

- <1> Style Change: “A black swan swimming in a pond with lush greenery in the background, oil painting style.”

<3> Compound Change: “A duck made of origami floating on a pond under a cherry blossom tree

in full bloom.”

|[Figure 684]<br><br>[Figure 685]|
|---|

|[Figure 686]<br><br>[Figure 687]|
|---|

|[Figure 688]<br><br>[Figure 689]|
|---|

|[Figure 690]<br><br>[Figure 691]|
|---|

|[Figure 692]<br><br>[Figure 693]|
|---|

|[Figure 694]<br><br>[Figure 695]|
|---|

|[Figure 696]<br><br>[Figure 697]|
|---|

|[Figure 698]<br><br>[Figure 699]|
|---|

|[Figure 700]<br><br>[Figure 701]|
|---|

[Figure 702]

[Figure 703]

- <2> Object Change: “A skateboarder in full gear maneuvering his skateboard over a dirt ramp in a BMX track.”

- <2> Background Change: “A BMX rider in full gear maneuvering his bike over a dirt ramp in a night-time cityscape with skyscrapers in the background.”

Video Type: Human

Original Prompt: “A BMX rider in full gear maneuvering his bike over a dirt ramp in a BMX track.”

Camera Motion: 3 Object Motion: 3 Scene Complexity: 3

- <1> Style Change: “A BMX rider in full gear maneuvering his bike over a dirt ramp in a BMX track, rendered in old movie style.”

<3> Compound Change: “A polar bear BMX rider in full gear maneuvering his bike over a ramp in a futuristic cyberpunk city, surrounded by neon billboards.”

|[Figure 704]<br><br>[Figure 705]|
|---|

|[Figure 706]<br><br>[Figure 707]|
|---|

|[Figure 708]<br><br>[Figure 709]|
|---|

|[Figure 710]<br><br>[Figure 711]|
|---|

|[Figure 712]<br><br>[Figure 713]|
|---|

|[Figure 714]<br><br>[Figure 715]|
|---|

|[Figure 716]<br><br>[Figure 717]|
|---|

|[Figure 718]<br><br>[Figure 719]|
|---|

|[Figure 720]<br><br>[Figure 721]|
|---|

[Figure 722]

[Figure 723]

- <1> Object Change: “Crystalline ice formations by a frozen water body with a mountain in the backdrop under an overcast sky.”
- <2> Background Change: “Rocks by a still water body with a mystical floating island in the sky backdrop, amidst a soft glow of sunset.”

Video Type: Landscape

Original Prompt: “Rocks by a still water body with a mountain in the backdrop under an overcast sky.”

Camera Motion: 2

- Object Motion: 1 Scene Complexity: 2

- <2> Style Change: “Rocks by a still water body with a mountain in the backdrop under an overcast sky, in Studio Ghibli style.”
- <3> Compound Change: “Luminous gemstones by the shores of a bioluminescent lake with an aurora borealis in the backdrop in a starry night scene.”

[Figure 724]

[Figure 725]

- <1> Object Change: “A vintage lantern tumbling across a grassy field beside a wire fence.”
- <2> Background Change: “A soccer ball rolling across the moon's surface with Earth in the distant skyline.”

Video Type: Object

Original Prompt: “A soccer ball rolling across a grassy field beside a wire fence.”

Camera Motion: 1 Object Motion: 2 Scene Complexity: 2

<1> Style Change: “A soccer ball rolling across a grassy field beside a wire fence, Comic Book Art style.”

- <3> Compound Change: “A glowing orb of energy bouncing across a futuristic cityscape at dusk.”

|[Figure 726]<br><br>[Figure 727]|
|---|

|[Figure 728]<br><br>[Figure 729]|
|---|

|[Figure 730]<br><br>[Figure 731]|
|---|

|[Figure 732]<br><br>[Figure 733]|
|---|

|[Figure 734]<br><br>[Figure 735]|
|---|

|[Figure 736]<br><br>[Figure 737]|
|---|

|[Figure 738]<br><br>[Figure 739]|
|---|

|[Figure 740]<br><br>[Figure 741]|
|---|

|[Figure 742]<br><br>[Figure 743]|
|---|

###### Figure 15. Illustrative examples of BalanceCC benchmark dataset.

[Figure 744]

[Figure 745]

Figure 16. Illustration of the interface to conduct user study. Initially, we provided a description of the evaluation criteria at the top of the page, along with corresponding notes for consideration. Additionally, to reduce user burden and avoid the confusion of displaying multiple videos simultaneously, for each video’s editing result, we randomly selected three from all nine options (eight comparative methods and our method) for users to rate on various criteria (Edit Accuracy, Aesthetic, Temporal Consistency, and Overall Impression). Finally, users were asked to choose the winner(s) among the three videos. Selecting multiple winners was allowed (up to two), but choosing none or all three was not permitted. Zoom in to see details.

|[Figure 746]<br><br>[Figure 747]|
|---|

|[Figure 748]<br><br>[Figure 749]|
|---|

|[Figure 750]<br><br>[Figure 751]|
|---|

|[Figure 752]<br><br>[Figure 753]|
|---|

|[Figure 754]<br><br>[Figure 755]|
|---|

|[Figure 756]<br><br>[Figure 757]|
|---|

Input Video

<Model: ReV Animated, kMechAnimal> “A mechanical bear is running.”

|[Figure 758]<br><br>[Figure 759]|
|---|

|[Figure 760]<br><br>[Figure 761]|
|---|

|[Figure 762]<br><br>[Figure 763]|
|---|

|[Figure 764]<br><br>[Figure 765]|
|---|

|[Figure 766]<br><br>[Figure 767]|
|---|

|[Figure 768]<br><br>[Figure 769]|
|---|

Edited Video

|[Figure 770]<br><br>[Figure 771]|
|---|

|[Figure 772]<br><br>[Figure 773]|
|---|

|[Figure 774]<br><br>[Figure 775]|
|---|

|[Figure 776]<br><br>[Figure 777]|
|---|

|[Figure 778]<br><br>[Figure 779]|
|---|

|[Figure 780]<br><br>[Figure 781]|
|---|

Input Video

<MajicMIX realistic> “A beautiful young girl is doing makeup.”

|[Figure 782]<br><br>[Figure 783]|
|---|

|[Figure 784]<br><br>[Figure 785]|
|---|

|[Figure 786]<br><br>[Figure 787]|
|---|

|[Figure 788]<br><br>[Figure 789]|
|---|

|[Figure 790]<br><br>[Figure 791]|
|---|

|[Figure 792]<br><br>[Figure 793]|
|---|

Edited Video

|[Figure 794]<br><br>[Figure 795]|
|---|

|[Figure 796]<br><br>[Figure 797]|
|---|

|[Figure 798]<br><br>[Figure 799]|
|---|

|[Figure 800]<br><br>[Figure 801]|
|---|

|[Figure 802]<br><br>[Figure 803]|
|---|

|[Figure 804]<br><br>[Figure 805]|
|---|

Input Video

<Model: ToonYou> “An anime-style tiger is walking.”

|[Figure 806]<br><br>[Figure 807]|
|---|

|[Figure 808]<br><br>[Figure 809]|
|---|

|[Figure 810]<br><br>[Figure 811]|
|---|

|[Figure 812]<br><br>[Figure 813]|
|---|

|[Figure 814]<br><br>[Figure 815]|
|---|

|[Figure 816]<br><br>[Figure 817]|
|---|

Edited Video

|[Figure 818]<br><br>[Figure 819]|
|---|

|[Figure 820]<br><br>[Figure 821]|
|---|

|[Figure 822]<br><br>[Figure 823]|
|---|

|[Figure 824]<br><br>[Figure 825]|
|---|

|[Figure 826]<br><br>[Figure 827]|
|---|

|[Figure 828]<br><br>[Figure 829]|
|---|

Input

Video

<Model: Counterfeit> “A girl with grey hair.”

|[Figure 830]<br><br>[Figure 831]|
|---|

|[Figure 832]<br><br>[Figure 833]|
|---|

|[Figure 834]<br><br>[Figure 835]|
|---|

|[Figure 836]<br><br>[Figure 837]|
|---|

|[Figure 838]<br><br>[Figure 839]|
|---|

|[Figure 840]<br><br>[Figure 841]|
|---|

Edited Video

|[Figure 842]<br><br>[Figure 843]|
|---|

|[Figure 844]<br><br>[Figure 845]|
|---|

|[Figure 846]<br><br>[Figure 847]|
|---|

|[Figure 848]<br><br>[Figure 849]|
|---|

|[Figure 850]<br><br>[Figure 851]|
|---|

|[Figure 852]<br><br>[Figure 853]|
|---|

Input Video

<Model: SD v1.5> “A man wanders in the field, with the Milky Way in the sky”

|[Figure 854]<br><br>[Figure 855]|
|---|

|[Figure 856]<br><br>[Figure 857]|
|---|

|[Figure 858]<br><br>[Figure 859]|
|---|

|[Figure 860]<br><br>[Figure 861]|
|---|

|[Figure 862]<br><br>[Figure 863]|
|---|

|[Figure 864]<br><br>[Figure 865]|
|---|

Edited

Video

|[Figure 866]<br><br>[Figure 867]|
|---|

|[Figure 868]<br><br>[Figure 869]|
|---|

|[Figure 870]<br><br>[Figure 871]|
|---|

|[Figure 872]<br><br>[Figure 873]|
|---|

|[Figure 874]<br><br>[Figure 875]|
|---|

|[Figure 876]<br><br>[Figure 877]|
|---|

Input Video

<Model: ToonYou> “A man is running on the beach, with sunset behind.”

|[Figure 878]<br><br>[Figure 879]|
|---|

|[Figure 880]<br><br>[Figure 881]|
|---|

|[Figure 882]<br><br>[Figure 883]|
|---|

|[Figure 884]<br><br>[Figure 885]|
|---|

|[Figure 886]<br><br>[Figure 887]|
|---|

|[Figure 888]<br><br>[Figure 889]|
|---|

Edited Video

Figure 17. Visualized results of CCEdit. ⟨·⟩ indicates the personalized T2I model we used.

|[Figure 890]<br><br>[Figure 891]|
|---|

|[Figure 892]<br><br>[Figure 893]|
|---|

|[Figure 894]<br><br>[Figure 895]|
|---|

|[Figure 896]<br><br>[Figure 897]|
|---|

|[Figure 898]<br><br>[Figure 899]|
|---|

|[Figure 900]<br><br>[Figure 901]|
|---|

|[Figure 902]<br><br>[Figure 903]|
|---|

Input Video

|[Figure 904]<br><br>[Figure 905]|
|---|

|[Figure 906]<br><br>[Figure 907]|
|---|

|[Figure 908]<br><br>[Figure 909]|
|---|

|[Figure 910]<br><br>[Figure 911]|
|---|

|[Figure 912]<br><br>[Figure 913]|
|---|

|[Figure 914]<br><br>[Figure 915]|
|---|

|[Figure 916]<br><br>[Figure 917]|
|---|

Tune-AVideo

|[Figure 918]<br><br>[Figure 919]|
|---|

|[Figure 920]<br><br>[Figure 921]|
|---|

|[Figure 922]<br><br>[Figure 923]|
|---|

|[Figure 924]<br><br>[Figure 925]|
|---|

|[Figure 926]<br><br>[Figure 927]|
|---|

|[Figure 928]<br><br>[Figure 929]|
|---|

|[Figure 930]<br><br>[Figure 931]|
|---|

vid2vid-

zero

|[Figure 932]<br><br>[Figure 933]|
|---|

|[Figure 934]<br><br>[Figure 935]|
|---|

|[Figure 936]<br><br>[Figure 937]|
|---|

|[Figure 938]<br><br>[Figure 939]|
|---|

|[Figure 940]<br><br>[Figure 941]|
|---|

|[Figure 942]<br><br>[Figure 943]|
|---|

|[Figure 944]<br><br>[Figure 945]|
|---|

Text2VideoZero

|[Figure 946]<br><br>[Figure 947]|
|---|

|[Figure 948]<br><br>[Figure 949]|
|---|

|[Figure 950]<br><br>[Figure 951]|
|---|

|[Figure 952]<br><br>[Figure 953]|
|---|

|[Figure 954]<br><br>[Figure 955]|
|---|

|[Figure 956]<br><br>[Figure 957]|
|---|

|[Figure 958]<br><br>[Figure 959]|
|---|

FateZero

|[Figure 960]<br><br>[Figure 961]|
|---|

|[Figure 962]<br><br>[Figure 963]|
|---|

|[Figure 964]<br><br>[Figure 965]|
|---|

|[Figure 966]<br><br>[Figure 967]|
|---|

|[Figure 968]<br><br>[Figure 969]|
|---|

|[Figure 970]<br><br>[Figure 971]|
|---|

|[Figure 972]<br><br>[Figure 973]|
|---|

Pix2Video

|[Figure 974]<br><br>[Figure 975]|
|---|

|[Figure 976]<br><br>[Figure 977]|
|---|

|[Figure 978]<br><br>[Figure 979]|
|---|

|[Figure 980]<br><br>[Figure 981]|
|---|

|[Figure 982]<br><br>[Figure 983]|
|---|

|[Figure 984]<br><br>[Figure 985]|
|---|

|[Figure 986]<br><br>[Figure 987]|
|---|

ControlVideo

|[Figure 988]<br><br>[Figure 989]|
|---|

|[Figure 990]<br><br>[Figure 991]|
|---|

|[Figure 992]<br><br>[Figure 993]|
|---|

|[Figure 994]<br><br>[Figure 995]|
|---|

|[Figure 996]<br><br>[Figure 997]|
|---|

|[Figure 998]<br><br>[Figure 999]|
|---|

|[Figure 1000]<br><br>[Figure 1001]|
|---|

Rerender A Video

|[Figure 1002]<br><br>[Figure 1003]|
|---|

|[Figure 1004]<br><br>[Figure 1005]|
|---|

|[Figure 1006]<br><br>[Figure 1007]|
|---|

|[Figure 1008]<br><br>[Figure 1009]|
|---|

|[Figure 1010]<br><br>[Figure 1011]|
|---|

|[Figure 1012]<br><br>[Figure 1013]|
|---|

|[Figure 1014]<br><br>[Figure 1015]|
|---|

TokenFlow

|[Figure 1016]<br><br>[Figure 1017]|
|---|

|[Figure 1018]<br><br>[Figure 1019]|
|---|

|[Figure 1020]<br><br>[Figure 1021]|
|---|

|[Figure 1022]<br><br>[Figure 1023]|
|---|

|[Figure 1024]<br><br>[Figure 1025]|
|---|

|[Figure 1026]<br><br>[Figure 1027]|
|---|

|[Figure 1028]<br><br>[Figure 1029]|
|---|

CCEdit (Ours)

- Figure 18. Qualitative comparison of different methods. The target prompt is “A dragonfly with shimmering wings perches on a plant amidst a field of golden grass.”

|[Figure 1030]<br><br>[Figure 1031]|
|---|

|[Figure 1032]<br><br>[Figure 1033]|
|---|

|[Figure 1034]<br><br>[Figure 1035]|
|---|

|[Figure 1036]<br><br>[Figure 1037]|
|---|

|[Figure 1038]<br><br>[Figure 1039]|
|---|

|[Figure 1040]<br><br>[Figure 1041]|
|---|

|[Figure 1042]<br><br>[Figure 1043]|
|---|

Input

Video

|[Figure 1044]<br><br>[Figure 1045]|
|---|

|[Figure 1046]<br><br>[Figure 1047]|
|---|

|[Figure 1048]<br><br>[Figure 1049]|
|---|

|[Figure 1050]<br><br>[Figure 1051]|
|---|

|[Figure 1052]<br><br>[Figure 1053]|
|---|

|[Figure 1054]<br><br>[Figure 1055]|
|---|

|[Figure 1056]<br><br>[Figure 1057]|
|---|

Tune-AVideo

|[Figure 1058]<br><br>[Figure 1059]|
|---|

|[Figure 1060]<br><br>[Figure 1061]|
|---|

|[Figure 1062]<br><br>[Figure 1063]|
|---|

|[Figure 1064]<br><br>[Figure 1065]|
|---|

|[Figure 1066]<br><br>[Figure 1067]|
|---|

|[Figure 1068]<br><br>[Figure 1069]|
|---|

|[Figure 1070]<br><br>[Figure 1071]|
|---|

vid2vidzero

|[Figure 1072]<br><br>[Figure 1073]|
|---|

|[Figure 1074]<br><br>[Figure 1075]|
|---|

|[Figure 1076]<br><br>[Figure 1077]|
|---|

|[Figure 1078]<br><br>[Figure 1079]|
|---|

|[Figure 1080]<br><br>[Figure 1081]|
|---|

|[Figure 1082]<br><br>[Figure 1083]|
|---|

|[Figure 1084]<br><br>[Figure 1085]|
|---|

Text2VideoZero

|[Figure 1086]<br><br>[Figure 1087]|
|---|

|[Figure 1088]<br><br>[Figure 1089]|
|---|

|[Figure 1090]<br><br>[Figure 1091]|
|---|

|[Figure 1092]<br><br>[Figure 1093]|
|---|

|[Figure 1094]<br><br>[Figure 1095]|
|---|

|[Figure 1096]<br><br>[Figure 1097]|
|---|

|[Figure 1098]<br><br>[Figure 1099]|
|---|

FateZero

|[Figure 1100]<br><br>[Figure 1101]|
|---|

|[Figure 1102]<br><br>[Figure 1103]|
|---|

|[Figure 1104]<br><br>[Figure 1105]|
|---|

|[Figure 1106]<br><br>[Figure 1107]|
|---|

|[Figure 1108]<br><br>[Figure 1109]|
|---|

|[Figure 1110]<br><br>[Figure 1111]|
|---|

|[Figure 1112]<br><br>[Figure 1113]|
|---|

Pix2Video

|[Figure 1114]<br><br>[Figure 1115]|
|---|

|[Figure 1116]<br><br>[Figure 1117]|
|---|

|[Figure 1118]<br><br>[Figure 1119]|
|---|

|[Figure 1120]<br><br>[Figure 1121]|
|---|

|[Figure 1122]<br><br>[Figure 1123]|
|---|

|[Figure 1124]<br><br>[Figure 1125]|
|---|

|[Figure 1126]<br><br>[Figure 1127]|
|---|

ControlVideo

|[Figure 1128]<br><br>[Figure 1129]|
|---|

|[Figure 1130]<br><br>[Figure 1131]|
|---|

|[Figure 1132]<br><br>[Figure 1133]|
|---|

|[Figure 1134]<br><br>[Figure 1135]|
|---|

|[Figure 1136]<br><br>[Figure 1137]|
|---|

|[Figure 1138]<br><br>[Figure 1139]|
|---|

|[Figure 1140]<br><br>[Figure 1141]|
|---|

Rerender A

Video

|[Figure 1142]<br><br>[Figure 1143]|
|---|

|[Figure 1144]<br><br>[Figure 1145]|
|---|

|[Figure 1146]<br><br>[Figure 1147]|
|---|

|[Figure 1148]<br><br>[Figure 1149]|
|---|

|[Figure 1150]<br><br>[Figure 1151]|
|---|

|[Figure 1152]<br><br>[Figure 1153]|
|---|

|[Figure 1154]<br><br>[Figure 1155]|
|---|

TokenFlow

|[Figure 1156]<br><br>[Figure 1157]|
|---|

|[Figure 1158]<br><br>[Figure 1159]|
|---|

|[Figure 1160]<br><br>[Figure 1161]|
|---|

|[Figure 1162]<br><br>[Figure 1163]|
|---|

|[Figure 1164]<br><br>[Figure 1165]|
|---|

|[Figure 1166]<br><br>[Figure 1167]|
|---|

|[Figure 1168]<br><br>[Figure 1169]|
|---|

CCEdit (Ours)

- Figure 19. Qualitative comparison of different methods. The target prompt is “A rider on a horse jumping over an obstacle in an equestrian competition, rendered in Van Gogh style with swirling skies and vibrant colors.”

