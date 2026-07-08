## VideoLCM: Video Latent Consistency Model

# arXiv:2312.09109v1[cs.CV]14Dec2023

Xiang Wang1∗ Shiwei Zhang2† Han Zhang3 Yu Liu2 Yingya Zhang2 Changxin Gao1 Nong Sang1† 1Key Laboratory of Image Processing and Intelligent Control, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology 2Alibaba Group 3Shanghai Jiao Tong University

{wxiang,cgao,nsang}@hust.edu.cn, {zhangjin.zsw,ly103369,yingya.zyy}@alibaba-inc.com, hzhang9617@gmail.com

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|
|---|---|---|---|---|---|

###### Text-to-videogeneration

 Young man with dwarfism recording a vlog at home 

|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|---|---|---|---|---|---|

 Beef burger in close up 

|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|
|---|---|---|---|---|---|

 Beautiful woman playing with her hair 

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

|[Figure 25]|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|---|---|---|---|---|---|

 The Close up of Smiling Woman Winking Eyes, White Background 

###### Compositionalvideosynthesis

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|[Figure 37]| |[Figure 38]| |[Figure 39]| |[Figure 40]| |[Figure 41]| |[Figure 42]| |
|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 43]| |[Figure 44]| |[Figure 45]| |[Figure 46]| |[Figure 47]| |[Figure 48]| |

 Monarch butterfly drinking nectar 

|[Figure 49]|[Figure 50]|[Figure 51]|[Figure 52]|[Figure 53]|[Figure 54]|
|---|---|---|---|---|---|

 Fruitfully young red grapes hanging on vineyard, hanging on a bush in a

beautiful sunny day

[Figure 55]

 First frame 

|[Figure 56]|[Figure 57]|[Figure 58]|[Figure 59]|[Figure 60]|[Figure 61]|
|---|---|---|---|---|---|

 woman lying and peacefully sleeping on sofa 

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

Figure 1. Video examples synthesized by the proposed VideoLCM with 4 inference steps. VideoLCM is a plug-and-play technique and can be integrated into text-to-video generation and compositional video synthesis paradigms.

### Abstract

Consistency models have demonstrated powerful capability in efficient image generation and allowed synthesis within a few sampling steps, alleviating the high computational cost in diffusion models. However, the consistency

∗ Intern at Alibaba Group. † Corresponding authors.

model in the more challenging and resource-consuming video generation is still less explored. In this report, we present the VideoLCM framework to fill this gap, which leverages the concept of consistency models from image generation to efficiently synthesize videos with minimal steps while maintaining high quality. VideoLCM builds upon existing latent video diffusion models and in-

corporates consistency distillation techniques for training the latent consistency model. Experimental results reveal the effectiveness of our VideoLCM in terms of computational efficiency, fidelity and temporal consistency. Notably, VideoLCM achieves high-fidelity and smooth video synthesis with only four sampling steps, showcasing the potential for real-time synthesis. We hope that VideoLCM can serve as a simple yet effective baseline for subsequent research. The source code and models will be publicly available.

### 1. Introduction

Recently, the field of video generation has witnessed tremendous advancements in synthesizing photo-realistic and temporally coherent video content, especially with the development of diffusion models [4, 6, 14, 17, 38, 44, 46, 47, 54, 58, 60]. Traditional diffusion-based methods such as videoLDM [5], Make-A-Video [38] and ModelScopeT2V [38], have achieved significant performance by incorporating additional temporal layers into existing image diffusion models [30, 31] to handle the temporal dynamics in videos. Nevertheless, these diffusion-based approaches inevitably require substantial sampling steps to synthesize videos during inference, e.g., 50-step DDIM sampling [40]. This limitation hinders the efficient and rapid synthesis of high-quality videos.

To address the challenge of high sampling cost in diffusion models, the concept of consistency models has been introduced in image generation [23, 24, 41, 53], achieving remarkable progress by enabling efficient image synthesis with a minimal number of steps (e.g., 4 steps vs. 50 steps). Despite its success, the application of the consistency model in the domain of video synthesis still remains largely unexplored.

To fill this research gap, we propose the VideoLCM framework. Our method builds upon existing latent diffusion models in video generation and leverages the idea of consistency distillation to train a video latent consistency model. By incorporating the VideoLCM framework, we aim to alleviate the need for extensive sampling steps while maintaining high-quality video synthesis. The quantitative and qualitative results demonstrate the effectiveness of our approach. Remarkably, our method achieves high-fidelity video synthesis with only 4∼6 sampling steps, showcasing its potential for fast and real-time synthesis. In comparison, previous methods such as ModelScopeT2V [44] and VideoLDM [5] typically require approximately 50 steps based on the DDIM solver to achieve similarly satisfactory results. In addition to text-to-video generation, we further extend the consistency model to compositional video synthesis. Experimental results indicate that in compositional video synthesis tasks, such as compositional depth-to-video synthesis, VideoLCM can produce visually satisfactory and

temporally continuous videos with even fewer steps, such as just 1 step.

In summary, the proposed VideoLCM bridges the gap between diffusion models and consistency models in video generation, enabling efficient synthesis of high-quality videos. By exploring the potential of consistency models in video generation, we aim to contribute to the field of fast video synthesis and provide a simplified and effective baseline for future research.

### 2. Related Work

The relevant fields related to this work include text-toimage generation, consistency model, and video generation. Next, we provide a brief review of these fields.

Text-to-image generation. In recent years, significant progress has been made in image generation with the development of generative models [7, 10, 13, 15, 19, 22, 34, 37], especially with the emergence of diffusion models [20, 27, 29, 31–33]. Typical methods for text-to-image generation, such as DALLE-2 [30], propose a two-stage approach where the input text is first converted into image embeddings using a prior model, followed by the generation of images based on these embeddings. Stable Diffusion [31] introduces a VAE-based approach in the latent space to decrease computational demand and optimizes the model with large-scale datasets [36]. Subsequent methods like ControlNet [57] and Composer [18] have incorporated additional conditional inputs, such as depth maps or sketches, for spatially controllable image synthesis.

Consistency model. To alleviate the limitation of requiring a large number of inference steps in diffusion models, the consistency model [41] has been developed. Built upon the probability flow ordinary differential equation (PF-ODE) , consistency models learn to map any point at any time step to the starting of the trajectory, i.e., the original clean image. The consistency model facilitates efficient one-step image generation without sacrificing the advantages of multi-step iterative sampling, thereby enabling more high-quality results through multi-step inference. Building upon the foundation of the consistency model, LCM [23] explores consistency models in the latent space to save memory consumption and improve inference efficiency. Subsequently, several methods [24, 35, 53] have also investigated efficient generation techniques and achieved impressive results. Inspired by the efficiency of the consistency model and its significant success in image generation, we extend the application of the consistency model to the domain of video generation.

Video generation. The pursuit of visually appealing and temporally coherent synthesis is crucial in video generation. Early methods [3, 39, 42, 48–50] primarily relied on generative adversarial networks (GANs), resulting in

| | |[Figure 74]| |[Figure 75]<br><br>[Figure 76]| |
|---|---|---|---|---|---|
| | | | | | |

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

🔥 Tunable

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Frozen

𝑥 𝑥 𝑥

[Figure 85]

[Figure 86]

[Figure 87]

🔥

Forward diffusion

DDIM step

EMA

𝜽 𝜽﹡

﹡

Student

###### Teacher

###### Student

(consistency model)

(diffusion model)

(consistency model)

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

𝑥

Loss

(Source video)

𝑥 𝑥 ∗

𝑥

- Figure 2. The overall pipeline of the proposed VideoLCM. Given a source video x0, a forward diffusion operation is first performed to add noise to the video. Then, the noised xn+k is entered into the student and teacher model to predict videos. xˆn is estimated by the teacher model and fed into the EMA student model. To learn self-consistency, a loss is imposed to constrain the output of the two student models to be consistent. The whole consistency distillation is conducted in the latent space, and conditional guidance is omitted for ease of presentation. The teacher model is a video diffusion model, and the student model shares the same network structure as the teacher model and is initialized with the parameters of the teacher model.

#### 3.1. Preliminaries

poor video quality and limited generalization to unseen domains [5]. With the rise of diffusion models [31], which provide stable training and impressive results, recent approaches [1, 4, 6, 12, 14, 16, 21, 25, 26, 38, 44, 45, 51, 52, 55, 56, 58, 61] have started exploring diffusion-based video generation. To generate temporally consistent videos, mainstream text-to-video methods such as ModelScopeT2V [44] and VideoLDM [5] achieve long-term temporal modeling by inserting temporal modules into a 2D UNet and training on large-scale datasets [2]. There are also some methods [26, 38, 43, 58, 61] focusing on using super-resolution and frame interpolation models to generate more realistic videos. To achieve spatiotemporal controllability, controllable video synthesis methods [8, 9, 11, 28, 46, 59, 60] have been proposed. In particular, VideoComposer [46] presents a compositional video synthesis paradigm that enables flexible control over the generated videos through textual, spatial, and temporal conditions. Despite significant advancements, these methods rely heavily on extensive iterative denoising processes to obtain satisfactory results, posing challenges for fast video generation. To address this issue, in this work, we propose the VideoLCM framework based on consistency models for fast and efficient video synthesis.

To achieve fast image generation, song et al. [41] brings into the conception of the consistency model, which aims to optimize a model that learns to map any point at any time step to the starting of the PF-ODE trajectory. Formally, the self-consistency property can be formulated as:

fθ(xt,t) = fθ(xt′,t′),∀t,t′ ∈ [ϵ,T] (1)

where ϵ is a time step, T is the overall denoising step, and xt denotes the noised input.

To accelerate the training and extract the strong prior knowledge of the established diffusion model [31], consistency distillation is usually adopted:

L(θ,θ∗;Φ) = E[d(fθ(xt

,tn))] (2)

,tn+1)),fθ∗(ˆxt

n+1

n

where Φ means the applied ODE solver and the model parameter θ∗ is obtained from the exponential moving average (EMA) of θ. xˆt

: xˆt

is the estimation of xt

n

n

,tn+1) (3)

n ← xt

+ (tn − tn−1)Φ(xt

n+1

n+1

LCM [23] conducts the above consistency optimization in the latent space and applies classifier-free guidance [15] in Eq. (3) to inject control signals, such as textual prompts. For more details, please refer to the original works [23, 41].

### 3. Method

The proposed VideoLCM builds upon the foundations of latent consistency models. We first briefly describe the preliminaries about latent consistency models. Then, we will present the details of the proposed VideoLCM. The overall structure of VideoLCM is displayed in Fig. 2.

#### 3.2. VideoLCM

Following LCM, the proposed VideoLCM is also established in the latent space to reduce the computational burden. To leverage the powerful knowledge within large-scale

pretrained video diffusion models and speed up the training process, we apply the consistency distillation strategy to optimize our model. Note that the pretrained video diffusion model can be the text-to-video generation model (e.g., ModelScopeT2V [44]) or the compositional video synthesis model (e.g., VideoComposer [46]).

In VideoLCM, we apply DDIM [40] as the basic ODE solver Ψ to estimate xˆt

: xˆt

n

,tn+1,tn,c) (4)

n ≈ xt

+ Ψ(xt

n+1

n+1

where c means the conditional inputs, which can be textual prompts in text-to-video generation or multiple combined signals in compositional video synthesis task.

Since classifier-free guidance is pivotal in synthesizing high-quality content [15], we also leverage classifier-free guidance in the consistency distillation stage and use a factor w to control the guidance scale:

n ≈xt

xˆt

,tn+1,tn,c) − wΨ(xt

+ (1 + w)Ψ(xt

(5)

n+1

n+1

,tn+1,tn,ϕ)

n+1

In LCM [23], w is variable and can be fed into the network for modulation, but this changes the structure of the initial network because a module encoding w needs to be added. In order to keep the initial parameters and design of the consistency model consistent with the teacher diffusion model, we train the consistency model with a fixed value of w, such as 9.0. Note that classifier-free guidance is only applied to the teacher diffusion model in training and is not required during the inference process of the consistency model.

VideoLCM is a plug-and-play technique compatible with text-to-video generation and compositional video synthesis. During the inference phrase, we can sample 4∼6 LCM steps to produce plausible results on text-to-video generation. For compositional video synthesis, take the compositional depth-to-video task as an example, 2∼4 steps are sufficient, and sometimes even 1 step.

### 4. Experiments

In this section, we first introduce the details of the experimental setup. Then, quantitative and qualitative comparisons will be represented to evaluate the effectiveness of the proposed VideoLCM framework.

#### 4.1. Experimental setup

Datasets. We train our video consistency model on two widely used datasets, i.e., WebVid10M [2] and LAION5B [36]. WebVid10M is a video-text dataset containing approximately 10.7M videos. To further enhance temporal diversity and improve visual quality, we additionally utilize about 1M internal video-text data to train VideoLCM. LAION-5B is an image-text dataset that is used to provide high-quality visual-text correspondence.

Table 1. Inference latency on text-to-video generation task. All experiments are performed on an NVIDIA A100 GPU. The inference overhead of generating eight videos at a time is reported.

|Method|Step Resolution Latency<br><br>|
|---|---|
|Baseline VideoLCM<br><br>|DDIM 50-step 16 × 256 × 256 60s LCM 4-step 16 × 256 × 256 10s<br><br>|
|Baseline VideoLCM<br><br>|DDIM 50-step 16 × 448 × 256 104s LCM 4-step 16 × 448 × 256 16s<br><br>|

Implementation details. For convenience, we directly leverage the existing pretrained video diffusion models in TF-T2V [47] as the teacher model and fix the network parameters in the consistency distillation process. TFT2V [47] is a technique that exploits text-free videos to scale up video diffusion models and can be applied to mainstream text-to-video generation and compositional video synthesis framework. The model structure of the consistency model is the same as the teacher diffusion model and is initialized with the teacher’s model parameters. AdamW optimizer with a learning rate of 1e-5 is adopted to train VideoLCM. The EMA rate used in our experiments is 0.95. We sample 16 frames and crop a 448 × 256 center region from each source video as model input. The training loss used in VideoLCM is a Huber loss by default. The entire network is trained on 4 NVIDIA A100 GPUs (one for image and three for video), requiring approximately 4k training iterations to produce relatively reliable video generation results.

#### 4.2. Time efficiency

We measure the inference time required for text-to-video synthesis using our proposed VideoLCM and compare it with the baseline method [47]. The comparative results are exhibited in Tab. 1. From the results, we can observe that since our method merely requires 4 LCM steps for inference, it is much faster than the baseline method with 50 DDIM steps. It should be noted that in addition to iterative denoising, the inference cost also includes text feature encoding, latent code decoding, etc. In addition, we notice that VideoLCM saves more time on high-resolution generation compared to baseline. For example, generating a video of 16 × 256 × 256 size saves 50 seconds (i.e., 60s − 10s), while 16x448x256 saves 88 seconds. The above comparison demonstrates the high efficiency of our approach.

#### 4.3. Ablation study on inference steps

In Fig. 3, we present the experimental visualization of varying inference steps in the text-to-video task. The results indicate that when the sampling step is too small, such as step = 1, the generated videos suffer from blurriness, with many details being inaccurately represented, and the temporal structure of objects cannot be preserved. As the number of iteration steps increases, the visual quality gradually

 Red apple on a paper bag 

|[Figure 100]|[Figure 101]|[Figure 102]|
|---|---|---|

|[Figure 103]|[Figure 104]|[Figure 105]|
|---|---|---|

DDIM steps = 50 (Baseline)

|[Figure 106]|[Figure 107]|[Figure 108]|
|---|---|---|

- LCM steps = 1

 Smiling successful bearded arabian businessman in suit enjoying phone chat 

|[Figure 109]|[Figure 110]|[Figure 111]|
|---|---|---|

|[Figure 112]|[Figure 113]|[Figure 114]|
|---|---|---|

- LCM steps = 2

|[Figure 115]|[Figure 116]|[Figure 117]|
|---|---|---|

|[Figure 118]|[Figure 119]|[Figure 120]|
|---|---|---|

|[Figure 121]|[Figure 122]|[Figure 123]|
|---|---|---|

LCM steps = 4

|[Figure 124]|[Figure 125]|[Figure 126]|
|---|---|---|

|[Figure 127]|[Figure 128]|[Figure 129]|
|---|---|---|

LCM steps = 6

|[Figure 130]|[Figure 131]|[Figure 132]|
|---|---|---|

|[Figure 133]|[Figure 134]|[Figure 135]|
|---|---|---|

LCM steps = 8

- Figure 3. Ablation study on text-to-video task under varying steps. Larger steps generally yield better visual quality and time continuity.

 instant noodle spicy salad with pork on plate - Asian food style 

- LCM steps = 1

 The Close up of Smiling Woman Winking Eyes, White Background 

- LCM steps = 2

DDIM steps = 50 (Baseline)

LCM steps = 4

LCM steps = 6

LCM steps = 8

|[Figure 136]|[Figure 137]|[Figure 138]|
|---|---|---|

|[Figure 139]|[Figure 140]|[Figure 141]|
|---|---|---|

|[Figure 142]|[Figure 143]|[Figure 144]|
|---|---|---|

|[Figure 145]|[Figure 146]|[Figure 147]|
|---|---|---|

|[Figure 148]|[Figure 149]|[Figure 150]|
|---|---|---|

|[Figure 151]|[Figure 152]|[Figure 153]|
|---|---|---|

|[Figure 154]|[Figure 155]|[Figure 156]|
|---|---|---|

|[Figure 157]|[Figure 158]|[Figure 159]|
|---|---|---|

|[Figure 160]|[Figure 161]|[Figure 162]|
|---|---|---|

|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Depth sequence

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

- Figure 4. Ablation study on compositional depth-to-video synthesis task under different inference steps. Since the additional depth sequence can provide prior guidance about structure and temporal, our method can produce plausible results with fewer steps or even only one step.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

 Aerial view of Brown castle in Portofino 

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

 Happy european woman doctor with glasses in medical mask wearing white medical coat and stethoscope 

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

 Colorful Macaroons are rotating  top view  

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

 Positive black guy crossing fingers, hopes for fortune 

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

 Rocks Splash Ocean Close Up   HD  

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

 For tangerines in colander falling water drops. 

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

 Happy woman gives green seedling to little daughter at home 

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

 girl paints eyes with mascara 

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

 Pretty Girl White Bikini Walking Beside Ocean 

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

 Winter Carpathian landscape, Christmas trees in the snow 

- Figure 5. Qualitative visualization results on text-to-video generation task. Videos are synthesized by performing 4 denoising steps.

improves, and the temporal structure can be better maintained. For example, when using 4∼6 steps, comparable results to DDIM’s 50 steps can be achieved, while significantly reducing the sampling steps and improving the generation speed. Moreover, we additionally perform an ablation study on compositional depth-to-video synthesis. As illustrated in Fig. 4, we observe that with only a few steps, such as 1 step, the generated results can display good visual quality. With an increase in the step size, many details became even more apparent. We attribute this phenomenon to the fact that compared to text-to-video generation, compositional depth-to-video synthesis can leverage additional structured control information, which reduces the difficulty of predicting videos from pure noise and enables achieving high-quality results with fewer steps. Empirically, we find

that 2∼4 steps can produce relatively stable compositional synthesis contents. The results on both text-to-video and compositional video generation demonstrate the effectiveness of the proposed VideoLCM, achieving a good balance between quality and speed.

#### 4.4. More qualitative results

To comprehensively showcase the capabilities of our method in handling various video generation tasks, we present additional visualization examples in Fig. 5. These results are generated with 4 LCM steps. From the visualization results, it is evident that our method achieves impressive generation outcomes in terms of visual quality and temporal continuity.

In Figs. 6 to 8, we present more results of compositional

|[Figure 238]|[Figure 239]|[Figure 240]|[Figure 241]|[Figure 242]|[Figure 243]|
|---|---|---|---|---|---|

 Portrait of infant boy looking to camera 

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

|[Figure 250]|[Figure 251]|[Figure 252]|[Figure 253]|[Figure 254]|[Figure 255]|
|---|---|---|---|---|---|

 Timelapse sunset floating mosque on the sea at Bagan Luar 

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

|[Figure 262]|[Figure 263]|[Figure 264]|[Figure 265]|[Figure 266]|[Figure 267]|
|---|---|---|---|---|---|

 Young handsome bearded Persian businessman against green background 

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

|[Figure 274]|[Figure 275]|[Figure 276]|[Figure 277]|[Figure 278]|[Figure 279]|
|---|---|---|---|---|---|

 Blue h2 hydrogen fuel tank renewable Ecological future Ecology concept grid energy storage 4k 

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

|[Figure 286]|[Figure 287]|[Figure 288]|[Figure 289]|[Figure 290]|[Figure 291]|
|---|---|---|---|---|---|

 A big tusker elephant walking across camera 

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

|[Figure 298]|[Figure 299]|[Figure 300]|[Figure 301]|[Figure 302]|[Figure 303]|
|---|---|---|---|---|---|

 Tall black athlete carrying basket ball in hands, thinking about game strategy 

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

|[Figure 310]|[Figure 311]|[Figure 312]|[Figure 313]|[Figure 314]|[Figure 315]|
|---|---|---|---|---|---|

 Man pushing trolley full of goods in diy warehouse, home improvement project 

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

|[Figure 322]|[Figure 323]|[Figure 324]|[Figure 325]|[Figure 326]|[Figure 327]|
|---|---|---|---|---|---|

 Rice sprinkled with bonito 

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

- Figure 6. Qualitative results on compositional depth-to-video synthesis task. Videos are synthesized by performing 4 denoising steps.

video generation using 4 sampling steps, including compositional depth-to-video synthesis (Fig. 6), compositional sketch-to-video synthesis (Fig. 7), and compositional video inpainting (Fig. 8). These qualitative results demonstrate stable generation quality and highlight the controllability of our method in accordance with the input conditioning. Our method can be applied to multiple mainstream video generation tasks mentioned above, revealing the generality of the VideoLCM framework and the vast potential for other future applications.

#### 4.5. Limitations

In our VideoLCM, we explore the application of the consistency model in video generation tasks, including text-to-video generation and compositional video synthesis. However, there are certain limitations: 1) Our method relies on a strong teacher model as the distillation target. 2) The consistency distillation process requires finetuning the model. While consistency distillation only requires a small number of training steps, it may lead to unsatisfactory results when the training data for the teacher model is unavail-

|[Figure 334]|[Figure 335]|[Figure 336]|[Figure 337]|[Figure 338]|[Figure 339]|
|---|---|---|---|---|---|

 Asian man stands confidently and inspired with his arms outstretched in rain background cloudy sky and lightning flashes 

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

|[Figure 346]|[Figure 347]|[Figure 348]|[Figure 349]|[Figure 350]|[Figure 351]|
|---|---|---|---|---|---|

 A girl is standing in smoke holding a mask in her hand. White smoke. 

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

|[Figure 358]|[Figure 359]|[Figure 360]|[Figure 361]|[Figure 362]|[Figure 363]|
|---|---|---|---|---|---|

 A big tusker elephant walking across camera 

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

|[Figure 370]|[Figure 371]|[Figure 372]|[Figure 373]|[Figure 374]|[Figure 375]|
|---|---|---|---|---|---|

 MS, Gentoo Penguin  Pygoscelis papua  walking on snow, Antarctica 

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

|[Figure 382]|[Figure 383]|[Figure 384]|[Figure 385]|[Figure 386]|[Figure 387]|
|---|---|---|---|---|---|

 Man wearing protective suit showing palms with inscription 

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

|[Figure 394]|[Figure 395]|[Figure 396]|[Figure 397]|[Figure 398]|[Figure 399]|
|---|---|---|---|---|---|

 Pleased brunette woman talking by smartphone and looking around 

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

|[Figure 406]|[Figure 407]|[Figure 408]|[Figure 409]|[Figure 410]|[Figure 411]|
|---|---|---|---|---|---|

 beauty mixed race bride in studio 

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

|[Figure 418]|[Figure 419]|[Figure 420]|[Figure 421]|[Figure 422]|[Figure 423]|
|---|---|---|---|---|---|

 View of ancient City wall at night, xian, shaanxi, China 

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

- Figure 7. Qualitative results on compositional sketch-to-video synthesis task. Videos are synthesized by performing 4 denoising steps.

able or from different domains. 3) Even though our method reduces the inference steps to 4∼6, real-time video generation, like image generation, is still not achieved. Exploring more stable and efficient video generation algorithms while ensuring the high quality of the generated video content is a promising future direction.

### 5. Conclusion

In this work, we propose the VideoLCM framework that extents latent consistency models to the video gener-

ation field. Our approach leverages existing latent video diffusion models and employs the consistency distillation technique to enable efficient and fast video synthesis. Experimental results demonstrate the effectiveness of our approach, with high-fidelity video synthesis achieved in just four steps, showcasing its real-time synthesis potential compared to prior methods requiring approximately 50 DDIM steps. We hope that VideoLCM can serve as a simple yet effective baseline for subsequent research work.

Acknowledgements. This work is supported by the Na-

|[Figure 430]| |[Figure 431]| |[Figure 432]| | |[Figure 433]| |[Figure 434]| |[Figure 435]| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 436]| |[Figure 437]| | |[Figure 438]| |[Figure 439]| |[Figure 440]| | |[Figure 441]| |

 Aucanada Lighthouse and beach in Majorca, Balearic Islands - Spain 

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

 Portrait of three millennial frenemies standing on the midway at a carnival 

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

|[Figure 454]| |[Figure 455]| |[Figure 456]| |[Figure 457]| |[Figure 458]| |[Figure 459]| |
|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 460]| |[Figure 461]| |[Figure 462]| |[Figure 463]| |[Figure 464]| |[Figure 465]| |

 Delicate white and pink flowers of the apple tree in the wind in the rays of the warm sun 

|[Figure 466]| |[Figure 467]| |[Figure 468]| |[Figure 469]| |[Figure 470]| |[Figure 471]| |
|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 472]| |[Figure 473]| |[Figure 474]| |[Figure 475]| |[Figure 476]| |[Figure 477]| |

 Excited older best employee get appreciated by business team 

|[Figure 478]| |[Figure 479]| |[Figure 480]| |[Figure 481]| | |[Figure 482]| |[Figure 483]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 484]| |[Figure 485]| |[Figure 486]| |[Figure 487]| |[Figure 488]| | |[Figure 489]| |

 Langkawi, Malaysia 

|[Figure 490]| |[Figure 491]| |[Figure 492]| | |[Figure 493]| |[Figure 494]| | |[Figure 495]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 496]| |[Figure 497]| |[Figure 498]| |[Figure 499]| | | |[Figure 500]| |[Figure 501]| |

 Man wearing protective suit showing palms with inscription 

|[Figure 502]| |[Figure 503]| | |[Figure 504]| |[Figure 505]| |[Figure 506]| | |[Figure 507]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 508]| | |[Figure 509]| |[Figure 510]| |[Figure 511]| | |[Figure 512]| |[Figure 513]| |

 Old church in England 

Figure 8. Qualitative visualizations on compositional video inpainting task. Videos are synthesized by performing 4 denoising steps.

tional Natural Science Foundation of China under grant U22B2053 and Alibaba Group through Alibaba Research Intern Program.

### References

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 3
- [2] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, pages 1728–1738, 2021. 3, 4
- [3] Yogesh Balaji, Martin Renqiang Min, Bing Bai, Rama Chellappa, and Hans Peter Graf. Conditional GAN with discrim-

- inative filter generation for text-to-video synthesis. In IJCAI, page 2, 2019. 2
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, pages 22563–22575, 2023. 2, 3
- [6] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. In ICCV, pages 23040–23050, 2023. 2, 3

- [7] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Textto-image generation via masked generative Transformers. arXiv preprint arXiv:2301.00704, 2023. 2
- [8] Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, TsungYi Lin, and Ming-Hsuan Yang. Motion-conditioned diffusion model for controllable video synthesis. arXiv preprint arXiv:2304.14404, 2023. 3
- [9] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840, 2023. 3
- [10] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 2
- [11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In ICCV, pages 7346–7356, 2023. 3
- [12] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In ICCV, pages 22930–22941, 2023. 3
- [13] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 27,

2014. 2

- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3
- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2, 3, 4
- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [17] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2023. 2
- [18] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. ICML, 2023. 2
- [19] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up GANs for text-to-image synthesis. In CVPR, pages 10124– 10134, 2023. 2
- [20] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, pages 6007–6017, 2023. 2
- [21] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant

- Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 3
- [22] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. In ICML, 2023. 2
- [23] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2, 3, 4
- [24] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023. 2
- [25] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In CVPR, pages 10209– 10218, 2023. 3
- [26] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 3
- [27] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2
- [28] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models. In CVPR, pages 18444–18455, 2023. 3
- [29] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, pages 16784–16804. PMLR, 2022. 2
- [30] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 2, 3
- [32] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023.
- [33] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022. 2
- [34] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 2

- [35] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 2
- [36] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35:25278– 25294, 2022. 2, 4
- [37] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in GANs. In CVPR, pages 1532–1540, 2021. 2
- [38] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. ICLR, 2023. 2, 3
- [39] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. StyleGAN-v: A continuous video generator with the price, image quality and perks of StyleGAN2. In CVPR, pages 3626–3636, 2022. 2
- [40] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 2, 4
- [41] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 2, 3
- [42] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. MocoGAN: Decomposing motion and content for video generation. In CVPR, pages 1526–1535, 2018. 2
- [43] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 3
- [44] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2, 3, 4
- [45] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint arXiv:2305.10874, 2023. 3
- [46] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. NeurIPS, 2023. 2, 3, 4
- [47] Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, and Nong Sang. A recipe for scaling up text-to-video generation with text-free videos. arXiv, 2023. 2, 4
- [48] Yaohui Wang, Piotr Bilinski, Francois Bremond, and Antitza Dantcheva. G3an: Disentangling appearance and motion for video generation. In CVPR, pages 5264–5273, 2020. 2
- [49] Yaohui Wang, Piotr Bilinski, Francois Bremond, and Antitza Dantcheva. Imaginator: Conditional spatio-temporal GAN for video generation. In WACV, pages 1160–1169, 2020.
- [50] Yuhan Wang, Liming Jiang, and Chen Change Loy. Styleinv: A temporal style modulated inversion network for uncondi-

- tional video generation. In ICCV, pages 22851–22861, 2023. 2
- [51] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. arXiv preprint arXiv:2312.04433, 2023. 3
- [52] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, pages 7623–7633, 2023. 3
- [53] Jie Xiao, Kai Zhu, Han Zhang, Zhiheng Liu, Yujun Shen, Yu Liu, Xueyang Fu, and Zheng-Jun Zha. Ccm: Adding conditional controls to text-to-image consistency models. 2023. 2
- [54] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2
- [55] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In CVPR, pages 18456–18466, 2023. 3
- [56] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 3
- [57] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3836–3847, 2023. 2
- [58] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2, 3
- [59] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 3
- [60] Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098,

2023. 2, 3

- [61] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 3

