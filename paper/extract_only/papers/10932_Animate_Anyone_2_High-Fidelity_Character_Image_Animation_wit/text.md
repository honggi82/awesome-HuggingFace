## Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordance

Li Hu∗ Guangyuan Wang∗ Zhen Shen Xin Gao Dechao Meng Lian Zhuo Peng Zhang Bang Zhang Liefeng Bo Tongyi Lab, Alibaba Group

https://humanaigc.github.io/animate-anyone-2/

# arXiv:2502.06145v1[cs.CV]10Feb2025

|MoreResultsfromAnimateAnyone2<br><br>[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]|
|---|

AnimateAnyone

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Motion

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Driving Video

AnimateAnyone2

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|Motion<br><br>Environment|
|---|

Figure 1. We propose Animate Anyone 2, which differs from previous character image animation methods that solely utilize motion signals to animate characters. Our approach additionally extracts environmental representations from the driving video, thereby enabling character animation to exhibit environment affordance. The generated results demonstrate that, beyond maintaining character consistency, Animate Anyone 2 can produce high-fidelity results that seamlessly integrate characters with the surrounding environment.

### Abstract

object interactions, we leverage an object guider to extract features of interacting objects and employ spatial blending for feature injection. We also introduce a pose modulation strategy that enables the model to handle more diverse motion patterns. Experimental results demonstrate the superior performance of the proposed method.

Recent character image animation methods based on diffusion models, such as Animate Anyone, have made significant progress in generating consistent and generalizable character animations. However, these approaches fail to produce reasonable associations between characters and their environments. To address this limitation, we introduce Animate Anyone 2, aiming to animate characters with environment affordance. Beyond extracting motion signals from source video, we additionally capture environmental representations as conditional inputs. The environment is formulated as the region with the exclusion of characters and our model generates characters to populate these regions while maintaining coherence with the environmental context. We propose a shape-agnostic mask strategy that more effectively characterizes the relationship between character and environment. Furthermore, to enhance the fidelity of

### 1. Introduction

The objective of character image animation is to synthesize animated video sequences utilizing a reference character image and a sequence of motion signals. Recent developments predominantly adopt diffusion-based frameworks [7, 15, 17, 42, 44, 48, 56, 60], achieving notable enhancements in appearance consistency, motion stability and character generalizability. These advancements exhibit substantial potential in areas such as filmmaking, advertising,

∗Equal contribution

and virtual character applications.

In recent cross-identity animation workflows, motion signals are typically extracted from disparate videos, while the character’s contextual environments are derived from static images. This setting introduces critical limitations: the spatial relationships between animated characters and their environments often lack authenticity, and intrinsic human-object interactions are disrupted. Consequently, most existing methods are predominantly limited to animating simple actions (e.g., individual gestures or dances) without adequately capturing the complex spatial and interactive relationships between characters and their surroundings. These limitations significantly hinder the advancement of character animation techniques.

Recent attempts to integrate character animation with scenes and objects, while promising, face significant challenges in generation quality and adaptability. For instance, MovieCharacter[28] synthesizes character videos by cascading the outputs from multiple algorithms, which introduces noticeable artifacts and unnatural visual discontinuities. AnchorCrafter[47] primarily focuses on human-object manipulation animation, with relatively simplistic character motion and object appearance. MIMO[25] addresses this challenge by composing characters, pre-processed backgrounds and occlusions, which are disentangled via depth. Such formulation for defining the relationship between characters and environments is suboptimal, limiting the ability to handle complex interactions.

In this paper, we propose to expand the scope of character animation by introducing Character Image Animation with Environment Affordance. Specifically, we define the research problem as follows: given a character image and a source video, the generated character animation should: 1) inherit character motion desired by the source video. 2) accurately demonstrate character-environment relationship consistent with the source video. This setting introduces novel challenges for character animation, as it requires that the model should effectively handle diverse and complex character motions, while ensuring precise interaction between characters and their environments throughout the animation process.

To achieve this, we introduce a novel framework Animate Anyone 2. As illustrated in Fig.1, unlike previous character animation methods that solely utilize motion signals, we additionally capture environmental representations from the source video as conditional inputs, which enables the model to learn the intrinsic relationship between character and environment in an end-to-end manner. We formulate the environment by removing the character regions and our model generates characters to populate these regions while maintaining coherence with the environmental context. We develop a shape-agnostic mask strategy that better represents the boundary relationship between character and their

contextual scenes, enabling effective learning for charactercontext integration while mitigating shape leakage issues. Second, to enhance the fidelity of object interactions, we introduce additional processing for interactive object regions. We design a lightweight object guider to extract interactive object features and propose a spatial blending mechanism to inject these features into the generation process. It facilitates the preservation of intricate interaction dynamics in the source video. Lastly, we propose depth-wise pose modulation approach for character motion modeling, empowering the model to handle more diverse and complex character poses with enhanced robustness.

The results in Fig.1 exhibit both high-quality character animation performance and remarkable environment affordance, manifested through three key advantages: 1) seamless scene integration; 2) coherent object interaction; and 3) robust handling of diverse and complex motions. Our approach is evaluated on corresponding benchmarks, achieving superior character animation results compared to existing methods. In summary, we highlight three key contributions of our paper.

- • We introduce Animate Anyone 2, a framework capable of animating character with environment affordance, achieving robust performance.
- • We propose a novel environment formulation and object injection strategy to achieve seamless characterenvironment integration.
- • We propose pose modulation strategy to enhance model robustness in challenging action scenarios.

### 2. Related Works

#### 2.1. Character Image Animation

Distinguished from GAN-based[1, 9, 18] approaches[6, 30, 34–36, 50, 54, 57], diffusion-based image animation methods[7, 15, 17, 20, 42, 44, 48, 49, 56, 60] have emerged as the current research mainstream. As the most representative approach, Animate Anyone[15] designs its framework based on Stable Diffusion[31], and the denoising network is structured as a 3D UNet[4, 12] for temporal modeling. It proposes ReferenceNet, a symmetric UNet[32] architecture, to preserve appearance consistency and employs pose guider to incorporate skeleton information as driving signals for stable motion control. The Animate Anyone framework achieves robust and generalizable character animation, from which we extensively drew inspiration.

Some works propose improvements upon foundational frameworks. MimicMotion[56] leverages pretrained image-to-video capabilities of Stable Video Diffusion[3], designing a PoseNet to inject skeleton information. UniAnimate[44] stacks reference images across temporal dimensions, utilizing mamba-based[10] temporal modeling techniques. Some works explore different motion con-

|[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>| |
|---|
<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>SourceVideo<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>|[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>|
|---|
<br><br>[Figure 40]<br><br>VideoRecon<br><br>Training<br><br>[Figure 41]<br><br>| |
|---|
<br><br>[Figure 42]<br><br>| |
|---|
<br><br>SpatialAttention<br><br>TemporalAttention<br><br>[Figure 43]<br><br>PoseModulation<br><br>E<br><br>E D<br><br>|[Figure 44]<br><br>[Figure 45]| |
|---|---|
| | |
<br><br>Inference<br><br>[Figure 46]<br><br>| |
|---|
<br><br>[Figure 47]<br><br>| |
|---|
<br><br>SpatialBlending<br><br>ObjectInjection<br><br>VideoGen<br><br>Training<br><br>Inference<br><br>DenoisingNet<br><br>ReferenceNet<br><br>Noise<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>|[Figure 52]|[Figure 53]|
|---|---|
| | |
<br><br>| |
|---|
|[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]|
| |
| |
<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>| |[Figure 80]|
|---|---|
| | |
<br><br>[Figure 81]<br><br>…<br><br>…<br><br>Environment Formulation<br><br>[Figure 82]<br><br>Framework<br><br>[Figure 83]<br><br>|E|
|---|
<br><br>VAEEncoder VAEDecoder<br><br>[Figure 84]<br><br>|D|
|---|
<br><br>SampleImage<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>ObjectGuider<br><br>[Figure 89]<br><br>E| |
|---|---|
|[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>Conv CrossAttn 2D<br><br>Conv 2D<br><br>Conv 3D<br><br>[Figure 94]<br><br>PoseModulation<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>Pose<br><br>Depth<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]|[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>AlphaBlending<br><br>[Figure 122]<br><br>[Figure 123]<br><br>SpatialBlending<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>ObjectLatents<br><br>NoiseLatents<br><br>C|

Figure 2. The framework of Animate Anyone 2. We capture environmental information from the source video. The environment is formulated as regions devoid of characters and incorporated as model input, enabling end-to-end learning of character-environment fusion. To preserve object interactions, we additionally inject features of objects interacting with the character. These object features are extracted by a lightweight object guider and merged into the denoising process via spatial blending. To handle more diverse motions, we propose a pose modulation approach to better represent the spatial relationships between body limbs.

trol signals. DisCo[42] and MagicAnimate[48] utilizes DensePose[11] as human body representations. Champ[60] employs the 3D parametric human model SMPL[24], integrating multi-modal information including depth, normal, and semantic signals derived from SMPL.

quence. AnchorCrafter[47], focusing on human-object interaction, first perceives HOI-appearances and injects HOImotion to generate anchor-style product promotion videos. MIMO[25] introduces spatial decomposed diffusion, decomposing videos into human, background and occlusion based on 3D depth and subsequently composing these elements to generate character video.

#### 2.2. Human-environment Affordance Generation

Numerous studies leverage diffusion models to generate human image or video that contextually integrate with scenes or interactive objects. Some studies[23, 26, 27, 33, 52] investigate inserting or inpainting human into given scenes to achieve scene affordance. [23] applies video self-supervised training to inpaint person into masked region with correct affordances. Text2Place[27] aims to place a person in background scenes by learning semantic masks using text guidance for localizing regions. InVi[33] achieves object insertion by first conducting image inpainting and subsequently generating frames using extendedattention mechanisms.

### 3. Method

In this section, we introduce Animate Anyone 2. In 3.1, we first elaborate on the overall framework. In 3.2, we delineate the strategy for environment formulation. In 3.3, we present the design of object injection. In 3.4, we provide a detailed exposition of pose modulation strategy.

#### 3.1. Framework

System Setting. The overall framework is illustrated in Fig.2. During training, we employ a self-supervised learning strategy. Given a reference video I1:N where N denotes the number of frames, we disentangle character and environment via a formulated mask (detailed in 3.2), ob-

Several works focus on character animation with scene or object interactions. MovieCharacter[28] composites the animated character results into person-removed video se-

|[Figure 128]<br><br>[Figure 129]|
|---|

|[Figure 130]<br><br>[Figure 131]|
|---|

taining separate character sequence Ic1:N and environment sequence Ie1:N. To facilitate more fidelity object interaction, we additionally extracted the sequence of objects Io1:N. Motion sequence Im1:N is extracted as driving signals. We randomly sample a character image Ic from Ic1:N with center crop and composite it onto a random background. Given image Ic, motion sequence Im1:N, environment sequence Ie1:N and object sequence Io1:N as inputs, our model reconstructs the reference video I1:N. During inference, given a target character image and a driving video, our method can animate the character with consistent actions and environmental relationship corresponding to the driving video.

###### or …

Figure 3. Different coefficients for mask formulation.

facilitate the model in learning character-environment integration more effectively, while preserving character shape consistency and environmental information integrity. Some studies[23, 33] leverage bounding boxes to represent generative regions. However, we observe artifacts or inconsistencies with the source video when dealing with complex scenes, due to insufficient conditioning. Conversely, directly using precise masks is also suboptimal, potentially introducing shape leakage. Due to the self-supervised training strategy, there exists strong correlation between character outlines and mask boundaries. Consequently, the model tends to use this information as additional guidance for animating character. However, during inference, when the target character differs from the source in body shape and clothing, the model may forcibly conform to the mask boundary, resulting in integration artifacts.

Diffusion Model. Our method is developed based on LDM[31]. It employs a pretrained VAE[21, 40] to transform images from pixel space to latent space: z=E(x). During training, random Gaussian noise ϵ is progressively added to image latents zt at different timesteps, The training objective can be formulated as follows:

t,c,ϵ,t(||ϵ − ϵθ(zt,c,t)||22) (1) where ϵθ represents the function of DenoisingNet. c represents conditional inputs. During inference, noise latents are iteratively denoised[13, 37] and reconstructed into images through the decoder of VAE: xrecon=D(z). The network design of DenoisingNet is derived from Stable Diffusion[31], inheriting its pretrained weights. We extend the original 2D UNet architecture to 3D UNet, incorporating the temporal layer design from AnimateDiff[12].

L = Ez

Shape-agnostic Mask. Therefore, we propose a shapeagnostic mask strategy for environment formulation, with the core idea of disrupting the correspondence between mask region and character outline during training. Specifically, for a character mask Mc in its bounding box of size h × w, we define two coefficients kh and kw. We divided the character mask Mc into kh×kw non-overlapping blocks, where kh ∈ (1,h),kw ∈ (1,w). We denote Pc(k) as the divided patches, where k is the index. We reformulate the mask Mc into a new mask Mf by propagating the patchwise maximum value:

Conditional Generation. We adopt the ReferenceNet architecture from [15] to extract appearance features of the character image Ic. In our framework, we simplify the computational complexity by merging these features exclusively in the midblock and upblock of the DenoisingNet decoder via spatial attention[41]. Besides, three conditional embeddings are extracted from the souce video: environment sequence Ie1:N, motion sequence Im1:N, and object sequence Io1:N. For environment sequence Ie1:N, we employ VAE encoder to encode the embedding and subsequently merge it with noise latents. For motion sequence Im1:N, we design pose modulation strategy (elaborated in 3.4) and the motion information is also merged into the noise latents. For object sequence Io1:N, after encoding via VAE encoder, we develop an object guider to extract multi-scale features and inject them into the DenoisingNet through spatial blending, which will be detailed in 3.3.

Pc(k)(i,j) (2)

Mf (i,j) = max

(i,j)∈Pc(k)

where Pc(k)(i,j) represents the value at position (i,j). The visualized process is presented in Fig.3. By employing this strategy, the formulated mask dynamically generate different shapes that deviate from the character boundaries, thereby compelling the network to learn context integration more effectively, unencumbered by predefined boundary constraints. During inference, we set kh = h/10 and kw = w/10.

#### 3.2. Environment Formulation

Random Scale Augmentation. Moreover, since the formulated mask is inherently larger than the original mask, this introduces an inevitable bias that constrains the generated character to be necessarily smaller than the given mask. To mitigate this bias, we employ random scale augmentation on source videos. Specifically, we extract the character together with the interacting objects based on their masks

Motivation. In our framework, the environment is formulated as a region excluding characters. During training, the model generates characters to populate these regions while maintaining coherence with the environmental context. The boundary relationship between characters and the environment is crucial. Appropriate boundary guidance can

and apply a random scaling operation. Subsequently, we recompose these scaled content back into the source video. This approach ensures that the formulated mask has a probabilistic chance of being smaller than the actual character region. During inference, the model is capable of animating the character flexibly without being constrained by the size of the mask.

#### 3.3. Object Injection

Object Guider. The environment formulation strategy may potentially lead to distortion of object regions. To enhance the preservation of object interactions, we propose to inject additional object-level features. Interactive objects can be extracted through two methods: 1) Leveraging VLM[2, 43] to obtain object localization; 2) Interactively confirming object positions via manual annotation. Then we employ SAM2[22, 29] to extract object mask, obtaining corresponding object image and encode it into object latents via VAE encoder. A naive approach to merging object features is to directly concatenate scene and object features before feeding them into the network. However, due to the intricate relationship between characters and objects, such method struggles to handle complex human-object interactions, often falling short in capturing both human and object details. Thus we design an object guider to extract object-level features. Unlike character features that require complex modeling, objects inherently preserve visual characteristics from the source video. Thus we implement object guider using a lightweight fully convolutional architecture. specifically, object latents are downsampled four times via 3 × 3 Conv2D to obtain multi-scale features. The channel dimensions of these features are aligned with those in the midblock and upblock of the DenoisingNet, facilitating subsequent feature fusion.

Spatial Blending. To recover the spatial relationships of human-object interaction, we employ spatial blending to inject features extracted by object guider into the DenoisingNet. Specifically, during the denoising process, spatial blending layer is performed after spatial attention layer. For noise latents znoise and object latents zobject, we concatenate their features and compute the alpha weight α through a Conv2D-Sigmoid layer. The spatial blending process can be mathematically formulated as follows:

α = F(cat(znoise,zobject)) (3)

zblend = α · zobject + (1 − α) · znoise (4) where F denotes the Conv2D-Sigmoid layer, which is initialized through zero convolution. zblend denotes the new noise latents after spatial blending. In each stage of the DenoisingNet decoder, we alternately apply spatial attention on character features and spatial blending of object features, enabling the generation of high-fidelity results with excellent details of character-object interactions.

#### 3.4. Pose Modulation

Motivation. Animate Anyone[15] employs a skeleton representation to capture character motion and utilizes pose guider for feature modeling. However, the skeleton representation lacks explicit modeling of inter-limb spatial relationships and hierarchical dependencies. Some existing methods[25, 60] adopt 3D mesh representations like SMPL to represent human bodies, but this tends to compromise the generalizability across characters and potentially introduces shape leakage due to its dense representation.

Depth-wise Pose Modulation. We propose to retain the skeleton signals while augmenting it with structured depth to enhance the representation of inter-limb spatial relationships. We refer to this approach as depth-wise pose modulation. For motion signals, we leverage Sapien[19] to extract the skeleton and depth information from the source video. The depth information is structurally processed via the skeleton to mitigate potential shape leakage in raw depth maps. Specifically, we first binarize the skeleton image to obtain skeleton mask, and subsequently extract the depth results within this masked region. Then we employ Conv2D with the same architectural design as the pose guider[15] to process the skeleton map and structured depth map. Then we merge the structured depth information into the skeleton features through a cross-attention mechanism. The key insight behind this approach is to enable each limb to incorporate spatial characteristics from other limbs, thereby facilitating a more nuanced understanding of limb interaction relationships. Given that pose information extracted from wild videos may contain errors, we utilize Conv3D to model temporal motion information, enhancing inter-frame connections and mitigating the impact of erroneous signals on individual frames.

### 4. Experiments 4.1. Implementations

To validate the generalizability of our method across more diverse scenarios, we curated a dataset of 100,000 character videos collected from the internet, encompassing a broader range of scene types, action categories, and human-object interaction cases. Experiments are conducted on 8 NVIDIA A100 GPUs. The training involves 100k steps with batch size of 8 and the video length in a batch is 16. Video frames are cropped at consistent positions to ensure that the character is fully contained within the 16frame sequence. The reference image is randomly sampled from the entire video sequence. We perform center cropping and remove the original background, compositing it with a new random background. This approach enables the model to automatically recognize characters within the image during inference without requiring additional segmentation, thereby mitigating potential accuracy limitations in-

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

- Figure 4. Qualitative Results. Animate Anyone 2 achieves consistent character animation while enabling the integration and interaction between characters and their environments, thereby realizing environment affordance.

herent in segmentation processes.

During long video inference, the video is segmented into multiple video clips, and inference is performed on each clip sequentially. Inspired by the motion frame technique in [38], we utilize the final frame of the previous video clip as the temporal reference to guide the transition between clips. This strategy ensures smooth transitions between different video clips, preventing appearance texture discontinuities or blurriness.

#### 4.2. Qualitative Results

Fig. 4 demonstrates that our approach not only animates diverse characters with high-fidelity performance, but also achieves remarkably seamless visual integration and interaction with their surrounding environments. This substantiates the versatility and robustness of our method, underscoring its significant potential for widespread applications.

#### 4.3. Comparisons

Metrics. We follow the previous evaluation metrics for character image animation. Specifically, for single-frame quality assessment, we employ PSNR[14], SSIM[45], and LPIPS[55]. For video fidelity, we utilize the Frechet Video Distance (FVD)[39].

Evaluation on TikTok Dataset. We conduct experiments on the TikTok Benchmark[16]. In this dataset, the video

Method SSIM ↑ PSNR ↑ LPIPS ↓ FVD ↓ MRAA [36] 0.672 29.39 0.672 284.82

DisCo [42] 0.668 29.03 0.292 292.80 MagicAnimate [48] 0.714 29.16 0.239 179.07 Animate Anyone [15] 0.718 29.56 0.285 171.90

Champ* [60] 0.802 29.91 0.234 160.82 UniAnimate* [44] 0.811 30.77 0.231 148.06

Ours 0.778 29.82 0.248 158.97 Ours* 0.812 30.82 0.223 144.65

Table 1. Quantitative comparison on Tiktok benchmark. * means utilizing other video data for pretraining.

backgrounds are static. Existing character animation approaches typically synthesize target videos with both characters and backgrounds by a single reference image. To ensure a fair comparison, we adjust the configuration of our method: instead of using the ground truth background, we employ the background from the reference image as the environmental input. This modification allows all methods to generate outputs conditioned exclusively on a single reference image. We implement two training settings of our approach: 1) trained exclusively on the Tiktok training set, and 2) first trained on our custom dataset and subsequently finetuned on the Tiktok training set. As shown in Tab. 1, when trained solely on the Tiktok training set, our method outperforms Magicanimate[48] and Animate Anyone[15]. After

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

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

Source Video Baseline MIMO Ours

Figure 6. Qualitative comparion. Our method demonstrates superior environment integration and object interaction.

Reference Animate Anyone Champ MimicMotion Ours

- Figure 5. Qualitative comparion for character animation. We normalize the background to a uniform color.

Method SSIM ↑ PSNR ↑ LPIPS ↓ FVD ↓ Baseline 0.785 28.71 0.291 195.45

##### Ours 0.794 28.83 0.276 186.17

Method SSIM ↑ PSNR ↑ LPIPS ↓ FVD ↓ Animate Anyone[15] 0.761 28.41 0.324 228.53

Table 3. Quantitative comparison with baseline on our dataset. Baseline refers to the pseudo character-environment integration.

Champ[60] 0.771 28.69 0.294 205.79 MimicMotion[56] 0.767 28.52 0.307 212.48

###### Ours 0.809 29.24 0.259 172.54

further evaluate the performance of character-environment affordance on our proposed dataset. We construct a baseline algorithm by directly compositing character animation results onto the original video background, creating a pseudo character-environment integration, similar to MovieCharacter [28]. we leverage ProPainter[59] to inpaint the character region. Quantitative evaluation is presented in Tab. 3. We conduct qualitative comparison illustrated in Fig. 6. Our approach demonstrates superior performance in terms of enhanced character-environment integration. We also compare our method with MIMO[25], which is the most relevant method to our task setting. Due to the absence of public source code, we conduct a qualitative comparison focused on character-environment integration performance. The result of MIMO are obtained from its official ModelScope link∗. As illustrated in Fig. 6. From the first group of the visualization, it can be observed that due to MIMO’s reliance on additional pre-processing algorithms for background inpainting, it tends to leave noticeable preprocessing artifacts and establish erroneous relationships between the background and the animated characters. In contrast, our proposed approach effectively mitigates these issues, enabling superior scene and character integration. The second group further illustrates MIMO’s limitations in handling relatively complex human-object interaction scenarios, whereas our method demonstrates enhanced robustness in intricate scenes.

Table 2. Quantitative comparison on our dataset. Our approach demonstrates superior performance across generalized scenarios.

incorporating pre-trained video data, our approach further surpasses Champ[60] and UniAnimate[44], achieving stateof-the-art performance.

Evaluation on Proposed Dataset. Due to the limitations of existing benchmarks[16, 36, 53] that exhibit domain proximity, these datasets cannot effectively evaluate the generalizability of models across diverse scenarios. Following [60], we establish a testset comprising 100 character videos from real-world scenarios to conduct additional evaluation. Since other methods cannot generate dynamic environment, we standardize the background of input images to a uniform color, thus isolating the impact of environment variations on the evaluation. For fair comparison, we finetune these methods on our custom training dataset. The quantitative comparison is shown in Tab. 2. Qualitative comparison is shown in Fig.5. Our results significantly outperform alternative approaches, which can be attributed to two key factors: (1) our proposed motion modeling demonstrates robust generalization across diverse motion patterns, and (2) our decoupled environment and character generation strategy enables the model to focus more precisely on character dynamics, mitigating interference from environment variations.

Evaluation for character-environment affordance. We

∗https://modelscope.cn/studios/iic/MIMO

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Image Source Video Precise Mask Bounding Box Ours

- Figure 7. Ablation study of environment formulation.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Source Video w/o Spatial Blending Ours

| |
|---|

- Figure 8. Qualitative ablation of object modeling method.

#### 4.4. Ablation Study

Environment Formulation. To demonstrate the effectiveness of our proposed environment formulation strategy, we explore alternative designs, including: 1) utilizing precise character masks from the source video, and 2) employing bounding box regions. Qualitative results are shown in Fig. 7. Using accurate masks can constrain the animated character’s shape within the predefined mask boundaries, potentially causing appearance deformation and inconsistency. Conversely, adopting bounding box regions may introduce scene context distortions and fusion artifacts in the proximity of character. Our method demonstrates superior capability in learning flexible character generation and environmental completion, achieving both character consistency and seamless character-scene integration.

Object Modeling. We conduct a comparison of different object modeling approaches: directly merging object features with noise latents without employing spatial blending. Quantitative result is shown in Tab 4. We further demonstrate the visualization results. As shown in Fig. 8, in complex interaction scenarios, it fails to comprehensively preserve the intrinsic features of interactive objects, resulting in local distortions and consequently misinterpreting their interaction relationships. The second comparison reveals that the interactions between characters and objects exhibit an artificial stitching effect, which consequently compromises the naturalness of their interactive relationships.

Pose Modulation. We evaluate the effectiveness of our

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Source Video w/o Pose Modulation Ours

Figure 9. Qualitative ablation of pose modulation.

Method SSIM ↑ PSNR ↑ LPIPS ↓ FVD ↓

w/o Spatial Blending 0.789 28.74 0.283 191.23 w/o Pose Modulation 0.769 28.56 0.301 211.15

###### Ours 0.794 28.83 0.276 186.17

Table 4. Quantitative ablation study.

proposed pose modulation strategy. Quantitative result is presented in Tab 4. Qualitative result is shown in Fig 9. Without employing the pose modulation method, character limb relationships may suffer from misalignment and spatial inconsistencies. Consequently, the model’s capability to generate accurate and plausible character poses becomes severely constrained. In contrast, our proposed approach, by incorporating depth-aware information, can more effectively learn and capture the complex spatial relationships between limbs, enabling robust performance across diverse and challenging motion scenarios.

### 5. Discussion and Conclusion

Limitations. Our approach may introduce visual artifacts when dealing with complex hand-object interactions that occupy a relatively small pixel region. In intricate humanobject interactions, deformation artifacts may emerge when source and target characters exhibit substantial shape discrepancies. The performance of object interaction is also influenced by SAM’s segmentation capabilities.

Potential Impact. The proposed method may be used to produce fake videos of individuals, which can be detected using some anti-spoofing techniques[5, 8, 46, 51, 58].

Conclusion. In this paper, we introduce Animate Anyone 2, a novel framework that enables character animation to exhibit environment affordance. We extract environmental information from driving videos, enabling the animated character to preserve its original environment. We pro-

pose a novel environment formulation and object injection strategy, facilitating seamless character-environment integration. Moreover, we propose pose modulation that empowers the model to robustly handle diverse motion patterns. Experimental results demonstrate that Animate Anyone 2 achieves high-fidelity generation performance.

### References

- [1] Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein generative adversarial networks. In International conference on machine learning, pages 214–223. PMLR, 2017. 2
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 5
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [5] Zinelabidine Boulkenafet, Jukka Komulainen, and Abdenour Hadid. Face anti-spoofing based on color texture analysis. In 2015 IEEE international conference on image processing (ICIP), pages 2636–2640. IEEE, 2015. 8
- [6] Caroline Chan, Shiry Ginosar, Tinghui Zhou, and Alexei A Efros. Everybody dance now. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5933–5942, 2019. 2
- [7] Di Chang, Yichun Shi, Quankai Gao, Hongyi Xu, Jessica Fu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. In Forty-first International Conference on Machine Learning, 2023. 1, 2
- [8] Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu, Jianfu Zhang, Weiqiang Wang, et al. Demamba: Ai-generated video detection on million-scale genvideo benchmark. arXiv preprint arXiv:2405.19707, 2024. 8
- [9] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [10] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2
- [11] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7297–7306, 2018. 3

- [12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 4
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4
- [14] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010. 6
- [15] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 1, 2, 4, 5, 6, 7
- [16] Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12753– 12762, 2021. 6, 7
- [17] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dreampose: Fashion video synthesis with stable diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22680–22690, 2023. 1, 2
- [18] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [19] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. In European Conference on Computer Vision, pages 206–228. Springer, 2025. 5
- [20] Jeongho Kim, Min-Jung Kim, Junsoo Lee, and Jaegul Choo. Tcan: Animating human images with temporally consistent pose guidance using diffusion models. In European Conference on Computer Vision, pages 326–342. Springer, 2024. 2
- [21] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 1416, 2014, Conference Track Proceedings, 2014. 4
- [22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 5
- [23] Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A Efros, and Krishna Kumar Singh. Putting people in their place: Affordance-aware human insertion into scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17089–17099, 2023. 3, 4
- [24] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 3

- [25] Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. Mimo: Controllable character video synthesis with spatial decomposed modeling. arXiv preprint arXiv:2409.16160,

2024. 2, 3, 5, 7

- [26] Mirela Ostrek, Carol O’Sullivan, Michael J Black, and Justus Thies. Synthesizing environment-specific people in photographs. In European Conference on Computer Vision, pages 292–309. Springer, 2024. 3
- [27] Rishubh Parihar, Harsh Gupta, Sachidanand VS, and R Venkatesh Babu. Text2place: Affordance-aware text guided human placement. In European Conference on Computer Vision, pages 57–77, 2024. 3
- [28] Di Qiu, Zheng Chen, Rui Wang, Mingyuan Fan, Changqian Yu, Junshi Huan, and Xiang Wen. Moviecharacter: A tuningfree framework for controllable character video synthesis. arXiv preprint arXiv:2410.20974, 2024. 2, 3, 7
- [29] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 5
- [30] Yurui Ren, Ge Li, Shan Liu, and Thomas H Li. Deep spatial transformation for pose-guided person image generation and animation. IEEE Transactions on Image Processing, 29: 8622–8635, 2020. 2
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4
- [32] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 2
- [33] Nirat Saini, Navaneeth Bodla, Ashish Shrivastava, Avinash Ravichandran, Xiao Zhang, Abhinav Shrivastava, and Bharat Singh. Invi: Object insertion in videos using off-the-shelf diffusion models. arXiv preprint arXiv:2407.10958, 2024. 3, 4
- [34] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019. 2
- [35] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. Animating arbitrary objects via deep motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2377–2386, 2019.
- [36] Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13653–13662, 2021. 2, 6, 7
- [37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Con-

- ference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 4
- [38] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260. Springer, 2025. 6
- [39] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [40] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 4
- [41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4
- [42] Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for referring human dance generation in real world. arXiv preprint arXiv:2307.00040, 2023. 1, 2, 3, 6
- [43] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 5
- [44] Xiang Wang, Shiwei Zhang, Changxin Gao, Jiayu Wang, Xiaoqiang Zhou, Yingya Zhang, Luxin Yan, and Nong Sang. Unianimate: Taming unified video diffusion models for consistent human image animation. arXiv preprint arXiv:2406.01188, 2024. 1, 2, 6, 7
- [45] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [46] Zezheng Wang, Zitong Yu, Chenxu Zhao, Xiangyu Zhu, Yunxiao Qin, Qiusheng Zhou, Feng Zhou, and Zhen Lei. Deep spatial gradient and temporal depth learning for face anti-spoofing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5042– 5051, 2020. 8
- [47] Ziyi Xu, Ziyao Huang, Juan Cao, Yong Zhang, Xiaodong Cun, Qing Shuai, Yuchen Wang, Linchao Bao, Jintao Li, and Fan Tang. Anchorcrafter: Animate cyberanchors saling your products via human-object interacting video generation. arXiv preprint arXiv:2411.17383, 2024. 2, 3
- [48] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024. 1, 2, 3, 6
- [49] Sunjae Yoon, Gwanhyeong Koo, Younghwan Lee, and Chang D Yoo. Tpc: Test-time procrustes calibration for diffusion-based human image animation. arXiv preprint arXiv:2410.24037, 2024. 2

- [50] Wing-Yin Yu, Lai-Man Po, Ray CC Cheung, Yuzhi Zhao, Yu Xue, and Kun Li. Bidirectionally deformable motion modulation for video-based human pose transfer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7502–7512, 2023. 2
- [51] Zitong Yu, Chenxu Zhao, Zezheng Wang, Yunxiao Qin, Zhuo Su, Xiaobai Li, Feng Zhou, and Guoying Zhao. Searching central difference convolutional networks for face anti-spoofing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5295– 5305, 2020. 8
- [52] Dongxu Yue, Maomao Li, Yunfei Liu, Qin Guo, Ailing Zeng, Tianyu Yang, and Yu Li. Addme: Zero-shot group-photo synthesis by inserting people into scenes. In European Conference on Computer Vision, pages 222–239. Springer, 2024. 3
- [53] Polina Zablotskaia, Aliaksandr Siarohin, Bo Zhao, and Leonid Sigal. Dwnet: Dense warp-based network for pose-guided human video generation. arXiv preprint arXiv:1910.09139, 2019. 7
- [54] Pengze Zhang, Lingxiao Yang, Jian-Huang Lai, and Xiaohua Xie. Exploring dual-task correlation for pose guided person image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7713–7722, 2022. 2
- [55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [56] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. arXiv preprint arXiv:2406.19680, 2024. 1, 2, 7
- [57] Jian Zhao and Hui Zhang. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3657–3666, 2022. 2
- [58] Jiachen Zhou, Mingsi Wang, Tianlin Li, Guozhu Meng, and Kai Chen. Dormant: Defending against pose-driven human image animation. arXiv preprint arXiv:2409.14424, 2024. 8
- [59] Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10477–10486, 2023. 7
- [60] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision, pages 145–162. Springer, 2025. 1, 2, 3, 5, 6, 7

