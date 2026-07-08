# arXiv:2403.13745v1[cs.CV]20Mar2024

## Be-Your-Outpainter: Mastering Video Outpainting through Input-Specific Adaptation

Fu-Yun Wang1* Xiaoshi Wu1* Zhaoyang Huang2 Xiaoyu Shi1 Dazhong Shen3 Guanglu Song4 Yu Liu3 Hongsheng Li1

1MMLab, CUHK 2Avolution AI 3Shanghai AI Lab 4SenseTime Research {fywang@link, hsli@ee}.cuhk.edu.hk

###### https://be-your-outpainter.github.io/

Abstract. Video outpainting is a challenging task, aiming at generating video content outside the viewport of the input video while maintaining inter-frame and intra-frame consistency. Existing methods fall short in either generation quality or flexibility. We introduce MOTIA (Mastering Video Outpainting Through Input-Specific Adaptation), a diffusionbased pipeline that leverages both the intrinsic data-specific patterns of the source video and the image/video generative prior for effective outpainting. MOTIA comprises two main phases: input-specific adaptation and pattern-aware outpainting. The input-specific adaptation phase involves conducting efficient and effective pseudo outpainting learning on the single-shot source video. This process encourages the model to identify and learn patterns within the source video, as well as bridging the gap between standard generative processes and outpainting. The subsequent phase, pattern-aware outpainting, is dedicated to the generalization of these learned patterns to generate outpainting outcomes. Additional strategies including spatial-aware insertion and noise travel are proposed to better leverage the diffusion model’s generative prior and the acquired video patterns from source videos. Extensive evaluations underscore MOTIA’s superiority, outperforming existing state-of-the-art methods in widely recognized benchmarks. Notably, these advancements are achieved without necessitating extensive, task-specific tuning.

### 1 Introduction

Video outpainting aims to expand the visual content out of the spatial boundaries of videos, which has important real-world applications [4,6,7]. For instance, in practice, videos are usually recorded with a fixed aspect ratio, such as in movies or short clips. This becomes an issue when viewing these videos on smartphones, which often have varying aspect ratios, resulting in unsightly black bars on the screen that detract from the viewing experience. Proper ways for video outpainting are crucial in solving this issue. By expanding the visual content beyond the original frame, it adapts the video to fit various screen sizes seamlessly. This process ensures that the audience enjoys a full-screen experience without any compromise in visual integrity. However, the challenges associated with video

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|
|---|---|---|---|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|[Figure 11]|[Figure 12]|[Figure 13]|[Figure 14]|[Figure 15]|
|---|---|---|---|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|[Figure 25]|
|---|---|---|---|---|

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|
|---|---|---|---|---|

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

|[Figure 41]|[Figure 42]|[Figure 43]|[Figure 44]|[Figure 45]|
|---|---|---|---|---|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

|[Figure 51]|[Figure 52]|[Figure 53]|[Figure 54]|[Figure 55]|[Figure 56]|
|---|---|---|---|---|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

|[Figure 61]|[Figure 62]|[Figure 63]|[Figure 64]|[Figure 65]|[Figure 66]|[Figure 67]|[Figure 68]|[Figure 69]|
|---|---|---|---|---|---|---|---|---|

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

###### Source videos Outpainting results by MOTIA

Fig. 1: MOTIA is a high-quality flexible video outpainting pipeline, leveraging the intrinsic data-specific patterns of source videos and image/video generative prior for state-of-the-art performance. Quantitative metric improvement of MOTIA is significant (Table 1).

outpainting are significant. It requires not just the expansion of each frame’s content but also the preservation of temporal (inter-frame) and spatial (intraframe) consistency across the video.

Currently, there are two primary approaches to video outpainting. The first employs optical flows and specialized warping techniques to extend video frames, involving complex computations and carefully tailored hyperparameters to ensure the added content remains consistent [6,8]. However, their results are far from satisfactory, suffering from blurred content. The other type of approach in video outpainting revolves around training specialized models tailored for video inpainting and outpainting with extensive datasets [7,33]. However, they have two notable limitations: 1) An obvious drawback of these models is their dependency on the types of masks and the resolutions of videos they can handle, which significantly constrains their versatility and effectiveness in real-world applications, as they may not be adequately equipped to deal with the diverse range

of video formats and resolutions commonly encountered in practical scenarios. 2) The other drawback is their inability to out-domain video outpainting, even intensively trained on massive video data. Fig. 2 shows a failure example of most advanced previous work [7] that the model faces complete outpainting failure, with only blurred corners. We show that a crucial reason behind this is that the model fails at capturing the intrinsic data-specific patterns from out-domain source (input) videos.

In this work, we propose MOTIA: Mastering Video Outpainting Through Input-Specific Adaptation, a diffusion-based method for open-domain video outpainting with arbitrary types of mask, arbitrary video resolutions and lengths, and arbitrary styles. At the heart of MOTIA is treating the source video itself as a rich source of information [18,23], which contains key motion and content patterns (intrinsic dataspecific patterns) necessary for effective outpainting. We demonstrate that the patterns learned from the source video, combined with the generative capabilities of diffusion models, can achieve surprisingly great outpainting performance.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

###### M3DDMMOTIA

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Fig. 2: Failure example of previous methods. Many previous methods including the intensively trained models on video outpainting still might suffer from generation failure, that the model simply generates blurred corners. MOTIA never encounters this failure.

Fig. 3 illustrates the workflow of MOTIA. MOTIA consists of two stages: input-specific adaptation and pattern-aware outpainting. During the input-specific adaptation stage, we conduct pseudo video outpainting learning on the source video (videos to be outpainted) itself. Specifically, at each iteration, we heuristically add random masks to the source video and prompt the base diffusion model to recover the masked regions by learning to denoise the video corrupted by white noise, relying on the extracted information from unmasked regions. This process not only allows the model to capture essential data-specific patterns from the source video but also narrows the gap between standard generation and outpainting. We insert trainable lightweight adapters into the diffusion model for tuning to keep the efficiency and stability. In the pattern-aware outpainting stage, we combine the learned patterns from the source video and the generation prior of the diffusion model for effective outpainting. To better leverage the generation ability of the pretrained diffusion model and the learned pattern from the source video, we propose spatial-aware insertion (SA-Insertion) of the tuned adapters for outpainting. Specifically, the insertion weights of adapters gradually decay as the spatial position of features away from the known regions. In this way, the outpainting of pixels near the known regions is more influenced by the

learned patterns, while the outpainting of pixels far from the known regions relies more on the original generative prior of diffusion model. To further mitigate potential denoising conflicts and enhance the knowledge transfer between known regions and unknown regions, we incorporate noise regret that we add noise and denoise periodically at early inference steps, which works for more harmonious outpainting results.

Extensively quantitative and qualitative experiments verify the effectiveness of our proposed method. MOTIA overcomes many limitations of previous methods and outperforms the state-of-the-art intensively trained outpainting method in standard widely used benchmarks. In summary, our contribution is three-fold:

- 1) We show that the data-specific patterns of source videos are crucial for effective outpainting, which is neglected by previous work. 2) We introduce an adaptation strategy to effectively capture the data-specific patterns and then propose novel strategies to better leverage the captured patterns and pretrained image/video generative prior for better results. 3) Vast experiments verify that our performance in video outpainting is great, significantly outperforming previous state-of-the-art methods in both quantitative metrics and user studies.
- 2 Related Works

In this section, we discuss related diffusion models and outpainting methods.

Diffusion models. Diffusion models (a.k.a., score-based models) [10, 11, 17, 21, 25] have gained increasing attention due to its amazing ability to generate highly-detailed images. Current successful video diffusion models [5,10,24,27] are generally built upon the extension of image diffusion models through inserting temporal layers. They are either trained with image-video joint tuning [12,24] or trained with spatial weights frozen [5] to mitigate the negative influence of the poor captions and visual quality of video data.

Ooutpainting methods. Video outpainting is largely built upon the advancements in image outpainting, where techniques ranged from patch-based methods (e.g., PatchMatch [4]) to more recent deep learning approaches like GANs [1,32]. Diffusion models [2,16], benefiting from the learned priors on synthesis tasks and the process of iterative refinement, achieve state-of-the-art performance on image outpainting tasks. The research focusing on video outpainting is comparatively few. Previous works typically apply optical flow for outpainting, which warps content from adjacent frames to the outside corners, but their results are far from satisfactory. Recently, M3DDM [7] trained a large 3D diffusion models with specially designed architecture for outpainting on massive video data, achieving much better results compared to previous methods, showcasing the huge potential of diffusion models on video outpainting. However, as we claimed, they have two main limitations: 1) The inflexibility for mask types and video resolutions. They can only outpaint video with resolution 256 × 256 with square type of masking. 2) Inability for out-domain video outpainting. As shown in Fig. 2, they encounter outpainting failure when processing out domain videos even intensively trained on massive video data.

[Figure 89]

Noise

U-Net ×𝑵

[Figure 90]

[Figure 91]

Input-Specific Adaptation

×𝑻 Timesteps

Input

Fully Insertion

Low-Rank Adapters

Pattern-Aware Outpainting

Mask

Insertion

[Figure 92]

[Figure 93]

Pseudo Outpainting

[Figure 94]

Frozen

SA-

[Figure 95]

Trainable

[Figure 96]

Augment

Spatial Layer

Input

[Figure 97]

Outpainting

[Figure 98]

Temporal Layer

Source Video

Noise

×𝑴

Noise Regret

- Fig. 3: Workflow of MOTIA. Blue lines represent the workflow of input-specific adaptation, and green lines represent the workflow of pattern-aware outpainting.

### 3 Preliminaries

Diffusion models [11] add noise to data through a Markov chain process. Given initial data x0 ∼ q(x0), this process is formulated as:

T

√αtxt−1,βtI), (1)

q(xt|xt−1), q(xt|xt−1) = N(xt|

q(x1:T|x0) =

t=1

where βt is the noise schedule and αt = 1−βt. The data reconstruction, or denoising process, is accomplished by the reverse transition modeled by pθ(xt−1|xt):

q(xt−1|xt,x0) = N(xt−1;µ˜t(xt,x0),β˜tI), (2) with µ˜t(xt,x0) = √1α

√1−α¯t√t αtϵ, α¯t = ts=1 αs, β˜t = 1−α¯

xt − 1−α

1−α¯t βt, and ϵ is the noise added to x0 to obtain xt.

t−1

t

Diffusion-based outpainting aims to predict missing pixels at the corners of the masked region with the pre-trained diffusion models. We denote the ground truth as x, mask as m, known region as (1 − m) ⊙ x, and unknown region as m ⊙ x. At each reverse step in the denoising process, we modify the known regions by incorporating the intermediate noisy state of the source data from the corresponding timestep in the forward diffusion process (which adds noise), provided that this maintains the correct distribution of correspondences. Specifically, each reverse step can be denoted as the following formulas:

√α¯tx0,(1 − α¯t)I , xunknownt−1 ∼ N (µθ (xt,t),Σθ (xt,t)), (3)

xknownt−1 ∼ N

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

- Fig. 4: Sample results of quantitative experiments. All videos are outpainted with a horizontal mask ratio of 0.66. Contents outside the yellow lines are outpainted by MOTIA.

xt−1 = m ⊙ xknownt−1 + (1 − m) ⊙ xunknownt−1 , (4) where the xknownt−1 is padded to the target resolution before the masked merging.

### 4 Methodology

This section presents MOTIA, a method demonstrating exceptional performance in video outpainting tasks. We begin by defining the concept of video outpainting and describing the foundational model in Section 4.1. and Section 4.2. Subsequently, we delve into the specifics of input-specific adaptation and pattern-aware

outpainting in Sections 4.3 and 4.4, respectively. Moreover, we show that our approach has great promise in extending its application to long video outpainting, which will be explored in Section 4.5.

##### 4.1 Problem Formulation

For a video represented as v ∈ Rt×h×w×d, where t denotes the number of frames, h denotes frame height, w denotes frame width, and d denotes channel depth. Video outpainting model f(v) is designed to generate a spatially expanded video

- v′ ∈ Rt×h

′×w′×d. This process not only increases the spatial dimensions (h′ > h,

- w′ > w), but also requires to ensure continuity and harmony between the newly expanded regions and the known regions. The transformation maintains the known regions unchanged, with f(v) acting as an identity in these regions.

##### 4.2 Network Expansion

Network inflation. MOTIA leverages the pre-trained text-to-image (T2I) model,

Stable Diffusion. In line with previous video editing techniques [30], we transform 2D convolutions into pseudo 3D convolutions and adapt 2D group normalizations into 3D group normalizations to process video latent features. Specifically, the 3 × 3 kernels in convolutions are replaced by 1 × 3 × 3 kernels, maintaining identical weights. Group normalizations are executed across both temporal and spatial dimensions, meaning that all 3D features within the same group are normalized simultaneously, followed by scaling and shifting.

Masked video as conditional input. Additionally, we incorporate a ControlNet [34], initially trained for image inpainting, to manage additional mask inputs. Apart from noise input, ControlNet can also process masked videos to extract effective information for more controllable denoising. In these masked videos, known regions have pixel values ranging from 0 to 1, while values of masked regions are set to −1.

Temporal consistency prior. To infuse the model with temporal consistency priors, we integrate temporal modules pre-trained on text-to-video (T2V) tasks. Note that although MOTIA relies on pre-trained video diffusion modules, applying these pre-trained temporal modules directly for video outpainting yields rather bad results, significantly inferior to all baseline methods (Table. 3). However, when equipped with our proposed MOTIA, the model demonstrates superior performance even in comparison to models specifically designed and trained for video outpainting, underscoring the efficacy of MOTIA.

##### 4.3 Input-Specific Adaptation

The input-specific adaptation phase is crucial in our video outpainting method, aiming to tailor the model for the specific challenges of outpainting. This phase involves training on the source video with a pseudo-outpainting task, importantly, enabling the model to learn intrinsic content and motion patterns (dataspecific patterns) within the source video as well as narrowing the gap between the standard generation process and outpainting.

Element-wise multiplication

Unknown

[Figure 108]

[Figure 109]

[Figure 110]

Element-wise addition

| | | | |
|---|---|---|---|
| |[Figure 111]<br><br>| | |
| | | | |
| | | | |

Trainable

Freeze

|exp(− 𝐾∥𝐩−𝐩c∥<br><br>𝑚𝑎𝑥<br><br>𝐩<br><br>∥𝐩−𝐩c∥<br><br>)|
|---|

Known

Attention

Low-Rank Adapter

Spatially Aware Insertion

|× 𝟑|
|---|

Latent Feature

[Figure 112]

[Figure 113]

[Figure 114]

|× 𝟐|
|---|

Query

Multi-Head Attention FeedForward

Key

Value

- Fig. 5: Spatial-aware insertion scales the insertion weights of adapters for better leveraging of learned patterns and generative prior.

[Figure 115]

Initial noise distribution

[Figure 116]

[Figure 117]

Ground truth distribution

[Figure 118]

Low-Density Region

[Figure 119]

Estimated distribution

[Figure 120]

Noise regret Fused score, applied in fact

[Figure 121]

Score of estimated distribution

[Figure 122]

Score of ground truth distribution

Low-Density Region

Generation failure

Fig. 6: Noise regret fixes possible generation failure/degradation caused by score conflicts.

Video augmentation. Initially, we augment the source video. Transformations like identity transformation, random flipping, cropping, and resizing can be employed. This step can potentially help the model better learn and adapt to diverse changes in video content. For longer video outpainting, as we will discuss later, instead of taking it as a whole, we randomly sample short video clips from it to reduce the cost of the adaptation phase.

Video masking. We then add random masks to the video. We adopt a heuristic approach that uniformly samples edge boundaries of 4 sides within given limits. The area enclosed by these boundaries is considered the known region, while the rest is the unknown region. This masked video serves as the conditional input for the ControlNet, simulating the distribution of known and unknown areas in actual outpainting scenarios.

Video noising. Additionally, we apply noise to the video following the DDPM [11]

by randomly selecting diffusion timesteps. This noisy video serves as an input for both the ControlNet and the Stable Diffusion model, training the model to adapt to various noise conditions.

Optimization. Finally, we optimize the model. To ensure efficiency, low-rank adapters are inserted into the layers of the diffusion model. We optimize only the parameters of these adapters while keeping the other parameters frozen. The loss function is

L = ϵ − ϵˆθ¯l,θ¯c,θa(vnoisy,vmasked,t) 2 , (5) where t represents the timestep in the process, ϵ is the added noise, vnoisy refers to the video perturbed by ϵ, and vmasked denotes the masked video. The parameters θl, θc, and θa correspond to the Diffusion Model, ControlNet, and adapters, respectively. The bar over these parameters indicates they are frozen during the optimization. This optimization process, including the steps of augmentation, masking, and noising, is repeated to update the lightweight adapters to capture the data-specific patterns from the source video.

##### 4.4 Pattern-Aware Outpainting

Following the initial phase of input-specific adaptation, our model shows promising results in video outpainting using basic pipelines as outlined in Eq. 3 and

Eq. 4, achieving commendable quality. However, we here introduce additional inference strategies that can be combined to better leverage the learned dataspecific patterns from the input-specific adaptation phase for better outpainting results. We call the outpainting process that incorporates these strategies pattern-aware outpainting.

Spatial-aware insertion. It is important to acknowledge that in the inputspecific adaptation phase, the model is fine-tuned through learning outpainting within the source video. However, at the outpainting phase, the model is expected to treat the entire source video as known regions and then fill the unknown regions at edges (i.e., generating a video with a larger viewport and resolution). This specificity may lead to a noticeable training-inference gap during outpainting, potentially affecting the outpainting quality. To balance the fine-tuned patterns with the diffusion model’s inherent generative prior, we introduce the concept of spatial-aware insertion (SA-Insertion) of adapters as shown in Fig. 5. The adaptation involves adjusting the insertion weight of tuned low-rank adapters based on the feature’s spatial position. We increase insertion weight near known areas to utilize the learned patterns while decreasing it in farther regions to rely more on the original generative capacity of the diffusion model. To be specific,

Wadapted⊤ xp = W⊤xp + α(p)(WupWdown)⊤ xp. (6) Here, p signifies the spatial position of x, W ∈ Rd

in×dout denotes the linear transformation in layers of diffusion model, Wdown ∈ Rd

in×r and Wup ∈ Rr×d

out

are the linear components of the adapter with rank r ≪ min(din,dout). The function α(p) is defined as:

K∥p − pc∥ maxp¯ ∥p¯ − pc∥

), (7)

α(p) = exp(−

where K is a constant for controlling decay speed, and pc represents the nearest side of the known region to p.

Noise regret. In the context of Eq. 3, merging noisy states from known and unknown regions in video outpainting tasks poses a technical problem. This process, similar to sampling from two different vectors, can disrupt the denoising direction. As depicted in Fig. 6, the estimated denoising direction initially points downwards to the left, in contrast to the true direction heading towards the top-right. This leads to a merged trajectory directed to a less dense top-left region, potentially resulting in generation failures (see Fig. 2), even in welltrained models. Given the significant impact of early steps on the generation’s structure, later denoising may not rectify these initial discrepancies. Inspired by DDPM-based image inpainting methods [16,22], we propose to re-propagate the noisy state into a noisier state by adding noise when denoising and then provide the model a second chance for re-denoising. This helps integrate known region data more effectively and reduces denoising direction conflicts. In detail, after obtaining vt during denoising, we conduct

vt+L = Πit=+tL+1αivt + 1 − Πit=+tL+1αiϵ,, (8)

where αi = 1 − βi and ϵ ∼ N(0,I). Then we restart the denoising process. We repeat this progress for M times. We only conduct it in the early denoising steps.

##### 4.5 Extension to Long Video Outpainting

We show that our method can be easily extended for long video outpainting. Specifically, for the stage of input-specific adaptation, instead of taking the long video as a whole for adaptation (Direct adaptation on long videos is costly and does not align with the video generation prior of the pretrained modules), we randomly sample short video clips from the long video for tuning to learn global patterns without requiring more GPU memory cost. For the stage of patternaware outpainting, we split the long video into short video clips with temporal overlapping (i.e., some frames are shared by different short video clips), and then conduct temporal co-denoising following Gen-L [28]. Specifically, the denoising result for jth frame of the long video at timestep t is approximated by the weighted sum of all the corresponding frames in short video clips that contain it,

j (Wi,j∗)2 ⊗ vti−1,j∗ i∈Ij Wi,j2 ∗

vt−1,j = i∈I

2 , (9)

where ⊗ denotes element-wise multiplication,vt−1,j denotes the noisy state of the jth frame at timestep t, vti−1,j∗ is the noisy state of jth frame predicted with only information from the ith video clip at timestep t, Wi,j∗ is the per-pixel weight, which is as 1 as default.

### 5 Experiments

##### 5.1 Experimental Setup

Benchmarks. To verify the effectiveness of MOTIA, we conduct evaluations on DAVIS [19] and YouTube-VOS [31], which are widely used benchmarks for video outpainting. Following M3DDM [7], we compare the results of different methods in the horizontal direction, using mask ratios of 0.25 and 0.66.

Evaluation metrics. Our evaluation approach utilizes four well-established metrics: Peak Signal to Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [29], Learned Perceptual Image Patch Similarity (LPIPS) [35], and Frechet Video Distance (FVD) [26]. For assessing PSNR, SSIM, and FVD, the generated videos are converted into frames within a normalized value range of [0,1]. LPIPS is evaluated over a range of [−1,1]. About the FVD metric, we adopt a uniform frame sampling, with 16 frames per video for evaluation following M3DDM.

Compared methods. The comparative analysis includes the following methods: 1) VideoOutpainting [6]: Dehan et al. [6] propose to tackle video outpainting by bifurcating foreground and background components. It conducts separate flow estimation and background prediction and then fuses these to generate a

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

M3DDMMOTIASDM

|[Figure 131]<br><br>|[Figure 132]|
|---|---|
|[Figure 133]|[Figure 134]|

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

VideoOut

-painting

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

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Source Videos

Fig. 7: Qualitative comparison. Other methods outpainting the source video with a mask ratio of 0.6. MOTIA outpainting the source video with a larger mask ratio of 0.66 while achieving obviously better outpainting results.

cohesive output. 2) SDM [7]: SDM considers the initial and terminal frames of a sequence as conditional inputs, merged with the context at the initial network layer. It is trained on video datasets including WebVid [3] and e-commerce [7]. 3) M3DDM [7]: M3DDM is an innovative pipeline for video outpainting. It adopts a masking technique allowing the original source video as masked conditions. Moreover, it uses global-frame features for cross-attention mechanisms, allowing the model to achieve global and long-range information transfer. It is intensively trained on vast video data, including WebVid and e-commerce, with a specialized architecture design for video outpainting. In this way, SDM could be viewed as a pared-down version of M3DDM, yet it is similarly intensively trained.

Implementation details. Our method is built upon Stable Diffusion v1-5. We add the ControlNet pretrained on image inpainting to enable the model to accept additional masked image inputs. The temporal modules are initialized with the weights from pretrained motion modules [9] to obtain additional motion priors. The motion modules are naive transformer blocks trained with solely text-tovideo tasks on WebVid. For the input-specific adaptation, the low-rank adapters are trained using the Adam optimizer. We set the learning rate to 10−4, and set the weight decay to 10−2. The LoRA rank and αlora are set to 16 and, 8, respectively. The number of training steps is set to 1000. We do not apply augmentation for simplicity. For both mask ratios of 0.66 and 0.25, we simply apply the same random mask strategy, which uniformly crops a square in the middle as the known regions. For the pattern-aware outpainting, the diffusion steps are set to 25 and the classifier-free guidance (CFG) scale is set to 7.5 and we only apply CFG at the first 15 inference steps. When adding noise regret to further improve the results, we set jump length L = 3, and repeat time M = 4.

Table 1: Quantitative comparison of video outpainting methods on DAVIS and YouTube-VOS datasets. ↑ means ‘better when higher’, and ↓ indicates ‘better when lower’.

DAVIS YouTube-VOS PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓ PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓

Method

VideoOutpainting [6] 17.96 0.6272 0.2331 363.1 18.25 0.7195 0.2278 149.7 SDM [7] 20.02 0.7078 0.2165 334.6 19.91 0.7277 0.2001 94.81 M3DDM [7] 20.26 0.7082 0.2026 300.0 20.20 0.7312 0.1854 66.62 MOTIA 20.36 0.7578 0.1595 286.3 20.25 0.7636 0.1727 58.99

We only apply noise regret in the first half inference steps. Note that our method is built upon LDM, which requires text-conditional inputs. For a fair comparison and to remove the influence of the choice of text prompt, we apply Blip [14] to select the prompt automatically. We observe dozens of prompt mistakes but do not modify them to avoid man-made influence.

##### 5.2 Qualitative Comparison

- Fig. 7 showcases a qualitative comparison of MOTIA against other methods. Outpainting a narrow video into a square format. MOTIA employs a mask ratio of 0.66, surpassing the 0.6 ratio utilized by other methods, and demonstrates superior performance even with this higher mask ratio. The SDM method only manages to blur the extremities of the video’s background, egregiously overlooking the primary subject and resulting in the outpainting failure as previously highlighted in Fig. 2. Dehan’s approach effectively outpaints the background but utterly fails to address the foreground, leading to notable distortions. In contrast, the M3DDM method adeptly handles both subject and background integration but is marred by considerable deviations in subject characteristics, such as incorrect brown coloration in the dog’s fur across several frames. Our method stands out by achieving optimal results, ensuring a harmonious and consistent outpainting of both the foreground and background.

##### 5.3 Quantitative Comparison

Table 1 summarizes the evaluation metrics of our method compared to other approaches. Our method achieves comparable results to the best method in PSNR. It shows significant improvements in video quality (SSIM), perceptual metric (LPIPS), and distribution similarity (FVD). Specifically, our SSIM, LPIPS, and FVD metrics show improvements of 7.00%, 21.27%, and 4.57% respectively on the DAVIS dataset, and 4.43%, 6.85%, and 11.45% on the YouTube-VOS dataset compared to the best-performing method.

##### 5.4 Ablation Study

Ablation study on input-specific adaptation. We conducted the ablation study on input-specific adaptation with the DAVIS dataset to verify its effective-

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

#### SD+T SD+T+C Direct-tune Input-Specific Adaptation

###### Fig. 8: Visual examples of ablation study on the proposed input-specific adaptation.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

###### Direct (Ours w/o SA + NR) SA (Ours w/o NR) Pattern-Aware Outpainting

Fig. 9: Visual examples of ablation study on pattern-aware outpainting.

ness, as shown in Fig. 8 and Table 2. “SD+T” represents the result of directly combining the temporal module with Stable Diffusion, which led to a complete outpainting failure. “SD+T+C” indicates the additional use of ControlNet, resulting in similarly poor outcomes. “Direct-tune” refers to the approach of directly fitting the original video without outpainting training; in this case, we observed a very noticeable color discrepancy between the outpainted and known areas. In contrast, our method achieved the best results, ensuring consistency in both the visual and temporal aspects. The metrics shown in Table 2 also support this observation, with MOTIA significantly outperforming the other baselines.

Ablation study on pattern-aware outpainting. Table 3 summarizes our ablation experiments for the pattern-aware outpainting part. We conducted extensive validation on the YouTube-VOS dataset. “Direct” refers to performing outpainting according to Eq. 3 directly after input-specific adaptation. “SA” denotes spatially-aware insertion, and “SA+NR” indicates the combined use of spatially-aware insertion and noise regret. The experimental results demonstrate that each of our components effectively enhances performance. Specifically, Combining both SA-Insertion and Noise regret, the PSNR, SSIM, LPIPS, and FVD metrics show improvements of 2.69%, 0.90%, 3.95%, and 11.32% respectively than directly applying Eq. 3. Fig. 9 presents the visual examples of ablation study on our proposed pattern-aware outpainting part. When removing NR, it might fail to align the texture colors or produce unreasonable details (e.g., arms in the middle of Fig. 9). When further removing SA, it could potentially generate

Table 2: Ablation study on inputspecific adaptation. Component PSNR↑ SSIM↑ LPIPS↓ FVD ↓

SD+T 15.59 0.6640 0.2979 672.7 SD+T+C 16.81 0.6961 0.2338 515.4 Direct-tune 19.42 0.7375 0.1784 312.1 MOTIA 20.36 0.7578 0.1595 286.3

Table 3: Ablation study on the proposed pattern-aware outpainting. Component PSNR↑ SSIM↑ LPIPS↓ FVD ↓

Direct 19.72 0.7568 0.1798 66.52 SA 19.97 0.7608 0.1752 58.40 SA+NR 20.25 0.7636 0.1727 58.99

Table 4: Computation complexity of MOTIA. Phase VARM↓ Time↓ Input-Specific Adapt 12.70 GB 309 Seconds Pattern-Aware Outpaint 5.80 GB 58 Seconds MOTIA (In total) 12.70 GB 367 Seconds

Table 5: User study comparison between M3DDM and MOTIA.

Method Visual-Quality Realism M3DDM 27.4% 42.8% MOTIA 72.6% 57.2%

unrealistic results caused by the overfitting to the target video (e.g., the white collar on the left of Fig. 9). Note that even though the FVD degrades in a very slight manner, all the other metrics increase and we qualitatively find it to be helpful for improving results.

##### 5.5 Discussions

Model and computation complexity. Model Complexity: The original model has 1.79 billion (including the auto-encoder and text encoder) parameters in total, while the added adapters contain 7.49 million parameters, leading to an increase of 0.42% in memory usage. Computation Complexity: We report the peak GPU VRAM and the time required for outpainting a target video from 512×512 to 512×1024 with 16 frames at two stages in Table 4. For longer videos, as described in Section 4.5, instead of processing the long video as a whole, we adapt only to short video clips sampled from the long video. This approach does not require additional time or GPU VRAM during the input-specific adaptation phase. Additionally, with temporal co-denoising [28], the GPU VRAM usage remains the same as that for short video during the pattern-aware outpainting phase, while the required time increases linearly with the video length.

User study. We conducted a user study between MOTIA and M3DDM, utilizing the DAVIS dataset with a horizontal mask of 0.66 as source videos. Preferences were collected from 10 volunteers, each evaluating 50 randomly selected sets of results based on visual quality (such as clarity, color fidelity, and texture detail) and realism (including motion consistency, object continuity, and integration with the background). Table 2 demonstrates that the outputs from MOTIA are preferred over those from M3DDM in both visual quality and realism.

Why MOTIA outperforms (Why previous methods fail). 1) Flexibility. Current video diffusion models are mostly trained with fixed resolution and length, lacking the ability to tackle videos with various aspect ratios and lengths. In contrast, the adaptation phase of MOTIA allows the model to better capture

the size, length, and style distribution of the source video, greatly narrowing the gap between pretrained weights and the source video. 2) Ability for capturing intrinsic patterns from source video. A crucial point for successful outpainting is the predicted score of diffusion models should be well-compatible with the original known regions of the source video. To achieve this, the model should effectively extract useful information from the source video for denoising. For instance, M3DDM concatenates local frames of source video at the input layers and incorporates the global frames through the cross-attention mechanism after passing light encoders. However, the information might not be properly handled especially for out-domain inputs, thus leading to outpainting failure. Instead, by conducting input-specific adaptation on the source video, the model can effectively capture the data-specific patterns in the source videos through gradient. Through this, MOTIA better leverage the data-specific patterns of the source video and image/video generative prior for outpainting. We hope this work inspires following research to exploit more from the source video itself instead of purely relying on the generative prior from intensive training on videos.

### 6 Conclusion

We present MOTIA, an innovative advancement in video outpainting. MOTIA relies on a combination of input-specific adaptation for capturing inner video patterns and pattern-aware outpainting to generalize these patterns for actual outpainting. Extensive experiments validate the effectiveness.

Limitations: MOTIA requires learning necessary patterns from the source video, when the source video contains little information, it poses a significant challenge for MOTIA to effectively outpainting it.

### References

- 1. Arora, R., Lee, Y.J.: Singan-gif: Learning a generative video model from a single gif. In: CVPR. pp. 1310–1319 (2021) 4
- 2. Avrahami, O., Lischinski, D., Fried, O.: Blended diffusion for text-driven editing of natural images. In: CVPR. pp. 18208–18218 (2022) 4
- 3. Bain, M., Nagrani, A., Varol, G., Zisserman, A.: Frozen in time: A joint video and image encoder for end-to-end retrieval. In: ICCV. pp. 1728–1738 (2021) 11
- 4. Barnes, C., Shechtman, E., Finkelstein, A., Goldman, D.B.: Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Trans. Graph. 28(3), 24 (2009) 1, 4
- 5. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: CVPR. pp. 22563–22575 (2023) 4
- 6. Dehan, L., Van Ranst, W., Vandewalle, P., Goedemé, T.: Complete and temporally consistent video outpainting. In: CVPRW. pp. 687–695 (2022) 1, 2, 10, 12, 3
- 7. Fan, F., Guo, C., Gong, L., Wang, B., Ge, T., Jiang, Y., Luo, C., Zhan, J.: Hierarchical masked 3d diffusion model for video outpainting. In: ACM MM. pp. 7890–7900 (2023) 1, 2, 3, 4, 10, 11, 12
- 8. Gao, C., Saraf, A., Huang, J.B., Kopf, J.: Flow-edge guided video completion. In: ECCV. pp. 713–729. Springer (2020) 2
- 9. Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., Dai, B.: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023) 11, 1
- 10. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022) 4
- 11. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. NeurIPS 33, 6840–6851 (2020) 4, 5, 8
- 12. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. arXiv:2204.03458 (2022) 4
- 13. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021) 1
- 14. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: ICML. pp. 12888–

12900. PMLR (2022) 12, 1

- 15. Liew, J.H., Yan, H., Zhang, J., Xu, Z., Feng, J.: Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749 (2023) 3
- 16. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising diffusion probabilistic models. In: CVPR. pp. 11461–11471 (2022) 4, 9
- 17. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021) 4
- 18. Nikankin, Y., Haim, N., Irani, M.: Sinfusion: Training diffusion models on a single image or video. arXiv preprint arXiv:2211.11743 (2022) 3
- 19. Pont-Tuset, J., Perazzi, F., Caelles, S., Arbeláez, P., Sorkine-Hornung, A., Van Gool, L.: The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675 (2017) 10, 2

- 20. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763. PMLR (2021) 1
- 21. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR. pp. 10684–10695 (2022) 4, 1
- 22. Rout, L., Parulekar, A., Caramanis, C., Shakkottai, S.: A theoretical justification for image inpainting using denoising diffusion probabilistic models. arXiv preprint arXiv:2302.01217 (2023) 9
- 23. Shaham, T.R., Dekel, T., Michaeli, T.: Singan: Learning a generative model from a single natural image. In: ICCV. pp. 4570–4580 (2019) 3
- 24. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without textvideo data. arXiv preprint arXiv:2209.14792 (2022) 4
- 25. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020) 4
- 26. Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018) 10
- 27. Voleti, V., Jolicoeur-Martineau, A., Pal, C.: Mcvd: Masked conditional video diffusion for prediction, generation, and interpolation. NeurIPS (2022) 4
- 28. Wang, F.Y., Chen, W., Song, G., Ye, H.J., Liu, Y., Li, H.: Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264

(2023) 10, 14

- 29. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004) 10
- 30. Wu, J.Z., Ge, Y., Wang, X., Lei, W., Gu, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565 (2022) 7
- 31. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327

(2018) 10, 2

- 32. Xu, T., Zhang, P., Huang, Q., Zhang, H., Gan, Z., Huang, X., He, X.: Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In: CVPR. pp. 1316–1324 (2018) 4
- 33. Yu, L., Cheng, Y., Sohn, K., Lezama, J., Zhang, H., Chang, H., Hauptmann, A.G., Yang, M.H., Hao, Y., Essa, I., et al.: Magvit: Masked generative video transformer. In: CVPR. pp. 10459–10469 (2023) 2
- 34. Zhang, L., Agrawala, M.: Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543 (2023) 7, 1
- 35. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR. pp. 586–595 (2018) 10

## Supplementary Material MOTIA: Mastering Video Outpainting through Input-Specific Adaptation

### I Detailed Implementation

Model architecture. MOTIA marries generative priors with pretrained models for fast adaptation and better generalization. Specifically, the basic components of MOTIA in model architecture aspect are:

- • Variational autoencoder [21]. The autoencoder consists of an encoder and decoder, with the encoder mapping the original video frames into latent space and the decoder decoding the video frames with latent codes.
- • CLIP text encoder [20]. CLIP is trained on vast text-image pairs, enabling its text encoder to contain meaningful and rich information for controlling image generation.
- • U-Net [21]. We apply Stable Diffusion v1-5 as our fundamental denoisier. The U-Net is conditioned on text embeddings of CLIP through cross-attention. To make it applicable to the 3D features of videos, we inflate the 2D convolutions and 2D group normalizations within it into pseudo-3D convolutions and 3D group normalizations.
- • Temporal module [9]. To equip the model with additional temporal priors, we initialize additional temporal attention layers with vanilla transformer architectures pretrained on large-scale text video datasets. Note that, we have shown that directly applying this temporal prior for video outpainting leads to poor results without our proposed input-specific adaptation process.
- • LoRA [13]. LoRA is proposed for the efficient fine-tuning of large models. It has been widely used in various diffusion-based applications, including video editing and manipulation. Therefore, we also choose LoRA as the basic learning component. Additionally, unlike previous works directly inserting the trained LoRA, we propose an effective strategy that adjusts the insertion weight of LoRA according to the spatial position of the given feature, achieving better balance in the learned patterns and generative priors of the pretrained model.
- • ControlNet [34]. ControlNet works as a plug-and-play module for Stable Diffusion, allowing it to accept additional input for better controlling the denoising results. We apply a ControlNet pretrained on Image Inpainting tasks, accepting the masked image to instruct the whole denoising process.
- • Blip [14]. Note that our method is built upon Stable Diffusion, which is a conditional denoiser, requiring appropriate text conditions to achieve good results. We apply Blip to automatically provide the captions to avoid manmade influence.

- Algorithm 1 Input-Specific Adaptation in MOTIA. Require: Source video v

- 1: Initialize Stable Diffusion (SD), ControlNet (C), Temporal Module (T) with frozen weights
- 2: Initialize trainable Low-Rank Adapters (loRA)
- 3: function Input-Specific Adaptation
- 4: Insert loRA fully into layers of SD ▷ Full-insertion
- 5: Add loRA to the optimizer
- 6: for i = 1 to iterations do
- 7: vaugment ← Augment(v)
- 8: t ∼ Uniform(1, T)
- 9: ϵ ∼ N(0, I)
- 10: vnoisy ← AddNoise(vaugment, t, ϵ)
- 11: vmask ← RandomMask(vaugment)
- 12: Optimize gap between predicted noise and ϵ
- 13: L = ϵ − ϵˆθ¯l,θ¯c,θa(vnoisy, vmasked, t) 2 ▷ Learning through pseudo outpainting task
- 14: Gradient backpropagation
- 15: Update optimizer
- 16: Zero gradients of optimizer
- 17: end for
- 18: end function

Pseudo algorithm code. The MOTIA framework operates in a two-fold manner: the input-specific adaptation hones the model’s ability to capture the essential content and motion patterns from the source video, while the pattern-aware outpainting generalizes the captured patterns to creatively expand the video’s horizon. The overall pipelines for input-specific adaptation and pattern-aware outpainting are shown in Algorithm 1 and Algorithm 2.

### II Benchmark Details

The quantitative metric evaluation of MOTIA is mostly based on DAVIS [19] and YouTube-VOS [31]. The DAVIS (Densely Annotated Video Segmentation) dataset is pivotal for video object segmentation research. DAVIS 2016 contains 50 videos (30 for training, 20 for testing), each featuring a single instance annotation per frame. DAVIS 2017 expands this scope with 150 videos in total (60 for training, 30 for validation, 60 for testing), annotating multiple instances per video. This dataset supports semi-supervised and unsupervised tasks, differing in the level of human input during testing. The YouTube-VOS dataset, designed for Video Object Segmentation (VOS), is a substantial benchmark with over 4,000 high-resolution YouTube videos, totaling over 340 minutes. It supports multiple VOS tasks, including semi-supervised and unsupervised video object segmentation. In our study, frames from these videos are used as inputs, cropped on the sides, without annotated foreground masks. Though designed for segmentation, these datasets are widely used to evaluate the performance of video outpainting

- Algorithm 2 Pattern-Aware Outpainting in MOTIA. Require: Source video v Ensure: Outpainted video voutpainted

- 1: function Pattern-Aware Outpainting
- 2: Insert loRA Spatial-awarely into layers of SD ▷ SA-Insertion
- 3: Repeat Time M, and Jump Length L ▷ Hyper-parameters for noise regret
- 4: vT ∼ N(0, I)
- 5: t ← T
- 6: m ← 0
- 7: while t ̸= 0 do
- 8: ϵ ∼ N(0, I) if t > 1 else ϵ = 0
- 9: vtknown−1 ←

√α¯t−1v0 + (1 − α¯t−1)ϵ

- 10: zt ∼ N(0, I) if t > 1 else zt = 0
- 11: vtunkonwn−1 = √α¯t−1 vt−

√1−α¯tϵθ(vt,t)

√α¯t + 1 − α¯t−1 − σt2 · ϵθ (xt, t) + σtzt

- 12: vt−1 ← m ⊙ vtknown−1 + (1 − m) ⊙ vtunknown−1
- 13: if (T − t + 1) mod L = 0 then
- 14: if m < M then
- 15: vt+L−1 ∼ N ti=+tL−1 αivt−1, 1 − ti=+tL−1 αiI ▷ Noise regret

- 16: t ← t + L − 1
- 17: m ← m + 1
- 18: else
- 19: m ← 0
- 20: end if
- 21: end if
- 22: end while
- 23: voutpainted = v0
- 24: return voutpainted
- 25: end function

and inpainting. MOTIA achieves superior performance compared to previous state-of-the-art methods [6,7,15].

### III Additional Results

We report additional results outpainted by MOTIA. Fig. 10 and Fig. 11 show the longer videos (8 seconds compared to baseline 2 seconds) outpainted by MOTIA. Fig. 12, Fig. 13, Fig. 14 show high resolution videos outpainted by MOTIA.

### IV Demo Video

We provide a demo video, which can be viewed on the anonymous project page or the supplementary video file, showing:

Outpainting results. The results cover videos with various subjects and styles in different resolutions and video lengths, showing the versatile applicability of MOTIA.

###### Baseline comparison. We compare the outpainting results of MOTIA and previous methods in different settings. The results show that MOTIA surpasses previous methods in visual quality, frame consistency, and the harmony of the outpaint scenes in videos.

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

###### Fig. 10: Results of MOTIA on long video outpainting, from 256 × 256 to 512 × 256.

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

###### Fig. 11: Results of MOTIA on long video outpainting, from 256 × 256 to 256 × 512.

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

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

