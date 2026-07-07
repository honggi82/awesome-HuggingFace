## Magic-Me: Identity-Specific Video Customized Diffusion

# arXiv:2402.09368v2[cs.CV]20Mar2024

Ze Ma1, Daquan Zhou1, Xue-She Wang1, Chun-Hsiao Yeh2, Xiuyu Li2, Huanrui Yang2, Zhen Dong2, Kurt Keutzer2, and Jiashi Feng1

- 1 ByteDance Inc.

{ze.ma1, daquanzhou, xueshe.wang, jshfeng}@bytedance.com

- 2 UC Berkeley

{daniel_yeh, xiuyu, huanrui, zhendong, keutzer}@berkeley.edu

Subjects Generated Videos

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

A V* man walking on the street, city, Cinematic Shot

V* man

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

V* woman

A V* woman turning in exquisite armor, cherry blossoms sway in the breeze

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

V* man

A V* man waving in superman costume, in the outer space, stars

Fig. 1: Video Custom Diffusion (VCD) results. With just a few images of a specific identity, the proposed framework can generate temporal consistent videos aligned with the given prompt.

Abstract. Creating content with specified identities (ID) has attracted significant interest in the field of generative models. In the field of textto-image generation (T2I), subject-driven creation has achieved great progress with the identity controlled via reference images. However, its extension to video generation is not well explored. In this work, we propose a simple yet effective subject identity controllable video generation framework, termed Video Custom Diffusion (VCD). With a specified identity defined by a few images, VCD reinforces the identity characteristics and injects frame-wise correlation at the initialization stage for stable video outputs. To achieve this, we propose three novel components that are essential for high-quality identity preservation and stable video generation: 1) a noise initialization method with 3D Gaussian

Noise Prior for better inter-frame stability; 2) an ID module based on extended Textual Inversion trained with the cropped identity to disentangle the ID information from the background 3) Face VCD and Tiled VCD modules to reinforce faces and upscale the video to higher resolution while preserving the identity’s features. We conducted extensive experiments to verify that VCD is able to generate stable videos with better ID over the baselines. Besides, with the transferability of the encoded identity in the ID module, VCD is also working well with personalized text-to-image models available publicly. The codes are available at https://github.com/Zhen-Dong/Magic-Me.

Keywords: Conditional Video Generation · Diffusion Models

### 1 Introduction

Recent advancements in text-to-video (T2V) generation [23–25,56,69,74] have facilitated the creation of realistic animations from text descriptions. However, achieving precise control over the generated identities remains a challenge. In real-world applications, there is often a requirement to render a specific identity guided by text-described contexts, a task known as identity-specific generation [6, 17, 32]. This is crucial in scenarios such as movie production, where a specific character needs to be directed to perform particular actions.

Controlling subject identity in video generation, especially for humans, remains a significant challenge. Previous works on the customization have primarily focused on styles and motions [64] or video editing [35]. While these approaches offer holistic control with conditions such as reference images [40], reference videos [15,64], or depth maps [67], they do not concentrate on identityspecific control. Moreover, as illustrated in the first two rows of Figure 2, the direct integration of image customization modules such as IP-Adapter [70] with the T2V method [19], struggle to generate stable videos with reasonable motions and accurate identity of the identity.

The observed failure cases underscore two key issues specific to humanfocused video customization. 1) In comparison to rigid objects, humans have articulated joints that can display complex poses and movements, such as lifting, walking, and dancing, making it more challenging to maintain stable motions across frames. 2) Human movement generally involves full-body motion, but a person’s distinctive characteristics are often found in the face, which occupies a relatively small area compared to the entire body. In the decoding process of VAEs, the face area on feature maps of the latent space is limited, making it difficult to render the intricate features of the face. Although existing customization methods [17,32,70] can customize close-up face images, they meet difficulties in accurately rendering images with small faces and are not well-suited for videos that feature full-body motions.

In our paper, the main focus is on human-focused identity customization, where the goal is to animate a human identity with diverse motions and scenes and preserve the identity’s characteristics. In the bottom row of Figure 2, our

###### subject

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

IP-Adapter

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

IP-Adapter Face

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

###### VCD(ours)

A <V*> man lifts one handed dumbbell in the gym

- Fig. 2: Comparison between the proposed Video Custom Diffusion (VCD) method3and previous approaches. The first two rows display results using adapters such as IP-Adapter and IP-Adapter Face [70], and the bottom row showcases results from VCD. We observe that body-focused adapters tend to overlook the identity, thus losing alignment, whereas face-focused adapters exhibit artifacts due to conflicts in model parameter spaces when coupled with an off-the-shelf motion module [19], which is discussed in the work [18]. Moreover, the small faces in the images appear plain and lack distinctive identity features. Our method, VCD, successfully generates stabilized videos that preserve distinctive facial characteristics of the identity.

method shows consistent full-body motions with identity’s features, resolving the two aforementioned issues in existing approaches.

To address the first issue, we analyze the exposure bias that leads to error accumulation during inference, and propose a novel 3D Gaussian Noise Prior to reconstruct the correlation across frames. This approach is training-free and enhances consistency in initialization during the inference phase. The covariance among the initialization noises for all frames is controlled by a covariance matrix. Consequently, the motions rendered are more stable in the generated videos.

For the second issue, we first build a robust ID module to balance the tradeoff between preserving identity and aligning the user prompt in the generated videos. Upon analyzing existing image customization methods such as TI [17], LoRA [26], and adapters [70], we observe that appending more parameters to the base model, as in IP-Adapter and LoRA, enhances identity preservation but makes the generated videos more susceptible to overfitting the reference images. In contrast, TI offers the greatest flexibility in generated motions and prompt alignment but exhibits poor identity recovery. To enhance TI’s describing capability, we extend it from a single text token to multiple ones to encode more accurate identity information. To further improve the quality of learned identity, updates for the ID tokens are based on the area of distinct characteristics of the subject, utilizing a prompt-to-segmentation sub-module to more effectively

3 Some of the samples are following the identity images used in [71]

distinguish the identity from the background. Empirical results demonstrate that the proposed ID module based on extended TI achieves a better balance between preserving identity and aligning user prompts.

Equipped with the proposed robust ID module, we integrate the ID module with a 3D Gaussian Noise Prior across three stages of Video Customized Diffusion (VCD): (1) Text-to-Video (T2V) VCD, (2) Face VCD, and (3) Tiled VCD. T2V VCD generates initial videos from the user prompt, incorporating identity characteristics in low resolution. Subsequently, during the Face VCD stage, the faces across all frames are cropped, upscaled, and regenerated with more ID-specific details. Face VCD employs a partial denoising to maintain context consistency around the face as in the initial video. Then the face videos are downscaled and reintegrated into the initial videos. Tiled VCD stage upscales and regenerates the composite video to render both the identity and the background in higher resolution. The resulting videos show more details of identity features while maintaining motion consistency across frames as shown in Figure 1.

The proposed VCD framework introduces a modular approach to ID-specific video generation. The three stages of VCD reuses the same ID module to achieve generation stability in the denoising. Furthermore, based on Stable Diffusion [1], the proposed VCD can replace the base model with any domain-specific finetuned model stemming from the same base, and integrates other conditional inputs such as poses, depths and emotions. This offers valuable flexibility for content generation in AIGC communities such as Civitai [13] and Hugging Face [27], allowing non-technical users to mix and match modules independently.

Our contributions are summarized as follows:

- 1. We introduce a novel framework, Video Custom Diffusion (VCD), dedicated to generating high-quality ID-specific videos in three cascaded stages. VCD demonstrates substantial improvement in aligning generated videos with reference images and user inputs.
- 2. To render stable motions for human identity, we propose a training-free 3D Gaussian Noise Prior for video frame initialization, reconstructing interframe correlation and thereby improving motion consistency.
- 3. To enhance the identity characteristics of human, we integrate a robust ID module with partially denoising processes in Face VCD and Tiled VCD to render more details of identities in higher resolution.

### 2 Related Works

#### 2.1 Subject-Driven Text-to-Image Generation

The progression of T2I diffusion models represents a remarkable stride in image generation, creating both realistic portraits and imaginative depictions of fantastical entities [8,43,46,49]. Recent efforts have spotlighted the customization of these generative models, wherein a pre-trained T2I diffusion model is employed

(2) Face VCD

|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]|
|---|

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]|
|---|

|[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]|
|---|

VAE

[Figure 40]

[Figure 41]

|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>A man, clad in a worn, dark brown cloak with a hood in the desert<br><br>| | | | |
|---|---|---|---|
<br><br><V*><br><br>3D Gaussian Noise Prior<br><br>Motion Module<br><br>VAE|
|---|

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

| |
|---|

2X

VAE

ESRGAN

(1) T2V VCD (3) Tiled VCD

###### ID Module

- Fig. 3: Framework of ID-specific Video Generation. The framework comprises T2V VCD, Face VCD, and Tiled VCD. The basic components ID module and 3D Gaussian Noise Prior are reused in these denoising stages.

alongside a minimal set of customized subject images, aiming to fine-tune the model and learn a unique identifier linked to the desired subject. Pioneering approaches, such as Textual Inversion [17], adapt a token embedding to learn a mapping between token and subject images without altering the model structure, while DreamBooth [47] involves the comprehensive model fine-tuning to learn the concept of subject as well as preserving the capability of general concept generation. This sparked a series of subsequent works, such as NeTI [5], focusing on fidelity and identity preservation of the subject. It further extended to multisubject generation [6,7,18,21,32,37,38,66], where the model is able to jointly learn multiple subjects, and compose them into a single generated image. Previous work [6] introduces the masked loss for image rendering, which we adopt in the sub-module prompt-to-segmentation to further distinguish identities from the background.

#### 2.2 Text-to-Video Generation

Advancing from image generation, T2V appears to be the next breakthrough in the novel applications of generative models. Compared to image generation, video generation is more challenging as it requires high computation costs to maintain long-term spatial and temporal consistency across multiple frames, needs to condition on the vague prompt of short video captioning, and lacks high-quality annotated datasets with video-text pairs. Early explorations utilize GAN and VAE-based methods to generate frames in an auto-regressive manner given a caption [34,42], yet these works are limited to low-resolution videos with simple, isolated motions. The next line of work adopts large-scale transformer architectures for long, HD quality video generations [25,54,56,69], yet suffering from significant training, memory, and computational costs. The recent success of diffusion models leads a new wave of video generation with diffusion-based architectures, with pioneering work like Video Diffusion Models [24] and Imagen Video [23] that introduce new conditional sampling techniques for spatial and temporal video extension. MagicVideo [74] significantly improves generation efficiency by generating video clips in a low-dimensional latent space, which is

later followed by Video LDM [9]. Mask Diffusion augments the cross-attention module to reinforce the text control-ability [75]

#### 2.3 Video Editing

Further advances take more control of the generated video. Tune-a-Video [64,65] allows changing video content while preserving motions by finetuning a T2I diffusion model with a single text-video pair. Text2Video-Zero [29] and Runway Gen [16] propose to combine a trainable motion dynamic module with a pretrained Stable Diffusion, further enabling video synthesis guided by both text and pose/edge/images, without using any paired text-video data. More recently, AnimateDiff [19] animates most of the existing personalized T2I models by distilling reasonable motion priors in the training of a motion module.

#### 2.4 Image Animation

Previous works on image animation mainly focus on extending a static image to a sequence of frames without any change of scene or modifying attributes of the character. Previous works take the subject from an image [14,51–53,55,61,68,73] or a video [20,39,58,59,63], and transfer the motion happened in another video to the subject. Our framework is able not only to animate a given frame but also to modify the attributes of the subject and change the background, all rendered in reasonable motions.

### 3 Preliminaries

Latent Diffusion Model. Our work is based on Stable Diffusion [1], a variant of Latent Diffusion Model (LDM) [46]. In the training, the diffusion model takes an image x0 and a condition c as the input and encodes x0 into a latent code z0 with an image encoder [28, 30, 72]. The latent code z0 is iteratively mixed with Gaussian noise ϵ through forward process, which can be transformed into a closed form:

zt = √α¯tz0 + √1 − α¯tϵ,ϵ ∼ N(0,I), (1)

where α¯t = ti=1 αi,αi ∈ (0,1). The diffusion model is trained to approximate the original data distribution

with a denoising objective:

0,c,ϵ,t[∥ϵθ(zt,c,t) − ϵ∥], (2)

Ez

where ϵθ is the model prediction. In the inference, given the random Gaussian noise initialization zT and condition c, diffusion model performs reverse process for t = T,...,1 to get the encoding of sampled image zˆ0 by the equation:

1 √αt

zˆt−1 =

z ˆt −

where σt = 1−α¯

1−α¯t βt,βt = 1 − αt.

t−1

1 − αt √1 − α¯t

ϵθ(ˆzt,c,t) + σtϵ, (3)

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

###### subject

[Figure 62]

𝛾 = 0

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

- 𝛾 = 0.1
- 𝛾 = 0.2

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

###### A V* man skiing in the snow on the frozen lake, with mountains in the distance A V* dog running on the snowy road, with trees on both sides

- Fig. 4: Influence of 3D Gaussian Noise Prior Covariance γ. We present results for highly-articulated subjects such as humans and animals. As the covariance hyperparameter γ increases, the movements across frames become more stable, and artifacts are suppressed. Setting γ = 0.2 renders the video almost still.

AnimateDiff. We use off-the-shelf motion module AnimateDiff [19] in the framework. This chosen motion module expands the network to encompass the temporal dimension. It transforms 2D convolution and attention layers into temporal pseudo-3D layers [24], adhering to the training objective outlined in Equation 2.

### 4 Method

We propose VCD framework consisting of three stages to enhance the ID features along with a 3D Gaussian Noise Prior to stabilize the human motions across the frames as illustrated in Figure 3. We first analyze the exposure bias in the diffusion model, and proopose 3D Gaussian Noise Prior in Section 4.1. Then we discuss the trade-off between the text-alignment and ID similarity for the existing methods, and propose our ID module that make a better balance in Section 4.2. In the following Section 4.3, we introduce T2V VCD that utilizes the off-the-shelf motion module of AnimateDiff [19], Face VCD to enhance the facial characteristics of the identity, and Tiled VCD that up-scale the videos to higher resolution by tiled denoising.

#### 4.1 3D Gaussian Noise Prior

By comparing Equation 2 with Equation 3, we observe that the inputs to the model ϵθ differ between training and inference stages. Specifically, during training, the model ϵθ(zt,c,t) uses zt that is sampled from an mixture of Gaussian noise and a video, thus zt typically exhibits temporal correlation across the frames. While during inference, the model ϵθ(ˆzt,c,t) receives zˆt that is initialized independently on each frame. The model needs to construct the temporal consistency from the stretch, a process that are not included in the training. Previously, such discrepancy are discussed in the context of exposure bias in both of Natural Language Processing and Computer Vision [41,45,50], and usually leads to accumulated errors in inference.

To mitigate this issue, we propose a training-free 3D Gaussian Noise Prior to reconstruct the temporal correlation in the initialization of zˆT in the inference.

###### Compare TI, LoRA and extended TI

Prompt-to-segmentation

0.35

0.301

0.3

The main subject is a man wearing the pink T-shirt

0.243

0.25

0.189

0.187

[Figure 80]

0.2

0.164

0.159

[Figure 81]

[Figure 82]

0.15

0.1

VAE

0.05

0

DINO CLIP-T

TI LoRA extended TI

<V*> man

Lmask

| | | | |
|---|---|---|---|

(b) Compare LoRA, TI and extended TI.

(a) Illustration of extended TI token training.

- Fig. 5: Extended TI training. Figure (a) shows that the extended TI are optimized against a masked subject area by prompt-to-segmentation. From the Figure (b) we can see extended TI achieves a better balance between the text alignment and identity similarity.

For videos comprising f frames, the 3D Gaussian Noise Prior samples from a Multivariate Gaussian distribution N(0,Σf(γ)). Here, Σf(γ) denotes the covariance matrix parameterized by γ ∈ (0,1).





1 γ γ2 ··· γf−1 γ 1 γ ··· γf−2

γ2 γ 1 ··· γf−3

. (4)

Σf(γ) =

 

 

... . γf−1 γf−2 γf−3 ··· 1

. . .

The covariance described above ensures that the initialized 3D noise exhibits a covariance of γ|m−n| at the same position between the m-th and n-th frames. The hyper-parameter γ represents a trade-off between the stability and magnitude of the motions, as demonstrated in Figure 4. A lower γ value leads to videos with dramatic movements but increased instability, while a higher γ results in more stable motion with reduced motion amplitude.

#### 4.2 ID Module

Although previous works have explored token embedding [17, 57] and network fine-tuning [12,18,32,47,70] for image identity customization, few have delved into identity customization in T2V generation. We find that weight tuning methods such as adapters usually result in frozen characters in the video generation or conflicts with other adapters [19], while LoRA is more prone to overfit the styles of the training images thus lose the alignment of user input, as shown in Figure 6. Interestingly, we find that TI shows the best capability to align with the user prompt, though the depicted character is inferior regarding similarity. We find that using a single token is not enough to encode the necessary information, and training ID tokens based on the entire image would introduce background interference which impede the encoding process. Thus, we make the following improvements on TI to build our ID module, as shown in Figure 5.

###### subject

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

LoRA

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

TI

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Extended TI

A <V*> man on a balcony with a cityscape at night, soft lights twinkling in the distance

- Fig. 6: Comparison of the ID Module [19]4. The first row shows the results using LoRA [26], The middle row displays outcomes TI [17], while the bottom row presents results from extended TI. We observe the LoRA prediction preserves more identity characteristics than TI, but is more overfitted to training images and show inferior user input alignment. With prompt-to-segmentation and sufficient ID tokens, the extended TI shows a better balance between user input alignment and similarity of the identity.

Extended TI with sufficient ID tokens. We find that with more learnable tokens, TI is able to preserve better visual features of identities. This approach, compared to LoRA or adapters, results in better text alignment and identity resemblance. Moreover, the proposed ID module requires only 16KB of storage, a notably compact parameter space compared to the 3.6G required for parameters in Stable Diffusion or 1.7MB for SVDiff [21] which is a compressed version of LoRA.

Prompt-to-segmentation. The interference of background noise in the training is an important issue for concept customization as noted in works [12, 21]. To remove encoded background noise, we propose a straightforward yet robust method: prompt-to-segmentation. We input the prompted class information into Grounding DINO [36] to obtain bounding boxes. These bounding boxes are then fed into SAM [31] to generate the segmentation mask of the subject. During training, we compute the loss only within the mask area.

#### 4.3 Video Customized Diffusion (VCD)

The proposed VCD consists of three stages as shown in 3: (1) T2V VCD, a vanilla T2V process that renders videos in low-resolution, (2) Face VCD, which further enhances the characteristics on the faces, (3) Tiled VCD, that up-scales the videos to higher resolution without any degradation of the identity. All three

stages reuse the same ID module and user prompt to ensure the text-alignment as well as ID similarity.

- (1) T2V VCD In this stage, we first utilize ID module and an off-the-shelf motion module [19] to render a low-resolution videos with the user specified identity. In this process, the frames are all initialized with 3D Gaussian Noise Prior, and the results show that the generated video is well aligned with user input, but the face is blurred, as the area of feature map is small and can’t recover enough details to reflect the intricate characteristics on the face.
- (2) Face VCD. Face VCD regenerates the face with more identity characteristics without changing the video’s resolutions. It first utilizes the prompt-tosegmentation sub-module to detect the bounding box of faces and segment the pixels on the face. Bounding boxes of faces are diluted to include the background around the faces and give more contexts for the ID module. The bounding boxes are then cropped and stacked to make a new face video. Face VCD up-scales the face video, and mixes it with partial Gaussian noise as in [64]. The ID module denoises the mixture of face video to recover more details of the faces. After partially denoising, the regenerated faces have more distinctive characteristics and they are down-scaled to the original resolution so that they can be pasted the initial videos. From the Figure 3 we can see even after the downsampling, the recovered faces in this stage still shows more characteristics and natural existence in the contextual background.
- (3) Tiled VCD. The output after Face VCD is still limited in resolution (512x512). Tiled VCD can further upscale the video while preserving the identity. The videos are first up-scaled to 1024x1024 by ESRGAN [33,60]. However, during the upscaling, ESRGAN has no prior knowledge about the user prompt or the specified identity. Thus, the up-scaled need to be denoised using ID module again. We segmented the up-scaled video into 4 tiles, each of them occupying 512x512 pixels as the original video. Then we add partial 3D Gaussian Noise Prior to the video for ID module to denoise. Each tile is partially denoised with the same ID module, with the details of identity rendered in higher resolution, and stitched together. Both of the identity and background are more clear in this stage.

### 5 Experiments

In this section, we presents the experimental evaluation of our method. We begin with an overview of the implementation details. Following this, we present qualitative results demonstrating the animation of various identities, controllable generation, as well as quantitative results. Finally, we conclude with ablation studies.

#### 5.1 Implementation Details

Training. Unless specified otherwise, the ID module is trained using Stable Diffusion 1.5 and employed with Realistic Vision during inference. Applying it directly

[Figure 98]

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

###### Subject

Cropped Face in T2V VCD

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

Cropped Face in Face VCD

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

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Cropped Face in Tiled VCD

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

T2V VCD

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Face VCD

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Tiled VCD

- Fig. 7: VCD Qualitative Results. We display the zoomed-in faces after each stage. From the figure, it is evident that the faces are significantly enhanced with the identity’s characteristics following Face VCD and Tiled VCD interventions. The prompt used is a V* man floating in a Superman costume, in outer space, among stars.

to Stable Diffusion 1.5 for video generation coupled with AnimateDiff [19] results in inferior videos. We use 4 text tokens for extended TI. We set the learning rates for extended token tokens at 1e-3. The batch size is fixed at 4. Each identity’s ID module undergoes 200 optimization steps during training. For the motion module, we adjust the γ in Equation 4 to 0.1. We denoise 80% in the Face VCD while 20% in Tiled VCD.

Datasets. To validate the effectiveness of our proposed VCD framework, we collected training images of 24 characters from CustomConcept101 [32] as well as from the internet, ensuring a diverse representation of humans and animals. For each subject, we tasked GPT-4V with creating 25 prompts for animations against various backgrounds. For evaluation purposes, the model generates four videos for each prompt, using different random seeds. This process results in a total of 2400 videos.

Evaluation Metrics. We evaluate the generated videos from three perspectives. (i) Identity alignment: The visual appearance of the generated identity should align with that in the reference images. We utilize CLIP-I [44] and DINO [10] to compute the similarity score between each pair of video frames and reference images. (ii) Text-alignment: The text-image similarity score is calculated in the CLIP feature space [22]. (iii) Temporal-smoothness: We assess temporal consistency in the generated videos by computing CLIP scores for all pairs of consecutive video frames. It’s important to note that temporal smoothness is influenced not only by the content consistency between consecutive frames but also by the motion’s magnitude. Therefore, it is important to consider text alignment, identity similarity, and temporal smoothness together when comparing results.

Baselines. Due to the lack of identity-specific T2V methods, we compare our method with direct composition of concept customization methods with AnimateDiff, such as CustomDiffusion [32], Textual Inversion (TI) [17], IP-Adapter [70], and LoRA [26,48], all combined with a 3D Gaussian Noise Prior. While recent advancements have introduced more novel customization methods for multiidentity customization, such as those found in [11, 18, 21, 62], the integration with such methods could be left for future work.

#### 5.2 Qualitative Results

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

V* man, in the forest, with sharp eyes

V* man

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

V* dog

V* dog on the coast, waves, wind, wind, tree, sky

- Fig. 8: Compatibility with Personalized Base Models. We list results from Realist Vision [3], ToonYou [4] and RCNZ Cartoon 3D [2] for each subject.

- Table 1: Quantitative comparison with baseline models. The best score is highlighted in bold. The symbol ↑ indicates that a higher score implies greater relevance. The proposed ID module achieves an optimal balance between video consistency and image alignment. SD: Stable Diffusion, RV: Realistic Vision, TI: Textual Inversion, DB: DreamBooth, IA Face: IP-Adapter Face.

SD [1] RV [3] TI [17] LoRA [26] DB [47] IA Face [70] Ours

DINO ↑ 0.267 0.345 0.352 0.522 0.530 0.455 0.585 CLIP-I ↑ 0.595 0.624 0.701 0.722 0.699 0.610 0.747 CLIP-T↑ 0.265 0.275 0.271 0.238 0.222 0.250 0.315 CLIP-I Temp Consist↑ 0.729 0.758 0.796 0.810 0.734 0.771 0.835

- Table 2: Influence of 3D Gaussian Noise Prior. We experiment with different level covariance in 3D Gaussian Noise Prior, with higher gamma, video temporal consistency improves while text alignment is decreasing. When gamma is higher than 0.2, the differences between frames are minimal, as illustrated in Figure 4.

CLIP-I Temp Consist↑ γ = 0 0.560 0.745 0.310 0.797

DINO↑ CLIP-I↑ CLIP-T↑

- γ = 0.1 0.585 0.747 0.315 0.835
- γ = 0.2 0.583 0.816 0.280 0.861

In this section, we first present the generated results, highlighting zoomed-in cropped faces in Figure 7 at each stage to showcase the effectiveness of our proposed framework. We observe that after the initial text-to-video (T2V) stage, faces appear plain and lack distinguishing features. However, following the Face VCD stage, even without any resolution changes on the displayed faces, they begin to closely resemble the specified identity. After Tiled VCD, which upscales the video clip as a whole, faces are rendered with enhanced details. Furthermore, we demonstrate that our proposed VCD framework and ID module can effectively render identities on different personalized models derived from the same SD1.5, as shown in Figure 8.

#### 5.3 Quantitative Results

We present the quantitative results in Table 1. Initially, we evaluate two pretrained models: Stable Diffusion (SD) and Realistic Vision. Realistic Vision, a community-developed model fine-tuned on SD, shows promising results in generating realistic images. As indicated in Table 1, Realistic Vision generally outperforms SD, leading us to adopt it as the base model when possible. However, for models like DreamBooth, which involve fine-tuning all weights in the UNet, replacing the base model weights is not feasible. Its performance is generally inferior to the others, highlighting the limitations of extensive fine-tuning. We also compare with LoRA, TI and IP-Adapter Face in the table. As vanilla IPAdapter model can only animate the image and can’t align with user input, we compare with IP-Adapter Face here.

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

subject A <V*>man boxing in the forest, waterfall

- Fig. 9: VCD with ControlNet. Our method can be integrated with conditional inputs such as depth maps and poses using ControlNet to achieve controllable video generation.

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

A <V*>man turning head in the office

0:15 frames <Angry*>

0:15 frames <Laugh*>

0:7 frames <Angry*> and 8:15 frames <Laugh*>

[Figure 290]

[Figure 291]

[Figure 292]

Subject

- Fig. 10: VCD with Controllable Emotions. To demonstrate the compatibility of the VCD framework, we present results featuring controllable emotions on human faces using embedding. In the first two rows, videos are generated with a consistent prompt, while the last row displays a transition of emotions, employing an angry motion for the first 8 frames followed by a laughing emotion for the remaining 8 frames.

#### 5.4 Controllable Generation

We demonstrate that our framework supports controllable generations, as illustrated in Figure 9 and Figure 10. In Figure 9, where vanilla VCD faces challenges in generating sudden movements such as boxing, we utilize ControlNet with reference motion to accurately render the identity in the desired poses. Furthermore, traditional conditional inputs fail to control characters’ expressions, a critical aspect for applications in the film industry. In Figure 10, we present controllable emotion generation and employ multi-prompt techniques across frames to facilitate the transition of emotions.

- Table 3: Ablation study for three stages of VCD. From the table, we can see each stage of VCD has increased the similarity between generated identities and reference ones as indicated by DINO.

CLIP-I Temp Consist↑

DINO↑ CLIP-I↑ CLIP-T↑

After T2V VCD 0.539 0.721 0.321 0.826 After Face VCD 0.551 0.752 0.322 0.825 After Tiled VCD 0.585 0.747 0.315 0.835

#### 5.5 Ablation Study

To validate the impact of proposed 3D Gaussian Noise Prior, we conduct a detailed ablation study as shown in Table 2. In the study, 3D Gaussian Noise Prior is crucial for video smoothness. Increasing gamma bring more temporal consistency while losing the text alignment. We conduct ablation study for three stages of VCD as in Table 3, and find that each stage could improve the identity similarity.

### 6 Limitations and Future Works

Our proposed framework has several areas to improve. First, it struggles when we try to make videos with several different identities, each with its own special token embedding. The resulting video degrades especially when these characters have to interact with each other. Second, the proposed frameworks are limited by the capacity of the motion module. Given the motion module that only generates short-time videos, it’s not easy to extend the length of the videos while keeping the same consistency and fidelity. We need to work on making the system capable of handling multiple identities that interact with each other, and ensuring it can keep up the quality in longer videos.

### 7 Conclusion

In this paper, we introduces Video Custom Diffusion (VCD), a framework designed for subject identity controllable video generation. By focusing on the encoding of identity information and frame-wise correlation, VCD paves the way for producing videos that not only maintain the subject’s identity across frames but do so with stability and clarity. Our novel contributions, including the ID module for precise identity disentanglement, the T2V VCD module for enhanced frame consistency, and the Face VCD and Tiled VCD modules for improved video quality, collectively establish a new standard for identity preservation in video content. The extensive experiments we conducted affirm VCD’s superiority over existing methods in generating high-quality videos that preserve the subject’s identity. Furthermore, the adaptability of our ID module to work with existing text-to-image models and controllable generations enhances VCD’s practicality, making it versatile for a broad range of applications.

### References

- 1. Stable diffusion. https://huggingface.co/runwayml/stable-diffusion-v1-5

(2022)

- 2. Rcnz cartoon 3d v1.0. https://civitai.com/models/66347?modelVersionId= 71009 (2023)
- 3. Realistic vision v5.1. https://civitai.com/models/4201/realistic-vision-v51

(2023)

- 4. Toonyou beta 3. https://civitai.com/models/30240?modelVersionId=78775

(2023)

- 5. Alaluf, Y., Richardson, E., Metzer, G., Cohen-Or, D.: A neural space-time representation for text-to-image personalization. arXiv preprint arXiv:2305.15391 (2023)
- 6. Avrahami, O., Aberman, K., Fried, O., Cohen-Or, D., Lischinski, D.: Break-ascene: Extracting multiple concepts from a single image. In: SIGGRAPH Asia 2023 Conference Papers. SA ’23, Association for Computing Machinery, New York, NY, USA (2023). https://doi.org/10.1145/3610548.3618154, https://doi.org/ 10.1145/3610548.3618154
- 7. Bai, J., Dong, Z., Feng, A., Zhang, X., Ye, T., Zhou, K., Shou, M.Z.: Integrating view conditions for image synthesis. arXiv preprint arXiv:2310.16002 (2023)
- 8. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Lia, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., Manassra, W., Dhariwal, P., Chu, C., Jiao, Y., Ramesh, A.: Improving image generation with better captions. https://cdn.openai.com/ papers/dall-e-3.pdf (2023)
- 9. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22563–22575 (2023)
- 10. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)
- 11. Chen, H., Wang, X., Zeng, G., Zhang, Y., Zhou, Y., Han, F., Zhu, W.: Videodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning (2023)
- 12. Chen, H., Zhang, Y., Wang, X., Duan, X., Zhou, Y., Zhu, W.: Disenbooth: Identitypreserving disentangled tuning for subject-driven text-to-image generation. arXiv preprint arXiv:2305.03374 (2023)
- 13. Civitai: Civitai. https://civitai.com/ (2022)
- 14. Dorkenwald, M., Milbich, T., Blattmann, A., Rombach, R., Derpanis, K.G., Ommer, B.: Stochastic image-to-video synthesis using cinns. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3742– 3753 (2021)
- 15. Esser, P., Chiu, J., Atighehchian, P., Granskog, J., Germanidis, A.: Structure and content-guided video synthesis with diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7346–7356 (October 2023)
- 16. Esser, P., Chiu, J., Atighehchian, P., Granskog, J., Germanidis, A.: Structure and content-guided video synthesis with diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7346–7356 (2023)
- 17. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)

- 18. Gu, Y., Wang, X., Wu, J.Z., Shi, Y., Chen, Y., Fan, Z., Xiao, W., Zhao, R., Chang, S., Wu, W., et al.: Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292

(2023)

- 19. Guo, Y., Yang, C., Rao, A., Wang, Y., Qiao, Y., Lin, D., Dai, B.: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023)
- 20. Ha, S., Kersner, M., Kim, B., Seo, S., Kim, D.: Marionette: Few-shot face reenactment preserving identity of unseen targets. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 10893–10900 (2020)
- 21. Han, L., Li, Y., Zhang, H., Milanfar, P., Metaxas, D., Yang, F.: Svdiff: Compact parameter space for diffusion fine-tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7323–7334 (October 2023)
- 22. Hessel, J., Holtzman, A., Forbes, M., Bras, R.L., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 (2021)
- 23. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022)
- 24. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. arXiv:2204.03458 (2022)
- 25. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868

(2022)

- 26. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)
- 27. HuggingFace: Huggingface. https://huggingface.co/ (2022)
- 28. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1125–1134 (2017)
- 29. Khachatryan, L., Movsisyan, A., Tadevosyan, V., Henschel, R., Wang, Z., Navasardyan, S., Shi, H.: Text2video-zero: Text-to-image diffusion models are zeroshot video generators. arXiv preprint arXiv:2303.13439 (2023)
- 30. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)
- 31. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.: Segment anything. arXiv:2304.02643 (2023)
- 32. Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept customization of text-to-image diffusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1931–1941 (2023)
- 33. Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z., et al.: Photo-realistic single image superresolution using a generative adversarial network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4681–4690 (2017)
- 34. Li, Y., Min, M., Shen, D., Carlson, D., Carin, L.: Video generation from text. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)
- 35. Liew, J.H., Yan, H., Zhang, J., Xu, Z., Feng, J.: Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749 (2023)

- 36. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Li, C., Yang, J., Su, H., Zhu, J., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499 (2023)
- 37. Liu, Z., Zhang, Y., Shen, Y., Zheng, K., Zhu, K., Feng, R., Liu, Y., Zhao, D., Zhou, J., Cao, Y.: Cones 2: Customizable image synthesis with multiple subjects. arXiv preprint arXiv:2305.19327 (2023)
- 38. Ma, J., Liang, J., Chen, C., Lu, H.: Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410 (2023)
- 39. Ni, H., Liu, Y., Huang, S.X., Xue, Y.: Cross-identity video motion retargeting with joint transformation and synthesis. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 412–422 (January 2023)
- 40. Ni, H., Shi, C., Li, K., Huang, S.X., Min, M.R.: Conditional image-to-video generation with latent flow diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18444–18455 (2023)
- 41. Ning, M., Sangineto, E., Porrello, A., Calderara, S., Cucchiara, R.: Input perturbation reduces exposure bias in diffusion models. arXiv preprint arXiv:2301.11706

(2023)

- 42. Pan, Y., Qiu, Z., Yao, T., Li, H., Mei, T.: To create what you tell: Generating videos from captions. In: Proceedings of the 25th ACM international conference on Multimedia. pp. 1789–1798 (2017)
- 43. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)
- 44. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021)
- 45. Ranzato, M., Chopra, S., Auli, M., Zaremba, W.: Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732 (2015)
- 46. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 47. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22500–22510 (2023)
- 48. Ryu, S.: Low-rank adaptation for fast text-to-image diffusion fine-tuning (2023)
- 49. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022)
- 50. Schmidt, F.: Generalization in generation: A closer look at exposure bias. In: Birch, A., Finch, A., Hayashi, H., Konstas, I., Luong, T., Neubig, G., Oda, Y., Sudoh, K. (eds.) Proceedings of the 3rd Workshop on Neural Generation and Translation. pp. 157–167. Association for Computational Linguistics, Hong Kong (Nov 2019). https://doi.org/10.18653/v1/D19-5616, https://aclanthology.org/D195616

- 51. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N.: Animating arbitrary objects via deep motion transfer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2377–2386 (2019)
- 52. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N.: First order motion model for image animation. In: Conference on Neural Information Processing Systems (NeurIPS) (December 2019)
- 53. Siarohin, A., Woodford, O., Ren, J., Chai, M., Tulyakov, S.: Motion representations for articulated animation. In: CVPR (2021)
- 54. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without textvideo data. arXiv preprint arXiv:2209.14792 (2022)
- 55. Tao, J., Wang, B., Xu, B., Ge, T., Jiang, Y., Li, W., Duan, L.: Structure-aware motion transfer with deformable anchor model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3637–3646 (2022)
- 56. Villegas, R., Babaeizadeh, M., Kindermans, P.J., Moraldo, H., Zhang, H., Saffar, M.T., Castro, S., Kunze, J., Erhan, D.: Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399 (2022)
- 57. Voynov, A., Chu, Q., Cohen-Or, D., Aberman, K.: P+: Extended textual conditioning in text-to-image generation (2023)
- 58. Wang, T.C., Liu, M.Y., Tao, A., Liu, G., Kautz, J., Catanzaro, B.: Few-shot video-to-video synthesis. In: Advances in Neural Information Processing Systems (NeurIPS) (2019)
- 59. Wang, T.C., Liu, M.Y., Zhu, J.Y., Liu, G., Tao, A., Kautz, J., Catanzaro, B.: Video-to-video synthesis. In: Advances in Neural Information Processing Systems (NeurIPS) (2018)
- 60. Wang, X., Xie, L., Dong, C., Shan, Y.: Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1905–1914 (2021)
- 61. Wang, Y., Yang, D., Bremond, F., Dantcheva, A.: Latent image animator: Learning to animate images via latent space navigation. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id= 7r6kDq0mK_
- 62. Wang, Z., Li, A., Xie, E., Zhu, L., Guo, Y., Dou, Q., Li, Z.: Customvideo: Customizing text-to-video generation with multiple subjects. arXiv preprint arXiv:2401.09962 (2024)
- 63. Wiles, O., Koepke, A., Zisserman, A.: X2face: A network for controlling face generation using images, audio, and pose codes. In: Proceedings of the European conference on computer vision (ECCV). pp. 670–686 (2018)
- 64. Wu, J.Z., Ge, Y., Wang, X., Lei, S.W., Gu, Y., Shi, Y., Hsu, W., Shan, Y., Qie, X., Shou, M.Z.: Tune-a-video: One-shot tuning of image diffusion models for textto-video generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7623–7633 (2023)
- 65. Wu, J.Z., Li, X., Gao, D., Dong, Z., Bai, J., Singh, A., Xiang, X., Li, Y., Huang, Z., Sun, Y., He, R., Hu, F., Hu, J., Huang, H., Zhu, H., Cheng, X., Tang, J., Shou, M.Z., Keutzer, K., Iandola, F.: Cvpr 2023 text guided video editing competition. arXiv preprint arXiv: 2310.16003 (2023)
- 66. Xiao, G., Yin, T., Freeman, W.T., Durand, F., Han, S.: Fastcomposer: Tuningfree multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431 (2023)

- 67. Xing, J., Xia, M., Liu, Y., Zhang, Y., Zhang, Y., He, Y., Liu, H., Chen, H., Cun, X., Wang, X., et al.: Make-your-video: Customized video generation using textual and structural guidance. arXiv preprint arXiv:2306.00943 (2023)
- 68. Xu, B., Wang, B., Tao, J., Ge, T., Jiang, Y., Li, W., Duan, L.: Move as you like: Image animation in e-commerce scenario. In: Proceedings of the 29th ACM International Conference on Multimedia. pp. 2759–2761 (2021)
- 69. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021)
- 70. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models (2023)
- 71. Yu, J., Wang, Y., Zhao, C., Ghanem, B., Zhang, J.: Freedom: Training-free energyguided conditional diffusion model. arXiv preprint arXiv:2303.09833 (2023)
- 72. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018)
- 73. Zhao, J., Zhang, H.: Thin-plate spline motion model for image animation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3657–3666 (2022)
- 74. Zhou, D., Wang, W., Yan, H., Lv, W., Zhu, Y., Feng, J.: Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018 (2022)
- 75. Zhou, Y., Zhou, D., Zhu, Z.L., Wang, Y., Hou, Q., Feng, J.: Maskdiffusion: Boosting text-to-image consistency with conditional mask. arXiv preprint arXiv:2309.04399

(2023)

