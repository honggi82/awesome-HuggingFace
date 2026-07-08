# arXiv:2504.21650v2[cs.CV]13May2025

## HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation

Haiyang Zhou∗

Wangbo Yu∗

Jiawen Guan

zhouhaiyang000@gmail.com School of Electronic and Computer Engineering, Peking University Shenzhen, China

yuwangbo98@gmail.com School of Electronic and Computer Engineering, Peking University Peng Cheng Laboratory Shenzhen, China

3291495912@qq.com Harbin Institute of Technology, Shenzhen Shenzhen, China

Yonghong Tian†

Li Yuan†

Xinhua Cheng

chengxinhua@stu.pku.edu.cn School of Electronic and Computer Engineering, Peking University Shenzhen, China

yhtian@pku.edu.cn School of Electronic and Computer Engineering, Peking University Peng Cheng Laboratory Shenzhen, China

yuanli-ece@pku.edu.cn School of Electronic and Computer Engineering, Peking University Peng Cheng Laboratory Shenzhen, China

### Pipeline

[Figure 1]

Panorama

[Figure 2]

Time Slice

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

- View1
- View2
- View3

[Figure 10]

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

4D Scene

Figure 1: HoloTime accepts either a user-provided or model-generated panoramic image as input, and transforms it into an immersive 360-degree 4D scene, enabling a virtual roaming experience.

思想自由 兼容并包

Abstract

assets, enabling a fully immersive 4D experience for users. Specifically, to tame video diffusion models for generating high-fidelity panoramic videos, we introduce the 360World dataset, the first comprehensive collection of panoramic videos suitable for downstream 4D scene reconstruction tasks. With this curated dataset, we propose Panoramic Animator, a two-stage image-to-video diffusion model that can convert panoramic images into high-quality panoramic videos. Following this, we present Panoramic SpaceTime Reconstruction, which leverages a space-time depth estimation method to transform the generated panoramic videos into 4D point clouds, enabling the optimization of a holistic 4D Gaussian Splatting representation to reconstruct spatially and temporally consistent 4D scenes. To validate the efficacy of our method, we conducted a comparative analysis with existing approaches, revealing its superiority

The rapid advancement of diffusion models holds the promise of revolutionizing the application of VR and AR technologies, which typically require scene-level 4D assets for user experience. Nonetheless, existing diffusion models predominantly concentrate on modeling static 3D scenes or object-level dynamics, constraining their capacity to provide truly immersive experiences. To address this issue, we propose HoloTime, a framework that integrates video diffusion models to generate panoramic videos from a single prompt or reference image, along with a 360-degree 4D scene reconstruction method that seamlessly transforms the generated panoramic video into 4D

∗Both authors contributed equally to this research. †Corresponding authors.

in both panoramic video generation and 4D scene reconstruction. This demonstrates our method’s capability to create more engaging and realistic immersive environments, thereby enhancing user experiences in VR and AR applications. More results are available on the project page: https://zhouhyocean.github.io/holotime/.

###### Keywords

Video Diffusion Model, 4D Scene Generation, Panoramic Video Generation, Panoramic Video Depth Estimation

###### 1 Introduction

The advancement of diffusion models for text-to-image generation [13, 41] has sparked a burgeoning interest in video and 3D content creation, leading to significant milestones. Building upon the foundation of 3D content generation, 4D content generation—which integrates the temporal dimension into 3D data—holds immense potential for applications and promises to revolutionize the fields of Augmented Reality (AR) and Virtual Reality (VR). Scene-level 4D content generation has the potential to provide users with a more liberating and immersive experience, allowing for greater freedom in virtual navigation. However, unlike the advancements in video and 3D content generation, 4D content generation has lagged behind due to the scarcity of high-quality annotated data, particularly for large-scale 4D scenes. Consequently, current 4D generation methods are primarily limited to generating object-level dynamics [1, 17, 27, 39, 59] or creating forward-facing scenes with restricted viewing freedom [55].

To cultivate a 360-degree immersive experience, an effective strategy is to harness the 360° field of view provided by panoramic images. Whileestablishedmethods[32,71,72]already utilizepanoramic images for 3D generation, introducing dynamics to these images for 4D generation is not straightforward due to the lack of large-scale panoramic video training data. To address these challenges, we introduce HoloTime, a 360-degree 4D scene generation framework. It accepts a user-provided or model-generated [21] panoramic image as input, and leverage a Panoramic Animator to transform the panoramic image into a panoramic video featuring realistic visual dynamics and a fixed camera pose. Following this, we present a Panoramic Space-Time Reconstruction technique that reconstructs a high-fidelity 4D scene based on the generated video, thereby enabling an immersive user experience. To build the framework, we first curated a large-scale dataset of fixed-camera panoramic videos, accompanied by corresponding prompt descriptions, which we term the 360World dataset. We then train the Panoramic Animator on this dataset to convert panoramic images into panoramic videos, employing a two-stage motion-guided generation process to enhance its performance. After generating the panoramic video, we utilize the Panoramic Space-Time Reconstruction technique for high-fidelity 4D scene reconstruction. Specifically, we begin by employing a panoramic optical flow estimation model [43] in conjunction with a narrow field-of-view depth estimation model [18] to perform space-time depth estimation, ensuring depth consistency across both temporal and spatial dimensions. This aligned depth is used to convert the panoramic video into dynamic point clouds,

optimizing a spatially and temporally consistent 4D-GS representation, which ultimately provides an immersive experience in virtual and augmented reality environments.

Our contributions are encapsulated as follows:

- • We present Panoramic Animator, accompanied by a twostage motion-guided generation strategy that seamlessly transformspanoramicimagesintodynamic panoramic videos, all while preserving the spatial characteristics of the original images, facilitating downstream 4D reconstruction tasks.
- • We introduce Panoramic Space-Time Reconstruction, which employs cutting-edge techniques to enable the temporal and spatial alignment of depth estimation for panoramic videos, enabling a holistic 4D scene reconstruction through the 4DGS representations.
- • We contribute the 360World dataset, the first comprehensive collection of panoramic videos captured with fixed cameras. We believe this dataset can not only helps address gaps in 360-degree 4D scene generation but also holds promise for advancing future 4D generation efforts.

2 Related Works

- 2.1 Diffusion-based Image and Video Generation.

As supported by previous works [13, 34, 38, 41, 42], diffusion models have markedly enhanced the capacity to generate 2D images. The application of diffusion models also extends to the domain of video generation. VDM [14] first introduced the space-time factorized U-Net architecture, extending the original 2D U-Net for video modeling and generation. Subsequently, many methods[5, 10, 12, 45, 50, 70] have followed this similar architecture, achieving significant progress in generating high-quality video content from text. Some models [4, 56, 69] utilize image control for video generation to achieve a higher degree of customization and consistency in the content produced. Recent studies [20, 31, 58] have employed the DiT (Diffusion Transformer) architecture for video generation, also achieving impressive performance. Although current video generation is largely confined to perspective videos, 360DVD [51] has conducted explorations to panoramic video text-driven generation through training an adapter. Furthermore, Imagine360 [46] is capable of lifting input perspective videos to panoramic videos.

- 2.2 3D Scene Generation.

The emergence of NeRF [33] and 3D Gaussian Splatting (3D-GS) [19] has propelled significant advancements in the field of 3D reconstruction [3, 9, 47, 49, 62, 63]. Additionally, there is growing exploration into leveraging 2D diffusion models to facilitate 3D generation. Early works [7, 26, 37, 48, 53, 65, 68] employ the Score Distillation Sampling (SDS) [36] to distill 3D representations from 2D diffusion models. However, most of these efforts focus on objectlevel generation, and multiple iterations of distillation could lead to excessive saturation. Some works [8, 35, 60, 61] are based on outpainting to generate larger-scale scenes and conduct lifting from 2D to 3D using depth estimation models. To achieve more immersive virtual roaming, many methods [32, 57, 71, 72] choose panoramic

images with a spherical field of view (FoV) as the scene representation. However, these methods can only create static 3D scenes without any motion, restricting their vibrancy.

###### 2.3 4D Scene Generation.

The achievements in video generation and 3D generation have indeed spurred the exploration of 4D generation tasks. Many objectlevel generation works [1, 17, 27, 39, 59] leverage video diffusion models and multi-view diffusion models [29, 30, 44] that have 3D aware ability for optimization or reconstruction of 4D objects. CAT4D [55] introduces a more unified multi-view video diffusion model for limited-scale 4D scene generation. VividDream [22] employs an outpainting method to expand the scene scale, but it suffers from artifacts. For the generation of panoramic 4D scenes, 4K4DGen [23] proposes a novel denoising method to adapt the perspective video diffusion model for panoramic video generation. However, the denoising of multiple small fixed window leads to restricted motion ranges. DynamicScaler [28] improves this by continuously shifting the angles of the windows during the denoising process. In contrast, our method directly animate the entire panoramic image, allowing for a more significant range of motion and better global consistency in the panoramic space.

3 Methodology

This section begins with a concise overview of the Diffusion Model in Sec. 3.1. Subsequently, we introduce Panoramic Animator and its techniques in Sec. 3.2 and delve into the process of conducting Panoramic Space-Time Reconstruction in Sec. 3.3. Ultimately, the explanation of 360World dataset is presented in Sec. 3.4. The overall framework of our method is illustrated in Fig. 2.

###### 3.1 Preliminaries

- 3.1.1 Diffusion Model. Diffusion models [13] typically consist of two processes: forward diffusion process 𝑞 and reverse denoising

process 𝑝𝜃. Given an input signal x0 ∼ (𝑞0, x0), the forward process gradually introduces noise to x0. As the time steps increase, the level of noise gradually intensifies, ultimately resembling a Gaussian noise. This could be defined as follows:

x𝑡 = Q(x0,𝑡) = 𝛼𝑡x0 + 𝜎𝑡𝜖 (1)

where 𝑡 = 1, . . .,𝑇 and 𝜖 ∼ N(0, I). The hyper-parameters 𝛼𝑡 and 𝜎𝑡 satisfy the equation 𝛼𝑡2 + 𝜎𝑡2 = 1. The reverse denoising process 𝑝𝜃 aims to optimize a noise predictor 𝜎𝜃 to effectively remove noise, using the following loss function:

L = E𝑡∼U(0,1),𝜖∼N(0,I) ∥𝜖 − 𝜖𝜃 (x𝑡,𝑡)∥22 (2) Latent Diffusion Models [41] enhance computational efficiency by employing a pre-trained VAE. Initially, the clean data 𝑥0 is encoded into latent code using the VAE encoder E. The diffusion and denoising processes are then performed in the latent space. Finally, the denoised latent code is decoded back to the original space using the VAE decoder D. In video diffusion models, the parameter 𝜖𝜃 typically employs either a U-Net architecture [4, 10, 45, 70] or a Transformer [20, 31, 58]. Considering that most open-source Imageto-Video (I2V) models currently use the U-Net architecture, we adopt this architecture in our study. We build our panoramic video

generation method based on DynamiCrafter [56], which effectively produces rich dynamic effects from input images.

###### 3.2 Panoramic Animator

Leveraging advanced I2V models, , we propose Panoramic Animator, a novel method composed of three proposed mechanisms for generating panoramic videos from panoramic images. we introduce Hybrid Data Fine-tuning (HDF) in Sec. 3.2.1 and Two-stage Motion Guided Generation (MGG) in Sec. 3.2.2 to improve the model’s performance, along with Panoramic Circular Techniques (PCT) in Sec. 3.2.3 to enhance the visual effects of panoramic videos.

- 3.2.1 Hybrid Data Fine-tuning. Due to the significant distribution differences between general and panoramic videos, we avoid directly fine-tuning with panoramic video data to preserve the effective temporal priors of the pretrained video model. We incorporate additional video data for hybrid data fine-tuning. Landscape timelapse videos exhibit significant motion. While typically recorded with perspective cameras, they share semantic and temporal similarities with panoramic videos, effectively bridging the gap between panoramic and standard videos in spatial-temporal data distribution and improving the model’s generalization capabilities. We choose the ChronoMagic-Pro dataset [66] for supplementation of training data. By searching for the keyword “landscape” in the annotated text, we filtered out irrelevant videos and collected 4, 455 relevant text-to-video pairs. These were then randomly mixed with our 360World dataset to create a hybrid dataset.
- 3.2.2 Two-Stage Motion Guided Generation. Panoramic videos offer a spherical viewing angle that contains a wealth of spatial information, typically showcasing localized fine motion rather than global large-scale motion. When training models with the same architecture and training data at varying resolutions, we find that the model prioritizes learning temporal information at lower resolutions and spatial information at higher resolutions. Therefore, we propose a two-stage motion-guided generation method: firstly, we generate a low-resolution coarse video that offers global motion guidance, and then generate refinement video with higher resolution.

We fine-tune the guidance model Mg and refinement model Mr

based on the pre-trained DynamiCrafter. During training, we finetune the spatial layers of the guidance model Mg using 360World dataset solely at low resolution, while keeping the temporal layers frozen. Subsequently, we fine-tune all the layers of the refinement model Mr using hybrid dataset at high resolution. In the inference phase, given the input panoramic image I and other optional conditions, like prompt, the guidance model Mg first generate a coarse video Vg from random latent Gaussian noise z𝑇g . This video Vg exhibits significant global motion aligned with the panoramic spatial features. We upscale Vg to high resolution Vg′ using a superresolution model. Subsequently, We encode the video Vg′ into latent code and add noise to 𝐾 (𝐾 < 𝑇) time steps. This could be represented by the following formula:

z𝐾r = Q(E(Vg′ ),𝐾) (3) We utilize the refinement model Mr for secondary denoising

process, using user input as conditional control to enhance local

[Figure 24]

###### Panoramic Animator

[Figure 25]

Panoramic Image Input T

[Figure 26]

###### Text Prompt

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

###### A cartoon-style cloud city where floating buildings drift among fluffy clouds as airships pass by.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

ℰ

[Figure 42]

Add Noise

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

[Figure 60]

[Figure 61]

Upscale

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

K

[Figure 70]

[Figure 71]

[Figure 72]

Upscale

[Figure 73]

Panoramic Space-Time Reconstruction

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Optical Flow

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

###### Time

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Space-Time Depth Estimation

4D Gaussians

思想自由 兼容并包

Figure 2: Overview of HoloTime. Beginning with a user-provided or model-generated panoramic image as input, we first use the Panoramic Animator to generate panoramic videos in two stages. The guidance model generates a coarse video in the first stage, which is then refined by the refinement model in the second stage, creating the final panoramic video for 4D reconstruction. Subsequently, we perform Panoramic Space-Time reconstruction to lift the panoramic video to a 4D scene. We employ optical flow for space-time depth estimation to achieve spatial and temporal alignment, thus obtaining a 4D initialized point cloud. Finally, we employ a 4D-GS method for the final scene reconstruction representation.

motion details, resulting in the final video Vr. This ensures that the generated video exhibits a strong dynamic effect on a global scale while also handling local details well.

- 3.2.3 Panoramic Circular Techniques. Horizontal end continuity is crucial for panoramic videos, influencing the user’s seamless experience in subsequent 4D reconstructions. To achieve this, we create repeated sections at the left and right ends of the panoramic video and conduct blending after each denoising step throughout the generation process for complete continuity. Specifically, during inference process, A section from the left end of the reference image I is first copied and concatenated to the right end. After each denoising step, the left part of the corresponding latent code is blended into the right part, and thereafter, the right part is blended into the left part, continuing this alternating mixing. Following 360DVD [51], We also modify the padding operation of the convolutional

layers during denoising process of refinement model Mr by applying code at suitable positions to pad the left and right boundaries, ensuring pixel-level continuity. After generating the final video, the repeated part is cropped to yield the final seamless continuous panoramic video.

###### 3.3 Panoramic Space-Time Reconstruction

In the 4D lifting phase, we first introduce Space Aligned Depth Estimation for single panoramic image in Sec. 3.3.1, and then extend it to Space-Time Depth Estimation for panoramic video in Sec. 3.3.2. Finally, we employ a 4D-GS representation method to complete the final 4D scene reconstruction in Sec. 3.3.3.

3.3.1 Space Aligned Depth Estimation. To estimate the depth of a single panoramic image using pre-trained perspective depth estimation models, 360MonoDepth[40] proposes a general spatial alignment-based method that projects the panoramic image into multiple perspective images for individual depth estimation. The resulting depth maps are aligned and back-projected to form the panoramic depth map. Building on this concept, we define N groups of outward perspective view directions in the spherical panoramic field of view (FoV), denoted as {v1, v2, ..., v𝑁 }, where v𝑛 ∈ Rℎ×𝑤×3. Each panoramic frame f𝑙 ∈ R𝐻×𝑊 ×3 (𝑙 = 1, 2, . . .,𝐿) will be reprojected into N perspective images corresponding to these view directions according to the equirectangular projection. Subsequently, a perspective depth estimation model is used to estimate the depth of the perspective images, yielding N depth maps, denoted as

{d𝑙1, d𝑙2, ..., d𝑙𝑁 }. Each depth map d𝑙𝑛 ∈ Rℎ×𝑤×3 is assigned a learn-

able scale factor 𝛼𝑛𝑙 ∈ R and a learnable shift factor 𝜷𝑙𝑛 ∈ Rℎ×𝑤 to adjust the depth values. Inspired by DreamScene360 [72], we

employ a learnable MLPs with parameters 𝚯𝑙 : R3 → R to model a static geometric field of the panoramic frame f𝑙. The input to the geometric field is the view directions v, while the output is the depth values along the corresponding ray directions from the origin. We define the optimization goal for space alignment as follows:

min E

𝜆depthLdepth + 𝜆scaleLscale + 𝜆shiftLshift (4)

𝑖∈𝑆

where Ldepth = || softplus(𝛼𝑖𝑙)d𝑙𝑖 + 𝜷𝑙𝑖 − MLPs(v𝑖;𝚯𝑙)|| Lscale = ∥ softplus(𝛼𝑖𝑙) − 1∥2 Lshift = ∑︁

(𝜷𝑙𝑖,𝑗,𝑘+1 − 𝜷𝑙𝑖,𝑗,𝑘)2 + (𝜷𝑙𝑖,𝑗+1,𝑘 − 𝜷𝑙𝑖,𝑗,𝑘)2

𝑗,𝑘

where 𝑆 denotes the index set for the perspective images involved in the current optimization. We initiate by estimating the depth map of the first frame f1, which is also the input reference image I, based on Eq. 4 with 𝑙 = 1, involving all N perspective images. After optimization, the depth map D1 ∈ R𝐻×𝑊 is determined using the geometric field 𝚯1 with the panoramic direction vpano ∈ R𝐻×𝑊 ×3 as input, computed as D𝑙 = MLPs(vpano;𝚯𝑙).

- 3.3.2 Space-Time DepthEstimation. Depthestimationforpanoramic videos requires not only spatial alignment within each individual frame but also temporal consistency across frames. To address this, we propose the Space-Time Depth Estimation. To more accurately comprehend the spatial motion information in panoramic videos, we employ a panoramic optical flow estimation model to derive the optical flow F ∈ R(𝐿−1)×𝐻×𝑊 ×2 of the panoramic video, representing pixel movement between consecutive frames. We define

the mask M𝑙 ∈ R𝐻×𝑊 to highlight pixels with motion distances, where 𝑙 = 1, . . .,𝐿 − 1. We then compute the post-motion positions

of these pixels to obtain M𝑙′ ∈ R𝐻×𝑊 , where 𝑙 = 2, . . .,𝐿. We use the following formula to calculate the motion region M𝑙 for each frame, identifying areas that have changed since the previous frame or will generate new motion in the next one:

𝐿

M𝑙 = M𝑙−1 ∨ M𝑙′ ∨ M𝑙, M˜ =

M𝑙 (5)

𝑙=1

The overall motion region mask M˜ is the union of the masks for each frame.

For subsequent panoramic frames (𝑙 > 1), We perform the spacetime depth estimation to ensure spatial and temporal consistency in the depth maps. This process leverages motion regions from optical flow for adaptive perspective selection. Additionally, the previously estimated depth maps offer supervision. The goal of optimization is extended for space-time depth alignment as follows:

min E

{Lspatial} + 𝜆firstLfirst + 𝜆preLpre (6)

𝑖∈𝑆

where Lspatial = 𝜆depthLdepth + 𝜆scaleLscale + 𝜆shiftLshift Lfirst = ∥(D1 − MLPs(vpano;𝚯𝑙)) ◦ ¬M˜ ∥ Lpre = ∥(D𝑙−1 − MLPs(vpano;𝚯𝑙)) ◦ M˜ ◦ ¬M𝑙 ∥

where 𝑆 = {𝑖 | ∃𝑗,𝑘 such that 𝛾(M𝑙, v𝑖)[𝑗,𝑘] = True}, and 𝛾 is the projection function that maps a panoramic image to a perspective image based on direction v. Consequently, only the perspective images that overlap with the motion region mask M𝑙 are involved for optimization of the current frame f𝑙. Other areas of the overall motion region M˜ are supervised using depth D𝑙−1 from the previous frame f𝑙−1, while non-motion regions are supervised using depth D1 from the first frame f1. Using Eq. 6, we could estimate the panoramic depth map D𝑙 (𝑙 > 1) frame by frame. thereby obtaining the depth of the entire panoramic video.

3.3.3 4D Scene Reconstruction. After completing the depth estima-

tion, the panoramic video and its depth map {[f𝑙, D𝑙]}, (𝑙 = 1, 2, ...𝐿) are converted into a 4D point cloud with temporal attributes, which serves as the initialization for the 4D scene. We choose Spacetime Gaussian [25] as the 4D representation of the scene. The video is projected into perspective views with varying FoV for supervision during training. Since the camera position of the panoramic video is fixed, we project the panoramic depth into the corresponding perspective depth map and apply depth-based warping to generate new views by perturbing the camera position relative to the original view. This process enriches the training set and improves scene completeness and rendering robustness.

###### 3.4 360World Dataset

Current large-scale text-video datasets like WebVid [2] primarily feature narrow-FoV perspective videos instead of panoramic videos. Furthermore, existing datasets [46, 51] for panoramic video generation primarily include footage captured with moving cameras, which makes them unsuitable for the task of 4D scene generation. To address data limitations, we present the 360World dataset, consisting of 7, 497 high-quality panoramic video clips with a total of 5, 380, 909 frames. Each clip is accompanied by textual descriptions sourced from open-domain content. The videos capture a wide array of real-world scenarios, from natural landscapes to urban environments, offering robust data support for generative models to comprehend dynamic panoramic scenes.

We collect original YouTube videos and annotate the segmented clips. we employ ShareGPT4Video [6], a Large Video-Language Model (LVLM) with strong video comprehension capabilities, to deeply analyze the video across both spatial and temporal dimensions and generate detailed text prompts of panoramic videos. Finally, we utilize a Large Language Model (LLM) for post-processing the text, summarizing and refining the detailed prompt by removing photography-related descriptive terms such as “camera” and “video”, resulting in final text prompts that effectively describe the scene content and dynamic motions.

###### 4 Experiments

In this section, we first introduce the implementation details in Sec. 4.1, followed by a multi-level and comprehensive evaluation. We compare our Panoramic Animator with other panoramic video generation methods in Sec. 4.2. We then evaluate HoloTime against other image-driven 4D scene generation methods in Sec. 4.3. Finally, we conduct ablation experiments in Sec. 4.4 to validate the effectiveness of proposed techniques.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

PanFusion 360DVD360DVD

A nighttime plaza with a fountain lit by changing colors and couples strolling under the stars.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

A golden wheat field swaying gently in the breeze, birds flying across the horizon.

#### 360DVD

Coastal cave, crashing waves.

PanFusion Ours+

Ours+Flux 360DVD Ours+Flux 360DVD Ours+Flux

FLUX(LoRA) Ours+

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

FLUX(LoRA) Ours+

A busy train station with trains arriving and departing, passengers rushing to catch their trains, and the sounds of announcements in the air.

A campfire in the wilderness, with people playing guitars under a sky full of shooting stars.

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

[Figure 125]

[Figure 126]

[Figure 127]

Ours+

A golden wheat field swaying gently in the breeze, birds flying across the horizon. Coastal cave, crashing waves.

Figure 3: Qualitative comparison of text-driven panoramic video generation. Our Panoramic Animator effectively achieves more coherent motion and avoids the occurrence of artifacts.

思想自由 兼容并包

###### Table 1: User study of text-driven panoramic video generation. Our approach can integrate different personalized textto-panorama generation models with outperformance results.

###### 4.1 Implementation Details

In the implementation of Panoramic Animator, the frame length 𝐿 of the video is set to 25. The resolution of the guidance model is 320×512. We only train the spatial layers from pre-trained weights for 5, 000 iterations on the 360World dataset. For the refinement model, a progressive training strategy from ViewCrafter [64] is applied. Initially,the entire UNet is trained on the hybrid data at 320 × 512 resolution for 5, 000 iterations, then the spatial layers are further trained at 576 × 1024 resolution on the 360World dataset for another 5, 000 iterations to upscale the resolution. Throughout the training, the batch size is set to 16, and the learning rate to 5×10−5. We employ Real-ESRGAN[52] as the super-resolution model. The final generated panoramic video frames are resized to 512 × 1024 to maintain the panoramic aspect ratio.

|Video Criteria Panorama Criteria| |
|---|---|
|Graphics Quality<br><br>Frame Consistency|End Continuity<br><br>Content Distribution<br><br>Motion Pattern|

Method

360DVD [51] 5.6692 5.9462 6.1308 5.7731 5.6385 Ours+PanFusion [67] 6.0962 6.3615 6.5308 5.9731 6.1692 Ours+FLUX (LoRA) 8.1423 8.1538 8.3500 8.2000 8.0308

###### 4.2 Panoramic Video Generation Comparisons

During the reconstruction phase, the generated panoramic video is first upscaled to 1024 × 2048 using Real-ESRGAN. The number of perspective views 𝐿 is set to 20, with each perspective image having a resolution of 512 × 512. Depth estimation for these images is conducted with Marigold [18]. PanoFlow [43] is utilized for optical flow estimation of the panoramic videos. We train the Spacetime Gaussian [25] lite model for 30,000 iterations with the default hyperparameter settings for each scene.

Due to the absence of available image-driven panoramic video generation methods currently, we compare our Panoramic Animator with 360DVD [51], a text-driven panoramic video generation method fine-tuned on AnimateDiff [10]. Our approach can integrate different text-to-panorama generation models to achieve textdriven panoramic video generation. PanFusion [67] and FLUX [21] with its Panorama LoRA [15] from Hugging Face are employed to conduct comparison. Given the input text prompt, They first creates a panoramic image, which is then used by the Panoramic Animator to generate the panoramic video.

##### Input Image Ours 3D Cin. Ours 3D Cin.

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

View1 View1 View2 View2

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

View1 View1 View2 View2

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

View1 View1 View2 View2

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

View1 View1 View2 View2

Figure 4: Qualitative comparison of image-driven 4D scene generation. Our method can generate more complex and diverse motions in the scene while maintaining spatial and temporal consistency in the dynamic scene.

- Table 2: Quantitative comparison of image-driven panoramic video generation. 360DVD* is fine-tuned on DynamiCrafter using the techniques proposed by 360DVD with the 360World dataset. “pers” indicates that the panoramic videos are projected into perspective videos for evaluation. Our method outperforms the baseline in terms of all metrics.

思想自由 兼容并包

|VBench (pers) ChronoMagic-Bench| |
|---|---|
|Subject Consistency Background Consistency Temporal Flickering Motion Smoothness Dynamic Degree<br><br>|GPT4o MTScore MTScore|

Method

360DVD* [51] 0.9508 0.9489 0.9797 0.9843 0.0667 2.2000 0.3556 Ours 0.9543 0.9501 0.9864 0.9903 0.0817 2.4111 0.3629

We use a Large Language Model (LLM) to generate a batch of text prompts for various scenes, emulating the captions from the 360World dataset, and use the generated prompts as the input for text-driven comparison. Fig. 8 shows the results of qualitative comparison, highlighting broad applicability of Panoramic Animator. We conduct a user study to comprehensively evaluate generated panoramic videos in terms of visual criteria and panoramic criteria. We follow the criteria set in 360DVD, including graphics quality, cross-frame consistency, left-right continuity, content distribution, and motion patterns. 26 participants rate each metric of the 10 sets of generated videos on a scale from 1 to 10. We calculate the average score of the different metrics. Table. 1 presents the results of user study, revealing that our method not only achieve high video quality but also effectively align with the characteristics of panoramic videos. This also demonstrates the strong adaptability of our method for multiple personalized text-to-panorama models.

To conduct a more precise and targeted comparison, we employ the techniques proposed by 360DVD to fine-tune the same base model DynamiCrafter with our 360World dataset, thereby obtaining

360DVD*, for the purpose of image-driven panoramic video generation comparison. We use multiple panoramic image generation models [21, 67, 71] to generate 90 panoramic images of different styles based on the generated prompts as input. Panoramic videos are projected into perspective videos to compute the metrics of VBench [16] for evaluating video details, including Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness and Dynamic Degree. The MTScore (Metamorphic Score) metrics of ChronoMagic-Bench [66] are used to directly assess the global motion of panoramic videos. Table. 2 demonstrates that our proposed techniques can achieve better temporal and motion details. Higher GPT4o MTScore and MTScore indicates that our method can generate more significant overall motion amplitude.

###### 4.3 4D Scene Generation Comparisons

We compare our framework with the optical flow-based 3D dynamic image technology, 3D-Cinemagraphy (3D-Cin.) [24]. Following the experimental setup of 4K4DGen, we employ 3D-Cin. to construct 4D scenes from the input panoramic images under the "circle" and

###### Table 3: Quantitative comparison and user study of imagedriven 4D scene generation. Our method outperforms the baselines in terms of all metrics.

[Figure 164]

[Figure 165]

[Figure 166]

|Q-Align User Study| |
|---|---|
|Image Quality<br><br>Image Aesthetics<br><br>Video Quality<br><br>Video Aesthetics|Graphics Quality<br><br>Temporal Coherence|

Method

(a) Input Image (b) Ours (full)

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

3D Cin. (zoom in) 1.9175 1.6271 2.266 1.8446 10.32% 11.29% 3D Cin. (circle) [24] 1.9433 1.6150 2.2823 1.8365 1.94% 5.16% Ours 2.2643 1.7627 2.6293 2.0350 87.74% 83.55%

###### Table 4: Quantitative ablation study of hybrid data finetuning (HDF) and motion guided generation (MGG). “pers” indicates that the panoramic videos are projected into perspective videos for evaluation.

(c) w/o HDF (d) w/o MGG

思想自由 兼容并包

###### Figure 5: Ablation study of hybrid data fine-tuning (HDF) and motion guided generation (MGG) for panoramic video generation.

|VBench (pers) ChronoMagic-Bench| |
|---|---|
|Temporal Flickering<br><br>Motion Smoothness<br><br>Dynamic Degree<br><br>|CHScore<br><br>GPT4o MTScore|

Method

[Figure 171]

w/o HDF 0.9860 0.9898 0.0694 533.8 2.4556 w/o MGG 0.9861 0.9899 0.0794 560.4 2.1444 Ours Full 0.9864 0.9903 0.0816 647.9 2.4111

Input Image Input Image Copy

(a) Concatenation

[Figure 172]

[Figure 173]

"zoom-in" settings, and then project the rendered videos into perspective videos for comparison. Fig. 4 shows the results of the qualitative comparison. It is evident that the optical flow-based method is mainly effective for creating flowing effects, making it suitable primarily for fluids like water, which limits its applications. In contrast, our method utilizes a video diffusion model, generating more complex texture variations and spatial motion, and thus exhibits superior generalization capabilities.

Middle Frame Middle Frame

(b) w/ PCT (c) w/o PCT

Figure 6: Ablation study of panoramic circular techniques (PCT). We concatenate the left and right ends of the generated frames to check for continuity. PCT effectively prevents the occurrence of discontinuous seam.

We employ the Q-Align [54] metrics to evaluate the quality and aesthetic scores of the perspective videos rendered from the generated scenes, as well as the individual video frames. We also conduct a 4D scene generation user study, where 31 participants evaluate 10 groups of generated 4D scenes and select the best method for each group of scenes based on the criteria of graphics quality and temporal coherence. Our method achieves better scores on all metrics in Table. 3, highlighting the quality of the generated scenes.

思想自由 兼容并包

We also assess the effectiveness of the temporal loss terms in Eq. 6 for Space-Time Depth Estimation. As illustrated in Fig. 7, for panoramic videos with significant spatial motion, the 𝜆first term ensures overall depth consistency between frames, while the 𝜆prev term mitigates artifacts in regions near motion.

###### 4.4 Ablation Study

We perform ablation experiments on Panoramic Animator and Panoramic Space-Time Reconstruction separately. Firstly, we evaluate the impact of our proposed Hybrid Data Fine-tuning (HDF) and Two-Stage Motion Guided Generation (MGG) on the Panoramic Animator, as illustrated in Fig. 5. We perform quantitative assessments of HDF and MGG in Table. 4. We employ three temporal metrics from VBench to assess projected perspective videos, as well as CHScore (Coherence Score) and GPT4o MTScore from ChronoMagic-Bench for panoramic videos. The results illustrate the contribution of HDF to enhancing temporal details and coherence, as well as the impact of MGG on the overall motion. Meanwhile, Fig. 6 demonstrates the effectiveness of PCT, which prevents the occurrence of discontinuous seam.

###### 5 Conclusions

In this work, we present HoloTime, an novel framework that transforms static panoramic images into large-scale 4D scenes. To tackle the limited availability of panoramic video datasets, we introduce 360World, a comprehensive dataset of fixed-camera panoramic videos. We propose the Panoramic Animator, which utilizes an I2V diffusion model for direct panoramic video generation. Moreover, we introduce Panoramic Space-Time Reconstruction, a method for 4D reconstruction of panoramic videos that ensures temporal and spatial consistency. Our approach outperforms existing methods by creating more captivating and realistic immersive dynamic environments, enhancing the virtual roaming experience.

- [12] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. 2022. Imagen Video: High Definition Video Generation with Diffusion Models. arXiv:2210.02303 [cs.CV] https://arxiv.org/abs/2210.02303
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 6840–6851. https://proceedings.neurips.cc/paper_files/paper/2020/ file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
- [14] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. 2022. Video Diffusion Models. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 8633–8646. https://proceedings.neurips.cc/paper_files/paper/2022/file/ 39235c56aef13fb05a6adc95eb9d8d66-Paper-Conference.pdf
- [15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. https://openreview.net/forum?id=nZeVKeeFYf9
- [16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 21807–21818.
- [17] Yanqin Jiang, Li Zhang, Jin Gao, Weiming Hu, and Yao Yao. 2024. Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id= sPUrdFGepF
- [18] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. 2024. Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 9492–9502.
- [19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.

2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139:1–139:14. https://doi.org/10.1145/3592433

- [20] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, MingHsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. 2024. VideoPoet: A Large Language Model for Zero-Shot Video Generation. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.
- [21] Black Forest Labs. 2023. FLUX. https://github.com/black-forest-labs/flux.
- [22] Yao-Chih Lee, Yi-Ting Chen, Andrew Wang, Ting-Hsuan Liao, Brandon Y. Feng, and Jia-Bin Huang. 2024. VividDream: Generating 3D Scene with Ambient Dynamics. arXiv:2405.20334 [cs.CV] https://arxiv.org/abs/2405.20334
- [23] Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhengzhong Tu, and Zhiwen Fan. 2024. 4K4DGen: Panoramic 4D Generation at 4K Resolution. arXiv:2406.13527 [cs.CV] https://arxiv.org/abs/2406.13527
- [24] Xingyi Li, Zhiguo Cao, Huiqiang Sun, Jianming Zhang, Ke Xian, and Guosheng Lin. 2023. 3D Cinemagraphy From a Single Image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4595–4605.
- [25] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2024. Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8508–8520.
- [26] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3D: High-Resolution Text-to-3D Content Creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 300–309.
- [27] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis.

2024. Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8576–8588.

- [28] Jinxiu Liu, Shaoheng Lin, Yinxiao Li, and Ming-Hsuan Yang. 2024. DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes. arXiv:2412.11100 [cs.CV] https://arxiv.org/abs/2412.11100
- [29] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot One Image to 3D Object. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 9298–9309.
- [30] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2024. SyncDreamer: Generating Multiview-consistent Images

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

l=1 l=6

(a) Input Image (b) Ours (full)

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

(c) w/o ℒfirst (d) w/o ℒprev

l=6 l=6

Figure 7: Ablation study of temporal loss terms for SpaceTime Depth Estimation. Based on the same first frame depth, our temporal loss terms contribute to higher temporal consistency of the middle frame’s depth.

思想自由 兼容并包

###### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B. Lindell. 2024. 4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 7996–8006.
- [2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. 2021. Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 1728–1738.
- [3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. 2021. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision. 5855–5864.
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. 2023. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv:2311.15127 [cs.CV] https://arxiv.org/abs/2311.15127
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023. Align Your Latents: High-Resolution Video Synthesis With Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 22563–22575.
- [6] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. 2024. ShareGPT4Video: Improving Video Understanding and Generation with Better Captions. arXiv:2406.04325 [cs.CV] https://arxiv. org/abs/2406.04325
- [7] Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, and Li Yuan. 2024. Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Semantic Prompts. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=O072Rc8uUy
- [8] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee.

2023. LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes. arXiv:2311.13384 [cs.CV] https://arxiv.org/abs/2311.13384

- [9] Chaoran Feng, Wangbo Yu, Xinhua Cheng, Zhenyu Tang, Junwu Zhang, Li Yuan, and Yonghong Tian. 2025. AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scenes. In AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, Toby Walsh, Julie Shah, and Zico Kolter (Eds.). AAAI Press, 2924–2932. https://doi.org/10.1609/AAAI.V39I3.32299
- [10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. arXiv:2307.04725 [cs.CV] https://arxiv.org/abs/2307.04725
- [11] Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, and Ziwei Liu. 2024. VEnhancer: Generative Space-Time Enhancement for Video Generation. arXiv:2407.07667 [cs.CV] https://arxiv.org/ abs/2407.07667

- from a Single-view Image. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=MN3yH2ovHb
- [31] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. 2024. Latte: Latent Diffusion Transformer for Video Generation. arXiv:2401.03048 [cs.CV] https://arxiv.org/abs/2401.03048
- [32] Yikun Ma, Dandan Zhan, and Zhi Jin. 2024. FastScene: Text-Driven Fast Indoor 3D Scene Generation via Panoramic Gaussian Splatting. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024. ijcai.org, 1173–1181. https://www.ijcai.org/ proceedings/2024/130
- [33] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
- [34] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2022. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. arXiv:2112.10741 [cs.CV] https://arxiv.org/abs/2112.10741
- [35] Hao Ouyang, Kathryn Heal, Stephen Lombardi, and Tiancheng Sun.

2023. Text2Immersion: Generative Immersive Scene with 3D Gaussians. arXiv:2312.09242 [cs.CV] https://arxiv.org/abs/2312.09242

- [36] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview.net/forum?id=FjNys5c7VyY
- [37] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. 2024. Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=0jHkUDyEO9
- [38] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.

2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv:2204.06125 [cs.CV] https://arxiv.org/abs/2204.06125

- [39] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. 2024. DreamGaussian4D: Generative 4D Gaussian Splatting. arXiv:2312.17142 [cs.CV] https://arxiv.org/abs/2312.17142
- [40] Manuel Rey-Area, Mingze Yuan, and Christian Richardt. 2022. 360MonoDepth: High-Resolution 360° Monocular Depth Estimation. In CVPR.
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis With Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 10684–10695.
- [42] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 36479–36494. https://proceedings.neurips.cc/paper_files/paper/2022/file/ ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf
- [43] Hao Shi, Yifan Zhou, Kailun Yang, Xiaoting Yin, Ze Wang, Yaozu Ye, Zhe Yin, Shi Meng, Peng Li, and Kaiwei Wang. 2023. PanoFlow: Learning 360° Optical Flow for Surrounding Temporal Understanding. IEEE Trans. Intell. Transp. Syst. 24, 5

(2023), 5570–5585. https://doi.org/10.1109/TITS.2023.3241212

- [44] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. 2024. MVDream: Multi-view Diffusion for 3D Generation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11,

2024. OpenReview.net. https://openreview.net/forum?id=FUgrjq2pbB

- [45] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. 2022. Make-A-Video: Text-to-Video Generation without Text-Video Data. arXiv:2209.14792 [cs.CV] https://arxiv.org/abs/2209.14792
- [46] Jing Tan, Shuai Yang, Tong Wu, Jingwen He, Yuwei Guo, Ziwei Liu, and Dahua Lin. 2024. Imagine360: Immersive 360 Video Generation from Perspective Anchor. arXiv:2412.03552 [cs.CV] https://arxiv.org/abs/2412.03552
- [47] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. 2022. Blocknerf: Scalable large scene neural view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8248–8258.
- [48] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. 2024. DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id= UyNXMqnN3c
- [49] Zhenyu Tang, Chaoran Feng, Xinhua Cheng, Wangbo Yu, Junwu Zhang, Yuan Liu, Xiaoxiao Long, Wenping Wang, and Li Yuan. 2025. NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations.

- arXiv:2503.23162 [cs.CV] https://arxiv.org/abs/2503.23162
- [50] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023. ModelScope Text-to-Video Technical Report. arXiv:2308.06571 [cs.CV] https://arxiv.org/abs/2308.06571
- [51] Qian Wang, Weiqi Li, Chong Mou, Xinhua Cheng, and Jian Zhang. 2024. 360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6913–6923.
- [52] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. 2021. Real-ESRGAN: Training Real-World Blind Super-Resolution With Pure Synthetic Data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. 1905–1914.
- [53] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan LI, Hang Su, and Jun Zhu. 2023. ProlificDreamer: High-Fidelity and Diverse Textto-3D Generation with Variational Score Distillation. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36. Curran Associates, Inc., 8406–8441. https://proceedings.neurips.cc/paper_files/paper/2023/file/ 1a87980b9853e84dfb295855b425c262-Paper-Conference.pdf
- [54] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin. 2023. Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels. arXiv:2312.17090 [cs.CV] https: //arxiv.org/abs/2312.17090
- [55] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T. Barron, and Aleksander Holynski. 2024. CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models. arXiv:2411.18613 [cs.CV] https://arxiv.org/ abs/2411.18613
- [56] Jinbo Xing, Menghan Xia, Yong Zhang, Hao Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. 2024. DynamiCrafter: Animating Open-Domain Images with Video Diffusion Priors. In Computer Vision

- ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLVI (Lecture Notes in Computer Science, Vol. 15104), Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 399–417. https://doi.org/10.1007/978-3-031-72952-2_23

- [57] Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Yixuan Li, Gordon Wetzstein, Ziwei Liu, and Dahua Lin. 2024. LayerPano3D: Layered 3D Panorama for HyperImmersive Scene Generation. arXiv:2408.13252 [cs.CV] https://arxiv.org/abs/ 2408.13252
- [58] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv:2408.06072 [cs.CV] https://arxiv.org/abs/2408.06072
- [59] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 2024. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency. arXiv:2312.17225 [cs.CV] https://arxiv.org/abs/2312.17225
- [60] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. 2024. WonderWorld: Interactive 3D Scene Generation from a Single Image. arXiv:2406.09394 [cs.CV] https://arxiv.org/abs/2406.09394
- [61] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T. Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, and Charles Herrmann. 2024. WonderJourney: Going from Anywhere to Everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 6658–6667.
- [62] Wangbo Yu, Yanbo Fan, Yong Zhang, Xuan Wang, Fei Yin, Yunpeng Bai, Yan-Pei Cao, Ying Shan, Yang Wu, Zhongqian Sun, et al. 2023. Nofa: Nerf-based oneshot facial avatar reconstruction. In ACM SIGGRAPH 2023 conference proceedings. 1–12.
- [63] Wangbo Yu, Chaoran Feng, Jiye Tang, Jiashu Yang, Zhenyu Tang, Xu Jia, Yuchao Yang, Li Yuan, and Yonghong Tian. 2024. EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images. arXiv:2405.20224 [cs.CV] https://arxiv. org/abs/2405.20224
- [64] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. 2024. ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis. arXiv:2409.02048 [cs.CV] https://arxiv.org/abs/2409.02048
- [65] Wangbo Yu, Li Yuan, Yan-Pei Cao, Xiangjun Gao, Xiaoyu Li, Wenbo Hu, Quan Long, Ying Shan, and Yonghong Tian. 2024. HiFi-123: Towards High-Fidelity One Image to 3D Content Generation. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXIII (Lecture Notes in Computer Science, Vol. 15131), Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer, 258–274. https://doi.org/10.1007/978-3-031-73464-9_16
- [66] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Ruijie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. 2024. ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation. arXiv:2406.18522 [cs.CV] https://arxiv.org/abs/2406.18522

[Figure 182]

- [67] Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. 2024. Taming Stable Diffusion for Text to 360◦ Panorama Image Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
- [68] Junwu Zhang, Zhenyu Tang, Yatian Pang, Xinhua Cheng, Peng Jin, Yida Wei, Xing Zhou, Munan Ning, and Li Yuan. 2025. Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting. In Computer Vision – ECCV 2024, Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer Nature Switzerland, Cham, 303– 320.
- [69] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. 2023. I2VGen-XL: High-Quality Imageto-Video Synthesis via Cascaded Diffusion Models. arXiv:2311.04145 [cs.CV] https://arxiv.org/abs/2311.04145
- [70] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng.

2023. MagicVideo: Efficient Video Generation With Latent Diffusion Models. arXiv:2211.11018 [cs.CV] https://arxiv.org/abs/2211.11018

- [71] Haiyang Zhou, Xinhua Cheng, Wangbo Yu, Yonghong Tian, and Li Yuan. 2024. HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions. arXiv:2407.15187 [cs.CV] https://arxiv.org/abs/2407.15187
- [72] Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, and Achuta Kadambi. 2025. DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting. In Computer Vision – ECCV 2024, Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer Nature Switzerland, Cham, 324–342.

A serene scene at the Arc de Triomphe in Paris under a clear sky. Vehicles enter and exit the frame, while pedestrians walk around the monument.

A serene town square centered around a fountain with a green statue. Chairs remain in place, and shadows shift slightly as time passes.

A tranquil river scene with lush greenery and a small waterfall. People stand, swim, and float in the river, shifting positions and interacting gently.

A serene tropical beach with clear waters and palm trees. The ocean gently laps against the shoreline, and small boats remain anchored offshore.

[Figure 183]

- A lively waterfront scene with pedestrians and colorful buildings. People stroll, gather in groups, and sit on benches while the surveillance cameras remain stationary.

A lively club scene with a DJ and a dimly lit atmosphere. The DJ manipulates the equipment while the crowd subtly moves in the background.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Figure 8: Samples in the 360World dataset.

- B Implementation Details

思想自由 兼容并包

###### A 360World Dataset Processing

We only download publicly available YouTube videos for training. We utilize keywords such as “panorama” and “VR” for retrieval and filtering. Given that each original video may contain multiple distinct scenes and transitions between them, we perform keyframebased slicing on the original videos to identify frames with significant changes. Due to the camera’s broad field of view (FoV), there are typically not large areas of pixel changes throughout the realworld panoramic video. Most motion is confined to specific local regions within the panoramic space. Therefore, we use the pixel distance between adjacent frames to determine whether a frame is a keyframe. Firstly, we perform pre-processing on each video. To facilitate the detection of significant changes between frames, we sample video frames at a fixed interval along the temporal dimension, thereby increasing the video speed.

###### B.1 Panoramic Animator

The training of Panoramic Animator is completed on 8 NVIDIA A100 GPUs with 80 GB RAM. For inference, we employ the DDIM sampler with classifier-free guidance. Initially, we copy the left one-fifteenth of the reference image and concatenate it to the right end. The timesteps 𝑇 is set to 1, 000, while 𝐾 is set to 800. After each denoising step, the left and right one-sixteenths are blended, as shown in Panoramic Circular Techniques (PCT). Finally, we crop the right one-sixteenth of the generated panoramic video.

For a sampled video with 𝐿 frames, each frame is converted into a grayscale image. The 𝑖-th grayscale image is defined as 𝑔𝑖 ∈ R𝐻×𝑊 . We calculate the pixel distance between current frame 𝑔𝑖 and the previous frame 𝑔𝑖−1. If the distance at any pixel exceeds the threshold 𝜃trans, that pixel is labeled a transition pixel. If the proportion of transition pixels across the entire frame surpasses the threshold 𝜃count across the entire frame, it signifies a significant change area, classifying it as a keyframe. The formal formula is as follows:

###### B.2 Panoramic Space-Time Reconstruction

In the phase of depth estimation, following DreamScene360 [72], we project each panorama frame into 20 perspective views based on the surface directions of a regular icosahedron. The hyperparameter settings of depth optimization are as follows: 𝜆depth = 𝜆shift = 𝜆first = 𝜆pre = 1, 𝜆scale = 0.1. When optimizing the spatial depth alignment of the first frame, we perform 3, 000 iterations, with 𝜆shift = 0 for the first 1, 500 iterations. Subsequent frame depth alignments consist of1, 000iterations each. continuously optimizing the current geometric field 𝚯𝑙 based on the optimization parameters 𝚯𝑙−1 from the proir.s

𝑓 (𝐷,𝜃1) = {(𝑖, 𝑗) | 𝑑𝑖𝑗 > 𝜃1, 1 ≤ 𝑖 ≤ ℎ, 1 ≤ 𝑗 ≤ 𝑤} (7)

In the training of Spacetime Gaussians [25], We set up 38 cameras with approximately 45 FoV (Field of View) to cover the spherical perspective of the panoramic video. The depth-based warping is applied to generate 4 supplementary views for each view, with camera position disturbances of 0.1 in the up, down, left, and right directions. Additionally, 20 cameras with approximately 75 FoV are set up. For each scene, We train the lite model of Spacetime Gaussians for 30, 000 iterations on single NVIDIA A800 GPU with 80 GB RAM, using the default hyperparameter settings.

True, if 𝑓 (|𝑔𝑖 − 𝑔𝑖−1| ,𝜃trans) > 𝜃count False, otherwise

(8)

T𝑖 =

After detecting keyframes, we slice the original video and discard frames that are too close to the keyframes to eliminate complex transitions involving multiple frames.

###### B.3 4D Point Cloud Initialization

To reduce memory usage and improve training efficiency of Spacetime Gaussianss, we decrease the initialized 4D point cloud. A 4D point cloud used for Spacetime Gaussian initialization has a time attribute 𝑡, which indicates the moment the point exists. First, we convert the panoramic video into a grayscale image and calculate the standard deviation of each pixel over time. For pixels with a standard deviation less than 20, we mark them as texture variation areas Mstd. For the first frame of the video at time 𝑡 = 0, all pixel points are merged into the 4D point cloud. For subsequent frames at time 𝑡, only points that covered by Mstd or the optical flow detected motion area M𝑙 of the current frame are merged into the 4D point cloud.

###### C Experiments Details

We use 90 prompts their generated panoramic images to evaluate panoramic video generation, as well as 20 generated panoramic images for assessing 4D scene generation. We employ a projection method based on a regular icosahedron in experiments to project panoramic videos to render perspective videos. All of the panoramic images used in experiments and presentations are generated by PanFusion [67], FLUX [21] (panorama LoRA), and panorama

generation method in Holodreamer [71]. When designing these experimental data, in addition to the realistic style, we assigned some other styles such as cartoon, cyberpunk, etc., to a portion of the prompts and images, in order to evaluate the out-of-distribution (OOD) generalization capability of our method.

We replace the original 360DVD base model with DynamiCrafter [56] and complete 15,000 iterations of training using our 360World dataset under the same hyperparameter settings to obtain 360DVD* for comparison in image-driven panorama generation.

###### D Visual Results

We provide a supplementary video containing the generated results, and we strongly recommend that you view it to have the most intuitive understanding of the generated 4D scenes and panoramic videos. We use Venhancer [11] to super-resolution the generated panoramic videos before Panoramic Space-Time Reconstruction for a better VR immersive experience of 4D scenes. This technique is applied solely to the 4D scene presentations in the supplementary video and is not used elsewhere in the video. Notably, it is also excluded from any comparative experiments or ablation studies in the main paper.

