arXiv:2506.02620v1[cs.GR]3Jun2025

# FlexPainter: Flexible and Multi-View Consistent Texture Generation

Dongyu Yan1,∗, Leyi Wu1,∗, Jiantao Lin1, Luozhou Wang1, Tianshuo Xu1, Zhifei Chen1, Zhen Yang1, Lie Xu2, Shunsi Zhang2, Yingcong Chen1,3,† 1HKUST(GZ), 2Quwan, 3HKUST

{starydyxyz@gmail.com; lwu398@connect.hkust-gz.edu.cn; yingcong.ian.chen@gmail.com}

[Figure 1]

Untextured mesh

[Figure 2]

Conditions Results

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Stylize it!

[Figure 9]

[Figure 10]

Stylize it!

[Figure 11]

“A pink tree stum”

[Figure 12]

“A tree stum in ice and snow style”

Figure 1. FlexPainter generates diverse, high-quality textures based on various flexible user prompts.

### Abstract

Texture map production is an important part of 3D modeling and determines the rendering quality. Recently, diffusionbased methods have opened a new way for texture generation. However, restricted control flexibility and limited prompt modalities may prevent creators from producing desired results. Furthermore, inconsistencies between generated multi-view images often lead to poor texture generation quality. To address these issues, we introduce FlexPainter, a novel texture generation pipeline that enables flexible multi-modal conditional guidance and achieves highly consistent texture generation. A shared conditional embedding space is constructed to perform flexible aggregation between different input modalities. Utilizing such embedding space, we present an image-based CFG method to de-

∗ Both authors contributed equally to this research. † Corresponding Author.

compose structural and style information, achieving reference image-based stylization. Leveraging the 3D knowledge within the image diffusion prior, we first generate multi-view images simultaneously using a grid representation to enhance global understanding. Meanwhile, we propose a view synchronization and adaptive weighting module during diffusion sampling to further ensure local consistency. Finally, a 3D-aware texture completion model combined with a texture enhancement model is used to generate seamless, high-resolution texture maps. Comprehensive experiments demonstrate that our framework significantly outperforms state-of-the-art methods in both flexibility and generation quality.

### 1. Introduction

Texturing is a crucial step in 3D modeling and an important aspect of computer graphics and vision. It has signif-

icant applications in games, film, VR/AR, and animation. With the development of diffusion models [1–6], works utilizing image generation models for texture generation have demonstrated strong potential [7–12]. However, extending these methods from general image generation to texture creation remains challenging.

A key challenge in multi-view-based texture generation is precisely controlling the generated results according to user expectations. Most of the existing texture generation methods, like TEXTure [7], Text2Tex [8], and MVPaint

- [13] use text as the condition. Other works like Paint3D
- [14] and FlexiTex [15] utilize IP-Adapter [16] to allow image-conditioned generation. Although high-quality texture maps can be generated, these methods only rely on a single modality as a condition, which may cause ambiguity. For example, text descriptions may fail to convey detailed color, style, or material information accurately. Meanwhile, finding an image condition that perfectly meets the user’s demand is difficult; mixed information within it may lead to unclear semantic intent and missing contextual details. Differently, StyleTex [17] uses a reference style image to generate stylized textures. However, using an additional GPT model for content-style separation increases the system’s complexity, and the style feature obtained may preserve unwanted structural information of the input image.

Another important challenge of texture generation is ensuring inter-view consistency, as global and local discrepancies can significantly degrade the plausibility and detail quality of the generated texture. Previous works [7, 8] usually generate one viewpoint at a time sequentially to paint the mesh. These generated views are independent, which may easily lead to inconsistencies. To overcome such problems, methods like SyncMVD [18] and TexFusion [19] merge different views in UV space during diffusion sampling and achieve view-synchronized generation. Although these methods can produce locally consistent images, there is still no information exchange between non-overlapping views, which causes the Janus problem. To gain holistic information across views, Meta TextureGen [20] and MVPaint [13] leverages a multi-view image grid as the generation target. However, after reprojecting into UV space, misalignments in the details may still occur, resulting in ghosting artifacts. Moreover, both types of the above methods employ a heuristic weight map to merge partial textures reprojected from different views. Such a simple weighting function can cause low robustness and adaptability to different inputs.

To address the aforementioned challenges, we introduce FlexPainter, a novel texture-generation pipeline with flexible input and consistent generation. Leveraging a shared, linear-structured conditional embedding space, we achieve the complementary integration of high-level semantics from texts and low-level details from images. Such

multi-granularity control helps generate textures that better meet designers’ needs in real-world production, reducing the time spent on repeated adjustments. With such embedding space, we further inject image embedding into the classifier-free guidance (CFG) [21] module. Such imagebased CFG not only improves generation quality but also provides flexible control. It can eliminate the structure and content information in the image prompt by feeding grayscaled versions as negative prompts, yielding aesthetic texture stylization.

For inconsistency, our method handles it from both global and local perspectives. We apply the multi-view image grid representation as the generation target to gain the model with global understanding. This way, attention between views can ensure a holistic understanding of the object, thereby maintaining global coherence. As for the local alignment, we apply view synchronization during diffusion sampling. At each denoising step, we reproject views back to UV space, aggregate them into a unified UV map, and then rasterize it into the image grid representation for the next step of generation. We use an additional adaptive WeighterNet to aggregate the partial UV maps. Compared with the traditional heuristic weighting function used in [13, 20, 22], our method can handle various generated inputs at different timesteps and output robust results. Finally, a diffusion-based texture completion module and a texture enhancement network are applied to obtain the full texture map with high-resolution details.

In summary, our contributions are as follows: (1) We propose FlexPainter, a carefully designed pipeline for generating high-quality and consistent textures, offering intuitive granularity control. (2) We conduct linear operations and image-based CFG in the conditional embedding space, achieving flexible generation and coherent stylization based on different user demands and prompt modalities. (3) We realize global and local consistent texture generation through a unified image grid representation combined with a view synchronization and weighting module.

### 2. Related Work

#### 2.1. Multi-View Generation

Multi-view generation tasks aim to produce multiple perspectives of an object with the reference image or text. Zero-1-to-3 [23] and Consistent-1-to-3 [24] generate novel views using a viewpoint-conditioned manner. Zero123++ [25], MVDream [26], and other works [27–30] manage to generate multi-view grid images simultaneously for better consistency. Some works [31–36] incorporate various modules, including video diffusion priors, iterative view synchronization with 3D proxy, and flexible prompt-based input to accomplish the task. Unlike the above methods, which mainly focus on subsequent 3D generation, multi-

view generation for texturing incorporates additional geometry constraints. It requires multi-view information to be projected and merged into UV space to obtain the final texture map, imposing stricter requirements on multi-view consistency.

#### 2.2. Texture Generation

Texture generation aims to produce UV texture maps of the target mesh with the given prompts as conditions. Traditional learning-free methods [37–40] rely on manual or algorithm-specific procedures to generate textures. Approaches like [41–44] performs texture generation based on GAN structure [45, 46]. With the rapid development of diffusion models, their adoption in texture generation has been explored. Approaches like [47–51] optimize texture fields based on Score Distillation Sampling (SDS). Other works [7–10] use pre-trained diffusion models to generate in the image domain and apply predefined rules to fuse images from different views into the UV domain.

Condition flexibility. Many texture generation methods attempt to allow multi-modality input to better guide texture generation. TEXTure [7] and TextureDreamer [51] achieve concept transfer using textual inversion [52] and DreamBooth [53] techniques. In addition to basic text input, FlexiTex [15] and Paint3D [14] allows for image conditioning by adopting IP-Adapter [16] module. However, these works still use fixed inputs and cannot integrate multiple conditional controls flexibly. Differently, StyleTex [17] proposes to operate embeddings in the CLIP [54] space to achieve stylized texture generation. While aiming to disentangle content and style, the additional language model introduces overhead and may undesirably preserve structural information from the input image.

UV space consistency. Direct back-projection of the diffusion-generated multi-view images may cause misalignments in UV space. To address the UV inconsistencies from different generated views, methods like TexFusion [19], SyncMVD [18], and others [9, 11, 12, 15, 22, 55] share noisy content across views via UV projection during denoising loop. However, these methods struggle to maintain global consistency due to insufficient holistic information, often producing artifacts and Janus Problem. Other works [13, 20, 56] leverage image grid representation for simultaneous view generation. Although attention between views can enhance global consistency, it may lead to local misalignment, affecting aggregated UV results. Point-UV [57], and TEXGen [58] take a different approach by directly generating texture maps using point and UV hybrid structures. However, they are trained from scratch, relying on limited 3D datasets, leading to limited generalizability.

### 3. Preliminaries

- 3.1. Latent Flow Matching Method

The flow matching method [59] is a generative method that transforms a noise distribution into a target distribution. It is modeled by velocity prediction that guides sampling along a smooth trajectory in data space. Specifically, the most commonly used rectified flow defines a straight line as the path:

xt = (1 − t)x0 + tϵ,

dx dt

= vt(xt,Θ), (1)

where xt is a sample point on the flow trajectory in time step t, x0, ϵ are the destinations from two distributions, and vt represents the velocity field modeled by a neural network with parameters Θ.

Flow matching-based image generation is typically extended to the latent space to improve scalability and efficiency. To this end, a variational autoencoder (VAE) [60] is used for the mapping between image space and latent space:

x0 = E(I), I = D(x0), (2)

where E and D denotes the encoder and decoder of the VAE, and I is a sample image in the original space.

- 3.2. Texture Rasterization and ReProjection

Texture rasterization and reprojection are fundamental techniques in computer graphics, enabling mapping between rendered images and UV textures. Rasterization involves projecting a 3D model onto a 2D image plane, determining visibility, and assigning texture values from UV space to image pixels. In contrast, reprojection involves backprojecting images from a certain view onto the 3D object, which reconstructs textures based on known depth and camera information. With camera c and texture map T, they can be noted as:

I = R(T,c), T = R−1(I,c). (3)

### 4. Method

Given a 3D object mesh, we aim to generate high-fidelity textures that reflect users’ creative intent through diverse prompts. We implement a prompt aggregation method on a multi-modal embedding space to achieve flexible conditioning. We also achieve aesthetic texture stylization with an image-based CFG method (Sec. 4.1). As for the generation pipeline, we use a multi-view image grid representation to strengthen global understanding. We propose a reprojection-based view synchronization and weighting module during diffusion sampling for better view consistency (Sec. 4.2). Finally, we perform texture completion and enhancement in the UV space to generate complete and detailed results (Sec. 4.3). The full pipeline of our method is shown in Fig. 3.

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

“a parrot with colorful feathers”

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“a sofa with pink leather and brown wooden legs”

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

- Figure 2. Example of modality aggregation and semantic manipulation by controlling the image embedding strength α. Each case’s first and second rows show text-to-image and image-to-image interpolation, respectively.

#### 4.1. Flexible Conditioning

Aggregation of multi-modal prompts. Accommodating multi-modal inputs to achieve desired results flexibly is a challenging topic. To enable text, images, and their interactions as conditions, we need a shared embedding space to align the information from each modality. Therefore, we use separated image and text embedders within the same embedding space to process inputs of different modalities. By concatenating the output embeddings, a unified condition can be formed to guide the generation process using cross-attention. Moreover, since our embedding space retains the superior properties of a linear space, we can further manipulate the strength and direction of the image embeddings to provide more flexible control. The final embedding v for the model can be obtained through the following equation:

n

αiI(Icondi ), (4)

v = T (wcond)

i=1

where T and I represents the text and image embedder respectively, wcond and Icondi are the text and i-th image condition. Such manipulation in embedding space results in meaningful generations, as illustrated in Fig. 2. Even with single-modal training, the linear structure and combination property can be well preserved, allowing advanced tasks such as text-assisted image prompt refinement and generation using reference-style images to be effectively implemented without special design.

Image-based CFG Traditional CFG is usually carried out using blank text, which lacks strong and precise control. Utilizing the above multi-modal embedding space, we can inject image information into the diffusion network to enhance its generative capabilities. Furthermore, we incorporate image embeddings into the negative prompts used in the CFG framework [21] to improve generation control. Em-

ploying image embeddings yields better conditional control and higher-quality generation results than textual-only embeddings as negative prompts. Moreover, we can utilize images containing undesired structures or specific colors as negative prompts to explicitly suppress their presence in the generated outputs. Building on this property, we further achieve reference image-conditioned texture stylization. Due to the entanglement between content and style, directly using the original reference image as a prompt may introduce unintended structural artifacts into the generated texture. To address this, we employ a grayscaled version of the reference image as a negative prompt to eliminate structural information while preserving stylistic features. This disentanglement of content and style enables more robust and flexible texture stylization, allowing the model to focus on transferring color and tone without copying shape or layout from the reference.

#### 4.2. Generation Pipeline

Representation of the multi-view images. We first perform generation in the image space by leveraging the strong priors in current state-of-the-art 2D diffusion models. However, generating a single image per sample can cause global inconsistencies due to the lack of communication between perspectives. We adopt image grid representation for synchronized multi-view generation in a single sample to facilitate information exchange between different views and maximize the use of 3D prior information from existing image diffusion models. Specifically, we arrange four evenly spaced surround views of the object in a two-by-two grid as our generation target. For geometry alignment, depth maps are rendered and concatenated with the noisy color images to form the input to the diffusion network. In particular. we use LoRA [61] to retarget the pre-trained model to generate pure background and coherent multi-view grid images. The image grid representation and simultaneous generation strategy allow attention between views, which can better ensure global understanding and consistency. The 3D knowledge within the model’s large amount of training data makes it much easier to adapt to the new task.

Reprojection-based view synchronization. The use of image grid representation allows us to generate globally consistent multi-views. However, local misalignments can still occur when the images are reprojected into UV space, affecting the overall texture quality. To address this problem, we propose a reprojection-based view-synchronization module to align different views in UV space at each sampling step. Specifically, after obtaining the predicted flow vt from the noisy latent xt at denoising step t, we first use it to predict the fully denoised latent xt0:

xt0 = xt − tvt(xt). (5)

[Figure 59]

[Figure 60]

###### View-Synchronization and 3D-aware WeighterNet

“A Porsche 911 gt3 sport car, yellow body, black roof”

###### Multi-View Image Grid Generation With Flexible Conditional Control

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

WeighterNet Conditions

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

position map normal map

[Figure 78]

𝒟𝒟 𝑥𝑥0𝑡𝑡 𝒟𝒟(𝑥𝑥0𝑡𝑡−1)

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

‘with green side stripes and green racing tail wing’

[Figure 84]

[Figure 85]

Reproject Rasterize & Step

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

camera rays UV masks

###### LoRA

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

| | | |
|---|---|---|
| | | |

[Figure 98]

| | | |
|---|---|---|
|Point|Bl|ocks|
| | | |
|UV|lo|cks|

‘A porsche 911 gt3 sport car in icy style’

[Figure 99]

[Figure 100]

𝑥𝑥𝑇𝑇

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

raw mesh

[Figure 107]

[Figure 108]

𝒩𝒩(0,𝐈𝐈) random noise depth grid

WeighterNet

as negative prompt

[Figure 109]

𝑇𝑇1:4𝑡𝑡 𝑇𝑇𝑡𝑡

[Figure 110]

[Figure 111]

Trainable LoRA layers Concatenation

Frozen Diffusion Transformer layers

Trainable WeighterNet

- Figure 3. Pipeline of our method. We first generate multi-view images using the conditional input from the user. The top two cases show the generation of using text-only or image-only conditions. Leveraging the linear operation in Equ.(4), we can also perform text-guided image refinement (shown in green) and stylization using reference image (shown in blue). The right side shows our view-synchronization and weighting module. Consistent multi-view images can be generated by reprojection and weighting during each sampling step.

Then, the latent xt0 is decoded, split into independent images, and reprojected into UV space, producing four partial

texture maps T1:4t :

I1:4t = S(D(xt0)), Tit = R−1(Iit,ci), i = 1,2,3,4,

(6)

where S denotes splitting operation that turns grid image into individuals images I1:4t , and ci represents the i-th camera. Next, a weighting function W is applied on the partial texture maps T1:4t to generate the fused texture map T˜t:

T˜t = W(T1:4t ). (7)

From T˜t, the view-synchronized latent x˜t0 can be obtained through operations inverse to the above process:

I˜1:4t = R(T˜t,c1:4), x˜t0 = E(F(I˜1:4t )). (8)

Finally, a view-synchronized flow prediction v˜t is computed to replace the original vt, and the subsequent sampling steps continues:

xt − x˜t0 t

. (9)

v˜t =

This reprojection-based view synchronization ensures that, at each diffusion step, the multi-view images to be denoised are derived from the same UV map. This way, the consistency between views can be further maintained.

Adaptive Weighting. Although multi-view consistency can be ensured using view synchronization, the generation

quality can be greatly affected by the choice of the weighting function W. Traditionally, the weights assigned to partial textures from different views are determined heuristically, using the cosine of the angle between the camera rays and the rendered normals of the mesh as the weighting factor. While such a weighting function provides a simple way to detect observation quality, it can not dynamically adjust to different input variations. It may introduce poorly generated regions into the final result. Therefore, we train a WeighterNet as the weighting function W, allowing it to discern generation quality and achieve robust weighting results. Due to the discontinuities in UV space, incorporating 3D structures to enhance the network’s spatial awareness is beneficial. Thus, we combine the UV blocks and point blocks by the mapping between 2D and 3D space. The WeighterNet also considers the input variation of different time steps by taking an additional t as input. The final format can be represented as:

T˜t = W(T1:4t ,R1:4,N1:4,P,t,Θ), (10)

where P represents the position map in the UV space. We use ground truth textures T¯ and its simulated denoised texture T˜t at time step t as a training pair. For supervision, perceptual loss and cycle loss are used:

4

vgg(R(T,c¯ i)) − vgg(R(W(T1:4t ,Θ),ci)),

Lpec =e−αt

i=1

4

R(Tit,ci) − R(W(T1:4t ,Θ),ci),

Lcyc =

i=1

(11)

where e−αt balances the penalties for different noise levels. As a result, our adaptive weighting function can provide adaptable view aggregation in a training-based manner.

#### 4.3. Texture Completion and Enhancement

We generate and blend the multi-view images in the previous stage, resulting in a single UV map T˜0. Although most of the object surface is covered, some regions still require texture completion due to occlusion. A straightforward approach is to use an image diffusion model to perform texture inpainting in the UV space. However, due to the discontinuity inherent in UV space, this method lacks an understanding of the spatial relationships, potentially leading to suboptimal results. Leveraging the 3D-aware architecture of TEXGen [58], we finetune it with our aggregated four-view UV map. Due to the explicitly introduced 3D knowledge and the texel-aligned input, our texture completion network produces reasonable and spatially correlated inpainting in the UV space.

Although our method can generate high-quality, 1K (10242 pixels) resolution texture maps through the above pipeline, some scenarios require a higher resolution for more detailed rendering. To address this limitation, we perform super-resolution on the generated texture maps. We finetune a Real-ESRGAN [62] model using our baked texture maps to perform 4 times up-scale. At can be produced. At last, precise and detailed texture maps at a resolution of 4K (40962 pixels)

### 5. Experiment

We conduct qualitative and quantitative experiments to verify the superiority of our method. Ablation studies are also performed to prove the effectiveness of our proposed modules. Furthermore, we demonstrate examples of our method in practical applications to illustrate its real-world significance.

#### 5.1. Implementation Details

We choose FLUX.1-dev [6] as our base generation model and utilize LoRA [61] to retarget it for multi-view image grid generation. We use T5 [63] text encoder and Redux [6] image encoder to align multi-modal information in a shared embedding space. For the WeighterNet, we use a Vision Transformer (ViT) [64] combined with a Point Transformer [65] to achieve UV and spatial perception. For texture completion and enhancement, we leverage the inpainting capabilities of TEXGen [58] and finetune it on our synthesized partial texture data. We then deploy Real-ESRGAN [62] finetuned on the UV map dataset to achieve texture refinement and super-resolution.

Data preparation. We select 100K 3D objects from the Objaverse dataset [66] as our training set. During training,

- Table 1. Quantitative evaluation result of text-to-texture generation. The best results are in bold, while the second-best ones are underlined. KID multiplied by 10−4.

FID ↓ KID ↓ User Preference TEXTure 78.269 107.062 4.4% Text2Tex 89.008 94.423 11.3%

SyncMVD 73.385 41.737 22.5% Paint3D 77.801 123.605 17.8% TEXGen 72.900 61.729 15.7%

Ours 71.621 58.465 28.3%

- Table 2. Quantitative evaluation result of image-to-texture generation. The best results are indicated in bold. KID multiplied by 10−4.

FID ↓ KID ↓ User Preference Paint3D 83.977 267.132 28.6%

##### Ours 59.492 62.089 71.4%

we randomly select rendered images and GPT-4V labeled caption as prompt. Four surrounding, relatively fixed depth and albedo images are also rendered as training data. For WeighterNet training, we simulate denoised texture maps T˜t at time step t by first adding noise on the ground-truth rendered views to get xt. Then, we can use Equ. (5) and (6) to get the simulate T˜t as training input.

Baselines and metrics. We choose TEXTure [7], Text2Tex [8], SyncMVD [18], Paint3D [14] and TEXGen [58] as our comparison baseline In particular, we use a depth-controlled Flux [6] model to generate the meshaligned front view required by TEXGen to ensure fairness. We randomly select 100 unseen 3D objects from Google Scanned Objects (GSO) [67] as the evaluation dataset to better demonstrate our generalization capabilities. Common metrics, including FID [68] and KID [69], are used for quantitative comparison. To further prove our effectiveness, we conduct a user study by asking users to select the texture generation result with the highest quality that best matches the text or image prompt. The results are represented as the proportion of user preference.

#### 5.2. Comparisons on Text-to-Texture

We conduct experiments on text-to-texture generation. We only use text as the condition for this task and set all αi to 0. The results are shown in Tab. 1 and Fig. 4. Our model achieves the best FID score and user preference in the quantitative comparison. The qualitative experiments show that our model can produce text-coherent results without ghost artifacts.

###### TEXTure Text2Tex Paint3D SyncMVD TEXGen Ours

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

“Areddemon

wings,anda

withhorns,

sword.” “Adogin

[Figure 124]

[Figure 125]

sketchstyle.”

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

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

“Abronzestatue

ofahorseontop

ofamarble

pedestal.”

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

modelofahorse,

Minecraftstyle.

“Awoodenblock

resembling

Figure 4. Qualitative results on text-to-texture generation.

Conditions Paint3D Ours

[Figure 158]

[Figure 159]

| |
|---|

[Figure 160]

[Figure 161]

[Figure 162]

| |
|---|

| |
|---|

| |
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 5. Qualitative results on image-to-texture generation.

#### 5.3. Comparisons of Image-to-Texture

We conduct quantitative experiments on image-to-texture generation using the rendered images as input. We use fixed trigger words as text prompts and set α0 = 1 for the image embedding. The results presented in Tab. 2 show that our method significantly outperforms others, indicating better image understanding and stronger generative capabilities. For the qualitative experiments, we use imperfectly-aligned, in-the-wild images as conditions to test the generalizability of our method. From the results shown in Fig. 5, we can tell that our method can infer semantic correspondences from the image information to generate results that meet both image and geometry conditions.

#### 5.4. Ablation Studies

We conduct ablation studies to demonstrate the effect of our view synchronization, WeighterNet (replaced by simple cosine weight), and the image-based CFG (replaced by normal text-based CFG) modules. Quantitative and qualitative results can be seen in Table 3 and Fig. 7, respectively. Although our multi-view generation model may still exhibit local misalignment in the UV space, our synchronization strategy and the 3D-aware weighting network can solve such a problem and generate flawless texture maps. Due to multiple encode-decode operations during the sampling process, the original text-based CFG can lead to a decline in generation quality. Meanwhile, our image-based CFG method can enhance condition consistency and generate higher-quality results.

#### 5.5. Applications

We show the applications of our method, including tasks of text-to-texture, image-to-texture, text-guided image refinement, and stylization using a reference image. The results are shown in Fig. 6. These results show that our method can cope with different user needs with different prompts in actual production and flexibly generate high-quality textures that meet user intentions.

### 6. Conclusion

We introduce FlexPainter, a novel framework for generating high-quality, consistent texture maps with flexible generation. We achieve multi-modality mixing and flexible conditional control by constructing a shared semantic space. The proposed image-based CFG also enables structure sup-

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

user’s demands

“A conch in rainbow colorful style”

“A conch in ice and snow covered style”

[Figure 189]

“A conch with a red heart on it”

[Figure 190]

stylize it! stylize it!

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Results

[Figure 198]

“A bird figurine with colorful feathers on a rock”

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

user’s demands

“A colorful bird figurine on a rock”

“A blue bird figurine on a rock in in a ice and snow covered style”

[Figure 203]

stylize it! stylize it!

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

results

- Figure 6. Our Applications includes tasks of text-to-texture, image-to-texture, text-guided image refinement, and reference image-based stylization.

[Figure 211]

[Figure 212]

[Figure 213]

w/o view-sync w/o WeighterNet w/o image CFG Our Full Model

[Figure 214]

| |
|---|

| |
|---|

- Figure 7. Qualitative results of our ablation study. Our full model achieves consistent and high-quality generation results, while the ablated methods suffer from ghosting artifacts and degraded quality.

### References

- [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [2] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [3] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [4] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [5] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [6] Black-Forest-Labs. Flux. https://github.com/ black-forest-labs/flux, 2023. 2, 6, 14
- [7] Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. Texture: Text-guided texturing of 3d shapes. In ACM SIGGRAPH 2023 conference proceedings, pages 1–11, 2023. 2, 3, 6
- [8] Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner. Text2tex: Text-driven texture synthesis via diffusion models. In Proceedings of the

Table 3. Quantitative ablation studies on our view synchronization, WeighterNet, and image-based CFG. The best results are indicated in bold. KID multiplied by 10−4.

w/o view-sync

w/o WeighterNet

w/o image CFG

our full model

FID ↓ 80.635 76.688 79.041 71.621 KID ↓ 77.883 59.674 65.482 58.465

pression of the reference image and achieves high-fidelity stylization. For texture coherency, we use a multi-view image grid as a generation target for global understanding. The reprojection-based view synchronization and weighting module enhances multi-view alignment for local consistency. Finally, a texture completion and enhancement module produces high-resolution textures. Extensive experiments show that FlexPainter outperforms existing methods, offering superior semantic alignment and multi-view consistency, making it a practical solution for 3D artists and designers. Finally, a texture completion and enhancement module generates full high-resolution textures, significantly improving the visual quality and detail of the final output.

- IEEE/CVF International Conference on Computer Vision, pages 18558–18568, 2023. 2, 6
- [9] Jinbo Wu, Xing Liu, Chenming Wu, Xiaobo Gao, Jialun Liu, Xinqi Liu, Chen Zhao, Haocheng Feng, Errui Ding, and Jingdong Wang. Texro: Generating delicate textures of 3d models by recursive optimization. arXiv preprint arXiv:2403.15009, 2024. 3
- [10] Xiaoyu Xiang, Liat Sless Gorelik, Yuchen Fan, Omri Armstrong, Forrest Iandola, Yilei Li, Ita Lifshitz, and Rakesh Ranjan. Make-a-texture: Fast shape-aware texture generation in 3 seconds. arXiv preprint arXiv:2412.07766, 2024. 3
- [11] Chenjian Gao, Boyan Jiang, Xinghui Li, Yingpeng Zhang, and Qian Yu. Genesistex: Adapting image denoising diffusion to texture space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4620–4629, 2024. 3
- [12] Dong Huo, Zixin Guo, Xinxin Zuo, Zhihao Shi, Juwei Lu, Peng Dai, Songcen Xu, Li Cheng, and Yee-Hong Yang. Texgen: Text-guided 3d texture generation with multi-view sampling and resampling. In European Conference on Computer Vision, pages 352–368. Springer, 2025. 2, 3
- [13] Wei Cheng, Juncheng Mu, Xianfang Zeng, Xin Chen, Anqi Pang, Chi Zhang, Zhibin Wang, Bin Fu, Gang Yu, Ziwei Liu, et al. Mvpaint: Synchronized multi-view diffusion for painting anything 3d. arXiv preprint arXiv:2411.02336, 2024. 2, 3
- [14] Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu, Yong Liu, and Gang Yu. Paint3d: Paint anything 3d with lighting-less texture diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4252–4262, 2024. 2, 3, 6
- [15] DaDong Jiang, Xianghui Yang, Zibo Zhao, Sheng Zhang, Jiaao Yu, Zeqiang Lai, Shaoxiong Yang, Chunchao Guo, Xiaobo Zhou, and Zhihui Ke. Flexitex: Enhancing texture generation with visual guidance. arXiv preprint arXiv:2409.12431, 2024. 2, 3
- [16] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3

- [17] Zhiyu Xie, Yuqing Zhang, Xiangjun Tang, Yiqian Wu, Dehan Chen, Gongsheng Li, and Xiaogang Jin. Styletex: Style image-guided texture generation for 3d models. ACM Transactions on Graphics (TOG), 43(6):1–14, 2024. 2, 3
- [18] Yuxin Liu, Minshan Xie, Hanyuan Liu, and Tien-Tsin Wong. Text-guided texturing by synchronized multi-view diffusion. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11,

2024. 2, 3, 6

- [19] Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, and Kangxue Yin. Texfusion: Synthesizing 3d textures with text-guided image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4169–4181, 2023. 2, 3
- [20] Raphael Bensadoun, Yanir Kleiman, Idan Azuri, Omri Harosh, Andrea Vedaldi, Natalia Neverova, and Oran Gafni.

- Meta 3d texturegen: Fast and consistent texture generation for 3d objects. arXiv preprint arXiv:2407.02430, 2024. 2, 3
- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2, 4, 14
- [22] Hongkun Zhang, Zherong Pan, Congyi Zhang, Lifeng Zhu, and Xifeng Gao. Texpainter: Generative mesh texturing with multi-view consistency. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2, 3
- [23] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023. 2
- [24] Jianglong Ye, Peng Wang, Kejie Li, Yichun Shi, and Heng Wang. Consistent-1-to-3: Consistent image to 3d view synthesis via geometry-aware diffusion models. In 2024 International Conference on 3D Vision (3DV), pages 664–674. IEEE, 2024. 2
- [25] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2
- [26] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2
- [27] Sixu Li, Chaojian Li, Wenbo Zhu, Boyang Yu, Yang Zhao, Cheng Wan, Haoran You, Huihong Shi, and Yingyan Lin. Instant-3d: Instant neural radiance field training towards ondevice ar/vr 3d reconstruction. In Proceedings of the 50th Annual International Symposium on Computer Architecture, pages 1–13, 2023. 2
- [28] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023.
- [29] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9970–9980, 2024.
- [30] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2
- [31] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2025. 2
- [32] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217, 2023.

- [33] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and Filippos Kokkinos. Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation. arXiv preprint arXiv:2402.08682, 2024.
- [34] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023.
- [35] Haohan Weng, Tianyu Yang, Jianan Wang, Yu Li, Tong Zhang, CL Chen, and Lei Zhang. Consistent123: Improve consistency for one image to 3d object synthesis. arXiv preprint arXiv:2310.08092, 2023.
- [36] Xinli Xu, Wenhang Ge, Jiantao Lin, Jiawei Feng, Lie Xu, HanFeng Zhao, Shunsi Zhang, and Ying-Cong Chen. Flexgen: Flexible multi-view generation from text and image inputs. arXiv preprint arXiv:2410.10745, 2024. 2
- [37] Johannes Kopf, Chi-Wing Fu, Daniel Cohen-Or, Oliver Deussen, Dani Lischinski, and Tien-Tsin Wong. Solid texture synthesis from 2d exemplars. In ACM SIGGRAPH 2007 papers, pages 2–es. 2007. 3
- [38] Sylvain Lefebvre and Hugues Hoppe. Appearance-space texture synthesis. ACM Transactions on Graphics (TOG), 25(3):541–548, 2006.
- [39] Greg Turk. Texture synthesis on surfaces. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques, pages 347–354, 2001.
- [40] Li-Yi Wei, Sylvain Lefebvre, Vivek Kwatra, and Greg Turk. State of the art in example-based texture synthesis. Eurographics 2009, State of the Art Report, EG-STAR, pages 93– 117, 2009. 3
- [41] Michael Oechsle, Lars Mescheder, Michael Niemeyer, Thilo Strauss, and Andreas Geiger. Texture fields: Learning texture representations in function space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4531–4540, 2019. 3
- [42] Yawar Siddiqui, Justus Thies, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. Texturify: Generating textures on 3d shape surfaces. In European Conference on Computer Vision, pages 72–88. Springer, 2022.
- [43] Alexey Bokhovkin, Shubham Tulsiani, and Angela Dai. Mesh2tex: Generating mesh textures from image queries. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8918–8928, 2023.
- [44] An-Chieh Cheng, Xueting Li, Sifei Liu, and Xiaolong Wang. Tuvf: Learning generalizable texture uv radiance fields. arXiv preprint arXiv:2305.03040, 2023. 3
- [45] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 3
- [46] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 3

- [47] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3
- [48] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023.
- [49] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22246–22256, 2023.
- [50] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12663–12673, 2023.
- [51] Yu-Ying Yeh, Jia-Bin Huang, Changil Kim, Lei Xiao, Thu Nguyen-Phuoc, Numair Khan, Cheng Zhang, Manmohan Chandraker, Carl S Marshall, Zhao Dong, et al. Texturedreamer: Image-guided texture synthesis through geometryaware diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4304–4314, 2024. 3
- [52] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [53] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 3
- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3
- [55] Shang Liu, Chaohui Yu, Chenjie Cao, Wen Qian, and Fan Wang. Vcd-texture: Variance alignment based 3d-2d codenoising for text-guided texturing. In European Conference on Computer Vision, pages 373–389. Springer, 2024. 3
- [56] Kangle Deng, Timothy Omernick, Alexander Weiss, Deva Ramanan, Jun-Yan Zhu, Tinghui Zhou, and Maneesh Agrawala. Flashtex: Fast relightable mesh texturing with lightcontrolnet. In European Conference on Computer Vision, pages 90–107. Springer, 2025. 3
- [57] Xin Yu, Peng Dai, Wenbo Li, Lan Ma, Zhengzhe Liu, and Xiaojuan Qi. Texture generation on 3d meshes with pointuv diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4206–4216, 2023. 3
- [58] Xin Yu, Ze Yuan, Yuan-Chen Guo, Ying-Tian Liu, Jianhui Liu, Yangguang Li, Yan-Pei Cao, Ding Liang, and Xiaojuan

- Qi. Texgen: a generative diffusion model for mesh textures. ACM Transactions on Graphics (TOG), 43(6):1–14, 2024. 3, 6, 14
- [59] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [60] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 3
- [61] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 4, 6
- [62] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914,

2021. 6, 14

- [63] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 6
- [64] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 6
- [65] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4840–4851, 2024. 6
- [66] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023. 6, 14
- [67] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 6
- [68] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [69] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 6
- [70] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 14

[Figure 215]

###### mesh result mesh result

“A cylindrical cosmetic bag with a flora pattern and a zipper closure”

“A dragon head with horns and stylized head”

“A bipedal theropod dinosaur, resembling a Tyrannosaurus Rex”

“A pink seashell”

“A colorful cylindrical pencil case with a pattern of geometric shapes and a zipper closure”

“A pink heart-shaped clock with an arrow and the words TO MY ONE & ONLY”

“A Don’t Wake Daddy board game box featuring various images and text describing the game”

“A wooden-framed picture of a building on a wall panel”

- Figure 8. More cases of our text-to-texture generation.

[Figure 216]

mesh condition result mesh condition result

- Figure 9. More cases of our imgae-to-texture generation.

[Figure 217]

[Figure 218]

###### condition result condition result

mesh mesh

[Figure 219]

“A bird figurine on a rock with colorful feathers”

“A bench in colorful style”

“A blue ceramic bird figurine on a rock in an ice and snow covered style”

###### “A bench in ice and snow covered style”

condition result condition result

[Figure 220]

mesh mesh

“An axe with a pink head and yellow handle”

“A mannequin wearing a pink flora dress with flowers on it”

“An axe in ice and snow covered style”

“A mannequin wearing a blue flora dress with flowers on it”

Figure 10. More cases of our applications.

## Supplementary Material for FlexPainter: Flexible and Multi-View Consistent Texture Generation

### A. Additional Implementation Details

#### A.1. Baking Texture

When preparing for texture data, we found one problem in Objaverse[66]. Instead of a single unified texture, some meshes are composed of multiple parts, each with its texture image. For those meshes, we use Blender’s Smart UV Project function to re-unfold the UVs and then bake the diffuse color to get a unified texture image. For the texture completion net, we bake 10K texture maps at 1K (10242 pixels) resolution; for the texture enhancement net, we bake the same texture maps at a 4K (40962 pixels) resolution.

#### A.2. Training of LoRA.

During LoRA training, we use Adam [70] optimizer with a learning rate of 10−5. The LoRA rank is set to 256. We train the model for 20K steps with 8 NVIDIA-A800 GPUs and a batch size of 4. We use ”’a grid of 2x2 multi-view image. white background.” for the trigger word.

#### A.3. Training of WeighterNet.

For the training of WeighterNet, we use Adam optimizer with a learning rate of 10−4. The weight of the supervision loss are set to: λpec = 1,λcyc = 0.5. Additionally, a smooth loss Lsm with weight λsm = 0.2 is used to regularize the training process:

Lsm = ((∇xI)2i,j + (∇yI)2i,j), (12) and the full supervision loss L is set to:

L = λpercLperc + λcycLcyc + λsmLsm (13)

We train the model for 100K step with 8 NVIDIA-A800 GPU and batchsize of 2.

#### A.4. Training of Completion and Enhancement Net.

For texture completion net, we start from the checkpoint provided by TEXGen [58]. However, since the original TEXGen is trained using a partial texture map reprojected from a single view, direct extension into a larger partial UV map from four views may result in domain variance. To this end, we generate a partial texture map using our four-view camera setting and finetune the original model. We use an Adam optimizer with a learning rate of 10−4 and train the model for 20K steps on 8 NVIDIA-A800 GPUs with a batch size 2.

For texture enhancement net, we choose RealESRGAN[62] as our base model. Given 4K (4096 × 4096)

baked texture maps, we generate corresponding low-quality images with the degradation process described in RealESRGAN[62]. During training, we use Adam [70] optimizer with learning rate of 10−4. Additionally, we utilize L1 Loss, GAN Loss, and Perceptual Loss for pixel level, content level, and style level supervision:

###### L = λpercLperc + λpLp + λgLg (14)

where λperc,λp,λg are set to 1.0, 1.0, 0.1 respectively. We train the model for 400K step with 1 NVIDIA-A800 GPU and batchsize of 12.

Detailed pipeline of the texture completion and enhancement module is shown in Fig.S1.

#### A.5. Implementaion of CFG

Since the original checkpoint from FLUX.1-dev [6] uses a distilled CFG mechanism, we implement an explicit CFG method following [21] during sampling. For all of our experiments, the distilled CFG scale is set to 6, and the explicit CFG scale is set to 2. We use the embedding of a white image as the negative embedding for the explicit CFG.

### B. More Qualitative Results

Our main paper introduces various applications with our model, including text-to-texture, image-to-texture, and texture stylization. We demonstrate more results in Fig.S2, Fig.S3. Additionally in Fig.S4 to Fig. Fig.S6, we detailed conditions and generation results shown in the teaser of the main paper. Also, we attached a video to the supplementary file to present the texture results using a dynamic approach.

### C. Details of our user study

In our user study, we use 120 cases for text-to-texture generation and 50 cases for image-to-texture generation. We shuffle all the cases and separate them randomly into 6 questionnaires and distribute them to 100 interviewees. We showcase examples of our questionnaires from Fig.S7 to Fig.S12. Notice that the cases are shown in video.

### D. Limitations and Future Work

Although our model has already achieved superior results, some challenges remain to be solved. While utilizing depth maps as geometric conditions is effective, it often results in the loss of geometric detail from the original mesh. This discrepancy can lead to the generation results in Stage I

Texture Completion Module

partial texture

Texture Enhancement Module

[Figure 221]

| | | |
|---|---|---|
|UV|Blo|cks|
| | | |
|Point|Bl|ocks|
| | | |

[Figure 222]

[Figure 223]

| |
|---|

| | | |
|---|---|---|
|R ESR|eal GA|-<br><br>N|
| | | |

| |
|---|

[Figure 224]

completed 1K texture 4K texture

[Figure 225]

|[Figure 226]|
|---|

|[Figure 227]|
|---|

[Figure 228]

[Figure 229]

partial mask

textured mesh

mask map position map

Figure S1. Detailed pipeline of the texture completion and enhancement module. We use a 3D-aware texture completion module containing UV blocks and point blocks to complete the partial texture. Then, we use a texture enhancement module to generate the super-resolution results.

not fully aligning with the original mesh. Moreover, we are still exploring whether explicit 3D information can be used as guidance to steer the inpainting process in Stage II. Due to the albedo data used during training, our generated textures are free from ambient lighting. However, we have observed that applying these textures in practical 3D generation pipelines still requires designers to spend considerable time adjusting the lighting. Therefore, our future work will explore methods to generate textures that align with designers’ expected lighting effects to achieve more flexible, thereby reducing design time.

Prompt TEXTure Text2Tex SyncMVD Paint3D TEXGen Ours

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

handle,cylindrical

design,tapered

symmetrical

AxeModel,

grip

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

handholdingagun,

[Figure 253]

stonegargoyle,a

andablackbox.

Abullstatue,a

cubeshaped,sixfaces,

Awoodencratemodel.

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

equal-sizedsides,. Awhitetoycement

truck.Rotatingdrum,

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

Truckcabin.

Asmallbluebird.rounded

stand-alonefigure,stable

projection,compactsize,

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

protrudingbeak,tail

body,smallhead,

base

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

geometricshape. Aflesh-colored

hammerwitha

Awoodentoy

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

seashell.Spiraled,

pointedtip.

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

cloth,andpaperson

itemssuchaspaint,

tablewithvarious

them,includinga

Asideboardand

metalboxand

cabinetwith

newspapers.

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

AblackNokiaE63

flipdesign,hinged

rectangularbody,

top,thinprofile.

cellphone.

varyingprotrusions,

truncatedtop,non-

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

uniformdiameter.

cylindricalbase,

irregularform,

Atreestump.

- Figure S2. More cases of our text-to-texture generation.

Ref. Image Paint3D Ours Ref. Image Paint3D Ours

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

- Figure S3. More cases of our image-to-texture generation.

[Figure 390]

condition result condition result

mesh mesh

“A conch shell in rainbow colorful style”

“A pink tree stum”

“A conch shell in ice and snow covered style”

“A tree stum in ice and snow covered style”

Figure S4. More cases of our applications.

[Figure 391]

condition result condition result

mesh mesh

“A box in colorful style”

“A green bowl”

“A bowl in ice and snow covered style”

“A box in ice and snow covered style”

Figure S5. More cases of our applications.

[Figure 392]

condition result condition result

mesh mesh

“A pink cup with a white handle, and a flower on the side”

“A pink barrel”

“A cup in ice style”

“A barrel with snow covered”

Figure S6. More cases of our applications.

[Figure 393]

[Figure 394]

###### Figure S7. Our user study interface.

[Figure 395]

[Figure 396]

###### Figure S8. Our user study interface.

[Figure 397]

[Figure 398]

###### Figure S9. Our user study interface.

[Figure 399]

[Figure 400]

###### Figure S10. Our user study interface.

[Figure 401]

[Figure 402]

###### Figure S11. Our user study interface.

[Figure 403]

[Figure 404]

###### Figure S12. Our user study interface.

