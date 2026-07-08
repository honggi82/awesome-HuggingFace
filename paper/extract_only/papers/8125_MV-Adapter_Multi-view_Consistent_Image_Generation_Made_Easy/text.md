# arXiv:2412.03632v1[cs.CV]4Dec2024

## MV-ADAPTER: MULTI-VIEW CONSISTENT IMAGE GENERATION MADE EASY

Zehuan Huang1, Yuan-Chen Guo2†, Haoran Wang3, Ran Yi3, Lizhuang Ma3, Yan-Pei Cao2 , Lu Sheng1 1School of Software, Beihang University 2VAST 3Shanghai Jiao Tong University Project page: https://huanngzh.github.io/MV-Adapter-Page/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(cartoon) 1 boy, kurosaki ichigo, bleach, black hakama, bankai…

(realistic) macro shot, parrot, colorful, dark shot, realistic …

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

(chalk sketch) an elderly woman with a serene expression…

(SDXL-Lightning) mecha vampire girl chibi

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

(ControlNet Tile) A character wearing blue and pink outfit…

###### (ControlNet Scribble) Sonic the Hedgehog

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Adaptability

Camera-guided Text-to-Multiview Generation Camera-guided Image-to-Multiview Generation

Versatility

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

The warrior Aragorn from Lord of the Rings

Geometry-guided Text-to-Multiview Generation Geometry-guided Image-to-Multiview Generation

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

A muscular cartoon cat wearing blue jeans and a brown belt…

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Input Generated Multi-view Images Input Generated Multi-view Images

Figure 1: MV-Adapter is a versatile plug-and-play adapter that turns existing pre-trained text-toimage (T2I) diffusion models to multi-view image generators. Row 1,2,3: results by integrating MVAdapter with personalized T2I models, distilled few-step T2I models, and ControlNets (Zhang et al., 2023), demonstrating its adaptability. Row 4,5: results under various control signals, including view-guided or geometry-guided generation with text or image inputs, showcasing its versatility.

ABSTRACT

Existing multi-view image generation methods often make invasive modifications to pre-trained text-to-image (T2I) models and require full fine-tuning, leading to (1) high computational costs, especially with large base models and highresolution images, and (2) degradation in image quality due to optimization difficulties and scarce high-quality 3D data. In this paper, we propose the first adapterbased solution for multi-view image generation, and introduce MV-Adapter, a versatile plug-and-play adapter that enhances T2I models and their derivatives without altering the original network structure or feature space. By updating fewer parameters, MV-Adapter enables efficient training and preserves the prior knowledge embedded in pre-trained models, mitigating overfitting risks. To efficiently model the 3D geometric knowledge within the adapter, we introduce innovative designs that include duplicated self-attention layers and parallel attention architec-

Corresponding authors: lsheng@buaa.edu.cn, caoyanpei@gmail.com; †: project lead

ture, enabling the adapter to inherit the powerful priors of the pre-trained models to model the novel 3D knowledge. Moreover, we present a unified condition encoder that seamlessly integrates camera parameters and geometric information, facilitating applications such as text- and image-based 3D generation and texturing. MV-Adapter achieves multi-view generation at 768 resolution on Stable Diffusion XL (SDXL), and demonstrates adaptability and versatility. It can also be extended to arbitrary view generation, enabling broader applications. We demonstrate that MV-Adapter sets a new quality standard for multi-view image generation, and opens up new possibilities due to its efficiency, adaptability and versatility.

- 1 INTRODUCTION Multi-view image generation is a fundamental task with significant applications in areas such as

- 2D/3D content creation, robotics perception, and simulation. With the advent of text-to-image (T2I) diffusion models (Ramesh et al., 2022; Nichol et al., 2022; Saharia et al., 2022; Ramesh et al., 2021; Balaji et al., 2022; Podell et al., 2024; Mokady et al., 2023), there has been considerable progress in generating high-quality single-view images. Extending these models to handle multiview generation holds the promise of unifying text, image, and 3D data into a cohesive framework.

Recent attempts on multi-view image generation (Shi et al., 2023b; Tang et al., 2023; 2024; Huang et al., 2024; Gao et al., 2024; Liu et al., 2023a; Long et al., 2024; Li et al., 2024; Kant et al., 2024; Zheng & Vedaldi, 2024; Wang & Shi, 2023) involve fine-tuning T2I models on large-scale

- 3D datasets (Deitke et al., 2023; Yu et al., 2023) and propose modeling 3D consistency across images by applying attention on relevant pixels in different views. However, this is computationally challenging when working with large base T2I models and high-resolution images, as it requires at least n view images to be processed simultaneously during training. Existing advanced methods (Li et al., 2023a; 2024) still struggle with 512 resolution, which is far from the 1024 or higher that modern T2I models can achieve. Moreover, the scarcity of high-quality 3D training data exacerbates the optimization difficulty when performing full model fine-tuning, resulting in a degradation in the quality of the generated multi-view images. These limitations primarily stem from the invasive changes to base models and full tuning.

To address these challenges, we propose the first adapter-based solution for multi-view image generation. The adapter mechanism plays a crucial role in this context for several reasons: First, adapters are easy to train. They require updating only a small number of parameters, making the training process faster and more memory-efficient. This property has become increasingly critical as state-of-the-art T2I models grow in scale, making full fine-tuning infeasible. Second, adapters help in preserving prior knowledge embedded in the pre-trained models. Adapters mitigate the risk of overfitting by constraining the optimization space through fewer trainable parameters, allowing the model to retain its learned priors while adapting to multi-view generation. Third, adapters offer adaptability and ease of use. They are plug-and-play modules and can be applied to different variants of base models, including fine-tuned versions (Ruiz et al., 2023) and LoRAs (Hu et al., 2021).

Building on the importance of adapters for the multi-view generation task and adhering to the principle of preserving the original network structure and feature space of the base T2I model, we propose MV-Adapter, a versatile plug-and-play adapter that enhances T2I models and their derivatives for multi-view generation under various conditions. To achieve this, we design an effective adapter framework with innovative features. Unlike existing methods (Shi et al., 2023b;a) that modify the base model’s self-attention layers to include multi-view or reference features, which disrupts learned priors and requires full model fine-tuning, we duplicate the self-attention layers to create new multiview attention and image cross-attention layers, and initializes the output projections to zero. We further enhance the effectiveness of our attention layers through a parallel organization structure, ensuring that the new layers fully inherit the powerful priors of the pre-trained self-attention layers, thus enabling efficient learning of geometric knowledge. Additionally, we introduce a unified condition embedding and encoder that seamlessly integrates camera parameters and geometric information into spatial map representations, enhancing the model’s versatility and applicability.

By leveraging our adapter design, we successfully achieve the multi-view generation at 768 resolution on Stable Diffusion XL (SDXL) (Podell et al., 2024). As shown in Fig. 1, our trained MVAdapter demonstrates both adaptability and versatility. It seamlessly applies to derivatives of the

base model (Ruiz et al., 2023; Hu et al., 2021; Zhang et al., 2023; Mou et al., 2024) for customized or controllable multi-view generation, while simultaneously supporting camera and geometry guidance, which benefits applications in 3D generation and texture generation. Moreover, MV-Adapter can be extended to arbitrary view generation, enabling broader applications.

In summary, our contributions are as follows: (1) We propose the first adapter-based approach that enhances efficiency and is able to work with larger base models for higher performance. (2) We introduce an innovative adapter framework that efficiently models 3D geometric knowledge and supports versatile applications like 3D generation and texture generation. (3) Our MV-Adapter can be extended to generate images from arbitrary viewpoints, facilitating a wider range of downstream tasks. (4) MV-Adapter provides a framework for decoupled learning that offers insights into modeling new types of knowledge, such as physical or temporal knowledge.

- 2 RELATED WORK

Text-to-image diffusion models. Text-to-image (T2I) generation (Ramesh et al., 2022; Nichol et al., 2022; Saharia et al., 2022; Ramesh et al., 2021; Balaji et al., 2022; Podell et al., 2024; Mokady et al., 2023) has made remarkable progress, particularly with the advancement of diffusion models (Ho et al., 2020; Song et al., 2020; Dhariwal & Nichol, 2021; Ho & Salimans, 2022). Guided diffusion (Dhariwal & Nichol, 2021) and classifier-free guidance (Ho & Salimans, 2022) improved text conditioning and generation fidelity. DALL-E2 (Ramesh et al., 2022) leverages CLIP (Radford et al., 2021) for better text-image alignment. The Latent Diffusion Model (Rombach et al., 2022), also known as Stable Diffusion, enhances efficiency by performing diffusion in the latent space of an autoencoder. Stable Diffusion XL (Podell et al., 2024), a two-stage cascade diffusion model, has greatly improved the generation of high-frequency details and image quality.

Derivatives and extensions of T2I models. To facilitate creation with pre-trained T2Is, various derivative models and extensions have been developed, focusing on model distillation for efficiency (Meng et al., 2023; Song et al., 2023; Luo et al., 2023; Lin et al., 2024) and controllable generation (Cao et al., 2024). These derivatives encompass personalization (Ruiz et al., 2023; Gal et al., 2022; Hu et al., 2021; Shi et al., 2024; Wang et al., 2024a; Ma et al., 2024; Song et al., 2024; Kumari et al., 2023; Ye et al., 2023), and spatial control (Mou et al., 2024; Zhang et al., 2023). Typically, they employ adapters or fine-tuning methods to extend functionality while preserving the original feature space of the pre-trained models. Our work adheres to non-intrusive principle, ensuring compatibility with these derivatives or extensions for broader applications.

Multi-view Generation with T2I models. Multi-view generation methods (Shi et al., 2023b; Tang et al., 2023; 2024; Huang et al., 2024; Gao et al., 2024; Liu et al., 2023a; Long et al., 2024; Li et al., 2024; Kant et al., 2024; Zheng & Vedaldi, 2024; Wang & Shi, 2023; Jeong et al., 2025) extend T2I models by leveraging large-scale 3D datasets (Deitke et al., 2023; Yu et al., 2023). For instance, MVDream (Shi et al., 2023b) integrates camera embeddings and expands the self-attention mechanism from 2D to 3D for cross-view connections, while SPAD (Kant et al., 2024) enhances spatial relational modeling by applying epipolar constraints to cross-view attention. Era3D (Li et al., 2024) introduces an efficient row-wise self-attention mechanism aligned with epipolar lines across views, facilitating high-resolution multi-view generation. However, these methods typically require extensive parameter updates, altering the feature space of pre-trained T2I models and limiting their compatibility with T2I derivatives. Our work addresses this by introducing a multi-view adapter that harmonizes with pre-trained T2Is, significantly expanding the potential for diverse applications.

- 3 PRELIMINARY

Here we introduce the preliminary of multi-view diffusion models (Shi et al., 2023b; Kant et al., 2024; Li et al., 2024), which can help understand the common strategies in modeling multi-view consistency within T2I models.

Multi-view diffusion models. Multi-view diffusion models enhance T2Is by introducing multiview attention mechanism, enabling the generation of images that are consistent across different viewpoints. Several studies (Shi et al., 2023b; Wang & Shi, 2023) extend the self-attention of T2Is to include all pixels across multi-view images. Let fin denotes the input of the attention block, the

dense multi-view self-attention extends fin from the view itself to the concatenated feature sequence from n views. While this approach captures global dependencies, it is computationally intensive, as it processes all pixels of all views. To mitigate the computational cost, epipolar attention (Kant et al., 2024; Huang et al., 2024) leverages geometric relationships between views. Specifically, methods like SPAD (Kant et al., 2024) extend the self-attention by restricting fin to the view itself as well as patches along its epipolar lines.

Furthermore, when generating orthographic views at an elevation angle of 0◦, the epipolar lines align with the image rows. Utilizing this property, row-wise self-attention (Li et al., 2024) is introduced after the original self-attention layers in T2I models. The process is defined as:

#### fself = SelfAttn(fin) + fin; fmv = MultiViewAttn(fself) + fself (1)

where MultiViewAttn performs attention across the same rows in different views, effectively enforcing multi-view consistency with reduced computational overhead.

- 4 METHOD

MV-Adapter is a plug-and-play adapter that learns multi-view priors transferable to derivatives of T2Is without specific tuning, and enable

Attention Modules of MV-Adapter

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- them to generate multi-view consistent images under various conditions. As shown in Fig. 2, at inference, our MV-Adapter, which contains a condition guider and the decoupled attention layers, can be inserted into a personalized or distilled T2I to constitute the multi-view generator.

[Figure 49]

Module Insert

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Text with Optional Image

[Figure 58]

[Figure 59]

Multi-view Images

In detail, as shown in Fig. 3, the condition guider in Sec. 4.1 encodes the camera or geometry information, which supports both camera-guided and geometry-guided generation. Within the decoupled attention mechanism in Sec. 4.2, the additional multi-view attention layers learn multiview consistency, while the optional image crossattention layers are for image-conditioned generation. These new layers are duplicated from pre-trained spatial self-attention and organized in a parallel architecture. Sec. 4.3 elaborates on the training and inference processes of the MV-Adapter.

Camera or Geometry Guidance

Image Layers of Pre-trained T2I MV-Adapter

[Figure 60]

| |
|---|

Figure 2: Inference pipeline.

- 4.1 CONDITION GUIDER

We design a general condition guider that supports encoding both camera and geometric representations, enabling T2I models to perform multi-view generation under various guidance.

Camera conditioning. To condition on the camera pose, we use a camera ray representation (“raymap”) that shares the same height and width as the latent representations in the pre-trained T2I models and encodes the ray origin and direction at each spatial location (Watson et al., 2022; Sajjadi

- et al., 2022; Gao et al., 2024).

Geometry conditioning. Geometry-guided multi-view generation helps applications like texture generation. To condition on the geometry information, we use a global, rather than view-dependent representation that contains position maps and normal maps (Li et al., 2023b; Bensadoun et al., 2024). Each pixel in the position map represents the coordinates of the point on the shape, which provide point correspondences across different views. Normal maps provide orientation information and capture fine geometric details, helping produce detailed textures. We concatenate the position map and normal map along to form a composite geometric conditioning input for each view.

Encoder design. To encode the camera or geometry representation, we design a simple and lightweight condition guider for the conditioning maps cm (cm ∈ Rn×6×h×w). The condition guider consists of a series of convolutional networks, which contain feature extraction blocks and downsampling layers to adapt the feature resolution to the features in the U-Net encoder. The extracted multi-scale features are then added to the corresponding scales in the U-Net, enabling the

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

| | |[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>| |[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>| |[Figure 77]<br><br>|
|---|---|---|---|---|---|---|
| | | |[Figure 78]| | | |

… …

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

|[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]|
|---|

|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]|
|---|

Image features

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

|[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]|
|---|

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

(Optional) Image Condition

… …

Guidance for Each View

Camera Geometry

𝓏  : 

𝓏  : 

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Multi-view Attention

| |
|---|

Decoupled Attention Layers

C

OR

Image Cross-Attention

Condition Guider

Positions Normals

Ray Coordinates

Text Cross-Attention

Spatial Self-Attention

Figure 3: Overview of MV-Adapter. Our MV-Adapter consists of two components: 1) a condition guider that encodes camera or geometry condition; 2) decoupled attention layers that contain multiview attention for learning multi-view consistency, and optional image cross-attention to support image-conditioned generation, where we use the pre-trained U-Net to encode fine-grained information of the reference image. After training, MV-Adapter can be inserted into any personalized or distilled T2I to generate multi-view images while leveraging the specific strengths of base models.

model to integrate the conditioning information seamlessly at multiple levels. In theory, the input to our encoder is not limited to specific types of conditions; it can also be extended to a wider variety of maps, such as depth maps and pose maps.

- 4.2 DECOUPLED ATTENTION

We introduce a decoupled attention mechanism, where we retain the original spatial self-attention layers and duplicate them to create new multi-view attention layers as well as image cross-attention layers for image-conditioned generation. These three types of attention layers are organized in a parallel architecture, which ensures that the new attention layers can fully inherit the powerful priors of the pre-trained self-attention layers, thus enabling efficient learning of geometric knowledge.

Duplication of spatial self-attention. Our design adheres to the principle of preserving the original network structure and feature space of the base T2I model. Existing methods like MVDream (Shi et al., 2023b) and Zero123++ (Shi et al., 2023a) modify the base model’s self-attention layers to include multi-view or reference features, which disrupts the learned priors and requires full model fine-tuning. Here we duplicate the structure and weights of spatial self-attention layers to create new multi-view attention and image cross-attention layers, and initialize the output projections of these new attention layers to zero. This allows the new layers to learn geometric knowledge without interfering with the original model, ensuring excellent adaptability.

Parallel attention architecture. In the pretrained T2I model, the spatial self-attention layer and text cross-attention layer are connected serially through residual connections. Suppose feature fin is the input of the attention block, we can express the process as

Serial Architecture Parallel Architecture (Ours)

Multi-view Features

Multi-view Features

ResidualConnection

Spatial Self-Attention

Image Cross-Attention

SpatialSelf-Attention

Muti-view Attention

fself = SelfAttn(fin) + fin; fcross = CrossAttn(fself) + fself

(2)

Muti-view Attention

Image Cross-Attention

A straightforward method to incorporate new attention layers is to append them after the original layers, connecting them in a serial manner. However, the sequential arrangement may not effectively utilize the image priors modeled by the pre-trained self-attention layers, as it requires the new layers to learn from scratch. Even if we initialize the new layers with the pre-trained weights, the features input to these serially organized layers are in a different domain, causing the initialization to be ineffective. To

| | |
|---|---|
| | |

Text Cross-Attention

Text Cross-Attention

Figure 4: Serial vs parallel architecture.

fully exploit the effective priors of the spatial self-attention layers, we adopt a parallel architecture, as shown in Fig. 4. The process can be formulated as

fself = SelfAttn(fin) + MultiViewAttn(fin) + ImageCrossAttn(fin,fref) + fin (3)

where fref refers to features of the reference image. Since the features fin fed into the new layers are the same as those to the self-attention layer, we can effectively initialize them with the pre-trained layers to transfer the image priors. We zero-initialize the output projection layer of the new layers to ensure that the initial output does not disrupt the original feature space. This architectural choice allows the model to build upon the established priors, facilitating efficient learning of multi-view consistency and image-conditioned generation, while preserving the original space of the base T2Is.

Details of multi-view attention. We design different strategies for multi-view attention to meet the specific needs of different applications. For 3D object generation, we enable the model to generate multi-view images at an elevation of 0◦ and employ row-wise self-attention (Li et al., 2024). For 3D texture generation, considering the view coverage requirements, in addition to the four views evenly at elevation 0◦, we add two views from top and bottom. We then perform both row-wise and column-wise self-attention, enabling efficient information exchange among all views. For arbitrary view generation, we employ full self-attention (Shi et al., 2023b) in our multi-view attention layers.

Details of image cross-attention. To condition on reference images ci and achieve, we propose a novel method for incorporating detailed information from the image without altering the original feature space of the T2I model. We employ the pre-trained and frozen T2I U-Net as our image encoder. We pass the clear reference image into this frozen U-Net, setting the timestep t = 0, and

- then extract multi-scale features from the spatial self-attention layers. These fine-grained features contain detailed information about the subject and are injected into the denoising U-Net through the decoupled image cross-attention layers. In this way, we leverage the rich representations learned by the pre-trained model, enabling precise control over the generated content.

- 4.3 TRAINING AND INFERENCE

During training, we only optimize the MV-Adapter, while freezing weights of the pre-trained T2I models. We train MV-Adapter on the dataset with pairs of a reference image, text and n views, using the same training objective as T2I models:

0 ),ϵ∼N(0,I),ct,ci,cm,t[∥ϵ − ϵθ(zt1:n,ct,ci,cm,t)∥22] (4)

L = EE(x1:n

where ct, ci and cm represent texts, reference images and conditioning maps (i.e., camera or geometry conditions) respectively. We randomly zero out the features of the reference image to drop image conditions, enabling classifier-free guidance at inference. Similar to prior work (Blattmann

- et al., 2023; Hoogeboom et al., 2023), we shift the noise schedule towards high noise levels as we move from the T2Is to the multi-view diffusion model that captures data of higher dimensionality. We shift the log signal-to-noise ratio by log(n), where n is the number of generated views.

- 5 EXPERIMENTS

We implemented MV-Adapter on Stable Diffusion V2.1 (SD2.1) and Stable Diffusion XL (SDXL), training a 512 × 512 adapter for SD2.1 and a 768 × 768 adapter for SDXL using a subset of the Objaverse dataset (Deitke et al., 2023). Detailed configurations are provided in the Appendix.

- 5.1 CAMERA-GUIDED MULTI-VIEW GENERATION

Evaluation on community models and extensions. We evaluated MV-Adapter using representative T2Is and extensions, including personalized models (Ruiz et al., 2023; Hu et al., 2021), efficient distilled models (Luo et al., 2023; Lin et al., 2024), and plugins such as ControlNet (Zhang et al.,

- 2023). We present six qualitative results in Fig. 5. More results can be found in the Appendix.

Comparison with baselines. For text-to-multiview generation, we compared our MV-Adapter with MVDream (Shi et al., 2023b) and SPAD (Kant et al., 2024) on 1,000 prompts from the Objaverse dataset. The results are presented in Fig. 6 and Table 1. For image-to-multiview generation,

###### Animagine-xl RealVisXL

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

(realistic) Melissa Benoist dressed in her Supergirl outfit, smiling…

(2d cartoon) 1 boy, male focus, Gojo Satoru, white hair, masterpiece

Watercolor Style SDXL

###### Pokemon Trainer Sprite PixelArt

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

(watercolor) Game art, Female Soldier, wearing Otter-style ral-wtrclr…

(pixel art) 1 girl angel with 2 large angel wings and a halo, wearing…

###### LCM SDXL ControlNet Openpose

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

(few step) Samurai koala bear

(pose control) Albert Einstein

- Figure 5: Results with community models and extensions. Each sample corresponds to a distinct T2I model or extension. Information about the models can be found in the Appendix.

Table 1: Quantitative comparison on cameraguided text-to-multiview generation.

Method FID↓ IS↑ CLIP Score↑

MVDream 32.15 14.38 31.76 SPAD 48.79 12.04 30.87 Ours (SD2.1) 31.24 15.01 32.04 Ours (SDXL) 29.71 16.38 33.17

Table 2: Quantitative comparison on cameraguided image-to-multiview generation.

Method PSNR↑ SSIM↑ LPIPS↓ ImageDream 19.280 0.8472 0.1218 Zero123++ 20.312 0.8417 0.1205 CRM 20.185 0.8325 0.1247 SV3D 20.042 0.8267 0.1396 Ouroboros3D 20.810 0.8535 0.1193 Era3D 20.890 0.8601 0.1199 Ours (SD2.1) 20.867 0.8695 0.1147 Ours (SDXL) 22.131 0.8816 0.1002

we conduct comparison with ImageDream (Wang & Shi, 2023), Zero123++ (Shi et al., 2023a), CRM (Wang et al., 2024b), SV3D (Voleti et al., 2024), Ouroboros3D (Wen et al., 2024), and Era3D (Li et al., 2024) on the Google Scanned Objects (GSO) dataset (Downs et al., 2022), as results shown in Fig. 7 and Table 2. Experiments indicate that, by preserving the original feature space of T2I models, our MV-Adapter achieves higher visual fidelity and consistency with conditions.

- 5.2 GEOMETRY-GUIDED MULTI-VIEW GENERATION

Evaluation on community models and extensions. We evaluated our geometry-guided model with T2I derivative models. The results in Fig. 8 demonstrate the adaptability of MV-Adapter in seamlessly integrating with different base models.

Comparison with baselines. We compare our text- and image-conditioned multi-view-based texture generation method (see Sec. 5.4) with four state-of-the-art methods, including TEXTure (Richardson et al., 2023), Text2Tex (Chen et al., 2023), Paint3D (Zeng et al., 2024), SyncMVD (Liu et al., 2023b), and FlashTex (Deng et al., 2024). For our image-to-texture model, we used ControlNet (Zhang et al., 2023) to generate reference images conditioned on text and depth maps. As shown in Fig. 10 and Table 3, compared to these project-and-inpaint or synchronized multi-view texturing methods, our approach fine-tunes additional modules to model geometric associations and preserves the generative capabilities of the base T2I model, thereby producing multi-

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Corgi riding a rocket

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

A character in blue and white armor

Input MVDream SPAD MV-Adapter (SD2.1) MV-Adapter (SDXL)

- Figure 6: Qualitative comparison on camera-guided text-to-multiview generation. our MV-Adapter achieves higher visual fidelity and image-text consistency.

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

Input ImageDream Zero123++ CRM SV3D Ouroboros3D Era3D MV-Adapter (SD2.1)

Figure 7: Qualitative comparison on camera-guided image-to-multiview generation.

view consistent and high-quality textures. Additionally, testing on a single RTX 4090 GPU revealed that our method achieves faster generation speeds than the others.

- 5.3 ABLATION STUDY We conduct ablation studies to evaluate the efficiency and adaptability of our MV-Adapter, as well

- as the detailed design of the adapter network.

Efficiency. To assess the training efficiency of our adapter design, we conducted comparison with Era3D (Li et al., 2024), which requires full training rather than fine-tuning only adapters like us. As shown in Table 4, when working with SDXL (Podell et al., 2024), our MV-Adapter significantly reduces training costs, facilitating high-resolution multi-view generation based on larger backbones.

Adaptability. We compare MV-Adapter with the full-trained text-to-multiview generation method MVDream (Shi et al., 2023b) regarding compatibility with T2I derivatives. MVDream, which fine-tunes the whole T2I model, cannot be easily replaced with other T2Is; thus, we integrate LoRA (Hu et al., 2021) for our experiments. As shown in Fig. 9, MVDream struggles to generate images that align with the text and style, whereas our MV-Adapter produces high-quality results, demonstrating its superior adaptability.

(3d style) 1 girl, blue eyes, upper body, mask, eyes half closed

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

MVDream MV-Adapter

Figure 9: Qualitative ablation study on the adaptability of MV-Adapter.

Parallel attention architecture. To assess the effectiveness of our proposed parallel attention architecture, we conducted ablation studies on image-to-multi-view generation setting. We report the quantitative and qualitative results of using serial or parallel architecture in Table 5 and Fig. 11. The results show that, the serial setting, which cannot leverage the pre-trained image prior, tends

- to produce artifacts and inconsistent details with the image input. In contrast, our parallel setting produces high-quality and highly consistent results with the reference image.

###### Dreamshaper Animagine-xl SDXL-Lightning

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

(general) portrait of a girl, (medieval armor), metal…

(2d cartoon) girl, moricalliope (new year), hololive…

(few step) stylized, a female's upper body. Blonde hair…

###### Watercolor Style SDXL RealVisXL LCM SDXL

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

(realistic) A beautiful Warrior Elf girl with a hood and boots

(few step) A 3D model of a female in a fantasy outfit…

(watercolor) A female soldier in the stylized suit…

Figure 8: Results of geometry-guided text-to-multiview generation with community models.

Table 3: Quantitative comparison on 3D texture generation. FID and KID (×10−4) are evaluated on multi-view renderings. Our models achieves best texture quality with faster inference.

Method FID↓ KID↓ Time↓ TEXTure 56.44 61.16 90s Text2Tex 58.43 60.81 421s Paint3D 44.38 47.06 60s SyncMVD 36.13 42.28 50s FlashTex 50.48 56.36 186s

Ours (SD2.1 - Text) 38.19 42.83 18s Ours (SD2.1 - Image) 33.93 38.73 19s Ours (SDXL - Text) 32.75 35.18 32s Ours (SDXL - Image) 27.28 29.47 33s

##### Table 4: Comparison of training costs with fulltuning methods (batch size set to 1).

Method Trainable params ↓

Memory usage↓

Training speed ↑

Era3D (SD2.1) 993M 36G 2.2iter/s Ours (SD2.1) 127M 17G 3.1iter/s

Era3D (SDXL) 3.1B >80G Ours (SDXL) 490M 60G 1.05iter/s

##### Table 5: Quantitative ablation studies on attention architecture.

Method PSNR↑ SSIM↑ LPIPS↓ Serial (SDXL) 20.687 0.8681 0.1149 Parallel (SDXL) 22.131 0.8816 0.1002

- 5.4 APPLICATIONS

3D generation. We Follow the existing pipelines (Li et al., 2024) to achieve 3D generation. After generating multi-view images from text or image conditions using MV-Adapter, we use StableNormal (Ye et al., 2024) to generate corresponding normal maps. The multiview images and normal maps are then fed into NeuS (Wang et al., 2021) to reconstruct the 3D mesh. We conducted comparison on 3D reconstruction with Era3D (Li et al., 2024), which shares a similar pipeline with our method. Results in Table 6 show that our SD2.1-based MV-Adapter is comparable to Era3D, but our SDXL-based model shows significantly higher performance. These findings underline the scalability of MV-Adapter and its ability to leverage the strengths of state-ofthe-art T2I models, providing benefits to 3D generation. More results can be found in the Appendix.

Table 6: Quantitative comparison on 3D reconstruction.

Method Chamfer Distance↓ Volume IoU↑

Era3D 0.0329 0.5118 Ours (SD2.1) 0.0317 0.5173 Ours (SDXL) 0.0206 0.5682

Texture generation. We use backprojection and incidence-based weighted blending techniques (Bensadoun et al., 2024) to map the generated multi-view images onto the UV texture map. Despite optimizing view distribution to enhance coverage, some areas may remain uncovered due to occlusions or extreme angles. To address this, we perform view coverage analysis to identify

A gray raccoon 3D model with a long tail, pointy ears, black eyes, and a pink nose.

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

###### MV-AdapterBaselines

FlashTex

TEXTure Text2Tex Paint3D SyncMVD

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

SD2.1 - Text SD2.1 - Image SDXL - Text SDXL - Image

Figure 10: Qualitative comparison on texture generation. We compare our text- and imageconditioned models with baseline methods.

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

Input Serial attention architecture Parallel attention architecture

Figure 11: Qualitative ablation study on the attention architecture.

uncovered regions, render images from the current 3D texture for those views, and refine them using an efficient inpainting model (Suvorov et al., 2022). We show more visual results in the Appendix.

Arbitrary view generation. Starting from text or an initial image, we first generate eight anchor views that broadly cover the object. For new target views, we cluster the viewpoints based on their spatial orientations and select the 4 nearest anchor views to guide the generation of target views. We concatenate these four input views into a single image and input it into the pre-trained T2I UNet to extract features. Implementation details and visual results are provided in the Appendix and supplementary materials.

- 6 CONCLUSION

In this paper, we present MV-Adapter, the first adapter-based solution for multi-view image generation. This versatile, plug-and-play adapter enhances text-to-image diffusion models and their derivatives without compromising quality or altering the original feature space. We introduce innovative adapter framework that includes duplicated self-attention layers and a parallel attention architecture, allowing the adapter to efficiently model 3D geometric knowledge. Additionally, we introduced a unified condition encoder that integrates camera parameters and geometric information into spatial map representations, enhancing the model’s versatility and applicability in 3D object generation and texture generation. Extensive evaluations highlight the efficiency, adaptability, and versatility of MV-Adapter across different models and conditions. Overall, MV-Adapter offers an efficient and flexible solution for multi-view image generation, significantly broadening the capabilities of pre-trained T2I models and presenting exciting possibilities for a wide range of applications.

REFERENCES

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

Raphael Bensadoun, Yanir Kleiman, Idan Azuri, Omri Harosh, Andrea Vedaldi, Natalia Neverova, and Oran Gafni. Meta 3d texturegen: Fast and consistent texture generation for 3d objects. arXiv preprint arXiv:2407.02430, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Pu Cao, Feng Zhou, Qing Song, and Lu Yang. Controllable generation with text-to-image diffusion models: A survey. arXiv preprint arXiv:2403.04279, 2024.

Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner. Text2tex: Text-driven texture synthesis via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18558–18568, 2023.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13142–13153, 2023.

Kangle Deng, Timothy Omernick, Alexander Weiss, Deva Ramanan, Jun-Yan Zhu, Tinghui Zhou, and Maneesh Agrawala. Flashtex: Fast relightable mesh texturing with lightcontrolnet. arXiv preprint arXiv:2402.13251, 2024.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pp. 2553–2560. IEEE, 2022.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pp. 13213–13232. PMLR, 2023.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Zehuan Huang, Hao Wen, Junting Dong, Yaohui Wang, Yangguang Li, Xinyuan Chen, Yan-Pei Cao, Ding Liang, Yu Qiao, Bo Dai, et al. Epidiff: Enhancing multi-view synthesis via localized epipolar-constrained diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9784–9794, 2024.

Yoonwoo Jeong, Jinwoo Lee, Chiheon Kim, Minsu Cho, and Doyup Lee. Nvs-adapter: Plug-andplay novel view synthesis from a single image. In European Conference on Computer Vision, pp. 449–466. Springer, 2025.

Yash Kant, Aliaksandr Siarohin, Ziyi Wu, Michael Vasilkovsky, Guocheng Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey Tulyakov, and Igor Gilitschenski. Spad: Spatially aware multiview diffusers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10026–10038, 2024.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1931–1941, 2023.

Black Forest Labs. Flux. [Online], 2024. https://github.com/black-forest-labs/ flux.

Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023a.

Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wenhan Luo, Ping Tan, et al. Era3d: High-resolution multiview diffusion using efficient row-wise attention. arXiv preprint arXiv:2405.11616, 2024.

Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arXiv preprint arXiv:2310.02596, 2023b.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023a.

Yuxin Liu, Minshan Xie, Hanyuan Liu, and Tien-Tsin Wong. Text-guided texturing by synchronized multi-view diffusion. arXiv preprint arXiv:2311.12891, 2023b.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9970–9980, 2024.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. Advances in Neural Information Processing Systems, 36, 2024.

Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–12, 2024.

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14297–14306, 2023.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6038–6047, 2023.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(5):4296–4304, Mar. 2024. doi: 10.1609/aaai.v38i5.28226. URL https://ojs.aaai.org/index.php/ AAAI/article/view/28226.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 16784–16804, 2022.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. Texture: Textguided texturing of 3d shapes. In ACM SIGGRAPH 2023 conference proceedings, pp. 1–11, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention– MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500– 22510, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Luˇci´c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6229–6238, 2022.

Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600, 2023.

Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8543–8552, 2024.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023a.

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023b.

Yukai Shi, Jianan Wang, He Cao, Boshi Tang, Xianbiao Qi, Tianyu Yang, Yukun Huang, Shilong Liu, Lei Zhang, and Heung-Yeung Shum. Toss: High-quality text-guided novel view synthesis from a single image. arXiv preprint arXiv:2310.10644, 2023c.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Kunpeng Song, Yizhe Zhu, Bingchen Liu, Qing Yan, Ahmed Elgammal, and Xiao Yang. Moma: Multimodal llm adapter for fast personalized image generation. arXiv preprint arXiv:2404.05674, 2024.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2149–2159, 2022.

Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, and Yasutaka Furukawa. Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. arXiv, 2023.

Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka Furukawa, and Rakesh Ranjan. Mvdiffusion++: A dense high-resolution multi-view diffusion model for single or sparse-view 3d object reconstruction. arXiv preprint arXiv:2402.12712, 2024.

Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint, 2024.

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024.

Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024a.

Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021.

Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024b.

Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. arXiv preprint arXiv:2210.04628, 2022.

Hao Wen, Zehuan Huang, Yaohui Wang, Xinyuan Chen, Yu Qiao, and Lu Sheng. Ouroboros3d: Image-to-3d generation via 3d-aware recursive diffusion. arXiv preprint arXiv:2406.03184, 2024.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. ACM Transactions on Graphics (TOG), 2024.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9150–9161, 2023.

Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu, Yong Liu, and Gang Yu. Paint3d: Paint anything 3d with lighting-less texture diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4252–4262, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023.

Chuanxia Zheng and Andrea Vedaldi. Free3d: Consistent novel view synthesis without 3d representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9720–9731, 2024.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024. URL https://github.com/hpcaitech/Open-Sora.

A APPENDIX

### A.1 BACKGROUND Stable Diffusion (SD) and Stable Diffusion XL (SDXL). We adopt Stable Diffusion (Rombach

- et al., 2022) and Stable Diffusion XL (Podell et al., 2024) as our base T2I models, since they have a well-developed community with many powerful derivatives for evaluation. SD and SDXL perform the diffusion process within the latent space of a pre-trained autoencoder E(·) and D(·). In training, an encoded image z0 = E(x0) is perturbed to zt at step t by the forward diffusion. The denoising network ϵθ learns to reverse this process by predicting the added noise, encouraged by an MSE loss:

L = EE(x

0),ϵ∼N(0,I),c,t[∥ϵ − ϵθ(zt,c,t)∥22] (5)

where c denotes the conditioning texts. In SD, ϵθ is implemented as a UNet (Ronneberger et al., 2015) consisting of pairs of down/up sample blocks and a middle block. Each block contains pairs of spatial self-attention layers and cross-attention layers, which are serially connected using the residual structure. SDXL leverages a three times larger UNet backbone than SD for high-resolution image synthesis, and introduces a refinement denoiser to improve the visual fidelity.

A.2 IMPLEMENTATION DETAILS Dataset. We trained MV-Adapter on a filtered high-quality subset of the Objaverse dataset (Deitke

- et al., 2023), comprising approximately 70,000 samples, with captions from Cap3D (Luo et al.,
- 2024). To accommodate the efficient multi-view self-attention mechanism, we rendered orthographic views to train the the model to generate n = 6 views per sample. For the camera-guided generation, we rendered views of 3D models with the elevation angle set to 0◦ and azimuth angles

- at {0◦,45◦,90◦,180◦,270◦,315◦}. This distribution aligns with the setting used in Era3D (Li et al.,

- 2024), facilitating the application of a similar image-to-3D pipeline for 3D generation tasks. For the geometry-guided generation, we included four views at an elevation of 0◦ with azimuth angles

of {0◦,90◦,180◦,270◦}, added two additional views from the top and bottom. In addition to the target views, we rendered five random views within a certain frontal range of the models to serve as reference images during training.

Training. We utilized two versions of Stable Diffusion (Rombach et al., 2022) as the base models for training. Specifically, we trained a 512-resolution model based on Stable Diffusion 2.1 (SD2.1) and a 768-resolution model based on Stable Diffusion XL (SDXL). During training, we randomly dropped the text condition with a probability of 0.1, the image condition with a probability of 0.1, and both text and image conditions simultaneously with a probability of 0.1. Following prior work (Hoogeboom et al., 2023; Blattmann et al., 2023), we shifted the noise schedule to higher noise levels by adjusting the log signal-to-noise ratio (SNR) by log(n), where n = 6 is the number of the generated views. For the specific training configurations, we used a learning rate of 5 × 10−5 and trained the MV-Adapter on 8 NVIDIA A100 GPUs for 10 epochs.

Inference. In our experimental setup, we generated multi-view images using the DDPM sampler (Ho et al., 2020) with classifier-free guidance (Ho & Salimans, 2022), and set the number of inference steps to 50. For generation conditioned solely on text (i.e., setting the weight of the image condition λi to 0), we set the guidance scale to 7.0. For image-conditioned generation, we set the guidance scale of image condition α and text condition β to 3.0. Following TOSS (Shi et al., 2023c), the calculation can be expressed as:

ϵˆθ(zt1:n,ct,ci,cm,t) = ϵθ(zt1:n,∅,∅,cm,t)

+ α ϵθ(zt1:n,∅,ci,cm,t) − ϵθ(zt1:n,∅,∅,cm,t)

+ β ϵθ(zt1:n,ct,ci,cm,t) − ϵθ(zt1:n,∅,ci,cm,t) (6)

where ct, ci and cm represent texts, reference images and conditioning maps (i.e., camera or geometry conditions) respectively. Since we did not drop cm during the training process, we do not use the classifier-free guidance method for it.

Comparison with baselines. We conducted comprehensive comparisons with baseline methods across three settings: text-to-multiview generation, image-to-multiview generation, and texture generation. In these experiments, we evaluated both versions of MV-Adapter based on Stable Diffusion 2.1 (SD2.1) (Rombach et al., 2022) and Stable Diffusion XL (SDXL) (Podell et al., 2024), demonstrating the performance gains brought by MV-Adapter due to its efficient training and scalability.

For text-to-multiview generation, we selected MVDream (Shi et al., 2023b) and SPAD (Kant et al., 2024) as baseline methods. MVDream extends the original self-attention mechanism of T2I models to the multi-view domain. SPAD introduces epipolar constraints into the multi-view attention mechanism. We tested on 1,000 prompts selected from the Objaverse dataset (Deitke et al., 2023). We computed Fr´echet Inception Distance (FID), Inception Score (IS), and CLIP Score on all generated views to assess the quality of the generated images and their alignment with the textual prompts.

For image-to-multiview generation, we compared our method with ImageDream (Wang & Shi, 2023), Zero123++(Shi et al., 2023a), CRM(Wang et al., 2024b), SV3D (Voleti et al., 2024), Ouroboros3D (Wen et al., 2024), and Era3D (Li et al., 2024). ImageDream, Zero123++, CRM, and Era3D generally fall into the category of modifying the original network architecture of T2I models to extend them for multi-view generation. SV3D and Ouroboros3D fine-tune text-to-video (T2V) models to achieve multi-view generation. We selected 100 assets covering multiple object categories from the Google Scanned Objects (GSO) dataset (Downs et al., 2022) as our test set. For each asset, we rendered input images from front-facing views, with input views randomly distributed in azimuth angles between −45◦ and 45◦ and elevation angles between −10◦ and 30◦. We evaluated the generated multi-view images by computing Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) between the generated images and the ground truth, assessing both the consistency and quality of the outputs.

For 3D texture generation, we compared our text-based and image-based models with projectand-paint methods such as TEXTure (Richardson et al., 2023), Text2Tex (Chen et al., 2023), and Paint3D (Zeng et al., 2024), the synchronized multi-view texturing method SyncMVD (Liu et al., 2023b), and the optimization-based method FlashTex (Deng et al., 2024). We randomly selected 200 models along with their captions from the Objaverse (Deitke et al., 2023) dataset for testing.

Table 7: Community models and extensions for evaluation. Category Model Name Domain Model Type

Dreamshaper1 General T2I Base Model RealVisXL2 Realistic T2I Base Model Animagine-xl3 2D Cartoon T2I Base Model 3D Render Style XL4 3D Cartoon LoRA Pokemon Trainer Sprite PixelArt5 Pixel Art LoRA Chalk Sketch SDXL6 Chalk Sketch LoRA Chinese Ink LoRA7 Color Ink LoRA Zen Ink Wash Sumi-e8 Wash Ink LoRA Watercolor Style SDXL9 Watercolor LoRA Papercut SDXL10 Papercut LoRA Furry Enhancer11 Enhancer LoRA White Pitbull Dog SDXL12 Concept LoRA Spider spirit fourth sister13 Concept LoRA

Personalized T2I

SDXL-Lightning14 Few Step T2I Base Model LCM-SDXL15 Few Step T2I Base Model

Distilled T2I

ControlNet Openpose16 Spatial Control Plugin ControlNet Scribble17 Spatial Control Plugin ControlNet Tile18 Image Deblur Plugin T2I-Adapter Sketch19 Spatial Control Plugin IP-Adapter20 Image Prompt Plugin

Extension

Multiple views were rendered from the generated 3D textures, and we computed FID and Kernel Inception Distance (KID) of them to evaluate the quality of the generated textures. Additionally, we recorded the texture generation time to assess the inference efficiency of each method.

Community models and extensions for evaluation. To ensure a comprehensive benchmark, we selected a diverse set of representative T2I derivative models and extensions from the community for evaluation. As illustrated in Table 7, these models include personalized models that encompass various domains such as anime, stylistic paintings, and realistic photographic images, as well as efficient distilled models and plugins for controllable generation. They cover a wide range of subjects,

- 1https://civitai.com/models/112902?modelVersionId=126688
- 2https://civitai.com/models/139562?modelVersionId=789646
- 3https://huggingface.co/cagliostrolab/animagine-xl-3.1
- 4https://huggingface.co/goofyai/3d render style xl

- 5https://civitai.com/models/159333/pokemon-trainer-sprite-pixelart?modelVersionId=443092
- 6https://huggingface.co/JerryOrbachJr/Chalk-Sketch-SDXL
- 7https://huggingface.co/ming-yang/sdxl chinese ink lora

- 8https://civitai.com/models/647926/zen-ink-wash-sumi-e-sdxl-pony-flux?modelVersionId=724876
- 9https://civitai.com/models/484723/watercolor-style-sdxl
- 10https://huggingface.co/TheLastBen/Papercut SDXL

- 11https://civitai.com/models/310964/furry-enhancer?modelVersionId=558568
- 12https://civitai.com/models/700883/white-pitbull-dog-sdxl?modelVersionId=787948
- 13https://civitai.com/models/689010/pony-black-myth-wukong-spider-spirit-fourth-

sister?modelVersionId=771146

- 14https://huggingface.co/ByteDance/SDXL-Lightning
- 15https://huggingface.co/latent-consistency/lcm-sdxl
- 16https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0
- 17https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0
- 18https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0
- 19https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0
- 20https://huggingface.co/h94/IP-Adapter

including portraits, animals, landscapes, and more. This selection enables a thorough evaluation of our approach across different styles and content, demonstrating the adaptability and generality of MV-Adapter in working with various T2I derivatives and extensions.

- A.3 ADDITIONAL DISCUSSIONS

- A.3.1 MV-ADAPTER VS. MULTI-VIEW LORA

LoRA (Low-Rank Adaptation) (Hu et al., 2021) offers an alternative approach to achieving plugand-play multi-view generation. Specifically, using a condition encoder to inject camera representations, we extend the original self-attention mechanism to operate across all pixels of multiple views. During training, we introduce trainable LoRA layers into the network, allowing these layers to learn multi-view consistency or, optionally, generate images conditioned on a reference view. This approach requires the spatial self-attention mechanism to simultaneously capture spatial image knowledge, ensure multi-view consistency, and align generated images with reference views.

However, the multi-view LoRA approach has a notable limitation. The “incremental changes” it introduces to the network are not orthogonal or decoupled from those induced by T2I derivatives, such as personalized T2I models or LoRAs. Specifically, layers fine-tuned by multi-view LoRA and those tuned by personalized LoRA often overlap. Note that each weight matrix learned by both represents a linear transformation defined by its columns, so it is intuitive that the merger would retain the information available in these columns only when the columns that are being added are orthogonal to each other (Shah et al., 2023). Clearly, the multi-view LoRA and personalized models are not orthogonal, which often leads to challenges in retaining both sets of learned knowledge. This can result in a trade-off where either multi-view consistency or the fidelity of concepts (such as style or subject identity) is compromised.

In contrast, our proposed decoupled attention mechanism encourages different attention layers to specialize in their respective tasks without needing to fine-tune the original spatial self-attention layers. In this design, the layers we train do not overlap with those in the original T2I model, thereby better preserving the original feature space and enhancing compatibility with other models.

We conducted a series of experiments to test these approaches. We trained two versions of multiview LoRA, targeting different modules: (1) inserting LoRA layers only into the attention layers, and (2) inserting LoRA layers into multiple layers, including the convolutional layers, down-sampling, up-sampling layers, etc. For both settings, we set the LoRA rank to 64 and alpha to 32. As shown in Fig. 12 and Fig. 13, while the multi-view LoRA approach can generate multi-view consistent images when the base model is not changed, it often struggles to maintain multi-view consistency when switching to a different base model or when integrating a new LoRA. In contrast, as demonstrated in Fig. 14, our MV-Adapter, equipped with the decoupled attention mechanism, maintains consistent multi-view generation even when used with personalized models.

Compared to the LoRA mechanism, our decoupled attention-based approach proves more robust and adaptable for extending T2I models to multi-view generation, offering greater flexibility and compatibility with various pre-trained models.

- A.3.2 IMAGE RESTORATION CAPABILITIES

During the training of MV-Adapter, we probabilistically compress the resolution of reference images in the training data pairs to enhance the robustness of multi-view generation from images. We observed that the model trained with this approach is capable of generating high-resolution, detailed multi-view images even when the input is low-resolution, as depicted in Fig. 15. Through such training strategy, MV-Adapter has inherent image restoration capabilities and automatically enhances and refines input images during the generation process.

- A.3.3 APPLICABILITY OF MV-ADAPTER

Beyond the demonstrated applications in 3D object generation and 3D texture mapping, the MVAdapter’s strong adaptability and versatility open up a wide array of potential uses in image creation and personalization. For instance, creators can integrate MV-Adapter with their personalized T2I models—customized for specific identities or artistic styles—to generate multi-view images

(Base model: SDXL) Daenerys Targaryen from game of throne, full body, blender 3d, art station

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

(Base model: Animagine-xl) 1 girl, pink hair, pink shirts, smile, shy, masterpiece

|[Figure 247]|[Figure 248]|
|---|---|

|[Figure 249]|
|---|

[Figure 250]

[Figure 251]

[Figure 252]

(LoRA: Watercolor Style) painting, Burmese Cat, wearing ral-wtrclr, Comic book art

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

| |
|---|

- Figure 12: Results of multi-view LoRA (set target modules to attention layers). The azimuth angles of the images from left to right are 0◦,45◦,90◦,180◦,270◦,315◦, corresponding to the front, frontleft, left, back, right, and front-right of the object.

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

|[Figure 268]|
|---|

[Figure 269]

[Figure 270]

(Base model: SDXL) Daenerys Targaryen from game of throne, full body, blender 3d, art station

(LoRA: Watercolor Style) painting, Burmese Cat, wearing ral-wtrclr, Comic book art

|[Figure 271]|[Figure 272]|[Figure 273]|
|---|---|---|

[Figure 274]

[Figure 275]

[Figure 276]

(Base model: Animagine-xl) 1 girl, pink hair, pink shirts, smile, shy, masterpiece

- Figure 13: Results of multi-view LoRA (set target modules to attention layers, convolutional layers, etc.). The azimuth angles of the images from left to right are 0◦,45◦,90◦,180◦,270◦,315◦, corresponding to the front, front-left, left, back, right, and front-right of the object.

that capture consistent perspectives of their unique concepts. Additionally, MV-Adapter can facilitate tasks like multi-view portrait generation, where a subject’s face is rendered consistently across different angles, or stylized multi-view illustrations that maintain artistic coherence across diverse perspectives.

(Base model: SDXL) Daenerys Targaryen from game of throne, full body, blender 3d, art station

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

(Base model: Animagine-xl) 1 girl, pink hair, pink shirts, smile, shy, masterpiece

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

(LoRA: Watercolor Style) painting, Burmese Cat, wearing ral-wtrclr, Comic book art

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

- Figure 14: Results of MV-Adapter, which introduces decoupled attention mechanism rather than LoRA. The azimuth angles of the images from left to right are 0◦,45◦,90◦,180◦,270◦,315◦, corresponding to the front, front-left, left, back, right, and front-right of the object.

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Input Generated Multi-view Images (Auto Restored)

- Figure 15: Results on camera-guided image-to-multiview generation with low-resolution images as input.

- A.3.4 EXTENDING MV-ADAPTER FOR ARBITRARY VIEW SYNTHESIS

In the main text, we introduced a novel adapter architecture—comprising parallel attention layers and a unified condition encoder—to achieve multi-view generation. We implemented efficient rowwise and column-wise attention mechanisms tailored for two specific applications: 3D object gener-

ation and 3D texture mapping, generating six views accordingly. However, our adapter framework is not limited to these configurations and can be extended to perform arbitrary view synthesis. To explore this capability, we designed a corresponding approach and conducted experiments, training a new version of MV-Adapter to handle arbitrary viewpoints.

Following CAT3D (Gao et al., 2024), we perform multiple rounds of multi-view generation, with the number of views generated each time set to n = 8. Starting from text or an initial single image as input, we first generate eight anchor views that broadly cover the object. In practice, these anchor views are positioned at elevations of 0◦ and 30◦, with azimuth angles evenly distributed around the circle (e.g. every 45◦). For generating new target views, we cluster the viewpoints based on their spatial orientations, grouping them into clusters of 8. We then select the 4 nearest known views from the already generated anchor views to serve as conditions guiding the generation of each target view.

In terms of implementation, the overall framework of our MV-Adapter remains unchanged. We adjust its inputs and specific attention components to accommodate arbitrary view synthesis. First, we set the number of input images to either 1 or 4. When using four input views, we concatenate them into a long image and input this into the pre-trained T2I U-Net to extract features. This simple yet effective method allows the images from the four views to interact within the pre-trained U-Net without requiring additional camera embeddings to represent these views. Second, we utilize full self-attention in the multi-view attention component, expanding the attention scope to enable the generation of target views with more flexible distributions.

To train an MV-Adapter capable of generating arbitrary viewpoints, we rendered data from 40 different views, with elevations of −10◦,0◦,10◦,20◦,30◦, and azimuth angles evenly distributed around 360 degrees at each elevation layer. We trained the model for 16 epochs. During the first 8 epochs, the model was trained using a setting of one conditional view and eight target anchor views. In the subsequent 8 epochs, we trained with an equal mixture of one condition plus eight target views and four conditions plus eight target views.

As shown in Fig. 16, the visualization results demonstrate that MV-Adapter can generate consistent, high-quality multi-view images beyond the six views designed for specific applications. This extension further verifies the scalability and practicality of our adapter framework, showcasing its potential for arbitrary view synthesis in diverse applications. More results can be found in the supplementary materials.

Azimuth 0 360

[Figure 303]

Elevation030

Figure 16: Visualization results using MV-Adapter to generate arbitrary viewpoints.

- A.4 LIMITATIONS AND FUTURE WORKS

Limitation: Dependency on image backbone. Within our MV-Adapter, we only fine-tune the additional multi-view attention and image cross-attention layers, and do not disturb the original structure or feature space. Consequently, the overall performance of MV-Adapter is heavily dependent on the base T2I model. If the foundational model struggles to generate content that aligns with the provided prompt or produces images of low quality, MV-Adapter is unlikely to compensate for these deficiencies. On the other hand, employing superior image backbones can enhance the synthetic results. We present a comparison of outputs generated using SDXL (Podell et al., 2024) and SD2.1 (Rombach et al., 2022) models in Fig. 17, which confirms this observation, particularly in text-conditioned multi-view generation. We believe that MV-Adapter can be further developed by utilizing advanced T2I models (Team, 2024; Labs, 2024) based on the DiT architecture (Peebles & Xie, 2023), to achieve higher visual quality in the generated images.

[Figure 304]

[Figure 305]

[Figure 306]

A DSLR photo of a frog wearing a sweater Mecha vampire girl chibi

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

SD2.1

SDXL

Figure 17: Qualitative comparison of our MV-Adapter based on SD2.1 and SDXL.

Future works: 3D scene generation, dynamic multi-view video generation, inspiration for modeling new knowledge. This paper provides extensive analyses and enhancements for our novel multi-view adapter, MV-Adapter. While our model has significantly improved efficiency, adaptability, versatility, and performance compared to previous models, we identify several promising areas for future work:

- • 3D scene generation. Our method can be extended to scene-level multi-view generation, accommodating both camera- and geometry-guided approaches with text or image conditions.
- • Dynamic multi-view video generation. Exploring dynamic multi-view video generation using a similar approach as MV-Adapter within text-to-video generation models (Zheng et al., 2024; Yang et al., 2024) presents a valuable opportunity for further advancements.
- • Inspiration for modeling new knowledge. Our approach of decoupling the learning of geometric knowledge from the image prior can be applied to learning zoom in/out effects, consistent lighting, and other viewpoint-dependent properties. It also provides valuable insights for modeling physical or temporal knowledge based on image priors.

- A.5 MORE COMPARISON RESULTS A.5.1 IMAGE-TO-MULTI-VIEW GENERATION

To provide a more in-depth analysis of our quantitative results on image-to-multi-view generation, we conducted a user study comparing MV-Adapter (based on SD2.1 (Rombach et al., 2022)) with baseline methods (Wang & Shi, 2023; Shi et al., 2023a; Wang et al., 2024b; Voleti et al., 2024; Wen et al., 2024; Li et al., 2024). The study aimed to evaluate both multi-view consistency and image quality preferences. We selected 30 samples covering a diverse range of categories, such as toy cars, medicine bottles, stationery, dolls, and sculptures. A total of 50 participants were recruited to provide their preferences between the outputs of different methods.

[Figure 316]

Figure 18: Results of user study on image-to-multi-view generation.

Participants were presented with pairs of multi-view images generated by MV-Adapter and the baseline methods. For each pair, they were asked to choose the one they preferred in terms of multi-view consistency and image quality. The results of the user study are summarized in Fig. 18. The findings indicate that, in terms of multi-view consistency, MV-Adapter performs comparably to Era3D, with preference rates of 25.07% and 22.33%, respectively. However, regarding image quality, MVAdapter demonstrates a significant advantage, receiving a higher preference rate of 36.80% compared to the baseline methods. The improved image quality can be attributed to MV-Adapter’s ability to leverage the strengths of the underlying T2I models without full fine-tuning, preserving the original feature space and benefiting from the high-quality priors of the base models.

- A.6 MORE VISUAL RESULTS

In Fig. 19 and Fig. 20, we show more visual results of MV-Adapter on camera-guided text-tomultiview generation with community models and extensions, such as ControlNet (Zhang et al., 2023) and IP-Adapter (Ye et al., 2023). In Fig. 21, we show more visual results on camera-guided image-to-multiview generation. In Fig. 22, we show more visual results on text-to-3D generation. In Fig. 23, we show more visual results on image-to-3D generation. In Fig. 24, we show more visual results on geometry-guided text-to-texture generation. In Fig. 25, we show more visual results on geometry-guided image-to-texture generation. Note that we have removed the background of the generated images in the visual results.

###### Dreamshaper

###### Dreamshaper

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

(general) A beautiful girl, (medieval armor), upper body…

(general) A well-dressed dog named Puproy Doggerson III wearing…

###### RealVisXL

###### RealVisXL

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

(realistic) horror surrealism, george morland, mysterious jungle…

(realistic) trench coat, fedora, private detective, [murder mystery]…

###### Animagine-xl

###### 3D Render Style XL

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

(2d cartoon) 1 girl, izayoi sakuya, touhou, maid headdress, maid…

(3d cartoon) 1 boy, jacket, beard

###### Chalk Sketch SDXL

###### Pokemon Trainer Sprite PixelArt

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

(chalk sketch) a lone cowboy dressed in traditional attire…

(pixelart) joker, simple background

###### Chinese Ink LoRA

###### Zen Ink Wash Sumi-e

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

(Chinese Ink) a cute fox

(Ink) A simple teapot

###### Papercut SDXL

###### Watercolor Style SDXL

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

(papercut) an owl

(watercolor) strong Ash Ketchum dressed in ral-wtrclr

###### Furry Enhancer

###### Spider spirit fourth sister

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

(enhancer) anthro red panda, cyberpunk, fangs

(concept) 1 girl, Black Myth Wukong, black hair, single hair bun…

###### SDXL Lightning

###### LCM SDXL

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

(few step) Daenerys Targaryen from game of throne, full body…

(few step) an astronaut riding a horse

Figure 19: Additional results on camera-guided text-to-multiview generation with community models.

(IP-Adapter) cartoon style, light (IP-Adapter) A ginger tabby cat standing upright…

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

###### (ControlNet Openpose) Luffy

(ControlNet Tile) A stylized white fox with pink fur on its ears and…

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

(ControlNet Scribble) Aladdin, a character from Disney’s Aladdin…

(T2I-Adapter Sketch) A 3D model of Finn the Human from the …

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

##### Figure 20: Additional results on camera-guided text-to-multiview generation with extensions.

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

Input Generated Multi-view Images Input Generated Multi-view Images

##### Figure 21: Additional results on camera-guided image-to-multiview generation.

Daenerys Targaryen from game of throne, full body

A DSLR photo of a frog wearing a sweater

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

DnD dwarf with hammer

Army Jacket, 3D scan

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Military Mech, future, scifi

Nike air max, realistic, 8k texture, photorealistic

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

Ironman Scifi helmet with blue glowing elements

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

Astronaut bumping high A Squirtle

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

- Figure 22: Visual results on text-to-3D generation.

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

Input Generated 3D Objects Input Generated 3D Objects

- Figure 23: Visual results on image-to-3D generation.

A blue low poly formula one car with number 33 on the body and a white circle on the hood…

A stylized squirrel holding an acorn, chubby body, short legs, a large fluffy tail, and big round eyes.

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

The 3D model is of the Super Sonic, a yellow anthropomorphic hedgehog with spiky hair…

A cartoon-styled rocket ship ride with a predomi-nantly orange body, a green base, white details.

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

A US army motorcycle with a medical cross on the sidecar, a headlight, a brown seat, a large wheel…

A robot with blue, red and gray colors, and has a flame-like pattern on the body…

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

A purple anthropomorphic chameleon with a yellow belly and purple eyes, wearing black and purple…

Mater, a rusty and beat-up tow truck from the 2006 Disney/Pixar animated film "Cars", with a rusty…

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

A young girl with black hair, wearing an orange dress and yellow shirt, from the waist up.

Coco Bandicoot, from the Crash Bandicoot series, wearing her signature orange shirt and blue overalls…

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

##### Figure 24: Additional results on geometry-guided text-to-texture generation.

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

Input Rendered Multi-view Images from Generated Texture

##### Figure 25: Additional results on geometry-guided image-to-texture generation.

