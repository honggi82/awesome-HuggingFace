# arXiv:2312.00085v3[cs.CV]30Jul2024

## Creating High-quality 3D Content by Bridging the Gap Between Text-to-2D and Text-to-3D Generation

YIWEI MA∗, Xiamen University, China YIJUN FAN∗, Xiamen University, China JIAYI JI, Xiamen University, China HAOWEI WANG, Xiamen University, China HAIBING YIN, Hangzhou Dianzi University, China XIAOSHUAI SUN†, Xiamen University, China RONGRONG JI, Xiamen University, China

In recent times, automatic text-to-3D content creation has made significant progress, driven by the development of pretrained 2D diffusion models. Existing text-to-3D methods typically optimize the 3D representation to ensure that the rendered image aligns well with the given text, as evaluated by the pretrained 2D diffusion model. Nevertheless, a substantial domain gap exists between 2D images and 3D assets, primarily attributed to variations in camera-related attributes and the exclusive presence of foreground objects. Consequently, employing 2D diffusion models directly for optimizing 3D representations may lead to suboptimal outcomes. To address this issue, we present X-Dreamer, a novel approach for high-quality text-to-3D content creation that effectively bridges the gap between text-to-2D and text-to-3D synthesis. The key components of X-Dreamer are two innovative designs: Camera-Guided Low-Rank Adaptation (CG-LoRA) and Attention-Mask Alignment (AMA) Loss. CG-LoRA dynamically incorporates camera information into the pretrained diffusion models by employing camera-dependent generation for trainable parameters. This integration makes the 2D diffusion model camera-sensitive. AMA loss guides the attention map of the pretrained diffusion model using the binary mask of the 3D object, prioritizing the creation of the foreground object. This module ensures that the model focuses on generating accurate and detailed foreground objects. Extensive evaluations demonstrate the effectiveness of our proposed method compared to existing text-to-3D approaches. Our project webpage: https://anonymous-11111.github.io/. Our code is available at https://github.com/xmu-xiaoma666/XDreamer.

CCS Concepts: • Applied computing → Media arts; Computer-aided design; Computer-Fine arts. Additional Key Words and Phrases: Text-to-3D Generation, Vision and Language, Domain Gap ACM Reference Format:

Yiwei Ma, Yijun Fan, Jiayi Ji, Haowei Wang, Haibing Yin, Xiaoshuai Sun, and Rongrong Ji. 2024. Creating High-quality 3D Content by Bridging the Gap Between Text-to-2D and Text-to-3D Generation. J. ACM 37, 4, Article 111 (August 2024), 23 pages. https://doi.org/XXXXXXX.XXXXXXX

∗Both authors contributed equally to this research. †Corresponding author

Authors’ Contact Information: Yiwei Ma, Xiamen University, Xiamen, China, yiweima@stu.xmu.edu.cn; Yijun Fan, Xiamen University, Xiamen, China, fyj08092000@163.com; Jiayi Ji, Xiamen University, Xiamen, China, jjyxmu@gmail.com; Haowei Wang, Xiamen University, Xiamen, China, wanghaowei@stu.xmu.edu.cn; Haibing Yin, Hangzhou Dianzi University, Hangzhou, China, yhb@hdu.edu.cn; Xiaoshuai Sun, Xiamen University, Xiamen, China, xssun@xmu.edu.cn; Rongrong Ji, Xiamen University, Xiamen, China, rrji@xmu.edu.cn.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM 1557-735X/2024/8-ART111 https://doi.org/XXXXXXX.XXXXXXX

[Figure 1]

[Figure 2]

Consider Camera Parameters

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Concentrate on the 3D Asset

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Cameraindependent

[Figure 13]

[Figure 14]

Cameradependent

Focus on the Foreground Object

[Figure 15]

Focus on the Entire Image

Text-to-2D generation Text-to-3D generation

Fig. 1. The outputs of the text-to-2D generation model (left) and the text-to-3D generation model (right) under the same text prompt, i.e., “A statue of Leonardo DiCaprio’s head”.

#### 1 Introduction

The field of text-to-3D synthesis, which seeks to generate superior 3D content predicated on input textual descriptions, has shown significant potential to impact a diverse range of applications. These applications extend beyond traditional areas such as architecture, animation, and gaming, and encompass contemporary domains like virtual and augmented reality.

In recent years, extensive research [8, 20, 76] has demonstrated significant performance improvement in the text-to-2D generation task [3, 53, 55, 57, 73] by leveraging pretrained diffusion models [17, 62, 63] on a large-scale text-image dataset [58]. Building on these advancements, DreamFusion [49] introduces an effective approach that utilizes a pretrained 2D diffusion model [57] to autonomously generate 3D assets from text, eliminating the need for a dedicated 3D asset dataset. A key innovation introduced by DreamFusion is the Score Distillation Sampling (SDS) algorithm. This algorithm aims to optimize a single 3D representation, such as NeRF [45], to ensure that rendered images from any camera perspective maintain a high likelihood with the given text, as evaluated by the pretrained 2D diffusion model. Inspired by the groundbreaking SDS algorithm, several recent works [7, 33, 43, 68, 71] have emerged, envisioning the text-to-3D generation task through the application of pretrained 2D diffusion models.

While text-to-3D generation has made significant strides through the utilization of pretrained text-

- to-2D diffusion models [26,57,77], it is crucial to recognize and address the persistent and substantial domain gap that remains between text-to-2D and text-to-3D generation. This distinction is clearly illustrated in Fig. 1. To begin with, the text-to-2D model produces camera-independent generation results, focusing on generating high-quality images from specific angles while disregarding other angles. In contrast, 3D content creation is intricately tied to camera parameters such as position, shooting angle, and field of view. As a result, a text-to-3D model must generate high-quality results across all possible camera parameters. This fundamental difference emphasizes the necessity for innovative approaches that enable the pretrained diffusion model to consider camera parameters. Furthermore, a text-to-2D generation model [6, 12, 14, 37, 50, 56, 64] must simultaneously generate both foreground and background elements while maintaining the overall coherence of the image. Conversely, a text-to-3D generation model [36, 52, 74] only needs to concentrate on creating the foreground object. This distinction allows text-to-3D models to allocate more resources and attention to precisely represent and generate the foreground object. Consequently, the domain

1

Creating High-quality 3D Content by Bridging the Gap Between Text-to-2D and Text-to-3D Generation 111:3

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

A marble bust of an angel, 3D model, high resolution.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

A DSLR photo of David Beckham's head, highly detailed, 8K, HD.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

A DSLR photo of the Terracotta Army, 3D model, high resolution.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

A DSLR photo of a corn, highly detailed, 8K, HD.

Fig. 2. Text-to-3D generation results of the proposed X-Dreamer.

gap between text-to-2D and text-to-3D generation poses a significant performance obstacle when directly employing pretrained 2D diffusion models for 3D asset creation.

In this study, we present a pioneering framework, X-Dreamer, designed to address the domain gap between text-to-2D and text-to-3D generation, thereby facilitating the creation of high-quality text-

- to-3D content. Our framework incorporates two innovative designs that are specifically tailored to

address the aforementioned challenges. Firstly, existing approaches [7, 33, 39, 40, 43, 68] commonly employ 2D pretrained diffusion models [55, 57] for text-to-3D generation, which lack inherent linkage to camera parameters. To address this limitation and ensure that our text-to-3D model produces results that are directly influenced by camera parameters, we introduce Camera-Guided Low-Rank Adaptation (CG-LoRA) to fine-tune the pretrained 2D diffusion model. Notably, the parameters of CG-LoRA are dynamically generated based on the camera information during each iteration, establishing a robust relationship between the text-to-3D model and camera parameters. Furthermore, pretrained text-to-2D diffusion models allocate attention to both foreground and background generation, whereas the creation of 3D assets necessitates a stronger focus on accurately generating foreground objects. To address this requirement, we introduce Attention-Mask Alignment (AMA) Loss, which leverages the rendered binary mask of the 3D object to guide the attention map of the pretrained 2D stable diffusion model [55]. By incorporating this module, X-Dreamer prioritizes the generation of foreground objects, resulting in a significant enhancement of the overall quality of the generated 3D content.

We present a compelling demonstration of the effectiveness of X-Dreamer in synthesizing highquality 3D assets based on textual cues, as shown in Fig. 2. By incorporating CG-LoRA and AMA loss to address the domain gap between text-to-2D and text-to-3D generation, our proposed framework exhibits substantial advancements over prior methods in text-to-3D generation. In summary, our study contributes to the field in three key aspects:

- • We propose a novel method, X-Dreamer, for high-quality text-to-3D content creation, effectively bridging the domain gap between text-to-2D and text-to-3D generation.
- • To enhance the alignment between the generated results and the camera perspective, we propose CG-LoRA, which leverages camera information to dynamically generate CG-LoRA parameters for 2D diffusion models.
- • To prioritize the creation of foreground objects in the text-to-3D model, we introduce AMA loss, which utilizes binary masks of the foreground 3D object to guide the attention maps of the 2D diffusion model.

#### 2 Related Work

- 2.1 Text-to-3D Content Creation

In recent years, there has been a significant surge in interest surrounding the evolution of text-to-3D generation [2, 9, 13, 15, 30, 31, 44, 49, 59, 73, 75]. This growing field has been propelled, in part, by advancements in pretrained vision-and-language models, such as CLIP [51], as well as diffusion models like Stable Diffusion [55] and Imagen [57]. Contemporary text-to-3D models can generally be classified into two distinct categories: the CLIP-based text-to-3D approach and the diffusion-based text-to-3D approach. The CLIP-based text-to-3D approach [23, 27, 41, 44, 46, 72] employs CLIP encoders [51] to project textual descriptions and rendered images derived from the

###### 3D object into a modal-shared feature space. Subsequently, CLIP loss [11, 24, 69] is harnessed to align features from both modalities, optimizing the 3D representation to conform to the textual description. Various scholars have made significant contributions to this field. For instance, Michel et al. [44] are pioneers in proposing the use of CLIP loss to harmonize the text prompt with the rendered images of the 3D object, thereby enhancing text-to-3D generation. Ma et al. [41] introduce dynamic textual guidance during 3D object synthesis to improve convergence speed and generation performance. However, these approaches have inherent limitations, as they tend to generate 3D representations with a scarcity of geometry and appearance detail. To overcome this shortcoming, the diffusion-based text-to-3D approach [7, 21, 25, 33, 49, 65] leverages pretrained text-to-2D diffusion models [55, 57] to guide the optimization of 3D representations. Central to

Stage1 : Geometry

[Figure 32]

[Figure 33]

[Figure 34]

Render Mask

[Figure 35]

[Figure 36]

Learning

"An ice cream sundae, ... view."

Attention Map

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

DMTET Render Image

Difussion U-Net

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

CGLoRA

[Figure 48]

DMTET

DMTET

Update

Guided Mesh

Camera Parameters

Geometry Initialization Geometry Modeling

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Stage2 : Appearance

[Figure 54]

Render Mask

[Figure 55]

"An ice cream sundae, ... view."

Learning

[Figure 56]

[Figure 57]

Attention Map

Render Image

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Difussion U-Net

Mesh

[Figure 65]

Camera Parameters

[Figure 66]

[Figure 67]

[Figure 68]

CGLoRA

[Figure 69]

[Figure 70]

[Figure 71]

###### Text-to-3D Result

Update

PBR material

Fig. 3. Overview of the proposed X-Dreamer, which consists of geometry learning and appearance learning.

these models [5, 22, 29, 54, 65] is the application of SDS loss [49] to align the rendered images stemming from a variety of camera perspectives with the textual description. Specifically, given the target text prompt, Lin et al. [33] leverage a coarse-to-fine pipeline to generate high-resolution 3D content. Chen et al. [7] decouple geometry modeling and appearance modeling to generate realistic 3D assets. For specific purposes, some researchers [60, 71] integrate trainable LoRA [18] branches into pretrained diffusion models. For instance, Seo et al. [60] put forth 3DFuse, a model that harnesses the power of LoRA to comprehend object semantics. Wang et al. [71] introduce ProlificDreamer, where the role of LoRA is to evaluate the score of the variational distribution for 3D parameters. However, the LoRA parameter begins its journey from random initialization and maintains its independence from the camera and text. To address these limitations, we present two innovative modules: CG-LoRA and AMA loss. These modules are designed to enhance the model’s ability to consider important camera parameters and prioritize the generation of foreground objects throughout the text-to-3D creation process.

#### 2.2 Low-Rank Adaptation (LoRA)

Low-Rank Adaptation (LoRA) [19] is a technique used to reduce memory requirements when fine-tuning a large model [1, 10, 28, 32, 34, 38, 70]. It involves injecting only a small set of trainable parameters into the pretrained model, while keeping the original parameters fixed. During the optimization process, gradients are passed through the fixed pretrained model weights to the LoRA adapter, which is then updated to optimize the loss function. LoRA has been applied in various fields, including natural language processing [4, 18], image synthesis [77] and 3D generation [60, 71]. To achieve low-rank adaptation, a linear projection with a pretrained weight matrix W0 ∈ R𝑑𝑖𝑛×𝑑𝑜𝑢𝑡 is augmented with an additional low-rank factorized projection. This augmentation is represented as W0 + ΔW = W0 + AB, where A ∈ R𝑑𝑖𝑛×𝑟, B ∈ R𝑟×𝑑𝑜𝑢𝑡, and 𝑟 ≪ min(𝑑𝑖𝑛,𝑑𝑜𝑢𝑡). During training, W0 remains fixed, while A and B are trainable. The modified forward pass, given the original forward

pass Y = XW0, can be formulated as follows:

##### Y = XW0 + XAB. (1)

In this paper, we introduce CG-LoRA, which involves the dynamic generation of trainable parameters for A based on camera information. This technique allows for integrating perspective information, including camera parameters and direction-aware descriptions, into the pretrained

- text-to-2D diffusion model. As a result, our method significantly enhances text-to-3D generation capabilities.

- 3 Approach

- 3.1 Architecture

In this section, we present a comprehensive introduction to the proposed X-Dreamer, which consists of two main stages: geometry learning and appearance learning. For geometry learning, we employ DMTet [61] as the 3D representation. DMTet is an MLP parameterized with Φ𝑑𝑚𝑡 and is initialized with a 3D ellipsoid using the mean squared error (MSE) loss L𝑀𝑆𝐸. Subsequently, we optimize DMTet and CG-LoRA using the SDS loss [49] L𝑆𝐷𝑆 and the proposed AMA loss L𝐴𝑀𝐴 to ensure the alignment between the 3D representation and the input text prompt. For appearance learning, we leverage bidirectional reflectance distribution function (BRDF) modeling [66] following the previous approach [7]. Specifically, we utilize an MLP with trainable parameters Φ𝑚𝑎𝑡 to predict surface materials. Similar to the geometry learning stage, we optimize Φ𝑚𝑎𝑡 and CG-LoRA using the SDS loss L𝑆𝐷𝑆 and the AMA loss L𝐴𝑀𝐴 to achieve alignment between the 3D representation and the text prompt. Fig. 3 provides a detailed depiction of our proposed X-Dreamer.

- 3.1.1 Geometry Learning. For geometry learning, an MLP network Φ𝑑𝑚𝑡 is utilized to parameterize DMTet as a 3D representation. To enhance the stability of geometry modeling, we employ a 3D ellipsoid as the initial configuration for DMTet Φ𝑑𝑚𝑡. For each vertex 𝑣𝑖 ∈ 𝑉𝑇 belonging to the tetrahedral grid 𝑇, we train Φ𝑑𝑚𝑡 to predict two important values: the SDF value 𝑠(𝑣𝑖) and the deformation offset 𝛿(𝑣𝑖). To initialize Φ𝑑𝑚𝑡 with the 3D ellipsoid, we sample a set of 𝑁 points {𝑝𝑖 ∈ R3}|𝑖𝑁=1 approximately distributed on the surface of an ellipsoid and compute the

corresponding SDF values {𝑆𝐷𝐹(𝑝𝑖)}|𝑖𝑁=1. Subsequently, we optimize Φ𝑑𝑚𝑡 using MSE loss. This optimization process ensures that Φ𝑑𝑚𝑡 effectively initializes DMTet to resemble the 3D ellipsoid. The formulation of the MSE loss is given by:

∑︁𝑁

1 𝑁

(𝑠(𝑝𝑖;Φ𝑑𝑚𝑡) − 𝑆𝐷𝐹(𝑝𝑖))2 . (2)

L𝑀𝑆𝐸 =

𝑖=1

After initializing the geometry, our objective is to align the geometry of DMTet with the input text prompt. Specifically, we generate the normal map 𝒏 and the object mask 𝒎 from the initialized DMTet Φ𝑑𝑚𝑡 by employing a differentiable rendering technique [66], given a randomly sampled camera pose 𝒄. Subsequently, we input the normal map 𝒏 into the frozen stable diffusion (SD) with a trainable CG-LoRA and update Φ𝑑𝑚𝑡 using the SDS loss, which is defined as follows:

𝜕𝒏 𝜕Φ𝑑𝑚𝑡

∇Φ𝑑𝑚𝑡 LSDS = E𝑡,𝝐 𝑤(𝑡) (𝝐ˆΘ(𝒏𝑡;𝒚,𝑡) − 𝝐)

, (3)

where Θ represents the parameter of SD, 𝝐ˆΘ(𝒏𝑡;𝒚,𝑡) denotes the predicted noise of SD given the noise level 𝑡 and text embedding 𝒚. Additionally, 𝒏𝑡 = 𝛼𝑡𝒏 + 𝜎𝑡𝝐, where 𝝐 ∼ N(0, 𝑰) represents noise sampled from a normal distribution. The implementation of 𝑤(𝑡), 𝛼𝑡, and 𝜎𝑡 is based on the DreamFusion [49].

[Figure 72]

+ Element-wise Addition C Concatenation

[Figure 73]

[Figure 74]

An ice cream sundae, front/side/back view.

[Figure 75]

[Figure 76]

[Figure 77]

###### Direction-aware Text

### +

[Figure 78]

[Figure 79]

[Figure 80]

C

[Figure 81]

Camera Parameters

[Figure 82]

[Figure 83]

[Figure 84]

Fig. 4. Illustration of Camera-Guided Low-Rank Adaptation (CG-LoRA).

[Figure 85]

Furthermore, to focus SD on generating foreground objects, we introduce an additional AMA loss to align the object mask 𝒎 with the attention map of SD, given by:

∑︁𝐿

1 𝐿

|𝒂𝑖 − 𝜂(𝒎)|, (4)

L𝐴𝑀𝐴 =

𝑖=1

where 𝐿 denotes the number of attention layers, and 𝒂𝑖 is the attention map of 𝑖-th attention layer. The function 𝜂(·) is employed to resize the rendered mask, ensuring its dimensions align with those of the attention maps.

- 3.1.2 Appearance Learning. After obtaining the geometry of the 3D object, our objective is to compute its appearance using the Physically-Based Rendering (PBR) material model [42]. The material model comprises the diffuse term 𝒌𝑑 ∈ R3, the roughness and metallic term 𝒌𝑟𝑚 ∈ R2, and the normal variation term 𝒌𝑛 ∈ R3. Firstly, the Diffuse Term 𝒌𝑑 ∈ R3 accurately represents how the material responds to diffuse lighting. This term is captured by a three-dimensional vector that

denotes the material’s diffuse color. Secondly, the Roughness and Metallic Term 𝒌𝑟𝑚 ∈ R2 effectively captures the material’s roughness and metallic properties. It is represented by a two-dimensional vector that conveys the values of roughness and metallicness. Roughness quantifies the surface smoothness, while metallicness indicates the presence of metallic properties in the material. These terms collectively contribute to the overall visual realism of the rendered scene. Additionally, the Normal Variation Term 𝒌𝑛 ∈ R3 plays a crucial role in characterizing variations in surface normals. This term, typically used in conjunction with a normal map, enables the simulation of fine details and textures on the material’s surface. During the rendering process, these three PBR material terms interact to accurately simulate the optical properties exhibited by real-world materials. This integration facilitates rendering engines in achieving high-quality rendering results and generating realistic virtual scenes.

For any point 𝑝 ∈ R3 on the surface of the geometry, we utilize an MLP parameterized by Φ𝑚𝑎𝑡 to obtain the three material terms, which can be expressed as follows:

(𝒌𝑑, 𝒌𝑛, 𝒌𝑟𝑚) = MLP (P(𝑝);Φ𝑚𝑎𝑡) , (5)

where P(·) represents the positional encoding using a hash-grid technique [47]. Subsequently, each pixel of the rendered image can be computed as follows:

𝑉 (𝑝,𝜔) = ∫

𝐿𝑖(𝑝,𝜔𝑖)𝑓 (𝑝,𝜔𝑖,𝜔)(𝜔𝑖 · 𝑛𝑝)d𝜔𝑖, (6)

Ω

where 𝑉 (𝑝,𝜔) denotes the rendered pixel value from the direction 𝜔 for the surface point 𝑝. Ω denotes a hemisphere defined by the set of incident directions 𝜔𝑖 satisfying the condition 𝜔𝑖 ·𝑛𝑝 ≥ 0, where 𝜔𝑖 denotes the incident direction, and 𝑛𝑝 represents the surface normal at point 𝑝. 𝐿𝑖(·) corresponds to the incident light from an off-the-shelf environment map, and 𝑓 (·) is the Bidirectional Reflectance Distribution Function (BRDF) related to the material properties (i.e., 𝒌𝑑, 𝒌𝑛, 𝒌𝑟𝑚). By aggregating all rendered pixel colors, we obtain a rendered image 𝒙 = {𝑉 (𝑝,𝜔)}. Similar to the geometry modeling stage, we feed the rendered image 𝒙 into SD. The optimization objective remains the same as Equ. 3 and Equ. 4, where the rendered normal map 𝒏 and the parameters of DMTet Φ𝑑𝑚𝑡 are replaced with the rendered image 𝒙 and the parameters of the material encoder Φ𝑚𝑎𝑡, respectively.

#### 3.2 Camera-Guided Low-Rank Adaptation

The domain gap between text-to-2D and text-to-3D generation presents a significant challenge, as discussed in Sec. 1, which leads to unsatisfied generated results. To address these issues, we propose Camera-Guided Low-Rank Adaptation (CG-LoRA) as a solution to bridge the domain gap. As depicted in Fig. 4, we leverage camera parameters and direction-aware text to guide the generation of parameters in CG-LoRA, enabling X-Dreamer to effectively incorporate camera perspective and direction information.

Specifically, given a text prompt 𝑇 and camera parameters 𝐶 = {𝑥,𝑦,𝑧,𝜙𝑦𝑎𝑤,𝜙𝑝𝑖𝑡,𝜃𝑓 𝑜𝑣} 1, we initially project these inputs into a feature space using the pretrained textual CLIP encoder E𝑡𝑥𝑡 (·) and a trainable MLP E𝑝𝑜𝑠(·):

𝒕 = E𝑡𝑥𝑡 (𝑇), (7) 𝒄 = E𝑝𝑜𝑠(𝐶), (8)

where 𝒕 ∈ R𝑑𝑡𝑥𝑡 and 𝒄 ∈ R𝑑𝑐𝑎𝑚 are textual features and camera features. Subsequently, we employ two low-rank matrices to project 𝒕 and 𝒄 into trainable dimensionality-reduction matrices within CG-LoRA:

𝑨𝑡𝑥𝑡 = Reshape(𝒕𝑾𝑡𝑥𝑡), (9) 𝑨𝑐𝑎𝑚 = Reshape(𝒄𝑾𝑐𝑎𝑚), (10)

where 𝑨𝑡𝑥𝑡 ∈ R𝑑×𝑟2 and 𝑨𝑐𝑎𝑚 ∈ R𝑑×𝑟2 are two dimensionality-reduction matrices of CG-LoRA. The function Reshape(·) is used to transform the shape of a tensor from R𝑑∗𝑟2 to R𝑑×𝑟2 . 2 𝑾𝑡𝑥𝑡 ∈ R𝑑𝑡𝑥𝑡×(𝑑∗𝑟2) and 𝑾𝑐𝑎𝑚 ∈ R𝑑𝑐𝑎𝑚×(𝑑∗𝑟2) are two low-rank matrices. Thus, we decompose them into the product of two matrices to reduce the trainable parameters in our implementation, i.e., 𝑾𝑡𝑥𝑡 = 𝑼𝑡𝑥𝑡𝑽𝑡𝑥𝑡 and 𝑾𝑐𝑎𝑚 = 𝑼𝑐𝑎𝑚𝑽𝑐𝑎𝑚, where 𝑼𝑡𝑥𝑡 ∈ R𝑑𝑡𝑥𝑡×𝑟′, 𝑽𝑡𝑥𝑡 ∈ R𝑟′×(𝑑∗𝑟2 ), 𝑼𝑐𝑎𝑚 ∈ R𝑑𝑐𝑎𝑚×𝑟′, 𝑽𝑐𝑎𝑚 ∈ R𝑟′×(𝑑∗𝑟2 ), 𝑟′ is a small number (i.e., 4). In accordance with LoRA [18], we initialize the dimensionality-expansion matrix 𝑩 ∈ R𝑟×𝑑 with zero values to ensure that the model begins training from the pretrained parameters of SD. Thus, the feed-forward process of CG-LoRA is

1The variables 𝑥, 𝑦,𝑧,𝜙𝑦𝑎𝑤,𝜙𝑝𝑖𝑡,𝜃𝑓𝑜𝑣 represent the x, y, z coordinates, yaw angle, pitch angle of the camera, and field of view, respectively. The roll angle 𝜙𝑟𝑜𝑙𝑙 is intentionally set to 0 to ensure the stability of the object in the rendered image. 2R𝑑∗𝑟2 denotes a one-dimensional vector. R𝑑×𝑟2 represents a two-dimensional matrix.

formulated as follows:

𝒚 = 𝒙𝑾 + [𝒙𝑨𝑡𝑥𝑡;𝒙𝑨𝑐𝑎𝑚]𝑩, (11)

where 𝑾 ∈ R𝑑×𝑑 represents the frozen parameters of the pretrained SD model, and [·; ·] is the concatenation operation alone the channel dimension. In our implementation, we integrate CGLoRA into the linear embedding layers of the attention modules in SD to effectively capture direction and camera information.

- 3.3 Attention-Mask Alignment Loss

Although SD is pretrained to generate 2D images that encompass both foreground and background elements, the task of text-to-3D generation demands a stronger focus on generating foreground objects. To address this specific requirement, we introduce Attention-Mask Alignment (AMA) Loss, which aims to align the attention map of SD with the rendered mask image of the 3D object. Specifically, for each attention layer in the pretrained SD, we compute the attention map between the query image feature 𝑸 ∈ R𝐻×ℎ×𝑤×𝐻𝑑 and the key CLS token feature 𝑲 ∈ R𝐻×𝐻𝑑 . The calculation is formulated as follows:

𝒂¯ = Softmax(

𝑸𝑲⊤

√

𝑑

), (12)

where 𝐻 denotes the number of attention heads in SD. ℎ, 𝑤, and 𝑑 represent the height, width, and channel dimensions of the image features, respectively. 𝒂¯ ∈ R𝐻×ℎ×𝑤 represents the attention map. Subsequently, we proceed to compute the overall attention map 𝒂ˆ ∈ Rℎ×𝑤 by averaging the attention values of 𝒂¯ across all attention heads. Since the attention map values are normalized using the softmax function, the activation values in the attention map may become very small when the image feature resolution is high. However, considering that each element in the rendered mask has a binary value of either 0 or 1, directly aligning the attention map with the rendered mask is not optimal. To address this, we propose a normalization technique that maps the values in the attention map from 0 to 1. This normalization process is formulated as follows:

𝒂 =

𝒂ˆ − min(𝒂ˆ) max(𝒂ˆ) − min(𝒂ˆ) + 𝜈

, (13)

where 𝜈 represents a small constant value (e.g., 1𝑒-6) that prevents division by zero in the denominator. Finally, we align the attention maps of all attention layers with the rendered mask of the 3D object using the AMA loss. The formulation of this alignment is presented in Equ. 4.

- 4 Experiments

- 4.1 Implementation Details.

We conduct the experiments using four Nvidia RTX 3090 GPUs and the PyTorch library [48]. To calculate the SDS loss, we utilize the Stable Diffusion implemented by HuggingFace Diffusers [67]. For the DMTet Φ𝑑𝑚𝑡 and material encoder Φ𝑚𝑎𝑡, we implement them as a two-layer MLP and a single-layer MLP, respectively, with a hidden dimension of 32. The values of 𝑑𝑐𝑎𝑚, 𝑑𝑡𝑥𝑡, 𝑟, 𝑟′, the batch size, the SDS loss weight, the AMA loss weight, and the aspect ratio of the perspective projection plane are set to 1024, 1024, 4, 4, 4, 1, 0.1, and 1 respectively. AMA loss is used after half of the training iterations, where CG-LoRA has a certain degree of perspective alignment ability. We optimize X-Dreamer for 2000 iterations for geometry learning and 1000 iterations for appearance learning. For each iteration,𝜙𝑝𝑖𝑡,𝜙𝑦𝑎𝑤, and𝜃𝑓 𝑜𝑣 are randomly sampled from (−15◦, 45◦), (−180◦, 180◦), and (25◦, 45◦), respectively.

[Figure 86]

[Figure 87]

111:10 Yiwei Ma et al.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 97]

###### A beautifully carved wooden queen chess piece.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 102]

[Figure 103]

A sliced loaf of fresh bread.

A cabbage, highly detailed.

A strawberry.

A hamburger.

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

A plate piled high with chocolate chip cookies.

A chocolate cupcake, highly detailed.

A DSLR photo of a brown cowboy hat.

A rocket.

Fig. 5. Text-to-3D generation results from an ellipsoid.

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

A beautifully carved wooden queen chess piece.

Barack Obama’s head.

A corgi, highly detailed.

Fig. 6. Text-to-3D generation results from coarse-grained guided meshes.

#### 4.2 Results of X-Dreamer

- 4.2.1 Text-to-3D generation from an ellipsoid. We present representative results of X-Dreamer for

text-to-3D generation, utilizing an ellipsoid as the initial geometry, as shown in Fig. 5. The results demonstrate the ability of X-Dreamer to generate high-quality and photo-realistic outputs that accurately correspond to the input text prompts.

- 4.2.2 Text-to-3D generation from coarse-grained meshes. While there is a wide availability of coarse-grained meshes for download from the internet, directly utilizing these meshes for 3D content creation often results in poor performance due to the lack of geometric details. However, when compared to a 3D ellipsoid, these meshes may provide better 3D shape prior information for X-Dreamer. Hence, instead of using ellipsoids, we can initialize DMTet with coarse-grained guided meshes as well. As shown in Fig. 6, X-Dreamer can generate 3D assets with precise geometric details based on the given text, even when the provided coarse-grained mesh lacks details. For instance, in the last column of Fig. 6, X-Dreamer accurately transforms the geometry from a cow

Table 1. Quantitative comparison of SOTA Methods: The top-performing and second-best results are highlighted in bolded and underlined, respectively.

|Method<br><br>|User Study Geo. Qua. App. Qua.<br><br>|CLIP Score ViT-B/32 ViT-B/16 ViT-L/14|OpenCLIP Score ViT-B/32 ViT-B/16 ViT-L/14|
|---|---|---|---|
| | | | |
|Dreamfusion [49] Magic3d [33] Fantasia3d [7] ProlificDreamer [71] X-Dreamer<br><br>|1.2 0.6<br><br>2.1 1.1 5.4 6.3<br><br><br>12.9 16.2 78.4 75.8<br><br>|30.5 31.1 26.3 28.4 28.6 24.6 30.4 30.1 24.8 30.8 31.5 26.4 32.0 32.2 27.2<br><br>|31.9 27.9 30.0<br><br>29.1 25.9 28.3<br><br>30.3 27.2 29.7<br><br><br>32.7 28.8 31.1<br><br><br>34.1 29.6 32.2|

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

[Figure 147]

Prompt: An onion, highly detailed.

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

Prompt: A statue of Leonardo DiCaprio's head.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Prompt: A pumpkin, highly detailed, 8K, HD.

DreamFusion Magic3D Fantasia3D ProlificDreamer X-Dreamer

Fig. 7. Comparison with SOTA methods. Our method yields results that exhibit enhanced fidelity and more details.

to a corgi based on the text prompt “A corgi, highly detailed." Therefore, X-Dreamer is also an exceptionally powerful tool for editing coarse-grained mesh geometry using textual inputs.

- 4.2.3 Quantitative Comparison. We conducted a meticulously organized user study involving 50 individuals to evaluate the performance of four state-of-the-art (SOTA) methods by assessing 50 generated results, i.e., DreamFusion [49], Magic3D [33], Fantasia3D [7], and ProlificDreamer [71]. Participants were asked to compare and select the best result based on two criteria: Geometry Quality (Geo. Qua.) and Appearance Quality (App. Qua.). The results, as presented in Tab. 1, indicate that our approach garnered a preference from 75.0% of the participants. This overwhelming preference underscores our method’s exceptional quality and superiority, establishing it as a pioneering solution in the field. The primary objective of these user studies was to evaluate the geometric and visual quality of the generated results, without specifically assessing their alignment with the text. To provide a more comprehensive evaluation, we further calculated the CLIP score and OpenCLIP score for both the generated results and the corresponding text prompts. As depicted in Tab. 1,

[Figure 171]

###### DreamFusion Magic3D Fantasia3D ProlificDreamer X-Dreamer

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Prompt: A plate piled high with chocolate chip cookies.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Prompt: A 3D model of an adorable cottage with a thatched roof.

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Prompt: A marble bust of a mouse.

Fig. 8. More comparison with State-of-the-Art (SOTA) methods.

our method outperformed other approaches in terms of both the CLIP score and OpenCLIP score. These findings demonstrate that the results generated by our method exhibit superior alignment with the text, further strengthening the efficacy of our approach.

- 4.2.4 Qualitative Comparison. To assess the effectiveness of X-Dreamer, we compare it with four SOTA methods. Since the codes of some methods [33, 49, 71] have not been publicly available, we present the results obtained by implementing these methods in threestudio [16]. The results are depicted in Fig. 7. In Fig. 8, we present a comprehensive comparison of our methods with four baselines, using the images provided in their original papers. When compared to the SDS-based methods [7, 33, 49], X-Dreamer outperforms them in generating superior-quality and realistic 3D assets. In addition, when compared to the VSD-based method [71], X-Dreamer produces 3D content with comparable or even better visual effects, while requiring significantly less optimization time. Specifically, the geometry and appearance learning process of X-Dreamer requires only approximately 27 minutes, whereas ProlificDreamer exceeds 8 hours.
- 4.2.5 Camera Control Comparison. During training, CG-LoRA aims to control the camera parameters. To investigate the effectiveness of this control, we visualize the image generated by Stable Diffusion (SD) with a pretrained CG-LoRA. As shown in Fig. 9, given a specific camera parameter, SD with CG-LoRA can generate images from that perspective. To make a comparative analysis, we also compare our results with Zero-1-to-3 [35], a method capable of generating images with desired viewing angles based on relative camera angles and reference images. Using the front view produced by our approach as a reference image, we leverage Zero-1-to-3 to generate images with varying camera parameters. Our observations indicate that our method can generate images that exhibit a general alignment with the angles produced by Zero-1-to-3.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

###### CG-LoRAZero-1-to-3

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

- Fig. 9. Camera Control Comparison with Zero-1-to-3 [35].

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

w/o CG-LoRA w/o AMA Loss X-Dreamer

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

Prompt: A 3D rendering of Batman, highly detailed.

Prompt: A DSLR photo of Lord Voldemort's head, highly detailed.

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

Prompt: A rocket.

- Fig. 10. Ablation studies of the proposed X-Dreamer.

#### 4.3 Ablation Study

- 4.3.1 Ablation on the proposed modules. To gain insights into the abilities of CG-LoRA and AMA loss, we perform ablation studies wherein each module is incorporated individually to assess its impact. As depicted in Fig. 10, the ablation results demonstrate a notable decline in the geometry and appearance quality of the generated 3D objects when CG-LoRA is excluded from X-Dreamer. For instance, as shown in the second row of Fig. 10, the generated Batman lacks an ear on the top of its head in the absence of CG-LoRA. This observation highlights the crucial role of CG-LoRA in injecting camera-relevant information into the model, thereby enhancing the 3D consistency. Furthermore, the omission of AMA loss from X-Dreamer also has a deleterious effect on the

Attention Map Rendered Mask Rendered Image

Attention Map Rendered Mask Rendered Image

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

w/AMALossw/oAMALoss

AppearanceLearning

GeometryLearning

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Prompt: A DSLR photo of a cat, highly detailed.

Fig. 11. Visualization of Attention Map, Rendered Mask, and Rendered Image with and without AMA Loss. For clarity, we only visualize the attention map of the first attention layer in SD.

geometry and appearance fidelity of the generated 3D assets. Specifically, as illustrated in the first row of Fig. 10, X-Dreamer successfully generates a photorealistic texture for the rocket, whereas the texture quality noticeably deteriorates in the absence of AMA loss. This disparity can be attributed to AMA loss, which directs the focus of the model towards foreground object generation, ensuring the realistic representation of both geometry and appearance of foreground objects. These ablation studies provide valuable insights into the individual contributions of CG-LoRA and AMA loss in enhancing the geometry, appearance, and overall quality of the generated 3D objects.

- 4.3.2 Attention map comparisons w/ and w/o AMA loss. AMA loss is introduced with the aim of guiding attention during the denoising process towards the foreground object. This objective is achieved by aligning the attention map of SD with the rendered mask of the 3D object. To evaluate the effectiveness of AMA loss in accomplishing this goal, we visualize the attention maps of SD with and without AMA loss at both the geometry learning and appearance learning stages. As depicted in Fig. 11, it can be observed that incorporating AMA loss not only results in improved geometry and appearance of the generated 3D asset, but also concentrates the attention of SD specifically on the foreground object area. The visualizations confirm the efficacy of AMA loss in directing the attention of SD, resulting in improved quality and foreground object focus during geometry and appearance learning stages.
- 4.3.3 Ablation on the parameter generation of CG-LoRA. In our implementation, the parameters of CG-LoRA are dynamically generated based on two key factors related to camera information: camera parameters and direction-aware text. To thoroughly investigate the impact of these camera information terms on X-Dreamer, we conducted ablation experiments, dynamically generating CG-LoRA parameters based solely on one of these terms. The results, as illustrated in Fig. 12, clearly demonstrate that when using only one type of camera information term, the geometry and appearance quality of the generated results are significantly diminished compared to those produced by X-Dreamer. For example, as shown in the third row of Fig. 12, when utilizing the CG-LoRA solely related to direction-aware text, the geometry quality of the generated cookies appears poor. Similarly, as depicted in the last row of Fig. 12, employing the CG-LoRA solely based on camera parameters results in diminished geometry and appearance quality for the pumpkin. These findings highlight the importance of both camera parameters and direction-aware text in generating high-quality 3D assets. The successful generation of parameters for CG-LoRA relies

W/ CG-LoRA (camera parameters)

W/ CG-LoRA (direction-aware text)

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

W/ CG-LoRA

A sliced loaf of fresh bread.

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

A cabbage, highly detailed.

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

A plate piled high with chocolate chip cookies.

A statue of Leonardo DiCaprio's head.

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

A blue tulip.

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

A pumpkin, highly detailed,8K,HD.

Fig. 12. Ablation Study of CG-LoRA.

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Prompt: A DSLR photo of Lord Voldemort's head, highly detailed.

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Prompt: A pumpkin, highly detailed,8K,HD.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

Prompt: Garlic with white skin, highly detailed,8K,HD.

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Prompt: A Vintage Style LED Lantern.

Fig. 13. Attention maps of different layers of SD for the geometry learning stage.

on the synergy between these two camera information terms, emphasizing their crucial role in achieving superior geometry and appearance in the generated assets.

#### 4.4 Attention Map of Different Layers

In this section, we present visualizations of the attention maps from various attention layers during the geometry learning stage and the appearance learning stage. As illustrated in Fig. 13 and Fig. 14, the attention maps generated by different attention layers consistently exhibit the ability to accurately locate the shape of the foreground object. This demonstrates the robustness and effectiveness of the attention mechanism in capturing the essential features of the foreground object. Furthermore, we observe that attention maps from different layers seem to exhibit variations in focus, with some emphasizing the foreground, some highlighting the edges, and some concentrating on the interior of the object. This divergence in focus can be attributed to the distinct roles played by different layers in the stable diffusion training process. Each layer contributes its unique perspective and emphasis, leading to variations in attention focus. Nevertheless, despite these variations, all attention maps generated by X-Dreamer successfully locate the foreground objects with precision. This outcome provides strong evidence that our proposed module effectively guides the model’s attention towards the foreground objects. By incorporating the attention mechanism into the training process, X-Dreamer ensures that the model focuses on the most relevant regions, resulting in accurate and reliable detection of the foreground objects throughout the geometry and appearance learning stages.

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Prompt: A DSLR photo of Lord Voldemort's head, highly detailed.

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

Prompt: A pumpkin, highly detailed,8K,HD.

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Prompt: Garlic with white skin, highly detailed,8K,HD.

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

Prompt: A Vintage Style LED Lantern.

Fig. 14. Attention maps of different layers of SD for the appearance learning stage.

#### 4.5 Rendered Images under Different Environment Maps

Our method, while employing a fixed environment for training the model, offers significant advantages and flexibility in the rendering of 3D assets generated by X-Dreamer. It is worth noting that these assets are not limited to a single environment; instead, they possess the capability to be rendered in diverse environments, expanding their applicability and adaptability. The visual evidence presented in Fig. 15 reinforces our claim. By utilizing different environment maps, we can readily observe the striking variations in the appearance of the rendered images of these 3D assets. This compelling demonstration showcases the inherent versatility and potential of X-Dreamer’s generated assets to seamlessly integrate into various rendering engines. The ability to effectively address the challenges posed by different lighting conditions is a crucial aspect of any rendering engine. By leveraging the capabilities of X-Dreamer, we unlock the possibility of using its 3D assets in a multitude of rendering engines. This versatility ensures that the assets can adapt and thrive in diverse lighting scenarios, enabling them to meet the demands of real-world applications in fields such as architecture, virtual reality, and computer graphics.

#### 4.6 Limitation

One limitation of the proposed X-Dreamer is its inability to generate multiple objects simultaneously. When the input text prompt contains two distinct objects, X-Dreamer may produce a mixed object that combines properties of both objects. A clearer understanding of this limitation can be gained by referring to Fig. 16. In the first row of Fig. 16, our intention is for X-Dreamer to generate “A sliced loaf of fresh bread and a cabbage." However, the resulting output consists of a bread with a few cabbage leaves, indicating the model’s difficulty in generating separate objects accurately.

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

A statue of Leonardo DiCaprio's head.

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

A plate piled high with chocolate chip cookies.

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

A 3D rendering of Batman, highly detailed.

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

A pumpkin, highly detailed,8K,HD.

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

A beautifully carved wooden queen chess piece.

Environment Map

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Fig. 15. Rendered images of generated 3D assets under different environment maps.

Similarly, in the second row of Fig. 16, we aim to generate “A blue and white porcelain vase and an apple." However, the output depicts a blue and white porcelain vase with a shape resembling an apple. Nevertheless, it is important to note that this limitation does not significantly undermine the value of our work. In applications such as games and movies, each object functions as an independent entity, capable of creating complex scenes through interaction with other objects. Our primary objective is to generate high-quality 3D objects, and while the current limitation exists, it does not diminish the overall quality and utility of our approach.

#### 5 Conclusion

This study introduces a groundbreaking framework called X-Dreamer, which is designed to enhance text-to-3D synthesis by addressing the domain gap between text-to-2D and text-to-3D generation. To achieve this, we first propose CG-LoRA, a module that incorporates 3D-associated

[Figure 404]

[Figure 405]

A sliced loaf of fresh bread and a cabbage.

[Figure 406]

[Figure 407]

A DSLR photo of a blue and white porcelain vase and an apple, highly detailed, 8K, HD.

Fig. 16. Bad cases of X-Dreamer.

information, including direction-aware text and camera parameters, into the pretrained Stable Diffusion (SD) model. By doing so, we enable the effective capture of information relevant to the

- 3D domain. Furthermore, we design AMA loss to align the attention map generated by SD with the rendered mask of the 3D object. The primary objective of AMA loss is to guide the focus of the textto-3D model towards the generation of foreground objects. Through extensive experiments, we have thoroughly evaluated the efficacy of our proposed method, which has consistently demonstrated its ability to synthesize high-quality and photorealistic 3D content from given text prompts. References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] Sudarshan Babu, Richard Liu, Avery Zhou, Michael Maire, Greg Shakhnarovich, and Rana Hanocka. 2023. Hyperfields: Towards zero-shot generation of nerfs from text. arXiv preprint arXiv:2310.17075 (2023).
- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. 2022. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324 (2022).
- [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- [5] Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, and Kangxue Yin. 2023. Texfusion: Synthesizing 3d textures with text-guided image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4169–4181.
- [6] Jingwen Chen, Yingwei Pan, Ting Yao, and Tao Mei. 2023. Controlstyle: Text-driven stylized image generation using diffusion priors. In Proceedings of the 31st ACM International Conference on Multimedia. 7540–7548.
- [7] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873 (2023).
- [8] Yang Chen, Yingwei Pan, Yehao Li, Ting Yao, and Tao Mei. 2023. Control3d: Towards controllable text-to-3d generation. In Proceedings of the 31st ACM International Conference on Multimedia. 1148–1156.
- [9] Zilong Chen, Feng Wang, and Huaping Liu. 2023. Text-to-3D using Gaussian Splatting. arXiv preprint arXiv:2309.16585

(2023).

- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2023. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238 (2023).
- [11] Ziluo Ding, Hao Luo, Ke Li, Junpeng Yue, Tiejun Huang, and Zongqing Lu. 2023. Clip4mc: An rl-friendly vision-language model for minecraft. arXiv preprint arXiv:2303.10571 (2023).
- [12] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, and Tat-Seng Chua. 2024. Dysen-VDM: Empowering Dynamicsaware Text-to-Video Diffusion with LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7641–7653.
- [13] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong-Li Lee, and Wynne Hsu. 2024. Video-ofthought: Step-by-step video reasoning from perception to cognition. In Proceedings of the International Conference on Machine Learning.
- [14] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. 2024. VITRON: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing. (2024).
- [15] Hao Fei, Shengqiong Wu, Meishan Zhang, Min Zhang, Tat-Seng Chua, and Shuicheng Yan. 2024. Enhancing videolanguage representations with structural spatio-temporal alignment. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
- [16] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, Zi-Xin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. 2023. threestudio: A unified framework for 3D content generation. https://github.com/threestudio-project/threestudio.
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.

- 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

[19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.

- 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9

- [20] Shuo Huang, Zongxin Yang, Liangting Li, Yi Yang, and Jia Jia. 2023. AvatarFusion: Zero-shot Generation of ClothingDecoupled 3D Avatars Using 2D Diffusion. In Proceedings of the 31st ACM International Conference on Multimedia. 5734–5745.
- [21] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. 2023. DreamWaltz: Make a Scene with Complex 3D Animatable Avatars. arXiv preprint arXiv:2305.12529 (2023).
- [22] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. 2024. Dreamwaltz: Make a scene with complex 3d animatable avatars. Advances in Neural Information Processing Systems 36

(2024).

- [23] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. 2022. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 867–876.
- [24] Ruixiang Jiang, Lingbo Liu, and Changwen Chen. 2023. Clip-count: Towards text-guided zero-shot object counting. In Proceedings of the 31st ACM International Conference on Multimedia. 4535–4545.
- [25] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. 2023. AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control. arXiv preprint arXiv:2303.17606 (2023).
- [26] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. 2023. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1931–1941.
- [27] Jiabao Lei, Yabin Zhang, Kui Jia, et al. 2022. Tango: Text-driven photorealistic and robust 3d stylization via lighting decomposition. Advances in Neural Information Processing Systems 35 (2022), 30923–30936.
- [28] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. 2024. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems 36 (2024).
- [29] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. 2023. Diffusion-sdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12642–12651.
- [30] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. 2023. SWEETDREAMER: ALIGNING GEOMETRIC PRIORS IN 2D DIFFUSION FOR CONSISTENT TEXT-TO-3D. arXiv preprint arXiv:2310.02596 (2023).
- [31] Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. 2024. Focaldreamer: Text-driven 3d editing via focal-fusion assembly. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 3279–3287.

- [32] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. 2023. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv preprint arXiv:2311.06607

(2023).

- [33] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 300–309.
- [34] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems 36 (2024).
- [35] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9298–9309.
- [36] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, Ming-Yu Liu, Sanja Fidler, and James Lucas. 2023. Att3d: Amortized text-to-3d object synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 17946–17956.
- [37] Lingxiao Lu, Jiangtong Li, Junyan Cao, Li Niu, and Liqing Zhang. 2023. Painterly image harmonization using diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia. 233–241.
- [38] Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. 2024. Feast Your Eyes: Mixture-ofResolution Adaptation for Multimodal Large Language Models. arXiv preprint arXiv:2403.03003 (2024).
- [39] Yiwei Ma, Jiayi Ji, Xiaoshuai Sun, Yiyi Zhou, and Rongrong Ji. 2023. Towards local visual modeling for image captioning. Pattern Recognition 138 (2023), 109420.
- [40] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. 2022. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval. In Proceedings of the 30th ACM International Conference on Multimedia. 638–647.
- [41] Yiwei Ma, Xiaoqing Zhang, Xiaoshuai Sun, Jiayi Ji, Haowei Wang, Guannan Jiang, Weilin Zhuang, and Rongrong Ji.

2023. X-Mesh: Towards Fast and Accurate Text-driven 3D Stylization via Dynamic Textual Guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2749–2760.

- [42] Stephen McAuley, Stephen Hill, Naty Hoffman, Yoshiharu Gotanda, Brian Smits, Brent Burley, and Adam Martinez.

2012. Practical physically-based shading in film and game production. In ACM SIGGRAPH 2012 Courses. 1–7.

- [43] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. 2023. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12663–12673.
- [44] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. 2022. Text2mesh: Text-driven neural stylization for meshes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13492– 13502.
- [45] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
- [46] Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Tiberiu Popa. 2022. Clip-mesh: Generating textured meshes from text using pretrained image-text models. In SIGGRAPH Asia 2022 conference papers. 1–8.
- [47] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41, 4 (2022), 1–15.
- [48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).
- [49] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).
- [50] Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, and Tat-Seng Chua. 2023. Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation. In Proceedings of the 31st ACM International Conference on Multimedia. 643–654.
- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.
- [52] Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. 2023. Dreambooth3d: Subject-driven text-to-3d generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2349–2359.
- [53] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1, 2 (2022), 3.
- [54] Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. 2023. Texture: Text-guided texturing of 3d shapes. In ACM SIGGRAPH 2023 Conference Proceedings. 1–11.

- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [56] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo.

2023. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10219–10228.

- [57] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35 (2022), 36479–36494.
- [58] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems 35 (2022), 25278–25294.
- [59] Etai Sella, Gal Fiebelman, Peter Hedman, and Hadar Averbuch-Elor. 2023. Vox-e: Text-guided voxel editing of 3d objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 430–440.
- [60] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, and Seungryong Kim. 2023. Let 2d diffusion model know 3d-consistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937 (2023).
- [61] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. 2021. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems 34 (2021), 6087–6101.
- [62] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.
- [63] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020).
- [64] Gan Sun, Wenqi Liang, Jiahua Dong, Jun Li, Zhengming Ding, and Yang Cong. 2024. Create your world: Lifelong text-to-image diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
- [65] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. 2023. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184 (2023).
- [66] Kenneth E Torrance and Ephraim M Sparrow. 1967. Theory for off-specular reflection from roughened surfaces. Josa 57, 9 (1967), 1105–1114.
- [67] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. 2022. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers.
- [68] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. 2023. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12619–12629.
- [69] Qiang Wang, Junlong Du, Ke Yan, and Shouhong Ding. 2023. Seeing in Flowing: Adapting CLIP for Action Recognition with Motion Prompts Learning. In Proceedings of the 31st ACM International Conference on Multimedia. 5339–5347.
- [70] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. 2024. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. Advances in Neural Information Processing Systems 36 (2024).
- [71] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. arXiv preprint arXiv:2305.16213

(2023).

- [72] Jiacheng Wei, Hao Wang, Jiashi Feng, Guosheng Lin, and Kim-Hui Yap. 2023. TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16805–16815.
- [73] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. 2024. NExT-GPT: Any-to-Any Multimodal LLM. In Proceedings of the International Conference on Machine Learning.
- [74] Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, and Shenghua Gao. 2023. Dream3d: Zero-shot text-to-3d synthesis using 3d shape prior and text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20908–20918.
- [75] Haibo Yang, Yang Chen, Yingwei Pan, Ting Yao, Zhineng Chen, and Tao Mei. 2023. 3dstyle-diffusion: Pursuing fine-grained text-driven 3d stylization with 2d diffusion models. In Proceedings of the 31st ACM International Conference on Multimedia. 6860–6868.
- [76] Chaohui Yu, Qiang Zhou, Jingliang Li, Zhe Zhang, Zhibin Wang, and Fan Wang. 2023. Points-to-3d: Bridging the gap between sparse points and shape-controllable text-to-3d generation. In Proceedings of the 31st ACM International Conference on Multimedia. 6841–6850.

###### [77] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

