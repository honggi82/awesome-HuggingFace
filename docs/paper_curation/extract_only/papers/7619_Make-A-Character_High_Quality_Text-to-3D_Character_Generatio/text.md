## Make-A-Character: High Quality Text-to-3D Character Generation within Minutes

# arXiv:2312.15430v1[cs.CV]24Dec2023

Jianqiang Ren, Chao He, Lin Liu, Jiahao Chen, Yutong Wang, Yafei Song, Jianfang Li, Tangli Xue, Siqi Hu, Tao Chen, Kunkun Zheng, Jianjing Xiang, Liefeng Bo Institute for Intelligent Computing, Alibaba Group

{jianqiang.rjq, yichao.hc, lorrain.ll, peter.cjh, yutong.yutongwang, huaizhang.syf, wuhui.ljf, xuetangli.xtl, husiqi.hsq, ct253279, kunkun.zkk, jianjing.xjj, liefeng.bo}@alibaba-inc.com

### Abstract

There is a growing demand for customized and expressive 3D characters with the emergence of AI agents and Metaverse, but creating 3D characters using traditional computer graphics tools is a complex and time-consuming task. To address these challenges, we propose a user-friendly framework named Make-A-Character (Mach) to create lifelike 3D avatars from text descriptions. The framework leverages the power of large language and vision models for textual intention understanding and intermediate image generation, followed by a series of human-oriented visual perception and 3D generation modules. Our system offers an intuitive approach for users to craft controllable, realistic, fully-realized 3D characters that meet their expectations within 2 minutes, while also enabling easy integration with existing CG pipeline for dynamic expressiveness. For more information, please visit the project page at https://human3daigc.github.io/MACH/.

### 1. Introduction

Realistic-looking 3D avatars have been widely utilized in the realms of video games, VR/AR, and film industry. With the rise of the Metaverse and AI agents, the demand for personalized and expressive character creation has surged in fields like virtual meetings, conversational agents, and intelligent customer service. However, for general users, creating a personalized 3D avatar using traditional digital creation tools is a complex and time-consuming task. To lower the barrier to 3D digital human creation, this work unveils an innovative system, named Make-A-Character (Mach), which harnesses the power of large language and vision foundation models to generate detailed and lifelike 3D avatars from simple text descriptions. Our system seamlessly converts textual descriptors into vivid visual avatars, providing users with a simple way to create custom avatars that resonate with their

intended personas.

We summarize the properties of our generated 3D characters as follows:

Controllable. Our system empowers users with the ability to customize detailed facial features, including the shape of the face, eyes, the color of the iris, hairstyles and colors, types of eyebrows, mouths, and noses, as well as the addition of wrinkles and freckles. This customization is facilitated by intuitive text prompts, offering a user-friendly interface for personalized character creation.

Highly-Realistic. The characters are generated based on a collected dataset of real human scans. Additionally, their hairs are built as strands rather than meshes. The characters are rendered using PBR (Physically Based Rendering) techniques in Unreal Engine, which is renowned for its highquality real-time rendering capabilities.

Fully-Completed. Each character we create is a complete model, including eyes, tongue, teeth, a full body, and garments. This holistic approach ensures that our characters are ready for immediate use in a variety of situations without the need for additional modeling.

Animatable. Our characters are equipped with sophisticated skeletal rigs, allowing them to support standard animations. This contributes to their lifelike appearance and enhances their versatility for various dynamic scenarios.

Industry-Compatible. Our method utilizes explicit 3D representation, ensuring seamless integration with standard CG pipelines employed in the game and film industries.

### 2. Method

Mach aims to create complete, lifelike, drivable 3D virtual avatars that are compatible with existing CG pipeline and offer flexible styling and animation ability. Therefore, we have opted for an explicit 3D representation(i.e., surface mesh and texture) rather than an implicit approach like NeRF. In terms of geometric base model selection, we conduct research on various models including BFM, FLAME [9], Daz 3D [4],

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

An old man with wrinkles on his face, he has gray hair.

A cool girl, sporting ear-length short hair, freckles on her cheek.

Chinese girl, Bobo haircut with a straight bangs, around 20 years old. She has a Vshaped face, cameo lipstick.

A boy with brown skin and black glasses, green hair.

A girl, she is thin, willow leaf eyebrow, oval face, long curly hair.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

A handsome man with glasses, brown hair and black eyes.

Pretty girl, diamond face, blue eyes.

A chubby lady with round face.

An elderly lady, gray short hair.

A middle-aged man, sword-like eyebrow, black short hair.

- Figure 1. The Mach system is capable of creating highly detailed and varied 3D avatars from text prompts. We demonstrate the versatility of these characters by showcasing their ability to express dynamic animations through various facial expressions.

Metahuman [11], and SMPL [10]. Ultimately, we choose MetaHuman because it includes a complete representation of the face and body, and it offers more nuanced expression animation capabilities, primarily because of its advanced facial rig system [15], which offers a powerful support for the vivid dynamic performance of virtual characters.

The architecture of Mach is illustrated in Figure 2. Given a text prompt, The Large Language Model (LLM) is leveraged for semantic comprehension, enabling the extraction of various facial attributes, such as face shape, eyes shape, mouth shape, hairstyle and color, glasses type. Some of these semantic attributes are then mapped to corresponding visual clues, which serve as fine guidance for generating a reference portrait image using Stable Diffusion [16] and ControlNet [24]. The reference portrait is guaranteed to be frontal with neutral expression owing to our posture control, which brings great convenience to head geometry and texture generation. We build a conversion mechanism between head mesh and triplane maps, thus we can directly optimize

- 2D maps instead of resorting to 3DMM methods, which offers flexible vertex-level control. Differentiable rendering and delighting techniques are utilized to extract and refine diffuse texture based on the reference image, and our hair generation module enhances the overall expressiveness by providing strand-level hair synthesis. For other accessories such as garments, glasses, eyelashes, and irises, we match them from the tagged 3D asset library with extracted semantic attributes, and finally assemble them into a complete 3D figure. The duration of the entire process is within 2 minutes. Detailed introductions to each module are provided in the following sections. 2.1. LLM-Powered Visual Prompt Generation

The utilization of large models is illustrated in Figure 3. Due to Stable Diffusion’s insensitivity to subtle facial at-

tributes (including face shape, eyebrows, eye shape, nose, mouth, etc.), it fails to provide finer-grained control over these attributes. To address this limitation, we perform facial attributes analysis on the text prompt using Qwen-14B [1] to acquire visual clues related to these attributes, and then apply ControlNet to regulate the fine-grained features of facial components. In the deployment of ControlNet, we integrate Openpose and canny maps to ensure a reasonable distribution of facial features, eventually obtaining reference images that are strongly correlated with the text prompts.

#### 2.2. Dense Landmark Detection

Face landmarks refer to identifiable points on face that correspond semantically across different individuals, such as the nose tip and mouth corners. They typically represent the face geometry on 2D image domain and are essential for reconstructing a 3D face from a single image. The traditional 68 or 98 face landmarks in [17] and [21] are considered sparse landmarks, meaning they cover only a limited area of face and leave regions like the forehead and cheeks without landmarks. The absence of landmarks in these areas such as the forehead makes it challenging to reconstruct the structure of the forehead accurately.

To overcome these limitations, previous works utilize supplementary information like image colors [5, 6], optical flow [2]. However, the reliability of such data on face images is often questionable, as it can change drastically with varying lighting conditions and different camera viewing angles. In contrast, we utilize dense facial landmarks as the primary information to reconstruct face and head geometry inspired by [19]. Since dense facial landmarks cannot be annotated manually, we follow the work [19] to adopt synthetic images for training data.

We established a multi-view capturing and processing pipeline (Figure 4) to produce uniformly topological head

[Figure 11]

|Text Prompt|
|---|

||ControlNet|
|---|
<br><br>|Stable Diffusion Model|
|---|
|
|---|

|Dense Facial Landmark|
|---|

|Geometry Generation|
|---|

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

|Texture Generation|
|---|

|Hair Generation|
|---|

|Assets Matching|
|---|

|[Figure 15]<br><br>LLM|
|---|

|[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]|
|---|

Tagged 3D Assets

- Figure 2. The framework utilizes the Large Language Model (LLM) to extract various facial attributes(e.g., face shape, eyes shape, mouth shape, hairstyle and color, glasses type). These semantic attributes are then mapped to corresponding visual clues, which in turn guide the generation of reference portrait image using Stable Diffusion along with ControlNet. Through a series of 2D face parsing and 3D generation modules, the mesh and textures of the target face are generated and assembled along with additional matched accessories. The parameterized representation enable easy animation of the generated 3D avatar.

Text

prompt DenoisingU-Net 𝐷

[Figure 23]

[Figure 24]

|LLM|
|---|

[Figure 25]

Text Encoder

ControlNet

[Figure 26]

Visual Clues

Noise

Reference Image

T

[Figure 27]

- Figure 3. We find that generating corresponding detailed facial attributes guided by only text prompt using the Stable Diffusion Model is challenging due to the absence of comprehensive facial annotations and corresponding image pairs. To address this issue, we employ the Large Language Model (LLM) to extract attributes and align them with low-level visual clues, such as posture and edge maps. These visual clues then direct the text-to-image (T2I) generation process via a ControlNet, enhancing the model’s ability to accurately render facial details.

[Figure 28]

[Figure 29]

(a) Face Scanner (b) Body Scanner

Figure 4. Our multi-view light stage setups are designed for capturing high-resolution 3D scans of heads and bodies.

Method Is Dense Re-projection error

StarLoss [25] Not 4.00 MediaPipe Yes 5.31 Ours Yes 3.19

Table 1. Re-projection errors on FaceScape [22] dataset.

geometries and facial textures from 1,000 real human scans. Additionally, we utilized a variety of digital assets, including 23 hairs, 45 clothes, 8 hats, and 13 moustaches, to create complete human head models. For generating different facial expressions, we employed the classic set of 52 blendshapes and rendered each model from multiple angles. For landmark detection, we employ the stacked hourglass networks [12] to regress heat maps for each facial landmark. Since the systematic definition of landmarks is different from traditional sparse landmarks, we conducted comparisons by calculating re-projection errors on the FaceScape [22] dataset, Specifically, we measured the re-projection errors for 98 landmarks excluding those along the jawline. The results are presented in Table 1.

#### 2.3. Geometry Generation

Once given the reference portrait image and corresponding dense facial landmarks, we reconstruct the head geometry under the guidance of these landmarks. We firstly establish a conversion mechanism between 3D mesh and 2D maps, this is accomplished by mapping each vertex’s position onto three orthogonal planes (i.e. Y-Z, X-Z, and X-Y planes) in accordance with its UV coordinates, thus we can represent the 3D mesh with a 3-channels image, referred to as triplane. This representation enhances the potential for geometry generation through a 2D diffusion model and facilitates the

[Figure 30]

[Figure 31]

[Figure 32]

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|Dense Landmark Detection|
|---|

Reference Image Dense Landmarks

(a) 68 Landmarks (b) 98 Landmarks (c) 431 Landmarks

Figure 5. The traditional 68 landmarks [17], 98 landmarks [21], and our 431 landmarks. The traditional landmarks are sparse on face. In contrast, our 431 landmarks are dense landmarks that cover the whole head.

Learnable Parameters

|Camera Position|Camera Rotation|Tri-plane Maps|
|---|---|---|

[Figure 35]

local smoothness constraints. For each 2D landmarks pk, we predefine its vertex correspondence Vk on base mesh, and introduce the landmark projection loss as:

[Figure 36]

- y

x

- z Vertexpositionsprojected onto the three planes( YZ, XZ, XY) according to UV

[Figure 37]

Canonical Space

K

mapping.

||pk − proj(Vk,Tcam,Rcam,Incam)||2 (1)

Llmk =

k=1

[Figure 38]

[Figure 39]

Vk = triplane[vk,uk,:] (2)

Merge with Body

where proj is an projection operation, Tcam and Rcam are translation and rotation of camera respectively, and Incam is fixed camera intrinsics. uk and vk are the uv-coordinate of vertex Vk.

The total variation loss is introduced to encourage local smoothness on mesh:

Figure 6. In the process of geometry generation, we optimize the camera parameters and triplane maps under the guidance of dense facial landmarks and the reference image. The position of each vertex is encoded into the triplane maps based on its corresponding UV coordinates.

Ltv = TV (triplane) (3)

We additionally add symmetric loss on triplane to encourage facial symmetry for aesthetics consideration:

Cam are the results of the geometry generation from the previous section, and Tex denotes the target texture we aim to fit. The loss between the rendered image and the target image is calculated by the following equation:

Lsym = ||triplane[...,0] + flip(triplane[...,0]))||2

(4)

- +||triplane[...,1] − flip(triplane[...,1]))||2
- +||triplane[...,2] − flip(triplane[...,2]))||2

IR = DR(Geo,Tex,Cam) (6) LTex = ||IR − IT||F + α ∗ TV (IR) (7)

The total loss function on geometry generation are given by:

L = Llmk + λtvLtv + λsymLsym (5)

where, DR denotes differentiable rendering, IR is the rendered image, IT is the target image, α is a weight coefficient, here set to 0.01.

#### 2.4. Texture Generation

##### 2.4.1 Texture Extraction

After fitting the geometry to match the reference image, we employ differentiable rendering method to get the required texture image. The camera setting used here are kept consistent with geometry generation stage. Since there is not always a one-to-one correspondence between pixel positions and UV coordinates, we adopt a multi-resolution approach for texture generation by progressively generating texture from low to high resolution. As shown in Figure 7, Geo and

##### 2.4.2 Diffuse Albedo Estimation

Using the texture obtained directly from differentiable rendering is not ideal, as it fails to fully separate the illumination and diffuse albedo components from the texture image. Consequently, when rendering under varied lighting conditions, the textures containing baked illumination may restult in noticeable unrealistic shadings and shadows. To

[Figure 40]

[Figure 41]

[Figure 42]

32

[Figure 43]

128

[Figure 44]

𝐶𝑎𝑚

[Figure 45]

[Figure 46]

[Figure 47]

|Camera Position|
|---|

512

[Figure 48]

|Camera Rotation|
|---|

[Figure 49]

differentiable rendering

1024

[Figure 50]

- (a) collected geometry and diffuse albedo

[Figure 51]

- (b) baked textures under different illuminations

[Figure 52]

[Figure 53]

geometry multi-resolution texture rendered image target image

𝐺𝑒𝑜 𝑇𝑒𝑥

L2 Loss + TV Loss

Figure 7. We employ a multi-resolution strategy, leveraging differentiable rendering methods, to obtain a detailed and coherent texture map. Our resolution hierarchy is defined as follows: [32, 128, 512, 1024].

Figure 8. The data acquisition and processing pipeline.

facial mask artifacts mask

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

tackle this issue, we introduce a neural delighting method to remove the undesired illumination from the texture image and get the render-ready diffuse albedo. It is noteworthy that our delighting algorithm works on texture images rather than portrait images. It is an intentional choice since texture images are free from occlusions and consistent across different poses and expressions of portraits, making the data acquisition and the algorithm learning more tractable.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

de-lighting blending

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

input texture facial region diffuse albedo corrected texture final texture

[Figure 72]

[Figure 73]

template make-ups

Figure 9. The proposed texture correction and completion pipeline.

##### 2.4.3 Texture correction and completion

Ground Truth Data Collection. As in [7], we capture faces of 193 individuals (including 100 females and 93 males aged between 20-60 years old) under uniform illumination. By reconstructing the geometries, we obtain the unwarpped high resolution diffuse albedo, as in Figure 8 (a).

After delighting, the generated diffuse albedo may still has artifacts in the vicinity of eyes, mouth and nostril regions. This is a result of the inherent limitations of the single frontal portrait image, which only provides limited textural information about a face. When mapped onto a 3D face geometry, the imperfect diffuse albedo introduces misaligned semantic features near eyes, mouth and nostril regions, leading to aesthetically unpleasant results. To address this issue, we utilize an off-the-shelf face parsing algorithm[23] to extract masks of these error-prone regions, which are then carefully dilated and merged with a template diffuse albedo using the Poisson method[13]. Additionally, We transfer the colors of the mouth and eyebrows from the portrait image to maintain the facial features. Finally, the facial region is Poisson blended with the template diffuse albedo to obtain the textures of the ears and neck. We also add make-ups around eyes and cheeks to improve the aesthetics. Figure 9 demonstrates the proposed texture correction and completion pipeline.

Training Data Generation. The textures under varying illumination are synthesized by baking lights into the ground truth diffuse albedo. In order to cover the wide range of natural lighting conditions, we bake 100 high dynamic range (HDR) lights (including indoor/outdoor, day/ night scenarios) for each ground truth data. To improve the data diversity and avoid overfitting, the skin colors of the ground truth diffuse albedos are augmented according to the Individual Typology Angle (ITA)[3]. Figure 8 (b) illustrates the baked textures in different illuminations.

Delighting Network. Without losing generality, we formulate the texture delighting problem as an image-to-image translation problem. Specifically, we employ the coarse-tofine pix2pixHD network[18], which takes the synthesized illuminated textures as input and generates visually appealing high-resolution diffuse albedos. As in [18], the loss function is defined as a weighted combination of GAN loss and VGG feature matching loss. We train the network at the resolution of 1024 using the default parameters.

#### 2.5. Hair Generation

To produce a high-fidelity 3D avatar, we render hair as individual strands rather than as meshes. We firstly synthesis various hairstyle images via SD models, and then conduct 3D strand-based hair reconstruction from these 2D images.

[Figure 74]

[Figure 75]

[Figure 76]

occupancy orientation

image2voxel hair mask

input image

[Figure 77]

[Figure 78]

postprocess

voxel2strand

final result

2d orientation

Figure 10. Hair generation pipeline.

We incorporate SOTA research such as NeuralHDHair [20], and train our model using the USC-HairSalon [8] dataset. The hair reconstruction process consists of two main stages. Initially, the occupancy and orientation fields of the hair in

- 3D space are estimated based on the input 2D image. Subsequently, leveraging the discrete voxel-level data generated in the first phase, geometric descriptions of tens of thousands of hair strands are generated. The complete hairstyle generation procedure also involves pre-processing operations such as face alignment, hair mask segmentation, and 2D orientation recognition. Additionally, post-processing operations are applied to enhance the initial generated hair results, including the removal of unreasonable strands, fitting the hair to the target head mesh, and performing geometric deformations to achieve high-quality reconstruction results. The overall pipeline is illustrated in the Figure 10.

Considering that real-time hairstyle generation is timeconsuming, we opt to generate diverse hairstyle assets offline. These generated hair assets, along with existing metahuman hairs, are labeled with descriptive attributes such as hairstyle type, length, and degree of crimp. This attribute labeling enables efficient matching processes.

#### 2.6. Assets Matching

To construct a fully-realized 3D character, we must integrate the generated head, hair, body, garments, and some accessories together. Each pre-produced asset is labeled with textual annotations, either manually or through annotation algorithm. To select the most suitable asset that matches the input prompt, we employ CLIP’s text encoder [14] to compute the cosine similarity between the features of the input prompt and the labels of the assets, the asset with the highest similarity is then selected.

### 3. Results

We present the visual results of the generated 3D avatars in Figure 1, accompanied by respective input prompts listed below each avatar. In Figure 12, we showcase the expressive animations achieved through facial rig control. These showcases were rendered using the Unreal Engine.

[Figure 79]

Figure 11. Strand-based hair generation is guided by hairstyle images that are generated using SD models.

### 4. Future Work

Our current version focuses on generating visually appealing 3D avatars of Asian ethnicity, as our selected SD model is primarily trained on Asian facial images. In the future, we will try to expand support for different ethnicities and styles. It is worth noting that our de-lighting datasets consist of clean face textures only, non-natural facial patterns like scribbles or stickers may be weakened in the generated avatars. Currently, our garments and body parts are pre-produced and matched based on textual similarity. However, we are actively working on developing cloth, expression, and motion generation techniques driven by text prompts.

### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 2
- [2] Chen Cao, Menglei Chai, Oliver Woodford, and Linjie Luo. Stabilized real-time face tracking via a learned dynamic rigidity prior. ACM Trans. Graph., 37(6), 2018. 2
- [3] A. CHARDON, I. CRETOIS, and C. HOURSEAU. Skin colour typology and suntanning pathways. International Journal of Cosmetic Science, page 191–208, 1991. 5

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Figure 12. Dynamic expression by employing facial rig control system [15].

- [4] Daz3D. Daz3d. https://www.daz3d.com/ technology/, 2023. 1
- [5] Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. Accurate 3d face reconstruction with weaklysupervised learning: From single image to image set. In IEEE Computer Vision and Pattern Recognition Workshops, 2019. 2
- [6] Yao Feng, Haiwen Feng, Michael J. Black, and Timo Bolkart. Learning an animatable detailed 3D face model from in-thewild images. ACM Transactions on Graphics (ToG), Proc. SIGGRAPH, 40(4):88:1–88:13, 2021. 2
- [7] Abhijeet Ghosh, Graham Fyffe, Borom Tunwattanapong, Jay Busch, Xueming Yu, and Paul Debevec. Multiview face capture using polarized spherical gradient illumination. ACM Transactions on Graphics, page 1–10, 2011. 5
- [8] Liwen Hu, Chongyang Ma, Linjie Luo, and Hao Li. Singleview hair modeling using a hairstyle database. ACM Transactions on Graphics (ToG), 34(4):1–9, 2015. 6
- [9] Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Transactions on Graphics, 36(6CD): 1–17, 2017. 1
- [10] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 2
- [11] Metahuman. Metahuman. https : / / www . unrealengine . com / en - US / metahuman, 2023. 2
- [12] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In European Conference on Computer Vision, 2016. 3
- [13] Patrick P´erez, Michel Gangnet, and Andrew Blake. Poisson image editing. ACM Transactions on Graphics, page 313–318,

2003. 5

- [14] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 6
- [15] RigLogic. rig-logic-whitepaper-v2-5c9f23f7e210. https : / / cdn2 - unrealengine - 1251447533 . file.myqcloud.com/rig-logic-whitepaperv2-5c9f23f7e210.pdf, 2023. 2, 7
- [16] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of

- the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [17] Christos Sagonas, Georgios Tzimiropoulos, Stefanos Zafeiriou, and Maja Pantic. 300 faces in-the-wild challenge: The first facial landmark localization challenge. In 2013 IEEE International Conference on Computer Vision Workshops, pages 397–403, 2013. 2, 4
- [18] Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. High-resolution image synthesis and semantic manipulation with conditional gans. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018. 5
- [19] Erroll Wood, Tadas Baltruˇsaitis, Charlie Hewitt, Matthew Johnson, Jingjing Shen, Nikola Milosavljevi´c, Daniel Wilde, Stephan Garbin, Toby Sharp, Ivan Stojiljkovi´c, et al. 3d face reconstruction with dense landmarks. In European Conference on Computer Vision, pages 160–177. Springer, 2022. 2
- [20] Keyu Wu, Yifan Ye, Lingchen Yang, Hongbo Fu, Kun Zhou, and Youyi Zheng. Neuralhdhair: Automatic high-fidelity hair modeling from a single image using implicit neural representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1526–1535, 2022. 6
- [21] Wayne Wu, Chen Qian, Shuo Yang, Quan Wang, Yici Cai, and Qiang Zhou. Look at boundary: A boundary-aware face alignment algorithm. In CVPR, 2018. 2, 4
- [22] Haotian Yang, Hao Zhu, Yanru Wang, Mingkai Huang, Qiu Shen, Ruigang Yang, and Xun Cao. Facescape: a large-scale high quality 3d face dataset and detailed riggable 3d face prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3
- [23] Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation, page 334–349.

2018. 5

- [24] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [25] Zhenglin Zhou, Huaxia Li, Hong Liu, Nanyang Wang, Gang Yu, and Rongrong Ji. Star loss: Reducing semantic ambiguity in facial landmark detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15475–15484, 2023. 3

