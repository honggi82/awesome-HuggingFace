# arXiv:2407.12705v2[cs.CV]6Aug2024

### IMAGDressing-v1: Customizable Virtual Dressing

#### Fei Shen1 , Xin Jiang1 , Xin He2, Hu Ye3, Cong Wang4, Xiaoyu Du1, Zechao Li1, Jinhui Tang1#

|1Nanjing University of Science and Technology 2Wuhan University of Technology 3Tencent AI Lab 4Nanjing University https://imagdressing.github.io/<br><br>realistic virtual try-on inpainting using latent enhancing consumers’ online existing VTON technologies<br><br>over garments, optional this issue, we define a<br><br>on generating freely edgarments and optional concomprehensive affinity met-<br><br>consistency between gengarments. Then, we propose<br><br>a garment UNet that<br><br>CLIP and texture features attention module, including<br><br>cross-attention, to integarment UNet into a frozen can control different scenes can be combined with other<br><br>and IP-Adapter, to enof generated images. of data, we release the in-<br><br>dataset, containing over dressed images, and estabassembly. Extensive experi-<br><br>performance under various and model will be available<br><br>achieve comprehensive and for merchants by utilizing faces, poses, and descrip-<br><br>significant potential for<br><br>and entertainment. focus on virtual try-on<br><br>al. 2021; Kim et al. 2024;<br><br>Yang et al. 2024) tasks given garments and fixed hu-<br><br>it also presents more sig-<br><br>|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>Consumer<br><br>Generate<br><br>Fit or not<br><br>[Figure 11]<br><br>[Figure 12]<br><br>Garment<br><br>[Figure 13]<br><br>[Figure 14]<br><br>User Photo<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>(Fixed Condition)|
|---|
<br><br>(a) Virtual Try-On<br><br>|[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>|[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>“a white background” Text<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Pose<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>Face|
|---|
<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>Merchant<br><br>Generate<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>Garment Display<br><br>Garment<br><br>[Figure 52]<br><br>[Figure 53]<br><br>(Optional Condition)<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]|
|---|
<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>(b) Virtual Dressing<br><br><br>Figure 1: Differences between virtual try-on and virtual dressing tasks in conditions and applicable scenarios.<br><br>To enhance the shopping experience for consumers in ecommerce, VTON (Han et al. 2018; Choi et al. 2021) tasks have become increasingly popular within the community. Early methods primarily relied on generative adversarial network (GAN) (Creswell et al. 2018), typically comprising a warping module to learn the semantic correspondence between clothes and the human body, and a generator module to synthesize the warped clothes onto the person’s image. However, GAN-based methods (Choi et al. 2021; Han et al. 2018) often suffer from instability due to the min-max training objective, and they have limitations in preserving texture details and handling complex backgrounds. Recently, latent diffusion models (Ramesh et al. 2022; Zhang, Rao, and Agrawala 2023; Saharia et al. 2022) have made signif-|
|---|

###### Abstract

Latest advances have achieved (VTON) through localized garment diffusion models, significantly en shopping experience. However, ex neglect the need for merchants to showcase garments comprehensively, including flexible control faces, poses, and scenes. To address virtual dressing (VD) task focused itable human images with fixed g ditions. Meanwhile, we design a c ric index (CAMI) to evaluate the erated images and reference gar IMAGDressing-v1, which incorporates captures semantic features from from VAE. We present a hybrid a a frozen self-attention and a trainable grate garment features from the g denoising UNet, ensuring users c through text. IMAGDressing-v1 c extension plugins, such as ControlNet hance the diversity and controllability Furthermore, to address the lack teractive garment pairing (IGPair) 300,000 pairs of clothing and d lish a standard pipeline for data a ments demonstrate that our IMAGDressing-v1 achieves stateof-the-art human image synthesis controlled conditions. The code a at https://github.com/muzishen/IMAGDressing.

#### Introduction

Virtual dressing (VD) aims to a personalized clothing displays f given garments and optional f tive texts. This technology holds practical applications in e-commerce However, existing works primarily (VTON) (Han et al. 2018; Choi et Morelli et al. 2023; Xu et al. 2024b; for consumers, which involve giv man conditions, lacking flexibility and editability. While VD offers greater freedom and appeal, nificant challenges.

icant advances in VTON applications. These methods (Kim et al. 2024; Zeng et al. 2024; Xu et al. 2024b) better retain the texture information of the input garments through a

# Corresponding author

multi-step denoising process, ultimately generating images of specific individuals wearing the target clothing. Nevertheless, as illustrated in Figure 1 (a), VTON is essentially a local image inpainting task for consumer scenarios, requiring only the faithful preservation of given garment features. It overlooks the need for comprehensive clothing displays in merchant scenarios, lacking the ability to customize faces, poses, and scenes.

To address this, as illustrated in Figure 1 (b), we define a virtual dressing (VD) task aimed at generating freely editable human images with fixed garment and optional conditions, and then design a comprehensive affinity metric index (CAMI) to evaluate the consistency between generated images and reference garments. The differences between VTON and VD are as follows: (1) User Experience. VTON synthesizes images based on given clothing and specific person conditions, providing users with a static experience of partial inpainting. In contrast, VD centers on clothing and combines it with optional conditions to synthesize images, offering users a more interactive and flexible experience. (2) Application Scenarios. VTON is primarily used for personalized services for consumers, such

- as trying on clothes online to see if they suit them. In comparison, VD is mainly used by merchants on e-commerce platforms to showcase clothing, providing a comprehensive view of clothing ensembles. (3) Accuracy Requirements. VTON focuses on ensuring natural transitions and detailed handling between the clothing and the given model’s body. Building on these requirements, VD further emphasizes the uniformity and aesthetics of clothing displays under given clothing conditions and optional elements.

Furthermore, this paper presents IMAGDressing-v1, a latent diffusion model specifically designed for custom virtual dressing for merchants. IMAGDressing-v1 consists primarily of a trainable garment UNet and a denoising UNet. Since the VAE can nearly losslessly reconstruct images, the garment UNet is used to simultaneously capture semantic features from CLIP and texture features from the VAE. The denoising UNet introduces a hybrid attention module, comprising a frozen self-attention and a trainable cross-attention modules, to integrate clothing features from the clothing UNet into it. This integration allows users to control different scenes through text prompts. Moreover, IMAGDressingv1 can be combined with other extensions, such as ControlNet (Zhang, Rao, and Agrawala 2023) and IP-Adapter (Ye et al. 2023), to enhance the diversity and controllability of generated images. Lastly, to address the issue of data scarcity, we have collected and released the large-scale interactive garment pairing (IGPair) dataset, containing over 300,000 pairs of clothing and dressed images. The contributions of this paper are summarized as follows:

- • This paper introduces a new virtual dressing (VD) task for merchants and designs a comprehensive affinity measurement index (CAMI) to evaluate the consistency between generated images and reference garment.
- • We propose IMAGDressing-v1, which includes a garment UNet for extracting fine-grained clothing features and a denoising UNet with a hybrid attention module to

- balance clothing features with text prompt control.
- • IMAGDressing-v1 can be combined with other extensions, such as ControlNet and IP-Adapter, to enhance the diversity and controllability of generated images.
- • We collect and release a large-scale interactive garment pairing (IGPair) dataset, containing over 300,000 pairs, available for the community to explore and research.

#### Related Work

##### Virtual Try-On

Early virtual try-on (Dong et al. 2020; Jo and Park 2019; Lee et al. 2020; Liu et al. 2020; Choi et al. 2021) typically utilized generative adversarial networks (GANs) (Creswell et al. 2018) and a two-stage strategy. Initially, they would warp the clothing to the desired shape, then use a GANbased generator to merge the warped clothing onto the human model. For instance, VITON-HD (Choi et al. 2021) addresses issues of clothing-body occlusion and mismatch by performing warping and segmentation simultaneously. GPVTON (Xie et al. 2023) introduces local warping and global parsing to independently model the deformation of different clothing regions, aiming for a more accurate fit. To achieve precise clothing deformation, some methods (Han et al. 2019; Lee et al. 2022) estimate a dense flow map to guide the reshaping process. Additionally, some approaches (Ge et al.

- 2021; Issenhuth, Mary, and Calauzenes 2020) use normalization or distillation strategies to address the misalignment between the warped clothing and the human body. However, GAN-based methods face instability due to the min-max nature of their training objectives and have limitations in preserving texture details and handling complex backgrounds.

Recent research (Morelli et al. 2023; Gou et al. 2023; Zhu et al. 2023) have incorporated pre-trained diffusion models as priors for VTON tasks. For example, LADIVTON (Morelli et al. 2023) and DCI-VTON (Gou et al.

- 2023) explicitly warp clothes to align them with the human body, then use diffusion models to merge the clothes with the body. TryOnDiffusion (Zhu et al. 2023) proposed an architecture with two parallel UNets and demonstrated the capability of diffusion-based virtual try-on by training on large-scale datasets. Similarly, OOTDiffusion (Xu et al.
- 2024b) and IDM (Choi et al. 2024) utilize parallel UNets for garment feature extraction and enhance integration through self-attention. StableVITON (Kim et al. 2024) introduces a zero-initialized cross-attention block to inject intermediate features of the spatial encoder into the UNet decoder. While diffusion-based VTON methods can combine clothing with a fixed model, producing fine-grained static images, VTON is essentially a local image inpainting task tailored for consumer scenarios, merely needing to faithfully preserve the given clothing features. As previously mentioned, VTON overlooks the need for comprehensive garment presentation in commercial contexts and cannot customize faces, poses, and scenes.

Latent Diffusion Model

While latent diffusion models (LDMs) (Rombach et al.

- 2022a; Wang et al. 2024; Shen et al. 2024; Wang et al. 2024)

Dataset Public Caption # Garments # Pairs Resolution TryOnGAN ✗ ✗ 52,000 52,000 512 × 512 Revery AI ✗ ✗ 321,000 321,000 512 × 512 VITON-HD ✓ ✗ 13,679 13,679 1024 × 768 Dress Code ✓ ✗ 53,792 53,792 1024 × 768 IGPair (Ours) ✓ ✓ 86,873 324,857 > 2K × 2K

Table 1: Comparison between IGPair and the widely used datasets.

have been widely used for text-to-image (T2I) generation and editing tasks, the inaccuracy of natural language limits fine-grained image control. Various methods have been proposed to add conditional control to T2I diffusion models to address this. For example, ControlNet (Zhang, Rao, and Agrawala 2023) and T2I Adapter (Mou et al. 2024) introduce additional conditional encoding modules, such as edges, depth, and human poses, to control diffusion models and text prompts. IP-Adapter (Ye et al. 2023) conditions T2I diffusion models on high-level semantics of reference images, using both text and visual cues to guide image generation. Uni-ControlNet (Zhao et al. 2024) proposes a unified framework that flexibly and composably handles different conditional controls within a single model to reduce computational costs. MasaCtrl (Cao et al. 2023) achieves consistent image generation and complex non-rigid image editing through self-attention transformation without additional training costs. Similarly, InstructPix2Pix (Brooks, Holynski, and Efros 2023) retrains LDMs by adding extra input channels to the first convolutional layer to follow editing instructions. PCDMs (Shen et al. 2023) proposes proposes a multistage conditional diffusion model for pose guided character image generation. In this paper, we leverage the capabilities of frozen LDMs in text-to-image generation to achieve garment-centric image generation and editing.

#### IGPair Dataset

- ➤ Q1: What kind of data is suitable for VD task? We identify three critical requirements for an ideal virtual dressing dataset: (1) it should be publicly accessible for research purposes; (2) it should include high-resolution images of both garment and models wearing the clothing; (3) it should encompass a variety of scenes and styles, with detailed textual descriptions. As shown in Table 1, the proposed IGPair dataset not only meets all the aforementioned requirements but also provides six times the number of image pairs compared to the largest publicly available dataset, VITONHD (Choi et al. 2021), surpassing TryOnGAN (Lewis, Varadharajan, and Kemelmacher-Shlizerman 2021), Revery AI (Li et al. 2021), VITON-HD (Choi et al. 2021), and Dress Code (Morelli et al. 2022) datasets. Notably, IGPair includes multiple models for each clothing item. It is also the only dataset with a resolution exceeding 2K × 2K. Additionally, IGPair is the only publicly available dataset that includes textual descriptions, diverse scenes, and various styles.
- ➤ Q2: How is the IGPair dataset collected and annotated? All images are sourced from the internet and encompass a variety of clothing styles, including casual, formal, athletic, fashionable, and vintage, etc. Initially, we collected 500,000

A woman is standing against a brick wall, wearing black pants and a gray tank top

The model is wearing an orange tie-dye camisole top with denim shorts and white socks

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

A young man is standing in front of a brick wall, wearing a white t-shirt with a black graphic and green short pants

a man in a beige shirt and black pants

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Figure 2: Sample pairs from the IGPair dataset, including pose keypoints, dense poses, and human body segmentation masks. More sample refer to supplementary material.

garment images, each accompanied by 2 to 5 images of people wearing the clothing from different perspectives. We then use classifiers to differentiate between clothing and human and employ a human pose estimator to select complete and usable images of clothing on models. After this automated stage, we manually verified all images. We categorize the garment into 18 types, and the dataset consists of 324,857 image pairs. To further enrich our dataset, we use OpenPose (Cao et al. 2017) to extract 18 key points for each human figure, DensePose (G¨uler, Neverova, and Kokkinos 2018) to compute dense poses for each reference model, and SCHP (Li et al. 2020) to generate segmentation masks for body parts and clothing items. We utilize BLIP2-OPT6.7B (Li et al. 2023), INTERNLM-XCOMPOSER2-VL7B (Dong et al. 2024), LLaVA-V1.5-13B (Liu et al. 2024), and Qwen-VL-Chat (Bai et al. 2023) to generate captions for the images. All model images are anonymized. Samples of human models and clothing pairs from our dataset, along with the corresponding additional information, are shown in Figure 2. More detail refer to supplementary material.

➤ Q3: How to evaluate the consistency between the generated image and the garment? We propose a comprehensive affinity metric index (CAMI) for evaluating VD task, which includes the unspecified score (CAMI-U) and the specified score (CAMI-S). CAMI-U represents the score for image generation without specified pose, face, and text scenarios. In contrast, CAMI-S represents the score for image generation with the specified pose, face, and text scenarios. CAMI-U focuses on the clothing images’ structure Ss, texture St, and keypoints Sk. CAMI-S builds upon CAMI-U by adding pose matching degree Sp, facial similarity Sf, and text-image matching degree Sc.

SCAMI-U = Ss + St + Sk, (1) SCAMI-S = SCAMI-U + Sp + Sf + Sc. (2)

More detailed settings are to be provided in the supplementary material. Additionally, we also utilize MP-LPIPS (Chen et al. 2024), and ImageReward (Xu et al. 2024a) to evaluate the quality of the generated images.

|[Figure 81]<br><br>[Figure 82]<br><br>VAE Encoder<br><br>VAE Decoder<br><br>VAE Encoder<br><br>Image Encoder<br><br>| | |
|---|---|
| | |
<br><br>···<br><br>Projection<br><br>Text<br><br>Encoder<br><br>Garment Image<br><br>Noisy Image<br><br>|“A woman wearing red short sleeves and black pants against a white background.”| |
|---|---|
| | |
<br><br>[Figure 83]<br><br>[Figure 84]<br><br>Caption<br><br>···<br><br>··· ···<br><br>Denoised Image<br><br>| | | | | |
|---|---|---|---|---|
|KK|VV|QQ| | |
| | | || | | |
|---|---|---|
|QQ KK|VV| |
| |
<br><br>| | | |
|---|---|---|
|QQ KK|VV| |
<br><br>[Figure 85]<br><br>Latent Noise Latent Garment<br><br>[Figure 86]<br><br>|[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>Trainable Module<br><br>Frozen Module|
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>Self Attention Cross Attention Hybrid Attention|
|---|
<br><br>[Figure 91]<br><br>Garment UNet<br><br>[Figure 92]<br><br>Dressing UNet<br><br>|[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>Optional Conditions<br><br>ControlNet IP-Adapter| |
|---|---|
| | |
<br><br>Hybrid Attention|
|---|

- Figure 3: Illustration of the proposed IMAGDressing-v1 framework. It mainly consists of a trainable garment UNet and a frozen denoising UNet. The former extracts fine-grained garment features, while the latter balances these features with text prompts. IMAGDressing-v1 is compatible with other community modules, such as ControlNet and IP-Adapter.

#### Methodology

##### Preliminaries

Unlike other pixel-based diffusion models, latent diffusion models (LDMs) (Rombach et al. 2022a) aim to perform the denoising process in the latent space to reduce computational costs. An LDM typically comprises a variational autoencoder (VAE) (Kingma and Welling 2013), a CLIP text encoder (Radford et al. 2021), and a denoising UNet. The VAE transforms images into latent space representations and vice versa. Specifically, the VAE encoder E compresses the original image x into a latent representation z, i.e., z = E(x), while the VAE decoder D reconstructs the image x from the latent representation z, i.e., x = D(z). The CLIP text encoder converts text prompts into token embeddings c. During the diffusion process, Gaussian noise ϵ is added to the latent representation z over timestep t to produce zt, where t ∈ [0,T]. The denoising UNet then iteratively denoises zt during the denoising process. To learn such a denoising UNet ϵθ parameterized by θ, for each timestep t, the training objective usually adopts a mean square error loss LLDM, as follows,

t,ϵ∼N(0,I),c,t ∥ϵθ (zt,c,t) − ϵt∥2 , (3)

LLDM = Ez

where zt = √αtz0 + √1 − αtϵt is the noisy latent at timestep t and ϵt is the added noise. Here, z0 = E (x0) and x0 represents the real data with a text condition c.

During the sampling stage, the predicted noise is calculated using both the conditional model ϵθ(xt,c,t) and the unconditional model ϵθ(xt,t) via classifier-free guidance (Ho and Salimans 2022).

ϵˆθ(xt,c,t) = wϵθ(xt,c,t) + (1 − w)ϵθ(xt,t). (4) Here, w is the guidance scale used to adjust the influence of the condition c.

##### IMAGDressing-v1

As shown in Figure 3, the proposed IMAGDressing-v1 mainly consists of a trainable garment UNet, architecturally same Stable Diffusion V1.5 (SD v1.5) 1. The difference lies in the garment UNet’s ability to simultaneously capture garment semantic features from CLIP and texture features from VAE, since the VAE can nearly losslessly reconstruct images. The lower part is a frozen denoising UNet, similar to SD v1.5, used for denoising the latent image under conditions. Unlike SD v1.5, we replace all self-attention modules with hybrid attention modules to more easily integrate garment features from the garment UNet and leverage the existing text-to-image capabilities for scene control via text prompts. Additionally, IMAGDressing-v1 includes an image encoder and projection layer for encoding garment features, as well as a text encoder for encoding textual features.

1https://huggingface.co/runwayml/stable-diffusion-v1-5

Garment UNet. Extracting fine-grained garment features is crucial for maintaining the consistency of garment details in VD task. To achieve this, the proposed garment UNet simultaneously extracts semantic information and texture features as garment characteristics. Specifically, given a garment image X ∈ R3×H×W, we first convert it into a latent

space representation Zg ∈ R4×H8 ×W8 using a frozen VAE Encoder 2. Simultaneously, token embeddings are extracted from X using a frozen CLIP image encoder 3 and a trainable projection layer, where we utilize a Q-Former (Li et al. 2023) as the projection layer. Subsequently, the garment features from garment UNet interact thoroughly in the crossattention mechanism, similar to the interaction between text and image in the original T2I model. Finally, the garment UNet aligns parallelly with the denoising UNet, injecting fine-grained features into the denoising UNet through hybrid attention. It is important to note that the garment UNet is only used to encode the reference image. Therefore, no noise is added to the reference image, and only a single forward pass is performed during the diffusion process.

Hybrid Attention. For VD task, an ideal denoising UNet should possess two key capabilities: (1) maintaining the original editing and generation abilities, and (2) incorporating additional garment features. The former can be achieved by freezing the modules of the denoising UNet, while the latter is accomplished through the proposed hybrid attention modules. Consequently, the architecture of the denoising UNet in IMAGDressing-v1 is similar to that of the original text-to-image SD v1.5 model, with the main difference being that we replace all self-attention modules in the denoising UNet with hybrid attention modules. As shown in Figure 3, the hybrid attention module consists of a frozen self-attention module and a learnable cross-attention module. Here, the weights of the self-attention of hybrid attention module are initialized using the self-attention’s weights from SD v1.5. Assuming Zd and Cg represent the query features and the garment features output by the garment UNet

- at corresponding positions, the output of hybrid attention Zh can be defined as follows:

###### Q(K′)⊤

###### QK⊤

V ′

√

√

Zh = Softmax

+λSoftmax

###### ,

###### V

d

d

Self Attention

Cross Attention

(5) where λ ∈ [0,1.5] is a hyperparameter used to regulate the strength of garment conditions. Q = ZdWq, K = ZdWk, V = ZdWv, K′ = CgWk′ , and V′ = CgWv′ . Here, Wq,Wk,Wv,W′k, and W′v are the weight matrices of the trainable linear projection layers. Noted that we share a query matrix Q for self attention and cross attention. In the hybrid attention module, the self-attention is frozen while the cross-attention is trainable. In other words, in Eq.5, only W′k and W′v are learnable. This approach allows us to retain the generative capabilities of the original

- 2https://huggingface.co/stabilityai/sd-vae-ft-mse
- 3https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-

b79K

|Method| |ImageReward (↑) MP-LPIPS (↓) CAMI-U (↑) CAMI-S (↑)<br><br>|
|---|---|---|
|Blip-Diffusion Versatile Diffusion IP-Adapter MagicClothing| |-2.224 0.1824 1.051 -<br>-2.055 0.4321 1.253 -<br>-2.267 0.4093 1.381 -<br>-0.164 0.1499 1.655 2.692<br><br><br>|
|Ours| |-0.095 0.1466 1.753 2.719|

Table 2: Quantitative comparison of the IMAGDressing-v1 with several state-of-the-art methods.

T2I model, such as scene generation.

Training and Inference. During training stage, we keep the parameters of the basic modules in the denoising UNet unchanged and only optimize the remaining modules. Let Ct represent the text condition, then the loss function LLDM is as follows,

t,ϵ∼N(0,I),Ct,Cg,t ∥ϵθ (zt,Ct,Cg,t) − ϵt∥2 ,

LLDM = Ez

(6) In the inference stage, we also use classifier-free guidance according to Eq. 7.

ϵˆθ(xt,Ct,Cg,t) = wϵθ(xt,Ct,Cg,t) + (1 − w)ϵθ(xt,t).

(7) ➤ Q4: How support customized generation? As shown in Figure 3, the weights of the basic modules are frozen in the denoising UNet, making the garment UNet essentially an adapter module compatible with other community adapters for customized face and pose generation. For instance, to generate images of people in a given outfit and consistent pose, IMAGDressing-v1 can be combined with ControlNetOpenpose. To generate specific individuals wearing specified clothing, IMAGDressing-v1 can be integrated with the IP-Adapter. Furthermore, if both pose and face need to be specified simultaneously, IMAGDressing-v1 can be used in conjunction with both ControlNet-Openpose and IPAdapter. Additionally, for VTON task, IMAGDressing-v1 also can be combined with ControlNet-Inpaint.

#### Experiments

##### Implementation Details

In our experiments, we initialize the weights of our garment UNet by inheriting the pre-trained weights of the UNet in Stable Diffusion v1.5 (Rombach et al. 2022b), and finetune its weight. Our model is trained on the paired images from the IGPair dataset at the resolution of 512 × 640. We adopt the AdamW optimizer with a fixed learning rate of 5e-5. The model is trained for 200,000 steps on 10 NVIDIA RTX3090 GPUs with a batch size of 5. At the inference stage, the images are generated with the UniPC sampler for 50 sampling steps and set the guidance scale w to 7.0. Please refer to the supplementary material for more details.

##### Main Comparisons

We compare our IMAGDressing-v1 with four state of- theart (SOTA) methods: Blip-Diffusion (Li, Li, and Hoi 2023), Versatile Diffusion (Xu et al. 2023), Versatile Diffusion (Xu et al. 2023), and MagicClothing (Chen et al. 2024).

Blip-Diffusion Versatile Diffusion IP-Adapter MagicClothing IMAGDressing-v1

MagicClothing IMAGDressing-v1

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

“A Beautiful Woman”

+ Pose

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

“A young woman”

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

“A black woman”

+ Face

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

“A beautiful woman on the beach”

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

“A beautiful lady standing on the lawn”

[Figure 147]

+ Pose&Face

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

“A man in the mall”

[Figure 158]

(a) Unspecific Conditions

###### (b) Specific Conditions

- Figure 4: Qualitative comparison with other SOTA methods under both unspecific and specific conditions, including BLIP-Diffusion (Li, Li, and Hoi 2023), Versatile Diffusion (Xu et al. 2023), IP-Adapter (Ye et al. 2023), and MagicClothing (Chen et al. 2024). Method ImageReward (↑) MP-LPIPS (↓) CAMI-U (↑) CAMI-S (↑)

- A0 (Base) -0.245 0.1537 1.575 2.578

- A1 (Base + IEB) -0.178 0.1504 1.637 2.625

- A2 (Base + IEB + HA) -0.095 0.1466 1.753 2.719

Table 3: Quantitative results for different Settings. IEB and HA denote the image encoder branch and hybrid attention.

- A0

（Base）

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

| |
|---|

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

| |
|---|

[Figure 172]

| |
|---|

[Figure 173]

| |
|---|

[Figure 174]

| |
|---|

- A1

(Base + IEB)

- A2

Quantitative Results. As shown in Table 2, since BlipDiffusion (Li, Li, and Hoi 2023), Versatile Diffusion (Xu et al. 2023), and IP-Adapter (Ye et al. 2023) are not specifically designed VD models, they struggle to extract finegrained garment features and generate character images that precisely match the text, pose, and garment attributes. This results in suboptimal performance across multiple metrics. Additionally, these models are incompatible with several plugins, making it impossible to compute the CAMI-S metric. Compared to MagicClothing (Chen et al. 2024), our IMAGDressing-v1 captures more detailed garment features through its image encoder branch and employs a hybrid attention mechanism. This mechanism integrates additional garment features while retaining the original text editing and generation capabilities. As a result, IMAGDressingv1 demonstrates superior performance, outperforming other SOTA methods across all evaluation metrics.

(Base+IEB+HA)

Figure 5: Ablation study of each component.

Diffusion (Li, Li, and Hoi 2023) and Versatile Diffusion (Xu et al. 2023) fail to faithfully reproduce garment textures. Although IP-Adapter maintains the overall appearance of the garments, it does not preserve the details well and, more importantly, does not follow the text prompts accurately. MagicClothing aligns closely with the text conditions; however, it struggles to retain the overall appearance and details of the garments, such as printed text or colors. In contrast,

Qualitative Results. Figure 4 illustrates the qualitative results of IMAGDressing-v1 compared to SOTA methods, including unspecific and specific condition generations. In Figure 4(a), under unspecific conditions, BLIP-

###### =0.1 =0.2 =0.3 =0.4 =0.5 =0.6 =0.7 =0.8 =0.9 =1.0 =1.1 =1.2 =1.3 =1.4 =1.5

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

Figure 6: Example results with different garment strength λ.

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

Figure 7: Examples of plug-in results of our IMAGDressing-v1 combined with ControlNet-Inpaint for virtual try-on.

Hyper-parameter λ. In Figure 6 demonstrates the effects of the hyper-parameter λ on generated samples with a fixed random seed. As λ increases to 1.0, the garment in the generated character becomes more similar to the input garment. A smaller λ ensures the generated results adhere more closely to the text prompts, while a larger λ biases the results towards the input garment. This indicates that λ effectively balances original editing and generation capabilities with additional garment features. Consequently, we empirically set λ to 1.0 in our experiments.

IMAGDressing-v1 not only adheres to the text prompts but also preserves fine-grained garment details, demonstrating superior performance in VD tasks. Additionally, our method supports customized text prompt scenarios, as shown in the last three rows of Figure 4 (a). Furthermore, Figure 4 (b) illustrates the qualitative results under specific conditions. We observe that IMAGDressing-v1 significantly outperforms MagicClothing in scenarios involving given poses, faces, or both. The results generated by IMAGDressing-v1 exhibit superior texture details and a more realistic appearance. This demonstrates the enhanced compatibility of IMAGDressingv1 with community adapters, which enhances the generated images’ diversity and controllability.

Potential application. Figure 7 illustrates a potential application of IMAGDressing-v1 in virtual try-on (VTON). By combining IMAGDressing-v1 with ControlNet-Inpaint and masking the garment area, we achieve VTON. The results demonstrate that IMAGDressing-v1 can achieve highfidelity VTON, showcasing significant potential.

##### Ablation Studies

Effectiveness of each component. Table 3 presents an ablation study to validate the effectiveness of the proposed image encoder branch (IEB) and hybrid attention (HA) module. Here, A0 (Base) denotes the setting without IEB and HA. We observe that A1, which uses IEB, shows improvements across all metrics, indicating that IEB effectively captures semantic garment features. Furthermore, A2 surpasses A1, demonstrating that the combination of IEB and HA further enhances quantitative results. Additionally, Figure 5 provides qualitative comparisons. We notice that A0 fails to adequately capture garment features in images with complex textures (2nd row). Although IEB (A1) partially addresses this issue, directly injecting IEB into the denoising UNet can lead to conflicts with the main model’s features, resulting in obscured garment details (3rd). Therefore, the HA module (A2) improves image fidelity by adjusting the intensity of garment details within the garment UNet (4th row), aligning with our quantitative results.

#### Conclusion

While recent advancements in VTON using latent diffusion models have enhanced the online shopping experience, they fall short of allowing merchants to showcase garments comprehensively with flexible control over faces, poses, and scenes. To bridge this gap, we introduce the virtual dressing (VD) task, designed to generate editable human images with fixed garments under optional conditions. Our proposed IMAGDressing-v1 employs a garment UNet and a hybrid attention module to integrate garment features, enabling scene control through text. It supports plugins like ControlNet and IP-Adapter for greater diversity and controllability. Additionally, we release the IGPair dataset with over 300,000 pairs of clothing and dressed images, providing a robust data assembly pipeline. Extensive experiments validate that IMAGDressing-v1 achieves state-of-the-art performance in controlled human image synthesis.

#### References

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023. Qwen-VL: A Versatile VisionLanguage Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966.

Brooks, T.; Holynski, A.; and Efros, A. A. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18392–18402.

Cao, M.; Wang, X.; Qi, Z.; Shan, Y.; Qie, X.; and Zheng,

- Y. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 22560–22570. Cao, Z.; Simon, T.; Wei, S.-E.; and Sheikh, Y. 2017. Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition, 7291–7299. Chen, W.; Gu, T.; Xu, Y.; and Chen, C. 2024. Magic Clothing: Controllable Garment-Driven Image Synthesis. CoRR, abs/2404.09512. Choi, S.; Park, S.; Lee, M.; and Choo, J. 2021. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14131–14140. Choi, Y.; Kwak, S.; Lee, K.; Choi, H.; and Shin, J. 2024. Improving diffusion models for virtual try-on. arXiv preprint arXiv:2403.05139. Creswell, A.; White, T.; Dumoulin, V.; Arulkumaran, K.; Sengupta, B.; and Bharath, A. A. 2018. Generative adversarial networks: An overview. IEEE signal processing magazine, 35(1): 53–65. Dong, H.; Liang, X.; Zhang, Y.; Zhang, X.; Shen, X.; Xie,
- Z.; Wu, B.; and Yin, J. 2020. Fashion editing with adversarial parsing learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8120– 8128. Dong, X.; Zhang, P.; Zang, Y.; Cao, Y.; Wang, B.; Ouyang, L.; Wei, X.; Zhang, S.; Duan, H.; Cao, M.; Zhang, W.; Li, Y.; Yan, H.; Gao, Y.; Zhang, X.; Li, W.; Li, J.; Chen, K.; He, C.; Zhang, X.; Qiao, Y.; Lin, D.; and Wang, J. 2024. InternLMXComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model. arXiv preprint arXiv:2401.16420. Ge, Y.; Song, Y.; Zhang, R.; Ge, C.; Liu, W.; and Luo, P.

- 2021. Parser-free virtual try-on via distilling appearance flows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8485–8493. Gou, J.; Sun, S.; Zhang, J.; Si, J.; Qian, C.; and Zhang, L.

2023. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, 7599– 7607.

G¨uler, R. A.; Neverova, N.; and Kokkinos, I. 2018. Densepose: Dense human pose estimation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, 7297–7306.

Han, X.; Hu, X.; Huang, W.; and Scott, M. R. 2019. Clothflow: A flow-based model for clothed person generation. In Proceedings of the IEEE/CVF international conference on computer vision, 10471–10480.

Han, X.; Wu, Z.; Wu, Z.; Yu, R.; and Davis, L. S. 2018. Viton: An image-based virtual try-on network. In Proceedings of the IEEE conference on computer vision and pattern recognition, 7543–7552.

Ho, J.; and Salimans, T. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.

Issenhuth, T.; Mary, J.; and Calauzenes, C. 2020. Do not mask what you do not need to mask: a parser-free virtual try-on. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX 16, 619–635. Springer.

Jo, Y.; and Park, J. 2019. Sc-fegan: Face editing generative adversarial network with user’s sketch and color. In Proceedings of the IEEE/CVF international conference on computer vision, 1745–1753.

Kim, J.; Gu, G.; Park, M.; Park, S.; and Choo, J. 2024. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8176–8185.

Kingma, D. P.; and Welling, M. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

Lee, C.-H.; Liu, Z.; Wu, L.; and Luo, P. 2020. Maskgan: Towards diverse and interactive facial image manipulation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 5549–5558.

Lee, S.; Gu, G.; Park, S.; Choi, S.; and Choo, J. 2022. Highresolution virtual try-on with misalignment and occlusionhandled conditions. In European Conference on Computer Vision, 204–219. Springer.

Lewis, K. M.; Varadharajan, S.; and KemelmacherShlizerman, I. 2021. Tryongan: Body-aware try-on via layered interpolation. ACM Transactions on Graphics (TOG), 40(4): 1–10.

Li, D.; Li, J.; and Hoi, S. C. H. 2023. BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-toImage Generation and Editing. In NeurIPS.

- Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, 19730–19742. PMLR.
- Li, K.; Chong, M. J.; Zhang, J.; and Liu, J. 2021. Toward accurate and realistic outfits visualization with attention to details. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 15546–15555.

Li, P.; Xu, Y.; Wei, Y.; and Yang, Y. 2020. Self-correction for human parsing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(6): 3260–3271.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26296–26306.

Liu, J.; Song, X.; Chen, Z.; and Ma, J. 2020. MGCM: Multi-modal generative compatibility modeling for clothing matching. Neurocomputing, 414: 215–224.

Morelli, D.; Baldrati, A.; Cartella, G.; Cornia, M.; Bertini, M.; and Cucchiara, R. 2023. LaDI-VTON: latent diffusion textual-inversion enhanced virtual try-on. In Proceedings of the 31st ACM International Conference on Multimedia, 8580–8589.

Morelli, D.; Fincato, M.; Cornia, M.; Landi, F.; Cesari, F.; and Cucchiara, R. 2022. Dress code: High-resolution multicategory virtual try-on. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2231–2235.

Mou, C.; Wang, X.; Xie, L.; Wu, Y.; Zhang, J.; Qi, Z.; and Shan, Y. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 4296–4304.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PMLR.

Ramesh, A.; Dhariwal, P.; Nichol, A.; Chu, C.; and Chen, M.

- 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2): 3. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Om-

- mer, B. 2022a. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695. Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Om-
- mer, B. 2022b. High-Resolution Image Synthesis with Latent Diffusion Models. In CVPR, 10674–10685. IEEE.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35: 36479–36494.

Shen, F.; Ye, H.; Liu, S.; Zhang, J.; Wang, C.; Han, X.; and Yang, W. 2024. Boosting Consistency in Story Visualization with Rich-Contextual Conditional Diffusion Models. arXiv preprint arXiv:2407.02482.

Shen, F.; Ye, H.; Zhang, J.; Wang, C.; Han, X.; and Wei, Y.

- 2023. Advancing Pose-Guided Image Synthesis with Progressive Conditional Diffusion Models. In The Twelfth International Conference on Learning Representations.

Wang, C.; Tian, K.; Guan, Y.; Zhang, J.; Jiang, Z.; Shen, F.; Han, X.; Gu, Q.; and Yang, W. 2024. Ensembling Diffusion Models via Adaptive Feature Aggregation. arXiv preprint arXiv:2405.17082.

Xie, Z.; Huang, Z.; Dong, X.; Zhao, F.; Dong, H.; Zhang, X.; Zhu, F.; and Liang, X. 2023. Gp-vton: Towards general purpose virtual try-on via collaborative local-flow globalparsing learning. In Proceedings of the IEEE/CVF Confer-

ence on Computer Vision and Pattern Recognition, 23550– 23559.

Xu, J.; Liu, X.; Wu, Y.; Tong, Y.; Li, Q.; Ding, M.; Tang, J.; and Dong, Y. 2024a. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36.

- Xu, X.; Wang, Z.; Zhang, E. J.; Wang, K.; and Shi, H. 2023. Versatile Diffusion: Text, Images and Variations All in One Diffusion Model. In ICCV, 7720–7731. IEEE.
- Xu, Y.; Gu, T.; Chen, W.; and Chen, C. 2024b. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779.

Yang, X.; Ding, C.; Hong, Z.; Huang, J.; Tao, J.; and Xu, X. 2024. Texture-Preserving Diffusion Models for HighFidelity Virtual Try-On. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7017–7026.

Ye, H.; Zhang, J.; Liu, S.; Han, X.; and Yang, W. 2023. IPAdapter: Text Compatible Image Prompt Adapter for Textto-Image Diffusion Models. CoRR, abs/2308.06721.

Zeng, J.; Song, D.; Nie, W.; Tian, H.; Wang, T.; and Liu, A.A. 2024. CAT-DM: Controllable Accelerated Virtual Tryon with Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8372–8382.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3836–3847.

Zhao, S.; Chen, D.; Chen, Y.-C.; Bao, J.; Hao, S.; Yuan, L.; and Wong, K.-Y. K. 2024. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36.

Zhu, L.; Yang, D.; Zhu, T.; Reda, F.; Chan, W.; Saharia, C.; Norouzi, M.; and Kemelmacher-Shlizerman, I. 2023. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4606–4615.

安踏安踏 ((ANTA)(ANTA)) 鸿星尔克鸿星尔克 ((ERKE)(ERKE)) 特步特步 ((XTEP)(XTEP)) 李宁李宁 ((LI-NING)(LI-NING)) 安踏安踏 ((ANTA)(ANTA)) 鸿星尔克鸿星尔克 ((ERKE)(ERKE)) 特步特步 ((XTEP)(XTEP)) 李宁李宁 ((LI-NING)(LI-NING))

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

[Figure 278]

Figure 8: More results of IMAGDressing-v1 synthesizing person images given garment with logos and optional faces.

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

Figure 9: More results of IMAGDressing-v1 generating person images with complex logos (such as text and dense letters) on garment and optional faces.

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

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

|"In the Park" "In the Library" "At the Elevator" "In the Mall" "On the Beach" "In the Cafeteria"|
|---|

Garment

Text Prompt

Figure 10: More results of IMAGDressing-v1 synthesizing person images based on clothing and different text prompts.

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

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

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

Figure 11: More results of IMAGDressing-v1 synthesizing person images given garment with logos, and optional faces and pose.

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

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

A smiling woman

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

A young girl on the beach A beautiful woman in the garden

Figure 12: More results of IMAGDressing-v1 synthesizing cartoon images based on clothing and different text prompts.

