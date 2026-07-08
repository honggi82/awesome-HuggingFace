## FashionComposer: Compositional Fashion Image Generation

Sihui Ji1,2,∗ Yiyang Wang1 Xi Chen1 Xiaogang Xu3 Hao Luo2,4 Hengshuang Zhao1,† 1The University of Hong Kong 2DAMO Academy, Alibaba Group 3Zhejiang University 4Hupan Lab

https://SihuiJi.github.io/FashionComposer-Page

Compositional Fashion Image Generation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

# arXiv:2412.14168v2[cs.CV]19Dec2024

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

Parametric Control for Human Figure, Pose, Views, etc.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

…

A woman is wearing a black tank top and a blue skirt, with sunglasses.

[Figure 30]

Virtual Try-on with Multiple Garments Human Album Generation with Consistent ID

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Figure 1. Demonstration for the applications of FashionComposer. FashionComposer takes different kinds of conditions (e.g., garment image, face image, parametric human model) equally as “assets” to composite diverse and realistic fashion images. Thus supporting various fashion-related applications like controllable model image generation, virtual try-on, human album generation, etc.

### Abstract

robust compositional capabilities. To accommodate multiple reference images (garments and faces) seamlessly, we organize these references in a single image as an “asset library” and employ a reference UNet to extract appearance features. To inject the appearance features into the correct pixels in the generated result, we propose subject-binding attention. It binds the appearance features from different “assets” with the corresponding text features. In this way, the model could understand each asset according to their semantics, supporting arbitrary numbers and types of reference images. As a comprehensive solution, FashionComposer also supports many other applications like human album generation, diverse virtual try-on tasks, etc.

We present FashionComposer for compositional fashion image generation. Unlike previous methods, FashionComposer is highly flexible. It takes multi-modal input (i.e., text prompt, parametric human model, garment image, and face image) and supports personalizing the appearance, pose, and figure of the human and assigning multiple garments in one pass. To achieve this, we first develop a universal framework capable of handling diverse input modalities. We construct scaled training data to enhance the model’s

*Work done when interning at DAMO Academy. † Corresponding author.

### 1. Introduction

In the e-commercial age, the fashion industry is inundated with large amounts of in-shop garment images. To better attract customers, they hire models to showcase how the clothes look on individuals. Virtual try-on technology, designed to generate an image of the specific human wearing provided garments, has become increasingly popular. Existing virtual try-on methods [8, 15, 29, 32] usually allow for trying on only a single garment and have significant limitations in terms of flexibility. For example, the poses of the synthesized images are fixed according to the human image, which imposes strict constraints on the diversity of body shapes and postures in try-on images. Furthermore, these methods are usually conditional on only one garment, thus are unable to try on outfits.

Facing the aforementioned challenges, we introduce FashionComposer, a flexible fashion image generator. Unlike existing strategies, the core feature of FashionComposer is compositionality, manifested in two aspects. First, it involves multi-modal inputs, including a language description for the target fashion image, a parametric human model that controls the human figure and posture, and garment images (face as optional) for reference. Second, it entails the composition of multiple visual assets. It allows users to drag their desired garment components and human faces into an “asset library” to customize the generation.

To achieve such compositionality, we design a diffusionbased framework with multi-modal inputs. We employ the Skinned Multi-Person Linear model (SMPL) [18] to control the human figure and posture and leverage a series of offthe-shelf models [2, 7, 11] to prepare multi-modal training data (e.g., the masks and language descriptions for different parts). To handle multiple visual assets (e,g., different garments, face, shoes) in one pass without increasing the computation burden, we propose to arrange all the referring elements in one image as an “asset library” and leverage one reference UNet [13] to extract the features. Considering the asset image might contain different numbers of garment images with different positions and sizes, it is hard for the model to understand the mapping relation between the asset image and the generation result. To solve this issue, we propose subject-binding attention, which binds the appearance features of the garments/faces with the corresponding text embeddings. Therefore, the features of each asset could be mapped to the correct pixels according to their semantics.

Besides pursuing composability, we further develop correspondence-aware attention and latent code alignment to support FashionComposer in generating an album of images with a consistent identity. As an all-round fashion image generator, FashionComposer also supports traditional virtual try-on and multi-garment virtual try-on tasks. In general, FashionComposer demonstrates an extraordinary capacity for multi-modal fashion generation. Our contri-

butions could be summarized in three folds.

- • We propose FashionComposer, which takes multi-modal input conditions and supports multiple fashion-related tasks in a unified framework.
- • We design subject-binding attention, which enables the composition of multiple visual assets in an effective and extensible manner.
- • We introduce techniques like correspondence-aware attention and latent code alignment to support multiple practical applications like album image generation, multigarment try-on, etc.

### 2. Related Work

Standard virtual try-on. The goal of the standard virtual try-on task is to generate images where a garment from an in-shop image is seamlessly integrated onto a reference person while preserving meticulous details. Warping-based methods [12, 28, 32] depend on garment deformation followed by fitting on the individual to generate the composite images, suffering from alignment issues. Diffusion-based methods can directly dress a target person in a specified garment without warping based on the strong ability of generative models. For maintenance of identity, LaDIVTON [21] uses CLIP [22] as image encoder for feature extraction while TryOnDiffusion [37] designs a two-UNets architecture, and OOTDiffusion [33] leverages reference UNet for maintaining details of clothes. However, a notable limitation of these methods is their restricted capacity for customization. Specifically, they are constrained to transferring only a single garment onto the reference person, hindering flexibility in the try-on process.

Multi-object customization. Discerning different objects while preserving their identities is a challenging task. Textual Inversion [9] and DreamBooth [24] propose to tune the text representation space to add new concepts for various objects in the text-to-image diffusion models. Cones [17] finds the neurons in the neural network of each corresponding referred object. Emu2 [27] is a multi-modal autoregressive model that integrates the input text prompt and different reference objects in a multi-modal sequence and generates images accordingly. Collage Diffusion [25] generates multiple objects by separating each object into different layers, which can transform a user-provided object layout into a high-resolution image. FastComposer [31] enables multi-person composition by binding the image embeddings of different people to the corresponding word embeddings, which can synthesize personalized, multi-person images without additional tuning. Nevertheless, these techniques either demand costly fine-tuning [9, 17, 24, 25], or struggle to keep detailed fidelity [9, 24, 25, 27, 31]. In contrast, FashionComposer can generate multi-garment fashion images without tuning and keep detailed fidelity for each garment with excellent superiority.

[Figure 40]

###### Asset Image

Subject-binding Attention (SBA)

Reference UNet

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

|SBA| |
|---|---|
|ConcatK,V| |

|CA|
|---|

|SBA| |
|---|---|
| | |
|ConcatK,V| |

|CA|
|---|

|SBA| |
|---|---|
| | |
|ConcatK,V| |

|CA|
|---|

SBA

SBA

SBA

…

…

CA SA

CA SA

CA SA

[Figure 47]

or

ConcatK,V

ConcatK,V

ConcatK,V

|A slim woman with short brown hair wears a green shirt and a yellow skirt.| |
|---|---|
| | |

Text Encoder

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

###### Try-on Task

|CA|
|---|

|CA|
|---|

|SA|
|---|

|CA|
|---|

|SA|
|---|

|SA|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

… …

VAE Decoder

CA

CA

CA

C

or

C

Model Image Denoising UNet

Pose Map Noise

Try-on Result

Generation Result

Agnostic Mask

Figure 2. Overall pipeline of FashionComposer. FashionComposer takes garments composition and optional face, text prompt, and a densepose map projected from SMPL as inputs. The text prompt is encoded and fused with UNets through cross-attention and subjectbinding attention, while the garment features are extracted and injected for denoising through Feature Injection Attention.

### 3. Method

with the distinction that the self-attention modules are substituted with subject-binding attention.

#### 3.1. Preliminaries

FashionComposer leverages different combinations of input to support different tasks, naturally formulating fashion image generation and conventional virtual try-on in the same framework. The only change we make to fit the original inpainting-based virtual try-on setting is to concatenate a 4-channel cloth-agnostic person image embedding encoded by VAE and a 1-channel binary cloth-agnostic mask along with noise rather than using a densepose map. The convolution-in channel of UNet is converted to 9 accordingly. FashionComposer achieves the leading performance for compositional fashion image generation and our finetuned virtual try-on model achieves superior qualitative and quantitative results on standard try-on benchmarks while supporting multiple garment try-on in one pass.

Stable Diffusion. We use Stable Diffusion (SD) v1.5 [23] as our backbone model. The UNet-based SD model comprises a variational autoencoder(VAE), a denoising UNet, and a text encoder. The VAE encoder E encodes the image x into a compressed latent representation z. The latent z is then perturbed by a Gaussian noise ϵ in the forward diffusion process, which is predicted by the UNet parameterized by θ for the backward diffusion process. The text prompt P is encoded by the text encoder T into text embeddings T (P), which is then injected into the UNet through cross-attention. During training, the network is optimized to minimize the loss function formulated as:

L = Et∼T,ϵ∼N(0,1),z∼E(x)[||ϵ − ϵθ(zt,t,T (P))||22], (1)

#### 3.3. Composition for Multi-modal Conditions

where the timestep t is sampled by a timestep scheduler T, and zt is the latent representation at the timestep t. During inference, the initial noise zT is sampled from N(0,1) and iteratively denoised by the denoising UNet that predicts the noise at each timestep conditioned on the text prompt P. The final latent code z0 is the input of the VAE decoder D to generate the final image x0 = D(z0).

Multi-modal input conditions. Unlike existing virtual tryon methods relying on only image inputs, FashionComposer seamlessly integrates various multi-modal conditions (i.e., text prompt, SMPL parameters, reference garments and optional face) in the fashion image generation. We condition the generator on different multi-modal conditions simultaneously using a unified framework. For the visual assets composition, we employ the reference UNet to extract its multi-level features, which are then involved in the denoising process by attention sharing. As for the text prompt, we encode it into text embeddings using CLIP’s text encoder. We opt to condition them on the cross-attention modules of both UNets for appearance guidance. Additionally, we opt for a lighter approach in handling the SMPL , as we

#### 3.2. Framework Overview

The overall pipeline of FashionComposer is demonstrated in Fig. 2. The densepose map projected from SMPL is initially concatenated with the noise, followed by the denoising UNet performing the denoising process. The reference UNet follows the same framework as the denoising UNet,

aim to avoid introducing overly intricate conditions into the generation process, which would compromise the generation quality. Therefore, we choose to simply concatenate the 2-channel densepose map projected from SMPL with the noisy latent along the feature channel and directly input them into the denoising UNet.

Adopting this approach, we effectively condition FashionComposer on diverse multi-modal inputs. FashionComposer demonstrates extraordinary abilities to follow different conditions and synthesize fashion images accordingly.

Multi-modal data construction. The ideal training samples consist of all in-shop garments and the dressed human images, along with text descriptions, and an index indicating the correlation between phrases in the text and the corresponding garments. However, the raw datasets we utilize [12, 16, 20] contain only a single corresponding in-shop garment image for each target image.

To address this problem, we use the Mask2FormerParsing [7, 35] to detect the human parsing maps of the person images. Then for our chosen garments without a corresponding in-shop garment image, we mask them out as the selected components of the garment’s composition. Subsequently, we construct the composition of in-shop garment, masked-out garment and face by randomly placing

- them without overlapping on a white background. We leverage Qwen-VL-Chat [2] to caption the target images. For each text prompt, we identify the phrases describing different components and classify them into corresponding categories by querying Qwen-14B-Chat [1] about the phrases related to each component in the sentence. The final joint multi-modal dataset includes 165k samples.

#### 3.4. Composition of Multiple Visual Assets

Preserving reference garments’ global features and intricate details is crucial for ensuring the fidelity of the synthesized images. Reference UNet [3, 5, 13, 36] is proven effective for preserving the fine details of the reference image. In this work, we follow this structure to extract the reference appearance features and design subject-binding attention to enable multiple references in one pass.

Basics for reference UNet. Specifically, the reference UNet follows the same framework as the denoising UNet initialized with the parameters from SD V1.5. The asset composition undergoes encoding by the reference UNet, which converts it into multi-level features across the different blocks of the UNet. Specifically, we extract the key and value tokens from the self-attention modules of the reference UNet, which can be formulated as kref,vref ∈ R(h×w)×d. Then we concatenate these tokens with their corresponding key and value tokens in the matching block of the denoising UNet, which can be formulated as [kden,kref],[vden,vref] ∈ R(2×h×w)×d (the operator [x1,x2,...] stands for concatenation). Consequently,

the self-attention operations in the denoising UNet are replaced with a newly defined attention formulated as:

qden · [kden,kref]T √

) · [vden,vref]. (2)

softmax(

d

Note that different from [13], we keep textual input for the cross-attention module. While reference UNet utilizes the existing feature modeling capacities of the pre-trained SD for accurately maintaining details, the model still retains the ability to align with the text prompt which facilitates the generation process under multiple guidance.

Subject-binding attention. A limitation of standard reference UNet is that it cannot deal with multiple reference images. A direct approach is equipping each reference image with a specific reference UNet. However, this approach is computationally expensive, especially for cases with large numbers of references. Facing this challenge, we propose subject-binding attention. Specifically, we support users to put all their reference elements (with arbitrary location, size, or number) in one asset image. Then, we let one reference UNet to extract the features for this asset image. Afterward, to make the model understand the “all-in-one” features, we bind each pixel with their corresponding text descriptions according to the following steps:

First, we find the corresponding tokens of each visual asset in different feature maps. Suppose we select the UNet block l here; then, the size of the feature map in this block is determined, denoted by h′ × w′. Then, given an asset’s composition G = {a1,a2,...an}, for each asset component ai in the asset’s composition, we directly downsample it to derive its corresponding area in the feature map. Since the key/value tokens in the self-attention module can be seen as feature maps flattened into sequences, denoted by K ∈ R(h

′×w′)×d, then the above downsample operation determines ni corresponding key/value points Ki = {k1,k2...kn

′×w′)×d,V ∈ R(h

i} for asset component ai in the self-attention module.

i},Vi = {v1,v2...vn

Second, we bind the selected points (tokens) with their corresponding text representations. Take the key points as an example. Each key point kj ∈ Rd is an element of the key token K. Now, we proceed to bind all key points in Ki with the text embeddings corresponding to asset ai. The text prompt is denoted by P = {w1,w2,...,ws}, then for asset ai it might have a corresponding phrase {wi

p} like “a red shirt”, whose text embedding is denoted as Pi ∈ Rl×c. We input this text embedding into an MLP, with each UNet block having one, and then add the output with each kj in Ki, formulated as:

,...wi

1

kj′ = MLPl(Pi) + kj. (3) Subsequently, we replace each original key point kj ∈

Ki in the key token with the subject-bind embedding kj′.

This procedure is repeated for each asset in the asset’s composition, and same for the value tokens. Fig. 2 illustrates the concrete procedure of our approach which provides the model with awareness of the semantic meaning of conditioned asset to better maintain details without confusion.

#### 3.5. Consistent Human Image Generation

Another feature of FashionComposer is generating a human album with a consistent ID. Specifically, we propose correspondence-aware attention and latent code alignment. Correspondence-aware attention. Inspired by crossframe attention used in text-to-video area [14], we first try to directly replace all key/value tokens of self-attention modules of the 2nd to Nth images with the key/value tokens of the 1st image. Although the synthesized human appearances demonstrate excellent consistency, the fidelity and quality of the garments are compromised. We attribute this to the excessive amount of information from the first image, which negatively impacts the subsequent images. Therefore, we propose correspondence-aware attention to leverage the information from the first image and the current image. Specifically, we extract the UV map coordinates for all SMPL parameters and map them on the densepose maps, where identical UV coordinates (u,v) denote the same human area across different SMPL settings. Then, we substitute the key/value tokens of the 2nd to the Nth image with those of the 1st image only if they share the same (u,v) coordinates. While such a design effectively generates an album of high-fidelity images, it exhibits poorer consistency in human appearances compared to cross-frame attention.

Latent code alignment. To enhance the consistency in human appearances, we propose latent code alignment to capitalize on the high consistency provided by cross-frame attention and the high fidelity offered by correspondenceaware attention. Specifically, we first generate an album of images using cross-frame attention and preserve their denoised latent codes {z01,z02,...z0N} in the final denoising timestep. Next, we utilize the densepose map to extract face masks representing facial areas across different images. We

- then apply these masks to remove the facial regions from these denoised latent codes and stitch them into the corresponding facial areas of the latent codes synthesized by correspondence-aware attention. This design does not require additional fine-tuning and synthesizes an album of a given human with both high consistency and fidelity.

### 4. Experiments 4.1. Implementation Details

Hyperparameters. During training, we rescale the image resolution to 512 × 384. We choose the AdamW optimizer [19] with an initial learning rate of 1e−4. The trainable modules are the self-attention modules, cross-attention modules of the denoising UNet and the reference UNet,

[Figure 56]

Figure 3. Qualitative comparison with multi-reference customization methods, including Emu2 [27], Collage Diffusion [25], Paint by Example [34] and AnyDoor [6].

the MLP used in text augmentation, and the convolution-in layer of the denoising UNet.

Evaluation metrics. We evaluate the similarity between the synthesized and the reference garments by calculating the CLIP-Score (CLIP-I) and DINO-Score following DreamBooth [24]. Furthermore, we evaluate the prompt consistency by calculating the CLIP text-image similarity (CLIPT) following Textual Inversion [9]. Additionally, we organize user studies with a group of 23 annotators to compare the generation results in quality and fidelity.

#### 4.2. Comparisons for Compositional Generation

We assess the fidelity-maintenance capability of our method in generating multi-guidance fashion images. Considering that only a few existing works match our task setting, we implement the comparison with two groups of methods.

General customization methods. To evaluate our generation ability conditioned on multi-garment, we compare with the leading general customization methods which receive multiple subjects for reference, comprising Emu2 [27] and Collage Diffusion [25]. We also choose AnyDoor [6] and Paint by Example [34] as a two-stage reference-based generation method, for it requires a pre-generated background image to place the reference objects. Specifically, we first leverage a text-to-image model SD V1.5 with denseposeControlNet to customize a background human, and then we place the reference garments in multiple passes.

In qualitative comparisons, we adapt the inputs correspondingly and make our best effort to adjust each method for better results as presented in Fig. 3. Collage Diffusion [25] tends to blend the identity of different garments and encounters difficulties in maintaining detailed fidelity. While Emu2 [27] demonstrates improved ability in discerning different garments, it still faces challenges in preserving fidelity. AnyDoor [6] and Paint by Example [34] generate images with better fidelity, but they both rely on the textto-image model to create a target human and necessitate inputting different garments in multiple forwards. In contrast, FashionComposer demonstrates promising performance for one-pass multi-reference customization with the best detail fidelity and an excellent understanding of garment subjects.

[Figure 57]

- Figure 4. Qualitative comparison with garment-centric fashion image synthesis methods, including StableGarment [30], IMAGDressing-v1 [26], and Magic Clothing [4], where ours better preserves the identity of the target objects. Note that all approaches do not finetune the model on the test samples.

Table 1. Comparison with multi-object reference generation methods. The first three rows represent one pass multi-reference customization methods and the last two rows represent two stage inpainting pipeline based on pre-generated base images.

customization ability with the condition of the face and pose image, we compare with recent garment-centric methods which receive garment image and text description as key conditions and images of face and pose as possible additional references.

Method CLIP-I↑ DINO↑ CLIP-T↑

As demonstrated in Fig. 4, all the generation methods keep high fidelity of fine-grained garment details while none of them can achieve highly flexible compositionality as our work does. StableGarment [30] could only be guided by textual description and upper cloth. IMAGDressingv1 [26] and Magic Clothing [4] can take all conditions except for lower cloth as guidance to generate a target person with a specified pose, face, and upper cloth. However, they run into trouble when keeping face identities and taking lower garments as references.

Ours 77.60 40.11 27.71 Emu2 69.70 35.96 20.54 Collage Diffusion 67.80 34.16 22.14

AnyDoor+ControlNet 72.40 37.94 27.00 Paint-by-example+ControlNet 64.50 34.60 23.77

In quantitive comparisons, we first prepare 100 multigarment reference images and corresponding prompts as input conditions. For multi-reference customization methods Emu2 and Collage Diffusion, we customize fashion images through one pass with possible additional input like bounding boxes for Emu2. For single reference-based methods including AnyDoor and Paint by Example, we need to generate densepose-conditioned person images and their clothagnostic image by utilizing Controlnet and human parsing model. Then we can implement the two-stage inpainting pipeline for image customization under multi-garment guidance. CLIP-I, CLIP-T, and DINO are used as the evaluation metrics for comparisons of image similarity between reference images and customized images as well as text-image similarity between text prompts and fashion images. Tab. 1 shows that our method outperforms all compared methods in both text alignment and image similarity.

#### 4.3. Comparisons of Virtual Try-on

Standard try-on. A quantitative comparison is conducted on VITON-HD [12] dataset with several state-of-the-art open-source virtual try-on methods. The comparison is performed under both paired and unpaired settings to measure the similarity between the synthesized results and ground truth and the generalization performance of the models. The results are presented in Tab. 2 and our model outperforms all others across all metrics except LPIPS. GP-VTON and DCI-VTON as warping-based methods, have advantages in SSIM and LPIPS but perform weaker in KID and FID. This result suggests that warping-based methods may fo-

Fashion image synthesis. To evaluate our garment-driven

[Figure 58]

- Figure 5. Diverse virtual try-on results of FashionComposer for upper, lower, and outfit try-on tasks.

[Figure 59]

- Figure 6. Qualitative comparison for the reference encoder. Reference UNet better preserves the fine details of the garments.

cus more on ensuring structural and perceptual similarity but lack realism and detailed naturalness than our diffusionbased model, which convincingly verifies that our method can successfully preserve the slight details and achieve a more realistic try-on effect for these garments.

Multi-garment try-on. Fig. 5 illustrates our results on various virtual try-on tasks, including upper garment try-on conducted on VITON-HD [12] dataset, and lower garment try-on as well as outfit try-on on DressCode [20] dataset. For uppers and lowers, our approach can generate flawless images preserving the consistent identity of the garment textures and devoid of artifacts. Regarding outfit try-on setting, our method can accurately recognize the type of elements in asset image even when shoes are included, and generate reasonable results with precise fitting between garment and body parts.

#### 4.4. Ablation Study

Reference UNet. We ablate different techniques to keep the detail fidelity of the reference garments. For DINOv2 embeddings of the garments, we condition them on the denoising UNet through cross-attention and omit the reference UNet and the text prompt. For ControlNet, we input the garment’s composition into the ControlNet branch in place of the reference UNet.

Fig. 6 presents the qualitative results of different techniques. All methods succeed in aligning the posture of the synthesized individual with the reference densepose map. In terms of detail fidelity, ControlNet can only grasp the

- Table 2. Quantitative comparison for the standard virtual tryon task on the VITON-HD test dataset.

Methods

VITON-HD Paired Unpaired SSIM ↑ FID ↓ KID ↓ LPIPS ↓ FID ↓ KID ↓

DCI-VTON [10] 0.8620 9.408 4.547 0.0606 12.531 5.251 StableVITON [15] 0.8543 6.439 0.942 0.0905 11.054 3.914 StableGarment [30] 0.8029 15.567 8.519 0.1042 17.115 8.851 MV-VTON [29] 0.8083 15.442 7.501 0.1171 17.900 8.861 GP-VTON [32] 0.8701 8.726 3.944 0.0585 11.844 4.310 LaDI-VTON [21] 0.8603 11.386 7.248 0.0733 14.648 8.754 OOTDiffusion [33] 0.8187 9.305 4.086 0.0876 12.408 4.689 Ours 0.8771 5.842 0.906 0.0727 9.205 1.3606

- Table 3. Quantitative study for the reference UNet. We compare with other options for the appearance encoders like DINOv2 and ControlNet. Reference UNet shows the best performance. Method CLIP-I↑ DINO↑ CLIP-T↑

DINOv2 Embeddings 76.80 38.22 26.17 ControlNet 75.94 33.47 27.10 Reference UNet 77.30 39.39 27.74

- Table 4. Quantitative study for subject-binding attention. Bind(1) means only augmenting the self-attention modules of the UNet down and up blocks with the smallest resolution. Conv-in refers to injecting the text embeddings through the Convolution-in layer of the reference UNet.

Method CLIP-I↑ DINO↑ CLIP-T↑ Quality↑ Fidelity↑ w/o Binding 77.30 39.39 27.74 84 74 Conv-in 77.60 39.39 27.86 90 122 Bind(1) 77.20 39.42 28.10 169 95 Bind(1,2,3) 77.60 40.11 27.71 140 192

main semantic information of the garments and it fails to maintain the fidelity. Conditioning the denoising UNet on DINOv2 embeddings improves the fidelity, but it still struggles with garments with more intricate textures and details. The Reference UNet demonstrates a significant advancement in maintaining detail fidelity. It faithfully preserves all the patterns, textures, and tiny details of each reference garment. Tab. 3 presents the quantitative ablation study on the effectivity of the reference UNet. Reference UNet surpasses other methods in image similarity (CLIP-I and DINO) and text-image similarity (CLIP-T).

Subject-binding attention. We observe different image fidelity when binding with text embeddings on different UNet blocks, including Bind(1,2,3) (i.e. on all blocks) and Bind(1) (i.e. on the smallest resolution). Intuitively, UNet blocks with higher resolution are likely to capture more detailed features, whereas the low-resolution blocks may primarily handle semantic information. Therefore, we ablate the selection of UNet blocks for applying attention-binding. We also try to directly inject mask map of asset image into the conv-in layer of the reference UNet.

Tab. 4 illustrates the quantitative impact of subjectbinding attention. Bind(1,2,3) demonstrates the best image similarities, indicating its extraordinary ability to maintain detail fidelity. Bind(1) shows worse image similarities. For the CLIP-T Score, all methods demonstrate compara-

[Figure 60]

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

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Garments Densepose Conv-in w/o binding Bind(1,2,3) Bind(1)

- Figure 7. Qualitative ablation study on subject-binding attention. Bind(1) means only modifying the self-attention modules of UNet blocks with the smallest resolution. Conv-in refers to injecting the mask map through the Convolution-in layer of the reference UNet. We highlight mistakes in rows 2-3 using red boxes.

ble generation abilities in maintaining semantic similarity, as the quantitative values are nearly identical. We also conduct a user study involving 23 annotators to compare the quality and fidelity of each method in Tab. 4. Specifically, we randomly pick 20 compositional garment images and correspondently generate 20 images for each method. Subsequently, we formulate two questions for each image: the first prompts the annotator to select the image with the highest quality, while the second focuses on selecting the image with the best fidelity. We aggregate the number of wins for each method in terms of quality and fidelity. Results show that injecting text representations(Bind) or mask map(Conv-in) indeed enhance the quality and fidelity of the generation results. Particularly, subject-binding attention significantly improves them, and binding text features in all UNet blocks significantly improves the detail fidelity, while restricting subject-binding attention in partial blocks sacrifices fidelity for slight improvement of quality.

Fig. 7 provides the qualitative ablation studies of subjectbinding attention. Without subject-binding attention, the model blends the identity of different garments. Injecting mask maps using the convolution-in layer mitigates the subject-blending issue. However, it compromises fidelity and still cannot well discern different garments. In contrast, subject-binding attention succeeds in distinguishing between different garments while only slightly compromising fidelity (not obvious). We try to mitigate this issue by restricting the subject-binding attention areas (Bind(1)), but it’s more harmful to the generation fidelity.

In conclusion, given all the results, we assert that the Bind(1,2,3) method achieves the most satisfactory balance between generation quality and fidelity in the multi-garment reference generation task. Therefore, we leverage it as our

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Densepose Baseline CFA CAA+LCA

Figure 8. Qualitative ablation study on synthesizing independently (baseline), cross-frame attention (CFA), and latent code alignment (LCA). Reference garments and text are omitted here.

default setting.

Human album generation. In Fig. 8, we analyze the effectiveness of our proposed modules targeting human album generation. In this task, we aim to generate a series of human images with the same identity. The baseline method is synthesizing images independently with the same appearance descriptions in the text prompt. We observe that the baseline method generates images with totally different appearances. Applying cross-frame attention generates appearances with high consistency across views but compromises posture alignment (row 3) and dressing naturalness. The model with correspondence-aware attention demonstrates better global consistency, but the appearance consistency is still not perfect. However, when incorporating correspondence-aware attention and latent code alignment, our method could generate desired results with both high consistency and fidelity (row 4).

### 5. Conclusion

We propose FashionComposer, a diffusion-based method for highly customized fashion image generation. The primary contribution is compositionality through referring to multi-modal input and multi-subject image. We propose subject-binding attention that binds visual features of various image components with their corresponding descriptions, enabling semantic distinctions between different garments while seamlessly compatible with Feature Injection Attention for fidelity preservation. However, the generation capacity of our model is still limited by the scale and bias of the training dataset in terms of race, gender, and body figure. A direct way to improve the diversity and fairness of generation is to further enhance data variety and amount.

### 6. Acknowledgements

This work was supported by DAMO Academy through DAMO Academy Research Intern Program.

### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv:2309.16609, 2023. 4
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv:2308.12966, 2023. 2, 4
- [3] Mengting Chen, Xi Chen, Zhonghua Zhai, Chen Ju, Xuewen Hong, Jinsong Lan, and Shuai Xiao. Wear-any-way: Manipulable virtual try-on via sparse correspondence alignment. arXiv:2403.12965, 2024. 4
- [4] Weifeng Chen, Tao Gu, Yuhao Xu, and Chengcai Chen. Magic clothing: Controllable garment-driven image synthesis. arXiv:2404.09512, 2024. 6
- [5] Xi Chen, Yutong Feng, Mengting Chen, Yiyang Wang, Shilong Zhang, Yu Liu, Yujun Shen, and Hengshuang Zhao. Zero-shot image editing with reference imitation. arXiv:2406.07547, 2024. 4
- [6] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In CVPR, 2024. 5
- [7] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, 2021. 2, 4
- [8] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. arXiv:2403.05139, 2024. 2
- [9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023. 2, 5
- [10] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of diffusion models for high-quality virtual try-on with appearance flow. In ACMMM, 2023. 7
- [11] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In CVPR, 2018. 2
- [12] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In CVPR, 2018. 2, 4, 6, 7
- [13] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv:2311.17117, 2023. 2, 4
- [14] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero:

- Text-to-image diffusion models are zero-shot video generators. arXiv:2303.13439, 2023. 5
- [15] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent diffusion model for virtual try-on. In CVPR, 2024. 2, 7
- [16] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In CVPR, 2016. 4
- [17] Zhiheng Liu, Ruili Feng, Kai Zhu, Yifei Zhang, Kecheng Zheng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones: Concept neurons in diffusion models for customized generation. arXiv:2303.05125, 2023. 2
- [18] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries. 2023. 2
- [19] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv:1711.05101, 2017. 5
- [20] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress Code: High-Resolution Multi-Category Virtual Try-On. In ECCV,

2022. 4, 7

- [21] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In ACMMM, 2023. 2, 7
- [22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [24] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 5
- [25] Vishnu Sarukkai, Linden Li, Arden Ma, Christopher R´e, and Kayvon Fatahalian. Collage diffusion. In WACV, 2024. 2, 5
- [26] Fei Shen, Xin Jiang, Xin He, Hu Ye, Cong Wang, Xiaoyu Du, Zechao Li, and Jinhui Tang. Imagdressing-v1: Customizable virtual dressing. arXiv:2407.12705, 2024. 6
- [27] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv:2312.13286, 2023. 2, 5
- [28] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristic-preserving image-based virtual try-on network. In ECCV, 2018. 2
- [29] Haoyu Wang, Zhilu Zhang, Donglin Di, Shiliang Zhang, and Wangmeng Zuo. Mv-vton: Multi-view virtual try-on with diffusion models. arXiv:2404.17364, 2024. 2, 7
- [30] Rui Wang, Hailong Guo, Jiaming Liu, and Huaxia Li. Stablegarment: Garment-centric generation via stable diffusion. arXiv:2403.10783, 2024. 6, 7

- [31] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multi-subject image generation with localized attention. arXiv:2305.10431, 2023. 2
- [32] Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, and Xiaodan Liang. Gp-vton: Towards general purpose virtual try-on via collaborative local-flow global-parsing learning. In CVPR,

2023. 2, 7

- [33] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv:2403.01779, 2024. 2, 7
- [34] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In CVPR, 2023. 5
- [35] Lu Yang, Wenhe Jia, Shan Li, and Qing Song. Deep learning technique for human parsing: A survey and outlook. arXiv:2301.00394, 2023. 4
- [36] Shilong Zhang, Lianghua Huang, Xi Chen, Yifei Zhang, Zhi-Fan Wu, Yutong Feng, Wei Wang, Yujun Shen, Yu Liu, and Ping Luo. Flashface: Human image personalization with high-fidelity identity preservation. arXiv:2403.17008,

2024. 4

- [37] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In CVPR, 2023. 2

