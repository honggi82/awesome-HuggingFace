## Edicho: Consistent Image Editing in the Wild

Qingyan Bai1 Hao Ouyang2 Yinghao Xu3 Qiuyu Wang2 Ceyuan Yang4 Ka Leong Cheng1 Yujun Shen2† Qifeng Chen1†

1HKUST 2Ant Group 3Stanford University 4CUHK

# arXiv:2412.21079v3[cs.CV]14Jan2025

[Figure 1]

[Figure 2]

[Figure 3]

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

[Figure 10]

[Figure 11]

[Figure 12]

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

Figure 1. Given two images in the wild, Edicho generates consistent editing versions of them in a zero-shot manner. Our approach achieves precise consistency for editing parts (left), objects (middle), and the entire images (right) by leveraging explicit correspondence.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

### 1. Introduction

### Abstract

[Figure 23]

The ability to consistently edit images across different instances is of paramount importance in the field of computer vision and image processing [1, 12, 61, 63]. Consistent image editing facilitates numerous applications, such as creating coherent visual narratives and maintaining characteristics in marketing materials. As in Fig. 1, sellers or consumers can enhance photos of their favorite products, such as toys or shoes, by applying consistent decorative elements, making each item appear more appealing or personalized. Similarly, during themed events like a masquerade ball or Halloween, families and friends may hope to uniformly style masks or dresses across their photos, ensuring a harmonious visual presentation. Another instance for content creators is consistently making multiple of your photos look like a graceful elf or an impressive Superman. By ensuring the edits applied to one image can be reliably replicated across others, we also enhance the efficiency and quality of tasks ranging from photo retouching to data augmentation for customization [31, 48] and 3D reconstruction [55].

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

As a verified need, consistent editing across in-thewild images remains a technical challenge arising from various unmanageable factors, like object poses, lighting conditions, and photography environments. Edicho1 steps in with a training-free solution based on diffusion models, featuring a fundamental design principle of using explicit image correspondence to direct editing. Specifically, the key components include an attention manipulation module and a carefully refined classifier-free guidance (CFG) denoising strategy, both of which take into account the pre-estimated correspondence. Such an inference-time algorithm enjoys a plug-and-play nature and is compatible to most diffusionbased editing methods, such as ControlNet and BrushNet. Extensive results demonstrate the efficacy of Edicho in consistent cross-image editing under diverse settings. We will release the code to facilitate future studies. Project page can be found here.

† Corresponding authors. 1“Edicho” is an abbreviation of “edit echo”, implying that the edit is

Despite the significance of consistent editing, achieving it across diverse images remains a challenging task. Previ-

echoed across images.

ous editing methods [6, 25, 60] often operate on a per-image basis, leading to variations that can disrupt the uniformity required in specific applications. Previous attempts to address this issue have encountered limitations. Learningbased methods [12, 58] that involve editing a single image and propagating the changes to others lack proper regularization and tend to produce inconsistent results. They struggle to acquire high-quality paired training data and fail to enforce the necessary constraints to maintain uniformity. Alternatively, strategies without optimization [1, 7, 18] rely on the implicit correspondence predicted from the attention features to achieve appearance transfer. Yet, due to the unstable implicit correspondence prediction, these approaches struggle to account for the intrinsic variations between images, leading to edits that appear inconsistent or distorted when applied indiscriminately.

Inspired by the property of diffusion models [19, 33, 47, 49] where intermediate features are spatially aligned with the generated image space, we propose a novel, trainingfree, and plug-and-play method that enhances consistent image editing through explicit correspondence between images. Different from previous training-free methods [1, 7, 18] relying on implicit correspondence from attention weights to transfer appearance, we propose to predict the correspondence between the inputs with a robust correspondence extractor before editing. Our approach then leverages the self-attention mechanism within neural networks to transfer features from a source image to a target image effectively. Specifically, we enhance the self-attention mechanism by warping the query features according to the correspondence between the source and target images. This allows us to borrow relevant attention features from the source image, ensuring that the edits remain coherent across different instances. To achieve finer control over the consistency of the edits, we further modify the classifier-free guidance (CFG) [14] computation by incorporating the precomputed correspondence. This modification guides the generation process, aligning it more closely with the desired edits while maintaining high image quality. During this design, we empirically observed that directly transferring the source noisy latent to the target image often results in blurred and over-smoothed images. Inspired by the concept of NULL-text Inversion [36], we discovered that fusing features from unconditional embeddings enhances consistency without compromising the image quality.

Moreover, our algorithm is specifically designed to handle in the wild images — those captured under diverse and uncontrolled real-world conditions. Benefiting from the correspondence, this capability ensures that our method remains robust against variations in lighting, backgrounds, perspectives, and occlusions commonly found in natural settings. By effectively processing in the wild images, the versatility of our method allows for additional numer-

ous practical applications. For instance, in a customized generation, our method enables the generation of more consistent image sets by editing, which is valuable for learning customized models for novel concepts and creating personalized content. Additionally, we can apply new textures consistently across different views of an object and acquire the corresponding 3D reconstructions of the edits, benefiting from the editing consistency.

In summary, we introduce explicit correspondence into the denoising process of diffusion models in order to achieve consistent image editing. We enhance the selfattention mechanism and modify classifier-free guidance to incorporate correspondence information, improving edit consistency without degrading image quality. We also further demonstrate that fusing features from unconditional embeddings enhances consistency, inspired by null-text inversion techniques. The final method, due to its trainingfree and plug-and-play nature, is able to function across various models and diverse tasks, enabling both global and local edits. We validate the effectiveness of the proposed method through extensive experiments, showing superior performance in both quantitative metrics and qualitative assessments.

### 2. Related Work

Generative models for image editing. Recently, diffusion models have shown unprecedented power in various generative tasks [2–5, 8, 10, 13, 14, 16, 20, 21, 28–31, 39– 41, 47–49, 51, 54, 56]. To unleash its potential in editing, PnP [52] proposes to borrow convolutional and attention features from the input image during generation to achieve manipulation. While MasaCtrl [7] and Cross-ImgaeAttention [1] modify self-attention modules for editing, by combining the target queries and source keys and values. Prompt2Prompt [35] focuses on the cross-attention layers in text-to-image models and proposes manipulating the textual embedding. Serving as the foundation of these editing methods, image inversion is also widely studied by researchers [17, 26, 34, 36, 49]. Different from the aforementioned training-free editing methods, Instruct-Pix2Pix [6], ControlNet [60, 62], T2I-Adapter [37], Composer [23], and BrushNet [25] learn editing models conditioned on the input images and instructions, which are based on or fine-tuned from the pre-trained latent diffusion models for better quality and training stability. Another branch of works [11, 12, 58] aims at exemplar-based editing, where a pre-trained diffusion model is finetuned to function conditioned on the exemplar image as well as the masked source image. [38] achieves pose transfer among the image batch by manipulating the StyleGAN latent codes following the exemplar image. Unlike the works discussed above, we focus on the task of consistent editing for images in the wild, and propose an explicit correspondence-guided

Input Explicit Implicit

the image pairs by existing visual understanding methods such as [32, 50, 55, 59]. Then we seek help from the pre-trained editing models [25, 60] built upon Stable Diffusion [47] to achieve editing, and guide their denoising process with these pre-computed explicit correspondences to ensure consistency. In this section, we first review some preliminary concepts of diffusion models, which is followed by a subsection discussing the correspondence extraction and comparisons. Then we introduce the correspondenceguided denoising process that includes two levels - the level of attention features and the level of noisy latents. Note that these designs on feature manipulations are only applied to a range of denoising steps and layers, in order to preserve the strong generative prior from the pre-trained models.

|[Figure 28]<br><br>|[Figure 29]<br><br>|[Figure 30]<br><br>|
|---|---|---|

- (a)

(c)

|[Figure 31]<br><br>|[Figure 32]<br><br>|[Figure 33]<br><br>|
|---|---|---|

- (b)

|[Figure 34]<br><br>|[Figure 35]<br><br>|[Figure 36]<br><br>|
|---|---|---|

#### 3.1. Preliminaries

Diffusion models are probabilistic generative models trained through a process of progressively adding and then removing noise. The forwarding process adds noise to the images as follows:

xt = √αt · x0 + √1 − αt · z, (1)

- Figure 2. Comparisons of the implicit and our explicit correspondence prediction for the images in the wild. The implicit correspondence from cross-image attention calculation is less accurate and unstable with the change of denoising steps and network layers.

where z ∼ N(0,I) and αt indicates the noise schedule. And a neural network ϵθ (xt,t) is trained to predict the adding noise z during the denoising backward process and finally achieves sampling from Gaussian noise xT ∼ N(0,I). In the formulation of latent diffusion models (LDMs) [47], a pair of pre-trained variational encoder E and decoder D serve perceptual compression and enable denoising from the noisy latents z in this latent space.

solution.

Correspondence from neural networks. The concept of correspondence is widely applicable and essential in various real-world scenarios [61, 63], where understanding relationships between data points is crucial. Neural networks have been broadly employed to find correspondences for image, video, and 3D scenes through supervised learning [24, 24, 32, 44, 55]. DIFT [50] proposes to extract semantic correspondence among in-the-wild images by directly matching the features from the pre-trained diffusion models. SD-DINO [59] further ensembles features from diffusion models and DINO [9] for correspondence matching. Once correspondences are established, they can be utilized in various applications. For instance, in object tracking, networks can maintain correspondences across video frames to follow objects through occlusions and transformations [15, 27, 40, 43, 57]. Our work leverages these principles by integrating correspondence into the diffusion model framework, enabling precise and consistent multi-image editing without additional training.

Classifier-free guidance (CFG) [14] is an innovative technique introduced to enhance the quality and diversity of generated images by diffusion models without relying on additional classifiers. Specifically, CFG introduces a mixing coefficient to blend the conditional and unconditional predictions made by the denoising model. The unconditional prediction is typically obtained by setting the condition to a null or default value.

Reference networks for editing. Recent editing methods [25, 60] purpose editing by learning an additional reference network over the pre-trained large diffusion models while keeping the pre-trained backbone fixed. This network-topology-preserving design successfully separates control signals and the pretrained generative prior.

#### 3.2. Correspondence Comparison and Prediction

Correspondence comparisons. In order to achieve the target of consistent editing, we start with the comparison between explicit and implicit correspondence for matching. Explicit extractors predict correspondence from the input images with a single-pass forwarding and apply this prediction to all the target network layers and denoising steps. While implicit extractors predict correspondences

### 3. Method

In this work, we focus on the task of consistent image editing, where multiple images are manipulated altogether to achieve consistent and unified looks. To achieve this, we first extract explicit semantic correspondence among

[Figure 37]

(a) Overview & Corr-Attention (b) CFG with Correspondence

[Figure 38]

[Figure 39]

끫欲(끫歸끫殬,끫歸끫殮)

[Figure 40]

Corr Cache

MD5 Check

[Figure 41]

Corr

[Figure 42]

Net

∅ 끫毎끫毂 c

끫毎끫毂

Corr-Attention Block

[Figure 43]

|[Figure 44]|
|---|

[Figure 45]

[Figure 46]

DiffusionModel

[Figure 47]

KV

끫歬끫殬,끫殮

KV Q

[Figure 48]

[Figure 49]

끫歬끫殬,끫殮

| | |
|---|---|
|끫毊|끫殮|
| | |
|끫毊|끫殬|
| | |

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

AttentionAttention

[Figure 56]

Image 끫歸끫殮

[Figure 57]

[Figure 58]

Warp

[Figure 59]

[Figure 60]

끫殈끫殮

[Figure 61]

[Figure 62]

[Figure 63]

Q

[Figure 64]

[Figure 65]

끫毎끫毂−1끫殾끫殾끫殾

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

끫毎끫毂−1끫毄 끫毎끫毂−1c

[Figure 71]

끫歼끫殬,끫殒끫殬

[Figure 72]

[Figure 73]

Image 끫歸끫殬

| | |
|---|---|
| | |

[Figure 74]

[Figure 75]

[Figure 76]

끫殈끫殬

[Figure 77]

[Figure 78]

[Figure 79]

Fusing

Zoomed-in Block

Inputs

Edits

Weighted Sum

끫毎끫毂−1

[Figure 80]

[Figure 81]

[Figure 82]

Prompt: “A fashionable sport shoe”

- Figure 3. Framework of Edicho. To achieve consistent editing, we first predict the explicit correspondence with extractors for the input images. The pre-computed correspondence is injected into the pre-trained diffusion models and guide the denoising in the two levels of (a) attention features and (b) noisy latents in CFG.

by calculating the similarities between attention queries and keys for each layer and denoising step. As in the previous training-free editing methods [1], these correspondences are subsequently applied to this layer and step for editing.

In Fig. 2, we present the correspondence prediction results using explicit and implicit methods. For explicit prediction in cases (a) and (b), we employed DIFT [50] and Dust3R [55] is utilized for (c). For the implicit approach, we followed Cross-Image-Attention [1] to compute the correspondence based on the attention similarity by querying the attention keys of the matching image with Qi · KjT and visualize the corresponding positions with maximum similarity, where i and j indicate the image indices. Additionally, for cases (a), (b), and (c), we selected different network layers and denoising steps (1, 10), (2, 15), (4, 25) for extraction to achieve a more comprehensive exploration where x and y in (x, y) represent the decoder layer number of the diffusion model and the denoising step. The visualized results in Fig. 2 indicate that the correspondences obtained by explicit prediction are notably more accurate than those obtained by implicit methods. Moreover, the predictions from implicit methods tend to become unstable with changes in network layers and denoising steps. These results are also aligned with previous work [50, 59] indicating that only specific layers or steps of generative models are suitable for effective visual understanding like point matching. The inaccurate correspondence matching leads to borrowing inaccurate features in performing cross-image attention, which hinders the editing consistency of the editing methods merely based on implicit attentions [1, 7, 18]. This further boosts our motivation of introducing more robust explicit correspondence to guide the denoising process. Additional details and comparisons for correspondence prediction are provided in Appendix.

Correspondence prediction. To achieve consistent editing for images Ii and Ij, the first step in our editing method involves extracting robust correspondence from the input images using a pre-trained correspondence extractor such as [50, 55]:

Ci,j = ϕ(Ii,Ij), (2)

where ϕ and C indicate the extractor and correspondence. In our practice, the extractor is instantiated as in DIFT [50]. To further optimize efficiency, we implement a strategy to avoid redundant computations of correspondences, particularly when the same images or image groups are processed multiple times. We achieve this by encoding each image group using an MD5 [46] hash function, creating a unique identifier for each. After storing the identifier (key) and correspondence (value) in a minor database, the input image group would first retrieve it before editing for acceleration.

#### 3.3. Attention Manipulation with Correspondence

Recall that the intermediate feature xi in self-attention blocks are firstly projected to queries Qi = fQ(xi), keys Ki = fK(xi), and values Vi = fV (xi) with the learned projection matrix fQ, fK, and fV . Then the attention features F could be computed by autonomously computing and assessing the relevance of these feature representations following [53]. Inspired by the comparisons between explicit and implicit correspondence, we propose to guide self-attention with explicit correspondence to achieve consistent editing, which is termed as Corr-Attention. For an image pair (Ii,Ij) among the inputs, we borrow features from the query matrix Qi to Qj to form a new query Qedit based on the explicit correspondence:

Qedit = Warp(Qi,Qj,Ci,j), (3)

Where the Warp function indicates the process of borrowing features by warping corresponding tokens to the source

based on the corresponding location denoted by correspondence. Considering (1) tokens of Qedit are borrowed from Qi and (2) to further improve consistency, we query Ki,Vi instead of Kj,Vj during the editing of Ij:

Fj = softmax

Qedit · KiT √dk · Vi, (4)

where dk indicates the dimension of Q and K, and Fj represents the attention outputs of Ij. By transferring attention features from the source, we effectively achieve editing consistency during the denoising process.

#### 3.4. Classifer-free Guidance with Correspondence

In order to retain a finer consistency over the edited images, we take a further step from the attention feature control and focus on the noisy latents in Classifier-free Guidance (CFG) [14]. Specifically, we extend the traditional CFG framework to facilitate synchronized editing of multiple images by leveraging explicit correspondences and propose Corr-CFG. NULL-text inversion [36] demonstrates that optimizing unconditional word embeddings can achieve precise image inversion and semantic editing. Inspired by this approach, our primary objective is to preserve the integrity of the pre-trained model’s powerful generative priors during the consistent editing process. To achieve this, we propose manipulating only the unconditional branch of zj within the Classifier-Free Guidance (CFG) framework under the guidance of correspondence. Recall that in CFG the denoising process are split into two branches of the conditional and unconditional, and the noise are estimated with neural network ϵθ:

zct−1 = ϵθ (zt,c), (5) zut−1 = ϵθ (zt,∅), (6)

where c represents the condition (text prompt) and ∅ indicates the null text. Specifically, we modified the unconditional noise component of zj and incorporated information from zi into it during the denoising process, which ensures coherent edits:

ϵuncondθ zjt = T ϵθ zit,∅ ,ϵθ zjt,∅ ,Ci,j , (7)

where T represents a fusing function that aligns the unconditional noises and t indicates the time-step:

T (zi,zj,Ci,j) = (1 − λ) · zj + λ · Inj(zi,zj,Ci,j,γ).

(8)

λ and γ ∈ (0,1] here are adjustable parameters. The function Inj indicates randomly choosing latent of zi in a portion of γ and injecting them into zj. At last, we apply the guidance and fuse the conditional and unconditional

predictions as in the prior paradigm [14]:

ϵguidedθ (zt,c) = ϵuncondθ (zt) + s · (ϵθ (zt,c) − ϵθ (zt,∅)),

(9) where s indicates the guidance scale. The final latents generated as such are at last sent to the VAE decoder [47] to be decoded into images.

### 4. Experiments

#### 4.1. Experimental Setup

Settings. We use Stable Diffusion [47] as the base model and adopt BrushNet [25] and ControlNet [60] as the reference networks for editing. We adopt the DDIM [49] scheduler and perform denoising for 50 steps. By default, the proposed correspondence-guided denoising strategy is applied from 4th to 40th steps and from the eighth attention layer to ensure consistency as well as preserve the strong generative prior. Note the optimal choice of these may vary when different base models are utilized. The testing samples are partially acquired from the internet, while others of them are from the dataset of DreamBooth [48] and Custom Diffusion [31].

Evaluation metrics. We follow Custom Diffusion [31] and adopt the prevalent multi-modal model CLIP [42] to evaluate various methods in terms of text alignment (TA) and editing consistency (EC). Specifically, on the one hand, feature similarity of the target prompt and model output are computed to judge the textual alignment. On the other hand, feature similarity of the edited images is adopted to evaluate the editing consistency. User studies (US) are also incorporated to further evaluate the practical applicability and user satisfaction.

Baselines. We include both local and global editing tasks, as well as numerous previous image editing methods, for comprehensive comparisons. Specifically, for the task of local editing, we include prior works of Adobe Firefly [45], Anydoor [12], and Paint-by-Example [58] for comparison. Among these aforementioned methods, Firefly is a state-of-the-art commercial inpainting tool developed by Adobe, which could repaint local regions of the input image following the given textual prompts. In order to achieve the task of consistent editing, the images among the set would be inpainted with the same detailed prompts. Both Anydoor and Paint-by-example are Latent Diffusion Models (LDMs) supporting repainting target regions with the given reference image. Thus, we sent an inpainted image to these models as the reference, expecting consistent editing results. While for global editing, we compare our approach with MasaCtrl [7], StyleAlign [18], and Crossimage attention [1]. The aforementioned methods achieve editing by manipulating and fusing attention features from various sources. Different from our method, they compute

Input Ours Adobe Firefly Anydoor Paint-by-Example

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Text Prompt: A dancing girl wearing Cheongsam

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Text Prompt: A girl wearing a ball mask

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

Text Prompt: A dog wearing a collar

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

Text Prompt: A cool sport shoe

- Figure 4. Qualitative comparisons on local editing with Adobe Firefly (AF) [45], Anydoor (AD) [12], and Paint-by-Example (PBE) [58]. The inpainted areas of the inputs are highlighted in red.

[Figure 125]

[Figure 126]

[Figure 127]

Text Prompt: A future oriented robot

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Input Ours MasaCtrl StyleAligned Cross-Image-Attention

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Text Prompt: The cat princess wearing a dress

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Text Prompt: A car from the future world

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Text Prompt: A beautiful elf in the western magical world

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

- Figure 5. Qualitative comparisons on global editing with MasaCtrl (MC) [7], StyleAligned (SA) [18], and Cross-Image-Attention (CIA) [1].

Table 1. Quantitative results respectively on local and global editing. We follow Custom Diffusion [31] to evaluate various methods on text alignment (TA) and editing consistency (EC).

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Edited&Input

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Method TA ↑ EC ↑

Method TA ↑ EC ↑

AF [45] 0.3082 0.8569 AD [12] 0.2981 0.8320 PBE [58] 0.2969 0.8683

MC [7] 0.3140 0.9258 SA [18] 0.3021 0.9099 CIA [1] 0.2914 0.8912

[Figure 171]

[Figure 172]

[Figure 173]

Ours 0.3176 0.8931

Ours 0.3228 0.9355

Input Edit #1 Ours w/o Corr-Attention

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

A [V] dog in a girl’s arms A person wearing [V] shoes

A [V] girl cooking in the kitchen

[Figure 181]

[Figure 182]

[Figure 183]

CustomizedGenerations

- (a)
- (b)

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

A [V] dog lying on the bed

A [V] shoe at the exhibition

A [V] girl on a chair on the beach

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Input Edit #1 Ours w/o Corr-CFG

[Figure 196]

- Figure 6. Ablation studies on the (a) correspondence-guided attention manipulation (Corr-Attention) and (b) correspondenceguided CFG (Corr-CFG).

[Figure 197]

[Figure 198]

[Figure 199]

A [V] dog on the beach

[V] shoes under the spotlight

A [V] school girl in the classroom

Figure 7. With outputs from our consistent editing method (upper) and the customization [48] techniques, customized generation (lower) could be achieved by injecting the edited concepts into the generative model.

implicit correspondences from attention weights to ensure consistency among the editing outputs.

#### 4.2. Evaluation

(EC) mentioned in Sec. 4.1. As illustrated in Tab. 1, for local editing, our method attained the best scores in both TA and EC for local editing tasks, demonstrating a significant improvement over the competing methods. For global editing tasks, our method continued to outperform the other counterparts, reaching a TA score of 0.3228 and an EC score of 0.9355. Overall, these results, along with the user studies in Appendix, clearly demonstrate the effectiveness of our method in achieving high text alignment and editing consistency across both local and global editing scenarios.

Qualitative results. We present a qualitative evaluation of the consistency editing methods, focusing on both local editing (image inpainting) and global editing (image translation). The comparisons for local editing in Fig. 4 include results from our method, Adobe Firefly (AF), Anydoor (AD), and Paint-by-Example (PBE). The results demonstrate that our approach consistently maintains the integrity of the input images across different modifications, including the cloth textures, mask and collar appearance, and even the eyelet amount of shoes, thanks to the introduction of explicit correspondence. The baselines of global editing mainly include the ones predicted merely by implicit attentions - MasaCtrl (MC), StyleAligned (SA), and Cross-Image-Attention (CIA). As in Fig. 5, our method also achieves superior consistency and thematic adherence among the edits, such as the dress of the cat. The implicit alternatives like MasaCtrl fail in the car roof, the high neckline of the elf, and the hole number of the robot.

#### 4.3. Ablation Studies

In order to validate the effectiveness of the proposed correspondence-guided attention manipulation (CorrAttention) and correspondence-guided CFG (Corr-CFG) introduced in 3.3 and 3.4, we conduct ablation studies by respectively disabling each of them and testing on the task of consistent editing. When the proposed correspondenceguided attention manipulation (Corr-Attention) is disabled, the diffusion model relies on implicit attention correspondence to maintain consistency just like previous methods [1, 7]. As demonstrated in Fig. 6 (a), the generative

Quantitative results. We conducted a comprehensive quantitative evaluation of our proposed method against several state-of-the-art image editing techniques, focusing on metrics of text alignment (TA) and editing consistency

[Figure 200]

[Figure 201]

[Figure 202]

Input Input

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

3D Reconstruction Matching of Edits Matching of Edits 3D Reconstruction

- Figure 8. We adopt the neural regressor Dust3R [55] for 3D reconstruction based on the edits by matching the 2D points in a 3D space.

[Figure 211]

[Figure 212]

- (a)

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

- (b)
- (c)

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

- Figure 9. Diverse results of consistent image inpainting (a) and translation (b) by the proposed method. Editing results for an image set of three images are demonstrated in (c).

process by introducing low-rank matrices as adaptation parameters. As in Fig. 7, the fine-tuned generative model could yield desirable images corresponding to the edits after concept injection. Thus novel concept generation and concept editing can be achieved in this way, serving as an application example of consistent editing.

3D reconstruction based on the consistent edits. Furthermore, consistent editing could also benifit 3D reconstruction of the edits. We achieve 3D reconstruction with a neural regressor [55] which could predict accurate 3D scene representations from the consistent image pairs. Taking the edited images as inputs, the learned neural regressor could predict the 3D point-based models and 2D matchings without additional inputs such as camera parameters. The reconstruction and matching results are presented in Fig. 8, both of which also suggest the editing consistency of the proposed method. The regressor respectively obtained 11,515 and 13,800 pairs of matching points for the two groups of edits, and only a portion are visualized for clear understanding.

model then would yield flowers of wrong amounts and at improper locations. The number of flowers and inconsistent textures speak for the effectiveness of introducing explicit correspondence to attention manipulation. Recall that correspondence-guided CFG (Corr-CFG) is designed for finer consistency control functioning in the latent space of LDMs, which is validated in Fig. 6 (b) where CorrCFG achieves in generating more consistent textures for the flower on the bowl and stripes at the bottom of the bowl. Additional ablation studies on attention manipulation and CFG could be found in Appendix.

Additional results. Diverse results of multi-image inpainting and translation by the proposed method are provided in Fig. 9 (a) and (b). Editing results for an image set including three images are demonstrated in Fig. 9 (c).

### 5. Conclusion

We introduce Edicho, a novel training-free method for consistent image editing across various images by leveraging explicit correspondence between them. Our approach enhances the self-attention mechanism and the classifierfree guidance computation by integrating correspondence information into the denoising process to ensure consistency among the edits. The plug-and-play nature of our method allows for seamless integration into various models and its applicability across a wide range of tasks. For limitations, sometimes the generated textures would be inconsistent due to the correspondence misalignment, which could be expected to be improved with better correspondence extractors. And inheriting from the pre-trained editing models, sometimes distorted textures would be generated.

#### 4.4. Additional Applications and Results

Customization based on the consistent edits. To further demonstrate the practical utility of the proposed method, we present an application example that integrates DreamBooth [48] and Low-Rank Adaptation (LoRA) [22] techniques for customized image generation based on the multiple edited images. Leveraging the edited outputs from our method, we employ DreamBooth to fine-tune a generative model for 500 steps for concept injection. We also integrate LoRA techniques to further enhance the efficiency of this

### References

- [1] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar AverbuchElor, and Daniel Cohen-Or. Cross-image attention for zero-shot appearance transfer. In ACM SIGGRAPH 2024 Conference Papers, 2024. 1, 2, 4, 5, 6, 7
- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM Trans. Graph., 2023. 2
- [3] Qingyan Bai, Zifan Shi, Yinghao Xu, Hao Ouyang, Qiuyu Wang, Ceyuan Yang, Xuan Wang, Gordon Wetzstein, Yujun Shen, and Qifeng Chen. Real-time 3d-aware portrait editing from a single image. In Eur. Conf. Comput. Vis., 2024.
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. InstructPix2Pix: Learning to follow image editing instructions. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [7] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. MasaCtrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Int. Conf. Comput. Vis., 2023. 2, 4, 5, 6, 7, 1
- [8] Mingdeng Cao, Xuaner Zhang, Yinqiang Zheng, and Zhihao Xia. Instruction-based image manipulation by watching how things move. arXiv preprint arXiv:2412.12087, 2024. 2
- [9] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Int. Conf. Comput. Vis., 2021. 3
- [10] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [11] Xi Chen, Yutong Feng, Mengting Chen, Yiyang Wang, Shilong Zhang, Yu Liu, Yujun Shen, and Hengshuang Zhao. Zero-shot image editing with reference imitation. arXiv preprint arXiv:2406.07547, 2024. 2
- [12] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 1, 2, 5, 6, 7
- [13] Ka Leong Cheng, Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Hao Ouyang, Qifeng Chen, and Yujun Shen. Learning naturally aggregated appearance for efficient 3d editing. arXiv preprint arXiv:2312.06657, 2023. 2
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Adv. Neural Inform. Process. Syst., 2021. 2, 3, 5
- [15] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman.

- Tapir: Tracking any point with per-frame initialization and temporal refinement. In Int. Conf. Comput. Vis., 2023. 3
- [16] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Int. Conf. Comput. Vis., 2023. 2
- [17] Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Anastasis Stathopoulos, Xiaoxiao He, Yuxiao Chen, et al. Proxedit: Improving tuning-free real image editing with proximal guidance. In IEEE Winter Conf. Appl. Comput. Vis., 2024. 2
- [18] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2, 4, 5, 6, 7
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Adv. Neural Inform. Process. Syst., 2020. 2
- [20] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Adv. Neural Inform. Process. Syst., 2022. 2
- [21] Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Int. Conf. Comput. Vis., 2023. 2
- [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In Int. Conf. Learn. Represent., 2022. 8
- [23] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023. 2
- [24] Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, and Kwang Moo Yi. Cotr: Correspondence transformer for matching across images. In Int. Conf. Comput. Vis., 2021. 3
- [25] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In Eur. Conf. Comput. Vis., 2024. 2, 3, 5, 1
- [26] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In Int. Conf. Learn. Represent., 2024. 2
- [27] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In Eur. Conf. Comput. Vis., 2024. 3
- [28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Adv. Neural Inform. Process. Syst., 2022. 2
- [29] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023.
- [30] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant

- Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Int. Conf. Comput. Vis., 2023.
- [31] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 1, 2, 5, 7
- [32] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In Eur. Conf. Comput. Vis., 2024. 3
- [33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In Int. Conf. Learn. Represent., 2022. 2
- [34] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023. 2
- [35] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [36] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2, 5
- [37] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Assoc. Adv. Artif. Intell., 2024. 2
- [38] Thao Nguyen, Utkarsh Ojha, Yuheng Li, Haotian Liu, and Yong Jae Lee. Edit one for all: Interactive batch image editing. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [39] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Int. Conf. Mach. Learn., 2021. 2
- [40] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. CoDeF: Content deformation fields for temporally consistent video processing. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 3
- [41] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In Int. Conf. Learn. Represent., 2023. 2
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Int. Conf. Mach. Learn., 2021. 5
- [43] Frano Rajiˇc, Lei Ke, Yu-Wing Tai, Chi-Keung Tang, Martin Danelljan, and Fisher Yu. Segment anything meets point tracking. arXiv:2307.01197, 2023. 3
- [44] Anurag Ranjan and Michael J Black. Optical flow estimation using a spatial pyramid network. In IEEE Conf. Comput. Vis. Pattern Recog., 2017. 3

- [45] Adobe reseachers. Adobe firefly: Free generative ai for creatives. https://firefly.adobe.com/generate/ inpaint, 2023. 5, 6, 7
- [46] Ronald Rivest. The md5 message-digest algorithm. Technical report, 1992. 4
- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2, 3, 5, 1
- [48] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In IEEE Conf. Comput. Vis. Pattern Recog.,

2023. 1, 5, 7, 8

- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Int. Conf. Learn. Represent., 2021. 2, 5, 1
- [50] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. In Adv. Neural Inform. Process. Syst.,

- 2023. 3, 4, 1

[51] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Trans. Graph.,

- 2024. 2

- [52] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Adv. Neural Inform. Process. Syst., 2017. 4
- [54] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024. 2
- [55] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 1, 3, 4, 8
- [56] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [57] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3
- [58] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2, 5, 6, 7
- [59] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan

- Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. In Adv. Neural Inform. Process. Syst., 2024. 3, 4
- [60] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., 2023. 2, 3, 5, 1
- [61] Pan Zhang, Bo Zhang, Dong Chen, Lu Yuan, and Fang Wen. Cross-domain correspondence learning for exemplarbased image translation. In IEEE Conf. Comput. Vis. Pattern Recog., 2020. 1, 3
- [62] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. In Adv. Neural Inform. Process. Syst., 2024. 2
- [63] Xingran Zhou, Bo Zhang, Ting Zhang, Pan Zhang, Jianmin Bao, Dong Chen, Zhongfei Zhang, and Fang Wen. Cocosnet v2: Full-resolution correspondence learning for image translation. In IEEE Conf. Comput. Vis. Pattern Recog., 2021. 1, 3

## Edicho: Consistent Image Editing in the Wild Appendix

### Appendix

results on the correspondence-guided CFG, where we guide the CFG in both conditional and unconditional noisy latents (instead of the unconditional only). The generation result of the ablated version turns out to be unnatural and has a fragmented look with chaotic textures, suggesting the superiority of our design in correspondence-guided CFG, which avoids deteriorating the prior by merely manipulating the unconditional latents.

- A. Overview We first illustrate the additional implementation details

- in App. B. Additional ablation studies are also included
- in App. C to support the effectiveness of the designed components. We also provide additional correspondence prediction comparisons in App. D. While App. E includes the user studies to validate the effectiveness of the proposed method in terms of user preference. App. F presents the additional qualitative results of the proposed method.

- B. Implementation Details

We adopt DDIM [49] and perform denoising for 50 steps at the resolution of 512. The proposed correspondence-guided denoising strategy is applied from 4th to 40th steps of the denoising process and from the eighth attention layer. λ and γ are respectively set to be 0.8 and 0.9. BrushNet [25] and ControlNet [60] are adopted as the reference network for the task of consistent local and global editing. We conduct on our experiments on a single A6000 GPU.

- C. Additional Ablation Studies

### D. Additional Correspondence Comparisons

Input Implicit (1, 10)

Explicit Implicit (3, 10)

|[Figure 231]<br><br>|
|---|

|[Figure 232]|[Figure 233]|[Figure 234]|
|---|---|---|

- (a)

(c)

- (b)

Input

Explicit Implicit (2, 20) Implicit (4, 20)

|[Figure 235]|
|---|

|[Figure 236]|[Figure 237]|[Figure 238]|
|---|---|---|

Input

Explicit Implicit (1, 39) Implicit (4, 39)

|[Figure 239]|
|---|

|[Figure 240]|[Figure 241]|[Figure 242]|
|---|---|---|

Input Edit #1 Ours Ablated

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

AttentionOutputConditionalNoise

- (a)
- (b)

[Figure 247]

Figure S2. Additional correspondence prediction comparisons. The numbers behind “Implicit” respectively indicate the network layer and denoising step for correspondence prediction.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

We also incorporate additional comparing results of explicit and implicit correspondence prediction. As in the main submission, the explicit ones are predicted with DIFT [50] and implicit ones are computed by querying the attention keys of the target images with the source attention queries following [1, 7]. Specifically, we first perform image inversion [49] and extract queries and keys of input images from the decoder of Stable Diffusion [47], as in [1]. Then the attention features are upsampled and the similarity is calculated based on the attention operation mentioned in the main submission. During this experiment, we change the network layer and denoising step attention feature extraction for comprehensive studies. These ranges are selected considering [1] performs editing during steps of 10-39 and decoder layers of 1-4. As in Fig. S2, the implicit correspondence turns out to be less accurate, degrading the consistent editing, where the numbers (x, y) behind

[Figure 253]

[Figure 254]

[Figure 255]

Figure S1. Additional ablations on the correspondence-guided attention (upper) and CFG (lower).

[Figure 256]

To further validate the effectiveness of the proposed components, we conduct experiments to ablate the correspondence-guided attention (upper) and CFG (lower). As in Fig. S1 (a), we first modify the correspondenceguided attention manipulation to warp the attention outputs instead of the queries. The distorted and inconsistent textures demonstrate that warping attention queries could better preserve the generative prior and achieve consistent results of high quality. Fig. S1 (b) demonstrates ablation

Input

Implicit

Explicit

As mentioned in the main submission, we conduct user studies to obtain results for user preference. For the task of local and global editing, respectively, the 30 individuals are asked to finish up to 20 questions where they choose the best option among the four provided based on overall consistency, generation quality, and instruction-following, resulting in 500 votes for each task. As in Fig. S4, our proposed solution has garnered significantly more preference compared to existing alternatives. In both evaluated tasks, over 60% of the participants opted for our approach. This endorsement from the user validates the practical value of our method as well as highlights its potential impact in realworld applications.

|[Figure 257]<br><br>|
|---|

|[Figure 258]|[Figure 259]|
|---|---|

- (a)

(c)

- (b)

Input

Explicit Implicit

|[Figure 260]<br><br>|[Figure 261]<br><br>|
|---|---|

|[Figure 262]|
|---|

### F. Additional Results

Input Explicit

Implicit

|[Figure 263]<br><br>|
|---|

|[Figure 264]<br><br>|[Figure 265]|
|---|---|

For better understanding, we also incorporate additional qualitative results for consistent local and global editing in Fig. S5. Editing results from the same initial noise are also provided in the figure, indicated as “Fixed Seed”. The inpainted regions for local editing are indicated with light red color.

- Figure S3. Additional correspondence prediction results with attention visualization. Regions with the highest attention weights are outlined with dashed circles.

“Implicit” indicate the layer and denoising step.

We further follow StyleAligned [18] to visualize the attention map for a better understanding of the editing process of implicit methods. We first select a point on the source image, and then compute the attention map based on the aforementioned attention features. The regions with the highest attention weights are outlined with dashed circles in Fig. S3, which suggests the implicit methods would query unreasonable regions that would cause undesirable and inconsistent textures. The attention features of (a), (b), and (c) are respectively extracted from the denoising step of 10, 20, and 35, where the total step is 50.

E. User Studies

[Figure 266]

[Figure 267]

- Figure S4. User study results of consistent local editing (left) and global editing (right).

Input Fixed Seed Ours #1 Ours #2

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

- Figure S5. Additional qualitative results of the proposed method for local (upper three) and global editing (lower three ones). The inpainted regions for local editing are indicated with the light red color. “Fixed Seed” indicates editing results from the same random seed (the same initial noise).

