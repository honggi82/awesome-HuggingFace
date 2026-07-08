# arXiv:2507.08772v2[cs.CV]30Oct2025

## From One to More: Contextual Part Latents for 3D Generation

Shaocong Dong1*, Lihe Ding2*, Xiao Chen2, Yaokun Li2, Yuxin Wang1, Yucheng Wang1, Qi Wang1, Jaehyeok Kim1, Chenjian Gao2, Zhanpeng Huang3, Zibin Wang3,

Tianfan Xue2,4†, Dan Xu1†

1HKUST 2CUHK 3SenseTime Research 4Shanghai AI Laboratory

{sdongae, danxu}@cse.ust.hk, {dl023, tfxue}@ie.cuhk.edu.hk {wangzb02}@gmail.com, {huangzhanpeng}@sensetime.com

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

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

TRELLIS Ours 3D box input and our part generation result

Rodin

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

App. 1: Articulated object generation App. 2: Mini-scene generation App. 3: Part editing

Figure 1. CoPart achieves high-quality part-based 3D generation and supports various applications.

### Abstract

To generate 3D objects, early research focused on multiview-driven approaches relying solely on 2D renderings. Recently, the 3D native latent diffusion paradigm has demonstrated superior performance in 3D generation, because it fully leverages the geometric information provided in ground truth 3D data. Despite its fast development, 3D diffusion still faces three challenges. First, the majority of these methods represent a 3D object by one single latent, regardless of its complexity. This may lead to detail loss when generating 3D objects with multiple complicated parts. Second, most 3D assets are designed parts by parts, yet the current holistic latent representation overlooks the independence of these parts and their interrelationships, limiting

the model’s generative ability. Third, current methods rely on global conditions (e.g., text, image, point cloud) to control the generation process, lacking detailed controllability. Therefore, motivated by how 3D designers create a 3D object, we present a new part-based 3D generation framework, CoPart, which represents a 3D object with multiple contextual part latents and simultaneously generates coherent 3D parts. This part-based framework has several advantages, including: i) reduces the encoding burden of intricate objects by decomposing them into simpler parts, ii) facilitates part learning and part relationship modeling, and iii) naturally supports part-level control. Furthermore, to ensure the coherence of part latents and to harness the powerful priors from foundation models, we propose a novel mutual guidance strategy to fine-tune pre-trained diffusion

models for joint part latent denoising. Benefiting from the part-based representation, we demonstrate that CoPart can support various applications including part-editing, articulated object generation, and mini-scene generation. Moreover, we collect a new large-scale 3D part dataset named Partverse from Objaverse through automatic mesh segmentation and subsequent human post-annotations. By training on the proposed dataset, CoPart achieves promising part-based 3D generation with high controllability. Project page: https://copart3d.github.io.

### 1. Introduction

With the emergence of large-scale 3D datasets [7], many techniques have been proposed to convert the raw 3D data into different representations for generative modeling. Pioneering multi-view driven works [31, 42] render 3D mesh into multi-view images and train multi-view diffusion models or large reconstruction models with only 2D image supervision. These methods can generate consistent multiview images of 3D objects but are poor at recovering highquality geometry since the accurate 3D shape supervision is omitted when converting mesh into multi-view images. Another 3D latent diffusion method CLAY [57] converts 3D meshes into latent tokens by a 3D VAE [20, 55] and trains a latent diffusion model. This method implicitly preserves previously overlooked geometric supervision through 3D occupancy auto-encoding, resulting in improved generation quality.

Despite all these advances of 3D latent diffusion, it still faces three challenges, making it still sub-optimal for 3D generation. First, the current methods treat intricate 3D meshes and simple ones equally, using the same number of tokens. However, the constrained representative ability of 3D VAE will inevitably cause information loss for complex data; and the unbalanced data distribution will make simpler geometries dominate the training process. Second, most 3D designers create complex 3D objects part by part, so they can spend more time adding detailed geometries for each part. On the contrary, state-of-the-art 3D generation algorithms neither utilize the part representations nor explicitly model the relationships between parts, limiting their ability to generate detailed and independent parts. For instance, when generating “a person with a hat,” most algorithms will fuse the head and hat together as a single object within a limited resolution of each local region, leading to low quality. However, users need two distinguishable detailed parts, especially for the face. Third, current methods utilize global conditions (e.g., text, image, point clouds) to control the generation process, which lacks detailed local controllability.

Based on these observations, we demonstrate that all these issues can be addressed through part-based 3D object

representation and generation, as part-based modeling can i) naturally distribute complexity across individual parts, ii) efficiently learn part-level information from the data, and iii) provide detailed control at the part level. Therefore, we propose a novel part-based 3D generation framework, CoPart, which represents a 3D object with multiple Contextual Part latents and generates part latents by learning a joint distribution across diverse 3D data.

While 3D part generation has been explored previously, our solution offers a very distinct perspective. Majority of 3D part generation models either i) are restricted by PartNet [33] categories with limited generalizability [11, 22, 34], or ii) adopt a “top-down” strategy [28], segmenting input images into part patches for individual reconstruction. The latter limits the model’s ability to leverage part information during training and depends heavily on segmentation quality. In contrast, CoPart adopts a “bottomup” framework that directly generates coherent parts and leverage the diverse Objaverse [7] dataset, ensuring greater generalizability.

Specifically, CoPart encodes each 3D part using a geometric token and an image token extracted from the part image, instead of one single global latent. This approach is beneficial for two reasons. First, each part has the complementary geometry and image tokens. Geometry tokens model the detailed shape, while the image tokens not only provide appearance information but also offer semantic cues for understanding part relationships. Second, decoupling geometry from image latents allows us to leverage the capabilities of pre-trained 3D and 2D autoencoders more effectively. For geometric tokens, since the part geometry is normally simpler than object geometry, they can be more efficiently encoded by a 3D part VAE [20]. For image tokens, since each part can be rendered in much higher resolution, 2D diffusion model can generate more detailed textures.

To learn the distribution of both geometry and image tokens of each part, we finetune the diffusion models for 3D geometries [24] and 2D images [4], leveraging their pretrained priors for better generation quality. To ensure both consistency between different parts and between geometric tokens and image tokens, we introduce a mutual guidance diffusion model, inspired by [8]. The mutual guidance facilitates information exchange between different parts as well as between each part’s geometric and image tokens, achieving both part consistency and geometry-image consistency. Furthermore, to eliminate the ambiguity of part order and effectively control the part generation using the input 3D bounding boxes, we propose a novel strategy to encode bounding box conditions to guide part generation. In this way, with the input of bounding boxes and text descriptions, we can generate high-quality 3D objects by decoding and assembling part latents, as shown in the first row of Fig. 1.

Collecting high-quality 3D part data for training is also non-trivial. One option is the PartNet dataset [33], but it only contains 24 categories of objects and has poor textures, restricting model generalizability. Another option is Objaverse [7], which offers more diversity but suffers from inconsistent part labels as different 3D designers prefer different ways to partition an object, often leading to overor under-segmentation. To address this, we first employ a mesh segmentation pipeline to automatically decompose objects into reasonable parts. Then we manually conduct simple post-processing, including filtering low-quality data and grouping the over-segmented parts. Additionally, we utilize a multi-modal vision-language model [25] to generate text prompts for each part. In this way, we obtain a high-quality 3D part dataset with 91k parts for 12k objects.

With the proposed CoPart model trained on our 3D part dataset, it unlocks many various new applications, as shown in the bottom of Fig. 1. First, we can run structure diffusion [29] to obtain bounding boxes and articulation information while using CoPart to generate parts, achieving novel articulated objects generation. Second, we can generate a mini-scene by considering each object in a scene as a part. Thirdly, we can achieve part-based 3D editing by resampling selected part latents. Experimental results also show that our method can generate high-quality 3D objects with more accurate parts compared with the previous works, and also support various applications as discussed above.

### 2. Related Work

#### 2.1. 3D Generation

In contrast to earlier category-specific 3D generation methods [2, 6, 10, 14, 18, 35, 43, 47, 52], contemporary 3D generative models are capable of producing diverse 3D objects conditioned on text or images. DreamFusion [37] and its subsequent works [27, 38, 46] introduce the Score Distillation Sampling (SDS) loss to adapt 2D diffusion models for 3D generation. Multi-view diffusion approaches [30, 31, 42] fine-tune 2D diffusion models to generate multiview consistent images. Meanwhile, LRM [16] trains a large-scale reconstruction model to predict 3D radiance fields from a single image. Based on these 3D foundations, some work can achieve precise part and detail level modifications [9]. More recently, CLAY [57] and its follow-ups [24, 54] directly train 3D-native diffusion models, achieving significantly improved performance.

#### 2.2. Part Generation

StructureNet [32] uses a graph network to understand the relationship between parts while Grass [23] adopting recursive autoencoders for shape structures. DSG-Net [53] proposes disentangled structure and geometry for 3d generation. Other methods [12, 13] employ 3D Gaussian mixture to represent parts. SPAGHETTI [15] and Neu-

ral Template [17] train an auto-encoder to map 3D objects into a part-aware latent space, enabling part-aware editing. SALAD [22] replaces the auto-encoder in SPAGHETTI with a diffusion model, achieving superior performance. DiffFacto [34] learns a controllable part-based point cloud diffusion model. PartNeRF [45] and NeuForm [26] also introduce part-based neural representations. While effective, these methods are limited by their reliance on categoryspecific part data, which restricts their generalizability. Part123 [28] leverages the powerful SAM [21] model to segment multi-view images and perform part-aware reconstruction. Concurrent to our work, PartGen [5], also adopts a ”top-down” strategy by first segmenting parts from multiview images and then performing part completion and reconstruction. However, this approach not only heavily depends on segmentation quality but also struggles with the limited information provided by small segmented patches, which constrains the quality of part reconstruction. In contrast, we propose a ”bottom-up” strategy that directly learns the part distribution from diverse data and jointly generates coherent parts.

### 3. Synchronized Part Latent Diffusion

An effective part-based 3D generative model should be able to generate consistent parts with both high-quality geometry and appearance. However, this is non-trivial for the following three reasons. First, consistency between different parts is hard to ensure. Second, it is not easy to efficiently leverage limited part data to achieve high-quality part-based 3D generation. Third, simultaneously generating parts introduces ambiguity in part ordering.

In this paper, we provide a synchronized part latent diffusion framework to address the above challenges as shown in Fig. 2. In Sec. 3.1, we first introduce our method to represent 3D objects using part latents. Next, in Sec. 3.2.1, we propose an effective framework to synchronize part latent diffusion through mutual guidance, and fine-tune the part latent diffusion model from pre-trained foundation models for efficient part data utilization. Finally, in Sec. 3.2.3, we discuss our approach to inject conditions to resolve the part order ambiguity and enhance controllability.

#### 3.1. Part Representation Encoding

To model the distribution of 3D parts, we decompose a 3D mesh M into part-based 3D representations that preserve both geometric and appearance information from the ground truth data. Our approach utilizes hybrid part latents to represent 3D texture parts through the combination of geometric tokens (encoded by a 3D part VAE) and image tokens (encoded by an image VAE), as detailed below.

Part Geometric Token Encoding. Given a 3D part geometry Mp segmented from M, we sample points P ∈ RS×3 and their corresponding normals Q ∈ RS×3 on the

[Figure 25]

###### (a) Mutual Guidance

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Noise

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

ssss ssss

2D Enc

ssss ssss

- 2D Latent Tokens

- 3D Latent Tokens

[Figure 40]

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

Box Img Token

Global Img Token

[Figure 54]

3D Enc

ssss ssss

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

ControlNet

Pre-trained 2D DiT

Part-0 (global)

###### Add box cond to 2D

[Figure 59]

Residual

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

Noise

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

2D Enc

[Figure 74]

ssss ssss

(b) Add box condition to 2D

###### 2D Latent Tokens

- 2D Cross-Part Attn
- 3D Cross-Part Attn

[Figure 75]

Part Mesh Box Mesh

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

3D Enc

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

- 2D Latent Tokens

- 3D Latent Tokens

[Figure 90]

- Part-1
- Part-2

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

3D Enc

[Figure 98]

Pre-trained 3D DiT

Noise

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

2D Enc

ssss ssss

[Figure 107]

Add box cond to 3D

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Part Token Box Token

[Figure 112]

Cross-Attn

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

3D Enc

###### 3D Latent Tokens

(c) Add box condition to 3D

Cross-Modality Attn

Figure 2. The framework of CoPart operates as follows: Gaussian noise is added to part image and geometric tokens extracted from the VAE, which are then fed into 3D and 2D denoisers. Mutual guidance (a) is introduced to facilitate information exchange between the 3D and 2D modalities (via Cross-Modality Attention) as well as between different parts (via Cross-Part Attention). Additionally, (b) the 3D bounding boxes are treated as cube meshes, and the extracted box tokens are injected into the 3D denoiser through cross-attention. Simultaneously, the boxes are rendered into 2D images and injected into the 2D denoiser via ControlNet.

part mesh’s surface, where S = 4096 is the number of sampled points. Then a 3D part VAE encoder E3D is used to extract 3D part geometric tokens L3D = E3D(P,Q) ∈ RT×D, where T and D represent the token length and dimensions respectively. To enhance the part-level representation learning, we fine-tune the part VAE from a pretrained holistic 3D VAE [24] using our part data. Additionally, we modify the VAE decoder D3D from [24] to predict Flexicube [41] parameters, enabling differentiable rendering. More implementation details can be found in the supplementary material. These designs allow us to incorporate normal and depth rendering losses to supervise the VAE fine-tuning.

Part Image Token Encoding. To encode part appearance, we render the part mesh Mp into multi-view partcentric images {Ok}vk=1. Using a pre-trained image VAE E2D [4], we obtain part image tokens: L2D = {Fk|Fk = E2D(Ok) ∈ RT×D}vk=1 , where v denotes the number of views, and T and D represent the token length and dimensions for each view respectively.

#### 3.2. Synchronized Diffusion

As introduced in Sec. 3.1, a 3D object can be represented by N part latents, comprising geometric tokens and image tokens: {Lp3D,Lp2D}Np=1. To enable effective part generation, we leverage the powerful priors from pre-trained geometric and image diffusion models, and further fine-tune them with our part data. Specifically, we fine-tune a pre-trained 3D latent diffusion [24] to generate geometric part latents Lp3D,

while fine-tuning pre-trained image diffusion models [4] to generate image part latents Lp2D.

To ensure consistent 3D object generation across all diffusion processes, we apply two types of synchronization. First is the inter-part synchronization, which ensures the part consistency. It synchronizes between parts Lib and Ljb, where b ∈ {3D,2D} denotes the modality and i ̸= j indicates different parts. Second is the intra-part synchronization between geometric and appearance representations Li3D and Li2D within the same i-th part, ensuring cross-modality (geometry and image modality) consistency. Details of both synchronization through guidance are described below.

##### 3.2.1. Mutual Guidance

The inter-part synchronization design is inspired by BiDiff [8], which synchronizes a 3D diffusion model and a 2D diffusion model through bidirectional guidance. We further extend this approach by adding mutual guidance between different part latents as well as between 3D and 2D modalities. The original bidirectional guidance, which replies on 2D-to-3D lifting and 3D-to-2D rendering, proves memory-intensive and inefficient for exchanging part-level information. Instead, we adopt a more effective implicit mutual guidance strategy that uses attention to exchange information between different parts and modalities (Fig. 2) (a), as we use Transformer-based diffusion models [36]. Specifically, given noisy part latents Lp,t3D and Lp,t2D = {Fkp,t}vk=1,p = 1,...,N at diffusion timestep t, we define

the intermediate features from the 3D and 2D diffusion as:

Gp = DiT(Lp,t3D, y, t), p = 1,...,N Ip = DiT(Lp,t2D, y, t), p = 1,...,N,

(1)

′×D′ denotes the intermediate 3D features of p-th part, Ip = {Fpk ∈ RT

where Gp ∈ RT

′×D′}vk=1 is the intermediate 2D features of p-th part conatining v views, and y represents additional conditions such as bounding boxes.

To ensure the inter-part consistency, we extend selected self-attention blocks in each modality to attend tokens from all parts:

Gp′ = Attention(Gp,{Gi}Ni=1) Fpk′ = Attention(Fpk,{Fik}Ni=1),

(2)

where Attention(query, key/value) is a standard attention block. This mechanism enables each part to be guided by other parts, facilitating inter-part mutual guidance and synchronization.

Similarly, to ensure cross-modality consistency, we add new attention blocks to exchange information between 3D and 2D features:

Gp′ = Gp + LN(Attention(Gp,{Fpk}vk=1)) Fpk′ = Fpk + LN(Attention(Fpk,Gp)),

(3)

where LN() is a linear layer initialized by zeros for training stability. Furthermore, to guarantee multi-view consistency

- for 2D branch, we also extend some self-attention blocks in 2D diffusion to attend tokens from other views of the same part:

Fpk′ = Attention(Fpk,{Fpk}vk=1). (4)

##### 3.2.2. Global Guidance

To further enhance inter-part consistency, we include a global branch for both 3D and 2D latent diffusions to jointly denoise holistic latents. This global branch functions as an additional “global part” that interacts with other part branches as mentioned in Sec. 3.2.1. In particular, the global branch shares parameters with the part branch, distinguished only by concatenating learnable part-global embeddings to the text embeddings. This architecture enables the network to differentiate between part and global branches, and it further regularizes the fine-tuning process by maintaining global supervision, thereby preventing significant deviation from the original pre-trained weights.

##### 3.2.3. Part Guidance Encoding

One challenge of this design is the part order ambiguity. While we pre-define the part order during training, the network can alter this order during inference, creating a discrepancy between training and inference phases. To alleviate the part order ambiguity and enhance controllability, we introduce two conditions to both the 3D and 2D diffusion denoising processing.

First, we incorporate part-level text prompts to guide the network to distinguish different parts, thereby boosting local semantic controllability. Second, we introduce 3D bounding box conditions for each part as an additional constraint. A naive approach is to use MLPs (multi-layer perceptrons) to encode the coordinates of 3D bounding boxes and use the concatenation of coordinate embeddings with timestep embeddings as conditions. However, learning the correlation between embedded coordinates and actual 3D parts location is not easy.

To solve this challenge in 3D bounding box encoding, we propose a novel strategy to encode 3D bounding boxes by treating each box as a mesh with six surfaces, and then extracting box geometric latents through the pre-trained 3D mesh VAE encoder, as illustrated in Fig. 2 (c). The encoding can be written as

Lpbox = E3D(Pboxp ,Qpbox),p = 1,...N, (5)

where Pboxp and Qpbox present the sampled points and normals on the p-th bounding box surfaces. In this way, we can

encode the 3D bounding box information to the same latent space of the original geometric latents Gp, and simply added to the geometry latent through an additional crossattention block:

Gp′ = Gp + LN(Attention(Gp,Lpbox)), (6) where LN() denotes a linear layer initialized to zeros.

For image tokens, we implement a dual-pathway approach to incorporate bounding boxes. First, we implicitly query bounding box information from 3D geometric tokens into image tokens through the cross-modal attention mechanism described in Eq. 2. Second, we encode 3D bounding boxes into the image latent space to guide the denoising of the global image branch. Specifically, we render 3D boxes into 2D to generate multi-view bounding box wireframe images. These wireframe images are then encoded into latent tokens by an image VAE and integrated into the 2D denoiser through a lightweight ControlNet [56], as illustrated in Fig. 2 (b). It is noteworthy that these bounding box 2D features are exclusively added to the global branch of the 2D diffusion model.

#### 3.3. Refinement

After fine-tuning on the part dataset, the synchronized part latent diffusion is capable of understanding 3D parts and jointly generating consistent part geometric tokens and image tokens. These tokens can be decoded into part meshes and multi-view part-centric images by VAE decoders D3D and D2D:

Mp = D3D(Lp,3D0), Op = D2D(Lp,2D0). (7)

To further improve quality, we leverage the 3D foundation model [49] as an additional enhancer, utilizing both the part

[Figure 121]

[Figure 122]

a firefighter in full gear

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

compose decompose decompose

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Ours

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

zoom in zoom in

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Rodin TRELLIS

Figure 3. Comparison with state-of-the-art 3D generators. CoPart can generate detailed and independent 3D parts.

### 4. Applications

images and geometry generated by our model. While the original pipeline of [49] takes a single image and a generated voxel as input, we found it incapable of understanding diverse 3D parts. Therefore, we modify this approach by replacing the voxels with our detailed part voxels extracted from the generated part meshes, thereby providing essential part geometry prior as follows:

One major advantage of our part-based representation of CoPart is that it enables us to directly achieve various applications without further training. These applications include part-based editing (Sec. 4.1), articulated object generation (Sec. 4.2), mini-scene generation and long part sequence sampling (Sec. 4.3).

#### 4.1. Part-based Editing

Mp′ = R(Op,V p), V p = Voxelize(T (Mp)), (8)

To enable selective part modification while keeping other parts unchanged, we design an inference-time resampling strategy. Specifically, given a mesh parts sequence {Mp′}Np=1 either from CoPart sampling or segmented from an existing mesh, we denote the parts that need to be edited as {Mp′}p∈C, where C is the selected index. To maintain the remaining parts unchanged during sampling while allowing them to provide information for new parts, we first encode the remaining part mesh back into contextual part latents:

where R represents the stage II enhancer, and T denotes a transformation to normalize Mp to [−1,1]. We can further enhance each part efficiently and assemble the parts by using the inverse transformation T −1:

M′ = {T −1(Mp′)}Np=1. (9)

#### 3.4. Optimization Loss

We supervise both the 3D and image branches of CoPart by the denoising loss in diffusion models:

Lp,3D0′ = E3D(Sample(Mp′)) Lp,2D0′ = E2D(Render(Mp′)).

(11)

N

1 N

3D,ϵp3d,t∥ϵp3d−N3d(Lp,t3D,Lp,t2D,t)∥22),

(ELp,t

Loss3D =

Then during each timestep of the new editing sampling process, we directly replace the noisy part latents

p=1

(10)

N

1 N

2D,ϵp2d,t∥ϵp2d−N2d(Lp,t3D,Lp,t2D,t)∥22),

{Lp,t3D, Lp,t2D}p/∈C by adding noise to {Lp,3D0′, Lp,2D0′}p/∈C: Lp,t3D = √αtLp,3D0′ + √1 − αtϵ3d,p ∈/ C Lp,t2D = √αtLp,2D0′ + √1 − αtϵ2d,p ∈/ C,

(ELp,t

Loss2D =

p=1

(12)

where N represents the denoiser for 3D and 2D and ϵ denotes Gaussian noise.

a black military fighter aircraft

a wheel chair

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Ours SALAD

Figure 4. Comparison with part-based generator SALAD.

Thor’s hammer A table of dinner

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

texture edit geometry edit

[Figure 152]

Rose gold hammer Pointed warhammer

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

(a) Part editing (b) Mini scene generation

Figure 5. Qualitative results of part editing and mini scene generation.

where αt is noise schedule and ϵ is random Gaussian noise

fer to the supplementary video for the visualization of generated articulated objects.

- for 3D or 2D. Thus, we can sample additional part latents from pure Gaussian noise while incorporating information from the fixed one by:

- 4.3. Mini-scene and Long Sequence Generation Our approach extends naturally to mini-scene generation, as scenes can be represented as layouts of bounding boxes, aligning with our box-guided part generation where each part corresponds to an object. Our training data includes mini-scenes, enabling direct scene generation through specified boxes and text (Fig. 5 (b)). For complex objects requiring long part sequences, GPU memory constraints during training limit the maximum number of parts N=8. We address it by adopting the strategy from Sec. 4.1: fixing some sampled part latents while replacing others with new box conditions sampled from Gaussian noise. This enables the generation of longer sequences without memory issues. More details can be found in the supplementary materials.
- 5. PartVerse Dataset

Lp,t3D−1,Lp,t2D−1=N({ Lp,t3D, Lp,t2D}p/∈C,{Lp,t3D,Lp,t2D}p∈C),

(13) where N is the diffusion denoiser. In this way, we can modify the text prompts to edit the selected parts as shown in Fig. 5 (a).

#### 4.2. Articulated Object Generation

The generation of articulated objects involves two key components: i) articulation generation, which includes the bounding boxes indicating the position of each part, and ii) part generation. To achieve this, we leverage an offthe-shelf method [29] to generate the bounding boxes along with their articulation relationships. Subsequently, we utilize CoPart to populate each bounding box with coherent parts based on text prompts. This approach enables the creation of novel articulated objects, such as an avocado swivel chair (Fig. 1), which cannot be realized using previous holistic generation methods. Additionally, the partbased generation approach provides the flexibility to manually define articulation information for each part. Please re-

To enhance the generalizability of CoPart, we introduce PartVerse, a new diverse 3D part dataset comprising 91k high-quality parts from 12k objects with detailed text descriptions. Unlike previous part datasets such as PartNet [33] which only contains 24 categories of daily objects, our PartVerse, curated and annotated from Objaverse [7],

[Figure 157]

Figure 6. PartVerse dataset processing pipeline. We follow the pipeline of “raw data - mesh segment algorithm - human post correction generate text caption” to produce part-level data.

Table 1. Quantitative comparison with SOTA methods. CLIP (N-T) and CLIP (I-T) [39, 57] gauge the geometry alignment of normal maps with the input text and similarity of render images with the input text, respectively. In addition, ULIP-T [50] was also experimented and user-prefer study were conducted. † Our method takes 12s when the number of parts is one.

Whole-aware Part-aware

Method

Time

CLIP (N-T) CLIP (I-T) ULIP-T CLIP (N-T) CLIP (I-T) ULIP-T

Shap-E [19] 0.1546 0.1607 0.1054 0.0875 0.1043 0.0795 3s Unique3D [48] 0.1865 0.2037 0.1528 0.1062 0.1380 0.1032 16s CraftMan [24] 0.1887 0.1966 0.1476 0.1026 0.1271 0.0997 8s Rodin [1] 0.2042 0.2416 0.1785 0.1425 0.1571 0.1244 Trellis [49] 0.2071 0.2360 0.1751 0.1274 0.1455 0.1119 10s Ours 0.2010 0.2387 0.1743 0.1607 0.1768 0.1355 65s†

exhibits enhanced diversity in 175 categories and realistic textures, significantly improving the model’s ability to generate high-quality 3D parts. As shown in Fig. 6, we provide an overview of the data collection and annotation pipeline used to create Partverse.

Automatic Mesh Segmentation. 3D artists follow a specific modeling pipeline when creating 3D objects, typically creating them part by part. We can restore this information from the raw 3D data. However, the original modeling steps might not always match semantic part boundaries - for instance, some artists might create textured surfaces as separate elements. A pre-labeling algorithm based on SAM-2 [40] and Samesh [44] was constructed using 3D model creation priors combined with semantic segmentation, which balances the mesh faces connectivity and visual semantic during segmentation. This algorithm can allow us to preliminarily obtain semantic parts. We specifically adjusted parameters to favor over-segmentation rather than under-segmentation, since it’s easier for annotators to merge extra segments than to split insufficient ones.

Human post-annotation. After initial segmentation, human reviewers first remove low-quality data including overly complex or unsuitable 3D objects for splitting. Using a Blender-based [3] annotation platform, they then refine the segmentation by merging over-segmented or splitting under-segmented parts according to guidelines: ensur-

ing clear part semantics and maximizing symmetry in part distribution. Based on these annotations, complete textured objects are finally split into individual part objects with textures preserved.

Part-level text caption. We also created textual descriptions for each part, covering appearance features, shape characteristics, and part-whole relationships. The process starts by rendering multi-view images of both complete objects and individual parts. We select the view showing maximum visible overlap between part and whole, then combine the object image with the highlighted part using a bounding box. This composite image is input to a vision-language model (VLM) to generate descriptions of the component and its relationship to the complete object.

### 6. Experimental Evaluation

This section presents our experimental results. For better visualization, please refer to our supplementary video.

Implementation details. We initialize our 2D and 3D denoisers using Pixart-α [4] and CraftMan [24], and fine-tune them on PartVerse dataset with 4 NVIDIA A100 GPUs. We set the batch size to 1 and limit the maximum part number to 8. We also perform random selecting or padding during training. Moreover, we adopt a progressive training strategy, which is detailed in the supplementary material.

##### Comparison with state-of-the-art 3D generators. We

compare CoPart with leading holistic 3D generators in Fig. 3. More results can be found in the supplementary materials. It is evident that CoPart outperforms state-of-the-art methods, particularly in the quality of small parts, owing to its part-based representation.

Quantitative comparison. As shown in Tab. 1, we conduct quantitative comparisons following [57]. For the Image-to-3D model, we first generate corresponding images from given texts by Flux.1 [51]. Different from [57], half of the test cases we use are part-aware, such as “a rifle stock”. This is reasonable, since a truly general 3D generation model should be able to handle all types. In addition, ULIP [50] was used in evaluation. As shown in Tab. 2, we also conducted a user study with 51 diverse participants from different professions by collecting their preferences for generating textured mesh. These results highlights the advantages of our part-based generation approach in producing decomposable and high-quality 3D assets. More details can be found in the supplementary materials.

Table 2. User study (%preference).

Method Whole-aware Part-aware Rodin [1] 33.3% 25.5%

PartGen [5] 11.8% 13.7% Ours 54.9% 60.8%

Comparison with part-based generation methods. We compare CoPart with the accessible state-of-the-art part generator SALAD [22]. Fig. 4 shows that CoPart can generate diverse objects with detailed parts while SALAD is constrained to generate shapes in PartNet distribution. More comparisons can be found in the supplementary materials.

Ablation study of global guidance. We ablate the effect of global guidance in Fig. 7. The results demonstrate that global guidance significantly enhances part coherence, especially in appearance.

Ablation study of refinement. We ablate the effect of refinement (Sec. 3.3), as shown in Fig. 8. The results show that providing both part images (Fig. 8 (a)) and geometries (Fig. 8 (b)) is essential for the enhancer to accurately comprehend part shapes. By integrating both modalities, the enhancer effectively optimizes the parts and enhances overall performance.

a yellow M4 rifle

[Figure 158]

[Figure 159]

(a) w/o global guidance (b) w global guidance

Figure 7. Ablation of global guidance.

### 7. Discussion of Part Assembly

As depicted in Sec. 3.3, the part latents are decoded in absolute positions and then normalized to perform refinement. After an inverse-normalization, the refined part meshes can be relocated to the 3D bonding boxes for assembling. Our precise strategy for injecting bounding box information generally ensures effective combinations of part meshes.

[Figure 160]

a pink airplane nose propeller

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

(a) generated multi-view image (b) generated part mesh (c) refine w geo (d) refine w/o geo

Figure 8. Ablation of refinement module.

However, assembly errors, such as “mesh clipping”, can occur when the user provides incorrect bounding boxes. We will explore techniques to optimize user-provided bounding boxes in future work.

### 8. Conclusion

We present CoPart for high-quality and diverse 3D part generation. Specifically, we utilize mutual guidance to ensure coherent part latent denoising and introduce 3D box conditions to eliminate part ambiguity. Furthermore, a larger scale 3D part-aware dataset is firstly collected from Objaverse, which can be widely used for various tasks. Our method outperforms SoTA results. We also discuss the limitations of our method in the supplementary material.

### References

- [1] Rodin. https://hyper3d.ai/. 8, 9
- [2] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3d point clouds. In ICML, 2018. 3
- [3] Blender Foundation. Blender, 2023. 8
- [4] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,

2023. 2, 4, 8

- [5] Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. Partgen: Part-level 3d generation and reconstruction with multi-view diffusion models. arXiv preprint arXiv:2412.18608, 2024. 3, 9
- [6] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In CVPR, 2019. 3
- [7] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2, 3, 7
- [8] Lihe Ding, Shaocong Dong, Zhanpeng Huang, Zibin Wang, Yiyuan Zhang, Kaixiong Gong, Dan Xu, and Tianfan Xue. Text-to-3d generation with bidirectional diffusion using both 2d and 3d priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5115–5124, 2024. 2, 4
- [9] Shaocong Dong, Lihe Ding, Zhanpeng Huang, Zibin Wang, Tianfan Xue, and Dan Xu. Interactive3d: Create what you want by interactive 3d generation. In CVPR, 2024. 3

- [10] Lin Gao, Jie Yang, Tong Wu, Yu-Jie Yuan, Hongbo Fu, YuKun Lai, and Hao Zhang. Sdm-net: Deep generative network for structured deformable mesh. ACM Transactions on Graphics (ToG), 38:1–15, 2019. 3
- [11] Lin Gao, Jie Yang, Tong Wu, Yu-Jie Yuan, Hongbo Fu, YuKun Lai, and Hao Zhang. Sdm-net: Deep generative network for structured deformable mesh. ACM Transactions on Graphics (TOG), 38(6):1–15, 2019. 2
- [12] Kyle Genova, Forrester Cole, Daniel Vlasic, Aaron Sarna, William T Freeman, and Thomas Funkhouser. Learning shape templates with structured implicit functions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7154–7164, 2019. 3
- [13] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna, and Thomas Funkhouser. Local deep implicit functions for 3d shape. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4857–4866,

2020. 3

- [14] Philipp Henzler, Niloy J. Mitra, and Tobias Ritschel. Escaping plato’s cave: 3d shape from adversarial rendering. In ICCV, 2019. 3
- [15] Amir Hertz, Or Perel, Raja Giryes, Olga Sorkine-Hornung, and Daniel Cohen-Or. Spaghetti: Editing implicit shapes through part aware generation. ACM Transactions on Graphics (TOG), 41(4):1–20, 2022. 3
- [16] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 3
- [17] Ka-Hei Hui, Ruihui Li, Jingyu Hu, and Chi-Wing Fu. Neural template: Topology-aware reconstruction and disentangled generation of 3d meshes. In CVPR, 2022. 3
- [18] Moritz Ibing, Gregor Kobsik, and Leif Kobbelt. Octree transformer: Autoregressive 3d shape generation on hierarchically structured sequences. arXiv preprint arXiv:2111.12480, 2021. 3
- [19] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 8
- [20] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2
- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 3
- [22] Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, and Minhyuk Sung. Salad: Part-level latent diffusion for 3d shape generation and manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14441– 14451, 2023. 2, 3, 9
- [23] Jun Li, Kai Xu, Siddhartha Chaudhuri, Ersin Yumer, Hao Zhang, and Leonidas Guibas. Grass: Generative recursive autoencoders for shape structures. ACM Transactions on Graphics (TOG), 36(4):1–14, 2017. 3
- [24] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity

- mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 2, 3, 4, 8
- [25] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 3

- [26] Connor Lin, Niloy Mitra, Gordon Wetzstein, Leonidas J Guibas, and Paul Guerrero. Neuform: Adaptive overfitting for neural shape editing. NeuIPS, 2022. 3
- [27] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, 2023. 3
- [28] Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. Part123: part-aware 3d reconstruction from a single-view image. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2, 3
- [29] Jiayi Liu, Hou In Ivan Tam, Ali Mahdavi-Amiri, and Manolis Savva. Cage: Controllable articulation generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17880–17889, 2024. 3, 7
- [30] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In ICCV, 2023. 3
- [31] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 2, 3
- [32] Kaichun Mo, Paul Guerrero, Li Yi, Hao Su, Peter Wonka, Niloy Mitra, and Leonidas J Guibas. Structurenet: Hierarchical graph networks for 3d shape generation. arXiv preprint arXiv:1908.00575, 2019. 3
- [33] Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A largescale benchmark for fine-grained and hierarchical part-level 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 909–918, 2019. 2, 3, 7
- [34] George Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shi-Min Hu, Ke Li, and Leonidas Guibas. Difffacto: Controllable part-based 3d point cloud generation with cross diffusion. In CVPR, 2023. 2, 3
- [35] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, , and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In CVPR, 2019. 3
- [36] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022. 4
- [37] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR,

2022. 3

- [38] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123:

- One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 3
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 8
- [40] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 8
- [41] Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. Flexible isosurface extraction for gradient-based mesh optimization. ACM Trans. Graph., 42(4), 2023. 4
- [42] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2, 3
- [43] Edward Smith and David Meger. Deep unsupervised learning using nonequilibrium thermodynamics. In CoRL, 2017. 3
- [44] George Tang, William Zhao, Logan Ford, David Benhaim, and Paul Zhang. Segment any mesh: Zero-shot mesh part segmentation via lifting segment anything 2 to 3d. arXiv preprint arXiv:2408.13679, 2024. 8
- [45] Konstantinos Tertikas, Despoina Paschalidou, Boxiao Pan, Jeong Joon Park, Mikaela Angelina Uy, Ioannis Emiris, Yannis Avrithis, and Leonidas Guibas. Generating part-aware editable 3d shapes without 3d supervision. In CVPR, 2023. 3
- [46] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. NeurIPS, 2024. 3
- [47] Jiajun Wu, Chengkai Zhang, Tianfan Xue, Bill Freeman, and Josh Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In NeurIPS, 2016. 3
- [48] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 8
- [49] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024. 5, 6, 8
- [50] Le Xue, Mingfei Gao, Chen Xing, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In CVPR, 2023. 8, 9

- [51] Chenglin Yang, Celong Liu, Xueqing Deng, Dongwon Kim, Xing Mei, Xiaohui Shen, and Liang-Chieh Chen. 1.58-bit flux. arXiv preprint arXiv:2412.18653, 2024. 9
- [52] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. Pointflow: 3d point cloud generation with continuous normalizing flows. In ICCV,

2019. 3

- [53] Jie Yang, Kaichun Mo, Yu-Kun Lai, Leonidas J Guibas, and Lin Gao. Dsg-net: Learning disentangled structure and geometry for 3d shape generation. ACM Transactions on Graphics (TOG), 42(1):1–17, 2022. 3
- [54] Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, Lifu Wang, Zhuo Chen, Sicong Liu, Yuhong Liu, Yong Yang, Di Wang, Jie Jiang, and Chunchao Guo. Tencent hunyuan3d-1.0: A unified framework for text-to-3d and image-to-3d generation, 2024. 3
- [55] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–16, 2023. 2
- [56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 5
- [57] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 2, 3, 8, 9

