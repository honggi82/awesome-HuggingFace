## PartGen: Part-level 3D Generation and Reconstruction with Multi-View Diffusion Models

# arXiv:2412.18608v2[cs.CV]29Dec2024

Minghao Chen1,2 Roman Shapovalov2 Iro Laina1 Tom Monnier2 Jianyuan Wang1,2 David Novotny2 Andrea Vedaldi1,2 1Visual Geometry Group, University of Oxford 2Meta AI

silent-chen.github.io/PartGen

Part-Aware Text-to-3D

Part-Aware Image-to-3D

Selected Parts Selected Parts

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Input Image

Text prompt

[Figure 9]

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

> A beagle in

a detective’s outfit

[Figure 22]

[Figure 23]

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

[Figure 41]

[Figure 42]

3D Decomposition

Selected Parts

3D Part Editing

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Unstructured 3D

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

Original “Magician hat” “Brown hat with police badge"

“Red hat with blue texture”

Figure 1. We introduce PartGen, a pipeline that generates compositional 3D objects similar to a human artist. It can start from text, an image, or an existing, unstructured 3D object. It consists of a multi-view diffusion model that identifies plausible parts automatically and another that completes and reconstructs them in 3D, accounting for their context, i.e., the other parts, to ensure that they fit together correctly. Additionally, PartGen enables 3D part editing based on text instructions, enhancing flexibility and control in 3D object creation.

#### Abstract

Text- or image-to-3D generators and 3D scanners can now produce 3D assets with high-quality shapes and textures. These assets typically consist of a single, fused representation, like an implicit neural field, a Gaussian mixture, or a mesh, without any useful structure. However, most applications and creative workflows require assets to be made of several meaningful parts that can be manipulated independently. To address this gap, we introduce PartGen, a novel approach that generates 3D objects composed of meaningful parts starting from text, an image, or an unstructured 3D object. First, given multiple views of a 3D object, generated

Work completed during Minghao’s internship at Meta.

or rendered, a multi-view diffusion model extracts a set of plausible and view-consistent part segmentations, dividing the object into parts. Then, a second multi-view diffusion model takes each part separately, fills in the occlusions, and uses those completed views for 3D reconstruction by feeding them to a 3D reconstruction network. This completion process considers the context of the entire object to ensure that the parts integrate cohesively. The generative completion model can make up for the information missing due to occlusions; in extreme cases, it can hallucinate entirely invisible parts based on the input 3D asset. We evaluate our method on generated and real 3D assets and show that it outperforms segmentation and part-extraction baselines by a large margin. We also showcase downstream applications such as 3D part editing.

#### 1. Introduction

High-quality textured 3D assets can now be obtained through generation from text or images [12, 14, 18, 51, 56, 58, 76, 83], or through photogrammetry techniques [15, 63, 89]. However, the resulting objects are unstructured, consisting of a single, monolithic representation, such as an implicit neural field, a mixture of Gaussians, or a mesh. This is not good enough in a professional setting, where the structure of an asset is also of paramount importance. While there are many aspects to the structure of a 3D object (e.g., the mesh topology), parts are especially important as they enable reuse, editing and animation.

In this paper, we thus consider the problem of obtaining structured 3D objects that are formed by a collection of meaningful parts, akin to the models produced by human artists. For example, a model of a person may be decomposed into its clothes and accessories, as well as various anatomical features like hair, eyes, teeth, limbs, etc. However, if the object is generated or scanned, different parts are usually ‘fused’ together, missing the internal surfaces and the part boundaries. This means that physically detachable parts appear glued together, with a jarring effect. Furthermore, parts carry important information and functionality that those models lack. For example, different parts may have distinct animations or different materials. Parts can also be replaced, removed, or edited independently. For instance, in video games, parts are often reconfigured dynamically, e.g., to represent a character picking up a weapon or changing clothes. Due to their semantic meaning, parts are also important for 3D understanding and applications like robotics, embodied AI, and spatial intelligence [48, 53].

Inspired by these requirements, we introduce PartGen, a method to upgrade existing 3D generation pipelines from producing unstructured 3D objects to generating objects as compositions of meaningful 3D parts. To do this, we address two key questions: (1) how to automatically segment a 3D object into parts, and (2) how to extract high-quality, complete 3D parts even when these are only partially—or not at all—visible from the exterior of the 3D object.

Crucially, both part segmentation and completion are highly ambiguous tasks. First, since different artists may find it useful to decompose the same object in different ways, there is no ‘gold-standard’ segmentation for any given 3D object. Hence, a segmentation method should model the distribution of plausible part segmentations rather than a single one. Second, current 3D reconstruction and generation methods only model an object’s visible outer surface, omitting inner or occluded parts. Therefore, decomposing an object into parts often requires completing these parts or even entirely hallucinating them.

To model this ambiguity, we base part segmentation and reconstruction on 3D generative models. We note that most state-of-the-art 3D generation pipelines [12, 14, 18, 39, 51,

56, 58, 76, 83] start by generating several consistent 2D views of the object, and then apply a 3D reconstruction network to those images to recover the 3D object. We build upon this two-stage scheme to address both part segmentation and reconstruction ambiguities.

In the first stage, we cast part segmentation as a stochastic multi-view-consistent colouring problem, leveraging a multi-view image generator fine-tuned to produce colourcoded segmentation maps across multiple views of a 3D object. We do not assume any explicit or even deterministic taxonomy of parts; the segmentation model is learned from a large collection of artist-created data, capturing how 3D artists decompose objects into parts. The benefits of this approach are twofold. First, it leverages an image generator which is already trained to be view-consistent. Second, a generative approach allows for multiple plausible segmentations by simply re-sampling from the model. We show that this process results in better segmentation than that obtained by fine-tuning a model like SAM [35] or SAM2 [70] for the task of multi-view segmentation: while the latter can still be used, our approach better captures the artists’ intent.

For the second problem, namely reconstructing a segmented part in 3D, an obvious approach is to mask the part within the available object views, and then use a 3D reconstructor network to recover the part in 3D. However, when the part is heavily occluded, this task amounts to amodal reconstruction, which is highly ambiguous and thus badly addressed by the deterministic reconstructor network. Instead, and this is our core contribution, we propose to tune another multi-view generator to complete the views of the part while accounting for the context of the object as a whole. In this manner, the parts can be reconstructed reliably even if they are only partially visible, or even not visible, in the original input views. Furthermore, the resulting parts fit together well and, when combined, form a coherent 3D object.

We show that PartGen can be applied to different input modalities. Starting from text, an image, or a areal-world 3D scan, PartGen can generate 3D assets with meaningful parts. We assess our method empirically on a large collection of 3D assets produced by 3D artists or scanned, both quantitatively and qualitatively. We also demonstrate that PartGen can be easily extended to the 3D part editing task.

- 2. Related Work
- 3D generation from text and images. The problem of generating 3D assets from text or images has been thoroughly studied in the literature. Some authors have built generators from scratch. For instance, CodeNeRF [30] learns a latent code for NeRF in a Variational Autoencoder fashion, and Shap-E [31] and 3DGen [21] does so using latent diffusion, PC2 [55] and Point-E [62] diffuse a point cloud, and MosaicSDF a semi-explicit SDF-based representation [95]. However, 3D training data is scarce, which

makes it difficult to train text-based generators directly.

DreamFusion [65] demonstrated for the first time that 3D assets can be extracted from T2I diffusion models with Score Distillation Sampling (SDS) loss. Variants of DreamFusion explore representations like hash grids [41, 66], meshes [41] and 3D Gaussians (3DGS) [8, 79, 97], tweaks to the SDS loss [27, 85, 87, 105], conditioning on an input image [54, 66, 78, 80, 99], and regularizing normals or depth [68, 74, 78].

Other works focus on improving the 3D awareness of the T2I model, simplifying extracting a 3D output and eschewing the need for slow SDS optimization. Inspired by 3DIM [88], Zero-1-to-3 [47] fine-tunes the 2D generator to output novel views of the object. Two-stage approaches [6, 9, 18, 22, 23, 25, 45, 49, 50, 56, 74, 81, 86, 90, 92–94] take the output of a text- or image-to-multi-view model that generates multiple views of the object and reconstruct the latter using multi-view reconstruction methods like NeRF [59] or 3DGS [32]. Other approaches reduce the number of input views generated and learn a fast feed-forward network for 3D reconstruction. Perhaps the most notable example is Instant3D [39] based on the Large Reconstruction Model (LRM) [26]. Recently, there are works focusing on 3D compositional generation [11, 40, 64, 106]. D3LL [17] learns 3D object composition through distilling from a 2D T2I generator. ComboVerse [7] starts from a single image, but mostly at the levels of different objects instead of their parts, performs single-view inpainting and reconstruction, and uses SDS optimization for composition.

3D segmentation. Our work decomposes a given 3D object into parts. Several works have considered segmenting 3D objects or scenes represented in an unstructured manner, lately as neural fields or 3D Gaussian mixtures. Semantic-NeRF [102] was the first to fuse 2D semantic segmentation maps in 3D with neural fields. DFF [36] and N3F [84] propose to map 2D features to 3D fields, allowing their supervised and unsupervised segmentation. LERF [33] extends this concept to language-aware features like CLIP [69]. Contrastive Lift [2] considers instead instance segmentation, fusing information from several independently-segmented views using a contrastive formulation. GARField [34] and OminiSeg3D [98] consider that concepts exist at different levels of scale, which they identify with the help of SAM [35]. LangSplat [67] leverages both CLIP and SAM, creating distinct 3D language fields to model each SAM scale explicitly, while N2F2 [3] automates binding the correct scale to each concept. Neural Part Priors [4] completes and decomposes 3D scans with learned part priors in a test-time optimization manner. Finally, Uni3D [103] learns a ‘foundation’ model for 3D point clouds that can perform zero-shot segmentation.

Primitive-based representations. Some authors proposed to represent 3D objects as a mixture of primitives [100], which can be seen as related to parts, although they are usually non-semantic. For example, SIF [19] represents an occupancy function as a 3D Gaussians mixture. LDIF [20] uses the Gaussians to window local occupancy functions implemented as neural fields [57]. Neural Template [28] and SPAGHETTI [1] learn to decompose shapes in a similar manner using an auto-decoding setup. SALAD [37] uses SPAGHETTI as the latent representation for a diffusionbased generator. PartNeRF [82] is conceptually similar, but builds a mixture of NeRFs. NeuForm [42] and DiffFacto [61] learn representations that afford part-based control. DBW [60] decomposes real-world scenes with textured superquadric primitives.

Semantic part-based representations. Other authors have considered 3D parts that are semantic. PartSLIP [46] and PartSLIP++ [104] use vision-language model to segment objects into parts using point clouds as representation. Part123 [44] is conceptually similar to Contrastive Lift [2], but applied to object than scenes, and to the output of a monocular reconstruction network instead of NeRF.

In this paper, we address a problem different from the ones above. We generate compositional 3D objects from various modalities using multi-view diffusion models for segmentation and completion. Parts are meaningfully segmented, fully reconstructed, and correctly assembled. We handle the ambiguity of these tasks in a generative way.

#### 3. Method

This section introduces PartGen, our framework for generating 3D objects that are fully decomposable into complete 3D parts. Each part is a distinct, human-interpretable, and self-contained element, representing the 3D object compositionally. PartGen can take different modalities as input (text prompts, image prompts, or 3D assets) and performs part segmentation and completion by repurposing a powerful multi-view diffusion model for these two tasks. An overview of PartGen is shown in Figure 2.

The rest of the section is organised as follows. In Sec. 3.1, we introduce the necessary background on multiview diffusion and how PartGen can be applied to text, image, or 3D model inputs briefly. Then, in Secs. 3.2 to 3.4 we describe how PartGen automatically segments, completes, and reconstructs meaningful parts in 3D.

##### 3.1. Background on 3D generation

First, we provide essential background on multi-view diffusion models for 3D generation [39, 74, 76]. These methods usually adopt a two-stage approach to 3D generation.

In the first stage, given a prompt y, an image generator Φ outputs several 2D views of the object from different van-

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

| | | |
|---|---|---|
| |Network| |
| |Reconstruction| |
| | | |

[Figure 82]

or

[Figure 83]

<Text Prompt>

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Conditional Multi-View Generator

ReconstructionNetwork

CompletionNetwork

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Muti-ViewPart

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Segmentation

[Figure 104]

Multi-View

[Figure 105]

[Figure 106]

[Figure 107]

+

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Grid View Segmentation Map

Completed Parts 3D Parts and Object

- Figure 2. Overview of PartGen. Our method begins with text, single images, or existing 3D objects to obtain an initial grid view of the object. This view is then processed by a diffusion-based segmentation network to achieve multi-view consistent part segmentation. Next, the segmented parts, along with contextual information, are input into a multi-view part completion network to generate a fully completed view of each part. Finally, a pre-trained reconstruction model generates the 3D parts.

tage points. Depending on the nature of y, the network Φ is either a text-to-image (T2I) model [39, 74] or a imageto-image (I2I) one [73, 86]. These are fine-tuned to output a single ‘multi-view’ image I ∈ R3×2H×2W, where views from the four cardinal directions around the object are arranged into a 2 × 2 grid. This model thus provides a probabilistic mapping I ∼ p(I | Φ,y). The 2D views I are subsequently passed to a Reconstruction Model (RM) [39, 76, 91] Ψ, i.e., a neural network that reconstructs the 3D object L in both shape and appearance. Compared to direct 3D generation, this two-stage paradigm takes full advantage of an image generation model pre-trained on internet-scale 2D data.

This approach is general and can be applied with various implementations of image-generation and reconstruction models. Our work in particular follows a setup similar to AssetGen [76]. Specifically, we obtain Φ by finetuning a pre-trained text-to-image diffusion model with an architecture similar to Emu [13], a diffusion model in a 8-channel latent space, the mapping to which is provided by a specially trained variational autoencoder (VAE). The detailed fine-tuning strategy can be found in Sec. 4.4 and supplementary material. When the input is a 3D model, we render multiple views to form the grid view. For the RM Ψ we use LightplaneLRM [5], trained on our dataset.

##### 3.2. Multi-view part segmentation

The first major contribution of our paper is a method for segmenting an object into its constituent parts. Inspired by multi-view diffusion approaches, we frame object decomposition into parts as a multi-view segmentation task, rather than as direct 3D segmentation. At a high-level, the goal is to map I to a collection 2D masks M1,...,MS ∈ {0,1}2H×2W, one for each visible part of the object. Both image I and masks Mi are multi-view grids.

Addressing 3D object segmentation through the lens of multi-view diffusion offers several advantages. First, it al-

lows us to repurpose existing multi-view models Φ, which, as described in Sec. 3.1, are already pre-trained to produce multi-view consistent generations in the RGB domain. Second, it integrates easily with established multi-view frameworks. Third, decomposing an object into parts is an inherently non-deterministic, ambiguous task as it depends on the desired verbosity level, individual preferences, and artistic intent. By learning this task with probabilistic diffusion models, we can effectively capture and model this ambiguity. We thus train our model on a curated dataset of artistcreated 3D objects, where each object L is annotated with a possible decomposition into 3D parts, L = (S1,...,SS). The dataset details are provided in Sec. 3.5.

Consider that the input is a multi-view image I, and the output is a set of multi-view part masks M1,M2,...,MS. To finetune our multi-view image generators Φ for mask prediction, we quantize the RGB space into Q different colors c1,...,cQ ∈ [0,1]3. For each training sample L = (Sk)Sk=1, we assign colors to the parts, mapping part Sk to color cπ

, where π is a random permutation on {1,...,Q} (we assume that Q ≥ S). Given this mapping, we render the segmentation map as a multi-view RGB image C ∈ [0,1]3×2H×2W (Fig. 4). Then, we fine-tune Φ to (1) take as conditioning the multi-view image I, and (2) to generate the color-coded multi-view segmentation map C, hence sampling a distribution C ∼ p(C | Φseg,I).

k

This approach can produce alternative segmentations by simply re-running Φseg, which is stochastic. It further exploits the fact that Φseg is stochastic to discount the specific ‘naming’ or coloring of the parts, which is arbitrary. Naming is a technical issue in instance segmentation which usually requires ad-hoc solutions, and here is solved ‘for free’.

To extract the segments at test time, we sample the image C and simply quantize it based on the reference colors c1,...,cQ, discarding parts that contain only a few pixels.

by the inpainting setup in [71]. We apply the pre-trained VAE separately to the masked image I ⊙ M and context image I, yielding 2 × 8 channels, and stack them with the 8D noise image and the unencoded part mask M to obtain the 25-channel input to the diffusion model. Example results are shown in Figure 5.

Whole object Part 1 Part 2

Part N

|[Figure 115]|
|---|
|[Figure 116]|
|[Figure 117]|

|[Figure 118]|[Figure 119]|
|---|---|
|[Figure 120]|[Figure 121]|
|[Figure 122]|[Figure 123]|

|[Figure 124]|
|---|
|[Figure 125]|
|[Figure 126]|

…

…

##### 3.4. Part reconstruction

Given a multi-view part image J, the final step is to reconstruct the part in 3D. Because the part views are now complete and consistent, we can simply use the RM to obtain a predicted reconstruction Sˆ = Ψ(J) of the part. We found that the model does not require special finetuning to move from objects to their parts, so any good quality reconstruction model can be plugged into our pipeline directly.

…

- Figure 3. Training data. We obtain a dataset of 3D objects decomposed into parts from assets created by artists. These come ‘naturally’ decomposed into parts according to the artist’s design.

##### 3.5. Training data

Implementation details. The network Φseg has the same architecture as the network Φ with some changes to allow conditioning on the multi-view image I: we encode it into latent space with the VAE and stack it with the noised latent as the input to the diffusion network.

To train our models, we require a dataset of 3D models consisting of multiple parts. We have built this dataset from a collection of 140k 3D-artist generated assets that we licensed for AI training from a commercial source. Each asset L is stored as a GLTF scene that contains, in general, several watertight meshes (S1,...,SS) that often align with semantic parts due to being created by a human who likely aimed to create an editable asset. Example objects from the dataset are shown in Fig. 3. We preprocess data differently for each of the three models we fine tuned.

##### 3.3. Contextual part completion

The method so far has produced a multi-view image I of the 3D object along with 2D segments M1,M2,...,MS. What remains is to convert those into the full 3D part reconstructions. Given a mask M, in principle we could simply submit the masked image I⊙M to the RM Ψ to obtain a 3D reconstruction of the part, i.e., Sˆ = Ψ(I ⊙M). However, in multi-view images, some parts can be heavily occluded by other parts and, in extreme cases, entirely invisible. While we could train the RM to handle such occlusions directly, in practice this does not work as part completion is inherently a stochastic problem, whereas the RM is deterministic.

Multi-view generator data. To train the multi-view generator models Φ, first of all, we have to render the target multiview images I consisting of 4 views to the full object. Following Instant3D[39], we rendered shaded colours I from the 4 views from the orthogonal azimuths and 20◦ elevation and arranged them in a 2 × 2 grid. In case of text conditioning, training data consists of the pairs {(In,yn)}Nn=1 of multi-view images and their text captions Following AssetGen [76], we choose 10k highest quality assets and generated their text captions using CAP3D-like pipeline [52] that used LLAMA3 model [16]. In case of image conditioning, we use all 140k models, and the conditioning yn comes in form of single renders from a randomly sampled direction (not just one of the four used in In).

To handle this ambiguity, we repurpose yet again the multi-view generator Φ, this time to perform part completion. The latter model is able to generate a 3D object from text or single image, so, properly fine-tuned, it should be able to hallucinate any missing portion of a part.

Formally, we consider fine-tuning Φ to sample a view J ∼ p(J | I ⊙ M), mapping the masked image I ⊙ M to the completed multi-view image J of the part. However, we note that sometimes parts are barely visible, so the masked

Part segmentation and completion data. To train the part segmentation and completion networks, we need to additionally render the multi-view part images and their depth maps. Since different creators have different ideas on part decomposition, we filter the dataset to avoid having excessively granular parts which likely lack semantic meaning. To this end, we first cull the parts that take less than 5% of the object volume, and then remove the assets that have more than 10 parts or consist of a single monolithic part. This results in the dataset of 45k objects contain the total of 210k parts. Given the asset L = (S1,...,SS), we ren-

- image I ⊙M provides very little information. Furthermore, we need the generated part to fit well with the other parts and the whole object. Hence, we provide to the model also the un-masked image I for context. Thus, condition p(J | I ⊙ M,I,M) on the masked image I ⊙ M, the unmasked image I, and the mask M. The importance of the context I increases with the extent of the occlusion.

Implementation details. The network architecture resembles that of Sec. 3.2, but extends the conditioning, motivated

Input SAM2 SAM2-finetune SAM2–4 view Part123 Ours Sample 1 Ours Sample 2 Ours Sample 3

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

[Figure 148]

[Figure 149]

[Figure 150]

- Figure 4. Examples of automatic multi-view part segmentations. By running our method several times, we obtain different segmentations, covering the space of artist intents.

Context Incomplete Part GT Ours Sample 1 Ours Sample 2 Ours Sample 3

Mask

Automatic Seeded Method mAP50↑ mAP75↑ mAP50↑ mAP75↑ Part123 [44] 11.5 7.4 10.3 6.5

|[Figure 151]|[Figure 152]| |[Figure 153]|
|---|---|---|---|
|[Figure 154]|[Figure 155]| |[Figure 156]|
|[Figure 157]| | | |
|[Figure 158]| | | |

|[Figure 159]|[Figure 160]|[Figure 161]|[Figure 162]|
|---|---|---|---|
| | | | |
|[Figure 163]|[Figure 164]|[Figure 165]|[Figure 166]|
| |[Figure 167]|[Figure 168]|[Figure 169]|
| |[Figure 170]|[Figure 171]|[Figure 172]|

SAM2† [70] 20.3 11.8 24.6 13.1 SAM2∗ [70] 37.4 27.0 44.2 30.1 SAM2 [70] 35.3 23.4 41.4 27.4 PartGen (1 sample) 45.2 32.9 44.9 33.5 PartGen (5 samples) 54.2 33.9 51.3 32.9 PartGen (10 samples) 59.3 38.5 53.7 35.4

- Table 1. Segmentation results. SAM2∗ is fine-tuned our data and SAM2† is fine-tuned for multi-view segmentation.

[Figure 174]

|[Figure 176]<br><br>Figure 5. with blue plausible|Qualita<br><br>borders outputs|tive are the across|
|---|---|---|

|of part Our<br><br>runs|[Figure 177]<br><br>comple lgorithm<br><br>. Even|[Figure 178]<br><br>tion. Th<br><br>produce if given|[Figure 179]<br><br>e images s various<br><br>an empty|
|---|---|---|---|

F results w inputs. p different part, PartGen attempts to generate internal structures inside the object, such as sand or inner wheels.

der a set of multi-view images {Js}Ss=1 (shown in Fig. 3) and the corresponding depth maps {δs}Ss=1 from the same viewpoints as above.

The segmentation diffusion network is trained on the dataset of pairs {(In,Mn)}Nn=1, where the segmentation map M = [Mk]Sk=1 is a stack of multi-view binary part masks Mk ∈ {0,1}2H×2W. Each mask shows the pixels where the appropriate part is visible in I: Mi,jk = [k = argminlδi,jl ], where k,l ∈ {1,...,S} and brackets denote Iverson brackets. The part completion network is trained on the dataset of triplets {(In′,Jn′,Mn′)}N

##### 4.1. Part segmentation

Evaluation protocol. We set up two settings for the segmentation tasks. One is automatic part segmentation, where the input is the multi-view image I and requires the method to output all parts of the object M1,...,MS. The other is seeded segmentation, where we assume that users give a point as an additional input for a specific mask. Now the segmentation algorithm is regarded as a black box Mˆ = A(I) mapping the multi-view image I to a ranked list of N part segmentations (which can in general partially overlap). This ranked list is obtained by scoring candidate regions and removing redundant ones. See the sup. mat. for more details. We then match these segments to the ground-truth segments Mk and report mean Average Precision (mAP). This precision can be low in practice due to the inherent ambiguity of the problem: many of the parts predicted by the algorithm will not match any particular artist’s choice.

′

n′=1. All the components are produces in the way described above.

#### 4. Experiments

Evaluation protocol. We first individually evaluate the two main components of our pipeline, namely part segmentation (Sec. 4.1) and part completion and reconstruction (Sec. 4.2). We then evaluate how well the decomposed reconstruction matches the original object (Sec. 4.3). For all experiments, we use the held out 100 objects from the dataset described in Sec. 3.5.

(a) 3D Decomposition

Reconstructed 3D Input Reconstructed 3D Example Parts Input Reconstructed 3D Example Parts

Input Example Parts

|[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>|[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]|[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]|
|---|---|---|
| | | |

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

View completion J 3D reconstruction S Method Compl. Multi-view Context CLIP↑ LPIPS↓ PSNR↑ CLIP↑ LPIPS↓ PSNR↑

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Oracle (Jˆ = J) GT — — 1.0 0.0 ∞ 0.957 0.027 18.91 PartGen (Jˆ = B(I ⊙ M,I)) ✓ ✓ ✓ 0.974 0.015 21.38 0.936 0.039 17.16

image_to_repr_3d_id00017_a_chihuahua_wearing_a_tutu_images_render

w/o context† (Jˆ = B(I ⊙ M)) ✓ ✓ ✗ 0.951 0.028 16.80 0.923 0.046 14.83 single view‡ (Jˆv = B(Iv ⊙ Mv,Iv)) ✓ ✗ ✓ 0.944 0.031 15.92 0.922 0.051 13.25

None (Jˆ = I ⊙ M) ✗ — — 0.932 0.039 13.24 0.913 0.059 12.32

- Table 2. Part completion results. We first evaluate view part completion by computing scores w.r.t. the ground-truth multi-view part

- image J. Then, we evaluate 3D part reconstruction by reconstructing each part S and rendering it. See text for details.

[Figure 246]

- (a) Part-Aware Text-to-3D Generated 3D Input Generated 3D Example Parts Input Generated 3D Example Parts

Input Example Parts

- (b) Part-Aware Image-to-3D Generated 3D Input Generated 3D Example Parts Input Generated 3D Example Parts

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

Input Example Parts

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

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

- (c) 3D Decomposition Reconstructed 3D Input Reconstructed 3D Example Parts Input Reconstructed 3D Example Parts

Input Example Parts

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

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

> A cat

[Figure 327]

> A gummy

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

> A chihuahua wearing a tutu

[Figure 334]

[Figure 335]

wearing a lion costume

bear driving a

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

convertible

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

|[Figure 431]|
|---|

[Figure 432]

[Figure 433]

[Figure 434]

- Figure 6. Examples of applications. PartGen can effectively generate or reconstruct 3D objects with meaningful and realistic parts in different scenarios: a) Part-aware text-to-3D generation; b) Part-aware image-to-3D generation; c) 3D decomposition.

Baselines. We consider the original and fine-tuned SAM2 [70] as our baselines for multi-view segmentation. We fine-tune SAM2 in two different ways. First, we finetune SAM2’s mask decoder on our dataset, given the ground truth masks and randomly selected seed points for different views. Second, we concatenate the four orthogonal views in a multi-view image I and fine-tune SAM2 to predict the multi-view mask M (in this case, the seed point randomly falls in one of the views). SAM2 produces three regions for each input image and seed point. For automatic segmentation, we seed SAM2 with a set of query points spread over the object, obtaining three different regions for each seed point. For seeded segmentation, we simply return the regions that SAM2 outputs for the given seed point. We also provide a comparison with recent work, Part123 [44].

marily because of the ambiguity of the segmentation task, which is better captured by our generator-based approach. We further provide qualitative results in Fig. 4.

##### 4.2. Part completion and reconstruction

We utilize the same test data as in Sec. 4.1, forming tuples (S,I,Mk,Jk) consisting of the 3D object part S, the full multi-view image I, the part mask Mk and the multi-view image Jk of the part, as described in Section 3.5. We choose one random part index k per model, and will omit it from the notation below to be more concise.

Evaluation protocol. The completion algorithm and its baselines are treated as a black box Jˆ = B(I ⊙ M,I) that predicts the completed multi-view image Jˆ. We then compare Jˆ to the ground-truth render J using Peak Signal to Noise Ratio (PSNR) of the foreground pixels, Learned Perceptual Image Patch Similarity (LPIPS) [101], and CLIP similarity [69]. The latter is an important metric since the

Results. We report the results in Tab. 1. As shown in the table, mAP results for our method are much higher than others, including SAM2 fine-tuned on our data. This is pri-

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

Sˆk = Φ(Jˆk), and reassemble the 3D object Lˆ by merging the 3D parts {Sˆ1,...,SˆN}. We then compare Lˆ =

k Φ(Jˆk) to the unsegmented reconstruction Lˆ = Φ(I) using the same protocol as for parts.

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

Results. Table 3 shows that our method achieves performance comparable to directly reconstructing the objects using the RM (Lˆ = Φ(I)), with the additional benefit of producing the reconstruction structured into parts, which are useful for downstream applications such as editing.

Original “White T-shirt with

“Hawaii shirt” “Cloth with colorful

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

logo”

texture”

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

##### 4.4. Applications

Original “Black magic hat” “White hat” “Cowboy hat”

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

Part-aware text-to-3D generation. First, we apply PartGen to part-aware text-to-3D generation. We train a textto-multi-view generator similar to [76], which takes a text prompt as input and outputs a grid of four views. For illustration, we use the prompts from DreamFusion [65]. As shown in Fig. 6, PartGen can effectively generate 3D objects with distinct and completed parts, even in challenging cases with heavy occlusions, such as the gummy bear. Additional examples are provided in the supp. mat.

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

Original “pink cup with square bottom”

“Green cup with cute logo”

“Yellow cup with a smile on it”

- Figure 7. 3D part editing. We can edit the appearance and shape of the 3D objects with text prompt.

Method CLIP↑ LPIPS↓ PSNR↑

Part-aware image-to-3D generation. Next, we consider part-aware image-to-3D generation using images from [23, 90]. Building upon the text-to-multi-view generator, we further fine-tune the generator to accept images as input with a strategy similar to [96]. Further training details are provided in supp. mat. Results are shown in Fig. 6 demonstrating that PartGen is successful in this case as well.

PartGen (Lˆ = k Φ(Jˆk)) 0.952 0.065 20.33 Unstructured (Lˆ = Φ(I)) 0.955 0.064 20.47

Table 3. Model reassembling result. The quality of 3D reconstruction of the object as a whole is close to that of the partbased compositional reconstruction, which proves that the predicted parts fit together well.

Real-world 3D object decomposition. PartGen can also decompose real-world 3D objects. We show this using objects from Google Scanned Objects (GSO) [15] for this purpose. Given a 3D object from GSO, we render different views to obtain a an image grid and then apply PartGen as above. The last row of Figure 6 shows that PartGen can effectively decompose real-world 3D objects too.

completion task is highly ambiguous, and thus evaluating semantic similarity can provide additional insights. We also evaluate the quality of the reconstruction of the predicted completions by comparing the reconstructed object part Sˆ = Φ(Jˆ) to the ground-truth part S using the same metrics, but averaged after rendering the part to four random novel viewpoints.

3D part editing. Finally, we show that once the 3D parts are decomposed, they can be further modified through text input. As illustrated in Fig. 7, a variant of our method enables effective editing of the shape and texture of the parts based on textual prompts. The details of the 3D editing model are provided in supplementary materials.

Results. We compare our part completion algorithm (Jˆ = B(I ⊙ M,I)) to several baselines and the oracle, testing using no completion (Jˆ = I ⊙ M), omitting context (Jˆ = B(I ⊙ M)), completing single views independently (Jˆv = B(Iv ⊙ Mv,Iv)), and the oracle (Jˆ = J). The latter provides the upper-bound on the part reconstruction performance, where the only bottleneck is the RM.

#### 5. Conclusion

As shown in the table Tab. 2, our model largely surpasses the baselines. Both joint multi-view reasoning and contextual part completion are important for good performance. We further provide qualitative results in Fig. 5.

We have introduced PartGen, a novel approach to generate or reconstruct compositional 3D objects from text, images, or unstructured 3D objects. PartGen can reconstruct in 3D parts that are even minimally visible, or not visible at all, utilizing the guidance of a specially-designed multiview diffusion prior. We have also shown several application of PartGen, including text-guided part editing. This is a promising step towards the generation of 3D assets that are more useful in professional workflows.

##### 4.3. Reassembling parts

Evaluation protocol. Starting from multi-view image I of a 3D object L, we run the segmentation algorithm to obtain segmentation (Mˆ 1,...,Mˆ S), reconstruct each 3D part as

#### References

- [1] Hertz Amir, Perel Or, Giryes Raja, Sorkine-Hornung Olga, and Cohen-Or Daniel. SPAGHETTI: editing implicit shapes through part aware generation. In ACM Transactions on Graphics, 2022. 3
- [2] Yash Sanjay Bhalgat, Iro Laina, Joao F. Henriques, Andrea Vedaldi, and Andrew Zisserman. Contrastive Lift: 3D object instance segmentation by slow-fast contrastive fusion. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2023. 3
- [3] Yash Sanjay Bhalgat, Iro Laina, Joao F. Henriques, Andrew Zisserman, and Andrea Vedaldi. N2F2: Hierarchical scene understanding with nested neural feature fields. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 3
- [4] Aleksei Bokhovkin and Angela Dai. Neural part priors: Learning to optimize part-based object completion in rgbd scans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9032– 9042, 2023. 3
- [5] Ang Cao, Justin Johnson, Andrea Vedaldi, and David Novotny. Lightplane: Highly-scalable components for neural 3d fields. arXiv preprint arXiv:2404.19760, 2024. 4, 2
- [6] Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. Generative novel view synthesis with 3D-aware diffusion models. In Proc. ICCV, 2023. 3
- [7] Yongwei Chen, Tengfei Wang, Tong Wu, Xingang Pan, Kui Jia, and Ziwei Liu. Comboverse: Compositional 3d assets creation using spatially-aware diffusion guidance. arXiv preprint arXiv:2403.12409, 2024. 3
- [8] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3D using Gaussian splatting. arXiv, 2309.16585, 2023. 3
- [9] Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, and Huaping Liu. V3D: Video diffusion models are effective 3D generators. arXiv, 2403.06738, 2024. 3
- [10] Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models. arXiv preprint arXiv:2407.15886, 2024. 1
- [11] Dana Cohen-Bar, Elad Richardson, Gal Metzer, Raja Giryes, and Daniel Cohen-Or. Set-the-scene: Global-local training for generating controllable nerf scenes. In Proc. ICCV Workshops, 2023. 3
- [12] CSM. CSM text-to-3D cube 2.0, 2024. 2
- [13] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam S. Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, Matthew Yu, Abhishek Kadian, Filip Radenovic, Dhruv Mahajan, Kunpeng Li, Yue Zhao, Vladan Petrovic, Mitesh Kumar Singh, Simran Motwani, Yi Wen, Yiwen Song, Roshan Sumbaly, Vignesh Ramanathan, Zijian He, Peter Vajda, and Devi Parikh. Emu: Enhancing image generation models using photogenic needles in a haystack. CoRR, abs/2309.15807, 2023. 4, 1

- [14] Deemos. Rodin text-to-3D gen-1 (0525) v0.5, 2024. 2
- [15] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 2, 8
- [16] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aur´elien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozi`ere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gr´egoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, and Kevin Stone. The Llama 3 herd of models. arXiv, 2407.21783, 2024. 5
- [17] Dave Epstein, Ben Poole, Ben Mildenhall, Alexei A. Efros, and Aleksander Holynski. Disentangled 3d scene generation with layout learning, 2024. 3
- [18] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, and Ben Poole. CAT3D: create anything in 3d with multi-view diffusion models. arXiv, 2405.10314, 2024. 2, 3
- [19] Kyle Genova, Forrester Cole, Daniel Vlasic, Aaron Sarna, William T. Freeman, and Thomas Funkhouser. Learning shape templates with structured implicit functions. In Proc. CVPR, 2019. 3
- [20] Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna, and Thomas A. Funkhouser. Local deep implicit functions for 3D shape. In Proc. CVPR, 2020. 3
- [21] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas Oguz. 3DGen: Triplane latent diffusion for textured mesh generation. corr, abs/2303.05371, 2023. 2
- [22] Junlin Han, Jianyuan Wang, Andrea Vedaldi, Philip Torr, and Filippos Kokkinos. Flex3d: Feed-forward 3d generation with flexible reconstruction model and input view curation. arXiv preprint arXiv:2410.00890, 2024. 3

- [23] Junlin Han, Filippos Kokkinos, and Philip Torr. Vfusion3d: Learning scalable 3d generative models from video diffusion models. In European Conference on Computer Vision, pages 333–350. Springer, 2025. 3, 8
- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proc. NeurIPS, 2020. 1
- [25] Lukas H¨ollein, Aljaz Bozic, Norman M¨uller, David Novotn´y, Hung-Yu Tseng, Christian Richardt, Michael Zollh¨ofer, and Matthias Nießner. ViewDiff: 3D-consistent image generation with text-to-image models. In Proc. CVPR, 2024. 3
- [26] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large reconstruction model for single image to 3D. In Proc. ICLR, 2024. 3
- [27] Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, and Lei Zhang. Dreamtime: An improved optimization strategy for text-to-3D content creation. CoRR, abs/2306.12422, 2023. 3
- [28] Ka-Hei Hui, Ruihui Li, Jingyu Hu, and Chi-Wing Fu. Neural template: Topology-aware reconstruction and disentangled generation of 3d meshes. In Proc. CVPR, 2022. 3
- [29] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier J. H´enaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and Jo˜ao Carreira. Perceiver IO: A general architecture for structured inputs & outputs. In Proc. ICLR, 2022. 1
- [30] Wonbong Jang and Lourdes Agapito. CodeNeRF: Disentangled neural radiance fields for object categories. In Proc. ICCV, 2021. 2
- [31] Heewoo Jun and Alex Nichol. Shap-E: Generating conditional 3D implicit functions. arXiv, 2023. 2
- [32] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaussian Splatting for real-time radiance field rendering. Proc. SIGGRAPH, 42(4), 2023. 3
- [33] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: language embedded radiance fields. In Proc. ICCV, 2023. 3
- [34] Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg, Matthew Tancik, and Angjoo Kanazawa. Garfield: Group anything with radiance fields. arXiv.cs, abs/2401.09419, 2024. 3
- [35] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. In Proc. CVPR,

2023. 2, 3

- [36] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing NeRF for editing via feature field distillation. arXiv.cs, 2022. 3
- [37] Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, and Minhyuk Sung. SALAD: part-level latent diffusion for 3D shape generation and manipulation. In Proc. ICCV, 2023. 3

- [38] D. Larlus, G. Dorko, D. Jurie, and B. Triggs. Pascal visual object classes challenge. In Selected Proceeding of the first PASCAL Challenges Workshop, 2006. 3
- [39] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3D: Fast text-to-3D with sparse-view generation and large reconstruction model. Proc. ICLR, 2024. 2, 3, 4, 5, 1
- [40] Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Text-driven 3d editing via focal-fusion assembly, 2023. 3
- [41] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3D: High-resolution text-to-3D content creation. arXiv.cs, abs/2211.10440, 2022. 3
- [42] Connor Lin, Niloy Mitra, Gordon Wetzstein, Leonidas J. Guibas, and Paul Guerrero. NeuForm: adaptive overfitting for neural shape editing. In Proc. NeurIPS, 2022. 3
- [43] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5404–5411,

2024. 1

- [44] Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. Part123: Part-aware 3d reconstruction from a single-view image. arXiv, 2405.16888, 2024. 3, 6, 7
- [45] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3D mesh in 45 seconds without pershape optimization. In Proc. NeurIPS, 2023. 3
- [46] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. PartSLIP: low-shot part segmentation for 3D point clouds via pretrained imagelanguage models. In Proc. CVPR, 2023. 3
- [47] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3D object. In Proc. ICCV, 2023. 3
- [48] Weiyu Liu, Jiayuan Mao, Joy Hsu, Tucker Hermans, Animesh Garg, and Jiajun Wu. Composable part-based manipulation. In CoRL 2023, 2023. 2
- [49] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. SyncDreamer: Generating multiview-consistent images from a single-view image. arXiv, 2309.03453, 2023. 3
- [50] Xiaoxiao Long, Yuanchen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3D: Single image to 3D using cross-domain diffusion. arXiv.cs, abs/2310.15008, 2023. 3
- [51] LumaAI. Genie text-to-3D v1.0, 2024. 2
- [52] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. arXiv preprint arXiv:2306.07279, 2023. 5
- [53] Oier Mees, Jessica Borja-Diaz, and Wolfram Burgard. Grounding language with visual affordances over unstruc-

- tured data. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK, 2023. 2
- [54] Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and Andrea Vedaldi. RealFusion: 360 reconstruction of any object from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [55] Luke Melas-Kyriazi, Christian Rupprecht, and Andrea Vedaldi. PC2: Projection-conditioned point cloud diffusion for single-image 3d reconstruction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [56] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and Filippos Kokkinos. IM-3D: Iterative multiview diffusion and reconstruction for high-quality 3D generation. In Proceedings of the International Conference on Machine Learning (ICML), 2024. 2, 3
- [57] L. Mescheder, M. Oechsle, M. Niemeyer, S. Nowozin, and A. Geiger. Occupancy Networks: Learning 3D reconstruction in function space. In Proc. CVPR, 2019. 3
- [58] Meshy. Meshy text-to-3D v3.0, 2024. 2
- [59] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In Proc. ECCV, 2020. 3
- [60] Tom Monnier, Jake Austin, Angjoo Kanazawa, Alexei Efros, and Mathieu Aubry. Differentiable blocks world: Qualitative 3d decomposition by rendering primitives. Advances in Neural Information Processing Systems, 36: 5791–5807, 2023. 3
- [61] George Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shi-Min Hu, Ke Li, and Leonidas Guibas. DiffFacto: controllable part-based 3D point cloud generation with cross diffusion. In Proc. ICCV, 2023. 3
- [62] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-E: A system for generating 3D point clouds from complex prompts. arXiv.cs, abs/2212.08751, 2022. 2
- [63] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Carl Yuheng Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception, 2023. 2
- [64] Ryan Po and Gordon Wetzstein. Compositional 3d scene generation using locally conditioned diffusion. ArXiv, abs/2303.12218, 2023. 3
- [65] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. In Proc. ICLR, 2023. 3, 8
- [66] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, and Bernard Ghanem. Magic123: One image to high-quality 3D object generation using both 2D and 3D diffusion priors. arXiv.cs,

abs/2306.17843, 2023. 3

- [67] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3D language Gaussian splatting. In Proc. CVPR, 2024. 3
- [68] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3D. arXiv.cs, abs/2311.16918, 2023. 3
- [69] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proc. ICML, pages 8748–8763, 2021. 3, 7, 1
- [70] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. arXiv, 2408.00714, 2024. 2, 6, 7
- [71] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proc. CVPR,

2022. 5

- [72] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 1
- [73] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multiview diffusion base model. arXiv.cs, abs/2310.15110, 2023. 4
- [74] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3D generation. In Proc. ICLR, 2024. 3, 4
- [75] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11987–11997, 2023. 2
- [76] Yawar Siddiqui, Filippos Kokkinos, Tom Monnier, Mahendra Kariya, Yanir Kleiman, Emilien Garreau, Oran Gafni, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and David Novotny. Meta 3D Asset Gen: Text-to-mesh generation with high-quality geometry, texture, and PBR materials. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 3, 4, 5, 8
- [77] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proc. ICLR, 2021. 1
- [78] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. DreamCraft3D: Hierarchical 3D generation with bootstrapped diffusion prior. arXiv.cs, abs/2310.16818, 2023. 3
- [79] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. DreamGaussian: Generative gaussian splatting for efficient 3D content creation. arXiv, 2309.16653,

2023. 3

- [80] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-It-3D: Highfidelity 3d creation from A single image with diffusion prior. arXiv.cs, abs/2303.14184, 2023. 3
- [81] Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka Furukawa, and Rakesh Ranjan. MVDiffusion++: A dense high-resolution multi-view diffusion model for single or sparse-view 3d object reconstruction. arXiv, 2402.12712,

2024. 3

- [82] Konstantinos Tertikas, Despoina Paschalidou, Boxiao Pan, Jeong Joon Park, Mikaela Angelina Uy, Ioannis Z. Emiris, Yannis Avrithis, and Leonidas J. Guibas. PartNeRF: Generating part-aware editable 3D shapes without 3D supervision. arXiv.cs, abs/2303.09554, 2023. 3
- [83] TripoAI. Tripo3D text-to-3D, 2024. 2
- [84] Vadim Tschernezki, Iro Laina, Diane Larlus, and Andrea Vedaldi. Neural Feature Fusion Fields: 3D distillation of self-supervised 2D image representation. In Proceedings of the International Conference on 3D Vision (3DV), 2022. 3
- [85] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich. Score Jacobian chaining: Lifting pretrained 2D diffusion models for 3D generation. In Proc. CVPR, 2023. 3
- [86] Peng Wang and Yichun Shi. ImageDream: Image-prompt multi-view diffusion for 3D generation. In Proc. ICLR,

2024. 3, 4

- [87] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. ProlificDreamer: Highfidelity and diverse text-to-3D generation with variational score distillation. arXiv.cs, abs/2305.16213, 2023. 3
- [88] Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. In Proc. ICLR, 2023. 3
- [89] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu. Omniobject3d: Largevocabulary 3d object dataset for realistic perception, reconstruction and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [90] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. InstantMesh: efficient 3D mesh generation from a single image with sparse-view large reconstruction models. arXiv, 2404.07191, 2024. 3, 8
- [91] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. GRM: Large gaussian reconstruction model for efficient 3D reconstruction and generation. arXiv, 2403.14621,

2024. 4

- [92] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, and Kai Zhang. DMV3D: Denoising multiview diffusion using 3D large reconstruction model. In Proc. ICLR, 2024. 3
- [93] Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, and Hongdong Li. ConsistNet: Enforcing 3D consistency for multiview images diffusion. arXiv.cs, abs/2310.10343, 2023.

- [94] Yunhan Yang, Yukun Huang, Xiaoyang Wu, Yuan-Chen Guo, Song-Hai Zhang, Hengshuang Zhao, Tong He, and Xihui Liu. DreamComposer: Controllable 3D object generation via multi-view conditions. arXiv.cs, abs/2312.03611,

2023. 3

- [95] Lior Yariv, Omri Puny, Natalia Neverova, Oran Gafni, and Yaron Lipman. Mosaic-SDF for 3D generative models. arXiv.cs, abs/2312.09222, 2023. 2
- [96] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arxiv:2308.06721,

2023. 8, 1

- [97] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. GaussianDreamer: Fast generation from text to 3D gaussian splatting with point cloud priors. arXiv.cs, abs/2310.08529,

2023. 3

- [98] Haiyang Ying, Yixuan Yin, Jinzhi Zhang, Fan Wang, Tao Yu, Ruqi Huang, and Lu Fang. Omniseg3d: Omniversal 3d segmentation via hierarchical contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20612–20622, 2024. 3
- [99] Wangbo Yu, Li Yuan, Yan-Pei Cao, Xiangjun Gao, Xiaoyu Li, Long Quan, Ying Shan, and Yonghong Tian. HiFi-123: Towards high-fidelity one image to 3D content generation. arXiv.cs, abs/2310.06744, 2023. 3
- [100] Guanqi Zhan, Qingnan Fan, Kaichun Mo, Lin Shao, Baoquan Chen, Leonidas J Guibas, Hao Dong, et al. Generative 3d part assembly via dynamic graph learning. Advances in Neural Information Processing Systems, 33:6315–6326,

2020. 3

- [101] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. CVPR, pages 586–595, 2018. 7
- [102] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J. Davison. In-place scene labelling and understanding with implicit scene representation. In Proc. ICCV,

2021. 3

- [103] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3D: Exploring unified 3D representation at scale. In Proc. ICLR, 2024. 3
- [104] Yuchen Zhou, Jiayuan Gu, Xuanlin Li, Minghua Liu, Yunhao Fang, and Hao Su. PartSLIP++: enhancing low-shot 3d part segmentation via multi-view instance segmentation and maximum likelihood estimation. arXiv, 2312.03015,

2023. 3

- [105] Junzhe Zhu and Peiye Zhuang. HiFA: High-fidelity text-to-3D with advanced diffusion guidance. CoRR, abs/2305.18766, 2023. 3
- [106] Yan Zizheng, Zhou Jiapeng, Meng Fanpeng, Wu Yushuang, Qiu Lingteng, Ye Zisheng, Cui Shuguang, Chen Guanying, and Han Xiaoguang. Dreamdissector: Learning disentangled text-to-3d generation from 2d diffusion priors. ECCV,

2024. 3

## PartGen: Part-level 3D Generation and Reconstruction with Multi-View Diffusion Models

### Supplementary Material

Input Target

This supplementary material contains the following parts:

[Figure 459]

[Figure 460]

[Figure 461]

> A red cylindrical cup with a smooth

- • Implementation Details. Detailed descriptions of the training and inference settings for all models used in PartGen are provided.
- • Additional Experiment Details. We describe the detailed evaluation metrics employed in the experiments and provide additional experiments.
- • Additional Examples. We include more outputs of our method, showcasing applications with part-aware text-to3D, part-aware image-to-3D, real-world 3D decomposition, and iteratively adding parts.
- • Failure Case. We analyse the modes of of failure of PartGen.
- • Ethics and Limitation. We provide a discussion on the ethical considerations of data and usage, as well as the limitations of our method.

matte finish a flat bottom

[Figure 462]

[Figure 463]

[Figure 464]

> A red necktie made of smooth shiny material

Input image Generated caption

Input image Generated caption

[Figure 465]

[Figure 466]

> A dark brown, tapered, wooden leg

> A dead tree trunk with a rough, brown

with a smooth, glossy surface and a pointed tip

texture and several thin, bare branches.

Figure 8. 3D part editing and captioning examples. The top section illustrates training examples for the editing network, where a mask, a masked image, and text instructions are provided as conditioning to the diffusion network, which fills in the part based on the given textual input. The bottom section demonstrates the input for the part captioning pipeline. Here, a red circle and highlights are used to help the large vision-language model (LVLM) identify and annotate the specific part.

#### A. Implementation Details

We provide the details of training used in PartGen (Appendices A.1 to A.4). In addition, we provide the implementation details for the applications: for part composition (Appendix A.5) and for part editing (Appendix A.6).

##### A.1. Text-to-multi-view generator

We fine-tune the text-to-multi-view generator starting with a pre-trained text-to-image diffusion model trained on billions of image-text pairs that uses an architecture and data similar to Emu [13]. We change the target image to a grid of 2×2 views as described in Section 3.5 following Instant 3D [39] via v-prediction [72] loss. The resolution of each view is 512 × 512, resulting in the total size of 1024 × 1024. To avoid the problem of the cluttered background mentioned in [39], we rescale the noise scheduler to force a zero terminal signal-to-noise ratio (SNR) following [43]. We use the DDPM scheduler with 1000 steps [24] for training. During the inference, we use DDIM [77] scheduler with 250 steps. The model is trained with 64 H100 GPUs with a total batch size of 512 and a learning rate 10−5 for 10k steps.

##### A.2. Image-to-multi-view generator

Building on the text-to-multi-view generator, we further fine-tune the model to accept images as input conditioning instead of text. The text condition is removed by setting it to a default null condition (an empty string). We concatenate the conditional image to the noised image along the spatial

dimension, following [10]. Additionally, inspired by IPadapter [96], we introduce another cross-attention layer into the diffusion model. The input image is first converted into tokens using CLIP [69], then reprojected into 157 tokens of dimension 1024 using a Perceiver-like architecture [29]. To train the model, we utilize all 140k 3D models of our data collection, selecting conditional images with random elevation and azimuth but fixed camera distance and field of view. We use the DDPM scheduler with 1000 steps [24], rescaled SNR, and v-prediction for training. Training is conducted with 64 H100 GPUs, a batch size of 512, and a learning rate of 10−5 over 15k steps.

##### A.3. Multi-view segmentation network

To obtain the multi-view segmentation network, we also fine-tune the pre-trained text-to-multi-view model. The input channels are expanded from 8 to 16 to accommodate the additional image input, where 8 corresponds to the latent dimension of the VAE used in our network. We create segmentation-image pairs as inputs. The training setup follows a similar recipe to that of the image-to-multi-view

Recall@k (IoU > 0.5) Comparison

0.7

0.6

Recall@k(IoU>0.5)

0.5

0.4

ours (1 sample) ours (5 sample) ours (10 sample)

0.3

SAM2 (finetuned)

SAM2 (original) SAM2 (4 views)

0.2

1 3 5 10

k

Recall@k (IoU > 0.75) Comparison

0.45

0.40

Recall@k(IoU>0.75)

0.35

0.30

0.25

ours (1 sample) ours (5 sample) ours (10 sample)

0.20

SAM2 (finetuned)

0.15

SAM2 (original) SAM2 (4 views)

0.10

1 3 5 10

k

- Figure 9. Recall curve of different methods. Our method achieve better performance comparing with SAM2 and its variants.

generator, employing a DDPM scheduler, v-prediction, and rescaled SNR. The network is trained with 64 H100 GPUs, a batch size of 512, a learning rate of 10−5, for 10k steps.

##### A.4. Multi-view completion network

The training strategy for the multi-view completion network mirrors that of the multi-view segmentation network, with the key difference in the input configuration. The number of input channels (in latent space) is increased to 25 by including the context image, masked image, and binary mask, where the mask remains a single unencoded channel. Example inputs are illustrated in Figure 5 of the main text. The network is trained with 64 H100 GPUs, a batch size of 512, a learning rate of 10−5, and for approximately 10k steps.

##### A.5. Parts assembly

When compositing an object from its parts, we observed that simply combining the implicit neural fields of parts reconstructed by the Reconstruction Model (RM) in the rendering process with their respective spatial locations achieves satisfactory results.

To describe this formally, we first review the rendering function of LightplaneLRM [5] that we use as our reconstruction model. LightplaneLRM employs a generalized

Emission-Absorption (EA) model for rendering, which calculates transmittance Tij, representing the probability of a photon emitted at position xij (the jth sampling point in the ith ray) reaching the sensor. Then the rendered feature (e.g. color) vi of ray ri is computed as:

R−1

(Ti,j−1 − Ti,j)fv(xij)

vi =

j=1

where fv(xij) denotes the feature of the 3D point xij; Ti,j = exp(− jk=0 ∆ · σ(xik)), where ∆ is the distance between two sampled points and σ(xik) is the opacity at position xik, Ti,j−1 − Ti,j captures the visibility of the point.

Now we show how we generalise it to rendering N parts.

Given feature functions fv1,...,fvN and their opacity functions σ1,··· ,σN, the rendered feature of a specific ray ri becomes:

R−1

N

(Tˆi,j−1 − Tˆi,j)wijh · fvh(xij).

vi =

j=1

h=1

where wijh = σh(xij)/ Nl=1 σl(xij) is the weight of the feature fvh(xij) at xij for part h; Tˆi,j = exp(− jk=0

N h=1 ∆·σh(xik)), ∆ is the distance between two sampled points and σh(xik) is the opacity at position xik for part h, and Tˆi,j−1 −Tˆi,j is the visibility of the point.

##### A.6. 3D part editing

As shown in the main text and Figure 7, once 3D assets are generated or reconstructed as a composition of different parts through PartGen, specific parts can be edited using text instructions to achieve 3D part editing. To enable this, we fine-tune the text-to-multi-view generator using part multi-view images, masks, and text description pairs. Example of the training data are shown in Figure 8 (top). Notably, instead of supplying the mask for the part to be edited, we provide the mask of the remaining parts. This design choice encourages the editing network to imagine the part’s shape without constraining the region where it has to project. The training recipe is similar to multi-view segmentation network.

To generate captions for different parts, we establish an annotation pipeline similar to the one used for captioning the whole object, where captions for various views are first generated using LLAMA3 and then summarized into a single unified caption using LLAMA3 as well. The key challenge in this variant is that some parts are difficult to identify without knowing the context information of the object. We thus employ the technique inspired by [75]. Specifically, we use red annulet and alpha blending to emphasize the part being annotated. Example inputs and generated captions are shown in Figure 8 (bottom). The network is trained with 64 H100 GPUs, a batch size of 512, and the learning rate of 10−5 over 10,000 steps.

|Input Object Part 1 Part 2 Part 3<br><br>[Figure 467]<br><br>[Figure 468]<br><br>[Figure 469]<br><br>[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]<br><br>[Figure 473]<br><br>[Figure 474]<br><br>[Figure 475]<br><br>[Figure 476]<br><br>[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>|[Figure 482]|
|---|
<br><br>[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]<br><br>[Figure 486]<br><br>[Figure 487]<br><br>[Figure 488]<br><br>[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]<br><br>[Figure 492]<br><br>[Figure 493]<br><br>[Figure 494]<br><br>[Figure 495]<br><br>> A dachshund<br><br>dressed up in a<br><br>hotdog costume<br><br>[Figure 496]<br><br>[Figure 497]<br><br>[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]<br><br>[Figure 501]<br><br>[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]<br><br>[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]<br><br>[Figure 513]<br><br>[Figure 514]<br><br>[Figure 515]<br><br>[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]<br><br>[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>[Figure 524]<br><br>[Figure 525]<br><br>[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]<br><br>[Figure 529]<br><br>[Figure 530]<br><br>[Figure 531]<br><br>[Figure 532]<br><br>[Figure 533]<br><br>[Figure 534]<br><br>[Figure 535]<br><br>[Figure 536]<br><br>[Figure 537]<br><br>[Figure 538]<br><br>[Figure 539]<br><br>[Figure 540]<br><br>[Figure 541]<br><br>[Figure 542]<br><br>[Figure 543]<br><br>> A panda<br><br>rowing a boat in<br><br>a pond<br><br>[Figure 544]|
|---|

- Figure 10. More examples. Additional examples illustrate that PartGen can process various modalities and effectively generate or reconstruct 3D objects with distinct parts.

#### B. Additional Experiment Details

Then, we assign to each segment Mˆ ∈ P a reliability score based on how frequently it overlaps with similar segments in the list, i.e.,

We provide a detailed explanation of the ranking rules applied to different methods and the formal definition of mean average precision (mAP) used in our evaluation protocol. Additionally, we report the recall at K in the automatic segmentation setting.

- 1

- 2

s(Mˆ ) = M ˆ ′ ∈ P : m(Mˆ ′,Mˆ ) >

where the Intersection over Union (IoU) [38] metric is given by:

Ranking the parts. For evaluation using mAP and recall at K, it is necessary to rank the part proposal. For our method, we run the segmentation network several times and concatenate the results into an initial set P of segment proposals.

m(M,Mˆ ) = IoU(M,Mˆ ) = |Mˆ ∩ M| + ϵ |Mˆ ∪ M| + ϵ

.

Input 1 Part 2 Parts 3 Parts Composed Object

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

…

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

…

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

> A chihuahua

…

wearing a tutu

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

…

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

Figure 11. Iteratively adding parts. We show that users can iteratively add parts and combine the results of PartGen pipeline.

The constant ϵ = 10−4 smooths the metric when both regions are empty, in which case m(ϕ,ϕ) = 1, and will be useful later.

Finally, we sort the regions M by decreasing score s(M) and, scanning the list from high to low, we incrementally remove duplicates down the list if they overlap by more than 1/2 with the regions selected so far. The final result is a ranked list of multi-view masks M = (Mˆ1,...,MˆN) where N ≤ |P| and:

- 1

- 2

∀i < j : s(Mˆi) ≥ s(Mˆj) ∧ m(Mˆi,Mˆj) <

.

Other algorithms like SAM2 come with their own region reliability metric s, which we use for sorting. We otherwise apply non-maxima suppression to their ranked regions in the same way as ours.

Computing mAP. The image I comes from an object L with parts (S1,...,SS) from which we obtain the groundtruth part masks S = (M1,...,MS) as explained in Section 3.5 in the main text. We assign ground-truth segments to candidates following the procedure: we go through the list M = (Mˆ1,...,MˆN) and match the candidates one by one to the ground truth segment with the highest IOU, exclude that ground-truth segment, and continue traversing the candidate list. We measure the degree of overlap

between a predicted segment and a ground truth segment as m(M,Mˆ ) ∈ [0,1]. Given this metric, we then report the mean Average Precision (mAP) metric at different IoU thresholds τ. Recall that, based on this definition, computing the AP curve for a sample involves matching predicted segments to ground truth segments in ranking order, ensuring that each ground truth segment is matched only once, and considering any unmatched ground truth segments.

In more detail, we start by scanning the list of segments Mˆk in order k = 1,2,.... Each time, we compare Mˆk to the ground truth segments S and define:

m(Mˆk,Ms).

s∗ = argmax s=1,...,S

If m(Mˆk,Ms∗) ≥ τ, then we label the region Ms as retrieved by setting yk = 1 and removing Ms from the list of ground truth segments not yet recalled by setting

###### S ← S \ {Ms∗}.

Otherwise, if m(Mˆk,Ms∗) < τ or if S is empty, we set yk = 0. We repeat this process for all k, which results in labels (y1,...,yN) ∈ {0,1}N. We then set the average precision (AP) at τ to be:

AP(M,S;τ) =

N

k

1 S

i=1

k=1

yiyk k

.

Input Generated Grid View Reconstructed 3D

Computing recall at K. For a given sample, we define recall at K the curve

[Figure 597]

[Figure 598]

[Figure 599]

> An orangutan using chopsticks to eat ramen

S

1 S

m(Mˆs,Mk) > τ .

R(K;M,S,τ) =

[Figure 600]

χ max

k=1,...,K

s=1

Hence, this is simply the fraction of ground truth segments recovered by looking up to position K in the ranked list of predicted segments. The results in Figure 9 demonstrate that our diffusion-based method outperforms SAM2 and its variants by a large margin and shows consistent improvement as the number of samples increases.

[Figure 601]

[Figure 602]

[Figure 603]

> a group of squirrels

rowing crew

[Figure 604]

(a) Grid view generation failure

Seeded part segmentation. To evaluate seeded part segmentation, the assessment proceeds as before, except that a single ground truth part S and mask M is considered at a time, and the corresponding seed point u ∈ M is passed to the algorithm (Mˆ1,...,MˆK) = A(I,u). Note that, because the problem is still ambiguous, it makes sense for the algorithm to still produce a ranked list of possible part segments.

Input Segmentation Map Reconstructed 3D

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

#### C. Additional Examples

[Figure 614]

More application examples. We provide additional application examples in Figure 10, showcasing the versatility of our approach to varying input types. These include partaware text-to-3D generation, where textual prompts guide the synthesis of 3D models with semantically distinct parts; part-aware image-to-3D generation, which reconstructs 3D objects from a single image while maintaining detailed part-level decomposition; and real-world 3D decomposition, where complex real-world objects are segmented into different parts. These examples demonstrate the broad applicability and robustness of PartGen in handling diverse inputs and scenarios.

(b) Segmentation Failure

Input Reconstructed 3D Depth map

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

Iteratively adding parts. As shown in Figure 11, we demonstrate the capability of our approach to compose a 3D object by iteratively adding individual parts to it. Starting with different inputs, users can seamlessly integrate additional parts step by step, maintaining consistency and coherence in the resulting 3D model. This process highlights the flexibility and modularity of our method, enabling finegrained control over the composition of complex objects while preserving the semantic and structural integrity of the composition.

(C) Reconstruction Model Failure

Figure 12. Failure Cases. (a) Multi-view grid generation failure, where the generated views lack 3D consistency. (b) Segmentation failure, where semantically distinct parts are incorrectly grouped together. (c) Reconstruction model failure, where the complex geometry of the input leads to inaccuracies in the depth map.

Note that this quantity is at most 1 because by construction N i=1 yi ≤ S as we cannot match more proposal than there

are ground truth regions. mAP is defined as the average of the AP over all test samples.

#### D. Failure Cases

As outlined in the method section, PartGen incorporates several steps, including multi-view grid generation, multiview segmentation, multi-view part completion, and 3D part reconstruction. Failures at different stages will result in specific issues. For instance, as shown in Figure 12(a), failures in grid view generation can cause inconsistencies

in 3D reconstruction, such as misrepresentations of the orangutan’s hands or the squirrel’s oars. The segmentation method can sometimes group distinct parts together, and limited, in our implementation, to objects containing no more than 10 parts, otherwise it merges different building blocks into a single part. Furthermore, highly complex input structures, such as dense grass and leaves, can lead to poor reconstruction outcomes, particularly in terms of depth quality, as illustrated in Figure 12(c).

#### E. Ethics and Limitation

Ethics. Our models are trained on datasets derived from artist-created 3D assets. These datasets may contain biases that could propagate into the outputs, potentially resulting in culturally insensitive or inappropriate content. To mitigate this, we strongly encourage users to implement safeguards and adhere to ethical guidelines when deploying PartGen in real-world applications.

Limitation. In this work, we focus primarily on objectlevel generation, leveraging artist-created 3D assets as our training dataset. However, this approach is heavily dependent on the quality and diversity of the dataset. Extending the method to scene-level generation and reconstruction is a promising direction but it will require further research and exploration.

