## StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors

Xiaokun Sun, Zeyu Cai, Ying Tai, Jian Yang, Zhenyu Zhang* Nanjing University ∗Corresponding Author

xiaokun sun@smail.nju.edu.cn, caizeyu010612@gmail.com, {yingtai, csjyang}@nju.edu.cn, zhangjesse@foxmail.com

Generation:

[Figure 1]

# arXiv:2412.11586v4[cs.CV]14Feb2026

“A handsome American man with slicked-back sandy blonde hair.”

“Brad Pitt with medium-length, tousled blonde hair.”

[Figure 3]

(b) Realistic Hair-Decoupled 3D Head Avatars

“Chris Hemsworth with side-swept sandy blonde hair.”

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

| |
|---|

“Marilyn Monroe with iconic short curly platinum blonde hair.”

| |
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

| |
|---|

###### …

(a) Input Prompts

(c) Strand-Based 3D Hair

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

(d) Hairstyle Transfer (e) Hairstyle Editing (f) Physics-Based Simulation

Figure 1. Given (a) input prompts, StrandHead generates (b) realistic 3D head avatars featuring strand-level attributes and (c) 3D hair strands by utilizing human-specific 2D generative priors and 3D hair strand geometric priors. By precisely capturing the internal geometry of hair strands, our approach enables seamless and flexible (d) hairstyle transfer and (e) editing, as well as (f) physics-based simulation.

#### Abstract

on text to strand generation and disentangled 3D head avatar modeling. The generated 3D hair can be applied on avatars for strand-level editing, as well as implemented in the graphics engine for physical simulation or other applications. Project page: https://xiaokunsun. github.io/StrandHead.github.io/.

While haircut indicates distinct personality, existing avatar generation methods fail to model practical hair due to the data limitation or entangled representation. We propose StrandHead, a novel text-driven method capable of generating 3D hair strands and disentangled head avatars with strand-level attributes. Instead of using large-scale hair-text paired data for supervision, we demonstrate that realistic hair strands can be generated from prompts by distilling 2D generative models pre-trained on human mesh data. To this end, we propose a meshing approach guided by strand geometry to guarantee the gradient flow from the distillation objective to the neural strand representation. The optimization is then regularized by statistically significant haircut features, leading to stable updating of strands against unreasonable drifting. These employed 2D/3D human-centric priors contribute to text-aligned and realistic 3D strand generation. Extensive experiments show that StrandHead achieves the state-of-the-art performance

#### 1. Introduction

Creating 3D head avatars is crucial for many applications including digital telepresence, gaming, movies, and AR/VR. Hairstyles reflecting personal characteristics greatly impact the fidelity and realism of digital humans. Traditional methods rely heavily on manual effort, making them time-consuming and labor-intensive. Recent advances [52, 75, 79, 88] achieve automatic 3D head generation with the research paradigm based on supervised learning. However, these methods require costly 3D training data, which significantly limits their generalization capabilities and restricts potential applications.

With the rapid development of text-driven generation

methods [47, 54], creating 3D head avatars from given prompts without requiring 3D-text datasets becomes possible. Recent studies [1, 12, 84, 94] unlock the potential of 2D diffusion models in modeling 3D head avatars by integrating them with 3D head priors [4, 26, 77]. HumanNorm [17] further improves domain-specific fidelity by fine-tuning diffusion models on high-quality 3D human meshes. While focusing on facial geometry and texture modeling, these methods utilize holistic meshes or NeRF [38] to represent 3D haircuts, failing to capture the internal geometric structure of hair strands, i.e., 3D curves. This limitation not only significantly reduces the realism of the generated avatars but also makes them incompatible with strand-based applications and simulation systems [2, 9, 10].

To accurately model 3D haircuts, recent efforts achieve strand-level hairstyle reconstruction [55, 64, 89] or generation [13, 92] using VAEs or parametric models. Despite their impressive performance, these methods require constrained multi-view images or manual latent space searching, preventing them from freely creating 3D hair based on user-friendly text. The most recent work, HAAR [65] pioneers text-to-strand by training a text-conditioned hair map diffusion model. However, due to its reliance on largescale and costly paired data including 9825 haircuts and descriptions, HAAR faces challenges in generating diverse hairstyles outside the training set. Additionally, HAAR overlooks hair texture and geometry that adapts to specific head shapes, further limiting its practical applications.

So can we use powerful human-specific 2D generative priors to create realistic and diverse strand-based hair from text? We address this meaningful and challenging problem by developing StrandHead, a novel framework that generates high-fidelity 3D head avatars with strand-accurate hair from prompts. As illustrated in Fig. 1, the generated head avatars feature diverse and realistic hair strands, enabling seamless and flexible strand-level transfer, editing, and physics-based simulation. Instead of using large-scale hair-prompt paired data, we achieve text-aligned 3D hair strand generation by effectively utilizing human-specific 2D generative priors and 3D hair geometric priors.

Specifically, we propose a novel differentiable prismatization algorithm inspired by the cylindrical structure of hair strands. This algorithm can efficiently and differentiably convert strands into watertight prismatic meshes, which enables smooth backpropagation of gradients from 2D diffusion models pre-trained on human mesh data to 3D hair strands using mesh-based differentiable renderers, thereby making it possible to use 2D generative priors for modeling 3D hairstyles. Then we propose two effective losses inspired by the distribution patterns of 3D hair geometric features to further regularize the hair shape. These rich 2D/3D human-centric priors work together to achieve text-aligned and realistic 3D hair strand generation. Extensive experi-

ments demonstrate that StrandHead outperforms the stateof-the-art (SOTA) methods in both head and hair generation tasks, and supports flexible haircut transfer and editing, as well as physical-based rendering and simulation.

Our main contributions are summarized as follows:

- • We propose StrandHead, a novel framework for generating realistic 3D head avatars with strand-level attributes. To the best of our knowledge, StrandHead is the first work to generate 3D hair strands by distilling human-specific

- 2D diffusion models.

- • We propose a differentiable prismatization algorithm that converts hair strands into watertight prismatic meshes. This ensures stable gradient flow from 2D generative priors to 3D hair strand representation, thereby achieving reliable end-to-end strand-based hair optimization.
- • Inspired by statistical 3D hair geometric features, we introduce two simple but solid regularization losses to supervise both local and global hair shapes, enabling reasonable and realistic hairstyle generation.

2. Related Work

Text-to-3D General Object Generation. Inspired by the success of text-to-image (T2I) generation [42, 50, 51, 54, 56, 86], many studies have explored using pre-trained vision-language models [49, 54] to achieve text-guided

- 3D content generation without large-scale 3D-text paired data. Early methods [18, 37, 39, 58, 69] employ the CLIP model [49] to supervise the alignment of 3D representations with prompts. DreamFusion [47] introduces the Score Distillation Sampling (SDS) loss, significantly enhancing the fidelity of generated 3D content by leveraging more powerful pre-trained diffusion models [54]. Subsequent works further advance text-driven 3D generation by improving 3D representations [5, 6, 80], optimization strategies [29, 68, 81], SDS loss [28, 71], and diffusion models [27, 48, 62]. Despite these advancements, current approaches focused on general content generation do not fully leverage the extensive prior knowledge of human heads and hair, limiting their ability to generate realistic, high-quality 3D head avatars with strand-based hair. Text-to-3D Head Avatar Generation. CLIPFace [1], T2P [87] and Describe3D [74] pioneer zero-shot text-driven 3D head generation by combining CLIP [49] with 3D parametric head models [4, 8, 26, 77]. With the introduction of SDS loss [47], DreamFace [85] and FaceG2E [76] leverage pre-trained T2I diffusion models to greatly improve generation quality within specific domains. HeadEvolver [70] enhances the expressiveness of head mesh deformations by introducing vector fields, while HeadSculpt [12], and HeadArtist [30] employ DMTet [60] instead of traditional meshes to capture geometric details. HeadStudio [94] incorporates 3DGS [20] to produce realistic and animatable 3D heads. HumanNorm [17], further

No Large-Scale 3D-Text Paired Data Text-to-Head

Head-HairDecoupled

Strand-Based Hair

Geometry & Texture

Task Method

[12, 17, 30, 94] [84]

✗ ✓

✗ ✗

✓ ✓

✓ ✓

[28, 48, 62, 80] [65]

✗ ✗

✗ ✓

✓ ✗

✓ ✗

Text-to-Hair

Text-to-Head-Hair Ours ✓ ✓ ✓ ✓

Table 1. Comparison with current related methods.

enhances human-specific fidelity by fine-tuning 2D diffusion models on high-quality human mesh data. Despite these advancements, these approaches treat the head and hair as a holistic model, limiting support for downstream applications such as hairstyle transfer and editing. To enable flexible head-hair-disentangled generation, TECA [84] independently represents the head with mesh and the hair with NeRF [38]. However, these methods focus only on the realistic external appearance of hair without modeling its internal geometric structure, restricting their application for strand-level editing and physics-based simulation.

Strand-Based Hair Creation. Automatic strand-based hair modeling has garnered significant attention from both industry [2, 9, 10] and academia [3, 14, 33, 82], as it enables downstream physics-based rendering and simulation Such methods can be broadly categorized into two types: (1) reconstructing hair from input images or videos and (2) generating hair based on control conditions. Works on the former one [15, 21, 32, 35, 41, 55, 57, 61, 64, 67, 72, 73, 78, 83, 89–91, 93] relies on classical 3D reconstruction frameworks [20, 36, 38, 45, 59] and incorporates hair-specific features [44] to realize accurate 3D strand reconstruction. However, these methods depend on constrained multi-view images, making hair creation costly and challenging in less controlled settings. Recently, 3D hair generation approaches [13, 92] have emerged using GANs or parametric models, but they struggle with controllable generation. The most relevant work to ours is HAAR [65], which pioneers text-to-strand by training a text-conditioned hair map diffusion model. However, HAAR relies on a large-scale paired dataset including 9825 haircuts and prompts with limited diversity, restricting its ability to generate novel hairstyles beyond the training data. Moreover, HAAR overlooks hair texture and geometry that adapts to specific head shapes, further limiting its application range.

Compared to previous methods, StrandHead generates 3D heads featuring strand-level hairstyles. Instead of requiring large-scale hair-text paired data, we achieve realistic and diverse 3D hair strand generation by using 2D/3D human-centric priors. We summarize the main differences between our methods and related works in Tab. 1.

#### 3. Method

##### 3.1. Preliminaries

FLAME [26] is a 3D parametric head model that represents the human head shape, pose, and expression using a compact set of parameters. Given a set of shape parameters β, pose parameters θpose, and expression parameters ψexp,

FLAME can create a 3D head mesh M.

HumanNorm [17] is a text-driven, high-quality 3D head generation method. In specific, it models the geometry and texture of the human head using DMTet [60] and a texture field. The alignment between the text y and the optimized representation θ is achieved using the following SDS loss [47] with T2I diffusion models ϕ:

∂x ∂θ

, (1)

∇θLSDS = Et,ϵ w(t)(ϵϕ(xt;y,t) − ϵ)

where x = g(θ) is the image rendered from θ by a differentiable renderer g, t is the time step, xt = x + ϵ is a noised version of x, ϵϕ(xt;y,t) is the denoised image, and w(t) is a weighting function. HumanNorm employs human-specific diffusion models fine-tuned on high-quality human mesh data to replace the general diffusion model, enhancing the fidelity of the generated human heads.

Neural Scalp Textures (NST) [55] is an efficient hair representation where each pixel stores a feature vector conveying the shape information of a single strand at the corresponding scalp location. Using a pre-trained hair strand generator [55] G, the low-dimensional 2D neural scalp texture T can be decoded into high-dimensional 3D strand polylines S = {si}N

i=1, where Ns is the number of strands

s

and si = {pij}Nj=1p consists of Np 3D points. This process is formulated as follows: S = G(T).

##### 3.2. Overview

Given a text prompt, StrandHead aims to create a realistic 3D head avatar with strand-based hair without relying on large-scale 3D-text paired data. Fig. 2 provides an overview of our three-stage pipeline. First, we generate a FLAMEaligned 3D bald head for accurate hair initialization using human-specific diffusion models (Sec. 3.3). In the subsequent stages, we model the reasonable geometry and realistic texture of 3D hair strands by leveraging rich 2D/3D human-centric priors (Sec. 3.4 and Sec. 3.5).

##### 3.3. Bald Head Generation

To achieve accurate hair initialization and optimize 3D hair strands in subsequent steps using human-specific 2D generative priors, we first need to obtain a reasonable and semantic-aligned 3D bald head from the text (Fig. 2-(a)). To achieve this goal, we enhance HumanNorm [17] by incorporating an evolving FLAME [26] model while optimizing the bald head DMTet θh. This provides accurate semantic information and prevents unnatural geometry. Please refer to the Supp. Mat. for details on bald head generation.

##### 3.4. Hair Geometry Generation

With a FLAME-aligned bald head, we can generate 3D hair in the scalp area. Specifically, we first initialize the haircut based on hairstyle descriptions, and then further sculpt the 3D hair shape utilizing 2D/3D human-centric priors.

[Figure 21]

[Figure 22]

[Figure 25]

𝑮𝑮

Optimized Neural Scalp Textures 𝑻𝑻

[Figure 27]

[Figure 28]

###### (c) Hair Texture Generation

[Figure 29]

[Figure 30]

(a) Bald Head Generation

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Differentiable Rendering

[Figure 35]

Normal- 𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡 Conditioned

“A bald handsome American man.”

[Figure 36]

[Figure 37]

[Figure 38]

𝒏𝒏𝐡𝐡+𝐬𝐬

[Figure 39]

[Figure 40]

Diffusion Model 𝑳𝑳𝑴𝑴𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

𝑳𝑳𝒃𝒃𝒃𝒃𝒐𝒐𝒃𝒃 𝑳𝑳𝒇𝒇𝒇𝒇𝒄𝒄𝒇𝒇 𝑳𝑳𝒄𝒄𝒐𝒐𝒄𝒄𝒄𝒄𝒐𝒐

Bald Head DMTet 𝜽𝜽𝒉𝒉

Bald Head Texture Field 𝝍𝝍𝒉𝒉

FLAME Mesh 𝑴𝑴𝒉𝒉 Hair Strand

Texture Field 𝝍𝝍𝒔𝒔

𝒄𝒄𝐡𝐡+𝐬𝐬

###### (b) Hair Geometry Generation

[Figure 41]

|[Figure 42]|
|---|

|[Figure 43]|
|---|

[Figure 44]

[Figure 45]

(a) Bald Head Generation

[Figure 46]

Strand Generator

Normal-Adapted Diffusion Model 𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

𝑳𝑳𝒐𝒐𝒐𝒐𝒐𝒐 + 𝑳𝑳𝒄𝒄𝒄𝒄𝒐𝒐

[Figure 47]

“A handsome American man with slicked-back

| |
|---|

| |
|---|

[Figure 48]

[Figure 49]

[Figure 50]

𝐡𝐡+𝐬𝐬

[Figure 51]

zz 𝒏𝒏

[Figure 52]

[Figure 53]

Differentiable Rendering

[Figure 54]

𝑮𝑮

Differentiable 𝑮𝑮 Prismatization

sandy blonde hair.”

DiffusionDepth-AdaptedModel 𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

Strand Generator

Hair Prismatic Meshes 𝑴𝑴𝒑𝒑

Neural Scalp Textures 𝑻𝑻

Hair Strands 𝑺𝑺

𝒅𝒅𝐡𝐡+𝐬𝐬

[Figure 56]

[Figure 57]

: Frozen Parameters : Optimized Parameters : Human-Specific 2D Generative Priors : 3D Hair Strand Geometric Priors

Figure 2. StrandHead includes three stages: (a) We first create a FLAME-aligned 3D bald head using the improved HumanNorm [17]. (b) Next, we introduce a differentiable prismatization algorithm to enable human-specific geometry-aware 2D diffusion models to supervise hair shape modeling. Additionally, two losses inspired by 3D hair geometric priors are applied to further regularize the hair geometry. (c) Finally, we use a human-specific normal-conditioned 2D diffusion model to generate lifelike hair textures.

𝑳𝑳𝒐𝒐𝒐𝒐𝒐𝒐 𝑳𝑳𝒄𝒄𝒄𝒄𝒐𝒐

Hair Initialization. To achieve reasonable and diverse hair initialization, we first utilize ChatGPT [43] to select the 20 most representative hairstyles from the USC-HairSalon Dataset [15]. We then optimize neural scalp textures to fit the selected hairstyle using the following loss function:

: Forward : Backward

Mesh-Based Differentiable Renderer

[Figure 58]

Optimized Neural Scalp Textures 𝑻𝑻

Differentiable Prismatization

Strand Generator

Reasonable Normals

2D Image SDS Loss Neural Scalp Texture

Hair Strand Hair Prismatic Mesh

(a) Gradient backpropagation process

: Strand Points : Strand Normals : Mesh Vertices

[Figure 59]

Np

Step 1 Step 2 Step 3 Step 4 Step 5

Ns

∥pˆij−pij∥2+λori(1−oˆij·oij)+λcur∥cˆij−cij∥1,

Lfit =

(b) A strand-to-mesh conversion example

i=1

j=1

[Figure 60]

[Figure 61]

(2) where pˆij and pij are the position of the j-th point on the i-th polyline of the GT and generated hair, respectively, and oˆij, oij, cˆij, and cij denote their orientation and curvature, respectively. Here, orientation oij = (pij+1 − pij)/∥pij+1 − pij∥2 represents the direction of change in strand position, and curvature cij = ∥oij − oij−1∥2 denote the rate of change in strand orientation. Given a hair description, we select an optimal pre-trained NST as initialization and further optimize it in subsequent stages.

|[Figure 62]|
|---|

[Figure 63]

|[Figure 64]<br><br>| |
|---|
|
|---|

| |
|---|

|[Figure 65]<br><br>| |
|---|
|
|---|

, (b) a strand-to-mesh conversion exam-ple and (c) advantages over other conversion methods

| |
|---|

Reliable Optimization Reasonable Normals Watertight Mesh Non-Watertight Mesh

Unstable Optimization

Ambiguous Normals

w/ differentiable prismatization w/ NeuralHaircut’s quad mesh

(c) Differentiable prismatization’s advantages

Figure 3. The differentiable prismatization algorithm’s (a) gradient backpropagation process, (b) a strand-to-mesh conversion example and (c) advantages over NeuralHaircut [64]. Non-watertight quad meshes can easily produce ambiguous normal maps, which significantly reduce the stability of hair shape modeling (see the drifting hair highlighted by the oval dotted box in (c)).

Discussion on Strand-Based Differentiable Rendering. With reasonable hair initialization, the next step is to leverage human-specific 2D diffusion models to further sculpt the hair geometry. However, due to the lack of a stable strand-based differentiable renderer, it is very challenging to optimize hair shape using 2D generative priors like other SDS-based methods. Besides, we expect hair strands to produce mesh-style smooth geometry and texture maps to fully exploit the powerful generative capabilities of 2D diffusion models pre-trained on human mesh data. A feasible alternative is to first differentiably convert the strands into quad meshes [82], similar to NeuralHaircut [64], and then leverage SDS-based mesh optimization frameworks to model hair strands. However, this non-watertight stripe-like mesh easily produces ambiguous normal maps or excessively thin sides (see the zoom image of Fig. 3-(c)), which significantly

“A charming woman with a straight nose and shoulder-length straight red hair.”

reduce the optimization stability (see the drifting hair highlighted by the oval dotted box in Fig. 3-(c)). Therefore, a differentiable strand-to-mesh conversion method with superior optimization properties is urgently needed.

Differentiable Prismatization (DP). To this end, we propose a novel differentiable prismatization algorithm inspired by the internal structure of 3D hair (i.e., hair fiber is a dielectric cylinder covered with tilted scales and with a pigmented interior [19, 34]). This algorithm can efficiently and differentiably convert strands into watertight prismatic meshes with arbitrary thickness and lateral edges to approximate the cylindrical structure of the hair. This enables

(a) Ablation study on self-evolving human prior loss

(a) Ablation study on self-evolving human prior loss

[Figure 66]

[Figure 67]

smooth backpropagation of gradients from the SDS loss to the 3D strand representation using mesh-based differentiable renderers [22], thereby making it possible to fully exploit 2D generative priors distilled from high-fidelity human mesh data for modeling realistic hair strands. Fig. 3-(a) visualizes the complete gradient backpropagation process.

NumberofHaircuts

NumberofHaircuts

(a) Ablation study on self-evolving human prior loss

𝐶𝐶𝐶𝐶ori

𝐶𝐶mean

In specific, given a hair strand s, our DP converts it into a watertight prismatic mesh with K lateral edges and radius R through the following five steps: (1) Compute the Initial Normal Vector. (2) Generate K Rotated Normals. (3) Translate to Form Lateral Edges. (4) Construct Lateral Faces. (5) Construct Top and Bottom Faces. An example of this conversion is shown in Fig. 3-(b). Unlike quad meshes [64], our DP effectively avoids ambiguous normals, enabling smooth gradient backpropagation and hair strand optimization, as shown in Fig. 3-(c). Further algorithm details are provided in the Supp. Mat.

(a) Distribution of Haircuts’ 𝐶𝐶mean

(b) Distribution of Haircuts’

𝐶𝐶𝐶𝐶ori

|[Figure 68]|
|---|

[Figure 69]

|[Figure 70]|
|---|

|[Figure 71]|
|---|

[Figure 72]

[Figure 73]

| |
|---|

| |
|---|

| |
|---|

𝐶𝐶𝐶𝐶ori = 0.9949 𝐶𝐶mean = 0.0211 𝐶𝐶𝐶𝐶ori = 0.9605 𝐶𝐶mean = 0.0493

𝐶𝐶𝐶𝐶ori = 0.9868 𝐶𝐶mean = 0.0290

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

𝐶𝐶𝐶𝐶ori = 0.9265 𝐶𝐶mean = 0.0598 𝐶𝐶𝐶𝐶ori = 0.9719 𝐶𝐶mean = 0.0870 𝐶𝐶𝐶𝐶ori = 0.9297 𝐶𝐶mean = 0.1059

(c) Some hairstyles from the USC-HairSalon dataset

Figure 4. Observation of hair geometric features: (1) neighboring strand orientations are highly consistent. (2) strand curvature is strongly and positively related to the haircut curliness.

With the help of our proposed differentiable prismatization algorithm, we optimize the neural scalp texture T by utilizing rich human-specific 2D generative priors through the following SDS losses (Fig. 2-(b)):

Np

Ns

oij · okj |A(i)|

1 NsNp

, (5)

CSori =

∂nh+s ∂T

∇TLhnSDS = Et,ϵ (ϵϕ

(nh+st ;yh+s,t) − ϵ)

, (3)

i=1

j=1 k∈A(i)

hn

Np

Ns

∂dh+s ∂T

1 NsNp

cij, (6)

∇TLhdSDS = Et,ϵ (ϵϕ

(dh+st ;yh+s,t) − ϵ)

Cmean =

, (4)

hd

i=1

j=1

where ϕhn and ϕhd are the human-specific normaladapted and depth-adapted diffusion models from HumanNorm [17], nh+s and dh+s are the rendered normal and depth maps of the head DMTet θh with hair prismatic mesh Mp = DP(G(T)), respectively. Here, DP(·) denotes the differentiable prismatization operation function, yh+s is the full text description of the head with hair.

However, due to the excessive flexibility of hair strands, relying solely on the SDS loss leads to unnatural strand orientations and unrealistic hairstyles (Fig. 9). Furthermore, the lack of constrained multi-view images prevents us from leveraging hair-specific features [44] to provide powerful supervision as in previous works [55].

Prior-Driven Losses. To address this issue, we propose two straightforward yet robust prior-driven losses based on observations of the 3D hair strand orientation and curvature to ensure the rationality of the generated hair strands. Specifically, we observe two geometric properties of hair: (1) neighboring strand orientations are highly consistent, and (2) strand curvature is strongly and positively correlated with the curliness of the hairstyle. To validate the properties, we calculate the cosine similarity of adjacent hair strand orientations CSori and the average curvature Cmean of 343 hairstyles in the USC-HairSalon dataset [15] using the following equations and visualize their distributions and some examples in Fig. 4:

where A(i) is the set of adjacent strands for the i-th strand, and |A(i)| is the number of neighboring strands.

As illustrated, over 95% of hairstyles exhibit an orientation similarity above 0.9, and the average curvature is significantly positively related to curliness, which successfully validates the properties. Based on these observations, we introduce orientation consistency loss and curvature regularization to guide the local and global strand shapes, formulated as follows:

Lori = 1 − CSori, (7) Lcur = ∥Cmean − Ctarget∥1, (8)

where Ctarget represents the target average curvature set according to the input hair description. These two losses, inspired by 3D hair geometry priors, regularize the hair shape by supervising the consistency of orientations between adjacent strands and the overall hairstyle’s curvature.

Additionally, we introduce a series of losses Lbbox, Lface, and Lcolli, which prevent the hair from exceeding the bounding box, obscuring the face, and colliding with the head, respectively. These losses can further improve the rationality of hair geometry. Their details are provided in the Supp. Mat. The final loss function for optimizing hair shape is expressed as follows:

Lhair-geo = LhdSDS + λhnSDSLhnSDS + λoriLori + λcurLcur

+ λbboxLbbox + λfaceLface + λcolliLcolli. (9)

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

[Figure 90]

[Figure 91]

“A mature European woman with a straight nose and mediumlength, straight auburn hair.”

“An elderly Caucasian grandmother with thin lips and a straight, short silver bob.”

“Brad Pitt with medium-length, tousled blonde hair.”

“Chris Hemsworth with sideswept sandy blonde hair.”

“Lionel Messi with a blonde mohawk hairstyle.”

“Scarlett Johansson with a textured blonde bob.”

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

“A charming Caucasian woman with a straight nose and shoulderlength curly sleek blonde hair.”

“Beyoncé with voluminous, curly honey blonde hair.”

“Dwayne Johnson with a short black afro.”

“A beautiful Caucasian girl with delicate features and long silky golden blonde hair.”

“A confident Hispanic man with a beard and a curly black afro.”

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

[Figure 110]

[Figure 111]

| |
|---|

| |
|---|

| |
|---|

“Long wavy chestnut brown hair.”

“A short, wavy brown bob.”

“A light brown mohawk hairstyle.”

“Long silky golden blonde hair.”

“Side-swept sandy blonde hair.”

| |
|---|

|[Figure 112]|
|---|

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

“Medium-length, straight auburn hair.”

“A sleek long black hairstyle.”

“Medium-length wavy caramel-colored hair.”

“A side-swept silver haircut.”

“Medium-length, tousled blonde hair.”

“A curly dark blonde haircut.”

“A curly black afro.”

Figure 5. Examples of high-fidelity and diverse 3D heads and strand-accurate haircuts generated by our method. The upper visualization includes rendered color and normal maps of the head and hair prismatic meshes. The lower visualization shows the physics-based hair strand rendering result using Blender [10]. For better strand-based visualization, we interpolate generated hair to approximately 10,000 strands and apply a consistent appearance. Please zoom in for detailed views, and refer to the Supp. Mat. for video demonstrations.

##### 3.5. Hair Texture Generation

server equipped with A6000 GPUs. Starting with a prompt such as “A man with brown hair”, we first generate a bald head using “A bald man”. Then, we generate the hair based on “A man with brown hair”. Further implementation details are available in the Supp. Mat.

Next, we fix the optimized hair strand geometry and model the realistic hair strand texture under the supervision of the SDS loss as follows (Fig. 2-(c)):

∂ch+s ∂ψs

∇ψsLhcSDS = Et,ϵ(ϵϕ

(ch+st ;nh+s,yh+s,t) − ϵ)

, (10)

#### 4. Experiments

hc

where ϕhc is the human-specific normal-conditioned diffusion model from HumanNorm [17], ch+s is the rendered color map of the head texture field ψh and hair strand texture field ψs. Since the vanilla SDS loss often leads to color oversaturation, we replace it with the following MSDS loss [17] to further enhance the texture’s realism in later iterations of texture optimization:

Some examples of 3D head models with strand-based hair generated by StrandHead are shown in Fig. 5. Due to space limitations, only the key experimental settings and results are presented here. For more results, evaluations, and details, please refer to our Supp. Mat.

##### 4.1. Experimental Settings

Baselines. We compare StrandHead with the SOTA methods for head avatar and haircut generation. The baseline methods for head generation include text-to-holistic-head works (HeadArtist [30], HumanNorm [17], and HeadStudio [94]) and text-to-decoupled-head work (TECA [84]). For text-to-hair generation, we consider general text-to3D methods (MVDream [62], GaussianDreamer [80], LucidDreamer [28], and RichDreamer [48]), text-to-strand method (HAAR [65]), and TECA [84].

∂ch+s ∂ψs

∇ψsLhcMSDS = Et,ϵ(H(ch+st ;nh+s,yh+s,t) − ϵ)

, (11)

where H(·) denotes the multi-step operation function. Additionally, we propose a strand-aware texture field that models orientation-dependent texture to better generate highfrequency color variations. Please refer to the Supp. Mat. for more their details.

##### 3.6. Implementation Details

Dataset Construction. To create the dataset, we use ChatGPT[43] to randomly generate 30 text descriptions of heads with hair. These include 30 descriptions for assessing

Our method is implemented using PyTorch [46] and ThreeStudio [11]. Experiments are conducted on an Ubuntu

MVDream GaussianDreamer LucidDreamer RichDreamer TECA HAAR StrandHead

HeadArtist HeadStudio HumanNorm TECA StrandHead

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

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

N/A

Text-to-Head

Text-to-Hair

“A handsome American man with slicked-back sandy blonde hair.”

“Medium-length, straight auburn hair.”

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

|[Figure 142]|
|---|

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

N/A

“Beyoncé with voluminous, curly honey blonde hair.”

“A short, wavy brown bob.”

- Figure 6. Qualitative comparisons with the SOTA methods. Since TECA [84] uses the vanilla NeRF to represent hair, rendering normals is not supported. HAAR [65] generates only the geometry of hair strands, so we first convert the strands into prismatic meshes using differentiable prismatization and then utilize TEXTure [53] to generate texture for visualization and comparison.

“A curly dark brown afro.”

[Figure 154]

[Figure 155]

| |
|---|

| |
|---|

| |
|---|

HAAR StrandHead

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

“A side-swept silver haircut.”

“Slicked-back ash blonde hair.” “Shoulder-length straight red hair.”

“A short wavy brown bob.”

| |
|---|

HAAR StrandHead

- Figure 7. Qualitative comparison with HAAR [65]. HAAR, which does not model heads, often produces unreasonable hair-head collisions (highlighted in the black box).

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

“Medium-length, straight auburn hair.”

[Figure 176]

[Figure 177]

Method BLIP-VQA ↑ BLIP2-VQA ↑ GQP ↑ TAP ↑

N/A

“A short, wavy brown bob.”

HeadArtist 0.7667 0.9667 1.00 2.33 HeadStudio 0.7833 0.8833 3.33 3.67

HumanNorm 0.7000 0.9500 7.67 7.67

“Slicked-back sandy blonde hair.”

“Slicked-back sandy blonde hair.”

TECA 0.7333 0.9500 34.33 28.33 StrandHead (Ours) 0.8500 0.9667 53.67 58.00

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

“Tight, curly dark brown hair..”

[Figure 182]

MVDream 0.9000 0.8333 24.67 20.00 GaussianDreamer 0.3333 0.3667 5.33 3.00

LucidDreamer 0.8000 0.9333 5.33 5.00

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

RichDreamer 0.8333 0.7667 4.00 5.67 TECA 0.7000 0.7667 1.67 3.67 HAAR 0.6333 0.2000 1.33 2.33

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

StrandHead (Ours) 0.9000 0.9000 57.67 60.33

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Table 2. Quantitative comparisons with the SOTA methods. The best and second-best results are highlighted in bolded and underlined, respectively. GQP: generation quality preference (%), TAP: text-image alignment preference (%).

“An elderly Caucasian grandmother with thin lips and a straight, short silver bob.”

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 203]

“A confident black woman with a curly dark brown afro.”

“A handsome American man with slicked-back sandy blonde hair.”

selecting 10 generated examples and asking 30 volunteers to assess (1) generation quality and (2) text-image alignment, and select the best methods.

[Figure 204]

[Figure 206]

[Figure 209]

“Beyoncé with voluminous, curly honey blonde hair.”

[Figure 211]

N/A

“An elderly Caucasian grandmother with thin lips and a straight, short silver bob.”

head generation and 30 for evaluating hair generation. The full list of prompts can be found in the Supp. Mat.

“A side-swept silver haircut.”

“Long, silky black wavy hair.”

##### 4.2. Comparisons of Head Generation

As illustrated in the upper section of Tab. 2, StrandHead outperforms all comparison methods across every evaluation metric. The qualitative results on the left side of Fig. 6 further highlight the advantages of our method. Compared to other head generation methods, the head avatars generated by StrandHead exhibit not only more refined facial geometry and texture details but also strand-accurate hair with a plausible appearance that integrates seamlessly with physics-based simulation systems. To the best of our knowledge, StrandHead is the first head generation framework that realizes strand-level hair modeling, a capability that holds substantial potential for advancing human-centric 3D AIGC applications in the industry.

Evaluation Metrics. Current text-to-3D methods typically use CLIP-based metrics to assess text-image alignment and output quality. However, research [16, 31] shows that these metrics struggle to capture fine-grained alignment between 3D content and input prompts, as validated by our experiments in the Supp. Mat. To address this limitation, we draw inspiration from Progressive3D [7] and Barbie [66], using BLIP-VQA [23, 24] and BLIP2-VQA [23, 25] for evaluation. Specifically, we convert each prompt into two questions to separately verify head and hair. We then feed the rendered image of the generated 3D content into the VQA model, asking each question in sequence and using the probability of a “yes” response as our evaluation metric. For example, the head prompt “A man with black hair” is transformed into “Is the person in the picture a man?” and “Does the person in the picture have black hair?” The hair prompt “Black hair” becomes “Is the object in the picture black hair?” Finally, we conduct a user study by randomly

##### 4.3. Comparisons of Hair Generation

As shown in the lower part of Tab. 2, our approach surpasses other methods across most evaluation metrics, and is only slightly lower than LucidDreamer (the latest open-source

(a) Ablation study on self-evolving head prior loss

“Medium-length wavy caramel-colored hair.”

“Slicked-back sandy blonde hair.”

“A curly dark brown afro.”

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

| |
|---|

| |
|---|

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

| |
|---|

| |
|---|

(a)

w/ human-specific 2D supervision

w/ general 2D supervision w/o 2D supervision

w/ w/o

w/ w/o

[Figure 222]

[Figure 223]

[Figure 224]

(a) (b)

(a) Ablation study on human-specific diffusion models’ supervision

Figure 9. Ablation study on (a) orientation consistency loss Lori and (b) re re tion loss

“A confident man with a broad jawline and a side-swept jet-black haircut.”

[Figure 225]

InitialHairstyle+HairstylePrompt

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

|curvature|
|---|

|regularization|
|---|

|Lcur.|
|---|

| |
|---|

“Long, straight platinum blonde hair.”

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

“Mediumlength, curly hair.”

“A weathered African man with a broad forehead and brown dreadlocks” “A beautiful girl with delicate features and long ponytail”

Figure 10. Failure cases.

[Figure 251]

w/ differentiable prismatization (c)w/ NeuralHaircut’s quad mesh

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

“Tight, curly dark

StrandHead over HAAR [65].

brown hair.”

Effectiveness of Prior-Driven Losses. Fig. 9-(a) demonstrates that incorporating Lori provides strong guidance for generating natural hair by supervising the consistency of orientations between adjacent strands. As shown in Fig. 9(b), our Lcur preserves the desired curliness by constraining the overall hairstyle’s curvature. In summary, these two losses, inspired by 3D hair geometry priors, effectively enhance the realism and rationality of the generated haircut.

[Figure 258]

[Figure 259]

[Figure 260]

“A bald beautiful girl with delicate features.”

“A bald handsome American man.”

“Bald Marilyn Monroe.”

Bald Head

(b) Generated strand-based hair under different bald heads

Figure 8. Analysis of human-specific 2D diffusion models.

text-to-3D method) on BLIP2-VQA, ranking the second. The qualitative comparisons on the right side of Fig. 6 display sample generation results. Compared to general textto-3D methods, our approach more precisely captures the internal structure of hair strands, resulting in realistic haircut geometry and appearance without incorporating incorrect content such as parts of the human head.

| |
|---|

| |
|---|

| |
|---|

#### 5. Conclusion

In this paper, a novel framework StrandHead is proposed to generate realistic hair-disentangled 3D head avatars using text prompts. With a series of 2D generative models pre-trained on human-centric data, StrandHead generates strand-based hair requiring no large-scale text-hair paired data for supervision. This goal is achieved mainly by two approaches: (1) proposing a novel differentiable prismatization method that transforms the hair strands into prismatic meshes, and (2) using the prior-driven losses inspired by the observation of real-world haircut and the statistical features of human-crafted 3D hair data. The former one produces watertight geometry more similar to the original hair strands, boosting a stable optimization with significantly less normal ambiguity. Besides, the latter one well constrains the optimization with reliable guidance to generate reasonable 3D haircuts. Extensive experiments demonstrate that StrandHead obtains the state-of-the-art performance on 3D head avatars and haircuts generation. The generated results are easily implemented in industrial software to produce physical simulation and high-fidelity rendering.

“Tight curly dark brown hair.”

Compared with HAAR [65], StrandHead relies on robust 2D/3D human-centric priors rather than large-scale paired datasets to supervise hair generation. This not only allows it to generate hairstyles that are uncommon in the dataset (see slicked-back and side-swept hairstyles in Fig. 7), but also models hair texture or geometry that adapts to specific head shapes (Fig. 8-(b)), thereby avoiding unnatural hairhead collisions (see the black box in Fig. 7).

##### 4.4. Ablation Study

Effectiveness of Human-Specific 2D Generative Priors. We demonstrate the importance of human-specific 2D generative priors from two aspects: (1) As shown in Fig. 8(a), ignoring 2D supervision fails to generate meaningful hair. While a general 2D diffusion model can slightly improve modeling performance, it still falls short of producing high-quality results. Only by incorporating human-specific 2D generative priors can one achieve realistic textures and reasonable shapes for hairstyles. (2) Fig. 8-(b) illustrates generated hair under varying bald head conditions while maintaining a fixed initial hairstyle and hair prompt. The hairstyles created under the supervision of human-specific 2D diffusion models exhibit clear geometric and textural variations that adapt to specific bald heads. This strongly demonstrates the necessity of considering the bald head for hair strand generation and highlights the advantages of

Limitations and Future Work. Due to the limited representation capabilities of the strand generator, StrandHead is unable to create highly complex 3D hairstyles (e.g., dreadlocks and ponytails in Fig. 10). Additionally, although the SDS-based method effectively leverages 2D generative priors, its high computational cost restricts practical application. In future work, we aim to address these challenges by incorporating richer hairstyle datasets and exploring deeper hair priors.

Acknowledgement. This work was supported by the National Science Foundation of China (62376121).

#### References

- [1] Shivangi Aneja, Justus Thies, Angela Dai, and Matthias Nießner. Clipface: Text-guided editing of textured 3d morphable models. ACM SIGGRAIPH, 2022. 2
- [2] Inc. Autodesk. Autodesk maya - 3d animation and modeling software. https://www.autodesk.com/ products/maya, 2024. 2, 3
- [3] Gaurav Bhokare, Eisen Montalvo, Elie Diaz, and Cem Yuksel. Real-time hair rendering with hair meshes. In ACM SIGGRAIPH, 2024. 3
- [4] Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023. 2
- [5] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Int. Conf. Comput. Vis., 2023. 2
- [6] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [7] Xinhua Cheng, Tianyu Yang, Jianan Wang, Yu Li, Lei Zhang, Jian Zhang, and Li Yuan. Progressive3d: Progressively local editing for text-to-3d content creation with complex semantic prompts. In Int. Conf. Learn. Represent., 2023. 7, 5
- [8] Hang Dai, Nick Pears, William Smith, and Christian Duncan. Statistical modeling of craniofacial shape and texture. Int. J. Comput. Vis., 2020. 2
- [9] Inc. Epic Games. Unreal engine - real-time 3d creation tool. https://www.unrealengine.com/, 2024. 2, 3
- [10] Blender Foundation. Blender - a 3d modelling and rendering software. https://www.blender.org/, 2024. 2, 3, 6, 7
- [11] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, ZiXin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/ threestudio, 2023. 6
- [12] Xiaoping Han, Yukang Cao, K. Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, and Kwan-Yee K. Wong. Headsculpt: Crafting 3d head avatars with text. In Adv. Neural Inform. Process. Syst., 2023. 2, 3
- [13] Chengan He, Xin Sun, Zhixin Shu, Fujun Luan, S¨oren Pirk, Jorge Alejandro Amador Herrera, Dominik L Michels, Tuanfeng Y Wang, Meng Zhang, Holly Rushmeier, and Yi Zhou. Perm: A parametric representation for multi-style 3d hair modeling. arXiv preprint arXiv:2407.19451, 2024. 2, 3
- [14] Jerry Hsu, Tongtong Wang, Zherong Pan, Xifeng Gao, Cem Yuksel, and Kui Wu. Real-time physically guided hair interpolation. ACM Trans. Graph., 2024. 3

- [15] Liwen Hu, Chongyang Ma, Linjie Luo, and Hao Li. Singleview hair modeling using a hairstyle database. ACM Trans. Graph., 2015. 3, 4, 5, 2
- [16] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. In Adv. Neural Inform. Process. Syst., 2023. 7, 5
- [17] Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying Feng, Yebin Liu, and Qing Wang. Humannorm: Learning normal diffusion model for high-quality and realistic 3d human generation. In IEEE Conf. Comput. Vis. Pattern Recog.,

2024. 2, 3, 4, 5, 6, 1

- [18] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2
- [19] James T Kajiya and Timothy L Kay. Rendering fur with three dimensional textures. ACM Siggraph Computer Graphics,

1989. 4

- [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 2023. 2, 3
- [21] Zhiyi Kuang, Yiyang Chen, Hongbo Fu, Kun Zhou, and Youyi Zheng. Deepmvshair: Deep hair modeling from sparse views. In ACM SIGGRAPH Asia, 2022. 3
- [22] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Trans. Graph., 2020. 5
- [23] Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library for language-vision intelligence. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2023. 7, 5
- [24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, 2022. 7, 5
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023. 7, 5
- [26] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 2017. 2, 3
- [27] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arXiv preprint arXiv:2310.02596, 2023. 2
- [28] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards highfidelity text-to-3d generation via interval score matching. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2, 3, 6
- [29] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2

- [30] Hongyu Liu, Xuan Wang, Ziyu Wan, Yujun Shen, Yibing Song, Jing Liao, and Qifeng Chen. Headartist: Textconditioned 3d head generation with self score distillation. In ACM SIGGRAIPH, 2024. 2, 3, 6, 1
- [31] Yujie Lu, Xianjun Yang, Xiujun Li, Xin Eric Wang, and William Yang Wang. Llmscore: Unveiling the power of large language models in text-to-image synthesis evaluation. In Adv. Neural Inform. Process. Syst., 2024. 7, 5
- [32] Haimin Luo, Min Ouyang, Zijun Zhao, Suyi Jiang, Longwen Zhang, Qixuan Zhang, Wei Yang, Lan Xu, and Jingyi Yu. Gaussianhair: Hair modeling and rendering with light-aware gaussians. arXiv preprint arXiv:2402.10483, 2024. 3
- [33] Ryota Maeda, Kenshi Takayama, and Takafumi Taketomi. Refinement of hair geometry by strand integration. In Comput. Graph. Forum, 2023. 3
- [34] Stephen R Marschner, Henrik Wann Jensen, Mike Cammarano, Steve Worley, and Pat Hanrahan. Light scattering from human hair fibers. ACM Trans. Graph., 2003. 4
- [35] Givi Meishvili, James Clemoes, Charlie Hewitt, Zafiirah Hosenie, Xian Xiao, Martin de La Gorce, Tibor Takacs, Tadas Baltrusaitis, Antonio Criminisi, Chyna McRae, et al. Hairmony: Fairness-aware hairstyle classification. arXiv preprint arXiv:2410.11528, 2024. 3
- [36] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In IEEE Conf. Comput. Vis. Pattern Recog., 2019. 3
- [37] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. In IEEE Conf. Comput. Vis. Pattern Recog.,

2022. 2

- [38] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Eur. Conf. Comput. Vis., 2020. 2, 3
- [39] Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Tiberiu Popa. Clip-mesh: Generating textured meshes from text using pretrained image-text models. In ACM SIGGRAPH Asia, 2022. 2
- [40] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 2022. 1
- [41] Giljoo Nam, Chenglei Wu, Min H Kim, and Yaser Sheikh. Strand-accurate multi-view hair capture. In IEEE Conf. Comput. Vis. Pattern Recog., 2019. 3
- [42] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [43] OpenAI. Chatgpt. https://openai.com/, 2024. 4, 6, 2
- [44] Sylvain Paris, Hector M Briceno, and Franc¸ois X Sillion. Capture of hair geometry from multiple images. ACM Trans. Graph., 2004. 3, 5
- [45] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning con-

- tinuous signed distance functions for shape representation. In IEEE Conf. Comput. Vis. Pattern Recog., 2019. 3
- [46] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Adv. Neural Inform. Process. Syst., 2019. 6
- [47] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In Int. Conf. Learn. Represent., 2022. 2, 3
- [48] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to3d. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2, 3, 6
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning,

2021. 2

- [50] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, 2021. 2
- [51] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [52] Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, and Michael J Black. Generating 3d faces using convolutional mesh autoencoders. In Eur. Conf. Comput. Vis., 2018. 1
- [53] Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. Texture: Text-guided texturing of 3d shapes. In ACM SIGGRAIPH, 2023. 7
- [54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2
- [55] Radu Alexandru Rosu, Shunsuke Saito, Ziyan Wang, Chenglei Wu, Sven Behnke, and Giljoo Nam. Neural strands: Learning hair geometry and appearance from multi-view images. In Eur. Conf. Comput. Vis., 2022. 2, 3, 5
- [56] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Adv. Neural Inform. Process. Syst., 2022. 2
- [57] Shunsuke Saito, Liwen Hu, Chongyang Ma, Hikaru Ibayashi, Linjie Luo, and Hao Li. 3d hair synthesis using volumetric variational autoencoders. ACM Trans. Graph.,

2018. 3

- [58] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2

- [59] Johannes L Sch¨onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In Eur. Conf. Comput. Vis.,

2016. 3

- [60] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. In Adv. Neural Inform. Process. Syst., 2021. 2, 3, 1
- [61] Yuefan Shen, Shunsuke Saito, Ziyan Wang, Olivier Maury, Chenglei Wu, Jessica Hodgins, Youyi Zheng, and Giljoo Nam. Ct2hair: High-fidelity 3d hair modeling using computed tomography. ACM Trans. Graph., 2023. 3
- [62] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In Int. Conf. Learn. Represent., 2023. 2, 3, 6
- [63] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Int. Conf. Learn. Represent., 2014. 2
- [64] Vanessa Sklyarova, Jenya Chelishev, Andreea Dogaru, Igor Medvedev, Victor Lempitsky, and Egor Zakharov. Neural haircut: Prior-guided strand-based hair reconstruction. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2, 3, 4, 5
- [65] Vanessa Sklyarova, Egor Zakharov, Otmar Hilliges, Michael J Black, and Justus Thies. Text-conditioned generative model of 3d strand-based human hairstyles. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2, 3, 6, 7, 8
- [66] Xiaokun Sun, Zhenyu Zhang, Ying Tai, Qian Wang, Hao Tang, Zili Yi, and Jian Yang. Barbie: Text to barbie-style 3d avatars. arXiv preprint arXiv:2408.09126, 2024. 7, 1
- [67] Yusuke Takimoto, Hikari Takehara, Hiroyuki Sato, Zihao Zhu, and Bo Zheng. Dr. hair: Reconstructing scalpconnected hair strands without pre-training via differentiable rendering of line segments. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3
- [68] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In Int. Conf. Learn. Represent.,

2023. 2

- [69] Can Wang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2
- [70] Duotun Wang, Hengyu Meng, Zeyu Cai, Zhijing Shao, Qianxi Liu, Lin Wang, Mingming Fan, Xiaohang Zhan, and Zeyu Wang. Headevolver: Text to head avatars via expressive and attribute-preserving mesh deformation. arXiv preprint arXiv:2403.09326, 2024. 2
- [71] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In Adv. Neural Inform. Process. Syst., 2024. 2
- [72] Keyu Wu, Yifan Ye, Lingchen Yang, Hongbo Fu, Kun Zhou, and Youyi Zheng. Neuralhdhair: Automatic high-fidelity hair modeling from a single image using implicit neural representations. In IEEE Conf. Comput. Vis. Pattern Recog.,

2022. 3

- [73] Keyu Wu, Lingchen Yang, Zhiyi Kuang, Yao Feng, Xutao Han, Yuefan Shen, Hongbo Fu, Kun Zhou, and Youyi Zheng.

- Monohair: High-fidelity hair modeling from a monocular video. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3
- [74] Menghua Wu, Hao Zhu, Linjiang Huang, Yi Zhuang, Yuanxun Lu, and Xun Cao. High-fidelity 3d face generation from natural language descriptions. IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [75] Sijing Wu, Yichao Yan, Yunhao Li, Yuhao Cheng, Wenhan Zhu, Ke Gao, Xiaobo Li, and Guangtao Zhai. Ganhead: Towards generative animatable neural head avatars. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 1
- [76] Yunjie Wu, Yapeng Meng, Zhipeng Hu, Lincheng Li, Haoqian Wu, Kun Zhou, Weiwei Xu, and Xin Yu. Text-guided 3d face synthesis-from generation to editing. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2
- [77] Haotian Yang, Hao Zhu, Yanru Wang, Mingkai Huang, Qiu Shen, Ruigang Yang, and Xun Cao. Facescape: a large-scale high quality 3d face dataset and detailed riggable 3d face prediction. In IEEE Conf. Comput. Vis. Pattern Recog., 2020. 2
- [78] Lingchen Yang, Zefeng Shi, Youyi Zheng, and Kun Zhou. Dynamic hair modeling from monocular videos using deep neural networks. ACM Trans. Graph., 2019. 3
- [79] Tarun Yenamandra, Ayush Tewari, Florian Bernard, HansPeter Seidel, Mohamed Elgharib, Daniel Cremers, and Christian Theobalt. i3dmm: Deep implicit 3d morphable model of human heads. In IEEE Conf. Comput. Vis. Pattern Recog., 2021. 1
- [80] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 2, 3, 6
- [81] Taoran Yi, Jiemin Fang, Zanwei Zhou, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Xinggang Wang, and Qi Tian. Gaussiandreamerpro: Text to manipulable 3d gaussians with highly enhanced quality. arXiv preprint arXiv:2406.18462, 2024. 2
- [82] Cem Yuksel, Scott Schaefer, and John Keyser. Hair meshes. ACM Trans. Graph., 2009. 3, 4
- [83] Egor Zakharov, Vanessa Sklyarova, Michael Black, Giljoo Nam, Justus Thies, and Otmar Hilliges. Human hair reconstruction with strand-aligned 3d gaussians. In Eur. Conf. Comput. Vis., 2024. 3
- [84] Hao Zhang, Yao Feng, Peter Kulits, Yandong Wen, Justus Thies, and Michael J. Black. Teca: Text-guided generation and editing of compositional 3d avatars. In International Conference on 3D Vision, 2024. 2, 3, 6, 7
- [85] Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, and Jingyi Yu. Dreamface: Progressive generation of animatable 3d faces under text guidance. ACM Trans. Graph., 2023. 2
- [86] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Int. Conf. Comput. Vis., 2023. 2
- [87] Rui Zhao, Wei Li, Zhipeng Hu, Lincheng Li, Zhengxia Zou, Zhen Xia Shi, and Changjie Fan. Zero-shot text-to-parameter

- translation for game character auto-creation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [88] Mingwu Zheng, Hongyu Yang, Di Huang, and Liming Chen. Imface: A nonlinear 3d morphable face model with implicit neural representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022. 1
- [89] Yujian Zheng, Zirong Jin, Moran Li, Haibin Huang, Chongyang Ma, Shuguang Cui, and Xiaoguang Han. Hairstep: Transfer synthetic to real using strand and depth maps for single-view 3d hair modeling. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2, 3
- [90] Yujian Zheng, Yuda Qiu, Leyang Jin, Chongyang Ma, Haibin Huang, Di Zhang, Pengfei Wan, and Xiaoguang Han. Towards unified 3d hair reconstruction from single-view portraits. arXiv preprint arXiv:2409.16863, 2024.
- [91] Yi Zhou, Liwen Hu, Jun Xing, Weikai Chen, Han-Wei Kung, Xin Tong, and Hao Li. Hairnet: Single-view hair reconstruction using convolutional neural networks. In Eur. Conf. Comput. Vis., 2018. 3
- [92] Yuxiao Zhou, Menglei Chai, Alessandro Pepe, Markus Gross, and Thabo Beeler. Groomgen: A high-quality generative hair model using hierarchical latent representations. ACM Trans. Graph., 2023. 2, 3
- [93] Yuxiao Zhou, Menglei Chai, Daoye Wang, Sebastian Winberg, Erroll Wood, Kripasindhu Sarkar, Markus Gross, and Thabo Beeler. Groomcap: High-fidelity prior-free hair capture. arXiv preprint arXiv:2409.00831, 2024. 3
- [94] Zhen Zhou, Fan Ma, Hehe Fan, and Yi Yang. Headstudio: Text to animatable head avatars with 3d gaussian splatting. In Eur. Conf. Comput. Vis., 2024. 2, 3, 6

## StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors

### Supplementary Material

###### (a) Geometry Modeling

###### (b) Texture Modeling

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

###### Normal-Adapted Diffusion Model

𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

[Figure 268]

Differentiable Rendering

Differentiable Rendering

𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

NormalConditioned Diffusion Model

𝒏𝒏𝐡𝐡

𝒏𝒏𝐡𝐡

[Figure 269]

[Figure 270]

[Figure 271]

𝑳𝑳𝑴𝑴𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

𝑳𝑳𝐩𝐩𝐩𝐩𝐩𝐩𝐩𝐩𝐩𝐩

Depth-Adapted Diffusion Model

𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

Bald Head Texture Field 𝝍𝝍𝒉𝒉

Bald Head DMTet 𝜽𝜽𝒉𝒉

Periodic Updating

𝒅𝒅𝐡𝐡

𝒄𝒄𝐡𝐡

“A bald handsome American man.”

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

: Human-Specific 2D Generative Priors : FLAME-Evolving Prior Loss

: Frozen Parameters : Optimized Parameters

Evolvable FLAME

FLAME Mesh 𝑴𝑴𝒐𝒐𝒏𝒏𝒐𝒐𝒊𝒊

[Figure 282]

Mesh 𝑴𝑴𝒐𝒐𝒏𝒏𝒐𝒐𝒊𝒊 Evolving

Figure 11. The process for generating a bald head model involves two steps: (a) Employing human-specific geometry-aware diffusion models and FLAME-evolving prior loss to model realistic and semantic-aligned bald head shapes. (b) Subsequently, using a normalconditioned diffusion model to generate lifelike head textures.

(c) Hair Texture Generation

###### (a) Bald Head Generation

Differentiable Rendering

#### 6. Supplementary Material

In addition, to obtain semantic information for accurate hair initialization, we introduce a FLAME-evolving prior loss, inspired by Barbie [66]. In specific, we freeze the FLAME shape parameters β but improve Minit to evolvable Mˆinit by adding learnable vertex-wise offsets. We periodically fit Mˆinit to the head DMTet every δ iteration, thereby obtaining accurate semantic information. Moreover, Mˆinit provide a reliable and diverse head prior to prevent unnatural geometry using the following formula:

Normal- 𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡 Conditioned

“A bald handsome American man.”

𝒏𝒏𝐡𝐡+𝐬𝐬

Diffusion Model 𝑳𝑳𝑴𝑴𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

Texture Modeling

𝒏𝒏𝐡𝐡

This document includes the following supplementary cont

[Figure 296]

|tent:|
|---|

Render

𝝍𝝍𝐡𝐡

Bald Head DMTet 𝜽𝜽𝒉𝒉

Bald Head Texture Field 𝝍𝝍𝒉𝒉

Hair Strand Texture Field 𝝍𝝍𝒔𝒔

FLAME Mesh 𝑴𝑴𝒉𝒉

Render

- • Bald Head Generation.
- • Implementation Details.
- • More Evaluations.
- • More Experiment Details and Results.
- • Prompt List.
- • Ethics Statement.

𝒄𝒄𝐡𝐡+𝐬𝐬

Normal-Conditioned Diffusion Model

𝒄𝒄𝐡𝐡

[Figure 297]

||
|---|

||
|---|

(b) Hair Geometry Generation

Normal-Adapted Diffusion Model 𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

𝑳𝑳𝒃𝒃𝒃𝒃𝒐𝒐𝒃𝒃 𝑳𝑳𝒇𝒇𝒇𝒇𝒄𝒄𝒇𝒇 𝑳𝑳𝒄𝒄𝒐𝒐𝒄𝒄𝒄𝒄𝒐𝒐

𝑳𝑳𝒐𝒐𝒐𝒐𝒐𝒐 𝑳𝑳𝒄𝒄𝒄𝒄𝒐𝒐

Differentiable Rendering

“A handsome American man with slicked-back sandy blonde hair.”

𝒏𝒏𝐡𝐡+𝐬𝐬

| |
|---|

| |
|---|

zz

Differentiable Prismatization

2 2

, (14)

𝑮𝑮

Lprior =

###### (p) − sMˆ

##### 6.1. Bald Head Generation

(p)

sθ

DiffusionDepth-AdaptedModel 𝑳𝑳𝑺𝑺𝑺𝑺𝑺𝑺𝐡𝐡𝐡𝐡

h

init

p∈P

Neural Scalp Textures 𝑻𝑻

Strand Generator

Following HumanNorm [17] and HeadArtist [30], We use DMTet [60] θh and a texture field ψh to model the geometry and appearance of the bald head respectively. These components are optimized separately in two stages, as illustrated in Fig. 11.

Hair Strands 𝑺𝑺 Hair Prismatic

𝒅𝒅𝐡𝐡+𝐬𝐬

Meshes 𝑴𝑴𝒑𝒑

are the signed distance functions (SDF) of the head DMTet θh and Mˆinit, respectively, and P is a set of randomly sampled points. Through this evolving process, Mˆinit gradually captures rich geometric features (e.g., beards and wrinkles), providing reliable yet diverse priors for subsequent geometry generation (Fig. 12). In summary, the loss function for optimizing the head geometry is as follows:

where sθ

###### and sMˆ

h

init

: Frozen Parameters : Optimized Parameters : Human-Specific 2D Generative Priors : 3D Hair Strand Geometric Priors

Head Geometry Modeling. Specifically, we first initialize the head DMTet θh utilizing the FLAME model Minit. We then refine it under the supervision of human-specific geometry-aware diffusion models (Fig. 11-(a)) by leveraging the following losses:

Lhead-geo = LhnSDS + LhdSDS + λpriorLprior. (15) Head Texture Modeling. Given the generated head geometry generated, we fix it and utilize a texture field ψh, which maps a query position to its color to generate head appearance. Specifically, we construct this field using MLP with multi-resolution hash encoding [40], and we optimize it using the following loss function (Fig. 11-(b)):

∂nh ∂θh

∇θhLhnSDS = Et,ϵ (ϵϕ

(nht;yh,t) − ϵ)

, (12)

hn

∂dh ∂θh

∇θhLhdSDS = Et,ϵ (ϵϕ

(dht;yh,t) − ϵ)

, (13)

hd

where yh is the input bald head text, ϕhn and ϕhd are the human-specialized normal-adapted and depth-adapted diffusion models, respectively, and nh and dh are the rendered normal and depth maps of the bald head.

∂ch ∂ψh

∇ψhLhcSDS = Et,ϵ (ϵϕ

(cht;nh,yh,t) − ϵ)

, (16)

hc

“A wrinkly bald African grandmother.”

where ϕhc is a human-specialized normal-conditioned diffusion model, and ch represents the rendered color image of the generated head. Since the vanilla SDS loss often leads to color oversaturation, we replace it with the following MSDS loss [17] to further enhance the texture’s realism in later iterations of texture optimization:

| |
|---|

| |
|---|

| |
|---|

[Figure 312]

[Figure 313]

[Figure 314]

| |
|---|

| |
|---|

| |
|---|

∂ch ∂ψh

∇ψhLhcMSDS = Et,ϵ (h(cht;nh,yh,t) − ϵ)

+

(17)

∂ch ∂ψh

∂V (ϵ) ∂ϵ

“A bald European grandfather with a gray beard.”

Et,ϵ (V (h(cht;nh,yh,t)) − V (ϵ))

,

| |
|---|

| |
|---|

| |
|---|

[Figure 315]

[Figure 316]

[Figure 317]

where V is the first k layers of the VGG network [63] h(·) denotes the multi-step image generation function of the normal-aligned diffusion model.

| |
|---|

| |
|---|

| |
|---|

##### 6.2. Implementation Details

Hyper-Parameters. For the bald head generation, our approach requires 10,000 iterations for geometry creation, 2,000 iterations for texture synthesis, and 5,000 iterations for appearance refinement using MSDS [17]. The loss weight λprior is set to 1 × 103. For hair generation, the process involves 5,000 iterations for geometry modeling, 2,000 iterations for texture generation, and 5,000 iterations for visual enhancement using MSDS [17]. The respective loss weights for λhnSDS, λori, λcur, λbbox, λface, and λcolli are assigned as 1×10−3, 1×104, 1×104, 1×103, 1×103, and

[Figure 318]

[Figure 319]

full w/o w/o evolving

Figure 12. Ablation study on FLAME-evolving prior loss.

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

“A medium-length straight hairstyle”

“A slicked-back hairstyle”

“A side-swept haircut”

“A short wavy hairstyle” “A short hairstyle”

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

- 1×103 respectively. The number of strand polylines Ns and strand points Np are configured as 3000 and 100, respectively. For different hairstyles—normal, straight, wavy, and curly—the target curvature Ctarget is defined as 5 × 10−2,
- 2 × 10−2, 1 × 10−1, and 2 × 10−1, respectively. All experiments are conducted on an Ubuntu server equipped with A6000 GPUs. Generating a bald head takes roughly 4 hours with 24GB of memory, and generating a haircut takes roughly 4 hours with 44GB of memory. Details of FLAME-Evolving Prior Loss. In the FLAME-

“A side-swept hairstyle”

“A slicked-back haircut”

“A mohawk hairstyle” “A short haircut”

“A short spiky haircut”

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

“A medium-length wavy hairstyle”

“A short bob haircut” “A straight bob haircut” “A wavy bob haircut” “A curly bob haircut”

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

evolving prior loss, we periodically optimize Mˆinit to fit the current human head geometry at every 1000 iterations, thus providing an effective and flexible human head prior constraint. The fitting loss function is as follows:

“A medium-length curly hairstyle”

“A long straight haircut”

“A long wavy haircut”

“A afro” “A short afro”

Figure 13. Initialize hairstyles. Since the USC-HairSalon Dataset [15] lacks afro hairstyles, we use HAAR [65] to generate the initial afro hairstyle.

Lfit = λchamfLchamf+λedgeLedge+λnorLnor+λlapLlap, (18)

where Lchamf is the Chamfer distance between Mˆinit and the current human head geometry, Ledge is the edge length regularization loss, Lnor is the normal consistency loss, and Llap is the Laplacian smoothness loss. The loss weights λchamf, λedge, λnor, and λlap are set to 1 × 102, 1 × 100, 1 × 10−2, and 1 × 10−1, respectively. The ablation study results for the FLAME-evolving prior loss are presented in Fig. 12. As shown, excluding Lprior results in exaggerated head shapes (e.g., overly pointed head tops), significantly compromising the realism of the outputs. Employing a non-evolving prior

loss achieves reasonable head proportions but fails to preserve finer geometric details such as beards and wrinkles. By leveraging the full Lprior, our approach generates head geometry with both accurate proportions and high detail fidelity.

Details of Hair Initialization. As mentioned in the main paper, we utilize ChatGPT[43] to select the most representative hairstyles from the USC-HairSalon Dataset [15]. Specifically, we first exclude some exaggerated hairstyles (e.g., those that are overly long or excessively messy). Next, ChatGPT is used to generate textual descriptions for each

[Figure 340]

[Figure 341]

- Figure 14. Illustration of converting a hair strand into a prismatic mesh using the differentiable prismatization algorithm.

w/ differentiable prismatization w/ NeuralHaircut’s quad mesh

[Figure 342]

[Figure 343]

“A charming Caucasian man with sharp features and slicked-back ash blonde hair.”

“A charming woman with a straight nose and shoulder-length straight red hair.”

[Figure 344]

[Figure 345]

“A confident man with a broad jawline and a side-swept jet-black haircut.”

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

“A cheerful fat woman with medium-length wavy caramel-colored hair.”

- Figure 15. Differentiable prismatization vs. NeuralHaircut’s quad mesh [64].

[Figure 358]

Figure 17. Ablation study on K and R.

- 2. Generate K Rotated Normals: Rotate this normal around the axis defined by the strand’s orientation K

times, each by 360

◦

K , to produce K normals.

- 3. Translate to Form Lateral Edges: Translate s along each of these K normals by R, generating K lateral edges of the prism.
- 4. Construct Lateral Faces: Connect the adjacent lateral edges’vertices to form the K lateral faces of the prism.
- 5. Construct Top and Bottom Faces: Connect the vertices at the ends of the lateral edges to form the top and bottom faces of the prism, completing the conversion from a hair strand to a watertight prismatic mesh.

Figure 14 shows an example of converting a hair strand into a prismatic mesh. Importantly, the proposed differentiable prismatization algorithm can be easily implemented on GPU, achieving flexible and fast prismatization of hair strands and paving a new way for hair modeling.

Specific to our experiment, each hair strand is converted into a watertight prismatic mesh with K = 4 lateral edges. The radius is defined as R = ANscalp

sπ , where Ascalp represents the surface area of the scalp mesh. During the optimization of hair textures, the radius R is further reduced

“Shoulder-length curly sleek blonde hair.”

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

Ascalp Nsπ

to

2 to achieve a more detailed appearance. As illustrated in Fig. 15, our proposed differentiable prismatization algorithm offers more stable gradient backpropagation compared to the quad mesh used by NeuralHaircut [64]. This mitigates abnormal normal problems caused by nonwatertight meshes, ensuring reliable strand-based hair optimization. Additionally, we present an ablation study on the effect of the number of hair strands in Fig. 16. It is observed that as the number of hair strands increases, the generated hairstyles become denser. Meanwhile, the overall shape and texture quality are maintained, demonstrating the robustness of differentiable prismatization. As shown in Fig. 17, we also conduct an ablation study on lateral edges K and the radius R. The results indicate that the number of lateral edges K has a minimal impact on the final output, which validates the effectiveness and robustness of our differentiable prismatization algorithm.

1000 hair strands 2000 hair strands 3000 hair strands

Figure 16. Ablation study on the number of hair strands.

hairstyle, which are then utilized to select the 20 most representative ones. As shown in Fig. 13, the selected hairstyles cover different lengths, curvatures, and styles, ensuring rich diversity. Finally, we optimize neural scalp textures (NST) to fit the selected hairstyle using Eq. (2), where λori and λcur are set 5 × 10−2, and 1 × 100, respectively.

Details of Differentiable Prismatization Algorithm. Given a hair strand s, our differentiable prismization algorithm converts it into a watertight prismatic mesh with K lateral edges and radius R through the following five steps: 1. Compute the Initial Normal Vector: Determine a nor-

mal to the hair strand s by taking the cross product of its orientation with a non-collinear reference point (typically the center of the head).

###### Details of Geometry-Aware Losses. The geometry-aware

“A bald European grandfather with a gray beard.”

“Wavy dark brown hair.”

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Change Image

Input Image

[Figure 371]

[Figure 372]

[Figure 373]

full loss w/o

w/o w/o

“A kind grandmother with big ears and a silver curly bob hairstyle.” CLIP

Figure 18. Ablation study on geometry-aware losses.

Input Text

“Slicked-back ash blonde hair.”

0.2988 0.2556 0.2977 0.3012 0.2691 0.2984 1 0 1 0

[Figure 374]

[Figure 375]

[Figure 376]

Open-CLIP Fashion-CLIP

| |
|---|

| |
|---|

| |
|---|

BLIP-VQA BLIP2-VQA

full w/o orientation w/o uv coordinates

Figure 19. Ablation study on the strand-aware texture field.

Figure 20. Quantitative comparisons for metrics including CLIP, Open-CLIP, Fashion-CLIP, BLIP-VQA, and BLIP2-VQA.

loss functions Lbbox, Lface, and Lcolli are formulated as:

max(0,sbbox(p)), (19)

Lbbox =

Retrieved Hair

Generated Hair

p∈S

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

max(0,−sface(p)), (20)

Lface =

p∈S

max(0,−shead(p)), (21)

Lcolli =

p∈S

“Slicked-back sandy

L2 Distance: 427.92

L2 Distance: 514.60

L2 Distance: 518.06

“Iconic short curly blondehair.” platinum blonde hair.”

where sbbox, sface, and shead are the SDF of the bounding box, the space in front of the face, and the head, respectively. Here, p represents the 3D points of hair strands S.

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

The results of the ablation study on geometry-aware losses are displayed in Fig. 18. Incorporating Lbbox, Lface, and Lcolli effectively prevents the hair from extending beyond the bounding box, obscuring the face, and colliding with the head. This significantly enhances the geometric rationality and realism of the generated hairstyles.

L2 Distance: 412.23

L2 Distance: 489.76

L2 Distance: 496.42

“Iconic short curly platinum blonde hair.”

Figure 21. The generated hair and the retrieved hair from USCHairSalon [15] by ranking the L2 distance of neural scale textures.

Details of Strand-Aware Texture Field. Due to the complexity of hair textures, a basic texture field cannot fully capture the lifelike appearance of strands. For a query point p, the basic texture field ϕb generates its color using the following formula:

Specifically, we introduce two improvements to the basic texture field: first, we replace Euclidean coordinates with scalp UV coordinates, which are more uniformly distributed. Second, we incorporate strand orientations as additional input information to model orientation-dependent texture variations. These enhancements enhance the realism of the generated results by better capturing highfrequency appearance variations and ensuring consistent colors across different faces of a single prismatic mesh since the input features are strand-based. As shown in Fig. 19, our proposed strand-aware texture field accurately models high-frequency appearance details by switching coordinate spaces and incorporating orientation information, resulting in more realistic strand textures.

###### c = ϕb(Euc(p)), (22)

where Euc(·) represents the Euclidean coordinates of the query point. To better model high-frequency color variations, we propose the strand-aware texture field ϕs which uses the following equation:

###### c = ϕs(UV (p),o), (23)

where UV (·) denotes the scalp UV coordinates of the query point, and o refers to its strand orientation.

Method FID ↓ CLIP ↑

HeadArtist 201.35 27.69 HeadStudio 271.37 27.61

HumanNorm 211.65 27.96

TECA 178.79 30.60 StrandHead (Ours) 176.95 30.95

MVDream 215.06 28.11 GaussianDreamer 249.96 25.05 LucidDreamer 231.65 25.05

RichDreamer 213.54 27.13 TECA 231.23 26.36 HAAR 335.07 25.42

StrandHead (Ours) 201.19 27.84

Table 3. Quantitative comparisons with the SOTA methods. The best and second-best results are highlighted in bolded and underlined, respectively.

##### 6.3. More Evaluations

For a more comprehensive comparison, we also provide evaluations based on FID and CLIP metrics. We compute the FID between the images rendered by each method and those generated by Flux. As shown in Tab. 3, our method still achieves good results. However, given the inherent flaws of these metrics, we recommend that readers refer to the metrics in the main paper.

##### 6.4. More Experiment Details and Results

Disadvantages of CLIP-Based Metrics. Recent studies [16, 31] have demonstrated that CLIP-based metrics are limited to assessing coarse text-image similarity and struggle to capture the fine-grained correspondence between 3D content and input prompts accurately. To address this limitation, we follow Progressive3D [7] and utilize finegrained text-to-image evaluation metrics, such as BLIPVQA [23, 24] and BLIP2-VQA [23, 25], to assess the generative capabilities of different methods.

As depicted in Fig. 20, when input images are replaced while keeping the input text unchanged, interesting observations can be made: Open-CLIP and Fashion-CLIP scores increase under these conditions. However, BLIP-VQA and BLIP2-VQA scores drop significantly. This highlights the limitations of CLIP-based metrics in evaluating fine-grained correspondences. In contrast, BLIP-VQA and BLIP2-VQA demonstrate superior performance in capturing these intricate relationships.

Clarification on Retrieval-Like Results. Since there is no GT available for text-driven hair generation, we display the generated hair alongside its top-3 nearest haircuts from the USC-HairSalon dataset [15]textures, as shown in Fig. 21. The similarity is ranked based on the L2 distance of neural scale textures. As illustrated, the generated hair is significantly different from the retrieved hair samples. This distinction clearly demonstrates that our method generates unique hairstyles rather than simply retrieving them from the dataset.

More Experiment Results. We present additional 3D hairdisentangled head avatars in Fig. 22 and 3D strand-based

hair generated by StrandHead in Fig. 23.

Effect of Human-Specific 2D Generative Priors. We demonstrate the importance of human-specific 2D generative priors from two aspects:

- (1) Fig. 24 displays optimized hair under varying text

conditions, while keeping the initial hairstyle and bald head constant. As shown, our method, leveraging human-specific

- 2D generative priors, accurately captures subtle changes in textual descriptions (e.g., variations in haircut length and curliness). The generated 3D strand-based hair not only exhibits a realistic and well-structured shape but also aligns closely with the given text conditions.

(2) Fig. 25 illustrates generated hair under varying bald head conditions while maintaining a fixed initial hairstyle and hair prompt. As observed, thanks to the powerful 2D diffusion models pre-trained on human data, the generated

- 3D hair strands exhibit geometry and texture variations that adapt to specific bald heads.

In summary, our robust optimization strategy ensures that the generated hair is not only highly consistent with the text prompts but also integrates harmoniously with the human head.

##### 6.5. Prompt List

The following are textual prompts for quantitative experiments:

- • A beautiful girl with delicate features and long, silky black wavy hair.
- • A cheerful fat European man with a short, spiky light brown haircut.
- • A confident black woman with a curly dark brown afro.
- • A handsome American man with slicked-back sandy blonde hair.
- • A kind grandmother with big ears and a silver curly bob hairstyle.
- • A lively European boy with tousled light brown hair.
- • A lively black girl with tight, curly dark brown hair.
- • A mature African man with deep-set eyes and a short, curly black afro.
- • A mature European woman with a straight nose and medium-length, straight auburn hair.
- • A middle-aged Hispanic woman with a strong jawline and medium-length, wavy magenta hair.
- • A muscular European man with a wide forehead and slicked-back black hair.
- • A serious white man with a sharp nose and a slicked-back, jet-black hairstyle.
- • A sexy woman with full lips and long, wavy chestnut brown hair.
- • A strong American man with a gray beard and a curly silver haircut.
- • A strong man with a broad nose and a short, spiky dark brown haircut.

- • A strong man with a strong jaw and a medium-length, wavy black hairstyle.
- • A stylish African woman with a sleek, shoulder-length black bob haircut.
- • A thin man with a sharp jawline and a spiky light brown mohawk hairstyle.
- • A wise elderly European grandfather with wrinkles and a side-swept silver haircut.
- • A wise, elderly grandfather with a classic short curly white haircut.
- • An elderly African grandmother with wrinkles and a curly gray afro.
- • An elderly Caucasian grandmother with thin lips and a straight, short silver bob.
- • Angelina Jolie with long, wavy chestnut brown hair.
- • Beyonc´e with voluminous, curly honey blonde hair.
- • Emma Watson with a short, wavy brown bob.
- • Lionel Messi with a blonde mohawk hairstyle.
- • Michael Jordan with a short, neatly-trimmed black buzz cut.
- • Morgan Freeman with a curly brown afro.
- • Rihanna with a sleek, long black hairstyle.
- • Taylor Swift with long, straight platinum blonde hair. 6.6. Ethics Statement

StrandHead provides an efficient solution for creating realistic 3D head avatars with strand-based hair using 2D/3D human-centric priors, enabling a wide range of applications. However, like many AI generative technologies, it carries the risk of misuse, such as the creation of misleading avatars. To mitigate these concerns, future research in generative AI should emphasize ethical considerations, develop effective safeguards, and promote responsible practices. By addressing these challenges, developers can reduce potential harm while maximizing the positive impact of these technologies.

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

“A mature European woman with a straight nose and medium-length, straight auburn hair.”

“A handsome Hispanic man with a strong jawline and a short curly deep brown haircut.”

“A handsome American man with slicked-back sandy blonde hair.”

“Rihanna with a sleek long black hairstyle.”

“A charming woman with a straight nose and shoulder-length, straight red hair.”

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

“Leonardo DiCaprio with a short, slicked-back light brown haircut.”

“A beautiful Caucasian girl with delicate features and long silky golden blonde hair.”

“A cheerful fat European man with a short, spiky light brown haircut.”

“Emma Watson with a short, wavy brown bob.”

“A confident black woman with a curly dark brown afro.”

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

“A wise elderly European grandfather with wrinkles and a side-swept silver haircut.”

“A charming Caucasian woman with a straight nose and shoulder-length curly sleek blonde hair.”

“A charming Caucasian man with sharp features and slicked-back, ash blonde hair.”

“A lively black girl with tight, curly dark brown hair.”

“A kind African grandfather with short, curly gray hair.”

Figure 22. Examples of 3D head avatars with strand-based attributes.

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

“Iconic short curly platinum blonde hair.”

“Slicked-back sandy blonde hair.”

“Shoulder-length, straight red hair.”

“A curly dark blonde haircut.”

“A tight curly afro.”

“A short neatlytrimmed black buzz cut.”

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

“Long silky black wavy hair.”

“Slicked-back, ash blonde hair.”

“A textured blonde bob.”

“Side-swept sandy blonde hair.”

“A straight, short silver bob.”

“Medium-length, tousled blonde hair.”

Figure 23. The physics-based hair strand rendering result using Blender [10].

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

“A beautiful girl with delicate features.”

“A long wavy haircut.”

“Long silky black straight hair.”

“Long silky black curly hair.”

“Long silky black wavy hair.”

“Medium-length silky black straight hair.”

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

InitialHairstyle+BaldHead

“A charming woman with a straight nose.”

“Shoulder-length straight red hair.”

“Shoulder-length wavy red hair.”

“A long straight haircut.”

“Short wavy red hair.” “Shoulder-length red hair.”

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

“A curly bob haircut.”

“Voluminous curly honey blonde hair.”

“Voluminous straight honey blonde hair.”

“Rihanna.”

“Long curly honey blonde hair.” “Voluminous honey blonde hair.”

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

“A wavy bob haircut.”

“Emma Watson.”

“A short wavy brown bob.” “A long wavy brown bob.” “A short brown bob.” “A short curly brown bob.”

Hairstyle Prompt

- Figure 24. The optimization results under different hair prompts. Starting from the same initial hairstyle, StrandHead demonstrates its capability to generate diverse hairstyles by adapting to varying text conditions.

InitialHairstyle+HairstylePrompt

Bald Head

[Figure 467]

“A long straight haircut.”

“A curly bob haircut.”

[Figure 468]

“A short bob haircut”

[Figure 469]

“A medium-length curly hairstyle”

[Figure 470]

“A short, curly dark blonde bob”

“Mediumlength, curly hair.”

“Long, straight platinum blonde hair.”

“Tight, curly dark brown hair.”

[Figure 471]

“A beautiful girl with delicate features.”

“A handsome American man.”

[Figure 472]

“Leonardo DiCaprio.”

[Figure 473]

“Marilyn Monroe.”

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

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

- Figure 25. The optimization results under different bald heads. Under consistent text conditions, the 3D hair generated by StrandHead exhibits specific geometry and texture variations that seamlessly adapt to the unique features of different bald heads.

