Pacific Graphics 2025 M. Christie, N. Pietroni, and Y.-S. Wang (Guest Editors)

COMPUTER GRAPHICS forum

Volume 44 (2025), Number 7

## StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation

Seungmi Lee∗ Kwan Yun∗ Junyong Noh

KAIST, Visual Media Lab ∗Contributed equally to this work

# arXiv:2508.11203v1[cs.GR]15Aug2025

### Abstract

We introduce StyleMM, a novel framework that can construct a stylized 3D Morphable Model (3DMM) based on user-defined text descriptions specifying a target style. Building upon a pre-trained mesh deformation network and a texture generator for original 3DMM-based realistic human faces, our approach fine-tunes these models using stylized facial images generated via text-guided image-to-image (i2i) translation with a diffusion model, which serve as stylization targets for the rendered mesh. To prevent undesired changes in identity, facial alignment, or expressions during i2i translation, we introduce a stylization method that explicitly preserves the facial attributes of the source image. By maintaining these critical attributes during image stylization, the proposed approach ensures consistent 3D style transfer across the 3DMM parameter space through imagebased training. Once trained, StyleMM enables feed-forward generation of stylized face meshes with explicit control over shape, expression, and texture parameters, producing meshes with consistent vertex connectivity and animatability. Quantitative and qualitative evaluations demonstrate that our approach outperforms state-of-the-art methods in terms of identity-level facial diversity and stylization capability. The code and videos are available at kwanyun.github.io/stylemm_page.

Categories and Subject Descriptors (according to ACM CCS): I.3.6 [Computer Graphics]: Methodology and TechniquesCCS Concepts

• Computing methodologies → Computer graphics; Computer vision representations;

1. Introduction

A parametric face model allows for the instant creation of diverse 3D avatars by varying adjustable parameters. This significantly reduces the labor and cost associated with individually modeling each avatar for digital content production. The resulting faces share a unified structural framework—including vertex connectivity, UV maps, and rigs—which enables efficient asset management in editing and reuse tasks. Moreover, the interactive generation process through input parameters facilitates seamless expansion into personalized and customized content. A well-known example is the 3D Morphable Face Model (3DMM), which parameterizes both the shape and expression of human faces and is widely employed for head avatar generation.

Beyond realistic human representations, a parametric face model designed for artistic expression or fictional appearances is particularly valuable in applications such as film, animation, and game production. We define such models as stylized 3D morphable face models (stylized 3DMMs). While stylized 3DMMs must capture a broad range of expressive styles beyond those represented in realistic face models, they should also support identity-level facial variation and maintain a consistent underlying structure across gener-

ated assets. Furthermore, to enhance user engagement and creative flexibility, the models should offer intuitive control, enabling creators to rapidly explore diverse appearance variations and achieve desired visual outcomes.

To formalize these requirements, we identify three key elements of a stylized 3DMM. The first two are adopted from the definition of realistic 3DMMs [EST∗20,BV99], while the third is specific to stylized models.

- 1. All generated faces share dense point-to-point correspondences, ensuring consistent mesh structure across instances. (Maintained Correspondence)
- 2. Facial shape and color are disentangled and can be independently controlled. (Disentangled Control)
- 3. Expressive stylization of both geometry and texture that extends beyond realistic face models. (Stylization Beyond Realistic Geometry and Texture)

Previous studies have successfully addressed one or two of these elements. Unfortunately, no existing method fulfills all three key requirements of a stylized 3DMM. A detailed discussion is provided in the following paragraphs.

Table 1: Comparison of different 3D face stylization methods. Circles indicate that the method fully meets a criterion, while triangles indicate partial fulfillment -for example, when the geometry satisfies the criterion but the texture does not. StyleMM satisfies all three elements of stylized parametric face models.

Maintained Correspondence

Disentangled Control

Beyond Realistic Geometry & Texture

Method

StyleMM (Ours) ⃝ ⃝ ⃝ LeGO [YYS∗24] ⃝ △ △ CLIPFace [ATDN22] ⃝ ⃝ △ Ultravatar [ZHXQ24] ⃝ △ △ HeadEvolver [WMC∗24] ⃝ × ⃝ ToonifyGB [JHH25] △ × ⃝

Recent neural rendering-based methods [ALZ∗23, KC23, HCH∗23, ZMF∗24, BBBGGD24, JHH25, SCL∗24] have demonstrated promising capabilities in generating stylized 3D faces. Because these methods do not use consistent connectivity, however, they lack registered mapping and structural coherence, making it difficult to maintain dense point-to-point correspondence across identities. Furthermore, shape and color are entangled in these models due to their representations, which employ NeRF [MST∗21] or 3D Gaussian Splatting [KKLD23].

Template face deformation methods based on iterative optimization [LYX∗24,WMC∗24,ZHXQ24,ZQL∗23] have also been proposed. These methods generally leverage priors from pre-trained diffusion models to refine both the geometry and texture of a realistic 3DMM. These techniques successfully produce high-quality stylized meshes typically for a single identity from each optimization process. One drawback of these methods is that geometric and textural components are typically entangled without unified texture mapping. Additionally, the per-character optimization required by these methods introduces substantial computational overhead, significantly restricting their scalability for broader applications.

A few studies have extended parametric face models to stylized domains through deep-learning-based surface deformation methods [JJK∗22,YYS∗24]. These approaches primarily focus on geometric stylization by deforming a template mesh according to input latent codes, thereby maintaining dense correspondences and enabling controlled geometric deformation. However, these methods do not incorporate texture manipulation, limiting their ability to independently control and stylize facial textures. Moreover, they require stylized 3D datasets for training, restricting accessibility for novice users and constraining the range of achievable stylistic diversity.

In this work, we propose StyleMM, an automated framework that fulfills the three essential elements of a stylized 3DMM: dense point-to-point correspondence, separate and independent control over shape and texture through effective disentanglement, and stylization capabilities that extend beyond the limits of realistic face models. As summarized in Table 1, StyleMM uniquely addresses these criteria by directly leveraging textual descriptions of desired styles, which eliminates the need for stylized 3D datasets. Specifically, StyleMM employs diffusion models and their text-guided

[Figure 1]

Figure 1: Comparison between EAS and baseline text based i2i methods. Top row: Disney character, Bottom row: green Orc. See Section 4.3.2 for experimental details.

image-to-image (i2i) translation capabilities. It fine-tunes a surface deformation network and a texture generator, initially designed for realistic 3DMM faces, using rendered stylized images as training targets. This approach ensures dense correspondence across generated meshes, independently controllable shape and texture parameters, and the capability for extensive geometric and textural stylization.

In the process of using stylized images as deformation targets, we found that existing i2i translation methods often alter facial expressions, alignments, or facial structures, posing a significant challenge for training 3D models based on rendered images (See Figure 1). Even a small misalignment can hinder the intended deformation of the mesh. To address this, we propose Explicit Attribute-preserving Stylization (EAS), which utilizes explicit facial attributes of sparse facial landmarks, head rotation, and expression. EAS helps preserve key facial attributes, improving the quality and consistency of image-based training.

Building on these images, we train a stylized 3DMM to model various identities within a given style. In order to derive 3D cues from stylized images with sparse geometric information, we adopt a progressive training strategy consisting of three stages: geometry warm-up for mesh deformation, joint fine-tuning of shape and texture, and further texture refinement. To stabilize face diversity of the original 3DMM during stylization, we introduce a Consistent Displacement Loss (CDL), which promotes coherent mesh deformation across diverse identities and facilitates the generalization of stylization under diverse shape conditions.

Our contributions can be summarized as follows:

- • We formalized the requirements for a stylized 3DMM and introduced the first method to fulfill them by fine-tuning the realistic face generator in a three-stage process.
- • We developed Explicit Attribute-preserving Stylization (EAS), a new facial image stylization pipeline using adjusted noise initialization.
- • We proposed the Explicit Attribute-preserving Module (EAM), which integrates facial landmarks, rotation, and expression into an existing diffusion model to preserve desired attributes during generation.
- • We formulated a novel CDL loss that enhances identity-level face diversity by preventing mode collapse.

- 2. Related Work

- 2.1. Morphable Face Model

Morphable face models, introduced by Blanz and Vetter [BV99], represent faces using dense geometry and texture modeled via principal component analysis (PCA), enabling intuitive control over attributes such as gender and fullness. The Basel Face Model [PKA∗09]—subsequently expanded to a dataset of 10,000 facial scans [BRZ∗16, BRP∗18]—laid the foundation for modern face modeling. Subsequently, FaceWarehouse [CWZ∗13] introduced multi-linear models to capture identity and expression variations. More recently, FLAME [LBB∗17] and ICTFaceKit [LBZ∗20] have leveraged pose-dependent corrective blendshapes and non-linear expression models, derived from larger facial-scan datasets, to further refine facial expressions. In this work, we utilize FLAME as the source model, distill it into a learning-based model, and then fine-tune it to obtain a new morphable model.

- 2.2. Text to 3D Face Generation

Recent advancements in text-to-3D generation, driven by progress in text-to-image models [RBL∗22, SCS∗22] and vision-language models [RKH∗21], have demonstrated remarkable capabilities. One line of research leverages neural fields such as NeRF [MST∗21], 3DGS [KKLD23], or DMTet [SGY∗21]. Despite their success in producing high-quality stylized characters [ALZ∗23, ZCF∗23, HSZ∗24, LWW∗24, ZMF∗24] via multiview rendering, generating meshes with a consistent structure remains challenging, limiting compatibility with existing graphics pipelines.

Another line of work builds upon established face models such

- as FLAME and ICT-FaceKit, achieving high-quality textures while preserving compatibility with standard graphics pipelines, owing to their consistent mesh structure. For instance, CLIPface [ATDN22] generates texture maps for a given FLAME model, while DreamFace [ZQL∗23] and UltraAvatar [ZHXQ24] sample or estimate blendshape parameters to obtain geometry while generating the textures. These methods confine shapes within the realistic 3DMM space, restricting geometric exaggerations or abstractions beyond the distribution of scanned-face datasets.

- 2.3. Surface Deformation Network

Learning implicit functions for 3D shapes has proven highly effective in representing complex geometries [PFS∗19, MON∗19, MPJ∗19, Lip21]. DIF-Net [DYT21] uses MLPs to learn a signed distance function (SDF) alongside a volumetric deformation function, while DD3C [JJK∗22] and LeGO [YYS∗24] adopt surface deformation for face stylization. Although DD3C and LeGO demonstrate the creation of impressive stylization results, both rely on ground-truth target meshes created by skilled artists for training, thereby limiting accessibility for novice users. Building on their finding that surface deformation is well-suited for facial modeling, we advance this approach by eliminating the need for ground-truth target meshes. Specifically, we use text descriptions to generate target images and then deform the surface mesh accordingly.

### 2.4. Text-Based Portrait Stylization

Text-based portrait stylization focuses on stylizing a given image while preserving the subject’s identity. Both GAN-based [GPM∗22, PWS∗21] and diffusionbased [LCW∗24, YZL∗23, WBW∗24] methods have made notable strides by leveraging pre-trained generative models to balance style and identity. Unfortunately, these approaches are not designed for 3D deformation, making it difficult to guarantee precise alignment with the source image. Therefore, we propose an explicitly aligned stylization approach that ensures not only identity and style preservation but also spatial alignment and retention of the subject’s expressions.

### 3. Methods

Our goal is to build a 3DMM for stylized faces, which reflects the style described by a user-provided text prompt. To achieve this, we fine-tune two pre-trained networks Dsrc and Gsrc, both originally trained on natural human 3DMM faces (Section 3.1). According to the given style description, our framework automatically generates a dataset of stylized facial images (Sections 3.2 and 3.3) and finetunes the networks by comparing rendered results with the image data (Section 3.4). After training, the resulting models, denoted as Dstyle and Gstyle, operate together as a stylized 3DMM conditioned on shape and texture, respectively. An overview of the StyleMM training process is shown in Figure 2.

[Figure 2]

Figure 2: StyleMM fine-tunes Deformation network Dsrc and texture generator Gsrc pre-trained on FLAME into stylized models Dstyle and Gstyle, respectively using Explicit Attribute-preserving Stylization (EAS). In this process, the Explicit Attribute-preserving Module (EAM) is a component of EAS that enables the preservation of alignments.

3.1. Pre-trained Networks 3.1.1. Surface Deformation Network

Recent studies have shown that a surface deformation network can efficiently modify geometry while preserving the underlying mesh structure in face stylization [JJK∗22, YYS∗24]. Therefore, we adopt the surface deformation network from LeGO [YYS∗24], which deforms a template face mesh using FLAME shape β and expression ψ parameters. Specifically, given vertices on a template meshV0 ∈ RN×3 (N is the number of vertices), the network outputs per-vertex offsets ∆V such that the final mesh is V = V0 +∆V.

This pre-trained model Dsrc provides the geometry of a 3DMM

[Figure 3]

- Figure 3: Overview of Explicit Attribute-preserving Stylization. Left: training process of EAM. Right: Inference process of EAS, equipped with EAM.

capable of generating diverse face shapes and expressions while maintaining consistent vertex connectivity.

### 3.1.2. Texture Generator

For texture synthesis, we utilize a pre-trained StyleGAN2 [KLA∗20] model capable of generating textures in UV space. Specifically, a latent code z is first mapped to an intermediate style vector w, which then modulates the convolution layers to generate a high-resolution RGB texture. Originally trained on natural human face datasets [BKZ∗23, KLA19], the model, denoted as Gsrc, can synthesize detailed facial textures.

### 3.2. Explicit Attribute-preserving Module

For image-based fine-tuning, our framework constructs a stylized image dataset by pairing each stylized image with its corresponding source image based on user-defined style text. Specifically, we render natural human faces using pre-trained networks Dsrc and Gsrc, and apply the desired style using EAS. Although previous portrait stylization methods are well-suited for both stylization and identity preservation [YZL∗23,LCW∗24], they often introduced changes in alignment or expression in our experiments (See Figure 1). Such inconsistencies can hinder building a 3D face model due to unaligned deformation targets.

To address this, we propose EAS, which is composed of SDXL [PEL∗23] and a newly proposed Explicit Attributepreserving Module (EAM) as shown in Figure 3. The EAM provides explicit conditions to SDXL to preserve attributes of the source face, such as scale, rotation, alignment, and expression. The EAM consists of three condition-encoding MLPs, an Attribute

Merger, a series of EAM blocks, and an SDXL Encoder. The weights of the SDXL Encoder are initialized using those of the pre-trained SDXL, following the approach of ControlNet [ZRA23]. Sparse landmarks (lmk), head rotation (θ), and facial expression (ψ) are passed through their respective encoding MLPs, merged before being fed to a series of EAM Blocks. Each EAM block accepts the processed feature as input, upsample them, and feed them to convolutional layers equipped with Adaptive Layer Normalization initialized with zero [PX23].

To train the EAM, we randomly sample shape parameters β, expression parameters ψ, and texture latent codes z from normal distribution, which represent a source human face. We then render the output faces with random rotation angles θ. Both θ and ψ are directly used as inputs to the EAM during training, while lmk is extracted using a pre-trained network [YZZZ24]. We use a sparse subset of five landmarks—two from the eyes, one from the nose, and two from the lips—to guide alignment between the stylized face and the source image. This selection allows stylization of facial features while avoiding constraints on outer facial shape (e.g., the chin), which is often meant to be stylized. The training loss for the EAM can be expressed as follows:

Ltrain = Ex0,t,txt,lmk,θ,ψ,ϵ∼N(0,1) ∥ϵθ(xt,t,txt,lmk,θ,ψ)−ϵ∥22 ,

(1)

where ϵ is the noise added during diffusion training, and t is the denoising timestep. xt is the noise-added image at timestep t and txt is the text input. Here, we train the EAM with full and partial conditions. For the full condition, we use all three inputs-lmk,θ,Ψwhile for the partial condition, only two input parameters are used. We omit each conditional parameter in 25% of the training samples.

This hybrid training scheme enables the EAM to learn without relying solely on any one parameter, while still effectively leveraging all three when available.

### 3.3. Explicit Attribute-preserving Stylization

We use the trained EAM during inference to stylize rendered faces while preserving their translation, rotation, and expression. To further accelerate inference and preserve the original structure and identity, we initialize the latent variable using noise added source image xt instead of generating images from random noise. Here, t is set to 19 out of total 25 DDIM [SME20] sampling. This adjusted initialization speeds up inference without degrading stylization quality. This is because the early denoising steps focus primarily on coarse geometric structure, while style and details emerge during the middle to later denoising steps [MHS∗21, KC23]. We observed that setting this t lower (e.g., t = 10) led to faster inference speed but significantly decreased stylization capability as shown in Figure 11. The resulting stylized images, paired with the corresponding rendered source images, serve as the synthetic paired data for our image-based fine-tuning process.

### 3.4. Image-Based Learning for 3DMM Stylization

To align the rendered output from the 3D network with the stylized facial images generated by EAS, which are generated under the same conditions, we fine-tune both the surface deformation network and the texture generator. In this process, we found that a single reconstruction loss is insufficient because the 2D supervision provides limited geometric information. Therefore, we divide the training into a three-stage process: an initial geometry warm-up phase, joint fine-tuning of shape and texture, and the final stage for texture refinement. An overview of this training pipeline is illustrated in Figure 4. We will introduce loss terms for the stylization and deformation stabilization in subsections 3.4.1 and 3.4.2 respectively.

- 3.4.1. Style Adaptation Losses To guide the model toward the target, we apply different loss terms

- at various training stages, each extracting meaningful cues from 2D images and transferring them to the 3D face model.

- 3.4.1.1. Geometry Warm-Up. Recall that Dsrc and Dstyle are the surface deformation networks that take the canonical 3D face geometry as input and predict per-vertex offsets to produce the human face and the stylized face, respectively. In this warm-up stage, we finetune Dsrc to become Dstyle to establish an accurate geometric foundation before jointly training Dstyle and Gstyle. Instead of relying on raw pixel values, which can be heavily influenced by textures, we employ 2D keypoint matching as a more stable guide for initial shape alignment. Specifically, we extract a set of facial landmarks from each stylized image using an off-the-shelf detector X-Pose [YZZZ24], and assign the corresponding 3D vertices in the mesh. By projecting these vertices onto the screen space, the discrepancies between the projected vertices and the detected 2D landmarks are penalized.

Lkp =∑

Π(vi)−ki (2)

i

where Π(·) denotes the camera projection function, vi ∈ R3 are the 3D vertices, ki ∈ R2 are the detected 2D landmarks. This approach encourages Dstyle to capture a coarse geometric structure, mitigating potential distractions from complex stylized textures.

- 3.4.1.2. Joint Fine-Tuning. After the warm-up, we jointly finetune the surface deformation network Dstyle and the texture generator Gstyle on the style data from EAS. The joint training combines geometric and textural information, ensuring improved reconstruction fidelity and style adherence. We employ a reconstruction loss between the rendered image Ir and the stylized image Is:

Lrecon = ∥Ir −Is∥2 +λCLIP ·CLIP(Ir,Is)+λDINO ·DINO(Ir,Is)

(3) where CLIP denotes the cosine similarity in CLIP [RKH∗21] embedding space, DINO denotes the cosine similarity of DINOv2 [ODM∗23] feature, and λCLIP and λDINO represent the weighting factor for each reconstruction loss, respectively. The reconstruction loss encourages the rendered image to closely match the stylized target in terms of geometry and texture.

Because the structural and textural information are entangled in the image domain, the reconstruction loss alone often fails to provide accurate geometric detail, especially for fine-grained facial parts. To address this, we employ a segmentation-guided alignment loss computed over facial part segmentation maps, which explicitly guides the model to match the spatial layout of key regions.

Lseg = ∑

c∈C

∥Mask(rc) −Masks(c)∥2. (4)

Here, Mask(rc) and Mask(sc) denote the segmentation masks of class c (eye, nose, ear, and background) from the rendered and stylized images, respectively. Note that this segmentation is generated using our stylized-face segmentation network which will be explained in Section 3.4.4.

- 3.4.1.3. Texture Refinement. In the final stage, we enhance the texture generator to improve the quality of reproduced fine-grained textural details from the style images. We employ a perceptual similarity loss based on LPIPS [ZIE∗18] for local fidelity.

LLPIPS = ∥φ(Ir)−φ(Is)∥2, (5) where φ(·) denotes a LPIPS feature extractor.

To further enhance global texture plausibility, we incorporate an adversarial loss with an additional trained discriminator.

LGAN = logD(Is)+log(1−D(Ir)), (6)

where D(·) denotes the discriminator. This adversarial loss encourages the generator to globally align the generated face with target style distribution when rendered.

### 3.4.2. Deformation Stabilization Losses

Because the style adaptation losses are primarily guided by 2D observations, they can produce unstable or collapsed deformations when applied to 3D models. To address this, we introduce complementary loss terms that stabilize the deformation process and preserve plausible facial geometry throughout training.

[Figure 4]

- Figure 4: Overview of the StyleMM training pipeline. Our method stylizes a 3D morphable face model through a three-stage process that leverages distinct loss functions. A deformation network Dstyle and a texture generator Gstyle are optimized using style-supervised 2D image pairs rendered using random shapes, textures, and viewpoints. The proposed Consistent Displacement Loss LCDL encourages locally consistent deformation patterns across different identities as shown in the bottom right of the figure.

Our goal is to generalize the stylization across the entire shape parameter space, while preserving the structural diversity of the original 3DMM. In our experiments, finetuning the 3D networks with stylized images—each depicting a different individuals and view—often resulted in convergence to a single dominant geometric form, limiting identity diversity. This may be caused by the inherent randomness of the diffusion model, such that regardless of facial attribute alignment, intensity or visual interpretation of the style may vary at each inference. To address this, we propose a Consistent Displacement Loss, denoted as LCDL. LCDL encourages coherent deformation by aggregating partial cues into consistent displacement patterns across identities in a batch.

LCDL = Var vertex_tangent V(i) −V(prei)

B i=1

(7)

where vertex_tangent(·) denotes projection onto the local tangent space of each vertex, and Var(·) computes the variance across a batch of B samples with different shape parameters. By aligning style deformation in the parameter space, LCDL helps preserve the shape diversity of the original 3DMM.

In addition, we apply an auxiliary regularization loss to preserve the pretrained network’s knowledge of plausible face geometry, including vertex positions, surface normals, and internal face angles.

Lreg = λv V −Vpre 22 +λn N −Npre 22

(8)

cos(α)−cos(αpre) 2

+λang ∑

∑

α∈{αt1,αt2,αt3}

t∈T

whereV, N, and α denote the current vertex positions, normals, and internal face angles produced by Dstyle given a shape parameter β and expression parameter ψ. Vpre, Npre, and αpre denote the corresponding outputs from the pre-trained model Dsrc under the same parameters.

### 3.4.3. Overall Training Objective

Based on the style adaptation and deformation stabilization losses introduced in the previous sections, we define the overall training objective for each stage of the stylization process. Here, each λ value is a weighting factor.

Geometry Warm-up:

Lwarm = λkpLkp +λregLreg +λCDLLCDL. (9) Joint Fine-tuning:

Ljoint = λreconLrecon +λsegLseg +λregLreg +λCDLLCDL. (10)

Texture Refinement:

Ltex = λLPIPSLLPIPS +λGANLGAN. (11) The style adaptation losses are applied differently at each of the three steps. Specifically, Lkp is applied during the geometry warmup stage, Lrecon and Lseg are applied during the joint fine-tuning stage, and LLPIPS and LGAN are applied during the texture refinement stage. In contrast, both of the deformation stabilization losses are employed during the warm-up and joint fine-tuning stages.

### 3.4.4. Image Segmentation

Facial part segmentations are required to apply the segmentationguided alignment loss on stylized images. The segmentations are also used for image masking to exclude regions such as the inner mouth, eyes, and background, which are irrelevant to stylized 3DMM construction. Because no existing method can accurately mask the desired parts of diverse stylized faces, and general segmentation models [KMR∗23] cannot segment detailed areas, we introduce a simple feature-based masking network fmask for stylized faces, trained in a few-shot manner. We manually create five

[Figure 5]

facial images for each of five different styles, resulting in 25 training pairs. Each pair is annotated with six segments: eyes, nose, ears, mouth, face, and outer face including hair and background.

For efficient training, we use features from pre-trained models [XOP∗23, ODM∗23, DOMB23], known for their efficacy in downstream tasks [ZHH∗23,YSS∗24,TJW∗23,YKS∗24], by fusing and decoding them into a mask using a zero-initialized Residual Convolutional network [HZRS16]. Specifically, we fuse the features from Stable Diffusion [RBL∗22] and DINOv2 [ODM∗23], inspired by SD-DINO [ZHH∗23]. The training is conducted with

Figure 5: Stylized faces rendered with the same UV pattern to visualize vertex correspondence. Key regions remain aligned across identities.

the cross-entropy loss L = −Nmask1 ∑Ni=mask1 log y ˆi,yi , where yˆ is the estimated mask label, y is the ground-truth mask label, and Nmask is a total number of mask labels. After training fmask, we obtain a mask M for each stylized image and use them during joint finetuning and texture refinement process. For a mesh, we define the part regions in the UV space, render masks separately, and apply them to ensure consistency.

### 4.2.2. Disentangled Control

To demonstrate that our stylization model functions as a 3DMM, we present parametric control results for shape, expression, and texture. Figure 6 visualizes this through the three sets of variations. The left block illustrates shape variations. It shows that the model inherits a structured control space from the pretrained 3DMM, where variations align with prominent facial features. The middle block demonstrates expression-level control. Because our framework maintains vertex-level correspondence even with pretrained 3DMM faces, it supports the direct reuse of expression blendshape basis defined in the original 3DMM. The right block shows diverse texture variations. The generated textures maintain identityrelevant details inherited from the latent space of the pretrained texture generator, while allowing for diverse stylization. Overall, StyleMM enables independent parametric control over key facial attributes, consistent with the original 3DMM.

- 4. Experiments

- 4.1. Implementation Details

Training EAM and EAS inference were conducted on a system with a single RTX A6000 GPU while StyleMM were trained and evaluated on a system with a single RTX 3090 GPU. The EAM was trained for a total of 20,000 iterations with a learning rate of 1e-

- 5. StyleMM was trained on 10,000 paired images with a learning rate of 5e-5 for both the surface deformation network and the texture generator. Warmup training was performed for 2 epochs with

### 4.2.3. Stylization Beyond Realistic Geometry and Texture

the weighting factors λkp, λreg, and λCDL from Eq. (9) set to 300, 4, and 4000, respectively. Joint fine-tuning was performed for 4

To examine the expressive capacity of our stylized 3DMM beyond the realistic 3DMM, we visualize the distribution of generated identities based on their mesh geometry. We randomly sampled a total of 1,000 face meshes across different styles and applied t-SNE to their vertex coordinates to project them into a 2D space. As shown in Figure 7, stylized identities form distinct clusters that extend beyond the region occupied by the original FLAME model. This suggests that our framework can generate geometries that surpass the anthropomorphic bounds of conventional 3DMMs, enabling more expressive and creative stylization. We also demonstrate expressive capacity in texture generation by visualizing the generated results in Figure 8. While each row shares the same texture code, the results show that Gstyle produces textures with expressiveness that surpasses realistic appearances.

epochs with the weighting factors λrecon, λseg, λreg, and λCDL from Eq. (10) set to 500, 100, 2, and 500 respectively. Texture refinement

was conducted for 4 epochs with the weighting factors λLPIPS and λGAN from Eq. (11) set to 50 and 1, respectively. For the sub-losses in Eq.(3), λCLIP was fixed at 0.2, while λDINO was set to 0.2 initially and disabled after 500 iterations. The regularization weights λv, λn, and λang in Eq. (8) were set to 1, 50, and 1000, respectively. The 3D pipeline required about 3 hours for training.

### 4.2. Qualitative Results

We evaluated our approach across three fundamental elements of a stylized 3DMM: (1) maintained correspondence, (2) disentangled control, and (3) stylization beyond realistic geometry and texture. Additional results are presented in Figure 18.

4.3. Comparison with Baselines 4.3.1. Evaluation Setting

### 4.2.1. Maintained Correspondence

Baselines. We compared our framework against both a conventional 3DMM, FLAME [LBB∗17], and two stylized face generation methods that produce mesh-based outputs while also preserving dense vertex-level correspondence and supporting disentangled parametric control, making them functionally comparable to 3DMMs: LeGO [YYS∗24] and ClipFace [ATDN22]. LeGO performs shape stylization, while ClipFace focuses on texture stylization built upon a 3DMM geometry. Because LeGO relies on oneshot learning based on 3D supervision from paired meshes of real

We first assessed whether the generated faces preserve dense pointto-point correspondences across different identities and styles. To visualize vertex alignment, we overlaid the same UV texture to several stylized outputs. As illustrated in Figure 5, key facial features consistently occupy the same semantic regions across different stylized faces. This demonstrates that our framework ensures reliable vertex indexing, which is essential for downstream tasks such as animation retargeting, texture editing, and semantic part manipulation.

[Figure 6]

Figure 6: Disentangled parametric control of shape (left), expression (middle), and texture (right), with other factors fixed.

[Figure 7]

Figure 7: The t-SNE projection of randomly sampled faces for each style. Our model extends beyond the realistic representation, distinctly diverging from the representation space of FLAME.

[Figure 8]

Figure 8: Generated Texture from Gsrc (Realistic) and variants of Gstyle. Each row shares the same texture code.

and stylized faces, we applied it only to a subset of styles for which such data are publicly available. Moreover, as it does not perform texture stylization, we generated a single representative texture using SyncMVD [LXLW23] and applied it uniformly across all outputs.

Evaluation Metrics. We evaluated stylized 3DMMs using the same set of 1,000 face meshes randomly sampled by varying parameters per style. Two metrics were computed over generated meshes and corresponding renderings:

• Face Diversity: Measures geometric variation among the sample faces by applying PCA to vertex positions. The cumulative

[Figure 9]

Figure 9: Qualitative comparison of randomly sampled faces generated from the realistic 3DMM and the stylized parametric face models.

Table 2: Quantitative comparison across six different styles. FLAME yields constant values. The values for LeGO and ClipFace are partially reported because LeGO requires manually crafted mesh pairs for each style while ClipFace relies on the FLAME mesh.

Method Face Diversity ↑ Style Score ↑ FLAME [LBB∗17] 12.211 – Style disney pixar child orc baby statue neanderthal disney pixar child orc baby statue neanderthal

LeGO [YYS∗24] 10.498 9.836 9.836 – – – 0.266 0.275 0.333 – – – ClipFace [ATDN22] – – – – – – 0.278 0.284 0.298 0.247 0.285 0.234 Ours 11.678 12.070 11.940 12.004 10.686 11.906 0.285 0.293 0.333 0.263 0.283 0.246

explained variance of the top 100 principal components is used as the metric. A higher value indicates a greater capacity to represent identity-level diversity in facial geometry.

• Style Score: Computes the average CLIP image-text similarity between a rendered image and a style prompt. A higher score reflects a stronger alignment to the input style.

### 4.3.2. Image Stylization

We first compared our stylization method with baselines, whose goal is to generate stylized faces while preserving attributes for future geometric training. The baselines consist of ControlNet [ZRA23], InstructPix2Pix [Li23], PhotoMaker [LCW∗24], and InstantID [WBW∗24]. As presented in Figure 1, ControlNet and InstructPix2Pix performed stylization following the prompt but ControlNet showed misalignment in head rotation and InstructPix2Pix removed the shoulder for the Disney character, both of which are inappropriate for deformation targets. InstantID changed the overall color of the images and failed to perform stylization correctly, following the prompt. Photomaker was unable to generate aligned images. In contrast, EAS generated stylized characters successfully following the prompt while preserving identity, expression, and alignment of the source image.

### 4.3.3. Stylized 3D Morphable Face Model

- Figure 9 presents a visual comparison of the results produced by our method and baselines in three representative styles: Pixar child, Green Orc, and LEGO. While FLAME produced realistic face models capable of representing diverse identities by changing

shape and appearance parameters, the appearances are limited to the realistic human face domain.

Because LeGO enables expressive shape stylization through parameter-based control similar to that of Dstyle of Ours, it effectively captured exaggerated facial structures. However, because it relies on a single manually crafted pair from an artist, it shows limited diversity of the generated meshes. Furthermore, because it could not generate textures for stylized characters, the resulting meshes fail to represent textural variation within a style. ClipFace accepts arbitrary text prompts without 3D data and provides flexible texture stylization over parameter space. While it enables promptdriven appearance control, we observed limitations in stylization quality: in some cases, the texture was overly transformed, reducing identity-level diversity; in others, the stylization was too subtle, appearing as minor artifacts over the original texture. In contrast, our method achieved identity-level face diversity comparable to FLAME, while successfully applying a wide range of text-driven styles. This enables expressive and semantically faithful stylization across a wide range of identities, without compromising structural coherence or parametric controllability.

### 4.3.4. Quantitative Results

To quantitatively evaluate our method as a stylized 3DMM, we assessed its ability to generate diverse shapes under a fixed style condition and its capacity to faithfully reflect the intended style input. As summarized in Table 2, our framework achieved the highest face diversity across most evaluated styles, indicating a stronger capability to represent identity-level shape variations under a shared

[Figure 10]

- Figure 10: Qualitative comparison for ablation study. Each row was are rendered using the same parameters. Close-up views of the eyes and mouth are provided for visualization.

stylization. Furthermore, it attained competitive or superior style scores, validating the fidelity of stylization. These results demonstrate that our model can generate expressive identity variations while maintaining consistent stylization and mesh structure—an essential feature for editable 3D avatar generation.

### 4.4. Ablation Study

We conducted a series of ablation studies to investigate the contribution of each component in our framework, covering both the image stylization stage and the 3D training phase as shown in Figure 10. For the ablation study, we used two different styles, Pixar child and green Orc. We first analyzed the impact of image stylization components. Removing either EAM or latent initialization disrupted source image alignment during i2i stylization, resulting in inconsistencies between mesh geometry and texture, such as mismatched lip texture and mouth shape. Additionally, the lack of consistent style guidance degraded the overall visual quality, producing noisy textures in the eye region without EAM.

Next, we ablated the components of the 3D training pipeline. Removing the LCDL term led to a collapse of identity diversity, causing all shapes to converge to similar forms. This undermines the expressive range originally supported by the 3DMM, as evidenced in the quantitative results presented in Table 3; notably, the diversity value dropped significantly for w/o LCDL. In the progressive three-stage training, skipping the geometry warm-up resulted in diminished geometric deformation—evident in smaller eye shapes and textures appear to compensate for missing geometry. Omitting the texture refinement phase produced textures with reduced stylistic detail and frown-like artifacts. These effects are reflected in the quantitative results reported in Table 3, where both face diversity and style score dropped under ablated settings. In contrast, Ours achieved the highest scores, demonstrating the necessity of each design choice.

We further examined the effect of our adjusted initialization strategy. As described in Section 3.3, instead of sampling from pure noise, we initialized the latent variable with xt to both accelerate in-

Table 3: Ablation study on the effect of each component in terms of face diversity and style fidelity.

Method Face Diversity ↑ Style Score ↑

w/o EAM 11.727 0.3046 w/o init 11.819 0.2972

w/o LCDL 1.812 0.3039 w/o warmup 11.657 0.3027 w/o texture refine - 0.3037

### ours 12.005 0.3136

[Figure 11]

Figure 11: Ablation study on adjusted initialization for EAS.

ference and better preserve the source’s structure and identity. Figure 11 compares the results: the model without using adjusted initialization achieved the strongest stylization but failed to maintain the source identity, and geometry shifted as observed in the neck region although overall facial alignment was enforced by EAM. Conversely, when using a lower value of t, alignment was preserved and inference became faster—because fewer diffusion steps were needed, but stylization fidelity declined sharply. The reason for the decline was that, with fewer steps (i.e., smaller t), EAS could not fully transform the source into the target style.

### 4.5. User Study

We conducted a user study to evaluate the geometric and textural diversity, style fidelity to the provided text, and overall quality of the face models generated by our method compared to LEGO+SyncMVD and CLIPFace. The study employed a 5-point Likert scale to assess three distinct styles—Disney character, Pixar child, and Green Orc—each represented by five sample images. To validate diversity, each sample image included eight randomly sampled face models from each method. A total of 21 participants (13 males, 8 females), with an average age of 28.4 years, took part in the study, providing perceptual ratings across all styles and images for each method. We conducted a one-way ANOVA to analyze statistical significance (p < 0.001). Subsequently, we performed Tukey’s HSD post hoc test, which revealed that all comparisons of our method versus each competitor were statistically significant (p < 0.001). The results, summarized in Figure 12, demonstrate that

[Figure 12]

[Figure 13]

Figure 12: User study results. Ours showed superior results compared to baselines in Diversity, Fidelity, and Naturalness.

[Figure 14]

Figure 14: Two-staged stylization pipeline.

[Figure 15]

Figure 13: Video-driven animation results of StyleMM.

our approach consistently outperforms both baselines in diversity, style fidelity, and perceived quality. These findings indicate that StyleMM offers superior performance relative to both alternative methods.

Figure 15: Stylization results from input images. When an input image is provided, the reconstruction stage generates a coarse face model (middle), and the refinement stage is applied to produce the final stylized 3D model (right).

- 5. Applications

- 5.1. Video-Driven Facial Animation

Driving animation using videos offers a practical way to create and control 3D characters, which is particularly valuable for film, game, and virtual production pipelines. By preserving the animatable structure of realistic 3DMM, StyleMM allows direct application of expression and pose 3DMM parameters obtained from video-based facial tracking systems. To perform video-driven facial animation transfer, we used off-the-shelf FLAME tracking [ZBT22, Yan23] for the optimization of the shape, expression, and pose parameters to match the face in each video frame. The optimized parameters are then fed into the corresponding Dstyle and Gstyle of the stylized 3DMM. As shown in Figure 13, the stylized avatar faithfully reproduces the performer’s dynamics without requiring additional rigging or manual retargeting. This seamless retargeting enables artists and developers to leverage off-the-shelf facial capture tools for real-time or offline generation of high-quality stylized animations, significantly reducing manual workload and accelerating creative workflows.

### 5.2. 3D Face Stylization

While our primary focus is on generating stylized 3DMMs, our method naturally extends to 3D face stylization. Given an input image, we first estimate shape, expression, and texture parameters

using off-the-shelf methods such as MICA [ZBT22] and FFHQUV [BKZ∗23]. These parameters are then fed into our stylized 3DMM networks, Dstyle and Gstyle, to reconstruct geometry and appearance that adhere to both the original identity and the desired target style. This process is presented in the first row of Figure 14. Although this StyleMM reconstruction already produces a stylized 3D face, it may lack the smoothness typically expected in highquality face stylizations. To address these issues, we introduce a subsequent texture refinement stage.

Following the approach of i2i translation studies [WXT∗24, MHS∗21, RBL∗22], our texture refinement process operates directly on the reconstructed UV map. Because the source image has already been stylized by StyleMM, we only inject a small amount of noise and perform a denoising step. We use this denoised image as a target stylized face and update the texture map iteratively. Whereas our EAS initialization adds noise over 76% of the total diffusion steps; this iterative refinement uses just 25% of the diffusion budget and runs for 400 optimization iterations. We employ a 4-step SDXL-Turbo diffusion model [PEL∗23,SLBR24] to ensure rapid convergence. As demonstrated in Figure 15, this refinement produces a smoother, higher-fidelity texture while preserving the stylized geometry obtained from the StyleMM networks.

[Figure 16]

Figure 16: Left: original StyleMM output; Right: eyeball postprocessing results.

### 5.3. Adding Facial Attribute

In our original StyleMM pipeline, we excluded the eyeball geometry during training, following the deformation-network strategy of LeGO [YYS∗24]. To restore facial details omitted from StyleMM’s parametric output, we performed post-processing by adding eyeballs. Using the eye socket center vertex and its distance to the surrounding eyelid vertices, we inserted the original FLAME rigid eyeball meshes and scaled them accordingly. For texture optimization, we utilized EAS stylization to generate references for the L2 loss applied to the rendered outputs. We then added a total variation loss to the texture map to promote spatially uniform optimization across the surface. During this optimization, all other geometry, texture, and network parameters were fixed. As shown in Figure 16, this approach produced realistic eyeballs in the StyleMM outputs. To evaluate the effect of this attribute addition, we conducted a user study with 15 participants, who compared results with and without eyeballs in terms of style fidelity to the textual prompt and overall naturalness. Using Pixar child and green orc models in an A/B test, the eyeball-enhanced versions were strongly preferred, achieving 96.67% in style fidelity and 91.33% in naturalness.

### 6. Conclusion

In this work, we have introduced StyleMM, a novel framework for constructing stylized 3DMMs directly from text-driven image translations. By defining the three core requirements of stylized 3DMMs—maintained correspondence, disentangled control, and stylization beyond realistic geometry and texture—we have positioned stylized 3DMMs as a distinct research paradigm, particularly important for applications in film, animation, and game production. Unlike prior face stylization methods, StyleMM simultaneously satisfies all three criteria by leveraging text-guided i2i translations to fine-tune both a surface deformation network and a texture generator originally designed for realistic 3DMMs.

Our approach makes stylized 3DMMs accessible to users without requiring a stylized 3D dataset by utilizing EAS, a novel image stylization method. Through the three-stage training pipeline and CDL, StyleMM maintains mesh correspondence across diverse identities, ensures independent control over geometry and texture, and achieves a wide spectrum of stylistic variations in both geometry and appearance. Quantitative and qualitative evaluations

[Figure 17]

Figure 17: Self-intersection under exaggerated expressions

demonstrate that StyleMM outperforms existing baselines by preserving identity-level variation while extending beyond realistic models into highly expressive, stylized domains.

### 6.1. Limitations and Future Work

A primary limitation of StyleMM lies in the trade-off between stylization capability and geometric consistency. In EAS, applying extreme stylization can introduce geometry and identity misalignments in the i2i-generated outputs, as shown by the W/O init results in Figure 11. This can result in unexpected mesh deformations. Although EAS mitigates misalignment of facial landmarks and expressions during stylization, residual discrepancies may still impair dense correspondence under highly artistic style transfers. To reduce misalignment for extreme artistic styles, future work could investigate improved alignment strategies that dynamically adjust landmark weighting during stylization. Integrating a confidenceweighted attribute preservation module could more effectively reconcile aggressive geometric deviations with consistent mesh structure.

In 3DMM fine-tuning, strong mesh stabilization losses are applied to maintain plausible geometry, but they can also suppress sharp stylistic details such as pointed ears and fine wrinkles, reducing intended geometric expressiveness. Incorporating multi-scale structural priors—such as Laplacian regularization or mesh spectral embeddings—into the CDL may better preserve both global and local geometry under strong style shifts, enabling more radical stylizations without compromising correspondence. Additionally, integrating displacement mapping could further enhance geometry by disentangling it from texture, although this may necessitate additional data for parameterization.

Another limitation is self-intersection under exaggerated expressions. While our method generally produces meshes of superior quality compared to previous studies due to the adoption of a pretrained surface deformation network, self-intersection can still occur with exaggerated expression as shown in Figure 17. One possible solution might be to incorporate more diverse expressions during training.

Pursuing these directions may help close the existing gaps between unconstrained text-driven stylization and strict geometric coherence, paving the way for stylized 3DMMs to become standard tools in animation, virtual production, and avatar creation not only for large groups of professional artists but also for small studios and game developers.

### Acknowledgements

This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded

by the Korea government(MSIT) (No.RS-2024-00439499, Generating Hyper-Realistic to Extremely-stylized Face Avatar with Varied Speech Speed and Context-based Emotional Expression)

### References

[ALZ∗23] ABDAL R., LEE H.-Y., ZHU P., CHAI M., SIAROHIN A., WONKA P., TULYAKOV S.: 3davatargan: Bridging domains for personalized editable avatars. arXiv preprint arXiv:2301.02700 (2023). 2, 3

[ATDN22] ANEJA S., THIES J., DAI A., NIESSNER M.: Clipface: Text-guided editing of textured 3d morphable models. arXiv preprint arXiv:2212.01406 (2022). 2, 3, 7, 9

[BBBGGD24] BATUHAN BILECEN B., BERKE GOKMEN A., GUZELANT F., DUNDAR A.: Identity preserving 3d head stylization with multiview score distillation. arXiv e-prints (2024), arXiv–2411. 2

[BKZ∗23] BAI H., KANG D., ZHANG H., PAN J., BAO L.: Ffhq-uv: Normalized facial uv-texture dataset for 3d face reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2023), pp. 362–371. 4, 11

[BRP∗18] BOOTH J., ROUSSOS A., PONNIAH A., DUNAWAY D., ZAFEIRIOU S.: Large scale 3d morphable models. International Journal of Computer Vision 126, 2 (2018), 233–254. 3

[BRZ∗16] BOOTH J., ROUSSOS A., ZAFEIRIOU S., PONNIAH A., DUNAWAY D.: A 3d morphable model learnt from 10,000 faces. In Proceedings of the IEEE conference on computer vision and pattern recognition (2016), pp. 5543–5552. 3

[BV99] BLANZ V., VETTER T.: A morphable model for the synthesis of 3d faces. In Proceedings of the 26th annual conference on Computer graphics and interactive techniques (1999), pp. 187–194. 1, 3

[CWZ∗13] CAO C., WENG Y., ZHOU S., TONG Y., ZHOU K.: Facewarehouse: A 3d facial expression database for visual computing. IEEE Transactions on Visualization and Computer Graphics 20, 3 (2013), 413–425. 3

[DOMB23] DARCET T., OQUAB M., MAIRAL J., BOJANOWSKI P.: Vision transformers need registers. arXiv preprint arXiv:2309.16588

(2023). 7

[DYT21] DENG Y., YANG J., TONG X.: Deformed implicit field: Modeling 3d shapes with learned dense correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021), pp. 10286–10296. 3

[EST∗20] EGGER B., SMITH W. A., TEWARI A., WUHRER S., ZOLLHOEFER M., BEELER T., BERNARD F., BOLKART T., KORTYLEWSKI A., ROMDHANI S., ET AL.: 3d morphable face models—past, present, and future. ACM Transactions on Graphics (ToG) 39, 5 (2020), 1–38. 1

[GPM∗22] GAL R., PATASHNIK O., MARON H., BERMANO A. H., CHECHIK G., COHEN-OR D.: Stylegan-nada: Clip-guided domain adaptation of image generators. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–13. 3

[HCH∗23] HAN X., CAO Y., HAN K., ZHU X., DENG J., SONG Y.-Z., XIANG T., WONG K.-Y. K.: Headsculpt: Crafting 3d head avatars with text. arXiv preprint arXiv:2306.03038 (2023). 2

[HSZ∗24] HUANG X., SHAO R., ZHANG Q., ZHANG H., FENG Y., LIU Y., WANG Q.: Humannorm: Learning normal diffusion model for high-quality and realistic 3d human generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024), pp. 4568–4577. 3

[HZRS16] HE K., ZHANG X., REN S., SUN J.: Identity mappings in deep residual networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14 (2016), Springer, pp. 630–645. 7

[JHH25] JU R.-Y., HUANG S.-Y., HUNG Y.-P.: Toonifygb: Styleganbased gaussian blendshapes for 3d stylized head avatars. arXiv preprint arXiv:2505.10072 (2025). 2

[JJK∗22] JUNG Y., JANG W., KIM S., YANG J., TONG X., LEE S.: Deep deformable 3d caricatures with learned shape control. In ACM SIGGRAPH 2022 Conference Proceedings (2022), pp. 1–9. 2, 3

[KC23] KIM G., CHUN S. Y.: Datid-3d: Diversity-preserved domain adaptation using text-to-image diffusion for 3d generative model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023), pp. 14203–14213. 2, 5

[KKLD23] KERBL B., KOPANAS G., LEIMKÜHLER T., DRETTAKIS G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1. 2, 3

[KLA19] KARRAS T., LAINE S., AILA T.: A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 4401–4410. 4

[KLA∗20] KARRAS T., LAINE S., AITTALA M., HELLSTEN J., LEHTINEN J., AILA T.: Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2020), pp. 8110–8119. 4

[KMR∗23] KIRILLOV A., MINTUN E., RAVI N., MAO H., ROLLAND C., GUSTAFSON L., XIAO T., WHITEHEAD S., BERG A. C., LO W.Y., ET AL.: Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision (2023), pp. 4015–4026. 6

[LBB∗17] LI T., BOLKART T., BLACK M. J., LI H., ROMERO J.: Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph. 36, 6 (2017), 194–1. 3, 7, 9

[LBZ∗20] LI R., BLADIN K., ZHAO Y., CHINARA C., INGRAHAM O., XIANG P., REN X., PRASAD P., KISHORE B., XING J., ET AL.: Learning formation of physically-based face attributes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2020), pp. 3410–3419. 3

[LCW∗24] LI Z., CAO M., WANG X., QI Z., CHENG M.-M., SHAN Y.: Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024), pp. 8640–8650. 3, 4, 9

[Li23] LI S.: Instruct-video2avatar: Video-to-avatar generation with instructions. arXiv preprint arXiv:2306.02903 (2023). 9

[Lip21] LIPMAN Y.: Phase transitions, distance functions, and implicit neural representations. arXiv preprint arXiv:2106.07689 (2021). 3

[LWW∗24] LIU H., WANG X., WAN Z., SHEN Y., SONG Y., LIAO J., CHEN Q.: Headartist: Text-conditioned 3d head generation with self score distillation. In ACM SIGGRAPH 2024 Conference Papers (2024), pp. 1–12. 3

[LXLW23] LIU Y., XIE M., LIU H., WONG T.-T.: Text-guided texturing by synchronized multi-view diffusion. arXiv preprint arXiv:2311.12891

(2023). 8

[LYX∗24] LIAO T., YI H., XIU Y., TANG J., HUANG Y., THIES J., BLACK M. J.: Tada! text to animatable digital avatars. In 2024 International Conference on 3D Vision (3DV) (2024), IEEE, pp. 1508–1519. 2

[MHS∗21] MENG C., HE Y., SONG Y., SONG J., WU J., ZHU J.-Y., ERMON S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021). 5, 11

[MON∗19] MESCHEDER L., OECHSLE M., NIEMEYER M., NOWOZIN S., GEIGER A.: Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 4460–4470. 3

[MPJ∗19] MICHALKIEWICZ M., PONTES J. K., JACK D., BAKTASHMOTLAGH M., ERIKSSON A.: Implicit surface representations as layers in neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (2019), pp. 4743–4752. 3

[MST∗21] MILDENHALL B., SRINIVASAN P. P., TANCIK M., BARRON J. T., RAMAMOORTHI R., NG R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65, 1 (2021), 99–106. 2, 3

[ODM∗23] OQUAB M., DARCET T., MOUTAKANNI T., VO H., SZAFRANIEC M., KHALIDOV V., FERNANDEZ P., HAZIZA D., MASSA F., EL-NOUBY A., ET AL.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023). 5, 7

[PEL∗23] PODELL D., ENGLISH Z., LACEY K., BLATTMANN A., DOCKHORN T., MÜLLER J., PENNA J., ROMBACH R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023). 4, 11

[PFS∗19] PARK J. J., FLORENCE P., STRAUB J., NEWCOMBE R., LOVEGROVE S.: Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 165–174. 3

[PKA∗09] PAYSAN P., KNOTHE R., AMBERG B., ROMDHANI S., VETTER T.: A 3d face model for pose and illumination invariant face recognition. In 2009 sixth IEEE international conference on advanced video and signal based surveillance (2009), Ieee, pp. 296–301. 3

[PWS∗21] PATASHNIK O., WU Z., SHECHTMAN E., COHEN-OR D., LISCHINSKI D.: Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF international conference on computer vision (2021), pp. 2085–2094. 3

[PX23] PEEBLES W., XIE S.: Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision (2023), pp. 4195–4205. 4

[RBL∗22] ROMBACH R., BLATTMANN A., LORENZ D., ESSER P., OMMER B.: High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022), pp. 10684–10695. 3, 7, 11

[RKH∗21] RADFORD A., KIM J. W., HALLACY C., RAMESH A., GOH G., AGARWAL S., SASTRY G., ASKELL A., MISHKIN P., CLARK J., ET AL.: Learning transferable visual models from natural language supervision. In International conference on machine learning (2021), PMLR, pp. 8748–8763. 3, 5

[SCL∗24] SONG L., CHEN L., LIU C., LIU P., XU C.: Texttoon: Realtime text toonify head avatar from single video. In SIGGRAPH Asia 2024 Conference Papers (2024), pp. 1–11. 2

[SCS∗22] SAHARIA C., CHAN W., SAXENA S., LI L., WHANG J., DENTON E. L., GHASEMIPOUR K., GONTIJO LOPES R., KARAGOL AYAN B., SALIMANS T., ET AL.: Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems 35 (2022), 36479–36494. 3

[SGY∗21] SHEN T., GAO J., YIN K., LIU M.-Y., FIDLER S.: Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems 34 (2021), 6087–6101. 3

[SLBR24] SAUER A., LORENZ D., BLATTMANN A., ROMBACH R.: Adversarial diffusion distillation. In European Conference on Computer Vision (2024), Springer, pp. 87–103. 11

[SME20] SONG J., MENG C., ERMON S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020). 5

[TJW∗23] TANG L., JIA M., WANG Q., PHOO C. P., HARIHARAN B.: Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems 36 (2023), 1363–1389. 7

[WBW∗24] WANG Q., BAI X., WANG H., QIN Z., CHEN A., LI H., TANG X., HU Y.: Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024). 3, 9

[WMC∗24] WANG D., MENG H., CAI Z., SHAO Z., LIU Q., WANG L., FAN M., ZHAN X., WANG Z.: Headevolver: Text to head avatars via expressive and attribute-preserving mesh deformation. arXiv preprint arXiv:2403.09326 (2024). 2

[WXT∗24] WU Y., XU H., TANG X., CHEN X., TANG S., ZHANG Z., LI C., JIN X.: Portrait3d: Text-guided high-quality 3d portrait generation using pyramid representation and gans prior. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–12. 11

[XOP∗23] XIE J., OUYANG H., PIAO J., LEI C., CHEN Q.: Highfidelity 3d gan inversion by pseudo-multi-view optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023), pp. 321–331. 7

[Yan23] YAN P.: flame-head-tracker. https://github.com/ PeizhiYan/flame-head-tracker, 2023. Accessed: 2025-06-

06. 11

[YKS∗24] YUN K., KIM Y., SEO K., SEO C. W., NOH J.: Representative feature extraction during diffusion process for sketch extraction with one example. arXiv preprint arXiv:2401.04362 (2024). 7

[YSS∗24] YUN K., SEO K., SEO C. W., YOON S., KIM S., JI S., ASHTARI A., NOH J.: Stylized face sketch extraction via generative prior with limited data. In Computer Graphics Forum (2024), vol. 43, Wiley Online Library, p. e15045. 7

[YYS∗24] YOON S., YUN K., SEO K., CHA S., YOO J. E., NOH J.: Lego: Leveraging a surface deformation network for animatable stylized face generation with one example. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024), pp. 4505– 4514. 2, 3, 7, 9, 12

[YZL∗23] YE H., ZHANG J., LIU S., HAN X., YANG W.: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023). 3, 4

[YZZZ24] YANG J., ZENG A., ZHANG R., ZHANG L.: X-pose: Detecting any keypoints. In European Conference on Computer Vision (2024), Springer, pp. 249–268. 4, 5

[ZBT22] ZIELONKA W., BOLKART T., THIES J.: Towards metrical reconstruction of human faces. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XIII (2022), Springer, pp. 250–269. 11

[ZCF∗23] ZHANG C., CHEN Y., FU Y., ZHOU Z., YU G., WANG B., FU B., CHEN T., LIN G., SHEN C.: Styleavatar3d: Leveraging image-text diffusion models for high-fidelity 3d avatar generation. arXiv preprint arXiv:2305.19012 (2023). 3

[ZHH∗23] ZHANG J., HERRMANN C., HUR J., POLANIA CABRERA L., JAMPANI V., SUN D., YANG M.-H.: A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems 36 (2023), 45533– 45547. 7

[ZHXQ24] ZHOU M., HYDER R., XUAN Z., QI G.: Ultravatar: A realistic animatable 3d avatar diffusion model with authenticity guided textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024), pp. 1238–1248. 2, 3

[ZIE∗18] ZHANG R., ISOLA P., EFROS A. A., SHECHTMAN E., WANG O.: The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition (2018), pp. 586–595. 5

[ZMF∗24] ZHOU Z., MA F., FAN H., YANG Z., YANG Y.: Headstudio: Text to animatable head avatars with 3d gaussian splatting. In European Conference on Computer Vision (2024), Springer, pp. 145–163. 2, 3

[ZQL∗23] ZHANG L., QIU Q., LIN H., ZHANG Q., SHI C., YANG W., SHI Y., YANG S., XU L., YU J.: Dreamface: Progressive generation of animatable 3d faces under text guidance. arXiv preprint arXiv:2304.03117 (2023). 2, 3

[ZRA23] ZHANG L., RAO A., AGRAWALA M.: Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision (2023), pp. 3836–3847. 4, 9

[Figure 18]

#### Figure 18: Results of StyleMM, shape and textures are randomly sampled from Dstyle and Gstyle.

