## En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data

# arXiv:2401.01173v1[cs.CV]2Jan2024

Yifang Men1, Biwen Lei1, Yuan Yao1, Miaomiao Cui1, Zhouhui Lian2, Xuansong Xie1

1Institute for Intelligent Computing, Alibaba Group 2Wangxuan Institute of Computer Technology, Peking University https://menyifang.github.io/projects/En3D/index.html

[Figure 1]

Figure 1. Given random noises or guided texts, our generative scheme can synthesize high-fidelity 3D human avatars that are visually realistic and geometrically accurate. These avatars can be seamlessly animated and easily edited. Our model is trained on 2D synthetic data without relying on any pre-existing 3D or 2D collections.

### Abstract

We present En3D, an enhanced generative scheme for sculpting high-quality 3D human avatars. Unlike previous works that rely on scarce 3D datasets or limited 2D collections with imbalanced viewing angles and imprecise pose priors, our approach aims to develop a zero-shot 3D generative scheme capable of producing visually realistic, geometrically accurate and content-wise diverse 3D humans without relying on pre-existing 3D or 2D assets. To address this challenge, we introduce a meticulously crafted workflow that implements accurate physical modeling to learn the enhanced 3D generative model from synthetic 2D data. During inference, we integrate optimization modules to bridge the gap between realistic appearances and coarse 3D shapes. Specifically, En3D comprises three modules: a 3D generator that accurately models generalizable 3D humans with realistic appearance from synthesized balanced,

diverse, and structured human images; a geometry sculptor that enhances shape quality using multi-view normal constraints for intricate human anatomy; and a texturing module that disentangles explicit texture maps with fidelity and editability, leveraging semantical UV partitioning and a differentiable rasterizer. Experimental results show that our approach significantly outperforms prior works in terms of image quality, geometry accuracy and content diversity. We also showcase the applicability of our generated avatars for animation and editing, as well as the scalability of our approach for content-style free adaptation.

### 1. Introduction

3D human avatars play an important role in various applications of AR/VR such as video games, telepresence and virtual try-on. Realistic human modeling is an essential task, and many valuable efforts have been made by lever-

aging neural implicit fields to learn high-quality articulated avatars [9, 11, 45, 52]. However, these methods are directly learned from monocular videos or image sequences, where subjects are single individuals wearing specific garments, thus limiting their scalability.

Generative models learn a shared 3D representation to synthesize clothed humans with varying identities, clothing and poses. Traditional methods are typically trained on 3D datasets, which are limited and expensive to acquire. This data scarcity limits the model’s generalization ability and may lead to overfitting on small datasets. Recently, 3Daware image synthesis methods [6, 20, 39] have demonstrated great potential in learning 3D generative models of rigid objects from 2D image collections. Follow-up works show the feasibility of learning articulated humans from image collections driven by SMPL-based deformations, but only in limited quality and resolution. EVA3D [18] represents humans as a composition of multiple parts with NeRF representations. AG3D [10] incorporates an efficient articulation module to capture both body shape and cloth deformation. Nevertheless, there remains a noticeable gap between generated and real humans in terms of appearance and geometry. Moreover, their results are limited to specific views (i.e., frontal angles) and lack diversity (i.e., fashion images in similar skin tone, body shape, and age).

The aim of this paper is to propose a zero-shot 3D generative scheme that does not rely on any pre-existing 3D or 2D datasets, yet is capable of producing high-quality 3D humans that are visually realistic, geometrically accurate, and content-wise diverse. The generated avatars can be seamlessly animated and easily edited. An illustration is provided in Figure 1. To address this challenging task, our proposed method inherits from 3D-aware human image synthesis and exhibits substantial distinctions based on several key insights. Rethinking the nature of 3D-aware generative methods from 2D collections [6, 10, 18], they actually try to learn a generalizable and deformable 3D representation, whose 2D projections can meet the distribution of human images in corresponding views. Thereby, it is crucial for accurate physical modeling between 3D objects and 2D projections. However, previous works typically leverage pre-existing 2D human images to estimate physical parameters (i.e., camera and body poses), which are inaccurate because of imprecise SMPL priors for highly-articulated humans. This inaccuracy limits the synthesis ability for realistic multi-view renderings. Second, these methods solely rely on discriminating 2D renderings, which is ambiguous and loose to capture inherent 3D shapes in detail, especially for intricate human anatomy.

To address these limitations, we propose a novel generative scheme with two core designs. Firstly, we introduce a meticulously-crafted workflow that implements accurate physical modeling to learn an enhanced 3D gener-

ative model from synthetic data. This is achieved by instantiating a 3D body scene and projecting the underlying 3D skeleton into 2D pose images using explicit camera parameters. These 2D pose images act as conditions to control a 2D diffusion model, synthesizing realistic human images from specific viewpoints. By leveraging synthetic view-balanced, diverse and structured human images, along with known physical parameters, we employ a 3D generator equipped with an enhanced renderer and discriminator to learn realistic appearance modeling. Secondly, we improve the 3D shape quality by leveraging the gap between high-quality multi-view renderings and the coarse mesh produced by the 3D generative module. Specifically, we integrate an optimization module that utilizes multi-view normal constraints to rapidly refine geometry details under supervision. Additionally, we incorporate an explicit texturing module to ensure faithful UV texture maps. In contrast to previous works that rely on inaccurate physical settings and inadequate shape supervision, we rebuild the generative scheme from the ground up, resulting in comprehensive improvements in image quality, geometry accuracy, and content diversity. In summary, our contributions are threefold:

- • We present a zero-shot generative scheme that efficiently synthesizes high-quality 3D human avatars with visual realism, geometric accuracy and content diversity. These avatars can be seamlessly animated and easily edited, offering greater flexibility in their applications.
- • We develop a meticulously-crafted workflow to learn an enhanced generative model from synthesized human images that are balanced, diverse, and also possess known physical parameters. This leads to diverse 3D-aware human image synthesis with realistic appearance.
- • We propose the integration of optimization modules into the 3D generator, leveraging multi-view guidance to enhance both shape quality and texture fidelity, thus achieving realistic 3D human assets.

- 2. Related work
- 3D Human Modeling. Parametric models [4, 21, 22, 30, 40] serve as a common representation for 3D human modeling, they allows for robust control by deforming a template mesh with a series of low-dimensional parameters, but can only generate naked 3D humans. Similar ideas have been extended to model clothed humans [2, 32], but geometric expressivity is restricted due to the fixed mesh topology. Subsequent works [7, 41, 41] further introduce implicit surfaces to produce complex non-linear deformations of 3D bodies. Unfortunately, the aforementioned approaches all require 3D scans of various human poses for model fitting, which are difficult to acquire. With the explosion of NeRF, valuable efforts have been made towards combining NeRF models with explicit human models [9, 11, 29, 45, 52]. Neural body [45] anchors a set of latent codes to the vertices

[Figure 2]

Figure 2. An overview of the proposed scheme, which consists of three modules: 3D generative modeling (3DGM), the geometric sculpting (GS) and the explicit texturing (ET). 3DGM using synthesized diverse, balanced and structured human image with accurate camera φ to learn generalizable 3D humans with the triplane-based architecture. GS is integrated as an optimization module by utilizing multi-view normal constraints to refine and carve geometry details. ET utilizes UV partitioning and a differentiable rasterizer to disentangles explicit UV texture maps. Not only multi-view renderings but also realistic 3D models can be acquired for final results.

of the SMPL model [30] and transforms the spatial locations of the codes to the volume in the observation space. HumanNeRF [52] optimizes for a canonical, volumetric Tpose of the human with a motion field to map the non-rigid transformations. Nevertheless, these methods are learned directly from monocular videos or image sequences, where subjects are single individuals wearing specific garments, thus limiting their scalability.

Generative 3D-aware Image Synthesis. Recently, 3Daware image synthesis methods have lifted image generation with explicit view control by integrating the 2D generative models [23–25] with 3D representations, such as voxels [16, 35, 36, 53], meshes [28, 50] and points clouds [1, 27]. GRAF [49] and π-GAN[5] firstly integrate the implicit representation networks, i.e., NeRF [34], with differentiable volumetric rendering for 3D scene generation. However, they have difficulties in training on highresolution images due to the costly rendering process. Subsequent works have sought to improve the efficiency and quality of such NeRF-based GANs, either by adopting a two-stage rendering process [6, 14, 37, 39, 55] or a smart sampling strategy [8, 60]. StyleSDF [39] combines a SDFbased volume renderer and a 2D StyleGAN network [24] for photorealistic image generation. EG3D [6] introduces a superior triplane representation to leverage 2D CNNbased feature generators for efficient generalization over 3D spaces. Although these methods demonstrate impressive quality in view-consistent image synthesis, they are limited to simplified rigid objects such as faces, cats and cars.

To learn highly articulated humans from unstructured

- 2D images, recent works [10, 12, 18, 19, 56, 58] inte-

grate the deformation field to learn non-rigid deformations based on the body prior of estimated SMPL parameters. EVA3D [18] represents humans as a composition of multiple parts with NeRF representations. Instead of directly rendering the image from a 3D representation, 3DHumanGAN [56] uses an equivariant 2D generator modulated by 3D human body prior, which enables to establish one-tomany mapping from 3D geometry to synthesized textures from 2D images. AG3D [10] combines the 3D generator with an efficient articulation module to warp from canonical space into posed space via a learned continuous deformation field. However, a gap still exists between the generated and real humans in terms of appearance, due to the imprecise priors from complex poses as well as the data biases from limited human poses and imbalanced viewing angles in the dataset.

### 3. Method Description

Our goal is to develop a zero-shot 3D generative scheme that does not rely on any pre-existing 3D or 2D collections, yet is capable of producing high-quality 3D humans that are visually realistic, geometrically accurate and content-wise diverse to generalize to arbitrary humans.

An overview of the proposed scheme is illustrated in Figure 2. We build a sequential pipeline with the following three modules: the 3D generative modeling (3DGM), the geometric sculpting (GS) and the explicit texturing (ET). The first module synthesizes view-balanced, structured and diverse human images with known camera parameters. Subsequently, it learns a 3D generative model from these synthetic data, focusing on realistic appearance mod-

eling (Section 3.1). To overcome the inaccuracy of the 3D shape, the GS module is incorporated during the inference process. It optimizes a hybrid representation with multiview normal constraints to carve intricate mesh details (Section 3.2). Additionally, the ET module is employed to disentangle explicit texture by utilizing semantical UV partitioning and a differentiable rasterizer (Section 3.3). By combining these modules, we are able to synthesize highquality and faithful 3D human avatars by incorporating random noises or guided texts/images (Section 3.4).

#### 3.1. 3D generative modeling

Without any 3D or 2D collections, we develop a synthesisbased flow to learn a 3D generative module from 2D synthetic data. We start by instantiating a 3D scene through the projection of underlying 3D skeletons onto 2D pose images, utilizing accurate physical parameters (i.e., camera parameters). Subsequently, the projected 2D pose images serve as conditions to control the 2D diffusion model [59] for synthesizing view-balanced, diverse, and lifelike human images. Finally, we employ a triplane-based generator with enhanced designs to learn a generalizable 3D representation from the synthetic data. Details are described as follows.

- 3D instantiation. Starting with a template body mesh (e.g., SMPL-X [44]) positioned and posed in canonical space, we estimate the 3D joint locations P3d by regressing them from interpolated vertices. We then project P3d onto 2D poses Pi,i = 1,...,k from K horizontally uniformly sampled viewpoints φ. In this way, paired 2D pose images and their corresponding camera parameters {Pi,φi} are formulated.

Controlled 2D image synthesis. With the pose image Pi, we feed it into off-the-shelf ControlNet [59] as the pose condition to guide diffusion models [47] to synthesize human images in desired poses (i.e., views). The text prompt T is also used for diverse contents. Given a prompt T, instead of generating a human image Is : Is = C(Pi,T) independently for each view φi, we horizontally concatenate K pose images Pi ∈ RH×W×3, resulting in Pi′ ∈ RH×KW×3 and feed Pi′ to C, along with a prompt hint of ‘multi-view’ in T. In this way, multi-view human images Is′ are synthesized with roughly coherent appearance. We split Is′ to single view images Iφ under specific views φ. This concatenation strategy facilitates the convergence of distributions in synthetic multi-views, thus easing the learning of common 3D representation meeting multi-view characteristics.

Generalizable 3D representation learning. With synthetic data of paired {Iφ,φ}, we learn the 3D generative module G3d from them to produce diverse 3D-aware human images with realistic appearance. Inspired by EG3D [6], we employ a triplane-based generator to produce a generalizable representation T and introduce a patch-composed neural renderer to learn intricate human representation efficiently. Specifically, instead of uniformly sampling 2D

pixels on the image I, we decompose patches in the ROI region including human bodies, and only emit rays towards pixels in these patches. The rays are rendered into RGB color with opacity values via volume rendering. Based on the decomposed rule, we decode rendered colors to multiple patches and re-combine these patches for full feature images. In this way, the representation is composed of effective human body parts, which directs the attention of the networks towards the human subject itself. This design facilitates fine-grained local human learning while maintaining computational efficiency.

For the training process, we employ two discriminators, one for RGB images and another for silhouettes, which yields better disentanglement of foreground objects with global geometry. The training loss for this module L3d consists of the two adversarial terms:

L3d = Ladv(Drgb,G3d) + λsLadv(Dmask,G3d), (1)

where λs denotes the weight of silhouette item. Ladv is computed by the non-saturating GAN loss with R1 regularization [33].

With the trained G3d, we can synthesize 3D-aware human images Igφ with view control, and extract coarse 3D shapes Mc from the density field of neural renderer using the Marching Cubes algorithm [31].

#### 3.2. Geometric sculpting

Our 3D generative module can produce high-quality and 3D-consistent human images in view controls. However, its training solely relies on discriminations made using 2D renderings, which can result in inaccuracies in capturing the inherent geometry, especially for complex human bodies. Therefore, we integrate the geometric sculpting, an optimization module leveraging geometric information from high-quality multi-views to carve surface details. Combined with a hybrid 3D representation and a differentiable rasterizer, it can rapidly enhance the shape quality within seconds.

DMTET adaption. Owing to the expressive ability of arbitrary topologies and computational efficiency with direct shape optimization, we employ DMTET as our 3D representation in this module and adapt it to the coarse mesh Mc via an initial fitting procedure. Specifically, we parameterize DMTET as an MLP network Ψg that learns to predict the SDF value s(vi) and the position offset δvi for each vertex vi ∈ V T of the tetrahedral grid (V T,T). A point set P = {pi ∈ R3} is randomly sampled near Mc and their SDF values SDF(pi) can be pre-computed. We adapt the parameters ψ of Ψg by fitting it to the SDF of Mc:

Lada =

||s(pi;ψ) − SDF(pi)||2. (2)

pi∈P

Geometry refinement. Using the adapted DMTET, we leverage the highly-detailed normal maps N derived from realistic multi-view images as a guidance to refine local surfaces. To obtain the pseudo-GT normals Nφ, we extract them from Igφ using a pre-trained normal estimator [54]. For the rendered normals Nˆφ, we extract the triangular mesh Mtri from (V T,T) using the Marching Tetrahedra (MT) layer in our current DMTET. By rendering the generated mesh Mtri with differentiable rasterization, we obtain the resulting normal map Nˆφ. To ensure holistic surface polishing that takes into account multi-view normals, we randomly sample camera poses φ that are uniformly distributed in space. We optimize the parameters of Ψg using the normal loss, which is defined as:

Lnorm = ||Nˆφ − Nφ||2. (3)

After rapid optimization, the final triangular mesh Mtri can be easily extracted from the MT layer. If the hands exhibit noise, they can be optionally replaced with cleaner geometry hands from SMPL-X, benefiting from the alignment of the generated body in canonical space with the underlying template body.

#### 3.3. Explicit texturing

With the final mesh, the explicit texturing module aims to disentangle a UV texture map from multi-view renderings Igφ. This intuitive module not only facilitates the incorporation of high-fidelity textures but also enables various editing applications, as verified in Section 4.4.

Given the polished triangular mesh Mtri and multiviews Igφ, we model the explicit texture map Tuv of Mtri with a semantic UV partition and optimize Tuv using a differentiable rasterizer R [26]. Specifically, leveraging the canonical properties of synthesized bodies, we semantically split Mtri into γ components and rotate each component vertically, thus enabling effective UV projection for each component with cylinder unwarping. We then combine the texture partitions together for the full texture Tuv. We optimize Tuv from a randomly initialized scratch using the texture loss, which consists of a multi-view reconstruction term and a total-variation (tv) term:

Ltex = Lrec + λtvLtv, (4) where λtv denotes the weight of the tv loss.

Multi-view guidance. To ensure comprehensive texturing in the 3D space, we render the color images R(Mtri,φ) and silhouettes S using R and optimize Tuv utilizing multiview weighted guidance. Their pixel-alignment distances to the original multi-view renderings Igφ are defined as the reconstruction loss:

Lrec =

wφ||R(Mtri,φ) · S − Igφ · S||2, (5)

φ∈Ω

[Figure 3]

Figure 3. The visualized flowchart of our method that synthesize textured 3D human avatars from input noises, texts or images.

where Ω is the set of viewpoints {φi,i = 1,...,k} and wφ denotes weights of different views. wφ equals to 1.0 for φ ∈ {front,back} and 0.2 otherwise.

Smooth constraint. To avoid abrupt variations and smooth the generated texture Tuv, we utilize the total-variation loss Ltv which is computed by:

1

h × w × c||∇x(Tuv) + ∇y(Tuv)||, (6) where x and y denote horizontal and vertical directions.

Ltv =

#### 3.4. Inference

Built upon the above modules, we can generate high-quality 3D human avatars from either random noises or guided inputs such as texts or images. The flowchart for this process is shown in Figure 3. For input noises, we can easily obtain the final results by sequentially using the 3DGM, GS and ET modules. For text-guided synthesis, we first convert the text into a structured image using our controlled diffusion C, and then inverse it to the latent space using PTI [46]. Specially, the GS and ET modules provide an interface that accurately reflects viewed modifications in the final 3D objects. As a result, we utilize the guided image to replace the corresponding view image, which results in improved fidelity in terms of geometry and texture. The same process is applied for input images as guided images.

### 4. Experimental Results

Implementation details. Our process begins by training the 3D generative module (3DGM) on synthetic data. During inference, we integrate the geometric sculpting (GS) and explicit texturing (ET) as optimization modules. For 3DGM, we normalize the template body to the (0,1) space and place its center at the origin of the world coordinate system. We sample 7(K = 7) viewpoints uniformly from the horizontal plane, ranging from 0◦ to 180◦ (front to back), with a camera radius of 2.7. For each viewpoint, we generate 100K images using the corresponding pose image. To ensure diverse synthesis, we use detailed descriptions of

[Figure 4]

Figure 4. Results of synthesized 3D human avatars at 5122.

age, gender, ethnicity, hairstyle, facial features, and clothing, leveraging a vast word bank. To cover 360◦ views, we horizontally flip the synthesized images and obtain 1.4 million human images at a resolution of 5122 in total. We train the 3DGM for about 2.5M iterations with a batch size of 32, using two discriminators with a learning rate of 0.002 and a generator learning rate of 0.0025. The training takes 8 days on 8 NVIDIA Tesla-V100. For GS, we optimize ψ for 400 iterations for DMTET adaption and 100 iterations for surface carving (taking about 15s in total on 1 NVIDIA RTX 3090 GPU). For ET, we set λuv = 1 and optimize Tuv for 500 iterations (around 10 seconds). We split Mtri into 5(γ = 5) body parts (i.e., trunk, left/right arm/leg) with cylinder UV unwarping. We use the Adam optimizer with learning rates of 0.01 and 0.001 for Ψg and Tuv, respectively. Detailed network architectures can be found in the supplemental materials (Suppl).

#### 4.1. 3D human generation

Figure 4 showcases several 3D human avatars synthesized by our pipeline, highlighting the image quality, geometry

accuracy, and diverse outputs achieved through our method. Additionally, we explore the interpolation of the latent conditions to yield smooth transitions in appearance, leveraging the smooth latent space learned by our generative model. For more synthesized examples and interpolation results, please refer to the Suppl.

#### 4.2. Comparisons

Qualitative comparison. In Figure 5, we compare our method with three baselines: EVA3D [18] and AG3D [10], which are state-of-the-art methods for generating 3D humans from 2D images, and EG3D [6], which serves as the foundational backbone of our method. The results of first two methods are produced by directly using source codes and trained models released by authors. We train EG3D using our synthetic images with estimated cameras from scratch. As we can see, EVA3D fails to produce 360◦ humans with reasonable back inferring. AG3D and EG3D are able to generate 360◦ renderings but both struggle with photorealism and capturing detailed shapes. Our method synthesizes not only higher-quality, view-consistent 360◦ im-

[Figure 5]

Figure 5. Qualitative comparison with three state-of-the-art methods: EVA3D [18], AG3D [10] and EG3D [6].

Table 1. Quantitative evaluation using FID, IS-360, normal accuracy (Normal) and identity consistency (ID).

Method FID ↓ IS-360 ↑ Normal ↓ ID↑ EVA3D [18] 15.91 3.19 30.81 0.72 AG3D [10] 10.93 3.28 20.83 0.69 EG3D [6] 7.48 3.26 12.74 0.71 Ours 2.73 3.43 5.62 0.74

ages but also higher-fidelity 3D geometry with intricate details, such as irregular dresses and haircuts.

Quantitative comparison. Table 1 provides quantitative results comparing our method against the baselines. We measure image quality with Frechet Inception Distance (FID) [17] and Inception Score [48] for 360◦ views (IS360). FID measures the visual similarity and distribution discrepancy between 50k generated images and all real images. IS-360 focuses on the self-realism of generated images in 360◦ views. For shape evaluation, we compute FID between rendered normals and pseudo-GT normal maps (Normal), following AG3D. The FID and Normal scores of EVA3D and AG3D are directly fetched from their reports. Additionally, we access the multi-view facial identity consistency using the ID metric introduced by EG3D.

Table 2. Results of models trained by replacing physical parameters with estimated ones (w/o SYN-P) or removing patchcomposed rendering (w/o PCR).

Ours Ours-w/o SYN-P Ours-w/o PCR FID ↓ 2.73 4.28 3.26

IS-360 ↑ 3.43 3.31 3.35

Our method demonstrates significant improvements in FID and Normal, bringing the generative human model to a new level of realistic 360◦ renderings with delicate geometry while also maintaining state-of-the-art view consistency.

#### 4.3. Ablation study

Synthesis flow and patch-composed rendering. We assess the impact of our carefully designed synthesis flow by training a model with synthetic images but with camera and pose parameters estimated by SMPLify-X [44] (w/o SYNP). As Table 2 shows, the model w/o SYN-P results in worse FID and IS-360 scores, indicating that the synthesis flow contributes to more accurate physical parameters for realistic appearance modeling. By utilizing patch-composed rendering (PCR), the networks focus more on the human region, leading to more realistic results.

[Figure 6]

Figure 6. Effects of the GS module to carve fine-grained surfaces.

[Figure 7]

Figure 7. Effects of the ET module for guided synthesis.

Geometry sculpting module (GS). We demonstrate the importance of this module by visualizing the meshes before and after its implementation. Figure 6 (b) shows that the preceding module yields a coarse mesh due to the complex human anatomy and the challenges posed by decomposing ambiguous 3D shapes from 2D images. The GS module utilizes high-quality multi-view outputs and employs a more flexible hybrid representation to create expressive humans with arbitrary topologies. It learns from pixel-level surface supervision, leading to a significant improvement in shape quality, characterized by smooth surfaces and intricate outfits (Figure 6 (c)).

Explicit texturing module (ET). This intuitive module not only extracts the explicit UV texture for complete 3D assets but also enables high-fidelity results for image guided synthesis. Following the flowchart in Figure 3, we compare the results produced with and without this module. Our method without ET directly generates implicit renderings through PTI inversion, as shown in Figure 7 (b). While it successfully preserves global identity, it struggles to synthesize highly faithful local textures (e.g., floral patterns). The ET module offers a convenient and efficient way to directly interact with the 3D representation, enabling the production of high-fidelity 3D humans with more consistent content including exquisite local patterns (Figure 7 (a, c)).

#### 4.4. Applications

Avatar animation. All avatars produced by our method are in a canonical body pose and aligned to an underlying 3D skeleton extracted from SMPL-X. This alignment allows for easy animation and the generation of motion videos, as demonstrated in Figure 1 and Suppl.

[Figure 8]

Figure 8. Results synthesized by adapting our method to various styles (e.g., Disney cartoon characters) or contents (e.g., portrait heads).

Texture doodle and local editing. Our approach benefits from explicitly disentangled geometry and texture, enabling flexible editing capabilities. Following the flowchart of text or image guided synthesis (Section 3.4), users can paint any pattern or add text to a guided image. These modifications can be transferred to 3D human models by inputting modified views into the texture module (e.g., painting the text ’hey’ on a jacket as shown in Figure 1 (d)). Our approach also allows for clothing editing by simultaneously injecting edited guide images with desired clothing into the GS and ET modules (e.g., changing a jacket and jeans to bodysuits in Figure 1 (e)). More results can be found in Suppl.

Content-style free adaption. Our proposed scheme is versatile and can be extended to generate various types of contents (e.g., portrait heads ) and styles (e.g., Disney cartoon characters). To achieve this, we fine-tune our model using synthetic images from these domains, allowing for flexible adaptation. We showcase the results in Figure 8. More results and other discussions (e.g., limitations, negative impact, etc.) can be found in Suppl.

### 5. Conclusions

We introduced En3D, a novel generative scheme for sculpting 3D humans from 2D synthetic data. This method overcomes limitations in existing 3D or 2D collections and significantly enhances the image quality, geometry accuracy, and content diversity of generative 3D humans. En3D comprises a 3D generative module that learns generalizable 3D humans from synthetic 2D data with accurate physical modeling, and two optimization modules to carve intricate shape details and disentangle explicit UV textures with high fidelity, respectively. Experimental results validated the superiority and effectiveness of our method. We also demonstated the flexibility of our generated avatars for animation and editing, as well as the scalability of our approach for synthesizing portraits and Disney characters. We believe that our solution could provide invaluable human assets for

the 3D vision community. Furthermore, it holds potential for use in common 3D object synthesis tasks.

### Acknowledgements

We would like to thank Mengyang Feng and Jinlin Liu for their technical support on guided 2D image synthesis.

### References

- [1] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3d point clouds. In International conference on machine learning, pages 40–49. PMLR, 2018. 3
- [2] Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian Theobalt, and Gerard Pons-Moll. Video based reconstruction of 3d people models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8387– 8397, 2018. 2
- [3] Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit Y. Ogras, and Linjie Luo. Panohead: Geometry-aware 3d fullhead synthesis in 360deg. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20950–20959, 2023.
- [4] Dragomir Anguelov, Praveen Srinivasan, Daphne Koller, Sebastian Thrun, Jim Rodgers, and James Davis. Scape: shape completion and animation of people. In ACM SIGGRAPH 2005 Papers, pages 408–416. 2005. 2
- [5] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5799–5809, 2021. 3
- [6] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123–16133, 2022. 2, 3, 4, 6, 7
- [7] Xu Chen, Tianjian Jiang, Jie Song, Jinlong Yang, Michael J Black, Andreas Geiger, and Otmar Hilliges. gdna: Towards generative detailed neural avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20427–20437, 2022. 2
- [8] Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong. Gram: Generative radiance manifolds for 3d-aware image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10673– 10683, 2022. 3
- [9] Zijian Dong, Chen Guo, Jie Song, Xu Chen, Andreas Geiger, and Otmar Hilliges. Pina: Learning a personalized implicit neural avatar from a single rgb-d video sequence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20470–20480, 2022. 2
- [10] Zijian Dong, Xu Chen, Jinlong Yang, Michael J Black, Otmar Hilliges, and Andreas Geiger. Ag3d: Learning to gen-

- erate 3d avatars from 2d image collections. arXiv preprint arXiv:2305.02312, 2023. 2, 3, 6, 7
- [11] Yao Feng, Jinlong Yang, Marc Pollefeys, Michael J Black, and Timo Bolkart. Capturing and animation of body and clothing from monocular video. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9, 2022. 2
- [12] Jianglin Fu, Shikai Li, Yuming Jiang, Kwan-Yee Lin, Chen Qian, Chen Change Loy, Wayne Wu, and Ziwei Liu. Stylegan-human: A data-centric odyssey of human generation. In European Conference on Computer Vision, pages 1–19. Springer, 2022. 3
- [13] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [14] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. Stylenerf: A style-based 3d-aware generator for high-resolution image synthesis. arXiv preprint arXiv:2110.08985, 2021. 3
- [15] Honglin He, Zhuoqian Yang, Shikai Li, Bo Dai, and Wayne Wu. Orthoplanes: A novel representation for better 3dawareness of gans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22996–23007, 2023.
- [16] Philipp Henzler, Niloy J Mitra, and Tobias Ritschel. Escaping plato’s cave: 3d shape from adversarial rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9984–9993, 2019. 3
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [18] Fangzhou Hong, Zhaoxi Chen, Yushi Lan, Liang Pan, and Ziwei Liu. Eva3d: Compositional 3d human generation from 2d image collections. arXiv preprint arXiv:2210.04888,

2022. 2, 3, 6, 7

- [19] Suyi Jiang, Haoran Jiang, Ziyu Wang, Haimin Luo, Wenzheng Chen, and Lan Xu. Humangen: Generating human radiance fields with explicit priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12543–12554, 2023. 3
- [20] Kyungmin Jo, Wonjoon Jin, Jaegul Choo, Hyunjoon Lee, and Sunghyun Cho. 3d-aware generative model for improved side-view image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22862– 22872, 2023. 2
- [21] Hanbyul Joo, Tomas Simon, and Yaser Sheikh. Total capture: A 3d deformation model for tracking faces, hands, and bodies. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8320–8329, 2018. 2
- [22] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7122–7131, 2018. 2
- [23] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks.

- In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 3
- [24] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 3
- [25] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34:852–863, 2021. 3
- [26] Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics (TOG), 39(6):1–14, 2020. 5
- [27] Ruihui Li, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng-Ann Heng. Pu-gan: a point cloud upsampling adversarial network. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7203–7212,

2019. 3

- [28] Yiyi Liao, Katja Schwarz, Lars Mescheder, and Andreas Geiger. Towards unsupervised learning of generative models for 3d controllable image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5871–5880, 2020. 3
- [29] Lingjie Liu, Marc Habermann, Viktor Rudnev, Kripasindhu Sarkar, Jiatao Gu, and Christian Theobalt. Neural actor: Neural free-view synthesis of human actors with pose control. ACM transactions on graphics (TOG), 40(6):1–16,

2021. 2

- [30] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. ACM Transactions on Graphics, 34(6),

2015. 2, 3

- [31] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field, pages 347–353. 1998. 4
- [32] Qianli Ma, Jinlong Yang, Anurag Ranjan, Sergi Pujades, Gerard Pons-Moll, Siyu Tang, and Michael J Black. Learning to dress 3d people in generative clothing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6469–6478, 2020. 2
- [33] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International conference on machine learning, pages 3481–

3490. PMLR, 2018. 4

- [34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3
- [35] Thu Nguyen-Phuoc, Chuan Li, Lucas Theis, Christian Richardt, and Yong-Liang Yang. Hologan: Unsupervised learning of 3d representations from natural images. In Proceedings of the IEEE/CVF International Conference on

Computer Vision, pages 7588–7597, 2019. 3

- [36] Thu H Nguyen-Phuoc, Christian Richardt, Long Mai, Yongliang Yang, and Niloy Mitra. Blockgan: Learning 3d object-aware scene representations from unlabelled images. Advances in neural information processing systems, 33:6767–6778, 2020. 3
- [37] Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11453–11464, 2021. 3
- [38] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3504–3515, 2020.
- [39] Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira Kemelmacher-Shlizerman. Stylesdf: High-resolution 3d-consistent image and geometry generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13503– 13513, 2022. 2, 3
- [40] Ahmed AA Osman, Timo Bolkart, and Michael J Black. Star: Sparse trained articulated human body regressor. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VI 16, pages 598–613. Springer, 2020. 2
- [41] Pablo Palafox, Aljaˇz Boˇziˇc, Justus Thies, Matthias Nießner, and Angela Dai. Npms: Neural parametric models for 3d deformable shapes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12695–12705,

2021. 2

- [42] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 165–174, 2019.
- [43] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2337–2346, 2019.
- [44] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975–10985, 2019. 4, 7
- [45] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9054–9063, 2021. 2
- [46] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on graphics (TOG), 42(1):1–13,

2022. 5

- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 4
- [48] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 7
- [49] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. Graf: Generative radiance fields for 3d-aware image synthesis. Advances in Neural Information Processing Systems, 33:20154–20166, 2020. 3
- [50] Attila Szab´o, Givi Meishvili, and Paolo Favaro. Unsupervised generative 3d shape learning from natural images. arXiv preprint arXiv:1910.00287, 2019. 3
- [51] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021.
- [52] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, pages 16210–16220, 2022. 2, 3
- [53] Jiajun Wu, Chengkai Zhang, Tianfan Xue, Bill Freeman, and Josh Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in neural information processing systems, 29, 2016. 3
- [54] Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J Black. Icon: Implicit clothed humans obtained from normals. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13286–13296. IEEE, 2022. 5
- [55] Yang Xue, Yuheng Li, Krishna Kumar Singh, and Yong Jae Lee. Giraffe hd: A high-resolution 3d-aware generative model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18440– 18449, 2022. 3
- [56] Zhuoqian Yang, Shikai Li, Wayne Wu, and Bo Dai. 3dhumangan: 3d-aware human image generation with 3d pose mapping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23008–23019, 2023. 3
- [57] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems, 34:4805–4815, 2021.
- [58] Jianfeng Zhang, Zihang Jiang, Dingdong Yang, Hongyi Xu, Yichun Shi, Guoxian Song, Zhongcong Xu, Xinchao Wang, and Jiashi Feng. Avatargen: a 3d generative model for animatable human avatars. In European Conference on Computer Vision, pages 668–685. Springer, 2022. 3
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 4

- [60] Peng Zhou, Lingxi Xie, Bingbing Ni, and Qi Tian. Cips-3d: A 3d-aware generator of gans based on conditionally-independent pixel synthesis. arXiv preprint arXiv:2110.09788, 2021. 3
- [61] Peihao Zhu, Rameen Abdal, Yipeng Qin, and Peter Wonka. Sean: Image synthesis with semantic region-adaptive normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5104– 5113, 2020.

