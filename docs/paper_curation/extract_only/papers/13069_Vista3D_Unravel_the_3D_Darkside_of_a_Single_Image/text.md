## Vista3D: Unravel the 3D Darkside of a Single Image

# arXiv:2409.12193v1[cs.CV]18Sep2024

###### Qiuhong Shen1, Xingyi Yang1, Michael Bi Mi2, and Xinchao Wang1⋆

1 National University of Singapore 2 Huawei Technologies Ltd {qiuhong.shen,xyang}@u.nus.edu xinchao@nus.edu.sg

Abstract. We embark on the age-old quest: unveiling the hidden dimensions of objects from mere glimpses of their visible parts. To address this, we present Vista3D, a framework that realizes swift and consistent 3D generation within a mere 5 minutes. At the heart of Vista3D lies a two-phase approach: the coarse phase and the fine phase. In the coarse phase, we rapidly generate initial geometry with Gaussian Splatting from a single image. In the fine phase, we extract a Signed Distance Function (SDF) directly from learned Gaussian Splatting, optimizing it with a differentiable isosurface representation. Furthermore, it elevates the quality of generation by using a disentangled representation with two independent implicit functions to capture both visible and obscured aspects of objects. Additionally, it harmonizes gradients from 2D diffusion prior with 3D-aware diffusion priors by angular diffusion prior composition. Through extensive evaluation, we demonstrate that Vista3D effectively sustains a balance between the consistency and diversity of the generated 3D objects. Demos and code will be available at https://github.com/florinshen/Vista3D.

Keywords: 3D Generation · 3D Reconstruction · Score Distillation

##### 1 Introduction

Since the earliest times, our ancestors gazed upon the luminous moon, a symbol of mystery and wonder. Its bright facade, an elegant sphere in the cosmos, has always made us think about what remains hidden: the moon’s obscure and elusive dark side. This curiosity, as ancient as human history itself, represents our innate desire to uncover the concealed dimensions that exist beyond the visible.

This quest, once purely philosophical, has now ventured into the realm of practicality, propelled by the advancements in 3D generative model [29,34,42,45, 48]. These technologies enable a broad range of applications, especially in gaming and virtual reality, allowing for the creation of rich, detailed environments and objects without extensive modeling.

Nevertheless, the development of robust large-scale 3D generative models remains a formidable challenge, predominantly due to the limited availability of 3D data. Numerous attempts [1,13,27] have been made to train 3D diffusion models on relatively small 3D datasets, condition on textual or visual prompts; Yet,

⋆ Corresponding Author.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

A grey BMW X5

A grey Volvo XC90

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A white skin horse with flowing mane A sleek silver horse with flowing mane

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

A pale cream-color bear pillow

[Figure 23]

[Figure 24]

A pale cream-color panda pillow

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Reference

A muscle dog with capital S A muscle dog wearing a superman cape

- Fig. 1: 3D Darkside of Single Image. By employing various text prompts, Vista3D is capable of unveiling the diversity of unseen views while retaining 3D consistency and detail. Two novel views and the normal map are visualized for each text prompt.

these endeavors often fall short in creating 3D objects with structural integrity and textural consistency.

This challenge is further compounded in the context of reconstructing 3D objects from single images. In this context, two primary approaches emerge. The first considers the task as a problem of sparse-view reconstruction. However, this often leads to blurred 3D outputs due to the neglect of unseen elements, resulting in excessively blurred 3D objects [8,52] as most views remain unseen.

On the other hand, the generative approach, which leverages large-scale 2D diffusion models [29,42], introduces its own set of challenges. Efforts to develop 3D-aware 2D diffusion models [19,21,30,32,34,39,40,51] involve fine-tuning 2D models with camera transformation modeling on 3D datasets [5,6]. Nevertheless, the prevalence of synthetic objects in these datasets can lead to a compromise in 2D diversity. This often results in the generation of oversimplified geometries and textures.

In this paper, we present Vista3D, a framework designed for reconstructing the unseen view (or "darkside") from a single image. Central to Vista3D is a dual-phase strategy: a coarse phase followed by a fine phase.

In the coarse phase, we leverage 3D Gaussian splatting [14] to swiftly create basic geometry and textures. To stabilize Gaussian Splatting optimization, we employ a gradient-based Top-K densification strategy, focusing on Gaussian points with the highest gradients. Additionally, we introduce two novel regularization terms targeting the Gaussian scale and transmittance values, significantly enhancing the convergence speed.

The fine phase then transforms this initial geometry into signed distance fields (SDF) for further optimization. Here, we employ FlexiCubes [38], an advanced differentiable isosurface technique, to refine the geometry. This refine-

ment aids in learning the signed distance fields (SDFs), deformation, and interpolation weights. The parameters are optimized by ensuring fidelity to the original image and guided by a score function derived from diffusion priors.

Despite these advancements, a unified representation and supervision across all views, both seen and unseen, prove insufficient for capturing the unique characteristics of different viewpoints and generating diverse, consistent 3D objects. To address this, we enhance the representation by implementing Disentangled Texture Representation, using two angularly disentangled networks for accurate texture prediction. Furthermore, our Angular-based Composition method amalgamates different diffusion priors, adjusting their gradients within specific angular bounds according to their gradient magnitudes. This strategic adjustment assures 3D consistency while promoting diversity in the unseen views.

Vista3D excels in efficiently generating diverse and consistent 3D objects from a single image within five minutes. Our extensive evaluations demonstrate its ability to maintain a flexible balance between the consistency and diversity of the generated 3D objects.

We summarize our contribution as follows:

- – We present Vista3D, a framework for revealing the 3D darkside of single images, efficiently generating diverse 3D objects using 2D priors.
- – We develop a transition from Gaussian Splatting to isosurface 3D representations, refining coarse geometry with a differentiable isosurface method and disentangled texture for textured mesh creation.
- – We propose an angular composition approach for diffusion priors, constraining their gradient magnitudes to achieve diversity on the 3D darkside without sacrificing 3D consistency.

##### 2 Related-works

- 2.1 3D Generation Conditioned on a Single Image

The objective of image-to-3D generation is to create 3D objects from a single reference image. Initial methods [8,52] approached this challenge as a variant of sparse view 3D reconstruction. However, these methods often resulted in blurred object outputs due to insufficient priors. Recently, drawing inspiration from textto-3D initiatives that utilize Score Distillation Sampling (SDS) to elevate 2D diffusion priors into 3D generative models, image-to-3D works [24,33,34,40,42] have adopted a similar approach for 3D object generation based on a single image. However, 2D diffusion priors alone cannot ensure 3D consistency, as they are typically trained solely on image datasets. To address this, several studies [19–21,39] have attempted to refine 2D diffusion priors with 3D data [5,6], enhancing their ability to model 3D consistency. A notable example is Zero-1-to-

###### 3, which can generate novel views condition on single image and camera position. Integrating this refined model with SDS [30,41] allows for the reconstruction of coherent 3D objects. Moreover, another stream of works [9,17,36,46,47,50,55] pretrained on large-scale 3D dataset [5] directly predicting the representation

of a 3D object from a single image. Diverging from previous works, our work does not solely view this as a 3D reconstruction issue. We redefine it as a 3D generation task aimed at uncovering the unseen 3D aspects behind a single image. Through a meticulously crafted framework, our method efficiently generate diverse and consistent 3D objects.

###### 2.2 3D Representations for Generation

Presently, most zero-shot text-to-3D and image-to-3D models utilize an optimization based pipeline, parameterizing the 3D object as a differentiable representation, which varies among different methods. The most prevalent representation in groundbreaking works like dreamfields [12], dreamfusion [29], and SJC [43] is Neural Radiance Fields (NeRF) [25]. However, training a NeRF is computationally intensive and takes long time to convergence. Magic3D [16] introduced a two-stage representation, initially learning a coarse NeRF, followed by refining the polygon mesh using a differentiable isosurface method, DMTet [37]. Fantasia3D [2] suggested directly optimizing DMTet [37] in separate phases for geometry and texture, but this often leads to mode collapse in the geometry phase and extends training time beyond NeRF. Gaussian Splatting [10,14,35,44,53] has gained attention for its efficiency in various 3D tasks, with several 3D generative models [3,4,41,49] incorporating it for effective generation. However, as a point-based representation, it cannot yield high-fidelity meshes. In our approach, we employ Gaussian Splatting exclusively to create coarse geometry. This coarse geometry is then transformed into SDF, optimized with a hybrid isosurface representation, FlexiCubes [38], to produce high-fidelity meshes. Additionally, we propose an angular disentangled texture representation, tailored to the specifics of this task.

##### 3 Methodology

In this section, we outline our framework to generate detailed 3D object from single image with 2D diffusion priors. As depicted in Figure 2, our exploration of the 3D darkside of a single image commences with the efficient generation of basic geometry (Section 3.1), represented through 3D Gaussian Splatting. In refinement stage (Section 3.2), we devise a method for transforming the rudimentary 3D Gaussian geometry into signed distance fields, and thereafter, we introduce a differentiable isosurface representation to further enhance the geometry and textures. To enable diverse 3D darkside of given single image, we present a novel approach to constrain two diffusion priors (Section 3.3), enabling the creation of varied yet coherent darkside textures by bounding gradient magnitude. With these approaches, our method can efficiently generate diverse, high-fidelity meshes from a single image.

[Figure 29]

[Figure 30]

[Figure 31]

Rasterization

𝐿 + 𝐿

[Figure 32]

[Figure 33]

3D Prior

#### Stage I

SDS

[Figure 34]

Top-K Densification Regularization

Gaussian Splatting

[Figure 35]

| | |
|---|---|
|SD|[Figure 36]<br><br>F|
| | |

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Input Image

𝜃

#### Stage II

[Figure 43]

3D Prior

2D Prior

𝐿 + 𝐿

𝜃

Angular-based Composition

Disentangled Texture Representation

FlexiCubes

SDS

- Fig. 2: Overview of Vista3D. We generate high-fidelity mesh from single image input in a coarse-to-fine manner. In the coarse stage, we utilize Gaussian Splatting to learn a coarse geometry with a 3D-aware 2D diffusion prior. We further extract sign distance fields from Gaussian Splatting for refinement. Another 2D diffusion prior is enabled with an angular-based composition to explore diverse darkside while retain 3D consistency in refinement stage.

###### 3.1 Coarse geometry from Gaussian Splatting

In the coarse stage of our framework, we focus on constructing a basic object geometry using Gaussian Splatting. This technique, as described in [14], represents 3D scenes as set of anisotropic 3D Gaussians. Compared to other neural inverse rendering methods, such as NeRF [25, 26], Gaussian Splatting demonstrates a notably faster convergence speed in inverse rendering tasks.

Some works [3, 41, 49] has attempted to introduce Gaussian Splatting into 3D generative models. In these methods, we found that directly using Gaussian splatting to generate detailed 3D objects requires optimizing a large number of 3D Gaussians, necessitating significant time for optimization and densification, which is still time-consuming. However, Gaussian Splatting can quickly create a coarse geometry from a single image using a limited number of 3D Gaussians within just one minute. Therefore, in our approach, we utilize Gaussian Splatting solely for the initial coarse geometry generation.

Specifically, each 3D Gaussians is parameterized by its central position x ∈ R3, scaling r ∈ R, rotation quaternion q ∈ R4, opacity α ∈ R, and spherical harmonics c ∈ R3 to represent color. To generate a coarse 3D object, we optimize a set of these Gaussian parameters Ψ = {Φi}, where Φi = {xi,ri,qi,αi,ci}. To render 3D Gaussians to 2D images, we utilized the highly-optimized tile based rasterization implementation [14].

To generate the coarse geometry of given single image Iref, we adopt Zero1-to-3 XL [5, 19] as 2D diffusion priors ϵϕ with pretrained parameters ϕ. This prior enables denoising of novel views based on the given image Iref and relative camera pose ∆π. Accordingly, we optimize the 3D Gaussians Ψ with SDS [29]:

∂IRπ ∂Ψ

∇ΨLSDS = Et,ϵ (ϵϕ (IRπ;t,Iref,∆π) − ϵ)

(1)

where π denotes the camera pose sampled around the object with fixed camera radius and FoV , IRπ is the rendered image from 3D Gaussian set Ψ with camera pose π, timestep t is annealed to weight the gaussian noise ϵ added to the rendered image. Beyond this basic approach, we introduce a Top-K Gradientbased Densification strategy to accelerate convergence and add two regularization terms to enhance the reconstructed geometry.

Top-K Gradient-based Densification. In the optimization process, we find the periodical densification [14] with naive gradient threshold is hard to tune due to the nature randomness of SDS. So we instead use a more robust densification strategy. Only gaussians points with top-k gradients will be densified during each interval, this simple strategy can stablize training cross various given images.

Scale & Transmittance Regularization. Additionally, We add two regularization terms to encourage Gaussian Splatting to learn more detailed geometry in this phase. A scale regularization is introduced to avoid too large 3d gaussians, and another transmittance regularization is adopted to encourage the geometry learning from transparent to solid. The overall loss function in this stage can be written as:

∇ΨLcoarse = λSDS∇ΨLSDS + λrgb∇ΨLrgb

+ λmask∇ΨLmask + λscale∇Ψ

∥si∥

i

(2)

Scale Regularization − λtr∇Ψmin(τ,

1 Nfg k

Tk)

;

Transmittance Regularization

where Lrgb and Lmask are two MSE loss computed between the rendered reference view and the given image. The term Tk = i αi ij−=11(1 − αj) denotes the transmittance value for the k-th pixel in IRπ, where Nfg is the total number of foreground pixels. Additionally, τ serves as a hyperparameter that is gradually annealed from 0.4 to 0.9, effectively regularizing transmittance over time.

###### 3.2 Mesh refinement and texture disentanglement

In the refinement stage, our focus shifts to transforming the coarse geometry, produced via Gaussian splatting, into signed distance fields (SDF) and refining its parameters using a hybrid representation.

This stage is crucial for overcoming the challenges presented in the coarse stage, notably the surface artifacts frequently introduced by Gaussian splatting. Due to the inability of Gaussian splatting to provide direct estimates of surface normals, we cannot employ traditional smoothing methods to alleviate these artifacts. To counter this, our method incorporates a hybrid mesh representation, which entails modeling the 3D object’s geometry as a differentiable isosurface and learning the texture using two distinct, disentangled networks. This dual approach not only smooths out the surface irregularities but also significantly improves the fidelity and overall quality of the 3D model.

Geometry representation. We utilize FlexiCubes to represent the geometry in our approach. FlexiCubes is a differentiable isosurface representation which allow local flexible adjustments to the extracted mesh geometry and connectivity [38]. The geometry of an object is depicted as a deformable voxel grid with learnable weights. Deformation δi ∈ R3 and sign distance field (SDF) si ∈ R is learnt for every vertices vi in the voxel grid. And interpolation weights β ∈ R20 and splitting weights γ ∈ R are learnt for each grid cell to position dual vertices and control quadrilaterals splitting. Triangle meshes can be extracted from it differentiablely through Dual Marching Cubes [28]. To bridge the gap between the learned coarse geometry and the isosurface representation, we initially extract a density field from Gaussian splattings using local density queries [41], followed by the application of marching cubes [22] to extract a base mesh Mcoarse. Subsequently, we query this base mesh at grid vertices vi to obtain the initial Signed Distance Field (SDF) s(vi). For stable optimization, the queried SDF is then scaled as follows:

ξ · s(vi) max{|sj| : sj ∈ S,sj < 0}

, where S = {si} (3)

s(vi) =

where sj < 0 indicates the field within the object. The scale factor ξ linearly increases from 1 to 3 during the optimization process.

Disentangled Texture Representation. For texture learning, we employ hash encoding followed by a MLP to directly learn albedo. However, distinct from text-to-3D tasks, we recognize two primary supervision sources in this task: the provided reference image and the SDS gradient from 2D Diffusion priors. Typically, a substantial loss weight λrgb is assigned for the reference image. This dominant reference image supervision can decelerate the convergence of textures in unseen views, particularly when unseen views significantly differ from the reference view.

To address this, we separate the texture into two hash encoding, utilizing a ratio that combines with the relative azimuth angle ∆θ = θπ − θref, where θπ represents the azimuth of the sampled camera pose π, and θref is the azimuth of the reference image. The hash encoding for a given query point κ in the rasterized triangle mesh is expressed as:

E = (1 − η)Hback(κ) + ηHref(κ) (4)

where Href and Hback denote learnable hash encoding facing forward and back, η = (cos(∆θ) + 1)/2 is the balance factor that varies with the sampled azimuth angle. Then the encoded feature E is fed into a MLP predict albedo values.

With these geometry and texture representation, we can render the 3D object to images by memory-efficient rasterization coupled with lambertian shading. Above learnable parameters Θ is refined with ∇ΘLrefine:

∇ΘLrefine = λSDS∇ΘLSDS

(5)

+ λSDF∇ΘLSDF + λconsistency∇ΘLconsistency

+ λrgbλSDS∇ΘLrgb + λmask∇ΘLmask;

where the LSDF is a simple SDF regulariztion term to avoid floaters, Lconsistency is a smooth loss applied on surface normals [16,24], Lrgb and Lmask are two MSE loss between the rendered reference view and the given image.

###### 3.3 Darkside Diversity via Prior Composition

In implementing our pipeline, we encountered a key challenge related to the lack of diversity in unseen views. This issue largely stems from the reliance on the Zero-1-to-3 XL prior, a model trained on synthetic 3D objects from ObjaverseXL [5]. While this prior is adept at handling 3D-aware generation based on reference images and relative camera poses, it tends to produce oversimplified or overly smooth results in unseen views. This limitation becomes especially pronounced when dealing with objects captured in the real world.

To address this, we integrate an additional prior from Stable-Diffusion, known for its ability to synthesize diverse images.

Darkside diversification with 2D diffusion. We introduce a second prior, ϵρ with pretrained parameters ρ, leading to two Score Distillation Sampling (SDS)

loss terms ∇LϕSDS and ∇LρSDS (Equation 1) for optimization. The optimal balance between these two priors remains relatively unexplored. While Magic123

[30] uses an empirical loss weight of 1/40 for the latter term, this approach may not fully harness the potential of the 2D prior. The key objective in introducing this 2D prior is to introduce greater diversity in unseen view. A small weight with ∇LρSDS may largely limit its effect.

To enhance the diversity in the unseen aspects of the given image, we employ a gradient constrain method to merge these two priors. We reformulate the SDS loss as a score function [29], ∇ΘLSDS(ϕ, x) = −Et,z

t|x∇Θlogpϕ(zt|y), where t is the timestep and zt is noise latent.

Here ∇LϕSDS is a 3D-aware term conditioned on y = {∆π,Iref}, while

∇LρSDS is a diverse text-to-image term conditioned on text prompt y = PT. With different condition y, the score function of these two SDS term varies. To

retain 3D consistency of unseen views, the magnitude of ∇Θlogpρ(zt|y) need to be constrained with respect to the 3D-aware term ∇Θlogpϕ(zt|y). And to avoid the texture to be over-smoothed by the 3D-aware diffusion model, the magnitude of ∇Θlogpϕ(zt|y) is indeed to be constrained with the ∇Θlogpρ(zt|y) term.

Angular-based Score Composition. Since the noise latents zt in both priors have different encoding spaces, direct evaluation of their magnitudes using the predicted noise difference ϵρ−ϵ is not feasible. Instead, we evaluate the magnitude of these terms by observing their gradient on the rendered image x, specifically ∇xLSDS. Consequently, we establish upper and lower bounds for the gradient magnitude ratio of these two SDS terms, allowing for a more accurate and feasible evaluation method:

Blower(η,ι) ≤ G = ||∇xLρSDS||2 ||∇xLϕSDS||2

≤ Bupper(η,ι) (6)

When this ratio exceeds Bupper, we adjust the magnitude of ∇xLρSDS using the factor Bupper/G. Conversely, if the ratio falls below Blower, we scale the

magnitude of ∇xLϕSDS using G/Blower. And this Bupper and Blower are regulated by the balance factor η, influenced by the camera pose, and by iterations ι,

facilitating a balance between diversity and 3D consistency.

##### 4 Experiments

- 4.1 Implementation Details

Coarse geometry learning. In this phase, the input image undergoes preprocessing with SAM [15,23,34], where the object is extracted and recentered. We initialize all 3D Gaussians with an opacity of 0.1 and a grey color, confined within a sphere of radius 0.5. The rendering resolution is progressively increased from 64 to 512. This stage involves a total of 500 optimization steps, with the densification and pruning of 3D Gaussians occurring every 100 iterations. The top-K densification starts at a ratio of 0.5 and gradually anneals to 0.1, while the pruning opacity remains constant at 0.1. After the first densification, transmittance regularization is activated and selectively applied to the top-80% opacity values of 3D Gaussians to avoid affecting transparent Gaussians. Scale regularization is enforced using L1 norm. The weights of λscale and λtr are maintained at 0.01 and 1, respectively, throughout the optimization, whereas λrgb and λmask are gradually increased from 0 to 10000 and 1000, respectively. The timestep for SDS is linearly annealed from 980 to 20. For camera pose sampling, the azimuth is sampled in the range of [−180,180] and elevation in [−45,45], with a fixed radius of r = 2. This phase of optimizing the coarse geometry takes about 30 s. Mesh refinement. In the refinement phase, we configure the grid size of FlexiCubes to 803 within the space [−1,1]3. The coarse geometry obtained from the initial stage is recentered and rescaled to initialize the Signed Distance Field (SDF) for the vertices of this grid. Interpolation weights are set to 1, and all deformations start at 0. For texture, we use two hash encodings with a two-layer Multilayer Perceptron (MLP). The batch size is maintained at 4. The learning rate for deformation and interpolation weights is 0.005, while it’s 0.001 for SDF, and 0.01 for texture parameters. The rendering resolution is gradually increased from 64 to 512. In Equation 5, the loss weights are set as follows: λrgb = 1500, λmask = 5000, λsdf = 1, and λSDS = 1. We develop two versions for optimization: Vista3D-S and Vista3D-L. Vista3D-S performs 1000 steps of optimization solely with the 3D-aware prior, aiming to generate 3D mesh within

###### 5 minutes. Vista3D-L undergoes 2000 steps of optimization with two diffusion priors to create more detailed 3D objects. The entire optimization process for Vista3D ranges from 15 to 20 minutes. In this stage, camera poses are sampled using a 3D-aware Gaussian unsampling strategy to expedite convergence (additional details are provided in the supplementary material). All experiments are conducted on an RTX3090 GPU. Score distillation sampling. In SDS optimization, the practice of linearly annealing the timestep t to adjust the noise level has been established as effective for producing higher-quality 3D objects [11]. However, in our experiments, we observed that linear annealing may not be the optimal strategy. Consequently, we

###### have implemented an interval annealing approach. In this approach, the timestep t is randomly sampled from an annealing interval rather than adhering to a fixed linear progression. This strategy has been found to effectively mitigate the artifacts commonly observed with linear annealing.

Reference Magic123 Vista3D-S ( 𝟐𝟎× faster)

DreamGaussian

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

2 hours 5 minutes

2 minutes

- Fig. 3: Qualitative Comparison on image-to-3D generation. We compare our Vista3D-S with DreamGaussian [41], and Magic123 [30]. Vista3D-S only takes 5 minutes to reconstruct single 3D object, yielding competitive geometry and more consistent textures compared to Magic123 [30] with 20× speedup.

Angular diffusion prior composition. In our model, we utilize two diffusion models: Zero-1-to-3 XL [5,19] and the Stable-Diffusion model [31]. For the StableDiffusion model, the timestep t is scaled by the factor η to ensure consistency with the reference view. When editing with both diffusion priors, we start with a large initial upper bound Bupper = 100, which is linearly annealed to 10 across optimization iterations. For front-facing views, where η > 0.75, we adjust the upper bound using the factor (1−η). The lower bound is specifically implemented for unseen views with η < 0.5, and its range is gradually reduced from 10 to 1 during the optimization process. For enhancements using the diffusion prior, we apply tighter constraints, with Bupper being reduced from 2 to 0.5. The text prompts utilized for the Stable-Diffusion model are derived from the image captions generated by GPT-4.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

a white standing panda, cartoon style

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

An anthropomorphic cat wearing a tweed outfit, complete with a matching cap

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

An anthropomorphic bull in a casual grey sweater and blue jeans

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

Two cute pandas stacked on top of each other

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

A dark orange and green dinosaur, lifelike style

Reference One-2-3-45 Wonder3D Vista3D-L (ours)

- Fig. 4: Qualitative Comparison with One-2-3-45 [18] and Wonder3D [21]. In this comparison, we render two views of each 3D object as generated by One-2-3-45 and Wonder3D. For Vista3D-L, we detail the text prompts utilized for the generation of each 3D object, showcasing three rendered views alongside a single normal map for a comprehensive comparison.

- 4.2 Qualitative Comparison In Figure 3, we show our efficient Vista3D-S is capable of generating competitive
- 3D objects with a 20× speedup compared to existing coarse-to-fine methods. For Vista3D-L, as depicted in Figure 1 and Figure 4, we highlight our angular gradient constraint which distinguishes our framework from previous image-to-3D methods, as it can explore the diversity of the backside of single images without sacrificing 3D consistency. In Figure 3, we primarily compare our Vista3D-S with two baselines, Magic123 [30] and DreamGaussian [41], for generating 3D objects from a single reference view. Regarding the quality of generated 3D objects, our method outperforms these two methods in terms of both geometry and texture. Regarding Vista3D-L, we compare it with two inference-only single view reconstruction models, specifically One-2-3-45 [18] and Wonder3D [21]. As shown in Fig. 4, One-2-3-45 tends to produce blurred texture and may result in incomplete geometry for more complex objects, while our Vista3D-L achieves more refined textures, particularly on the backside of 3D objects, using userspecified text prompts. And Wonder3D often resorts to simpler textures due to its primary training on synthetic datasets [5], which occasionally leads to out-of-

distribution issues for certain objects. In contrast, Vista3D-L offers zero-shot 3D object reconstruction by controlling two diffusion priors, enabling more detailed and consistent textural. Moreover, given that only a single reference view of the object is provided, we posit that the object should be amenable to editing during optimization with user-specified prompts. To illustrate this, we display several results in Figure 1 that emphasize the potential for editing.

| |Type|CLIP-Similarity ↑|Time Cost ↓<br><br>|
|---|---|---|---|
|One-2-3-45 [18] Point-E [27] Shape-E [13] Zero-1-to-3 [19] DreamGaussian [41] Magic123 [30] DreamCraft3D [40]|Inference Inference Inference Optimization Optimization Optimization Optimization<br><br>|0.594 0.587 0.591 0.778 0.738 0.802 0.842|45 s 78 s 27 s 30 min 2 min 2 h 3.5 h<br><br>|
|Vista3D-S Vista3D-L|Optimization Optimization<br><br>|0.831 0.868|5 min 15 min<br><br>|

Table 1: Quantitative Comparisons on generation quality in terms of CLIP-Similarity for image-to-3D task. Average generation time is reported.

###### 4.3 Quantitative Comparison

In our evaluation, we employ the CLIP-similarity metric [19, 24, 30] to assess the performance of our method in 3D reconstruction using the RealFusion [24] dataset, which comprises 15 diverse images. Consistent with the settings used in previous studies, we sample 8 views evenly across an azimuth range of [−180,180] degrees at zero elevation for each object. The cosine similarity is then calculated using the CLIP features of these rendered views and the reference view. Table 1 highlights that Vista3D-S attains a CLIP-similarity score of 0.831, with an average generation time of just 5 minutes, thereby surpassing the performance of the Magic123 [30]. Furthermore, when compared to another optimization-based method, DreamGaussian [41], Vista3D-S may take longer at 5 minutes, but it significantly improves consistency, as evidenced by the higher CLIP-Similarity score. For Vista3D-L, we apply an enhancement-only setting. By employing angular diffusion prior composition, our method achieves a higher CLIP-Similarity of 0.868. The capabilities of Vista3D-L, especially in generating objects with more detailed and realistic textures through prior composition, are demonstrated

- in Figure 4. Additionally, we conduct quantitative experiments on the Google Scanned Object (GSO) [7] Dataset, following the setting in SyncDreamer [20]. We evaluate each method using 30 objects and computed PSNR, SSIM, and LPIPS [54] between the rendered views of the 3D object and 16 ground-truth anchor views. The results, as shown in Tab. 2, reveal that our Vista3D-L achieves SOTA performance among these methods with a large margin. Vista3D-S also demonstrates competitive performance, albeit with a single diffusion prior.

| |PSNR ↑ SSIM ↑ LPIPS ↓|
|---|---|
|RealFusion [24] Make-it-3D [42] Zero-1-to-3 [19] One-2-3-45 [18] SyncDreamer [20] DreamGaussian [41] Magic123 [30]|15.26 0.722 0.283 15.79 0.741 0.245 18.93 0.779 0.166 17.47 0.768 0.184 20.05 0.798 0.146<br><br>23.43 0.832 0.092<br><br>24.89 0.875 0.084<br><br><br>|
|Vista3D-S Vista3D-L|25.42 0.912 0.073<br><br>26.31 0.929 0.062<br>|

Table 2: Quantitative Comparison on the GSO [7] dataset

###### 4.4 User study

In our user study, we evaluate reference view consistency and overall 3D model quality [41]. The evaluation encompasses four methods: DreamGaussian [41], Magic123 [30], and our own Vista3D-S and Vista3D-L. We recruited 10 participants for this user study. Each was asked to sort generated 3D object from different methods in terms of view consistency and overall quality respectively. Thus, the scores presented for each metric range from 1 to 4. The results, presented in Table 3, reveal that our Vista3D-S outperforms the previous methods in both view consistency and overall quality. Furthermore, the adoption of the angular prior composition in Vista3D-L leads to additional improvements in both the consistency and quality of the generated 3D objects.

| |DreamGaussian [41] Magic123 [30] Vista3D-S Vista3D-L|
|---|---|
|View Consistency ↑ Overall Quality ↑|1.78 2.11 2.87 3.24<br><br>2.02 1.83 2.81 3.33<br>|

Table 3: User study of Vista3D. We conduct user study in terms of view consistency and overall quality, the score ranges from 1 to 4, the higher the better.

###### 4.5 Ablation Study

Coarse-to-fine framework. Our framework integrates a coarse stage to learn initial geometry then a fine stage to refine geometry and shade textures. We validate the necessity of such a coarse-to-fine pipeline in Figure 5 (a). We first commence with isosurface representation to learn geometry directly, finding the geometry optimization is prone to collapse without preliminary geometry initialization. Thus, a coarse initialization becomes imperative. Beside, we present the normal map of a rough mesh extracted from 3DGS from the coarse stage. It is observed that the coarse stage tends to generate rough even non-watertight geometry, both difficult to mitigate. These findings demonstrate that combining both stages is crucial for the optimal performance of Vista3D.

Disentangled Texture. For validating the effectiveness of the disentangled texture, we compare adopting both hash encodings with single hash encoding

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Ref FlexiCubes 3DGS Ours Single hash-encoding Both hash-encoding

a) Ablation study of overall framework b) Ablation study of angular hash-encoding

Fig. 5: Ablation study of overall framework and disentangled texture.

- in Figure 5 (b). With both hash-encodings, the artifacts on the reconstructed robot are notably reduced, especially at the backside. Further, we visualize the disentangled texture in supplementary Figure 6(b). Specifically, when visualizing Href, Hback is set as 0 in Equation 4, and vice versa. From the shown visualization, we can clearly find that the facing-forward hash encoding Href mainly encodes the detail features consistent with the given reference view. While the

back hash encoding Hback mainly encodes the features in the unseen views. The textures of the facing-forward view and back views are disentangled and learned in two separate hash encodings, which can facilitate learning better textures near the reference view and in unseen views.

##### 5 Conclusion

In this paper, we present a coarse-to-fine framework Vista3D to delve into the 3D darkside of a single input image. This framework facilitates user-driven editing through text prompts or enhances generation quality using image captions. The generation process begins with a coarse geometry obtained through Gaussian Splatting, which is subsequently refined using an isosurface representation complemented by disentangled textures. The design of these 3D representations enables the generation of textured meshes within a mere 5 minutes. Additionally, the angular composition of diffusion priors empowers our framework to reveal the diversity of unseen views while maintaining 3D consistency. Our approach surpasses previous methods in terms of realism and detail, striking an optimal balance between generation time and the quality of the textured mesh. We hope our contributions will inspire future advancements and foster future exploration into the 3D darkside of single images.

##### Acknowledgement

This project is supported by the Ministry of Education, Singapore, under its Academic Research Fund Tier 2 (Award Number: MOE-T2EP20122-0006), and the National Research Foundation, Singapore, under its Medium Sized Center for Advanced Robotics Technology Innovation.

## Vista3D: Unravel the 3D Darkside of a Single Image

### Supplementary Material

##### 1 More experimental results

###### 1.1 More ablation studies

[Figure 145]

[Figure 146]

[Figure 147]

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

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Front Back Front Back

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Front Back Front Back

[Figure 175]

[Figure 176]

𝐻 𝐻

Coarse stage Without Top-K densification

Without Transmittance regularization

Without scaling regularization

𝐻 𝐻

Coarse stage Without Top-K densification

Without Transmittance regularization

Without scaling regularization

(b) Visualization of the disentangled texture. Here we showcase a generated 3D object. The left side is visualized from the facing-forward hash encoding Href, while the right side is visualized from the back hash encoding Hback.

(a) Ablation study of the coarse stage. Here we conduct four settings on the coarse stage, including w/o Top-K densification, w/o transmittance and scaling regularization for comparison.

Fig. 6: Ablation study of the coarse stage and disentangled texture.

Top-k densification. We compare our densification strategy against a naive gradient threshold approach. This comparison is illustrated in the second column of Figure 6a. Using a naive gradient threshold often results in excessive densification of 3D Gaussians, causing geometry to appear swollen. Furthermore, finding an appropriate gradient threshold is challenging, as it varies from case to case. In contrast, our method deterministically controls the densification ratio throughout the optimization process. Consequently, the total number of 3D Gaussians at convergence is solely influenced by the hyperparameter of pruning opacity, effectively maintaining the number of 3D Gaussians within a reasonable range and yielding more accurate geometry.

Regularization with 3DGS. In the third and fourth columns of Figure 6a, we conduct ablation experiments on the two regularization terms specified in Equation 2: transmittance regularization and scale regularization. Removing the transmittance regularization tends to produce objects with holes, resulting in coarse meshes from these 3D Gaussians that are often not watertight, complicating refinement stage optimization. On the other hand, excluding only the scale regularization often leads to coarser details in the geometry. This may be caused by Gaussians with larger scales oversmoothing the local geometries.

The effect of prior composition. To explore the 3D dark side of a single image, we introduce a gradient constraint-based method in Sec. 3.3 to control two diffusion priors in the image-to-3D task. Here we conduct an ablation study to validate the effectiveness of this component. As shown in Fig. 7, without this

- 2 Shen et al.

score composition, though detailed texture on the backside can still be generated, results in degraded consistency between front views and reference images. Another setting involves a naive weighting strategy; we follow Magic123 [30] to set a weighting factor of 1/40 on the SDS term LρSDS with diffusion prior ϵρ. With this setting, the backside of the generated 3D objects appears overly smoothed. In contrast, incorporating score composition enables our Vista3D to robustly generate textures that are both detailed and consistent across the front and back views of 3D objects.

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

w/o score composition with score composition naive weighting

Fig. 7: Ablation Study of Score Composition. Without score composition, the consistency between the reference view and front view is degraded. Applying naive weighting results in over-smoothed textures on back views.

###### 1.2 More qualitative results

Figure 8 showcases the qualitative results of Vista3D-L with diffusion prior composition compared to Vista3D-S with a single diffusion prior. Particularly in scenarios where the provided reference view is less informative, such as when only a side or back view of an object is available, Vista3D-L demonstrates a superior ability to generate more detailed textures compared to Vista3D-S, especially when specific text prompts are used. For example, in the case of the astronaut, Vista3D-S tends to produce oversmoothed textures. In contrast, when using Vista3D-L, the textures generated are notably more vivid and detailed.

##### 2 Camera Pose Sampling

As illustrated in Fig. 9, our approach adopts a 3D-aware camera pose sampling strategy in the refinement stage, diverging from the standard uniform sampling used in previous image-to-3D works [30,41,42]. This approach not only speeds up convergence but also enhances visual quality.

Specifically, for a given conditional reference image Iref, the pre-trained Zero1-to-3 model [19] ϵϕ is capable of approximating the underlying 3D object distribution PI

(x). Leveraging this, we employ its estimated empirical error for 3D-aware sampling.

ref

In this sampling stage, camera poses are sampled from a sphere surface surrounding the central object, divided evenly into N sub-regions Ri with azimuth

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

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Reference Vista3D-S Vista3D-L

Fig. 8: Qualitative Comparison between Vista3D-S and Vista3D-L

ranging from [−180,180] degrees, as shown on the left side of Figure 9. Memory queues of fixed length T are established for each sub-region to store empirical errors estimated during the SDS optimization, directly derived from SDS as (ϵϕ − ϵ) in Equation 1.

When performing pose sampling, an empirical Probability Density Function (PDF) P3d(Ri) is created from these N memory queues. Additionally, given the supplementary supervision from the reference image Iref for forward-facing camera poses, we integrate Gaussian unsampling to reduce sampling frequency on forward-facing poses and increase it for unseen views. This unsampling employs a rejection sampling with a truncated Gaussian distribution, depicted on the right side of Figure 9. Each sub-region is mapped onto this truncated Gaussian PDF, with regions overlapping significantly with the reference view being more likely to be sampled.

In this process, a camera pose is sampled by initially performing Gaussian unsampling to determine a rejection index n ∈ [0,N − 1]. Subsequently, we modify the empirical PDF by setting P3d(Rn) = 0 and normalizing it. A subregion index is then sampled from this discrete PDF P˜3d(Ri), and a camera pose is uniformly sampled from this chosen sub-region.

In our implementation, we configure N = 5, and initially perform uniform camera pose sampling during the first 100 iterations. For the Gaussian Unsampling, we utilize a truncated Gaussian distribution spanning [−1,1], with N(0,0.5). This distribution is evenly divided into N intervals to facilitate the sampling process.

[Figure 231]

###### 4 Shen et al.

[Figure 232]

Sampled Pose

[Figure 233]

Default Azimuth

Fig. 9: 3D-aware Pose Sampling, Camera poses are sampled from an empirical PDF with a truncated Gaussian unsampling.

##### 3 Timestep Sampling in SDS

Pioneering work DreamFusion [29] randomly sample timestep t from U(20,980) in the SDS optimization. However, Dreamtime [11] critiques this strategy, suggesting that such random sampling is misaligned with the Denoising Diffusion Probabilistic Models (DDPM) sampling process and leads to inefficient and inaccurate optimization in SDS. Dreamtime suggests a deterministic Time Prioritized (TP) strategy where each iteration step is assigned a unique, decrementally decreasing timestep t.

However, we observed that this deterministic approach falls short in SDS optimization. Artifacts generated by large timesteps are not effectively compensated for by smaller timesteps, often exacerbating the problem. To rectify this, we propose an interval-based annealing method for the timestep. Specifically, we define a maximum timestep tmax and a minimum timestep tmin for each optimization interval, updating them every 50 optimization steps. The timestep is then sampled from the dynamically adjusted interval U(tmin,tmax). This approach effectively alleviates the artifacts that larger timesteps tend to cause.

##### 4 Limitations

Despite Vista3D demonstrating prowess in exploring the 3D dark side of a single image, we acknowledge several limitations for future exploration. Employing a Score Distillation Sampling (SDS) based architecture, Vista3D necessitates optimization for each 3D object it generates, positioning its efficiency a notch below that of purely feed-forward image-to-3D methods. The amount of public

- 3D data is relatively limited, often resulting in the generation of simplistic 3D

###### objects by feed-forward methodologies. Vista3D leverages diffusion prior composition to facilitate the reconstruction of more diverse 3D objects. This strategy holds promise for the creation of additional 3D data, potentially alleviating the current data scarcity and enabling the development of more sophisticated pretrained image-to-3D models.

##### References

- 1. Cao, Z., Hong, F., Wu, T., Pan, L., Liu, Z.: Large-vocabulary 3d diffusion model with transformer. arXiv preprint arXiv:2309.07920 (2023) 1
- 2. Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In: ICCV (October 2023) 4
- 3. Chen, Z., Wang, F., Liu, H.: Text-to-3d using gaussian splatting. arXiv preprint arXiv:2309.16585 (2023) 4, 5
- 4. Chung, J., Lee, S., Nam, H., Lee, J., Lee, K.M.: Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384 (2023) 4
- 5. Deitke, M., Liu, R., Wallingford, M., Ngo, H., Michel, O., Kusupati, A., Fan, A., Laforte, C., Voleti, V., Gadre, S.Y., et al.: Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663 (2023) 2, 3, 5, 8, 10, 11
- 6. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: CVPR. pp. 13142–13153 (2023) 2, 3
- 7. Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T.B., Vanhoucke, V.: Google scanned objects: A high-quality dataset of 3d scanned household items. In: 2022 International Conference on Robotics and Automation (ICRA). pp. 2553–2560. IEEE (2022) 12, 13
- 8. Duggal, S., Pathak, D.: Topologically-aware deformation fields for single-view 3d reconstruction. In: CVPR. pp. 1536–1546 (2022) 2, 3
- 9. Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023) 3
- 10. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–11 (2024) 4
- 11. Huang, Y., Wang, J., Shi, Y., Qi, X., Zha, Z.J., Zhang, L.: Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422 (2023) 9, 4
- 12. Jain, A., Mildenhall, B., Barron, J.T., Abbeel, P., Poole, B.: Zero-shot text-guided object generation with dream fields. In: CVPR. pp. 857–866. IEEE (2022) 4
- 13. Jun, H., Nichol, A.: Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463 (2023) 1, 12
- 14. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 2, 4, 5, 6
- 15. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023) 9
- 16. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: CVPR. pp. 300–309 (2023) 4, 8
- 17. Liu, M., Shi, R., Chen, L., Zhang, Z., Xu, C., Chen, H., Zeng, C., Gu, J., Su, H.: One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion (2023) 3

- 18. Liu, M., Xu, C., Jin, H., Chen, L., Xu, Z., Su, H., et al.: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928 (2023) 11, 12, 13
- 19. Liu, R., Wu, R., Van Hoorick, B., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9298–9309 (2023) 2, 3, 5, 10, 12, 13
- 20. Liu, Y., Lin, C., Zeng, Z., Long, X., Liu, L., Komura, T., Wang, W.: Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023) 3, 12, 13
- 21. Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al.: Wonder3d: Single image to 3d using crossdomain diffusion. arXiv preprint arXiv:2310.15008 (2023) 2, 3, 11
- 22. Lorensen, W.E., Cline, H.E.: Marching cubes: A high resolution 3d surface construction algorithm. In: Seminal graphics: pioneering efforts that shaped the field, pp. 347–353 (1998) 7
- 23. Lu, J., Yang, X., Wang, X.: Unsegment anything by simulating deformation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

(2024) 9

- 24. Melas-Kyriazi, L., Laina, I., Rupprecht, C., Vedaldi, A.: Realfusion: 360deg reconstruction of any object from a single image. In: CVPR. pp. 8446–8455 (2023) 3, 8, 12, 13
- 25. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021) 4, 5
- 26. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022) 5
- 27. Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., Chen, M.: Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751

(2022) 1, 12

- 28. Nielson, G.M.: Dual marching cubes. In: IEEE visualization 2004. pp. 489–496. IEEE (2004) 7
- 29. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. In: ICLR. OpenReview.net (2023) 1, 2, 4, 5, 8
- 30. Qian, G., Mai, J., Hamdi, A., Ren, J., Siarohin, A., Li, B., Lee, H.Y., Skorokhodov,

I., Wonka, P., Tulyakov, S., et al.: Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843 (2023) 2, 3, 8, 10, 11, 12, 13

- 31. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR. pp. 10684–10695 (2022) 10
- 32. Sargent, K., Li, Z., Shah, T., Herrmann, C., Yu, H.X., Zhang, Y., Chan, E.R., Lagun, D., Fei-Fei, L., Sun, D., et al.: Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994 (2023) 2
- 33. Seo, J., Jang, W., Kwak, M.S., Ko, J., Kim, H., Kim, J., Kim, J.H., Lee, J., Kim, S.: Let 2d diffusion model know 3d-consistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937 (2023) 3
- 34. Shen, Q., Yang, X., Wang, X.: Anything-3d: Towards single-view anything reconstruction in the wild (2023) 1, 2, 3, 9

- 35. Shen, Q., Yang, X., Wang, X.: Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally. arXiv preprint arXiv:2409.08270 (2024) 4
- 36. Shen, Q., Yi, X., Wu, Z., Zhou, P., Zhang, H., Yan, S., Wang, X.: Gamba: Marry gaussian splatting with mamba for single view 3d reconstruction. arXiv preprint arXiv:2403.18795 (2024) 3
- 37. Shen, T., Gao, J., Yin, K., Liu, M.Y., Fidler, S.: Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems 34, 6087–6101 (2021) 4
- 38. Shen, T., Munkberg, J., Hasselgren, J., Yin, K., Wang, Z., Chen, W., Gojcic, Z., Fidler, S., Sharp, N., Gao, J.: Flexible isosurface extraction for gradient-based mesh optimization. ACM Trans. Graph. 42(4), 37–1 (2023) 2, 4, 7
- 39. Shi, R., Chen, H., Zhang, Z., Liu, M., Xu, C., Wei, X., Chen, L., Zeng, C., Su, H.: Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110 (2023) 2, 3
- 40. Sun, J., Zhang, B., Shao, R., Wang, L., Liu, W., Xie, Z., Liu, Y.: Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818 (2023) 2, 3, 12
- 41. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G.: Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023) 3, 4, 5, 7, 10, 11, 12, 13, 2
- 42. Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., Chen, D.: Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. In: ICCV. pp. 22819–22829 (October 2023) 1, 2, 3, 13
- 43. Wang, H., Du, X., Li, J., Yeh, R.A., Shakhnarovich, G.: Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In: CVPR. pp. 12619– 12629 (2023) 4
- 44. Wang, S., Yang, X., Shen, Q., Jiang, Z., Wang, X.: Gflow: Recovering 4d world from monocular video. arXiv preprint arXiv:2405.18426 (2024) 4
- 45. Wu, Z., Zhou, P., Yi, X., Yuan, X., Zhang, H.: Consistent3d: Towards consistent high-fidelity text-to-3d generation with deterministic sampling prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9892–9902 (2024) 1
- 46. Xu, J., Cheng, W., Gao, Y., Wang, X., Gao, S., Shan, Y.: Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 (2024) 3
- 47. Xu, Y., Tan, H., Luan, F., Bi, S., Wang, P., Li, J., Shi, Z., Sunkavalli, K., Wetzstein, G., Xu, Z., et al.: Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. arXiv preprint arXiv:2311.09217 (2023) 3
- 48. Yang, X., Wang, X.: Hash3d: Training-free acceleration for 3d generation. arXiv preprint arXiv:2404.06091 (2024) 1
- 49. Yi, T., Fang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., Wang, X.: Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. arXiv preprint arXiv:2310.08529 (2023) 4, 5
- 50. Yi, X., Wu, Z., Shen, Q., Xu, Q., Zhou, P., Lim, J.H., Yan, S., Wang, X., Zhang, H.: Mvgamba: Unify 3d content generation as state space sequence modeling. arXiv preprint arXiv:2406.06367 (2024) 3
- 51. Yi, X., Wu, Z., Xu, Q., Zhou, P., Lim, J.H., Zhang, H.: Diffusion time-step curriculum for one image to 3d generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9948–9958 (2024) 2
- 52. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: CVPR. pp. 4578–4587 (2021) 2, 3

- 53. Yu, Z., Sattler, T., Geiger, A.: Gaussian opacity fields: Efficient and compact surface reconstruction in unbounded scenes. arXiv preprint arXiv:2404.10772 (2024) 4
- 54. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018) 12
- 55. Zou, Z.X., Yu, Z., Guo, Y.C., Li, Y., Liang, D., Cao, Y.P., Zhang, S.H.: Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10324–10335 (2024) 3

