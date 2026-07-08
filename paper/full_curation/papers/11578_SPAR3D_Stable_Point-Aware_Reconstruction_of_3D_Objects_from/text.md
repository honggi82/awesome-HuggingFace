## SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images

Zixuan Huang1,2∗ Mark Boss1 Aaryaman Vasishta1 James M. Rehg2 Varun Jampani1 1Stability AI, 2UIUC

# arXiv:2501.04689v1[cs.CV]8Jan2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Edit Edit

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Input Image Point Cloud Mesh Output Input Image Point Cloud Mesh Output

Figure 1. We present SPAR3D, a state-of-the-art 3D reconstructor that reconstructs high-quality 3D meshes from single-view images. SPAR3D enjoys a fast reconstruction speed at 0.7 seconds and supports interactive user edits.

### Abstract

We study the problem of single-image 3D object reconstruction. Recent works have diverged into two directions: regression-based modeling and generative modeling. Regression methods efficiently infer visible surfaces, but struggle with occluded regions. Generative methods handle uncertain regions better by modeling distributions, but are computationally expensive and the generation is often misaligned with visible surfaces. In this paper, we present SPAR3D, a novel two-stage approach aiming to take the

*Work done at Stability AI.

best of both directions. The first stage of SPAR3D generates sparse 3D point clouds using a lightweight point diffusion model, which has a fast sampling speed. The second stage uses both the sampled point cloud and the input image to create highly detailed meshes. Our two-stage design enables probabilistic modeling of the ill-posed single-image 3D task while maintaining high computational efficiency and great output fidelity. Using point clouds as an intermediate representation further allows for interactive user edits. Evaluated on diverse datasets, SPAR3D demonstrates superior performance over previous state-of-the-art methods, at an inference speed of 0.7 seconds. Project page with

code and model: https://spar3d.github.io

### 1. Introduction

Reconstructing 3D objects from monocular images is a fundamental problem in computer vision. An efficient reconstruction system opens up a wide range of applications, including augmented reality, filmmaking, and manufacturing. Monocular 3D reconstruction is also a complex inverse problem: while the visible surface can be estimated from shading, predicting the occluded surface necessitates a strong 3D object prior. Our field has seen a divergence in two different directions: feedforward regression [2, 10, 19, 24, 25, 27, 37, 53, 54, 59–62, 65, 66, 69] and diffusion-based generation [6, 8, 9, 26, 29, 31–35, 39, 46– 48, 68, 71]. Despite the significant progress made in both directions, each has fundamental limitations.

Regression-based models are highly effective in adhering to the visible surface in the image, and the inference speed is typically fast. However, they make the oversimplified assumption of bijective mapping between images and 3D. This assumption introduces ambiguity in the learning objective, leading to poorly estimated surfaces and textures in occluded regions. On the other hand, diffusion-based approaches are generative and do not predict the statistical mean. However, their iterative sampling at inference time is computationally inefficient when modeling high-resolution 3D. Additionally, previous studies such as [27] indicate that diffusion-generated 3D models exhibit worse alignment to the surface visible in the input image. How can we take the best of both worlds while avoiding their limitations?

In light of this, we propose SPAR3D, which breaks the 3D reconstruction process down into two stages: the point sampling stage and the meshing stage. The point sampling stage uses diffusion models to generate sparse point clouds, followed by the meshing stage, transforming point clouds into highly detailed meshes. Our main idea is to offload the uncertainty modeling to the point sampling stage, where the low resolution of the point clouds allows rapid iterative sampling. The subsequent meshing stage leverages the local image features to transform the point cloud into a detailed mesh of high output fidelity. Reducing the meshing uncertainty with point clouds further facilitates unsupervised learning of inverse rendering, which reduces the baked-in lighting in the textures. Our two-stage design enables SPAR3D to significantly outperform previous regressive methods, while preserving high computational efficiency and fidelity to input observation.

A key design choice of our method is the usage of point clouds to connect the two stages. To ensure fast reconstruction, our intermediate representation needs to be lightweight so it can be efficiently generated. However, it should provide enough guidance to the meshing stage. This inspires us

to use point clouds, which are perhaps the most computationally efficient 3D representation because all information bits are used to represent the surface. Moreover, the lack of connectivity, typically considered as the drawback of point clouds, now turns into an advantage with our two-stage approach for editing purposes. When the back surface does not align with user expectations, local edits can be easily made on the low-resolution point clouds without worrying about topologies (see Fig. 1 bottom). Feeding edited point clouds into the meshing stage produces better meshes tailored towards user requirements.

Our experiments demonstrate the superiority of SPAR3D over previous state-of-the-art methods, with solid quantitative and qualitative results on various data sources. SPAR3D also exhibits a strong generalization ability to inthe-wild images and AI-generated images. With a total inference time below 0.7 seconds, SPAR3D is not only efficient but also allows for easy user-driven edits, offering a practical solution to the task of monocular 3D reconstruction. We hope that this is a meaningful step towards scalable generation of high-quality 3D assets.

### 2. Related Work

Feedforward 3D reconstruction methods address the problem of 3D object reconstruction by learning a feedforward model in a regression-based manner. Earlier works [10, 19, 25, 37, 59, 62, 66] in this field typically predict only the geometry and train on small datasets [5, 50], which limits their generalization ability. Recently, larger 3D datasets [11, 42] have been collected, unlocking the potential to train feedforward 3D models at scale [24, 27, 61]. These models exhibit great generalization ability to unseen images, and excel at producing reconstructions that tightly align with the observed cues in the input image. In particular, LRM [24] and follow-up works [2, 53, 54, 60, 65, 69] show that properly designed large transformer models can be trained using only rendering losses to capture object geometry and texture in great detail. Despite the high fidelity and computational efficiency of these models, the oversimplified bijective assumption in these regressive approaches results in oversmoothed unseen surfaces. Multi-view diffusion models [34, 35, 45, 46] have been considered as a remedy for this, where additional viewpoints are synthesized as input to the feedforward model [53, 60, 65]. However, the inconsistency across viewpoints often leads to significant artifacts on the reconstructed surfaces, and the computational efficiency of these approaches is severely affected by the slow multi-view generation process. Our model also aims to overcome the learning ambiguity in regressive approaches, but our point sampling approach is inherently 3Dconsistent and computationally efficient, and further allows easy user edits.

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]| |
|---|---|
| | |

[Figure 21]

| | | |
|---|---|---|
| | | |

DINOv2 Encoder

Triplane Transformer

[Figure 22]

Triplane

[Figure 23]

Texture

Geometry

Input Image

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

Point Cloud Denoiser

Illumination Encoder

RENI++ Decoder

[Figure 27]

Noisy Points

Sampled Points

Environment Map

- Figure 2. SPAR3D Overview. Conditioned on the input image, SPAR3D first leverages a point diffusion model to generate a sparse point cloud. The triplane transformer then uses the sampled point cloud and image features to produce high-resolution triplane features. The triplane features are then queried to reconstruct the geometry, texture, and illumination of the object in the image.

Generative 3D modeling learns the image-conditioned distribution of 3D assets instead of a deterministic mapping. Early 3D generative works use GAN [4, 16, 28, 55], normalizing flow [30, 67] or VAE [17, 38, 64] as the generative framework. Inspired by the success of 2D diffusion models [14, 43], 3D diffusion models [6, 8, 9, 26, 29, 31– 35, 39, 46–48, 68, 71] have also been extensively explored in recent works. Despite the advantage of probabilistic modeling that avoids over-smoothed results, diffusionbased 3D generation has two drawbacks: 1) not aligning well with input observations, and 2) having low inference speed at high resolution. Our work inherits the advantage of probabilistic modeling, while avoiding the drawbacks by using diffusion to generate only sparse point clouds.

Optimization-based single-view 3D leverages 2D generative priors to recover 3D from single-view images. These works [12, 20, 36, 52] rely on SDS-type loss [41, 58] and generate 3D assets by optimizing for each object image separately. These methods achieve promising results without large-scale annotation. However, the lack of a strong explicit 3D prior makes the optimization process inefficient and prone to local minima.

### 3. Method

SPAR3D Overview. Given the input image I ∈ R3×h×w, our method produces a 3D mesh with PBR materials, including albedo, metallic, roughness and surface normals. The main goal of our work is to develop a model that enjoys the benefits of distribution learning through diffusion models, while not suffering from the low output fidelity and computational inefficiency. To this end, we design a twostage model that consists of the point sampling stage and the meshing stage (see Fig. 2). At the point sampling stage, a point diffusion model learns the conditional distribution of point clouds given the input image. This stage is com-

putationally efficient given the low resolution of the point clouds. The regression-based meshing stage transforms the sampled point cloud into a highly detailed mesh that aligns with the visible surface. The reduced uncertainty with point sampling further facilitates the learning of materials and illumination in an unsupervised manner during the meshing stage. This reduces baked-in lighting artifacts and results in better modeling of specular surfaces. Finally, by using sparse point clouds as the intermediate representation, SPAR3D enables human editing in the loop.

#### 3.1. Point Sampling Stage

Overview. The point sampling stage produces a sparse point cloud as the input to the meshing stage. The core of the point sampling stage is a point diffusion model, which generates point clouds p0 ∈ Rn×6 conditioned on the input image I. The six channels include three XYZ channels and three RGB channels. In our work, the resolution of the point cloud n is set to 512.

Point Diffusion Framework. Our diffusion framework is based on DDPM [23], which consists of two processes: 1) the forward process which adds noise to the original point cloud, and 2) the backward process where the denoiser learns to remove the noise. At timestep t ∈ [0,T], the diffusion process combines Gaussian noise ϵ ∼ N(0,I) with a point cloud p0 as

pt = √α¯tp0 + √1 − α¯tϵ, (1)

where α¯t denotes the noise schedule. We use the sigmoid noise schedule proposed in [7], combined with input scaling

and the renormalization trick. The denoiser ϵθ(pt,t;c) then learns to recover the noise from pt and is supervised by

0,ϵ∥ϵ − ϵθ(pt,t;c)∥22. (2) Here c denotes the image condition tokens. During inference, we use the DDIM sampler [49] to generate point cloud

Lsimple(θ) = Et,p

samples. Samples generated directly often align poorly with the condition, hence we use the classifier-free guidance (CFG) [22] to improve sampling fidelity.

Denoiser Design. We use a transformer denoiser similar to Point-E [39], where the noisy point cloud pt ∈ Rn×6 is linearly mapped to a set of point tokens x ∈ Rn×d. We use DINOv2 [40] to encode the input image I as conditioning tokens c ∈ Rc×d. The conditions and the point tokens are then concatenated together as input to the transformer, which predicts the added noise on each point.

Albedo Point clouds. In the meshing stage, we estimate the materials and lighting alongside the geometry. However, this decomposition is inherently ambiguous because there are countless combinations of lighting and albedo that can explain the same input image. It is challenging to learn this highly uncertain decomposition during the regressive meshing stage alone. We therefore reduce the uncertainty at the point sampling stage, by directly generating albedo point clouds with diffusion models. Sampling albedo point clouds as input to the meshing stage drastically reduces the ambiguity of inverse rendering and stabilizes the decomposition learning.

#### 3.2. Meshing Stage

Overview. The meshing stage produces a textured mesh from the input image and the point cloud. The backbone of our meshing model is a large triplane transformer, which predicts triplane features from the image and point cloud conditions. We estimate the geometry, texture and lighting of the current object from the triplane, and metallic/roughness from the image features. The geometry and materials are fed into our differentiable renderer during training, so that we can apply rendering loss to supervise our model.

Triplane Transformer. Our triplane transformer consists of three submodules: the point cloud encoder, the image encoder, and the transformer backbone. We use a simple transformer encoder to encode the point cloud as a set of point tokens. Given the low resolution of the point clouds, each point can be directly mapped to a single token. Our image encoder is DINOv2 [40], which produces local image embeddings. Our triplane transformer follows a similar design to PointInfinity [26] and SF3D [2], which produces triplanes at high resolution of 384 × 384 by using a computationally-detached two-stream design.

Surface Estimation. To estimate the geometry, the triplanes are queried with a shallow MLP to produce density values. Similar to [2, 60, 65], we convert the implicit density field to explicit surface using differentiable Marching Tetrahedron (DMTet) [44]. We additionally use two MLP heads to predict vertex offsets and surface normals together

with density. These two attributes reduce the artifacts introduced by the Marching Tetrahedron and lead to locally smoother surfaces.

Material and Illumination Estimation. We perform inverse rendering and jointly estimate materials (albedo, metallic and roughness) and illumination alongside the geometry. The task is highly ill-posed and Neural-PIL [1] showed that an illumination prior can reduce the ambiguity. We build our illumination estimator upon the learningbased illumination prior from RENI++ [18]. RENI++ is originally an unconditional generative model for HDR illumination generation. We learn an encoder to map triplane features into the latent space of RENI++. This allows us to estimate the environment illumination in the input image. The albedo is estimated from triplane similar to geometry, where a shallow MLP predicts the albedo value for each 3D location. For metallic and roughness, we follow SF3D [2] and learn to estimate them with a probabilistic approach via a Beta prior. We find that the CLIP encoder used in SF3D is unstable when the object size changes. We therefore replace their CLIP encoder with AlphaCLIP [51] to alleviate this issue using foreground object masks.

Differentiable Rendering. We implement a differentiable renderer that renders images based on the predicted environment map, PBR materials and geometry surface (see Fig. 3). We use a differentiable mesh rasterizer and add a differentiable shader. Specifically, we leverage the standard simplified Disney PBR model [3] in our shader. As we use RENI++ to reconstruct environment maps, we need to explicitly integrate the incoming radiance. Here, we opt to use the Monte Carlo Integration. Given the low sample counts we can computationally afford during training, we rely on Multiple Importance Sampling (MIS) with the balanced heuristic [56] to reduce integration variance. Additionally, to better model the self-occlusion which has been typically ignored in prior works, we implement a visibility test for better shadow modeling. We take inspiration from real-time graphics and model the visibility test as a screen-space method using the depth map from our rasterizer. An overview of this test is shown in Fig. 4. Specifically, we ray-march a short distance (0.25) in 6 steps for all proposed sample directions from MIS, and project the position back to image space. If the current ray depth is farther away than the sampled value from the depth map, then the ray is marked as shadowed.

Loss Function. Our main loss function is the rendering loss that compares renderings from novel views to the grountruth (GT) images. Specifically, our rendering loss is a linear combination of 1) the L2 distance between the rendered and GT images, 2) the perceptual distance between the rendered and GT images measured by LPIPS [70], and 3) the L2 distance between the rendered opacity and the GT

[Figure 28]

Surface Point p0

[Figure 29]

Surface

|Metallic|
|---|

p1

Disney BRDF

|Roughness|
|---|

Shadowed

p2

Depth test

Input Image

GT Image

Monte Carlo MIS Sampling

[Figure 30]

Surface

|Geometry|
|---|

Rendering Loss

Sampled ray

[Figure 31]

| | | |
|---|---|---|
| | | |

|Albedo|
|---|

Visibility Test

[Figure 32]

|Env Map|
|---|

[Figure 33]

|Normal|
|---|

Camera

Triplane

Rendered

- Figure 3. Our Differentiable Renderer. We estimate geometry, albedo, lighting, and normal maps from the triplane and metallic/roughness values from the image. We rasterize and interpolate these values as input to our shader (omitted here for simplicity). Our shader uses the Disney BRDF [3] and performs Monte Carlo integration. We further perform visibility testing to improve shadow modeling. Finally, we compare the rendered image with the GT image and minimize the rendering loss.

Figure 4. Shadow Modeling. We perform visibility testing in screen-space by marching along sampled rays. If any point along the ray has a ray depth which is farther away than the depth map, we consider the entire ray as shadowed.

foreground mask. Apart from the rendering loss, we also follow SF3D and apply the mesh and shading regularization that regularizes the surface smoothness and the inverse rendering respectively.

#### 3.3. Interactive Editing

A unique advantage of our two-stage design is that it naturally supports interactive editing of unseen regions in our produced mesh. In most circumstances, the visible surface is determined by the input image and remains highly accurate, while the unseen surface is mainly based on the sampled point cloud, which might not align with user intention. In this case, editing the unseen surface of the mesh is feasible by altering the point cloud. Point clouds are perhaps one of the most flexible 3D representation for editing purposes because there are no topology constraints. Given the low resolution of our point clouds, editing the point cloud is fairly efficient and intuitive. Users can easily delete, duplicate, stretch or recolor points in the point cloud. Our efficient meshing model is able to produce the adjusted mesh in 0.3 seconds, which makes this process fairly interactive.

#### 3.4. Implementation Details

Point Sampling Stage. Our point diffusion model has 16 transformer blocks in total. Each transformer block consists of two Layer Normalization layer, one Multi-Head Attention (MHA) layer and one MLP. We use a feature dimension of 1024 and 16 attention heads in each MHA layer. With many emissive objects in our dataset, albedo can be visually distinct from the input image and hard to learn. Therefore, instead of directly generating albedo point clouds in the point sampling stage, we learn to generate white-lit point clouds as a proxy target.

Meshing Stage. Our triplane transformer consists of 4 two-stream blocks [26]. Each two-stream block consists

of three self-attentions and two cross-attentions. The main computation is carried out using 3,072 latent tokens, each with a feature dimension of 1024. The MHA includes 16 attention heads. The point cloud encoder is a vanilla transformer with 12 layers and 512 feature dimension, and the image encoder is DINOv2-large. We use a tetrahedra resolution of 160 for DMTet. In our differentiable shader, we follow Hasselgren et al. [21] and use a detached biased sampling scheme. We sample based on the specular lobe (GGX [57]), the 2D piecewise-linear distribution of the environment map luminance and the hemispherical distribution. Specifically, we include 6 samples from the GGX lobe, 6 samples from the 2D piecewise-linear distribution of the luminance, and 4 samples from the hemispherical distribution. The main body of the shader is implemented in PyTorch, while the screen-space shadowing and the 2D piecewise-linear distribution computation for the environment map are implemented as custom CUDA kernels for efficiency. The training of our meshing stage includes multiple phases, where we increase the rendering resolution and decrease the batch size at later training phases. We use GT point clouds as input when training the meshing model. The curation of our training data follows TripoSR [54].

### 4. Experiments

#### 4.1. Evaluation

Datasets. We used two datasets for evaluation, GSO [15] and OmniObject3D [63]. We follow TripoSR [54] and remove simple box or cylindrical objects to avoid bias on simple geometries. Each of the evaluation sets consists of around 250 objects. We render the objects with diverse azimuth angles at different elevations, with randomly sampled HDRI environment maps. We also vary the focal length of the camera to create more diverse test cases.

Method CD↓ FS@0.1↑ FS@0.2↑ FS@0.5↑ PSNR↑ SSIM↑ LPIPS↓ Time (s)↓ Shap-E [29] 0.204 0.359 0.638 0.922 15.3 0.802 0.205 3.1 LN3Diff [31] 0.174 0.422 0.703 0.949 17.1 0.819 0.169 5.1 LGM [53] 0.196 0.356 0.635 0.936 17.0 0.818 0.184 41.0 CRM [60] 0.161 0.437 0.735 0.961 17.5 0.830 0.169 7.4 TripoSR [54] 0.145 0.501 0.784 0.968 18.5 0.837 0.151 0.2 InstantMesh [65] 0.135 0.545 0.812 0.971 18.1 0.838 0.146 36.1

- SF3D [2] 0.137 0.540 0.806 0.970 18.0 0.839 0.145 0.3 SPAR3D (ours) 0.120 0.584 0.850 0.983 18.6 0.836 0.139 0.7

Table 1. Quantitative Comparisons on GSO [15]. SPAR3D performs favorably to other state-of-the-art methods.

Method CD↓ FS@0.1↑ FS@0.2↑ FS@0.5↑ PSNR↑ SSIM↑ LPIPS↓ Time (s)↓ Shap-E [29] 0.212 0.349 0.624 0.909 14.8 0.8006 0.205 3.1 LN3Diff [31] 0.160 0.480 0.744 0.957 16.7 0.819 0.161 5.0 LGM [53] 0.200 0.366 0.638 0.924 16.1 0.810 0.188 42.0 CRM [60] 0.155 0.482 0.765 0.962 17.0 0.828 0.162 7.0 TripoSR [54] 0.144 0.537 0.785 0.963 18.0 0.835 0.147 0.2 InstantMesh [65] 0.145 0.546 0.790 0.962 17.2 0.832 0.150 34.7

- SF3D [2] 0.138 0.554 0.800 0.967 17.4 0.836 0.145 0.3 SPAR3D (ours) 0.122 0.587 0.845 0.978 17.9 0.832 0.140 0.7

Table 2. Quantitative Comparisons on OmniObject3D [63]. SPAR3D performs favorably to other state-of-the-art methods.

Metrics. To evaluate the geometry quality of the reconstructed meshes, we use follow prior works [54, 65] and use Chamfer Distance (CD) and F-score (FS) as our evaluation metrics. CD measures the alignment between two point clouds and is defined as the average of accuracy and completeness:

- 1

- 2|S1| x∈S

- 1

- 2|S2| y∈S

∥x−y∥2+

∥x−y∥2

d(S1, S2) =

min

min

y∈S2

x∈S1

1

2

(3)

FS evaluates point cloud alignment by calculating the Fscore with a predefined threshold. Predicted points that lie within the distance threshold are considered as correct predictions. A higher FS means better alignment between the reconstructed shape and the groundtruth. To evaluate the texture quality, we compute standard image metrics, including PSNR, SSIM and LPIPS, between images rendered from the predicted mesh and the groundtruth images.

Protocol. To calculate the metrics that are comparable across methods, the meshes need to lie in the same coordinate system. To this end, we perform brute-force search in rotations to align each predicted mesh with the groundtruth mesh. Both the prediction and the groundtruth are normalized before the brute-force alignment, and the alignment is further refined with ICP.

Baselines. We compare SPAR3D with other efficient methods for single-view 3D generation or reconstruction [2, 53, 54, 60, 65]. We use the official implementation for all baselines, and we evaluate the produced meshes under the same protocol. Specifically, we compare against

TripoSR [54], LGM [53], CRM [60], InstantMesh [65], LN3Diff [31], Shap-E [29] and SF3D [2]. Among these baselines, TripoSR and SF3D are pure regression-based approaches; LGM, CRM and InstantMesh use multiview diffusion to generate pseudo multi-view images; LN3Diff and Shap-E are purely diffusion-based 3D generative models.

#### 4.2. Main Results

Quantitative Comparison. We compare SPAR3D to other baselines on GSO and Omniobject3D quantitatively. As shown in Tab. 1 and Tab. 2, SPAR3D outperforms all other regressive or generative baselines significantly across most metrics on both datasets. For SSIM, we observe that SPAR3D is slightly worse than the strongest baseline for this metric. We find that this relates to the Monte Carlo noise from our shader. SPAR3D is also among the fastest reconstruction models with an inference speed of 0.7 seconds per object, which is significantly faster than 3D or multiview diffusion-based approaches.

Qualitative Results. We show qualitative results of different methods in Fig. 5. The reconstructed meshes from pure regression-based approaches such as SF3D or TripoSR align with the input image well, but the backside is often less accurate and over-smoothed. Multi-view diffusionbased methods such as LGM, CRM and InstantMesh show more details on the backside. However, the inconsistency in the synthesized views leads to clear artifacts and overall worse results. Pure generative approaches such as Shap-E and LN3Diff are able to produce sharp surfaces in their generation. However, many details are erroneous hallucinations

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Input Image

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

Shap-E

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

LN3Diff

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

LGM

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

CRM

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

TripoSR

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Instant Mesh

[Figure 112]

[Figure 113]

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

SF3D

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

[Figure 134]

[Figure 135]

Ours

- Figure 5. Qualitative Comparison. We compare SPAR3D to other state-of-the-art methods visually. SPAR3D not only aligns better with the visible surfaces from images, but also generates higher-quality geometries and textures for the occluded surfaces.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Image Point Cloud Mesh Output Image Point Cloud Mesh Output Image Point Cloud Mesh Output

[Figure 142]

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

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

- Figure 6. Generalization Results. We show qualitative results of SPAR3D on in-the-wild images from 2D generative models (top 2 rows) and ImageNet (bottom 2 rows). The reconstructed meshes exhibit accurate geometric structures with great textures, demonstrating a strong generalization performance of SPAR3D.

that do not accurately follow the input images, and the visible surfaces are often reconstructed incorrectly. Compared to prior art, the meshes produced by SPAR3D not only faithfully resemble the input image, but also exhibit wellgenerated occluded parts with reasonable details. In Fig. 6, we further show qualitative results of SPAR3D on in-thewild images. The images are generated using text-to-image generative models or from the ImageNet validation set [13].

The high quality of the reconstructed meshes demonstrates a strong generalization performance of SPAR3D.

#### 4.3. Editing Results

The usage of explicit point clouds as an intermediate representation enables interactive editing of the generated meshes. Users can easily alter the unseen surface of the mesh by manipulating the point cloud. In Fig. 7, we show a

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

Edit

Edit

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

Edit

Edit

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

Point Cloud Mesh Output Point Cloud Mesh Output

- Figure 7. Editing Results. We show qualitative examples of interactive editing with SPAR3D. On the left two examples, we add a handle to the mug and a tail to the elephant doll by duplicating existing points. On the right two examples, we move or delete points to fix imperfections and to improve local details on the mesh. All the edits are performed in Blender within a minute.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

few editing examples with SPAR3D, either by adding major object parts to the reconstruction, or improving undesirable generated details.

#### 4.4. Ablation

Squirrel Image

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

We ablate the key idea of SPAR3D, the point sampling stage, which can be seen as an addition to standard regression approaches. We consider a variant of our model (SPAR3D w/o Point), where we remove the point sampling stage and make SPAR3D a full regressive model. We compare this variant with our full model on both GSO and Omniobject3D. As shown in Tab. 3, our full SPAR3D significantly outperforms the regressive variant, which validates the effectiveness of our design.

Horse Point Cloud

Mesh Output (6 views)

Figure 8. Generated Mesh with Conflicting Cues. Under conflicting cues from images and point clouds, our model reconstructs the visible surface based on the image, while generating the backside surface based on the point cloud.

Method CD↓ FS@0.1↑ PSNR↑ LPIPS↓ SPAR3D w/o Point 0.136 0.506 18.5 0.146 SPAR3D 0.120 0.584 18.6 0.139 SPAR3D w/o Point 0.140 0.509 17.8 0.146 SPAR3D 0.122 0.587 17.9 0.140

age. In Fig. 8, we feed the input image of a squirrel and the point cloud of a horse to the meshing model. As shown in the figure, the reconstructed mesh indeed aligns with the squirrel image well on the visible surface, while the back surface mainly adheres to the point cloud. This result validates our assumption.

Table 3. Ablation Study on GSO (top 2 rows) and Omniobject3D (bottom 2 rows). Removing the point sampling stage leads to significant performance drop.

### 5. Conclusion

#### 4.5. Analysis

We present SPAR3D, a simple yet effective approach for single-view 3D reconstruction. The core of our model is a two-stage design based on point sampling. We first generate a sparse point cloud via point diffusion, and then reconstruct a highly detailed mesh from both the point cloud and the image. This design enables us to take the best of regressionbased and generative modeling. Evaluated on standard benchmarks and in-the-wild images, SPAR3D significantly outperforms previous state-of-the-art methods with a fast inference speed. We will release our model upon publication, and we hope our effort is useful for future research towards scalable generation of high-quality 3D content.

We further design experiments to understand how SPAR3D works. Our key assumption when designing SPAR3D is that the two-stage design effectively separates the uncertain part (back-surface modeling) and the deterministic part (visible surface modeling) of the monocular 3D reconstruction problem. Ideally, the meshing stage should mainly rely on the input image for reconstructing the visible surface, while relying on the point cloud to generate the back surface. To see whether this is true, we design an experiment where we artificially use point clouds that conflict with the input im-

### References

- [1] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan T. Barron, and Hendrik P.A. Lensch. Neural-pil: Neural pre-integrated lighting for reflectance decomposition. NeurIPS, 2021. 4
- [2] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun Jampani. Sf3d: Stable fast 3d mesh reconstruction with uv-unwrapping and illumination disentanglement. arXiv preprint, 2024. 2, 4, 6, 1
- [3] Brent Burley. Physically-based shading at disney. ACM Transactions on Graphics (SIGGRAPH), 2012. 4, 5
- [4] Ruojin Cai, Guandao Yang, Hadar Averbuch-Elor, Zekun Hao, Serge Belongie, Noah Snavely, and Bharath Hariharan. Learning gradient fields for shape generation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 364–381. Springer, 2020. 3
- [5] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015. 2
- [6] Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2416–2425, 2023. 2, 3
- [7] Ting Chen. On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972, 2023. 3
- [8] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4456–4465, 2023. 2, 3
- [9] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2262–2272, 2023. 2, 3
- [10] Christopher B Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, and Silvio Savarese. 3d-r2n2: A unified approach for single and multi-view 3d object reconstruction. In European conference on computer vision, pages 628–644. Springer,

2016. 2

- [11] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. arXiv preprint arXiv:2212.08051, 2022. 2
- [12] Congyue Deng, Chiyu Jiang, Charles R Qi, Xinchen Yan, Yin Zhou, Leonidas Guibas, Dragomir Anguelov, et al. Nerdi: Single-view nerf synthesis with language-guided diffusion as general image priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20637–20647, 2023. 3

- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 7
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 3
- [15] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 5, 6
- [16] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. Advances In Neural Information Processing Systems, 35:31841–31854, 2022. 3
- [17] Lin Gao, Tong Wu, Yu-Jie Yuan, Ming-Xian Lin, Yu-Kun Lai, and Hao Zhang. Tm-net: Deep generative networks for textured meshes. ACM Transactions on Graphics (TOG), 40

(6):1–15, 2021. 3

- [18] James AD Gardner, Bernhard Egger, and William AP Smith. Reni++ a rotation-equivariant, scale-invariant, natural illumination prior. arXiv preprint arXiv:2311.09361, 2023. 4
- [19] Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan Russell, and Mathieu Aubry. AtlasNet: A Papier-Mˆach´e Approach to Learning 3D Surface Generation. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [20] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. Nerfdiff: Single-image view synthesis with nerf-guided distillation from 3d-aware diffusion. In International Conference on Machine Learning, pages 11808–11826. PMLR,

2023. 3

- [21] Jon Hasselgren, Nikolai Hofmann, and Jacob Munkberg. Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising. Adv. Neural Inform. Process. Syst. (NeurIPS), 2022. 5
- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [24] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2
- [25] Zixuan Huang, Varun Jampani, Anh Thai, Yuanzhen Li, Stefan Stojanov, and James M Rehg. Shapeclipper: Scalable 3d shape learning from single-view images via geometric and clip-based consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2023. 2

- [26] Zixuan Huang, Justin Johnson, Shoubhik Debnath, James M Rehg, and Chao-Yuan Wu. Pointinfinity: Resolutioninvariant point diffusion models. In Proceedings of the

- IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10050–10060, 2024. 2, 3, 4, 5
- [27] Zixuan Huang, Stefan Stojanov, Anh Thai, Varun Jampani, and James M Rehg. Zeroshape: Regression-based zero-shot shape reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10061–10071, 2024. 2
- [28] Le Hui, Rui Xu, Jin Xie, Jianjun Qian, and Jian Yang. Progressive point cloud deconvolution generation network. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16, pages 397–413. Springer, 2020. 3
- [29] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 2, 3, 6
- [30] Roman Klokov, Edmond Boyer, and Jakob Verbeek. Discrete point flow networks for efficient point cloud generation. In European Conference on Computer Vision, pages 694–710. Springer, 2020. 3
- [31] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. arXiv preprint arXiv:2403.12019, 2024. 2, 3, 6
- [32] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusionsdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12642–12651, 2023.
- [33] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885, 2023.
- [34] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9298–9309, 2023. 2
- [35] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 2, 3
- [36] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Realfusion: 360deg reconstruction of any object from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8446–8455, 2023. 3
- [37] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4460–4470, 2019. 2
- [38] Paritosh Mittal, Yen-Chi Cheng, Maneesh Singh, and Shubham Tulsiani. Autosdf: Shape priors for 3d completion, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 306–315, 2022. 3

- [39] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 2, 3, 4
- [40] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4
- [41] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023. 3
- [42] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10901–10911, 2021. 2
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [44] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems, 34:6087–6101,

2021. 4

- [45] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model, 2023. 2
- [46] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2, 3
- [47] Jaehyeok Shim, Changwoo Kang, and Kyungdon Joo. Diffusion-based signed distance fields for 3d shape generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20887–20897, 2023.
- [48] J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20875–20886, 2023. 2, 3
- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 3
- [50] Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Tianfan Xue, Joshua B Tenenbaum, and William T Freeman. Pix3d: Dataset and methods for single-image 3d shape modeling. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2974–2983, 2018. 2
- [51] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alpha-

- clip: A clip model focusing on wherever you want, 2023. 4
- [52] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22819–22829, 2023. 3
- [53] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint

- arXiv:2402.05054, 2024. 2, 6

[54] Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, , Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint

- arXiv:2403.02151, 2024. 2, 5, 6

- [55] Diego Valsesia, Giulia Fracastoro, and Enrico Magli. Learning localized generative models for 3d point clouds via graph convolution. In International conference on learning representations, 2018. 3
- [56] Eric Veach. Robust Monte Carlo Methods for Light Transport Simulation. PhD thesis, Stanford University, 1997. 4
- [57] Bruce Walter, Stephen R. Marschner, Hongsong Li, and Kenneth E. Torrance. Microfacet models for refraction through rough surfaces. Eurographics Symposium on Rendering,

2007. 5

- [58] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12619–12629, 2023. 3
- [59] Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh models from single rgb images. In Proceedings of the European conference on computer vision (ECCV), pages 52–67,

2018. 2

- [60] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024. 2, 4, 6
- [61] Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, and Georgia Gkioxari. Multiview compressive coding for 3d reconstruction. arXiv preprint arXiv:2301.08247, 2023. 2
- [62] Jiajun Wu, Yifan Wang, Tianfan Xue, Xingyuan Sun, Bill Freeman, and Josh Tenenbaum. Marrnet: 3d shape reconstruction via 2.5 d sketches. Advances in neural information processing systems, 30, 2017. 2
- [63] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. arXiv preprint arXiv:2301.07525, 2023. 5, 6
- [64] Zhijie Wu, Xiang Wang, Di Lin, Dani Lischinski, Daniel Cohen-Or, and Hui Huang. Sagnet: Structure-aware generative network for 3d-shape modeling. ACM Transactions on Graphics (TOG), 38(4):1–14, 2019. 3

- [65] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

2024. 2, 4, 6

- [66] Qiangeng Xu, Weiyue Wang, Duygu Ceylan, Radomir Mech, and Ulrich Neumann. Disn: Deep implicit surface network for high-quality single-view 3d reconstruction. Advances in neural information processing systems, 32, 2019. 2
- [67] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. Pointflow: 3d point cloud generation with continuous normalizing flows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4541–4550, 2019. 3
- [68] Lior Yariv, Omri Puny, Oran Gafni, and Yaron Lipman. Mosaic-sdf for 3d generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4630–4639, 2024. 2, 3
- [69] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. arXiv preprint arXiv:2404.19702, 2024. 2
- [70] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 4
- [71] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, BIN FU, Tao Chen, Gang YU, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. In Thirtyseventh Conference on Neural Information Processing Systems, 2023. 2, 3

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

Input Image SF3D Albedo SPAR3D Albedo SF3D Relight SPAR3D Relight

- Figure 9. Decomposition and Relighting Results. We show decomposed albedo and relighting results of SPAR3D in comparison with SF3D. The albedo estimated by SPAR3D has less baked-in lighting compared with SF3D and results in better relighting outcomes.

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Input Image Mesh Output Input Image Mesh Output Input Image Mesh Output

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

- Figure 10. Additional In-the-wild Results. We show additional results of SPAR3D on in-the-wild images. The reconstructed meshes achieve high fidelity and exhibit great surface details.

This appendix is structured as follows: in Appendix A we discuss the limitations of our approach; in Appendix B we provide two additional illustrations of our architecture; in Appendix C we show decomposition and relighting results of our model in comparison with SF3D [2]; in Appendix D we present additional in-the-wild results.

### A. Limitations

The main limitations of SPAR3D are twofold. First, the point clouds generated during the point sampling stage occasionally exhibit artifacts, such as small surface spikes or detached parts. While these imperfections can typically be remedied through SPAR3D’s editing capabilities with minimal effort (see Fig. 7 in the main paper), exploring more principled solutions (e.g. improving the denoiser design or diffusion samplers) could further enhance the utility and robustness of our method.

Second, although SPAR3D learns material decomposi-

tion during training, the accuracy of these decompositions can sometimes be suboptimal. This limitation is primarily due to the inherent ambiguity of inverse rendering from a single image, especially when learned in an unsupervised manner. Unsupervised decomposition learning is useful given the scarcity of 3D assets containing high-quality Physically Based Rendering (PBR) materials and is scalable to real-world multi-view datasets. However, investigating semi-supervised learning techniques may offer a pathway to more plausible material estimations in future work.

### B. Additional Illustrations of our Architecture

We show additional illustrations of our point cloud denoiser and our meshing model in Fig. 11 and Fig. 12. We hope these illustrations facilitate a better understanding of our architecture.

[Figure 290]

[Figure 291]

DINOv2 Encoder

[Figure 292]

Transformer Blocks

Concat

Decode

Tokenize

- Figure 11. Point Cloud Denoiser Architecture. We illustrate the architecture of our point cloud denoiser. The point cloud denoiser takes the noisy point cloud and the image as input, and produces a denoised point cloud. The image and the noisy point cloud are encoded as latent vectors and concatenated together. The concatenated latent vectors are processed by a set of transformer blocks and decoded as the denoised point cloud.

[Figure 293]

[Figure 294]

DINOv2 Encoder

Tokenize

Triplane Transformer

| | |
|---|---|
| | |

Triplane Tokens

[Figure 295]

[Figure 296]

[Figure 297]

Detokenizer

Triplane

Illumimation Encoder

RENI++ Decoder Lighting

Albedo Density Deform Normal

DMTet Geometry

Texture

- Figure 12. Meshing Model Architecture. We illustrate the architecture of our meshing model, which takes the point cloud and the image as input, and produces a textured mesh and an environment map as output. Specifically, the meshing model first encodes the image and the point cloud as latent vectors. The learnable triplane tokens are then processed by the triplane transformer conditioned on the latent vectors. We query the triplane with MLPs to obtain albedo, density, vertex deformation and surface normal, which are converted to a textured mesh using DMTet. The triplane also produces an environment map using the illumination prior from RENI++. The metallic and roughness values are estimated from the image directly and are omitted here for simplicity.

### C. Decomposition Results

We show decomposition and relighting results of SPAR3D in comparison with SF3D, which is a full regressive method. As shown in Fig. 9, our estimated albedo often has less baked-in lighting artifacts compared with SF3D, which improves the quality of relighting under different illumination conditions.

### D. Additional In-the-wild Results

We present additional reconstruction results on in-the-wild images. In Fig. 10, we show the reconstructions of SPAR3D on images from 3D-Arena (Ebert, 2024). On this data source, SPAR3D also achieves high reconstruction quality. This further validates the strong generalization ability of SPAR3D.

