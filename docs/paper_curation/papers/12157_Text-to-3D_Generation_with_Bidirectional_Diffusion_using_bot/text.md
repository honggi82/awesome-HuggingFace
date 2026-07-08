## Text-to-3D Generation with Bidirectional Diffusion using both 2D and 3D priors

# arXiv:2312.04963v1[cs.CV]7Dec2023

Lihe Ding1,4*, Shaocong Dong2*, Zhanpeng Huang3, Zibin Wang3†, Yiyuan Zhang1, Kaixiong Gong1, Dan Xu2, Tianfan Xue1 1The Chinese University of Hong Kong 2Hong Kong University of Science and Technology 3SenseTime 4Shanghai AI Laboratory

{dl023, gk023, tfxue}@ie.cuhk.edu.hk, {sdongae, danxu}@cse.ust.hk {wangzb02, yiyuanzhang.ai}@gmail.com, {huangzhanpeng}@sensetime.com

A yellow and green oil painting style eagle head

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

(a) Shap-E

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

(b) Zero-123 (c) ProlificDreamer

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

≈ 20min

### ≈40s

Bidirectional Diffusion: Generalizable 3D Generation Efficiently Refine by Optimization Methods

Figure 1. Our BiDiff can efficiently generate high-quality 3D objects. It alleviates all these issues in previous 3D generative models: (a) low-texture quality, (b) multi-view inconsistency, and (c) geometric incorrectness (e.g., multi-face Janus problem). The outputs of our model can be further combined with optimization-based methods (e.g., ProlificDreamer) to generate better 3D geometries with slightly longer processing time (bottom row).

#### Abstract

Most 3D generation research focuses on up-projecting 2D foundation models into the 3D space, either by minimizing 2D Score Distillation Sampling (SDS) loss or fine-tuning on multi-view datasets. Without explicit 3D priors, these methods often lead to geometric anomalies and multi-view

*Equal contribution. Part of this work was done when Lihe Ding and Shaocong Dong interned at Sensetime.

†Corresponding author.

inconsistency. Recently, researchers have attempted to improve the genuineness of 3D objects by directly training on 3D datasets, albeit at the cost of low-quality texture generation due to the limited texture diversity in 3D datasets. To harness the advantages of both approaches, we propose Bidirectional Diffusion (BiDiff), a unified framework that incorporates both a 3D and a 2D diffusion process, to preserve both 3D fidelity and 2D texture richness, respectively. Moreover, as a simple combination may yield inconsistent generation results, we further bridge them with novel bidi-

rectional guidance. In addition, our method can be used as an initialization of optimization-based models to further improve the quality of 3D model and efficiency of optimization, reducing the generation process from 3.4 hours to 20 minutes. Experimental results have shown that our model achieves high-quality, diverse, and scalable 3D generation. Project website: https://bidiff.github.io/.

#### 1. Introduction

Recent advancements in text-to-3D generation [22] mainly focus on lifting 2D foundation models into 3D space. One of the most popular solutions [17, 27] uses 2D Score Distillation Sampling (SDS) loss derived from a 2D diffusion model to supervise 3D generation. While these methods can generate high-quality textures, they often lead to geometric ambiguity, such as the multi-face Janus problem [23], due to the lack of 3D constraints (Fig. 1(c)). Moreover, these optimization methods are time-consuming, taking hours to generate one object. Zero-123[18] tries to alleviate the problem by fine-tuning the 2D diffusion models on multi-view datasets, but it still cannot guarantee geometric consistency (Fig. 1(b)).

To ensure better 3D consistency, another solution is to directly learn 3D structures from 3D datasets [14, 25]. However, many existing 3D datasets [2, 5] only contain handcrafted objects or lack high-quality 3D geometries, with textures very different from real-world objects. Moreover, 3D datasets are often much smaller than, and also difficult to scale up to, their 2D counterparts. As a result, the 3D diffusion models (Fig. 1 (a)) normally cannot generate detailed textures and complicated geometry, even if they have better 3D consistency compared to up-projecting 2D diffusion models.

Therefore, a straightforward way to leverage the advantages of both methods is to combine both 2D and 3D diffusion models. However, a simple combination may result in inconsistent generative directions as they are learned in two independent diffusion processes. In addition, the two diffusion models are represented in separate 2D and 3D spaces without knowledge sharing.

To overcome these problems, we propose Bidirectional Diffusion (BiDiff), a method to seamlessly integrate both 2D and 3D diffusion models within a unified framework. Specifically, we employ a hybrid representation in which a signed distance field (SDF) is used for 3D feature learning and multi-view images for 2D feature learning. The two representations are mutually transformable by rendering 3D feature volume into 2D features and back-projecting 2D features to 3D feature volume. Starting from pretrained 3D and 2D diffusion models, the two diffusion models are jointly finetuned to capture a joint 2D and 3D prior facilitating 3D generation.

However, correlating the 2D and 3D representations is not enough to combine two diffusion processes, as they may deviate from each other in the following diffusion steps. To solve this problem, we further introduce bidirectional guidance to align the generative directions of the two diffusion models. At each diffusion step, the intermediate results from the 3D diffusion scheme are rendered into 2D images as guidance signals to the 2D diffusion model. Meanwhile, the multi-view intermediate results from the 2D diffusion process are also back-projected to 3D, guiding the 3D diffusion. The mutual guidance regularizes the two diffusion processes to learn in the same direction.

The proposed bidirectional diffusion poses several advantages over the previous 3D generation models. First, users can separately control the generation of 2D texture and 3D geometry, as shown in Fig. 2, because the 2D diffusion model focuses on texture generation and the 3D diffusion model focuses on geometry. This is impossible for previous 3D diffusion methods. Secondly, compared to 3D-only diffusion models [14], our method takes advantage of a 2D diffusion model trained on much larger datasets. Therefore, it can generate more diversified objects and create a completely new object like “A strong muscular chicken” illustrated in Fig 2. Thirdly, compared to previous optimization methods [27, 37] that often take several hours to generate one object, we utilize a fast feed-forward joint 2D-3D diffusion model for scalable generation, which only takes about 40 seconds to generate one object.

Moreover, because of the efficacy of BiDiff, we also propose an optional step to utilize its output as an initialization for the existing optimization-based methods (e.g., ProlificDreamer [37]). This optional step can further improve the quality of a 3D object, as demonstrated in the bottom row of Fig. 1. Also, the good initialization from BiDiff helps to reduce optimization time from around 3.4 hours to 20 minutes, and concurrently resolves geometrical inaccuracy issues, like multi-face anomalies. Moreover, this two-step generation enables creators to rapidly adjust prompts to obtain a satisfactory preliminary 3D model through a lightweight feed-forward generation process, subsequently refining it into high-fidelity results.

Through training on ShapeNet [2] and Objaverse 40K [5], our framework is shown to generate high-quality textured 3D objects with strong generalizability. In summary, our contributions are as follows: 1) We propose BiDiff, a joint 2D-3D diffusion model, that can generate high-quality, 3Dconsistent, and diversified 3D objects; 2) We propose a novel training pipeline that utilizes both pretrained 2D and 3D generative foundation models; 3) We propose the first diffusionbased 3D generation model that allows independent control of texture and geometry; 4) We utilize the outputs from BiDiff as a strong initialization for the optimization-based methods, generating high-quality geometries while ensuring

###### An ancient Gothic tower.

A strong muscular man.

An ancient Chinese tower.

A strong muscular chicken.

A golden skull. A crystal skull.

[Figure 20]

[Figure 21]

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

###### A bule and red Superman clothes style car.

A house in Van Gogh starry sky style.

A blue and red Superman clothes.

A board with Van Gogh‘s starry sky style painting on it.

A blue and white porcelain burger.

A blue and white porcelain teapot.

- Figure 2. Texture Control (Top): we change the texture while maintaining the overall shape. Shape Control (Bottom): we fix texture patterns and generate various shapes. that users receive quick feedback for each prompt update.

spaces [4, 19]. The synthesized multi-view images rendered from 3D datasets were also utilized to provide cross-view 3D consistent knowledge [18]. These methods normally highlight fast inference and 3D consistent results. However, due to inferior 3D dataset quality and size, these methods generally yield visually lower-quality results with limited diversity. Recently a few methods [28, 33] explored to combine 2D priors and 3D priors from individual pre-trained diffusion models, but they often suffer from inconsistent between two generative processes.

#### 2. Related Work

Early 3D generative methods adopt various 3D representations, including 3D voxels [10, 34, 38], point clouds [1, 40], meshes [9, 12], and implicit functions [3, 26] for categorylevel 3D generations. These methods directly train the generative model on a small-scale 3D dataset, and, as a result, the generated objects may either miss tiny geometric structures or lose diversity. Even though there are large-scale [5] or high-quality 3D datasets [39] in recent years, they are still much smaller than the datasets used for 2D image generation training.

#### 3. Method

As many previous studies [18, 28] have illustrated, both 2D texture and 3D geometry are important for 3D object generation. However, incorporating 3D structural priors and 2D textural priors is challenging: i) combining both 3D and

With the powerful text-to-image synthesis models [29– 31], a new paradigm emerges for 3D generation without large-scale 3D datasets by leveraging 2D generative model. One line of works utilizes 2D priors from pre-trained textto-image model (known as CLIP) [13, 15] or 2D diffusion generative models [17, 22, 35] to guide the optimization of underlying 3D representations. However, these models could not guarantee cross-view 3D consistency and the per-instance optimization scheme suffers both high computational cost and over-saturated problems. Later on, researchers improve these models using textual codes or depth maps [6, 21, 32], and [37] directly model 3D distribution to improve diversity. These methods alleviate the visual artifacts but still cannot guarantee high-quality 3D results.

- 2D generative models into a single cohesive framework is not trivial; ii) in both training and inference, two generative models may lead to opposite generative directions.

To tackle these problems, we propose BiDiff, a novel bidirectional diffusion model that marries a pretrained 3D diffusion model with another 2D one using bidirectional guidance. Fig. 3 illustrates the overall architecture of our framework. Details of each component will be discussed below. Specifically, in Sec. 3.1, we will introduce our novel hybrid representation that includes both 2D and 3D information, and the bidirectional diffusion model built on top of this hybrid representation. In Sec. 3.2 and Sec. 3.3, to ensure the two generative models lead to the same generative direction, we will introduce how to add bidirectional guidance to both

- 3D and 2D diffusion models. In Sec. 3.4, we discuss one advantage of BiDiff, which is independent control of texture

Another line of works learn 3D priors directly from 3D datasets. As the diffusion model has been the de-facto network backbone for most recent generative models, it has been adapted to learn 3D priors using implicit spaces such as point cloud features [25, 42], NeRF parameters [7, 14], or SDF

- 2D Noise

Distillation

SDS loss

Optimization

step t+1 step t step t-1 step 0

- 3D Noise

3DPipeline (b)

###### (a)

3D Denoising

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Noisy Input

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

SDF Prediction

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- 2D-3D Control

[Figure 86]

[Figure 87]

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

[Figure 100]

[Figure 101]

[Figure 102]

- 3D-2D Control

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

3D Foundation Model

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

3D SDF (t step)

3D SDF (t-1 step)

Feature Volume

Volume Rendering

[Figure 111]

[Figure 112]

Recon loss

2D Pipeline

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

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Volume Encoding

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Feature Volume

Noisy Input

[Figure 142]

2D Denoising

Multi-view images (t step)

Multi-view images (t-1 step)

- Figure 3. The BiDiff framework operates as follows: (a) At each step of diffusion, we render the 3D diffusion’s intermediate outputs into

- 2D images, which then guide the denoising of the 2D diffusion model. Simultaneously, the intermediate multi-view outputs from the 2D diffusion are re-projected to assist the denoising of the 3D diffusion model. Red arrows show the bidirectional guidance, which ensures that both diffusion processes evolve coherently. (b) We use the outcomes of the 2D-3D diffusion as a strong starting point for optimization methods, allowing for further refinement with fewer optimization steps.

and geometry generation, as shown in Fig. 2. Finally, in Sec. 3.5, we discuss another advantage of BiDiff, which is to use the results from BiDiff as a strong initialization for optimization-based methods to obtain more delicate results efficiently.

##### 3.1. Bidirectional Diffusion

To incorporate both 2D and 3D priors, we represent a 3D object using a hybrid combination of two formats: Signed Distance Field (SDF) F and multi-view image set V =

Ii Mi=1, where F is computed from signed distance values on an N × N × N grid, and Ii is the i-th image from a multi-view image set of size M. This hybrid representation is shown on the left side of Fig. 3.

With this representation, we learn a joint distribution {F,V} utilizing two distinct diffusion models: a 3D diffusion model D3d in the SDF space (the green 3D denoising block in Fig. 3) and a 2D multi-view diffusion model D2d within the image domain (the blue 2D denoising block in Fig. 3). Specifically, given a timestep t, we add Gaussian noises to both SDF and multi-view images as

Ft = √αtF0 + √1 − αtϵ3d and Iti = √αtI0i + √1 − αtϵi2d for ∀i,

(1)

where ϵ ∼ N(0,I) is random noise, and αt is a noise schedule which is different in 3D and 2D. Subsequently, the straightforward way is to separately train these two diffusion models by minimizing the following two objectives:

t,ϵ3d,t∥ϵ3d − D3d(Ft,t)∥22,

Lsimple3d = EF

N

1 N

t,ϵi2d,t∥ϵi2d − D2d(Iti,t)∥22),

(EIi

Lsimple2d =

i=1

(2)

where ϵ3d and ϵ2d are Gaussian noises ϵ3d,ϵi2d ∼ N(0,I), SDF and image set are sampled from forward diffusion pro-

cesses Ft ∼ q(Ft),Iti ∼ q(Iti), and timestep is uniformly sampled t ∼ U[1,T].

However, this simple combination does not consider the correlations between 3D and 2D diffusion, which may hinder the understanding of 2D and 3D consistency, leading to inconsistent generation between 3D geometry and 2D multiview images.

We resolve this problem by a novel Bidirectional Diffusion. In this model, the consistency between 3D and 2D diffusion output is enforced through bidirectional guidance. First, we add guidance from the 2D diffusion process to the 3D generative process, which is the red arrow pointing to the “2D-3D control”. Specifically, during each denoising step t, we feed the denoised multi-view images

N i=1 in previous step into the 3D diffusion model as ϵ′3d = D3d(Ft,Vt′+1,t). This guidance steers the current 3D denoising direction to ensure 2D-3D con-

Vt′+1 = Iti+1

sistency. It’s worth mentioning that the denoised output Vt′+1 from the previous step t + 1 is inaccessible in training, therefore we directly substitute it with the ground truth Vt. In inference, we utilize the denoised images from the previous step. Then we could obtain the denoised radiance field F0′ given the 2D guided noise prediction ϵ′3d by F0′ = √1α

√1 − αtϵ′3d).

(Ft −

t

Secondly, we also add guidance from the 3D diffusion process to the 2D generative process. Specifically, using the same camera poses, we render multi-view images Hti derived from the radiance field F0′ by the 3D diffusion model: Hti = R(F0′,Pi),i = 1,...M, where Pi is the ith camera pose. These images are further used as guidance to the 2D multi-view denoising process D2d by ϵ′2d = D2d(Vt, Hti

N i=1 ,t).. This guidance is the red arrow

pointing to the “3D-2D control” in Fig. 3.

Our method can seamlessly integrate and synchronize both the 3D and 2D diffusion processes within a unified framework. In the following sections, we will delve into each component in detail.

##### 3.2. 3D Diffusion Model with 2D Guidance

Our 3D diffusion model aims to generate a neural surface field (NeuS) [20], with novel 2D-to-3D guidance derived from the denoised 2D multi-view images. To train our 3D diffusion model, at each training timestep t, we add noise to a clean radiance field, yielding a noisy one Ft. This field, combined with the timestep t embeddings and the text embeddings, is then passed through 3D sparse convolutions to generate a 3D feature volume M as: M = Sp3DConv(Ft,t,text). Then we sample N × N × N grid points from M and project these points onto all denoised multi-view images Vt′+1 from the previous step of the 2D diffusion model. At each grid point p, we aggregate the interpolated 2D feature at its 2D projected location on each view, and calculate the mean and variance over all N interpolated features to obtain the image-conditioned feature volume N:

N(p) = [Mean(Vt′+1(π(p))),Var(Vt′+1(π(p)))], (3)

where π denotes the projection operation from 3D to 2D image plane. We fuse these two feature volumes with further sparse convolutions for predicting the clean F0.

One important design of our 3D diffusion model is that it incorporates geometry priors derived from the 3D foundation model, Shap-E [14]. Shap-E is a latent diffusion [22] model trained on several millions 3D objects, and thus ensures the genuineness of generated 3D objects. Still, we do not want Shap-E to limit the creativity of our 3D generative model, and try to preserve the capability of generating novel objects that Shap-E cannot.

To achieve this target, we design a feature volume G to represent a radiance field converted from the Shap-E latent code C. It is implemented using NeRF MLPs by setting their parameters to the latent code C: G(p) = MLP(λ(p);θ = C), where λ denotes the positional encoding operation.

One limitation of directly introducing Shap-E latent code is that the network is prone to shortcut the training process, effectively memorizing the radiance field derived from ShapE. To generate 3D objects beyond Shap-E model, we add Gaussian noise at level t0 to the clean latent code, resulting in

the noisy latent representation Ct0

, where t0 represents a predefined constant timestep. Subsequently, the noisy radiance field Gt0

. This design establishes a coarse-to-fine relationship between the 3D prior and the ground truth, prompting the 3D diffusion process to leverage the 3D prior without excessively depending on it. In this way, we can get the fused feature volume as:

is decoded by substituting C with Ct0

S = U([M,Sp3DConv(N),Sp3DConv(Gt0

)]), (4)

where U denotes 3D sparse U-Net. Then we can query features from S for each grid point p and decode it to SDF values through several MLPs: F0′(p) = MLP(S(p),λ(p)),

- where S(p) represents the interpolated features from S at position p. In Sec. 4.2 and Fig. 4, our experiments also demonstrate that our model can generate 3D objects beyond Shap-E model.

3.3. 2D Diffusion Model with 3D Guidance

Our 2D diffusion model simultaneously generates multiview images by jointly denoising multi-view noisy images

Vt = Iti

M i=1. To encourage 2D-3D consistency, the 2D diffusion model is also guided by the 3D radiance field output from 3D diffusion process mentioned above. Specifically, for better image quality, 2D multi-view diffusion model is built on the multiple independently frozen 2D foundation models (e.g., DeepFloyd [8]) to harness the potent 2D priors. Each of these frozen 2D foundation models (the dark blue network in Fig. 3) is modulated by view-specific 3D-consistent residual features and responsible for the denoising of a specific view, as described below.

First, to achieve 3D-to-2D guidance, we render multiview images from the 3D denoised radiance field F0′ and feed them to 2D denoising model. Note that the radiance field consists of a density field and a color field. The density field is constructed from the signed distance field (SDF) generated by our 3D diffusion model using S-density introduced in NeuS [36]. To obtain the color field, we apply another color MLP to the feature volume in the 3D diffusion process.

Upon obtaining the color field c and density field σ, we conduct volumetric rendering on each ray r(m) = o + md which extends from the camera origin o along a direction d

to produce multi-view consistent images Hi Mi=1:

Cˆ(r) =

∞

0

T(m)σ(r(m)))c(r(m)),d)dm, (5)

- where T(m) = exp(− 0 m σ(r(s))ds) handles occlusion. Secondly, we use these rendered multi-view images as

guidance for the 2D foundation model. We first use a shared feature extractor E to extract hierarchical multi-view consistent features from these images. Then each extracted feature is added as residuals to the decoder of its corresponding frozen 2D foundation denoising U-Net (the red arrow pointing to “3D-2D Control” in Fig. 3), achieving multi-view

modulation and joint denoising following ControlNet [43] as fˆik = fik + ZeroConv(E(Hi)[k]), where fki denotes the original feature maps of the k-th decoder layer in 2D foundation model, E(Hi)[k] denotes the k-th residual features of the i-th view, and ZeroConv [43] is 1 × 1 convolution which is initialized by zeros and gradually updated during training. Experimental results show that this 3D-to-2D guidance helps to ensure multi-view consistency and facilitate geometry understanding.

##### 3.4. Separate Control of Geometry and Texture

One advantage of BiDiff is that it naturally separates 2D texture generation using 2D diffusion model from 3D geometry generation using 3D diffusion model. Because of this, users can separately control geometry and texture generation, as shown in Fig. 2.

To achieve this, we first propose a prior enhancement strategy to empower a manual control of the strength of 3D and 2D priors independently. Inspired by the classifier-free guidance [11], during training, we randomly drop the information from 3D priors by setting condition feature volume from G to zero and weaken the 2D priors by using empty text prompts. Consequently, upon completing the training, we can employ two guidance scales, γ3d and γ2d, to independently modulate the influence of these two priors.

Specifically, to adjust the strength of 3D prior, we calculate the difference between 3D diffusion outputs with and without conditional 3D feature volumes, and add them back to 3D diffusion output:

ϵˆ3d =D3d(Ft,Vt′+1,t) + γ3d · ((D3d(Ft,Vt′+1,t|G)− D3d(Ft,Vt′+1,t)).

(6)

Then we can control the strength of 3D prior by adjusting the weight γ3d of this difference term. When γ3d = 0, it will completely ignore 3D prior. When γ3d = 1, this is just the previous model that uses both 3D prior and 2D prior. When γ3d > 1, the model will produce geometries close to the conditional radiance field but with less diversity.

Also, we can similarly adjust the strength of 2D priors by adding differences between 2D diffusion outputs with and without conditional 2D text input:

M i=1 ,t)+

ϵˆ2d =D2d(Vt, Hti

M i=1 ,t|text))−

(7)

γ2d · ((D2d(Vt, Hti

M i=1 ,t)).

D2d(Vt, Hti

Increasing γ2d results in more coherent textures with text, albeit at the expense of diversity. It is worth noting that while we adjust the 3D and 2D priors independently via Eq. (6) and Eq. (7), the influence inherently propagates to the other domain due to the intertwined nature of our bidirectional diffusion process.

With these two guidance scales γ3d and γ2d, we can easily achieve a separate control of geometry and texture. First, to only change texture while keep geometry untouched, we just fix the initial 3D noisy SDF grids and the conditional radiance field Ct0

, while enlarge its influence by Eq. (7). On the other hand, to only change geometry while keep texture style untouched, we can maintain keywords in text prompts and enlarge its influence by Eq. (6). By doing so, the shape will be adjusted by the 3D diffusion process.

##### 3.5. Optimization with BiDiff Initialization

The generated radiance field F0 using BiDiff can be further used as a strong initialization of the optimization-based methods [37]. This additional step can further improve the quality of the 3D model, as shown in Fig. 1 and Fig. 5. Importantly, compared to the geometries directly generated by optimization, our BiDiff can output more diversified geometry and generated geometries better aligns with users’ input text, and also has more accurate 3D geometry. Therefore, the optimization started from this strong initialization can be rather efficient (≈ 20min) and avoid incorrect geometries like multi-face and floaters.

Specifically, we first convert generated radiance field F0 from BiDiff into a higher resolution one F0 that supports 512 × 512 resolution image rendering, as shown on the right of Fig. 3. This process is achieved by a fast NeRF distillation operation (≈ 2min). The distillation first bounds the occupancy grids of F0 with the estimated binary grids (transmittance > 0.01) from the original radiance field F0, then overfits F0 to F0 by minimizing both the L1 distance between two density fields and L1 distance between their renderings 2D images under random viewpoints. Thanks to this flexible and fast distillation operation, we can efficiently convert generated radiance field from BiDiff into any 3D representations an optimization-based method requires. In our experiments, since we are using ProlificDreamer [37], we use the InstantNGP [24] as the high-resolution radiance field.

After initialization, we optimize F0 by SDS loss following the previous methods [27, 37]. It is noteworthy that since we already have a good initialized radiance field, we only need to apply a small noise level SDS loss. Specifically, we set the ratio range of denoise timestep topt to [0.02, 0.5] during the entire optimization process.

#### 4. Experiment

In this section, we described our experimental results. We train our framework on the ShapeNet-Chair [2] and Objaverse LVIS 40k datasets [5]. We use the pre-trained DeepFloyd-IF-XL [8] as our 2D foundation model and ShapE [14] as our 3D priors. We adopt the SparseNeuS [20] as the neural surface field presentation with N = 128. For the 3D-to-2D guidance, We follow the setup of ControlNet [43]

A chair made of Minecraft bedrock blocks. The blocks should be seamlessly integrated into the chair's structure.

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

A chair designed in the shape of a cactus, with prickly spines and a green, desert-inspired texture.

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

A beautiful dress made out of fruit, on a mannequin.

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

A red Volkswagen Beetle car.

[Figure 181]

[Figure 182]

Ours Shap-E

- Figure 4. Qualitative sampling results of Bidirectional Diffusion model, including multi-view images and 3D mesh from diffusion sampling. The top two rows are the results on the Shapenet-Chair, and the bottom two rows are the results on the Objaverse. We compared the results of Shap-E in the last column.

Table 1. CLIP R-precision.

Decouple geometry and texture control. Lastly, we illustrate that our BiDiff can separately control geometry and texture generation. First, as illustrated in the first row of Fig. 2, when the 3D prior is fixed, we have the flexibility to manipulate the 2D diffusion model using varying textual prompts to guide the texture generation process. This capability enables the generation of a diverse range of textured objects while maintaining a consistent overall shape. Second, when we fix the textual prompt for the 2D priors (e.g., "a xxx with Van Gogh starry sky style"), we can adjust the 3D diffusion model by varying the conditional radiance field derived from the 3D priors. This procedure results in the generation of a variety of shapes, while maintaining a similar texture, as shown in the second row of Fig. 2.

to render M = 8 multi-view images with 64 × 64 resolution using SparseNeuS. We train our framework on 4 NVIDIA A100 GPUs for both ShapeNet and Objaverse 40k experiments with batch size of 4. During sampling, we set the

Method R-P time DreamFusion 0.67 1.1h

ProlificDreamer 0.83 3.4h Ours-sampling 0.79 40s

- 3D and 2D prior guidance scale to 3.0 and 7.5 respectively. More details on data processing and model architecture are included in supplementary material. We discuss the evaluation and ablation study results below. Also, please refer to supplementary webpages and videos for more visual results.

Ours-post 0.85 20min

##### 4.1. Text-to-3D Results

ShapeNet-Chair results. The first and second rows of Fig. 4 present our results trained on the ShapeNet-Chair dataset. Although the chair category often contains complicated geometric details, our framework demonstrates the capability to capture those fine details. Concurrently, our approach exhibits a remarkable capability to produce rich and diverse textures by merely modulating the textual prompts, leading to compelling visual outcomes.

##### 4.2. Comparison with other Generation Models

Objaverse-40K results. Scaling to a much larger 3D dataset, Objaverse-40K, our framework’s efficacy becomes increasingly pronounced. The bottom two rows of Fig. 4 are results from the Objaverse dataset. Compared to objects generated by Shap-E, our model closely adheres to the given textual prompts. This again shows that the proposed BiDiff learns to model both 2D textures and 3D geometries better compared with 3D-only solutions, and is capable of generating more diverse geometries.

Comparison with optimization methods. Our framework is capable of simultaneously generating multi-view consistent images alongside a 3D mesh in a scalable manner. In contrast, the SDS-based methods [27, 37] utilize a oneby-one optimization approach. Tab. 1 reports the CLIP R-Precision [14] and inference time on 50 test prompts manually derived from the captioned untrained Objaverse to quantitatively evaluate these methods. Also, optimization methods, Dreamfusion [27] and ProlificDreamer [37],

[Figure 183]

A golden skull

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Input(by SD 2.1) 90° -150°

Zero-1-to-3 PolificDreamer

time ≈ 6h

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

Bidirectional Diffusion Results Refinement Results

time ≈ 20min

- Figure 5. Comparison with other optimization or multi-view diffusion based works. We show both multi-view images (left) and 3D results (right). Zero-1-to-3 [18] is not good at predicting results from a large perspective, and PolificDreamer [37] suffers from the multi face problem. Our method has excellent robustness and can obtain high-quality results in a short period of time.

###### An oral cavity style chair, includes oral cavity elements like red tongue and white teeth.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

w/o

- 2D prior

w/o

- 3D prior

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

w/o prior enhancement

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

normal

Figure 6. Ablation of prior and prior enhancement.

are expensive, taking several hours to generate a single object. Moreover, these optimization methods may lead to more severe multi-face problems. In contrast, our method can produce realistic objects with reasonable geometry in only 40 seconds. Furthermore, BiDiff can serve as a strong prior for optimization-based methods and significantly boost their performance. Initializing the radiance field in ProlificDreamer [37] with our outputs shows remarkable improvements in both quality and computational efficiency, as shown in Fig. 5.

Comparison with multi-view methods Given one reference image, the multi-view method Zero-1-to-3 [18] pro-

[Figure 220]

[Figure 221]

[Figure 222]

Diffusion Output ~  ( .  ,  .  ) ~  ( .  ,  . )

Figure 7. Ablation of range of noise level t for SDS.

duces images from novel viewpoints by fine-tuning a pretrained 2D diffusion model on multi-view datasets. However, this method employs cross-view attention to establish multiview correspondence without an inherent understanding of 3D structures, inevitably leading to inconsistent multi-view images as shown in Fig. 5. Moreover, the Zero-123 series cannot directly generate the 3D mesh, requiring substantial post-processing (SDS loss) to acquire the geometry. In contrast, our framework also incorporates 3D priors, in addition to 2D priors, and thus can generate more accurate 3D geometries.

##### 4.3. Abalation Studies

We perform comprehensive ablation studies on the ShapeNetChair dataset [2] to evaluate the importance of each component below. More ablation results can be found in the supplementary material.

3D priors. To assess the impact of 3D priors, we eliminate the conditional radiance field from Shap-E and train the 3D geometry generation from scratch. The experimental results in the second row of Fig. 6 demonstrate that in the absence of the 3D priors, our framework can only generate common

objects in the training set.

- 2D priors. To delve into the impact of 2D priors, we randomly initiate the parameters of the 2D diffusion model, instead of fine-tuning on a pretrained model. The results in the first row of Fig. 6 show that in the absence of 2D priors, the textures generated tend to fit the stylistic attributes of the synthetic training data. Conversely, with 2D priors, we can produce more realistic textures. Prior enhancement strategy. As discussed in Sec. 3.4, we can adjust the influence of both 3D and 2D priors by the prior enhancement strategy. Fig. 6 also shows the results of not using this strategy. It shows that the prior enhancement strategy plays a vital role in achieving diverse and flexible
- 3D generation. Range of noise level for SDS. The results in Fig. 7 illustrate the impact of the noise level during the entire optimization process, as discussed in Sec. 3.5. The 3D object generated with a smaller noise range is closer to the diffusion

output. By adjusting the range of the noise level topt, we can effectively control the texture similarity between geometries before and after the optimization.

#### 5. Conclusion

In this paper, we propose Bidirectional Diffusion, which incorporates both 3D and 2D diffusion processes into a unified framework. Furthermore, Bidirectional Diffusion leverages the robust priors from 3D and 2D foundation models, achieving generalizable geometry and texture understanding.

#### References

- [1] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, and Leonidas Guibas. Learning representations and generative models for 3d point clouds. In International conference on machine learning, pages 40–49. PMLR, 2018. 3
- [2] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An informationrich 3d model repository. arXiv preprint arXiv:1512.03012,

2015. 2, 6, 8, 1

- [3] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 3
- [4] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tuyakov, Alex Schwing, and Liangyan Gui. SDFusion: Multimodal 3d shape completion, reconstruction, and generation. arXiv, 2022. 3
- [5] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. arXiv preprint arXiv:2212.08051,

2022. 2, 3, 6, 1

- [6] C. Deng, C. Jiang, C. R. Qi, X. Yan, Y. Zhou, L. Guibas, and D. Anguelov. Nerdi: Single-view nerf synthesis with

- language-guided diffusion as general image priors. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20637–20647, 2023. 3
- [7] Ziya Erkoç, Fangchang Ma, Qi Shan, Matthias Nießner, and Angela Dai. Hyperdiffusion: Generating implicit neural fields with weight-space diffusion, 2023. 3
- [8] Deep Floyd. If project. https://github.com/deepfloyd/IF, 2023. 5, 6
- [9] Lin Gao, Jie Yang, Tong Wu, Yu-Jie Yuan, Hongbo Fu, YuKun Lai, , and Hao Zhang. Sdm-net: Deep generative network for structured deformable mesh. ACM Transactions on Graphics (TOG), 38:1–15, 2019. 3
- [10] Philipp Henzler, Niloy J. Mitra, and Tobias Ritschel. Escaping plato’s cave: 3d shape from adversarial rendering. In The IEEE International Conference on Computer Vision (ICCV),

2019. 3

- [11] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6
- [12] Moritz Ibing, Gregor Kobsik, and Leif Kobbelt. Octree transformer: Autoregressive 3d shape generation on hierarchically structured sequences. arXiv preprint arXiv:2111.12480, 2021. 3
- [13] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 867–876,

2022. 3

- [14] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 2, 3, 5, 6, 7, 1
- [15] Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Popa Tiberiu. Clip-mesh: Generating textured meshes from text using pretrained image-text models. SIGGRAPH Asia 2022 Conference Papers, 2022. 3
- [16] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1
- [17] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, MingYu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to3d content creation. arXiv preprint arXiv:2211.10440, 2022. 2, 3
- [18] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. arXiv preprint arXiv:2303.11328,

2023. 2, 3, 8

- [19] Zhen Liu, Yao Feng, Michael J. Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. Meshdiffusion: Score-based generative 3d mesh modeling. In International Conference on Learning Representations, 2023. 3
- [20] Xiaoxiao Long, Cheng Lin, Peng Wang, Taku Komura, and Wenping Wang. Sparseneus: Fast generalizable neural surface reconstruction from sparse views. In European Conference on Computer Vision, pages 210–227. Springer, 2022. 5, 6

- [21] Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and Andrea Vedaldi. Realfusion: 360 reconstruction of any object from a single image. In CVPR, 2023. 3
- [22] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. arXiv preprint arXiv:2211.07600,

2022. 2, 3, 5

- [23] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12663–12673, 2023. 2
- [24] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1– 102:15, 2022. 6
- [25] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 2, 3
- [26] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, , and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 165–174, 2019. 3
- [27] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 6, 7
- [28] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 3
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [31] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 3
- [32] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, and Seungryong Kim. Let 2d diffusion model know 3dconsistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937, 2023. 3
- [33] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv:2308.16512, 2023. 3

- [34] Edward Smith and David Meger. Deep unsupervised learning using nonequilibrium thermodynamics. In Conference on Robot Learning, pages 87–96. PMLR, 2017. 3
- [35] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. arXiv preprint arXiv:2212.00774, 2022. 3
- [36] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 5
- [37] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2, 3, 6, 7, 8, 4
- [38] Jiajun Wu, Chengkai Zhang, Tianfan Xue, Bill Freeman, and Josh Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In Advances in neural information processing systems, pages 82–90, 2016. 3
- [39] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Chen Qian Jiaqi Wang, Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [40] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. Pointflow: 3d point cloud generation with continuous normalizing flows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4541–4550, 2019. 3
- [41] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578–4587, 2021. 1
- [42] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. arXiv preprint arXiv:2210.06978, 2022. 3
- [43] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023. 6, 3

## Text-to-3D Generation with Bidirectional Diffusion using both 2D and 3D priors Supplementary Material

In the supplementary material, we first introduce the data processing pipeline in (§ 5.1), then provide more implementation details of the model architecture (§ 5.2), more training details in (§ 5.3), and give more ablation results in (§ 5.4).

##### 5.1. Data Processing

As mentioned in the main paper, we use 6k ShapNetChair [2] and LVIS Objaverse 40k [5] as our training datasets. We obtain the Objaverse 40k dataset by filtering objects with LVIS category labels in the 800k Objaverse data. To process data for the 2D diffusion process, we use Blender to render each 3D object into 8 images with a fixed elevation of 30◦ and evenly distributed azimuth from −180◦ to 180◦. These fixed view images serve as the ground truth multiview image set V. In addition, we also randomly render 16 views to supervise the novel view rendering of the denoised radiance field F0′. All the images are rendered at a resolution of 256 × 256. Since we adopt the DeepFloyd as our

- 2D foundation model which runs at a resolution of 64 × 64, the rendered images are downsampled to 64 × 64 during training. To process data for the 3D diffusion, we compute the signed distance of each 3D object at each N × N × N grid point within a [−1,1] cube, where N is set to 128 in our experiments. To obtain the latent code C for each object, we use the encoder in Shap-E [14] to encode each object and

apply t0 = 0.4 level Gaussian noise to C to get noisy Ct0

, and then decode the condition radiance field during training.

Furthermore, both the ShapNet-Chair and Objaverse dataset contains no text prompts, so we use Blip-2 [16] to generate labels for the Objaverse object by rendering the image from a positive view. For evaluation, we manually choose 50 text prompts from the Objaverse dataset without LVIS label, ensuring the text prompts have not been trained during training.

- 5.2. Model Architecture Details

Our framework contains a 3D denoising network built upon

- 3D SparseConv U-Net and a 2D denoising network built upon 2D U-Net. Below we provide more details for each of them.

- 5.2.1 3D Denoising Network Given the input feature volume

Sin = Concat(M,Sp3DConv(N),

(8)

Sp3DConv(Gt0

))

as discussed in Section 3.2 of the main paper, we use a 3D sparse U-Net U to denoise the signed distance field. Specifi-

cally, we first use a 1×1×1 convolution to adjust the number of input channels to 128. Then we stack four 3×3×3 sparse 3D convolution blocks to extract hierarchical features while obtaining downsampled 8 × 8 × 8 feature grids. It is noteworthy that we inject the timestep and text embeddings into each sparse convolution block to make the network aware of the current noise level and text information. In practice, we first use an MLP to project the scalar timestep t to highdimensional features and fuse it with the text embeddings with another MLP to get the fused embeddings as follows:

emb = MLP2(Concat(embtext,MLP1(t))), (9)

where embtext denotes the text embeddings. Then in each sparse convolution block, we project the fused embeddings to scale β and shift γ:

β,γ = Chunk(MLPproj(GeLU(emb))), (10)

where GeLU is activated function, Chunk operation splits the projected features into two equal parts along the channel dimension. After that, we introduce modulation to the sparse convolution by:

Sk+1 = (1+β)(SparseConv(GroupNorm(Sk)))+γ, (11)

where k denotes the feature level, Sk and Sk+1 are the input and output of the k-th level sparse convolution block. Subsequently, we use 4 sparse deconvolution blocks to upsample the bottleneck feature grids with residuals linked from the extracted hierarchical features:

Sk′ = SparseDeConv(Sk′+1) + Sk, (12)

where Sk′+1 and Sk′ are the input and output of the k-th level sparse de-convolution block, and obtain the output features

S of the 3D U-Net.

To obtain the denoised signed distance field, we first query each 3D position p in the fused feature grid S to fetch its feature S(p) by Trilinear Interpolation. Then we apply several MLPs (we adopt the ResNetFC blocks in [41]) to predict the signed distance at position p:

F0′ = MLP(S(p),λ(p)), (13) where λ(p) is the positional encoding:

λ(p) =(sin(20ωp),cos(20ωp),sin(21ωp),cos(21ωp),

...,sin(2L−1ωp),cos(2L−1ωp)).

(14) L is set to 6 in all experiments.

[Figure 223]

###### Figure 8. More ablation results showing the importance of both 2D and 3D priors in our model.

[Figure 224]

A silver platter piled high with fruits.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

A lemur taking notes in a journal.

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

An orangutan playing accordion with its hands spread wide.

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

A bear dancing ballet.

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

A pig wearing a backpack.

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

An airplane made out of wood.

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

A car made out pizza.

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

A lionfish.

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

A llama wearing a suit.

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

- Figure 9. More generated 3D objects by our model. Left side shows the diffusion output and right side shows the 3D object after optimization.

###### 5.2.2 2D Denoising Network

Our 2D denoising network contains a U-Net of the 2D foundation model (DeepFloyd) and a ControlNet [43] modulation module to jointly denoise the multi-view image set. In prac-

M i=1 from the 2D

tice, given the M noisy images Vt = Iti

diffusion process and M rendered images Hi Mi=1 from the

###### 3D diffusion process as mentioned in Section 3.3 of the main paper, we first reshape both of them from [B,M,C,H,W] to [B × M,C,H,W], where B,C,H,W denote batch size,

channel, height, width, respectively. Then we feed the noisy images to the frozen encoder E∗ of DeepFloyd to get encoded features:

M i=1),t,embtext). (15)

P = E∗(Reshape( Iti

P = pk Kk=1 where pk denotes the k-th features of the total K feature levels. Simultaneously, we feed the rendered

images to the trainable copy encoder E of ControlNet to obtain the hierarchical 3D consistent condition features:

Q = E(Reshape( Hi Mi=1),t,embtext), (16)

where Q = qk Kk=1. Subsequently, we decode P with the frozen decoder D∗ of DeepFloyd and the condition residual

features Q. Specifically, in the k-th decoding stage, we first apply zero-convolutions to the condition feature qk and then add it to the original decoded features as residuals:

fˆk = pk + Dk∗−1(pk−1) + ZeroConv(qk), (17)

where Dk∗−1 denotes the k − 1-th frozen decoding layer of DeepFloyd. In this way, we can denoise the multi-view noisy

images in a unified manner by introducing the 3D consistent condition signal as guidance. In practice, we set M = 8 in our experiments.

###### A lionfish. A llama wearing a suit.

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

PolificDreamer (6h+)

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

Our Bidirectional Diffusion Results (40s)

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Refinement Results (20min)

- Figure 10. Comparison between our results with the object directly generated by the optimization method (ProlificDreamer).

##### 5.3. More Training Details

We train our framework on 4 NVIDIA A100 GPUs with a batch size of 4. For ShapeNet-Chair, the training takes about 8 hours to converge. For Objaverse 40k, the training takes 5 days. We use the AdamW optimizer with β = (0.9,0.999) and weight decay = 0.01. Notably, we set the learning rate of the 2D diffusion model to 2 × 10−6 while using a much larger learning rate of 5 × 10−5 for the 3D diffusion model.

##### 5.4. More Experiments

###### 5.4.1 Ablation for Priors

In Fig. 8, we provide additional results for the ablation of 3D and 2D priors mentioned in Sec. 4.3. Our method can produce more realistic textures with 2D priors and more robust geometry with 3D priors.

###### 5.4.2 Visualization of 2D-3D Denoising

We also demonstrated the visualization of 2D and 3D denoising processes during bidirectional diffusion sampling as shown in Fig. 11. The top two lines show the rendering views

[Figure 328]

Figure 11. Visualization of our 2D and 3D denoising processes (the maximum diffusion step is 1,000). The top two rows show the rendering views of the implicit field during the 3D denoising process, and the bottom two rows show the 2D sample results during the 2D denoising process.

of the implicit field during the 3D denoising process, and the bottom two lines show the 2D sample results during the 2D denoising process. 3D and 2D representations are jointly denoised, and in the early step of diffusion sampling, 3D representations can provide basic geometric shapes, which guides 2D diffusion to generate geometrically reasonable images. In the later step of sampling, texture generation is dominated by 2D diffusion.

###### 5.4.3 More Results

In Fig. 9, we provide more high-quality results generated by our entire framework. And in Fig. 10, we demonstrated a comparison with the previous state-of-the-art optimization method [37]]. Our approach not only significantly reduces time costs but is also more robust in understanding geometry.

