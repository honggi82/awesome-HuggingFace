## GECO: Generative Image-to-3D within a SECOnd

Chen Wang1 Jiatao Gu2 Xiaoxiao Long3 Yuan Liu3 Lingjie Liu1 1University of Pennsylvania 2Apple 3The University of Hong Kong

View 1 View 2 View 3

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

# arXiv:2405.20327v2[cs.CV]20Aug2024

[Figure 8]

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

Input Images Generated Textured Meshes

Figure 1. We propose GECO, a framework for feed-forward image-to-3D generation that produces texture meshes in 0.64s on a single L40 GPU. Here we show both the texture and geometry renderings of the generated meshes.

### Abstract

hand, reconstruction-based approaches are more efficient but tend to compromise quality due to their limited ability to handle uncertainty. We introduce GECO, a novel method for high-quality 3D generative modeling that operates within a second. Our approach addresses the prevalent issues of uncertainty and inefficiency in existing methods

Recent years have seen significant advancements in 3D generation. While methods like score distillation achieve impressive results, they often require extensive per-scene optimization, which limits their time efficiency. On the other

[Figure 37]

[Figure 38]

Input Image

through a two-stage approach. In the first stage, we train a single-step multi-view generative model with score distillation. Then, a second-stage distillation is applied to address the challenge of view inconsistency in the multi-view generation. This two-stage process ensures a balanced approach to 3D generation, optimizing both quality and efficiency. Our comprehensive experiments demonstrate that GECO achieves high-quality image-to-3D mesh generation with an unprecedented level of efficiency. We will make the code and model publicly available.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

One-step Multi-View Image Generator

Mesh Reconstructor

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

~ 0.3s

~ 0.3s

Gaussian Noise

Multi-View Images Textured Mesh

Figure 2. Overall pipeline of our feedforward 3D generator, which achieves image-to-3D mesh generation within one second given a conditional image and noise.

first-stage diffusion model, and as a result, the predicted 3D is generally better than pure reconstruction-based models. However, the multi-view diffusion stage still needs multiple network inferences and takes over 7 seconds due to the iterative sampling nature of diffusion models, and thus bottlenecks time-sensitive applications.

### 1. Introduction

3D digital assets encapsulate the geometry and appearance of objects from the real world. The role of 3D assets is pivotal across a wide range of applications, including movies, digital games, virtual reality, and robotics. Despite their importance, generating 3D assets is often labor-intensive and typically restricted to skilled professionals. Automatic and efficient techniques for generating high-fidelity 3D models will greatly simplify the workload and open up the creation process to beginners. Therefore, in this paper, we study the problem of efficiently producing high-quality 3D assets using a single input image, aiming for fast and faithful reproduction of the original object in the image.

To address these issues, we present GECO, a generative approach that can generate high-fidelity 3D objects within one second. Specifically, we learn a feed-forward generator similar to reconstruction-based models [19, 88] while taking additional noise as inputs for handling uncertainties. Training such a model from scratch is a non-trivial task due to mode collapse. Instead, we parameterize our model using multi-view images as an intermediate representation (Fig. 2), and propose a novel two-stage distillation approach for training (Fig. 3). For the first stage, we follow variational score distillation (VSD [72]) and learn a single-step multi-view generator directly from a pre-trained multi-view diffusion model [57]. Since the outputs of the single-step multi-view generator are not perfectly multi-view consistent, the reconstructed 3D model from the multi-view outputs tends to have incorrect geometry. To tackle this problem, we propose a second-stage training by fine-tuning a pretrained reconstruction-based method [76] with outputs of the single-step multi-view diffusion model from the first stage. We generate pseudo ground truth images using the multi-step diffusion model and the pretrained reconstruction model, which is more consistent, to train our secondstage model with reconstruction losses. Notably, this training strategy enables using images from arbitrary viewpoints to supervise high-quality reconstruction.

Dreamfusion [48] and the follow-up works [4, 29, 31, 50, 67, 72] propose to distill 3D neural representations [23, 42, 44] from pretrained large-scale 2D diffusion models [51, 52] with score distillation techniques. These methods generate high-quality 3D assets with text or image input, however, facing the major drawback that they require 30 minutes of per-scene optimization for only one object, which raises practical concerns in real-time applications. On the other hand, to accelerate 3D generation, reconstruction-based models (e.g., PixelNeRF [80], LRM [19], TripoSR [69]) train a deterministic feed-forward model for predicting 3D representations given a single input image. By leveraging large-scale 3D datasets, such models exhibit impressive generalization ability over unseen objects, and only require less than a second to obtain the 3D. However, the uncertainty issue of the single image to 3D prediction is fundamentally unsolvable for deterministic methods: unseen regions of a 3D object cannot be fully recovered from the single image input, causing blurriness and incorrect geometries.

We conduct extensive quantitative and qualitative comparisons on GSO [9] dataset. We also test GECO on more challenging in-the-wild input images (some of them are shown in Fig. 1). The results show that our method can well resolve the uncertainty of image-to-3D generation, while being highly efficient in rendering and mesh extraction. Compared to previous feed-forward baselines, our method synthesizes high-quality texture and geometry even for the back view of the input object.

Tackling the uncertainty issues, various methods have been proposed for incorporating generative models such as diffusion models for text-to-3D generation tasks [11, 25, 66, 73]. For instance, InstantMesh [76] employs a multiview diffusion model [57, 59] to first synthesize multi-view consistent images given a single image, and predict the final 3D representations based on the predicted images. In such case, the uncertainty problem can be addressed by the

Our contributions can be summarized as the following:

- • We design a novel feed-forward model for single-imageto-3D generation that, for the first time, handles the uncertainty issue while achieving high efficiency.
- • We propose a two-stage distillation method that effi-

ciently distills a pre-trained multi-view diffusion model and a reconstruction-based model into a feedforward image-to-3D generation model.

• Extensive experiments demonstrate that GECO achieves high-quality 3D generation within one second, outperforming reconstruction-based methods in terms of quality and existing diffusion-based methods in generation speed.

### 2. Related Work

Acceleration of Diffusion Models Diffusion Models [8, 15, 61, 62], also known as score-based generative models, achieved tremendous success for various generative tasks, including text [10], image [51, 52], video [16, 17] and 3D [11, 31]. The continuous form of diffusion models are SDEs that transform between data distribution and a prior distribution [62]. The SDEs also have corresponding probability flow ODE with the same marginal distribution [61, 62]. One of the major drawbacks of diffusion models is that they require hundreds of denoising steps to generate the final output. Researchers have proposed efficient diffusion samplers [1, 22, 36, 85] to reduce the sampling steps of pretrained diffusion models to less than 50. Another line of work formulates the acceleration problem under the framework of knowledge distillation [14], where a fast student model is distilled from the teacher model. The pioneering work of Salimans and Ho [53] progressively reduces the number of steps for StableDiffusion by training multiple student models. Consistency models [39, 63] and BOOT [12] learn a one-step generator that matches the output of the teacher model along the ODE trajectory at each timestep by bootstrapping in a forward or backward manner respectively. Recently, ADD [55] and DMD [79] introduced score distillation [48, 72] for diffusion distillation.

3D Generation with Diffusion Models Researchers have explored directly training diffusion models on 3D representations, e.g. point clouds, triplanes, neural fields [6, 21, 38, 43, 60, 74]. However, they require exhaustive 3D data and computation resources and are also limited to categorylevel shape generation with simple textures. Other works proposed to learn 3D models from 2D pretrained diffusion models with score distillation [48, 70, 72] by matching the distribution of 3D renderings with that of 2D images. Follow-up works further improves the quality by using high-resolution guidance [4, 29, 50, 64], disentangling geometry and apperance [4, 50], and introducing advanced diffusion guidance [24, 27, 41, 64, 86]. Currently, these methods achieve high-fidelity 3D generation with detailed texture. Besides, the same objective is also widely utilized in scene-level generation [18, 46], 3D editing [26, 87], texturing [41, 78] and articulated object generation [2, 20, 28].

As an intermediate 3D representation, the generation of multi-view images using diffusion models has been explored. The advantage of multi-view images is that they are

batched 2D projections and can be directly processed by existing image diffusion models with minor changes. Existing works [32, 34, 37, 54, 57–59, 68, 75] fine-tuned from pretrained StableDiffusion variants to generate view consistent multi-view images, which is then fused or reconstructed to 3D representations. However, they still require several seconds to perform diffusion sampling and our work addresses this by learning to generate multi-view images in one step.

Efficient 3D Generation Methods based on score distillation often require several minutes of optimization to obtain one 3D model even with efficient 3D Gaussians [5, 67]. Some works [35, 49] use score distillation to train a hypernetwork of neural fields, enabling 3D generation from direct inference but having limited generalization ability. Recently, LRM [19] and TripoSR [69] trains a reconstruction model on large-scale datasets [7] and enable image-to-3D in seconds. TriplaneGaussian [88] uses 3D Gaussians to assist the generation of triplanes for LRM. However, the major problem of reconstruction-based methods is that they do not consider the uncertain nature of 3D generation, so the back views of the generated objects are often blurry. Based on LRM, Instant3D [25] and InstantMesh [76] samples multiview images with 2D diffusion for LRM reconstruction, and DMV3D [77] directly trains a 3D diffusion with LRM. All three works improve quality but sacrifice efficiency.

### 3. Preliminaries

#### 3.1. Multi-view Diffusion Models

Diffusion Models [15, 61] learn the data distribution by estimating the noised data distribution (or score) along a Markov Chain. Diffusion models consist of a forward process that gradually removes information from data by adding Gaussian noises and a reverse process that generates data starting from random noise. Given x0 ∼ q(x0), the forward process q is a Markov chain that adds gaussian noise to x0 and generates latent x1,...,xT of the same dimension with q(xt|x) = N(αtx,σt2I). Ideally, the final latent xT will follow a standard Gaussian distribution: p(xT) = N(xT;0,I). The reverse process starts denoising from xT by learning the Gaussian transitions from xt to xt−1 that is defined as pθ(x0:T) := p(xT) Tt=1 pθ(xt−1|xt). Further, pθ(xt−1|xt) = N(xt−1,µθ(xt,t),σt2I) and µθ is the learnable component. The sampling of diffusion models often takes more than 50 steps to obtain high-quality results.

Multi-view Diffusion Models learn the joint probability distribution of multi-view images [32, 34, 57]. This kind of model treats multi-view renderings of an object on a fixed set of viewpoints as the data point. The learning process of multi-view diffusion models is similar to standard image diffusion models except that noises are added and de-

[Figure 47]

- Figure 3. The two-stage learning pipeline for GECO. Stage I: the multi-view generator is optimized with VSD [72] objective with a pretrained multi-view diffusion model [57]; Stage II: the full model is optimized by predicting the rendering from the pre-trained reconstruction model [66] under the same image and noise condition.

noised simultaneously to those images. They also need special design to maintain the consistency of different viewpoints. However, the problem with these models is that multiple inference steps are required, and the results are not view-consistent enough. Furthermore, post-processing is also needed to reconstruct 3D geometry and appearance from the multi-view image outputs of these models.

els efficiently using a two-stage distillation approach given pre-trained multi-view diffusion and reconstruction models, where we first learn an efficient multi-view generator based on variational score distillation (VSD, Sec. 4.1), and then finetune our model with a 3D consistent distillation algorithm (Sec. 4.2). The learning pipeline is shown in Fig. 3.

#### 3.2. 3D Reconstruction Models

Reconstruction models aim to produce 3D representations of an object from a single view or multiple views. PixelNeRF [80] achieves single-view 3D reconstruction by projecting the input image features to 3D and applying volume rendering for learning 3D representations. The recent work, LRM [19], greatly boosts the reconstruction quality of PixelNeRF by leveraging a large transformer model and a huge amount of data. However, these methods generally synthesize blurry results from unseen viewpoints because they don’t model the uncertainty and only use regression losses to train. This issue can be addressed by using multi-view inputs for the reconstruction model. For example, by using multi-view images generated by multi-view diffusion models as the input, the 3D reconstruction model, Instant3D [25], LGM [66], and InstantMesh [76] can reconstruct 3D models from text or image prompts. Specifically, InstantMesh [76] reconstructs 3D meshes with an isosurface extraction module, i.e., FlexiCubes [56] representation from multi-view images.

### 4. Method

In this section, we introduce GECO – a novel image-to3D generative model that achieves both efficient sampling and high-quality generation. More precisely, given a single image of an object and a random noise z, GECO learns a single-step generator to output 3D representations (we mainly experimented on meshes in this paper) of the object. An illustration of our proposed model is shown in Fig. 2 where multi-view images are used as an intermediate representation similar to [25, 76]. We learn our mod-

#### 4.1. Stage I: Multi-view Score Distillation

Variational Score Distillation (VSD) VSD [72] is an extension of Score Distillation Sampling (SDS), which was first introduced by DreamFusion [48] for distilling pretrained 2D diffusion knowledge into 3D. The core idea of SDS is to match the score function between the output of a learnable parametric image generator and the real data estimated by a pretrained diffusion model. Given a datapoint x = g(θ) generated by the differentiable image generator g parametrized with θ, SDS adds Gaussian noise of level t and turns it into xt. It then uses a pre-trained diffusion model with denoising function ϵϕ(xt;y,t) to predict the noise with condition y to optimize θ. ProlificDreamer [72] proposed VSD to further improve SDS by directly optimizing the distribution of θ such that the rendering distribution q(x|y) with condition y align with the pretrained diffusion model p(x|y) by minimizing their KL divergence: DKL(q(x|y)||p(x|y)). In practice, this is achieved by learning a separate “student model” that estimates the score function of the learned 3D models. The learned score will be used for back-propagation to learn 3D distribution.

Generative Modeling with VSD The original ProlificDreamer parameterized the 3D distribution using a fixed number of particles [72], which, however, does not allow us to draw new samples from the learned distribution. To facilitate learning a 3D generative model that can handle novel scenes, we propose to replace the original parameterization with a learnable generator G(θ) that transforms a random Gaussian noise ϵ input to a data sample. The training ob-

jective of G is derived as follows:

∇θLVSD = Et,ϵ w(t)(ϵpre(xt;y,t) − ϵstu(xt;y,t))∂G∂θ(θ,z)

(1)

where x0 = G(θ,z) is the clean sample of the generator output given noise z ∈ N(0,I) and xt is the noisy version of x0, t is the diffusion timestep, ϵpre and ϵstu are the predictions of the pretrained diffusion model and the student model respectively. The student model is trained online on the output of G to estimate the score of the generated samples:

Lstu = Et,ϵ∥ϵstu(xt;y,t)) − ϵ∥22 (2)

Multi-view Distillation Ideally, our goal is to learn a 3D generator that directly maps random noises to 3D representations using VSD, and the 2D renderings of the generator become the input of the 2D diffusion models. Here, we can leverage large-scale pre-trained multi-view diffusion models [34, 57] as our teacher models to improve the learning of 3D inductive bias. A natural design would be to parametrize G(θ,z) with a 3D generator, such as a triplane generator [3]. However, we found that training a generator from scratch without proper initialization would lead to severe mode collapse, i.e. all the samples drawn from the generator will become identical. This observation coincides with the finding in recent work [12, 40, 79] for the distillation of 2D Diffusion models trained on single-view images.

To circumvent this problem, we propose to first learn multi-view images as an intermediate representation using VSD. This allows us to use the same architecture and initial parameters as the pretrained model for our generator G which is essentially a single-step multi-view generator. In GECO, we employ Zero123Plus [57] as our teacher model because it provides photorealistic and highly consistent 6view renderings. In contrast to [57] that uses reference attention [82] to concatenate the self-attention matrices of the noised condition image, we directly used the self-attention matrices of the clean condition image to preserve the information. As mentioned earlier, we initialize the generator with pretrained Zero123Plus with the additional conversion from v-prediction to x0-prediction.

#### 4.2. Stage II: 3D Consistent Distillation

After the multi-view images of the object are obtained, our next step is to estimate the 3D representation of the object from the multi-view images. One potential solution is to apply a pretrained 3D reconstruction network R that takes multi-view images as input and outputs a 3D representation. However, one major drawback of this approach is that the output of the one-step multi-view generator G(θ), which is also the input to the reconstruction network R, has low multi-view consistency compared to the ground-truth multiview images, which causes training-testing mismatch in the

reconstruction model, resulting in incorrect geometry in the output 3D reconstruction.

As shown in Fig. 3, we propose a second distillation stage to resolve this inconsistency issue, which finetunes a reconstruction model as part of the generative model. Considering that the multi-view generation of the teacher diffusion model is much more consistent than the learned single-step generator, we can use the 3D representation reconstructed from these images as pseudo ground truth to refine the reconstruction model. Namely, given a condition image y and sampled noise z, we conduct the deterministic DDIM sampling [61] using Zero123Plus [57] to obtain xmv. 3D representations are then reconstructed based on the pretrained 3D reconstructor R(xmv). With the reconstructed mesh, we render from random viewpoints to create a set of pseudo ground truth images {Iisyn(z),i = 1,...,N}. We collect such paired dataset D = (z, {Iisyn(z)|i = 1,...,N}) for each sampled noise z, and use them for training the final generator which includes our pretrained single-step multiview generator (described in Sec. 4.1) and a pretrained 3D reconstructor [25, 76]. Here, I(z)i represents the i−th view rendered from the generator given z. In practice, we finetune our final generator by minimizing the difference between the renderings of the generator’s output and the corresponding pseudo ground truth images {Iisyn(z),i = 1,...,N} rendered from the same viewpoint in terms of the RGB loss and LPIPS [84] loss:

L3D = Ez,Isyn(z) LMSE(Irgb(z),Irgbsyn(z)) + λ · LLPIPS(Irgb(z),Irgbsyn(z))

(3)

Note that the 3D reconstructor in the final generator can be regarded as a refinement module that tackles the multiview inconsistency issue for 3D reconstruction. Furthermore, this training strategy allows us to go beyond the fixed six-view setting specified in Zero123Plus [57] and use the renderings from arbitrary viewpoints for training, which is an important factor for high-quality 3D reconstruction.

### 5. Experiments

#### 5.1. Implementation Details

Datasets We train our model on the LVIS subset of the Objaverse [7] dataset, which contains approximately 46,000 objects. For each scene, we only need images at one viewpoint to be the condition image of Zero123Plus [57].

Multi-view Score Distillation For Stage I training, the multi-view generator, pretrained teacher Zero123Plus model, and student Zero123Plus model as shown in Fig. 3 are all initialized with the fine-tuned white background Zero123Plus [57] in InstantMesh [76]. We train the generator and student model on a single NVIDIA L40 GPU for 5,000 steps. In each iteration, the generator and student model are updated alternatively. The t for the student

- Table 1. Quantitative comparison of novel view synthesis of GECO and the baselines on GSO [9] dataset. We report PSNR, SSIM [71], LPIPS [84] for novel view synthesis, CD and volume IoU for geometry. For the runtime, “Get 3D” refers to the time that the model predicts the 3D representations (e.g. triplanes, gaussians) from single view inputs, “3D to mesh” is the time that converts the 3D representations to meshes. We tested all methods on NVIDIA L40. The best results are bolded and the second best results are underlined.

Novel View Synthesis Geometry Runtime PSNR↑ SSIM↑ LPIPS↓ CD↓ vIoU↑ Get 3D (s)↓ 3D to Mesh (s)↓ Total ↓

Method

TriplaneGaussian [88] 18.52 0.817 0.191 0.036 0.492 0.11 140.0 140.1 OpenLRM [13, 19] 18.15 0.810 0.173 0.035 0.557 0.22 1.06 2.49 TripoSR [69] 18.33 0.812 0.172 0.033 0.577 0.16 1.30 1.46 LGM [66] 18.11 0.805 0.178 0.038 0.478 1.28 145.9 147.1 InstantMesh [76] (Our teacher) 19.15 0.822 0.152 0.028 0.626 7.06 0.30 7.36 Ours 19.31 0.825 0.154 0.029 0.599 0.34 0.30 0.64

model training is randomly sampled from [0.02,0.98]. We use a fixed guidance scale of 4 for the generator and the pretrained teacher model, and a guidance scale of 1 for the student model. The generator and the student model are both optimized by the Adam optimizer with learning rate 1e-6, and betas (0.9,0.999). We found it is crucial to balance the learning rate of the generator and student model, otherwise the generator will not converge to reasonable results.

3D Consistent Distillation We adopt InstantMesh [76] as our reconstruction network R. For each condition image, we ran Zero123Plus with deterministic 75-step DDIM scheduler [61] to obtain the pseudo ground truth six views and use it as input of the InstantMesh [76] to inference 3D meshes. Then we render 50 images at random viewpoints to save them for Stage II training. In Stage II, we use a learning rate of 1e-6 and a batch size of 8 to train 10 epochs.

Inference The whole pipeline of GECO takes about 0.64s for each scene to generate 3D meshes on a single NVIDIA L40 GPU, including 0.28s for multi-view image generation and 0.06s for flexicube reconstruction and 0.30s for mesh extraction. It consumes about 10 GB of GPU memory during inference.

#### 5.2. Experiment Protocol

Evaluation Dataset and Metrics Following prior works [30–32], we adopt the Google Scanned Object (GSO) [9] dataset to perform the quantitative comparison of all the methods. We use the same randomly sampled 30 objects ranging from daily objects to animals in SyncDreamer [32]. For each object, we render an image with a size of 512 × 512 as the input view with zero elevation and render another two sets for evaluation: the first set consists of 6 images from the same viewpoint as in Zero123Plus [57], the second consists of evenly sampled 15 images around the object with zero elevation. For novel view synthesis, we employ commonly used metrics for evaluation, including PSNR, SSIM [71] and LPIPS [84]. For geometry evaluation, we report chamfer distance (CD) and Volume IoU (vIoU).

Baselines We mainly compare with recent methods that focus on feed-forward 3D generation, including LRM [19], TriplaneGaussian [88] and TripoSR [69]. We use the community version OpenLRM [13] for LRM comparison since the original model is not publicly available. We also include LGM [66], which generates 3D Gaussians from multi-view images in seconds.

#### 5.3. Results

Qualitative Comparison Fig. 4 demonstrates the renderings of GECO and other baselines. We urge readers to view our supplemental video to judge the multi-view consistency of the results. Due to the reconstruction nature, the baseline methods fail to generate reasonable textures at unseen viewpoints, producing incorrect geometry and blurry renderings. Our method handles the uncertainty through multiview image generation, so even at the back viewpoints, we can synthesize details that are highly consistent with the input image. From the geometry renderings, we can see that our method also generates consistent and smooth geometry with the input image at the back viewpoints, outperforming other baselines. Video comparisons with baselines can be found in the supplementary.

Fig. 5 further shows comparisons with other methods that also generate multi-view images first and then reconstruct 3D meshes with InstantMesh. It can be seen from Fig. 5 that the output of Zero123 [31] is not consistent across different viewpoints because each view is generated separately, e.g. the shoes are in a pair in one view but not in another. Therefore, the rendered images from 3D meshes tend to be blurry and lack of details. Our method generates multi-view images that are much more consistent and look similar to 75-step sampling of Zero123Plus [57]. The highquality multi-view generation provides a basis for the 3D generation stage.

Quantitative Comparison The quantitative comparison is shown in Table 1. Our method achieves the best results in novel view synthesis and geometry metrics. The improvement results from the uncertainty handling of our approach in the occluded regions. In contrast, TriplaneGaussian [88],

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

TripoSRTriplaneGaussianOpenLRMOursInput

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

[Figure 103]

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

- Figure 4. Qualitative comparison against baseline methods. GECO outperforms the baselines, especially from the unseen views. For each method, the first row and second row are the texture and geometry renderings respectively.

LRM [19] and TripoSR [69] cannot produce sharp predictions for unseen parts, as they are deterministic models. In terms of runtime, thanks to the Flexicubes technique, our method achieves a higher rendering speed and more efficient conversion to meshes than other methods. Triplane-

Gaussian achieves highly efficient rendering with 3D Gaussians, but requires a slow mesh conversion process with Point-E [45]. OpenLRM and TripoSR render much slower due to triplane representation. Our method also achieves comparable performance compared with our teacher model

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

InputZero123

“Star Character”

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Zero123Plus

(1 step)

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

“Yellow Suitcase”

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

Zero123Plus

[Figure 171]

(75 steps)

Ours

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

“Strawberry Cake”

Multi-view Generation Mesh Rendering Multi-view Generation Mesh Rendering

Text Prompt Text to image Renderings

- Figure 5. Comparison of GECO with the baselines that use different multi-view image generation methods and then reconstruct 3D meshes. Our one-step multi-view generator produces much better results than Zero123 [31] and is comparable to Zero123Plus [57] 75-step sampling, leading to better 3D renderings.

- Table 2. Quantataively results on the renderings and geometry across different settings. We report PSNR, SSIM [71], LPIPS [84], CD and vIoU on the GSO [9] dataset.

Method PSNR↑ SSIM↑ LPIPS↓ CD↓ vIoU↑ 1-step Zero123Plus with InstantMesh [57] 14.79 0.791 0.262 0.052 0.462 Ours w/o Stage II 18.75 0.812 0.157 0.029 0.585 Ours 19.30 0.825 0.154 0.029 0.599

[Figure 178]

- Figure 6. The Stage-II training alleviates the view inconsistency issue in the multi-view diffusion outputs, resulting in higherquality results with less bad geometry and overexposure.

Figure 7. 3D generation given text prompts.

multi-view images and high-quality 3D renderings.

Effectiveness of Stage-II We compare the generation results before and after stage-II training in Fig. 6. We can see from the results that after the stage-II training, the incorrect geometry on the object has gone. This is because the Stage-II training distills a more robust 3D generator that can handle the inconsistency of input images benefiting from the multi-view supervision from the pseudo ground truth images. Also, the overexposure problem in the generated multi-view images from stage-II is also resolved (rightmost example in Fig. 6). For Table 2, we can also see that both the novel view synthesis metrics and geometry metrics are improved after stage-II.

### 6. Conclusion and Future Work

In this work, we present GECO, a generative framework for 3D content generation. We found that directly learning a 3D generative model that generalizes well involves learning from massive 3D data. Therefore, we use the intermediate representation and employ a multi-view image generation and reconstruction framework. The uncertainty of 3D generation is well addressed in the multi-view image diffusion stage that enjoys the rich prior of pretrained 2D image diffusion models. Then, 3D can be obtained through multi-view reconstruction. We further jointly learn the multi-view image generator and reconstructor to improve the 3D consistency. The whole pipeline is feed-forward and requires less than one second.

InstantMesh [76] and better results than LGM [66], while being much faster.

Text-to-image-to-3D Generation Our method can also be combined with text-to-image diffusion models for 3D generation from text prompts. We first use SD-XL [47] to generate images and then run GECO to synthesize 3D. The results are shown in Fig. 7.

Our approach still has some limitations. First, our training process involves two stages, including distilling a multiview image diffusion model and leveraging it to learn a reconstruction model. Second, the results of our work are bounded by the multi-step sampling results of the multiview diffusion models, which might still not be as consistent as renderings of 3D representations. The first stage might also influence the diversity of the generator. Future work can consider learning a one-step 3D generative model that can produce 3D representations directly, either by training from scratch or distilling a 3D diffusion model.

#### 5.4. Ablation Study

Quality of Multi-View Generator We evaluated the generated 6 views of our multi-view image generator on the same GSO [9] objects we used in Sec. 5.2. Results show that our multi-view generator after VSD distillation achieves PSNR/SSIM/LPIPS of 17.28/0.784/0.197, which is close to the results of 75-step sampling Zero123Plus [57] (18.00/0.790/0.190). From Fig. 5, we can see that 1-step inference in the original Zero123Plus cannot produce reasonable results, while our method is able to generate sharp

### References

- [1] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analyticdpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. ICLR, 2021. 3
- [2] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and KwanYee K Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models. arXiv preprint arXiv:2304.00916, 2023. 3
- [3] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In CVPR, pages 16123–16133, 2022. 5
- [4] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. ICCV, 2023. 2, 3
- [5] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using gaussian splatting. arXiv preprint arXiv:2309.16585, 2023. 3
- [6] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In CVPR, pages 2262–2272, 2023. 3
- [7] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663, 2023. 3, 5, 12
- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794,

2021. 3

- [9] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 2, 6, 8, 12
- [10] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. ICLR, 2023. 3
- [11] Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M Susskind, Christian Theobalt, Lingjie Liu, and Ravi Ramamoorthi. Nerfdiff: Single-image view synthesis with nerf-guided distillation from 3d-aware diffusion. In ICML, pages 11808–

11826. PMLR, 2023. 2, 3

- [12] Jiatao Gu, Chen Wang, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Data-free distillation of diffusion models with bootstrapping. In ICML, 2024. 3, 5
- [13] Zexin He and Tengfei Wang. Openlrm: Open-source large reconstruction models. https://github.com/ 3DTopia/OpenLRM, 2023. 6, 12
- [14] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 3
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 3, 12

- [16] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [17] Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In ICLR Workshop on Deep Generative Models for Highly Structured Data, 2022. 3
- [18] Lukas H¨ollein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2room: Extracting textured 3d meshes from 2d text-to-image models. ICCV, 2023. 3
- [19] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. ICLR, 2024. 2, 3, 4, 6, 7, 12
- [20] Tomas Jakab, Ruining Li, Shangzhe Wu, Christian Rupprecht, and Andrea Vedaldi. Farm3d: Learning articulated 3d animals by distilling 2d diffusion. arXiv preprint

- arXiv:2304.10535, 2023. 3

[21] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint

- arXiv:2305.02463, 2023. 3

- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. NeurIPS, 35:26565–26577, 2022. 3
- [23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4), 2023. 2, 12
- [24] Jeong-gi Kwak, Erqun Dong, Yuhe Jin, Hanseok Ko, Shweta Mahajan, and Kwang Moo Yi. Vivid-1-to-3: Novel view synthesis with video diffusion models. arXiv preprint arXiv:2312.01305, 2023. 3
- [25] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. ICLR, 2024. 2, 3, 4, 5
- [26] Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Textdriven 3d editing via focal-fusion assembly. arXiv preprint arXiv:2308.10608, 2023. 3
- [27] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. arXiv preprint arXiv:2311.11284, 2023. 3
- [28] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxaing Tang, Yangyi Huang, Justus Thies, and Michael J Black. Tada! text to animatable digital avatars. arXiv preprint arXiv:2308.10899, 2023. 3
- [29] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, pages 300–309, 2023. 2, 3
- [30] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single

- image to 3d mesh in 45 seconds without per-shape optimization. NeurIPS, 36, 2024. 6
- [31] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In ICCV, pages 9298– 9309, 2023. 2, 3, 6, 8
- [32] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. ICLR, 2024. 3, 6, 12
- [33] Ying-Tian Liu, Yuan-Chen Guo, Vikram Voleti, Ruizhi Shao, Chia-Hao Chen, Guan Luo, Zixin Zou, Chen Wang, Christian Laforte, Yan-Pei Cao, et al. threestudio: a modular framework for diffusion-guided 3d generation. 12
- [34] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023. 3, 5
- [35] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, Ming-Yu Liu, Sanja Fidler, and James Lucas. Att3d: Amortized text-to-3d object synthesis. arXiv preprint arXiv:2306.07349, 2023. 3
- [36] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. NeurIPS, 35:5775–5787, 2022. 3
- [37] Yuanxun Lu, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Xun Cao, and Yao Yao. Direct2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion. arXiv preprint arXiv:2311.15980, 2023. 3
- [38] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In CVPR, pages 2837–2845,

2021. 3

- [39] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3
- [40] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. NeurIPS, 36, 2024. 5
- [41] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In CVPR, pages 12663–12673, 2023. 3
- [42] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [43] Norman M¨uller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias Nießner. Diffrf: Rendering-guided 3d radiance field diffusion. In CVPR, pages 4328–4338, 2023. 3

- [44] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 41(4):1–15, 2022. 2
- [45] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 7
- [46] Ryan Po and Gordon Wetzstein. Compositional 3d scene generation using locally conditioned diffusion. arXiv preprint arXiv:2303.12218, 2023. 3
- [47] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 8
- [48] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. ICLR,

2023. 2, 3, 4

- [49] Guocheng Qian, Junli Cao, Aliaksandr Siarohin, Yash Kant, Chaoyang Wang, Michael Vasilkovsky, Hsin-Ying Lee, Yuwei Fang, Ivan Skorokhodov, Peiye Zhuang, et al. Atom: Amortized text-to-mesh using 2d diffusion. arXiv preprint arXiv:2402.00867, 2024. 3
- [50] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. ICLR, 2024. 2, 3
- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 2, 3
- [52] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022. 2, 3
- [53] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. ICLR, 2022. 3
- [54] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. Zeronvs: Zero-shot 360-degree view synthesis from a single real image. arXiv preprint arXiv:2310.17994, 2023. 3
- [55] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042, 2023. 3
- [56] Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. Flexible isosurface extraction for gradient-based mesh optimization. ACM Trans. Graph., 42(4):37–1, 2023. 4, 12
- [57] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2, 3, 4, 5, 6, 8, 12

- [58] Yukai Shi, Jianan Wang, He Cao, Boshi Tang, Xianbiao Qi, Tianyu Yang, Yukun Huang, Shilong Liu, Lei Zhang, and Heung-Yeung Shum. Toss: High-quality text-guided novel view synthesis from a single image. arXiv preprint arXiv:2310.10644, 2023.
- [59] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. ICLR, 2024. 2, 3
- [60] J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In CVPR, pages 20875–20886,

2023. 3

- [61] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. ICLR, 2021. 3, 5, 6
- [62] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ICLR, 2021. 3
- [63] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. ICML, 2023. 3
- [64] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818, 2023. 3
- [65] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. arXiv preprint arXiv:2312.13150, 2023. 13
- [66] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024. 2, 4, 6, 8, 13
- [67] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. ICLR, 2024. 2, 3
- [68] Shitao Tang, Jiacheng Chen, Dilin Wang, Chengzhou Tang, Fuyang Zhang, Yuchen Fan, Vikas Chandra, Yasutaka Furukawa, and Rakesh Ranjan. Mvdiffusion++: A dense high-resolution multi-view diffusion model for single or sparse-view 3d object reconstruction. arXiv preprint

- arXiv:2402.12712, 2024. 3

[69] Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d object reconstruction from a single image. arXiv preprint

- arXiv:2403.02151, 2024. 2, 3, 6, 7, 12

- [70] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR, pages 12619–12629, 2023. 3
- [71] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 6, 8
- [72] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. NeurIPS, 36, 2024. 2, 3, 4

- [73] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024. 2
- [74] Tong Wu, Zhibing Li, Shuai Yang, Pan Zhang, Xingang Pan, Jiaqi Wang, Dahua Lin, and Ziwei Liu. Hyperdreamer: Hyper-realistic 3d content generation and editing from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10, 2023. 3
- [75] Desai Xie, Jiahao Li, Hao Tan, Xin Sun, Zhixin Shu, Yi Zhou, Sai Bi, S¨oren Pirk, and Arie E Kaufman. Carve3d: Improving multi-view reconstruction consistency for diffusion models with rl finetuning. arXiv preprint arXiv:2312.13980,

- 2023. 3

[76] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

- 2024. 2, 3, 4, 5, 6, 8, 12, 13

- [77] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. ICLR, 2024. 3
- [78] Yu-Ying Yeh, Jia-Bin Huang, Changil Kim, Lei Xiao, Thu Nguyen-Phuoc, Numair Khan, Cheng Zhang, Manmohan Chandraker, Carl S Marshall, Zhao Dong, et al. Texturedreamer: Image-guided texture synthesis through geometryaware diffusion. arXiv preprint arXiv:2401.09416, 2024. 3
- [79] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2023. 3, 5
- [80] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In CVPR, pages 4578–4587, 2021. 2, 4
- [81] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A largescale dataset of multi-view images. In CVPR, pages 9150– 9161, 2023. 12
- [82] Lyuming Zhang. Reference-only control, 2022. 5
- [83] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 12
- [84] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595,

2018. 5, 6, 8

- [85] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpmsolver-v3: Improved diffusion ode solver with empirical model statistics. NeurIPS, 36, 2024. 3
- [86] Junzhe Zhu, Peiye Zhuang, and Sanmi Koyejo. Hifa: Highfidelity text-to-3d generation with advanced diffusion guidance. In ICLR, 2024. 3
- [87] Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and Guanbin Li. Dreameditor: Text-driven 3d scene editing with

neural fields. In SIGGRAPH Asia 2023 Conference Papers, pages 1–10, 2023. 3

[88] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. CVPR, 2024. 2, 3, 6, 12

### A. Datasets

Objaverse We use the Objaverse 1.0 [7] LVIS subset1 for our training, which contains around 46K 3D objects. Since our distillation method only requires the condition image as the input, we render one image for each scene at a random viewpoint with 49.1 FOV and a camera radius of 1.5.

GSO We use the same randomly selected 30 objects as in [32] from Google Scanned Objects (GSO) [9] dataset for evaluation. For each object, we use blender scripts to render an image with a size of 512 × 512 as the input view with zero elevation and azimuth. We render another two sets of images for evaluation: the first consists of 6 images from the same viewpoint as in Zero123Plus [57], and the second consists of evenly sampled 15 images around the object with zero elevation.

### B. Implementation Details B.1. Pretrained Models

Zero123Plus Model We use the white background Zero123Plus v1.1 model fine-tuned by InstantMesh [76] without the depth ControlNet [83] part. The input to the model is a white background image and the output is a set of six novel view images at elevation (30,−20,30,−20,30,−20) and azimuth (30,90,150,210,270,330). The elevation angles are absolute and the azimuth angles are relative to the input view. During inference, Zero123Plus forwards the UNet two times. For the first time, the UNet takes the condition image as input and the keys and values matrices of the self-attention matrices are stored. For the second time, the UNet takes the noisy image as input and concatenates the stored self-attention matrices for conditioning.

InstantMesh 3D meshes [23] are reconstructed from six input images at random poses of an input object. It mainly adopts the transformer architecture in LRM [13, 19] with two major differences: first, the input of the network is extended from single-view to six-views; second, the output 3D representation is changed from triplanes to a differentiable iso-surface extraction module, i.e., Flexicubes [56] to enable efficient rendering and mesh extraction.

1https://objaverse.allenai.org/docs/objaverse1.0#lvis-annotations

#### B.2. Training Details

GECO can be trained quite efficiently, here we provide the details our training:

- Stage I: Multi-view Score Distillation When we train the multi-view generator, we have three UNets that all initialized with the same architecture and parameters. The teacher model is freezing, and the multi-view generator and the student model are trained. The training procedure is similar to the standard VSD training implemented in threestudio [33]. For each iteration, we first random sample a batch of Gaussian noises z and use it as the multi-view generator input. The generator output is sent to the pretrained teacher and student model for VSD loss. The gradients are backpropagated to the generator. Afterward, we add noises at level t in [0.02,0.98] to the generator output and train the student model to predict the added noises. We follow the original Zero123Plus [57] model and use the DDPM Scheduler [15] with v-prediction to train the student model. The training takes about 4 hours on a single NVIDIA L40 GPU.
- Stage II: 3D Consistent Distillation For this part, we follow the InstantMesh [76] architecture design. During training, we use a batch size of 4 on 4 NVIDIA L40 GPUs for 5 epochs. The whole training takes about 10 hours. For each scene, we always use the fixed 6 viewpoints from the multiview diffusion model as input and randomly sample another 4 views from the pseudo ground truth set to compute loss.

### C. Baselines

We compare with other methods that can achieve 3D generation in a feed-forward manner. Current works that can achieve this goal are reconstruction-based methods, which train a model to predict novel views given an input image with a regression loss. We select the most representative and state-of-the-art methods for comparison: LRM [19], TriplaneGaussian [88] and TripoSR [69].

LRM trains a large-scale transformer model that uses the attention operations between learnable input embeddings and input image features to directly output triplanes. Since the official code is not public, we use the community opensource version OpenLRM [13] for comparison2. We use the openlrm-mix-large-1.1 model trained on Objaverse [7] and MVImgNet [81] datasets as the comparison.

TriplaneGaussian utilizes two transformer-based networks: a point decoder and a triplane decoder, to reconstruct 3D objects. The triplane features and point features are combined to decode 3D Gaussians for fast novel view synthesis. We use the official code for evaluation3.

- 2https://github.com/3DTopia/OpenLRM
- 3https : / / github . com / VAST - AI - Research /

TriplaneGaussian

TripoSR follows the design principle of LRM and uses a transformer network to predict triplanes from a single image. The main difference is that TripoSR curated and rendered a new set of 3D object data and employed mask loss and patched rendering loss. We use the official code for evaluation4.

### D. Additional Results

#### D.1. Qualitative Comparison with LGM and InstantMesh

Besides comparing with feedforward baselines, we also compare with methods that use multi-view diffusion models and multi-view reconstruction models for 3D generation: LGM [66] and InstantMesh [76]. LGM first predicts multi-view images and then uses an asymmetric UNet architecture that outputs 3D Gaussians stored by splatter images [65] of size 128 × 128 × 14 from 256 × 256 input images. The 14 channels of splatter images all the parameters of 3D Gaussians, including color, position, rotation, scale, etc. The whole process takes less than 2 seconds to generate 3D Gaussians, but it takes about 2 minutes to extract meshes from the Gaussians. InstantMesh is our teacher model that uses 75 steps of Zero123Plus to generate multiview images and produces meshes directly from these images. The qualitative results shown in Fig. 8 show that LGM produces multi-view inconsistent renderings, floating artifacts and bad geometry, presumably due to the 3D Gaussian representation. InstantMesh generates sharp renderings and smooth surfaces. Our method also has comparable results with our teacher InstantMesh, while being more than 10 times faster.

#### D.2. 2D Renderings

More renderings of the 3D meshes generated by our method can be found in Fig. 9.

4https://github.com/VAST-AI-Research/TripoSR

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

InputLGM (147s)InstantMesh (7.3s)Ours (0.6s)

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

[Figure 231]

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

Figure 8. Qualitative comparison of our method and LGM, InstantMesh, the two columns for each method are the RGB and geometry renderings respectively.

View 1 View 2 View 3

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

[Figure 263]

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

Input Images Texture and Geometry Renderings

Figure 9. Additional results of our method.

