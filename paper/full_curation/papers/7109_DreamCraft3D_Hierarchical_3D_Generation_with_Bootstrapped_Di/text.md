# arXiv:2310.16818v2[cs.CV]26Oct2023

## DREAMCRAFT3D: HIERARCHICAL 3D GENERATION WITH BOOTSTRAPPED DIFFUSION PRIOR

### Jingxiang Sun1∗, Bo Zhang3†, Ruizhi Shao1, Lizhen Wang1, Wen Liu2, Zhenda Xie2, Yebin Liu1† 1 Tsinghua University, 2 DeepSeek AI, 3 Independent Researcher

ABSTRACT

We present DreamCraft3D, a hierarchical 3D content generation method that produces high-fidelity and coherent 3D objects. We tackle the problem by leveraging a 2D reference image to guide the stages of geometry sculpting and texture boosting. A central focus of this work is to address the consistency issue that existing works encounter. To sculpt geometries that render coherently, we perform score distillation sampling via a view-dependent diffusion model. This 3D prior, alongside several training strategies, prioritizes the geometry consistency but compromises the texture fidelity. We further propose Bootstrapped Score Distillation to specifically boost the texture. We train a personalized diffusion model, Dreambooth, on the augmented renderings of the scene, imbuing it with 3D knowledge of the scene being optimized. The score distillation from this 3D-aware diffusion prior provides view-consistent guidance for the scene. Notably, through an alternating optimization of the diffusion prior and 3D scene representation, we achieve mutually reinforcing improvements: the optimized 3D scene aids in training the scene-specific diffusion model, which offers increasingly view-consistent guidance for 3D optimization. The optimization is thus bootstrapped and leads to substantial texture boosting. With tailored 3D priors throughout the hierarchical generation, DreamCraft3D generates coherent 3D objects with photorealistic renderings, advancing the state-of-the-art in 3D content generation. Code available at https://github.com/deepseek-ai/DreamCraft3D.

1 INTRODUCTION

The remarkable success of 2D generative modeling (Saharia et al., 2022; Ramesh et al., 2022; Rombach et al., 2022; Gu et al., 2022) has profoundly shaped the way that we create visual content. 3D content creation, which is crucial for applications like games, movies and virtual reality, still presents a significant challenge for deep generative networks. While 3D generative modeling has shown compelling results for certain categories (Wang et al., 2023a; Chan et al., 2022; Zhang et al., 2023b), generating general 3D objects remains formidable due to the lack of extensive 3D data. Recent research effort has sought to leverage the guidance of pretrained text-to-image generative models (Poole et al., 2022; Lin et al., 2023; Tang et al., 2023) and showcases promising results.

The idea of leveraging pretrained text-to-image (T2I) models for 3D generation is initially proposed by DreamFusion (Poole et al., 2022). A score distillation sampling (SDS) loss is enforced to optimize the 3D model such that its renderings at random viewpoints match the text-conditioned image distribution as interpreted by a powerful T2I diffusion model. DreamFusion inherits the imaginative power of 2D generative models and can yield highly creative 3D assets. To deal with the oversaturation and blurriness issues, recent works adopt stage-wise optimization strategies (Mildenhall et al., 2021) or propose improved 2D distillation loss (Wang et al., 2023b), which leads to an enhancement in photo-realism. However, the majority of current research falls short of synthesizing complex content as achieved by 2D generative models. In addition, these works are often plagued with the “Janus issue”, where 3D renderings that appear plausible individually show semantic and stylistic inconsistencies when examined holistically.

*Work partially done during the internship at DeepSeek AI. †Corresponding authors.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

A beagle in a detective's outfit

Humoristic san goku body mixed with wild boar head running

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

A cute rabbit in a stunning, detailed Chinese coat

A DSLR photo of spiderman

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Portrait painting of batman with black leather armor, ultra realistic

A DSLR photo of a chimpanzee dressed like Henry VIII king of England

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

An astronaut in a space suit, perched atop a jagged lunar rock, gazing up at the stars.

Isometric view of a MINI cute hyperrealistic futuristic soldier cat wearing cyberpunk

- Figure 1: By lifting 2D images to 3D, DreamCraft3D achieves 3D generation with rich details and holistic 3D consistency. Please refer to the Appendix and the demo video for more results.

In this paper, we propose DreamCraft3D, an approach to produce complex 3D assets while maintaining holistic 3D consistency. Our approach explores the potential of hierarchical generation. We draw inspiration from the manual artistic process: an abstract concept is first solidified into a 2D draft, followed by the sculpting of rough geometry, the refinement of the geometric details and the painting of high-fidelity textures. We adopt a similar approach, breaking down the challenging 3D generation into manageable steps. Starting with a high-quality 2D reference image generated from a text prompt, we lift it into 3D via stages of geometry sculpting and texture boosting. Contrary to prior approaches, our work highlights how careful consideration of each stage can unleash the full potential of hierarchical generation, resulting in superior-quality 3D creation.

The geometry sculpting stage aims to produce plausible and consistent 3D geometry from the 2D reference image. On top of using the SDS loss for novel views and photometric loss at the reference view, we introduce multiple strategies to promote geometric consistency. Foremost, we leverage an off-the-shelf viewpoint-conditioned image translation model, Zero-1-to-3 (Liu et al., 2023b), to model the distribution of novel views based on the reference image. Since this view-conditioned diffusion model is trained on diverse 3D data (Deitke et al., 2023), it provides a rich 3D prior that complements the 2D diffusion prior. Additionally, we find annealing the sampling timestep and progressively enlarging training views are crucial to further improve coherency. During optimization, we transition from implicit surface representation (Wang et al., 2021) to mesh representation (Shen

- et al., 2021) for coarse-to-fine geometry refinement. Through these techniques, the geometry sculpting stage produces sharp, detailed geometry while effectively suppressing most geometric artifacts.

We further propose bootstrapped score distillation to substantially boost the texture. Existing viewconditioned diffusion models trained on limited 3D often struggle to match the fidelity of modern 2D diffusion models. Instead, we finetune the diffusion model according to multi-view renderings of the 3D instance being optimized. This personalized 3D-aware generative prior becomes instrumental in augmenting the 3D texture while ensuring view consistency. Importantly, we find that alternatively optimizing the generative prior and 3D representation leads to mutually reinforcing improvements. The diffusion model benefits from training on improved multi-view renderings, which in turn provides superior guidance for optimizing the 3D texture. In contrast to prior works (Poole et al., 2022; Wang et al., 2023b) that distill from a fixed target distribution, we learn from a distribution that gradually evolves according to the optimization state. Through this “bootstrapping”, our approach captures increasingly detailed texture while keeping the view consistency.

- As shown in Figure 1, our method is capable of producing creative 3D assets with intricate geometric structures and realistic textures rendered coherently in 360◦. Compared to optimization-based approaches (Poole et al., 2022; Lin et al., 2023), our method offers substantially improved texture and complexity. Meanwhile, compared to image-to-3D techniques (Tang et al., 2023; Qian et al., 2023), our work excels at producing unprecedented realistic renderings in 360◦renderings. These results suggest the strong potential of DreamCraft3D in enabling new creative possibilities in 3D content creation. The full implementation will be made publicly available.

- 2 RELATED WORK
- 3D generative models have been intensively studied to generate 3D assets without tedious manual creation. Generative adversarial networks (GANs) (Chan et al., 2021; 2022; 2021; Xie et al., 2021; Zeng et al., 2022; Skorokhodov et al., 2023; Gao et al., 2022; Tang et al., 2022; Xie et al., 2021; Sun et al., 2023; 2022) have long been the prominent techniques in the field. Auto-regressive models have been explored (Sanghi et al., 2022; Mittal et al., 2022; Yan et al., 2022; Zhang et al., 2022; Yu et al., 2023), which learn the distribution of these 3D shapes conditioned on images or texts. Diffusion models (Wang et al., 2023a; Cheng et al., 2023; Li et al., 2023; Nam et al., 2022; Zhang et al., 2023a; Nichol et al., 2022; Jun & Nichol, 2023; Bautista et al., 2022; Gupta et al., 2023) have also shown significant recent success in learning probabilistic mappings from text or images to 3D shape latent. However, these methods require 3D shapes or multi-view data for training, raising challenges when generating in-the-wild 3D assets due to the scarcity of diverse 3D data (Chang et al., 2015; Deitke et al., 2023; Wu et al., 2023) compared to 2D.

3D-aware image generation aims to render images in novel views while offering some level of 3D consistency. These works (Sargent et al., 2023; Skorokhodov et al., 2023; Xiang et al., 2023) often rely on a pretrained monocular depth prediction model to synthesize view-consistent images.

While they achieve photo-realistic renderings for categories of ImageNet, they fall short in producing results in large views. There are a few recent attempts (Watson et al., 2022; Liu et al., 2023b) that train view-dependent diffusion models on 3D data and demonstrate promising novel view synthesis capability for open domain. However, these are inherently 2D models and cannot ensure perfect view consistency.

Lifting 2D to 3D approaches improve a 3D scene representation by seeking guidance using estblished 2D text-image foundation models. Early works (Jain et al., 2022; Lee & Chang, 2022; Hong

- et al., 2022) utilize the pretrained CLIP (Radford et al., 2021) model to maximize the similarity between rendered images and text prompt. DreamFusion (Poole et al., 2022) and SJC (Wang et al., 2022), on the other hand, propose to distill the score of image distribution from a pretrained diffusion model and demonstrate promising results. Recent works have sought to further enhance the texture realism via coarse-to-fine optimization (Lin et al., 2023; Chen et al., 2023), improved distillation loss (Wang et al., 2023b), shape guidance (Metzer et al., 2023) or lifting 2D image to 3D (Deng et al., 2023; Tang et al., 2023; Qian et al., 2023; Liu et al., 2023a). Recently, Raj et al. (2023) proposes to finetune a personalized diffusion model for 3D consistent generation. However, producing globally consistent 3D remains challenging. In this work, we meticulously design 3D priors through the whole hierarchical generation process, achieving unprecedented coherent 3D generation.

- 3 PRELIMINARIES

DreamFusion (Poole et al., 2022) achieves text-to-3D generation by utilizing a pretrained text-toimage diffusion model ϵϕ as an image prior to optimize the 3D representation parameterized by θ. The image x = g(θ), rendered at random viewpoints by a volumetric renderer, is expected to represent a sample drawn from the text-conditioned image distribution p(x|y) modeled by a pretrained diffusion model. The diffusion model ϕ is trained to predict the sampled noise ϵϕ(xt;y,t) of the noisy image xt at the noise level t, conditioned on the text prompt y. A score distillation sampling (SDS) loss encourages the rendered images to match the distribution modeled by the diffusion model. Specifically, the SDS loss computes the gradient:

∂x ∂θ

, (1)

∇θLSDS(ϕ,g(θ)) = Et,ϵ ω(t)(ϵϕ(xt;y,t) − ϵ)

which is the per-pixel difference between the predicted and the added noise upon the rendered image, where ω(t) is the weighting function.

One way to improve the generation quality of a conditional diffusion model is to use the classifierfree guidance (CFG) technique to steer the sampling slightly away from the unconditional sampling, i.e., ϵϕ(xt;y,t)+ϵϕ(xt;y,t)−ϵϕ(xt,t,∅), where ∅ represents the “empty” text prompt. Typically, the SDS loss requires a large CFG guidance weight for high-quality text-to-3D generation, yet this will bring side effects like over-saturation and over-smoothing (Poole et al., 2022).

Recently, Wang et al. (2023b) proposed a variational score distillation (VSD) loss that is friendly to standard CFG guidance strength and better resolves unnatural textures. Instead of seeking a single data point, this approach regards the solution corresponding to a text prompt as a random variable. Specifically, VSD optimizes a distribution qµ(x0|y) of the possible 3D representations µ(θ|y) corresponding to the text y, to be closely aligned with the distribution defined by the diffusion timestep t = 0, p(x0|y), in terms of KL divergence:

LVSD = DKL(qµ(x0|y)||p(x0|y)). (2) Wang et al. (2023b) further shows that this objective can be optimized by matching the score of noisy real images and that of noisy rendered images at each time t, so the gradient of LVSD is

∂x ∂θ

. (3)

∇θLVSD(ϕ,g(θ)) = Et,ϵ ω(t)(ϵϕ(xt;y,t) − ϵlora(xt;y,t,c))

Here, ϵlora estimates the score of the rendered images using a LoRA (Low-rank adaptation) (Hu et al., 2021) model. The obtained variational distribution yields samples with high-fidelity textures. However, this loss is applied for texture enhancement and is helpless to the coarse geometry initially learned by SDS. Moreover, both the SDS and VSD attempt to distill from a fixed target 2D distribution which only assures per-view plausibility rather than a global 3D consistency. Consequently, they suffer from the same appearance and semantic shift issue that hampers the perceived 3D quality.

LSDS, L3D-SDS

From Implicit surface (NeuS) to 3D Mesh (DMTet)

Text-to-Image

[Figure 33]

Diffusion

Noise

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

“an astronaut in sand beach”

[Figure 38]

View-conditioned

Diffusion

Reference Novel View

Estimated Normal

Image

Geometry Sculpting Result

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

LBSD

Novel Views

[Figure 44]

Text-to-Image Diffusion

DreamBooth

Reference view

[Figure 45]

[Figure 46]

[Figure 47]

“a [V] astronaut”

Texture Boosting Result

Augmented

renderings

- Figure 2: DreamCraft3D leverages a 2D image generated from the text prompt and uses it to guide the stages of geometry sculpting and texture boosting. When sculpting the geometry, the viewconditioned diffusion model provides crucial 3D guidance to ensure geometric consistency. We then dedicately improve the texture quality by conducting a cyclic optimization. We augment the multi-view renderings and use them to finetune a diffusion model, DreamBooth, to offer multi-view consistent gradients to optimize the scene. We term the loss that distills from an evolving diffusion prior as bootstrapped distillation sampling (LBSD in the figure).

- 4 DREAMCRAFT3D

We propose a hierarchical pipeline for 3D content generation as illustrated in Figure 2. Our method first leverages a state-of-the-art text-to-image generative model to generate a high-quality 2D image from a text prompt. In this way, we can leverage the full power of state-of-the-art 2D diffusion models to depict intricate visual semantics described in the text, retaining the creative freedom as 2D models. We then lift this image to 3D through cascaded stages of geometric sculpting and texture boosting. By decomposing the problem, we can apply specialized techniques at each stage. For geometry, we prioritize multi-view consistency and global 3D structure, allowing for some compromise on detailed textures. With the geometry fixed, we then focus solely on optimizing realistic and coherent texture, for which we jointly learn a 3D-aware diffusion prior that bootstraps the 3D optimization. In the next, we elaborate on key design considerations for the two phases.

- 4.1 GEOMETRY SCULPTING

- At this stage, we aim to craft a 3D model such that it matches the appearance of the reference image xˆ at the same reference view while maintaining plausibility under different viewing angles. To achieve this, we encourage plausible image renderings for each randomly sampled view, recog-

nizable by a pretrained diffusion model. This is achieved using the SDS loss LSDS, as defined in Equation 1. In order to effectively utilize guidance from the reference image, we penalize the pho-

tometric difference between the rendered image and the reference via Lrgb = ∥mˆ ⊙ (xˆ − g(θ; ˆc))∥2 at the reference view cˆ. The loss is computed only within the foreground region denoted by the

mask mˆ . Meanwhile, we implement the mask loss Lmask = ∥mˆ − gm(θ; ˆc)∥2 to encourage scene sparsity, where gm renders the silhouette. In addition, akin to (Deng et al., 2023), we fully exploit the geometry prior inferred from the reference image, and enforce the consistency with the depth and normal map computed for the reference view. The corresponding depth and normal loss are

respectively computed as:

conv(d,dˆ) σ(d)σ(dˆ)

n · nˆ ∥n∥2 · ∥nˆ∥2

, (4)

Ldepth = −

, Lnormal = −

where conv(·) and σ(·) represent the covariance and variance operators respectively, and the depth dˆand the normal nˆ at the reference view are computed using the off-the-shelf single-view estimator (Eftekhar et al., 2021). The depth loss adopts the form of negative Pearson correlation Ldepth to account for the scale mismatch in depth.

Despite these, maintaining consistent semantics and appearance across back-views remains a challenge. Thus, we employ additional techniques to produce coherent, detailed geometry.

3D-aware diffusion prior. We argue that the 3D optimization with per-view supervision alone is under-constrained. Hence, we utilize a view-conditioned diffusion model, Zero-1-to-3, which is trained on a large scale of 3D assets and offers an improved viewpoint awareness. The Zero-1-to-3 is a fine-tuned 2D diffusion model, which hallucinates the image in a relative camera pose c given the reference image xˆ. This 3D-aware model encodes richer 3D knowledge of the visual world and allows us to better extrapolate the views given a reference image. As such, we distill the probability density from this model and compute the gradient of a 3D-aware SDS loss for novel views:

∂x ∂θ

]. (5)

∇θL3D-SDS(ϕ,g(θ)) = Et,ϵ[ω(t)(ϵϕ(xt;xˆ,c,y,t) − ϵ)

This loss effectively alleviates 3D consistency issues like Janus problem. However, the finetuning on limited categories of 3D data of inferior rendering quality impairs the diffusion model’s generation capability, so the 3D-aware SDS loss alone is prone to induce deteriorated quality when lifting general images to 3D. Therefore, we employ a hybrid SDS loss, which incorporates both the 2D and 3D diffusion priors simultaneously. Formally, this hybrid SDS loss provides the gradient as:

#### ∇θLhybrid(ϕ,g(θ)) = ∇θLSDS(ϕ,g(θ)) + µ∇θL3D-SDS(ϕ,g(θ)), (6)

where we choose µ = 2 to emphasize the weight of the 3D diffusion prior. When computing LSDS, we adopt the DeepFloyd IF base model (Shonenkov et al., 2023), a diffusion model that operates at 64 × 64 resolution pixel space and better captures coarse geometry.

Progressive view training. However, directly deriving the free views in 360◦may still result in geometric artifacts, such as extra chair legs, due to the ambiguity inherent in a single reference image. To solve this, we progressively enlarge the training views, gradually propagating the wellestablished geometry to 360◦results.

Diffusion timestep annealing. To align with the coarse-to-fine progression of 3D optimization, we adopt a diffusion timestep annealing strategy similar to Huang et al. (2023). At the start of optimization, we prioritize sampling larger diffusion timestep t from the range [0.7,0.85] when computing Equation 6 to provide the global structure. As training proceeds, we linearly anneal the t sampling range to [0.2,0.5] over hundreds of iterations. This annealing strategy allows the model to first establish a plausible global geometry in the early optimization phase before refining the structural details.

Detailed structural enhancement. We initially optimize an implicit surface representation with the corresponding volume rendering as in NeuS (Wang et al., 2021) to establish the coarse structure. Then, following Lin et al. (2023), we use this result to initialize a textured 3D mesh representation using a deformable tetrahedral grid (DMTet) (Shen et al., 2021) to facilitate high-resolution details. Moreover, this representation disentangles the learning of geometry and texture. Hence, at the end of this structural enhancement, we are able to solely refine the texture and better preserve highfrequency details from the reference image.

- 4.2 TEXTURE BOOSTING VIA BOOTSTRAPPED SCORE SAMPLING

The geometry sculpting stage prioritizes the learning of coherent and detailed geometry but leaves the texture blurry. This is due to our reliance on a 2D prior model that operates at a coarse resolution, and the limited sharpness offered by the 3D-aware diffusion model. Additionally, texture issues such as over-smoothing and over-saturation arise from excessively large classifier-free guidance.

To augment the texture realism, we use variational score distillation (VSD) loss, as detailed in Equation 3. We switch to the Stable Diffusion model (Rombach et al., 2021) in this stage which offers high-resolution gradients. To promote realistic rendering, we exclusively optimize the mesh texture with the tetrahedral grid fixed. In this learning stage, we do not leverage the Zero-1-to-3 model as the 3D prior since it adversely impacts the texture quality. Nonetheless, the inconsistent textures may come back, resulting in bizarre 3D outcomes.

We observe that the multi-view renderings from the last stage, despite some blurriness, exhibit good

- 3D consistency. One idea is to adapt a pretrained 2D diffusion model using these rendering results, enabling the model to form a concept about the scene’s surrounding views. In light of this, we finetune the diffusion model with the multi-view image renderings {x}, using DreamBooth (Ruiz

- et al., 2023). Specifically, we incorporate the text prompts containing a unique identifier and the subject’s class name (e.g., “A [V] astronaut” in Figure 2). During finetuning, the camera parameter of each view is introduced as an additional condition. In practice, we train the DreamBooth with “augmented” image renderings, xr = rt′(x). We introduce Gaussian noises, in an amount specified by the diffusion timestep t′, to the multi-view renderings, i.e., xt′ = αt′x0 + σt′ϵ (αt′,σt′ > 0 are hyperparameters), which are restored using the diffusion model. By choosing a large t′, these augmented images reveal high-frequency details at the cost of the fidelity to the original renderings. The DreamBooth model trained on these augmented renderings can serve as a 3D prior to guide texture refinement.

Further, we propose to alternatively optimize the 3D scene to facilitate a bootstrapped optimization (Figure 2). Initially, the 3D mesh yields blurry multi-view renderings. We adopt a large diffusion t′ to augment their texture quality while introducing some 3D inconsistency. The DreamBooth model trained on these augmented renderings obtains a unified 3D concept of the scene to guide texture refinement. As the 3D mesh reveals finer textures, we reduce the diffusion noises introduced to the image renderings, so the DreamBooth model learns from more consistent renderings and better captures the image distribution faithful to evolving views. In this cyclic process, the 3D mesh and diffusion prior mutually improve in a bootstrapped manner. Formally, we derive the 3D optimization gradient using the following bootstrapped score distillation (BSD) loss:

∂x ∂θ

]. (7)

∇θLBSD(ϕ,g(θ)) = Et,ϵ,c[ω(t)(ϵDreamBooth(xt;y,t,rt′(x),c) − ϵlora(xt;y,t,x,c))

Contrary to prior works (Poole et al., 2022; Wang et al., 2023b) that distill the score function from a fixed 2D model, our BSD loss learns from an evolving model which becomes increasingly 3D consistent by drawing feedback from the ongoing crafted 3D model. In our experiments, we alternate the optimization twice, which suffices to produce consistent textures with rich details.

- 5 EXPERIMENTS

- 5.1 IMPLEMENTATION DETAILS

Architectural details. In the geometry sculpting stage, we use Neus and textured 3D mesh representations. We employ Instant NGP (M¨uller et al., 2022), optimizing from a 64 to a 384 resolution. For the textured mesh, we use DMTet at a 128 grid and 512 rendering resolution.

Optimization. During mesh refinement, we iteratively render a guided normal map and an RGB image, enhancing geometric detail and optimizing our texture prediction network for consistency. Considering the given plausible global geometric structure, our approach eschews the use of a 3D prior during texture optimization. We leverage random sampling of the camera radius and field-ofview (FOV) angles, aligning with Dreamfusion’s methodology. This results in improved texture and geometry details via alternate rendering of normal maps and RGB images.

- 5.2 COMPARISONS WITH THE STATE OF THE ARTS

Baselines. We conduct a comparative analysis of our technique against five baseline methods. The first three are text-to-3D methods: DreamFusion (Poole et al., 2022), Magic3D (Lin et al., 2023) and ProlificDreamer (Wang et al., 2023b). We also compare our method against two image-to-3d methods: Make-it-3D (Tang et al., 2023) and Magic123 (Qian et al., 2023). For DreamFusion, Magic3D,

Super Saiyan Goku unleashe a massive energy wave

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

A beagle in a detective's outfit

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

A Minion wearing the cloths of Spiderman

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Reference Ours Magic123 Make-It-3D DreamFusion Magic3D ProlificDreamer

- Figure 3: Qualitative comparison with baselines. Our method generates sharper and more plausible details in both geometry and texture. Note that our method generates rich texture detail at novel views and eliminates multi-face Janus problems.

Rendering Views

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Times

“3D Pixar Lionel Messi artfully kicking paint-filled bottles”

DreamBooth

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Reference

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Figure 4: Improved view consistency and texture fidelity along bootstrapping.

Magic123 and ProlificDreamer, we utilize their implementations in the Threestudio library (Guo et al., 2023) for comparison. For Make-it-3D, we use its official implementation.

Datasets. We establish a test benchmark that includes 300 images, which is a mix of real pictures and those produced by Stable Diffusion (Rombach et al., 2021) and Deep Floyd. Each image in this benchmark comes with an alpha mask for the foreground, a predicted depth map, and a text prompt. For real images, the text prompts are sourced from an image caption model. We intend to make this test benchmark accessible to the public.

Table 1: Quantitative comparison against prior 2D-to3D lifting methods. The metrics are measured on 300 generated samples.

CLIP ↑ Contextual ↓ PSNR ↑ LPIPS ↓ Make-it-3D 0.872 1.609 18.937 0.054

Magic123 0.843 1.628 22.838 0.053 DreamCraft3D 0.896 1.579 31.801 0.005

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

(a) w/o Zero1-to-3 (b) SDS (c) VSD (d) 1-round BSD (e) 2-round BSD

Figure 6: Ablation study of the effectiveness of 3D prior and our proposed BSD (Bootstrapped Score Distillation). (a) geometry sculpting stage without 3D prior. (b) texture optimization with SDS loss. (c) VSD loss produces richer texture detail while suffering from texture inconsistency. (d) BSD improves the texture consistency with one round DreamBooth. (e) Two-round BSD adds more details to the generated result.

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

“a DSLR photo of a corgi wearing a beret and holding a baguette, standing up on two hind legs”

Figure 7: Continual improvement of geometry and texture quality through multiple stages.

Quantitative comparison. To generate compelling 3D content that resembles the input image and consistently conveys semantics from various perspectives, we compare our technique with established baselines using a quantitative analysis. Our evaluation employed four metrics: LPIPS (Zhang et al., 2018) and PSNR for fidelity measurement at the reference viewpoint; Contextual Distance (Mechrez et al., 2018) for pixel-level congruence assessment; and CLIP score (Radford et al., 2021) to estimate semantic coherence. Results depicted in Table 1 indicate that our approach significantly surpasses the baselines in maintaining both texture consistency and fidelity.

User study. To substantiate the robustness and quality of our proposed model, we executed a user study employing 15 distinct pairs of prompts and images. Each participant was provided with four free-view rendering video alongside their corresponding textual input, and asked to choose their top preferred 3D model. The study gathered 480 responses from a total of 32 participants, the analysis of which is depicted in Figure 5. On an average basis, our model was favored by 92% of users over alternative models, outperforming the baselines by a large margin. This result provides compelling evidence of the resilience and superior quality inherent to our proposed method.

[Figure 98]

Qualitative comparison. Figure 3 compares our method with the baselines. All the text-to-3D methods suffer from multi-view consistency issues. While ProlificDreamer offers realistic textures, it fails to form a plausible 3D object. Image-to-3D methods like Make-it-3D create quality frontal views but struggle with geometry. Magic123, enhanced by Zero1-to-3, fares better in geometry regularization, but both generate overly smoothed textures and geometric details. In contrast, our Bootstrapped Score Distillation method improves imagination diversity while maintaining semantic consistency.

Figure 5: User study.

- 5.3 ANALYSIS

The effect of 3D prior. In our paper we claim that the guidance offered by a 3D prior enhances the generation of globally plausible geometry. To ascertain its impact, an ablation study is conducted, where the 3D prior is deactivated. Figure 6 demonstrates that, in the absence of the 3D prior, the resultant character tends to exhibit the multifaceted Janus issue and suffers from irregular geometry. This observation underlines the significance of a viewpoint-aware 3D prior in regulating a globally consistent shape.

The effect of BSD. Figure 6 also presents an ablation study encompassing three texture optimization techniques: (1) BSD, (2) VSD, and (3) Score Distillation Sampling (SDS) with the traditional stable diffusion. The application of SDS has been observed to generate novel-view textures that are excessively smooth and over-saturated. In contrast, while VSD using standard stable diffusion can produce realistic textures, it yields a notably high inconsistency. In contrast, our proposed approach successfully generates textures that strike a balance between realism and consistency.

Visualization of multiple stages. Figure 7 provides the visualization of the intermediate rendering results for each stage in our hierarchical pipeline. In the geometry sculpting stage, we convert Neus to DMTet to improve high-resolution geometry details. However, the improvement in texture is negligible. On the contrary, in the texture stage, we significantly improve the texture quality with our proposed BSD.

DreamBooth times. Figure 4 illustrates multi-view datasets for DreamBooth. The initial stage involves the introduction of substantial noise into each image to amplify detail richness, leading to inconsistent denoised images. However, as the textured mesh undergoes optimization, the produced renderings evolve towards increased consistency and photorealism, thereby enhancing the quality of the input dataset tailored for DreamBooth.

- 6 CONCLUSION

We have presented DreamCraft3D, an innovative approach that advances the field of complex 3D asset generation. This work introduces a meticulous geometry sculpting phase for producing plausible and coherent 3D geometries and a novel Bootstrapped Score Distillation strategy. The latter, by distilling from an optimizing 3D-aware diffusion prior and adapting to multi-view renderings of the instance being optimized, significantly improves texture quality and consistency. DreamCraft3D produces high-fidelity 3D assets with compelling texture details and multi-view consistency. We believe this work represents an important step towards democratizing 3D content creation and shows great promise in future applications.

REFERENCES

Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, et al. Gaudi: A neural architect for immersive 3d scene generation. Advances in Neural Information Processing Systems, 35:25102–25116, 2022.

Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5799–5809, 2021.

Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16123–16133, 2022.

Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023.

Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4456–4465, 2023.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13142–13153, 2023.

Congyue Deng, Chiyu Jiang, Charles R Qi, Xinchen Yan, Yin Zhou, Leonidas Guibas, Dragomir Anguelov, et al. Nerdi: Single-view nerf synthesis with language-guided diffusion as general image priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20637–20647, 2023.

Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10786–10796, 2021.

Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. Advances In Neural Information Processing Systems, 35:31841–31854, 2022.

Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10696–10706, 2022.

Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, ChiaHao Chen, Zi-Xin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/ threestudio, 2023.

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023.

Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. ACM Transactions on Graphics (TOG), 41(4):1–19, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Yukun Huang, Jianan Wang, Yukai Shi, Xianbiao Qi, Zheng-Jun Zha, and Lei Zhang. Dreamtime: An improved optimization strategy for text-to-3d content creation. arXiv preprint arXiv:2306.12422, 2023.

Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. 2022.

Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023.

Han-Hung Lee and Angel X Chang. Understanding pure clip guidance for voxel grid nerf models. arXiv preprint arXiv:2209.15172, 2022.

Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusion-sdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12642–12651, 2023.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023a.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. arXiv preprint arXiv:2303.11328, 2023b.

Roey Mechrez, Itamar Talmi, and Lihi Zelnik-Manor. The contextual loss for image transformation with non-aligned data. In Proceedings of the European conference on computer vision (ECCV), pp. 768–783, 2018.

Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12663–12673, 2023.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

Paritosh Mittal, Yen-Chi Cheng, Maneesh Singh, and Shubham Tulsiani. Autosdf: Shape priors for 3d completion, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 306–315, 2022.

Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1–102:15, July 2022. doi: 10.1145/3528223.3530127. URL https://doi.org/10.1145/3528223. 3530127.

Gimin Nam, Mariem Khlifi, Andrew Rodriguez, Alberto Tono, Linqi Zhou, and Paul Guerrero. 3d-ldm: Neural implicit 3d shape generation with latent diffusion models. arXiv preprint arXiv:2212.00842, 2022.

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, HsinYing Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508, 2023.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22500– 22510, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Kamal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18603–18613, 2022.

Kyle Sargent, Jing Yu Koh, Han Zhang, Huiwen Chang, Charles Herrmann, Pratul Srinivasan, Jiajun Wu, and Deqing Sun. Vq3d: Learning a 3d-aware generative model on imagenet. arXiv preprint arXiv:2302.06833, 2023.

Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems, 34:6087–6101, 2021.

Alex Shonenkov, Misha Konstantinov, Daria Bakshandaeva, Christoph Schuhmann, Ksenia Ivanova, and Nadiia Klokova. DeepFloyd IF: A modular cascaded diffusion model. https://github. com/deep-floyd/IF/tree/develop, 2023.

Ivan Skorokhodov, Aliaksandr Siarohin, Yinghao Xu, Jian Ren, Hsin-Ying Lee, Peter Wonka, and Sergey Tulyakov. 3d generation on imagenet. arXiv preprint arXiv:2303.01416, 2023.

Jingxiang Sun, Xuan Wang, Yichun Shi, Lizhen Wang, Jue Wang, and Yebin Liu. Ide-3d: Interactive disentangled editing for high-resolution 3d-aware portrait synthesis. ACM Trans. Graph., 41(6), nov 2022. ISSN 0730-0301. doi: 10.1145/3550454.3555506. URL https://doi.org/10. 1145/3550454.3555506.

Jingxiang Sun, Xuan Wang, Lizhen Wang, Xiaoyu Li, Yong Zhang, Hongwen Zhang, and Yebin Liu. Next3d: Generative neural texture rasterization for 3d-aware head avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20991–21002, June 2023.

Junshu Tang, Bo Zhang, Binxin Yang, Ting Zhang, Dong Chen, Lizhuang Ma, and Fang Wen. Explicitly controllable 3d-aware portrait generation. arXiv preprint arXiv:2209.05434, 2022.

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023.

Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. arXiv preprint arXiv:2212.00774, 2022.

Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021.

Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4563–4573, 2023a.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023b.

Daniel Watson, William Chan, Ricardo Martin-Brualla, Jonathan Ho, Andrea Tagliasacchi, and Mohammad Norouzi. Novel view synthesis with diffusion models. arXiv preprint arXiv:2210.04628, 2022.

Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 803–814, 2023.

Jianfeng Xiang, Jiaolong Yang, Binbin Huang, and Xin Tong. 3d-aware image generation using 2d diffusion models. arXiv preprint arXiv:2303.17905, 2023.

Chulin Xie, Chuxin Wang, Bo Zhang, Hao Yang, Dong Chen, and Fang Wen. Style-based point generator with adversarial rendering for point cloud completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4619–4628, 2021.

Xingguang Yan, Liqiang Lin, Niloy J Mitra, Dani Lischinski, Daniel Cohen-Or, and Hui Huang. Shapeformer: Transformer-based shape completion via sparse representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6239–6249, 2022.

Wang Yu, Xuelin Qian, Jingyang Huo, Tiejun Huang, Bo Zhao, and Yanwei Fu. Pushing the limits of 3d shape generation at scale. arXiv preprint arXiv:2306.11510, 2023.

Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. arXiv preprint arXiv:2210.06978, 2022.

Biao Zhang, Matthias Nießner, and Peter Wonka. 3dilg: Irregular latent grids for 3d generative modeling. In NeurIPS, 2022.

Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. arXiv preprint arXiv:2301.11445, 2023a.

Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, and Min Zheng. Avatarverse: High-quality & stable 3d avatar creation from text and pose. arXiv preprint arXiv:2308.03610, 2023b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

A APPENDIX

- A.1 IMPLEMENTATION DETAILS

Algorithms for Bootstrapped Score Distillation. We provide a summarized algorithm of bootstrapped score distillation in Algorithm 1. The “Bootstrapped Score Distillation” algorithm starts by initializing n (n = 1 in our case) meshes and a pretrained text-to-image diffusion model ϵDreamBooth parameterized by ϕ. The algorithm then enters an iterative loop: in each iteration, it renders the mesh to obtain multi-view images x, augments these images with Gaussian noises to form xt′ = αt′x + σt′ϵ, and fine-tunes the pretrained diffusion model ϵDreamBooth based on these augmented image renderings. Within each iteration, there’s an inner loop running for n steps, where a random mesh and camera pose are sampled, and a 2D image is rendered from the chosen pose. Then, updates are performed on θ and ϕ using gradients calculated from the difference between the pretrained score function and the predicted score function, and from the L2 norm between the predicted score and real noise, respectively. These iterations continue until convergence, and the final refined mesh(s) are returned.

Structure-aware latent regularization. To maintain the high-quality output produced by BSD while reducing noise and inconsistencies, we further incorporate a control net-guided inpainting

Algorithm 1 Bootstrapped Score Distillation

Input: Number of particles n (≥ 1). Pretrained text-to-image diffusion model ϵDreamBooth. Learning rate η1 and η2 for 3D structures and diffusion model parameters, respectively. A prompt y. Number

of images m and Camera poses {c(ri)}mi=1 for the multi-view datasets.

- 1: initialize n meshes {θ(i)}ni=1, a noise prediction model ϵϕ parameterized by ϕ.
- 2: while not converged do
- 3: Render the mesh to get multi-view images x = g(θ,cr).
- 4: Augment image renderings with Gaussian noises: xt′ = αt′x + σt′ϵ.
- 5: Finetune ϵDreamBooth on augmented image renderings xr = rt′(x).
- 6: for i in T steps do
- 7: Randomly sample θ ∼ {θ(i)}ni=1 and a camera pose c.
- 8: Render the 3D structure θ at pose c to get a 2D image x0 = g(θ,c).
- 9: θ ← θ − η1Et,ϵ,c ω(t)(ϵDreamBooth(xt,t,y) − ϵϕ(xt,t,c,y)) ∂g∂θ(θ,c)

- 10: ϕ ← ϕ − η2∇ϕEt,ϵ||ϵϕ(xt,t,c,y) − ϵ||22.
- 11: end for
- 12: end while
- 13: return

diffusion model that regularizes the generated textures. Specifically, for a rendered image x from an arbitrary viewpoint, the visible section under the reference view is initially computed. This invariant portion during the generation process allows our inpainting model to fill in the remaining segments. As these remaining parts adhere to geometric constraints, we integrate geometric normal information through a control net. Ultimately, this method permits us to enforce view consistency and generate realistic results using a control-net guided inpainting diffusion model. To preserve the high-quality generation output, we avoid utilizing this image directly as a loss against the rendered image. Instead, we subtly introduce it by constraining the norm of the latent variables:

#### Lreg(ϕ,g(θ)) = Σ(∥E(x)∥2 − ∥E(xreg)∥2)2. (8)

Architectural details. In the Neus approach (Wang et al., 2021), we employ a single-layer MultiLayer Perceptron (MLP) with 32 hidden units to simultaneously predict RGB color, volume density, and normal. The inputs to this MLP are the concatenated feature vectors derived from multiresolution hash encoding sampled with trilinear interpolation. To sparsify the Instant NGP representation, we implement density-based pruning every 10 iterations within an octree structure, as suggested by Magic3D (Lin et al., 2023). In our experiments, we use a bounding sphere with a radius of 2. For the density prediction, we utilise the softplus activation function and, following the approach of Poole et al. 2022, include an initial spatial density bias in order to encourage optimization in favor of the object-centric neural field.

Camera and light augmentations. We follow Magic3D to add random augmentations to the camera and light sampling for rendering the shaded images. Differently, we sample the point light location such that the angular distance from the random camera center location (w.r.t. the origin) is sampled from ϕcam ∼ U(0,π/3) with a random point light distance rcam ∼ (7.5,10), and (b) we freeze the material augmentation unlike Dreamfusion and Magic3D, as we found it is bad for training convergence (c) In the coarse neus stage, we propose a fixed-random mixed camera pose strategy. Specifically, following the common practice of the current text-to-3D methods, random camera view sampling benefits scene optimization. However, Zero1-to-3 needs fixed camera intrinsic parameters. Therefore, we let half of the GPUs sample the camera distance from U(3.2,3.5), and the Field-of-View from U(10,20), while the left GPUs are fixed to the default camera intrinsic.

Time annealing. At the beginning of the geometry sculpting stage, We utilize a simple two-stage annealing of time step t in the score distillation objective. For the first several iterations we sample time steps t ∼ U(0.7,0.85) and then anneal into t ∼ U(0.2,0.50). We refer the readers to ProlificDreamer (Wang et al., 2023b). For the left iterations, we fix time steps to U(0.2,0.50). We also utilize a simple two-stage time annealing for the multi-view dataset generation, that is, for the first updating step, we select a time step t = 0.5 for all rendered images and then anneal it into t = 0.1 along the later updating steps.

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

“a DSLR photo of a bagel filled with cream cheese and lox”

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Figure 8: DreamCraft3D skillfully generates an assortment of visually compelling 3D models when provided with a textual description.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Figure 9: Failure case. Our method learns incorrect geometry for elephant nose.

- A.2 ADDITIONAL EXPERIMENTS

Diversity. Prior studies frequently yield models of limited diversity with disproportionately smooth textures. Our approach to superior text-to-3D generation initially translates the text prompt into a reference image via 2D diffusion before implementing our proprietary image-based 3D creation methodology. Figure 8 demonstrates the proficiency of our method in generating an array of diverse models from a single text prompt, all characterized by their remarkable quality.

- A.3 LIMITATIONS

Our approach occasionally incorporates frontal-view geometric details into texture, as depicted in Figure 9, due to depth ambiguity and inaccuracies in the depth prior. Furthermore, we do not expressly segregate material and lighting from the 2D reference image, an aspect deferred for future exploration.

- A.4 ADDITIONAL QUALITATIVE RESULTS

Figure 10−Figure 13 provides more results produced by DreamCraft3D. Our method is able to produce photo-realistic 3D assets with compelling textural details. Moreover, our method shows significantly improved 3D consistency. Please find video results in the supplementary video.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Mictlantecuhtli

A brightly colored mushroom growing on a log

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

President Obama

Lebron James wearing Lakers jersey, holding_a basketball

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

A DSLR photo of a delicious chocolate brownie dessert with ice cream on the side

Wes Anderson style Red Panda, reading a book, super cute, highly detailed and colored

Figure 10: Additional results of DreamCraft3D.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Super Saiyan Goku unleashes a massive energy wave while standing on top of a mountain

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Mech robot with large weapons on top with hexagonal bases

Super Mario

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

3d stylized game little building 3d render of a statue of an astronaut

Figure 11: Additional results of DreamCraft3D.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

A delicious hamburger

A blue jay standing on a large basket of rainbow macarons

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

A DSLR photo of a plate of fried chicken and waffles with maple syrup on them

An ice cream cone

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Nike Lebron 10 "Cork" Sneakers Nike “Air Jordan 1” Sneakers

Figure 12: Additional results of DreamCraft3D.

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Cute asuka anime figure, 3d render, full body, with a black handbag

Baby yoda in the style of Mormookiee

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

A banana peeling itself

[Figure 199]

[Figure 200]

Teddy with sunglasses and swimming trunks

Realistic photo of pistachio mousse cake

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

A DSLR photo of a bear dressed in medieval armor Neymar 3d character cartoon disney pixar render

Figure 13: Additional results of DreamCraft3D.

