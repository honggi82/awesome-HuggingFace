# arXiv:2307.04787v1[cs.CV]4Jul2023

## Collaborative Score Distillation for Consistent Visual Synthesis

Subin Kim∗1 Kyungmin Lee∗1 June Suk Choi1 Jongheon Jeong1 Kihyuk Sohn2 Jinwoo Shin1 1KAIST 2Google Research ∗{subin-kim, kyungmnlee}@kaist.ac.kr

### Abstract

Generative priors of large-scale text-to-image diffusion models enable a wide range of new generation and editing applications on diverse visual modalities. However, when adapting these priors to complex visual modalities, often represented as multiple images (e.g., video), achieving consistency across a set of images is challenging. In this paper, we address this challenge with a novel method, Collaborative Score Distillation (CSD). CSD is based on the Stein Variational Gradient Descent (SVGD). Specifically, we propose to consider multiple samples as “particles” in the SVGD update and combine their score functions to distill generative priors over a set of images synchronously. Thus, CSD facilitates seamless integration of information across 2D images, leading to a consistent visual synthesis across multiple samples. We show the effectiveness of CSD in a variety of tasks, encompassing the visual editing of panorama images, videos, and 3D scenes. Our results underline the competency of CSD as a versatile method for enhancing inter-sample consistency, thereby broadening the applicability of text-to-image diffusion models.2

### 1 Introduction

Text-to-image diffusion models [1, 2, 3, 4] have been scaled up by using billions of image-text pairs [5, 6] and efficient architectures [7, 8, 9, 4], showing impressive capability in synthesizing high-quality, realistic, and diverse images with the text given as an input. Furthermore, they have branched into various applications, such as image-to-image translation [10, 11, 12, 13, 14, 15, 16], controllable generation [17], or personalization [18, 19]. One of the latest applications in this regard is to translate the capability into other complex modalities, viz., beyond 2D images [20, 21] without modifying diffusion models using modality-specific training data. This paper focus on the problem of adapting the knowledge of pre-trained text-to-image diffusion models to more complex highdimensional visual generative tasks beyond 2D images without modifying diffusion models using modality-specific training data.

We start from an intuition that many complex visual data, e.g., videos and 3D scenes, are represented as a set of images constrained by modality-specific consistency. For example, a video is a set of frames requiring temporal consistency, and a 3D scene is a set of multi-view frames with view consistency. Unfortunately, image diffusion models do not have a built-in capability to ensure consistency between a set of images for synthesis or editing because their generative sampling process does not take into account the consistency when using the image diffusion model as is. As such, when applying image diffusion models on these complex data without consistency in consideration, it results in a highly incoherent output, as in Figure 2 (Patch-wise Crop), where one can easily identify where images

∗Equal contribution 2Visualizations are available at the website https://subin-kim-cv.github.io/CSD.

Preprint. Under review.

[Figure 1]

[Figure 2]

Parametric generator, Source,

Beyond a single 2D image with fixed resolution e.g., panorama image, video, 3D scene

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

“Turn it into a Van

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Source

Gogh style painting”

Diffusion U-Net

Diffusion U-Net

Diffusion U-Net

Diffusion U-Net

Diffusion U-Net

Diffusion U-Net

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

Collaborative Score Distillation (CSD) Backpropagate

[Figure 61]

Edited with CSD

- Figure 1: Method overview. CSD-Edit enables various visual-to-visual translations with two novel components. First, a new score distillation scheme using Stein variational gradient descent, which considers inter-sample relationships (Section 3.1) to synthesize a set of images while preserving modality-specific consistency constraints. Second, our method edits images with minimal information given from text instruction by subtracting image-conditional noise estimate instead of random noise during score distillation (Section 3.2). By doing so, CSD-Edit is used for text-guided manipulation of various visual domains, e.g., panorama images, videos, and 3D scenes (Section 3.3).

are stitched. Such behaviors are also reported in video editing, thus, recent works [22, 23, 24, 25] propose to handle video-specific temporal consistency when using the image diffusion model.

Here, we take attention to an alternative approach, Score Distillation Sampling (SDS) [26], which enables the optimization of arbitrary differentiable operators by leveraging the rich generative prior of text-to-image diffusion models. SDS poses generative sampling as an optimization problem by distilling the learned diffusion density scores. While Poole et al. [26] has shown the effectiveness of SDS in generating 3D objects from the text by resorting on Neural Radience Fields [27] priors which inherently suppose coherent geometry in 3D space by density modeling, it has not been studied for consistent visual synthesis of other modalities.

In this paper, we propose Collaborative Score Distillation (CSD), a simple yet effective method that extends the singular of the text-to-image diffusion model for consistent visual synthesis. The crux of our method is two-fold: first, we establish a generalization of SDS by using Stein variational gradient descent (SVGD), where multiple samples share their knowledge distilled from diffusion models to accomplish inter-sample consistency. Second, we present CSD-Edit, an effective method for consistent visual editing by leveraging CSD with Instruct-Pix2Pix [14], a recently proposed instruction-guided image diffusion model (See Figure 1).

We demonstrate the versatility of our method in various applications such as panorama image editing, video editing, and reconstructed 3D scene editing. In editing a panorama image, we show that CSD-Edit obtains spatially consistent image editing by optimizing multiple patches of an image. Also, compared to other methods, our approach achieves a better trade-off between source-target image consistency and instruction fidelity. In video editing experiments, CSD-Edit obtains temporal consistency by taking multiple frames into optimization, resulting in temporal frame-consistent video editing. Furthermore, we apply CSD-Edit to 3D scene editing and generation, by encouraging consistency among multiple views.

### 2 Preliminaries

#### 2.1 Diffusion models

Generative modeling with diffusion models consists of a forward process q that gradually adds Gaussian noise to the input x0 ∼ pdata(x), and a reverse process p which gradually denoises from the Gaussian noise xT ∼ N(0,I). Formally, the forward process q(xt|x0) at timestep t is given by q(xt|x0) = N(xt;αtx0,σt2I), where σt and αt2 = 1 − σt2 are pre-defined constants designed for

[Figure 62]

[Figure 63]

Source Instruct-Pix2Pix, Patch-wise Crop

[Figure 64]

[Figure 65]

Instruct-Pix2Pix + MultiDiffusion, =7.5 Instruct-Pix2Pix + MultiDiffusion, =15

ωy

ωy

[Figure 66]

[Figure 67]

[Figure 68]

CSD-Edit (Ours), =7.5 CSD-Edit (Ours), =15

ωy

ωy

“Turn it into a Van-Gogh style painting”

- Figure 2: Panorama image editing. (Top right) Instruct-Pix2Pix [14] on cropped patches results in inconsistent image editing. (Second row) Instruct-Pix2Pix with MultiDiffusion [28] edits to consistent

image, but less fidelity to the instruction, even with high guidance scale ωy. (Third row) CSD-Edit provides consistent image editing with better instruction-fidelity by setting proper guidance scale.

effective modeling [8, 29, 30]. Given enough timesteps, reverse process p also becomes a Gaussian and the transitions are given by posterior q with optimal MSE denoiser [31], i.e., pϕ(xt−1|xt) = N(xt−1;xt − xˆϕ(xt;t),σt2I), where xˆϕ(xt;t) is a learned optimal MSE denoiser. Ho et al. [7] proposed to train an U-Net [32] autoencoder ϵϕ(xt;t) by minimizing following objective:

LDiff(ϕ;x) = Et∼U(0,1),ϵ∼N(0,I) w(t)∥ϵϕ(xt;t) − ϵ∥22 , xt = αtx0 + αtϵ (1) where w(t) is a weighting function for each timestep t. Text-to-image diffusion models [1, 2, 4, 3] are trained by Eq. (1) with ϵϕ(xt;y,t) that estimates the noise conditioned on the text prompt y. At inference, those methods rely on Classifier-free Guidance (CFG) [33], which allows higher quality sample generation by introducing additional parameter ωy ≥ 1 as follows:

ϵωϕ(xt;y,t) = ϵϕ(xt;t) + ωy ϵϕ(xt;y,t) − ϵϕ(xt;t) (2) By setting the appropriate guidance scale ωy > 0, one can improve fidelity to the text prompt at the cost of diversity. Throughout the paper, we refer pωϕy(xt;y,t) a conditional distribution of a text y.

Instruction-based image editing by Instruct-Pix2Pix. Recently, many works have demonstrated the capability of diffusion models in editing or stylizing images [10, 13, 11, 12, 14]. Among them, Brooks et al. [14] proposed Instruct-Pix2Pix, where they finetuned Stable Diffusion [4] models with the source image, text instruction, edited image (edited by Prompt-to-Prompt [12]) triplet to enable instruction-based editing of an image. Given source image x˜ and instruction y, the noise estimate at time t is given as

ϵωϕs,ωy(xt;x˜,y,t) = ϵϕ(xt;t) + ωs ϵϕ(xt;x˜,t) − ϵϕ(xt;t)

(3)

+ ωy ϵϕ(xt;x˜,y,t) − ϵϕ(xt;x˜,t) ,

where ωy is CFG parameter for text as in Eq. (2) and ωs is an additional CFG parameter that controls the fidelity to the source image x˜.

#### 2.2 Score distillation sampling

Poole et al. [26] proposed Score Distillation Sampling (SDS), an alternative sample generation method by distilling the rich knowledge of text-to-image diffusion models. SDS allows optimization

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

CSD-Edit(Ours)FateZeroPix2VideoSourceGen-1

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

“Make it as a painting of Claude Monet”

- Figure 3: Video editing. Qualitative results on the lucia video in DAVIS 2017 [34]. CSD shows frame-wise consistent editing providing coherent content across video frames e.g., consistent color and background without changes in person. Compared to Gen-1 [21], a video editing method trained on a large video dataset, CSD-Edit shows high-quality video editing results reflecting given prompts.

of any differentiable image generator, e.g., Neural Radiance Fields [27] or the image space itself. Formally, let x = g(θ) be an image rendered by a differentiable generator g with parameter θ, then SDS minimizes density distillation loss [35] which is KL divergence between the posterior of x = g(θ) and the text-conditional density pωϕ:

LDistill θ;x = g(θ) = Et,ϵ αt/σt DKL q xt|x = g(θ) ∥pωϕ(xt;y,t) . (4) For an efficient implementation, SDS updates the parameter θ by randomly choosing timesteps t ∼ U(tmin,tmax) and forward x = g(θ) with noise ϵ ∼ N(0,I) to compute the gradient as follows:

∂x ∂θ

∇θLSDS θ;x = g(θ) = Et,ϵ w(t) ϵωϕ(xt;y,t) − ϵ

. (5)

Remark that the U-Net Jacobian ∂ϵωϕ(zt;y,t)/∂zt is omitted as it is computationally expensive to compute, and degrades performance when conditioned on small noise levels. The range of timesteps

tmin and tmax are chosen to sample from not too small or large noise levels, and the guidance scales are chosen to be larger than those used for image generation.

#### 2.3 Stein variational gradient descent

The original motivation of Stein variational gradient descent (SVGD) [36] is to solve a variational inference problem, where the goal is to approximate a target distribution from a simpler distribution by minimizing KL divergence. Formally, suppose p is a target distribution with a known score function ∇x log p(x) that we aim to approximate, and q(x) is a known source distribution. Liu and Wang [36] showed that the steepest descent of KL divergence between q and p is given as follows:

Eq(x) f(x)⊤∇x log p(x) + Tr(∇xf(x)) , (6)

where f : RD → RD is any smooth vector function that satisfies lim∥x∥→∞ p(x)f(x) = 0. Remark that Eq. (6) becomes zero if we replace q(x) with p(x) in the expectation term, which is known as

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

CSD-Edit(Ours)Instruct-NeRF2NeRF

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Original NeRF (b)“Make him smile”

(a)“What if he were an anime character?”

- Figure 4: 3D NeRF scene editing. Visualizing novel-views of edited Fangzhou NeRF scene [38]. CSD-Edit leads to high-quality editing of 3D scenes and better preserves semantics of source scenes, e.g., obtains sharp facial details (left) and makes him smile without giving beard (right).

Stein’s identity [37]. Here, the choice of the critic f is crucial in its convergence and computational tractability. To that end, Liu and Wang [36] proposed to constrain f in the Reproducing Kernel Hilbert Space (RKHS) which yields a closed-form solution. Specifically, given a positive definite kernel k : RD × RD → R+, Stein variational gradient descent provides the greedy directions as follows:

x ← x − η∆x, ∆x = Eq(x′) k(x,x′)∇x′ log p(x′) + ∇x′k(x,x′) , (7)

with small step size η > 0. The SVGD update in Eq. (7) consists of two terms that play different roles: the first term moves the particles towards the high-density region of target density p(x), where the direction is smoothed by kernels of other particles. The second term acts as a repulsive force that prevents the mode collapse of particles. One can choose different kernel functions, while we resort to

standard Radial Basis Function (RBF) kernel k(x,x′) = exp(−h1∥x−x′∥22) with bandwidth h > 0.

### 3 Method

In this section, we introduce Collaborative Score Distillation (CSD) for consistent synthesis and editing of multiple samples. We first derive a collaborative score distillation method using Stein variational gradient descent (Section 3.1) and propose an effective image editing method using CSD, i.e., CSD-Edit, that leads to coherent editing of multiple images with instruction (Section 3.2). Lastly, we present various applications of CSD-Edit in editing panorama images, videos, and 3D scenes (Section 3.3).

#### 3.1 Collaborative score distillation

Suppose a set of parameters {θi}Ni=1 that generates images x(i) = g(θi). Similar to SDS, our goal is to update each θi by distilling the smoothed densities from the diffusion model by minimizing KL divergence in Eq. (4). On the contrary, CSD solves Eq. (4) using SVGD demonstrated in Section 2.3

so that each θi can be updated in sync with updates of other parameters in the set {θi}Ni=1. At each update, CSD samples t ∼ U(tmin,tmax) and ϵ ∼ N(0,I), and update each θi as follows:

N

∂x(i) ∂θi

w(t) N

k(x(tj),x(ti))(ϵωϕ(x(tj);y,t) − ϵ) + ∇x(j)

k(x(tj),x(ti))

, (8)

∇θiLCSD θi =

t

j=1

for each i = 1,2,...,N. We refer to Appendix A for full derivation. Note CSD is equivalent to SDS in Eq. (5) when N = 1, showing that CSD is a generalization of SDS to multiple samples. As the pairwise kernel values are multiplied by the noise prediction term, each parameter update on θi is affected by other parameters, i.e., the scores are mixed with importance weights according to the affinity among samples. The more similar samples tend to exchange more score updates, while different samples tend to interchange the score information less. The gradient of the kernels acts as a repulsive force that prevents the mode collapse of samples. Moreover, we note that Eq. (8) does not make any assumption on the relation between θi’s or their order besides them being a set of images

to be synthesized coherently with each other. As such, CSD is also applicable to arbitrary image generators, as well as text-to-3D synthesis in DreamFusion [26], which we compare in Section 4.4.

#### 3.2 Text-guided editing by collaborative score distillation

In this section, we introduce a text-guided visual editing method using Collaborative Score Distillation (CSD-Edit). Given source images x˜(i) =g(θ˜i) with parameters θ˜i, we optimize new target parameters {θi}Ni=1 with x(i) =g(θi) such that 1) each x(i) follows the instruction prompt, 2) preserves the semantics of source images as much as possible, and 3) the obtained images are consistent with each other. To accomplish these, we update each parameter θi, initialized with θ˜i, using CSD with noise estimate ϵωϕy,ωs of Instruct-Pix2Pix. However, this approach often results in blurred outputs, leading to the loss of details of the source image (see Figure 7). This is because the score distillation term subtracts random noise ϵ, which perturbs the undesirable details of source images.

We handle this issue by adjusting the noise prediction term that enhances the consistency between source and target images. Subtracting a random noise ϵ in Eq. (5) when computing the gradient is a crucial factor, which helps optimization by reducing the variance of a gradient. Therefore, we amend the optimization by changing the random noise into a better baseline function. Since our goal is to edit an image with only minimal information given text instructions, we set the baseline by the image-conditional noise estimate of the Instruct-Pix2Pix model without giving text instructions on the source image. To be specific, our CSD-Edit is given as follows:

N

∂x(i) ∂θi

w(t) N

k(x(tj),x(ti))∆E(ti) + ∇x(j)

k(x(tj),x(ti))

∇θiLCSD−Edit θi =

,

(9)

t

j=1

∆E(ti) = ϵωϕy,ωs(x(ti);x˜,y,t) − ϵω

ϕ (x˜(ti);x˜,t).

s

In Section 4.4, we validate our findings on the effect of baseline noise on image editing performance. We notice that CSD-Edit presents an alternative way to utilize Instruct-Pix2Pix in image-editing without any finetuning of diffusion models, by posing an optimization problem.

#### 3.3 CSD-Edit for various complex visual domains

Panorama image editing. Diffusion models are usually trained on a fixed resolution (e.g., 512×512 for Stable Diffusion [4]), thus when editing a panorama image (i.e., an image with a large aspect ratio), the editing quality significantly degrades. Otherwise, one can crop an image into smaller patches and apply image editing on each patch. However this results in spatially inconsistent images (see Figure 2, Patch-wise Crop, Appendix E). To that end, we propose to apply CSD-Edit on patches to obtain spatially consistent editing of an image, while preserving the semantics of source image. Following [28], we sample patches of size 512×512 that overlap using small stride and apply CSD-Edit on the latent space of Stable Diffusion [4]. Since we allow overlapping, some pixels might be updated more frequently. Thus, we normalize the gradient of each pixel by counting the appearance.

Video editing. Editing a video with an instruction should satisfy the following: 1) temporal consistency between frames such that the degree of changes compared to the source video should be consistent across frames, 2) ensuring that desired edits in each edited frame are in line with the given prompts while preserving the original structure of source video, and 3) maintaining the sample quality in each frame after editing. To meet these requirements, we randomly sample a batch of frames and update them with CSD-Edit to achieve temporal consistency between frames.

3D scene editing. We consider editing a 3D scene reconstructed by a Neural Radiance Field (NeRF) [27], which represents volumetric 3D scenes using 2D images. To edit reconstructed

- 3D NeRF scenes, it is straightforward to update the training views with edited views and finetune the NeRF with edited views. Here, the multi-view consistency between edited views should be considered since inconsistencies between edits across multiple viewpoints lead to blurry and undesirable artifacts, hindering the optimization of NeRF. To mitigate this, Haque et al. [39] proposed Instruct-NeRF2NeRF, which performs editing on a subset of training views and updates them sequentially at training iteration with intervals. However, image-wise editing results in inconsistencies between views, thus they rely on the ability of NeRF in achieving multi-view consistency. Contrary

0.9

CLIPImageSimilarity

0.8

0.7

CSD-Edit (Ours)

0.6

Instruct-Pix2Pix + MultiDiffusion

Instruct-Pix2Pix, Downscaling

0.05 0.10 0.15 0.20 0.25 CLIP Directional Similarity

- Figure 5: Panorama image editing. Comparison of CSD-Edit with baselines at different guidance scales ωy ∈ {3.0,5.0,7.5,10.0}.

- Table 1: Video editing. Quantitative comparison of CSD-Edit with baselines on video editing. Bold indicates the best results.

CLIP Directional CLIP Image LPIPS Similarity ↑ Consistency ↑ ↓

FateZero [22] 0.314 0.948 0.267 Pix2Vid [25] 0.230 0.949 0.283

CSD-Edit (Ours) 0.320 0.957 0.236

- Table 2: 3D scene editing. Quantitative comparison of CSD-Edit with baselines on 3D scene editing. Bold indicates the best results.

CLIP Directional CLIP Image LPIPS Similarity ↑ Consistency ↑ ↓

IN2N [14] 0.230 0.994 0.048 CSD-Edit (Ours) 0.239 0.995 0.043

to Instruct-NeRF2NeRF, we update the dataset with multiple consistent views through CSD-Edit, which serves as better training resources for NeRF, leading to less artifacts and better preservation of source 3D scene.

### 4 Experiments

#### 4.1 Text-guided panorama image editing

For the panorama image-to-image translation task, we compare CSD-Edit with different versions of Instruct-Pix2Pix: one is which using naive downsizing to 512×512 and performing Instruct-Pix2Pix, and another is updating Instruct-Pix2Pix on the patches as in MultiDiffusion [28] (Instruct-Pix2Pix + MultiDiffusion). For comparison, we collect a set of panorama images (i.e., which aspect ratio is higher than 3), and edit each image to various artistic styles and different guidance scales ωy. For evaluation, we use pre-trained CLIP [40] to measure two different metrics: 1) consistency between source and target images by computing similarity between two image embeddings, and 2) CLIP directional similarity [41] which measures how the change in text agrees with the change in the images. The experimental details are in Appendix D.1.

- In Figure 5, we plot the CLIP scores of different image editing methods with different guidance scales. We notice that CSD-Edit provides the best trade-off between the consistency between source and target images and fidelity to the instruction. Figure 2 provides a qualitative comparison between panorama image editing methods. Remark that Instruct-Pix2Pix + MultiDiffusion is able to generate spatially consistent images, however, the edited images show inferior fidelity to the text instruction even when using a large guidance scale. Additional qualitative results are in Appendix E.

#### 4.2 Text-guided video editing

For the video editing experiments, we primarily compare CSD-Edit with existing zero-shot video editing schemes that employ text-to-image diffusion models such as FateZero [22], and Pix2Video [25]. To emphasize the effectiveness of CSD-Edit against learning-based schemes, we also compare it with Gen-1 [21], a state-of-the-art video editing method trained on a large-scale video dataset. For quantitative evaluation, we report CLIP image-text directional similarity as in Section 4.1 to measure alignment between changes in texts and images. Also, we measure CLIP image consistency and LPIPS [42] between consecutive frames to evaluate temporal consistency. We utilize video sequences from the popular DAVIS [34] dataset at a resolution of 1920 × 1080. Please refer to Appendix D.2 for a detailed description of the baseline methods and experimental setup.

Table 1 summarize quantitative comparison between CSD-Edit and the baselines. We notice that CSD-Edit consistently outperforms the existing zero-shot video editing schemes in terms of both temporal consistency and fidelity to given text prompts. Moreover, Figure 3 qualitatively demonstrate the superiority of CSD over the baselines on video-stylization and object-aware editing tasks. Impressively, CSD shows comparable editing performance to Gen-1 even without training on a large-scale video dataset and any architectural modification to the diffusion model. Additional qualitative results are in Appendix E.

Text-to-3D Generation Text-to-2D Generation

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

SDS CSD (Ours)

SDS CSD (Ours)

“a DSLR photo of a corgi wearing a top hat”

- Figure 6: Text-to-3D generation. (Left) CSD helps capturing coherent geometry in synthesizing 3D object. (Right) CSD generates coherent images conditioned on view-dependent prompts.

#### 4.3 Text-guided 3D scene editing

For the text-guided 3D scene editing experiments, we mainly compare our approach with InstuctNeRF2NeRF (IN2N) [39]. For a fair comparison, we exactly follow the experimental setup which they used, and faithfully find the hyperparameters to reproduce their results. For evaluation, we render images at the novel views (i.e., views not seen during training), and report CLIP image similarity and LPIPS between consecutive frames in rendered videos to measure multi-view consistency, as well as CLIP image-text similarity to measure fidelity to the instruction. Detailed explanations for each dataset sequence and training details can be found in Appendix D.3.

Figure 4 and Table 2 summarize the comparison between CSD-Edit and IN2N. We notice that CSD-Edit enables a wide-range control of 3D NeRF scenes, such as delicate attribute manipulation (e.g., facial expression alterations) and scene-stylization (e.g., conversion to the animation style). Especially, we notice two advantages of CSD-Edit compared to IN2N. First, CSD-Edit presents high-quality details to the edited 3D scene by providing multi-view consistent training views during NeRF optimization. In Figure 4, one can observe that CSD-Edit captures sharp details of anime character, while IN2N results in blurry face. Second, CSD-Edit is better at preserving the semantics of source 3D scenes, e.g., backgrounds or colors. For instance in Figure 4, we notice that CSD-Edit allows subtle changes in facial expressions without changing the color of the background or adding a beard to the face.

#### 4.4 Ablation study

CSD for text-to-3D generation. We explore the effectiveness of CSD in text-to-3D generation tasks following DreamFusion [26]. We train a coordinate MLP-based NeRF architecture from scratch using text-to-image diffusion models. Since the pixel-space diffusion model that DreamFusion used [26] is not publicly available, we used an open-source implementation of pixel-space text-to-image diffusion model.3 When using CSD for text-to-3D generation, we empirically observe that using LPIPS [43] as a distance for RBF kernel worked well. We refer to Appendix B.2 for details.

Given a set of text prompts, we run both DreamFusion and DreamFusion with CSD with a fixed seed.

- In Figure 6, we visualize generated examples. Remark that DreamFusion and DreamFusion + CSD tend to generate similar objects, but we observe that CSD often adds better details that complement the poor quality of one that made by DreamFusion. For instance, in Figure 6, CSD removes blurry artifacts in the synthesized 3D NeRF scene, which is often caused by inconsistent view distillation. Also in Figure 6, we verify that the CSD generates more coherent images when conditioned on view-dependent prompts which were used in DreamFusion. We refer to Appendix B.2 for more examples of text-to-3D generation.

Ablation on the components of CSD. To demonstrate the effect of our method, we present an ablation study on a video editing experiment. To verify the role of communication between samples using SVGD, we compare the editing results with and without SVGD. Also, to verify the role of baseline noise in CSD-Edit, we provide result when using random noise as baseline. As shown in Figure 7, CSD-Edit consistently edits a source video adding a red cap on a man’s head when given

3https://github.com/deep-floyd/IF

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

Source Random noise

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

Without SVGD CSD-Edit

“Give him a cap”

- Figure 7: Ablation study. Given a source video (top left), CSD-Edit without SVGD results in inconsistent frames (bottom left), and subtracting random noise in CSD-Edit results in loss of details (top right). CSD-Edit obtains consistency between frames without loss of semantics (bottom right).

the instruction “give him a cap.” However, without SVGD, the edits between frames are inconsistent, for example, blue caps or red caps appear both on the edited frames. In addition, if we set the baseline noise as the random noise injected into the source and target image, each frame gets blurry and loses the original structures, e.g., blurred legs and backgrounds.

### 5 Related work

Following remarkable success of text-to-image diffusion models [4, 20, 1, 2, 44], numerous works have attempted to exploit rich knowledge of text-to-image diffusion models for various visual editing tasks including images [10, 45, 13, 46, 14, 12, 15], videos [47, 25], 3D scenes [39], etc. However, extending existing image editing approaches to more complex visual modalities often faces a new challenge; consistency between edits, e.g., spatial consistency in high-resolution images, temporal consistency in videos, and multi-view consistency in 3D scenes. While prior works primarily focus on designing task-specific methods [24, 22, 25] or model fine-tuning for complex modalities [47], we present a modality-agnostic novel method for editing, effectively capturing consistency between samples.

The most related to our work is DreamFusion [26], which introduced Score Distillation Sampling (SDS) for creation of 3D assets, leveraging the power of text-to-image diffusion models. Despite the flexible merit of SDS to enable the optimization of arbitrary differentiable operators, most works mainly focus on applying SDS to enhance the synthesis quality of 3D scenes by introducing 3D specific frameworks [48, 49, 50, 51, 52]. Although there exists some work to apply SDS for visual domains other than 3D scenes, they have limited their scope to image editing [53], or image generation [54]. Here, we clarify that our main focus is not to improve the performance of SDS for a specific task, but rather to shift the focus to generalizing it from a new perspective in a principled way. To the best of our knowledge, we are the first to center our work on the generalization of SDS and introduce a novel method that simply but effectively adapts text-to-image diffusion models to diverse high-dimensional visual syntheses beyond a single 2D image with fixed resolution.

### 6 Conclusion

In this paper, we propose Collaborative Score Distillation (CSD) for consistent visual synthesis and manipulation. CSD is built upon Stein variational gradient descent, where multiple samples share their knowledge distilled from text-to-image diffusion models during the update. Furthermore, we propose CSD-Edit that gives us consistent editing of images by distilling minimal, yet sufficient information from instruction-guided diffusion models. We demonstrate the effectiveness of our method in text-guided translation of diverse visual contents, such as in high-resolution images, videos, and real 3D scenes, outperforming previous methods both quantitatively and qualitatively.

Limitations. Since we use pre-trained text-to-image diffusion models, there are some cases where the results are imperfect due to the inherent inability of diffusion models in understanding language. Also, our method might be prone to the underlying societal biases in diffusion models. See Appendix F.

Societal impact. Our method enables consistent editing of visual media. On the other hand, our method is not free from the known issues that text-to-image models carry when used by malicious users. We expect future research on the detection of generated visual content. See Appendix G.

### References

- [1] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35: 36479–36494, 2022.
- [2] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [3] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [4] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [5] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [6] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.
- [7] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.
- [8] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [9] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [10] C. Meng, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [11] O. Bar-Tal, D. Ofri-Amar, R. Fridman, Y. Kasten, and T. Dekel. Text2live: Text-driven layered image and video editing. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XV, pages 707–723. Springer, 2022.
- [12] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or. Prompt-toprompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [13] B. Kawar, S. Zada, O. Lang, O. Tov, H. Chang, T. Dekel, I. Mosseri, and M. Irani. Imagic: Text-based real image editing with diffusion models. arXiv preprint arXiv:2210.09276, 2022.
- [14] T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022.
- [15] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022.

- [16] A. Voynov, K. Aberman, and D. Cohen-Or. Sketch-guided text-to-image diffusion models. arXiv preprint arXiv:2211.13752, 2022.
- [17] L. Zhang and M. Agrawala. Adding conditional control to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.
- [18] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [19] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2022.
- [20] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [21] P. Esser, J. Chiu, P. Atighehchian, J. Granskog, and A. Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023.
- [22] C. Qi, X. Cun, Y. Zhang, C. Lei, X. Wang, Y. Shan, and Q. Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023.
- [23] L. Khachatryan, A. Movsisyan, V. Tadevosyan, R. Henschel, Z. Wang, S. Navasardyan, and H. Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.
- [24] S. Liu, Y. Zhang, W. Li, Z. Lin, and J. Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023.
- [25] D. Ceylan, C.-H. P. Huang, and N. J. Mitra. Pix2video: Video editing using image diffusion. arXiv preprint arXiv:2303.12688, 2023.
- [26] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [27] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamorthi, and R. Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision, 2020.
- [28] O. Bar-Tal, L. Yariv, Y. Lipman, and T. Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2, 2023.
- [29] D. Kingma, T. Salimans, B. Poole, and J. Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021.
- [30] T. Karras, M. Aittala, T. Aila, and S. Laine. Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364, 2022.
- [31] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In F. Bach and D. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR. URL https: //proceedings.mlr.press/v37/sohl-dickstein15.html.
- [32] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
- [33] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [34] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv:1704.00675, 2017.
- [35] A. Oord, Y. Li, I. Babuschkin, K. Simonyan, O. Vinyals, K. Kavukcuoglu, G. Driessche, E. Lockhart, L. Cobo, F. Stimberg, et al. Parallel wavenet: Fast high-fidelity speech synthesis. In International conference on machine learning, pages 3918–3926. PMLR, 2018.
- [36] Q. Liu and D. Wang. Stein variational gradient descent: A general purpose bayesian inference algorithm. Advances in neural information processing systems, 29, 2016.
- [37] J. Gorham and L. Mackey. Measuring sample quality with kernels. In International Conference on Machine Learning, pages 1292–1301. PMLR, 2017.
- [38] C. Wang, R. Jiang, M. Chai, M. He, D. Chen, and J. Liao. Nerf-art: Text-driven neural radiance fields stylization. arXiv preprint arXiv:2212.08070, 2022.
- [39] A. Haque, M. Tancik, A. Efros, A. Holynski, and A. Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. 2023.
- [40] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [41] R. Gal, O. Patashnik, H. Maron, A. H. Bermano, G. Chechik, and D. Cohen-Or. Stylegan-nada: Clip-guided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41

(4):1–13, 2022.

- [42] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.
- [43] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [44] Y. Balaji, S. Nah, X. Huang, A. Vahdat, J. Song, K. Kreis, M. Aittala, T. Aila, S. Laine, B. Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [45] G. Couairon, J. Verbeek, H. Schwenk, and M. Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.
- [46] D. Valevski, M. Kalman, Y. Matias, and Y. Leviathan. Unitune: Text-driven image editing by fine tuning an image generation model on a single image. arXiv preprint arXiv:2210.09477, 2022.
- [47] J. Z. Wu, Y. Ge, X. Wang, W. Lei, Y. Gu, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou. Tune-avideo: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022.
- [48] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin. Magic3d: High-resolution text-to-3d content creation. arXiv preprint arXiv:2211.10440, 2022.
- [49] C. Tsalicoglou, F. Manhardt, A. Tonioni, M. Niemeyer, and F. Tombari. Textmesh: Generation of realistic 3d meshes from text prompts. arXiv preprint arXiv:2304.12439, 2023.
- [50] L. Melas-Kyriazi, C. Rupprecht, I. Laina, and A. Vedaldi. Realfusion: 360 {\deg} reconstruction of any object from a single image. arXiv preprint arXiv:2302.10663, 2023.
- [51] R. Chen, Y. Chen, N. Jiao, and K. Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023.
- [52] J. Tang, T. Wang, B. Zhang, T. Zhang, R. Yi, L. Ma, and D. Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023.

- [53] A. Hertz, K. Aberman, and D. Cohen-Or. Delta denoising score. arXiv preprint arXiv:2304.07090, 2023.
- [54] K. Song, L. Han, B. Liu, D. Metaxas, and A. Elgammal. Diffusion guided domain adaptation of image generators. arXiv preprint arXiv:2212.04473, 2022.
- [55] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
- [56] Y. Du, C. Durkan, R. Strudel, J. B. Tenenbaum, S. Dieleman, R. Fergus, J. Sohl-Dickstein, A. Doucet, and W. Grathwohl. Reduce, reuse, recycle: Compositional generation with energybased diffusion models and mcmc. arXiv preprint arXiv:2302.11552, 2023.
- [57] N. Liu, S. Li, Y. Du, A. Torralba, and J. B. Tenenbaum. Compositional visual generation with composable diffusion models. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVII, pages 423–439. Springer, 2022.
- [58] J. Tang. Stable-dreamfusion: Text-to-3d with stable-diffusion, 2022. https://github.com/ashawkey/stable-dreamfusion.
- [59] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770– 778, 2016.
- [60] X. Xie, P. Zhou, H. Li, Z. Lin, and S. Yan. Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. arXiv preprint arXiv:2208.06677, 2022.
- [61] G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave, V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and L. Schmidt. Openclip, July 2021. URL https://doi.org/10.5281/zenodo.5143773. If you use this software, please cite it as below.
- [62] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, J. Kerr, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, et al. Nerfstudio: A modular framework for neural radiance field development. arXiv preprint arXiv:2302.04264, 2023.

## Appendix

##### Website: https://subin-kim-cv.github.io/CSD

- A Technical details In this section, we provide detailed explanations on the proposed methods, CSD and CSD-Edit.

CSD derivation. Consider a set of parameters {θi}Ni=1 which generates images x(i) = g(θi). For each timestep t ∼ U(tmin,tmax), we aim at minimizing the following KL divergence

DKL q(xt(i)|x(i) = g(θi))∥pϕ(xt;y,t)

for each i = 1,2,...,N via SVGD using Eq. (7). To this end, we approximate the score function, (i.e., gradient of log-density) by the noise predictor from diffusion model as follows:

∇x(i)

t

log pϕ(x(ti);y,t) ≈ −

ϵϕ(x(ti);y,t) σt

.

Then, the gradient of score function with respect to parameter θi is given by

∇θi

log pϕ(x(ti);y,t) = ∇x(i)

t

log pϕ(x(ti);y,t)

∂x(ti) ∂θi ≈ −

αt σt

ϵϕ(x(ti);y,t)

∂x(i) ∂θ

, (10)

for each i = 1,...N. Finally, to derive CSD, we plug Eq. (10) to Eq. (7) to attain Eq. (8). Also, we subtract the noise ϵ, which helps reducing the variance of gradient for better optimization. Following DreamFusion [26], we do not compute the Jacobian of U-Net. At high level, CSD takes the gradient update on each x(i) using SVGD and update θi by simple chain rule without computing the Jacobian. This formulation makes CSD as a straightforward generalization to SDS for multiple samples and leads to effective gradient for optimizing with consistency among batch of samples.

CSD-Edit derivation. As mentioned above, we subtract the random noise to reduce the variance of CSD gradient estimation. This is in a similar manner to the variance reduction in policy gradient [55], where having proper baseline function results in faster and more stable optimization. Using this analogy, our intuition is build upon that setting better baseline function can ameliorate the optimization of CSD. Thus, in image-editing via CSD-Edit, we propose to use image-conditional noise estimate as a baseline function. This allows CSD-Edit to optimize the latent driven by only the influence of instruction prompts. We notice that similar observations were proposed in Delta Denoising Score (DDS) [53], where they introduced an image-to-image translation method that is based on SDS, and the difference of the noise estimate from target prompt and that from source prompt are used. Our CSD can be combined with DDS by changing the noise difference term as follows:

∆Et = ϵϕ(xt;ytgt,t) − ϵϕ(x˜t;ysrc,t),

where x and x˜ are target and source images, ytgt and ysrc are target and source prompts. However, we found that CSD-Edit with InstructPix2Pix is more amenable in editing real images as it does not require source prompt. Finally, we remark that CSD-Edit can be applied to various text-to-image diffusion models such as ControlNet [17], which we leave it for the future work.

- B Additional experiments

#### B.1 Compositional editing

Recent works have shown the ability of text-to-image diffusion models in compositional generation of images handling multiple prompts [56, 57]. Here, we show that CSD-Edit can extend this ability to compositional editing, even at panorama-scale images which require a particular ability to maintain far-range consistency. Specifically, we demonstrate that one can edit a panorama image to follow different prompts on different regions while keeping the overall context uncorrupted.

0.8

CLIPImageSimilarity

0.7

0.6

0.5

CSD-Edit

Without SVGD Random Noise

0.4

0.10 0.15 0.20 0.25 0.30 CLIP Directional Similarity

- Figure 8: Ablation study. Ablation study on the components of CSD-

Edit at different guidance scales ωy ∈ {3.0,5.0,7.5,10.0}.

Table 3: Text-to-3D. Quantitative comparison between CSD and SDS under on text-to-3D generation via DreamFusion [26]

CLIP Similarity CLIP Similarity FID

Color ↑ Geo ↑ ↓

SDS [26] 0.437 0.322 259.4 CSD (Ours) 0.447 0.345 247.1

Given multiple textual prompts {yk}Kk=1, the compositional noise estimate is given by

K

ϵϕ(xt;{yk}Kk=1,t) =

αkϵωϕ(xt;yk,t),

k=1

where αk are hyperparameters that regularize the effect of each prompt. When applying compositional generation to the panorama image editing, the challenge lies in obtaining image that is smooth and natural within the region where the different prompts are applied. To that end, for each patch of an image, we set αk to be the area of the overlapping region between the patch and region where prompt yk is applied. Also, we normalize to assure k αk = 1. In Figure 9, we illustrate some examples on compositional editing of a panorama image. For instance, given an image, one can change into different weathers, different seasons, or different painting styles without leaving artifacts that hinder the spatial consistency of an image.

#### B.2 Text-to-3D generation with CSD

As of Section 4.4, we present a detailed study on the effect of CSDin text-to-3D generation, particularly focusing on the DreamFusion architecture [26]. We follow the most of experimental setups from those conducted by Poole et al. [26]. Our experiments in this section are based on StableDreamFusion [58], a public re-implementation of DreamFusion, given that currently the official implementation of DreamFusion is not available on public.

Setup. We use vanilla MLP based NeRF architecture [27] with 5 ResNet [59] blocks. Other regularizers such as shading, camera and light sampling are set as default in [58]. We use viewdependent prompting given the sampled azimuth angle and interpolate by the text embeddings. We use Adan [60] optimizer with learning rate warmup over 2000 steps from 10−9 to 2 × 10−3 followed by cosine decay down to 10−6. We use batch size of 4 and optimize for 10000 steps in total, where most of the case sufficiently converged at 7000 to 8000 steps. For the base text-to-image diffusion model, we adopt DeepFloyd-IF-XL-v1.0 since we found it way better than the default choice of Stable Diffusion in a qualitative manner. While the original DreamFusion [26] used guidance scale of 100 for their experiments, we find that guidance scale of 20 works well for DeepFloyd. We selected 30 prompts used in DreamFusion gallery4 and compare their generation results via DreamFusion from the standard SDS and those from our proposed CSD. We use one A100 (80GB) GPU for each experiment, and it takes ∼5 hours to conduct one experiment.

For CSD implementation, we use LPIPS [42] as a distance of RBF kernel. Note that LPIPS gives more computational cost than the usual ℓ2-norm based RBF kernel. The LPIPS is computed between two rendered views of size 64×64. For the kernel bandwidth, we use h = med

2

logB, where med is a median of the pairwise LPIPS distance between the views, B is the batch size.

For evaluation, we render the scene at the elevation at 30 degree and capture at every 30 degree of azimuth angle. Then we compute the CLIP image-text similarity between the rendered views and

4https://dreamfusion3d.github.io/gallery.html

input prompts. We measure similarities for both textured views (RGB) and textureless depth views (Depth). We also report Frechet Inception Distance (FID) between the RGB images and ImageNet validation dataset to evaluate the quality and diversity of rendered images compared to natural images.

Results. In Table 3, we report the evaluation results of CSD on text-to-3D generation comparison to DreamFusion. Remark that CSD presents better CLIP image-text similarities in both RGB and Depth views. Also, CSD achieves lower FID score showing its better quality on generated samples. Since we used the same random seed in generating both CSD and DreamFusion, the shapes and colors are similar. However, the results show that CSD obtains finer details in its generations.

In Figure 13, we qualitatively compare the baseline DreamFusion (SDS) and ours. We empirically observe three benefits of using CSD over SDS. First, CSD provides better quality compared to SDS. SDS often suffers from Janus problem, where multiple faces appear in a 3D object. We found that CSD often resolves Janus problem by showing consistent information during training. See the first row of Figure 13. Second, CSD can give us better fine-detailed quality. The inconsistent score distillation often gives us blurry artifact or undesirable features left in the 3D object. CSD can handle this problem and results in higher-quality generation, e.g., Figure 13 second row. Lastly, CSD can be used for improving diversity. One problem of DreamFusion, as acclaimed by the authors, is that it lacks sample diversity. Thus, it often relies on changing random seeds, but it largely alters the output. On the other hand, we show that CSD can obtain alternative sample with only small details changed, e.g., Figure 13 third row. Even when SDS is successful, CSD can be used in generating diverse sample.

### C Ablation study

In addition to the qualitative examples shown in Section 4.4, we present an additional ablation study on (a) the effect of SVGD and (b) subtracting random noise in CSD-Edit in panorama image editing experiments. Following the experimental setup in Section 4.1, we select 16 images and apply 5 different artistic stylization using CSD-Edit, CSD-Edit without SVGD, and CSD-Edit without subtracting image-conditional noise estimate. Again, we measure the CLIP image similarity and CLIP directional similarity for the evaluation.

In Figure 8, we plot the results of the ablation study. Remark that CSD-Edit without SVGD radically changes the image due to the absence of consistency regularization. As illustrated in Figure 7, CSD-Edit via subtracting random noise instead of image-conditional noise results in blurry outputs. Here, we also quantitatively show that it results in significant degrade in CLIP image similarity and CLIP directional similarity, losing the details of the source image. In Figure 12, we depict the qualitative results on our ablation study.

### D Implementation details

Setup. For the experiments with CSD-Edit, we use the publicly available pre-trained model of Instruct-Pix2Pix [14]5 by default. We perform CSD-Edit optimization on the output space of Stable Diffusion [4] autoencoder. We use SGD optimizer with step learning rate decay, without adding weight decay. We set tmin = 0.2 and tmax = 0.5, where original SDS optimization for DreamFusion used tmin = 0.2 and tmax = 0.98. This is because we do not generally require a large scale of noise in editing. We use the guidance scale ωy ∈ [3.0,15.0] and image guidance scale ωs ∈ [1.5,5.0]. We find that our approach is less sensitive to the choice of image guidance scale, yet a smaller image guidance scale is more sensitive to editing. All experiments are conducted on AMD EPYC 7V13 64-Core Processor and a single NVIDIA A100 80GB. Throughout the experiments, we use OpenCLIP [61] ViT-bigG-14 model for evaluation.

#### D.1 Panorama image editing

To edit a panorama image, we first encode into the Stable Diffusion latent space (i.e., downscale by 8), then use a stride size of 16 to obtain multiple patches. Then we select a B batch of patches to perform CSD-Edit. Note that we perform CSD-Edit and then normalize by the number of appearances as

5https://github.com/timothybrooks/instruct-pix2pix

mentioned in Section 3.3. Note that our approach performs well even without using small batch size, e.g., for an image of resolution 1920×512, there are 12 patches and we use B = 4.

For experiments, we collect 32 panorama images and conduct 5 artistic stylizations: “turn into Van Gogh style painting”, “turn into Pablo Picasso style painting”, “turn into Andy Warhol style painting”, “turn into oriental style painting”, and “turn into Salvador Dali style painting”. We use learning rate of 2.0 and image guidance scale of 1.5, and vary the guidance scale from 3.0 to 10.0.

#### D.2 Video editing

We edit video sequences in DAVIS 2017 [34] by sampling 24 frames at the resolution of 1920×1080 from each sequence. Then, we resize all frames into 512×512 resolution and encode all frames each using Stable Diffusion. We use learning rate [0.25,2] and optimize them for [200,500] iterations.

#### D.3 3D scene editing

Following Instruct-NeRF2NeRF [39], we first pretrain NeRF using the nerfacto model from NeRFStudio [62], training it for 30,000 steps. Next, we re-initialize the optimizer and finetune the pre-trained NeRF model with edited train views. In contrast to Instruct-NeRF2NeRF, which edits one train view with Instruct-Pix2Pix after every 10 steps of update, we edit a batch of train views (batch size of 16) with CSD-Edit after every 2000 steps of update. The batch is randomly selected among the train views without replacement.

### E Additional qualitative results

Source

[Figure 148]

“Turnintosunny weather” “Turnintorainy weather” “Turnintosnowy weather”

[Figure 149]

“Turnintospring” “Turnintofall”

[Figure 150]

“TurnintoVan Gogh style painting” “TurnintoPaul Gauguin style painting”

[Figure 151]

- Figure 9: Compositional image editing. CSD-Edit demonstrates the ability to edit consistently and coherently across patches in panorama images. This provides the unique capability to manipulate each patch according to different instructions while maintaining the overall structure of the source image. Remarkably, CSD-Edit ensures a smooth transition between patches, even when different instructions are applied.

[Figure 152]

“Turnsheepsinto wolves” “Turnsheepsintokangaroos”

[Figure 153]

[Figure 154]

[Figure 155]

“Turnsheepsinto polar bears” “Turnsheepsintoreindeers”

[Figure 156]

[Figure 157]

Source

[Figure 158]

“Turnpenguins into chickens” “Turnpenguins into bears”

[Figure 159]

[Figure 160]

“Turnpenguins into pandas” “Turnpenguins into sea lions”

[Figure 161]

[Figure 162]

- Figure 10: Object editing. CSD-Edit can edit many objects in a wide panorama image consistently in accordance with the given instruction while preserving the overall structure of source images.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

CSD-Edit(Ours)CSD-Edit(Ours)SourceFateZeroPix2VideoFateZeroPix2VideoSource

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

“Turn a bear into a tiger”

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

“Give him a yellow T-shirt”

- Figure 11: Video editing. CSD-Edit demonstrates various editing from an object (e.g., tiger) to attributes (e.g., color) while providing consistent edits across frames and maintaining the overall structure of a source video.

[Figure 195]

“Turninto Van Gogh style painting”

[Figure 196]

[Figure 197]

[Figure 198]

“Turninto Pablo Picasso style painting”

[Figure 199]

[Figure 200]

[Figure 201]

“Turninto Andy Warhol style painting”

[Figure 202]

[Figure 203]

[Figure 204]

“Turninto oriental style painting”

[Figure 205]

[Figure 206]

[Figure 207]

“Turninto Salvador Dali style painting”

[Figure 208]

[Figure 209]

[Figure 210]

CSD-Edit CSD-Edit without SVGD CSD-Edit with Random Noise

- Figure 12: Ablation study: SVGD and random noise. As illustrated, edits across different patches are not consistent without SVGD. Also, when using random noise as baseline noise, it loses the content and the detail of the source image.

“a fox holding a videogame controller”

“a crocodile playing a drum set”

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

“a plush dragon toy”

“ChichenItza, aerial view”

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

“a beautiful dress made out of fruit, on a mannequin”

“a squirrel in samurai armor wielding a katana”

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

SDS CSD (Ours)

SDS CSD (Ours)

- Figure 13: Text-to-3D generation examples. (First row) CSD helps to capture coherent geometry compared to using SDS. (Second row) CSD allows learning finer details than SDS. (Third row) CSD can provide diverse and high-quality samples without changing random seeds.

[Figure 259]

[Figure 260]

Source “Turnflowersintored roses”

[Figure 261]

[Figure 262]

Source “Turn itinto an old vintage photo”

- Figure 14: Limitations. (First row) CSD-Edit often manipulates undesirable contents due to the inherent inability of Instruct-Pix2Pix model. (Second row) CSD-Edit often produces artifacts on the image due to the patch-wise update.

### F Limitations

As our method leverages pre-trained Instruct-Pix2Pix, it inherits the limitations of it such as undesirable changes to the image due to the biases. Also, as described in [14], Instruct-Pix2Pix is often unable to change viewpoints, isolate a specific object, or reorganize objects within the image.

When editing a high-resolution image by dividing it into patches, it often remains an artifact at the edge of the patches, especially at the corner side of an image. This is due to that the patches at the corner are less likely to be sampled during the optimization. See Figure 14 for examples.

When editing a video, the edited video often shows a flickering effect due to the inability of the Stable Diffusion autoencoder in compressing the video. We believe that using CSD-Edit with video diffusion models trained on video datasets can possibly overcome this problem.

### G Broader Impact

Our research introduces a comprehensive image editing framework that encompasses various modalities, including high-resolution images, videos, and 3D scenes. While it is important to acknowledge that our framework might be potentially misused to create fake content, this concern is inherent to image editing techniques as a whole. Furthermore, our method relies on generative priors derived from large text-to-image diffusion models, which may inadvertently contain biases due to the autofiltering process applied to the vast training dataset. These biases influence the score distillation process, where the undesired results may come out. However, we propose that employing Consistent Score Distillation (CSD) can assist us in identifying and understanding such undesirable biases. By leveraging the inter-sample relationships and aiming for consistent generation and manipulation of visual content, our method provides a valuable avenue for comprehending the interaction between samples and prompts. Exploring this aspect further represents an intriguing future direction.

