## Smooth Diffusion: Crafting Smooth Latent Spaces in Diffusion Models

# arXiv:2312.04410v1[cs.CV]7Dec2023

Jiayi Guo1,2*, Xingqian Xu1,3*, Yifan Pu2, Zanlin Ni2, Chaofei Wang2, Manushree Vasu1, Shiji Song2, Gao Huang2†, Humphrey Shi1,3†

1SHI Labs @ Georgia Tech & UIUC 2Tsinghua University 3Picsart AI Research (PAIR) https://github.com/SHI-Labs/Smooth-Diffusion

- Task 1: Image Interpolation
- Task 2: Image Inversion and Reconstruction Task 3: Image Editing

“A realistic dog”

[Figure 1]

Smooth Diffusion

(Ours)

[Figure 2]

Stable Diffusion

Image A Interpolation Image B

“A train going back to its course filled with people” Replace Item: “rabbit”→ “cat”

Add Item: +“bacon”

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

“A mouse is next to a keyboard on a desk”

Transfer Style: “watercolor style” Drag Point

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

Source Smooth Diff. Stable Diff.

Source Smooth Diff. Stable Diff. Source Smooth Diff. Stable Diff.

Figure 1. Smooth Diffusion for downstream image synthesis tasks. Our method formally introduces latent space smoothness to diffusion models like Stable Diffusion [59]. This smoothness dramatically aids various tasks in: 1) improving continuity of transitions in image interpolation, 2) reducing approximation errors in image inversion, & 3) better preserving unedited contents in image editing.

### Abstract

Recently, diffusion models have made remarkable progress in text-to-image (T2I) generation, synthesizing images with high fidelity and diverse contents. Despite this advancement, latent space smoothness within diffusion models remains largely unexplored. Smooth latent spaces ensure that a perturbation on an input latent corresponds to a steady change in the output image. This property proves beneficial in downstream tasks, including image interpolation, inversion, and editing. In this work, we expose the non-smoothness of diffusion latent spaces by observing noticeable visual fluctuations resulting from minor latent variations. To tackle this issue, we propose Smooth Diffusion, a

*Equal contribution. †Corresponding authors.

new category of diffusion models that can be simultaneously high-performing and smooth. Specifically, we introduce Step-wise Variation Regularization to enforce the proportion between the variations of an arbitrary input latent and that of the output image is a constant at any diffusion training step. In addition, we devise an interpolation standard deviation (ISTD) metric to effectively assess the latent space smoothness of a diffusion model. Extensive quantitative and qualitative experiments demonstrate that Smooth Diffusion stands out as a more desirable solution not only in T2I generation but also across various downstream tasks. Smooth Diffusion is implemented as a plug-and-play Smooth-LoRA to work with various community models. Code is available at https://github.com/SHI-Labs/Smooth-Diffusion.

### 1. Introduction

In recent years, diffusion models [13, 23, 59] have rapidly grown into very powerful tools for generative AI, particularly for text-to-image generation. The remarkable ability of diffusion models, generating high-quality photorealistic images from open-book contexts, has been highlighted in many research and commercial products. Such success has also inspired various diffusion-based downstream tasks, including image interpolation [27, 75], inversion [14, 42, 51, 68, 74], editing [20, 40, 43, 52, 66, 72, 78, 79], etc.

Despite the great success in the generation field, diffusion models occasionally produce low-quality results with undesirable and unpredictable behaviors. Specifically speaking, for image interpolation, the Stable Diffusion Walk (SDW) [27] test examines latent space with spherical linear interpolations, usually resulting in highly fluctuated outputs with unpredictable visual appearance. Examples can be found in Fig. 1 Task 1, in which such interpolation exhibits undesired sharp changes as well as “cartoonization” on photorealistic dog images, highlighted in the red box. For the image inversion task shown in Fig. 1 Task 2, a naive application of DDIM inversion [68] cannot reconstruct images faithfully from the sources. Instead, it generates incorrect colors and object orientations, and misinterprets the computer mouse as an animal mouse. For the image editing task shown in Fig. 1 Task 3, one may notice that only minor text prompt editing can lead to major updates on image contents and layouts, in which the object (i.e. the cat’s pose, the horse’s location, the shape of the pizza) can be wildly and incorrectly altered. Moreover, current diffusion models are unsuited to drag-based editing [66] because a fine-engineered drag method still has a noticeably large chance of breaking objects’ shape and semantics.

In this work, we step into an important but underexplored area: to improve the latent space smoothness of diffusion models. Our motivation to enhance latent smoothness comes from the real-world demand to improve the output qualities of the aforementioned downstream tasks. A smooth latent space implies a robust visual variation under a minor latent change. Therefore, enhancing such smoothness could help improve the continuity of image interpolation, expand the capacity of image inversion, and maintain correct semantics in image editing. Notably, prior works in GANs [30, 31, 65] have demonstrated that the smooth latent space of the generator can significantly improve downstream tasks’ quality, offering additional evidence of the importance of this area.

To achieve our goal, we propose Smooth Diffusion, a new category of diffusion models that can be simultaneously high-performing and smooth. We start our exploration by first formalizing the objective for Smooth Diffusion, in which fixed-size perturbations ∆ϵ on a latent noise ϵ should produce smooth visual changes ∆ x0 on the

synthetic image x0, rounded to a constant ratio C. Although one may think that according to the formulation, the smoothness constraint could be an accessible traintime loss. Actually, there is no direct application of such regularization from inference to training, and the challenge lies in the fact that in each training iteration (i.e., back-propagation), diffusion models optimize only a “t-step snapshot” instead of the entire T-step diffusion process.

Therefore, we introduce Step-wise Variation Regularization, a novel regularization that seamlessly incorporates our Smooth Diffusion’s inference-time objective to training. This regularization aims to bound the 2-norm of output variation ∆ x0 given a fixed-size change ∆xt in input xt at an arbitrary step t. The rationale of the reformulation is intuitive: If xt and x0 exhibit smooth changes at any t, then the relation between the latent noise ϵ (i.e. xT) and x0 is just the accumulation of smooth variations and thus can be smooth as well. More details can be found in Sec. 3.

In practice, our Smooth Diffusion is trained on top of a well-known text-to-image model: Stable Diffusion [59]. We examine and demonstrate that Smooth Diffusion dramatically improves the latent space smoothness over its baseline. Meanwhile, we conduct extensive research across numerous downstream tasks, including but not limited to image interpolation, inversion, editing, etc. Both qualitative and quantitative results support our conclusion that Smooth Diffusion can be the next-gen high-performing generative model not only for the baseline text-to-image task but across various downstream tasks.

### 2. Related Work

Diffusion models are initiated from a family of prior works including but not limited to [10, 63, 67, 73]. Since then, DDPM [23] introduced an image-based noise prediction model, becoming one of the most popular image generation research. Later works [13, 45, 68] extended DDPM, demonstrating that diffusion models perform on-par and even surpass GAN-based methods [16, 28–31]. Recently, generating images from text prompts (T2I) become an emerging field, among which diffusion models [17, 46, 56, 59, 61] have become quite visible to the public. For example, Stable Diffusion (SD) [59] consists of VAE [34] and CLIP [55], diffuses latent space, and yields an outstanding balance between quality and speed. Following SD [59], researchers also explored diffusion approaches for controls such as ControlNet [15, 25, 44, 54, 77, 82, 83, 86–88, 92] and multimodal such as Versatile Diffusion [8, 39, 70, 85]. Works from a different track reduce diffusion steps to improve speed [7, 32, 37, 41, 62, 69, 89, 93], or restrict data and domain for few-shot learning [19, 24, 38, 60], all had successfully maintained a high output quality.

Smooth latent space was one of the prominent prop-

erties of SOTA GAN works [11, 29–31], while exploring such property went through the decade-long GAN research [5, 16], whose goals were mainly robust training. Ideas such as Wasserstein GAN [6, 18] had proved to be effective, which enforced the Lipschitz continuity on discriminator via gradient penalties. Another technique, namely path length regularization, related to the Jacobian clamping in [48], was adapted in StyleGAN2 [30] and later became a standard setting for GAN-based generators [12, 35, 84, 91]. Benefiting from the smoothness property, researchers managed to manipulate latent space in many downstream research projects. Works such as [9, 47, 65, 80] explored latent space disentanglement. GAN-inverse [3, 4, 49, 81] had also proved to be feasible, along with a family of image editing approaches [50, 53, 57, 58, 71, 94]. As aforementioned, our work aims to investigate the latent space smoothness for diffusion models, which by far remains unexplored.

### 3. Methodology

In this section, we first introduce preliminaries of our method, including diffusion process [23], diffusion inversion [13, 42, 68] and low-rank adaptation [24] (Sec. 3.1). Then Smooth Diffusion is proposed with its definition, objective (Sec. 3.2) and regularization function (Sec. 3.3).

##### 3.1. Preliminaries

Diffusion process [23] is a kind of Markov chain that gradually adds random noise ϵt ∼ N(0,I) to ground truth signal x0 ∼ p(x0), making xT in a total of T steps. At each step, The noisy data xt is computed as:

xt = 1 − βtxt−1 + βtϵt, t = 1,2,··· ,T, (1)

where βt is the preset diffusion rate at step t. By making αt = 1 − βt, αt = Tt=1 αt and ϵ ∼ N(0,I), we have the following equivalents:

xt = √αtxt−1 + √1 − αtϵt

(2)

= √αtx0 + √1 − αtϵ, t = 1,2,··· ,T.

A diffusion model ϵθ(xt,t) is then trained to estimate ϵt from xt, by which one can predict the original signal x0 by gradually remove noise from the degraded xT [68]. This is commonly known as the backward diffusion process:

αt−1 αt

1 αt−1 − 1 −

1 αt − 1 · ϵθ( xt,t).

xt +

xt−1 =

(3) Diffusion inversion [13, 42, 68] targets to recover the exact backward diffusion process (i.e. xt,ϵθ( xt,t),t = 1,...,T) from a known final prediction x0. One of the common technique for such inversion is DDIM inversion [13, 68], which reverses Eq. (3) under a local linear approximation:

αt+1 αt

1 αt+1 − 1 −

1 αt − 1 · ϵθ( xt,t),

xt+1 =

xt +

(4) where xt represent the estimated xt at time t. However, DDIM inversion is only a rough estimation. For textto-image diffusion, a more advanced technique, Null-Text Inversion [42], optimizes additional null-text embeddings {∅t}Tt=1 for each step t, simulating the backward process with ϵθ(xt,t,ξ,∅t), where ξ is the input text embedding. The predicted null-text ∅t is the null input of the classifierfree guidance [22] with a guidance scale w:

ϵθ(xt,t,ξ,∅t) = w · ϵθ(xt,t,ξ) + (1 − w) · ϵθ(xt,t,∅t).

(5) Low-rank adaptation (LoRA) [24] is initially proposed to efficiently adapt large pretrained models to downstream tasks. The key assumption of LoRA is that the weight changes required during adaptation maintain a low rank. Given a pretrained model weight W0 ∈ Rd×k, its updated weight ∆W is expressed as a low rank decomposition:

W0 + ∆W = W0 + BA, (6) where B ∈ Rd×r, A ∈ Rr×k and r ≪ min(d,k). During adaptation, W0 is frozen, while B and A are trainable.

##### 3.2. Smooth Diffusion

As previously mentioned, modern diffusion models (DM) do not guarantee latent space smoothness, creating not only research gaps between GANs and diffusions but also unexpected challenges in downstream tasks. To address these issues, we propose Smooth Diffusion, a novel class of high-performing diffusion models with enhanced smoothness over its latent space. The underlining of Smooth Diffusion is the newly proposed training scheme in which we carried out a Step-wise Variation Regularization to enhance model smoothness.

To better explain our aims, we adopt the same terminologies from the standard inference-time diffusion process (Fig. 2a), involving a T steps procedure that transforms the random noise ϵ (i.e., xT) to the prediction x0. The overall objective of Smooth Diffusion can then be written in Eq. 7: in which we expect that a fixed-size change ∆ϵ on ϵ (i.e., ∆xT on xT) will finally lead to a non-zero, fixed-size change ∆ x0 on x0, up to a constant ratio C:

∥∆ x0∥2 ⇔ C∥∆xT∥2 = C∥∆ϵ∥2, ∀ϵ, (7)

Notice that by definition, xT is the initial input of the backward diffusion loop in Eq. 3. Since xT is close to ϵ ∼ N(0,1), for simplicity, we make them equivalent in all the following equations.

Nevertheless, one may notice that our inference-time objective in Eq. 7 cannot be directly transformed into a train-

𝑇 steps

𝛼 𝒙

|+| |
|---|---|
| | |

𝒙

𝝐 (or 𝒙 )

𝒙 𝒙 1 − 𝛼 𝝐

DM DM

(a) Inference-time Diffusion: Denoisng prediction through 𝑻 steps

(b) Training-time Diffusion: Denoisng prediction at a single step 𝒕

𝑇 steps

𝛼 𝒙 𝝐 + Δ𝝐

|+| |
|---|---|
| | |

𝒙 + Δ 𝒙

DM

#### DM

𝒙 + Δ𝒙 𝒙 + Δ 𝒙 1 − 𝛼 (𝝐 + Δ𝝐)

(or 𝒙 + Δ𝒙 )

𝐶 Δ𝝐 𝟐 = 𝐶 Δ𝒙 ⇔ Δ 𝒙 ,∀𝝐 𝐶 1 − 𝛼 Δ𝝐 𝟐 = 𝐶 Δ𝒙 ⇔ Δ 𝒙 ,∀𝝐

(c) Inference-time Smooth Diffusion: Variation constraint through 𝑻 steps

(d) Training-time Smooth Diffusion: Variation constraint at a single step 𝒕

- Figure 2. Illustration of Smooth Diffusion. Smooth Diffusion (c) enforces the ratio between the variation of the input latent (∥∆ϵ∥2 or ∥∆xT∥2) and the variation of the output prediction (∥∆ x0∥2) is a constant C. Training-time Diffusion (b) optimizes a “t-step snapshot” of the denoising prediction process in Inference-time Diffusion (a). Similarly, we propose Training-time Smooth Diffusion (d) to optimize a “t-step snapshot” of the variation constraint in Inference-time Smooth Diffusion (c). DM: Diffusion model.

ing loss function. This is because, in one training iteration (i.e., back-propagation), diffusion models optimize only a “t-step snapshot” of the diffusion process (Fig. 2b), where t is uniformly sampled from 1 to T. Hence, the proposed “global” objective (Eq. 7) for the entire T-step process is not accessible in training. Therefore, we need to reformulate our global objective into a step-wise objective shown in Eq. 8, which can later be integrated into the diffusion training process as a loss function:

the following regularization loss at any x0,ϵ, and step t:

√1 − αt∥JTϵ ∆ x0∥2 − a 2 , (9)

Lreg = E∆ x

0,ϵ

where ∆ x0 is the normally sampled pixel intensities normalized to unit length, ϵ is a normally sampled noise

in√1Eq.− αt2∥,JandTϵ ∆ xa0∥is2 computedthe exponentialonline movingduring training.average Inof practice, we compute Eq. 9 via standard backpropagation with the following identity:

√1 − αt∥JTϵ ∆ x0∥2 = ∥∇ϵ(√1 − αt x0 · ∆ x0)∥2. (10)

∥∆ x0∥2 ⇔ C∥∆xt∥2 = C√1 − αt∥∆ϵ∥2, ∀ϵ, (8)

The identity holds since ∆ x0 is independently sampled, and uncorrelated with ϵ.

where C is a non-zero constant. This step-wise objective indicates that at each training step, variations ∆ϵ on ϵ should

imply√1 − αvariationst. The rationale∆xt onofxEq.t with8 is aintuitive:ratio proportionalIf xt and xto0 show smooth changes at any t, then the relation between the latent noise ϵ (i.e. xT) and x0 is just the accumulation of smooth variations and thus can be smooth as well.

Next, we prove that the proposed objective in Eq. 9 exactly matches our optimization goal in Eq. 8. One preliminary result, proven in [30], is that in high dimensions, Eq. 9 is minimized when Jϵ is orthogonal at any ϵ up to a global scaling factor K (i.e. Jϵ · JTϵ = K · I). By applying the orthogonality of Jϵ, we have the following:

∂ϵ

##### 3.3. Step-wise Variation Regularization

JTϵ ∆ x0 = KJ−ϵ 1∆ x0 = K

∂ x0 · ∆ x0 = K∆ϵ. (11) When Lreg in Eq. 9 reaches its optimal, we then have:

While the motivation and formulation of the Smooth Diffusion objective are presented, how to realize such an objective remains unexplained. Therefore, in this section, we introduce Step-wise Variation Regularization to effectively integrate the step-wise objective into diffusion training.

a = √1 − αt∥JTϵ ∆ x0∥2 = √1 − αtK∥∆ϵ∥2. (12)

Notice that a = a∥∆ x0∥2, since ∥∆ x0∥2 = 1 is the aforementioned random unit length vector. Hence, we can finally reformulate the expression:

We draw inspiration from the regularization techniques [30, 48] adopted in GAN training. The core idea of Step-wise Variation Regularization is to bound the Jacobian matrix Jϵ = ∂ x0/∂ϵ of the diffusion system by minimizing

√1 − αt∥∆ϵ∥2

∥∆ x0∥2 = K a

(13)

√1 − αt∥∆ϵ∥2,

= C

which exactly matches our proposed objective in Eq. 8.

To summarize, during training, the Smooth Diffusion objective encompasses a combination of Lbase and Lreg:

L = Lbase + λLreg, (14)

where Lbase denotes the basic training objective of a diffusion model and λ represents a ratio parameter controlling the intensity of Step-wise Variation Regularization.

### 4. Experiments

##### 4.1. Experimental Setup

Baselines and settings. We select the Stable Diffusion [59] as the primary baseline for all tasks. Additionally, for image interpolation, we adopt a VAE-space interpolation and ANID [75] as competitors. For image inversion, we integrate Smooth Diffusion and Stable Diffusion with DDIM inversion [68] and Null-text inversion [42]. For text-based image editing, SDEdit [40], Prompt-to-Prompt (P2P) [20], Plug-and-Play (PnP) [72], Diffusion Disentanglement (Disentangle) [79], Pix2Pix-Zero [52] and Cycle Diffusion [78] are chosen as SOTA approaches. For drag-based image editing, we compare Smooth Diffusion with Stable Diffusion within the framework of DragDiffusion [66].

Implementation details. Smooth Diffusion is trained atop pretrained Stable Diffusion-V1.5 [59], using LoRA [24] finetuning technique. The UNet of Smooth Diffusion is set as trainable with a LoRA rank of 8, while the VAE and text encoder are frozen. We leverage the LAION Aesthetics 6.5+ as the training dataset, which contains 625K imagetext pairs with predicted aesthetics scores of 6.5 or higher from LAION-5B [64]. Smooth diffusion is typically trained for 30K iterations with a batch size of 96, 3 samples per GPU, a total of 4 A100 GPUs, and a gradient accumulation of 8. The AdamW [33] optimizer is adopted with a constant learning rate of 1 × 10−4 and a weight decay of 1 × 10−4. The ratio parameter λ in Eq. 14 is set to 1. During inference, the total number of diffusion steps is set to 50 and the classifier-free guidance [22] scale is set to 7.5.

Evaluation metrics. To evaluate the general text-to-image generation performance, we report the popular FID [21] and CLIP Score [55] on the MS-COCO validation set [36]. To assess the latent space smoothness, we propose an interpolation standard deviation (ISTD) as an evaluation metric. In specific, we randomly draw 500 text prompts from the MSCOCO validation set. For each prompt, we sample a pair of Gaussian noises and uniformly interpolate them from one to the other 9 times with mix ratios from 0.1 to 0.9. Fed into diffusion models together with a prompt, we could obtain a total of 11 generated images, 2 from the source Gaussian noises and 9 from the interpolated noises. We calculate the standard deviation of L2 distances between every two ad-

jacent images in the pixel space. Finally, we average the standard deviations over 500 prompts as ISTD. Ideally, a zero value of ISTD indicates that consistent and uniform visual fluctuations in the pixel space for identical fixedsize changes in the latent space, resulting in a smooth latent space. For image inversion, mean square error (MSE), LPIPS [90], SSIM [76] and PSNR [26] are adopted to evaluate the image reconstruction capability.

##### 4.2. Latent Space Interpolation

Qualitative comparison. The most straightforward way to demonstrate the smoothness of the latent space is through the observation of interpolation results between latent noises. In Fig. 3, we present interpolation comparisons between Smooth Diffusion and Stable Diffusion using real images. To generate these comparisons, we utilize the NTI [42] to invert a pair of real images into latent noises xT, sharing the same {∅t}Tt=1. We then perform uniform spherical linear interpolations between latent noises (also known as Stable Diffusion Walk [27]), resulting in 9 intermediate noises with mix ratios from 0.1 to 0.9. Subsequently, we concatenate the 11 images produced from these noises to create an image transition sequence in the figures.

Notably, as highlighted by the red boxes, Stable Diffusion exhibits significant visual fluctuations during the transition. In particular, the interpolated images may introduce new attributes that are unrelated to the source images, e.g., the undesired grasslands in the second row of Fig. 3. In contrast, our approach, Smooth Diffusion, not only avoids introducing obvious irrelevant attributes in the interpolated images but also ensures that the visual effects change smoothly throughout the transition. Additional interpolation results can be seen in supplementary materials.

In addition to Stable Diffusion, Fig. 3 also includes two other baseline methods for comparison: 1) VAE Interpolation (VAE Inter.), which performs interpolations within the VAE space of Stable Diffusion. However, the results closely resemble pixel-space interpolations, with significant degradation of visual details, particularly in the highlighted red box area. 2) ANID [75], which first adds noise to real images and subsequently denoises the interpolated noisy images using Stable Diffusion. In Fig. 3, ANID with a 50step scheduler exhibits highly blurred interpolation results. When ANID operates with a default 200-step scheduler, the blurring can be alleviated, but the quality of the interpolated images remains far from satisfactory.

Quantitative comparison. The goal of Smooth Diffusion is to enhance the latent space smoothness without image generation performance degradation compared to Stable Diffusion. In pursuit of this goal, we employ the ISTD introduced in Sec. 4.1 to evaluate the latent space smoothness. Additionally, we utilize FID [21] and CLIP Score [55] to assess generators’ overall performance. The results pre-

“A church”

[Figure 21]

Smooth Diffusion

(50 steps) Stable Diffusion (50 steps)

[Figure 22]

[Figure 23]

VAE Inter.

[Figure 24]

ANID (50 steps)

[Figure 25]

ANID (200 steps)

Image A Interpolation Image B

- Figure 3. Image interpolation comparison results. For Smooth Diffusion and Stable Diffusion [59], real images (Image A and B) are inverted into latents using NTI [42]. We perform spherical linear interpolations between latents (also known as Stable Diffusion Walk [27]) and concatenate the resulting images as a transition sequence. VAE Inter. performs interpolations within the VAE space of Stable Diffusion. ANID [75] first adds noise to real images and subsequently denoises the interpolated noisy images using Stable Diffusion.

Method ISTD (↓) FID (↓) CLIP Score (↑) Stable Diffusion 38.63 12.70 31.46

Smooth Diffusion 16.54 12.10 31.54

- Table 1. Quantitative evaluations of image interpolation and text-to-image generation. We evaluate Smooth Diffusion and Stable Diffusion [59] with ISTD, FID [21] and CLIP Score [55]. The better results are in bold.

sented in Tab. 1 demonstrate that Smooth Diffusion significantly outperforms Stable Diffusion in terms of ISTD, indicating a substantial improvement in the latent space smoothness. Furthermore, Smooth Diffusion exhibits superior performance in both FID and CLIP Score, suggesting that the enhancement of latent space smoothness and the overall image generation quality are not mutually exclusive but complement each other when the regularization term is applied with a suitable strength ratio.

- 4.3. Image Inversion and Reconstruction

“A man using his laptop computer while a cat sits on his lap”

|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|---|---|---|---|---|

“A large tower that has a big clock at top”

|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|
|---|---|---|---|---|

“A room with a couch, table set with dinnerware and a television”

|[Figure 36]|[Figure 37]|[Figure 38]|[Figure 39]|[Figure 40]|
|---|---|---|---|---|

Smooth Diff.

Smooth Diff.

Stable Diff. +NTI

Stable Diff. +DDIM

Source

+NTI (Ours)

+DDIM (Ours)

Figure 4. Image reconstruction comparison results. We integrate Smooth Diffusion and Stable Diffusion [59] with NTI [42] (column 2 & 3) and DDIM inversion [68] (column 4 & 5).

tively compare the image inversion and reconstruction performance of these integrated models using 500 randomly sampled images from the MS-COCO validation set [36].

Previous research [30] in the realm of GANs discovered that a smoother latent space has a positive impact on the accuracy of image inversion and reconstruction. We empirically validate this finding within the context of diffusion models. In specific, two representative inversion techniques, DDIM inversion [68] and Null-text inversion (NTI) [42] are adopted and integrated with Smooth Diffusion and Stable Diffusion separately. We both qualitatively and quantita-

As illustrated in the two rightmost columns of Fig. 4, when employing a straightforward DDIM inversion, Smooth Diffusion outperforms Stable Diffusion by a considerable margin in terms of reconstruction quality. This improvement is evident in various aspects, such as an accurate generation of character identities, a faithful recreation

“A chocolate cake with cream on it” →“A chocolate cake with strawberries on it”

|[Figure 41]|
|---|

|[Figure 42]|[Figure 43]|
|---|---|

|[Figure 44]|[Figure 45]|[Figure 46]|[Figure 47]|[Figure 48]|[Figure 49]|
|---|---|---|---|---|---|

Local Edit (Replace Item)

“A banana on the table” →“A banana and an apple on the table”

|[Figure 50]|
|---|

|[Figure 51]|[Figure 52]|
|---|---|

|[Figure 53]|[Figure 54]|[Figure 55]|[Figure 56]|[Figure 57]|[Figure 58]|
|---|---|---|---|---|---|

Local Edit (Add Item)

“A young girl face with long black hair” →“A young girl face with long black hair, cartoon style”

|[Figure 59]|
|---|

|[Figure 60]|[Figure 61]|
|---|---|

|[Figure 62]|[Figure 63]|[Figure 64]|[Figure 65]|[Figure 66]|[Figure 67]|
|---|---|---|---|---|---|

Global Edit (Transfer Style)

Source

Smooth Diff. Stable Diff. SDEdit P2P PnP Disentangle Pix2Pix-Zero Cycle Diff.

- Figure 5. Text-based image editing comparison results. We compare Smooth Diffusion and Stable Diffusion [59] (column 2 & 3), considering both local and global edits through the straightforward pipeline described in Sec. 4.4. Additionally, we present results from SOTA approaches, including SDEdit [40], P2P [20], PnP [72], Disentangle [79], Pix2Pix-Zero [52], and Cycle Diffusion [78], as references.

Method MSE (↓) LPIPS (↓) SSIM (↑) PSNR (↑) Stable Diff. + DDIM 0.1756 0.5385 0.2662 13.97

Smooth Diff. + DDIM 0.1086 0.4326 0.3418 16.17

Stable Diff. + NTI 0.0156 0.1656 0.6068 25.63 Smooth Diff. + NTI 0.0153 0.1635 0.6102 25.74 VAE Reconstruction 0.0148 0.1590 0.6136 25.98

- Table 2. Quantitative evaluations of image reconstruction. We integrate Stable Diffusion and Smooth Diffusion [59] with DDIM inversion [68] (row 2 & 3) and NTI [42] (row 4 & 5). MSE, LPIPS [90], SSIM [76] and PSNR [26] are evaluated. VAE Reconstruction results are provided as the optimal values.

of the city view behind the tower, and a correct reproduction of room layouts. This phenomenon underscores the fact that the latent space of Smooth Diffusion is more tolerant of the errors introduced by the local linear approximation in DDIM inversion. Consequently, the reconstruction results produced by Smooth Diffusion manage to retain the contents of the source images to a greater extent. On the other hand, when the optimization-based NTI technique is employed, the disparity between Smooth Diffusion and Stable Diffusion is not as pronounced. Nonetheless, there are still instances where Stable Diffusion exhibits subpar results, such as the ruined man’s face in Fig. 4.

To quantify the image reconstruction performance, MSE, LPIPS [90], SSIM [76] and PSNR [26] are reported in Tab. 2. Notably, the reconstruction error encompasses two components: 1) the error from different inversion methods and U-Net parameters and 2) the error from the shared pretrained VAE [34]. Hence, we included the VAE reconstruction errors as optimal values for our method. The results exhibit a consistent outperformance of Smooth Diffusion over Stable Diffusion across all metrics, whether using DDIM inversion or NTI. Moreover, “Smooth Diffusion +

NTI” performs results close to VAE reconstruction, indicating its superiority attributed to a smoother latent space.

##### 4.4. Image Editing

The superiority of Smooth Diffusion in image inversion and reconstruction has motivated us to explore its potential for enhancing image editing tasks. In this section, we delve into two typical image editing scenarios: text-based image editing and drag-based image editing.

Text-based image editing. There have been numerous methods [20, 40, 52, 72, 78, 79] proposed in the literature, each with its own unique designs aimed at achieving the SOTA performance. In contrast, we adopt a simpler pipeline akin to the image inversion and reconstruction process discussed in Sec. 4.3. The key distinction lies in our approach to modify the text prompt during the later time steps of the reconstruction process. In specific, the original ϵθ(xt,t,C,∅t) in Eq. (5) during NTI reconstruction (diffusion sampling) process is replaced with:

ϵθ(xt,t,Csrc,∅t), t > T × r, ϵθ(xt,t,Ctrg,∅t), t ≤ T × r,

(15)

ϵθ(xt,t,C,∅t) =

where Csrc represents the source text prompt for inversion, while Ctrg corresponds to the target text prompt for editing. The parameter r serves as a threshold, determining when to switch from Csrc to Ctrg. In practice, r is typically chosen within {0.6, 0.7, 0.8, 0.9}, with the exact value depending on the specific input images and target visual effects.

Through this straightforward pipeline, we conducted a comparative analysis of the editing performance between Smooth Diffusion and Stable Diffusion, as presented in the three left-most columns of Fig. 5. We also included editing results obtained from SOTA methods as references. Our

“A photo of a cat”

|[Figure 68]|[Figure 69]|[Figure 70]|[Figure 71]|
|---|---|---|---|

“A photo of a landscape”

|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]|
|---|---|---|---|

“A photo of flowers”

|[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|
|---|---|---|---|

Source User Edit Smooth Diff. Stable Diff.

- Figure 6. Drag-based image editing comparison results. We implement Smooth Diffusion and Stable Diffusion [59] within the framework of DragDiffusion [66], respectively.

evaluation encompasses both local and global editing tasks. The local editing tasks involve replacing items (e.g., changing “cream” to “strawberries”) and adding items (e.g., “apple”). On the other hand, the global editing tasks pertain to global style transfer, such as transforming an image into a “cartoon style”. It is evident that while Stable Diffusion excels in achieving precise image reconstruction with NTI, as discussed in Sec. 4.3, even minor modifications to the text prompt can significantly impact the content of the generated images. For instance, it can affect elements like the style of the cake, the shape of the banana, and the haircut of the girl. In contrast, Smooth Diffusion not only accurately generates edited images in accordance with the target text prompts but also effectively preserves the unedited contents. Furthermore, when compared to SOTA methods, even with this straightforward pipeline, Smooth Diffusion consistently delivers competitive results across all cases.

Drag-based image editing. As an emerging research avenue in the community, drag-based image editing [43, 50, 66] has garnered considerable attention recently. DragDiffusion [66] first introduces a framework for drag-based image editing employing Stable Diffusion. In the task 3 of Fig. 1 and Fig. 6, we showcase that by integrating Smooth Diffusion into the DragDiffusion framework, some previously unsuccessful editing operations with Stable Diffusion can be enabled. As illustrated, Smooth Diffusion achieves operations such as making the tree grow taller without damaging existing branches (Fig. 1), rotating the cat head, creating a new mountain top without destroying the original one, and letting new flowers grow in the vase (Fig. 6). These operations, however, fail with Stable Diffusion, indicating the non-smoothness of its latent space.

##### 4.5. Ablation Studies

Regularization ratio. In Tab. 3, we examine the impact of different strength ratios λ in Eq. (14). This ratio adjusts the intensity of the step-wise variation regularization. Specifically, when a weaker regularization is applied (e.g., λ = 0.1), we observe a slight improvement in the CLIP Score. However, there is a significant increase in ISTD, indicating a notable degradation in latent space smoothness. In contrast, employing a stronger regularization (e.g., λ = 10) leads to a smoother latent space, as demonstrated by the decrease in ISTD. However, in this case, we observe an unexpected increase in FID, indicating a notable decline in the quality of generated images. Therefore, selecting an appropriate trade-off value for λ becomes crucial based on the specific experimental settings. In our default setting, we find that λ = 1 serves as a suitable value.

Ratio ISTD (↓) FID (↓) CLIP Score (↑)

0.1 24.23 12.15 31.56 1 (default) 16.54 12.11 31.49 10 11.51 17.44 31.41

- Table 3. Ablation results of different regularization ratios. The best results are in bold, and the second-best results are underlined.

LoRA rank. In Tab. 4, we examine the impact of different ranks of the LoRA component utilized in our Smooth diffusion. We discover that LoRA ranks within the range of [4,16] are all suitable values for our default setting. We select a default rank of 8 because of its lowest ISTD among the first three rows in Tab. 4. Furthermore, we train a fully finetuned model, referred to as ”full,” which showcases a further decrease in ISTD. However, this comes at the expense of significantly degrading the quality of the generated images, as indicated by an increased FID and decreased CLIP Score. This decline in performance underscores the vulnerability of fully fine-tuned models to collapse within our default setting, emphasizing the need for additional meticulous design considerations.

Rank ISTD (↓) FID (↓) CLIP Score (↑) 4 16.76 12.36 31.49

8 (default) 16.54 12.11 31.54 16 16.65 11.49 31.61 full 11.52 27.27 28.86

- Table 4. Ablation results of different LoRA ranks. The best results are in bold, and the second-best results are underlined.
- 5. Conclusion

In this article, we explored Smooth Diffusion, an innovative diffusion model that enhances latent space smoothness

for generation. Smooth Diffusion adopts the novel Stepwise Variation Regularization, which successfully maintains variation between arbitrary input latent and generated images at a more bounded range. Smooth Diffusion was trained on top of the prevailing text-to-image model, from which we carried out extensive research, including but not limited to interpolation, inversion, and editing, all of which had shown competitive performance. Through qualitative and quantitative measurements, we demonstrated that Smooth Diffusion managed to make a smoother latent space without compromising the output quality. We believe that Smooth Diffusion will become a valuable solution for other challenging tasks, such as video generation, in the future.

### References

- [1] OpenJourney-V4. https : / / huggingface . co / prompthero/openjourney-v4, 2023. 12, 14
- [2] RealisticVision-V2. https://huggingface.co/ SG161222/Realistic_Vision_V2.0, 2023. 12, 14
- [3] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2StyleGAN++: How to edit the embedded images? In CVPR, 2020. 3
- [4] Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Restyle: A residual-based stylegan encoder via iterative refinement. In ICCV, 2021. 3
- [5] Martin Arjovsky and L´eon Bottou. Towards principled methods for training generative adversarial networks. arXiv:1701.04862, 2017. 3
- [6] Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein generative adversarial networks. In ICML, 2017. 3
- [7] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. AnalyticDPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In ICLR, 2022. 2
- [8] Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. One transformer fits all distributions in multi-modal diffusion at scale. In ICML, 2023. 2
- [9] David Bau, Jun-Yan Zhu, Hendrik Strobelt, Bolei Zhou, Joshua B Tenenbaum, William T Freeman, and Antonio Torralba. Gan dissection: Visualizing and understanding generative adversarial networks. In ICLR, 2018. 3
- [10] Yoshua Bengio, Eric Laufer, Guillaume Alain, and Jason Yosinski. Deep generative stochastic networks trainable by backprop. In ICML, 2014. 2
- [11] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv:1809.11096, 2018. 3
- [12] Eric R Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In CVPR, 2021. 3
- [13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 2, 3

- [14] Wenkai Dong, Song Xue, Xiaoyue Duan, and Shumin Han. Prompt tuning inversion for text-driven image editing using diffusion models. arXiv:2305.04441, 2023. 2
- [15] Vidit Goel, Elia Peruzzo, Yifan Jiang, Dejia Xu, Nicu Sebe, Trevor Darrell, Zhangyang Wang, and Humphrey Shi. Pairdiffusion: Object-level image editing with structure-andappearance paired diffusion models. arXiv:2303.17546,

2023. 2

- [16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS,

2014. 2, 3

- [17] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022. 2
- [18] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of wasserstein gans. NeurIPS, 30, 2017. 3
- [19] Jiayi Guo, Chaofei Wang, You Wu, Eric Zhang, Kai Wang, Xingqian Xu, Humphrey Shi, Gao Huang, and Shiji Song. Zero-shot generative model adaptation via image-specific prompt learning. In CVPR, 2023. 2
- [20] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv:2208.01626,

2022. 2, 5, 7

- [21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5, 6
- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshops, 2021. 3, 5
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. 2, 3, 5
- [25] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv:2302.09778, 2023. 2
- [26] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 2008. 5, 7
- [27] Andrej Karpathy. Stable Diffusion Walk. https : / / gist . github . com / karpathy / 00103b0037c5aaea32fe1da1af553355, 2022. 2, 5, 6
- [28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 2
- [29] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In NeurIPS, 2020. 3

- [30] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020. 2, 3, 4, 6
- [31] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In NeurIPS, 2021. 2, 3
- [32] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 2
- [33] Diederik P Kingma and Jimmy Ba. ADAM: A method for stochastic optimization. In ICLR, 2015. 5
- [34] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2015. 2, 7, 12
- [35] Chieh Hubert Lin, Hsin-Ying Lee, Yen-Chi Cheng, Sergey Tulyakov, and Ming-Hsuan Yang. Infinitygan: Towards infinite-pixel image synthesis. In ICLR, 2021. 3
- [36] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 5, 6, 12
- [37] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS,

2022. 2

- [38] Haoming Lu, Hazarapet Tunanyan, Kai Wang, Shant Navasardyan, Zhangyang Wang, and Humphrey Shi. Specialist diffusion: Plug-and-play sample-efficient fine-tuning of text-to-image diffusion models to learn any unseen style. In CVPR, 2023. 2
- [39] Weijian Mai and Zhijun Zhang. Unibrain: Unify image reconstruction and captioning all in one diffusion model from human brain activity. arXiv:2308.07428, 2023. 2
- [40] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2, 5, 7
- [41] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, 2023. 2
- [42] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 2, 3, 5, 6, 7, 12, 13, 14
- [43] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv:2307.02421, 2023. 2, 8
- [44] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv:2302.08453, 2023. 2
- [45] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 2
- [46] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022. 2

- [47] Yotam Nitzan, Amit Bermano, Yangyan Li, and Daniel Cohen-Or. Face identity disentanglement via latent space mapping. TOG, 2020. 3
- [48] Augustus Odena, Jacob Buckman, Catherine Olsson, Tom Brown, Christopher Olah, Colin Raffel, and Ian Goodfellow. Is generator conditioning causally related to gan performance? In ICML, 2018. 3, 4
- [49] Xingang Pan, Xiaohang Zhan, Bo Dai, Dahua Lin, Chen Change Loy, and Ping Luo. Exploiting deep generative prior for versatile image restoration and manipulation. TPAMI, 2021. 3
- [50] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your GAN: Interactive point-based manipulation on the generative image manifold. In SIGGRAPH, 2023. 3, 8
- [51] Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. Effective real image editing with accelerated iterative diffusion inversion. In ICCV, 2023. 2
- [52] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In SIGGRAPH, 2023. 2, 5, 7
- [53] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. StyleCLIP: Text-driven manipulation of StyleGAN imagery. In ICCV, 2021. 3
- [54] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv:2305.11147, 2023. 2
- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 5, 6
- [56] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv:2204.06125, 2022. 2
- [57] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a stylegan encoder for image-to-image translation. In CVPR, 2021. 3
- [58] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Trans. Graph., 2021. 3
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2, 5, 6, 7, 8, 12, 13
- [60] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2
- [61] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad

- Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [62] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. 2
- [63] Tim Salimans, Diederik Kingma, and Max Welling. Markov chain monte carlo and variational inference: Bridging the gap. In ICML, 2015. 2
- [64] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022. 5
- [65] Yujun Shen, Ceyuan Yang, Xiaoou Tang, and Bolei Zhou. InterfaceGAN: Interpreting the disentangled face representation learned by GANs. TPAMI, 2020. 2, 3
- [66] Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. DragDiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv:2306.14435, 2023. 2, 5, 8, 15
- [67] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [68] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2020. 2, 3, 5, 6, 7, 12, 14
- [69] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 2
- [70] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. arXiv:2305.11846, 2023. 2
- [71] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for StyleGAN image manipulation. TOG, 2021. 3
- [72] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, 2023. 2, 5, 7
- [73] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661– 1674, 2011. 2
- [74] Bram Wallace, Akash Gokul, and Nikhil Naik. EDICT: Exact diffusion inversion via coupled transformations. In CVPR, 2023. 2
- [75] Clinton Wang and Polina Golland. Interpolating between images with diffusion models. In ICML Workshops, 2023. 2, 5, 6
- [76] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. TIP, 2004. 5, 7
- [77] Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. arXiv:2305.01115, 2023. 2
- [78] Chen Henry Wu and Fernando De la Torre. A latent space of stochastic diffusion models for zero-shot image editing and guidance. In ICCV, 2023. 2, 5, 7
- [79] Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu

- Chang. Uncovering the disentanglement capability in textto-image diffusion models. In CVPR, 2023. 2, 5, 7
- [80] Weihao Xia, Yujiu Yang, Jing-Hao Xue, and Wensen Feng. Controllable continuous gaze redirection. In ACM MM,

2020. 3

- [81] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. Gan inversion: A survey. TPAMI, 2022. 3
- [82] Dejia Xu, Xingqian Xu, Wenyan Cong, Humphrey Shi, and Zhangyang Wang. Reference-based painterly inpainting via diffusion: Crossing the wild reference domain gap. arXiv:2307.10584, 2023. 2
- [83] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion: Taking” text” out of text-to-image diffusion models. arXiv:2305.16223, 2023. 2
- [84] Xingqian Xu, Shant Navasardyan, Vahram Tadevosyan, Andranik Sargsyan, Yadong Mu, and Humphrey Shi. Image completion with heterogeneously filtered spectral hints. In WACV, 2023. 3
- [85] Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi. Versatile diffusion: Text, images and variations all in one diffusion model. In ICCV, 2023. 2
- [86] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IPAdapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv:2308.06721, 2023. 2
- [87] Eric Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, and Humphrey Shi. Forget-me-not: Learning to forget in text-toimage diffusion models. arXiv preprint arXiv:2303.17591, 2023.
- [88] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2
- [89] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022. 2
- [90] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5, 7
- [91] Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, I Eric, Chao Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. In ICLR, 2020. 3
- [92] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. UniControlNet: All-in-one control to text-to-image diffusion models. In NeurIPS, 2023. 2
- [93] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: A unified predictor-corrector framework for fast sampling of diffusion models. In NeurIPS, 2023. 2
- [94] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. Indomain gan inversion for real image editing. In ECCV, 2020. 3

### Supplementary Materials A. Implementation Details

This section elaborates on details briefly introduced in the main paper. These include the notation, the basic training objective, the interpolation standard deviation (ISTD) metric, and our utilization of Null-text inversion (NTI) [42] for real-image interpolation.

##### A.1. Notation

Stable Diffusion [59] employs an efficient “latent” diffusion pipeline. Here the “latent” refers to using an individually trained (VAE) [34] to compress an input image x0 into its VAE-space representation z0:

z0 = E(x0), x0 = D(z0), (16)

where E and D represent the encoder and decoder of the VAE, respectively. For simplicity, we exclude this conversion process and only use “x”-based notations in the main paper. Although we chose Stable Diffusion as our baseline due to its popularity and high performance, our training pipeline is not specifically tailored for latent diffusion models and is compatible with other diffusion models.

##### A.2. Basic Training Objective

Smooth Diffusion’s training objective comprises two key components: 1) a basic training objective primarily centered on noise prediction but flexible in formulation for different diffusion models, and 2) our proposed Step-wise Variation Regularization term. In our experiments, the basic training objective is:

0,ϵ,t∥ϵ − ϵθ(xt,t)∥22, (17) which is a commonly adopted training objective across many diffusion models, e.g., Stable Diffusion [59].

Lbase = Ex

##### A.3. ISTD

The goal of ISTD is to quantify the deviation of pixel-space changes given the same fixed-step changes in latent space. A lower deviation implies the input latents and output images are more likely to change smoothly. In our experiments, we first randomly draw 500 text prompts from the MS-COCO validation set [36]. For each prompt, we then sample two random Gaussian noises, ϵa and ϵb. Next, we execute uniform spherical linear interpolations (slerp) between ϵa and ϵb for 11 times, varying the mixing ratio η from 0 to 1:

ϵη = slerp(ϵa,ϵb,η), η = 0,0.1,0.2,··· ,1. (18) We employ the testing diffusion model to generate 11

interpolated images { xη0}1η=0 from {ϵη}1η=0. Notice that Eq. 18 guarantees that the latent space changes between every two adjacent latents (i.e., ϵη and ϵη+0.1) are the same.

Hence, we calculate the L2 distances between every two adjacent images (i.e., xη0 and xη0+0.1 ) and compute the standard deviation of these distances. Finally, ISTD is the average of standard deviations over 500 different text prompts. For a fair comparison, the text prompts and the noises for each prompt are the same for different testing models.

##### A.4. NTI for real-image interpolation

NTI is initially designed to transform a real image x0 into a latent xT, along with a series of learnable null-text embeddings {∅t}Tt=1 for each step t. The optimization for each ∅t is formulated as:

∥ xt−1 − DDIM( xt,t,ξ,∅t)∥22. (19)

min

∅t

where { xt}Tt=1 represents intermidiate noisy images estimated by DDIM inversion [68]. For simplicity,

DDIM( xt,t,ξ,∅t) denotes the DDIM sampling process at step t, utilizing the text embedding ξ, the null-text embedding ∅t and the classifier-free guidance scale w = 7.5.

For real-image interpolation, we optimize a shared series of {∅t}Tt=1 for two real images, xa0 and xb0:

∥ xat−1 − DDIM( xat ,t,ξ,∅t)∥22+ ∥ xbt−1 − DDIM( xbt,t,ξ,∅t)∥22.

min

∅t

(20)

In our experiments, we only interpolate the latents xaT

and xbT following Eq. 18 and use the same null-text embeddings {∅t}Tt=1 for all interpolated images.

### B. Additional Results

This section provides additional visual results of Smooth Diffusion. We display image interpolation results in Fig. 7 and Fig. 8, image inversion and reconstruction results in Fig. 9, and image editing results in Fig. 10.

Reusability. The LoRA component of Smooth Diffusion remains adaptable to other models sharing the same architecture as Stable Diffusion. However, the effectiveness of this reusability is not guaranteed. We evaluate the integration of this LoRA component into two popular community models, RealisticVision-V2 [2] and OpenJourney-V4 [1]. As depicted in Fig. 8, this integration also enhances the latent space smoothness of these models. This reusability makes our method eliminate the need for repeated training and become a plug-and-play module across various models.

“A woman face”

[Figure 80]

Smooth Diffusion

[Figure 81]

Stable Diffusion

“A cute rabbit”

[Figure 82]

###### Smooth Diffusion

[Figure 83]

Stable Diffusion

“A beautiful landscape”

[Figure 84]

Smooth Diffusion

[Figure 85]

Stable Diffusion

“A chocolate cake”

[Figure 86]

Smooth Diffusion

[Figure 87]

Stable Diffusion

Image A Interpolation Image B

- Figure 7. Additional image interpolation results with Smooth Diffusion. For Smooth Diffusion and Stable Diffusion [59], real images (Image A and B) are inverted into latents using Null-text inversion [42]. We perform spherical linear interpolations between latents and concatenate the resulting images as a transition sequence.

“A basket of apples”

[Figure 88]

###### Smooth RealisticVision-V2

[Figure 89]

RealisticVision-V2

“A robot horse”

[Figure 90]

###### Smooth OpenJourney-V4

| |
|---|

[Figure 91]

OpenJourney-V4

Image A Interpolation Image B

- Figure 8. Image interpolation results with community models. We apply the LoRA component of Smooth Diffusion to RealisticVisionV2 [2] and OpenJournery-V4 [1] and perform spherical linear interpolations in their latent spaces.

[Figure 92]

“A city bus is parked on the curb waiting for people” “A dog that is wearing a dog collar smiling” “A hand holding a smart phone with apps on a screen”

[Figure 93]

“A plate of cooked food in seen in this image” “A skateboard that has its wheels on the floor”

[Figure 94]

[Figure 95]

“The train engine number 6309 is operated by BNSF” “Large four sided clock hangs on the corner of the building”

[Figure 96]

[Figure 97]

“An older Dodge pickup sits parked next to another older pickup”

[Figure 98]

“A woman walking down a street talking on a cell phone”

Smooth Diffusion

+NTI

Smooth Diffusion

+DDIM

Source

Smooth Diffusion

+NTI

Smooth Diffusion

+DDIM

Source

Smooth Diffusion

+NTI

Smooth Diffusion

+DDIM

Source

[Figure 99]

[Figure 100]

- Figure 9. Additional image inversion and reconstruction results with Smooth Diffusion. We integrate Smooth Diffusion with two typical diffusion inversion techniques, Null-text inversion [42] and DDIM inversion [68].

“A basket of apples”

“A bird standing on a branch”

“A dog standing on the bench”

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

“dog” ↓ “toy dog”

“apples” ↓ “oranges”

“bird” ↓ “Lego bird”

Local Edit (Replace Item)

“A dog in a jacket”

“A young girl” “A beach”

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

+“wearing a hat”

+“wearing sunglasses”

Local Edit (Add Item)

+“with a turtle on it”

“A river with trees on both sides”

“A red car with a black roof”

“A sitting cat”

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Global Edit (Transfer Style)

“Van Gogh painting” →

“Watercolor drawing” →

“anime painting” →

“A photo of a tree”

“An oil painting of a mountain”

“A photo of a river”

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Drag Edit (Move Point)

Smooth

Smooth

Smooth

Source User Edit Diffusion

Source User Edit Diffusion

Source User Edit Diffusion

- Figure 10. Additional image editing results with Smooth Diffusion. Both text-based image editing and drag-based image editing are evaluated. For text-based image editing, we consider both local and global edits to test Smooth Diffusion. For drag-based image editing, Smooth Diffusion is integrated into the framework of DragDiffusion [66].

