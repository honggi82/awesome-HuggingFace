# arXiv:2405.11252v1[cs.CV]18May2024

## Dreamer XL: Towards High-Resolution Text-to-3D Generation via Trajectory Score Matching

#### Xingyu Miao1 Haoran Duan2 Varun Ojha2 Jun Song3 Tejal Shah2 Yang Long1 Rajiv Ranjan2

1Durham University, UK 2Newcastle University, UK 3China University of Geosciences, China

### Abstract

In this work, we propose a novel Trajectory Score Matching (TSM) method that aims to solve the pseudo ground truth inconsistency problem caused by the accumulated error in Interval Score Matching (ISM) when using the Denoising Diffusion Implicit Models (DDIM) inversion process. Unlike ISM which adopts the inversion process of DDIM to calculate on a single path, our TSM method leverages the inversion process of DDIM to generate two paths from the same starting point for calculation. Since both paths start from the same starting point, TSM can reduce the accumulated error compared to ISM, thus alleviating the problem of pseudo ground truth inconsistency. TSM enhances the stability and consistency of the model’s generated paths during the distillation process. We demonstrate this experimentally and further show that ISM is a special case of TSM. Furthermore, to optimize the current multi-stage optimization process from high-resolution text to 3D generation, we adopt Stable Diffusion XL for guidance. In response to the issues of abnormal replication and splitting caused by unstable gradients during the 3D Gaussian splatting process when using Stable Diffusion XL, we propose a pixel-by-pixel gradient clipping method. Extensive experiments show that our model significantly surpasses the state-of-the-art models in terms of visual quality and performance. Code: https://github.com/xingy038/Dreamer-XL.

### 1 Introduction

In recent years, Virtual Reality (VR) and Augmented Reality (AR) have increasingly become a part of our daily lives, and the demand for high-quality 3D content has increased significantly. 3D technology has become extremely important, allowing us to visualize, understand and interact with complex objects and environments. It also plays a key role in various fields such as architecture, animation, gaming and virtual reality. In addition, 3D technology shows broad application prospects in retail [39], online meetings [21], education [27] and other fields [18, 4]. Despite its wide application, the complexity of creating 3D content poses considerable challenges: generating high-quality 3D models requires computional time, effort, and expertise. Given these challenges, methods for generating 3D from text have become particularly important in recent years [11, 43, 16, 31]. These methods create accurate 3D models directly from natural language descriptions, thereby reducing manual input in traditional 3D modeling processes. Once the text-to-3D method can efficiently generate large amounts of data, it will not only shorten the production time of 3D content, but also reduce costs and improve production efficiency.

Typically, text-to-3D generation methods utilize pre-trained text-to-image diffusion models [29] as an image prior to training neural parametric 3D models such as Neural Radiance Fields (NeRF) [20] and 3D Gaussian splitting [8]. These approaches enable the rendering of consistent images that are aligned with the text. This process essentially relies on Score Distillation Sampling (SDS) [24].

Preprint. Under review.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

“Luffy model with detailed facial features.” “A Mario 3D model, red hat, white gloves, dressed in his iconic overalls, 8K.”

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

“A Gundam model, with detailed panel lines and decals.” “An action figure of Iron Man, Marvel’s Avengers, HD.”

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

“A pyramid shaped burrito with a slice cut out of it.”

“A Supercar made out of toy bricks.”

“Viking axe, fantasy, weapon, blender, 8k, HDR.”

“A cuddly panda bear, with black and white fur.”

“A delicious hamburger.”

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

“a DSLR photo of a bagel filled with cream cheese and lox.”

“A forbidden castle high up in the mountains.”

“An icecream.” “A pineapple.” “A highly-detailed sandcastle.”

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

“The warm, gooey brownie was flecked with melting chocolate.”

“The vibrant red rose, with delicate petals and thorny stem.”

“A delicate porcelain teacup, adorned with delicate floral patterns.”

“a DSLR photo of a football helmet.”

“An avocado.”

- Figure 1: Example of text-to-3D content generated from scratch by our Dreamer XL. Our Dreamer XL is based on 3D Gaussian splatting using stable diffusion XL. Please zoom in for details.

Through SDS, the model can distill the capabilities of the pre-trained 2D diffusion model to obtain rendered images, and optimize the parameters of the 3D model through backpropagation, so that the 3D model can be effectively trained even without actual image data. However, since random noise generates inconsistent pseudo-baselines, the results obtained by SDS optimization of 3D models tend to be averaged, leading to problems such as over-smoothing.

Although some recent work [38, 10] devote themselves to solving the over-smoothing problems, they inevitably lead to the generation of low-resolution and average results due to the inherent limitations of stable diffusion models and their sampling methods. For example, [10], inspired by the DDIM inversion process, proposed interval score matching, which can generate relatively consistent results. However, due to the inherent cumulative errors in the DDIM inversion process, it may lead to the averaging of results in certain regions. Furthermore, most existing methods do not yet support the new high-resolution Stable Diffusion XL (SDXL) [23, 15]. To achieve an output of 1024x1024 high resolution, multi-stage optimization is necessary. The main reason is the inherent instability of the Variational Autoencoder (VAE) in the SDXL [23] architecture, which is particularly evident during the optimization process of the 3D Gaussian Splatting. In this process, the gradients directly affect the duplication and deletion of point clouds in 3D space. The anomalous gradients introduced by SDXL severely hinder the optimization process of 3D Gaussian Splatting, leading to generated 3D models that lose complex texture details, have blurred appearances, and exhibit abnormal colors. In severe cases, this can cause the 3D models to fail to converge.

In this work, we aim to overcome the above limitations. The reverse process of DDIM was adopted by [10], effectively reducing the higher reconstruction errors generated by the diffusion model’s one-step reconstruction and generating relatively consistent pseudo-ground truth. However, the

inherent accumulation errors in the reverse process of DDIM still result in semantic changes in these pseudo-ground truths, leading to partially averaged reconstruction results in certain regions, resulting in erroneous and unrealistic outcomes in these regions. To address this issue, we propose a novel method called Trajectory Score Matching (TSM). We through simple improvements to Interval Score Matching (ISM) [10], effectively alleviate the average effect of inconsistent pseudo-ground truths caused by inherent accumulation errors. We demonstrate that our TSM has smaller accumulation errors compared to ISM, and ISM can be considered as a special case of TSM. Through experiments, we prove that the effects produced by our TSM are superior to ISM, yielding highly realistic and detailed results. In order to achieve high-resolution output, previous methods need to undergo multistage training. Our model directly uses SDXL that supports high resolution as guidance without going through multi-stage training. This not only reduces training costs, but also simplifies the training process. However, we still need to solve the inherent gradient instability problem of SDXL. Therefore, we propose a pixel-by-pixel gradient clipping method, which effectively alleviates the inherent gradient instability of SDXL. In summary, the contributions of our work are as follows:

- • We investigate and analyze the inherent accumulated errors produced by DDIM inversion process in interval score matching (ISM), resulting in the presence of inconsistent pseudoground truth.
- • To address the aforementioned limitation, we introduce a novel Trajectory Score Matching (TSM) method. Unlike the single path of ISM, TSM improves ISM to a dual path, effectively alleviating the inconsistent pseudo-ground truth issue generated from the inherent accumulated error of DDIM.
- • To simplify the training process and generate high-resolution, high-quality text-to-3D results. We are the first to leverage SDXL for guidance based on 3D Gaussian splatting. In addition, we also introduce a novel gradient clipping method, which effectively solves the problem of SDXL in gradient stability. Extensive experiments demonstrate that our method significantly outperforms the current state-of-the-art methods.

### 2 Related Work

- 2D diffusion. Score-based generative models and diffusion models [33, 34, 5, 1, 29, 23] have shown excellent performance in image synthesis [3], especially by introducing latent diffusion models (LDM) [28] into stable diffusion to generate high-resolution images in latent space. Podel et al. [23] further extended this model to a larger latent space in SDXL, VAE and U-net, achieving higher resolution (1024×1024). Zhang et al. [42] enhanced the functionality of these models by generating controllable images for different input types. At the same time, the diffusion model also showed impressive performance in converting text into image synthesis, which opens up the possibility of using this technology to directly generate 3D images from text [2, 6, 11, 19, 24, 38].

Text-to-3D Generation. Early attempts from text-to-3D are mainly guided by the use of multimodal information from CLIP [26] to achieve information conversion from text-to-3D, with DreamField [7] being a pioneer in this direction. However, the multi-modal information of CLIP can only provide rough alignment, and the results of using it for 3D distillation are often unsatisfactory. Zero-1-to-3 [14] first introduces camera parameters to fine-tune the 2D pre-trained stable diffusion model, enabling it to generate multi-view images, and then use the multi-view images for 3D reconstruction. This improvement not only facilitates more accurate 3D reconstructions, but also inspires a wealth of derivative research [17, 13, 25, 12, 30]. In addition, another research direction explores the possibility of using pre-trained 2D diffusion models to directly optimize 3D representations. These methods often combine differentiable 3D representation techniques, such as NeRF [20], NeuS [37], and 3D Gaussian Splatting [8], and optimize model parameters through backpropagation techniques. Dreamfusion [24] first introduces SDS to optimize 3D representations directly from pre-trained 2D text-to-image diffusion models. Similarly, Score Jacobian Chaining [36] proposes an alternative method that achieves parameterization effects similar to SDS. ProlificDreamer [38] conducted an in-depth analysis of the objective function of SDS and proposed a particle-based variational framework called Variational Score Distillation (VSD), which significantly improves the quality of generated content. The latest research combines SDS with Gaussian Splatting to accelerate the optimization process. Consistent3D [40] analyzes SDS from the latest perspective of ordinary differential equations (ODE) and proposes a method called Consistency Distillation Sampling (CSD)

[Figure 62]

- Figure 2: ISM example [10]. We notice that using the same initial value x0 but under different noise {ϵ1,ϵ2,ϵ3,ϵ4}, the generated results still show certain inconsistencies. This is due to the error accumulation inherent in the DDIM inversion process. These inconsistencies can lead to errors or inconsistencies in some areas during the optimization of the 3D model.

to solve the challenges of SDS in over-smoothing and inconsistency issues. Similarly, LucidDreamer [10] analyzed the loss function of SDS and proposed interval score matching (ISM), which is very similar to the idea of CSD. However, ISM utilizes the reversible diffusion trajectory of DDIM [32] when calculating the two interval steps. DDIM will inevitably produce inherent accumulated errors in this process, resulting in inconsistent reconstruction results. In this work, we empirically follow the mature mainstream architecture method of 3D Gaussian Splatting [10, 35, 41] as the baseline of our approach. On this basis, inspired by the recent Consistency trajectory model [9], we propose to optimize the 3D model from two trajectories, and use the less noisy trajectory to guide another noisier trajectory to alleviate the inconsistency problem.

- 3 Methods

This section presents the preliminaries on the inverse DDIM process, SDS and ISM (see Section 3.1). We then propose the Trajectory Score Matching (TSM) method (see Section 3.2), which generates dual paths from the same starting point using the reverse process of DDIM. This enhances the stability and consistency of the model along the entire generative path during the distillation process. We further indicate that ISM is a special case of TSM (see Section 3.3). Additionally, we investigate the challenges of optimizing 3D models using the SDXL architecture (see Section 3.4).

#### 3.1 Preliminaries

Review of DDIM inversion We first consider the most common sampling scheme is that of DDIM [32] where intermediate steps are calculated as:

√1 − αtϵϕ(xt,t,∅) √αt

xt −

xt−1 = √αt−1

+ 1 − αt−1ϵϕ(xt,t,∅), (1)

where xt and xt−1 represent the noisy latent, {αt}Tt=0 (where a0 = 1,aT = 0) indicates a set of time steps indexing a strictly monotonically increasing noise schedule. The ϵϕ(xt,t,∅) denotes the predicted denoising direction (by stable diffusion model) with the given condition y (Here, the condition is null, i.e., unconditioned ∅).

As noted in DDIM [32], the above denoising process can approximate the inverse transition from xt to xt−1, which can be expressed as:

√1 − αtϵϕ(xt,t,∅) √αt−1

xt = √αt

xt−1 −

+ 1 − αt−1ϵϕ(xt,t,∅)

(2)

√1 − αtϵϕ(xt−1,t − 1,∅) √αt−1

√αt

xt−1 −

≈

+ 1 − αt−1ϵϕ(xt−1,t − 1,∅),

where the approximation is a linearization assumption that ϵϕ(xt,t,∅) ≈ ϵϕ(xt−1,t − 1,∅). This approximation inevitably introduces errors, resulting in inconsistencies between the diffusion states in the forward and backward processes.

Text-to-3D generation by interval score matching (ISM) The concepts of the ISM is first introduced by LucidDreamer [10] to address the issues of over-smoothness and inconsistency inherent

in the original SDS method. The 3D model leverages a differentiable function x = g(θ,c) to render images, where θ represents the trainable 3D parameters and c is camera parameter. The gradient of the ISM loss for θ is expressed as follows:

∂x ∂θ

, (3)

∇θLISM(θ) := Et,c ω(t)(ϵϕ(xt,t,y) − ϵϕ(xs,s,∅))

where 0 < s < t, the noisy latent xt and xs are calculated by DDIM inversion process, the ϵϕ(xt,t,y) is the predicted denoising direction given the conditioned y, and ω(t) is a time-dependent weighting function. In DDIM inversion process, the generation of accumulated error is inevitable. Especially for conditional models, these errors are further magnified. Therefore, inconsistent pseudo-ground truths will be generated when optimizing the 3D model, thus affecting the optimization quality of the final result.

#### 3.2 Trajectory Score Matching

To address the inherent accumulated error in the DDIM inversion process, which leads to the production of inconsistent pseudo-ground truths and consequently suboptimal 3D models. Inspired by recent work [9], we propose a new approach, Trajectory Score Matching (TSM), which utilizes dual paths originating from the same starting point to minimize error accumulation during iterations. Specifically, similar to ISM [10], our TSM also utilizes the DDIM inversion to predict an invertible noisy latent trajectory. For a given timestep s (where 0 < s < t), the corresponding noise latent xs can be obtained using Equation (2). Considering xs as the starting latent, it is possible to approximate two noise latents, xµ and xt, on the latent trajectory, where 0 < s < µ < t ≤ T. This can be expressed as follows:

+ √1 − αsϵϕ(xs,s,∅), (4)

xs − 1 − αµϵϕ(xs,s,∅) √αs

xµ = √αµ

√1 − αtϵϕ(xs,s,∅) √αs

+ √1 − αsϵϕ(xs,s,∅). (5)

xt = √αt

xs −

Then, we can integrate DDIM inversion and DDIM denoising with the same step size. We define the naive objective of 3D distillation as follows:

LTSM(θ) := Et,c ω(t)||ϵϕ(xt,t,y) − ϵϕ(xµ,µ,∅)||2 , (6)

where xt and xµ is generated through DDIM inversion from x0. Followiing [10], the gradient of TSM loss over θ is:

∂x ∂θ

. (7)

∇θLTSM(θ) := Et,c ω(t)(ϵϕ(xt,t,y) − ϵϕ(xµ,µ,∅))

The optimization goal of TSM is to maintain the consistency of x0 updates as much as possible to reduce the error introduced by DDIM inversion. Since TSM uses the same noise latent during the inversion process, its cumulative error is relatively small. The algorithm flow of TSM is shown in the Algorithm 1. Among them, the blue part marks the differences from ISM.

##### 3.3 Comparison with ISM We now analyze the differences between TSM and ISM theoretically.

TSM has smaller error For ISM, the optimized goal is the minimum predicted noise from noise latent xs and xt. The noise latent can be obtained from Equation (2) and then using the pre-trained stable diffusion model to predict noise. During the inversion process of DDIM, xs is approximated by the prediction noise at s − 1, and xt is approximated by the prediction noise at s. For simplicity, we can regard the optimization goal of ISM as minimizing noise latent, thus the accumulated error generated by ISM during the optimization process can be expressed as:

√1 − αtϵϕ(xs,s,∅) √αs

+ √1 − αsϵϕ(xs,s,∅)

xt − xs = √αt

xs −

(8)

√1 − αsϵϕ(xs−1,s − 1,∅) √αs−1

√αs

xs−1 −

−

+ 1 − αs−1ϵϕ(xs−1,s − 1,∅)

Algorithm 1 Trajectory Score Matching

- 1: Initialization: DDIM inversion step size δT and δS, the target prompt y, the offset rate γ ∈ [0,1]
- 2: while θ is not converged do
- 3: Sample: x0 = g(θ,c),t ∼ U(1,1000)
- 4: let s = t − δT, n = s/δS, and µ = s + γδT
- 5: for i = [0,...,n − 1] do
- 6: x(i+1)δ

S

= √α(i+1)δ

S

xiδS −

√

1−α(i+1)δS ϵϕ(xiδS ,iδS,∅)

√αiδS + √1 − αsϵϕ(xiδ

S

,iδS,∅)

- 7: end for
- 8: predict ϵϕ(xs,s,∅), then step xs → xt and xs → xµ via xt = √αt xs−

√1−αtϵϕ(xs,s,∅)

√αs + √1 − αsϵϕ(xs,s,∅) xµ = √αµ xs−

√

1−αµϵϕ(xs,s,∅)

√αs + √1 − αsϵϕ(xs,s,∅)

- 9: predict ϵϕ(xt,t,y), ϵϕ(xµ,µ,∅) and compute TSM gradient ∇θLTSM = ω(t)(ϵϕ(xt,t,y) − ϵϕ(xµ,µ,∅))
- 10: update x0 with ∇θLTSM
- 11: end while

For our TSM, we can also give similar expresses as:

√1 − αtϵϕ(xs,s,∅) √αs

xt − xµ = √αt

xs −

√αµ

−

xs − 1 − αµϵϕ(xs,s,∅) √αs

+ √1 − αsϵϕ(xs,s,∅)

+ √1 − αsϵϕ(xs,s,∅)

(9)

Compared with ISM, which is optimized for a single path, TSM is optimized for a dual path starting from the same noise latent. We show that the accumulated error is smaller in TSM, as shown below.

Theorem 1 Consider three timesteps 0 < s < µ < t, where µ is defined as µ = γ(t − s) + s, with γ ∈ [0,1]. Then, we have:

xt − xs > xt − xµ ⇒ xµ > xs, (10) Proof: Assume xµ ≤ xs. Utilizing Equation (8) and Equation (9), we aim to demonstrate that:

+ √1 − αsϵϕ(xs,s,∅)

xs − 1 − αµϵϕ(xs,s,∅) √αs

√αµ

(11)

√1 − αsϵϕ(xs−1,s − 1,∅) √αs−1

√αs

xs−1 −

≤

+ 1 − αs−1ϵϕ(xs−1,s − 1,∅).

Where α represents a set of timesteps with a strictly monotonically increasing noise schedule, hence αs−1 < αs < αµ. Considering the αs < αµ, the term involving √αµ should yield a smaller value compared to the term involving √αs, which presents a contradiction to our initial assumptions. Considering the ϵϕ terms, due to the inversion process of DDIM, ϵϕ(xs,s,∅) ≈ ϵϕ(xs−1,s − 1,∅) and √1 − αs−1 > √1 − αs. This also contradicts the assumption. Therefore, our initial assumption xµ ≤ xs must be false. Consequently, we conclude that xµ > xs, thus proving the theorem.

ISM is a special case of TSM Theoretically, the optimization objectives of ISM and TSM are the same, however, TSM considers optimization steps that are closer together compared to ISM. Specifically, we can regard ISM as a special case of TSM. Between time steps s and t, we choose any time step µ = γ(t − s) + s, where µ = s if and only if γ = 0. Therefore, our hypothesis is confirmed, and ISM is indeed a special case of TSM when µ = s.

#### 3.4 The abnormal gradient from advanced pipeline

Previous methods have shown that increasing rendering resolution and training batch size can significantly improve visual quality. Although increasing the resolution of rendering can significantly

DreamFusion(SDS) Fantasia3D ProlificDreamer (VSD) LucidDreamer (ISM) Ours

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

“A DSLR photo of the Imperial State Crown of England.”

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

“A DSLR photo of a Schnauzer wearing a pirate hat .”

- Figure 3: Comparison with state-of-the-art baseline methods in text-to-3D generation. Experimental results show that our method can generate 3D content that is more consistent with input text prompts and has more detailed details. All results of this work are generated on a single A100 GPU. Please zoom in to see more details.

improve the visual quality, most text-to-3D generation methods mainly use guidance based on Stable Diffusion 2.1 and only support 512 × 512 resolution. Due to the impact of low resolution, local details are still blurred. Consequently, we experimented with using Stable Diffusion XL as guidance, which supports 1024 × 1024 resolutions. The more advanced model Stable Diffusion XL has a different architecture from the previous one, and the VAE of this model is unstable. Although it has a certain impact on NeRF-based methods, it is not serious. However, this instability poses significant challenges for methods that employ 3D Gaussian splatting. In 3D Gaussian splatting, the reliability of operations like copying and deleting point clouds is heavily dependent on gradient stability. If the average positional gradient g of the Gaussian view space exceeds a preset threshold, regions with under- or over-reconstruction of color c and depth d are intensively corrected. SDXL gradients are usually large and unstable, and high average gradient values in this case may cause normal areas to still be densified. An intuitive method is leveraging the gradient clip technical to handle this issue, previous related work [22] has explored for NeRF-based method, which is not very suitable for 3D Gaussian splitting. Thus, we propose an improved gradient clip method for 3D Gaussian splitting. Specifically, we still use the [22] method for gradient clipping of color c. For depth d, we calculate its scaling factor independently for each depth element and perform pixel-by-pixel pruning. The pruning gradient of depth gˆd can be expressed as:

s |gd|

,c , (12)

gˆd = gd · min min

where gd is the gradient of depth, s is scale of Gaussian and c is the threshold. In this way, we can ensure that the updated direction of the depth gradient remains unchanged and has no effect on the gradient of the normal region.

### 4 Experiments

#### 4.1 Qualitative Results

Text-to-3D Generation We show the generated results of Dreamer XL in Figure 1. The results show that Dreamer XL is capable of generating high-quality 3D content accurately based on the input text, and it performs exceptionally well in producing realistic and complex appearances, effectively avoiding common issues such as excessive smoothing or oversaturation. For example, it can finely reproduce the texture details of objects like teacups. Moreover, our framework can generate objects that are close to reality and create imaginary ones. This flexibility offers possibilities for various application scenarios.

Comparison with State-of-the-Art Methods We compare our approach with four state-of-the-art text-to-3D baselines: DreamFusion [24] proposes Score Distillation Sampling (SDS) leveraging a

Stable Diffusion 2.1 base Stable Diffusion XL 1.0 “A portrait of IRONMAN, white hair, head, photorealistic, 8K, HDR.” “Zombie JOKER, head, photorealistic, 8K, HDR.”

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

###### ISM

ISM

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

TSM (Ours)

TSM (Ours)

- Figure 4: Comparison with the generation results of different stable diffusion models. Compared with ISM, our TSM performs better in the clarity and consistency of local details. Please zoom in to see the circled region for more details.

Table 1: Quantitative evaluation. We compare with recent text-to-3D conversion methods. CLIPscore is used to measure the alignment between text and 3D content, while A-LPIPS is used to evaluate the degree of artifacts caused by inconsistencies in 3D content.

CLIP-Score ↑ A-LPIPS ↓ CLIP-L/14 OpenCLIP-L/14 VGG Alex

Methods

DreamFusion 0.232 0.165 0.081 0.080 Fantasia3D 0.233 0.207 0.077 0.082 ProlificDreamer 0.255 0.221 0.178 0.103 LucidDreamer 0.278 0.234 0.065 0.059 Ours 0.297 0.243 0.052 0.041

pre-trained 2D text-to-image diffusion model for text-to-3D synthesis; Fantasia3D [2] disentangle geometric and appearance attributes to simulate real-world physical environments; ProlificDreamer [38] introduces Variational Score Distillation (VSD), a particle-based variational framework to address issues of oversaturation, oversmoothing, and low diversity; LucidDreamer [10] introduces Interval Score Matching (ISM), utilizing deterministic diffusion trajectories and interval-based score matching to alleviate oversmoothing problems, and employs a 3D Gaussian splatting for 3D representation. The comparison results are shown in Figure 3. The results generated by our method are significantly clearer than other baseline results. For example, the crown shows a more precise geometric structure and a more realistic color, and the Schnauzer’s hair texture and overall body shape show obvious advantages. We can observe that our method significantly outperforms existing methods in both visual quality and consistency.

Comparison with ISM in detail As shown in Figure 4, we show the generation results of ISM and TSM using the same prompt on different stable diffusion models. In Iron Man, it can be seen that the ISM has significant inconsistencies on the left and right sides of the neck, while our TSM maintains consistency in this region. In Joker, the ISM has shallower wrinkles on the head compared to our TSM, which is due to the averaging effect caused by error accumulation. Furthermore, ISM also shows significant inconsistency in the neck region.

#### 4.2 Quantitative Results

Currently, there are no standardized evaluation metrics specifically dedicated to text-to-3D. This is primarily due to the subjective nature of the task and the presence of multiple dimensions that are difficult to quantify. To maintain consistency with existing text-to-3D evaluation methods, we adopt CLIP-based metrics for quantitative analysis. Specifically, we employ variants of the CLIP model, including OpenCLIP ViT-L/14 and CLIP ViT-L/14, to calculate the average CLIP score between the text and its corresponding 3D render. Furthermore, considering the importance of view consistency, we follow previous work calculating A-LPIPS to determine view consistency, quantifying visual

[Figure 109]

𝛾 = 0.1 𝛾 = 0.3 𝛾 = 0.5 𝛾 = 0.7 𝛾 = 0.9

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

“A portrait of IRONMAN, white hair, head, photorealistic, 8K, HDR.”

- Figure 5: Ablation on offset rate. γ = 0.3 achieves optimal visual quality and ensures high consistency between the generated results and the original text. artifacts caused by view inconsistency through calculating the average LPIPS score between adjacent

- 3D scene images. We adopt A-LPIPS as an alternative metric to quantify view consistency, and present it alongside the CLIP scores in our report.

Consistent with the qualitative results, we compare our method with four state-of-the-art text-to-3D methods. Compared with the current best-performing LucidDreamer, our method improves CLIPScore (CLIP-L/14) and CLIP-Score (OpenCLIP-L/14) by 6.83% and 3.85% respectively. At the same time, our method reduces the performance by 20.00% and 30.51% respectively on the A-LPIPS (VGG) and A-LPIPS (Alex) evaluation metrics, showing significant advantages in image authenticity and visual consistency. This improvement is mainly due to our adoption of the more advanced SDXL as a guidance model, which has lower accumulated error. Overall, these results highlight the superior performance of our approach in terms of image quality and text consistency.

#### 4.3 Ablation Study

Ablation on offset rate γ We investigate the impact of the offset rate γ on the generated results (Figure 5), and the best results are achieved when γ is set to 0.3. If γ is set too low, it will result in a loss of color details; if it is set too high, it may destroy the consistency between the generated results and the text. That is, when µ is too close to s, the average effect is too heavy, and its effect is similar to that of ISM. When µ is too close to t, although the cumulative error is reduced, the updated gradient will become very small, easily causing the model to fall into a local optimal state. However, for simple scenes, the results are best when µ is close to t, and the analysis in Appendix A.3.

[Figure 115]

[Figure 116]

w/o gradient clipping

[Figure 117]

[Figure 118]

Ablation on pixel-by-pixel gradient clipping As shown in Figure 6, when pixel-by-pixel gradient clipping is not applied, the instability of the gradients causes abnormal splitting and duplication in normal areas, filling the depth map with noise and making it rough and uneven, thus making the entire facial appearance abnormal. However, after applying pixel-by-pixel gradient clipping, it is clearly observed that the depth map becomes smoother, the texture returns to normal, and the normal facial features are displayed. This comparison demonstrates the effectiveness of our method.

w/ gradient clipping

Figure 6: Ablation on pixelby-pixel gradient clipping.

- 5 Conclusion In this work, we investigate the inconsistency problem produced by ISM during the generation of

- 3D results. To alleviate this problem, we introduce TSM, which leverages dual paths to reduce error accumulation and thereby improve inconsistency. In addition, to simplify the generation of a high-resolution training process, we adopt SDXL as guidance and propose a pixel-by-pixel gradient clipping method to alleviate the abnormal splitting of normal regions in 3D Gaussian splatting caused by SDXL gradient instability. Our experimental results demonstrate that our method can effectively generate high-resolution, high-quality 3D results.

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [2] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22246–22256, 2023.
- [3] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [4] Haoran Duan, Yang Long, Shidong Wang, Haofeng Zhang, Chris G. Willcocks, and Ling Shao. Dynamic unary convolution in transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11):12747–12759, 2023.
- [5] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [6] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. arXiv preprint arXiv:2205.08535, 2022.
- [7] Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. arXiv, December 2021.
- [8] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023.
- [9] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.
- [10] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. arXiv preprint arXiv:2311.11284, 2023.
- [11] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023.
- [12] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885, 2023.
- [13] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-23-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36, 2024.
- [14] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object, 2023.
- [15] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [16] Baorui Ma, Haoge Deng, Junsheng Zhou, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Geodream: Disentangling 2d and geometric priors for high-fidelity and consistent 3d generation. arXiv preprint arXiv:2311.17971, 2023.
- [17] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shapeguided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12663–12673, 2023.
- [18] Xingyu Miao, Yang Bai, Haoran Duan, Fan Wan, Yawen Huang, Yang Long, and Yefeng Zheng. Conrf: Zero-shot stylization of 3d scenes with conditioned radiation fields, 2024.
- [19] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13492–13502, 2022.

- [20] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.
- [21] Hideyuki Nakanishi, Chikara Yoshida, Toshikazu Nishimura, and Toru Ishida. Freewalk: A 3d virtual space for casual meetings. IEEE MultiMedia, 6(2):20–28, 1999.
- [22] Zijie Pan, Jiachen Lu, Xiatian Zhu, and Li Zhang. Enhancing high-resolution 3d generation through pixel-wise gradient clipping. In International Conference on Learning Representations (ICLR), 2024.
- [23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [24] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.
- [25] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023.
- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [27] Ilknur Reiso˘glu, Burcu Topu, Rabia Yılmaz, T Karaku¸s Yılmaz, and Yuksel Gökta¸s. 3d virtual learning environments in education: A meta-review. Asia Pacific Education Review, 18:81–100, 2017.
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [29] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [30] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023.
- [31] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.
- [32] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, October 2020.
- [33] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
- [34] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [35] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023.
- [36] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12619–12629, 2023.
- [37] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021.
- [38] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024.

- [39] Andrew Wodehouse and Mohammed Abba. 3d visualisation for online retail: factors in consumer behaviour. International Journal of Market Research, 58(3):451–472, 2016.
- [40] Zike Wu, Pan Zhou, Xuanyu Yi, Xiaoding Yuan, and Hanwang Zhang. Consistent3d: Towards consistent high-fidelity text-to-3d generation with deterministic sampling prior. arXiv preprint arXiv:2401.09050, 2024.
- [41] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In CVPR, 2024.
- [42] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.
- [43] Joseph Zhu and Peiye Zhuang. Hifa: High-fidelity text-to-3d with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023.

### A Supplemental material

#### A.1 Implementation details

During the optimization process, we train the model for 2500 iterations. To optimize the 3D Gaussian model, we set the learning rates for opacity, scaling, and rotation to 0.05, 0.005, and 0.001 respectively. Furthermore, the learning rate of the camera encoder is set to 0.001. During training, RGB images and corresponding depth maps from 3D Gaussians are used for rendering. Gaussian densification and pruning processes are performed between 100 and 1500 iterations. We select the publicly available stable diffusion of text to images as a guidance model and choose the checkpoint of Stable Diffusion XL1. The guidance scale is 7.5 for all diffusion guidance models.

#### A.2 More Qualitative Results

As shown in Figure 7, we present additional generation results. It can be observed that our Dreamer XL is capable of generating 3D models that are visually high-quality, closely approximate reality, and maintain good consistency.

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

“A DSLR photo of a Cream Cheese Donut.”

“A pillow.” “a red apple.” “A ripe strawberry.”

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

“Marble bust of Theodoros Kolokotronis.”

“a DSLR photo of a cat wearing armor.”

“A plate piled high with chocolate chip cookies.”

“A durian, 8k, HDR.”

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

“A DSLR photo of A Rugged, vintageinspired hiking boots with a weathered leather finish, best quality, 8K, HD.”

“a DSLR photo of A very beautiful tiny human heart organic sculpture made of copper wire and threaded pipes, very intricate, curved, Studio lighting, high resolution.”

“A 3D model of an adorable cottage with a thatched roof.”

“A DSLR photo of A Stylish Air Jordan shoes, best quality, 8K, HD.”

Figure 7: More results generated by our Dreamer XL framework. Please zoom in for details.

#### A.3 More Ablation Study

Ablation on offset rate γ for simple scene Here, we provide more ablation on offset rate γ for simple scene. As shown in Figures 8 and 9, if γ is set too low, the similarity between text and images decreases, and the images appear unrealistic. Raising γ can mitigate the problem of accumulated errors, but it also leads to a loss of some details.

Ablation on pixel-by-pixel gradient clipping Here, we present additional result in Figures 10 and 11. In Figure 10, once gradient clipping is not applied, the generated 3D model tends to be darker in color and lacks realism. In Figure 11, without gradient clipping, the generated objects appear more blurry, and it is evident from the depth map that there are numerous noise and spikes.

###### 1https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

[Figure 155]

𝛾 = 0.1 𝛾 = 0.3 𝛾 = 0.5 𝛾 = 0.7 𝛾 = 0.9

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

“The vibrant red rose, with delicate petals and thorny stem. ”

- Figure 8: More Ablation on offset rate γ for simple scene.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

𝛾 = 0.1 𝛾 = 0.3 𝛾 = 0.5 𝛾 = 0.7 𝛾 = 0.9

“A durian, 8k, HDR. ”

- Figure 9: More Ablation on offset rate γ for simple scene.

[Figure 167]

[Figure 168]

w/o gradient clipping w/ gradient clipping

[Figure 169]

[Figure 170]

- Figure 10: More Ablation on pixel-by-pixel gradient clipping.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

w/o gradient clipping w/ gradient clipping

- Figure 11: More Ablation on pixel-by-pixel gradient clipping.

#### A.4 Limitations

While our method can generate high-quality and relatively realistic 3D models, qualitative results show a significant limitation in our approach regarding light handling. Specifically, we have observed anomalous blue reflections in many scenes. Through our experiments, we have identified this problem as primarily caused by our use of SDXL. When SDXL is applied, the blue channel values in rendered images tend to be large, resulting in numerous areas exhibiting abnormal blue hues after normalization. Despite our attempts, including parameter adjustments and different normalization methods, we have yet to find a viable solution. We speculate that this may be attributed to the gradient or training data of SDXL. Additionally, it’s worth noting that while our research aims to enhance the quality of generated models, it may inadvertently contribute to the advancement of deepfake technology.

