# arXiv:2503.09601v2[cs.CV]13Mar2025

## RewardSDS: Aligning Score Distillation via Reward-Weighted Sampling

### Itay Chachy Guy Yariv Sagie Benaim Hebrew University of Jerusalem

{itay.chachy, guy.yariv, sagie.benaim}@mail.huji.ac.il

“A penguin with a brown bag in the snow.”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

“A man with a beard, wearing a suit, holding a pink briefcase.”

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

MVDream RewardSDS

Figure 1. RewardSDS is a plug-and-play score distillation approach that allows for a reward-aligned generation. It can be applied to various tasks and extend diverse set of distillation approaches, boosting their performance and alignment. Here, we demonstrate it by replacing the standard SDS of the state-of-the-art MVDream [43] approach with RewardSDS for text-to-3D generation.

### Abstract

### 1. Introduction

Score Distillation Sampling (SDS) has emerged as an effective technique for leveraging 2D diffusion priors for tasks such as text-to-3D generation. While powerful, SDS struggles with achieving fine-grained alignment to user intent. To overcome this, we introduce RewardSDS, a novel approach that weights noise samples based on alignment scores from a reward model, producing a weighted SDS loss. This loss prioritizes gradients from noise samples that yield aligned high-reward output. Our approach is broadly applicable and can extend SDS-based methods. In particular, we demonstrate its applicability to Variational Score Distillation (VSD) by introducing RewardVSD. We evaluate RewardSDS and RewardVSD on text-to-image, 2D editing, and text-to-3D generation tasks, showing significant improvements over SDS and VSD on a diverse set of metrics measuring generation quality and alignment to desired reward models, enabling state-of-the-art performance. Project page is available at https://itaychachy. github.io/reward-sds/.

Diffusion models have shown remarkable success in generating high-fidelity and diverse images [37, 38, 40] and videos [5, 12, 44]. However, their success often hinges on the availability of large-scale datasets, a requirement that poses a significant challenge in modalities like 3D content generation. In such data-scarce scenarios, leveraging diffusion models as priors becomes crucial, allowing one to improve generation quality. To this end, Score Distillation Sampling (SDS) was introduced [34, 50], distilling a pretrained 2D diffusion model into a 3D representation by optimizing a loss that encourages rendered samples to have high scores under an underlying 2D diffusion model. While SDS has shown great promise, achieving fine-grained control and alignment with user intent remains a challenge. A different line of recent work has shown that one can utilize reward-based sample selection to align generative models such as LLMs [7, 36, 46] or diffusion models [19, 28, 49]. Motivated by these works, we ask: How can one effectively harness reward-based sample selection to align outputs produced by score distillation?

Building on SDS, several works have explored methods to improve the optimization process. Variational Score Distillation (VSD) [53] optimized the 3D parameters as a random variable instead of a constant using a particle-based variational framework. Several other methods were proposed, either modifying or generalizing SDS [20, 24, 31, 57, 58], or adapting it to the editing setting [10, 23]. In the context of SDS alignment, recently, DreamReward [56] proposed to train a reward model on multi-view images, and subsequently use it to adjust the SDS score. In contrast, our method is compatible with any pre-trained reward model, including those trained on 2D images, eliminating the need for costly multi-view annotations. These works rely on sampling from an underlying pre-trained diffusion model and weighting the contribution of each sample equally. However, some noise samples may correspond to high-reward regions in the output space, while others may lead to low-reward regions. As such, we propose a novel adaptation of SDS, called RewardSDS. We begin by rendering an image x from an underlying model θ. We then draw N noises corresponding to timestep t from a Gaussian distribution, resulting in noisy samples x1t,...xNt . Each noisy sample xit is assigned an alignment score. This score is obtained by first denoising xit using the diffusion model, and then feeding the denoised sample into a given reward model to obtain a reward value, which we term the “alignment score”. The overall loss is then given by a weighted sum of the SDS losses computed for individual noisy samples x1t,...xNt , where the weight is derived from the alignment score. As our adaptation is general and can be broadly applied, we also demonstrate its applicability to Variational Score Distillation (VSD), referred to as RewardVSD. We validate our approach using both RewardSDS and RewardVSD. We first present a controlled study on zeroshot text-to-image generation using a diverse set of prompts from Drawbench [40] and MS-COCO [26]. We evaluate the generated outputs both in terms of image quality and text alignment using a diverse set of metrics. Our evaluation is conducted across multiple pretrained reward models, demonstrating significantly improved performance and better alignment to each reward model. We demonstrate a similar improvement on 2D editing in comparison to Delta Denoising Score [10].

We then consider text-guided 3D generation. Specifically, we follow the state-of-the-art work of MVDream [43] utilizing a diffusion model pretrained on muliview images for our prior. We subsequently optimize either a NeRF [33] or a 3DGs [22] backbone using score distillation on this pretrained diffusion model, following the pipeline of DreamFusion [34] or DreamGaussian [47], replacing the use of the SDS loss with RewardSDS. This results in significantly improved performance. In particular, in comparison to MVDream, which uses a NeRF backbone with standard SDS,

the use of RewardSDS instead significantly boosts performance and reward-model alignment. We also demonstrate the superiority of RewardVSD to standard VSD. Lastly, we provide an extensive ablation study and analyze the time vs performance tradeoff of our approach.

### 2. Related Work

Diffusion Models Alignment Fine-tuning diffusion models for alignment with human preferences has been extensively explored through methods such as reinforcement learning, which optimizes reward signals [4], direct gradient backpropagation from reward functions [55], direct preference optimization based on diffusion likelihood [49], and stochastic optimal control frameworks [8].

Sample selection and optimization have also emerged as an important paradigm for improving diffusion model sample quality and alignment at test time. Some methods focus on metric-guided selection, using random search guided by a reward model [19, 28], or similarity to references [41, 48]. Recent work [1, 59] approximate “good” noise distributions with neural networks for efficient sampling. Unlike these works we consider alignment in the context of score distillation for tasks such as text-to-3D generation.

Score Distillation Sampling Score Distillation Sampling (SDS) [34, 50] has emerged as a essential technique for extending pre-trained 2D diffusion models to modalities like 3D content generation, where large-scale datasets are scarce. SDS leverages the 2D prior from text-to-image diffusion models to optimize rendering parameters, such as Neural Radiance Fields. However, the original SDS formulation exhibits limitations, notably artifacts like oversaturation and over-smoothing.

Several works have aimed to improve the core distillation process with different strategies. Variational Score Distillation (VSD) [53] reformulates SDS as particlebased variational inference, optimizing a distribution of 3D scenes. Noise-Free Score Distillation [20] addresses oversmoothing by eliminating noise terms in the SDS objective. Classifier Score Distillation [57] re-evaluates the role of classifier-free guidance in score distillation. DreamFlow [24] suggests using a predetermined timestep schedule for distillation. RecDreamer [58] proposed a 3D noise consistent distillation approach. JointDreamer [18] further improves multi-view consistency by using view-aware models within a joint distillation framework. SteinDreamer [51] employs Stein’s identity to reduce gradient variance in SDS. DreamTime [15] proposed an improved optimization strategy for score distillation. Furthermore, works like Debiased-SDS [13], Perp-Neg [2], DreamControl [14], and ESD [52] explore improved control over pose and biases in SDS-based generation. SDS-Bridge [31] presents a unified score distillation framework formulated as a bridge between distributions. Delta Denoising Score (DDS) [10] and Pos-

terior Denoising Score [23] adapt SDS for image editing by modifying the gradient direction to better preserve input image details during editing. Beyond these core algorithmic improvements, research has also explored diverse applications of SDS such as generating SVG graphics [16, 17], sketches [54], textures [25, 32], typography [16], and dynamic 4D scenes [3, 45]. Importantly, these advancements are orthogonal to our reward-based approach and offer potential for synergistic combination.

In the context of SDS alignment, DreamReward [56] has recently proposed to align SDS by training a reward model to score a set of multiview images. They then use this model to adjust the SDS score provided by the underlying 2D diffusion model. Our method differs in two main aspects: (1). Our method can be aligned with any pretrained reward model, specifically a reward model trained to align 2D images as well as non-differentiable reward models. In contrast, DreamReward require the expensive annotation of scores for multiview data. (2). DreamReward still perform an expectation over all noises corresponding to a given timestep. Our method, on the other hand, weighs different noises according to their alignment score. As such, one can view our contribution as orthogonal.

### 3. Method

We begin by outlining related background, specifically score distillation sampling (SDS) and variational score distillation (VSD). We then outline our approach as it applies to both SDS and VSD. An illustration is provided in Fig. 2. Score Distillation Sampling (SDS) The optimization process in SDS is derived from the training objective of diffusion models. Given a clean image x0 and a text prompt y, diffusion models are trained to predict the noise ϵt added at timestep t to a noisy image xt. This training objective can be expressed as:

t∼N(0,I) w(t)∥ϵϕ(xt,y,t) − ϵt∥22

L(x0) = Et∼U(0,1),ϵ

where ϵϕ(xt,y,t) is the noise predictor (typically a U-Net), w(t) is a weighting function, and xt is obtained through the forward diffusion process:

xt = αtx0 + σtϵt, ϵt ∼ N(0,I) (1)

where αt,σt are hyperparameters, chosen such that σt2 + αt2 = 1, and σt gradually increases from 0 to 1.

In SDS, when x0 = g(θ) is a rendered image from a differentiable generator g parameterized by θ, the parameters θ are updated by backpropagating the gradient of the loss:

∇θLSDS(x0 = g(θ)) = Et,ϵ

t

∂x0 ∂θ

w(t)(ϵϕ(xt,y,t) − ϵt)

.

This gradient update guides the parameters θ such that the rendered images g(θ) increasingly resemble samples from

the text-conditioned 2D diffusion model. Intuitively, SDS perturbs a rendered image x0 by adding Gaussian noise to obtain xt, then uses the pre-trained diffusion model to predict the noise ϵϕ(xt,y,t) that should be present in xt to move it towards the real image distribution. The difference between the predicted noise and the added noise, weighted by w(t) and backpropagated through ∂x0

∂θ , provides an approximation of the gradient for optimizing θ.

Variational Score Distillation (VSD) Despite its effectiveness, SDS often suffers from issues like over-saturation and over-smoothing. To address these limitations and improve generation quality and diversity, Variational Score Distillation (VSD) [53] was proposed. VSD reformulates SDS within a particle-based variational inference framework. Instead of optimizing a single 3D scene representation, VSD optimizes a distribution µ(θ|y) over scene parameters by introducing multiple particles {θi}ni=1. The VSD objective is defined as:

µ∗ = arg min

Et

µ

σt αt

ω(t)DKL(qtµ(xt|y)∥pt(xt|y))

where qtµ(xt|y) is the distribution of rendered images from the particle set, and pt(xt|y) is the target distribution from the pre-trained diffusion model. To solve this optimization, VSD fine-tunes an additional U-Net, ϵϕ, using LoRA, to estimate the score of the proxy distribution qtµ(xt|y). That is ϵϕ is finetuned on rendered images by {θ(i)}Ki=1 with the standard diffusion objective:

w(t)||ϵϕ(xt,y,t) − ϵt)||22 (2) The gradient for each particle θi in VSD, ∇θi

Et,ϵ

t

LV SD(θi) is then computed as:

Et,ϵ

t

∂g(θi) ∂θi

w(t)(ϵpretrain(xt,y,t) − ϵϕ(xt,y,t))

(3)

where ϵpretrain is the original pre-trained diffusion denoiser, and ϵϕ is the fine-tuned score estimator.

RewardSDS Our core idea is to incorporate a reward model R to guide the SDS optimization by prioritizing noise samples that are more likely to produce high-quality, aligned outputs. To this end, we rank a set of N noise

samples {ϵ(ti)}Ni=1 drawn at each iteration based on the reward scores they induce in the rendered image. Specifically, for a given timestep t and a set of noise samples

{ϵt(i)}Ni=1 ∼ N(0,I), we generate a set of noisy images {xt(i)}Ni=1 using Eq. 1:

x(ti) = αtx0 + σtϵ(ti), i = 1,2,...,N,

where x0 = g(θ) is the rendered image. We then evaluate each noisy image xt(i) by first denoising it and then using a

Scores

Weights

Images

[Figure 13]

[Figure 14]

Render

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

T2I

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

|[Figure 27]|
|---|

+

Reward

.

.

.

. .

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

NeRF / 3DGS

Losses

T2I

[Figure 34]

Reward Weighted SDS Loss

“Two dogs in the park.”

Backpropagate

.

Figure 2. RewardSDS illustration. An image is first rendered from a given view and N random noises are applied (at a given timestep). The noisy images are then scored by denoising them and applying a reward model on the output. These scores are then mapped to corresponding weights, which are used to weigh the contribution of each noisy sample in score distillation.

### 4. Experiments

reward model R to obtain a set of reward scores {r(i)}Ni=1, where r(i) = R(x(ti)). R may optionally accept a text input.

We begin by evaluating our method on zero-shot text-toimage generation, reporting the effect of different reward models. Here, we directly optimize an image and compare the use of RewardSDS and RewardVSD to standard SDS and VSD, respectively. Second, we evaluate our approach on text-to-3D generation. Next we demonstrate the applicability of our approach for image editing by incorporating it into DDS [11], yielding RewardDDS. Lastly, we conduct an extensive ablation study and analyze optimization time vs performance tradeoff of using our method. Detailed implementation information for all experiments is provided in Appendix B.

Based on these reward scores, the SDS loss for each

noise sample ϵ(ti) is then weighted by a factor w(i) that is derived from r(i) (see Sec. 4 for details).

The Reward SDS loss is then computed as a weighted sum of the individual SDS losses, where {w(i)}Ni=1 are the weights assigned based on the reward ranking, and w(t) is the standard SDS weighting function. The gradient for updating the parameters θ, ∇θLR−SDS(x0 = g(θ)), is then given by:

N

1 N

∂x0 ∂θ

w(i) w(t)(ϵϕ(x(ti),y,t) − ϵ(ti))

(4)

Et

Reward models. Following Ma et al. [29], we incorporate CLIPScore [35] (using the ViT-L/14 variant), ImageReward [55], and the Aesthetic Score Predictor [42] as optional reward models. We consider three separate model types, each optimized with respect to the reward produced by a different reward model. Each model serves a distinct role: CLIPScore validates text-image alignment by comparing visual and textual features. The Aesthetic Score Predictor assesses aesthetic quality, as it is trained to predict human ratings of synthesized images’ visual appeal. ImageReward evaluates both alignment and aesthetics, learning general human preferences through a carefully curated annotation pipeline that includes ratings and rankings for text-image alignment, aesthetic quality, and harmlessness.

i=1

RewardVSD In RewardVSD, for each particle θi in the particle set {θi}Ki=1, we sample a set of N noise samples {ϵ(ti,j)}Nj=1. For each noise sample ϵ(ti,j), we compute the noisy image x(ti,j) and its corresponding reward score r(i,j) = R(xt(i,j)). We then rank the noise samples {ϵ(ti,j)}Nj=1 based on their reward scores {r(i,j)}Nj=1 and assign weights {w(i,j)}Nj=1 accordingly. The gradient for updating particle θi, ∇θi

LR−V SD(θi), is:

  1

N

w(i,j) w(t)(ϵpretrain(x(ti,j),y,t) (5)

Et

Metrics. As metrics, we utilize the three presented reward models. While each model type is optimized with respect to a specific reward model, we evaluate it with respect to all reward models. Additionally, we utilize Gemini 2.0

N

j=1

∂g(θi) ∂θi

−ϵϕ(x(ti,j),y,t))

Flash 1 as an LLM Grader, asking it to score each promptimage pair on a scale from 1 to 10 based on criteria such

- as accuracy to the prompt, creativity and originality, visual quality and realism, and consistency and cohesion. We refer to this metric as the LLM Grader and present the average score. The exact prompt used as input for the LLM is sourced from Ma et al. [29], specifically in Fig. 16. To further validate our approach, we construct user studies, where we assess the realism of generated outputs and their alignment to the input text.

#### 4.1. Zero-Shot Text-to-Image Generation

We begin by performing a zero-shot text-to-image generation where we optimize a latent map of size 64 × 64 × 4 (corresponding to an image) in the Stable Diffusion’s latent space [21, 30, 53]. As opposed to text-to-3D, where the choice of model significantly impacts the result, this experiment allows us to measure the impact of our reward-based score distillation more directly.

Effect of different reward models. First, we evaluate the effect of different reward models on our approach. We randomly selected 25 prompts from the Drawbench benchmark [40], a diverse, general-purpose dataset of text prompts spanning multiple categories. We optimized RewardSDS and RewardVSD, using each of the three reward models noted above. Additionally, we report results for the SDS and VSD baselines alone. As shown in Tab. 1, models trained on each of the three reward models effectively improve the baselines. For both RewardSDS and RewardVSD, using the ImageReward model achieves the best overall results, attaining the highest LLM Grader score and the strongest performance on its own reward model. This suggests that ImageReward provides a well-balanced optimization that enhances alignment, aesthetics, and overall human preference. Not surprisingly, in all cases, each reward model excels when evaluated with its corresponding reward model. However, we find that optimizing with respect to one reward model also results in an improved performance with respect to the other reward models.

Qualitative results showcasing RewardSDS with different reward models, along with the SDS baseline, are presented in Fig.3, while results for RewardVSD with different reward models, along with the VSD baseline, are shown in Appendix A.1. These examples illustrate the effect of integrating each reward model into image generation, along with the baseline.

Larger-scale comparison to baselines. To further evaluate our approach, we conduct a larger-scale comparison to baselines. We randomly sample 100 prompts: 50 from MS-COCO [27] and 50 from Drawbench [40]. We use ImageReward as the reward model and report results for CLIPScore, Aesthetic Score, and the LLM Grader. To verify that

1https://ai.google.dev/gemini-api

Reward Model CLIP↑ Aesthetic↑ ImageReward↑ LLM-G↑ RewardSDS

SDS Baseline 27.93 5.42 0.59 6.74 CLIP 28.96 5.49 0.78 6.92 Aesthetic 27.30 5.67 0.64 6.83 ImageReward 28.84 5.54 1.21 7.19

RewardVSD

VSD Baseline 27.41 5.15 0.51 6.73 CLIP 27.99 5.21 0.73 6.87 Aesthetic 27.37 5.35 0.55 6.74 ImageReward 27.80 5.24 1.11 7.03

Table 1. Effect of different reward models on generated outputs using RewardSDS and RewardVSD. Each row represents results obtained by applying our method with a different reward model. The first row corresponds to the baseline SDS or VSD, where no reward model is used. We report scores from three reward models: CLIP (CLIPScore), Aesthetic (Aesthetic Score), ImageReward, along with LLM-G (LLM Grader).

“A black colored car.”

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

“A green apple and a black backpack.”

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

“A white car and red sheep.”

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

SDS Baseline CLIPScore Aesthetic Score ImageReward

Figure 3. Qualitative comparison of generated outputs using different reward models for RewardSDS and the SDS baseline.

better alignment to the given reward model is not compensated by image realism or by alignment to the input text, we conduct a user study in which 50 users are asked to score from 1 (lowest) to 5 (highest): (1). How realistic is the generated image?, (2). How aligned is the generated image to the input text? For each of the 100 prompts used above, we generated corresponding images using SDS, VSD, RewardSDS and RewardVSD. Users are then presented the image corresponding for each method at random,

Method CLIP↑ Aesthetic↑ LLM-G↑ Align.↑ Real.↑ MS-COCO

Method CLIP↑ Aesthetic↑ LLM-G↑ Align.↑ Real.↑ Gaussian Splatting

SDS 27.32 5.59 6.90 2.85 2.66

MVDream 24.71 5.73 4.90 2.74 2.28 RewardSDS 25.24 5.79 5.31 4.11 3.13

- RewardSDS 27.86 5.84 7.10 3.93 3.18

VSD 27.40 5.39 6.85 3.15 2.31 RewardVSD 27.79 5.46 7.03 4.08 2.80

Drawbench SDS 27.65 5.70 6.73 3.18 2.34

- RewardSDS 28.29 5.86 7.11 3.48 2.66

NeRF

MVDream 26.19 5.83 5.86 3.51 3.14 RewardSDS 27.12 5.97 6.07 4.21 3.79

Table 3. Comparison of text-guided 3D generation using MVDream as our backbone, with and without reward-based sampling. We evaluate CLIPScore (CLIP), Aesthetic Score (Aesthetic), and LLM Grader (LLM-G). Additionally, we assess alignment and realism using a user study (MOS on a scale of 1 to 5).

VSD 26.74 5.32 6.64 2.85 2.34 RewardVSD 27.95 5.55 7.02 3.82 2.95

Table 2. Comparison of zero-shot text-to-image generation using RewardSDS/RewardVSD using ImageReward compared to SDS/VSD. We evaluate CLIPScore (CLIP), Aesthetic Score (Aesthetic), and LLM Grader (LLM-G). We also assess image alignment and realsim (user study MOS on a scale of 1 to 5).

a 3DGs [22] backbone using score distillation on this pretrained diffusion model, following the pipeline of DreamFusion [34] or DreamGaussian [47], replacing the use of the SDS loss with RewardSDS. 3DGs was trained on 30 random prompts from the DreamFusion Gallery 2, while NeRF was trained using 22 hand-crafted prompts. We refer readers to Appendix C for the full prompt list. We measure CLIPScore, Aesthetic Score, and LLM Grader as automatic metrics of 10 randomly sampled views from each generated scene and report the average score. We also consider a user study to assess alignment and realism, which follows the procedure of the 2D setting, but where we use 10 random views from each scene.

“A man riding a snowboard on top of a snow covered slope.”

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

“A television is on a tv stand next to a bureau.”

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

As shown in Tab. 3, our method consistently improves all metrics using both 3D Gaussian Splatting and NeRF backbones, demonstrating its effectiveness in enhancing textscene alignment and overall visual quality in 3D generation. Qualitative results, shown in Fig.5, illustrate the superiority of our method over the baseline in both settings. Additional examples can be found in the Appendix A.3, specifically on the attached webpage. In addition, we qualitatively assess the effect of using different reward models. Specifically, in Fig. 6 we demonstrate the effect of using RewardSDS with ImageReward in contrast to Aesthetic reward. In Appendix A.3, we also provide additional results for standard DreamGaussian [47] training (without MVDream pertaining), using both RewardSDS and RewardVSD, showcasing our advantage to standard SDS and VSD.

“A sheep to the right of a wine glass.”

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

SDS RewardSDS VSD RewardVSD

- Figure 4. Qualitative comparison of zero-shot text-to-image generation using SDS, RewardSDS (ours), VSD, RewardVDS (ours)

and asked to rank each image from 1-5 on questions (1)-(2) above. As shown in Tab. 2, our reward-based sampling consistently improves generation performance across all metrics. Qualitative comparisons between RewardSDS, SDS, RewardVSD, and VSD are shown in Fig. 4.

#### 4.3. Image Editing

We extend the editing method of DDS [11].

DDS can be expressed as the difference of SDS of the source prompt and the SDS score of the target prompt. As such, our RewardDDS is simply the difference between two RewardSDS scores. We use five noise candidates and ImageReward as the reward model. We evaluate RewardDDS

#### 4.2. Text-to-3D Generation

We now evaluate our approach on text-to-3D generation. Specifically, we follow the state-of-the-art work of MVDream [43] by first training a multiview diffusion model. We subsequently optimize either a NeRF [33] or

2https://dreamfusion3d.github.io/gallery.html

“Mini China town.”

“A cartoon cat eating a cheesecake.”

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

“A skyscraper that reaches the cloud.”

“A bulldog wearing a black pirate hat.”

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

ImageReward Aesthetic Score

“Two dogs in the park.”

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

- Figure 6. Qualitative comparison of text-to-3D generation using different reward models. We consider a NeRF backbone optimized with RewardSDS, either using the ImageReward reward model or Aesthetic Score reward model. As can be seen, using aesthetic reward results in adding bushes (top row) and a different (more aesthetic) color (both rows).

Method CLIPScore↑ Aesthetic Score↑ LLM Grader↑

DDS 24.00 5.77 6.89 RewardDDS 24.19 5.80 7.09

Table 4. Quantitative comparison of image editing.

“A white lion.” -> “A white wolf.”

“A photo of a flower.” -> “A photo of a snowflake.”

Input DDS RewardDDS

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

- Figure 7. Qualitative comparison of image editing. We compare DDS to our adaptation, RewardDDS.

“A basketball player dunking a basketball.”

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

MVDream RewardSDS

(NeRF Backbone)

“A banana on the left of an apple.”

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

“A DSLR photo of the Imperial State Crown of England.”

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

“Two dogs in the park.”

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

“Two computer monitors on a brown wooden desk.”

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

against DDS on 100 randomly sampled examples from the InstructPix2Pix dataset [6], pairing each target prompt with the generated image and computing automatic metrics (CLIPScore, Aesthetic Score, and LLM Grader). Tab. 4 shows that RewardDDS outperforms DDS across all metrics. A qualitative comparison is given in Fig. 7.

MVDream RewardSDS

(3DGs Backbone)

- Figure 5. Qualitative comparison of text-to-3D generation based on (a) NeRF and (b) 3DGs, in comparison to MVDream.

#### 4.4. Ablation Studies

Our method allows for a diverse set of design choices, which we analyze here. We consider a common experimental setup for all ablations: We evaluate zero-shot text-toimage generation on 25 randomly sampled prompts from the Drawbench benchmark using ImageReward as the reward model and the LLM Grader as the primary metric. We use RewardSDS as our method, while the baseline is SDS.

Noise weighting. We ablate the choice of noise weighting (w(i)), applied to the N candidate noise samples drawn

- at each iteration. Specifically, we compare the following strategies: (i) Random, where a weight of 1 is assigned to a randomly selected candidate and 0 to all others; (ii) Softmax, where a softmax function is applied to the reward scores; (iii) Winner-takes-all, in which only the candidate with the highest reward is used (assigned a weight of 1, while the rest is assigned a weight of 0);

(iv) Two winners-take-all (as in (iii) but with two highest-reward candidates); (v) Step towards best, away from worst, which takes a positive step toward the candidate with the highest reward (weight 0.9) while subtracting the influence of the lowest-reward candidate (weight -0.1); (vi) Step towards top-2, away from bottom-2, which is as in (v) but with two highest and two-lowest reward samples. Tab. 5 summarizes the performance of each scheme in terms of the LLM Grader score.

Scheme LLM-G↑

- (i) 6.69
- (ii) 7.08
- (iii) 7.05
- (iv) 7.13
- (v) 7.12
- (vi) 7.17

Table 5. Effect of different noise weighing schemes.

Reward-based optimization steps. We now consider whether RewardSDS can be applied for a smaller number of steps K out of the total number of optimization steps, while using standard SDS for the remaining steps. We vary K from 0 to 1000 (out of a total of 1000 optimization steps), in increments of 100. As shown in Fig. 8 (top graph), performance improves as K increases, with noticeable saturation around K ≈ 600 steps. Importantly, even for K = 100, we observe superior results compared to K = 0.

Effect of the number of noises sampled. Next, we ablate the number of noise samples sampled (and weighted) at each optimization step, denoted as N (as in Eq. 4). We test four different values of N: 1, 3, 7, and 10, where N = 1 serves as the baseline (no reward model). The results, presented in Fig. 8 (middle graph), indicate that increasing N consistently improves performance. Note the significant improvement already from N = 1 to N = 3. Qualitative results, shown in Appendix A.2 further illustrate this.

Number of inference steps for noise selection. We consider the effect of the number of denoising steps (S) used to denoise a noisy sample. We consider three different values: S = 1, S = 8, and S = 15, alongside a baseline

[Figure 105]

- Figure 8. The top graph shows the impact of the number of rewardbased steps (K), the middle graph presents the impact of the number of reward-based steps (S), and the bottom graph illustrates the impact of the number of considered noises (N).

[Figure 106]

- Figure 9. Tradeoff between running time and generation quality.

without using the reward model (S = 0). As can be seen in Fig. 8 (bottom graph), increasing S consistently improves performance, with minimal increase in steps (S = 1) already outperforms the baseline.

Time vs. quality tradeoff. In the ablations above, by experimenting with N, K, and S we observed two main conclusions: (i) increasing N, K, and S improves performance, albeit at the cost of longer running time, and (ii) even small increases in these parameters already yield improvements over the baseline. Therefore, we further analyze the concrete tradeoff between quality and running time. Specifically, we evaluate four scenarios: (i) Baseline – regular SDS, i.e., N = 1,K = 0,S = 0. (ii) Small scale – minimal additional running time, i.e., N = 2,K = 100,S = 1. (iii) Medium scale – with N = 5,K = 500,S = 8. (iv) Large scale – with N = 10,K = 1000,S = 15. In addition to reporting the LLM Grader’s score, we also report the average running time required to optimize a single image (in seconds). The results, as shown in Fig. 9, suggest that the small-scale setting provides improvements over the baseline while maintaining a similar order of magnitude in running time (67 sec. vs. 45 sec.), while medium and large-scale settings significantly outperform the baseline in quality.

Different SDS-based methods. Lastly, we demonstrate the versatility of our approach by showcasing its plug-andplay nature. Specifically, we integrate our method on SDSBridge [30], resulting in RewardSDS-Bridge. As SDS-

Method CLIPScore Aesthetic Score LLM Grader

SDS-Bridge 27.01 5.34 6.44 RewardSDS-Bridge 27.94 5.58 7.31

Table 6. Comparison of SDS-Bridge and RewardSDS-Bridge on zero-shot text-to-image generation.

Bridge adapts the input condition, our adaptation is otherwise the same as for SDS. We perform further evaluation via CLIPScore, Aesthetic Score, and the LLM Grader. Tab. 6 presents the results for SDS-Bridge and RewardSDSBridge, with our method consistently improving performance across all metrics. Qualitative comparisons can be found in Appendix A.4.

### 5. Conclusion

To conclude, we have presented RewardSDS, a novel approach for addressing the critical challenge of aligning score distillation with user intent through reward-weighted noise sample selection. Across extensive evaluations encompassing zero-shot text-to-image generation, text-to-3D creation, and image editing, RewardSDS, and its VSD variant, RewardVSD, consistently and significantly outperformed standard SDS and VSD baselines across diverse metrics, including CLIPScore, Aesthetic Score, ImageReward, LLM Grader assessments, and user studies, with a model based on ImageReward notably demonstrating robust performance. The general nature of RewardSDS facilitates seamless integration with various SDS extensions and pre-trained reward models, offering a flexible framework for improved alignment.

### References

- [1] Donghoon Ahn, Jiwon Kang, Sanghyun Lee, Jaewon Min, Minjae Kim, Wooseok Jang, Hyoungwon Cho, Sayak Paul, SeonHwa Kim, Eunju Cha, et al. A noise is worth diffusion guidance. arXiv preprint arXiv:2412.03895, 2024. 2
- [2] Mohammadreza Armandpour, Ali Sadeghian, Huangjie Zheng, Amir Sadeghian, and Mingyuan Zhou. Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. arXiv preprint arXiv:2304.04968, 2023. 2
- [3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006, 2024. 3
- [4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 2
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,

- Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [6] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions,

2023. 7

- [7] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher R´e, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024. 1
- [8] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861, 2024. 2
- [9] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, Zi-Xin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/ threestudio, 2023. 13
- [10] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2328–2337, 2023. 2
- [11] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score, 2023. 4, 6
- [12] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1
- [13] Susung Hong, Donghoon Ahn, and Seungryong Kim. Debiasing scores and prompts of 2d diffusion for view-consistent text-to-3d generation. Advances in Neural Information Processing Systems, 36:11970–11987, 2023. 2
- [14] Tianyu Huang, Yihan Zeng, Zhilu Zhang, Wan Xu, Hang Xu, Songcen Xu, Rynson WH Lau, and Wangmeng Zuo. Dreamcontrol: Control-based text-to-3d generation with 3d self-prior. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5364–5373,

2024. 2

- [15] Yukun Huang, Jianan Wang, Yukai Shi, Boshi Tang, Xianbiao Qi, and Lei Zhang. Dreamtime: An improved optimization strategy for diffusion-guided 3d generation. arXiv preprint arXiv:2306.12422, 2023. 2
- [16] Shir Iluz, Yael Vinker, Amir Hertz, Daniel Berio, Daniel Cohen-Or, and Ariel Shamir. Word-as-image for semantic typography. ACM Transactions on Graphics (TOG), 42(4): 1–11, 2023. 3
- [17] Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1911–1920, 2023. 3
- [18] Chenhan Jiang, Yihan Zeng, Tianyang Hu, Songcun Xu, Wei Zhang, Hang Xu, and Dit-Yan Yeung. Jointdreamer: Ensuring geometry consistency and text congruence in text-to-3d generation via joint score distillation. In European Confer-

- ence on Computer Vision, pages 439–456. Springer, 2024. 2
- [19] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. If at first you don’t succeed, try, try again: Faithful diffusion-based text-to-image generation by selection. arXiv preprint arXiv:2305.13308, 2023. 1, 2
- [20] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. arXiv preprint arXiv:2310.17590, 2023. 2
- [21] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation, 2023. 5
- [22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering, 2023. 2, 6
- [23] Juil Koo, Chanho Park, and Minhyuk Sung. Posterior distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13352–13361, 2024. 2, 3
- [24] Kyungmin Lee, Kihyuk Sohn, and Jinwoo Shin. Dreamflow: High-quality text-to-3d generation by approximating probability flow. arXiv preprint arXiv:2403.14966, 2024. 2
- [25] Kehan Li, Yanbo Fan, Yang Wu, Zhongqian Sun, Wei Yang, Xiangyang Ji, Li Yuan, and Jie Chen. Learning pseudo 3d guidance for view-consistent texturing with 2d diffusion. In European Conference on Computer Vision, pages 18–34. Springer, 2024. 3
- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 2
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft coco: Common objects in context, 2015. 5
- [28] Yujian Liu, Yang Zhang, Tommi Jaakkola, and Shiyu Chang. Correcting diffusion generation through resampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8713–8723, 2024. 1, 2
- [29] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, YuChuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Saining Xie. Inference-time scaling for diffusion models beyond scaling denoising steps, 2025. 4, 5
- [30] David McAllister, Songwei Ge, Jia-Bin Huang, David W. Jacobs, Alexei A. Efros, Aleksander Holynski, and Angjoo Kanazawa. Rethinking score distillation as a bridge between image distributions, 2024. 5, 8
- [31] David McAllister, Songwei Ge, Jia-Bin Huang, David Jacobs, Alexei Efros, Aleksander Holynski, and Angjoo Kanazawa. Rethinking score distillation as a bridge between image distributions. Advances in Neural Information Processing Systems, 37:33779–33804, 2025. 2
- [32] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF

- conference on computer vision and pattern recognition, pages 12663–12673, 2023. 3
- [33] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis, 2020. 2, 6
- [34] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion, 2022. 1, 2, 6
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 4
- [36] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 1
- [37] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen, Heewoo Jun, Pamela Brown, Christine Clark, Alec Radford, Ilya Sutskever, et al. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1
- [39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 12
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 1, 2, 5
- [41] Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, and Gal Chechik. Generating images of rare concepts using pretrained diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4695–4703, 2024. 2
- [42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 4
- [43] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation, 2024. 1, 2, 6, 13
- [44] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 1

- [45] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280,

2023. 3

- [46] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024. 1
- [47] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653,

2023. 2, 6, 13

- [48] Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. Realfill: Reference-driven generation for authentic image completion. ACM Transactions on Graphics (TOG), 43(4):1–12, 2024. 2
- [49] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 1, 2
- [50] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12619–12629, 2023. 1, 2
- [51] Peihao Wang, Zhiwen Fan, Dejia Xu, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, et al. Steindreamer: Variance reduction for text-to-3d score distillation via stein identity. arXiv preprint arXiv:2401.00604, 2023. 2
- [52] Peihao Wang, Dejia Xu, Zhiwen Fan, Dilin Wang, Sreyas Mohan, Forrest Iandola, Rakesh Ranjan, Yilei Li, Qiang Liu, Zhangyang Wang, et al. Taming mode collapse in score distillation for text-to-3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9037–9047, 2024. 2
- [53] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation, 2023. 2, 3, 5
- [54] Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. Diffsketcher: Text guided vector sketch synthesis through latent diffusion models. Advances in Neural Information Processing Systems, 36:15869–15889, 2023. 3
- [55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation, 2023. 2, 4
- [56] Junliang Ye, Fangfu Liu, Qixiu Li, Zhengyi Wang, Yikai Wang, Xinzhou Wang, Yueqi Duan, and Jun Zhu. Dreamreward: Text-to-3d generation with human preference. In European Conference on Computer Vision, pages 259–276. Springer, 2024. 2, 3

- [57] Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, SongHai Zhang, and Xiaojuan Qi. Text-to-3d with classifier score distillation. arXiv preprint arXiv:2310.19415, 2023. 2
- [58] Chenxi Zheng, Yihong Lin, Bangzhen Liu, Xuemiao Xu, Yongwei Nie, and Shengfeng He. Recdreamer: Consistent text-to-3d generation via uniform score distillation. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [59] Zikai Zhou, Shitong Shao, Lichen Bai, Zhiqiang Xu, Bo Han, and Zeke Xie. Golden noise for diffusion models: A learning framework. arXiv preprint arXiv:2411.09502,

2024. 2

### A. Additional Results

Method CLIPScore Aesthetic LLM Grader

#### A.1. Qualitative Comparison of the Effect of Different Reward Models on RewardVSD

SDS 21.49 5.34 3.85 RewardSDS 22.87 5.53 4.29

To better understand the effect of different reward models in the RewardVSD framework, we provide a qualitative comparison. Fig. 10 illustrates the generated outputs using different reward models and the VSD baseline.

VSD 21.38 5.14 3.67 RewardVSD 22.03 5.41 4.11

Table 7. Comparison of text-guided 3D generation using standard SDS and VSD (with stable-diffusion-2.1-base), with and without reward-based sampling, over 30 randomly sampled prompts from DreamFusion gallery, with each score averaged over 10 randomly rendered views.

“Two dogs on the street.”

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

process. As discussed in the ablation studies, increasing N leads to better alignment with the desired reward model and enhances the overall image quality. Fig. 11 provides qualitative examples illustrating how different values of N affect the final generated outputs. As shown, larger values of N result in more refined and coherent images, whereas smaller values introduce more variability and potential misalignment.

“A white car and a red sheep.”

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

VSD Baseline CLIPScore Aesthetic Score ImageReward

#### A.3. Additional Text-to-3D results

- Figure 10. Qualitative comparison of generated outputs using different reward models for RewardVSD and the VSD baseline. Each row corresponds to a different reward model, with the input prompts shown at the bottom, taken from Drawbench.

RewardSDS

“A green apple and a black backpack.”

N = 1

N = 7

N = 3

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

“A banana on the left of an apple.”

RewardVSD RewardVSD RewardVSD

- Figure 11. Qualitative results illustrating the effect of the number of considered noises (N). The top row presents the baseline method, while the input prompts are displayed below the images.

Tab. 7 provides a comparison of text-guided 3D generation using standard SDS and VSD (with stable-diffusion-2.1base) in comparison to RewardSDS and RewardVSD, over 30 randomly sampled prompts from DreamFusion gallery, with each score averaged over 10 randomly rendered views. Unlike the main text, here we train on a standard diffusion model, not pretrained on multiview images as in MVDream. In Fig. 12, we provide corresponding text-to-3D results for SDS compared to RewardSDS and VSD compared to our RewardVSD.

#### A.4. Qualitative Comparison of RewardSDSBridge vs. SDS-Bridge

Building upon the comparisons presented in the ablation study, we provide further qualitative results comparing RewardSDS-Bridge with SDS-Bridge in Fig. 13. These results highlight the improved sample quality and alignment with text conditions achieved by our approach.

### B. Implementation details.

All experiments, including both our method and the baselines, we use a single L40s GPU. For all zero-shot textto-image, and image editing experiments, optimization is performed using the ADAM optimizer with a learning rate of 0.01. As the text-to-image model, we employ Stable Diffusion 2.1 Base [39], setting the classifier-free guidance (CFG) scale to 100 for SDS-based experiments and 7.5 for VSD-based experiments. Each 2D image is generated in 1,000 optimization steps using N = 10 (the num-

#### A.2. Qualitative Comparison of the Effect of Number of Considered Noises

The number of noise samples (N) considered at each optimization step plays a crucial role in guiding the generation

SDS

RewardSDS SDS RewardSDS

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

“A DSLR photo of a bagel filled with cream cheese and lox.”

“A silver platter piled with fruits."

RewardVSD RewardVSD

VSD VSD

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

“Colorful camping tent in a patch of grass.” “A palm tree, low poly, 3D model.”

- Figure 12. Qualitative results illustrating text-to-3D results for SDS compared to RewardSDS and for VSD compared to our RewardVSD (without MVDream’s pretraining).

“A yellow book and a red vase.”

RewardSDSBridge

SDS-Bridge

“A couch on the left of a chair.”

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

RewardSDSBridge

SDS-Bridge

- Figure 13. Qualitative comparison between SDS-Bridge and RewardSDS-Bridge.

and we employ the public implementations of DreamGaussian [47] for 3DGs backbone optimization (leaving their second stage of texture refinement as is) and MVDream [9] (without shading) for NeRF based optimization. We use the same settings as in the public implementations, and in NeRF training, we apply our method only in the first half of the optimization as we found this is sufficient for convergence with our method. Regarding the weighting strategy, we use the same one as in the 2D. Optimization of 3DGs takes 60 minutes and NeRF takes 10 hours.

ber of noise samples drawn at each iteration), K = 1000 (the number of reward-based optimization steps used during generation), and S = 15 (the number of inference steps applied during noise selection to obtain refined reward estimates). The weighting strategy assigns a weight of 0.9 to the two highest scoring samples, a weight of -0.1 to the two lowest scoring samples, and 0 to the rest. With those settings, image optimization takes 2227 seconds. For image editing experiments, generation is performed in 200 steps, taking 211 seconds per image, and we use N = 5, K = 200, and S = 1 as the hyperparameters, with the same weighting strategy as above. For text-guided 3D generation, MVDream [43] is used as our text-to-image prior,

### C. Hand-Crafted Prompts for NeRF-Based MVDream Evaluation

As described in Section 4.2, to evaluate NeRF-based 3D generation using MVDream, with and without our method, we utilized a set of 22 hand-crafted prompts. These prompts were carefully designed to cover a diverse range of objects, scenes, and subjects, ensuring a comprehensive assessment of text-to-3D generation quality. The full list of prompts is provided in Table 8.

# Prompt

- 1 A basketball player dunking a basketball
- 2 A basketball player in a red jersey, high resolution, 4K
- 3 A bulldog wearing a black pirate hat
- 4 A cartoon cat eating a cheesecake, realistic
- 5 a DSLR photo of a ghost eating a hamburger
- 6 A guitar player on stage, high quality, realistic, HD, 8K
- 7 A man with a beard, wearing a suit, holding a pink briefcase, high quality, realistic, HD
- 8 A penguin with a brown bag in the snow
- 9 A man with a red scarf, highly detailed, 4K
- 10 A shitzu with a bowtie, high quality, realistic, HD, 8K
- 11 A skyscraper that reaches the clouds, high quality, realistic
- 12 A tiger in the jungle, high quality, realistic, HD
- 13 A white sofa next to a brown wooden table
- 14 A young girl flying a kite, high quality
- 15 An astronaut riding a horse
- 16 Argentinian football player, celebrating a goal, HD
- 17 Corgi riding a rocket
- 18 King Kong climbing the Empire State Building
- 19 Mini China town, highly detailed, 8K, HD
- 20 Red drum set, high quality, realistic, HD, 8K
- 21 Two dogs in the park
- 22 World cup trophy, high quality, realistic, HD

Table 8. List of hand-crafted prompts used to evaluate NeRFbased MVDream with and without RewardSDS.

