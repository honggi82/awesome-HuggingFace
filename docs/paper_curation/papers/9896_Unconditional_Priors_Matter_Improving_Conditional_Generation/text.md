# arXiv:2503.20240v3[cs.CV]10Feb2026

### Unconditional Priors Matter! Improving Conditional Generation of Fine-Tuned Diffusion Models

Prin Phunyaphibarn Phillip Y. Lee Jaihoon Kim Minhyuk Sung KAIST

{prin10517, phillip0701, jh27kim, mhsung}@kaist.ac.kr

Zero-1-to-3: Novel View Synthesis

###### Versatile Diffusion: Image Variations

Input Input Input Input

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

|120°|
|---|

|120°|
|---|

+

+

Zero-1-to-3 + Ours Versatile Diffusion + Ours Versatile Diffusion + Ours

Zero-1-to-3 + Ours

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

InstructPix2Pix: Image Editing DynamiCrafter: Image+Text-to-Video

Input

Input

Input

[Figure 13]

[Figure 14]

[Figure 15]

“Make both shoes purple with white laces.”

“Make the fairy hold a crystal ball.”

“A woman carrying a bundle of plants over their head.”

+

## +

+

DynamiCrafter

InstructPix2Pix + Ours InstructPix2Pix + Ours + Ours

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

Figure 1. Unconditional Priors Matter in CFG-Based Conditional Generation. Fine-tuned conditional diffusion models often show drastic degradation in their unconditional priors, adversely affecting conditional generation when using techniques such as CFG [28]. We demonstrate that leveraging a diffusion model with a richer unconditional prior and combining its unconditional noise prediction with the conditional noise prediction from the fine-tuned model can lead to substantial improvements in conditional generation quality. This is demonstrated across diverse conditional diffusion models including Zero-1-to-3 [46], Versatile Diffusion [64], InstructPix2Pix [7], and DynamiCrafter [62].

##### Abstract

Classifier-Free Guidance (CFG) is a fundamental technique in training conditional diffusion models. The common practice for CFG-based training is to use a single network to learn both conditional and unconditional noise prediction, with a small dropout rate for conditioning. However, we observe that the joint learning of unconditional noise with limited bandwidth in training results in poor priors for the unconditional case. More importantly, these poor unconditional noise predictions become a serious reason for degrading the quality of conditional generation. In-

spired by the fact that most CFG-based conditional models are trained by fine-tuning a base model with better unconditional generation, we first show that simply replacing the unconditional noise in CFG with that predicted by the base model can significantly improve conditional generation. Furthermore, we show that a diffusion model other than the one the fine-tuned model was trained on can be used for unconditional noise replacement. We experimentally verify our claim with a range of CFG-based conditional models for both image and video generation, including Zero-1-to-3, Versatile Diffusion, DiT, DynamiCrafter, and InstructPix2Pix.

##### 1. Introduction

In recent years, diffusion models [29, 56, 59] have shown great success in generation tasks, becoming the de facto standard generative model across many data modalities such as images [50–53], video [4, 5, 30, 66], and audio [13, 33, 44]. The success of diffusion models is not only due to their high-quality results and ease of training, but also the simplicity of adapting them into conditional diffusion models. While previous generative models such as GANs [23] and VAEs [39] require separate training for each conditional generation task, making it costly to create various conditional generative models, diffusion models introduced a considerably more effective approach: training an unconditional model (or a conditional model with simple conditions, such as text) as a base and branching out into multiple conditional models.

At the core of the extendability of diffusion models in easily converting an unconditional (or less conditioned) base model into a conditional (or more conditioned) model is the Classifier-Free Guidance (CFG) [28] technique. CFG proposed to learn to predict both unconditional and conditional noises using a single neural network, without introducing another network, such as a classifier, as in the classifier-guidance [15] approach. CFG combines unconditional and conditional noise predictions to generate data conditioned on a given input. It has been widely adopted not only for training a conditional model from scratch but also for fine-tuning a base model to incorporate other conditions, by adding encoders for the conditional input. Many successful conditional generative models have been fine-tuned using CFG from a base model. For example, Zero-1-to3 [46] and Versatile Diffusion [64] use variants of Stable Diffusion [52] (SD) as a base, with additional encoders to incorporate the input image as conditions, while InstructPix2Pix [7] uses SD1.5 as a base and incorporates text editing instructions and input reference images as conditions to perform instruction-based image editing.

Despite its successes and widespread usage, fine-tuning a conditional model from a base model using the CFG technique has limitations, most notably producing lower-quality results for unconditional generation. This is because both conditional and unconditional noise are learned by the same noise prediction network, thus sharing the limited capacity of the neural network. Typically, the bandwidth allocated for the unconditional noise is even more limited by setting a 5-20% drop rate of the condition, an issue which is exacerbated when the training data is limited or the model is fine-tuned multiple times. More importantly, the low quality of unconditional noise also negatively affects the quality of conditional generation, since conditional generation is performed by combining both conditional and unconditional noise predictions in the CFG formulation.

The crucial oversight in this practice is that the base

model already provides useful guidance for unconditional generation, and the quality of its generated outputs is generally much better than that of the fine-tuned model. Hence, we demonstrate that in conditional generation using a finetuned diffusion model and CFG, simply replacing the unconditional noise of the fine-tuned diffusion model in CFG with that of the base model leads to significant improvements. This is a training-free solution that requires no additional training or modifications to the neural networks. It also highlights that when fine-tuning a diffusion model with additional conditioning using CFG, the unconditional noise does not need to be learned jointly.

Surprisingly, we show that the unconditional noise does not need to come from the base unconditional model used for fine-tuning, but can also come from other pretrained diffusion models. This further eliminates the need to jointly learn both unconditional and conditional noise using the same noise prediction network when a pretrained unconditional diffusion model is available.

##### 2. Related Works

Guidance in Diffusion Models. Classifier-Free Guidance (CFG) [28] has become the de facto guidance technique for conditional generation with diffusion models, leading to notable improvements in both condition alignment and image quality. However, recent research has highlighted some of its limitations. Kynk¨a¨anniemi et al. [41] have shown that the specific timesteps at which CFG is applied significantly impact image diversity, and proposed to restrict CFG to certain intervals.

Another line of work [1, 31] addresses the limited applicability of CFG for text-based conditions when using offthe-shelf diffusion models like Stable Diffusion [52]. These approaches introduce a guidance technique that extends to a broader range of generation tasks, including unconditional generation, inverse problems, and conditional generation with non-text conditions (e.g., depth maps [67]). However, most works have not explored how the dynamics of CFG shift when a diffusion model is fine-tuned for a specific task [7, 46, 64]. Motivated by transfer learning, Zhong et al. [70] propose to improve the generation quality of fine-tuned DiTs [50] by using the unconditional noise from the original model. However, they consider only small class-conditional DiTs while our method applies to popular large-scale diffusion models across different architectures (even when the base and fine-tuned models have different network architectures) and in cases where the base and fine-tuned model conditions have different modalities. Based directly on empirical observations of unconditional noise degradation in large-scale diffusion models that occurs during fine-tuning, we propose a solution by combining noise predictions from various diffusion models.

Karras et al. [38] proposed Autoguidance which uses the

noise estimate from an under-trained version of itself, instead of unconditional noise, to improve image quality. In Autoguidance, both versions of the model are conditioned on the same condition to isolate the image quality improvement from the improvement in condition-alignment. In contrast, our method combines models whose conditions have different modalities, correcting the degradation that stems from the quality of the unconditional prior used in the CFG formulation. Furthermore, Autoguidance emphasizes the importance of designing the degradations to match the degradation of the conditional model. They show that certain degradations will degrade rather than improve the generation quality. One of our main contributions is showing that the fine-tuned unconditional degradation hurts rather than helps the quality of conditional generation.

Merging Diffusion Models. Aligned with the mixtureof-experts [8] and model merging [65] literature on foundation models, there is growing research on methods for merging diffusion models to enable effective composition of multiple conditions. Diffusion Soup [3] directly merges weights of different diffusion models, Mix-of-Show [25] combines the weights of LoRA adapters [32], and MaxFusion [48] merges intermediate model features. Notably, leveraging the iterative denoising process of diffusion models, merging their noise estimates has emerged as a simple yet powerful technique for composing conditions. By merging noise estimates from the same diffusion model with different input conditions, it becomes possible to generate outputs that contain a combination of these conditions [2, 17, 18, 21, 22, 68]. Interestingly, multiple studies have shown that noise estimates from different diffusion models [11, 20, 47] can also be merged effectively. In this work, we extend this approach, demonstrating how merging noise estimates can enhance generation quality when applying CFG to fine-tuned models.

##### 3. Background

###### 3.1. Diffusion Models

Diffusion models [29, 56, 57] generate data by sampling from a given distribution (e.g., Gaussian) and applying iterative denoising. In the forward process, random noise is applied to clean data x0 following:

xt = √α¯tx0 + √1 − α¯tϵ (1)

where ϵ ∼ N(0,I) and α¯t ∈ [0,1]1 [29]. In the reverse process, the noisy data xt is denoised by modeling the transition as a Gaussian distribution:

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),σt2I) (2)

1With variance schedule {βt}Tt=1, αt := 1−βt and α¯t := ts=1 αs.

where the variance σt2 is predefined, and predicting the posterior mean µθ(xt,t) can be reparameterized as a noise prediction task using Tweedie’s formula [19] as below:

√1 − α¯tϵθ(xt)) (3) =: µ˜ (xt,g(xt,ϵθ(xt))) (4)

1 √α¯t

(xt −

µθ(xt,t) = µ˜t xt,

where ϵθ(xt)2 is the noise prediction from a diffusion model, and µ˜t(xt,x0) is the forward process posterior mean. Eq. 3 can be further interpreted as updating the posterior mean towards the prediction of a clean observation from xt using Tweedie’s formula. We define this clean observation predicted from Tweedie’s formula as the function g(·,·) and denote the predicted clean observation by x0|t.

DDIM Sampling. DDIM [57] enables efficient sampling for diffusion models by modeling the non-Markovian transition q(xt−1|xt,x0), conditioned on x0. One denoising step of DDIM is presented as the following deterministic transition:

xt−1 = √α¯t−1g(xt,ϵθ(xt)) + 1 − α¯t−1ϵθ(xt). (5)

###### 3.2. Classifier-Free Guidance (CFG) [28]

For a diffusion model to perform conditional generation given a condition c, it needs to sample from the conditional distribution p(x|c). One approach is to use a classifier to guide the sampling process toward the conditional distribution [15]; however, it comes at the cost of training a separate classifier. Alternatively, Ho and Salimans [28] eliminated the need for a separate classifier by introducing Classifier-Free Guidance (CFG), a straightforward modification to the training and sampling process of diffusion models. In CFG training, the model learns to predict the noise in xt at timestep t not only when a condition c is given, but also when a null condition ∅ is given. That is, the diffusion model performs both conditional (i.e., ϵθ(xt,c)) and unconditional (i.e., ϵθ(xt,∅)) noise prediction. This is achieved by setting the condition c to the null condition ∅ with a certain probability during training. With a model trained using this technique, CFG can be applied in the sampling process by replacing ϵθ(xt) in Eq. 5 with:

ϵ(θγ)(xt,c) = ϵθ(xt,∅) + γ(ϵθ(xt,c) − ϵθ(xt,∅)), (6)

where γ is a guidance scale. A detailed algorithm of CFG with DDIM sampling [57] is shown in Alg. 1.

Analysis of CFG. Based on the connection between diffusion models and score-based models [58, 59], ϵθ(xt,c) and ϵθ(xt,∅) model the conditional score ∇xt

log p(xt|c)

2We omit timestep t in ϵθ(xt) and g(xt, ϵθ(xt)) for brevity.

Algorithm 1 DDIM Sampling with CFG

- 1: xT ∼ N(0,I)
- 2: for t = T,...,1 do
- 3: ϵ(θγ)(xt,c) = ϵθ(xt,∅) + γ(ϵθ(xt,c) − ϵθ(xt,∅))
- 4: x0|t = g(xt,ϵ(θγ)(xt,c))
- 5: xt−1 = √α¯t−1x0|t + √1 − α¯t−1ϵ(θγ)(xt,c)

- 6: end for
- 7: return x0

log p(xt) (up to a scaling factor), respectively. We can interpret the CFG noise ϵ(θγ)(xt,c) in Eq. 6 as an approximation of the true score ∇xt

and the unconditional score ∇xt

γ

t|c) p(xt)

log pγ(xt|c) where pγ(xt|c) := p(xt) p(x

is

the gamma-powered distribution. We can rewrite pγ(xt|c) as follows:

γ

p(xt,c) p(c)p(xt)

∝ p(xt)p(c|xt)γ.

pγ(xt|c) = p(xt)

Thus, CFG guides the samples via the implicit classifier p(c|xt)γ. Using γ > 1 results in sharpening the mode corresponding to c which leads to better conditionalignment [28] and image quality [38].

##### 4. Unconditional Priors Matter

In this section, we discuss the negative impact of degraded unconditional noise estimates in diffusion models fine-tuned on a narrower task-specific data distribution (Sec. 4.1). We then present a simple yet effective approach to enhance the generation quality of these models by leveraging richer unconditional noise estimates from other pretrained diffusion models (Sec. 4.2).

###### 4.1. Poor Unconditional Priors Affect Conditional Generation

The CFG training technique, introduced in Sec. 3, is also commonly used for fine-tuning an existing diffusion model to incorporate new types of input conditions [7, 36, 46, 55, 64]. Consider a pretrained diffusion model, such as Stable Diffusion [52], referred to as the base model, parameterized by ψ. This base model can be fine-tuned on task-specific datasets to incorporate certain types of input conditions, such as camera poses [46] or a reference image [64]. We refer to the resulting model as the fine-tuned model, parameterized by θ.

After fine-tuning the base model with CFG training, conditional generation with the fine-tuned model is performed by combining conditional and unconditional noise predic-

tions following Eq. 6, yielding the CFG noise ϵ(θγ)(xt,c). However, we observe a significant quality drop in uncon-

ditional generation with the fine-tuned model compared to the base model. As shown in Fig. 2, the unconditional outputs from fine-tuned models [7, 46, 64] clearly lack detailed

semantics and exhibit lower image quality. These results are expected, as (1) the unconditional distribution is inherently more complex than the conditional distribution, and (2) only a small fraction of the training data is utilized in each training iteration due to the low CFG dropping probability (typically 5-20%).

Importantly, we observe that the quality of the conditional generation in these fine-tuned models is negatively impacted by poor unconditional priors. This degradation can be understood through the CFG [28] formulation: CFG is designed to sample from the gamma-powered distribution pγ(xt|c) ∝ p(xt)p(c|xt)γ. Poor unconditional priors introduce approximation errors to p(xt), which in turn affects both p(xt) and p(c|xt) ∝ p(x

t|c)

p(xt) . Although a key advantage of CFG is the joint modeling of the unconditional and conditional distributions, we find that under limited data or multiple fine-tuning, the fine-tuned model loses the rich unconditional prior of the base model, leading to quality degradation.

###### 4.2. Finding Richer Unconditional Priors

Can we improve the quality of conditional generation by incorporating better unconditional priors during sampling? Note that for fine-tuned diffusion models, we have access to a diffusion model with reliable unconditional priors: its base model. Therefore, we propose a simple yet effective fix, combining the unconditional noise prediction from the base model with the conditional noise prediction from the fine-tuned model. For this, the CFG noise in line 3 of Alg. 1 is modified as follows:

ϵ(θ,ψγ)(xt,c) = ϵψ(xt,∅) + γ(ϵθ(xt,c) − ϵψ(xt,∅)), (7)

where ϵψ and ϵθ denote the base model and fine-tuned model, respectively. Then DDIM sampling step becomes:

xt−1 = √α¯t−1g(xt,ϵ(θ,ψγ)(xt,c)) + 1 − α¯t−1ϵ(θ,ψγ)(xt,c)

Surprisingly, this simple modification results in significant improvements in the output quality of conditional generation. We demonstrate this through both qualitative and quantitative evaluations in Sec. 5.

A natural next question that arises is: should the unconditional noise come from the base model or from another unconditional model? We find that the base model does not necessarily need to be the true base model from which the new conditional model was fine-tuned from, but can instead be another diffusion model with good unconditional priors. In our experiments, we show that even though some models have been fine-tuned on SD1.x, using unconditional predictions of SD2.1 or PixArt-α results in further improvements as shown in Sec. 5.

Combining Diffusion Models. Although the base model and the fine-tuned model have different model weights,

Stable Diffusion v1.4 [52] Versatile Diffusion [64] Zero 1-to-3 [46] InstructPix2Pix [7]

[Figure 30]

- Figure 2. Unconditional samples from different diffusion models. Stable Diffusion [4], which often serves as the base model for finetuning conditional diffusion models, generates plausible images, whereas other fine-tuned diffusion models fail to sample realistic images.

their noise predictions can be combined as done in previous works [11, 20, 47]. Based on the connection between energy-based models (EBMs) and diffusion models [45], at each timestep, t, our method is equivalent to sampling from the time-annealed distribution pψ(xt)1−γpθ(xt|c)γ where pψ(xt) and pθ(xt) are the distributions modeled by the base and fine-tuned models, respectively.

Notably, pψ(xt) can be modeled by any pretrained diffusion model which may have different weights or even different architecture from the fine-tuned model so long as pψ(xt) is a better approximation of the true unconditional distribution than pθ(xt).

##### 5. Experiments

We validate our method on five conditional diffusion models, each trained for distinct conditional generation tasks: Zero-1-to-3 [46], Versatile Diffusion (VD) [64], DiT [50], DynamiCrafter [62], and InstructPix2Pix [7]. For experiments on models fine-tuned from Stable Diffusion, we present the results using unconditional noise predictions from both the true base model of the fine-tuned networks and other diffusion models, Stable Diffusion 2.1 (SD2.1) [52] and PixArt-α [10]. Notably, even when the fine-tuned model is a UNet, our method yields improvements when using PixArt-α, which is a DiT, in place of the base model. We refer readers to supplementary for details on the experimental setup for each application.

###### 5.1. Single-Condition CFG Formulation

In this section, we provide experimental results of our method when applied to diffusion models that use singlecondition CFG formulation. Models of this category samples using noise of the form provided in Eq. 6.

Zero-1-to-3 [46] Zero-1-to-3 is a conditional diffusion model for novel view synthesis, taking a reference image and relative camera poses as input. It is fine-tuned from

Method LPIPS↓ PSNR↑ SSIM↑

Zero-1-to-3 [46] 0.182 16.647 0.824 Ours w/ SD1.4 0.163 17.514 0.842 Ours w/ SD2.1 0.158 17.801 0.848

Ours w/ PixArt-α 0.169 17.069 0.825

- Table 1. Novel View Synthesis with Zero-1-to-3 [46]. Our method improves quality of novel view images (bold represents the best, and underline, the second best method).

Method FID ↓ FDDINOv2 ↓ CLIP-I ↑ DINOv2 ↑

VD [64] 8.38 167.65 0.93 0.91 Ours w/ SD1.4 6.68 156.77 0.94 0.92 Ours w/ SD2.1 7.80 151.48 0.94 0.92

Ours w/ PixArt-α 6.29 148.48 0.94 0.92

- Table 2. Image Variations with Versatile Diffusion [64]. Our sampling method achieves best performances across all metrics (bold represents the best, and underline, the second best method).

Stable Diffusion Image Variations (SD-IV) [36], which itself is originally fine-tuned from SD1.4. Due to the multiple fine-tuning stages, we opted to use SD1.4 as the base model. We evaluate the samples on the Google Scanned Objects (GSO) dataset [16] using LPIPS [69], PSNR, and SSIM. As shown in Tab. 1, our method, which incorporates unconditional noise predictions from base models, achieves significant improvements across all three metrics, with the best performances observed when using SD2.1 as the unconditional prior. Fig. 3 shows that using improved unconditional noise from the base model enhances lighting quality (row 1), reduces color saturation (row 2) and shape distortions (rows 3 and 4).

Versatile Diffusion [64] Versatile Diffusion (VD) is a multi-task diffusion model designed to handle text-toimage, image variations, and image-to-text tasks within a unified architecture. VD is progressively fine-tuned from SD1.4 in three stages to handle additional image conditions on top of text condition. Due to the cascaded finetuning scheme, VD displays the worst image-unconditional generation quality as shown in Fig. 2. We focus on using VD for image variations to generate semantically similar

Input Image

Ground Truth

Zero-1-to-3 (Baseline)

w/ SD1.4 (Ours)

w/ SD2.1 (Ours)

w/ PixArt-α (Ours)

[Figure 31]

- Figure 3. Novel View Synthesis with Zero-1-to-3 [46]. Outputs from Zero-1-to-3 often show inaccuracies in lighting or shape distortions during novel view synthesis. By incorporating unconditional noise predictions from Stable Diffusion [52] or PixArt-α [10], our method achieves clear improvements in output quality.

images from a reference image.

We report FID [27] and FDDINOv2 [60] on COCOCaptions [43] for image quality assessment and CLIP-I [26] and DINOv2 [49] image similarity metrics to evaluate condition alignment.

As shown in Tab. 2, using unconditional noise prediction from the base models yields better FID and FDDINOv2 while retaining similar CLIP-I and DINOv2 image similarity, showing a performance improvement while maintaining condition alignment.

As shown in Fig. 4, VD often generates images with highly saturated colors (rows 1 and 2) and distorted objects (row 3) while our method corrects both.

DiT [50] Although the experiments so far have been conducted using diffusion models with the UNet architecture, we show that our method holds for fine-tuned DiTs [50] as well. Since there are no publicly available fine-tuned DiT models, we fine-tune DiT-XL/2 on the standard downstream tasks SUN397 [61], Food101 [6], and Caltech101 [24]. The FID for the different fine-tuning tasks are shown in Tab. 3. Incorporating the unconditional noise from the base DiTXL/2 results in improved FID. We also observe larger ben-

efits when the fine-tuning dataset is large (Food101 and SUN397) corroborating our observation that the degradation in unconditional priors is amplified by the limited computational budget. While we observe no improvement for Caltech101, the dataset is over 10 times smaller than both SUN397 and Food101, thus the model is given sufficient time to fit the dataset despite the low CFG condition drop rate (10%). In practice, large-scale fine-tuning like in Zero1-to-3 and Versatile Diffusion often suffer from limited data and computation. In these practical settings, the degradation of the unconditional prior becomes highly detrimental as we have shown.

Method SUN397 Food101 Caltech101 Fine-tuned DiT-XL/2 17.12 18.31 24.05

Ours 14.51 17.67 24.15

Table 3. Class-conditional generation with DiT[50]. FID-5k evaluated on three fine-tuning tasks (SUN397 [61], Food101 [6], and Caltech101 [24]). Our method improves FID of the fine-tuned models (bold represents the best method).

###### VD (Baseline)

w/ SD1.4 (Ours)

w/ SD2.1 (Ours)

w/ PixArt-α (Ours)

Input Image

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 4. Image Variations with Versatile Diffusion [64]. Versatile Diffusion often suffers from style and detail degradation—excessive saturation (rows 1 and 3) or loss of key content (row 2). In contrast, our method, leveraging SD1.4, SD2.1, or PixArt-α as unconditional priors, achieves noticeable improvements in performance.
- 5.2. Dual-Condition CFG Formulation

DynamiCrafter [62] We apply our method to DynamiCrafter [62], a text-and-image-to-video diffusion model fine-tuned from the text-to-video diffusion model VideoCrafterT2V [9]. DynamiCrafter incorporates an image condition cI as c1 and a text condition cT as c2. For our method, we replace the DynamiCrafter unconditional noise with the VideoCrafterT2V unconditional noise. We report quantitative results using VBenchI2V [12, 35] which measures video quality and temporal consistency across multiple dimensions. For more details on the metrics, please refer to VBench [35]. Quantitative results are reported in Tab. 4. Our method outperforms the baseline in 7 out of 9 metrics, yielding more consistent videos with higher aesthetic quality. As shown in Fig. 5, our method is more temporally consistent (first video) and less distorted (second video). Video results are also included in the supplementary.

In this section, we provide experimental results on diffusion models which use the dual-condition CFG formulation. Diffusion models in this category are conditioned on two conditions and sample using the modified CFG noise

ϵθ(xt,c1,c2) = ϵθ(xt,∅,∅)

- + γ1(ϵθ(xt,c1,∅) − ϵθ(xt,∅,∅))
- + γ2(ϵθ(xt,c1,c2) − ϵθ(xt,c1,∅)). (8)

Notably, the dual-condition CFG formulation has two unconditional terms trained using CFG condition dropout: ϵθ(xt,∅,∅) and ϵθ(xt,c1,∅). We replace ϵθ(xt,∅,∅) with the base model unconditional noise prediction ϵψ(xt,∅). However, since the other unconditional term ϵθ(xt,c1,∅) is not replaced, we observe less improvements in this case than in the single-condition CFG formulation. This is to be expected as the quality degradation stems from training using a low dropout rate for the condition which is applied to both ϵθ(xt,∅,∅) and ϵθ(xt,c1,∅), only one of which is replaced by the better base model unconditional prior.

InstructPix2Pix [7] InstructPix2Pix (IP2P) tackles instruction-based image editing by fine-tuning SD1.5 to condition on both text (editing instruction, as c2) and image (as c1) to generate the edited image. We evaluate the performance on the EditEvalv2 benchmark [34]. To assess identity preservation, we compute the CLIP image

I2V Background DynamiCrafter [62] 90.80 96.73 95.21 96.67 59.59 57.06 64.10 92.77 94.56

Method Subject

Background Consistency

Temporal Flickering

Motion Smoothness

Dynamic Degree

Aesthetic Quality

Imaging Quality

I2V Subject

Consistency

Ours 91.49 97.03 95.34 96.86 57.32 57.51 63.15 93.49 94.72

Table 4. Video Generation with DynamiCrafter [62]. All metrics are scored out of 100, higher indicates better performance.

Input Generated Frames

[Figure 35]

[Figure 36]

Figure 5. Image-to-Video Generation with DynamiCrafter [62]. Our method is more temporally consistent (lighting on the biker) and less distorted (the hand and face in the second video).

similarity (C-I) [26] between the edited and input images. To evaluate the faithfulness of the edited images, we measure CLIP text alignment (C-T), CLIP Directional Similarity (C-D), Image Reward (IR) [63], and PickScore (PS) [40] based on the edited image prompt. The reported PS are compared against IP2P. As shown in Tab. 5, our method shows better alignment with the prompt while preserving the identity of the source image. We observe improvements in both IR and PS which have been observed to better align with human preference [40, 63] with slight underperformance in CLIP-T (for SD1.5 and SD2.1). Qualitative results are shown in Fig. 6. Our method generates faithful, high-fidelity edited images (rows 1 and 2) whereas IP2P creates distorted images (row 3).

Method C-I ↑ C-T ↑ C-D ↑ IR ↑ PS ↑ IP2P [7] 0.909 0.294 0.174 -0.510 −

Ours w/ SD1.5 0.911 0.291 0.186 -0.460 0.514 Ours w/ SD2.1 0.913 0.290 0.184 -0.464 0.518

Ours w/ PixArt-α 0.915 0.297 0.185 -0.363 0.532

- Table 5. Image Editing with InstructPix2Pix (IP2P) [7]. We normalize text and image similarity scores: C-I, C-T, and C-D. (bold represents the best, and underline represents the second best method.)

Input Image

###### IP2P (Baseline)

w/ PixArt-α (Ours) ‘‘Make the gown crystal ’’

w/ SD1.5 (Ours)

w/ SD2.1 (Ours)

[Figure 37]

‘‘turn the kitten into a sculpture ’’

[Figure 38]

‘‘change the woman to a storm-trooper ’’

[Figure 39]

Figure 6. Image Editing with InstructPix2Pix (IP2P) [7]. Applying our method improves alignment with the editing prompt while preserving the identity of the source image.

##### 6. Conclusion

We presented a novel, training-free approach to improving the generation quality of a CFG-based fine-tuned conditional diffusion model by replacing the low-quality unconditional noise with richer unconditional noise from a separate pretrained base diffusion model. We validated our approach across a range of diffusion models trained for distinct conditional generation, including image variation [64], image editing [7], novel view synthesis [46], and video generation [62]. Notably, we find that the separate pretrained diffusion model can have different weights and architecture from the original base model.

Limitations and Discussions. Although our method is training-free, it involves loading a second model into memory which increases memory cost. Furthermore, we can no longer parallelize computation as done in CFG, resulting in slight inference time overhead. However, the inference speed is only slightly affected as shown in the supplementary. Our method significantly improves diverse fine-tuned diffusion models, but is less effective for adapterbased fine-tuning methods such as ControlNet [67] and GLIGEN [42], which exhibit less degradation in unconditional priors. Identifying unconditional priors for these advanced fine-tuning techniques would be a valuable future direction.

##### Acknowledgements.

This work was supported by the IITP grants (RS-202400399817, RS-2025-25441313, RS-2025-25443318, RS-2025-02653113); and the Industrial Technology Innovation Program (RS-2025-02317326), all funded by the Korean government (MSIT and MOTIE), as well as by the DRB-KAIST SketchTheFuture Research Center.

##### References

- [1] Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance. In ECCV,

2024. 2

- [2] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 3
- [3] Benjamin Biggs, Arjun Seshadri, Yang Zou, Achin Jain, Aditya Golatkar, Yusheng Xie, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. Diffusion soup: Model merging for text-to-image diffusion models. In ECCV, 2024. 3
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 5
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 2
- [6] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part VI 13, pages 446–461. Springer, 2014. 6, 12, 16
- [7] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 1, 2, 4, 5, 7, 8, 13, 18
- [8] Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts. arXiv preprint arXiv:2407.06204, 2024. 3
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 7
- [10] Junsong Chen, YU Jincheng, GE Chongjian, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Repre-

sentations. 5, 6

- [11] Ziyang Chen, Daniel Geng, and Andrew Owens. Images that sound: Composing images and sounds on a single canvas. In NeurIPS, 2024. 3, 5
- [12] VBench Contributors. Vbench. GitHub repository, 2023. 7
- [13] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre D´efossez. Simple and controllable music generation. In NeurIPS, 2023. 2
- [14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 12
- [15] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2, 3
- [16] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 5, 12
- [17] Yilun Du and Leslie Kaelbling. Compositional generative modeling: A single model is not all you need. 2024. 3
- [18] Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In International conference on machine learning, pages 8489–8510. PMLR, 2023. 3
- [19] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602– 1614, 2011. 3
- [20] Rohit Gandikota, Joanna Materzynska, Jaden FiottoKaufman, and David Bau. Erasing concepts from diffusion models. In ICCV, 2023. 3, 5
- [21] Daniel Geng, Inbum Park, and Andrew Owens. Visual anagrams: Generating multi-view optical illusions with diffusion models. In CVPR, 2024. 3
- [22] Daniel Geng, Inbum Park, and Andrew Owens. Factorized diffusion: Perceptual illusions by noise decomposition. In ECCV, 2024. 3
- [23] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS,

2014. 2

- [24] Gregory Griffin, Alex Holub, Pietro Perona, et al. Caltech256 object category dataset. Technical report, Technical Report 7694, California Institute of Technology Pasadena,

2007. 6, 12, 16

- [25] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. In NeurIPS, 2023. 3
- [26] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 6, 8

- [27] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 6
- [28] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 1, 2, 3, 4
- [29] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [30] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 2022. 2
- [31] Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In ICCV, 2023. 2
- [32] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In arXiv preprint arXiv:2106.09685, 2021. 3
- [33] Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. 2023. 2
- [34] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, and Liangliang Cao. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525, 2024. 7, 13
- [35] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7, 13
- [36] Justin. Experiments with stable diffusion. https: / / github . com / justinpinkney / stable diffusion. 4, 5
- [37] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022. 12
- [38] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. arXiv preprint arXiv:2406.02507, 2024. 2, 4
- [39] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2
- [40] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In NeurIPS, 2023. 8
- [41] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724, 2024. 2
- [42] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee.

- Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 8
- [43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 6, 12
- [44] Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024. 2
- [45] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In ECCV, 2022. 5
- [46] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023. 1, 2, 4, 5, 6, 8, 12, 13, 14
- [47] Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, and Vishal M Patel. Unite and conquer: Plug & play multi-modal synthesis using diffusion models. In CVPR,

- 2023. 3, 5

[48] Nithin Gopalakrishnan Nair, Jeya Maria Jose Valanarasu, and Vishal M Patel. Maxfusion: Plug&play multi-modal generation in text-to-image diffusion models. In ECCV,

- 2024. 3

- [49] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6
- [50] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 5, 6, 12, 16

- [51] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024.
- [52] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4, 5, 6
- [53] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [54] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques

- for training gans. Advances in neural information processing systems, 29, 2016. 12
- [55] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 4
- [56] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. 2015. 2, 3
- [57] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 12
- [58] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. 3
- [59] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 2, 3
- [60] George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L Caterini, Eric Taylor, and Gabriel Loaiza-Ganem. Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 6
- [61] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010. 6, 12, 16
- [62] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025. 1, 5, 7, 8, 13, 17
- [63] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. In NeurIPS, 2023. 8
- [64] Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi. Versatile diffusion: Text, images and variations all in one diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7754–7765, 2023. 1, 2, 4, 5, 7, 8, 12, 13, 15
- [65] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities. arXiv preprint arXiv:2408.07666, 2024. 3
- [66] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. IJCV, 2024. 2
- [67] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 8

- [68] Qinsheng Zhang, Jiaming Song, Xun Huang, Yongxin Chen, and Ming-Yu Liu. Diffcollage: Parallel generation of large content with diffusion models. In CVPR, 2023. 3
- [69] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5, 13
- [70] Jincheng Zhong, XiangCheng Zhang, Jianmin Wang, and Mingsheng Long. Domain guidance: A simple transfer approach for a pre-trained diffusion model. In The Thirteenth International Conference on Learning Representations. 2

### Unconditional Priors Matter! Improving Conditional Generation of Fine-Tuned Diffusion Models

#### Supplementary Material

In this supplementary material, we first provide additional evidence for the fine-tuned models’ poor unconditional priors by quantitatively showing that the base model has better unconditional generation quality than the finetuned models in Sec. A. In Sec. B, we include more details about the experimental setups for Zero-1-to-3, Versatile Diffusion, DiT, DynamiCrafter, and InstructPix2Pix. We include more qualitative results in Sec. E and more ablation studies on the CFG scale in Sec. C. Finally, we provide details on the inference speed and memory cost of our method in Sec. D.

##### A. Quantitative Evaluation of Unconditional Samples

In the main paper, we argued that the poor unconditional priors from the fine-tuned models degrade the quality of the conditional generation. We qualitatively showed in Fig. 2 of the main paper that the fine-tuned models exhibit poor unconditional generation quality. In this section, we quantitatively show that the base models have better unconditional generation quality than the fine-tuned models. We unconditionally sample 5000 images from each of SD1.4, SD2.1, PixArt-α, Zero-1-to-3, Versatile Diffusion, and InstructPix2Pix, and evaluate the image quality using Inception Score (IS) [54]. The results are shown in Tab. 6. We observe that the fine-tuned models indeed have quantitatively worse unconditional generation than the base models. Thus, in the main paper, we proposed replacing the poor unconditional noise from the fine-tuned models with the good unconditional noise from the base model which improves the conditional generation quality.

Method IS ↑

SD1.4 14.085 SD2.1 12.640

PixArt-α 9.224 Versatile Diffusion 2.704

Zero-1-to-3 9.140 InstructPix2Pix 5.852

- Table 6. Image Model Unconditional Generation. We sample using the unconditional noise predictions from each model. The unconditional samples from SD1.4, SD2.1, and PixArt-α are higher quality than those of the fine-tuned models. (bold represents the best performance.)

##### B. Experiment Details

For all experiments, we use the DDIM [57] sampler. When applying our method to a base model with a different variance schedule, we use the fine-tuned model’s variance schedule as the reference and choose the base model timestep that yields the closest available variance to the finetuned model’s variance.

###### B.1. Zero-1-to-3 [46]

We evaulate our method using the Google Scanned Objects (GSO) dataset [16] which consists of over a thousand scanned objects. We render six views for each object at fixed radii and elevation with azimuths uniformly spaced 60◦ apart from each other. The first view is used as the reference image and Zero-1-to-3 is used to generate the remaining five images for evaluation. We use 50 steps of DDIM and a CFG scale of γ = 5.0.

###### B.2. Versatile Diffusion [64]

We use the COCO-Captions [43] 2014 validation set as the ground truth dataset. We randomly select 30,000 images from the validation set as input conditions to Versatile Diffusion and compute the FID and FDDINOv2 against the full validation set. We use 50 steps of DDIM and a CFG scale of γ = 2.0.

###### B.3. DiT [50]

We sample the images using γ = 1.5 and 50 steps of DDIM. The base model used is DiT-XL/2 trained on ImageNet 256×256 [14]. The fine-tuning is done on each of the datasets using 20,000 steps with batch size 64 and learning rate 0.0001. To account for the impact of random variation, we compute the FID three times and report the minimum, as done by Karras et al. [37]. We provide additional details on each of the dataset below.

SUN397 [61] SUN397 [61] is a dataset used for testing algorithms for scene recognition consisting of 108,754 images distributed among 397 categories.

Food101 [6] Food101 [6] consists of 101,000 images split among 101 food categories. Each category contains 250 test images and 750 training images.

Caltech101 [24] Caltech101 [24] contains images of objects belonging to 101 classes, containing 9,145 images in

total. Each class contains between 40 and 800 images with a typical edge length of between 200 and 300 pixels.

###### B.4. DynamiCrafter [62]

We sample 256 × 256 resolution videos using 50 steps of DDIM with a CFG scale of γT = 7.5 and γI = 1.5. Although the original paper uses a CFG scale of γT = γI =

- 7.5, we find that their choice of CFG scale results in mostly static images, as shown in their low dynamic degree of 40.57% in the VBench benchmark [35]. In contrast, the baseline DynamiCrafter with our choice of CFG scale has a higher dynamic degree of 59.59%.

###### B.5. InstructPix2Pix [7]

We evaluate the performance of InstructPix2Pix (IP2P) using the EditEvalv2 benchmark [34] which consists of 150 high quality images with edits from 7 categories.

IP2P uses a dual text-image CFG formulation: ϵθ(xt,cI,cT) = ϵθ(xt,∅,∅)

+ γI(ϵθ(xt,cI,∅) − ϵθ(xt,∅,∅))

+ γT(ϵθ(xt,cI,cT) − ϵθ(xt,cI,∅)) (9)

For our method, we replace the IP2P fully unconditional score ϵθ(xt,∅,∅) with the unconditional score from SD1.5 or SD2.1. We use 100 steps of DDIM with a CFG scale of γI = 1.5 and γT = 7.5.

##### C. Choice of CFG Scale

In this section, we provide an ablation study on the choice of CFG scale γ for Zero-1-to-3 [46] and Versatile Diffusion [64]. The results shown in Tab. 7 and 8 indicate that our method consistently yields improved results across CFG scales.

γ 3.0 4.0 5.0 6.0 7.0 8.0

Zero-1-to-3 [46] 0.192 0.170 0.182 0.179 0.178 0.178 Ours w/ SD1.4 0.170 0.165 0.163 0.163 0.161 0.161 Ours w/ SD2.1 0.165 0.161 0.158 0.159 0.158 0.160

Ours w/ PixArt-α 0.173 0.171 0.169 0.168 0.171 0.170

Table 7. Zero-1-to-3 [46] (CFG Scales). We report the LPIPS [69] (lower is better) of applying our method to Zero-1-to-3 using various CFG scales (bold represents the best, and underline represents the second best method).

γ 2.5 3.0 4.0 5.0 7.5

Versatile Diffusion [64] 37.96 40.19 42.07 42.33 44.80 Ours w/ SD1.4 35.67 35.24 35.45 35.60 36.07 Ours w/ SD2.1 38.29 37.44 37.83 38.44 37.71

Ours w/ PixArt-α 37.55 39.03 39.62 40.24 40.89

- Table 8. Versatile Diffusion [64] (CFG Scales). We report the FID-5k (lower is better) of applying our method to Versatile Diffusion using various CFG scales (bold represents the best, and underline represents the second best method).

- D. Memory and Inference Speed

As shown in Tab. 9, the inference speed is only slightly affected by our method.

Method

Memory (GB) Speed (seconds/sample)

Baseline Ours Baseline Ours

Zero-1-to-3 [46] 4.93 10.06 2.92 3.59 VD 5.68 10.80 7.20 8.17 DiT 3.11 5.65 4.24 4.96 IP2P 5.13 10.14 19.45 21.43

DynamiCrafter 19.17 29.03 125.15 142.84

Table 9. Memory and Inference Speed on an RTX3090 using float32 precision.

- E. Additional Qualitative Results

We provide additional qualitative results for Zero-1-to-3 (Fig. 7), Versatile Diffusion (Fig. 8), DiT (Fig. 9), DynamiCrafter (Fig. 10), and InstructPix2Pix (Fig. 11).

Input Image

Ground Truth

Zero-1-to-3 (Baseline)

w/ SD1.4 (Ours)

w/ SD2.1 (Ours)

w/ PixArt-α (Ours)

[Figure 40]

- Figure 7. Novel View Synthesis with Zero-1-to-3 [46]. Zero-1-to-3 tends to produce views that have inaccurate lighting, coloring, or shape. Combining Zero-1-to-3 with the unconditional noise from SD1.4, SD2.1, or PixArt-α corrects these inaccuracies.

###### VD (Baseline)

w/ SD1.4 (Ours)

w/ SD2.1 (Ours)

w/ PixArt-α (Ours)

Input Image

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

- Figure 8. Image Variations with Versatile Diffusion [64]. Images generated from Versatile Diffusion tend to be oversaturated and distorted. Combining Versatile Diffusion with the unconditional noise predictions from SD1.4, SD2.1, or PixArt-α corrects these artifacts. 15

[Figure 47]

###### Figure 9. Class-conditional generation with DiT [50]. Class-conditional generation using DiT fine-tuned on SUN397 [61], Food101 [6], and Caltech101 [24].

###### Input Generated Frames

[Figure 48]

[Figure 49]

[Figure 50]

- Figure 10. Image-to-Video Generation with DynamiCrafter [62]. Our method is more temporally consistent (number of horses in the first video and train color in the second video) and less distorted (hand and face in the last video).

Input Image

###### IP2P (Baseline)

w/ PixArt-α (Ours) ‘‘Change the horses to unicorns ’’

w/ SD1.5 (Ours)

w/ SD2.1 (Ours)

[Figure 51]

‘‘Replace the spaceship with an eagle ’’

[Figure 52]

‘‘Change the weather to sunny ’’

[Figure 53]

‘‘Change the beach to grass in the painting ’’

[Figure 54]

‘‘Change to a kids crayon drawing ’’

[Figure 55]

- Figure 11. Image Editing with InstructPix2Pix (IP2P) [7]. InstructPix2Pix tends to produce distorted edits. Replacing the IP2P fully unconditional noise with the unconditional noise from SD1.5, SD2.1, or PixArt-α corrects these distortions and improves image quality.

