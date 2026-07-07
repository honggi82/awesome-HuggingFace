# arXiv:2409.18124v5[cs.CV]18Jan2025

LOTUS: DIFFUSION-BASED VISUAL FOUNDATION MODEL FOR HIGH-QUALITY DENSE PREDICTION

Jing He1✱ Haodong Li1✱ Wei Yin2 Yixun Liang1 Leheng Li1 Kaiqiang Zhou3 Hongbo Zhang3 Bingbing Liu3 Yingcong Chen1,4 1HKUST(GZ) 2University of Adelaide 3Noah’s Ark Lab 4HKUST

{jhe812, hli736}@connect.hkust-gz.edu.cn; yingcongchen@ust.hk

[Figure 1]

[Figure 2]

[Figure 3]

DepthAnything V2

Lotus (Ours)

[Figure 4]

[Figure 5]

[Figure 6]

DepthAnything V2

Lotus (Ours)

[Figure 7]

[Figure 8]

[Figure 9]

DSINE

Lotus (Ours)

[Figure 10]

[Figure 11]

[Figure 12]

Lotus (Ours)

DSINE

Avg. Rank

Avg. Rank Omnidata

[Figure 13]

[Figure 14]

OASIS Omnidata EESNU

DPT

HDN

GenPercept

GenPercept Omnidata V2

Diffusion-E2E-FT

DepthAnything V2

DSINE

Diffusion-E2E-FT

###### Lotus-D

Lotus-D

DepthAnything

Avg. Rank

Avg. Rank

Marigold

GeoWizard

Marigold(LCM)

GeoWizard StableNormal

Marigold Lotus-G

Lotus-G

Training Data

Training Data

Figure 1: We present Lotus, a diffusion-based visual foundation model for dense geometry prediction. With minimal training data, Lotus achieves promising performance in zero-shot depth and normal estimation. “Avg. Rank” indicates the average ranking across all metrics, where lower values are better. Bar length represents the amount of training data used.

✱Both authors contributed equally (order randomized). Corresponding author.

ABSTRACT

Leveraging the visual priors of pre-trained text-to-image diffusion models offers a promising solution to enhance zero-shot generalization in dense prediction tasks. However, existing methods often uncritically use the original diffusion formulation, which may not be optimal due to the fundamental differences between dense prediction and image generation. In this paper, we provide a systemic analysis of the diffusion formulation for the dense prediction, focusing on both quality and efficiency. And we find that the original parameterization type for image generation, which learns to predict noise, is harmful for dense prediction; the multistep noising/denoising diffusion process is also unnecessary and challenging to optimize. Based on these insights, we introduce Lotus, a diffusion-based visual foundation model with a simple yet effective adaptation protocol for dense prediction. Specifically, Lotus is trained to directly predict annotations instead of noise, thereby avoiding harmful variance. We also reformulate the diffusion process into a single-step procedure, simplifying optimization and significantly boosting inference speed. Additionally, we introduce a novel tuning strategy called detail preserver, which achieves more accurate and fine-grained predictions. Without scaling up the training data or model capacity, Lotus achieves promising performance in zero-shot depth and normal estimation across various datasets. It also enhances efficiency, being significantly faster than most existing diffusion-based methods. Lotus’ superior quality and efficiency also enable a wide range of practical applications, such as joint estimation, single/multi-view 3D reconstruction, etc. Project page: lotus3d.github.io.

- 1 INTRODUCTION

Dense prediction is a fundamental task in computer vision, benefiting a wide range of applications, such as 3D/4D reconstruction (Huang et al., 2024; Long et al., 2024; Wang et al., 2024; Lei et al., 2024), tracking (Xiao et al., 2024; Song et al., 2024), and autonomous driving (Yurtsever et al., 2020; Hu et al., 2023). Estimating pixel-level geometric attributes from a single image requires comprehensive scene understanding. Although deep learning has advanced dense prediction, progress is limited by the quality, diversity, and scale of training data, leading to poor zero-shot generalization. Instead of merely scaling data and model size, recent works (Lee et al., 2024; Ke et al., 2024; Fu et al., 2024; Xu et al., 2024) leverage diffusion priors for zero-shot dense prediction. These studies demonstrate that text-to-image diffusion models like Stable Diffusion (Rombach et al., 2022), pretrained on billions of images, possess powerful and comprehensive visual priors to elevate dense prediction performance. However, most of these methods directly inherit the pre-trained diffusion models for dense prediction tasks, without exploring more suitable diffusion formulations. This oversight often leads to challenging issues. For example, Marigold (Ke et al., 2024) directly finetunes Stable Diffusion for image-conditioned depth generation. While it significantly improves depth estimation, its performance is still constrained by overlooking the fundamental differences between dense prediction and image generation. Especially, its efficiency is also severely limited by standard iterative denoising processes and ensemble inferences.

Motivated by these concerns, we systematically analyze the diffusion formulation, trying to find a better formulation to fit the pre-trained diffusion model into dense prediction. Our analysis yields several important findings: ① The widely used parameterization, i.e., noise prediction, for diffusionbased image generation is ill-suited for dense prediction. It results in large prediction errors due to harmful prediction variance at initial denoising steps, which are subsequently propagated and magnified throughout the entire denoising process (Sec. 4.1). ② Multi-step diffusion formulation is computation-intensive and is prone to sub-optimal with limited data and resources. These factors significantly hinder the adaptation of diffusion priors to dense prediction tasks, leading to decreased accuracy and efficiency (Sec. 4.2). ③ Though remarkable performance achieved, we observed that the model usually outputs vague predictions in highly-detailed areas (Fig. 8). This vagueness is attributed to catastrophic forgetting: the pre-trained diffusion models gradually lose their ability to generate detailed regions during fine-tuning (Sec. 4.3).

single-step

𝑥 -prediction

𝐳𝐱

Image x

𝒛 𝐱

𝑡 = T Training Objective:

[Figure 15]

[Figure 16]

❄

[Figure 17]

[Figure 18]

concat.

𝐳𝐱 − 𝑓 𝐳𝒕𝐲,𝐳𝐱,𝑡,𝑠

🔥

[Figure 19]

denoiser U-Net 𝑓

[Figure 20]

(image reconstruction)

ℰ

+

(

[Figure 21]

[Figure 22]

[Figure 23]

(

[Figure 24]

𝐳𝐲 − 𝑓 𝐳𝒕𝐲,𝐳𝐱,𝑡,𝑠

add noise

(predict annotation)

𝐳𝒕𝐲

𝐳𝐲

𝒛 𝐲

switcher 𝑠

Annotation y

𝑡 = T

detail preserver

single-step

Figure 2: Adaptation protocol of Lotus. After the pre-trained VAE encoder E encodes the image x and annotation y to the latent space: ① the denoiser U-Net model fθ is fine-tuned using x0-prediction; ② we employ single-step diffusion formulation at time-step t = T for better convergence; ③ we propose a novel detail preserver, to switch the model either to reconstruct the image or generate the dense prediction via a switcher s, ensuring a more fine-grained prediction. The noise zyT in bracket is used for our generative Lotus-G and is omitted for the discriminative Lotus-D.

Following our analysis, we propose Lotus, a diffusion-based visual foundation model for dense prediction, featuring a simple yet effective fine-tuning protocol (see Fig. 2). First, Lotus is trained to directly predict annotations, thereby avoiding the harmful variance associated with standard noise prediction. Next, we introduce a one-step formulation, i.e., one step between pure noise and clean output, to facilitate model convergence and achieve better optimization performance with limited high-quality data. It also considerably boosts both training and inference efficiency. Moreover, we implement a novel detail preserver through a task switcher, allowing the model either to generate annotations or reconstruct the input images. It can better preserve the fine-grained details in input image during dense annotation generation, achieving higher performance without compromising efficiency, requiring additional parameters, or being affected by surface textures.

[Figure 25]

Figure 3: Inference time comparison in depth estimation between Lotus and SoTA methods. Lotus is hundreds of times faster than Marigold and slightly faster than DepthAnything V2 at high resolutions. DepthAnything V2’s inference time at 2048 × 2048 is not plotted because it requires > 80GB graphic memory.

- 1. On Single A800
- 2. We keep the original shape of input image during inference.
- 3. DA在2048下爆显存啦！

To validate Lotus, we conduct extensive experiments on two primary geometric dense prediction tasks: zero-shot monocular depth and normal estimation. The results demonstrate that Lotus achieves promising, and even superior, performance on these tasks across a wide range of evaluation datasets. Compared to traditional discriminative methods, Lotus delivers remarkable results with only 59K training samples by effectively leveraging the powerful diffusion priors. Among generative approaches, Lotus also outperforms previous methods in both accuracy and efficiency, being significantly faster than methods like Marigold (Ke et al., 2024) (Fig. 3). Beyond these improvements, Lotus seamlessly supports various applications, such as joint estimation, single/multi-view 3D reconstruction, etc.

In conclusion, our key contributions are as follows:

- • We systematically analyze the diffusion formulation and find their parameterization type, designed for image generation, is unsuitable for dense prediction and the computationintensive multi-step diffusion process is also unnecessary and challenging to optimize.
- • We propose a novel detail preserver that ensures more accurate dense predictions especially in detail-rich areas, without compromising efficiency, introducing additional network parameters, or being affected by surface textures.

- • Based on our insights, we introduce Lotus, a diffusion-based visual foundation model for dense prediction with simple yet effective fine-tuning protocol. Lotus achieves promising performance on both zero-shot monocular depth and surface normal estimation. It also enables a wide range of applications.

- 2 RELATED WORKS

- 2.1 TEXT-TO-IMAGE GENERATIVE MODELS

In the field of text-to-image generation, the evolution of methodologies has transitioned from generative adversarial networks (GANs) (Goodfellow et al., 2014; Zhang et al., 2017; 2018; 2021; He et al., 2022; Karras et al., 2019; 2020; 2021; Zhang et al., 2017; 2018; Xu et al., 2018; Zhang et al.,

- 2021) to advanced diffusion models (Ho et al., 2020; Ramesh et al., 2022; Saharia et al., 2022; Ramesh et al., 2021; Nichol et al., 2021; Chen et al., 2023; Rombach et al., 2022; Ramesh et al., 2021; He et al., 2024). A series of diffusion-based methods such as GLIDE (Nichol et al., 2021), DALL·E2 (Ramesh et al., 2022), and Imagen (Saharia et al., 2022) have been introduced, offering enhanced image quality and textual coherence. The Stable Diffusion (SD) (Rombach et al., 2022), trained on large-scale LAION-5B dataset (Schuhmann et al., 2022), further enhances the generative quality, becoming the community standard. In our paper, we aim to leverage the comprehensive and encyclopedic visual priors of SD to facilitate zero-shot generalization for dense prediction tasks.

- 2.2 GENERATIVE MODELS FOR DENSE PERCEPTION

Currently, a notable trend involves adopting pre-trained generative models, particularly diffusion models, into dense prediction tasks. Marigold (Ke et al., 2024) and GeoWizard (Fu et al., 2024) directly apply the standard diffusion formulation and the pre-trained parameters, without addressing the inherent differences between image generation and dense prediction, leading to constrained performance. Their efficiency is also severely limited by standard iterative denoising processes and ensemble inferences. In this paper, we propose a novel diffusion formulation tailored to the of dense prediction. Aiming to fully leveraging the pre-trained diffusion’s powerful visual priors, Lotus enables more accurate and efficient predictions, finally achieving promising performance.

More recent works, GenPercept (Xu et al., 2024) and StableNormal (Ye et al., 2024), also adopted single-step diffusion. However, GenPercept (Xu et al., 2024) first removes noise input for deterministic characteristic based on DMP (Lee et al., 2024), and then adopts one-step strategy to avoid surface texture interference. It lacks systematic analysis of the diffusion formulation, only treats the U-Net as a deterministic backbone and still falls short in performance. In contrast, Lotus systematically analyzes the standard stochastic diffusion formulation for dense prediction and proposes innovations such as the detail preserver to improve accuracy especially in detailed area, finally delivering much better results (Tab. 1). Additionally, Lotus is a stochastic model. In contrast to GenPercept’s deterministic nature, Lotus enables uncertainty predictions. StableNormal (Ye et al., 2024) predicts normal maps through a two-stage process. While the first stage produces coarse normal maps with single-step diffusion, the second stage performs refinement still with iterative diffusion which is computation-intensive. In comparison, Lotus not only achieves fine-grained predictions thanks to our novel detail preserver without extra stages or parameters, but also delivers much superior results (Tab. 2) thanks to our designed diffusion formulation that better fits the pre-trained diffusion for dense prediction. Recently, a concurrent work, Diffusion-E2E-FT (Garcia et al., 2024), has also achieved promising results in a single step. Its main contribution lies in addressing the issue where Marigold (Ke et al., 2024) and similar models (Fu et al., 2024) use inconsistent pairings of time-step and noise, resulting in poor predictions. By setting the “time-step spacing” to “trailing” mode in schedulers, it prevents “GT” signal leakage during inference, improving accuracy. While the performance of Lotus-D and Diffusion-E2E-FT is similar, Lotus is based on a systematic analysis of stochastic diffusion for dense prediction, with innovations like the detail preserver to enhance accuracy, particularly in detailed areas. Additionally, unlike the deterministic Diffusion-E2E-FT, Lotus (Lotus-G) is a stochastic model that enables uncertainty predictions.

- 2.3 MONOCULAR DEPTH AND NORMAL PREDICTION

Monocular depth and normal prediction are two crucial dense prediction tasks. Solving them typically demands comprehensive scene understanding capability. Starting from (Eigen et al., 2014), early CNN-based methods for depth prediction, such as (Fu et al., 2018; Lee et al., 2019; Yuan et al., 2022), focus only on specific domains. Subsequently, in pursuit of a generalizable depth estimator, many methods expand model capacity and train on larger and more diverse datasets, such

- as DiverseDepth (Yin et al., 2021a) and MiDaS (Ranftl et al., 2020). DPT (Ranftl et al., 2021) and Omnidata (Eftekhar et al., 2021) are further proposed based on vision transformer (Ranftl et al.,

- 2021), significantly enhancing performance. LeRes (Yin et al., 2021b) and HDN (Zhang et al.,
- 2022) further introduce novel training strategies and multi-scale depth normalization to improve predictions in detailed areas. More recently, the DepthAnything series (Yang et al., 2024a;b) and Metric3D series (Yin et al., 2023; Hu et al., 2024) collect and leverage millions of training data to develop more powerful estimators. Normal prediction follows the same trend. Starting with the early CNN-based methods like OASIS (Chen et al., 2020), EESNU (Bae & Davison, 2021) and Omnidata series (Eftekhar et al., 2021; Kar et al., 2022) expand the model capacity and scale up the training data. Recently, DSINE (Bae & Davison, 2024) achieves SoTA performance by rethinking inductive biases for surface normal estimation. In our paper, we focus on leveraging pre-trained diffusion priors to enhance zero-shot dense predictions, rather than expanding model capacity or relying on large training data, which avoids the need for intensive resources and computation.

- 3 PRELIMINARIES

Diffusion Formulation for Dense Prediction. Following (Ke et al., 2024) and (Fu et al., 2024), we also formulate dense prediction as an image-conditioned annotation generation task based on Stable Diffusion (Rombach et al., 2022), which performs the diffusion process in low-dimensional latent space for computational efficiency. First, the auto-encoder, which consists an encoder E(·) and a decoder D(·), is trained to map between RGB space and latent space, i.e., E(x) = zx, D(zx) ≈ x. The auto-encoder also maps between dense annotations and latent space effectively, i.e., E(y) = zy, D(zy) ≈ y (Ke et al., 2024; Fu et al., 2024; Xu et al., 2024; Ye et al., 2024). Following (Ho et al., 2020), Stable Diffusion establishes a pair of forward nosing and reversal denoising processes in latent space. In forward process, Gaussian noise is gradually added at levels t ∈ [1,T] into sample zy to obtain the noisy sample zyt :

### zyt = √αtzy + √1 − αtϵ, (1)

where ϵ ∼ N(0,I), αt := ts=1(1 − βs), and {β1,β2,...,βT} is the noise schedule with T steps. At time-step T, the sample zy is degraded to pure Gaussian noise. In the reversal process, a neural

network fθ, usually a U-Net model (Ronneberger et al., 2015), is trained to iteratively remove noise from zyt to predict the clean sample zy. The network is trained by sampling a random t ∈ [1,T] and minimizing the loss function Lt.

Parameterization Types. To enable gradient computation for network training, there are two basic parameterizations of the loss function Lt. ① ϵ-prediction (Ho et al., 2020): the model fθ learns to predict the added noise ϵ; ② x0-prediction (Ho et al., 2020): the model fθ learns to directly predict the clean sample zy. The loss functions for these parameterizations are formulated as:

### ϵ-prediction: Lϵt = ||ϵ − fθϵ(zyt ,zx,t)||2, x0-prediction: Lzt = ||zy − fθz(zyt ,zx,t)||2.

(2)

where fθ∗ is the denoiser model to be learnt, ∗ ∈ {ϵ,z}. ϵ-prediction is commonly chosen as the standard for parameterizing the denoising model, as it empirically achieves high-quality image gen-

eration with fine details and realism.

Denoising Process. DDIM (Song et al., 2020) is a key technique for multi-step diffusion models to achieve fast sampling, which implements an implicit probabilistic model that can significantly reduce the number of denoising steps while maintaining output quality. Formally, the denoising process from zyτ to zyτ−1 is:

### zyτ−1 = ατ−1ˆzyτ + direction(zyτ ) + στϵτ, (3)

multi-step

𝐳𝐱

Image x

[Figure 26]

##### ❄ 𝑡 ∈ [1,T]

[Figure 27]

[Figure 28]

concat .

𝜖-prediction

[Figure 29]

[Figure 30]

ℰ

🔥

Training Objective:

denoiser U-Net 𝑓 𝜖 − 𝑓 𝐳𝒕𝐲,𝐳𝐱,𝑡

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

𝜖

add noise

(predict noise)

𝐳𝒕𝐲

𝐳𝐲

Annotation y

𝑡 ∈ [1,T]

multi-step

- Figure 4: Adaptation protocol of Direct Adaptation. Starting with a pre-trained Stable Diffusion

model, image x and annotation y are encoded using the pre-trained VAE. The noisy annotation zyt is obtained by adding noise at level t ∈ [1,T]. The U-Net input layer is coupled to accommodate the

concatenated inputs and then fine-tuned using the standard diffusion objective, ϵ-prediction, under the original multi-step formulation.

where ˆzyτ is the predicted clean sample at the denoising step τ, direction(zyτ ) represents the direction pointing to zyτ and στ can be set to 0 if deterministic denoising is needed. And τ ∈ {τ1,τ2,...,τS}, an increasing sub-sequence of the time-step set [1,T], is used for fast sampling. During inference, DDIM iteratively denoises the sample from τS to τ1 to obtain the clean one.

- 4 METHODOLOGY

We start our analysis by directly adapting the original diffusion formulation with minimal modifications as illustrated in Fig. 4. We call this starting point as “Direct Adaptation”1. Direct Adaptation is optimized using the standard diffusion objective as formulated in Eq. 2 (first row) and inferred by standard multi-step DDIM sampler. As shown in Tab. 3, Direct Adaptation fails to achieve satisfactory performance. In following sections, we will systematically analyze the key factors that affect adaptation performance step by step: parameterization types (Sec. 4.1); number of time-steps (Sec. 4.2); and the novel detail preserver (Sec. 4.3).

- 4.1 PARAMETERIZATION TYPES

The type of parameterization is crucial, it not only determines the loss function discussed in Sec. 3, but also influences the inference process (Eq. 3). During inference, the predicted clean sample ˆzyτ , a key component in Eq. 3, is calculated according to different parameterizations 2.

√1 − ατfθϵ(zyτ ,zx,τ)), x0-prediction: ˆzyτ = fθz(zyτ ,zx,τ).

1 √ατ

ϵ-prediction: ˆzyτ =

(zyτ −

(4)

In the community, ϵ-prediction is chosen as the standard for image generation. However, it is not effective for dense prediction task. In the following, we will discuss the impact of different parameterization types in denoising inference process for dense prediction task.

Insights from the literature (Benny & Wolf, 2022; Salimans & Ho, 2022) reveal that ϵ-prediction introduces larger pixel variance compared to x0-prediction, especially at the initial denoising steps (large τ). This variance mainly originates from the noise input. Specifically, for ϵ-prediction in Eq. 4, at initial denoising step, τ → T, the value √1α

→ +∞. Thus, the prediction variance

τ

from fθϵ(zyτ ,zx,τ) will be amplified significantly, resulting in large variance of predicted ˆzyτ . In contrast, there is no coefficient for x0-prediction to re-scale the model output, achieving more stable predictions of ˆzyτ at initial denoising steps. Subsequently, the predicted ˆzyτ is used in Eq. 3, where its coefficient √ατ−1 are same across the two parameterizations, and other terms are of the same order

1Details of Direct Adaptation will be provided in Sec. B of the supplementary materials. 2The latest parameterization, v-prediction, combines ϵ-prediction and x0-prediction, producing results that

are intermediate between the two. Please see Sec. D of the supplementary materials for more details.

𝜀-prediction, seed 1 𝜀-prediction, seed 2

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

𝑥 -prediction, seed 1 𝑥 -prediction, seed 2

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Input Image

𝜏 = 1000 𝜏 = 600 𝜏 = 200 𝜏 = 1

𝜏 = 1000 𝜏 = 600 𝜏 = 200 𝜏 = 1

- Figure 5: Comparisons among different parameterizations using various seeds. All models are trained on Hypersim (Roberts et al., 2021) and tested on the input image for depth estimation. The standard DDIM sampler is used with 50 denoising steps. Four steps are selected for clear illustration. From left (larger τ) to right (smaller τ) is the iterative denoising process.

of magnitude. Therefore, the ˆzyτ predicted by ϵ-prediction, which has larger variance, exerts a more significant influence on denoising process. Since the process is iterative, this influence is continually preserved and maybe amplified.

We take the depth estimation as an example. During the inference process, we compute the predicted depth map ˆzyτ at each denoising step τ. As illustrated in Fig. 5, the depth maps predicted by ϵ-prediction significantly vary under different seeds while those predicted by x0-prediction are more consistent. Although the large variance enhances diversity for image generation, it lead to unstable predictions in dense prediction tasks, potentially resulting in significant errors. For example in Fig. 5, the “dark gray cabinet” (highlighted in red circles) maybe wrongly considered as an “opened door” with significantly larger depth. While the predicted depth map looks more and more plausible, the error gradually propagates to the final prediction (τ = 1) along the denoising process, indicating the persistent influence of the large variance. We further quantitatively measure the predicted depth maps by the absolute mean relative error (AbsRel) on NYUv2 dataset (Silberman et al., 2012). As shown in Fig. 6, ϵ-prediction exhibits higher error with much larger variance compared to x0-prediction at the initial denoising steps (τ → T), and the prediction error propagates with a higher slope. In contrast, x0-prediction, directly predicting ˆzyτ without any coefficients to amplify the prediction variance, yields more stable and correct dense predictions than ϵ-prediction. In conclusion, to mitigate the errors from large variance that adversely affect the performance of dense prediction, we replace the standard ϵ-prediction with the more tailored x0-prediction.

NYUv2 AbsRel vs. Denoising Step

Pred. Type=x0

- 8

- 9

- 10

- 11

- 12

Pred. Type=

NYUv2AbsRel(%)

1000 800 600 400 200 0 Denoising Step ( )

Figure 6: Quantitative evaluation of the predicted depth maps ˆzyτ along the denoising process. The experimental settings are same as Fig. 5. Six steps are selected for illustration. The banded regions around each line indicate the variance, wider areas representing larger variance.

- 4.2 NUMBER OF TIME-STEPS

Although x0-prediction can improve the prediction quality, the multi-step diffusion formulation still leads to the propagation of predicted errors during the denoising process (Fig. 5, 6). Furthermore, utilizing multiple time-steps enhances the model’s capacity, typically requiring large-scale training data to optimize and is beneficial for complex tasks such as image generation. However, for simpler tasks like dense prediction, where large-scale, high-quality training data is also scarce, employing multiple time-steps can make the model difficult to optimize. Additionally, training/inferring a multi-step diffusion model is slow and computation-intensive, hindering its practical application.

Therefore, to address these challenges, we propose fine-tuning the pre-trained diffusion model with fewer training time steps. Specifically, the original set of training time-steps is defined as [1,T] = {1,2,3,...,T}, where T denotes the total number of original training time-steps. We fine-tune the

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

Input Image Reconstruction

Input Image Reconstruction

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

|[Figure 61]|[Figure 62]|
|---|---|

[Figure 63]

[Figure 64]

| | | |
|---|---|---|
|w/o Preserver| | |

| | | |
|---|---|---|
|w/ Preserver| | |

|w/ Preserver|
|---|

|w/o Preserver|
|---|

- Figure 8: Depth maps w/ and w/o the detail preserver and reconstruction outputs. Fine-tuning the diffusion model for dense prediction tasks can potentially degrade its ability to generate highly detailed images, resulting in blurred predictions in regions with rich detail. To preserve these finegrained details, we introduce a detail preserver that incorporates an additional reconstruction task, enhancing the model’s capacity to produce more accurate dense annotations.

pre-trained diffusion model using a sub-sequence derived from this set. We define the length of this sub-sequence as T′, where T′ ⩽ T and T is divisible by T′. This sub-sequence is obtained by evenly sampling the original set at intervals, defined as:

### {ti = i · k | i = 1,2,...,T′}, (5)

where k = T/T′ is the sampling interval. During inference, the DDIM denoises the sample from noise to annotation using the same sub-sequence if T′ ⩽ 50, otherwise we use 50 denoising steps.

As illustrated in Fig. 7, we conduct experiments by varying the number of time-steps T′ under x0-prediction. The results clearly show that the performance gradually improves as the number of time-steps is reduced, no matter the training data scales, culminating in the best result when reduced to only a single step. We further consider more strict scenarios with more limited training

data to assess its impact on model optimization. As depicted in Fig. 7, these experiments reveal that the multi-step formulation is more sensitive to increases in training data scales compared with single-step. Notably, the single-step formulation consistently yields lower prediction errors and demonstrates greater stability. Although it is conceivable that multi-step and single-step formulations might achieve comparable performance with unlimited high-quality data, it’s expensive and sometimes impractical in dense prediction.

NYUv2 AbsRel vs. Training Data

- 6

- 7

- 8

- 9

- 10

- 11

- T'=1

- T'=2

T'=5

T'=10

NYUv2AbsRel(%)

T'=100

T'=1000

5K 10K 19K 39K Training Data

Decreasing the number of denoising steps can reduce the optimization space of the diffusion model, leading to more effective and efficient adaption, as suggested by the above phenomenon. Therefore, for better adaptation performance under limited resource, we reduce the number of training time-steps of diffusion formulation to only one, and fixing the only timestep t to T. Additionally, the single-step formulation is much more computationally efficient. It also naturally prevents the harmful error propagation as discussed in Sec. 4.1, further enhancing the diffusion’s adaptation performance in dense prediction.

Figure 7: Comparisons among various training time-steps and data scales evaluated on NYUv2 in depth estimation. All models are fine-tuned on Hypersim using x0-prediction. During inference, if T′ > 50, the DDIM sampler is used with 50 denoising steps; otherwise, the number of denoising steps is equal to T′. The results demonstrate improved performance with decreased training time-steps. The single-step diffusion formulation (T′ = 1) exhibits best performance across different data volumes.

Input Image Seed 0 Seed 1 Seed 2 Seed 3 Uncertainty Map

Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

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

Input Image Seed 0 Seed 1 Seed 2 Seed 3 Uncertainty Map

- Figure 9: Depth maps of multiple inferences and uncertainty maps. Areas like the sky, object edges, and intricate details (e.g., cat whiskers) typically exhibit high uncertainty.

- 4.3 DETAIL PRESERVER

Despite the effectiveness of the above designs, the model still struggles with processing detailed areas (Fig. 8, w/o Preserver). The original diffusion model excels at generating detailed images. However, when adapted to predict dense annotations, it can lose such detailed generation ability, due to unexpected catastrophic forgetting (Zhai et al., 2023; Du et al., 2024). This leads to challenges in predicting dense annotations in intricate regions.

To preserve the rich details of the input images, we introduce a novel regularization strategy called Detail Preserver. Inspired by previous works (Long et al., 2024; Fu et al., 2024), we utilize a task switcher s ∈ {sx,sy}, enabling the denoiser model fθ to either generate annotation or reconstruct the input image. When activated by sy, the model focuses on predicting annotation. Conversely, when sx is selected, it reconstructs the input image. The switcher s is a one-dimensional vector encoded by the positional encoder and then added with the time embeddings of diffusion model, ensuring seamless domain switching without mutual interference. This dual capability enables the diffusion model to make detailed predictions and thus leading to better performance. Overall, the loss function Lt is:

Lt = ||zx − fθ(zyt ,zx,t,sx)||2 + ||zy − fθ(zyt ,zx,t,sy)||2, (6) where t = T and thus zyt is a pure Gaussian noise.

- 4.4 STOCHASTIC NATURE OF DIFFUSION MODEL

One major characteristic of generative models is their stochastic nature, which, in image generation, enables the production of diverse outputs. In perception tasks like dense prediction, this stochasticity has the potential to allow the model generating predictions with uncertainty maps. Specifically, for any input image, we can conduct multiple inferences using different initialization noises and aggregate these predictions to calculate its uncertainty map. Thanks to our systematic analysis and tailored fine-tuning protocol, our method effectively

𝐳𝑻𝐲

reduces excessive flickering (large variance), only allowing for more accurate uncertainty calculations in naturally uncertain areas, such as the sky, object edges, and fine details (e.g. cat whiskers), as shown in Fig. 9.

(

(

| | | | | | |
|---|---|---|---|---|---|
| | | |[Figure 89]<br><br>[Figure 90]| | |
| | | | | | |
| | | | | | |

Image x

concat.

denoiser U-Net 𝑓

[Figure 91]

[Figure 92]

ℰ

𝐳𝐱

𝑡 = T switcher 𝑠

Most existing perception models are deterministic. To align with these, we can remove the noise input zyt and only input the encoded image features zx to the U-Net denoiser. The model still performs well. In this paper, we finally present two versions of Lotus: Lotus-G (generative) with noise input and Lotus-D (discriminative) without noise input, catering to different needs.

[Figure 93]

| | | | | | |
|---|---|---|---|---|---|
| | | | | |[Figure 94]|
| | | | | | |
| | | | | | |

𝒟

Prediction 𝒛𝐲

Figure 10: Inference Pipeline of Lotus. The noise zyT in bracket is used for Lotus-G and omitted for Lotus-D.

- 4.5 INFERENCE

The inference pipeline is illustrated in Fig. 10. We initialize the annotation map with standard Gaussian noise zyT, and encode the input image into its latent code zx. The noise zyT and the image zx are concatenated and fed into the denoiser U-Net model. In our single-step formulation, we set t = T and the switcher to sy. The denoiser U-Net model then predicts the latent code of the annotation map. The final annotation map is decoded from the predicted latent code via the VAE decoder. For deterministic prediction, we eliminate the Gaussian noise zyT and only feed the latent code of the input image into U-Net.

- 5 EXPERIMENTS

- 5.1 EXPERIMENTAL SETTINGS

Implementation details. We implement Lotus based on Stable Diffusion V2 (Rombach et al., 2022), without text conditioning. During training, we fix the time-step t = 1000. For depth estimation, we predict in disparity space, i.e., d = 1/d′, where d represents the values in disparity space and d′ denotes the true depth. For more details, please see Sec. A.1 of the supplementary materials.

Training Datasets. Both depth and normal estimation are trained on two synthetic dataset covering indoor and outdoor scenes: ① Hypersim (Roberts et al., 2021) is a photorealistic synthetic dataset featuring 461 indoor scenes. We use the official training split, which contains approximately 54K samples. After filtering out incomplete samples, around 39K samples remain, all resized to 576×768 for training. ② Virtual KITTI (Cabon et al., 2020) is a synthetic street-scene dataset with five urban scenes under various imaging and weather conditions. We utilize four of these scenes for training, comprising about 20K samples. All samples are cropped to 352 × 1216, with the far plane at 80m.

Following Marigold (Ke et al., 2024), we probabilistically choose one of the two datasets and then draw samples from it for each batch (Hypersim 90% and Virtual KITTI 10%).

Evaluation Datasets and Metrics. ① For zero-shot affine-invariant depth estimation, we evaluate Lotus on NYUv2 (Silberman et al., 2012), ScanNet (Dai et al., 2017), KITTI (Geiger et al., 2013), ETH3D (Schops et al., 2017), and DIODE (Vasiljevic et al., 2019) using absolute mean relative error (AbsRel), and also report δ1 and δ2 values. ② For surface normal prediction, we employ NYUv2, ScanNet, iBims-1 (Koch et al., 2018), Sintel (Butler et al., 2012), and OASIS (Chen et al., 2020) datasets, reporting mean angular error (m.) as well as the percentage of pixels with an angular error below 11.25◦ and 30◦. Please see Sec. A.2 of the supplementary materials for further details on the evaluation datasets and metrics.

- 5.2 QUANTITATIVE COMPARISONS

- ① For depth estimation (Tab. 1), Lotus-G demonstrates promising performance across all evaluation datasets, achieving the overall best rank compared to other generative baselines. Notice that we only require single step denoising process, significantly boosting the inference speed as shown in Fig. 3. Lotus-D also performs well, achieving comparable results to DepthAnything series. It is worthy to notice that Lotus is trained on only 0.059M images compared to DepthAnything’s 62.6M images.
- ② For normal estimation (Tab. 2), both Lotus-G and Lotus-D outperform all other generative and discriminative methods on zero-shot surface normal estimation with significant margins. Please see Sec. H of the supplementary materials for Qualitative Comparisons.

- 5.3 ABLATION STUDY

As shown in Tab. 3, we conduct ablation studies to validate our designs. Starting with “Direct Adaptation”, we incrementally test the effects of different components, such as parameterization types, the single-step diffusion process, and the detail preserver. Initially, we train the model using only the Hypersim dataset to establish a baseline. We then expand the training dataset using a mixture dataset strategy by including Virtual KITTI, aiming to enhance the model’s generalization ability across different domains. For depth estimation, we further train the model in the disparity space to improve the accuracy. The findings from these ablations validate the effectiveness of our proposed adaptation protocol, demonstrating that each design plays a vital role in optimizing the diffusion models for dense prediction tasks.

- Table 1: Quantitative comparison on zero-shot affine-invariant depth estimation between Lotus and SoTA methods. The upper section lists discriminative methods, the lower lists generative ones. The best and second best performances are highlighted. Lotus-G outperforms all others methods while Lotus-D is only slightly inferior to DepthAnything. §indicates results revised by ourselves, following Marigold (Ke et al., 2024). ⋆denotes the method relies on pre-trained Stable Diffusion.

Method

Training NYUv2 (Indoor) KITTI (Outdoor) ETH3D (Various) ScanNet (Indoor) DIODE (Various) Avg

Data↓ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ Rank

DiverseDepth 320K 11.7 87.5 - 19.0 70.4 - 22.8 69.4 - 10.9 88.2 - 37.6 63.1 - 10.6 MiDaS 2M 11.1 88.5 - 23.6 63.0 - 18.4 75.2 - 12.1 84.6 - 33.2 71.5 - 10.2 LeRes 354K 9.0 91.6 - 14.9 78.4 - 17.1 77.7 - 9.1 91.7 - 27.1 76.6 - 7.8 Omnidata 12.2M 7.4 94.5 - 14.9 83.5 - 16.6 77.8 - 7.5 93.6 - 33.9 74.2 - 7.5 DPT 1.4M 9.8 90.3 - 10.0 90.1 - 7.8 94.6 - 8.2 93.4 - 18.2 75.8 - 5.8 HDN 300K 6.9 94.8 - 11.5 86.7 - 12.1 83.3 - 8.0 93.9 - 24.6 78.0 - 5.3

GenPercept⋆§ 74K 5.6 96.0 99.2 13.0 84.2 97.2 7.0 95.6 98.8 6.2 96.1 99.1 35.7 75.6 86.6 4.9 Diffusion-E2E-FT⋆ 74K 5.4 96.5 99.1 9.6 92.1 98.0 6.4 95.9 98.7 5.8 96.5 98.8 30.3 77.6 87.9 3.6 DepthAnything V2 62.6M 4.5 97.9 99.3 7.4 94.6 98.6 13.1 86.5 97.5 4.2 97.8 99.3 26.5 73.4 87.1 3.5 Lotus-D (Ours)⋆ 59K 5.1 97.2 99.2 8.1 93.1 98.7 6.1 97.0 99.1 5.5 96.5 99.0 22.8 73.8 86.2 3.0 DepthAnything 62.6M 4.3 98.1 99.6 7.6 94.7 99.2 12.7 88.2 98.3 4.3 98.1 99.6 26.0 75.9 87.5 2.4

GeoWizard⋆§ 280K 5.6 96.3 99.1 14.4 82.0 96.6 6.6 95.8 98.4 6.4 95.0 98.4 33.5 72.3 86.5 3.3 Marigold(LCM)⋆§ 74K 6.1 95.8 99.0 9.8 91.8 98.7 6.8 95.6 99.0 6.9 94.6 98.6 30.7 77.5 89.3 2.9 Marigold⋆ 74K 5.5 96.4 99.1 9.9 91.6 98.7 6.5 95.9 99.0 6.4 95.2 98.8 30.8 77.3 88.7 2.1 Lotus-G (Ours)⋆ 59K 5.4 96.8 99.2 8.5 92.2 98.4 5.9 97.0 99.2 5.9 95.7 98.8 22.9 72.9 86.0 1.3

- Table 2: Quantitative comparison on zero-shot surface normal estimation between Lotus and SoTA methods. Discriminative methods are shown in the upper section, generative methods in the lower. Both Lotus-D and Lotus-G outperform all other methods. ‡refers the Marigold normal model as detailed in this link. ⋆denotes the method relies on pre-trained Stable Diffusion.

Method

Training NYUv2 (Indoor) ScanNet (Indoor) iBims-1 (Indoor) Sintel (Outdoor) OASIS (Various) Avg. Data↓ m.↓ 11.25◦↑ 30◦↑ m.↓ 11.25◦↑ 30◦↑ m.↓ 11.25◦↑ 30◦↑ m.↓ 11.25◦↑ 30◦↑ m.↓ 11.25◦↑ 30◦↑ Rank

OASIS 110K 29.2 23.8 60.7 32.8 15.4 52.6 32.6 23.5 57.4 43.1 7.0 35.7 - - - 7.8 Omnidata 12.2M 23.1 45.8 73.6 22.9 47.4 73.2 19.0 62.1 80.1 41.5 11.4 42.0 24.9 31.0 71.4 5.9 EESNU 2.5M 16.2 58.6 83.5 - - - 20.0 58.5 78.2 42.1 11.5 41.2 27.7 24.0 66.6 5.8 GenPercept§⋆ 74K 18.2 56.3 81.4 17.7 58.3 82.7 18.2 64.0 82.0 37.6 16.2 51.0 26.3 26.9 71.1 4.9 Omnidata V2 12.2M 17.2 55.5 83.0 16.2 60.2 84.7 18.2 63.9 81.1 40.5 14.7 43.5 24.2 27.7 74.2 4.4 DSINE 160K 16.4 59.6 83.5 16.2 61.0 84.4 17.1 67.4 82.3 34.9 21.5 52.7 24.4 28.8 72.0 3.1 Diffusion-E2E-FT§⋆ 74K 16.5 60.4 83.1 14.7 66.1 85.1 16.1 69.7 83.9 33.5 22.3 53.5 23.2 29.4 74.5 1.9 Lotus-D (Ours)⋆ 59K 16.2 59.8 83.9 14.7 64.0 86.1 17.1 66.4 83.0 32.3 22.4 57.0 22.3 31.8 76.1 1.4

Marigold‡⋆ 74K 20.9 50.5 - 21.3 45.6 - 18.5 64.7 - - - - - - - 3.6 GeoWizard§⋆ 280K 18.9 50.7 81.5 17.4 53.8 83.5 19.3 63.0 80.3 40.3 12.3 43.5 25.2 23.4 68.1 3.1 StableNormal§⋆ 250K 18.6 53.5 81.7 17.1 57.4 84.1 18.2 65.0 82.4 36.7 14.1 50.7 26.5 23.5 68.7 2.1 Lotus-G (Ours)∗ 59K 16.5 59.4 83.5 15.1 63.9 85.3 17.2 66.2 82.7 33.6 21.0 53.8 22.7 29.4 75.8 1.0

6 CONCLUSION AND FUTURE WORK

In this paper, we introduce Lotus, a diffusion-based visual foundation model for dense prediction. Through systematic analysis and tailored diffusion formulation, Lotus finds a way to better fit the rich visual prior from pre-trained diffusion models into dense prediction. Extensive experiments demonstrate that Lotus achieves promising performance on zero-shot depth and normal estimation with minimal training data, paving the way of various practical applications. Please see the supplementary materials for our discussion about Applications (Sec. I) and Future Work (Sec. J).

- Table 3: Ablation studies on the step-by-step design of our adaptation protocol for fitting pre-trained diffusion models into dense prediction. Here we show the results in monocular depth estimation.

Training NYUv2 (Indoor) KITTI (Outdoor) ETH3D (Various) ScanNet (Indoor)

Method

Data AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑ AbsRel↓ δ1↑ δ2↑

Direct Adaptation 39K 11.551 87.692 96.122 20.164 70.403 90.996 19.894 76.464 87.960 15.726 78.885 93.651 + x0-prediction 39K 8.332 92.769 97.941 17.008 74.969 93.611 11.075 87.952 94.978 10.212 89.130 97.181 + Single Time-step 39K 5.587 96.272 99.113 13.262 83.210 97.237 7.586 94.143 97.678 6.262 95.394 98.791 + Detail Preserver 39K 5.555 96.303 99.118 13.170 83.657 97.454 7.147 95.000 98.058 6.201 95.470 98.814

+ Mixture Dataset 59K 5.425 96.597 99.156 11.324 87.692 97.780 6.172 96.077 98.980 6.024 96.026 99.730 → − Noise Input 59K 5.334 96.729 99.198 9.334 92.813 98.795 6.846 95.290 98.899 5.982 96.287 99.087 + Disparity Space (Lotus-G) 59K 5.379 96.736 99.155 8.521 92.206 98.374 5.878 97.024 99.233 5.925 95.727 98.839 → − Noise Input (Lotus-D) 59K 5.123 97.182 99.134 8.117 93.097 98.654 6.147 96.964 99.077 5.494 96.534 99.039

SUPPLEMENTARY MATERIALS OF LOTUS: DIFFUSION-BASED VISUAL FOUNDATION MODEL FOR HIGH-QUALITY DENSE PREDICTION

- A EXPERIMENTAL SETTINGS

- A.1 IMPLEMENTATION DETAILS

We implement Lotus based on Stable Diffusion V2 (Rombach et al., 2022), with text conditioning disabled. Both the depth and normal maps are normalized to the range [−1,1] to match the designed input value range of the VAE. During training, we fix the time-step t = 1000. To optimize the model, we utilize the standard Adam optimizer with the learning rate 3×10−5. All experiments are conducted on 8 NVIDIA A800 GPUs and the total batch size is 128. For our discriminative variant, we train for 4,000 steps, which takes ∼8.1 hours, while for the generative variant, we extend training to 10,000 steps, requiring ∼20.3 hours.

- A.2 EVALUATION DATASETS AND METRICS

Evaluation Datasets. ① For affine-invariant depth estimation, we evaluate on 4 real-world datasets that are not seen during training: NYUv2 (Silberman et al., 2012) and ScanNet (Dai et al., 2017) all contain images of indoor scenes; KITTI (Geiger et al., 2013) contains various outdoor scenes; ETH3D (Schops et al., 2017), a high-resolution dataset, containing both indoor and outdoor scenes. ② For surface normal prediction, we employ 4 datasets for evaluation: NYUv2 (Silberman et al., 2012), ScanNet (Dai et al., 2017), and iBims-1 (Koch et al., 2018) contain real indoor scenes; Sintel (Butler et al., 2012) contains highly dynamic outdoor scenes.

Metrics. ① For affine-invariant depth, we follow the evaluation protocol from (Ranftl et al., 2020; Ke et al., 2024; Yang et al., 2024a;b), aligning the estimated depth predictions with available ground truths using least-squares fitting. The accuracy of the aligned predictions is assessed using the

absolute mean relative error (AbsRel), i.e., M1 Mi=1 |ai − di|/di, where M is the total number of pixels, ai is the predicted depth map and di represents the ground truth. We also report δ1 and δ2, the proportion of pixels satisfying Max(ai/di,di/ai) < 1.25 and < 1.252 respectively.

② For surface normal, following (Bae & Davison, 2024; Ye et al., 2024), we evaluate the predictions of Lotus by measuring the mean angular error for pixels with available ground truth. Additionally, we report the percentage of pixels with an angular error below 11.25◦ and 30◦.

For all tasks, we report the Avg. Rank, which indicates the average ranking of each method across various datasets and evaluation metrics. A lower value signifies better overall performance.

- B DETAILS OF DIRECT ADAPTION

As illustrated in Fig. 4 of the main paper, our Direct Adaption means directly adapting the standard diffusion formulation for dense prediction task with minimal modifications. Specifically, starting with the pre-trained Stable Diffusion model, image x and annotation y are encoded using the pretrained VAE encoder. Noise is added to the encoded annotation to obtain the noisy annotation zyt

- at noise level t ∈ [1,T]. The encoded image zx is then concatenated with the noisy annotation

zyt to form the input of the denoiser U-Net model. To handle this concatenated input, the U-Net input layer is duplicated (from 4 channels to 8 channels) and its original weights are halved as

initialization, which prevents activation inflation (Ke et al., 2024). Direct Adaptation is optimized using the standard multi-step formulation the standard diffusion objective, ϵ-prediction, as described in Eq. 2 (first row) of the main paper. To analyze the original diffusion formulation more effectively, we avoid specialized techniques introduced in prior methods (Ke et al., 2024; Fu et al., 2024; Xu et al., 2024; Ye et al., 2024), such as annealed multi-resolution noise (AMRN).

The AMRN strategy aims to reduce the model’s variance, which has a similar effect to our design, x0-pred., but through a different solution. This diminishes the impact of our method. Therefore,

Table 4: Experiments based on Marigold w/ AMRN.

NYUv2 KITTI AbsRel↓ δ1↑ AbsRel↓ δ1↑

Index Method

- 1-1 ϵ-pred. 6.746 95.021 11.827 87.065

- 1-2 ϵ-pred. + single step 6.691 94.552 13.395 76.269

- 1-3 ϵ-pred. + single step + detail preserver 6.547 94.772 12.815 77.829

- 2-1 v-pred. 6.358 95.188 10.796 89.726

- 2-2 v-pred. + single step 5.499 96.415 11.132 88.520

- 2-3 v-pred. + single step + detail preserver 5.422 96.517 10.761 89.826

- 3-1 x0-pred. 6.262 95.501 10.769 89.643

- 3-2 x0-pred. + single step 5.495 96.431 11.237 88.457

- 3-3 x0-pred. + single step + detail preserver 5.418 96.542 10.651 89.887

Table 5: Experiments based on Marigold w/o AMRN.

NYUv2 KITTI AbsRel↓ δ1↑ AbsRel↓ δ1↑

Index Method

- 1-1 ϵ-pred. 13.110 85.083 17.655 75.581

- 1-2 ϵ-pred. + single step 6.605 94.583 13.406 76.298

- 1-3 ϵ-pred. + single step + detail preserver 6.582 94.768 12.823 77.983

- 2-1 v-pred. 10.634 89.448 14.328 84.026

- 2-2 v-pred. + single step 5.498 96.562 11.173 88.314

- 2-3 v-pred. + single step + detail preserver 5.459 96.657 10.814 89.081

- 3-1 x0-pred. 8.058 92.834 12.177 86.301

- 3-2 x0-pred. + single step 5.477 96.615 11.166 88.640

- 3-3 x0-pred. + single step + detail preserver 5.396 96.717 10.575 89.804

it is preferable to validate the effect of our designs w/o AMRN. We validate this claim using the Marigold codebase, both w/ and w/o AMRN, as shown in the Tab. 4 and Tab. 5, respectively. In Tab. 5, the performance of multi-step models follows the order: ϵ-pred. < v-pred. < x0-pred. However, in Tab. 4, the differences between three parameterization types are minimal, particularly the performance of v-pred. and x0-pred. are nearly identical. This can be attributed to the influence of AMRN, which is specifically designed for multi-step diffusion models to reduce variance and enhance performance. As a result, x0-pred. shows no significant difference in reducing variance compared to the other two parameterizations. In Tab. 5, when the number of time-steps is reduced to one, the performance of the model improves regardless of the parameterization type used. However, in Tab. 4, the effect of single-step is unstable. This unexpected phenomenon arises from the complex, multifaceted effects of AMRN when transitioning from multi-step to single-step: ① AMRN significantly improves the multi-step model, but its effect is lost when the number of time-steps is reduced to one. ② In the single-step model, convergence is easier with limited data, leading to a slight improvement in performance. However, this also leads to catastrophic forgetting, which reduces the model’s ability to handle detailed areas, especially on the KITTI dataset. In both Tab. 4 and Tab. 5, Detail Preserver further enhances the performance of single-step model, particularly on the KITTI dataset, which contains more complex and detailed areas, such as pedestrians and fences, compared to the NYUv2 dataset. In both Tab. 4 and Tab. 5, when using a single step (t = T), according to vt = √α¯Tϵ −

√1 − α¯Tz, since √α¯T ≈ 0 when t = T, v-pred. becomes equivalent to x0-pred. This explains why the performances of v-pred. and x0-pred. are nearly identical in single-step, with only minor differences. In conclusion, these experiments show that AMRN, which has a similar effect to our designs but is achieved through a different solution, diminishing the impact of our proposed designs. Therefore, it is preferable to validate the effect of our designs w/o AMRN. The experiments on Marigold w/o AMRN (Tab. 5) validate the effectiveness of our proposed designs, as stated in our main paper, where the best protocol is x0-pred. + single step + detail preserver.

- C ANALYSIS OF “DIRECTION(zyτ)” IN DDIM PROCESS (EQ. 4)

In addition to the predicted clean sample ˆzyτ , Eq. 4 of the main paper includes another term, “direction(zyτ )”. It is calculated according to different parameterization types:

ϵ-prediction: d = wτ · fθϵ x0-prediction: d = wτ · [

1 √1 − ατ

(zyτ −

√ατfθz)]

(7)

where d represents the term “direction(zyτ )”, wτ = √1 − ατ−1 is the weight at denoising step τ. And fθϵ and fθz denote the model outputs for different parameterizations. For clarity, the input of the model fθ is omitted. As shown in Eq. 7, for x0-prediction, when τ → 1, i.e., at the end of the denoising process, the factor √1 − ατ → 0, which may amplify variance from fθz. However, its influence is limited. The reasons are as follows: ① The rate of change of √1 − ατ from T to 1 is initially slow and then accelerates. As a result, the factor remains close to 1 for most of the denoising process, only close to 0 in the final steps. ② In x0-prediction, compared to the initial denoising steps, the gap between network output fθz and zyτ in the final steps is much weaker and gradually approaching zero. With √ατ → 1 as τ → 1, we can get zyτ −

√ατfθz → 0, which may also indicate the limited influence of factor √1 − ατ.

- D PERFORMANCE OF v-PREDICTION

1000 800 600 400 200 0 Denoising Step ( )

- 8

- 9

- 10

- 11

- 12

NYUv2AbsRel(%)

NYUv2 AbsRel vs. Denoising Step

Pred. Type=x0

Pred. Type=

Pred. Type=v

Figure 11: Quantitative evaluation of the predicted depth maps ˆzyτ along the denoising process. The experimental settings are same as Fig. 5 and 6. Six steps are selected for illustration. The banded regions around each line indicate the variance, wider areas representing larger variance.

In sec. 4.1, we discussed two basic parameterization types: ϵ-prediction and x0-prediction. The latest parameterization, v-prediction (Salimans & Ho, 2022), combines these two basic parameterizations to avoid the invalid prediction values of ϵ-prediction at some timesteps for progressive distillation. Specifically, the U-Net denoiser model fθ learns to predict the combination of added noise ϵ and the clean sample zy: v = √ατϵ −

√1 − ατzy, where

√ατ2 + √1 − ατ2 = 1. During inference, according to the Eq. 4 of main paper, the pre-

diction ˆzyτ = √ατzyτ −

√1 − ατfθv, where

fθv represents the predicted combination, striking a balance between ϵ (ϵ-prediction) and zy

(x0-prediction). As shown in Fig. 11, we conduct experiments based on the settings in Fig. 5 and 6 of the main paper. The results indicate

that the performance of v-prediction falls between that of x0-prediction and ϵ-prediction, with moderate variance. However, for dense prediction tasks, minimizing variance is crucial to avoid unsta-

ble prediction. Therefore, v-prediction may not be the optimal choice. In contrast, x0-prediction achieves the best performance with the lowest variance, which is why we replace the standard ϵ-

prediction with the more suitable x0-prediction.

- E EXPERIMENTS ON MORE DENSE PREDICTION TASKS: SEMANTIC SEGMENTATION AND DIFFUSE REFLECTANCE

To validate the generalization ability of our method on other dense prediction tasks, we further train it on semantic segmentation and diffuse reflectance prediction. Both tasks are trained using the training set of the Hypersim dataset (Roberts et al., 2021) and evaluated on their corresponding test sets. For semantic segmentation, we report the mean intersection over union (mIoU) and mean accuracy (mAcc). For diffuse reflectance prediction, we evaluate using the L1 and L2 distances to the ground truth. To enable fast evaluation, we randomly select 500 paired testing samples. In our experiments, we do not redesign any specific modules or loss functions for these tasks and maintain the original training protocol of Lotus unchanged. As shown in Tab. 6 and Tab. 7, we compare our

[Figure 95]

Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

[Figure 96]

[Figure 97]

(a) Semantic Segmentation

(b) Diffuse Reflectance

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

Input Image Ground Truth Prediction

Ground Truth Prediction

- Figure 12: Experiments of Lotus on (a) semantic segmentation and (b) diffuse reflectance. The high-quality results indicate that our method, even without task-specific designs, can be effectively applied not only to geometric dense prediction tasks, but also to semantic dense prediction tasks.

Table 6: The quantitative results of semantic segmentation on Hypersim (Roberts et al., 2021) testing set. Mean values are reported from 10 independent runs.

Method mIoU ↑ mAcc ↑

Direct Adaption 14.1 61.3 Lotus-G 21.2 65.6

Table 7: The quantitative results of diffuse reflectance prediction on Hypersim (Roberts et al., 2021) testing set. Mean values are reported from 10 independent runs.

Method L1 ↓ L2 ↓ Direct Adaption 0.198 0.206 Lotus-G 0.109 0.135

method with the baseline, Direct Adaption (Fig. 4 in the main paper), to assess its effectiveness. The results show that our method outperforms the baseline across all metrics. Additionally, we provide qualitative visualizations for these two tasks in Fig. 12, demonstrating accurate and high-quality results. Both the quantitative and qualitative results indicate that our method, even without taskspecific designs, can be effectively applied not only to geometric dense prediction tasks, as shown in the main paper, but also to semantic dense prediction tasks.

- F FREQUENCY DOMAIN ANALYSIS OF THE DETAIL PRESERVER TAKE MONOCULAR DEPTH ESTIMATION AS AN EXAMPLE

We use fast Fourier transform (FFT) to compute the Discrete Fourier Transform (DFT) of the input images and depth map estimations with and without Detail Preserver. The entire 2D frequency domains are divided into 8 frequency groups exponentially using the base of 2, i.e., the first group covers the 2D frequency map in a circle with a radius of 2, the second group covers the annular region with radii from 2 to 4, the third group covers radii from 4 to 8, and so on. This exponential grouping allows us to analyze the frequency components across progressively larger ranges, capturing both low-frequency and high-frequency characteristics.

[Figure 111]

[Figure 112]

(b) Frequency energy ratio between input image and GT depth.

(a) Frequency domain energy distribution comparisons among input image, and depth estimations w/ and w/o Detail Preserver.

[Figure 113]

[Figure 114]

(c) Frequency energy ratio between input image and depth estimations w/ Detail Preserver.

(d) Frequency energy ratio between the input image and depth estimations w/o Detail Preserver.

## Figure 13: Frequency Domain Analysis of the Detail Preserver We use Hypersim (Roberts et al.,

2021) dataset to transfer the input image and depth estimation w/ and w/o Detail Preserver into 2D frequency domains, using FFT. 100 pairs of {input image, depth estimation w/ Detail Preserver, depth estimation w/o Detail Preserver} are randomly selected for this frequency domain analysis. Hypersim is a photorealistic synthetic dataset. Not only can Hypersim offer dense GT labels without None areas (which is important during FFT), its depth annotations are much fine-grained compared with real-world datasets like NYUv2 (Silberman et al., 2012) and KITTI Geiger et al. (2013).

In order to more clearly demonstrate the effect of our proposed Detail Preserver, we first analysis the experiments using Hypersim (Roberts et al., 2021) dataset to display the difference in frequency domain energy between the details from both geometry and texture (the input images); and the details from purely the geometry (the GT depth maps). As shown in Fig. 13b, the frequency domain energy between the input images and the depth annotations are plotted. Clearly we can see that the input images has much higher frequency energy in high-frequency areas, i.e., group 4, 5, 6, and 7, indicating that the details in surface textures mainly contribute to high-frequency energy; while the details in geometries, which can be expressed by depth maps, are mainly concentrated into (relative) middle and low frequency areas, i.e., group 0, 1, 2, and 3.

As shown in Fig. 13a , collaborating with the Detail Preserver effectively drag the frequency domain energy of depth estimation to the input image, especially on middle and low frequency domains, i.e., the frequency group 0, 1, 2 and 3, highlighting the Detail Preserver’s effectiveness in enhancing the geometrical details that should be reflected into depth predictions, like the fences around roads and houses (Fig. 8 of our main paper).While for high-frequency components, i.e., the frequency group 4, 5, 6, and 7, which may be primarily caused by the highly detailed textures, like the signs on the road and patterns on house surfaces, the energy in these areas between depth estimations with and without Detail Preserver is quite similar, indicating that the Detail Preserver does not copy this high-frequency and geometry-independent texture.

By comparing Fig. 13b, 13c and 13d together, we can see that Detail Preserver effectively enhances the details of geometries. This insight is evident by this phenomenon: the frequency domain energy ratio between input and depth estimation w/ Detail Preserver, is closer to the frequency domain energy ratio between input and GT depth, compared with the frequency domain energy ratio between input and depth estimation w/o Detail Preserver.

- G THE EFFECT OF DIFFERENT TIME-STEPS t IN ONE-STEP DIFFUSION

In Sec. 4.2 of our main paper, we reduce the number of training time-steps of diffusion formulation to only one, and fixing the only time-step t to T following the diffusion formulation. In this section, we evaluate the effect of different time-steps t in one-step diffusion, rather than exclusively fixing t = T, to validate that the rule of basic diffusion formulations should better be followed. Violating it will lead to performance degradation. As shown in Tab. 8, we conduct experiments on Hypersim dataset (Roberts et al., 2021) and evaluated on NYUv2 dataset (Silberman et al., 2012), without employing the detail preserver or mixture dataset training. The results indicate that the model performs best when t = T (t = 1000). Changing t leads to a slight degradation in performance.

Table 8: The effect of different time-steps t in one-step diffusion. In this experiment, the models are trained on Hypersim dataset (Roberts et al., 2021) and evaluated on NYUv2 dataset (Silberman et al., 2012), without employing the detail preserver or mixture dataset training.

Time-step t = 1000 t = 750 t = 500 t = 250 t = 1 AbsRel ↓ 5.587 5.631 5.727 5.663 5.737 δ1 ↑ 96.272 96.165 96.087 96.141 96.080

- H QUALITATIVE COMPARISONS

In Fig. 14, we further compare the performance of our Lotus with other methods in detailed areas. The quantitative results obviously demonstrate that our method can produce much finer and more accurate depth predictions, particularly in complex regions with intricate structures, which sometimes cannot be reflected by the metrics. Also, as illustrated in Fig. 15, Lotus consistently provides accurate surface normal predictions, effectively handling complex geometries and diverse environments, highlighting its robustness on fine-grained prediction.

- I APPLICATIONS OF LOTUS

Thanks to its superiority, Lotus can seamlessly support a variety of applications. Fig. 16 illustrates four key applications: ① Depth to Point Cloud. The depth maps estimated by Lotus are projected into 3D point clouds; ② Joint Estimation. By incorporating a task switcher, Lotus can perform multiple tasks simultaneously, such as joint depth and normal map estimation with 100% shared network parameters; ③ Single-View Reconstruction. Using Lotus’s normal predictions, high-quality meshes can be reconstructed through through Bilateral Normal Integration (Cao et al., 2022); ④ Multi-View Reconstruction. Leveraging per-view depth and normal predictions from Lotus, highquality meshes can be reconstructed with MonoSDF (Yu et al., 2022), without RGB supervision, showcasing Lotus’s robustness and accurate spatial understanding. These applications emphasize the importance of Lotus in the field of computer vision. Its accuracy and efficiency will help in addressing increasingly complex problems.

- J FUTURE WORK

While we have applied Lotus to two geometric dense prediction tasks, it can be seamlessly adapted to other dense prediction tasks requiring per-pixel alignment with great potential, such as panoramic segmentation and image matting. Additionally, our performance is slightly behind DepthAnything (Yang et al., 2024a) which utilizes large-scale training data. In the future, scaling up the training data, as reveal in Fig. 7 and Tab. 3 (“Mixture Dataset”) of the main paper, has great potential to further enhance Lotus’s performance.

[Figure 115]

## Figure 14: Qualitative comparison on zero-shot affine-invariant depth estimation. Lotus demonstrates higher accuracy especially in detailed areas.

[Figure 116]

#### Figure 15: Qualitative comparison on zero-shot surface normal estimation. Lotus offers improved accuracy particularly in complex regions.

[Figure 117]

- Figure 16: Applications of Lotus. ① Depth to 3D Point Clouds. ② Joint Estimation: Simultaneous depth and normal estimation with 100% shared parameters. ③ Single-View Reconstruction: Reconstructing 3D meshes from normal predictions. ④ Multi-View Reconstruction: Reconstructing high-quality meshes using depth/normal predictions without RGB supervision.

REFERENCES

Gilwon Bae and Andrew J Davison. Aleatoric uncertainty in monocular surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), pp. 1472–1485, 2021.

Gilwon Bae and Andrew J Davison. Rethinking inductive biases for surface normal estimation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Yaniv Benny and Lior Wolf. Dynamic dual-output diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11482–11491, 2022.

Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pp. 611–625. Springer, 2012.

Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020.

Xu Cao, Hiroaki Santo, Boxin Shi, Fumio Okura, and Yasuyuki Matsushita. Bilateral normal integration. In ECCV, 2022.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Weifeng Chen, Shengyi Qian, David Fan, Noriyuki Kojima, Max Hamilton, and Jia Deng. Oasis: A large-scale dataset for single image 3d in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 679–688, 2020.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.

Wenyu Du, Shuang Cheng, Tongxu Luo, Zihan Qiu, Zeyu Huang, Ka Chun Cheung, Reynold Cheng, and Jie Fu. Unlocking continual learning abilities in language models. arXiv preprint arXiv:2406.17245, 2024.

Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10786–10796, 2021.

David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014.

Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2002–2011, 2018.

Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. arXiv preprint arXiv:2403.12013, 2024.

Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. arXiv preprint arXiv:2409.11355, 2024.

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237, 2013.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Jing He, Yiyi Zhou, Qi Zhang, Jun Peng, Yunhang Shen, Xiaoshuai Sun, Chao Chen, and Rongrong Ji. Pixelfolder: An efficient progressive pixel synthesis network for image generation. arXiv preprint arXiv:2204.00833, 2022.

Jing He, Haodong Li, Yongzhe Hu, Guibao Shen, Yingjie Cai, Weichao Qiu, and Ying-Cong Chen. Disenvisioner: Disentangled and enriched visual prompt for customized image generation. arXiv preprint arXiv:2410.02067, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. arXiv preprint arXiv:2404.15506, 2024.

Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17853–17862, 2023.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024. doi: 10.1145/3641519.3657428.

O˘guzhan Fatih Kar, Teresa Yeo, Andrei Atanov, and Amir Zamir. 3d common corruptions and data augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18963–18974, 2022.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8110–8119, 2020.

Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34:852–863, 2021.

Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9492–9502, 2024.

Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Korner. Evaluation of cnn-based single-image depth estimation methods. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pp. 0–0, 2018.

Hsin-Ying Lee, Hung-Yu Tseng, and Ming-Hsuan Yang. Exploiting diffusion prior for generalizable dense prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7861–7871, 2024.

Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019.

Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421, 2024.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9970–9980, 2024.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020.

Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 12179–12188, 2021.

Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10912–10922, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention– MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3260–3269, 2017.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pp. 746–760. Springer, 2012.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Yunzhou Song, Jiahui Lei, Ziyun Wang, Lingjie Liu, and Kostas Daniilidis. Track everything everywhere fast and robustly, 2024.

Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.

Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764, 2024.

Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Diffusion models trained with large data are transferable visual models. arXiv preprint arXiv:2403.06090, 2024.

Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1316–1324, 2018.

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10371–10381, 2024a.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024b.

Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. arXiv preprint arXiv:2406.16864, 2024.

Wei Yin, Yifan Liu, and Chunhua Shen. Virtual normal: Enforcing geometric constraints for accurate and robust depth prediction. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):7282–7295, 2021a.

Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 204–213, 2021b.

Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9043–9053, 2023.

Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, and Ping Tan. Neural window fully-connected crfs for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3916–3925, 2022.

Ekim Yurtsever, Jacob Lambert, Alexander Carballo, and Kazuya Takeda. A survey of autonomous driving: Common practices and emerging technologies. IEEE access, 8:58443–58469, 2020.

Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the catastrophic forgetting in multimodal large language models. arXiv preprint arXiv:2309.10313, 2023.

Chi Zhang, Wei Yin, Billzb Wang, Gang Yu, Bin Fu, and Chunhua Shen. Hierarchical normalization for robust monocular depth estimation. Advances in Neural Information Processing Systems, 35: 14128–14139, 2022.

Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pp. 5907–5915, 2017.

Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan++: Realistic image synthesis with stacked generative adversarial networks. IEEE transactions on pattern analysis and machine intelligence, 41(8):1947–1962, 2018.

Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-to-image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 833–842, 2021.

