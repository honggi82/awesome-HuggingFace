# arXiv:2412.09013v2[cs.CV]13Mar2025

## Arbitrary-steps Image Super-resolution via Diffusion Inversion

Zongsheng Yue1,2, Kang Liao2, Chen Change Loy2 1Xi’an Jiaotong University, Xi’an, China 2S-Lab, Nanyang Technological University, Singapore

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

DiffBIR-50 (7937ms)

OSEDiff-1 (176ms)

StableSR-50 (3459ms)

ResShift-4 (319ms)

SinSR-1 (138ms)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Ours-3 (176ms)

Ours-4 (207ms)

Ours-2 (149ms)

Ours-1 (117ms)

Ours-5 (244ms)

Zoomed LR

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

OSEDiff-1

StableSR-50

DiffBIR-50

ResShift-4

SinSR-1

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Ours-1

Ours-2

Ours-3

Ours-4

Ours-5

Zoomed LR

Figure 1. Qualitative comparisons of our proposed method to recent state-of-the-art diffusion-based approaches on two real-world examples, where the number of sampling steps is annotated in the format “Method name-Steps”. We provide the runtime (in milliseconds) highlighted by red in the sub-caption of the first example , which is tested on ×4 (128 → 512) SR task on an A100 GPU. Our method offers an efficient and flexible sampling mechanism, allowing users to freely adjust the number of sampling steps based on the degradation type or their specific requirements. In the first example, mainly degraded by blurriness, multi-step sampling is preferable to single-step sampling as it progressively recovers finer details. Conversely, in the second example with severe noise, a single sampling step is sufficient to achieve satisfactory results, whereas additional steps may amplify the noise and introduce unwanted artifacts. (Zoom-in for best view)

### Abstract

This study presents a new image super-resolution (SR) technique based on diffusion inversion, aiming at harnessing the rich image priors encapsulated in large pre-trained diffusion models to improve SR performance. We design a Partial noise Prediction strategy to construct an intermediate state of the diffusion model, which serves as the starting sampling point. Central to our approach is a deep noise predictor to estimate the optimal noise maps for the forward diffusion process. Once trained, this noise

predictor can be used to initialize the sampling process partially along the diffusion trajectory, generating the desirable high-resolution result. Compared to existing approaches, our method offers a flexible and efficient sampling mechanism that supports an arbitrary number of sampling steps, ranging from one to five. Even with a single sampling step, our method demonstrates superior or comparable performance to recent state-of-the-art approaches. The code and model are publicly available at https: //github.com/zsyOAOA/InvSR.

### 1. Introduction

Image super-resolution (SR) is a fundamental yet challenging problem in computer vision, aiming to restore a high-resolution (HR) image from a given low-resolution (LR) observation. The main challenge of SR arises from the complexity and often unknown nature of the degradation model in real-world scenarios, making SR an illposed problem. Recent breakthroughs in diffusion models [16, 45, 48], particularly large-scale text-to-image (T2I) models, have demonstrated remarkable success in generating high-quality images. Owing to the strong generative capability of these T2I models, recent studies have begun to use them as a reliable prior to alleviate the ill-posedness of SR. This work follows this research line, further exploring the potential of diffusion priors in SR.

The prevailing SR approaches leveraging diffusion priors usually attempt to modify the intermediate features of the diffusion network, either through optimization [7, 23, 56] or fine-tuning [30, 56, 60, 64], to better align them with the given LQ observations. In this work, we propose a new technique based on diffusion inversion to harness diffusion priors. Unlike existing approaches, it attempts to find an optimal noise map as the input of the diffusion model, without any modification to the diffusion network itself, thereby maximizing the utility of diffusion prior.

While considerable advances have been made in generative adversarial networks (GANs) [14] inversion for various applications [62, 75], including SR [4, 15, 41], extending these principles to diffusion models presents unique challenges, particularly for SR tasks that demand high fidelity preservation. In particular, the multi-step stochastic sampling process of diffusion models makes inversion nontrivial. The straightforward inversion approach to optimize the distinct noise maps at each diffusion step is expensive and complex. Additionally, the iterative inference mechanism would accumulate prediction errors and randomness at each step, which can significantly compromise fidelity. Therefore, recent diffusion inversion methods have mainly focused on tasks with lower fidelity requirements, such as image editing [13, 36].

In this work, we reformulate diffusion inversion for the more challenging task of SR. To enable diffusion inversion for SR, we introduce a deep neural network called noise predictor to estimate the noise map from a given LR image. In addition, a Partial noise Prediction (PnP) strategy is devised to construct an intermediate state for the diffusion model, serving as the starting point for sampling. This is made possible by adding noise onto the LR image according to the diffusion model’s forward process, where the noise predictor predicts the added noise instead of random sampling. This approach is driven by the following key motivations:

• Rationality. LR and HR images differ only in highfrequency details. With the addition of appropriate noise,

the LR image becomes indistinguishable from its HR counterpart. Thus, the noisy LR can serve as a proxy for deriving the inversion trajectory during reverse diffusion.

- • Complexity. Rather than predicting noise maps for all diffusion steps, the PnP strategy simplifies the inversion task by limiting predictions to the starting step, thereby reducing the overall complexity of the inversion process.
- • Flexibility. The noise predictor can be trained to predict noise maps for multiple predefined starting steps. During inference, we can freely select a starting step from them and then use any existing sampling algorithm with an arbitrary number of steps, offering favorable flexibility in controlling the sampling process.
- • Fidelity. The starting steps during training are carefully selected to have a high signal-to-noise ratio (SNR), ensuring robust fidelity preservation for SR. In practice, we enforce an SNR threshold greater than 1.44, corresponding to the timestep of 250 in Stable Diffusion [43].
- • Efficiency. As the sampling process begins from a step earlier than 250 (SNR larger than 1.44), the PnP strategy effectively reduces the number of sampling steps to fewer than five when combined with off-the-shelf accelerated sampling algorithms [22, 46]. This addresses the common inefficiency issue in diffusion-based SR approaches.

Unlike most existing diffusion-based methods that rely on fixed sampling steps, our flexible sampling mechanism offers a versatile solution for handling varying degrees of degradation in SR. In SR, it is common to encounter different types and intensities of corruption. Intuitively, the number of sampling steps should adapt to the specific degradation conditions. For example, as shown in Fig. 1, multi-step sampling is preferable to single-step sampling in the first case, as it effectively reduces blurriness and restores finer details. In contrast, for the second example with severe noise, a single sampling step achieves satisfactory results, while additional steps may amplify the noise and introduce unwanted artifacts. Our method uniquely allows users to adjust sampling to suit different degradation types.

The main contributions of this work are twofold. First, we propose a novel SR approach based on diffusion inversion, which effectively leverages the diffusion prior by integrating an auxiliary noise predictor while keeping the entire diffusion backbone fixed. Second, our method introduces a flexible and efficient sampling mechanism that allows for arbitrary sampling steps, ranging from one to five. Remarkably, even when the steps are reduced to just one, our approach still achieves superior or comparable performance to recent dedicated one-step diffusion methods.

### 2. Related Work

Diffusion Prior for SR. Existing diffusion prior-based SR approaches can be broadly categorized into two classes. The first class of methods involves re-optimizing the in-

termediate results of the diffusion model to ensure consistency with the given LR images via pre-defined or estimated degradation models. Representative works include DDRM [23], CCDF [7], and DDNM [56], among others [6, 8, 11, 38, 47, 63, 67]. While effective, these methods are limited by their computational complexity, as they require solving an optimization problem at each diffusion step, leading to slow inference. Furthermore, they often rely on manually defined assumed degradation models and thus cannot handle the blind SR problem in real-world scenarios. The second class directly fine-tunes a pre-trained large T2I model for the SR task. StableSR [53] pioneers this paradigm by incorporating spatial feature transform layers [54] to guide the T2I model toward generating HR outputs. Subsequent works follow by proposing various fine-tuning strategies to exploit diffusion priors, including DiffBIR [30], SeeSR [60], PASD [64], S3Diff [71], and so on [27, 40, 49, 59, 61, 66]. These methods have achieved impressive performance, validating the effectiveness of diffusion priors for SR.

Diffusion Inversion. Diffusion inversion focuses on determining the optimal noise map set that, when processed through the diffusion model, reconstructs a given image. DDIM [46] first addressed this by generalizing the diffusion model via a class of non-Markovian processes, thereby establishing a deterministic generation process. Subsequent approaches, such as those by Rinon et al. [12] and Mokady et al. [36], proposed optimizing the text embedding to better align with the desired textual guidance. Recent efforts have further refined the optimization strategies for both the textual and visual prompts [35, 39], as well as for intermediate noise maps [13, 19, 20, 33, 51, 72], leading to notable enhancements in inversion quality. Despite these advances, existing methods mainly focus on image editing and cannot meet the high-fidelity requirements of SR.

In this work, we tailor the diffusion inversion technique for SR. While Chihaoui et al. [5] have recently explored diffusion inversion for image restoration, their method relies on solving an optimization problem at each inversion step, significantly limiting its inference efficiency. In contrast, our approach introduces a noise prediction module that, once trained, enables efficient inversion without requiring iterative optimization during inference. This leads to substantial improvements in both the efficiency and practicality of diffusion inversion for SR tasks.

### 3. Methodology

In this section, we present the proposed diffusion inversion technique for SR. To maintain consistency with the notations used in diffusion models, we denote the LR image as y0 and the corresponding HR image as x0.

#### 3.1. Motivation

The diffusion model [16, 45] was first introduced as a probabilistic generative model inspired by nonequilibrium thermodynamics. Subsequently, Song et al. [48] reformulated it within the framework of stochastic differential equations (SDEs). In this paper, we propose a general diffusion inversion technique that is applicable to both the probabilistic and SDE-based diffusion formulations. To facilitate understanding, we employ the probabilistic framework of the Denoising Diffusion Probabilistic Model (DDPM) [16] throughout our presentation.

The DDPM framework [16] is indeed a Markov chain of length T, where the forward process is characterized by a Gaussian transition kernel:

q(xt|xt−1) = N(xt; 1 − βtxt−1,βtI), (1)

where βt is a pre-defined hyper-parameter controlling variance schedule. Notably, this transition kernel allows the derivation of the marginal distribution q(xt|x0), i.e.,

q(xt|x0) = N(xt;√α¯tx0,(1 − α¯t)I), (2)

where α¯t = ts=1 αs, αs = 1 − βs. The reverse process aims to generate a high-quality image from an initial ran-

dom noise map xT ∼ N(0,I), which can be expressed as:

xt−1 = gθ(xt,t) + σtzt−1, t = T,··· ,1, (3) where

1 − αt √1 − α¯t

1 √αt

ϵθ(xt,t) , (4)

xt −

gθ(xt,t) =

ϵθ(xt,t) is a pre-trained denoising network parameterized by θ. The noise term zt satisfies z0 = 0 and zt ∼ N(0,I) for t = 1,··· ,T − 1.

Equation (3) indicates that the synthesized image x0 is fully determined by the set of noise maps M = xT ∪ {zt}tT=1−1. In the context of SR, our goal is to generate an HR image x0 conditioned on an LR image y0. To this end, we propose diffusion inversion to find an optimal set of noise maps M∗ that reconstruct the target HR image x0 via the reverse process of Eq. (3). In the following sections, we detail how to achieve this goal by training a noise predictor.

#### 3.2. Diffusion Inversion

To achieve diffusion inversion, we introduce a noise prediction network with parameter w, denoted as fw, which takes the LR image y0 and the timestep t as input and outputs the desired noise maps M∗. Unlike the strategy [5] of directly optimizing M∗ for each testing image, we train such a noise predictor to enable fast sampling during inference, thereby significantly improving the inference efficiency. To ensure the output of fw conforms to Gaussian distribution, we adopt the reparameterization trick of VAE [25], which predicts the mean and variance parameters of Gaussian distribution rather than directly estimating the noise map.

##### 3.2.1. Problem Simplification

Training this noise predictor is inherently challenging. The noise map set M consists of T noise maps (typically T = 1000 in most current diffusion models), corresponding to each step of the diffusion process. Naturally, it is non-trivial to simultaneously estimate such a large number of noise maps using a single, compact network. What’s worse, the iterative sampling paradigm of diffusion models can gradually accumulate prediction errors, which may adversely affect the final SR performance.

To address these challenges, we design a Partial Noise Prediction (PnP) strategy. Specifically, let’s consider diffusion inversion in the context of SR, where the observed LR image y0 only slightly deviates from the target HR image x0 in most cases, primarily in high-frequency components. This observation inspires us to initiate the sampling process from an intermediate timestep N (N < T), effectively reducing the number of noise maps in M from T to N, i.e., M = {zt}Nt=1. Furthermore, given the high-fidelity requirements of SR, we constrain xN to have a relatively high SNR, implying mild noise corruption. This constraint encourages the selection of a smaller N, and in practice, we set N ≤ 250, corresponding to an SNR threshold of 1.44 in the widely used Stable Diffusion [43].

In addition, we further compress the set of the noise maps M = {zt}Nt=1 by integrating existing diffusion acceleration algorithms [22, 46]. The common idea of these algorithms is to skip certain steps during inference, which are selected based on specific rules [29], e.g., “linspace” and “trailing”. Combining with this skipping strategy, the noise map set is simplified as follows:

i}Mi=1, (5)

M = {zκ

where {κ1,··· ,κM} ⊆ {1,··· ,N}. In practice, we set M ≤ 5, thus largely reducing the prediction burden on the noise predictor and improving the sampling efficiency.

##### 3.2.2. Inversion Trajectory

i}Mi=1 and the noise prediction network fw, our goal is to restore the HR image x0 from a given LR observation y0, following an inversion trajectory defined by:

Given the set of noise maps M = {zκ

fw(y0,κi−1), (6)

##### xκ

= gθ(xκ

,κi) + σκ

i−1

i

i

where κ0 = 0, and gθ(·,·) is defined in Eq. (4). The key to initiating this inversion trajectory is constructing the starting state xκ

from the LR image y0. The marginal distribution q(xκ

M

M|x0), as defined in Eq. (2), suggests to achieve xκ

as follows:

M

= α¯κ

##### xκ

M

M

ξ, ξ ∼ N(0,I). (7)

x0 + 1 − α¯κ

M

In the context of SR, since the HR image x0 is not accessible during testing, we thus construct an analogous formulation for xκ

directly from the LR image y0 using the noise predictor fw(·), namely

M

fw(y0,κM). (8)

y0 + 1 − α¯κ

= α¯κ

##### xκ

M

M

M

This design is inspired by the observation that the LR image y0 and the HR image x0 become increasingly indistinguishable when perturbed by Gaussian noise with an appropriate magnitude. Therefore, we aim to seek an optimal noise map fw(y0,κM) to perturb y0 in such a way that the pre-trained diffusion model can generate the corresponding x0 from xκ

that is defined in Eq. (8).

M

To summarize, we establish an inversion trajectory by combining Eqs. (6) and (8), which can be used to solve the SR problem via iterative generation along this trajectory.

##### 3.2.3. Model Training

Given a pre-trained large-scale diffusion model ϵθ(·), an estimation of the HR image x0 can be obtained from xκ

by taking a reverse diffusion step:

i

xˆ0←κ

i

1 √α¯κ

=

i

,κi) , (9)

i − 1 − α¯κ

xκ

ϵθ(xκ

i

i

where xκ

is defined by Eq. (8) for i = M and Eq. (6) for i < M. It is thus possible to train the noise predictor fw(·) by minimizing the distance between xˆ0←κ

i

and x0.

i

However, directly training with this objective is computationally impractical. Specifically, as shown in Eq. (6), calculating xκ

(i < M) necessitates recurrent application of the large-scale diffusion model ϵθ, which leads to prohibitive GPU memory usage. To circumvent this, we adopt an alternative version for xκ

i

based on the marginal distribution in Eq. (2), i.e.,

i

fw(y0,κi), i < M. (10)

x0 + 1 − α¯κ

= α¯κ

##### xκ

i

i

i

This modification also aligns better with the training process of the employed diffusion model, allowing for more effective leveraging of the prior knowledge embedded in it. We now detail the training procedure step by step:

Gaussian Constraint. The pre-trained diffusion model is a powerful denoiser tailored for Gaussian noise with zero mean and varying variances. Hence, it is reasonable to enforce the predicted noise map by fw to obey a Gaussian distribution. For the initial state xκ

, it is observed that the predicted noise map fw(y0,κM) exhibits a mean shift, which is evident when comparing Eqs. (7) and (8), due to the substitution of y0 for x0. Moreover, the visualization presented in Figs. 2 and 3 further validates this observation, illustrating that the predicted noise map is clearly correlated with the LR image. Therefore, we do not consider the Gaussian constraint for xκ

M

.

M

LR Image Noise Map

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

Figure 2. Inference flow of our proposed method, wherein {τi}Si=1 denotes the inversion timesteps. Note that the predicted noise map

zτS exhibits an obvious correlation with the LR image, indicating the non-zero mean property of its statistical distribution.

as defined in Eq. (10), the predicted noise map fw(yκ

Conversely, for the intermediate state xκ

i

,κi) should be enforced to follow N(0,I). This naturally raises an interesting question: Is it necessary to predict the noise map using fw instead of random sampling?

i

First, the proposed PnP strategy requires the timestep κi to satisfy a high SNR constraint, indicating that xκ

is corrupted by only mild Gaussian noise. Second, the pretrained large-scale diffusion model, specifically designed for Gaussian denoising, performs robustly, especially for timesteps κi with low noise levels. Thus, introducing an extra noise predictor model fw for the intermediate timesteps, even conditioned on the additional LR observation, does not yield significant performance gains. This is also empirically illustrated by an ablation study provided in supp. Third, predicting the noise map both for the initial and intermediate state would increase the prediction burden on fw and make the training process more challenging. Considering these reasons together, we discard the noise prediction for the intermediate states, which further reduces the noise map set to M = {zκ

i

M}, leading to a more elegant diffusion inversion technique, as detailed in Alg. 1.

Arbitrary-step Inversion. As analyzed above, noise map prediction is only required for the starting state, as defined in Eq.(8), to initialize the reverse sampling process. To further enhance the flexibility of the sampling process, the noise predictor is trained to estimate the noise maps for multiple pre-selected steps via time embedding. Once trained, the starting timestep can be freely chosen during inference, which results in a fidelity-realism trade-off, as analyzed in Sec. 4.2. Note that the total number of sampling steps is determined by both the selected starting timestep and the skipping stride of the accelerated sampling algorithm for diffusion models. For clarity, a detailed inference procedure is provided in Fig. 2 and Alg. 1. Given practical efficiency considerations, we focused on the number of sampling steps

Algorithm 1 Inference Require: LR image y0, noise predictor fw, pre-trained

diffusion model ϵθ, inversion timesteps {τi}Si=1 ⊂ {κi}Mi=1

- 1: xτ

S

= √α¯τ

S

y0 + √1 − α¯τ

S

fw(y0,τS)

- 2: for i = S,··· ,1 do
- 3: zτ

i ∼ N(0,I) if τi > 1 else zτ

i

= 0

- 4: xτ

i−1

= gθ(xτ

i

,τi) + στ

i

zτ

i

, where gθ is defined in Eq. (4)

- 5: end for
- 6: return x0

ranging from one to five in this study.

Loss Function. To train the noise predictor, we adopt an L2 loss L2, a LPIPS [74] loss Ll, and a GAN [14] loss Lg following recent SR approaches [4, 55]. Let’s denote the set of pre-selected starting timesteps for training as S ⊆ {κ1,··· ,κM}, the overall loss function is defined as:

L2(xˆ0←t, x0) + λlLl(xˆ0←t, x0) + λgLg(xˆ0←t, x0), (11)

t∈S

where xˆ0←t is defined in Eq. (9), λl and λg are hyperparameters. The adversarial loss Lg is implemented using a hinge loss, with a discriminator architecture based on diffusion UNet [16] enhanced with a multi-in, multi-out strategy [65]. For the base model ϵθ, we use SD-Turbo [44], which operates in the latent space of VQGAN [10]. We thus compute the whole loss in the latent space, significantly reducing the required GPU memory. To facilitate training, the LPIPS loss is also fine-tuned in the latent space.

### 4. Experiments

In this section, we first provide an analysis of the proposed method and then conduct extensive experiments to evaluate its performance on one synthetic and two real-world datasets. Our investigation focuses mainly on the ×4 SR task following previous works [55, 73]. To ease the presentation, we refer to our method as InvSR, standing for Diffusion Inversion-based Super-Resolution.

#### 4.1. Experimental Setup

Training Details. Following the setup of recent works [59, 60], we trained the noise predictor on the LSDIR [28] dataset and a subset of 20k face images from the FFHQ [21] dataset. At each iteration, we randomly cropped an image patch with a resolution of 512 × 512 from the source image and synthesized the LR image using the pipeline of RealESRGAN [55]. The text prompt was fixed as a general description1 in both the training and testing phases. To optimize the network parameters, we adopted the Adam [26]

1“Cinematic, high-contrast, photo-realistic, 8k, ultra HD, meticulous detailing, hyper sharpness, perfect without deformations.”

- Table 1. Quantitative results of InvSR with various numbers of sampling steps ranging from one to five on the ImageNet-Test dataset.

|#Steps|Index of the sampled timesteps<br><br>|Metrics|
|---|---|---|
| | |PSNR↑ SSIM↑ LPIPS↓ NIQE↓ PI↓ CLIPIQA↑ MUSIQ↑|
|5|{250, 200, 150, 100, 50}<br><br>|22.70 0.6412 0.2844 4.8757 3.4744 0.6733 69.8427|

{250, 150, 50} 22.92 0.6478 0.2762 4.7980 3.4002 0.6823 70.4688 {200, 100, 50} 23.41 0.6609 0.2648 4.5089 3.2074 0.6851 70.7024 {150, 100, 50} 23.84 0.6713 0.2575 4.2719 3.0527 0.6823 70.4569

3

{250} 23.84 0.6713 0.2575 4.5287 3.1748 0.7132 72.5773 {200} 24.14 0.6789 0.2517 4.3815 3.0866 0.7093 72.2909 {150} 24.42 0.6851 0.2469 4.2194 2.9979 0.7019 71.7100 {100} 24.66 0.6891 0.2450 4.0606 2.8951 0.6912 70.8251

1

algorithm with default settings of PyTorch [42]. The training process takes over 100k iterations with a batch size of 64 and a fixed learning rate of 5e−5. The hyper-parameters λl and λg in the loss function were set to 2.0 and 0.1, respectively. The architecture of the noise predictor was based on the encoder of VQGAN [10], containing two down-sampling blocks, each equipped with a self-attention layer [50].

In the training stage, we randomly select a starting timestep from S = {250,200,150,100} to train the noise predictor at each iteration. During inference, five inversion steps, i.e., M = {250,200,150,100,50}, are used throughout our experiments.

Testing Datasets. To evaluate the performance of InvSR, we constructed a synthetic dataset named ImageNet-Test, comprising 3,000 images from the validation set of ImageNet [9]. The LR and HR images, with resolutions of 128 × 128 and 512 × 512, respectively, were synthesized using the degradation settings of ResShift [69]. Notably, we selected the HR images from ImageNet rather than the commonly used datasets in SR, such as Set5 [1], Set14 [70], and Urban100 [17], mainly because these datasets only contain very few source images, which fails to thoroughly assess various methods under complicated degradation types.

We further conducted experiments on two real-world datasets to validate the effectiveness of InvSR. The first dataset is RealSR [3], which contains 100 real images captured by Canon 5D3 and Nikon D810 cameras. The second dataset, RealSet80 [69], comprises 80 LR images widely used in existing literature [18, 30–32, 55, 68].

Compared Methods. We evaluate the effectiveness of InvSR in comparison to nine recent methods, including two GAN-based methods, namely BSRGAN [73] and RealESRGAN [55], as well as seven diffusion-based methods, including LDM [43], StableSR [53], DiffBIR [30], SeeSR [60], ResShift [68, 69], SinSR [57], and OSEDiff [59]. For LDM, StableSR, DiffBIR, and SeeSR, we all use 50 sampling steps for fair comparison. In the case of ResShift, SinSR, and OSEDiff, we adhere to the number of sampling steps suggested by their official guidelines.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

(a) Zoomed LR (b) Noise Map (c) InvSR-1

Figure 3. From left to right: (a) zoomed LR image, (b) predicted noise map by our method for the initial timestep, (c) superresolved results by our method with a single sampling step.

Metrics. The performance of various methods was assessed across seven metrics, including three reference metrics, namely PSNR, SSIM [58], LPIPS [74], as well as four nonreference metrics, namely NIQE [34], PI [2], MUSIQ [24], and CLIPIQA [52]. For evaluations on the datasets of ImageNet-Test and RealSR, all seven metrics were adopted to ensure a holistic assessment. For the dataset of RealSet80, however, only non-reference metrics were employed since the ground truth images are not accessible. Notably, PSNR and SSIM are calculated in the luminance (Y) channel of YCbCr space, while other metrics are directly computed in the standard RGB (sRGB) space.

#### 4.2. Model Analysis

Arbitrary-steps Sampling. Recent efficient diffusionbased SR approaches, such as ResShift [69], SinSR [57], and OSEDiff [59], constrain the sampling process to a predefined number of steps, consistent with their training configuration. In contrast, the proposed InvSR supports sampling with an arbitrary number of steps, significantly enhancing flexibility and adaptability to varying degradation types, as demonstrated in Fig. 1 and Fig. 6.

- Table 2. Quantitative comparisons of different methods on ImageNet-Test and RealSR. The number of sampling steps is marked in the format of “Method name-Steps” for diffusion-based methods. The best and second-best results are highlighted in bold and underlined.

|Datasets<br><br>|Methods|Metrics|
|---|---|---|
| | |PSNR↑ SSIM↑ LPIPS↓ NIQE↓ PI↓ CLIPIQA↑ MUSIQ↑ #Params (M)|
|ImageNet-Test<br><br>|BSRGAN [73] RealESRGAN [55]<br><br>|27.05 0.7453 0.2437 4.5345 3.7111 0.5703 67.7195 16.70 26.62 0.7523 0.2303 4.4909 3.7234 0.5090 64.8186 16.70<br><br>|
| |LDM-50 [43] StableSR-50 [53] DiffBIR-50 [30] SeeSR-50 [60]|27.19 0.7285 0.2286 5.2411 4.2554 0.5554 62.8776 113.60<br><br>24.77 0.6908 0.2591 4.5120 3.1473 0.7067 71.2811 152.70<br><br>25.72 0.6695 0.2795 4.5875 3.2260 0.6900 69.7089 385.43<br><br>26.69 0.7422 0.2187 4.3825 3.4742 0.5868 71.2412 751.50<br><br><br>|
| |ResShift-4 [69]|27.33 0.7530 0.1998 5.8700 4.3643 0.6147 65.5860 118.59|
| |SinSR-1 [57] OSEDiff-1 [59] InvSR-1 (Ours)<br><br>|26.98 0.7304 0.2209 5.2623 3.8189 0.6618 67.7593 118.59<br><br>23.95 0.6756 0.2624 4.7157 3.3775 0.6818 70.3928 8.50<br>24.14 0.6789 0.2517 4.3815 3.0866 0.7093 72.2900 33.84<br>|

|RealSR|BSRGAN [73] RealESRGAN [55]<br><br>|26.51 0.7746 0.2685 4.6501 4.4644 0.5439 63.5869 16.70 25.85 0.7734 0.2728 4.6766 4.4881 0.4898 59.6803 16.70<br><br>|
|---|---|---|
| |LDM-50 [43] StableSR-50 [53] DiffBIR-50 [30] SeeSR-50 [60]<br><br>|26.75 0.7711 0.2945 4.8712 5.0025 0.4907 54.3910 113.60 26.27 0.7755 0.2671 5.1745 4.8209 0.5209 60.1758 152.70 24.83 0.6642 0.3864 3.7366 3.3661 0.6857 65.3934 385.43 26.20 0.7555 0.2806 4.5358 4.1464 0.6824 66.3757 751.50<br><br>|
| |ResShift-4 [69]|25.77 0.7453 0.3395 6.9113 5.4013 0.5994 57.5536 118.59|
| |SinSR-1 [57] OSEDiff-1 [59] InvSR-1 (Ours)<br><br>|26.02 0.7097 0.3993 6.2547 4.7183 0.6634 59.2981 118.59<br><br>23.89 0.7030 0.3288 5.3310 4.3584 0.7008 65.4806 8.50<br>24.50 0.7262 0.2872 4.2189 3.7779 0.6918 67.4586 33.84<br><br><br>|

We further provide a comprehensive comparison of InvSR with one, three, and five sampling steps, as summarized in Table. 1. Three key observations can be obtained from these results: i) With a fixed number of sampling steps, e.g., one or three, varying the starting timestep enables a trade-off between fidelity (measured by reference metrics) and realism (measured by non-reference metrics). Specifically, using larger starting timesteps favors improved realism at the expense of fidelity. ii) As expected, reference metrics deteriorate with increased sampling steps due to the introduction of additional randomness. iii) Interestingly, non-reference metrics also exhibit a decline when using more sampling steps. This is mainly because most testing images contain some noise, which can lead to undesired artifacts if multiple sampling steps are used, thus degrading the overall image quality. However, using more sampling steps can effectively recover intricate fine-grained structures in cases involving substantial blur, as evidenced by the first examples in Fig. 1 and Fig. 6.

Initial Noise Prediction Figure 3 presents the noise map predicted by our method for the initial timestep, exhibiting a strong correlation with image content. This visualization aligns well with the theoretical analysis in Sec 3.2.3, empirically validating that our noise predictor can effectively find an LR-dependent noise map to facilitate the SR task.

#### 4.3. Comparison to State of the Arts

Considering that recent studies [57, 59] mainly focus on developing one-step diffusion-based methods, we thus evalu-

Table 3. Quantitative comparisons of various methods on RealSet80. The number of sampling steps is marked in the format of “Method name-Steps” for diffusion-based methods. The best and second-best results are highlighted in bold and underlined.

|Methods<br><br>|Metrics|
|---|---|
| |NIQE↓ PI↓ CLIPIQA↑ MUSIQ↑|
|BSRGAN [73] RealESRGAN [55]|4.4408 4.0276 0.6263 66.6288 4.1568 3.8852 0.6189 64.4957<br><br>|
|LDM-50 [43] StableSR-50 [53] DiffBIR-50 [30] SeeSR-50 [60]|4.3248 4.2545 0.5511 55.8246 4.5593 4.0977 0.6214 62.7613<br><br>3.8630 3.2117 0.7404 67.9806<br><br>4.3678 3.7429 0.7114 69.7658<br><br><br>|
|ResShift-4 [69]|5.9866 4.8318 0.6515 61.7967<br><br>|
|SinSR-1 [57] OSEDiff-1 [59] InvSR-1 (Ours)<br><br>|5.6243 4.2830 0.7228 64.0573 4.3457 3.8219 0.7093 68.8202 4.0284 3.4666 0.7291 69.8055<br><br>|

ate InvSR against these methods under a one-step configuration to ensure a fair comparison.

Synthetic Dataset. Table 2 reports a comprehensive evaluation of various methods on ImageNet-Test dataset, encompassing seven quantitative metrics, with additional qualitative comparisons included in Fig. 8 of supp. Notably, compared to the recent state-of-the-art (SotA) one-step method OSEDiff [59], InvSR demonstrates evident superiority across all the seven metrics. Moreover, even compared to multi-step methods with 50 sampling steps, such as StableSR and DiffBIR, InvSR still achieves comparable performance in distortion-oriented metrics, including

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

LDM-50

DiffBIR-50

BSRGAN

StableSR-50

SeeSR-50

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

ResShift-4

OSEDiff-1

RealESRGAN

SinSR-1

Zoomed LR

InvSR-1 (Ours)

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

DiffBIR-50

BSRGAN

StableSR-50

LDM-50

SeeSR-50

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

OSEDiff-1

Zoomed LR InvSR-1 (Ours)

RealESRGAN

ResShift-4

SinSR-1

Figure 4. Visual results of different methods on two typical real-world examples from RealSet80 dataset. For clear comparisons, the number of sampling steps is annotated in the format “Method name-Steps” for diffusion-based approaches. (Zoom-in for best view)

PSNR and SSIM, while outperforming these methods in perception-oriented metrics, such as LPIPS, NIQE, PI, and MUSIQ. These results indicate that InvSR effectively balances both performance and efficiency, advancing the field of diffusion-based SR approaches. Additionally, InvSR maintains a moderate model size with about 34 million learnable parameters, further enhancing its practicality for real-world applications.

Real-world Dataset. To evaluate real-world datasets, we mainly focus on the non-reference metrics. Table 2 and 3 provide a detailed comparison of InvSR against recent SotA methods on the datasets of RealSR and RealSet80, respectively. It can be easily observed that InvSR achieves superior performance across most non-reference metrics compared to recent one-step methods under fair comparison and second-best results compared to existing multi-step methods. To further substantiate these conclusions, we present visual comparisons of two real-world examples in Fig. 4, and more examples can be found in Fig. 9 of supp. In the first example, where the LR image contains evident compression noise, InvSR successfully removes these artifacts and generates clear results, while other methods struggle with remaining artifacts. In the second example, which is

degraded by noticeable blurriness, InvSR produces sharper image structures, such as the tile edges on the wall. These quantitative and qualitative evaluations highlight the great potential of InvSR to solve the real-world SR task.

### 5. Conclusion

We proposed InvSR, a new SR method based on diffusion inversion. Our method introduces a noise prediction network designed to estimate an optimal noise map, enabling the construction of an intermediate state of a pre-trained diffusion model as the starting sampling point. This design is appealing in two aspects: first, InvSR can sufficiently harness the prior knowledge encapsulated in the pre-trained diffusion model, thereby facilitating SR performance. Second, InvSR offers a flexible sampling strategy capable of starting from various intermediate states of the diffusion model by incorporating a time-dependent architecture of the noise predictor. This flexibility allows users to freely adjust the sampling steps according to the degradation type or their specific requirements. Even after reducing the sampling steps to just one, InvSR still exhibits significant superiority beyond recent one-step diffusion-based methods, suggesting its effectiveness and efficiency.

Acknowledgement. This study is supported under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). The work was completed when Zongsheng Yue was with S-Lab.

### References

- [1] Marco Bevilacqua, Aline Roumy, Christine Guillemot, and Marie Line Alberi-Morel. Low-complexity single-image super-resolution based on nonnegative neighbor embedding.

2012. 6

- [2] Yochai Blau, Roey Mechrez, Radu Timofte, Tomer Michaeli, and Lihi Zelnik-Manor. The 2018 pirm challenge on perceptual image super-resolution. In Eur. Conf. Comput. Vis. Worksh., pages 0–0, 2018. 6
- [3] Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image super-resolution: A new benchmark and a new model. In Int. Conf. Comput. Vis., pages 3086–3095, 2019. 6
- [4] Kelvin CK Chan, Xiangyu Xu, Xintao Wang, Jinwei Gu, and Chen Change Loy. Glean: Generative latent bank for image super-resolution and beyond. IEEE Trans. Pattern Anal. Mach. Intell., 45(3):3154–3168, 2022. 2, 5
- [5] Hamadi Chihaoui, Abdelhak Lemkhenter, and Paolo Favaro. Blind image restoration via fast diffusion inversion. In Adv. Neural Inform. Process. Syst., 2024. 3
- [6] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In Adv. Neural Inform. Process. Syst., pages 25683–25696, 2022. 3
- [7] Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In IEEE Conf. Comput. Vis. Pattern Recog., pages 12413–12422, 2022. 2, 3
- [8] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In Int. Conf. Learn. Represent., 2023. 3
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conf. Comput. Vis. Pattern Recog., pages 248–255, 2009. 6
- [10] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In IEEE Conf. Comput. Vis. Pattern Recog., pages 12873–12883,

2021. 5, 6

- [11] Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative diffusion prior for unified image restoration and enhancement. In IEEE Conf. Comput. Vis. Pattern Recog., pages 9935–9946,

2023. 3

- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3

- [13] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. ReNoise: Real image inversion through iterative noising. In Eur. Conf. Comput. Vis., 2024. 2, 3
- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Adv. Neural Inform. Process. Syst., 27, 2014. 2, 5, 12
- [15] Jinjin Gu, Yujun Shen, and Bolei Zhou. Image processing using multi-code gan prior. In IEEE Conf. Comput. Vis. Pattern Recog., pages 3012–3021, 2020. 2
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Adv. Neural Inform. Process. Syst., pages 6840–6851, 2020. 2, 3, 5
- [17] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5197–5206,

2015. 6

- [18] Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey, and Luc Van Gool. Dslr-quality photos on mobile devices with deep convolutional networks. In Int. Conf. Comput. Vis., 2017. 6
- [19] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In Int. Conf. Learn. Represent., 2024. 3
- [20] Wonjun Kang, Kevin Galim, and Hyung Il Koo. Eta inversion: Designing an optimal eta function for diffusion-based real image editing. In Eur. Conf. Comput. Vis., 2024. 3
- [21] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4401–4410,

2019. 5

- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Adv. Neural Inform. Process. Syst., pages 26565– 26577, 2022. 2, 4
- [23] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In Adv. Neural Inform. Process. Syst., pages 23593–23606, 2022. 2, 3
- [24] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Int. Conf. Comput. Vis., pages 5148–5157, 2021. 6
- [25] Diederik P Kingma. Auto-encoding variational bayes. In Int. Conf. Learn. Represent., 2014. 3
- [26] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Int. Conf. Learn. Represent.,

2015. 5

- [27] Jianze Li, Jiezhang Cao, Zichen Zou, Xiongfei Su, Xin Yuan, Yulun Zhang, Yong Guo, and Xiaokang Yang. Distillation-free one-step diffusion for real-world image super-resolution. arXiv preprint arXiv:2410.04224, 2024. 3
- [28] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, Rakesh Ranjan, Radu Timofte, and Luc Van Gool. Lsdir: A large scale dataset for image restoration. In IEEE Conf. Comput. Vis. Pattern Recog. Worksh., pages 1775– 1787, 2023. 5

- [29] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF Winter Conf. on Applications of Comput. Vision, pages 5404–5411, 2024. 4
- [30] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Ben Fei, Bo Dai, Wanli Ouyang, Yu Qiao, and Chao Dong. DiffBIR: Towards blind image restoration with generative diffusion prior. arXiv preprint arXiv:2308.15070, 2023. 2, 3, 6, 7
- [31] David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Int. Conf. Comput. Vis., pages 416–423, 2001.
- [32] Yusuke Matsui, Kota Ito, Yuji Aramaki, Azuma Fujimoto, Toru Ogawa, Toshihiko Yamasaki, and Kiyoharu Aizawa. Sketch-based manga retrieval using manga109 dataset. Multimedia Tools and Applications, 76:21811–21838, 2017. 6
- [33] Barak Meiri, Dvir Samuel, Nir Darshan, Gal Chechik, Shai Avidan, and Rami Ben-Ari. Fixed-point inversion for text-toimage diffusion models. arXiv preprint arXiv:2312.12540,

2023. 3

- [34] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012. 6
- [35] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023. 3
- [36] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 6038–6047, 2023. 2, 3
- [37] Chong Mou, Yanze Wu, Xintao Wang, Chao Dong, Jian Zhang, and Ying Shan. Metric learning based interactive modulation for real-world super-resolution. In Eur. Conf. Comput. Vis., pages 723–740. Springer, 2022. 12
- [38] Naoki Murata, Koichi Saito, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration. In Int. Conf. Mach. Learn., pages 25501–25522. PMLR,

2023. 3

- [39] Thao Nguyen, Yuheng Li, Utkarsh Ojha, and Yong Jae Lee. Visual instruction inversion: Image editing via image prompting. In Adv. Neural Inform. Process. Syst., 2024. 3
- [40] Mehdi Noroozi, Isma Hadji, Brais Martinez, Adrian Bulat, and Georgios Tzimiropoulos. You only need one step: Fast super-resolution with stable diffusion via scale distillation. In Eur. Conf. Comput. Vis., 2024. 3
- [41] Xingang Pan, Xiaohang Zhan, Bo Dai, Dahua Lin, Chen Change Loy, and Ping Luo. Exploiting deep generative prior for versatile image restoration and manipulation. IEEE Trans. Pattern Anal. Mach. Intell., 44(11):7474–7489,

2021. 2

- [42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An

- imperative style, high-performance deep learning library. In Adv. Neural Inform. Process. Syst., 2019. 6
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., pages 10684–10695, 2022. 2, 4, 6, 7, 12
- [44] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In Eur. Conf. Comput. Vis., pages 87–103. Springer, 2024. 5
- [45] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Int. Conf. Mach. Learn., pages 2256–2265. PMLR, 2015. 2, 3
- [46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Int. Conf. Learn. Represent., 2021. 2, 3, 4
- [47] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In Int. Conf. Learn. Represent., 2023. 3
- [48] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In Int. Conf. Learn. Represent., 2021. 2, 3
- [49] Haoze Sun, Wenbo Li, Jianzhuang Liu, Haoyu Chen, Renjing Pei, Xueyi Zou, Youliang Yan, and Yujiu Yang. Coser: Bridging image and language for cognitive super-resolution. In IEEE Conf. Comput. Vis. Pattern Recog., pages 25868– 25878, 2024. 3
- [50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Adv. Neural Inform. Process. Syst., 2017. 6
- [51] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In IEEE Conf. Comput. Vis. Pattern Recog., pages 22532–22541,

2023. 3

- [52] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 6
- [53] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C.K. Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. Int. J. Comput. Vis., pages 1–21, 2024. 3, 6, 7
- [54] Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Recovering realistic texture in image super-resolution by deep spatial feature transform. In IEEE Conf. Comput. Vis. Pattern Recog., 2018. 3
- [55] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Int. Conf. Comput. Vis. Worksh., pages 1905–1914, 2021. 5, 6, 7
- [56] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. In Int. Conf. Learn. Represent., 2023. 2, 3
- [57] Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C

- Kot, and Bihan Wen. Sinsr: diffusion-based image superresolution in a single step. In IEEE Conf. Comput. Vis. Pattern Recog., pages 25796–25805, 2024. 6, 7
- [58] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 13(4):600–612,

2004. 6

- [59] Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for real-world image super-resolution. In Adv. Neural Inform. Process. Syst.,

2024. 3, 5, 6, 7, 13

- [60] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. SeeSR: Towards semantics-aware real-world image super-resolution. In IEEE Conf. Comput. Vis. Pattern Recog., pages 25456–25467, 2024. 2, 3, 5, 6, 7
- [61] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool. Diffir: Efficient diffusion model for image restoration. In Int. Conf. Comput. Vis., pages 13095–13105, 2023. 3
- [62] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. Gan inversion: A survey. IEEE Trans. Pattern Anal. Mach. Intell., 45(3):3121–3138,

2022. 2

- [63] Jie Xiao, Ruili Feng, Han Zhang, Zhiheng Liu, Zhantao Yang, Yurui Zhu, Xueyang Fu, Kai Zhu, Yu Liu, and ZhengJun Zha. DreamClean: Restoring clean image using deep diffusion prior. In Int. Conf. Learn. Represent., 2024. 3
- [64] Tao Yang, Rongyuan Wu, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixel-aware stable diffusion for realistic image super-resolution and personalized stylization. In Eur. Conf. Comput. Vis., 2024. 2, 3
- [65] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In Adv. Neural Inform. Process. Syst., 2024. 5
- [66] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild. In IEEE Conf. Comput. Vis. Pattern Recog., pages 25669–25680, 2024. 3
- [67] Zongsheng Yue and Chen Change Loy. DifFace: Blind face restoration with diffused error contraction. IEEE Trans. Pattern Anal. Mach. Intell., 2024. 3
- [68] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image superresolution by residual shifting. In Adv. Neural Inform. Process. Syst., pages 13294–13307, 2023. 6
- [69] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Efficient diffusion model for image restoration by residual shifting. IEEE Trans. Pattern Anal. Mach. Intell., 2024. 6, 7
- [70] Roman Zeyde, Michael Elad, and Matan Protter. On single image scale-up using sparse-representations. In Int. Conf. on Curves and Surfaces, pages 711–730. Springer, 2012. 6
- [71] Aiping Zhang, Zongsheng Yue, Renjing Pei, Wenqi Ren, and Xiaochun Cao. Degradation-guided one-step image super-resolution with diffusion priors. arXiv preprint arXiv:2409.17058, 2024. 3

- [72] Guoqiang Zhang, Jonathan P Lewis, and W Bastiaan Kleijn. Exact diffusion inversion via bidirectional integration approximation. In Eur. Conf. Comput. Vis., pages 19–36, 2024. 3
- [73] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Int. Conf. Comput. Vis., pages 4791–4800, 2021. 5, 6, 7
- [74] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conf. Comput. Vis. Pattern Recog., pages 586–595, 2018. 5, 6, 12
- [75] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. Indomain gan inversion for real image editing. In Eur. Conf. Comput. Vis., pages 592–608. Springer, 2020. 2

## Arbitrary-steps Image Super-resolution via Diffusion Inversion Supplementary Material

This supplemental material mainly contains:

- • Sec. A discusses the selection of the number of sampling steps.
- • Performance comparison of InvSR with various base diffusion models in Sec. B.1.
- • Ablation studies on the intermediate noise prediction model in Sec. B.2.
- • Ablation studies on the loss function in Sec. B.3.
- • Discussions on the efficiency and limitation in Sec. B.4.
- • Visual comparisons on ImageNet-Test dataset in Fig. 8.
- • More visual comparisons on real-world examples in Fig. 9.

### A. Discussion on Sampling Steps

The proposed method, named InvSR, enables a flexible sampling mechanism that allows an arbitrary number of sampling steps. This naturally raises an interesting question: how do we determine an appropriate number of sampling steps for general image super-resolution (SR) tasks? We answer this question from two aspects.

First, as shown in Tables 2 and 3, and Fig. 3 of the main text, InvSR achieves promising results with only a single sampling step, evidently outperforming recent state-of-theart (SotA) one-step methods. Therefore, we recommend setting the sampling steps to one for most real-world applications, effectively balancing efficiency and performance.

Second, we can also adjust sampling steps according to the type of image degradation. Generally, image degradations can be categorized into two main classes: blurriness and noise. As illustrated in Fig. 1 and Fig. 6, multi-step sampling would incorrectly amplify noise, leading to undesirable artifacts for images with heavy noise. In contrast, for images primarily suffering from blurriness, multi-step sampling proves beneficial, as it generates more detailed and realistic image structures. In practice, we could first estimate the noise level using some off-the-shelf degradation estimation models, such as Mou et al. [37]. Based on the estimated noise level, one can determine whether a onestep or multi-step sampling is more appropriate. In cases where multi-step sampling is favored, the number of sampling steps can be freely adjusted to achieve a satisfactory result.

### B. Experiments B.1. Base Diffusion Model

For the pre-trained diffusion models used in InvSR, we considered two prevailing variants of Stable Diffusion [43],

[Figure 63]

[Figure 64]

[Figure 65]

Zoomed LR InvSR with SD-2.0 InvSR with SD-Turbo

Figure 5. A typical visual comparison of the proposed InvSR based on different diffusion models: SD-2.0 and SD-Turbo. Note that these results are achieved with five sampling steps.

namely SD-2.02 and SD-Turbo3. Table 4 provides a quantitative comparison of InvSR equipped with these two base models. When reducing the sampling steps to one, InvSR demonstrated similar performance with both SD-2.0 and SD-Turbo. However, in the multi-step sampling scenarios, the model based on SD-Turbo exhibited more stable performance, particularly in terms of reference metrics. Furthermore, a visual comparison under five sampling steps, as illustrated in Fig. 5, reveals that the SD-2.0-based model produced noticeable artifacts, aligning with the quantitative results. We thus employed SD-Turbo as our base model throughout this study.

#### B.2. Intermediate Noise Prediction

In our proposed diffusion inversion framework, we opt to sample the noise maps randomly rather than employing a noise prediction model for intermediate timesteps. This choice is motivated by the high SNR (signal-to-noise ratio) constraint imposed on the inversion timesteps, as elaborated in Sec. 3.2.3 of the main text. To further validate this choice, we introduced an additional baseline, denoted as “InvSR-Int”, which integrates an extra noise predictor specifically trained for intermediate timesteps. Table 5 reports a detailed comparison between InvSR and InvSR-Int. It can be observed that the performance differences between these two models are negligible. Therefore, we omit the intermediate noise prediction in InvSR, further simplifying the overall framework.

#### B.3. Loss Functions

In addition to the commonly used L2 loss, we incorporate LPIPS [74] loss and GAN [14] loss to train our noise predictor, as formulated in Eq. (11). The hyper-parameters of λl and λg are introduced to control the importance of the LPIPS and GAN losses, respectively. Table 6 provides

- 2https://huggingface.co/stabilityai/stable-diffusion-2-base
- 3https://huggingface.co/stabilityai/sd-turbo

- Table 4. Quantitative comparisons of the proposed InvSR equipped with two different based models, namely SD-2.0 and SD-Turbo, on the dataset of ImageNet-Test.

|Base models|#Steps<br><br>|Index of the sampled timesteps<br><br>|Metrics|
|---|---|---|---|
| | | |PSNR↑ SSIM↑ LPIPS↓ NIQE↓ PI↓ CLIPIQA↑ MUSIQ↑|
|SD-Turbo SD-2.0<br><br>|5|{250, 200, 150, 100, 50}<br><br>|22.70 0.6412 0.2844 4.8757 3.4744 0.6733 69.8427 21.40 0.6063 0.3274 5.1508 3.8709 0.6467 67.6056|

SD-Turbo

3 {150, 100, 50}

23.84 0.6713 0.2575 4.2719 3.0527 0.6823 70.4569 SD-2.0 23.13 0.6566 0.2776 4.2449 3.1467 0.6722 69.5178

SD-Turbo

1 {200}

24.14 0.6789 0.2517 4.3815 3.0866 0.7093 72.2909 SD-2.0 23.36 0.6637 0.2647 4.3304 3.1545 0.6969 71.4974

- Table 5. Quantitative comparisons of InvSR to the baseline method InvSR-Int that combines an additional noise predictor for the intermediate timesteps on the dataset of ImageNet-Test.

|Methods<br><br>|#Steps<br><br>|Index of the sampled timesteps<br><br>|Metrics|
|---|---|---|---|
| | | |PSNR↑ SSIM↑ LPIPS↓ NIQE↓ PI↓ CLIPIQA↑ MUSIQ↑|
|InvSR InvSR-Int<br><br>|5<br><br>|{250, 200, 150, 100, 50}|22.70 0.6412 0.2844 4.8757 3.4744 0.6733 69.8427 22.70 0.6412 0.2844 4.8785 3.4718 0.6734 69.8466|

- Table 6. Quantitative ablation studies on the loss function in Eq. (11), wherein the hyper-parameters λl and λg control the weight importance of the LPIPS loss and the GAN loss, respectively.

|Methods<br><br>|Hyper-parameters<br><br>| |Metrics|
|---|---|---|---|
| |λl (LPIPS loss)|λg (GAN loss)|PSNR↑ SSIM↑ LPIPS↓ NIQE↓ PI↓ CLIPIQA↑ MUSIQ↑|
|Baseline1<br>Baseline2<br>Baseline3 InvSR-1<br>|0.0 2.0 0.0 2.0<br><br>|0.0 0.0 0.1 0.1<br><br>|26.71 0.7365 0.2850 9.2792 6.4147 0.6168 64.6069 26.24 0.7274 0.2841 8.4367 5.7973 0.6501 66.1726 24.11 0.6809 0.2599 4.4518 3.1229 0.7078 72.5045 24.14 0.6789 0.2517 4.3815 3.0866 0.7093 72.2909|

- Table 7. Efficiency comparisons of different methods on the x4 (128 → 512) SR task, where the runtime results are tested on an NVIDIA A100 GPU with 40GB memory. For diffusion-based SR approaches, the number of sampling steps is annotated in the format of “Method name-Steps”.

|Metrics|Methods|
|---|---|
| |BSRGAN RealESRGAN StableSR-50 DiffBIR-50 SeeSR-50 ResShift-4 SinSR-1 OSEDiff-1 InvSR-1<br><br>|
|#Params (M) Runtime (ms)<br><br>|16.70 16.70 152.70 385.43 751.50 118.59 118.59 8.50 33.84 65 65 3459 7937 6438 319 138 176 117|

a quantitative comparison of various baseline models under different loss configurations, and Fig. 7 demonstrates a typical visual example. We can observed that Baseline1 trained solely with the L2-based diffusion loss produces over-smooth outputs, which is consistent with its superior PSNR scores. Incorporating the GAN loss enhances the generation of finer image details but may introduce undesirable artifacts. The addition of the LPIPS loss can mitigate these artifacts to a certain extent, striking a balance between perceptual quality and artifact suppression. Therefore, this study employs both LPIPS and GAN losses to achieve optimal performance.

#### B.4. Efficiency and Limitation

Table 7 lists an efficiency comparison of various methods on the x4 (128 → 512) SR task. It can be observed that the proposed InvSR demonstrates advantages in runtime among one-step diffusion-based approaches. Despite

having a larger number of parameters compared to the recent SotA method OSEDiff [59], InvSR achieves a 50% reduction in inference time. This is mainly because OSEDiff relies on an additional image captioning model, whereas InvSR does not. However, it is noteworthy that InvSR still lags behind GAN-based methods in efficiency due to its reliance on the large-scale Stable Diffusion model. To address the high-efficiency demand in real-world applications, future work will explore model quantization techniques to further accelerate the inference speed.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Zoomed LR InvSR-1 InvSR-3 InvSR-5

- Figure 6. Qualitative comparisons of the proposed InvSR with different sampling steps, where the number of sampling steps is annotated in the format “InvSR-Steps”. In the first example, mainly degraded by blurriness, multi-step sampling is preferable to single-step sampling as it progressively recovers finer details. Conversely, in the second example with severe noise, a single sampling step is sufficient to achieve satisfactory results, whereas additional steps may amplify the noise and introduce unwanted artifacts. (Zoom-in for best view)

(a) Zoomed LR (b) w/o LPIPS, w/o GAN (c) w/o GAN (d) w/o LPIPS (e) InvSR-1

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

- Figure 7. Visual comparisons of the proposed method with various loss configurations. (a) Zoomed LR image, (b) Baseline1 with λl = 0 and λg = 0, (c) Baseline2 with λl = 2.0 and λg = 0, (d) Baseline3 with λl = 0 and λg = 0.1, (e) recommended settings of λl = 2.0 and λg = 0.1. (Zoom-in for best view)

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Zoomed LR

LDM-50

DiffBIR-50

BSRGAN

StableSR-50

SeeSR-50

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

ResShift-4

OSEDiff-1

RealESRGAN

SinSR-1

Ground Truth

InvSR-1 (Ours)

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Zoomed LR

LDM-50

DiffBIR-50

BSRGAN

StableSR-50

SeeSR-50

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Ground Truth

ResShift-4

OSEDiff-1

RealESRGAN

SinSR-1

InvSR-1 (Ours)

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Zoomed LR

LDM-50

DiffBIR-50

BSRGAN

StableSR-50

SeeSR-50

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Ground Truth

OSEDiff-1

RealESRGAN

ResShift-4

SinSR-1

InvSR-1 (Ours)

- Figure 8. Visual comparisons of various methods on three typical examples from ImageNet-Test. For diffusion-based methods, the number of sampling steps is annotated in the format of “Method name-Steps”. (Zoom-in for best view)

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

LDM-50

StableSR-50

SeeSR-50

DiffBIR-50

BSRGAN

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

ResShift-4

SinSR-1

OSEDiff-1

Zoomed LR

RealESRGAN

InvSR-1 (Ours)

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

LDM-50

StableSR-50

SeeSR-50

DiffBIR-50

BSRGAN

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

ResShift-4

SinSR-1

Zoomed LR

OSEDiff-1

RealESRGAN

InvSR-1 (Ours)

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

LDM-50

StableSR-50

SeeSR-50

DiffBIR-50

BSRGAN

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

ResShift-4

SinSR-1

OSEDiff-1

Zoomed LR

RealESRGAN

InvSR-1 (Ours)

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

LDM-50

DiffBIR-50

BSRGAN

StableSR-50

SeeSR-50

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Zoomed LR

OSEDiff-1

RealESRGAN

ResShift-4

SinSR-1

InvSR-1 (Ours)

- Figure 9. Visual comparisons of various methods on four real-world examples from RealSet80. For diffusion-based methods, the number of sampling steps is annotated in the format of “Method name-Steps”. (Zoom-in for best view)

