## Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation

# arXiv:2606.01048v1[cs.CV]31May2026

Ziyue Lin1*, Jiahe Hou2*, Hongyu Xia1*, Xinrui Xie3, Feifei Wang1, Yuyin Zhou4, Wei Wang2 Jiawei Liu2†, Liangqiong Qu1† 1The University of Hong Kong 2Shenyang Institute of Automation, Chinese Academy of Sciences 3The Chinese University of Hong Kong 4University of California, Santa Cruz {ziyue lin,xiahyu}@connect.hku.hk, liujiawei@sia.cn, liangqqu@hku.hk

#### Abstract

We propose Decoupled Residual Denoising Diffusion models (DRDD) for unified and data-efficient image-to-image (I2I) translation. While diffusion models have advanced I2I translation in terms of quality and diversity, we uncover a previously under-explored property in diffusion models. Crucially, beyond its conventional role of manifold lifting (i.e., moving data off low-dimensional manifolds), injecting Gaussian noise facilitates domain harmonization by implicitly aligning feature distributions across domains, a property particularly advantageous for unified I2I translation. However, existing diffusion models prematurely erode this harmonization effect, as noise and residuals are simultaneously removed in a single coupled diffusion process. To address this, DRDD decouples the diffusion process into two sequential and independent diffusion stages: (1) a stochastic noise diffusion for domain harmonization and manifold lifting, and (2) a deterministic residual diffusion that learns the core semantic mapping entirely within the fixed-noise domain. This decoupling preserves harmonization and manifold lifting effects throughout the transformation, substantially simplifying the learning of unified mappings across diverse tasks and domains. Notably, the noise diffusion stage is trained exclusively on abundant, unpaired targetdomain images, greatly improving data efficiency. Comprehensive theoretical and empirical analysis demonstrates that DRDD is broadly compatible with mainstream diffusion models and consistently delivers robust, unified I2I translation, even under limited paired data. Our code is available at https://github.com/HKU-HealthAI/DRDD.

#### 1. Introduction

Image-to-image (I2I) translation aims to map an input image from a source domain to a target domain, a fundamental task

*Equally contributed. †Corresponding author.

in computer vision with wide-ranging applications [47, 53], including image restoration [28, 69], super-resolution [52], image style translation [6, 42], among others [37]. Early approaches to I2I translation have been dominated by Generative Adversarial Networks (GANs) [5, 13, 71]. However, GANs suffer from training instability and limited mode coverage [43]. More recently, denoising diffusion models have set new benchmarks for output quality and diversity by learning a reversible process of iteratively adding and removing noise [12, 17, 31]. Early diffusion-based I2I methods, such as SR3 [52] and WeatherDiff [73], typically initiate the reverse process from pure Gaussian noise, using the input image solely as a conditioning signal. To more stably preserve structural information of the input and reduce inference uncertainty, advanced approaches, e.g., RDDM [34] and I2SB [32], no longer initialize from pure noise; instead, they start from noise-carrying input image [10, 38]. Even with variant starting points for reverse sampling, they share an underlying principle: image-to-image translation is achieved through a single, coupled reverse process, where noise and residuals are simultaneously removed at each diffusion step.

Despite these promising advances, applying these coupled diffusion models to unified I2I translation, where one model must handle multiple distinct tasks and domains, remains challenging. The primary difficulty stems from the substantial domain gaps between different I2I translation tasks, as well as the challenge of collecting large-scale paired source-to-target images that adequately cover this diversity. Here, we re-examine the diffusion paradigm to enhance its suitability for data-efficient and unified I2I translation. A key insight from our investigation concerns the role of injected Gaussian noise in diffusion models. Beyond its conventional functions of moving data off low-dimensional manifolds and enriching training signals for score estimation [23, 55], we theoretically and empirically demonstrate that injecting a certain level of Gaussian noise can act as a “domain harmonizer”, pulling feature representations from disparate domains closer together (see Fig. 1(a), (b) and Proposition

[Figure 1]

- Figure 1. Left: Domain gap reduction via noise introduction. t-SNE plot of feature representations across three I2I translation tasks. The original source domain (a) shows a significant domain gap, complicating unified I2I translation. Introducing noise (Source+Noise domain, b) reduces the domain gap, as shown by noticeably closer feature representation. Here, noise-carrying domains refer to domains with noise added to original images (e.g., Source+Noise and Target+Noise domains) as noise-carrying domains, excluding pure noise. Right: Reverse process comparison. During the inference stage, traditional coupled diffusion models perform domain shifting from the noise-carrying input to the target by simultaneously removing the residual (the difference between the source and target) and the noise [(b) → (d)]. Different from them, DRDD first performs residual removal exclusively within the noise-carrying domain, and then performs a denoising process to transform noisy-carrying target into clean target [(b) → (c) → (d)].

3.1). This emergent property, which is particularly beneficial for unified I2I translation tasks, reveals an under-explored utility of noise in diffusion processes. Nevertheless, in prevailing coupled diffusion models [10, 32, 34, 38], the reverse process progressively removes the injected noise, thereby eroding this harmonization benefit before the source-totarget mapping is completed. This ultimately undermines the model’s effectiveness in a unified, data-efficient setting.

To fully unleash the role of injected Gaussian noise, we propose Decoupled Residual Denoising Diffusion models for I2I Translation (denoted as DRDD). As shown in Fig. 1 and Fig. 2, DRDD fundamentally decouples the conventional single forward diffusion process into two independent and sequential diffusion processes: (1) a stochastic noise diffusion stage first injects Gaussian perturbations for domain harmonization and manifold lifting, projecting the target domain into a harmonized, noise-carrying domain; (2) this is followed by a deterministic residual diffusion stage that learns the target-to-source mapping within this fixed-noise domain. The reverse process is symmetrically decoupled. It first performs residual removal within a noise-carrying domain (with a predefined and fixed noise level) to achieve the core source-to-target transformation, thereby preserving

the domain harmonization and manifold lifting effects. This is followed by a denoising stage for fidelity refinement. This entire reverse process is clearly visualized in Fig. 1(b) → Fig. 1(c) → Fig. 1(d) and Fig. 2.

This novel decoupling mechanism offers two key advantages. First, by performing the core source-to-target mapping before any noise is removed, the domain harmonization and manifold lifting effects of the initial noise persist, thereby substantially simplifying the learning of a unified image mapping. Second, the design inherently enhances data efficiency, as the denoising stage is trained exclusively on target-domain images. This enables DRDD to leverage abundant unpaired data to boost final fidelity. As a result, DRDD establishes a new paradigm for unified and data-efficient I2I transformation, delivering robust performance across diverse tasks, even with limited paired data. Notably, both theoretical analysis and empirical results confirm that our residual and noise decoupling idea is compatible with multiple popular diffusion paradigms, including DDPM [17], DDIM [54], and SDE-based diffusion models [56]. Our core contributions are listed as follows:

• We uncover and formalize a novel role of Gaussian noise as a “domain harmonizer” in diffusion models. We theo-

retically and empirically demonstrate that controlled noise injection can effectively bridge the representation gap between disparate domains, a property particularly advantageous for unified I2I translation.

- • We propose DRDD, a novel diffusion method that decouples standard diffusion into sequential noise diffusion and residual diffusion stages. This decoupling strategically separates domain harmonization from semantic mapping, ensuring the harmonization effect persists throughout the core source-to-target transformation by performing residual removal entirely within the noisy domain.
- • Our decoupled design establishes a new paradigm for dataefficient and unified I2I translation. It not only simplifies the learning of a unified mapping across tasks but also enables the denoising stage to be trained exclusively on abundant, unpaired target-domain images, achieving robust performance with limited paired data. We further validate the broad compatibility of our framework with mainstream diffusion paradigms.

#### 2. Related Works

Image-to-image translation (I2I) aims to transfer images from a source domain to a target domain while preserving the content representations, with wide applications in many computer vision tasks [6, 24, 30, 42, 52, 53, 63, 66, 70]. Diffusion models have shown their impressive performance in I2I tasks. SR3 [52] first applies diffusion models in I2I, focusing on super-resolution and pioneers the usage of the input image as a condition for sampling from pure noise to a clear image. Subsequent works [6, 37, 53] extend diffusionbased I2I works to image inpainting, style transformation and colorization. Meanwhile, mainstream diffusion-based I2I approaches like RDDM [34] and others [10, 32, 38] enhance performance by initializing the reverse process with a noisy input to better preserve input information and reduce uncertainty. This strategy drives the model to perform denoising and the required domain translation (e.g., residual removal) simultaneously within a single, coupled reverse process.

#### 3. Method 3.1. Motivation

The widespread popularity of diffusion models originates with denoising diffusion probabilistic models (DDPMs) [17]. Since then, denoising and diffusion have been tightly coupled across a range of generative and I2I translation tasks. This close coupling has sometimes fostered the impression that “the denoising network itself is responsible for producing a clean target image containing semantic information.”

We revisit this impression for two reasons. (1) If we examine only the objective function of DDPM [17], the network learns solely denoising capabilities, which has no direct con-

nection to generating a clear target image. We argue that the generative semantic capability of diffusion models does not stem from the denoising network itself, nor even from the denoising process1, but rather a) from the mutual representation between the predicted noise ϵθ and the predicted target image I0θ (i.e., I0θ = f(ϵθ), see Eq.9 in DDIM [54] and Eq.16 in RDDM [34]); b) from the sampling formula derived from the mutual representation. (2) When diffusion models were extended from image generation to I2I translation, researchers observed that predicting residual [36, 62], target images [3, 10] or its linear transformation terms [32] often yielded better results than predicting noise. But noise injecting is still used empirically in I2I translations because its addition has been observed to improve performance [10]. These observations collectively suggest that the role of noise is nuanced. However, the community lacks a clear understanding of noise’s role in I2I translation.

In this paper, we discover an additional role for noise in I2I translation tasks, i.e., domain harmonization that injects noise can reduce the distance between feature representations across different domains (see Fig. 1(a) and (b)), with proof provided in Section 3.2. To leverage this new discovery, in

- Section 3.3, we thoroughly decouple the traditional singlestage coupled forward diffusion process into a two-stage process, involving noise diffusion and residual diffusion. In
- Section 3.4, we introduce the decoupled reverse process involving the residual removal stage and denoising stage. 3.2. The Role of Noise in Diffusion Models

Conventional functions of noise in diffusion models are moving data off low-dimensional manifolds and enriching training signals for score estimation [23, 55], with the noise being controlled over time schedules. Beyond this, we discover that without time-step control, a certain level of fixed Gaussian noise can act as a “domain harmonizer” in unified I2I tasks, minimizing the distribution gap of features across different domains. Here we give a mathematical expression:

Proposition 3.1. Let P and Q be two distinct probability distributions over a space X. Suppose that we inject Gaussian noise N(0,σ2) (with σ ̸= 0) to both distributions and denote Pσ and Qσ as the resulting distributions. Then, the Kullback-Leibler (KL) divergence between Pσ and Qσ is less than the KL divergence between P and Q:

DKL(Pσ ∥ Qσ) < DKL(P ∥ Q) (1)

The proof is provided in Appendix A.1. Although this “domain harmonizer” is particularly beneficial for unified I2I

1RDDM [34] reveals that, whether reverse sampling begins from pure noise (e.g., vanilla DDPM [17] and SR3 [52]) or from noise-carrying input image (e.g., I2SB [32] and IR-SDE [38] with additional diffusion terms), they all involve residual diffusion and noise diffusion. Our experiments in Fig. 2 also show that the denoising process itself only learns to remove noise and cannot generate semantic information corresponding to clear target images.

[Figure 2]

- Figure 2. Proposed DRDD framework. DRDD decouples the traditional single forward diffusion processes into two sequential and independent process: a noise diffusion stage that injects Gaussian noise into the target image, followed by a residual diffusion stage that conducts deterministic target-to-source transformation, but now within a noise-carrying level. The reverse diffusion process is correspondingly decoupled into a residual removal stage and a denoising stage.

translation tasks, the conventional coupled diffusion process removes the injected noise simultaneously with residuals, thereby undermines such harmonizing benefit.

- 3.3. Decoupled Forward Process

diffusion, models the deterministic target-to-source process by injecting the residual Ires:

It(2) = It(2)−1 + αtIres = I0(2) + α¯tIres (3)

where αt is the residual schedule coefficient (α¯t = t i=1 αi ) and It(2) denotes the image at the forward dif-

DRDD decouples the forward diffusion process into two sequential and independent stages: a stochastic noise diffusion followed by a deterministic residual diffusion. As illustrated in the top row of Fig. 2, the forward process starts from target

fusion step t in the residual diffusion stage. When t = T2 (total steps of residual diffusion) and α¯T

###### = 1, IT(2)

yields the final stage, completing the forward process:

2

2

= I0(2) + Ires = I0(1) + β¯T1ε + Ires = Iin + β¯T1ε. (4)

IT(2)

image I0(1), where we inject Gaussian noises to obtain noisecarrying target IT(1)

2

. Then this is followed by a deterministic residual diffusion stage that models target-to-source mapping within a fixed-noise domain, leading to noise-carrying

##### 3.4. Decoupled Reverse Process

1

Correspondingly, the reverse process is decoupled into two independent stages: residual-removal and denoising. Each stage is managed by a network trained with distinct objectives. As shown in Fig. 2, the reverse process starts from

input image IT(2)

. This entire process is visualized in Fig. 2 through “Forward Process”: I0(1) → IT(1)

2

###### = I0(2) → IT(2)

.

1

2

Given a paired input image Iin and target image I0(1), the noise diffusion stage perturbs I0 by progressively injecting Gaussian noise as:

noise-carrying image IT(2)

, where DRDD first performs residual removal exclusively within the noise-carrying domain and obtains I0(2). Then it conducts a denoising process to transform noise-carrying target IT(1)

2

It(1) = It(1)−1 + βtεt−1 = I0(1) + β¯t ε, (2)

into clean target I0(1). This entire reverse process is visualized in Fig. 2 through “Reverse Process”: IT(2)

1

where εt−1,...,ε ∼ N(0,I), and βt is the noise coefficient schedule that controls the noise diffusion speed

→ I0(1). In the residual-removal stage (IT(2)

###### → I0(2) = IT(1)

1

2

→ I0(2)), DRDD aims to remove residuals from IT(2)

(β¯t = ti=1 βi2). It(1) denotes the image at the forward diffusion step t in the noise diffusion stage. This injected noise serves as ”domain harmonizer” as well as utilizes the original manifold lifting ability. We then pass this terminal state of diffusion stage to the residual diffusion stage as

2

, which involves estimation of the residuals injected during the forward process, as described in Eq. 3. To this end, we train a residual re-

2

moval network denoted as Iresθ (It(2),t,Iin). Given the current image It(2), timestep t and the degraded image Iin, the network learns to predict residuals in a noise-carrying domain. Using Eq. 3, we obtain the estimated target images

the initial state by setting I0(2) := IT(1)

(see Fig. 2). We define residual as the difference between the source and target images: Ires = Iin − I0. The subsequent stage, residual

1

Algorithm 1: Training Algorithm Input: Target image: I0; Input image: Iin

Residual image: Ires = Iin − I0.

- 1 repeat

- 2 t ∼ Uni(1,...T1),ϵ ∼ N(0,I),It(1) = I0 +βtϵ;

- 3 Take gradient descent step on ∇θ = ∥ϵ − ϵθ(It(1),t)∥1
- 4 until converged;
- 5 repeat

- 6 t ∼ Uni(1,...T2),It(2) = IT(1)

+ αtIres; 78 Take gradient descent step on

1

∇θ∥Ires − Iresθ (It(2),Iin,t)∥1 9 until converged;

I0(2)(θ) = It(2) − α¯tIresθ . Given I0(2)(θ) and Iresθ , the generation process of residual removal is defined as:

pθ(It(2)−1 | It(2)) := q(It(2)−1 | It(2),I0(2)(θ),Iresθ )

= N It(2)−1;I0(2)(θ) + α¯t−1Iresθ , 0 . (5) With Eq. 3 and 5, It−1 can be sampled from It via:

It(2)−1 = It(2) − αtIresθ (It(2),Iin,t). (6)

By performing the core source-to-target mapping before any noise is removed, the domain-harmonizing and manifold lifting effects are preserved, thereby substantially simplifying the learning of a unified image mapping.

→ I0(1)), we train a denoise network ϵθ which learns to remove Gaussian noises. Using Eq. 2, we obtained the estimated target image I0(1)(θ) = It(1) − β¯t ϵθ. Given It(1) and model prediction of the noise ϵθ, we factor this variational distribution qσ as:

In the denoising stage (IT(1)

1

pθ It(1)−1 | It(1) := qσ It(1)−1 | It(1),I0(1)(θ)

(It(1) − I0(1)(θ)) β¯t

= N(It−1; β¯t2−1 − σt2

,σt2I), (7)

where σt2 = ηβt2β¯t2−1/β¯t2 and η controls whether the generation process is random (η = 1) or deterministic (η = 0). With Eq. 2 and 7, the iterative process is (see Appendix A.2):

It(1)−1 = It(1) − (β¯t − β¯t2−1 − σt2) ϵθ(It(1), t) + σtεt. (8)

The complete sampling algorithm is shown in Alg. 2.

Training Objectives. We derive the following simplified loss function for training the residual removal network Iresθ and the denoising network ϵθ (see proofs in Appendix A.3):

Lres(θ) = E Ires − Iresθ (It(2),t,Iin)

. (9)

1

Lϵ(θ) = E ϵ − ϵθ(It(1),t)

. (10)

1

Algorithm 2: Sampling Algorithm Input: Input image: Iin.

- 1 ϵ ∼ N(0,I);
- 2 IT(2)

2

= Iin + βT

1

ϵ;

- 3 for t = T2,...,1 do

- 4 It(2)−1 = It(2) − αtIresθ (It(2),Iin,t);
- 5 end
- 6 IT(1)

1

= I0(2);

- 7 for t = T1,...,1 do

- 8 It(1)−1 = It(1) − (β¯t − β¯t2−1 − σt2) · ϵθ(It(1),t) + σtεt;

- 9 end
- 10 return I0(1)

The complete training algorithm is shown in Alg. 1. According to Eq. 10, since denoising network is trained solely on clean images with no need of corresponding source domain images, DRDD significantly enhances data efficiency. Besides, we can initialize the denoising network with weights pretrained on large scale natural image datasets. Although our derivation builds upon the DDPM [17] and DDIM [54] framework, it is also compatible with score-based SDE [56] methods. The detailed derivation is in Appendix A.4.

#### 4. Experiments

In the experiments, we first validate DRDD’s effectiveness in unified I2I translation through two challenging settings: (1) multi-task benchmarks (All-in-One-5 [8] and CDD-11 [15]), where a single model handles multiple distinct restoration tasks; and (2) cross-domain single I2I translation task using our self-collected MNMD benchmark, where images from disparate domains within a single task. Next, we show that DRDD’s advantages extend to standard single I2I translation tasks, where its domain harmonization mitigates instancelevel distribution shifts. We then conduct data efficiency analyses, verifying DRDD’s robust performance with limited paired data on both task-specific and unified settings. Further, we demonstrate DDRD’s compatibility with other diffusion backbones. Finally, we theoretically and empirically investigate the optimal noise injection level for DRDD.

##### 4.1. Experiment Settings

Datasets. We use a diverse set of widely-used datasets across various I2I translation tasks. For unified image restoration, we experiment on All-in-One-5 [8] and CDD-11 [15], which includes various restoration tasks. We also construct an MNMD benchmark, which contains various types of noise and covers multiple domains. We use All-in-One-3 [8] and Low-Light [61] to validate our data efficiency. Studies on inpainting (CelebA-HQ [19]), super-resolution (FFHQ [20])

- Table 1. Performance comparisons of five unified multi-task image restoration tasks on All-in-One-5 dataset [8]. Denoising results are reported at the noise level σ = 25. SSIM (↑), LPIPS (↓) and FID (↓) are reported. Best results are highlighted in red, while the second-best results are blue. Diffusion-based methods are denoted by “*”. Our DRDD demonstrates superior or competitive performance compared to recent models, especially in perceptual metrics. Due to space limitation, PSNR results and computational costs are provided in Appendix C.5

.

Low-Light Deraining Denoising Deblurring Dehazing Average

Method

SSIM / LPIPS / FID SSIM / LPIPS / FID SSIM / LPIPS / FID SSIM / LPIPS / FID SSIM / LPIPS / FID SSIM / LPIPS / FID DA-CLIP* [39] 0.819 / 0.115 / 36.2 0.973 / 0.169 / 9.25 0.809 / 0.108 / 34.4 0.829 / 0.135 / 16.2 0.959 / 0.015 / 3.82 0.876 / 0.108 / 20.0 DiffuIR* [69] 0.804 / 0.204 / 77.7 0.961 / 0.042 / 18.3 0.856 / 0.114 / 34.9 0.793 / 0.182 / 24.1 0.930 / 0.046 / 13.6 0.869 / 0.117 / 33.7 AdAIR [9] 0.844 / 0.120 / 48.9 0.978 / 0.015 / 8.73 0.888 / 0.109 / 39.2 0.857 / 0.189 / 19.9 0.975 / 0.015 / 14.5 0.909 / 0.089 / 26.1 VLUNet [67] 0.832 / 0.144 / 60.4 0.981 / 0.012 / 6.17 0.890 / 0.098 / 36.4 0.840 / 0.214 / 24.0 0.979 / 0.012 / 12.9 0.904 / 0.096 / 27.9 DFPIR [57] 0.843 / 0.122 / 50.6 0.977 / 0.017 / 8.32 0.889 / 0.091 / 35.0 0.873 / 0.164 / 17.0 0.978 / 0.013 / 13.6 0.912 / 0.081 / 24.9 DRDD(Ours)* 0.864 / 0.103 / 35.4 0.978 / 0.014 / 8.06 0.893 / 0.097 / 28.9 0.881 / 0.134 / 15.8 0.972 / 0.012 / 3.54 0.916 / 0.073 / 18.3

##### 4.2. Performance on I2I Translation Tasks

and other task-specific I2I translation further demonstrate our framework’s capability on single task I2I translation. The configuration of the aforementioned datasets is detailed in Appendix B.

To comprehensively evaluate the unified capability of DRDD on I2I translation tasks, we design experiments from three perspectives: handling multiple restoration tasks, performing a single task across multiple domains, and conducting taskspecific I2I within a single domain.

Model Architecture. Following design in ADM [12], the denoising network in DRDD, DiffUIR [69] and RDDM [34] adopt the same UNet backbone and hyperparameter settings, where the channel depth is C = 128 with channel multiplier = (1,1,2,2,4,4). Our denoising model follows the U-Net architecture in [12], where the channel depth is C = 64 with channel multiplier = (1,2,4,8).

Multi-task Unified Restoration. Following recent studies [57, 67], we evaluate the effectiveness of our method in handling various image degradation types on the All-in-One-5 benchmark (see Appendix. B.2). As shown in Tab. 1, DRDD achieves state-of-the-art (SOTA) performance on most image restoration tasks. Notably, DRDD consistently outperforms both recent diffusion-based approaches (DA-CLIP [39], DiffuIR [46]) and other non-diffusion based (DFPIR [57], VLUNet [67], and AdAIR [9]) in all three metrics. This advantage is also clearly reflected in Fig. 4-a and Fig. 4-b, where DRDD produces restorations with richer details and fewer artifacts compared to other methods.

As for inference, we adopt the DDIM[54] sampling strategy as described by [54], with the sampling step size set to 2 for both the denoising and residual removal stages in all experiments. We provide inference steps and corresponding model performance in Appendix C.5.2.

[Figure 3]

L H

Avg.

The CDD-11 dataset [15], which is used to assess robustness and generalization under complex scenarios, contains 11 different degradation types (Appendix. B.3). Fig. 3 presents a comparison between DRDD and five recent approaches [15, 27, 48, 65, 72] in CDD-11. DRDD consistently outperforms recent SOTA models across most degradation categories, achieving the highest average SSIM score. In particular, DRDD demonstrates clear advantages in challenging composite scenarios (e.g., the L+H+S and L+H+R in Fig. 3), where other methods such as PromptIR [48], WGWSNet [72], and MoCE-IR-S [65] experience notable performance drops. These results demonstrate the superiority of the DRDD framework on unified image restoration benchmarks.

[Figure 4]

[Figure 5]

R

L+H+S

S

L+H+R

L+H

H+S

H+R L+S

L+R

AirNet PromptIR WGWSNet OneRestore MoCE-IR-S DRDD(Ours)

L : Low-Light H : Dehaze R : Derain S : Desnow

Single Task I2I Translation in Multi-Domain. Recent All-in-One models often neglect performance drops from domain shifts. We believe that residual removal in the noisecarrying domain reduces degradation conflicts, while serving as a domain harmonizer that aligns feature representations across domains and mitigates domain gap. Therefore, we

- Figure 3. Quantitative comparison to state-of-the-art on 11 degradation tasks and their average. SSIM (↑) is reported. Our DRDD method consistently outperforms recent SOTA models, with favorable results in complex composited degradation scenarios. All experiments are conducted on the CDD11 dataset [15].

[Figure 6]

- (a)
- (b)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Center & Irregular Mask DRDD(Ours) GroundTruth DRDD(Ours) GroundTruth

- (c)

Low-Light Image VLU-NET DiffUIR DFPIR DRDD(Ours) GroundTruth

[Figure 13]

Blur Image VLU-NET DiffUIR DFPIR DRDD(Ours) GroundTruth

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

(d)

Low-Resolution

- Figure 4. Visual results of state-of-the-art methods and our proposed DRDD. (a) Comparison of low-light enhancement results on the LoLV1 dataset [61]. (b) Comparison of blur restoration results on the GoPro dataset [44]. (c) Face inpainting results (center and irregular mask) in CelebA-HQ [19]. (d) Super-Resolution result in FFHQ [20]. Zoom in for best view. More visual results are provided in Appendix D.

- Table 2. Performance comparison of several methods on MNMD dataset. Best results are highlighted in Bold.

Natural Medical Remote Average SSIM↑ LPIPS↓ SSIM↑ LPIPS↓ SSIM↑ LPIPS↓ SSIM↑ LPIPS↓ RDDM [34] 0.8333 0.1703 0.8343 0.1917 0.8542 0.1485 0.8406 0.1702 IR-SDE [38] 0.8062 0.1129 0.8492 0.0510 0.8090 0.0999 0.8215 0.0879 VLUNET [67] 0.9308 0.0782 0.9267 0.0840 0.9249 0.0710 0.9274 0.0784 DRDD(Ours) 0.9391 0.0492 0.9324 0.0629 0.9300 0.0539 0.9338 0.0553

Method

focus on a multi-domain image denoising task and establish a challenging multi-domain benchmark by adding different kinds of noise to natural (WED+BSD400 [49]), remote sensing (UC-Merced [45]), and medical (BrainWeb [7, 21, 22]) image datasets. As shown in Tab. 2, we compare our proposed DRDD model with several SOTA restoration methods, including RDDM [34], IR-SDE [38], and VLUNET [67]. Across all domains, our model consistently achieves the highest SSIM and lowest LPIPS on natural, medical, and remote sensing images. This demonstrates that DRDD effectively handles multi-domain image restoration within a single task. The complete benchmark construction pipeline is detailed in the Appendix B.4.

Single-Task I2I Translation in Single Domain. Distribution gaps can occur even within a single I2I translation task in one domain due to diverse input characteristics. Here, we further verify the effectiveness of DRDD on single-task I2I translation problems, where the data is sourced from a single domain. Specifically, we conduct experiments on image inpainting, super-resolution, and other single tasks, such as

image deraining and low-light enhancement. For image inpainting, DRDD is compared with CTSDG [14], MISF [29], and TransRef [35] on the CelebA-HQ [19] dataset under various mask patterns and resolutions. See Fig. 4-c and Appendix C.1 for all the quantitative and qualitative comparisons. These extensive results demonstrate that DRDD also achieves superior performance on single I2I translation tasks.

##### 4.3. Performance on Limited Training Data

We now validate another key advantage of our decoupled design: its ability to achieve superior performance with limited paired data. We randomly sub-sample the training set to 75%, 50% and 25% while keeping the validation set fixed. As shown in Fig. 5, on two representative datasets (LowLight and All-in-One-3, see details in Appendix B.1), DRDD achieves better performance than existing baselines, especially under limited data conditions. Here we initialize the training of denoising network with pretrained weights on ImageNet [11]. Notably, as the amount of training data decreases, the relative performance drop of DRDD remains substantially smaller compared to other methods, underscoring the data efficiency of our approach. These results demonstrate that DRDD can effectively maintain restoration quality even with severely reduced training data, highlighting its practical superiority in data-constrained cases.

##### 4.4. Extensions to other Diffusion Paradigms

We further incorporate the decoupling strategy into a SDEbased diffusion model built upon IR-SDE [38] to ensure

###### Data Pruning for Low Light

0.206

0.881 0.872 0.874

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |0.175|0.153| | | |
| | |0.145|0.144| | |
| |0.125| |0.135| | |
| | |0.105|0.102| | |
| | | | | | |
| | | | | | |

0.88

0.20

0.18

0.850 0.845

0.86

0.851 0.852 0.839

0.16

###### LPIPS

0.835

SSIM

0.136

0.84

0.14

0.131

0.82

0.12

0.814

0.799

0.10

0.80

0.796

0.089

25 50 75 100

25 50 75 100

Data Percentage (%)

Data Percentage (%)

###### Data Pruning for All-in-One3

0.952

0.050

| | | |0.950<br><br>0.951| |
|---|---|---|---|---|
| |0.947<br><br>0.945|0.949<br><br>0.946| | |
| |0.940|0.942|0.944| |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
|0.046|0.042<br><br>0.046|0.044| | |
|0.041|0.040|0.039<br><br>0.041| | |
| | | | | |

0.951

0.950

0.048

0.948

###### LPIPS

SSIM

0.043

0.945

0.042

0.040

0.038

0.940

25 50 75 100

25 50 75 100

Data Percentage (%)

Data Percentage (%)

DiffUIR with pretrained sampler

VLU-NET

DRDD(Ours) with pretrained denoiser

- Figure 5. Data Pruning on All-in-One-3 [8] and Low-Light dataset. SSIM (↑) and LPIPS (↓) are reported. As the training data decreases, DRDD’s performance drop is much smaller than other methods.

the compatibility of our framework. As shown in Tab. 3, the decoupled SDE consistently outperforms the baseline SDE on two important tasks: deraining and inpainting. In addition, it achieves comparable performance in denoising while obtaining better FID scores. These results indicate that the decoupling mechanism can be effectively extended to other diffusion frameworks.

##### 4.5. Investigation of Noise Injection Level

The reverse process of DRDD starts from a noise-carrying image, where a Gaussian noise with a predefined and fixed noise level is injected. The intensity of such Gaussian noise is critical to overall performance (see Proposition 3.1). Here, we explore this issue from both theoretical and experimental perspectives. We define two distinct distances:

###### A(σ) = ∆(Psσ,Ptσ), B(σ) = ∆(Psσ,Ps) (11)

where A(σ) quantifies the distance between noise-carrying target distribution Pσt and the noise-carrying source distribution Pσs while B(σ) quantifies the distance between the noise-carrying source distribution Pσs and the original source distribution Ps. ∆ denotes the Maximum Mean Discrepancy (MMD) in this case, and the full formula is provided in the Appendix A.5. As the noise level σ increases, both A(σ) and B(σ) increase monotonically. We aim to find a noise level that A(σ) is small enough (since a larger distance complicates translation), and B(σ) is also minimized (to prevent

Table 3. Performance comparison of decoupled and coupled SDEbased diffusion methods on single task I2I. Results are evaluated on the CelebA-HQ [19], Rain100 [49], and BSD400 [2] datasets.

Inpainting Deraining Denoise

Method

LPIPS↓ FID↓ PSNR↑ SSIM↑ LPIPS↓ SSIM↑ LPIPS↓ FID↓ IR-SDE [38] 0.0517 15.14 27.2 0.856 0.083 0.833 0.1014 33.29 De-IRSDE(Ours) 0.0490 15.10 28.1 0.862 0.076 0.827 0.1069 31.87

significant input corruption), which leads to:

J(σ;λ) = λ A(σ) + (1 − λ) B(σ), λ ∈ [0,1]. (12)

where A(σ) and B(σ) represent the normalized values of A(σ) and B(σ), respectively. By minimizing this objective function, we obtain the optimal noise level σJ⋆ = arg minJ(σ;λ). The value of λ can be adjusted based on the desired balance between the two distances. We calculate the results on the All-in-One-5 datasets, taking λ = 0.5, and the obtained optimal β¯ (noise intensity) is around 1.1 to 1.2. To further validate our Eq. 12, we conduct quantitative experiments across different noise levels. As shown in Fig. 6, the models achieve optimal performance when the noise intensity is set to 1.0, with stable and superior results observed in the range of 0.8 to 1.3. These findings align with our theoretical expectations.

SSIM and LPIPS under Different Noise Levels

0.120

| |
|---|

0.896

0.115

0.110

0.888

SSIM

LPIPS

0.105

SSIM

0.880

LPIPS

0.100

0.872

0.095

0.864

| |
|---|

| |
|---|

0.090

| |
|---|

| |
|---|

| |
|---|

0.3 0.6 0.9 1.2 1.5 1.8

Noise Level

Figure 6. Performance comparison on the All-in-One-5 dataset[8] under varying noise injection level.

#### 5. Conclusion

In this paper, we present DRDD, a novel diffusion model that decouples standard diffusion into sequential noise diffusion and residual diffusion stages. Our work discovers a novel role of Gaussian noise as a “domain harmonizer”, which leads us to rethink conventional coupled diffusion models and propose a novel decoupled framework. By leveraging its decoupled mechanism, DRDD not only simplifies the learning of a unified mapping across tasks but also enables the denoising stage to be trained exclusively on unpaired images. Comprehensive theoretical and empirical analyses demonstrate DRDD’s effectiveness. We believe this work opens new perspectives on noise utilization in generative models and provides a solid foundation for unified image translation systems.

#### 6. Acknowledgements

This work was supported by National Natural Science Foundation of China (62306253, T2522030), Early Career Scheme from the Research Grants Council of Hong Kong SAR (27207025, 27204623), Guangdong Natural Science Fund-General Programme (2024A1515010233), China Postdoctoral Science Foundation under Grant Number 2025M781669, and the Fundamental Research Project of SIA (2025JC1K05).

#### References

- [1] Abdelrahman Abdelhamed, Stephen Lin, and Michael S Brown. A high-quality denoising dataset for smartphone cameras. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1692–1700, 2018. 8
- [2] Pablo Arbelaez, Michael Maire, Charless Fowlkes, and Jitendra Malik. Contour detection and hierarchical image segmentation. IEEE transactions on pattern analysis and machine intelligence, 33(5):898–916, 2010. 8, 4, 5, 6, 7, 10
- [3] Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise. arXiv preprint arXiv:2208.09392,

2022. 3

- [4] Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning. Springer, 2006. 2
- [5] Xinyuan Chen, Chang Xu, Xiaokang Yang, Li Song, and Dacheng Tao. Gated-gan: Adversarial gated networks for multi-collection style transfer. IEEE Transactions on Image Processing, 28(2):546–560, 2019. 1
- [6] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models, 2021. 1, 3
- [7] D Louis Collins, Alex P Zijdenbos, Vasken Kollokian, John G Sled, Noor Jehan Kabani, Colin J Holmes, and Alan C Evans. Design and construction of a realistic digital brain phantom. IEEE transactions on medical imaging, 17(3):463–468, 2002. 7, 5
- [8] Marcos V Conde, Gregor Geigle, and Radu Timofte. Instructir: High-quality image restoration following human instructions. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 5, 6, 8
- [9] Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan. AdaIR: Adaptive all-in-one image restoration via frequency mining and modulation. In The Thirteenth International Conference on Learning Representations, 2025. 6, 7, 9
- [10] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration, 2024. 1, 2, 3
- [11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. 7

- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1, 6, 4
- [13] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014. 1
- [14] Xiefan Guo, Hongyu Yang, and Di Huang. Image inpainting via conditional texture and structure dual generation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14134–14143, 2021. 7
- [15] Yu Guo, Yuan Gao, Yuxu Lu, Huilin Zhu, Ryan Wen Liu, and Shengfeng He. Onerestore: A universal restoration framework for composite degradation. In European conference on computer vision, pages 255–272. Springer, 2024. 5, 6, 4
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 4
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851. Curran Associates, Inc., 2020. 1, 2, 3, 5
- [18] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 6
- [19] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018. 5, 7, 8
- [20] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 5, 7, 10
- [21] Remi K-S Kwan, Alan C Evans, and G Bruce Pike. An extensible mri simulator for post-processing evaluation. In International conference on visualization in biomedical computing, pages 135–140. Springer, 1996. 7, 4, 5
- [22] RK-S Kwan, Alan C Evans, and G Bruce Pike. Mri simulation-based evaluation of image-processing and classification methods. IEEE transactions on medical imaging, 18(11):1085–1097, 1999. 7, 5
- [23] Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji, and Stefano Ermon. The principles of diffusion models, 2025. 1, 3
- [24] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network, 2017. 3
- [25] Chulwoo Lee, Chul Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation of 2d histograms. IEEE transactions on image processing, 22(12): 5372–5384, 2013. 8
- [26] Boyi Li, Wenqi Ren, Dengpan Fu, Dacheng Tao, Dan Feng, Wenjun Zeng, and Zhangyang Wang. Benchmarking single-

- image dehazing and beyond. IEEE Transactions on Image Processing, 28(1):492–505, 2019. 4, 10
- [27] Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, and Xi Peng. All-in-one image restoration for unknown corruption. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17452–17462,

2022. 6

- [28] Hao Li, Xiang Chen, Jiangxin Dong, Jinhui Tang, and Jinshan Pan. Foundir: Unleashing million-scale training data to advance foundation models for image restoration. In ICCV,

2025. 1

- [29] Xiaoguang Li, Qing Guo, Di Lin, Ping Li, Wei Feng, and Song Wang. Misf: Multi-level interactive siamese filtering for high-fidelity image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1869–1878, 2022. 7
- [30] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833– 1844, 2021. 3
- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1
- [32] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima Anandkumar. I2sb: Imageto-image schr¨odinger bridge, 2023. 1, 2, 3
- [33] Jiaying Liu, Dejia Xu, Wenhan Yang, Minhao Fan, and Haofeng Huang. Benchmarking low-light image enhancement and beyond. International Journal of Computer Vision, 129(4):1153–1184, 2021. 4
- [34] Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, and Liangqiong Qu. Residual denoising diffusion models, 2024. 1, 2, 3, 6, 7, 9
- [35] Taorong Liu, Liang Liao, Delin Chen, Jing Xiao, Zheng Wang, Chia-Wen Lin, and Shin’ichi Satoh. Transref: Multi-scale reference embedding transformer for reference-guided image inpainting. Neurocomputing, 632:129749, 2025. 7
- [36] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In Proc. ICLR, 2023. 3
- [37] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models, 2022. 1, 3
- [38] Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sj¨olund, and Thomas B. Sch¨on. Image restoration with mean-reverting stochastic differential equations, 2023. 1, 2, 3, 7, 8, 6
- [39] Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sj¨olund, and Thomas B. Sch¨on. Controlling vision-language models for multi-task image restoration, 2024. 6
- [40] Kede Ma, Kai Zeng, and Zhou Wang. Perceptual quality assessment for multi-exposure image fusion. IEEE Transactions on Image Processing, 24(11):3345–3356, 2015. 8
- [41] Kede Ma, Zhengfang Duanmu, Qingbo Wu, Zhou Wang, Hongwei Yong, Hongliang Li, and Lei Zhang. Waterloo exploration database: New challenges for image quality assessment models. IEEE Transactions on Image Processing,

26(2):1004–1016, 2017. 4, 5, 6, 7

- [42] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations,

2022. 1, 3

- [43] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge?, 2018. 1
- [44] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3883–3891, 2017. 7, 4, 5
- [45] Maxim Neumann, Andre Susano Pinto, Xiaohua Zhai, and Neil Houlsby. In-domain representation learning for remote sensing, 2019. 7, 4, 5
- [46] Mang Ning, Mingxiao Li, Jianlin Su, Haozhe Jia, Lanmiao Liu, Martin Beneˇs, Wenshuo Chen, Albert Ali Salah, and Itir Onal Ertugrul. Dctdiff: Intriguing properties of image generative modeling in the dct space, 2025. 6
- [47] Yingxue Pang, Jianxin Lin, Tao Qin, and Zhibo Chen. Imageto-image translation: Methods and applications, 2021. 1
- [48] Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Khan. Promptir: Prompting for all-in-one image restoration. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 6
- [49] Rui Qian, Robby T Tan, Wenhan Yang, Jiajun Su, and Jiaying Liu. Attentive generative adversarial network for raindrop removal from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2482–2491, 2018. 7, 8, 4, 5
- [50] Jaesung Rim, Haeyun Lee, Jucheol Won, and Sunghyun Cho. Real-world blur dataset for learning and benchmarking deblurring algorithms. In European conference on computer vision, pages 184–201. Springer, 2020. 9
- [51] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer,

2015. 4

- [52] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement, 2021. 1, 3
- [53] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022. 1, 3
- [54] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 2, 3, 5, 6

- [55] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 1, 3
- [56] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations,

2021. 2, 5

- [57] Xiangpeng Tian, Xiangyu Liao, Xiao Liu, Meng Li, and Chao Ren. Degradation-aware feature perturbation for all-in-one image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28165–28175, 2025. 6
- [58] Hebaixu Wang, Jing Zhang, Haonan Guo, Di Wang, Jiayi Ma, and Bo Du. Dgsolver: Diffusion generalist solver with universal posterior sampling for image restoration. arXiv preprint arXiv:2504.21487, 2025. 4
- [59] Shuhang Wang, Jin Zheng, Hai-Miao Hu, and Bo Li. Naturalness preserved enhancement algorithm for non-uniform illumination images. IEEE transactions on image processing, 22(9):3538–3548, 2013. 8
- [60] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 4
- [61] Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retinex decomposition for low-light enhancement. arXiv preprint arXiv:1808.04560, 2018. 5, 7, 4
- [62] Jay Whang, Mauricio Delbracio, Hossein Talebi, Chitwan Saharia, Alexandros G Dimakis, and Peyman Milanfar. Deblurring via stochastic refinement. In Proc. CVPR, pages 16293–16303, 2022. 3
- [63] Xing Xie, Jiawei Liu, Ziyue Lin, Huijie Fan, Zhi Han, Yandong Tang, and Liangqiong Qu. Unleashing the potential of large language models for text-to-image generation through autoregressive representation alignment, 2025. 3
- [64] Wenhan Yang, Robby T Tan, Jiashi Feng, Jiaying Liu, Zongming Guo, and Shuicheng Yan. Deep joint rain detection and removal from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1357–1366, 2017. 6, 8
- [65] Eduard Zamfir, Zongwei Wu, Nancy Mehta, Yuedong Tan, Danda Pani Paudel, Yulun Zhang, and Radu Timofte. Complexity experts are task-discriminative learners for any image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12753–12763, 2025. 6
- [66] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022. 3, 6
- [67] Haijin Zeng, Xiangming Wang, Yongyong Chen, Jingyong Su, and Jie Liu. Vision-language gradient descent-driven all-in-one deep unfolding networks. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7524–7533, 2025. 6, 7
- [68] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 4
- [69] Dian Zheng, Xiao-Ming Wu, Shuzhou Yang, Jian Zhang, JianFang Hu, and Wei-Shi Zheng. Selective hourglass mapping for universal image restoration based on diffusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 25445–25455, 2024. 1, 6, 7

- [70] Weiying Zheng, Ziyue Lin, Pengxin Guo, Yuyin Zhou, Feifei Wang, and Liangqiong Qu. Fedvlmbench: Benchmarking federated fine-tuning of vision-language models, 2025. 3
- [71] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks, 2020. 1
- [72] Yurui Zhu, Tianyu Wang, Xueyang Fu, Xuanyu Yang, Xin Guo, Jifeng Dai, Yu Qiao, and Xiaowei Hu. Learning weathergeneral and weather-specific features for image restoration under multiple adverse weather conditions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21747–21758, 2023. 6
- [73] Ozan Ozdenizci¨ and Robert Legenstein. Restoring vision in adverse weather conditions with patch-based denoising diffusion models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8):10346–10357, 2023. 1

## Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation

### Supplementary Material

#### A. Derivations and Proofs

##### A.1. Proofs of Proposition 3.1

In this section, we give a proof to the aforementioned proposition 3.1.

Proposition 3.1. Let P and Q be two distinct probability distributions over a space X. Suppose that we inject Gaussian noise N(0,σ2) (with σ ̸= 0) to both distributions and denote Pσ and Qσ as the resulting distributions. Then, the Kullback-Leibler (KL) divergence between Pσ and Qσ is less than the KL divergence between P and Q:

DKL(Pσ ∥ Qσ) < DKL(P ∥ Q) (13)

Proof: Let P and Q be two distinct probability distributions (P ̸= Q) with corresponding probability density functions

- p(x) and q(x). The KL divergence is defined as:

DKL(P ∥ Q) = p(x)log

- p(x)

- q(x)

dx (14)

where the integral is taken over the entire support of x. To add Gaussian noise to each data distribution in a random manner, we consider a Gaussian kernel K(y|x), which represents the probability of output y given input x. The Gaussian kernel is defined as:

K(y|x) =

- 1

√

- 2πσ2

exp −

(y − x)2 2σ2

, (15)

where σ2 is the variance of the noise. Adding Gaussian noise, the distributions for P and Q are modified to Pσ and Qσ, with the corresponding probability densities:

- pσ(y) = K(y|x)p(x)dx, (16)
- qσ(y) = K(y|x)q(x)dx. (17)

According to the definition:

DKL(Pσ ∥ Qσ) = pσ(y)log

- pσ(y)

- qσ(y)

dy

= DKL(P(Y ) ∥ Q(Y )) (18) Define the joint density p(x,y) = p(x)K(y|x) and

- q(x,y) = q(x)K(y|x). Note that due to Gaussian noise being independent, K(y|x) does not depend on p or q, so p(x|y) = K(y|x) for both P and Q. Here, KL divergence

DKL(P ∥ Q) is related to x, while DKL(Pσ ∥ Qσ) is related to y. We can now write the joint KL divergence as:

- p(x,y)

- q(x,y)

DKL(P(X,Y ) ∥ Q(X,Y )) = p(x,y)log

dxdy

(19) Since

- p(x)K(y|x)

- q(x)K(y|x)

- p(x,y)

- q(x,y)

- p(x)

- q(x)

(20)

=

=

DKL(P(X,Y ) ∥ Q(X,Y )) = p(x)K(y|x)log

- p(x)

- q(x)

dxdy (Eq. 20)

- p(x)

- q(x)

dx ( K(y|x)dy = 1)

= p(x)log

= DKL(P ∥ Q) (21) The joint KL divergence can be decomposed as:

DKL(P(X,Y ) ∥ Q(X,Y )) = DKL(P(Y ) ∥ Q(Y ))

+ DKL(P(X|Y ) ∥ Q(X|Y )|P(Y )) (22)

where DKL(P(Y ) ∥ Q(Y )) is the KL divergence of the marginals Pσ and Qσ, i.e., DKL(Pσ ∥ Qσ). The second term DKL(P(X|Y ) ∥ Q(X|Y )|P(Y )) is positive (P ̸= Q), hence

###### DKL(P(X,Y ) ∥ Q(X,Y )) < DKL(P(Y ) ∥ Q(Y )) (23)

According to Eq. 18 and Eq. 21, we transfer Eq. 23 to:

DKL(Pσ ∥ Qσ) < DKL(P ∥ Q) (24)

This means that, for most cases, if P ̸= Q, adding Gaussian noise with σ ̸= 0 will decrease the KL divergence between P and Q.

##### A.2. Proofs in Our Method

In this section, we give a detailed explanation to section 3.2, including detailed explanation of forward sampling process and proof of reverse sampling process.

###### Reverse Sampling steps of Residual-Removal Stage. Given Eq. 3 and Eq. 5, we have:

It(2)−1 = I0θ + α¯t−1Iresθ + σϵt

= (It(2) − α¯tIresθ ) + α¯t−1Iresθ + σϵt

= It(2) − (¯αt − α¯t−1)Iresθ + σϵt

= It(2) − α¯tIresθ + σϵt

= It(2) − αtIresθ

(25)

Finally, we have

It(2)−1 = It(2) − αtIresθ (It(2),Iin,t). (Eq. 6) (26)

###### Reverse Sampling steps of Denoising Stage. Given

It(1) = I0(1) + β¯t ε, (Eq. 2) (27) and

pθ It(1)−1 | It(1) := qσ It(1)−1 | It(1),I0(1)(θ) (Eq. 7)

(It(1) − I0(1)(θ)) β¯t

= N(It−1;I0(1)(θ) + β¯t2−1 − σt2

,σt2I), (28)

It(1)−1

(It(1) − I0(1)(θ)) β¯t

= I0(1)(θ) + β¯t2−1 − σt2

+ σtεt

(It1 − (It1 − β¯tϵt)) β¯t

= (It1 − β¯tϵt) + β¯t2−1 − σt2

+ σtεt

= (It(1) − β¯tϵt) + ϵt β¯t2−1 − σt2 + σtεt

= It(1)−1 = It(1) − (β¯t − β¯t2−1 − σt2)ϵθ(It(1),t) + σtεt.

(29)

where σt2 = ηβt2β¯t2−1/β¯t2. When generation process is deterministic (η = 0), we have:

###### It(1)−1 = It(1) − (β¯t − β¯t−1)Iϵθ It(1), t (30)

Derivation of Eq. 7 (Eq. 28). Similar to proof in RDDM [34] A.2, we have:

q(It|I0) = N(It;I0,β¯t2I). (31)

Similar to the evolution from DDPM [17] to DDIM [54], we can prove the statement with an induction argument for t from T to 1. Assuming that Eq. 31 holds at T, we just need

to verify q(It−1|I0) at t − 1 from q(It|I0) at t using Eq. 31. Given:

q(It|I0) = N(It;I0,β¯t2I), (32) qσ(It−1|It,I0) = N(It−1;I0 + β¯t2−1 − σt2

(It − I0) β¯t

,σt2I),

(33) q(It−1|I0) := N(˜µt−1,Σ˜t−1) (34) Similar to obtaining p(y) from p(x) and p(y|x) using Eq.2.113-Eq.2.115 in [4], the values of µ˜t−1 and Σ˜t−1 are derived as follows:

(I0) − (I0) β¯t

µ˜t−1 = I0 + β¯t2−1 − σt2

= I0, (35)

 

 

2

β¯t2−1 − σt2 β¯t

β¯t2I = β¯t2−1I. (36)

Σ˜t−1 = σt2I +

Therefore, q(It−1|I0) = N(It−1;I0,β¯t2−1I). In fact, the case (t = T) already holds, thus Eq. 31 holds for all t. We can derive Eq. 7 and Eq. 28 from Eq. 33.

##### A.3. Derivation of Training Objectives

In this section, we give a proof to the training objectives. According to Eq. 5, we derive the training objective of residualremoval process as follows:

Lres(θ)

= DKL q(It(2)−1 | It(2),I0(2)(θ),Iresθ ) ∥ pθ It(2)−1 | It(2)

= E It − αtIres − It − αtIresθ 2

= E Ires − Iresθ (It(2),t,Iin)

. (Eq. 10) (37)

1

According to Eq. 7, we derive the training objective of denoising process as follows:

Lϵ(θ)

= DKL q(It(1)−1 | It(1),I0(1)(θ) ∥ pθ It(1)−1 | It(1)

2

βt2 βt

βt2 βt

ϵθ

= E It −

ϵ − It −

= E ϵ − ϵθ(It(1),t)

1

. (Eq. 10) (38)

##### A.4. Derivation of Decoupled SDE

This section introduces decoupling paradigm in SDE-based diffusion models and show the process of generating samples with reverse-time SDEs. The forward process formula is as follows:

dx = θt (µ − x)dt + σt dw, (39)

where t denote the continuous time variable, w is a standard Wiener process, µ is the state mean, and θt,σt are timedependent positive parameters that characterize the speed of the mean-reversion and the stochastic volatility, respectively. In 39, θt (µ − x)dt is the drift term, which governs the evolutionary trend of the state, while σt dw denotes the random noise disturbance affecting the state. We then reverse the SDE to derive an image restoration SDE. During the testing phase, only the score ∇x log pt(x) needs to be predicted in this formula:

dx = θt (µ − x) − σt2 ∇x log pt(x) dt + σt d ˆw. (40) We decouple the forward process into a two-stage procedure, adding noise and degradation-specific information respectively, as follows:

dx(1) = σt dw (41)

dx(2) = θt (µ − x(2))dt + σt dw, (42) Finally, we can reverse the SDE by predicting the score in Formula 40 for each of the two stages. Given an initial state x0, for any state xi at discrete time i > 0, the optimum residual reversing solution x∗i−1 in (39) is given by:

1 − e−2θ¯

i−1

′

x(2)i−1∗ =

i(x(2)i − µ)

1 − e−2θ¯i e−θ

(43)

′ i

1 − e−2θ

1 − e−2θ¯i e−θ¯

(x(2)0 − µ) + µ.

+

i−1

For noise reversing x(1)i−1∗ is given by:

1 − e−2θ¯

i−1

x(1)i−1∗ =

(x(1)i − x(1)0 ) + x(1)0 . (44)

1 − e−2θ¯i e−θ

i

##### A.5. Noise-Level Selection via Dual MMD Distances

In this section, we give a detailed exploration of aforementioned Eq. 12 in Section 4.5.

A(σ) = ∆(Psσ,Ptσ), B(σ) = ∆(Psσ,Ps) (45) Here, A(σ) and B(σ) represent the measures of distance (in this case, based on MMD) between the distributions Psσ and Ptσ for A(σ), and Psσ and Ps for B(σ), where: - Psσ is the distribution of the source after adding Gaussian noise N(0,σ2) with σ being the noise strength parameter. - Ptσ and Ps are the target and reference distributions used for comparison.

∥x − x′∥2 2σ2

∆ = Ex,x′∼P exp −

∥y − y′∥2 2σ2 − 2Ex∼P,y∼Q exp −

+ Ey,y′∼Q exp −

∥x − y∥2 2σ2

, (46)

In the above equation, ∆ is the Maximum Mean Discrepancy between two distributions, P and Q. Here, σ is a hyperparameter controlling the width of the Gaussian kernel, which affects the sensitivity of the kernel to the differences between samples from the distributions. The term ∥x − x′∥2 is the squared Euclidean distance between two samples, and P and Q represent the two distributions being compared.

R

R

1 R

1 R

Br(σ). (47)

A(σ) =

Ar(σ), B(σ) =

r=1

r=1

In this equation, A(σ) and B(σ) are the average estimates of A(σ) and B(σ), computed over R independent samples. Each sample, Ar(σ) and Br(σ), is calculated for the r-th sample from the dataset. Here, R represents the total number of samples used in the averaging process.

- A(σ) =

A(σ) − minτ∈[0,∞) A(τ) maxτ∈[0,∞) A(τ) − minτ∈[0,∞) A(τ) + ϵ

, (48)

- B(σ) =

B(σ) − minτ∈[0,∞) B(τ) maxτ∈[0,∞) B(τ) − minτ∈[0,∞) B(τ) + ϵ

. (49)

The equations above represent the normalized versions of

- A(σ) and B(σ), denoted as A(σ) and B(σ), respectively. The normalization is done to scale the values of A(σ) and
- B(σ) to a range between 0 and 1. The small constant ε is added to the denominator to avoid division by zero and ensure numerical stability.

J(σ;λ) = λ A(σ) + 1 − λ B(σ), λ ∈ [0,1].

(50)

The function J(σ;λ) is the weighted trade-off between A(σ) and B(σ), controlled by the parameter λ. This trade-off allows us to balance the importance of the two terms, with λ taking values in the range [0,1].

σJ⋆ = arg min

J(σ;λ). (51)

σ∈[0,∞)

Finally, the optimal σJ⋆ is selected by minimizing the tradeoff function J(σ;λ) over the range σ ∈ [0,∞). The goal is to find the value of σ that minimizes the trade-off between A(σ) and B(σ), optimizing the performance based on the specific task.

#### B. Experiment Settings and Dataset

Experiment Settings. All experiments are conducted on NVIDIA A6000 48GB GPUs. The network is optimized with L2 loss. Data augmentation includes random horizontal and vertical flips for all tasks and histogram equalization for low-light images. During training, 256 × 256 patches are randomly cropped from

[Figure 20]

Figure 7. Mindmap of experiments and corresponding datasets. Table 4. Details for All-in-One image restoration benchmarks.

##### B.1. All-in-One-3 & Low Light

In this section, we introduce the datasets we used for experiment “4.3.Performance on Limited Training Data”.

Task Training Dataset Size Testing Dataset Size

WED+BSD400 [2, 41] 5,144 CBSD68 [2] 68

All-in-One-3

RESIDE-OTS [26] 72,135 SOTS [26] 492 Rain100L [49] 200 Rain100L 100

All-in-One-3. All-in-One-3 is a common setting for unified multiple image restoration tasks training, which contains “Noise+Haze+Rain”.

GoPro [44] 2,111 GoPro 1,111

WED+BSD400 [2, 41] 5,144 CBSD68 [2] 68 LoLV1 [61] 485 LoLV1 15

All-in-One-5

Image dehazing: For image dehazing, we use the outdoor synthesis dataset of RESIDE-OTS [26] with 72,135 pairs for training and SOTS-Outdoor [26] dataset for testing with 492 image pairs. These pairs are captured in realworld outdoor scenes and are specifically designed to benchmark the performance of dehazing algorithms under varying weather and lighting conditions, ensuring effective evaluation of dehazing performance across diverse environments.

RESIDE-OTS [26] 72,135 SOTS [26] 492 Rain100L [49] 200 Rain100L 100

UC-Merced [45] 17,010 UC-Merced 630 WED+BSD400 46,296 CBSD68 204 BrainWeb [21] 4,689 BrainWeb-test 174

MNMD

CDD-11 CDD-11 [15] 20,790 CDD-11 2,310 Low-Light

LoLV1 [61] 485 LoLV1 15

LOL-VE [49] 400 LOL-VE 100

Image deraining: we use the Rain100L [49] dataset with 200 pairs of images for training and 100 pairs for testing. Detailed introduction to Rain100L is provided in Appendix B.7.

augmented images as network inputs. Unless otherwise specified, the batch size is set to 8, the initial learning rate is 8e-5, and the total number of training steps is 300,000.

Image denoising: We conduct training using a merged dataset of BSD400 [2] and WED [41] with 400 and 4,744 clear images, respectively. Noisy images are generated with Gaussian noise (σ = (15, 25, 50)). Testing is performed on CBSD68 [2] datasets with 68 samples. Detailed introduction of BSD400 [2] and WED [41] is provided in Appendix B.8.

Model Architecture. Our residual removal model sets the basic U-Net [51] architecture and the hyper-parameters are as follows: C = 64, channel multiplier = (1,2,4,8). Our denoising model follows the U-Net architecture in [12], and the parameters are as follows C = 128, channel multiplier = (1,1,2,2,4,4).

Low-Light. Following previous image restoration settings [58], we combine data samples in LOLv1 and VE-LOLL [33] as Low-Light benchmark. VE-LOL-L: This dataset is specifically designed to benchmark low-light image enhancement techniques. The VELOLL dataset consists of 400 synthetically paired images for training and 100 paired images for testing, with each pair consisting of a low-light image and its corresponding well-exposed reference image. LOLv1: See details in Appendix B.2.

Datasets. Overall dataset structures are displayed in Fig. 7. The detailed dataset introduction are provided later from B.1 to B.8.

Metrics. To comprehensively evaluate restoration performance, we adopt both distortion-based metrics (PSNR, SSIM [60]) and perceptual metrics (LPIPS [68], FID [16]). Distortion metrics assess pixel-level fidelity, while perceptual metrics compare feature space similarities to better reflect human visual perception. Following standard practice in All-in-One image restoration, all metrics are computed on the RGB channels of full-resolution images.

##### B.2. All-in-One-5

All-in-One-5 contains datasets from the aforementioned three-task setting(All-in-One-3) as well as additional datasets: GoPro [44] for motion deblurring, and LOLv1 [61]

for lowlight image enhancement. The overall dataset contains “Noise+Haze+Rain+Light+Blur” in total.

Image Deblurring. The GoPro [44] dataset is a standard benchmark for dynamic scene deblurring, created to enable supervised learning for blind deblurring models. The authors recorded high-frame-rate video and used the high-fps frames as sharp reference images. Instead of convolving with simple uniform kernels, they synthesized realistic, spatially varying motion blur by averaging consecutive sharp frames. The dataset release provides 2,111 blurry-sharp image pairs for training and 1,111 pairs for evaluation.

Low Light Enhancement. LOLv1 [61] is a standard supervised benchmark for low-light enhancement. It consists of paired low-light and well-exposed reference images. The authors collected these pairs by capturing the same scenes under low and normal lighting conditions using a variety of consumer cameras and phones. This process yielded real captures (not purely synthetic) that include realistic sensor noise and color shifts, making the dataset a robust resource. We use 485 image pairs from LoLV1 for training and 15 pairs for testing.

##### B.3. CDD-11

The CDD-11 (Composite Degradation Dataset) is a synthetic dataset designed for training and evaluating image restoration models under composite degradation conditions. CDD-11 was introduced in the OneRestore [15], from which highlights the importance of dealing with multiple degradation types simultaneously, such as low-light, haze, rain, and snow, rather than addressing each degradation type in isolation. The dataset consists of 11 different degradation conditions, including combinations like “low haze,” “haze rain,” “low rain,” and “low snow,” which containing 20,790 image pairs for training and 2,310 image pairs for testing in total. CDD-11 serves as a benchmark for evaluating models that aim to perform restoration under mixed degradation conditions, reflecting real-world scenarios more accurately than datasets with only single degradation types.

##### B.4. Multi-Noise and Multi-Domain

In this section, we introduce a novel denoising benchmark designed for the single task I2I in multi-domains. We name this new benchmark Multi-Noise and Multi-Domain (MNMD). As shown in Tab. 4, MNMD consists of image pairs from various sources, including 46,296 natural images (from WED and BSD400 [49]), 17,010 remote sensing images (from UC-Merced [45]), and 4,689 medical images (from BrainWeb [7, 21, 22]). To better simulate noise-related degradations under different conditions, we introduce three types of noise, each with multiple intensity levels: Gaussian noise is added as x′ = x + n, where n ∼ N(0,σ2). The parameter σ = (15,25,50) controls the standard deviation, reflecting the noise strength; Salt-and-Pepper noise is applied by

randomly selecting a fraction d (scale 0.014, 0.039, 0.154) of pixels and replacing each with either the minimum or maximum possible value, thereby simulating random pixel corruption; and Poisson noise is simulated by replacing each pixel with a value drawn from a Poisson distribution whose mean is x × peak, with x being the original pixel value and peak (26, 102, 283) acting as a noise scaling factor.

For evaluation, we use 204 noisy natural images from CBSD68 [2], as well as 630 and 174 images from UCMerced and BrainWeb, respectively. The test images are degraded with Gaussian noise (σ = 15), salt-and-pepper noise (density = 0.039), and Poisson noise (scale = 102). This setup forms a comprehensive test set for evaluating denoising performance across multiple domains and noise types.

The UC-Merced Land Use Dataset (UC-Merced [45] is a widely-used remote sensing image dataset designed for land use scene classification, featuring 21 distinct categories of US urban and semi-urban scenes. Each category contains 100 images, resulting in a total of 2,100 images in the dataset. The images have a resolution of 256×256 pixels and were manually extracted from the United States Geological Survey (USGS) National Map Urban Area Imagery collections for various US urban areas. The medical images are stem from BrainWeb [7, 21, 22], which is a website for synthesizing medical images. Detailed introduction of BSD400 [2] and WED [41] is provided in Appendix B.8.

##### B.5. Image Inpainting

The CelebA-HQ [19] (CelebFaces High Quality) dataset is a high-resolution version of the CelebA dataset, specifically designed for training and evaluating face-related image processing tasks, including image inpainting. CelebA-HQ consists of 30,000 high-resolution facial images, each with a resolution of 256x256 pixels. The dataset includes a wide variety of celebrity faces with various attributes such as age, gender, and facial expressions, making it suitable for tasks like face generation and image inpainting. For image inpainting tasks, we adopt the CelebA-HQ dataset for both training and testing, with our training set containing 28,000 image pairs and our test set consisting of 2,000 image pairs. Each image in the dataset is paired with a mask that specifies the region to be inpainted, allowing models to learn to fill in missing parts of the face.

##### B.6. Super-Resolution

For super-resolution, we adopt the FFHQ [20] dataset for training and testing. FFHQ (Flickr-Faces-HQ) dataset is a high-quality facial image dataset consists of 70,000 samples, which is primarily used for computer vision and deep learning research, particularly in applications like facial image generation, editing, and expression recognition. Released by NVIDIA in 2018, the dataset aims to provide a high-

resolution standard for facial image generation and recognition tasks. In our experiment, we input 16x16 images and output high-resolution 128x128 images. During training, we employ bicubic interpolation to upsample the 16x16 input images to 128x128, using the upsampled images as input for training.

##### B.7. Deraining

For deraining, we adopt the Rain100 dataset [64] for both training and testing. Rain100 consists of two subsets, Rain100H (Heavy Rain) and Rain100L (Light Rain), each containing paired rainy and ground-truth clean images. The full dataset contains 2,200 image pairs: 300 for Rain100L and 1,900 for Rain100H. For our experiments, we use 100 image pairs from each subset for testing and the remaining pairs for training.

##### B.8. Denoising

For image denoising, we leverage two widely used datasets: BSD400 [2] and WED [41] for training, while CBSD68 [2] and Urban100 [18] for testing. Our training set consists of 5,144 image pairs (400 from BSD400 and 4,744 from WED), while our testing set consists of 168 image pairs (68 from CBSD68 and 100 from Urban100).

Training datasets. We use BSD400 and WED to build our training set. BSD400 is a widely used dataset of 400 natural images, usually adopted as a training set in imagerestoration research. The WED (Waterloo Exploration Database) is a much larger collection created for imagequality assessment, containing thousands of diverse, highquality natural images. Combining WED with BSD400 provides a larger and more varied pool of clean images, which is essential for training models that can generalize well to diverse real-world noise. Many recent denoising pipelines [8, 48, 66] merge these two datasets to expand their training data, and we follow this paradigm to utilize BSD400+WED as our training set.

Testing datasets. Our models are evaluated on two challenging test sets, CBSD68 and Urban100. CBSD68 is the color version of the BSD68 test set, a benchmark of 68 natural images commonly used to evaluate color-image denoising. Urban100 is a 100-image dataset of high-resolution urban scenes, initially developed for single-image super-resolution but now widely adopted as a challenging benchmark for denoising. Urban scenes often feature strong, repeating geometric patterns (like bricks and windows) that are difficult to reconstruct, making Urban100 a robust test for a model’s ability to recover fine structural detail.

#### C. Additional Experiments

##### C.1. Implementation Details of Methods in Experiments

In this section, we introduce the training details of our comparison methods.

DRDD(Ours). Without specific mentioned, training settings follow experiments setting details in Appendix B. In Experiment “4.3.Performance on Limited Training Data”. We start training denoising U-NET from pre-trained parameters, where C = 256, channel multiplier = (1,1,2,2,4,4).

RDDM. [34] All experimental settings are kept consistent with those in the paper for both image deraining and image inpainting tasks.

DiffuIR. It [69] uses a selective hourglass mapping strategy within a diffusion model to handle multiple image restoration tasks with a single, efficient model. All experimental settings are kept consistent with those in the paper.

AdAIR [9] adaptively restores images suffering from various degradations by mining and modulating frequency components, enabling unified and effective all-in-one image restoration. For training stage, it is conducted with a batch size of 32 in the all-in-one setting and a batch size of 8 in the single-task setting. The model is trained on cropped image patches of size 128 × 128 pixels for 150 epochs, which is approximately equivalent to 300,000 steps in DRDD.

DA-CLIP [39] employs an image controller to identify degradation types and adaptively restore images affected by diverse distortions, thereby providing a unified and effective solution for multi-task image restoration. For training stage, we use a batch size of 16 in training for All-in-One-5 task. The model is trained for 300,000 steps on cropped image patches of size 256 × 256 pixels.

DFPIR. [57] It uses a unified model with a degradationaware feature perturbation mechanism, which introduces channel-wise and attention-wise perturbations to mitigate task interference. We follow the experiment settings in the paper. IR-SDE [38] adaptively restores images suffering from various degradations by modeling the degradation process with mean-reverting stochastic differential equations, enabling unified image restoration. For training stage, all tasks are trained with a batch size of 8 for a total of 500,000 steps. For the image inpainting task, the model is trained on cropped image patches of size 64 × 64 pixels, whereas for the other tasks, is trained on 128 × 128 pixels.

De-IRSDE is a decoupling version of IRSDE [38]. All experimental settings are kept consistent with those of IRSDE [38].

VLUNET [67] adaptively restores images suffering from various degradations by leveraging vision-language models to automatically select degradation-specific transforms, enabling unified and effective all-in-one image restoration within a deep unfolding framework. For training stage, we

- Table 5. Comparison of single task image restoration approaches on deraining and denoising with two metrics (SSIM / LPIPS). Best results are highlighted in Bold.

(a) Deraining Results

Method

Rain100H Rain100L SSIM / LPIPS SSIM / LPIPS

RDDM* .8806 / .1170 .9432 / .0250 IRSDE* .9041 / .0470 .9805 / .0140 DRDD (Ours) .9375 / .0400 .9839 / .0180

(b) Denoising on CBSD68 and Kodak24 with σ ∈ {15, 25, 50}

Method

CBSD68 Kodak24 σ=15 σ=25 σ=50 σ=15 σ=25 σ=50 SSIM / LPIPS SSIM / LPIPS SSIM / LPIPS SSIM / LPIPS SSIM / LPIPS SSIM / LPIPS

AdAIR .9340 / .0534 .8898 / .0967 .8003 / .1895 .9269 / .0712 .8841 / .1114 .8036 / .1966 VLUNet .9341 / .0568 .8902 / .0984 .8039 / .1938 .9270 / .0745 .8838 / .1146 .8079 / .2040 DRDD (Ours) .9346 / .0502 .8920 / .0877 .8013 / .1750 .9274 / .0680 .8879 / .1032 .8047 / .1880

- Table 6. Performance comparison of inpainting methods at 256×256 (center), 256×256 (irregular), and 64×64 (center) resolutions. Best results and the second-best results are highlighted in red and blue.

Method

256×256 (Center) 256×256 (Irregular) 64×64 (Center)

LPIPS↓ FID↓ LPIPS↓ FID↓ LPIPS↓ FID↓ RDDM [34] 0.0862 15.67 0.0963 8.52 0.0568 14.75 CTSDG [14] 0.0798 11.64 0.0722 7.87 0.0498 15.68 MISF [29] 0.0764 11.72 0.0803 7.68 0.0695 20.43 TransRef [35] 0.0745 9.13 0.0692 7.80 0.0490 10.17 DRDD(Ours) 0.0542 10.30 0.0528 7.15 0.0382 10.02

- Table 7. Performance comparison of RDDM and DRDD on Edges2Handbags and Edges2Shoes dataset.

Table A2

Edges2Handbags-256×256 Edges2Shoes-256×256 SSIM↑ FID↓ LPIPS↓ SSIM↑ FID↓ LPIPS↓

RDDM 0.645 5.72 0.256 0.652 23.57 0.178 DRDD 0.723 4.76 0.247 0.782 9.658 0.154

use a batch size of 8 in training for all tasks. The model is trained for 200 epochs on cropped image patches of size 128 × 128 pixels, which is roughly equivalent to 400,000 steps in DRDD.

TransRef. [35] For training stage, we use a batch size of

- 8 for training across all tasks. The model is trained on the original image pixels for 400,000 steps.

Figure 8. Visual comparison of RDDM and DRDD on Edges2Handbags dataset.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Input RDDM DRDD (Ours)

Furthermore, we conducted experiments at both 64×64 and 256×256 resolutions to further confirm the model’s adaptability to different input scales. As shown in Tab. 6, our method achieved the best or second-best results across all configurations, demonstrating its superior performance and robustness under diverse task conditions.

Image Deraining. We compare DRDD’s performance on image deraining through Rain100H and Rain100L datasets with RDDM [34] and DiffuIR [69]. Results are shown in Tab. 5.

Edges to Objects. As shown in Tab. 7 and Fig. 8, we evaluated DRDD on style transfer task using edges2handbags & edges2shoes datasets. DRDD significantly outperforms its baseline RDDM. Visual results confirm that the injected noise does not adversely affect DRDD.

Image Denoising. We compare DRDD’s performance on image denoising through BSD400 [2] and WED [41] with AdaIR[9] and VLUNet [67]. Results are shown in Tab. 5.

CTSDG. [14] For training stage, we use a batch size of 4 for training across all tasks. The model is trained on cropped image patches of size 128 × 128 pixels for 150,000 steps.

##### C.2. More Single I2I Tasks in Single Domain.

Image Inpainting. To validate the generalization capability of the proposed framework across different tasks, we conducted inpainting experiments under various settings. Specifically, we evaluated the inpainting performance at different resolutions using two mask patterns (center and irregular) on the CelebA-HQ [19] dataset. It is worth noting that irregular masks more closely reflect the restoration demands of complex occlusions encountered in real-world scenarios, whereas rectangular masks are typically used to simulate cases in which local image information is entirely missing.

Super Resolution. To further demonstrate the generalization ability of our method across different tasks, we conducted a qualitative experiment on the FFHQ [20] dataset to evaluate DRRD’s performance in the image super-resolution task.

##### C.3. Ablation Studies.

We compare DRDD against a matched “coupled” baseline with the same architecture. Although only a single neural network is used, we assign it the combined number of parameters of two decoupled neural networks, along with the

Table 8. Ablation Studies and Further Investigaitons on All-in-One-5 daytaset. SSIM (↑) and LPIPS (↓) are reported on the full RGB images. The best performances are highlighted.

Dehazing Deraining Denoising Deblurring Low-Light

Method

Average SOTS Rain100L BSD68σ=25 GoPro LOLv1

DRDD w.o Denoising Network .9690 .0149 .9706 .0244 .8771 .1136 .8241 .1923 .8469 .1251 .8971 .0941 DRDD with General Denoising Network .9652 .0145 .9701 .0248 .8867 .1054 .8712 .1748 .8490 .1232 .9084 .0885 Entangled Baseline .9601 .0136 .9727 .0231 .8818 .1153 .8405 .1849 .8283 .1555 .8966 .0985 DRDD (Ours) .9715 .0122 .9777 .0153 .8891 .0977 .8806 .1342 .8640 .1033 .0916 .0762

double cumulative inference steps. We test it on All-in-One5 benchmark and results are displayed in Tab. 8.

##### C.4. Further Investigations of Denoising Model

Investigating the Training of Denoising Models on Isolated Datasets. Based on the properties we previously discussed, the denoising model can be trained on a isolate dataset. To further validate this, We train the denoising model for 300,000 steps on the All-in-One-3 dataset and combined it with the residual removal model, which was trained for 300,000 steps on the All-in-One-5 dataset. The results in Tab. 8 show that despite the denoising model never having encountered low-light or deblurring data, it still learned a certain level of generalized denoising capability from the other datasets. Such a strategy can significantly reduce the number of training-required parameters.

Investigating Inference Without Denoising Models. In our decoupled model, the residual removal model is responsible for directional semantic elimination in the noise domain, while the denoising model is tasked with translating the data back to the noise-free domain and performing refined restoration. The necessity for a dedicated denoising model, rather than simply subtracting the noise added in the first step, arises because during directional semantic transformation, the residual removal model also exerts some influence on the noise. This perspective has been confirmed by experiment results in Tab. 8, DRDD without Denoising Network.

##### C.5. Cost and Efficiency

###### C.5.1. Parameters, Flops and Inference Time

In this section, we present the key performance metrics of our model. The floating-point operations (FLOPs) were computed by performing a forward inference using a randomly generated matrix of size 256×256×3. For measuring inference time, the model was trained on the AIOIR5 architecture and evaluated on the Rain100L dataset. The reported inference time was obtained under the condition where the model runs for two steps on each of its two sub-modules: residual removing and denoising.

###### C.5.2. Influence of sampling steps

We conducted experiments to investigate the influence of different sampling steps on the results using the All-in-One-3 dataset. The results, shown in Tab. 10, indicate that there is

- Table 9. Computational resource comparisons: Parameters, FLOPs, and Runtime. FLOPs are measured on the patch size of 256 × 256 × 3, while Runtime are measured on Rain100L testing set. The sampling steps of denoising model and residual-removal model are both 2.

| |Method|Para (M) FLOPs (G) Step<br><br>|Latency (s)<br><br>|SSIM↑ LPIPS↓ FID↓|
|---|---|---|---|---|
| |AdAIR VLUNet|29 138 1 123 143 1<br><br>|0.24 0.74<br><br>|0.909 0.089 26.1 0.904 0.096 27.9|
|Diff.|DiffUIR<br><br>DA-CLIP<br><br>RDDM - S RDDM - L<br><br>DRDD - S DRDD - L<br><br>|138 284 3 174 380 3 138 278 4<br><br>138×2 278×2 4<br><br>7+35 32+69 2 + 2 7+138 32+278 2 + 2<br><br>|0.75 0.76 0.92 1.84 0.10+0.23 0.10+0.45<br><br>|0.869 0.117 33.7 0.876 0.108 20.0 0.878 0.105 27.8 0.897 0.098 23.6 0.908 0.080 19.6 0.916 0.073 18.3<br><br>|

- Table 10. Performance of different sampling steps. SSIM (↑) and LPIPS (↓) are reported.

Sampling Steps Dehazing Denoise Deraining

de-res de-noise SSIM↑ LPIPS↓ SSIM↑ LPIPS↓ SSIM↑ LPIPS↓ 2 2 0.9787 0.0104 0.8919 0.0929 0.9767 0.0149 2 5 0.9790 0.0103 0.8900 0.0940 0.9774 0.0146 5 2 0.9791 0.0101 0.8918 0.0930 0.9767 0.0149 5 5 0.9794 0.0100 0.8899 0.0940 0.9774 0.0146

10 10 0.9792 0.0101 0.8838 0.0951 0.9777 0.0144

- Table 11. Performance comparison between our method and other universal image restoration methods on real unseen datasets. Best results are highlighted in red, while the second-best results are blue.

Low-Light Deraining Denoising Deblurring

Method

NIQE↓ NIQE↓ PSNR↑ / SSIM↑ / LPIPS↓ PSNR↑ / SSIM↑ / LPIPS↓

VLUNet 4.40 3.73 24.6 / 0.490 / 0.664 26.8 / 0.822 / 0.175 DiffuIR 3.89 4.49 28.6 / 0.674 / 0.569 28.4 / 0.857 / 0.186 DRDD(Ours) 4.31 3.80 34.8 / 0.865 / 0.274 28.5 / 0.861 / 0.172

no significant difference between 2-step and 10-step inference when using DDIM sampling.

##### C.6. Generalization Ability

To further validate the proposed method’s generalization capability on unseen data distribution, we evaluated its performance on unseen data across four representative image restoration tasks: low-light enhancement, deraining, denoising, and deblurring. The corresponding evaluation metrics are presented in Table 11. Specifically, the low-light enhancement task was conducted using a combined dataset comprising MEF [40], NPE [59], and DICM [25]; the deraining task employed the Practical [64] dataset; the denoising task utilized the SIDD [1] dataset; and the deblurring task

was assessed using the RealBlur [50] dataset.

##### C.7. PSNR results

We provide the PSNR results of All-in-One-5 model as below. As shown in Tab. 1 of the paper, DRDD consistently achieves the best FID, LPIPS among all the methods, the best PSNR, SSIM among diffusion-based approaches, while remaining highly competitive on PSNR, SSIM with nondiffusion methods.

|Table C.7<br><br>|Task Dataset<br><br>|Low-Light Deraining Denoising Deblurring Dehazing LOLv1 Rain100L CBSD68 GoPro SOTS|
|---|---|---|
|Non-Diff.<br><br>|DFPIR AdAIR VLU-NET<br><br>|23.80 37.50 31.26 28.80 31.24 22.94 37.85 31.29 28.11 29.93 22.25 38.36 31.39 28.85 30.56|
|Diff.<br><br>|DiffuIR DA-CLIP DRDD (Ours)|19.33 34.88 30.12 26.48 30.27<br><br>20.12 35.86 25.17 27.34 26.88<br><br><br>23.00 36.86 31.47 29.08 30.56|

##### C.8. Comparing data pruning results with more methods

We provide comparison with AdaIR [9] and RDDM [34] on pruned All-in-One-3 dataset, SSIM and LPIPS are reported.

|Table C.8<br><br>|25% 50% 75% 100%|
|---|---|
| |SSIM↑ / LPIPS↓ SSIM↑ / LPIPS↓ SSIM↑ / LPIPS↓ SSIM↑ / LPIPS↓|
|RDDM AdAIR<br><br>|.929 / .058 .935 / .056 .941 / .049 .942 / .047 .944 / .050 .948 / .046 .949 / .045 .951 / .042|
|DRDD (Ours)<br><br>|.947 / .041 .949 / .040 .950 / .039 .951 / .038|

##### C.9. Sensitivity Analysis of Noise Injection Level

Noise Injection level is relatively insensitive within the range of 0.8–1.3 (calculated via Eq. 12 across different datasets), as validated by Fig. 6 on the All-in-One-5 dataset and Table A5 (PSNR metric) on the Rain100H and Edges2Bags datasets. Outside this 0.8–1.3 range, sensitivity increases. If optimal performance is required, experimentation in 0.8–1.3 range is necessary.

|Table C.9<br><br>|Recommend σ|0.1|0.5|0.8<br><br>|1.0<br><br>|1.5<br><br>|2.0|
|---|---|---|---|---|---|---|---|
|Rain100H Edges2Bags<br><br>|0.8-1.2 0.8-1.1<br><br>|29.64 16.34|30.97 19.22<br><br>|32.01 20.05|32.33 19.81<br><br>|31.94 18.76<br><br>|31.68 17.92|

#### D. More Visual Comparisons

As shown in Fig. 9 - Fig.10, we provide additional visual examples of DRDD on various image-to-image transformation tasks, including restoration and inpainting. These results serve as supplementary visualizations to those presented in the main paper.

[Figure 30]

- (a)
- (b)
- (c)

Haze Image VLU-NET DiffUIR DFPIR DRDD(Ours) GroundTruth

[Figure 31]

Noise Image VLU-NET DiffUIR DFPIR DRDD(Ours) GroundTruth

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

Low-Resolution DRDD(Ours) GroundTruth

Figure 9. Visual results of state-of-the-art methods and our proposed DRDD. (a) Comparison of haze image restoration results on the SOTS dataset [26]. (b) Comparison of noise restoration results on the CBSD68 dataset [2]. (c) Super-Resolution result in FFHQ [20]. Zoom in for best view.

[Figure 44]

[Figure 45]

[Figure 46]

Irregular Mask CTSDG MISF

[Figure 47]

[Figure 48]

[Figure 49]

TransRef DRDD(Ours) GroundTruth

Figure 10. Irregular Mask inpainting results of state-of-the-art methods and our proposed DRDD. Zoom in for best view.

