### ReNoise: Real Image Inversion Through Iterative Noising

## arXiv:2403.14602v1[cs.CV]21Mar2024

Daniel Garibi1 Or Patashnik1 Andrey Voynov2 Hadar Averbuch-Elor1 Daniel Cohen-Or1 1Tel Aviv University 2Google Research

https://garibida.github.io/ReNoise-Inversion/

Inversion Edits

SDXLTurbo

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Input Image DDIM Inversion ReNoise Inversion “Dog Made of Lego” “Dog” -> “Poodle” “… in the snow

“Neon art of…”

Inversion Edits

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

LCM-LoRA

Input Image DDIM Inversion ReNoise Inversion “Crochet racoon” “Raccoon” -> “Koala”“Raccoon” -> “Panda”

“… in the snow”

Figure 1. Our ReNoise inversion technique can be applied to various diffusion models, including recent few-step ones. This figure illustrates the performance of our method with SDXL Turbo and LCM models, showing its effectiveness compared to DDIM inversion. Additionally, we demonstrate that the quality of our inversions allows prompt-driven editing. As illustrated on the right, our approach also allows for prompt-driven image edits.

#### Abstract

Recent advancements in text-guided diffusion models have unlocked powerful image manipulation capabilities. However, applying these methods to real images necessitates the inversion of the images into the domain of the pretrained diffusion model. Achieving faithful inversion remains a challenge, particularly for more recent models trained to generate images with a small number of denoising steps. In this work, we introduce an inversion method with a high quality-to-operation ratio, enhancing reconstruction accuracy without increasing the number of operations. Building on reversing the diffusion sampling process, our method employs an iterative renoising mechanism at each inversion sampling step. This mechanism refines the approximation of a predicted point along the forward diffusion trajectory, by iteratively applying the pretrained diffusion model, and averaging these predictions. We evaluate the performance of our ReNoise technique using various sampling algorithms and models, including recent accelerated diffusion models. Through comprehensive evaluations and comparisons, we show its effectiveness in terms of both accuracy and speed. Furthermore, we confirm that our method preserves editability by demonstrating text-driven image editing on real images.

#### 1. Introduction

Large-scale text-to-image diffusion models have revolutionized the field of image synthesis [20, 38, 40, 41]. In particular, many works have shown that these models can be employed for various types of image manipulation [6– 9, 15, 17, 23, 30, 35, 36, 48]. To edit real images, many of these techniques often require the inversion of the image into the domain of the diffusion model. That is, given a real image z0, one has to find a Gaussian noise zT, such that denoising zT with the pretrained diffusion model reconstructs the given real image z0. The importance of this task for real image manipulation has prompted many efforts aimed at achieving accurate reconstruction [16, 21, 31, 32].

The diffusion process consists of a series of denoising steps {ϵθ(zt,t)}1t=T, which form a trajectory from the Gaussian noise to the model distribution (see Figure 2). Each denoising step is computed by a trained network, typically implemented as a UNet, which predicts zt−1 from zt [20]. The output of the model at each step forms a direction from zt to zt−1 [45]. These steps are not invertible, in the sense that the model was not trained to predict zt from zt−1. Thus, the problem of inverting a given image is a challenge, and particularly for real images, as they are not necessarily in the model distribution (see Figure 3).

Data

Gaussian

Distribution

Distribution

|𝑧0|
|---|

𝑧𝑡−1

𝑧𝑡

|𝑧𝑇|
|---|

|𝑧𝑡+1|
|---|

Figure 2. The diffusion process samples a Gaussian noise and iteratively denoises it until reaching the data distribution. At each point along the denoising trajectory, the model predicts a direction, ϵθ(zt), to step to the next point along the trajectory. To invert a given image from the distribution, the direction from zt to zt+1 is approximated with the inverse of the direction from zt to zt−1 denoted by a dotted blue line.

In this paper, we present an inversion method with a high quality-to-operation ratio, which achieves superior reconstruction accuracy for the same number of UNet operations. We build upon the commonly used approach of reversing the diffusion sampling process, which is based on the linearity assumption that the direction from zt to zt+1 can be approximated by the negation of the direction from zt to zt−1 [12, 44] (see Figure 2). To enhance this approximation, we employ the fixed-point iteration methodology [10]. Specifically, given zt, we begin by using the common approximation to get an initial estimate for zt+1, denoted by zt(0)+1. Then, we iteratively renoise zt, following the direction implied by zt(+1k) to obtain zt(+1k+1). After repeating this renoising process several times, we apply an averaging on zt(+1k) to form a more accurate direction from zt to zt+1.

We show that this approach enables longer strides along the inversion trajectory while improving image reconstruction. Therefore, our method can also be effective with diffusion models trained to generate images using a small number of denoising steps [28, 43]. Furthermore, despite the need to repeatedly renoise in each step of the inversion process, the longer strides lead to a more favorable tradeoff of UNet operations for reconstruction quality.

Through extensive experiments, we demonstrate the effectiveness of our method in both image reconstruction and inversion speed. We validate the versatility of our approach across different samplers and models, including recent time-distilled diffusion models (e.g., SDXL-Turbo [43]). Importantly, we demonstrate that the editability of the inversion achieved by our method allows a wide range of textdriven image manipulations (see Figure 1).

#### 2. Related Work

Image Editing via Diffusion Models Recent advancements in diffusion models [12, 20] have resulted in unprecedented diversity and fidelity in visual content creation guided by free-form text prompts [37, 38, 40, 41]. Textto-image models do not directly support text-guided image editing. Therefore, harnessing the power of these models

Original Image DDIM Inversion + 1 ReNoise Step

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Figure 3. Comparing reconstruction results of plain DDIM inversion (middle column) on SDXL to DDIM inversion with one ReNoise iteration (rightmost column).

for image editing is a significant research area and many methods have utilized these models for different types of image editing [6–9, 11, 14, 15, 17, 18, 23, 30, 35, 36, 48, 49]. A common approach among these methods requires inversion [21, 32, 44, 50] to edit real images, i.e., obtaining a latent code zT such that denoising it with the pretrained diffusion model returns the original image. Specifically, in this approach two backward processes are done simultaneously using zT. One of the processes reconstructs the image using the original prompt, while the second one injects features from the first process (e.g., attention maps) to preserve some properties of the original image while manipulating other aspects of it.

Inversion in Diffusion Models Initial efforts in image inversion for real image editing focused on GANs [2– 5, 13, 34, 39, 47, 53–55]. The advancements in diffusion models, and in diffusion-based image editing in particular have recently prompted works studying the inversion of a diffusion-based denoising process. This inversion depends on the sampler algorithm used during inference, which can be deterministic [44] or non-deterministic [20, 22]. Inversion methods can be accordingly categorized into two: methods that are suitable for deterministic sampling, and methods suitable for non-deterministic sampling.

Methods that approach the deterministic inversion commonly rely on the DDIM sampling method [44], and build upon DDIM inversion [12, 44]. Mokady et al. [32] observed that the use of classifier-free guidance during inference magnifies the accumulated error of DDIM inversion and therefore leads to poor reconstruction. Following this observation, several works [16, 31, 32] focused on solving this issue by replacing the null text token with a different embedding, which is found after an optimization process or by a closed solution. However, excluding [32] which requires a lengthy optimization, these methods are limited by

×𝑇 𝑆𝑡𝑒𝑝𝑠

[Figure 23]

UNET

𝛿 ( )

[Figure 24]

𝑧 ( )

[Figure 25]

𝛿 ( )

UNET

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

𝑧 (   )

𝛿 (   )

[Figure 33]

Enhance Editability

UNET

[Figure 34]

𝑧 𝑧

𝑧 𝑧

Average

[Figure 35]

𝑧 ( )

𝛿 ( )

[Figure 36]

Enhance Editability

UNET

[Figure 37]

- Figure 4. Method overview. Given an image z0, we iteratively compute z1, ..., zT, where each zt is calculated from zt−1. At each time step, we apply the UNet (ϵθ) K +1 times, each using a better approximation of zt as the input. The initial approximation is zt−1. The next one, zt(1), is the result of the reversed sampler step (i.e., DDIM). The reversed step begins at zt−1 and follows the direction of ϵθ(zt−1, t). At the k renoising iteration, zt(k) is the input to the UNet, and we obtain a better zt approximation. For the lasts iterations, we optimize ϵθ(zt(k), t) to increase editability. As the final denoising direction, we use the average of the UNet predictions of the last few iterations.

ate high-quality images, recent accelerated models achieve high-quality synthesis with 1-4 steps only. These new methods pave the way for interactive editing workflows. However, as we show in this paper, using current methods for the inversion of an image with a small number of steps degrades the reconstruction quality in terms of accuracy [12, 44] or editability [21, 51].

the reconstruction accuracy of DDIM inversion, which can be poor, especially when a small number of denoising steps is done. In our work, we present a method that improves the reconstruction quality of DDIM inversion and therefore can be integrated with methods that build on it.

Another line of work [21, 51] tackles the inversion of DDPM sampler [20]. In these works [21, 51], instead of inverting the image into an initial noise zT, a series of noises {zT,ϵT,...,ϵ1} is obtained. The definition of this noises series ensures that generating an image with it returns the original input image. However, these methods require a large number of inversion and denoising steps to allow image editing. Applying these methods with an insufficient number of steps leads to too much information encoded in {ϵT,...,ϵ1} which limits the ability to edit the generated image. As shall be shown, The editability issue of these methods is particularly evident in few-steps models [27, 28, 43].

#### 3. Method

##### 3.1. ReNoise Inversion

Reversing the Sampler Samplers play a critical role in the diffusion-based image synthesis process. They define the noising and denoising diffusion processes and influence the processes’ trajectories and quality of the generated images. While different samplers share the same pre-trained UNet model (denoted by ϵθ) as their backbone, their sampling approaches diverge, leading to nuanced differences in output. The goal of the denoising sampler is to predict the latent code at the previous noise level, zt−1, based on the current noisy data zt, the pretrained UNet model, and a sampled noise, ϵt. Various denoising sampling algorithms adhere to the form:

Most relevant to our work, two recent inversion methods [29, 33] also use the fixed-point iteration technique. Specifically, they improve the reconstruction accuracy of DDIM inversion [44] with Stable Diffusion [40] without introducing a significant computational overhead. In our work, we focus on the problem of real image inversion for recently introduced few-step diffusion models, where the difficulties encountered by previous methods are amplified. Furthermore, we show that our inversion method successfully works with various models and different samplers.

zt−1 = ϕtzt + ψtϵθ(zt,t,c) + ρtϵt, (1)

where c represents a text embedding condition, and ϕt, ψt, and ρt denote sampler parameters. At each step, these parameters control the extent to which the previous noise is removed (ϕt), the significance assigned to the predicted noise from the UNet (ψt), and the given weight to the additional noise introduced (ρt).

Few Steps Models Recently, new methods [27, 28, 42, 43, 46] that fine-tune text-to-image diffusion models enabled a significant reduction of the number of steps needed for high-quality image generation. While standard diffusion models typically require 50 denoising steps to gener-

[Figure 38]

|𝑧𝑡−2|
|---|

𝜖𝜃(𝑧𝑡−1,𝑡)

|𝑧𝑡−1|
|---|

𝜖𝜃(𝑧𝑡1 ,𝑡)

[Figure 39]

−𝜖𝜃(𝑧𝑡−1,𝑡)

[Figure 40]

−𝜖𝜃(𝑧𝑡1 ,𝑡) Known Latent

[Figure 41]

|𝑧𝑡(1)|
|---|

[Figure 42]

Estimations of next Latent

[Figure 43]

Target Latent

[Figure 44]

Denoising Step

[Figure 45]

|𝑧𝑡|
|---|

Corresponding Inverse Step

|𝑧𝑡(2)|
|---|

- Figure 5. Geometric intuition for ReNoise. At each inversion step, we are trying to estimate zt (marked with a red star) based on zt−1. The straightforward approach is to use the reverse direction of the denoising step from zt−1, assuming the trajectory is approximately linear. However, this assumption is inaccurate, especially in few-step models, where the size of the steps is not small. We use the linearity assumption only as an initial estimation and keep improving the estimation. We recalculate the denoising step from the previous estimation (which is closer to zt) and then proceed with its opposite direction from zt−1 (see the orange vectors).

A given image z0 can be inverted by reformulating Equation 1 and applying it iteratively:

zt−1 − ψtϵθ(zt,t,c) − ρtϵt ϕt

, (2)

zt =

where for non-deterministic samplers, a series of random noises {ϵt}Tt=1 is sampled and used during both inversion and image generation processes. However, directly computing zt from Equation 2 is infeasible since it relies on ϵθ(zt,t,c), which, in turn, depends on zt, creating a circular dependency. To solve this implicit function, Dhariwal et al. [12] propose using the approximation ϵθ(zt,t,c) ≈ ϵθ(zt−1,t,c):

zt−1 − ψtϵθ(zt−1,t,c) − ρtϵt ϕt

zt(1) =

. (3)

This method has several limitations. First, the assumption underlying the approximation used in [12] is that the number of inversion steps is large enough, implying a trajectory close to linear. This assumption restricts the applicability of this inversion method in interactive image editing with recent few-step diffusion models [27, 28, 43, 46], as the inversion process would take significantly longer than inference. Second, this method struggles to produce accurate reconstructions in certain cases, such as highly detailed images or images with large smooth regions, see Figure 3. Moreover, we observe that this inversion method is sensitive to the prompt c and may yield poor results for certain prompts.

zt(1) zt(2)

zt(5)

zt

zt(4) zt(3)

Figure 6. Schematic illustration of the ReNoise convergence process to the true inversion of zt−1. While estimates may converge non-monotonically to the unknown target zt, we found that averaging them improves true value estimation. Typically, the initial iteration exhibits an exponential decrease in the norm between consecutive elements.

ReNoise In a successful inversion trajectory, the direction from zt−1 to zt aligns with the direction from zt to zt−1 in the denoising trajectory. To achieve this, we aim to improve the approximation of ϵθ(zt,t,c) in Eq. 2 compared to the one used in [12]. Building on the fixed-point iteration technique [10], our approach better estimates the instance of zt that is inputted to the UNet, rather than relying on zt−1.

Intuitively, we utilize the observation that zt(1) (from Eq. 3) offers a more precise estimate of zt compared to zt−1. Therefore, employing zt(1) as the input to the UNet is likely to yield a more accurate direction, thus contributing to reducing the overall error in the inversion step. We illustrate this observation in Figure 5. Iterating this process generates a series of estimations for zt, denoted by {zt(k)}Kk=1+1. While the fixed-point iteration technique [10] does not guarantee convergence of this series in the general case, in Section 4, we empirically show that convergence holds in our setting. However, as the convergence is not monotonic, we refine our prediction of zt by averaging several {zt(k)}, thus considering more than a single estimation of zt. See Figure 6 for an intuitive illustration.

In more detail, our method iteratively computes estimations of zt during each inversion step t by renoising the noisy latent zt−1 multiple times, each with a different noise prediction (see Figure 4). Beginning with zt(1), in the k-th renoising iteration, the input to the UNet is the result of the previous iteration, zt(k). Then, zt(k+1) is calculated using the inverted sampler while maintaining zt−1 as the starting point of the step. After K renoising iterations, we obtain a set of estimations {zt(k)}Kk=1+1. The next point on the inversion trajectory, zt, is then defined as their weighted average, where wk is the weight assigned to zt(k). For a detailed description of our method, refer to Algorithm 1.

##### 3.2. Reconstruction-Editability Tradeoff

Enhance Editability The goal of inversion is to facilitate the editing of real images using a pretrained image generation model. While the the renoising approach attains highly accurate reconstruction results, we observe that the resulting zT lacks editability. This phenomenon can be attributed

to the reconstruction-editability tradeoff in image generative models [47]. To address this limitation, we incorporate a technique to enhance the editability of our method.

It has been shown [35] that the noise maps predicted during the inversion process often diverge from the statistical properties of uncorrelated Gaussian white noise, thereby affecting editability. To tackle this challenge, we follow pix2pix-zero [35] and regularize the predicted noise at each step, ϵθ(zt,t,c), using the following loss terms.

First, we encourage ϵθ(zt,t,c) to follow the same distribution as ϵθ(zt′,t,c), where zt′ represents the input image z0 with added random noise corresponding to the noise level at timestep t. We do so by dividing ϵθ(zt,t,c) and ϵθ(zt′,t,c) into small patches (e.g., 4×4), and computing the KL-divergence between corresponding patches. We denote this loss term by Lpatch-KL. Second, we utilize Lpair proposed in pix2pix-zero [35], which penalizes correlations between pairs of pixels. We leverage these losses to enhance the editability of our method, and denote the combination of them as Ledit. For any renoising iteration k where wk > 0, we regularize the UNet’s prediction ϵθ(zt(k),t,c) using Ledit before computing zt(k+1). See line 9 in Algorithm 1.

Noise Correction in Non-deterministic Samplers Nondeterministic samplers, in which ρt > 0, introduce noise (ϵt) at each denoising step. Previous methods [21, 51] suggested using ϵt to bridge the gap between the inversion and denoising trajectories in DDPM inversion. Specifically, given a pair of points zt−1,zt on the inversion trajectory, we denote by zˆt−1 the point obtained by denoising zt. Ideally, zt−1 and zˆt−1 should be identical. We define:

1 ρt

(zt−1 − ϕtzt − ψtϵθ(zt,t,c)). (4)

ϵt =

Integrating this definition into Eq. 1 yields zˆt−1 = zt−1. However, we found that replacing ϵt with the above definition affects editability. Instead, we suggest a more tender approach, optimizing ϵt based on Eq. 4 as our guiding objective:

1 ρt

(zt−1 − ϕtzt − ψtϵθ(zt,t,c)). (5)

ϵt = ϵt − ∇ϵt

This optimization improves the reconstruction fidelity while preserving the distribution of the noisy-latents.

#### 4. Convergence Discussion

In this section, we first express the inversion process as a backward Euler process and our renoising iterations as fixed-point iterations. While these iterations do not converge in the general case, we present a toy example where they yield accurate inversions. Then, we analyze the convergence of the renoising iterations in our real-image inversion scenario and empirically verify our method’s convergence.

Algorithm 1: ReNoise Inversion

- 1 Input: An image z0, number of renoising steps K, number of inversion steps T, a series of renoising weights {wk}Kk=1.
- 2 Output: A noisy latent zT and set of noises {ϵt}Tt=1.
- 3 for t = 1,2,...,T do

- 4 sample ϵt ∼ N(0,I)
- 5 zt(0) ← zt−1
- 6 zt(avg) ← 0
- 7 for k = 1,...,K do

- 8 δtk ← ϵθ(zt(k−1),t)
- 9 δtk ← Enhance-editability(δtk,wk)
- 10 zt(k) ← Inverse-Step(zt−1,δtk)
- 11 end // Average ReNoised predictions
- 12 zt(avg) ← Kk=1 wk · zt(k)
- 13 ϵt ← Noise-Correction(zt(avg),t,ϵt,zt−1)
- 14 end
- 15 return (zT,{ϵt}Tt=1)
- 16
- 17 Function Inverse-Step(zt−1, δt, t):

- 18 return ϕ1

t

zt−1 − ψ

t

ϕt δt − ρ

t

ϕtϵt

- 19
- 20 Function Enhance-editability(δt(k), wk):

- 21 if wk > 0 then

- 22 δtk ← δtk − ∇δtkLedit(δtk)
- 23 end
- 24 return δtk
- 25
- 26 Function Noise-Correction(zt, t, ϵt, zt−1):

- 27 δt ← ϵθ(zt,t)
- 28 ϵt ← ϵt − ∇ϵt

1 ρt (zt−1 − ϕtzt − ψtδt)

- 29 return ϵt

Inversion Process as Backward Euler The denoising process of diffusion models can be mathematically described as solving an ordinary differential equation (ODE). A common method for solving such equations is the Euler method, which takes small steps to approximate the solution. For ODE in the form of y′(t) = f(t,y(t)), Euler solution is defined as:

yn+1 = yn + h · f(tn,yn),

where h is the step size. The inversion process can be described as solving ODE using the backward Euler method (or implicit Euler method) [1]. This method is similar to forward Euler, with the difference that yn+1 appears on both

15

(k)(k+1)− zztt

10

5

0

1 2 3 4 5 6

ReNoise step k

- Figure 7. Average distance between consequent estimations zt(k),

and zt(k+1). Vertical bars indicate the standard deviation. The averages are computed over 32 images and 10 different timesteps.

sides of the equation:

###### yn+1 = yn + h · f(tn+1,yn+1).

For equations lacking an algebraic solution, several techniques estimate yn+1 iteratively. As we described in Section 3.1, the inversion process lacks a closed-form solution, as shown in Equation 2. To address this, the ReNoise method leverages fixed-point iterations, which we refer to as reonising iterations, to progressively refine the estimate of yn+1:

yn(0)+1 = yn, yn(k+1+1) = yn + h · f(tn+1,yn(k+1) ).

In our ReNoise method, we average these renoising iterations to mitigate convergence errors, leading to improvement in the reconstruction quality.

Renoising Toy Example We begin with the simple toy example, the diffusion of a shifted Gaussian. Given the initial distribution µ0 ∼ N(a,I), where a is a non-zero shift value and I is the identity matrix. The diffusion process defines the family of distributions µt ∼ N(ae−t,I), and the probability flow ODE takes the form dzdt = −ae−t (see [24] for details). The Euler solver step at a state (zt,t), and timestep ∆t moves it to (zt(1)+∆t,t + ∆t) = (zt − ae−t · ∆t,t + ∆t). Notably, the backward Euler step at this point does not lead to zt. After applying the first renoising iteration, we get (zt(2)+∆t,t+∆t) = (zt−ae−(t+∆t)·∆t,t+∆t) and the backward Euler step at this point leads exactly to (zt,t). Thus, in this simple example, we successfully estimates the exact pre-image after a single step. While this convergence cannot be guaranteed in the general case, in the following, we discuss some sufficient conditions for the algorithm’s convergence and empirically verify them for the image diffusion model.

ReNoise Convergence During the inversion process, we aim to find the next noise level inversion, denoted by zˆt, such that applying the denoising step to zˆt recovers the previous state, zt−1. Given the noise estimation ϵθ(zt,t) and a given zt−1, the ReNoise mapping defined in Section 3.1 can

be written as G : zt → InverseStep(zt−1,ϵθ(zt,t)). For example, in the case of using DDIM sampler the mapping

(zt−1 − ψtϵθ(zt,t)). The point zˆt, which is mapped to zt−1 after the denoising step, is a stationary point of this mapping. Given zt(1), the first approximation of the next noise level zt, our goal is to show that the sequence zt(k) = Gk−1(zt(1)),k → ∞ converges. As the mapping G is continuous, the limit point would be its stationary point. The definition of G gives:

###### is G(zt) = ϕ1

t

∥zt(k+1) − zt(k)∥ = ∥G(zt(k)) − G(zt(k−1))∥, where the norm is always assumed as the l2-norm. For the ease of the notations, we define ∆(k) = zt(k) − zt(k−1). For convergence proof, it is sufficient to show that the sum of norms of these differences converges, which will imply that zt(k) is the Cauchy sequence. Below we check that in practice ∥∆(k)∥ decreases exponentially as k → ∞ and thus has finite sum. In the assumption that G is C2-smooth, the Taylor series conducts:

∥∆(k+1)∥ = ∥G(zt(k)) − G(zt(k−1))∥ = ∥G(zt(k−1))+

∂G ∂z |z(k−1) t

·∆(k)+O(∥∆(k)∥2)−G(zt(k−1))∥ = ∥

∂G ∂z |z(k−1) t

· ∆(k) + O(∥∆(k)∥2)∥ ≤

∂G ∂z |z(k−1) t

∥ · ∥∆(k)∥ + O(∥∆(k)∥2) =

∥

ψt ϕt · ∥

∂ϵθ ∂z |z(k−1)

###### ∥ · ∥∆(k)∥ + O(∥∆(k)∥2)

t

Thus, in a sufficiently small neighborhood, the convergence dynamics is defined by the scaled Jacobian norm

ψt ϕt · ∥∂ϵ

∥. In the Appendix A, we show this scaled norm estimation for the SDXL diffusion model for various steps and ReNoise iterations indices (k). Remarkably, the ReNoise indices minimally impact the scale factor, consistently remaining below 1. This confirms in practice the convergence of the proposed algorithm. Notably, the highest scaled norm values occur at smaller t (excluding the first step) and during the initial renoising iteration. This validates the strategy of not applying ReNoise in early steps, where convergence tends to be slower compared to other noise levels. Additionally, the scaled norm value for the initial t approaches 0, which induces almost immediate convergence.

∂z |z(k−1)

θ

t

Figure 7 illustrates the exponential decrease in distances

between consecutive elements zt(k) and zt(k+1), which confirms the algorithm’s convergence towards the stationary point of the operator G.

The proposed averaging strategy is aligned with the conclusions described above, and also converges to the desired stationary point. In The Appendix A, we present a validation for this claim.

###### SDXL SDXL Turbo LCM

27.5

Method

Configuration

22

- 26

- 27

- 28

- 29

- 30

- 31

DDIM w/ ReNoise

w/o Edit Loss

25.0

DDIM Inversion

w/ Edit Loss

20

22.5

18

20.0

PSNR

###### PSNR

###### PSNR

16

17.5

15.0

Method

Configuration

Method

Configuration

14

DDIM w/ ReNoise

w/o Edit Loss

DDIM w/ ReNoise

w/o Edit Loss

12.5

DDIM Inversion

w/ Edit Loss

DDIM Inversion

w/ Edit Loss

12

Euler w/ ReNoise

w/ Edit Loss and NC

LCM w/ ReNoise

10.0

Euler Inversion

LCM Inversion

10

60 80 100 120 140 160 180 200

5 10 15 20 25 30 35 40

5 10 15 20 25 30 35 40

Number of UNet Operations

Number of UNet Operations

Number of UNet Operations

- Figure 8. Image reconstruction results comparing sampler reversing inversion techniques across different samplers (e.g., vanilla DDIM inversion) with our ReNoise method using the same sampler. The number of denoising steps remains constant. However, the number of UNet passes varies, with the sampler reversing approach increasing the number of inversion steps, while our method increases the number of renoising iterations. We present various configuration options for our method, including options with or without edit enhancement loss and Noise Correction (NC).

#### 5. Experiments

In this section, we conduct extensive experiments to validate the effectiveness of our method. We evaluate both the reconstruction quality of our inversion and its editability. To demonstrate the versatility of our approach, we apply it to four models, SD [40], SDXL [37], SDXL Turbo [43], and LCM-LoRA [28], with SDXL Turbo and LCM-LoRA being few-step models. Additionally, we use various sampling algorithms including both deterministic and non-deterministic ones. Implementation details for each model are provided in Appendix B. Following previous works [4, 32], we quantitatively evaluate our method with three metrics: L2, LPIPS [52], and PSNR. Unless stated otherwise, for both inversion and generation we use the prompt obtained from BLIP2 [25].

##### 5.1. Reconstruction and Speed

We begin by evaluating the reconstruction-speed tradeoff. The main computational cost of both the inversion and denoising processes is the forward pass through the UNet. In each renoising iteration, we perform one forward pass, which makes it computationally equal to a standard inversion step (as done in DDIM Inversion for example). In the following experiments, we compare the results of a sampler reversing with our method, where we match the number of UNet passes between the methods. For example, 8 steps of sampler reversing are compared against 4 steps with one renoising iteration at each step.

Qualitative Results In Figure 9 we show qualitative results of image reconstruction on SDXL Turbo [43]. Here, we utilize DDIM as the sampler, and apply four denoising steps for all configurations. Each row exhibits results obtained using a different amount of UNet operations. In our method, we apply four inversion steps, and a varying number of renoising iterations. As can be seen, the addition of renoising iterations gradually improves the reconstruction results. Conversely, employing more inversion steps proves insufficient for capturing all details in the image, as evident

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Original

Image 4UNet

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Operations 8UNet

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Operations 20UNet

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Operations 40UNet

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Operations

DDIM Inversion

w/ ReNoise

DDIM Inversion

w/ ReNoise

Figure 9. Qualitative comparison between DDIM Inversion to our ReNoise method using the DDIM sampler on SDXL Turbo. The first row presents the input images. In each subsequent row, we present the reconstruction results of both approaches, each utilizing the same number of UNet operations in the inversion process. To generate the images, we use 4 denoising steps in all cases. DDIM Inversion performs more inversion steps (i.e., smaller strides), while our method performs more renoising steps.

by the background of the car, or even detrimental to the reconstruction, as observed in the Uluro example.

Quantitative Results For the quantitative evaluation, we use the MS-COCO 2017 [26] validation dataset. Specifically, we retain images with a resolution greater than 420 × 420, resulting in a dataset containing 3,865 images.

We begin by evaluating both the sampler reversing approach and our ReNoise method, while varying the num-

Image Reconstruction With a Fixed Number of UNet Operations Inversion Inference ReNoise L2 ↓ PSNR ↑ LPIPS ↓

Steps Steps Steps

50 50 0 0.00364 26.023 0.06273 75 25 0 0.00382 25.466 0.06605 80 20 0 0.00408 25.045 0.07099 90 10 0 0.01023 20.249 0.10305 25 25 2 0.00182 29.569 0.03637 20 20 3 0.00167 29.884 0.03633 10 10 8 0.00230 28.156 0.04678

Table 1. Image reconstruction results with a fixed number of 100 UNet operations. Each row showcases the results obtained using different combinations of inversion steps, denoising steps, and renoising iterations, totaling 100 operations. As observed, allocating some of the operations to renoising iterations improves the reconstruction quality while maintaining the same execution time.

ber of UNet operations during the inversion process and keeping the number of denoising steps fixed. This experiment is conducted using various models (SDXL, SDXL Turbo, LCM) and samplers. For all models, we utilize the DDIM [44] sampler. In addition, we employ the AncestralEuler scheduler for SDXL Turbo, and the default LCM sampler for LCM-LoRA. We set the number of denoising steps to 50 for SDXL, and to 4 for SDXL Turbo and LCMLoRA. Quantitative results, using PSNR as the metric, are presented in Figure 8. We evaluate our method using different configurations. The x-axis refers to the number of UNet operations in the inversion process. LPIPS metrics results are provided in Appendix C.

As depicted in the graphs, incorporating additional renoising iterations proves to be more beneficial for image reconstruction compared to adding more inversion steps. Note that the performance of the Ancestral-Euler and LCM samplers noticeably degrades when the number of inversion steps exceeds the number of denoising steps. Unlike DDIM, these samplers have Φt ≈ 1, resulting in an increase in the latent vector’s norm beyond what can be effectively denoised in fewer steps. In this experiment, we maintain the same number of UNet operations for both ReNoise and the sampler reversing approach. However, in ReNoise, the number of inversion steps remains fixed, and the additional operations are utilized for renoising iterations, refining each point on the inversion trajectory. Consequently, our method facilitates improved reconstruction when using these noise samplers.

We continue by evaluating both the sampler reversing approach and our method while maintaining a fixed total number of UNet operations for the inversion and denoising processes combined. The results for SDXL with DDIM are presented in Table 1. The table displays various combinations of inversion, denoising, and renoising steps, totaling 100 UNet operations. Despite employing longer strides

Original Image ←− Editing Results −→

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

“cat” “koala” “cat statue” “bear”

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

“person” “panda mask” “purple shirt” “astronaut”

Figure 10. LCM Editing Results. Each row showcases one image. The leftmost image is the original, followed by three edited versions. The text below each edited image indicates the specific word or phrase replaced or added to the original prompt for that specific edit.

along the inversion and denoising trajectories, our ReNoise method yields improved reconstruction accuracy, as evident in the table. Furthermore, a reduced number of denoising steps facilitates faster image editing, especially since it commonly involves reusing the same inversion for multiple edits.

##### 5.2. Reconstruction and Editability

In Figure 10, we illustrate editing results generated by our method with LCM LoRA [28]. These results were obtained by inverting the image using a source prompt and denoising it with a target prompt. Each row exhibits an image followed by three edits accomplished by modifying the original prompt. These edits entail either replacing the object word or adding descriptive adjectives to it. As can be seen, the edited images retain the details present in the original image. For instance, when replacing the cat with a koala, the details in the background are adequately preserved.

##### 5.3. Ablation Studies

Image Reconstruction Figure 11 qualitatively demonstrates the effects of each component in our method, highlighting their contribution to the final outcome. Here, we use SDXL Turbo model [43], with the Ancestral-Euler sampler, which is non-deterministic. As our baseline, we simply reverse the sampler process. The reconstruction, while semantically capturing the main object, fails to reproduce the image’s unique details. For example, in the middle column, the image contains a bird standing on a branch, but the branch is in a different pose and the bird is completely different. Using 9 ReNoise iterations significantly improves the reconstruction, recovering finer details like the bird’s original pose and branch texture. However, some subtle details, such as the bird’s colors or the color in Brad Pitt’s image, remain incomplete. Averaging the final iterations effectively incorporates information from multiple predic-

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Original

Image Euler

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Inversion With

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

ReNoise With

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Averaging WithEdit

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Losses WithNoise

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Correction

Figure 11. Ablation study on SDXL Turbo. The first row presents the input image. In each subsequent row, we show the reconstruction results using an additional component of our inversion method. The images in the bottom row represent the results obtained by our full method.

tions, leading to a more robust reconstruction that captures finer details. Regularize the UNet’s noise prediction with Ledit can introduce minor artifacts to the reconstruction, evident in the smoother appearance of the hair of the two people on the left, or in the cake example. Finally, we present our full method by adding the noise correction technique.

Table 2 quantitatively showcases the effect each component has on reconstruction results. As can be seen, the best results were obtained by our full method or by averaging the last estimations of zt. Our final method also offers the distinct advantage of getting an editable latent representation.

In Appendix C, we present an ablation study to justify our editability enhancement and noise correction components.

##### 5.4. Comparisons

Inversion for Non-deterministic Samplers. In Figure 12 we show a qualitative comparison with “an edit-friendly DDPM” [21] where we utilize SDXL Turbo [43]. Specifically, we assess the performance of the edit-friendly DDPM method alongside our ReNoise method in terms of both reconstruction and editing.

We observe that in non-deterministic samplers like DDPM, the parameter ρ0 in Equation 1 equals zero. This means that in the final denoising step, the random noise addition is skipped to obtain a clean image. In long diffusion

Ablation - Image Reconstruction L2 ↓ PSNR ↑ LPIPS↓

Euler Inversion 0.0700 11.784 0.20337 + 1 ReNoise 0.0552 12.796 0.20254 + 4 ReNoise 0.0249 16.521 0.14821 + 9 ReNoise 0.0126 19.702 0.10850 + Averaging ReNoise 0.0087 21.491 0.08832 + Edit Losses 0.0276 18.432 0.12616 + Noise Correction 0.0196 22.077 0.08469

Table 2. Quantitative ablation study on SDXL Turbo. We demonstrate the impact of each component of our inversion method on reconstruction results. The results improve with additional renoising iterations and significant enhancements occur through averaging final estimations. Additionally, we observe a reconstructioneditability trade-off, with edit losses causing degradation that is effectively mitigated by Noise Correction.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

EditFriendlyOurs

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Original Reconstruction “ginger cat” “wood” → “metal”

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

EditFriendlyOurs

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Original Reconstruction “cat” → “dog” “wooden cat”

Figure 12. Comparison with edit-friendly DDPM Inversion with SDXL Turbo. We invert two images with the prompts: “a cat laying in a bed made out of wood” (left) and “a cat is sitting in front of a mirror” (right) and apply two edits to each image.

processes (e.g., 50-100 steps), the final denoising step often has minimal impact as the majority of image details have already been determined. Conversely, shorter diffusion processes rely on the final denoising step to determine fine details of the image. Due to focusing solely on noise correction to preserve the original image in the inversion process, the edit-friendly DDPM struggles to reconstruct fine details of the image, such as the shower behind the cat in the right example. However, our ReNoise method finds an inversion trajectory that faithfully reconstructs the image and does not rely solely on noise corrections. This allows us to better reconstruct fine details such as the shower.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Original Image

Null Text Inversion

Negative Prompt Inv

DDIM inv w/ ReNoise

Figure 13. Image reconstruction comparisons with Stable Diffusion. We present the results of Null-Text Inversion (NTI), Negative-Prompt Inversion (NPI), and our method. While NTI and our method achieve comparable results, ours demonstrates significant speed improvement.

Additionally, encoding a significant amount of information within only a few external noise vectors, ϵt, limits editability in certain scenarios, such as the ginger cat example. It is evident that the edit-friendly DDPM method struggles to deviate significantly from the original image while also failing to faithfully preserve it. For instance, it encounters difficulty in transforming the cat into a ginger cat while omitting the preservation of the decoration in the top left corner.

Null-prompt Inversion Methods In Figure 13, we present a qualitative comparison between our method and null-text based inversion methods. For this comparison, we utilize Stable Diffusion [40] since these methods rely on a CFG [19] mechanism, which is not employed in SDXL Turbo [43]. Specifically, we compare DDIM Inversion [44] with one renoising iteration to Null-Text Inversion (NTI)[32] and Negative-Prompt Inversion (NPI)[31]. Both NTI and NPI enhance the inversion process by replacing the null-text token embedding when applying CFG. Our method achieves results comparable to NTI, while NPI highlights the limitations of plain DDIM inversion. This is because NPI sets the original prompt as the negative prompt, essentially resulting in an inversion process identical to plain DDIM inversion. Regarding running time, our ReNoise inversion process takes 13 seconds, significantly faster than NTI’s 3 minutes. For comparison, plain DDIM inversion and NPI each take 9 seconds.

#### 6. Conclusion

In this work, we have introduced ReNoise, a universal approach that enhances various inversion algorithms of diffusion models. ReNoise gently guides the inversion curve of a real image towards the source noise from which a denoising process reconstructs the image. ReNoise can be considered as a meta-algorithm that warps the trajectory of

any iterative diffusion inversion process. Our experiments demonstrate that averaging the last few renoising iterations significantly enhances reconstruction quality. For a fixed amount of computation, ReNoise shows remarkably higher reconstruction quality and editability. The method is theoretically supported and our experiments reconfirm its effectiveness on a variety of diffusion models and sampling algorithms. Moreover, the method is numerically stable, and always converges to some inversion trajectory that eases hyperparameters adjustment.

Beyond the net introduction of an effective inversion, the paper presents a twofold important contribution: an effective inversion for few-steps diffusion models, which facilitates effective editing on these models.

A limitation of ReNoise is the model-specific hyperparameter tuning required for Edit Enhancement and Noise Correction. While these hyperparameters remain stable for a given model, they may vary across models, and tuning them is necessary to achieve high reconstruction quality while maintaining editability.

While ReNoise demonstrates the potential for editing few-step diffusion models, more extensive testing with advanced editing methods is needed. It is worth noting that no such editing has been demonstrated for the few-step diffusion models. We believe and hope that our ReNoise method will pave the way for fast and effective editing methods based on the few-steps models. We also believe that ReNoise can be adapted to the challenging problem of inverting video-diffusion models.

#### References

- [1] Numerical Differential Equation Methods, chapter 2, pages 45–121. John Wiley and Sons, Ltd, 2003. 5
- [2] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2019. 2
- [3] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan++: How to edit the embedded images? In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020.
- [4] Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Restyle: A residual-based stylegan encoder via iterative refinement. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021. 7
- [5] Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit H. Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing, 2021. 2
- [6] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar AverbuchElor, and Daniel Cohen-Or. Cross-image attention for zeroshot appearance transfer, 2023. 1, 2
- [7] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. arXiv preprint arXiv:2206.02779, 2022.
- [8] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Pro-

- ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022.
- [9] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions,

2023. 1, 2

- [10] R.L. Burden, J.D. Faires, and A.M. Burden. Numerical Analysis. Cengage Learning, 2015. 2, 4
- [11] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. ArXiv, abs/2210.11427, 2022. 2
- [12] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021. 2, 3, 4, 15
- [13] Tan M. Dinh, Anh Tuan Tran, Rang Nguyen, and Binh-Son Hua. Hyperinverter: Improving stylegan inversion via hypernetwork. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [14] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. 2023. 2
- [15] Songwei Ge, Taesung Park, Jun-Yan Zhu, and Jia-Bin Huang. Expressive text-to-image generation with rich text. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1, 2
- [16] Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Anastasis Stathopoulos, Xiaoxiao He, Yuxiao Chen, Di Liu, Qilong Zhangli, Jindong Jiang, Zhaoyang Xia, Akash Srivastava, and Dimitris Metaxas. Improving tuning-free real image editing with proximal guidance, 2023. 1, 2
- [17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022. 1, 2
- [18] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. 2023. 2
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 10
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 1, 2, 3
- [21] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations, 2023. 1, 2, 3, 5, 9, 15
- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 2
- [23] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Conference on Computer Vision and Pattern Recognition 2023,

2023. 1, 2

- [24] Valentin Khrulkov, Gleb Ryzhakov, Andrei Chertkov, and Ivan Oseledets. Understanding ddpm latent codes through optimal transport. In The Eleventh International Conference on Learning Representations, 2022. 6

- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 7, 13
- [26] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll’a r, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312,

2014. 7

- [27] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference, 2023. 3, 4
- [28] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023. 2, 3, 4, 7, 8, 14
- [29] Barak Meiri, Dvir Samuel, Nir Darshan, Gal Chechik, Shai Avidan, and Rami Ben-Ari. Fixed-point inversion for textto-image diffusion models, 2023. 3
- [30] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations, 2022. 1, 2
- [31] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models, 2023. 1, 2, 10, 15
- [32] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models, 2022. 1, 2, 7, 10
- [33] Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. Effective real image editing with accelerated iterative diffusion inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15912–15921, 2023. 3
- [34] Gaurav Parmar, Yijun Li, Jingwan Lu, Richard Zhang, JunYan Zhu, and Krishna Kumar Singh. Spatially-adaptive multilayer selection for gan inversion and editing. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022. 2
- [35] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Proceedings. ACM, 2023. 1, 2, 5, 14, 15
- [36] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar AverbuchElor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 1, 2
- [37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 2, 7, 14, 15
- [38] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 1, 2

- [39] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a stylegan encoder for image-to-image translation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 1, 2, 3, 7, 10, 14
- [41] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding, 2022. 1, 2
- [42] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2021. 3
- [43] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation, 2023. 2, 3, 4, 7, 8, 9, 10, 14, 15
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. 2, 3, 8, 10
- [45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020. 1
- [46] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023. 3, 4
- [47] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. ACM Trans. Graph., 40(4), 2021. 2, 5
- [48] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. pages 1921–1930, 2023. 1, 2
- [49] Andrey Voynov, Amir Hertz, Moab Arar, Shlomi Fruchter, and Daniel Cohen-Or. Anylens: A generative diffusion model with any rendering lens. 2023. 2
- [50] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. arXiv preprint arXiv:2211.12446, 2022. 2
- [51] Chen Henry Wu and Fernando De la Torre. Unifying diffusion models’ latent space, with applications to cyclediffusion and guidance. arXiv preprint arXiv:2210.05559, 2022. 3, 5
- [52] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 7
- [53] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. Indomain gan inversion for real image editing. In Proceedings of European Conference on Computer Vision (ECCV), 2020. 2
- [54] Jun-Yan Zhu, Philipp Kr¨ahenb¨uhl, Eli Shechtman, and Alexei A. Efros. Generative visual manipulation on the natural image manifold. In Proceedings of European Conference on Computer Vision (ECCV), 2016.

[55] Peihao Zhu, Rameen Abdal, Yipeng Qin, John Femiani, and Peter Wonka. Improved stylegan embedding: Where are the good latents?, 2020. 2

# Appendices

#### A. Convergence Discussion

##### A.1. ReNoise As Contraction mapping

In this section, we continue the convergence discussion of our method.

Scaled Jacobian Norm of ReNoise Mapping Figure 14 shows this scaled norm estimation for the SDXL diffusion model for various steps and ReNoise iterations indices (k). Remarkably, the ReNoise indices minimally impact the scale factor, consistently remaining below 1. This confirms in practice the convergence of the proposed algorithm. Notably, the highest scaled norm values occur at smaller t (excluding the first step) and during the initial renoising iteration. This validates the strategy of not applying ReNoise in early steps, where convergence tends to be slower compared to other noise levels. Additionally, the scaled norm value for the initial t approaches 0, which induces almost immediate convergence, making ReNoise an almost identical operation.

Validation for the Averaging Strategy Notably, the proposed averaging strategy is aligned with the conclusions described in the main paper and also converges to the desired stationary point. To verify this claim we will show that if

a sequence zt(+1k) converges to some point zt+1, then the averages z¯t(+1k) = k1 ki=1 zt(+1i) converges to the same point. That happens to be the stationary point of the operator G. We demonstrate it with the basic and standard calculus. Assume that zt(+1k) = ε(t+1k) +zt+1 with ∥ε(t+1k) ∥ → 0 as k → ∞. For a fixed ε we need to show that there exists K so that ∥z¯t(+1k) − zt+1∥ < ϵ for any k > K. One has

ε(t+1i) k

k

z¯t(+1k) − zt+1 =

i=1

There exists m such that ∥ε(t+1k) ∥ < 0.5 · ε once k > m. Then we have

+ ∥ ki=m+1 ε(t+1i) ∥

ε(t+1i) k ∥ ≤

∥ mi=1 ε(t+1i) ∥ k

k

∥

k ≤ ∥ mi=1 ε(t+1i) ∥ k

i=1

∥ mi=1 ε(t+1i) ∥ k

·(k − m) k ≤

ε 2

ε 2

+

+

given that m is fixed, we can always take k sufficiently large such that

∥ mi=1 ε(t+1i) ∥ k

ε 2

<

this ends the proof. The very same computation conducts a similar result if the elements’ weights wk are non-equal.

zt(1) zt(2)

zt(2) zt(3)

1.0

1.0

zt

0.5

0.5

t

0 200 400 600 800 t

0 200 400 600 800 t

zt(3) zt(4)

zt(4) zt(5)

1.0

1.0

zt

0.5

0.5

t

0 200 400 600 800 t

0 200 400 600 800 t

- Figure 14. Scaled Jacobian norm of the mapping zt(k) → zt(k) calculated for various noise levels t, iterations k. We report the average and the standard deviation calculated over 32 images. Values below 1 indicate the exponential convergence of the ReNoise algorithm.

Original Image ←− Editing Results −→

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

“kitten” “broccoli” “made of lego” “plush toy”

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

“...in the woods” “...in the beach” “...in the desert” “...in the snow”

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

“lion” “tiger” “monkey” “panda”

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

“monkey” “beaver” “raccoon” “...in the snow”

- Figure 15. SDXL Turbo editing results. Each row showcases one image. The leftmost image is the original, followed by three edited versions. The text below each edited image indicates the specific word or phrase replaced or added to the original prompt for that specific edit. B. Implementation Details

We used BLIP-2 [25] to generate captions for the input images, which were then used as prompts for the diffusion models. In Table 3 , we provide all hyperparameters of ReNoise inversion per model, optimized for the best reconstruction-editability trade-off:

Where {wi} are the renoising estimations averaging weights, and λpair and λpatch-KL are the weights we assign to each component of the edit enhancement loss:

Ledit = λpair · Lpair + λpatch-KL · Lpatch-KL

###### Implementation details

Model Name Stable Diffusion SDXL SDXL Turbo LCM LoRA Noise Sampler DDIM DDIM Ancestral-Euler DDIM Number of denoising steps 50 50 4 4 Number of renoising iterations 1 1 9 7

Weights for t < 250 w1, w2 = 0.5 w1, w2 = 0.5 w1, ..., w4 = 0.25 w1, ..., w4 = 0.25 Weights for t > 250 w2 = 1.0 w2 = 1.0 w8, ..., w10 = 0.33 w6, ..., w8 = 0.33 λpair 10 10 10 20 λpatch-KL 0.05 0.055 0.055 0.075

Table 3. Implementation details of ReNoise with Stable Diffusion [40], SDXL [37], SDXL Turbo [43] and LCM LoRA [28].

SDXL SDXL Turbo LCM

0.22

Method

Configuration

Method

Configuration

Method

Configuration

0.225

0.07

DDIM w/ ReNoise

w/o Edit Loss

DDIM w/ ReNoise

w/o Edit Loss

DDIM w/ ReNoise

w/o Edit Loss

0.20

DDIM Inversion

w/ Edit Loss

DDIM Inversion

w/ Edit Loss

DDIM Inversion

w/ Edit Loss

0.200

LCM w/ ReNoise

Euler w/ ReNoise

w/ Edit Loss and NC

0.18

0.06

LCM Inversion

0.175

Euler Inversion

0.16

0.150

LPIPS

LPIPS

LPIPS

0.05

0.14

0.125

0.04

0.100

0.12

0.075

0.10

0.03

0.050

0.08

60 80 100 120 140 160 180 200

5 10 15 20 25 30 35 40

5 10 15 20 25 30 35 40

Number of UNet Operations

Number of UNet Operations

Number of UNet Operations

Figure 16. Image reconstruction results comparing sampler reversing inversion techniques across different samplers (e.g., vanilla DDIM inversion) with our ReNoise method using the same sampler. The number of denoising steps remains fixed. However, the number of UNet passes varies, with the number of inversion steps increasing in the sampler reversing approach, and the number of renoising iterations increasing in our method. We present various configuration options for our method, including options with or without edit enhancement loss and Noise Correction (NC).

#### C. Additional Experiments

Editing Results With SDXL Turbo Figure 15 showcases additional image editing examples achieved using our ReNoise inversion method. These edits are accomplished by inverting the image with a source prompt, and then incorporating a target prompt that differs by only a few words during the denoising process.

Reconstruction and Speed We continue our evaluation of the reconstruction-speed tradeoff from Section 5.1 in the main paper. Figure 16 presents quantitative LPIPS results for the same configuration described in the main paper. As expected, LPIPS scores exhibit similar trends to the PSNR metric shown previously.

Image Editing Ablation Figure 18 visually illustrates the impact of edit enhancement losses and noise correction when editing inverted images using SDXL Turbo [43]. While achieving good reconstructions without the Ledit regularization, the method struggles with editing capabilities (second column). Although the Ledit regularization enhances editing capabilities, it comes at the cost of reduced reconstruction accuracy of the original image, as evident in the two middle columns. In the third column, we use LKL as defined in pix2pix-zero [35]. While Lpatch-KL surpasses LKL in original image preservation, further improvements

Original Image

Pix2Pix zero w/ ReNoise

Pix2Pix zero

[Figure 144]

[Figure 145]

[Figure 146]

“Cat” −→ “Dog”

[Figure 147]

[Figure 148]

[Figure 149]

“Dog” −→ “Cat”

Figure 17. Zero-Shot Image-to-Image Translation editing results. This figure compares editing results with Stable Diffusion [40] achieved using two inversion methods: pix2pix-zero [35] and our proposed ReNoise inversion. As observed, ReNoise inversion preserves image details while effectively incorporating the desired edits.

are necessary. These improvements are achieved by using the noise correction technique. To correct the noise, we can either override the noise ϵt in Equation 1 in the main paper (fifth column), or optimize it (sixth column). As observed, overriding the noise ϵt affects editability, while optimizing

Our w/ Noise Correction Optimize

Our w/ Noise Correction Override

Our w/ Lpatch-KL and Lpair

Original Image

Our w/o Edit Losses

Our w/ LKL and Lpair

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Rec.Edit

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 18. Ablation study on SDXL Turbo for image editing. The first row displays reconstructed images, while the second row showcases edits replacing “cat” with “raccoon”. Each column represents a different inversion configuration. From left to right, the second column demonstrates results with renoising iterations and estimations averaging. In the following two columns we employ different edit enhancement losses. Finally, the last two columns present results using both noise correction (NC) and Ledit. In the fifth column, NC overrides the noise ϵt, while in the sixth column, we present the results of our full method, where NC optimizes ϵt.

EditFriendlyOurs

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Original Reconstruction “ginger cat” “wood” → “metal”

EditFriendlyOurs

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Original Reconstruction “black&white dog” “...with glasses”

- Figure 19. Comparison with edit-friendly DDPM Inversion with SDXL Turbo. We invert two images with the prompts: “a cat laying in a bed made out of wood” (top) and “a dog sitting on the beach with its tongue out” (bottom) and apply two edits to each image.

that rely on inversion methods like DDIM [12], negative prompt inversion [31] and more. It seamlessly integrates with these existing approaches, boosting their performance without requiring extensive modifications. Figure 17 showcases image editing examples using Zero-Shot Image-toImage Translation (pix2pix-zero) [35]. We compared inversions with both the pix2pix-zero inversion method and our ReNoise method. Our method demonstrably preserves finer details from the original image while improving editability, as exemplified by the dog-to-cat translation.

Inversion for Non-deterministic Samplers In Figure 19 we show more qualitative comparisons with “an editfriendly DDPM” [21] where we utilize SDXL Turbo [43]. As can be seen, encoding a significant amount of information within only a few external noise vectors, ϵt, limits editability in certain scenarios, such as the ginger cat example. It is evident that the edit-friendly DDPM method struggles to deviate significantly from the original image in certain aspects while also failing to faithfully preserve it in others. For instance, it encounters difficulty in transforming the cat into a ginger cat while omitting the preservation of the decoration in the top left corner. In addition, the image quality of edits produced by edit-friendly DDPM is lower, as demonstrated in the dog example.

Improving DDIM Instabilities As mentioned in Section 3.1 of the main paper, DDIM inversion [12] can exhibit instabilities depending on the prompt. Figure 20 demonstrates this with image reconstruction on SDXL [37] using an empty prompt. Notably, incorporating even a single renoising iteration significantly improves inversion stability.

it achieves good results in terms of both reconstruction and editability. Therefore, in our full method, we use Lpatch-KL, Lpair, and optimization-based noise correction.

Editing With ReNoise The ReNoise technique provides a drop-in improvement for methods (e.g., editing methods)

Original Image DDIM Inversion +1 ReNoise Step Original Image DDIM Inversion +1 ReNoise Step

[Figure 178]

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

- Figure 20. Comparing reconstruction results with an empty prompt of plain DDIM inversion on SDXL to DDIM inversion with one ReNoise iteration.

