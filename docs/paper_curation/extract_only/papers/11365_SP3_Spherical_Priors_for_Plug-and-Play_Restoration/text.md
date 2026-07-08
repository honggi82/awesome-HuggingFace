# arXiv:2606.16396v1[cs.CV]15Jun2026

## SP3: Spherical Priors for Plug-and-Play Restoration

Sean Man Technion – Israel Institute of Technology sean.man@campus.technion.ac.il

#### Matan Kleiner∗

Technion – Israel Institute of Technology matankleiner@campus.technion.ac.il

#### Ron Raphaeli∗

Independent Researcher

Or Ronai Technion – Israel Institute of Technology or.ronai@campus.technion.ac.il

### Abstract

In this paper, we introduce SP3, a novel Plug-and-Play algorithm that accelerates maximum a posteriori image restoration by replacing denoisers with Spherical Encoders (SE) as generative priors. SP3 approximates the intractable proximal prior step by utilizing the SE tightly structured latent space as a robust projection onto the natural image manifold. Alternating this projection with a closed-form dataconsistency step, via Half-Quadratic Splitting, achieves stable convergence without requiring gradient computation during inference. This unique formulation unlocks “anytime” restoration capabilities, producing sharp, plausible images from the first iteration. Evaluations across a variety of image restoration tasks demonstrate that SP3 achieves perceptual quality comparable to state-of-the-art zero-shot diffusion and flow methods while being 3-630× faster.

### 1 Introduction

Recovering clean images from degraded measurements is a fundamental task in image processing, as photos are often acquired as degraded. A popular technique for solving image restoration is by optimizing the maximum a posteriori (MAP) objective [1]. MAP estimator maximizes the log-likelihood of a data fidelity term and the log-probability of a natural image prior term. After maximizing these two objectives, the resulted image is both natural looking and loyal to the measurements.

Although the data-fidelity term is often straightforward to obtain, specifying an explicit, tractable prior function is challenging. As a result, many approaches have been developed to replace the prior term with easy-to-compute alternatives [1, 2]. One prominent example is the Plug-and-Play (PnP) restoration framework [1], which replaces the explicit prior optimization step within an iterative algorithm with a denoising operator. This framework is particularly attractive because it allows using arbitrary pretrained denoisers. Consequently, better denoisers, such as neural and generative denoisers, can be incorporated with minimal modifications to the underlying reconstruction algorithm [3–6].

With improvements in generative modeling, early methods with a structured latent space, such as generative adversarial network (GAN) [7], variational autoencoders (VAE) [8], and normalizing flows [9], were used as an image prior in restoration tasks [10–16]. These methods often solve MAP optimization problem in the models’ latent space, where the prior term is known and easy to sample from, such as a Gaussian. However, they mostly yield poor results [6]. Recent work focuses on utilizing diffusion and flow models [17–23] as generative priors for zero-shot image restoration [24–36]. Yet, their reliance on gradients computed during inference, combined with their iterative nature, significantly increases restoration time.

∗denotes equal contribution

Preprint.

AFHQ Deblurring

[Figure 1]

DPS OT-ODE

RED

100

D-Flow

| |
|---|

SP3

PnP Flow

DiffPIR

3KID←×10

10

1

1 100

← Runtime [sec]

Figure 1: SP3. We introduce a Plug-and-Play image restoration algorithm (left) that iteratively projects degraded observations onto a tightly structured latent space using a Spherical Encoder (SE) generative prior. The trajectory along the sphere illustrates our gradient-free, alternating optimization approach, which navigates the image manifold to enforce data consistency. The magnified crops highlight the algorithm’s “anytime” generation capability, demonstrating how it stably converges to produce perceptually sharp and plausible details from the very first iteration. (right) SP3 achieves perceptual quality comparable to state-of-the-art zero-shot methods, while being 3 − 630× faster.

In this work, we use the recently proposed Sphere Encoder (SE) [37], a generative model that maps images onto a latent spherical manifold (and vice versa) as a prior for image restoration. Following the original motivation behind Plug-and-Play, enforcing a prior information through a projection operator, we design Spherical Prior for Plug-and-Play Restoration (SP3). SP3, schematically illustrated in the left part of Figure 1, alternates, via Half-Quadratic Splitting (HQS), between a prior step and a data fidelity step. The prior step projects the current estimate onto the learned spherical manifold, refines the resulting latent representation, and projects it back to the image manifold, where the data-fidelity step applies a data proximal operator. Importantly, our method stably converges without requiring gradient computation or backpropagation at all.

Unlike diffusion/flow-based methods, SP3 is extremely fast, achieving ×3 − 630 speed improvement compared to previous work, illustrated in the right part of Figure 1. Moreover, one attractive property of SP3, emerges from using the SE prior, is anytime restoration – producing a single step reconstruction that could be further refined to improve its perceptual quality (left part of Figure 1). SP3 achieves these fast and anytime results while producing results on par to different PnP and diffusion/flow based methods, across different noisy restoration problems.

### 2 Background and related work

#### 2.1 Sphere Encoder

Sphere Encoder (SE) [37] is a recent generative framework that, similar to VAEs [8], relies on an autoencoder [38] to encode images onto a structured latent space and decode latent vectors back to the image space. Unlike VAEs that promote the latent space to follow a Gaussian distribution, SE promotes the latent space to be uniformly distributed on a sphere. By inducing a latent sphere, SE overcomes many of the shortcomings associated with VAEs, especially the posterior hole problem [39– 41], which results in unrealistic image samples.

To create the spherical latent space, SE employs a spherifying function (RMS normalization), denoted by f(·), to the output of the encoder E. Namely, a latent vector is given by v = f(E(x)). During training, instead of decoding the latent vector v with the decoder D, the SE framework first add noise to the latent vector, followed by spherification, S(v,σ) = f(v + σ · e), where e ∼ N(0,I) is a white Gaussian noise vector and σ is a scalar that controls the noise magnitude. Adding noise to the latent vectors during training results in a latent sphere that is entirely covered and densely populated.

The training objective of SE compose of three reconstruction loss functions:

LSE = LL1+perc.(D(vn),x) + LL1+perc.(D(vN),sg(D(vn))) + Lcos. sim.(v,E(D(vN))), (1) where L1 denotes smoothed L1 loss [42], perc. denotes perceptual loss [43], cos. sim denotes cosine similarity, vϵ = v + ϵ, and n = σ1e, N = σ2e with σ2 > σ1. In this work, we use the SE framework as a prior for image restoration, and illustrate how its unique features are crucial to the success of SP3.

#### 2.2 Zero-shot generative priors for image restoration

Image restoration aims to recover a clean natural image, xˆ, from a degraded measurement y. The degraded measurement is considered to be the result of a degradation operator, A, applied to an unknown clean image x, with the addition of white Gaussian noise n ∼ N(0,σn2I),

Table 1: SE baseline quantitative comparisons. Comparisons between SP3 and baselines that solve MAP optimization problem in SE latent space. Results are given for deblurring on AFHQ-Cat. We highlight the best method in each metric.

S-GD S-GPD SP3 PSNR [dB] ↑ 18.5 23.1 25.7 LPIPS ↓ 0.58 0.26 0.12 KID×103 ↓ 34.3 5.49 2.64 Time [sec] 1.32 1.17 0.28

y = Ax + n. (2)

The degradation operator A is linear for many common degradation types, such as inpainting, super-resolution, deblurring, and denoising. Following the notations of Blau & Michaeli [44], the natural image x is a realization of a random

- variable X with probability density function pX. The measurement y is a realization of a random
- variable Y , which relates to X via the conditional probability density function pY |X. In general,

an image restoration algorithm is an estimator Xˆ that generates reconstructions according to pXˆ|Y , where X and Xˆ are statistically independent given Y .

The success of deep learning in image processing tasks led to a wide adoption of deep networks for image restoration. Early approaches trained regression based networks for image restoration [45–51], while recent approaches trained conditional generative models [52–65]. Despite the impressive results of these methods, they are often constrained by degradation processes seen during training.

An alternative, flexible, approach leverages pretrained models as image priors for zero-shot image restoration. Frameworks such as PnP [1] and RED [2] demonstrate that off-the-shelf deep [3, 4] and generative [5, 66, 6] denoisers can effectively regularize the restoration process. This paradigm has expanded beyond denoisers to incorporate diverse restoration models [67, 68]. Driven by advances in generative modeling, recent frameworks integrates GANs [7], VAEs [8], normalizing flows [9], diffusion [17–20], and flow models [21–23] as powerful image priors.

Clean Degraded

A common direction to solve a MAP problem in a generative model’s latent space [10, 69, 12– 14, 16], requires optimizing the latent vector such that applying the model and the degradation operator A, will result in an image similar to the observed measurement, y. The results of these methods are often poor, as they require propagating gradients through the generative model.

[Figure 2]

This approach is not easily applied to diffusion/flow models due to their iterative nature and non-trivial latent space. Nevertheless, many methods were developed to apply the image prior learned by diffusion/flow models for zeroshot restoration, often by combining the diffusion model and the acquisition process, via Bayes’ rule [24–36]. Their reliance on gradients computed during inference, amalgamated with their iterative nature, results in high-quality restorations at the cost of slow inference.

SP3 S-PGD S-GD

Figure 2: SE baseline qualitative comparisons for AFHQ deblurring. While S-GD fails to fully project onto the natural image manifold, and SPGD struggles to remain faithful to the original data, SP3 successfully balances both constraints to restore a sharp, high-fidelity image.

SE projection operator SE vs. VAE Sphere noise

0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

300

0.04 0.08 0.12 0.16 0.24

1.0

###### 3KID←×10

3KID←×10

0.5

30

LPIPS←

LPIPS←

100

0.8

0.4

0.6

VAE type

0.4

0.3

SD-VAE EQ-VAE

10

10

0 5 10 15 20

0 5 10 15 20

0 5 10 15 20

0 5 10 15 20

- FLUX.1

- FLUX.2

Steps

Steps

Steps

Steps

Figure 3: SE as a projection operator to the clean image manifold. In both panels, the left plot shows LPIPS, and the right shows KID (on a log scale). Step 0 shows the results of the blurred, noisy input before projections. (left) Performance of iteratively applying SE as a projection operator for different values of latent noise. SE projects degraded images to the image manifold, as illustrated by the decrease in the KID score with consecutive steps. (right) Different state-of-the-art VAEs are poor projection operators. SE is illustrated with latent noise of 0.12 on the right.

Evaluating image restoration methods requires using perception (whether a reconstructed image looks natural) and distortion (whether a reconstructed image resembles the unknown original image) metrics. It is well established that there is an inherent tradeoff between these metrics [44, 70], therefore, many methods excel at either of these metrics, while sacrificing the other.

In this work, we utilize the recent SE [37] for Plug-and-Play/zero-shot image restoration. In contrast to previous approaches that relied on propagated gradients through the generative model, we use a fast, gradient-free iterative process that takes advantage of SE strong generative prior.

### 3 Method

To achieve a MAP estimator, we need to obtain a prior over natural images, to guide the restoration process towards a plausible-looking image. Additionally, a measurement-enforcing term is needed to ground the solution in the known, available data.

In this section, we first define the MAP objective within the SE latent space and evaluate two baseline optimization strategies that rely exclusively on the decoder. Next, we analyze the Spherical Encoder’s behavior as a projection operator, highlighting its unique stability and manifold-mapping properties in comparison to standard VAEs. Building on these insights, we introduce SP3: a simple, fast, and effective PnP algorithm. By embedding the SE projection within a Half-Quadratic Splitting (HQS) framework, SP3 alternates between manifold refinement and proximal data-consistency updates to achieve high-fidelity restoration without requiring expensive gradient backpropagation through the decoder or encoder.

- 3.1 Naive MAP with a sphere prior Given a pretrained spherical prior, we can formulate the restoration task as a MAP estimation problem:

− log p(y|x) − log p(x) , (3) where the right term results from applying Bayes’ theorem and taking the negative logarithm.

p(x|y) = arg min

arg max

x

x

Assuming that the decoder D is injective, such that every distinct latent vector v maps to a unique image x, we can reparameterize the problem over the latent space V rather than the image space X. Furthermore, following the assumption of additive Gaussian noise in the measurements as defined in Equation (2), the objective becomes:

∥AD(v) − y∥22 − log p(v) . (4) This formulation permits us to treat the prior term independently.

arg min

v

We consider two baseline solutions for handling the spherical prior. The first approach strictly enforces the prior by assuming the distribution is uniform over the shell of a sphere S = {v ∈ V | ∥v∥22 = L} and zero elsewhere, p(v) ∝ δ(∥v∥22 − L), allowing us to define IS(v) as an indicator function that evaluates to 0 if v ∈ S and ∞ otherwise. This leads to a constrained optimization problem:

∥AD(v) − y∥22 s.t. IS(v) = 0, (5)

arg min

v

While this is intractable, we approach a local solution using projected gradient descent (S-PGD), where in each iteration the gradient is projected onto the tangent plane to v, and the updated latent vector is retracted back onto the sphere to ensure a valid solution.

An alternative approach that facilitates standard gradient descent (S-GD) involves a “soft” enforcement of the prior. Here, we assume the distribution is Gaussian with its mean concentrated on the shell of the sphere, yielding:

arg min

v

∥AD(v) − y∥22 + λ ∥v∥22 − L 22 . (6)

Notably, in both Equations (5) and (6), the encoder is entirely absent; gradients are propagated only through the decoder. This is an unfortunate omission, as the encoder is an integral part of the prior, having been trained jointly with the decoder to map images into the latent manifold [37]. Our proposed method, SP3, utilizes both the encoder and the decoder to improve restoration performance, as we discuss in the next subsections.

Shown qualitatively in Figure 2, S-GD fails to recover sharp details, while S-PGD tends to deviate from the measurement constraints. In contrast, SP3 produces high-fidelity details while remaining faithful to the observed data. Quantitative results in Table 1 further demonstrate that SP3 outperforms both S-GD and S-PGD across PSNR, LPIPS [71], KID [72], and computational runtime.

#### 3.2 Spherical Encoder as projection operator

As established, formulating the optimization exclusively over the decoder space discards the rich manifold information captured by the encoder. To fully utilize the pretrained spherical prior, we introduce a projection operator that leverages the complete autoencoder architecture.

Denote by P(·)=D (S (E (·),σ)) the projection operator onto the image manifold using an SE and latent random noise of strength σ. Such an operator holds several unique properties that make it attractive for image restoration: (1) P projects degraded, out-of-distribution (OOD), images to semantically similar natural images; (2) each application of P produces a probable sample from the image manifold; and (3) successive applications further refine the sample’s quality.

We demonstrate those properties on the left side of Figure 3: We take 100 images from the AFHQ-Cat dataset and degrade them using Gaussian blur and additive Gaussian noise. On the degraded inputs, we repeatedly apply P with varying amounts of latent noise. We evaluate the results using LPIPS to measure distortion relative to the clean version of each input and KID to measure perceptual quality relative to the entire dataset.

Empirically, the initial projection validates our first claim: within the first few steps, both LPIPS and KID significantly drop from their starting values across all evaluated noise levels. This indicates a rapid and effective mapping of OOD inputs back toward the natural image distribution.

However, the inclusion of latent noise (σ >0) within S is critical to satisfying our second claim. As observed in the KID trajectories, a purely deterministic projection (σ =0) slowly diverge, and its perceptual quality remains noticeably worse than configurations with injected latent noise. These higher noise levels effectively stabilize the operator, ensuring it consistently produces more probable, realistic samples from the actual image manifold.

Successive applications of P further validate our third claim. Unconstrained, repeated projections push the sample deeper into the prior manifold, refining global perceptual quality (illustrated by a stable KID score), but inevitably drift from the clean images which created the degraded inputs (illustrated by the increase in LPIPS score).

Crucially, the ability to stably apply P in this manner is not a universal property of autoencoders. On the right side of Figure 3, we compare the SE projection against popular VAEs [73–76]. Instead of refining the sample, successive applications of encoding and decoding using VAEs cause both the LPIPS distortion and the KID perceptual metric to increase dramatically within just a few steps, effectively pushing the sample far from the natural image manifold. This stark contrast highlights the uniqueness of the SE: its tightly structured, bounded latent space, prevents such catastrophic divergence, uniquely enabling the stable, iterative behavior necessary in our case.

Algorithm 1: SP3: Spherical Priors for Plug-andPlay Restoration

Input: y; A; E; D; S; proxλ; K; σ Output: restored image xK

- 1 x0 ← initialization from y;
- 2 for k ← 1 to K do

- 3 vk ← E(xk−1) ; // Encode
- 4 vk ← S(vk, σ) ; // Noisy spherify
- 5 xprior ← D(vk) ; // Decode
- 6 xk ← proxλ(xprior) ; // Data proximal

- 7 return xK;

Denois. Deblur. Inp.

Paint. Inp. Rand. Inp. SR

2− xxk+1k

10−2

10−3

10−4

0 5 10 15 20 Steps

Figure 4: SP3 empirical convergence. MSE of consecutive iterations converges to zero. AFHQ Cat, y-axis is given in log-scale.

This eventual rise in distortion during unconstrained projection motivates our proposed method. We embed P within an alternating PnP framework by interleaving the unconstrained manifold projection P with a rigorous data-consistency step. This way, we harness P effectiveness without sacrificing data fidelity. Our method, SP3, effectively balances perceptual realism with adherence to the physical measurement constraints defined in Equation (2).

#### 3.3 SP3

In previous sections it became evident that the naive baseline solutions, S-PGD and S-GD, yield suboptimal results. Using only the decoder, these methods fail to fully exploit the learned representations of the spherical prior. Disregarding the encoder during the restoration process effectively discards a crucial component.

To address this, we approach the MAP estimation problem from an alternative perspective. Utilizing IS(v), the strictly constrained optimization problem in Equation (5) can be equivalently rewritten as an unconstrained objective:

∥AD(v) − y∥22 + IS(v) . (7)

arg min

v

To decouple the forward measurement model from the non-linear generative prior, we introduce an auxiliary image-space variable x and enforce the constraint x=D(v). Therefore:

arg min

v,x

∥Ax − y∥22 + IS(v) s.t. x = D(v). (8)

Following the Half-Quadratic Splitting (HQS) method, we relax this exact constraint by introducing a quadratic penalty term parameterized by λ > 0. This yields the unconstrained penalty function:

Lλ(x,v) = ∥Ax − y∥22 + IS(v) + λ∥x − D(v)∥22. (9)

We minimize Lλ(x,v) by alternating between updating the latent representation v and the image variable x. Given a fixed intermediate state xk−1 from the previous iteration, updating the latent representation requires solving the following subproblem:

IS(v) + λ∥xk−1 − D(v)∥22 . (10)

vk = arg min

v

Solving Equation (10) exactly is intractable. The objective combines a non-smooth indicator function IS(v) with a non-linear decoder D, which precludes standard gradient-based optimization. Instead of solving this directly, we approximate the minimizer by accounting for the SE training objectives. During training, the loss function defined in Equation (1) encourages the encoder to behave such that x ≈ P(x), while forcing the latent to be on the learned manifold. Therefore, we bypass the optimization in Equation (10) and replace it directly with our projection operator P(·) introduced in Section 3.2.

By applying this operator, the encoder-decoder pair acts as an efficient PnP projector that maps the unregularized state xk−1 to a prior-consistent state:

xprior = P(xk−1). (11)

Inpainting

Denoising

Super-Resolution

100

60

3KID←×10

100

75

AFHQ

40

DPS OT-ODE

50

50

| |
|---|

20

D-Flow

25

PnP Flow

0

0

Flow Priors

DDRM DiffPIR PnP GS RED

0.1 0.2 0.3

0.5 1.0

0.2 0.4 0.6

Deblurring

Random Inpainting

Paintbrush Inpainting

80

200

60

3KID←×10

DPIR

60

CelebA

150

SP3

40

40

100

20

50

20

0.05 0.10 0.15

0.1 0.2

0.1 0.2 0.3

← LPIPS

← LPIPS

← LPIPS

- Figure 5: Perception-distortion evaluation. Comparison between SP3 (red sphere) and competing methods on perception (KID) and distortion (LPIPS) metrics on AFHQ (top) and CelebA (bottom). SP3 achieves at least comparable perception-distortion results while being significantly faster.

Given this fixed manifold projection xprior, the alternating update for x becomes a standard regularized inverse problem:

∥Ax − y∥22 + λ∥x − xprior∥22 . (12)

xk = arg min

x

Because A is a linear operator, Equation (12) represents a quadratic data-consistency step with a closed-form proximal solution:

xk = proxλ(xprior) = (A⊤A + λI)−1(A⊤y + λxprior). (13) Thus, each iteration of our method alternates between ensuring data consistency with the measurements and explicitly projecting the result onto the learned image manifold. This formulation gives rise to our simple and efficient restoration algorithm, summarized in Algorithm 1. Note that during the algorithm, differentiation is not required, thereby significantly reducing computational time.

To conclude this section, we discuss the convergence behavior of SP3. By construction, our alternating update scheme is an instance of HQS. In classical optimization, if the projection operator P(x) perfectly solves the prior proximal subproblem, HQS is guaranteed to converge to a stationary point of the MAP objective, even in non-convex settings [77, 78].

In the context of PnP methods, theoretically guaranteeing this convergence requires treating the update step as a fixed-point iteration, xk+1 = T(xk) = proxλ(P(xk)) [79]. Formal convergence to a fixed point x∞ ∈ Fix(T) requires strong assumptions regarding the contractive or averaged nature of the projection operator P(x) (see Appendix C for an extended fixed-point theorem and proofs). Because guaranteeing these properties for highly non-linear generative priors is restrictively strong in practice, we validate the convergence of SP3 empirically. As illustrated in Figure 4, the empirical sequence generated by our method rapidly stabilizes. Across all evaluated restoration tasks on the AFHQ-Cat dataset, the residual converges to near-zero within the first few iterations, demonstrating that SP3 robustly reaches a stable fixed point on the image manifold.

### 4 Experiments

We compare SP3 with zero-shot image restoration methods. The compared methods are divided into three categories: (1) PnP methods utilizing neural denoisers [2, 4, 79]; (2) PnP methods utilizing generative (diffusion/flow based) denoisers [5, 6]; and (3) diffusion/flow based zero-shot

Clean Degraded SP3 DDRM DPS DiffPIR PnP-Flow OT-ODE

[Figure 3]

[Figure 4]

- Figure 6: Qualitative comparison of image restoration. Results for AFHQ-Cat (top rows) and CelebA (bottom rows). The six rows correspond to the following degradation in descending order: inpainting, denoising, super-resolution, deblurring, paintbrush-inpainting, and random inpainting.

methods [24, 25, 31, 33, 32]. Following the recent PnP-Flow [6], we test the different methods on both AFHQ-Cat [80] at 256px resolution and CelebA [81] at 128px resolution. We use the test sets defined by [6]. All the results describe in this section are for 20 SP3 steps, unless specifically mentioned otherwise. Below we provide details about the experimental setting, where additional implementation details are provided in Appendix B.

Priors. For methods utilizing flow models, we use the priors trained by PnP-Flow [6]. For diffusionbased methods, we use publicly available diffusion models [82, 83]. Classical PnP methods use the DRUnet and GS-DRUnet implemented by the DeepInv library [84]. For SE, we use the official code [85] to train an AFHQ-Cat model using the official hyper-parameters described in [37], and a CelebA model using the same settings for 100 epochs. To ensure fair comparison between SE and diffusion/flow-based methods, we validate that the generative models used as the backbone for all methods mostly achieve similar KID results for generation. See Appendix B for additional details.

Degradations. For each dataset, we evaluate all methods across six noisy image restoration tasks, in the form of Equation (2): (1) Gaussian denoising with σ = 0.4 for AFHQ-Cat and σ = 0.2 for CelebA; (2) Deblurring using a 61 × 61 Gaussian kernel with σblur = 3.0 for AFHQ-Cat and σblur = 1.0 for CelebA; (3) Super-resolution using a 4× bicubic downsampling kernel for AFHQ-Cat and 2× for CelebA; (4) Box-inpainting with a centered square mask of size 80px for AFHQ-Cat and 40px for CelebA; (5) Random pixel inpainting with 70% masked pixels; and (6) paintbrush inpainting. For random inpainting, we add Gaussian noise with σ = 0.02 on AFHQ-Cat and σ = 0.01 on CelebA; for all other restoration tasks, we use σ = 0.1 on AFHQ-Cat and σ = 0.05 on CelebA.

Metrics. We measure both distortion and perception metrics as they are at odds with each other [44, 70]. We use LPIPS [71] as our main distortion metric, and KID [72] as our perception index. In

CelebA Deblurring

AFHQ Super-Resolution

3KID←×10

100

| | | | |
|---|---|---|---|
| | | | |
| | | | |

100

10

20

5

1

1 10 100

1 10 100

← Runtime [sec]

← Runtime [sec]

D-Flow

PnP Flow

DiffPIR PnP GS RED

DPIR

SP3

DPS OT-ODE

Flow Priors

DDRM

| |
|---|

- Figure 7: Perceptual quality vs. running time. Comparison between SP3 (red sphere) and competing methods on perception (KID) and running time (sec). SP3 achieves at least comparable perceptual quality at fraction of the time, achieving 3-630× speedup. SP3 results are displayed for 1, 3, 5, 10, and 20 steps.

Clean Degraded

[Figure 5]

1 step 3 steps 5 steps

Figure 8: Anytime results. Results of anytime restoration for super-resolution on AFHQ.

Appendix A we report the complete set of results in table form, including PSNR as an additional distortion metric that favors smooth reconstructions. Moreover, we report the per-image wall-clock runtime of each method on a single NVIDIA L40S GPU.

Comparisons. Figure 5 illustrates quantitative results of SP3 (red sphere) and the competing methods on a perception-distortion plane [44]. The top part shows results for AFHQ, and the bottom part for CelebA, with each plot showing a different degradation. SP3 achieves comparable results to those of the best competing methods, typically DPS [25] and OT-ODE [31]. Qualitative comparisons illustrated in Figure 6, where SP3 produces sharp results with fine details that are consistent with the degraded input and loyal to the clean image. See Appendix A for additional visual comparisons.

Speed & anytime generation. Figures 5 and 6 do not reveal one of the key advantages of SP3, its speed and anytime capabilities. Because SE projects degraded images onto the clean-image manifold, SP3 achieves perceptual results comparable to those of competing methods even after a single step. This is illustrated in Figure 7, where we compare the different methods on the perception-runtime plane. Even after a single step (leftmost point), SP3 is better than several competing methods, and after 20 steps (rightmost point), SP3 is comparable to the best competing methods while being significantly faster. The speed advantage stems from the small number of steps required, without the need to compute gradients. Concretely, SP3 using 20 steps is ×8 faster than OT-ODE and ×60 faster than DPS, which are its most direct competitors in terms of visual quality as seen in Figures 5 and 6. Finally, Figure 8 presents an example of the restoration trajectory, demonstrating SP3 anytime capability. Even after a single step, we obtain a usable, sharp result that can be further refined.

### 5 Conclusion and limitations

We introduce SP3, a fast, gradient-free Plug-and-Play algorithm that leverages Spherical Encoders to solve the MAP image restoration problem. By interleaving robust manifold projections with a closed-form data-consistency step, SP3 achieves anytime generation and perceptual quality matching state-of-the-art diffusion and flow methods, at a fraction of the computational cost.

Despite its empirical success, our method presents some limitations. As the projection operator blindly forces intermediate states back onto the clean image manifold, SP3 strongly prioritizes perceptual realism. Consequently, it naturally lacks in classical distortion metrics and cannot provide Minimum Mean Square Error (MMSE) solutions, which inherently favor smoother, averaged outputs. Additionally, while SP3 efficiently solves the MAP objective, it is not a posterior sampler like DPS [25]. Investigating how the structural latent noise could be actively leveraged to sample from and explore the posterior distribution [86, 87] is a promising direction for future work.

Societal impacts. Our method restores clean images from partial measurements, benefiting applications like smartphone photography and noisy imaging. However, it poses dual-use risks, including unauthorized surveillance and deepfake generation.

### Acknowledgments and Disclosure of Funding

The authors acknowledge Modal Labs for providing some of the cloud computing platform and GPUs resources (8× NVIDIA L40S) used for conducting part of the experiments presented in this paper.

### References

- [1] Singanallur V. Venkatakrishnan, Charles A. Bouman, and Brendt Wohlberg. Plug-and-Play priors for model based reconstruction. In 2013 IEEE Global Conference on Signal and Information Processing, pages 945–948, December 2013.
- [2] Yaniv Romano, Michael Elad, and Peyman Milanfar. The Little Engine That Could: Regularization by Denoising (RED). SIAM Journal on Imaging Sciences, 10(4):1804–1844, January 2017.
- [3] Kai Zhang, Wangmeng Zuo, Shuhang Gu, and Lei Zhang. Learning deep cnn denoiser prior for image restoration. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3929–3938, 2017.
- [4] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-andPlay Image Restoration With Deep Denoiser Prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6360–6376, October 2022.
- [5] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising Diffusion Models for Plug-and-Play Image Restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1219–1229, 2023.
- [6] Ségolène Tiffany Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl. Pnp-flow: Plugand-play image restoration with flow matching. In The Thirteenth International Conference on Learning Representations, 2025.
- [7] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014.
- [8] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [9] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In International Conference on Learning Representations, 2017.
- [10] Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G Dimakis. Compressed sensing using generative models. In International conference on machine learning, pages 537–546. PMLR, 2017.
- [11] Paul Hand, Oscar Leong, and Vlad Voroninski. Phase retrieval under a generative prior. Advances in Neural Information Processing Systems, 31, 2018.
- [12] Muhammad Asim, Max Daniels, Oscar Leong, Ali Ahmed, and Paul Hand. Invertible generative models for inverse problems: mitigating representation error and dataset bias. In International conference on machine learning, pages 399–409. PMLR, 2020.
- [13] Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, and Cynthia Rudin. Pulse: Selfsupervised photo upsampling via latent space exploration of generative models. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 2437–2445, 2020.
- [14] Jay Whang, Qi Lei, and Alex Dimakis. Solving inverse problems with a flow-based noise model. In International Conference on Machine Learning, pages 11146–11157. PMLR, 2021.
- [15] Xinyi Wei, Hans Van Gorp, Lizeth Gonzalez-Carabarin, Daniel Freedman, Yonina C Eldar, and Ruud JG van Sloun. Deep unfolding with normalizing flow priors for inverse problems. IEEE Transactions on Signal Processing, 70:2962–2971, 2022.

- [16] Jean Prost, Antoine Houdard, Andrés Almansa, and Nicolas Papadakis. Inverse problem regularization with hierarchical variational autoencoders. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22894–22905, 2023.
- [17] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, pages 2256–2265. PMLR, June 2015.
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.
- [19] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In International Conference on Learning Representations, October 2020.
- [20] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.
- [21] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.
- [22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.
- [23] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal of Machine Learning Research, 26(209):1– 80, 2025.
- [24] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising Diffusion Restoration Models. Advances in Neural Information Processing Systems, 35:23593–23606, December 2022.
- [25] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion Posterior Sampling for General Noisy Inverse Problems. In The Eleventh International Conference on Learning Representations, September 2022.
- [26] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model. (arXiv:2212.00490), December 2022.
- [27] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-Guided Diffusion Models for Inverse Problems. In International Conference on Learning Representations, September 2022.
- [28] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alexandros G. Dimakis, and Sanjay Shakkottai. Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models, July 2023.
- [29] Peiqing Yang, Shangchen Zhou, Qingyi Tao, and Chen Change Loy. PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance. Advances in Neural Information Processing Systems, 36:32194–32214, December 2023.
- [30] Zongsheng Yue and Chen Change Loy. DifFace: Blind Face Restoration with Diffused Error Contraction. (arXiv:2212.06512), December 2023.
- [31] Ashwini Pokle, Matthew J. Muckley, Ricky T. Q. Chen, and Brian Karrer. Training-free linear image inverses via flows. Transactions on Machine Learning Research, 2024.
- [32] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. In International Conference on Machine Learning, pages 3462–3483. PMLR, 2024.

- [33] Yasi Zhang, Peiyu Yu, Yaxuan Zhu, Yingshan Chang, Feng Gao, Ying N Wu, and Oscar Leong. Flow priors for linear inverse problems via iterative corrupted trajectory matching. Advances in Neural Information Processing Systems, 37:57389–57417, 2024.
- [34] Ron Raphaeli, Sean Man, and Michael Elad. Silo: Solving inverse problems with latent operators. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10570–10580, October 2025.
- [35] Sean Man, Guy Ohayon, Ron Raphaeli, Matan Kleiner, and Michael Elad. Elad: Blind face restoration using expectation-based likelihood approximation and diffusion prior. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–12, 2025.
- [36] Tomer Garber and Tom Tirer. Zero-shot image restoration using few-step guidance of consistency models (and beyond). In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2398–2407, 2025.
- [37] Kaiyu Yue, Menglin Jia, Ji Hou, and Tom Goldstein. Image generation with a sphere encoder. arXiv preprint arXiv:2602.15030, 2026.
- [38] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504–507, 2006.
- [39] Bin Dai and David Wipf. Diagnosing and enhancing VAE models. In International Conference on Learning Representations, 2019.
- [40] Danilo Jimenez Rezende and Fabio Viola. Taming vaes. arXiv preprint arXiv:1810.00597, 2018.
- [41] Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian Goodfellow. Adversarial autoencoders. In International Conference on Learning Representations, 2016.
- [42] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015.
- [43] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016.
- [44] Yochai Blau and Tomer Michaeli. The Perception-Distortion Tradeoff. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6228–6237, 2018.
- [45] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2):295–307, 2015.
- [46] Jian Sun, Wenfei Cao, Zongben Xu, and Jean Ponce. Learning a convolutional neural network for non-uniform motion blur removal. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 769–777, 2015.
- [47] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In European conference on computer vision, pages 649–666. Springer, 2016.
- [48] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE Transactions on Image Processing, 26(7):3142–3155, July 2017.
- [49] Idan Kligvasser, Tamar Rott Shaham, and Tomer Michaeli. xunit: Learning a spatial activation function for efficient image restoration. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2433–2442, 2018.
- [50] Javier Gurrola-Ramos, Oscar Dalmau, and Teresa E. Alarcón. A Residual Dense U-Net Neural Network for Image Denoising. IEEE Access, 9:31742–31754, 2021.

- [51] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. SwinIR: Image Restoration Using Swin Transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1833–1844, 2021.
- [52] Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4681–4690, 2017.
- [53] Andreas Lugmayr, Martin Danelljan, Luc Van Gool, and Radu Timofte. SRFlow: Learning the Super-Resolution Space with Normalizing Flow. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 715–732, Cham,

2020. Springer International Publishing.

- [54] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training RealWorld Blind Super-Resolution With Pure Synthetic Data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1905–1914, 2021.
- [55] Guy Ohayon, Theo Adrai, Gregory Vaksman, Michael Elad, and Peyman Milanfar. High Perceptual Quality Image Denoising With a Posterior Sampling CGAN. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1805–1813, 2021.
- [56] Jay Whang, Mauricio Delbracio, Hossein Talebi, Chitwan Saharia, Alexandros G. Dimakis, and Peyman Milanfar. Deblurring via Stochastic Refinement. pages 16293–16303, 2022.
- [57] Sean Man, Guy Ohayon, Theo Adrai, and Michael Elad. High-Perceptual Quality JPEG Decoding via Posterior Sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1272–1282, 2023.
- [58] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-Image Diffusion Models. In ACM SIGGRAPH 2022 Conference Proceedings, SIGGRAPH ’22, pages 1–10, New York, NY, USA, July 2022. Association for Computing Machinery.
- [59] Yuchao Gu, Xintao Wang, Liangbin Xie, Chao Dong, Gen Li, Ying Shan, and Ming-Ming Cheng. VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 126–143, Cham, 2022. Springer Nature Switzerland.
- [60] Shangchen Zhou, Kelvin Chan, Chongyi Li, and Chen Change Loy. Towards Robust Blind Face Restoration with Codebook Lookup Transformer. Advances in Neural Information Processing Systems, 35:30599–30611, December 2022.
- [61] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022.
- [62] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. Transactions on Machine Learning Research, 2023. Featured Certification, Outstanding Certification.
- [63] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior. (arXiv:2308.15070), April 2024.
- [64] Guy Ohayon, Tomer Michaeli, and Michael Elad. Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration. (arXiv:2410.00418), October 2024.
- [65] Noam Elata, Hyungjin Chung, Jong Chul Ye, Tomer Michaeli, and Michael Elad. Invfusion: Bridging supervised and zero-shot diffusion for inverse problems. arXiv preprint arXiv:2504.01689, 2025.

- [66] Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A Variational Perspective on Solving Inverse Problems with Diffusion Models. In The Twelfth International Conference on Learning Representations, October 2023.
- [67] Yuyang Hu, Mauricio Delbracio, Peyman Milanfar, and Ulugbek Kamilov. A restoration network as an implicit prior. In The Twelfth International Conference on Learning Representations, 2024.
- [68] Matthieu Terris, Ulugbek S Kamilov, and Thomas Moreau. Fire: Fixed-points of restoration priors for solving inverse problems. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23185–23194, 2025.
- [69] Xingang Pan, Xiaohang Zhan, Bo Dai, Dahua Lin, Chen Change Loy, and Ping Luo. Exploiting deep generative prior for versatile image restoration and manipulation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):7474–7489, 2021.
- [70] Dror Freirich, Tomer Michaeli, and Ron Meir. A Theory of the Distortion-Perception Tradeoff in Wasserstein Space. In Advances in Neural Information Processing Systems, volume 34, pages 25661–25672. Curran Associates, Inc., 2021.
- [71] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.
- [72] Mikołaj Bi´nkowski, Danica J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, February 2018.
- [73] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-Resolution Image Synthesis With Latent Diffusion Models. pages 10684–10695, 2022.
- [74] Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Eq-vae: Equivariance regularized latent space for improved generative image modeling. arXiv preprint arXiv:2502.09509, 2025.
- [75] Black Forest Labs. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [76] Black Forest Labs. FLUX.2: Analyzing and enhancing the latent space of FLUX – representation comparison, 2025.
- [77] Jérôme Bolte, Shoham Sabach, and Marc Teboulle. Proximal alternating linearized minimization for nonconvex and nonsmooth problems. Math. Program., 146(1–2):459–494, August 2014.
- [78] Mila Nikolova and Michael K. Ng. Analysis of half-quadratic minimization methods for signal and image recovery. SIAM Journal on Scientific Computing, 27(3):937–966, 2005.
- [79] Samuel Hurault, Arthur Leclaire, and Nicolas Papadakis. Gradient step denoiser for convergent plug-and-play. In International Conference on Learning Representations, 2022.
- [80] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8188–8197, 2020.
- [81] Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaoou Tang. From facial parts responses to face detection: A deep learning approach. In Proceedings of the IEEE international conference on computer vision, pages 3676–3684, 2015.
- [82] Chen Henry Wu and Fernando De la Torre. A latent space of stochastic diffusion models for zero-shot image editing and guidance. In ICCV, 2023.
- [83] rdruce. ddpm-celeb-128. https://huggingface.co/rdruce/ddpm-celeb-128. Accessed: 2026-05-07.

- [84] Julián Tachella, Matthieu Terris, Samuel Hurault, Andrew Wang, Leo Davy, Jérémy Scanvic, Victor Sechaud, Romain Vo, Thomas Moreau, Thomas Davies, Dongdong Chen, Nils Laurent, Brayan Monroy, Jonathan Dong, Zhiyuan Hu, Minh-Hai Nguyen, Florian Sarron, Pierre Weiss, Paul Escande, Mathurin Massias, Thibaut Modrzyk, Brett Levac, Tobías I. Liaudat, Maxime Song, Johannes Hertrich, Sebastian Neumayer, and Georg Schramm. Deepinverse: A python package for solving imaging inverse problems with deep learning. Journal of Open Source Software, 10(115):8923, 2025.
- [85] Meta. Sphere encoder in pytorch. https://github.com/facebookresearch/ sphere-encoder.
- [86] Noa Cohen, Hila Manor, Yuval Bahat, and Tomer Michaeli. From Posterior Sampling to Meaningful Diversity in Image Restoration. October 2023.
- [87] Elias Nehme and Tomer Michaeli. Generative AI for Solving Inverse Problems in Computational Imaging. XRDS, 31(2):32–37, January 2025.
- [88] R. Tyrrell Rockafellar and Roger J-B Wets. Variational Analysis, volume 317. Springer Science & Business Media, 2009.
- [89] Heinz H Bauschke and Patrick L Combettes. Correction to: convex analysis and monotone operator theory in hilbert spaces. In Convex analysis and monotone operator theory in Hilbert spaces, pages C1–C4. Springer, 2020.

### A Additional Comparisons

Figures S1-S6 and Figures S7-S12 illustrate qualitative comparisons of all methods on AFHQ-Cat and CelebA, respectively.

Tables S1 and S2, and Tables S3 and S4 provide PSNR and LPIPS for distortion metrics, and KID for perception metric, for all methods AFHQ-Cat and CelebA, respectively.

Table S5 provides running time (in seconds) of all methods for a single image.

#### A.1 Qualitative comparisons

Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 6]

[Figure 7]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 8]

[Figure 9]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

- Figure S1: Qualitative comparison of image restoration results on AFHQ-Cat (deblurring).

[Figure 10]

[Figure 11]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 12]

[Figure 13]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

- Figure S2: Qualitative comparison of image restoration results on AFHQ-Cat (denoising).

[Figure 14]

[Figure 15]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 16]

[Figure 17]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

- Figure S3: Qualitative comparison of image restoration results on AFHQ-Cat (inpainting).

[Figure 18]

[Figure 19]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 20]

[Figure 21]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

- Figure S4: Qualitative comparison of image restoration results on AFHQ-Cat (paintbrush-inpainting).

[Figure 22]

[Figure 23]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 24]

[Figure 25]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

- Figure S5: Qualitative comparison of image restoration results on AFHQ-Cat (random inpainting).

[Figure 26]

[Figure 27]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff

[Figure 28]

[Figure 29]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

- Figure S6: Qualitative comparison of image restoration results on AFHQ-Cat (super-resolution).

[Figure 32]

[Figure 33]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

##### Figure S7: Qualitative comparison of image restoration results on CelebA (deblurring).

Clean Degraded SP3 DDRM DPS PnP-Diff PnP-GS

[Figure 34]

[Figure 35]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff PnP-GS

[Figure 36]

[Figure 37]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

##### Figure S8: Qualitative comparison of image restoration results on CelebA (denoising).

[Figure 40]

[Figure 41]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

##### Figure S9: Qualitative comparison of image restoration results on CelebA (inpainting).

Clean Degraded SP3 DDRM DPS PnP-Diff PnP-GS

[Figure 42]

[Figure 43]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff PnP-GS

[Figure 44]

[Figure 45]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

##### Figure S10: Qualitative comparison of image restoration results on CelebA (paintbrush-inpainting).

[Figure 48]

[Figure 49]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

##### Figure S11: Qualitative comparison of image restoration results on CelebA (random-inpainting).

Clean Degraded SP3 DDRM DPS PnP-Diff PnP-GS

[Figure 50]

[Figure 51]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED Clean Degraded SP3 DDRM DPS PnP-Diff PnP-GS

[Figure 52]

[Figure 53]

PnP-Flow OT-ODE D-Flow Flow Priors DPIR RED

##### Figure S12: Qualitative comparison of image restoration results on CelebA (super-resolution).

#### A.2 Quantitative results

Method Denoising Deblurring Inpainting KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑

- DPS 3.63 0.159 26.38 2.69 0.178 24.72 3.91 0.107 25.85 OT-ODE 7.71 0.146 27.32 6.74 0.155 26.79 8.53 0.129 22.95 DiffPIR 5.74 0.235 23.05 128.68 0.584 16.71 15.06 0.267 23.72 PnP-Flow 36.74 0.259 29.49 64.01 0.446 25.91 58.89 0.328 25.57 DDRM 48.83 0.336 26.74 41.93 0.325 27.34 40.21 0.220 26.54 RED 55.42 0.414 25.99 49.44 0.398 28.25 106.65 0.206 17.20 DPIR 66.70 0.336 27.97 40.84 0.401 28.08 122.08 0.230 17.27 Flow Priors 38.86 0.700 19.33 21.18 0.297 26.46 30.95 0.301 24.34 D-Flow 60.17 1.039 14.97 7.81 0.170 27.20 20.70 0.224 23.71 SP3 4.09 0.174 22.84 2.56 0.118 25.75 6.68 0.103 25.12

Table S1: Quantitative evaluation on AFHQ cats. KID is scaled by ×103.

Method Paintbrush Inp. Random Inp. Superresolution KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑

- DPS 4.53 0.103 26.92 3.15 0.100 29.26 3.50 0.201 24.32 OT-ODE 9.10 0.120 24.14 4.34 0.118 28.80 6.40 0.173 26.34 DiffPIR 16.69 0.263 24.26 9.55 0.159 27.26 5.59 0.242 22.60 PnP-Flow 57.43 0.321 26.34 66.44 0.438 25.16 98.44 0.666 18.09 DDRM 40.20 0.213 27.55 20.93 0.114 32.32 50.64 0.371 25.42 RED 59.57 0.159 18.19 360.24 1.171 8.31 45.91 0.373 27.95 DPIR 96.43 0.198 18.29 333.30 1.139 7.97 47.11 0.398 27.42 Flow Priors 23.60 0.278 25.23 16.51 0.145 29.77 55.70 0.417 24.41 D-Flow 19.14 0.217 24.40 6.64 0.061 32.24 35.15 0.568 21.28 SP3 6.91 0.099 25.73 3.71 0.085 27.62 3.47 0.151 24.24

Table S2: Quantitative evaluation on AFHQ cats. KID is scaled by ×103.

Method Denoising Deblurring Inpainting KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑

OT-ODE 7.70 0.044 30.47 5.90 0.037 32.80 8.29 0.050 27.57 DPS 17.52 0.041 31.30 14.67 0.041 32.09 16.16 0.055 25.14 DPIR 44.45 0.071 32.02 19.48 0.046 35.42 121.67 0.250 14.88 PnP-Flow 45.58 0.114 29.95 57.95 0.184 27.90 52.23 0.136 27.61 DiffPIR 64.02 0.109 26.47 272.39 0.951 10.73 74.04 0.105 22.61 RED 62.99 0.126 30.02 29.82 0.049 34.81 136.61 0.261 14.85 Flow Priors 86.68 0.234 24.73 22.96 0.064 31.52 12.27 0.058 29.65 DDRM 103.42 0.113 29.89 65.43 0.067 32.95 68.42 0.089 26.80 D-Flow 95.72 0.390 20.72 67.70 0.027 30.60 48.90 0.025 30.74 PnP-GS 187.39 0.418 21.46 21.22 0.057 33.41 128.87 0.155 22.97 SP3 57.10 0.147 25.45 7.95 0.057 31.13 34.38 0.0522 27.23

Table S3: Quantitative evaluation on CelebA. KID is scaled by ×103.

Method Paintbrush Inp. Random Inp. Superresolution KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑ KID ↓ LPIPS ↓ PSNR ↑

OT-ODE 12.01 0.099 22.93 20.53 0.084 27.61 6.28 0.047 31.30 DPS 22.72 0.104 21.49 7.79 0.026 31.91 15.86 0.045 31.63 DPIR 162.28 0.291 12.54 425.53 1.018 8.45 25.84 0.067 32.75 PnP-Flow 54.00 0.160 25.06 73.93 0.233 23.32 80.71 0.272 22.14 PnP-Diff 83.56 0.175 19.07 81.34 0.118 26.70 49.54 0.067 28.76 RED 224.26 0.304 12.47 434.20 1.015 8.87 29.05 0.062 33.65 Flow Priors 33.91 0.097 24.97 18.45 0.072 29.05 33.75 0.119 28.53 DDRM 86.61 0.160 23.21 26.63 0.041 31.04 87.84 0.091 30.51 D-Flow 51.96 0.045 27.43 7.08 0.021 33.28 71.54 0.044 28.62 PnP-GS 112.76 0.233 19.45 343.59 0.638 12.46 24.44 0.072 31.33 SP3 50.00 0.088 23.48 22.39 0.080 26.06 13.80 0.067 29.46

Table S4: Quantitative evaluation on CelebA. KID is scaled by ×103.

#### Method Runtime [sec]

OT-ODE 2.29 DPS 17.77 DPIR 0.044 PnP-Flow 3.20 DiffPIR 0.83 RED 0.37 Flow Priors 29.88 DDRM 0.81 D-Flow 183.18 PnP-GS 0.31 SP3 (1 step) 0.016 SP3 (3 step) 0.043 SP3 (5 step) 0.073 SP3 (10 step) 0.14 SP3 (20 step) 0.28

Table S5: Average inference time per image. Reported per-image wall-clock runtimes were measured using a single NVIDIA L40S GPU for all methods.

### B Implementation details

SP3 (Algorithm 1) is primarily governed by two key hyper-parameters: the data prox regularization coefficient, λ, and the injected latent noise scale, σ. Note that σ denotes the relative noise strength with respect to the maximum noise level the prior was trained on (equivalent to r in [37]). In addition to these parameters, the number of forward steps can be adjusted; however, this serves as a flexible parameter rather than a strict constraint due to the anytime restoration capabilities of SP3.

Furthermore, the algorithm requires an initialization strategy for the starting state, x0, which we tailor to the specific degradation. For denoising and deblurring, we apply the adjoint operator. For inpainting problems, we utilize a GPU-friendly, convolution-based median filter approximation. Finally, for super-resolution, we initialize using standard bicubic upsampling.

The optimal values for λ and σ across the evaluated inverse problems are detailed in Tables S6 and S7. These configurations were determined empirically via a grid search on a held-out validation set.

- Table S6: Hyper-parameters for AFHQ Cat

Problem Steps Init Mode λ σ

Denoising 20 A⊤ 0.04 0.08 Gaussian Deblurring 20 A⊤ 1.50 0.32 Inpainting 20 Conv 0.30 0.10 Paintbrush Inpainting 20 Conv 0.30 0.10 Random Inpainting 20 Conv 1.60 0.05 Super-resolution 20 Bicubic 2.20 0.15

- Table S7: Hyper-parameters for CelebA

Problem Steps Init Mode λ σ

Denoising 20 A⊤ 0.10 0.05 Gaussian Deblurring 20 A⊤ 0.80 0.05 Inpainting 20 Conv 0.45 0.05 Paintbrush Inpainting 20 Conv 0.80 0.15 Random Inpainting 20 Conv 0.45 0.05 Super-resolution 20 Bicubic 1.20 0.05

Each experiment used a single NVIDIA L40S GPU or NVIDIA A40 GPU, each with 49GB of memory.

#### B.1 Comparisons between different priors

To ensure fair comparison between Spherical Encoder and diffusion/flow-based methods, we validate that the generative models used as the backbone for all methods mostly achieve similar KID results for generation, as illustrated in Table S8.

Prior AFHQ-Cat CelebA Diffusion 14.1 104.4∗ Flow 10.78 11.4 Sphere Encoder 9.5 10.4

Table S8: KID ×103 of different priors on AFHQ-Cat and CelebA. ∗Despite the high KID number achieved by the diffusion model prior [83], diffusion based methods (e.g., DPS) achieve state-of-theart results.

#### B.2 Licences

AFHQ [80] is released under CC BY-NC 4.0; CelebA [81] is released under a custom non-commercial research license.

### C Theoretical Analysis and Convergence

While classical Half-Quadratic Splitting guarantees convergence for specific regularizers [78], extending these guarantees to highly non-linear deep generative priors requires treating the alternating updates as a non-convex optimization problem. Under the assumption that the objective satisfies the Kurdyka-Łojasiewicz property, alternating minimization schemes are guaranteed to converge to a critical point [77]. More recently, similar theoretical extensions have been specifically proven for HQS-based Plug-and-Play architectures [79].

To formally analyze the convergence behavior of SP3 as a fixed-point iteration, we follow standard fixed-point and convex optimization theory [88, 89]. While the assumptions required for formal fixed-point convergence are restrictive and rarely strictly satisfied by deep generative models, they provide the mathematical framework that motivates our empirical analysis in Section 3.

#### C.1 Preliminaries and Notations

Let X = Rn be a finite-dimensional image space, V = Rm be a finite-dimensional latent space, and Y = Rk be a finite-dimensional measurement space, with a linear forward operator A : X → Y containing a non-trivial null-space.

Our algorithm, SP3, updates the image state via the operator T:

T(x) = proxηℓ(P(x)), (S1)

where P(x) = D(S(E(x))) is the generative prior mapping defined in the main text. The data-fidelity proximal operator is defined for a convex loss ℓ(x) = ∥Ax − y∥2 as:

- 1

- 2η∥x − z∥2 . (S2)

proxηℓ(z) = arg min

ℓ(x) +

x∈X

Note that setting η = 21λ exactly recovers the update step defined in Equation (12).

Definition 1 (Nonexpansive and Averaged Operators). An operator T : X → X is nonexpansive if it is 1-Lipschitz continuous (∥T (x) − T (y)∥ ≤ ∥x − y∥). It is α-averaged if there exists a nonexpansive operator R and a constant α ∈ (0,1) such that T = (1 − α)I + αR. An operator is firmly nonexpansive if it is exactly 1/2-averaged.

Lemma 1 (Composition). The composition of an α1-averaged operator and an α2-averaged operator is an α3-averaged operator, where α3 ∈ (0,1) [89].

#### C.2 Proof of Convergence

Because ℓ(x) is closed, proper, and convex, its proximal operator proxηℓ is firmly nonexpansive (1/2-averaged). If we assume the deep prior P acts as an averaged operator, we can guarantee convergence to a fixed point.

Theorem 1 (Convergence to a fixed point). Assume that the prior mapping P is α-averaged on X and the fixed-point set Fix(T) = {x : T(x) = x} is nonempty. Then, the sequence generated by xk+1 = T(xk) converges to a fixed point x∞ ∈ Fix(T).

Proof. By Lemma 1, the composition of the 1/2-averaged proxηℓ and the α-averaged P means the update operator T is β-averaged for some β ∈ (0,1). A fundamental property of a β-averaged operator is that for any x,y ∈ X:

1 − β β ∥(I − T)(x) − (I − T)(y)∥2. (S3)

∥T(x) − T(y)∥2 ≤ ∥x − y∥2 −

Let x∗ ∈ Fix(T). Substituting y = x∗ (noting T(x∗) = x∗) and our iteration xk+1 = T(xk), we obtain:

1 − β β ∥xk − xk+1∥2. (S4)

∥xk+1 − x∗∥2 ≤ ∥xk − x∗∥2 −

Because the subtracted term is non-negative, ∥xk+1 − x∗∥ ≤ ∥xk − x∗∥. Therefore, the sequence {xk} is Fejér monotone, bounded, and non-increasing in distance to any fixed point.

Summing Equation (S4) from k = 0 to K − 1 yields a telescoping sum bounding the residuals:

K−1

1 − β β

∥xk − xk+1∥2 ≤ ∥x0 − x∗∥2. (S5)

k=0

Because the infinite sum is bounded by a finite constant, the individual terms must approach zero (∥xk − xk+1∥ → 0), demonstrating the sequence is asymptotically regular. In a finite-dimensional space, the Krasnosel’skii-Mann theorem dictates that a sequence which is both bounded and asymptotically regular under an averaged operator strictly converges to a point in the fixed-point set. Thus, xk → x∞ ∈ Fix(T). ■

Remark 1 (On Linear Convergence). If the data fidelity term ℓ(x) was strongly convex, its proximal operator would be a strict contraction, yielding a linear convergence rate via the Banach Fixed Point Theorem. However, in typical image restoration tasks (e.g., deblurring, inpainting), the forward operator A has a non-trivial null-space, meaning ℓ(x) is not strongly convex. Consequently, theoretical bounds on the convergence rate are difficult to establish, which motivates the empirical convergence validation provided in Figure 4.

