## Scale Space Diffusion

Soumik Mukhopadhyay∗

soumik@umd.edu

Prateksha Udhayanan∗

pudhayan@umd.edu

University of Maryland, College Park

Abhinav Shrivastava

abhinav2@umd.edu

# arXiv:2603.08709v1[cs.CV]9Mar2026

#### Abstract

Diffusion models degrade images through noise, and reversing this process reveals an information hierarchy across timesteps. Scale-space theory exhibits a similar hierarchy via low-pass filtering. We formalize this connection and show that highly noisy diffusion states contain no more information than small, downsampled images - raising the question of why they must be processed at full resolution. To address this, we fuse scale spaces into the diffusion process by formulating a family of diffusion models with generalized linear degradations and practical implementations. Using downsampling as the degradation yields our proposed Scale Space Diffusion. To support Scale Space Diffusion, we introduce Flexi-UNet, a UNet variant that performs resolutionpreserving and resolution-increasing denoising using only the necessary parts of the network. We evaluate our framework on CelebA and ImageNet and analyze its scaling behavior across resolutions and network depths. Our project website is available publicly.

#### 1. Introduction

Diffusion models [16, 36] are a class of generative models that achieve image synthesis by reversing an iterative noising process. It has been observed that states at different stages of the diffusion process encode different types of information [28]. As shown along the y-axis of Fig. 1(a), increasing diffusion noise progressively removes fine facial details while retaining only coarse structure. Eventually, with sufficient noising, even this structural information is lost. This illustrates that diffusion timesteps form an intrinsic information hierarchy.

A similar property underlies scale space theory [24], a fundamental subfield of computer vision. Scale spaces also represent image signals in an information-hierarchical manner through successive low-pass filtering. Along the x-axis of Fig. 1(a), we see the loss of details as the resolution decreases in a Gaussian pyramid, mirroring the information

*Equal contribution.

[Figure 1]

[Figure 2]

ScaleSpaceDiffusion

[Figure 3]

[Figure 4]

[Figure 5]

Resize + noise

DiffusionNoising

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Scale Space (Gaussian Pyramid)

(a)

- 4

CelebA-64x64 (1M iterations)

- 5

2

CelebA-128x128 (300k iterations)

10

FID

CelebA-256x256 (300k iterations)

10

5

40 60 80 Train time (hrs)

DDPM Ours (multi-scale)

(b)

Figure 1. (a) Our proposed Scale Space Diffusion fuses scale spaces into diffusion models. (b) We show trends in image generation performance versus time for our proposed Flexi-UNet for CelebA64, CelebA-128, and CelebA-256. Multiple point on the same plot represent our models with different number levels (i.e., number of intermediate resolutions). We see immense gains in efficiency with resolution scaling while having reasonable performance.

dissipation in the diffusion process. The main distinction lies in the mechanism of information degradation: diffusion uses iterative noising, whereas scale spaces use progressive blurring or downsampling.

We investigate this relationship between diffusion and scale spaces formally through a preliminary mathematical modeling of information in both processes. This reveals striking parallels in their information content, suggesting a fundamental connection between the two. Intuitively, one may ask why completely noisy images should be processed at high resolution when they contain information equivalent to that of a tiny image. These parallels indicate that the two axes in Fig. 1(a) correspond to different but compatible ways of information degradation.

In this work, we revisit pixel diffusion to achieve a unification of scale spaces and the diffusion process. Previous attempts at this either operate only at the highest resolution [3, 18], making them computationally inefficient, or rely on simplistic covariance assumptions [1] that may not hold in practice, or perform noisy scale shifting using high-frequency [2] or decorrelation noise [6, 19, 20], which

remain inference-time approximations. Unlike pyramidal flow-matching approaches that approximate scale changes only during inference, our formulation integrates scale transitions directly into the diffusion process. In contrast, we first develop a mathematical theory for diffusion processes under generalized linear degradations, yielding a family of diffusion processes. We further illustrate how these can be implemented in modern deep-learning frameworks. Next, using image resizing as the linear degradation, we realize a fusion of scale spaces with diffusion. We term this process Scale Space Diffusion (SSD). Denoising diffusion probabilistic models (DDPM) [16] emerge as a special case of SSD, corresponding to the trivial case of resizing to the same size, i.e., the identity operator. These generalized degradations naturally induce non-isotropic posteriors, which we handle through an implicit sampling procedure.

To realize the general version of Scale Space Diffusion, we require a neural network architecture capable of reversing the downsizing degradation, i.e., it must be able to upsample a noisy state. A na¨ıve approach could use a UNet [33] directly, but this would require even small-scale images to pass through the full network, leading to unnecessary computational cost. To address this, we propose a novel convolutional neural network (ConvNet) architecture that augments the standard UNet to use only the relevant levels of the network. It supports both resolution-preserving diffusion steps and next-resolution upscaling at all stages of a Gaussian pyramid. We denote this architecture as Flexi-UNet.

We analyze our framework and architectures on unconditional image generation using commonly used datasets of CelebA [25] and ImageNet [8]. To study the scaling properties of our method, we conduct experiments at multiple resolutions of CelebA dataset as shown in Fig. 1 (b). We observe that our models are faster during both training and inference while achieving reasonable FID scores. The key contributions of this work are:

- 1. We uncover and analyze the relationship between the states of diffusion models and the levels of scale spaces.
- 2. We build the mathematical foundation for a family of generalized linear diffusion processes, and techniques to implement them in modern deep-learning frameworks. With resizing as the choice for the linear degradation, we realize the fusion of diffusion and scale spaces, which we term Scale Space Diffusion.
- 3. To enable Scale Space Diffusion, we introduce a novel architecture Flexi-UNet capable of handling both resolution-changing as well as resolution-preserving reverse diffusion across multiple resolutions.

#### 2. Related Work

Diffusion Models. Diffusion Models have become the de facto standard for image generation in recent times. Early works such as DDPM [16] achieved high-quality image gen-

eration without adversarial training, but relied on simulating a Markov chain with a large number of steps for sampling. DDIM [35] accelerated the sampling process, while methods such as LDM [32] performed denoising in a compact latent space rather than directly in the pixel-space. Recently, DiTs have become popular, replacing traditional UNet based backbones with transformer architecture [31]. Motivated by the goal of scaling diffusion models for high-resolution image generation while maintaining architectural simplicity and high-frequency image details, we propose an end-toend Scale Space Diffusion model that performs denoising directly in the pixel domain.

Scale-Space Theory. Scale-space theory [24] is a fundamental concept in computer vision, that provides a framework for multi-scale image representation and analysis. It has been widely used in visual understanding tasks [5, 27]. The underlying idea of representation at multiple scales has been smartly used in the context of generative models to progressively generate images at increasing resolutions. In GANbased approaches, Progressive GAN [21] has shown excellent results in generating high-resolution images by learning to generate at increasing resolutions during the training process. In some other works such as LAPGAN [9], multiple GANs, one for each scale, are used to upscale the image by producing a residual, similar to a Laplacian pyramid.

Several works in the space of diffusion models have also drawn inspiration from scale-space theory. Cascaded diffusion model [17] consists of a series of diffusion models that generate images of increasing resolutions, where the base model produces a low-resolution image and subsequent super-resolution models refine it using the upsampled version of the low-resolution image as a condition. Matryoshka Diffusion [12] model proposes a diffusion process that denoises inputs at multiple resolutions jointly.

However, none of these approaches directly incorporate scale-space theory in the diffusion process because the noise component of the noisy intermediate state leads to correlated noise pixels at an upsampled state. Some works solve this by adding additional noise at the higher resolution. Relay Diffusion [37] imagines a low-resolution generation as a high-resolution image with block noise and trains a model to denoise it at higher resolution with a weighted combination of block noise and high-resolution noise. Laplacian Diffusion Models [2] train separate models for different resolutions and add a Laplacian residual of high-resolution noise during resolution transitions. However, simply adding high-resolution noise does not fully resolve the distribution mismatch between noisy states at different resolutions. Pyramidal Flow Matching [20] addresses this issue by adding decorrelated noise while also rolling the diffusion process back to a noisier timestep. PixelFlow [6] and Region Adaptive Latent Sampling [19] build on this idea. BottleneckSampling [38] as opposed to increasing scales introduces a bottleneck scale for better generation, while Decomposed

Flow Matching [13] predicts Laplacian residuals of clean images. UDPM [1] tries to add blurring and subsampling into the diffusion process, assumes isotropic posterior covariance to simplify their reverse diffusion derivation, which may not hold, given that the blurring kernels usually overlap in most implementations of resizing. We show through Scale Space Diffusion that end-to-end training of a single diffusion model capable of handling multiple resolutions, with a generalized mathematical formulation for resolution transitions, helps to achieve faster generation, while preserving high-quality.

#### 3. Scale Spaces vis-`a-vis Diffusion Timesteps

In this section, we outline the motivation behind our approach, which originates from a simple but compelling intuition. Consider the intermediate states of a diffusion model

- (Fig. 2(a), bottom) and the scales of a Gaussian pyramid
- (Fig. 2(b), bottom). If one squints and focuses on the third image from the left along the t-axis, the overall structure of the face begins to emerge, which is remarkably similar to the information present in the images corresponding to smaller spatial scales along the r-axis of the Gaussian pyramid. As we move rightward along either axis (i.e., decreasing t or increasing r), it becomes evident how finer details are added progressively.

This observation suggests a striking correspondence in the information hierarchy between diffusion timesteps and scale-space resolutions (or scales). Our goal is to quantify this correspondence. To do so, we first review the standard diffusion process, and then formalize our intuition by mathematically characterizing the amount of information present across diffusion states.

##### 3.1. Preliminary: Standard Diffusion Process

In standard denoising diffusion probabilistic models (DDPM) [16], the forward diffusion process is modeled as a Markov chain that progressively noises a signal by adding Gaussian noise. For x0 ∼ q(x0), where q(x0) is the data distribution, the process is defined as:

xt = √αtxt−1 + √1 − αtϵ, ϵ ∼ N(0,I) (1)

where {βt}Tt=1 is the variance schedule (with αt := 1 − βt). This expression, when applied iteratively over t, leads to an

alternative definition that expresses the noisy state as a linear combination of the signal x0 and the noise ϵ:

xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I) (2)

where α¯t := ti=0 αi. Diffusion models aim to reverse this process by approximating the posterior distribu-

tion q(xt−1|x0,xt) using a neural network (with parameters θ) that predicts the noise ϵ in Eq. 2. This model, ϵθ(xt,t), is trained using a simplified loss function Lsimple = Ex

0,t,ϵ ∥ϵθ(xt,t) − ϵ∥22 . The model can also be parameterized to predict x0 instead of ϵ.

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

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Figure 2. Information Analysis. (a) Amount of information present in a diffusion state as diffusion step t changes. (b) Amount of information present in images at various resolutions (scales).

##### 3.2.InformationDegradationinDiffusionandScale Spaces

Diffusion States. In this section, we formally model the information degradation over the diffusion process. Eq. 2 has two terms – a signal term and a noise term. One way to model the amount of information present in xt is to compute the percentage of pixels for which the noise term dominates the signal term, i.e., |

√1 − α¯t ϵ| > |

√α¯t x0|. In other words, we are looking for the probability that |ϵ| is greater than s(t)|x0|, where s(t) =

√√1−α¯αt¯t is the square root of the signal-to-noise coefficient ratio. We have P (|ϵ| > s(t)|x0|) = (1 − Φ(s(t)|x0|))+Φ(−s(t)|x0|) = 2Φ(−s(t)|x0|), where Φ is the CDF of the standard normal distribution, and the second equality follows from symmetry of the standard normal distribution. Hence, P (|ϵ| ≤ s(t)|x0|) = 1 − 2Φ(−s(t)|x0|). Now to obtain the expected fraction of signal-dominated pixels, a proxy for information, we average this probability over the data distribution q(x0). For simplicity, let us assume x0 ∼ U(−1,1). Then the variation of information over timestep t can be written as:

Info(t) = Ex

0∼U(−1,1)[1 − 2Φ(−s(t)|x0|)]

1

= 1 − 2

pU(−1,1)(x)Φ(−s(t)|x|)dx

−1

1

- 1

- 2

= 1 − 2

Φ(−s(t)|x|)dx

−1

1

1

= 1 −

Φ(−s(t)|x|)dx = 1 − 2

Φ(−s(t)x)dx,

−1

0

where we use the fact that the uniform distribution has density pU = 12 over [−1,1], and for the final equality we split the integral about x = 0. Using this simplification, Info(t) can be numerically computed as a function of t, as shown in Fig. 2(a).

Scale Spaces. Similar to the approximation of information across diffusion steps, here we want to approximate the information as a function of image resolution (i.e., scale). A simple way to model this is to assume:

Info ∝ Area.

Let us consider a normalized resolution r ∈ [0,1], where 0 represents no pixels and 1 represents the highest resolution. Under this assumption, the information can be written as:

###### Info(r) = r2.

This implies, for example, that if the spatial dimensions of an image are halved, then the information becomes one quarter, which may not be strictly true due to redundancy in pixel space. However, the monotonic trend should still hold. This trend is visualized in Fig. 2(b).

Notice how there is a similarity in the trends of information degradation as t increases versus as r decreases. This analysis quantifies our main intuition regarding the similarity in the information trends across diffusion steps and scale spaces. Given this insight, we aim to leverage this intuition to construct a framework that realizes scale spaces within the current formulation of diffusion models.

In our initial attempts to incorporate scale spaces into diffusion models, we tried to frame this problem as jumping across the same timesteps of independent diffusion processes at varying scales. However, this led to an accumulation of errors during the iterative inference procedure, resulting in suboptimal outputs. Methods such as Pyramidal Flow Matching [6, 19, 20] address this issue by adding decorrelation noise when transitioning across scales and then backtracking in time so that an appropriate noise level is selected. This strategy helps mitigate the error accumulation. Nonetheless, it does not actually resolve the underlying issue – the diffusion process itself is not mathematically modeled to handle scale changes. In this work, we aim to fill this gap.

#### 4. Scale Space Diffusion (SSD)

In this section, we introduce a new family of diffusion processes that use a generalized linear degradation operation for degrading the signal, in addition to the standard additive Gaussian noise. We then show how this formulation can be implemented in deep learning frameworks such as PyTorch [30] for any choice of a linear degradation that is available as a function call. In our case, we choose a downsizing operator as our linear degradation. Next, we present our training and sampling pipelines. Finally, we introduce our architecture that can handle scale-preservation and scale-changing transitions at multiple resolutions.

##### 4.1. Generalized Linear Diffusion Process

###### 4.1.1. Extension to Linear Degradation

We now replace the scalar coefficient of xt−1 in Eq. 1, i.e., √αt, with a more generic linear operator Mt. For example, blurring or downsampling can serve as such a linear operator. Let us assume a Gaussian distribution for this updated formulation for the transition distribution q(xt|xt−1) as xt = Mtxt−1 + ηt,ηt ∼ N(0,Σt|t−1). Here, we do not assume Σt|t−1 to be isotropic.

Now, repeatedly sampling the next state using the transition distribution, we want to derive an equation analogous to Eq. 2, which provides us xt given x0. It is clear that this will also be a Gaussian distribution q(xt|x0) = N(µt,Σt). The only constraint we want to enforce is isotropy, i.e., Σt = σt2I. For the coefficient of x0, instead of √α¯t = √αt√αt−1 ...√α1 in Eq. 2, we get M1:t = MtMt−1 ...M1, i.e., µt = M1:tx0. Hence, q(xt|x0) = N(M1:tx0,σt2I), which can be expressed as:

xt = M1:tx0 + σtϵ, ϵ ∼ N(0,I) (3)

Using Theorem 1, similar to blurring diffusion [18], the transition distribution q(xt|xt−1) is given by:

xt = Mtxt−1 + ηt, ηt ∼ N(0,Σt|t−1), where Σt|t−1 = Σt − MtΣt−1MtT.

(4)

For the isotropic marginals Σt = σt2I and Σt−1 = σt2−1I, we obtain Σt|t−1 = σt2I − σt2−1MtMtT. For positive semidefinite feasibility we require σt2I ⪰ σt2−1MtMtT, i.e., σt2 ≥ σt2−1λmax(MtMtT).

As shown in Theorem 2, the reverse diffusion step, i.e., the posterior distribution q(xt−1|xt,x0), conditioned additionally on x0, is also a normal distribution:

q(xt−1|xt,x0) = N(µt→t−1,Σµ

),

t→t−1

where Σt→t−1 = (Σ−t−11 + MtTΣ−t|t1−1Mt)−1, and

µt→t−1 = Σt→t−1(Σ−t−11µt−1 + MtTΣ−t|t1−1xt)

(5)

Using the Woodbury matrix identity and isotropic covariance assumption, this simplifies to (Theorem 3):

σt4−1 σt2

Σt→t−1 = σt2−1I −

MtTMt

σt2−1 σt2

MtT(xt − Mtµt−1)

µt→t−1 = µt−1 +

(6)

Please refer to Table 1 for the comparison of our Generalized Linear Diffusion Process framework against DDPM and Blurring Diffusion (BD).

DDPM as a special case of SSD. When Mt = √α¯tI and σt = √1 − α¯t, the forward, marginal, and posterior distributions of SSD collapse to those of the DDPM model.

###### 4.1.2. Implementation Details

Choice of Mt. We derived the above framework so that we can introduce scale spaces from Gaussian pyramids into the diffusion process. Although Mt may be any arbitrary linear operator, for our purposes we select it to be a resize operator, which effectively blurs and downsamples the image, and then multiplies it by at = √α¯t, as shown in Algo. 1. Note that this changes the dimensionality of the signal, in contrast

- Table 1. Comparison between the formulations of the forward, marginal, and posterior distributions of DDPM and Blurring Diffusion (BD) against our Scale Space Diffusion. For Blurring Diffusion, we use ‘a’ instead of α used in their paper, to not confuse it with the α in DDPM. Note that BD applies a change of variable ut = V Txt, where V T is the Discrete Cosine Transform, before performing diffusion, i.e., diffusion in frequency space. BD and DDPM have equivalent formulations when at = √α¯t and σt = √1 − α¯t. While the formulations share structural similarities, Scale Space Diffusion extends the framework to support general linear degradations (e.g., downscaling), which are not handled by DDPM or BD. We highlight analogous terms with consistent background colors for easier correspondences across different formulations. Legend: Forward mean, var Marginal mean, var Posterior mean, var Distributions DDPM [16] Blurring Diffusion [18] Scale Space Diffusion

ut = at|t−1 ut−1 + σt|t−1 ϵ, ϵ ∼ N(0, I) where at|t−1 =

xt = √αt xt−1 + √1 − αt ϵ, ϵ ∼ N(0, I)

xt = µt|t−1 + ηt|t−1, ηt|t−1 ∼ N(0, Σt|t−1) where µt|t−1 = Mtxt−1 = M1:t(M1:t−1)−1xt−1,

at at−1

,

Forward q(xt|xt−1)

Σt|t−1 = Σt − Mt Σt−1 MtT

σt|t−1 = σt2 − a2t|t−1 σt2−1

xt = √α¯t x0 + √1 − α¯t ϵ, ϵ ∼ N(0, I) ut = at u0 + σt ϵ, ϵ ∼ N(0, I) xt = µt + ηt, ηt ∼ N(0, Σt)

Marginal q(xt|x0)

where µt = M1:tx0, Σt = σt2I

xt−1 = µ˜t−1 + β˜t−1 ϵ, ϵ ∼ N(0, I) where β˜t−1 = 1−1−α¯αt¯−1

ut−1 = µt→t−1 + σt→t−1 ϵ, ϵ ∼ N(0, I)

xt−1 = µt→t−1 + ηt→t−1, ηt→t−1 ∼ N(0, Σt→t−1) where Σt→t−1 = (Σ−t−11 + MtT Σ−t|t1−1 Mt)−1

−1

βt

a2t|t−1 σt2|t−1

t

where σt2→t−1 = 1

+

,

−1

αt 1 − αt

σt2−1

4 t−1

= 1 1 − α¯t−1

= σt2−1I − σ

, µ˜t−1 =

+

σt2 MtTMt, µt→t−1 = Σt→t−1(MtT Σ−t|t1−1xt + Σ−t−11 µt−1)

Posterior q(xt−1|xt, x0)

at|t−1 σt2|t−1

√αt(1−α¯t−1)

at−1 σt2−1

1−α¯t xt + α¯1t−−α1¯βt

µt→t−1 = σt2→t−1

x0

xt +

x0

t

√αt 1 − αt

2 t−1

α¯t−1 1 − α¯t−1

= µt−1 + σ

σt2 MtT(xt − Mtµt−1)

= β˜t−1

xt +

x0

with previous diffusion formulations. However, since we make no assumptions about dimensionality, our framework remains valid regardless. Furthermore, with this choice of Mt, we also define a resolution schedule r(t) that maps diffusion timestep (t) to the corresponding resolution, such that the resolution monotonically decreases as t increases (Fig. 5). Refer to the supplementary Sec. 8.2.1 for another degradation example.

Calculating the Transpose. Since operators like image resizing are implicit, we may not have the matrix form available, making it non-trivial to apply the transpose MtT. To address this, we use a vector-Jacobian product of the function call Mt(·), i.e., MTv = torch.autograd.grad(Mt(x), x,

grad outputs=v)[0], which, for linear operators, does not depend on x, as shown in Algo. 2. This computes the derivative of the inner product ⟨v,Mtx⟩ with respect to x, i.e., ∇x⟨v,Mtx⟩ = MtTv.

Sampling from a Non-Isotropic Gaussian Distribution. A neat trick to sample from a non-isotropic Gaussian distribution with covariance matrix Σ is to first sample a standard Gaussian noise ϵ ∼ N(0,I), and then multiply with the square root of the covariance matrix, so that Σ12 ϵ ∼ N(0,Σ12 I(Σ12 )T) = N(0,Σ). In our case, we need to sample noise from Σt→t−1 from Eq. 6, which depends on implicit operators Mt(·) and MtT(·). Thus, we need a way to apply Σt→t−1(·) implicitly to a standard Gaussian noise ϵ. For this purpose, we use the Lanczos algorithm [11, 23], which numerically computes A(x) given an implicit symmetric linear operator A(·) and vector x. When the Lanczos algorithm is applied with a square root spectral function over the eigenvalues, we can obtain A12 x. In our case, this gives ηt→t−1 = Σ

- 1

- 2

t→t−1ϵ ∼ N(0,Σt→t−1) as shown in Algo. 3.

- Algorithm 1 Implicit Linear Operator

# M resizes and attenuates signal x def M(x, a_t, a_t_minus1, size_out):

return (a_t / a_t_minus1) * F.interpolate( x, size=size_out, mode="bilinear", align_corners=False, antialias=True)

- Algorithm 2 Implicit Linear Operator’s Transpose

# M_T applies the transpose of M on v def M_T(M, v, a_t, a_t_minus1, M_input_shape):

size_out = v.shape()[-2:] with torch.enable_grad():

x = torch.zeros(M_input_shape, requires_grad=True) out = M(x, a_t, a_t_minus1, size_out) # calculate MˆTv = d<v,Mx>/dx (g,) = torch.autograd.grad(out, x, grad_outputs=v,

retain_graph=False) return g

- Algorithm 3 Sampling Non-Isotropic Gaussian Noise

# Sample noise from posterior covariance Sigma_{t-->t-1} def sample_non_isotropic_noise(M, M_T, sigma_t,

sigma_tminus1, x):

rho = (sigma_tminus1 ** 2) / (sigma_t ** 2) # Define matvec operator A

A = lambda v: v - rho * M_T(M(v)) # Lanczos approximation of Aˆ{1/2}v y = lanczos(A, x, f=lambda l: l.sqrt()) return sigma_tminus1 * y

##### 4.2. Training and Sampling

To reverse the diffusion process using Eq. 6, our model must predict µt−1 = M1:t−1x0, which, with our choice of Mt, reduces to a scaled version of an image x0 at resolution r(t − 1). To train such a model, using our Generalized Linear Diffusion Process, we need to first adapt Lsimple. When predicting x0, the loss becomes Lsimple = Ex

0,t,ϵ[s2(t)∥xθ0(xt,t) − x0∥22], where s2(t) is the signal to noise ratio, as shown in [34]. In Min-SNR-γ [14] instead

###### Algorithm 4 Train

def train_iter(x, t, a_t_minus1, model, opt): opt.zero_grad() t_minus1 = (t-1).clamp(min=0) # clean image at res r(t-1) = M_{1:t-1}(x) / a_{t-1} x_start_t_minus1 = cummulative_M[t_minus1](x)/a_t_minus1 # Using Eq.3 x_t = diffuse(x, t) pred_x_start_t_minus1 = model(x_t, t) # Using Eq.7 loss = min_snr_5(t) * ((pred_x_start_t_minus1 -

x_start_t_minus1) ** 2) loss.backward() opt.step() return loss

###### Algorithm 5 Sampling

# get x_{t-1} given x_t def sample_iter(x_t, t, model):

pred_x_start_t_minus1 = model(x_t, t) mu_t_minus1 = a_t_minus1 * pred_x_start_t_minus1 # Using Eq.6

posterior_noise = calculate_posterior_noise(t) posterior_mean = calculate_posterior_mean(x_t,

mu_t_minus1, t) x_t_minus1 = posterior_mean + posterior_noise return x_t_minus1

of the s2(t) weighting, they use min(s2(t),γ), with γ = 5, which improves the performance of x0 parameterization significantly. Following this, our loss function evaluates to:

0,t,ϵ min(s2(t),γ) xr0(,θt−1)(xt,t) −

L = Ex

1 at−1

M1:t−1x0

(7)

2 2

where we predict an unscaled µt−1 using a neural network xr0(,θt−1) (Algo. 4). Note that the input resolution r(t) of xt may be smaller than the resolution of the output at r(t−1) as seen in Fig. 3 (left). In standard diffusion training, timesteps are simply sampled uniformly for each batch. However, this is non-trivial in our setting because the (r(t),r(t − 1)) pairs may not match. To solve this, we first uniformly sample a single t, and if r(t) = r(t − 1), then uniformly sample the batch size number of ti’s that have r(ti) = r(t). Otherwise, if r(t) ̸= r(t − 1), then we fill the entire batch with the same t, so there is no size mismatch. Since not all t’s change resolution, many of the Mt’s can be replaced by scalar multiplication with (at/at−1) = √αt.

For sampling (Algo. 5), we start from a random Gaus-

sian noise at the lowest resolution r(T). Our model xr0(,θt−1) predicts a clean image at the next resolution r(t − 1), using

which we can calculate µt−1 and denoise using the posterior distribution (Eq. 6). This also involves sampling from Σt→t−1, which may not be isotropic, and hence we use Algo. 3 to sample noise from this distribution. Eq. 6 is equivalent to DDPM sampling when r(t) = r(t − 1), so the non-isotropic noise sampling can be replaced with normal torch.randn() calls for resolution-preserving steps.

##### 4.3. Architecture

We adapt the UNet architecture from Ablated Diffusion Model (ADM) [10] to design our proposed model FlexibleUNet (Flexi-UNet), which supports multi-resolution inputs and outputs to fully realize the scale-space formulation. Because Scale Space Diffusion embeds a resizing operator in the forward diffusion process, the spatial resolution of xt varies across timesteps, and the reverse model must therefore operate on variable-sized noisy states and sometimes predict a higher-resolution output at the next scale (Fig. 3).

A standard diffusion model, such as ADM [10], is trained to operate at a single fixed resolution throughout all timesteps, and even multiresolution UNet variants only process multiple scales within a fixed-resolution diffusion process. In contrast, SSD requires an architecture that natively handles different input resolutions across timesteps. To address this, we explore two architectural designs.

Full UNet (Single Path). The base UNet architecture inherently supports variable-size inputs and outputs, and in principle can operate on any spatial resolution R×R as long as the kernel sizes, strides, padding, and pooling operations produce valid feature maps at every layer. However, this design has two key limitations for Scale Space Diffusion. First, it requires the input and output resolutions to be equal. In our setting, certain timesteps involve a resolution transition, which would require the model to output at a higher resolution. To handle this, the input must be manually upsampled before entering the UNet whenever such a transition occurs.

Second, the depth of the UNet determines how many distinct spatial scales it can represent. For a UNet with L downsampling blocks, the smallest internal resolution is

, which fixes the total number of scales to L. This number is typically small and does not grow with the input resolution. For example, the ADM architecture uses 4 feature map resolutions for 64×64, 5 for 128×128, and 6 for 256×256, meaning that across all these models the number of downsampling stages remains fixed at 4. Thus, even at higher resolutions, the network cannot represent more than a handful of scales, limiting the usefulness of a scale-space formulation where many more levels naturally exist.

R 2L−1

Flexi-UNet. These limitations motivate our proposed architecture, Flexi-UNet, where different subsets of UNet layers are dynamically activated based on the input resolution. High-resolution inputs traverse the full UNet, while lowerresolution inputs are routed only through the deeper layers, effectively bypassing the early and late blocks. Since each block expects a specific channel dimensionality, we insert 1×1 conv layers to map the input features to the appropriate channel size while preserving spatial resolution.

For denoising steps that do not involve a resolution change, the active pathway through the UNet remains symmetric, using the same number of downsampling and upsampling blocks. When a resolution increase is required, the

Training Inference

[Figure 41]

Reverse Sampling:

Forward Sampling:

(See Eq. (6))

(See Eq. (3))

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

Architecture Details

Input

Skip connections

|0|0|0|0|
|---|---|---|---|

Predicted

Active Layers Inactive Layers

Input, Noisy Image, at Diffusion step and scale Predicted, Clean Image at scale from

Resolution change Diffusion step Same resolution Diffusion steps

Figure 3. Overview. Left: During training xt’s at resolution r(t) are sampled using Eq. 3, and our model is trained to predict clean image xr0,θ(t−1) using the loss as in Eq. 7. Our Flexi-UNet is able to process both resolution-preserving and resolution-changing steps at multiple resolution using only parts of the network. Right-top: During sampling, Eq. 6 is used to progressively denoise and upsample to generate images. Right-bottom: Our Flexi-UNet has additional 1×1 Conv layers to take inputs at any UNet encoder block and get outputs form any decoder blocks. For resolution changing, the skip connections are fed with zero-filled tensors.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

𝑟 = 64

pathway becomes asymmetric: the model uses one additional upsampling block compared to the number of downsampling blocks encountered. In these cases, the skip connections that would normally come from the bypassed encoder blocks are replaced with zero tensors (Fig. 3). This design allows the model to share parameters across resolutions while supporting valid diffusion dynamics during resolution transitions.

𝑟 = 32 𝑟 = 16

[Figure 68]

𝑟 = 8

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

𝒓𝑟 = 𝟏𝟐𝟖128

𝑟 = 64 𝑟 = 32

[Figure 75]

[Figure 76]

𝑟 = 16 𝑟 = 8

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

𝑟 = 256

𝒓 = 𝟐𝟓𝟔

#### 5. Experiments

[Figure 83]

[Figure 84]

[Figure 85]

Datasets. We perform experiments and analyze the performance of Scale Space Diffusion on the CelebA dataset[25] and the ImageNet dataset [8]. The CelebA dataset consists of around 200K training images, while the ImageNet dataset contains around 1.3 million images from 1000 different classes. We use JPEG images for these datasets. We conduct experiments at 64×64 resolution for both CelebA and ImageNet. We additionally show experiments on 128×128 and 256×256 for CelebA dataset. The CelebA experiments helps us understand our method’s scalability with increasing resolutions, while ImageNet helps in evaluating the model’s ability to learn complex and diverse distributions.

- Figure 4. Visual samples. Top: ImageNet-64 unconditional generation. For the top-most sample we also show model prediction at various scales (8, 16, 32, 64) during SSD. Bottom: CelebA-256 unconditional generation. For the top-most sample we also show model predictions at various scales (8, 16, 32, 64, 128, 256).

[Figure 86]

Schedule FID Time (hrs)

equal 9.64 12.88 ConvexDecay 2 11.03 11.71 ConvexDecay 0.5 4.87 13.81 SigmoidLikeDecay 3 7.08 13.06 TanhLikeDecay 3 8.09 12.50

- Figure 5. Resolution Schedules. Mapping diffusion timesteps t to resolution r across 4 scales. Both discrete and continuous variants are shown. The right shows FIDs at 500k iterations (batch size 8). comparison to Scale Space Diffusion. We implement Blurring Diffusion using the pseudo-code provided in their paper.

Unconditional Image Generation. We analyze and evaluate Scale Space Diffusion on unconditional image generation, as it allows us to clearly study how scale-space theory integrates with the diffusion process.

Implementation Details. We use the ADM [10] repository

- as our base codebase and build our baselines (DDPM [16], Blurring Diffusion [18]), as well as Scale Space Diffusion on top of it. For DDPM, we consider two standard parametriza-

For the diffusion process, we follow the linear noise schedule proposed in DDPM [16] and use the standard setting of 1000 timesteps. For training, we use AdamW [22, 26] optimizer with a fixed learning rate. We conducted all our

tions as our baselines, the ϵ-prediction, and the x0-prediction formulation. We train the baseline model with Min-SNR-γ weighting for the x0-parametrization to ensure an accurate

- Table 2. Main Results. Unconditional image generation results on CelebA dataset over multiple resolutions. Training time is specified in hours. Average GFlops per iteration. Effective batch size is 128 for resolutions 64 and 128, 64 for resolution 256. Here BD refers to Blurring Diffusion [18] and all SSD models use our FlexiUNet architecture.

Method

CelebA-64 (1M iters) CelebA-128 (300K iters) CelebA-256 (300K iters) FID Time GFlops FID Time GFlops FID Time GFlops DDPM-ϵ 2.22 70.30 60.05 4.16 50.50 132.30 5.52 87.31 497.03

DDPM-x0 2.98 70.71 – 3.50 50.33 – 5.47 87.33 – BD 2.06 71.79 – 3.67 – – 4.76 88.08 –

- SSD (2L) 2.14 62.63 50.61 – – – – – –
- SSD (3L) 3.61 56.13 44.27 6.53 31.71 87.38 7.79 59.00 317.36
- SSD (4L) 4.28 52.38 38.48 – – – 10.52 51.70 272.98
- SSD (5L) – – – 10.47 25.41 66.72 – – 237.70
- SSD (6L) – – – – – – 13.50 42.88 209.69

- Table 3. ImageNet64 Results. Unconditional image generation results on ImageNet-64 dataset.

Figure 6. Temporal Scaling. Training time of proposed SSD with our FlexiUNet across multiple resolutions.

Training time across scales and resolutions

DDPM

Ours (multi-scale)

80

Trainingtime(hrs)

Method FID DDPM-ϵ 12.82 DDPM-x0 13.07 Blurring Diffusion 15.34 SSD (2L) 13.08 SSD (4L) 17.89

60

40

20

64 128 256 Resolution

experiments on NVIDIA H100 and NVIDIA RTX A4000 GPUs. We maintained consistent combinations of learning rate and batch-size across dataset and resolutions. For 64×64 and 128×128, we used an effective batch size of 128, and trained the models either on a single H100, or on 4 A4000 GPUs, with a per-GPU batch-size of 32. For 256×256, we used an effective batch size of 64 due to memory constraints and trained them on 2 H100s with a per-GPU batch size of 32. Our learning rate was set to 1×10−4 for the 64×64 and 128×128 models, and 5×10−5 for 256×256 model, following linear learning rate scaling.

Evaluation. We evaluate our models using the exponential moving average (EMA) weights with a decay rate of 0.9999. We assess the quality of generated images by computing FID [15] scores on 50k samples w.r.t. the training set. We further compare Scale Space Diffusion with the DDPM baseline model in terms of training time, and FLOPs (Floating Point Operations) per forward pass. In addition to FLOPs, we report the sampling latency as the total time to generate a single image. All speed and compute metrics are measured on a single NVIDIA GH200 node.

Main results. Our results are presented in Table 2 and 3. We train the baseline models and Scale Space Diffusion model for 1 million iterations for CelebA-64 and 300k iterations for CelebA-128 and CelebA-256. We report the total training time, average GFLOPs per iteration, and the FID value. We notice that increasing the number of levels significantly decreases training time and GFLOPs. SSD (6L) at 256 resolution takes less than half the time as the baseline DDPM. Table 3 shows that SSD, trained for 1 million iterations,

Table 4. Architecture Ablation. FID (at 500K iterations on CelebA-64) and Inference time (in secs/generation, 1000 steps, batch size=1, on 1× A4000) of network architecture variants at 2 and 4 levels.

FID Inference time res. 64 res.64 res. 256

Method

Full Unet, 2L 2.33 16.19 43.07 Flexi-UNet, 2L 2.26 15.38 38.99

Full Unet, 4L 4.90 16.28 34.74 Flexi-UNet, 4L 4.87 13.43 31.08

achieves comparable performance to baselines even on the harder ImageNet-64 benchmark. Figure 6 shows that training time for SSD scales well with increasing resolution. Please refer to the supplementary Sec. 8.4 for more comparisons.

Qualitative results. We present qualitative results of our method on ImageNet-64 with SSD (4L) and CelebA-256 with SSD (6L) in Fig. 4. We also show multiple intermediate predictions of the model in SSD.

Individual effectiveness of our mathematical formulation vs architecture. In the supplementary Sec. 8.2, we first show that our Generalized Linear Diffusion Process works, albeit suboptimally, even without Flexi-UNet, and also with an alternate degradation. Approximating anisotropic gaussian noise with isotropic leads to saturation artifacts, showing the need for our anisotropic sampling. Secondly, we show that Flexi-UNet is effective on its own for other formulations of iterative multi-resolution pixel-space generation, albeit suboptimal. We do this by applying it to approximate multiresolution diffusion as in PyramidalFlow [6, 19, 20].

Resolution Schedule. r(t) specifies the spatial resolution as a function of diffusion timestep. We present 5 different resolution schedules in Figure 5 (left) at 4 levels (64,32,16,8) (refer to the supplementary Sec. 8.3 for details) and note their effects for a CelebA-64 model in Fig. 5 (right). We observe that schedules that spend the least number of timesteps at the higher resolutions train the fastest, but also yield the worst FID (i.e. ConvexDecay 2). In contrast, the model trained with ConvexDecay 0.5, which spends the most steps at the highest resolution, achieves the best FID, but requires the longest training time. We use this for all our experiments.

Full UNet vs Flexi-UNet. In Table 4, we observe that FlexiUNet has slightly better FID for both 2L and 4L, while being faster than the Full UNet. Hence, we use Flexi-UNet.

Sampling. Use of Lanczos has negligible overhead. Further, SSD does not suffer from performance drop, like DDPM, on sampling steps reduction. (Refer supplementary Sec. 8.5.)

Conclusion. We showed that diffusion models and scale spaces share an information hierarchy, and we quantified this connection mathematically. Observing that highly noised diffusion states contain only low-resolution information, we introduced a generalized family of diffusion models that embeds scale-space structure into the forward process, yielding Scale Space Diffusion. To realize this in practice, we proposed the Flexi-UNet architecture and demonstrated its effectiveness on unconditional image generation.

Acknowledgment. This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA R&D Contract No. 140D0423C0076. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon.

### Scale Space Diffusion Supplementary Material

Figure 7. Animation of the predicted clean image xr0,θ(t−1) over the generation process for gradual downsizing degradation operator in

SSD framework. (Best viewed in Adobe Reader).

Figure 8. Animation of the noisy intermediate state xt over the generation process for the gradual downsizing degradation operator in SSD framework. (Best viewed in Adobe Reader).

#### Contents

- 6. Clarifications 1
- 7. Future Works 1
- 8. Additional Material 2

- 8.1. Hyperparameters . . . . . . . . . . . . . . . 2
- 8.2. Parts of our Approach . . . . . . . . . . . . 2
- 8.3. Resolution Schedules . . . . . . . . . . . . . 4
- 8.4. More Comparisons . . . . . . . . . . . . . . 5
- 8.5. Quantitative Results . . . . . . . . . . . . . 5
- 8.6. Qualitative Results . . . . . . . . . . . . . . 6

- 9. Mathematical Derivations 20

- 9.1. Forward Transition . . . . . . . . . . . . . . 20
- 9.2. Posterior Distribution . . . . . . . . . . . . . 20
- 9.3. Posterior Under Isotropic Marginals . . . . . 21

#### 6. Clarifications

We add some clarifications for our main paper here.

1. All variable names used in Algo. 1, 2, and 3 have their usual meanings. For example, a t, a t minus1 denote the signal coefficients at, at−1 respectively, while sigma t, sigma tminus1 denote the noise schedule σt, σt−1 respectively. size out denotes the (height, width) of the output of M operator.

#### 7. Future Works

The focus of this work has been to analyze the connection between diffusion models and scale space theory, while proposing to merge them using Scale Space Diffusion with FlexiUNet. We do not use any advanced techniques to tune our framework or architectures for the most optimal performance. Instead, we use the standard hyperparameters from the base

codebase to keep the choices simple and the number of experiments under check given the expense of each training. The use of advanced techniques is out of scope for this work given the conference length manuscript.

However, there are multiple future exploration directions which have high potential for improvement in performance. For example, adapting newer diffusion samplers instead of using DDPM-style samplers can improve both performance and inference speeds. Similarly, progressive curriculum learning for different layers or resolutions, as done by works with multi-resolution trainings [12, 21], should also yield improvement in training optimization.

Why not use a Transformer-based architecture? The two most popularly used architectures in diffusion are – convolutional UNet [33] based ADM [10], and vision transformer (ViT) based DiT [31]. Another popular architecture is UViT [4], that combines the skip connections from UNet with a ViT architecture. One thing to note is that, DiT was designed for latent spaces and hence did not take into consideration the blowing up of the quadratic complexity of the attention mechanism when applied in the pixel space [6]. UViT acknowledges this issue, and explicitly works in a latent space for higher resolutions. Newer works like HDiT [7] try to mitigate this issue using neighborhood attention instead of global attention is all layers. But such non-trivial design decisions in the architecture can develop into confounding factors. Since our goal is to understand how scale-spaces can be integrated into diffusion models, for simplicity we stick to the standard ADM base architecture, a widely used pixel and latent diffusion architecture [32]. Nonetheless, for future work, a similar integration of scale-space theory should also be explored with transformer based architectures.

Hyperparameter CelebA-64 CelebA-128 CelebA-256 ImageNet-64 Noise Schedule Linear Linear Linear Linear Denoising Steps 1000 1000 1000 1000 Optimizer AdamW AdamW AdamW AdamW Batch Size 128 128 64 128 Learning Rate 0.0001 0.0001 0.00005 0.0001 Number of Iterations 1 million 300k 300k 1 million

- Table 5. Hyperparameters for all datasets.

Implementation Choice DDPM-ϵ DDPM-x0 Blurring Diffusion SSD Reverse Process Variance fixed-large fixed-large fixed-small fixed-small Loss Lsimple Lsimple + Min-SNR-5 Lsimple Lsimple + Min-SNR-5

- Table 6. Additional implementation details.
- Table 7. Inference time. By default we use DDPM sampling, but we also show 25† steps DDIM [35] speeds.

Method #Steps Speedup (Inference Time) FID

DDPM-x0

1000 1.00 × 2.98 250 4.18 × 14.00 25† 38.87 × 4.70

DDPM-ϵ

1000 1.05 × 2.22 250 4.18 × 11.02 25† 38.06 × 3.76

SSD(Flexi-UNet, 2L)

1000 1.18 × 2.14

- 250 4.80 × 2.87

SSD(Flexi-UNet, 4L)

1000 1.58 × 4.28

- 250 5.91 × 4.90

- Table 8. Inference time (in secs) per gen at 64 res (1000 steps, bs=1, 1 A4000): Lanczos sampling vs. torch.randn call.

Method SSD (2L) SSD (4L)

w/ Lanczos 15.38 13.43 w/o Lanczos 15.35 13.40

#### 8. Additional Material

##### 8.1. Hyperparameters

The set of hyperparameters that we use for each dataset is summarized in Table 5. We also note additional experimental details in Table 6.

##### 8.2. Parts of our Approach

Our approach consists of two parts. The first part is the Scale Space Diffusion mathematical formulation and the second part is the Flexi-UNet architecture. In the main paper, we have presented the combination of both parts as our complete approach. But here we also want to show that each part is effective on its own. So, in Section 8.2.1, we explore whether the mathematics behind SSD can be applied without a modified architecture, while in Section 8.2.2, we check if Flexi-UNet can be used without our mathematical framework, summarized in Table 9.

###### 8.2.1. Validity of SSD

One way to verify whether SSD framework works without using a modified architecture is to assume that the actual states of the diffusion model are at a certain resolution, but when passing through the model, we resize them to the model

Table 9. Parts of our approach, and validity of each part.

SSD Flexi-UNet

- Section 8.2.1 ✓ ✗
- Section 8.2.2 ✗ ✓ Main paper ✓ ✓

Table 10. Results of only SSD (w/o Flexi-UNet) on CelebA-32. (Here we resize the inputs to the model input resolution.)

Method FID DDPM-ϵ 2.85 SSD (w/o Flexi-UNet, 5L) 5.55 SSD (w/o Flexi-UNet, gradual downsizing) 4.10

input size. Similarly, the model outputs are resized to the required output resolution before applying losses. We test this with CelebA-32 dataset just to check the correctness of SSD. For this, we use a DDPM reimplementation (not ours) optimized for resolution 32 images [39], since ADM’s codebase does not support that resolution, and a smaller resolution is faster to verify on. We train these models for 300 epochs and use 5 steps of resolutions (2, 4, 8, 16, 32). We note their FIDs in Table 10.

Alternative Degradation. All the degradations used in this work till now have been 2× downsampling. However, given the general nature of the theory, it is not limited to just this choice. Here we test using a gradual downsizing instead of 2× downsizing steps. In this degradation, whenever the resolution changes, it does so by only 1 pixel at a time. We try going from 2 → 32. We report its FID in Table 10. We show some static visual results in Fig. 9. We show some interesting animated visualizations (view in Adobe Reader) in Fig. 7, and Fig. 8.

Effect of Isotropic Approximation. Another thing we wanted to test was whether we could approximate the nonisotropic Gaussian noise sampling (Algo. 3) with isotropic Gaussian noise. For testing purposes, during the generation procedure (of the gradual downsizing degradation case), in the resolution changing steps, we first use Algo. 3 to sample non-isotropic noise, and then find the mean and variance over the height and width dimensions of this noise tensor. Instead of using the sampled non-isotropic noise for the stochasticity in Eq. 6, we instead use an isotropic noise sampled using torch.randn() with the calculated mean and variance. As seen in Fig. 10, this leads to the colors becoming flat and saturated, despite having facial structures. This shows that the assumption of isotropic covariance for the reverse process may not actually be valid, as assumed in [1]. And we need to sample from non-isotropic Gaussians depending upon the linear operator.

[Figure 87]

[Figure 88]

[Figure 89]

(a) Generated Samples (b) Predicted clean images xr0,θ(t−1) (c) Noisy states xt

Figure 9. Visual results of SSD with gradual downsizing degradation (1 pixel downsizing instead of 2× downsizing)

[Figure 90]

Figure 10. Effect of using isotropic noise instead of non-isotropic noise in the reverse diffusion process of SSD.

###### 8.2.2. Effectiveness of Flexi-UNet

In this section, we demonstrate that Flexi-UNet can naturally accommodate different formulations of the diffusion process to support multi-resolution inputs and outputs. To do so, we build upon previous works that introduce corrective noise when an upsampling operation is performed in the diffusion process [6, 19, 20]. We implement these ideas within Flexi-UNet to both validate the flexibility of our architecture and quantify the computational benefits obtained from operating across resolutions. A key challenge addressed in these works is the distribution mismatch that arises when a noisy latent is upsampled. Prior works [19] show that applying a 2× nearest-neighbor upsampling step produces a block-structured covariance that deviates from the forward diffusion trajectory. Their solution injects structured noise and identifies an adjusted timestep that realigns

the upsampled latent with the original process. While this motivates our analysis, our setting is different from this in two ways: a) we operate entirely in pixel-space rather than latent space, and b) we consider multiple (more than one) upsampling stages throughout the denoising process. With these conditions in mind, our setup is as follows:

Let xrt be a valid DDPM forward state at a timestep t for resolution r:

√α¯t xr0, (1 − α¯t)I , Let xRt = Upsample(xrt),

xrt ∼ N

√α¯t Uxr0, (1 − α¯t)UU⊤

xRt ∼ N

where U is the Upsampling matrix.

Let xRs be a valid DDPM forward state at some other timestep s for resolution R:

√α¯s xR0 , (1 − α¯s)I ,

xRs ∼ N

The upsampled state xRt has covariance proportional to UU⊤, which differs from the isotropic Gaussian noise assumed by the DDPM forward process at resolution R. To correct this mismatch, we add corrective noise and roll back to a previous timestep. Let the corrected sample be

x˜Rt = axRt + bz, z ∼ N 0, I − cUU⊤ . Then the distribution of x˜Rt is

x˜Rt ∼ N a√α¯t Uxr0, a2(1 − α¯t)UU⊤ + b2 I − cUU⊤ .

We make an approximation to match the mean and covariance of x˜Rt to xRs

- a2α¯t = α¯s
- b2 = 1 − α¯s

a2(1 − α¯t) = b2c

(8)

Table 11. Results of Flexi-UNet (w/o SSD) on CelebA-64. Computed at 500k iterations. Inference time is computed as the average time (in minutes) to generate a batch of samples (256 samples).

Method FID Inference Time

Flexi-UNet (w/o SSD, Equal, 2L) 2.44 15.52 Flexi-UNet (w/o SSD, Equal, 4L) 5.79 13.32 Flexi-UNet (w/ SSD, ConvexDecay0.5, 2L) 2.26 14.98 Flexi-UNet (w/ SSD, ConvexDecay0.5, 4L) 4.87 11.20

Solving the three equations mentioned in Equation 8, gives us

a2(1 − α¯t) b2

- α¯s(1 − α¯t)

- α¯t(1 − α¯s)

(9)

=

c =

We first obtain the value of α¯s that satisfies Equation 9 for a given choice of c, and then obtain the corresponding timestep s. We sweep through values of c in range 0 ≤ c ≤ 0.25 (as mentioned in [19]) to produce different values of s. We compute all such candidate values of s and pick the best s empirically. For each value of c, we generate the corrected samples x˜Rt and the corresponding DDPM forward samples xRs using 2048 training images. We then compute the Jensen–Shannon divergence between these distributions to obtain the final backtracking index s as the one that produces the minimum JS divergence.

This experiment serves as our validation of our proposed method Flexi-UNet. During training, we follow a specific resolution schedule, so that for each timestep t, the model

receives a state xrt(t). To support distribution correction, we additionally include timestep s, x˜Rt to the training samples. During inference, the denoising process follows the standard reverse diffusion trajectory, with the following change: whenever the process reaches a timestep that has an upsampling step, the model rolls back to a slightly earlier timestep and continues denoising from that point at the higher resolution. This experiment illustrates the computational advantages of operating at multiple resolutions, using an architecture like Flexi-UNet, as a lot of the early denoising occurs

- at lower spatial resolutions. However, this setup requires rollback around each upsampling point, creating overlapping steps in the reverse process. While this model provides computational savings, there is an additional overhead of denoising for additional timesteps.

In Table 11, we show the FID values obtained for this experiment after training the model for 500k iterations. We compare the performance of Flexi-UNet trained with SSD to Flexi-UNet trained without SSD. We observe that FlexiUNet with SSD has better FID values, while also being faster at inference.

##### 8.3. Resolution Schedules

Here, we will define the functions that we used for the resolution schedules. We define what the resolution of the image should be given the diffusion timestep t, using a func-

tion r(t). As shown in Fig. 5, we use a discrete version of the resolution schedule, but it is based on a continuous function. Suppose for the discrete version we use a list of resolutions [rmin,2rmin,...,2n−2rmin,2n−1rmin] where rmin is the smallest resolution and n is the number of resolutions. For the continuous version, let’s first define normalized time τ = t/(T − 1), where T denotes the number of diffusion states. Then the normalized time to resolution schedule is defined as:

rcont(τ) = rmin · 2(n−1)f(τ) where f(τ) is the exponential schedule function that works as the multiplier to the exponent of 2. For example, when f(τ) = 0, then rcont(τ) = rmin, while when f(τ) = 1, then rcont(τ) = rmax = 2n−1rmin.

For the discrete version, we want to similarly sample from R = [rmin,2rmin,...,2n−2rmin,2n−1rmin = rmax], using the same schedule but over these discrete values. So, here we instead index the schedule function i(τ) that gives the index to select from R given τ.

###### r(τ) = R[i(τ)]

Similar to f, when i(τ) = 0, we have r(τ) = rmin, and when i(τ) = 1, r(τ) = rmax. Now we can introduce our schedules.

- 8.3.1. Equal This is the easiest linear schedule.

- • Continuous: f(τ) = 1 − τ
- • Discrete: i(τ) = n − 1 − ⌊nτ⌋

- 8.3.2. ConvexDecay γ

With a γ > 0 parameter, this function can simulate a convex or concave function depending on this parameter.

- • Continuous: f(τ) = 1 − (1 − τ)γ
- • Discrete: i(τ) = n − 1 − ⌊nf(τ)⌋ For γ > 1, it shows slow decay first, then faster, while for γ < 1, fast decay first, then slower.

- 8.3.3. TanhLikeDecay γ

Here we wanted a function that looks like tanh(·) function, which is steep at the highest and the lowest timesteps but is flat in the middle. This essentially spends more time in the middle resolutions. We approximate this using a polynomial.

First, we define a polynomial over a variable u ∈ [−0.5,0.5] as follows:

x(u,γ) = sign(u)|u|γ + 0.5 p(x) = −2x3 + 3x2 − 0.5

The polynomial p(x) is monotonically increasing in the range of [−0.5,0.5] for x ∈ [0,1], while x(u) is a function that looks like the tanh(·) function but is centered around 0.5. Essentially, p(x(u)) looks like the tanh shape and is

centered around the origin, but has varying range dependent on γ. We want this function to be equal to 1 at u = 0.5 and -1 at u = −0.5. To achieve that, we normalize this function:

p(x(u,γ)) p(x(0.5,γ))

pˆ(u,γ) =

Finally, to shift this function from [−0.5,0.5] → [−1,1] to [0,1] → [0,1], we apply the following transformation:

tanh like(u,γ) = 0.5 · pˆ(x(u − 0.5,γ)) + 0.5

[Figure 91]

Figure 11. Visualization of tanh like(·) for different γ’s.

Now, based on this definition, we can define the schedule.

- • Continuous: f(τ) = 1 − tanh like(1 − τ,γ)

- • Discrete: i(τ) = ⌊nf(τ)⌋

###### 8.3.4. SigmoidLikeDecay γ

Here we want a simoid-like curve, i.e., steep in the middle while flatter at the beginning and the end. We can such a curve by inverting pˆ(·). Following similar stretching and normalization, we can define another function that goes from [−0.5,0.5] → [−1,1] as:

(0.5 · pˆ)−1(u,γ) (0.5 · pˆ)−1(−0.5,γ)

hˆ(u) =

. Using the same shifting to transform [−0.5,0.5] → [−1,1] to [0,1] → [0,1], we have:

sigmoid like(u,γ) = 0.5 · hˆ(x(u − 0.5,γ)) + 0.5 Finally, we define the schedule.

- • Continuous: f(τ) = 1 − sigmoid like(1 − τ,γ)

- • Discrete: i(τ) = ⌊nf(τ)⌋

##### 8.4. More Comparisons

Upsampling Diffusion Probabilistic Models (UDPM) [1]. The mathematical formulation and implementation of

Table 12. FID comparison of SSD and super-resolved DDPM (64 res.) using an OpenImages pretrained 4× LDM.

Method FID SSD (3L, res. 256) 7.79 low-res diffusion (res. 64) + super-res (4×) 7.91

Table 13. Inference time per batch: SSD and LDMs (1000 steps, bs=32, A4000).

Method Inference time (secs)

SSD (6L, res. 256) 495 LDM (res. 256) 515

Table 14. Comparison with UDPM at 64 resolution: FID, training (1 H100), and inference speed (1 A4000, bs=256)

Method Inference Inference Training Time FID steps Time / batch (250K iters) (250K iters) (in secs) (in hours)

DDPM-ϵ 1000 1018.07 17.575 2.36 SSD (2L) 1000 898.71 15.658 2.68 SSD (4L) 1000 672.09 13.095 4.1

UDPM 3 1.88 30.58 7.51 UDMP (w/o Adv. & Perceptual loss) 3 1.84 31.63 98.61

[Figure 92]

Figure 12. UDPM generations w/ (left) and w/o (right) adversarial and perceptual losses.

SSD can be viewed as a generalization of UDPM. However, UDPM should be considered as a GAN instead of diffusion, as their performance degrades without perceptual and adversarial losses (Table 14) and generations are washed out (Fig. 12, right). Nonetheless, even without extra losses, SSD outperforms UDPM in FID and training time. Furthermore, UDPM has not been tested at resolutions higher than 64.

Latent Diffusion Models (LDM) [32]. LDMs operate in latent space with different architectures and rely on a compute-intensive pipeline, including two-stage VAE training on large-scale datasets such as OpenImages, making fair comparison difficult. Nonetheless, Table 13 shows SSD (6L) is faster than LDM. Moreover, SSD can be applied in latent space as a multi-resolution interpolation degradation, enabling more efficient Scale Space LDMs.

Low-res diffusion + super-res. Another baseline could be using a low-resolution generation and applying a superresolution model over it. Table 12 shows that even with a pretrained LDM super-res model trained for 3× more iterations, and on a large dataset, SSD has better performance. Adding multiple stages normally leads to distribution shifts as well as more inference steps coming from different stages. PixelFlow [6], DFM [13]. These are flow-based DiT models with differential equation solver-based sampling, and hence, are hard to compare fairly against. Nonetheless, in a fair setting in section 8.2.2, we recreate a multi-res pixel diffusion similar to PixelFlow, and show that using Flexi-UNet formulation outperforms it (Table 11).

##### 8.5. Quantitative Results

Number of Inference Steps. In Table 7, we compare inference speed across different samplers and denoising steps. We report DDPM sampling with the default 1000 steps, a reduced 250 step process, and DDIM with 25 steps. For our

method, we report results with 1000 and 250 DDPM steps, since SSD is formulated in the DDPM setting. We observe that reducing the number of diffusion steps leads to a much larger performance degradation for DDPM-ϵ and DDPM-x0 than for our approach. This aligns with prior observations that DDPM-ϵ models trained with the Lsimple loss (with fixed sigmas) deteriorate substantially when the number of sampling steps is reduced [29], which is reflected in our results as well.

We note that SSD degrades far less when reducing the sampling steps to 250, while also providing substantial inference speedups. However, to ensure a fair comparison against baselines, we report all final quantitative results in the paper using the standard 1000-step setting. The speedup column in Table 7 reports the speedup obtained in generating a batch of 256 samples relative to the time taken by DDPM-x0.

Lanczos sampling overhead. Table 8 shows that the overhead of using Lanczos instead of torch.randn call is negligible, since it is applied only in the resolution-changing steps (1× in SSD (2L), 2× in SSD (3L)). Refer to Table 7 for comparison against DDPM.

##### 8.6. Qualitative Results

We show qualitative results of SSD on every setting that we have trained and noted in Tables 2, and 3. For every setting, we show the progression of noisy states xt and predicted clean images xr0(,θt−1) during generation, and a grid of generated images. The results start on the next page.

[Figure 93]

Figure 13. Progression of noisy states xt during generation using SSD (Flexi-UNet, 3L) on CelebA-256.

[Figure 94]

Figure 14. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 3L) on CelebA-256.

[Figure 95]

Figure 15. Generated Samples using SSD (Flexi-UNet, 3L) on CelebA-256.

[Figure 96]

Figure 16. Progression of noisy states xt during generation using SSD (Flexi-UNet, 4L) on CelebA-256.

[Figure 97]

Figure 17. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 4L) on CelebA-256.

[Figure 98]

Figure 18. Generated Samples using SSD (Flexi-UNet, 4L) on CelebA-256.

[Figure 99]

Figure 19. Progression of noisy states xt during generation using SSD (Flexi-UNet, 6L) on CelebA-256.

[Figure 100]

Figure 20. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 6L) on CelebA-256.

[Figure 101]

Figure 21. Generated Samples using SSD (Flexi-UNet, 6L) on CelebA-256.

[Figure 102]

Figure 22. Progression of noisy states xt during generation using SSD (Flexi-UNet, 3L) on CelebA-128.

[Figure 103]

Figure 23. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 3L) on CelebA-128.

[Figure 104]

Figure 24. Generated Samples using SSD (Flexi-UNet, 3L) on CelebA-128.

[Figure 105]

Figure 25. Progression of noisy states xt during generation using SSD (Flexi-UNet, 5L) on CelebA-128.

[Figure 106]

Figure 26. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 5L) on CelebA-128.

[Figure 107]

Figure 27. Generated Samples using SSD (Flexi-UNet, 5L) on CelebA-128.

[Figure 108]

Figure 28. Progression of noisy states xt during generation using SSD (Flexi-UNet, 2L) on CelebA-64.

[Figure 109]

Figure 29. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 2L) on CelebA-64.

[Figure 110]

Figure 30. Generated Samples using SSD (Flexi-UNet, 2L) on CelebA-64.

[Figure 111]

Figure 31. Progression of noisy states xt during generation using SSD (Flexi-UNet, 3L) on CelebA-64.

[Figure 112]

Figure 32. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 3L) on CelebA-64.

[Figure 113]

Figure 33. Generated Samples using SSD (Flexi-UNet, 3L) on CelebA-64.

[Figure 114]

Figure 34. Progression of noisy states xt during generation using SSD (Flexi-UNet, 4L) on CelebA-64.

[Figure 115]

Figure 35. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 4L) on CelebA-64.

[Figure 116]

Figure 36. Generated Samples using SSD (Flexi-UNet, 4L) on CelebA-64.

[Figure 117]

[Figure 118]

[Figure 119]

- Figure 37. Progression of noisy states xt during generation using SSD (Flexi-UNet, 2L) on ImageNet-64. Here we show the progression of

- 3 samples; each pair of rows corresponds to a single sample.

[Figure 120]

[Figure 121]

[Figure 122]

- Figure 38. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 2L) on ImageNet-64. Here we show the progression of 3 samples; each pair of rows corresponds to a single sample.

[Figure 123]

Figure 39. Generated Samples using SSD (Flexi-UNet, 2L) on ImageNet-64.

[Figure 124]

[Figure 125]

[Figure 126]

###### Figure 40. Progression of noisy states xt during generation using SSD (Flexi-UNet, 4L) on ImageNet-64. Here we show the progression of 3 samples; each pair of rows corresponds to a single sample.

[Figure 127]

[Figure 128]

[Figure 129]

###### Figure 41. Progression of predicted clean images xr0,θ(t−1) during generation using SSD (Flexi-UNet, 4L) on ImageNet-64. Here we show the progression of 3 samples; each pair of rows corresponds to a single sample.

[Figure 130]

###### Figure 42. Generated Samples using SSD (Flexi-UNet, 4L) on ImageNet-64.

- 9. Mathematical Derivations In this section, we provide derivations for various mathematical results provided in the main paper.

##### 9.1. Forward Transition

- Theorem 1 (Forward Transition). Let a generalized linear diffusion process be defined by xt = Mtxt−1 + ηt, ηt ∼ N(0,Σt|t−1), (10)

and suppose the marginal distribution satisfies

q(xt | x0) = N(µt,Σt). (11) Then the transition mean and covariance are given by

µt = M1:tx0 (12)

Σt|t−1 = Σt − MtΣt−1MtT. (13) Proof. The mean part is true by design of the cumulative linear operator. Here we derive Σt|t−1.

xt = Mtxt−1 + ηt, ηt ∼ N(0,Σt|t−1)

= Mt(M1:t−1x0 + ϵt−1) + ηt, ϵt−1 ∼ N(0,Σt−1)

= M1:tx0 + Mtϵt−1 + ηt. Hence,

Cov(xt | x0) = Cov(Mtϵt−1 + ηt | x0)

= Cov(Mtϵt−1 | x0) + Cov(ηt) (independence)

= MtΣt−1MtT + Σt|t−1. Σt = MtΣt−1MtT + Σt|t−1,

=⇒ Σt|t−1 = Σt − MtΣt−1MtT.

| |
|---|

9.2. Posterior Distribution

- Theorem 2 (Posterior Distribution). Consider the linear generalized linear diffusion process xt = Mtxt−1 + ηt, ηt ∼ N(0,Σt|t−1), (14)

with marginals

q(xt−1 | x0) = N(µt−1,Σt−1), (15)

q(xt | x0) = N(µt,Σt). (16) Then the posterior distribution

q(xt−1 | xt,x0) (17) is Gaussian:

q(xt−1 | xt,x0) = N(µt→t−1,Σt→t−1), (18) with

Σt→t−1 = (Σ−t−11 + MtTΣ−t|t1−1Mt)−1, (19) µt→t−1 = Σt→t−1 Σ−t−11µt−1 + MtTΣ−t|t1−1xt . (20)

q(xt| xt−1,x0)q(xt−1| x0) q(xt| x0)

q(xt−1| xt,x0) =

q(xt| xt−1)q(xt−1| x0) q(xt| x0)

=

(a)

∝ exp − (xt − Mtxt−1)⊤Σ−t|t1−1(xt − Mtxt−1) − (xt−1 − µt−1)⊤Σ−t−11(xt−1 − µt−1) (b)

+ (xt − µt)⊤Σ−t 1(xt − µt)

= exp − xTt Σ−t|t1−1xt − (xTt Σ−t|t1−1Mtxt−1 + (xTt Σ−t|t1−1Mtxt−1)T) + xTt−1MtTΣ−t|t1−1Mtxt−1

− xTt−1Σ−t−11xt−1 − (µTt−1Σ−t−11xt−1 + (µTt−1Σ−t−11xt−1)T) + µTt−1Σt−1µt−1

###### + C1(x0,xt)

= exp − xTt−1(Σ−t−11 + MtTΣ−t|t1−1Mt)xt−1

+ (xTt Σ−t|t1−1Mt + µTt−1Σ−t−11)xt−1 + ((xTt Σ−t|t1−1Mt + µTt−1Σ−t−11)xt−1)T

###### + C2(x0,xt)

In Eq. a, we first use Bayes’ rule, and then use the Markov chain assumption. In Eq. b, we then substitute the marginal (Eq. 3) and forward transition (Eq. 4) distributions. Then we start collecting the terms quadratic (red) and linear (blue) in xt−1. From the quadratic and linear terms, we can complete the square and hence extract the mean and variance of the posterior normal distribution:

Σt→t−1 = (Σ−t−11 + MtTΣ−t|t1−1Mt)−1 µt→t−1 = Σt→t−1(xTt Σ−t|t1−1Mt + µTt−1Σ−t−11)T

= Σt→t−1(MtTΣ−t|t1−1xt + Σ−t−11µt−1).

The last step comes from the fact that for a symmetric matrix A, (A−1)T = A−1, and covariance matrices are symmetric.

##### 9.3. Posterior Under Isotropic Marginals

- Theorem 3 (Closed-Form Posterior Under Isotropic Marginals). Assume isotropic marginals

| |
|---|

Σt = σt2I, Σt−1 = σt2−1I. (21) Then the posterior covariance simplifies to

σt4−1 σt2

Σt→t−1 = σt2−1I −

MtTMt, (22) and the posterior mean simplifies to

µt→t−1 = µt−1 +

σt2−1 σt2

MtT (xt − Mtµt−1). (23)

Σt→t−1 = (Σ−t−11 + MtTΣ−t|t1−1Mt)−1

= Σt−1 − Σt−1MtT(Σt|t−1 + MtΣt−1MtT)−1MtΣt−1 (c) = Σt−1 − Σt−1MtTΣ−t 1MtΣt−1 (d)

σt4−1 σt2

= σt2−1I −

MtTMt (e)

We start from Eq. 5 derived in the previous section. In Eq. c, we used the Woodbury matrix identity (A + UCV )−1 = A−1 − A−1U(C−1 + V A−1U)−1V A−1 with A = Σ−t−11,U = MtT,C = Σ−t|t1−1,V = Mt. In Eq. d, we substitute the value of Σt|t−1 from Eq. 4. Finally, in Eq. e we substitute the isotropic values for Σt and Σt−1.

µt→t−1 = Σt→t−1(Σ−t−11µt−1 + MtTΣ−t|t1−1xt) = Σt→t−1Σ−t−11µt−1 + Σt→t−1MtTΣ−t|t1−1xt

= (Σt−1 − Σt−1MtTΣ−t 1MtΣt−1)Σ−t−11µt−1 + Σt→t−1MtTΣ−t|t1−1xt (f)

= (I − Σt−1MtTΣ−t 1Mt)µt−1 + Σt→t−1MtTΣ−t|t1−1xt

= (I − Σt−1MtTΣ−t 1Mt)µt−1 + (Σ−t−11 + MtTΣt|t−1Mt)−1MtTΣ−t|t1−1xt (g) = (I − Σt−1MtTΣ−t 1Mt)µt−1 + Σ−t−11MtT(Σt|t−1 + MtΣt−1MtT)−1xt (h) = (I − Σt−1MtTΣ−t 1Mt)µt−1 + Σ−t−11MtTΣ−t 1xt (i) = µt−1 + Σt−1MtTΣ−t 1(xt − Mtµt−1)

σt2−1 σt2

MtT(xt − Mtµt−1) (j)

= µt−1 +

Starting from µt→t−1 in Eq. 5, in Eq. f we substitute Σt→t−1 from Eq. d, and then in Eq g we substitute Σt→t−1 from Eq. 5. In Eq. h, we use a corollary of Woodbury identity (A + UCV )−1UC = A−1U(C−1 + V A−1U)−1, with the same substitution as described above. In Eq. i, we substitute the value of Σt|t−1 from Eq. 4, and finally, we substitute the isotropic values for Σt and Σt−1.

| |
|---|

#### References

- [1] Shady Abu-Hussein and Raja Giryes. Udpm: Upsampling diffusion probabilistic models. arXiv preprint arXiv:2305.16269,

2023. 1, 3, 2, 5

- [2] Yuval Atzmon, Maciej Bala, Yogesh Balaji, Tiffany Cai, Yin Cui, Jiaojiao Fan, Yunhao Ge, Siddharth Gururani, Jacob Huffman, Ronald Isaac, et al. Edify image: High-quality image generation with pixel space laplacian diffusion models. arXiv preprint arXiv:2411.07126, 2024. 1, 2
- [3] Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise. NeurIPS, 2023. 1
- [4] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 1
- [5] John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 2009. 2
- [6] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025. 1, 2, 4, 8, 3, 5
- [7] Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z Kaplan, and Enrico Shippole. Scalable high-resolution pixel-space image synthesis with hourglass diffusion transformers. In Forty-first International Conference on Machine Learning, 2024. 1
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 2, 7
- [9] Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. NeurIPS, 2015. 2
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021. 6, 7, 1
- [11] Gene H Golub and G´erard Meurant. Matrices, moments and quadrature with applications. Princeton University Press,

2009. 5

- [12] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Joshua M Susskind, and Navdeep Jaitly. Matryoshka diffusion models. In ICLR,

2024. 2, 1

- [13] Moayed Haji-Ali, Willi Menapace, Ivan Skorokhodov, Arpit Sahni, Sergey Tulyakov, Vicente Ordonez, and Aliaksandr Siarohin. Improving progressive generation with decomposable flow matching. arXiv preprint arXiv:2506.19839, 2025. 3, 5
- [14] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7441–7451, 2023. 5
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two

- time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 8
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 1, 2, 3, 5, 7
- [17] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR, 2022. 2
- [18] Emiel Hoogeboom and Tim Salimans. Blurring diffusion models. ICLR, 2023. 1, 4, 5, 7, 8
- [19] Wongi Jeong, Kyungryeol Lee, Hoigi Seo, and Se Young Chun. Upsample what matters: Region-adaptive latent sampling for accelerated diffusion transformers. arXiv preprint arXiv:2507.08422, 2025. 1, 2, 4, 8, 3
- [20] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. 1, 2, 4, 8, 3
- [21] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. ICLR, 2018. 2, 1
- [22] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 7
- [23] Cornelius Lanczos. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. Journal of research of the National Bureau of Standards, 45(4):255–282, 1950. 5
- [24] Tony Lindeberg. Scale-space theory: A basic tool for analyzing structures at different scales. Journal of applied statistics, 21(1-2):225–270, 1994. 1, 2
- [25] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015. 2, 7
- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [27] David G Lowe. Object recognition from local scale-invariant features. In Proceedings of the seventh IEEE international conference on computer vision, pages 1150–1157. Ieee, 1999. 2
- [28] Soumik Mukhopadhyay, Matthew Gwilliam, Yosuke Yamaguchi, Vatsal Agarwal, Namitha Padmanabhan, Archana Swaminathan, Tianyi Zhou, Jun Ohya, and Abhinav Shrivastava. Do text-free diffusion models learn discriminative visual representations? In European Conference on Computer Vision, pages 253–272. Springer, 2024. 1
- [29] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 6

- [30] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 4

- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2, 1

- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 1, 5
- [33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 2, 1
- [34] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 5
- [35] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 2

- [36] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ICLR, 2021. 1
- [37] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 2
- [38] Ye Tian, Xin Xia, Yuxi Ren, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Yunhai Tong, Ling Yang, and Bin Cui. Trainingfree diffusion acceleration with bottleneck sampling. arXiv preprint arXiv:2503.18940, 2025. 2
- [39] XIANG Weilai. FutureXiang/Diffusion, 2025. original-date: 2022-10-18T11:42:46Z. 2

