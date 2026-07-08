## Gaussian Mixture Flow Matching Models

# arXiv:2504.05304v3[cs.LG]30Aug2025

#### Hansheng Chen1 Kai Zhang2 Hao Tan2 Zexiang Xu3 Fujun Luan2 Leonidas Guibas1 Gordon Wetzstein1 Sai Bi2

https://github.com/Lakonik/GMFlow

Noise Data

### Abstract

𝑞𝑞𝜽𝜽 𝒙𝒙𝑡𝑡3 𝒙𝒙𝑇𝑇 𝑞𝑞𝜽𝜽 𝒙𝒙𝑡𝑡2 𝒙𝒙𝑡𝑡3 𝑞𝑞𝜽𝜽 𝒙𝒙𝑡𝑡1 𝒙𝒙𝑡𝑡2 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡1

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Diffusion models approximate the denoising distribution as a Gaussian and predict its mean, whereas flow matching models reparameterize the Gaussian mean as flow velocity. However, they underperform in few-step sampling due to discretization error and tend to produce oversaturated colors under classifier-free guidance (CFG). To address these limitations, we propose a novel Gaussian mixture flow matching (GMFlow) model: instead of predicting the mean, GMFlow predicts dynamic Gaussian mixture (GM) parameters to capture a multi-modal flow velocity distribution, which can be learned with a KL divergence loss. We demonstrate that GMFlow generalizes previous diffusion and flow matching models where a single Gaussian is learned with an L2 denoising loss. For inference, we derive GM-SDE/ODE solvers that leverage analytic denoising distributions and velocity fields for precise few-step sampling. Furthermore, we introduce a novel probabilistic guidance scheme that mitigates the over-saturation issues of CFG and improves image generation quality. Extensive experiments demonstrate that GMFlow consistently outperforms flow matching baselines in generation quality, achieving a Precision of 0.942 with only 6 sampling steps on ImageNet 256×256.

Network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡3

Network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡2

Network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡1

Network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑇𝑇

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑇𝑇 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡3 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡2 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡1

(a) Flow – DDPM 4 steps (learned variance)

𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡 = 𝒩𝒩 𝒖𝒖;𝝁𝝁,𝑠𝑠2𝑰𝑰

𝑞𝑞𝜽𝜽 𝒙𝒙𝑡𝑡3 𝒙𝒙𝑇𝑇 𝑞𝑞𝜽𝜽 𝒙𝒙𝑡𝑡2 𝒙𝒙𝑡𝑡3 𝑞𝑞𝜽𝜽 𝒙𝒙𝑡𝑡1 𝒙𝒙𝑡𝑡2 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡1

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

GM network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡3

GM network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡2

GM network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡1

GM network 𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑇𝑇

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑇𝑇 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡3 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡2 𝑞𝑞𝜽𝜽 𝒙𝒙0 𝒙𝒙𝑡𝑡1

32

(b) GMFlow – GM-SDE 4 steps

𝐴𝐴𝑘𝑘𝒩𝒩 𝒖𝒖; 𝝁𝝁𝑘𝑘,𝑠𝑠2𝑰𝑰

𝑞𝑞𝜽𝜽 𝒖𝒖 𝒙𝒙𝑡𝑡 =

𝑘𝑘

Figure 1. Comparison between vanilla diffusion (with flow velocity parameterization) and GMFlow on a 2D checkerboard distribution. (a) The vanilla diffusion model predicts mean velocity, modeling the denoising distribution qθ(x0|xt) as a single Gaussian, and then samples from a Gaussian transition distribution qθ(xt−∆t|xt). (b) GMFlow predicts a GM for velocity and yields a multi-modal GM denoising distribution, from which the transition distribution can be analytically derived for precise next-step sampling, allowing more accurate few-step sampling (4 steps in this case).

### 1. Introduction

Diffusion probabilistic models (Sohl-Dickstein et al., 2015; Ho et al., 2020), score-based models (Song & Ermon, 2019; Song et al., 2021b), and flow matching models (Lipman et al., 2023; Liu et al., 2022) form a family of generative models that share an underlying theoretical framework and

have made significant advances in image and video generation (Yang et al., 2023; Po et al., 2024; Rombach et al.,

1Stanford University, CA 94305, USA 2Adobe Research, CA 95110, USA 3Hillbot. Correspondence to: Hansheng Chen <hanshengchen@stanford.edu>.

- 2022; Saharia et al., 2022b; Podell et al., 2024; Chen et al., 2024; Esser et al., 2024; Blattmann et al., 2023; Hong et al.,
- 2023; HaCohen et al., 2024; Kong et al., 2025). Standard diffusion models approximate the denoising distribution as

Preliminary work.

a Gaussian and train neural networks to predict its mean, and optionally its variance (Nichol & Dhariwal, 2021; Bao et al., 2022b;a). Flow matching models reparameterize the Gaussian mean as flow velocity, formulating an ordinary differential equation (ODE) that maps noise to data. These formulations remain dominant in image generation as per user studies (Artificial Analysis, 2025; Jiang et al., 2024).

However, vanilla diffusion and flow models require tens of sampling steps for high-quality generation, since Gaussian approximations hold only for small step sizes, and numeric ODE integration introduces discretization errors. In addition, high quality generation requires a higher classifier-free guidance (CFG) scale (Ho & Salimans, 2021), yet stronger CFG often leads to over-saturated colors (Saharia et al., 2022a; Kynk¨a¨anniemi et al., 2024) due to out-of-distribution (OOD) extrapolation (Bradley & Nakkiran, 2024), thus limiting overall image quality even with hundreds of steps.

To address these limitations, we deviate from previous single-Gaussian assumption and introduce Gaussian mixture flow matching (GMFlow). Unlike vanilla flow models, which predict the mean of flow velocity u, GMFlow predicts the parameters of a Gaussian mixture (GM) distribution, representing the probability density function (PDF) of u. This provides two key benefits: (a) our GM formulation captures more intricate denoising distributions, enabling more accurate transition estimates at larger step sizes and thus requiring fewer steps for high-quality generation (Fig. 1); (b) CFG can be reformulated by reweighting the GM probabilities rather than extrapolation, thus bounding the samples within the conditional distribution and avoiding over-saturation, thereby improving overall image quality.

We train GMFlow by minimizing the KL divergence between the predicted velocity distribution and the ground truth distribution, which we show is a generalization of previous diffusion and flow matching models (§ 3.1). For inference, we introduce novel SDE and ODE solvers that analytically derive the reverse transition distribution and flow velocity field from the predicted GM, enabling fast and precise few-step sampling (§ 3.3). Meanwhile, we develop a probabilistic guidance approach for conditional generation, which reweights the GM PDF using a Gaussian mask to enhance condition alignment (§ 3.2).

For evaluation, we compare GMFlow against vanilla flow matching baselines on both 2D toy dataset and ImageNet (Deng et al., 2009). Extensive experiments reveal that GMFlow consistently outperforms baselines equipped with advanced solvers (Lu et al., 2022; 2023; Zhao et al., 2023; Karras et al., 2022). Notably, on ImageNet 256×256, GMFlow excels in both Precision and FID metrics with fewer than 8 sampling steps; with 32 sampling steps, GMFlow achieves a state-of-the-art Precision of 0.950 (Fig. 6).

The main contributions of this paper are as follows:

- • We propose GMFlow, a generalized formulation of diffusion models based on GM denoising distributions.
- • We introduce a GM-based sampling framework consisting of novel SDE/ODE solvers and probabilistic guidance.
- • We empirically validate that GMFlow outperforms flow matching baselines in both few- and many-step settings.

### 2. Diffusion and Flow Matching Models

In this section, we provide background on diffusion and flow matching models as the basis for GMFlow. Note that we introduce flow matching as a special diffusion parameterization since they largely overlap in practice (Albergo & Vanden-Eijnden, 2023; Gao et al., 2024).

Forward diffusion process. Let x ∈ RD be a data point sampled from a distribution with its PDF denoted by p(x). A typical diffusion model defines a time-dependent interpolation between the data point and a random Gaussian noise ϵ ∼ N(0,I), yielding the noisy data xt = αtx + σtϵ, where t ∈ [0,T] denotes the diffusion time, and αt,σt are the predefined time-dependent monotonic coefficients (noise schedule) that satisfy the boundary condition x0 = x, xT ≈ ϵ. Apparently, the marginal PDF of the noisy data p(xt) can be written as the data distribution p(x) convolved by a Gaussian kernel p(xt|x0) = N(xt;αtx0,σt2I), i.e., p(xt) = R

p(xt|x0)p(x0)dx0. Alternatively, p(xt) can be expressed as a previous PDF p(xt−∆t) convolved by a transition Gaussian kernel p(xt|xt−∆t), given by:

D

p(xt|xt−∆t)

βt,∆t

αt2 αt2−∆t

αt αt−∆t

σt2 −

σt2−∆t I , (1)

= N xt;

xt−∆t,

where the transition variance is denoted by βt,∆t for brevity. A series of convolutions from 0 to T constructs a diffusion process over time. With infinitesimal step size ∆t, this process can be continuously modeled by the forward-time SDE (Song et al., 2021b).

Reverse denoising process. Diffusion models generate samples by first sampling the noise xT ← ϵ and then reversing the diffusion process to obtain x0. This can be achieved by recursively sampling from the reverse transition distribution p(xt−∆t|xt) = p(x

t−∆t)p(xt|xt−∆t)

p(xt) . DDPM (Ho et al., 2020) approximate p(xt−∆t|xt) with a Gaussian and reparameterize its mean in terms of residual noise, which is then learned by a neural network. Its sampling process is equivalent to a first-order solver of a reverse-time SDE. Song et al. (2021b) reveal that the time evolution of the marginal distribution p(xt) described by the SDE is the same as that described by a flow ODE, which maps the noise xT to data x0 deterministically. In particular, flow matching models

train a neural network to directly predict the ODE velocity field dxt

dt . They typically adopt a linear noise schedule, which defines T := 1,αt := 1 − t,σt := t, thus yielding a simplified velocity formulation dxt

0∼p(x0|xt)[u], with the random flow velocity u defined as:

dt = Ex

xt − x0 σt

. (2)

u :=

This reveals that the velocity field dxt

dt represents the mean of the random velocity u over the denoising distribution

0)p(xt|x0)

- p(x0|xt) = p(x

p(xt) .

Flow matching loss. Flow models stochastically regress

dxt

dt using randomly paired samples of x0 and xt. Let µθ(xt) denote a flow velocity neural network with learnable parameters θ, the L2 flow matching loss is given by:

L = Et,x

0,xt

- 1

- 2∥u − µθ(xt)∥2 . (3)

In practice, additional condition signals c (e.g., class label or text prompt) can be fed to the network µθ(xt,c), making it a conditioned diffusion model that learns p(x0|c).

Limitations. Sampling errors in flow models can arise from two sources: (a) discretization errors in SDE/ODE solvers, and (b) inaccurate flow prediction µθ(xt,c) due to underfitting (Karras et al., 2024). While discretization errors can be reduced by reducing step size, it increases the number of function evaluations (NFE), causing significant computational overhead. To mitigate prediction inaccuracies, CFG (Ho & Salimans, 2021) performs an extrapolation of conditional and unconditional predictions, given by wµθ(xt,c) + (1 − w)µθ(xt), where w ∈ [1,+∞) is the guidance scale. Such a method improves image quality and condition alignment at the expense of diversity. However, higher w may lead to OOD samples and cause image oversaturation, which is often mitigated with heuristics such as thresholding (Saharia et al., 2022a; Sadat et al., 2025).

### 3. Gaussian Mixture Flow Matching Models

In this section, we introduce our GMFlow models, covering its parameterization and loss function (§ 3.1), probabilistic guidance mechanism (§ 3.2), GM-SDE/ODE solvers (§ 3.3), and other practical designs for image generation (§ 3.4). Table 1 summarizes the key differences between GMFlow and vanilla flow models. Algorithms 1 and 2 present the outlines of training and sampling schemes, respectively.

#### 3.1. Parameterization and Loss Function

Different from vanilla flow models that regress mean velocity, we model the velocity distribution p(u|xt) as a Gaussian

Table 1. Comparison between vanilla flow models and GMFlow.

Vanilla diffusion flow GMFlow

Transition assumption

∆t is small → p(xt−∆t|xt) is approximately a Gaussian

Derive p(xt−∆t|xt) from the predicted GM

GM params in qθ(u|xt) = Kk=1 AkN(u; µk, s2I)

Network output

Mean of flow velocity µθ(xt)

Training loss Et,x

- 1

- 2∥u − µθ(xt)∥2 Et,x

0,xt[−log qθ(u|xt)]

0,xt

Sampling methods

1st-ord: GM-SDE/ODE 2nd-ord: GM-SDE/ODE 2 Guidance Mean extrapolation

1st-ord: Euler, DDPM... 2nd-ord: DPM++, UniPC...

GM reweighting

w(u)

wµθ(xt, c) + (1 − w)µθ(xt)

Z qθ(u|xt, c)

mixture (GM):

K

AkN(u;µk,Σk), (4)

qθ(u|xt) =

k=1

where {Ak,µk,Σk} are dynamic GM parameters predicted by a network with parameters θ, and K is a hyperparameter specifying the number of mixture components. To enforce

k Ak = 1 with Ak ≥ 0, we employ a softmax activation Ak = expa

k expak , where {ak ∈ R} are pre-activation logits. We train GMFlow by matching the predicted distribution qθ(u|xt) with the ground truth velocity distribution p(u|xt). Specifically, we minimize the Kullback–Leibler (KL) divergence (Kullback & Leibler, 1951) Eu∼p(u|x

k

t)[log p(u|xt)−log qθ(u|xt)], where log p(u|xt) does not affect backpropagation and can be omitted. During training, we sample u ∼ p(u|xt) by drawing a pair of x0, xt, and then calculate the velocity using Eq. (2). The resulting loss function is therefore reformulated as:

0,xt[−log qθ(u|xt)]. (5) In Eq. (6), we present an expanded form of this loss function. Why choosing Gaussian mixture? While there are a large family of parameterized distributions, we choose Gaussian mixture for its following desired properties:

L = Et,x

• Mean alignment. The mean of the ground truth distribu-

- tion p(u|xt) is crucial since it decides the velocity term in the flow ODE and the drift term in the reverse-time SDE (§ C.1). When the loss in Eq. (5) is minimized (assuming sufficient network capacity), the GM distribu-
- tion qθ(u|xt) guarantees that its mean aligns with that of p(u|xt). Theorem 3.1. Given any distribution p(u) and any sym-

metric positive definite matrices {Σk}, if {a∗k,µ∗k} are the optimal GM parameters w.r.t. with the objective

k,µk} Eu∼p(u)[−log k AkN(u;µk,Σk)], then the GM mean satisfies k A∗kµ∗k = Eu∼p(u)[u].

min{a

We provide the proof of the theorem in § C.2. It’s worth pointing out that not all distributions satisfy this property

(e.g. the mean of Laplace distribution aligns to the median instead of the mean).

- • Analytic calculation. GM enables necessary calculations to be approached analytically (e.g., for deriving the mean and the transition distribution), as detailed in § D.
- • Expressiveness. GMs can approximate intricate multimodal distributions. With sufficient number of components, GMs are theoretically universal approximators (Huix et al., 2024).

Simplified covariances. The expansion of Eq. (5) involves the inverse covariance matrix Σ−k 1, which can lead to training instability. To mitigate this, we simplify each covariance to a scaled identity matrix, i.e., Σk = s2I, where s ∈ R+ is the predicted standard deviation shared by all mixture components. It’s worth pointing out that this simplification does not limit the GM’s expressiveness, which is mainly dominated by K. Moreover, Theorem 3.1 implies that the structure of the covariance matrix is irrelevant to the accuracy of the mean velocity, mitigating the need for intricate covariances. Under this simplification, the expanded GM KL loss is reduced to:

K

L = Et,x

0,xt − log

exp

k=1

- 1

- 2s2∥u − µk∥2 − D log s + log Ak , (6)

−

which can be interpreted as a hybrid of regression loss to the centroids and classification loss to the components.

Special cases of GMFlow. GMFlow generalizes several formulations of previous diffusion and flow models. In a special case where K = 1,s = 1, Eq. (6) is identical to the L2 loss in Eq. (3). Therefore, GMFlow is a generalization of vanilla diffusion and flow models. In another case where {µk} are velocities towards predefined tokens and s ≈ 0, then {Ak} represent token probabilities, making Eq. (6) analogous to categorical diffusion objectives (Gu et al., 2022; Dieleman et al., 2022; Campbell et al., 2022).

u-to-x0 reparameterization. While the neural network directly outputs the u distribution, we can flexibly reparameterize it into an x0 distribution by substituting u = x

t−x0

σt into Eq. (4), yielding qθ(x0|xt) =

k AkN(x0;µxk,s2xI), with the new parameters µxk = xt − σtµk and sx = σts. The velocity KL loss is thus equivalent to an x0 likelihood loss L = Et,x

0,xt[−log qθ(x0|xt)].

#### 3.2. Probabilistic Guidance via GM Reweighting

Vanilla CFG suffers from over-saturation due to unbounded extrapolation, which overshoots samples beyond the valid data distribution. In contrast, GMFlow can provide a well-

defined conditional distribution qθ(u|xt,c). This allows us to formulate probabilistic guidance, a principled approach that reweights the predicted distribution while preserving its intrinsic bounds and structure.

To reweight the GM PDF qθ(u|xt,c) analytically, we multiply it by a Gaussian mask w(u):

w(u) Z

qθ(u|xt,c), (7)

qw(u|xt,c) :=

where Z is a normalization factor. The reweighted qw(u|xt,c) remains a GM with analytically derived parameters (see § D.2 for derivation), and is treated as the model output for sampling.

Then, our goal is to design w(u) so that it enhances condition alignment without inducing OOD samples. To this end, we approximate the conditional and unconditional GM predictions qθ(u|xt,c) and qθ(u|xt) as isotropic Gaussian surrogates N(u;µc,s2cI) and N(u;µu,s2uI) by matching the mean and total variance of the GM (see § A.1 for details). Using these approximations, we define the unnormalized Gaussian mask as:

w(u) := N u;µc + ws˜ c∆µn, 1 − w˜2 s2cI N(u;µc,s2cI)

, (8)

where w˜ ∈ [0,1) is the probabilistic guidance scale, and ∆µn := µ

c−µu ∥µc−µu∥/

D is the normalized mean difference.

√

Intuitively, the numerator in Eq. (8) shifts the conditional mean by the bias ws˜ c∆µn to enhance conditioning, while reducing the conditional variance according to bias–variance decomposition. Notably, for any w˜, a sample u from the numerator Gaussian satisfies Eu ∥u − µc∥2/D ≡ s2c. This ensures that the original bounds of the conditional distribution are preserved.

Meanwhile, the denominator in Eq. (8) cancels out the original mean and variance of the conditional GM, so that the reweighted GM qw(u|xt,c) approximately inherits the adjusted mean and variance of the numerator. Since the “masking” operation retains the original GM components, the fine-grained structure of the conditional GM is preserved.

With preserved bounds and structure, probabilistic guidance reduces the risk of OOD samples compared to vanilla CFG, effectively preventing over-saturation and enhancing overall image quality.

#### 3.3. GM-SDE and GM-ODE Solvers

In this subsection, we show that GMFlow enables unique SDE and ODE solvers that greatly reduce discretization errors by analytically deriving the reverse transition distribution and flow velocity field.

GM-SDE solver. Given the predicted x0-based GM

- qθ(x0|xt) = k AkN(x0;µxk,s2xI), the reverse transition distribution qθ(xt−∆t|xt) can be analytically derived (see § C.3 for derivation):

qθ(xt−∆t|xt)

K

AkN xt−∆t;c1xt + c2µxk, c3 + c22s2x I , (9)

=

k=1

2 t−∆t

with the coefficients c1 = σ

αt−∆t, c2 = βt,σ∆2 t

αt

αt−∆t, c3 = βt,σ∆2 t

σt2

t

σt2−∆t. By recursively sampling from qθ(xt−∆t|xt), we obtain a GM approximation to the reverse-time SDE solution. Alternatively, we can implement the solver by first sampling xˆ0 ∼ qθ(x0|xt) and then sampling p(xt−∆t|xt,xˆ0) = N(xt−∆t;c1xt + c2xˆ0,c3I) (Song et al., 2021a) (see § C.3 for details). Theoretically, if qθ(x0|xt) is accurate, the GM-SDE solution incurs no error even in a single step. In practice, a smaller step size ∆t increases reliance on the mean (see § C.4 for details) over the shape of the distribution, which can be more accurate as per Theorem 3.1.

t

GM-ODE solver. While GMFlow supports standard ODE integration by converting its GM prediction into the current mean velocity, it also enables a unique sampling scheme with reduced discretization errors by analytically deriving the flow velocity field for any time τ < t, thereby facilitating sub-step integration without additional neural network evaluations. Specifically, given the x0-based GM qθ(x0|xt) at xt, we show in § C.5 that the denoising distribution at xτ can be derived as:

p(xτ|x0) Z · p(xt|x0)

qθ(x0|xt), (10)

qˆ(x0|xτ) =

where Z is a normalization factor, and qˆ(x0|xτ) is also a GM with analytically derived parameters. This allows us to instantly estimate GMFlow’s next-step prediction at xτ from its current prediction at xt, without re-evaluating the neural network. The velocity can therefore be derived by reparameterizing qˆ(x0|xτ) in terms of u and computing its mean. This enables the GM-ODE solver, which integrates a curved trajectory along the analytic velocity field via multiple Euler sub-steps between t and t − ∆t (Algorithm 2 line 18–22). Theoretically, if qθ(x0|xt) is accurate, the GMODE solution also incurs no error. In practice, the velocity field is only locally accurate as τ → t and xτ → xt, thus multiple network evaluations are still required.

Second-order multistep GM solvers. Vanilla diffusion models often use Adams–Bashforth-like second-order multistep solvers (Lu et al., 2023), which extrapolate the mean x0 predictions of the last two steps to estimate the next midpoint, and then apply an Euler update. Analogously, we extend this approach to GMs. Given GM predictions at times t and t + ∆t, we first convert the latter to time t via

Eq. (10), yielding qˆ(x0|xt). In an ideal scenario where both GMs are accurate, qˆ(x0|xt) perfectly matches qθ(x0|xt); otherwise, their discrepancy is extrapolated following the Adams–Bashforth scheme. To perform GM extrapolation, we adopt a GM reweighting scheme similar to that in § 3.2. More details are provided in § A.2.

#### 3.4. Practical Designs

Pixel-wise factorization. For high-dimensional data such as images, Gaussian mixture models often suffer from mode collapse. To address this, we treat each pixel (in the latent grid for latent diffusion (Rombach et al., 2022)) as an independent low-dimensional GM, making the training loss the sum of per-pixel GM KL terms.

Spectral sampling. Due to the factorization, image generation under GM-SDE solvers performs the sampling step xˆ0 ∼ qθ(x0|xt) independently for each pixel, neglecting spatial correlations. To address this, we adopt a spectral sampling technique that imposes correlations through the frequency domain. Specifically, we generate a spatially correlated Gaussian random field from a learnable power spectrum. The per-pixel Gaussian samples are then mapped to GM samples via Knothe–Rosenblatt transport (Knothe, 1957; Rosenblatt, 1952). More details are provided in § A.3.

Transition loss. Empirically, we observe that the GM KL loss may still induce minor gradient spikes when s becomes small. To stabilize training, we adopt a modified loss based on the transition distributions (Eq. (9)), formulated as Ltrans = Et,x

t−∆t,xt[−log qθ(xt−∆t|xt)], where we define ∆t := λt with the transition ratio hyperparameter λ ∈ (0,1]. When λ = 1, the loss is equivalent to the original; when λ < 1, the transition distribution has a lower bound of variance c3, which improves training stability. The modified training scheme is described in Algorithm 1.

### 4. Experiments

To evaluate the effectiveness of GMFlow, we compare it against vanilla flow matching baselines on two datasets: (a) a simple 2D checkerboard distribution, which facilitates visualization of sample histograms and analysis of underlying mechanisms; (b) the class-conditioned ImageNet dataset (Deng et al., 2009), a challenging benchmark that demonstrates the practical advantages of GMFlow for large-scale image generation.

#### 4.1. Sampling a 2D Checkerboard Distribution

In this subsection, we compare GMFlow with the vanilla flow model on a simple 2D checkerboard distribution, following the experimental settings of Lipman et al. (2023). All configurations adopt the same 5-layer MLP architecture for denoising 2D coordinates, differing only in their output

𝑁𝑁𝑁𝑁𝑁𝑁 = 2 𝑁𝑁𝑁𝑁𝑁𝑁 = 4 𝑁𝑁𝑁𝑁𝑁𝑁 = 8 𝑁𝑁𝑁𝑁𝑁𝑁 = 2 𝑁𝑁𝑁𝑁𝑁𝑁 = 4 𝑁𝑁𝑁𝑁𝑁𝑁 = 8

𝑁𝑁𝑁𝑁𝑁𝑁 = 1 𝑁𝑁𝑁𝑁𝑁𝑁 = 2 𝑁𝑁𝑁𝑁𝑁𝑁 = 4 𝑁𝑁𝑁𝑁𝑁𝑁 = 8 𝑁𝑁𝑁𝑁𝑁𝑁 = 16 𝑁𝑁𝑁𝑁𝑁𝑁 = 32

[Figure 19]

[Figure 20]

[Figure 21]

DDPM large variance

- 𝐾𝐾 = 1
- 𝐾𝐾 = 2

(a)SDE method comparison(b)ODE method comparison

[Figure 22]

[Figure 23]

- (DDIM 𝜂𝜂 = 0)

[Figure 24]

DDPM small variance

- (DDIM 𝜂𝜂 = 1)

[Figure 25]

[Figure 26]

[Figure 27]

DPM++ 2M SDE

𝐾𝐾 = 8

GM-SDE

GM-SDE 2

[Figure 28]

[Figure 29]

[Figure 30]

GM-SDE 2 𝐾𝐾 = 64

- 𝐾𝐾 = 1
- 𝐾𝐾 = 2

[Figure 31]

[Figure 32]

[Figure 33]

Euler

[Figure 34]

[Figure 35]

[Figure 36]

DPM++ 2M

𝐾𝐾 = 8

[Figure 37]

GM-ODE GM-ODE 2

###### (a) GMFlow 𝐾𝐾 and solver comparison

UniPC

𝑁𝑁𝑁𝑁𝑁𝑁 = 2 𝑁𝑁𝑁𝑁𝑁𝑁 = 4 𝑁𝑁𝑁𝑁𝑁𝑁 = 8 𝑁𝑁𝑁𝑁𝑁𝑁 = 2 𝑁𝑁𝑁𝑁𝑁𝑁 = 4 𝑁𝑁𝑁𝑁𝑁𝑁 = 8

[Figure 38]

[Figure 39]

[Figure 40]

GM-ODE 2 𝐾𝐾 = 64

𝐾𝐾 = 8

GM-SDE 2 GM-SDE 2 w/o 𝑞𝑞 𝒙𝒙0 𝒙𝒙𝑡𝑡 conversion

Figure 2. Comparison among vanilla flow models with different solvers and GMFlow. For both SDE and ODE, our method achieves higher quality in few-step sampling.

###### (b) Ablation study on the 𝑞𝑞 𝒙𝒙0 𝒙𝒙𝑡𝑡 conversion

- Figure 3. (a) Comparison of first- and second-order GM-SDE and GM-ODE solvers with varying NFE and GM components K. Increasing K improves few-step sampling results. Second-order solvers produce sharper histograms with fewer outliers than firstorder solvers. (b) Ablation study on the qˆ(x0|xt) conversion in second-order solvers. Removing this conversion causes samples to overshoot and concentrate at the edges.

DiT

[Figure 41]

𝒄𝒄,𝑡𝑡

Stop grad Exp

𝑠𝑠

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Soft max

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

𝒙𝒙𝑡𝑡

𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

Pixel-wise {𝝁𝝁𝑘𝑘}

𝐾𝐾 × 𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

Pixel-wise {𝐴𝐴𝑘𝑘}

𝐾𝐾 × 1 × 𝐻𝐻 × 𝑊𝑊

Embedding 2-Layer MLP

- Figure 4. Architecture of GMFlow-DiT. The original DiT (Peebles & Xie, 2023) is shown in blue, and the modified output layers are shown in purple.

channels. For GMFlow, we train multiple models with different numbers of GM components K (with λ = 0.9); for GM-ODE sampling, we use n = ⌈128/NFE⌉ sub-steps.

Comparison against flow model baseline. In Fig. 2, we compare the 2D sample histograms of GMFlow using second-order GM solvers (GM-SDE/ODE 2) against the vanilla flow matching baseline using established SDE and ODE solvers (Lu et al., 2023; Zhao et al., 2023; Ho et al., 2020; Song et al., 2021a). Notably, vanilla flow models require approximately 8 steps to achieve a reasonable histogram and 16–32 steps for high-quality sampling. In contrast, GMFlow (K = 64) can approximate the checkerboard in a single step and achieve high-quality sampling in 4 steps. Moreover, for vanilla flow models, samples generated by first-order solvers (DDPM, Euler) tend to concentrate toward the center, whereas those from second-order solvers (DPM++, UniPC) concentrate near the outer edges. In contrast, GMFlow samples are highly uniform. This validates GMFlow’s advantage in few-step sampling and multi-modal distribution modeling.

the expressiveness of GMFlow.

Second-order GM solvers. Fig. 3 (a) also compares our first- and second-order GM solvers across various NFEs and GM component counts. Comparing the left (first-order) and right (second-order) columns, we observe that secondorder solvers produce sharper histograms and better suppress outliers, particularly when NFE and K are small. With K = 8, the GM probabilities are sufficiently accu-

Impact of the GM component count K. Fig. 3 (a) demonstrates that increasing K significantly improves few-step sampling results, leading to sharper histograms. Notably, GM-SDE sampling with K = 1 is theoretically equivalent to DDPM with learned variance (Nichol & Dhariwal, 2021), which produces a blurrier histogram. This further highlights

GM-ODE 2 UniPC Euler GM-ODE 2 UniPC Euler GM-ODE 2 UniPC Euler GM-ODE 2 UniPC Euler

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

𝑁𝑁𝑁𝑁𝑁𝑁 = 4

𝑁𝑁𝑁𝑁𝑁𝑁 = 6

𝑁𝑁𝑁𝑁𝑁𝑁 = 8

𝑁𝑁𝑁𝑁𝑁𝑁 = 32

Figure 5. Qualitative comparisons (at best Precision) among GMFlow (GM-ODE 2) and vanilla flow model baselines (UniPC and Euler). GMFlow produces consistent results across various NFEs, whereas baselines struggle in few-step sampling, exhibiting distorted structures.

rate that the second-order solvers do not make a difference, aligning with our theoretical analysis.

|| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
<br><br>| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
<br><br>𝑁𝑁𝑁𝑁𝑁𝑁 𝑁𝑁𝑁𝑁𝑁𝑁 Best PrecisionBest FID<br><br>SDE solvers ODE solvers<br><br>(a) Best FID and best Precision vs. NFE|
|---|

Ablation studies. To validate the importance of the qˆ(x0|xt) conversion in second-order GM solvers, we conduct an ablation study by removing the conversion and directly extrapolating the x0 distributions of the last two steps, similar to DPM++ (Lu et al., 2023). As shown in Fig. 3 (b), removing this conversion introduces an overshooting bias, similar to DPM++ 2M SDE (NFE = 8), underscoring its importance in maintaining sample uniformity.

#### 4.2. ImageNet Generation

For image generation evaluation, we benchmark GMFlow against vanilla flow baselines on class-conditioned ImageNet 256×256. We train a Diffusion Transformer (DiTXL/2) (Peebles & Xie, 2023; Vaswani et al., 2017) using the flow matching objective (Eq. (3)) as the baseline, and then adapt it into GMFlow-DiT by expanding its output channels and training it with the transition loss (λ = 0.5). As illustrated in Fig. 4, GMFlow-DiT produces K weight maps and mean maps as the pixel-wise parameters {Ak,µk}. Meanwhile, a tiny two-layer MLP separately predicts s using only the time t and condition c as inputs. Empirically, we find this design to be more stable than predicting s using the main DiT. The adaptation results in only a 0.2% increase in network parameters for K = 8. During inference, we apply the orthogonal projection technique by Sadat et al. (2025) to both the baseline and GMFlow since it universally improves Precision. Additional training and inference details are provided in § A.4.

|| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
<br><br>Precision<br><br>Recall Recall<br><br>SDE solvers ODE solvers<br><br>(b) Precision–Recall curves at 𝑁𝑁𝑁𝑁𝑁𝑁 = 8|
|---|

Precision

Figure 6. (a) Comparison of the best Precision and best FID among GMFlow and vanilla flow model baselines using different solvers across varying NFEs on ImageNet. For best FID, GMFlow significantly outperforms the baselines in few-step sampling; for best Precision, GMFlow consistently excels across different NFEs. (b) Precision-Recall curves of different methods at NFE = 8. Points corresponding to the best FID and best Precision are marked on the curves. GMFlow achieves superior Precision and Recall.

Evaluation protocol. For quantitative evaluation, we adopt the standard metrics used in ADM (Dhariwal & Nichol,

- 2021), including Fr´echet Inception Distance (FID) (Heusel et al., 2017), Inception Score (IS), and Precision–Recall (Kynk¨a¨anniemi et al., 2019). These metrics are highly sen-

sitive to the classifier-free guidance (CFG) scale: while smaller CFG scales (w ≈ 1.4) often yield the best FID by balancing diversity and quality, human users generally favor

higher CFG scales (w > 3.0) for the best perceptual quality, which also leads to the best Precision. The best Recall and IS values typically occur outside these CFG ranges and are less representative of typical usage. Therefore, for a fair and complete evaluation, we sweep over the CFG scale w or the probabilistic guidance scale w˜ for each model, and report the best FID and best Precision. We also present Precision–Recall curves, illustrating the quality–diversity trade-off more comprehensively. Additionally, following Sadat et al. (2025), we report the Saturation metric at the best Precision setting to assess over-saturation.

Comparison against flow model baselines. For baselines, we test the vanilla flow model using various first-order solvers (DDPM (Ho et al., 2020), Euler (DDIM) (Song et al., 2021a)) and advanced second-order solvers (DPM++ (Lu et al., 2023), DEIS (Zhang & Chen, 2023), UniPC (Zhao et al., 2023), SA-Solver (Xue et al., 2023)). Details on adapting these solvers for flow matching are presented in § A.5. In Fig. 6 (a), we compare these baselines with our GMFlow (K = 8) model equipped with second-order GM solvers (GM-SDE/ODE 2). GMFlow consistently achieves superior Precision in both SDE and ODE sampling across various NFEs. Notably, GM-ODE 2 reaches a Precision of 0.942 in just 6 steps, while GM-SDE 2 attains a state-of-the-art Precision of 0.950 in 32 steps. For FID, GMFlow outperforms baselines in few-step settings (NFE ≤ 8) and remains competitive in many-step settings (NFE ≥ 32). Fig. 6 (b) further illustrates that the Precision-Recall curves of firstand second-order GM solvers consistently outperform those of their baseline counterparts. Qualitative comparisons are presented in Fig. 5.

Saturation assessment. Table 2 compares the Saturation metrics of different methods at their best Precision. GMFlow (K = 8) effectively reduces over-saturation, achieving Saturation levels closest to real data. Visual comparisons in Fig. 5 further support this finding. Additionally, ablating the orthogonal projection technique results in the same trend: GMFlow consistently achieves the best Saturation, while baseline methods perform even worse without orthogonal projection.

Impact of the GM component count K. In Fig. 7, we analyze the impact of the number of GM components on the metrics. Few-step sampling (NFE = 8) benefits the most from increasing GM components, particularly with the GMSDE 2 solver. The metrics generally saturate at K = 8. Beyond this point, the GM-SDE 2 Precisions decline because spectral sampling introduces larger numerical errors when more Gaussian components are used. Additionally, Table 3 presents the time-averaged negative log-likelihood (NLL) values of data under the predicted distribution qθ(x0|xt,c), showing that increasing the number of Gaussians significantly reduces NLL, suggesting its potential benefits for

- Table 2. ImageNet evaluation results at best Precision (NFE = 32). The reported Saturation values (Sadat et al., 2025) are relative to the real data statistics (Saturation=0.318).

Method

Orthogonal

projection Guidance Precision↑ Saturation

DDPM small ✓ w = 3.3 0.944 +0.032 DPM++ 2M SDE ✓ w = 2.9 0.938 +0.032 GM-SDE ✓ w˜ = 0.47 0.950 −0.019 GM-SDE 2 ✓ w˜ = 0.39 0.950 −0.019

Euler ✓ w = 3.3 0.934 +0.052 UniPC ✓ w = 3.3 0.931 +0.048 GM-ODE ✓ w˜ = 0.47 0.947 −0.024 GM-ODE 2 ✓ w˜ = 0.47 0.947 −0.024

DDPM small w = 3.3 0.939 +0.056 DPM++ 2M SDE w = 2.9 0.933 +0.054 GM-SDE w˜ = 0.27 0.943 +0.023

Euler w = 3.3 0.931 +0.064 UniPC w = 3.3 0.929 +0.060 GM-ODE w˜ = 0.27 0.933 +0.016

Best FID

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Best Precision

𝐾𝐾 𝐾𝐾

Figure 7. Best FIDs and best Precisions of GMFlow models with varying numbers of GM components K.

- Table 3. Validation NLL of ImageNet training images (latents) under different GMFlow configurations.

K = 16 +spectral

Method K = 1 K = 4 K = 8 K = 16

NLL (bits/dim) 0.346 0.263 0.242 0.224 0.173

posterior sampling applications.

Ablation studies. Table 4 presents ablation study results evaluating the impact of various design choices in our method. For GM-SDE, removing spectrum sampling (A2) leads to a noticeable degradation in FID. Replacing our probabilistic guidance with vanilla CFG (A3)—i.e., naively shifting the mean of the predicted GM—results in severe quality degradation (lower Precision) and over-saturation, which is also evident in Fig. 8. Replacing GM-SDE 2 with the second-order DPM++ SDE solver (A4) worsens both FID and Precision. Reducing the transition loss to the original KL loss (A5) degrades FID. Finally, for GM-ODE 2, directly applying ODE integration without taking sub-steps (B1) leads to significantly worse FID.

Inference time. GMFlow only alters the output layer and uses solvers based on simple arithmetic operations. As a result, it adds only 0.005 sec of overhead per step (batch size 125, A100 GPU) compared to its flow-matching counterpart,

which is minimal compared to the total inference time of 0.39 sec per step—most of which is spent on DiT.

Table 4. Ablation studies on GMFlow (K = 8, NFE = 8) using ImageNet evaluation.

Best Precision

Saturation @Best Prec.

ID Method Best FID↓

↑

### 5. Related Work

- A0 Full model (GM-SDE 2) 3.43 0.939 −0.003
- A1 A0 → GM-SDE 6.96 (+3.53) 0.938 (−0.001) +0.009
- A2 A1 w/o spec. sampling 8.98 (+2.02) 0.940 (+0.002) −0.022
- A3 A2 → Vanilla CFG 9.02 (+0.04) 0.917 (−0.023) +0.049
- A4 A0 → DPM++ 2M SDE 4.59 (+1.16) 0.912 (−0.027) −0.062
- A5 A0 → λ = 1.0 4.49 (+1.06) 0.941 (+0.002) −0.019

Prior works (Nichol & Dhariwal, 2021; Bao et al., 2022b;a) extend standard diffusion models by learning the variance of denoising distributions, which is effectively a special case of GMFlow with K = 1 and learnable s. GMS (Guo et al.,

- B0 Full model (GM-ODE 2) 2.77 0.946 −0.028
- B1 B0 w/o sub-steps 7.47 (+4.70) 0.947 (+0.001) −0.031

- 2023) further extends this approach to third-order moments and fits a bimodal GM to the moments during inference. In § B.3, we present a comparison between GMFlow (K = 2) and GMS. Conversely, Xiao et al. (2022) employ a generative adversarial network (Goodfellow et al., 2014) to capture a more expressive distribution, though adversarial diffusion typically relies on additional objectives for better quality and stability (Jolicoeur-Martineau et al., 2021; Kim et al., 2024). However, none of these methods address deterministic ODE sampling.

A growing line of work trains networks to directly predict ODE integrals or denoised samples, enabling extremely fewstep sampling (Salimans & Ho, 2022; Song et al., 2023; Yin et al., 2024b; Sauer et al., 2024). These approaches fall under the category of distillation methods, whose quality typically remains bounded by the many-step performance of standard diffusion models, unless supplemented by additional adversarial training (Kim et al., 2024; Yin et al.,

- 2024a).

- A2: Prob. guidance 𝑤𝑤 = 0.55

[Figure 54]

- A3: Vanilla CFG

𝑤𝑤 = 4.90

Figure 8. Ablation study on probabilistic guidance, comparing samples from Table 4 A2 and A3.

over-saturation. Extensive experiments demonstrated that GMFlow significantly improves few-step generation while enhancing overall sample quality. This framework lays the foundation for potential future research in both theoretical and practical directions, including applications such as posterior sampling with GMFlow priors.

Efforts to refine CFG beyond thresholding (Saharia et al.,

- 2022a) and orthogonal projection (Sadat et al., 2025) include disabling CFG in early steps (Kynk¨a¨anniemi et al., 2024), which compromises Precision, and adding Langevin corrections (Bradley & Nakkiran, 2024), which reduces efficiency.

### Limitations

To apply Gaussian mixture to high-dimensional data, we adopted pixel-wise factorization for image generation, which may not fully exploit the potential of GMFlow, leaving rooms for further development.

Beyond diffusion models, Gaussian mixtures have also been employed in other generative models. DeLiGAN (Gurumurthy et al., 2017) and GM-GAN (Ben-Yosef & Weinshall, 2018) enhance the latent expressiveness of GANs through the introduction of mixture priors. GIVT (Tschannen et al., 2024) models the output of an autoregressive Transformer as a Gaussian mixture distribution, enabling continuous data sampling and outperforming its quantization-based counterpart.

### Acknowledgments

We thank Liwen Wu, Lvmin Zhang, and Guandao Yang for discussions and feedback. Part of this work was done while Hansheng Chen was an intern at Adobe Research. This project was partially supported by Qualcomm Innovation Fellowship, ARL grant W911NF-21-2-0104, and Vannevar Bush Faculty Fellowship.

### 6. Conclusion

In this work, we introduced GMFlow, a generalization of diffusion and flow models that represents flow velocity as a Gaussian mixture, offering greater expressiveness for capturing complex multi-modal structures. We derived principled SDE/ODE solvers unique to this approach and proposed a novel probabilistic guidance technique to eliminate

### Impact Statement

This paper presents work whose goal is to advance the field of generation models. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In ICLR, 2023.

Artificial Analysis. Artificial analysis text to image model leaderboard, 2025. URL https://huggingface. co/spaces/ArtificialAnalysis/ Text-to-Image-Leaderboard. Accessed: 2025-01-25.

Bao, F., Li, C., Sun, J., Zhu, J., and Zhang, B. Estimating the optimal covariance with imperfect mean in diffusion probabilistic models. In ICML, 2022a.

Bao, F., Li, C., Zhu, J., and Zhang, B. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In ICLR, 2022b. URL https: //openreview.net/forum?id=0xiJLKH-ufZ.

Ben-Yosef, M. and Weinshall, D. Gaussian mixture generative adversarial networks for diverse datasets, and the unsupervised clustering of images, 2018. URL https://arxiv.org/abs/1808.10356.

Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V., and Rombach, R. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. URL https://arxiv.org/abs/ 2311.15127.

Bradley, A. and Nakkiran, P. Classifier-free guidance is a predictor-corrector, 2024. URL https://arxiv. org/abs/2408.09000.

Campbell, A., Benton, J., Bortoli, V. D., Rainforth, T., Deligiannidis, G., and Doucet, A. A continuous time framework for discrete denoising models. In NeurIPS, 2022.

Chen, J., YU, J., GE, C., Yao, L., Xie, E., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic textto-image synthesis. In ICLR, 2024. URL https:// openreview.net/forum?id=eAKmQPe3m1.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009. doi: 10.1109/CVPR.2009. 5206848.

Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8-bit optimizers via block-wise quantization. In ICLR, 2022.

Dhariwal, P. and Nichol, A. Q. Diffusion models beat GANs on image synthesis. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), NeurIPS, 2021. URL https://openreview.net/forum? id=AAWuCvzaVt.

Dieleman, S., Sartran, L., Roshannai, A., Savinov, N., Ganin, Y., Richemond, P. H., Doucet, A., Strudel, R., Dyer, C., Durkan, C., Hawthorne, C., Leblond, R., Grathwohl, W., and Adler, J. Continuous diffusion for categorical data, 2022. URL https://arxiv.org/abs/ 2211.15089.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., and Rombach, R. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Gao, R., Hoogeboom, E., Heek, J., Bortoli, V. D., Murphy, K. P., and Salimans, T. Diffusion meets flow matching: Two sides of the same coin, 2024. URL https:// diffusionflow.github.io/.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N., and Weinberger, K. (eds.), NeurIPS, volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips.

cc/paper_files/paper/2014/file/ 5ca3e9b122f61f8f06494c97b1afccf3-Paper. pdf.

Gu, S., Chen, D., Bao, J., Wen, F., Zhang, B., Chen, D., Yuan, L., and Guo, B. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022.

Guo, H. A., Lu, C., Bao, F., Pang, T., YAN, S., Du, C., and Li, C. Gaussian mixture solvers for diffusion models. In NeurIPS, 2023. URL https://openreview.net/ forum?id=0NuseeBuB4.

Gurumurthy, S., Sarvadevabhatla, R. K., and Babu, R. V. Deligan : Generative adversarial networks for diverse and limited data. In CVPR, 2017.

HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., and Bibi, O. Ltx-video: Realtime video latent diffusion, 2024. URL https: //arxiv.org/abs/2501.00103.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. In NeurIPS Workshop, 2021.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In NeurIPS, 2020.

Hong, W., Ding, M., Zheng, W., Liu, X., and Tang, J. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023. URL https: //openreview.net/forum?id=rB6TpjAuSRy.

Huix, T., Korba, A., Durmus, A., and Moulines, E. Theoretical guarantees for variational inference with fixedvariance mixture of gaussians. In ICML, 2024.

Jiang, D., Ku, M., Li, T., Ni, Y., Sun, S., Fan, R., and Chen, W. GenAI arena: An open evaluation platform for generative models. In NeurIPS Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum? id=0Gmi8TkUC7.

Jolicoeur-Martineau, A., Pich´e-Taillefer, R., Mitliagkas, I., and des Combes, R. T. Adversarial score matching and improved sampling for image generation. In ICLR, 2021. URL https://openreview.net/forum? id=eLfqMl3z3lq.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.

Karras, T., Aittala, M., Kynk¨a¨anniemi, T., Lehtinen, J., Aila, T., and Laine, S. Guiding a diffusion model with a bad version of itself. In NeurIPS, 2024. URL https: //openreview.net/forum?id=bg6fVPVs3s.

Kim, D., Lai, C.-H., Liao, W.-H., Murata, N., Takida, Y., Uesaka, T., He, Y., Mitsufuji, Y., and Ermon, S. Consistency trajectory models: Learning probability flow ODE trajectory of diffusion. In ICLR, 2024. URL https:

//openreview.net/forum?id=ymjI8feDTD. Knothe, H. Contributions to the theory of convex bodies.

Michigan Mathematical Journal, 4(1):39–52, 1957.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., Wu, K., Lin, Q., Yuan, J., Long, Y., Wang, A., Wang, A., Li, C., Huang, D., Yang, F., Tan, H., Wang, H., Song, J., Bai, J., Wu, J., Xue, J., Wang, J., Wang, K., Liu, M., Li, P., Li, S., Wang, W., Yu, W., Deng, X., Li, Y., Chen, Y., Cui, Y., Peng, Y., Yu, Z., He, Z., Xu, Z., Zhou, Z., Xu, Z., Tao, Y., Lu, Q., Liu, S., Zhou, D., Wang, H., Yang, Y., Wang, D., Liu, Y., Jiang, J., and Zhong, C. Hunyuanvideo: A systematic framework for large video generative models, 2025. URL https://arxiv.org/abs/2412.03603.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images, 2009.

Kullback, S. and Leibler, R. A. On information and sufficiency. The Annals of Mathematical Statistics, 22(1): 79–86, 1951.

Kynk¨a¨anniemi, T., Karras, T., Laine, S., Lehtinen, J., and Aila, T. Improved precision and recall metric for assessing generative models. In NeurIPS, 2019.

Kynk¨a¨anniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., and Lehtinen, J. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. In NeurIPS, 2024.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In ICLR, 2023. URL https://openreview.net/forum? id=PqvMRDCJT9t.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In ICLR, 2019.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), NeurIPS, 2022. URL https://openreview.net/forum? id=2uAaGwlP_V.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpmsolver++: Fast solver for guided sampling of diffusion probabilistic models, 2023. URL https://arxiv. org/abs/2211.01095.

Nichol, A. and Dhariwal, P. Improved denoising diffusion probabilistic models. In ICML, 2021.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In ICCV, 2023.

Po, R., Yifan, W., Golyanik, V., Aberman, K., Barron, J. T., Bermano, A. H., Chan, E. R., Dekel, T., Holynski, A., Kanazawa, A., Liu, C. K., Liu, L., Mildenhall, B., Nießner, M., Ommer, B., Theobalt, C., Wonka, P., and Wetzstein, G. State of the art on diffusion models for visual computing. In Eurographics STAR, 2024.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. URL https://openreview.

net/forum?id=di52zR8xgf.

Rezende, D. and Mohamed, S. Variational inference with normalizing flows. In Bach, F. and Blei, D. (eds.), ICML,

volume 37 of Proceedings of Machine Learning Research, pp. 1530–1538, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/ rezende15.html.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234–241, 2015.

Rosenblatt, M. Remarks on a multivariate transformation. The Annals of Mathematical Statistics, 23(3):470–472, 1952.

Sadat, S., Hilliges, O., and Weber, R. M. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In ICLR, 2025.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan,

- B., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. Photorealistic text-to-image diffusion models with deep language understanding. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), NeurIPS, volume 35, 2022a.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan,

- B., Salimans, T., et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022b.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.

Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. In ECCV, pp. 87–103, 2024.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pp. 2256–2265, 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In ICLR, 2021a.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In ICLR, 2021b.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In ICML, 2023.

Tschannen, M., Eastwood, C., and Mentzer, F. Givt: Generative infinite-vocabulary transformers. In ECCV, pp. 292–309, 2024.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), NeurIPS, volume 30. Curran Associates, Inc., 2017.

von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman, W., Xu, Y., Liu, S., and Wolf, T. Diffusers: State-of-the-art diffusion models. https://github.

com/huggingface/diffusers, 2022.

Xiao, Z., Kreis, K., and Vahdat, A. Tackling the generative learning trilemma with denoising diffusion GANs. In ICLR, 2022. URL https://openreview.net/ forum?id=JprM0p-q0Co.

Xue, S., Yi, M., Luo, W., Zhang, S., Sun, J., Li, Z., and Ma, Z.-M. SA-solver: Stochastic adams solver for fast sampling of diffusion models. In NeurIPS, 2023. URL https://openreview.net/forum? id=f6a9XVFYIo.

Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., and Yang, M.-H. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4), November 2023. ISSN 0360-0300. doi: 10.1145/3626235. URL https:// doi.org/10.1145/3626235.

Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, W. T. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024a. URL https://openreview.net/forum? id=tQukGCDaNT.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In CVPR, 2024b.

Zhang, Q. and Chen, Y. Fast sampling of diffusion models with exponential integrator. In ICLR, 2023. URL https: //openreview.net/forum?id=Loek7hfb46P.

Zhao, W., Bai, L., Rao, Y., Zhou, J., and Lu, J. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. In NeurIPS, 2023.

### A. Additional Technical Details

#### A.1. Details on Gaussian Surrogates

For a GM distribution of the form Kk=1 AkN(µk,s2I), we approximate it with an isotropic Gaussian surrogate

N(µ′,s′2I) by matching the first two moments. The surrogate mean is computed as:

µ′ =

K

Akµk. (11)

k=1

To determine the surrogate variance, we equate the trace of the GM’s covariance matrix (i.e., the GM’s total variance) with the trace of the surrogate covariance, which yields:

1 D

s′2 =

K

Ak∥µk − µ′∥2 + s2. (12)

k=1

#### A.2. Details on Second-Order GM Solvers

For GM extrapolation, we follow the method in § 3.2 and fit the isotropic Gaussian surrogates N(x0;µ+,s2+I) :≈ qθ(x0|xt), N(x0;µ−,s2−I) :≈ qˆ(x0|xt). We then define a Gaussian mask:

2

N x0;µ+ + ∆µ, s2+ − ∥∆µ∥

D I N(x0;µ+,s2+I)

, (13)

ρ(x0) =

+−µ−

where ∆µ = µ

2 , so that µ+ + ∆µ represents the extrapolated mean at the next midpoint. The final extrapolated GM is obtained via the reweighting formulation:

ρ(x0) Z

qθ(x0|xt). (14)

qext(x0|xt) =

We then feed qext(x0|xt) to the first-order GM-SDE or ODE solver as a substitution for qθ(x0|xt).

Empirically, we observe that second-order GM solvers using the above formulation slightly underperform with higher guidance scales, which is a common issue with multistep solvers (Lu et al., 2023; Zhao et al., 2023). To address this, we rescale the mean difference ∆µ by an empirical

2+ca)s2c

factor max(0,1 − (w˜

∆t2 ) (with the hyperparameter ca = 0.005), so that a high w˜ practically disables multistep extrapolation.

#### A.3. Details on Spectral Sampling

Due to pixel-wise factorization, image generation under GM-SDE solvers performs the sampling step xˆ0 ∼ qθ(x0|xt) independently for each pixel, neglecting spatial correlations. Spectral sampling addresses this by establishing an invertible mapping between a frequency-space

- Algorithm 1: Complete GMFlow training scheme.

Input: Data distribution p(x0, c), transition ratio λ Output: Network params θ

- 1 Initialize network params θ
- 2 for sample {x0, c} ∼ p(x0, c) do

- 3 if use logit-normal then

- 4 Sample t ∼ LogitNormal(0, 1)

- 5 else

- 6 Sample t ∼ U(0, 1)

- 7 ∆t ← λt
- 8 Sample xt−∆t ∼ p(xt−∆t|x0)
- 9 Sample xt ∼ p(xt|xt−∆t)
- 10 Predict GM params in qθ(xt−∆t|xt, c) // Eq. (9)
- 11 Ltrans ← − log qθ(xt−∆t|xt, c)
- 12 Predict magnitude spectrum sF
- 13 Lspec ← − log N vec(zr); 0, diag(sF)2 // Eq. (20)
- 14 Backpropagate and update θ

- Algorithm 2: Complete GMFlow sampling scheme.

Input: Steps NFE, sub-steps n, guidance scale w˜,

condition c, network params θ Output: x0

- 1 t ← 1, ∆t ← NFE1 , x1 ∼ N(0, I), Cache ← {}

- 2 while t > 0 do

- 3 Predict GM params in qθ(x0|xt, c)
- 4 if w˜ > 0 then

- 5 Predict GM params in qθ(x0|xt)
- 6 Compute qw(x0|xt, c) // Eq. (7)
- 7 qθ(x0|xt, c) param← qw(x0|xt, c)

- 8 if use 2nd-order then

- 9 if Cache ̸= {} then

- 10 Compute qˆ(x0|xt, c) from Cache // Eq. (10)
- 11 Cache ← {xt, qθ(x0|xt, c)}
- 12 Compute qext(x0|xt, c) // Eq. (14)
- 13 qθ(x0|xt, c) param← qext(x0|xt, c)

- 14 else

- 15 Cache ← {xt, qθ(x0|xt, c)}

- 16 if use GM-SDE then

- 17 if use spectral sampling then

- 18 Predict magnitude spectrum sF
- 19 Sample xˆ0 ∼ qθS(x0|xt, c) // Eq. (19)

- 20 else

- 21 Sample xˆ0 ∼ qθ(x0|xt, c)

- 22 Sample xt−∆t ∼ N(xt−∆t; c1xt + c2xˆ0, c3I)

- 23 else if use GM-ODE then

- 24 τ ← t, h ← ∆t/n
- 25 while τ > t − ∆t do

- 26 Compute qˆ(x0|xτ, c) // Eq. (10)
- 27 xτ−h ← xτ − hEx0∼qˆ(x0|xτ,c)[xτσ−x0

τ

]

- 28 τ ← τ − h

- 29 t ← t − ∆t

distribution (power spectrum) and the pixel-space GM distribution. During training, the power spectrum is optimized via

𝒔𝒔F∘2

Pixel-wise {𝝁𝝁x𝑘𝑘}

Pixel-wise mean ∑𝑘𝑘 𝐴𝐴𝑘𝑘𝝁𝝁x𝑘𝑘

Var of pixel means

Compute statistics

𝐷𝐷

𝐾𝐾 × 𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

[Figure 55]

1

𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

Pixel-wise {𝐴𝐴𝑘𝑘}

2-Layer MLP

Pixel-wise var 𝐶𝐶1 ∑𝑘𝑘 𝐴𝐴𝑘𝑘 𝝁𝝁x𝑘𝑘 − ∑𝑘𝑘 𝐴𝐴𝑘𝑘𝝁𝝁x𝑘𝑘 2 + 𝑠𝑠x2

Softmax

Stop grad

Mean of pixel vars

𝐾𝐾 × 1 × 𝐻𝐻 × 𝑊𝑊

1

1 × 𝐻𝐻 × 𝑊𝑊

𝑠𝑠x

𝒙𝒙0-based GM output

Viewed as

𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

𝜻𝜻 Convert to 𝒛𝒛 and run iFFT KR transport 𝐶𝐶 ×𝐻𝐻 ×𝑊𝑊

Reshape

Sample from 𝒩𝒩 𝟎𝟎,diag 𝒔𝒔F 2

𝒛𝒛r

𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

Gaussian random field

𝒙𝒙0

𝐶𝐶 × 𝐻𝐻 × 𝑊𝑊

Figure 9. The optional spectral sampling pipeline used during inference. The spectrum MLP takes the statistics of pixel-wise GMs as input, and predicts the power spectrum s◦F2. Alternatively, if probabilistic guidance or second-order GM solvers are employed, one can re-use the mean and variance from the numerator in the Gaussian mask to compute the input statistics, which serves as a good approximation.

a likelihood loss, while during inference, frequency-space samples are transformed back into pixel space to introduce spatial correlation.

We establish an invertible mapping using two building blocks: the fast Fourier transform (FFT) and Knothe– Rosenblatt (KR) transport (Knothe, 1957; Rosenblatt, 1952). The FFT bridges a frequency-space Gaussian distribution and a pixel-space Gaussian distribution, while KR transport bridges each per-pixel Gaussian to its corresponding Gaussian mixture (GM) distribution.

During training, given a real image x0 ∈ RC×H×W and the factorized denoising distribution qθ(x0|xt) =

H,W i=1,j=1

K k=1 Ak,ijN(x0,ij|µk,ij,s2I), KR transport

defines an invertible mapping for each pixel:

##### Tij : x0,ij  → ζij, (15)

such that each ζij can be viewed as a standard Gaussian sample. We then assemble all ζij into a tensor ζ ∈ RC×H×W and apply a forward 2D FFT with orthogonal normalization, yielding the complex frequency representation z = FFT(ζ) ∈ CC×H×W. Since z is Hermitian symmetric, we can derive a real-valued representation zr while preserving invertibility:

##### zr = Re[z] + Im[z]. (16)

Finally, we impose a zero-mean Gaussian prior on zr, given by N vec(zr);0,diag(sF)2 , where sF ∈ RD+ represents the magnitude spectrum. To dynamically model sF ∈ RD+, we use a tiny two-layer MLP, which takes the mean of perpixel GM variances and the variance of per-pixel GM means as input, and outputs the power spectrum s◦F2 with a softmax activation, where (·)◦2 stands for element-wise square. With this invertible mapping and the spectral Gaussian prior, we can derive the model’s spectrum-enhanced denoising PDF using the change of variables technique in a similar way to normalizing flow models (Rezende & Mohamed, 2015),

which can be expressed as:

∂vec(zr) ∂x0 N vec(zr);0,diag(sF)2 ,

qθS(x0|xt) = det

(17) where the absolute determinant can be easily derived using the properties of FFT and KR transport:

∂vec(zr) ∂x0

∂vec(zr) ∂vec(ζ) · det

∂vec(ζ) ∂x0

det

= det

qθ(x0|xt) N(vec(ζ);0,I)

= 1 ·

qθ(x0|xt) N(vec(zr);0,I)

. (18)

=

Substituting Eq. (18) into Eq. (17), we obtain:

qθS(x0|xt) = qθ(x0|xt)N vec(zr);0,diag(sF)2 N(vec(zr);0,I)

. (19)

With the derived PDF, the entire model can be trained by minimizing the negative log-likelihood (NLL) of data samples under the PDF, which inherently includes the GM KL loss (Eq. (5) and (6)) as one of its terms. Therefore, the total loss consists of the GM KL loss (replaced with the transition loss in practice) and an additional spectral loss, defined as:

0,xt −log N vec(zr);0,diag(sF)2 N(vec(zr);0,I)

Lspec = Et,x

- 1

- 2

diag(sF)−1 − I vec(zr) 2

= Et,x

0,xt

+ log det(diag(sF)) . (20)

In practice, we stop the gradient flow through KR transport to prevent spectral learning from influencing the main GMFlow model.

During inference, we employ the spectral sampling pipeline illustrated in Fig. 9.

w/ sub-steps

w/o sub-steps

w/ sub-steps

w/o sub-steps

#### A.4. ImageNet Experiment Details.

[Figure 56]

[Figure 57]

We train both the baseline and GMFlow-DiT on ImageNet 256×256 with a batch size of 4096 images across 16 A100 GPUs, using a total training schedule of 200K iterations. We adopt the 8-bit AdamW (Dettmers et al., 2022; Loshchilov & Hutter, 2019) optimizer with a fixed learning rate of 0.0002. Following Stable Diffusion 3 (Esser et al., 2024), both models sample t from a logit-normal distribution during training (Algorithm 1), which accelerates convergence.

𝑁𝑁𝑁𝑁𝑁𝑁 = 4

𝑁𝑁𝑁𝑁𝑁𝑁 = 6

While we set the transition ratio to λ = 0.5 in the main experiments, the results in Fig. 7 and Table 3 are based on an earlier design iteration that uses randomly sampled transition ratios λ ∼ LogitNormal(0,1). This setting slightly increases Precision and decreases Recall, though the overall differences remain minor.

𝑁𝑁𝑁𝑁𝑁𝑁 = 8

𝑁𝑁𝑁𝑁𝑁𝑁 = 32

We densely evaluate the models across different guidance scales to identify the optimal FID and Precision. For vanilla CFG, we use guidance scales w from the set {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.1, 2.3, 2.6, 2.9, 3.3, 3.7, 4.3, 4.9,

Figure 10. Qualitative comparison (at best Precision) from the ablation study on GM-ODE sub-steps. Without sub-steps (forcing n = 1), few-step sampling tends to produce over-smoothed textures and poor detail.

###### 5.7, 6.5}; for probabilistic guidance, we use probabilistic guidance scales w˜ from the set {0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19, 0.23, 0.27, 0.33, 0.39, 0.47, 0.55, 0.65, 0.75}. In practice (Algorithm 2), we

implement probabilistic guidance on x0-based GMs, which is equivalent to guidance on u.

For DPM++ (Lu et al., 2023), DEIS (Zhang & Chen, 2023), UniPC (Zhao et al., 2023), and SA-Solver (Xue et al., 2023), we use their Diffusers implementations (von Platen et al., 2022) and rescale their noise schedules to match the flow matching noise schedule. This approach is similar to how the EDM Euler solver (Karras et al., 2022) rescales a variance-preserving diffusion model into a variance-exploding one at test time.

For inference with GM-ODE solvers, we generally set the number of sub-steps to n = max NFE 128 ,2 , which performs well when NFE ≥ 8. For the exception when NFE = 4 or 6, we observe that reducing n to 8 yields better performance. In Table 3, the time-averaged NLL values are computed on 50K samples from the training dataset using the following equation:

1 D

### B. Additional Experiment Results

0,xt[−log2 qθ(x0|xt)], (21)

Et,x

NLL =

#### B.1. Ablation Study on GM-ODE Sub-Steps

where t ∼ U(0,1) and D = C × H × W. This is equivalent to the original training loss (λ = 1) with a uniform time sampling scheme. When spectral prior is enabled, we replace qθ(x0|xt) with qθS(x0|xt) (Eq. (19)).

In addition to the results in Table 4 (B0 and B1), Fig. 10 presents additional qualitative results from the ablation study on GM-ODE sub-steps, using K = 8 and the GM-ODE 2 solver. As shown in the figure, sub-steps are essential for producing detailed textures when NFE < 8.

#### A.5. Adapting Diffusion Solvers for Flow Matching

For DDPM solvers (Ho et al., 2020), we implement them as special cases of GM-SDE solver with K = 1. The original DDPM solvers include a large variance variant (β) and a small variance variant (β˜). These are equivalent to setting s = √ 1

and s = 0, respectively.

α2t+σt2

For DDIM solvers (Song et al., 2021a), its stochastic variant (η = 1) is equivalent to DDPM with small variance, whereas its deterministic variant (η = 0) is equivalent to Euler solver in flow matching models.

#### B.2. Additional Qualitative Comparison

Fig. 11 presents a comparison among uncurated random samples (conditioned on random class labels) from GMFlow and vanilla flow matching baselines, showcasing their respective best results under the many-step setting. Overall, the images generated by GMFlow exhibit more natural color saturation and, in some cases, improved structural coherence. These observations are consistent with the quantitative results shown in Fig. 6 and Table 2.

Table 5. FIDs on CIFAR-10 unconditional image generation. Competitor results are sourced from Guo et al. (2023).

NFE 10 25 50 100

DDPM large 205.31 84.71 37.35 14.81 DDPM small 34.76 16.18 11.11 8.38 SN-DDPM 16.33 6.05 4.19 3.83 GMS 13.80 5.48 4.00 3.46 GM-SDE 2 9.11 4.16 3.79 3.76

- B.3. Comparison with Moment Matching Methods

Table 5 presents a quantitative comparison among GMFlow (K = 2), GMS (Guo et al., 2023), SN-DDPM (Bao et al., 2022a), and DDPM (Ho et al., 2020) for CIFAR-10 (Krizhevsky et al., 2009) unconditional image generation using SDE sampling. The GMFlow model is trained from scratch using the same U-Net architecture (Ronneberger et al., 2015; Ho et al., 2020) as its competitors, but with modified output layers as in GM-DiT. The results demonstrate that GMFlow significantly outperforms its competitors in few-step sampling.

- C. Additional Theoretical Analysis

- C.1. Details on the SDE and ODE Background

For notation references, this subsection briefly recaps the SDE and ODE formulation of diffusion models by Song et al. (2021b).

The forward-time SDE of the diffusion process can be derived by taking an infinitesimal step size ∆t when applying the transition Gaussian kernel in Eq. (1), which yields:

dx = ftxt dt + gt dwt (22) where ft = α1

dt ,gt = dβd∆t,∆tt ∆t=0, and wt is the standard Wiener process. The corresponding reverse-time SDE is:

dαt

t

dxt = ftxt − gt2st(xt) dt + gt dw¯t, (23) with the score function st(xt) := ∇xt

log p(xt) = Ex

αtx0−xt

σt2 and the reverse-time standard Wiener process w¯. Using the Fokker–Planck equation, we can prove that the time evolution of the PDF p(xt) described by the SDEs is equivalent to that described by an ODE:

0∼p(x0|xt)

- 1

- 2

gt2st(xt) dt, (24) which maps the noise xt to a deterministic data point x0.

dxt = ftxt −

#### C.2. Proof of Theorem 3.1

Proof. Let p(u) be the PDF of an arbitrary distribution on RD, and {Σk}Kk=1 be a set of arbitrary D × D symmetric

positive definite matrices. We aim to show that if

K

{a∗k,µ∗k} = arg min {ak,µk}

Eu∼p(u) −log

AkN(u;µk,Σk) ,

k=1

(25) with ak ∈ R, µk ∈ RD, Ak = expa

k=1 expak , then the mean alignment property holds:

k K

K

A∗kµ∗k = Eu∼p(u)[u], (26)

k=1

∗ k

with A∗k = expa

k=1 expa∗k . For brevity, we define

K

qk(u) := A∗kN(u;µ∗k,Σk), (27) q(u) :=

K

qk(u). (28)

k=1

Since {a∗k,µ∗k} minimizes the objective, the first-order optimality conditions imply

∂Eu∼p(u)[−log q(u)] ∂a∗

k

∂ − log q(u) ∂a∗

= Eu∼p(u)

k

qk(u) q(u)

= Eu∼p(u) A∗k −

qk(u) q(u)

= A∗k − Eu∼p(u)

= 0, (29) and

∂Eu∼p(u)[−log q(u)] ∂µ∗

k

∂ − log q(u) ∂µ∗

= Eu∼p(u)

k

qk(u) q(u)

- 1

- 2

Σ−

k (µ∗k − u)

= Eu∼p(u)

qk(u) q(u)

qk(u) q(u) − Eu∼p(u)

- 1

- 2

= Σ−

k µ∗kEu∼p(u)

#### u

= 0. (30) From the above, it follows that

µ∗kEu∼p(u)

A∗k = Eu∼p(u)

qk(u) q(u)

= Eu∼p(u)

qk(u) q(u)

, (31)

qk(u) q(u)

u . (32)

Substituting Eq. (31) into Eq. (32), we have

A∗kµ∗k = Eu∼p(u)

qk(u) q(u)

u . (33)

Summing over all k, we conclude that

K

A∗kµ∗k = Eu∼p(u)

k=1

K k=1 qk(u)

u = Eu∼p(u)[u].

q(u)

(34)

| |
|---|

#### C.3. Derivation of the Reverse Transition Distribution

Given the ground truth data distribution p(x0) and the forward diffusion Gaussian p(xt|x0) = N(xt;αtx0,σt2I), the ground truth denoising distribution p(x0|xt) can be derived using Bayes’ theorem:

p(x0)p(xt|x0) p(xt)

. (35)

p(x0|xt) =

Conversely, let us assume that we know the ground truth denoising distribution p(x0|xt). By rearranging Eq. (35), we can derive the data distribution as:

p(xt)p(x0|xt) p(xt|x0)

. (36)

p(x0) =

With the data distribution, we can apply the forward diffusion process and derive the noisy data distribution at t −∆t:

p(xt−∆t|x0)p(x0)dx0

p(xt−∆t) =

RD

p(xt)p(x0|xt) p(xt|x0)

dx0. (37)

p(xt−∆t|x0)

=

RD

Finally, the reverse transition distribution p(xt−∆t|xt) can be derived using Bayes’ theorem:

p(xt−∆t|xt)

p(xt−∆t)p(xt|xt−∆t) p(xt)

=

p(xt|xt−∆t)p(xt−∆t|x0) p(xt|x0)

p(x0|xt)dx0, (38)

=

RD

where p(xt|xt−∆t) is the forward transition Gaussian defined in Eq. (1). The term p(x

t|xt−∆t)p(xt−∆t|x0)

p(xt|x0) can be fused into one Gaussian PDF using the conflation operation described in § D.1, which yields:

p(xt−∆t|xt,x0) = N(xt−∆t;c1xt + c2x0,c3I), (39)

with the coefficients c1,c2,c3 defined in § 3.3. Therefore, Eq. (38) can be simplified into:

p(xt−∆t|xt) =

p(xt−∆t|xt,x0)p(x0|xt)dx0, (40)

RD

which is a convolution between p(xt−∆t|xt,x0) and

- p(x0|xt). Sampling from p(xt−∆t|xt) can therefore be simulated by first sampling x0 ∼ p(x0|xt) and then sampling xt−∆t ∼ p(xt−∆t|xt,x0). In general, given any empirical denoising distribution
- qθ(x0|xt), we can perform stochastic denoising sampling by substituting p(x0|xt) ≈ qθ(x0|xt) into the above sampling process. Specifically for GMFlow, qθ(x0|xt) is a GM PDF. § D.3 shows that the convolution of a GM and a Gaussian is also a GM with analytically derived parameters. Therefore, GMFlow models the reverse transition

- distribution as a GM qθ(xt−∆t|xt), given by Eq. (9).

- C.4. Additional Analysis of the Reverse Transition GM

By computing the first-order Taylor approximation of Eq. (9) w.r.t. ∆t, we can re-write the GM reverse transition distribution as:

qθ(xt−∆t|xt)

=

K

k=1

AkN xt−∆t;xt − ftxt − gt2µsk ∆t,gt2∆tI

+ O(∆t)

= N xt−∆t;xt − ftxt − gt2

K

k=1

Akµsk ∆t,gt2∆tI

+ O(∆t), (41)

where µsk := −σ1

t

(xt + αtµk). The term Kk=1 Akµsk represents the model’s prediction of the score function st(xt), and is a linear transformation of the predicted mean velocity Kk=1 Akµk. Recursively sampling xt−∆t using the first-order approximation in Eq. (41) is equivalent to solving the reverse-time SDE in Eq. (23) (with predicted score) using the Euler–Maruyama method, a simple firstorder SDE solver.

Eq. (41) is consistent with the standard diffusion model assumption that the reverse transition distribution is approximately Gaussian for small ∆t. Even with GM parameterization, SDE sampling is primarily influenced by the mean prediction when ∆t is sufficiently small. This underscores the significance of mean alignment not only for ODE solving but also for SDE solving.

- C.5. Derivation of qˆ(x0|xτ)

Let us assume that we know the ground truth denoising

- distribution at xt, i.e., p(x0|xt). Eq. (36) shows us how to derive the data distribution p(x0) from p(x0|xt). From p(x0), the denoising distribution at xτ can be derived using

Bayes’ theorem:

p(x0)p(xτ|x0) p(xτ)

p(x0|xτ) =

p(xt)p(x0|xt)p(xτ|x0) p(xt|x0)p(xτ)

=

p(xτ|x0) Z · p(xt|x0)

p(x0|xt), (42) where Z = p(x

=

τ)

p(xt) is a normalization factor. Substituting p(x0|xt) ≈ qθ(x0|xt) into the above equation yields the denoising distribution conversion rule in Eq. (10). The conversion involves conflations of Gaussians and a GM, which can be approached analytically as shown in § D.

### D. Gaussian Mixture Math References

#### D.1. Conflation of Two Gaussians

Let p1(x) = N(x;µ1,Σ1), p2(x) = N(x;µ2,Σ2) be two multivariate Gaussian PDFs, their powered conflation is defined as:

pγ

1 (x)pγ

1

2

2 (x) Z

p′(x) =

, (43)

where γ1,γ2 ∈ R, assuming γ1Σ−1 1 + γ2Σ−2 1 is positive definite, and Z = R

pγ

1 (x)pγ

2 (x)dx is a normalization factor. It’s easy to prove that, p′(x) can also be expressed as a Gaussian N(x;µ′,Σ′), with the new parameters:

1

2

D

Σ′ = γ1Σ−1 1 + γ2Σ−2 1 −1, (44) µ′ = Σ′(γ1Σ−1 1µ1 + γ2Σ−2 1µ2). (45)

#### D.2. Conflation of a Gaussian and a GM

Let p1(x) = N(x;µ,Σ) be a Gaussian PDF, and p2(x) = K k=1 AkN(x;µk,Σk) be a GM PDF, where Ak = exp ak

k=1 expak with logit ak. Their conflation is defined as: p′(x) =

K

p1(x)p2(x) Z

, (46)

where Z = R

p1(x)p2(x)dx is a normalization factor. Since the GM PDF is a sum of Gaussians, the conflation of a Gaussian and a GM expands to a sum of conflations of Gaussians, which simplifies to a sum of Gaussians. Therefore, p′(x) can also be expressed as a GM

D

K k=1 A′kN(x;µ′k,Σ′k), with the new parameters:

Σ′k = (Σ−1 + Σ−k 1)−1, (47) µ′k = Σ′k(Σ−1µ + Σ−k 1µk), (48)

expa′k K k=1 expa′

A′k =

, (49)

k

where the new logit a′k is given by:

- 1

- 2

(µ − µk)T(Σ + Σk)−1(µ − µk). (50)

a′k = ak −

#### D.3. Convolution of a Gaussian and a GM

Let p(x1|x2) = N(x1;µ + cx2,Σ) be a conditional Gaussian PDF, where c ∈ R is a linear coefficient, and

p(x2) = Kk=1 AkN(x2;µk,Σk) be a GM PDF, where Ak = expa

k=1 expak with logit ak. Their convolution yields the marginal PDF of x1:

k K

p(x1) =

p(x1|x2)p(x2)dx2. (51)

RD

Since the GM PDF is a sum of Gaussians, the convolution of a Gaussian and a GM expands to a sum of convolution of Gaussians, which simplifies to a sum of Gaussians. Therefore p(x1) can also be expressed as a GM

K k=1 AkN(x;µ′k,Σ′k), with the new parameters:

Σ′k = Σ + c2Σk, (52) µ′k = µ + cµk. (53)

### E. Notation

Table 6. A summary of frequently used notations. Notation Description

D Data dimension.

t ∈ [0,T] Diffusion time. Flow matching models define T := 1. αt Noise schedule coefficient. Flow matching models define αt := 1 − t. σt Noise schedule coefficient. Flow matching models define σt := t.

ϵ Standard Gaussian noise. x,x0 ∈ RD Data.

xt := αtx + σtϵ Noisy data.

p(x),p(x0) PDF of data (ground truth). p(xt) PDF of noisy data (ground truth).

p(xt|x0) = N(xt;αtx0,σt2I) PDF of forward diffusion distribution. p(xt|xt−∆t) = N(xt; α

αt−∆tx0,βt,∆tI) PDF of forward transition distribution. βt,∆t = σt2 − α

t

2 t

α2t−∆tσt2−∆t Variance of forward transition distribution.

- p(x0|xt) = p(x

0)p(xt|x0)

p(xt) PDF of reverse denoising distribution (ground truth).

- p(xt−∆t|xt) = p(x

t−∆t)p(xt|xt−∆t)

p(xt) PDF of reverse transition distribution (ground truth). θ Neural network parameters.

- qθ(x0|xt) PDF of reverse denoising distribution, predicted by a network.

- qθ(xt−∆t|xt) PDF of reverse transition distribution, predicted by a network.

t−x0

u := x

σt Random flow velocity. Ex

0∼p(x0|xt)[u] Mean flow velocity at xt (ground truth).

- p(u|xt) PDF of velocity distribution at xt (ground truth), derived from p(x0|xt).
- qθ(u|xt) PDF of velocity distribution at xt, predicted by a network. µθ(xt) Mean flow velocity at xt, predicted by a network.

K Number of Gaussian components. k Index of Gaussian components.

Ak := expa

k=1 expak . Mixture weight of the k-th Gaussian component. µk Mean of the k-th Gaussian component. Σk Covariance of the k-th Gaussian component. GMFlow defines Σk := s2I. ak Pre-activation logit.

k K

s Shared standard deviation.

µxk := xt − σtµk Mean of the k-th Gaussian component after u-to-x0 conversion. sx := σts Shared standard deviation after u-to-x0 conversion. c Condition.

- w ∈ [1,+∞) CFG scale. w˜ ∈ [0,1) Probabilistic guidance scale.
- xˆ Intermediate sample of denoised data.

NFE Number of function (network) evaluations, a.k.a. sampling steps. n Number of sub-steps in GM-ODE solvers. τ Sub-step diffusion time in GM-ODE solvers.

qˆ(x0|xτ) PDF of reverse denoising distribution, derived from qθ(x0|xt) (Eq. (10)). qˆ(x0|xt) PDF of reverse denoising distribution, derived from qθ(x0|xt+∆t).

λ ∈ (0,1] Transition ratio.

GM-SDE 2 DPM++ 2M SDE SA-Solver DDPM small DDPM large GM-ODE 2 UniPC DPM++ 2M DEIS Euler

[Figure 58]

[Figure 59]

Figure 11. Uncurated samples (at best Precision, NFE = 32) from GMFlow and vanilla flow matching baselines using different solvers. The images generated by GMFlow exhibit more natural color saturation and, in some cases, improved structural coherence.

