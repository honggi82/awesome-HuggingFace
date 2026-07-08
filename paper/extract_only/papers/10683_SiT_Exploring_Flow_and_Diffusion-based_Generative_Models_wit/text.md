# arXiv:2401.08740v2[cs.CV]23Sep2024

## SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden†, and Saining Xie†

New York University

Abstract. We present Scalable Interpolant Transformers (SiT), a family of generative models built on the backbone of Diffusion Transformers (DiT). The interpolant framework, which allows for connecting two distributions in a more flexible way than standard diffusion models, makes possible a modular study of various design choices impacting generative models built on dynamical transport: learning in discrete or continuous time, the objective function, the interpolant that connects the distributions, and deterministic or stochastic sampling. By carefully introducing the above ingredients, SiT surpasses DiT uniformly across model sizes on the conditional ImageNet 256 × 256 and 512 × 512 benchmark using the exact same model structure, number of parameters, and GFLOPs. By exploring various diffusion coefficients, which can be tuned separately from learning, SiT achieves an FID-50K score of 2.06 and 2.62, respectively. Code is available here: https://github.com/willisma/SiT

### 1 Introduction

Contemporary success in image generation has come from a combination of algorithmic advances, improvements in model architecture, and progress in scaling neural network models and data. State-of-the-art diffusion models [25,53] proceed by incrementally transforming data into Gaussian noise as prescribed by an iterative stochastic process, which can be specified either in discrete or continuous time. At an abstract level, this corruption process can be viewed as defining a time-dependent distribution that is iteratively smoothed from the original data distribution into a standard normal distribution. Diffusion models learn to reverse this corruption process and push Gaussian noise backwards along this connection to obtain data samples. The objects learned to perform this transformation conventionally predict either the noise in the corruption process [25] or the score of the distribution that connects the data and the Gaussian [64], though alternatives of these choices exist [28,56]. While diffusion models originally represented these objects with a U-Net architecture [25,54], recent work has highlighted that architectural advances in vision such as the

† Equal advising.

Table 1: Scalable Interpolant Transformers. We systematically vary the following aspects of a generative model: time discretization, model prediction, interpolant, and sampler. The resulting Scalable Interpolant Transformer (SiT) model, under identical training compute, consistently outperforms the Diffusion Transformer (DiT) in generating 256×256 ImageNet images. All models employ a patch size of 2. In this work, we ask the question: What is the source of the performance gain?

Model Params(M) Training Steps FID ↓

DiT-S 33 400K 68.4 SiT-S 33 400K 57.6

DiT-B 130 400K 43.5 SiT-B 130 400K 33.0

DiT-L 458 400K 23.3 SiT-L 458 400K 18.8

DiT-XL 675 400K 19.5 SiT-XL 675 400K 17.2

DiT-XL 675 7M 9.6 SiT-XL 675 7M 8.3

DiT-XL (cfg=1.5) 675 7M 2.27 SiT-XL (cfg=1.5) 675 7M 2.06

Vision Transformer (ViT) [21] can be incorporated into the standard diffusion model pipeline to improve performance [50].

Orthogonally, significant research effort has gone into exploring the structure of the noising process, which has been shown to lead to performance benefits [33, 36,37,60]. Yet, many of these efforts do not move past the notion of passing data through a diffusion process with an equilibrium distribution, which is a restricted type of connection between the data and the Gaussian. Recently-introduced stochastic interpolants [2] lift this constraint and introduce more flexibility in the noise-data connection. In this paper, we systematically explore the effect of this flexibility on performance in large scale image generation.

Intuitively, we expect that the difficulty of the learning problem can be related to both the specific connection chosen and the object that is learned. Our aim is to clarify these design choices, so as to simplify the learning problem and thereby improve performance. To understand where potential benefits arise in the learning problem, we start with Denoising Diffusion Probabilistic Models (DDPMs) and sweep through adaptations of: (i) which object to learn, and (ii) which interpolant to choose to reveal best practices.

In addition to the learning problem, there is a sampling problem that must be solved at inference time. It has been acknowledged for diffusion models that sampling can be either deterministic or stochastic [63], and the choice of sampling method can be made after the learning process. Yet, the diffusion coefficients used for stochastic sampling are typically presented as intrinsically tied to the forward noising process, which need not be the case in general.

[Figure 1]

- Fig. 1: Selected samples from SiT-XL models trained on ImageNet [55] at 512 × 512 and 256 × 256 resolution with cfg = 4.0, respectively.

200K 300K 400K 500K 600K 700K Training Steps

45

50

55

60

65

70

75

80

FID-50K

SiT-S DiT-S

200K 300K 400K 500K 600K Training Steps

25

30

35

40

45

50

55

FID-50K

SiT-B DiT-B

200K 300K 400K 500K 600K 700K 800K Training Steps

10

15

20

25

30

35

FID-50K

SiT-L DiT-L

200K 300K 400K 500K 600K 700K 800K Training Steps

12.5

15.0

17.5

20.0

22.5

25.0

FID-50K

SiT-XL DiT-XL

- Fig. 2: SiT improves FID across all model sizes. FID-50K over training iterations for both DiT and SiT. All results are produced by a Euler-Maruyama sampler using 250 integration steps. Across all model sizes, SiT converges much faster.

Throughout this paper, we explore how the design of the interpolant and the use of the resulting model as either a deterministic or a stochastic sampler impact performance. We gradually transition from a typical denoising diffusion model to an interpolant model by taking a series of orthogonal steps in the design space. As we progress, we carefully evaluate how each move away from the diffusion model impacts the performance. In summary, our main contributions are:

- – We systematically study the SiT design space through the combinations of the four key components: time discretization, model prediction, interpolant, and sampler.
- – We provide theoretical motivation for the choice of each component and study how they lead to improved practical performance.
- – We exploit the tunability of the diffusion coefficient of the stochastic sampler, and show that its adaptation can tighten control of the KL-divergence between the model and the target. We show how this leads to empirical benefits without any additional re-training.

- – Combining the best design choices identified in each component, our SiT model surpasses Diffusion Transformer(DiT) on both 256×256 and 512×512 image resolution, achieving FID-50K scores of 2.06 and 2.62, respectively, without modifying any structure or hyperparameter of the model.

### 2 SiT: Scalable Interpolant Transformers

We begin by recalling the main ingredients for building flow-based and diffusionbased generative models.

#### 2.1 Flows and diffusions

Flow and diffusion models both utilize stochastic processes to gradually turn noise ε ∼ N(0,I) into data x∗ ∼ p(x) for the generating task. Such time-dependent processes can be summarized as follow

xt = αtx∗ + σtε, (1)

where αt is a decreasing function of t and σt is an increasing function of t. Stochastic interpolants and other flow matching methods [2,4,41,43] restrict the process (1) on t ∈ [0,1], and set α0 = σ1 = 1, α1 = σ0 = 0, so that xt interpolates exactly between x∗ at time t = 0 and ε and time t = 1. By contrast, score-based diffusion models [33,37,64] set both αt and σt indirectly through a forward-time stochastic differential equation (SDE) with N(0,I) as its equilibrium distribution, i.e. xt converges to N(0,I) only if t → ∞.

Despite the nuances in formulating the stochastic processes xt, common to both stochastic interpolants and score-based diffusion models is the observation that xt can be sampled dynamically using either a reverse-time SDE or a probability flow ordinary differential equation (ODE).

Probability flow ODE. The marginal probability distribution pt(x) of xt in (1) coincides with the distribution of the probability flow ODE with a velocity field

X˙ t = v(Xt,t), (2) where v(x,t) is given by the conditional expectation

v(x,t) = E[x˙t|xt = x],

(3)

= α˙tE[x∗|xt = x] + σ˙tE[ε|xt = x].

The correspondence between pt(x) and (2) and the formulation of (3) is derived in Appendix A.1. By solving (2) backwards in time from XT = ε ∼ N(0,I), we can generate samples from p0(x), which approximates the ground-truth data distribution p(x). We refer to (2) as a flow-based generative model.

Reverse-time SDE. The time-dependent probability distribution pt(x) of xt also coincides with the distribution of the reverse-time SDE [5]

dXt = v(Xt,t)dt −

wts(Xt,t)dt + √wtdW¯ t, (4)

- 1

- 2

where W¯ t is a reverse-time Wiener process, wt > 0 is an arbitrary time-dependent diffusion coefficient, v(x,t) is the velocity defined in (3), and s(x,t) = ∇log pt(x) is the score. Similar to v, this score is given by the conditional expectation

s(x,t) = −σt−1E[ε|xt = x]. (5)

Again, the correspondence between pt(x) and (4) and the formulation of (5) is derived in Appendix A.3. Solving the reverse SDE (4) backwards in time from XT = ε ∼ N(0,I) enables generating samples from the approximated data distribution p0(x) ∼ p(x). We refer to (4) as a stochastic generative model.

Design choices. Score-based diffusion models typically tie the choice of αt, σt, and wt in (4) to the drift and diffusion coefficients used in the forward SDE that generates xt (see (10) below). The stochastic interpolant framework decouples the formulation of xt from the forward SDE and shows that there is more flexibility in the choices of αt, σt, and wt. Below, we will exploit this flexibility to construct generative models that outperform score-based diffusion models on standard benchmarks in image generation task.

#### 2.2 Estimating the score and the velocity

Practical use of the probability flow ODE (2) and the reverse-time SDE (4) as generative models relies on our ability to estimate the velocity v(x,t) and/or score s(x,t) fields that enter these equations. The key observation made in score-based diffusion models is that the score can be estimated parametrically as sθ(x,t) using the loss

T

E[∥σtsθ(xt,t) + ε∥2]dt. (6)

Ls(θ) =

0

This loss can be derived by using (5) along with standard properties of the conditional expectation. Similarly, the velocity in (3) can be estimated parametrically as vθ(x,t) via the loss

Lv(θ) =

T

E[∥vθ(xt,t) − α˙tx∗ − σ˙tε∥2]dt. (7)

0

We note that any time-dependent weight can be included under the integrals in both (6) and (7). These weight factors are key in the context of score-based models when T becomes large [36]; in contrast, with stochastic interpolants where T = 1 without any bias, these weights are less important and might impose numerical stability issue (see Appendix B).

Model prediction. We observed that only one of sθ(x,t) and vθ(x,t) is needed to be estimated in practice. This follows directly from the constraint

x = E[xt|xt = x],

= αtE[x∗|xt = x] + σtE[ε|xt = x],

which can be used to re-express the score (5) in terms of the velocity (3) as

(8)

αtv(x,t) − α˙tx α˙tσt − αtσ˙t

s(x,t) = σt−1

. (9)

We include a detailed derivation in Appendix A.4. Notably, given the simply linear relationship posed by (9), we can also express v(x,t) in terms of s(x,t). We will use this relation to specify our model prediction. In our experiments, we typically learn the velocity field v(x,t) and use it to express the score s(x,t) when using an SDE for sampling.

Note that by our definitions α˙t < 0 and σ˙t > 0, so that the denominator of (9) is never zero. Yet, σt vanishes at t = 0, making the σt−1 in (9) cause a singularity1. This suggests the choice wt = σt in (4) to cancel this singularity, for which we will explore the performance in the numerical experiments.

Time discretization. The objective functions specified above are defined over a continuous time domain, as opposed to DDPM which couples the time grid used in learning to that used in sampling. Learning in continuous time allows us to specify a discretization used in sampling a posteriori, which allows for flexibility in both sampling efficiency and performance.

#### 2.3 Specifying the interpolating process

In Sec. 2.1 we present the general definition of interpolants (αt and σt) for both stochastic interpolant and score-based diffusion. In this section we dive into more details and specify the three choices of interpolants to explore in the experiments.

Score-based diffusion. We follow [64] and use the standard variance-preserving (VP) SDE in forward-time

dXt = −

- 1

- 2

βtXtdt + βtdWt (10)

for some βt > 0, xt’s perturbation kernel pt(xt|x0) = N(αtxt,σt2I) is defined by

- 1

- 2

t

t

SBDM-VP: αt = e−

0 βsds, σt = 1 − e−

0 βsds. (11)

1 We remark that s(x, t) can be shown to be non-singular at t = 0 analytically if the data distribution p(x) has a smooth density [2], though this singularity appears in numerical implementations and losses in general.

−−−−−−−−−−−−−−−−−−→Increasing transformer sizes

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

- Fig. 3: Increasing transformer size increases sample quality. Best viewed zoomedin. We sample from all 4 of our SiT model (SiT-S, SiT-B, SiT-L and SiT-XL) after 400K training steps using the same latent noise and class label.

The only design flexibility in (11) comes from the choice of βt, as it determines both αt and σt2. For example, setting βt = 1 leads to αt = e−t and σt = √1 − e−2t. This choice necessitates taking T sufficiently large [25] or searching for more appropriate choices of βt [16,60,64] to reduce the bias. To be specific, such bias comes from the mismatch between the condition ε ∼ N(0,I) used in practice for sampling and the density of the process x1 ̸∼ N(0,I), as stated in Sec. 2.1.

General interpolants. In the stochastic interpolant framework, the process (1) is defined explicitly and without any reference to a forward SDE, creating more flexibility in the choice of αt and σt. Specifically, any choice satisfying:

- (i) αt2 + σt2 > 0 for all t ∈ [0,1];
- (ii) αt and σt are differentiable for all t ∈ [0,1];
- (iii) α1 = σ0 = 0, α0 = σ1 = 1;

gives a process that interpolates without bias between xt=0 = x∗ and xt=1 = ε. In our numerical experiments, we exploit this design flexibility to test, in particular, the choices

Linear: αt = 1 − t, σt = t, GVP: αt = cos(21πt), σt = sin(12πt),

(12)

where GVP refers to a generalized VP which has constant variance across time for any endpoint distributions with the same variance. We note that the fields v(x,t) and s(x,t) entering (2) and (4) depend on the choice of αt and σt, and typically must be specified before learning3. This is in contrast to the diffusion coefficient w(t), as we now describe.

- 2 VP is the only linear scalar SDE with an equilibrium distribution [60]; interpolants extend beyond αt2+σt2 = 1 by foregoing the requirement of an equilibrium distribution.
- 3 The requirement to learn and sample under one choice of path specified by αt, σt, at training time may be relaxed and is explored in [1].

#### 2.4 Specifying the diffusion coefficient

As stated earlier, the SBDM diffusion coefficient used in (4) is usually taken to match that of (10). That is, one sets wt = βt. In the stochastic interpolant framework, this choice is again subject to greater flexibility: any wt ≥ 0 can be used. Interestingly, this choice can be made after learning, as it does not affect the velocity v(x,t) or the score s(x,t). In our experiments, we exploit this flexibility by considering the following choices:

- (i) wt = σt; this is used to eliminate the singularity at t = 0 following the explanation at the end of Sec. 2.2;
- (ii) wt = sin2(πt); this also eliminates the singularity at t = 0, and allows us to explore the effect of removing diffusivity at times close to t = 1 in sampling.
- (iii) wt can be chosen to minimize an upper bound on the KL divergence DKL(p(x)∥p0(x)), where p(x) denotes the true data distribution and p0(x) refers to the density of xt at t = 0. Disregarding the simulation cost of integrating the SDE (4), it can be shown (see Appendix A.5) that the following choice of wt minimizes the KL upper bound:

wt = wtKL ≡ 2 σ ˙tσt −

α˙tσt2 αt

. (13)

Under the SBDM-VP interpolant, wtKL coincides with βt; this aligns with the claim made in [63].

- (iv) If the SDE in (iii) becomes hard to integrate because of the magnitude of

wtKL near t = 1, one may wish to regularize the diffusion coefficient to reduce the integration cost. For example, difficulties may arise for the Linear and

GVP interpolants, because wtKL → ∞ as t → 1 given the presence of αt in the denominator of (13). Including the integration cost of (4), it can also be

shown (see Appendix A.5) that an optimal regularized wt is given by

wtKL,η ≡ wtKL Lt Lt + 2η(wtKL)2

, (14)

where Lt is the value of Lv in Sec. 2.2 at time t, and η is any non-negative constant. With η = 0, we recover wtKL. For score models, we first convert to a velocity model following (9), then calculate the corresponding Lv. As t → 1, wtKL,η approaches a limit at L

2η . If Lt is defined everywhere on [0,1], then wtKL,η will be well-behaved on [0,1].

t→1

#### 2.5 Interpolant Transformer Architecture

The backbone architecture and capacity of generative models are both crucial for producing high-quality samples. In order to eliminate any confounding factors and focus on our exploration, we strictly follow the standard Diffusion Transformer

(DiT) [50] and its configurations. This way, we can also test the scalability of our model across various model sizes.

Here we briefly introduce the model design. Generating high-resolution images with diffusion models can be computationally expensive. Latent diffusion models (LDMs) [53] address this by first downsampling images into a smaller latent embedding space using an encoder E, and then training a diffusion model on z = E(x). New images are created by sampling z from the model and decoding it back to images using a decoder x = D(z).

Similarly, SiT is a latent generative model, and we use the same pre-trained VAE encoder and decoder models originally used in Stable Diffusion [53]. SiT processes a spatial input z (shape 32 × 32 × 4 for 256 × 256 × 3 images) by first ‘patchifying’ it into T linearly embedded tokens of dimension d. We always use a patch size of 2 in these models as they achieve the best sample quality. We then apply standard ViT [21] sinusoidal positional embeddings to these tokens. We use a series of N SiT transformer blocks, each with hidden dimension d.

Our model configurations—SiT-{S,B,L,XL}—vary in model size (parameters) and compute (flops), allowing for a model scaling analysis. For class-conditional generation on ImageNet, we use the AdaLN-Zero block [50] to process additional conditional information (times and class labels). SiT architectural details are listed in Appendix E.

The complete SiT design space that we explore consists of the choice of time discretization and the model prediction (Sec. 2.2), the choice of the interpolant (Sec. 2.3), the choice of sampler and diffusion coefficient (Sec. 2.4), and the model size (Sec. 2.5).

### 3 Experiments

To provide a more detailed answer to the question raised in Table 1 and make a fair comparison between DiT and SiT, we gradually transition from a DiT model (discrete, score prediction, VP interpolant) to a SiT model (continuous, velocity prediction, Linear interpolant) and present the impacts on performance.

Experimental setup. In the transition experiments, we use SiT-B models trained on 256 × 256 image resolution on the ImageNet as our backbone. We fix training steps to be 400K throughout the transition. For solving the ODE (2), we adopt a fixed step second-order Heun integrator; for solving the SDE (4), we used a first-order Euler-Maruyama integrator. With both solver choices we limit the number of function evaluations (NFE) to be 250 to match the number of sampling steps used in DiT. All metrics presented are FID-50K scores evaluated on the ImageNet training set unless otherwise stated.

We also scale up our SiT model to the XL configuration and train on both 256 × 256 and 512 × 512 resolution on ImageNet. We strictly follow the training settings of DiT and did not tune any hyperparameters.

#### 3.1 Model Parameterization

Discrete- to continuous-time. Continuous time training has been previously studied from the perspective of improved likelihood bounds [37,64]. As mentioned in Section 2.2, here we focus on the fact that training in continuous time allows us to decouple discretization choices in sampling from the particular training method, which allows for finding the right discretization for various choices of diffusion coefficients that we are free to choose after training. We observe a marginal performance increase in Table 2 by switching to continuous time.

We additionally observe in Figure 5 that flexibility in integration allows one to trade-off number of functional evaluations and FID performance.

Model parameterization. To clarify the role of the model parameterization in the context of SBDM-VP, we now compare learning (i) a score model using (6) (Ls), (ii) a weighted score model (Lsλ

), or (iii) a velocity model using (7)(Lv). We observe a significant performance increase with Lsλ

and Lv in Table 3.

In accordance with the observation made in [36], we carefully choose a λ(t) such that λs

is made equivalent to λv. We will provide detailed derivations in Appendix A.3, and demonstrate such λ is closely related to the maximum likelihood weighting proposed in [63,66]. Furthermore, we note that λ(t) → ∞ as t → 0, thus compensating for the vanishing gradient of the score objective when near the data. This could also account for the performance gain from λs to λs

λ

.

λ

Table 2: Discrete vs. continuous.

Model Objective FID DDPM Noise LNs 44.2 SBDM-VP Score Ls 43.6

##### Table 3: Effect of parameterizations.

Interpolant Model Objective FID SBDM-VP Score Ls 43.6 SBDM-VP Score Lsλ 39.1 SBDM-VP Velocity Lv 39.8

Choices of interpolant. Sec. 2 highlights that there are many possible ways to build a connection between the data distribution and a Gaussian by varying the choice of αt and σt in the definition of the interpolant (1). To understand the role of this choice, we now study the benefits of moving away from the commonly-used SBDM-VP setup. We consider learning a velocity model v(x,t) with the Linear and GVP interpolants presented in (12), which make the interpolation between the Gaussian and the data distribution exact on [0,1]. We benchmark these models against the SBDM-VP in Table 4, where we find that both the GVP and Linear interpolants obtain significantly improved performance.

One possible explanation for this observation is given in Fig. 4, where we see that the path length (transport cost) is reduced when changing from SBDM-VP to GVP or Linear. We note that this is equivalently reducing curvatures in the

ODE trajectories from SBDM-VP to Linear, which is known to reduce the timediscretization errors in sampling [40,43], and thus contributing to the performance. Numerically, we also note that for SBDM-VP, σ˙t = βte−

t

0 βsds/(2σt) becomes singular at t = 0: this can pose numerical difficulties inside Lv, leading to difficulty in learning near the data distribution. This issue does not appear with the GVP and Linear interpolants.

Table 4: Effect of interpolant. Interpolant Model Objective FID SBDM-VP Velocity Lv 39.8

Linear Velocity Lv 34.8 GVP Velocity Lv 34.6

Table 5: ODE vs. SDE, wt = wtKL. Interpolant Model Objective ODE SDE SBDM-VP Velocity Lv 39.8 37.8

Linear Velocity Lv 34.8 33.6 GVP Velocity Lv 34.6 32.9

#### 3.2 Deterministic vs stochastic sampling

As shown in Sec. 2, given a learned model, we can sample using either the probability flow equation (2) or an SDE (4). In Tab. 5 we illustrate the discrepancy between the two methods when using the same trained velocity model. We find performance improvements by sampling with an SDE over the ODE, which is in line with the bounds given in [2]: the SDE has better control over the KL divergence between the model density at t = 0 and the ground truth data distribution. We also note that the performance of ODE and SDE integrators may differ under different computation budgets. As shown in Fig. 5, the ODE converges faster with fewer NFE, while the SDE is capable of reaching a much lower final FID score when given a larger computational budget.

Tunable diffusion coefficient. Motivated by the improved performance of SDE sampling, we now consider the effect of tuning the diffusion coefficient in inference. As shown in Table 6, we sweep through all different combinations of our model prediction and interpolant, and present the result. We find that the optimal choice for sampling is both model prediction and interpolant dependent.

According to Sec. 2.4, the choice of wt = wtKL would ideally minimize the upper bound for the KL divergence DKL(p(x)|∥p0(x)) and make the SDE approximate the data distribution more closely, barring integration costs. This theoretical result is supported by empirical observation for the SBDM-VP and GVP interpolants presented in Table 6. For Linear interpolants, the cost-regularized version wtKL,η provides the best FID, because the SDE for the Linear interpolant with wtKL becomes hard to integrate at the endpoint. Generally speaking, the score models perform worse than the velocity models, which may be due to the singularity of the objective in (6). Moreover, the efficacy of using wtKL in this context is also reduced, for similar reason. For example, reverting (9) to obtain vθ(x,t)

0.7

0.6

2 ||v(x,t)p(x)dxdtθt

0.5

SBDM-VP

Linear

0.4

GVP

0.3

0.2

100K 200K 300K 400K 500K 600K 700K Training Steps

Fig. 4: Path length. The path length C(v) = 0 1 E[|v(xt, t)|2]dt arising from the velocity field at different training steps; each curve is approximated by 10000 datapoints at each training step.

ODE

44

SDE: σt

SDE: wtKL

42

SDE: sin(πt)2

SDE: wtKL,η

40

FID-10K

38

36

34

8 16 32 64 128 256 512 1024 NFE

Fig. 5: Comparison of ODE and SDE w/ choices of diffusion coefficients. We evaluate each sampler using a 400K steps trained SiT-B model with Linear interpolant and learning the v(x, t).

- Table 6: Evaluation of our SDE samplers. The last three columns specify different diffusion coefficients wt. To make the SBDM-VP competitive, we perform evaluation on the weighted score model Lsλ. We mark the optimal wt for each interpolant.

Interpolant Model Objective wt = wtKL wt = σt wt = sin2(πt) wt = wtKL,η SBDM-VP velocity Lv 37.8 38.7 39.2 41.1

score Lsλ 35.7 37.1 37.7 38.9 GVP velocity Lv 32.9 33.4 33.6 33.2 score Ls 37.8 33.5 33.2 33.3 Linear velocity Lv 33.6 33.5 33.3 33.0 score Ls 41.0 35.3 34.4 34.9

from sθ(x,t) will result in a singularity at t = 1 in Lt used to choose wtKL,η in (14). Lastly, for SBDM-VP we observe worse result from wtKL,η as opposed to wtKL. Different from Linear and GVP, as stated in Sec. 2.4 and Sec. 3.1, wtKL is well-defined everywhere on [0,1] for SBDM-VP, whereas wtKL,η suffers from the singularity issue posed by Lv near t = 0. These observations supports our claim made before, that the optimal choice of wt will always be model prediction and interpolant dependent.

We also note that the influences of different diffusion coefficients can vary across different model sizes. Empirically, we observe the best choice for our SiT-XL is a velocity model with Linear interpolant and sampled with wtKL,η.

#### 3.3 Classifier-free guidance

Classifier-free guidance (CFG) [27] often leads to improved performance for score-based models. In this section, we give a concise justification for adopting

it on the velocity model, and then empirically show that the drastic gains in performance for DiT case carry across to SiT.

Guidance for a velocity field means that: (i) that the velocity model vθ(x,t;y) takes class labels y during training, where y is occasionally masked with a null token ∅; and (ii) during sampling the velocity used is vθζ(x,t;y) = ζvθ(x,t;y) + (1 − ζ)vθ(x,t;∅) for a fixed ζ > 0. In Appendix C, we show that this indeed corresponds to sampling the tempered density p(xt)p(y|xt)ζ as proposed in [48]. Given this observation, one can leverage the usual argument for classifier-free guidance of score-based models.

We observed similar performance improvement with our SiT-XL models under identical computation budget and CFG scale as DiT-X: models. For SiT-XL 256 × 256, we follow identical settings in DiT and train the model for 7M steps. We show samples in Fig. 1, and report the result in Table 7. For SiT-XL 512×512, we train the model for 3M steps under the same setting and report the result in Table 7. Under both training settings we observe performance advantage of SiT. We display more samples in Fig. 1 and in Appendix F.

- Table 7: Benchmarking class-conditional image generation on ImageNet 256 × 256 and 512 × 512. SiT-XL surpasses DiT-XL in both resolutions.

Class-Conditional ImageNet 256 × 256 Model FID↓ sFID↓ IS↑ Precision↑ Recall↑

Class-Conditional ImageNet 512 × 512 Model FID↓ sFID↓ IS↑ Precision↑ Recall↑

BigGAN-deep [10] 6.95 7.36 171.4 0.87 0.28 StyleGAN-XL [57] 2.30 4.02 265.12 0.78 0.53

BigGAN-deep [10] 8.43 8.13 177.90 0.88 0.29 StyleGAN-XL [57] 2.41 4.06 267.75 0.77 0.52

Mask-GIT [12] 6.18 - 182.1 - -

ADM [19] 10.94 6.02 100.98 0.69 0.63 ADM-G, ADM-U 3.94 6.14 215.84 0.83 0.53

Mask-GIT [12] 7.32 - 156.0 - -

ADM [19] 23.24 10.19 58.06 0.73 0.60 ADM-G, ADM-U 3.85 5.86 221.72 0.84 0.53

CDM [26] 4.88 - 158.71 - RIN [31] 3.42 - 182.0 - Simple Diffusion(U-Net) [28] 3.76 - 171.6 - Simple Diffusion(U-ViT, L) 2.77 - 211.8 - VDM++ [36] 2.12 - 267.7 - DiT-XL(cfg = 1.5) [50] 2.27 4.60 278.24 0.83 0.57 SiT-XL(cfg = 1.5, ODE) 2.15 4.60 258.09 0.81 0.60 SiT-XL(cfg = 1.5, SDE) 2.06 4.49 277.50 0.83 0.59

Simple Diffusion(U-Net) [28] 4.28 - 171.0 - Simple Diffusion(U-ViT, L) 4.53 - 205.3 - -

VDM++ [36] 2.65 - 278.1 - DiT-XL(cfg = 1.5) [50] 3.04 5.02 240.82 0.84 0.54 SiT-XL(cfg = 1.5, SDE) 2.62 4.18 252.21 0.84 0.57

### 4 Related Work

Transformers. The transformer architecture [67] has emerged as a powerful tool for application domains as diverse as vision [21,49], language [68,69], quantum chemistry [23], active matter systems [9], and biology [11]. Several works have built on DiT and have made improvements by modifying the architecture to internally include masked prediction layers [22,70]; these choices are orthogonal to this work and may be fruitfully combined in future work.

Training and Sampling in Diffusions. Diffusion models arose from [25,61,64] and have close historical relationship with denoising methods [29,30,59]. Various

efforts have gone into improving the sampling algorithms behind these methods in the context of DDPM [62] and SBDM [33,63]; these are also orthogonal to our studies and may be combined to push for better performance in future work. Improved Diffusion ODE [71] also studies several combinations of model parameterizations (velocity versus noise) and paths (VP versus Linear). Unlike our work, they focus on lower dimensional experiments, benchmark with likelihoods, and do not consider SDE sampling.

Interpolants and flow matching. Velocity parameterizations using the Linear interpolant were also studied in [41,43], and were generalized to the manifold setting in [6]. A trade-off in bounds on the KL divergence between the target distribution and the model arises when considering sampling with SDEs versus ODE; [2] shows that minimizing the objectives presented in this work controls KL for SDEs, but not for ODEs. Error bounds for SDE-based sampling with score-based diffusion models are studied in [13,14,38,39], for ODE-base sampling are explored in [7,15], in addition to the Wasserstein bounds provided in [4].

Other related works make improvements by changing how noise and data are sampled during training. [52,65] compute mini-batch optimal couplings between the Gaussian and data distribution to reduce the transport cost and gradient variance; [3] instead build the coupling by flowing directly from the conditioning variable to the data for image-conditional tasks. Finally, various work considers learning a stochastic bridge connecting two arbitrary distributions [18,44,51,58]. These directions are compatible with our investigations; they specify the learning problem for which one can vary the choices of model parameterizations, interpolant schedules, and sampling algorithms.

Diffusion in Latent Space. Generative modeling in latent space [53, 66] is a tractable approach for modeling high-dimensional data. The approach has been applied beyond images to video generation [8], which is a yet-to-be explored and promising application area for velocity trained models. [17] also train velocity models in the latent space of the pre-trained Stable Diffusion VAE. They demonstrate promising results for the DiT-B backbone with a final FID-50K of 4.46; their study was one motivation for the investigation in this work regarding which aspects of these models contribute to the gains in performance over DiT.

### 5 Conclusion

In this work, we have presented Scalable Interpolant Transformers, a simple and powerful framework for image generation tasks. Within the framework, we explored the tradeoffs between a number of key design choices: the choice of a continuous or discrete-time model, the choice of interpolant, the choice of model prediction, and the choice of diffusion coefficient. We highlighted the advantages and disadvantages of each choice and demonstrated how careful decisions can lead to significant performance improvements. Many concurrent works [24,32,42,47] explore similar approaches in a wide variety of downstream tasks, and we leave the application of SiT to these tasks for future works.

### Acknowledgements

We would like to thank Adithya Iyer, Sai Charitha Akula, Fred Lu, Jiatao Gu, and Edwin P. Gerber for helpful discussions and feedback. The research is partly supported by the Google TRC program.

### References

- 1. Albergo, M.S., Boffi, N.M., Lindsey, M., Vanden-Eijnden, E.: Multimarginal generative modeling with stochastic interpolants. arXiv preprint arXiv:2310.03695

(2023)

- 2. Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E.: Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 (2023)
- 3. Albergo, M.S., Goldstein, M., Boffi, N.M., Ranganath, R., Vanden-Eijnden, E.: Stochastic interpolants with data-dependent couplings. arXiv preprint arXiv:2310.03725 (2023)
- 4. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. In: ICLR (2023)
- 5. Anderson, B.D.: Reverse-time diffusion equation models. Stochastic Processes and their Applications (1982)
- 6. Ben-Hamu, H., Cohen, S., Bose, J., Amos, B., Grover, A., Nickel, M., Chen, R.T., Lipman, Y.: Matching normalizing flows and probability paths on manifolds. In: ICML (2022)
- 7. Benton, J., Deligiannidis, G., Doucet, A.: Error bounds for flow matching methods. arXiv preprint arXiv:2305.16860 (2023)
- 8. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: CVPR (2023)
- 9. Boffi, N.M., Vanden-Eijnden, E.: Deep learning probability flows and entropy production rates in active matter. arXiv preprint arXiv:2309.12991 (2023)
- 10. Brock, A., Donahue, J., Simonyan, K.: Large scale gan training for high fidelity natural image synthesis. In: ICLR (2019)
- 11. Chandra, A., Tünnermann, L., Löfstedt, T., Gratz, R.: Transformer-based deep learning for predicting protein properties in the life sciences. Elife 12, e82819 (2023)
- 12. Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: Maskgit: Masked generative image transformer. In: CVPR (2022)
- 13. Chen, H., Lee, H., Lu, J.: Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In: ICML (2023)
- 14. Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., Zhang, A.: Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In: ICLR (2023)
- 15. Chen, S., Daras, G., Dimakis, A.: Restoration-degradation beyond linear diffusions: A non-asymptotic analysis for DDIM-type samplers. In: ICML (2023)
- 16. Chen, T.: On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972 (2023)
- 17. Dao, Q., Phung, H., Nguyen, B., Tran, A.: Flow matching in latent space. arXiv preprint arXiv:2307.08698 (2023)
- 18. De Bortoli, V., Thornton, J., Heng, J., Doucet, A.: Diffusion schrödinger bridge with applications to score-based generative modeling. In: NeurIPS (2021)

- 19. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. In: NIPS

(2021)

- 20. Dockhorn, T., Vahdat, A., Kreis, K.: Score-based generative modeling with criticallydamped langevin diffusion. In: ICLR (2022)
- 21. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)
- 22. Gao, S., Zhou, P., Cheng, M.M., Yan, S.: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389 (2023)
- 23. von Glehn, I., Spencer, J.S., Pfau, D.: A Self-Attention Ansatz for Ab-initio Quantum Chemistry. In: ICLR (2023)
- 24. Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Fei-Fei, L., Essa, I., Jiang, L., Lezama, J.: Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662 (2023)
- 25. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS

- (2020)

26. Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M., Salimans, T.: Cascaded diffusion models for high fidelity image generation. arXiv preprint arXiv:2106.15282

- (2021)

- 27. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- 28. Hoogeboom, E., Heek, J., Salimans, T.: simple diffusion: End-to-end diffusion for high resolution images. In: ICML (2023)
- 29. Hyvärinen, A.: Estimation of non-normalized statistical models by score matching. JMLR (2005)
- 30. Hyvärinen, A.: Sparse code shrinkage: Denoising of nongaussian data by maximum likelihood estimation. Neural Computation (1999)
- 31. Jabri, A., Fleet, D., Chen, T.: Scalable adaptive computation for iterative generation. In: ICML (2023)
- 32. Jakab, T., Li, R., Wu, S., Rupprecht, C., Vedaldi, A.: Farm3D: Learning articulated 3d animals by distilling 2d diffusion. In: 3DV (2024)
- 33. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusionbased generative models. In: NeurIPS (2022)
- 34. Kidger, P.: On Neural Differential Equations. Ph.D. thesis, University of Oxford

(2021)

- 35. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR (2015)
- 36. Kingma, D.P., Gao, R.: Understanding the diffusion objective as a weighted integral of elbos. arXiv preprint arXiv:2303.00848 (2023)
- 37. Kingma, D.P., Salimans, T., Poole, B., Ho, J.: Variational diffusion models. In: NeurIPS (2021)
- 38. Lee, H., Lu, J., Tan, Y.: Convergence for score-based generative modeling with polynomial complexity. In: NeurIPS (2022)
- 39. Lee, H., Lu, J., Tan, Y.: Convergence of score-based generative modeling for general data distributions. In: ALT (2023)
- 40. Lee, S., Kim, B., Ye, J.C.: Minimizing trajectory curvature of ode-based generative models. In: ICML (2023)
- 41. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: ICLR (2023)
- 42. Liu, R., Wu, R., Hoorick, B.V., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1to-3: Zero-shot one image to 3d object. In: ICCV (2023)

- 43. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: ICLR (2023)
- 44. Liu, X., Wu, L., Ye, M., Liu, Q.: Let us build bridges: Understanding and extending diffusion generative models. arXiv preprint arXiv:2208.14699 (2022)
- 45. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)
- 46. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In: NeurIPS (2022)
- 47. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic differential equations. In: ICLR (2022)
- 48. Nichol, A., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: ICML (2021)
- 49. Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, L., Shazeer, N., Ku, A., Tran, D.: Image Transformer. In: ICML (2018)
- 50. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: ICCV (2023)
- 51. Peluchetti, S.: Non-denoising forward-time diffusions. In: ICLR (2022)
- 52. Pooladian, A.A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., Chen, R.T.Q.: Multisample flow matching: Straightening flows with minibatch couplings. In: ICML (2023)
- 53. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)
- 54. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI (2015)
- 55. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. IJCV (2015)
- 56. Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. In: ICLR (2022)
- 57. Sauer, A., Schwarz, K., Geiger, A.: Stylegan-xl: Scaling stylegan to large diverse datasets. In: SIGGRAPH (2022)
- 58. Shi, Y., Bortoli, V.D., Campbell, A., Doucet, A.: Diffusion schrödinger bridge matching. In: NIPS (2023)
- 59. Simoncelli, E.P., Adelson, E.H.: Noise removal via bayesian wavelet coring. In: ICIP

(1996)

- 60. Singhal, R., Goldstein, M., Ranganath, R.: Where to diffuse, how to diffuse, and how to get back: Automated learning for multivariate diffusions. In: ICLR (2023)
- 61. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: ICML (2015)
- 62. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: ICLR (2021)
- 63. Song, Y., Durkan, C., Murray, I., Ermon, S.: Maximum likelihood training of score-based diffusion models. In: NeurIPS (2021)
- 64. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Score-based generative modeling through stochastic differential equations. In: ICLR

(2021)

- 65. Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport. In: ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems (2023)
- 66. Vahdat, A., Kreis, K., Kautz, J.: Score-based generative modeling in latent space. In: NIPS (2021)
- 67. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NIPS (2017)

- 68. Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D.F., Chao, L.S.: Learning deep transformer models for machine translation. In: ACL (2019)
- 69. Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., Ahmed, A.: Big Bird: Transformers for Longer Sequences. In: NeurIPS (2020)
- 70. Zheng, H., Nie, W., Vahdat, A., Anandkumar, A.: Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305 (2023)
- 71. Zheng, K., Lu, C., Chen, J., Zhu, J.: Improved techniques for maximum likelihood estimation for diffusion odes. In: ICML (2023)

### A Proofs

In all proofs below, we use · for dot product and assume all bold notations (x, ε, etc.) are real-valued vectors in Rd. Most proofs are derived from [2].

- A.1 Proof of the probability flow ODE (2) with the velocity in Eq. (3).

Consider the time-dependent probability density function (PDF) pt(x) of xt = αtx∗ + σtε defined in Eq. (1). By definition, its characteristic function pˆt(k) =

Rd eik·xpt(x)dx is given by

pˆt(k) = E[eik·x

] (15)

t

where E denotes expectation over x∗ and ε. Taking time derivative on both sides, and using the tower property of conditional expectation, we have

∂tpˆt(k) = ik · E[x˙teik·x

] (16)

t

[E[x˙teik·x

|xt = x]] (17)

= ik · Ex∼p

t

t

[E[(α˙tx∗ + σ˙tε)eik·x

|xt = x]] (18)

= ik · Ex∼p

t

t

[E[(α˙tx∗ + σ˙tε)|xt = x]eik·x] (19)

= ik · Ex∼p

t

[v(x,t)eik·x] (20)

= ik · Ex∼p

t

where v(x,t) = E[(α˙tx∗ + σ˙tε)|xt = x] = α˙tE[x∗|xt = x] + σ˙tE[ε|xt = x] is the velocity defined in Eq. (3). Explicitly, Eq. (20) reads

∂t

eik·xpt(x)dx = ik ·

Rd

from which we deduce

v(x,t)eik·xpt(x)dx (21)

Rd

eik·x∂tpt(x)dx =

Rd

= −

v(x,t) · ∇x[eik·x]pt(x)dx (22)

Rd

∇x · [v(x,t)pt(x)]eik·xdx (23)

Rd

where ∇x · [vpt] = di=1 ∂x∂

[vipt] is the divergence operator and we used integration by parts to get the second equality. By the properties of Fourier transform, Eq. (23) implies that pt(x) satisfies the transport equation

i

∂tpt(x) + ∇x · (v(x,t)pt(x)) = 0. (24)

Solving this equation by the method of characteristic leads to probability flow ODE (2).

#### A.2 Proof of the SDE (4)

We show that the SDE (4) has marginal density pt(x) with any choice of wt ≥ 0. To this end, recall that solution to the SDE

dXt = [v(Xt,t) −

wts(Xt,t)]dt + √wtdW¯ t

- 1

- 2

has a PDF that satisfies the Fokker-Planck equation

∂tpt(x) = −∇x · [v(x,t) −

- 1

- 2

wts(x,t)]pt(x) −

- 1

- 2

wt∆xpt(x) (25)

2

where ∆x is the Laplace operator defined as ∆x = ∇x · ∇x = di=0 ∂

∂x2i . Reorganizing the equation and usng the definition of the score s(x,t) = ∇x log pt(x) = p−t 1(x)∇xpt(x), we have

- 1

- 2

- 1

- 2

] −

∂tpt(x) = −∇x · [v(x,t)pt(x)]

wt∇x · [∇x log pt(x)pt(x)

+

wt∆xpt(x)

= ∂tpt(x) by Eq. (24)

= ∇xpt(x)

(26)

- 1

- 2

- 1

- 2

wt∆xpt(x) (27)

wt∇x · ∇xpt(x) −

=⇒ 0 =

By definition of Laplace operator, the last equation holds for any wt ≥ 0. When wt = 0, the Fokker-Planck equation reduces to a continuity equation, and the SDE reduces to an ODE, so the connection trivially holds.

#### A.3 Proof of the expression for the score in Eq. (5)

We show that s(x,t) = −σt−1E[ε|xt = x]. Letting fˆ(k,t) = E[εeiσ

tk·ε], we have

i σt∇kE[eiσ

fˆ(k,t) = −

tk·ε] (28)

Since ε ∼ N(0,I), we can compute the expectation explicitly to obtain

i σt

- 1

- 2σt2|k|2) (29)

fˆ(k,t) = −

(∇ke−

- 1

- 2σt2|k|2 (30)

= iσtke− Since x∗ and ε are independent random variable, we have E[εeik·x

- 1

- 2σt2|k|2E[eiα

] = fˆ(k,t)E[eiα

tk·x∗] = iσtke−

tk·x∗] combine this

= iσtkpˆt(k) (31)

t

where pˆt(k) is the characteristic function of xt = αtx∗ + σtε defined in Eq. (15). The left hand-side of this equation can also be written as:

E[εeik·x

] =

t

=

whereas the right hand-side is

E[εeik·x

|xt = x]pt(x)dx (32)

t

Rd

E[ε|xt = x]eik·xpt(x)dx, (33)

Rd

eik·xpt(x)dx (34)

iσtkpˆt(k) = iσtk

Rd

∇x[eik·x]pt(x)dx (35)

= σt

Rd

eik·x∇xpt(x)dx (36)

= −σt

Rd

eik·xs(x,t)pt(x)dx (37)

= −σt

Rd

where we used integration by parts to get the third equality, and again the definition of the score to get the last.

Comparing Eq. (33) and Eq. (37) we deduce that, when σt ̸= 0,

s(x,t) = ∇x log pt(x) = −σt−1E[ε|xt = x] (38) Further, setting wt to σt in Eq. (4) gives

1 2

- 1

- 2

wts(xt,t) = −

E[ε|xt = x] (39)

for all t ∈ [0,1]. This bypass the constraint of σt ̸= 0 and effectively eliminate the singularity at t = 0.

#### A.4 Proof of Eq. (9)

We note that there exists a straightforward connection between v(x,t) and s(x,t). From Eq. (1), we have

v(x,t) = α˙tE[x∗|xt = x] + σ˙tE[ε|xt = x] (40)

xt − σtε

αt |xt = x] + σ˙tE[ε|xt = x] (41)

= α˙tE[

α˙t αt

α˙tσt αt

)E[ε|xt = x] (42)

x + (˙σt −

=

α˙tσt αt

α˙t αt

)(−σts(x,t)) (43)

x + (˙σt −

=

α˙t αt

x − λtσts(x,t) (44)

=

where we defined

α˙tσt αt

. (45) Given Eq. (44) is linear in terms of s, reverting it will lead to Eq. (9).

λt = σ˙t −

Note that we can also plug Eq. (44) into the loss Lv in Eq. (7) to deduce that

T

α˙t αt

+λt(−σtsθ(xt,t)) − α˙tx∗ − σ˙tε∥2]dt (46)

E[∥

Lv(θ) =

#### x

0

Expand to xt = αtx∗ + σtε

T

α˙tσt αt

ε + λt(−σtsθ(xt,t)) − α˙tx∗ − σ˙tε∥2]dt (47)

E[∥α˙tx∗ +

=

0

T

E[∥λt(−σtsθ(xt,t)) − λtε∥2]dt (48)

=

0

T

λ2tE[∥σtsθ(xt,t) + ε∥2]dt (49) ≡ Lsλ

=

0

(θ) (50) which defines the weighted score objective Lsλ

(θ). This observation is consistent with the claim made in [36] that the score objective with different monotonic weighting functions coincides with losses for different model parameterizations. In Appendix B we show that λt corresponds to the square of the maximum

- likelihood weighting proposed in [63] and [66].

#### A.5 Proof for the optimal wt for tightening the KL bound Lemma 2.22 in [2] asserts that:

- 1

- 2

DKL(p(x)∥pθ(x)) ≤

1

wt−1

0

|b(x,t) − bθ(x,t)|2pt(x)dtdx (51)

Ω

where p(x) denotes the true data distribution, pθ(x) denotes the approximated data distribution by our model at time t = 0, and pt corresponds to the marginal density in Appendix A.2. We further use b and bθ to refer to the ground truth and approximated drift for the reverse SDE, respectively; that is, b(x,t) = v(x,t) − 12wts(x,t). Following Appendix A.4, b(x,t) can be expressed in terms of v(x,t)

b(x,t) = (1 +

wt λtσt

- 1

- 2

)v(x,t) −

- 1

- 2

wtα˙t αtλtσt

and similarly for bθ(x,t). Plug back, Eq. (51) becomes

x (52)

- 1

- 2

DKL(p(x)∥pθ(x)) ≤

1

wt−1(1 +

0

- 1

- 2

wt λtσt

)2

|v(x,t) − vθ(x,t)|2pt(x)dtdx

Ω

(53)

Since v(x,t) = E[x˙|xt = x] from Eq. (1), we have

|v(x,t) − vθ(x,t)|2pt(x)dx = E[|v(x,t) − vθ(x,t)|2] ≤ E[|x˙ − vθ(x,t)|2] ≡ Lt (54)

Ω

where Lt is the loss at time t of our model after optimization. With Eq. (55) we can further simplify Eq. (54) to be

1

- 1

- 2

- 1

- 2

wt λtσt

wt−1(1 +

)2Ltdt (55)

DKL(p(x)∥pθ(x)) ≤

0

We note the minimum of the integrand in Eq. (55) is achieved at wt = 2λtσt with a value of 2 L

λtσt. We note that such wt is the exact choice of wtKL in Sec. 2.4.

t

For SBDM-VP interpolant, such wtKL coincides with βt in [64], and is well defined and positive everywhere on [0,1]. For GVP and Linear interpolant however, this diffusion coefficient is zero at t = 0 and infinity at t = 1. Since σt = O(t) at t = 0, the integrand 2 L

λtσt is not integrable at t = 0, making the bound in Eq. (55) trivially ∞ unless limt→0 Lt = 0.

t

We note the bound proposed in Eq. (55) does not account for the cost of time-integration of the SDE. Assuming that the non-uniform integration step ∆t one must take to maintain a given precision is inversely proportional to wt, that is ∆t = O(wt−1), we can account for the integration cost by adding a term to Eq. (55)

2

1

1

1 2

- 1

- 2

wt λtσt

wt−1 1 +

wtdt (56)

DKL(p(x)∥pθ(x)) ≤

Ltdt + η

0

0

where η > 0 is a parameter that controls the integration error: the higher the η, the smaller the cost but the higher the error, and vice-versa. The minimum of the integrand in Eq. (56) is

 1 +

  (57)

2

4ηλ2tσt2 + Lt Lt

2Lt λtσt

- 1

- 2

wt λtσt

wt−1 1 +

Lt + ηwt =

min

wt

and it is achieved at

wt = 2λtσt Lt 4ηλ2tσt2 + Lt

(58)

This is exactly the choice of wtKL,η in Sec. 2.4. We note that such diffusion coefficient is well defined everywhere on [0,1] if Lt is also well defined everywhere, and as t → 1, it approaches a finite limit at L

2η . We also note that the integrand in Eq. (56) would still be ∞ at time 0 given the

t→1

λtσt, unless limt→0 Lt = 0. Theoretically, this is not an unreasonable assumption for both coefficients, as we know the closed form of v(x,0) = E[x˙0|xt=0 = x] =

1

α˙0E[x∗] and could optimize our model vθ(x,t) to directly approximate this value at t = 0. In practice, we found the numerical stability of wtKL,η could lead to better results.

### B Connection with Score-based Diffusion As shown in [64], the reverse-time SDE from Eq. (10) is

1 2

βtXt − βts(Xt,t)]dt + βtdW¯ t (59)

dXt = [−

Let us show this SDE is Eq. (4) for the specific choice wt = βt. To this end, notice that the solution Xt to Eq. (59) for the initial condition Xt=0 = x∗ with x∗ fixed is Gaussian distributed with mean and variance given respectively by

- 1

- 2

t

E[Xt] = e−

0 βsdsx∗ ≡ αtx∗ (60) var[Xt] = 1 − e−

t

0 βsds ≡ σt2 (61)

Using Eq. (44), the velocity of the score-based diffusion model can therefore be expressed as

- 1

- 2

- 1

- 2

- 1

- 2

t

t

βt(1 − e−

βte−

0 βsds) −

0 βsds)s(x,t) (62)

βtx + (−

v(x,t) = −

- 1

- 2

1 2

βts(x,t) (63) we see that 2λtσt is precisely βt, making λt correspond to the square of maximum

= −

βtx −

- likelihood weighting proposed in [64]. Further, if we plug Eq. (63) into Eq. (4), we arrive at Eq. (59).

A useful observation for choosing velocity versus noise model. We see that in the velocity model, all of the path-dependent terms (αt, σt) are inside the squared loss, and in the score model, the terms are pulled out (apart from the necessary σt in score matching loss) and get squared due to coming out of the norm. So which is more stable depends on the interpolant. In the paper we see that for SBDM-VP, due to the blowing up behavior of σ˙t near t = 0, both Lv and Lsλ are unstable.

for SBDM-VP, as the blowing up λt near t = 0 will compensate for the diminishing gradient inside the squared norm, where Lv would simply experience gradient explosion resulted from σ˙t. The behavior is different for the Linear and GVP interpolant, where the source of instability is αt−1 near t = 1. We note Lv is stable since αt−1 gets cancelled out inside the squared norm, while in Lsλ

Yet, shown in Tab. 3, we observed better performance with Lsλ

it remains in λt outside the norm.

### C Sampling with Guidance

Let pt(x|y) be the density of xt = αtx∗ + σtε conditioned on some extra variable y. By argument similar to the one given in Appendix A.1, it is easy to see that pt(x|y) satisfies the transport equation (compare Eq. (24))

∂tpt(x|y) + ∇x · (v(x,t|y)pt(x,|y)) = 0, (64)

where (compare Eq. (3))

v(x,t|y) = E[x˙t|xt = x,y] = α˙tE[x∗|xt = x,y] + σ˙tE[ε|xt = x,y] (65)

Proceeding as in Appendix A.3 and Appendix A.4, it is also easy to see that the score s(x,t|y) = ∇x log pt(x|y) is given by (compare Eq. (5))

s(x,t|y) = −σt−1E[ε|xt = x,y] (66) and that v(x,t|y) and s(x,t|y) are related via (compare Eq. (44))

α˙t αt

x − λtσts(x,t|y) (67)

v(x,t|y) =

Consider now

sζ(x,t|y) ≡ (1 − ζ)s(x,t) + ζs(x,t|y) (68)

- = ∇log pt(x) − ζ∇log pt(x) + ζ∇log pt(x|y) (69)
- = ∇log pt(x) − ζ∇log pt(x) + ζ∇log pt(y|x) + ζ∇log pt(x) (70)

= ∇log pt(x) + ζ∇log pt(y|x) (71) = ∇log[pt(x)pζt(y|x)] (72)

where we have used the fact ∇x log pt(x|y) = ∇x log pt(y|x) + ∇x log pt(x) that follows from pt(x|y)p(y) = pt(y|x)pt(x), and ζ to be some constant greater than 1. Eq. (72) shows that using the score mixture sζ(x,t|y) = (1−ζ)s(x,t)+ζs(x,t|y), and the velocity mixture associated with it, namely,

vζ(x,t|y) = (1 − ζ)v(x,t) + ζv(x,t|y) (73)

α˙t αt

x − λtσt[(1 − ζ)s(x,t) + ζs(x,t|y)] (74)

=

α˙t αt

x − λtσtsζ(x,t|y), (75)

=

allows one to to construct generative models that sample the tempered distribution pt(xt)pζt(y|xt) following classifier guidance [19]. Note that pt(x)pζt(y|x) ∝ pζt(x|y)p1t−ζ(x), so we can also perform classifier free guidance sampling [27]. Empirically, we observe significant performance boost by applying classifier free guidance, as showed in Tab. 1 and Tab. 7.

### D Sampling with ODE and SDE

In the main body of the paper, we used a second order Heun integrator for solving the ODE in Eq. (2) and a first order Euler-Maruyama integrator for solving the SDE in Eq. (4). We summarize all results in Tab. 8, and present the implementations below.

- Table 8: FID-50K scores produced by ODE and SDE. We demonstrate the comparison between ODE and SDE across all of our model sizes. All statistics are produced without classifier free guidance. Each cell in the table is showing [ODE results] / [SDE results]. We note the better performances of SDE observed in all model sizes are in line with the bounds given in [2], and that ODE has its advantage in lower NFE region, as shown in Fig. 5

Model Training Steps(K) FID↓ sFID↓ IS↑ Precision↑ Recall↑ SiT-S 400 58.97 / 57.64 8.95 / 9.05 23.34 / 24.78 0.40 / 0.41 0.59 / 0.60 SiT-B 400 34.84 / 33.02 6.59 / 6.46 41.53 / 43.71 0.52 / 0.53 0.64 / 0.63 SiT-L 400 20.01 / 18.79 5.31 / 5.29 67.76 / 72.02 0.62 / 0.64 0.64 / 0.64 SiT-XL 400 18.04 / 17.19 5.17 / 5.07 73.90 / 76.52 0.63 / 0.65 0.64 / 0.63 SiT-XL 7000 9.35 / 8.26 6.38 / 6.32 126.06 / 131.65 0.67 / 0.68 0.68 / 0.67

It is feasible to use either a velocity model vθ or a score model sθ in applying the above two samplers. If learning the score for the deterministic Heun sampler, we could always convert the learned sθ to vθ following Appendix A.4. However, as there exists potential numerical instability (depending on interpolants) in σ˙t, αt−1 and λt, it’s recommended to learn vθ in sampling with deterministic sampler instead of sθ. For the stochastic sampler, it’s required to have both vθ and sθ in integration, so we always need to convert from one (either learning velocity or score) to obtain the other. Under this scenario, the numerical issue from Appendix A.4 can only be avoided by clipping the time interval near t = 0. Empirically we found clipping the interval by h = 0.04 and doing a long last step from t = 0.04 to 0 can greatly benefit the performance. A detailed summary of sampler configuration is provided in Appendix E.

Additionally, we could replace vθ and sθ by vθζ and sζθ presented in Appendix C as inputs of the two samplers and enjoy the performance improvements coming along with guidance. As guidance requires evaluating both conditional and unconditional model output in a single step, it will impose twice the computational cost when sampling.

We also note that our models are compatible with more advanced samplers [37, 46]. We do not include the evaluations of those samplers in our work for the sake of direct comparison with the DDPM model, and we leave the investigation of potential performance improvements to future work.

Comparison between DDPM and Euler-Maruyama We primarily investigate and report the performance comparison between DDPM and Euler-Maruyama samplers. We set our Euler sampler’s number of steps to be 250 to match that of DDPM during evaluation. This comparison is made direct and fair, as the DDPM method can be viewed as a discretized version of Euler’s method.

Comparison between DDIM and Heun We also investigate the performance difference produced by deterministic samplers between DiT and our models. In Fig. 6, we show the FID-50K results for both DiT models sampled with

DDIM and SiT models sampled with Heun. We note that this is not directly an apples-to-apples comparison, as DDIM can be viewed as a discretized version of the first order Euler’s method, while we use the second order Heun’s method in sampling SiT models, due to the large discretization error with Euler’s method in continuous time. Nevertheless, we control the NFEs for both DDIM (250 sampling steps) and Heun (250 NFE).

80

SiT-B (ODE)

SiT-L (ODE)

SiT-XL (ODE)

SiT-S (ODE)

55

DiT-B (DDIM)

DiT-L (DDIM)

DiT-XL (DDIM)

30

DiT-S (DDIM)

75

35

50

70

30

25

45

FID-50K

FID-50K

FID-50K

FID-50K

65

25

40

60

20

20

35

55

15

15

30

50

25

45

10

200K 300K 400K 500K 600K 700K Training Steps

200K 300K 400K 500K 600K 700K Training Steps

200K 300K 400K 500K 600K 700K Training Steps

200K 300K 400K 500K 600K 700K Training Steps

- Fig. 6: SiT observes improvement in FID across all model sizes. We show FID-50K over training iterations for both DiT and SiT models. Across all model sizes, SiT converges faster. We acknowledge this is not directly an apples-to-apples comparison. This is because DDIM is essentially a discrete form of the first-order Euler’s method, whereas in sampling SiT, we employ the second-order Heun’s method. Nevertheless, both the SiT and DiT results are produced by a deterministic sampler with a 250 NFE.

Algorithm 1 Deterministic Heun Sampler

procedure HeunSampler(vθ(x, t, y), ti∈{0,···,N}, αt, σt)

sample x0 ∼ N(0, I) ▷ Generate initial sample ∆t ← t1 − t0 ▷ Determine fixed step size for i ∈ {0, · · · , N − 1} do

di ← vθ(xi, ti, y) x˜i+1 ← xi + ∆tdi ▷ Euler Step at ti di+1 ← vθ(x˜i+1, ti+1, y) xi+1 ← xi + ∆t2 [di + di+1] ▷ Explicit trapezoidal

rule at ti+1 end for return xN

end procedure

### E Additional Implementation Details

We implemented our models in JAX following the DiT PyTorch codebase by [50]4, and referred to [2]5, [64]6, and [20]7 for our implementation of the Euler-Maruyama

- sampler. For the Heun sampler, we directly used the one from diffrax [34]8, a JAX-based numerical differential equation solver library.

- 4 https://github.com/facebookresearch/DiT
- 5 https://github.com/malbergo/stochastic-interpolants
- 6 https://github.com/yang-song/score_sde
- 7 https://github.com/nv-tlabs/CLD-SGM
- 8 https://github.com/patrick-kidger/diffrax

Algorithm 2 Stochastic Euler-Maruyama Sampler

procedure EulerSampler(vθ(x, t, y), wt, ti∈{0,···,N}, T, αt, σt)

sample x0 ∼ N(0, I) ▷ Generate initial sample sθ ← convert from vθ following Appendix A.4

▷ Obtain ∇x log pt(x) in Eq. (4) ∆t ← t1 − t0 ▷ Determine fixed step size for i ∈ {0, · · · , N − 1} do

sample εi ∼ N(0, I) dεi ← εi ∗

√

∆t

di ← vθ(xi, ti, y) + 12wtisθ(xi, ti, y) ▷

Evaluate drift term at ti x¯i+1 ← xi + ∆tdi xi+1 ← x¯i+1 + √wtidεi ▷ Evaluate

diffusion term at ti end for h ← T − tN ▷ Last step size; T denotes the

time where xT = x∗

d ← vθ(xN, tN, y) + 12wtN sθ(xN, tN, y) x ← xN + h ∗ d ▷ Last step; output noiseless

sample without diffusion

return x end procedure

Architectural Configurations We follow the identical transformer architectures in DiT and have four different configurations: SiT-{S,B,L,XL}, varying in model size (parameters) and compute (flops). A detailed summarization is presented below.

- Table 9: Details of SiT models. We follow DiT [50] for the Small (S), Base (B), Large (L) and XLarge (XL) model configurations.

Model Layers N Hidden size d Heads SiT-S 12 384 6 SiT-B 12 768 12 SiT-L 24 1024 16 SiT-XL 28 1152 16

Training configurations We trained all of our models following identical structure and hyperparameters retained from DiT [50]. We used AdamW [35,45] as optimizer for all models. We use a constant learning rate of 1×10−4 and a batch size of 256. We used random horizontal flip with probability of 0.5 in data augmentation. We did not tune the learning rates, decay/warm up schedules, AdamW parameters, nor use any extra data augmentation or gradient clipping during training. Our

largest model, SiT-XL, trains at approximately 6.8 iters/sec on a TPU v4-64 pod following the above configurations. This speed is slightly faster compared to DiT-XL, which trains at 6.4 iters/sec under identical settings.

Sampling configurations We maintain an exponential moving average (EMA) of all models weights over training with a decay of 0.9999. All results are sampled from the EMA checkpoints, which is empirically observed to yield better performance. We summarize the start and end points of our deterministic and stochastic samplers with different interpolants below, where each t0 and tN are carefully tuned to optimize performance and avoid numerical instability during integration.

Table 10: Sampler configurations

Interpolant Model Objective Heun Euler-Maruyama t0 tN t0 tN

SBDM-VP velocity Lv 1 1e-5 1 4e-2

score Lsλ 1 1e-5 1 4e-2 GVP velocity Lv 1 0 1 4e-2

score Ls 1 - 1e-5 0 1 - 1e-3 4e-2 LIN velocity Lv 1 0 1 4e-2

score Ls 1 - 1e-5 0 1 - 1e-3 4e-2

FID calculation We calculate FID scores between generated images (10K or 50K) and all available real images in ImageNet training dataset. We observe small performance variations between TPU-based FID evaluation and GPU-based FID evaluation (ADM’s TensorFlow evaluation suite [19]9). To ensure consistency with the basline DiT, we sample all of our models on GPU and obtain FID scores using the ADM evaluation suite.

9 https://github.com/openai/guided-diffusion/tree/main/evaluations

### F Additional Visual results

[Figure 8]

##### Fig. 7: Uncurated 512 × 512 SiT-XL

- samples. Classifier-free guidance scale = 4.0 Class label = "volcano"(980)

[Figure 9]

Fig. 8: Uncurated 512 × 512 SiT-XL samples. Classifier-free guidance scale = 4.0 Class label = "arctic fox"(279)

[Figure 10]

Fig. 9: Uncurated 512 × 512 SiT-XL samples. Classifier-free guidance scale = 4.0 Class label = "loggerhead turtle"(33)

[Figure 11]

Fig. 10: Uncurated 512 × 512 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "balloon"(417)

[Figure 12]

Fig. 11: Uncurated 512 × 512 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "red panda"(387)

[Figure 13]

Fig. 12: Uncurated 512 × 512 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "geyser"(974)

[Figure 14]

Fig. 13: Uncurated 256 × 256 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "macaw"(88)

[Figure 15]

Fig. 14: Uncurated 256 × 256 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "golden retriever"(207)

[Figure 16]

Fig. 15: Uncurated 256 × 256 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = " ice cream"(928)

[Figure 17]

Fig. 16: Uncurated 256 × 256 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "cliff"(972)

[Figure 18]

Fig. 17: Uncurated 256 × 256 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "husky"(250)

[Figure 19]

Fig. 18: Uncurated 256 × 256 SiTXL samples. Classifier-free guidance scale = 4.0 Class label = "valley"(979)

