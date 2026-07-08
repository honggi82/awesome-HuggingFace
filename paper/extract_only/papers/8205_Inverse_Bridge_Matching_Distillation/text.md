## Inverse Bridge Matching Distillation

# arXiv:2502.01362v2[cs.LG]18Aug2025

Nikita Gushchin12 David Li*13 Daniil Selikhanovych*145 Evgeny Burnaev12 Dmitry Baranchuk4 Alexander Korotin12

### Abstract

Super-resolutionSketch-to-ImageInpaintingNormal-to-ImageJPEGrestoration

Learning diffusion bridge models is easy; making them fast and practical is an art. Diffusion bridge models (DBMs) are a promising extension of diffusion models for applications in image-toimage translation. However, like many modern diffusion and flow models, DBMs suffer from the problem of slow inference. To address it, we propose a novel distillation technique based on the inverse bridge matching formulation and derive the tractable objective to solve it in practice. Unlike previously developed DBM distillation techniques, the proposed method can distill both conditional and unconditional types of DBMs, distill models in a one-step generator, and use only the corrupted images for training. We evaluate our approach for both conditional and unconditional types of bridge matching on a wide set of setups, including super-resolution, JPEG restoration, sketch-to-image, and other tasks, and show that our distillation technique allows us to accelerate the inference of DBMs from 4x to 100x and even provide better generation quality than used teacher model depending on particular setup. We provide the code at https: //github.com/ngushchin/IBMD.

### 1. Introduction

Diffusion Bridge Models (DBMs) represent a specialized class of diffusion models designed for data-to-data tasks, such as image-to-image translation. Unlike standard diffusion models, which operate by mapping noise to data (Ho et al., 2020; Sohl-Dickstein et al., 2015), DBMs construct diffusion processes directly between two data distributions (Peluchetti, 2023a; Liu et al., 2022b; Somnath et al., 2023;

*Equal contribution 1Skolkovo Institute of Science and Technology 2Artificial Intelligence Research Institute 3Moscow Institute of Physics and Technology 4Yandex Research 5HSE University. Correspondence to: Nikita Gushchin <n.gushchin@skoltech.ru>, Alexander Korotin <a.korotin@skoltech.ru>.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

Input IBMD (Ours) Teacher

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Figure 1. Outputs of DBMs models distilled by our Inverse Bridge Matching Distillation (IBMD) approach on various image-toimage translation tasks and datasets ( 5). Teachers use NFE≥ 500 steps, while IBMD distilled models use NFE≤ 4.

Zhou et al., 2024a; Yue et al., 2024; Shi et al., 2023; De Bortoli et al., 2023). This approach allows DBMs to modify only the necessary components of the data, starting from an input sample rather than generating it entirely from Gaus-

sian noise. As a result, DBMs have demonstrated impressive performance in image-to-image translation problems.

The rapid development of DBMs has led to two dominant approaches, usually considered separately. The first branch of approaches (Peluchetti, 2023a; Liu et al., 2022b; 2023a; Shi et al., 2023; Somnath et al., 2023) considered the construction of diffusion between two arbitrary data distributions performing Unconditional Bridge Matching (also called the Markovian projection) of a process given by a mixture of diffusion bridges. The application of this branch includes different data like images (Liu et al., 2023a; Li et al., 2023), audio (Kong et al., 2025) and biological tasks (Somnath

- et al., 2023; Tong et al., 2024) not only in paired but also in unpaired setups using its relation to the Schr¨odinger Bridge problem (Shi et al., 2023; Gushchin et al., 2024). The second direction follows a framework closer to classical diffusion models, using forward diffusion to gradually map to the point of different distibution rather than mapping distribution to distribution as in previous case (Zhou et al., 2024a; Yue et al., 2024). While these directions differ in theoretical formulation, their practical implementations are closely related; for instance, models based on forward diffusion can be seen as performing Conditional Bridge Matching with additional drift conditions (De Bortoli et al., 2023).

Similar to classical DMs, DBMs also exhibit multi-step sequential inference, limiting their adoption in practice. Despite the impressive quality shown by DBMs in the practical tasks, only a few approaches were developed for their acceleration, including more advanced sampling schemes (Zheng

- et al., 2024; Wang et al., 2024) and consistency distillation (He et al., 2024), adapted for bridge models. While these approaches significantly improve the efficiency of DBMs, some unsolved issues remain. The first one is that the developed distillation approaches are directly applicable only for DBMs based on the Conditional Bridge Matching, i.e., no universal distillation method can accelerate any DBMs. Also, due to some specific theoretical aspects of DBMs, consistency distillation cannot be used to obtain the single-step model (He et al., 2024, Section 3.4).

Contributions. To address the above-mentioned issues of DBMs acceleration, we propose a new distillation technique based on the inverse bridge matching problem, which has several advantages compared to existing methods:

- 1. Universal Distillation. Our distillation technique is applicable to DBMs trained with both conditional and unconditional regimes, making it the first distillation approach introduced for unconditional DBMs.
- 2. Single-Step and Multi-step Distillation. Our distillation is capable of distilling DBMs into generators with any specified number of steps, including the distillation of DBMs into one-step generators.
- 3. Target data-free distillation. Our method does not

require the target data domain to perform distillation.

4. Better quality of distilled models. Our distillation technique is tested on a wide set of image-to-image problems for conditional and unconditional DBMs in both one and multi-step regimes. It demonstrates improvements compared to the previous acceleration approaches including DBIM (Zheng et al., 2024) and CDBM (He et al., 2024).

### 2. Background

In this paper, we propose a universal distillation framework for both conditional and unconditional DBMs. To not repeat fully analogical results for both cases, we denote by this color the additional conditioning on xT used for the conditional models, i.e. for the unconditional case this conditioning is not used.

#### 2.1. Bridge Matching

We start by recalling the bridge matching method (Peluchetti, 2023b;a; Liu et al., 2022b; Shi et al., 2023). Consider two probability distributions p(x0) and p(xT) on RD dimensional space, which represent target and source domains, respectively. For example, in an image inverse problem, p(x0) represents the distribution of clean images and p(xT) the distribution of corrupted images. Also consider a coupling p(x0,xT) of these two distributions, which is a probability distribution on RD × RD. Coupling p(x0,xT) can be provided by paired data or constructed synthetically, i.e., just using the independent distribution p(x0,xT) = p(x0)p(xT). Bridge Matching aims to construct the diffusion that transforms source distribution p(xT) to target distribution p(x0) based on given coupling p(x0,xT) and specified diffusion bridge.

Diffusion bridges. Consider forward-time diffusion Q called ”Prior” on time horizon [0,T] represented by the stochastic differential equation (SDE):

Prior Q : dxt = f(xt,t)dt + g(t)dwt, (1) f(xt,t) : RD × [0,T] → RD, g(t) : [0,T] → RD,

where f(xt,t) is a drift function, g(t) is the noise schedule function and dwt is the differential of the standard Wiener process. By q(xt|xs), we denote the transition probability density of prior process Q from time s to time t. Diffusion bridge is a conditional process Q|x

0,xT , which is obtained by pinning down starting and ending points x0 and xT. This diffusion bridge can be derived from prior process Q using the Doob-h transform (Doob & Doob, 1984):

0,xT : x0,xT are fixed, (2) dxt = {f(xt,t)dt + g2(t)∇xt

Diffusion Bridge Q|x

log q(xT|xt)}dt + g(t)dwt, For this diffusion bridge we denote the distribution at time t of the diffusion bridge Q|x

0,xT by q(xt|x0,xT).

[Figure 16]

- Figure 2. Overview of (Conditional) Bridge Matching with x0 reparameterization. The process begins by sampling a pair (x0, xT) from the data coupling p(x0, xT). An intermediate sample xt is then drawn from the diffusion bridge q(xt|x0, xT) at a random time t ∼ U[0, T]. The model x0 is trained with an MSE loss to reconstruct x0 from xt. In the conditional setting (dashed red path), x0 is also conditioned on xT as an additional input, leveraging information about the terminal state to improve reconstruction.

Mixture of bridges. Bridge Matching procedure starts with creating a mixture of bridges process Π. This process is represented as follows:

Mixture of Bridges Π : Π(·) = Q|x

0,xT (·)p(x0,xT)dx0dxT. (3)

Practically speaking, the definition (3) means that to sample from a mixture of bridges Π, one first samples the pair (x0,xT) ∼ p(x0,xT) from data coupling and then samples trajectory from the bridge Q|x

0,xT (·).

Bridge Matching problem. The mixture of bridges Π cannot be used for data-to-data translation since it requires first to sample a pair of data and then just inserts the trajectory. In turn, we are interested in constructing a diffusion, which can start from any sample xT ∼ p(xT) and gradually transform it to x0 ∼ p(x0). This can be done by solving the Bridge Matching problem (Shi et al., 2023, Proposition 2)

Time moment t here is sampled according to the uniform distribution on the interval [0,T].

Relation Between Flow and Bridge Matching. The Flow Matching (Liu et al., 2023b; Lipman et al., 2023) can be seen as the limiting case σ → 0 of the Bridge Matching for particular example see (Shi et al., 2023, Appendix A.1).

2.2. Augmented (Conditional) Bridge Matching and Denoising Diffusion Bridge Models (DDBM)

For a given coupling p(x0,xT) = p(x0|xT)p(xT), one can use an alternative approach to build a data-to-data diffusion. Consider a set of Bridge Matching problems indexed by xT between p0 = p(x0|xT) and p(xT) = δx

(x) (delta measure centered at xT). This approach is called Augmented Bridge Matching (De Bortoli et al., 2023). The key difference of this version in practice is that it introduces the condition of the drift function v∗(xt,t,xT) on the starting point xT in the reverse time diffusion (5):

T

Bridge Matching problem: (4) BM(Π) def= arg min

KL(Π||M),

M∈M

where M is the set of Markovian processes associated with some SDE and KL(Π||M) is the KL-divergence between a constructed mixture of bridges Π and diffusion M. It is known that the solution of Bridge Matching is the reversedtime SDE (Shi et al., 2023, Proposition 9):

The SDE of Bridge Matching solution : (5) dxt = {ft(xt) − g2(t)v∗(xt,t)}dt + g(t)dw¯t, xT ∼ pT(xT),

where w¯ is a standard Wiener process when time t flows backward from t = T to t = 0, and dt is an infinitesimal negative timestep. The drift function v∗ is obtained solving the following problem (Shi et al., 2023; Liu et al., 2023a):

Bridge Matching problem with a tractable objective: (6)

log q(xt|x0)∥2 , (x0,xT) ∼ p(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT).

Ex

0,t,xt ∥vϕ(xt,t) − ∇xt

min

ϕ

dxt = {ft(xt) − g2(t)v∗(xt,t,xT)}dt + g(t)dw¯t.

The drift function v∗ can be recovered almost in the same way just by the addition of this condition on xT:

Augmented (Conditional) Bridge Matching Problem. min

log q(xt|x0)∥2 , (x0,xT) ∼ p(x0,xT),and xt ∼ q(xt|x0,xT).

Ex

0,t,xt,xT ∥vϕ(xt,t,xT) − ∇xt

ϕ

Since the difference is the addition of conditioning on xT, we call this approach Conditional Bridge Matching.

Relation to DDBM. As was shown in the Augmented Bridge Matching (De Bortoli et al., 2023), the conditional Bridge Matching is equivalent to the Denoising Diffusion Bridge Model (DDBM) proposed in (Zhou et al., 2024a). The difference is that in DDBM, the authors learn the score function of s(xt,xT,t) conditioned on xT of a process for which x0 ∼ p(x0|xT) and q(xt) ∼ q(xt|x0,xT): Then, it is combined with the drift of forward Doob-h transform (5) to get the reverse SDE drift v(xt,t,xT):

v(xt,t,xT) = s(xt,xT,t) − ∇xt

log q(xT|xt),

dxt = {f(xt,t)dt − g2(t)v(xt,t,xT)}dt + g(t)dw¯t, or reverse probability flow ODE drift:

- 1

- 2

log q(xT|xt), dxt = {f(xt,t)dt − g2(t)vODE(xt,t,xT)}dt, which is used for consistency distillation in (He et al., 2024).

s(xt,xT,t) − ∇xt

vODE(xt,t,xT) =

#### 2.3. Practical aspects of Bridge Matching

Priors used in practice. In practice (He et al., 2024; Zhou et al., 2023; Zheng et al., 2024), the drift of the prior process is usually set to be f(xt,t) = f(t)xt, i.e, it depends linearly on xt. For this process the transitional distribution q(xt|x0) = N(xt|αtx0,σt2I) is Gaussian, where:

dσt2 dt − 2

dlog αt dt

dlog αt dt

, g2(t) =

σt2.

f(t) =

The bridge process distribution is also a Gaussian q(xt|x0,xT) = N(xT|atxT + btx0,c2tI) with coefficients:

SNRT SNRt

SNRT SNRt

αt αT

,bt = αt 1 −

at =

,

SNRT SNRt

c2t = σt2 1 −

,

2 t

where SNRt = α

σt2 is the signal-to-noise ratio at time t.

Data prediction reparameterization. The regression target of the loss function (6) for the priors with the drift v(xt,t) is given by ∇xt

t−αtx0

log q(xt|x0) = −x

σt2 . Hence, one can use the parametrization v(xt,t,xT) = −x

t−αt x0(xt,t,xT ) σt2

and solve the equivalent problem: Reparametrized (Conditional) Bridge Matching problem:

0,t,xt,xT λ(t)∥ xϕ0(xt,t,xT) − x0∥2 , (7) (x0,xT) ∼ p(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT), where λ(t) is any positive weighting function. Note that xT is used only for the Conditional Bridge Matching model.

Ex

min

ϕ

#### 2.4. Difference Between Acceleration of Unconditional and Conditional DBMs

Since both conditional and unconditional approaches learn drifts of SDEs, they share the same problems of long inference. However, these models significantly differ in the approaches that can accelerate them. The source of this difference is that Conditional Bridge Matching considers the set of problems of reversing diffusion, which gradually transforms distribution p(x0|xT) to the fixed point xT. Furthermore, the forward diffusion has simple analytical drift

and Gaussian transitional kernels. Thanks to it, for each xT to sample, one can use the probability flow ODE and ODEsolvers or hybrid solvers to accelerate sampling (Zhou et al., 2024a) or use consistency distillation of bridge models (He et al., 2024). Another beneficial property is that one can consider a non-Markovian forward process to develop a more efficient sampling scheme proposed in DBIM (Zheng et al., 2024) similar to Denoising Diffusion Implicit Models (Song et al., 2021). However, in the Unconditional Bridge Matching problem, the forward diffusion process, which maps p(x0) to p(xT) without conditioning on specific point xT, is unknown. Hence, the abovementioned methods cannot be used to accelerate this model type.

### 3. IBMD: Inverse Bridge Matching Distillation

This section describes our proposed universal approach to distill the both Unconditional and (Conditional) Bridge Matching models v∗ (called the teacher model) into a fewstep generator using only the corrupted data pT(xT). The key idea of our method is to consider the inverse problem of finding the mixture of bridges Πθ, for which Bridge Matching provides the solution vθ with the same drift as the given teacher model v∗. We formulate this task as the optimization problem ( 3.1). However, gradient methods cannot solve this optimization problem directly due to the absence of tractable gradient estimation. To avoid this problem, we prove a theorem that allows us to reformulate the inverse problem in the tractable objective for gradient optimization ( 3.2). Then, we present the fully analogical results for the Conditional Bridge Matching case in ( 3.3). Next, we present the multi-step version of distillation ( 3.5) and the final algorithm ( 3.4). We provide the proofs for all considered theorems and propositions in Appendix A.

#### 3.1. Bridge Matching Distillation as Inverse Problem

In this section, we focus on the derivation of our distillation method for the case of Unconditional Bridge Matching. Consider the fitted teacher model v∗(xt,t), which is an SDE drift of some process M∗ = BM(Π∗), where Π∗ constructed using some data coupling p∗(x0,xT) = p∗(x0|xT)p(xT). We parametrize pθ(x0,xT) = pθ(x0|xT)p(xT) and aim to find such Πθ build on pθ(x0,xT), that BM(Π∗) = BM(Πθ). In practice, we parametrize pθ(x0|xT) by the stochastic generator Gθ(xT,z),z ∼ N(0,I), which generates samples based on input xT ∼ p(xT) and the gaussian noise z. Now, we formulate the inverse problem as follows:

KL(BM(Πθ)||M∗). (8)

min

θ

Note, that since the objective (8) is the KL-divergence between BM(Πθ) and M∗, it is equal to 0 if and only if BM(Πθ) and M∗ coincide. Furthermore, using the disinte-

[Figure 17]

- Figure 3. Overview of our method Inverse Bridge Matching Distillation (IBMD). The goal is to distill a trained (Conditional) Bridge Matching model into a generator Gθ(z, xT), which learns to produce samples using the corrupted data p(xT). Generator Gθ(z, xT) defines the coupling pθ(x0, xT) = pθ(x0|xT)p(xT) and we aim to learn the generator in such way that Bridge Matching with pθ(x0, xT)

produces the same (Conditional) Bridge Matching model xϕ0 = xθ0. To do so, we learn a bridge model xϕ0 using coupling pθ in the same way as the teacher model was learned. Then, we use our novel objective given in Theorem 3.2 to update the generator model Gθ.

gration and Girsanov theorem (Vargas et al., 2021; Pavon & Wakolbinger, 1991), we have the following result:

Proposition 3.1 (Inverse Bridge Matching problem). The inverse problem (8) is equivalent to

t,t λ(t)||v(xt,t) − v∗(xt,t)||2 , s.t. (9) v = arg min

Ex

min

θ

t,t,x0 ∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 , (x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT), where λ(t) is any positive weighting function.

Ex

v′

Thus, this is the constrained problem, where the drift v is the result of Bridge Matching for coupling pθ(x0,xT) parametrized by the generator Gθ. Unfortunately, there is no clear way to use this objective efficiently for optimizing a generator Gθ since it would require gradient backpropagation through the argmin of the Bridge Matching problem.

#### 3.2. Tractable objective for the inverse problem

In this section, we introduce our new unconstrained reformulation for the inverse problem (9), which admits direct optimization using gradient methods:

Theorem 3.2 (Tractable inverse problem reformulation). The constrained inverse problem (9) w.r.t θ is equivalent to the unconstrained optimization problem:

t,t,x0 λ(t)∥v∗(xt,t) − ∇xt

log q(xt|x0)∥2 − min ϕ

Ex

min

θ

log q(xt|x0)∥2 , (x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT),

Ex

t,t,x0 λ(t)∥vϕ(xt,t) − ∇xt

Where the constraint in the original inverse problem (9) is relaxed by introducing the inner bridge matching problem.

This is the general result that can applied with any diffusion bridge. For the priors with with drift f(xt,t) = f(t)xt, we present its reparameterized version.

Proposition 3.3 (Reparameterized tractable inverse problem). Using the reparameterization ( 2.3) for the prior with the linear drift f(xt,t) = f(t)xt, the inverse problem in Theorem 3.2 is equivalent to:

t,t,x0 λ(t)∥ x∗0(xt,t) − x0∥2 − min ϕ

Ex

min

θ

t,t,x0 λ(t)∥ xϕ0(xt,t) − x0∥2 ,

Ex

(x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT). Thanks to the unconstrained reformulation, this problem admits explicit gradients with respect to the generator Gθ, as all samples (x0,xT,xt) are obtained via reparameterizable transformations: x0 = Gθ(xT,z) with z ∼ N(0,I), and xt ∼ q(xt | x0,xT), where q(xt | x0,xT) is a Gaussian distribution (under priors with linear drift f(xt,t) = f(t)xt). This enables differentiability of the entire objective, which involves an expectation over pθ(x0,xT), and allows optimization using standard gradient-based methods.

Interpretation of the auxiliary model ϕ. Note that the minimal value of the inner problem is the averaged variance of x0 ∼ pθ(x0 | xt,xT):

2

t,t,x0 λ(t) xϕ0(xt,t) − x0

Ex

min

=

ϕ

θ(x0|xt)[x0] − x0 2 = Ex

Ex

t,t,x0 λ(t) Ep

θ(x0|xt)[x0] − x0 2

t,t λ(t) Ep

θ(x0|xt) Ep

##### .

Variance of pθ(x0|xt)

For t = T, this is directly the variance of the generator x0 ∼ pθ(x0 | xT). Since this part comes with a negative

sign in the objective, its minimization enforces the generator to produce more diverse outputs and avoid collapsing.

#### 3.3. Distillation of conditional Bridge Matching models

Since Conditional Bridge Matching is, in essence, a set of Unconditional Bridge Matching problems for each xT ( 2.2), the analogical results hold just by adding the conditioning on xT for v, i.e., using v(xt,t,xT) or x0, i.e. using x0(xt,t,xT). Here, we provide the final reparametrized formulation, which we use in our experiments:

Theorem 3.4 (Reparameterized tractable inverse problem for conditional bridge matching).

t,t,x0,xT λ(t)∥ x∗0(xt,t,xT) − x0∥2 − (10) min ϕ

Ex

min

θ

t,t,x0,xT λ(t)∥ xϕ0(xt,t,xT) − x0∥2 , (x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT). where λ(t) is some positive weight function.

Ex

To use it in practice, we parameterize x0(xt,t,xT) by a neural network with an additional condition on xT.

#### 3.4. Algorithm

We provide a one-step Algorithm 1 that solves the inverse Bridge Matching problem in the reformulated version that we use in our experiments. We provide a visual abstract of it in Figure 3. Note that a teacher in the velocity parameterization v∗(xt,t) can be easily reparameterized ( 2.3) in x0-prediction model using x∗(xt,t) = σ

2 t v∗(xt,t)+xt

αt .

#### 3.5. Mulitistep distillation

We also present a multi-step modification of our distillation technique if a one-step generator struggles to distill the models, e.g., in inpainting setups, where the corrupted image xT contains less information. Our multi-step technique is inspired by similar approaches used in diffusion distillation methods (Yin et al., 2024a, DMD) and aims to avoid training/inference distribution mismatch.

We choose N timesteps {0 < t1 < t2 < ... < tN = T} and add additional time input to our generator Gθ(xt,z,t). For the conditional Bridge Matching case, we also add conditions on xT and use Gθ(xt,z,t,xT). To perform inference, we alternate between getting prediction from the generator x0 = Gθ(xt,z,t) and using posterior sampling q(xt

) given by the diffusion bridge. To train the generator in the multi-step regime, we use the same procedure as in one step except that to get input xt for intermediate times tn < tN, we first perform inference of our generator to get x0 and then use bridge q(xt| x0,xT).

n−1| x0,xt

n

### 4. Related work

Diffusion Bridge Models (DBMs) acceleration. Unlike a wide scope of acceleration methods developed for classical diffusion/flow models, only a few approaches were developed for DBM acceleration. Acceleration methods include more advanced samplers (Zheng et al., 2024; Wang et al., 2024) based on a reformulated forward diffusion process as a non-markovian process inspired by Denoising Diffusion Implicit Models (Song et al., 2021). Also, there is a distillation method based on the distilling probability-flow ODE into a few steps using consistency models (He et al., 2024), which is applicable only for conditional DBMs. However, for theoretical reasons (He et al., 2024, Section 3.4), consistency models for Diffusion Bridges cannot be distilled into one-step generators. Unlike existing distillation methods, our method is applicable to both conditional and unconditional DBMs and can distill into a one-step generator.

Related diffusion and flow models distillation techniques. Among the methods developed for the distillation of classical diffusion and flow models, the most related to our work are methods based on simultaneous training of few-step generators and auxiliary ”fake” model, that predict score or drift function for the generator (Yin et al., 2024b;a; Zhou et al., 2024b; Huang et al., 2024). Unlike these approaches, we consider the distillation of Diffusion Bridge Models the generalization of flow and diffusion models.

Furthermore, previous distillation methods for diffusion and flow models rely on marginal-based losses such as Fisher divergence, these approaches do not account for the full structure of path measures. This limitation becomes critical in the context of Diffusion Bridge Models (DBMs), where the dynamic aspects of the forward and reverse processes play a fundamental role. To better motivate the need for our KL-based objective, we next discuss the conceptual differences between KL divergence of path measures and Fisher divergence, illustrating why Fisher-based objectives like those used in SiD (Zhou et al., 2024b) are insufficient in the general setting of bridge matching. Consider two reverse-time diffusions D1 and D2 given by the same starting distribution p(xT) and SDEs:

D1 : dxt = v(xt,t)dt + g2(t)dw¯t, xT ∼ p(xT), D2 : dxt = v(xt,t)dt + g2(t)dw¯t, xT ∼ p(xT)

Let p(xt) and p(xt) be the corresponding marginals. Then the KL divergence and Fisher divergence are given by:

- 1

- 2g2(t)∥v(xt,t) − v(xt,t)∥2 +

KL(D1||D2) = Et,p

t(xt)

, (11)

KL(p(xT)|| p(xT)) =0 if p(xT )= p(xT )

log p(xt)∥2.

DFisher(D1||D2)=Et,p(x

t)∥∇xt

log p(xt) − ∇xt

In SiD (Zhou et al., 2024b), Fisher divergence (DFisher) is averaged over time and compares only marginal distributions p(xt) and p(xt) of path measures. However, path measures with the same marginal distributions might not be equal; thus, in general, minimizing Fisher divergence does not guarantee that D1 ≈ D2 as stochastic processes. For classical diffusion models, the forward drift f(xt,t) is fixed, and reverse drifts are fully determined by score functions:

v(xt,t) = f(xt,t) − g2(t)∇xt

log p(xt), v(xt,t) = f(xt,t) − g2(t)∇xt

log p(xt).

Substituting these into the KL expression (11) shows that in this specific setting — with a fixed forward SDE — KL divergence between path measures becomes equivalent (up to a constant) to the time-averaged Fisher divergence between the marginals. This explains why Fisher-based methods like SiD (Zhou et al., 2024b) may succeed in this context.

However, this equivalence breaks down in the case of unconditional bridge matching. Here, the forward drift f(xt,t) is not fixed and depends on the data coupling p(x0,xT). In turn, the forward drift fθ(xt,t) for the generated coupling pθ(x0,xT) also depends on θ. As a result, f(xt,t) ̸= fθ(xt,t), and the reverse drifts cannot be expressed solely in terms of marginal scores. Hence, KL divergence in the case of unconditional bridge matching is not equivalent to Fisher divergence between marginals. This difference is expected since, in the case of an unconditional diffusion bridge, one does not have a fixed forward process, which specifies the ”dynamic part” of the measure. This highlights the importance of using KL divergence between path measures as a high-level objective instead of the previously used Fisher Divergence.

- 5. Experiments This section highlights the applicability of our IBMD distillation method in both unconditional and conditional settings. To demonstrate this, we conducted experiments utilizing pretrained unconditional models used in I2SB paper (Liu et al., 2023a). Then we evaluated IBMD in conditional settings using DDBM (Zhou et al., 2024a) setup ( 5.2). For clarity, we denote our models as IBMD-DDBM and IBMD-I2SB, indicating that the teacher model is derived from DDBM or I2SB framework, respectively. We provide all the technical details in Appendix B.

- 5.1. Distillation of I2SB (5 setups) Since known distillation and acceleration techniques are designed for the conditional models, there is no clear baseline for comparison. Thus, this section aims to demonstrate that our distillation technique significantly decreases NFE required to obtain the same quality of generation.

Experimental Setup. To test our approach for unconditional models, we consider models trained and published in

Algorithm 1 Inverse Bridge Matching Distillation (IBMD) Input :Teacher network x∗0 : RD × [0,T] × RD → RD;

Bridge q(xt|x0,xT) used for training x∗; Generator network Gθ : RD × RD → RD; Bridge network xϕ0 : RD × [0,T] × RD → RD; Input distribution p(xT) accessible by samples; Weights function λ(t) : [0,T] → R+; Batch size N; Number of student iterations K; Number of bridge iterations L.

Output:Learned generator Gθ of coupling pθ(x0,xT) for

which Bridge Matching outputs drift v ≈ v∗. // Conditioning on xT is used only for distillation of Conditional Bridge Matching models.

- for k = 1 to K do

- for l = 1 to L do Sample batch xT ∼ p(xT)

Sample batch of noise z ∼ N(0,I) x0 ← Gθ(xT,z) Sample time batch t ∼ U[0,T] Sample batch xt ∼ q(xt|x0,xT) Lϕ ← N1 Nn=1 λ(t)|| xϕ0(xt,t,xT) − x0||2 n Update ϕ by using ∂∂ϕ Lϕ

Sample batch xT ∼ p(xT) Sample batch of noise z ∼ N(0,I) x0 ← Gθ(xT,z) Sample time batch t ∼ U[0,T] Sample batch xt ∼ q(xt|x0,xT) Lθ ←N1 Nn=1 λ(t)|| x∗0(xt,t,xT) − x0||2 − λ(t)|| xϕ0(xt,t,xT) − x0||2 n Update θ by using ∂ Lθ

∂θ

I2SB paper (Liu et al., 2023a), specifically (a) two models for the 4x super-resolution with bicubic and pool kernels, (b) two models for JPEG restoration using quality factor QF= 5 and QF= 10, and (c) a model for center-inpainting with a center mask of size 128 × 128 all of which were trained on ImageNet 256 × 256 dataset (Deng et al., 2009).

For all the setups we use the same train part of ImageNet dataset, which was used to train the used models. For the evaluation we follow the same protocol used in the I2SB paper, i.e. use the full validation subset of ImageNet for super-resolution task and the 10′000 subset of validation for other tasks. We report the same FID (Heusel et al., 2017) and Classifier Accuracy (CA) using pre-trained ResNet50 model metrics used in the I2SB paper. We present our results in Table 1, Table 3, Table 2, Table 4 and Table 6. We provide the uncurated samples for all setups in Appendix C.

Results. For both super-resolution tasks (see Table 1, Table 3), our 1-step distilled model outperformed teacher model inference using all 1000 steps used in the training.

- Table 1. Results on the image super-resolution task. Baseline results are taken from I2SB (Liu et al., 2023a).

4× super-resolution (bicubic) ImageNet (256 × 256) NFE FID ↓ CA ↑

DDRM (Kawar et al., 2022) 20 21.3 63.2 DDNM (Wang et al., 2023) 100 13.6 65.5 ΠGDM (Song et al., 2023) 100 3.6 72.1 ADM (Dhariwal & Nichol, 2021) 1000 14.8 66.7 CDSB (Shi et al., 2022) 50 13.6 61.0 I2SB (Liu et al., 2023a) 1000 2.8 70.7

IBMD-I2SB (Ours) 1 2.6 72.3

- Table 2. Results on the image JPEG restoration task with QF=5. Baseline results are taken from I2SB (Liu et al., 2023a).

JPEG restoration, QF= 5. ImageNet (256 × 256) NFE FID ↓ CA ↑

DDRM (Kawar et al., 2022) 20 28.2 53.9 ΠGDM (Song et al., 2023) 100 8.6 64.1 Palette (Saharia et al., 2022) 1000 8.3 64.2 CDSB (Shi et al., 2022) 50 38.7 45.7 I2SB (Liu et al., 2023a) 1000 4.6 67.9 I2SB (Liu et al., 2023a) 100 5.4 67.5

IBMD-I2SB (Ours) 1 5.2 66.6

Note that our model does not use the clean training target data p(x0), only the corrupted p(xT), hence this improvement is not due to additional training using paired data. We hypothesize that it is because the teacher model introduces approximation error during many steps of sampling, which may accumulate. For both JPEG restoration (see Table 2, Table 4), our 1-step distilled generator provides the quality of generation close to the teacher model and achieves around 100x time acceleration. For the inpainting problem (see Table 6), we present the results for 1,2 and 4 steps distilled generator. Our 2 and 4-step generators provide a quality similar to the teacher I2SB model, however, there is still some gap for the 1-step model. These models provide around 5x time acceleration. We hypothesize that this setup is harder since it requires to generate the entire center fragment from scratch, while in other tasks, there is already some good approximation given by corrupted images.

- 5.2. Distillation of DDBM (3 setups) This section addresses two primary objectives: (1) demonstrating the feasibility of conditional model distillation within our framework and (2) comparing with the CDBM (He et al., 2024) - a leading approach in Conditional Bridge Matching distillation, presented into different models: CBD (consistency distillation) and CBT (consistency training).

Experimental Setup. For evaluation, we use the same setups used in competing methods (He et al., 2024; Zheng

- Table 3. Results on the image super-resolution task. Baseline results are taken from I2SB (Liu et al., 2023a).

4× super-resolution (pool) ImageNet (256 × 256) NFE FID ↓ CA ↑

DDRM (Kawar et al., 2022) 20 14.8 64.6 DDNM (Wang et al., 2023) 100 9.9 67.1 ΠGDM (Song et al., 2023) 100 3.8 72.3 ADM (Dhariwal & Nichol, 2021) 1000 3.1 73.4 CDSB (Shi et al., 2022) 50 13.0 61.3 I2SB (Liu et al., 2023a) 1000 2.7 71.0

IBMD-I2SB (Ours) 1 2.5 72.5

- Table 4. Results on the image JPEG restoration task with QF=10. Baseline results are taken from I2SB (Liu et al., 2023a).

JPEG restoration, QF= 10. ImageNet (256 × 256) NFE FID ↓ CA ↑

DDRM (Kawar et al., 2022) 20 16.7 64.7 ΠGDM (Song et al., 2023) 100 6.0 71.0 Palette (Saharia et al., 2022) 1000 5.4 70.7 CDSB (Shi et al., 2022) 50 18.6 60.0 I2SB (Liu et al., 2023a) 1000 3.6 72.1 I2SB (Liu et al., 2023a) 100 4.4 71.6

IBMD-I2SB (Ours) 1 3.7 72.4

et al., 2024). For the image-to-image translation task, we utilize the Edges→Handbags dataset (Isola et al., 2017) with a resolution of 64 × 64 pixels and the DIODE-Outdoor dataset (Vasiljevic et al., 2019) with a resolution of 256×256 pixels. For these tasks, we report FID and Inception Scores (IS) (Barratt & Sharma, 2018). For the image inpainting task, we use the same setup of center-inpainting as before.

Results. We utilized the same teacher model checkpoints and as in CDBM. We present the quantitative and qualitative results of IBMD on the image-to-image translation task in

- Table 5 and in Figures 12, 10 respectively. The competing methods, DBIM (Zhou et al., 2024a, Section 4.1) and CDBM (He et al., 2024, Section 3.4), cannot use single-step inference due to the singularity at the starting point xT.

We trained our IBMD with 1 and 2 NFEs on the Edges→Handbags dataset. We surpass CDBM at 2 NFE, outperform the teacher at 100 NFE, and achieve performance comparable to the teacher at 50 NFE with 1 NFE, resulting in a 50× acceleration. For the DIODE-Outdoor setup, we trained IBMD with 1 and 2 NFEs. We surpassed CBD in FID at 2 NFE, achieving results comparable to CBT with a slight drop in performance and maintaining strong performance at 1 NFE with minor quality reductions.

For image inpainting, we show in Table 6 quantitative results and in Figure 9 the quantitative results. We train IBMD with 4 NFE in this setup. It outperforms CBD and CBT at 4

- Table 5. Results on the Image-to-Image Translation Task (Training Sets). Methods are grouped by NFE (> 2, 2, 1), with the best metrics bolded in each group. Baselines results are taken from CDBM.

NFE

Edges → Handbags (64 × 64) DIODE-Outdoor (256 × 256)

FID ↓ IS ↑ FID ↓ IS ↑ DDIB (Su et al., 2023) ≥ 40 186.84 2.04 242.3 4.22 SDEdit (Meng et al., 2022) ≥ 40 26.5 3.58 31.14 5.70 Rectified Flow (Liu et al., 2022a) ≥ 40 25.3 2.80 77.18 5.87 I2SB (Liu et al., 2023a) ≥ 40 7.43 3.40 9.34 5.77 DBIM (Zheng et al., 2024) 50 1.14 3.62 3.20 6.08 DBIM (Zheng et al., 2024) 100 0.89 3.62 2.57 6.06 CBD (He et al., 2024)

2

1.30 3.62 3.66 6.02 CBT (He et al., 2024) 0.80 3.65 2.93 6.06

- IBMD-DDBM (Ours) 0.67 3.69 3.12 5.92

Pix2Pix (Isola et al., 2017)

1

74.8 4.24 82.4 4.22

- IBMD-DDBM (Ours) 1.26 3.66 4.07 5.89

- Table 6. Results on the Image Inpainting Task. Methods are grouped by NFE (> 4, 4, 2, 1), with the best metrics bolded in each group. Baselines results are taken from CDBM.

model and teacher model on these sets in Figures 13 and 11 in Appendix C. We also provide the uncurated samples on the train part in Figures 12 and 10 to compare models’ behavior on train and test sets. From these results, we see that the teacher model exhibits overfitting on both setups, e.g., it produces exactly the same images as corresponding reference images. In turn, on the test sets, teacher models work well for the handbag setups, while on the test set of DIODE images, it exhibits mode collapse and produces gray images. Nevertheless, our distilled model shows exactly the same behavior in both sets, i.e., our IBMD approach precisely distills the teacher model as expected.

ImageNet (256 × 256) NFE FID ↓ CA ↑

Inpainting, Center (128 × 128)

DDRM (Kawar et al., 2022) 20 24.4 62.1 ΠGDM (Song et al., 2023) 100 7.3 72.6 DDNM (Wang et al., 2023) 100 15.1 55.9 Palette (Saharia et al., 2022) 1000 6.1 63.0 I2SB (Liu et al., 2023a) 10 5.4 65.97 DBIM (Zheng et al., 2024) 50 3.92 72.4 DBIM (Zheng et al., 2024) 100 3.88 72.6 CBD (He et al., 2024)

5.34 69.6

### 6. Discussion

- CBT (He et al., 2024) 4.77 70.3 IBMD-I2SB (Ours) 5.1 70.3 IBMD-DDBM (Ours) 4.03 72.2 CBD (He et al., 2024)

2

5.65 69.6

- CBT (He et al., 2024) 5.34 69.8 IBMD-I2SB (Ours) 5.3 65.7

4

Potential impact. DBMs are used for data-to-data translation in different domains, including images, audio, and biological data. Our distillation technique provides a universal and efficient way to address the long inference of DBMs, making them more affordable in practice.

- IBMD-DDBM (Ours) 4.23 72.3

IBMD-I2SB (Ours)

1

6.7 65.0

- IBMD-DDBM (Ours) 5.87 70.6

Limitations. Our method alternates between learning an additional bridge model and updating the student, which may be computationally expensive. Moreover, the student optimization requires backpropagation through the teacher, additional bridge, and the generator network, making it 3x time more memory expensive than training the teacher.

NFE with a significant gap, surpassing both at 2 NFE and maintaining strong performance at 1 NFE while achieving teacher-level results (with 50 NFE) with a 12.5× speedup.

### Acknowledgements

Concerns regarding the evaluation protocol used in prior works. For Edges-Handbags and DIODE-Outdoor setups, we follow the evaluation protocol originally introduced in DDBM (Zhou et al., 2024a) and later used in works on acceleration of DDBM (Zheng et al., 2024; He et al., 2024). For some reason, this protocol implies evaluation of the train set. Furthermore, test sets of these datasets consist of a tiny fraction of images (around several hundred), making the usage of standard metrics like FID challenging due to high statistical bias or variance of their estimation. Still, to assess the quality of the distilled model on the test sets, we provide the uncurated samples produced by our distill

The work was supported by the grant for research centers in the field of AI provided by the Ministry of Economic Development of the Russian Federation in accordance with the agreement 000000C313925P4F0002 and the agreement with Skoltech №139-10-2025-033.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Barratt, S. and Sharma, R. A note on the inception score. arXiv preprint arXiv:1801.01973, 2018.

De Bortoli, V., Liu, G.-H., Chen, T., Theodorou, E. A., and Nie, W. Augmented bridge matching. arXiv preprint arXiv:2311.06978, 2023.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Doob, J. L. and Doob, J. Classical potential theory and its probabilistic counterpart, volume 262. Springer, 1984.

Gushchin, N., Selikhanovych, D., Kholkin, S., Burnaev, E., and Korotin, A. Adversarial schr¨odinger bridge matching. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https:

//openreview.net/forum?id=L3Knnigicu.

He, G., Zheng, K., Chen, J., Bao, F., and Zhu, J. Consistency diffusion bridge models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Huang, Z., Geng, Z., Luo, W., and Qi, G.-j. Flow generator matching. arXiv preprint arXiv:2410.19310, 2024.

Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A. Image-toimage translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1125–1134, 2017.

Kawar, B., Elad, M., Ermon, S., and Song, J. Denoising diffusion restoration models. Advances in Neural Information Processing Systems, 35:23593– 23606, 2022.

Li, B., Xue, K., Liu, B., and Lai, Y.-K. Bbdm: Image-toimage translation with brownian bridge diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, pp. 1952–1961, 2023.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

net/forum?id=PqvMRDCJT9t.

Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E. A., Nie, W., and Anandkumar, A. I2sb: Image-to-image schr\” odinger bridge. The Fortieth International Conference on Machine Learning, 2023a.

Liu, X., Gong, C., et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2022a.

Liu, X., Wu, L., Ye, M., and qiang liu. Let us build bridges: Understanding and extending diffusion generative models. In NeurIPS 2022 Workshop on Score-Based Methods, 2022b. URL https://openreview.net/forum? id=0ef0CRKC9uZ.

Liu, X., Gong, C., and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023b. URL https:

//openreview.net/forum?id=XVjTT1nw5z.

Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., and Ermon, S. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Pavon, M. and Wakolbinger, A. On free energy, stochastic control, and schr¨odinger processes. In Modeling, Estimation and Control of Systems with Uncertainty: Proceedings of a Conference held in Sopron, Hungary, September 1990, pp. 334–348. Springer, 1991.

Peluchetti, S. Diffusion bridge mixture transports, schr¨odinger bridge problems and generative modeling. Journal of Machine Learning Research, 24(374):1–51, 2023a.

Peluchetti, S. Non-denoising forward-time diffusions. arXiv preprint arXiv:2312.14589, 2023b.

Kong, Z., Shih, K. J., Nie, W., Vahdat, A., Lee, S.g., Santos, J. F., Jukic, A., Valle, R., and Catanzaro, B. A2sb: Audio-to-audio schrodinger bridges. arXiv preprint arXiv:2501.11311, 2025.

Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., Fleet, D., and Norouzi, M. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pp. 1–10, 2022.

Shi, Y., De Bortoli, V., Deligiannidis, G., and Doucet,

- A. Conditional simulation using diffusion schr¨odinger bridges. In Uncertainty in Artificial Intelligence, pp. 1792–1802. PMLR, 2022.

Shi, Y., Bortoli, V. D., Campbell, A., and Doucet, A. Diffusion schr¨odinger bridge matching. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum? id=qy07OHsJT5.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Somnath, V. R., Pariset, M., Hsieh, Y.-P., Martinez, M. R., Krause, A., and Bunne, C. Aligned diffusion schr¨odinger bridges. In Uncertainty in Artificial Intelligence, pp. 1985–1995. PMLR, 2023.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=St1giarCHLP.

Song, J., Vahdat, A., Mardani, M., and Kautz, J. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023.

Su, X., Song, J., Meng, C., and Ermon, S. Dual diffusion implicit bridges for image-to-image translation. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

net/forum?id=5HLoTvVGDe.

Tong, A. Y., Malkin, N., Fatras, K., Atanackovic, L., Zhang, Y., Huguet, G., Wolf, G., and Bengio, Y. Simulationfree schr¨odinger bridges via score and flow matching. In International Conference on Artificial Intelligence and Statistics, pp. 1279–1287. PMLR, 2024.

Vargas, F., Thodoroff, P., Lamacraft, A., and Lawrence, N. Solving schr¨odinger bridges via maximum likelihood. Entropy, 23(9):1134, 2021.

Vasiljevic, I., Kolkin, N., Zhang, S., Luo, R., Wang, H., Dai, F. Z., Daniele, A. F., Mostajabi, M., Basart, S., Walter, M. R., et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.

Wang, Y., Yu, J., and Zhang, J. Zero-shot image restoration using denoising diffusion null-space model. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

net/forum?id=mRieQgMtNTQ.

Wang, Y., Yoon, S., Jin, P., Tivnan, M., Song, S., Chen, Z., Hu, R., Zhang, L., Chen, Z., Wu, D., et al. Implicit image-to-image schr¨odinger bridge for image restoration. Zhiqiang and Wu, Dufan, Implicit Image-to-Image Schr¨odinger Bridge for Image Restoration, 2024.

Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, W. T. Improved distribution matching distillation for fast image synthesis. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL https:

//openreview.net/forum?id=tQukGCDaNT.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.

Yue, Z., Wang, J., and Loy, C. C. Resshift: Efficient diffusion model for image super-resolution by residual shifting. Advances in Neural Information Processing Systems, 36, 2024.

Zheng, K., He, G., Chen, J., Bao, F., and Zhu, J. Diffusion bridge implicit models. In The Thirteenth International Conference on Learning Representations, 2024.

Zhou, L., Lou, A., Khanna, S., and Ermon, S. Denoising diffusion bridge models. In The Twelfth International Conference on Learning Representations, 2023.

- Zhou, L., Lou, A., Khanna, S., and Ermon, S. Denoising diffusion bridge models. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum? id=FKksTayvGo.

- Zhou, M., Zheng, H., Wang, Z., Yin, M., and Huang, H. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024b.

### A. Proofs

Since all our theorems, propositions and proofs for the inverse Bridge Matching problems which is formulated for the already trained teacher model using some diffusion bridge, we assume all corresponding assumptions used in Bridge Matching. Extensive overview of them can be found in (Shi et al., 2023, Appendix C).

Proof of Proposition 3.1. Since both BM(Πθ) and M∗ given by reverse-time SDE and the same distribution pT(xT) the KL-divergence expressed in the tractable form using the disintegration and Girsanov theorem (Vargas et al., 2021; Pavon & Wakolbinger, 1991):

KL(BM(Πθ)||M∗) = Ex

t,t g2(t)||v(xt,t) − v∗(xt,t)||2 , (x0,xT) ∼ pθ(x0,xT),t ∼ U([0,T]),xt ∼ q(xt|x0,xT).

The expectation is taken over the marginal distribution p(xt) of Πθ since it is the same as for BM(Πθ) (Shi et al., 2023, Proposition 2). In turn, the drift v(xt,t) is the drift of Bridge Matching using Πθ, i.e. BM(Πθ):

t,t,x0 ∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 ,

Ex

v = arg min

v′

(x0,xT) ∼ pθ(x0,xT),t ∼ U([0,T]),xt ∼ q(xt|x0,xT). Combining this, the inverse problem can be expressed in a more tractable form: min θ

t,t g2(t)||v(xt,t) − v∗(xt,t)||2 , s.t. (12) v = arg min

Ex

t,t,x0 ∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 dt,

Ex

v′

(x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT). We can add positive valued weighting function λ(t) for the constraint:

v = arg min

v′

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 dt,

Ex

since it is the MSE regression and its solution is conditional expectation for any weights given by: v(xt,t) = Ex

log q(xt|x0)]. We can add positive valued weighting function λ(t) for the main functional:

0|xt,t ∇xt

t,t λ(t)||v(xt,t) − v∗(xt,t)||2 ,

Ex

since it does not change the optimum value (which is equal to 0) and optimal solution, which is the mixture of bridges with the same drift as the teacher model.

| |
|---|

Proof of Theorem 3.2. Consider inverse bridge matching optimization problem: min

t,t λ(t)||v(xt,t) − v∗(xt,t)||2 , s.t. (13) v = arg min

Ex

θ

t,t,x0 ∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 , (x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT).

Ex

v′

First, note that since v = arg minv′ Ex

log q(xt|x0)∥2 , i.e. minimizer of MSE functional it is given by conditional expectation as:

t,t,x0 ∥v′(xt,t) − ∇xt

log q(xt|x0)|xt,t . (14) Then note that:

v(xt,t) = Ex

0|xt,t ∇xt

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 =

Ex

min

v′

log q(xt|x0)∥2 = Ex

Ex

t,t,x0 λ(t)∥v(xt,t) − ∇xt

t,t,x0 λ(t)||v(xt,t)||2

log q(xt|x0)||2 =

−2Ex

log q(xt|x0)⟩ + Ex

t,t,x0 λ(t)⟨v(xt,t),∇xt

t,t,x0 λ(t)||∇xt

Ext,t λ(t)||v(xt,t)||2

log q(xt|x0)||2 =

t,t λ(t)||v(xt,t)||2 − 2Ex

+ Ex

Ex

t,t λ(t) v(xt,t),Ex

t,t,x0 λ(t)||∇xt

0|xt,t ∇xt

log q(xt|x0)

=v(xt,t)

t,t λ(t)||v(xt,t)||2 − 2Ex

t,t λ(t)||v(xt,t)||2 + Ex

log q(xt|x0)||2 = −Ex

Ex

t,t,x0 λ(t)||∇xt

log q(xt|x0)||2 . (15) Hence, we derive that

t,t λ(t)||v(xt,t)||2 + Ex

t,t,x0 λ(t)||∇xt

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 . Now we use it to reformulate the initial objective:

t,t λ(t)||v(xt,t)||2 = Ex

log q(xt|x0)||2 − min

Ex

Ex

t,t,x0 λ(t)||∇xt

v′

t,t λ(t)||v(xt,t) − v∗(xt,t)||2 = Ex

Ex

t,t λ(t)||v∗(xt,t)||2 = Ex

t,t λ(t)⟨v(xt,t),v∗(xt,t)⟩ + Ex

t,t λ(t)||v(xt,t)||2 − 2Ex

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)||2] − min

log q(xt|x0)∥2

Ex

−

t,t,x0 λ(t)||∇xt

v′

=Ext,t λ(t)||v(xt,t)||2

t,t λ(t)||v∗(xt,t)||2 = Ex

t,t λ(t)⟨v(xt,t),v∗(xt,t)⟩ + Ex

2Ex

t,t λ(t)⟨v(xt,t),v∗(xt,t)⟩ + Ex

t,t λ(t)||v∗(xt,t)||2

log q(xt|x0)||2 − 2Ex

t,t,x0 λ(t)||∇xt

Ext,t,x0 λ(t)||v∗(xt,t)||2

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 Therefore, we get:

Ex

min

v′

−

t,t λ(t)||v(xt,t) − v∗(xt,t)||2 = Ex

Ex

t,t,x0 λ(t)||v∗(xt,t)||2 − min v′

t,t λ(t)⟨v(xt,t),v∗(xt,t)⟩ + Ex

log q(xt|x0)||2 − 2Ex

t,t,x0 λ(t)||∇xt

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2

Ex

To complete the proof, we use the relation v(xt,t) = Ex

log q(xt|x0)|xt,t from Equation 14. Integrating these components, we arrive at the final result:

0|xt,t ∇xt

t,t λ(t)||v(xt,t) − v∗(xt,t)||2 = Ex

Ex

t,t,x0 λ(t)||v∗(xt,t)||2 − min v′

log q(xt|x0)|xt,t ,v∗(xt,t) + Ex

log q(xt|x0)||2 − 2Ex

t,t λ(t) Ex

t,t,x0 λ(t)||∇xt

0|xt,t ∇xt

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 = Ex

Ex

t,t,x0 λ(t)||v∗(xt,t)||2 − min v′

log q(xt|x0),v∗(xt,t)⟩ + Ex

log q(xt|x0)||2 − 2Ex

t,t,x0 λ(t)||∇xt

t,t,x0 λ(t)⟨∇xt

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 = Ex

Ex

t,t,x0 λ(t)∥v∗(xt,t) − ∇xt

t,t,x0 λ(t)∥v′(xt,t) − ∇xt

log q(xt|x0)∥2 − min

log q(xt|x0)∥2 .

Ex

v′

| |
|---|

Proof of Proposition 3.3. Consider the problem from Proposition 3.2:

min

θ

t,t,x0 λ(t)∥v∗(xt,t) − ∇xt

log q(xt|x0)∥2 − min

Ex

ϕ

log q(xt|x0)∥2 ,

Ex

t,t,x0 λ(t)∥vϕ(xt,t) − ∇xt

t−αtx0

log q(xt|x0) = −x

For the priors with the drift f(t)x the regression target is ∇xt

σt2 . Hence one can use the parametrization v(xt,t) = −x

t−αt x0(xt,t)

σt2 We use reparameterization of both v∗ and vϕ given by:

xt − αt xϕ0(xt,t) σt2 and get:

xt − αt x∗0(xt,t) σt2

v∗(xt,t) = −

, vϕ(xt,t) = −

min

θ

t,t,x0 λ(t)∥v∗(xt,t) − ∇xt

log q(xt|x0)∥2 − min

Ex

Ex

ϕ

αt2 σt4

∥ x∗0(xt,t) − x0∥2 − min

Ex

Ex

min

t,t,x0 λ(t)

ϕ

θ

def=λ′(t)

min

θ

t,t,x0 λ′(t)∥ x∗0(xt,t) − x0∥2 − min

Ex

ϕ

log q(xt|x0)∥2 =

t,t,x0 λ(t)∥vϕ(xt,t) − ∇xt

αt2 σt4

∥ xϕ0(xt,t) − x0∥2 =

t,t,x0 λ(t)

def=λ′(t)

t,t,x0 λ′(t)∥ xϕ0(xt,t) − x0∥2 ,

Ex

where λ′(t) is just another positive weighting function.

| |
|---|

Proof of Theorem 3.4. In a fully analogical way, as for the unconditional case we consider the set of the Inverse Bridge Matching problems indexes by xT:

min

θ

KL(BM(Πθ|x

T

)||M|∗x

T

) x

,

T

where M|∗x

is a Mixture of Bridges for each xT constructed using bridge q(xt|x0,xT) and coupling pθ(x0|xT)δx

is a result of Bridge Matching conditioned on xT and Πθ|x

T

T

(x).

T

By employing the same reasoning as in the proof of Proposition 3.1, the inverse problem can be reformulated as follows:

t,t,xT g2(t)||v(xt,t,xT) − v∗(xt,t,xT)||2 , s.t. v = arg min

Ex

min

θ

t,t,x0,xT ∥v′(xt,t,xT) − ∇xt

log q(xt|x0)∥2 dt, (x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT).

Ex

v′

Following the proof of Theorem 3.2, we obtain a tractable formulation incorporating a weighting function:

t,t,x0,xT λ(t)∥v∗(xt,t,xT) − ∇xt

log q(xt|x0)∥2 − min ϕ

Ex

min

θ

log q(xt|x0)∥2 .

Ex

t,t,x0,xT λ(t)∥vϕ(xt,t,xT) − ∇xt

Utilizing the reparameterization under additional conditions ( 2.3), we obtain the following representations:

xt − αt xϕ0(xt,t,xT) σt2

xt − αt x∗0(xt,t,xT) σt2

v∗(xt,t,xT) = −

, vϕ(xt,t,xT) = −

.

Consequently, applying the proof technique from Proposition 3.3, we derive the final expression:

min

θ

t,t,x0 λ(t)∥ xϕ0(xt,t,xT) − x0∥2 , (x0,xT) ∼ pθ(x0,xT), t ∼ U([0,T]), xt ∼ q(xt|x0,xT).

t,t,x0 λ(t)∥ x∗0(xt,t,xT) − x0∥2 −min

Ex

Ex

ϕ

| |
|---|

|Task|Dataset<br><br>|Teacher<br><br>|NFE|L/K ratio<br><br>|LR|Grad Updates<br><br>|Noise|
|---|---|---|---|---|---|---|---|
|4× super-resolution (bicubic) 4× super-resolution (pool) JPEG restoration, QF = 5 JPEG restoration, QF = 10 Center-inpainting (128 × 128) Sketch to Image Sketch to Image Normal to Image Normal to Image Center-inpainting (128 × 128)|ImageNet ImageNet ImageNet ImageNet ImageNet Edges → Handbags Edges → Handbags DIODE-Outdoor DIODE-Outdoor ImageNet<br><br>|I2SB I2SB I2SB I2SB I2SB DDBM DDBM DDBM DDBM DDBM<br><br>|1 1 1<br><br>1 4<br><br>2<br><br><br>1<br><br>2 1 4<br><br><br>|5:1 5:1 5:1 5:1 5:1 5:1 5:1 5:1 5:1 1:1|5e-5 5e-5 5e-5 5e-5 5e-5 1e-5 1e-5 1e-5 1e-5 3e-6<br><br>|3000 3000 2000 3000 2000 300 14000 500 3700 3000<br><br>|✓ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✓<br><br>|

Table 7. Table entries specify experimental configurations: NFE indicates multi-step training (Sec. 3.5); L/K represents bridge/student gradient iteration ratios (Alg. 3.4); Grad Updates shows student gradient steps; Noise notes stochastic pipeline incorporation.

### B. Experimental details

All hyperparameters are listed in Table 7. We used batch size 256 and ema decay 0.99 for setups. For each setup, we started the student and bridge networks using checkpoints from the teacher models. In setups where the model adapts to noise: (1) We added extra layers for noise inputs (set to zero initially), (2) Noise was concatenated with input data before input it to the network. Datasets, code sources, and licenses are included in Table 8.

Training time. We present the training time of each in Table 9. About 75% of this training time is used to get the last 10-20% decrease of FID (e.g., drop from 3.6 to 2.5 FID in pooling SR setup or from 4.3 to 3.8 FID in JPEG with), while training for the first 25% of time already provides a good-quality model. On Sketch-to-image and Normal-to-image in multistep regime with 2 NFEs, convergence appears faster than in the corresponding single-step version.

Table 8. The used datasets, codes and their licenses.

|Name|URL<br><br>|Citation<br><br>|License|
|---|---|---|---|
|Edges→Handbags DIODE-Outdoor ImageNet Guided-Diffusion I2SB DDBM DBIM|GitHub Link Dataset Link Website Link GitHub Link GitHub Link GitHub Link GitHub Link<br><br>|(Isola et al., 2017) (Vasiljevic et al., 2019) (Deng et al., 2009) (Dhariwal & Nichol, 2021) (Liu et al., 2023a) (Zhou et al., 2023) (Zheng et al., 2024)<br><br>|BSD MIT \<br><br>MIT CC-BY-NC-SA-4.0<br><br>\ \|

|Task|Teacher<br><br>|Dataset|Approximate time on 8×A100<br><br>|NFE|
|---|---|---|---|---|
|4× super-resolution (bicubic) 4× super-resolution (pool) JPEG restoration, QF = 5 JPEG restoration, QF = 10 Center-inpainting (128×128) Center-inpainting (128×128) Sketch to Image Sketch to Image Normal to Image Normal to Image<br><br>|I2SB I2SB I2SB I2SB I2SB DDBM DDBM DDBM DDBM DDBM|Imagenet Imagenet Imagenet Imagenet Imagenet Imagenet<br><br>Edges/Handbags Edges/Handbags DIODE-Outdoor DIODE-Outdoor<br><br>|40 hours 40 hours 40 hours 40 hours 24 hours 12 hours 40 hours 1 hour 48 hours 7 hours<br><br>|1 1 1 1 4 4<br><br>1<br>2<br><br><br>1<br>2<br>|

Table 9. Training times and NFE across different tasks, teachers, and datasets.

#### B.1. Distillation of I2SB models.

We extended the I2SB repository (see Table 8), integrating our distillation framework. The following sections outline the setups, adapted following the I2SB.

Multi-step implementation In this setup, we use the student model’s full inference process during multi-step training (Section 3.5). This means that x0 is generated with inferenced of the model Gθ through all timesteps (T = tN,...,t1 = 0) in the multi-step sequence. The generated x0 is subsequently utilized in the computation of the bridge Lϕ or student Lθ objective functions, as formalized in Algorithm 1.

4× super-resolution. Our implementation of the degradation operators aligns with the filters implementation proposed in DDRM (Kawar et al., 2022). Firstly, we synthesize images at 64 × 64 resolution, then upsample them to 256 × 256 to ensure dimensional consistency between clean and degraded inputs. For evaluation, we follow established benchmarks (Saharia et al., 2022; Song et al., 2023) by computing the FID on reconstructions from the full ImageNet validation set, with comparisons drawn against the training set statistics.

JPEG restoration. Our JPEG degradation implementation, employing two distinct quality factors (QF=5, QF=10), follows (Kawar et al., 2022). FID is evaluated on a 10,000-image ImageNet validation subset against the full validation set’s statistics, following baselines (Saharia et al., 2022; Song et al., 2023).

Inpainting. For the image inpainting task on ImageNet at 256 × 256 resolution, we utilize a fixed 128 × 128 centrally positioned mask, aligning with the methodologies of DBIM (Zheng et al., 2024) and CDBM (He et al., 2024). During training, the model is trained only on the masked regions, while during generation, the unmasked areas are deterministically retained from the initial corrupted image xT to preserve structural fidelity of unmasked part of images. We trained the model with 4 NFEs via the multi-step method (Section 3.5) and tested it with 1,2, and 4 NFEs.

#### B.2. Distillation of DDBM models.

We extended the DDBM repository (Table 8) by integrating our distillation framework. Subsequent sections outline the experimental setups, adapted from the DDBM (Zheng et al., 2024).

Multi-step implementation In this setup, the multi-step training (Section 3.5) adopts the methodology of DMD (Yin et al., 2024a), wherein a timestep t is uniformly sampled from the predefined sequence (t1,...,tN). The model Gθ then generates x0 by iteratively reversing the process from the terminal timestep tN = T to the sampled intermediate timestep t. This generated x0 is subsequently used to compute the bridge network’s loss Lϕ or the student network’s loss Lθ, as detailed in Algorithm 1.

Edges → Handbags The model was trained utilizing the Edges→Handbags image-to-image translation task (Isola et al., 2017), with the 64 × 64 resolution images. Two versions were trained under the multi-step regime (Section 3.5), with 2 and 1 NFEs during training. Both models were evaluated using the same NFE to match training settings.

DIODE-Outdoor Following prior work (Zhou et al., 2023; Zheng et al., 2024; He et al., 2024), we used the DIODE outdoor dataset, preprocessed via the DBIM repository’s script for training/test sets (Table 8). Two versions were trained under the multi-step regime (Section 3.5), with 2 and 1 NFEs during training. Both models were evaluated using the same NFE to match training settings.

Inpainting All setups matched those in Section B.1 inpainting, except we use a CBDM checkpoint (Zheng et al., 2024). This checkpoint is adjusted by the authors to: (1) condition on xT and (2) ImageNet class labels as input to guide the model. Also this is the same checkpoint used in both CDBM (He et al., 2024) and DBIM (Zheng et al., 2024) works.

### C. Additional results

[Figure 18]

###### Figure 4. Uncurated samples for IBMD-I2SB distillation of 4x-super-resolution with bicubic kernel on ImageNet 256 × 256 images.

[Figure 19]

###### Figure 5. Uncurated samples for IBMD-I2SB distillation of 4x-super-resolution with pool kernel on ImageNet 256 × 256 images.

[Figure 20]

###### Figure 6. Uncurated samples for IBMD-I2SB distillation of Jpeg restoration with QF=5 on ImageNet 256 × 256 images.

[Figure 21]

###### Figure 7. Uncurated samples for IBMD-I2SB distillation of Jpeg restoration with QF=10 on ImageNet 256 × 256 images.

[Figure 22]

###### Figure 8. Uncurated samples for IBMD-I2SB distillation trained for inpaiting with NFE= 4 and inferenced with different inference NFE on ImageNet 256 × 256 images. 21

[Figure 23]

###### Figure 9. Uncurated samples for IBMD-DDBM distillation trained for inpaiting with NFE= 4 and inferenced with different inference NFE on ImageNet 256 × 256 images. 22

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

