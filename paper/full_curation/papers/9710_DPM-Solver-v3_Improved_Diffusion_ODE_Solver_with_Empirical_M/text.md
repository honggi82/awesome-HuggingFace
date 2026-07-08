# arXiv:2310.13268v3[cs.CV]28Oct2023

## DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics

Kaiwen Zheng∗†1, Cheng Lu∗1, Jianfei Chen1, Jun Zhu‡123 1Dept. of Comp. Sci. & Tech., Institute for AI, BNRist Center, THBI Lab 1Tsinghua-Bosch Joint ML Center, Tsinghua University, Beijing, China 2Shengshu Technology, Beijing 3Pazhou Lab (Huangpu), Guangzhou, China {zkwthu,lucheng.lc15}@gmail.com; {jianfeic, dcszj}@tsinghua.edu.cn

### Abstract

Diffusion probabilistic models (DPMs) have exhibited excellent performance for high-fidelity image generation while suffering from inefficient sampling. Recent works accelerate the sampling procedure by proposing fast ODE solvers that leverage the specific ODE form of DPMs. However, they highly rely on specific parameterization during inference (such as noise/data prediction), which might not be the optimal choice. In this work, we propose a novel formulation towards the optimal parameterization during sampling that minimizes the firstorder discretization error of the ODE solution. Based on such formulation, we propose DPM-Solver-v3, a new fast ODE solver for DPMs by introducing several coefficients efficiently computed on the pretrained model, which we call empirical model statistics. We further incorporate multistep methods and a predictorcorrector framework, and propose some techniques for improving sample quality at small numbers of function evaluations (NFE) or large guidance scales. Experiments show that DPM-Solver-v3 achieves consistently better or comparable performance in both unconditional and conditional sampling with both pixelspace and latent-space DPMs, especially in 5∼10 NFEs. We achieve FIDs of 12.21 (5 NFE), 2.51 (10 NFE) on unconditional CIFAR10, and MSE of 0.55 (5 NFE, 7.5 guidance scale) on Stable Diffusion, bringing a speed-up of 15%∼30% compared to previous state-of-the-art training-free methods. Code is available at https://github.com/thu-ml/DPM-Solver-v3.

### 1 Introduction

Diffusion probabilistic models (DPMs) [47, 15, 51] are a class of state-of-the-art image generators. By training with a strong encoder, a large model, and massive data as well as leveraging techniques such as guided sampling, DPMs are capable of generating high-resolution photorealistic and artistic images on text-to-image tasks. However, to generate high-quality visual content, DPMs usually require hundreds of model evaluations to gradually remove noise using a pretrained model [15], which is much more time-consuming compared to other deep generative models such as generative adversarial networks (GANs) [13]. The sampling overhead of DPMs emerges as a crucial obstacle hindering their integration into downstream tasks.

To accelerate the sampling process of DPMs, one can employ training-based methods [37, 53, 45] or training-free samplers [48, 51, 28, 3, 52, 56]. We focus on the latter approach since it requires no

†Work done during an internship at Shengshu Technology

*Equal contribution ‡Corresponding author

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

[Figure 1]

[Figure 2]

[Figure 3]

(a) DPM-Solver++ [32] (MSE 0.60) (b) UniPC [58] (MSE 0.65) (c) DPM-Solver-v3 (Ours, MSE 0.55)

- Figure 1: Random samples of Stable-Diffusion [43] with a classifier-free guidance scale 7.5, using only 5 number of function evaluations (NFE) and text prompt “A beautiful castle beside a waterfall in the woods, by Josef Thoma, matte painting, trending on artstation HQ”.

extra training and is more flexible. Recent advanced training-free samplers [56, 31, 32, 58] mainly rely on the ODE form of DPMs, since its absence of stochasticity is essential for high-quality samples in around 20 steps. Besides, ODE solvers built on exponential integrators [18] converge faster. To change the original diffusion ODE into the form of exponential integrators, they need to cancel its linear term and obtain an ODE solution, where only the noise predictor needs to be approximated, and other terms can be exactly computed. Besides, Lu et al. [32] find that it is better to use another ODE solution where instead the data predictor needs to be approximated. How to choose such model parameterization (e.g., noise/data prediction) in the sampling of DPMs remains unrevealed.

In this work, we systematically study the problem of model parameterization and ODE formulation for fast sampling of DPMs. We first show that by introducing three types of coefficients, the original ODE solution can be reformulated to an equivalent one that contains a new model parameterization. Besides, inspired by exponential Rosenbrock-type methods [19] and first-order discretization error analysis, we also show that there exists an optimal set of coefficients efficiently computed on the pretrained model, which we call empirical model statistics (EMS). Building upon our novel ODE formulation, we further propose a new high-order solver for diffusion ODEs named DPM-Solverv3, which includes a multistep predictor-corrector framework of any order, as well as some novel techniques such as pseudo high-order method to boost the performance at extremely few steps or large guidance scale.

We conduct extensive experiments with both pixel-space and latent-space DPMs to verify the effectiveness of DPM-Solver-v3. Compared to previous fast samplers, DPM-Solver-v3 can consistently improve the sample quality in 5∼20 steps, and make a significant advancement within 10 steps.

### 2 Background

#### 2.1 Diffusion Probabilistic Models

Suppose we have a D-dimensional data distribution q0(x0). Diffusion probabilistic models (DPMs) [47, 15, 51] define a forward diffusion process {qt}Tt=0 to gradually degenerate the data x0 ∼ q0(x0) with Gaussian noise, which satisfies the transition kernel q0t(xt|x0) = N(αtx0,σt2I), such that qT(xT) is approximately pure Gaussian. αt,σt are smooth scalar functions of t, which are called noise schedule. The transition can be easily applied by xt = αtx0 + σtϵ,ϵ ∼ N(0,I). To train DPMs, a neural network ϵθ(xt,t) is usually parameterized to predict the noise ϵ by minimizing Ex

0∼q0(x0),ϵ∼N(0,I),t∼U(0,T) w(t)∥ϵθ(xt,t) − ϵ∥22 , where w(t) is a weighting function. Sampling of DPMs can be performed by solving diffusion ODE [51] from time T to time 0:

g2(t) 2σt

dxt dt

ϵθ(xt,t), (1) where f(t) = d logα

= f(t)xt +

2 t

dt , g2(t) = dσ

dt − 2d logα

dt σt2 [23]. In addition, the conditional sampling by DPMs can be conducted by guided sampling [10, 16] with a conditional noise predictor ϵθ(xt,t,c), where c is the condition. Specifically, classifier-free guidance [16] combines the unconditional/con-

t

t

ditional model and obtains a new noise predictor ϵ′θ(xt,t,c) := sϵθ(xt,t,c) + (1 − s)ϵθ(xt,t,∅),

where ∅ is a special condition standing for the unconditional case, s > 0 is the guidance scale that controls the trade-off between image-condition alignment and diversity; while classifier guidance [10] uses an extra classifier pϕ(c|xt,t) to obtain a new noise predictor ϵ′θ(xt,t,c) := ϵθ(xt,t) − sσt∇x log pϕ(c|xt,t).

In addition, except for the noise prediction, DPMs can be parameterized as score predictor sθ(xt,t) to predict ∇x log qt(xt,t), or data predictor xθ(xt,t) to predict x0. Under variance-preserving (VP) noise schedule which satisfies αt2 + σt2 = 1 [51], “v” predictor vθ(xt,t) is also proposed to predict αtϵ − σtx0 [45]. These different parameterizations are theoretically equivalent, but have an impact on the empirical performance when used in training [23, 59].

#### 2.2 Fast Sampling of DPMs with Exponential Integrators

Among the methods for solving the diffusion ODE (1), recent works [56, 31, 32, 58] find that ODE solvers based on exponential integrators [18] are more efficient and robust at a small number of function evaluations (<50). Specifically, an insightful observation by Lu et al. [31] is that, by changeof-variable from t to λt := log(αt/σt) (half of the log-SNR), the diffusion ODE is transformed to

dxλ dλ

α˙λ αλ

xλ − σλϵθ(xλ,λ), (2) where α˙λ := dα

=

dλ . By utilizing the semi-linear structure of the diffusion ODE and exactly computing the linear term [56, 31], we can obtain the ODE solution as Eq. (3) (left). Such exact computation of the linear part reduces the discretization errors [31]. Moreover, by leveraging the equivalence of different parameterizations, DPM-Solver++ [32] rewrites Eq. (2) by the data predictor xθ(xλ,λ) := (xλ − σλϵθ(xλ,λ))/αλ and obtains another ODE solution as Eq. (3) (right). Such solution does not need to change the pretrained noise prediction model ϵθ during the sampling process, and empirically outperforms previous samplers based on ϵθ [31].

λ

λt

λt

xt αt

xs αs −

xs σs

xt σt

e−λϵθ(xλ,λ)dλ,

eλxθ(xλ,λ)dλ (3)

=

=

+

λs

λs

However, to the best of our knowledge, the parameterizations for sampling are still manually selected and limited to noise/data prediction, which are not well-studied.

### 3 Method

We now present our method. We start with a new formulation of the ODE solution with extra coefficients, followed by our high-order solver and some practical considerations. In the following discussions, we assume all the products between vectors are element-wise, and f(k)(xλ,λ) = dkf(xλ,λ)

dλk is the k-th order total derivative of any function f w.r.t. λ.

#### 3.1 Improved Formulation of Exact Solutions of Diffusion ODEs

As mentioned in Sec. 2.2, it is promising to explore the semi-linear structure of diffusion ODEs for fast sampling [56, 31, 32]. Firstly, we reveal one key insight that we can choose the linear part according to Rosenbrock-type exponential integrators [19, 18]. To this end, we consider a general form of diffusion ODEs by rewriting Eq. (2) as

dxλ dλ

=

α ˙λ αλ − lλ xλ

, (4)

##### −(σλϵθ(xλ,λ) − lλxλ)

linear part

non-linear part,:=Nθ(xλ,λ)

where lλ is a D-dimensional undetermined coefficient depending on λ. We choose lλ to restrict the Frobenius norm of the gradient of the non-linear part w.r.t. x:

lλ∗ = argmin

λ(xλ)∥∇xNθ(xλ,λ)∥2F, (5)

Epθ

lλ

where pθλ is the distribution of samples on the ground-truth ODE trajectories at λ (i.e., model distribution). Intuitively, it makes Nθ insensitive to the errors of x and cancels all the “linearty” of

Nθ. With lλ = lλ∗, by the “variation-of-constants” formula [1], starting from xλ

at time s, the exact solution of Eq. (4) at time t is

s

λt

λs lλdλ xλ

λt

λ

e−

λs lτdτ fθ(xλ,λ) approximated

dλ , (6)

s

−

#### xλ

= αλ

e

t

t

αλ

λs

s

- where fθ(xλ,λ) := N

θ(xλ,λ)

αλ . To calculate the solution in Eq. (6), we need to approximate fθ for each λ ∈ [λs,λt] by certain polynomials [31, 32].

Secondly, we reveal another key insight that we can choose different functions to be approximated instead of fθ and further reduce the discretization error, which is related to the total derivatives of the approximated function. To this end, we consider a scaled version of fθ i.e., hθ(xλ,λ) := e−

λ

λs sτdτfθ(xλ,λ) where sλ is a D-dimensional coefficient dependent on λ, and then Eq. (6) becomes

λt

λs lλdλ xλ

λt

λ

e−

λs(lτ+sτ)dτ hθ(xλ,λ) approximated

dλ . (7)

s

−

#### xλ

= αλ

e

t

t

αλ

λs

s

Comparing with Eq. (6), we change the approximated function from fθ to hθ by using an additional scaling term related to sλ. As we shall see, the first-order discretization error is positively related to

λ

the norm of the first-order derivative h(1)θ = e−

λs sτdτ(fθ(1) − sλfθ). Thus, we aim to minimize

∥fθ(1) − sλfθ∥2, in order to reduce ∥h(1)θ ∥2 and the discretization error. As fθ is a fixed function depending on the pretrained model, this minimization problem essentially finds a linear function of

- fθ to approximate fθ(1). To achieve better linear approximation, we further introduce a bias term

bλ ∈ RD and construct a function gθ satisfying gθ(1) = e−

λ

λs sτdτ(fθ(1) − sλfθ − bλ), which gives gθ(xλ,λ) := e−

λ

λs sτdτfθ(xλ,λ) −

λ

λs

e−

r

λs sτdτbrdr. (8) With gθ, Eq. (7) becomes

xλ

t

= αλ

t

e−

λt

λs lλdλ

linear coefficient

xλ

s

αλ

s

−

λt

λs

e

λ

λs(lτ+sτ)dτ

scaling coefficient

gθ(xλ,λ)

approximated

+

λ

λs

e−

r

λs sτdτbrdr

bias coefficient

dλ . (9)

Such formulation is equivalent to Eq. (3) but introduces three types of coefficients and a new parameterization gθ. We show in Appendix I.1 that the generalized parameterization gθ in Eq. (8) can cover a wide range of parameterization families in the form of ψθ(xλ,λ) = α(λ)ϵθ(xλ,λ) + β(λ)xλ +γ(λ). We aim to reduce the discretization error by finding better coefficients than previous works [31, 32].

Now we derive the concrete formula for analyzing the first-order discretization error. By replacing gθ(xλ,λ) with gθ(xλ

s

,λs) in Eq. (9), we obtain the first-order approximation xˆλ

t

= αλ

t

e−

λt

λs lλdλ xλs

αλs − λ λt

s

e

λ

λs(lτ+sτ)dτ gθ(xλ

s

,λs) + λ λ

s

e−

r

λs sτdτbrdr dλ . As

- gθ(xλ

,λs) = gθ(xλ,λ) + (λs − λ)gθ(1)(xλ,λ) + O((λ − λs)2) by Taylor expansion, it follows that the first-order discretization error can be expressed as xˆλ

s

λt

λt

λ

λs lτdτ(λ−λs) fθ(1)(xλ,λ)−sλfθ(xλ,λ)−bλ dλ +O(h3),

e−

λs lλdλ

t−xλ

= αλ

e

t

t

λs

(10) where h = λt − λs. Thus, given the optimal lλ = lλ∗ in Eq. (5), the discretization error xˆλ

t − xλ

t

mainly depends on fθ(1) − sλfθ − bλ. Based on this insight, we choose the coefficients sλ,bλ by solving

2 2

λ(xλ) fθ(1)(xλ,λ) − sλfθ(xλ,λ) − bλ

s∗λ,b∗λ = argmin

. (11)

Epθ

sλ,bλ

For any λ, lλ∗,s∗λ,b∗λ all have analytic solutions involving the Jacobian-vector-product of the pretrained model ϵθ, and they can be unbiasedly evaluated on a few datapoints {x(λn)}K ∼ pθλ(xλ) via MonteCarlo estimation (detailed in Section 3.4 and Appendix C.1.1). Therefore, we call lλ,sλ,bλ empirical model statistics (EMS). In the following sections, we’ll show that by approximating gθ with Taylor expansion, we can develop our high-order solver for diffusion ODEs.

###### xs

xˆt

- gs(0)
- gs(1)

(ts,gs)

estimation

(ti

,gi

)

1

1

gs(n)

(ti

,gi

)

n

n

(a) Local Approximation

t0 = 1 ti−2 ti−1 ti

Predictor

x0 xˆi−2 xˆi−1 xˆi ϵˆi−2

ϵˆ· := ϵθ(xˆ·,t·)

ϵˆi−1

ϵˆi

cache

|gˆi−2|
|---|

|gˆi−1|
|---|

|gˆi| |
|---|---|
| | |

gˆ· = a·,i−1xˆ· + b·,i−1ϵˆ· + c·,i−1

Corrector

xˆci

(b) Multistep Predictor-Corrector

- Figure 2: Illustration of our high-order solver. (a) (n + 1)-th order local approximation from time s to time t, provided n extra function values of gθ. (b) Multistep predictor-corrector procedure as our global solver. A combination of second-order predictor and second-order corrector is shown. a·,i−1, b·,i−1, c·,i−1 are abbreviations of coefficients in Eq. (8).

#### 3.2 Developing High-Order Solver

In this section, we propose our high-order solver for diffusion ODEs with local accuracy and global convergence order guarantee by leveraging our proposed solution formulation in Eq. (9). The proposed solver and analyses are highly motivated by the methods of exponential integrators [17, 18] in the ODE literature and their previous applications in the field of diffusion models [56, 31, 32, 58]. Though the EMS are designed to minimize the first-order error, they can also help our high-order solver (see Appendix I.2).

λt

λ

For simplicity, denote A(λs,λt) := e−

λs(lτ+sτ)dτ, Bλ

λs lτdτ, Eλ

(λ) := e

(λ) :=

s

s

r λs sτdτbrdr. Though high-order ODE solvers essentially share the same mathematical principles, since we utilize a more complicated parameterization gθ and ODE formulation in Eq. (9) than previous works [56, 31, 32, 58], we divide the development of high-order solver into simplified local and global parts, which are not only easier to understand, but also neat and general for any order.

λ λs e−

#### 3.2.1 Local Approximation

Firstly, we derive formulas and properties of the local approximation, which describes how we transit locally from time s to time t. It can be divided into two parts: discretization of the integral in Eq. (9) and estimating the high-order derivatives in the Taylor expansion.

#### Discretization. Denote gs(k) := gθ(k)(xλ

,λs). For n ≥ 0, to obtain the (n+1)-th order discretization

s

of Eq. (9), we take the n-th order Taylor expansion of gθ(xλ,λ) w.r.t. λ at λs: gθ(xλ,λ) = n k=0

(λ−λs)k

k! gs(k) + O((λ − λs)n+1). Substituting it into Eq. (9) yields

n

λt

λt

(λ − λs)k k!

xt αt

xs αs −

gs(k)

dλ +O(hn+2)

(λ)dλ−

= A(λs,λt)

#### Eλ

(λ)Bλ

#### Eλ

(λ)

s

s

s

λs

λs

k=0

(12) Here we only need to estimate the k-th order total derivatives gθ(k)(xλ

,λs) for 0 ≤ k ≤ n, since the other terms are determined once given λs,λt and lλ,sλ,bλ, which we’ll discuss next. High-order derivative estimation. For (n+1)-th order approximation, we use the finite difference of gθ(xλ,λ) at previous n+1 steps λi

s

,λs to estimate each gθ(k)(xλ

,λs). Such derivation is to match the coefficients of Taylor expansions. Concretely, denote δk := λi

,...,λi

n

1

s

), and the estimated high-order derivatives gˆs(k) can be solved by the following linear system:

k −λs, gi

:= gθ(xλ

,λi

ik

k

k

  

  

δ1 δ12 ··· δ1n

... .

. .

δn δn2 ··· δnn

  

   =

gˆs(1) . gˆs(n) n!

  

   (13)

1 − gs

gi

#### . gi

n − gs

Then by substituting gˆs(k) into Eq. (12) and dropping the O(hn+2) error terms, we obtain the (n+1)-th order local approximation:

xˆt αt

= A(λs,λt)

xs αs −

n

λt

gˆs(k)

(λ)dλ −

#### Eλ

(λ)Bλ

s

s

λs

k=0

λt

(λ − λs)k k!

dλ (14)

#### Eλ

(λ)

s

λs

- where gˆθ(0)(xλ

,λs) = gs. Eq. (13) and Eq. (14) provide an update rule to transit from time s to time t and get an approximated solution xˆt, when we already have the solution xs. For (n + 1)-th order approximation, we need n extra solutions xλ

s

. We illustrate the procedure in Fig. 2(a) and summarize it in Appendix C.2. In the following theorem, we show that under some assumptions, such local approximation has a guarantee of order of accuracy.

and their corresponding function values gi

ik

k

- Theorem 3.1 (Local order of accuracy, proof in Appendix B.2.1). Suppose xλ

ik

are exact (i.e., on the ground-truth ODE trajectory passing xs) for k = 1,...,n, then under some regularity conditions detailed in Appendix B.1, the local truncation error xˆt − xt = O(hn+2), which means the local approximation has (n + 1)-th order of accuracy.

Besides, we have the following theorem showing that, whatever the order is, the local approximation is unbiased given our choice of sλ,bλ in Eq. (11). In practice, the phenomenon of reduced bias can be empirically observed (Section 4.3).

- Theorem 3.2 (Local unbiasedness, proof in Appendix B.4). Given the optimal s∗λ,b∗λ in Eq. (11), For

the (n + 1)-th order approximation, suppose xλ

i1

,...,xλ

in

are on the ground-truth ODE trajectory passing xλ

s

, then Epθ

λs(xs) [xˆt − xt] = 0. 3.2.2 Global Solver

Given M + 1 time steps {ti}Mi=0, starting from some initial value, we can repeat the local approximation M times to make consecutive transitions from each ti−1 to ti until we reach an acceptable solution. At each step, we apply multistep methods [1] by caching and reusing the previous n values at timesteps ti−1−n,...,ti−2, which is proven to be more stable when NFE is small [32, 56]. Moreover, we also apply the predictor-corrector method [58] to refine the approximation at each step without introducing extra NFE. Specifically, the (n + 1)-th order predictor is the case of the local approximation when we choose (ti

n

,...,ti

1

,s,t) = (ti−1−n,...,ti−2,ti−1,ti), and the (n + 1)-th order corrector is the case when we choose (ti

n

,...,ti

1

,s,t) = (ti−n,...,ti−2,ti,ti−1,ti). We present our (n + 1)-th order multistep predictor-corrector procedure in Appendix C.2. We also illustrate a second-order case in Fig. 2(b). Note that different from previous works, in the local transition from ti−1 to ti, the previous function values gˆi

k

(1 ≤ k ≤ n) used for derivative estimation are dependent on i and are different during the sampling process because gθ is dependent on the current ti−1 (see Eq. (8)). Thus, we directly cache xˆi,ϵˆi and reuse them to compute gˆi in the subsequent steps. Notably, our proposed solver also has a global convergence guarantee, as shown in the following theorem. For simplicity, we only consider the predictor case and the case with corrector can also be proved by similar derivations in [58].

- Theorem 3.3 (Global order of convergence, proof in Appendix B.2.2). For (n+1)-th order predictor,

if we iteratively compute a sequence {xˆi}Mi=0 to approximate the true solutions {xi}Mi=0 at {ti}Mi=0, then under both local and global assumptions detailed in Appendix B.1, the final error |xˆM −xM| = O(hn+1), where | · | denotes the element-wise absolute value, and h = max1≤i≤M(λi − λi−1).

#### 3.3 Practical Techniques

In this section, we introduce some practical techniques that further improve the sample quality in the case of small NFE or large guidance scales.

Pseudo-order solver for small NFE. When NFE is extremely small (e.g., 5∼10), the error at each timestep becomes rather large, and incorporating too many previous values by high-order solver at each step will cause instabilities. To alleviate this problem, we propose a technique called pseudoorder solver: when estimating the k-th order derivative, we only utilize the previous k + 1 function

values of gθ, instead of all the n previous values as in Eq. (13). For each k, we can obtain gˆs(k) by

solving a part of Eq. (13) and taking the last element:

  

  

  

  , k = 1,2,...,n (15)

  

   =

· .

δ1 δ12 ··· δ1k

1 − gs

gi

... .

#### . gi

. .

gˆs(k) k!

δk δk2 ··· δkk

k − gs

In practice, we do not need to solve n linear systems. Instead, the solutions for gˆs(k),k = 1,...,n have a simpler recurrence relation similar to Neville’s method [36] in Lagrange polynomial interpolation. Denote i0 := s so that δ0 = λi

0 − λs = 0, we have

- Theorem 3.4 (Pseudo-order solver). For each k, the solution in Eq. (15) is gˆs(k) = k!D0(k), where

Dl(0) := gi

, l = 0,1,...,n

l

(16)

Dl(+1k−1) − Dl(k−1) δl+k − δl

Dl(k) :=

, l = 0,1,...,n − k

Proof in Appendix B.3. Note that the pseudo-order solver of order n > 2 no longer has the guarantee of n-th order of accuracy, which is not so important when NFE is small. In our experiments, we mainly rely on two combinations: when we use n-th order predictor, we then combine it with n-th order corrector or (n + 1)-th pseudo-order corrector.

Half-corrector for large guidance scales. When the guidance scale is large in guided sampling, we find that corrector may have negative effects on the sample quality. We propose a useful technique called half-corrector by using the corrector only in the time region t ≤ 0.5. Correspondingly, the strategy that we use corrector at each step is called full-corrector.

#### 3.4 Implementation Details

In this section, we give some implementation details about how to compute and integrate the EMS in our solver and how to adapt them to guided sampling.

Estimating EMS. For a specific λ, the EMS lλ∗,s∗λ,b∗λ can be estimated by firstly drawing K (1024∼4096) datapoints xλ ∼ pθλ(xλ) with 200-step DPM-Solver++ [32] and then analytically computing some terms related to ϵθ (detailed in Appendix C.1.1). In practice, we find it both convenient and effective to choose the distribution of the dataset q0 to approximate pθ0. Thus, without further specifications, we directly use samples from q0.

Estimating integrals of EMS. We estimate EMS on N (120 ∼ 1200) timesteps λj

,λj

,...,λj

0

1

N

and use trapezoidal rule to estimate the integrals in Eq. (12) (see Appendix I.3 for the estimation error analysis). We also apply some pre-computation for the integrals to avoid extra computation costs during sampling, detailed in Appendix C.1.2.

Adaptation to guided sampling. Empirically, we find that within a common range of guidance scales, we can simply compute the EMS on the model without guidance, and it can work for both unconditional sampling and guided sampling cases. See Appendix J for more discussions.

#### 3.5 Comparison with Existing Methods

By comparing with existing diffusion ODE solvers that are based on exponential integrators [56, 31, 32, 58], we can conclude that (1) Previous ODE formulations with noise/data prediction are special cases of ours by setting lλ,sλ,bλ to specific values. (2) Our first-order discretization can be seen as improved DDIM. See more details in Appendix A.

### 4 Experiments

In this section, we show that DPM-Solver-v3 can achieve consistent and notable speed-up for both unconditional and conditional sampling with both pixel-space and latent-space DPMs. We conduct extensive experiments on diverse image datasets, where the resolution ranges from 32 to 256. First, we present the main results of sample quality comparison with previous state-of-the-art training-free

30

| |Heun's (EDM)<br><br>DPM-Solver++<br><br>UniPC<br><br>DPM-Solver-v3| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

†DEIS

DPM-Solver++

18

DPM-Solver++

UniPC

25

25

16

UniPC

DPM-Solver-v3

DPM-Solver-v3

14

20

20

12

FID

FID

FID

15

15

10

10

8

10

6

5

5

4

0

5 6 8 10 12 15 20

5 6 8 10 12 15 20

5 6 8 10 12 15 20

NFE

NFE

NFE

(a) CIFAR10 (ScoreSDE, Pixel DPM)

(b) CIFAR10 (EDM, Pixel DPM)

(c) LSUN-Bedroom (Latent-Diffusion, Latent DPM)

- Figure 3: Unconditional sampling results. We report the FID↓ of the methods with different numbers of function evaluations (NFE), evaluated on 50k samples. †We borrow the results reported in their original paper directly.

5 6 8 10 12 15 20

NFE

8

10

12

14

16

FID

DPM-Solver++

UniPC

DPM-Solver-v3

(a) ImageNet-256 (Guided-Diffusion, Pixel DPM) (Classifier Guidance, s = 2.0)

5 6 8 10 12 15 20

NFE

0.05

0.10

0.15

0.20

0.25

RMSE

DPM-Solver++

UniPC

DPM-Solver-v3

(b) MS-COCO2014 (Stable-Diffusion, Latent DPM) (Classifier-Free Guidance, s = 1.5)

5 6 8 10 15 20

NFE

0.3

0.4

0.5

0.6

0.7

MSE

DPM-Solver++

UniPC

DPM-Solver-v3

(c) MS-COCO2014 (Stable-Diffusion, Latent DPM) (Classifier-Free Guidance, s = 7.5)

- Figure 4: Conditional sampling results. We report the FID↓ or MSE↓ of the methods with different numbers of function evaluations (NFE), evaluated on 10k samples.

methods. Then we illustrate the effectiveness of our method by visualizing the EMS and samples. Additional ablation studies are provided in Appendix G. On each dataset, we choose a sufficient number of timesteps N and datapoints K for computing the EMS to reduce the estimation error, while the EMS can still be computed within hours. After we obtain the EMS and precompute the integrals involving them, there is negligible extra overhead in the sampling process. We provide the runtime comparison in Appendix E. We refer to Appendix D for more detailed experiment settings.

#### 4.1 Main Results

We present the results in 5 ∼ 20 number of function evaluations (NFE), covering both few-step cases and the almost converged cases, as shown in Fig. 3 and Fig. 4. For the sake of clarity, we mainly compare DPM-Solver-v3 to DPM-Solver++ [32] and UniPC [58], which are the most state-of-the-art diffusion ODE solvers. We also include the results for DEIS [56] and Heun’s 2nd order method in EDM [21], but only for the datasets on which they originally reported. We don’t show the results for other methods such as DDIM [48], PNDM [28], since they have already been compared in previous works and have inferior performance. The quantitative results on CIFAR10 [24] are listed in Table 1, and more detailed quantitative results are presented in Appendix F.

Unconditional sampling We first evaluate the unconditional sampling performance of different methods on CIFAR10 [24] and LSUN-Bedroom [55]. For CIFAR10 we use two pixel-space DPMs,

- 0.5

- 1.0 lλ

1.00

lλ sλ bλ

1.5

sλ bλ

0.75

1.0

0.50

0.5

0.25

0.0

0.00

0.0

−0.5

−0.25

−0.5

−0.50

lλ sλ bλ

−1.0

−1.0

−0.75

−1.5

−1.5

−1.00

−4 −2 0 2 4 6

−2 −1 0 1 2 3

−4 −2 0 2 4 6

λ

λ

λ

(a) CIFAR10 (ScoreSDE)

(b) CIFAR10 (EDM)

(c) MS-COCO2014 (Stable-Diffusion, s = 7.5)

Figure 5: Visualization of the EMS lλ,sλ,bλ w.r.t. λ estimated on different models.

[Figure 4]

[Figure 5]

[Figure 6]

(a) DPM-Solver++ [32] (FID 18.59) (b) UniPC [58] (FID 12.24) (c) DPM-Solver-v3 (Ours) (FID 7.54)

Figure 6: Random samples of Latent-Diffusion [43] on LSUN-Bedroom [55] with only NFE = 5.

- Table 1: Quantitative results on CIFAR10 [24]. We report the FID↓ of the methods with different numbers of function evaluations (NFE), evaluated on 50k samples. †We borrow the results reported in their original paper directly.

NFE

Method Model

5 6 8 10 12 15 20 25 †DEIS [56]

15.37 \ \ 4.17 \ 3.37 2.86 \

DPM-Solver++ [32] 28.53 13.48 5.34 4.01 4.04 3.32 2.90 2.76 UniPC [58] 23.71 10.41 5.16 3.93 3.88 3.05 2.73 2.65 DPM-Solver-v3 12.76 7.40 3.94 3.40 3.24 2.91 2.71 2.64

ScoreSDE [51]

320.80 103.86 39.66 16.57 7.59 4.76 2.51 2.12 DPM-Solver++ [32] 24.54 11.85 4.36 2.91 2.45 2.17 2.05 2.02 UniPC [58] 23.52 11.10 3.86 2.85 2.38 2.08 2.01 2.00 DPM-Solver-v3 12.21 8.56 3.50 2.51 2.24 2.10 2.02 2.00

Heun’s 2nd [21]

EDM [21]

one is based on ScoreSDE [51] which is a widely adopted model by previous samplers, and another is based on EDM [21] which achieves the best sample quality. For LSUN-Bedroom, we use the latentspace Latent-Diffusion model [43]. We apply the multistep 3rd-order version for DPM-Solver++, UniPC and DPM-Solver-v3 by default, which performs best in the unconditional setting. For UniPC, we report the better result of their two choices B1(h) = h and B2(h) = eh − 1 at each NFE. For our DPM-Solver-v3, we tune the strategies of whether to use the pseudo-order predictor/corrector at each NFE on CIFAR10, and use the pseudo-order corrector on LSUN-Bedroom. As shown in Fig. 3, we find that DPM-Solver-v3 can achieve consistently better FID, which is especially notable when NFE is 5∼10. For example, we improve the FID on CIFAR10 with 5 NFE from 23 to 12 with ScoreSDE, and achieve an FID of 2.51 with only 10 NFE with the advanced DPM provided by EDM. On LSUN-Bedroom, with around 12 minutes computing of the EMS, DPM-Solver-v3 converges to the FID of 3.06 with 12 NFE, which is approximately 60% sampling cost of the previous best training-free method (20 NFE by UniPC).

Conditional sampling. We then evaluate the conditional sampling performance, which is more widely used since it allows for controllable generation with user-customized conditions. We choose two conditional settings, one is classifier guidance on pixel-space Guided-Diffusion [10] model trained on ImageNet-256 dataset [9] with 1000 class labels as conditions; the other is classifierfree guidance on latent-space Stable-Diffusion model [43] trained on LAION-5B dataset [46] with CLIP [41] embedded text prompts as conditions. We evaluate the former at the guidance scale of 2.0, following the best choice in [10]; and the latter at the guidance scale of 1.5 (following the original paper) or 7.5 (following the official code) with prompts random selected from MS-COCO2014 validation set [26]. Note that the FID of Stable-Diffusion samples saturates to 15.0∼16.0 even within 10 steps when the latent codes are far from convergence, possibly due to the powerful image decoder (see Appendix H). Thus, following [32], we measure the mean square error (MSE) between the generated latent code xˆ and the ground-truth solution x∗ (i.e., ∥xˆ−x∗∥22/D) to evaluate convergence, starting from the same Gaussian noise. We obtain x∗ by 200-step DPM-Solver++, which is enough to ensure the convergence.

We apply the multistep 2nd-order version for DPM-Solver++, UniPC and DPM-Solver-v3, which performs best in conditional setting. For UniPC, we only apply the choice B2(h) = eh − 1, which performs better than B1(h). For our DPM-Solver-v3, we use the pseudo-order corrector by default, and report the best results between using half-corrector/full-corrector on Stable-Diffusion (s = 7.5). As shown in Fig. 4, DPM-Solver-v3 can achieve better sample quality or convergence at most NFEs, which indicates the effectiveness of our method and techniques under the conditional setting. It’s worth noting that UniPC, which adopts an extra corrector, performs even worse than DPM-Solver++ when NFE<10 on Stable-Diffusion (s = 7.5). With the combined effect of the EMS and the halfcorrector technique, we successfully outperform DPM-Solver++ in such a case. Detailed comparisons can be found in the ablations in Appendix G.

#### 4.2 Visualizations of Estimated EMS

We further visualize the estimated EMS in Fig. 5. Since lλ,sλ,bλ are D-dimensional vectors, we average them over all dimensions to obtain a scalar. From Fig. 5, we find that lλ gradually changes from 1 to near 0 as the sampling proceeds, which suggests we should gradually slide from data prediction to noise prediction. As for sλ,bλ, they are more model-specific and display many fluctuations, especially for ScoreSDE model [51] on CIFAR10. Apart from the estimation error of

the EMS, we suspect that it comes from the fluctuations of ϵ(1)θ , which is caused by the periodicity of trigonometric functions in the positional embeddings of the network input. It’s worth noting that the

fluctuation of sλ,bλ will not cause instability in our sampler (see Appendix J).

#### 4.3 Visual Quality

We present some qualitative comparisons in Fig. 6 and Fig. 1. We can find that previous methods tend to have a small degree of color contrast at small NFE, while our method is less biased and produces more visual details. In Fig. 1(b), we can observe that previous methods with corrector may cause distortion at large guidance scales (in the left-top image, a part of the castle becomes a hill; in the left-bottom image, the hill is translucent and the castle is hanging in the air), while ours won’t. Additional samples are provided in Appendix K.

### 5 Conclusion

We study the ODE parameterization problem for fast sampling of DPMs. Through theoretical analysis, we find a novel ODE formulation with empirical model statistics, which is towards the optimal one to minimize the first-order discretization error. Based on such improved ODE formulation, we propose a new fast solver named DPM-Solver-v3, which involves a multistep predictor-corrector framework of any order and several techniques for improved sampling with few steps or large guidance scale. Experiments demonstrate the effectiveness of DPM-Solver-v3 in both unconditional conditional sampling with both pixel-space latent-space pre-trained DPMs, and the significant advancement of sample quality in 5∼10 steps.

Limitations and broader impact Despite the great speedup in small numbers of steps, DPM-Solverv3 still lags behind training-based methods and is not fast enough for real-time applications. Besides, we conducted theoretical analyses of the local error, but didn’t explore the global design spaces, such as the design of timestep schedules during sampling. And commonly, there are potential undesirable effects that DPM-Solver-v3 may be abused to accelerate the generation of fake and malicious content.

### Acknowledgements

This work was supported by the National Key Research and Development Program of China (No. 2021ZD0110502), NSFC Projects (Nos. 62061136001, 62106123, 62076147, U19A2081, 61972224, 62106120), Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University. J.Z is also supported by the XPlorer Prize.

### References

- [1] Kendall Atkinson, Weimin Han, and David E Stewart. Numerical solution of ordinary differential equations, volume 108. John Wiley & Sons, 2011.
- [2] Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, and Bo Zhang. Estimating the optimal covariance with imperfect mean in diffusion probabilistic models. In International Conference on Machine Learning, pages 1555–1584. PMLR, 2022.
- [3] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-DPM: An analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations, 2022.
- [4] Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad. An estimator for the diagonal of a matrix. Applied numerical mathematics, 57(11-12):1214–1229, 2007.
- [5] Mari Paz Calvo and César Palencia. A class of explicit multistep exponential integrators for semilinear problems. Numerische Mathematik, 102:367–381, 2006.
- [6] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In International Conference on Learning Representations, 2021.
- [7] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations, 2022.
- [8] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusionbased semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations, 2022.
- [9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255. IEEE, 2009.
- [10] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, volume 34, pages 8780– 8794, 2021.
- [11] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Genie: Higher-order denoising diffusion solvers. In Advances in Neural Information Processing Systems, 2022.
- [12] Alfredo Eisinberg and Giuseppe Fedele. On the inversion of the vandermonde matrix. Applied mathematics and computation, 174(2):1384–1397, 2006.
- [13] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27, pages 2672–2680, 2014.
- [14] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.
- [17] Marlis Hochbruck and Alexander Ostermann. Explicit exponential Runge-Kutta methods for semilinear parabolic problems. SIAM Journal on Numerical Analysis, 43(3):1069–1090, 2005.
- [18] Marlis Hochbruck and Alexander Ostermann. Exponential integrators. Acta Numerica, 19:209– 286, 2010.

- [19] Marlis Hochbruck, Alexander Ostermann, and Julia Schweitzer. Exponential rosenbrock-type methods. SIAM Journal on Numerical Analysis, 47(1):786–803, 2009.
- [20] Michael F Hutchinson. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 18(3):1059– 1076, 1989.
- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems, 2022.
- [22] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In Advances in Neural Information Processing Systems, 2022.
- [23] Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Advances in Neural Information Processing Systems, 2021.
- [24] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [25] Shengmeng Li, Luping Liu, Zenghao Chai, Runnan Li, and Xu Tan. Era-solver: Errorrobust adams solver for fast sampling of diffusion probabilistic models. arXiv preprint arXiv:2301.12935, 2023.
- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
- [27] Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, and Zhou Zhao. Diffsinger: Singing voice synthesis via shallow diffusion mechanism. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 11020–11028, 2022.
- [28] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2021.
- [29] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2022.
- [30] Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Maximum likelihood training for score-based diffusion odes by high order denoising score matching. In International Conference on Machine Learning, pages 14429–14460. PMLR, 2022.
- [31] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, 2022.
- [32] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpmsolver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
- [33] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.
- [34] Chenlin Meng, Ruiqi Gao, Diederik P Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In NeurIPS 2022 Workshop on Score-Based Methods, 2022.
- [35] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.
- [36] Eric Harold Neville. Iterative interpolation. St. Joseph’s IS Press, 1934.

- [37] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.
- [38] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pages 16784–16804. PMLR, 2022.
- [39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'AlchéBuc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.
- [40] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov, and Jiansheng Wei. Diffusion-based voice conversion with fast maximum likelihood sampling scheme. In International Conference on Learning Representations, 2022.
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [42] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022.
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022.
- [45] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.
- [46] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022.
- [47] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015.
- [48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.
- [49] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [50] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. In Advances in Neural Information Processing Systems, volume 34, pages 1415–1428, 2021.
- [51] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

- [52] Hideyuki Tachibana, Mocho Go, Muneyoshi Inahara, Yotaro Katayama, and Yotaro Watanabe. Itô-taylor sampling scheme for denoising diffusion probabilistic models using ideal derivatives. arXiv e-prints, pages arXiv–2112, 2021.
- [53] Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In International Conference on Learning Representations, 2022.
- [54] Suttisak Wizadwongsa and Supasorn Suwajanakorn. Accelerating guided diffusion sampling with splitting numerical methods. In The Eleventh International Conference on Learning Representations, 2022.
- [55] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. LSUN: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.
- [56] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In The Eleventh International Conference on Learning Representations, 2022.
- [57] Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. In Advances in Neural Information Processing Systems, 2022.
- [58] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: A unified predictorcorrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867, 2023.
- [59] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Improved techniques for maximum likelihood estimation for diffusion odes. arXiv preprint arXiv:2305.03935, 2023.

### A Related Work

Diffusion probabilistic models (DPMs) [47, 15, 51], also known as score-based generative models (SGMs), have achieved remarkable generation ability on image domain [10, 21], yielding extensive applications such as speech, singing and video synthesis [6, 27, 14], controllable image generation, translation and editing [38, 42, 43, 35, 57, 8], likelihood estimation [50, 23, 30, 59], data compression [23] and inverse problem solving [7, 22].

#### A.1 Fast Sampling Methods for DPMs

Fast sampling methods based on extra training or optimization include learning variances in the reverse process [37, 2], learning sampling schedule [53], learning high-order model derivatives [11], model refinement [29] and model distillation [45, 33, 49]. Though distillation-based methods can generate high-quality samples in less than 5 steps, they additionally bring onerous training costs. Moreover, the distillation process will inevitably lose part of the information of the original model, and is hard to be adapted to pre-trained large DPMs [43, 44, 42] and conditional sampling [34]. Some of distillation-based methods also lack the ability to make flexible trade-offs between sample speed and sample quality.

In contrast, training-free samplers are more lightweight and flexible. Among them, samplers based on diffusion ODEs generally require fewer steps than those based on diffusion SDEs [51, 40, 3, 48, 32], since SDEs introduce more randomness and make the denoising harder in the sampling process. Previous samplers handle the diffusion ODEs with different methods, such as Heun’s methods [21], splitting numerical methods [54], pseudo numerical methods [28], Adams methods [25] or exponential integrators [56, 31, 32, 58].

#### A.2 Comparison with Existing Solvers Based on Exponential Integrators

- Table 2: Comparison between DPM-Solver-v3 and other high-order diffusion ODE solvers based on exponential integrators.

DPM-Solver-v3 (Ours) First-Order DDIM DDIM DDIM DDIM Improved DDIM Taylor Expanded Predictor ϵθ for t ϵθ for λ xθ for λ xθ for λ gθ for λ Solver Type (High-Order) Multistep Singlestep Multistep Multistep Multistep Applicable for Guided Sampling ✓ ✗ ✓ ✓ ✓ Corrector Supported ✗ ✗ ✗ ✓ ✓ Model-Specific ✗ ✗ ✗ ✗ ✓

DEIS [56]

DPM-Solver [31]

DPM-Solver++ [32]

UniPC [58]

In this section, we make some theoretical comparisons between DPM-Solver-v3 and existing diffusion ODE solvers that are based on exponential integrators [56, 31, 32, 58]. We summarize the results in Table 2, and provide some analysis below.

Previous ODE formulation as special cases of ours. First, we compare the formulation of the ODE solution. Our reformulated ODE solution in Eq. (9) involves extra coefficients lλ,sλ,bλ, and it corresponds to a new predictor gθ to be approximated by Taylor expansion. By comparing ours with previous ODE formulations in Eq. (3) and their corresponding noise/data prediction, we can easily figure out that they are special cases of ours by setting lλ,sλ,bλ to specific values:

- • Noise prediction: lλ = 0,sλ = −1,bλ = 0
- • Data prediction: lλ = 1,sλ = 0,bλ = 0

It’s worth noting that, though our ODE formulation can degenerate to previous ones, it may still result in different solvers, since previous works conduct some equivalent substitution of the same order in the local approximation (for example, the choice of B1(h) = h or B2(h) = eh − 1 in UniPC [58]). We never conduct such substitution, thus saving the efforts to tune it.

Moreover, under our framework, we find that DPM-Solver++ is a model-agnostic approximation of DPM-Solver-v3, under the Gaussian assumption. Specifically, according to Eq. (5), we have

lλ∗ = argmin

lλ

λ(xλ)∥σλ∇xϵθ(xλ,λ) − lλ∥2F, (17)

Epθ

If we assume qλ(xλ) ≈ N(xλ|αλx0,σλ2I) for some fixed x0, then the optimal noise predictor is

xλ − αtx0 σλ

. (18)

ϵθ(xλ,λ) ≈ −σλ∇x log qλ(xλ) =

It follows that σλ∇xϵθ(xλ,λ) ≈ I, thus lλ∗ ≈ 1 by Eq. (5), which corresponds the data prediction model used in DPM-Solver++. Moreover, for small enough λ (i.e., t near to T), the Gaussian

assumption is almost true (see Section 4.2), thus the data-prediction DPM-Solver++ approximately computes all the linear terms at the initial stage. To the best of our knowledge, this is the first explanation for the reason why the data-prediction DPM-Solver++ outperforms the noise-prediction DPM-Solver.

First-order discretization as improved DDIM Previous methods merely use noise/data parameterization, whether or not they change the time domain from t to λ. While they differ in high-order cases, they are proven to coincide in the first-order case, which is DDIM [48] (deterministic case, η = 0):

αt αs

xˆs − αt

xˆt =

However, the first-order case of our method is

σs αs −

σt αt

ϵθ(xˆs,λs) (19)

λt

αt αs

(λ)dλ x ˆs − σs

xˆt =

A(λs,λt) 1 + lλ

#### Eλ

s

s

λs

λt

− αtA(λs,λt)

#### Eλ

(λ)Bλ

(λ)dλ

s

s

λs

λt

(λ)dλ ϵθ(xˆs,λs)

#### Eλ

s

λs

(20)

which is not DDIM since we choose a better parameterization by the estimated EMS. Empirically, our first-order solver performs better than DDIM, as detailed in Appendix G.

### B Proofs

#### B.1 Assumptions

In this section, we will give some mild conditions under which the local order of accuracy of Algorithm 1 and the global order of convergence of Algorithm 2 (predictor) are guaranteed.

- B.1.1 Local First, we will give the assumptions to bound the local truncation error.

- Assumption B.1. The total derivatives of the noise prediction model d

kϵθ(xλ,λ)

dλk ,k = 1,...,n exist and are continuous.

- Assumption B.2. The coefficients lλ,sλ,bλ are continuous and bounded. d

klλ

dλk , d

ksλ

dλk , d

kbλ

dλk ,k = 1,...,n exist and are continuous.

- Assumption B.3. δk = Θ(λt − λs),k = 1,...,n

Assumption B.1 is required for the Taylor expansion which is regular in high-order numerical methods. Assumption B.2 requires the boundness of the coefficients as well as regularizes the coefficients’ smoothness to enable the Taylor expansion for gθ(xλ,λ), which holds in practice given the smoothness of ϵθ(xλ,λ) and pθλ(xλ). Assumption B.3 makes sure δk and λt − λs is of the same order, i.e., there exists some constant rk = O(1) so that δk = rk(λt − λs), which is satisfied in regular multistep methods.

- B.1.2 Global Then we will give the assumptions to bound the global error.

- Assumption B.4. The noise prediction model ϵθ(x,t) is Lipschitz w.r.t. to x.
- Assumption B.5. h = max1≤i≤M(λi − λi−1) = O(1/M).
- Assumption B.6. The starting values xˆi,1 ≤ i ≤ n satisfies xˆi − xi = O(hn+1).

- Assumption B.4 is common in the analysis of ODEs, which assures ϵθ(xˆt,t)−ϵθ(xt,t) = O(xˆt−xt).
- Assumption B.5 implies that the step sizes are rather uniform. Assumption B.6 is common in the convergence analysis of multistep methods [5].

#### B.2 Order of Accuracy and Convergence

In this section, we prove the local and global order guarantees detailed in Theorem 3.1 and Theorem 3.3.

#### B.2.1 Local

Proof. (Proof of Theorem 3.1) Denote h := λt − λs. Subtracting the Taylor-expanded exact solution in Eq. (12) from the local approximation in Eq. (14), we have

n

gˆθ(k)(xλ

,λs) − gθ(k)(xλ

λt

,λs) k!

(λ)(λ−λs)kdλ+O(hn+2)

s

s

xˆt −xt = −αtA(λs,λt)

#### Eλ

s

λs

k=1

(21) First we examine the order of A(λs,λt) and λ λt

(λ)(λ − λs)kdλ. Under Assumption B.2, there exists some constant C1,C2 such that −lλ < C1,lλ + sλ < C2. So

#### Eλ

s

s

λt

A(λs,λt) = e−

λs lτdτ < eC

(22)

1h

= O(1)

λt

λt

λ

λs(lτ+sτ)dτ(λ − λs)kdλ

(λ)(λ − λs)kdλ =

#### Eλ

e

s

λs

λs

(23)

λt

2(λ−λs)(λ − λs)kdλ

eC

<

λs

= O(hk+1)

(k) θ (xλs,λs)−gθ(k)(xλs,λs)

Next we examine the order of gˆ

k! . Under Assumption B.1 and Assumption B.2, since gθ is elementary function of ϵθ and lλ,sλ,bλ, we know gθ(k)(xλ

,λs),k = 1,...,n exist and are continuous. Adopting the notations in Eq. (13), by Taylor expansion, we have

s

= gs + δ1gs(1) + δ12gs(2) + ··· + δ1ngs(n) + O(δ1n+1) gi

#### gi

1

= gs + δ2gs(1) + δ22gs(2) + ··· + δ2ngs(n) + O(δ2n+1)

2

... gi

= gs + δngs(1) + δn2gs(2) + ··· + δnngs(n) + O(δnn+1)

n

Comparing it with Eq. (13), we have









- gˆs(1) − gs(1)
- gˆs(2)−gs(2) 2!

  

  

- O(δ1n+1)
- O(δ2n+1)

- δ1 δ12 ··· δ1n
- δ2 δ22 ··· δ2n

=

 

 

... .

 

 

. .

. O(δnn+1)

.

δn δn2 ··· δnn

gˆs(n)−gs(n) n!

(24)

(25)

From Assumption B.3, we know there exists some constants rk so that δk = rkh,k = 1,...,n. Thus





  

  

  

  ,

  

  

   =

  

- O(δ1n+1)
- O(δ2n+1)

O(hn+1) O(hn+1)

- r1 r12 ··· r1n
- r2 r22 ··· r2n

- δ1 δ12 ··· δ1n
- δ2 δ22 ··· δ2n

h

h2

=

 

 

...

... .

... .

. O(hn+1)

. .

. .

. O(δnn+1)

hn

rn rn2 ··· rnn

δn δn2 ··· δnn

(26) And finally, we have





- gˆs(1) − gs(1)
- gˆs(2)−gs(2) 2!

  

−1 

  

  

  

  

h−1

O(hn+1) O(hn+1)

r1 r12 ··· r1n r2 r22 ··· r2n

h−2

 

=

... .

...

 

 

. O(hn+1)

. .

.

h−n

rn rn2 ··· rnn

gˆs(n)−gs(n) n!

(27)

  

  

O(hn) O(hn−1)

=

. O(h1)

Substitute Eq. (22), Eq. (23) and Eq. (27) into Eq. (21), we can conclude that xˆt−xt = O(hn+2).

| |
|---|

#### B.2.2 Global

First, we provide a lemma that gives the local truncation error given inexact previous values when estimating the high-order derivatives.

- Lemma B.7. (Local truncation error with inexact previous values) Suppose inexact values xˆλ

,k =

ik

1,...,n and xˆs are used in Eq. (13) to estimate the high-order derivatives, then the local truncation error of the local approximation Eq. (14) satisfies

n

αtA(λs,λt) αs

) + O(hn+1) (28)

O(∆λ

∆s + O(h) O(∆s) +

∆t =

ik

k=1

where ∆· := xˆ· − x·,h := λt − λs. Proof. By replacing x· with xˆ· in Eq. (13) and subtracting Eq. (12) from Eq. (14), the expression for the local truncation error becomes

λt

αtA(λs,λt) αs

∆s − αtA(λs,λt)(gθ(xˆλ

,λs) − gθ(xλ

∆t =

,λs))

#### Eλ

(λ)dλ

s

s

s

λs

(29)

n

gˆθ(k)(xλ

,λs) − gθ(k)(xλ

λt

,λs) k!

(λ)(λ − λs)kdλ + O(hn+2)

s

s

− αtA(λs,λt)

#### Eλ

s

λs

k=1

And the linear system for solving gθ(k)(xλ

,λs),k = 1,...,n becomes

s





- gˆs(1)
- gˆs(2) 2!

  

  

  

   (30)

- δ1 δ12 ··· δ1n
- δ2 δ22 ··· δ2n

- 1 − gˆs

gˆi

- 2 − gˆs

gˆi

=

... .

 

 

. gˆi

. .

.

δn δn2 ··· δnn

n − gˆs

gˆs(n) n!

where gˆ· = gθ(xˆλ

). Thus, under Assumption B.1, Assumption B.2 and Assumption B.3, similar to the deduction in the last section, we have 

,λ·). Under Assumption B.4, we know gˆ· − g· = O(∆λ

·

·





−1 

- gˆs(1) − gs(1)
- gˆs(2)−gs(2) 2!

  

  

  

  

O(hn+1 + ∆s + ∆λ

h−1

- r1 r12 ··· r1n
- r2 r22 ··· r2n

) O(hn+1 + ∆s + ∆λ

i1

h−2

)

i2

=

 

 

... .

...

 

 

. .

.

.

h−n

rn rn2 ··· rnn

O(hn+1 + ∆s + ∆λ

gˆs(n)−gs(n) n!

)

in

(31)

Besides, under Assumption B.2, the orders of the other coefficients are the same as we obtain in the last section:

λt

(λ)(λ − λs)kdλ = O(hk+1) (32) Thus

A(λs,λt) = O(1),

#### Eλ

s

λs

n

gˆθ(k)(xλ

,λs) − gθ(k)(xλ

λt

,λs) k!

(λ)(λ − λs)kdλ

s

s

#### Eλ

s

λs

k=1

⊤ 







λt

- gˆs(1) − gs(1)
- gˆs(2)−gs(2) 2!

- λs Eλ

s

(λ)(λ − λs)1dλ

- λt

- λs Eλ

s

(λ)(λ − λs)2dλ

.

- λt

=

 

 

 

 

.

gˆs(n)−gs(n) n!

(λ)(λ − λs)ndλ

λs Eλ

s

−1 



  

  

⊤ 

  

  

  

O(hn+1 + ∆s + ∆λ

h−1

- r1 r12 ··· r1n
- r2 r22 ··· r2n

- O(h2)
- O(h3)

) O(hn+1 + ∆s + ∆λ

i1

h−2

)

i2

 

=

 

 

... .

...

. .

. O(hn+1)

.

h−n

rn rn2 ··· rnn

O(hn+1 + ∆s + ∆λ

)

in



−1 

  

  

  

⊤ 

O(hn+1 + ∆s + ∆λ

- r1 r12 ··· r1n
- r2 r22 ··· r2n

) O(hn+1 + ∆s + ∆λ

O(h) O(h)

i1

)

i2

 

=

 

 

... .

. O(h)

. .

.

rn rn2 ··· rnn

O(hn+1 + ∆s + ∆λ

)

in

n

O(h)O(hn+1 + ∆s + ∆λ

=

)

ik

k=1

(33) Combining Eq. (29), Eq. (32) and Eq. (33), we can obtain the conclusion in Eq. (28).

| |
|---|

- Then we prove Theorem 3.3 below.

- Proof. (Proof of Theorem 3.3)

As we have discussed, the predictor step from tm−1 to tm is a special case of the local approximation Eq. (14) with (ti

,s,t) = (tm−n−1,...,tm−2,tm−1,tm). By Lemma B.7 we have

,...,ti

n

1

n

αt

A(λt

) αt

,λt

O(∆m−k−1) + O(hn+1) (34)

m−1

m

m

∆m−1 + O(h)

∆m =

m−1

k=0

It follows that there exists constants C,C0 irrelevant to h, so that

n

αt

A(λt

) αt

,λt

m−1

m

m

|∆m| ≤

+ Ch |∆m−1| + Ch

m−1

k=0

|∆m−k−1| + C0hn+2 (35)

Denote fm := max0≤i≤m |∆i|, we then have

αt

A(λt

) αt

,λt

+ C1h fm−1 + C0hn+2 (36)

m−1

m

m

|∆m| ≤

m−1

Since αtmA(λt

m−1,λtm)

αtm−1 → 1 when h → 0 and it has bounded first-order derivative due to Assumption B.2, there exists a constant C2, so that for any C ≥ C2, αtmA(λt

m−1,λtm)

αtm−1 + Ch > 1 for sufficiently small h. Thus, by taking C3 = max{C1,C2}, we have

αt

A(λt

) αt

,λt

+ C3h fm−1 + C0hn+2 (37)

m−1

m

m

fm ≤

m−1

m−1,λtm)

Denote Am−1 := αtmA(λt

αtm−1 + C3h, by repeating Eq. (37), we have

 

 C0hn+2 (38)

M−1

M−1

M

fM ≤

Ai fn +

Aj

i=n

i=n+1

j=i

By Assumption B.5, h = O(1/M), so we have

M−1

M−1

C3h αt

αt

αt

A(λt

) αt

,λt

i−1

n

M

M

1 +

Ai =

A(λt

,λt

)

i−1

n

i

i

i=n

i=n

M−1

C4 αt

αt

αt

A(λt

) αt

,λt

i−1

n

M

M

≤

1 +

(39)

)M ≤

A(λt

,λt

i−1

n

i

i

i=n

M−n

σ M

αt

A(λt

) αt

,λt

n

M

M

1 +

n

≤ C5eσ

where σ = maxn≤i≤M−1 αt

i−1C4

αtiA(λti−1,λti). Then denote β := maxn+1≤i≤M αt

M A(λti,λtM )

αti , we have

M−1

M

M

M−i

αt

A(λt

) αt

,λt

σ M

i

M

M

Aj ≤

1 +

i

i=n+1

j=i

i=n+1

M−n−1

i

σ M

≤ β

1 +

(40)

i=0

M−n

βM σ

σ M

− 1 ≤ C6 (eσ − 1)M

=

1 +

Then we substitute Eq. (39) and Eq. (40) into Eq. (38). Note that M = O(1/h) by Assumption B.5, and fn = O(hn+1) by Assumption B.6, finally we conclude that |∆M| ≤ fM = O(hn+1).

| |
|---|

- B.3 Pseudo-Order Solver First, we provide a lemma that gives the explicit solution to Eq. (15).

- Lemma B.8. The solution to Eq. (15) is

k

gˆs(k) k!

p − gi

gi

(41)

0

=

k q=0,q̸=p(δp − δq)

p=1

Proof. Denote





- δ1 δ12 ··· δ1k
- δ2 δ22 ··· δ2k

Rk =

 

 

... .

. .

δk δk2 ··· δkk

Then the solution to Eq. (15) can be expressed as

(42)

gˆs(k) k!

k

(Rk−1)kp(gi

p − gi

=

p=1

) (43)

0

where (Rk−1)kp is the element of Rk−1 at the k-th row and the p-th column. From previous studies of the inversion of the Vandermonde matrix [12], we know

1 δp kq=1,q̸=p(δp − δq)

1 (δp − δ0) kq=1,q̸=p(δp − δq)

(Rk−1)kp =

(44)

=

Substituting Eq. (44) into Eq. (43), we finish the proof.

- Then we prove Theorem 3.4 below:

- Proof. (Proof of Theorem 3.4) First, we use mathematical induction to prove that

k

Dl(k) = D˜l(k) :=

p=1

l+p − gi

gi

, 1 ≤ k ≤ n,0 ≤ l ≤ n − k (45)

l

k q=0,q̸=p(δl+p − δl+q)

For k = 1, Eq. (45) holds by the definition of Dl(k). Suppose the equation holds for k, we then prove it holds for k + 1.

Define the Lagrange polynomial which passes (δl+p,gi

) for 0 ≤ p ≤ k:

l+p − gi

l

k

k

x − δl+q δl+p − δl+q

Pl(k)(x) :=

, 1 ≤ k ≤ n,0 ≤ l ≤ n − k (46)

l+p − gi

gi

l

p=1

q=0,q̸=p

Then D˜l(k) = Pl(k)(x)[xk] is the coefficients before the highest-order term xk in Pl(k)(x). We then prove that Pl(k)(x) satisfies the following recurrence relation:

(x − δl)Pl(+1k−1)(x) − (x − δl+k)Pl(k−1)(x) δl+k − δl

Pl(k)(x) = P˜l(k)(x) :=

(47)

By definition, Pl(+1k−1)(x) is the (k − 1)-th order polynomial which passes (δl+p,gi

) for 1 ≤ p ≤ k, and Pl(k−1)(x) is the (k − 1)-th order polynomial which passes (δl+p,gi

l+p − gi

l

) for

l+p − gi

l

0 ≤ p ≤ k − 1. Thus, for 1 ≤ p ≤ k − 1, we have

(δl+p − δl)Pl(+1k−1)(δl+p) − (δl+p − δl+k)Pl(k−1)(δl+p) δl+k − δl

P˜l(k)(δl+p) =

(48)

l+p − gi

= gi

l

For p = 0, we have

(δl − δl)Pl(+1k−1)(δl) − (δl − δl+k)Pl(k−1)(δl) δl+k − δl

P˜l(k)(δl) =

l − gi

= gi

l

for p = k, we have

(49)

(δl+k − δl)Pl(+1k−1)(δl+k) − (δl+k − δl+k)Pl(k−1)(δl+k) δl+k − δl

P˜l(k)(δl+k) =

(50)

l+k − gi

= gi

l

Therefore, P˜l(k)(x) is the k-th order polynomial which passes k +1 distince points (δl+p,gi

l+p −gi

)

l

for 0 ≤ p ≤ k. Due to the uniqueness of the Lagrange polynomial, we can conclude that Pl(k)(x) = P˜l(k)(x). By taking the coefficients of the highest-order term, we obtain

D˜l(+1k−1) − D˜l(k−1) δl+k − δl

D˜l(k) =

(51)

where by the induction hypothesis we have Dl(+1k−1) = D˜l(+1k−1),Dl(k−1) = D˜l(k−1). Comparing Eq. (51) with the recurrence relation of Dl(k) in Eq. (16), it follows that Dl(k) = D˜l(k), which completes the mathematical induction.

Finally, by comparing the expression for D˜l(k) in Eq. (45) and the expression for gˆs(k) in Lemma B.8, we can conclude that gˆs(k) = k!D0(k).

#### B.4 Local Unbiasedness

Proof. (Proof of Theorem 3.2) Subtracting the local exact solution in Eq. (9) from the (n + 1)-th order local approximation in Eq. (14), we have the local truncation error

n

λt

λt

(λ − λs)k k!

gˆθ(k)(xλ

xˆt − xt = αtA(λs,λt)

(λ)gθ(xλ,λ)dλ −

#### Eλ

dλ

,λs)

#### Eλ

(λ)

s

s

s

λs

λs

k=0

λt

(λ)(gθ(xλ,λ) − gθ(xλ

= αtA(λs,λt)

#### Eλ

,λs))dλ

s

s

λs

n

λt

(λ − λs)k k!

gˆθ(k)(xλ

− αtA(λs,λt)

dλ

#### Eλ

(λ)

,λs)

s

s

λs

k=1

λt

(λ)(gθ(xλ,λ) − gθ(xλ

= αtA(λs,λt)

#### Eλ

,λs))dλ

s

s

λs

n

n

λt

(λ − λs)k k!

(Rn−1)kl(gθ(xλ

− αtA(λs,λt)

) − gθ(xλ

#### Eλ

(λ)

dλ

,λi

,λs))

s

il

s

l

λs

k=1

l=1

(52) where xλ is on the ground-truth ODE trajectory passing xλ

, and (Rn−1)kl is the element of the inverse matrix Rn−1 at the k-th row and the l-th column, as discussed in the proof of Lemma B.8. By Newton-Leibniz theorem, we have

s

λ

gθ(1)(xτ,τ)dτ (53) Also, since xλ

gθ(xλ,λ) − gθ(xλ

,λs) =

s

λs

,l = 1,...,n are on the ground-truth ODE trajectory passing xλ

, we have

il

s

λil

gθ(1)(xτ,τ)dτ (54) where

) − gθ(xλ

gθ(xλ

,λi

,λs) =

il

s

l

λs

τ

gθ(1)(xτ,τ) = e−

λs srdr fθ(1)(xτ,τ) − sτfθ(xτ,τ) − bτ (55)

Note that sλ,lλ are the solution to the least square problem in Eq. (11), which makes sure Epθ

τ(xτ) fθ(1)(xτ,τ) − sτfθ(xτ,τ) − bτ = 0. It follows that Epθ

λs(xλs) fθ(1)(xτ,τ) − sτfθ(xτ,τ) − bτ = 0, since xτ is on the ground-truth ODE trajectory passing xλ

,λs)] = 0 and Epθ

. Therefore, we have Epθ

λs(xλs) [gθ(xλ,λ) − gθ(xλ

s

s

,λs) = 0. Substitute them into Eq. (52), we conclude that Epθ

) − gθ(xλ

λs(xλs) gθ(xλ

,λi

il

s

l

λs(xλs) [xˆt − xt] = 0.

| |
|---|

### C Implementation Details

#### C.1 Computing the EMS and Related Integrals in the ODE Formulation

The ODE formulation and local approximation require computing some complex integrals involving lλ,sλ,bλ. In this section, we’ll give details about how to estimate lλ∗,s∗λ,b∗λ on a few datapoints, and how to use them to compute the integrals efficiently.

#### C.1.1 Computing the EMS

First for the computing of lλ∗ in Eq. (5), note that

∇xNθ(xλ,λ) = σλ∇xϵ(xλ,λ) − diag(lλ) (56) Since diag(lλ) is a diagonal matrix, minimizing Epθ

λ(xλ) ∥∇xNθ(xλ,λ)∥2F is equivalent to minimizing Epθ

λ(xλ) ∥diag−1(∇xNθ(xλ,λ))∥22 = Epθ

λ(xλ) ∥diag−1(σλ∇xϵ(xλ,λ)) − lλ∥22 , where

diag−1 denotes the operator that takes the diagonal of a matrix as a vector. Thus we have lλ∗ = Epθ

##### λ(xλ) diag−1(σλ∇xϵ(xλ,λ)) .

However, this formula for lλ∗ requires computing the diagonal of the full Jacobian of the noise prediction model, which typically has O(d2) time complexity for d-dimensional data and is unacceptable when d is large. Fortunately, the cost can be reduced to O(d) by utilizing stochastic diagonal estimators and employing the efficient Jacobian-vector-product operator provided by forward-mode automatic differentiation in deep learning frameworks.

For a d-by-d matrix D, its diagonal can be unbiasedly estimated by [4]

diag−1(D) =

s

(Dvk) ⊙ vk ⊘

k=1

s

vk ⊙ vk (57)

k=1

where vk ∼ p(v) are d-dimensional i.i.d. samples with zero mean, ⊙ is the element-wise multiplication i.e., Hadamard product, and ⊘ is the element-wise division. The stochastic diagonal estimator is analogous to the famous Hutchinson’s trace estimator [20]. By taking p(v) as the Rademacher distribution, we have vk ⊙ vk = 1, and the denominator can be omitted. For simplicity, we use regular multiplication and division symbols, assuming they are element-wise between vectors. Then lλ∗ can be expressed as:

##### lλ∗ = Epθ

λ(xλ)p(v) [(σλ∇xϵθ(xλ,λ)v)v] (58) which is an unbiased estimation when we replace the expectation with mean on finite samples xλ ∼ pθλ(xλ),v ∼ p(v). The process for estimating lλ∗ can easily be paralleled on multiple devices by computing (σλ∇xϵθ(xλ,λ)v)v on separate datapoints and gather them in the end.

Next, for the computing of s∗λ,b∗λ in Eq. (11), note that it’s a simple least square problem. By taking partial derivatives w.r.t. sλ,bλ and set them to 0, we have

 

λ(xλ) fθ(1)(xλ,λ) − s∗λfθ(xλ,λ) − b∗λ fθ(xλ,λ) = 0 Epθ

Epθ

(59)

λ(xλ) fθ(1)(xλ,λ) − s∗λfθ(xλ,λ) − b∗λ = 0



And we obtain the explicit formula for s∗λ,b∗λ

λ(xλ) fθ(xλ,λ)fθ(1)(xλ,λ) − Epθ

λ(xλ) fθ(1)(xλ,λ) Epθ

Epθ

λ(xλ) [fθ(xλ,λ)]Epθ

s∗λ =

(60)

λ(xλ)[fθ(xλ,λ)fθ(xλ,λ)] − Epθ

λ(xλ)[fθ(xλ,λ)]Epθ

λ(xλ)[fθ(xλ,λ)]

b∗λ = Epθ

λ(xλ)[f(1)(xλ,λ)] − s∗λEpθ

λ(xλ)[fθ(xλ,λ)] (61) which are unbiased least square estimators when we replace the expectation with mean on finite samples xλ ∼ pθλ(xλ). Also, the process for estimating s∗λ,b∗λ can be paralleled on multiple devices by computing fθ, fθ(1), fθfθ, fθfθ(1) on separate datapoints and gather them in the end. Thus, the estimation of s∗λ,b∗λ involving evaluating fθ and fθ(1) on xλ. fθ is a direct transformation of ϵθ and requires a single forward pass. For fθ(1), we have

∂fθ(xλ,λ) ∂λ

dxλ dλ

fθ(1)(xλ,λ) =

+ ∇xfθ(xλ,λ)

l˙λαλ − α˙λlλ αλ2

lλ αλ

α ˙λ αλ

= e−λ ϵ(1)θ (xλ,λ) − ϵθ(xλ,λ) −

xλ −

xλ − σλϵθ(xλ,λ)

l˙λxλ αλ

= e−λ (lλ − 1)ϵθ(xλ,λ) + ϵ(1)θ (xλ,λ) −

(62) After we obtain lλ, l˙λ can be estimated by finite difference. To compute ϵ(1)θ (xλ,λ), we have

dxλ dλ

ϵ(1)θ (xλ,λ) = ∂λϵθ(xλ,λ) + ∇xϵθ(xλ,λ)

= ∂λϵθ(xλ,λ) + ∇xϵθ(xλ,λ)

α ˙λ αλ

xλ − σλϵθ(xλ,λ)

(63)

which can also be computed with the Jacobian-vector-product operator. In conclusion, for any λ, lλ∗,s∗λ,b∗λ can be efficiently and unbiasedly estimated by sampling a few datapoints xλ ∼ pθλ(xλ) and using the Jacobian-vector-product.

#### C.1.2 Integral Precomputing

In the local approximation in Eq. (14), there are three integrals involving the EMS, which are A(λs,λt), λ λt

s)k

(λ)dλ, λ λt

(λ)(λ−λ

k! dλ. Define the following terms, which are also evaluated at λj

#### Eλ

(λ)Bλ

#### Eλ

s

s

s

s

s

and can be precomputed in O(N) time:

,λj

,...,λj

0

1

N

Lλ =

Sλ =

λ

lτdτ

λT

λ

sτdτ

λT

- Bλ =

λ

λT

e−

- r

λT

- sτdτbrdr =

λ

λT

e−S

r

brdr

- Cλ =

Iλ =

λ

u

u λT

(lτ+sτ)dτ

e

λT

λT

λ

r λT

(lτ+sτ)dτdr =

e

λT

e−

- r

λT

- sτdτbrdr du =

λ

#### eL

r+Srdr

λT

λ

eL

u+SuBudu

λT

(64)

Then for any λs,λt, we can verify that the first two integrals can be expressed as

λs−Lλt λt

A(λs,λt) = eL

(λ)dλ = e−L

t − Cλ

s − Bλ

t − Iλ

#### Eλ

(λ)Bλ

λs(Cλ

(Iλ

))

s

s

s

s

λs

(65)

which can be computed in O(1) time. For the third and last integral, denote it as Eλ(k)

s,λt, i.e.,

Eλ(k)

s,λt =

λt

(λ − λs)k k!

dλ (66)

#### Eλ

(λ)

s

λs

We need to compute it for 0 ≤ k ≤ n and for every local transition time pair (λs,λt) in the sampling process. For k = 0, we have

Eλ(0)

s,λt = e−L

λs−Sλs(Iλ

t − Iλ

) (67)

s

which can also be computed in O(1) time. But for k > 0, we no longer have such a simplification technique. Still, for any fixed timestep schedule {λi}Mi=0 during the sampling process, we can use a lazy precomputing strategy: compute Eλ(k)

i−1,λi,1 ≤ i ≤ M when generating the first sample, store it with a unique key (k,i) and retrieve it in O(1) in the following sampling process.

#### C.2 Algorithm

We provide the pseudocode of the local approximation and global solver in Algorithm 1 and Algorithm 2, which concisely describes how we implement DPM-Solver-v3.

- Algorithm 1 (n + 1)-th order local approximation: LUpdaten+1

Require: noise schedule αt,σt, coefficients lλ,sλ,bλ Input: transition time pair (s,t), xs, n extra timesteps {ti

k}nk=1, gθ values (gi

n

,...,gi

1

,gs) at {(xλ

ik

,ti

k

)}nk=1 and (xs,s) Input Format: {ti

n

,gi

n},...,{ti

1

,gi

1},{s,xs,gs},t

- 1: Compute A(λs,λt), λ λt

s

Eλ

s

(λ)Bλ

s

(λ)dλ, λ λt

s

Eλ

s

(λ)(λ−λ

s)k

k! dλ (Appendix C.1.2)

- 2: δk = λi

k − λs, k = 1,...,n

- 3:



 

- gˆs(1)
- gˆs(2) 2!

.

gˆs(n) n!



 

←

  

- δ1 δ12 ··· δ1n
- δ2 δ22 ··· δ2n

. .

... .

δn δn2 ··· δnn

  

−1 

 

gi

- 1 − gs

gi

- 2 − gs

. gi

n − gs

   (Eq. (13))

- 4: xˆt ← αtA(λs,λt)

xs αs −

λt

λs

Eλ

s

(λ)Bλ

s

(λ)dλ −

n

k=0

gˆs(k)

λt

λs

Eλ

s

(λ)

(λ − λs)k k!

dλ (Eq. (14))

Output: xˆt

- Algorithm 2 (n + 1)-th order multistep predictor-corrector algorithm

Require: noise prediction model ϵθ, noise schedule αt,σt, coefficients lλ,sλ,bλ, cache Q1,Q2 Input: timesteps {ti}Mi=0, initial value x0

- 1: Q1 cache← x0
- 2: Q2 cache← ϵθ(x0,t0)
- 3: for m = 1 to M do
- 4: nm ← min{n + 1,m}
- 5: xˆm−n

m

,...,xˆm−1 fetch← Q1

- 6: ϵˆm−n

m

,...,ϵˆm−1 fetch← Q2

- 7: gˆl ← e−

λl λm−1

sτdτ σλ

l

ϵˆl − lλ

l

xˆl αλ

l

−

λl

λm−1

e−

r λm−1

sτdτbrdr, l = m − nm,...,m − 1 (Eq. (8))

- 8: xˆm ← LUpdaten

m

({tm−n

m

,gˆm−n

m},...,{tm−2,gˆm−2},{tm−1,xˆm−1,gˆm−1},tm)

- 9: if m ̸= M then
- 10: ϵˆm ← ϵθ(xˆm,tm)
- 11: gˆm ← e−

λm λm−1

sτdτ σλ

m

ϵˆm − lλ

m

xˆm αλ

m

−

λm

λm−1

e−

r λm−1

sτdτbrdr (Eq. (8))

- 12: xˆcm ← LUpdaten

m

({tm−n

m+1,gˆm−n

m+1},...,{tm−2,gˆm−2},{tm,gˆm}, {tm−1,xˆm−1,gˆm−1},tm)

- 13: ϵˆcm ← ϵˆm + lλ

m

(xˆcm − xˆm)/σλ

m

(to ensure gˆmc = gˆm)

- 14: Q1 cache← xˆcm
- 15: Q2 cache← ϵˆcm
- 16: end if
- 17: end for Output: xˆM

### D Experiment Details

In this section, we provide more experiment details for each setting, including the codebases and the configurations for evaluation, EMS computing and sampling. Unless otherwise stated, we utilize the forward-mode automatic differentiation (torch.autograd.forward_ad) provided by PyTorch [39] to compute the Jacobian-vector-products (JVPs). Also, as stated in Section 3.4, we draw datapoints

xλ from the marginal distribution qλ defined by the forward diffusion process starting from some data distribution q0, instead of the model distribution pθλ.

#### D.1 ScoreSDE on CIFAR10

Codebase and evaluation For unconditional sampling on CIFAR10 [24], one experiment setting is based on the pretrained pixel-space diffusion model provided by ScoreSDE [51]. We use their official codebase of PyTorch implementation, and their checkpoint checkpoint_8.pth under vp/cifar10_ddpmpp_deep_continuous config. We adopt their own statistic file and code for computing FID.

EMS computing We estimate the EMS at N = 1200 uniform timesteps λj

by drawing K = 4096 datapoints xλ

,λj

,...,λj

0

1

N

0 ∼ q0, where q0 is the distribution of the training set. We compute two sets of EMS, corresponding to start time ϵ = 10−3 (NFE≤ 10) and ϵ = 10−4 (NFE>10) in the sampling process respectively. The total time for EMS computing is ∼7h on 8 GPU cards of NVIDIA A40.

Sampling Following previous works [31, 32, 58], we use start time ϵ = 10−3 (NFE≤ 10) and ϵ = 10−4 (NFE>10), end time T = 1 and adopt the uniform logSNR timestep schedule. For DPM-Solver-v3, we use the 3rd-order predictor with the 3rd-order corrector by default. In particular, we change to the pseudo 3rd-order predictor at 5 NFE to further boost the performance.

#### D.2 EDM on CIFAR10

Codebase and evaluation For unconditional sampling on CIFAR10 [24], another experiment setting is based on the pretrained pixel-space diffusion model provided by EDM [21]. We use their official codebase of PyTorch implementation, and their checkpoint edm-cifar10-32x32-uncond-vp.pkl. For consistency, we borrow the statistic file and code from ScoreSDE [51] for computing FID.

EMS computing Since the pretrained models of EDM are stored within the pickles, we fail to use torch.autograd.forward_ad for computing JVPs. Instead, we use torch.autograd.functional.jvp, which is much slower since it employs the double backward trick. We estimate two sets of EMS. One corresponds to N = 1200 uniform timesteps λj

0 ∼ q0, where q0 is the distribution of the training set. The other corresponds to N = 120,K = 4096. They are used when NFE<10 and NFE≥10 respectively. The total time for EMS computing is ∼3.5h on 8 GPU cards of NVIDIA A40.

and K = 1024 datapoints xλ

,λj

,...,λj

0

1

N

Sampling Following EDM, we use start time tmin = 0.002 and end time tmax = 80.0, but adopt the uniform logSNR timestep schedule which performs better in practice. For DPM-Solver-v3, we use the 3rd-order predictor and additionally employ the 3rd-order corrector when NFE≤ 6. In particular, we change to the pseudo 3rd-order predictor at 5 NFE to further boost the performance.

#### D.3 Latent-Diffusion on LSUN-Bedroom

Codebase and evaluation The unconditional sampling on LSUN-Bedroom [55] is based on the pretrained latent-space diffusion model provided by Latent-Diffusion [43]. We use their official codebase of PyTorch implementation and their default checkpoint. We borrow the statistic file and code from Guided-Diffusion [10] for computing FID.

EMS computing We estimate the EMS at N = 120 uniform timesteps λj

by drawing K = 1024 datapoints xλ

,λj

,...,λj

0

1

N

0 ∼ q0, where q0 is the distribution of the latents of the training set. The total time for EMS computing is ∼12min on 8 GPU cards of NVIDIA A40.

Sampling Following previous works [58], we use start time ϵ = 10−3, end time T = 1 and adopt the uniform t timestep schedule. For DPM-Solver-v3, we use the 3rd-order predictor with the pseudo 4th-order corrector.

#### D.4 Guided-Diffusion on ImageNet-256

Codebase and evaluation The conditional sampling on ImageNet-256 [9] is based on the pretrained pixel-space diffusion model provided by Guided-Diffusion [10]. We use their official codebase of PyTorch implementation and their two checkpoints: the conditional diffusion model

256x256_diffusion.pt and the classifier 256x256_classifier.pt. We adopt their own statistic file and code for computing FID.

EMS computing We estimate the EMS at N = 500 uniform timesteps λj

by drawing K = 1024 datapoints xλ

,λj

,...,λj

0

1

N

0 ∼ q0, where q0 is the distribution of the training set. Also, we find that the FID metric on the ImageNet-256 dataset behaves specially, and degenerated lλ (lλ = 1) performs better. The total time for EMS computing is ∼9.5h on 8 GPU cards of NVIDIA A40.

- Sampling Following previous works [31, 32, 58], we use start time ϵ = 10−3, end time T = 1 and adopt the uniform t timestep schedule. For DPM-Solver-v3, we use the 2nd-order predictor with the pseudo 3rd-order corrector.

D.5 Stable-Diffusion on MS-COCO2014 prompts

Codebase and evaluation The text-to-image sampling on MS-COCO2014 [26] prompts is based on the pretrained latent-space diffusion model provided by Stable-Diffusion [43]. We use their official codebase of PyTorch implementation and their checkpoint sd-v1-4.ckpt. We compute MSE on randomly selected captions from the MS-COCO2014 validation dataset, as detailed in Section 4.1.

EMS computing We estimate the EMS at N = 250 uniform timesteps λj

0

,λj

1

,...,λj

N

by drawing K = 1024 datapoints xλ

0 ∼ q0. Since Stable-Diffusion is trained on the LAION-5B dataset [46], there is a gap between the images in the MS-COCO2014 validation dataset and the images generated by Stable-Diffusion with certain guidance scale. Thus, we choose q0 to be the distribution of the latents generated by Stable-Diffusion with corresponding guidance scale, using 200-step DPMSolver++ [32]. We generate these latents with random captions and Gaussian noise different from those we use to compute MSE. The total time for EMS computing is ∼11h on 8 GPU cards of NVIDIA A40 for each guidance scale.

- Sampling Following previous works [32, 58], we use start time ϵ = 10−3, end time T = 1 and adopt the uniform t timestep schedule. For DPM-Solver-v3, we use the 2nd-order predictor with the pseudo 3rd-order corrector.

- D.6 License Table 3: The used datasets, codes and their licenses.

Name URL Citation License

CIFAR10 https://www.cs.toronto.edu/~kriz/cifar.html [24] \ LSUN-Bedroom https://www.yf.io/p/lsun [55] \ ImageNet-256 https://www.image-net.org [9] \ MS-COCO2014 https://cocodataset.org [26] CC BY 4.0 ScoreSDE https://github.com/yang-song/score_sde_pytorch [51] Apache-2.0 EDM https://github.com/NVlabs/edm [21] CC BY-NC-SA 4.0 Guided-Diffusion https://github.com/openai/guided-diffusion [10] MIT Latent-Diffusion https://github.com/CompVis/latent-diffusion [43] MIT Stable-Diffusion https://github.com/CompVis/stable-diffusion [43] CreativeML Open RAIL-M DPM-Solver++ https://github.com/LuChengTHU/dpm-solver [32] MIT UniPC https://github.com/wl-zhao/UniPC [58] \

We list the used datasets, codes and their licenses in Table 3.

### E Runtime Comparison

As we have mentioned in Section 4, the runtime of DPM-Solver-v3 is almost the same as other solvers (DDIM [48], DPM-Solver [31], DPM-Solver++ [32], UniPC [58], etc.) as long as they use the same NFE. This is because the main computation costs are the serial evaluations of the large neural network ϵθ, and the other coefficients are either analytically computed [48, 31, 32, 58], or precomputed (DPM-Solver-v3), thus having neglectable costs.

Table 4 shows the runtime of DPM-Solver-v3 and some other solvers on a single NVIDIA A40 under different settings. We use torch.cuda.Event and torch.cuda.synchronize to accurately compute the runtime. We evaluate the runtime on 8 batches (dropping the first batch since it contains

- Table 4: Runtime of different methods to generate a single batch (second / batch, ±std) on a single NVIDIA A40, varying the number of function evaluations (NFE). We don’t include the runtime of the decoding stage for latent-space DPMs.

Method

NFE

5 10 15 20 CIFAR10 [24], ScoreSDE [51] (batch size = 128)

DPM-Solver++ [32] 1.253(±0.0014) 2.503(±0.0017) 3.754(±0.0042) 5.010(±0.0048) UniPC [58] 1.268(±0.0012) 2.532(±0.0018) 3.803(±0.0037) 5.080(±0.0049) DPM-Solver-v3 1.273(±0.0005) 2.540(±0.0023) 3.826(±0.0039) 5.108(±0.0055)

CIFAR10 [24], EDM [21] (batch size = 128)

DPM-Solver++ [32] 1.137(±0.0011) 2.278(±0.0015) 3.426(±0.0024) 4.569(±0.0031) UniPC [58] 1.142(±0.0016) 2.289(±0.0019) 3.441(±0.0035) 4.590(±0.0021) DPM-Solver-v3 1.146(±0.0010) 2.293(±0.0015) 3.448(±0.0018) 4.600(±0.0027)

LSUN-Bedroom [55], Latent-Diffusion [43] (batch size = 32)

DPM-Solver++ [32] 1.302(±0.0009) 2.608(±0.0010) 3.921(±0.0023) 5.236(±0.0045) UniPC [58] 1.305(±0.0005) 2.616(±0.0019) 3.934(±0.0033) 5.244(±0.0043) DPM-Solver-v3 1.302(±0.0010) 2.620(±0.0027) 3.932(±0.0028) 5.290(±0.0030)

ImageNet256 [9], Guided-Diffusion [10] (batch size = 4)

DPM-Solver++ [32] 1.594(±0.0011) 3.194(±0.0018) 4.792(±0.0031) 6.391(±0.0045) UniPC [58] 1.606(±0.0026) 3.205(±0.0025) 4.814(±0.0049) 6.427(±0.0060) DPM-Solver-v3 1.601(±0.0059) 3.229(±0.0031) 4.807(±0.0068) 6.458(±0.0257)

MS-COCO2014 [26], Stable-Diffusion [43] (batch size = 4)

DPM-Solver++ [32] 1.732(±0.0012) 3.464(±0.0020) 5.229(±0.0027) 6.974(±0.0013) UniPC [58] 1.735(±0.0012) 3.484(±0.0364) 5.212(±0.0015) 6.988(±0.0035) DPM-Solver-v3 1.731(±0.0008) 3.471(±0.0011) 5.211(±0.0030) 6.945(±0.0022)

extra initializations) and report the mean and std. We can see that the runtime is proportional to NFE and has a difference of about ±1% for different solvers, which confirms our statement. Therefore, the speedup for the NFE is almost the actual speedup of the runtime.

F Quantitative Results

- Table 5: Quantitative results on LSUN-Bedroom [55]. We report the FID↓ of the methods with different numbers of function evaluations (NFE), evaluated on 50k samples.

Method Model

NFE

5 6 8 10 12 15 20 DPM-Solver++ [32]

Latent-Diffusion [43]

18.59 8.50 4.19 3.63 3.43 3.29 3.16 UniPC [58] 12.24 6.19 4.00 3.56 3.34 3.18 3.07 DPM-Solver-v3 7.54 4.79 3.53 3.16 3.06 3.05 3.05

We present the detailed quantitative results of Section 4.1 for different datasets in Table 1, Table 5,

- Table 6 and Table 7 respectively. They clearly verify that DPM-Solver-v3 achieves consistently better or comparable performance under various settings, especially in 5∼10 NFEs.

### G Ablations

In this section, we conduct some ablations to further evaluate and analyze the effectiveness of DPM-Solver-v3.

- Table 6: Quantitative results on ImageNet-256 [9]. We report the FID↓ of the methods with different numbers of function evaluations (NFE), evaluated on 10k samples.

Method Model

NFE

5 6 8 10 12 15 20 DPM-Solver++ [32]

Guided-Diffusion [10] (s = 2.0)

16.87 13.09 9.95 8.72 8.13 7.73 7.48 UniPC [58] 15.62 11.91 9.29 8.35 7.95 7.64 7.44 DPM-Solver-v3 15.10 11.39 8.96 8.27 7.94 7.62 7.39

- Table 7: Quantitative results on MS-COCO2014 [26] prompts. We report the MSE↓ of the methods with different numbers of function evaluations (NFE), evaluated on 10k samples.

Method Model

NFE 5 6 8 10 12 15 20 DPM-Solver++ [32]

Stable-Diffusion [43] (s = 1.5)

0.076 0.056 0.028 0.016 0.012 0.009 0.006 UniPC [58] 0.055 0.039 0.024 0.012 0.007 0.005 0.002 DPM-Solver-v3 0.037 0.027 0.024 0.007 0.005 0.001 0.002

DPM-Solver++ [32]

Stable-Diffusion [43] (s = 7.5)

0.60 0.65 0.50 0.46 0.42 0.38 0.30 UniPC [58] 0.65 0.71 0.56 0.46 0.43 0.35 0.31 DPM-Solver-v3 0.55 0.64 0.49 0.40 0.45 0.34 0.29

- G.1 Varying the Number of Timesteps and Datapoints for the EMS

Table 8: Ablation of the number of timesteps N and datapoints K for the EMS, experimented with ScoreSDE [51] on CIFAR10 [24]. We report the FID↓ with different numbers of function evaluations (NFE), evaluated on 50k samples.

N K

NFE 5 6 8 10 12 15 20

1200 512 18.84 7.90 4.49 3.74 3.88 3.52 3.12 1200 1024 15.52 7.55 4.17 3.56 3.37 3.03 2.78 120 4096 13.67 7.60 4.09 3.49 3.24 2.90 2.70 250 4096 13.28 7.56 4.00 3.45 3.22 2.92 2.70 1200 4096 12.76 7.40 3.94 3.40 3.24 2.91 2.71

First, we’d like to investigate how the number of timesteps N and the number of datapoints K for computing the EMS affects the performance. We conduct experiments with the DPM ScoreSDE [51] on CIFAR10 [24], by decreasing N and K from our default choice N = 1200,K = 4096.

We list the FID results using the EMS of different N and K in Table 8. We can observe that the number of datapoints K is crucial to the performance, while the number of timesteps N is less significant and affects mainly the performance in 5∼10 NFEs. When NFE>10, we can decrease N to as little as 50, which gives even better FIDs. Note that the time cost for computing the EMS is proportional to NK, so how to choose appropriate N and K for both efficiency and accuracy is worth studying.

- G.2 First-Order Comparison

As stated in Appendix A, the first-order case of DPM-Solver-v3 (DPM-Solver-v3-1) is different from DDIM [48], which is the previous best first-order solver for DPMs. Note that DPM-Solver-v3-1 applies no corrector, since any corrector has an order of at least 2.

In Table 9 and Figure 7, we compare DPM-Solver-v3-1 with DDIM both quantitatively and qualitatively, using the DPM ScoreSDE [51] on CIFAR10 [24]. The results verify our statement that DPM-Solver-v3-1 performs better than DDIM.

- Table 9: Quantitative comparison of first-order solvers (DPM-Solver-v3-1 and DDIM [48]), experimented with ScoreSDE [51] on CIFAR10 [24]. We report the FID↓ with different numbers of function evaluations (NFE), evaluated on 50k samples.

NFE 5 6 8 10 12 15 20

Method

DDIM [48] 54.56 41.92 27.51 20.11 15.64 12.05 9.00 DPM-Solver-v3-1 39.18 29.82 20.03 14.98 11.86 9.34 7.19

DDIM [48]

NFE = 5 NFE = 10 NFE = 20

[Figure 7]

[Figure 8]

[Figure 9]

FID 54.56 FID 20.11 FID 9.00

DPM-Solver-v3-1 (Ours)

[Figure 10]

[Figure 11]

[Figure 12]

FID 39.18 FID 14.98 FID 7.19

Figure 7: Random samples by first-order solvers (DPM-Solver-v3-1 and DDIM [48]) of ScoreSDE [51] on CIFAR10 dataset [24], using 5, 10 and 20 NFE.

#### G.3 Effects of Pseudo-Order Solver

We now demonstrate the effectiveness of the pseudo-order solver, including the pseudo-order predictor and the pseudo-order corrector.

Pseudo-order predictor The pseudo-order predictor is only applied in one case (at 5 NFE on CIFAR10 [24]) to achieve maximum performance improvement. In such cases, without the pseudoorder predictor, the FID results will degenerate from 12.76 to 15.91 for ScoreSDE [51], and from 12.21 to 12.72 for EDM [21]. While they are still better than previous methods, the pseudo-order predictor is proven to further boost the performance at NFEs as small as 5.

Pseudo-order corrector We show the comparison between true and pseudo-order corrector in Table 10. We can observe a consistent improvement when switching to the pseudo-order corrector. Thus, it suggests that if we use n-th order predictor, we’d better combine it with pseudo (n + 1)-th order corrector rather than (n + 1)-th order corrector.

#### G.4 Effects of Half-Corrector

We demonstrate the effects of half-corrector in Table 11, using the popular Stable-Diffusion model [43]. We can observe that under the relatively large guidance scale of 7.5 which is necessary for producing samples of high quality, the corrector adopted by UniPC [58] has a negative effect on the convergence to the ground-truth samples, making UniPC even worse than DPM-Solver++ [32]. When we employ the half-corrector technique, the problem is partially alleviated. Still, it lags behind our DPM-Solver-v3, since we further incorporate the EMS.

- Table 10: Effects of pseudo-order corrector under different settings. We report the FID↓ with different numbers of function evaluations (NFE).

Method

NFE 5 6 8 10 12 15 20 LSUN-Bedroom [55], Latent-Diffusion [43]

4th-order corrector 8.83 5.28 3.65 3.27 3.17 3.14 3.13 →pseudo (default) 7.54 4.79 3.53 3.16 3.06 3.05 3.05

ImageNet-256 [9], Guided-Diffusion [10] (s = 2.0)

3rd-order corrector 15.87 11.91 9.27 8.37 7.97 7.62 7.47 →pseudo (default) 15.10 11.39 8.96 8.27 7.94 7.62 7.39

MS-COCO2014 [26], Stable-Diffusion [43] (s = 1.5)

3rd-order corrector 0.037 0.028 0.028 0.014 0.0078 0.0024 0.0011 →pseudo (default) 0.037 0.027 0.024 0.0065 0.0048 0.0014 0.0022

- Table 11: Ablation of half-corrector/full-corrector on MS-COCO2014 [26] prompts with StableDiffusion model [43] and guidance scale 7.5. We report the MSE↓ of the methods with different numbers of function evaluations (NFE), evaluated on 10k samples.

Method Corrector Usage

NFE 5 6 8 10 12 15 20 DPM-Solver++ [32] no corrector 0.60 0.65 0.50 0.46 0.42 0.38 0.30 UniPC [58]

full-corrector 0.65 0.71 0.56 0.46 0.43 0.35 0.31

→half-corrector 0.59 0.66 0.50 0.46 0.41 0.38 0.30 DPM-Solver-v3

full-corrector 0.65 0.67 0.49 0.40 0.47 0.34 0.30 →half-corrector 0.55 0.64 0.51 0.44 0.45 0.36 0.29

- G.5 Singlestep vs. Multistep

- Table 12: Quantitative comparison of single-step methods (S) vs. multi-step methods (M), experimented with ScoreSDE [51] on CIFAR10 [24]. We report the FID↓ with different numbers of function evaluations (NFE), evaluated on 50k samples.

NFE 5 6 8 10 12 15 20 25

Method

DPM-Solver (S) [31] 290.51 23.78 23.51 4.67 4.97 3.34 2.85 2.70 DPM-Solver (M) [32] 27.40 17.85 9.04 6.41 5.31 4.10 3.30 2.98 DPM-Solver++ (S) [32] 51.80 38.54 12.13 6.52 6.36 4.56 3.52 3.09 DPM-Solver++ (M) [32] 28.53 13.48 5.34 4.01 4.04 3.32 2.90 2.76 DPM-Solver-v3 (S) 21.83 16.81 7.93 5.76 5.17 3.99 3.22 2.96 DPM-Solver-v3 (M) 12.76 7.40 3.94 3.40 3.24 2.91 2.71 2.64

As we stated in Section 3.2.2, multistep methods perform better than singlestep methods. To study the relationship between parameterization and these solver types, we develop a singlestep version of DPM-Solver-v3 in Algorithm 3. Note that since (n + 1)-th order singlestep solver divides each step into n + 1 substeps, the total number of timesteps is a multiple of n + 1. For flexibility, we follow the adaptive third-order strategy of DPM-Solver [31], which first takes third-order steps and then takes first-order or second-order steps in the end.

- Algorithm 3 (n + 1)-th order singlestep solver Require: noise prediction model ϵθ, noise schedule αt,σt, coefficients lλ,sλ,bλ, cache Q1,Q2 Input: timesteps {ti}(i=0n+1)M, initial value x0

- 1: Q1 cache← x0
- 2: Q2 cache← ϵθ(x0,t0)
- 3: for m = 1 to M do
- 4: i0 ← (n + 1)(m − 1)
- 5: for i = 0 to n do
- 6: xˆi

0

,...,xˆi

0+i

fetch← Q1

- 7: ϵˆi

0

,...,ϵˆi

0+i

fetch← Q2

- 8: gˆl ← e−

λl λi0

sτdτ σλ

l

ϵˆl − lλ

l

xˆl αλ

l

−

λl

λi0

e−

- r

λi0

- sτdτbrdr, l = i0,...,i0 + i (Eq. (8))

- 9: xˆi

0+i+1 ← LUpdatei+1({ti

0+1,gˆi

0+1},...,{ti

0+i,gˆi

0+i},{ti

0

,xˆi

0

,gˆi

0},ti

0+i+1)

- 10: if (i + 1)m ̸= (n + 1)M then
- 11: ϵˆi

0+i+1 ← ϵθ(xˆi

0+i+1,ti

0+i+1)

- 12: Q1 cache← xˆi

0+i+1

- 13: Q2 cache← ϵˆi

0+i+1

- 14: end if
- 15: end for
- 16: end for Output: xˆ(n+1)M

Since UniPC [58] is based on a multistep predictor-corrector framework and has no singlestep version, we only compare with DPM-Solver [31] and DPM-Solver++ [32], which uses noise prediction and data prediction respectively. The results of ScoreSDE model [51] on CIFAR10 [24] are shown in Table 12, which demonstrate that different parameterizations have different relative performance under singlestep and multistep methods.

Specifically, for singlestep methods, DPM-Solver-v3 (S) outperforms DPM-Solver++ (S) across NFEs, but DPM-Solver (S) is even better than DPM-Solver-v3 (S) when NFE≥10 (though when NFE<10, DPM-Solver (S) has worst performance), which suggests that noise prediction is best strategy in such scenarios; for multistep methods, we have DPM-Solver-v3 (M) > DPM-Solver++ (M) > DPM-Solver (M) across NFEs. Moreover, DPM-Solver (S) even outperforms DPM-Solver++ (M) when NFE≥20, and the properties of different parameterizations in singlestep methods are left for future study. Overall, DPM-Solver-v3 (M) achieves the best results among all these methods, and performs the most stably under different NFEs.

### H FID/CLIP Score on Stable-Diffusion

- Table 13: Sample quality and text-image alignment performance on MS-COCO2014 [26] prompts with Stable-Diffusion model [43] and guidance scale 7.5. We report the FID↓ and CLIP score↑ of the methods with different numbers of function evaluations (NFE), evaluated on 10k samples.

NFE 5 6 8 10 12 15 20 DPM-Solver++ [32]

Method Metric

FID 18.87 17.44 16.40 15.93 15.78 15.84 15.72

CLIP score 0.263 0.265 0.265 0.265 0.266 0.265 0.265 UniPC [58]

FID 18.77 17.32 16.20 16.15 16.09 16.06 15.94 CLIP score 0.262 0.263 0.265 0.265 0.265 0.265 0.265

FID 18.83 16.41 15.41 15.32 15.13 15.30 15.23 CLIP score 0.260 0.262 0.264 0.265 0.265 0.265 0.265

DPM-Solver-v3

In text-to-image generation, since the sample quality is affected not only by discretization error of the sampling process, but also by estimation error of neural networks during training, low MSE (faster convergence) does not necessarily imply better sample quality. Therefore, we choose the MSCOCO2014 [26] validation set as the reference, and additionally evaluate DPM-Solver-v3 on Stable-Diffusion model [43] by the standard metrics FID and CLIP score [41] which measure the sample quality and text-image alignment respectively. For DPM-Solver-v3, we use the full-corrector strategy when NFE<10, and no corrector when NFE≥10.

The results in Table 13 show that DPM-Solver-v3 achieves consistently better FID and similar CLIP scores. Notably, we achieve an FID of 15.4 in 8 NFE, close to the reported FID of Stable-Diffusion v1.4.

Still, we claim that FID is not a proper metric for evaluating the convergence of latent-space diffusion models. As stated in DPM-Solver++ and Section 4.1, we can see that the FIDs quickly achieve 15.0∼16.0 within 10 steps, even if the latent code does not converge, because of the strong image decoder. Instead, MSE in the latent space is a direct way to measure the convergence. By comparing the MSE, our sampler does converge faster to the ground-truth samples of Stable Diffusion itself.

### I More Theoretical Analyses

#### I.1 Expressive Power of Our Generalized Parameterization

Though the introduced coefficients lλ,sλ,bλ seem limited to guarantee the optimality of the parameterization formulation itself, we claim that the generalized parameterization gθ in Eq. (8) can actually cover a wide range of parameterization families in the form of ψθ(xλ,λ) = α(λ)ϵθ(xλ,λ)+ β(λ)xλ + γ(λ). Considering the paramerization on [λs,λt], by rearranging the terms, Eq. (8) can be written as

gθ(xλ,λ) = e−

λs sτdτ σλ αλ

λ

ϵθ(xλ,λ) − e−

λs sτdτ lλ αλ

λ

xλ −

λ

e−

λs

r

λs sτdτbrdr (68)

We can compare the coefficients before ϵθ and xλ in gθ and ψθ to figure out how lλ,sλ,bλ corresponds to α(λ),β(λ),γ(λ). In fact, we can not directly let gθ equal ψθ, since when λ = λs, we

have λ λ

(·) = 0, and the coefficient before ϵθ in gθ is fixed. Still, ψθ can be equalized to gθ by a

s

linear transformation, which only depends on λs and does not affect our analyses of the discretization error and solver.

Specifically, assuming ψθ = ωλ

, by corresponding the coefficients we have

gθ + ξλ

s

s



= eλ

ωλ

α(λs) ξλ

s

 

s

λ

e−

λs sτdτ σλ

= γ(λs) lλ = −σλαβ((λλ)) sλ = −1 − α



α(λ) = ωλ

s

αλ β(λ) = −ωλ

s

λ

(69)

e−

λs sτdτ lλ

⇒

αλ γ(λ) = −ωλ

s

′(λ) α(λ)



r

λ λs e−

λs sτdτbrdr + ξλ



s

s

′(λ) α(λ)

bλ = −e−λγ

Therefore, as long as α(λ) ̸= 0 and α(λ),γ(λ) have first-order derivatives, our proposed parameterization gθ holds the same expressive power as ψθ, while at the same time enabling the neat optimality criteria of lλ,sλ,bλ in Eq. (5) and Eq. (11).

#### I.2 Justification of Why Minimizing First-order Discretization Error Can Help Higher-order Solver

The EMS s∗λ,b∗λ in Eq. (11) are designed to minimize the first-order discretization error in Eq. (10). However, the high-order solver is actually more frequently adopted in practice (specifically, third-

order in unconditional sampling, second-order in conditional sampling), since it incurs lower sampling errors by taking higher-order Taylor expansions to approximate the predictor gθ.

In the following, we show that the EMS can also help high-order solver. By Eq. (52) in Appendix B.4, the (n + 1)-th order local error can be expressed as

λt

(λ)(gθ(xλ,λ) − gθ(xλ

xˆt − xt = αtA(λs,λt)

#### Eλ

,λs))dλ

s

s

λs

n

n

λt

(λ − λs)k k!

(Rn−1)kl(gθ(xλ

− αtA(λs,λt)

) − gθ(xλ

#### Eλ

(λ)

,λi

,λs))

dλ

s

il

s

l

λs

k=1

l=1

(70) By Newton-Leibniz theorem, it is equivalent to

λt

xˆt − xt = αtA(λs,λt)

λs

n

− αtA(λs,λt)

k=1

λ

#### Eλ

(λ)

s

λs

n

(Rn−1)kl

l=1

gθ(1)(xτ,τ)dτ dλ

λil

gθ(1)(xλ,λ)dλ

λs

λt

(λ − λs)k k!

dλ

#### Eλ

(λ)

s

λs

(71)

We assume that the estimated EMS are bounded (in the order of O(1), Assumption B.2 in Appendix B.1), which is empirically confirmed as in Section 4.2. By the definition of gθ in Eq. (8), we have gθ(1)(xτ,τ) = e−

τ

λs srdr fθ(1)(xτ,τ) − sτfθ(xτ,τ) − bτ . Therefore, Eq. (11) controls

∥gθ(1)∥2 and further controls ∥xˆt − xt∥2, since other terms are only dependent on the EMS and are bounded.

#### I.3 The Extra Error of EMS Estimation and Integral Estimation

Analysis of EMS estimation error In practice, the EMS in Eq. (5) and Eq. (11) are estimated on finite datapoints by the explicit expressions in Eq. (58) and Eq. (63), which may differ from the true lλ∗,s∗λ,b∗λ. Theoretically, on one hand, the order and convergence theorems in Section 3.2 are irrelevant to the EMS estimation error: The ODE solution in Eq. (9) is correct whatever lλ,sλ,bλ are, and we only need the assumption that these coefficients are bounded (Assumption B.2 in Appendix B.1) to prove the local and global order; on the other hand, the first-order discretization error in Eq. (10) is vulnerable to the EMS estimation error, which relates to the performance at few steps. Empirically, to enable fast sampling, we need to ensure the number of datapoints for estimating EMS (see ablations in Table 8), and we find that our method is robust to the EMS estimation error given only 1024 datapoints in most cases (see EMS computing configs in Appendix D).

Analysis of integral estimation error Another source of error is the process of integral estimation ( λ λt

s)k

(λ)dλ and λ λt

(λ)(λ−λ

k! dλ in Eq. (14)) by trapezoidal rule. We can analyze the estimation error by the error bound formula of trapezoidal rule: suppose we use uniform discretization on [a,b] with interval h to estimate a b f(x)dx, then the error E satisfies

#### Eλ

(λ)Bλ

#### Eλ

s

s

s

s

s

(b − a)h2 12

max|f′′(x)| (72)

|E| ≤

Under Assumption B.2, the EMS and their first-order derivative are bounded. Denote f1(λ) = Eλ

s)k

(λ)(λ−λ

k! , then

(λ)Bλ

(λ),f2(λ) = Eλ

s

s

s

λ

r

λ

e−

λs sτdτbrdr

λs(lτ+sτ)dτ

f1(λ) = e

λs

λ

λ

r

λ

r

λ

e−

e−

λs sτdτbrdr

λs lτdτ

λs sτdτbrdr + bλe

λs(lτ+sτ)dτ

f1′(λ) = (lλ + sλ)e

λs

λs

λ

λ

λ

r

λ

e−

e−

λs(lτ+sτ)dτ

λs sτdτbrdr + (lλ + sλ)2e

λs(lτ+sτ)dτ

f1′′(λ) = (lλ′ + s′λ)e

λs

λs

λ

λ

r

λ

r

λ

e−

e−

λs sτdτbrdr

λs lτdτ

λs sτdτbrdr + b′λe

λs lτdτ

+ (lλ + sλ)bλe

λs

λs

λ

λ

r

λ

r

λ

λs(lτ−sτ)dτ

e−

e−

λs sτdτbrdr

λs sτdτbrdr + b2λe

λs lτdτ

+ bλlλe

λs

λs

r

λs sτdτbrdr

(73)

λs(lτ+sτ)dτ (λ − λs)k

λ

f2(λ) = e

k! f2′(λ) = (lλ + sλ)e

λs(lτ+sτ)dτ (λ − λs)k−1

λs(lτ+sτ)dτ (λ − λs)k k!

λ

λ

+ e

(k − 1)! f2′′(λ) = (lλ′ + s′λ)e

λs(lτ+sτ)dτ (λ − λs)k k!

λs(lτ+sτ)dτ (λ − λs)k k!

λ

λ

(74)

+ (lλ + sλ)2e

λs(lτ+sτ)dτ (λ − λs)k−1 (k − 1)!

λs(lτ+sτ)dτ (λ − λs)k−1 k − 1!

λ

λ

+ (lλ + sλ)e

+ (lλ + sλ)e

λs(lτ+sτ)dτ (λ − λs)k−2 (k − 2)!

λ

+ e

Since lλ,sλ,bλ,lλ′ ,s′λ,b′λ are all O(1), we can conclude that f1′′(λ) = O(h),f2′′(λ) = O(hk−2), and the errors of λ λt

s)k

(λ)dλ, λ λt

(λ)(λ−λ

k! dλ are O(h20h2),O(h20hk−1) respectively, where h0 is the stepsize of EMS discretization (h0 = λ

#### Eλ

#### Eλ

(λ)Bλ

s

s

s

s

s

t−λs

n , n corresponds to 120∼1200 timesteps for our EMS computing), and h = λt − λs. Therefore, the extra error of integral estimation is under high order and ignorable.

### J More Discussions

#### J.1 Extra Computational and Memory Costs

The extra memory cost of DPM-Solver-v3 is rather small. The extra coefficients lλ,sλ,bλ are discretized and computed at N timesteps, each with a dimension D same as the diffused data. The extra memory cost is O(ND), including the precomputed terms in Appendix C.1.2, and is rather small compared to the pretrained model (e.g. only ∼125M in total on Stable-Diffusion, compared to ∼4G of the model itself).

The pre-computation time for estimating EMS is rather short. The EMS introduced by our method can be effectively estimated on around 1k datapoints within hours (Appendix D), which is rather short compared to the long training/distillation time of other methods. Moreover, the integrals of these extra coefficients are just some vector constants that can be pre-computed within seconds, as shown in Appendix C.1.2. The precomputing is done only once before sampling.

The extra computational overhead of DPM-Solver-v3 during sampling is negligible. Once we obtain the estimated EMS and their integrals at discrete timesteps, they can be regarded as constants. Thus, during the subsequent sampling process, the computational overhead is the same as previous training-free methods (such as DPM-Solver++) with negligible differences (Appendix E).

sλ bλ

1.5

λ λT sτdτ

λ λT bτdτ

1.0

0.5

0.0

−0.5

−1.0

−1.5

−4 −2 0 2 4 6

λ

Figure 8: Visualization of the EMS sλ,bλ and their integrals w.r.t. λ, estimated on ScoreSDE [51] on CIFAR10 [24]. sλ,bλ are rather fluctuating, but their integrals are smooth enough to ensure the stability of DPM-Solver-v3.

#### J.2 Flexibility

The pre-computed EMS can be applied for any time schedule during sampling without re-computing EMS. Besides, we compute EMS on unconditional models and it can be used for a wide range of guidance scales (such as cfg=7.5 in Stable Diffusion). In short, EMS is flexible and easy to adopt in downstream applications.

- • Time schedule: The choice of N,K in EMS is disentangled with the timestep scheduler in sampling. Once we have estimated the EMS at N (e.g., 1200) timesteps, they can be flexibly adapted to any schedule (uniform λ/uniform t...) during sampling, by corresponding the actual timesteps during sampling to the N bins. For different time schedule, we only need

to re-precompute Eλ(k)

s,λt in Appendix C.1.2, and the time cost is within seconds.

- • Guided sampling: We compute the EMS on the unconditional model for all guided cases. Empirically, the EMS computed on the model without guidance (unconditional part) performs more stably than those computed on the model with guidance, and can accelerate the sampling procedure in a wide range of guidance scales (including the common guidance scales used in pretrained models). We think that the unconditional model contains some common information (image priors) for all the conditions, such as color, sketch, and other image patterns. Extracting them helps correct some common biases such as shallow color, low saturation level and lack of details. In contrast, the conditional model is dependent on the condition and has a large variance.

Remark J.1. EMS computed on models without guidance cannot work for extremely large guidance scales (e.g., cfg scale 15 for Stable Diffusion), since in this case, the condition has a large impact on the denoising process. Note that at these extremely large scales, the sampling quality is very low (compared to cfg scale 7.5) and they are rarely used in practice. Therefore, our proposed EMS without guidance is suitable enough for the common applications with the common guidance.

#### J.3 Stability

As shown in Section 4.2, the estimated EMS sλ,bλ appear much fluctuating, especially for ScoreSDE on CIFAR10. We would like to clarify that the unstable sλ,bλ is not an issue, and our sampler is stable:

- • The fluctuation of sλ,bλ on ScoreSDE is intrinsic and not due to the estimation error. As we increase the number of samples to decrease the estimation error, the fluctuation is not reduced. We attribute it to the periodicity of trigonometric functions in the positional timestep embedding as stated in Section 4.2.

- • Moreover, we only need to consider the integrals of sλ,bλ in the ODE solution Eq. (9). As shown in Figure 8, the integrals of sλ,bλ are rather smooth, which ensures the stability of our method. Therefore, there is no need for extra smoothing.

#### J.4 Practical Value

When NFE is around 20, our improvement of sample quality is small because all different fast samplers based on diffusion ODEs almost converge. Therefore, what matters is that our method has a faster convergence speed to good sample quality. As we observed, the less diverse the domain is, the more evidently the speed-up takes effect. For example, on LSUN-Bedroom, our method can achieve up to 40% faster convergence.

To sum up, the practical value of DPM-Solver-v3 embodies the following aspects:

- 1. 15∼30% speed-up can save lots of costs for online text-to-image applications. Our speedups are applicable to large text-to-image diffusion models, which are an important part of today’s AIGC community. As the recent models become larger, a single NFE requires more computational resources, and 15∼30% speed-up can save a lot of the companies’ expenses for commercial usage.
- 2. Improvement in 5∼10 NFEs benefits image previewing. Since the samples are controlled by the random seed, coarse samples with 5∼10 NFEs can be used to preview thousands of samples with low costs and give guidance on choosing the random seed, which can be then used to generate fine samples with best-quality sampling strategies. This is especially useful for text-to-image generation. Since our method achieves better quality and converges faster to the ground-truth sample, it can provide better guidance when used for preview.

### K Additional Samples

We provide more visual samples in Figure 9, Figure 10, Figure 11 and Table 14 to demonstrate the qualitative effectiveness of DPM-Solver-v3. It can be seen that the visual quality of DPM-Solver-v3 outperforms previous state-of-the-art solvers. Our method can generate images that have reduced bias (less “shallow”), higher saturation level and more visual details, as mentioned in Section 4.3.

DPM-Solver++ [32]

UniPC [58]

DPM-Solver-v3 (Ours)

NFE = 5 NFE = 10

[Figure 13]

[Figure 14]

FID 28.53 FID 4.01

[Figure 15]

[Figure 16]

FID 23.71 FID 3.93

[Figure 17]

[Figure 18]

FID 12.76 FID 3.40

Figure 9: Random samples of ScoreSDE [51] on CIFAR10 dataset [24] with only 5 and 10 NFE.

Heun’s 2nd [21]

DPM-Solver++ [32]

UniPC [58]

DPM-Solver-v3 (Ours)

NFE = 5 NFE = 10

[Figure 19]

[Figure 20]

FID 320.80 FID 16.57

[Figure 21]

[Figure 22]

FID 24.54 FID 2.91

[Figure 23]

[Figure 24]

FID 23.52 FID 2.85

[Figure 25]

[Figure 26]

FID 12.21 FID 2.51

- Figure 10: Random samples of EDM [21] on CIFAR10 dataset [24] with only 5 and 10 NFE.

DPM-Solver++ [32] (FID 11.02)

UniPC [58] (FID 10.19)

DPM-Solver-v3 (Ours) (FID 9.70)

[Figure 27]

[Figure 28]

[Figure 29]

- Figure 11: Random samples of Guided-Diffusion [10] on ImageNet-256 dataset [9] with a classifier guidance scale 2.0, using only 7 NFE. We manually remove the potentially disturbing images such as those containing snakes or insects.

- Table 14: Additional samples of Stable-Diffusion [43] with a classifier-free guidance scale 7.5, using only 5 NFE and selected text prompts. Some displayed prompts are truncated.

DPM-Solver++ [32] (MSE 0.60)

UniPC [58] (MSE 0.65)

DPM-Solver-v3 (Ours) (MSE 0.55)

Text Prompts

[Figure 30]

[Figure 31]

[Figure 32]

“pixar movie still portrait photo of madison beer, jessica alba, woman, as hero catgirl cyborg woman by pixar, by greg rutkowski, wlop, rossdraws, artgerm, weta, marvel, rave girl, leeloo, unreal engine, glossy skin, pearlescent, wet, bright morning, anime, sci-fi, maxim magazine cover”

[Figure 33]

[Figure 34]

[Figure 35]

“oil painting with heavy impasto of a pirate ship and its captain, cosmic horror painting, elegant intricate artstation concept art by craig mullins detailed”

[Figure 36]

[Figure 37]

[Figure 38]

“environment living room interior, mid century modern, indoor garden with fountain, retro, m vintage, designer furniture made of wood and plastic, concrete table, wood walls, indoor potted tree, large window, outdoor forest landscape, beautiful sunset, cinematic, concept art, sunstainable architecture, octane render, utopia, ethereal, cinematic light”

[Figure 39]

[Figure 40]

[Figure 41]

“the living room of a cozy wooden house with a fireplace, at night, interior design, concept art, wallpaper, warm, digital art. art by james gurney and larry elmore.”

[Figure 42]

[Figure 43]

[Figure 44]

“Full page concept design how to craft life Poison, intricate details, infographic of alchemical, diagram of how to make potions, captions, directions, ingredients, drawing, magic, wuxia”

“Fantasy art, octane render, 16k, 8k, cinema 4d, back-lit, caustics, clean environment, Wood pavilion architecture, warm led lighting, dusk, Landscape, snow, arctic, with aqua water, silver Guggenheim museum spire, with rays of sunshine, white fabric landscape, tall building, zaha hadid and Santiago calatrava, smooth landscape, cracked ice, igloo, warm lighting, aurora borialis, 3d cgi, high definition, natural lighting, realistic, hyper realism”

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

“tree house in the forest, atmospheric, hyper realistic, epic composition, cinematic, landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, unreal engine”

“A trail through the unknown, atmospheric, hyper realistic, 8k, epic composition, cinematic, octane render, artstation landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, 8k resolution, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, rendering by octane, unreal engine”

[Figure 51]

[Figure 52]

[Figure 53]

“postapocalyptic city turned to fractal glass, ctane render, 8 k, exploration, cinematic, trending on artstation, by beeple, realistic, 3 5 mm camera, unreal engine, hyper detailed, photo–realistic maximum detai, volumetric light, moody cinematic epic concept art, realistic matte painting, hyper photorealistic, concept art, cinematic epic, octane render, 8k, corona render, movie concept art, octane render, 8 k, corona render, trending on artstation, cinematic composition, ultra–detailed, hyper–realistic, volumetric lighting”

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

““WORLDS”: zoological fantasy ecosystem infographics, magazine layout with typography, annotations, in the style of Elena Masci, Studio Ghibli, Caspar David Friedrich, Daniel Merriam, Doug Chiang, Ivan Aivazovsky, Herbert Bauer, Edward Tufte, David McCandless”

