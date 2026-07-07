# arXiv:2505.07447v2[cs.LG]20May2025

## Unified Continuous Generative Models

#### Peng Sun1,2 Yi Jiang2 Tao Lin1,∗

1Westlake University 2Zhejiang University

sunpeng@westlake.edu.cn, yi_jiang@zju.edu.cn, lintao@westlake.edu.cn

### Abstract

Recent advances in continuous generative models, encompassing multi-step processes such as diffusion and flow-matching (typically requiring 8-1000 sampling steps) and few-step methods like consistency models (typically 1-8 steps), have yielded impressive generative performance. Existing work, however, often treats these approaches as distinct learning paradigms, leading to disparate training and sampling methodologies. We propose a unified framework designed for the training, sampling, and understanding of these models. Our implementation, the Unified Continuous Generative Models Trainer and Sampler (UCGM-{T,S}), demonstrates state-of-the-art (SOTA) capabilities. For instance, on ImageNet 256 × 256 using a 675M diffusion transformer model, UCGM-T trains a multi-step model achieving 1.30 FID in 20 sampling steps, and a few-step model achieving 1.42 FID in 2 sampling steps. Furthermore, applying our UCGM-S to a pre-trained model from prior work improves its FID from 1.26 (at 250 steps) to 1.06 in only 40 steps, incurring no additional cost. Code: https://github.com/LINs-lab/UCGM.

### 1 Introduction

[Figure 1]

[Figure 2]

###### (a) NFE = 40, FID = 1.48. (b) NFE = 2, FID = 1.75.

Figure 1: Generated samples from two 675M diffusion transformers trained with our UCGM on ImageNet-1K 512×512. The figure showcases generated samples illustrating the flexibility of Number of Function Evaluation (NFE) and superior performance achieved by our UCGM. The left subfigure presents results with NFE = 40 (multi-step), while the right subfigure shows results with NFE = 2 (few-step). Note that the samples are sampled without classifier-free guidance (CFG) or other guidance techniques.

Continuous generative models, encompassing diffusion models [15, 38], flow-matching models [25, 30], and consistency models [41, 29], have demonstrated remarkable success in synthesizing highfidelity data across diverse applications, including image and video generation [33, 3, 30, 48, 16, 4].

Training and sampling of these models necessitate substantial computational resources [18, 20]. Moreover, current research largely treats distinct model paradigms independently, leading to paradigmspecific training and sampling methodologies. This fragmentation introduces two primary challenges:

∗Corresponding author.

Preprint. Under review.

- Table 1: Existing continuous generative paradigms as special cases of our UCGM. Prominent continuous generative models, such as Diffusion, Flow Matching, and Consistency models, can be formulated as specific parameterizations of our UCGM. The columns detail the required parameterizations for the transport coefficients α(·), γ(·), αˆ(·), γˆ(·) and parameters λ, ρ, ν of UCGM. Note that σ(t) is defined as e4(2.68t−1.59) in this table.

Paradigm UCGM-based Parameterization

Type e.g., α(t) = γ(t) = αˆ(t) = γˆ(t) = λ ∈ [0,1] ρ ∈ [0,1] ν ∈ {1,2} Diffusion EDM[18] √ σ(t)

√ 2σ(t)

√ −0.5

√ 1

0 ≥ 0 2 Flow Matching

σ2(t)+ 14

σ2(t)+ 14

σ2(t)+ 14

σ2(t)+ 14

OT[25] t 1 − t 1 −1 0 ≥ 0 1

Consistency sCM[29] sin(t · π2) cos(t · π2) cos(t · π2) sin(t · −2π) 1 1 1

(a) a deficit in unified theoretical and empirical understanding, which constrains the transfer of advancements across different paradigms; and (b) limited cross-paradigm generalization, as algorithms optimized for one paradigm (e.g., diffusion models) are often incompatible with others.

To address these limitations, we introduce UCGM, a novel framework that establishes a unified foundation for the training, sampling, and conceptual understanding of continuous generative models. Specifically, the unified trainer UCGM-T is built upon a unified training objective, parameterized by a consistency ratio λ ∈ [0,1]. This allows a single training paradigm to flexibly produce models tailored for different inference regimes: models behave akin to multi-step diffusion or flow-matching approaches when λ is close to 0, and transition towards few-step consistency-like models as λ approaches 1. Moreover, this versatility can extend to compatibility with various noise schedules (e.g., linear, triangular, quadratic) without requiring bespoke algorithm modifications.

Complementing the unified trainer UCGM-T, we propose a unified sampling algorithm, UCGM-S, designed to work seamlessly with models trained via our objective. Crucially, UCGM-S enhances and accelerates sampling from pre-trained models, encompassing those from distinct prior paradigms and models trained with UCGM-T. The unifying nature of our UCGM is further underscored by its ability to encapsulate prominent existing continuous generative paradigms as specific instantiations of UCGM, as detailed in Tab. 1. Moreover, as illustrated in Fig. 1, models trained with UCGM can achieve excellent sample quality across a wide range of Number of Function Evaluations (NFEs).

A key innovation within UCGM is the introduction of self-boosting techniques for both training and sampling. The training-time self-boosting mechanism enhances model quality and training efficiency (cf., Sec. 3.1), significantly reducing or eliminating the need for computationally expensive guidance techniques [14, 19] during inference. The sampling-time self-boosting, through our proposed estimation extrapolation (cf., Sec. 3.2), markedly improves generation fidelity while minimizing NFEs without requiring additional cost. In summary, our contributions are:

- (a) A unified trainer (UCGM-T) that seamlessly bridges few-step (e.g., consistency models) and multi-step (e.g., diffusion, flow-matching) generative paradigms, accommodating diverse model architectures, latent autoencoders, and noise schedules.
- (b) A versatile and unified sampler (UCGM-S) compatible with our trained models and, importantly, adaptable for accelerating and improving pre-trained models from existing yet distinct paradigms.
- (c) A self-boosting training mechanism enhances model performance and efficiency while reducing reliance on external guidance techniques. Separately, a computation-free self-boosting sampling technique significantly enhances generation quality with reduced inference costs.

Extensive experiments validate the effectiveness and efficiency of UCGM. Our approach consistently matches or surpasses SOTA methods across various datasets, architectures, and resolutions, for both few-step and multi-step generation tasks (cf., the experimental results in Sec. 4).

### 2 Preliminaries

Given a training dataset D, let p(x) represent its underlying data distribution, or p(x|c) under a condition c. Continuous generative models seek to learn an estimator that gradually transforms a simple source distribution p(z) into a complex target distribution p(x) within a continuous space. Typically, p(z) is represented by the standard Gaussian distribution N(0,I). For instance, diffusion

models generate samples by learning to reverse a noising process that gradually perturbs a data sample x ∼ p(x) into a noisy version xt = α(t)x + σ(t)z, where z ∼ N(0,I). Over the range t ∈ [0,T], the perturbation intensifies with increasing t, where higher t values indicate more pronounced noise. Below, we introduce three prominent learning paradigms for deep continuous generative models.

Diffusion models [15, 40, 18]. In the widely adopted EDM method [18], the noising process is defined by setting α(t) = 1, σ(t) = t. The training objective is given by Ex,z,t ω(t)∥fθ(xt,t) − x∥22

where ω(t) is a weighting function. The diffusion model is parameterized by fθ(xt,t) = cskip(t)xt +cout(t)Fθ(cin(t)xt,cnoise(t)) where Fθ is a neural network, and the coefficients cskip, cout, cin, and cnoise are manually designed. During sampling, EDM solves the Probability Flow Ordinary Differential Equation (PF-ODE) [40]: dxt

dt = [xt − fθ(xt,t)]/t, integrated from t = T to t = 0.

Flow matching [25]. Flow matching models are similar to diffusion models but differ in the transport process from the source to the target distribution and in the neural network training objective. The forward transport process utilizes differentiable coefficients α(t) and γ(t), such that xt = α(t)z+γ(t)x. Typically, the coefficients satisfy the boundary conditions α(1) = γ(0) = 1 and α(0) = γ(1) = 0. The training objective is given by Ex,z,t ω(t) Fθ(xt,t) − (dα

dt x) 22 . Similar to diffusion models, the reverse transport process (i.e., sampling process) begins at t = 1 with x1 ∼ N(0,I) and solves the PF-ODE: dxt

dt z + dσ

t

t

dt = Fθ(xt,t), integrated from t = 1 to t = 0.

Consistency models [41, 29]. A consistency model fθ(xt,t) is trained to map the noisy input xt directly to the corresponding clean data x in one or few steps by following the sampling trajectory of the PF-ODE starting from xt. To be valid, fθ must satisfy the boundary condition fθ(x,0) ≡ x. Inspired by EDM [18], one approach to enforce this condition is to parameterize the consistency model as fθ(xt,t) = cskip(t)xt + cout(t)Fθ (cin(t)xt,cnoise(t)) with cskip(0) = 1 and cout(0) = 0. The training objective is defined between two adjacent time steps with a finite distance: Ex

t,t [ω(t)d(fθ(xt,t),fθ−(xt−∆t,t − ∆t))], where θ− denotes stopgrad(θ), ∆t > 0 is the distance between adjacent time steps, and d(·,·) is a metric function. Discrete-time consistency models are sensitive to the choice of ∆t, necessitating manually designed annealing schedules [39, 11] for rapid convergence. This limitation is addressed by proposing a training objective for continuous consistency models [29], derived by taking the limit as ∆t → 0.

In summary, both diffusion and flow-matching models are multi-step frameworks operating within a continuous space, whereas consistency models are designed as few-step approaches.

### 3 Methodology

We first introduce our unified training objective and algorithm, UCGM-T, applicable to both few-step and multi-step models, including consistency, diffusion, and flow-matching frameworks. Additionally, we present UCGM-S, our unified sampling algorithm, which is effective across all these models.

#### 3.1 Unifying Training Objective for Continuous Generative Models

We first propose a unified training objective for diffusion and flow-matching models, which constitute all multi-step continuous generative models. Moreover, we extend this unified objective to encompass both few-step and multi-step models.

Unified training objective for multi-step continuous generative models. We introduce a generalized training objective below that effectively trains generative models while encompassing the formulations presented in existing studies:

1 ω(t) ∥Fθ(xt,t) − zt∥22 , (1)

L(θ) := E(z,x)∼p(z,x),t

where time t ∈ [0,1], ω(t) is the weighting function for the loss, Fθ is a neural network2 with parameters θ, xt = α(t)z + γ(t)x, and zt = αˆ(t)z + γˆ(t)x. Here, α(t), γ(t), αˆ(t), and γˆ(t) are

2For simplicity, unless otherwise specified, we assume that any conditioning information c is incorporated into the network input. Thus, Fθ(xt, t) should be understood as Fθ(xt, t, c) when c is applicable.

- Algorithm 1 (UCGM-T). A Unified and Efficient Trainer for Few-step and Multi-step Continuous Generative Models (including Diffusion, Flow Matching, and Consistency Models)

Require: Dataset D, transport coefficients {α(·), γ(·), αˆ(·), γˆ(·)}, neural network Fθ, enhancement

ratio ζ, Beta distribution parameters (θ1,θ2), learning rate η, stop gradient operator sg. Ensure: Trained neural network Fθ for generating samples from p(x).

- 1: repeat
- 2: Sample z ∼ N(0,I), x ∼ D, t ∼ ϕ(t) := Beta(θ1,θ2)
- 3: Compute input data, such as xt = α(t) · z + γ(t) · x and xλt = α(λt) · z + γ(λt) · x
- 4: Compute model output Ft = Fθ(xt,t) and set z⋆ = z and x⋆ = x
- 5: if ζ ∈ (0,1) then
- 6: Let F∅t = Fθ−(xt,t,∅) to get enhanced x⋆ = ξ(x,t,fx(sg(Ft),xt,t),fx(F∅t ,xt,t)) and z⋆ = ξ(z,t,fz(sg(Ft),xt,t),fz(F∅t ,xt,t)){Note that ξ(a,t,b,d) := a +

ζ + 1t>s 2 1 − ζ · (b − 1t>s · a − d(1 − 1t>s)), where 1(·) is the indicator function}

- 7: end if
- 8: if λ ∈ [0,1) then
- 9: Compute x⋆t = α(t) · z⋆ + γ(t) · x⋆ and x⋆λt = α(λt) · z⋆ + γ(λt) · x⋆
- 10: Compute ∆fxt = fx(sg(Ft),x⋆t,t) · (t−1λt) − fx(Fθ−(xλt,λt),x⋆λt,λt) · (t−1λt){Note that for λ = 0, fx(Fθ−(x0,0),x⋆0,0) = x⋆}

- 11: else if λ = 1 then
- 12: Comupte x⋆t+ϵ = α(t + ϵ) · z⋆ + γ(t + ϵ) · x⋆ and xt−ϵ = α(t − ϵ) · z⋆ + γ(t − ϵ) · x⋆
- 13: Let ∆fxt = fx(Fθ−(xt+ϵ,t+ϵ),x⋆t+ϵ,t+ϵ)·(21ϵ)−fx(Fθ−(xt−ϵ,t−ϵ),x⋆t−ϵ,t−ϵ)·(21ϵ)

- 14: end if
- 15: Compute Ftargett = sg(Ft) − α(t)·γˆ(t4)α−(tαˆ)(t)·γ(t) · clip(∆f

x t ,−1,1)

sin(t)

- 16: Compute loss Lt(θ) = cos(t) Ft − Ftargett 22 and update θ ← θ − η∇θ 0 1 ϕ(t)Lt(θ)dt
- 17: until Convergence

the unified transport coefficients defined for UCGM. Additionally, to efficiently and robustly train multi-step continuous generative models using objective (1), we propose three necessary constraints:

- (a) α(t) is continuous over the interval t ∈ [0,1], with α(0) = 0, α(1) = 1, and dαd(tt) ≥ 0.

- (b) γ(t) is continuous over the interval t ∈ [0,1], with γ(0) = 1, γ(1) = 0, and dγd(tt) ≤ 0.

- (c) For all t ∈ (0,1), it holds that |α(t)·γˆ(t)−αˆ(t)·γ(t)| > 0 to ensure that α(t)·γˆ(t)−αˆ(t)·γ(t) is non-zero and can serve as the denominator in (3).

Under these constraints, diffusion and flow-matching models are special cases of our unified training objective (1) with additional restrictions (App. D.2.4 details EDM models transformation to UCGM):

- (a) For example, following EDM [18, 20], by setting α(t) = 1 and σ(t) = t, diffusion models based on EDM can be derived from (1) provided that the constraint γ(t)/α(t) = t is satisfied3.
- (b) Similarly, flow-matching models can be derived only when αˆ(t) = dαd(tt) and γˆ(t) = dγd(tt) (see Sec. 2 for more technical details about EDM-based and flow-based models).

Unified training objective for both multi-step and few-step models. To facilitate the interpretation of our technical framework, we define two prediction functions based on model Fθ as:

α(t) · Ft − αˆ(t) · xt α(t) · γˆ(t) − αˆ(t) · γ(t)

γˆ(t) · xt − γ(t) · Ft α(t) · γˆ(t) − αˆ(t) · γ(t)

fx(Ft,xt,t) :=

& fz(Ft,xt,t) :=

, (2)

where we define Ft := Fθ(xt,t). The training objective (1) can thus become (cf., App. D.1.2):

1

ωˆ(t)∥fx(Fθ(xt,t),xt,t) − x∥22 . (3) To align with the gradient of our training objective (1), we define a new weighting function ωˆ(t) in (3) as ωˆ(t) := (α(t)α·γ(ˆt()t·)α−(tαˆ)(·tω)(·tγ)(t))2 . To unify few-step models (such as consistency models) with

L(θ) = E(z,x)∼p(z,x),t

3In EDM, with σ(t) = t, the input of neural network Fθ is cin(t)xt = cin(t) · (x + t · z). Although cin(t) can be manually adjusted, the coefficient before z remains t times that of x.

multi-step models, we adopt a modified version of (3) by incorporating a consistency ratio λ ∈ [0,1]:

1 ωˆ(t) ∥fx(Fθ(xt,t),xt,t) − fx(Fθ−(xλt,λt),xλt,λt)∥22 , (4)

L(θ) = E(z,x)∼p(z,x),t

where consistency models and conventional multi-steps models are special cases within the context of (4) (cf., App. D.1.2 and App. D.1.4). Specifically, setting λ = 0 yields diffusion and flow-matching models, while setting λ → 1 − ∆t with ∆t → 0 recovers consistency models. Following previous

studies [41], we set ωˆ(t) = tan(4 t). As a result, the explicit minimization objective L(θ) is given by:

2

4α(t)∆fxt sin(t) · (α(t) · γˆ(t) − αˆ(t) · γ(t))

, (5)

E(z,x)∼p(z,x),t cos(t) Fθ(xt,t) − Fθ−(xt,t) +

2

where the detailed derivation from (4) to (5) is provided in App. D.1.1, and we define ∆fxt in (5) as

fx(Fθ−(xt,t),xt,t) − fx(Fθ−(xλt,λt),xλt,λt) t − λt

∆fxt :=

. (6)

However, optimizing the unified objective in (5) presents a challenge: stabilizing the training process as λ approaches 1. In this regime, the training dynamics resemble those of consistency models, known for unstable gradients, especially with BF16 precision [41, 29]. To address this, we propose several stabilizing training techniques stated below.

Stabilizing gradient as λ → 1. We identify that the instability in objective (5) primarily arises from numerical computational errors in the term ∆fxt , which subsequently affect the training target Ftargett . Specifically, our theoretical analysis reveals that as λ → 1, ∆fxt approaches df

x(F θ−(xt,t),xt,t)

dt . (6) then serves as a first-order difference approximation of df

x(F θ−(xt,t),xt,t)

dt , which would become highly susceptible to numerical precision errors, primarily due to catastrophic cancellation. To mitigate this issue, we propose a second-order difference estimation technique by redefining ∆fxt as

- 1

- 2ϵ

(fx(Fθ−(xt+ϵ,t + ϵ),xt+ϵ,t + ϵ) − fx(Fθ−(xt−ϵ,t − ϵ),xt−ϵ,t − ϵ)) . To further stabilize the training, we implement the following two strategies for ∆fxt :

∆fxt =

(a) We adopt a distributive reformulation of the second-difference term to prevent direct subtraction between nearly identical quantities, which can induce catastrophic cancellation, especially under limited numerical precision (e.g., BF16). Specifically, we factor out the shared scaling coefficient

1 2ϵ, namely, ∆fxt = fx(Fθ−(xt+ϵ,t+ϵ),xt+ϵ,t+ϵ)·21ϵ−fx(Fθ−(xt−ϵ,t−ϵ),xt−ϵ,t−ϵ)·21ϵ. In this paper, we consistently set ϵ to 0.005. See App. D.2.3 for further analysis of this technique.

(b) We observe that applying numerical truncation [29] to ∆fxt enhances training stability. Specifically, we clip ∆fxt to the range [−1,1], which prevents abnormal numerical outliers.

Unified distribution transformation of time. Previous studies [49, 8, 41, 29, 18, 20] employ non-linear functions to transform the time variable t, initially sampled from a uniform distribution t ∼ U(0,1). This transformation shifts the distribution of sampled times, effectively performing importance sampling and thereby accelerating the training convergence rate. For example, the

lognorm function flognorm(t;µ,σ) = 1+exp(−µ−1 σ·Φ−1(t)) is widely used [49, 8], where Φ−1(·) denotes the inverse Cumulative Distribution Function (CDF) of the standard normal distribution.

In this work, we demonstrate that most commonly used non-linear time transformation functions can be effectively approximated by the regularized incomplete beta function: fBeta(t;a,b) =

1 0 τa−1(1−τ)b−1 dτ, where a detailed analysis defers to App. D.2.1. Consequently, we simplify the process by directly sampling time from a Beta distribution, i.e., t ∼ Beta(θ1,θ2), where θ1 and θ2 are parameters that control the shape of distribution (cf., App. C.1.3 for their settings).

- 0 τa−1(1−τ)b−1 dτ/

t

Learning enhanced target score function. Directly employing objective (5) to train models for estimating the conditional distribution p(x|c) results in models incapable of generating realistic samples without Classifier-Free Guidance (CFG) [14]. While enhancing semantic information, CFG approximately doubles the number of function evaluations, incurring significant computational overhead.

- Algorithm 2 (UCGM-S). A Unified and Efficient Sampler for Few-step and Multi-step Continuous Generative Models (including Diffusion, Flow Matching, and Consistency Models)

Require: Initial x˜ ∼ N(0,I), transport coefficients {α(·), γ(·), αˆ(·), γˆ(·)}, trained model Fθ,

sampling steps N, order ν ∈ {1,2}, time schedule T , extrapolation ratio κ, stochastic ratio ρ. Ensure: Final generated sample x˜ ∼ p(x) and history samples {xˆi}Ni=0 over generation process.

- 1: Let N ← ⌊(N + 1)/2⌋ if using second order sampling (ν = 2) {Adjusts total steps to match first-order evaluation count}
- 2: for i = 0 to N − 1 do
- 3: Compute model output F = Fθ−(x˜,ti), and then xˆi = fx(F,x˜,ti) and zˆi = fz(F,x˜,ti)
- 4: if i ≥ 1 then
- 5: Compute extrapolated estimation zˆ = zˆi + κ · (zˆi − zˆi−1) and xˆ = xˆi + κ · (xˆi − xˆi−1)
- 6: end if
- 7: Sample z ∼ N(0,I) {An example choice of ρ for performing SDE-similar sampling is:

ρ = clip(|t

i−ti+1|·2α(ti)

α(ti+1) ,0,1)}

- 8: Compute estimated next time sample x′ = α(ti+1) · (√1 − ρ · zˆ + √ρ · z) + γ(ti+1) · xˆ

- 9: if order ν = 2 and i < N − 1 then
- 10: Compute prediction F′ = Fθ(x′,ti+1), xˆ′ = fx(F′,x′,ti+1) and zˆ′ = fz(F′,x′,ti+1)
- 11: Compute corrected next time sample x′ = x˜ · γ(t

i+1)

γ(ti) + α(ti+1) − γ(t

i+1)α(ti)

γ(ti) · xˆ+xˆ

′

2

- 12: end if
- 13: Reset x˜ ← x′
- 14: end for

A recent work [44] proposes modifying the target score function (see definition in [40]) from ∇xt

log pt(xt|c)(pt,θ(xt|c)/pt,θ(xt))ζ , where ζ ∈ (0,1) denotes the enhancement ratio. By eliminating dependence on CFG, this approach enables high-fidelity sample generation with significantly reduced inference cost.

log(pt(xt|c)) to an enhanced version ∇xt

Inspired by this, we propose enhancing the target score function in a manner compatible with our unified training objective (5). Specifically, we introduce a time-dependent enhancement strategy:

- (a) For t ∈ [0,s], enhance x and z by applying x⋆ = x + ζ · fx(Ft,xt,t) − fx(F∅t ,xt,t) , z⋆ = z+ζ· fz(Ft,xt,t) − fz(F∅t ,xt,t) . Here, F∅t = Fθ−(xt,t,∅) and Ft = Fθ−(xt,t).
- (b) For t ∈ (s,1], enhance x and z by applying x⋆ = x + 12 (fx(Ft,xt,t) − x) and z⋆ = z +

- 1

- 2 (fz(Ft,xt,t) − z). We consistently set s = 0.75 (cf., App. D.1.6 for more analysis).

An ablation study for this technique is shown in Sec. 4.4, and the training process is shown in Alg. 1.

#### 3.2 Unifying Sampling Process for Continuous Generative Models

In this section, we introduce our unified sampling algorithm applicable to both consistency models and diffusion/flow-based models.

For classical iterative sampling models, such as a trained flow-matching model fθ, sampling from the learned distribution p(x) involves solving the PF-ODE [40]. This process typically uses numerical ODE solvers, such as the Euler or Runge-Kutta methods [30], to iteratively transform the initial Gaussian noise x˜ into a sample from p(x) by solving the ODE (i.e., dx˜t

dt = fθ(x˜t,t)), Similarly, sampling processes in models like EDM [18, 20] and consistency models [41] involve a comparable gradual denoising procedure. Building on these observations and our unified trainer UCGM-T, we first propose a general iterative sampling process with two stages, i.e., (a) and (b):

- (a) Decomposition: At time t, the current input x˜t is decomposed into two components: x˜t = α(t) · zˆt + γ(t) · xˆt. This decomposition uses the estimation model Fθ. Specifically, the model output Ft = Fθ−(x˜t,t) is computed, yielding the estimated clean component xˆt = fx(Ft,x˜t,t) and the estimated noise component zˆt = fz(Ft,x˜t,t).
- (b) Reconstruction: The next time step’s input, t′, is generated by combining the estimated components: x˜t′ = α(t′) · zˆt + γ(t′) · xˆt. The process then iterates to stage (a).

We then introduce two enhancement techniques below to optimize the sampling process:

- (i) Extrapolating the estimation. Directly utilizing the estimated xˆt and zˆt to reconstruct the subsequent input x˜t′ can result in significant estimation errors, as the estimation model Fθ does not perfectly align with the target function Ftarget for solving the PF-ODE.

Note that CFG [14] guides a conditional model using an unconditional model, namely, fθ(x˜,t) = fθ(x˜,t)+κ· f∅θ (x˜,t) − fθ(x˜,t) , where κ is the guidance ratio. This approach can be interpreted as leveraging a less accurate estimation to guide a more accurate one [19].

Extending this insight, we propose to extrapolate the next time-step estimates xˆt′ and zˆt′ using the previous estimates xˆt and zˆt, formulated as: xˆt′ ← xˆt′ +κ·(xˆt′ −xˆt) and zˆt′ ← zˆt′ +κ·(zˆt′ −zˆt), where κ ∈ [0,1] is the extrapolation ratio. This extrapolation process can significantly enhance sampling quality and reduce the number of sampling steps. Notably, this technique is compatible with CFG and does not introduce additional computational overhead (see Sec. 4.2 for experimental details and App. D.1.8 for theoretical analysis).

- (ii) Incorporating stochasticity. During the aforementioned sampling process, the input x˜t is deterministic, potentially limiting the diversity of generated samples. To mitigate this, we introduce

√1 − ρ · zˆt + √ρ · z + γ(t′) · xˆt, where z ∼ N(0,I) is a random noise vector, and ρ is the stochasticity ratio. This stochastic term acts as a random perturbation to x˜t, thereby enhancing the diversity of generated samples.

a stochastic term ρ to x˜t, defined as: x˜t′ = α(t′) ·

We find that setting ρ = λ consistently yields optimal performance in terms of generation quality across all experiments, and we leave the analysis of this phenomenon for future research. Furthermore, empirical investigation of κ indicates that the range [0.2,0.6] is consistently beneficial (cf., Sec. 4.4 and App. C.1.3). Model performance remains relatively stable within this range.

Unified sampling algorithm UCGM-S. Putting all these factors together, here we introduce a unified sampling algorithm applicable to consistency models and diffusion/flow-based models, as presented in Alg. 2. This framework demonstrates that classical samplers, such as the Euler sampler utilized for flow-matching models [30], constitute a special case of our UCGM-S (cf. App. D.1.7 for analysis). Extensive experiments (cf., Sec. 4) demonstrate two key features of this algorithm:

- (a) Reduced computational resources: It decreases the number of sampling steps required by existing models while maintaining or enhancing performance.
- (b) High compatibility: It is compatible with existing models, irrespective of their training objectives or noise schedules, without necessitating modifications to model architectures or tuning.

### 4 Experiment

This section details the experimental setup and evaluation of our proposed methodology, UCGM{T,S}. Note that our approach relies on specific parameterizations of the transport coefficients α(·), γ(·), αˆ(·), and γˆ(·), as detailed in Alg. 1 and Alg. 2. Therefore, Tab. 6 summarizes the parameterizations used in experiments, including configurations for compatibility with prior methods.

#### 4.1 Experimental Setting

Datasets. We utilize ImageNet-1K [5] at resolutions of 512 × 512 and 256 × 256 as our primary datasets, following prior studies [20, 41] and adhering to ADM’s data preprocessing protocols [6]. Additionally, CIFAR-10 [22] at a resolution of 32 × 32 is employed for ablation studies.

For both 512 × 512 and 256 × 256 images, experiments are conducted using latent space generative modeling in line with previous works. Specifically: (a) For 256 × 256 images, we employ multiple widely-used autoencoders, including SD-VAE [34], VA-VAE [49], and E2E-VAE [23]. (b) For 512 × 512 images, a DC-AE (f32c32) [3] with a higher compression rate is used to conserve computational resources. When utilizing SD-VAE for 512 × 512 images, a 2× larger patch size is applied to maintain computational parity with the 256×256 setting. Consequently, the computational burden for generating images at both 512 × 512 and 256 × 256 resolutions remains comparable across our trained models4. Further details on datasets and autoencoders are provided in App. C.1.1.

4Previous works often employed the same autoencoders and patch sizes for both resolutions, resulting in higher computational costs for generating 512 × 512 images. For example, the DiT-XL/2 model requires 524.60 GFLOPs for 512 × 512 generation, in contrast to 118.64 GFLOPs for 256 × 256.

- Table 2: System-level quality comparison for multi-step generation task on class-conditional ImageNet-1K. Notation A⊕B denotes the result obtained by combining methods A and B. ↓/↑ indicate a decrease/increase, respectively, in the metric compared to the baseline performance of the pre-trained models.

512 × 512 256 × 256 METHOD NFE (↓) FID (↓) #Params #Epochs METHOD NFE (↓) FID (↓) #Params #Epochs Diffusion & flow-matching Models

ADM-G [6] 250×2 7.72 559M 388 ADM-G [6] 250×2 4.59 559M 396 U-ViT-H/4 [1] 50×2 4.05 501M 400 U-ViT-H/2 [1] 50×2 2.29 501M 400 DiT-XL/2 [33] 250×2 3.04 675M 600 DiT-XL/2 [33] 250×2 2.27 675M 1400 SiT-XL/2 [30] 250×2 2.62 675M 600 SiT-XL/2 [30] 250×2 2.06 675M 1400 MaskDiT [54] 79×2 2.50 736M - MDT [10] 250×2 1.79 675M 1300 EDM2-S [20] 63 2.56 280M 1678 REPA-XL/2 [52] 250×2 1.96 675M 200 EDM2-L [20] 63 2.06 778M 1476 REPA-XL/2 [52] 250×2 1.42 675M 800 EDM2-XXL [20] 63 1.91 1.5B 734 Light.DiT [49] 250×2 2.11 675M 64 DiT-XL/1⊕[3] 250×2 2.41 675M 400 Light.DiT [49] 250×2 1.35 675M 800 U-ViT-H/1⊕[3] 30×2 2.53 501M 400 DDT-XL/2 [47] 250×2 1.31 675M 256 REPA-XL/2 [52] 250×2 2.08 675M 200 DDT-XL/2 [47] 250×2 1.26 675M 400 DDT-XL/2 [47] 250×2 1.28 675M - REPA-E-XL [23] 250×2 1.26 675M 800

###### GANs & masked & autoregressive models

VQGAN⊕[7] 256 18.65 227M - VQGAN⊕[43] - 2.18 3.1B 300 MAGVIT-v2 [51] 64×2 1.91 307M 1080 MAR-L [24] 256×2 1.78 479M 800 MAR-L [24] 256×2 1.73 479M 800 MAR-H [24] 256×2 1.55 943M 800 VAR-d36-s [45] 10×2 2.63 2.3B 350 VAR-d30-re [45] 10×2 1.73 2.0B 350

Ours: UCGM-S sampling with models trained by prior works

UCGM-S⊕[20] 40↓23 2.53↓0.03 280M - UCGM-S⊕[47] 100↓400 1.27↑0.01 675M UCGM-S⊕[20] 50↓13 2.04↓0.02 778M - UCGM-S⊕[49] 100↓400 1.21↓0.14 675M UCGM-S⊕[20] 40↓23 1.88↓0.03 1.5B - UCGM-S⊕[23] 80↓420 1.06↓0.20 675M UCGM-S⊕[47] 200↓300 1.25↓0.03 675M - UCGM-S⊕[23] 20↓480 2.00↑0.74 675M -

Ours: models trained and sampled using UCGM-{T,S} (setting λ = 0)

⊕DC-AE [3] 40 1.48 675M 800 ⊕SD-VAE [34] 60 1.41 675M 400 ⊕DC-AE [3] 20 1.68 675M 800 ⊕VA-VAE [49] 60 1.21 675M 400 ⊕SD-VAE [34] 40 1.67 675M 320 ⊕E2E-VAE [23] 40 1.21 675M 800 ⊕SD-VAE [34] 20 1.80 675M 320 ⊕E2E-VAE [23] 20 1.30 675M 800

Neural network architectures. We evaluate UCGM-S sampling using models trained with established methodologies. These models employ various architectures from two prevalent families commonly used in continuous generative models: (a) Diffusion Transformers, including variants such as DiT [33], UViT [1], SiT [30], Lightening-DiT [49], and DDT [47]. (b) UNet-based convolutional networks, including improved UNets [18, 40] and EDM2-UNets [20]. For training models specifically for UCGM-T, we consistently utilize DiT as the backbone architecture. We train models of various sizes (B: 130M, L: 458M, XL: 675M parameters) and patch sizes. Notation such as XL/2 denotes the XL model with a patch size of 2. Following prior work [49, 47], minor architectural modifications are applied to enhance training stability (details in App. C.1.2).

Implementation details. Our implementation is developed in PyTorch [31]. Training employs AdamW [28] for multi-step sampling models. For few-step sampling models, RAdam [26] is used to improve training stability. Consistent with standard practice in generative modeling [52, 30], an exponential moving average (EMA) of model weights is maintained throughout training using a decay rate of 0.9999. All reported results utilize the EMA model. Comprehensive hyperparameters and additional implementation details are provided in App. C.1.3. Consistent with prior work [40, 15, 25, 2], we adopt standard evaluation protocols. The primary metric for assessing image quality is the Fréchet Inception Distance (FID) [13], calculated on 50,000 images (FID-50K).

#### 4.2 Comparison with SOTA Methods for Multi-step Generation

Our experiments on ImageNet-1K at 512×512 and 256×256 resolutions systematically validate the three key advantages of UCGM: (1) sampling acceleration via UCGM-S on pre-trained models, (2) ultra-efficient generation with joint UCGM-T + UCGM-S, and (3) broad compatibility.

- UCGM-S: Plug-and-play sampling acceleration without additional cost. UCGM-S provides free sampling acceleration for pre-trained generative models. It reduces the required Number of Function Evaluations (NFEs) while preserving or improving generation quality, as measured by FID. Applied to 512 × 512 image generation, the approach demonstrates notable efficiency gains:

###### Table 3: System-level quality comparison for few-step generation task on class-conditional ImageNet-1K. 512 × 512 256 × 256

METHOD NFE (↓) FID (↓) #Params #Epochs METHOD NFE (↓) FID (↓) #Params #Epochs Consistency training & distillation

sCT-M [29] 1 5.84 498M 1837 iCT [39] 2 20.3 675M -

2 5.53 498M 1837 Shortcut-XL/2 [9] 1 10.6 676M 250 sCT-L [29] 1 5.15 778M 1274 4 7.80 676M 250

2 4.65 778M 1274 128 3.80 676M 250 sCT-XXL [29] 1 4.29 1.5B 762 IMM-XL/2 [55] 1×2 7.77 675M 3840

2 3.76 1.5B 762 2×2 5.33 675M 3840 sCD-M [29] 1 2.75 498M 1997 4×2 3.66 675M 3840

2 2.26 498M 1997 8×2 2.77 675M 3840 sCD-L [29] 1 2.55 778M 1434 IMM (ω = 1.5) 1×2 8.05 675M 3840

2 2.04 778M 1434 2×2 3.99 675M 3840 sCD-XXL [29] 1 2.28 1.5B 921 4×2 2.51 675M 3840

2 1.88 1.5B 921 8×2 1.99 675M 3840

GANs & masked & autoregressive models

BigGAN [2] 1 8.43 160M - BigGAN [2] 1 6.95 112M StyleGAN [36] 1×2 2.41 168M - GigaGAN [17] 1 3.45 569M MAGVIT-v2 [51] 64×2 1.91 307M 1080 StyleGAN [36] 1×2 2.30 166M VAR-d36-s [45] 10×2 2.63 2.3B 350 VAR-d30-re [45] 10×2 1.73 2.0B 350

###### Ours: models trained and sampled using UCGM-{T,S} (setting λ = 0)

⊕DC-AE [3] 32 1.55 675M 800 ⊕VA-VAE [49] 16 2.11 675M 400 ⊕DC-AE [3] 16 1.81 675M 800 ⊕VA-VAE [49] 8 6.09 675M 400 ⊕DC-AE [3] 8 3.07 675M 800 ⊕E2E-VAE [23] 16 1.40 675M 800 ⊕DC-AE [3] 4 74.0 675M 800 ⊕E2E-VAE [23] 8 2.68 675M 800

###### Ours: models trained and sampled using UCGM-{T,S} (setting λ = 1)

- ⊕DC-AE [3] 1 2.42 675M 840 ⊕VA-VAE [49] 2 1.42 675M 432

- ⊕DC-AE [3] 2 1.75 675M 840 ⊕VA-VAE [49] 1 2.19 675M 432

- ⊕SD-VAE [34] 1 2.63 675M 360 ⊕SD-VAE [34] 1 2.10 675M 424

- ⊕SD-VAE [34] 2 2.11 675M 360 ⊕E2E-VAE [23] 1 2.29 675M 264

- (a) For the diffusion-based models, such as a pre-trained EDM2-XXL model, UCGM-S reduced NFEs from 63 to 40 (a 36.5% reduction), concurrently improving FID from 1.91 to 1.88.
- (b) When applied to the flow-based models, such as a pre-trained DDT-XL/2 model, UCGM-S achieved an FID of 1.25 with 200 NFEs, compared to the original 1.28 FID requiring 500 NFEs. This demonstrates a performance improvement achieved alongside enhanced efficiency.

This approach generalizes across different generative model frameworks and resolutions. For instance, on 256 × 256 resolution using the flow-based REPA-E-XL model, UCGM-S attained 1.06 FID at 80 NFEs, which surpasses the baseline performance of 1.26 FID achieved at 500 NFEs.

In summary, UCGM-S acts as a broadly applicable technique for efficient sampling, demonstrating cases where performance (FID) improves despite a reduction in sampling steps.

- UCGM-T + UCGM-S: Synergistic efficiency. The combination of UCGM-T training and UCGM-S sampling yields highly competitive generative performance with minimal NFEs:

- (a) 512 × 512: With a DC-AE autoencoder, our framework achieved 1.48 FID at 40 NFEs. This outperforms DiT-XL/1⊕DC-AE (2.41 FID, 500 NFEs) and EDM2-XXL (1.91 FID, 63 NFEs), with comparable or reduced model size.
- (b) 256×256: With an E2E-VAE autoencoder, we attained 1.21 FID at 40 NFEs. This result exceeds prior SOTA models like MAR-H (1.55 FID, 512 NFEs) and REPA-E-XL (1.26 FID, 500 NFEs).

Importantly, models trained with UCGM-T maintain robustness under extremely low-step sampling regimes. At 20 NFEs, the 256 × 256 performance degrades gracefully to 1.30 FID, a result that still exceeds the performance of several baseline models sampling with significantly higher NFEs.

In summary, the demonstrated robustness and efficiency of UCGM-{T,S} across various scenarios underscore the high potential of our UCGM for multi-step continuous generative modeling.

#### 4.3 Comparison with SOTA Methods for Few-step Generation

As evidenced by the results in Tab. 3, our UCGM-{T,S} framework exhibits superior performance across two key settings: λ = 0, characteristic of a multi-step regime akin to diffusion and flowmatching models, and λ = 1, indicative of a few-step regime resembling consistency models.

Few-step regime (λ = 1). Configured for few-step generation, UCGM-{T,S} achieves SOTA sample quality with minimal NFEs, surpassing existing specialized consistency models and GANs:

256

[Figure 3]

140

Diff. Transport

Diff.

4.07 2.07 2.15 2.34 2.56

- 1

- 2

- 3

- 4

1.0

128

= 0.0

EDM

TrigLinear

120

- = 0.5

| | | |
|---|---|---|
| | | |

- = 1.0

Linear

Random

64

4.14 1.69 0.84 0.56 0.47

()logFID-50K

0.75

TrigFlow

100

32

FID-50K

FID-50K

###### Valueof

Few Steps Multi. Steps

| |
|---|

4.22 1.81 0.50 0.19 0.19

80

0.5

16

| |
|---|

8

60

4.31 2.15 0.75 0.27 0.19

0.25

4

40

- 1

- 2

4.41 2.48 1.07 0.43 0.24

0.0

20

4 8 16 32 64

1 2 4 8 16 32 64

0

= 0.45 = 0.0

Sampling Steps

Sampling Steps

(a) Various λ and sampling steps.

(b) Different ζ and transport types.

###### (c) Various κ and sampling steps.

Figure 2: Ablation studies of UCGM on ImageNet-1K 256×256. These studies evaluate key factors of the proposed UCGM. Ablations presented in (a) and (c) utilize XL/1 models with the VA-VAE autoencoder. For the results shown in (b), B/2 models with the SD-VAE autoencoder are used to facilitate more efficient training.

- (a) 512 × 512: Using a DC-AE autoencoder, our model achieves an FID of 1.75 with 2 NFEs and 675M parameters. This outperforms sCD-XXL, a leading consistency distillation model, which reports 1.88 FID with 2 NFEs and 1.5B parameters.
- (b) 256 × 256: Using a VA-VAE autoencoder, our model achieves an FID of 1.42 with 2 NFEs. This is a notable improvement over IMM-XL/2, which obtains 1.99 FID with 8 × 2 = 16 NFEs, demonstrating higher sample quality while requiring 8× fewer sampling steps.

In summary, these results demonstrate the capability of UCGM-{T,S} to deliver high-quality generation with minimal sampling cost, which is advantageous for practical applications.

Multi-step regime (λ = 0). Even when models are trained for multi-step generation, it nonetheless demonstrates competitive performance even when utilizing a moderate number of sampling steps.

- (a) 512 × 512: Using a DC-AE autoencoder, our model obtains an FID of 1.81 with 16 NFEs and 675M parameters. This result is competitive with or superior to existing methods such as VAR-d30-s, which reports 2.63 FID with 10 × 2 = 20 NFEs and 2.3B parameters.
- (b) 256 × 256: Using an E2E-VAE autoencoder, our model achieves an FID of 1.40 with 16 NFEs. This surpasses IMM-XL/2, which obtains 1.99 FID with 8 × 2 = 16 NFEs, demonstrating improved quality at the same sampling cost.

In summary, our UCGM-{T,S} framework demonstrates versatility and high performance across both few-step (λ = 1) and multi-step (λ = 0) sampling regimes. As shown, it consistently achieves SOTA or competitive sample quality relative to existing methods, often requiring fewer sampling steps or parameters, which are important factors for efficient high-resolution image synthesis.

- 4.4 Ablation Study over the Key Factors of UCGM Unless otherwise specified, experiments in this section are conducted with κ = 0.0 and λ = 0.0.

Effect of λ in UCGM-T. Fig. 2a demonstrates that varying λ influences the range of effective sampling steps for trained models. For instance, with λ = 1, optimal performance is attained at 2 sampling steps. In contrast, with λ = 0.5, optimal performance is observed at 16 steps.

Impact of ζ and transport type in UCGM. The results in Fig. 2b demonstrates that UCGM-{T, S} is applicable with various transport types, albeit with some performance variation. Investigating these performance differences constitutes future work. The results also illustrate that the enhanced training objective (achieved with ζ = 0.45 compared to ζ = 0.0, per Sec. 3) consistently improves performance across all tested transport types, underscoring the efficacy of this technique.

Setting different κ in UCGM-S. Experimental results, depicted in Fig. 2c, illustrate the impact of κ on the trade-off between sampling steps and generation quality: (a) High κ values (e.g., 1.0 and 0.75) prove beneficial for extreme few-step sampling scenarios (e.g., 4 steps); (b) Moreover, mid-range κ values (0.25 to 0.5) achieve superior performance with fewer steps compared to κ = 0.0.

Conclusion We introduce UCGM, a unified and efficient framework for the training and sampling of few-step and multi-step continuous generative models. Extensive experiments demonstrate UCGM achieves SOTA performance across various tasks, underscoring the efficacy of its constituent techniques. Additional experimental results and theoretical analysis are provided in App. C and App. D.

### References

- [1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023.
- [2] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.
- [3] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.
- [4] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Enze Xie, and Song Han. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025.
- [5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [7] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.
- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [9] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.
- [10] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23164–23173, 2023.
- [11] Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024.
- [12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.
- [13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [16] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.
- [17] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10124–10134, 2023.

- [18] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022.
- [19] Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024.
- [20] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024.
- [21] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [22] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 and cifar-100 datasets. URl: https://www. cs. toronto. edu/kriz/cifar. html, 6(1):1, 2009.
- [23] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.
- [24] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024.
- [25] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [26] Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. On the variance of the adaptive learning rate and beyond. arXiv preprint arXiv:1908.03265, 2019.
- [27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [29] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.
- [30] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.
- [31] A Paszke. Pytorch: An imperative style, high-performance deep learning library. arXiv preprint arXiv:1912.01703, 2019.
- [32] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikitlearn: Machine learning in python. the Journal of machine Learning research, 12:2825–2830, 2011.
- [33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [35] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

- [36] Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.
- [37] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [38] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [39] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.
- [40] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [41] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [42] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [43] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.
- [44] Zhicong Tang, Jianmin Bao, Dong Chen, and Baining Guo. Diffusion models without classifierfree guidance. arXiv preprint arXiv:2502.12154, 2025.
- [45] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.
- [46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [47] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.
- [48] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024.
- [49] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025.
- [50] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [51] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.
- [52] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [53] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.
- [54] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023.
- [55] Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. arXiv preprint arXiv:2503.07565, 2025.

### Contents

- 1 Introduction 1
- 2 Preliminaries 2
- 3 Methodology 3

- 3.1 Unifying Training Objective for Continuous Generative Models . . . . . . . . . . 3
- 3.2 Unifying Sampling Process for Continuous Generative Models . . . . . . . . . . . 6

- 4 Experiment 7

- 4.1 Experimental Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.2 Comparison with SOTA Methods for Multi-step Generation . . . . . . . . . . . . . 8
- 4.3 Comparison with SOTA Methods for Few-step Generation . . . . . . . . . . . . . 9
- 4.4 Ablation Study over the Key Factors of UCGM . . . . . . . . . . . . . . . . . . . 10

- A Broader Impacts 15
- B Limitations 15
- C Detailed Experiment 15

- C.1 Detailed Experimental Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- C.1.1 Detailed Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.1.2 Detailed Neural Architecture . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.1.3 Detailed Implementation Details . . . . . . . . . . . . . . . . . . . . . . . 16

- C.2 Experimental Results on Small Datasets . . . . . . . . . . . . . . . . . . . . . . . 17
- C.3 Detailed Comparison with SOTA Methods for Multi-step Generation . . . . . . . . 18
- C.4 Detailed Comparison with SOTA Methods for Few-step Generation . . . . . . . . 19
- C.5 Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- C.5.1 Analysis of Consistency Ratio λ . . . . . . . . . . . . . . . . . . . . . . . 20
- C.5.2 Analysis of Transport Types . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D Theoretical Analysis 22

- D.1 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D.1.1 Unified Training Objective . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.1.2 Learning Objective when λ = 0 . . . . . . . . . . . . . . . . . . . . . . . 24
- D.1.3 Closed-form Solution Analysis when λ = 0 . . . . . . . . . . . . . . . . . 28
- D.1.4 Learning Objective as λ → 1 . . . . . . . . . . . . . . . . . . . . . . . . . 37
- D.1.5 Analysis on the Optimal Solution for λ ∈ [0,1] . . . . . . . . . . . . . . . 41
- D.1.6 Enhanced Target Score Function . . . . . . . . . . . . . . . . . . . . . . . 43
- D.1.7 Unified Sampling Process . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- D.1.8 Extrapolating Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . 44

- D.2 Other Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45

- D.2.1 Beta Transformation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- D.2.2 Kumaraswamy Transformation . . . . . . . . . . . . . . . . . . . . . . . . 46
- D.2.3 Derivative Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- D.2.4 Calcluation of Transport . . . . . . . . . . . . . . . . . . . . . . . . . . . 53

### A Broader Impacts

This paper proposes a unified implementation and theoretical framework for recent popular continuous generative models, such as diffusion models, flow matching models, and consistency models. This work should provide positive impacts for the generative modeling community.

### B Limitations

Integration of training acceleration techniques. This work does not explore the integration of advanced training acceleration methods for diffusion models, such as REPA [52].

Exploration of downstream applications. The current study focuses on establishing the foundational framework. Comprehensive exploration of its application to complex downstream generative tasks, including text-to-image and text-to-video generation, is reserved for future research.

### C Detailed Experiment

#### C.1 Detailed Experimental Setting

- C.1.1 Detailed Datasets Image datasets. We conduct experiments on two datasets: CIFAR-10 [21], ImageNet-1K [5]:

- (a) CIFAR-10 is a widely used benchmark dataset for image classification and generation tasks. It consists of 60,000 color images, each with a resolution of 32 × 32 pixels, categorized into 10 distinct classes. The dataset is divided into 50,000 training images and 10,000 test images.
- (b) ImageNet-1K is a large-scale dataset containing over 1.2 million high-resolution images across 1,000 categories.

Latent space datasets. However, directly training diffusion transformers in the pixel space is computationally expensive and inefficient. Therefore, following previous studies [52, 30], we train our diffusion transformers in latent space instead. Tab. 4 presents a comparative analysis of various Variational Autoencoder (VAE) architectures. SD-VAE is characterized by a higher spatial resolution in its latent representation (e.g., H/8 × W/8) combined with a lower channel capacity (4 channels). Conversely, alternative models such as VA-VAE, E2E-VAE, and DC-AE achieve more significant spatial compression (e.g., H/16 × W/16 or H/32 × W/32) at the expense of an increased channel depth (typically 32 channels).

A key consideration is that the computational cost of a diffusion transformer subsequently processing these latent representations is primarily dictated by their spatial dimensions, rather than their channel capacity [3]. Specifically, if the latent map is processed by a transformer by dividing it into nonoverlapping patches, the cost is proportional to the number of these patches. This quantity is given by (H/Compression Ratio/Patch Size) × (W/Compression Ratio/Patch Size). Here, H and W are the input image dimensions, Compression Ratio refers to the spatial compression factor of the VAE (e.g., 8, 16, 32 as detailed in Tab. 4), and Patch Size denotes the side length of the patches processed by the transformer.

Table 4: Comparison of different VAE architectures in terms of latent space dimensions and channel capacity. The table contrasts four variational autoencoder variants (SD-VAE, VA-VAE, E2E-VAE, and DC-AE) by their spatial compression ratios (latent size) and feature channel dimensions. Here, H and W denote input image height and width (e.g., 256 × 256 or 512 × 512), respectively.

SD-VAE (both ema and mse versions) [34] VA-VAE [49] E2E-VAE [23] DC-AE (f32c32) [3] Latent Size (H/8) × (W/8) (H/16) × (W/16) (H/16) × (W/16) (H/32) × (W/32)

Channels 4 32 32 32

- C.1.2 Detailed Neural Architecture

Diffusion Transformers (DiTs) represent a paradigm shift in generative modeling by replacing the traditional U-Net backbone with a Transformer-based architecture. Proposed by Scalable Diffusion Models with Transformers [33], DiTs exhibit superior scalability and performance in image generation tasks. In this paper, we utilize three key variants—DiT-B (130M parameters), DiT-L (458M parameters), and DiT-XL (675M parameters).

To improve training stability, informed by recent studies [49, 47], we incorporate several architectural modifications into the DiT model: (a) SwiGLU feed-forward networks (FFN) [37]; (b) RMSNorm [53] without learnable affine parameters; (c) Rotary Positional Embeddings (RoPE) [42]; and (d) parameter-free RMSNorm applied to Key (K) and Query (Q) projections in self-attention layers [46].

- C.1.3 Detailed Implementation Details Experiments were conducted on a cluster equipped with 8 H800 GPUs, each with 80 GB of VRAM.

Hyperparameter configuration. Detailed hyperparameter configurations are provided in Tab. 5 to ensure reproducibility. The design of time schedules for sampling processes varies in complexity. For few-step models, typically employing 1 or 2 sampling steps, manual schedule design is straightforward. However, the time schedule T utilized by our UCGM-S often comprises a large number of time points, particularly for a large number of sampling steps N. Manual design of such dense schedules is challenging and can limit the achievable performance of our UCGM-{T,S}, as prior work [49, 47] has established that carefully designed schedules significantly enhance multi-step models, including flow-matching variants. To address this, we propose transforming each time point t ∈ T using a generalized Kumaraswamy transformation: fKuma(t;a,b,c) = (1 − (1 − ta)b)c. This choice is motivated by the common practice in prior studies of applying non-linear transformations to individual time points to construct effective schedules. A specific instance of such a transformation is the timeshift function fshift(t;s) = 1+(sst−1)t, where s > 0 [49]. We find that the Kumaraswamy transformation, by appropriate selection of parameters a,b,c, can effectively approximate fshift and other widely-used functions (cf., App. D.2.2), including the identity function f(t) = t [52, 23]. Empirical evaluations suggest that the parameter configuration (a,b,c) = (1.17,0.8,1.1) yields robust performance across diverse scenarios, corresponding to the "Auto" setting in Tab. 5.

Detailed implementation techniques of enhancing target score function. We enhance the target score function for conditional diffusion models by modifying the standard score ∇xt

log pt(xt|c) [40] to an enhanced version derived from the density pt(xt|c)(pt,θ(xt|c)/pt,θ(xt))ζ. This corresponds to a target score of ∇xt

log pt,θ(xt)). The objective is to guide the learning process towards distributions that yield higher quality conditional samples.

log pt(xt|c) + ζ (∇xt

log pt,θ(xt|c) − ∇xt

Accurate estimation of the model probabilities pt,θ is crucial for the effectiveness of this enhancement. We find that using parameters from an Exponential Moving Average (EMA) of the model during training improves the stability and quality of these estimates, resulting better x⋆ and z⋆ in Alg. 1.

When training few-step models, direct computation of the enhanced target score gradient typically requires evaluating the model with and without conditioning (for the pt,θ terms), incurring additional computational cost. To address this, we propose an efficient approximation that leverages a wellpre-trained multi-step model, denoted by parameters θ⋆. Instead of computing the score gradient explicitly, the updates for the variables x⋆ and z⋆ (as used in Alg. 1) are calculated based on features or outputs derived from a single forward pass of the pre-trained model θ⋆.

Specifically, we compute Ft = Fθ⋆(xt,t), representing features extracted by the pre-trained model θ⋆ at time t given input xt. The enhanced updates x⋆ and z⋆ are then computed as follows:

- (a) For t ∈ [0,s], the updates are: x⋆ ← x+ζ·(fx(Ft,xt,t) − x), z⋆ ← z+ζ·(fz(Ft,xt,t) − z).
- (b) For t ∈ (s,1], the updates are: x⋆ ← x + 21 (fx(Ft,xt,t) − x) and z⋆ ← z + 1 2 (fz(Ft,xt,t) − z).

We consistently set the time threshold s = 0.75. This approach allows us to incorporate the guidance from the enhanced target signal with the computational cost equivalent to a single forward evaluation of the pre-trained model θ⋆ per step. The enhancement ratio ζ is constrained to [0,∞) in this case.

Baselines. We compare our approach against several SOTA continuous and discrete generative models. We broadly categorize these baselines by their generation process:

- (a) Multi-step models. These methods typically synthesize data through a sequence of steps. We include various diffusion models, encompassing classical formulations like DDPM and score-based models [38, 15], and advanced variants focusing on improved sampling or performance in latent spaces [6, 18, 33, 54, 1]. We also consider flow-matching models [25], which leverage continuous

- Table 5: Hyperparameter configurations for UCGM-{T,S} training and sampling on ImageNet-1K. We maintain a consistent batch size of 1024 across all experiments. Training durations (epoch counts) are provided in other tables throughout the paper. The table specifies optimizer choices, learning rates, and key parameters for both UCGM-T and UCGM-S variants across different model architectures and datasets.

Task Optimizer UCGM-T UCGM-S

Resolution VAE/AE Model Type lr (β1,β2) Transport (θ1,θ2) λ ζ ρ κ T ν Multi-step model training and sampling

E2E-VAE XL/1 AdamW 0.0002 (0.9,0.95) Linear (1.0,1.0) 0 0.67 0 0.5 Auto 1 256 SD-VAE XL/2 AdamW 0.0002 (0.9,0.95) Linear (2.4,2.4) 0 0.44 0 0.21 Auto 1

VA-VAE XL/1 AdamW 0.0002 (0.9,0.95) Linear (1.0,1.0) 0 0.47 0 0.5 Auto 1 512 DC-AE XL/1 AdamW 0.0002 (0.9,0.95) Linear (1.0,1.0) 0 0.57 0 0.46 Auto 1

SD-VAE XL/4 AdamW 0.0002 (0.9,0.95) Linear (2.4,2.4) 0 0.60 0 0.4 Auto 1 Few-step model training and sampling

E2E-VAE XL/1 RAdam 0.0001 (0.9,0.999) Linear (0.8,1.0) 1 1.3 1 0 {1,0.5} 1 256 SD-VAE XL/2 RAdam 0.0001 (0.9,0.999) Linear (0.8,1.0) 1 2.0 1 0 {1,0.3} 1

VA-VAE XL/2 RAdam 0.0001 (0.9,0.999) Linear (0.8,1.0) 1 2.0 1 0 {1,0.3} 1 512 DC-AE XL/1 RAdam 0.0001 (0.9,0.999) Linear (0.8,1.0) 1 1.5 1 0 {1,0.6} 1

SD-VAE XL/4 RAdam 0.0001 (0.9,0.999) Linear (0.8,1.0) 1 1.5 1 0 {1,0.5} 1

- Table 6: Comparison of different transport types employed during the sampling and training phases of our UCGM-{T, S}. “TrigLinear” and “Random” are introduced herein specifically for ablation studies. “TrigLinear” is constructed by combining the transport coefficients of “Linear” and “TrigFlow”. “Random” represents a randomly designed transport type used to demonstrate the generality of our UCGM. Other transport types are adapted from existing methods and transformed into the transport coefficient representation used by UCGM.

Linear ReLinear TrigFlow EDM (σ(t) = e4·(2.68t−1.59)) TrigLinear Random α(t) t 1 − t sin(t · π2) σ(t)/√

σ2(t)+0.25 sin(t · π2) sin(t · π2) γ(t) 1 − t t cos(t · π2) 1/√

σ2(t)+0.25 cos(t · π2) 1 − t αˆ(t) 1 −1 cos(t · π2) −0.5/√

σ2(t)+0.25 1 1 γˆ(t) −1 1 −sin(t · π2) 2σ(t)/√

σ2(t)+0.25 −1 −1 − e−5t e.g., [30, 52] [49, 47] [29, 4] [41, 20, 18] N/A N/A

normalizing flows and demonstrate favorable training properties, along with subsequent scaling efforts [30, 52, 49]. Additionally, we also include autoregressive models [24, 45, 51] as the baselines, which generate data sequentially, often in discrete domains.

- (b) Few-step models. These models are designed for efficient, often single-step or few-step, generation. This category includes generative adversarial networks [12], which achieve efficient one-step synthesis through adversarial training, and their large-scale variants [2, 36, 17]. We also evaluate consistency models [41], proposed for high-quality generation adaptable to few sampling steps, and subsequent techniques aimed at improving their stability and scalability [39, 29, 55].

Crucially, we demonstrate the compatibility of UCGM-S with models pre-trained using these methods. We show how these models can be represented within the UCGM framework by defining the functions α(·), γ(·), αˆ(·), and γˆ(·). Detailed parameterizations are provided in Tab. 6, with guidance for their specification presented in App. D.2.4.

- C.2 Experimental Results on Small Datasets

Since most existing few-step generation methods [41, 11] are limited to training models on lowresolution, small-scale datasets like CIFAR-10 [21], we conduct our comparative experiments on CIFAR-10 to ensure fair comparison. To demonstrate the versatility of our UCGM, we employ both the "EDM" transport (see Tab. 6 for definition) and the standard 56M-parameter UNet architecture, following established practices in prior work [41, 11]. As shown in Tab. 7, our UCGM achieves

- Table 7: System-level quality comparison for few-step generation task on unconditional CIFAR-10 (32×32).

Metric PD [35] 2-RF [27] DMD [50] CD [41] sCD [29] iCT [39] ECT [11] sCT [29] IMM [55] UCGM FID (↓) 4.51 4.85 3.77 2.93 2.52 2.83 2.46 3.60 2.11 2.97 2.06 3.20 1.98 2.82 2.17 NFE (↓) 2 1 1 2 2 1 2 1 2 1 2 1 2 1 2

SOTA performance with just 1 NFE (Neural Function Evaluation) while maintaining competitive results for 2 NFEs. These results underscore UCGM’s robust compatibility across diverse datasets, network architectures, and transport types.

#### C.3 Detailed Comparison with SOTA Methods for Multi-step Generation

- Table 8: System-level quality comparison for multi-step generation task on class-conditional ImageNet-1K. Notation A⊕B denotes the result obtained by combining methods A and B. ↓/↑ indicate a decrease/increase, respectively, in the metric compared to the baseline performance of the pre-trained models.

METHOD VAE/AE Patch Size Activation Size NFE (↓) FID (↓) IS (↑) #Params #Epochs 512 × 512 Diffusion & flow-matching models

ADM-G [6] - - - 250×2 7.72 172.71 559M 388 U-ViT-H/4 [1] SD-VAE [34] 4 16×16 50×2 4.05 263.79 501M 400 DiT-XL/2 [33] SD-VAE [34] 2 32×32 250×2 3.04 240.82 675M 600 SiT-XL/2 [30] SD-VAE [34] 2 32×32 250×2 2.62 252.21 675M 600 MaskDiT [54] SD-VAE [34] 2 32×32 79×2 2.50 256.27 736M EDM2-S [20] SD-VAE [34] - - 63 2.56 - 280M 1678 EDM2-L [20] SD-VAE [34] - - 63 2.06 - 778M 1476 EDM2-XXL [20] SD-VAE [34] - - 63 1.91 - 1.5B 734 DiT-XL/1⊕[3] DC-AE [3] 1 16×16 250×2 2.41 263.56 675M 400 U-ViT-H/1⊕[3] DC-AE [3] 1 16×16 30×2 2.53 255.07 501M 400 REPA-XL/2 [52] SD-VAE [34] 2 32×32 250×2 2.08 274.6 675M 200 DDT-XL/2 [47] SD-VAE [34] 2 32×32 250×2 1.28 305.1 675M -

###### GANs & masked & autoregressive models

VQGAN⊕[7] - - - 256 18.65 - 227M MAGVIT-v2 [51] - - - 64×2 1.91 324.3 307M 1080 MAR-L [24] - - - 256×2 1.73 279.9 479M 800 VAR-d36-s [45] - - - 10×2 2.63 303.2 2.3B 350

Ours: UCGM-S sampling with models trained by prior works

EDM2-S [20] SD-VAE [34] - - 40↓23 2.53↓0.03 - 280M EDM2-L [20] SD-VAE [34] - - 50↓13 2.04↓0.02 - 778M EDM2-XXL [20] SD-VAE [34] - - 40↓23 1.88↓0.03 - 1.5B DDT-XL/2 [47] SD-VAE [34] 2 32×32 200↓300 1.25↓0.03 - 675M -

Ours: models trained and sampled using UCGM-{T,S} (setting λ = 0)

Ours-XL/1 DC-AE [3] 1 16×16 40 1.48 - 675M 800

- Ours-XL/1 DC-AE [3] 1 16×16 20 1.68 - 675M 800 Ours-XL/4 SD-VAE [34] 4 16×16 40 1.67 - 675M 320 Ours-XL/4 SD-VAE [34] 4 16×16 20 1.80 - 675M 320

256 × 256

Diffusion & flow-matching models

ADM-G [6] - - - 250×2 4.59 186.70 559M 396 U-ViT-H/2 [1] SD-VAE [34] 2 16 × 16 50×2 2.29 263.88 501M 400 DiT-XL/2 [33] SD-VAE [34] 2 16 × 16 250×2 2.27 278.24 675M 1400 SiT-XL/2 [30] SD-VAE [34] 2 16 × 16 250×2 2.06 277.50 675M 1400 MDT [10] SD-VAE [34] 2 16 × 16 250×2 1.79 283.01 675M 1300 REPA-XL/2 [52] SD-VAE [34] 2 16 × 16 250×2 1.96 264.0 675M 200 REPA-XL/2 [52] SD-VAE [34] 2 16 × 16 250×2 1.42 305.7 675M 800 Light.DiT [49] VA-VAE [49] 1 16 × 16 250×2 2.11 - 675M 64 Light.DiT [49] VA-VAE [49] 1 16 × 16 250×2 1.35 - 675M 800 DDT-XL/2 [47] SD-VAE [34] 2 16 × 16 250×2 1.31 308.1 675M 256 DDT-XL/2 [47] SD-VAE [34] 2 16 × 16 250×2 1.26 310.6 675M 400 REPA-E-XL [23] E2E-VAE[23] 1 16 × 16 250×2 1.26 314.9 675M 800

GANs & masked & autoregressive models

VQGAN⊕[43] - - - - 2.18 - 3.1B 300 MAR-L [24] - - - 256×2 1.78 296.0 479M 800 MAR-H [24] - - - 256×2 1.55 303.7 943M 800 VAR-d30-re [45] - - - 10×2 1.73 350.2 2.0B 350

Ours: UCGM-S sampling with models trained by prior works

DDT-XL/2 [47] SD-VAE [34] 2 16 × 16 100↓400 1.27↑0.01 - 675M Light.DiT [49] VA-VAE [49] 1 16 × 16 100↓400 1.21↓0.14 - 675M REPA-E-XL [23] E2E-VAE[23] 1 16 × 16 80↓420 1.06↓0.20 - 675M REPA-E-XL [23] E2E-VAE[23] 1 16 × 16 20↓480 2.00↑0.74 - 675M -

Ours: models trained and sampled using UCGM-{T,S} (setting λ = 0)

- Ours-XL/2 SD-VAE [34] 2 16 × 16 60 1.41 - 675M 400 Ours-XL/1 VA-VAE [49] 1 16 × 16 60 1.21 - 675M 400 Ours-XL/1 E2E-VAE [23] 1 16 × 16 40 1.21 - 675M 800 Ours-XL/1 E2E-VAE [23] 1 16 × 16 20 1.30 - 675M 800

#### C.4 Detailed Comparison with SOTA Methods for Few-step Generation

###### Table 9: System-level quality comparison for few-step generation task on class-conditional ImageNet-1K (512 × 512).

METHOD VAE/AE Patch Size Activation Size NFE (↓) FID (↓) IS #Params #Epochs 512 × 512 Consistency training & distillation

- sCT-M [29] - - - 1 5.84 - 498M 1837
- sCT-M [29] - - - 2 5.53 - 498M 1837

- sCT-L [29] - - - 1 5.15 - 778M 1274
- sCT-L [29] - - - 2 4.65 - 778M 1274

- sCT-XXL [29] - - - 1 4.29 - 1.5B 762
- sCT-XXL [29] - - - 2 3.76 - 1.5B 762

- sCD-M [29] - - - 1 2.75 - 498M 1997
- sCD-M [29] - - - 2 2.26 - 498M 1997

- sCD-L [29] - - - 1 2.55 - 778M 1434
- sCD-L [29] - - - 2 2.04 - 778M 1434

- sCD-XXL [29] - - - 1 2.28 - 1.5B 921
- sCD-XXL [29] - - - 2 1.88 - 1.5B 921 GANs & masked & autoregressive models

BigGAN [2] - - - 1 8.43 - 160M StyleGAN [36] - - - 1×2 2.41 267.75 168M MAGVIT-v2 [51] - - - 64×2 1.91 324.3 307M 1080 VAR-d36-s [45] - - - 10×2 2.63 303.2 2.3B 350

###### Ours: models trained and sampled using UCGM-{T,S} (setting λ = 0)

Ours-XL/1 DC-AE [3] 1 16×16 32 1.55 - 675M 800 Ours-XL/1 DC-AE [3] 1 16×16 16 1.81 - 675M 800 Ours-XL/1 DC-AE [3] 1 16×16 8 3.07 - 675M 800 Ours-XL/1 DC-AE [3] 1 16×16 4 74.0 - 675M 800

###### Ours: models trained and sampled using UCGM-{T,S} (setting λ = 1)

- Ours-XL/1 DC-AE [3] 1 16×16 1 2.42 - 675M 840
- Ours-XL/1 DC-AE [3] 1 16×16 2 1.75 - 675M 840

- Ours-XL/4 SD-VAE [34] 4 16×16 1 2.63 - 675M 360
- Ours-XL/4 SD-VAE [34] 4 16×16 2 2.11 - 675M 360 256 × 256

###### Consistency training & distillation

iCT [39] - - - 2 20.3 - 675M Shortcut-XL/2 [9] SD-VAE [34] 2 16×16 1 10.6 - 676M 250 Shortcut-XL/2 [9] SD-VAE [34] 2 16×16 4 7.80 - 676M 250 Shortcut-XL/2 [9] SD-VAE [34] 2 16×16 128 3.80 - 676M 250

- IMM-XL/2 [55] SD-VAE [34] 2 16×16 1×2 7.77 - 675M 3840
- IMM-XL/2 [55] SD-VAE [34] 2 16×16 2×2 5.33 - 675M 3840 IMM-XL/2 [55] SD-VAE [34] 2 16×16 4×2 3.66 - 675M 3840 IMM-XL/2 [55] SD-VAE [34] 2 16×16 8×2 2.77 - 675M 3840

- IMM (ω = 1.5) SD-VAE [34] 2 16×16 1×2 8.05 - 675M 3840
- IMM (ω = 1.5) SD-VAE [34] 2 16×16 2×2 3.99 - 675M 3840 IMM (ω = 1.5) SD-VAE [34] 2 16×16 4×2 2.51 - 675M 3840 IMM (ω = 1.5) SD-VAE [34] 2 16×16 8×2 1.99 - 675M 3840

###### GANs & masked & autoregressive models

BigGAN [2] - - - 1 6.95 - 112M GigaGAN [17] - - - 1 3.45 225.52 569M StyleGAN [36] - - - 1×2 2.30 265.12 166M VAR-d30-re [45] - - - 10×2 1.73 350.2 2.0B 350

###### Ours: models trained and sampled using UCGM-{T,S} (setting λ = 0)

Ours-XL/1 VA-VAE [49] 1 16×16 16 2.11 - 675M 400 Ours-XL/1 VA-VAE [49] 1 16×16 8 6.09 - 675M 400 Ours-XL/1 E2E-VAE [23] 1 16×16 16 1.40 - 675M 800 Ours-XL/1 E2E-VAE [23] 1 16×16 8 2.68 - 675M 800

###### Ours: models trained and sampled using UCGM-{T,S} (setting λ = 1)

Ours-XL/1 VA-VAE [49] 1 16×16 2 1.42 - 675M 432

- Ours-XL/1 VA-VAE [49] 1 16×16 1 2.19 - 675M 432
- Ours-XL/2 SD-VAE [34] 2 16×16 1 2.10 - 675M 424 Ours-XL/1 E2E-VAE [23] 1 16×16 1 2.29 - 675M 264

| |
|---|

| |
|---|

| |
|---|

Few-step Model (i.e., =0)

Few-step Model (i.e., =0)

Few-step Model (i.e., =0)

Multi-step Model (i.e., =1)

Multi-step Model (i.e., =1)

Multi-step Model (i.e., =1)

Initial Point x1 (0,I)

Initial Point x1 (0,I)

Initial Point x1 (0,I)

RealDataDensity

RealDataDensity

RealDataDensity

(a) Two Moons

(b) S-Curve

###### (c) Swiss Roll

- Figure 3: Case studies of UCGM on three synthetic datasets. These intuitive studies evaluate the ability of our UCGM to capture the latent data structure for both few-step generation (λ = 1) and multi-step generation (λ = 0) tasks.

[Figure 4]

- Figure 4: Intermediate images generated during 60-step sampling from UCGM-S. Columns display

intermediate images xˆt produced at different timesteps t during a single sampling trajectory, ordered from left to right by decreasing t. Rows correspond to models trained with λ ∈ {0.0, 0.5, 1.0}, ordered from top to bottom. Note that the initial noise for generating these images is the same.

#### C.5 Case Studies

In this section, we provide several case studies to intuitively illustrate the technical components proposed in this paper.

#### C.5.1 Analysis of Consistency Ratio λ

We evaluate our approach on three synthetic benchmark datasets from scikit-learn [32]: the Two Moons (non-linear separation, see Fig. 3a), S-Curve (manifold structure, see Fig. 3b), and Swiss Roll (non-linear dimensionality reduction, see Fig. 3c). These studies yield two primary observations:

- (a) Our UCGM successfully captures the structure of the data distribution and maps initial points sampled from a Gaussian distribution to the target distribution, regardless of whether the task is few-step (λ = 1) or multi-step (λ = 0) generation.
- (b) Models trained for multi-step (λ = 0) and few-step (λ = 1) generation map the same initial Gaussian noise to nearly identical target data points.

To further validate these findings and explore additional properties of the consistency ratio λ, we conduct experiments on a real-world dataset (ImageNet-1K). Specifically, we trained three models with three different settings of λ ∈ {0.0,0.5,1.0}.

The experimental results presented in Fig. 4 demonstrate the following:

- (a) For λ = 1.0, high visual fidelity is achieved early in the sampling process. In contrast, for λ = 0.0, high visual fidelity emerges in the mid to late stages. For λ = 0.5, high-quality images appear in the mid-stage of sampling.
- (b) Despite being trained with different settings of λ values, the models produce remarkably similar generated images.

[Figure 5]

- Figure 5: Visualization of generated images (512 × 512) from pre-trained EDM2-S [20].

[Figure 6]

- Figure 6: Visualization of generated images (512 × 512) from pre-trained DDT-XL/2 [47].

In summary, we posit that while the setting of λ affects the dynamics of the generation process, it does not substantially impact the final generated image quality. Detailed analysis of these phenomena is provided in App. D.1.2, App. D.1.4 and App. D.1.5.

#### C.5.2 Analysis of Transport Types

Generated samples, obtained using UCGM-S with two distinct pre-trained models from prior works, are presented in Fig. 6 and Fig. 5. When using the identical initial Gaussian noise for both models, the generated images exhibit notable visual similarity. This observation is unexpected, considering the models were trained independently [20, 47] using distinct algorithms, transport formulations, network architectures, and data augmentation strategies. The similarity suggests that despite these differences, the learned probability flow ODEs may be converging to similar solutions. See App. D.1.3 for a comprehensive analysis of this phenomenon.

### D Theoretical Analysis D.1 Main Results

#### D.1.1 Unified Training Objective

Problem setup. Let (V,⟨·,·⟩) be a real inner-product space and Θ ⊆ Rp an open parameter domain. We consider

A : Θ → V, B ∈ V (constant w.r.t. θ ∈ Θ), and define the objective

1 ω

A(θ) − B 2, ω > 0. We denote by ∇θA(θ) ∈ Rp×dimV the Jacobian matrix of A.

J (θ) =

|Lemma 1 (Gradient of a Squared Norm) . If v : Θ → V is C1, then<br><br>∇θ v(θ) 2 = 2 ∇θv(θ) ⊤ v(θ).|
|---|

Proof. Define f : V → R by f(v) = ⟨v,v⟩. Its Fréchet derivative is Df(v)[h] =

2

d dϵ

= 2⟨v,h⟩. By the chain rule,

v + ϵh

ϵ=0

∇θ∥v(θ)∥2 = ∇θv(θ) ⊤Df v(θ) = 2 ∇θv(θ) ⊤ v(θ).

|Lemma 2 (Stop-Gradient Simplification) . If B does not depend on θ, then<br><br>∇θ A(θ) − B 2 = 2 ∇θA(θ) ⊤ A(θ) − B .|
|---|

Proof. Set v(θ) = A(θ) − B. Since ∇θv = ∇θA, Lem. 1 applies directly.

|Lemma 3 (Finite-Difference Definition) . Let t > 0, λ ∈ (0,1), and A0 : {λt,t} → V. Define<br><br>∆A :=<br><br>A0(t) − A0(λt) t − λt<br><br>. Then<br><br>A0(t) − A0(λt) = (t − λt)∆A.|
|---|

Proof. Immediate from the definition.

|Theorem 1 (Gradient Approximation via Finite Difference) . Under the above hypotheses, let<br><br>J (θ) =<br><br>1 ω<br><br>A(θ) − A0(λt) 2, and assume A(θ) ≈ A0(t). Then<br><br>∇θJ (θ) =<br><br>2 ω ∇θA(θ) ⊤ A(θ)−A0(λt) ≈<br><br>2(t − λt) ω ∇θA(θ) ⊤∆A ∝ ∇θA(θ), ∆A<br><br>|
|---|

##### .

Proof. Combine Lem. 2 and Lem. 3, then absorb the scalar 2(t−ωλt) into the learning rate. The only non-rigorous step is the approximation A(θ) ≈ A0(t).

| |
|---|

|Lemma 4 . Let Fθ : X → V be C1 in θ, let y ∈ V, and let F− : X → V be independent of θ. Define<br><br>L(θ) = Ex Fθ(x) − F−(x) + y 2, G(θ) = Ex Fθ(x), y . Then<br><br>∇θG(θ) =<br><br>1<br><br>2 ∇θL(θ) − Ex ∇θFθ(x) ⊤ Fθ(x) − F−(x) .<br><br><br>In particular, if Fθ(x) ≈ F−(x) then<br><br>∇θG(θ) ≈<br><br>1<br><br>2 ∇θL(θ).<br>|
|---|

Proof. By Lem. 1,

∇θ Fθ − F− + y 2 = 2 ∇θFθ ⊤ Fθ − F− + y . Taking expectation and dividing by 2 gives

- 1

- 2∇θL = E (∇θFθ)⊤(Fθ − F−) + E (∇θFθ)⊤y .

On the other hand,

∇θG = ∇θE⟨Fθ,y⟩ = E (∇θFθ)⊤y . Rearranging yields the stated identity. Derivation of the training objective. We begin with the original training objective:

1 ωˆ(t)

fx Fθ(xt,t),xt,t − fx Fθ−(xλt,λt),xλt,λt 22 , where θ− denotes the stop-gradient copy of θ.

L(θ) = E(z,x)∼p(z,x), t

- Step 1. By Lem. 2,

∇θL = E(z,x), t

2 ωˆ(t) ∇θ fx Fθ(xt,t),xt,t ⊤ fx(Fθ(xt,t),xt,t)−fx(Fθ−(xλt,λt),xλt,λt) .

- Step 2. Define

A0(s) := fx Fθ−(xs,s),xs,s , ∆A :=

A0(t) − A0(λt) t − λt

. Subsequently, based on Lem. 3 and Thm. 1, we obtain:

∇θL = Et

2(t − λt) ωˆ(t) ∇θ fx T ∆A ∝ Et ∇θ fx, ∆A .

- Step 3. Since

α(s)h − αˆ(s)x α(s) ˆγ(s) − αˆ(s)γ(s)

tan(t) 4

fx(h,x,s) =

, one checks

, ωˆ(t) =

α(t) α(t) ˆγ(t) − αˆ(t)γ(t)

1 ωˆ(t)

4cos(t) sin(t)

∇hfx =

. Hence

,

=

α(t)

∇θ fx Fθ(xt,t),xt,t =

α(t) ˆγ(t) − αˆ(t)γ(t) ∇θ Fθ(xt,t), and therefore

4α(t) cos(t) sin(t) α(t) ˆγ(t) − αˆ(t)γ(t)

∇θL ∝ Et ∇θ Fθ(xt,t), y , y =

∆A.

| |
|---|

- Step 4. Finally apply Lem. 4 with Fθ = Fθ(xt,t) and F− = Fθ−(xt,t). We have

2 2

L(θ) = E(z,x)∼p(z,x), t Fθ(xt,t) − Fθ−(xt,t) + y

. Pulling the overall cos(t) inside the norm yields the final training objective

4α(t)∆fxt sin(t) α(t) ˆγ(t) − αˆ(t)γ(t)

L(θ) = E(z,x)∼p(z,x), t cos(t) Fθ(xt,t) − Fθ−(xt,t) +

This completes the derivation.

2 2

.

| |
|---|

#### D.1.2 Learning Objective when λ = 0

Recall that (z,x) ∼ p(z,x) is a pair of latent and data variables (typically independent), and let t ∈ [0,1]. We have four differentiable scalar functions α,γ,α,ˆ γˆ: [0,1] → R , the noisy interpolant xt = α(t)z + γ(t)x and Ft = Fθ(xt,t). We define the x- and z-prediction functions by

α(t)Ft − αˆ(t)xt α(t) ˆγ(t) − αˆ(t)γ(t)

γˆ(t)xt − γ(t)Ft α(t) ˆγ(t) − αˆ(t)γ(t)

fx(Ft,xt,t) =

, and fz(Ft,xt,t) =

. Finally, let ωˆ(t) > 0 be a weight function. We consider the x- and z-prediction losses

Lx(θ) = E(z,x)∼p(z,x), t

1 ωˆ(t)

fx(Ft,xt,t) − x 22 ,

1 ωˆ(t)

fz(Ft,xt,t) − z 22 . Recall that our unified loss function is defined by:

Lz(θ) = E(z,x)∼p(z,x), t

1 ωˆ(t) ∥fx(Fθ(xt,t),xt,t) − fx(Fθ−(xλt,λt),xλt,λt)∥22 .

L(θ) = E(z,x)∼p(z,x),t

We have L(θ) = Lx(θ) when λ = 0, since fx(F0,x0,0) = 0. Then, we define the direct-field loss LF (θ) = E(z,x), t w(t) Ft − (ˆα(t)z + γˆ(t)x) 22 , w(t) > 0.

|Lemma 5 (Equivalence of x-prediction and direct-field loss) . For all θ,<br><br>fx(Ft,xt,t) − x =<br><br>α(t) α(t) ˆγ(t) − αˆ(t)γ(t)<br><br>Ft − (ˆα(t)z + γˆ(t)x) . Hence<br><br>Lx(θ) = E(z,x), t<br><br>α(t)2 ωˆ(t) α(t) ˆγ(t) − αˆ(t)γ(t) 2<br><br>Ft − (ˆα(t)z + γˆ(t)x) 22 ,<br><br>so Lx is equivalent to LF with<br><br>w(t) =<br><br>α(t)2 ωˆ(t) α(t) ˆγ(t) − αˆ(t)γ(t) 2<br><br>.|
|---|

Proof. Compute

α(t)Ft − αˆ(t)xt

fx(Ft,xt,t) − x =

α(t) ˆγ(t) − αˆ(t)γ(t) − x. Since xt = α(t)z + γ(t)x, the numerator becomes

α Ft − αˆ αz + γx − α γˆ − α γˆ x = α(t) Ft − α ˆ(t)z + γˆ(t)x .

Dividing by α γˆ − α γˆ yields the desired factorization. Substituting into Lx gives the weight w(t) as above.

| |
|---|

|Lemma 6 (Equivalence of z-Prediction and Direct-Field Loss) . For all θ,<br><br>fz(Ft,xt,t) − z =<br><br>γ(t) α(t) ˆγ(t) − αˆ(t)γ(t)<br><br>T(t,z,x) − Ft . Hence<br><br>Lz(θ) = E(z,x), t<br><br>γ(t)2 ωˆ(t) α(t) ˆγ(t) − αˆ(t)γ(t) 2<br><br>Ft − (ˆα(t)z + γˆ(t)x) 22 ,<br><br>so Lz is equivalent to LF with<br><br>w(t) =<br><br>γ(t)2 ωˆ(t) α(t) ˆγ(t) − αˆ(t)γ(t) 2<br><br>.|
|---|

Proof. Compute

γˆ(t)xt − γ(t)Ft

fz(Ft,xt,t) − z =

α(t) ˆγ(t) − αˆ(t)γ(t) − z. Using xt = αz + γx, the numerator is

γˆ(αz + γx) − γ Ft − α γˆ − α γˆ z = γ(t) α ˆ(t)z + γˆ(t)x − Ft . Dividing by α γˆ − α γˆ gives the factorization. Substitution into Lz yields the stated equivalence. Then, when λ = 0, we aim to derive the Probability Flow Ordinary Differential Equation (PFODE) [40] corresponding to a defined forward process from time 0 to 1.

| |
|---|

|Lemma 7 (Probability Flow ODE for the linear Gaussian forward process) . Let p(x) be a data distribution on Rd, and let z ∼ N(0,Id) be independent of x. Let α,γ : [0,1] → R be continuously differentiable scalar functions satisfying<br><br>α(0) = 0, α(1) = 1, γ(0) = 1, γ(1) = 0, and assume γ(t) = 0 for t ∈ (0,1). Define the forward process<br><br>xt = α(t)z + γ(t)x, t ∈ [0,1],<br><br>so that x0 = x ∼ p(x) and x1 = z ∼ N(0,I). Let pt(xt) denote the marginal density of xt. Then the Probability Flow ODE for this process,<br><br>dxt dt<br><br>= f(xt,t) − 12 g(t)2 ∇xt<br><br>log pt(xt), takes the explicit form<br><br>dxt dt<br><br>=<br><br>γ′(t) γ(t)<br><br>xt − α(t)α′(t) −<br><br>γ′(t) γ(t)<br><br>α(t)2 ∇xt<br><br>log pt(xt). (7)|
|---|

##### ̸

Proof. We first represent the forward process xt as the solution of the linear SDE

##### dxt = f(xt,t)dt + g(t)dwt,

where wt is a standard d-dimensional Wiener process, and where f(·,t) and g(t) are to be determined so that xt = α(t)z + γ(t)x in law.

- 1. Drift term via the conditional mean. Since z and x are independent, E[xt | x0 = x] = γ(t)x.

Differentiating in t gives

d dt

E[xt | x0] = γ′(t)x.

On the other hand, if f(xt,t) = H(t)xt for some matrix H(t), then

d dt

E[xt | x0] = H(t)E[xt | x0] = H(t)γ(t)x. Comparison yields H(t) = γ′(t)/γ(t)Id, so

γ′(t) γ(t)

f(xt,t) =

#### xt.

- 2. Diffusion term via the conditional variance. The covariance of xt given x0 is Var(xt | x0) = α(t)2 Id.

For a linear SDE with drift matrix H(t) and scalar diffusion g(t), the covariance Σ(t) satisfies the Lyapunov equation

dΣ(t) dt

= H(t)Σ(t) + Σ(t)H(t)⊤ + g(t)2 Id. Substitute Σ(t) = α(t)2Id and H(t) = γ

′(t)

γ(t) Id. Since ddt α(t)2 = 2α(t)α′(t), we get

2α(t)α′(t)Id = 2

γ′(t) γ(t)

α(t)2 Id + g(t)2 Id. Rearranging yields

g(t)2 = 2α(t)α′(t) − 2

γ′(t) γ(t)

α(t)2.

- 3. Probability Flow ODE. By general theory (see, e.g., de Bortoli et al.), the probability flow ODE associated with the SDE dxt = f(xt,t)dt + g(t)dwt is

dxt dt

= f(xt,t) − 12 g(t)2 ∇xt

log pt(xt). Substituting the expressions for f and g2 above gives

γ′(t) γ(t)

dxt dt

′(t)

xt − α(t)α′(t) − γ

γ(t) α(t)2 ∇xt

=

log pt(xt), i.e.,

γ′(t) γ(t)

γ′(t) γ(t)

xt, g(t)2 = 2α(t)α′(t) − 2

α(t)2. which is exactly the claimed formula (7).

f(xt,t) =

|Lemma 8 (Tweedie formula [40] for the linear Gaussian model) . Under the linear Gaussian interpolation model xt | x ∼ N γ(t)x, α2(t)I , the conditional expectation of x given xt is<br><br>E[x | xt] =<br><br>xt + α2(t)∇xt<br><br>log pt(xt) γ(t)<br><br>.|
|---|

Proof. We write the conditional expectation by Bayes’ rule:

1 pt(xt)

E[x | xt] = xp(x | xt)dx =

xpt(xt | x)p(x)dx,

where pt(xt) = pt(xt | x)p(x)dx. Since pt(xt | x) = (2πα2(t))−d/2 exp −2α21(t)∥xt − γ(t)x∥2 , we have

1 α2(t)

(xt − γ(t)x)pt(xt | x).

∇xt

pt(xt | x) = −

Differentiating the marginal,

1 α2(t)

∇xt

pt(xt) = ∇xt

pt(xt | x)p(x)dx = −

(xt − γ(t)x)pt(xt | x)p(x)dx.

Multiply by −α2(t) and split:

−α2(t)∇xt

pt(xt) = xt pt(xt) − γ(t) xpt(xt | x)p(x)dx.

Rearrange and divide by γ(t)pt(xt):

1 pt(xt)

xt + α2(t)∇xt

pt(xt)/pt(xt) γ(t)

xpt(xt | x)p(x)dx =

=

xt + α2(t)∇xt

log pt(xt) γ(t)

.

Hence E[x | xt] = (xt + α2(t)∇xt

log pt(xt))/γ(t), as claimed.

|Lemma 9 (Optimal predictors as conditional expectations) . For each fixed t and observed xt, the pointwise minimizers fx⋆ and fz⋆ for the objective function L(θ) satisfy<br><br>fx⋆(Ft,xt,t) = E[x | xt], fz⋆(Ft,xt,t) = E[z | xt].|
|---|

Proof. Fix t and xt. By Lem. 5 and Lem. 6, we conclude that the minimizers of L(θ) are equivalent to those of Lx and Lz. Then, up to an additive constant independent of fx, the contribution of (t,xt) to Lx is

##### Jx fx(Ft,xt,t) = E ∥fx(Ft,xt,t) − x∥22 | xt .

For any random vector X, the function w  → E∥w − X∥2 is uniquely minimized at w = E[X]. Therefore

fx⋆(Ft,xt,t) = arg min

E ∥w − x∥2 | xt = E[x | xt]. The same argument applies to

w

Jz fz(Ft,xt,t) = E ∥fz(Ft,xt,t) − z∥22 | xt , yielding

fz⋆(Ft,xt,t) = E[z | xt].

|Theorem 2 . Under the linear Gaussian interpolation model xt = α(t)z + γ(t)x, with z ∼ N(0,I) independent of x, we have<br><br>fx⋆(Ft,xt,t) =<br><br>xt + α2(t)∇xt<br><br>log pt(xt) γ(t)<br><br>, fz⋆(Ft,xt,t) = α(t)∇xt<br><br>log pt(xt). Then for every t,<br><br>α′(t)fz⋆(Ft,xt,t)+γ′(t)fx⋆(Ft,xt,t) =<br><br>γ′(t) γ(t)<br><br>xt − α(t)α′(t) −<br><br>γ′(t) γ(t)<br><br>α2(t) ∇xt<br><br>log pt(xt). As a result, by Lem. 7, we conclude:<br><br>α′(t)fz⋆(Ft,xt,t) + γ′(t)fx⋆(Ft,xt,t) =<br><br>dxt dt<br><br>|
|---|

Proof. Tweedie formula for fx⋆(Ft,xt,t). According to Lem. 9 and Lem. 8, we have

xt + α2(t)∇xt

log pt(xt) γ(t)

fx⋆(Ft,xt,t) = E[x | xt] =

.

Derivation of E[z | xt] for fz⋆(Ft,xt,t). From xt = α(t)z + γ(t)x we solve z = (xt − γ(t)x)/α(t). Taking conditional expectation and substituting the above,

1 α(t)

E[z | xt] =

xt − γ(t)E[x | xt]

xt + α2(t)∇xt

1 α(t)

log pt(xt) γ(t)

xt − γ(t)

= −α(t)∇xt

log pt(xt). Thus, according to Lem. 9, we can obtain

=

fz⋆(Ft,xt,t) = −α(t)∇xt

log pt(xt).

#### Combine to obtain the claimed identity.

α′(t)fz⋆(Ft,xt,t) + γ′(t)fx⋆(Ft,xt,t)

xt + α2(t)∇xt

log pt(xt) γ(t)

= α′(t) −α(t)∇xt

log pt(xt) + γ′(t)

γ′(t) γ(t)

= −α(t)α′(t)∇xt

log pt(xt) +

γ′(t) γ(t)

α2(t)∇xt

xt +

log pt(xt)

γ′(t) γ(t)

γ′(t) γ(t)

xt − α(t)α′(t) −

α2(t) ∇xt

=

log pt(xt). This matches the claimed formula.

|Remark 1 (Velocity field of the flow ODE) . Given x and z, the field v(z,x)(y,t) = α′(t)z + γ′(t)x could transport z to x, so the velocity field of the flow ODE can be computed as<br><br>v∗(xt,t) = E(z,x)|x<br><br>t<br><br>v(z,x)(xt,t)|xt<br><br>= E(z,x)|x<br><br>t<br><br>[α′(t)z + γ′(t)x|xt]<br><br>= α′(t) · E[z|xt] + γ′(t) · E[x|xt]<br><br>= α′(t) · fz⋆(Ft,xt,t) + γ′(t) · fx⋆(Ft,xt,t).|
|---|

#### D.1.3 Closed-form Solution Analysis when λ = 0 Corollary 1 (Closed-form PF–ODE for an arbitrary Gaussian mixture in Rd) . Let

K

wj N x; mj, Σj , wj > 0,

p(x) =

wj = 1,

j=1

j

be a Gaussian-mixture density on Rd. Let α,γ satisfy the hypotheses of Lem. 7, and define the forward map

xt = α(t)z + γ(t)x, x ∼ p(x), z ∼ N(0,I). For each component j set

µj(t) = γ(t)mj, Σj(t) = γ(t)2 Σj + α(t)2 I, ϕj(xt) = N xt; µj(t),Σj(t) so that

K

wj N xt; µj(t), Σj(t) .

pt(xt) =

j=1

Then the Probability-Flow ODE (7) admits the closed-form drift

K

γ′(t) γ(t)

γ′(t) γ(t)

dxt dt

wj ϕj(xt) pt(xt)

xt + α(t)α′(t) −

Σj(t)−1 xt − µj(t) .

α(t)2

=

j=1

Proof. Step 1. Affine transform of a Gaussian mixture. Conditioned on the j-th component, x ∼ N(mj,Σj), and hence

xt = α(t)z + γ(t)x (j) ∼ N γ(t)mj, α(t)2I + γ(t)2Σj . Defining

µj(t) = γ(t)mj, Σj(t) = γ(t)2 Σj + α(t)2 I, we conclude that the marginal of xt is

pt(xt) =

K

wj N xt; µj(t),Σj(t) .

j=1

- Step 2. Score of the mixture. Set

ϕj(xt) = N xt; µj(t),Σj(t) , pt(xt) =

K

j=1

wj ϕj(xt).

Then by the usual mixture-rule,

∇xt

log pt =

1 pt(xt)

K

j=1

wj ϕj(xt)∇xt

log ϕj(xt).

Since for each Gaussian component

∇xt

log ϕj(xt) = −Σj(t)−1 xt − µj(t) , we obtain the closed-form score

∇xt

log pt(xt) = −

1 pt(xt)

K

j=1

wj N xt; µj(t),Σj(t) Σj(t)−1 xt − µj(t) .

- Step 3. Substitution into the PF–ODE. By Lem. 7, the Probability–Flow ODE reads

γ′(t) γ(t)

γ′(t) γ(t)

dxt dt

xt − α(t)α′(t) −

α(t)2 ∇xt

=

log pt(xt). Substituting the expression for ∇log pt above (and observing that the two “−” signs cancel) yields dxt dt

K

γ′(t) γ(t)

γ′(t) γ(t)

wj N xt; µj(t),Σj(t) pt(xt)

xt + α(t)α′(t)−

Σj(t)−1 xt−µj(t) ,

α(t)2

=

j=1

which is exactly the claimed closed-form drift.

Corollary 2 (Closed-form PF–ODE for a symmetric two-peak Gaussian mixture) . Let p(x) be the one-dimensional, symmetric, two-peak Gaussian mixture

p(x) = 21 N x;−m,σ2 + 12 N x;+m,σ2 , and let α,γ be as in Lem. 7. Define

xt = α(t)z + γ(t)x, Σt = γ(t)2 σ2 + α(t)2, µ±(t) = ±γ(t)m. Then the marginal density of xt is

pt(xt) = 21 N xt;µ−(t),Σt + 21 N xt;µ+(t),Σt ,

and the Probability-Flow ODE (7) admits the closed-form drift

γ′(t) γ(t)

γ′(t) γ(t)

dxt dt

1 Σt

γ(t)m Σt

xt + α(t)α′(t) −

α(t)2

xt − γ(t)m tanh

=

xt .

Proof. Step 1. Marginal law under the affine map. Conditional on x = ±m, one has

##### xt = αz + γx (x = ±m) ∼ N ±γm, α2 + γ2σ2 = N µ±(t),Σt .

Since each peak has weight 21, the marginal of xt is 12N(µ−,Σt) + 12N(µ+,Σt).

- Step 2. Score of the bimodal mixture. Write ϕ±(xt) = N(xt;µ±(t),Σt), so pt = 21(ϕ− + ϕ+). Then

d dxt

log pt =

1 pt

- 1

- 2 ϕ− ∇log ϕ− + ϕ+ ∇log ϕ+ , ∇log ϕ± = −

xt − µ±(t) Σt

. Hence

d dxt

log pt = −

1 2pt Σt

ϕ−(xt − µ−) + ϕ+(xt − µ+) . Define

r±(xt) =

ϕ±(xt) ϕ−(xt) + ϕ+(xt)

, ϕ− + ϕ+ = 2pt. Then

d dxt

log pt = −

1 Σt

r−(xt − µ−) + r+(xt − µ+) . A direct computation shows r+ − r− = tanh

γm Σt

xt , r−(xt + γm) + r+(xt − γm) = xt − γm tanh

γm Σt

xt . Therefore

d dxt

log pt = −

1 Σt

xt − γm tanh γmΣ

t

xt .

- Step 3. Substitution into the PF–ODE. By Lem. 7,

dxt dt

=

γ′ γ

xt − α α′ −

γ′ γ

α2

d dxt

log pt.

log pt carries a “−)” sign, the two negatives cancel, yielding exactly

Since ddx

t

γ′ γ

γ′ γ

dxt dt

1 Σt

xt − γm tanh γmΣ

xt + α α′ −

α2

xt , as claimed.

=

t

|Remark 2 (OU-type schedule for the symmetric bimodal case) . Specialize Cor. 2 to the Ornstein–Uhlenbeck-type schedule with<br><br>γ(t) = e−st, α(t) = 1 − e−2st,<br><br>and noise variance σ2 in each mixture component. Then the marginal variance is<br><br>Σt = γ(t)2 σ2 + α(t)2 = σ2e−2st + (1 − e−2st), and one obtains the closed-form drift of the Probability-Flow ODE:<br><br>|dxt dt<br><br>= −sxt +<br><br>s Σt<br><br>xt − me−st tanh<br><br>me−st Σt<br><br>xt .|
|---|
|
|---|

Proof. We start from the general drift in Cor. 2:

γ′ γ

γ′ γ

dxt dt

1 Σt

γ m Σt

xt + α α′ −

α2

xt − γ m tanh

xt . We now substitute γ(t) = e−st, α(t) = √1 − e−2st and compute each piece in detail: Derivative of γ:

=

γ′(t) γ(t)

γ′(t) = −se−st, =⇒

= −s.

Marginal variance Σt:

Σt = γ(t)2 σ2 + α(t)2 = σ2 e−2st + (1 − e−2st). Square of α and its derivative:

d dt

α(t)2 = 2se−2st =⇒ 2α α′ = 2se−2st =⇒ α α′ = se−2st. Combination term

α(t)2 = 1 − e−2st,

γ′ γ

α2 = se−2st − (−s)(1 − e−2st) = s e−2st + 1 − e−2st = s.

α α′ −

Substitution into the general drift formula gives dxt dt

1 Σt

−st m

xt − e−st m tanh e

= −sxt + s

Σt xt . Hence the final, closed-form Probability-Flow ODE is

me−st Σt

dxt dt

s Σt

xt − me−st tanh

= −sxt +

xt , where Σt = σ2e−2st + (1 − e−2st).

|Remark 3 (Triangular schedule for the symmetric bimodal case) . Specialize Cor. 2 to the trigonometric schedule<br><br>γ(t) = cos π2 t , α(t) = sin π2 t , with noise variance σ2 in each mixture component. Then<br><br>Σt = γ(t)2 σ2 + α(t)2 = σ2 cos2 π2 t + sin2 π2 t , and the closed-form drift of the Probability-Flow ODE is<br><br>|dxt dt<br><br>= −<br><br>π 2<br><br>tan π2 t xt +<br><br>π 2 tan π2 t<br><br>Σt<br><br>xt − cos π2 t m tanh<br><br>cos(π2t)m Σt<br><br>xt .|
|---|
|
|---|

Proof. We begin with the general drift in Cor. 2:

γ′ γ

γ′ γ

dxt dt

1 Σt

γ m Σt

xt + α α′ −

α2

xt − γ m tanh

xt . For γ(t) = cos(π2t), α(t) = sin(π2t),

=

γ′ γ

γ′(t) = −π2 sin π2 t = −π2 α(t),

= −π2 tan π2 t . And

α′(t) = π2 cos π2 t = π2 γ(t), so that

γ′ γ

α3 γ

π 2

π 2

π 2

α γ

π 2

α α′ −

α2 =

α2 + γ2 =

tan π2 t . Substituting into the general formula immediately yields the boxed drift.

α γ +

=

| |
|---|

|Remark 4 (Linear schedule for the symmetric bimodal case) . Specialize Cor. 2 to the "Linear" schedule<br><br>γ(t) = 1 − t, α(t) = t, t ∈ [0,1]. Then the marginal variance is<br><br>Σt = γ(t)2 σ2 + α(t)2 = (1 − t)2 σ2 + t2, and one obtains the closed-form drift of the Probability-Flow ODE:<br><br>|dxt dt<br><br>= −<br><br>xt 1 − t<br><br>+<br><br>t (1 − t)Σt<br><br>xt − m(1 − t) tanh<br><br>m(1 − t) Σt<br><br>xt .|
|---|
|
|---|

Proof. We begin with the general drift formula from Cor. 2:

γ′(t) γ(t)

γ′(t) γ(t)

1 Σt

dxt dt

γ(t)m Σt

xt + α(t)α′(t) −

α(t)2

xt − γ(t)m tanh

xt . We substitute γ(t) = 1 − t and α(t) = t and compute each piece:

=

- 1. Derivative of γ:

γ′(t) = −1, =⇒

γ′(t) γ(t)

= −

1 1 − t

.

- 2. Marginal variance: Σt = (1 − t)2 σ2 + t2.
- 3. Square of α and its derivative:

α(t)2 = t2,

d dt

α(t)2 = 2t =⇒ 2α α′ = 2t =⇒ α(t)α′(t) = t.

- 4. Combination term:

γ′ γ

t2 1 − t

1 1 − t

t 1 − t

α α′ −

α2 = t − −

t2 = t +

=

.

Substituting these into the general drift gives

m(1 − t) Σt

dxt dt

xt 1 − t

t (1 − t)Σt

= −

xt − m(1 − t) tanh

+

xt , which is the claimed closed-form Probability-Flow ODE.

Remark 5 (OU-type schedule for the Hermite–Gaussian n = 1 case) . Apply Lem. 7 to the one-dimensional Hermite–Gaussian initial density

2/2, x > 0, and the OU-type schedule

p1(x) ∝ xe−x

γ(t) = e−st, α(t) = 1 − e−2st. Then the Probability–Flow ODE (7) reduces to the scalar form

|dxt dt<br><br>= −<br><br>s xt<br><br>, t ∈ [0,1],|
|---|

and integrating from t = 1 (with x(1) = x1) to any t ∈ [0,1] yields the explicit solution

|xt = x21 + 2s(1 − t).<br><br>|
|---|

Proof. By Lem. 7, the drift of the Probability–Flow ODE is

γ′(t) γ(t)

dxt dt

′(t)

xt − α(t)α′(t) − γ

γ(t) α(t)2 ∂x

lnpt(xt). Under γ(t) = e−st and α(t) = √1 − e−2st one computes

=

t

γ′ γ

γ′ γ

α2 = s(1 − e−2st), hence

= −s, 2α α′ = 2se−2st =⇒ α α′ = se−2st, −

γ′ γ

α2 = se−2st + s(1 − e−2st) = s. Moreover, one checks that the marginal density remains pt(x) ∝ xe−x

α α′ −

2/2, so ∂x lnpt(x) = x1 − x. Therefore

dxt dt

s xt

= −sxt − s x 1

− xt = −

. Separating variables,

t

t

xt

x2t − x21 2

s x

dx dt

ds =⇒

xdx = −s

= −s(t − 1), whence

= −

=⇒

1

x1

x2t = x21 + 2s(1 − t), xt = x21 + 2s(1 − t), taking the positive root on x > 0.

|Lemma 10 (Picard–Lindelöf existence and uniqueness) . Let v: R × [0,1] → R be continuous in t and satisfy the uniform Lipschitz condition<br><br>|v(x,t) − v(y,t)| ≤ L|x − y|, ∀x,y ∈ R, t ∈ [0,1],<br><br>for some constant L < ∞. Then for any t0 ∈ [0,1] and any initial value x(t0) = x0, there exists δ > 0 and a unique function<br><br>x ∈ C1 [t0 − δ,t0 + δ] ∩ [0,1] solving the ODE<br><br>dx dt<br><br>(t) = v x(t),t , x(t0) = x0.|
|---|

Proof. Fix t0 ∈ [0,1] and x0 ∈ R. Choose δ > 0 so small that (t0 − δ,t0 + δ) ⊂ [0,1] and Lδ < 1. Define the closed ball

BR = x ∈ C([t0 − δ,t0 + δ],R) : ∥x − x0∥∞ ≤ R with R > 0 to be chosen. Consider the operator

t

(Γx)(t) = x0 +

##### v x(s),s ds.

t0

Since v is continuous on the compact set BR × [t0 − δ,t0 + δ], it is bounded by some M < ∞. If we choose R = Mδ, then Γ maps BR into itself:

t

|v(x(s),s)|ds ≤ M δ = R. Moreover, for any x,y ∈ BR and any t in the interval,

∥Γx − x0∥∞ ≤ sup

t

t0

t

##### |v(x(s),s) − v(y(s),s)|ds ≤ Lδ ∥x − y∥∞ < ∥x − y∥∞,

|(Γx)(t) − (Γy)(t)| ≤

t0

so Γ is a contraction. By the Banach fixed-point theorem, Γ has a unique fixed point in BR, which is precisely the unique C1 solution of the ODE on [t0 − δ,t0 + δ] ∩ [0,1].

| |
|---|

|Lemma 11 (Gronwall’s inequality and no blow-up) . Let x ∈ C1([0,1]) satisfy<br><br>|x′(t)| ≤ K 1 + |x(t)| , t ∈ [0,1], for some constant K ≥ 0. Then<br><br>|x(t)| ≤ |x(1)| + 1 eK(1−t) − 1, ∀t ∈ [0,1], and in particular x does not blow up in finite time on [0,1].|
|---|

Proof. Define

y(t) = |x(t)| + 1 ≥ 1. Since y(t) is Lipschitz, for almost every t we have

d dt |x(t)| + 1 = sgn(x(t))x′(t),

y′(t) =

and hence

y′(t) ≥ −|x′(t)| ≥ −K 1 + |x(t)| = −K y(t). Equivalently,

y′(t) + K y(t) ≥ 0. Multiply both sides by the integrating factor eKt:

d dt

eKty(t) = eKt y′(t) + K y(t) ≥ 0.

Thus the function t  → eKty(t) is non-decreasing on [0,1]. For any t ≤ 1 we then have

eKty(t) ≤ eK·1y(1) =⇒ y(t) ≤ y(1)eK(1−t) = |x(1)| + 1 eK(1−t). Rewriting y(t) = |x(t)| + 1 gives

|x(t)| ≤ |x(1)| + 1 eK(1−t) − 1,

as claimed. In particular |x(t)| < ∞ for all t ∈ [0,1], so no finite-time blow-up occurs.

|Lemma 12 (Gaussian convolution preserves linear-growth bound) . Let p0 ∈ C1(R) be a probability density satisfying<br><br>∂x log p0(x) ≤ A + B |x|, A,B < ∞, ∀x ∈ R, and assume furthermore that ∥p0∥∞ = supx∈R p0(x) ≤ M < ∞. For each σ > 0, define the Gaussian kernel ϕσ(u) = √21πσ exp − u<br><br>2<br><br>2σ2 , and set pσ(x) = (p0 ∗ ϕσ)(x) = R p0(y)ϕσ(x − y)dy. Then pσ ∈ C∞(R) and there exist<br><br>A(σ) = A + B M σ π2, B(σ) = B, such that<br><br>∂x log pσ(x) ≤ A(σ) + B(σ)|x|, ∀x ∈ R.|
|---|

Proof. Smoothness and differentiation under the integral. Since ϕσ ∈ C∞(R) decays rapidly and p0 ∈ L∞(R), by dominated convergence we may differentiate under the integral to get

p′σ(x) =

p0(y)∂xϕσ(x − y)dy =

R

Noting ∂yϕσ(x − y) = −ϕ′σ(x − y), we rewrite

p0(y)ϕ′σ(x − y)dy.

R

p′σ(x) = −

p0(y)∂yϕσ(x − y)dy.

R

[Figure 7]

[Figure 8]

###### (a) OU-type. (b) Linear.

Figure 7: Comparison of two optimal Probability-Flow ODE trajectories on 1D data. Starting from identical initial noise distributions and noise points, we apply two distinct transport types—OU-type and Linear—to analyze their trajectories. The results show that both types successfully converge to the same target distribution (a bimodal Gaussian) and accurately match the same target data points, despite following different ODE paths.

Integration by parts. Integrating the above in y and using that p0(y)ϕσ(x − y) → 0 as |y| → ∞, we obtain

p′σ(x) =

p′0(y)ϕσ(x − y)dy =

(∂y log p0)(y)p0(y)ϕσ(x − y)dy.

R

R

Bounding ∂x log pσ. Hence

∂x log pσ(x) = |p′σ(x)| pσ(x)

(∂y log p0)(y)p0(y)ϕσ(x − y)dy

=

pσ(x) ≤

A + B|y| p0(y)ϕσ(x − y)dy pσ(x)

|∂y log p0(y)|p0(y)ϕσ(x − y)dy pσ(x) ≤

= A + B |y|p0(y)ϕσ(x − y)dy pσ(x)

.

Change of variables. Set u = y − x. Then

|y|p0(y)ϕσ(x−y)dy = |x+u|p0(x+u)ϕσ(u)du ≤ |x|pσ(x)+ |u|p0(x+u)ϕσ(u)du. Hence

|y|p0(y)ϕσ(x − y)dy pσ(x) ≤ |x| + |u|p0(x + u)ϕσ(u)du pσ(x)

.

Using the L∞-bound on p0. Since p0(x + u) ≤ M,

|u|p0(x + u)ϕσ(u)du ≤ M |u|ϕσ(u)du = M σ π2. Conclusion. Combining the above estimates yields

∂x log pσ(x) ≤ A + B |x| + M σ π2 = A + B M σ π2 + B |x|. Thus one may set

A(σ) = A + B M σ π2, B(σ) = B, and the lemma follows.

Theorem 3 (Monotonicity and uniqueness of the 1D probability-flow map) . Let p0(x) be a probability density on R satisfying the linear-growth bound

∂x log p0(x) ≤ A + B |x|, A,B < ∞, ∀x ∈ R.

Let z ∼ N(0,1) be independent of x0, and let α,γ : [0,1] → R be C1 functions with

α(0) = 0, α(1) = 1, γ(0) = 1, γ(1) = 0, γ(t) ̸= 0 ∀t ∈ (0,1). Define the forward process

xt = α(t)z + γ(t)x0, t ∈ [0,1], so that x0 ∼ p0 and x1 ∼ N(0,1). Let pt denote the density of xt. By Lem. 7, the velocity field:

γ′(t) γ(t)

γ′(t) γ(t)

x − α(t)α′(t) −

α(t)2 ∂x log pt(x).

v(x,t) =

Consider the backward ODE ddt xt = v xt,t , Then for each x1 ∈ R there is a unique C1 solution

- t  → xt(x1) on [0,1], and the map

g(x1) = x0(x1) = F0−1 F1(x1)

is strictly increasing on R and is the unique increasing transport pushing p1 onto p0.

Proof. (1) Global existence and uniqueness. Since

##### xt = α(t)z + γ(t)x0, pt = p0 ∗ N 0,α(t)2 ,

standard Gaussian-convolution estimates imply ∂x log pt(x) ≤ At + Bt|x| for some continuous At,Bt (cf., Lem. 12). Hence there exists K < ∞ such that

##### |v(x,t)| ≤ K (1 + |x|), ∂xv(x,t) ≤ K, ∀x ∈ R, t ∈ [0,1].

In particular v is globally Lipschitz in x (uniformly in t) and of linear growth. By the Lem. 10 together with Lem. 11 to prevent finite-time blow-up, the backward ODE admits for each x1 a unique C1 solution on [0,1].

- (2) Conservation of the CDF. Let

Ft(x) =

x

−∞

pt(u)du (the CDF of pt).

Since pt satisfies the continuity equation ∂tpt + ∂x(v pt) = 0, along any characteristic t  → xt one computes

d dt

Ft(xt) =

xt

−∞

∂tpt(u)du + pt(xt)

dxt dt

= − v pt x−∞t + pt(xt)v(xt,t) = 0, using limu→−∞ pt(u) = 0. Hence Ft(xt) = F1(x1) for all t ∈ [0,1].

- (3) Quantile representation. Evaluating at t = 0 gives F0 x0(x1) = F1(x1).

Since F0: R → (0,1) is strictly increasing and onto, it has an inverse F0−1, and thus

x0(x1) = F0−1 F1(x1) .

- (4) Monotonicity and uniqueness. If x1 < y1 then F1(x1) < F1(y1), so

g(x1) = F0−1 F1(x1) < F0−1 F1(y1) = g(y1), showing g is strictly increasing. In one dimension the strictly increasing transport between two given laws is unique, so g is the unique increasing map pushing p1 onto p0. A case study presented in Fig. 7 validates this theorem, considering the specific schedules discussed in Rem. 4 and Rem. 2.

| |
|---|

|Lemma 13 (Monotone transport from Gaussian to P) . Let Z ∼ N(0,1) be a standard normal random variable and let X be a random variable with distribution P on R, having cumulative distribution function (CDF) FP. Define<br><br>Φ(z) = Pr[Z ≤ z], FP−1(u) = inf{x : FP(x) ≥ u}, u ∈ (0,1).<br><br>Then there exists a non-decreasing continuous function g(z) = FP−1 Φ(z) such that g(Z) =d X if and only if P has no atoms (i.e. FP is continuous). Moreover, if FP is strictly increasing then g is unique.|
|---|

Proof. Existence. Since Φ : R → (0,1) is continuous and strictly increasing, the random variable

##### U = Φ(Z)

is distributed uniformly on (0,1). Hence for any x ∈ R,

Pr FP−1(U) ≤ x = Pr U ≤ FP(x) = FP(x),

so FP−1(U) has distribution P. The quantile function FP−1 is non-decreasing and, by standard results on generalized inverses (see e.g. Billingsley, Probability and Measure), is continuous on (0,1) if and

only if FP is continuous. Therefore

g(z) = FP−1 Φ(z)

is non-decreasing and continuous exactly when FP is continuous, and in that case g(Z) =d X.

Necessity. Suppose P has an atom at x0, i.e. Pr[X = x0] = p > 0. If there were a continuous non-decreasing g with g(Z) =d X, then to produce a point-mass p at x0 it would have to be constant on a set of positive Pr-mass in the continuous law of Z. But continuity of g then forces it to be constant on a strictly larger interval, yielding a mass > p at x0, a contradiction. Thus FP must be continuous.

Uniqueness. Let g1,g2 be two continuous non-decreasing functions with gi(Z) =d P. Define for

- u ∈ (0,1) hi(u) = gi Φ−1(u) , i = 1,2.

Each hi is continuous, non-decreasing, and pushes Unif(0,1) onto P. When FP is strictly increasing, its quantile FP−1 is the unique such map (classical uniqueness of quantile functions for atomless laws). Hence h1 ≡ h2 ≡ FP−1 on (0,1), and therefore g1 ≡ g2 on R.

| |
|---|

#### D.1.4 Learning Objective as λ → 1

|Lemma 14 (Lp-estimate for the difference of two absolutely continuous functions) . Let I = [a,b] be a compact interval and (E,∥ · ∥) a Banach space. Suppose f,g : I → E are absolutely continuous with Bochner–integrable derivatives f′,g′. Fix 1 ≤ p ≤ ∞. Then<br><br>∥f − g∥Lp(I;E) ≤ (b − a)1/p f(a) − g(a) +<br><br>b<br><br>a<br><br>(b − s)1/p f′(s) − g′(s) ds,<br><br>where for p = ∞ one interprets (b − s)1/p = 1. Moreover, if 1 < p < ∞ and p′ denotes the conjugate exponent 1/p + 1/p′ = 1, then by Hölder’s inequality one further deduces<br><br>∥f − g∥Lp(I;E) ≤ (b − a)1/p f(a) − g(a) + p−p1<br><br>1/p′<br><br>(b − a)∥f′ − g′∥Lp(I;E).|
|---|

Proof. Since f and g are absolutely continuous on [a,b], the Fundamental Theorem of Calculus in the Bochner setting gives, for each t ∈ [a,b],

t

f′(s) − g′(s) ds.

f(t) − g(t) = f(a) − g(a) +

a

Set X(s) = f′(s) − g′(s). Then for every t ∈ [a,b],

t

f(t) − g(t) ≤ f(a) − g(a) +

X(s)ds . We now distinguish two cases.

a

- Case 1: 1 ≤ p < ∞. Taking the Lp–norm in the variable t over [a,b] and applying Minkowski’s integral inequality for Bochner integrals yields

∥f − g∥L

p t

≤ f(a) − g(a) 1 L

p([a,b]) +

t

a

X(s)ds

Lpt

= (b − a)1/p f(a) − g(a) +

b

a

t

a

X(s)ds

p

dt

1/p

≤ (b − a)1/p f(a) − g(a) +

b

a

1[s,b](·)X(s) Lp

t

ds.

Here we have written a t X(s)ds = a b 1[a,t](s)X(s)ds and used the fact that

1[s,b](t) Lp

t

=

b

a

1[s,b](t)dt

1/p

= (b − s)1/p. Hence

∥f − g∥Lp(I;E) ≤ (b − a)1/p f(a) − g(a) +

b

a

(b − s)1/p X(s) ds, which is the claimed Lp–estimate.

- Case 2: p = ∞. Taking the essential supremum in t ∈ [a,b] in the pointwise bound ∥f(t) − g(t)∥ ≤ ∥f(a) − g(a)∥ + a t ∥X(s)∥ds gives immediately

b

∥f − g∥L∞(I;E) ≤ ∥f(a) − g(a)∥ +

∥X(s)∥ds,

a

which agrees with the above formula when (b − s)1/p = 1. Refinement for 1 < p < ∞. Let p′ be the conjugate exponent, 1/p + 1/p′ = 1. Applying Hölder’s inequality to the integral a b(b − s)1/p ∥X(s)∥ ds gives

b

b

1/p′ b

1/p

′/p ds

(b − s)1/p ∥X(s)∥ds ≤

(b − s)p

∥X(s)∥p ds

##### .

a

a

a

Since p′/p = 1/(p − 1), a direct computation yields

b−a

b

p − 1 p

′/p ds =

′

u1/(p−1) du =

(b − s)p

(b − a)p

. Hence

a

0

b

1/p′

1/p′

′/p ds

= p−p1

(b − s)p

(b − a), and we arrive at

a

b

1/p′

(b − s)1/p ∥X(s)∥ds ≤ p−p1

(b − a)∥X∥Lp(I;E).

a

Combining this with the previous display completes the proof of the refined estimate.

| |
|---|

|Lemma 15 (Uniqueness of absolutely continuous functions) . Let I = [a,b] be a compact interval and (E,∥ · ∥) a Banach space. Suppose f,g : I → E are absolutely continuous with Bochner–integrable derivatives f′,g′. If<br><br>f(a) = g(a) and f′(t) = g′(t) for almost every t ∈ I, then f(t) = g(t) for all t ∈ I.|
|---|

Proof. Apply Lem. 14 (the Lp–estimate for differences) in the case p = ∞. Since in this case one has

b − s 1/p = 1, ∥f(a) − g(a)∥ = 0, ∥f′(s) − g′(s)∥ = 0 a.e., the conclusion of Lem. 14 reads

b

∥f′(s) − g′(s)∥ds = 0.

∥f − g∥L∞(I;E) ≤ ∥f(a) − g(a)∥ +

a

Hence ∥f − g∥L∞(I;E) = 0, which means

∥f(t) − g(t)∥ = 0,

sup

t∈I

so f(t) = g(t) for every t ∈ I.

|Theorem 4 (Pathwise consistency via zero total derivative) . Let p(x) be a data distribution on Rd, and let z ∼ N(0,Id) be independent of x. Let α,γ : [0,1] → R be C1 scalar functions satisfying<br><br>α(0) = 0, α(1) = 1, γ(0) = 1, γ(1) = 0, γ(t) = 0 ∀t ∈ (0,1). Define the forward process<br><br>xt = α(t)z + γ(t)x, t ∈ [0,1],<br><br>so that x0 = x ∼ p(x) and x1 = z ∼ N(0,I). Let pt be the law of xt. By Lem. 7 the corresponding Probability Flow ODE is<br><br>v(xt,t) =<br><br>d dt<br><br>xt =<br><br>γ′(t) γ(t)<br><br>xt − α(t)α′(t) −<br><br>γ′(t) γ(t)<br><br>α(t)2 ∇xt<br><br>log pt(xt). Given any point xt, define<br><br>g(xt,t) = x0 = xt +<br><br>0<br><br>t<br><br>v(xu,u)du.<br><br>Let (z,x) ∼ p(x) ⊗ N(0,I) and t ∼ Unif[0,1] be all mutually independent. Write E(z,x) for expectation over (z,x) and E(z,x),t for expectation over (z,x) and t. Suppose<br><br>E(z,x) f(x0,0) − g(x0,0) = 0, E(z,x),t<br><br>d dt<br><br>f(xt,t) = 0. Then<br><br>E(z,x),t f(xt,t) − g(xt,t) = 0.|
|---|

̸

Proof. Fix a draw (z,x). Along its forward trajectory xt = α(t)z + γ(t)x, define the two curves

f(t) = f xt,t , g(t) = g xt,t . We check the hypotheses of Lem. 15 for f,g : [0,1] → Rd. Absolute continuity. Since f is C1 in (x,t) and t  → xt is C1, the composition f(t) = f(xt,t) is absolutely continuous, with

f′(t) =

d dt

f xt,t , existing a.e.

Also

0

t

v(xu,u)du = x0 −

g(t) = xt +

v(xu,u)du

t

0

is the sum of a C1 function and an absolutely continuous integral, hence itself absolutely continuous. Coincidence of initial values. From E(z,x)∥f(x0,0) − g(x0,0)∥ = 0 we get f(x0,0) = g(x0,0) almost surely, so f(0) = g(0) for almost every (z,x).

Coincidence of derivatives a.e. By Tonelli–Fubini,

1

0 = E(z,x),t d dtf(xt,t) =

d dtf(xt,t) dt dP(z,x).

0

Hence for almost every (z,x), 0 1 ∥∂tf(xt,t)∥dt = 0, which forces ∂tf(xt,t) = 0 for almost all t. Thus

f′(t) = 0 for a.e. t ∈ [0,1]. On the other hand

dxt dt − v(xt,t) = v(xt,t) − v(xt,t) = 0, ∀t ∈ [0,1].

g′(t) =

Conclusion by uniqueness. We have shown f,g are absolutely continuous, f(0) = g(0), and

- f′(t) = g′(t) for almost every t. By Lem. 15, f(t) = g(t) for all t ∈ [0,1] (almost surely in (z,x)). Hence f(xt,t) = g(xt,t) a.s., and taking expectation yields E(z,x),t f(xt,t) − g(xt,t) = 0.

|Remark 6 (Consistency-training loss) . By Thm. 4, to enforce f(xt,t) ≈ g(xt,t) = x0 along the PF–ODE flow, we suggests two equivalent training objectives:<br><br>1. Continuous PDE-residual loss<br><br>LPDE = Et,x<br><br>t<br><br>∂tf(xt,t) + v(xt,t)·∇xt<br><br>f(xt,t)<br><br>2<br><br>.<br><br>2. Finite-difference consistency loss<br><br><br>Lcons = Et,x<br><br>0,z f xt+∆t, t + ∆t − f xt,t<br><br>2<br><br>, where xt = α(t)z + γ(t)x0 and similarly for xt+∆t.|
|---|

Proof. We begin from the requirement that f(xt,t) remain constant along the flow:

d dt

f(xt,t) = ∂t + v(xt,t) · ∇xt

This is exactly the linear transport PDE

dxt dt = v(xt,t)

f(xt,t) = ∂tf(xt,t) +

·∇xt

f(xt,t) = 0.

##### (∂t + v · ∇)f(x,t) = 0.

To train a network f to satisfy it, one may minimize the L2-residual over the joint law of t and xt, yielding

2

LPDE = Et,x

∂tf(xt,t) + v(xt,t)·∇xt

f(xt,t)

. In practice, computing the spatial gradient ∇xt

t

f can be expensive. Instead, we use a small time increment ∆t and the finite-difference approximation

##### f(xt+∆t, t + ∆t) − f(xt,t) ≈ ∆t ∂tf + v · ∇f (xt,t).

Squaring and taking expectations over t,x0,z then yields the discrete consistency loss

2

Lcons = Et,x

0,z f xt+∆t, t + ∆t − f xt,t

##### .

This completes the derivation of both forms of the consistency-training objective.

| |
|---|

Recall that (z,x) ∼ p(z,x) is a pair of latent and data variables (typically independent), and let t ∈ [0,1]. We have four differentiable scalar functions α,γ,α,ˆ γˆ: [0,1] → R , the noisy interpolant xt = α(t)z + γ(t)x and Ft = Fθ(xt,t). We define the x- and z-prediction functions by

α(t)Ft − αˆ(t)xt α(t) ˆγ(t) − αˆ(t)γ(t)

γˆ(t)xt − γ(t)Ft α(t) ˆγ(t) − αˆ(t)γ(t)

fx(Ft,xt,t) =

, and fz(Ft,xt,t) =

. Since

α(0) · Fθ(x0,0) − αˆ(0) · x0 α(0) · γˆ(0) − αˆ(0) · γ(0)

fx(F0,x0,0) =

0 · Fθ(x0,0) − αˆ(0) · x0 0 · γˆ(0) − αˆ(0) · 1

=

0 − αˆ(0) · x0 0 − αˆ(0)

=

= x0

fx satisfies the boundary condition of consistency models [41] and Thm. 4. To better understand the unified loss, let’s analyze a bit further. For simplicity we use the notation fθ(xt,t) := fx(Fθ(xt,t),xt,t), the training objective is then equal to

1 ωˆ(t)∥fθ(xt,t) − fθ−(xλt,λt)∥22 .

L(θ) = Et,(z,x)

Let ϕt(x) be the solution of the PF-ODE determined by the velocity field v∗(xt,t) = E(z,x)|x

v(z,x)(xt,t)|xt (where v(z,x)(y,t) = α′(t)z + γ′(t)x) and an initial value x at time

t

t = 0. Define gθ(x,t) := fθ(ϕt(x),t) that moves along the solution trajectory. When λ → 1, the gradient of the loss tends to

L(θ) 2(1 − λ)

fθ(xt,t) − fθ(xλt,λt) t − λt

t ωˆ(t) · E(z,x) lim

= Et

⟨

,∇θfθ(xt,t)⟩

∇θ

lim

λ→1

λ→1

t ωˆ(t) · E(z,x)⟨

dfθ(xt,t) dt

,∇θgθ(ϕ−t 1(xt),t)⟩ The inner expectation can be computed as:

= Et

dfθ(xt,t) dt

,∇θgθ(ϕ−t 1(xt),t)⟩

E(z,x),x

t⟨

t⟨∂1fθ(xt,t) · v(z,x)(xt,t) + ∂2fθ(xt,t),∇θgθ(ϕ−t 1(xt),t)⟩

= E(z,x),x

t⟨∂1fθ(xt,t) · (α′(t)z + γ′(t)x) + ∂2fθ(xt,t),∇θgθ(ϕ−t 1(xt),t)⟩

= E(z,x),x

t⟨∂1fθ(xt,t) · (α′(t)z + γ′(t)x) + ∂2fθ(xt,t),∇θgθ(ϕ−t 1(xt),t)⟩

= Ex

E(z,x)|x

t

[α′(t)z + γ′(t)x|xt] + ∂2fθ(xt,t),∇θgθ(ϕ−t 1(xt),t)⟩

= Ex

t⟨∂1fθ(xt,t) · E(z,x)|x

t

t⟨∂1fθ(xt,t) · v∗(xt,t) + ∂2fθ(xt,t),∇θgθ(ϕ−t 1(xt),t)⟩

= Ex

t⟨∂2gθ(ϕ−t 1(xt),t),∇θgθ(ϕ−t 1(xt),t)⟩

= Ex

- 1

- 2∥gθ(ϕ−t 1(xt),t) − gθ−(ϕ−t 1(xt),t) + ∂2gθ(ϕ−t 1(xt),t)∥22

= ∇θEϕ−1

t (xt)

Thus from the perspective of gradient, when λ → 1 the training objective is equivalent to

t ωˆ(t) · ∥gθ(ϕ−t 1(xt),t) − gθ−(ϕ−t 1(xt),t) + ∂2gθ(ϕ−t 1(xt),t)∥22

Eϕ−1

t (xt),t

which naturally leads to the solution gθ(x,t) = x (since gθ(x,0) ≡ x), or equivalently fx(Fθ∗(xt,t),xt,t) = fθ∗(xt,t) = ϕ−t 1(xt), that is the definition of consistency function.

#### D.1.5 Analysis on the Optimal Solution for λ ∈ [0,1]

Below we provide some examples to illustrate the property of the optimal solution for the unified loss by considering some simple cases of data distribution.

(for simplicity define fθ(xt,t) = fx(Fθ(xt,t),xt,t)) Assume x ∼ N(µ,Σ). For r < t the conditional mean

E [xr|xt] = γ(r)µ + (γ(r)γ(t)Σ + α(r)α(t)I) γ(t)2Σ + α(t)2I −1 (xt − γ(t)µ) , denote

K(r,t) := (γ(r)γ(t)Σ + α(r)α(t)I) γ(t)2Σ + α(t)2I −1 , using above equations we can get the optimal solution for diffusion model:

##### fDMθ∗ (xt,t) = E [x|xt] = µ + K(0,t)(xt − γtµ).

Now consider a series of t together: t = tT > tT−1 > ... > t1 > t0 ≈ 0. This series could be obtained by tj−1 = λ · tj,j = T,...,0, for instance. With an abuse of notation, denote xt

j as xj and α(tj) as αj, γ(tj) as γj. Since t0 ≈ 0,x0 ≈ x, we could conclude the trained model fθ∗(x1,t1) = Ex|x

##### [x|x1], and concequently fθ∗(xj+1,tj+1) = Ex

1

j|xj+1 [fθ∗(xj,tj)|xj+1] , j = 1,...,T − 1. Using the property of the conditional expectation, we have Ex

[fθ∗(xj,tj)] = Ex [x] ,∀j. Using the expressions above we have

j

fθ∗(x1,t1) = µ + K(t0,t1)(x1 − γ1µ) and

- j
- k=1

K(tk−1,tk) · (xt − γtµ), j = 2,...,T

fθ∗(xj,tj) = µ +

Further denote cj = jk=1 αk−1αk + γk−1γk and assume Σ = I,α = sin(t),γ(t) = cos(t). For appropriate choice of the partition scheme (e.g. even or geometric), the coefficient cj can converge as T grows. For instance, when evenly partitioning the interval [0,t], we have:

T

t T

))T = 1.

αk−1αk + γk−1γk = lim

(cos(

lim

c(t) = lim

T→∞

T→∞

T→∞

k=1

Thus the trained model can be viewed as an interpolant between the consistency model(λ → 1 or T → ∞) and the diffusion model(λ → 0 or T → 1):

fθ∗(xt,t) = µ + c(t)(xt − γ(t)µ), fCMθ∗ (xt,t) = µ + (xt − γ(t)µ), fDMθ∗ (xt,t) = µ + γ(t)(xt − γ(t)µ).

The expression of fCMθ∗ can be obtained by first compute the velocity field v∗(xt,t) = E [α′(t)z + γ′(t)x|xt] = γ′(t)µ then solve the initial value problem of ODE to get x(0).

The above optimal solution can be possibly obtained by training. For example if we set the parameterizition as fθ(xt,t) = (1 − γtct)θ + ctxt, the gradient of the loss can be computed as (let r = λ · t):

∇θ∥fθ(xt,t) − fθ−(xr,r)∥22 = 2(1 − γtct)[(αtγt − αrγr)z + (γrcr − γtct)(θ − x)] ,

∇θEz,x ∥fθ(xt,t) − fθ−(xr,r)∥22 = 2(1 − γtct)(γrcr − γtct)(θ − µ),

2(1 − γtct)(γrcr − γtct) ωˆ(t)

∇θL(θ) = Et

(θ − µ)

2(1 − γtct)(γrcr − γtct) ωˆ(t)

= C(θ − µ), C = Et

.

Use gradient descent to update θ during training:

dθ(s) ds

= −∇θL(θ) = −C(θ − µ). The generalization loss thus evolves as:

d∥θ(s) − µ∥2 ds

dθ(s) ds ⟩

= ⟨θ(s) − µ,

= ⟨θ(s) − µ,−C(θ(s) − µ)⟩

= −C∥θ(s) − µ∥2 ,

=⇒ ∥θ(s) − µ∥2 = ∥θ(0) − µ∥2e−Cs .

- D.1.6 Enhanced Target Score Function Recall that CFG proposes to modify the sampling distribution as

p˜θ(xt|c) ∝ pθ(xt|c)pθ(c|xt)ζ , Bayesian rule gives

pθ(xt|c)pθ(c) pθ(xt)

pθ(c|xt) =

, so we can futher deduce

p˜θ(xt|c) ∝ pθ(xt|c)pθ(c|xt)ζ = pθ(xt|c)(

pθ(xt|c)pθ(c) pθ(xt)

)ζ

pθ(xt|c) pθ(xt)

)ζ .

∝ pθ(xt|c)(

When t ∈ [0,s] (s = 0.75), inspired by above expression and a recent work [44], we choose to use below as the target score function for training

ζ

pt,θ(xt|c) pt,θ(xt)

∇xt

log pt(xt|c)

which equals to

∇xt

log pt(xt|c) + ζ (∇xt

log pt,θ(xt|c) − ∇xt

log pt,θ(xt)) . For fz⋆ we originally want to learn:

##### fz⋆(Ft,xt,t) = −α(t)∇xt

log pt(xt), now it turns to

ζ

pt,θ(xt|c) pt,θ(xt)

fz⋆(Ft,xt,t) = −α(t)∇xt

log pt(xt|c)

= −α(t)[∇xt

log pt(xt|c) + ζ (∇xt

log pt,θ(xt|c) − ∇xt

log pt,θ(xt))]

= −α(t)∇xt

log pt(xt|c) + ζ (−α(t)∇xt

log pt,θ(xt|c) + α(t)∇xt

log pt,θ(xt))

log pt(xt|c) + ζ fz(Ft,xt,t) − fz(F∅t ,xt,t) , thus in training we set the objective for fz as:

= −α(t)∇xt

z⋆ ← z + ζ · fz(Ft,xt,t) − fz(F∅t ,xt,t) . Similarly, since fx⋆ = xt+α

2(t)∇xt log pt(xt)

γ(t) is also linear in the score function, we can use the same strategy to modify the training objective for fx:

##### x⋆ ← x + ζ · fx(Ft,xt,t) − fx(F∅t ,xt,t) .

When t ∈ (s,1] (s = 0.75), we further slightly modify the target score function to

∇xt

log pt(xt|c) + ζ (∇xt

log pt,θ(xt|c) − ∇xt

log pt(xt)), ζ = 0.5 which corresponds to the following training objective:

- 1

- 2

- 1

- 2

(fx(Ft,xt,t) − x) ,z⋆ ← z +

(fz(Ft,xt,t) − z) .

x⋆ ← x +

- D.1.7 Unified Sampling Process Deterministic sampling. When the stochastic ratio ρ = 0, let’s analyze a apecial case where the coefficients satisfying αˆ(t) = dαd(tt),γˆ(t) = dγd(tt). Let ∆t = ti+1 − ti, for the core updating rule we have:

x′ = α(ti+1) · zˆ + γ(ti+1) · xˆ

= (α(ti) + α′(ti)∆t + o(∆t)) · zˆ + (γ(ti) + γ′(ti)∆t + o(∆t)) · xˆ

= (α(ti)zˆ + γ(ti)xˆ) + (ˆα(ti)zˆ + γˆ(ti)xˆ) · ∆t + o(∆t)

= (α(ti)fz(F,x˜,ti) + γ(ti)fx(F,x˜,ti)) + (ˆα(ti)fz(F,x˜,ti) + γˆ(ti)fx(F,x˜,ti)) · ∆t + o(∆t)

= (α(ti)

γˆ(ti) · x˜ − γ(ti) · F(x˜,ti) α(ti) · γˆ(ti) − αˆ(ti) · γ(ti)

+ γ(ti)

α(ti) · F(x˜,ti) − αˆ(ti) · xt α(ti) · γˆ(ti) − αˆ(ti) · γ(ti)

)

+ (ˆα(ti)

γˆ(ti) · x˜ − γ(ti) · F(x˜,ti) α(ti) · γˆ(ti) − αˆ(ti) · γ(ti)

+ γˆ(ti)

α(ti) · F(x˜,ti) − αˆ(ti) · xt α(ti) · γˆ(ti) − αˆ(ti) · γ(ti)

) · ∆t + o(∆t)

= x˜ + F(x˜,ti) · ∆t + o(∆t)

In this case F(·,·) tries to predict the velocity field of the flow model, and we can see that the term x˜ + F(x˜,ti) · ∆t corresponds to the sampling rule of the Euler ODE solver.

Stochastic sampling. As for case when the stochastic ratio ρ ̸= 0, follow the Euler-Maruyama numerical methods of SDE, the noise injected should be a Gaussian with zero mean and variance proportional to ∆t, so when the updating rule is x′ = α(ti+1) · (√1 − ρ · zˆ + √ρ · z) + γ(ti+1) · xˆ, the coefficient of z should satisfy

α(ti+1)√ρ ∝

√

∆t, ρ ∝

∆t α2(ti+1) In practice, we set

ρ =

2∆t · α(ti) α2(ti+1)

.

which corresponds to g(t) = 2α(t) for the SDE dx = f(x,t)dt + g(t)dw.

- D.1.8 Extrapolating Estimation

|Theorem 5 (Local Truncation error of the extrapolated update) . Let {x˜i} be the sequence defined by the extrapolated update<br><br>x˜i+1 = x˜i + h vi + κ(vi − vi−1) + h2 ϵi, h = ti+1 − ti,<br><br>where vi = v(x˜i,ti) and ϵi = O(1). Denote by x(ti+1) the exact solution of x˙ = v(x,t) at time ti+1. Then the local truncation error satisfies<br><br>x(ti+1) − x˜i+1 = h2 12 − κ v′(x˜i,ti) − ϵi + O(h3),<br><br>where v′(x˜i,ti) denotes the total derivative of v along the trajectory. In particular, choosing κ = 12 cancels the O(h2) term (up to ϵi), yielding a second-order method.<br><br>|
|---|

Proof. 1. By Taylor’s theorem in time,

##### vi−1 = v(x˜i−1,ti−1) = vi − hv′(x˜i,ti) + O(h2).

- 2. Substitute into the update rule: x˜i+1 = x˜i + h vi + κ(vi − vi−1) + h2 ϵi

= x˜i + h vi + κ vi − (vi − hv′ + O(h2)) + h2 ϵi

= x˜i + hvi + κh2 v′(x˜i,ti) + h2 ϵi + O(h3).

- 0

- 1

- 2

- 3

- 4

- 5

3.5

1 =2, 2 =8 1 =5, 2 =5 1 =8, 2 =2

1 =0.5, 2 =0.5

1 =1, 2 =1 1 =5, 2 =5 1 =20, 2 =20

Prob.DensityFunc.

Prob.DensityFunc.

Prob.DensityFunc.

1 =0.5, 2 =2 1 =2, 2 =0.5

3.0

20

2.5

15

2.0

1.5

10

1.0

5

0.5

0.0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Time t

Time t

Time t

(a) Skewed and symmetric.

(b) Increasingly concentrated.

(c) J- and U-shaped.

Figure 8: Probability density functions of the Beta distribution over the domain t ∈ [0, 1] for various shape-parameter θ1, θ2.

- 3. The exact solution expands as

x(ti+1) = x(ti) + hv(x(ti),ti) + h

2

2 v′(x(ti),ti) + O(h3). Replacing x(ti) by x˜i in the leading terms gives

x(ti+1) = x˜i + hvi + h

2

2 v′(x˜i,ti) + O(h3).

- 4. Subtracting yields the local truncation error:

2

2 v′ + O(h3) − x ˜i + hvi + κh2v′ + h2ϵi + O(h3)

x(ti+1) − x˜i+1 = x ˜i + hvi + h

= h2 2 1 − κ v′(x˜i,ti) − ϵi + O(h3). This completes the proof.

|Remark 7 (Error reduction via the extrapolation ratio κ) . From the local truncation error estimate<br><br>x(ti+1) − x˜i+1 = h2 2 1 − κ v′(x˜i,ti) − ϵi + O(h3), define<br><br>E(κ) = (12 − κ)v′(x˜i,ti) − ϵi, E(0) = 12 v′(x˜i,ti) − ϵi. Note that<br><br>min<br><br>κ∈[0,1]<br><br>∥E(κ)∥ ≤ ∥E(0)∥.<br><br>By selecting an appropriate κ value, the O(h2) coefficient—and thus the leading part of the local truncation error—is is smaller (or at least not larger) in norm than in the case κ = 0.|
|---|

D.2 Other Techniques

- D.2.1 Beta Transformation

We utilize three representative cases to illustrate how the Beta transformation fBeta(t;θ1,θ2) generalizes time warping mechanisms for t ∈ [0,1].

Standard logit-normal time transformation [49, 8]. For t ∼ U(0,1), the logit-normal transformation flognorm(t;0,1) = 1+exp(−1Φ−1(t)) generates a symmetric density profile peaked at t = 0.5, consistent with the central maximum of the logistic-normal distribution. Analogously, the Beta transformation fBeta(t;θ1,θ2) (with θ1,θ2 > 1) produces a density peak at t = θ

1−1

θ1+θ2−2. When θ1 = θ2 > 1, this reduces to t = 0.5, mirroring the logit-normal case. Both transformations concentrate sampling density around critical time regions, enabling importance sampling for accelerated

training. Notably, this effect can be equivalently achieved by directly sampling t ∼ Beta(θ1,θ2).

Uniform time distribution [49, 52, 30, 25]. The uniform limit case emerges when θ1 = θ2 = 1, reducing fBeta(t;1,1) to an identity transformation. This corresponds to a flat density p(t) = 1, reflecting no temporal preference—a baseline configuration widely adopted in diffusion and flowbased models.

Approximately symmetrical time distribution [41, 39, 18, 20]. For near-symmetric configurations where θ1 ≈ θ2 > 1, the Beta transformation induces quasi-symmetrical densities with tunable central sharpness. For instance, setting θ1 = θ2 = 2 yields a parabolic density peaking at t = 0.5, while θ1 = θ2 → 1+ asymptotically approaches uniformity. This flexibility allows practitioners to interpolate between uniform sampling and strongly peaked distributions, adapting to varying requirements for temporal resolution in training. Such approximate symmetry is particularly useful in consistency models where balanced gradient propagation across time steps is critical.

Furthermore, Fig. 8 further demonstrates the flexibility of the beta distribution.

#### D.2.2 Kumaraswamy Transformation

|Lemma 16 (Piecewise monotone error) . Suppose f,g are continuous and nondecreasing on [0,1], and agree at<br><br>0 = x0 < x1 < ··· < xn = 1, i.e. f(xj) = g(xj) for j = 0,...,n. Let ∆j = g(xj) − g(xj−1). Then for every t ∈ [xj−1,xj],<br><br>|f(t) − g(t)| ≤ ∆j . In particular, if each ∆j ≤ 14, then ∥f − g∥L∞ ≤ 14.<br><br>|
|---|

Proof. On [xj−1,xj] monotonicity gives

f(t) − g(t) ≤ f(xj) − g(xj−1) = g(xj) − g(xj−1) = ∆j, and similarly g(t) − f(t) ≤ ∆j.

|Theorem 6 (L2 approximation bound of monotonic functions by generalized Kumaraswamy transformation) . Let G = g ∈ C([0,1]) : g nondecreasing, g(0) = 0, g(1) = 1 , and define<br><br>for a,b,c > 0, fa,b,c(t) = 1 − (1 − ta)b c, t ∈ [0,1]. Then<br><br>sup<br><br>g∈G<br><br>inf<br><br>a,b,c>0<br><br>1<br><br>0<br><br>fa,b,c(t) − g(t) 2 dt ≤<br><br>1 16<br><br>.|
|---|

Proof. Let g ∈ G. By continuity and the Intermediate-Value Theorem there exist

0 < t1 < t0 < t2 < 1, g(t1) = 41, g(t0) = 12, g(t2) = 34. We will choose (a,b,c) > 0 so that

fa,b,c(tj) = g(tj) (j = 1,0,2), and then apply the piecewise monotone Lem. 16 on the partition

0, t1, t0, t2, 1

to conclude ∥fa,b,c − g∥L∞ ≤ 41 and hence ∥fa,b,c − g∥2L2 ≤ 161 .

#### Existence via the implicit function theorem. Define

 

 .

- fa,b,c(t1) − 41 fa,b,c(t0) − 21

- fa,b,c(t2) − 43

F : R3>0 −→ R3, F(a,b,c) =

Then F is C1, and at the “base point” (a,b,c) = (1,1,1) with (t1,t0,t2) = (14, 12, 34) we have f1,1,1(t) = t so F(1,1,1) = 0, and the Jacobian ∂F/∂(a,b,c) there is invertible. By the Implicit

Function Theorem, for each fixed (t1,t0,t2) near (41, 12, 34) there is a unique local solution (a,b,c).

Global non-degeneracy of the Jacobian. In order to continue this local solution to all triples 0 < t1 < t0 < t2 < 1, we show det ∂(a,b,c)F(a,b,c) never vanishes. Set

u(t) = 1 − (1 − ta)b, uj = u(tj) ∈ (0,1), fj = ucj. Then

∂afj = cujc−1 ∂auj, ∂bfj = cujc−1 ∂buj, ∂cfj = ucj lnuj. Hence

 

 .

cuc1−1u1,a cuc1−1u1,b uc1 lnu1 cuc0−1u0,a cuc0−1u0,b uc0 lnu0 cuc2−1u2,a cuc2−1u2,b uc2 lnu2

detJ = det

Factor c from the first two columns and ucj−1 from each row:

detJ = c2 (u1u0u2)c−1 det

- u1,a u1,b u1 lnu1 u0,a u0,b u0 lnu0
- u2,a u2,b u2 lnu2

.

Now

uj,b = −(1 − taj)b ln(1 − taj) = −(1 − uj)ln(1 − taj),

taj lntj 1 − taj

uj,a = b(1 − taj)b−1taj lntj = −b(1 − uj)

.

A direct—but straightforward—expansion shows

- u1,a u1,b u1 lnu1 u0,a u0,b u0 lnu0
- u2,a u2,b u2 lnu2

u1u0u2 (1 − u1)(1 − u0)(1 − u2)

= c−2b

(u0 − u1)(u2 − u1)(u2 − u0).

det

Therefore

(u0 − u1)(u2 − u1)(u2 − u0) (1 − u1)(1 − u0)(1 − u2)

detJ(a,b,c) = b(u1u0u2)c

> 0,

since 0 < u1 < u0 < u2 < 1 and a,b,c > 0. Hence the Jacobian is everywhere non-zero, and the local solution by the Implicit Function Theorem extends along any path in the connected domain

{0 < t1 < t0 < t2 < 1}. We obtain a unique (a,b,c) > 0 solving

fa,b,c(tj) = g(tj), j = 1,0,2, for every choice 0 < t1 < t0 < t2 < 1.

Completing the error estimate. By construction fa,b,c(0) = 0, fa,b,c(1) = 1, and fa,b,c(tj) =

- g(tj) for j = 1,0,2. On the partition 0, t1, t0, t2, 1

the increments of g are each 1/4. The piecewise monotone error Lem. 16 yields ∥fa,b,c − g∥L∞ ≤ 41, hence

1

1 16

fa,b,c(t) − g(t) 2 dt ≤ ∥f − g∥2L∞ ≤

. Since g was arbitrary in G, we conclude

0

1

1 16

fa,b,c(t) − g(t) 2 dt ≤

sup

inf

.

a,b,c>0

g∈G

0

This completes the proof.

| |
|---|

Setting and notation. Fix a positive real number s > 0 and consider the shift function

st 1 + (s − 1)t

, t ∈ [0,1]. For a,b,c > 0, define the Kumaraswamy transform as

fshift(t;s) =

b c

fKuma(t;a,b,c) = 1 − 1 − ta

, t ∈ [0,1]. Notice that when a = b = c = 1 one obtains

fKuma(t;1,1,1) = 1 − 1 − t1 1 = t, so that the identity function appears as a special case. We work in the Hilbert space L2([0,1]) with the inner product

1

⟨f,g⟩ =

f(t)g(t)dt. Accordingly, we introduce the error functional

0

2 2

2 2

and Jid := id − fshift(·;s)

J(a,b,c) := fKuma(·;a,b,c) − fshift(·;s)

. It is known that for s ̸= 1 one has

J(a,b,c) < Jid. The goal is to quantify this improvement by optimally adjusting all three parameters (a,b,c). Quadratic approximation around the identity. Since the interesting behavior occurs near the identity (a,b,c) = (1,1,1), we reparameterize as

inf

a,b,c

 

  :=

 

 , with ∥θ∥ ≪ 1.

- a − 1
- b − 1
- c − 1

α β γ

θ :=

Thus, we study the function

fKuma(t;1 + α,1 + β,1 + γ) in a small neighborhood of (1,1,1). Writing

c

F(a,b,c;t) := fKuma(t;a,b,c) = 1 − (1 − ta)b

, a second–order Taylor expansion around (a,b,c) = (1,1,1) gives

fKuma(t;1 + α,1 + β,1 + γ) = t +

3

θi gi(t) +

i=1

3

1 2

θiθj hij(t) + O(∥θ∥3), (8)

i,j=1

where

∂2 ∂θi∂θj

∂ ∂θi

and hij(t) =

fKuma(t;1 + θ)

fKuma(t;1 + θ)

. A short calculation yields:

gi(t) =

θ=0

θ=0

- (a) With respect to a (noting that for b = c = 1 one has fKuma(t;a,1,1) = ta):

g1(t) =

∂fKuma ∂a

(t;1,1,1) =

d da

ta

a=1

= tlnt.

- (b) With respect to b (since for a = 1, c = 1 we have fKuma(t;1,b,1) = 1 − (1 − t)b):

∂fKuma ∂b

(t;1,1,1) = −(1 − t) ln(1 − t).

g2(t) =

- (c) With respect to c (noting that for a = b = 1 we have fKuma(t;1,1,c) = tc):

∂fKuma ∂c

g3(t) =

(t;1,1,1) = tlnt. Thus, we observe that

g1(t) = g3(t), which indicates an inherent redundancy in the three-parameter model. In consequence, the Gram matrix (defined below) will be of rank at most two.

Next, define the difference between the identity and the shift functions:

t(1 − t) 1 + (s − 1)t

st 1 + (s − 1)t

g(t) := id(t) − fshift(t;s) = t −

= (1 − s)

. Then, Jid = ⟨g,g⟩. Also, introduce the first-order moments and the Gram matrix:

vi := ⟨g,gi⟩, Gij := ⟨gi,gj⟩, i,j = 1,2,3. Inserting the expansion (8) into the error functional gives

3

J(1 + θ) = fKuma(·;1 + θ) − fshift(·;s) 22 = Jid − 2

θi vi +

i=1

Thus, the quadratic approximation (or model) of the error is

3

θiθj Gij + O(∥θ∥3).

i,j=1

J(θ) := Jid − 2θ⊤v + θ⊤Gθ.

Since the Gram matrix G is positive semidefinite (and has a nontrivial null-space due to g1 = g3), the minimizer is determined only up to the null-space. To select the unique (minimum–norm) minimizer, we choose

θ⋆ = G†v,

where G† denotes the Moore-Penrose pseudoinverse. The quadratic model is then minimized at

Jmin = Jid − v⊤G†v.

A scaling argument now shows that for any sufficiently small ε > 0 one has

J(1 + εθ⋆) ≤ J(εθ⋆) = Jid − ε2 v⊤G†v < Jid, so that the full nonlinear functional is improved by following the direction of θ⋆. For convenience we introduce the explicit improvement factor

v⊤G†v Jid(s) ∈ (0,1), s ̸= 1, (9)

ρ3(s) :=

so that our main bound can be written succinctly as

J(a,b,c) ≤ 1 − ρ3(s) Jid(s). (s > 0, s ̸= 1) (10)

min

a,b,c>0

#### Computation of the Gram matrix G. We now compute the inner products

Gij = ⟨gi,gj⟩, i,j = 1,2,3.

Since the functions g1 and g3 are identical, only two independent functions appear in the system. A standard fact from Beta-function calculus is that

1

2 (n + 1)3

tn ln2 tdt =

, n > −1. Thus, one has

0

- ⟨g1,g1⟩ =

1

0

t2 ln2 tdt =

2 33

=

2 27

,

- ⟨g2,g2⟩ =

1

2 27

(1 − t)2 ln2(1 − t)dt =

,

0

since the change of variable u = 1 − t yields the same result.

1

3π2 − 37 108

⟨g1,g2⟩ = −

t(1 − t)lnt ln(1 − t)dt =

. It is now convenient to express the Gram matrix with an overall factor:

0

  ,r =

  

1 r 1 r 1 r 1 r 1

3π2 − 37 8

2 27

.

G =

Since g1 = g3, it is clear that the columns (and rows) corresponding to parameters a and c are identical, so that rank(G) = 2. One can compute the Moore-Penrose pseudoinverse G† by eliminating one of the redundant rows/columns, inverting the resulting 2 × 2 block, and then re-embedding into R3×3. One obtains

  .

  

1 −2r 1 −2r 4 −2r 1 −2r 1

27 8(1 − r2)

G† =

#### Computation of the first-order moments vi. Recall that

st 1 + (s − 1)t

g(t) = id(t) − fshift(t;s) = t −

. This expression can be rewritten as

1 1 + (s − 1)t

g(t) = (1 − s)t(1 − t)Ds(t), with Ds(t) :=

. Then, the first–order moments read

1

v1 = v3 = (1 − s)

t(1 − t)Ds(t) tlntdt,

0

1

v2 = −(1 − s)

t(1 − t)Ds(t) (1 − t)ln(1 − t)dt.

0

These integrals can be expressed in closed form (involving logarithms and powers of (s − 1)); in the case s ̸= 1 at least one of the vi is nonzero so that ρ3(s) > 0.

A universal numerical improvement. Since projecting onto the three-dimensional subspace spanned by {g1,g2,g3} is at least as effective as projecting onto any one axis, we immediately deduce that

ρ3(s) ≥ ρ1(s),

where the one-parameter improvement factor is defined by

v1(s)2 ⟨g1,g1⟩Jid(s)

.

ρ1(s) :=

By an elementary (albeit slightly tedious) estimate — for example, using the bounds 12 ≤ Ds(t) ≤ 2 valid for |s − 1| ≤ 1 — one can show that

49 1536

ρ1(s) ≥

. Hence, one deduces that

49 1536 ≈ 0.0319, for |s − 1| ≤ 1.

ρ3(s) ≥

In particular, for s ∈ [0.5,2] \ {1} the optimal three-parameter Kumaraswamy transform reduces the squared L2 error by at least 3.19% compared with the identity mapping. Analogous bounds can be obtained on any compact subset of (0,∞) \ {1}.

Interpretation of the bound. Inequality (10) strengthens the known qualitative result (namely, that the three-parameter model can outperform the identity mapping) in two important respects:

- (a) Quantitative improvement: The explicit factor ρ3(s) is computable via one-dimensional integrals, providing a concrete measure of the error reduction.
- (b) Utilization of all three parameters: Even though the redundancy (i.e. g1 = g3) implies that the Gram matrix is singular, the full three-parameter model still offers strict improvement; indeed,

one has ρ3(s) ≥ ρ1(s) > 0 for s ̸= 1. (Equality would require, hypothetically, that v2(s) = 0, which does not occur in practice.)

Summary. For every shift parameter s > 0 with s ̸= 1 there exist parameters (a,b,c) (in a neighborhood of (1,1,1)) such that

2 2

2 2

≤ 1 − ρ3(s) id − fshift(·;s)

fKuma(·;a,b,c) − fshift(·;s)

, with the improvement factor ρ3(s) defined in (9) and satisfying

ρ3(s) ≥ 0.0319 on s ∈ [0.5,2] \ {1}.

Thus, the full three-parameter Kumaraswamy transform not only beats the identity mapping but does so by a quantifiable margin.

#### D.2.3 Derivative Estimation

|Proposition 1 (Error estimates for forward and central difference quotients) . Let f ∈ C3(I) where I ⊂ R is an open interval, and let t ∈ I. For 0 < ε small enough that [t − ε,t + ε] ⊂ I, define the forward and central difference quotients<br><br>D+f(t) =<br><br>f(t + ε) − f(t) ε<br><br>, D0f(t) =<br><br>f(t + ε) − f(t − ε) 2ε<br><br>. Then<br><br>D+f(t) = f′(t) +<br><br>ε 2<br><br>f′′(t) +<br><br>ε2 6<br><br>f(3)(t + θ1ε), for some 0 < θ1 < 1,<br><br>D0f(t) = f′(t) +<br><br>ε2 12<br><br>f(3)(t + θ2ε) + f(3)(t − θ3ε) , for some 0 < θ2,θ3 < 1. In particular,<br><br>D+f(t) − f′(t) = O(ε), D0f(t) − f′(t) = O(ε2),<br><br>so for sufficiently small ε, the forward-difference error exceeds the central-difference error.|
|---|

Proof. By Taylor’s theorem with Lagrange remainder, for some 0 < θ1 < 1,

f(t + ε) = f(t) + f′(t)ε + 21f′′(t)ε2 + 16f(3)(t + θ1ε)ε3. Dividing by ε gives the formula for D+f(t). Hence

D+f(t) − f′(t) =

1 2

f′′(t)ε +

1 6

f(3)(t + θ1ε)ε2 = O(ε).

Similarly, applying Taylor’s theorem at t + ε and t − ε,

f(t + ε) = f(t) + f′(t)ε + 21f′′(t)ε2 + 16f(3)(t + θ2ε)ε3, f(t − ε) = f(t) − f′(t)ε + 12f′′(t)ε2 − 16f(3)(t − θ3ε)ε3,

for some 0 < θ2,θ3 < 1. Subtracting and dividing by 2ε yields the formula for D0f(t) and

ε2 12

D0f(t) − f′(t) =

f(3)(t + θ2ε) + f(3)(t − θ3ε) = O(ε2). This completes the proof.

| |
|---|

|Proposition 2 . Let f : R → R be differentiable, let t ∈ R and ε > 0. In BF16 arithmetic (1-bit sign, 8-bit exponent, 7-bit significand) with unit roundoff η = 2−7, define<br><br>f± = f(t ± ε), ∆ = f+ − f−,<br><br>E1 =<br><br>fl(f+) − fl(f−) 2ε<br><br>, E2 = fl<br><br>f+ 2ε − fl<br><br>f− 2ε<br><br>. Suppose in addition that<br><br>(1) |∆| < 2−126, so that ∆ (and any nearby perturbation) lies in the BF16 subnormal range;<br>(2) writing fl(f±) = f±(1 + δ±) with |δ±| ≤ η, one has f+δ+ − f−δ− < 2−126, so f˜+ − f˜− remains subnormal;<br>(3) f±/(2ε) ≥ 2−126, so each product f±/(2ε) lies in the normalized range;<br>(4) |f+| + |f−| = O(|∆|), so that any rounding in the two multiplications is not amplified by a large subtraction.<br><br><br>Then the “subtract-then-scale” formula E1 may incur a relative error of order O(1), whereas the “scale-then-subtract” formula E2 retains a relative error of order O(η).|
|---|

Proof. We use two BF16 rounding models: (i) if x ∈ [2−126,2128) then fl(x) = x(1 + δ), |δ| ≤ η; (ii) for any x (including subnormals), fl(x) − x ≤ 21 ulp(x), where ulpsub = 2−133 for subnormals. Set f˜± = fl(f±) = f±(1 + δ±), |δ±| ≤ η.

- Error in E1. By (1) and (2), f˜+ − f˜− = ∆ + (f+δ+ − f−δ−) lies in the subnormal range. Hence d = fl(f˜+ − f˜−) = (f˜+ − f˜−) + ed, |ed| ≤ 12 ulpsub = 2−134.

Thus

d = ∆ + (f+δ+ − f−δ−) + ed, |ed|/|∆| = O(2−134/|∆|)gη. Dividing by 2ε and rounding gives

E1 = fl d/(2ε) =

d 2ε

(1 + δq), |δq| ≤ η, so the relative error in E1 can be O(1).

- Error in E2. By (3), each f±/(2ε) is normalized, so

f± 2ε

f± 2ε

(1 + δ±′ ), |δ±′ | ≤ η. Subtracting and rounding (still normalized) gives

g± = fl

=

E2 = fl(g+ − g−) = (g+ − g−)(1 + δd′ ), |δd′ | ≤ η. Since

f+δ+′ − f−δ−′ 2ε

∆ 2ε

g+ − g− =

, we obtain

+

f+δ+′ − f−δ−′ 2ε

∆ 2ε

(1 + δd′ ). The second term has magnitude ≤ η |f

(1 + δd′ ) +

E2 =

+|+|f−|

2ε (1 + η), and by (4) its relative size to ∆/(2ε) is O η |f

+|+|f−|

|∆| = O(η).

Hence E1 may suffer O(1) relative error, while E2 attains O(η) relative accuracy under (1)–(4).

| |
|---|

- D.2.4 Calcluation of Transport

Transport transformation from EDM to UCGM. Take the formula (8) from EDM [18], one can deduce:

2

1 cout(σ)

Eσ,x,n λ(σ)cout(σ)2 Fθ(cin(σ) · (x + n); cnoise(σ)) −

(x − cskip(σ) · (x + n))

2

2

σdata2 σ2 + σdata2 · (x + σz))

σdata2 + σ2 σ · σdata

1 4

1 σ2 + σdata2 · (x + σz);

= Eσ,x,z Fθ(

ln(σ)) −

(x −

2

2

σ2 σ2 + σdata2 · x −

σdata2 σ2 + σdata2 · σz)

σdata2 + σ2 σ · σdata

1 σ2 + σdata2 · (x + σz);

1 4

= Eσ,x,z Fθ(

ln(σ)) −

(

2

2

σσdata−1 σdata2 + σ2 · x −

1 σ2 + σdata2 · (x + σz);

1 4

σdata σ2 + σdata2 · z)

= Eσ,x,z Fθ(

ln(σ)) − (

2

2

σσdata−1 σdata2 + σ2 · x)

ln(σ)) − ( −σdata σ2 + σdata2 · z +

σ σ2 + σdata2 · z +

1 σ2 + σdata2 · x;

1 4

= Eσ,x,z Fθ(

2

   Fθ(

  

2

· x) − ( −0.5 σ2 + 14

σ σ2 + 41

1 σ2 + 14

2σ σ2 + 14

= Eσ,x,z

· z +

· z +

· x)

2

where σdata = 21, n = σ · z.

