## Differentiable Solver Search for Fast Diffusion Sampling

# arXiv:2505.21114v1[cs.CV]27May2025

#### Shuai Wang1 Zexian Li2 Qipeng Zhang2 Tianhui Song1 Xubin Li2 Tiezheng Ge2 Bo Zheng2 Limin Wang13

[Figure 1]

Figure 1: Visualization of searched Solver Parameters of DDPM/VP and Rectified Flow. We limited the order of solver coefficients of the last two steps for 5/6 NFE. The left images show the absolute value of searched coefficients {cji}. The right image shows the searched timesteps of different NFE and fitted curves.

### Abstract

score of 2.33 with only 10 steps. Notably, our searched solver significantly outperforms traditional solvers(even some distillation methods). Moreover, our searched solver demonstrates generality across various model architectures, resolutions, and model sizes.

Diffusion models have demonstrated remarkable generation quality, but at the cost of numerous function evaluations. Advanced ODE-based solvers have recently been developed to mitigate the substantial computational demands of reverse-diffusion solving under limited sampling steps. However, these solvers, heavily inspired by Adams-like multistep methods, rely solely on t-related Lagrange interpolation. We show that t-related Lagrange interpolation is suboptimal for diffusion models and reveal a compact search space comprised of time steps and solver coefficients. Building on our analysis, we propose a novel differentiable solver search algorithm to identify more optimal solver. Equipped with the searched solver, rectified-flow models, e.g., SiT-XL/2 and FlowDCN-XL/2, achieve FID scores of 2.40 and 2.35, respectively, on ImageNet-256 × 256 with only 10 steps. Meanwhile, DDPM model, DiT-XL/2, reaches a FID

### 1. Introduction

Image generation is a fundamental task in computer vision research, which aims at capturing the inherent data distribution of original image datasets and generating high-quality synthetic images through distribution sampling. Diffusion models (Ho et al., 2020; Song et al., 2020b; Karras et al., 2022; Liu et al., 2022; Lipman et al., 2022; Wang et al., 2025) have recently emerged as highly promising solutions to learn the underlying data distribution in image generation, outperforming GAN-based models (Brock et al., 2018; Sauer et al., 2022) and Auto-Regressive models (Chang et al., 2022) by a significant margin.

However, diffusion models necessitate numerous denoising steps during inference, which incur a substantial computational cost, thereby limiting the widespread deployment of pre-trained diffusion models. To achieve fast diffusion sampling, the existing studies have explored two distinct approaches. Training-based techniques by distilling the fast ODE trajectory into the model parameters, thereby circumventing redundant refinement steps. In addition, solver-

1State Key Lab of Novel Software Technology, Nanjing University, Nanjing, China. 2Taobao & Tmall Group of Alibaba, Hangzhou, China. 3Shanghai AI Lab, Shanghai, China.. Correspondence to: Limin Wang <lmwang@nju.edu.cn>.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

based methods (Lu et al., 2023; Zhang & Chen, 2023; Song et al., 2020a) tackle the fast sampling problem by designing high-order numerical ODE solvers.

For training-based acceleration, (Salimans & Ho, 2022) aligns the single-step student denoiser with the multi-step teacher output, thereby reducing inference burdens. The consistency model concept, introduced by (Song et al., 2023), directly teaches the model to produce consistent predictions at any arbitrary timesteps. Building upon (Song et al., 2023), subsequent works (Zheng et al., 2024; Kim et al., 2023; Wang et al., 2024a; Song et al., 2025) propose improved techniques to mitigate discreet errors in LCM training. Furthermore, (Lin et al., 2024; Kang et al., 2024; Yin et al., 2024; Zhou et al., 2024; Wang et al., 2023) leverage adversarial training and distribution matching to enhance the quality of generated samples. To improve the training efficiency of distribution matching. However, training-based methods introduce changes to the model parameters, resulting in an inability to fully exploit the pre-training performance.

Solver-based methods rely heavily on the ODE formulation in the reverse-diffusion dynamics and hand-crafted multi-step solvers. (Lu et al., 2023; 2022) and (Zhang & Chen, 2023) point out the semi-linear structure of the diffusion ODE and propose an exponential integrator to tackle faster sampling in diffusion models. (Zhao et al., 2023) further enhances the sampling quality by borrowing the predictor-corrector structure. Thanks to the multistep-based ODE solver methods, high-quality samples can be generated within as few as 10 steps. To further improve efficiency, (Gao et al., 2023) tracks the backward error and determines the adaptive step. Moreover, (Karras et al., 2022; Lu et al., 2022) propose a handcrafted timesteps scheduler to sample respaced timesteps. (Xue et al., 2024) argues that timesteps sampled in (Karras et al., 2022; Lu et al., 2022) are suboptimal, thus proposing an online optimization algorithm to find the optimal sampling timesteps for generation. Apart from timesteps optimization, (Shaul et al., 2023) learns a specific path transition to improve the sampling efficiency.

In contrast to training-based acceleration methods, solverbased approaches do not necessitate parameter adjustments and preserve the optimal performance of the pre-trained model. Moreover, solvers can be seamlessly applied to any arbitrary diffusion model trained with a similar noise scheduler, offering a high degree of flexibility and adaptability. This motivates us to investigate the generative capabilities of pre-trained diffusion models within limited steps from a diffusion solver perspective.

Current state-of-the-art diffusion solvers (Lu et al., 2023; Zhao et al., 2023) adopt Adams-like multi-step methods that use the Lagrange interpolation function to minimize integral errors. We argue that an optimal solver should be tailored to specific pre-trained denoising functions and

their corresponding noise schedulers. In this paper, we explore solver-based methods for fast diffusion sampling by improving diffusion solvers using data-driven approaches without destroying the pre-training internality in diffusion models. Inspired by (Xue et al., 2024), we investigate the sources of error in the diffusion ODE and discover that the interpolation function form is inconsequential and can be reduced to coefficients. Furthermore, we define a compact search space related to the timesteps and solver coefficients. Therefore, we propose a differentiable solver search method to identify the optimal parameters in the compact search space.

Based on our novel differentiable solver search algorithm, we investigate the upper bound performance of pre-trained diffusion models under limited steps. Our searched solver significantly improves the performance of pre-trained diffusion models, and outperforms traditional solvers with a large gap. On ImageNet-256 × 256, armed with our solver, rectified-flow models of SiT-XL/2 and FlowDCNXL/2 achieve 2.40 and 2.35 FID respectively under 10 steps, while DDPM model of DiT-XL/2 achieves 2.33 FID. Surprisingly, our findings reveal that when equipped with an optimized high-order solver, the DDPM can achieve comparable or even surpass the performance of rectified flow models under similar NFE constraints.

To summarize, our contributions are

- • We reveal that the interpolation function choice is not important and can be reduced to coefficients through the pre-integral technique. We demonstrate that the upper bound of discretization error in reverse-diffusion ODE is related to both timesteps and solver coefficients and define a compact solver search space.
- • Based on our analysis, we propose a novel differentiable solver search algorithm to find the optimal solver parameter for given diffusion models.
- • For DDPM/VP time scheduling, armed with our searched solver, DiT-XL/2 achieves 2.33 FID under 10 steps, beating DPMSolver++/UniPC by a significant margin.
- • For Rectified-flow models, armed with our searched solver, SiT-XL/2 and FlowDCN-XL/2 achieve 2.40 and 2.35 FID respectively under 10 steps on ImageNet256 × 256.
- • For Text-to-Image diffusion models like FLUX, SD3, PixArt-Σ, our solver searched on ImageNet-256 × 256 consistently yields better images compared to traditional solvers with the same CFG scale.

### 2. Related Work

Diffusion Model gradually adds x0 with Gaussian noise ϵ to perturb the corresponding known data distribution p(x0) into a simple Gaussian distribution. The discrete perturbation function of each t satisfies N(xt|αtx0,σt2I), where αt,σt > 0. It can also be written as Equation (1).

##### xt = αtxreal + σtϵ (1)

Moreover, as shown in Equation (2), Equation (1) has a forward continuous-SDE description, where f(t) = d logα

t

dt and g(t) = dσ

2 t

dt − d logα

dt σt2. (Anderson, 1982) establishes a pivotal theorem that the forward SDE has an equivalent reverse-time diffusion process as in Equation (3), so the generating process is equivalent to solving the diffusion SDE. Typically, diffusion models employ neural networks and distinct prediction parametrization to estimate the score function ∇logx px

t

(xt) along the sampling trajectory (Song et al., 2020b; Karras et al., 2022; Ho et al., 2020).

t

dxt = f(t)xtdt + g(t)dw (2) dxt = [f(t)xt − g(t)2∇x log p(xt)]dt + g(t)dw (3)

(Song et al., 2020b) also shows that there exists a corresponding deterministic process Equation (4) whose trajectories share the same marginal probability densities of Equation (3).

dxt = [f(t)xt −

- 1

- 2

g(t)2∇xt

log p(xt)]dt (4)

Rectified Flow Model simplifies diffusion model under the framework of Equation (2) and Equation (3). Different from (Ho et al., 2020) introduces non-linear transition scheduling, the rectified-flow model adopts linear function to transform data to standard Gaussian noise.

2023) discovered the semi-linear structure in DDPM/VP reverse ODEs. Furthermore, (Zhao et al., 2023) enhanced the sampling quality by borrowing the predictor-corrector structure. Thanks to the multi-step ODE solvers, high-quality samples can be generated within as few as 10 steps. To further improve efficiency, (Gao et al., 2023) tracks the backward error and determines the adaptive step. Moreover, (Karras et al., 2022; Lu et al., 2022) proposed a handcrafted timestep scheduler to sample respaced timesteps. However, (Xue et al., 2024; Sabour et al., 2024; Chen et al., 2024a) argued that the handcrafted timesteps are suboptimal, and thus proposed an online optimization algorithm to find the optimal sampling timestep for generation. Apart from timestep optimization, (Shaul et al., 2023) learned a specific path transition to improve the sampling efficiency.

### 3. Problem Definition

As rectified-flow constitutes a simple yet elegant formulation within the diffusion family, we choose rectified-flow as the primary subject of discussion in this paper to enhance readability. Importantly, our proposed algorithm is not constrained to rectified-flow models. We explore its applicability to other diffusion models such as DDPM/VP in Section 6.

Recall the continuous integration of reverse-diffusion in Equation (7) with the pre-defined interval {t0,t1,...tN}. Given the pre-trained diffusion models and their corresponding ODE defined in Equation (4), before we tackle the integration of interval [ti,ti+1], we have already obtained the sampled velocity field of previous timestep {(xj,tj,vj = vθ(xj,tj)}ij=0. Here, we directly denote xt

as xi for presentation clarity:

i

xi+1 = xi +

ti+1

vθ(xt,t)dt (7)

ti

xt = txreal + (1 − t)ϵ (5) Instead of estimating the score function ∇xt

log pt(xt), rectified-flow models directly learn a neural network vθ(xt,t) to predict the velocity field vt = dxt = (xreal−ϵ).

1

||vθ(xt,t) − vt||2dt] (6)

L(θ) = E[

0

Solver-based Fast Sampling Method does not necessitate parameter adjustments and preserves the optimal performance of the pre-trained model. It can be seamlessly applied to an arbitrary diffusion model trained with a similar noise scheduler, offering a high degree of flexibility and adaptability. Solvers heavily rely on the reverse diffusion ODE in Equation (4). Current solvers are mainly focused on DDPM/VP noise schedules. (Lu et al., 2022; Zhang & Chen,

As shown in Equation (8), we strive to develop a more optimal solver that minimizes the integral error while enhancing image quality under limited sampling steps (NFE) without requiring any parameter adjustments for the pretrained model.

1

vθ(xt,t)dt)||]. (8)

Φ = arg minE[||Φ(ϵ,vθ) − (ϵ +

0

### 4. Analysis of reverse-diffusion ODE Sampling

Initially, we revisit the multi-step methods commonly used by (Zhao et al., 2023; Zhang & Chen, 2023; Lu et al., 2023) and identify potential limitations. Specifically, we argue that the Lagrange interpolation function used in AdamsBashforth methods is suboptimal for diffusion models. Moreover, we show that the specific form of the interpolation function is inconsequential, as pre-integration and

[Figure 2]

Figure 2: Generated images from Flux.1-dev with Guidance=2.0 and our solver (searched on SiT-XL/2). Euler-Shift3 is the default solver provided by diffusers and Flux community. Our solver(DS-Solver) achieves better visual quality from 5 to 10 steps(NFE).

expectation estimation ultimately reduce it to a set of coefficients. Inspired by (Xue et al., 2024), we prove that timesteps and these coefficients effectively constitute our search space.

#### 4.1. Recap the multi-step methods

As shown in Equation (9), the Euler method employs vi as an estimate of Equation (9) throughout the interval [ti,ti+1]. Higher-order multistep solvers further improve the estimation quality of the integral by incorporating interpolation functions and leveraging previously sampled values.

xi+1 = xi + (ti+1 − ti)vθ(xi,ti). (9)

The most classic multi-step solver Adams–Bashforth method (Bashforth & Adams, 1883)(deemed as Adams for brevity) incorporates the Lagrange polynomial to improve the estimation accuracy within a given interval.

xi+1 ≈ xi +

ti+1

ti

- i
- j=0

xi+1 ≈ xi +

vj

- i
- j=0

i

t − tk tj − tk

)vjdt (10)

(

k=0,k̸=j

i

ti+1

t − tk tj − tk

)dt (11)

(

ti

k=0,k̸=j

As Equation (11) states, t ti+1

( ik=0,k̸=j t−t

tj−tk )dt of the Lagrange polynomial can be pre-integrated into a constant coefficient, resulting in only naive summation being required for ODE solving. Current SoTA multi-step solvers (Lu et al., 2023; Zhao et al., 2023) are heavily inspired by Adams–Bashforth-like multi-step solvers. These

k

i

solvers employ the Lagrange interpolation function or difference formula to estimate the value in the given interval.

However, the Lagrange interpolation function and other similar methods only take t into account while the v(x,t) also needs x as inputs. Using first-order Taylor expansion of x at xi and higher-order expansion of t at ti, we can readily derive the error bound of the estimation.

4.2. Focus on Solver coefficients instead of the interpolation function

In contrast to typical problems of solving ordinary differential equations, when considering reverse-diffusion ODEs along with prerained models, a compact searching space is present. We define a universal interpolation function P, which has no explicit form. P measures the distance of (xt,t) between previous sampled points {(xj,tj)}ij=0 to determine the interpolation weight for {vj}ij=0.

- i
- j=0

ti+1

P(xt,t,xj,tj)vjdt. (12)

xi+1 ≈ xi +

ti

- i
- j=0

ti+1

P(xt,t,xj,tj)dt. (13)

≈ xi +

vj

ti

Assumption 4.1. We assume that the remainder term of the universal interpolation function ij=0 P(xt,t,xj,tj)vj for v(x,t) is bound as O(dxm) + O(dtn), where O(dxm) is the mth-order infinitesimal for dx, O(dtm) is the nth-order infinitesimal for dt.

Equation (13) has a recurrent dependency, as xt also relies

on ij=0 P(xt,t,xj,tj)vjdt. To eliminate the recurrent dependency, shown in Equation (14), we simply use the first

order Taylor expansion of x(t) at xi to replace the original form. Recall that vi is already determined by xi and ti, thus the partial integral of Equation (14) can be formulated as Equation (15). Unlike naive Lagrange interpolation, Cj(xi) is a function of the current xi instead of a constant scalar. Learning a Cj(xi) function will cause the generalization to be lost. This limits the actual usage in diffusion model sampling.

xi+1 ≈ xi +

xi+1 ≈ xi +

- i
- j=0

vj

ti+1

P(xi + vi(t − ti),t,xj,tj)dt

ti

(14)

- i
- j=0

vjCj(xi)(ti+1 − ti) (15)

Theorem 4.2. Given sampling time interval [ti,ti+1] and suppose Cj(xi) = gj(xi) + bji, Adams-like linear multi-step methods have an error expectation of (ti+1 − ti)Ex

i|| ij=0 vjgj(xi)||. replacing Cj(x) with Ex

[Cj(xi)] is the optimal choice and owns an error expectation of (ti+1 − ti)Ex

i

i|| ij=0 vj[gj(xi) − Ex

gj(xi)||. We place the proof in Appendix H.

i

According to Theorem 4.2, we opt to replace Cj(xi) with its expectation Ex

[Cj(xi)], thus we obtain diffusionscheduler-related coefficients while maintaining generalization ability.

i

Finally, given the predefined time intervals, we obtain the optimization target Equation (16), where cji = Ex

[Cj(xi)]. The expectation can be deemed as optimized through massive data and gradient descent.

i

xi+1 ≈ xi +

- i
- j=0

vjcji(ti+1 − ti) (16)

#### 4.3. Optimal search space for a solver

Assumption 4.3. As shown in Equation (17), the pre-trained velocity model vθ is not perfect and the error between vθ and ideal velocity field vˆ is L1-bounded, where η is a constant scalar.

||vˆ − vθ|| ≤ η ≪ ||vˆ|| (17)

Previous discussions assume we have a perfect velocity function. However, the ideal velocity is hard to obtain, we only have pre-trained velocity models. Following Equation (16), we can expand Equation (16) from ti=0 to ti=N to obtain the error bound caused by non-ideal velocity estimation.

Theorem 4.4. The error caused by the non-ideal velocity estimation model can be formulated in the following

equation. We can employ triangle inequalities to obtain the error-bound(L1) of ||xN − xˆN||, the proof can be found in the Appendix I.

N−1

- i
- j=0

||xN − xˆN|| ≤ η

i=0

|cji(ti+1 − ti)|

Based on Theorem 4.4, since the error bound is related to timesteps and solver coefficients, we can define a much more compact search space consisting of {cji}Nj<i,j=0,i=1 and {ti}Ni=0.

Theorem 4.5. Based on Theorem 4.4 and Theorem 4.2. We can derive the total upper error bound(L1) of our solver search method and other counterparts.

The total upper error bound of Our solver search is:

N−1

- i
- j=0

gj(xi) + bji|

η|Ex

(ti+1 − ti)(

i

i=0

- i
- j=0

vjgj(xi) − Ex

+Ex

gj(xi)||)

i||

i

Compared to Adams-like linear multi-step methods. Our searched solver has a small upper error bound. The proof can be found in the Appendix I.

Through Theorem 4.5, our searched solvers own a relatively small upper error bound. Thus we can theoretically guarantee optimal compared to Adams-like methods.

### 5. Differentiable solver search.

Through previous discussion and analysis, we identify {cji}Nj<i,j=0,i=1 and {ti}Ni=0 as the target search items. To this end, we propose a differentiable data-driven solver search approach to determine these searchable items.

Timestep Parametrization. As shown in Algorithm 1, we employ unbounded parameters {ri,}iN=0−1 as the optimization subject, as the integral interval is from 0 to 1, we convert ri into time-space deltas ∆ti with softmax normalization function to force their summation to 1. We can access timestep ti+1 through ti+1 = ti + ∆ti. We initialize {ri}Ni=0−1 with 1.0 to obtain a uniform timestep distribution. Coefficients Parametrization. Inspired by (Xue et al., 2024). Given Equation (16) and Equation (7), when the velocity field vθ(x,t) yields constant value, an implicit constraint ik=0 cik = 1 emerges. This observation motivates us to re-parameterize the diagonal value of M as {1 − ij−=01 cji,}iN=0−1. We initialize {cki ,} with zeros to mimic the behavior of the Euler solver.

Algorithm 2 Differentiable Solver Search

Algorithm 1 Solver Parametrization

Require: vθ model, {∆ti, }Ni=0−1, M, A buffer Q. Compute {x˜l, }Ll=0 = Euler(ϵ, vθ) . for i = 0 to N − 1 do

Requires: {ri, } and {cji, } TimeDeltas: ∆t0, ∆t1, ..., ∆tn−1. SolverCoefficients: M ∈ RN×N {∆ti, }=Softmax({ri})

Q buffer← vθ(xti, ti) Compute v = ij=0 MijQj. ti+1 = ti + ∆ti xti+1 = xti + v∆ti





1 c01 1 − c01

M =

...

 

 

. . .

end for return: x˜tn−1, L({x˜l}Ll=0, {xi}Ni=0)

c0n−1 c1n−1 · · · 1 − nk=0−1 ckn−1

Mono-alignment Supervision. We take the L-step Euler solver’s ODE trajectory {x˜}Ll=0 as reference. We minimize the gap between the target and source trajectories with the MSE loss. We also adopt Huber loss as auxiliary supervision for xt

.

N

### 6. Extending to DDPM/VP framework

Applying our differentiable solver search to DDPM is infeasible. However, (Song et al., 2020b) suggests that there ex-

ists√βta}continuouscorrespondingSDEtoprocessdiscretewithDDPM.{f(t) =This−12βmotivatest;g(t) = us to transform the search space from the infeasible discrete space to its continuous SDE counterpart. (Lu et al.,

- 2022) and (Zhang & Chen, 2023) discover the semi-linear structure of diffusion and propose exponential integral with ϵ parametrization to tackle the fast sampling prob-

t

0 −12βsds, σt = 1 − e

lem of DDPM models, where αt = e

t

0 −βsds and λt = log α

σt . (Lu et al., 2023) further discovers that x parametrization is more powerful for diffusion sampling under limited steps, where x¯ = x

t

t−σϵ

αt .

σt σs

xt =

xs + σt

λt

eλx¯θ(xt(λ),t(λ))dλ (18)

λs

We opt to follow the x¯ parametrization as DPM-Solver++. However, we find directly interpolating eλxθ(xt,t) as a whole part is hard for searching, and yields worse results. To avoid conflating the interpolation coefficients with exponential integral, we employ ωt = α

σt and transform Equation (18) into Equation (19) with a similar interpolation format as Equation (15), where t(ω) maps ω to timestep.

t

xt ≈

σt σs

x¯s + σt(ωt − ωs)

i

cki xθ(x¯k,tk) (19)

k=1

### 7. Experiment

We demonstrate the efficiency of our differentiable solver search by conducting experiments on publicly available diffusion models. Specifically, we utilize DiT-XL/2 (Peebles

[Figure 3]

[Figure 4]

(a) FID of Search Model (b) FID of RefTraj Steps

Figure 3: Ablations studies of Differentiable Solver Search. We evaluate the searched solver on SiT-XL/2, and report the FID performance curve of searched solvers.

& Xie, 2023) trained with DDPM scheduling and rectifiedflow models SiT-XL/2 (Ma et al., 2024) and FlowDCNXL/2 (Wang et al., 2024b). Our default training setting employs the Lion optimizer (Chen et al., 2024c) with a constant learning rate of 0.01 and no weight decay. We sample 50,000 images for the entire search process. Notably, searching with 50,000 samples using FlowDCN-B/2 requires approximately 30 minutes on 8 × H20 computation cards. During the search, we deliberately avoid using CFG to construct reference and source trajectories, thereby preventing misalignment.

#### 7.1. Rectified Flow Models

We search solver with FlowDCN-B/2, FlowDCN-S/2 and SiT-XL/2. We compare the search solver’s performance with the second-order and fourth-order Adam multistep method on SiT-XL/2, FlowDCN-XL/2 trained on ImageNet 256 × 256 and FlowDCN-XL/2 trained on ImageNet 512 × 512.

Search Model. We tried different search models among different size and architecture. We report the FID performance of SiT-XL/2 in Figure 3a. Surprisingly, we find that the FID performance of SiT-XL/2 equipped with the solver searched using FlowDCN-B/2 outperforms the solver searched on SiT-XL/2 itself. The reconstruction error(in Appendix) between the sampled result produced by Euler-250 steps is as expected. These findings suggest that there exists a minor discrepancy between FID and the pursuit of minimal error

[Figure 5]

[Figure 6]

[Figure 7]

(a) SiT-XL/2-R256 (b) FlowDCN-XL/2-R256 (c) FlowDCN-XL/2-R512

- Figure 4: The same searched solver on different Rectified-Flow Models. R256 and R512 indicate the generation resolution of given model. We search solver with FlowDCN-B/2 on ImageNet-256 × 256 and evaluate it with SiT-XL/2 and FlowDCN-XL/2 on different resolution datasets. Our searched solver outperforms traditional solvers by a significant margin. More metrics(sFID, IS, Precision, Recall) are places at Appendix

in the current solver design.

Step of Reference Trajectory. We provide reference trajectory {x˜}Ll=0 of different sampling step L for differentiable solver search. We take FlowDCN-B/2 as the search model and report the FID measured on SiT-XL/2 in Figure 3b. As the sampling step of reference trajectory increases, the FID of SiT-XL/2 further improves and becomes better. However, the performance improvement is not significant when the number of steps is 5 or 6, which suggests that there is a limit to the improvement achievable with an extremely small number of steps.

ImageNet 256 × 256. We validate the searched solver on SiT-XL/2 and FlowDCN-XL/2. We arm the pre-trained model with CFG of 1.375. As shown in Figure 4a, our searched solver improves FID performance significantly and achieves 2.40 FID under 10 steps. As shown in Figure 4b, our searched solver achieves 2.35 FID under 10 steps, beating traditional solvers by large margins.

We also provide the comparison with recent solver-based distillation (Zhao et al., 2024) to demonstrate the efficiency of our searched solver in Table 1. Our searched solver achieves better metric performance under similar NFE with much fewer parameters.

ImageNet 512 × 512. Since (Ma et al., 2024) has not released SiT-XL/2 trained on 512×512 resolution, we directly report the performance collected from FlowDCN-XL/2. We arm FlowDCN-XL/2 with CFG of 1.375 and four channels. Our searched solver achieves 2.77 FID under 10 steps, beating traditional solver by a large margin, even slightly outperforming the Euler solver with 50 steps(2.81FID).

Text to Image. Shown in Figure 2, we apply our solver search on FlowDCN-B/2 and SiT-XL/2 to the most advanced Rectified-Flow model Flux.1-dev and SD3 (Esser et al., 2024). We find Flux.1-Dev would produce grid points in

SiT-XL-R256 NFE-CFG Params FID IS

Heun 16x2 0 3.68 / Heun 22x2 0 2.79 / Heun 30x2 0 2.42 /

- Adam2 15x2 / 2.49 236

- Adam2 16x2 0 2.42 237

- Adam4 15x2 / 2.33 242

- Adam4 16x2 0 2.27 243

- FlowTurbo (7+3)x2 2.9 × 107 3.93 224

- FlowTurbo (8+2)x2 2.9 × 107 3.63 / FlowTurbo (12+2)x2 2.9 × 107 2.69 / FlowTurbo (17+3)x2 2.9 × 107 2.22 248

- ours 6x2 21 3.57 214

- ours 7x2 28 2.78 229

- ours 8x2 36 2.65 234 ours 10x2 55 2.40 238 ours 15x2 55 2.24 244

Table 1: Comparsion with Distillation methods. Our searched solver achieves much better result under the same NFE with much fewer parameters.

generation. To alleviate the grid pattern, we decouple the velocity field into mean and direction, only apply our solver to the direction, and replace the mean with an exponential decayed mean. The details can be found in the appendix.

We also provide result of distillation on SD1.5 and SDXL with solver search in Appendix F.

#### 7.2. DDPM/VP Models

We choose the open-source DiT-XL/2(Peebles & Xie, 2023) model trained on ImageNet 256 × 256 as the search model to carry out the experiments. We compare the performance of the searched solver with DPM-Solver++ and UniPC on

[Figure 8]

- Figure 5: The images generated from PixArt-Σ with CFG=2.0 equipped with Our DS-Solver ( searched on DiT-XL/2-R256 ).In comparison to DPM-Solver++ and UniPC, our results consistently exhibit greater clarity and possess more details. Our solver achieves better quality from 5 to 10 steps(NFE).

Methods \NFEs 5 6 7 8 9 10 DPM-Solver++ with uniform-λ (Lu et al., 2023) 38.04 20.96 14.69 11.09 8.32 6.47 DPM-Solver++ with uniform-t (Lu et al., 2023) 31.32 14.36 7.62 4.93 3.77 3.23 DPM-Solver++ with uniform-λ-opt (Xue et al., 2024) 12.53 5.44 3.58 7.54 5.97 4.12 DPM-Solver++ with uniform-t-opt (Xue et al., 2024) 12.53 5.44 3.89 3.81 3.13 2.79 UniPC with uniform-λ (Zhao et al., 2023) 41.89 30.51 19.72 12.94 8.49 6.13 UniPC with uniform-t (Zhao et al., 2023) 23.48 10.31 5.73 4.06 3.39 3.04 UniPC with uniform-λ-opt (Xue et al., 2024) 8.66 4.46 3.57 3.72 3.40 3.01 UniPC with uniform-t-opt (Xue et al., 2024) 8.66 4.46 3.74 3.29 3.01 2.74 Searched-Solver 7.40 3.94 2.79 2.51 2.37 2.33

- Table 2: FID (↓) of different NFEs on DiT-XL/2-R256 . -opt indicates online optimization of the timesteps scheduler. Methods \NFEs 5 6 7 8 9 10 UniPC with uniform-λ (Zhao et al., 2023) 41.14 19.81 13.01 9.83 8.31 7.01 UniPC with uniform-t (Zhao et al., 2023) 20.28 10.47 6.57 5.13 4.46 4.14 UniPC with uniform-λ-opt (Xue et al., 2024) 11.40 5.95 4.82 4.68 6.93 6.01 UniPC with uniform-t-opt (Xue et al., 2024) 11.40 5.95 4.64 4.36 4.05 3.81 Searched-solver(searched on DiT-XL/2-R256) 10.28 6.02 4.31 3.74 3.54 3.64

- Table 3: FID (↓) of different NFEs on DiT-XL/2-R512. -opt indicates online optimization of the timesteps scheduler.

ImageNet 256 × 256 and ImageNet 512 × 512. ImageNet 256 × 256. Following (Peebles & Xie, 2023) and (Xue et al., 2024), We arm pre-trained DiT-XL/2 with

#### 7.3. Visualization Of Solver Parameters

GenEval Metrics

Solver Steps CFG

Overall Color.Attr Two.Obj Pos

Searched Coefficients are visualized in Figure 1. The absolute value of searched coefficients corresponding to DDPM/VP shares a different pattern, coefficients in DDPM/VP are more concentrated on the diagonal while rectified-flow demonstrates a more flattened distribution. This indicates there exists a more curved sampling path in DDPM/VP compared to rectified-flow.

5 2.0 6.50 33.08 4.75 0.40519

DPM++

- 8 2.0 5.25 39.65 5.75 0.43074

UniPC

5 2.0 6.50 34.85 5.25 0.41387

- 8 2.0 6.72 40.66 6.00 0.44134

Ours

5 2.0 5.25 37.37 4.75 0.41933

- 8 2.0 7.25 42.68 7.50 0.45064

Searched Timesteps are visualized in Figure 1. Compared to DDPM/VP, rectified-flow models more focus on the more noisy region, exhibiting small time deltas at the beginning. We fit the searched timestep of different NFE with polynomials and provide the respacing curves in Equation (20) and Equation (21). t ∈ [0,1], and t = 0 indicates the most noisy timestep.

- Table 4: Results on GenEval Benchmark for PixArt at 512 Resolution.Our searched solver achieves better performance compared with UniPC/DPM++ on PixArt-512 × 512.

Steps FID sFID IS PR Recall

DPM++ 5 60.0 209 25.59 0.36 0.20 DPM++ 8 38.4 116.9 33.0 0.50 0.36 DPM++ 10 35.6 114.7 33.7 0.53 0.37

UniPC 5 57.9 206.4 25.88 0.38 0.20 UniPC 8 37.6 115.3 33.3 0.51 0.36 UniPC 10 35.3 113.3 33.6 0.54 0.36

Ours 5 46.4 204 28.0 0.46 0.23 Ours 8 33.6 115.2 32.6 0.54 0.39 Ours 10 33.4 114.7 32.5 0.55 0.39

- Table 5: Metrics of different NFEs on PixArt-α (Our Solver are searched on ImageNet 256x256).

ReFlow : − 1.96t4 + 3.51t3 − 0.97t2 + 0.43t (20) DDPM/VP : − 2.73t4 + 6.30t3 − 4.744t2 + 2.17t (21)

### 8. Conclusion

We have found a compact solver search space and proposed a novel differentiable solver search algorithm to identify the optimal solver. Our searched solver outperforms traditional solvers by a significant margin. Equipped with the searched solver, DDPM/VP and Rectified Flow models significantly improve under limited sampling steps. However, our proposed solver still has several limitations which we plan to address in future work.

CFG of 1.5 and apply CFG only on the first three channels. As shown in Table 2, our searched solver improves FID performance significantly and achieves 2.33 FID under 10 steps.

### 9. Limitations

In the main paper, we demonstrate text-to-image visualization with a small CFG value. However, it is intuitive that using a larger CFG would result in superior image quality. We attribute the inferior performance of large CFGs in our solver to the limitations of current naive solver structures and searching techniques. We hypothesize that incorporating predictor-corrector solver structures would enhance numerical stability and yield better images. Additionally, training with CFGs may also be beneficial.

ImageNet 512×512. We directly apply the solver searched on 256 × 256 resolution to ImageNet 512 × 512. The result is also great to some extent, DiT-XL/2(512 × 512) achieves 3.64 FID under 10 steps, outperforming DPM-Solver++ and UniPC with a large gap.

Text to Image. As we search solver with DiT and its corresponding noise scheduler, so it is infeasible to apply our solver to other DDPM models with different βmin and βmax. Fortunately, we find (Chen et al., 2024b) and (Chen et al., 2023) also employ the same βmin and βmax as DiT. So we can provide the visualization results of our searched solver on PixArt-Σ and PixArt-α. Our visualization result is produced with CFG of 2. We take PixArt-alpha as the text-to-image model. We follow the evaluation pipeline of ADM and take COCO17-Val as the reference batch. We generate 5k images using DPM-Solver++, UniPC and our solver searched on DiT-XL/2-R256. Also, we provided the performance results on GenEval Benchmark (Ghosh et al.,

### Impact Statement

This paper proposes a search-based solver for fast diffusion sampling. We acknowledge that it could lower the barrier for creating diffusion-based AIGC contents.

Acknowledgements. This work is supported by National Key R&D Program of China (No. 2022ZD0160900), Jiangsu Frontier Technology Research and Development Program (No. BF2024076), and the Collaborative Innovation Center of Novel Software Technology and Industrializa-

- 2023) in Section 7.2.

tion, Alibaba Group through Alibaba Innovative Research Program.

### References

Anderson, B. D. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313– 326, 1982.

Bashforth, F. and Adams, J. C. An attempt to test the theories of capillary action by comparing the theoretical and measured forms of drops of fluid. University Press, 1883.

Brock, A., Donahue, J., and Simonyan, K. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022.

Chen, D., Zhou, Z., Wang, C., Shen, C., and Lyu, S. On the trajectory regularity of ode-based diffusion sampling. arXiv preprint arXiv:2405.11326, 2024a.

Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., et al. Pixart-\alpha: Fast training of diffusion transformer for photorealistic textto-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Kang, M., Zhang, R., Barnes, C., Paris, S., Kwak, S., Park, J., Shechtman, E., Zhu, J.-Y., and Park, T. Distilling diffusion models into conditional gans. arXiv preprint arXiv:2405.05967, 2024.

Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35: 26565–26577, 2022.

Kim, D., Lai, C.-H., Liao, W.-H., Murata, N., Takida, Y., Uesaka, T., He, Y., Mitsufuji, Y., and Ermon, S. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.

Lin, S., Wang, A., and Yang, X. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., and Li, Z. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024b.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36, 2024c.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Gao, Y., Pan, Z., Zhou, X., Kang, L., and Chaudhari, P. Fast diffusion probabilistic model sampling through the lens of backward error analysis. arXiv preprint arXiv:2304.11446, 2023.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Lu, C., Zhou, Y., Bao, F., Chen, J., LI, C., and Zhu, J. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 5775–5787, 2022.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. Dpmsolver++: Fast solver for guided sampling of diffusion probabilistic models, 2023.

Ma, N., Goldstein, M., Albergo, M. S., Boffi, N. M., VandenEijnden, E., and Xie, S. Sit: Exploring flow and diffusionbased generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Sabour, A., Fidler, S., and Kreis, K. Align your steps: Optimizing sampling schedules in diffusion models. arXiv preprint arXiv:2404.14507, 2024.

Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Sauer, A., Schwarz, K., and Geiger, A. Stylegan-xl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pp. 1–10, 2022.

Shaul, N., Perez, J., Chen, R. T., Thabet, A., Pumarola, A., and Lipman, Y. Bespoke solvers for generative flow models. arXiv preprint arXiv:2310.19075, 2023.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. arXiv:2010.02502, October 2020a. URL https://arxiv.org/abs/2010.02502.

Song, T., Feng, W., Wang, S., Li, X., Ge, T., Zheng, B., and Wang, L. Dmm: Building a versatile image generation model via distillation-based model merging. arXiv preprint arXiv:2504.12364, 2025.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Wang, F.-Y., Huang, Z., Bergman, A. W., Shen, D., Gao, P., Lingelbach, M., Sun, K., Bian, W., Song, G., Liu, Y., et al. Phased consistency model. arXiv preprint arXiv:2405.18407, 2024a.

Wang, S., Teng, Y., and Wang, L. Deep equilibrium object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6296–6306, 2023.

Wang, S., Li, Z., Song, T., Li, X., Ge, T., Zheng, B., and Wang, L. Flowdcn: Exploring dcn-like architectures for fast image generation with arbitrary resolution. arXiv preprint arXiv:2410.22655, 2024b.

Wang, S., Tian, Z., Huang, W., and Wang, L. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.

Xue, S., Liu, Z., Chen, F., Zhang, S., Hu, T., Xie, E., and Li, Z. Accelerating diffusion sampling with optimized time steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8292– 8301, 2024.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024.

Zhang, Q. and Chen, Y. Fast sampling of diffusion models with exponential integrator. In The Eleventh International Conference on Learning Representations, 2023.

Zhao, W., Bai, L., Rao, Y., Zhou, J., and Lu, J. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. arXiv preprint arXiv:2302.04867, 2023.

Zhao, W., Shi, M., Yu, X., Zhou, J., and Lu, J. Flowturbo: Towards real-time flow-based image generation with velocity refiner. arXiv preprint arXiv:2409.18128, 2024.

Zheng, J., Hu, M., Fan, Z., Wang, C., Ding, C., Tao, D., and Cham, T.-J. Trajectory consistency distillation. arXiv preprint arXiv:2402.19159, 2024.

Zhou, M., Zheng, H., Wang, Z., Yin, M., and Huang, H. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.

### A. More Metrics of Searched Solver

We adhere to the evaluation guidelines provided by ADM and DM-nonuniform, reporting only the FID as the standard metric in Figure 4a. To clarify, we do not report selective results on rectified flow models; we present sFID, IS, PR, and Recall metrics for SiT-XL(R256), FlowDCN-XL/2(R256), and FlowDCN-B/2(R256). Our solver searched on FlowDCN-B/2, consistently outperforms the handcrafted solvers across FID, sFID, IS, and Recall metrics.

### B. Computational complexity compared to other methods.

For sampling. When performing sampling over n time steps, our solver caches all pre-sampled predictions, resulting in a memory complexity of O(n). The model function evaluation also has a complexity of O(n) (O(2 × n) for CFG enabled). It is important to note that the memory required for caching predictions is negligible compared to that used by model weights and activations. Besides classic methods, we have also included a comparison with the latest Flowturbo published on NeurIPS24.

| |Steps|NFE|NFE-CFG<br><br>|Cache Pred|Order<br><br>|search samples|
|---|---|---|---|---|---|---|
|Adam2|n<br><br>|n<br><br>|2n|2<br><br>|2<br><br>|/|
|Adam4<br><br>|n|n<br><br>|2n|4<br><br>|4<br><br>|/|
|heun|n|2n<br><br>|4n<br><br>|2<br><br>|2|/|
|DPM-Solver++<br><br>|n<br><br>|n|2n<br><br>|2|2<br><br>|/|
|UniPC<br><br>|n|n<br><br>|2n<br><br>|3|3<br><br>|/|
|FlowTurbo|n<br><br>|>n<br><br>|>2n|2|2<br><br>|540000(Real)|
|our<br><br>|n<br><br>|n<br><br>|2n|n<br><br>|n<br><br>|50000(Generated)|

For searching. Solver-based algorithms, limited by their searchable parameter sizes, demonstrate significantly lower performance in few-step settings compared to distillation-based algorithms(5/6steps), making direct comparisons inappropriate. Consequently, we selected algorithms that are both acceleratable on ImageNet and comparable in performance, including popular methods such as DPM-Solver++, UniPC, and classic Adams-like linear multi-step methods. Since our experiments primarily utilize SiT, DiT, and FlowDCN that trained on the ImageNet dataset. We also provide fair comparisons by incorporating the latest acceleration method, FlowTurbo. Additionally, we have included results from the heun method as reported in FlowTurbo.

### C. Ablation on Search Samples

We ablate the number of search samples on the 10-step and 8-step solver settings. Samples means the total training samples the searched solver has seen. Unique Samples means the total distinct samples the searched solver has seen. Our searched solver converges fast and gets saturated near 30000 samples.

|iters(10-step-solver)|samples<br><br>|unique samples<br><br>|FID<br><br>|IS|PR|Recall|
|---|---|---|---|---|---|---|
|313|10000<br><br>|10000<br><br>|2.54<br><br>|239|0.79|0.59|
|626<br><br>|20000|10000<br><br>|2.38|239|0.79<br><br>|0.60|
|939|30000<br><br>|10000|2.49<br><br>|240<br><br>|0.79|0.59|
|1252<br><br>|40000<br><br>|10000|2.29|239|0.80|0.60|
|1565<br><br>|50000|10000<br><br>|2.41<br><br>|240|0.80<br><br>|0.59|
|626|20000<br><br>|20000<br><br>|2.47<br><br>|237<br><br>|0.78|0.60|
|939|30000<br><br>|30000<br><br>|2.40<br><br>|238<br><br>|0.79|0.60|
|1252|40000<br><br>|40000|2.48<br><br>|237<br><br>|0.80|0.59|
|1565|50000<br><br>|50000|2.41<br><br>|239<br><br>|0.80|0.59|

### D. Solver Across different variance schedules

Since our solvers are searched on a specific noise scheduler and its corresponding pre-trained models, applying the searched coefficients and timesteps to other noise schedulers yields meaningless results. We have tried applied searched solver on

[Figure 9]

[Figure 10]

[Figure 11]

(a) SiT-XL/2-R256 (b) FlowDCN-XL/2-R256 (c) FlowDCN-XL/2-R512

[Figure 12]

[Figure 13]

[Figure 14]

(d) SiT-XL/2-R256 (e) FlowDCN-XL/2-R256 (f) FlowDCN-XL/2-R512

[Figure 15]

[Figure 16]

[Figure 17]

(g) SiT-XL/2-R256 (h) FlowDCN-XL/2-R256 (i) FlowDCN-XL/2-R512

[Figure 18]

[Figure 19]

[Figure 20]

(j) SiT-XL/2-R256 (k) FlowDCN-XL/2-R256 (l) FlowDCN-XL/2-R512

- Figure 6: The same searched solver on different Rectified-Flow Models. R256 and R512 indicate the generation resolution of given model. We search solver with FlowDCN-B/2 on ImageNet-256 × 256 and evaluate it with SiT-XL/2 and FlowDCN-XL/2 on different resolution datasets. Our searched solver outperforms traditional solvers by a significant margin.

|iters(8-step-solver)<br><br>|samples<br><br>|unique samples<br><br>|FID|IS|PR<br><br>|Recall|
|---|---|---|---|---|---|---|
|313<br><br>|10000<br><br>|10000<br><br>|2.99|228|0.78<br><br>|0.59|
|626|20000|10000<br><br>|2.78|229<br><br>|0.79|0.60|
|939|30000<br><br>|10000<br><br>|2.72<br><br>|235<br><br>|0.79|0.60|
|1252|40000<br><br>|10000|2.67<br><br>|228|0.79<br><br>|0.60|
|1565<br><br>|50000|10000<br><br>|2.69|235<br><br>|0.79<br><br>|0.59|
|626|20000<br><br>|20000<br><br>|2.70<br><br>|231<br><br>|0.79|0.59|
|939<br><br>|30000<br><br>|30000|2.82<br><br>|232|0.79|0.59|
|1252<br><br>|40000<br><br>|40000|2.79<br><br>|231|0.79<br><br>|0.60|
|1565|50000|50000<br><br>|2.65<br><br>|234|0.79<br><br>|0.60|

SiT(Rectified flow) and DiT(DDPM with βmin = 0.1,βmax = 20) to SD1.5(DDPM with βmin = 0.085,βmax = 12), but the results were inconclusive. Notably, despite sharing the DDPM name, DiT and SD1.5 employ distinct βmin,βmax values, thereby featuring different noise schedulers. A more in-depth discussion of these experiments can be found in Section(Extend to DDPM/VP).

### E. Solver for different variance schedules

As every DDPM has a corresponding continuous VP scheduler, so we can transform the discreet DDPM into continuous VP, thus we successfully searched better solver compared to DPM-Solvers. The details can be found in Section 6. To put it simply, under the empowerment of our high-order solver, the performance of DDPM and FM does not differ significantly (8, 9, 10 steps), which contradicts the common belief that FM is stronger at limited sampling steps.

### F. Text to image Distillation Experiments

We unify distillation and solver search to obtain high-quality multi-step generative models. We adopt adversarial training and trajectory supervision. We will open source the training code of unified training techniques.

Table 6: Performance comparison on validation set of COCO-2017.

Method Res. Time (↓) # Steps # Param. FID (↓) SDv1-5+DPMSolver (Upper-Bound) (Lu et al., 2022) 512 0.88s 25 0.9B 20.1

Rectified Flow 512 0.88s 25 0.9B 21.65 Rectified Diffusion 512 0.88s 25 0.9B 21.28 Rectified Flow 512 0.21s 4 0.9B 103.48 PeRFlow 512 0.21s 4 0.9B 22.97 Rectified Diffusion 512 0.21s 4 0.9B 20.64 Ours(Distillation+solver search) 512 0.21s 4 0.9B 18.99

PeRFlow-SDXL 1024 0.71s 4 3B 27.06 Rectified Diffusion-SDXL 1024 0.71s 4 3B 25.81 Ours(LORA+Distillation+solver search) 1024 0.71s 4 3B 21.3

### G. Limitations.

We place the limitation at the appendix, in order to provide more discussion space and obtain more insights from reviews. We copy the original limitation content and add more.

Misalignd Reconstrucion loss and Performance. Our proposed methods are specifically designed to minimize integral error within a limited number of steps. However, ablation studies reveal a mismatch between FID performance and Reconstruction error. To address this issue, we plan to enhance our searched solver by incorporating distribution matching supervision, thereby better aligning sampling quality.

Larger CFG Inference. In the main paper, we demonstrate text-to-image visualization with a small CFG value. However, it is intuitive that utilizing a larger CFG would result in superior image quality. We attribute the inferior performance of

Table 7: Performance comparison on COCO-2014.

Method Res. Time (↓) # Steps # Param. FID (↓) Stable Diffusion XL (3B) and its accelerated or distilled versions

SDXL-Turbo 512 0.34s 4 3B 23.19 SDXL-Lightning 1024 0.71s 4 3B 24.56 DMDv2 1024 0.71s 4 3B 19.32 LCM 1024 0.71s 4 3B 22.16 Phased Consistency Model 1024 0.71s 4 3B 21.04 PeRFlow-XL 1024 0.71s 4 3B 20.99 Rectified Diffusion-XL (Phased) 1024 0.71s 4 3B 19.71 Ours(LORA+Distillation+solver search) 1024 0.71s 4 3B 11.4

large CFGs on our solver to the limitations of current naive solver structures and searching techniques. We hypothesize that incorporating predictor-corrector solver structures would enhance numerical stability and yield better images. Additionally, training with CFGs may also be beneficial.

Resource Consumption We can hard code the searched coefficients and timesteps into the program files. However, Compared to hand-crafted solvers, our solver still needs a searching process.

### H. Proof of pre-integral error expectation

- Theorem H.1. Given sampling time interval [ti,ti+1] and suppose Cj(x) = gj(x) + bji, Adams-like linear multi-step

i|| ij=0 vjgj(xi)||. Our solver search(replacing Cj(x) with Ex

methods will introduce an upper error bound of (ti+1 − ti)Ex

i|| ij=0 vj[gj(xi) − Ex

[Cj(xi)]) owns an upper error bound of (ti+1 − ti)Ex

i

gj(xi)||

i

Proof. Suppose Cj(xi) = gj(xi) + bji. Adams-like linear multi-step methods would not consider x-related interpolation. thus pre-integral coefficients of Adams-like linear multi-step methods will only reduce into b. We obtain the error expectation of the pre-integral of Adams-like linear multi-step methods:

- i
- j=0

- i
- j=0

vjbji(ti+1 − ti)|| (22)

Ex

vj[Cj(xi)](ti+1 − ti) −

i||

- i
- j=0

vj(ti+1 − ti)[Cj(xi) − bji|| (23)

=Ex

i||

i

vjgj(xi)|| (24)

=(ti+1 − ti)Ex

i||

j=0

We obtain the error expectation of the pre-integral of our solver search methods:

- i
- j=0

- i
- j=0

[Cj(xi)](ti+1 − ti)|| (25)

Ex

#### vjEx

i||

vj[Cj(xi)](ti+1 − ti) −

i

- i
- j=0

iCj(xi)|| (26)

=Ex

vj(ti+1 − ti)[Cj(xi) − Ex

i||

i

gj(xi)|| (27)

vj[gj(xi) − Ex

=(ti+1 − ti)Ex

i||

i

j=0

Next, define the optimization problem:

E = Ex

i||

- i
- j=0

vj[gj(xi) − aj]||22.

We suppose different vj are orthogonal and ||vj||22 = 1. As we leave cij as the expectation of Cj(xi), we will demonstrate this choice is optimal.

∂E ∂aj

(||vj||22(gj(xi) − aj)) (28)

= −2Ex

i

= 0, we obtain: aj = Ex

igi(xi)||vj||22 Exi||vj||22

iCj(xi) − bji. So our searched solver has a lower and optimal error expectation:

Let ∂a∂E

= Ex

gj(xi) = Ex

i

j

(ti+1 − ti)Ex

i||

- i
- j=0

vj[gj(xi) − Ex

gj(xi)]|| ≤ (ti+1 − ti)Ex

i||

i

- i
- j=0

vjgj(xi)|| (29)

Recall Assumption 4.1, the integral upper error bound of universal interpolation P will be:

- i
- j=0

ti+1

ti+1

P(xt,t,xj,tj)dt||. (30)

||

v(xt,t)dt −

vj

ti

ti

i

ti+1

ti+1

P(xt,t,xj,tj)vjdt||. (31)

=||

v(xt,t)dt −

ti

ti

j=0

- i
- j=0

ti+1

P(xt,t,xj,tj)vj]dt||. (32)

=||

[v(xt,t) −

ti

- i
- j=0

ti+1

P(xt,t,xj,tj)vj||dt. (33)

||v(xt,t) −

<

ti

<(ti+1 − ti)[O(dxm) + O(dtn)] (34)

Combining Equation (34) and the error expectation of the pre-integral part, we will get the total error bound of the solver

search.

- i
- j=0

ti+1

[Cj(xi)](ti+1 − ti)||. (35)

#### vjEx

v(xt,t)dt −

||

i

ti

- i
- j=0

ti+1

ti+1

P(xt,t,xj,tj)dt+ (36)

v(xt,t)dt −

=||

vj

ti

ti

- i
- j=0

- i
- j=0

ti+1

[Cj(xi)](ti+1 − ti)||. (37)

#### vjEx

P(xt,t,xj,tj)dt −

vj

i

ti

i

ti+1

ti+1

P(xt,t,xj,tj)dt||+ (38)

<||

v(xt,t)dt −

vj

ti

ti

j=0

i

i

ti+1

[Cj(xi)](ti+1 − ti)||. (39)

#### vjEx

||

P(xt,t,xj,tj)dt −

vj

i

ti

j=0

j=0

i

ti+1

ti+1

P(xt,t,xj,tj)dt||+ (40)

=||

v(xt,t)dt −

vj

ti

ti

j=0

i

- i
- j=0

[Cj(xi)](ti+1 − ti)||. (41)

#### vjEx

||

vj[Cj(xi)](ti+1 − ti) −

i

j=0

i

<(ti+1 − ti)[O(dxm) + O(dtn)] + (ti+1 − ti)Ex

gj(xi)]|| (42)

vj[gj(xi) − Ex

i||

i

j=0

- i
- j=0

<(ti+1 − ti)([O(dxm) + O(dtn)] + Ex

gj(xi)]||) (43)

vj[gj(xi) − Ex

i||

i

i|| ij=0 vj[gj(xi) − Ex

Since ((O(dxm) + O(dtn)) is much smaller than Ex

gj(xi)]||. We can omit the ((O(dxm) + O(dtn)) term.

i

| |
|---|

### I. Proof of total upper error bound

- Theorem I.1. Compared to Adams-like linear multi-step methods. Our Solver search has a small upper error bound. The total upper error bound of Adams-like linear multi-step methods is:

N−1

1 N

(

i=0

- i
- j=0

η|bji| + Ex

i||

)

The total upper error bound of Our solver search is:

- i
- j=0

vj[gj(xi)]||)

N−1

- i
- j=0

(ti+1 − ti)

i=0

gj(xi) + bji| + Ex

η|Ex

i||

i

- i
- j=0

vjgj(xi) − Ex

gj(xi)||)

i

Proof. We donate the continuous integral result of the ideal velocity field vˆ as xˆ, the solved integral result of the ideal velocity field vˆ as xˆN, the continuous integral result of the pre-trained velocity model vθ as xˆ, the solved integral result of the pre-trained velocity model vθ as xN.

N−1

xN = ϵ +

i=0

- i
- j=0

vjcji(ti+1 − ti) (44)

The error caused by the non-ideal velocity estimation model can be formulated in the following equation. we can employ triangular inequalities to obtain the error-bound ||xN − xˆN||, which is related to solver coefficients and timestep choices.

N−1

i

(vj − vˆj)cji(ti+1 − ti)|

||xN − xˆN|| = |

i=0

j=0

N−1

i

|(vj − vˆj)cji(ti+1 − ti)|

≤

i=0

j=0

N−1

- i
- j=0

|vj − vˆj)| × |cji(ti+1 − ti)|

≤

i=0

N−1

- i
- j=0

|cji(ti+1 − ti)|

≤ η

i=0

The total error of our searched solver is:

||xN − xˆ||

=||xN − xˆN + xˆN − xˆ|| ≤||xN − xˆN|| + ||xˆN − xˆ|| ≤η

N−1

- i
- j=0

|cji(ti+1 − ti)|+

i=0

N−1

(ti+1 − ti)(O(dxm) + O(dtn) + Ex

i||

i=0

N−1

- i
- j=0

|cji(ti+1 − ti)| + (ti+1 − ti)Ex

≈

i||

η

i=0

N−1

- i
- j=0

gj(xi) + bji| + Ex

η|Ex

(ti+1 − ti)

i||

=

i

i=0

- i
- j=0

vj[gj(xi) − Ex

gj(xi)]||)

i

- i
- j=0

vj[gj(xi) − Ex

gj(xi)]||)

i

i

vj[gj(xi) − Ex

##### gj(xi)]||)

i

j=0

The total error of Adams-like linear multi-step method is:

N−1

1 N

(

i=0

i

η|bji| + Ex

i||

)

j=0

- i
- j=0

vj[gj(xi)]||)

i|| ij=0 vj[gj(xi)]||) is not equal between different timestep intervals, Optimized timesteps owns smaller upper error bound than uniform timesteps. Recall that η ≪ ||vj||, the error is mainly determined by Ex

Obviously, as ( ij=0 η|bji|+Ex

i|| ij=0 vj[gj(xi)]||. Recall that Ex

i|| ij=0 vj[gj(xi)]||, thus our solver search has a minimal upper error bound because we search coefficients and timesteps simultaneously.

i|| ij=0 vj[gj(xi) − Ex

gj(xi)]|| ≤ Ex

i

| |
|---|

- J. Searched Parameters We provide the searched parameters ∆t and cji. Note cji needs to be converted into M follwing Algorithm 1.

- J.1. Solver Searched on SiT-XL/2 NFE TimeDeltas ∆t Coeffcients cji

- 5



 

0.0424 0.1225 0.2144 0.3073 0.3135



 



 

- 0.0 0.0 0.0 0.0 0.0

−1.17 0.0 0.0 0.0 0.0

- 1.07 −1.83 0.0 0.0 0.0 0.0 0.0 −0.93 0.0 0.0 0.0 0.0 0.0 −0.71 0.0



 

- 6



 

0.0389 0.0976 0.161 0.2046 0.2762 0.2217



 



 

- 0.0 0.0 0.0 0.0 0.0 0.0

−1.04 0.0 0.0 0.0 0.0 0.0

- 1.62 −2.98 0.0 0.0 0.0 0.0

−1.32 2.52 −2.04 0.0 0.0 0.0 0.0 0.0 0.0 −0.76 0.0 0.0 0.0 0.0 0.0 0.0 −0.66 0.0



 

- 7



 

0.0299 0.0735 0.1119 0.1451 0.1959 0.2698 0.1738



 



 

- 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−0.93 0.0 0.0 0.0 0.0 0.0 0.0

- 1.23 −2.31 0.0 0.0 0.0 0.0 0.0

−0.59 1.53 −2.09 0.0 0.0 0.0 0.0 −0.09 −0.07 0.99 −1.91 0.0 0.0 0.0

0.05 −0.21 0.09 0.55 −1.47 0.0 0.0 −0.05 0.19 −0.31 0.37 0.67 −1.79 0.0



 

- 8



 

0.0303 0.0702 0.0716 0.1112 0.1501 0.1833 0.2475 0.1358



 



 

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−0.92 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.78 −1.7 0.0 0.0 0.0 0.0 0.0 0.0 0.06 0.52 −1.76 0.0 0.0 0.0 0.0 0.0

−0.02 −0.16 0.98 −1.8 0.0 0.0 0.0 0.0 −0.02 −0.12 0.22 0.24 −1.36 0.0 0.0 0.0

−0.1 0.06 −0.02 0.18 0.12 −1.1 0.0 0.0 −0.16 0.14 −0.02 −0.02 0.38 0.32 −1.72 0.0



 

- 9



 

0.028 0.0624 0.0717 0.0894 0.1092 0.1307 0.1729 0.2198 0.1159



 



 

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−0.93 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.63 −1.29 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.39 −0.11 −1.41 0.0 0.0 0.0 0.0 0.0 0.0

−0.07 −0.05 0.83 −1.59 0.0 0.0 0.0 0.0 0.0

0.07 −0.11 0.27 0.27 −1.53 0.0 0.0 0.0 0.0 −0.05 0.03 0.01 0.15 0.17 −1.15 0.0 0.0 0.0 −0.21 0.27 −0.07 −0.03 0.19 0.09 −0.99 0.0 0.0 −0.15 0.15 0.03 −0.09 0.25 0.25 0.21 −1.71 0.0



 

- 10









0.0279 0.0479 0.0646 0.0659 0.1045 0.1066 0.1355 0.1622 0.1942 0.0908

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−0.95 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.59 −1.17 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.35 −0.11 −1.45 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−0.13 0.01 0.75 −1.49 0.0 0.0 0.0 0.0 0.0 0.0 0.05 −0.05 0.31 0.29 −1.59 0.0 0.0 0.0 0.0 0.0 0.05 −0.03 −0.09 0.23 0.17 −1.19 0.0 0.0 0.0 0.0

−0.03 0.07 −0.09 −0.03 0.27 −0.03 −0.91 0.0 0.0 0.0 −0.15 0.17 0.03 −0.09 0.05 0.09 0.05 −0.79 0.0 0.0 −0.17 0.11 0.15 0.03 0.05 0.25 0.05 −0.07 −1.49 0.0

 

 

 

 

#### J.2. Solver Searched on FlowDCN-B/2 NFE TimeDeltas ∆t Coeffcients cji

- 5



 

0.0521 0.1475 0.2114 0.2797 0.3092



 



 

- 0.0 0.0 0.0 0.0 0.0

−1.26 0.0 0.0 0.0 0.0

- 1.38 −2.26 0.0 0.0 0.0 0.0 0.0 −0.92 0.0 0.0 0.0 0.0 0.0 −0.7 0.0



 

- 6



 

0.0391 0.0924 0.165 0.2015 0.2511 0.2511



 



 

- 0.0 0.0 0.0 0.0 0.0 0.0

−1.22 0.0 0.0 0.0 0.0 0.0

- 1.12 −2.0 0.0 0.0 0.0 0.0 −0.3 0.9 −1.56 0.0 0.0 0.0

0.0 0.0 0.0 −0.74 0.0 0.0 0.0 0.0 0.0 0.0 −0.62 0.0



 

- 7



 

0.0387 0.0748 0.103 0.1537 0.184 0.234 0.2117



 



 

- 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−1.11 0.0 0.0 0.0 0.0 0.0 0.0

- 1.03 −1.99 0.0 0.0 0.0 0.0 0.0 0.07 0.43 −1.57 0.0 0.0 0.0 0.0

−0.21 −0.15 1.53 −2.29 0.0 0.0 0.0 −0.05 0.07 −0.23 0.61 −1.33 0.0 0.0 −0.17 0.31 −0.41 0.17 0.59 −1.31 0.0



 

- 8



 

0.0071 0.0613 0.078 0.1163 0.1421 0.188 0.2077 0.1996



 



 

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−2.43 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.61 −1.55 0.0 0.0 0.0 0.0 0.0 0.0 0.99 −0.11 −2.07 0.0 0.0 0.0 0.0 0.0 0.05 −0.49 1.33 −1.93 0.0 0.0 0.0 0.0 0.05 −0.33 0.23 0.73 −1.71 0.0 0.0 0.0

−0.09 0.25 −0.29 0.05 0.61 −1.45 0.0 0.0 −0.23 0.21 −0.01 −0.25 0.25 0.41 −1.25 0.0



 

- 9



 

0.0017 0.051 0.0636 0.0911 0.1007 0.1443 0.1694 0.191 0.1872



 



 

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −6.19 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −0.11 −0.81 0.0 0.0 0.0 0.0 0.0 0.0 0.0

0.73 −0.17 −1.37 0.0 0.0 0.0 0.0 0.0 0.0 0.31 −0.05 0.19 −1.45 0.0 0.0 0.0 0.0 0.0 0.03 −0.23 0.29 0.35 −1.35 0.0 0.0 0.0 0.0

−0.19 0.05 0.01 0.21 0.25 −1.23 0.0 0.0 0.0 −0.23 0.21 −0.13 0.17 0.09 0.09 −1.09 0.0 0.0 −0.17 0.15 0.11 −0.19 0.03 0.23 0.17 −1.21 0.0



 

- 10









0.0016 0.0538 0.0347 0.0853 0.0853 0.1198 0.1351 0.165 0.1788 0.1406

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−7.8801 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −0.4 −0.74 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.48 −0.18 −0.86 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.26 −0.04 −0.04 −1.28 0.0 0.0 0.0 0.0 0.0 0.0

0.0 −0.06 0.26 0.26 −1.42 0.0 0.0 0.0 0.0 0.0

−0.1 −0.06 0.08 0.2 0.22 −1.24 0.0 0.0 0.0 0.0 −0.18 0.14 −0.08 0.1 0.08 0.14 −1.06 0.0 0.0 0.0 −0.12 0.16 −0.1 0.04 0.08 0.06 0.08 −1.02 0.0 0.0 −0.16 0.02 0.14 0.0 −0.14 0.08 0.14 0.34 −1.38 0.0

 

 

 

 

#### J.3. Solver Searched on DiT-XL/2 NFE TimeDeltas ∆t Coeffcients cji









0.0 0.0 0.0 0.0 0.0 −1.43 0.0 0.0 0.0 0.0

0.2582 0.1766 0.1766 0.2156 0.1731

5

0.93 −1.55 0.0 0.0 0.0 0.0 0.0 −0.69 0.0 0.0 0.0 0.0 0.0 −0.59 0.0

 

 

 

 









0.0 0.0 0.0 0.0 0.0 0.0 −1.36 0.0 0.0 0.0 0.0 0.0

0.2483 0.1506 0.1476 0.1568 0.1733 0.1233

0.9 −1.84 0.0 0.0 0.0 0.0

6

−0.08 0.5 −1.08 0.0 0.0 0.0 0.0 0.0 0.0 −0.56 0.0 0.0 0.0 0.0 0.0 0.0 −0.56 0.0

 

 

 

 









- 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −1.38 0.0 0.0 0.0 0.0 0.0 0.0
- 1.08 −2.02 0.0 0.0 0.0 0.0 0.0 −0.28 0.78 −1.52 0.0 0.0 0.0 0.0

0.2241 0.1415 0.1205 0.1158 0.1443 0.1627 0.0911

7

−1.4901e − 08 −0.1 0.64 −1.5 0.0 0.0 0.0 0.06 −0.06 −0.06 0.26 −1.0 0.0 0.0 0.0 −0.1 0.02 0.2 0.26 −1.12 0.0

 

 

 

 









0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −1.14 0.0 0.0 0.0 0.0 0.0 0.0 0.0

0.2033 0.1476 0.1094 0.099 0.1116 0.1233 0.131 0.0748

0.8 −1.76 0.0 0.0 0.0 0.0 0.0 0.0 0.02 0.48 −1.62 0.0 0.0 0.0 0.0 0.0

8

−0.12 0.06 0.62 −1.42 0.0 0.0 0.0 0.0 0.04 −0.1 0.12 0.16 −1.04 0.0 0.0 0.0 0.06 −0.04 −0.06 0.08 −0.08 −0.56 0.0 0.0

 

 

 

 

−0.02 −0.04 −0.04 0.12 0.14 0.04 −0.9 0.0









0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −1.28 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

0.1959 0.1313 0.1142 0.0863 0.0898 0.0916 0.1119 0.1054 0.0735

0.78 −1.62 0.0 0.0 0.0 0.0 0.0 0.0 0.0 −0.02 0.44 −1.48 0.0 0.0 0.0 0.0 0.0 0.0

9

−0.1 0.16 0.36 −1.3 0.0 0.0 0.0 0.0 0.0

−0.06 −0.04 0.22 0.12 −1.08 0.0 0.0 0.0 0.0 0.08 −0.1 −0.04 0.24 −0.06 −0.86 0.0 0.0 0.0 0.04 −0.04 −0.04 0.0 0.06 −0.08 −0.5 0.0 0.0

 

 

 

 

−0.04 0.0 0.0 −0.02 0.14 0.02 0.0 −0.74 0.0









0.2174 0.1123 0.1037 0.0724 0.0681 0.0816 0.0938 0.0977 0.0849 0.0681

0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

−1.17 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.35 −0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 −0.11 −0.99 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.03 0.05 −0.07 −0.85 0.0 0.0 0.0 0.0 0.0 0.0

10

−0.03 0.03 0.25 −0.09 −0.93 0.0 0.0 0.0 0.0 0.0 −0.01 −0.03 −0.01 0.21 −0.11 −0.67 0.0 0.0 0.0 0.0

0.01 −0.03 −0.03 0.07 0.09 −0.03 −0.81 0.0 0.0 0.0 0.03 −0.03 −0.03 −0.03 0.05 0.01 −0.11 −0.27 0.0 0.0

 

 

 

 

−0.01 −0.01 −0.01 −0.01 0.03 0.07 −0.01 −0.05 −0.57 0.0

