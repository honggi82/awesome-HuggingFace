# arXiv:2411.00322v1[cs.LG]1Nov2024

## Constant Acceleration Flow

Dogyun Park Korea University gg933@korea.ac.kr

Taehoon Lee Korea University 98hoon@korea.ac.kr

Sojin Lee Korea University sojin_lee@korea.ac.kr

Sihyeon Kim Korea University sh_bs15@korea.ac.kr

Youngjoon Hong∗ KAIST hongyj@kaist.ac.kr

Hyunwoo J. Kim∗ Korea University hyunwoojkim@korea.ac.kr

### Abstract

Rectified flow and reflow procedures have significantly advanced fast generation by progressively straightening ordinary differential equation (ODE) flows. They operate under the assumption that image and noise pairs, known as couplings, can be approximated by straight trajectories with constant velocity. However, we observe that modeling with constant velocity and using reflow procedures have limitations in accurately learning straight trajectories between pairs, resulting in suboptimal performance in few-step generation. To address these limitations, we introduce Constant Acceleration Flow (CAF), a novel framework based on a simple constant acceleration equation. CAF introduces acceleration as an additional learnable variable, allowing for more expressive and accurate estimation of the ODE flow. Moreover, we propose two techniques to further improve estimation accuracy: initial velocity conditioning for the acceleration model and a reflow process for the initial velocity. Our comprehensive studies on toy datasets, CIFAR-10, and ImageNet 64×64 demonstrate that CAF outperforms state-of-the-art baselines for one-step generation. We also show that CAF dramatically improves few-step coupling preservation and inversion over Rectified flow. Code is available at https://github.com/mlvlab/CAF.

### 1 Introduction

Diffusion models [1, 2] learn the probability flow between a target data distribution and a simple Gaussian distribution through an iterative process. Starting from Gaussian noise, they gradually denoise to approximate the target distribution via a series of learned local transformations. Due to their superior generative capabilities compared to other models such as GANs and VAEs, diffusion models have become the go-to choice for high-quality image generation. However, their multi-step generation process entails slow generation and imposes a significant computational burden. To address this issue, two main approaches have been proposed: distillation models [3, 4, 5, 6, 7, 8, 9] and methods that simplify the flow trajectories [10, 11, 12, 13, 14] to achieve fewer-step generation. An example of the latter is rectified flow [10, 11, 13], which focuses on straightening ordinary differential equation (ODE) trajectories. Through repeated applications of the rectification process, called reflow, the trajectories become progressively straighter by addressing the flow crossing problem. Straighter flows reduce discretization errors, enabling fewer steps in the numerical solution and, thus, faster generation.

Rectified flow [10, 13] defines the straight ODE flow over time t with a drift force v, where each sample xt transforms from x0 ∼ π0 to x1 ∼ π1 under a constant velocity v = x1 − x0. It

∗Corresponding authors.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

Sampling trajectory

Sampling trajectory

𝑣

𝑥

𝑥

𝑥 𝑥

𝑣(𝑥 ,𝑡)

𝑎(𝑥 ,𝑡)

𝑥

𝑥

𝒗𝜽 𝒙𝒕𝟏,𝒕

𝒂𝝓 𝒙𝒕𝟏,𝒕,𝒗𝟎𝟏

𝑎(𝑥 ,𝑡)

𝑣(𝑥 ,𝑡)

𝑥

𝑥

𝑥

𝑥

𝑣

𝑥 = 𝑥

𝑥 = 𝑥

Flow crossing

Flow crossing

(a) Rectified Flow

(b) Constant Acceleration Flow

- Figure 1: Initial Velocity Conditioning (IVC). We illustrate the importance of IVC to address the flow crossing problem, which hinders the learning of straight ODE trajectories during training. In

Fig. 1a, Rectified flow suffers from approximation errors at the overlapping point xt (where x1t = x2t), resulting in curved sampling trajectories due to flow crossing. Conversely, Fig. 1b demonstrates that CAF, utilizing IVC, successfully estimates ground-truth trajectories by minimizing the ambiguity at xt.

approximates the underlying velocity v with a neural network vθ. Then, it iteratively applies the reflow process to avoid flow crossing by rewiring the flow and building deterministic data coupling. However, constant velocity modeling may limit the expressiveness needed for approximating complex couplings between π0 and π1. This results in sampling trajectories that fail to converge optimally to the target distribution. Moreover, the interpolation paths after the reflow may still intersect—a phenomenon known as flow crossing—which leads to curved rectified flows because the model estimates different targets for the same input. As illustrated in Fig. 1a, instead of following the intended path from x10 to x11, a sampling trajectory from Rectified flow erroneously diverts towards x21 due to the flow crossing. Such flow crossing can make the accurate learning of straight ODE trajectories more challenging.

In this paper, we introduce the Constant Acceleration Flow (CAF), a novel ODE framework based on a constant acceleration equation, as outlined in (4). Our CAF generalizes Rectified flow by introducing acceleration as an additional learnable variable. This constant acceleration modeling offers the ability to control flow characteristics by manipulating the acceleration magnitude and enables a direct closed-form solution of the ODE, supporting precise and efficient sampling in just a few steps. Additionally, we propose two strategies to address the flow crossing problem. The first one is initial velocity conditioning (IVC) for the acceleration model, and the second one is to employ reflow to enhance the learning of initial velocity. Fig. 1b presents that CAF, with the proposed strategies, can accurately predict the ground-truth path from x10 to x11, even when flow crossing occurs. Through extensive experiments, from toy datasets to real-world image generation on CIFAR-10 [15] and ImageNet 64×64, we demonstrate that our CAF exhibits superior performance over Rectified flow and state-of-the-art baselines. Notably, CAF achieves superior Fréchet Inception Distance (FID) scores on CIFAR-10 and ImageNet 64×64 in conditional settings, recording FIDs of 1.39 and 1.69, respectively, thereby surpassing recent strong methods. Moreover, we show that CAF provides more accurate flow estimation than Rectified flow by assessing the ‘straightness’ and ‘coupling preservation’ of the learned ODE flow. CAF is also capable of few-step inversion, making it effective for real-world applications such as box inpainting.

To summarize, our contributions are as follows:

- • We propose Constant Acceleration Flow (CAF), a novel ODE framework that integrates acceleration as a controllable variable, enhancing the precision of ODE flow estimation compared to the constant velocity framework.
- • We propose two strategies to address the flow crossing problem: initial velocity conditioning for the acceleration model and a reflow procedure to improve initial velocity learning. These strategies ensure a more accurate trajectory estimation even in the presence of flow crossings.
- • Through extensive experiments on synthetic and real datasets, CAF demonstrates remarkable performance, especially achieving the superior FID on CIFAR-10 and ImageNet 64×64 over strong baselines. We also demonstrate that CAF learns more accurate flow than Rectified flow by assessing the straightness, coupling preservation, and inversion.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

𝜋 = 0 𝜋 = 1 Generated

2-Rectified Flow CAF (Ours) ℎ = 0 CAF (Ours) ℎ = 1 CAF (Ours) ℎ = 2

- Figure 2: 2D synthetic dataset. We compare results between 2-Rectified flow and our Constant

Acceleration Flow (CAF) on 2D synthetic data. π0 (blue) and π1 (green) are source and target distributions parameterized by Gaussian mixture models. Here, the number of sampling steps is

N = 1. While 2-Rectified flow frequently generates samples that deviate from π1, CAF more accurately estimates the target distribution π1. The generated samples (orange) from CAF form a more similar distribution as the target distribution π1.

### 2 Related work

Generative models. Learning generative models involves finding a nonlinear transformation between two distributions, typically denoted as π0 and π1, where π0 is a simple distribution like a Gaussian, and π1 is the complex data distribution. Various approaches have been developed to achieve this transformation. For example, variational autoencoders (VAE) [16, 17] optimize the Evidence Lower Bound (ELBO) to learn a nonlinear mapping from the latent space distribution π0 to the data distribution π1. Normalizing flows [18, 19, 20] construct a series of invertible and differentiable mappings to transform π0 into π1. Similarly, GANs [21, 22, 23, 24, 25] earn a generator that transforms

- π0 into π1 through an adversarial process involving a discriminator. These models typically perform a one-step generation from π0 to π1. In contrast, diffusion models [2, 26, 27, 28, 29, 30] propose learning the probability flow between the two distributions through an iterative process. This iterative process ensures stability and precision, as the model incrementally learns to reverse a diffusion process that adds noise to data. Diffusion models have demonstrated superior performance across various domains, including images [12, 31, 32, 33], 3D [34, 35, 36, 37], and video [38, 39, 40].

Few-step diffusion models Addressing the slow generation speed of diffusion models has become a major focus in recent research: Distillation methods [3, 4, 5, 6, 7, 8, 9] seek to optimize the inference steps of pre-trained diffusion models by amortizing the integration of ODE flow. Consistency models [6, 7, 8] train a model to map any point on the pre-trained diffusion trajectory back to the data distribution, enabling fast generation. Rectified flow [10, 11, 13] is another direction, which focuses on straightening ODE trajectories under a constant velocity field. By straightening the flow and reducing path complexity, it allows for fast generation through efficient and accurate numerical solutions with fewer Euler steps. Recent methods such as AGM [41] also introduce acceleration modeling based on Stochastic Optimal Control (SOC) theory instead of relying solely on velocity. However, AGM predicts time-varying acceleration, which still requires multiple iterative steps to solve the differential equations. In contrast, our proposed CAF ODE assumes that the acceleration term is constant with respect to time. Therefore, there is no need to iteratively solve complex timedependent differential equations. This simplification allows for a direct closed-form solution that supports efficient and accurate sampling in just a few steps.

### 3 Preliminary

Rectified flow [10, 13] is an ordinary differential equation-based framework for learning a mapping between two distributions π0 and π1. Typically, in image generation, π0 is a simple tractable distribution, e.g., the standard normal distribution, defined in the latent space and π1 is the image distribution. Given empirical observations of x0 ∼ π0 and x1 ∼ π1 over time t ∈ [0,1], a flow is defined as

dxt dt

= v(xt,t), (1)

[Figure 5]

[Figure 6]

[Figure 7]

𝜋 = 0 𝜋 = 1 Sampling direction

|𝑎 > 0|𝑎 = 0|𝑎 < 0|
|---|---|---|
| |[Figure 8]|[Figure 9]<br><br>[Figure 10]|

ℎ = 0 ℎ = 1 ℎ = 2

- Figure 3: Sampling trajectories of CAF with different h. The sampling trajectories of CAF are displayed for different values of h, which determines the initial velocity and acceleration. π0 and

- π1 are mixtures of Gaussian distributions. We sample across sampling steps of N = 7 to show how sampling trajectories change with h.

where xt = I(x0,x1,t) is a time-differentiable interpolation between x0 and x1, and v : Rd × [0,1] → Rd is a velocity field defined on data-time domain. Rectified flow learns the velocity field v with a neural network vθ by minimizing the following mean square objective:

0,x1∼γ,t∼p(t) ∥v(xt,t) − vθ(xt,t)∥2 , (2)

Ex

min

θ

where γ represents a coupling of (π0,π1) and p(t) is a time distribution defined on [0,1]. The choice of interpolation I leads to various algorithms, such as Rectified flow [10], ADM [30], EDM [29],

and LDM [42]. Specifically, Rectified flow proposes a simple linear interpolation between x0 and x1 as xt = (1 − t)x0 + tx1, which induces the velocity field v in the direction of (x1 − x0), i.e., v(xt,t) = x1 − x0. This means the Rectified flow transports x0 to x1 along a straight trajectory with a constant velocity. After training vθ, we can generate a sample x1 using off-the-shelf ODE solvers Φ, such as the Euler method:

##### xt+∆t = xt + ∆t · vθ(xt,t), t ∈ {0,∆t,...,(N − 1) · ∆t}, (3)

where ∆t = N1 and N is the total number of steps. To achieve faster generation with fewer steps without sacrificing accuracy, it is crucial to learn a straight ODE flow. Straight ODE flow minimize

numerical errors encountered by the ODE solver.

Reflow and flow crossing. The trajectories of interpolants xt may intersect—a phenomenon known as flow crossing—due to stochastic coupling between π0 and π1 (e.g., random pairing of x0 and x1). These intersections introduce approximation errors in the neural network, leading to curved sampling trajectories [10]. Our toy experiment, illustrated in Fig. 1a, clearly demonstrates this issue: the simulated sampling trajectories become curved due to flow crossing, rendering one-step simulation inaccurate. To address this problem, Rectified flow [10] introduces a reflow procedure. This procedure iteratively straightens the trajectories by reconstructing a more deterministic and direct pairing of x0 and x1 without altering the marginal distributions. Specifically, the reflow procedure involves generating a new coupling γ of (x0,x1 = Φ(x0;vθk)) using a pre-trained Rectified flow model vθk, where k denotes the iteration of the reflow procedure, and Φ(x0;vθk) = x0 + 0 1 vθk(xt,t)dt. By iteratively refining the coupling and the velocity field, the reflow procedure reduces flow crossing, resulting in straighter trajectories and improved accuracy in fewer steps.

### 4 Method

We aim to develop a generative model based on the ODE framework that enables faster generation without compromising quality. To achieve this, we propose a novel approach called Constant Acceleration Flow (CAF). Specifically, CAF formulates an ODE trajectory that transports xt with a constant acceleration, offering a more expressive and precise estimation of the ODE flow compared to constant velocity models. Additionally, we propose two novel techniques that address the problem of flow crossing: 1) initial velocity conditioning and 2) reflow procedure for learning initial velocity. The overall training pipeline is presented in Alg. 1.

#### 4.1 Constant Acceleration Flow

We propose a novel ODE framework based on the constant acceleration equation, which is driven by the empirical observations x0 ∼ π0 and x1 ∼ π1 over time t ∈ [0,1] as:

dxt = v(x0,0)dt + a(xt,t)tdt, (4) where v : Rd × [0] → Rd is the initial velocity field and a : Rd × [0,1] → Rd is the acceleration field. We abbreviate time variable t for notation simplicity, i.e., v(x0,0) = v(x0), a(xt,t) = a(xt). By integrating both sides of (4) with respect to t and assuming a constant acceleration field, i.e., a(xt

),∀t1,t2 ∈ [0,1], we derive the following equation:

) = a(xt

1

2

- 1

- 2

a(xt)t2. (5) Given the initial velocity field v, the acceleration field a can be derived as

xt = x0 + v(x0)t +

a(xt) = 2(x1 − x0) − 2v(x0), (6) by setting t = 1 and the constant acceleration assumption. Then, we propose a time-differentiable interpolation I as:

xt = I(x0,x1,t,v(x0)) = (1 − t2)x0 + t2x1 + v(x0)(t − t2), (7)

by substituting (6) to (5). Using this result, we can easily simulate an intermediate sample xt on our CAF ODE trajectory.

Learning initial velocity field. Selecting an appropriate initial velocity field is crucial, as different initial velocities lead to distinct flow dynamics. Here, we define the initial velocity field as a scaled displacement vector between x1 and x0:

v(x0) = h(x1 − x0), (8) where h ∈ R is a hyperparameter that adjusts the scale of the initial velocity. This configuration enables straight ODE trajectories between distributions π0 and π1, similar to those in Rectified flow. However, varying h changes the flow characteristics: 1) h = 1 simulates constant velocity flows, 2) h < 1 leads to a model with a positive acceleration, and 3) h > 1 results in a negative acceleration, as illustrated in Fig. 3. Empirically, we observe that the negative acceleration model is more effective for image sampling, possibly due to its ability to finely tune step sizes near data distribution.

The initial velocity field is learned using a neural network vθ, which is optimized by minimizing the distance metric d(·,·) between the target and estimated velocities as

0,x1∼γ,t∼p(t),xt∼I [d(v(x0),vθ(xt))], (9)

Ex

min

θ

where p(t) is a time distribution defined on [0,1]. Note that our velocity model learns target initial velocity defined at t = 0. This differs from Rectified flow, which learns target velocity field defined over t ∈ [0,1].

Learning acceleration field. Under the assumption of constant acceleration, the acceleration field is derived from (6) as

a(xt) = 2(x1 − x0) − 2v(x0). (10) We learn the acceleration field using a neural network aϕ by minimizing the distance metric d(·,·) as:

0,x1∼γ,t∼p(t),xt∼I [d(a(xt),aϕ(xt))]. (11) In Sec. C, we theoretically show that CAF ODE preserves the marginal data distribution.

Ex

min

ϕ

- Algorithm 1 Training process of Constant Acceleration Flow Require: deterministic coupling γ, initial velocity scale h, vθ,aϕ.

- 1: while not converge do
- 2: x0,x1 ∼ γ, t ∼ Unif([0,1])
- 3: v(x0) = h(x1 − x0) ▷ Target initial velocity
- 4: xt = I(x0,x1,t,v(x0)) ▷ Eq. (7)
- 5: Lvel = d(v(x0),vθ(xt))
- 6: θ ← θ − ∇Lvel ▷ update θ using SGD with gradient
- 7: end while
- 8: while not converge do
- 9: x0,x1 ∼ γ, t ∼ Unif([0,1]),vˆθ = vθ(x0)
- 10: a(xt) = 2(x1 − x0) − 2vˆθ ▷ Target acceleration
- 11: xt = I(x0,x1,t,vˆθ) ▷ Eq. (7)
- 12: Lacc = d(sg[a(xt)],aϕ(xt,vˆθ))
- 13: ϕ ← ϕ − ∇Lacc ▷ update ϕ using SGD with gradient
- 14: end while
- 15: return vθ,aϕ

#### 4.2 Addressing flow crossing

Rectified flow addresses the issue of flow crossing by a reflow procedure. However, even after the procedure, trajectories may still intersect each other. Such intersections hinder learning straight ODE trajectories, as demonstrated in Fig. 1a. Similarly, our acceleration model also encounters the flow crossing problem. This leads to inaccurate estimation, as the model struggles to predict estimation on these intersections correctly. To further address the flow crossing, we propose two techniques.

Initial velocity conditioning (IVC). We propose conditioning the estimated initial velocity vˆθ = v(x0) as the input of the acceleration model, i.e., aϕ(xt,vˆθ). This approach provides the acceleration model with auxiliary information on the flow direction, enhancing its capability to distinguish correct estimations and mitigate ambiguity at the intersections of trajectories, as illustrated in Fig. 1. Our IVC circumvents the non-intersecting condition required in Rectified flow (see Theorem 3.6 in [10]), which is a key assumption for achieving a straight coupling γ. By reducing the ambiguity arising from intersections, CAF can learn straight trajectories with less constrained couplings, which is quantitatively assessed in Tab. 4.

To incorporate IVC into learning the acceleration model, we reformulate (11) as:

0,x1∼γ,t∼p(t),xt∼I [d(sg[a(xt)],aϕ(xt,vˆθ))]. (12)

Ex

min

ϕ

where sg[·] indicates stop-gradient operation. Since our velocity model learns to predict the initial velocity (see (9)), we ensure that the model can handle both forward and reverse CAF ODEs, which start from x0 and x1, respectively. Thus, our acceleration model can generalize across different flow directions, enabling inversion as demonstrated in Sec. B.2.

Reflow for initial velocity. It is also important to improve the accuracy of the initial velocity model. Following [10], we address the inaccuracy caused by stochastic pairing of x0 and x1 by employing a pre-trained generative model ψ, which constructs a more deterministic coupling γ of x0 and x1. We subsequently use this new coupling γ to train the initial velocity and acceleration models.

#### 4.3 Sampling

After training the initial velocity and acceleration models, we generate samples using the CAF ODE introduced in (4). The discrete sampling process is given by:

xt+∆t = xt + ∆t · vθ(x0) + t′ · ∆t · aϕ(xt,t,vθ(x0)), (13)

where N is the total number of steps, ∆t = N1 , t = i·∆t, and t′ = (2i2+1)·∆t where i ∈ {0,...,N−1} (See Alg. 2). We adopt t′ since it empirically improves accuracy, especially in the small N regime.

Notably, when N = 1 (one-step generation), t′ simplifies to 12, leading to the closed-form solution in

(5). See Alg. 3 for inversion algorithm.

- Algorithm 2 Sampling process of Constant Acceleration Flow

- Require: velocity model vθ, acceleration model aϕ, sampling steps N, π0.

- 1: x0 ∼ π0
- 2: vˆθ ← vθ(x0)
- 3: for i = 0 to N − 1 do
- 4: t ← Ni

- 5: t′ ← 22i+1N

- 6: aˆϕ ← aϕ(xt,vθ)
- 7: xt+ 1

N

← xt + N1 vˆθ + t

′

N aˆϕ

- 8: end for
- 9: return x1

### 5 Experiment

We evaluate the proposed Constant Acceleration Flow (CAF) across various scenarios, including both synthetic and real-world datasets. In Sec. 5.1, our investigation begins with a simple twodimensional synthetic dataset, where we compare the performance of Rectified flow and CAF to clearly demonstrate the effectiveness of our model. Next, we extend our experiments to real-world image datasets, specifically CIFAR-10 (32×32) and ImageNet (64×64), in Sec. 5.2. These experiments highlight CAF’s ability to generate high-quality images with a single sampling step. Furthermore, we conduct an in-depth analysis of CAF through evaluations of coupling preservation, straightness, inversion tasks, and an ablation study in Sec. 5.3.

#### 5.1 Synthetic experiments

We demonstrate the advantages of the Constant Acceleration Flow (CAF) over the constant velocity flow model, Rectified Flow [10], through synthetic experiments. For the neural networks, we use multilayer perceptrons (MLPs) with five hidden layers and 128 units per layer. Initially, we train

- 1-Rectified flow on 2D synthetic data to establish a deterministic coupling. We then train both CAF and 2-Rectified flow. For CAF, we incorporate the initial velocity into the acceleration model by concatenating it with the input, ensuring that the model capacities of both CAF and 2-Rectified flow

remain comparable. We set d as l2 distance. Fig. 2 presents samples generated from CAF in one step and from 2-Rectified flow in two steps. Our CAF more accurately approximates the target distribution

π1 than 2-Rectified flow. In particular, CAF with h = 2 (negative acceleration) learns the most accurate distribution. In contrast, 2-Rectified flow frequently generates samples that significantly

deviate from π1, indicating its difficulty in accurately estimating straight ODE trajectories. This experiment shows that reflowing alone may not overcome the flow crossing problem, leading to poor estimations, whereas our proposed acceleration modeling and IVC effectively address this issue. Moreover, Fig. 3 shows sampling trajectories from CAF trained with different hyperparameters h. It clearly demonstrates that h controls the flow dynamics as we intended: h > 1 indicates negative acceleration, h = 1 represents constant velocity, and h < 1 corresponds to positive acceleration flows. Additional synthetic examples are provided in Fig. 6.

#### 5.2 Real-data experiments

To further validate the effectiveness of our approach, we train CAF on real-world image datasets, specifically CIFAR-10 at 32×32 resolution and ImageNet at 64×64 resolution. To create a deterministic coupling γ, we utilize the pre-trained EDM models [29] and adopt the U-Net architecture of ADM [30] for the initial velocity and acceleration models. In the acceleration model, we double the input dimension of first layer to concatenate the initial velocity to the input xt of the acceleration model, which marginally increases the total number of parameters. We set h = 1.5 and d as LPIPS-Huber loss [43] for all real-data experiments.

Baselines and evaluation. We evaluate state-of-the-art diffusion models [1, 2, 7, 28, 29], GANs [22, 23, 24], and few-step generation approaches [6, 7]. We primarily assess the image generation quality of our method using the Fréchet Inception Distance (FID) [50] and Inception Score (IS) [51]. Additionally, we evaluate diversity using the recall metric following [6, 7, 10].

Table 1: Performance on CIFAR-10.

Unconditional Conditional

Model N

FID↓ FID↓ GAN Models

BigGAN [22] 1 8.51 StyleGAN-Ada [23] 1 2.92 2.42 StyleGAN-XL [24] 1 - 1.85 Diffusion/Consistency Models

Score SDE [1] 2000 2.20 DDPM [2] 1000 3.17 VDM [27] 1000 7.41 LSGM [28] 138 2.10 DDIM [26] 10 13.36 -

35 2.01 1.82

EDM [29]

5 37.75 35.54 CT [6]

2 5.83 -

- 1 8.70 -

Diffusion/Consistency Models – Distillation

Diff-Instruct [9] 1 4.53 DMD [44] 1 3.77 DFNO [5] 1 3.78 TRACT [45] 1 3.78 KD [46] 1 9.36 -

CD [6]

- 2 2.93 -

- 1 3.55 -

CTM [7]

- 2 1.87 1.63

- 1 1.98 1.73

Rectified Flow Models

2-Rectified Flow [10]

- 2 7.89 3.74 1 11.81 6.88

2-Rectified Flow + Distill [10] 1 4.84 -

CAF (Ours) 1 4.81 2.68 CAF + GAN (Ours) 1 1.48 1.39

Table 2: Performance on ImageNet 64 × 64.

Model N FID↓ IS↑ Rec↑ GAN Models

BigGAN-deep [22] 1 4.06 - 0.48 StyleGAN-XL [24] 1 2.09 82.35 0.52 Diffusion/Consistency Models

50 13.7 - 0.56 10 18.3 - 0.49

DDIM [26]

DDPM [2] 250 11.0 - 0.58 iDDPM [47] 250 2.92 - 0.62 ADM [30] 250 2.07 - 0.63

79 2.44 48.88 0.67

EDM [29]

5 55.3 - DPM-solver [48]

20 3.42 - 10 7.93 - -

20 3.10 - 10 6.65 - -

DEIS [49]

2 11.1 - 0.56

CT [6]

- 1 13.0 - 0.47

Diffusion/Consistency Models – Distillation

Diff-Instruct [9] 1 5.57 - DMD [44] 1 2.62 - TRACT [45] 1 7.43 - DFNO [5] 1 7.83 - 0.61 PD [3] 1 15.39 - 0.62

CD [6]

- 2 4.70 - 0.64

- 1 6.20 40.08 0.57

CTM [7]

- 2 1.73 64.29 0.57 1 1.92 70.38 0.57

Rectified Flow Models

CAF (Ours) 1 6.52 37.45 0.62 CAF + GAN (Ours) 1 1.69 62.03 0.64

Distillation. Distilling a few-step student model from a pre-trained teacher model has recently become essential for high-quality few-step generation [6, 7, 10, 11]. InstaFlow [11] has observed that learning straighter trajectories and achieving good coupling significantly enhance distillation performance. Moreover, CTM [7] and DMD [44] incorporate an adversarial loss as an auxiliary loss to facilitate the training of the student model. We empirically found that incorporating the adversarial loss alone was sufficient to achieve superior performance for one-step sampling without introducing instability. For training details, please refer to Sec. A.

CIFAR-10. We present the experimental results on CIFAR-10 in Tab. 1. Our base unconditional CAF model (4.81 FID, N = 1) significantly improves the FID compared to recent state-of-the-art diffusion models (without distillation), including DDIM [26] (13.36 FID, N = 10), EDM (37.75 FID, N = 5), and 2-Rectified flow (7.89 FID, N = 2) in a few-step generation (e.g., N < 10). We retrained 2-Rectified flow using the official codes of [10], achieving a slightly better performance than the officially reported performance (12.21 FID) for one-step generation [10]. CAF’s remarkable 3.08 FID improvement over 2-Rectified flow (N = 2) highlights the effectiveness of acceleration modeling in a fast generation. Our approach is also effective in class-conditional generation, where the base CAF model (2.68 FID, N = 1) shows a significant FID improvement over EDM (35.54 FID, N = 5) and 2-Rectified flow (3.74 FID, N = 2). Additionally, after adversarial training, CAF achieves a superior FID of 1.48 for unconditional generation and 1.39 for conditional generation with N = 1. Lastly, we qualitatively compare the 2-Rectified flow and our CAF in Fig. 4, where CAF generates more vivid samples with intricate details than 2-Rectified flow.

ImageNet. We extend our evaluation to the ImageNet dataset at 64×64 resolution to demonstrate the scalability and effectiveness of our CAF model on more complex and higher-resolution images. Similar to the results on CIFAR-10, our base conditional CAF model significantly improves the FID compared to recent state-of-the-art diffusion models (without distillation) in the small N regime (e.g., N < 10). Specifically, CAF (6.52 FID, N = 1) outperforms models such as DPM-solver [48] (7.93 FID, N = 10), CT [6] (11.1 FID, N = 2), and EDM [29] (55.3 FID, N = 5). This validates that the superior performance of CAF can be effectively generalized to complex and large-scale datasets. Additionally, after adversarial training, CAF outperforms or is competitive with state-of-the-art distillation baselines in one-step generation. Notably, CAF achieves the best FID performance of 1.69, surpassing strong baselines. We also demonstrate one-step qualitative results in Fig. 14.

CAF (Ours) 2-RF

2-RF CAF (Ours)

|[Figure 11]|[Figure 12]|[Figure 13]|[Figure 14]|
|---|---|---|---|
|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|
|[Figure 19]|[Figure 20]|[Figure 21]|[Figure 22]|
|[Figure 23]|[Figure 24]|[Figure 25]|[Figure 26]|

|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|---|---|---|---|
|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|
|[Figure 35]|[Figure 36]|[Figure 37]|[Figure 38]|
|[Figure 39]|[Figure 40]|[Figure 41]|[Figure 42]|

1 step 10 steps

(a)

2-RF + Distlled

CAF (Ours) + Distilled

2-RF + Distlled

CAF (Ours) + Distilled

|[Figure 43]|[Figure 44]|[Figure 45]|[Figure 46]|
|---|---|---|---|
|[Figure 47]|[Figure 48]|[Figure 49]|[Figure 50]|
|[Figure 51]|[Figure 52]|[Figure 53]|[Figure 54]|
|[Figure 55]|[Figure 56]|[Figure 57]|[Figure 58]|

|[Figure 59]|[Figure 60]|[Figure 61]|[Figure 62]|
|---|---|---|---|
|[Figure 63]|[Figure 64]|[Figure 65]|[Figure 66]|
|[Figure 67]|[Figure 68]|[Figure 69]|[Figure 70]|
|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|

1 step

10 steps

(b)

- Figure 4: Qualitative results on CIFAR-10. We compare the quality of generated images from

- 2-Rectified flow and CAF (Ours) with N = 1 and 10. Each image x1 is generated from the same x0 for both models. CAF generates more vivid images with intricate details than 2-RF for both N.

Table 3: Coupling preservation.

Metric 2-Rectified Flow CAF (ours)

LPIPS ↓ 0.092 0.041 PSNR ↑ 29.79 33.16

Table 4: Flow straightness comparison.

Dataset 2-Rectified Flow CAF (ours)

2D 0.065 0.058 CIFAR-10 0.043 0.034

Table 5: Ablation study on CIFAR-10 (N = 1).

Constant

v0 Reflow

Config

FID

| |acceleration condition procedure|↓<br><br>|
|---|---|---|
|A<br>B<br>|✗ ✗ ✗<br>✗ ✗ ✔<br>|378 6.88<br><br>|
|C<br><br>D<br>E<br>F<br>|✔(h=1.5) ✗ ✔<br>✔(h=1.5) ✔ ✔<br>✔(h=1) ✔ ✔<br>✔(h=0.5) ✔ ✔<br>|3.82 2.68 3.02 2.73|

#### 5.3 Analysis

Coupling preservation. We evaluate how accurately CAF and Rectified flow approximate the deterministic coupling obtained from pre-trained models via a reflow procedure. To analyze this, we first conduct synthetic experiments where the interpolant paths I are crossed, as illustrated in Fig. 5a. Due to the flow crossing, the sampling trajectory of Rectified flow fails to preserve the ground-truth coupling (interpolation path I), leading to a curved sampling trajectory. In contrast, our CAF learns the straight interpolation paths by incorporating acceleration, demonstrating superior coupling preservation ability.

Moreover, we evaluate the coupling preservation ability on real data from CIFAR-10. We randomly sample 1K training pairs (x0,x1) from the deterministic coupling γ and measure the similarity between x1 and xˆ1, where xˆ1 is a generated sample from x0. In other words, we measure the distance between a ground truth image and a generated image corresponding to the same noise. If the coupling is well-preserved, the distance should be small. We use PSNR and LPIPS [52] as distance measures. The result in Tab. 3 demonstrates that CAF better preserves coupling. In terms of PSNR, CAF outperforms Rectified flow by 3.37. This is consistent with the qualitative result in Fig. 5b, where xˆ1 from CAF resembles more to x1 (ground truth) than xˆ1 from Rectified flow.

Flow straightness. To evaluate the straightness of learned trajectories, we introduce the Normalized Flow Straightness Score (NFSS). Similar to previous works [10, 11], we measure flow straightness S by the L2distance between the normalized displacement vector (x0 −x1) and the normalized velocity vector x˙t as below:

2

x1 − x0 ∥x1 − x0∥2

x˙t ∥x˙t∥2

. (14)

S = Ex

−

0,x1,t

2

Here, a smaller value of S indicates a straighter trajectory. We compare S between CAF and Rectified flow using synthetic and real-world datasets, as presented in Tab. 4. For Rectified flow, we use x˙t = vθ(xt), while for CAF, we use x˙t = vθ(x0) + aϕ(xt)t. The results show that CAF outperforms Rectified flow in flow straightness.

𝜋 = 0 𝜋 = 1 Interpolation path Sampling trajectory

|[Figure 75]<br><br>[Figure 76]| |
|---|---|
|[Figure 77]<br><br>[Figure 78]| |

|[Figure 79]|[Figure 80]|[Figure 81]|
|---|---|---|

|[Figure 82]|[Figure 83]|[Figure 84]|
|---|---|---|

CAF(Ours)RF

2-RF (0.250)

CAF

2-RF (0.181)

CAF

GT (0.081)

GT (0.048)

|[Figure 85]|[Figure 86]|[Figure 87]|
|---|---|---|

|[Figure 88]|[Figure 89]<br><br>[Figure 90]|[Figure 91]|
|---|---|---|

GT (0.057) GT (0.302)2-RF (0.023)CAF

2-RF (0.328)

CAF

|[Figure 92]|[Figure 93]|[Figure 94]|
|---|---|---|

|[Figure 95]|[Figure 96]|[Figure 97]|
|---|---|---|

GT (0.048) GT (0.265)2-RF (0.050)CAF

2-RF (0.343)

CAF

Iterations

(b)

(a)

- Figure 5: Experiments for coupling preservation. (a) We plot the sampling trajectories during training where their interpolation paths I are crossed. Due to the flow crossing, RF (top) rewires the coupling, whereas CAF (bottom) preserves the coupling of training data. (b) CAF accurately generates target images from the given noise (e.g., a car from the car noise), while RF often fails (e.g., a frog from the car noise). LPIPS [52] values are in parentheses.

Inversion We further demonstrate CAF’s capability in real-world applications by conducting zeroshot tasks such as reconstruction and box inpainting using inversion. We provide implemenetation details and algorithms in Sec. B.2. As shown in the Tab. 6 and 7, our method achieves lower reconstruction errors (CAF: 46.68 PSNR vs. RF: 33.34 PSNR) and better zero-shot inpainting capabilities even with fewer steps compared to baselines. These improvements are attributed to CAF’s superior coupling preservation capability. Moreover, we present qualitative comparisons between CAF and the baselines in Fig. 12 and 13, which further validates the quantitative results.

Ablation study. We conduct an ablation study to evaluate the effectiveness of components in our framework under the one-step generation setting (N = 1). We examine the improvements achieved by 1) constant acceleration modeling, 2) initial velocity (v0) conditioning, and 3) the reflow procedure for v0. The configurations and results are outlined in Tab. 5. Specifically, A and B correspond to 1-Rectified flow and 2-Rectified flow, respectively. Configurations C to F represent our CAF frameworks, with C being our CAF without IVC. By comparing A,B,C, and F, we demonstrate that all three components in our framework substantially improve the performance. In addition, we analyze the final model across various acceleration scales controlled by h. The performance difference between D and F is relatively small, indicating that our framework is robust to the choice of hyperparameters. Empirically, we observe that configuration F, i.e., CAF (h = 1.5) with negative acceleration, achieves the best FID of 2.68. Notably, our CAF without v0 conditioning, still outperforms rectified flow (configuration B) by 3.06 FID. This highlights the critical role of constant acceleration modeling in enhancing the quality of few-step generation. Also, we verify the significance of reflowing by comparing configurations A and B, which achieve 378 FID and 6.88 FID, respectively.

- 6 Conclusion

In this paper, we have introduced the Constant Acceleration Flow (CAF) framework, which enhances precise ODE trajectory estimation by incorporating a controllable acceleration variable into the ODE framework. To address the flow crossing problem, we proposed two strategies: initial velocity conditioning and a reflow procedure. Our experiments on toy datasets, real-world dataset demonstrate CAF’s capabilities and scalability, achieving state-of-the-art FID scores. Furthermore, we conducted extensive ablation studies and analyses—including assessments of flow straightness, coupling preservation, and real-world applications—to validate and deepen our understanding of the effectiveness of our proposed components in learning accurate ODE trajectories. We believe that CAF offers a promising direction for efficient and accurate generative modeling, and we look forward to exploring its applications in more diverse settings such as 3D and video.

### Acknowledgement

This work was supported by ICT Creative Consilience Program through the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (IITP-2024-RS-2020-II201819, 10%), the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (NRF-2023R1A2C2005373, 45%), and the Virtual Engineering Platform Project (Grant No. P0022336, 45%), funded by the Ministry of Trade, Industry & Energy (MoTIE, South Korea).

### References

- [1] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, ICLR, 2021.
- [2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, NeurIPS, 2020.
- [3] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, ICLR, 2022.
- [4] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Conference on Computer Vision and Pattern Recognition, CVPR, 2023.
- [5] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In International Conference on Machine Learning, ICML, 2023.
- [6] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, ICML, 2023.
- [7] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. In International Conference on Learning Representations, ICLR, 2024.
- [8] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [9] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. In Advances in Neural Information Processing Systems, NeurIPS, 2024.
- [10] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, ICLR, 2023.
- [11] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In International Conference on Learning Representations, ICLR, 2023.
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.
- [13] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.
- [14] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, ICLR, 2022.
- [15] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
- [16] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, ICLR, 2014.
- [17] Aaron Van Den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, NeurIPS, 2017.

- [18] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. In International Conference on Learning Representations, ICLR, 2017.
- [19] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In Advances in Neural Information Processing Systems, NeurIPS, 2018.
- [20] Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. Ot-flow: Fast and accurate continuous normalizing flows via optimal transport. In Association for the Advancement of Artificial Intelligence, AAAI, 2021.
- [21] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, NeurIPS, 2014.
- [22] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations, ICLR, 2018.
- [23] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In Advances in Neural Information Processing Systems, NeurIPS, 2020.
- [24] Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. In SIGGRAPH, 2022.
- [25] Yujin Kim, Dogyun Park, Dohee Kim, and Suhyun Kim. Naturalinversion: Data-free image synthesis improving real-world consistency. In Association for the Advancement of Artificial Intelligence, AAAI, 2022.
- [26] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, ICLR, 2020.
- [27] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Advances in Neural Information Processing Systems, NeurIPS, 2021.
- [28] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In Advances in Neural Information Processing Systems, NeurIPS, 2021.
- [29] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems, NeurIPS, 2022.
- [30] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, NeurIPS, 2021.
- [31] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023.
- [32] Sojin Lee, Dogyun Park, Inho Kong, and Hyunwoo J Kim. Diffusion prior-based amortized variational inference for noisy inverse problems. In European Conference on Computer Vision, ECCV, 2024.
- [33] Juyeon Ko, Inho Kong, Dogyun Park, and Hyunwoo J Kim. Stochastic conditional diffusion models for robust semantic image synthesis. In International Conference on Machine Learning, ICML, 2024.
- [34] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In International Conference on Computer Vision, ICCV,

- 2023.

[35] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In International Conference on Learning Representations, ICLR,

- 2024.

- [36] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024.
- [37] Dogyun Park, Sihyeon Kim, Sojin Lee, and Hyunwoo J Kim. Ddmi: Domain-agnostic latent diffusion models for synthesizing high-quality implicit neural representations. In International Conference on Learning Representations, ICLR, 2024.

- [38] RunwayML Team. Runwayml - gen2. 2023.
- [39] Pika Art. Pika art – home. 2023.
- [40] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [41] Tianrong Chen, Jiatao Gu, Laurent Dinh, Evangelos A Theodorou, Joshua Susskind, and Shuangfei Zhai. Generative modeling with phase stochastic bridges. In International Conference on Learning Representations, ICLR, 2024.
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition, CVPR, 2022.
- [43] Sangyun Lee, Zinan Lin, and Giulia Fanti. Improving the training of rectified flows. In arXiv preprint arXiv:2405.20320, 2024.
- [44] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Conference on Computer Vision and Pattern Recognition, CVPR, 2024.
- [45] David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. In arXiv preprint arXiv:2303.04248, 2023.
- [46] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. In arXiv preprint arXiv:2101.02388, 2021.
- [47] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, ICML, 2021.
- [48] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, NeurIPS, 2022.
- [49] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In arXiv preprint arXiv:2204.13902, 2022.
- [50] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, NeurIPS, 2017.
- [51] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, NeurIPS, 2016.
- [52] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Conference on Computer Vision and Pattern Recognition, CVPR, 2018.
- [53] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, ICLR, 2019.
- [54] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. In International Conference on Learning Representations, ICLR, 2022.
- [55] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, ICML, 2023.
- [56] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, ICML, 2023.
- [57] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Conference on Computer Vision and Pattern Recognition, CVPR, 2023.
- [58] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Conference on Computer Vision and Pattern Recognition, 2024.

### A Implementation details

We utilize the pre-trained EDM model [29] to build the deterministic coupling γ for training our models. To construct deterministic couplings for CIFAR-10 and ImageNet, we select N = 18 and N = 40, respectively, using deterministic sampling following the protocol in [29]. For CIFAR-10 and ImageNet, we generate 1M and 3M pairs, respectively. We use the batch size of 2048 and train for 700K/700K iterations on ImageNet. For CIFAR-10, we use the batch size of 512 and train for 500K/500K iterations. For all experiments, we use AdamW [53] optimizer with a learning rate of 0.0001 and apply an Exponential Moving Average (EMA) with a 0.999 decay rate. For training acceleration model, we initialize it with initial velocity model for faster convergence.

For adversarial training, we employ adversarial loss Lgan using real data x1,real from [24]:

[log(1 − dη(xˆ1))], (15)

Lgan,η(ϕ) = Ex

[log dη(x1,real)] + Ex

1,real

0

where dη is a discriminator and xˆ1 = x0 + vθ(x0) + 12aϕ(x0,vθ(x0)). In the end, we use the following combined loss to update the acceleration model:

L(ϕ,η) = Lacc(ϕ) + λganLgan(ϕ,η), (16) where Lacc corresponds to (12) and λ are weight hyperparameters. Following [42, 54], we employ adaptive weighting as λgan = ∥∇ϕ

lLacc(ϕ)∥

∥∇ϕlLgan(ϕ,η)∥, where ϕl is the last layer of the acceleration model. Without Lacc, we found the training unstable and frequently exhibit mode collapse issue, which is a common problem with adversarial training. We follow the training configuration from StyleGANXL [24]. We bilinearly upscale the image to 224×224 resolution and use EfficientNet [55] and DeiTbase [56] for extracting features. During the adversarial training, we only optimize the acceleration model and discriminator with the learning rate of 2e-5 and 1e-3, respectively. We keep the parameters of the initial velocity model fixed for stable training. The total training takes about 21 days with 8 NVIDIA A100 GPUs for ImageNet, and takes 10 days 8 NVIDIA RTX3090 GPUs for CIFAR-10.

### B Additional results

#### B.1 Additional qualitative results

2D toy dataset. In Fig. 6, we provide additional generation results and sampling trajectories on various 2D synthetic datasets with N = 1, demonstrating the effectiveness of our approach for fast generation. Fig. 7 provides additional examples of coupling preservation on 2-RF and CAF.

Real-world dataset. In Fig. 8 and 9, we show additional generation results from our base CAF model on CIFAR-10 with N = 1,10, and 50. In Fig. 10, we compare the generation result between 2-RF and CAF distilled versions. Fig. 11 shows sampling results from our base CAF models with different hyperparameters h. Lastly, Fig. 14 shows the generation results on ImageNet with N = 1.

#### B.2 Real-world applications

Inversion techniques are essential for real-world applications such as image and video editing [57, 58]. However, existing methods typically require 25–100 steps for accurate inversion, which can be computationally intensive. In contrast, our method significantly reduces the inference time by enabling inversion in just a few steps (e.g., N < 20). We demonstrate this efficiency in two tasks: reconstruction and box inpainting.

To reconstruct x1, we first invert x1 to obtain xˆ0, as described in Alg. 3. We then use the generation process (Alg. 2) with xˆ0 and same initial velocity vθ(x1) used in Alg. 3 to generate xˆ1. For box inpainting, we inject conditional information—the non-masked image region—into the iterative inversion and generation procedures, as detailed in Alg. 4. As demonstrated in Tab. 6 and 7, our method achieves better reconstruction quality (CAF: 46.68 PSNR vs. RF: 33.34 PSNR) and zeroshot inpainting capability even with fewer steps compared to baseline methods. Qualitative results are presented in Fig. 12 and 13, which further illustrate the effectiveness of our approach. This demonstrate that our method can be efficiently used for real-world applications, offering both speed and accuracy advantages over existing techniques.

- Algorithm 3 Inversion process of Constant Acceleration Flow

- Require: velocity model vθ, acceleration model aϕ, sampling steps N, π1. 1: x1 ∼ π1 2: vˆθ ← vθ(x1) 3: for i = N to 1 do 4: t ← Ni

5: t′ ← 22i−N1 6: aˆϕ ← aϕ(xt,vˆθ)

7: xt− 1

N

← xt − N1 vˆθ − t

′

N aˆϕ

8: end for 9: return x0

- Algorithm 4 Box inpainting of Constant Acceleration Flow

Require: velocity model vθ, acceleration model aϕ, sampling steps N, reference image x¯1, binary

image mask Ω where 1 indicates the missing pixels.

- 1: σ ∼ N(0,I)
- 2: x¯ ← x¯1 ⊙ (1 − Ω) + σ ⊙ Ω ▷ Create image with missing pixels and add noise σ
- 3: vˆθ ← vθ(x¯)
- 4: for i = N to 1 do ▷ Inversion steps
- 5: t ← Ni , t′ ← 22i−N1

- 6: aˆϕ ← aϕ(xt,vˆθ)
- 7: xt− 1

N

← xt − N1 vˆθ − t

′

N aˆϕ

- 8: xt− 1

N

← xt− 1

N

⊙ (1 − Ω) + (1 − t)σ ⊙ Ω, σ ∼ N(0,I)

- 9: end for
- 10: vˆθ ← vθ(x0)
- 11: for j = 0 to N − 1 do ▷ Generation steps
- 12: t ← Nj , t′ ← 22jN+1

- 13: aˆϕ ← aϕ(xt,vˆθ)
- 14: xt+ 1

N

← xt + N1 vˆθ + t

′

N aˆϕ

- 15: xt+ 1

N

← x¯1 ⊙ (1 − Ω) + xt+ 1

N

⊙ Ω

- 16: end for
- 17: return inpainted image x1

#### B.3 Comparison with previous acceleration modeling literatures

Here, we elaborate on the crucial differences between AGM [41] and CAF. The main distinction is that CAF assumes constant acceleration, whereas AGM predicts time-dependent acceleration. Since the CAF ODE assumes that the acceleration term is constant with time, there is no need to solve time-dependent differential equations iteratively. This allows for a closed-form solution that supports efficient and accurate sampling, given that the learned velocity and acceleration models are accurate. Specifically, the solution for CAF ODE is given by:

1

1

a(xt) · tdt (17)

v(x0) + a(xt) · tdt = x0 + v(x0) +

x1 = x0 +

0

0

1

- 1

- 2

a(xt) (18)

= x0 + v(x0) + a(xt)

tdt = x0 + v(x0) +

0

The integral simplifies thanks to the constant acceleration assumption, leading to one-step sampling. In contrast, AGM’s acceleration is time-varying, meaning that the differential equation cannot be reduced in an analytic form. It requires multiple steps to approximate the true solution accurately. In Tab. 8, we systemically compare AGM with our CAF, where CAF consistently outperforms AGM. Moreover, we conducted additional experiments where AGM was trained with deterministic couplings as in our reflow setting. Incorporating reflow into AGM did not improve its performance in the few-step regime, which further highlights the distinct advantage of CAF over AGM.

Table 6: Reconstruction error.

Model N PSNR ↑ LPIPS ↓

CM - N/A N/A CTM - N/A N/A EDM 4 13.85 0.447 2-RF 2 33.34 0.094 2-RF 1 29.33 0.204

CAF (Ours) 1 46.68 0.007 CAF (+GAN) (Ours) 1 40.84 0.028

Table 7: Box inpainting.

Model NFE FID ↓ CM 18 13.16 CTM - N/A EDM - N/A 2-RF 12 16.41 CAF (Ours) 12 10.39 CAF (+GAN) (Ours) 12 10.91

Table 8: Comparison between AGM and CAF.

Model Acceleration Closed-form solution Reflow for velocity FID on CIFAR-10 ↓

AGM [41] Time-varying No No 11.88 (N = 5) AGM (enhanced ver.) Time-varying No Yes 15.23 (N = 5)

CAF (Ours) Constant Yes Yes 4.81 (N = 1)

### C Marginal preserving property of Constant Acceleration Flow

We demonstrate that the flow generated by our Constant Acceleration Flow (CAF) ordinary differential equation (ODE) maintains the marginal of the data distribution, as established by the definitions and theorem in [10].

- Definition C.1. For a path-wise continuously differentiable process x = {xt : t ∈ [0,1]}, we define its expected velocity vx and acceleration ax as follow:

vx(x,t) = E

dxt dt | xt = x , ax(x,t) = E

d2xt dt2 | xt = x , ∀x ∈ supp(xt). (19)

For x ∈/ supp(xt), the conditional expectation is not defined and we set vx and ax arbitrarily, for example vx(x,t) = 0 and ax(x,t) = 0.

- Definition C.2. [10] We denote that x is rectifiable if vx is locally bounded and the solution to the integral equation of the form

t

vx(zt,t)dt, ∀t ∈ [0,1], z0 = x0, (20)

zt = z0 +

0

exists and is unique. In this case, z = {zt : t ∈ [0,1]} is called the rectified flow induced by x. Theorem 1. [10] Assume x is rectifiable and z is its rectified flow. Then Law(zt) = Law(xt),∀t ∈ [0,1].

Refer to [10] for the proof of Theorem 1.

We will now show that our CAF ODE satisfies Theorem 1 by proving that our proposed ODE (4) induces z, which is the rectified flow as defined in Definition C.2. In (4), we define the CAF ODE as

d2xt

dxt dt

dxt dt t=0

dt2 · t. (21) By taking the conditional expectation on both sides, we obtain

=

+

vx(x,t) = vx(x,0) + ax(x,t) · t, (22)

from Definition C.1. Then, the solution of the integral equation of CAF ODE is identical to the solution in Definition C.2 by (22):

t

vx(z0,0) + ax(zt,t) · tdt (23)

zt = z0 +

0

t

vx(zt,t)dt. (24)

= z0 +

0

This indicates that z induced by CAF ODE is also a rectified flow. Therefore, the CAF ODE satisfies the marginal preserving property, i.e., Law(zt) = Law(xt), as stated in Theorem 1.

### D Limitation and Broader impacts

#### D.1 Limitations

One limitation of our model is the increased number of function evaluations (NFE) required for N-step generation. While Rectified flow achieves an NFE of N by only computing the velocity at each step, our method necessitates an additional computation, resulting in a total NFE of N + 1. This is because we compute the initial velocity at the beginning and the acceleration at each step. Although this extra evaluation slightly increases the computational burden, it is relatively minor in terms of overall performance and still enables efficient few-step generation. Moreover, this additional step can be reduced by jointly predicting velocity and acceleration terms with a single model, which we leave for future work. Another limitation is the additional effort required to generate supplementary data. We utilize generated data to create a deterministic coupling of noise and data samples for training CAF. While generating more data enhances our model’s performance, it can increase GPU usage, leading to higher carbon emissions.

#### D.2 Broader Impacts

Recent advancements in generative models hold significant potential for societal benefits across a wide array of applications, such as image and video generation and editing, medical imaging analysis, molecular design, and audio synthesis. Our CAF framework contributes to enhancing the efficiency and performance of existing diffusion models, offering promising directions for positive impacts across multiple domains. This suggests that in practical applications, users can utilize generative models more rapidly and accurately, enabling a broad spectrum of activities. However, it is crucial to acknowledge potential risks that must be carefully managed. The increased accessibility of generative models also broadens the potential for misuse. As these technologies become more widespread, the possibility of their exploitation for fraudulent activities, privacy breaches, and criminal behavior increases. It is vital to ensure their ethical and responsible use to prevent negative impacts. Establishing regulated ethical standards for developing and deploying generative AI technologies is necessary to prevent such misuse. Additionally, imposing restricted access protocols or verification systems to trace and authenticate generated contents will help ensure their responsible use.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

𝜋 = 0 𝜋 = 1 Generated

2-Rectified Flow CAF (Ours) ℎ = 0 CAF (Ours) ℎ = 1 CAF (Ours) ℎ = 2

(a) Generation results

[Figure 102]

[Figure 103]

[Figure 104]

𝜋 = 0 𝜋 = 1 Sampling direction

ℎ = 0 ℎ = 1 ℎ = 2

(b) Sampling trajectories with different h

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

𝜋 = 0 𝜋 = 1 Generated

2-Rectified Flow CAF (Ours) ℎ = 0 CAF (Ours) ℎ = 1 CAF (Ours) ℎ = 2

(c) Generation results

[Figure 109]

[Figure 110]

[Figure 111]

𝜋 = 0 𝜋 = 1 Sampling direction

ℎ = 0 ℎ = 1 ℎ = 2

(d) Sampling trajectories with different h

- Figure 6: Experiments on various 2D synthetic dataset. We compare results between 2-Rectified

Flow and our Constant Acceleration Flow (CAF) on 2D synthetic data. π0 (blue) and π1 (green) are source and target distributions parameterized by Gaussian mixture models. The generated samples

(orange) from CAF form a more similar distribution as the target distribution π1.

|[Figure 112]|[Figure 113]|[Figure 114]|[Figure 115]|[Figure 116]|[Figure 117]|[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|[Figure 129]|[Figure 130]|[Figure 131]|
|[Figure 132]|[Figure 133]|[Figure 134]|[Figure 135]|[Figure 136]|[Figure 137]|[Figure 138]|[Figure 139]|[Figure 140]|[Figure 141]|

GT

CAF (Ours)

2-RF

|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|[Figure 146]|[Figure 147]|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 152]|[Figure 153]|[Figure 154]|[Figure 155]|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|[Figure 160]|[Figure 161]|
|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|[Figure 166]|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|

GT

CAF (Ours)

2-RF

|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]|[Figure 177]|[Figure 178]|[Figure 179]|[Figure 180]|[Figure 181]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 182]|[Figure 183]|[Figure 184]|[Figure 185]|[Figure 186]|[Figure 187]|[Figure 188]|[Figure 189]|[Figure 190]|[Figure 191]|
|[Figure 192]|[Figure 193]|[Figure 194]|[Figure 195]|[Figure 196]|[Figure 197]|[Figure 198]|[Figure 199]|[Figure 200]|[Figure 201]|

GT

CAF (Ours)

2-RF

|[Figure 202]|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 212]|[Figure 213]|[Figure 214]|[Figure 215]|[Figure 216]|[Figure 217]|[Figure 218]|[Figure 219]|[Figure 220]|[Figure 221]|
|[Figure 222]|[Figure 223]|[Figure 224]|[Figure 225]|[Figure 226]|[Figure 227]|[Figure 228]|[Figure 229]|[Figure 230]|[Figure 231]|

GT

CAF (Ours)

2-RF

- Figure 7: Additional visualizations of coupling preservation on CIFAR-10. CAF accurately generates target images (x1) from the given noise (x0), while Rectified Flow often fails to preserve coupling of x0 and x1 .

|[Figure 232]|[Figure 233]|[Figure 234]|[Figure 235]|[Figure 236]|[Figure 237]|[Figure 238]|[Figure 239]|[Figure 240]|[Figure 241]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 242]|[Figure 243]|[Figure 244]|[Figure 245]|[Figure 246]|[Figure 247]|[Figure 248]|[Figure 249]|[Figure 250]|[Figure 251]|
|[Figure 252]|[Figure 253]|[Figure 254]|[Figure 255]|[Figure 256]|[Figure 257]|[Figure 258]|[Figure 259]|[Figure 260]|[Figure 261]|

1 step

10 steps

50 steps

|[Figure 262]|[Figure 263]|[Figure 264]|[Figure 265]|[Figure 266]|[Figure 267]|[Figure 268]|[Figure 269]|[Figure 270]|[Figure 271]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 272]|[Figure 273]|[Figure 274]|[Figure 275]|[Figure 276]|[Figure 277]|[Figure 278]|[Figure 279]|[Figure 280]|[Figure 281]|
|[Figure 282]|[Figure 283]|[Figure 284]|[Figure 285]|[Figure 286]|[Figure 287]|[Figure 288]|[Figure 289]|[Figure 290]|[Figure 291]|

1 step

10 steps

50 steps

|[Figure 292]|[Figure 293]|[Figure 294]|[Figure 295]|[Figure 296]|[Figure 297]|[Figure 298]|[Figure 299]|[Figure 300]|[Figure 301]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 302]|[Figure 303]|[Figure 304]|[Figure 305]|[Figure 306]|[Figure 307]|[Figure 308]|[Figure 309]|[Figure 310]|[Figure 311]|
|[Figure 312]|[Figure 313]|[Figure 314]|[Figure 315]|[Figure 316]|[Figure 317]|[Figure 318]|[Figure 319]|[Figure 320]|[Figure 321]|

1 step

10 steps

50 steps

|[Figure 322]|[Figure 323]|[Figure 324]|[Figure 325]|[Figure 326]|[Figure 327]|[Figure 328]|[Figure 329]|[Figure 330]|[Figure 331]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 332]|[Figure 333]|[Figure 334]|[Figure 335]|[Figure 336]|[Figure 337]|[Figure 338]|[Figure 339]|[Figure 340]|[Figure 341]|
|[Figure 342]|[Figure 343]|[Figure 344]|[Figure 345]|[Figure 346]|[Figure 347]|[Figure 348]|[Figure 349]|[Figure 350]|[Figure 351]|

1 step

10 steps

50 steps

###### Figure 8: Qualitative results on unconditional generation (CIFAR-10). We illustrate generating images with varying sampling steps, demonstrating consistency quality even for a one-step generation.

airplane automobile bird cat deer dog frog horse ship truck

|[Figure 352]|[Figure 353]|[Figure 354]|[Figure 355]|[Figure 356]|[Figure 357]|[Figure 358]|[Figure 359]|[Figure 360]|[Figure 361]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 362]|[Figure 363]|[Figure 364]|[Figure 365]|[Figure 366]|[Figure 367]|[Figure 368]|[Figure 369]|[Figure 370]|[Figure 371]|
|[Figure 372]|[Figure 373]|[Figure 374]|[Figure 375]|[Figure 376]|[Figure 377]|[Figure 378]|[Figure 379]|[Figure 380]|[Figure 381]|

1 step

10 steps

50 steps

|[Figure 382]|[Figure 383]|[Figure 384]|[Figure 385]|[Figure 386]|[Figure 387]|[Figure 388]|[Figure 389]|[Figure 390]|[Figure 391]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 392]|[Figure 393]|[Figure 394]|[Figure 395]|[Figure 396]|[Figure 397]|[Figure 398]|[Figure 399]|[Figure 400]|[Figure 401]|
|[Figure 402]|[Figure 403]|[Figure 404]|[Figure 405]|[Figure 406]|[Figure 407]|[Figure 408]|[Figure 409]|[Figure 410]|[Figure 411]|

1 step

10 steps

50 steps

|[Figure 412]|[Figure 413]|[Figure 414]|[Figure 415]|[Figure 416]|[Figure 417]|[Figure 418]|[Figure 419]|[Figure 420]|[Figure 421]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 422]|[Figure 423]|[Figure 424]|[Figure 425]|[Figure 426]|[Figure 427]|[Figure 428]|[Figure 429]|[Figure 430]|[Figure 431]|
|[Figure 432]|[Figure 433]|[Figure 434]|[Figure 435]|[Figure 436]|[Figure 437]|[Figure 438]|[Figure 439]|[Figure 440]|[Figure 441]|

1 step

10 steps

50 steps

|[Figure 442]|[Figure 443]|[Figure 444]|[Figure 445]|[Figure 446]|[Figure 447]|[Figure 448]|[Figure 449]|[Figure 450]|[Figure 451]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 452]|[Figure 453]|[Figure 454]|[Figure 455]|[Figure 456]|[Figure 457]|[Figure 458]|[Figure 459]|[Figure 460]|[Figure 461]|
|[Figure 462]|[Figure 463]|[Figure 464]|[Figure 465]|[Figure 466]|[Figure 467]|[Figure 468]|[Figure 469]|[Figure 470]|[Figure 471]|

1 step

10 steps

50 steps

- Figure 9: Qualitative results on conditional generation (CIFAR-10). We illustrate generating images with varying sampling steps, demonstrating consistency quality even for a one-step generation.

[Figure 472]

###### Figure 10: Comparisons on unconditional generation (CIFAR-10). We compare distilled model from 2-Rectified Flow (2-RF+Distill+GAN) and CAF (CAF+Distill+GAN) with qualitative results.

1 step h = 1

h = 2

|[Figure 473]|[Figure 474]|[Figure 475]|[Figure 476]|
|---|---|---|---|
|[Figure 477]|[Figure 478]|[Figure 479]|[Figure 480]|
|[Figure 481]|[Figure 482]|[Figure 483]|[Figure 484]|
|[Figure 485]|[Figure 486]|[Figure 487]|[Figure 488]|
|[Figure 489]|[Figure 490]|[Figure 491]|[Figure 492]|
|[Figure 493]|[Figure 494]|[Figure 495]|[Figure 496]|

|[Figure 497]|[Figure 498]|[Figure 499]|[Figure 500]|
|---|---|---|---|
|[Figure 501]|[Figure 502]|[Figure 503]|[Figure 504]|
|[Figure 505]|[Figure 506]|[Figure 507]|[Figure 508]|
|[Figure 509]|[Figure 510]|[Figure 511]|[Figure 512]|
|[Figure 513]|[Figure 514]|[Figure 515]|[Figure 516]|
|[Figure 517]|[Figure 518]|[Figure 519]|[Figure 520]|

10 steps h = 1

h = 2

|[Figure 521]|[Figure 522]|[Figure 523]|[Figure 524]|
|---|---|---|---|
|[Figure 525]|[Figure 526]|[Figure 527]|[Figure 528]|
|[Figure 529]|[Figure 530]|[Figure 531]|[Figure 532]|
|[Figure 533]|[Figure 534]|[Figure 535]|[Figure 536]|
|[Figure 537]|[Figure 538]|[Figure 539]|[Figure 540]|
|[Figure 541]|[Figure 542]|[Figure 543]|[Figure 544]|

|[Figure 545]|[Figure 546]|[Figure 547]|[Figure 548]|
|---|---|---|---|
|[Figure 549]|[Figure 550]|[Figure 551]|[Figure 552]|
|[Figure 553]|[Figure 554]|[Figure 555]|[Figure 556]|
|[Figure 557]|[Figure 558]|[Figure 559]|[Figure 560]|
|[Figure 561]|[Figure 562]|[Figure 563]|[Figure 564]|
|[Figure 565]|[Figure 566]|[Figure 567]|[Figure 568]|

- Figure 11: Unconditional generation for different h on CIFAR-10. We display qualitative results of CAF for different values of h, indicating that our framework is robust to the choice of h.

|[Figure 569]|[Figure 570]|[Figure 571]|[Figure 572]|[Figure 573]|[Figure 574]|[Figure 575]|[Figure 576]|[Figure 577]|
|---|---|---|---|---|---|---|---|---|
|[Figure 578]|[Figure 579]|[Figure 580]|[Figure 581]|[Figure 582]|[Figure 583]|[Figure 584]|[Figure 585]|[Figure 586]|

(a) Ground Truth

|[Figure 587]|[Figure 588]|[Figure 589]|[Figure 590]|[Figure 591]|[Figure 592]|[Figure 593]|[Figure 594]|[Figure 595]|
|---|---|---|---|---|---|---|---|---|
|[Figure 596]|[Figure 597]|[Figure 598]|[Figure 599]|[Figure 600]|[Figure 601]|[Figure 602]|[Figure 603]|[Figure 604]|

(b) CAF (Ours) (1 step, PSNR=46.68, LPIPS=0.007)

|[Figure 605]|[Figure 606]|[Figure 607]|[Figure 608]|[Figure 609]|[Figure 610]|[Figure 611]|[Figure 612]|[Figure 613]|
|---|---|---|---|---|---|---|---|---|
|[Figure 614]|[Figure 615]|[Figure 616]|[Figure 617]|[Figure 618]|[Figure 619]|[Figure 620]|[Figure 621]|[Figure 622]|

(c) RF (1 step, PSNR=29.33, LPIPS=0.204)

#### Figure 12: Reconstruction results using inversion.

|[Figure 623]|[Figure 624]|[Figure 625]|[Figure 626]|[Figure 627]|[Figure 628]|[Figure 629]|[Figure 630]|[Figure 631]|
|---|---|---|---|---|---|---|---|---|
|[Figure 632]|[Figure 633]|[Figure 634]|[Figure 635]|[Figure 636]|[Figure 637]|[Figure 638]|[Figure 639]|[Figure 640]|

(a) Masked Images

|[Figure 641]|[Figure 642]|[Figure 643]|[Figure 644]|[Figure 645]|[Figure 646]|[Figure 647]|[Figure 648]|[Figure 649]|
|---|---|---|---|---|---|---|---|---|
|[Figure 650]|[Figure 651]|[Figure 652]|[Figure 653]|[Figure 654]|[Figure 655]|[Figure 656]|[Figure 657]|[Figure 658]|

- (b) CAF (ours) (12 step, FID=10.39)
- (c) 2-RF (12 step, FID=16.41)

|[Figure 659]|[Figure 660]|[Figure 661]|[Figure 662]|[Figure 663]|[Figure 664]|[Figure 665]|[Figure 666]|[Figure 667]|
|---|---|---|---|---|---|---|---|---|
|[Figure 668]|[Figure 669]|[Figure 670]|[Figure 671]|[Figure 672]|[Figure 673]|[Figure 674]|[Figure 675]|[Figure 676]|

- (d) CM (18 step, FID=13.16)

|[Figure 677]|[Figure 678]|[Figure 679]|[Figure 680]|[Figure 681]|[Figure 682]|[Figure 683]|[Figure 684]|[Figure 685]|
|---|---|---|---|---|---|---|---|---|
|[Figure 686]|[Figure 687]|[Figure 688]|[Figure 689]|[Figure 690]|[Figure 691]|[Figure 692]|[Figure 693]|[Figure 694]|

- Figure 13: Zero-shot box inpainting results. We use a 16×16 size mask for masked images in (a). For consistency model in (d), we followed their official code for inpainting.

|[Figure 695]|[Figure 696]|[Figure 697]|[Figure 698]|[Figure 699]|[Figure 700]|[Figure 701]|[Figure 702]|[Figure 703]|
|---|---|---|---|---|---|---|---|---|
|[Figure 704]|[Figure 705]|[Figure 706]|[Figure 707]|[Figure 708]|[Figure 709]|[Figure 710]|[Figure 711]|[Figure 712]|
|[Figure 713]|[Figure 714]|[Figure 715]|[Figure 716]|[Figure 717]|[Figure 718]|[Figure 719]|[Figure 720]|[Figure 721]|
|[Figure 722]|[Figure 723]|[Figure 724]|[Figure 725]|[Figure 726]|[Figure 727]|[Figure 728]|[Figure 729]|[Figure 730]|
|[Figure 731]|[Figure 732]|[Figure 733]|[Figure 734]|[Figure 735]|[Figure 736]|[Figure 737]|[Figure 738]|[Figure 739]|
|[Figure 740]|[Figure 741]|[Figure 742]|[Figure 743]|[Figure 744]|[Figure 745]|[Figure 746]|[Figure 747]|[Figure 748]|
|[Figure 749]|[Figure 750]|[Figure 751]|[Figure 752]|[Figure 753]|[Figure 754]|[Figure 755]|[Figure 756]|[Figure 757]|
|[Figure 758]|[Figure 759]|[Figure 760]|[Figure 761]|[Figure 762]|[Figure 763]|[Figure 764]|[Figure 765]|[Figure 766]|
|[Figure 767]|[Figure 768]|[Figure 769]|[Figure 770]|[Figure 771]|[Figure 772]|[Figure 773]|[Figure 774]|[Figure 775]|
|[Figure 776]|[Figure 777]|[Figure 778]|[Figure 779]|[Figure 780]|[Figure 781]|[Figure 782]|[Figure 783]|[Figure 784]|
|[Figure 785]|[Figure 786]|[Figure 787]|[Figure 788]|[Figure 789]|[Figure 790]|[Figure 791]|[Figure 792]|[Figure 793]|
|[Figure 794]|[Figure 795]|[Figure 796]|[Figure 797]|[Figure 798]|[Figure 799]|[Figure 800]|[Figure 801]|[Figure 802]|
|[Figure 803]|[Figure 804]|[Figure 805]|[Figure 806]|[Figure 807]|[Figure 808]|[Figure 809]|[Figure 810]|[Figure 811]|
|[Figure 812]|[Figure 813]|[Figure 814]|[Figure 815]|[Figure 816]|[Figure 817]|[Figure 818]|[Figure 819]|[Figure 820]|
|[Figure 821]|[Figure 822]|[Figure 823]|[Figure 824]|[Figure 825]|[Figure 826]|[Figure 827]|[Figure 828]|[Figure 829]|
|[Figure 830]|[Figure 831]|[Figure 832]|[Figure 833]|[Figure 834]|[Figure 835]|[Figure 836]|[Figure 837]|[Figure 838]|

###### Figure 14: Qualitative results on conditional generation for ImageNet 64×64 (N = 1, FID=1.69).

