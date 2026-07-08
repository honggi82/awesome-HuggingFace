# arXiv:2604.10465v1[cs.LG]12Apr2026

## Rethinking the Diffusion Model from a Langevin Perspective*

Candi Zheng Yuan Lan

Department of Mathematics, The Hong Kong University of Science and Technology

Abstract

Diffusion models are often introduced from multiple perspectives, such as VAEs, score matching, or flow matching, accompanied by dense and technically demanding mathematics that can be difficult for beginners to grasp. One classic question is: how does the reverse process invert the forward process to generate data from pure noise? This article systematically organizes the diffusion model from a fresh Langevin perspective, offering a simpler, clearer, and more intuitive answer. We also address the following questions: how can ODE-based and SDE-based diffusion models be unified under a single framework? Why are diffusion models theoretically superior to ordinary VAEs? Why is flow matching not fundamentally simpler than denoising or score matching, but equivalent under maximum-likelihood? We demonstrate that the Langevin perspective offers clear and straightforward answers to these questions, bridging existing interpretations of diffusion models, showing how different formulations can be converted into one another within a common framework, and offering pedagogical value for both learners and experienced researchers seeking deeper intuition.

### Contents

- 1 Introduction 2
- 2 Langevin Dynamics as ’Identity’ Operation 3
- 3 Spliting the Identity into Forward and Reverse Processes 4

- 3.1 The Forward Diffusion Process for Noising . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2 The Reverse Diffusion Process for Denoising . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.3 Converting Between Different Model Types . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Forward–Reverse Duality 9
- 5 Unifying Training of Diffusion Models as Maximal likelihood 12
- 6 Conclusion 14 A Optional Derivations 15

- A.1 Why p(x) is stationary under Langevin dynamics . . . . . . . . . . . . . . . . . . . . 15
- A.2 Derivation Step 1: from forward SDE to Fokker–Planck . . . . . . . . . . . . . . . . 16
- A.3 Derivation Step 2: KL decay and squared-score objective . . . . . . . . . . . . . . . . 17
- A.4 Derivation: equivalence between DSM and SM . . . . . . . . . . . . . . . . . . . . . 18

maczheng@ust.hk

∗This article is adapted and organized from the ICLR Blogpost Track post: https://iclr-blogposts.github. io/2026/blog/2026/rethinking-diffusion-langevin/.

### 1 Introduction

Modern diffusion models are built upon two fundamental processes: the forward process, which gradually corrupts data with noise during training, and the reverse process, which generates data by sampling from noise. The development of diffusion models has diverged into several valuable perspectives, illuminating different aspects of these processes. Most interpretations fall into three main frameworks: the variational autoencoder (VAE) perspective, the score-based perspective, and the flow-based perspective. Although there are many tutorials available, learning the core theory of diffusion models remains challenging for beginners due to mathematically dense derivations and fragmented intuitions scattered across these different perspectives.

The VAE perspective treats the forward diffusion process as an encoder that adds noise to the data and the reverse process as a decoder that removes noise, with the Evidence Lower Bound (ELBO) serving as the training objective [3, 7]. This framework is straightforward for those familiar with VAEs. However, it is not obvious why the iterative denoising in diffusion models outperforms the one-step decoding typical of ordinary VAEs.

The score-based perspective [9] places a clearer emphasis on the paired relationship between the forward and reverse processes, which contributes to the superiority of diffusion models. It typically introduces the forward process first, then directly presents the reverse process by reversetime diffusion [1] without derivation. Understanding the derivation of the reverse process usually requires familiarity with advanced mathematical concepts such as the Kolmogorov backward equations, which makes it less accessible. Additionally, the score matching objective is specifically tailored for score models, making it less straightforward to generalize to other approaches such as flow matching models.

A third valuable viewpoint is the flow-based perspective [6], which has rapidly gained popularity in modern diffusion models. Although this approach is theoretically equivalent to both the VAE and score-based frameworks [2], it distinguishes itself by highlighting a clear and intuitive straight-line interpolation between data and noise. This conceptual clarity makes the flow-based perspective accessible and attractive. However, this apparent simplicity can be misleading: it can create the impression that flow matching is fundamentally simpler than denoising or score matching, rather than a mathematically equivalent reformulation.

In this article, we systematically organize the theory of diffusion models and present a perspective that is both mathematically simple and intuitively clear: the Langevin perspective. This approach, relying only on basic techniques from stochastic differential equations (SDEs), provides a straightforward derivation of the reverse process and explains why flow matching is not fundamentally simpler than denoising or score matching, but is equivalent to them under maximum likelihood.

[Figure 1]

### 2 Langevin Dynamics as ’Identity’ Operation

This section will show that Langevin dynamics acts as an ‘identity’ operation on distributions, mapping a sample from a distribution to another sample from the same distribution.

Langevin dynamics [5] is a stochastic process for sampling from a target probability distribution p(x). One common form is the SDE

dxt = g(t)s(xt)dt + 2g(t)dWt, (1) where s(x) = ∇x log p(x).

At first sight, the extra term dWt may make this SDE look much more complicated than an ordinary differential equation (ODE). In fact, it is best to think of it as an ODE with an additional infinitesimal random perturbation at each step. Informally, one can write

dWt = √

dtϵ, (2)

where ϵ is a standard Gaussian random noise. The remaining terms are familiar: s(x) is the score function of p(x), and g(t) is an arbitrary positive function rescaling the time t.

This dynamics is often used as a Monte Carlo sampler to draw samples from p(x), since p(x) is its stationary distribution—the distribution that xt converges to and remains at as t → ∞, regardless of the initial distribution of x0. For an intuitive derivation of this statement, see Section A.1.

Langevin dynamics, while widely used for sampling from complex distributions, becomes inefficient in high-dimensional or multimodal settings due to slow mixing and sensitivity to hyperparameters such as step size and noise scale. Nevertheless, it plays a crucial foundational role in diffusion models because of the following property:

For a target distribution p(x), Langevin dynamics acts as an identity operation on the distribution, transforming a sample from p(x) into a new, independent sample from the same distribution.

This “identity on distribution” view is the key bridge to diffusion models. Forward and reverse processes can be interpreted as a split of this identity into a noising phase and a denoising phase.

[Figure 2]

- Figure 1: Langevin dynamics acts as an identity operation on p(x): starting from a sample x ∼ p(x), it produces a new sample x′ from the same distribution.

The identity viewpoint in Fig. 1 will be the organizing principle for the rest of this article.

### 3 Spliting the Identity into Forward and Reverse Processes

One key reason Langevin dynamics struggles in high-dimensional settings is the challenge of initialization [8]. The score function required by it is learned from real data and is therefore reliable only near true data points, while being poorly estimated elsewhere. Yet in generative modeling we need to start from locations that may be far from the data manifold. Finding an initialization that is both realistic and close enough to the true data manifold is difficult, making effective generation with Langevin dynamics challenging in practice. In short, Langevin dynamics is well-suited for generating new samples from an existing one, but ill-suited for generating samples entirely from scratch.

An enhancement to Langevin dynamics is the Annealed Langevin dynamics [8]. Instead of using a single Langevin sampler, this method involves training a sequence of Langevin dynamics, each corresponding to a different level of noise added to the data. Starting from pure noise, the method gradually reduces the noise level, switching between these samplers at each step. In this way, samples are progressively transformed from random noise into data-like samples, using Langevin dynamics that are effective for each stage of noise contamination. This approach highlights the importance of using multiple noise levels.

Diffusion models take this concept a step further by completely separating the training and inference processes: one process trains the model at different noise levels, while another process samples from noise to generate data. In this section, we show that the forward and reverse processes in diffusion models are splits of a single Langevin dynamics, decomposing the identity operation into a noising phase and a denoising phase.

#### 3.1 The Forward Diffusion Process for Noising

The forward diffusion process in diffusion model generates the necessary training data: clean images and their progressively noised counterparts. In continuous time, a very general way to describe such a process is by an Itô SDE of the form

dxt = f(xt,t)dt + g(t)dWt, t ∈ [0,T]. (3)

Table 1: Forward processes across model parameterizations.

Model Type Noise-level parameter Forward process Forward SDE

VP αt = e−t xt = √αt x0 + √1 − αt ϵ dxt = −12xt dt + dWt VE-Karras σ zσ = z0 + σ ϵ dzσ = √2σ dWσ Rectified flow s rs = (1 − s)r0 + sϵ drs = −1r−ss ds + 12−ss dWs

[Figure 3]

- Figure 2: Overview of forward processes across VP, VE-Karras, and rectified-flow parameterizations (exported from the original interactive visualization https://iclr-blogposts.github.io/2026/ blog/2026/rethinking-diffusion-langevin/).

where t ∈ [0,T] is the forward diffusion time, xt is the noise-contaminated image at time t, Wt is a Brownian motion, f(xt,t) is the drift, and g(t) scales the injected noise. Different choices of f and g correspond to different forward-diffusion parameterizations used in diffusion models.

In practice, diffusion models are usually instantiated by choosing specific parameterizations of this SDE. The most common ones are the variance-preserving (VP) process, implemented in DDPMs as an Ornstein–Uhlenbeck dynamics that gently pulls samples toward the origin while injecting noise so that the marginal converges to a standard Gaussian; the variance-exploding (VE) process, where there is no restoring drift and the noise scale grows with time so that the variance “explodes”; and flow-matching formulations, which view generation as following a timedependent flow that implements a “straight line” interpolation between data and noise under a carefully designed schedule.

Table 1 summarizes these three forward processes of different model types, as well as their corresponding SDEs expressed in terms of their respective noise-levels. In what follows, we adopt Karras’ notation for the VE parameterization [4].

Each forward process has a characteristic way of mixing data and noise: the VP model uses the Ornstein–Uhlenbeck process, so samples drift toward the origin while their uncertainty grows; the VE-Karras model adds noise directly to the data without a restoring drift, so the mean stays fixed while the sample cloud expands outward; and the Rectified flow model is a stochastic forward

Table 2: Conversion between forward-process variable parameterizations.

Given parameterization Equivalent VP Equivalent VE-Karras Equivalent Rectified-flow

xt √αt σ =

xt √αt + √1 − αt s =

zσ =

rs =

√1 − αt √αt + √1 − αt

1 − αt αt

/

VP (xt,αt)

zσ √1 + σ2 αt =

xt =

zσ 1 + σ s =

rs =

1 1 + σ2

/

σ 1 + σ

VE-Karras (zσ,σ)

rs (1 − s)2 + s2

xt =

rs 1 − s σ =

zσ =

(1 − s)2 (1 − s)2 + s2

/

s 1 − s

Rectified flow (rs,s)

αt =

process as well, not a deterministic straight-line interpolation. This behavior is illustrated in Fig. 2.

Despite their differences, all above SDEs are fundamentally equivalent; they differ only by how time and state are reparameterized. For clarity, Table 2 gives a direct conversion between any two parameterizations [10]. Using this table, one can directly translate between any two parameterizations whenever needed. No matter which notation we choose, a forward diffusion step with a step size of ∆t acts as adding more noise to data, which is displayed in Fig. 3.

[Figure 4]

- Figure 3: A forward diffusion step with step size ∆t adds Gaussian noise to data, pushing samples closer to a Gaussian distribution.

#### 3.2 The Reverse Diffusion Process for Denoising

The reverse diffusion process is the conjugate of the forward process. While the forward process evolves pt(x) toward Gaussian noise, the reverse process reverses this evolution, restoring Gaussian noise to pt.

The concept behind the reverse process is intuitive: since Langevin dynamics acts as an identity operation on a distribution—preserving it unchanged—any forward process composed with its corresponding reverse process should similarly yield a Langevin dynamics. Specifically, at any time t, combining the forward and reverse processes should reproduce the Langevin dynamics for the distribution pt(x), as illustrated in Fig. 4.

To formalize this, consider the VP case with the following Langevin dynamics for pt(x) with a time variable τ, distinguished from the forward diffusion time t. This dynamics can be decomposed into forward and reverse components as follows:

dxτ = s(xτ,t)dτ + √2dWτ

- 1

- 2

- 1

- 2

. (4)

xτ dτ + dWτ(1)

xτ + s(xτ,t) dτ + dWτ(2)

= −

+

Forward

Reverse

Table 3: Langevin split of different model types.

Model Type Langevin dynamics Reverse Split Forward Split

VP-SDE dx = sx dτ + √2dWτ dx = 12x + sx dτ + dWτ dx = −12xdτ + dWτ VP-ODE dx = 12sx dτ + dWτ dx = 12 (x + sx)dτ dx = −12xdτ + dWτ VE-Karras dz = τ sz dτ + √2τ dWτ dz = τ sz dτ dz = √2τ dWτ

Rectified flow dr = 1+ττ sr dτ + 1+2ττ dWτ dr = τ sr+r

1−τ dτ dr = −1−rτ dτ + 12−ττ dWτ

[Figure 5]

- Figure 4: The forward and reverse diffusion processes compose to reproduce Langevin dynamics.

where s(x,t) = ∇x log pt(x) is the score function of pt(x). Here, we split the noise term √2dWτ into two independent Gaussian increments, dWτ(1) and dWτ(2), such that their sum equals the original noise: √2dWτ = dWτ(1) + dWτ(2). This split is possible because Gaussian random variables satisfy the property that their sum is Gaussian, and independent Gaussians add in variance; specifically, if dWτ(1) and dWτ(2) are independent standard Brownian increments (each with variance dτ), their sum has variance 2dτ, matching the original √2dWτ. This decomposition now lets us directly answer the first question posed in the abstract:

How does the reverse process invert the forward process to generate data from pure noise?

The “Forward” part in this decomposition corresponds to the forward diffusion process, effectively increasing the forward diffusion time t by dτ, bringing the distribution to pt+dτ(x). Since the forward and reverse components combine to form an “identity” Langevin dynamics, the “Reverse” part must reverse the forward process, decreasing the forward diffusion time t by dτ and restoring the distribution back to pt(x).

We can therefore read off the reverse process as

- 1

- 2

dxt′ =

xt′ + s(xt′,t) dt′ + dWt′. (5)

This reverse diffusion process is itself a standalone SDE that advances reverse time t′. If xt′ ∼ qt′(x), then a step with increment dt′ = ∆t′ moves it to xt′+∆t′ ∼ qt′+∆t′(x).

Having analyzed the VP case in detail, we can now apply the same decomposition approach to other diffusion schemes, which involve different choices of Langevin dynamics. This brings us to the second question raised in the abstract:

How can ODE-based and SDE-based diffusion models be unified under a single framework?

Table 3 provides a direct answer: these models are unified by decomposing different Langevin dynamics. We have decomposed the VP model into both SDE and ODE versions, as well as other parameterizations, relating their Langevin dynamics to the corresponding forward and reverse processes.

A key observation from this table is that the Langevin split is not unique. For the same VP model, we present two distinct splittings, the SDE and ODE versions, which are decompositions of different Langevin dynamics differing in their time scaling functions g(τ). The ODE version corresponds to a splitting where the reverse process contains no stochastic term dW.

Besides the decomposition of Langevin dynamics, we still have one problem: note that the

s(xt′,t) term in the reverse process still depends on the forward time t, not the reverse time t′; we need the relationship between the forward time t and the reverse time t′ to close the equation. Note that a single reverse-time step dt′ can be understood in two complementary ways:

- 1. As an undoing of the forward diffusion: one step of the reverse diffusion process with dt′ = ∆t removes a small amount of noise and therefore reduces the forward diffusion time by ∆t.
- 2. As forward evolution in its own clock: the reverse diffusion process is itself a well-defined SDE/ODE in the variable t′, so one step with dt′ = ∆t simply advances the reverse diffusion time from t′ to t′ + ∆t.

Together, these two viewpoints determine how the forward and reverse clocks are related. Since a positive reverse-time step dt′ > 0 both decreases the forward time t and increases the reverse time t′, their infinitesimal increments must satisfy

dt = −dt′. (6)

which means that t′ runs in the opposite direction to t. To make t′ lie in the same range [0,T] as the forward diffusion time, we can define

t = T − t′. (7)

so that t = 0 corresponds to t′ = T and t = T corresponds to t′ = 0. In this notation, the reverse diffusion process of VP is

dxt′ =

- 1

- 2

xt′ + s(xt′,T − t′) dt′ + dWt′. (8)

in which t′ ∈ [0,T] is the reverse time, s(x,t) = ∇x log pt(x) is the score function of the density of xt in the forward process.

The same reasoning applies not only to SDE reverse processes but also to ODE reverse processes. The full summary is listed in Table 4.

In this table, ϵ and v are just different ways of writing expressions based on the basic score functions. The score functions themselves are

sx(x,t) = ∇xt log p(xt), sz(z,σ) = ∇zσ log p(zσ), sr(r,s) = ∇rs log p(rs). (9) These reverse equations become more intuitive when we visualize how samples move under each

parameterization, as shown in Fig. 5:

Table 4: Reverse diffusion processes across model types.

Model Type Reverse Process Relation to Score Reverse Time Reverse time domain

VP-SDE dxt′ = 12xt′ + s(xt′,T − t′) dt′ + dWt′ s(x,t) = sx(x,t) t′ = T − t t′ ∈ [0,T] VP-ODE dxt′ = 21 [xt′ + s(xt′,T − t′)]dt′ s(x,t) = sx(x,t) t′ = T − t t′ ∈ [0,T] VE-Karras dzσ′ = −ϵ(zσ′,Σ − σ′)dσ′ ϵ(z,σ) = −σsz(z,σ) σ′ = Σ − σ σ′ ∈ [0,Σ]

Rectified flow drs′ = −v(rs′,1 − s′)ds′ v(r,s) = −ssr1(r−,ss)+r s′ = 1 − s s′ ∈ [0,1]

Table 5: Conversion between model prediction.

Given prediction Equivalent VP score sx Equivalent VE noise ϵ Equivalent RF velocity v

√

/

√1 − αt sx(xt,αt) v(rs,s) = − x

√αtt − 1−αt+

αt(1−αt)

√αt sx(xt,αt) VE noise ϵ(zσ,σ) sx(xt,αt) = −

VP score sx(xt,αt)

ϵ(zσ,σ) = −

###### /

√1+σ2

v(rs,s) = (1 + σ)ϵ(zσ,σ) − zσ RF velocity v(rs,s) sx(xt,αt) = −

σ ϵ(zσ,σ)

√

/

(1−s)2+s2

s (rs + (1 − s)v(rs,s)) ϵ(zσ,σ) = rs + (1 − s)v(rs,s)

In this single-data-point example, the reverse trajectories reveal a clear geometric difference between the parameterizations. The VP-SDE and VP-ODE flows bend along a curved path as they return to the target point, whereas the VE-Karras and Rectified flow trajectories move approximately along a straight line toward that point. It is important to emphasize that this straight-line behavior is a special feature of the one-point setting shown in the example, not the general case. For a general data distribution, the learned reverse vector fields vary across space, so all of these reverse trajectories are typically curved. Nevertheless, one could still expect the VE-Karras and Rectified flow trajectories to have smaller curvature than the VP trajectories.

#### 3.3 Converting Between Different Model Types

Despite their different geometric behaviors, all model types we discussed above are inherently equivalent parameterizations. Although VP uses the score sx, VE-Karras uses the noise prediction ϵ, and Rectified flow uses the velocity field v as their native outputs, these model types are mathematically equivalent parameterizations. Combined with the previous conversion table for the forward-process variables, we can therefore convert these fields into one another exactly [10].

Table 5 summarizes these conversions. From this table, we can see directly that the velocity learned in flow matching is equivalent to the noise prediction and the score under a change of parameterization. Its main advantage is therefore not that it produces truly straight-line trajectories, but that it is often expected to produce trajectories with smaller curvature.

### 4 Forward–Reverse Duality

We have established that a single reverse step undoes a forward step: advancing the reverse time t′ by an amount corresponds to decreasing the forward time t by the same amount. Now, let us examine what happens when we combine multiple forward and reverse steps to reveal the deeper duality between them. In fact, the forward process transforms a data distribution into noise, while the reverse process, starting from noise, generates samples from the same data distribution.

Consider the following sequence: begin with a data sample x0, propagate it through the forward process to obtain xT, then use xT as the starting point x0′ for the reverse process and evolve it to xT′. Part of this forward–reverse cycle is illustrated in Fig. 6.

[Figure 6]

- Figure 5: Reverse trajectories under different parameterizations (exported from the original interactive visualization https://iclr-blogposts.github.io/2026/blog/2026/ rethinking-diffusion-langevin/).

The green arrows represent consecutive forward process steps that advance the forward diffusion time t, while the blue arrows indicate consecutive reverse process steps that advance the reverse diffusion time t′. We examine the relationship between xt in the forward diffusion process and xt′=T−t in the reverse diffusion process. The composition of a forward and a reverse step constitutes a Langevin dynamics step. This allows us to connect x in the forward process with those in the reverse process through Langevin dynamics steps, as illustrated in Fig. 7.

Each horizontal row in this picture corresponds to consecutive steps of Langevin dynamics, which alter the samples while maintaining the same probability density. This illustrates the duality between the forward and reverse diffusion processes: while xt (forward) and x(T−t)′ (reverse) are distinct samples, they obey the same probability distribution.

To formalize the duality, let pt(x) denote the density of the forward process at time t, and let qt′(x) denote the density of the reverse process at reverse time t′. If we initialize

q0(x) = pT(x), (10) then their evolution are related by

qt′(x) = pT−t′(x). (11)

[Figure 7]

- Figure 6: Part of a forward–reverse diffusion cycle: the last two steps of the forward process (green arrows, increasing t) followed by the first two steps of the reverse process (blue arrows, increasing t′ while decreasing t).

[Figure 8]

- Figure 7: Each horizontal row shows a Langevin dynamics step that maps a forward sample xt to a new reverse sample x(T−t)′ from the same probability density.

In diffusion models, the terminal time T is chosen sufficiently large that the forward-process distribution pT(x) converges to a simple Gaussian distribution. This ensures that the reverse process can start from the same Gaussian distribution q0(x) at t′ = 0. By then evolving the reverse process through time t′ from 0 to T, we obtain samples that follow the original data distribution:

qT(x) = p0(x) (data distribution). (12) This exact recovery of the data distribution p0 through a forward–reverse duality brings us to

the third question from the abstract.

##### Why are diffusion models theoretically superior to ordinary VAEs?

The above duality means that if we run the reverse process from time t′ = 0 to t′ = T, the final samples follow exactly the same distribution as the original training data p0. In other words, the forward and reverse processes form an exact prior–posterior pair: the forward process maps data to noise, and the reverse process maps noise back to data. In practice, training introduces

approximation error, but the theoretical target is exact equality. Ordinary VAEs, by contrast, only require the decoder to approximate the encoder posterior, with no guarantee of exactness even at the ELBO optimum.

Now we have demonstrated that reverse diffusion—the dual of the forward process—can generate image data from noise. However, this requires access to the score function at every time step t. In practice, we approximate this function using a neural network. In the next section, we will explain how to train such score networks.

### 5 Unifying Training of Diffusion Models as Maximal likelihood

In this section, we derive the training objective directly from the maximum-likelihood framework. By doing so, we reveal the fundamental connection between diffusion model loss and exact maximum likelihood, and show that score matching, denoising, and flow matching are equivalent manifestations of this same objective rather than fundamentally different levels of simplicity.

Training the diffusion model involves addressing two fundamental questions: (1) What mathematical quantity should we model, and (2) What objective function should guide the training? Here, we start by analyzing the Kullback–Leibler (KL) divergence.

Suppose we have two distributions p(x,t) and q(x,t) that both evolve under the same forward diffusion process. Think of p as the true data distribution pushed forward by the diffusion dynamics, and q as the model distribution. At any fixed time t, their KL divergence is

KL(pt∥qt) = p(x,t)log

- p(x,t)

- q(x,t)

dx. (13)

Maximum likelihood training aims to minimize the KL divergence KL(p0∥q0) at time t = 0, where p0 is the true data distribution and q0 is the model distribution. However, in diffusion models, we introduce a forward process that evolves distributions over time t, and we learn a reverse process that maps from noisy states at different times back to clean data. This temporal structure suggests that rather than focusing solely on the KL divergence at t = 0, we should consider how this divergence evolves throughout the entire diffusion process. The key insight is to distribute the KL minimization objective across all diffusion times by examining the time derivative of KL(pt∥qt) along the forward dynamics.

Formally, we can rewrite the time-zero KL as an integral over its time derivative: KL(p0∥q0) = KL(p0∥q0) − KL(p∞∥q∞)

(14)

∞ 0

d dt

= −

KL(pt∥qt) dt,

where the second equality uses KL(p∞∥q∞) = 0 at infinitely large time, since both p and q converge to the same Gaussian noise distribution.

This naturally identifies the instantaneous contribution to the likelihood objective as

d dt

Lt := −

KL(pt∥qt). (15)

Thus minimizing KL(p0∥q0) is equivalent to minimizing these contributions on average over diffusion time.

We now show that as long as the forward diffusion process takes the form dx = f(x,t)dt + g(t)dW, (16)

Table 6: Training targets and losses under different parameterizations.

Model Type Noise-state relation Network output sθ w.r.t. NN ∇log p(xt | x0) Loss Lt

2

VP xt = √αt x0 + √1 − αt ϵ sθ(xt,t) sθ(xt,t) −√1−ϵ αt

- 1

- 2Ex

t∼pt(·|x0) −√1−ϵ αt − sθ(xt,t)

0∼p0Ex

σ∼pσ(·|z0) ∥ϵθ(zσ,σ) − ϵ∥2 Rectified flow rs = (1 − s)r0 + sϵ vθ(rs,s) −vθ(rs,s)(1−s)−rs

θ(zσ,σ)

σ −σϵ σ1Ez

VE-Karras zσ = z0 + σ ϵ ϵθ(zσ,σ) −ϵ

0∼p0Ez

s∼ps(·|r0) ∥ϵ − r0 − vθ(rs,s)∥2

s −ϵs 1−ssEr

0∼p0Er

the instantaneous contribution is Lt =

- 1

- 2

g(t)2 p(x,t) ∇log p(x,t) − ∇log q(x,t) 2dx

(17)

- 1

- 2

g(t)2Ex∼p(x,t) ∇log p(x,t) − ∇log q(x,t) 2.

=

Equation (17) shows that the score functions ∇log p(x,t) and ∇log q(x,t) for the true data distribution and the model distribution appear naturally inside the objective. Hence, the score function naturally arises as the quantity we should model. Full derivations of the Fokker–Planck equation and KL decay are provided in Sections A.2 and A.3.

In practice, we approximate the model score ∇log q(x,t) using a neural network. For standard

score-based models, we model sθ(x,t) directly. For VE-Karras and rectified-flow parameterizations, we instead model related quantities such as noise prediction ϵ or velocity v, which can be converted back to a score.

The only thing remains to handle is the score of the true data distribution ∇log p(x,t), which should be approximated by an empirical value from samples since we do not know its value. In fact,

argminsθEx0∼p0Ext∼pt(·|x0) ∥∇log p(xt|x0) − sθ∥2 = argminsθEx∼p(x,t) ∥∇log p(x,t) − sθ∥2 . (18) The left-hand side is the denoising score matching loss, while the right-hand side is the score matching loss. Their equivalence is shown in Section A.4.

This tells us that training the diffusion model, we only need to figure out the ∇log p(xt | x0), then minimize the loss

- 1

- 2

g(t)2Ex0∼p0 Ext∼pt(·|x0) ∥∇log p(xt | x0) − sθ∥2 . (19) Equipped with this instantaneous maximum-likelihood objective, we can now address the fourth

Lt =

and final question from the abstract.

Why flow matching is not fundamentally simpler than denoising or score matching, but equivalent under maximum-likelihood?

With the maximum-likelihood objective derived above, we can compare different parameterizations in a common framework and see explicitly why flow matching is not a fundamentally simpler alternative, but an equivalent reformulation of denoising and score matching.

Table 6 shows the loss functions for different diffusion model types. For the VP model, the loss directly trains a score function. For the VE-Karras model, the loss trains a network ϵθ to predict the Gaussian noise added to the data; this is the familiar epsilon-prediction parameterization. Other choices such as x0-prediction or v-prediction are algebraically equivalent reformulations of the same objective.

For the rectified-flow model, it looks like learning a constant velocity, but that is not the case. Note that with rs = (1 − s)r0 + sϵ we have r1 = ϵ, so the loss can be written as

∥r1 − r0 − vθ(rs,s)∥2 . (20)

Table 7: Unified summary of forward process, reverse process, and objective.

Model Type Forward Process Reverse Process Loss (up to a weight factor)

VP-SDE xt = √αt x0 + √1 − αt ϵ dxt′ = 12xt′ + s(xt′,T − t′) dt′ + dWt′ Ex0∼p0Ext∼pt(·|x0) −√1−ϵ αt − sθ(xt,t) 2 VP-ODE xt = √αt x0 + √1 − αt ϵ dxt′ = 21 [xt′ + s(xt′,T − t′)]dt′ Ex0∼p0Ext∼pt(·|x0) −√1−ϵ αt − sθ(xt,t) 2 VE-Karras zσ = z0 + σ ϵ dzσ′ = −ϵ(zσ′,Σ − σ′)dσ′ Ez0∼p0Ezσ∼pσ(·|z0) ∥ϵθ(zσ,σ) − ϵ∥2 Rectified flow rs = (1 − s)r0 + sϵ drs′ = −v(rs′,1 − s′)ds′ Er0∼p0Ers∼ps(·|r0) ∥ϵ − r0 − vθ(rs,s)∥2

If we interpret r0 and r1 as particle positions at times s = 0 and s = 1, then r1 − r0 is the average velocity over [0,1], which motivates viewing vθ as a velocity field and writing the reverse process as dr = −v(r,s)ds. This has led to the intuition that rectified flows are trained on simple straight lines and are therefore conceptually simpler than diffusion models. However, vθ(r,s) still depends on time s, so the velocity changes over time and trajectories are not truly straight in state–time space. More importantly, Table 6 shows that this velocity field is algebraically tied to the same underlying score function that appears in denoising and score matching. Under the maximum-likelihood objective, flow matching is therefore best understood not as a fundamentally simpler class, but as an equivalent parameterization of the same diffusion objective.

A note on loss weighting is also important. In practice, the coefficient outside the L2 norm, such

as 12, σ1, or 1−ss, is often omitted or replaced with a custom weighting schedule to improve training performance. This is valid because modifying this coefficient only changes the relative importance

of the loss across different time steps t—it does not affect the optimal solution at any individual time t. In other words, reweighting adjusts how much we prioritize learning at different noise levels, but the target (the true score or velocity) remains unchanged.

Combining all results from previous discussion, we summarize the forward, reverse, and loss for each diffusion type in Table 7.

### 6 Conclusion

From the Langevin perspective, diffusion models become conceptually simple: the forward and reverse processes are just a carefully chosen split of Langevin dynamics, which itself is an “identity map”. This viewpoint simultaneously explains how sampling inverts noising, unifies SDE and ODE formulations as different splittings of the same dynamics, and clarifies why diffusion models implement exact maximum likelihood in a way ordinary VAEs do not.

It also shows why flow matching is not fundamentally simpler than denoising or score matching, but instead an equivalent way of estimating the same underlying score field under the maximumlikelihood objective that governs Langevin dynamics. We hope this perspective helps demystify diffusion models to learners, so that new variants can be understood not as disconnected tricks, but

- as different parameterizations and discretizations of a single, coherent Langevin story.

### Acknowledgements

This work was supported in part by the General Research Fund 16302823, an Area of Excellence project (AoE/E-601/24-N), and a Theme-based Research Project (T32-615/24-R) from the Research Grants Council of the Hong Kong Special Administrative Region, China. We also acknowledge funding from the Hong Kong Innovation and Technology Commission (ITCPD/17-9).

Appendix. All optional derivations from the original blog are migrated to Sections A.1 to A.4.

### A Optional Derivations

#### A.1 Why p(x) is stationary under Langevin dynamics

- 1. Set g(t) = 1 by rescaling time as t′ = 0 t g(τ)dτ. Under this change of variables, the dynamics become

dxt′ = s(xt′)dt′ + √2dWt′,

which is equivalent to the case g(t′) = 1. Thus, g(t) only sets the time unit and does not affect the stationary distribution.

- 2. Let us consider the dynamics in energy form as dxt = −∇E(x)dt + √2dWt.

The random term dWt’s role is to perturb the system into complete, uniform chaos. The only position information is injected by the energy E(x). Thus, the stationary distribution shall have the form p(x) = f(E(x)) for some function f.

- 3. Consider N independent copies x1,...,xN. Their joint density must be the product form

- f(E(x1))···f(E(xN)). From another point of view, when treating them as a single system, the total energy is additive:

E(x1,...,xN) = E(xi). Therefore, the joint stationary density of N independent copies must also be the addition form

- g( E(xi)) for some function g. The only function f that turns product form into addition form is the exponential: f(E) = e−βE. This yields

p(x) ∝ e−βE(x).

- 4. To find β, take E(x) = 21∥x∥2. This gives the well known Ornstein–Uhlenbeck process

dxt = −x dt + √2dWt with known stationary N(0,I), density ∝ e−12∥x∥2. Matching forms gives β = 1. Thus, the dynamics

dxt = −∇E(x)dt + √2dWt has stationary distribution ∝ e−E(x), and

dxt = ∇x log p(x)dt + √2dWt has stationary distribution p(x).

#### A.2 Derivation Step 1: from forward SDE to Fokker–Planck

Given the SDE

dx = f(x,t)dt + g(t)dW,

we first ask: how does the probabilistic density pt(x) evolve in time? The answer is the Fokker–Planck equation, which describes the time evolution of the probability density p(x,t) induced by the SDE:

- 1

- 2

∂p ∂t

g(t)2 ∇2p.

:= −∇ · [f(x,t)p] +

This PDE shows how drift f and diffusion g jointly shape the distribution. Rigorous derivations can be found in standard references; here we only sketch an intuitive 1D argument for the drift part:

Drift term f. Start with a 1D motion with constant velocity v, so dx = v dt. After time t, a particle now at position x must have come from x − vt at time 0, so

p(x,t) = p(x − vt,0). Differentiating this identity w.r.t. t gives the continuity equation

∂p ∂t

∂ ∂x

+

v p(x,t) = 0.

For a general 1D deterministic dynamics dx = f(x,t)dt, the same reasoning yields

∂ ∂x

∂p ∂t

+

f(x,t)p(x,t) = 0.

We keep f(x,t) inside the ∂x because this term represents the probability flux. This guarantees conservation: integrating the total derivative ∂x(fp) over all space gives zero (assuming p vanishes

- at boundaries), preserving the total probability. Noise term g dW. Consider now the pure diffusion SDE dx = g dW with constant g and initial

condition x(0) = 0. At time t, the accumulated Brownian motion from 0 to t is Gaussian with variance t, so x(t) is Gaussian with variance g2t and density

p(x,t) =

x2 2g2t

- 1

- 2πg2t

exp −

.

One can check directly that this density satisfies the diffusion equation

∂2p ∂x2

- 1

- 2

∂p ∂t −

g2

= 0.

Combining drift and diffusion, we obtain that

∂2p ∂x2

- 1

- 2

∂ ∂x

∂p ∂t

g(t)2

= −

[f(x,t)p] +

,

which is the 1D specialization of the Fokker–Planck equation stated above.

#### A.3 Derivation Step 2: KL decay and squared-score objective

We now analyze how the KL divergence between two solutions of the same Fokker–Planck equation evolves in time.

Assume that both p(x,t) and q(x,t) satisfy the same Fokker–Planck equation with drift f(x,t) and diffusion strength g(t):

1 2

- 1

- 2

∂p ∂t

∂q ∂t

g(t)2∇2p,

g(t)2∇2q.

:= −∇ · fp +

:= −∇ · fq +

Define

- p(x,t)

- q(x,t)

KL pt∥qt := p(x,t) log

dx.

Step 1: Differentiate the KL. Differentiating under the integral sign and using ∂tpdx = 0 (mass conservation), we obtain

d dt

- p

- q

KL pt∥qt := log

∂tpdx −

- p

- q

∂tq dx.

Introduce the Fokker–Planck operator

- 1

- 2

g(t)2∇2u,

Lu = −∇ · (fu) +

so that ∂tp = Lp and ∂tq = Lq. Let r = p/q. Then

d dt

KL pt∥qt := log r Lpdx − r Lq dx.

- Step 2: Drift does not change the KL. For the drift operator −∇ · (fu), integration by

parts (with vanishing boundary terms) gives

log r − ∇ · (fp) dx = pf · ∇log r dx,

r − ∇ · (fq) dx = q f · ∇r dx. Using r = p/q and ∇log r = ∇r/r, one checks

pf · ∇log r − q f · ∇r = 0, so the drift part cancels exactly and does not affect KL(pt∥qt).

- Step 3: Diffusion decreases the KL. For the diffusion operator 12g(t)2∇2u, integration by

parts yields

- 1

- 2

- 1

- 2

g(t)2∇2pdx = −

g(t)2 ∇log r · ∇pdx,

log r ·

- 1

- 2

- 1

- 2

g(t)2 ∇r · ∇q dx. Using

g(t)2∇2q dx = −

r ·

∇p = p∇log p, ∇q = q ∇log q, ∇r = ∇

- p

- q

= r ∇log p − ∇log q ,

we obtain

∇log r · ∇p := p ∇log p − ∇log q · ∇log p, ∇r · ∇q := p ∇log p − ∇log q · ∇log q. Subtracting these contributions gives

- 1

- 2

1 2

- 1

- 2

g(t)2 p(x,t) ∇log p − ∇log q 2 dx.

g(t)2 ∇log r · ∇pdx +

g(t)2 ∇r · ∇q dx := −

−

- Step 4: Conclusion. Putting drift and diffusion together,

- 1

- 2

d dt

g(t)2 p(x,t) ∇log p(x,t) − ∇log q(x,t) 2dx ≤ 0.

KL pt∥qt := −

Thus, along the forward diffusion process, the KL divergence between any two solutions of the same Fokker–Planck equation is non-increasing: diffusion strictly contracts KL (with equality only if the scores ∇log p and ∇log q coincide almost everywhere). This monotone decrease of KL(pt∥qt) justifies decomposing the global maximum-likelihood objective into local-in-time, squared-score terms associated with each diffusion step.

#### A.4 Derivation: equivalence between DSM and SM

We now prove that the denoising score matching (DSM) loss and the score matching (SM) loss at time t have the same minimizer.

Step 1: Define the two losses. Let us write the denoising score matching (DSM) loss at time t as

LDSM(sθ) := Ex0∼p0 Ext∼pt(·|x0) ∇xt log pt(xt | x0) − sθ(xt,t) 2, and the score matching (SM) loss on the marginal pt(xt) as

LSM(sθ) := Ext∼pt ∇xt log pt(xt) − sθ(xt,t) 2.

Here pt(xt) = pt(xt | x0)p0(x0)dx0 is the marginal of the forward process at time t. Step 2: Introduce conditional and marginal scores. Define the conditional score

s(xt | x0) := ∇xt log pt(xt | x0), and the marginal score

s(xt,t) := ∇xt log pt(xt).

##### Step 3: Expand both objectives. Using ∥a − b∥2 = ∥a∥2 + ∥b∥2 − 2⟨a,b⟩, we can expand

both objectives. For DSM,

LDSM(sθ) = Ex0,xt sθ(xt,t) 2 − 2Ex0,xt sθ(xt,t),s(xt | x0)

+ Ex0,xt s(xt | x0) 2, where expectations are taken under the joint p0(x0)pt(xt | x0). Similarly, for SM we have LSM(sθ) = Ext sθ(xt,t) 2 − 2Ext sθ(xt,t),s(xt,t)

+ Ext s(xt,t) 2.

- Step 4: Match the first and last terms. The first terms coincide, because the marginal of

the joint distribution is exactly pt(xt):

Ex0,xt sθ(xt,t) 2 = pt(xt) sθ(xt,t) 2 dxt = Ext sθ(xt,t) 2. The last terms,

Ex0,xt∥s(xt | x0)∥2 and

Ext∥s(xt,t)∥2,

do not depend on sθ at all, so they can only shift the loss by a constant.

Step 5: Handle the cross term. The only subtle point is the cross term. Because the inner product is linear, it is enough to prove that, for any (scalar) test function f(xt),

Ex0,xt f(xt)s(xt | x0) = Ext f(xt)s(xt,t) , and then apply this to each coordinate of sθ(xt,t).

By definition of the score,

Therefore,

s(xt | x0) := ∇xt log pt(xt | x0) := ∇xtpt(xt | x0)

.

pt(xt | x0)

Ex0,xt f(xt)s(xt | x0) = p0(x0)pt(xt | x0)f(xt) ∇xtpt(xt | x0)

dxt dx0

pt(xt | x0)

= f(xt)∇xtpt(xt | x0)p0(x0) dxt dx0.

Under mild regularity conditions we can interchange the order of integration and differentiation, obtaining

Ex0,xt f(xt)s(xt | x0) = f(xt)∇xt pt(xt | x0)p0(x0)dx0 dxt

= f(xt)∇xtpt(xt)dxt

= pt(xt)f(xt)∇xt log pt(xt)dxt

= Ext f(xt)s(xt,t) .

Taking f(xt) to be each component of sθ(xt,t) shows that the DSM and SM cross terms are identical:

Ex0,xt sθ(xt,t),s(xt | x0) = Ext sθ(xt,t),s(xt,t) . Conclusion. Putting everything together, we have

LDSM(sθ) := LSM(sθ) + C,

where C is a constant independent of sθ. Hence both objectives are minimized by the same function, namely the true marginal score

s⋆θ(xt,t) = ∇xt log pt(xt).

### References

- [1] Brian D. O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 1982. URL https://doi.org/10.1016/0304-4149(82)90051-5.
- [2] Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin Patrick Murphy, and Tim Salimans. Diffusion models and gaussian flow matching: Two sides of the same coin. In The Fourth Blogpost Track at ICLR 2025, 2025. URL https://openreview.net/forum? id=C8Yyg9wy0s.
- [3] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239, 2020. URL https://arxiv.org/abs/2006.11239.
- [4] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364, 2022. URL https: //arxiv.org/abs/2206.00364.
- [5] Paul Langevin. Sur la théorie du mouvement brownien. Comptes Rendus de l’Académie des Sciences, 146:530–533, 1908.
- [6] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. URL https: //arxiv.org/abs/2209.03003.
- [7] Calvin Luo. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970, 2022. URL https://arxiv.org/abs/2208.11970.
- [8] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 2019. URL https://arxiv. org/abs/1907.05600.
- [9] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. URL https://arxiv.org/abs/2011.13456.
- [10] Candi Zheng, Yuan Lan, and Yang Wang. Lanpaint: Training-free diffusion inpainting with asymptotically exact and fast conditional sampling. Transactions on Machine Learning Research,

2025. URL https://openreview.net/forum?id=JPC8JyOUSW.

