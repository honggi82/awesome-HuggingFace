# arXiv:2410.10792v1[cs.LG]14Oct2024

## SEMANTIC IMAGE INVERSION AND EDITING USING RECTIFIED STOCHASTIC DIFFERENTIAL EQUATIONS

### Litu Rout1,2, Yujia Chen1, Nataniel Ruiz1, Constantine Caramanis2, Sanjay Shakkottai2, Wen-Sheng Chu1 1 Google, 2 UT Austin

{litu.rout,constantine,sanjay.shakkottai}@utexas.edu {yujiachen,natanielruiz,wschu}@google.com

|[Figure 1]|
|---|

|[Figure 2]|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(a) Ref. style “a girl” “a panda”

(b) Ref. style “face of a boy”

“a dwarf”

|[Figure 7]|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

(c) Ref. content “sleeping cat” “silver cat sculpture” “origami cat” “lion” “tiger”

|[Figure 13]|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

(d) Ref. content “smiling cartoon” “angry cartoon” “girl” “old man” “young boy+glasses”

Figure 1: Rectified flows for image inversion and editing. Our approach efficiently inverts reference style images in (a) and (b) without requiring text descriptions of the images and applies desired edits based on new prompts (e.g. “a girl” or “a dwarf”). For a reference content image (e.g. a cat in (c) or a face in (d)), it performs semantic image editing (e.g. “sleeping cat”) and stylization (e.g. “a photo of a cat in origmai style”) based on prompts, without leaking unwanted content from the reference image. Input images have orange borders.

ABSTRACT

Generative models transform random noise into images; their inversion aims to transform images back to structured noise for recovery and editing. This paper addresses two key tasks: (i) inversion and (ii) editing of a real image using stochastic equivalents of rectified flow models (such as Flux). Although Diffusion Models (DMs) have recently dominated the field of generative modeling for images, their inversion presents faithfulness and editability challenges due to nonlinearities in drift and diffusion. Existing state-of-the-art DM inversion approaches rely on training of additional parameters or test-time optimization of latent variables; both are expensive in practice. Rectified Flows (RFs) offer a promising alternative to diffusion models, yet their inversion has been underexplored. We propose RF inversion using dynamic optimal control derived via a linear quadratic regulator. We prove that the resulting vector field is equivalent to a rectified stochastic differential equation. Additionally, we extend our framework to design a stochastic sampler for Flux. Our inversion method allows for state-of-the-art performance in zero-shot inversion and editing, outperforming prior works in stroke-to-image synthesis and semantic image editing, with large-scale human evaluations confirming user preference.

- 1 INTRODUCTION

Vision generative models typically transform noise into images. Inverting such models, given a reference image, involves finding the structured noise that can regenerate the original image. Efficient inversion must satisfy two crucial properties. First, the structured noise should produce an image that is faithful to the reference image. Second, the resulting image should be easily editable using new prompts, allowing fine modifications over the image.

Diffusion Models (DMs) have become the mainstream approach for generative modeling of images (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020), excelling at sampling from high-dimensional distributions (Ramesh et al., 2021; Saharia et al., 2022; Ramesh et al., 2022; Rombach et al., 2022; Podell et al., 2023; Pernias et al., 2024). The sampling process follows a Stochastic Differential Equation known as reverse SDE (Anderson, 1982; Efron, 2011; Song et al.,

- 2021b). Notably, these models can invert a given image. Recent advances in DM inversion have shown a significant impact on conditional sampling, such as stroke-to-image synthesis (Meng et al.,
- 2022), image editing (Hertz et al., 2022; Mokady et al., 2023; Couairon et al., 2023; Rout et al.,
- 2023a;b; 2024a; Delbracio & Milanfar, 2023) and stylization (Hertz et al., 2023; Rout et al., 2024b).

Despite its widespread usage, DM inversion faces critical challenges in faithfulness and editability. First, the stochastic nature of the process requires fine discretization of the reverse SDE (Ho et al., 2020; Song et al., 2021b), which increases expensive Neural Function Evaluations (NFEs). Coarse discretization, on the other hand, leads to less faithful outputs (Meng et al., 2022), even with deterministic methods like DDIM (Song et al., 2021a;b). Second, nonlinearities in the reverse trajectory introduce unwanted drift, reducing the accuracy of reconstruction (Karras et al., 2024). While existing methods enhance faithfulness by optimizing latent variables (Rout et al., 2024a) or prompt embeddings (Mokady et al., 2023; Miyake et al., 2023), they tend to be less efficient, harder to edit, and rely on complex attention processors to align with a given prompt (Hertz et al., 2022; Rout et al.,

- 2024a). These added complexities make such methods less suitable for real-world deployment.

For inversion and editing, we introduce a zero-shot conditional sampling algorithm using Rectified Flows (RFs) (Liu et al., 2022; Albergo & Vanden-Eijnden, 2023; Lipman et al., 2022; Esser et al., 2024), a powerful alternative to DMs. Unlike DMs, where sampling is governed by a reverse SDE, RFs use an Ordinary Differential Equation known as reverse ODE, offering advantages in both efficient training and fast sampling. We construct a controlled forward ODE, initialized from a given image, to generate the initial conditions for the reverse ODE. The reverse ODE is then guided by an optimal controller, obtained through solving a Linear Quadratic Regulator (LQR) problem. We prove that the resulting new vector fields have a stochastic interpretation with an appropriate drift and diffusion. We evaluate RF inversion on stroke-to-image generation and image editing tasks, and show extensive qualitative results on other applications like cartoonization. Our method significantly improves photo realism in stroke-to-image generation, surpassing a state-of-the-art (SoTA) method (Mokady et al., 2023) by 89%, while maintaining faithfulness to the input stroke. In addition, we show that RF inversion outperforms DM inversion (Meng et al., 2022) in faithfulness by 4.7% and in realism by 13.8% on LSUN-bedroom dataset (Wang et al., 2017). Figure 1 and Figure 2 show the qualitative results of our approach and a graphical illustration, respectively.

Our theoretical and practical contributions can be summarized as:

- • We present an efficient inversion method for RF models, including Flux, that requires no additional training, latent optimization, prompt tuning, or complex attention processors.
- • We develop a new vector field for RF inversion, interpolating between two competing objectives: consistency with a possibly corrupted input image, and consistency with the “true” distribution of clean images (§3.3). We prove that this vector field is equivalent to a rectified SDE that interpolates between the stochastic equivalents of these competing objectives (§3.4). We extend the theoretical results to design a stochastic sampler for Flux.
- • We demonstrate the faithfulness and editability of RF inversion across three benchmarks: (i) LSUN-Bedroom, (ii) LSUN-Church, and (iii) SFHQ, on two tasks: stroke-to-image synthesis and image editing. In addition, we provide extensive qualitative results and conduct large-scale human evaluations to assess user preference metrics (§5).

- 2 RELATED WORKS

DM Inversion. Diffusion models have become the mainstream approach for generative modeling, making DM inversion an exciting area of research (Meng et al., 2022; Couairon et al., 2023; Song

Figure 2: Graphical model illustrating (a) DDIM inversion and (b) RF inversion. Due to nonlinearities in DM trajectory, the DDIM inverted latent x1 significantly deviates from the original image y0. RF inversion without controller reduces this deviation, resulting in x1. With controller, RF inversion further eliminates the reconstruction error, making x1 nearly identical to y0, which enhances the faithfulness.

𝑦 𝑦 𝑦 𝑦

𝑦 𝑦

𝑦

𝑦

𝑥

𝑥

𝑥

𝑥

𝑥

𝑥 𝑥̅

𝑥

𝑥

Fwd. latent DDIM step Rev. latent

𝑥 Fwd. latent

Rev. latent w/o controller

Rev. latent w/ controller

𝑥

(a) DDIM Inversion (b) RF Inversion

- et al., 2021b; Hertz et al., 2023; Mokady et al., 2023; Rout et al., 2024a). Among training-free methods, SDEdit (Meng et al., 2022) adds noise to an image and uses the noisy latent as structured noise. For semantic image editing based on a given prompt, it simulates the standard reverse SDE starting from this structured noise. SDEdit requires no additional parameter training, latent variable optimization, or complex attention mechanisms. However, it is less faithful to the original image because adding noise in one step is equivalent to linear interpolation between the image and noise, while the standard reverse SDE follows a nonlinear path (Liu et al., 2022; Karras et al., 2022).

An alternate method, DDIM inversion (Song et al., 2021a;b), recursively adds predicted noise at each forward step and returns the final state as the structured noise (illustrated by Yt process in Figure 2(a)). However, DDIM inversion often deviates significantly from the original image due to nonlinearities in the drift and diffusion coefficients, as well as inexact score estimates (Mokady et al., 2023). To reduce this deviation, recent approaches optimize prompt embeddings (Mokady et al., 2023) or latent variables (Rout et al., 2024a), but they have high time complexity. Negative prompt inversion (Miyake et al., 2023) speeds up the inversion process but sacrifices faithfulness. Methods like CycleDiffusion (Wu & De la Torre, 2023) and Direction Inversion (Ju et al., 2023) use inverted latents as references during editing, but they are either computationally expensive or not applicable to rectified flow models like Flux or SD3 (Esser et al., 2024).

DM Editing. Efficient inversion is crucial for real image editing. Once a structured noise is obtained by inverting the image, a new prompt is fed into the T2I generative model. Inefficient inversion often fails to preserve the original content and therefore requires complex editing algorithms. These editing algorithms can be broadly classified into (i) attention control, such as prompt-to-prompt (Hertz

- et al., 2022), plug-and-play (PnP) (Tumanyan et al., 2023), (ii) optimization-based methods like DiffusionCLIP (Kim et al., 2022), DiffuseIT (Kwon & Ye, 2023), STSL (Rout et al., 2024a), and (iii) latent masking to edit specific regions of an image using masks provided by the user (Nichol et al.,

- 2022) or automatically extracted from the generative model (Couairon et al., 2023). We focus on efficient inversion, avoiding the need for complex editing algorithms.

Challenges in RF Inversion. Previous inversion or editing approaches have been tailored towards diffusion models and do not directly apply to SoTA rectified flow models like Flux. This limitation arises because the network architecture of Flux is MM-DiT (Peebles & Xie, 2023), which is fundamentally different from the traditional UNet used in DMs (Ho et al., 2020; Song et al., 2021a;b). In MM-DiT, text and image information are entangled within the architecture itself, whereas in UNet, text conditioning is handled via cross-attention layers. Additionally, Flux primarily uses T5 text encoder, which lacks an aligned latent space for images, unlike CLIP encoders. Therefore, extending these prior methods to modern T2I generative models requires a thorough investigation. We take the first step by inverting and editing a given image using Flux.

RF Inversion and Editing. DMs (Ho et al., 2020; Song et al., 2021a; Rombach et al., 2022) traditionally outperform RFs (Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2023) in high-resolution image generation. However, recent advances have shown that RF models like Flux can surpass SoTA DMs in text-to-image (T2I) generation tasks (Esser et al., 2024). Despite this, their inversion and editing capabilities remain underexplored. In this paper, we introduce an efficient RF inversion method that avoids the need for training additional parameters (Hu et al., 2021; Ruiz et al., 2023), optimizing latent variables (Rout et al., 2024a), prompt tuning (Mokady et al.,

- 2023), or using complex attention processors (Hertz et al., 2022). While our focus is on inversion and editing, we also show that our framework can be easily extended to generative modeling.

Filtering, Control and SDEs. There is a rich literature on the connections between nonlinear filtering, optimal control and SDEs (Fleming & Rishel, 1975; Øksendal, 2003; Tzen & Raginsky, 2019; Zhang & Chen, 2022). These connections are grounded in the Fokker-Planck equation (Øksendal, 2003), which RF methods (Lipman et al., 2022; Liu et al., 2022; Albergo & Vanden-Eijnden, 2023;

Albergo et al., 2023) heavily exploit in sampling. Our study focuses on rectified flows for conditional sampling, and shows that the resulting drift field also has an optimal control interpretation.

- 3 METHOD

- 3.1 PRELIMINARIES

In generative modeling, the goal is to sample from a target distribution p0 given a finite number of samples from that distribution. Rectified flows (Lipman et al., 2022; Liu et al., 2022) represent

a class of generative models that construct a source distribution q0 and a time varying vector field vt(xt) to sample p0 using an ODE:

dXt = vt(Xt)dt, X0 ∼ q0, t ∈ [0,1]. (1)

Starting from X0 = x0, the ODE (1) is integrated from t : 0 → 1 to yield a sample x1 distributed according to p0 (e.g., the distribution over images). A common choice of q0 is standard Gaussian N (0,I) and vt (Xt) = −u(Xt,1 − t;φ), where u is a neural network parameterized by φ. The neural network is trained using the conditional flow matching objective as discussed below.

Training Rectified Flows. To train a neural network to serve as the vector field for the ODE (1), we couple samples from p0 with samples from q0 – which we call p1 to simplify the notation – via a linear path: Yt = tY1 + (1 − t)Y0. The resulting marginal distribution of Yt becomes:

pt(yt) = EY

1∼p1 [pt(yt|Y1)] = pt(yt|y1)p1(y1)dy1. (2)

Given an initial state Y0 = y0 and a terminal state Y1 = y1, the linear path induces an ODE: dYt = ut (Yt|y1)dt with the conditional vector field ut (Yt|y1) = y1 − y0. The marginal vector field is derived from the conditional vector field using the following relation (Lipman et al., 2022):

ut(yt) = EY

1∼p1 ut (yt|Y1)

pt(yt|Y1) pt(yt)

= ut (yt|y1)

pt(yt|y1) pt(yt)

p1(y1)dy1. (3)

We can then use a neural network u(yt,t;φ), parameterized by φ, to approximate the marginal vector field ut(yt) through the flow matching objective defined as:

LFM(φ) := Et∼U[0,1],Y

t∼pt ∥ut(Yt) − u(Yt,t;φ)∥22 . (4) For tractability, we can instead consider a different objective, called conditional flow matching:

LCFM(φ) := Et∼U[0,1],Y

t∼pt(·|Y1),Y1∼p1 ∥ut(Yt|Y1) − u(Yt,t;φ)∥22 . (5)

LCFM and LFM have the identical gradients (Lipman et al., 2022, Theorem 2), and are hence equivalent. However, LCFM(φ) is computationally tractable, unlike LFM(φ), and therefore preferred during training. Finally, the required vector field in (1) is computed as vt (Xt) = −u(Xt,1 − t;φ). In this way, rectified flows sample a data distribution by an ODE with a learned vector field.

- 3.2 CONNECTION BETWEEN RECTIFIED FLOWS AND LINEAR QUADRATIC REGULATOR

The unconditional rectified flows (RFs) (e.g., Flux) from Section §3.1 above, enable image generation by simulating the vector field vt(·) initialized with a sample of random noise. Subsequently, by simulating the reversed vector field −v1−t(·) starting from the image, we get back the sample of noise that we started with. We formalize this statement below.

- Proposition 3.1. Given an image y0 and the vector field vt(·) of the generative ODE (1), suppose the structured noise y1 is obtained by simulating an ODE:

dYt = ut(Yt)dt, Y0 = y0, t ∈ [0,1]. (6) If ut(·) = −v1−t(·) and X0 = y1, then the ODE (1) recovers the original image, i.e., X1 = y0.

Implication. Rectified flows enable exact inversion of a given image when the vector field of the generative ODE (1) is precisely known. Employing ODE (6) for the structured noise and ODE (1) to transform that noise back into an image, RF inversion accurately recovers the given image.

Suppose instead that we start with a corrupted image and simulate the reversed vector field −v1−t(·). Then we obtain a noise sample. There are two salient aspects of this noise sample. First, it is

consistent with the original image: when processed through vt(·) it results in the same corrupted image. Second, if the image sample is “atypical” (e.g., corrupted, or, say, a stroke painting as in §5), then the sample of noise is also likely to be atypical. In other words, the noise sample is only consistent to the (possibly corrupted) image sample.

Our goal is to modify the pipeline above so that even when we start with a corrupted image, we can get back a clean image (see stroke-to-image synthesis in Figure 5), but for this, we need to processs by vt(·) a noise sample that is closer to being “typical”. More generally, the goal is to create a pipeline that supports semantic editing of real images (§5), e.g., changing age, or gender without relying on additional training, optimization, or complex attention processors.

Thus, as a first step, we derive an optimal controller that takes a minimum energy path to convert any image Y0 (whether corrupted or not) to a given sample of random noise Y1 ∼ p1 – i.e., noise that is typical for p1. Specifically, we consider optimal control in a d-dimensional vector space Rd:

1

- 0
- 1

- 2 ∥c(Zt,t)∥22 dt +

λ 2 ∥Z1 − Y1∥22 , dZt = c(Zt,t)dt, Z0 = y0, Y1 ∼ p1, (7)

V (c) :=

where λ is the weight assigned to the terminal cost and V (c) denotes the total cost of the control c : Rd × [0,1] → Rd. The minimization of V (c) over the admissible set of controls, denoted by C, is known as the Linear Quadratic Regulator (LQR) problem. The solution of the LQR problem (7) is given in Proposition 3.2, which minimizes the quadratic transport cost of the dynamical system.

- Proposition 3.2. For Z0 = y0 and Y1 = y1, the optimal controller of the LQR problem (7), denoted by c∗ (·,t) is equal to the conditional vector field ut (·|y1) of the rectified linear path Yt = tY1 + (1 − t)Y0 when Y0 = y0, i.e., c∗ (zt,t) = ut (zt|y1) = (y1 − zt)/(1 − t).
- 3.3 INVERTING RECTIFIED FLOWS WITH DYNAMIC CONTROL

So far, we have two vector fields. The first, from the RFs, transforms an image Y0 typical for distribution p0 to a typical sample of random Gaussian noise Y1 ∼ p1. As discussed above, if the image sample is atypical, then the sample of noise is also likely to be atypical.

We also have a second vector field resulting from the optimal control formulation that transforms any image (whether corrupted or not) to a noise sample that is typical-by-design from the distribution p1. Therefore, this sample, when passed through the rectified flow ODE (1) results in a “typical” image from the “true” distribution p0. This image is clean, i.e., typical for p0, but it is not related to the image Y0. Our controlled ODE, defined below, interpolates between these two differing objectives – consistency with the given (possibly corrupted) image, and consistency with the distribution of images p0 – with a tunable parameter γ:

#### dYt = ut(Yt) + γ (ut(Yt|y1) − ut(Yt)) dt, Y0 = y0, (8)

where ut(Yt|y1) = c∗(Yt,t) is computed based on the insights from Proposition 3.2, and ut(Yt) = −v1−t(Yt) as established in Proposition 3.1. Here, we call γ ∈ [0,1] the controller guidance. Thus, ODE (8) generalizes (6) to editing applications, while keeping its inversion accuracy comparable.

When γ = 1, the drift field of the ODE (8) becomes optimal controller of LQR problem (7), ensuring that the structured noise Y1 = y1 adheres to the distribution p1. Consequently, initializing the generative ODE (1) with y1 results in samples with high likelihood under the data distribution p0.

Conversely, when γ = 0, the system follows the ODE (6) described in Proposition 3.1, resulting a structured noise Y1 that is not guaranteed to follow the noise distribution p1. However, initializing the generative ODE (1) with this noise precisely recovers the reference image y0.

Beyond this vector field interpolation intuition, we show in the next section §3.4 that the controlled ODE (8) has an SDE interpretation. As is well known (Ho et al., 2020; Song et al., 2021a; Meng

- et al., 2022; Song et al., 2021b), SDEs are robust to initial conditions, in proportion to the variance of the additive noise. Specifically, errors propagate over time in an ODE initialized with an incorrect or corrupted sample. However, SDEs (Markov processes) under appropriate conditions converge to samples from a carefully constructed invariant distribution with reduced sensitivity to the initial condition, resulting in a form of robustness to initialization. As we see, the parameter γ (the controller

guidance) appears in the noise term to the SDE, thus the SDE analysis in the next section again provides intuition on the trade-off between consistency to the (corrupted) image and consistency to the terminal invariant distribution.

Remark 3.3. We note that our analysis extends to the case where γ is time-varying, though we omit these results for simplicity of notation. This is useful in practice, especially when y0 is a corrupted image, because for large γ the stochastic evolution (22) moves toward a sample from the invariant measure N (0,I). This noise encodes clean images. Starting from this noise, the corresponding reverse process operates in pure diffusion mode, resulting in a clean image. As the process approaches the terminal state, γ is gradually reduced to ensure that y0 is encoded through ut(·) into the final structured noise sample.

- 3.4 CONTROLLED RECTIFIED FLOWS AS STOCHASTIC DIFFERENTIAL EQUATIONS

An SDE (Ho et al., 2020) is known to have an equivalent ODE formulation (Song et al., 2021a) under certain regularity conditions (Anderson, 1982; Song et al., 2021b). In this section, we derive the opposite: an SDE formulation for our controlled ODE (8) from §3.3. Let Wt be a d-dimensional Brownian motion in a filtered probability space (Ω,F,{Ft},P).

- Theorem 3.4. Fix any T ∈ (0,1). For any t ∈ [0,T], the controlled ODE (8) is explicitly given by:

(1 − γ)t

1 1 − t

1 − t ∇log pt(Yt) dt, Y0 ∼ p0. (9) Its density evolution is identical to the density evolution of the following SDE:

dYt = −

(Yt − γy1) −

2(1 − γ)t 1 − t

1 1 − t

dWt, Y0 ∼ p0. (10) Finally, denoting pt(·) as the marginal pdf of Yt, the density evolution is explicitly given by:

dYt = −

(Yt − γy1)dt +

(1 − γ)t 1 − t ∇log pt(Yt) pt(Yt) . (11)

1 1 − t

∂pt(Yt) ∂t

= ∇ ·

(Yt − γy1) +

Properties of SDE (10). Elaborating on the intuition discussed at the end of §3.3, when the controller guidance parameter γ = 0, it becomes the stochastic equivalent of the standard RFs; see Lemma A.2 for a precise statement. The resulting SDE is given by

1 1 − t

2t 1 − t

dWt, Y0 ∼ p0, (12)

dYt = −

Ytdt +

which improves faithfulness to the image Y0. When γ = 1, the SDE (10) solves the LQR problem (7) and drives towards the terminal state Y1 = y1. This improves the generation quality, because the sample Y1 is from the correct noise distribution p1 as previously discussed in §3.3. Therefore, a suitable choice of γ retains faithfulness while simultaneously applying the desired edits.

Finally, we assume T = 1 − δ for sufficiently small δ (such that 0 < δ ≪ 1) to avoid irregularities at the boundary. This is typically considered in practice for numerical stability (even for diffusion models). Thus, in practice, the final sample y1−δ is returned as y1.

Comparison with DMs. Analogous to the SDE (12), the stochastic noising process of DMs is typically modeled by the Ornstein-Uhlenbeck (OU) process, governed by the following SDE:

√

2dWt. (13) The corresponding ODE formulation is given by:

dYt = −Ytdt +

#### dYt = [−Yt − ∇log pt(Yt)]dt. (14)

Instead, our approach is based on rectified flows (1), which leads to a different ODE and consequently translates into a different SDE. As an additional result, we formalize the ODE derivation in Lemma A.1. In Lemma A.2, we show that the marginal distribution of this ODE is equal to that of an SDE with appropriate drift and diffusion terms. In Proposition A.3, we show that the stationary distribution of this new SDE (12) converges to the standard Gaussian N(0,I) in the limit as t → 1.

The standard OU process (13) interpolates between the data distribution at time t = 0 and a standard Gaussian as t → ∞. The SDE (12), however, interpolates between the data distribution at time t = 0 and a standard Gaussian at t = 1. In other words, it effectively “accelerates” time as it progresses to achieve the terminal Gaussian distribution. This is accomplished by modifying the coefficients of drift and diffusion as in (12) to depend explicitly on time t. Thus, a sample path of (12) appears like a noisy line, unlike that of the OU process (see Appendix C.3 for numerical simulations).

- 3.5 CONTROLLED REVERSE FLOW USING RECTIFIED ODES AND SDES

In this section, we develop an ODE and an SDE similar to our discussions above, but for the reverse direction (i.e., from noise to images).

Reverse process using ODE. Starting from the structured noise y1 obtained by integrating the controlled ODE (8), we construct another controlled ODE (15) for the reverse process (i.e., noise to

image). In this process, the optimal controller uses the reference image y0 for guidance:

#### dXt = vt(Xt) + η (vt(Xt|y0) − vt(Xt)) dt, X0 = y1, t ∈ [0,1], (15)

where η ∈ [0,1] is the controller guidance parameter as before that controls faithfulness and editability of the given image y0. Similar to the analysis in Proposition 3.2, vt(Xt|y0) is obtained by solving the modified LQR problem (16):

V (c) =

1

- 0
- 1

- 2 ∥c(Zt,t)∥22 dt +

λ 2 ∥Z1 − y0∥22 , dZt = c(Zt,t)dt, Z0 = y1. (16)

- 0−Zt

- 1−t . Our controller steers the samples toward the given image

Solving (16), we get c(Zt,t) = y y0. Thus, the controlled reverse ODE (15) effectively reduces the reconstruction error incurred in the standard reverse ODE (1) of RF models (e.g. Flux).

Reverse process using SDE. Finally, in Theorem 3.5, we provide the stochastic equivalent of our controlled reverse ODE (15) for generation. Recall that we initialize with the terminal structured noise by running the controlled forward ODE (8), along with a reference image y0. As discussed above, we terminate the inversion process at a time T = 1 − δ for numerical stability, resulting in a vector y1−δ. Our reverse SDE thus starts at a corresponding time δ with this vector y1−δ at initialization, and terminates at time T′ < 1.

- Theorem 3.5. Fix any T′ ∈ (δ,1), and for any t ∈ [δ,T′], the density evolution of the controlled ODE (15) initialized at X0 = y1−δ is identical to the density evolution of the following SDE:

(1 − t − η)Xt + ηty0 t(1 − t)

2(1 − t)(1 − η) t ∇log p1−t(Xt) dt +

2(1 − t)(1 − η) t

+

dWt.

dXt =

(17) Furthermore, denoting qt(·) as the marginal pdf of Xt, its density evolution is given by:

1 − t − η t(1 − t)

(1 − t) t

∂qt(Xt) ∂t

η 1 − t

(1 − η)∇log p1−t(Xt) qt(Yt) . (18)

= ∇ · −

Xt +

y0 +

Properties of SDE (17). When the controller parameter η = 0, we obtain a stochastic sampler (22) for the pre-trained Flux, as given in Lemma A.4 and compared qualitatively in Figure 24. This case of our SDE (17) corresponds to the stochastic variant of standard RFs (Liu et al., 2022; Lipman et al., 2022; Albergo & Vanden-Eijnden, 2023). Our key contribution lies in conditioning on X1 = y0 for inverting rectified flows. Importantly, our explicit construction does not require additional training or test-time optimization, enabling for the first time an efficient sampler for zeroshot inversion and editing using Flux. When η = 1, the score term and Brownian motion vanish from the SDE (17). The resulting drift becomes y

- 0−Xt

- 1−t , the optimal controller for the LQR problem (16),

exactly recovering the given image y0.

Remark 3.6. Similar to Remark 3.3, our analysis extends to the case when η is time-varying. This is useful in editing, as it allows the flow to initially move toward the given image y0 by choosing a large η. As the flow approaches y0 on the image manifold, η is gradually reduced, ensuring that the text-guided edits are enforced through the unconditional vector field vt(·) provided by Flux.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

0 0.2 0.4 0.6 0.8 Original

- Figure 4: Effect of controller guidance η given the original image and the prompt: “A young man”. Increasing η improves the faithfulness to the original image, which is reconstructed at η=1.

- 4 ALGORITHM: INVERSION AND EDITING VIA CONTROLLED ODES We describe the algorithm for RF inversion and editing using our controlled ODEs (8) and (15).

Problem Setup. The user provides a text “prompt” to edit reference content, which could be a corrupt or a clean image. For the corrupt image guide, we use the dataset from SDEdit (Meng et al., 2022), which contains color strokes to convey high-level details. In this setting, the reference guide

- y0 is typically not a realistic image under the data distribution p0. The objective is to transform this guide into a more realistic image under p0 while maintaining faithfulness to the original guide.

For the clean image guide, the user provides a real image y0 along with an accompanying text “prompt” to specify the desired edits. The task is to apply text-guided edits to y0 while preserving its content. Examples include face editing, where the text might instruct change in age or gender.

Procedure. Our algorithm has two key steps: inversion and editing. We discuss each step below.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Original RF inversion

Figure 3: Inverting flows by controlled ODEs (8) and (15).

Inversion. The first step involves computing the structured noise

- y1 by employing our controlled ODE (8), initialized at the reference content Y0 = y0. To compute the unconditional vector field, we use the pre-trained Flux model u(·,·,·;φ), which requires

three inputs: the state Yt, the time t, and the prompt embedding Φ(prompt). During the inversion process, we use null prompt in the

Flux model, i.e., ut(yt) = u(yt,t,Φ(“”);φ). For the conditional vector field, we apply the analytical solution derived in Proposition 3.2. The inversion process yields a latent variable that is then used to initialize our controlled ODE (15), i.e., X0 = y1. In this phase, we again use the null prompt to compute the vector field vt(xt) = −u(xt,1 − t,Φ(“”);φ): see Figure 3 for the final output. Editing. The second step involves text-guided editing of the reference content y0. This process is governed by our controlled ODE (15), where the vector field is computed using the desired text prompt within Flux: vt(Xt) = −u(xt,1 − t,Φ(prompt);φ). The controller guidance η in (15) balances faithfulness and editability: higher η improves faithfulness but limits editability, while lower η allows significant edits at the cost of reduced faithfulness. Consequently, the controller guidance η provides a smooth interpolation between faithfulness and editability, a crucial feature in semantic image editing. Motivated by Remark 3.3 and 3.6, we consider a time-varying controller guidance ηt, such that for a fixed η ∈ [0,1] and τ ∈ [0,1], ηt = η ∀t ≤ τ and 0 otherwise. Figure 4 illustrates the effect of controller guidance η for τ = 0.3; see Appendix C.2 for a detailed ablation study.

- 5 EXPERIMENTAL EVALUATION

We show that RF inversion outperforms DM inversion across three benchmarks: LSUN-church, LSUN-bedroom (Wang et al., 2017), and SFHQ (Beniaguev, 2022) on two tasks: Stroke2Image generation and semantic image editing. Stroke2Image generation shows the robustness of our algorithm to initial corruption. In semantic image editing, we emphasize the ability to edit clean images without additional training, optimization, or complex attention processors.

Baselines. As this paper focuses on inverting flows, we compare with SoTA inversion approaches, such as NTI (Mokady et al., 2023), DDIM Inversion (Song et al., 2021a), and SDEdit (Meng et al., 2022). We use the official NTI implementation for both NTI and DDIM inversion, and Diffusers

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

Input SDEdit DDIM Inversion NTI NTI+P2P Ours

- Figure 5: Stroke2Image generation. Our method generates photo-realistic images of bedroom or church given stroke paints, showing robustness to initial corruptions.

[Figure 41]

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

Original SDEdit DDIM Inversion NTI NTI+P2P Ours

- Figure 6: Image editing for adding face accessories. Prompt: “face of a man/woman wearing glasses”. The proposed method better preserves the identity while applying the desired edits.

library for SDEdit. Hyper-parameters for all these baselines are tuned for optimal performance. We compare with NTI for both direct prompt change and with prompt-to-prompt Hertz et al. (2022) editing. All methods are training-free; however, NTI (Mokady et al., 2023) solves an optimization problem at each denoising step during inversion and uses P2P (Hertz et al., 2022) attention processor during editing. We follow the evaluation protocol from SDEdit (Meng et al., 2022). More qualitative results and comparison are in Appendix §C.

Stroke2Image generation. As discussed in §4, our goal is to generate a photo-realistic image from a stroke paint (a corrupted image) and the text prompt “photo-realistic picture of a bedroom”. In this case, the high level details in the stroke painting guide the reverse process toward a clean image.

In Figure 5, we compare RF inversion (ours) with DM inversions. DM inversions propagate the corruption from the stroke painting into the structured noise, which leads to outputs resembling the input stroke painting. NTI optimizes null embeddings to align the reverse process with the DDIM forward trajectory. Although adding P2P to the NTI pipeline helps localized editing as in Figure 6, for corrupted images, it drives the reverse process even closer to the corruption. In contrast, our controlled ODE (8) yields a structured noise that is consistent with the corrupted image and also the invariant terminal distribution, as discussed in §3.3, resulting in more realistic images.

- In Table 1, we show that our method outperforms prior works in faithfulness and realism. On the test split of LSUN bedroom dataset, our approach is 4.7% more faithful and 13.79% more realistic than the best optimization free method SDEdit-SD1.5. Ours is 73% more realistic than the optimizationbased method NTI, but comparable in L2. As discussed, NTI+P2P gets closer toward the corrupt image, which gives a very low L2 error, but the resulting image becomes unrealistic. Our approach is 89% more realistic than NTI+P2P. We observe similar gains on LSUN church dataset.

User study. We conduct a user study using Amazon Mechanical Turk to evaluate the overall performance of the our method. With 3 responses for each question, we collected in total 9,000 compar-

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

(a) (b)

Original “laugh” “angry” “A woman” “old” “older”

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

(c) (d) Original + “pepperoni” + “mushroom”

man à woman

- Figure 7: Editing (a) stylized expression, (b) age, (c) gender, and (d) object insert. Given an original image and a text prompt, our algorithm performs semantic image editing in the wild.

Table 1: Quantitative results for Stroke2Image generation. L2 and Kernel Inception Distance (KID) capture faithfulness and realism, respectively. Optimization-based methods are colored gray. User Pref. shows the percentage of users that prefer our method over each alternative in pairwise comparisons (and ties). E.g.: 62.11% (+ 8% ties) prefer ours over SDEdit-Flux for LSUN Bedroom.

LSUN Bedroom LSUN Church Method L2 ↓ KID ↓ User Pref. (%) ↑ L2 ↓ KID ↓ User Pref. (%) ↑

SDEdit-SD1.5 86.72 0.029 59.67 (5.33) 90.72 0.089 65.33 (4.11) SDEdit-Flux 94.89 0.032 62.11 (8.00) 92.47 0.081 66.22 (5.22) DDIM Inv. 87.95 0.113 82.56 (1.67) 97.36 0.107 85.44 (2.78) NTI 82.77 0.095 80.89 (4.33) 87.88 0.098 77.11 (4.89) NTI+P2P 46.46 0.234 98.11 (1.78) 34.48 0.168 99.22 (0.56) Ours 82.65 0.025 - 80.36 0.059 -

isons from 126 participants. As given in Table 1, our method outperforms all the other baselines by at least 59.67% in terms of overall satisfaction. More details are provided in Appendix §C.6.

Semantic Image Editing. Given a clean image and a text “prompt”, the objective is to modify the image according to the given text while preserving the contents of the image (identity for face images). In rectified linear paths, editing from a noisy latent becomes straightforward, further enhancing the efficiency of our approach. Compared with SoTA approaches (Figure 6), our method requires no additional optimization or complex attention processors as in NTI (Mokady

- et al., 2023)+P2P(Hertz et al., 2022). Thus, it is more efficient than a current SoTA approach, and importantly, more faithful to the original image while applying the desired edits.

- In Table 2, we show that our method outperforms the optimization-free methods by at least 29% in face reconstruction, 6.6% in DINO patch-wise similarity, and 26.4% in CLIPImage similarity while being comparable in prompt alignment metric CLIP-T. Importantly, our approach offers 54.11% gain in runtime, though it uses a larger (∼12X) model, while staying comparable to NTI+P2P.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

In Figure 7, we showcase four complex editing tasks: (a) prompt-based stylization with the prompt: “face of a boy in disney 3d cartoon style”, where facial expressions, such as “laugh” or “angry” are used for editing; (b) ability to control the age of a person; (c) interpolating between two concepts: “A man” ↔ “A woman”; (d) sequentially inserting pepperoni and mushroom to an image of a pizza. We provide more examples of editing in the wild in Appendix §C.

Original SDEdit (Flux) Flux Inversion Ours

Figure 8: Comparison using Flux backbone.

Comparison using the same backbone: Flux. In Figure 8, we compare our method with SDEdit and DDIM inversion both adapted to Flux. NTI optimizes null embeddings to align with forward latents before applying text-guided edits via P2P, an approach well-suited for DMs that use both

Table 2: Quantitative results for face editing on SFHQ for “wearing glasses”.

Method Face Rec. ↓ DINO ↑ CLIP-T ↑ CLIP-I ↑ Runtime(s) ↓

SDEdit-SD1.5 0.626 0.885 0.300 0.712 8 SDEdit-Flux 0.632 0.892 0.292 0.710 24 DDIM Inv. 0.709 0.884 0.311 0.669 15 NTI 0.707 0.876 0.304 0.666 78 NTI+P2P 0.443 0.953 0.293 0.845 85 Ours 0.442 0.951 0.300 0.900 39

null and text embedding. However, this strategy cannot be applied to Flux, as it does not explicitly use null embedding. Consequently, we only reimplement SDEdit and DDIM inversion for Flux and compare them to our method. Since all methods leverage the same generative model, the improvements clearly stem from our controlled ODEs, grounded in a solid theoretical foundation (§3).

- 6 CONCLUSION

We present the first efficient approach for inversion and editing with the state-of-art rectified flow models such as Flux. Our method interpolates between two vector fields: (i) the unconditional RF field that transforms a “clean” image to “typical” noise, and (ii) a conditional vector field derived from optimal control that transforms any image (clean or not) to “typical” noise. Our new field thus navigates between these two competing objectives of consistency with the given (possibly corrupted) image, and consistency with the distribution of clean images. Theoretically, we show that this is equivalent to a new rectified SDE formulation, sharing this intuition of interpolation. Practically, we show that our method results in state-of-art zero-shot performance, without the need of additional training, optimization of latent variables, prompt tuning, or complex attention processors. We demonstrate the effectiveness of our method in stroke-to-image synthesis, face editing, object insertion, and stylization tasks, with large-scale human evaluation confirming user preference.

Limitation. The lack of comparison with expensive diffusion-based editing solutions may be viewed as a limitation. However, these implementations are either not available for Flux or not directly applicable due to Flux’s distinct multi-modal architecture. The key contribution of this paper lies in its theoretical foundations, validated using standard benchmarks and relevant baselines.

Reproducibility. The pseudocode and hyper-parameter details have been provided to reproduce the reported results in this paper.

BROADER IMPACT STATEMENT

Semantic image inversion and editing have both positive and negative social impacts.

On the positive side, this technology enables (i) the generation of photo-realistic images from high level descriptions, such as stroke paintings, and (ii) the modification of clean images by changing various attributes like the age, gender, or adding glasses (§5).

On the negative side, it can be misused by malicious users to manipulate photographs of individuals with inappropriate or offensive edits. Additionally, it carries the inherent risks associated with the underlying generative model.

To mitigate the negative social impacts, we enable safety features such as NSFW filters in the underlying generative model. Furthermore, we believe watermarking images generated by this technology can reduce misuse, especially in inversion and editing applications.

ACKNOWLEDGMENTS

This research has been supported by NSF Grant 2019844, a Google research collaboration award, and the UT Austin Machine Learning Lab.

REFERENCES

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.

Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.

Tamer Basar, Sean Meyn, and William R Perkins. Lecture notes on control system theory and design. arXiv preprint arXiv:2007.01367, 2020.

David Beniaguev. Synthetic faces high quality (sfhq) dataset, 2022. URL https://github. com/SelfishGene/SFHQ-dataset.

Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8861–8870, 2024.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusionbased semantic image editing with mask guidance. In ICLR 2023 (Eleventh International Conference on Learning Representations), 2023.

Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. Transactions on Machine Learning Research, 2023.

Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Wendell H Fleming and Raymond W Rishel. Deterministic and stochastic optimal control, volume 1. Springer Science & Business Media, 1975.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2022.

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. arXiv preprint arXiv:2312.02133, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.

Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. Humansd: A native skeleton-guided diffusion model for human image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15988–15998, 2023.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24174–24184, 2024.

Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2426–2435, 2022.

Gihyun Kwon and Jong Chul Ye. Diffusion-based image translation using disentangled style and content representation. In The Eleventh International Conference on Learning Representations,

2023. URL https://openreview.net/forum?id=Nayau9fwXU. Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2022.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6038–6047, 2023.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pp. 16784–16804. PMLR, 2022.

Bernt Øksendal. Stochastic differential equations. Springer, 2003. William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of

the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville. W¨urstchen: An efficient architecture for large-scale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=gU58d5QeGv.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821–8831. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Litu Rout, Advait Parulekar, Constantine Caramanis, and Sanjay Shakkottai. A theoretical justification for image inpainting using denoising diffusion probabilistic models. arXiv preprint arXiv:2302.01217, 2023a.

Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alexandros G Dimakis, and Sanjay Shakkottai. Solving inverse problems provably via posterior sampling with latent diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b. URL https://openreview.net/forum?id=XKBFdYwfRo.

Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and WenSheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024a.

Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free personalization of diffusion models using stochastic optimal control. arXiv preprint arXiv:2405.17401, 2024b.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22500– 22510, 2023.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6527–6536, 2024.

Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 2256–2265, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/sohl-dickstein15.html.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https://openreview.net/ forum?id=St1giarCHLP.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/ forum?id=PxTIG12RRHS.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1921–1930, 2023.

Belinda Tzen and Maxim Raginsky. Theoretical guarantees for sampling and inference in generative models with latent diffusions. In Conference on Learning Theory, pp. 3084–3114. PMLR, 2019.

Limin Wang, Sheng Guo, Weilin Huang, Yuanjun Xiong, and Yu Qiao. Knowledge guided disambiguation for large-scale scene classification with multi-resolution cnns. IEEE Transactions on Image Processing, 26(4):2055–2068, 2017.

Chen Henry Wu and Fernando De la Torre. A latent space of stochastic diffusion models for zeroshot image editing and guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7378–7387, 2023.

Qinsheng Zhang and Yongxin Chen. Path integral sampler: A stochastic control approach for sampling. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=_uCb2ynRu7Y.

- A ADDITIONAL THEORETICAL RESULTS

In this section, we present the theoretical results omitted from the main draft due to space constraints. We formalize the ODE derivation of the standard rectified flows in Lemma A.1.

- Lemma A.1. Given a coupling (Y0,Y1) ∼ p0 × p1, consider the noising process Yt = tY1 + (1 − t)Y0. Then, the rectified flow ODE formulation with the optimal vector field is given by

dYt = −

1 1 − t

Yt −

t

1 − t∇log pt(Yt) dt, Y0 ∼ p0. (19) Furthermore, denoting pt(·) as the marginal pdf of Yt, its density evolution is given by:

∂pt(Yt) ∂t

= ∇ ·

1 1 − t

Yt +

t 1 − t ∇log pt(Yt) pt(Yt) . (20)

In Lemma A.2, we show that the marginal distribution of the rectified flow (6) is equal to that of an SDE with appropriate drift and diffusion terms.

- Lemma A.2. Fix any T ∈ (0,1), and for any t ∈ [0,T], the density evolution (20) of the rectified flow model (19) is identical to the density evolution of the following SDE:

1 1 − t

2t 1 − t

dWt, Y0 ∼ p0. (21)

dYt = −

Ytdt +

In Proposition A.3, we show that the stationary distribution of the SDE (21) converges to the standard Gaussian N(0,I) in the limit as t → 1.

Proposition A.3. Fix any T ∈ (0,1), and for any t ∈ [0,T], the density evolution for the rectified flow ODE (6) is same as that of the SDE (12). Furthermore, denoting pt(·) as the marginal pdf of Yt, its stationary distribution pt(Yt) ∝ exp(−∥Y

t∥2

2t ), which converges to N (0,I) as t → 1.

We note that Lemma A.1 and Lemma A.2 follow from the duality between the heat equation and the continuity equation (Øksendal, 2003), where it is classically known that one can interpret a diffusive term as a vector field that is affine in the score function, and vice-versa. This connection has been carefully used to study a large family of stochastic interpolants (that generalize rectified flows) in (Albergo & Vanden-Eijnden, 2023; Albergo et al., 2023), and which can lead to a family of ODE-SDE pairs. In the lemmas above, we have provided explicit coefficients that have been directly derived, instead of using the stochastic interpolant formulation. Our key contribution lies in constructing a controlled ODEs (8) and (15), along with their equivalent SDEs (10) and (17) in Theorem 3.4 and Theorem 3.5, respectively. This aids faithfulness and editability as discussed in §4.

In Lemma A.4, we derive a rectified SDE that transforms noise into images by reversing the stochastic equivalent of rectified flows (12).

Lemma A.4. Fix any small δ ∈ (0,1), and for any t ∈ [δ,1], the process Xt governed by the SDE:

2(1 − t) t ∇log p1−t(Xt) dt +

2(1 − t) t

1 t

dWt, X0 ∼ p1, (22) is the time-reversal of the SDE (12).

Xt +

dXt =

Implication. The reverse SDE (22) provides a stochastic sampler for SoTA rectified flow models like Flux. Unlike diffusion-based generative models that explicitly model the score function ∇log pt(·) in (22), rectified flows model a vector field, as discussed in §3.1. However, given a neural network u(yt,t;φ)) approximating the vector field ut(yt), Lemma A.1 offers an explicit formula for computing the score function:

1 − t t

1 t

u(Yt,t;φ). (23)

Yt −

∇log pt(Yt) = −

This score function is used to compute the drift and diffusion coefficients of the SDE (22), resulting in a practically implementable stochastic sampler for Flux. This extends the applicability of Flux to downstream tasks where SDE-based samplers have demonstrated practical benefits, as seen in diffusion models (Ho et al., 2020; Song et al., 2021b; Rombach et al., 2022; Podell et al., 2023).

- B TECHNICAL PROOFS This section contains technical proofs of the theoretical results presented in this paper.

- B.1 PROOF OF PROPOSITION 3.2

Proof. The standard approach to solving an LQR problem is the minimum principle theorem that can be found in control literature (Fleming & Rishel, 1975; Basar et al., 2020). We follow this approach and provide the full proof below for completeness.

The Hamiltonian of the LQR problem (7) is given by

H(zt,pt,ct,t) =

- 1

- 2 ∥ct∥2 + pTt ct. (24)

For c∗t = −pt, the Hamiltonian attains its minumum value: H(zt,pt,c∗t,t) = −21 ∥pt∥2. Using minimum principle theorem (Fleming & Rishel, 1975; Basar et al., 2020), we get

dpt dt

= ∇zt

H (zt,pt,c∗t,t) = 0; (25) dzt dt

= ∇pt

H (zt,pt,c∗t,t) = −pt; (26) z0 = y0; (27) p1 = ∇z1

λ 2 ∥z1 − y1∥22 = λ(z1 − y1). (28)

From (25), we know pt is a constant p. Using this constant in (26) and integrating from t → 1, we have z1 = zt − p(1 − t). Substituting z1 in (27),

p = λ(zt − p(1 − t) − y1) = λ(zt − y1) − λ(1 − t)p, which simplifies to

p = (1 + λ(1 − t))−1 λ(zt − y1) =

1 λ

+ (1 − t)

−1

(zt − y1). Taking the limit λ → ∞, we get p = z

t−y1 1−t and the optimal controller c∗t = y

1−zt 1−t . Since

ut(zt|y1) = y1 − y0, the proof follows by substituting y0 = z

t−ty1 1−t .

| |
|---|

- B.2 PROOF OF PROPOSITION 3.1

Proof. Initializing the generative ODE (1) with the structured noise y1, we get

dXt dt

= vt(Xt), X0 = y1, ∀t ∈ [0,1]. (29) Substituting ut(·) = −v1−t(·) in ODE (6),

dYt dt

= ut(Yt) = −v1−t(Yt), Y0 = y0, ∀t ∈ [0,1]. Replacing t → (1 − t),

dY1−t dt

= vt(Y1−t), ∀t ∈ [0,1]. (30) Since (29) and (30) hold ∀t ∈ [0,1] and X0 = y1, then Xt = Y1−t that implies X1 = Y0 = y0.

| |
|---|

- B.3 PROOF OF THEOREM 3.4

Proof. From Proposition 3.2, we have ut(Yt|Y1) = Y

1−Yt 1−t . In Lemma A.1, we show that

t 1 − t∇log pt(Yt) .

1 1 − t

Yt −

ut(Yt) = −

Now, the controlled ODE (8) becomes:

dYt = ut(Yt) + γ (ut(Yt|Y1) − ut(Yt)) dt, Y0 ∼ p0, Y1 = y1

Y1 − Yt 1 − t

1 1 − t

t 1 − t∇log pt(Yt) + γ

= (1 − γ) −

Yt −

dt

1 1 − t

t 1 − t

γ 1 − t

= −

Yt −

(1 − γ)∇log pt(Yt) +

Y1 dt

1 1 − t

t 1 − t

= −

(Yt − γY1) −

(1 − γ)∇log pt(Yt) dt.

Using continuity equation (Øksendal, 2003), the density evolution of the controlled ODE (8) then becomes:

∂pt(Yt) ∂t

1 1 − t

t 1 − t

(1 − γ)∇log pt(Yt) pt(Yt) . (31) Applying Fokker-Planck equation (Øksendal, 2003) to the SDE (10), we have

= ∇ ·

(Yt − γY1) +

1 1 − t

t 1 − t

∂pt(Yt) ∂t

(1 − γ)∇pt(Yt) , which can be rearranged to equal (31) completing the proof.

+ ∇ · −

(Yt − γY1) pt(Yt) = ∇ ·

| |
|---|

- B.4 PROOF OF LEMMA A.1 Proof. Given (Y0,Y1) ∼ p0 × p1, the conditional flow matching loss (5) can be reparameterized as:

0,Y1)∼p1×p0 ∥(Y1 − Y0) − u(Yt,t;φ)∥22 , Yt = tY1 + (1 − t)Y0, where the optimal solution is given by the minimum mean squared estimator:

LCFM(φ) := Et∼U[0,1],(Y

0,Y1)∼p1×p0 [Y1 − Y0|Yt = yt]. (32) Since Yt = tY1 + (1 − t)Y0, we use Tweedie’s formula (Efron, 2011) to compute

ut(yt) = E(Y

t2

1 1 − t

1 − t∇log pt(yt). (33) Using the above relation, we obtain the following:

E[Y0|Yt = yt] =

yt +

1 t

E[Y1|Yt = yt] =

E[Yt − (1 − t)Y0|Yt = yt]

1 t

(yt − (1 − t)E[Y0|Yt = yt])

=

t2 1 − t∇log pt(yt)

1 t

1 1 − t

yt − (1 − t)

=

yt +

= −t ∇log pt(yt). (34) Combining (33) and (34) using linearity of expectation, we get

ut(yt) = E[Y1|Yt = yt] − E[Y0|Yt = yt] (35)

t2 1 − t∇log pt(yt) (36)

1 1 − t

= −t ∇log pt(yt) −

yt −

1 1 − t

t 1 − t∇log pt(yt), (37)

= −

yt −

The density evolution of Yt now immediately follows from the continuity equation (Øksendal, 2003) applied to (19).

- B.5 PROOF OF LEMMA A.2 Proof. The Fokker-Planck equation of the SDE (12) is given by

∂pt(Yt) ∂t

+ ∇ · −

1 1 − t

Yt pt(Yt) = ∇ ·

t

1 − t ∇pt(Yt) . (38) Rearranging (38) by multiplying and dividing pt(Yt) in the right hand side, we get

∂pt(Yt) ∂t

= ∇ ·

1 1 − t

Yt +

t

1 − t ∇log pt(Yt) pt(Yt) . (39) To conclude, observe that that the density evolution above is identical to (20).

| |
|---|

- B.6 PROOF OF PROPOSITION A.3

Proof. The optimal vector field of the rectified flow ODE (6) is given by Lemma A.1. The proof then immediately follows from the Fokker-Planck equations in Lemma A.1 and Lemma A.2. From Lemma A.2, the density evolution of the SDE (12) is given by

∂pt(Yt) ∂t

= ∇ ·

1 1 − t

Yt +

t

1 − t ∇log pt(Yt) pt(Yt) . The stationary (or steady state) distribution satisfies the following:

∂pt(Yt) ∂t

= 0 = ∇ ·

1 1 − t

Yt +

t

1 − t ∇log pt(Yt) pt(Yt) . Using the boundary conditions (Øksendal, 2003), we get

1 1 − t

Yt +

t

1 − t ∇log pt(Yt) = 0, which immediately implies pt(Yt) ∝ e−

∥Yt∥2

2t .

| |
|---|

- B.7 PROOF OF THEOREM 3.5 Proof. Using Fokker-Planck equation (Øksendal, 2003), Lemma A.4 implies

∂qt(Xt) ∂t

= ∇ · −qt(Xt)

1 − t t ∇log qt(Xt) .

1 t

Xt +

Therefore, the optimal vector field vt(Xt) of the controlled ODE (15) is given by

1 − t

1 t

t ∇log p1−t(Xt). (40) The LQR problem (16) is identical to the LQR problem (7) with changes in the initial and terminal states. Similar to Proposition 3.2, we compute the closed-form solution for the conditional vector field of the ODE (15) as:

vt(Xt) =

Xt +

X1 − Xt 1 − t

. (41) Combining (40) and (41), we have

vt(Xt|X1) =

dXt = [vt(Xt) + η(vt(Xt|X1) − vt(Xt))]dt

1 − t t ∇log p1−t(Xt) + η

X1 − Xt 1 − t

1 t

= (1 − η)

Xt +

dt

(1 − η)(1 − t) t ∇log p1−t(Xt) dt

(1 − η)(1 − t) − ηt t(1 − t)

η 1 − t

=

Xt +

X1 +

1 − t − η t(1 − t)

(1 − η)(1 − t) t ∇log p1−t(Xt) dt.

η 1 − t

=

Xt +

X1 +

The resulting continuity equation (Øksendal, 2003) becomes:

1 − t − η t(1 − t)

(1 − η)(1 − t) t ∇log p1−t(Xt) qt(Xt)

∂qt(Xt) ∂t

η 1 − t

= ∇ · −

Xt +

X1 +

1 − t − η t(1 − t)

2(1 − η)(1 − t) t ∇log p1−t(Xt) qt(Xt)

η 1 − t

= ∇ · −

Xt +

X1 +

(1 − η)(1 − t)

t ∇log p1−t(Xt) qt(Xt) . Using time-reversal property from Propsition 3.2, the above expression simplifies to

+

1 − t − η t(1 − t)

∂qt(Xt) ∂t

η 1 − t

+ ∇ ·

Xt +

X1 +

(1 − η)(1 − t)

t ∇qt(Xt) , which yields the following SDE:

= ∇ ·

2(1 − η)(1 − t) t ∇log p1−t(Xt) qt(Xt)

1 − t − η t(1 − t)

η 1 − t

dXt =

Xt +

and thus, completes the proof.

2(1 − η)(1 − t) t ∇log p1−t(Xt) dt +

X1 +

2(1 − η)(1 − t) t

dWt,

| |
|---|

- B.8 PROOF OF LEMMA A.4

Proof. It suffices to show that the Fokker-Planck equations of the SDE (22) and (12) are the same after time-reversal. Let qt(·) denote the marginal pdf of Xt such that q0(·) = p1(·). The FokkerPlanck equations of the SDE (22) becomes

2(1 − t) t ∇log p1−t(Xt) = ∇ ·

1 − t

1 t

∂qt(Xt) ∂t

+ ∇ · qt(Xt)

t ∇qt(Xt) , which can be rearranged to give

Xt +

2(1 − t) t ∇log p1−t(Xt) +

1 − t t ∇qt(Xt)

1 t

∂qt(Xt) ∂t

= ∇ · −qt(Xt)

Xt +

2(1 − t) t ∇log p1−t(Xt) −

1 − t

1 t

= ∇ · −qt(Xt)

t ∇log qt(Xt) Since Yt is the time-reversal process of Xt as discussed in Proposition (3.1),

Xt +

1 − t

∂qt(Xt) ∂t

1 t

t ∇log qt(Xt) . Substituting t → 1 − t,

= ∇ · −qt(Xt)

Xt +

∂q1−t(X1−t) ∂t

1 1 − t

t

1 − t∇log q1−t(X1−t) , which implies the density evolution of (12):

= ∇ · q1−t(X1−t)

X1−t +

∂pt(Yt) ∂t

1 1 − t

t

1 − t∇log pt(Yt) . This completes the proof of the statement.

= ∇ · pt(Yt)

Yt +

- C ADDITIONAL EXPERIMENTS

This section substantiates our contributions further by providing additional experimental details.

Baselines. We use the official NTI codebase1 for the implementations of NTI (Mokady et al., 2023), P2P (Hertz et al., 2022), and DDIM (Song et al., 2021a) inversion. We use the official Diffusers implementation2 for SDEdit and Flux3. We modify the pipelines for SDEdit and DDIM inversion to adapt to the Flux backbone.

For completeness, we include qualitative comparison with a leading training-based approach InstructPix2Pix (Brooks et al., 2023)4 and a higher-order differential equation based LEDIT++ (Brack et al., 2024)5 (§C). Table 3 summarizes the requirements of the compared baselines.

Table 3: Requirements of compared baselines. Our method outperforms prior works while requiring no additional training, optimization of prompt embedding, or attention manipulation scheme.

### Method Training Optimization Attention Manipulation

SDEdit (Meng et al., 2022) DDIM (Song et al., 2021a) NTI (Mokady et al., 2023) NTI+P2P (Hertz et al., 2022) LEDIT++ (Brack et al., 2024) InstructPix2Pix (Brooks et al., 2023) Ours

Metrics. Following SDEdit (Meng et al., 2022), we measure faithfulness using L2 loss between the stroke input and the output image, and assess realism using Kernel Inception Distance (KID) between real and generated images. Stroke inputs are generated from RGB images using the algorithm provided in SDEdit. Given the subjective nature of image editing, we conduct a large-scale user study to calculate the user preference metric.

For face editing, we evaluate identity preservation, prompt alignment, and overall image quality using a face recognition metric (Ruiz et al., 2024), CLIP-T scores (Radford et al., 2021), and using CLIP-I scores (Radford et al., 2021), respectively. For the face recognition score, we calculate the L2 distance between the face embedding of the original image and the edited image, obtained from Inception ResNet trained on CASIA-Webface dataset. Similar to SDEdit (Meng et al., 2022), we conduct extensive experiments on Stroke2Image generation, and showcase additional capabilities qualitatively on a wide variety of semantic image editing tasks.

Algorithm. The pseudo-code for getting the structured noise is provided in Algorithm 1, and transforming that noise back to an image is given in Algorithm 2.

C.1 HYPER-PARAMETER CONFIGURATIONS

In Table 4, we provide the hyper-parameters for the empirical results reported in §5. We use a fix γ = 0.5 in our controlled forward ODE (8) and a time-varying guidance parameter ηt in our controlled reverse ODE (15), as motivated in Remark 3.3 and Remark 3.6. Thus, our algorithm introduces one additional hyper-parameter ηt into the Flux pipeline. For each experiment, we use a fixed time-varying schedule of ηt described by starting time (s), stopping time τ, and strength (η). We use the default config for Flux model: 3.5 for classifier-free guidance and 28 for the total number of inference steps.

- 1https://github.com/google/prompt-to-prompt
- 2https://github.com/huggingface/diffusers
- 3https://github.com/black-forest-labs/flux
- 4https://huggingface.co/spaces/timbrooks/instruct-pix2pix
- 5https://huggingface.co/spaces/editing-images/leditsplusplus

Algorithm 1: Controlled Forward ODE (8) Input: Discretization steps N, reference image y0, prompt embedding network Φ, Flux model

u(·,·,·;φ), Flux noise scheduler σ : [0,1] → R Tunable parameter: Controller guidance γ Output: Structured noise Y1

- 1 Initialize Y0 = y0
- 2 Fix a noise sample y1
- 3 for i = 0 to N − 1 do

- 4 Current time step: ti = Ni

- 5 Next time step: ti+1 = i+1N

- 6 Unconditional vector field: ut

i

(Yt

i

) = u(Yt

i

,ti,Φ(“”);φ) ▷ Proposition 3.1

- 7 Conditional vector field: ut

i

(Yt

i|y1) = y1−Yt

i

1−ti ▷ Proposition 3.2

- 8 Controlled vector field: uˆt

i

(Yt

i

) = ut

i

(Yt

i

) + γ (ut

i

(Yt

i|y1) − ut

i

(Yt

i

)) ▷ODE (8)

- 9 Next state: Yt

i+1

= Yt

i

+ uˆt

i

(Yt

i

)(σ(ti+1) − σ(ti))

- 10 end
- 11 return Y1

Algorithm 2: Controlled Reverse ODE (15) Input: Discretization steps N, reference text “prompt”, reference image y0, prompt embedding

network Φ, Flux model u(·,·,·;φ), Flux noise scheduler σ : [0,1] → R, structured noise y1

Tunable parameter: Controller guidance η Output: Edited image X1

- 1 Initialize X0 = y1
- 2 for i = 0 to N − 1 do

- 3 Current time step: ti = Ni

- 4 Next time step: ti+1 = i+1N

- 5 Unconditional vector field: vt

i

(Xt

i

) = −u(Xt

i

,1 − ti,Φ(prompt);φ) ▷ Proposition 3.1

- 6 Conditional vector field: vt

i

(Xt

i|y0) = y0−Xt

i

1−ti ▷ Proposition 3.2

- 7 Controlled vector field: vˆt

i

(Xt

i

) = vt

i

(Xt

i

) + η (vt

i

(Xt

i|y0) − vt

i

(Xt

i

)) ▷ODE (15)

- 8 Next state: Xt

i+1

= Xt

i

+ vˆt

i

(Xt

i

)(σ(ti+1) − σ(ti))

- 9 end
- 10 return X1

Table 4: Hyper-parameter configuration of our method for inversion and editing tasks.

Task Starting Time (s) Controller Guidance (ηt) Stopping Time (τ) Strength (η)

Stroke2Image 3 5 0.9 Object insert 0 6 1.0 Gender editing 0 8 1.0 Age editing 0 5 1.0 Adding glasses 6 25 0.7 Stylization 0 6 0.9

- C.2 ABLATION STUDY

In this section, we conduct ablation study for our controller guidance parameter ηt. We consider two different time-varying schedules for ηt, and show that our controller strength allows for a smooth interpolation between unconditional and conditional generation.

- In Figure 9, we show the effect of starting time in controlling the faithfulness of inversion; starting time s ∈ [0,1] is defined as the time at which our controlled reverse ODE (15) is initialized. The initial state Xs = y1−s is obtained by integrating the controlled forward ODE (8) from 0 → 1 − s.

0 2 4 6 8 10

12 15 20 25 27 Original

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

- Figure 9: Effect of starting time. Prompt: “A young man”. The number below each figure denotes the starting time scaled by 28 (the total number of denoising steps) for better interpretation. In the

absence of controller guidance (ηt = 0), increasing the starting time (s) in our controlled ODE (15) improves faithfulness to the original image.

In Figure 10, we study the effect of stopping time. We find that increasing controller guidance ηt by increasing the stopping time τ guides the reverse flow towards the original image. However, we observe a phase transition around τ = 0.14 = 4/28, indicating that the resulting drift in our controlled reverse ODE (15) is dominated by the conditional vector field vt(Xt|y0) for t ≥ τ. Therefore, the reverse flow solves the LQR problem (16) and drives toward the terminal state (i.e., the original image).

0 2 4 6 8 10

12 15 20 25 27 Original

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

- Figure 10: Effect of controller guidance. Prompt: “A young man”. For a fixed starting time s = 0,

- In Figure 11, we visualize the effect of our controller guidance for another time-varying schedule.

consider a time-varying controller guidance schedule ηt = η ∀t ≤ τ and 0 otherwise. The number below each figure denotes the stopping time τ scaled by 28 (the total number of denoising steps) for

better interpretation. Increasing τ increases the controller guidance (ηt) that improves faithfulness to the original image.

We make a similar observation as in Figure 10: increasing ηt improves faithfulness. However, we notice a smooth transition from the unconditional to the conditional vector field, evidence from the smooth interpolation between “A young man” at the top left (η = 0) and the original image at the bottom right.

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

0 0.1 0.2 0.3 0.4 0.5

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

0.6 0.7 0.8 0.9 1.0 Original

- Figure 11: Effect of controller guidance for another time-varying schedule. Prompt: “A young man”. The number below each figure denotes the starting time scaled by 28 (the total number of denoising steps) for better interpretation. For a fixed starting time s = 0 and stopping time τ = 8, consider a time-varying controller guidance schedule ηt = η ∀t ≤ τ and 0 otherwise. Increasing η increases the controller guidance (ηt) that improves faithfulness to the original image.

- C.3 NUMERICAL SIMULATION

In this section, we design synthetic experiments to compare reconstruction accuracy of DM and RF inversion. Given Y0 ∼ p0, where the data distribution p0 := N(µ,I) and the source distribution q0 := N(0,I), we numerically simulate the ODEs and SDEs associated with DM and RF inversion; see our discussion in §3.

For µ = 10, we fix γ = 0.5 in the controlled forward ODE (8), and η = 0.5 in the controlled reverse ODE (15). These ODEs are simulated using the Euler discretization scheme with 100 steps. Additionally, we simulate the uncontrolled rectified flow ODEs (6) → (1) as a special case of our controlled ODEs (8) → (15) by setting γ = η = 0, and the deterministic diffusion model DDIM (Song et al., 2021a) in the same experimental setup.

The inversion accuracy is reported in Table 5. Observe that RF inversion has less L2 and L1 error compared to DDIM inversion (14). The minimum error is obtained by setting γ = η = 0 (i.e., reversing the standard rectified flows), which supports our discussion in §3.3.

Furthermore, we simulate the stochastic samplers corresponding to these ODEs in Table 5, highlighted in orange. Similar to the deterministic samplers, we observe that stochastic equivalents of rectified flows more accurately recover the original sample compared to diffusion models. Our controller in RF Inversion (10) → (17) effectively reduces the reconstruction error in the uncontrolled RF Inversion (12) → (22), which are special cases when γ = η = 0. Thus, we demonstrate that (controlled) rectified stochastic processes are better at inverting a given sample from the target distribution, outperforming the typical OU process used in diffusion models (Song & Ermon, 2019; Ho et al., 2020; Song et al., 2021a;b).

- In Figure 12, we compare sample paths of diffusion models and recitified flows using 10 IID samples

drawn from p0. In Figure 13, we visualize paths for those samples using our controlled ODEs and SDEs with γ = η = 0.5.

- C.4 ADDITIONAL RESULTS ON STROKE2IMAGE GENERATION

In Figure 14 and Figure 15, we show additional qualitative results on Stroke2Image generation. Our method generates more realistic images compared to leading training-free approaches in semantic image editing including optimization-based NTI (Mokady et al., 2023) and attention-based NTI+P2P (Hertz et al., 2022). Furthermore, it gives a competitive advantage over the training-based approach InstructPix2Pix (Brooks et al., 2023).

Table 5: DM and RF inversion accuracy. Stochastic samplers are highlighted in orange.

Method L2 Error L1 Error DDIM Inversion (14) 6.024 19.038 DDPM Inversion (13) 6.007 15.758 RF Inversion (γ = η = 0) (8) → (15) 0.092 0.20 RF Inversion (γ = η = 0) (10) → (17) 3.564 8.795 RF Inversion (γ = 0.5,η = 0) (8) → (15) 4.777 11.628 RF Inversion (γ = 0,η = 0.5) (8) → (15) 1.219 3.074 RF Inversion (γ = 0.5,η = 0.5) (8) → (15) 0.628 1.643

- RF Inversion (γ = η = 0.5) (10) → (17) 0.269 0.694

- RF Inversion (γ = η = 1.0) (10) → (17) 0.003 0.010

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

10

10

10

5

5

5

0

0

0

0.0 2.5 5.0 7.5 10.0

0.0 2.5 5.0 7.5 10.0

0.00 0.25 0.50 0.75 1.00

(a) DDPM (13) Fwd.

(b) DDIM (14) Fwd.

(c) SDE (12) Fwd.

10

10

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

10

5

5

5

0

0

0

0.0 2.5 5.0 7.5 10.0

0.0 2.5 5.0 7.5 10.0

0.00 0.25 0.50 0.75 1.00

(e) DDPM (13) Rev.

(f) DDIM (14) Rev.

(g) SDE (12) Rev.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

10

5

0

0.00 0.25 0.50 0.75 1.00

###### (d) RF (6) Fwd.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

10

5

0

0.00 0.25 0.50 0.75 1.00

(h) RF (6) Rev.

- Figure 12: Sample paths of DMs and RFs. Top row corresponds to the forward process {Yt}, and bottom row, reverse process {Xt}. In each plot, time is along the horizontal axis and the process, along the vertical axis. The sample paths of RFs are straighter than that of DMs, allowing coarse discretization and faithful reconstruction.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.00 0.25 0.50 0.75 1.00

0

5

10

(a) ODE (8) Fwd.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.00 0.25 0.50 0.75 1.00

0

5

10

(b) ODE (15) Rev.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.00 0.25 0.50 0.75 1.00

0

5

10

(c) SDE (10) Fwd.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.00 0.25 0.50 0.75 1.00

0

5

10

(d) SDE (17) Rev.

- Figure 13: Sample paths of our controlled ODEs and SDEs. (a,c) The optimal controller ut(Yt|Y1) steers Yt towards the terminal state Y1 ∼ p1 during inversion. (b,d) Similarly, vt(Xt|Y0) guides Xt towards the reference image Y0 ∼ p0, significantly reducing the reconstruction error.

In Figure 16, we demonstrate the robustness of our approach to corruption at initialization. All the methods transform the stroke input (corrupt image) to a structured noise, which is again transformed back to a similar looking stroke input, highlighting the faithfulness of these methods. However, unlike our approach, the resulting images in other methods are not editable given a new prompt.

- C.5 ADDITIONAL RESULTS ON SEMANTIC IMAGE EDITING

Figure 17 illustrates a smooth interpolation between “A man” → “A woman” (top row) and “A woman” → “A man” (bottom row). The facial expression and the hair style are gradually morphed from one person to the other.

In Figure 18, we show the ability to regulate the extent of age editing. Given an image of a young woman and the prompt “An old woman”, we gradually reduce the controller strength ηt to make the person look older. Similarly, we reduce the strength to make an old man look younger.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Input SDEdit DDIM Inversion NTI NTI+P2P LEDIT++ InstructPix2Pix Ours

- Figure 14: Stroke2Image generation. Additional qualitative results on LSUN-Bedroom dataset comparing our method with SoTA training-free and training-based editing approaches.

Input SDEdit DDIM Inversion NTI NTI+P2P LEDIT++ InstructPix2Pix Ours

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

- Figure 15: Stroke2Image generation. Additional qualitative results on LSUN-Church dataset comparing our method with SoTA training-free and training-based editing approaches.

- Figure 19 shows the insertion of multiple objects by text prompts, such as “pepperoni”, “mushroom”, and “green leaves” to an image of a pizza. Interestingly, pepperoni is not deleted while inserting mushroom, and mushroom is not deleted while inserting green leaves. The product is finally presented in a lego style.
- Figure 20 captures a variety of facial expressions that stylize a reference image. Given the original image and text prompt: e.g. “Face of a girl in disney 3d cartoon style”, we first invert the image to generate the stylized version of the original image. Then, we add the prompt for the expression (e.g., “surprised”) at the end of the prompt and run our editing algorithm (15) with this new prompt: “Face of a girl in disney 3d cartoon style, surprised”. By changing the expression, we are able to preserve the identity of the stylized girl and generate prompt-based facial expressions.

[Figure 173]

Stroke Input

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Inversion

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Editing

SDEdit DDIM NTI NTI+P2P Ours

- Figure 16: Robustness. For inversion, all methods perform well at recovering the stroke input when given a null prompt. However, when a new prompt like “a photo-realistic picture of a bedroom” is provided, only our method successfully generates realistic images. The other methods continue to suffer from the initial corruption, failing to make the output more realistic.

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

- Figure 17: Gender editing. Our method smoothly interpolates between “A man” ↔ “A woman”.

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

“A woman” “old” “older” “A man” “young” “younger”

Figure 18: Age editing. Our method regulates the extent of age editing.

- Figure 21 shows stylization based on a single reference style image and 12 different text prompts, covering both living and non-living objects. The generated images contain various style attributes that includes melting elements, golden color, and 3d rendering from the reference style image.
- Figure 22 visualizes stylization results based on different reference style images. In this experiment, we use text prompt to describe both the content of the generated image and the style of the given reference style image.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Original + “pepperoni” + “mushroom” + “green leaves” + “in lego style”

Figure 19: Object insert. Text-guided insertion of multiple objects sequentially.

“Face of a girl in disney 3d cartoon style”

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Original “surprised” “scared” “frowning” “yawning” “laughing”

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

“grinning” “sad” “smiling” “winking” “gasping” “angry” “Face of a boy in disney 3d cartoon style”

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Original “laugh” “smirking” “frown” “angry” “yelling”

- Figure 20: Stylization using reference text. Stylization of a reference image given prompt-based facial expressions in “disney 3d cartoon style”.

- C.6 HUMAN EVALUATION

We conduct a user study on the test splits of both LSUN Bedroom and LSUN Church dataset using Amazon Mechanical Turk, with 126 participants in total. As shown in Figure 23, each question was accompanied by an explanation of the task, the question, and the evaluation criteria. Participants were shown a pair of stroke-to-image outputs from different models, in random order, along with the input stroke image. They were asked to select one of three options based on their preference using the following two criteria:

- 1. Realism: which of these two images look more like a real, photorealistic image?
- 2. Faithfulness: which of these two images match more closely to the input stroke image?

We collect 3 responses per question. With 300 images in the test dataset and 10 pairwise comparisons, we gathered 9,000 responses for this evaluation. The example in Figure 23 is for the LSUN Church dataset; for LSUN Bedroom dataset, we simply replace the word “church” to “bedroom” in the instructions.

[Figure 225]

“melting golden 3d rendering” (reference style image)

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

“a butterfly” “a boat” “a piano”

“a baby penguin”

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

“a robot” “an f1 car” “a wood cabin” “a bench”

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

“a boy” “a dragon” “a cat” “a dwarf”

- Figure 21: Stylization using a single reference image and various text prompts. Given a reference style image (e.g. “melting golden 3d rendering” at the top) and various text prompts (e.g. “a dwarf in melting golden 3d rendering style”), our method generates images that are consistent with the reference style image and aligned with the given text prompt.

- C.7 GENERATIVE MODELING USING RECTIFIED STOCHASTIC DIFFERENTIAL EQUATIONS

In Figure 24, we compare images generated by Flux (an ODE-based sampler (19)). The similarity between the images generated by the ODE and SDE versions of Flux strengthens the practical significance of our theoretical results (§3).

|[Figure 238]|
|---|

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Flux Ours Flux Ours

“melting golden” “line drawing”

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Flux “3d rendering” Ours Flux “wooden sculpture” Ours

- Figure 22: Stylization using a single prompt and various reference style images: “melting golden”, “line drawing”, “3d rendering”, and “wooden sculpture”. Given a style image (e.g. “3d rendering”) and a text prompt (e.g. “face of a boy in 3d rendering style”), our method generates images that are consistent with the reference style image and the text prompt. The standard output from Flux is obtained by disabling our controller, which clearly highlights the importance of the controller.

[Figure 250]

- Figure 23: Interface for human evaluation. Each participant is asked to select their preferred image based on two criteria: realism and faithfulness.

FluxODE

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

FluxSDE

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

50 100 300 500 700 1000 Prompt: “This dreamlike digital art captures a vibrant, kaleidoscopic bird in a lush rainforest”.

FluxODE

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

FluxSDE

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

50 100 300 500 700 1000 Prompt: “A futuristic robot depicted entirely out of fractals”.

- Figure 24: T2I generation using rectified SDE (22) for different number of discretization steps marked along the X-axis. The stochastic equivalent sampler FluxSDE generates samples visually comparable to FluxODE at different levels of discretization.

