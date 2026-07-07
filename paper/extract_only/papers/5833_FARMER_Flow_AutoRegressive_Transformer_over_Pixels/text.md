# arXiv:2510.23588v2[cs.CV]30Oct2025

[Figure 1]

## FARMER: Flow AutoRegressive Transformer over Pixels

### Guangting Zheng1,3, Qinyu Zhao1,4, Tao Yang1, Fei Xiao1, Zhijie Lin2, Jie Wu1, Jiajun Deng5, Yanyong Zhang3, Rui Zhu1† 1ByteDance Seed China,

3University of Science and Technology of China, 4Australian National University,

2ByteDance Seed Singapore,

5National University of Singapore

†Project lead

### Abstract

Directly modeling the explicit likelihood of the raw data distribution is key topic in the machine learning area, which achieves the scaling successes in Large Language Models by autoregressive modeling. However, continuous AR modeling over visual pixel data suffer from extremely long sequences and high-dimensional spaces. In this paper, we present FARMER, a novel end-to-end generative framework that unifies Normalizing Flows (NF) and Autoregressive (AR) models for tractable likelihood estimation and high-quality image synthesis directly from raw pixels. FARMER employs an invertible autoregressive flow to transform images into latent sequences, whose distribution is modeled implicitly by an autoregressive model. To address the redundancy and complexity in pixel-level modeling, we propose a self-supervised dimension reduction scheme that partitions NF latent channels into informative and redundant groups, enabling more effective and efficient AR modeling. Furthermore, we design a one-step distillation scheme to significantly accelerate inference speed and introduce a resampling-based classifier-free guidance algorithm to boost image generation quality. Extensive experiments demonstrate that FARMER achieves competitive performance compared to existing pixel-based generative models while providing exact likelihoods and scalable training.

Date: October 31, 2025 Correspondence: zhurui.kim@bytedance.com

### 1 Introduction

Explicitly modeling a normalized likelihood P(x) over the high-dimensional data distribution is challenging. Popular generative paradigms such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and diffusion/score-based models do not provide tractable likelihoods—VAEs optimize a lower bound, GANs learn implicit generators without likelihoods, and diffusion/score-based models offer likelihoods only via variational bounds or costly numerical estimation by probability-flow ODE. In contrast, Autoregressive (AR) models directly factorize sequence likelihoods via the chain rule and lead to the scaling successes of Large Language Models [1, 2, 16, 59, 60]. However, modeling the likelihood over continuous, high-dimensional image pixels remains notably challenging compared to the discrete text. Continuous AR over visual pixels has been explored for years—from convolutional PixelRNN/PixelCNN [65, 66] to Image Transformer [46] and iGPT [5]. Despite these efforts, continuous AR suffers from extremely long sequences, making training and sampling costly and brittle to long-range dependencies. This gap motivates revisiting how we parameterize continuous

× Long Sequence Dependency

× Large Gap in Data Distribution

× Long Sequence Dependency

× Large Gap in Data Distribution

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|0>0@@,1,1@,1 @ @),),11 @),1))| | | | | |)| | | | | |

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |max<br><br>0<br><br>max>0 @<br><br>0<br><br>max >0<br><br>0<br><br>>0 @| | | | | |

× Long Sequence Dependency

× Large Gap in Data Distribution

in High-dimensional Space

vs Standard Gaussian × Limited Expressivity × Limited Controllable Sampling

in High-dimensional Space

vs Standard Gaussian × Limited Expressivity × Limited Controllable Sampling

in High-dimensional Space

vs Standard Gaussian × Limited Expressivity × Limited Controllable Sampling

2 @3&1 )

2 @3&1 )

@,2 @3&1 )

max

>0

max

>0

max

>

,

,

× Modeling Difficulty × Sampling Difficulty

× Modeling Difficulty × Sampling Difficulty

Repeat

Repeat

0

0

× Modeling Difficulty × Sampling Difficulty

Repeat

0

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

Causal Autoregressive

Causal Autoregressive

Causal Autoregressive

√ Powerful Expressivity

√ Invertible Architecture

√ Powerful Expressivity

√ Invertible Architecture

√ Powerful Expressivity

√ Invertible Architecture

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
|:,:, , : :), :),),)))| | | | | | | | | | |

max

A(@0 : ; 0, 1)

max

A(@0 : ; 0, 1)

max

A(@0 : ; 0, 1)

0

0

0

1 @2

1 @2

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

Dim

Dim

@2

|Dimension Split<br><br>Dimension @ Split<br><br>Dimension Split<br><br>@| | | | | |@1| | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

max

>0

:

max

>0 :

D

max

>0

0

0

0

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|

@0 : @

@0 : @

@0 : @

√ Easy Modeling via

√ Easy Modeling via

√ Easy Modeling via

Dimension Reduction √ Controllable Sampling √ Exact Likelihood Estimate

Dimension Reduction √ Controllable Sampling √ Exact Likelihood Estimate

Normalizing Flow

###### Normalizing Flow

Normalizing Flow

###### Normalizing Flow

###### Causal Autoregressive

Dimension Reduction √ Controllable Sampling √ Exact Likelihood Estimate

###### Causal Autoregressive

Normalizing Flow

###### Normalizing Flow

###### Causal Autoregressive

| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|

Condition

Condition

Condition

Image :

Image :

Image :

Image :

Image :

Image :

Image :

Image :

Image :

(a) Pixel Autoregressive models

(b) Normalizing Flow

(c) Flow Autoregressive Transformer (FARMER)

- Figure 1 Autoregressive (AR) models offer strong expressivity but struggle with pixel modeling and sampling due to the long sequences required for high-resolution images. Normalizing flows (NFs) employ invertible mappings to transform complex image distributions to a standard Gaussian, but the substantial gap between two distributions leads to degraded sampling quality. FARMER unifies NF and AR within a single framework, using the NF component to transform images into latent sequences, whose distribution is implicitly modeled by the AR component for easier modeling and controllable sampling. Furthermore, FARMER adopts a self-supervised dimension reduction method to partition NF latent channels into distinct groups, making AR modeling feasible and scalable.

densities over high-dimensional pixel spaces and how we couple them with scalable sequence models.

At the same time, Normalizing Flow (NF) [18, 32, 79] has seen a resurgence for image generation. By providing exact likelihoods via invertible and differentiable mappings, NF offers an attractive route for revitalizing continuous AR modeling and a principled latent representation. For instance, JetFormer [64] and STARFlow [18] each design a new NF Transformer as the visual tower: JetFormer employs Jet [32] to enable end-to-end continuous AR modeling over raw image pixels, while STARFlow extends TARFlow [79] and demonstrates that continuous Autoregressive Flow can achieve competitive generation quality. But recent NF works [12, 13, 18, 29, 32, 54, 79] predominantly map the data distribution to a standard Gaussian. This is a challenging objective, as forcing a high-dimensional and highly dispersed data distribution onto a simple isotropic Gaussian can introduce discontinuities or distortions, thus complicating the sampling process from the latent space and transforming back to the data space.

Inspired by the great work of Jetformer [64], we propose a framework named FARMER that leverages the strengths of both Normalizing Flows and Autoregressive models. As shown in Figure 1, rather than mapping the data distribution to a fixed standard Gaussian, we employ an NF to transform images into a latent sequence whose distribution is modeled implicitly by an AR model. Concretely, we implement the NF with an Autoregressive Flow (AF) architecture, ensuring causal modeling for NF/AR within FARMER. The two components are optimized jointly in an end-to-end fashion, preserving the tractable, exact likelihoods of NFs while endowing the target distribution with the expressivity of AR modeling. Beyond this design, two inherent challenges remain: (i) Continuous AR over pixels: Natural images are highly redundant. Without compression via VAEs [28, 55] or discrete tokenizers [51, 67], directly modeling all pixels forces the AR model to handle extremely long-range pixel dependencies, and thus results in unstable training and sample quality degrading. (ii) Slow reverse inference in AF: While AF substantially enhances the mapping capability via next-token modeling, they incur slow inference because the reverse inference process is strictly sequential.

To mitigate the redundancy in pixel AR modeling, we introduce a self-supervised dimension reduction mechanism that partitions NF latent channels into informative and redundant groups without information

loss. The key insight is to factorize the token likelihood P(z | c) as

N

P(z | c) = P(zR | zI,c) P(zI | c) =

PN+1 ziR | zI,c

i=1

N

Pi ziI | z<iI ,c ,

i=1

where zI denotes the informative channels and zR the redundant channels of each token. Concretely, the informative channels ziI are modeled in the standard autoregressive manner, i.e., conditioned on the preceding informative tokens z<iu and context c. The redundant channels ziR across all tokens are modeled jointly by a shared distribution conditioned on the entire sequence of informative channels zI and context c. This construction allows us to treat the redundant channels of all tokens as a single additional token, effectively converting N high-dimensional tokens into N+1 lower-dimensional tokens. Maximizing the resulting token likelihood encourages FARMER to disentangle information across channel groups, i.e., concentrating contour and structural features in zI, while assigning detail and color information to zR, as illustrated in Figure 7.

For the slow reverse issue of AF, we propose a one-step distillation scheme for efficient inference, which distills a single-step student reverse path from the teacher’s forward path, thereby avoiding the causal reverse process of AF models. Finally, we present a resampling-based Classifier-Free Guidance (CFG) algorithm that significantly improves generation quality in this framework. In summary, we summarize our contributions as follows:

- • We introduce FARMER, an elegant and powerful framework that jointly optimizes Autoregressive Flow and Autoregressive Transformer for continuous image pixel likelihood estimation.
- • We propose a self-supervised dimension reduction approach that simplifies modeling of high-dimensional visual data.
- • We develop a one-step distillation method that accelerates AF reverse process by a factor of 22× with only 60 additional training epochs, while maintaining comparable generation quality.
- • We introduce a novel resampling-based CFG algorithm that substantially enhances generation quality.

### 2 Preliminary

#### 2.1 Normalizing Flows

Normalizing Flow [12, 13, 29, 30, 32, 44, 48, 54, 66, 79] maps a complex data distribution x ∼ pdata(x) into a simple one z ∼ pZ(z). The target distribution pZ(z) is usually chosen as a standard Gaussian, which is easy for density estimation and sampling. This transformation is achieved by applying a sequence of invertible functions F = fn ◦ fn−1 ◦ ··· ◦ f1. Accordingly, the forward and inverse mappings are:

z = F(x) = fn ◦ fn−1 ◦ ··· ◦ f1(x), x = F−1(z) = f1−1 ◦ f2−1 ◦ ··· ◦ fn−1(z). (1) Using the change-of-variables formula, NFs can calculate the exact probability density of a data point x as:

pdata(x) = pZ(z) det

∂z ∂x

= pZ(F(x)) det

∂F(x) ∂x

, (2)

where det ∂F∂x(x) denotes the determinant of the Jacobian matrix of the transformation F. To facilitate training via maximum likelihood estimation, the learning objective is commonly formulated in terms of Negative Log-Likelihood (NLL):

−log pZ(F(x)) − log det

min

F

∂F(x) ∂x

. (3)

Previous works [18, 79] consider pZ as the standard Gaussian distribution N(0,1), so Eq. 3 can be written as:

∂F(x) ∂x

0.5||F(x)||22 − log det

. (4)

min

F

- 2.2 AutoRegressive Models

AutoRegressive models formulate the likelihood of a token sequence z = (z1,z2,...,zN) by factorizing it into a product of next-token conditional probabilities:

p(z) =

N

i=1

p(zi|z<i), (5)

where z<i = (z1,...,zi−1) conditions only on the previous tokens (z1,...,zi−1) to predict the next token. Such AR paradigm has achieved remarkable scalability and tremendous success in language models [1, 2, 16, 59, 60]. Furthermore, it has also demonstrated promising capabilities in visual generation [21, 36, 39, 58, 62].

- 3 Approach

- 3.1 Mapping Image to AR Distributions via Invertible Flows

As aforementioned in Eq (4), mapping high-dimensional and highly dispersed image data distribution to a simple isotropic Gaussian distribution via an NF can induce out-of-distribution issues and degrade the sampling quality [18]. Inspired by JetFormer [64], we propose a framework that combines the strengths of NF and AR models. Rather than using a fixed standard normal Gaussian, we employ an NF to transform images into a latent sequence whose distribution is modeled implicitly by an AR model. Then the NF and AR components are optimized jointly in an end-to-end fashion, preserving the tractable, exact likelihoods of NFs while endowing the target distribution with the expressivity of AR modeling. The overall objective is to maximize the log-likelihood of the via the change-of-variables formula:

log pdata(x) =

N

i=1

log p(zi|z<i) + log det(

∂F(x) ∂x

) , (6)

where z = F(x) denotes the forward mapping of the NF. The target distribution over z is parameterized autoregressively. To enhance the expressivity of the AR base, following JetFormer and GIVT [63], we model each conditional probability p(zi|z<i) with a Gaussian mixture model (GMM). The conditional log-likelihood for each token zi is:

log p(zi|z<i) = log(

K

k=1

πi,kN(zi;µi,k,σi,k2 )), (7)

where the mixture weights πi,k, means µi,k, and deviations σi,k2 are predicted by the AR model conditioned on preceding tokens z<i. Furthermore, different from Jetformer, we implement the NF model as an Autoregressive Flow (AF) [30, 44]. AF is a powerful universal approximator for distributions that adopt an autoregressive structure: the transformation of each token zi is conditioned only on the preceding tokens z<i. Such AF architecture ensures that the entire pipeline maintains a consistent and powerful causal formulation. Notably, when the number of mixture components in GMM is set to one (K = 1), the entire network, which composes an AF with an AR model, reduces to a single and deeper Autoregressive Flow. We provide a formal proof of this equivalence in Section A.1.

- 3.2 Flow AutoRegressive Transformer

We devise Flow AutoRegressive transforMER models (FARMER) that unify an invertible autoregressive flow with an autoregressive model into a single framework, which enables end-to-end training on raw image pixels by mapping the data onto an implicit target distribution modeled by the AR.

Dequantize and Patchify. Specifically, given an input image I ∈ RH×W×C, FARMER first adds Gaussian noise to I. It is a common practice [47, 64, 79] to add a small amount of noise to raw image I to dequantize the discrete pixel values and create a more continuous data distribution. Following Jetformer [64], we enhance this technique by employing a noise augmentation strategy with annealed noise levels. During training, we add Gaussian noise with a standard deviation N(0,σ2) to I, where the noise level σ is annealed from 0.1 to

C 1 2 3 N

###### …

|!53|
|---|

Patchify …

[Figure 29]

|!73|
|---|

|!23|
|---|

|!33|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

…

|[Figure 34]<br><br>"27 =("23−$ "423 )·'("423 )<br><br>[Figure 35]<br><br>[Figure 36]<br><br>Token Permutation Forward<br><br>$("423 ) '("423 )<br><br>[Figure 37]<br><br>[Figure 38]<br><br>Token Permutation Inverse<br><br>|!53|
|---|
<br><br>|!23|
|---|
<br><br>!73<br><br>|!33|
|---|
<br><br>…<br><br>|!57|
|---|
<br><br>|!27|
|---|
<br><br>|!77|
|---|
<br><br>|!37|
|---|
<br><br>…<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Transformer Blocks|
|---|

|Causal Transformer 1|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

Dimension Split

|Causal Transformer 2|
|---|

…

Autoregressive Flow

…

[Figure 44]

[Figure 45]

Transformer Blocks

|Causal Transformer n|
|---|

&8

&8

&8

&8

&8

"× $8 %8

$8 %8

$8 %8

$8 %8

$8 %8

…

…

Cond

|Causal Transformer|
|---|

Autoregressive Model

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

GMMs …

…

###### …

…

|!57|
|---|

Target:

|!77|
|---|

|!27|
|---|

|!37|
|---|

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

…

NLL Loss

1 2 3 4

1 2 N

FARMER Architecture

Causal Transformer (AF) t=2

Causal Transformer (AR)

- Figure 2 Overview of FARMER. Left, FARMER consists an autoregressive flow (AF) and an autoregressive (AR) model. The AF maps image patches to latent sequences, while the AR predicts Gaussian Mixture Models (GMMs) conditioned on these latents, optimizing their likelihood end-to-end. Middle, Each AF block performs an invertible next-token transformation of the input sequence to obtain a new sequence. Right, AR splits latent channels into informative and redundant groups, modeling each informative token’s likelihood via a GMM conditioned on its previous tokens, and redundant tokens jointly via a shared GMM conditioned on all informative tokens. This separation enables disentangling structural and detailed information.

0.005 via a cosine decay schedule. Then we patchify the noised image with a downsampling factor p to obtain the patch representation I′ ∈ Rh×w×d, where h = H/p, w = W/p, and d = C · p2. Finally, we reshape I′ into a sequence of N = h · w continuous-valued visual tokens X = {x1,x2,...,xN}, with each token xi ∈ Rd. Notably, there is no dimension compression in the whole patchify process.

Forward and Reverse of Autoregressive Flow. During training, FARMER utilizes an autoregressive flow F to map token sequence X ∈ RN×d to latents Z ∈ RN×d, i.e., Z = F(X). By design, F is invertible (see

- Figure 2) and composed of n invertible blocks: F = fn ◦ fn−1 ◦ ··· ◦ f1. Letting Z0 = X and Zn = Z, the forward transformation for the t-th AF block, Zt = ft(Zt−1), is defined for each token zit 1 as follows:

z1t−1 if i = 1, zit−1 − µt(z<it−1) ⊙ σt(z<it−1) if i > 1,

zit =

(8)

where z<it−1 represents the preceding tokens {z1t−1,...,zit−−11}. The bias factor µt(z<it−1) and the scaling factor σt(z<it−1) are predicted by t-th block conditioned on the preceding tokens z<it−1 in a causal manner. Accordingly, the inverse of t-th block, ft−1, can be derived by algebraically solving for zt−1 from the Eq (8). For each token, the inverse transformation Zt−1 = ft−1(Zt) is defined as (see Figure 3a):

z1t if i = 1, zit ⊘ σt(z<it−1) + µt(z<it−1) if i > 1,

zit−1 =

(9)

where ⊘ denotes element-wise division. For training via the change-of-variables formula, it is essential that the Jacobian determinant of each block ft can be efficiently computable. Such autoregressive flow architecture enables that the Jacobian ∂Z

t

∂Zt−1 is lower triangular, so its determinant equals the product of its diagonal terms (i.e., the scaling factor σt). Consequently, the block-wise log-determinant is:

N

d

∂Zt ∂Zt−1 =

log [σt(z<it−1)]j

log det

i=1

j=1

1Subscripts denote indexing i along the token sequence dimension, and superscripts denote the t-th AF block indices.

By the chain rule, the total log-det of F is the sum over blocks, which in our case reduces to:

∂Z ∂X

log det

n

n

∂Zt ∂Zt−1 =

=

log det

t=1

t=1

N

d

log [σt(z<it−1)]j . (10)

i=1

j=1

Permutation. To improve the expressiveness of AF, we follow TARFlow [79] and apply a permutation to the token sequence as shown in Figure 2. Specifically, at the beginning of the t-th AF block, we apply the forward permutation πt to Zt−1, which reverses the token order. After the forward AF transformation Zt = ft(Zt−1), we apply the corresponding inverse permutation πt−1 to Zt to restore the original ordering.

AR Modeling. After the AF forward mapping, we get the latent representation Z = {z1,z2,...,zN} from the input image. Then we model its probability distribution with a large causal AR Transformer. The AR Transformer is conditioned on an embedding c ∈ R1×D which encodes conditional information such as a class label. To amplify its effect, we replicate condition embedding for M times and prepend to the latent sequence Z. By the chain rule,

N

P(Z|c) =

p(zi|z<i,c).

i=1

For each token, the AR Transformer predicts the parameters of a K-component Gaussian Mixture Model (GMM) distribution Gi:

p(zi|z<i,c) =

K

πk(z<i,c)N zi ; µk(z<i,c),diag(σk2(z<i,c)) , (11)

k=1

where πk ∈ R,µk ∈ Rd,σk ∈ Rd are the mixture weights, means, and standard deviations of the k-th GMM component. To highlight the conceptual link to the invertible flow, Eq (11) can be reformulated as:

p(zi|z<i,c) =

K

1 σk(z<i,c)

πk(z<i,c)N (zi − µk(z<i,c)) ⊙ diag(

); 0,Id

k=1

1 σk(z<i,c)

, (12)

This formulation reveals that each GMM component models zi by a simple and invertible affine transformation (shifting by µk and scaling by σ1

) to a random variable drawn from a standard Gaussian distribution. This reveals that each GMM component performs an invertible affine normalization, i.e., (z

k

i−µk)

σk ∼ N(0,I). Learning Objective. As described in Eq (6), Eq (10) and Eq (11), the training loss of FARMER is the negative log-likelihood (NLL) of data and averaged over all dimensions:

1 N · d

L = −

N

∂Z ∂X

log p(zi|z<i,c) + log det

i=1

. (13)

#### 3.3 Self-supervised Dimension Reduction

A fundamental challenge in pixel AR modeling is redundancy: natural images are intrinsically low-dimensional signals whose spectrum are dominated by low frequencies [64]. Although an invertible AF can faithfully map the data distribution, its bijective nature preserves dimensionality. For a 256 × 256 × 3 image with patch size 16, the latent sequence has N = (256/16)2 = 256 tokens, each of dimension d = 768. This high-dimensional latent Z exacerbates two issues: (i) per-token AR modeling with a K-component GMM in Rd becomes exceptionally challenging. (ii) The enlarged latent volume expands the sampling space, reducing efficiency and often degrading sample quality.

Prior work like RealNVP [13] factors out half of the dimensions and model them with Gaussian priors. Jetformer [64] follows a similar strategy: it models the informative dimensions ZI autoregressively and assigns the redundant dimensions ZR a standard Gaussian prior, effectively assuming

P(Z | c) = P(ZR)P(ZI | c),

i.e., ZR is independent of both ZI and c. This is a strong assumption that is often violated in practice: informative and redundant parts typically remain correlated, so enforcing independence can discard information. Moreover, decoupling ZR from c and ZI restricts how other modalities interact with the full latent, leading to suboptimal performance on multi-modal tasks.

To this end, we propose a novel self-supervised dimension reduction technique to address the above issues. It reduces the complexity of AR modeling, shrinks the sampling space, and lowers the computational cost, all while avoiding information loss. As shown in Figure 2 we split the latent Z ∈ RN×d channel-wise into an informative part ZI ∈ RN×d

I

R

, with d = dI + dR. Then we correctly factorize the joint probability via the chain rule:

and a redundant part ZR ∈ RN×d

P(Z | c) = P(ZI | c)P(ZR | ZI,c).

Rather than assuming that ZR is independent of (ZI,c) in Jetformer, we explicitly condition ZR on both c and ZI, where ZI serves as the global image context. Furthermore, we constrain all tokens in ZR to share a GMM distribution, while modeling tokens in ZI in a token-by-token manner. This design encourages self-supervised disentanglement of distinct information across channel groups, without relying on a standard Gaussian prior.

For P(ZI|c), we model each informative token ziI autoregressively with an individual GMM distribution Gi predicted by the AR Transformer conditioned on c and the preceding z<iI , thereby being capable of capturing complex distributions. In contrast, for P(ZR|ZI,c), we use the entire informative sequence ZI (global context) together with c to predict a single shared GMM GN+1 for all redundant tokens ziR, By maximizing the combined likelihoods, our method successfully encourages complex contour and structural information to be reserved into ZI, while the simple color and fine-detail information is relegated to ZR, as shown in Figure 7 and discussed in Section 4.3.

After dimension reduction, the final training loss L is rewritten as the sum of the NLL for both components:

N

N

∂Z ∂X

1 N · D

log p(ziR|z≤I N,c) + log det

log p(ziI|z<iI ,c) +

. (14)

L = −

i=1

i=1

#### 3.4 Resampling-based Classifier-Free Guidance

Classifier-Free Guidance (CFG) has become a standard technique for improving sample quality in diffusion [40, 49, 55] and autoregressive models [36, 58, 62]. Conceptually, CFG steers the sampling process from a base distribution towards a target conditional distribution. For FARMER, the guided log-probability for a latent token z can be formulated as:

log p′(z) ∝ log pc(z) + w · (log pc(z) − log pu(z)) = log pu(z) + (w + 1) · (log pc(z) − log pu(z)), (15)

where pc(z) = p(z|c) is the conditional GMM distribution, pu(z) = p(z|∅) is the unconditional GMM, and w is the guidance scale. However, the guided distribution p′(z) is a product and sum of GMMs, which does not correspond to any known tractable distribution, making direct sampling infeasible.

To make it practical, we introduce a novel Resampling-based CFG. The key insight is that the target distribution p′(z) can be decomposed into two components as shown in Eq (15): the first term (blue) is a tractable GMM distribution and can be sampled directly, while the second term (red) is not samplable but allows evaluation of the sample probability under such distribution. Therefore, we approximate the sampling from p′(z) via a three-step resampling scheme as detailed in Algorithm 1. For For each token zi, the procedure is: (i) Propose. Sample s candidates from the conditional GMM pc(zi) and s′ candidates from the unconditional GMM pu(zi) respectively. (ii) Weigh. Compute the corresponding log probability of each candidates as the second term in Eq (15), and normalize these weights. (iii) Resample. Resample from the categorical distribution that consists of the normalized weights of all candidates, to obtain the final sample. In summary, the probability where candidate z is selected in the “propose” step is pc(z), and that in the “resample” step is p

w matches the target probability p′(z). More details are provided in the Section A.2.

w

c(z) pu(z)

. This resampling procedure ensures that the overall probability pc(z) p

c(z) pu(z)

Forward Forward

Forward Forward

Forward

Forward

Teacher forward path

Teacher forward path

Forward path

Forward path

One-step

One-step

One-step

One-step

𝑍 𝑍 𝑍 … 𝑍 𝑍

𝑍 𝑍 𝑍 … 𝑍 𝑍

Forward Forward Forward

Forward Forward Forward

Reverse path

Reverse path

𝑍 𝑍 𝑍 … 𝑍 𝑍

𝑍 𝑍 𝑍 … 𝑍 𝑍

Sequential

Sequential

Reverse Reverse Reverse

Reverse Reverse Reverse

MSELoss

MSELoss

+Noise

+Noise

…

…

| || |𝑧 𝑧| |
|---|---|---|
<br><br>| | | | | |
|---|---|---|---|---|
| |𝑧 𝑧| | | |
<br><br>Step 1<br><br>𝑧 = 𝑧<br><br>Step 1<br><br>𝑧 = 𝑧| |
|---|---|---|

|[Figure 56]<br><br>[Figure 57]<br><br>Transformer Blocks<br><br>𝜇(𝑧 ) 𝜎(𝑧 )<br><br>| |𝑧 𝑧| |
|---|---|---|
<br><br>𝑧 𝑧<br><br>| | | | |
|---|---|---|---|
| |𝑧 𝑧| | |
<br><br>Step 3<br><br>𝑧 =<br><br>𝑧 𝜎 𝑧<br><br>− 𝜇 𝑧<br><br>[Figure 58]<br><br>[Figure 59]<br><br>Transformer Blocks<br><br>𝜇(𝑧 ) 𝜎(𝑧 )<br><br>𝑧 𝑧 Step 3<br><br>𝑧 =<br><br>𝑧 𝜎 𝑧<br><br>− 𝜇 𝑧| |
|---|---|

| ||𝑧𝑧<br><br>| || |𝑧 𝑧| |
|---|---|---|
<br><br>Step 2Step 2| |
|---|---|---|---|
|TrT|ar|ansfnsfo|[Figure 60]<br><br>[Figure 61]<br><br>ormerrmer BlocksBlocks<br><br>[Figure 62]<br><br>[Figure 63]|
| | | | |
<br><br>| | | | |
|---|---|---|---|
| |𝑧 𝑧| | |
<br><br>𝑧 =<br><br>𝑧 𝜎 𝑧<br><br>𝑧 = − 𝜇 𝑧<br><br>𝑧 𝜎 𝑧<br><br>− 𝜇 𝑧| |
|---|---|---|

|Step NStep N|
|---|

Forward Forward Forward

Forward Forward Forward

𝑍 𝑍 𝑍 … 𝑍 𝑍

𝑍 𝑍 𝑍 … 𝑍 𝑍

…

…

Student reverse path

Student reverse path

One-step Distillation

One-step Distillation

(a) Autoregressive Flow Reverse Process

(b) One-Step Distillation Process

- Figure 3 One-Step Distillation. (a) The autoregressive flow (AF) reverse process reconstructs tokens sequentially, conditioning each token on previous ones, which leads to slow inference. (b) Our method distills a one-step student reverse path from the frozen teacher forward path in an end-to-end manner, approximating the reverse process of each AF block by the corresponding student AF block’s forward process, thereby enabling 22× faster AF reverse process and 4× overall inference speed-up.

#### 3.5 Fast Inferring via One-Step Distillation

A significant drawback of the Autoregressive Flow is the slow inference speed, caused by its sequential and token-by-token reverse process. As shown in Eq (9), during the inverse mapping (ft−1) of AF block t, the calculation of each token zt−1,i is conditioned on the preceding tokens zt−1,<i, leading to a complexity of O(N ×n). Such autoregressive dependency brings a substantial inference speed bottleneck, and such limitation is also noted in recent AF-Transformer works like TARFlow [79] and STARFlow [18] whose token sequence length is 1024 with the patchsize of 8.

Beneficial from the invertible nature of Normalizing Flow, whose forward and reverse paths are exact inverses, we can train a new AF whose forward path mirrors the original AF’s reverse path. Furthermore, because the forward/reverse path of NF consist of finite steps, we can invert the original AF’s forward path (Z0,Z1,...,Zn) to obtain its reverse path (Zn,Zn−1,...,Z0), and utilize such reverse path to supervise the new AF, thereby avoiding the original AF to perform slow reverse process to obtain its reverse path.

As shown in Figure 3 and inspired by the generative distillation works [56, 68, 75], we propose a one-step distillation scheme that learns a single-step student reverse path from the trained teacher’s forward path while maintaining competitive sample quality. Algorithm 2 details the procedure: we first obtain a teacher AF model, trained within the FARMER framework. Then, we initialize the student by copying the teacher AF and enable its attention bidirectional. At each distill iteration, we forward training data z0 to the latent zn by the teacher AF. In this way, we obtain a teacher forward path F(Z0) = (Z0,Z1,...,Zn). We use its reversal (Zn,Zn−1,...,Z0) as the supervision target for the student’s forward path G(Z˜n) = (Z˜n,Z˜n−1,...,Z˜0). Specifically, to enhance the robustness of the student AF, we add a small Gaussian noise to Zn as Z˜ and take Z˜ as the input of the student AF. Then, the output latent Z˜t−1 of each t-th student AF block is supervised by minimizing the Mean Squared Error (MSE) against the Zt−1 from the teacher path. By distilling one such student AF model, we significantly accelerate the reverse process from 0.1689 to 0.0076 seconds per image. As discussed in Section 4.3 and Table 5, such one-step distillation brings a 22× acceleration for AF reverse process while maintaining comparable generation quality.

Notably, different from the progressive distillation for diffusion models, our approach offers three main advantages: it distills the entire AF model in an end-to-end manner, ensuring robustness to accumulative inference error; it eliminates the need for teacher models to run the inference process, thereby accelerating the distillation process; and it requires only 60 additional training epochs on the AF.

Algorithm 2 One-step sampling distillation Require: Trained teacher AF (frozen)

Algorithm 1 Resampling-based CFG method

Require: AR model Pθ and AF model Fθ Require: Guidance scale w

n ◦ fη

n−1 ◦ ··· ◦ fη

Fη = fη

1

for i ∈ [0,...,N + 1] do ▷ Sampel tokens

Require: Data set D Require: Student AF Gθ = gθ

- # step1:Propose candidates

Gc,i = Pθ(z<iu ;c) ▷ Predict GMMc Gu,i = Pθ(z<iu ;∅) ▷ Predict GMMu zi,j ∼ Gc,i, j ∈ [0,..,s] ▷ Sample from pc(z) zi,j ∼ Gu,i, j ∈ [s + 1,..,s + s′] ▷ from pu(z)

- # step2:Weigh candidates if j ∈ [0,...,s] then ▷ Calculate weights

πj = w · (log Gc,i(zi,j) − log Gu,i(zi,j))

else

πj = (w+1)·(log Gc,i(zi,j)−log Gu,i(zi,j)) end if π1,...,πs+s′ = logsoftmax(π1,...,πs+s′)

- # step3:Resample from candidates if i ≤ N then ▷ For informative tokens

1 ◦ gθ

2 ◦ ··· ◦ gθ

n

for m epochs do

for K iterations do x ∼ D x = Patchify(x) Z0 := x for n Teacher AF blocks do

Zt,_ = fη

(Zt−1) ▷ Teacher Transform

t

end for

Z := Zn ϵ ∼ N(0,I) ▷ Sample noise s ∼ U[0,0.3] ▷ Sample scale Z˜ = Z + s · ϵ ▷ Add noise to latent Z˜n := Z˜

idx ∼ Categorical(π1,...,πs+s′) ziu := zi,idx

# Distill a one-step student reverse # path from the teacher forward path for n Student AF reversed blocks do

else ▷ For redundant tokens

Z˜t−1,_ = gθ

(Z˜t) ▷ Student Transform Lθ

for k ∈ [0,...,N] do ▷ s,s′ larger in here idxk ∼ Categorical(π1,...,πs+s′) zkd := zidx

t

= ∥Z˜t−1 − Zt−1∥22 ▷ MSE loss

t

end for

k

Lθ = n1 nt Lθ

end for

t

end if end for z = concat[[z1u,zid],...,[zNu ,zNd ]] x = Fθ−1(z) ▷ Reverse to data

θ ← θ − γ∇θLθ

end for end for

### 4 Experiments

#### 4.1 Experimental Settings

Datasets. We empirically verify the merits of the proposed FARMER for image generation on ImageNet [10] dataset at 256 × 256 resolution, which consits of 1,281,167 training images from 1K different classes.

Network Architectures. We design two model scales: FARMER-1.1B/1.9B. The number of invertible AF blocks is set to 28 and 32 respectively. Each AF block contains 4/6 Transformer layers for FARMER-1.1B/1.9B. For the AR Transformer module, the number of Transformer blocks is 12 and 24 respectively. For the GMM prediction heads, the informative dimensions (dI) are set to 128 with K = 64 mixtures, while redundant dimensions (dR) are set to 640 with K = 200 mixtures. Table 1 summarizes the detailed architectural configurations.

Table 1 The architecture configurations of FARMER in two different scales (i.e., 1.1B and 1.9B).

Autoregressive Transformer Invertible Autoregressive Flow

Model

Params

Layers Hidden size Params AF Blocks Layers Hidden size Params

FARMER-1.1B 12 768 295M 28 4 768 828M 1.1B FARMER-1.9B 24 1024 498M 32 6 768 1.4B 1.9B

Training Setup. We train the models using AdamW optimizer (β1 = 0.9,β2 = 0.95) with weight decay of 0.03 for 320 epochs. A cosine learning rate schedule is applied, starting from 1 × 10−4 to 1 × 10−6, with 5,000-step

- Table 2 System performance comparison on ImageNet 256 × 256 class-conditioned generation. “↓” or “↑” indicate lower or higher values are better. Metrics include Fréchet inception distance (FID), inception score (IS), precision and recall. Resampling-based CFG is applied on FARMER.

Types Models Params Epochs FID↓ IS↑ Pre.↑ Rec.↑ Latent Generative Models

LDM-4 [55] 400M + 86M 170 3.6 247.7 0.87 0.48 DiT-XL [49] 675M + 86M 1400 2.27 278.2 0.83 0.57 SiT-XL [40] 675M + 86M 1400 2.06 270.3 0.82 0.59

Diff.

FlowDCN [70] 618M + 86M 400 2.00 263.1 0.82 0.58

REPA [78] 675M + 86M 800 1.42 305.7 0.80 0.64 DDT-XL [72] 675M + 86M 400 1.26 310.6 0.79 0.65 REPA-E [35] 675M + 86M 800 1.12 302.9 0.79 0.66

GIVT [63] 1.67B+53M 500 2.59 - 0.81 0.57

AR

MAR-AR [36] 479M+66M 800 4.69 244.6 - -

MAR-L [36] 479M + 66M 800 1.78 296.0 0.81 0.60 NF

STARFlow [18] one-step denoise 1.4B+86M 320 2.96 - - STARFlow [18] finetune decoder 1.4B+86M 320 2.40 - - -

Pixel Generative Models GAN BigGAN [3] 112M / 6.95 224.5 0.89 0.38

ADM [11] 554M 400 4.59 186.7 0.82 0.52

CDM [23] - 2160 4.88 158.7 - SimpleDiffusion [24] 2.0B 800 2.77 211.8 - -

Diff.

PixelFlow-XL/4 [6] 677M 320 1.98 282.1 0.81 0.60 PixNerd-XL/16 [71] 700M 320 1.93 298 0.80 0.60 SiD2 patch 1 [25] - 1280 1.38 - - AR FractalMAR-H [37] 844M 600 6.15 348.9 0.81 0.46 NF

TARFlow [79] patch 8 1.3B 320 5.56 - - STARFlow [18] patch 8 1.4B 320 4.69 - - -

JetFormer [64] 2.8B 500 6.64 - 0.69 0.56 FARMER 1.1B patch 16 1.1B 320 5.40 212.23 0.78 0.45

NF+AR

FARMER 1.1B patch 8 1.1B 320 5.02 237.00 0.80 0.45 FARMER 1.9B patch 16 1.9B 320 3.96 250.64 0.79 0.50

FARMER 1.9B patch 8 1.9B 320 3.60 269.21 0.81 0.51

linear warmup. Gaussian noise with a cosine decay from 0.1 to 0.005 is added to the raw image.

Evaluation Metrics. To assess sample quality, we use Fréchet Inception Distance (FID) [22], Inception Score (IS) [57] and Precision/Recall [33] on 50K generated samples to measure the image quality on ImageNet-256.

#### 4.2 Results

System-level Comparison. As shown in Table 2, we compare FARMER with various generative models, including both latent and pixel-based approaches, on the class-conditional ImageNet 256×256 benchmark. Notably, FARMER significantly outperforms JetFormer [64], the most comparable baseline to our model, reducing the FID by 3.04. Furthermore, FARMER demonstrates superior generation quality compared to the NF-based models, TARFlow [79] and STARFlow [18]. FARMER also achieves competitive performance and faster convergence speed against mainstream Generative Adversarial Networks (GANs), diffusion models, and AR models. While methods like PixelFlow [6] and PixNerd [71] employ complex multi-stage pipelines to achieve better results, our approach remains highly competitive by utilizing a simple, single-stage, end-to-end training strategy. Compared to latent generative models, our method maintains strong generative performance. Latent generative models often benefit from a well-structured continuous latent space, modeled by VAEs, that facilitates high-quality sampling. However, by operating directly in pixel space, our model gains direct access

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

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

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

###### Figure 4 Qualitative Results. Images generated by FARMER on ImageNet 256x256.

FARMER MAR DiT FARMER MAR DiT

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Figure 5 Qualitative Comparison. Images of class 0 in ImageNet generated by FARMER, MAR, and DiT.

to the raw data distribution. This approach can potentially capture more detailed data semantics without the information bottleneck imposed by VAEs.

Qualitative Results. To qualitatively evaluate FARMER, we show 28 generated images by FARMER-1.9B in Figure 4, sampling using resampling-based classifier-free guidance. As shown, our FARMER generates diverse images with high quality. A key advantage of FARMER over latent generative models is its ability to preserve fine-grained details. This is because our end-to-end training directly accesses the raw data distribution, and the invertible nature of NFs prevents information loss. As shown in Figure 5, our FARMER can reconstruct intricate features, such as faces, which are often blurred or distorted by the compression of VAEs.

#### 4.3 Experimental Analysis

Ablation Study. Here we investigate the impact of each component within the FARMER framework on overall performance. Table 3 reports the performance of FARMER-1.1B with GMM components number (K=1024) across the ablated runs of different components on ImageNet 256×256 dataset for class-conditional image generation. Natural images typically possess a high degree of redundancy, and low-dimensional signals with low-frequency components dominating the spectrum [64]. Direct transformation of original images using normalizing flows yields latent representations with unchanged dimensionality. Partitioning these high-dimensional latents into equal-length, high-dimensional tokens complicates AR modeling and sampling. By introducing a self-supervised dimension reduction design as Eq (14), the FID notably decreases from 61.17

- Table 3 Ablation study of FARMER. We demonstrate relative impact of various components on generation quality. Self-supervised Dim. Reduce Condition Repeat Final Permute CFG Method FID↓ IS↑

✗ ✗ ✗ ✗ 61.17 22.10 ✓ ✗ ✗ ✗ 49.29 30.61 ✓ ✓ ✗ ✗ 45.34 33.87 ✓ ✗ ✓ ✗ 45.69 33.73 ✓ ✓ ✓ ✗ 44.56 33.17

✓ ✓ ✓ Naive Method 8.66 233.84 ✓ ✓ ✓ Resampling-based 5.67 215.53

Table 4 Impact of Normalizing Flow Architectures.

NF Architectures FID IS Forward Speed (s/img) Reverse Speed (s/img)

Jet 106.23 13.14 0.0065 0.0099 AF 5.55 194.63 0.0066 0.1689 AF+One-step Distll. 5.63 193.49 0.0066 0.0076

to 49.29, and IS also improves from 22.10 to 30.61. Next, we repeat the class embedding 64 times to enhance the conditional guidance, the FID further decreases to 45.34. If we consider the AR model as a block of AF, adding a token permutation operation between AF and AR is beneficial to preserve the fixed dependency between token sequences. The FID further decreases to 44.56. CFG is essential for improving generation quality in modern generative models during sampling. We first adopt a naive CFG sampling method from Jetformer [64], the FID score notably decreases to 8.66. Then, we upgrade the CFG sampling method to the resampling-based method described in Section 3.4, the FID score further decreases to 5.67. Together, these design choices enable FARMER to achieve strong performance across most evaluation metrics.

Impact of Normalizing Flow Architectures. The architecture design of NF is an important research topic and has been extensively studied [12, 13, 29, 30, 32, 44, 54, 66, 79]. Different NF architectures exhibit distinct characteristics in terms of representational capacity, training speed, and inference efficiency. Here we primarily compare two architectures, Jet and AF, which have demonstrated strong performance in modern generative models Jetformer [64] and Tarflow [64], respectively. For a fair comparison, we employ similar network parameters, same block numbers, the same layers per block, and the same AR models. Their representational capacity is evaluated using the FID metric, while forward and reverse speeds are also reported. Table 4 summarizes these results. Specifically, in each transformation of Jet, Jet first computes an affine transform from one half of the input latent channels by a Jet block and then applies it to the other half of the input channels; this pattern applies to both forward and reverse passes. Jet is constructed by stacking N such transformations. This simple and efficient design enables Jet to achieve fast forward and reverse computations, but it also limits its representational capacity, leading to a failure to separate different information of the image into two channel groups. As described in Section 3.2, in each transformation of AFs, each token is updated based on preceding tokens through the block, resulting in a reverse process where each token must be generated one by one. This enhances representation ability but leads to slow reverse speed. To address this, we introduce a one-step distillation strategy. By distilling a student AF model from the trained and frozen teacher AF model over only 60 additional training epochs on the NF, we significantly improve the reverse speed from 0.1689 seconds per image to 0.0076 seconds per image. This approach provides a fast and expressive architecture for both training and inference.

Dimension Reduction Method Comparison. We also compare our self-supervised dimension reduction method with the approach adopted in JetFormer [64]. JetFormer assumes that a subset of channels is redundant and independent from the remaining channels and maximizes the likelihood of these redundant channels under the standard Gaussian prior. This assumption may result in information loss, thereby degrading generation quality. In contrast, our self-supervised method models redundant channels as being conditionally dependent on informative channels which encapsulate the global information of images. Our method achieves improved generative performance, reducing FID from 7.81 to 5.67, and increasing IS from 182.87 to 215.53.

[Figure 98]

[Figure 99]

###### (a) Impact of GMM mixture component number (b) Impact of informative dimension Figure 6 The ablation study of different properties.

0.0 0.2 0.4 0.8 1.0 1.2 1.5 2.0 4.0

||[Figure 100]<br><br>[Figure 101]|
|---|
|
|---|

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

- Figure 7 The impact of redundant channels. The numbers above indicate scaling factors applied to the variance of the shared GMM distribution for redundant channels. Adjusting this variance controls sampling diversity: larger variance yields more diverse, potentially out-of-distribution samples, while smaller variance limits diversity. Visualization results demonstrate that the self-supervised dimension reduction effectively separates structural information from color details.

Impact of GMM Mixture Component Number. We analyze the impact of the number of GMM mixtures predicted by the AR models, which reflects the complexity of the approximated distribution. A larger number of mixtures enables the model to represent more complex distributions; however, it also increases sampling difficulty and computational cost during training. As shown in Figure 6a, the FID varies only slightly across different mixture numbers and attains its optimal value at 64 mixtures. Notably, reducing the number further—to 32 mixtures—prevents the model from performing effective dimension reduction, resulting in a significant decline in generation quality. Therefore, we set the number of mixtures to 64 to balance generation quality and training cost.

Impact of the Informative Dimension. We analyze the impact of the informative dimension, which reflects how information is separated and allocated by the NF models. As shown in Figure 6b, the FID initially decreases as the informative dimension increases and achieves the optimal value at 128. Further increasing the dimension leads to a rise in FID. This phenomenon demonstrates a trade-off: increasing the informative dimension allows capturing more information, but also makes AR modeling and sampling more challenging. Therefore, we set the informative dimension to 128.

Information Separation of Different Dimension Groups. Here, we visualize the information contained in the informative and redundant channels. Specifically, during inference, we first predict all tokens of the informative channels in a token-by-token manner. Subsequently, based on these tokens, we predict a shared GMM distribution for the redundant channels. By adjusting the variance of each Gaussian component in the GMM, different distributions are obtained, from which we sample all tokens of the redundant channels. As shown in Figure 7, reducing the variance causes sampled tokens of redundant channels to concentrate around the means of the Gaussians, resulting in reduced diversity and smoother color regions, while the global structure of the images remains largely unaffected. Conversely, increasing the variance enhances diversity

but raises the risk of sampling out-of-distribution values, which can lead to color artifacts or, in extreme cases, failure to generate coherent images. These observations demonstrate that our self-supervised dimension reduction method successfully decouples structural contour information from fine color details.

Table 5 Inference Speed Accelerate.

AR infer. time (% in total)

NF reverse time (% in total)

Method Epochs FID IS

Total time

FARMER 280 5.55 194.63 0.0500s (22.8%) 0.1689s (77.2%) 0.2189s w/. One-step Distll. 280+60 5.63 193.49 0.0500s (88.2%) 0.0076s (13.4%) 0.0567s

Inference Speed Acceleration. As shown in Table 5, the baseline FARMER requires 0.2189 seconds per image for inference, where the AR Transformer accounts for 0.0500 seconds and the NF reverse process dominates with 0.1689 seconds. By applying the proposed one-step distillation strategy, the NF reverse time is dramatically reduced from 0.1689 to 0.0076 seconds, yielding a 22× acceleration for this component. Consequently, the total inference time decreases from 0.2189 to 0.0567 seconds per image, nearly a 4× overall speed-up, while maintaining comparable image quality (FID 5.63 vs. 5.55, IS 193.49 vs. 194.63). These results demonstrate that one-step distillation effectively eliminates the sequential bottleneck of the reverse process, enabling FARMER to achieve both high fidelity and efficient generation.

Impact of Logdet As defined in the training objective (see Eq (14)), the log-determinant (logdet) loss term quantifies the volume change induced by the transformation from the original space to the target latent space. As illustrated in Figure 8, samples with abnormal logdet values often exhibit a blurred appearance and lack fine-grained details. Excessively large logdet values indicate that certain regions of the latent space are strongly compressed in the data space, which can lead to significant errors when reversing the transformation and reconstructing the data. This suggests that maintaining stable logdet values is crucial for high-fidelity and detail-preserving generation.

𝑃 𝑥 :13.8 𝑃 𝑧 :0.23 𝑫𝒆𝒕:𝟓𝟗.𝟒

𝑃 𝑥 :14.1 𝑃 𝑧 :0.23 𝑫𝒆𝒕:𝟔𝟎.𝟕

𝑃 𝑥 :13.8 𝑃 𝑧 :0.23 𝑫𝒆𝒕:𝟓𝟗.𝟎

𝑃 𝑥 :13.3 𝑃 𝑧 :0.23 𝑫𝒆𝒕:𝟓𝟕.𝟑

𝑃 𝑥 :13.5 𝑃 𝑧 :0.23 𝑫𝒆𝒕:𝟓𝟕.𝟖

𝑃 𝑥 :13.2 𝑃 𝑧 :0.23 𝑫𝒆𝒕:𝟓𝟔.𝟕

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

- Figure 8 The sample images with abnormal log-determinant values. High logdet values cause strong compression in parts of the data space, leading to blurred textures and missing fine-scale details in the generated images.

### 5 Conclusion

We introduce FARMER, a novel generative framework that integrate invertible AF with AR model, enabling end-to-end training directly on raw image pixels. FARMER learns by mapping the data distribution to an distribution modeled by the AR model and maximizing the negative log-likelihood of the raw images. This design permits both high-quality image synthesis and explicit likelihood estimation. Furthermore, we propose key techniques: a self-supervised dimension reduction to alleviate the complexity of AR modeling/sampling, a resampling-based CFG strategy to enhance image quality, and a one-step distillation scheme to accelerate the inference speed. Through the contributions, FARMER demonstrates competitive performance in image generation relative to pixel-based and latent generative models. However, beyond the curse of high-dimensionality that we have addressed, two challenges persist in NF–AR, i.e., (i) dequantization relying on noise injection and (ii) the complications arising from the log-determinant loss. We leave these for future works.

### 6 Related Work

#### 6.1 Continuous AR

A common paradigm in autoregressive image generation is to quantize images into discrete tokens [4, 34, 50, 51, 58, 67, 77] and train autoregressive models over them, as exemplified by LlamaGen [58], Janus-Pro [7], and SimpleAR [69]. However, this design suffers from a key bottleneck: quantization inevitably introduces information loss, which limits the fidelity of generated images [19, 36, 63].

To address this issue, GIVT [63] uses continuous latents obtained from a VAE to encode images and trains an AR model to predict GMM parameters for approximating token distributions. ARINAR [81] further predicts GMM parameters of each token in Gaussian-to-Gaussian paradigm. Since GMMs have limited expressive power, Tschannen et al. further introduce a NF to transform GMM samples into tokens, thereby improving generation quality. Jetformer [64] goes one step further by discarding the VAE and directly training AR and NF models in the pixel space.

Another line of work explores continuous token modeling by combining AR with diffusion models. In MAR [36], the AR backbone first outputs a conditioning vector for each token, and the diffusion head then generates the next tokens conditioned on this vector. Building on this idea, several other continuous-token approaches have been proposed [8, 9, 20, 27, 38, 41, 47, 52, 53, 61, 73, 74, 76, 82]. For example, FlowAR [52] employs a VAR [62] backbone with flow matching as the generative head; Hi-MAR [82] pivots on low-resolution image tokens to trigger hierarchical autoregressive modeling in a multi-phase manner; xAR [53] autoregressively generates next groups of tokens through flow matching. Although diffusion-based methods are effective at sampling continuous tokens, they require iterative noise-to-token denoising, which limits the model ability to perceive and understand images. In contrast, our model directly fits the token distribution without relying on noise sampling.

#### 6.2 Autoregressive Normalizing Flow

Normalizing flows (NF) [12–15, 17, 29, 31, 42, 43, 45, 54, 66] provide a powerful framework for density estimation, visual generation, and text generation [80], via invertible transformations, enabling exact likelihood computation and efficient sampling. However, the representational capacity of NFs is limited by the expressiveness of these invertible transformations. To address this limitation, autoregressive normalizing flows have been proposed, where each token is transformed conditioned on previous tokens. There has been a long line of work on autoregressive normalizing flows, with representative approaches including IAF [30], MAF [44], neural autoregressive flows [26], and T-NAF [48]. More recently, the resurgence of NFs has attracted renewed interest. TARFlow [79] leverages causal Transformers and simplifies the log-determinant term in the loss function, leading to notable improvements in generation quality. STARFlow extends TARFlow into the VAE latent space and demonstrates that continuous AR flows can deliver competitive generative performance. Meanwhile, JetFormer [64] integrates Jet [32] to enable fully end-to-end continuous AR modeling directly over raw image pixels.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

- [3] Andrew Brock. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018.

- [4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022.

- [5] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning. PMLR, 2020.

- [6] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025.

- [7] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [9] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

- [10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

- [12] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

- [13] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. In International Conference on Learning Representations, 2017.

- [14] Felix Draxler, Peter Sorrenson, Lea Zimmermann, Armand Rousselot, and Ullrich Köthe. Free-form flows: Make any architecture a normalizing flow. In International Conference on Artificial Intelligence and Statistics, pages 2197–2205. PMLR, 2024.

- [15] Felix Draxler, Stefan Wahl, Christoph Schnörr, and Ullrich Köthe. On the universality of volume-preserving and coupling-based normalizing flows. arXiv preprint arXiv:2402.06578, 2024.

- [16] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [17] Robert Giaquinto and Arindam Banerjee. Gradient boosted normalizing flows. Advances in Neural Information Processing Systems, 33:22104–22117, 2020.

- [18] Jiatao Gu, Tianrong Chen, David Berthelot, Huangjie Zheng, Yuyang Wang, Ruixiang Zhang, Laurent Dinh, Miguel Angel Bautista, Josh Susskind, and Shuangfei Zhai. Starflow: Scaling latent normalizing flows for high-resolution image synthesis. arXiv preprint arXiv:2506.06276, 2025.

- [19] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024.

- [20] Tiankai Hang, Jianmin Bao, Fangyun Wei, and Dong Chen. Fast autoregressive models for continuous latent generation. arXiv preprint arXiv:2504.18391, 2025.

- [21] Wanggui He, Siming Fu, Mushui Liu, Xierui Wang, Wenyi Xiao, Fangxun Shu, Yi Wang, Lei Zhang, Zhelun Yu, Haoyuan Li, et al. Mars: Mixture of auto-regressive models for fine-grained text-to-image synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 17123–17131, 2025.

- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017.

- [23] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022.

- [24] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023.

- [25] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024.

- [26] Chin-Wei Huang, David Krueger, Alexandre Lacoste, and Aaron Courville. Neural autoregressive flows. In International conference on machine learning, pages 2078–2087. PMLR, 2018.

- [27] Guolin Ke and Hui Xue. Hyperspherical latents improve continuous-token autoregressive generation. arXiv preprint arXiv:2509.24335, 2025.

- [28] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

- [29] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31, 2018.

- [30] Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems, 29, 2016.

- [31] Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence, 43(11):3964–3979, 2020.

- [32] Alexander Kolesnikov, André Susano Pinto, and Michael Tschannen. Jet: A modern transformer-based normalizing flow. arXiv preprint arXiv:2412.15129, 2024.

- [33] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems, 32, 2019.

- [34] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022.

- [35] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.

- [36] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024.

- [37] Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models. arXiv preprint arXiv:2502.17437, 2025.

- [38] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

- [39] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024.

- [40] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.

- [41] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751, 2025.

- [42] Bálint Máté, Samuel Klein, Tobias Golling, and François Fleuret. Flowification: Everything is a normalizing flow. Advances in Neural Information Processing Systems, 35:35478–35489, 2022.

- [43] Emile Mathieu and Maximilian Nickel. Riemannian continuous normalizing flows. Advances in neural information processing systems, 33:2503–2515, 2020.

- [44] George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. Advances in neural information processing systems, 30, 2017.

- [45] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.

- [46] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning. PMLR, 2018.

- [47] Marco Pasini, Javier Nistal, Stefan Lattner, and George Fazekas. Continuous autoregressive models with noise augmentation avoid error accumulation. arXiv preprint arXiv:2411.18447, 2024.

- [48] Massimiliano Patacchiola, Aliaksandra Shysheya, Katja Hofmann, and Richard E Turner. Transformer neural autoregressive flows. arXiv preprint arXiv:2401.01855, 2024.

- [49] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

- [50] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning. Pmlr, 2021.

- [51] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

- [52] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Flowar: Scale-wise autoregressive image generation meets flow matching. arXiv preprint arXiv:2412.15205, 2024.

- [53] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025.

- [54] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pages 1530–1538. PMLR, 2015.

- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- [56] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

- [57] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in Neural Information Processing Systems, 29, 2016.

- [58] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

- [59] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [60] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

- [61] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025.

- [62] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

- [63] Michael Tschannen, Cian Eastwood, and Fabian Mentzer. GIVT: Generative infinite-vocabulary Transformers. arXiv:2312.02116, 2023.

- [64] Michael Tschannen, André Susano Pinto, and Alexander Kolesnikov. Jetformer: An autoregressive generative model of raw images and text. arXiv preprint arXiv:2411.19722, 2024.

- [65] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016.

- [66] Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning. PMLR, 2016.

- [67] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

- [68] Steven Walton, Valeriy Klyukin, Maksim Artemev, Denis Derkach, Nikita Orlov, and Humphrey Shi. Distilling normalizing flows. In Proceedings of the Computer Vision and Pattern Recognition Conference, 2025.

- [69] Junke Wang, Zhi Tian, Xun Wang, Xinyu Zhang, Weilin Huang, Zuxuan Wu, and Yu-Gang Jiang. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455, 2025.

- [70] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Exploring dcn-like architecture for fast image generation with arbitrary resolution. Advances in Neural Information Processing Systems, 37:87959–87977, 2024.

- [71] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025.

- [72] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.

- [73] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

- [74] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025.

- [75] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

- [76] Hu Yu, Hao Luo, Hangjie Yuan, Yu Rong, and Feng Zhao. Frequency autoregressive image generation with continuous tokens. arXiv preprint arXiv:2503.05305, 2025.

- [77] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.

- [78] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

- [79] Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, David Berthelot, Jiatao Gu, Huangjie Zheng, Tianrong Chen, Miguel Angel Bautista, Navdeep Jaitly, and Josh Susskind. Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329, 2024.

- [80] Ruixiang Zhang, Shuangfei Zhai, Jiatao Gu, Yizhe Zhang, Huangjie Zheng, Tianrong Chen, Miguel Angel Bautista, Josh Susskind, and Navdeep Jaitly. Flexible language modeling in continuous space with transformer-based autoregressive flows. arXiv preprint arXiv:2507.00425, 2025.

- [81] Qinyu Zhao, Stephen Gould, and Liang Zheng. Arinar: Bi-level autoregressive feature-by-feature generative models. arXiv preprint arXiv:2503.02883, 2025.

- [82] Guangting Zheng, Yehao Li, Yingwei Pan, Jiajun Deng, Ting Yao, Yanyong Zhang, and Tao Mei. Hierarchical masked autoregressive models with low-resolution token pivots. arXiv preprint arXiv:2505.20288, 2025.

## Appendix

### A Discussions

#### A.1 FARMER reduces to Autoregressive Flow when K = 1

When the number of components (K) in the Gaussian Mixture Model (GMM) predicted by FARMER is set to one (K = 1), FARMER reduces to an Autoregressive Flow (AF). In this case, each token zi in the sequence is modeled by a conditional Gaussian distribution, where the mean and variance are functions of the preceding tokens z<i. The optimization objective for each token becomes:

log p(zi|z<i) = log N(zi;µ(z<i),σ2(z<i)) (16) This can be further expressed as:

log p(zi|z<i) = log N

zi − µ(z<i) σ(z<i)

;0,Id

∂ [zi − µ(z<i)]/σ(z<i) ∂zi

, (17)

where N(·;0,Id) denotes the standard normal density, and the second term inside the log corresponds to the change of variables formula (the volume correction by the Jacobian determinant).

Expanding the log yields two components:

log p(zi|z<i) = log N

zi − µ(z<i) σ(z<i)

;0,Id + log

1 σ(z<i)

, (18)

i−µ(z<i)

zi is transformed to new token z

σ(z<i) by the predicted results (µ(z<i),σ(z<i)) of the AR model conditioned on preceding tokens z<i, and this transformation is invertible; the first term is the log-likelihood of new token zi′ under the standard Gaussian distribution, and the second term is the log-determinant of the Jacobian of the affine transformation. Thus, the AR model can be considered as the last block of AFs.

This confirms that when K = 1, FARMER reduces to an Autoregressive Flow.

#### A.2 Resample-based CFG

While the main text (Section 3.4) outlines the proposed Resampling-based Classifier-Free Guidance (CFG) method, we further elaborate on additional tunable parameters that enhance generation quality and control diversity. Each of the three stages—Propose, Weigh, and Resample—introduces dedicated temperature coefficients and sampling numbers that can be adjusted.

Propose stage. In the proposal step, candidate samples are drawn from the conditional GMM pc(zi) and the unconditional GMM pu(zi). To control the diversity at this stage, we introduce two distinct temperature coefficients:

- • Weight temperature Tπ: applied multiplicatively to the mixture weights πk(z<i) of the GMM components before normalization. This modulates the relative selection probability among Gaussian components.

- • Variance temperature Tσ: applied multiplicatively to the variance σk(z<i) of each Gaussian component, scaling the spread of proposals.

Additionally, the number of samples drawn from pc and pu can differ; we denote these by sc and su. This allows balancing between strongly conditioned proposals and broader unconditional exploration.

Weigh stage. Given candidate samples zi,j, their importance weights are computed as:

log ωi,j = w · log pc(zi,j;Tπ,v,Tσ,v) − log pu(zi,j;Tπ,v,Tσ,v) ,

where Tπ,v and Tσ,v are temperature coefficients for the evaluation distributions in this stage (not necessarily equal to those used in the Propose stage). These temperatures control the sharpness or smoothness of the scoring in the log-probability space.

Resample stage. Finally, the normalized weights ωi,j define a categorical distribution. To further modulate selection sharpness, we introduce a resampling temperature Ts applied uniformly to all log-weights before normalization:

pfinal(zi,j) ∝ exp log ωi,j ∗ Ts . Higher Ts emphasizes high-weight proposals, while lower Ts encourages diversity. Summary table of parameter choices. Table 6 summarizes the temperature and sampling configurations used for different model variants evaluated in this work.

Table 6 Temperature and sampling parameters for Resampling-based CFG in different models.

Model Tπ Tσ sc su Tπ,v Tσ,v Ts CFG

FARMER 1.1B (patch 16) 1.0 0.9 5 5 0.2 0.9 1.1 2.5 FARMER 1.1B (patch 8) 1.0 1.0 5 5 0.2 0.9 1.1 2.0 FARMER 1.9B (patch 16) 1.0 0.9 5 5 0.2 0.9 1.1 3.5 FARMER 1.9B (patch 8) 1.0 1.0 5 5 0.1 1.0 1.1 1.5

