arXiv:2506.10038v1[cs.GR]10Jun2025

# Ambient Diffusion mni: Training Good Models with Bad Data

### Giannis Daras ∗

Massachusetts Institute of Technology gdaras@mit.edu

Adam Klivans The University of Texas at Austin klivans@utexas.edu

Adrian Rodriguez-Munoz ∗ Massachusetts Institute of Technology adrianrm@mit.edu

Antonio Torralba Massachusetts Institute of Technology torralba@mit.edu

Constantinos Daskalakis Massachusetts Institute of Technology costis@csail.mit.edu

## Abstract

We show how to use low-quality, synthetic, and out-of-distribution images to improve the quality of a diffusion model. Typically, diffusion models are trained on curated datasets that emerge from highly filtered data pools from the Web and other sources. We show that there is immense value in the lower-quality images that are often discarded. We present Ambient Diffusion Omni, a simple, principled framework to train diffusion models that can extract signal from all available images during training. Our framework exploits two properties of natural images – spectral power law decay and locality. We first validate our framework by successfully training diffusion models with images synthetically corrupted by Gaussian blur, JPEG compression, and motion blur. We then use our framework to achieve stateof-the-art ImageNet FID and we show significant improvements in both image quality and diversity for text-to-image generative modeling. The core insight is that noise dampens the initial skew between the desired high-quality distribution and the mixed distribution we actually observe. We provide rigorous theoretical justification for our approach by analyzing the trade-off between learning from biased data versus limited unbiased data across diffusion times.

## 1 Introduction

Large-scale, high-quality training datasets have been a primary driver of recent progress in generative modeling. These datasets are typically assembled by filtering massive collections of images sourced from the web or proprietary databases [24, 42, 51, 56, 57]. The filtering process—which determines which data is retained—is crucial to the quality of the resulting models [12, 26, 24, 31, 26]. However, filtering strategies are often heuristic and inefficient, discarding large amounts of data [49, 42, 24, 12]. We demonstrate that the data typically rejected as low-quality holds significant, underutilized value.

Extracting meaningful information from degraded data requires algorithms that explicitly model the degradation process. In generative modeling, there is growing interest in approaches that learn to generate directly from degraded inputs [15, 16, 17, 13, 7, 46, 38, 50, 5, 1, 2, 53, 69, 45, 62, 44]. A

∗Equal contribution.

Preprint.

key limitation of existing methods is their reliance on knowing the exact form of the degradation. In real-world scenarios, image degradations—such as motion blur, sensor artifacts, poor lighting, and low resolution—are often complex and lack a well-defined analytical description, making this assumption unrealistic. Even within the same dataset, from ImageNet to internet scale text-to-image datasets, there are samples of heterogeneous qualities [27], as shown in Figures 4, 25, 28, 26. Given access to this mixed-bag of datapoints, we would like to sample from a tilted continuous measure of high-quality images, without sacrificing the diversity present in the training points.

Text-to-image Results

"Close-up of a fire spitting dragon, cinematic shot."

"A photograph of an astronaut riding a pig."

"A realistic photo of a camel in Antarctica."

"A realistic photo of a dolphin on a mountain top."

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

BaselineAmbient-oBaselineAmbient-o

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Image et Results

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 1: Effect of using Ambient-o for (a) training a text-to-image model (Micro-Diffusion [52]) and (b) a class-conditional model for ImageNet (EDM-2 [35]). All generations are initialized with the same noise. The baseline models are trained using all the data equally. Ambient-o changes the way the data is used during the diffusion process based on its quality. This leads to significant visual improvements without sacrificing diversity, as would happen with a filtering approach (see Fig. 8).

The training objective of diffusion models naturally decomposes sampling from a target distribution into a sequence of supervised learning tasks [29, 59, 60, 14, 18, 9, 10]. Due to the power-law structure of natural image spectra [63], high diffusion times focus on generating globally coherent, semantically meaningful content [21], while low diffusion times emphasize learning high-frequency details.

Our first key theoretical insight is that low-quality samples can still be valuable for training in the high-noise regime. As noise increases, the diffusion process contracts distributional differences (see

Theorem 4.2), reducing the mismatch between the high-quality target distribution and the available mixed-quality data. At the same time, incorporating low-quality data increases the sample size, reducing the variance of the learned estimator. Our analysis formalizes this bias–variance trade-off and motivates a principled algorithm for training denoisers at high diffusion times using noisy, heterogeneous data.

For low diffusion times, our algorithm leverages a second key property of natural images: locality. We show a direct relationship between diffusion time and the optimal receptive field size for denoising. Specifically, small image crops suffice at lower noise levels. This allows us to borrow high-frequency details from out-of-distribution or synthetic images, as long as the marginal distributions of the crops match those of the target data.

We introduce Ambient Diffusion Omni (Ambient-o), a simple and principled framework for training diffusion models using arbitrarily corrupted and out-of-distribution data. Rather than filtering samples based on binary ‘good’ or ‘bad’ labels, Ambient-o retains all data and modulates the training process according to each sample’s utility. This enables the model to generate diverse outputs without compromising image quality. Empirically, Ambient-o advances the state of the art in unconditional generation on ImageNet and enhances diversity in text-conditional generation without sacrificing fidelity. Theoretically, it achieves improved bounds for distribution learning by optimally balancing the bias–variance trade-off: low-quality samples introduce bias, but their inclusion reduces variance through increased sample size.

We will release all our code and trained models in the following URL: https://github.com/giannisdaras/ambient-omni.

## 2 Background and Related Work

Diffusion Modeling. Diffusion models transform the problem of sampling from p0 into the problem of learning denoisers for smoothed versions of p0 defined as pt = p0 ⊛ N(0,σ2(t)I). We typically denote with X0 ∼ p0 the R.V. distributed according to the distribution of interest and Xt = X0 + σ(t)Z, the R.V. distributed according to pt. The target is to estimate the set of optimal l2 denoisers, i.e., the set of the conditional expectations: {E[X0|Xt = ·]}Tt=1. Typically, this can be achieved through supervised learning by minimizing the following loss (or a re-parametrization of it):

0,xt|t ||hθ(xt,t) − x0||2 , (2.1)

J(θ) = Et∈U[0,T]Ex

that is optimized over a function family H = {hθ : θ ∈ Θ} parametrized by network parameters θ. For sufficiently expressive families, the minimizer is indeed: hθ∗(x,t) = E[X0|Xt = x].

Learning from noisy data. The diffusion modeling framework described above assumes access to samples from the distribution of interest p0. An interesting variation of this problem is to learn to sample from p0 given access to samples from a tilted measure p˜0 and a known degradation model. In Ambient Diffusion [15], the goal is to sample from p0 given pairs (Ax0,A) for a matrix A : Rm×n,m < n, that is distributed according to a known density p(A). The techniques in this work were later generalized to accommodate additive Gaussian Noise [13, 16, 1] in the measurements. More recently there have been efforts to further broaden the family of degradation models considered through Expectation-Maximization approaches that involve multiple training runs [50, 5].

Recent work from [16] has shown that, at least for the Gaussian corruption model, leveraging the low-quality data can tremendously increase the performance of the trained generative models. In particular, the authors consider the setting where we have access to a few samples from p0, let’s denote them D0{x(0i)}N

, let’s denote them Dtn{x(ti)

#### }N

i=1, where ptn = p0 ⊛ N(0,σ2(tn)I) is a smoothed version of p0 at a known noise level tn. The clean samples are used to learn denoisers for all noise levels t ∈ [0,T] while the noisy samples are used to learn denoisers only for t ≥ tn, using the training objective:

i=1 and many samples from pt

1

2

n

n

N2

2

t|x(tin) α(t)hθ(xt,t) + (1 − α(t))xt − x(ti)

, (2.2)

Jambient(θ) = Et∈U(t

Ex

n,T]

n

i=1

2(t)−σ2(tn)

with α(t) = σ

σ2(t) . Note that the objective of equation 2.2 only requires samples from pt

n

(instead of p0) and can be used to train for all times t ≥ tn. This algorithm uses N1 + N2 datapoints

to learn denoisers for t > tn and only N1 datapoints to learn denoisers for t ≤ tn. The authors show that even for N1 << N2, the model performs similarly to the setting of training with (N1 + N2) clean datapoints. The main limitation of this method and its related works is that the degradation process needs to be known. However, in many applications, we have data from heterogeneous sources and various qualities, but there is no analytic form or any prior on the corruption model.

Data filtering. One of the most crude, but widely used, approaches for dealing with heterogeneous data sources is to remove the low-quality data and train only the high-quality subset [42, 24, 22]. While this yields better results than naively training on the entire distribution, it leads to a decrease in diversity and relies on heuristics for optimizing the filtering. An alternative strategy is to train on the entire distribution and then fine-tune on high-quality data [12, 52]. This approach better trades the quality-diversity trade-off but still incurs a loss of diversity and is hard to calibrate.

Training with synthetic data. A lot of recent works have shown that synthetic data can improve the generative capabilities of diffusion models when mixed properly with real data from the distribution of interest [23, 3, 4]. In this work, we show that it helps significantly to view synthetic data as corrupted versions of the samples from the real distribution and incorporate this perspective into the training objective.

## 3 Method

We propose a new framework that extends beyond [16] to enable training generative models directly from arbitrarily corrupted and out-of-distribution data, without requiring prior knowledge of the degradation process. We begin by formalizing the setting of interest.

Problem Setting. We are given a dataset D = {w0(i)}Ni=1 consisting of N datapoints. Each point in D is drawn from a mixture distribution p˜0, which mixes p0 (the distribution of interest) and an alternative distribution q0 that may contain various forms of degradation or out-of-distribution content. We assume access to two labeled subsets, SG,SB, where points in SG are known to come from the clean distribution p0, and points in SB from the corrupted distribution q0. While this assumption simplifies the initial exposition, we relax it in Section E.1. We focus on the practically relevant regime where |SG|≪ |D|—i.e., access to high-quality data is severely limited. The objective is to learn a generative model that (approximately) samples from the clean distribution p0, leveraging both clean and corrupted samples in its training.

We now describe how degraded and out-of-distribution samples can be effectively leveraged during training in both the high-noise and low-noise regimes of the diffusion process.

### 3.1 Learning in the high-noise regime (leveraging low-quality data)

Addition of gaussian noise contracts distribution distances. The first key idea of our method is that, at high diffusion times t, the noised target distribution pt and the noised corrupted distribution p˜t become increasingly similar (Theorem 4.2), effectively attenuating the discrepancy introduced by corruption. This effect is illustrated in Figure 2 (top), where we compare a clean image and its degraded counterpart (in this case, corrupted by Gaussian blur). As the diffusion time t increases, the noised versions of both samples become visually indistinguishable. Consequently, samples from p˜0 can be leveraged to learn (the score of) pt, for t > tminn . We formalize this intuition in Section 4, and we also quantify that for large t there are statistical efficiency benefits for using a large sample from p˜0 versus a small sample from p0 .

Heuristic selection of the noise level. From the discussion so far, it follows that to use samples from p˜0, we need to assign them to a noise level tminn . One can select this noise level empirically, i.e. we can ablate this parameter by training different models and selecting the one that maximizes the generative performance. However, this approach requires multiple trainings, which can be costly. Instead, we can find the desired noise level in a principled way as detailed below.

Training a classifier under additive Gaussian noise. To identify the appropriate noise level, we train a time-conditional classifier to distinguish between the noised distributions pt and qt across various diffusion times t. We use a single neural network cnoiseθ (xt,t) that is conditioned on the

Blurry image unsafe to use Merging point σmin Distributions merge and blurry image is safe to use

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | |[Figure 18]| | | | | |

Clean image

Blurry image

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

Classifieroutput

### Figure 2: A time-dependent classifier trained to distinguish noisy clean and blurry images (blur

kernel standard deviation σB = 0.6). At low noise the classifier is able to perfectly identify the blurry images, and outputs a probability close to 0. As the noise increases and the information in the image is destroyed, the clean and blurry distributions converge and the classifier outputs a prediction close to 0.5. The red line plots the threshold (selected at τ = 0.45), which is crossed at σt = 1.64.

diffusion time t, following the approach of time-aware classifiers used in classifier guidance [20]. The classifier is trained using labeled samples from SG (clean) and SB (corrupted) via the following objective:

t|y0 −log(1 − cnoiseθ (yt,t)) (3.1)

t|x0 −log cnoiseθ (xt,t) +

Ey

Ex

Jnoise(θ) =

y0∈SB

x0∈SG

Annotation. Once the classifier is trained, we use it to determine the minimal level of noise that must be added to the low-quality distribution q0 so that it closely approximates a smoothed version of the high-quality distribution p0. Formally, we compute:

 

 

1 |SB| y

t|y0 cnoiseθ (yt,t) > τ

tminn = inf

, (3.2)

Ey

t ∈ [0,T] :





0∈SB

for τ = 0.5 − ϵ and for some ϵ > 0. Subsequently, we form the annotated dataset Dannot = {(w0(i)+σtmin

Z(i),tminn )}Ni=1∪{(x0,0)|x0 ∈ SG}, where the random variables Z(i) are i.i.d. standard normals. In particular, our annotated dataset indicates that we should only use the samples from D for diffusion times t ≥ tminn , for which the distributions have approximately merged and hence it is safe to use them. In fact, the optimal classifier assigns time tn that corresponds to the first time for which dTV(pt,qt) ≤ ϵ.

n

Sample dependent annotation. One potential issue with the aforementioned annotation approach is that all the samples in D are treated equally. But, as we noted, the points in D could be drawn from a distribution p˜0 that mixes p0 and q0. In this case, all the samples in D that came from the p0 component, will still get a high annotation time, leading to information loss. Instead, we can opt-in

for a sample-wise annotation scheme, where each sample w0(i) gets assigned a time tmini based on: tmini = inf{t ∈ [0,T] : Ew

t|w0(i) cnoiseθ (wt,t) > τ}, for τ = 0.5 − ϵ and for some ϵ > 0.

From arbitrary corruption to additive Gaussian noise. The afore-described approach reduces our problem of learning from data with arbitrary corruption to the setting of learning from data corrupted with additive Gaussian noise. The price we pay for this reduction is the information loss due to the extra noise we add to the samples during the annotation stage. We can now extend the objective function (2.2) to train our diffusion model. Suppose our annotated dataset is comprised of

samples {(x(ti)

min i

,tmini )}. Then our objective becomes:

Jambient−o(θ) = Et∈U[0,T]

i:tmini <t

Ex

t|x(i)

tmin i

α(t,tmini )hθ(xt,t) + (1 − α(t,tmini ))xt − x(ti)

min i

2

,

2(t)−σ2(tmini )

where α(t,tmini ) = σ

σ2(t) .

Learning something from nothing? The proposed framework comes with limitations worth considering. First, unless the diffusion noise level tends to infinity, the distributions pt and qt never fully converge—there is always a bias when treating samples from qt as if they were from pt. Moreover, the method is particularly well-suited to certain types of corruptions but is less effective for others. Because the addition of Gaussian noise suppresses high-frequency components—due to the spectral power law of natural images—our approach is most effective for corruptions that primarily degrade high frequencies (e.g., blur). In contrast, degradations that affect low-frequency content—such as color shifts, contrast reduction, or fog-like occlusions—are more challenging. This limitation is illustrated in Figure 3: masked images, for example, require significantly more noise to become usable compared to high-frequency corruptions like blur. In the extreme, the method reduces to a filtering approach, as infinite noise nullifies all information in the corrupted samples.

HQ images used by all methods

| |
|---|

HQ data excluded by  ltering but used by Ambient-

[Figure 19]

LQ data included by training on everything

[Figure 20]

HQ

Gaussian Blur

JPEG

MotionBlur

0

Masking

σt

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

0.0

0.2 0.3 0.5 0.7 0.9 1.3 2.1 32

- Figure 3: Visual summary of our method for using low-quality data at high-noise. We see how the various corrupted images become indistinguishable from the High Quality (HQ) after a minimum noise level. These noisy versions of Low Quality (LQ) images are actually high-quality data, which filtering approaches discard, but Ambient Omni uses.

### 3.2 Learning in the low-noise regime (synthetic and out-of-distribution data)

So far, our algorithm implicitly results in varying amounts of training data across diffusion noise levels. At high noise, the model can leverage abundant low-quality data, whereas at low noise levels, it must rely solely on the limited set of high-quality samples. We now extend the algorithm to enable the use of synthetic and out-of-distribution data for learning denoisers at low-noise diffusion times.

To achieve this, we leverage another fundamental property of natural images: locality. At low diffusion times, the denoising task can be solved using only a small local region of the image, without requiring full spatial context. We validate this hypothesis experimentally in the Experiments Section (Figures 15, 16, 17, 18), where we show that there is a mapping between diffusion time t and the

crop size needed to perform the denoising optimally at this diffusion time. Intuitively, the higher the noise, the more context is required to accurately reconstruct the image. Conversely, for lower noise, the local information within a small neighborhood suffices to achieve effective denoising. We use crop(t) to denote the minimal crop size needed to perform optimal denoising at time t. If there are two distributions p0 and p˜0 that agree on their marginals (i.e. crops), they can be used interchangeably for low-diffusion times. Note that the distributions don’t have to agree globally, they only have to agree on a local (patch) level. Formally, let A(t) be a random patch selector of size crop(t). Let also p0,p˜0 two distributions that satisfy:

#### A(t)#p0 = A(t)#˜p0, (3.3)

where A(t)#p0 denotes the pushforward measure2 of p0 under A(t). Then, the cropped portions of the tilted distributions provide equivalent information to the crops of the original distribution for denoising.

Training a crops classifier. Note that the condition of Equation (3.3) can be trivially satisfied if A(t) masks all the pixels or even if A(t) just selects a single pixel. We are interested in finding what is the maximum crop size for which this condition is approximately true. Once again, we can use a classifier to solve this task. The input to the classifier, ccropsθ , is a crop of an image that either arises from p0 or p˜0, and the classifier needs to classify between these two cases.

Annotation and training using the trained classifier. Once the classifier is trained, we are now interested in finding the biggest crop size for which the distributions p0,p˜0 cannot be confidently distinguished. Formally,

 

 

1 |SB| y

[ccropsθ (A(t)(yt))] > τ

tmaxn = sup

, (3.4)

t ∈ [0,T] :





0∈SB

for τ = 0.5 − ϵ and for some small ϵ > 03. For times t ≤ tmaxn , the out-of-distribution images from p˜0 can be used with the regular diffusion objective as images from p0, as for these times the denoiser only looks at crops and at the crop level the distributions have converged.

The donut paradox. Each sample can be used for t ≥ tmini and for t ≤ tmaxi , but not for t ∈ (tmaxi ,tmini ). We call this the donut paradox as there is a hole in the middle of the diffusion trajectory for which we have fewer available data. These times do not have enough noise for the distributions to merge globally, but also the required receptive field for denoising is big enough so that there are differences on a crop level. We show an example of this effect in Figure 14.

Table 1: ImageNet results with and without classifier-free guidance.

|ImageNet-512<br><br>|Train FID ↓| |Test FID ↓<br><br>| |Model Size Mparams NFE|
|---|---|---|---|---|---|
| |FID<br><br>no CFG w/ CFG|FIDv2 no CFG w/ CFG|FID no CFG w/ CFG|FIDv2 no CFG w/ CFG| |
|EDM2-XS Ambient-o-XS<br><br>|3.57 2.91 3.59 2.89|103.39 79.94 107.26 79.56|3.77 3.68 3.69 3.58|115.16 93.86 115.02 92.96|125 63 125 63|
|EDM2-XXL<br><br>Ambient-o-XXL Ambient-o-XXL+crops|1.91 (1.93) 1.81 1.99 1.87 1.91 1.80|42.84 33.09<br>43.38 33.34 42.84 32.63<br>|2.88 2.73 2.81 2.68 2.78 2.53|56.42 46.22 56.40 46.02 56.39 45.78|1523 63 1523 63 1523 63|

- 4 Theory We study the 1-d case, but all our claims easily extend to any dimension. We compare two algorithms:

- Algorithm 1. Algorithm 1 trains a diffusion model using access to n1 samples from a target density p0, assumed to be supported in [0,1] and be λ1-Lipschitz.
- Algorithm 2. Algorithm 2 trains a diffusion model using access to n1 + n2 samples from a density p˜0 that is a mixture of the a target density p0 and another density q0, assumed to be supported in [0,1]

n1+n2 p0 + n

and be λ2-Lipschitz: p˜0 = n

n1+n2 q0.

1

2

- 2Given measure spaces (X1, Σ1) and (X2, Σ2), a measurable function f : X1 → X2, and a probability measure p : Σ1 → [0, ∞), the pushforward measure f#p is defined as (f#p)(B) := p(f−1(B)) ∀B ∈ Σ2.
- 3We subtract an ϵ to allow for approximate mixing of the two distributions and hence smaller annotation times.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

High quality images Low quality images

Figure 4: Results using CLIP to obtain the high-quality and the low-quality sets of ImageNet.

We want to compare how well these algorithms estimate the distribution pt := p0 ⊛ N(0,σt2). We use pˆ(1)t ,pˆ(2)t to denote the estimates obtained for pt by Algorithms 1 and 2 respectively.

Diffusion modeling is Gaussian kernel density estimation. We start by making a connection between the optimal solution to the diffusion modeling objective and kernel density estimation. Given

a finite dataset {W(i)}ni=1, the optimal solution to the diffusion modeling objective should match the empirical density at time t, which is:

n

W(i) − x σt

1 nσt

, (4.1)

ϕ

pˆt(x) =

i=1

2/2 is the Gaussian kernel. We observe that equation 4.1 is identical to a Gaussian kernel density estimate, given samples {W(i)}ni=1

where ϕ(u) = √12πe−u

4. We establish the following result for Gaussian kernel density estimation.

- Theorem 4.1 (Gaussian Kernel Density Estimation). Let {W(i)}ni=1 be a set of n independent samples from a λ-Lipschitz density p. Let pˆ be the empirical density, pσ := p ⊛ N(0,σ2) and pˆσ = pˆ⊛ N(0,σ2). Then, with probability at least 1 − δ with respect to the sample randomness,

log n + log(1 ∨ λ) + log 2/δ σ2n

1 n

1 σ2n

dTV(pσ,pˆσ) ≲

. (4.2)

+

+

The proof of this result is given in the Appendix. Comparing the performance of Algorithms 1 and 2. Applying Theorem 4.1 directly to the p0 density, we immediately get that the estimate pˆ(1)t (x) obtained by Algorithm 1 satisfies:

log n1 + log(1 ∨ λ1) + log 2/δ σt2n1

1 n1

1 σt2n1

dTV(pt,pˆ(1)t ) ≲

. (4.3)

+

+

Let us now see what we get by applying Theorem 4.1 to Algorithm 2, which uses samples from the tilted distribution p˜0. Since this distribution is n1

n1+n2 λ1 + n

n1+n2 λ2 -Lipschitz, we get that:

2

log(n1 + n2) + log(1 ∨ n

n1+n2 λ1 + n

n1+n2 λ2) + log 2/δ σt2(n1 + n2)

1

2

1 (n1 + n2)

1 σt2(n1 + n2)

dTV(˜pt,pˆ(2)t ) ≲

+

+

,

4This connection has been observed in prior works too, e.g., see [32, 8].

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

(a) High quality crops (b) Low quality crops

Figure 5: Results using CLIP to find (a) high-quality and (b) low-quality crops on ImageNet.

where p˜t := p˜0 ⊛ N(0,σt2). Further, we have that: dTV(pt,pˆ(2)t ) ≤ dTV(˜pt,pt) + dTV(˜pt,pˆ(2)t ). We already have a bound for the second term. To bound the first term, we prove the following theorem.

- Theorem 4.2 (Distance contraction under noise). Consider distributions P and Q supported on a subset of Rd with diameter D. Then

D 2σ

dTV(P ⊛ N(0,σ2I),Q ⊛ N(0,σ2I)) ≤ dTV(P,Q) ·

.

· n

Applying this theorem we get that: dTV(˜pt,pt) ≤ 21σ

n1+n2 dTV(p0,q0), where for the second inequality we used that dTV(p0,p˜0) ≤ n

#### dTV(˜p0,p0) ≤ 21σ

2

t

t

n1+n2dTV(p0,q0). Putting everything together, Algorithm (2) achieves an estimation error:

2

1 (n1 + n2)

+

1 σt2(n1 + n2)

+

dTV(pt,pˆ(2)t ) ≲

log(n1 + n2) + log(1 ∨ n

n1+n2 λ1 + n

n1+n2 λ2) + log 2/δ σt2(n1 + n2)

1

2

+

n2 σt(n1 + n2)

dTV(p0,q0).

Comparing this with the bound obtained in Equation 4.3, we see that if n2 is sufficiently larger than n1 or if λ2 ≤ λ1, there is a tminn such that for any t ≥ tminn , the upper-bound obtained by Algorithm 2 is better than the upper-bound obtained by Algorithm 1. That implies that for high-diffusion times, using biased data might be helpful for learning, as the bias term (final term) decays with the amount of noise. Going back to equation 4, note that the switching point t ≥ tminn depends on the distance dTV(˜pt,pt) that decays as shown in Theorem 4.2. Once this distance becomes small enough, our computations above suggest that we benefit from biased data. The classifier of Section 3.1, if optimal, exactly tracks the distance dTV(˜pt,pt) and, as a result, tracks the switching point.

## 5 Experiments

Controlled experiments to show utility from low-quality data. To verify our method, we first do synthetic experiments on artificially corrupted data. We use EDM [34] as our baseline, and we train networks on CIFAR-10 and FFHQ. For the first experiments, we only use the high-noise part of our Ambient-o method (Section 3.1). We underline that for all of our experiments, we only change the way we use the data, and we keep all the optimization and network hyperparameters as is. We compare against using all the data as equal (despite the corruption) and the filtering strategy of only training on the clean samples. For evaluation, we measure FID [28] with respect to the full uncorrupted dataset (which is not available during training).

For the blurring experiments, we use a Gaussian kernel with standard deviation σB = 0.4,0.6,0.8,1.0, and we corrupt 90% of the data. We show some example corrupted images in Appendix Figure 10a. To perform the annotations for our method, we train a blurry image vs clean image classifier under noise, as explained in Section 3.1. For the experiments in the main paper, we use a balanced dataset for the training of the classifier. We ablate the effect of having fewer training samples for the classifier training in Appendix Section D where we show that reducing the number of clean samples available for classifier training leads to a small drop in performance. Once equipped with the trained classifier, each sample is annotated on its own based on the amount of noise that is needed to confuse the classifier (sample dependent annotation). We present our results in Table 2a. As shown, for all corruption strengths, Ambient Omni, significantly outperforms the two baseline methods. In the one to the last column of Table 2a, we further show the average annotation of the classifier. As expected, the average assigned noise level increases as the corruption intensifies.

Ablations. We ablate the choice of using a fixed annotation vs sample-adaptive annotations in Appendix Table 7. We find that using sample-adaptive annotations achieves improved results. Nevertheless, both annotation methods yield improvements over the training on filtered data and the training on everything baselines. To show that our method works for more corruption types, we perform an equivalent experiment with JPEG compressed data at different compression ratios and we achieve similar results, presented in Appendix Table 3. We ablate the impact of the amount of training data and the number of training iterations on the classifier annotations in Appendix Section D. We show results for motion blur (Figure 11 and Section B.1) and for the FFHQ dataset (Table 4).

Table 2: In a controlled experiment with restricted access only to 10% of the clean dataset, our method of Ambient-o uses corrupted and out-of-distribution data to improve performance.

(a) Gaussian blurred data at different levels.

(b) Additional out-of-distribution data.

Method Parameters Values (σB) σ¯min

tn FID Only Clean (10%) - - 8.79

1.0

45.32 0.8 28.26 0.6 11.42

All data

0

- 0.4 2.47

Ambient-o

- 1.0 2.84 6.16 0.8 1.93 6.00 0.6 1.38 5.34 0.4 0.22 2.44

Source Data Additional Data Method σ¯max

tn FID

None – – 12.08 Cats Fixed σ 0.2 11.14 Cats Fixed σ 0.1 9.85 Cats Fixed σ 0.05 10.66 Cats Fixed σ 0.025 12.07 Cats Classifier 0.09 8.92

Dogs (10%)

Procedural Classifier 0.042 10.98 Cats (10%)

None – – 5.20 Dogs Classifier 0.13 5.11

Wildlife Classifier 0.08 4.89

Controlled experiments to show utility from out-of-distribution images. We now want to validate the method developed in Section 3.2 for leveraging crops from out-of-distribution data. To start with,

we want to find the mapping between diffusion times and the size of the receptive field required for an optimal denoising prediction. To do so, we take a pre-trained denoising diffusion model and measure the denoising loss at a given location as we increase the size of the context. We provide the corresponding plot in the Supplemental Figures 17, 15. The main finding is that while providing more context always leads to a decrease in the average loss, for sufficiently small noise levels, the loss nearly plateaus before the full image context is provided. That implies that the perfect denoiser for a given noise level only needs to look at a localized part of the image.

Equipped with the mapping between diffusion times and crop sizes, we now proceed to a fun experiment. Specifically, we show that it is possible to use images of cats to improve a generative model for dogs (!) and vice-versa. The cats here represent out-of-distribution data that can be used to improve the performance in the distribution of interest (in our toy example, dogs distribution). To perform this experiment, we train a classifier that discriminates between cats and dog images by looking at crops of various sizes (Section 3.2). Figure 6 shows the predictions of an 8 × 8 crops-classifier for an image of a cat, illustrating that there are a number of crops that are misclassified as crops from a dog image. We report results for this experiment in Table 2b and we observe improvements in FID arising from using out-of-distribution data. Beyond natural images, we show that it is even possible to use procedurally generated data from Shaders [6] to (slightly) improve the performance. Figure 21 shows an example of such an image and the corresponding predictions of a crops classifier. Table 2b contains more results and ablations between annotating all the out-of-distribution at a single noise level vs. sample-dependent annotations.

[Figure 33]

Figure 6: Patch level probabilities for dogness in a cat image.

Takeaway 1: It is possible to use low-quality in-distribution images and high-quality out-of-distribution images to produce high-quality in-distribution images.

Corruptions of natural datasets – ImageNet results. Up to this point, our corrupted data has been artificially constructed to study our method in a controlled setting. However, it turns out that even in real datasets such as ImageNet, there are images with significant degradations such as heavy blur, low lighting, and low contrast, and also images with fantastic detail, clear lightning, and sharp contrast. Here, the high-quality and the low-quality sets are not given and hence we have to estimate them. We opt to use the CLIP-IQA quality metric [64] to separate ImageNet into high-quality (top 10% CLIP-IQA) and low-quality (bottom 90% CLIP-IQA) sets. Figure 4 shows some of the top and bottom quality images according to our metric. Given the high-quality and low-quality sets, we are now back to the previous setting where we can use the developed Ambient-o methodology. We underline that there is a rich literature regarding quality-assessment methods [66, 67, 47, 65] and the performance of our method depends on how the high-quality and the low-quality sets are defined, since the rest of the samples are annotated based on which set they are closer to.

We use Ambient-o to refer to our method that uses low-quality data at high diffusion times (Section 5) and Ambient-o+crops to refer to the extended version of our method that uses crops from potentially low-quality images at low-diffusion times. Perhaps surprisingly, there are images in ImageNet that have lower global quality but high-quality crops that we can use for low-noise. We present results in Table 1, where we show the best FID [28] and FDDINOv2 obtained by different methods. We show the highest and lowest quality crops, alongside their associated full images, of ImageNet according to CLIP in Figure 5.

As shown in the Table, our method leads to state-of-the-art FID scores, improving over the previous state-of-the-art baseline EDM-2 [35] at both the low and high parameter count settings. The benefits are more pronounced when we measure test FID as our method memorizes significantly less due to the addition of noise during the annotation stage of our pipeline (Section 3.1). Beyond FID, we

provide qualitative results in Figure 1 (bottom) and Appendix Figures 12, 13. We further show that the quality of the generated images measured by CLIP increased compared to the baseline in Appendix Table 5. The observed improvements are proof that the ability to learn from data with heterogeneous qualities can be truly impactful for realistic settings beyond synthetic corruptions typically studied in prior work.

Takeaway 2: Real datasets contain heterogeneous samples. Ambient-o explicitly accounts for quality variability during training, leading to improved generation quality.

Text-to-image results. For our final set of experiments, we show how Ambient-o can be used to improve the performance of text-to-image diffusion models. We use the code-base of MicroDiffusion [52], as it is open-data and trainable with modest compute (≈ 2 days on 8-H100 GPUs). Sehwag et al. [52] use four main datasets to train their model: Conceptual Captions (12M) [54], Segment Anything (11M) [40], JourneyDB (4.2M) [61], and DiffusionDB (10.7M) [68]. Of these four, DiffusionDB is of significantly lower quality than the others as it contains solely synthetic data from an outdated diffusion model. This presents an opportunity for the use of our method. Can we use this lower-quality data and improve the performance of the trained network?

100

Baseline

Ambient-o

| |
|---|

80

80

73

72

Percentage(%)

65

65

65

64

61

57

60

55

50 50

44

42

38

40

35

35

35

34

28

26

20

20

0

ColorsConflictingCountingDALL-EDescriptionsGMetal.MisspellingPositionalRareWordsReddit Text

(a) DrawBench

100

Baseline Ambient-o

80

70

Percentage(%)

68

66

66

66

59

59

59

57

56

60

54

53

46

45

43

42

40

40

40

40

33

33

33

31

29

20

0

AbstractAnimalsArtifactsFood&BeverageArts IllustrationsIndoorScenesOutdoorScenesProduce&PlantsPeople WorldKnowledgeVehicles

(b) PartiPrompts

Figure 7: Assessing image quality with GPT-4o on DrawBench and PartiPrompts.

We set σmin = 2 for all samples from DiffusionDB and σmin = 0 for all other datasets and we train a diffusion model with Ambient-o. We note that we did not ablate this hyperparameter and it is quite likely that improved results would be obtained by tuning it or by training a high-quality vs low-quality data classifier for the annotation. Despite that, our trained model achieves a remarkable FID of 10.61 in COCO, significantly improving the baseline FID of 12.37 (Table 9). We present qualitative results in Figure 1 and GPT-4o evaluations on DrawBench and PartiPrompt in Figure 7. Ambient-o and baseline generations for different prompts can be found in Figure 1. As an additional ablation, we compared our method with the recipe of doing a final fine-tuning on the highest-quality subset, as done in the works of [52, 12]. Compared to this baseline, our method obtained slightly worse COCO FID (10.61 vs 10.27) but obtained much greater diversity, as seen visually in Figure 8 and quantitatively through > 13% increases in DINO Vendi Diversity on prompts from DiffDB (3.22 vs 3.65.). This corroborates our intuition that data filtration leads to decreased diversity. Ambient-o uses all the data but can strike a fine balance between high-quality and diverse generation.

Takeaway 3: Ambient-o treats synthetic data as corrupted data. This leads to superior visual quality and increased diversity compared to only relying on real samples.

[Figure 34]

[Figure 35]

(a) "the great battle of middle earth, unreal engine, trending on artstation, masterpiece"

[Figure 36]

[Figure 37]

(b) "an abominable snowman trapped in ice by greg rutkowski"

- Figure 8: Examples of mode collapse. Left: baseline model finetuned on a high-quality subset. Right: Ambient-o model using all the data. As shown, finetuning decreases output diversity.

(a) Measuring fidelity and prompt alignment of generated images on COCO dataset.

Method FID-30K (↓) Clip-FD-30K (↓) Clip-score (↑) Baseline 12.37 10.07 0.345

Ambient-o 10.61 9.40 0.348

(b) Measuring performance on the GenEval benchmark.

Objects Method Overall Single Two Counting Colors Position

Color attribution Baseline 0.44 0.97 0.33 0.35 0.82 0.06 0.14

Ambient-o 0.47 0.97 0.40 0.36 0.82 0.11 0.14

- Figure 9: Quantitative benefits of Ambient-o on COCO [43] zero-shot generation and GenEval [25].

## 6 Limitations and Future Work

Our work opens several avenues for improvement. On the theoretical side, we aim to establish matching lower bounds to demonstrate that learning from the mixture distribution becomes provably optimal beyond a certain noise threshold. Algorithmically, while our method performs well under high-frequency corruptions, it remains an open question whether more effective training strategies could be used for different types of corruptions (e.g., masking). Moreover, real-world datasets often exhibit patch-wise heterogeneity—for example, facial regions are frequently blurred for privacy, leading to uneven corruption across image crops. We plan to investigate patch-level noise annotations to better capture this structure in future work. Computationally, the full-version of our algorithm requires the training of classifiers for annotations that increases the runtime. This overhead can be avoided by using hand-picked annotation times based on quality proxies as done in our synthetic data experiment. Finally, we believe the true potential of Ambient-o lies in scientific applications, where data often arises from heterogeneous measurement processes.

## 7 Conclusion

Is it possible to get good generative models from bad data? Our framework extracts value from low-quality, synthetic, and out-of-distribution sources. At a time when the ever-growing data demands of GenAI are at odds with the need for quality control, Ambient-o lights a path for both to be achieved simultaneously.

## 8 Acknowledgements

This research has been supported by NSF Awards CCF-1901292, ONR grants N00014-25-1-2116, N00014-25-1-2296, a Simons Investigator Award, and the Simons Collaboration on the Theory of Algorithmic Fairness. The experiments were run on the Vista GPU Cluster through the Center for Generative AI (CGAI) and the Texas Advanced Computing Center (TACC) at UT Austin. Adrián Rodríguez-Muñoz is supported by the La Caixa Fellowship (LCF/BQ/EU22/11930084).

## References

- [1] Asad Aali, Marius Arvinte, Sidharth Kumar, and Jonathan I Tamir. Solving inverse problems with score-based generative priors learned from noisy data. arXiv preprint arXiv:2305.01166, 2023.
- [2] Asad Aali, Giannis Daras, Brett Levac, Sidharth Kumar, Alex Dimakis, and Jon Tamir. Ambient diffusion posterior sampling: Solving inverse problems with diffusion models trained on corrupted data. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=qeXcMutEZY.
- [3] Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G Baraniuk. Self-consuming generative models go mad. arXiv preprint arXiv:2307.01850, 4:14, 2023.
- [4] Sina Alemohammad, Ahmed Imtiaz Humayun, Shruti Agarwal, John Collomosse, and Richard Baraniuk. Self-improving diffusion models with synthetic data. arXiv preprint arXiv:2408.16333, 2024.
- [5] Weimin Bai, Yifei Wang, Wenzheng Chen, and He Sun. An expectation-maximization algorithm for training clean diffusion models from corrupted observations. arXiv preprint arXiv:2407.01014, 2024.
- [6] Manel Baradad, Chun-Fu Chen, Jonas Wulff, Tongzhou Wang, Rogerio Feris, Antonio Torralba, and Phillip Isola. Procedural image programs for representation learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=wJwHTgIoE0P.
- [7] Ashish Bora, Eric Price, and Alexandros G Dimakis. Ambientgan: Generative models from lossy measurements. In International conference on learning representations, 2018.
- [8] Zdravko I Botev, Joseph F Grotowski, and Dirk P Kroese. Kernel density estimation via diffusion. 2010.
- [9] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215, 2022.
- [10] Sitan Chen, Giannis Daras, and Alex Dimakis. Restoration-degradation beyond linear diffusions: A non-asymptotic analysis for DDIM-type samplers. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 4462–4484. PMLR, 7 2023. URL https://proceedings. mlr.press/v202/chen23e.html.
- [11] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8188–8197, 2020.
- [12] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, Matthew Yu, Abhishek Kadian, Filip Radenovic, Dhruv Mahajan, Kunpeng Li, Yue Zhao, Vladan Petrovic, Mitesh Kumar Singh, Simran Motwani, Yi Wen, Yiwen Song, Roshan Sumbaly, Vignesh Ramanathan, Zijian He, Peter Vajda, and Devi Parikh. Emu: Enhancing image generation models using photogenic needles in a haystack, 2023.
- [13] Giannis Daras, Yuval Dagan, Alexandros G Dimakis, and Constantinos Daskalakis. Consistent diffusion models: Mitigating sampling drift by learning to be consistent. arXiv preprint arXiv:2302.09057, 2023.
- [14] Giannis Daras, Mauricio Delbracio, Hossein Talebi, Alex Dimakis, and Peyman Milanfar. Soft diffusion: Score matching with general corruptions. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=W98rebBxlQ.
- [15] Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, and Adam Klivans. Ambient diffusion: Learning clean distributions from corrupted data. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/ forum?id=wBJBLy9kBY.

- [16] Giannis Daras, Alexandros G Dimakis, and Constantinos Daskalakis. Consistent diffusion meets tweedie: Training exact ambient diffusion models with noisy data. arXiv preprint arXiv:2404.10177, 2024.
- [17] Giannis Daras, Yeshwanth Cherapanamjeri, and Constantinos Costis Daskalakis. How much is a noisy image worth? data scaling laws for ambient diffusion. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=qZwtPEw2qN.
- [18] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. arXiv preprint arXiv:2303.11435, 2023.
- [19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.
- [20] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [21] Sander Dieleman. Diffusion is spectral autoregression, 2024. URL https://sander.ai/ 2024/09/02/spectral-autoregression.html.
- [22] Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, and Aleksander Madry. Optimizing ml training with metagradient descent. arXiv preprint arXiv:2503.13751, 2025.
- [23] Damien Ferbach, Quentin Bertrand, Avishek Joey Bose, and Gauthier Gidel. Self-consuming generative models with curated data provably optimize human preferences. arXiv preprint arXiv:2407.09499, 2024.
- [24] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.
- [25] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment, 2023. URL https://arxiv.org/abs/2310.11513.
- [26] Sachin Goyal, Pratyush Maini, Zachary C Lipton, Aditi Raghunathan, and J Zico Kolter. Scaling laws for data filtering–data curation cannot be compute agnostic. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22702–22711, 2024.
- [27] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.
- [28] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [29] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.
- [30] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive Mixtures of Local Experts. Neural Computation, 3(1):79–87, March 1991. ISSN 0899-7667. doi: 10.1162/neco.1991.3.1.79. URL https://doi.org/10.1162/neco.1991.3.1.79. _eprint: https://direct.mit.edu/neco/article-pdf/3/1/79/812104/neco.1991.3.1.79.pdf.
- [31] Yiding Jiang, Allan Zhou, Zhili Feng, Sadhika Malladi, and J Zico Kolter. Adaptive data optimization: Dynamic sample selection with scaling laws. arXiv preprint arXiv:2410.11820, 2024.
- [32] Mason Kamb and Surya Ganguli. An analytic theory of creativity in convolutional diffusion models. arXiv preprint arXiv:2412.20292, 2024.
- [33] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4401–4410, 2019.
- [34] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. arXiv preprint arXiv:2206.00364, 2022.

- [35] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proc. CVPR, 2024.
- [36] Sergey Kastryulin, Dzhamil Zakirov, and Denis Prokopenko. PyTorch Image Quality: Metrics and measure for image quality assessment, 2019. URL https://github.com/photosynthesis-team/piq. Open-source software available at https://github.com/photosynthesis-team/piq.
- [37] Sergey Kastryulin, Jamil Zakirov, Denis Prokopenko, and Dmitry V. Dylov. Pytorch image quality: Metrics for image quality assessment, 2022. URL https://arxiv.org/abs/2208. 14818.
- [38] Varun A Kelkar, Rucha Deshpande, Arindam Banerjee, and Mark A Anastasio. Ambientflow: Invertible generative models from incomplete, noisy measurements. arXiv preprint arXiv:2309.04856, 2023.
- [39] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [40] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything, 2023. URL https://arxiv.org/abs/2304.02643.
- [41] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.
- [42] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024.
- [43] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015. URL https://arxiv.org/abs/1405.0312.
- [44] Yvette Y Lin, Angela F Gao, and Katherine L Bouman. Imaging an evolving black hole by leveraging shared structure. ICASSP, 2024.
- [45] Zeyuan Liu, Zhihe Yang, Jiawei Xu, Rui Yang, Jiafei Lyu, Baoxiang Wang, Yunjian Xu, and Xiu Li. Adg: Ambient diffusion-guided dataset recovery for corruption-robust offline reinforcement learning. arXiv preprint arXiv:2505.23871, 2025.
- [46] Haoye Lu, Qifan Wu, and Yaoliang Yu. SFBD: A method for training diffusion models with noisy data. In Frontiers in Probabilistic Inference: Learning meets Sampling, 2025. URL https://openreview.net/forum?id=6HN14zuHRb.
- [47] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.
- [48] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL https://arxiv.org/abs/2212.09748.
- [49] Guilherme Penedo, Hynek Kydlíˇcek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024.
- [50] François Rozet, Gérôme Andry, François Lanusse, and Gilles Louppe. Learning diffusion priors from observations by expectation maximization. arXiv preprint arXiv:2405.13712, 2024.
- [51] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

- [52] Vikash Sehwag, Xianghao Kong, Jingtao Li, Michael Spranger, and Lingjuan Lyu. Stretching each dollar: Diffusion training from scratch on a micro-budget. arXiv preprint arXiv:2407.15811, 2024.
- [53] Kulin Shah, Alkis Kalavasis, Adam R. Klivans, and Giannis Daras. Does generation require memorization? creative diffusion models using ambient diffusion, 2025.
- [54] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Iryna Gurevych and Yusuke Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1238. URL https://aclanthology.org/P18-1238/.
- [55] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017. URL https://arxiv.org/abs/1701.06538.
- [56] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. arXiv preprint arXiv:2212.03860, 2022.
- [57] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. arXiv preprint arXiv:2305.20086, 2023.
- [58] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [59] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.
- [60] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [61] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, and Hongsheng Li. Journeydb: A benchmark for generative image understanding, 2023. URL https://arxiv. org/abs/2307.00716.
- [62] Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Josh Tenenbaum, Frédo Durand, Bill Freeman, and Vincent Sitzmann. Diffusion with forward models: Solving stochastic inverse problems without direct supervision. Advances in Neural Information Processing Systems, 36:12349–12362, 2023.
- [63] Antonio Torralba, Phillip Isola, and William T Freeman. Foundations of computer vision. MIT Press, 2024.
- [64] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023.
- [65] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 2555–2563, 2023.
- [66] Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment. In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003, volume 2, pages 1398–1402. Ieee, 2003.
- [67] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612, 2004.
- [68] Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. arXiv preprint arXiv:2210.14896, 2022.
- [69] Yasi Zhang, Tianyu Chen, Zhendong Wang, Ying Nian Wu, Mingyuan Zhou, and Oscar Leong. Restoration score distillation: From corrupted diffusion pretraining to one-step high-quality generation. arXiv preprint arXiv:2505.13377, 2025.

## A Theoretical Results

- A.1 Kernel Estimation Assumption A.1. The density p is λ lipschitz.

Let {X(i)}ni=1 a set of n independent samples from a density p that satisfies Assumption A.1. Let pˆ be the empirical density on those samples.

We are interested in bounding the total variation distance between pσ := p ⊛ N(0,σ2) and pˆσ = pˆ⊛ N(0,σ2). In particular,

n

X(i) − x σ

1 nσ

, (A.1)

pˆσ(x) =

ϕ

i=1

2/2 is the Gaussian kernel. We want to argue that the TV distance between

where ϕ(u) = √12πe−u

pσ and pˆσ is small given sufficiently many samples n. For simplicity, let’s fix the support of p to be [0,1]. We have:

- 1

- 2

dTV(pσ,pˆσ) =

L−1

1

|pσ(x) − pˆσ(x)|dx =

0

l=0

(l+1)/L

|pσ(x) − pˆσ(x)|dx (A.2)

l/L

Now let us look at one of the terms of the summation.

(l+1)/L

(l+1)/L

|pσ(x) − pσ(l/L) + pσ(l/L) − pˆσ(x)|dx (A.3)

|pσ(x) − pˆσ(x)|dx =

l/L

l/L

(l+1)/L

(l+1)/L

|pσ(l/L) − pˆσ(x)|dx. (A.4) We first work on the first term. Using Lemma A.6:

≤

|pσ(x) − pσ(l/L)|dx +

l/L

l/L

(l+1)/L

(l+1)/L

|x − l/L|dx (A.5)

|pσ(x) − pσ(l/L)|dx ≤ λ

l/L

l/L

λ 2L2

. (A.6)

=

Next, we work on the second term.

(l+1)/L

(l+1)/L

|pσ(l/L) − pˆσ(l/L) + pˆσ(l/L) − pˆσ(x)|dx (A.7)

|pσ(l/L) − pˆσ(x)|dx =

l/L

l/L

(l+1)/L

(l+1)/L

|pˆσ(l/L) − pˆσ(x)|dx. (A.8)

≤

|pσ(l/L) − pˆσ(l/L)|dx +

l/L

l/L

According to Lemma A.5, we have that pˆσ is λˆ = σ 1

2√2πe Lipschitz. Then, the second term becomes: (l+1)/L

λˆ 2L2

(l+1)/L

|pˆσ(l/L) − pˆσ(x)|dx ≤ λˆ

. (A.9)

|l/L − x|dx =

l/L

l/L

It remains to bound the following term

(l+1)/L

|pσ(l/L) − pˆσ(l/L)|dx = |pσ(l/L) − pˆσ(l/L)| L

l/L

We will be applying Hoeffding’s Inequality, stated below:

(A.10)

Theorem A.2 (Hoeffding’s Inequality). Let Y1,...,Yn be independent random variables in [a,b] with mean µ. Then,

Pr

n

1 n

Yi − µ ≥ t ≤ 2exp −2nt2/(b − a)2 . (A.11)

i=1

Recall that pˆσ can be written as

1 n

pˆσ(x) =

n

ϕ((X(i) − x)/σ) σ

1 n

=

i=1

n

Yi, (A.12)

i=1

(i)−x)/σ)

in terms of the random variables Yi := ϕ((X

σ . These random variables are supported in 0, √21πσ

. So, for any x, we have that:

2

#### Pr(|pˆσ(x) − E[ˆpσ(x)]| ≥ t) ≤ 2exp −4πσ2nt2 . (A.13)

Taking t = log(24πσL/δ2n ) and using the above inequality and the union bound, we have that, with probability at least 1 − δ, for all l ∈ {0,1,...,L − 1}:

log(2L/δ) 4πσ2n

. (A.14)

|pˆσ(l/L) − E[ˆpσ(l/L)]| ≤

Let us now compute the expected value of pˆσ(x).

1 σ

=

E[ˆpσ(x)] = E

n

1 nσ

=

i=1

x − u σ

p(u)ϕ

n

X(i) − x σ

1 nσ

ϕ

i=1

X(i) − x σ

E ϕ

(A.15)

(A.16)

du ≡ (p ⊛ N(0,σ2))(x) = pσ(x). (A.17)

Combining equation A.14 and equation A.17, we get:

|pˆσ(l/L) − pσ(x)| ≤

log(2L/δ) 4πσ2n

. (A.18)

Putting everything together we have:

λ 2L

1 2Lσ2√2πe

log(2L/δ) 4πσ2n

dTV(pσ,pˆσ) ≤

+

. Choosing L = n · max{λ,1} we get that:

+

log n + log(1 ∨ λ) + log 2/δ σ2n

1 n

1 σ2n

dTV(pσ,pˆσ) ≲

+

+

.

- A.2 Evolution of parameters under noise Proof of theorem 4.2: We will use the following facts:

- Fact 1 (Direct corollary of the optimal coupling theorem). There exists a coupling γ of P and Q, which samples a pair of random variables (X,Y ) ∼ γ such that Prγ[X ̸= Y ] = dTV(P,Q).
- Fact 2. For any x,y ∈ Rd: dTV(N(x,σ2I),N(y,σ2I)) ≤ ∥x − y∥/2σ

Proof. The KL divergence between N(µ1,Σ1) and N(µ2,Σ2) is

tr(Σ−2 1Σ1) + (µ2 − µ1)Σ−2 1(µ2 − µ1) − d + log |Σ2| |Σ1|

- 1

- 2

KL(N(µ1,Σ1),N(µ2,Σ2)) =

Applying this general result to our case:

.

- 1

- 2

KL(N(x,σ2I),N(y,σ2I)) =

We conclude by applying Pinsker’s inequality.

∥x − y∥2 σ2

.

| |
|---|

- A corollary of Fact 2 and the optimal coupling theorem is the following:

- Fact 3. Fix arbitrary x,y ∈ Rd. There exists a coupling γx,y of N(0,σ2I) and N(0,σ2I), which

samples a pair of random variables (Z,Z′) ∼ γx,y such that Prγ

#### [x+Z ̸= y +Z′] = ∥x−y∥/2σ.

x,y

Now let us denote by P˜ = P ⊛ N(0,σ2I) and Q˜ = Q ⊛ N(0,σ2I). To establish our claim in the theorem statement, it suffices to exhibit a coupling γ˜ of P˜ and Q˜ which samples a pair of random variables (X,˜ Y˜) ∼ γ˜ such that: Prγ˜[X˜ ̸= Y˜] ≤ dTV(P,Q) · 2Dσ. We define coupling γ˜ as follows: 1) sample (X,Y ) ∼ γ (as specified in Fact 1), 2) sample (Z,Z′) ∼ γX,Y (as specified in Fact 3) and

- 3) output (X,˜ Y˜) := (X + Z,Y + Z′). Let us argue the following:

- Lemma A.3. The afore-described sampling procedure γ˜ is a valid coupling of P˜ and Q˜.

Proof. We need to establish that the marginals of γ˜ are P˜ and Q˜. We will only show that for (X,˜ Y˜) ∼ γ˜ according to the afore-described sampling procedure, the marginal distribution of X˜ is P˜, as the proof for Y˜ is identical. Since γ is a coupling of P and Q, for (X,Y ) ∼ γ, the marginal distribution of X is P. By Fact 3, conditioning on any value of X and Y , the marginal distribution of Z is N(0,σ2I). Thus, X˜ = X + Z, where X ∼ P and independently Z ∼ N(0,σ2I), and thus the distribution of X˜ is P˜.

| |
|---|

- Lemma A.4. Under the afore-described coupling γ˜: Prγ˜[X˜ ̸= Y˜] ≤ dTV(P,Q) · 2Dσ.

Proof. Notice that, when X = Y , by Fact 3, Z = Z′ with probability 1, and therefore X˜ = Y˜. So for event X˜ ̸= Y˜ to happen, it must be that X ̸= Y happens and, conditioning on this event, that X + Z ̸= Y + Z′ happens. By Fact 1, Prγ[X ̸= Y ] = dTV(P,Q). By Fact 3, for any realization of (X,Y ), Prγ

X,Y

[X + Z ̸= Y + Z′] = ∥X2−σY ∥ ≤ 2Dσ, where we used that P and Q are supported on a set with diameter D. Putting the above together, the claim follows.

| |
|---|

- A.3 Auxiliary Lemmas

- Lemma A.5 (Lipschitzness of the empirical density). For a collection of points X(1),...,X(n)

(i)−x

2/2 is the Gaussian kernel. Then pσ is σ 1

consider the function pˆσ(x) = nσ1 ni=1 ϕ X

σ , where ϕ(u) = √12πe−u

2√2πe -Lipschitz.

Proof. Let us compute the derivative of pˆσ:

n

x − X(i) σ

1 nσ

d dx

pˆ′σ(x) =

(A.19)

ϕ

i=1

n

X(i) − x σ2

1 √2πnσ

exp −(X(i) − x)2/(2σ2)

(A.20)

=

i=1

1 √2πσ2

exp(−u2/2)u (A.21)

≤

max

u

1 σ2√2πe

. (A.22)

≤

| |
|---|

- Lemma A.6 (Lipschitzness of a density convolved with a Gaussian). Let p be a density that is λ-Lipschitz. Let pσ = p ⊛ N(0,σ2I). Then, pσ is also λ-Lipschitz. Proof. Let us denote with ϕσ(·) the Gaussian density with variance σ2. We have that:

pσ(x) − pσ(y) = (p(x − τ) − p(y − τ))ϕσ(τ)dτ ⇒ (A.23)

|pσ(x) − pσ(y)|≤ |p(x − τ) − p(y − τ)|ϕσ(τ)dτ (A.24)

≤ λ|x − y|· ϕσ(τ)dτ (A.25) = λ|x − y|. (A.26)

| |
|---|

## B Additional Results

### B.1 CIFAR-10 controlled corruptions

Figures 10a,11,10b show gaussian blur, motion blur, and JPEG corrupted CIFAR-10 images respectively at different levels of severity. Appendix Table 3 shows results for JPEG compressed data at different levels of compression. We also tested our method for motion blurred data with high severity, visualized in the last row of Appendix Figure 11), obtaining a best FID of 5.85 (compared to 8.79 of training on only the clean data).

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

(a) CIFAR-10 images corrupted with blur at increasing levels (σB = 0.4, 0.6, 1.0).

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(b) CIFAR-10 images corrupted with JPEG at compression rates: 25%, 18%, 15% respectively.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Figure 11: CIFAR-10 images corrupted with motion blur at increasing levels of corruption.

Table 3: Results for learning from JPEG compressed data on CIFAR-10.

Method Dataset Clean (%) Corrupted (%) JPEG Compression (Q) σ¯min

tn FID Only Clean Cifar-10 10 0 – – 8.79

15% 1.60 6.67 18% 1.40 6.43 25% 1.27 6.34 50% 1.03 5.94 75% 0.81 5.57 90% 0.63 4.72

Ambient Omni Cifar-10 10 90

### B.2 FFHQ-64x64 controlled corruptions

In Appendix 4 we show additional results for learning from blurred data on the FFHQ dataset. Similarly to the main paper, we observe that our Ambient-o algorithm leads to improvements over just using the high-quality data that are inversely proportional to the corruption level.

Table 4: Results for learning from blurred data, FFHQ.

Method Dataset Clean (%) Corrupted (%) Parameters Values (σB) σ¯min

tn FID Only Clean FFHQ 10 0 - - 5.12

10 90 0.8 2.89 4.95 10 90 0.6 2.12 4.65 10 90 0.4 0.63 3.32

Ambient Omni FFHQ

### B.3 ImageNet results

In the main paper, we used FID as a way to measure the quality of generated images. However, FID is computed with respect to the test dataset that might also have samples of poor quality. Further, during FID computation, quality and diversity are entangled. To disentangle the two, we generate images using the EDM-2 baseline and our Ambient-o model and we use CLIP to evaluate the quality of the generated image (through the CLIP-IQA metric implemented in the PIQ package [37, 36]). We present results and win-rates in Table 5. As shown, Ambient-o achieves a better per-image quality compared to the baseline despite using exactly the same model, hyperparameters, and optimization algorithm. The difference comes solely from better use of the available data.

- Table 5: Additional comparison between EDM-2 XXL and our Ambient-o model using the CLIP IQA metric for image quality assesment. Ambient-o leads to improved scores despite using the exact same architecture, data and hyperparameters. For this experiment, we use the models with guidance optimized for DINO FD since they are the ones producing the higher quality images.

Metric EDM-2 [35] XXL Ambient-o XXL crops

Average CLIP IQA score 0.69 0.71 Median CLIP IQA score 0.79 0.80

Win-rate 47.98% 52.02%

C Ambient diffusion implementation details and loss ablations

Similar to the EDM-2 [35] paper, we use a pre-condition weight to balance the importance of different diffusion times. Specifically, we modulate the EDM2 weight λ(σ) by a factor:

λamb(σ,σmin) = σ4/(σ2 − σmin2 )2 (C.1)

for our ambient loss based on a similar analysis to [35]. We further use a buffer zone around the annotation time of each sample to ensure that the loss doesn’t have singularities due to divisions by 0. We ablate the precondition term and the buffer size in Appendix Table 6.

- Table 6: Ablation study of ambient weight and stability buffer on Cifar-10 with 10% clean data and 90% corrupted data with blur of 0.6.

Method FID ↓ No ambient preconditioning weight and no buffer: λamb(σ,σmin) = 1 & σ > σmin 5.49 Adding ambient preconditioning weight:

+ Weight λamb(σ,σmin) = σ4/(σ2 − σmin2 )2 5.36 Adding stability buffer/clipping:

+ Clip λamb(σ,σmin) at 2.0 5.35 + Clip λamb(σ,σmin) at 4.0 5.69 + Buffer λamb(σ,σmin) at 2.0 i.e. σ > √2σmin 5.40 + Buffer λamb(σ,σmin) at 4.0 i.e. σ > (2/√3)σmin 5.34

For our ablations, we focus on the setting of training with 10% clean data and 90% corrupted data with Gaussian blur of σB = 0.6. Using no ambient pre-conditioning and no buffer, we obtain an FID of 5.56. In the same setting, adding the ambient pre-conditioning weight λamb(σ,σmin) improves FID by 0.13 points. Next, we ablate two strategies to mitigate the impact of the singularity of λamb(σ,σmin) at σ = σmin. The first strategy clips the ambient pre-conditioning weight at a specified maximum value λMAXamb , but still trains for σ arbitrarily close to σmin. The second strategy also specifies a maximum value, but imposes a buffer

1 λMAXamb − 1

σmin (C.2)

σ > 1 +

that restricts training to noise levels σ such that λamb(σ,σmin) ≤ λMAXamb . Clipping the ambient weight to λMAXamb = 2.0 minimally improves FID to 5.35, but clipping to 4.0 significantly worsens it to

- 5.69. Adding a buffer at λMAXamb = 2.0 slightly worsens FID to 5.40, but slackening the buffer to 4.0 minimally improves FID to 5.34. We opt for the buffering strategy in favor of the clipping strategy since performance appears convex in the buffer parameter, and because it obtains the best FID.

## D Classifier annotation ablations

Balanced vs unbalanced data: We ablate the impact of classifier training data on the setting of CIFAR-10 with 10% clean data and 90% corrupted data with gaussian blur with σB = 0.6. When

annotating with a classifier trained on the same unbalanced dataset we train the diffusion model on we obtained a best FID of 6.04, compared to the 5.34 obtained if we train on a balanced dataset.

Training iterations: We ablate the impact of classifier training iterations on the setting of CIFAR-10 with 10% clean data and 90% corrupted data with JPEG compression at compression rate of 18%, training the classifier with a balanced dataset. We report minute variations in the best FID, obtaining

- 6.50, 6.58, and 6.49 when training the classifier for 5e6, 10e6, and 15e6 images worth of training respectively.

- Table 7: Comparison with baselines for training with data corrupted by Gaussian Blur at different levels. The dataset used in this experiment is CIFAR-10.

Method Clean (%) Corrupted (%) Parameters Values (σB) σ¯min

tn FID Only Clean 10 0 - - 8.79

1.0

45.32 0.8 28.26

No annotations 10 90

0

- 0.4 2.47

Single annotation 10 90

- 1.0 2.32 6.95 0.8 1.89 6.66 0.4 0.00 2.47

10 90 1.0 2.84 6.16 10 90 0.8 1.93 6.00 10 90 0.4 0.22 2.44

Classifier annotations

## E Training Details

### E.1 Formation of the high-quality and low-quality sets.

In the theoretical problem setting we assumed the existence of a good set SG from the clean distribution and a bad set SB from the corrupted distribution. In practice, we do not actually possess these sets initially, but we can construct them so long as we have access to a measure of "quality". Given a function on images which tells us wether its good enough to generate or not e.g. CLIP-IQA quality [64] greater than some threshold, we can define our good set SG as the good enough images and SB as the complement. From this point on we can apply the methodology of ambient-o as developed, either employing classifier annotations as in our pixel diffusion experiments, or fixed annotations as in our large scale ImageNet and text-to-image experiments.

### E.2 Datasets

CIFAR-10. CIFAR-10 [41] consists of 60,000 32x32 images of ten classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck).

FFHQ. FFHQ [33] consists of 70,000 512x512 images of faces from Flickr. We used the dataset at 64x64 resolution for our experiments.

AFHQ. AFHQ [11] consists of 5,653 images of cats, 5,239 images of dogs and 5,000 images of wildlife, for a total of 15,892 images.

ImageNet. ImageNet [19] consists of 1,281,167 images of variable resolution from 1000 classes. Conceptual Captions. Conceptual Captions [54] consists of 12M (image url, caption) pairs.

Segment Anything. Segment Anything [40] consists of 11.1M high-resolution images annotated with segmentation masks. Since the original dataset did not have real captions, we use the same LLaVA generated captions created by the MicroDiffusion [52] paper.

JourneyDB. JourneyDB consists of 4.4M synthetic image-caption pairs from Midjourney [61].

DiffusionDB. DiffusionDB consists of 14M synthetic image-caption pairs, mostly generated from Stable Diffusion models [68]. We use the same 10.7M quality-filtered subset created by the MicroDiffusion paper [52].

### E.3 Diffusion model training

CIFAR-10. We use the EDM [34] codebase as a reference to train class-conditional diffusion models on CIFAR-10. The architecture is a Diffusion U-Net [58] with ~55M paramemeters. We use the Adam optimizer [39] with learning rate 0.001, batch size 512, and no weight decay. While the original EDM paper trained for 200 × 106 images worth of training, when training with corrupted data we saw best results around 20×106 images. On a single 8xV100 node we achieved a throughput of 0.8s per 1k images, for an average of 4.4h per training run.

FFHQ. Same as for CIFAR-10, except learning was set to 2e − 4, we trained for a maximum of 100 × 106 images worth of training, and saw best results around 30 × 106 images worth.

AFHQ. Same as FFHQ.

ImageNet. We use the EDM2 [35] codebase as a reference to train class-conditional diffusion models on ImageNet. The architecture is a Diffusion U-Net [58] with ~125M paramemeters. We use the Adam optimizer [39] with reference learning rate 0.012, batch size 2048, and no weight decay. Same as the original codebase, we trained for ~2B worth of images. On 32 H200 GPUs, XS models took ~3 days to train, while XXL models took ~7 days.

MicroDiffusion. We use the MicroDiffusion codebase [52] as a reference to train text-to-image models on an academic budget. We follow their recipe exactly, changing only the standard denoising diffusion loss to the ambient diffusion loss. The architecture is a Diffusion Transformer [48] utilizing Mixture-of-Experiments (MoE) feedforward layers [55, 30], with ~1.1B paramemeters. We use the AdamW optimizer [39] with reference learning rates 2.4e − 4/8e − 5/8e − 5/8e − 5 for each of the four phases and batch size 2048 for all phases. On 8 H200 GPUs, training takes ~2 days to train.

### E.4 Classifier training

Classifier training is done using the same optimization recipe (optimizer, learning rate, batch size, etc.) as diffusion model training, except we change the architecture to an encoder-only "Half-Unet", simply by removing the decoder half of the original UNet architecture. The training of the classifier is substantially shorter compared to the diffusion training since classification is task is easier than generation.

## F Additional Figures

[Figure 65]

##### Figure 12: Uncurated generations from our Ambient-o XXL model trained on ImageNet.

[Figure 66]

##### Figure 13: Uncurated generations from our Ambient-o+crops XXL model trained on ImageNet.

Data Availability vs. Noise Level

Amountofdataavailablefortraining

6000

100% - Amount of data available without corruptions

4500

3000

1500

10% - Amount of data available by filtering

0 2 4 6 8 10 12

Noise stddev

- Figure 14: Amount of samples available at each noise level when training a generative model for dogs in the following setting: (1) we have 10% of the dogs dataset uncorrupted, (2) we have the other

90% of the dogs dataset corrupted with gaussian blur with σB = 0.6, and (3) we have 100% of the clean dataset of cats. At low noise levels, we can train on both the high quality dogs and a lot of the cats, resulting in > 100% of samples available relative to the original dogs dataset size. As the noise level starts to increase, we stop being able to use to the out-of-distribution cat samples, but start gaining some blurry dog samples. As the noise level approaches the maximum all the blurry dogs become available for training, such that the amount of data available approaches 100%.

10 15 20 25 30 Context Size

0.00

0.01

0.02

0.03

0.04

0.05

MSELosson2x2CenterPatch

Loss vs Context Size

=0.05 =0.15 =0.25 =0.35 =0.45 =0.55 =0.65 =0.75 =0.85 =0.95

- Figure 15: ImageNet-512x512: denoising loss of an optimally trained model, measured at 2 × 2 center patch, as we increase the context size given to the model (horizontal axis) and the noise level (different curves). As expected, for higher noise, more context is needed for optimal denoising. The large dot on each curve marks the point where the loss nearly plateaus.

30

EffectiveReceptiveFieldSize

25

20

15

10

0.2 0.4 0.6 0.8 1.0

Noise Level ( )

- Figure 16: ImageNet-512x512: context size needed to be within ϵ = 1e − 3 of the optimal loss for different noise levels. As expected, for higher noise, more context is needed for optimal denoising.

10 20 30 40 50 60 Context Size

0.00

0.01

0.02

0.03

0.04

0.05

MSELossona2x2patch

Loss vs Context Size

=0.05 =0.15 =0.25 =0.35 =0.45 =0.55 =0.65 =0.75 =0.85 =0.95

- Figure 17: FFHQ: denoising loss of an optimally trained model, measured at 2 × 2 center patch, as we increase the context size given to the model (horizontal axis) and the noise level (different curves). As expected, for higher noise, more context is needed for optimal denoising. The large dot on each curve marks the point where the loss nearly plateaus.

60

50

EffectiveReceptiveFieldSize

40

30

20

10

0.2 0.4 0.6 0.8 1.0

Noise Level ( )

- Figure 18: FFHQ: context size needed to be within ϵ = 1e − 3 of the optimal loss for different noise levels. As expected, for higher noise, more context is needed for optimal denoising.

[Figure 67]

(a) Cat image and classification probabilities over patches.

[Figure 68]

(b) Cat image and classification probabilities over patches.

- Figure 19: Two examples of cats from the AFHQ dataset. We partition each cat into non overlapping patches and we compute the probabilities of the patch belonging to an image of a dog using a cats vs dogs classifier trained on patches. The cat on the right has a lot more patches that could belong to a dog image according to the classifier, possibly due to the color or the texture of the fur.

[Figure 69]

(a) Cat annotated by a cats vs. dogs classifier that operates with crops of size 8.

[Figure 70]

(b) Cat annotated by a cats vs. dogs classifier that operates with crops of size 16.

[Figure 71]

(c) Cat annotated by a cats vs. dogs classifier that operates with crops of size 24.

- Figure 20: Patch-based annotations of a cat image from AFHQ using cats vs. dogs classifiers trained on different patch sizes.

[Figure 72]

##### Figure 21: Patch level probabilities for dogness in a synthetic image (procedural program). The cat has more useful patches than this non-realistic procedural program.

[Figure 73]

(a) Synthetic image and classification probabilities over patches.

[Figure 74]

(b) Synthetic image and classification probabilities over patches.

##### Figure 22: Two examples of procedurally generated images. We partition each image into non overlapping patches and we compute the probabilities of the patch belonging to an image of a dog using a synthetic image vs dogs classifier trained on patches. The image on the right has a lot more patches that could belong to a dog image according to the classifier, possibly due to the color or the texture.

[Figure 75]

(a) Cat image and classification probabilities over patches.

[Figure 76]

(b) Cat image and classification probabilities over patches.

##### Figure 23: Two examples of cat images. We partition each image into nonoverlapping patches and we compute the probabilities of the patch belonging to an image of wildlife using a cats vs wildlife classifier trained on patches. The image on the right has a lot more patches that could belong to a wildlife image according to the classifier, possibly due to the color or the texture.

[Figure 77]

[Figure 78]

(a) Example batch. (b) Noisy batch.

Figure 24: Example batch.

[Figure 79]

(a) Highest quality images from CC12M according to CLIP.

[Figure 80]

(b) Lowest quality images from CC12M according to CLIP.

Figure 25: CLIP annotations for quality of images from CC12M.

[Figure 81]

(a) Highest quality images from SA1B according to CLIP.

[Figure 82]

(b) Lowest quality images from SA1B according to CLIP.

Figure 26: CLIP annotations for quality of images from SA1B.

[Figure 83]

(a) Highest quality images from DiffDB according to CLIP.

[Figure 84]

(b) Lowest quality images from DiffDB according to CLIP.

Figure 27: CLIP annotations for quality of images from DiffDB.

[Figure 85]

(a) Highest quality images from JDB according to CLIP.

[Figure 86]

(b) Lowest quality images from JDB according to CLIP.

Figure 28: CLIP annotations for quality of images from JDB.

###### Image Quality Distribution

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

0.8

CLIP-IQAQualityScore

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Fraction of Dataset (sorted by quality)

Figure 29: Distribution of image qualities according to CLIP for ImageNet-512.

