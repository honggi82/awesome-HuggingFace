## FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models

Vladimir Kulikov Matan Kleiner Inbar Huberman-Spiegelglas Tomer Michaeli Technion – Israel Institute of Technology

# arXiv:2412.08629v2[cs.CV]22Jul2025

Real image Edited image Real image Edited image Real image Edited image

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

this home Cat Raccoon FLOW

LOVE

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

BREAD BACON Mountain Volcano Lizard Dragon

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

White dog w/ cat

Dalmatian dog w/o cat

Man jumping Steak

…in Pixar style Bread

Figure 1. FlowEdit. We present an inversion-free, optimization-free and model agnostic method for text-based image editing using pretrained flow models. As opposed to the editing-by-inversion paradigm, FlowEdit constructs an ODE that directly maps the source image distribution to the target image distribution (corresponding to the source and target prompts). This ODE achieves a lower transport cost and thus leads to better structure preservation, achieving state of the art results on complex editing tasks. From left to right, top to bottom, the first five images were obtained with FLUX and the rest with Stable Diffusion 3. Text indicates changes in the prompts.

##### Abstract

Editing real images using a pre-trained text-to-image (T2I) diffusion/flow model often involves inverting the image into its corresponding noise map. However, inversion by itself is typically insufficient for obtaining satisfactory results, and therefore many methods additionally intervene in the sampling process. Such methods achieve improved results but are not seamlessly transferable between model architectures. Here, we introduce FlowEdit, a text-based editing

method for pre-trained T2I flow models, which is inversionfree, optimization-free and model agnostic. Our method constructs an ODE that directly maps between the source and target distributions (corresponding to the source and target text prompts) and achieves a lower transport cost than the inversion approach. This leads to state-of-the-art results, as we illustrate with Stable Diffusion 3 and FLUX. Code and examples are available on the project’s webpage.

##### 1. Introduction

Diffusion and flow models generate samples through an iterative process, which starts from pure noise and employs denoising-like steps. Beyond data generation, many methods employ pre-trained diffusion/flow models for editing real signals [5, 8, 15, 17–19, 22, 23, 26, 32, 33, 40, 49, 53]. While some techniques rely on test-time optimization [18,

- 23], many methods aim for optimization-free approaches. A common first step in the latter category of methods is inversion, i.e. extracting the initial noise vector that presumably generates the signal one wishes to edit. This noise vector is then used to generate the edited signal by injecting a different condition to the model, such as a modified text prompt [22, 35, 50, 52].

In the context of image editing, many works noted that this editing-by-inversion paradigm often leads to insufficient fidelity to the source image (see e.g. [22]). To overcome this, some methods attempt to reduce the errors incurred in the inversion process [14, 35]. However, this rarely eliminates the problem as even when editing generated images, for which the initial noise is precisely known, editing-by-inversion commonly fails to preserve structure. Other methods intervene in the sampling process by injecting internal model representations (e.g. attention maps) recorded during the inversion process [6, 19, 49]. These methods achieve improved fidelity but are not easily transferable across model architectures and sampling schemes.

In this work, we present FlowEdit, a text-based editing method for pre-trained text-to-image (T2I) flow models, which breaks away from the editing-by-inversion paradigm. FlowEdit is optimization-free and model agnostic, making it easily transferable between models. Rather than mapping the image to noise and the noise back to an edited image, it constructs a direct path between the source and target distributions (corresponding to the source and target text prompts). This path is shorter than that achieved by inversion, and thus maintains better fidelity to the source image. We evaluate FlowEdit on diverse edits using FLUX [27] and Stable Diffusion 3 (SD3) [12], and show that it achieves state-of-the-art results across the board (see Fig. 1).

Our method is motivated by a novel reinterpretation of editing-by-inversion as a direct path between the source and target distributions. Armed with this observation, we derive an alternative direct path that has a lower transport cost, as we show on synthetic examples.

2. Related work

To edit a real image using a pre-trained diffusion model, some methods use optimization over the image itself [18,

- 24, 25, 36]. These methods utilize the generative prior of a T2I diffusion model as a loss, which they optimize to push the image to comply with the user-provided prompt. Re-

cently, Yang et al. [54] suggested a similar method for flow models. These methods are resource intensive at test-time.

Many zero-shot editing methods do not require optimization. The first step of most of these methods is image-tonoise inversion [5, 6, 9, 19, 22, 45, 48–50, 52]. However, the noise map obtained through naive inversion [46, 52] is generally unsuitable for effective editing [19, 22, 35]. This is often attributed to the inaccurate inversion process, leading to attempts to improve the inversion accuracy [16, 34, 35, 50]. Yet, even exact inversion on synthetic data leads to unsatisfactory editing [22]. To address this, many methods extract during the inversion stage structural information implicitly encoded within the model architecture. They then inject it in the sampling process to achieve better structure preservation [5, 6, 19, 37, 49]. These methods are typically tailored for specific model architectures and sampling methods, limiting their transferability to new settings, such as to flow models [12, 27]. Several recent methods were proposed for editing with flow models [3, 44, 51]. However, these methods rely on inversion and in some cases also on feature injection, and thus suffer from the same limitations as their diffusion-model counterparts.

Unlike previous works, our method does not rely on inversion. It maps from source to target distributions without traversing through the Gaussian distribution. It also avoids optimization and does not intervene in the model internals, making it easily adaptable to new models.

##### 3. Preliminaries

###### 3.1. Rectified Flow models

Generative flow models attempt to construct a transportation between the distributions of two random vectors, X0 and X1, defined by an ordinary differential equation (ODE) over time t ∈ [0,1],

dZt = V (Zt,t)dt. (1) Here, V is a time-dependent velocity field, usually parameterized by a neural network, which has the property that if the boundary condition at t = 1 is Z1 = X1, then Z0 is distributed like X0. It is common to choose X1 ∼ N(0,I), which allows to easily draw samples from the distribution of X0. This is done by initializing the ODE at t = 1 with a sample of a standard Gaussian vector, and numerically solving the ODE backwards in time to obtain a sample Z0 from the distribution of X0 at time t = 0.

Rectified flows [2, 29, 31] are a particular type of flow models, which are trained such that the marginal at time t corresponds to a linear interpolation between X0 and X1,

Zt ∼ (1 − t)X0 + tX1. (2)

Rectified flows have the desired property that their sampling paths are relatively straight. This allows using a small number of discretization steps when solving the ODE.

###### (a) Editing by ODE Inversion

###### (b) Reinterpretation of Editing by Inversion

###### (c) Editing Using FlowEdit

𝒩(0,𝐼) 𝒩(0,𝐼)

𝒩(0,𝐼)

𝑍1src

𝑍1tar

𝑍1src

𝑍1tar

𝑍መ𝑡src 𝑍መ𝑡tar 𝑉𝑡tar

𝑍𝑡src 𝑍𝑡tar

𝑍𝑡src 𝑍𝑡tar

𝑉𝑡src

𝑉𝑡tar

𝑉𝑡src

𝑍0src

𝑍0src

𝑍መ0src

𝑍0tar

𝑍0tar

𝑍𝑡FE 𝑉𝑡Δ

𝑍1inv 𝑍0inv 𝑍1FE

𝑍𝑡inv

𝑍0FE

𝑉𝑡Δ

𝑍0src 𝑍1src/𝑍1tar 𝑍0tar 𝑍1inv 𝑍𝑡inv 𝑍0inv 𝑍1FE 𝑍𝑡FE 𝑍0FE

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

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

… …

… …

… … … …

Tiger → Cat

Tiger → Cat

Tiger → Noise → Cat

- Figure 2. Editing by inversion vs. FlowEdit. (a) In inversion based editing, the source image Z0src is first mapped to the noise space by solving the forward ODE conditioned on the source prompt (left path). Then, the extracted noise is used to solve the reverse ODE

conditioned on the target prompt to obtain Z0tar (right path). The images at the bottom visualize this transition. (b) We reinterpret inversion as a direct path between the source and target distributions (bottom path). This is done by using the velocities calculated during the inversion and sampling (green and red arrows) to calculate an editing direction (orange arrow) that drives the evolution of the direct path Ztinv through an ODE. The resulting path is noise-free, as demonstrated by the images at the bottom. (c) FlowEdit traverses a shorter direct path, ZtFE, without relying on inversion. At each timestep, we directly add random noise to Zˆ0src to obtain Zˆtsrc and use that direction to create Zˆttar from ZtFE(gray parallelogram). We then calculate the corresponding velocities and average over multiple realizations (not shown in the figure) to obtain the next ODE step (orange arrow). The images at the bottom demonstrate our noise-free path.

Text-to-image flow models employ a velocity field V (Xt,t,C) that depends on a text prompt C. Such models are trained on pairs of text and image data, (C,X0), so as to allow sampling from the conditional distribution of X0|C.

###### 3.2. Image editing using ODE inversion

Suppose we are given a real image, Xsrc, which we want to edit by providing a text prompt csrc describing the image and a text prompt ctar describing the desired edit. A common approach to do so is by using a pre-trained text-conditioned diffusion/flow model and employing inversion. Specifically, let us denote the text conditioned velocities as V src(Xt,t) ≜ V (Xt,t,csrc) and V tar(Xt,t) ≜ V (Xt,t,ctar). Methods relying on inversion start by extracting the initial noise map corresponding to the source image. This is done by traversing the forward process defined by the ODE

dZtsrc = V src(Ztsrc,t)dt, (3)

starting at t = 0 from the source image, Z0src = Xsrc, and reaching a noise map Z1src at t = 1. The process Ztsrc is the path shown on the left of Fig. 2a. Then, the reverse ODE,

dZttar = V tar(Zttar,t)dt, (4) is solved backward in time, starting from t = 1 with the extracted noise, Z1tar = Z1src, and reaching an edited image Z0tar at t = 0. This is the path shown on the right of Fig. 2a.

Editing by inversion tends to produce unsatisfactory results, and is thus usually accompanied by feature map injection. While the sub-optimal results are commonly blamed on the inaccurate discretization of the ODE, they are also encountered when editing generated images and using their ground-truth initial noise maps [22]. Our method retains the simplicity of editing by inversion and achieves state of the art results without any intervention in the model internals.

##### 4. Reinterpretation of editing by inversion

Before presenting FlowEdit, we reinterpret the editing-byinversion approach. This will serve to motivate our method.

Inversion based editing transports between the source and target distributions, while passing through the distribution of Gaussian noise. However, this approach can also be expressed as a direct path between the source and target distributions. Namely, we can construct a path Ztinv that starts at t = 1 in the source distribution and reaches the target distribution at t = 0. Specifically, given the forward and reverse flow trajectories, Ztsrc and Zttar, we define

Ztinv = Z0src + Zttar − Ztsrc. (5)

Note that when going from t = 1 to t = 0 we transition from the source image, Z1inv = Z0src = Xsrc (because Z1tar = Z1src), to the edited image, Z0inv = Z0tar. This path is depicted

|| |
|---|
<br><br>Source distribution<br><br>| |
|---|
<br><br>𝑍𝑡src|𝑍𝑡FE|
|---|---|
| |𝑍<br><br>𝑍𝑡tar|
| | |

||FlowEdit|
|---|
<br><br>Target distribution<br><br>inv|
|---|
||Editing by<br><br>Inversion|
|---|
|

𝑡

- Figure 3. Source to target pairings for editing-by-inversion and FlowEdit. Samples from the two modes in the source distribution are colored in blue and red (top left). As can be seen in the right panes, FlowEdit results in separated blue and red modes,

- as opposed to editing-by-inversion, where the modes are intermixed, indicating higher transport cost in terms of MSE. This can be explained by the Gaussian in the bottom left, through which the editing-by-inversion path must traverse before reaching the target. Please see the supplementary video for animated visualizations.
- at the bottom of Fig. 2b, in which the gray parallelogram visualizes the relation (5).

Let us express (5) as an ODE. Differentiating both sides of (5) and substituting (3) and (4), this relation becomes

dZtinv = Vt∆(Ztsrc,Zttar)dt, (6)

where Vt∆(Ztsrc,Zttar) = V tar(Zttar,t)−V src(Ztsrc,t). This is visualized by the green, red, and orange arrows in Fig. 2b.

Now, isolating Zttar from (5) and substituting it into (6), we get that the direct path is the solution of the ODE

dZtinv = Vt∆(Ztsrc,Ztinv + Ztsrc − Z0src)dt (7) with boundary condition Z1inv = Z0src at t = 1.

There exist, of course, many paths connecting Z0src and Z0tar. What is so special about the path defined by (7)? It turns out that images along this path are noise-free, as shown at the bottom of Fig. 2b. The reason this happens is that the noisy images Zttar and Ztsrc contain roughly the same noise constituent. Therefore, the vectors V tar(Zttar,t) and V src(Ztsrc,t), which point at the source and target distributions, respectively, remove roughly the same noise component, so that the difference vector Vt∆(Ztsrc,Zttar) (orange arrow in Fig. 2b), encompasses the difference only between the clean image predictions.

How do images evolve along this direct path? At the beginning of the path (t close to 1), the noise level in Ztsrc and

Zttar is substantial, and therefore the vector Vt∆(Ztsrc,Zttar) captures only differences in coarse image structures. As t gets smaller, the noise level drops and higher frequency contents are unveiled. In other words, the path (7) constitutes a sort of autoregressive coarse-to-fine evolution from Z0src to Z0tar. This can be seen in the images at the bottom of Fig. 2b, where the first features that get modified are the coarse structures, and the last features to get updated are the fine textures. See App. J for additional illustrations.

##### 5. FlowEdit

The fact that editing-by-inversion can be expressed as a direct ODE does not imply that it induces a desirable pairing between source and target samples. Figure 3 shows a simple example, where the source and target distributions are Gaussian mixtures with modes centered at

{(−√152,−√152),(√152, √152)} and {(−15,0),(0,15)}, respectively. A good editing method should map each mode in the

source distribution to its closest mode in the target distribution. In the context of image editing, this means that source images are modified as little as possible, while transporting to the target distribution. However, as seen on the bottom right of Fig. 3, this is not what happens in inversion. The reason the induced pairings look the way they look can be understood by inspecting how samples are mapped to their “initial” Gaussian noise component (bottom left).

Our goal is to construct an alternative mapping, which leads to lower distances between the source and target samples. Our method, which we coin FlowEdit, is illustrated on the top right of Fig. 3. As can be seen, it maps each mode in the source distribution to its nearest mode in the target distribution. Compared to editing-by-inversion, FlowEdit achieves less than twice as low a transportation cost, measured by the average squared distance between source samples and their paired target samples. We emphasize that our method is based on a heuristic, which is not guaranteed to precisely map to the target distribution. In particular, it does not necessarily coincide with the optimal transport mapping. Yet, as we will see in the context of images, it achieves state-of-the-art results in terms of structure preservation (i.e. small transportation cost) while adhering to the target text prompt (generating samples in the target distribution).

How can we depart from the pairings dictated by inversion? Our key idea is to use multiple random pairings, and average their associated velocity fields. More concretely, we propose to use the same ODE as (7), but to replace the inversion path Ztinv by a different random process with the same marginals. Then, at each timestep, we average the directions corresponding to different realizations of that process. Specifically, consider the alternative forward process

###### Zˆtsrc = (1 − t)Z0src + tNt, (8)

Algorithm 1 Simplified algorithm for FlowEdit

Input: real image Xsrc, ti Ti=0,nmax,navg Output: edited image Xtar

###### Init: ZtFE

= X0src

max

###### for i = nmax to 1 do

i ∼ N(0, 1) Ztsrc

Nt

Optionally average navg samples

← (1 − ti)Xsrc + tiNt

i

i

− Xsrc Vt∆

+ Ztsrc

← ZtFE

Zttar

i

i

i

← V tar(Zttar

,ti) − V src(Ztsrc

,ti) ZtFE

i

i

i

← ZtFE

+ (ti−1 − ti)Vt∆

i−1

i

i

end for Return: Z0FE = X0tar

where Nt is a Gaussian process that is statistically independent of Z0src and has marginals Nt ∼ N(0, 1) for every t ∈ [0,1]. We construct a path ZtFE by solving the ODE

dZtFE = E Vt∆(Zˆtsrc,ZtFE + Zˆtsrc − Z0src) Z0src dt (9)

with boundary condition Z1FE = Z0src at t = 1. Note that the expectation here is w.r.t. Zˆtsrc (equivalently Nt). Also note that we need not specify the covariance function of Nt as (9) depends only on the marginals of the process Zˆtsrc and not on its entire distribution law.

A schematic illustration is shown in Fig. 2c. The point Zˆtsrc on the left corresponds to a single draw of Nt. The distribution of Zˆtsrc at time t is shown in cyan. For each such draw, we calculate Zˆttar = ZtFE + Zˆtsrc − Z0src, as in the editing-by-inversion ODE. This is indicated by the gray parallelogram. Now, given Zˆtsrc and Zˆttar, we compute the velocity field Vt∆(Zˆtsrc,Zˆttar) = V tar(Zˆttar,t) − V src(Zˆtsrc,t) to obtain an update direction. This is illustrated by the green, red, and orange arrows. We repeat this multiple times and average the resulting orange arrows (not shown in the figure) to obtain the final update direction.

###### 5.1. Practical considerations

In practice, a discrete set of timesteps {ti}Ti=0 is used to drive the editing process, where T is the number of discretization steps. Additionally, the expectation in (9) is approximated by averaging navg model predictions at each timestep. As opposed to the theoretical expectation operator, when taking navg to be small, the covariance function of Nt starts playing a role. To obtain a good approximation, it is possible to exploit the averaging that naturally occurs across timesteps. We do so by choosing the covariance function of Nt to satisfy E[NtNs] = 0 for every |t−s| > δ, where δ is the ODE discretization step, so that the noise becomes independent across timesteps.

Note that the averaged velocity term in (9) corresponds to the difference E[V tar(Zˆttar,t)|Z0src]−E[V src(Zˆtsrc,t)|Z0src],

Generated

FlowEdit

Editing by Inversion

cats distribution

dogs distribution

dogs distribution

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Figure 4. Cats and dogs experiment. We generated 1000 cat images using diverse prompts and edited them to dog images using both FlowEdit and inversion. FlowEdit outperforms editing by inversion, achieving a lower transport costs and better FID and KID scores (computed against 1000 generated dog images).

which could be computed by sampling Zˆttar and Zˆtsrc independently. However, we choose to compute them with the same noise instances, which further improves robustness to small values of navg. This is aligned with the observation of Hertz et al. [18] in the context of diffusion models, that calculating a difference between correlated noisy marginals reduces artifacts. See Sec. 5.2 for further discussion.

Similarly to [22, 33], we define an integer that determines the starting timestep for the process 0 ≤ nmax ≤ T, meaning that the process is initialized with ZtFE

= Xsrc. When nmax = T, the full edit path is traversed and the strongest edit is obtained. When nmax < T, the first (T −nmax) timesteps are skipped, effectively shortening the edit path. This is equivalent to inversion, where weaker edits are obtained by inverting up to timestep nmax, and sampling from there. We illustrate the effect of nmax in App. C.

nmax

Algorithm 1 provides a simplified overview of our algorithm. For a more detailed version, please see Alg. S1.

###### 5.2. Relation to optimization based methods

Superficially, FlowEdit may seem similar to optimization based methods like DDS [18] and PDS [25]. These techniques, introduced for diffusion rather than flow, attempt to minimize a loss that measures the alignment between the model’s noise predictions with the source and target prompts. They approximate the gradient of the loss using only forward-passes through the model. For example, in DDS, the update step is the difference between the two noise predictions, which can be considered similar to our Vt∆(Zˆtsrc,Zˆttar) direction. However, as we show in App. L, writing Vt∆(Zˆtsrc,Zˆttar) in terms of noise predictions reveals that it is different from both the DDS and the PDS updates.

One may still wonder whether FlowEdit can be viewed as an optimization process, only for a different loss. Specifically, similarly to DDS, which attempts to minimize the norm of the difference between noise predictions in diffusion models, maybe FlowEdit attempts to minimize the norm of the difference between velocity predictions in flow

Real image Edited image Real image Edited image Real image Edited image

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Bikes Vespa Rabbit Puppy Beer

Milk

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Luna Sol Wooden statue Crown Top hat

Woman

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Coconut Baseball Pink toy hourse Wolf howling Husky looking

Horse

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Penguins Origami Clownfish Goldfish German shepherd Deer

- Figure 5. FlowEdit results. FlowEdit successfully edits diverse images using various editing prompts. The edits preserve the structure of the original image, changing only the specified region. FLUX was used for the first and third rows and SD3 for the second and fourth rows.

models? In App. L, we show that this optimization viewpoint is unjustified even for DDS. Namely, the DDS iterations do not decrease the DDS loss; they rather tend to increase it. Moreover, if allowed to continue beyond the default number of iterations, the quality of the edited image deteriorates. The optimization viewpoint is unnatural also because unlike standard optimizers, FlowEdit does not choose timesteps at random but rather uses a monotonically decreasing schedule {ti}n

i=0 . Additionally, it must use a learning rate of exactly dt = ti−1−ti at iteration i (namely, the sum of all step-sizes must be 1). Any slight deviation from this learning rate causes the results to deteriorate significantly (see App. L).

max

###### 5.3. Comparison to editing by (exact) inversion

To demonstrate the reduced transport cost of FlowEdit compared to editing-by-inversion, we evaluate both methods on

a synthetic dataset of model-generated images. This way, the initial noise maps are known, ensuring the inversion is exact and eliminating potential issues of approximate inversion. The dataset consists of 1000 cat images generated by SD3 using variations of the prompt “a photo of cat” generated by Llama3 [11]. We edit these images using both FlowEdit and inversion, with the target prompt identical to the source prompt, except for replacing “cat” with “dog”. We calculate the transport cost for both methods by measuring the MSE between source and edited images in the model’s latent space, as well as LPIPS [55] on the decoded images. As expected, FlowEdit achieves a lower transport cost (1376 vs. 2239 for MSE, 0.15 vs. 0.25 for LPIPS), indicating superior preservation of the structure and semantics of the source image. Figure 4 shows a small qualitative comparison for the cat-to-dog edits.

To assess the alignment of our edits with the target distri-

Stable Diffusion 3 FLUX

Real image SDEdit 0.4 ODE Inv. iRFDS FlowEdit SDEdit 0.75 ODE Inv. RF Inv. RF Edit FlowEdit

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

A large tiger standing in a swamp → A large lion standing in a swamp

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

A tall white lighthouse, illuminated by bright light → The Big Ben, illuminated by bright light

figure

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

A gas station with a CAFE sign → A gas station with a FREE sign

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

A three layer cake decorated with fruits → A three layer cake decorated with strawberries

- Figure 6. Qualitative comparisons. With both SD3 (left) and FLUX (right), FlowEdit better adheres to the target prompt while simultaneously preserving the structure of the original image. The value next to SDEdit indicates the strength.

bution, we generated 1000 dog images with SD3, using the same target prompts but with “dog” instead of “cat”. We followed by calculating FID [20] and KID [4] between the generated dog images and the edited dog images, for both methods. Our method achieves lower FID (51.14 vs. 55.88) and KID (0.017 vs. 0.023), indicating that similarly to inversion, our ODE path is able to produce images from the target distribution. See App. M for more details and results.

##### 6. Experiments

Implementation details. In our experiments we use the official weights of SD3 medium [47] and FLUX.1 dev [28], available at HuggingFace, as the base T2I flow models. For SD3 we use T = 50 steps, with nmax = 33. SD3 employs CFG [21] for their text conditioning. We set the source and target scales to 3.5 and 13.5, respectively. For FLUX we use T = 28 steps, with nmax = 24. FLUX takes CFG as an input conditioning, which we set to the values of 1.5 for the source conditioning, and 5.5 for the target. For both methods we use navg = 1 (see Sec. 5.1 and App. K). We use these hyperparameters for all the results in Figs. 5,6,7.

Dataset. Our dataset consists of a diverse set of over 70 real images of dimension 10242 from the DIV2K dataset [1] and from royalty free online sources [38, 39]. Each image has a source prompt, which was obtained from LLaVA-

1.5 [30] and manually refined. The source prompts have little effect on FlowEdit’s results. In particular, they can be omitted (App. I). For each image, several handcrafted target prompts are provided to facilitate diverse edits. Overall, the dataset consists of over 250 text-image pairs, and is used to evaluate our and the competing methods. The dataset, including source and target prompt pairs, is available on our official github repository.

Competing methods. We compare our method against several competing text-based real image editing methods that use flow models. The first is editing by ODE inversion, which we apply with the same hyperparameters as our method. The second is SDEdit [33], which is easily applied to flow models by adding noise to the source image up to a specified nmax step and then performing regular sampling conditioned on a target prompt to obtain the edit. The parameter nmax controls the strength of the edit. Additionally, we compare to iRFDS [54], which is an SDS-based editing method for flow models. We use their official implementation and hyperparameters, which are available only for SD3. Furthermore, we compare to RF-Inversion [44]. As an official implementation is not available at the time of writing, we implement it based on the provided pseudo-code and the hyperparameters reported in the SM, which are available only for FLUX. Finally, we compare to RF Edit [51]

Stable Diﬀusion 3

###### FLUX

0.45

0.35

0.35

←LPIPS

0.25

ODE Inv.

| |
|---|

0.25

ODE Inv.

SDEdit RF Inv. RF Edit Ours

| |
|---|

SDEdit

| |
|---|

| |
|---|

0.15

| |
|---|

iRFDS

0.15

Ours

0.33 0.34 0.35 CLIP →

0.31 0.32 0.33 0.34 CLIP →

- Figure 7. Quantitative comparisons. FlowEdit achieves a favorable balance between text adherence (CLIP) and structure preservation (LPIPS) compared to other methods. Connected markers represent different hyperparameters (see App. B).

using their official implementation, which is also available only for FLUX. Whenever possible, we performed a hyperparameters search for each method to identify the optimal settings for editing. The final parameters used to create the edited images in the paper are detailed in App. B. We do not compare FlowEdit to text-based image editing techniques for diffusion models [6, 22, 49], as these are not easily adapted to flow models, and a direct comparison would not be fair due to model differences.

Qualitative evaluation. Figures 1, 5, S1 show editing results obtained with FlowEdit across a diverse set of images and target prompts, including localized edits to text and objects, modifications of multiple objects at once, pose changes, etc. Our edits exhibit good structural preservation of the source image, and simultaneously maintain good adherence to the target text. Figure 6 shows comparisons between FlowEdit and the competing methods for both SD3 and FLUX. FlowEdit is the only method that consistently adheres to the text prompt while preserving the structure of the original image. For example, FlowEdit is the only method that can both change the large text on the gas station sign (third row) and preserve the cars and background of the source image. Some methods, like RF-Inversion, are able to change the text on the sign but fail to maintain the structure of the original image, while others, such as iRFDS, somewhat preserve the structure but fail to fully change the sign. In the last row, while all methods perform the required edit (changing the decoration to strawberries), only FlowEdit (for both SD3 and FLUX) preserves the background accurately. Other methods introduce additional, unintended elements such as furniture and flowers. See App. B for more qualitative comparisons.

Quantitative evaluation. We numerically evaluate the results of FlowEdit and the other methods using LPIPS [55] to measure the semantic structure preservation (lower is better) and CLIP [41] to assess text adherence (higher is bet-

Real image Edited image Real image Edited image

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

House by a lake Lighthouse

…in minecraft style …in watercolor style

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Kid running Dog

…in anime style …in Disney style

Figure 8. Text-based style editing. FlowEdit changes the style of an image at the cost of slightly deviating from the original image structure. SD3 was used for the first row and FLUX for the second. Hyperparameters used for these results are reported in App. D.

ter). Figure 7 displays these results for SD3 (left) and FLUX (right), with varied hyperparameters. The plots show that FlowEdit achieves a favorable balance between structure preservation and text adherence. The other methods either maintain the structure of original image at the cost of a weak edit or modify the image with little regard to its original semantics. Additional quantitative evaluations and details, including additional similarity metrics beyond LPIPS, are provided in App. B.

Text-based style editing. The hyperparameters used for the previous experiments lead to strong structure preservation and are thus not optimal for style editing. Figure 8 demonstrates the results of FlowEdit for text-based style editing, where by allowing some deviation from the original structure, we gain stylistic flexibility. To achieve these results, we simply remove the dependence on the source image in the final generation steps, as discussed in App. D.

##### 7. Conclusion and limitations

We introduced FlowEdit – an inversion-free, optimizationfree and model agnostic method for text-based image editing using pre-trained flow models. Our approach constructs a direct ODE between the source and target distributions (corresponding to source and target text prompts), without passing through the standard Gaussian distribution as in inversion-based editing. Evaluations on synthetic datasets show that FlowEdit achieves lower transport costs, and thus stronger structure preservation. This translates to state-ofthe-art performance across various editing tasks, as we illustrated with FLUX and SD3. While FlowEdit’s strong structure preservation is beneficial for precise editing tasks, it can become a limitation when substantial modifications to large regions of the image are desired. This is illustrated in Fig. S6 in the context of pose and background editing.

##### References

- [1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 126–135, 2017.
- [2] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.
- [3] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel CohenOr. Stable flow: Vital layers for training-free image editing. arXiv preprint arXiv:2411.14430, 2024.
- [4] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In International Conference on Learning Representations, 2018.
- [5] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8861–8870, 2024.
- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22560–22570, 2023.
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [8] Nathaniel Cohen, Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Slicedit: Zeroshot video editing with text-to-image diffusion models using spatio-temporal slices. In Proceedings of the 41st International Conference on Machine Learning, pages 9109–9137. PMLR, 2024.
- [9] Gilad Deutch, Rinon Gal, Daniel Garibi, Or Patashnik, and Daniel Cohen-Or. Turboedit: Text-based image editing using few-step diffusion models. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024.
- [10] Sander Dieleman. Diffusion is spectral autoregression, 2024.
- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [13] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity

- using synthetic data. Advances in Neural Information Processing Systems, 36, 2024.
- [14] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. In Computer Vision

– ECCV 2024, pages 395–413, 2024.

- [15] Ren´e Haas, Inbar Huberman-Spiegelglas, Rotem Mulayoff, Stella Graßhof, Sami S Brandt, and Tomer Michaeli. Discovering interpretable directions in the semantic latent space of diffusion models. In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pages 1–9. IEEE, 2024.
- [16] Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Anastasis Stathopoulos, Xiaoxiao He, Yuxiao Chen, et al. Proxedit: Improving tuning-free real image editing with proximal guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4291–4301, 2024.
- [17] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Editing 3d scenes with instructions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19740–19750, 2023.
- [18] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2328–2337, 2023.
- [19] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023.
- [20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.
- [22] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469– 12478, 2024.
- [23] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [24] Jeongsol Kim, Geon Yeong Park, and Jong Chul Ye. Dreamsampler: Unifying diffusion sampling and score distillation for image manipulation. In Computer Vision – ECCV 2024, pages 398–414. Springer Nature Switzerland, 2024.
- [25] Juil Koo, Chanho Park, and Minhyuk Sung. Posterior distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13352–13361, 2024.
- [26] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. In The

- Eleventh International Conference on Learning Representations, 2023.
- [27] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. Accessed: 202411-14.
- [28] Black Forest Labs. Official weights of FLUX.1 dev. https : / / huggingface . co / black - forest labs/FLUX.1-dev, 2024. Accessed: 2024-11-14.
- [29] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.
- [30] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.
- [31] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.
- [32] Hila Manor and Tomer Michaeli. Zero-shot unsupervised and text-based audio editing using DDPM inversion. In Proceedings of the 41st International Conference on Machine Learning, pages 34603–34629. PMLR, 2024.
- [33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.
- [34] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pages 2063–2072, 2025.
- [35] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
- [36] Hyelin Nam, Gihyun Kwon, Geon Yeong Park, and Jong Chul Ye. Contrastive denoising score for text-guided latent diffusion image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9192–9201, 2024.
- [37] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023.
- [38] Pexels. Pexels - free stock photos & videos. https:// www.pexels.com, 2024. Accessed: 2024-11-14.
- [39] PxHere. Pxhere - free images & free stock photos. https: //pxhere.com/, 2024. Accessed: 2024-11-14.
- [40] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15932–15942, 2023.

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [42] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. In The Eleventh International Conference on Learning Representations, 2023.
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [44] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. In The Thirteenth International Conference on Learning Representations, 2025.
- [45] Dvir Samuel, Barak Meiri, Haggai Maron, Yoad Tewel, Nir Darshan, Shai Avidan, Gal Chechik, and Rami Ben-Ari. Lightning-fast image inversion and editing for text-to-image diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.
- [46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.
- [47] StablityAI. Official weights of SD3 medium diffusers. https://huggingface.co/stabilityai/ stable-diffusion-3-medium-diffusers, 2024. Accessed: 2024-11-14.
- [48] Linoy Tsaban and Apolin´ario Passos. Ledits: Real image editing with ddpm inversion and semantic guidance. arXiv preprint arXiv:2307.00522, 2023.
- [49] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023.
- [50] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22532–22541, 2023.
- [51] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024.
- [52] Chen Henry Wu and Fernando De la Torre. A latent space of stochastic diffusion models for zero-shot image editing and guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7378–7387, 2023.
- [53] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.

- [54] Xiaofeng Yang, Chen Cheng, Xulei Yang, Fayao Liu, and Guosheng Lin. Text-to-image rectified flow as plug-and-play priors. In The Thirteenth International Conference on Learning Representations, 2025.
- [55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

- A. Additional results Figure S1 shows additional editing results obtained with FlowEdit.

Real image Edited image Real image Edited image Real image Edited image

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Bulldog Lion Cat Fox Bronze

Cat and dog

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Dog Lego LOVE Parrot Glass sculpture

STOP

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

CAFE FOOD Silver statue Grizzly bear Black bear

Cat

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Kid Robot Dogs Wolves Mountain Cake

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Lizard Frog Tree Cherry blossom Female deer

Male deer

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Owl

Glass sculpture Lighthouse Rapunzel’s tower Man Golden statue

- Figure S1. Additional FlowEdit results. FLUX was used for the first, third, and fifth rows, and SD3 for the second, fourth and sixth rows.

##### B. Comparisons

###### B.1. Additional qualitative comparisons

- Figure S2 shows additional comparisons between FlowEdit and the competing methods with both SD3 (left) and FLUX (right). The value next to SDEdit indicates the editing strength (see App. B.2).

Stable Diffusion 3 FLUX

Real image SDEdit 0.4 ODE Inv. iRFDS FlowEdit SDEdit 0.75 ODE Inv. RF Inv. RF Edit FlowEdit

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

A German shepherd dog standing in a snowy field → A husky dog standing in a snowy field

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

A colorful parrot perching on a tree branch → A gray pigeon perching on a tree branch

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

A board display the words “FREE WIFI” → A board display the words “FREE HUGS”

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

A slice of red cake with white frosting → A slice of chocolate cake with white frosting

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

A neon sign of a restaurant called Luna → A neon sign of a restaurant called Sol

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

A dalmatian dog looking up → A Cheetah looking to the camera

###### Figure S2. Additional qualitative comparisons.

###### B.2. Additional details on the experiment settings

We compare FlowEdit against the baseline methods of ODE inversion (Sec. 3.2) and SDEdit using both FLUX and SD3, as well as against RF-Inversion, RF Edit (FLUX) and iRFDS (SD3). We were not able to compare to Stable Flow [3], as their method is not compatible with Nvidia RTX A6000 (which we used for running FlowEdit, as well as all the other methods) even with cpu offloading and when changing the image size from 1024 × 1024 to 512 × 512.

Figure 7 in the main text presents the CLIP vs. LPIPS results for FlowEdit and the competing methods with both SD3 and FLUX. Below, we detail the hyperparameters used for SD3 (Tab. S1) followed by those for FLUX (Tabs. S2, S3, S4). The results shown in Figs. 6, S2 and Tabs. S5 and S6 were obtained using the parameters listed in Tabs. S1, S2 and S3. The bold entries indicate the specific settings used when multiple hyperparameter options were tested.

- B.2.1. Stable Diffusion 3 hyperparameters In Fig. 7 in the main text both FlowEdit and ODE Inversion are shown with three options for the CFG target scale, as detailed

- in Tab. S1, from left to right. For SDEdit the different values of nmax represents different strength settings ranging from 0.2 to 0.8 in intervals of 0.1. The markers in the figure indicate results with these strength values from left to right.

Table S1. SD3 hyperparameters.

T steps nmax CFG @ source CFG @ target SDEdit 50 10, 15, 20, 25, 30, 35, 40 - 13.5

ODE Inv. 50 33 3.5 13.5, 16.5, ,19.5 iRFDS official implementation and hyperparameters

FlowEdit 50 33 3.5 13.5, 16.5, ,19.5

B.2.2. FLUX hyperparameters In Fig. 7 in the main text both FlowEdit and ODE Inversion are shown with three options for the CFG target scale, as detailed

- in Tab. S2, from left to right.

For SDEdit the different values of nmax represent different strength values, corresponding to 0.25, 0.5, 0.75. The markers in the figure indicate results with these strength values from left to right.

Table S2. FLUX hyperparameters.

T steps nmax CFG @ source CFG @ target SDEdit 28 7, 14, 21 - 5.5

ODE Inv. 28 20, 24 1.5 3.5, 4.5, 5.5 FlowEdit 28 24 1.5 3.5, 4.5, 5.5

For RF-Inversion, we explore multiple sets of hyperparmeters, as the paper does not report specific ones for general editing. Following the SM of their work, we experimented with several combinations, detailed in Tab. S3 using their notations.

Table S3. RF-Inv. hyperparameters.

T steps s starting time τ stopping time η strength 28 0 8, 7, 6 0.9,1.0

For RF Edit, we explore multiple injection scales as the paper does not report the injection scale used for image editing. All hyperparameters used for RF Edit are listed in Tab. S4 using their notations.

Lastly, as can be seen in Fig. 7, ODE Inversion on FLUX with these hyperparameters achieves a high CLIP score at the cost of a high LPIPS score, indicating it does not balance these metrics effectively. By varying nmax, ODE Inversion could achieve lower (better) LPIPS scores at the cost of lower (worse) CLIP scores. Figure S3 illustrates the CLIP and LPIPS scores for the different methods using FLUX. Specifically, ODE Inversion is also shown with nmax = 20 in addition to nmax = 24 as

Table S4. RF Edit. hyperparameters.

Steps Guidance Injection

30 2 2, 3, 4, 5

shown in the main text. However, with these adjusted hyperparameters, ODE Inversion struggles to adhere to text prompts. RF-Inversion results with η = 1.0 are also illustrated in Fig. S3 and Tab. S6. Again, with this hyperparameter this method struggles to adhere to text prompts.

0.35

0.30

0.25

←LPIPS

| |
|---|

ODE Inv. nmax = 24 ODE Inv. nmax = 20 SDEdit

0.20

| |
|---|

0.15

- RF Inv. η = 0.9

- RF Inv. η = 1.0

| |
|---|

RF Edit

0.10

Ours

0.31 0.32 0.33 0.34 CLIP →

- Figure S3. Additional quantitative comparisons. In addition to the results shown in Fig. 7 for FLUX, we include additional hyperparameters: ODE Inversion with nmax = 20 and RF-Inversion with η = 1.0. These configurations also struggle to achieve a good balance between the CLIP and LPIPS metrics. In contrast, FlowEdit demonstrates the best balance.

###### B.3. Metrics comparisons

In addition to quantifying the structure preservation using LPIPS, as described in the main text, we now report alternative metrics. Specifically, we use DreamSim [13] as well as cosine similarity in the CLIP [41] and DINO [7] embedding spaces, between the source and target images’ embeddings. For DreamSim, a lower score indicates better structure preservation. For CLIP images and DINO, a higher score (bounded by 1) means higher structure preservation.

Tables S5, S6 illustrate the results of these metrics, alongside LPIPS and CLIP for the hyperparameters described above. As can be seen, FlowEdit is the only method that is able to both adhere to the text prompt and preserve the structure of the source image.

- Table S5. SD3 metrics. The first , second and third best scores are highlighted for each metric. CLIP-T measures adherence to text, while the other four scores measure structure preservation.

CLIP-T ↑ CLIP-I ↑ LPIPS ↓ DINO ↑ DreamSim ↓

SDEdit 0.2 0.33 0.885 0.251 0.634 0.213 SDEdit 0.4 0.34 0.854 0.316 0.564 0.273

ODE Inv. 0.337 0.813 0.318 0.549 0.326

iRFDS 0.335 0.822 0.376 0.534 0.327 FlowEdit 0.344 0.872 0.181 0.719 0.253

- Table S6. FLUX metrics. The first , second and third best scores are highlighted for each metric. CLIP-T measures adherence to text, while the other four scores measure structure preservation.

CLIP-T ↑ CLIP-I ↑ LPIPS ↓ DINO ↑ DreamSim ↓ SDEdit 0.5 0.316 0.902 0.264 0.637 0.18

SDEdit 0.75 0.331 0.862 0.348 0.557 0.26

ODE Inv. 0.341 0.822 0.374 0.505 0.328 RF Inv. 0.334 0.856 0.34 0.558 0.266

RF Edit 2 0.344 0.833 0.335 0.53 0.32 RF Edit 5 0.332 0.876 0.22 0.65 0.22

FlowEdit 0.337 0.875 0.223 0.682 0.252

##### C. Effect of nmax

As described in the main text, we define an integer to determine the starting timestep of the process, where 0 ≤ nmax ≤ T. The process is initialized with ZtFE

= Xsrc. When nmax = T, the full edit path is traversed and the strongest edit is obtained. For nmax < T, the first (T − nmax) timesteps are skipped, effectively shortening the edit path. This is equivalent to inversion, where weaker edits are obtained by inverting up to timestep nmax, and sampling from there. Figure S4 illustrates the effect of nmax on the results. The CFG and T used for these editing results are the same as mentioned in the main text, except for nmax whose value is specified in the figure.

nmax

|𝒏𝐦𝐚𝐱 = 𝟑𝟑 Ours|
|---|

|𝑛max = 10|
|---|

|𝑛max = 20|
|---|

|𝑛max = 30|
|---|

|𝑛max = 40|
|---|

|𝑛max = 50|
|---|

Real image

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

StableDiffusion3FLUX

|Clownfish swimming in a coral reef → Sea turtle swimming in a coral reef|
|---|

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

A steak accompanied by leaf salad → A schnitzel accompanied by leaf salad

𝒏𝐦𝐚𝐱 = 𝟐𝟒 Ours

𝑛max = 18 𝑛max = 19

𝑛max = 26

𝑛max = 27

𝑛max = 28

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

|A glass of milk with a straw → A glass of milkshake with cherry and a straw|
|---|

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

|A coconut shell filled with splashing water → A human head shell filled with splashing water|
|---|

###### Figure S4. Effect of nmax.

##### D. Text-based style editing

FlowEdit excels in structure preserving edits. This trait is usually desired in text-based image editing, yet in some cases it could be too limiting. One such scenario is text-based style editing, where we would like to slightly deviate from the original structure in order to achieve a stronger stylistic effect. While nmax allows some control over the structure, it might not be enough by itself to control the finer details required for style changes, i.e. higher frequency textures.

To achieve better control over the finer details, we define a new hyperparameter, nmin, that controls the structure deviation at lower noise levels, and effectively allows stronger modifications to the higher frequencies. Specifically, when i < nmin we apply regular sampling with the target text, rather than following FlowEdit as described in the main text. These steps are further detailed in our full algorithm, Alg. S1.

Figure S5 illustrates the effect of nmin for text-based style editing. For small nmin the edited image preserves the structure of the original image (especially in the higher frequencies), while for higher nmin values it slightly deviates from it, achieving a stronger edit. These results were obtained using FLUX with T = 28 steps, nmax = 21 and CFG scales of 2.5 and 6.5 for the source and target conditionings, respectively. The nmin values are mentioned in the figure.

Real image 𝑛min = 0 𝑛min = 6 𝑛min = 12 𝑛min = 18

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

A bear walking in a river in Pixar style

- Figure S5. Effect of nmin. When nmin is small, the edited image remains close to the original image and struggles to align with the text. In contrast, larger values of nmin result in better adherence to the text but at the cost of greater deviation from the original image.

The table below includes the hyperparameters used for the text-based style editing results in Fig. 8.

Table S7. Hyperparameters used for the results displayed in Fig. 8.

Model nmin nmax CFG @ source CFG @ target House by a lake in minecraft style SD3 15 31 3.5 13.5

Lighthouse in watercolor painting style SD3 15 31 3.5 13.5

Kid running in anime style FLUX 14 21 2.5 6.5 Dog in Disney style FLUX 14 24 1.5 4.5

- E. Hyperparmeters used for Fig. 1 The results in Fig. 1 were achieved using the hyperparmeters described in Tab. S8 below.

Table S8. Hyperparameters used for the results displayed in Fig. 1.

Model nmin nmax CFG @ source CFG @ target this → home FLUX 0 24 1.5 3.5

Cat → Raccoon FLUX 0 24 1.5 4.5 LOVE → FLOW FLUX 0 24 1.5 4.5

Bread → Bacon FLUX 0 24 1.5 3.5 Mountain → Volcano FLUX 0 24 1.5 5.5

Lizard → Dragon SD3 0 33 3.5 13.5 Man jumping in Pixar style SD3 15 21 3.5 13.5 Bread → Steak SD3 0 33 3.5 13.5 White dog w/ cat → Dalmatian w/o cat SD3 0 33 3.5 13.5

##### F. Full Algorithm

Algorithm S1 Full FlowEdit algorithm

Input: real image Xsrc, ti Ti=0,nmax,nmin,navg Output: edited image Xtar

###### Init: ZtFE

= X0src

max

for i = nmax to nmin+1 do

i ∼ N(0, 1) Ztsrc

Nt

Optionally average navg samples

← (1 − ti)Xsrc + tiNt

i

i

Zttar

− Xsrc Vt∆

← ZtFE

+ Ztsrc

i

i

i

← V tar(Zttar

,ti) − V src(Ztsrc

,ti) ZtFE

i

i

i

← ZtFE

+ (ti−1 − ti)Vt∆

i−1

i

i

end for if nmin = 0 then

Return: Z0FE = X0tar else

min ∼ N(0, 1) Ztsrc

Nn

###### )Xsrc + tn

###### ← (1 − tn

###### Nn

min

min

min

nmin

− Xsrc for i = nmin to 1 do

###### + Ztsrc

###### ← ZtFE

###### Zttar

nmin

nmin

nmin

Zttar

← Zttar

+ (ti−1 − ti)V tar(Zttar

,ti)

i−1

i

i

end for Return: Z0tar = X0tar

end if

##### G. Limitations

FlowEdit excels in structure preserving edits, and is therefore often limited in its ability to make substantial modifications to large regions of the image. This is illustrated in Fig. S6 for pose and background editing, respectively. In these cases, FlowEdit does not fully modify the image according to the target prompt and it fails at preserving the source identity. To obtain a greater deviation from the source image, it is possible to increase nmin, as we illustrate in App. D. This is helpful for style editing, but is often still not enough for background and pose edits.

Real image Edited image Real image Edited image

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Sitting Grass

Jumping Snow

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Sitting Grass Snow

Standing

- Figure S6. Limitations. FlowEdit is often fails at making substantial modifications to large region of the image, as in pose (left) and background (right) editing.

##### H. Effect of random noise on the results

FlowEdit adds random noise to the source image. If the number of samples over which we average is small, then the algorithm is effectively stochastic. Namely, it produces different editing results with different random seeds. These variations can lead to diverse edits for the same text-image pair. As shown in Fig. S7, the rocks are transformed into different bonsai trees, and the tent appears in various locations within the image.

However, these changes can also result in failure cases, as illustrated in Fig. S8. For example, when editing a white horse into a brown one, the edited results sometimes show the horse with more than four legs or suffer from other artifacts.

The editing results in both figures were obtained using SD3 and the hyperparameter mentioned in the main text.

Real image Edited image

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Rocks Bonsai tree

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Tree with tent

Figure S7. FlowEdit diverse results due to different added noise.

Real image Edited image

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Brown bear Polar bear

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

White horse Brown horse

Figure S8. Failure cases due to different added noise.

##### I. Effect of source prompt on the results

In the experiments detailed in the main text we paired each image with a source prompt, generated by LLaVA-1.5 and manually refined. However, a source prompt is not required for FlowEdit and in fact has little effect on FlowEdit results, as detailed in Tab. S9. We first tested FlowEdit’s sensitivity to the source prompt by generating variations on all the source prompts in the dataset using ChatGPT. We then completely omitted the source prompt and used an empty prompt. Both changes have little effect on the results. These results were obtained using SD3.

Table S9. Effect of source prompt on the results. CLIP-T measures adherence to text, while the other four scores measure structure preservation.

CLIP-T ↑ CLIP-I ↑ LPIPS ↓ DINO ↑ DreamSim ↓

original source prompt 0.344 0.872 0.181 0.719 0.253 different source prompt 0.343 0.872 0.181 0.713 0.254

w/o source prompt 0.343 0.872 0.192 0.709 0.254

##### J. Illustration of the noise-free path between the source and target distributions

As explained in Sec. 4 in the main text, the path defined by (7) is noise-free, and it constitutes an autoregressive coarse-to-fine evolution from Z0src to Z0tar. Figure S9 illustrates this evolution for both synthetic and real images using editing by inversion (top). This path starts from the source image, Z1inv = Z0src. Moving along this path requires adding the vector Vt∆(Ztsrc,Zttar) to Z1inv. Illustrations of these vectors along the path are shown beneath the images in Fig. S9. Each Vt∆(Ztsrc,Zttar) image is the result of the difference between the two images above it.

It can be seen that for large t values, Vt∆(Ztsrc,Zttar) contains mainly low frequency components. As t decreases, higherfrequency components become increasingly visible. This stems from the fact that Vt∆(Ztsrc,Zttar) corresponds to the difference between V tar(Zttar,t) and V src(Ztsrc,t). At large t, the noise level is substantial and therefore this vector captures mainly lowfrequency components. As t decreases, the vector begins to capture higher-frequency details. This process constitutes an autoregressive coarse-to-fine evolution that starts from Z0src and ends at Z0tar, similarly to the evolution of the diffusion/flow process itself [10, 42].

Figure S9 also illustrates the evolution along the FlowEdit path and the Vt∆(Ztsrc,Zttar) along it. These Vt∆(Ztsrc,Zttar) vectors have the same characteristics as in the case of editing by inversion.

The autoregressive coarse-to-fine evolution from source to target is also schematically illustrated in Fig. S10 and empirically shown in Fig. S11. This evolution is illustrated in the frequency domain, using the power spectral density (PSD) transformation, following Rissanen et al. [42].

By arranging the images Ztinv/FE along this path in a video, we can animate the interpolation between the source and target images. Additionally, by using the resulting target image as the input for subsequent editing steps, we can create a smooth animation that transitions from a source image to multiple edits. This interpolation between the source and target image can be seen in Fig. S12 for editing by inversion and in Fig. S13 for FlowEdit. This noise-free path reveals, for example, how gradually a tiger becomes a bear. Furthermore, the interpolation between a cat image and a fox image, going through lion, tiger and a bear can also be seen at the end of the supplementary video.

Tiger → Cat

𝑍1inv 𝑍𝑡inv 𝑍0inv

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

#### … … …

…

[Figure 279]

[Figure 280]

[Figure 281]

EditingbyInversion

𝑉𝑡Δ

… …

Cat → Lion

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

… … … …

[Figure 290]

[Figure 291]

[Figure 292]

𝑉𝑡Δ

… …

Tiger → Cat

𝑍1FE 𝑍𝑡FE 𝑍0FE

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

… … … …

[Figure 301]

[Figure 302]

[Figure 303]

𝑉𝑡Δ

… …

FlowEdit

Cat → Lion

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

… … … …

[Figure 312]

[Figure 313]

[Figure 314]

𝑉𝑡Δ

… …

Figure S9. Illustration of the noise free path and V ∆.

𝑍FE = 𝑋src

𝑍𝑡FE

𝑍𝑡FE

𝑍0FE = 𝑍0tar

|PSD|1|
|---|---|
| | |

…

PSD

PSD

PSD

PSD

Freq.

Freq.

Freq.

Freq.

𝑍𝑡src

𝑍𝑡tar

𝑍𝑡src 𝑍𝑡tar

𝑍𝑡src 𝑍𝑡tar

PSD

PSDPSD

PSD

PSDPSD

PSD

PSD

Freq. Freq.

Freq. Freq.

Freq. Freq.

𝑍𝑡src + 𝑑𝑡𝑉𝑡src 𝑍𝑡tar + 𝑑𝑡𝑉𝑡tar

𝑍𝑡src + 𝑑𝑡𝑉𝑡src 𝑍𝑡tar + 𝑑𝑡𝑉𝑡tar

𝑍src + 𝑑𝑡𝑉src

𝑍𝑡tar + 𝑑𝑡𝑉𝑡tar

|PSD|𝑡 𝑡<br><br>|
|---|---|
| | |
| | |

𝑍𝑡tar

𝑍𝑡tar

PSD

PSD

PSD

PSD

Freq. Freq.

Freq. Freq.

Freq. Freq.

|PSD|𝑑𝑡𝑉𝑡Δ<br><br>|
|---|---|
| | |

| |
|---|

Noise Constituent

𝑑𝑡𝑉𝑡Δ

𝑑𝑡𝑉𝑡Δ

| |
|---|

𝑋src Constituent 𝑋tar Constituent

PSD

PSD

PSD

| |
|---|

Freq.

Freq.

Freq.

- Figure S10. Schematic illustration of the spectral behavior of FlowEdit’s marginals. Green and red markings on the graph depict the source and target image’s spectral constituents, respectively. The gray markings depict the added white Gaussian noise. The top row illustrates our direct ODE path (9). The second row illustrates our noisy marginals (Sec. 5). The third row illustrates the result of a flow step with step-size dt towards the clean distributions. The fourth row illustrates dtVt∆. The vector dtVt∆ captures the differences between the new source and target frequencies that were “revealed” during the denoising step. This dtVt∆ is used to drive our ODE process to the next timestep. This is in line with the noise-free, coarse-to-fine behavior we observed in our experiments, and as seen with the image examples in Fig. S9. We also provide an empirical evaluation for the illustration in S11.

|𝑍𝑡tar|
|---|

|𝑍𝑡FE|
|---|

|𝑍𝑡src|
|---|

|𝑉𝑡Δ|
|---|

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

- Figure S11. Empirical evaluation of the spectral behavior of FlowEdit’s marginals. The PSD transform [42] is shown for example images along the FlowEdit path (left), the corresponding noisy versions of the source and target images (middle), and the associated velocity fields (right). The PSDs were averaged over four different edits. The example images are the results of one of these edits, corresponding to

penguins → origami penguins. The first column pair shows the PSD and images of our ODE path ZtFE (9), progressing from top to bottom. It can be seen that the spectra of these clean images is nearly the same across all timesteps. The second and third column pairs show the

noisy marginals, and as can be seen both Zˆtsrc and Zˆttar are masked with the same amount of noise, and are valid inputs to the trained flow models. Finally, the last column pair represents Vt∆, which is noise free, and holds more low-frequency edit information in the starting timesteps, and higher frequencies at the later ones. Importantly, these statistics hold for both (reinterpreted) editing-by-inversion and our method.

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

LionTigerCatLionTigerBearBearFox→→→→

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

- Figure S12. Results of editing by inversion along the noise-free path. The cat image on the top left is used as the input to editing by inversion, where the target prompt is a lion. Then, the lion image is used as input and so forth. It can be seen that the edited images do not fully preserve the structure and fine details of the original image, e.g the grass around the cat.

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

LionTigerCatLionTigerBearBearFox→→→→

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

- Figure S13. Editing results of FlowEdit along the noise free path. The cat image on the top left is used as the input to FlowEdit, where the target prompt is a lion. Then, the lion image is used as input and so forth. It can be seen that the edited images preserve the structure and the fine details of the original image, e.g the grass around the cat, even after multiple edits.

##### K. Effect of navg

FlowEdit operates by evaluating Vt∆(·) on multiple realizations of Zˆtsrc and Zˆttar at each t. Then, this Vt∆(·) is used to drive our ODE (9). This becomes impractical in cases where model evaluations are expensive (SD3/FLUX). Fortunately, averaging

also occurs across timesteps when the noises {Nt} are chosen to be independent across timesteps. This means that with large enough T, we can use a smaller navg, reducing expensive model evaluations. Specifically, the default values of T for SD3 and FLUX are high enough for our method to perform well with navg = 1, relying purely on averaging across t.

- Figure S14 illustrates this effect. As the value of navg increases, the LPIPS distance decreases. We used SD3 and navg

values of 1, 3, 5, 10. For a large number of discretization and editing steps, i.e. T = 50, nmax = 33 (orange curve), the number of averaging steps has little effect on the results, as averaging already occurs between timesteps. Hence, increasing navg only slightly improves the LPIPS score and has a negligible effect on the CLIP score (note that the horizontal axis ticks spacing is 0.001). These hyperparameters, T = 50, nmax = 33, with navg = 1, are the hyperparameters used in our method. However, for a small number of discretization and editing steps, i.e. T = 10, nmax = 7 (purple curve), increasing navg has a substantial effect. It reduces the LPIPS distance by ∼ 0.3 and increases the CLIP score by 0.035. In both experiments, the CFG scales are the same as those described in the main text.

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>T = 50, nmax = 33<br><br>T = 10, nmax = 7| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.20

0.19

←LPIPS

0.18

0.17

0.339 0.340 0.341 0.342 0.343 0.344 0.345 CLIP →

Figure S14. CLIP vs. LPIPS for different values of navg. From top to bottom, the markers correspond to navg of 1, 3, 5, 10.

##### L. Further discussion on the relation to optimization based methods

As detailed in the main text, it is unnatural to view FlowEdit as an optimization based method, like DDS or PDS, because (i) it does not choose timesteps t at random but rather following a monotonically decreasing schedule {ti}n

i=0 , and (ii) it must use a learning rate of exactly dt = ti−1 − ti at iteration i. In Sec. L.3 below, we show that even a slight change of the step size in FlowEdit leads to rapid deterioration of output quality. This suggests that it is more natural to view FlowEdit as solving an ODE rather than an optimization problem.

max

Point (i) above is a major distinction between FlowEdit and DDS/DPS, as it guarantees that the model is always fed with inputs from the distribution on which it has been trained. When sampling timesteps at random, as in DDS and DPS, there is a possibility of drawing a small timestep t at the beginning of the process, when the image has not yet been modified. This leads to out-of-distribution inputs to the model. For example, in the cat→lion edit of Fig. S13, if DDS draws a small t at the beginning of the process, then the model is fed with a cat image that is only slightly noisy (where the cat is clearly visible), and is tasked with denoising it conditioned on the text “lion”. Clearly, a slightly noisy cat image is out-of-distribution under the condition “lion”.

However, it turns out that there is an even more fundamental issue with the optimization viewpoint. Specifically, we claim that the optimization viewpoint is not fully justified even for DDS, as the loss that it attempts to minimize does not actually decrease throughout the DDS iterations. We thoroughly evaluate this in the next subsection.

###### L.1. Delta denoising loss

The DDS method was presented as an iterative approach for minimizing the delta denoising (DD) loss. The DD loss for matched and unmatched image-text embedding pairs z,y,ˆz,yˆ respectively, is defined as

LDD(ϕ,z,y,ˆz,y,ϵˆ ) =

1

∥ϵω(zt,y,t) − ϵω(ˆzt,y,tˆ )∥2dt, (S1)

0

where t is the diffusion timestep and ϵω(·) is the trained (noise-predicting) model with guidance scale ω [18]. The DDS iterations constitute an approximation for a stochastic gradient descent process. However, as we illustrate in Fig. S15, this approximation is quite poor. Specifically, as can be seen for two editing examples, the DDS iterations do not decrease the DD loss. They rather tend to increase it. Furthermore, as illustrated in Fig. S15, if the optimization is allowed to continue, the editing results deteriorate.

Tiger → Lion

Mountain → Volcano

| |
|---|

| |
|---|

when continuing the optimization

when continuing the optimization

when DDS stops

when DDS stops

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

###### Figure S15. DDS optimization process.

These experiments were carried out using the official DDS implementation1, with SD 1.5 [43]. We allowed the DDS optimization process to continue beyond the default 200 iterations used in the official implementation by running the optimization process an additional 1800 iteration, to a total of 2000 iterations. Every 100 iterations we calculated the DD loss ((S1)) for 19 timesteps between 50 and 950 with spacing of 50. We then summed the loss for all these timesteps to obtain the loss vs. iteration graphs in Fig. S15.

We do not compare FlowEdit with DDS results directly, as the comparison will not be fair due to the different backbone models, but we rather shed light on this strange behavior of the DDS optimization process.

###### L.2. FlowEdit with different step sizes

When using FlowEdit with a step size that is either smaller or larger than dt = ti−1 − ti, the results deteriorate, as illustrated in Fig. S16, qualitatively and quantitatively. We evaluate FlowEdit using SD3 with step-size, dt, scaled by a factor of c. Only for c = 1 we get a favorable balance between the CLIP and LPIPS metrics, i.e. a high CLIP score and a low LPIPS score. Furthermore, we see that even slight deviations from c = 1 lead to a significant drop in performance. This can also be clearly seen in the qualitative example, where the edited result adheres to the text prompt and is free of artifacts only for c = 1.

This behavior indicates that FlowEdit cannot be considered as a gradient descent (GD) optimization over a loss function. In GD, step sizes with similar values yield similar results, and the optimization dynamics are generally smooth, without abrupt changes. This is different from FlowEdit results illustrated in Fig. S16, where a big difference in the results occurs with c values around 1. In fact, as FlowEdit solves an ODE that traverses a path between the source distribution and the target distribution, the step sizes along this path need to sum to 1. Arbitrarily scaling the step-size by some constant violates this requirement and therefore leads to deteriorated results.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

[Figure 397]

### Cat ↓ Lion

𝑐 = 0.5 𝑐 = 0.75 𝑐 = 1 𝑐 = 1.25 𝑐 = 1.5

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

Figure S16. FlowEdit results with a scaled step size. The scale parameter c that multiplies the step-size is indicated next to each point.

1https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb

- L.3. Mathematical discrepancies between the FlowEdit, DDS, PDS update steps Regardless of the key differences in the ODE vs. optimization viewpoints, here we focus on the difference between the update

steps themselves, translated to a common diffusion denoising formulation. To do so, we rewrite our ODE step Vt∆ in terms of differences in noise predictions. Specifically, assuming Xt = (1 − t)X0 + tϵt, we will express the velocity field, which is given by V (xt,t) = E[ϵt − X0|Xt = xt] (see [31]) in terms of the noise prediction ϵˆ(xt,t) ≜ E[ϵ|xt = xt]. We have

V (xt,t) = E[ϵ − X0|Xt = xt]

1 1 − t

E[(1 − t)ϵ − (1 − t)X0|Xt = xt]

=

1 1 − t

E[ϵ − ((1 − t)X0 + tϵ)|Xt = xt]

=

1 1 − t

(E[ϵ|Xt = xt] − xt)

=

1 1 − t

(ˆϵ(xt,t) − xt). (S2) Therefore,

=

Vt∆(Zˆtsrc,Zˆttar) = V tar(Zˆttar,t) − V src(Zˆtsrc,t)

1 1 − t

1 1 − t

ϵ ˆ(Zˆttar,t) − Zˆttar −

ϵ ˆ(Zˆtsrc,t) − Zˆtsrc

=

1 1 − t

ϵ ˆ(Zˆttar,t) − ϵˆ(Zˆtsrc,t) − Z ˆttar − Zˆtsrc

=

1 1 − t

ϵ ˆ(Zˆttar,t) − ϵˆ(Zˆtsrc,t) − ZtFE − tX0src + tϵt − (1 − t)X0src − tϵt

=

1 1 − t

ϵ ˆ(Zˆttar,t) − ϵˆ(Zˆtsrc,t) − ZtFE − X0src , (S3)

=

where the fourth transition comes from our construction of Zˆtsrc = (1 − t)Z0src + tϵt and Zˆttar = ZtFE + Zˆtsrc − Z0src which we discussed in the main text.

As we discussed earlier, the update step in DDS is η(t)(ˆϵ(Zttar,t) − ϵˆ(Ztsrc,t), and is different from our update step, since it only relies on the differences between the noise predictions.

Regarding PDS[25], the proposed update step in Eq. (14) in their paper is of the form

###### ∇LPDS = ψ(t)(ZtPDS − X0src) + χ(t)(ˆϵ(Zttar,t) − ϵˆ(Ztsrc,t), (S4)

where ZtPDS is the variable that is being optimized, and ψ(t),χ(t) are diffusion-dependent coefficients. Note that their update step can only coincide with ours (up to a scalar multiplier) if ∀t : ψ(t) = −χ(t), which is not the case in their paper. Moreover, the authors of PDS state that their optimization scheme works when t is selected at random, and if chosen sequentially (mimicking FlowEdit) the gradients zero out (see the paragraph after Eq. (29) in PSD supplementary material). This further exacerbates the difference between the methods.

We conclude that even though DDS and PDS appear similar, when reformulating our update step in diffusion terms, the mathematical differences become clearer. This is added to the fact that both DDS and PDS attempt to solve an optimization problem (minimize differences between two vectors), which as we showed earlier is not fully justified.

##### M. Additional details about the Cats-Dogs experiment

In Sec. 5.3 we described experiments evaluating the reduced transport cost of FlowEdit compared to editing by inversion starting with generating a synthetic dataset of cat images. To generate the 1000 cat images we used variations of the prompt “a photo of cat” generated by Llama3 [11] 8B instruction model. Specifically, we requested 1000 source prompts describing cat images by providing the instruction: “Output 1000 possible prompts to a text-to-image model. These prompts should be a variation of ‘a photo of a cat.’, by adding descriptions and adjectives.” We used these 1000 prompts as input to SD3 and generated with them 1000 cats images. Examples of these generations can be seen in the upper part of Fig. S17, covering the blue shape.

To create corresponding edited dog images, we applied both editing by inversion and FlowEdit using target prompts identical to the source prompt, except for replacing the word “cat” with “dog”. Examples of these editing results can be seen in the middle of Fig. S17, covering the orange shape for FlowEdit results and covering the yellow shape for editing by inversion. Specifically, the four dog images in the middle of each shape are the results of editing the four cat images in the middle of the blue shape. It can be seen that FlowEdit editing are better when compare to editing by inversion results. We also calculate the transport cost between the cats distribution and both dogs distributions. We did it by calculating the average squared distance between SD3 latent representation of the cat images and their paired dog images, for both edits. We also calculated LPIPS between the original cat images and their paired dog images. As presented in Sec 5.3, in both metrics, FlowEdit achieved lower transport cost, compared to editing by inversion.

In addition, we generated 1000 dog images using SD3 and the same text prompts used to generate the cat images, but with “cat” replaced by “dogs”. Examples of these generations can be seen in the bottom part of Fig. S17, covering the green shape. To asses the alignment between the edited dogs distribution and the generated dogs distributions we used FID and KID scores. As shown in Sec 5.3, FlowEdit achieved lower FID and lower KID scores, indicating our ability to produce images from the target distribution.

Generated cats distribution

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

FlowEdit

Editing by inversion

dogs distribution

dogs distribution

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

Perceptual similarity

Generated dogs

distribution

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

###### Figure S17. Illustration of the Cats-Dogs experiment.

