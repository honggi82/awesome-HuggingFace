# Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model

Jing He, Haodong Li*, Mingzhi Sheng*, Ying-Cong Chen

## arXiv:2512.01030v3[cs.CV]18May2026

[Figure 1]

[Figure 2]

[Figure 3]

Input Image MoGe-2

Lotus-2 (Ours)

[Figure 4]

[Figure 5]

[Figure 6]

Input Image MoGe-2

Lotus-2 (Ours)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Input Image MoGe-2

Lotus-2 (Ours)

Input Image MoGe-2

Lotus-2 (Ours)

Fig. 1: We present Lotus-2, a two-stage deterministic framework for monocular geometric dense prediction. Our method leverages pre-trained generative model as a deterministic world prior to achieve new state-of-the-art accuracy while requiring remarkably minimal data (trained on only 0.66% of the samples used by MoGe-2 [1]). The decoupled, two-stage design ensures both structurally correct inference and high-fidelity detail refinement. This figure demonstrates Lotus-2’s robust zeroshot generalization with sharp geometric details, especially in challenging cases like oil paintings and transparent objects.

Abstract—Recovering pixel-wise geometric properties from a single image is fundamentally ill-posed due to appearance ambiguity and non-injective mappings between 2D observations and 3D structures. While discriminative regression models achieve strong performance through large-scale supervision, their success is bounded by the scale, quality, and diversity of available data, as well as by limited physical reasoning. Recent diffusion models exhibit powerful world priors that encode geometry and semantics learned from massive image–text data, yet directly reusing their stochastic generative formulation is suboptimal for deterministic geometric inference: the former is optimized for diverse and high-fidelity image generation, whereas the latter requires stable and accurate predictions. In this work, we propose Lotus-2, a two-stage deterministic framework for stable, accurate and fine-grained geometric dense prediction, aiming to provide an optimal adaptation protocol to fully exploit the pre-trained generative priors. Specifically, in the first stage, the core predictor employs a single-step deterministic formulation with a clean-data objective and a lightweight local continuity module (LCM) to generate globally coherent structures without grid artifacts. In the second stage, the detail sharpener performs a constrained multi-step rectified-flow refinement within the manifold defined by the core predictor, enhancing fine-grained geometry through noise-free deterministic flow matching. Using only 59K training samples—less than 1% of existing largescale datasets—Lotus-2 establishes new state-of-the-art results in monocular depth estimation and highly competitive surface normal prediction. These results demonstrate that diffusion models can serve as deterministic world priors, enabling highquality geometric reasoning beyond traditional discriminative and generative paradigms. Project page: lotus-2.github.io.

Index Terms—Dense prediction, depth estimation, normal estimation, diffusion model, rectified flow, image generation.

I. INTRODUCTION

Geometric dense prediction aims to recover pixel-wise geometric or physical properties, such as depth, surface normal, or albedo, from a single image. This problem lies at the foundation of modern visual understanding and serves as a cornerstone for various downstream applications, including controllable image generation [2], [3], 3D/4D reconstruction [4]–[6], and autonomous driving [7]–[9]. The mapping from image appearance to underlying geometry is inherently ill-posed: a single image can correspond to multiple plausible 3D interpretations. Consequently, a model must infer a physically plausible and globally coherent structure beyond what is directly observable from appearance.

Traditional approaches have long attempted to solve this problem through either geometric reasoning or discriminative learning. Early multi-view geometry and photometric consistency methods rely on strong assumptions about scene structure, lighting, and reflectance, making them unsuitable for single-view and complex real-world scenarios. With the rise of deep learning, discriminative models [1], [10]–[16] have become the dominant paradigm by directly regressing geometric quantities from single images. While such models have achieved remarkable progress through increasingly powerful architectures and large-scale training, their performance

Work done at the Hong Kong University of Science and Technology (Guangzhou). * denotes equal contribution. The authors’ e-mail addresses are: {jhe812, hli736, msheng758}@connect.hkust-gz.edu.cn, yingcongchen@ust.hk. Corresponding Author: Ying-Cong Chen.

remains fundamentally constrained by the scale, quality and diversity of available data. Human perception leverages strong world priors to resolve the ambiguity of geometric dense prediction, however, discriminative models trained on limited data distributions lack such mechanisms. Consequently, they perform poorly in rare and challenging scenes, involving transparency, reflection, and low texture, where inference requires reasoning beyond observable appearance. Even recent large-scale efforts—such as MoGe [1], [15] and DepthAnything [13], [14], trained on millions of samples—still rely heavily on distributional coverage rather than true scene understanding from world modeling, see Fig. 1 for reference.

The emergence of diffusion models such as Stable Diffusion [17] and FLUX [18] has revealed a new paradigm for visual reasoning. Trained on billions of diverse image–text pairs (e.g., LAION-5B [19]), these models exhibit remarkable capability in synthesizing geometrically coherent and physically consistent imagery across diverse scenes. This success suggests that diffusion backbones implicitly encode world priors—rich internal representations of geometry and semantics accumulated through large-scale generative training.

With this intuition, recent works have attempted to repurpose the world priors for dense prediction [20]–[26]. While these studies validate the promise of generative world priors, most of them directly adopt the original generative formulation of diffusion models without rethinking its suitability for dense prediction. For example, Marigold [21] fine-tunes Stable Diffusion by reformulating depth estimation as an image-conditioned depth generation problem. Although this design benefits from the pre-trained priors, it overlooks the fundamental difference between dense prediction and image generation: the former requires deterministic and accurate inference, whereas the latter optimizes for diverse and highfidelity generation through stochastic multi-step sampling. This fundamental mismatch often results in inconsistent and inaccurate geometric structure. Post-processing (e.g., test-time ensembling [21], [25]) doesn’t solve it in a native manner, and needs repeated predictions and may produce blurry results.

Motivated by these limitations, we revisit the role of diffusion-based generative models in dense prediction and propose a new perspective: their true value lies not in the generative mechanism itself, but in the world priors encoded within their pre-trained weights. Instead of treating diffusion as a stochastic generator, we view it as a structured world prior that can guide the inference towards deterministic and geometrically accurate dense prediction. Based on this insight, we introduce Lotus-2, a two-stage deterministic framework that decouples accurate global geometry prediction from meticulous detail sculpting, effectively combining the strengths of regression and generative expressiveness.

In the first stage, a core predictor extracts globally coherent and accurate geometry through a simple yet effective adaptation of the rectified-flow formulation in FLUX [18]. By systematically analyzing the key designs of stochastic generative formulation, including the stochasticity, multi-step sampling and parameterization type, we identify that a singlestep deterministic formulation under a clean-data prediction yields much more stable and accurate results than the original

stochastic multi-step residual-based design. This single-step predictor is further enhanced with a lightweight local continuity module (LCM), which mitigates grid artifacts introduced by the non-parametric Pack–Unpack operations in FLUX while maintaining architectural compatibility and efficiency.

In the second stage, an optional detail sharpener performs a detail refinement through a deterministic multi-step rectifiedflow process. It operates within the constrained manifold defined by the core predictor and learns the transition from the “accurate” to “accurate and fine-grained” annotation, progressively enriching geometric details while preserving global structure and accuracy. This design bridges the gap between regression and generative modeling: the former ensures structural stability and correctness, while the latter contributes finegrained realism. Consequently, Lotus-2 effectively leverages the generative priors in a disciplined and interpretable manner, achieving both geometric consistency and high-frequency detail fidelity without sacrificing efficiency and stability.

In summary, our key contributions are:

- • Revisiting the role of diffusion models for dense prediction. We reformulate diffusion-based generative models from stochastic image generators to structured world priors, emphasizing that their strength lies in the world modeling capability embedded within pre-trained weights rather than in the sampling trajectory itself.
- • A two-stage deterministic framework integrating the strengths of regression and generative refinement. We propose Lotus-2, which decouples structure prediction and detail refinement: a core predictor performs singlestep, clean-data regression for accurate and stable geometric estimation, while an optional detail sharpener applies multi-step rectified-flow refinement within the constrained manifold defined by the predictor.
- • A principled adaptation of the rectified-flow formulation. Through systematic analysis of several key designs in the original stochastic generative formulation, including stochasticity, multi-step sampling, parameterization type and local continuity, we demonstrate that the single-step clean-data deterministic design achieves higher accuracy and better optimization stability than the traditional formulation optimized for image generation.
- • State-of-the-art performance. With only 59K training samples—merely 0.66% of the data used by MoGe [1], [15] and 0.09% of that used by DepthAnything [13], [14], Lotus-2 achieves new state-of-the-art results on monocular depth estimation and highly competitive results on normal estimation.

II. RELATED WORK

- A. Traditional Paradigms for Geometric Dense Prediction

Recovering geometric properties, specifically monocular depth estimation and surface normal prediction, has been a central pursuit in computer vision. Traditional efforts to address the inherent ill-posed nature of this task have generally been categorized into two major paradigms: (1) physicsbased geometric reasoning and (2) data-driven discriminative learning.

Early physics-based geometric reasoning methods focus on leveraging established geometric and photometric constraints, such as structure from motion (SfM) [27], [28], photometric stereo [29], and algorithms based on multi-view geometry [30], [31]. They rely on a set of strong assumptions about the scene. For instance, they often require multiple views of the scene, precise camera calibration, or strict adherence to the Lambertian reflectance model. While theoretically sound under constrained or ideal conditions, these dependencies render them brittle and highly impractical for single-view geometric dense prediction in unconstrained, real-world environments.

With the advent of deep learning, the field shifts toward the discriminative learning paradigm. Models like the pioneering works [10], [12], [32], [33] and the recent large-scale stateof-the-art efforts, such as MoGe [1], [15] and Depth Anything [13], [14], have achieved remarkable empirical performance by directly regressing geometric quantities from input images. These successes are primarily attributed to increasingly powerful architectures (like Vision Transformers [34]) and, more importantly, supervision at massive scales.

However, these discriminative models face two fundamental limitations rooted in their data-driven nature. First, despite the scale of modern datasets, the quantity, quality, and diversity of geometric ground truth data remain fundamentally constrained (only million-scale compared to the billion-scale data used for pre-training large-scale generative models). Second, these models rely heavily on distributional coverage—memorizing patterns across the training data—rather than learning the intrinsic physical laws that govern scene structure. Consequently, they struggle severely with out-of-distribution (OOD) scenarios, including highly reflective surfaces, transparent objects, or rare scene compositions, where inference requires true geometric reasoning that transcends memorized patterns. These limitations motivate us to explore the powerful world priors embedded within pre-trained large-scale generative models. The world priors offer a superior foundation because: (1) they have been exposed to vast amounts of high-quality data; and (2) they possess world intrinsic knowledge of geometry, semantics, and physical structure accumulated through largescale generative training.

B. World Priors from Generative Models

The dependence of data-driven discriminative models on finite supervised data underscores the necessity of a superior source of structural knowledge, divorced from expensive geometric annotation. This required “world prior” has been implicitly encoded within the weights of large-scale generative models.

Unlike early generative approaches such as VAEs [35], [36] and GANs [37]–[41], and even initial diffusion models [42]– [47], which are often trained on restricted domain-specific data and thus contain limited world knowledge, recent advancements focus on large-scale training. By leveraging billions of diverse image-text pairs (e.g., LAION-5B [19]), modern diffusion models have acquired an extraordinary capacity to synthesize geometrically coherent and physically consistent imagery across diverse scenes. Crucially, the sheer volume

of the training data significantly surpasses the quantity of all available dense geometric annotation datasets [48]–[52]. This success implies that the diffusion backbone implicitly encodes powerful world priors—rich internal representations of geometry, semantics, and physical structure—thereby offering a new paradigm for visual reasoning.

The landscape of these pre-trained large-scale diffusion models has rapidly evolved: StabilityAI’s release of Stable Diffusion 1.x and 2.x [17], [53], based on the DDPM [42] training paradigm and a UNet [54] structure, initially revolutionizes the field. Subsequent efforts focused on efficiency and quality, such as Playground’s aesthetic enhancement efforts [55] and PixArt-α’s exploration [56] of the DiT [57] structure for computational efficiency. More recently, the emergence of the rectified-flow [58] and flow-matching [59] formulations, explored in models like Stable Diffusion 3.x [60], AuraFlow [61], and significantly, FLUX [18], represents the latest technological frontier. FLUX built upon the DiT architecture and the rectified-flow formulation, achieves the highest aesthetic quality through meticulous training and data preparation, leading to exceptionally natural, realistic, and geometrically consistent visual synthesis. Given the visual quality and superior physical consistency, the pre-trained FLUX model is the optimal choice as the world prior for our geometric dense prediction.

C. Repurposing Generative Priors for Dense Prediction

Building upon the insight that pre-trained generative weights encode crucial world priors, the community has explored various strategies to adapt this knowledge for geometric dense prediction. These methods can be broadly categorized into three distinct technical trajectories. The most dominant group follows the “stochastic generative formulation”, retaining the original multi-step diffusion pipeline. This includes works like Marigold [21], GeoWizard [25], DepthFM [62] and recent DICEPTION [26]. While these models validate the necessity of world knowledge from generative priors, their adherence to “stochastic multi-step sampling” leads to fundamental performance limitations: poor inference efficiency and unacceptable structural variance due to their non-deterministic nature. All of these methods rely on random noise, and different noises result in diverse geometric structure. This diversity is desirable in image generation, however, it results in inconsistent and physically implausible geometric structures in dense geometric prediction. A second group focuses on accelerating the inference speed. Works like Diffusion-E2E-FT [22], [63] directly fine-tune the generative backbone as a deterministic feedforward model to achieve fast, stable results. However, this single-step strategy often struggles to produce the fine-grained geometric details, which is crucial for high fidelity. The third group attempts a coarse-to-fine strategy, exemplified by StableNormal [64], which uses a two-stage approach combining initial prediction with subsequent refinement. However, its second stage still relies on the stochastic generative formulation, compromising the inherent need for high stability in geometric inference. In contrast to all these approaches, our proposed Lotus-2 employs a purely deterministic and noisefree rectified-flow strategy for both stages of prediction. By

utilizing the superior FLUX backbone and decoupling the inference into structure prediction (core predictor) and detail refinement (detail sharpener), we overcome the limitations of stochasticity, efficiency, and detail loss, positioning Lotus-2 as the premier solution for physically consistent and fine-grained geometric reasoning.

III. PRELIMINARIES

Our Lotus-2 framework is founded on the mathematical formalism of rectified-flow and the architectural foundation of FLUX model. This section introduces the necessary technical background related to our methodology.

A. Rectified-Flow Formulation

The rectified-flow (RF) formulation [58], [59] provides a robust and deterministic framework for modeling the transformation between two arbitrary probability measures via an ordinary differential equation (ODE). Specifically, given a source distribution p1 and a target distribution p0, the ODE on time-step t ∈ [0,1] is defined as: dzt = v(zt,t)dt, which maps z1 ∼ p1 to z0 ∼ p0 under the velocity vector field v(zt,t). Crucially, the core principle of RF is to transport samples along the straight-line path:

zt = tz1 + (1 − t)z0, (1) thus the target vector field v is given by v = dz

dt = z1 − z0. This straight-line mechanism fundamentally differs from the high-curvature paths of denoising diffusion models [42], which ensures a high efficiency and reduced error accumulation. For training, the velocity vector field v(zt,t) is parameterized by a neural network fθ, which is optimized by minimizing the distance to the target vector field v. The loss function is thus defined as:

t

Lt = ||v − fθ(zt,t)||2 (2) = ||(z1 − z0) − fθ(zt,t)||2. (3)

In practice, the expectation over the continuous time t ∈ [0,1] is approximated by randomly sampling a discrete time-step value from a pre-defined set at each training iteration. Given a total of T training time-steps, the pre-defined time-step set is:

i T | i = 1,2,...,T}. (4)

{ti =

During sampling (inference), the discrete Euler solver is used to iteratively generate the target sample (t = 0) from the source (t = 1). Formally, the iterative sampling process from current state zt

is given by: zt

to next state zt

curr

next

,t), (5)

curr − η · fθ(zt

= zt

next

curr

where tnext < tcurr and η (0 < η ≤ 1) denotes the step size, which is determined by the total number of inference timesteps Tinf.

Multi-step

[Figure 13]

[Figure 14]

𝑡 ∈ [0,1]

Predicting velocity (noise→annotation)

Annotation y

𝛆

[Figure 15]

🔥

[Figure 16]

[Figure 17]

❄

𝐿 = 𝐯 − 𝑓 𝐳𝐭,𝐳𝐱,𝑡 , 𝐯 = 𝛆 − 𝐳𝐲

Transformer 𝑓

| | | | | | |
|---|---|---|---|---|---|
| | |[Figure 18]<br><br>[Figure 19]| | | |
| | | | | | |
| | | | | | |

[Figure 20]

|Eq. (9)|
|---|

ℰ

𝐳𝐲

𝐳𝐭

[Figure 21]

[Figure 22]

Image x

𝐳𝐱

- Fig. 2: Adaptation protocol of stochastic formulation (Stochastic-DA). This framework models a conditional generative flow by estimating the velocity field from a random noise latent ϵ to the annotation latent zy, conditioned on the image latent

- zx. The target velocity vector is v = ϵ−zy. This reliance on noise initialization inherently leads to non-deterministic variance in deterministic geometric prediction.

IV. LOTUS-2

- B. Architectural Foundation of FLUX

We leverage the architecture and weights of FLUX [18], which utilizes a pre-trained variational autoencoder (VAE) to compress high-dimensional image data x into a compact latent space Z. The VAE consists of an encoder E and a decoder D, where E(x) = zx maps the image to a latent code, and D(zx) = ˆx attempts to reconstruct the image from the latent code. The rectified-flow formulation of FLUX operates within this VAE latent space Z.

In this section, we present Lotus-2, a two-stage deterministic framework for stable, accurate and high-fidelity dense prediction, aiming to provide an optimal adaptation protocol to effectively and efficiently leverage the pre-trained world priors of FLUX [18]. We argue that directly inheriting the stochastic generative formulation—which is optimized for image synthesis—introduces instability and unnecessary complexity for deterministic geometric tasks. The image synthesis aims at diverse and high-fidelity generation through stochastic multi-step sampling, while the dense prediction requires a deterministic and accurate inference. This fundamental misalignment results in high structural variance and significant prediction errors for dense prediction, thereby compromising overall accuracy. To better exploit the generative world priors, we propose a decoupled, two-stage adaptation protocol. We first introduce the Core Predictor (Sec. IV-A) derived through a systematic analysis of the standard generative formulation, including its stochasticity (Sec. IV-A1), multi-step iterative sampling (Sec. IV-A2), parameterization type (Sec. IV-A3), and local continuity (Sec. IV-A4). This core predictor is dedicated solely to achieving highly accurate and robust global geometry estimation. Subsequently, we address the challenge of fine-grained fidelity by proposing the Detail Sharpener (Sec. IV-B), which employs a constrained multistep rectified-flow formulation designed only for meticulous detail sculpting within the established structural manifold. This decoupled, two-stage approach successfully achieves both structural accuracy and fine-grained fidelity, with its complete inference process detailed in Sec. IV-C.

In the specific task of image generation, the starting distribution p1 is set to standard Gaussian noise in the latent space, i.e., z1 ∼ N(0,I). The target distribution p0 is the distribution of real, clean image latent, i.e., z0 = E(x) = zx. Based on this setup, the loss function in Eq. 2 is rewritten by:

#### Lt = ||(ϵ − zx) − fθ(zt,t)||2. (6)

Here, zt = tϵ + (1 − t)zx is the linear interpolation between the noise and the target latent code. FLUX adopts the DiT (Diffusion Transformer) [57] architecture as its model fθ.

The Pack-Unpack Operations in FLUX. To reduce computational overhead and memory usage, FLUX applies paired Pack and Unpack operations around the DiT model in the latent space. Pack is a parameter-free down-sampling procedure that rearranges the latent feature by grouping every nonoverlapping 2 × 2 patch into the channel dimension,

H

2 ×W2 ×4C. (7)

Pack : RH×W×C → R

Conversely, Unpack restores the original resolution by inverting this rearrangement,

H 2 ×

W

2 ×4C → RH×W×C. (8)

Unpack : R

A. Core Predictor: Robust and Accurate Geometric Prediction 1) Analysis-1: Stochastic vs. Deterministic Formulation: Initial efforts to leverage diffusion priors for geometric dense prediction (e.g., Marigold [21], GeoWizard [25]) inherit the model’s original stochastic generative formulation. We term

This Pack-Unpack operation, while efficient, introduces a critical challenge: because it is parameter-free, it can introduce noticeable local pixel discontinuities (“grid-artifacts”). This issue is especially severe under the single-step formulation, degrading the overall quality and realism of the outputs.

Image x

𝐳𝐱

Multi-step

[Figure 23]

[Figure 24]

❄

[Figure 25]

Predicting velocity (image→annotation)

𝑡 ∈ [0,1]

ℰ

[Figure 26]

🔥

[Figure 27]

𝐿 = 𝐯 − 𝑓 𝐳𝐭,𝑡 , 𝐯 = 𝐳𝐱 − 𝐳𝐲

Transformer 𝑓

[Figure 28]

[Figure 29]

[Figure 30]

|Eq. (11)|
|---|

𝐳𝐭

𝐳𝐲

Annotation y

- Fig. 3: Adaptation protocol of deterministic formulation (Deterministic-DA). This architecture shifts the paradigm to a noise-free rectified-flow formulation. It directly estimates the velocity field from the source image latent zx to the target annotation latent zy, where the target velocity vector is v = zx − zy. This deterministic setup ensures stability and structural consistency for geometric dense prediction.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Input Image

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- Stochastic, seed 1
- Stochastic, seed 2

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Deterministic, noise-free

Inference process: from 𝑡 = 1 to 𝑡 = 0

Average

[Figure 44]

- Fig. 4: Comparison between stochastic and deterministic formulation. The figure visualizes the iterative inference process from t = 1 to t = 0. The stochastic formulation (Stochastic-DA) exhibits significant structural variance: distinct random noise initializations yield inconsistent geometric structures across the entire inference process (highlighted in red circles). While averaging is employed to mitigate the variance, the final prediction remains compromised by the blending of conflicting structural hypotheses. In contrast, the deterministic formulation (Deterministic-DA) ensures a noisefree and stable trajectory, preventing structural variance and improving geometric coherence and prediction accuracy.

function for optimizing this stochastic generative formulation is given by:

#### Lt = ||(ϵ − zy) − fθ(zt,zx,t)||2. (10)

The core limitation of this approach is its inherent nondeterministic variance. Because the inference must begin with an initial sample of pure Gaussian noise, z1 = ϵ ∼ N(0,I), different random initializations lead to diverse outputs, resulting in inconsistent geometric structures for the same input image, as illustrated in Fig. 4. This variance is beneficial for diverse image generation; however, it leads to physically implausible geometric structures for dense prediction, thus hindering accuracy. While ensemble averaging is commonly used to mitigate this variance, it inherently introduces prediction bias and also compromises overall accuracy by blending both correct and incorrect structural hypotheses.

To resolve this fundamental mismatch, we discard the stochastic conditional generative formulation and shift the paradigm to a purely deterministic flow matching between two distributions. We formulate the problem as learning a noise-free transformation between the image feature zx and the geometric feature zy, directly utilizing the inherent determinism of the rectified-flow framework. We term this approach as Deterministic Direct Adaptation (Deterministic-DA) of the rectified-flow formulation. The architecture for this approach is illustrated in Fig. 3. Specifically, Deterministic-DA defines the two distributions as the image and annotation spaces, respectively: the source is the image latent zx and the target is the annotation latent zy. The latent feature at time t is defined as:

this approach as Stochastic Direct Adaptation (Stochastic-DA). In this setup, the process is framed as an image-conditioned geometric generation task: the model learns the flow from pure Gaussian noise ϵ ∼ N(0,I) to the target geometry zy conditioned on the input image zx as illustrated in Fig. 2. Specifically, the latent feature at time t is defined as:

#### zt = tzx + (1 − t)zy, (11)

where the model fθ is trained to predict the velocity v = zx − zy. The training objective for this deterministic flow is:

#### zt = tϵ + (1 − t)zy. (9)

#### Lt = ||(zx − zy) − fθ(zt,t)||2. (12)

This approach is inherently noise-free during both training and inference. As shown in Fig. 4 and Tab. III, this deterministic approach significantly improves structural consistency and prediction accuracy compared to its stochastic counterpart.

The neural network fθ is trained to predict the velocity field v = ϵ − zy by incorporating the image latent zx as a conditional input (typically concatenated along the channel dimension of the input feature to the DiT backbone). The loss

Image x

𝐳𝐱

Single-step

[Figure 45]

[Figure 46]

❄

[Figure 47]

𝑡 = 1

Predicting clean-data

[Figure 48]

🔥

ℰ

[Figure 49]

Transformer 𝑓 𝐿 = 𝐳𝐲 − Λ(𝑓 𝐳𝐭,𝑡 )

[Figure 50]

[Figure 51]

|Eq. (11)| |
|---|---|
| | |

LCM Λ

𝐳𝐭

𝐳𝐲

Annotation y

- Fig. 5: Adaptation protocol of the core predictor in Lotus-2. It adopts a single-step formulation (t = 1) with clean-data prediction to efficiently exploit the world priors of pre-trained FLUX model, where input latent zt is equivalent to the image latent zx, i.e, zt = z1 = zx, according to Eq. 11. In addition, since FLUX includes a pair of Pack-Unpack operations around the diffusion transformer fθ, we employ a local continuity module (LCM) Λ to mitigate grid artifacts caused by this Unpack operation.

5K 10K 19K 39K 59K Training Data

4

6

8

10

12

14

16

18

NYUv2AbsRel(%)

NYUv2 AbsRel vs. Training Data

- T=1

- T=2

T=5

T=10

T=100

T=1000

- Fig. 6: Comparisons among various training time-steps and data scales evaluated on NYUv2 in depth estimation. During inference, if the number of training time-steps T > 50, the inference time-steps are fixed at Tinf = 50; otherwise, Tinf = T. The results show that, when adapting the pretrained rectified-flow model to dense prediction, reducing the number of training time-steps leads to improved performance. In particular, the single-step formulation (T = 1) achieves the best performance across all data scales.

reducing the number of training time-steps T. This is achieved by modifying the value of T in Eq. 4 to define new, smaller time-step sets for training. The results clearly show that the performance gradually improves as the number of time-steps T is reduced, culminating in the best result when reduced to only a single step. Under a stricter setting with more limited training data, the multi-step formulation is more sensitive to variations in training data scale compared to the singlestep formulation. The single-step formulation demonstrates greater stability and yields lower prediction errors. While it is plausible that, given unlimited high-quality data, both multi- and single-step formulations could reach comparable performance, such a setting is often costly and impractical for dense prediction tasks.

Reducing the number of training time steps T constrains the optimization space of rectified-flow formulation, thereby enabling more effective and efficient adaptation for geometric dense prediction. Motivated by this observation, we adopt the single-step formulation (T = 1, i.e., t = 1 in Eq. 4). This single-step formulation further enhances computational efficiency.

3) Analysis-3, Parameterization Types: Under the singlestep formulation derived above, the model degenerates into a regression task, which is trained to predict the velocity given the input image with a fixed time-step t = 1. The velocity v = zx − zy is the residual between the input image zx and its annotation zy. We refer to this parameterization as Residual Prediction. During inference, the final prediction is obtained using the single-step Euler solver:

2) Analysis-2, Multi-Step Iterative Sampling: While the multi-step formulation enhances the capacity of generative models, it is optimized for high-fidelity image synthesis and demands large-scale training data. For dense geometric prediction, where high-quality supervision data is scarce, this inherited multi-step formulation is computationally intensive and makes the model difficult to optimize effectively. Furthermore, the prediction errors are accumulated during this multi-step iterative sampling, further compromising the accuracy. The iterative nature also hinders its practical application due to slow inference speeds.

ˆzy = zx − fθ(zx,t), (13) where fθ(zx,t) is the predicted residual.

However, such residual prediction is problematic for dense prediction tasks for two reasons: ① Predicting zx−zy requires the model to simultaneously learn image reconstruction and geometric estimation, which belong to substantially different distributions. This increases optimization difficulty and ultimately degrades accuracy; ② The predicted residual is dominated by high-frequency appearance signals of the input image, such as textures, illumination, and color. Although

To address these challenges, we propose fine-tuning the pretrained rectified-flow model with fewer training time-steps. As illustrated in Fig. 6, we conduct experiments by gradually

[Figure 52]

[Figure 53]

[Figure 54]

8

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Input Image Residual Prediction Clean-Data Prediction

Input Image 𝑤/𝑜 LCM 𝑤/𝑜 Pack-Unpack 𝒘/ LCM

- Fig. 7: Predictions under Different Model Parameterization Types. Red circles highlight regions with obvious appearance artifacts when residual prediction is used. In contrast, cleandata prediction produces more accurate predictions without interference from image appearance.

Fig. 8: The effects of different strategies for eliminating grid-like artifacts. “w/o LCM” refers to only single-step formulation with clean-data prediction, which produces noticeable grid-like artifacts due to the discontinuity introduced by Pack-Unpack. Removing Pack-Unpack entirely alleviates this issue but compromises both accuracy and efficiency. In contrast, LCM effectively resolves the artifacts while improving accuracy and preserving model efficiency. (Zoom in for clearer observation.)

the term zx in Eq. 13 attempts to remove these appearance components during inference, however, imperfect prediction makes this removal unreliable, and appearance interference inevitably leaks into the final result.

To overcome these limitations and better exploit pre-trained visual priors, we propose fine-tuning the model with CleanData Prediction, i.e., directly predicting the clean annotation

As shown in Fig .8, LCM effectively mitigates the local discontinuities introduced by Pack-Unpack, thereby eliminating grid artifacts. Furthermore, Tab. III demonstrates that LCM not only improves visual quality but also enhances prediction accuracy.

- zy. The clean-data prediction offers a simpler and more direct training objective, alleviates optimization difficulty, and eliminates appearance interference, thereby yielding superior performance.

For comparison, we additionally evaluate a straightforward alternative: entirely removing the Pack-Unpack operations from FLUX architecture. While the removal of Pack-Unpack does eliminate grid artifacts (see the “w/o Pack-Unpack” cases in Fig. 8), this approach suffers from two severe drawbacks: ① since the input–output dimensionality of the diffusion transformer changes, additional linear layers are required to align the dimensions, which shifts the feature space away from the pre-trained priors, degrading the prediction accuracy (see Tab. III); ② the absence of Pack-Unpack drastically compromises model efficiency, leading to much slower inference speed. Therefore, LCM offers an effective solution to the local discontinuity problem, while preserving the pre-trained priors and maintaining model efficiency.

As shown in Fig. 7, residual prediction produces predictions corrupted by image patterns (see red circles), whereas cleandata prediction yields accurate results without such interference. Consistently, Tab. III shows that clean-data prediction achieves significantly higher accuracy than the original residual prediction. Therefore, to mitigate appearance interference and improve prediction quality, we adopt clean-data prediction as the parameterization type.

4) Analysis-4, Local Continuity: The FLUX architecture employs non-parametric Pack and Unpack operations to reduce computational overhead in the latent space for image generation (Sec. III-B). While efficient, the non-parametric nature of the Unpack operation, which rearranges feature channels back to spatial resolution after the diffusion transformer model, introduces spatial discontinuities at the boundaries of the 2×2 latent patches. This localized discontinuity, which lacks constraints on local spatial coherence, is detrimental to geometric fidelity in the final output (Fig. 8, “w/o LCM”).

5) Finalized Architecture and Objective: The final core predictor is built upon the foundational Deterministic-DA and integrates all derived components: the single-step formulation, the clean-data prediction, and the local continuity module (LCM), as shown in Fig. 5. This comprehensive design transforms the unstable and iterative generative flow into a highly efficient and structurally robust formulation, optimizing for deterministic geometric dense prediction. The overall training objective is defined as:

To address this issue without compromising efficiency, we propose the lightweight Local Continuity Module (LCM) after the Unpack operation of diffusion transformer backbone, as shown in Fig. 5. LCM consists of two 3 × 3 convolutional layers with an intermediate GELU activation [65] to introduce nonlinearity, which is formally defined as:

Lt = ||zy − Λ(fθ(zt,t))||2, (15) where t = 1 and the input latent zt = z1 = zx. B. Detail Sharpener: High-Fidelity Geometric Refinement

ˆzy = Λ fθ(zt,t) , Λ(h) = ϕ2 ⊛ γ(ϕ1 ⊛ h), (14) where Λ(·) denotes the LCM, ⊛ is the convolution operator, ϕ1 and ϕ2 are convolutional kernels, and γ(·) is the GELU activation.

The single-step core predictor excels at predicting accurate and globally coherent structure, but often produces predictions

Coarse anno. 𝐲𝐜

𝐳𝐲𝐜

|[Figure 71]|
|---|

[Figure 72]

❄

| | | | |
|---|---|---|---|
| | | | |
| | | |[Figure 73]|

𝑡 ∈ [0,1]

ℰ

[Figure 74]

🔥

|[Figure 75]|
|---|

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 76]| | |

𝐿 = 𝐯 − 𝑔 𝐳𝐭,𝑡 , 𝐯 = 𝐳𝐲𝐜 − 𝐳𝐲𝐟 𝐳𝐲𝐟

[Figure 77]

Transformer 𝑔

|Eq. (16)| |
|---|---|
| | |

𝐳𝐭

Fine anno. 𝐲𝐟

- Fig. 9: The training pipeline of detail sharpener. Starting from a structurally correct but coarse annotation predicted by the core predictor, the detail sharpener learns the transition from coarse to fine-grained annotation via a constrained multi-step rectified-flow within the manifold defined by the core predictor.

[Figure 78]

Image x

ℰ

[Figure 79]

𝐳𝐱 𝐳 𝐲𝐜

Transformer 𝑓 LCM Λ

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | |[Figure 80]|

Transformer 𝑔

𝑡 ∈ [0, 1] × 𝑇

[Figure 81]

Prediction 𝐲

𝒟

Detail Sharpener

| | | | |
|---|---|---|---|
| | | | |
| | | |[Figure 82]|

𝐳 𝐲𝐟

Core Predictor

𝑡 = 1

- Fig. 10: The inference pipeline of Lotus-2. It is a decoupled, two-stage deterministic pipeline that bridges the regression and geometric refinement. First, the core predictor produces stable and structurally consistent prediction via single-step regression. The detail sharpener then employs a constrained multi-step rectified-flow formulation for iterative refinement without any

stochastic noise. The refinement uses Tinf′ ≤ 10 steps, adjustable based on the desired level of sharpness. This design ensures both structural consistency and fine-grained fidelity in minimal steps.

f. Thus, the training objective of detail sharpener is defined as:

that are coarse and blurry in high-frequency detail areas, lacking fine-grained fidelity (see “w/o Sharpener” cases of Fig. 11). This limitation stems from the inherent difficulty of the single-step formulation in resolving high-frequency details. In contrast, multi-step flow (e.g., Deterministic-DA) retains the complexity to model high-frequency dynamics and can produce sharper details; however, due to its optimization difficulty and the accumulation of high errors across multiple steps, it is prone to geometric hallucination (see the “Deterministic-DA” cases of Fig. 11), sacrificing structural correctness and overall accuracy.

### v = zy

−zy

c

) − gθ(zt,t)||2. (17) We set the number of training steps T′ = 10 to balance optimization and refinement. During inference, the number of inference steps Tinf′ can be flexibly chosen up to T′, depending on the desired level of sharpness.

Lt = ||(zy

### − zy

c

f

As shown in Fig. 11, the detail sharpener noticeably enhances the sharpness while successfully avoiding the structural hallucinations observed in Deterministic-DA. Tab. III further confirms that incorporating the detail sharpener does not compromise geometric accuracy established by the core predictor.

To simultaneously achieve accuracy and fine-grained fidelity, we introduce the Detail Sharpener, a constrained multistep rectified-flow model designed solely for geometric refinement within the manifold defined by the core predictor. Specifically, we first obtain a structurally correct but coarse prediction via the single-step core predictor, and then employ detail sharpener to progressively refine the high-frequency details. With this design, structural correctness is guaranteed by the core predictor, while the detail sharpener is solely responsible for enhancing sharpness.

C. Inference

Lotus-2 executes a two-stage deterministic inference pipeline, as illustrated in Fig. 10. The core predictor is dedicated to ensuring structural correctness and efficiency, while the detail sharpener is solely responsible for highfidelity refinement. Rooted in our philosophy of deterministic modeling, both the core predictor and the detail sharpener are noise-free, guaranteeing structural consistency and stability for deterministic geometric dense prediction. The complete inference process proceeds as follows:

As illustrated in Fig. 9, the detail sharpener is trained to learn a noise-free rectified-flow transformation from a coarse prediction zy

c to its high-fidelity ground-truth zy

f. The flow is defined between the two known geometric states:

- 1) The input image x is first encoded into the VAE latent space using the encoder E, yielding the image latent zx.
- 2) The image latent zx is passed through the core predic-

+ (1 − t)zy

### zt = tzy

. (16) The model gθ is fine-tuned from FLUX to predict the velocity

c

f

tor to generate the accurate but coarse prediction ˆzy

c.

[Figure 83]

- Fig. 11: Comparisons in Detail Sharpness. “w/o Sharpener” denotes predictions directly obtained by the core predictor, which suffer from blurry and coarse details. The “w/ Sharpener” cases demonstrate that the detail sharpener noticeably enhances the sharpness of fine-grained structures, particularly along boundaries, while avoiding the geometric hallucinations observed in Deterministic-DA, such as the misaligned chair backrest and stair railing.

space using the VAE decoder D to produce the final geometric prediction yˆ.

V. EXPERIMENTS

In this section, we systematically validate the design principles of Lotus-2: leveraging pre-trained generative priors as a stable and deterministic flow for structurally correct and high-fidelity geometric dense prediction. We first detail the experimental setup, then present a quantitative comparison against state-of-the-art methods, followed by comprehensive ablation studies validating our methodological contributions.

[Figure 84]

[Figure 85]

[Figure 86]

Input Image

| |
|---|

[Figure 87]

A. Experimental Settings

- 1) Implementation Details: We implement the proposed

Lotus-2, which includes both the core predictor and the detail sharpener, by fine-tuning the pre-trained FLUX model [18] without utilizing the text conditioning. Our design adapts the rectified-flow formulation by setting the core predictor to a single-step formulation (T = 1, t = 1) with clean-data prediction and the detail sharpener to a constrained multistep rectified-flow formulation (T′ = 10, with time-steps defined by Eq. 4) within the manifold defined by the core predictor. For optimization, we use the Adam optimizer with a learning rate of 1 × 10−4. All models are trained on 8 NVIDIA H100 GPUs (80G) with a total batch size of 64. To adapt the large-scale pre-trained architecture, we employ the parameter-efficient method LoRA [66], using a rank of 128 for depth estimation and 256 for normal estimation. For depth estimation, we operate in the disparity space, i.e., d = 1/d′, where d′ is the true depth. During inference, the core predictor directly predicts the coarse but structurally correct prediction in a single inference step, while the detail sharpener utilizes the Euler sampler with Tinf′ = T′ = 10 steps for refinement.

- 2) Training Datasets: A core demonstration of this work

[Figure 88]

[Figure 89]

[Figure 90]

Deterministic-DA

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

𝑤/𝑜 Sharpener

[Figure 95]

is the ability to achieve SoTA performance using extremely limited supervised data. Both depth and normal estimation tasks are trained solely on a small collection of synthetic data, totaling approximately 59K samples—a fraction of the millions used by large-scale discriminative models.

[Figure 96]

[Figure 97]

𝒘/ Sharpener

[Figure 98]

- • Hypersim [67]: A photorealistic synthetic dataset of 461 indoor scenes. We utilize the official training split, retaining approximately 39K samples after filtering. All samples are resized to 576 × 768.
- • Virtual KITTI (VKITTI) [68]: A synthetic street-scene dataset covering five urban scenes. We use four scenes, comprising about 20K samples, cropped to 352 × 1216.

To train the detail sharpener, we implement the methodology described in Sec. IV-B: we first generate coarse predictions (yc) on Hypersim and VKITTI using the trained core predictor, and then train the detail sharpener on the flow defined between these coarse predictions and the ground truth (yf).

This step guarantees global structural correctness and is performed with maximum efficiency (1 step).

- 3) The coarse prediction ˆzy

c is then fed into the detail sharpener to obtain the sharp and high-fidelity result ˆzy

f. This iterative refinement is achieved by the discrete Euler solver (Eq. 5). Note that this refinement is optional based on the desired level of sharpness.

- 4) The final refined latent ˆzy

3) Evaluation Datasets: We evaluate the generalization capability of Lotus-2 across five real-world datasets for depth estimation and four for surface normal prediction, none of which were seen during training.

f is decoded back to the pixel

- TABLE I: Quantitative comparison on zero-shot affine-invariant depth estimation between Lotus-2 and SoTA methods. The best and second best performances are highlighted. §indicates results re-evaluated by ourselves using the evaluation protocol of Marigold [21]. ⋆denotes the method relies on pre-trained text-to-image generative models. Our Lotus-2 achieves the best overall performance among all methods.

Training NYUv2 (Indoor) KITTI (Outdoor) ETH3D (Various) ScanNet (Indoor) DIODE (Various) Avg. Data↓ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ Rank

Method

DiverseDepth 320K 11.7 87.5 19.0 70.4 22.8 69.4 10.9 88.2 37.6 63.1 19.5 MiDaS 2M 11.1 88.5 23.6 63.0 18.4 75.2 12.1 84.6 33.2 71.5 18.7 LeRes 354K 9.0 91.6 14.9 78.4 17.1 77.7 9.1 91.7 27.1 76.6 15.7 Omnidata 12.2M 7.4 94.5 14.9 83.5 16.6 77.8 7.5 93.6 33.9 74.2 15.4 DPT 1.4M 9.8 90.3 10.0 90.1 7.8 94.6 8.2 93.4 18.2 75.8 12.5

GeoWizard⋆§ 280K 5.6 96.3 14.4 82.0 6.6 95.8 6.4 95.0 33.5 72.3 12.4 HDN 300K 6.9 94.8 11.5 86.7 12.1 83.3 8.0 93.9 24.6 78.0 12.2

GenPercept⋆§ 74K 5.6 96.0 13.0 84.2 7.0 95.6 6.2 96.1 35.7 75.6 11.5 Marigold(LCM)⋆§ 74K 6.1 95.8 9.8 91.8 6.8 95.6 6.9 94.6 30.7 77.5 10.5 MoGe-2§ 8.9M 3.6 98 11.8 89.2 16.6 81.5 3.5 98.2 39.3 70.0 10.4 Marigold⋆ 74K 5.5 96.4 9.9 91.6 6.5 95.9 6.4 95.2 30.8 77.3 9.2 DICEPTION⋆ 500K 7.2 93.9 7.5 94.5 5.3 96.7 7.5 93.8 24.3 74.1 9.2 DepthAnything V2 62.6M 4.5 97.9 7.4 94.6 13.1 86.5 4.2 97.8 26.5 73.4 7.3 Diffusion-E2E-FT⋆ 74K 5.4 96.5 9.6 92.1 6.4 95.9 5.8 96.5 30.3 77.6 7.1 Lotus-G⋆ 59K 5.4 96.8 8.5 92.2 5.9 97.0 5.9 95.7 22.9 72.9 7.1 DepthFM-ID⋆ 81.4K 5.5 96.3 8.9 91.3 5.8 96.2 6.3 95.4 21.2 80.0 6.9 MoGe§ 9M 3.6 97.9 7.3 95.2 8.4 93.0 3.5 98.4 36.3 71.2 6.9 DepthAnything 62.6M 4.3 98.1 7.6 94.7 12.7 88.2 4.3 98.1 26.0 75.9 6.2 Lotus-D⋆ 59K 5.1 97.2 8.1 93.1 6.1 97.0 5.5 96.5 22.8 73.8 6.0 Lotus-2⋆ 59K 4.1 97.6 6.7 94.5 4.6 98.1 4.2 97.6 22.1 75.2 3.6

- • Affine-Invariant Depth Estimation: We evaluate across diverse indoor (NYUv2 [69], ScanNet [70]), outdoor (KITTI [71]), and high-resolution mixed scenes (ETH3D [72], DIODE [73]).
- • Surface Normal Prediction: We use NYUv2, ScanNet, and iBims-1 [74] for real indoor evaluation, and Sintel [75] for highly dynamic synthetic outdoor scenes.

4) Metrics: We employ widely accepted metrics for both affine-invariant depth estimation and surface normal prediction.

- • Affine-Invariant Depth Estimation: Following standard protocols [21], [32], we firstly align predictions to ground truth via least-squares fitting before evaluation. We report two primary metrics: the absolute mean relative error (AbsRel), defined as M1 Mi=1 |ai − di|/di (lower is better); and the δ1 value, which is the proportion of pixels satisfying Max(ai/di,di/ai) < 1.25 (higher is better).

- • Surface Normal Prediction: Following [64], [76], we measure the mean angular error (lower is better) and the percentage of pixels with an angular error below 11.25◦ (higher is better).

For overall comparison, we report the Avg. Rank, which is the average ranking of each method across all datasets and metrics.

- A lower Avg. Rank signifies superior overall performance.
- B. Comparison with State-of-the-Art

We benchmark Lotus-2 against recent state-of-the-art methods in both affine-invariant monocular depth estimation and

surface normal prediction, including both large-scale discriminative models (e.g., DepthAnything [13], [14], MoGe [1], [15]) and generative prior adaptation methods (e.g., Marigold [21], GeoWizard [25]).

- 1) Affine-Invariant Depth Estimation: As presented in

Tab. I, Lotus-2 establishes a new state-of-the-art in affineinvariant monocular depth estimation across the five real-world datasets. Notably, Lotus-2 achieves the best Avg. Rank despite being trained on only 59K samples. This result decisively validates the power of leveraging large-scale generative models as deterministic world priors, allowing Lotus-2 to surpass massive data-trained discriminative methods.

- 2) Surface Normal Prediction: For surface normal predic-

tion, Lotus-2 demonstrates highly competitive performance (Tab. II), showcasing the effectiveness of our deterministic adaptation in capturing complex geometry. Crucially, as highlighted in Fig. 1, our deterministic adaptation of world priors ensures robust and structurally correct geometric prediction, enabling strong generalization even in challenging or rare scenes. This robust foundation, coupled with our noise-free multi-step refinement (detail sharpener), proves highly effective at capturing the high-frequency surface detail required for local geometry, significantly outperforming other SoTA approaches.

C. Ablation Studies

1) Ablation on the Core Predictor: The core predictor is the structural foundation of Lotus-2. We systematically validate its design in Tab. III by incrementally incorporating the

- TABLE II: Quantitative comparison on zero-shot surface normal estimation between Lotus-2 and SoTA methods. ‡refers the Marigold normal model as detailed in this link. §indicates results re-evaluated by us using the evaluation protocol of DSINE [76]. Our Lotus-2 demonstrates highly competitive quantitative performance, crucially delivering the robust and finegrained qualitative results as highlighted in Fig. 1.

Method

Training NYUv2 (Indoor) ScanNet (Indoor) iBims-1 (Indoor) Sintel (Outdoor) Avg. Data↓ mean↓ 11.25◦↑ mean↓ 11.25◦↑ mean↓ 11.25◦↑ mean↓ 11.25◦↑ Rank

OASIS 110K 29.2 23.8 32.8 15.4 32.6 23.5 43.1 7.0 13.5 Omnidata 12.2M 23.1 45.8 22.9 47.4 19.0 62.1 41.5 11.4 11.9 GeoWizard⋆§ 280K 18.9 50.7 17.4 53.8 19.3 63.0 40.3 12.3 10.4 StableNormal⋆§ 250K 18.6 53.5 17.1 57.4 18.2 65.0 36.7 14.1 8.4 GenPercept⋆§ 74K 18.2 56.3 17.7 58.3 18.2 64.0 37.6 16.2 8.3 EESNU 2.5M 16.2 58.6 - - 20.0 58.5 42.1 11.5 7.3 Omnidata V2 12.2M 17.2 55.5 16.2 60.2 18.2 63.9 40.5 14.7 8.1 Marigold⋆‡ 74K 20.9 50.5 21.3 45.6 18.5 64.7 - - 8.1 Lotus-G⋆ 59K 16.5 59.4 15.1 63.9 17.2 66.2 33.6 21.0 5.4 DSINE 160K 16.4 59.6 16.2 61.0 17.1 67.4 34.9 21.5 4.9 Diffusion-E2E-FT⋆§ 74K 16.5 60.4 14.7 66.1 16.1 69.7 33.5 22.3 3.4 Lotus-D⋆ 59K 16.2 59.8 14.7 64.0 17.1 66.4 32.3 22.4 3.4 Lotus-2⋆ 59K 16.9 59.0 14.2 66.8 15.4 70.4 30.3 27.6 2.9 MoGe-2§ 8.9M 14.7 62.3 12.8 68.4 14.7 70.4 29.3 24.8 1.1

- TABLE III: Ablation studies of the proposed Lotus-2. The second portion of the table contains the key components of the core predictor, sequentially demonstrating the performance gains conferred by each design. The final row validates the detail sharpener. The shaded row (w/o Pack-Unpack) is included as an auxiliary ablation to validate the effect of the local continuity module (LCM). The results below are evaluated in monocular depth estimation across four datasets.

NYUv2 (Indoor) KITTI (Outdoor) ETH3D (Various) ScanNet (Indoor)

Method

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑

Stochastic-DA 8.261 93.468 13.196 78.204 17.384 77.842 9.373 91.569 Deterministic-DA 7.812 94.262 10.212 89.900 10.766 94.762 8.488 92.897 + Single-Step Formulation 5.910 96.939 8.833 92.088 5.858 96.952 7.121 96.331 + Clean-Data Prediction 4.384 97.627 6.843 94.325 4.980 97.552 4.446 97.529 + Local Continuity Module 4.128 97.608 6.576 94.682 4.625 98.004 4.174 97.575

(w/o Pack-Unpack) 4.817 97.383 6.966 94.203 5.728 97.252 4.723 97.168 + Detail Sharpener 4.122 97.623 6.725 94.492 4.643 98.101 4.188 97.597

core contributions, showing consistent performance superiority across all four evaluation datasets.

We begin by validating the necessity of the deterministic formulation. Moving from the stochastic generative formulation (Stochastic-DA) to the noise-free deterministic formulation (Deterministic-DA) yields an immediate improvement in accuracy. This validates our core hypothesis that deterministic geometric prediction requires a stable flow (Sec. IV-A1). Next, adopting the single-step formulation (T = 1) also provides a significant performance increase, confirming the single-step mechanism is the optimal strategy for efficiently leveraging pre-trained world priors under limited data (Sec. IV-A2). Following this, switching to clean-data prediction from residual prediction consistently achieves higher structural accuracy. This confirms that its value lies in both eliminating highfrequency appearance interference (Fig. 7) and providing a more direct and effective optimization target (Sec. IV-A3). Finally, we validate the local continuity module (LCM). This lightweight module successfully eliminates grid artifacts

(Fig. 8) and provides the final accuracy boost. This contrasts with the “w/o Pack-Unpack” alternative, which compromises efficiency and degrades performance due to feature space misalignment (Sec. IV-A4).

2) Ablation on the Detail Sharpener: The detail sharpener is responsible for high-fidelity refinement via a constrained multi-step flow. This ablation validates the contribution of the detail sharpener to high-fidelity geometry through qualitative results, quantitative metrics, and spectral analysis.

As qualitatively demonstrated in Fig. 11, the detail sharpener achieves noticeable refinement in high-frequency areas. Quantitatively, the final line item in Tab. III shows that the multi-step flow of the detail sharpener maintains the near-optimal accuracy achieved by the core predictor. This preservation of accuracy confirms that the detail sharpener successfully operates on a decoupled objective—enhancing local fidelity—without compromising the structural accuracy established by the core predictor, thus validating the success of our two-stage design.

- Fig. 12: Spectral analysis of high-fidelity refinement. This plot compares the average log-power (y-axis) across spatial frequencies (x-axis) on NYUv2 dataset to validate the contribution of detail sharpener. The decay of the core predictor (w/o sharpener) curve confirms its coarse nature, while the Lotus-2 (w/ sharpener) curve shows recovery of highfrequency power.

To rigorously quantify the contribution of the detail sharpener to fine-grained fidelity, specifically its effect on highfrequency detail areas, we conduct a spectral analysis using the 1D radially averaged power spectrum as illustrated in Fig. 12. The results show that the prediction from the core predictor exhibits a clear decay in power at high frequencies, confirming its output is structurally correct but coarse. In contrast, both Deterministic-DA and our Lotus-2 retain significantly more high-frequency power, indicating successful detail refinement. This provides quantitative, signal-level evidence that the detail sharpener is essential for high-fidelity geometric prediction.

VI. CONCLUSION

In this work, we addressed the fundamental challenge of geometric dense prediction—the task’s ill-posed nature—by proposing a critical shift in how large-scale generative models are leveraged. We established the principle that for deterministic geometric inference, the power of diffusion backbones lies not in their stochastic sampling process but in their implicitly embedded deterministic world priors. Directly reusing the original stochastic generative flow proves suboptimal, leading to structural variance and unacceptable inconsistency in geometric outputs.

To fully exploit these priors in a disciplined and stable manner, we introduced Lotus-2, a novel two-stage deterministic framework that decouples the inference process into two specialized, noise-free rectified-flow mappings.

The first stage, the core predictor, is implemented for maximum structural accuracy and efficiency. Through systematic ablation, we validated the necessity of our derived design choices: the deterministic shift, the highly efficient single-step formulation (T = 1), and the clean-data prediction objective, which together transform the complex generative flow into a

robust geometric regressor. The lightweight local continuity module (LCM) further ensures fidelity by suppressing architectural artifacts without compromising efficiency.

The second stage, the detail sharpener, solves the final limitation of single-step regression—coarse high-frequency details. It performs a constrained multi-step refinement within the geometry manifold established by the core predictor. This process is inherently noise-free and is optimized to selectively enhance high-fidelity geometry without compromising the established global structural correctness, successfully validating the benefits of our decoupled design.

The experimental results decisively confirm our core hypothesis. By training on only 59K synthetic samples—less than 1% of existing large-scale datasets—Lotus-2 achieved new state-of-the-art performance in monocular depth estimation and demonstrated highly competitive results in surface normal prediction. This unprecedented data efficiency, combined with high inference stability and fine-grained fidelity, validates the efficacy of our deterministic adaptation protocol.

Ultimately, this work demonstrates that the vast knowledge accumulated by generative diffusion models can be repurposed to enable efficient, accurate, and physically consistent geometric reasoning, setting a new paradigm for structured prediction tasks beyond traditional discriminative and generative methods. This finding opens promising avenues for future research into extracting and utilizing structured knowledge from foundational generative models.

REFERENCES

- [1] R. Wang, S. Xu, Y. Dong, Y. Deng, J. Xiang, Z. Lv, G. Sun, X. Tong, and J. Yang, “Moge-2: Accurate monocular geometry with metric scale and sharp details,” arXiv preprint arXiv:2507.02546, 2025. 1, 2, 3, 11
- [2] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3836–3847. 2
- [3] L. Hu, “Animate anyone: Consistent and controllable image-to-video synthesis for character animation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8153–8163. 2
- [4] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024 conference papers, 2024, pp. 1–11. 2
- [5] X. Long, Y.-C. Guo, C. Lin, Y. Liu, Z. Dou, L. Liu, Y. Ma, S.-H. Zhang, M. Habermann, C. Theobalt et al., “Wonder3d: Single image to 3d using cross-domain diffusion,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9970–9980. 2
- [6] L. Jiang, J. Lin, K. Chen, W. Ge, X. Yang, Y. Jiang, Y. Lyu, X. Zheng, Y. Li, and Y. Chen, “Dimer: Disentangled mesh reconstruction model,” arXiv preprint arXiv:2504.17670, 2025. 2
- [7] Z. Li, Z. Yu, D. Austin, M. Fang, S. Lan, J. Kautz, and J. M. Alvarez, “Fb-occ: 3d occupancy prediction based on forward-backward view transformation,” arXiv preprint arXiv:2307.01492, 2023. 2
- [8] Z. Li, W. Wang, H. Li, E. Xie, C. Sima, T. Lu, Q. Yu, and J. Dai, “Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2
- [9] S. Gu, W. Yin, B. Jin, X. Guo, J. Wang, H. Li, Q. Zhang, and X. Long, “Dome: Taming diffusion model into high-fidelity controllable occupancy world model,” arXiv preprint arXiv:2410.10429, 2024. 2
- [10] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014. 2, 3
- [11] W. Yuan, X. Gu, Z. Dai, S. Zhu, and P. Tan, “Neural window fullyconnected crfs for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 3916–3925. 2

- [12] A. Eftekhar, A. Sax, J. Malik, and A. Zamir, “Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10786–10796. 2, 3
- [13] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 10371–10381. 2, 3, 11
- [14] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in Neural Information Processing Systems, vol. 37, pp. 21875–21911, 2024. 2, 3, 11
- [15] R. Wang, S. Xu, C. Dai, J. Xiang, Y. Deng, X. Tong, and J. Yang, “Moge: Unlocking accurate monocular geometry estimation for opendomain images with optimal training supervision,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5261–

5271. 2, 3, 11

- [16] H. Li, H. Lu, and Y.-C. Chen, “Bi-tta: Bidirectional test-time adapter for remote physiological measurement,” in European Conference on Computer Vision. Springer, 2024, pp. 356–374. 2
- [17] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695. 2, 4
- [18] BFL.ai. (2024, Aug.) Bfl.ai announces the flux.1 suite of models. [Online]. Available: https://bfl.ai/announcements/24-08-01-bfl 2, 4, 5, 10
- [19] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion5b: An open large-scale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25278–25294, 2022. 2, 3
- [20] H.-Y. Lee, H.-Y. Tseng, and M.-H. Yang, “Exploiting diffusion prior for generalizable dense prediction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7861–7871. 2
- [21] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9492–9502. 2, 4, 5, 11
- [22] J. He, H. Li, W. Yin, Y. Liang, L. Li, K. Zhou, H. Zhang, B. Liu, and Y.-C. Chen, “Lotus: Diffusion-based visual foundation model for highquality dense prediction,” arXiv preprint arXiv:2409.18124, 2024. 2, 4
- [23] H. Li, W. Zheng, J. He, Y. Liu, X. Lin, X. Yang, Y.-C. Chen, and C. Guo, “Da 2: Depth anything in any direction,” arXiv preprint arXiv:2509.26618, 2025. 2
- [24] J. Wang, C. Lin, C. Guan, L. Nie, J. He, H. Li, K. Liao, and Y. Zhao, “Jasmine: Harnessing diffusion prior for self-supervised depth estimation,” arXiv preprint arXiv:2503.15905, 2025. 2
- [25] X. Fu, W. Yin, M. Hu, K. Wang, Y. Ma, P. Tan, S. Shen, D. Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image, in European Conference on Computer Vision. Springer, 2024, pp. 241–258. 2, 4, 5, 11
- [26] C. Zhao, M. Liu, H. Zheng, M. Zhu, Z. Zhao, H. Chen, T. He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks, arXiv preprint, 2025. 2, 4
- [27] C. Tomasi and T. Kanade, “Shape and motion from image streams under orthography: a factorization method,” International journal of computer vision, vol. 9, no. 2, pp. 137–154, 1992. 3
- [28] N. Snavely, S. M. Seitz, and R. Szeliski, “Modeling the world from internet photo collections,” International journal of computer vision, vol. 80, no. 2, pp. 189–210, 2008. 3
- [29] R. J. Woodham, “Photometric method for determining surface orientation from multiple images,” Optical engineering, vol. 19, no. 1, pp. 139–144, 1980. 3
- [30] D. Scharstein and R. Szeliski, “A taxonomy and evaluation of dense two-frame stereo correspondence algorithms,” International journal of computer vision, vol. 47, no. 1, pp. 7–42, 2002. 3
- [31] R. Hartley and A. Zisserman, Multiple view geometry in computer vision. Cambridge university press, 2003. 3
- [32] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 3, pp. 1623–1637, 2020. 3, 11

- [33] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 12179–12188. 3
- [34] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020. 3
- [35] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30,

2017. 3

- [36] A. Razavi, A. Van den Oord, and O. Vinyals, “Generating diverse high-fidelity images with vq-vae-2,” Advances in neural information processing systems, vol. 32, 2019. 3
- [37] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014. 3
- [38] J. He, Y. Zhou, Q. Zhang, J. Peng, Y. Shen, X. Sun, C. Chen, and R. Ji, “Pixelfolder: An efficient progressive pixel synthesis network for image generation,” arXiv preprint arXiv:2204.00833, 2022. 3
- [39] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4401–

4410. 3

- [40] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Analyzing and improving the image quality of stylegan,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8110–8119. 3
- [41] T. Karras, M. Aittala, S. Laine, E. H¨ark¨onen, J. Hellsten, J. Lehtinen, and T. Aila, “Alias-free generative adversarial networks,” Advances in Neural Information Processing Systems, vol. 34, pp. 852–863, 2021. 3
- [42] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020. 3, 4
- [43] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020. 3
- [44] X.-L. Li, H. Li, H.-X. Chen, T.-J. Mu, and S.-M. Hu, “Discene: Object decoupling and interaction modeling for complex scene generation,” in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1–12. 3
- [45] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, “Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 6517–6526. 3
- [46] X. Yang, J. Lin, Y. Xu, H. Li, and Y. Chen, “Advancing highfidelity 3d and texture generation with 2.5 d latents,” arXiv preprint arXiv:2505.21050, 2025. 3
- [47] J. He, H. Li, Y. Hu, G. Shen, Y. Cai, W. Qiu, and Y.-C. Chen, “Disenvisioner: Disentangled and enriched visual prompt for customized image generation,” arXiv preprint arXiv:2410.02067, 2024. 3
- [48] W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, and S. Scherer, “Tartanair: A dataset to push the limits of visual slam,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020, pp. 4909–4916. 4
- [49] Z. Li and N. Snavely, “Megadepth: Learning single-view depth prediction from internet photos,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2041–2050. 4
- [50] Q. Wang, S. Zheng, Q. Yan, F. Deng, K. Zhao, and X. Chu, “Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation,” arXiv preprint arXiv:1912.09678,

2019. 4

- [51] J. Cho, D. Min, Y. Kim, and K. Sohn, “Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes,” arXiv preprint arXiv:2110.11590, 2021. 4
- [52] Y. Yao, Z. Luo, S. Li, J. Zhang, Y. Ren, L. Zhou, T. Fang, and L. Quan, “Blendedmvs: A large-scale dataset for generalized multi-view stereo networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 1790–1799. 4
- [53] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. M¨uller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952,

2023. 4

- [54] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 2015, pp. 234–241. 4

- [55] D. Li, A. Kamko, E. Akhgari, A. Sabet, L. Xu, and S. Doshi, “Playground v2. 5: Three insights towards enhancing aesthetic quality in textto-image generation,” arXiv preprint arXiv:2402.17245, 2024. 4
- [56] J. Chen, J. Yu, C. Ge, L. Yao, E. Xie, Y. Wu, Z. Wang, J. Kwok, P. Luo, H. Lu et al., “Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,” arXiv preprint arXiv:2310.00426, 2023. 4
- [57] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205. 4, 5
- [58] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” arXiv preprint arXiv:2209.03003, 2022. 4
- [59] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747,

2022. 4

- [60] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. M¨uller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in Forty-first international conference on machine learning, 2024. 4
- [61] cloneofsimo and Team Fal, “Introducing AuraFlow v0.1, an open exploration of large rectified flow models,” July 2024, accessed: 2025-02-25. [Online]. Available: https://blog.fal.ai/auraflow/ 4
- [62] M. Gui, J. Schusterbauer, U. Prestel, P. Ma, D. Kotovenko, O. Grebenkova, S. A. Baumann, V. T. Hu, and B. Ommer, “Depthfm: Fast generative monocular depth estimation with flow matching,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, 2025, pp. 3203–3211. 4
- [63] G. M. Garcia, K. Abou Zeid, C. Schmidt, D. De Geus, A. Hermans, and B. Leibe, “Fine-tuning image-conditional diffusion models is easier than you think,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 753–762. 4
- [64] C. Ye, L. Qiu, X. Gu, Q. Zuo, Y. Wu, Z. Dong, L. Bo, Y. Xiu, and X. Han, “Stablenormal: Reducing diffusion variance for stable and sharp normal,” arXiv preprint arXiv:2406.16864, 2024. 4, 11
- [65] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016. 8
- [66] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022. 10
- [67] M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, R. Webb, and J. M. Susskind, “Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10912–10922. 10
- [68] Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” arXiv preprint arXiv:2001.10773, 2020. 10
- [69] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12. Springer, 2012, pp. 746–

760. 11

- [70] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839. 11
- [71] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” The International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013. 11
- [72] T. Schops, J. L. Schonberger, S. Galliani, T. Sattler, K. Schindler, M. Pollefeys, and A. Geiger, “A multi-view stereo benchmark with highresolution images and multi-camera videos,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 3260–

3269. 11

- [73] I. Vasiljevic, N. Kolkin, S. Zhang, R. Luo, H. Wang, F. Z. Dai, A. F. Daniele, M. Mostajabi, S. Basart, M. R. Walter et al., “Diode: A dense indoor and outdoor depth dataset,” arXiv preprint arXiv:1908.00463,

2019. 11

- [74] T. Koch, L. Liebel, F. Fraundorfer, and M. Korner, “Evaluation of cnnbased single-image depth estimation methods,” in Proceedings of the European Conference on Computer Vision (ECCV) Workshops, 2018, pp. 0–0. 11
- [75] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black, “A naturalistic open source movie for optical flow evaluation,” in Computer Vision– ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12. Springer, 2012, pp. 611–625. 11

[76] G. Bae and A. J. Davison, “Rethinking inductive biases for surface normal estimation,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 11, 12

