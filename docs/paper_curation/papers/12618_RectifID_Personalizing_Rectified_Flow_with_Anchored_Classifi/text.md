# arXiv:2405.14677v4[cs.CV]10Dec2024

## RectifID: Personalizing Rectified Flow with Anchored Classifier Guidance

Zhicheng Sun1 , Zhenhao Yang3, Yang Jin1, Haozhe Chi1, Kun Xu2, Kun Xu2, Liwei Chen2, Hao Jiang1, Yang Song, Kun Gai2, Yadong Mu1∗

1Peking University, 2Kuaishou Technology, 3University of Electronic Science and Technology of China {sunzc,myd}@pku.edu.cn

### Abstract

Customizing diffusion models to generate identity-preserving images from userprovided reference images is an intriguing new problem. The prevalent approaches typically require training on extensive domain-specific images to achieve identity preservation, which lacks flexibility across different use cases. To address this issue, we exploit classifier guidance, a training-free technique that steers diffusion models using an existing classifier, for personalized image generation. Our study shows that based on a recent rectified flow framework, the major limitation of vanilla classifier guidance in requiring a special classifier can be resolved with a simple fixed-point solution, allowing flexible personalization with off-the-shelf image discriminators. Moreover, its solving procedure proves to be stable when anchored to a reference flow trajectory, with a convergence guarantee. The derived method is implemented on rectified flow with different off-the-shelf image discriminators, delivering advantageous personalization results for human faces, live subjects, and certain objects. Code is available at https://github.com/feifeiobama/RectifID.

### 1 Introduction

Recent advances in diffusion models (Sohl-Dickstein et al., 2015; Song and Ermon, 2019; Ho et al., 2020) have ignited a surge of research into their customizability. A prominent example is personalized image generation, which aims to integrate user-defined subjects into the generated image. This plays a pivotal role in AI art creation, empowering users to produce identity-consistent images with greater customizability beyond text prompts. Nevertheless, there remain significant challenges in accurately preserving the subject’s identity and being flexible to a variety of personalization needs.

Existing personalization methods are limited in these two aspects, as they require an extra finetuning or pre-training stage. For example, the pioneering works (Gal et al., 2023; Ruiz et al., 2023a) finetune conditional embeddings or model parameters per subject, resulting in suboptimal efficiency and identity consistency due to lack of domain knowledge. On the other hand, the recently prevailing tuning-free methods (Wei et al., 2023; Ye et al., 2023; Li et al., 2024; Wang et al., 2024b) pre-train a conditioning adapter to encode subject features into the generation process. However, their models must be pre-trained on extensive domain-specific data, e.g. LAION-Face 50M (Zheng et al., 2022), which is costly in the first place, and cannot be transferred flexibly across different data domains, e.g. from human faces to common live subjects and objects, and even to multiple subjects.

To address both challenges of identity consistency and flexibility, we advocate a training-free approach that utilizes the guidance of a pre-trained discriminator without extra training of the generative model. This methodology is well-known as classifier guidance (Dhariwal and Nichol, 2021), which modifies

∗Corresponding author.

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

Training-free

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]|
|---|

|face discriminator<br><br>[Figure 4]<br><br>object discriminator<br><br>[Figure 5]<br><br>Off-the-shelf classifier|
|---|

[Figure 6]

Diffusion model (rectified flow)

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]|
|---|

ref img

|[Figure 10]|[Figure 11]|
|---|---|
| | |

- Figure 1: Illustration of training-free classifier guidance. Left: an off-the-shelf discriminator can be reused to steer the existing diffusion model, e.g. rectified flow, to generate identity-preserving images. Right: personalized image generation results for human faces and objects using our proposed method.

an existing denoising process using the gradient from a pre-trained classifier. The rationale behind our exploitation is twofold: first, it directly harnesses the discriminator’s domain knowledge for identity preservation, which may be a cost-effective substitute for training on domain-specific datasets; secondly, keeping the diffusion model intact allows for plug-and-play combination with different discriminators, as shown in Fig. 1, which enhances its flexibility across various personalization tasks. However, the original classifier guidance is largely limited in the reliance on a special classifier trained on noised inputs. Despite recent efforts to approximate the guidance (Kim et al., 2022a; Liu et al.,

- 2023b; Wallace et al., 2023; Ben-Hamu et al., 2024), they have mainly focused on computational efficiency, and have yet to achieve sophisticated performance on personalization tasks.

Technically, to extend classifier guidance for personalized image generation, our work builds on a recent framework named rectified flow (Liu et al., 2023a) featuring strong theoretical properties, e.g. the straightness of its sampling trajectory. By approximating the rectified flow to be ideally straight, the original classifier guidance is reformulated as a simple fixed-point problem concerning only the trajectory endpoints, thus naturally overcoming its reliance on a special noise-aware classifier. This allows flexible reuse of image discriminators for identity preservation in personalization tasks. Furthermore, we propose to anchor the classifier-guided flow trajectory to a reference trajectory to improve the stability of its solving process, which provides a convergence guarantee in theoretical scenarios and proves even more crucial in practice. Lastly, a clear connection is established between our derived anchored classifier guidance and the existing approximation practices.

The derived method is implemented for a practical class of rectified flow (Yan et al., 2024) assumed to be piecewise straight, in combination with face or object discriminators. This provides flexibility for a range of personalization tasks on human faces, live subjects, certain objects, and multiple subjects. Extensive experimental results on these tasks clearly validate the effectiveness of our approach. Our contributions are summarized as follows: (1) We propose a training-free approach to flexibly personalize rectified flow, based on a fixed-point formulation of classifier guidance. (2) To improve its stability, we anchor the flow trajectory to a reference trajectory, which yields a theoretical convergence guarantee when the flow is ideally straight. (3) The proposed method is implemented on a relaxed piecewise rectified flow and demonstrates advantageous results in various personalization tasks.

### 2 Background

Personalized image generation studies incorporating user-specified subjects into the text-to-image generation pipeline. To preserve the subject’s identity, the seminal works Textual Inversion (Gal et al., 2023) and DreamBooth (Ruiz et al., 2023a) finetune conditional embeddings or model parameters for each subject, which imposes high computational costs. Subsequent literature resorts to more efficient parameters (Hu et al., 2022; Han et al., 2023; Yuan et al., 2023) or a pre-trained subject encoder (Wei et al., 2023; Ye et al., 2023) to allow personalization within a few minutes or even without tuning. At the other end, a recent trend is the reuse of existing discriminators to improve identity consistency, such as extracting discriminative face features as the condition (Ye et al., 2023; Wang et al., 2024b) or as a training objective for the encoder (Peng et al., 2024; Gal et al., 2024; Guo et al., 2024). However, these models require extensive pre-training on domain-specific data, e.g. LAION-Face 50M (Zheng et al., 2022). In contrast, our method is a training-free approach that exploits existing discriminators based on the recent rectified flow model, allowing flexible personalization for a variety of tasks.

Rectified flow is an instance of flow-based generative models (Song et al., 2021; Xu et al., 2022; Liu et al., 2023a; Albergo and Vanden-Eijnden, 2023; Lipman et al., 2023). They aim to learn a velocity field v that maps random noise z0 ∼ π0 to samples from a complex distribution z1 ∼ πdata via an ordinary differential equation (ODE):

dzt = v(zt,t)dt. (1)

Instead of directly solving the ODE (Chen et al., 2018), rectified flow (Liu et al., 2023a) simply learns a linear interpolation between the two distributions by minimizing the following objective:

1

E ∥(z1 − z0) − v(zt,t)∥2 dt. (2)

min

v

0

This procedure straightens the flow trajectory and thus allows faster sampling. Ideally, a well-trained rectified flow is a straight flow with uniform velocity v(zt,t) = v(z0,0) following:

zt = z0 + v(zt,t)t. (3) Recently, rectified flow has shown promising efficiency (Liu et al., 2024b) and quality (Esser et al.,

- 2024; Yan et al., 2024) in text-to-image generation. Our work extends its capabilities and theoretical properties to personalized image generation via classifier guidance.

Classifier guidance, initially proposed for class-conditioned diffusion models (Dhariwal and Nichol, 2021), introduces a test-time mechanism to adjust the predicted noise ϵ(zt,t) based on the guidance from a classifier. Given condition c and classifier output p(c|zt), the adjustment is formulated as:

log p(c|zt), (4)

ϵˆ(zt,t) = ϵ(zt,t) + s · σt∇zt

where s denotes the guidance scale, and σt is determined by the noise schedule. Noteworthy, the condition c is not restricted to class labels, but can be extended to text (Nichol et al., 2022) and beyond.

However, it is largely limited by the reliance on a noise-aware classifier for noised inputs zt, which restricts the use of most pre-trained discriminators that only predict the likelihood p(c|z1) on clean images. Consequently, its usefulness is limited in practice. See Appendix B for more related work.

### 3 Method

This work aims at customizing rectified flow with classifier guidance. We show that the above limit of classifier guidance may be solved with a simple fixed-point solution for rectified flow (Section 3.1). To improve its stability, Section 3.2 proposes a new anchored classifier guidance with a convergence guarantee. Lastly, the implementation and applications are described in Sections 3.3 and 3.4.

#### 3.1 Classifier Guidance for Rectified Flow

This section first derives the vanilla classifier guidance for rectified flow, and then present an initial attempt to remove the need for the noise-aware classifier p(c|zt), which is based on a new fixed-point solution of classifier guidance assuming that the rectified flow is ideally straight.

The classifier guidance can be derived as modifying the potential associated with the rectified flow. According to the Helmholtz decomposition, a velocity field v may be decomposed into:

ϕ(zt,t) + r(zt,t), (5) where ϕ is a scaler potential and r is a divergence-free rotation field. They can be determined by solving the Poisson’s equation ∇2ϕ(zt,t) = ∇ · v(zt,t), but this is beyond our focus. We directly add a new potential proportional to the log-likelihood to simulate classifier guidance, as follows:

v(zt,t) = ∇zt

[ϕ(zt,t) + s · log p(c|zt)] + r(zt,t), (6)

vˆ(zt,t) = ∇zt

where s denotes the guidance scale, andˆis used to distinguish the new flow from the original one. Subtracting the above two equations yields the vanilla classifier guidance, similar in form to Eq. (4):

log p(c|zt). (7) While this classifier guidance should allow for test-time conditioning of rectified flow, it cannot be applied in the absence of noise-aware classifier p(c|zt). In the following, we show that this limitation may be overcome by exploiting the straightness property of rectified flow.

vˆ(zt,t) = v(zt,t) + s · ∇zt

|[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|
|---|

Classifier guidance

Targettrajectory

𝑧̂

|[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]|
|---|

𝑧 𝑧

Reference trajectory

- Figure 2: Illustration of anchored classifier guidance for rectified flow. Left: we propose to guide the flow trajectory while implicitly enforcing it to flow straight and stay close to a reference trajectory. Right: comparison of the new trajectory with the reference trajectory (in the last three sampling steps).

Attempt to bypass noise-aware classifier. We make a key observation that the intermediate classifier guidance ∇zt

log p(c|zt) can be circumvented by approximating the new flow trajectory to be straight (an ideal guidance should preserve the properties of rectified flow) and focusing on the endpoint z1. Formally, substituting t = 1 in Eqs. (3) and (7) allows skipping any intermediate guidance terms:

z1 = z0 + vˆ(z1,1)

(8)

= z0 + v(z1,1) + s · ∇z1

log p(c|z1).

Interestingly, this turns out to be a fixed-point problem w.r.t. z1, suggesting that the classifier-guided flow trajectory could be solved iteratively by numerical methods such as the fixed-point iteration, without knowing the noise-aware classifier. This greatly enhances the flexibility of classifier guidance to a variety of off-the-shelf image discriminators. However, our further analysis reveals both empirical (Section 4.3) and theoretical evidence questioning the convergence of this iterative approach:

Proposition 1. There exist Lipschitz continuous functions v(z1,1) and ∇z1

log p(c|z1), such that the fixed-point iteration for solving the target trajectory based on Eq. (8) is not guaranteed to converge by the Banach fixed-point theorem (Banach, 1922), irrespective of the choice of s > 0.

Proof. Consider the following construction. Let v(z1,1) and ∇z1 log p(c|z1) be identical functions with a Lipschitz constant greater than 1. Then, the Lipschitz constant of the right-hand side of the fixed-point equation is greater than 1 for any s > 0. This violates the Banach fixed-point theorem’s requirement for a Lipschitz constant strictly less than 1, thus convergence is not guaranteed.

| |
|---|

- Proposition 1 shows that the derived fixed-point solution may not always be practical. Intuitively,

even with a small perturbation at z1, the target flow trajectory estimated by Eq. (8) could diverge significantly after iterated updates, which hinders the controllability of rectified flow. This motivates us to anchor the target flow trajectory to a reference trajectory to stabilize its solving process.

#### 3.2 Anchored Classifier Guidance

This section establishes a new type of classifier guidance based on a reference trajectory. The idea is to constrain the new trajectory to be straight and near the reference trajectory, as illustrated in Fig. 2. It provides a better convergence guarantee and a certain degree of interpretability.

Let zˆt and zt represent two flow trajectories originating from the common starting point z0 with or without classifier guidance. The symbol ˆ denotes the new trajectory with classifier guidance. Assuming the two trajectories are close and straight (ideally preserving the characteristics of rectified flow), their difference can be estimated based on Eq. (7) and the first-order Taylor expansion:

vˆ(zˆt,t) − v(zt,t) = v(zˆt,t) + s · ∇zˆt

log p(c|zˆt) − v(zt,t) ≈ [∇zt

v(zt,t)](zˆt − zt) + s · ∇zˆt

log p(c|zˆt)

(9)

= [∇zt

v(zt,t)t](vˆ(zˆt,t) − v(zt,t)) + s · ∇zˆt

log p(c|zˆt)

= [I − ∇zt

z0](vˆ(zˆt,t) − v(zt,t)) + s · ∇zˆt

log p(c|zˆt),

where the final step is derived from Eq. (3). From here, a new form of classifier guidance is obtained:

log p(c|zˆt). (10)

vˆ(zˆt,t) = v(zt,t) + s · [∇z0

zt]∇zˆt

This new classifier guidance anchors the target velocity to a predetermined reference velocity v(zt,t) that is dependent only on t and irrelevant to the current state zˆt, thereby constraining the target flow trajectory near the reference trajectory and improving its controllability. Next, we extend its applicability to the more common scenarios where the noise-aware classifier p(c|zˆt) is absent.

Bypassing noise-aware classifier. To circumvent the intermediate classifier guidance, we follow the previous practice of substituting t = 1 into Eqs. (3) and (10), yielding a fixed-point problem w.r.t. zˆ1:

log p(c|zˆ1). (11)

zˆ1 = z1 + s · [∇z0

z1]∇zˆ1

As can be seen, the target endpoint zˆ1 is also anchored to a known reference point z1, which should enhance its stability in the solving process via fixed-point iteration or alternative numerical methods. Below, we exemplify its favorable theoretical property using the fixed-point iteration:

#### Proposition 2. Suppose ∇zˆ1

log p(c|zˆ1) is Lipschitz continuous w.r.t. zˆ1, the fixed-point iteration to solve the target trajectory by Eq. (11) exhibits at least linear convergence with a properly chosen s.

Proof. Denote the Frobenius norm of ∇z0

log p(c|zˆ1) as L2. The Lipschitz constant of the right side of the equation w.r.t. zˆ1 is upper bounded by s · L1 · L2. By choosing a sufficiently small s < 1/(L1 · L2), the Lipschitz constant of the right side is reduced to less than 1, thus ensuring linear convergence by the Banach fixed-point theorem.

z1 as L1, and the Lipschitz constant of ∇zˆ1

| |
|---|

Interpretation of new classifier guidance. In addition to the above convergence guarantee, our new classifier guidance can be interpreted by connecting with gradient backpropagation. From Eq. (10) one could obtain an estimate of the intermediate classifier guidance (see Appendix A for derivation):

log p(c|zˆ1). (12)

log p(c|zˆt) = [∇zt

z1]∇zˆ1

∇zˆt

This suggests that our method is secretly estimating the intermediate classifier guidance with gradient backpropagation. While this is implicitly assumed or directly used in recent works that adapt classifier guidance to flow-based models (Wallace et al., 2023; Liu et al., 2023b; Ben-Hamu et al., 2024), it is explicitly derived here based on a very different assumption (the straightness of the flow trajectory). Such a connection helps to rationalize both our adopted assumption and the existing practice.

#### 3.3 Practical Algorithm

Extension to piecewise rectified flow. The above analyses are performed based on the assumption that the rectified flow is well-trained and straight, which is often not the case in reality. In fact, existing rectified flow usually require multiple sampling steps due to the inherent curvature in the flow trajectory. Inspired by Yan et al. (2024), we adopt a relaxed assumption during implementation that the rectified flow is piecewise linear. Let there be K time windows {[tk,tk−1)}1k=K where 1 = tK > ··· > tk > tk−1 > ··· > t0 = 0, and the flow trajectory is assumed straight within each time window, then the inference procedure can be expressed as:

+ v(zt,t)(t − tk−1), (13)

zt = zt

k−1

where k is the index of the time window [tk,tk−1) that t belongs to. Note that this framework is also compatible with the vanilla rectified flow by setting K to the number of sampling steps.

The previously derived fixed-point iteration in Eq. (11) cannot be applied directly, since its assumption that the target and reference trajectory segments share the same starting point (e.g. zˆt

)

= zt

k−1

k−1

may be violated after updates. A quick fix is to reinitialize the reference trajectory every round with predictions for updated target starting points. This allows to formulate the following problem:

zte

= zte

∇zˆtk

+ s · ∇ztk−1

log p(c|zˆt

zˆt

)

k

k

k

k

(14)

z1e ∇zˆ1

= zte

+ s · ∇ztk−1

log p(c|zˆ1),

k

where the last step is obtained by recursively applying Eq. (12) to backpropagate the guidance signal, and a superscript e is introduced to denote the endpoint of the previous trajectory segment, as the above fix may disconnect different segments of the reference trajectory. Meanwhile, a straight-through estimator (Bengio et al., 2013) is applied to allow computing the Jacobian across different trajectory segments by estimating the Jacobian between the adjacent points zt

and zte

with I.

k

k

Algorithm 1 Anchored Classifier Guidance Input: rectified flow v, classifier p(c|·), sampling steps K, iterations N.

Initialize reference trajectory zt[0]

from v. ▷ Eq. (13) Initialize target trajectory zˆt[0]

k

← zt[0]

. for i ← 0 to N − 1 do

k

k

Update reference trajectory with predicted starting points zt[i+1]

. ▷ Eq. (15) Update target trajectory zˆt[i+1]

k

with classifier output p(c|zˆ1[i]). ▷ Eq. (16) Output: target trajectory zˆt[N]

k

subject to condition c.

k

Solving target flow trajectory. The target trajectory under classifier guidance, subject to Eq. (14), can be estimated iteratively by starting with zˆt[0]

= zt[0]

and performing the following iterations: zt[i+1]

k

k

= zt[i]

+ zte[i+1]

− zte[i]

+ zˆt[i]

− zte[i]

, (15)

k

k

k

k

k

k

current offset

predicted update

zˆt[i+1]

= zte[i+1]

z1e[i+1] ∇zˆ[i]

log p(c|zˆ1[i]), (16)

+ s · ∇z[i+1]

k

k

tk−1

1

where the superscript [i] is used to indicate the target and reference trajectories at the i-th iteration. Specifically, Eq. (15) implements the prediction of updated target starting points by extrapolating from history updates, and Eq. (16) tackles the derived problem. Note that there are more sophisticated methods for predicting target starting points and solving this problem, e.g. quasi-Newton methods, but we opt for simplicity here and leave their exploration to future work. The complete procedure for implementing the proposed classifier guidance is summarized by Algorithm 1.

#### 3.4 Applications

The proposed algorithm is flexible for various personalized image generation tasks on human faces and common subjects. Given a reference image zref and our generated image zˆ1, we use their feature similarity on an off-the-shelf discriminator f, e.g. the face specialist ArcFace (Deng et al., 2019) or a self-supervised backbone DINOv2 (Oquab et al., 2023), as classifier guidance. In addition, to improve the guidance signal, a face detector or an open-vocabulary object detector g is employed to locate the identity-relevant region for feature extraction. Formally, the classifier output is as follows:

p(c|zˆ1[i]) = sim f ◦ g(zˆ1[i]),f ◦ g(zref) . (17)

More details are described in Appendix C. Notably, both configurations can be flexibly extended to a multi-subject scenario by incorporating a bipartite matching step between multiple detected subjects.

4 Experiments

#### 4.1 Experimental Settings

Datasets. Our method does not involve training data, as it operates only at test time. For face-centric evaluation, we follow Pang et al. (2024) to evaluate on 20 prompts with the first 200 images from CelebA-HQ (Liu et al., 2015; Karras et al., 2018) as reference images. For subject-driven generation, we conduct qualitative studies on a subset of examples from the DreamBooth dataset (Ruiz et al., 2023b), spanning 10 subjects across two live subject categories and three object categories.

Metrics. Three metrics are considered: identity similarity, prompt consistency, and computation time. The first two are measured using an ArcFace model (Deng et al., 2019) and CLIP encoders (Radford

- et al., 2021), while the latter is tested on an NVIDIA A800 GPU. We reproduce the latest methods IP-Adapter (Ye et al., 2023), PhotoMaker (Li et al., 2024), and InstantID (Wang et al., 2024b) for a comprehensive comparison, and also include the existing baselines in Pang et al. (2024).

Implementation details. We experiment with a frozen piecewise rectified flow (Yan et al., 2024) finetuned from Stable Diffusion 1.5 (Rombach et al., 2022) with 4 equally divided time windows. The number of sampling steps is set to a minimum K = 4 given the memory overhead of backpropagation.

Table 1: Quantitative comparison for face-centric personalization. The inference time is measured in seconds on an NVIDIA A800. Unlike the previous state-of-the-art methods that require training on large face datasets (the number of images is listed for reference), our method achieves superior performance in a training-free manner, by exploiting the guidance from an off-the-shelf discriminator.

Method Base model Training Identity ↑ Prompt ↑ Time ↓ Textual Inversion (Gal et al., 2023) SD 2.1 - 0.2115 0.2498 6331 DreamBooth (Ruiz et al., 2023a) SD 2.1 - 0.2053 0.3015 623 NeTI (Alaluf et al., 2023) SD 1.4 - 0.3789 0.2325 1527 Celeb Basis (Yuan et al., 2023) SD 1.4 - 0.2070 0.2683 140 Cross Initialtion (Pang et al., 2024) SD 2.1 - 0.2517 0.2859 346 IP-Adapter (Ye et al., 2023) SD 1.5 10M 0.4778 0.2627 2 PhotoMaker (Li et al., 2024) SDXL 112K 0.2271 0.3079 4 InstantID (Wang et al., 2024b) SDXL 60M 0.5806 0.3071 6

- RectifID (20 iterations) SD 1.5 - 0.4860 0.2995 9 RectifID (100 iterations) SD 1.5 - 0.5930 0.2933 46
- RectifID (20 iterations) SD 2.1 - 0.5034 0.3151 20

Input Celeb Basis IP-Adapter PhotoMaker InstantID RectifID

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

western vibes, sunset, rugged landscape

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

in a cowboy hat

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

colorful mural on a street wall

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

as a knight in plate armor

- Figure 3: Qualitative comparison for face-centric personalization. See Figs. 9 to 12 for more samples.

A naive implementation takes 14GB of GPU memory, which fits on a range of consumer-grade GPUs. More results on alternative rectified flows can be found in Appendix D.4. For hyperparameters, the guidance scale is fixed to s = 1 in quantitative evaluation. Meanwhile, for stability, the gradient is normalized following Karunratanakul et al. (2024). The number of iterations is set to N = 100.

#### 4.2 Main Results

Face-centric personalization. Table 1 and Fig. 3 compare our method (denoted RectifID) with extensive baselines. Overall, our training-free approach achieves state-of-the-art performance in quantitative evaluations. Specifically, we observe that: (1) our SD 1.5-based implementation yields the highest identity similarity of all, and leads in prompt consistency among SD 1.x-based methods. It is also computationally efficient, e.g. taking less time than existing tuning-based methods, and outperforming the training-based IP-Adapter (Ye et al., 2023) in a near inference time of 9 seconds vs. 2 seconds. (2) By simply replacing the base diffusion model with SD 2.1 at its default image size, our prompt consistency further surpasses SDXL (Podell et al., 2024)-based models. Note, however,

Input Textual Inversion∗ DreamBooth∗ BLIP-Diffusion Emu2 RectifID

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

in a firefighter outfit

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

on pink fabric

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

in a purple wizard outfit

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

in the jungle

- Figure 4: Qualitative comparison for subject-driven generation. ∗ denotes finetuned with multiple images of the target subject to achieve sufficient identity consistency. See Fig. 13 for more samples.

Input FastComposer RectifID Input Cones 2 RectifID

[Figure 66]

[Figure 67]

enjoying at an amusement park

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

in a purple wizard outfit

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

sailing near the Sydney Opera House

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

in the jungle

[Figure 80]

[Figure 81]

- Figure 5: Qualitative comparison for multi-subject personalization. See Fig. 14 for more samples.

that the rest of this paper still uses SD 1.5 for a fair comparison to SD 1.x-based baselines in various personalization tasks, excluding potential improvements from using better base models. (3) In general, our method takes a big step towards bridging the substantial performance gap with training-based personalization methods by exploring the effectiveness of training-free classifier guidance.

For face-centric qualitative comparison in Fig. 3, our method remains advantageous as its generated images by the guidance of the face discriminator exhibit high identity consistency. In comparison, InstantID (Wang et al., 2024b) delivers a near level of consistency by controlling face landmarks, but sometimes distorts the face shape (the first and third images) and contains much less natural variation. More generated samples are provided in Figs. 11 and 12 in the appendix.

Subject-driven generation. Our approach is flexibly extended beyond human faces towards more subjects, including certain common animals and regularly shaped objects. To validate our flexibility, Fig. 4 qualitatively compares it on three cats or dogs and a regularly shaped can, where the images generated by our method achieve highly competitive identity and prompt consistency. In comparison, the state-of-the-art method Emu2 (Sun et al., 2024), as a generalist multimodal large language model, yields high identity similarity largely by reconstructing the input image, which limits its usefulness. The tuning-based Textual Inversion (Gal et al., 2023) and DreamBooth (Ruiz et al., 2023a) only work well with multiple images and exhibit inferior prompt consistency due to finetuned model parameters or prompt embeddings. See Fig. 13 in the appendix for additional results from more subjects.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Gradient descent of noise

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

RectifID w/o anchor

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

RectifID

s × 0.5 s × 1 s × 2 N = 20 N = 50 N = 100

- Figure 6: Comparison with alternative designs at varying guidance scale (or learning rate) and iterations. The prompts are “cave mural depicting a person” and “a person as a priest in blue robes”. The base learning rate for gradient descent is 0.4, with momentum of 0.9 and an ℓ2 regularizer of 1.0.

Multi-subject personalization. Our method can be further extended to multi-subject scenarios via a bipartite matching step. Figure 5 compares it to the domain experts FastComposer (Xiao et al., 2023) and Cones 2 (Liu et al., 2023c) on composing multiple faces, live subjects and objects. As can be seen, our method achieves overall advantageous identity consistency, in spite of differences in non-persistent attributes such as hairstyle. Image semantics and quality are also well preserved, as exemplified by the amusement park details in the first image, with some others even surpassing the SD 2.1-based specialized model Cones 2. More generated samples can be found in Fig. 14 in the appendix.

#### 4.3 Ablation Study

To justify the effectiveness of our proposed classifier guidance, Fig. 6 and Table 2 compare it with two variants: the previously derived guidance without anchor, namely using Eq. (8), and a gradient descent method on the initial noise similar to DOODL (Wallace et al., 2023) and D-Flow (Ben-Hamu et al., 2024). The figure depicts that the gradient descent is unstable (left) and converges relatively slowly (right) despite using momentum and ℓ2 regularization. And its identity preservation is sensitive to the learning rate. Though our new fixed-point formulation allows for a more stable layout, the initial version fails to converge as the face feature keeps drifting. In contrast, our full method exhibits better stability (left) and faster convergence (right) by implicitly regularizing the flow trajectory to be close and straight. This is further supported by the quantitative comparison, where our method delivers better identity and prompt consistency than the alternatives. Further analysis for hyperparameter sensitivity is provided in Appendix D.3.

Table 2: Quantitative comparison with alternative designs. The number of iterations is 100, and the remaining settings for gradient descent follow Fig. 6.

Method Identity ↑ Prompt ↑

Gradient descent of noise 0.5249 0.2842 RectifID w/o anchor 0.1158 0.2916

RectifID 0.5930 0.2933

#### 4.4 Generalization

To validate the generalizability of our approach to broader application scenarios, we have extended it to more controllable generation tasks by directly using the guidance functions from Universal Guidance (Bansal et al., 2024). The experimental results under the guidance of segmentation map or style image are illustrated in Fig. 7. As shown, our classifier guidance can perform both tasks without any additional tuning, faithfully following the various forms of control signals provided by the user. This confirms the adaptability of our approach for various controllable generation tasks. Additional generalization analysis of our method for broader diffusion models is presented in Appendix D.1.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- (a) Segmentation map

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

- (b) Style transfer

- Figure 7: Experimental results for more controllable generation tasks. The first column shows the guidance, and the rest are the generated results. Our method is extended to various controllable generation tasks by incorporating the guidance functions from Universal Guidance (Bansal et al., 2024).

### 5 Conclusion

This work presents a training-free personalized image generation method using anchored classifier guidance. It extends the applicability of the original classifier guidance based on two key findings: first, by developing on a rectified flow framework assuming ideal straightness, the classifier guidance can be transformed into a new fixed-point formulation involving only clean image-based discriminators; secondly, anchoring the flow trajectory to a reference trajectory greatly improves its solving stability. The derived anchored classifier guidance allows flexible reuse of existing image discriminators to improve identity consistency, as validated by extensive experiments on various personalized image generation tasks for human faces, live subjects, certain objects, and multiple subjects.

Acknowledgement: This research work is supported by National Key R&D Program of China (2022ZD0160305), a research grant from China Tower Corporation Limited, Qinglonghu Laboratory, Beijing 100871 and a grant from Beijing Aerospace Automatic Control Institute.

### References

Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-to-image personalization. ACM Transactions on Graphics, 42(6):1–10, 2023.

Michael Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In International Conference on Learning Representations, 2023.

Xiang An, Xuhan Zhu, Yuan Gao, Yang Xiao, Yongle Zhao, Ziyong Feng, Lan Wu, Bin Qin, Ming Zhang, Debing Zhang, et al. Partial FC: Training 10 million identities on a single machine. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pages 1445–1449, 2021.

Stefan Banach. Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fundamenta Mathematicae, 3(1):133–181, 1922.

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In International Conference on Learning Representations, 2024.

Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-Flow: Differentiating through flows for controlled generation. In International Conference on Machine Learning, 2024.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. PhotoVerse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, pages 6572–6583, 2018.

Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In Advances in Neural Information Processing Systems, pages 25683–25696, 2022.

Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations, 2023.

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. ArcFace: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4690–4699, 2019.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, pages 8780–8794, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In International Conference on Learning Representations, 2023.

Rinon Gal, Or Lichter, Elad Richardson, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. LCM-lookahead for encoder-based text-to-image personalization. In Proceedings of the European Conference on Computer Vision, 2024.

Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2426–2436, 2023.

Jia Guo, Jiankang Deng, Alexandros Lattas, and Stefanos Zafeiriou. Sample and computation redistribution for efficient face detection. In International Conference on Learning Representations, 2022.

Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. PuLID: Pure and lightning ID customization via contrastive alignment. In Advances in Neural Information Processing Systems, 2024.

Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. SVDiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023.

Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, WeiHsiang Liao, Yuki Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, et al. Manifold preserving guided diffusion. In International Conference on Learning Representations, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851, 2020.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Haibo Jin, Shengcai Liao, and Ling Shao. Pixel-in-pixel net: Towards efficient facial landmark detection in the wild. International Journal of Computer Vision, 129(12):3174–3194, 2021.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

Korrawe Karunratanakul, Konpat Preechakul, Emre Aksan, Thabo Beeler, Supasorn Suwajanakorn, and Siyu Tang. Optimizing diffusion noise can serve as universal motion priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. DiffusionCLIP: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2426–2435, 2022a.

Kihong Kim, Yunho Kim, Seokju Cho, Junyoung Seo, Jisu Nam, Kychul Lee, Seungryong Kim, and KwangHee Lee. DiffFace: Diffusion-based face swapping with facial guidance. arXiv preprint arXiv:2212.13344, 2022b.

Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22691–22702, 2023.

Dongxu Li, Junnan Li, and Steven Hoi. BLIP-Diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. In Advances in Neural Information Processing Systems, pages 30146–30166, 2023.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. PhotoMaker: Customizing realistic human photos via stacked ID embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Hanwen Liu, Zhicheng Sun, and Yadong Mu. Countering personalized text-to-image generation with influence watermarks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12257–12267, 2024a.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023a.

Xingchao Liu, Lemeng Wu, Shujian Zhang, Chengyue Gong, Wei Ping, and Qiang Liu. FlowGrad: Controlling the output of generative ODEs with gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24335–24344, 2023b.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. InstaFlow: One step is enough for high-quality diffusion-based text-to-image generation. In International Conference on Learning Representations, 2024b.

Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. In Advances in Neural Information Processing Systems, pages 57500–57519, 2023c.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3730–3738, 2015.

Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In Proceedings of the European Conference on Computer Vision, pages 728–755, 2022.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. DiffEditor: Boosting accuracy and flexibility on diffusion-based image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8488–8497, 2024.

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pages 16784–16804, 2022.

##### OpenAI. GPT-4V(ision) system card. https://openai.com/research/gpt-4v-system-card,

- 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2023.

Lianyu Pang, Jian Yin, Haoran Xie, Qiping Wang, Qing Li, and Xudong Mao. Cross initialization for personalized text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. PortraitBooth: A versatile portrait model for fast identitypreserved personalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023a.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. HyperDreamBooth: Hypernetworks for fast personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023b.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In Proceedings of the European Conference on Computer Vision, 2024.

Patrick Schramowski, Manuel Brack, Björn Deiseroth, and Kristian Kersting. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22522–22531, 2023.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems, pages 25278–25294, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265, 2015.

Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, and Arash Vahdat. Loss-guided diffusion models for plug-and-play controllable generation. In International Conference on Machine Learning, pages 32483–32498, 2023.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, pages 11918–11930, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Zhentao Tan and Yadong Mu. Learning solution-aware transformers for efficiently solving quadratic assignment problem. In International Conference on Machine Learning, 2024.

Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc N Tran, and Anh Tran. AntiDreamBooth: Protecting users from personalized text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2116–2127, 2023.

Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7280–7290, 2023.

Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. In Advances in Neural Information Processing Systems, 2024a.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. InstantID: Zero-shot identitypreserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024b.

Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. In International Conference on Learning Representations, 2023.

Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. ELITE: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023.

Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. FastComposer: Tuning-free multi-subject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023.

Yilun Xu, Ziming Liu, Max Tegmark, and Tommi Jaakkola. Poisson flow generative models. In Advances in Neural Information Processing Systems, pages 16782–16795, 2022.

Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. PeRFlow: Piecewise rectified flow as universal plug-and-play accelerator. In Advances in Neural Information Processing Systems, 2024.

Lingxiao Yang, Shutong Ding, Yifan Cai, Jingyi Yu, Jingya Wang, and Ye Shi. Guidance with spherical gaussian constraint for conditional diffusion. In International Conference on Machine Learning, 2024.

Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Zou, and Stefano Ermon. TFG: Unified training-free guidance for diffusion models. In Advances in Neural Information Processing Systems, 2024.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. FreeDoM: Trainingfree energy-guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23174–23184, 2023.

Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. In Advances in Neural Information Processing Systems, pages 72958–72982, 2023.

Yasi Zhang, Peiyu Yu, Yaxuan Zhu, Yingshan Chang, Feng Gao, Ying Nian Wu, and Oscar Leong. Flow priors for linear inverse problems via iterative corrupted trajectory matching. arXiv preprint arXiv:2405.18816, 2024.

Yinglin Zheng, Hao Yang, Ting Zhang, Jianmin Bao, Dongdong Chen, Yangyu Huang, Lu Yuan, Dong Chen, Ming Zeng, and Fang Wen. General facial representation learning in a visual-linguistic manner. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18697–18709, 2022.

Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1219–1229, 2023.

### A Detailed Derivations

Most of the equations in our paper are accompanied by their derivations, except for Eqs. (11) and (12). For the sake of completeness, their derivations are supplemented here.

- Equation (11): We substitute t = 1 into Eqs. (3) and (10) to obtain: zˆ1 = z0 + vˆ(zˆ1,1)

- = z0 + v(z1,1) + s · [∇z0

z1]∇zˆ1

log p(c|zˆ1)

- = z1 + s · [∇z0

z1]∇zˆ1

log p(c|zˆ1).

(18)

- Equation (12): By applying Eq. (10) twice and utilizing the straightness property of rectified flow, we have:

log p(c|zˆt) = 1/s · [∇zt

z0](vˆ(zˆt,t) − v(zt,t))

∇zˆt

= 1/s · [∇zt

z0](vˆ(zˆ1,1) − v(z1,1))

(19)

= [∇zt

z0][∇z0

z1]∇zˆ1

log p(c|zˆ1)

= [∇zt

z1]∇zˆ1

log p(c|zˆ1).

### B Related Work on Classifier Guidance

Since the proposal of classifier guidance (Dhariwal and Nichol, 2021) which uses a special noiseaware classifier as training-free guidance for diffusion models, many efforts have been made to extend its applicability to off-the-shelf loss guidance. They can be grouped into three categories: (1) Early literature focuses on simpler objectives for linear inverse problems such as image super-resolution, deblurring, and inpainting (Chung et al., 2022, 2023; Wang et al., 2023; Zhu et al., 2023; Zhang et al., 2024). (2) These methods can be extended to more complex discriminators through various approximations. Yu et al. (2023); Song et al. (2023) use Tweedie’s formula and Monte Carlo method, respectively, to estimate the integrated classifier guidance. Bansal et al. (2024); He et al. (2024) perform updates directly in the clean data space, with the latter imposing an additional manifold constraint. Similarly, Gaussian spherical constraint is explored in Yang et al. (2024). Mou et al. (2024) advance these techniques to more versatile editing tasks. The above methods are further unified in Ye et al. (2024). (3) A recent line of work directly uses gradient descent with specific diffusion models. To enable their gradient computation, DiffusionCLIP (Kim et al., 2022a) relies on shortened ODE trajectories, while FlowGrad (Liu et al., 2023b) adopts a non-uniform ODE discretization and decomposed gradient computation. DOODL (Wallace et al., 2023), DNO (Karunratanakul et al., 2024) and D-Flow (BenHamu et al., 2024) use invertible models or flow models to backpropagate gradient to the initial noise. Our proposed method is related to the third category, focusing on rectified flow whose approximation remains understudied. Moreover, it features a fixed-point formulation with a convergence guarantee for ideal rectified flow, which allows potentially better stability over the existing approaches, e.g. gradient descent on initial noise, as empirically validated in Section 4.3.

Recent studies also explore the use of pre-trained classifiers to guide personalized image generation, but mostly during model training. DiffFace (Kim et al., 2022b) and PhotoVerse (Chen et al., 2023) directly apply a face discriminator to noised images to compute an identity loss. PortraitBooth (Peng et al., 2024) improves loss quality by computing it at less noisy stages. More relevant to our work, LCM-Lookahead (Gal et al., 2024) and PuLID (Guo et al., 2024) utilize distilled diffusion models to generate images in few steps, allowing for direct gradient backpropagation of image-space losses. However, these personalization methods must first be trained on extensive face recognition data, e.g. LAION-Face 50M (Zheng et al., 2022). In comparison, we harness off-the-shelf face discriminators at test time based on the methodology of classifier guidance, enabling more flexible customization without training. And it can be generalized to other subjects by simply replacing the classifier.

### C Experimental Settings

Method. We mainly experiment on the recently proposed piecewise rectified flow (Yan et al., 2024). The model is finetuned from Stable Diffusion 1.5 (Rombach et al., 2022) on the LAION-Aesthetic-5+ dataset (Schuhmann et al., 2022) without special pre-training on human faces or subjects. We adopt a fixed image size of 512 × 512, number of sampling step K = 4, and a classifier-free guidance scale

of 3.0 during quantitative evaluation. The newly incorporated anchored classifier guidance uses a default guidance scale s = 1.0 and number of iterations N = 100. For qualitative studies, a few results are generated with a slightly different guidance scale s = 0.5 or 2.0 for better visual quality or identity consistency. But in general, our method is not very sensitive to these hyperparameters. In terms of computational and memory overhead, our method takes less than 0.5s per iteration on an NVIDIA A800 GPU and fits on consumer-grade GPUs such as NVIDIA RTX 4080, the latter of which may be further improved with gradient checkpointing or the techniques in Liu et al. (2023b).

For face-centric personalization, our method is implemented with the antelopev2 model pack from the InsightFace library.2 Specifically, it detects and crops the face regions with an SCRFD model (Guo

- et al., 2022), and then extracts face features using an ArcFace model (Deng et al., 2019) trained on Glint360K (An et al., 2021), consistent with most personalization methods that use a face model (Ye et al., 2023; Wang et al., 2024b; Gal et al., 2024). The resulting face feature is compared with the reference image to compute a cosine loss, which serves as the classifier guidance signal.

For subject-driven generation, we use an open-vocabulary object detector OWL-ViT (Minderer et al., 2022) to detect the region of interest, and extract visual features with DINOv2 (Oquab et al., 2023). The extracted feature is compared with the reference to calculate a cosine loss as the guidance signal. While this guidance already works well for various live subjects including multiple dogs and cats, we add an optional ℓ1 loss with a coefficient of 10.0 to help preserve the identity of certain objects, such as cans, vases and duck toys. The current implementation is still limited in not capturing the details of some irregularly shaped objects, e.g. plush toys, and we expect to resolve this issue with an improved discriminator, or by combining with existing image prompt techniques (Ye et al., 2023).

For multi-subject personalization, we consider a simplified case of exactly two subjects and perform the following: first detect the two subjects, then enumerate all possible bipartite matches with the reference subjects to minimize the matching cost. For more complex scenarios, a possible workaround is to formulate it as an quadratic assignment problem and apply efficient solvers (Tan and Mu, 2024).

Evaluation. For face-centric personalization, we follow Pang et al. (2024) in evaluating on the first 200 images in the CelebA-HQ dataset (Liu et al., 2015; Karras et al., 2018) with 20 text prompts including 15 realistic prompts and 5 stylistic prompts. The evaluation process reuses the code from Celeb Basis (Yuan et al., 2023),3 which first detects the face region using a PIPNet (Jin et al., 2021) with a threshold of 0.5 and then computes the cosine similarity on face features extracted by an ArcFace model (Deng et al., 2019). It should be noted that our method adopts a different face detector and different alignment and cropping methods, so it does not overfit the evaluation protocol. For the baselines, in addition to those compared in Pang et al. (2024), we also evaluate the recently proposed IP-Adapter (Ye et al., 2023), PhotoMaker (Li et al., 2024), and InstantID (Wang et al., 2024b) on their recommended settings. The number of their sampling steps is set to 30 for a fair comparison. The checkpoint version of IP-Adapter is ip-adapter-full-face_sd15. The image size is set to 512×512 for IP-Adapter and 1024×1024 for PhotoMaker and InstantID based on SDXL (Podell et al., 2024). In the qualitative analysis, we also include Celeb Basis (Yuan et al., 2023) as a baseline method.

For subject-driven generation, we perform a qualitative rather than a quantitative comparison on a subset of the DreamBooth dataset (Ruiz et al., 2023a) due to the previously mentioned limitations. Nevertheless, many subjects are considered during the qualitative study, including 7 live subjects from two categories (cats and dogs) and 3 regularly shaped objects from different categories (cans, vases, and teapots). For the baselines, we incorporate Textual Inversion (Gal et al., 2023), DreamBooth (Ruiz

- et al., 2023a), BLIP-Diffusion (Li et al., 2023), and Emu2 (Sun et al., 2024) for extensive comparison. Their diffusion models and hyperparameter settings follow the official or Diffusers implementation.4

Licenses. The piecewise rectified flow (Yan et al., 2024) used in the main experiments is released under the BSD-3-Clause License and the 2-rectified flow (Liu et al., 2023a, 2024b) used in Appendix D.4 is released under the MIT License. The InsightFace library for face detection and recognition is released under the MIT License, while its pre-trained models are available for non-commercial research purposes only. The OWL-ViT (Minderer et al., 2022) and DINOv2 (Oquab et al., 2023) models for object detection and feature extraction are released under the Apache-2.0 License. For evaluation, the code of Celeb Basis (Yuan et al., 2023) is licensed under the MIT License. The

- 2https://github.com/deepinsight/insightface
- 3This is mentioned at https://github.com/lyuPang/CrossInitialization#metrics. 4For example, we use https://huggingface.co/docs/diffusers/training/text_inversion.

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

painting in Van Gogh style

reading on the train

Greek sculpture

concert poster

faded film in the snow (a) SD-Turbo

Input

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

stained glass window

pointillism painting

manga drawing (b) Phased consistency model

in a police outfit

oil painting sketch

Input

- Figure 8: Generalization to few-step diffusion models, including SD-Turbo (Sauer et al., 2024) and phased consistency model (Wang et al., 2024a), both distilled from SD and using 4 sampling steps in inference. The results show that our method is effective for personalizing broader diffusion models.

CelebA-HQ (Liu et al., 2015; Karras et al., 2018) and DreamBooth (Ruiz et al., 2023a) datasets for quantitative and qualitative evaluation are released under the CC BY-NC 4.0 and CC-BY-4.0 licenses.

### D Additional Results

#### D.1 Generalization

While our classifier guidance is derived based on rectified flow, the same idea can be generalized to some few-step diffusion models by assuming straightness of their trajectories within each time step. We empirically demonstrate this in Fig. 8 with two popular few-step diffusion models, SDTurbo (Sauer et al., 2024) and phased consistency model (Wang et al., 2024a). As the results indicate, our method effectively personalizes these diffusion models to generate identity-preserving images. We will continue to explore this approach for other generative models in future research.

#### D.2 More Visualizations

More examples of our generated image are provided in Figs. 9 to 14. For face-centric personalized image generation, it is shown that our method can follow a variety of text prompts to generate both realistic or stylistic images while preserving the user-specified identity from a diverse group of people. Although there exist minor differences in the person’s age and hairstyles, the face looks very similar to the reference image. In particular, the method demonstrates good generalizability among different piecewise rectified flows based on SDXL (Podell et al., 2024) and SD 1.5 (Rombach et al., 2022). For subject-driven generation, new results are presented from additional subject types, including different breeds of cats and dogs, and some regularly shaped objects such as vases, demonstrating the flexibility of our approach across different use cases. For multi-subject generation, it naturally blends multiple human faces or live subjects into a single image while maintaining the visual quality

and semantics, revealing a wider range of potentially interesting applications. Overall, our method demonstrates to be effective and flexible for various personalization tasks.

#### D.3 Ablation Study

In addition to the ablation experiments in the main paper, we perform a sensitivity analysis of the two hyperparameters in our method, namely the guidance scale s and the number of iterations N. Specifically, we study the effect of different s under a fixed N = 20 and then the effect of different N under a fixed s = 1.0. The results are summarized in Fig. 15. As can be observed, increasing both hyperparameters from 0 leads to a significant improvement in identity consistency, confirming the effectiveness of our proposed classifier guidance. Also, the performance is stable over a fairly wide range of hyperparameters, indicating that our approach is not very sensitive to hyperparameters. Furthermore, we find that the use of a small classifier guidance scale is actually beneficial for prompt consistency, possibly because it enhances the visual features, as demonstrated in Fig. 2.

#### D.4 Experiments with 2-Rectified Flow

Figures 16 and 17 present additional qualitative results on a vanilla 2-rectified flow (Liu et al., 2023a, 2024b) finetuned on Stable Diffusion 1.4 (Rombach et al., 2022). As can be seen, our method continues to deliver satisfactory identity preservation when moving to a different model, despite a noticeable drop in the generation quality. Concretely, it integrates target subjects with some quite challenging prompts, such as a person swimming or getting a haircut, while showing very little interference with the original background, e.g. jungles and cityscapes. These results clearly validate the effectiveness of our proposed method in alternative rectified flow models.

Note that we also experimented with K = 1 on a single-step InstaFlow (Liu et al., 2024b) distilled from this 2-rectified flow, but found that it tended to converge to slightly distorted images. This may be attributed to the larger modeling error inherent in InstaFlow’s distillation process, which reduces the effectiveness of our approach assuming each flow trajectory segment is well-trained and straight.

### E Broader Impacts and Limitations

Broader impacts. The proposed method can be integrated with the emerging rectified flow-based models to enhance identity preservation and versatile control over existing AI art production pipelines. However, as a training-free personalization technique, it may increase the risk of image faking and have negative societal effects. Some immediate remedies include text-based prompt filters and AIgenerated image detection, but it remains an open problem for a more principled solution, for which we advocate further research on data watermarking and model unlearning as potential mitigations. To further clarify it, we provide a more detailed explanation of these mitigations below:

- • Prompt filtering, model unlearning: Since our method keeps the diffusion model intact, existing techniques for regulating diffusion models can be applied seamlessly, including prompt filters or unlearning methods. The former can be applied explicitly like the text filters in SD models, or implicitly via CFG as in Schramowski et al. (2023). The latter approaches involve finetuning the diffusion model to remove the ability to generate harmful content (Gandikota et al., 2023; Kumari et al., 2023).
- • Data watermarking: To prevent misuse of personal images, one could add a protective watermark to their images (Van Le et al., 2023; Liu et al., 2024a). With this watermark or perturbation, the image can no longer be learned by common personalization methods. However, it is unclear how robust these watermarks are to training-free methods such as ours. An alternative watermarking scheme is to embed special watermark to the images generated by our proposed model, which would be invisible to the users yet identifiable by us (i.e., the tech provider). Images with such watermarks will be marked as being artificial.
- • AI-generated image detection: As a post-hoc safety measure, it helps to distinguish fake images generated by the attackers from real images. Beyond above watermark-based scheme, more sophisticated data-driven methods have attracted increasing interest from the AI community. Despite that current methods still lack accuracy, we believe that developing reliable and widely available AI-generated image detectors is an important research direction.

Limitations. Our theoretical guarantee is limited to ideal rectified flow and cannot be generalized to more complex flow-based models. Empirically, anchoring the new flow trajectory to a reference trajectory only proves effective for faces, live subjects and certain regularly shaped objects, and remains insufficient for many objects with large structural variations, e.g. plush toys. Furthermore, while our method is training-free, its inference time has yet to match several training-based baselines, which may be addressed by applying more advanced numerical solvers to the derived problem.

Another important issue is the lack of pre-trained discriminators. To address this in the short term, we suggest first training a specialized discriminator and then applying our classifier guidance. There are two reasons for doing this instead of finetuning the generator directly: (1) training/finetuning a discriminator is usually more efficient and stable than training/finetuning a generator; (2) it can take full advantage of domain images that have no captions or even labels by using standard contrastive learning loss. In the future, scaling up vision-language models may be a general solution for these domains. The current models such as GPT-4V (OpenAI, 2023) have demonstrated certain generalizability across visual understanding tasks. As they continue to improve in generalizability and robustness, they will become a viable source for guiding diffusion models in new domains.

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

wear a magician hat and a blue coat in a garden

graduating after finishing PhD

as White Queen

wearing headphones

pencil drawing

concert poster

Input

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

holding roses in front of the Eiffel Tower

wear a magician hat and a blue coat in a garden

in a police outfit

reading on the train

buckled in his seat on a plane

sipping coffee at a cafe

Input

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

stained glass window

painting in Van Gogh style

watercolor painting

colorful mural on a street wall

buckled in his seat on a plane

driving a car

Input

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

wearing a brown jacket and a hat, holding a whip

cooking a gourmet meal in blue suit

cubism painting

graduating after finishing PhD

pencil drawing

ink painting

Input

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

holding roses in front of the Eiffel Tower

typing a paper on a laptop

piloting a fighter jet

wearing a hat

concert poster in a chef outfit

Input

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

wear a magician hat and a blue coat in a garden

piloting a fighter jet

pencil drawing

Input cave mural

driving a car swimming

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

wear a magician hat and a blue coat in a garden

as an amazon warrior

in a cowboy hat

watercolor painting

pencil drawing

wearing a hat

Input

- Figure 9: Additional face-centric personalization results with piecewise rectified flow (Yan et al.,

- 2024) based on SDXL (Podell et al., 2024). Our method is compatible with more advanced base models and provides sophisticated personalization results.

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

piloting a fighter jet

in a cowboy hat

colorful mural on a street wall

watercolor painting

Input Banksy art

concert poster

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

pointillism painting

piloting a fighter jet

in a police outfit

playing the LEGO toys

wearing a hat faded film

Input

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

wear a magician hat and a blue coat in a garden

cubism painting

piloting a fighter jet

buckled in his seat on a plane

in a police outfit

watercolor painting

Input

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

wearing a scifi spacesuit in space

surrounded by bookshelves

reading on the train

coding in front of a computer

swimming in a chef outfit

Input

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

pointillism painting

in a cowboy hat

sipping coffee at a cafe

wearing a hat faded film concert poster

Input

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

holding roses in front of the Eiffel Tower

wearing a sunglass on a boat

western vibes, sunset, rugged landscape

stained glass window

piloting a fighter jet

concert poster

Input

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

wear a magician hat and a blue coat in a garden

western vibes, sunset, rugged landscape

wearing a scifi spacesuit in space

as White Queen

reading on the train

colorful mural on a street wall

Input

- Figure 10: Additional face-centric personalization results with piecewise rectified flow (Yan et al.,

2024) based on SDXL (Podell et al., 2024). Our method is compatible with more advanced base models and provides sophisticated personalization results.

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

sitting in the cafe, comic

sitting in a cozy cafe

in a cowboy hat

Input concert poster cave mural in the jungle

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

casting a fire ball, digital art

in Pixar style, surprised

wear a sweater outdoors

colorful graffiti

watercolor painting

swimming

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

casting a fire ball, digital art

writing a novel in a library

playing the guitar

graduating after finishing PhD

colorful graffiti

wear a sweater outdoors

Input

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

western vibes, sunset, rugged landscape

trying on hats in a vintage boutique

colorful mural on a street wall

in assassin’s creed

as White Queen

oil painting

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

western vibes, sunset, rugged landscape

colorful graffiti

watercolor painting

Greek sculpture

concert poster in the jungle

Input

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

casting a fire ball, digital art

writing a novel in a library

as a priest in blue robes

chalk art on a sidewalk

Greek sculpture

Input cave mural

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

graduating after finishing PhD

chalk art on a sidewalk

selfie on a yacht

Input concert poster driving a car ink painting

- Figure 11: Additional face-centric personalization results with piecewise rectified flow (Yan et al., 2024), which is based on Stable Diffusion 1.5 (Rombach et al., 2022). Our method achieves high identity consistency. See Fig. 16 for results on the vanilla 2-rectified flow (Liu et al., 2023a, 2024b).

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

holding a dog in cherry blossoms

colorful graffiti

graduating after finishing PhD

sitting in the cafe, comic

playing guitar in urban

Greek sculpture

Input

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

holding roses in front of the Eiffel Tower

wear a magician hat and a blue coat in a garden

as a knight in plate armor

colorful graffiti

ink painting

Input cave mural

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

camping in the outdoors

colorful mural on a street wall

Input cave mural

on the beach

faded film line art

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

camping in the outdoors

wear a sweater outdoors

colorful mural on a street wall

as a knight in plate armor

as White Queen

Input oil painting

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

pointillism painting

in a cowboy hat

sipping coffee at a cafe

pencil drawing

wearing a hat

in the rain

Input

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

wear a magician hat and a blue coat in a garden

sailing near the Sydney Opera House

casting a fire ball, digital art

cubism painting

watercolor painting

wear a sweater outdoors

Input

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

western vibes, sunset, rugged landscape

wearing a sunglass on a boat

as a priest in blue robes

stained glass window

in the jungle concert poster

Input

- Figure 12: Additional face-centric personalization results with piecewise rectified flow (Yan et al., 2024), which is based on Stable Diffusion 1.5 (Rombach et al., 2022). Our method achieves high identity consistency. See Fig. 16 for results on the vanilla 2-rectified flow (Liu et al., 2023a, 2024b).

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

in a firefighter outfit

in a purple wizard outfit

wearing a rainbow scarf

Input on pink fabric

in the jungle in the snow

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

wearing a santa hat

in a firefighter outfit

in a purple wizard outfit

in a police outfit

in front of a mountain

shiny

Input

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

wearing a santa hat

in a firefighter outfit

in a purple wizard outfit

in a police outfit

wearing a yellow shirt

in the snow

Input

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

wearing a santa hat

in a firefighter outfit

in a purple wizard outfit

in a police outfit

wearing a rainbow scarf

wet

Input

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

wearing a santa hat

in a firefighter outfit

in a purple wizard outfit

in front of a blue house

in the jungle wet

Input

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

in front of the Eiffel Tower

on a purple rug in the forest

with autumn leaves

on a cobblestone street

in front of a city

in the snow

Input

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

on a purple rug in the forest

with autumn leaves

Input on pink fabric

in the jungle on a mirror in the snow

- Figure 13: Additional subject-driven generation results with piecewise rectified flow (Yan et al., 2024), which is based on Stable Diffusion 1.5 (Rombach et al., 2022). Our approach preserves the identity of both live subjects and some regularly shaped objects. Please see Fig. 17 for more examples using the vanilla 2-rectified flow (Liu et al., 2023a, 2024b).

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

reading on the train

colorful mural on a street wal

graduating after finishing PhD

having dinner together

Input movie poster playing guitar

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

enjoying at an amusement park

trying on hats in a vintage boutique

Greek sculpture

cubism painting

swimming

Input movie poster

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

enjoying at an amusement park

sailing near the Sydney Opera House

camping in the outdoors

in front of a mountain

graduating after finishing PhD

ice sculpture

Input

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

Input shiny purple in the jungle wet on the beach in the snow

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

wearing a santa hat

in a purple wizard outfit

in the jungle wet on the beach in the snow

Input

- Figure 14: Additional multi-subject personalization results with piecewise rectified flow (Yan et al., 2024), which is based on Stable Diffusion 1.5 (Rombach et al., 2022). Our approach can naturally compose multiple subjects into the generated image while preserving their identities.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.0 0.5 1.0 2.0 Guidance scale s

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Identitysimilarity•

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 20 50 100 Number of iterations N

0.26

0.28

0.30

0.32

+Promptsimilarity

- Figure 15: Ablation study of hyperparameters. Left: ablation study of guidance scale s under N = 20. Right: ablation study of the number of iterations N under s = 1.0. Our method remains effective over a reasonably wide range of hyperparameters.

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

cubism painting

watercolor painting

stained glass window

writing a novel in a home library

Greek sculpture

line art

Input

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

driving a motorbike in yellow jacket

wearing a sunglass and a life jacket

wearing a sunglass on a boat

as an amazon warrior

in a comic book

ink painting

Input

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

wearing a brown jacket and a hat, holding a whip

having a haircut in a classic barbershop

playing the LEGO toys

colorful mural on a street wall

typing a paper on a laptop

swimming

Input

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

sipping coffee at a cafe

reading on the train

in a police outfit

wearing a sweater outdoors

pencil drawing

Input cave mural

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

trying on hats in a vintage boutique

western vibes, sunset, rugged landscape

in a cowboy hat

coding in a home office

colorful graffiti

as White Queen

Input

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

wearing an oversized sweater

wear a magician hat and a blue coat in a garden

eating bread in front of the Eiffel Tower

in assassin’s creed

in the snow

driving a car

Input

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

pointillism painting

graduating after finishing PhD

holding a bottle of red wine

Input oil painting

happy surprised

- Figure 16: Face-centric personalization results with vanilla 2-rectified flow (Liu et al., 2023a, 2024b), which is based on Stable Diffusion 1.4 (Rombach et al., 2022). Our method preserves their identities well while remaining faithful to the text prompt during generation.

[Figure 472]

Input

[Figure 473]

Input

[Figure 474]

Input

[Figure 475]

Input

[Figure 476]

Input

[Figure 477]

Input

[Figure 478]

Input

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

wearing a santa hat

in a firefighter outfit

in a police outfit

in a purple wizard outfit

purple on pink fabric

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

wearing a santa hat

in a firefighter outfit

in a police outfit

in a chef outfit in the jungle in the snow

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

wearing a santa hat

in a firefighter outfit

in a police outfit

in a purple wizard outfit

in the jungle in the snow

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

wearing a santa hat

in a firefighter outfit

in a police outfit

on the beach in the jungle in the snow

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

wearing a santa hat

wearing a red hat

wearing a rainbow scarf

in front of a mountain

shiny

in the snow

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

on a purple rug in the forest

on a cobblestone street

in front of a mountain

on pink fabric

in the jungle

in the snow

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

with autumn leaves

in front of the Eiffel Tower

in front of a city

in front of a mountain

on the beach in the snow

- Figure 17: Subject-driven generation results with vanilla 2-rectified flow (Liu et al., 2023a, 2024b), which is based on Stable Diffusion 1.4 (Rombach et al., 2022). Examples for additional categories of cats, dogs, and objects are included to demonstrate the effectiveness of our approach.

