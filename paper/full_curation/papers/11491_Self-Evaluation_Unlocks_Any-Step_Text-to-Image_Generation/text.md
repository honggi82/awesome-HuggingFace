### Self-Evaluation Unlocks Any-Step Text-to-Image Generation

# arXiv:2512.22374v1[cs.CV]26Dec2025

Xin Yu1,2 Xiaojuan Qi1*† Zhengqi Li2 Kai Zhang2 Richard Zhang2 Zhe Lin2 Eli Shechtman2 Tianyu Wang2† Yotam Nitzan2†

1The University of Hong Kong 2Adobe Research

#### Abstract

We introduce the Self-Evaluating Model (Self-E), a novel, from-scratch training approach for text-to-image generation that supports any-step inference. Self-E learns from data similarly to a Flow Matching model, while simultaneously employing a novel self-evaluation mechanism: it evaluates its own generated samples using its current score estimates, effectively serving as a dynamic self-teacher. Unlike traditional diffusion or flow models, it does not rely solely on local supervision, which typically necessitates many inference steps. Unlike distillation-based approaches, it does not require a pretrained teacher. This combination of instantaneous local learning and self-driven global matching bridges the gap between the two paradigms, enabling the training of a high-quality text-to-image model from scratch that excels even at very low step counts. Extensive experiments on large-scale text-to-image benchmarks show that Self-E not only excels in few-step generation, but is also competitive with state-of-the-art Flow Matching models at 50 steps. We further find that its performance improves monotonically as inference steps increase, enabling both ultra-fast few-step generation and high-quality long-trajectory sampling within a single unified model. To our knowledge, Self-E is the first from-scratch, any-step text-to-image model, offering a unified framework for efficient and scalable generation.

#### 1. Introduction

Diffusion models [19, 53, 54] and flow matching models [26, 29] currently dominate text-to-image generation due to their stability, scalability, and strong visual fidelity [2, 40, 60, 61]. These models are trained to approximate local supervision from data – either the score function or the instantaneous velocity field – which specifies how a noisy sample should infinitesimally move toward the data manifold at each timestep. Because this supervision is inherently

*Corresponding author. †Project lead.

local, it provides only short-range guidance: each update corrects small deviations but lacks a holistic global view of the target distribution. Consequently, diffusion and flowbased models typically require dozens of sequential steps to reliably traverse the curved reverse trajectory from noise to data, making inference computationally expensive and limiting their use in time-sensitive applications.

A dominant strategy for reducing inference steps is distillation, where a pretrained teacher supervises a student model [25, 34, 43–46, 63, 64]. Although these methods differ technically, they share a core principle: the student is optimized with global objectives that match the teacher’s distributions or trajectories, rather than data-derived local velocities, so that it can perform few-step inference. A key limitation, however, is the reliance on a strong pretrained teacher. This has recently motivated growing interest in self-contained, from-scratch training frameworks that natively yield few-step models. A prominent line of work is consistency-based methods [4, 15, 22, 30, 42, 55, 59], which essentially learn the underlying flow maps [4] – or, equivalently, the average velocity [15] between two points along the reverse trajectory – so that, in principle, the model can follow a one-step shortcut instead of integrating many instantaneous velocities at test time. However, these objectives are typically unstable to optimize from scratch [59, 72] or suffer from quality degradation [70], and have so far scaled reliably only on simpler benchmarks such as ImageNet [8], while large text-to-image systems that do succeed in this regime still rely heavily on distillation [42, 70], undermining the original teacher-free motivation.

In this paper, we present the Self-Evaluating Model (SelfE, pronounced like selfie) – a novel, self-contained, fromscratch training framework enabling any-step text-to-image inference. The model learns simultaneously from data, which provides local velocity supervision, and through a novel self-evaluation mechanism supervising the global distribution. The core idea is conceptually simple yet powerful: the model evaluates its own generated samples using its current local score estimate, effectively serving as a dynamic self-teacher. This self-evaluation becomes an increasingly

[Figure 1]

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

- Figure 1. Qualitative Any-Step Generation. We showcase diverse text-to-image results from our model at different inference step counts, demonstrating coherent semantics, strong text alignment. Text prompts are provided in the supplementary material.

accurate guidance signal – allowing the model to improve itself. By combining instantaneous local learning with selfdriven global matching, Self-E naturally bridges the gap between flow-based and distillation-based paradigms. As a result, it can be trained entirely from scratch while supporting any-step text-to-image inference, generating high-quality images even at very low step counts (see Fig. 1).

To the best of our knowledge, Self-E is the first native any-step text-to-image model, concurrent with TiM [59]. We conduct extensive experiments on large-scale text-to-image generation and show that Self-E achieves both strong fewstep quality and graceful scaling across inference budgets. In the few-step (< 8) setting, Self-E surpasses the performance of diffusion and flow-based models including FLUX1-dev [2], SDXL [37], SANA [61], the distillation-based LCM [33], and concurrent any-step model TiM [59]. Remarkably, although targeting this few-step generation, Self-E is also competitive or even surpasses some state-of-the-art flow-based methods at the 50 step setting. We also note that Self-E’s performance improves monotonically as inference steps increase, enabling both ultra-fast few-step generation and high-quality long-trajectory sampling within a single unified model.

#### 2. Background: Flow Matching

Generative models aim to learn a parameterized distribution pθ(x0|c) that approximates the real data distribution

q(x0|c) where c is a condition such as text prompt. Flow matching and diffusion models achieve this by learning the instantaneous velocity field, or equivalently the score function, induced by a continuous-time forward diffusion process. Specifically, given real data samples x0 ∼ q(x0|c), these models define a trajectory indexed by t ∈ [0,1]:

xt = αtx0 + σtϵ, ϵ ∼ N(0,I), (1)

where coefficients (αt,σt) form a noise scheduler and defines the noisy distribution q(xt|c). Flow matching specifically refers to a particular parameterization that explicitly matches the velocity field:

dxt dt

vt(xt) :=

dαt dt

dσt dt

ϵ, (2)

=

x0 +

where coefficients (αt,σt) = (1 − t,t).

Conditional Flow Matching (CFM) trains a neural network Vθ(xt,t,c) to predict the marginal velocity field (i.e., the expectation of instantaneous velocity) [26] by minimizing the mean squared error between predicted and conditional velocities:

0,ϵ ∥Vθ(xt,t,c) − vt(xt)∥2 . (3)

LCFM(θ) = Et,x

At inference time, we use the predicted velocity to follow the trajectory and generate samples. However, because the velocity is a local quantity, a single-step estimate of the original sample x0 often falls short. Intuitively, a single step

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 2. Self-Evaluating Model. (a) Overview. The model is trained with two complementary objectives: learning from data (b) and

self-evaluation (c). (b) Learning from data. Given a real sample x0, we add noise to obtain xt and train Gtθ→s with an x0-prediction loss, providing local trajectory supervision. (c) Self-evaluation with classifier score. When s < t, we re-noise the generated xˆ0 to xˆs and run the same network in evaluation mode (stop-gradient) twice: once with condition c and once with the null prompt ϕ. The difference between these outputs yields a self-evaluation score, which is treated as a feedback gradient on xˆ0 and back-propagated through the denoising path, enforcing global distribution matching in a teacher-free manner.

only captures the immediate direction and cannot account for the curvature of the trajectory, so it typically recovers just the average of the possible original samples. Formally, a naive one-step estimate is

xˆ0 = xt − tVθ∗(xt,t,c) ≈ E[x0|xt,c], (4) where θ∗ minimizes Eq. (3).

- 3. Self-Evaluating Model

serve a fundamentally different purpose. Self-consistency methods essentially learn a specific underlying Flow Map [4] or an average velocity [15] along the reverse trajectory, i.e., the integral of local velocities. In contrast, our goal is to directly predict samples whose marginal distribution pθ(xs|c) matches the real distribution q(xs|c), without constraining the reverse transition to follow any particular trajectory.

##### 3.1. Learning from Data

Our model is always trained on real data using the conditional flow matching loss in Eq. (3) which is equivalent to learning expectation of x0 prediction from Gθ(xt,t,s,c) through

We introduce the Self-Evaluating Model, a novel text-toimage pretraining approach enabling flexible, any-step inference. As illustrated in Fig. 2(a), the core idea is simple yet effective: the model simultaneously learns from data while performing self-evaluation. Conceptually, the loss function of our model is formulated as:

0,ϵ ∥Gθ(xt,t,s,c) − x0∥2 . (7)

Ldata(θ) = Es,t,x

where s ≤ t are randomly sampled during training. In particular, when s = t, our model is optimized solely by this loss. The optimally trained Gθ(xt,t,t,c) serves as an estimate of the conditional expectation E[x0|xt,c]. However, the expectation itself may not be a meaningful sample in q(x0|c). Since this supervision is derived from the data distribution, we refer to this process as learning from data.

L(θ) = Ldata(θ) + λLself-evaluate(θ). (5)

The learning-from-data component Ldata(θ) (Sec. 3.1) provides local trajectory supervision, effectively estimating the conditional expectation E[x0|xt,c]. Meanwhile, the selfevaluation component Lself-evaluate (Sec. 3.2) targets global distribution matching, encouraging the model-generated output xˆ0 to be a realistic sample drawn from the true distribution q(x0). We demonstrate that surprisingly, this can be achieved through self-evaluation of the generated images by the model itself.

##### 3.2. Learning by Self-Evaluation

When s < t, we introduce another objective which targets at global distribution matching. We interpret xˆ0 = Gθ(xt,t,s,c) as a sample from an implicit distribution pθ(x0|xt,t,s,c). Our goal is then to ensure that the marginal distribution:

Model Parametrization. Formally, given a noisy input xt, we train a model Gθ(xt,t,s,c) to predict the clean data sample xˆ0 = Gθ(xt,t,s,c), parameterized as:

pθ(xs|c) = q(xs|x0)pθ(x0|xt,t,s,c)q(xt|c)dxtdx0

(8) closely matches the real distribution q(xs|c). To accomplish this, we consider the reverse KL divergence between pθ(xs|c) and q(xs|c):

xˆ0 = Gθ(xt,t,s,c) = xt − tVθ(xt,t,s,c), (6)

where Vθ(xt,t,s,c) denotes a neural network analogous to Vθ from Eq. (3), but noted distinctively as it takes in two time variables. Our two time variables intuitively remind of self-consistency-based models [12, 15, 22, 42], but here they

DKL pθ(xs|c)∥q(xs|c) = Ex

(9)

s∼pθ(xs|c) [log pθ(xs|c) − log q(xs|c)].

The gradient of this KL divergence for per-sample optimization involves the difference between corresponding score functions:

log q(xˆs|c), (10) where we denote xˆs as a sample from pθ(xs|c).

log pθ(xˆs|c) − ∇xˆs

δ(xˆs) = ∇xˆs

Key Observation. Both score functions in Eq. (10) are intractable in practice. Specifically, ∇xˆs

log q(xˆs|c) represents the real-data score, which serves as the key driving force directing the sample towards regions of higher data density. In contrast, ∇xˆs

log pθ(xˆs|c) is termed the fake score, guiding the sample away from its current position and typically preventing mode collapse. To make optimization possible, prior methods use pre-trained diffusion to model real-data score, i.e., distill from a teacher model [58, 63, 64]. We argue that, obtaining a perfect real score is unnecessary; instead, we leverage the currently trained model Gθ(xs,s,s,c) to provide feedback for global distribution matching, which is a self-evaluation process.

Formally, according to Tweedie’s formula [6, 10, 39], the score function is related to the conditional expectation:

αsE[x0|xs,c] − xs σs2

. (11)

∇xs

log q(xs|c) =

Note that our current model Gθ(xs,s,s,c) progressively learns the expectation E[x0|xs,c] from the data, so we can use it to approximate the real score. Although this estimate is not fully accurate before convergence, it can still effectively guide training, since the “student” model itself is also far from converged in the early stages. Moreover, in practice the real score is typically evaluated under classifier-free guidance (CFG), and the reverse KL objective is inherently mode-seeking. Together, these properties provide stronger guidance for model optimization.

Self-Evaluation Score. We now concretely describe how we use the in-training model Gθ(xs,s,s,c), which progressively learns the expectation E[x0|xs,c], to approximate the score terms in Eq. (10) and Eq. (11). In common practice [63– 65], the real score is evaluated within a conditionally sharpened distribution via classifier-free guidance (CFG) [18], defined as:

∇xˆs

log qw(xˆs|c) = ∇xˆs

log q(xˆs|c)

(12)

q(xˆs|c) q(xˆs|ϕ)

+ (ω − 1)∇xˆs

,

log

where ω is a guidance scale, ϕ is a null prompt, denoting the unconditional distribution. By subtracting the fake score and applying appropriate transformations, we rewrite Eq. (10) into two distinct terms: which we call a classifier score term and an auxiliary term, i.e.:

δ(xˆs) = (ω − 1) ∇xˆs

log q(xˆs|ϕ) − ∇xˆs

log q(xˆs|c) Classifier score term

+ ∇xˆs

log pθ(xˆs|c) − ∇xˆs

log q(xˆs|c) Auxiliary term (optional)

###### .

(13)

Empirically, we observe that using only the classifier score term is sufficiently effective and even improves convergence (see Tab. 2). This observation is consistent with prior work [65], which also found classifier scores effective when performing score distillation for 3D generation [38]. Consequently, we omit the auxiliary term and thereby avoid co-training an additional model for the fake score during the early stages of training. Although this setting no longer corresponds to exact distribution matching, it still provides a meaningful learning signal: intuitively, the classifier score encourages the model to generate samples that align with an implicit classifier q(c|x) [18, 65].

The fake score primarily helps prevent mode collapse; in our case, we note that the learning-from-data component can already fulfill this role. When the model is close to convergence, i.e., in later training stages, we can optionally re-introduce the auxiliary term to perform more accurate distribution matching, which helps reduce artifacts (see Fig. 6). Even then, we do not require an additional copy of the model; instead, we simply utilize a specialized prompt to estimate the generated score. We provide more details about this case in the supplementary material.

We now formally describe our practical implementation of the self-evaluation score using the in-training network. The detailed procedure of self-evaluation with only the classifier score is illustrated in Fig. 2(c). We employ two stopgradient forward passes with self-generated samples as input. In particular, we add noise to the generated sample xˆ0: xˆs = αsxˆ0 + σsϵ, and define a pseudo-target as:

xself := sg[xˆ0 − [Gθ(xˆs,s,s,ϕ) − Gθ(xˆs,s,s,c)]], (14)

where sg denotes the stop-gradient operation. Minimizing the mean squared error (MSE) between this pseudo-target and our model’s prediction induces a gradient with respect to xˆs that precisely matches the desired direction, i.e., the classifier score in Eq. (13). We provide the proof in the appendix. Thus, our self-evaluation loss is expressed as:

0,ϵ ∥Gθ(xt,t,s,c) − xself∥2 .

Lself-evaluate(θ) = Et,s,x

(15)

##### 3.3. Final Objective

Our final per-sample objective is a hybrid loss function that combines data-driven supervision with global distribution

matching via self-evaluation. Formally, it is defined as:

Ls,t(θ) = ∥xˆ0 − x0∥22 + λs,t ∥xˆ0 − xself∥22, (16)

where the weight λs,t controls the relative contribution of the self-evaluation term and is given by:

σt αt −

σs αs

. (17)

λs,t =

Note that λs,t = 0 when t = s, in which case the objective reduces to a purely data-driven reconstruction loss.

In practice, large values of λs,t can overpower the datadriven loss, leading to undesired color bias. To mitigate this effect, we introduce an energy-preserving normalization of the effective training target, inspired by Zhang et al. [66], which addresses a similar issue with high CFG values. From the gradient of Eq. (16), the implicit regression target can be expressed as:

x0 + λs,txself 1 + λs,t

. (18)

xtar =

We normalize this target to preserve the energy of the clean sample x0:

x0 + λs,txself ∥x0 + λs,txself∥2

∥x0∥2. (19)

xrenorm =

Empirically, this normalization yields slightly improved visual quality and stability (see Tab. 2). Replacing xtar with xrenorm, our practical per-pair objective becomes:

Ls,t(θ) = ∥xˆ0 − xrenorm∥22. (20) Finally, the overall training loss is obtained by averaging

over all possible timestep pairs:

L(θ) = Es,t [ws,t Ls,t(θ)], (21) where ws,t denotes the sampling weight for each pair (s,t).

##### 3.4. Inference

Our model supports inference with an arbitrary number of steps by iteratively removing noise, similar to diffusion and flow matching models. Given a predefined inference step budget N and a corresponding time scheduler {tk}, where 1 = t1 > t2 > ··· > tN = 0, we sequentially predict a denoising direction at each timestep tk and take a step towards the next timestep tk+1. Formally, each inference step is defined as:

,tk,sk,c). (22)

k − (tk − tk+1)Vθ(xt

###### xt

= xt

k+1

k

By default, we set the target timestep sk to be the next timestep tk+1. Nevertheless, we find that setting the timestep sk to other values in the interval [tk+1,tk] might lead to improved results in some cases. We demonstrate this phenomenon and provide some suggestions to setting this hyperparameter in the supplementary material. We employ energy-preserving classifier-free guidance [66] with ω = 5.

#### 4. Experiment

We conduct two complementary sets of experiments to validate our approach. First, we train a 2B-parameter model with 512×512-resolution images and compare against stateof-the-art text-to-image models spanning the landscape of training paradigms (Sec. 4.1). Second, we perform controlled ablation studies with 0.5B-parameter models under identical training conditions to isolate the contributions of key design choices (Sec. 4.2). In both, we adopt a latent transformer architecture similar to FLUX [2, 11], with minor modifications to accommodate the additional timestep input s, which mirrors the typical handling of timestep input t. Additional details about architecture, data, and hyperparameters are provided in the supplementary material.

##### 4.1. Comparison with Prior Work

We compare our method with several state-of-the-art textto-image approaches spanning the landscape of training paradigms. Most closely related to our setting are fromscratch any-step methods. Since all published works in this category have been demonstrated only on small-scale datasets, such as CIFAR10 [24] and ImageNet [15, 72], we compare our model with the concurrent Transition Models (TiM) [59], which are the first to scale this family of approaches to text-to-image generation. In addition, we include standard flow-matching and diffusion baselines – FLUX.1dev [2], SDXL [37], and SANA-1.5 [61]. Finally, we compare with Latent Consistency Models (LCM) [33], SDXLTurbo [48], and SD3.5-Turbo [47], which employ different distillation methods from the pretrained Stable Diffusion model [40] for few-step sampling. Note that these models, as well as other distillation-based approaches [63, 64], are not trained from scratch and require a pretrained teacher.

Following the evaluation protocol of Deng et al. [7], we report quantitative results on the GenEval benchmark [16]. As shown in Tab. 1, our method consistently outperforms other methods across all inference step counts, achieving notably higher scores overall. In the few-step regime, our model outperforms the second-best method by a large margin. In Fig. 3, we visualize generated images from representative methods across 2, 4, 8, and 50 inference steps. Our approach consistently produces high-quality, detailed, and text-coherent images at all step counts. In the extreme few-step setting (2 steps), FLUX, SANA, and SDXL fail to generate meaningful images, while LCM and TiM produce recognizable objects but suffer from significant degradation in structure and semantic coherence. In contrast, our method yields clear, semantically aligned, and visually detailed results even under this challenging configuration. As the number of inference steps increases to 4 and 8 steps, all methods progressively improve, yet our approach maintains a clear advantage in both detail and text alignment. At 50 steps, our model attains image quality comparable or better than SANA, and SDXL,

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

[Figure 41]

- Figure 3. Qualitative Any-Step Comparison. Generated images from all methods at various inference steps. Our approach consistently produces detailed, semantically accurate, and visually appealing images aligned with textual prompts at all step counts. In extremely few-step scenarios (e.g., 2-step), FLUX, SANA, and SDXL fail to generate recognizable results, while LCM and TiM exhibit semantic and structural degradation. When using more inference steps, all methods improve, but our method retains superior quality, realism, and text alignment. At 50 steps, normal Flow Matching realm, our method is competitive with FLUX, despite FLUX being a much larger model.

while LCM and TiM exhibit saturation artifacts.

##### 4.2. Ablation Studies

To isolate the advantages of our approach over alternatives and to assess the effects of key design choices, we conduct controlled ablation experiments. We use 0.5B-parameter models trained on identical datasets under consistent conditions, enabling direct comparison without confounding factors. All ablations use 256 × 256 resolution and batch size 1024. Other settings follow the setup in Sec. 4.1 and are detailed in the supplementary material.

Comparison with Pretraining Alternatives. Here we compare our approach with two paradigms: standard Flow Matching and native few-step methods. For native few-step methods, we choose to experiment with a second method

of this family – the recently proposed, Inductive Moment Matching (IMM) [72], as our baseline. IMM can be viewed as an extension of trajectory-based models to the distribution level via moment matching. We report GenEval results in Tab. 2 and provide corresponding qualitative comparisons in Figure 4. Our approach consistently outperforms both Flow Matching and IMM across all step counts. In Figure 5, we further plot GenEval scores for our method and Flow Matching throughout training, clearly demonstrating that our approach not only converges to superior performance but also maintains this advantage throughout the entire training.

Design Choices. We further investigate two design choices and analyze their individual effects by training each variant for 100k iterations. In Tab. 2 we present results comparing models trained without energy-preserving target normal-

- Table 1. Quantitative Comparison on GenEval [16]. Our method is consistently SOTA across all step counts and improves monotonically with more steps on GenEval Overall (2→4→8→50: 0.753→0.781→0.785→0.815). Notably, we achieve large margins in the few-step regime (e.g., +0.12 at 2-step over the best prior methods), while remaining the top performer at 8 and 50 Steps.

Steps Method Overall ↑ Single Object ↑ Two Object ↑ Attribute Binding ↑ Colors ↑ Counting ↑ Position ↑

SDXL [37] 0.0021 0.0130 0.0000 0.0000 0.0000 0.0000 0.0000 FLUX.1-Dev [2] 0.0998 0.2969 0.0227 0.0025 0.1835 0.0656 0.0275

LCM [33] 0.2624 0.7937 0.0985 0.0050 0.4761 0.1812 0.0200 SANA-1.5 [61] 0.1662 0.5531 0.0707 0.0075 0.2234 0.1125 0.0030

2 Steps

TiM [59] 0.6338 0.9469 0.7071 0.4375 0.8723 0.4188 0.4200 SDXL-Turbo [48] 0.4622 0.9781 0.3308 0.1500 0.7527 0.4594 0.1025 SD3.5-Turbo [47] 0.3635 0.7125 0.2879 0.1650 0.5691 0.2812 0.1650

Ours 0.7531 0.9812 0.8838 0.5900 0.8218 0.6094 0.6325

SDXL [37] 0.1576 0.5281 0.0758 0.0125 0.2606 0.0437 0.0250 FLUX.1-Dev [2] 0.3198 0.6469 0.2955 0.0550 0.4202 0.2437 0.2575

LCM [33] 0.3277 0.9344 0.1667 0.0150 0.5372 0.2656 0.0475 SANA-1.5 [61] 0.5725 0.9219 0.6313 0.2525 0.6968 0.5125 0.4200

4 Steps

TiM [59] 0.6867 0.9531 0.7601 0.5225 0.9016 0.5031 0.4800 SDXL-Turbo [48] 0.4766 0.9781 0.4040 0.1400 0.7713 0.4562 0.1100 SD3.5-Turbo [47] 0.7194 0.9344 0.8510 0.5650 0.7952 0.5656 0.6050

Ours 0.7806 0.9688 0.9141 0.6250 0.8936 0.6219 0.6600

SDXL [37] 0.3759 0.8812 0.2702 0.0675 0.6569 0.2594 0.1200 FLUX.1-Dev [2] 0.5893 0.8844 0.7298 0.2175 0.7314 0.4625 0.5100

LCM [33] 0.3398 0.9281 0.1818 0.0300 0.5319 0.3094 0.0575 SANA-1.5 [61] 0.7788 0.9812 0.8864 0.5800 0.9202 0.6750 0.6300

8 Steps

TiM [59] 0.7143 0.9656 0.8232 0.5750 0.8936 0.5156 0.5125 SDXL-Turbo [48] 0.4652 0.9688 0.3763 0.1300 0.7500 0.4562 0.1100 SD3.5-Turbo [47] 0.7071 0.9437 0.8232 0.5450 0.8271 0.5312 0.5725

Ours 0.7849 0.9688 0.9141 0.6225 0.8830 0.6688 0.6525

SDXL [37] 0.4601 0.9688 0.4217 0.1300 0.8138 0.3312 0.0950 FLUX.1-Dev [2] 0.7966 0.9781 0.9318 0.5600 0.9096 0.7500 0.6500

LCM [33] 0.3303 0.8938 0.2247 0.0075 0.5319 0.2812 0.0425 SANA-1.5 [61] 0.8062 0.9844 0.9192 0.7175 0.9229 0.7031 0.5900

50 Steps

TiM [59] 0.7797 0.9656 0.8864 0.7300 0.9069 0.6344 0.5550 SDXL-Turbo [48] 0.3983 0.9156 0.2980 0.0700 0.6702 0.3563 0.0800 SD3.5-Turbo [47] 0.6114 0.8656 0.7449 0.4050 0.6995 0.4281 0.5250

Ours 0.8151 0.9875 0.9394 0.6700 0.8910 0.7000 0.7025

ization (i.e. using Eq. (16) instead of Eq. (20)) and models trained with the auxiliary term from Eq. (13) included throughout all training iterations. We observe that target normalization generally improves performance, except in the extreme two-step inference setting, so we adopt this strategy for all our experiments. In contrast, introducing the auxiliary term from scratch significantly degrades performance. Therefore, we rely primarily on the classifier score in the early stages of training, which both reduces computational cost and stabilizes optimization. However, we find that incorporating the auxiliary term in later training stages is beneficial, notably mitigating oversaturated stripe artifacts in two-step generations, as shown in Fig. 6. Consequently, we adopt this hybrid schedule – classifier-score-only early,

with including auxiliary term for refinement later – as our final training strategy in the main experiments.

#### 5. Related Work

Diffusion and Flow Matching. Diffusion models [19, 49, 52, 54] and flow-matching models [1, 26–28] have become two of the most popular frameworks for generative modeling in recent years. These models are trained to learn either a score function or a velocity field that reverses a noising process, transporting samples from the clean data distribution back to a simple prior distribution such as a Gaussian. Both diffusion and flow-matching approaches have been successfully scaled to a wide range of generative tasks, including

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

[Figure 53]

- Figure 4. Controlled Ablation Study. We compare our method to alternative pretraining methods - Flow Matching and IMM. Full prompts appear in supplementary. Our method produces favorable results across all step budgets.

- Table 2. Controlled Ablation Study. We report overall scores on GenEval [16]. The upper block compares our method with two alternative design choices of omitting the target normalization or incorporating the auxiliary term throughout all training steps. Reported after 100K iterations. The bottom block compares our method with alternative pretraining methods - Flow Matching and IMM. Reported after 300K iterations.

Method 2 Steps ↑ 4 Steps ↑ 8 Steps ↑ 50 Steps ↑ 100k Iterations

w/o target norm. 0.5555 0.6156 0.6521 0.7018 w/ aux. term 0.3307 0.4304 0.5153 0.6166 Ours 0.5439 0.6381 0.6819 0.7160

300k Iterations

Flow Matching [26] 0.2523 0.6075 0.7155 0.7311 IMM [72] 0.2617 0.5994 0.7112 0.7472 Ours 0.6097 0.7121 0.7490 0.7543

text-to-image synthesis [2, 5, 11, 37, 40, 71], text-to-video generation [3, 13, 23, 36, 56, 62], and large language modeling [35]. Despite their impressive performance, diffusion and flow-matching models are fundamentally designed to predict local properties of the data distribution. As a result, they typically require many iterative denoising steps to produce high-quality samples, which can pose significant computational challenges during inference.

Accelerating Diffusion/Flow Matching. There has been a rich body of literature focused on reducing the number of denoising steps required by diffusion and flow matching. Training-free approaches typically employ high-order solvers to better approximate the underlying differential equations [9, 21, 31, 32, 41, 67, 69], but these methods still struggle to achieve high-quality samples within ten denoising steps. Another major line of work aims to accel-

[Figure 54]

- Figure 5. Training Progress Comparison. GenEval scores across different inference steps (2, 4, 8, and 50) for our method and Flow Matching over training iterations (from 50k to 300k). Our approach consistently outperforms Flow Matching at all inference steps, indicating its superior effectiveness and robustness.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

- Figure 6. (Left) Models trained only with the classifier score component from Eq. (13) have clear checkerboard artifacts in extreme few-step regime, 2 steps in this example. (Right) Incorporating the auxiliary term from Eq. (13) in later stages of training helps mitigating these artifacts. Results are from our 2B model.

erate diffusion models through distillation. Early distillation techniques train a student model to match the long-step transitions along the trajectory produced by a multi-step teacher [26, 43]. Consistency Models (CMs) and their variants [14, 30, 51, 55] instead learn a direct flow map that transports a noisy input directly to its corresponding clean sample by following the PF-ODE trajectory. Flow-map models further generalize this paradigm by learning mappings between arbitrary pairs of points (s, t) along the PF-ODE trajectory [4, 12, 17, 20, 22, 42, 57, 68]. More recent work, such as TiM [59] and MeanFlow [15], attempts to learn such flow-map models through large-scale pre-training; however, we observe that these techniques remain difficult to scale effectively to text-to-image generation.

Another approach to obtain few-step models is distribution-matching distillation [44–46, 63, 64, 73, 74], where different divergence metrics are employed as training losses and applied to samples at different noise levels to move student generated sample towards teacher’s learned distribution. Our work is inspired by the distribution-matching viewpoint, but differs in that we apply this idea during the pre-training stage of text-to-image models.

#### 6. Conclusion

In this paper, we introduce the Self-Evaluating Model (SelfE), a novel pretraining framework for text-to-image generation capable of flexible, any-step inference entirely from scratch. Departing from prior approaches dependent on pretrained teacher models, Self-E leverages dynamically learned local scores to self-assess generated samples, establishing an internal feedback loop that seamlessly integrates local trajectory learning with global distribution matching. Comprehensive evaluations on the GenEval benchmark demonstrate Self-E’s state-of-the-art performance across diverse inference budgets, particularly excelling in few-step generation scenarios. Furthermore, Self-E’s performance monotonically improves with increased inference steps, indicating its capability to scale from rapid generation to high-quality long-trajectory sampling. We hope Self-E offers a fresh perspective on designing teacher-free pretraining methods for any-step image generation and inspires future work on transferring such self-evaluating models to downstream tasks.

#### References

- [1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022. 7
- [2] Black-Forest-Labs. Flux.1 [dev], 2024. 12B parameter rectified flow transformer, text-to-image. 1, 2, 5, 7, 8, 13
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 8
- [4] Nicholas M. Boffi, Michael S. Albergo, and Eric VandenEijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. Trans. Mach. Learn. Res., 2025, 2024. 1, 3, 8
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 8
- [6] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. Advances in Neural Information Processing Systems, 35:25683–25696, 2022. 4
- [7] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 5
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 1
- [9] Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Genie:

- Higher-order denoising diffusion solvers. Advances in Neural Information Processing Systems, 35:30150–30166, 2022. 8
- [10] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614,

2011. 4

- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 5, 8, 13
- [12] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024. 3, 8
- [13] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025. 8
- [14] Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024. 8
- [15] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 1, 3, 5, 8
- [16] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 5, 7, 8
- [17] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. ArXiv, abs/2403.06807, 2024. 8
- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 7
- [20] Zheyuan Hu, Chieh-Hsin Lai, Yuki Mitsufuji, and Stefano Ermon. Cmt: Mid-training for efficient learning of consistency, mean flow, and flow map models. ArXiv, abs/2509.24526,

2025. 8

- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 8
- [22] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023. 1, 3, 8
- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 8
- [24] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. 5
- [25] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 1

- [26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1, 2, 7, 8
- [27] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [28] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 7
- [29] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023. 1
- [30] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 1, 8
- [31] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022. 8
- [32] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, pages 1–22, 2025. 8
- [33] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2, 5, 7
- [34] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14297–14306, 2023. 1
- [35] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. 8
- [36] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, and Yang You. Open-sora 2.0: Training a commercial-level video generation model in 200k. arXiv preprint arXiv:2503.09642, 2025. 8
- [37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 5, 7, 8
- [38] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 4

- [39] Herbert E Robbins. An empirical bayes approach to statistics. In Breakthroughs in Statistics: Foundations and basic theory, pages 388–394. Springer, 1992. 4
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 5, 8
- [41] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your steps: Optimizing sampling schedules in diffusion models. arXiv preprint arXiv:2404.14507, 2024. 8
- [42] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. ArXiv, abs/2506.14603, 2025. 1, 3, 8
- [43] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 1, 8
- [44] Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. ArXiv, abs/2406.04103, 2024. 8
- [45] Axel Sauer, Dominik Lorenz, A. Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, 2023.
- [46] Axel Sauer, Frederic Boesel, Tim Dockhorn, A. Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. SIGGRAPH Asia 2024 Conference Papers, 2024. 1, 8
- [47] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 5, 7
- [48] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2024. 5, 7

- [49] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 7
- [50] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 15

- [51] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023. 8
- [52] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 7
- [53] Yang Song and Stefano Ermon. Generative Modeling by Estimating Gradients of the Data Distribution, 2020. 1
- [54] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1, 7
- [55] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. 1, 8

- [56] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 8
- [57] Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, Hongsheng Li, and Xiaogang Wang. Phased consistency models. In Neural Information Processing Systems, 2024. 8
- [58] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 4
- [59] Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint arXiv:2509.04394, 2025. 1, 2, 5, 7, 8
- [60] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629,

2024. 1

- [61] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025. 1, 2, 5, 7
- [62] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 8
- [63] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455– 47487, 2024. 1, 4, 5, 8
- [64] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 1, 4, 5, 8
- [65] Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, SongHai Zhang, and Xiaojuan Qi. Text-to-3d with classifier score distillation. arXiv preprint arXiv:2310.19415, 2023. 4
- [66] Kai Zhang, Fujun Luan, Sai Bi, and Jianming Zhang. Ep-cfg: Energy-preserving classifier-free guidance. arXiv preprint arXiv:2412.09966, 2024. 5
- [67] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022. 8
- [68] Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation. ArXiv, abs/2402.19159, 2024. 8
- [69] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpmsolver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502–55542, 2023. 8

- [70] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025. 1
- [71] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 8
- [72] Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. arXiv preprint arXiv:2503.07565, 2025. 1, 5, 6, 8
- [73] Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. ArXiv, abs/2410.14919, 2024. 8
- [74] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. ArXiv, abs/2404.04057, 2024. 8

## Appendix

This appendix provides additional details and results complementing the main paper. In Sec. S.1, we provide proofs showing that our proposed self-evaluation loss correctly induces the desired optimization gradients. We distinguish two scenarios: deriving the classifier-score gradient and deriving the full reverse KL divergence gradient. For the latter, we include additional implementation details on training. Sec. S.2 contains extended information on our training and inference implementation. In Sec. S.3, we present additional experimental results and further discussions regarding the choice of the second timestep input. Prompts corresponding to image examples shown in the main paper are provided in Sec. S.4. Finally, in Sec. S.5, we discuss limitations of our method and propose directions for future work.

#### S.1. Derivation of the Self-Evaluation Loss

Setup. We follow the forward noising in Eq. (1) and the model parameterization in Eqs. (6)–(7) (main paper). Throughout, we use the same network head Gθ as in the main text, and stop-gradient is denoted by sg[·]. All gradients are taken w.r.t. xs that is obtained by re-noising the model prediction xˆ0 as in Sec. 3.2.

Posterior means. By Tweedie’s formula applied to Eq. (1), the (data) conditional and unconditional posterior means, and the (model) conditional posterior mean, satisfy

1 αs

Eq[x0|xs,c] =

1 αs

Eq[x0|xs] =

xs + σs2∇xs

log q(xs|c) ,

xs + σs2∇xs

log q(xs) ,

- Ep

θ

[x0|xs,c] =

1 αs

xs + σs2∇xs

log pθ(xs|c) .

(s.1)

Subtracting the first two lines of (s.1) gives

Eq[x0|xs] − Eq[x0|xs,c]

=

σs2 αs

(∇xs

log q(xs) − ∇xs

log q(xs|c)),

(s.2)

and subtracting the first and third lines yields

Ep

θ

[x0|xs,c] − Eq[x0|xs,c]

=

σs2 αs

(∇xs

log pθ(xs|c) − ∇xs

log q(xs|c)).

(s.3)

- S.1.1. Self-evaluation without auxiliary term

We use the self-evaluation pseudo-target from the main paper,

- Eq. (14), xself := sg x ˆ0 − Gθ(xˆs,s,s,ϕ) − Gθ(xˆs,s,s,c) ,

and the per-sample squared loss (whose expectation over (t,s,x0,ε) gives Eq. (15)):

Lself := ∥xˆ0 − xself∥22 . (s.4)

- Result 1. Under the posterior-mean approximation Gθ(xs,s,s,c) ≈ Eq[x0|xs,c] and Gθ(xs,s,s,ϕ) ≈ Eq[x0|xs], the gradient of (s.4) w.r.t. xˆs is

∇xˆs Lself = ∂xˆ

0 ∂xˆs

⊤

∇xˆ0 Lself =

2 αs

(xˆ0 − xself)

=

2 αs

(Gθ(xˆs,s,s,ϕ) − Gθ(xˆs,s,s,c)) ≈

2 αs

(Eq[x0|xˆs] − Eq[x0|xˆs,c])

=

2σs2 αs2

(∇xˆs

log q(xˆs) − ∇xˆs

log q(xˆs|c)),

(s.5) where the last equality uses (s.2). Hence gradient descent on Eq. (15) moves xˆs in the direction of the classifier score ∇xˆs

log q(c|xˆs).

S.1.2. Self-evaluation with auxiliary term

We optionally add a branch prompted by cfake to estimate the model posterior mean Gθ(xs,s,s,cfake) ≈ Ep

θ

[x0|xs,c].

Define ∆θ(xs,c) := k(Gθ(xs,s,s,ϕ) − Gθ(xs,s,s,c))

+ (1 − k)(Gθ(xs,s,s,cfake) − Gθ(xs,s,s,c)),

(s.6) and the target xself := sg[xˆ0 − ∆θ(xˆs,c)], and the persample squared loss Lself := ∥xˆ0 − xself∥22 .

- Result 2. Proceeding as in (s.5),

2 αs

∇xˆs Lself =

∆θ(xˆs,c) ≈

2 αs

[k(Eq[x0|xˆs] − Eq[x0|xˆs,c])

+ (1 − k)(Ep

[x0|xˆs,c] − Eq[x0|xˆs,c])]

θ

2σs2 αs2

[k(∇xˆs

log q(xˆs|ϕ) − ∇xˆs

log q(xˆs|c))

=

+ (1 − k)(∇xˆs

log pθ(xˆs|c) − ∇xˆs

log q(xˆs|c))],

(s.7) where we used (s.2) and (s.3). Equation (s.7) is proportional to the full ideal vector field in Eq. (13) once we set k = (w − 1)/w. In practice, we set k = 0.9.

Training. To realize Gθ(xs,s,s,cfake) ≈ Ep

[x0|xs,c], we use model samples and reuse the same conditional FM loss as Eq. (7): draw xˆ0 ∼pθ(x0|c) and xˆs ∼pθ(xs|x0,c),

θ

and cfake is constructed by concatenating the phrase ‘fake image’ with the original prompt, and then minimize

Lfake = E ∥Gθ(sg[xˆs],s,s,cfake) − sg[xˆ0]∥22 . (s.8)

In practice we follow the training schedule in Sec. 4 of the main paper: use only the classifier term early, and enable the auxiliary term later to refine artifacts, while keeping the overall objective identical to Eqs. (16)–(21).

#### S.2. Implementation Details

We adopt a latent transformer architecture similar to FLUX [2, 11] for our experiments, with minor modifications to accommodate our new s-input. Specifically, the design of the modules handling s mirrors those handling t.

We employ a 2B-parameter model trained on mixedresolution and varying aspect-ratio text-to-image datasets. Initially, the model is trained at an approximate resolution of 2562 pixels for 500k iterations with a batch size of 1024. Subsequently, we introduce higher-resolution data of 5122 pixels, maintaining a balanced batch proportion (1:1) between the lower-resolution and higher-resolution data, with a total batch size of 768, continuing training until reaching 710k iterations. At iteration 550k, we additionally introduce training with the auxiliary term. We use the Adam optimizer with β1 = 0.9,β2 = 0.95, a learning rate warmup for 1000 iterations, and linearly decay the learning rate from 3×10−4 to 1 × 10−5. For model evaluation, we maintain an exponential moving average (EMA) with a decay rate of 0.9999. Additionally, during training in the self-evaluation forward pass, the conditional branch utilizes the EMA model, while the unconditional branch employs the non-EMA model.

Architecture. We adopt a FLUX-style latent transformer and keep the notation consistent with the main paper: the denoiser’s raw prediction is Vθ(·) and the sample head is Gθ(xt,t,s,c) = xt − tVθ(xt,t,s,c) (cf. Eq. (6)–(7) in the main paper). Our implementation consists of four modules: (a) a VAE, (b) a patchifier, (c) frozen text encoders, and (d) a dual-time denoiser.

(a-b) VAE and patch tokens. We use the FLUX.1-dev auto-encoder with z-channels = 16, and compression factors [1,8,8] for [frames,H,W]. Images are tokenized by a patchifier with patch size [1,2,2]. Thus, each image produces a sequence of Limg = (H/16)×(W/16) tokens, each of dimension dimg = 16 × 2 × 2 = 64.

(c) Text and global conditioning. We use a frozen T5XXL encoder to obtain token embeddings of dimension dtxt = 4096. Additionally, we compute a global pooled CLIP embedding (ViT-L/14) of dimension dvec = 768. Both encoders are kept frozen during training, and their outputs are linearly projected to Rd

model before entering the denoiser.

(d) Denoiser. Our 2B model has a model width dmodel = 2048, head size dhead = 128 (thus 16 heads), and a total of 8 Double-Stream blocks followed by 16 Single-Stream blocks. Positional encoding uses multi-axis RoPE over (t,y,x) with axis dimensions [16,56,56], whose sum matches dhead and whose three axes correspond to time and the two spatial directions. Inputs are linearly projected to dmodel: text via R4096 → R2048, image tokens via R64 → R2048, and the global CLIP vector via a two-layer MLP R768 → R2048. For ablations, we also train a smaller 0.5B variant with dmodel = 1024, dhead = 64, and RoPE axis dimensions [8,28,28], while keeping all other components identical.

Particularly, the denoiser has two time inputs: the primary time t and an auxiliary time s used by the self-evaluation mechanism (Sec. 3.2). In practice, we encode t and the gap t − s with sinusoidal features followed by small MLPs:

et = MLPt(Sinusoid(t)),es = MLPs(Sinusoid(t − s)),

(s.9) and form a combined time embedding

###### e˜t = et + es. (s.10)

This combined embedding e˜t simply replaces the original single-time embedding in the backbone: every module that previously consumed et now receives e˜t. Consequently, the only architectural change relative to FLUX is the additional auxiliary term es added on top of et, while all downstream conditioning and modulation remain unchanged.

Timestep Scheduler. We first sample the primary time t from a logit-normal distribution defined on (0,1):

traw = σ(z), z ∼ N(0,1), (s.11)

where σ(·) denotes the sigmoid function. This raw time is further adjusted by a length-dependent warping function. Specifically, given the latent patch length L, we define a linear shift µ(L) interpolating between 0.5 at length 512 and 1.15 at length 4096, then compute the warped primary time as:

eµ(L) eµ(L) + (1/traw − 1)

. (s.12)

t =

For the secondary time s, we set s = t with probability p = 0.5. For the remaining half of the cases, we sample s uniformly from the interval:

s ∼ U((1 − τ)t, t), (s.13)

where τ is a linear annealing weight, transitioning from 0 to 1 over the first 300,000 training iterations. As a result, the effective lower bound (1 − τ)t decreases gradually from approximately t towards 0 during training. For the weighting function ws,t in Eq. (20), we set it to 1/t2.

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

[Figure 92]

[Figure 93]

[Figure 94]

###### Figure S.1. More results with 2 and 4 steps. We showcase diverse text-to-image results from our model at 2 and 4 inference step counts, demonstrating coherent semantics, strong text alignment.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

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

- Figure S.2. More results with 8 and 50 steps. We showcase diverse text-to-image results from our model at 8 and 50 inference step counts, demonstrating coherent semantics, strong text alignment.

Inference. For inference, we employ an initially linear timestep scheduler with a length-dependent warping func-

tion, same with Eq. (s.12). We use a DDIM-style update with an η-controlled noise level, following Song et al. [50];

setting η = 0 recovers deterministic DDIM, while η = 1 corresponds to the original DDPM ancestral sampling. In our case, we use η = 1.

#### S.3. Additional Experimental Results

- S.3.1. Alternative s-Scheduler

We investigate alternative strategies for selecting the secondary timestep sk during inference, given a transition from tk to tk+1. During training, the selection of sk affects two aspects simultaneously: it determines the noise level for the smoothed data distribution used in the reverse KL divergence, and it specifies the self-evaluation weighting factor λs,t. These dual roles suggest alternative choices for sk

might yield intermediate and potentially improved behaviors. An intriguing direction for future work would be decoupling the dependence between sk and the weighting factor λs,t, making λs,t independently tunable.

We illustrate our empirical observations in Fig. S.3, highlighting two notable special cases:

- 1. When sk = tk, the model utilizes only the flow matching loss. Consequently, its behavior closely resembles standard Flow Matching, performing poorly at very low inference steps but improving significantly with more steps.
- 2. When sk = tk+1, the model excels in few-step generation. However, as the number of inference steps increases (e.g., at 50 steps), we occasionally observe it underper-

forms compared to sk = tk (see the last two examples in Fig. S.3).

Additionally, we explore a special inference setting—onestep generation without classifier-free guidance. As shown in Fig. S.4, we interpolate between sk = tk (represented as s = 1) and sk = tk+1 (represented as s = 0). Both extreme cases fail to yield meaningful images, whereas the midpoint choice s = 0.5 achieves a favorable balance between texture detail and overall image coherence.

- S.3.2. More Results

We present more results at different inference budgets in Fig. S.1 and Fig. S.2.

#### S.4. Prompts of Results

We provide the text prompts used for the qualitative results shown in the main paper.

##### S.4.1. Prompts of Figure 1.

###### 2-step:

• The word “Self-E” appearing faintly through condensation on a train window, blurred landscape passing behind, city lights refracting, cinematic melancholic tone.

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

Figure S.3. Visualization of two special cases for choosing the secondary timestep sk during inference. Top rows: sk = tk, bottom rows: sk = tk+1.

- • Portrait of a wolf under snowfall, frost collecting on its muzzle and fur, visible texture and natural grain, calm expression, photoreal cold-environment realism.
- • An oil painting of a woman with her hair turning into waves, seascape blending with portrait, tactile brushwork, painterly surreal tone.

###### 4-step:

- • A volcano erupting with petals instead of lava, clouds of color drifting across the sky, surreal cinematic beauty.
- • A cat composed of smoke sitting on a rooftop, its form dissolving into the night air, glowing eyes reflecting city lights, detailed cinematic surrealism.
- • A plate of pastries beside a teacup, sunlight highlighting golden crusts, powdered sugar shimmering under a warm

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Figure S.4. One-step generation without classifier-free guidance. We show results of when selecting different s.

glow, photoreal comforting realism. 8-step:

- • Portrait of a jungle guardian with vine tattoos and greengold war paint, wet skin glistening under filtered sunlight, 85mm, macro detail on skin texture, cinematic naturalism.
- • A bison standing in a foggy grassland at dawn, dew on tall grass, sun barely visible through haze, fur glistening with moisture, cinematic atmospheric realism.
- • A cozy cottage built entirely from red and white yarn, knitted walls and woven roof shingles, soft texture visible in each thread, golden sunlight casting gentle shadows, photoreal tactile realism.

###### 50-step:

- • A human face emerging from cracked porcelain, half side smooth and half crumbled revealing crystalline interior, emotional surreal realism.
- • A queen in jeweled crown standing under golden archway, sunlight refracting through gems, detailed embroidery on gown, distant cityscape visible behind, regal photoreal tone, 9:16.
- • A rabbit made of transparent glass jumping across a shallow creek, sunlight refracting rainbow light through its body, ripples and stones visible beneath, forest on both sides, 16:9 photoreal wide scene.
- • A close-up underwater portrait of a woman leaning forward on a large rectangular glowing sign that reads “SelfE,” the sign filling the lower part of the frame like a real physical board. Neon hues of cyan, pink, and gold from the illuminated surface ripple through the clear turquoise water, casting colorful reflections across her face. She smiles brightly, blue eyes open with confidence, freckles and natural skin texture visible under shifting light. Transparent fish swim nearby among coral branches, tiny bubbles rising through the calm cinematic 9:16 scene.
- • A valley full of blooming lupines and daisies, 16:9 panoramic view, rolling hills leading toward mountain

horizon, warm afternoon light highlighting color contrast, photoreal cinematic realism.

##### S.4.2. Prompts of Figure 4.

- • A colorful chalkboard artwork spelling “SELFE” in bright pastel colors—blue, pink, yellow, and green—each letter outlined softly, chalk dust particles floating through air, faint eraser marks around, warm nostalgic classroom atmosphere.
- • A small home bar setup with wine bottles, glass of whiskey half full, sliced lemon on napkin, reflections on wooden counter, photoreal cinematic tone.
- • A cat sleeping on cloud drifting above mountain range, soft pink sunrise illuminating fur, photoreal dreamlike realism.
- • A royal guard in ornate jade armor, sword reflecting sunlight, palace gardens behind full of flowers and fountains, silk banners waving in soft breeze, cinematic elegant realism.

##### S.4.3. Prompts of Figure 5.

- • A high-altitude thunderhead above a wheat plain; sculpted cumulonimbus, sunlit anvil, tiny barn for scale, global contrast, 24mm vastness, dramatic meteorological realism.
- • A house constructed from luminous jelly bricks glowing at night, detailed transparency and refraction, cinematic realism.

#### S.5. Limitations and Future Work

While our method significantly surpasses existing fromscratch training methods in few-step generation, it still has some limitations. Notably, our current approach, although effective in significantly reducing the number of inference steps, cannot fully compete with the quality obtained by 50-step inference when employing extremely few steps (e.g., 1–2 steps). In these cases, the generated images may lack sufficiently sharp details.

Additionally, given that our proposed paradigm fundamentally differs from existing consistency-based methods, it remains at an early stage of exploration. Several critical design choices, such as loss weighting schemes and inference strategies, have not yet been thoroughly optimized. We believe further systematic exploration of these aspects could lead to considerable improvements.

Nonetheless, we emphasize that our method introduces a genuinely novel training paradigm, distinct from the consistency-training family. Empirically, we observe that our method inherently produces robust structure and semantic coherence, exhibiting a clear trend of generating coherent structures first, followed by iterative refinement of details.

Looking forward, we identify several promising avenues for future work:

- 1. Improving training strategies and inference-time scheduling to further enhance generation quality.
- 2. Investigating the efficacy of our approach for downstream task fine-tuning.
- 3. Exploring scalability and potential adaptations of the proposed paradigm to video generative models.
- 4. Extending our method to unconditional generative settings, as the current approach relies on conditional guidance to derive the classifier scores.

