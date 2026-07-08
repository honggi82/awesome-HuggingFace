# arXiv:2509.04394v1[cs.LG]4Sep2025

## Transition Models: Rethinking the Generative Learning Objective

Zidong Wang1,2,*, Yiyuan Zhang1,2,∗,‡, Xiaoyu Yue2,3, Xiangyu Yue1,

Yangguang Li1,†, Wanli Ouyang1,2, Lei Bai2,† 1MMLab, CUHK 2Shanghai AI Lab 3USYD

{wangzd2022, yiyuanzhang.ai}@gmail.com, {xyyue, wlouyang}@ie.cuhk.edu.hk

Code: https://github.com/WZDTHU/TiM

[Figure 1]

### Abstract

GenEval Score ↑

A fundamental dilemma in generative modeling persists: iterative diffusion models achieve outstanding fidelity, but at a significant computational cost, while efficient few-step alternatives are constrained by a hard quality ceiling. This conflict between generation steps and output quality arises from restrictive training objectives that focus exclusively on either infinitesimal dynamics (PF-ODEs) or direct endpoint prediction. We address this challenge by introducing an exact, continuous-time dynamics equation that analytically defines state transitions across any finite time interval ∆t. This leads to a novel generative paradigm, Transition Models (TiM), which adapt to arbitrary-step transitions, seamlessly traversing the generative trajectory from single leaps to fine-grained refinement with more steps. Despite having only 865M parameters, TiM achieves state-of-the-art performance, surpassing leading models such as SD3.5 (8B parameters) and FLUX.1 (12B parameters) across all evaluated step counts. Importantly, unlike previous few-step generators, TiM demonstrates monotonic quality improvement as the sampling budget increases. Additionally, when employing our native-resolution strategy, TiM delivers exceptional fidelity at resolutions up to 4096 × 4096.

Number of Function Evaluations

[Figure 2]

GenEval Score ↑

Resolution

[Figure 3]

GenEval Score ↑

Aspect Ratio

Figure 1. TiM’s superior performance across different NFEs, resolutions, and aspect ratios. On the GenEval [27] benchmark, TiM outperforms Flux.1 models [5, 6] at different NFEs (top, 1024×1024), at higher resolutions (middle, 1024×1024 to 4096 × 4096), and diverse aspect ratios (bottom, 2 : 5 to 5 : 2).

### 1. Introduction

Function Evaluations (NFEs)—approximately proportional to the number of integration steps—leading to increased inference latency and compute cost.

Diffusion models have emerged as the dominant paradigm in visual content generation, producing state-of-the-art results across various domains [9, 20, 35, 52, 54, 83]. They generate samples from noise via iterative denoising, a process that can be formulated as numerical integration of either the reverse-time Stochastic Differential Equation (SDE) or the corresponding Probability-Flow Ordinary Differential Equation (PF-ODE), with related discrete-time solvers also widely used [46, 66, 69]. Despite its effectiveness, iterative denoising often entails a large Number of

In contrast, recent approaches reduce step counts by avoiding explicit multi-step integration. Consistency models [45, 67, 70] impose PF-ODE self-consistency across different noise levels, while distribution-distillation methods [42, 61, 62, 84, 93] train students to approximate teacher distributions with fewer denoising steps. Shortcut [21], FlowMap [7, 57], and MeanFlow [26, 53] learn the average (shortcut) velocity along the flow-matching trajectory via a self-consistency objective. The principle is that a single large step should approximate the integral of

*: Equal contribution. ‡: Project lead. †: Corresponding authors: bailei@pjlab.org.cn, liyangguang256@gmail.com.

all smaller, instantaneous steps. However, by averaging the entire trajectory, they irrevocably discard the fine-grained local dynamics necessary for high-fidelity refinement. This leads to performance saturation—while effective for fewstep generation, it offers no gains from additional sampling budget. Moreover, despite these methods deliver strong few-step results; their compute–quality scaling is typically weaker than that of high-NFE diffusion models: quality gains plateau after only a few steps, and asymptotic performance remains below traditional multi-step diffusion.

Thus, the entire field converges on a fundamental, yet flawed, compromise [26, 31, 44, 70, 84]: models either achieve high fidelity at the cost of computational efficiency (e.g., diffusion models), or they gain efficiency by sacrificing the very dynamics needed for high-fidelity refinement (e.g., few-step models). The root of this dilemma is not architectural, but a learning objective. It stems from a foundational choice in how these models are taught to generate, not the specific components they are built from. This trade-off is a direct and inevitable consequence of the chosen granularity of supervision. On one hand, local training methods that model instantaneous dynamics (such as those consistent with PF-ODEs/SDEs [31, 44, 69]) achieve high accuracy with small step sizes (∆t) and scale well to manystep generation. However, their performance degrades significantly in few-step regimes. On the other hand, finitehorizon training, which learns a direct mapping over a fixed interval (flow-map and consistency models [7, 26, 70]), excels at few-step generation. Yet, these models see diminishing returns from additional intermediate steps unless specifically trained with complex, multi-interval objectives. This reveals a persistent dilemma: objectives that model instantaneous dynamics and those that learn finite-interval mappings each entail inherent limitations. This motivates the question: What is an appropriate learning objective for generative models?

We attempt to answer this question from the following perspectives:

- 1) Diffusion training [46, 47, 91] learns a local PF-ODE

field whose numerical integration is accurate only in the small-step limit ∆t→0. With large steps, the discretization error dominates; therefore, the objective should be flexible in terms of step sizes.

- 2) Few-step objectives supervise an endpoint map but

do not learn a compositional flow: without an approximate semigroup over time, extra steps change the path rather than refine it, causing schedule sensitivity and early saturation. Therefore, the objective requires a consistency along the trajectory, where intermediate steps act as refinements along a single trajectory, rather than deviations onto new ones, which makes the sampler insensitive to step schedules and enables steady quality improvements with more steps.

Consequently, we argue that a generative model should

learn a versatile denoising operator, parameterized by the desired interval ∆t. By learning the transitions between any state xt to a previous state xt−∆t for an arbitrary ∆t, the generative model is no longer approximating a differential equation or a statistic map. Instead, it is learning the solution manifold1 of the generative process itself. This formulation inherently unifies the local and finite-horizon perspectives, yielding a sampler that is both a powerful fewstep generator and a precise, refinable integrator. Since the training objective is to learn the transitions between any state to a previous state, thus, it is named as Transition Models (TiM), which parameterize state-to-state transitions along the PF-ODE trajectory for arbitrary time intervals.

We validate TiM’s effectiveness through extensive experiments on text-to-image and class-guided image generation. As shown in Figure 1, TiM shows superior performance across different NFEs, resolutions, and aspect ratios. On the GenEval [27] benchmark, our compact 865M parameter model, TiM, establishes a new state-of-the-art. It achieves a score of 0.67 with a single function evaluation (1-NFE) and scales to 0.83 at 128-NFE, outperforming billion-scale industrial models including SD3.5-Large [20] (8B) and FLUX.1-Dev [5] (12B).

### 2. Related Work

Diffusion and Consistency Models. Generative modeling has seen two dominant paradigms. Diffusion models [31, 35, 44] iteratively solve a PF-ODE/SDE, achieving high quality but requiring many function evaluations (NFEs). In contrast, Consistency Models [70] learn a direct mapping for few-step generation but suffer from performance saturation and complex training requirements (e.g., pre-training and stabilization [45, 67]). While recent methods like FlowMap [7, 57] and MeanFlow [26, 53] enable training CM-like models from scratch, they inherit the same limitation of stagnating quality with more steps.

To break this impasse, we propose a new learning principle: mastering state transitions over arbitrary time intervals. This transforms the model from a brittle integrator or fixed-endpoint mapper into a robust navigator on the data manifold, preserving few-step efficiency while supporting monotonic refinement by using more steps.

Text-to-Image Generation with Few-steps. Efficient textto-image (T2I) sampling is currently dominated by distillation. These methods fall into two main camps: distribution distillation (e.g., SD-Turbo [61, 62], DMD [84, 85]), which matches the teacher’s output distribution, and trajectory distillation (e.g., LCM [48], PCM [77]), which mimics its generation path. Hybrid methods [55] combine both.

1solution manifold of a PF-ODE is the high-dimensional geometric surface formed by the collection of all possible generative trajectories that lead from noise to data.

Few-step Models:

Transition Models:

[Figure 4]

[Figure 5]

[Figure 6]

###### Diffusion Models:

Local PF-ODE Supervision.

###### Global Endpoint Supervision.

Arbitrary-Interval Supervision.

State-to-State transition:

Statistical mapping:  Consistency Mapping  Distribution Matching

Solving SDE/ODE through:

[Figure 7]

 Arbitrary state interval.

 Score Prediction

 Velocity Prediction

[Figure 8]

 Learning the solution mainfold.

 Data Prediction

[Figure 9]

Few-step sampling to reduce accumulated errors.

[Figure 10]

Multi-step sampling to reduce discretization errors.

[Figure 11]

###### Any-step sampling.

Few-step.

Re-adding Noise

Increasing NFEs to reduce discretization errors.

Multi-step.

Discretization Errors

Accumulated Errors

One-step.

Figure 2. Illustration of Different Generative Paradigms. While conventional diffusion models learn the local vector field and few-step models learn a fixed endpoint map (a single large step), our Transition Models (TiM) are trained to master arbitrary state-to-state transitions. This approach allows TiM to learn the entire solution manifold of the generative process, unifying the few-step and many-step regimes within a single, powerful model.

However, all these approaches are fundamentally limited: 1) they require a large, pre-trained teacher, leading to complex and costly pipelines, and 2) they produce brittle, few-step-only models whose quality stagnates or degrades with more steps. We bypass these limitations entirely by introducing TiM, the first T2I generator trained from scratch that masters arbitrary-step sampling, delivering strong fewstep results that monotonically improve with more compute.

### 3. Transition Models

In this section, we first analyze the limitations of PFODE supervision in diffusion models, which constrain the state transition to a local, infinitesimal interval. To address the limitations, we generalize diffusion’s local state transition to an arbitrary-interval state transition, as illustrated in Fig. 2, from which we we derive a novel mathematical identity that links the state xt, the interval ∆t, and the network fθ. From this identity, we formulate a training objective governing state evolution over any interval ∆t, and further propose two theoretically motivated improvements for scalable and stable training. Finally, we present the architecture improvements for effective transition modeling.

#### 3.1. Limitation of PF-ODE Supervision

Given the noise distribution ε ∼ N(0,I) and the data distribution x ∼ pdata(x), diffusion models learn to map the noise distribution to the data distribution. Given time range t ∈ [0,T], the forward process utilizes coefficients αt and σt, such that xt = αtx + σtε, which can be described by an SDE [69]:

dxt = f(xt,t)dt + g(t)dw, (1)

where w is the standard Wiener process, f(xt,t) = α˙

αt xt is the drift coefficient and g(t) = 2σtσ˙t − 2α˙

t

αt σt2 is the diffusion coefficient [35, 45, 69, 72]. Anderson [3] and Song et al. [69] have shown that the forward process can be reversed by solving the reverse-time SDE from or equivalently the probability flow ODE (PF-ODE)2:

t

dxt dt

- 1

- 2

g(t)2∇xt

= f(xt,t) −

log pt(xt)

(2)

dαt dt

dσt dt

=

x +

ε,

where ∇xt

denotes the score function.

log pt(xt) = −σε

t

Thus, a diffusion model can be parameterized as fθ(xt,t) = Fθ(xt,cnoise(t)), where θ denotes the parameters of the neural network and cnoise(t) is the time scaling function. The training objective can be given by:

Ex,ε,t[w(t)d(fθ(xt,t) − (ˆαtx + σˆtε))], (3)

where αˆt and σˆt are the coefficients of diffusion target, w(t) is a weighting function, d(·,·) is a metric function such as

the L2 loss d(x,y) = ∥x − y∥22.

Despite different transports 3 have instantiate coefficients αt and σt, the training objectives are equivalent to supervising the PF-ODE field 4. During sampling, diffusion models

2Song et al. [69] have shown that the PF-ODE trajectory has the same marginal probability as the reverse-time SDE: dxt = [f(xt, t) −

- 1

- 2g(t)2∇xt log pt(xt)]dt + g(t)dw.
- 3For convenience, we elaborate the coefficients αt, σt, αˆt, and σˆt of

different diffusion transports in Tab. 7

- 4For example, in VE-SDE [69],with coefficients αt = 1, σt = t, the

PF-ODE is: ddxtt = ε, and the training objective is −ε. In OT-FM [43], with αt = 1 − t, σt = t, the PF-ODE is: ddxtt = ε − x, which directly matches the training objective.

solve this PF-ODE, integrated from t = T to t = 0 using numerical solvers. To reduce discretization error and preserve the learned continuous-time dynamics, practical solvers [47, 66, 91] typically require a small step size (i.e., ∆t → 0) or many sub-steps per interval (i.e., high-order solvers), thus inducing huge NFEs.

#### 3.2. State Transition

The derivation begins with the general mathematical form for a state transition between points (xt,xr) on a PF-ODE trajectory, as given in Eq. (6). The central principle is to treat this form not as a numerical approximation, but as an exact identity that must hold for any interval ∆t = t − r. It allows us to formulate a general state transition dynamic (Eq. (8)) that is valid across any interval. Consequently, the model’s training objective is no longer constrained to approximating a local solution of the PF-ODE. Instead, it is trained to learn the entire solution manifold of the generative process. By internalizing this global structure, the model inherently acquires the ability to perform inference over arbitrary step sizes, from large, single leaps to finegrained, iterative refinement. We illustrate our derivation process step-by-step as follows:

State Transition. Given noisy state xt = αtx + σtε, a diffusion model fθ(xt,t) is optimized towards the target αˆtx + σˆtε, leading to the x-prediction and ε-prediction:

αtfθ(xt,t) − αˆtxt σˆtαt − αˆtσt

σˆtxt − σtfθ(xt,t) σˆtαt − αˆtσt

. (4)

, εˆ =

xˆ =

Using the prediction xˆ and εˆ, arbitrary previous state xr (r < t) can be represented as:

xr = αrxˆ + σrεˆ

(5)

(αrσˆt − σrαˆt)xt + (σrαt − αrσt)fθ(xt,t) σˆtαt − αˆtσt

.

=

This represents the general form of a first-order state transition on the PF-ODE Trajectory.

State Transition Identity. Different from the diffusion model, our transition model learns the state transition fθ(xt,t,r) = Fθ(xt,cnoise(t),cnoise(r)) between state xt and xr. By introducing fθ(xt,t,r) to Eq. (5), we obtain:

(αrσˆt − σrαˆt)xt + (σrαt − αrσt)fθ(xt,t,r) σˆtαt − αˆtσt

.

xr =

(6) Here, we define At,r := α

rσˆt−σrαˆt

rαt−αrσt σˆtαt−αˆtσt , and fθ,t,r := fθ(xt,t,r) for simplicity, then we differentiate both sides with respect to t and rearranging yields:

σˆtαt−αˆtσt , Bt,r := σ

d dt

dxr dt

(At,rxt + Bt,rfθ,t,r) =⇒

=

dxt dt

dBt,r dt − Bt,r

dfθ,t,r dt

dAt,r dt

= −fθ,t,r

xt

+ At,r

,

(7)

which can be further simplified as follows (detailed in Appx. A.1):

d(Bt,r · (ˆαtx + σˆtε − fθ,t,r)) dt

= 0 =⇒

d(ˆαtx + σˆtε − fθ,t,r) dt time-slope matching

dBt,r dt

(ˆαtx + σˆtε − fθ,t,r

= 0.

)

+ Bt,r

PF-ODE supervision

(8)

We denote Equation (8) as the State Transition Identity, a

product-derivative invariant. The State Transition Identity, d dt(Bt,r · h(t)) = 0, where h(t) = αˆtx + σˆtε − fθ,t,r is the instantaneous residual, imposes a powerful two-fold

constraint on the generative model fθ.

- • Implicit Trajectory Consistency: The identity dictates that the weighted residual Bt,rh(t) must be constant for any starting time t leading to the same target xr. This directly enforces path consistency: the direct map (t → r) must be equivalent to any composition of intermediate steps, such as (t → s)◦(s → r). This property (Eq. (8) ), absent in standard consistency models, is the core mechanism that makes TiM robust to sampling schedules and enables monotonic refinement.
- • Time-Slope Matching: Unpacking the product rule re-

veals that (ddtBt,r)h(t) + Bt,r(ddth(t)) = 0. Unlike conventional diffusion training, which only minimizes the

residual’s value (h(t) → 0), our objective forces the model to also minimize the residual’s temporal derivative (ddth(t) → 0). This higher-order supervision compels the model to learn a smoother solution manifold, preserving coherence during large-step sampling and ensuring stable refinement with smaller steps.

Derived from State Transition Identity (Eq. (8)), we obtain the learning target fˆ:

dfθ−,t,r dt

Bt,r dBt,r dt

dˆσt dt

dˆαt dt

fˆ = αˆtx + σˆtε +

, (9)

ε −

x +

where θ− indicates the fixed network parameter θ and dfθ−,t,r

dt is the time derivative of the network.

#### 3.3. Scalability and Stability in TiM Training

###### Remark 1: Making TiM Training Scalable.

A critical challenge in implementing our training target (Eq. (9)) is the computation of the network’s time derivative, dfθd−t,t,r . Prior work, such as MeanFlow [26, 53, 57] and sCM [45], relies on the Jacobian-Vector Product (JVP) for this task. However, JVP presents a fundamental roadblock to scalability. It is not only compute-intensive but, more cripplingly, its reliance on backward-mode automatic differentiation is incompatible with essential training optimizations, including FlashAttention [17] and distributed frameworks of FSDP [89]. This incompatibility has effectively rendered JVP-based methods impractical for training billion-parameter foundation models.

Operator Training FID

Method

FLOPs Latency Throughput Memory NFE=1 NFE=8 NFE=50

(G) (ms) (/s) (GiB)

JVP 48.29 213.14 1.80 14.89 49.75 26.22 18.11 DDE 24.14 110.08 2.40 15.23 49.91 26.09 17.99

- Table 1. Derivative Calculation Comparison. We utilize a TiMB/4 model for latency, throughput, and memory measurement, with a batch size of 256 on a NVIDIA-A100 GPU using BF16 precision.

We break this barrier with the Differential Derivation Equation (DDE), a principled and highly efficient finitedifference approximation:

dfθ−,t,r dt ≈

fθ−(xt+ϵ,t + ϵ,r) − fθ−(xt−ϵ,t − ϵ,r) 2ϵ

. (10) As shown in Tab. 1, DDE is not only ∼2× faster than JVP but, crucially, its forward-pass-only structure is natively compatible with FSDP. This compatibility transforms a previously unscalable training process into one ready for largescale deployment, making TiM the first model of its kind practical for from-scratch, billion-parameter pre-training 5. Remark 2: Making TiM Training Stable.

In addition to scalability, a key challenge in training with arbitrary intervals is managing gradient variance. For example, transitions over very large intervals (∆t → t) are easier to make loss spikes. To mitigate this, we introduce a loss weighting scheme that prioritizes short-interval transitions, which are more frequent and provide a more stable learning signal.

The weighting function, w(t,r), is a composition of a time-warping function τ(·) and a kernel function k(·,·):

w(t,r) = k(τ(t),τ(r)). (11)

Here, τ(·) is a monotonic function that re-parameterizes the time axis. For our final model, we use a tangent space transformation, which effectively stretches the time domain, yielding the specific weighting:

- 1

- 2, (12)

w(t,r) = (σdata + tan(t) − tan(r))−

where σdata is the standard deviation of the clean data 6.

Learning Objective. Our theoretical framework culminates in a scalable and stable learning objective. We train the network fθ to predict the dynamic target fˆ in Eq. (9). To manage gradient variance and ensure stable convergence, this is weighted by the interval function w(t,r) from Eq. (11). This results in the final TiM objective:

###### Ex,ε,t,r w(t,r) · d fθ(xt,t,r) − fˆ . (13)

5We provide a detailed analysis of DDE in the Appendix Tab. 10 6In the Appendix, we conduct an in-depth comparison of alternative

weighting schemes is provided in Tab. 12.

This objective generalizes the standard PF-ODE supervision to arbitrary state-to-state transitions. The practical implementation, enabled by our efficient DDE calculation, is detailed in Algorithm 1. We summarize the specific parameterizations for various transport choices in Tab. 7.

#### 3.4. Improved Architectures

We conduct a series of experiments to explore architectural modifications based on DiT [52] for effective state transition learning in Tab. 4. We illustrate the exact architectures in the Appendix A.

Decoupled Time and Interval Embeddings. To enable the model to distinguish between the absolute time t and the transition interval ∆t, we introduce a decoupled embedding strategy. We employ two independent time encoders, ϕt and ϕ∆t, to parameterize these two quantities. Their outputs are summed to form the final time-conditioning vector:

###### Et,∆t = ϕt(t) + ϕ∆t(∆t). (14)

This time embedding is then integrated with task-specific conditioning as follows:

- • For class-guided generation, the class embedding Ec is added to the time embedding, and the resulting sum, Et,∆t + Ec, modulates the AdaLN layers of the model.
- • For text-to-image generation, the conditioning pathways

are separated. The time embedding Et,∆t solely modulates the AdaLN layers, while textual features from the prompt are injected via dedicated cross-attention mechanisms.

Interval-Aware Attention. We assume that the optimal way to model spatial dependencies is conditional on the transition interval ∆t. A large step (∆t → t) may require global, coarse-grained restructuring, while a small step (∆t → 0) demands fine-grained, local refinement. Standard self-attention, which is agnostic to this context, is inappropriate for this task. We therefore introduce the IntervalAware Attention, a mechanism that infuses the transition interval’s magnitude directly into the query, key, and value computations. Specifically, we project both the spatial tokens z and the interval embedding E∆t into a shared representational space before the attention calculation:

q = zWq + bq + E∆tWq′ , k = zWk + bk + E∆tWk′ , v = zWv + bv + E∆tWv′ .

(15)

Here, (Wq,Wk,Wv) are the primary projection matrices for the spatial tokens, while (Wq′ ,Wk′ ,Wv′ ) are dedicated projection matrices that modulate the attention based on the interval embedding.

Model Param. NFE Overall↑ Single Obj. Two Obj. Counting Colors Position Attr. Binding Autoregressive Models

Emu3-Gen [79] - - 0.54 0.98 0.71 0.34 0.81 0.17 0.21 GPT-4o [1] - - 0.84 0.99 0.92 0.85 0.92 0.75 0.61

###### Multi-step Diffusion Models

SD2.1 [56] 865M 100 0.50 0.98 0.51 0.44 0.85 0.07 0.17 SD-XL [54] 2.6B 100 0.55 0.98 0.74 0.39 0.85 0.15 0.23 Seedream2.0 [28] - - 0.84 1.0 0.98 0.91 0.94 0.47 0.75 SD3.5-Medium [20] 2B 100 0.63 0.98 0.78 0.50 0.81 0.24 0.52 SD3.5-Large [20] 8B 128 0.69 0.99 0.89 0.67 0.81 0.24 0.56 SANA-1.5 [81] 4.6B 40 0.81 0.99 0.93 0.86 0.84 0.59 0.65 FLUX.1-Dev [5] 12B 128 0.65 0.98 0.79 0.69 0.76 0.21 0.48

###### Few-step Distilled Diffusion Models

SDXL-LCM [48] 2.6B 8 0.40 0.97 0.50 0.12 0.67 0.09 0.07 SDXL-Turbo [62] 2.6B 8 0.50 0.99 0.75 0.07 0.89 0.11 0.20 Hyper-SDXL [55] 2.6B 8 0.46 0.92 0.58 0.26 0.78 0.11 0.15 SANA-Sprint [16] 1.6B 8 0.72 1.0 0.88 0.56 0.87 0.56 0.47 SD3.5-Turbo [61] 8B 8 0.66 0.99 0.81 0.62 0.79 0.25 0.48 FLUX.1-Schnell [5] 12B 1 0.68 0.99 0.88 0.63 0.78 0.27 0.53

###### Transition Models

1 0.67 0.98 0.75 0.52 0.80 0.54 0.44 TiM 865M 8 0.76 0.99 0.87 0.61 0.88 0.63 0.61

128 0.83 1.0 0.91 0.73 0.91 0.73 0.71

- Table 2. System-level quality comparison of TiM and SOTA methods on GenEval benchmark. In the table, 1-NFE denotes a single sampling step; 8-NFE corresponds to four sampling steps with CFG, and other multi-NFE follow the same convention. Compared with multi-step diffusion models and few-step distilled models, TiM offers any-step generation, delivering strong few-step performance and exhibiting consistent, stable improvements as NFE increases.

MJHQ30K DPGBench

Model Param. NFE

FID↓ CLIP↑ Overall↑ Global Entity Attribute Relation Other

PixArt-α [12] 610M 100 6.14 27.55 71.11 74.97 79.32 78.60 82.57 76.96 PixArt-Σ [14] 610M 100 6.15 28.26 80.54 86.89 82.89 88.94 86.59 87.68 SDXL [54] 2.6B 100 6.63 29.03 74.65 83.27 82.43 80.91 86.76 80.41 Playground v2.5 [39] 2.6B 100 6.09 29.13 75.47 83.06 82.59 81.20 84.08 83.50 Hunyuan-DiT [41] 1.5B 100 6.54 28.19 78.87 84.59 80.59 88.01 74.36 86.41 SD3.5-Medium [20] 2B 100 11.92 27.83 84.08 87.90 91.01 88.83 80.70 88.68 SD3.5-Turbo [61] 8B 8 11.97 27.35 79.03 80.12 86.13 84.73 91.86 78.29 SD3.5-Large [20] 8B 32 14.68 27.88 83.21 84.27 88.99 87.35 93.28 80.35 FLUX.1-Schnell [6] 12B 8 7.94 28.14 84.94 86.62 90.82 88.35 93.45 82.00 FLUX.1-dev [5] 12B 32 9.19 27.27 83.32 81.46 90.02 87.50 92.72 82.39

1 6.68 24.80 74.93 82.98 83.64 83.54 91.99 63.20 TiM 865M 8 5.28 26.10 81.30 82.01 88.31 87.81 93.37 70.80

32 5.65 26.31 82.71 82.67 89.40 88.48 93.31 79.20

Table 3. System-level quality comparison on MJHQ30K and DPGBench benchmarks.

### 4. Experiments

#### 4.1. Setup

We use SD-VAE [56] for ImageNet-256 × 256 experiments and DC-AE [13] for text-to-image (T2I) experiments. Model architecture follows DiT [52], except the modifica-

tions in Sec. 3.4. For T2I generation, we use 33M images from public datasets [2, 11, 15, 18, 33, 63–65]. We train the T2I model with 865M parameters using the nativeresolution training strategies for about 30 days using 16 NVIDIA-A100 GPUs. Gemma3-1B-it [74] is utilized as a text encoder. See more details in Appx. C.2. We report the

Method NFE=1 NFE=8 NFE=50 Training Objective

- (a) Baseline (SiT-B/4 [49]) 309.5 77.26 20.35
- (b) TiM-B/4 (w/ JVP) 49.75 26.22 18.11
- (c) TiM-B/4 (w/ DDE) 49.91 26.09 17.99 Architecture

- (d) Vanilla Architecture 56.22 28.75 20.37
- (e) + Decoupled Time Embedding (De-TE) 49.91 26.09 17.99
- (f) + Interval-Aware Attention (IA-Attn) 48.38 26.10 17.85
- (g) + De-TE + IA-Attn 48.30 25.05 17.43 Training Strategy (on top of (g))

- (h) + Time-weighting 47.46 24.62 17.10

- Table 4. Ablation studies of Transition Models on the standard ImageNet-256 benchmark (FID↓). We analyze the effect of training objectives, architecture, and training strategies.

Method NFE=1 NFE=8 NFE=32 NFE=128

SD3.5-Turbo [61] 0.50 0.66 0.70 0.70 FLUX.1-Schnell [6] 0.68 0.67 0.63 0.58 SD3.5-Large [20] 0.00 0.50 0.69 0.70 FLUX.1-Dev [5] 0.00 0.40 0.64 0.65

TiM 0.67 0.76 0.80 0.83

- Table 5. Benchmarking generation quality across NFEs on the GenEval benchmark (score↑). We compare a single TiM model against diffusion models (i.e., SD3.5-Large and FLUX.1-Dev) and distilled models (i.e., SD3.5-Turbo and FLUX.1-Schnell).

Number of Function Evaluations (NFE) to quantify sampling steps. When classifier-free guidance (CFG) is used, NFE doubles, because each step requires two model evaluations: one conditioned and one unconditioned. We provide the T2I experiments and ablation experiments on ImageNet256 × 256 below and more results on class-guided image generation in Appxs. D.1 and D.2.

Native-Resolution Training. Previous methods [25, 28, 80] have shown the success of native-resolution training on resolution generalization; thus, we adopt this strategy for text-to-image generation, which preserves the original image resolution and aspect ratio information to the greatest extent possible. Given the wide resolution range, we increase noise for higher-resolution images and decrease it for lower-resolution ones. Following Esser et al. [20], we therefore apply resolution-dependent timestep shifting. Please see more details in Appx. C.2.

Sampling. Since TiM learns the arbitrary state transition on the diffusion trajectory, it supports arbitrary-step sampling when producing images. Given a set of timesteps T = {ti}0i=N where tN = T,t0 = 0, we obtain the next state xt

based on Eq. (5), as illustrated in Algorithm 2.

given the current state xt

n−1

n

#### 4.2. Text-to-Image Generation

TiM establishes a new state-of-the-art in performance, efficiency, and flexibility across diverse benchmarks (Tabs. 2,

3, 5 and 6). It achieves an SOTA FID of 5.25 on MJHQ30K while resolving the core speed-quality trade-off. On GenEval, TiM’s 1-NFE performance surpasses 8-NFE distilled models (e.g., SDXL-Turbo), while its 128-NFE quality rivals closed-source models. This unique scalability starkly contrasts with competitors like SD3.5-Large, which collapse at a few steps, and FLUX.1-Schnell, which degrades at many steps. TiM alone shows monotonic quality improvement with NFE. This efficiency is further proven on DPGBench, where 8-NFE TiM outperforms 100-NFE baselines like SDXL. Finally, TiM demonstrates superior generalization across diverse resolutions and aspect ratios, validating its fundamentally more robust design.

#### 4.3. Ablation Studies

We conduct a series of ablation studies to validate our design choices, building from a standard diffusion baseline (SiT-B/4 [49] here) to our final TiM configuration. We use a 131M parameter model trained on ImageNet-256 × 256 for 80 epochs and report FID at 1, 8, and 50 NFEs, corresponding to single-step, few-step, and multi-step generation7. The results are summarized in Table 4.

Transition Objective. As shown in Table 4 (a vs. c), switching from the standard SiT objective to our TiM objective delivers a dramatic improvement in few-step performance, reducing the 1-NFE FID by over 6× (309.5 → 49.91) while maintaining strong many-step quality. This confirms that learning arbitrary transitions is critical for few-step generation. Furthermore, our proposed DDE method (c) achieves this performance while being far more scalable than JVP (b), making large-scale training practical. Architectural Contributions. We next analyze the impact of our architectural innovations on top of a vanilla TiM baseline (d). Both the Decoupled Time Embedding (e) and Interval-Aware Attention (f) individually provide substantial gains across all sampling steps. Crucially, combining them (g) yields the best performance, lowering the 8-NFE FID from 33.08 to 29.21. This demonstrates that enabling the model to explicitly reason about both absolute time and the transition interval is complementary and essential for optimal performance.

Training Strategy. Building on our best architecture (g), we apply our proposed interval weighting scheme. This final step provides a consistent boost across the board (h), further refining the model and achieving our best FID scores of 47.46 / 24.62 / 17.10.

### 5. Conclusion and Limitations

This paper introduces the Transition Models (TiM), a novel generative model that learns to navigate the entire generative trajectory with unprecedented flexibility. The success

71-NFE: single sampling step; 8-NFE: 4 sampling steps with CFG; 50NFE: 25 sampling steps with CFG.

Aspect Ratio Resolution

Method NFE

###### 2 : 5 9 : 16 2 : 3 3 : 2 16 : 9 5 : 2 1280 1536 2048 2560 3072 4096

SD3.5-Turbo [61] 8 ✗ 0.53 0.60 0.58 0.30 ✗ 0.61 ✗ ✗ ✗ ✗ ✗ FLUX.1-Schnell [6] 8 0.57 0.61 0.63 0.62 0.59 0.57 0.64 0.58 0.46 0.14 ✗ ✗ TiM 8 0.55 0.58 0.63 0.64 0.58 0.56 0.70 0.61 0.49 0.48 0.45 0.39

SD3.5-Large [20] 32 0.25 0.48 0.60 0.57 0.16 ✗ 0.63 ✗ ✗ ✗ ✗ ✗ FLUX.1-Dev [5] 32 0.48 0.59 0.62 0.60 0.59 0.57 0.62 0.58 0.49 0.27 ✗ ✗ TiM 32 0.66 0.67 0.72 0.72 0.62 0.64 0.75 0.69 0.63 0.62 0.59 0.53

- Table 6. Benchmarking resolution generation capabilities on GenEval Benchmark. For aspect ratio generalization, the exact resolutions are: {1024 × 2560, 1024 × 1856, 1024 × 1536, 1536 × 1024, 1856 × 1024, 2560 × 1024}. ✗: when GenEval score falls below 0.10, we interpret it as evidence that the model fails to generalize to that resolution.

FLUX.1-Schnell 12B

FLUX.1-Dev 12B

SD3.5-Turbo 8B

SD3.5-Large 8B

SDXL-Turbo 2.6B

TiM (Ours) 865M

SDXL 2.6B

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

NFE=1

A beatiful photos of mountains in the background and a lake with cherry blossoms in the foreground.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

NFE=8

Exotic Shorthair cat on a cloud in heaven, stream, coquettish, immortal, fluffy, shiny fur, petals, Fairy tale.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

NFE=32

Photography, portrait headroom, a 14th Century Alchemist with a short beard and dark eyes wearing tattered gray cloak, devious, powerful, Autumn background.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

NFE=128

Extreme closeup portrait of a wolf in the swedish mountains at sunset life style stock image popular no text prompt trend. pinterest contest winner.

Figure 3. Qualitative Analysis between TiM and existing methods under different NFEs. TiM delivers superior fidelity and text alignment across all NFEs. In contrast, multi-step diffusion and few-step distilled models exhibit pronounced step–quality trade-offs: SDXL, SD3.5-Large, and FLUX.1-Dev fail to generate images at low NFEs, while SDXL-Turbo, SD3.5-Turbo, and FLUX.1-Schnell produce over-saturated outputs at high NFEs.

of our compact 865M model in outperforming multi-billion parameter giants is not just a new state-of-the-art; it is a testament to a more efficient and powerful paradigm. By achieving monotonic quality improvement from one step to many, and scaling to ultra-high resolutions, TiM demonstrates that a unified model is not only possible but superior. We believe this work paves the way for a new generation of foundation models that are at once efficient, scalable, and promising in their creative potential.

Limitations. Although TiM delivers a significant contribution to the fundamental generative models, ensuring content safety and controllability remains an open challenge, and model fidelity can degrade in scenarios requiring fine-grained detail, such as rendering text and hands. We also observe occasional artifacts at high resolutions (e.g., 3072 × 4096), likely attributable to biases in the underlying autoencoder.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 6

- [2] adebyollin. Megalith-huggingface. https://huggingface.co/datasets/madebyollin/megalith-10m. 6
- [3] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313– 326, 1982. 3
- [4] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023. 23
- [5] black-forest labs. Flux.1-dev. https://huggingface. co/black-forest-labs/FLUX.1-dev, . 1, 2, 6, 7, 8
- [6] black-forest labs. Flux.1-schnell. https : / / huggingface . co / black - forest - labs / FLUX.1-schnell, . 1, 6, 7, 8
- [7] Nicholas Matthew Boffi, Michael Samuel Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. Transactions on Machine Learning Research, 2025. 1, 2
- [8] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 22, 23
- [9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. Accessed: 2024-5-1. 1

- [10] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 22
- [11] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In CVPR,

2021. 6

- [12] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 6
- [13] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024. 6, 19, 21
- [14] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024. 6

- [15] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 6
- [16] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025. 6
- [17] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023. 4
- [18] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431,

2021. 6

- [19] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021. 21, 23
- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. 2024. 1, 2, 6, 7, 8, 20
- [21] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024. 1, 17, 19, 22
- [22] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024. 22, 23
- [23] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389, 2023. 22
- [24] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389,

2023. 22

- [25] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025. 7
- [26] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 1, 2, 4, 17, 19, 22
- [27] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 1, 2
- [28] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025. 6, 7

- [29] Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. Diffit: Diffusion vision transformers for image generation. arXiv preprint arXiv:2312.02139, 2023. 23
- [30] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 21
- [31] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2, 16, 20
- [32] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. 2023. 22, 23
- [33] jackyhate. text-to-image-2m. https://huggingface.co/datasets/jackyhate/text-to-image2M. 6
- [34] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10124–10134, 2023. 22
- [35] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. NeurIPS, 2022. 1, 2, 3, 15, 16, 17, 20
- [36] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024. 16, 17, 20, 23
- [37] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023. 17, 19
- [38] T. Kynk¨a¨anniemi, T. Karras, S. Laine, and T Lehtinen, J.and Aila. Improved precision and recall metric for assessing generative models. NeurIPS, 2019. 21
- [39] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 6
- [40] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 22
- [41] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 6
- [42] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 1
- [43] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3, 13, 16, 19, 20

- [44] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 2, 16, 20
- [45] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 1, 2, 3, 4, 13, 15, 16, 17, 20
- [46] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775– 5787, 2022. 1, 2
- [47] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, pages 1–22, 2025. 2, 4
- [48] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2, 6
- [49] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740,

2024. 7, 22, 23

- [50] C. Nash, J. Menick, S. Dieleman, and P. W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 21
- [51] Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T Freeman, and Yu-Xiong Wang. Randar: Decoder-only autoregressive visual generation in random orders. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 45–55, 2025. 22
- [52] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 5, 6, 22, 23
- [53] Yansong Peng, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. Flow-anchored consistency models. arXiv preprint arXiv:2507.03738, 2025. 1, 2, 4, 17
- [54] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 6
- [55] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. Advances in Neural Information Processing Systems, 37:117340–117362, 2025. 2, 6
- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 6, 21, 22
- [57] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603, 2025. 1, 2, 4
- [58] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X Chen. Improved techniques for training gans. NeurIPS, 2016. 21

- [59] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, 2022. 23
- [60] Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022. 22
- [61] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 1, 2, 6, 7, 8
- [62] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2024. 1, 2, 6

- [63] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 6
- [64] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. 2018.
- [65] Vasu Singla, Kaiyu Yue, Sukriti Paul, Reza Shirkavand, Mayuka Jayawardhana, Alireza Ganjdanesh, Heng Huang, Abhinav Bhatele, Gowthami Somepalli, and Tom Goldstein. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328, 2024. 6
- [66] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1, 4
- [67] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In The Twelfth International Conference on Learning Representations. 1, 2, 17, 22
- [68] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 2019. 16
- [69] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1, 2, 3, 13, 14, 15, 16, 20
- [70] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252. PMLR, 2023. 1, 2, 17, 18
- [71] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 22
- [72] Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025. 3, 13, 15
- [73] Zhicong Tang, Jianmin Bao, Dong Chen, and Baining Guo. Diffusion models without classifier-free guidance. arXiv preprint arXiv:2502.12154, 2025. 20

- [74] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 6
- [75] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 22, 23
- [76] Fu-Yun Wang, Zhengyang Geng, and Hongsheng Li. Stable consistency tuning: Understanding and improving consistency models. arXiv preprint arXiv:2410.18958, 2024. 23
- [77] Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in Neural Information Processing Systems, 37:83951–84009, 2025. 2, 17, 19
- [78] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Exploring dcn-like architecture for fast image generation with arbitrary resolution. Advances in Neural Information Processing Systems, 37:87959–87977, 2024. 22, 23
- [79] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 6
- [80] Zidong Wang, Lei Bai, Xiangyu Yue, Wanli Ouyang, and Yiyuan Zhang. Native-resolution image synthesis. arXiv preprint arXiv:2506.03131, 2025. 7
- [81] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025. 6
- [82] Wanghan Xu, Xiaoyu Yue, Zidong Wang, Yao Teng, Wenlong Zhang, Xihui Liu, Luping Zhou, Wanli Ouyang, and Lei Bai. Exploring representation-aligned latent space for better generation. arXiv preprint arXiv:2502.00359, 2025. 22
- [83] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1
- [84] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 1, 2
- [85] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in Neural Information Processing Systems, 37: 47455–47487, 2025. 2
- [86] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh

- Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 22, 23
- [87] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 22, 23
- [88] Xiaoyu Yue, Zidong Wang, Zeyu Lu, Shuyang Sun, Meng Wei, Wanli Ouyang, Lei Bai, and Luping Zhou. Diffusion models need visual priors for image generation. arXiv preprint arXiv:2410.08531, 2024. 22
- [89] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, ChienChin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023. 4, 20
- [90] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. 2023. 23
- [91] Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpmsolver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502–55542, 2023. 2, 4
- [92] Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. arXiv preprint arXiv:2503.07565, 2025. 22
- [93] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024. 1

### Appendix

We include additional derivations, experimental details, and results in the appendix.

- • In Appx. A, we provide a detailed formula derivation of the State Transition Identity, training, and sampling algorithms.
- • In Appx. B, we discuss TiM’s relationships with existing methods, including diffusion models, consistency models, and other training approaches.
- • In Appx. C, we provide the implementation details of text-to-image generation, including native-resolution training, resolution-dependent timestep shifting, model-guidance training, and from-scratch training.
- • In Appx. D, we provide additional ablation results and results on class-guided image generation.
- • In Appx. E, we provide more qualitative results of TiM. A. Transition Model Framework

In this section, we first provide the derivation of the TiM identity equation Eq. (8). Then we provide the training and sampling algorithms. Finally, we provide a systematic analysis of the connections with other existing methods.

#### A.1. TiM Identity Equation Derivation

We demonstrate the derivation from Eq. (7) to the TiM identity equation Eq. (8). We start from the detailed expansion of each term of Eq. (7). Firstly, we have:

xt = αtx + σtε, (16) dxt dt

dαt dt

dσt dt

ε, (17)

=

x +

where dxt

dt is the PF-ODE of diffusion and Eq. (17) has already been proved in previous works [43, 45, 69, 72]. For At,r = αrσˆt−σrαˆt σˆtαt−αˆtσt , Bt,r = σ

rαt−αrσt

σˆtαt−αˆtσt , we have: dAt,r dt

dAt,r dαt ·

dAt,r dσt ·

dAt,r dˆαt ·

dAt,r dˆσt ·

dαt dt

dσt dt

dˆαt dt

dˆσt dt

(18) dBt,r dt

=

+

+

+

dBt,r dαt ·

dBt,r dσt ·

dBt,r dˆαt ·

dBt,r dˆσt ·

dαt dt

dσt dt

dˆαt dt

dˆσt dt

(19)

=

+

+

+

We use Ct,r = σˆtαt − αˆtσt for simplicity, which is the denominator of At,r and Bt,r. For Eq. (18) and Eq. (19), each term is calculated as:

 

 

rαˆt−αrσˆt)ˆσt

rσˆt−σrαˆt)σt

dAt,r

dBt,r

(ˆσtαt−αˆtσt)2 = −σˆtACt,r

(ˆσtαt−αˆtσt)2 = σtACt,r

dαt = (σ

dαt = (α

t,r dAt,r

t,r dBt,r

rσˆt−σrαˆt)ˆαt

rαˆt−αrσˆt)αt

(ˆσtαt−αˆtσt)2 = αˆtACt,r

(ˆσtαt−αˆtσt)2 = −αtACt,r

dσt = (α

dσt = (σ

. (20)

;

t,r dAt,r

t,r dBt,r

rσt−σrαt)σt

rαt−αrσt)σt

(ˆσtαt−αˆtσt)2 = −σˆtBCt,r

(ˆσtαt−αˆtσt)2 = σtBCt,r

dˆαt = (α

dˆαt = (σ





t,r dAt,r

t,r dBt,r

tσr−αrσt)αt

rσt−σrαt)αt

(ˆσtαt−αˆtσt)2 = αˆtBCt,r

(ˆσtαt−αˆtσt)2 = −αtBCt,r

dˆσt = (α

dˆσt = (α

t,r

t,r

Substituting Eq. (20) into Eq. (18) and Eq. (19), we have:

dAt,r dt

At,r Ct,r ·

dαt dt

At,r Ct,r ·

dσt dt − σˆt

Bt,r Ct,r ·

dˆαt dt

Bt,r Ct,r ·

dˆσt dt

, (21) dBt,r dt

= −σˆt

+ αˆt

+ αˆt

At,r Ct,r ·

dαt dt − αt

At,r Ct,r ·

dσt dt

Bt,r Ct,r ·

dˆαt dt − αt

Bt,r Ct,r ·

dˆσt dt

. (22)

= σt

+ σt

There exists some symmetry between the above two equations, which is the key to our TiM identity. Combining Eqs. (16),

(17) and (21), we have:

dxt dt

dAt,r dt

xt

+ At,r

dαt dt

= (At,r

+ αt

dAt,r dt

dσt dt

)x + (At,r

+ σt

dAt,r dt

)ε. (23)

The coefficient of x in the above equation can be decomposed as:

dαt dt

dAt,r dt

At,r

+ αt

dAt,r dαt

dαt dt

=(At,r + αt

)

+ αt

At,r Ct,r ·

dαt dt

At,r Ct,r ·

= − αˆtσt

+ αˆtαt

dBt,r dt

= − αˆt

+ (ˆαtσt − σˆtαt)

dBt,r dt − Bt,r

dˆαt dt

= − αˆt

.

dAt,r dσt ·

dσt dt

dAt,r dˆαt ·

+ αt

dσt dt − σˆtαt

Bt,r Ct,r ·

dˆαt dt

Bt,r Ct,r ·

αˆt dt

dˆαt dt

dAt,r dˆσt ·

dˆσt dt

+ αt

Bt,r Ct,r ·

dˆσt dt

+ αˆtαt

Similarly, the coefficient of ε in the Eq. (23) can be decomposed as:

(24)

dσt dt

dAt,r dt

At,r

+ σt

dAt,r dαt ·

dαt dt

=σt

+ (At,r + σt

At,r Ct,r ·

dˆσt dt

At,r Ct,r ·

= − σˆtσt

+ σˆtαt

dBt,r dt

= − σˆt

+ (ˆαtσt − σˆtαt)

dBt,r dt − Bt,r

dˆσt dt

= − σˆt

.

dAt,r dσt

dAt,r dσt ·

dσt dt

+ σt

)

dˆσt dt − σˆtσt

Bt,r Ct,r ·

Bt,r dt ·

dˆσt dt

dAt,r dˆαt ·

dˆαt dt

dAt,r dˆσt ·

dˆσt dt

+ σt

dˆσt dt

Bt,r Ct,r ·

dˆσt dt

+ αˆtσt

Substituting Eqs. (23) to (25) into Eq. (7), we have:

(25)

dxt dt

dBt,r dt

dfθ,t,r dt

dAt,r dt

+ At,r

+ fθ,t,r

+ Bt,r

= 0.

xt

dBt,r dt − Bt,r

dˆαt dt

dBt,r dt − Bt,r

dˆσt dt

dBt,r dt

dfθ,t,r dt

⇒(−αˆt

)x + (−σˆt

)ε + fθ,t,r

+ Bt,r

= 0.

dBt,r dt

dˆαt dt

dˆσt dt −

dfθ,t,r dt

⇒(ˆαtx + σˆtε − fθ,t,r)

+ Bt,r(

+

) = 0

d(ˆαtx + σˆtε − fθ,t,r) dt

dBt,r dt

⇒(ˆαtx + σˆtε − fθ,t,r)

+ Bt,r

= 0

d(Bt,r · (ˆαtx + σˆtε − fθ,t,r)) dt

⇒

= 0.

This is the TiM identity equation in Eq. (8), the proof is completed.

(26)

#### A.2. TiM Training Algorithm

We provide the detailed training algorithm of TiM in Algorithm 1. It is noteworthy that the TiM models are entirely trained from scratch.

#### A.3. TiM Sampling Algorithms

We provide the TiM sampling algorithm in Algorithm 2. For multi-step sampling, we can further incorporate stochasticity into the sampling process for improved diversity. In multi-step scenarios, the TiM sampling is similar to the diffusion sampling process, but with a new condition for the next step. Therefore, we can construct a stochastic sampling from the SDE (stochastic differential equation) diffusion process. Given xt = αt + σtε,Song et al. [69] has shown that the SDE forward and reverse are:

forward : dxt = f(xt,t) + g(t)dw, reverse : dxt = [f(xt,t) −

(27)

- 1

- 2

g(t)2∇xt

log pt(xt)]dt + g(t)dw.

- Algorithm 1 Training Algorithm of Diffusion Transition Models (TiM).

Input: dataset D with standard deviation σd, model fθ, diffusion parameterization {αt,σt,αˆt,σˆt}, weighting w∆t, learning rate η, time distribution T , constant ϵ, constant c.

Init: Iters ← 0 repeat

xd ∼ D, x = cdata(xd), ε ∼ N(0,I), r < t ∼ T , xt ← αtx + σtε Bt,r ← (σrαt − αrσt)/(ˆσtαt − αˆtσt) dfθ−

dt = 21ϵ(fθ−(xt+ϵ,t + ϵ,r) − fθ−(xt−ϵ,t − ϵ,r)) ▷ DDE Calculation fˆ ← αˆtx + σˆtε + (dˆα

t

dt · x + dˆσ

t

dt · ε − df

θ−

dt ) · Bt,r/dBdt,rt ▷ TiM Target

L(θ) ← ∥fθ − fˆ∥22 + Lcos(fθ,fˆ) L(θ) ← w∆t · L(θ)/(∥L(θ)∥ + c) θ ← θ − η∇θL(θ)

Iters ← Iters + 1 until convergence

- Algorithm 2 Piecewise Sampling Algorithm of Diffusion Transition Models (TiM).

Input: sampling step N, miximum timestep Tmax, model fθ, diffusion parameterization {αt,σt,αˆt,σˆt}, stochasticity ratio ρ.

Init: data xN ∼ N(0,I), timesteps T = {ti}0i=N where tN = Tmax,t0 = 0 for i = N to 1 do

+ σ

i+1σˆti−σti+1αˆti σˆtiαti−αˆtiσti xt

= αt

ti+1αti −αti+1

σti

,ti,ti+1) if ρ > 0 then:

σˆtiαti−αˆtiσti f(xt

###### xt

i

i

i+1

εˆ ← αt

,ti,t0) − αˆt

σˆtiαtiαˆtiσti fθ(xt

σˆtiαti−αˆtiσti xt

i

i

i

i

εi ∼ N(0,I) dt = ti − ti+1 xt

√

σt′

− αt′

σ′

ti − α′

i+1 ← xt

i+1 − ρ(αt

)εˆdt − 2ρ(αt

σt

tiσt

)εi

dt

i

i

i

i

i

i

end if xi = xt

i+1

###### end for

Previous works[35, 45, 69, 72] has provided the explicit form of f(xt,t), g(t) and ∇xt

log pt(xt):

α˙t αt

α˙t αt

ε σt

xt, g(t) = 2σtsigma˙ t − 2

σt2, ∇xt

log pt(xt) = −

f(xt,t) =

where α˙t and σ˙t represent the derivation of αt and σt respectively. For PF-ODE, it is defined as:

, (28)

- 1

- 2

dxt dt

g(t)2∇xt

log pt(xt) = α˙tx + σ˙tε. (29) For reverse-SDE, it is defined as:

= f(xt,t) −

vt =

dxt = [f(xt,t) − g(t)2∇xt

log pt(xt)]dt + g(t)dw

- 1

- 2

- 1

- 2

g(t)2∇xt

g(t)2∇xt

= [f(xt,t) −

dt−

log pt(xt)]

log pt(xt)dt + g(t)dw

PF-ODE Term

Stochastic Term

α˙t αt

α˙t αt

σt2dw.

= vtdt + [˙σt −

σt]εdt + 2σtσ˙t − 2

(30)

In the TiM sampling, we can take the stochastic term in the above equation to enhance diversity. To balance the stochasticity and stability, we incorporate a scaling factor s(t) = ραt, leading to a scaled g˜ = ραtg(t) = 2ρ(αtσtσ˙t − 2α˙tσt2). Therefore, the stochastic term is: ρ[αtσ˙t − α˙tσt]εdt + 2ρ(αtσtσ˙t − 2α˙tσt2)dw.

Diffusion Parameterization Transition Parameterization cnoise(t) = αt = σt = αˆt = σˆt = dˆα

Transport

dt = Bt,r = dBdt,rt = OT-FM [43, 44] t 1 − t t −1 1 0 0 r − t −1 TrigFlow [45] t cos(t) sin(t) −sin(t) cos(t) −cos(t) −sin(t) sin(r − t) −cos(r − t) EDM [35, 36] 14 ln(t) t 1

dt = dˆσ

t

t

− σ

Eq. (35) Eq. (36) Eq. (37) Eq. (38) VP-SDE [31, 69] (T − 1)t β21

√ t

t σd

√

√

d

2+σd2

t2+σd2

t2+σd2

t2+σd2

√r−βt

√βt

0 1 0 0 β

· dβ

√−1

t

dt VE-SDE [68, 69] ln(12t) 1 t 0 −1 0 0 t − r 1

t +1

βt2+1

βr2+1

βr2+1

Table 7. Transition parameterization for different diffusion transports. For VP-SDE, T is set to 1000, and βt = e21βdt2+βmint − 1, where βd = 19.9 and βmin = 0.1 by default. We provide the details of timestep sampling in Tab. 8. Song et al. [69] has shown that VP-SDE is equivalent to DDPM [31] while VE-SDE is equivalent to score matching [68], so we adopt their notations for uniformity. For EDM, its TiM parameterization is too complex; we provide them in Eqs. (35) to (38).

### B. Connections with Existing Methods

In this section, we highlight the connection between TiM and other existing methods. We first demonstrate the properties of TiM compared with diffusion models. Then we demonstrate the connections of TiM with other training strategies.

Transport Noise Level Timestep Time Range Time Scaling

OT-FM ln(σ) ∼ N(Pmean,Pstd2 ) t = 1+σσ t ∈ [0,1] cnoise(t) = t Trigflow ln(σ) ∼ N(Pmean,Pstd2 ) t = arctan(σσ

###### ) t ∈ [0, π2] cnoise(t) = t EDM ln(σ) ∼ N(Pmean,Pstd2 ) t = σ t ∈ [0,+∞) cnoise(t) = 14 ln(t) VP σ ∼ U(ϵt,1) t = σ t ∈ [ϵt,1] cnoise(t) = (T − 1)t VE σ ∼ U(ϵt,1) t = σmax(σ

d

2 min

###### σmax2 )σ t ∈ [σmin,σmax] cnoise(t) = ln(12t)

Table 8. Time distribution of diffusion diffusion transports.

#### B.1. Connections with Diffusion Models

TiM generalizes the standard diffusion models. As a complement to Tab. 7, we elucidate the time distribution of different diffusion transports in Tab. 8. Our TiMs share these parameters with diffusion models, but learn a different objective. We show that the TiM training objective Eq. (13) generalizes the standard diffusion objective Eq. (3). Specifically, the TiM identity equation reduces to the diffusion identity equation in the limit as t → r. Recall that Bt,r = σ

rαt−αrσt

σˆtαt−αˆtσt , the training target of TiM when t → r becomes:

dfθ−,t,r dt

Bt,r dBt,r dt

dˆσt dt · ε −

dˆαt dt · x +

fˆ = lim

lim

α ˆtx + σˆtε +

t→r

t→r

= αˆtx + σˆtε + lim t→r

= αˆtx + σˆtε + lim t→r

= αˆtx + σˆtε + lim t→r

= αˆtx + σˆtε.

dfθ−,t,r dt

Bt,r dBt,r dt

dˆαt dt · x +

dˆσt dt · ε −

σrαt−αrσt σˆtαt−αˆtσt dBt,r dt

dfθ−,t,r dt

dˆαt dt · x +

dˆσt dt · ε −

0

dBt,r dt

dfθ−,t,r dt

dˆσt dt · ε −

dˆαt dt · x +

(31)

The above target is the diffusion target. This target lacks the modeling of state transitions from state to state, thus limiting the arbitrary-step generation capabilities of diffusion models.

EDM parametrization. EDM [35, 36] parameterizes the diffusion model as:

σd2 t2 + σd2

t · σd t2 + σd2

Dθ(x + tε,t) =

Fθ

(x + tε) +

x + tε

,

t2 + σd2

1 4

ln(t) . (32)

2+σd2 t2σd2 , leading to training objective as:

It adopts the x-prediction in its training and use time weighting w(t) = t

t2 + σd2 t2σd2 ∥Dθ(x + tε,t) − x∥22

L(θ) =

t2 + σd2 t2σd2

=

σd2 t2 + σd2

t · σd t2 + σd2

(x + tε) +

Fθ

= Fθ

###### x + tε

,

t2 + σd2

1 4

t σd t2 + σd2

ln(t) − (

1 4

x + tε

ln(t) − x

,

t2 + σd2

2

σd t2 + σd2

x −

ε)

2

2

2

(33)

Therefore, let cnoise(t) = 41 ln(t), the original EDM parameterization can be unified into our parameterization with the following coefficients:

1 t2 + σd2

t t2 + σd2

t σd t2 + σd2

αt =

, σt =

, αˆt =

Therefore, the TiM parameterization is defined as:

σd t2 + σd2

σˆt = −

(34)

t2 σd(t2 + σd2)23

dˆαt dt

1 σd t2 + σd2

, (35) dˆσt dt

= −

+

tσd (t2 + σd2)32

, (36)

=

(t − r)σd2 t2 + σd2 (t2 + σd3) r2 + σd2

, (37)

Bt,r =

t(t − r)(t2 + σd3) − 2t(t − r)(t2 + σd2) + (t2 + σd2)(t2 + σd3) (t2 + σd3)2 t2 + σd2 r2 + σd2

dBt,r dt

= σd2

. (38)

#### B.2. Connections to Other training Methods

In this section, we discuss the connections of TiM with other training strategies, including continuous-time consistency models [45, 70], consistency trajectory models [37], phased consistency models [77], Shortcut models [21], and MeanFlow models [26, 53].

Continuous-time consistency models. The TiM objective Eq. (13) generalizes the continuous-time consistency models. Specifically, the CTM objective reduces to the continuous-time CM objective when r = 0. For TiM, let r = 0 and d(x,y) = ∥x − y∥22, the training objective becomes:

  fθ(xt,t,0) − (ˆαtx + σˆtε +

 

2

dfθ−,t,0 dt

Bt,0 dBt,0 dt

dˆσt dt · ε −

dˆαt dt · x +

∇θEx,ε,t

2

(39)

dfθ−,t,0 dt −

dˆαt dt · x −

dˆσt dt · ε .

Bt,0 dBt,0 dt

=Ex,ε,t [∇θfθ,t,0]T fθ−,t,0 − αˆtx − σˆtε +

Continuous-time consistency models [45, 67, 70] are trained to map the noisy input xt directly to the clean data x in one or a few steps. Given model Fθ, the consistency models are formulated as:

###### Dθ(xt,t) = cskip(t)xt + cout(t)Fθ(cin(t)xt,cnoise(t)). (40)

Using the parameters αt, σt, αˆt, and σˆt, consistency parameterization corresponds to the transition from xt to x0:

σˆtxt − σtFθ(cin(t)xt,cnoise(t)) σˆtαt − αˆtσt

, (41)

Dθ(xt,t) =

σˆtαt−αˆtσt = At,0 and − σ

where σˆt

σˆtαt−αˆtσt = Bt,0 correspond to TiM parameterizations.

t

When using loss function d(x,y) = ∥x − y∥22, [70] show that the gradient of continuous-time consistency models is:

dDθ−(xt,t) dt

t,t DθT(xt,t)

∇θEx

dDθ−(xt,t) dt

t,t [Bt,0fθcm(xt,t)]T

=∇θEx

(42)

dDθ−(xt,t) dt

t,t Bt,0[∇θfθcm(xt,t)]T

=Ex

dfθcm− dt

dAt,0 dt

xt dt

dBt,0 dt

t,t Bt,0∇θ[fθcm(xt,t)]T

fθcm− + Bt,0

=Ex

xt + At,0

+

,

where fθcm(xt,t) = Fθ(cin(t)xt,cnoise(t))) represents the network in consistency models. As xt = αtx + σtε, dx

dt = dαt

t

dt x + σ

dtε, we have:

t

dfθcm− dt

dAt,0 dt

dBt,0 dt

xt dt

fθcm− + Bt,0

xt + At,0

+

dαt dt

dσt dt

dAt,0 dt

(αtx + σtε) + At,0(

x +

z) +

=

dAt,0 dt

dαt dt

dAt,0 dt

=(αt

+ At,0

)x + (σt

+ At,0

dfθcm− dt

dBt,0 dt

fθcm− + Bt,0

dfθcm− dt

dBt,0 dt

dσt dt

fθcm− + Bt,0

)z +

.

(43)

Based on Eqs. (24) and (25), we have:

dAt,0 dt

dαt dt

αt

+ At,0

dBt,0 dt − Bt,0

dˆαt dt

dAt,0 dt

= −αˆt

, σt

dσt dt

dBt,0 dt − Bt,0

dˆσt dt

= −σˆt

+ At,0

Therefore, we have:

dfθcm− dt

dAt,0 dt

xt dt

dBt,0 dt

fθcm− + Bt,0

xt + At,0

+

dfθcm− dt −

dˆαt dt

dˆσt dt

dBt,0 dt

(fθcm− − αˆtx − σˆtz).

x −

=Bt,0(

) +

Substituting the above equation into Eq. (42), the gradient of continuous-time consistency models is:

. (44)

(45)

dDθ−(xt,t) dt

t,t DθT(xt,t)

∇θEx

dfθcm− dt −

dˆαt dt

dˆσt dt

dBt,0 dt

(fθcm− − αˆtx − σˆtz) ,

t,t Bt,0[∇θfθ]T Bt,0(

=Ex

x −

) +

(46)

dfθcm− dt −

dBt,0 dt

Bt,0 dBt,0 dt

dˆαt dt

dˆσt dt

)[∇θfθ]T fθcm− − αˆtx − σˆtz +

=Ex

x −

t,t (Bt,0

(

) ,

Note that TiM network fθ(xt,t,0) corresponds to the network fθcm(xt,t) in consistency models. The only difference between Eq. (46) and Eq. (39) is a term (Bt,0dBdtt,0), which can be bridged by a weighting function.

Consistency trajectory models, phased consistency models, and shortcut models. These models learn to transition from one state to another state in a discrete manner, while our TiM generalizes this to the continuous-time domain. The core of our method is the TiM identity equation Eq. (8), which determines the function for the transition between two arbitrary states.

For consistency trajectory models (CTM) [37] and phased consistency models (PCM) [77], they targets at intermediate state xr, where 0 ⩽ r ⩽ tn−1, thus leading to the identity equation:

,tn−1,r), (47)

Ψ(xt

,fθ(xt

,tn,r),r) = Ψ(xt

,fθ(xt

n−1

n−1

n

n

where 0 < t1 < ··· < tn < ··· < tN = T represents the discrete timesteps, and Ψ is an ODE solver to obtain the state at timemstep r. It is noteworthy that PCM splits the entire trajectory into several segments and learns this identity on each segment independently.

Shortcut models [21] adopts the OT-flow-matching [43] as the transport, the ODE solver is: Ψ(xt,fθ(xt,t,r),r) = xt − (t − r)fθ(xt,t,r). The original identity equation is: (t − r)fθ(xt,t,r) = (t − s)fθ(xt,t,s) + (s − r)fθ(xs,s,r). This identity equation of shortcut models can be rearranged as:

(t − r)fθ(xt,t,r) = (t − s)fθ(xt,t,s) + (s − r)fθ(xs,s,r)

=⇒xt − (t − r)fθ(xt,t,r) = xt − (t − s)fθ(xt,t,s) − (s − r)fθ(xs,s,r)

=⇒xt − (t − r)fθ(xt,t,r) = xs − (s − r)fθ(xs,s,r)

=⇒Ψ(xt,fθ(xt,t,r),r) = Ψ(xs,fθ(xs,s,r),r),

where s = t+2r. Based on Eq. (47) and Eq. (48), when tn−1 = t

n+r

2 , CTMs are equivalent to shortcut models.

(48)

MeanFlow models. We show that the TiM Eq. (8) generalizes the MeanFlow [26]. In particular, the training objective of TiM reduces to the MeanFlow objective in the OT-FM [43] transport setting.

As in Tab. 7, OT-FM uses the parameterization {αt = 1−t,σt = t,αˆt = −1,σˆt = 1}, leading to the TiM parameterization {Bt,r = r − t, dBdt,rt = −1}. Therefore, the TiM training objective becomes:

dfθ−,t,r dt

. (49)

Ex,ε,t d fθ(xt,t,r) − z − x − (t − r)

This corresponds to the training objective of MeanFlow.

### C. Implementation Details

#### C.1. Model Architecture

We illustrate the model architecture in Fig. 4. As we incorporate decoupled time embedding and interval-aware attention designs into DiT architecture, we use LoRA-AdaLN to avoid increasing model size. Specifically, given attention hidden size D, LoRA rank is set as r = 31D, such as D = 1152 and r = 384 in XL-models. For text-to-image generation, we incorporate caption features through CrossAttention mechanism, as in Fig. 5.

##### C.2. Text-to-Image Training Details Native-Resolution Training. We adopt the VAE-specific native resolution training for text-to-image generation. As we

use DC-AE [13] with 32 downsampling scale, an image with shape H × W is resized to shape (32 · ⌊32H ⌋) × (32 · ⌊W32⌋). For example, an image with shape 1025 × 513 is resized to 1024 × 512, , preserving resolution and aspect ratio as much as

possible. Images of the same size are grouped into resolution buckets for batching. We set the base batch size as 16 on a single GPU for 1024 × 1024 resolution bucket, then for H × W resolution bucket,

the minimal batch size is B = ⌊102416×H×1024×W ⌉. For instance, the 512 × 512 resolution bucket holds the minimal batch size as B = 64, while the 2048 × 2048-resolution bucket holds the minimal batch size as B = 4. The maximum resolution is

4096 × 4096 with B = 1. For data parallelism, each device processes distinct buckets with their corresponding batch sizes, maintaining a similar token budget.

Resolution-Dependent Timestep Shifting. Sampling from a single timestep distribution is suboptimal across resolutions ranging from less than 256 × 256 to 4096 × 4096 pixels. Intuitively, higher-resolution images require stronger corruption (more noise) to destroy the signal, while lower-resolution images require less noise. Given an image with n = H1 × W1

× N TiM-T2I Block

###### × N

Prediction

Feed Forward

- 1
- 2, 2

- 2
- 3, 3

- 3

2

Scale

Linear & Reshape

Multi-head Self-Attn

Feed-Forward

Cross-Attn Block

2, 2

Embed

Q, K, V

Scale, Shift

⊕

Layer Norm

Layer Norm

LoRA AdaLN

Self-Attn Block

LLM Decoder 1, 1

Linear Linear

TiM Block

1

Δt Emb

Scale

Tokens

MHSA

A magpie with a glossy black head, wings, and tail, and a white belly, perches on a wooden post of a wire fence.

Patcify Embed

Embed

Patcify

1, 1

- Linear-1

- Linear-2

Scale, Shift

Rank r

Layer Norm

[Figure 42]

LoRA AdaLN

Timestep & Class label

Noisy Data

Timestep & Interval

Cond

Tokens Cond

Caption Noisy Data

Figure 4. TiM Model Architecture.

###### Figure 5. TiM T2I block.

pixels and its high-resolution counterpart with m = H2 × W2 pixels, Esser et al. [20] provides an equation to map the the timestep tn to tm:

- m

- n tn

. (50)

tm =

1 + ( mn − 1)tn

In our practice, we set the base pixel number as n = 1024 × 1024, and apply this mapping to all sampled timesteps.

Model-Guidance Training. Tang et al. [73] propose a model-guidance training target to improve sampling fidelity. We adopt this approach for text-to-image training. Under our formulation, the target becomes:

Bt,r dBt,r dt

fˆ = αˆtx + σˆtε − fθuncond∗,t,t ) +

dfθ−,t,r dt

dˆσt dt · ε −

dˆαt dt · x +

+ (ω − 1)(fθcond∗,t,t, (51)

where ω denotes the Classifier-Free Guidance (CFG) scale, θ∗ is the Exponential Moving Average (EMA) of θ, fθcond∗,t,t and fθuncond∗,t,t respectively represent the conditional and unconditional outputs.

From-Scratch Training. The TiM-T2I model contains 865M parameters with the patch size of 1. We train from scratch for about 30 days across 16 NVIDIA-A100 GPUs with a constant learning rate of 4 × 10−4, using PyTorch-FSDP [89] and half-precision (torch.bfloat16) for memory efficiency. Following Tang et al. [73], we use model-guidance target Eq. (51) with CFG scale ω = 1.75 after 100K iterations.

### D. Additional Results

Transport NFE=1 NFE=8 NFE=50

OT-FM [43, 44] 49.91 26.09 17.99 TrigFlow [45] 67.32 25.14 18.28 EDM [35, 36] 53.64 37.01 24.06

VP-SDE [31, 69] 78.98 37.44 35.72

Table 9. Ablation Studies on different transports.

Method ϵ Speed 1-step 4-step 50-step JVP n.a. 1.8 iter/s 49.75 26.22 18.11

- DDE 0.0001 2.4 iter/s 111.25 23.34 18.38

- DDE 0.0002 2.4 iter/s 80.14 23.83 17.58 DDE 0.0005 2.4 iter/s 67.09 24.33 16.93

- DDE 0.001 2.4 iter/s 48.83 24.73 17.03

- DDE 0.002 2.4 iter/s 49.07 25.54 17.59 DDE 0.005 2.4 iter/s 49.91 26.09 17.99

- DDE 0.01 2.4 iter/s 50.05 26.53 18.33

- DDE 0.02 2.4 iter/s 49.72 26.67 18.33 DDE 0.05 2.4 iter/s 49.90 27.05 18.79

t = r r = 0 1-step 4-step 50-step

0% 0% 52.08 31.52 24.85 0% 10% 53.46 32.49 25.74

10% 10% 51.74 29.75 22.56 20% 10% 50.98 28.41 20.74 30% 10% 50.09 27.01 19.20 40% 10% 49.88 26.42 18.54 50% 10% 47.46 24.62 17.10 60% 10% 48.29 24.55 16.58 70% 10% 48.44 24.05 16.32 80% 10% 48.26 22.88 15.34

Table 10. The impacts of DDE ϵ on generation performance.

Table 11. Timestep sampling comparison.

transform(t) = t transform(t) = t/(1 − t) transform(t) = tan(t)

Weighting

NFE=1 NFE=8 NFE=50 NFE=1 NFE=8 NFE=50 NFE=1 NFE=8 NFE=50

- (a) Reciprocal 48.25 25.29 17.42 56.65 24.33 16.58 49.65 25.22 17.39

- (b) SMS 49.01 25.76 17.75 72.56 25.15 17.01 48.93 25.23 17.19

- (c) Sqrt 48.24 25.82 17.87 49.93 24.73 16.85 47.46 24.62 17.10

- (d) Square 48.55 25.31 17.11 70.83 25.79 17.80 48.86 24.91 17.15

Table 12. Ablation studies on different time weighting functions.

#### D.1. Additional Ablations

We provide additional ablation results in this section.

Transport Comparison. We conduct ablation studies on different transports in Tab. 9. We find that different transports affect the convergence speed, where OT-FM and TrigFlow perform best, EDM is slightly worse, and VP-SDE performs the worst. Thus, we adopt OT-FM in all experiments.

Differential Derivation Equation Calculation. As we incorporate a small quantity ϵ into Eq. (10) to calculate the time derivative of network. We systematically evaluate the impact of ϵ on numerical accuracy in Tab. 10and observe that ϵ ∈ [0.001,0.01] yields high precision. For training stability, we adopt ϵ = 0.005 in all experiments.

Timestep Sampling. Using the TiM-B/4 model, we observe improved performance when a portion of timesteps is fixed to t = r, as in Tab. 11. The best results are obtained with 50% of samples using t = r (diffusion training) and 10% using r = 0 (consistency training).

Time Weighting. Using the TiM-B/4 model, we provide a systematic analysis of time weighting as in Eq. (11). We study three types of transformations: (1) transform(t) = t, (2) transform(t) = 1−t t, (3) transform(t) = tan(t); and four types of weighting functions: (1) Reciprocal: w fn(t,r) = σ 1

d+t−r, (2) Soft-Min-SNR (SMS): w fn(t,r) = σ2 1

d+(t−r)2,

d+t−r)2, where σd is the standard deviation of clean data x (σd = 1 in our dataset). Empirically, the combination w(t) = w(t,r) = √ 1

d+t−r, (4) Square: w fn(t,r) = (σ 1

(3) SQRT: w fn(t,r) = √σ 1

achieves the best performance, slightly surpassing the best results reported in Tab. 4.

σd+tan(t)−tan(r)

##### D.2. Class-Guided Image Generation We provide the results of class-guided image generation in this section.

Setup. We use SD-VAE [56] for ImageNet-256 × 256 and DC-AE [13] for ImageNet-512 × 512, with patch sizes of 2 and 1, respectively. We train an XL-model with 664M parameters for 750K iterations with a batch size of 512 (300 epochs), using a constant learning rate of 2 × 10−4 and AdamW optimizer. We report FID [30], sFID [50], Inception Score (IS) [58], Precision and Recall [38] using ADM evaluation suite [19].

BigGAN [8] - 112M 1 6.95 7.36 171.4 0.87 0.28 StyleGAN-XL [60] - 166M 1 2.30 4.02 265.12 0.78 0.53 GigaGAN [34] - 569M 1 × 2 3.45 - 225.52 0.84 0.61

Masked and Autoregressive Models Mask-GIT [10] 555 - - 6.18 - 182.1 - MagViT-v2 [86] 1080 307M - 1.78 - 319.4 - LlamaGen-XL [71] 300 775M - 2.62 5.59 244.08 0.81 0.58 LlamaGen-XXL [71] 300 1.4B - 2.34 5.97 253.90 0.81 0.59 LlamaGen-3B [71] 300 3.1B - 2.18 5.97 263.3 0.81 0.58 VAR [75] 350 2.0B - 1.80 - 365.4 0.83 0.57 MAR [40] 800 943M - 1.55 - 303.7 0.81 0.62 RandAR-XL [51] 300 775M - 2.22 - 314.21 0.80 0.60 RandAR-XXL [51] 300 1.4B - 2.15 - 321.97 0.79 0.62

Multi-step Diffusion Models LDM-4-G [56] 170 395M 250 × 2 3.60 5.12 247.67 0.87 0.48 SimpleDiffusion [32] 800 2B 250 × 2 2.44 - 256.3 Flag-DiT-3B∗ [22] 200 4.23B 250 × 2 1.96 4.43 284.8 0.82 0.61 Large-DiT-3B∗ [22] 340 4.23B 250 × 2 2.10 4.52 304.36 0.82 0.60 MDT [23] 1300 676M 250 × 2 1.79 4.57 283.01 0.81 0.61 MDTv2 [24] 700 676M 250 × 2 1.63 4.45 311.73 0.79 0.65 DiT-XL [52] 1400 675M 250 × 2 2.27 4.60 278.24 0.83 0.57 SiT-XL [49] 1400 675M 250 × 2 2.06 4.49 277.50 0.83 0.59 FlowDCN-XL [78] 400 675M 250 × 2 2.00 4.37 263.16 0.82 0.58 SiT-REPA-XL [87] 800 675M 250 × 2 1.42 4.70 305.7 0.80 0.65 DoD-XL [88] 300 613M 250 × 2 1.73 5.14 304.31 0.79 0.64 SiT-RealS-XL [82] 400 675M 250 × 2 1.82 4.45 268.54 0.81 0.60

###### Few-step Consistency Models

MeanFlow-XL† [26] 1000 675M 1 3.43 - - - iCT-XL [67] - 675M 1 × 2 20.30 - - - Shortcut-XL [21] 250 675M 1 × 2 10.60 - - - -

4 × 2 7.80 - - - -

IMM-XL [92] 3840 675M 1 × 2 7.77 - - - 2 × 2 3.99 - - - 4 × 2 2.51 - - - -

Any-step Transition Models TiM-XL 300 664M 1† 3.26 4.37 210.33 0.75 0.59

1 7.11 4.97 140.39 0.71 0.63

- 1 × 2 6.14 6.21 151.79 0.74 0.59

- 2 × 2 3.61 6.74 189.99 0.79 0.58 4 × 2 2.62 5.57 203.41 0.79 0.60

250 × 2 1.65 5.02 248.12 0.79 0.63

Table 13. Performance comparison on ImageNet-256 × 256 class-guided generation. ∗: Flag-DiT-3B and Large-DiT-3B actually have 4.23 billion parameters, where 3B means the parameters of all transformer blocks. †: means using model-guidance in the training, therefore eliminating the usage of CFG.

Performance Analysis. We provide the results on ImageNet-256 × 256 and ImageNet-512 × 512 in Tabs. 13 and 14 respectively. Across both ImageNet-256×256 and ImageNet-512×512, TiM-XL demonstrates strong performance-efficiency

BigGAN [8] - 160M 1 7.5 - 152.8 - StyleGAN-XL [59] - 168M 1 × 2 2.41 4.06 267.75 0.77 0.52

###### Masked and Autoregressive Models

VAR [75] 1080 307M - 2.63 - 303.2 - MAGVITv2 [86] 350 2.3B - 1.91 - 324.3 - -

Multi-step Diffusion Models SimpleDiffusion [32] 800 2B 250 × 2 3.02 - 248.7 - DiffiT [29] - 561M 250 × 2 2.67 - 252.12 0.83 0.55 MaskDiT [90] 800 - 250 × 2 2.50 5.10 256.27 0.83 0.56 Large-DiT-3B [22] 368 4.23B 250 × 2 2.52 5.01 303.70 0.82 0.57 ADM-G,U [19] 400 774M 250 × 2 3.85 5.86 221.72 0.84 0.53 U-ViT-H/2 [4] 400 501M 250 × 2 4.05 6.44 263.79 0.84 0.48 DiT-XL [52] 600 675M 250 × 2 3.04 5.02 240.82 0.84 0.54 EDM2-L [36] 1468 778M 64 × 2 1.87 - - - EDM2-XL [36] 1048 1.1B 64 × 2 1.80 - - - EDM2-XXL [36] 734 1.5B 64 × 2 1.73 - - - SiT-XL [49] 600 675M 250 × 2 2.62 4.18 252.21 0.84 0.57 FlowDCN-XL [78] - 675M 250 × 2 2.44 4.53 252.8 0.84 0.54 SiT-REPA-XL [87] 200 675M 250 × 2 2.08 4.19 274.6 0.83 0.58

Few-step Consistency Models sCT-L [76] 1273 778M 1 5.15 - - - -

2 4.65 - - - sCT-XL [76] 1117 1.1B 1 4.33 - - - -

2 3.73 - - - sCT-XXL [76] 762 1.5B 1 4.29 - - - -

2 3.76 - - - -

###### Any-step Transition Models TiM-XL 300 664M 1 5.07 4.29 160.06 0.79 0.59

- 1 × 2 4.79 4.36 171.73 0.82 0.57

- 2 × 2 4.01 4.22 171.51 0.83 0.58 4 × 2 2.55 4.24 207.07 0.83 0.57

250 × 2 1.69 4.66 247.52 0.82 0.62

Table 14. Performance comparison on ImageNet-512 × 512 class-guided generation.

trade-offs: at low NFE (1 to 4 × 2), it can compete with few-step consistency models, achieving comparable FID with fewer training epochs and smaller model size. When increasing NFEs, TiM-XL matches or surpasses many multi-step diffusion models in FID, despite training for only 300 epochs. Notably, TiM demonstrates remarkable generation quality and shows stable gains as NFE increases.

### E. Qualitative Results

We provide the qualitative results in Fig. 6.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

###### 4096x4096

1536x1024 2560x1024

An oil-on-canvas masterpiece that captures the dynamic essence of a blue night sky, bursting with the energy of swirling azure hues and speckled with stars that seem to explode in shades of yellow. Dominating the celestial scene is a luminous, fuzzy-the canvas. Below this cosmic display, a tranquil village is depicted to the right, its quaint houses huddled in repose. On the left, a towering cypress tree stretches upwards, its branches undulating like flames, creating a stark contrast against the serene sky. In the distance, amidst the gentle roll of blue hills, the spire of a church stands tall, a silent sentinel overlooking the sleepy hamlet.

knight in full silver armor with a red scarf towering over the camera, sunset in the background, fantasy, mountains, clouds seeming to protude from his figure, photorealistic

realistic, male, surfer, 80s, chill, relaxed, stoner, aviators, beach, illustration, shaggy hair, round face, round chin, brown hair, photorealistic

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

1024x1280 1024x2560 1024x1536

A close-up image of an intricately designed lotus flower, which appears to be crafted entirely from crystal-clear water droplets. The flower is set against a backdrop of soft green lily pads floating on a tranquil pond. Sunlight filters through the scene, highlighting the delicate texture and the shimmering surface of the water-formed petals.

a puppy and a young cat in a cozy room, close up photorealistic image with high details. Picture shows sun flares with warm light

portrait of wolf, lots of colour, pen and soft watercolour in style of sarah taylor art

[Figure 55]

[Figure 56]

1024x4096

fantasy, a majestic sky filled with stars and galaxies, over looking a serene lake.

Figure 6. High-resolution and multi-aspect generations from TiM (128 NFEs). TiM attains up to 4096 × 4096 resolution and reliably handles multiple aspect ratios, including 1024 × 4096 and 2560 × 1024.

