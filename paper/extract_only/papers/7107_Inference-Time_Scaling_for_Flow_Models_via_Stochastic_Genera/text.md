## Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing

#### Jaihoon Kim∗ Taehoon Yoon∗ Jisung Hwang∗ Minhyuk Sung KAIST

###### {jh27kim,taehoon,4011hjs,mhsung}@kaist.ac.kr

# arXiv:2503.19385v5[cs.CV]24Oct2025

“Five horses, three cars, one train, five airplanes.”

|NFEs : 10 NFEs : 100 NFEs : 200 NFEs : 300 NFEs : 400 NFEs : 500<br><br>FLUX + Ours<br><br>|Stable Diffusion<br><br>NFEs : 2500|
|---|---|
||[Figure 1]<br><br>[Figure 2]|
|---|
<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|[Figure 8]|

2

RSS : 27 RSS : 14 RSS : 10 RSS : 5 RSS : 1 RSS : 0 RSS : 36

Logical

Counting Comparison

“The arcade machine is bigger than the television but smaller than the refrigerator.”

“Every painting in the gallery is framed and hung straight, except for one that is hanging crooked.”

“Two cups, three paintings, four lamps, and four bananas decorated the art studio.”

|[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

|[Figure 13]<br><br>[Figure 14]|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

Spatial Relation

Aesthetic Concept Erasure

“A large suitcase is placed beside an open closet, with a folded jacket resting on top where a pair of shoes sit side by side in front of it.”

(+) “A nurse” (−) “Stethoscope, hat, mask”

“Duck”

|[Figure 18]<br><br>[Figure 19]|
|---|

|[Figure 20]<br><br>[Figure 21]|
|---|

|[Figure 22]<br><br>[Figure 23]|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

Figure 1: Diverse applications of our inference-time scaling method. Pretrained flow models struggle to generate images that align with complex prompts (left side of each case), whereas our inference-time scaling effectively extends their capabilities to achieve precise alignment (red box).

### Abstract

We propose an inference-time scaling approach for pretrained flow models. Recently, inference-time scaling has gained significant attention in LLMs and diffusion models, improving sample quality or better aligning outputs with user preferences by leveraging additional computation. For diffusion models, particle sampling has allowed more efficient scaling due to the stochasticity at intermediate denoising steps. On the contrary, while flow models have gained popularity–offering faster generation and high-quality outputs–efficient inference-time scaling methods used for diffusion models cannot be directly applied due to their deterministic generative process. To enable efficient inference-time scaling for flow models, we propose three key ideas: 1) SDE-based generation, enabling particle sampling in flow models, 2) Interpolant conversion, broadening the search space, and 3) Rollover Budget Forcing (RBF), maximizing compute utilization. Our experiments show that SDE-based generation and variance-preserving (VP) interpolant-based generation, improves the performance of particle sampling methods for inference-time scaling in flow models. Additionally, we demonstrate that RBF with VP-SDE achieves the best performance, outperforming all previous inference-time scaling approaches. Project page: flow-inference-time-scaling.github.io.

*Equal contribution.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

### 1 Introduction

Over the past years, scaling laws of AI models have mainly focused on increasing model size and training data. However, recent advancements have shifted attention toward inference-time scaling [57, 59], leveraging computational resources during inference to enhance model performance. OpenAI o1 [42] and DeepSeek R1 [11] exemplify this approach, demonstrating consistent output improvements with increased inference computation. Recent research in LLMs [41] attempting to replicate such improvements has introduced test-time budget forcing, achieving high efficiency with limited token sampling during inference.

For diffusion models [54, 56], which are widely used for generation tasks, research on inference-time scaling has been growing in the context of reward-based sampling [23, 30, 53]. Given a reward function that measures alignment with user preferences [26] or output quality [50, 31], the goal is to find the sample from the learned data distribution that best aligns with the reward through repeated sampling. Fig. 1 showcases diverse applications of inference-time scaling using our method, enabling the generation of faithful images that accurately align with complex user descriptions involving objects quantities, logical relationships, and conceptual attributes. Notably, naïve generation from text-toimage models [48, 28] often fails to fully meet user specifications, highlighting the effectiveness of inference-time scaling.

Our goal in this work is to extend the inference-time scaling capabilities of diffusion models to flow models. Flow models [32] power state-of-the-art image [15, 28] and video generation [7, 72], achieving high-quality synthesis with few inference steps, enabled by trajectory stratification techniques during training [35]. Beyond just speed, recent pretrained flow models, equipped with enhanced text-image embeddings [46] and advanced architectures [15], significantly outperform previous pretrained diffusion models in both image and video generation quality.

Despite their advantages in generating high-quality results more efficiently than diffusion models, flow models have an inherent limitation in the context of inference-time scaling. Due to their ODE-based deterministic generative process, they cannot directly incorporate particle sampling at intermediate steps, a key mechanism for effective inference-time scaling in diffusion models. Building on the formulation of stochastic interpolant framework [1], we adopt an SDE-based sampling method for flow models at inference-time, enabling particle sampling for reward alignment.

To further expand the exploration space, we consider not only stochasticity but also the choice of the interpolant. While typical flow models use a linear interpolant, diffusion models commonly adopt a Variance-Preserving (VP) interpolant [56, 18]. Inspired by this, for the first time, we incorporate the VP interpolant into the particle sampling of flow models and demonstrate its effectiveness in increasing sample diversity, enhancing the likelihood of discovering high-reward samples.

We emphasize that while we propose converting the generative process of a pretrained flow model to align with that of diffusion models—i.e.,VP-SDE-based generation—inference-time scaling with flow models offers significant advantages over diffusion models. Flow models, particularly those with rectification fine-tuning [35, 36], produce much clearer expected outputs at intermediate steps, enabling more precise future reward estimation and, in turn, more effective particle sampling.

We additionally explore a strategy for tight budget enforcement in terms of the number of function evaluations (NFEs) of the velocity prediction network. Previous particle-sampling-based inferencetime scaling approaches for diffusion models [30, 53] allocate the NFEs budget uniformly across timesteps in the generative process, which we empirically found to be ineffective in practice. To optimize budget utilization, we propose Rollover Budget Forcing, a method that adaptively reallocates NFEs across timesteps. Specifically, we perform a denoising step upon identifying a new particle with a higher expected future reward and allocate the remaining NFEs to subsequent timesteps.

Experimentally, we demonstrate that our inference-time SDE conversion and VP interpolant conversion enable efficient particle sampling in flow models, leading to consistent improvements in reward alignment across two challenging tasks: compositional text-to-image generation and quantity-aware image generation. Additionally, our Rollover Budget Forcing (RBF) provides further performance gains, outperforming all previous particle sampling approaches. We also demonstrate that for differentiable rewards, such as aesthetic image generation, integrating RBF with a gradient-based method [8] creates a synergistic effect, leading to further performance improvements.

In summary, we introduce an inference-time scaling for flow models, analyzing three key factors:

- • ODE vs. SDE: We introduce an SDE generative process for flow models to enable particle sampling.
- • Interpolant: We demonstrate that replacing the linear interpolant of flow models with Variance Preserving interpolant expands the search space, facilitating the discovery of higher-reward samples.
- • NFEs Allocation: We propose Rollover Budget Forcing that adaptively allocates NFEs across timesteps to ensure efficient utilization of the available compute budget.

### 2 Related Work

- 2.1 Reward Alignment in Diffusion Models

In the literature of diffusion models, reward alignment approaches can be broadly categorized into fine-tuning-based methods [5, 69, 62, 9, 43, 67] and inference-time-scaling-based methods [30, 53, 13, 64, 6]. While fine-tuning diffusion models enables the generation of samples aligned with user preferences, it requires fine-tuning for each task, potentially limiting scalability. In contrast, inference-time scaling approaches offer a significant advantage as they can be applied to any reward without requiring additional fine-tuning. Moreover, inference-time scaling can also be applied to fine-tuned models to further enhance alignment with the reward. Since our proposed approach is an inference-time scaling method, we focus our review on related literature in this domain.

When the reward is differentiable, gradient-based methods [8, 3, 71, 16, 17, 63, 4] have been extensively studied. We note that inference-time scaling can be integrated with gradient-based approaches to achieve synergistic performance improvements.

- 2.2 Particle Sampling with Diffusion Models

The simplest iterative sampling method that can be applied to any generative model is Best-of-N (BoN) [57, 59, 58], which generates N samples and selects the one with the highest reward. For diffusion models, however, incorporating particle sampling during the denoising process has been shown to be far more effective than naïve BoN [53, 30]. This idea has been further developed through various approaches that sample particles at intermediate steps. For instance, SVDD [30] proposed selecting the particle with the highest reward at every step. CoDe [53] extends this idea by selecting the highest-reward particle only at specific intervals. On the other hand, methods based on Sequential Monte Carlo (SMC) [64, 6, 23, 13] employ a probabilistic selection approach, in which particles are sampled from a multinomial distribution according to their importance weights. Despite the success of particle sampling approaches for diffusion models, they have not been applicable to flow models due to the absence of stochasticity in their generative process. In this work, we present the first inference-time scaling method for flow models based on particle sampling by introducing stochasticity into the generative process and further increasing sampling diversity through trajectory modification.

- 2.3 Inference-Time Scaling with Flow Models

To our knowledge, Search over Paths (SoP) [39] is the only inference-time scaling method proposed for flow models, which applies a forward kernel to sample particles from the deterministic sampling process of flow models. However, SoP does not explore the possibility of modifying the reverse kernel, which could enable the application of more diverse particle-sampling-based methods [30, 53, 23]. To the best of our knowledge, we are the first to investigate the application of particle sampling to flow models through the lens of the reverse kernel.

### 3 Problem Definition and Background

- 3.1 Inference-Time Reward Alignment Given a pretrained flow model that maps the source distribution, a standard Gaussian distribution

p1, into the data distribution p0, our objective is to generate high-reward samples x0 ∈ Rd from the pretrained flow model without additional training–a task known as inference-time reward alignment.

We denote the given reward function as r : Rd → R, which measures text alignment or user preference for a generated sample. Following previous works [27, 60, 61], our objective can be formulated as finding the following target distribution:

p∗0 = arg max

, (1)

Ex0∼q [r(x0)]

###### −β DKL [q∥p0]

q

Reward

KL Regularization

which maximizes the expected reward while the KL divergence term prevents p∗0(x0) from deviating too far from p0(x0), with its strength controlled by the hyperparameter β. As shown in previous work [45], the target distribution p∗0 can be computed as:

r(x0) β

1 Z

p∗0(x0) =

, (2)

p0(x0) exp

where Z is a normalization constant. We present details in Appendix A.1. However, sampling from the target distribution is non-trivial.

A notable approach for sampling from the target distribution is particle sampling, which maintains a set of candidate samples—referred to as particles—and iteratively propagates high-reward samples while discarding lower-reward ones. When combined with the denoising process of diffusion models, particle sampling can improve the efficiency of limited computational resources in inference-time scaling. In the next section, we review particle sampling methods used in diffusion models and, for the first time, we explore insights for adapting them to flow models.

- 3.2 Particle Sampling Using Diffusion Model A pretrained diffusion model generates data by drawing an initial sample from the standard Gaussian

distribution and iteratively sampling from the learned conditional distribution pθ(xt−∆t|xt). Building on this, previous works [29, 61] have shown that data from the target distribution in Eq. 2 can be generated by performing the same denoising process while replacing the conditional distribution pθ(xt−∆t|xt) with the optimal policy:

pθ(xt−∆t|xt) exp v(xtβ−∆t) pθ(xt−∆t|xt) exp v(xtβ−∆t) dxt−∆t

p∗θ(xt−∆t|xt) =

, (3)

where the details are presented in Appendix A.2. We denote v(·) : Rd → R as the optimal value function that estimates the expected future reward of the generated samples at current timestep. Following previous works [8, 23, 30, 3], we approximate the value function using the posterior mean computed via Tweedie’s formula [47], given by v(xt) ≈ r(x0|t), where x0|t := Ex

0∼pθ(x0|xt) [x0]. Since directly sampling from the optimal policy distribution in Eq. 3 is nontrivial, one can first approximate the distribution using importance sampling while taking pθ(xt−∆t|xt) as the proposal distribution:

wt(−i)∆t K j=1 wt(−j)∆t

K

, {x(t−i)∆t}Ki=1 ∼ pθ(xt−∆t|xt), (4)

p∗θ(xt−∆t|xt) ≈

δx(i)

t−∆t

i=1

where K is the number of particles, wt(−i)∆t = exp v(x(t−i)∆t)/β is the weight, and δx(i)

is a

t−∆t

Dirac distribution. SVDD [30] proposed an approximate sampling method for the optimal policy by selecting the sample with the largest weight from Eq. 4.

Notably, a key factor in seeking high-reward samples using particle sampling is defining the proposal distribution to sufficiently cover the distribution of high-reward samples. Consider a scenario where high-reward samples reside in a low density region of the original data distribution, which is common when generating complex or highly specific samples that deviate from the mode of the pretrained model distribution. In this case, the proposal distribution must have a sufficiently large variance to effectively explore these low density regions. This highlights the importance of the stochasticity of the proposal distribution, which has been instrumental in the successful adoption of particle sampling in diffusion models. In contrast, flow models [32] employ a deterministic sampling process, where all particles xt−∆t drawn from xt are identical. This restricts the applicability of particle sampling methods in flow models. One of the main contributions is the investigation of how these particle sampling methods can be efficiently applied to flow models.

To this end, we propose an inference-time approach that introduces stochasticity into the generative process of flow models to enable particle sampling. We first transform the deterministic sampling process of flow models into a stochastic process (Sec. 4.2). We further identify a sampling trajectory that expands the search space of the flow models (Sec. 4.3). Note that while stochastic sampling and trajectory conversion have been studied in prior works, their primary goals have been to improve sample quality [68, 70, 49, 24, 38] or to accelerate inference [19, 52, 51, 22]. To the best of our

[Figure 27]

Figure 2: Comparison of Linear-ODE, LinearSDE, and VP-SDE. The visualization shows how trajectories evolve under different dynamics starting from the same noise latent.

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

(a) Linear-ODE (b) Linear-SDE (c) VP-SDE

Figure 3: Sample diversity test using FLUX [28] under linear and VP interpolant. All samples share the same initial latent. Prompt:

“A steaming cup of coffee”.

knowledge, we are the first to investigate sampling stochasticity and trajectory conversion for efficient particle-based sampling in flow models.

Additionally, previous particle sampling methods in diffusion models allocated a fixed computational budget (i.e., a uniform number of particles) across all denoising timesteps, potentially limiting exploration. We explore sampling with the rollover strategy, which adaptively allocates compute across timesteps during the sampling process (Sec. 6).

### 4 SDE-Based Particle Sampling in Flow Models

In this section, we review flow and diffusion models within the unified stochastic interpolant framework (Sec. 4.1) and introduce our inference-time approaches for efficient particle sampling in flow models (Sec. 4.2 and 4.3).

#### 4.1 Background: Stochastic Interpolant Framework

At the core of both diffusion and flow models is the construction of probability paths {pt}0≤t≤1, where xt ∼ pt serves as a bridge between x1 ∼ p1 and x0 ∼ p0:

xt = αtx0 + σtx1, (5)

where αt and σt are smooth functions satisfying α0 = σ1 = 1, α1 = σ0 = 0, and α˙t < 0,σ˙t > 0; we denote the dot as a time derivative. This formulation provides a flexible choice of interpolant

(αt,σt) which determines the sampling trajectory.

#### 4.2 Inference-Time SDE Conversion

Flow models [32, 35] learn the velocity field ut : Rd → Rd, which enables sampling of x0 by solving the Probability Flow-ODE [56] backward in time:

dxt = ut(xt)dt. (6)

The deterministic process in Eq. 6 accelerates the sampling process enabling few-step generation of high-fidelity samples. However, as discussed in Sec. 3.2, the deterministic nature of this sampling process limits the applicability of particle sampling in flow models.

To address this, we transform the deterministic sampling process into a stochastic process. The reverse-time SDE that shares the same marginal densities as the deterministic process in Eq. 6:

gt2

2 ∇log pt(xt), (7) where ft(xt) and gt represent the drift and diffusion coefficient, respectively, and w is the standard Wiener process. This conversion introduces a noise schedule gt, which can be freely chosen. Although SiT [38] arrives at the same conclusion, we provide a more comprehensive proof in Appendix B. In our case, we set gt = t2, scaled by a factor of 3. Note that in the case where gt = 0 the process reduces to deterministic sampling in Eq. 6.

dxt = ft(xt)dt + gtdw, ft(xt) = ut(xt) −

Using the velocity ut(xt) predicted by a pretrained flow model, the score function ∇log pt(xt) appearing in the drift coefficient ft(xt) can be computed as:

∇log pt(xt) =

αtut(xt) − α˙txt α˙tσt − αtσ˙t

1 σt

. (8)

This enables the conversion of the deterministic sampling to stochastic sampling, which we refer to as inference-time SDE conversion. Given the drift coefficient term ft(xt) and diffusion coefficient gt, the proposal distribution in the discrete-time domain is derived as follows:

pθ(xt−∆t|xt) = N(xt − ft(xt)∆t, gt2∆t I). (9) While previous works have proposed converting an SDE to an ODE to improve sampling efficiency [22, 55, 37, 56], the reverse approach—transforming an ODE into an SDE—has received relatively less attention and has primarily focused on improving sample quality [68, 38]. To the best of our knowledge, this work is the first to explore SDE conversion in flow models specifically to expand the search space of proposal distribution for efficient particle sampling.

Since flow models utilize the linear interpolant (αt = 1 − t,σt = t), we refer to the generative processes of the flow models using Eq. 6 and Eq. 7 as Linear-ODE and Linear-SDE, respectively. In Fig. 2 (left), we visualize the sampling trajectories of Linear-ODE and Linear-SDE. The samples generated using Linear-ODE are identical and collapse to a single point, restricting exploration. In contrast, Linear-SDE introduces sample variance, allowing for broader exploration and increasing the likelihood of discovering high-reward samples.

In Fig. 3 (a-b), we visualize images sampled from Linear-ODE and Linear-SDE using FLUX [28]. As discussed previously, the particles drawn from the proposal distribution of Linear-ODE are identical. In contrast, Linear-SDE introduces variation across different particles, thereby expanding the search space for identifying high-reward samples. In the next section, we introduce inference-time interpolant conversion, which further increases the search space.

#### 4.3 Inference-Time Interpolant Conversion

To further expand the search space of Linear-SDE, we draw inspiration from the effective use of particle sampling in diffusion models, where we identified a key difference: the interpolant. While the forward process in diffusion models follows the Variance Preserving (VP) interpolant

t

t

- 1

- 2

0 βsds), with βs denoting a predefined variance schedule, flow models adopt a linear interpolant.

(αt = exp−

0 βsds,σt = 1 − exp−

As shown in the previous works [33, 52], we note that given a velocity model ut based on an interpolant (αt, σt) (e.g., linear), one can transform the vector field and generate a sample based on a new interpolant (¯αs,σ¯s) (e.g., VP) at inference-time. The two paths x¯s = α¯sx0 + σ¯sx1 and xt = αtx0 + σtx1 are connected through scale-time transformation:

ts = ρ−1(¯ρ(s)) cs = σ¯s/σt

, (10) where ρ(t) = α

x¯s = csxt

s

s

σt and ρ¯(s) = α¯

σ¯s define the signal-to-noise ratio of the original and the new interpolant, respectively. The velocity for the new interpolant is given as:

t

s

σt2s σ ¯sα¯˙s − α¯sσ¯˙s σ¯s2 (σtsα˙ts − αtsσ˙ts)

σtsσ¯˙s − σ¯sσ˙tst˙s σt2s

x ¯s cs

c˙s cs

t˙s =

x¯s + cst˙suts

. (11)

u¯s(x¯s) =

, c˙s =

Plugging the transformed velocity into the proposal distribution in Eq. 9 after computing the score using Eq. 8 gives our efficient proposal distribution.

gs2 2 ∇log p¯s(x¯s) ∆s, gs2∆s I . (12)

p¯θ(x¯s−∆s|x¯s) = N x ¯s − u ¯s(x¯s) −

Since the new trajectory follows the VP interpolant, we refer to this as VP-SDE. We visualize VP-SDE sampling in Fig. 2 (right). At inference-time, we query the velocity of the new interpolant from the original interpolant (purple arrow). In Fig. 3 (c), we visualize the sample diversity under VP-SDE using FLUX [28] which generates more diverse samples than Linear-SDE. This property of VP-SDE effectively expands the search space, improving particle sampling efficiency in flow models. In Sec. 5, we provide further analysis on how interpolant conversion contributes to sample diversity.

Previous works focused on interpolant conversion that enables stable training [49, 12, 24] and accelerated inference [51, 22, 52]. We utilize interpolant conversion to enhance the sample diversity in particle sampling, which has not been unexplored before. Importantly, while we modify the generative process of flow models to align with that of diffusion models, inference-time scaling with flow models still provides distinct advantages. The rectified trajectories of flow models [35, 36, 28] allow for a much clearer posterior mean, leading to more precise future reward estimation and, in turn, more effective particle filtering.

### 5 Analysis of the Interpolant Conversion and Sample Diversity

In this section, we analyze how the interpolant conversion affects the variance of the proposal distribution and explain why VP-SDE yields higher sample diversity than Linear-SDE, as illustrated in Fig. 3. To investigate this behavior, Fig. 4 visualizes the logSNR (log αt2/σt2 ) of commonly used interpolants including linear and VP across timesteps t ∈ (0,1).

Then consider initializing the Linear-SDE timesteps {ts}0≤s≤1 using the timestep conversion in Eq. 10. This ensures that the log-SNR of the corresponding latents xt

matches that of the VP-SDE latents x¯s at each step (see the horizontal dashed line in Fig. 4). Under this condition, the proposal distributions of the two processes are expressed as follows:

s

[Figure 40]

Figure 4: Interpolant log-SNR. Dashed lines show a reference SNR and timestep.

)∆ts, gt2

Linear-SDE (ts): pθ(xt

s−∆ts | xt

) = N xt

s − ft

(xt

∆ts I

s

s

s

s

VP-SDE: p¯θ(x¯s−∆s | x¯s) = N x ¯s − ¯fs(x¯s)∆s, gs2∆s I = p¯θ(· | csxt

). Since ∆ts < ∆s at early denoising steps, timestep conversion results in smaller variance gt2

s

∆ts than VP-SDE for a fixed diffusion coefficient. Hence, to match the variance of Linear-SDE (ts) to that of VP-SDE, one can scale the diffusion coefficient to gt′

s

= gs/cs ∆s/∆ts. Note that this scaling significantly increases the stochasticity at early denoising steps as cs ≈ 1 and ∆s/∆ts ≫ 1. While this scaling can enhance sample diversity, applying it in isolation injects excessive noise, causing samples to deviate from the predefined denoising trajectory and ultimately degrading output quality.

s

In fact, interpolant conversion counteracts excessive noise injection by pairing diffusion coefficient scaling with timestep conversion. The two mechanisms act synergistically to increase the sample diversity without harming the sample quality. In Sec. 7, we validate this analysis with an ablation study that isolates the effect of each factor.

Comparison under Identical Timestep and Diffusion Coefficient. We next analyze the case where both Linear-SDE and VP-SDE operate under identical, fixed timestep schedules and diffusion coefficients. While this setting yields identical proposal distribution variances, Fig. 4 shows that the VP interpolant maintains a consistently lower log-SNR, indicating that at any given timestep, VP-SDE samples contain a larger noise component (see the vertical dashed line). Consequently, the VP-SDE proposal distribution effectively samples from a noisier latent at each step, resulting in higher sample diversity. This reflects the observation that noisier latents produce more diverse samples [40]. While this work focuses on the interpolant perspective, a systematic exploration of timestep scheduling and diffusion coefficient scaling remains a promising direction for future research.

### 6 Rollover Budget Forcing

In the previous sections, we have introduced our inference-time approaches to expand the search space of proposal distribution. Here, we propose a new budget-forcing strategy to maximize the use of limited compute in inference-time scaling. To the best of our knowledge, previous particle sampling methods for diffusion models [30, 53] employ a fixed number of particles across all denoising steps. However, our analysis shows that this uniform allocation may lead to inefficiency, where the NFEs required at each denoising step to obtain a sample xt−∆t with a higher reward than the current sample xt significantly varies across different runs. We present the analysis results in Appendix C.

This motivates us to adopt a rollover strategy that adaptively allocates NFEs across timesteps. Given a total NFEs budget, the NFEs quota Q is allocated uniformly across timesteps. Then at each timestep, if a particle xt−∆t yields a higher reward than the current sample xt within the quota, we immediately proceed to the next timestep from the newly identified high-reward sample, rolling over the remaining NFEs to the next step. If the allocated quota is exhausted without identifying a better sample, we select the particle with the highest expected future reward from the current set, following the strategy used in SVDD [30]. The pseudocode of RBF is presented in Appendix D. In the next section, we demonstrate the effectiveness of RBF, along with SDE conversion and interpolant conversion.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

- Figure 5: Quantitative results of compositional text-to-image generation. † denotes the given reward used in inference-time scaling (left). Notably, performance consistently improves from Linear-ODE to Linear-SDE and VP-SDE for both given and held-out rewards (left, middle), without significant quality degradation, as evidenced by the comparable aesthetic score [50] (right).

### 7 Applications

In this section, we present the experimental results of particle sampling methods for inference-time reward alignment. In Appendix, we present i) implementation details of the search algorithms, ii) aesthetic image generation, iii) comparisons between diffusion and flow models, iv) scaling behavior comparison of Best-of-N (BoN) and RBF, and v) additional qualitative results.

#### 7.1 Experiment Setup

Tasks. In this section, we present the results for the following applications: compositional text-toimage generation and quantity-aware image generation, where the rewards are non-differentiable. For the differentiable reward case, we consider aesthetic image generation (Appendix E.1). In compositional text-to-image generation, we use all 121 text prompts from GenAI-Bench [21] that contain three or more advanced compositional elements. For quantity-aware image generation, we use 100 randomly sampled prompts from T2I-CompBench++ [20] numeracy category.

For all applications, we use FLUX [28] as the pretrained flow model. We fix the total number of function evaluations (NFEs) to 500 and set the number of denoising steps to 10, which allocates 50 NFEs per denoising step. As a reference, we also include the results of the base pretrained models without inference-time scaling. Additionally, we present a comparison between flow models and diffusion models in Appendix E.2.

Baselines. We evaluate inference-time search algorithms discussed in Sec. 2, including Best-of-N (BoN), Search over Paths (SoP) [39], SMC [23], CoDe [53], and SVDD [30]. We categorize BoN and SoP as Linear-ODE-based methods, as their generative processes follow the deterministic process in Eq. 6. For SMC, we adopt DAS [23]; however, when the reward is non-differentiable, we use the reverse transition kernel of the pretrained model as the proposal distribution.

#### 7.2 Compositional Text-to-Image Generation

Evaluation Metrics. In this work, we refer to the reward used for inference-time scaling as the given reward. Here, the given reward is VQAScore, measured with CLIP-FlanT5 [31], which evaluates text-image alignment. For the held-out reward, which is not used during inference, we evaluate the score using a different model, InstructBLIP [10]. Additionally, we evaluate aesthetic score [50] to assess the quality of the generated images.

Inference-Time SDE and Interpolant Conversion. The quantitative and qualitative results of compositional text-to-image generation are presented in Fig. 5 and Fig. 6, respectively. As discussed in Sec. 4.2, the deterministic sampling process in flow models limits the effectiveness of particle sampling, whereas introducing stochasticity significantly expands the search space and improves performance—highlighting a key contribution of our work: enabling effective particle sampling in flow models. The results in Fig. 5 support this finding, showing that Linear-SDE (yellow) consistently improves the given reward (left in Fig. 5) over the Linear-ODE (green) across all particle sampling methods, even surpassing BoN and SoP [39], which were previously the only available inference-time

Linear-ODE Linear-SDE VP-SDE Linear-ODE Linear-SDE VP-SDE “Three small, non-blue boxes on a large blue box.”

“Five origami cranes hang from the ceiling, only one of which is red, and the others are all white.”

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

CoDe[53]

SMC[23]

“A mouse pad has two pencils on it, the shorter pencil is green and the longer one is not.”

“Five ants are carrying biscuits, and an ant that is not carrying biscuits is standing on a green leaf directing them.”

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

SVDD[30]

RBF

- Figure 6: Qualitative results of compositional text-to-image generation. We use VQAScore [31], which measures text-image alignment, as the given reward for inference-time scaling. SDE and interpolant conversion enable more effective exploration during inference, enhancing the performance of all particle sampling methods [23, 53, 30], including RBF.

scaling approaches for ODE-based flow models. Additionally, through inference-time interpolant conversion, VP-SDE (red) further improves performance across all particle sampling methods on both given and held-out rewards (left, middle in Fig. 5) by expanding the search space, demonstrating the effectiveness of our proposed distribution. Notably, particle sampling methods with LinearSDE and VP-SDE generate high-reward samples without significantly compromising image quality, as evidenced by aesthetic scores that remain comparable to the base FLUX model [28] (right in Fig. 5). Qualitatively, SDE conversion and interpolant conversion shown in Fig. 6 bring consistent performance improvements (see Appendix G.1 for additional results).

Rollover Budget Forcing. As discussed in Sec. 6, instead of fixing the number of particles throughout the denoising process, we explore adaptive budget allocation through RBF. We demonstrate that budget forcing provides additional performance improvements, outperforming the previous particle sampling methods in the given reward (left in Fig. 5). We present qualitative comparisons of inference-time scaling methods in Appendix G.2.

Ablation study of interpolant conversion. Building on the analysis in Sec. 5, we examine how interpolant conversion contributes to sample diversity and reward alignment through its two underlying mechanisms, timestep conversion and diffusion coefficient scaling. Tab. 1 extends the results of Fig. 5 by isolating the effect of each component to sample diversity, measured by LPIPSMPD [23], and reward alignment.

Table 1: Ablation Study of Interpolant Conversion. † denotes the given reward.

Method LPIPS-MPD ↑ VQAScore† ↑ Inst. BLIP ↑

Linear-ODE – 0.788 0.789 Linear-SDE 0.158 0.900 0.813

+ Adapt. Time. 0.270 0.908 0.813 + Adapt. Diff. 0.429 0.702 0.571 VP-SDE 0.509 0.925 0.843

We observe that timestep conversion (row 3) yields only modest diversity gains: the benefit of sampling at lower log-SNR (Fig. 4) is offset by smaller discretization steps that reduce proposal variance, limiting improvements in reward alignment. On the other hand, applying diffusion coefficient scaling without timestep conversion (row 4) increases sample diversity but simultaneously leads to a significant drop in reward alignment indicating excessive noise injection. Lastly, the VP-SDE interpolant conversion (row 5) synergistically combines both components, achieving high sample diversity without sacrificing quality and consequently yielding the highest reward.

#### 7.3 Quantity-Aware Image Generation

Evaluation Metrics. Here, the given reward is the negation of the Residual Sum of Squares (RSS) between the target counts and the detected object counts, computed using GroundingDINO [34] and SAM [25] (details in Appendix F). Additionally, we report object count accuracy, which evaluates whether all object quantities are correctly shown in the image. For the held-out reward, we report

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

- Figure 7: Quantitative results of quantity-aware image generation. † denotes the given reward, RSS [34], with the y-axis truncated for better visualization (left). We observe consistent performance improvements by converting Linear-ODE to Linear-SDE, and VP-SDE for most cases.

Linear-ODE Linear-SDE VP-SDE Linear-ODE Linear-SDE VP-SDE

“Four balloons, one cup, four desks, two dogs and four microwaves.”

“Four candles, two balloons, one dog, two tomatoes and three helicopters.”

SMC[23]

[Figure 62]

[Figure 63]

[Figure 64]

CoDe[53]

[Figure 65]

[Figure 66]

[Figure 67]

“Eight chairs.” “One egg, three camels, four cars and four pillows.”

SVDD[30]

[Figure 68]

[Figure 69]

[Figure 70]

RBF

[Figure 71]

[Figure 72]

[Figure 73]

- Figure 8: Qualitative results of quantity-aware image generation. At inference-time, we guide generation using the negation of RSS [34] (Residual Sum of Squares) as the given reward, which measures the discrepancy between detected and target object counts. SDE and interpolant conversion expands the search space to identify high reward samples.

VQAScore measured with CLIP-FlanT5 [31]. As in the previous application, we evaluate the quality of the generated images using the aesthetic score [50].

Results. The quantitative and qualitative results are presented in Fig. 7 and Fig. 8, respectively. The trend in Fig. 7 align with those in Sec. 7.2, demonstrating that SDE conversion and interpolant conversion synergistically enhance the identification of high-reward samples. Notably, particle sampling methods with Linear-SDE already outperform Linear-ODE-based methods (BoN and SoP [39]), while interpolant conversion further improves accuracy, achieving a 4 ∼ 6× improvement over the base model [28]. Our RBF achieves the highest accuracy, outperforming all other particle sampling methods. Qualitatively, Fig. 8 shows that SDE and interpolant conversion effectively identify high-reward samples that accurately match the specified object categories and quantities. Additional qualitative comparisons of the inference-time scaling methods are provided in Appendix G.2.

### 8 Conclusion and Limitation

We introduced a novel inference-time scaling method for flow models with three key contributions: (1) ODE-to-SDE conversion for particle sampling in flow models, (2) Linear-to-VP interpolant conversion for enhanced diversity and search efficiency, and (3) Rollover Budget Forcing (RBF) for adaptive compute allocation. We demonstrated the effectiveness of VP-SDE-based generation in applying off-the-shelf particle sampling to flow models and showed that our RBF combined with VP-SDE generation outperforms previous methods. However, our method introduces additional inference-time overhead, which could become a bottleneck when the base model prediction is computationally intensive. Also, since the pretrained model may have been trained on uncurated datasets, our approach may produce undesirable outputs upon malicious attempts.

### Acknowledgments

This work was supported by the NRF of Korea (RS-2023-00209723); IITP grants (RS-2022II220594, RS-2023-00227592, RS-2024-00399817, RS-2025-25441313, RS-2025-25443318, RS2025-02653113); and the Technology Innovation Program (RS-2025-02317326), all funded by the Korean government (MSIT and MOTIE), as well as by the DRB-KAIST SketchTheFuture Research Center.

### References

- [1] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv, 2023.
- [2] Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 1982.
- [3] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In CVPRW, 2023.
- [4] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. 2024.
- [5] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv, 2024.
- [6] Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, and Eric Moulines. Monte carlo guided diffusion for bayesian linear inverse problems. In ICLR, 2024.
- [7] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, Yifei Hu, Ting-Che Lin, Shilong Zhang, Fu Li, Chuan Li, Xing Wang, Yanghua Peng, Peize Sun, Ping Luo, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Goku: Flow based video generative foundation models. arXiv preprint arXiv:2502.04896, 2025.
- [8] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In ICLR, 2023.
- [9] Kevin Clark, Paul Vicol, Kevin Swersky, and Fleet David J. Directly fine-tuning diffusion models on differentiable rewards. In ICLR, 2024.
- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023.
- [11] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021.
- [13] Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In ICLR, 2024.
- [14] Arnaud Doucet, Nando De Freitas, Neil James Gordon, et al. Sequential Monte Carlo methods in practice. Springer, 2001.
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 2024.
- [16] Luca Eyring, Shyamgopal Karthik, Karsten Roth, Alexey Dosovitskiy, and Zeynep Akata. Reno: Enhancing one-step text-to-image models through reward-based noise optimization. In NeurIPS, 2024.
- [17] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang. InitNO: Boosting text-to-image diffusion models via initial noise optimization. In CVPR, 2024.
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

- [19] Peter Holderrieth, Marton Havasi, Jason Yim, Neta Shaul, Itai Gat, Tommi Jaakkola, Brian Karrer, Ricky TQ Chen, and Yaron Lipman. Generator matching: Generative modeling with arbitrary markov processes. In ICLR, 2024.
- [20] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation . IEEE Transactions on Pattern Analysis Machine Intelligence, 2025.
- [21] Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. Genai arena: An open evaluation platform for generative models. arXiv, 2024.
- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. 2022.
- [23] Sunwoo Kim, Minkyu Kim, and Dongmin Park. Test-time alignment of diffusion models without reward over-optimization. In ICLR, 2025.
- [24] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In NeurIPS, 2021.
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In ICCV, 2023.
- [26] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In NIPS, 2023.
- [27] Tomasz Korbak, Hady Elsahar, Germán Kruszewski, and Marc Dymetmant. On reinforcement learning and distribution matching for fine-tuning language models with no catastrophic forgetting. In NeurIPS, 2022.
- [28] Black Forest Labs. FLUX. https://github.com/black-forest-labs/flux, 2024.
- [29] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv, 2018.
- [30] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Aviv Regev, Sergey Levine, and Masatoshi Uehara. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv, 2024.
- [31] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. arXiv, 2024.
- [32] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.
- [33] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024.
- [34] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024.
- [35] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.
- [36] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and qiang liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In ICLR, 2024.
- [37] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NIPS, 2022.
- [38] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024.
- [39] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Saining Xie. Inference-time scaling for diffusion models beyond scaling denoising steps. arXiv, 2025.

- [40] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
- [41] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.
- [42] OpenAI. Learning to Reason with LLMs. https://openai.com/index/ learning-to-reason-with-llms/, 2024.
- [43] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv, 2023.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. 2021.
- [45] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NIPS, 2023.
- [46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 2020.
- [47] Herbert E. Robbins. An Empirical Bayes Approach to Statistics. Springer, 1992.
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [49] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.
- [50] C. Schuhmann. Laion aesthetics. https://laion.ai/blog/laion-aesthetics, 2022.
- [51] Neta Shaul, Ricky TQ Chen, Maximilian Nickel, Matthew Le, and Yaron Lipman. On kinetic optimal probability paths for generative models. 2023.
- [52] Neta Shaul, Juan Perez, Ricky TQ Chen, Ali Thabet, Albert Pumarola, and Yaron Lipman. Bespoke solvers for generative flow models. arXiv preprint arXiv:2310.19075, 2023.
- [53] Anuj Singh, Sayak Mukherjee, Ahmad Beirami, and Hadi Jamali-Rad. Code: Blockwise control for denoising diffusion models. arXiv, 2025.
- [54] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. 2015.
- [55] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. ICLR, 2021.
- [56] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.
- [57] Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback. In NeurIPS, 2020.
- [58] Luming Tang, Nataniel Ruiz, Chu Qinghao, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, and Michael Rubinstein. Realfill: Reference-driven generation for authentic image completion. ACM TOG, 2024.
- [59] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv, 2023.

- [60] Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M Tseng, Tommaso Biancalani, and Sergey Levine. Fine-tuning of continuous-time diffusion models as entropy-regularized control. arXiv, 2024.
- [61] Masatoshi Uehara, Yulai Zhao, Ehsan Hajiramezanali, Gabriele Scalia, Gökcen Eraslan, Avantika Lal, Sergey Levine, and Tommaso Biancalani. Bridging model-based optimization and generative modeling via conservative fine-tuning of diffusion models. In NeurIPS, 2024.
- [62] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024.
- [63] Bram Wallace, Akash Gokul, Stefano Ermon, and Nikhil Naik. End-to-end diffusion latent optimization improves classifier guidance. In ICCV, 2023.
- [64] Luhuan Wu, Brian L Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. In NeurIPS, 2023.
- [65] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In ICCV, 2023.
- [66] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.
- [67] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In NeurIPS, 2023.
- [68] Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, and Tommi Jaakkola. Restart sampling for improving generative processes. In NeurIPS, 2023.
- [69] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Qimai Li, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In CVPR, 2024.
- [70] Kyeongmin Yeo, Jaihoon Kim, and Minhyuk Sung. Stochsync: Stochastic diffusion synchronization for image generation in arbitrary spaces. In ICLR, 2025.
- [71] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energyguided conditional diffusion model. In ICCV, 2023.
- [72] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024.

### NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- • You should answer [Yes] , [No] , or [NA] .
- • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- • Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

- • Delete this instruction block, but keep the section heading “NeurIPS Paper Checklist",
- • Keep the checklist subsection headings, questions/answers and guidelines below.
- • Do not modify the questions and only use the provided macros for your answers.

- 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes] Justification: Our analysis and experiments support the claims in the abstract and introduction. Guidelines:

- • The answer NA means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

- 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We present limitations in the last section. Guidelines:

- • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate "Limitations" section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes] Justification: We present proofs in Appendix. Guidelines:

- • The answer NA means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] Justification: We provide implementation details and experiment setups. Guidelines:

- • The answer NA means that the paper does not include experiments.

- • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: Code is publicly released. Guidelines:

- • The answer NA means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes] Justification: We specify hyperparameters in the paper. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No] Justification: Due to computational constraints, we were unable to include error bars. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: We specify memory usage and inference time in the paper. Guidelines:

- • The answer NA means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: This work conforms with the NeurIPS Code of Ethics. Guidelines:

- • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes] Justification: We discuss societal impacts in the paper. Guidelines:

- • The answer NA means that there is no societal impact of the work performed.
- • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes] Justification: We do not provide or use prompts that may generate harmful or disruptive content. Guidelines:

- • The answer NA means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: We have cited the relevant works. Guidelines:

- • The answer NA means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA] Justification: We do not release new assets. Guidelines:

- • The answer NA means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Does not involve research with human subjects. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] Justification: Does not involve research with human subjects. Guidelines:

- • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] Justification: Core method development of our work did not involve usage of LLMs. Guidelines:

- • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

Appendix

- A Proofs

- A.1 Derivation of the Target Distribution

From Eq. 1, we obtain the target distribution p∗0, which maximizes the reward while maintaining proximity to the distribution of the pretrained model p0:

p∗0(x0) = arg max

Ex

0∼q [r(x0)] − βDKL [q∥p0],

q

q(x0) p0(x0)

Ex

0∼q r(x0) − β log

= arg max

q

q(x0) p0(x0) −

1 β

Ex

= arg min

0∼q log

r(x0)

q

q(x0) p0(x0)

1 β

dx0 −

q(x0)log

q(x0)r(x0)dx0.

= arg min

q

This can be solved via calculus of variation where the functional J is given as follows:

q(x0) p0(x0) −

1 β

J [q(x0)] := q(x0) log

r(x0) dx0.

Substituting q˜(x0,ϵ) := q(x0) + ϵη(x0) gives:

q˜(x0,ϵ) p0(x0) −

1 β

J [˜q(x0,ϵ)] = q ˜(x0,ϵ) log

r(x0) dx0,

where η(x0) is an arbitrary smooth function, and ϵ is a scalar parameter. Introducing a Lagrange multiplier λ to constraint q(x0)dx0 = 1 gives:

q˜(x0,ϵ) p0(x0) −

J [˜q(x0,ϵ)] = q ˜(x0,ϵ) log

1 β

r(x0) + λq˜(x0,ϵ)dx0

:= f{q˜;x0}dx0

Then the problem boils down to finding a function q˜(x0,ϵ) satisfying:

∂J ∂ϵ ϵ=0

= 0

This can be solved using the Euler-Lagrange equation: ∂f ∂q −

d dx0

∂f ∂q′ = 0,

where q′ is a derivative of q with respect to x0 and tilde notation is dropped since the condition is to be satisfied at ϵ = 0.

Note that q′ does not appear in f, so the Euler-Lagrange equation simplifies to:

∂f ∂q

∂ ∂q

q(x0) p0(x0) −

1 β

=

q(x0) log

r(x0) + λq(x0) = 0

q(x0) p0(x0) −

1 β

r(x0) + 1 + λ = 0. (13)

= log

Solving Eq. 13 gives the target distribution p∗0, which minimizes the objective function in Eq. 1:

p∗0(x0) = p0(x0)exp

r(x0) β − 1 − λ (14)

Lastly, the Lagrangian multiplier λ is obtained from the normalization constraint, exp(λ) =

p0(x0)exp r(x

0)

β − 1 dx0. Plugging this into Eq. 14 gives the target distribution presented in Eq. 2:

p0(x0)exp r(x

0) β

p∗0(x0) =

, (15)

p0(x0)exp r(x

0)

β dx0

#### A.2 Derivation of the Optimal Policy

Here, we provide the derivations of the optimal policy given in Eq. 3 for completeness, which is proposed in previous works [60, 61].

To sample from the target distribution defined in Eq. 15, previous studies utilize an optimal policy p∗θ(xt−∆t|xt). The optimal value function v(xt) is defined as the expected future reward at current timestep t:

v(xt) = β log Ex

0∼pθ(x0|xt) exp

r(x0) β

(16)

The optimal policy is the policy that maximizes the objective function:

p∗θ(xt−∆t|xt) = arg max

Ex

t−∆t∼q(·|xt) [v(xt−∆t)] − βDKL [q(·|xt)∥pθ(·|xt)]

q(·|xt)

pθ(xt−∆t|xt)exp β 1v(xt−∆t) pθ(xt−∆t|xt)exp β 1v(xt−∆t) dxt−∆t

=

pθ(xt−∆t|xt)exp β 1v(xt−∆t) exp β 1v(xt)

=

(17)

(18)

where the last equality follows from the soft-Bellman equations [61]. For completeness, we present the theorem.

Theorem 1. (Theorem 1 of Uehara et al. [61]). The induced distribution of the optimal policy in

- Eq. 17 is the target distribution in Eq. 15.

p∗0(x0) = p1(x1)

s=T

1

p∗θ(x s

T −T1 |x s

) dx 1 T :1.

T

However, computing the optimal value function in Eq. 16 is non-trivial. Hence, we follow the previous works [23, 30] and approximate it using the posterior mean x0|t := Ex

0∼pθ(x0|xt) [x0]:

r(x0) β

v(xt) = β log exp

r(x0|t) β

≈ β log exp

pθ(x0|xt)dx0

= r(x0|t). (19)

### B Choice of Diffusion Coefficient

Ma et al. [38] have shown that the diffusion coefficient can be chosen freely within the stochastic interpolant framework [1]. Here, we present a more comprehensive proof. We use w interchangeably to denote the standard Wiener process for both forward and reverse time flows.

Proposition 1. For a linear stochastic process xt = αtx0 + σtx1 and the Probability-Flow ODE dxt = ut(xt)dt that yields the marginal density pt(xt), the following forward and reverse SDEs

with an arbitrary diffusion coefficient gt ≥ 0 share the same marginal density: Forward SDE: dxt = ut(xt) +

gt2

2 ∇log pt(xt) dt + gtdw (20) Reverse SDE: dxt = ut(xt) −

gt2 2 ∇log pt(xt) dt + gtdw. (21)

Proof. When velocity field ut generates a probability density path pt, it satisfies the continuity equation:

∂ ∂t

pt(xt) = −∇ · (pt(xt)ut(xt)). (22)

Similarly, for the SDE dxt = ft(xt)dt + gtdw, the Fokker-Planck equation describes the time evolution of p˜t:

∂ ∂t

- 1

- 2

gt2∇2p˜t(xt) (23)

p˜t(xt) = −∇ · (˜pt(xt)ft(xt)) +

where ∇2 denotes the Laplace operator. To find an SDE that yields the same marginal probability density as the ODE, we equate the probability density functions in Eq. 23 and Eq. 22, resulting in the following equation:

- 1

- 2

gt2∇2pt(xt) = −∇ · (pt(xt)ut(xt)) ∇ · (pt(xt)(ft(xt) − ut(xt))) =

−∇ · (pt(xt)ft(xt)) +

- 1

- 2

gt2∇2pt(xt) (24)

This implies that any SDE with drift coefficient ft(xt) and diffusion coefficient gt that satisfies Eq. 24 will generate pt. One particular choice is to set pt(xt)(ft(xt) − ut(xt)) proportional to ∇pt(xt), i.e.,pt(xt)(ft(xt) − ut(xt)) = At∇pt(xt). Then Eq. 24 can be rewritten as:

- 1

- 2

At∇2pt(xt) =

gt2∇2pt(xt),

which leads to the relation At = 12gt2. Similarly, the drift coefficient is given by:

gt2∇pt(xt) pt(xt)

- 1

- 2

ft(xt) = ut(xt) +

- 1

- 2

gt2∇log pt(xt) Thus, a family of SDEs that generate pt takes the following form:

= ut(xt) +

- 1

- 2

gt2∇log pt(xt) dt + gtdw,

dxt = ut(xt) +

which is the forward SDE presented in Eq. 20. Similarly, the reverse SDE in Eq. 21 can be derived by applying the time reversal formula, following Anderson et al. [2].

| |
|---|

Corollary 1. If diffusion coefficient is chosen as gt = 2 σtσ˙t − σt2α˙

αt then the score function ∇log pt(xt) inside the forward SDE vanish and it can be written as:

t

α˙t αt

α˙t αt

dw (25)

xtdt + 2 σtσ˙t − σt2

dxt =

Proof. Velocity field ut(xt) for linear stochastic process Xt = αtX0 + σtX1 is given as:

α˙t αt

α˙t αt ∇log pt(xt) (26)

xt − σtσ˙t − σt2

ut(xt) =

Plugging this equation into forward SDE Eq. 20, we can immediately see that when gt =

##### 2 σtσ˙t − σt2α˙

αt the score function term vanishes and the remaining terms constitute Eq. 25.

t

| |
|---|

[Figure 74]

Figure 9: Analysis of number of function evaluations (NFEs) across timesteps. The NFEs required to achieve a higher reward for each timestep. The plot illustrates the ±1 sigma variation band. The blue-dotted line represents the uniform allocation of compute (NFEs) across timesteps. We observe that the NFEs required to identify a higher-reward sample may exceed the uniformly allocated budget (blue dotted line).

### C Adaptive Time Scheduling and Rollover Strategy

In this section, we provide details of adaptive time scheduling and NFE analysis result which inspired rollover strategy.

Adaptive Time Scheduling. As discussed in Sec. 4.3, to maximize the exploration space in VPSDE sampling, we design the time scheduler to take smaller steps during the initial phase—when variance is high—and gradually increase the step size in later stages. Specifically, we define the time scheduler as tnew = 1 − (1 − t)2. While this approach can be problematic when the number of steps is too low—resulting in excessively large discretization steps in later iterations—we find that using a reasonable number of steps (e.g.,10) works well in practice, benefiting from the few-step generation capability of flow models. This setup effectively balances a broad exploration space with fast inference time, highlighting one of the key advantages of flow models over diffusion models.

NFE Analysis. As discussed in Sec. 6, we analyze the number of function evaluations (NFEs) required to obtain a sample with a higher reward than the current one. In Fig. 9, we visualize the variance band of the required NFEs across timesteps, with the blue-dotted line representing the uniform allocation used in previous particle sampling methods [30, 53]. Notably, uniform compute allocation may constrain exploration and fail to identify high-reward samples, as evidenced by crossings within the variance band. This observation motivates the use of a rollover strategy to optimize compute utilization efficiently. As demonstrated in Sec. 7, our experiments confirm that RBF provides additional improvements over previous particle sampling methods [30, 53].

[Figure 75]

Figure 10: Schematics of inference-time search algorithms. Linear-ODE-based methods, BoN and SoP use a deterministic sampling process, whereas particle-sampling-based methods follow a stochastic process. Note that RBF adaptively allocates NFEs across denoising timesteps.

### D Search Algorithms

In this section, we introduce the inference-time search algorithms discussed in Sec. 2 along with their implementation details. An illustrative figure of the algorithms is provided in Fig. 10. Here, we

define the batch size (N) as the number of initial latent samples and the particle size (K) as the number of samples drawn from the proposal distribution pθ(xt−∆t|xt) at each denoising step.

Best-of-N (BoN) [57, 58] is a form of rejection sampling. Given N generated samples {x(0i)}Ni=1, BoN selects the sample with the highest reward.

r(x(0i)).

x0 = arg max {x(0i)}Ni=1

- As presented in Sec. 7, we fixed the total compute budget to 500 NFEs and the number of denoising steps to 10, which sets the batch size of BoN to N = 50.

Search over Paths (SoP) [39] begins by sampling N initial noises and running the ODE solver up to a predefined timestep t0. Then the following two operations iterate until reaching t = 0:

- 1. Applying the forward kernel: For each sample in the batch at time t, K particles are sampled using the forward kernel, which propagates them from t to t + ∆f.
- 2. Solving the ODE: The resulting N · K particles are then evolved from t + ∆f to t + ∆f − ∆b by solving the ODE. The top N candidates with the highest rewards are selected.

We followed the original implementations [39] for ∆f and ∆b. We used N = 2 and K = 5.

Sequential Monte Carlo (SMC) [23, 14] extends the idea of importance sampling to a time-sequential setting by maintaining N samples and updating their importance weights over time:

wt(−i)∆t =

p∗θ(xt−∆t|xt) q(xt−∆t|xt)

wt(i) =

- pθ(xt−∆t|xt)exp v(x(t−i)∆t)/β

- q(xt−∆t|xt)exp v(x(ti))/β

wt(i),

where q(xt−∆t|xt) is a proposal distribution and the last equality follows from the optimal policy Eq. 18. We used the reverse process of the pretrained model as the proposal distribution, which leads to the following importance weight equation:

wt(−i)∆t =

exp v(x(t−i)∆t)/β exp v(x(ti))/β

wt(i). (27)

- At each step when effective sample size Nj=1 wt(j)

2

/ Ni=1(wt(i))2 is below the threshold, we

perform resampling, i.e.„ indices {a(ti)}Ni=1 are first sampled from a multinomial distribution based on the normalized importance weights:

 .

 N,

N

wt(i)

{a(ti)}Ni=1 ∼ Multinomial

N j=1 wt(j)

i=1

These ancestor indices a(ti) are then used to replicate high-weight particles and discard low-weight ones, yielding the resampled set {x(a

(i) t )

t }Ni=1. If resampling is not performed, the indices are simply set as a(ti) = i. Lastly, one-step denoised samples are obtained from {x(a

(i) t )

t }Ni=1:

(i) t ) t ).

x(t−i)∆t ∼ pθ(xt−∆t|x(a

When resampling is performed, the importance weights are reinitialized to one, i.e.„ wt = 1. The importance weights for the next step, wt−∆t are subsequently computed according to Eq. 27, regardless of whether resampling was applied. We used N = 50 for all applications.

Controlled Denoising (CoDe) [53] extends BoN by incorporating an interleaved selection step after every L denoising steps.

exp v(x(t−i)L∆t)/β

xt−L∆t = arg max

{x(ti−)L∆t}Ki=1

We used N = 2, K = 25, and L = 2 for all applications. SVDD [30] approximates the optimal policy in Eq. 3 by leveraging weighted K particles:

wt(−i)∆t K j=1 wt(−j)∆t

K

p∗θ(xt−∆t|xt) ≈

δx(i)

t−∆t

i=1

{x(t−i)∆t}Ki=1 ∼ pθ(xt−∆t|xt) wt(−i)∆t = exp v(x(t−i)∆t)/β .

(28)

At each timestep, the approximate optimal policy in Eq. 28 is sampled by first drawing an index at−∆t from a categorical distribution:

  (29)

 

K

wt(−i)∆t K j=1 wt(−j)∆t

at−∆t ∼ Categorical

i=1

This index is then used to select the sample from {x(t−i)∆t}Ki=1, i.e.„ xt−∆t ← x(t−at∆−∆t t). In practice, SVDD uses β = 0, replacing sampling from the categorical distribution with a direct arg max

operation, i.e.„ selecting the particle with the largest importance weight. Following the original implementation [30], we used N = 2 and K = 25 for all applications.

Rollover Budget Forcing (RBF) adaptively allocates compute across denoising timesteps. At each timestep, when a particle with a higher reward than the previous one is discovered, it immediately takes a denoising step, and the remaining NFEs are rolled over to the next timestep, ensuring efficient utilization of the available compute. To maintain consistency with SVDD [30], we set N = 2, with the compute initially allocated uniformly across all timesteps. We present the pseudocode for sampling from the stochastic proposal distribution with interpolant conversion in Alg. 1. Specifically, the pseudocode for RBF with SDE conversion and interpolant conversion is

provided in Alg. 2. Here, we denote {S(i)}Mi=1 as a sequence of timesteps in descending order, where S(1) = 1 and S(M) = 0, and M is the total number of denoising steps.

### E Additional Results

#### E.1 Aesthetic Image Generation

In this section, we demonstrate that inference-time scaling can also be applied to gradient-based methods, such as DPS [8], for differentiable rewards. Specifically, we consider aesthetic image generation and show that RBF leads to synergistic performance improvements. We first derive the formulation of the proposal distribution for differentiable rewards and then present qualitative and quantitative results.

E.1.1 Gradient-Based Guidance Uehara et al. [61] have shown that the marginal distribution p∗t(xt) is computed as follows:

p∗t(xt) ∝ exp

v(xt) β

pt(xt) ≈ exp

r(x0|t) β

pt(xt),

Algorithm 2: Rollover Budget Forcing (RBF) Inputs: Number of denoising steps M,

Algorithm 1: stoch_denoise: 1-step stochastic denoising Inputs: original velocity field u,

##### timesteps {S(i)}Mi=1, NFE quota {Q(i)}Mi=1

original interpolant (α,σ), new interpolant (¯α,σ¯), diffusion coefficient g, current sample x¯s, current timestep s, denoising step size ∆s

Outputs: Aligned sample x¯0

- 1 x¯1 ∼ N(0,I) r∗ ← r(x¯0|1)
- 2 for i ∈ {1,...,M} do

- 3 s ← S(i) ∆s ← S(i) − S(i+1) q ← Q(i)
- 4 for j ∈ {1,...,q} do

- 5 x¯(sj−)∆s ← stoch_denoise(x¯s,s,∆s) // Alg. 1
- 6 if r∗ < r(x¯(0j|s)−∆s) then

- 7 Q(i+1) ← Q(i+1) + Q(i) − j // Sec. 6
- 8 r∗ ← r(x¯(0j|s)−∆s) x¯s−∆s ← x¯(sj−)∆s
- 9 break
- 10 if j = q then

- 11 k∗ ← arg maxk∈{1,...,q} r(x¯(0k|s)−∆s)
- 12 x¯s−∆s ← x¯(k

Outputs: Stochastically denoised sample x¯s−∆s

- 1 ts ← ρ−1(¯ρ(s)) cs ← σ¯s/σt

s

- 2 u¯s ← c˙

s

csx¯s + cst˙sut

s

x ¯s cs // Eq. 11

- 3 ss ← σ¯1

s

α¯su¯s−α¯˙ sx¯s α¯˙sσ¯s−α¯sσ¯˙s // Eq. 8

- 4 fs = u¯s − g

2 s

2 ss // Eq. 7

- 5 z ∼ N(0,I)
- 6 x¯s−∆s ← x¯s − fs∆s + gs

√

∗) s−∆s

∆s z

SVDD [30] + DPS [8]

RBF (Ours) + DPS [8]

FLUX [28] DPS [8]

“Bird”

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Table 2: Quantitative results of aesthetic image generation. † denotes the given reward used in inference time. The best result in each row is highlighted in bold.

“Bat”

Model Aesthetic

ImageReward [67] (held-out)

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Score† [50]

FLUX [28] 5.795 0.991

DPS [8] 6.438 0.605 SVDD [30]+DPS [8] 6.887 1.077 RBF (Ours)+DPS [8] 7.170 1.152

Figure 11: Qualitative results of aesthetic image generation. At inference-time, we guide generate using the aesthetic score [50] as the given reward, which assesses visual appeal.

where the approximation follows from Eq. 19. When the reward is differentiable (e.g.„ aesthetic score [50]), one can simulate samples from p∗t(xt) by computing its score function:

r(x0|t) β

∇log p∗t(xt) = ∇log exp

pt(xt)

1 β ∇r(x0|t)

. (30)

##### +∇log pt(xt)

=

Pretrained Score

Guidance

For differentiable rewards, we incorporate the gradient-based guidance defined in Eq. 30 into the SDE sampling process described in Eq. 7. Notably, this approach is orthogonal to inference-time

#### Table 3: Comparison of diffusion and flow models.

|Type<br><br>|Model<br><br>|ImageReward [67] HPS [65] PickScore [26] CLIP Score [44]<br><br>|Steps|
|---|---|---|---|
|Diffusion<br><br>|SD2 [48] SANA-1.5 [66]|0.429 0.280 0.218 0.269 0.894 0.284 0.222 0.270<br><br>|50 20|
|Flow|SD3 [15] FLUX [28]<br><br>|1.154 0.294 0.226 0.277 1.054 0.290 0.226 0.275<br><br>|28 5<br><br>|

scaling, and RBF can be additionally utilized to further enhance performance. In the next section, we experimentally demonstrate that RBF can be effectively integrated with gradient-based guidance.

#### E.1.2 Aesthetic Image Generation Results

The aesthetic image generation task aims to sample images that best capture human preferences, such as visual appeal. We use 45 animal prompts from previous work, DDPO [5]. The aesthetic score [50] serves as the given reward, while ImageReward [67] is used as the held-out reward.

We present quantitative and qualitative results of aesthetic image generation in Tab. 2 and Fig. 11. Notably, RBF, implemented with DPS [8], achieves significant improvements on both the given and held-out rewards, even surpassing SVDD [30]. Qualitatively, RBF effectively adapts the pretrained flow model to better align with human preferences, particularly in terms of visual appeal.

#### E.2 Comparison of Diffusion and Flow Models

We present quantitative comparisons between text-to-image diffusion and flow models in Tab. 3, using compositional text prompts from GenAI-Bench [21]. As shown, flow-based models outperform diffusion models across all evaluation metrics assessing image quality [67, 65, 26] and text alignment [44, 67]. In the flow-based models, FLUX [28] achieves competitive performance while requiring fewer steps compared to Stable Diffusion 3 [15].

#### E.3 Scaling Behavior Comparison

As discussed in Sec. 4, expanding the exploration space and applying budget forcing significantly enhance the efficiency of RBF, leading to superior performance improvements over BoN. Here, we compare the scaling behavior of BoN, a representative Linear-ODE-based method, with RBF across different numbers of function evaluations (NFEs).

We report qualitative and quantitative scaling results for quantity-aware image generation (Fig. 12, Tab. 4) and for compositional text-to-image generation (Fig. 13, Tab. 5), respectively. Our results indicate that allocating more compute leads to performance improvements for both BoN and RBF. However, the performance of BoN plateaus after 300 NFEs, whereas RBF continues to scale and achieves the highest reward in both tasks. Notably, RBF shows similar trend in the held-out reward, outperforming BoN and demonstrating its efficiency.

Time Complexity and Compute Analysis. We present time complexity of scaling methods in Tab. 6. Let S as the number of denoising steps, N as the NFE budget, and cs and cv as the costs of the denoising and verification, respectively. Since all methods share the same NFE budget N, the total denoising cost is fixed at N · cd. For the verification cost, although BoN has the lowest cost, RBF consistently outperforms BoN across all NFE budget regimes in both compositional text-to-image generation and quantity-aware image generation tasks (Fig. 12 and Fig. 13) while incurring only a marginal increase in verification overhead.

Additionally, at inference time, a user can specify the compute budget (NFEs), which determines the total runtime of our method. We report the runtime of RBF in Tab. 7. Under a 500-NFE budget, scaling for compositional text-to-image generation (VQAScore [31]) requires 635.01 seconds per image. Runtime can be reduced by lowering the NFE budget—at the cost of reward performance—and further accelerated by decreasing output resolution or increasing batch size.

#### Table 4: Quantitative results of quantityaware image generation in NFE scaling expriment. We use the same 100 prompts from T2I-CompBench [20]. † denotes the given reward.

NFEs RSS†

Acc. ↑ VQAScore [31] (held-out) ↑

Aesthetic Score [50] ↑

[34] ↓

50 4.360 0.400 0.758 5.408 100 3.280 0.510 0.750 5.522 300 2.190 0.570 0.755 5.463 500 1.760 0.580 0.756 5.420

BoN

1000 1.340 0.590 0.759 5.466

50 3.250 0.410 0.756 5.560 100 1.860 0.590 0.760 5.627 300 0.690 0.720 0.779 5.503 500 0.540 0.800 0.769 5.581

(Ours)RBF

1000 0.290 0.880 0.777 5.526

#### Table 5: Quantitative results of compositional text-to-image generation in NFE scaling expriment. We use the 121 prompts from GenAIBench [21]. † denotes the given reward.

NFEs VQAScore †

Inst.BLIP [10] (held-out) ↑

Aesthetic [50] ↑

[31] ↑

50 0.8310 0.8011 5.2246 100 0.8459 0.7959 5.2594 300 0.8775 0.8250 5.1414 500 0.8790 0.8200 5.1620

BoN

1000 0.8886 0.8269 5.2055

50 0.8577 0.8253 5.2704 100 0.8824 0.8212 5.3213 300 0.9146 0.8387 5.2837 500 0.9250 0.8430 5.2370

(Ours)RBF

1000 0.9283 0.8369 5.2593

#### Table 6: Time complexity of scaling methods.

SVDD [30], RBF

Base BoN SMC [23]

S ·cd N ·cd + NS ·cv N ·cd+N ·cv N ·cd+N ·cv

[Figure 84]

1000

500

300

100

300 1000 500

100

50

50

Figure 12: Quantity-aware image generation scaling behavior comparison of BoN and RBF. We plot the known reward (RSS) [34] against accuracy for different numbers of function evaluations: {50,100,300,500,1,000}. Note that the horizontal axis is displayed on a logarithmic scale.

[Figure 85]

300 500 1000

50

100 300

1000

500

50

100

Figure 13: Compositional text-to-image generation scaling behavior comparison of BoN and RBF. We plot the known reward (VQAScore) [31] against the held-out reward [10] for different numbers of function evaluations: {50,100,300,500,1,000}.

#### Table 7: Runtime of RBF.

50 100 300 500 1000

Runtime (sec) 84.00 140.11 383.89 635.01 1243.68 VQAScore [31] 0.858 0.882 0.915 0.925 0.928

For all experiments, we use FLUX [28], which requires approximately 32GB of GPU memory, accounting for the majority of overall memory usage. All evaluations are performed on an NVIDIA RTX A6000 GPU.

### F Implementation Details

#### F.1 Choice of Hyperparameters

We report quantitative results on aesthetic score [50] and diversity [23] for images generated under different settings of the number of denoising steps and the diffusion coefficient. As shown in Tab. 8(a), the number of denoising steps beyond 10 gives marginal gains. Hence, we fixed the number of denoising steps to 10 to ensure fair and efficient evaluation across all methods. Note that once the

Table 8: Choice of hyperparameters. Evaluation of the images generated with different (a) number of denoising steps and (b) diffusion coefficient.

Steps Aesthetic [50] Diversity [23]

10 5.635 0.084 20 5.680 0.103

(a) Number of denoising steps

Aesthetic [50] g(t) = t2

Diversity [23] g(t) = t2

Norm Aesthetic [50]

Diversity [23] g(t) = t

g(t) = t

1 5.635 0.084 5.652 0.083 3 5.168 0.153 5.436 0.158 5 4.608 0.223 4.838 0.187

(b) Diffusion coefficient

number of denoising steps is fixed, the total particle count per step is automatically determined by dividing the total NFE budget by the number of steps. Additionally, Tab. 8(b) reports results obtained under varying diffusion coefficients scaled by different norms. We found that using g(t) = 3t2 consistently offered the best trade-off between sample diversity and output fidelity, so we adopt this setting for all SDE sampling.

#### F.2 Compositional Text-to-Image Generation

In the compositional text-to-image generation task, we use the VQAScore as the reward, which evaluates image-text alignment using a visual question-answering (VQA) model (CLIP-FlanT5 [31] and InstructBLIP [10]). Specifically, VQAScore measures the probability that a given attribute or object is present in the generated image. To compute the reward, we scale the probability value by setting β = 0.1 in Eq. 3.

#### F.3 Quantity-Aware Image Generation

In quantity-aware image generation, text prompts specify objects along with their respective quantities. To generate images that accurately match the specified object counts, we use the negation of the Residual Sum of Squares (RSS) as the given reward. Here, RSS is computed to measure the discrepancy between the detected object count Cˆi and the target object count Ci in the text prompt: RSS = ni=1 Ci − Cˆi

2

, where n is the total number of object categories in the prompt. We

additionally report accuracy, which is defined as 1 when RSS = 0 and 0 otherwise. For the held-out reward, we report VQAScore measured with CLIP-FlanT5 [31] model.

Object Detection Implementation Details. To compute the given reward, RSS, it is necessary to detect the number of objects per category, Cˆi. Here, we leverage the state-of-the-art object detection model, GroundingDINO [34] and the object segmentation model SAM [25], which is specifically used to filter out duplicate detections.

We observe that naïvely using the detection model [34] to compute RSS leads to poor detection accuracy due to two key issues: inner-class duplication and cross-class duplication. Inner-class duplication occurs when multiple detections are assigned to the same object within a category, leading to overcounting. This often happens when an object is detected both individually and as part of a larger group. Cross-class duplication arises when an object is assigned to multiple categories due to shared characteristics (e.g.„ a toy airplane being classified as both a toy and an airplane), making it difficult to assign it to a single category.

To address inner-class duplication, we refine the object bounding boxes detected by GroundingDINO [34] using SAM [25] and filter out overlapping detections. Smaller bounding boxes are prioritized, and larger ones that significantly overlap with existing detections are discarded. This ensures that each object is counted only once within its category. To resolve cross-class duplication, we assign each object to the category with the highest GroundingDINO [34] confidence score which prevents duplicate counting across multiple classes.

#### More qualitative results are presented in the following pages.

### G Additional Qualitative Results

#### G.1 Comparisons of Inference-Time SDE Conversion and Interpolant Conversion

Linear-ODE Linear-SDE VP-SDE Linear-ODE Linear-SDE VP-SDE

“Six people gathered for a picnic.” “Four candles, two balloons, one dog, two tomatoes and three helicopters.”

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

SMC[23]

Quantity

“Four rabbits, three apples, two mice and four televisions.” “Seven pigs snorted and played in the mud.”

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

SMC[23]

Quantity

“Two frogs in tracksuits, competing in a high jump. The frog in blue tracksuit jumps higher than the frog not in blue tracksuit.”

“Three purple gemstones and one pink gemstone, with the pink gemstone having the smoothest looking surface.”

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

SMC[23]

Composition

Linear-ODE Linear-SDE VP-SDE Linear-ODE Linear-SDE VP-SDE

“Seven helmets” “Four couches, three candles, two fish, one frog and three plates.”

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

CoDe[53]

Quantity

“Seven lamps.” “Seven desks.”

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

CoDe[53]

Quantity

“A frog with a baseball cap is crouching on a lotus leaf, and another frog without a cap is crouching on a bigger lotus leaf.”

“In a collection of hats, each one is plain, but one is adorned with feathers.”

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

CoDe[53]

Composition

#### Figure 14: Additional qualitative results of inference-time SDE conversion and interpolant conversion.

Linear-ODE Linear-SDE VP-SDE Linear-ODE Linear-SDE VP-SDE

“Six bottles.” “Five hamburgers sizzled on the grill.”

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

SVDD[30]

Quantity

“Two men, four vases, four chickens and four ships.” “Six bicycles.”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

SVDD[30]

Quantity

“Two people and two bicycles in the street, the bicycle with the larger wheels belongs to the taller person.”

“There are two cups on the table, the cup without coffee is on the left of the other filled with coffee.”

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

SVDD[30]

Composition

Linear-ODE Linear-SDE VP-SDE Linear-ODE Linear-SDE VP-SDE

“Two giraffes, three eggs, two breads, three microwaves and four strawberries.”

“Four pears, four desks, three paddles and two rabbits.”

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

(Ours)RBF

Quantity

“Four balloons, one cup, four desks, two dogs and four microwaves.” “Seven women.”

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

(Ours)RBF

Quantity

“Two birds are chasing each other in the air, with the one flying higher having a long tail and the other bird having a short tail.”

“Three sailboats on the water, each with sails of a different color.”

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

(Ours)RBF

Composition

#### Figure 15: Additional qualitative results of inference-time SDE conversion and interpolant conversion.

#### G.2 Comparisons of Inference-Time Scaling

BoN SoP [39] SMC [23] CoDe [53] SVDD [30] RBF (Ours) “In a room, all the chairs are occupied except one.”

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

“Three mugs are placed side by side; the two closest to the faucet each contain a toothbrush, while the one furthest away is empty.”

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

“Three flowers in the meadow, with only the red rose blooming; the others are not open.”

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

“In a pack of wolves, each one howls at the moon, but one remains silent.”

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

“An open biscuit tin contains three biscuits, one without sultanas is square-shaped and the other two are round-shaped.”

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

“A rose that is not fully bloomed is higher than a rose that is already in bloom.”

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

“There are two colors of pots in the flower garden; all green pots have tulips in them and all yellow pots have no flowers in them.”

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

#### Figure 16: Additional qualitative results of compositional text-to-image generation task.

BoN SoP [39] SMC [23] CoDe [53] SVDD [30] RBF (Ours) “Seven balloons, four bears and four swans.”

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

“Six horses and six deer and four balloons.”

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

“Eight apples, three bicycles and five rabbits.”

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

“Six helicopters buzzed over eight pillows.”

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

“Five swans and seven ducks swam in the pond.”

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

“Four drums, seven tomatoes, and five candles.”

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

“Three chickens, four birds, and eight pears.”

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

“Six airplanes flying over a desert with seven camels walking below.”

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

#### Figure 17: Additional qualitative results of quantity-aware image generation task.

