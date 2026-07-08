## π-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs

Siting Wang1,2,3 Xiaofeng Wang1,4 Zheng Zhu1,† Minnan Pei2,3 Xinyu Cui2,3,5 Cheng Deng6 Jian Zhao5 Guan Huang1 Haifeng Zhang2,3 Jun Wang7,†

# arXiv:2603.02083v2[cs.RO]9Mar2026

### Abstract

Flow-based vision-language-action (VLA) models excel in embodied control but suffer from intractable likelihoods during multi-step sampling, hindering online reinforcement learning. We propose π-StepNFT (Step-wise Negative-aware Fine-Tuning), a critic-and-likelihood-free framework that requires only a single forward pass per optimization step and eliminates auxiliary value networks. We identify that wider exploration spaces necessitate finer-grained, step-wise guidance for alignment. Empirically, π-StepNFT unlocks latent potential on LIBERO with competitive few-shot robustness. Moreover, it achieves superior generalization on ManiSkill, outperforming value-based baselines in OOD scenarios by preventing overfitting to multimodal features. This property offers a scalable solution promising for complex real-world applications. Our implementation builds upon RLinf and is publicly available at https://wangst0181.github.io/pi-StepNFT/.

### 1. Introduction

Vision-language-action (VLA) models have recently demonstrated increasingly general capabilities, enabling robots to follow open-ended natural language instructions and perform complex tasks across diverse environments. While early approaches discretized actions into tokens (Zitkovich et al., 2023; Kim et al., 2024) or mapped observations to continuous regression features (Kim et al., 2025b), recent advances have converged on flow-matching-based policies (Black et al., 2026; 2025; Intelligence et al., 2025; Ye et al., 2025; Bjorck et al., 2025). These models, by integrating large-scale vision-language pretraining with generative action prediction, have established a new standard for complex manipulation tasks.

1GigaAI 2Institute of Automation, Chinese Academy of Sciences 3University of Chinese Academy of Sciences 4Tsinghua University 5Zhongguancun Academy 6The University of Edinburgh 7University College London. Correspondence to: Zheng Zhu <zhengzhu@ieee.org>.

Preprint. March 10, 2026.

A recent systematic analysis (Pan et al., 2025) suggests that after supervised fine-tuning (SFT) converges, the output of flow-based VLAs often collapses to a single mode. Consequently, their efficacy stems from a mechanism of stochasticity injection combined with supervised iterative correction. Injecting noise during training and iterative correction during inference enable the policy to resist error accumulation and adhere closely to the expert manifold during closed-loop execution. However, we contend that relying heavily on the density of expert demonstrations, SFT merely establishes a foundational behavioral manifold that resembles a narrow line, where the model often lacks the ability to recover once it deviates due to micro-perturbations during testing. To overcome this fragility, reinforcement learning (RL) is required, aiming not to learn from scratch, but to explore an expanded manifold around the expert trajectory, endowing the model with the local error-correction capabilities needed to mitigate state deviations.

However, training flow-based VLAs with RL faces fundamental bottlenecks. The deterministic nature of ordinary differential equation (ODE) trajectories confines the action exploration space entirely to the initial noise distribution. To this end, the multi-step ODE integration renders exact action likelihoods computationally intractable, as calculating gradients requires expensive Jacobian trace estimation or backpropagation through the solver. Faced with these challenges, existing solutions either bypass likelihoods by employing latent space value distillation (Li et al., 2025b) or training separate value functions to introduce explicit conditioning on trajectory quality (Intelligence et al., 2025), or approximate likelihoods via Gaussian parameterization at each denoising step (Chen et al., 2025). In contrast, DiffusionNFT (Zheng et al., 2026) offers a likelihood-free alternative from the image generation domain. It optimizes the flow field directly on the forward diffusion process and defines a contrastive improvement direction by splitting samples into positive and negative subsets.

Crucially, directly transferring Diffusion-NFT to embodied control reveals a fundamental domain gap. While the standard deterministic ODE sampling yields a safe but narrow manifold, it lacks the exploratory capacity for selfimprovement. Attempting to broaden this scope via stochastic differential equation (SDE) theoretically enables mani-

SFT (ODE): Narrow Expert Manifold Naïve SDE: Wider Space but Misaligned 𝝅𝝅-StepNFT: Wider Space & Finer Step

Stochasticity Injection Supervised Iterative Correction

4

Step-wise Noise-aware

3

Noise-aware Terminal Matching

Gaussian Noise Prior

[Figure 1]

1

[Figure 2]

[Figure 3]

𝒙𝒙𝒕𝒕

[Figure 4]

[Figure 5]

[Figure 6]

𝝁𝝁𝟎𝟎 𝒙𝒙𝒕𝒕

𝝁𝝁𝒕𝒕

𝒙𝒙𝟏𝟏′ 𝒗𝒗𝜽𝜽

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

𝒙𝒙𝒕𝒕

[Figure 11]

𝒙𝒙𝟏𝟏

𝒙𝒙𝟎𝟎

𝓛𝓛

𝒙𝒙𝟏𝟏 𝒙𝒙𝒕𝒕−

𝒙𝒙𝟎𝟎 𝒙𝒙𝟏𝟏 𝒙𝒙𝟎𝟎

𝒙𝒙𝟏𝟏′

Finer

𝒙𝒙𝒕𝒕

𝒙𝒙𝟎𝟎

Coarse

2

Solely Terminal with Noise

Figure 1. Comparison of training paradigms. Left (ODE): Terminal supervision is well-posed for deterministic ODEs but results in a narrow expert manifold. Middle (Naive SDE): Stochastic rollouts introduce a wider exploration space, but coarse terminal supervision fails to correct deviations, leading to misalignment. Right (π-StepNFT): Our method leverages the wider space from SDE but applies finer, step-wise ranking guidance to ensure robust alignment with the expert manifold.

fold expansion during exploration, yet in practice, it results in a wider space but misaligned policy. This failure stems from the nature of supervision. Unlike image generation that targets static distribution matching, embodied control is sequential and thus sensitive to compounding errors. Moreover, embodied control typically uses a short denoising path to meet interaction latency, and prior empirical evidence indicates diminishing returns from moderately longer paths (Chen et al., 2025), making fine-grained, step-wise denoising supervision feasible in practice.

put x0 to the immediate next denoising step xt−, using a noise-based regression to generate the precise local gradients needed for robust alignment. Third, regarding suppressed successful exploration, we identify that the previous objective in Diffusion-NFT suffers from an “implicit penalty”, which inadvertently suppresses policy updates to minimize the magnitude of branch separation. To overcome this, we introduce a logistic contrastive ranking loss, establishing a “push-pull dynamics”: maximizing the likelihood of successful trajectories while suppressing failed ones. This noise-based, step-wise, bidirectional and penalty-free signal enforces strict preference separation, enabling more aggressive and precise policy improvement.

As illustrated in Figure 1, under deterministic ODE rollouts, the intermediate state xt stays on a narrow trajectory, making “point-level” terminal matching ❶ on x0 relatively wellposed. However, under SDE rollouts, injected noises accumulate along the denoising path, and naively regressing the final x0 forces unstable point-to-point matching with highvariance gradients. A noise-aware view ❷ reveals that each denoising update induces a Gaussian transition, suggesting “region-level” supervision with variance-normalization; yet using only the solely terminal ❸ still yields a coarse correction that lags behind on-policy drift. This motivates step-wise supervision ❹ that targets the immediate next solver state xt−, providing fine-grained local guidance to stabilize stochastic exploration and accelerate convergence. To address these issues, we propose π-StepNFT, a criticand-likelihood-free online RL framework tailored for embodied VLAs. We achieve robust manifold expansion by systematically redesigning the interplay between exploration and supervision. First, regarding lack of exploration, effective policy improvement necessitates a wider behavioral space. We introduce an SDE-based sampling mechanism during training that augments deterministic rollouts with structured noise, forcing the model to traverse adjacent states to effectively inflate the behavioral manifold. Second, for supervision target mismatch, anchoring this expanded exploration demands finer-grained step-wise supervision. We shift the prediction target from the final denoised out-

In summary, our main contributions are as follows:

- • We propose π-StepNFT, a critic-and-likelihood-free online RL framework tailored for flow-based VLAs, which eliminates the need to train auxiliary value networks that are prone to multimodal overfitting, and requires only a single forward pass per optimization step.
- • We identify that wider exploration spaces induced by SDE necessitate finer-grained guidance. By shifting the supervision target from the terminal to the immediate next denoising step and incorporating a logistic contrastive ranking objective, we resolve the supervision mismatch and reinforce successful exploration.
- • We validate our approach through extensive experiments on LIBERO (Liu et al., 2023) and ManiSkill (Mu et al.,

2021) benchmarks. On LIBERO, π-StepNFT unlocks the policy’s potential in few-shot settings, achieving a 32.9% improvement over SFT. On ManiSkill, it demonstrates superior generalization in visually diverse OOD settings by preventing critic-induced multimodal overfitting, outperforming critic-based baselines by 11.1% in unseen scenarios, highlighting its potential for complex real-world deployment.

### 2. Related Works

#### 2.1. Online RL for VLAs

Recent VLA models have evolved from discrete tokenization (Zitkovich et al., 2023; Kim et al., 2024; Liu et al., 2026) to continuous flow-based policies (Ghosh et al., 2024; Black et al., 2026; 2025; Intelligence et al., 2025), which establish strong priors for manipulation. However, fine-tuning these flow-based VLAs faces the challenge of intractable action likelihoods due to multi-step ODE sampling. Existing solutions generally adopt two strategies to circumvent this: bypassing likelihood calculation via value distillation or preference feedback (e.g., GR-RL (Li et al., 2025b), π0∗.6), or approximating likelihoods by transforming ODEs into SDEs (e.g., πRL (Chen et al., 2025)). Similar to test-time scaling strategies (Yang et al., 2025; Song et al., 2025), noise injection in SDEs facilitates exploration, yet existing methods often struggle to balance exploration width with supervision granularity.

#### 2.2. Policy Optimization for Generative Models

To handle intractable likelihoods in generative policy optimization, prior works typically follow three paradigms. Explicit gradient and advantage methods (Black et al., 2023; Liu et al., 2025; Zhang et al., 2025b) treat denoising as a sequential process but often require expensive backpropagation through solvers. Reward-Weighted methods (Pfrommer et al., 2025; Fan et al., 2025; McAllister et al., 2025) avoid exact likelihoods by re-weighting regression targets, yet they can suffer from high variance in gradient estimation. Preference and contrastive methods (Wallace et al., 2024; Zhang

- et al., 2025a) offer a more stable alternative by aligning distributions via ranking. Most notably, Diffusion-NFT (Zheng
- et al., 2026) proposes a likelihood-free framework using implicit forward-process updates. Our work extends this efficient paradigm to embodied control, addressing the unique supervision gaps that arise when applying it to multi-step VLA policies. Please refer to Appendix B for detailed related works.

guity when both the RL objective and the flow dynamics appear in the same derivations.

#### 3.1. Flow Matching for VLA Models

We consider a VLA policy that generates continuous actions x0 ∈ Rd conditioned on context c. Flow-matching (Lipman et al., 2022) learns a time-dependent vector field vθ(x,t,c) to generate data by transforming a noise distribution x1 ∼ N(0,I) to the data distribution x0 over time t ∈ [0,1].

The standard flow-matching objective regresses the network prediction vθ onto the target field ut = x1 − x0 by minimizing LCFM(θ) = Et,x

0,x1[∥vθ(xt,t,c) − ut∥2], where xt = tx1 + (1 − t)x0.

ODE Sampling. In the standard deterministic setting, inference is performed by numerically integrating the ODE dx = vθ(x,t,c)dt from t = 1 to t = 0. Using a discrete step size δt > 0, the Euler update rule for the next step xt− (where t− = t − δt) is:

xt− = xt − vθ(xt,t,c)δt. (1)

While efficient for generation, this deterministic trajectory lacks the exploratory capability required for reinforcement learning.

SDE Sampling. To enable exploration, we adopt the reversetime SDE formulation (Liu et al., 2025), which injects stochasticity while preserving the marginal distribution. The Euler-Maruyama discretized update is given by:

σt2 2t

(xt + (1 − t)vθ(xt,t)) (−δt)

xt− = xt + vθ(xt,t) +

+ σt δtϵ, (2)

where ϵ ∼ N(0,I) provides exploration noise. This update step induces a Gaussian transition density qθ,t(xt− | xt,c) = N µθ,t(xt),Σt . Crucially, the mean of this transition is an affine transformation of the network output:

µθ,t(xt,c) = Ut(xt,t) + Bt(t)vθ(xt,t,c), (3)

### 3. Preliminaries

We use two time scales throughout the paper: environment steps and denoising time. An episode trajectory is indexed by environment steps i = 0,...,H − 1, yielding τ = {(si,ai)}Hi=0−1. Separately, the flow policy uses a continuous denoising time t ∈ [0,1] (or a discretization) to generate an action. When discretized, we use K solver steps with schedule 1 = t0 > t1 > ··· > tK = 0, and denote intermediate sampler states by {xt

j}Kj=0, where K is typically small in embodied control due to real-time constraints. Unless stated otherwise, t refers to denoising time in the sampler, while the environment index is i, avoiding ambi-

where Ut and Bt are pre-determined coefficients derived from the noise schedule (detailed in Appendix A.1).

This linear relationship allows us to propagate gradients efficiently from the transition target to the policy parameters without backpropagating through the ODE solver.

#### 3.2. RL Fine-tuning and the Likelihood Gap

We formulate the fine-tuning task as maximizing the expected return J(θ) over trajectories τ = (s0,a0,...): J(θ) = Eτ∼p

θ(τ)[R(τ)], where the trajectory distribution is determined by the environment dynamics and the policy: pθ(τ) = p(s0) Hi=0−1 πθ(ai|si)p(si+1|si,ai).

Algorithm 1 π-StepNFT: Step-wise Negative-aware FineTuning with Contrastive Ranking

Wider Space Needs Finer Steps. While effective for image generation, we observe that directly transferring the ODE-based formulation to embodied control yields suboptimal convergence. We attribute this domain gap to two critical factors, requiring us to establish a Wider Space for exploration anchored by Finer Steps of supervision (made practical by the typically short denoising path in embodied control):

Require: Flow policy πθold, simulator E, env steps H, solver steps K, schedule {tj}Kj=0, hyperparams β,λTR. Initialize θ ← θold, buffer D ← ∅. for each iteration m do

- // Phase 1: Data Collection for each task (initial state s0,language prompt c) do

Rollout H env steps using πθold. for i = 0 to H − 1 do

Run Flow-SDE sampler; get chain {xt

j}Kj=0. Sample j ∼ U{0,...,K − 1}; set t ← tj. Set xt ← xt

j

and xt− ← xt

j+1

.

Set vtold ← πθold(c,si,xt,t). Record di = (xt,xt−,vtold,t,si,c). Execute xt

K

in E and update si

end for Observe terminal r ∈ {0,1}; D ← {(di,r)}Hi=0−1.

end for

- // Phase 2: Optimization

- • Lack of Exploration: Deterministic ODE rollouts quickly collapse to a narrow manifold, failing to discover diverse solutions in high-dimensional action spaces. We instead adopt an SDE-based formulation to inject controlled noise. This active expansion creates the necessary wider space for the policy to traverse and learn from adjacent regions around the expert trajectory.
- • Supervision Target Mismatch: Operating within this

wider space renders standard terminal-x0 supervision unstable, as injected noises accumulate and amplify variance over the rollout horizon. To counteract this, we require finer steps of guidance: we supervise the immediate one-step transition xt → xt− with variance normalization, providing the precise, low-variance local gradients needed for robust alignment.

for each batch (xt,xt−,vtold,t,s,c,r) ∼ D do Pred vθ,t ← πθ(c,s,xt,t); Drift ∆vθ ← vθ,t −vtold. Construct mirrors: vθ± ← vtold ± β∆vθ. Calc means/var: µ±θ,t,Σt ← Mean Var(xt,vθ±,t). Calc errors: Eθ,t± ← ∥xt− − µ±θ,t∥2Σ−1

.

t

Set y ← 2r − 1 and ∆Eθ ← Eθ,t+ − Eθ,t− . Ltotal ← softplus 12y∆Eθ + λTR∥∆vθ∥2. Update θ ← θ − η∇θLtotal.

#### 4.1. Step-wise Transitions and Mirror Errors

We conduct rollouts using the Flow-SDE solver described in Section 3.1 with a rollout policy πθold which is updated with EMA across iterations. Each episode yields an environment trajectory τ = {(si,ai)}Hi=0−1. At each environment step i, the policy generates ai by running a K-step solver with schedule 1 = t0 > t1 > ··· > tK = 0, producing sampler states {x(ti)

end for θold ← αmθold + (1 − αm)θ; clear D.

end for Output: Optimized policy πθ.

}Kj=0; we execute the chunked terminal sample x(i)tK in the simulator as ai. For efficiency, we uniformly sample one solver index j ∼ U{0,...,K − 1} and define a single solver transition (xt,xt−,t) = (x(ti)

Standard policy gradient methods rely on the score function: ∇θJ(θ) = Eτ [ i ∇θ log πθ(ai|si) · Ψi], where Ψt is the advantage or return (e.g., REINFORCE (Williams, 1992), PPO (Schulman et al., 2017)).

j

,x(ti)

,tj), where t− denotes the next solver time point in the discretization. We additionally record the rollout velocity vold = πθold(c,xt,t). Each episode also provides a terminal optimality signal r(τ) ∈ [0,1].

j+1

j

However, for flow-based policies, calculating the explicit log-likelihood log πθ(ai|si) is computationally expensive and numerically unstable, as it requires integrating the instantaneous change of variables (Jacobian trace) along the entire generation trajectory. This intractability prevents the direct application of standard RL algorithms, motivating our likelihood-free approach.

Following Diffusion-NFT, we construct two “mirrored” velocity candidates v+θ and v−θ, symmetric around vold along the update direction ∆vθ = vθ − vold:

vθ+ = (1 − β)vold + βvθ, (4) vθ− = (1 + β)vold − βvθ, (5)

### 4. Method

We introduce π-StepNFT (Step-wise Negative-aware FineTuning), an online RL framework designed for flow-based VLA models, as shown in Algorithm 1. Our method is inspired by Diffusion-NFT (Zheng et al., 2026), which finetunes diffusion models using a weighted-MSE objective on the final denoised output x0 generated by ODE rollouts.

where β > 0 is a trust-region hyperparameter controlling how far we deviate from the rollout policy to estimate a local improvement signal. This construction ensures symmetry: vθ+ − vold = vold − vθ− = β∆vθ.

Importantly, under the Flow-SDE transition Eq. (3), the onestep mean is an affine function of the velocity, so these two velocity candidates induce two Gaussian transition means µ±θ,t = µt(v±θ) with shared covariance Σt. We then compute the variance-normalized step errors against the sampled next state xt−:

, Eθ,t− = ∥xt− − µ−θ,t∥2Σ−1

Eθ,t+ = ∥xt− − µ+θ,t∥2Σ−1

. (6)

t

t

Intuitively, Eθ,t+ measures how well the positive mirrored branch explains the observed stochastic transition, while

Eθ,t− measures the negative branch. Normalizing by Σt

(which reflects the injected noise level at solver time t) stabilizes gradient scales across timesteps. We next show how this yields a step-wise contrastive objective over mirrored perturbations.

#### 4.2. π-StepNFT: Step-wise Contrastive Objective

Given a sampled solver transition (xt → xt−) with terminal signal r(τ), we define y = 2r(τ) − 1. We use oracle to denote the ideal, outcome-conditioned comparison between p(xt− | xt,c,o=1) and p(xt− | xt,c,o=0), which is not directly observable. Our method replaces this with a computable ranking surrogate: we construct two symmetric perturbations around the rollout policy along the update direction and rank the two branches by which one assigns higher likelihood to the observed transition.

Definition 4.1 (π-StepNFT Objective). For a sampled solver transition tuple (xt,xt−,t,c) with episode label y ∈ [−1,1], let Eθ,t+ and Eθ,t− denote the step-wise errors defined in Section 4.1. The step-level objective is:

ℓt(θ) = softplus

- 1

- 2

y · (Eθ,t+ − Eθ,t− ) . (7)

Minimizing ℓt encourages Eθ,t+ < Eθ,t− when the episode is successful (y > 0) and reverses the inequality for failures

(y < 0). Intuitively, this ranks two local transition hypotheses along the update direction ∆vθ = vθ − vold, using the episode label as a weak preference signal.

Lemma 4.2 (Log-Likelihood Ratio). Under the shared covariance Σt, the difference in squared errors is proportional to the log-likelihood ratio of the two mirrored branches:

qθ,t+ (xt− | xt,c) qθ,t− (xt− | xt,c)

= −

log

- 1

- 2

(Eθ,t+ − Eθ,t− ). (8)

Proof. See Appendix A.2.

Lemma 4.2 shows that minimizing ℓt(θ) adjusts the constructed transition log-ratio log q

+ θ

qθ− according to the episode label y. The two densities qθ± arise from symmetric perturbations ±β∆vθ of the rollout policy, so this

log-ratio provides a directional signal on the same observed solver transition (xt → xt−). In contrast, DPO (Rafailov et al., 2023) ranks outcome-conditioned distributions under a shared context, whereas we rank update perturbations using only episode-level feedback. For y = +1, minimizing the loss increases the log-ratio, favoring updates that make the observed transition more likely under the positive perturbation; for y = −1, it encourages the opposite preference. Thus, ℓt yields a low-variance step-wise surrogate, and in Section 4.3 we show that under small-step assumptions its induced updates align with the oracle improvement direction from outcome-conditioned posterior splits.

#### 4.3. Validity and Optimized Direction

This section closes the conceptual loop between our constructed step-wise objective and the oracle improvement signal.

Oracle direction from posterior splits (not directly computable). Let o ∈ 0,1 denote the latent episode outcome under context c. Posterior splits (Appendix A.4) induce an outcome-conditioned decomposition of the rollout posterior at solver time t, which defines the oracle mean gap ∆µ⋆t(xt,c) (Lemma A.4). Intuitively, ∆µ⋆t captures how the one-step transition mean would change if we could condition the rollout transition on success versus failure; we treat it as an ideal local improvement direction in mean space. Importantly, this oracle quantity is a reference defined under outcome conditioning, and is not directly observable from online rollouts.

Proposition 4.3 (Bayes Monotonicity). For fixed (xt,c), the posterior P(o = 1 | xt−,xt,c) is strictly increasing in the oracle likelihood ratio p(xt−|xt,c,o=1)

p(xt−|xt,c,o=0). Proof. See Appendix A.3.

Proposition 4.3 provides the key monotonic link: increasing the oracle transition ratio on the observed transition (xt → xt−) strictly increases the posterior probability of success. This motivates seeking a step-wise objective that increases a success-vs-failure transition ratio, while acknowledging that the oracle-conditioned densities p(· | o) are inaccessible.

Computable surrogate via mirrored transitions (what we actually optimize). Since the oracle success–failure ratio

p(|o=1) p(|o=0) is not observable online, we optimize a constructed step-wise surrogate based on the mirrored perturbations defined in Section 4.1. By Lemma 4.2, our ranking loss is equivalent to increasing (for y > 0) or decreasing (for y < 0) the constructed transition ratio q

+ θ

qθ− evaluated on the observed transition (xt → xt−). We next show that, under a small-step regime, the expected gradient induced by this constructed ratio aligns with the oracle mean-gap direction.

Theorem 4.4 (Gradient Form and Small-Step Alignment). Let et = xt− − µoldt be the residual of the rollout mean, dt = µ+θ,t − µoldt be the displacement in mean space and Bt

be the affine coefficient from Eq. (3).

- (a) The error difference satisfies Eθ,t+ − Eθ,t− = −4⟨Σ−t 1et,dt⟩. (9)
- (b) Consequently, the gradient of the step loss ℓt is

−∇θℓt(θ) ∝ σ(zt)y

∂vθ ∂θ

⊤

BtΣ−t 1et, (10) where zt is the softplus logit and σ(·) is the sigmoid.

- (c) In the binary-success setting and for small updates

(vθ ≈ vold so σ(zt) ≈ const), the conditional expected direction aligns with the oracle mean gap:

⊤

∂vθ ∂θ

BtΣ−t 1∆µ⋆t(xt,c),

E[−∇θℓt(θ) | xt,c] ∥

(11)

where ∆µ⋆t is defined by posterior splits in Appendix A.4.

- Proof. See Appendix A.5.

- Theorem 4.4 provides the missing “closed loop”: posterior

splits define an oracle local improvement signal ∆µ⋆t, while our mirrored construction yields a computable surrogate ratio whose small-step expected gradient provably points in the same mean-space direction.

- 4.4. Comparison with Diffusion-NFT (Weighted-MSE)

Diffusion-NFT optimizes a reward-weighted regression objective. In our step-wise setting, the analogous form is ℓwMSEt (θ) = r Eθ,t+ + (1 − r)Eθ,t− , where Eθ,t± are the mirrored step errors from Section 4.1. We next show that this weighted-MSE contains an implicit separation penalty that can suppress branch separation (and hence policy updates), whereas our logistic ranking objective isolates the directional alignment signal and induces a clearer push–pull behavior.

- Theorem 4.5 (Separation Penalty in wMSE). The wMSE loss decomposes as:

ℓwMSEt (θ) = const − 2y⟨Σ−t 1et,dt⟩ + ∥dt∥2Σ−1

t

. (12)

- Proof. See Appendix A.6.

Here, defined in Section 4.3, et = xt− − µoldt is the rollout residual and dt = µ+θ,t − µoldt is the mirrored mean displacement. The middle term is the directional alignment

signal driven by y, while the last term ∥dt∥2Σ−1

is a separation penalty that discourages large branch displacement irrespective of y.

t

In contrast, the core directional term in π-StepNFT (derived in Theorem 4.4) depends only on the error difference:

Eθ,t+ − Eθ,t− = −4⟨Σ−t 1et,dt⟩. (13)

Implicit Penalty: The decomposition of the objective above shows that wMSE optimizes the same alignment term plus

an additional quadratic penalty ∥dt∥2Σ−1

. This penalty explicitly discourages branch separation (and thus suppresses the magnitude of the policy update), even when the data suggests a strong corrective move (i.e., large alignment between et and dt). In contrast, by using a logistic ranking loss, π-StepNFT removes this intrinsic suppression and preserves the alignment signal as the dominant driver for policy improvement.

t

Push-pull dynamics: In the binary case r ∈ {0,1}, the wMSE objective reduces to fitting only one branch: it pulls the selected branch toward the observed transition but does not explicitly push the other branch away. By contrast, our logistic ranking enforces a strict ordering between Eθ,t+ and Eθ,t− : for successful episodes it simultaneously pulls the positive branch and pushes the negative branch away (and vice versa for failures). This bidirectional signal yields stronger separation and typically sharper gradients during fine-tuning, which translates into faster convergence and higher asymptotic performance in our experiments.

### 5. Experiments

#### 5.1. Experimental Setup

Evaluation Benchmarks. We evaluate on 2 multitask benchmarks. For LIBERO (Liu et al., 2023), we follow the standard protocol across four suites (Spatial, Object, Goal, Long), reporting average success rates over 500 episodes (50 states × 10 sub-tasks) per suite. For ManiSkill (Mu et al., 2021), we adopt the PutOnPlateInScene multitask setting from RL4VLA (Liu et al., 2026) tested for generalization, which defines 4,352 compositional tasks derived from 16 objects, 17 receptacles, and 16 tabletop scenes.

Model Architectures. We employ π0 and π0.5, OpenPi’s flow-based VLAs combining a PaliGemma-3B (Beyer et al., 2024; Steiner et al., 2024) backbone with a ∼300M parameter flow-matching action expert. π0.5 incorporates an improved training paradigm. Adhering to official configurations, π0 uses vision, text, and proprioception, while π0.5 omits proprioception on LIBERO; this modality setting remains consistent across SFT and RL phases.

SFT Initialization. We initialize our policy using πRL checkpoints. For LIBERO, to prevent performance saturation from masking RL gains, we train on pruned subsets of the total 1,692 trajectories: π0 uses 58 trajectories for Spatial/Object/Goal and 208 for Long; π0.5 uses a unified

- Table 1. Success rates (%) on LIBERO in the few-shot setting.

Model Spatial Object Goal Long Avg. ∆ Avg. # Full SFT

Octo (Ghosh et al., 2024) 78.9 85.7 84.6 51.1 75.1 OpenVLA (Kim et al., 2024) 84.7 88.4 79.2 53.7 76.5 πfast (Pertsch et al., 2025) 96.4 96.8 88.6 60.2 85.5 OpenVLA-OFT (Kim et al., 2025b) 91.6 95.3 90.6 86.5 91.0 π0 96.8 98.8 95.8 85.2 94.2 π0.5 98.8 98.2 98.0 92.4 96.9 # Few-shot SFT + RL

π0

SFT 65.3 64.4 49.8 51.2 57.6 —

πRL (Flow-SDE + PPO) 98.4 99.4 96.2 90.2 96.0 +38.4 πRL (Flow-SDE + GRPO) 97.8 97.8 83.2 81.4 90.0 +32.4 π-StepNFT 93.5 98.0 83.7 86.7 90.5 +32.9

# Few-shot SFT + RL

π0.5

SFT 84.6 95.4 84.6 43.9 77.1 —

πRL (Flow-SDE + PPO) 99.6 100 98.8 93.0 97.9 +20.8 πRL (Flow-SDE + GRPO) 97.4 99.8 91.2 77.6 91.5 +14.4 π-StepNFT 97.8 100 98.2 79.8 94.0 +16.9

- Table 2. Success rates (%) on ManiSkill across In-Distribution (IND) and Out-Of-Distribution (OOD) settings.

OOD Vision Semantic Execution Avg.

Model IND

Full SFT 38.4 32.6 8.4 13.2 18.1 πRL (Flow-SDE + PPO) 78.8 61.1 25.4 31.5 39.3 π-StepNFT 79.2 69.1 49.1 33.1 50.4

π0

Full SFT 40.1 40.2 16.6 22.4 26.4 πRL (Flow-SDE + PPO) 90.9 68.0 34.5 45.4 49.3 π-StepNFT 85.4 76.9 56.6 45.1 59.5

π0.5

1.0

1.0

0.8

0.8

###### SuccessRate

SuccessRate

0.6

0.6

0.4

0.4

xt ( 0 =0.1) xt ( 0 =0.1)

SDE w/ Mean Correction

0.2

0.2

x0 ( 0 =0.9) x0 ( 0 =0.1)

SDE w/o Mean Correction

Deterministic ODE

0.0

0.0

0 50 100 150 200

0 50 100 150 200

Training Steps

Training Steps

(a) Stochastic rollouts. ODE vs. Flow-SDE variants.

(b) Regression target. x0 vs. step-wise xt− supervision.

- Figure 2. Flow-SDE sampling and step-wise supervision improve on-policy stability.

0 50 100 150 200

Training Steps

0.0

0.2

0.4

0.6

0.8

1.0

SuccessRate

Contrastive Ranking

wMSE

Positive Branch

Negative Branch

(a) Loss formulation. wMSE vs. contrastive ranking.

0 50 100 150 200

Training Steps

0.0

0.2

0.4

0.6

0.8

1.0

SuccessRate

Terminal Binary Reward

Normalized GRPO Adv

Normalized GAE Adv

(b) Critic-free learning. Sparse labels vs. advantage signals.

- Figure 3. Contrastive ranking enables stable critic-free learning.

few-shot set of 40 trajectories (1 per sub-task). As for ManiSkill, we use the full 16,384 trajectories due to task complexity.

RL Training Protocol. We freeze the VLM backbone and fine-tune only the action expert. Training utilizes the RLinf (Yu et al., 2025) framework, which maximizes throughput by co-locating the environment, rollout policy, and actor on the same GPU. Main experiments were conducted on 8× NVIDIA H100 (80GB) GPUs; ablations used 8× NVIDIA RTX 4090 (48GB). Hyperparameters are detailed in the Appendix C.2.

- 5.2. Main Results LIBERO: Unlocking potential from few-shot SFT.

Table 1 reveals that SFT baselines are constrained by a narrow expert manifold, yielding initial success rates of only 57.6% (π0) and 77.1% (π0.5). π-StepNFT unlocks the model’s latent capacity via “wider space” exploration, significantly boosting average performance to 90.5% and 94.0%, respectively. Notably, on short-horizon tasks (e.g., Object), our method achieves performance comparable to PPO. Regarding alignment, while critic-based methods (PPO) maintain an advantage in long-horizon tasks due to temporal credit assignment, our method notably outperforms the critic-free GRPO baseline (e.g., 86.7% vs. 81.4% on π0 Long), demonstrating that step-wise supervision offers

highly competitive guidance without the need for estimating advantages.

#### ManiSkill: Critic-free generalization.

Unlike LIBERO, ManiSkill features high visual diversity, requiring generalization to unseen textures, objects and positions (OOD). Value-based methods estimate values from vision-language embeddings, which often causes critics to overfit to nuisance visual features and specific language prompts rather than task semantics. π-StepNFT bypasses this by relying on ground-truth outcomes. As shown in Table 2, while PPO is competitive in IND settings, π-StepNFT dominates in OOD scenarios. For π0, it achieves an OOD average of 50.4% (+11.1% over PPO), nearly doubling success rates on Semantic shifts (unseen objects/instructions) to 49.1%. This robust trend holds for π0.5 (59.5% OOD average vs. 49.3%), confirming that critic-free supervision effectively mitigates visual overfitting.

#### 5.3. Ablations Studies

To verify the efficacy of π-StepNFT, we conduct a component-wise analysis aligned with the challenges identified in Section 1. First, we decouple the effects of stochastic exploration, investigating whether SDE sampling with noise-aware correction is strictly necessary for manifold expansion compared to deterministic ODEs. Second, we analyze supervision granularity, contrasting our step-wise

1.0

1.0

1.0

0.8

0.8

0.8

SuccessRate

SuccessRate

SuccessRate

0.6

0.6

0.6

=0.1

:0.0001 0.995

=0.05

- =0.5

- =1.0

- =2.0

0.4

0.4

0.4

:0.01 0.995

- =0.1

- =0.2

- =0.3

:0.1 0.995

0.2

0.2

0.2

:0.1 (Constant)

=5.0

=0.5

:0.9 0.995

0.0

0.0

0.0

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training Steps

Training Steps

Training Steps

(c) Decay α. Figure 4. Hyperparameter sensitivity analysis. Configuration selected for main experiments is highlighted by the bold pink curves.

(a) Noise level σ.

(b) Trust region size β.

xt− target against the standard terminal x0 regression to demonstrate its impact on training stability and convergence speed. Third, we evaluate the learning objective, comparing our contrastive ranking formulation against weighted-MSE to highlight the benefits of removing the implicit separation penalty. Finally, we assess the necessity of explicit critics, showing that our likelihood-free framework achieves competitive performance using only sparse binary rewards, and provide a sensitivity analysis for key hyperparameters.

Impact of Stochastic Exploration. To isolate the benefit of exploration, we compare rollout strategies while fixing the regression target to the final denoised output x0 and using a conservative EMA schedule (α: 0.9 → 0.995). As shown in Figure 2a, deterministic ODE rollouts ❶(Figure 1) plateau early, confirming that restricted state coverage hinders policy improvement. While standard SDE ❷ widens the visited manifold, significant performance gains are only realized when the objective explicitly accounts for injected noise via mean correction ❸ (Eq. (3)). This indicates that effective exploration requires not just traversing a wider space, but utilizing a learning signal that mathematically aligns the noisy transition back to the policy’s velocity field.

Regression Target Granularity. We evaluate the efficacy of terminal x0 versus step-wise xt− regression targets under stochastic rollouts in Figure 2b. Empirical results demonstrate that supervision via x0 induces significant instability, necessitating overly conservative synchronization to prevent policy collapse. Conversely, the step-wise target xt− ❹ facilitates stable, near on-policy learning even under aggressive updates, thereby accelerating convergence. This suggests that precise, local supervision is essential to counteract the distribution shift introduced by active exploration, whereas terminal targets provide gradients that are too coarse for effective manifold adherence .

Objective Formulation: Ranking vs. wMSE. We benchmark the proposed contrastive ranking objective against a weighted-MSE (wMSE) baseline and single-branch ablations in Figure 3a. We observe that utilizing exclusively Positive or Negative branches yields partial improvement. This confirms that valid gradient signals exist in both directions, yet combining them yields superior performance.

Crucially, the wMSE baseline underperforms because, in the binary reward setting, it degenerates to fitting a single branch . Consequently, it fails to leverage both positive and negative signals simultaneously. In contrast, our ranking objective establishes a ”push-pull” dynamic by utilizing both branches to enforce strict preference separation. This effectively removes the ”implicit separation penalty” that otherwise suppresses the policy update magnitude.

Necessity of Value Estimation. We investigate the tradeoff between supervision density and training complexity in Figure 3b. Although dense step-level value estimates can in principle improve long-horizon credit assignment, sparse trajectory-level outcomes remain highly competitive for general manipulation tasks. Empirically, binary supervision yields smoother training because it relies on accurate environment feedback rather than approximate value estimates. Unlike image generation, where sample quality varies continuously and advantage-style soft weighting is more natural, embodied control typically has discrete success-or-failure outcomes, similar to mathematical reasoning (Chen et al., 2026). Correspondingly, treating r as a bounded trajectory success probability r ∈ [0,1] avoids the instability of unbounded advantage scores and reduces the need for normalization and clipping. Our probability-based formulation is also compatible with denser supervision: the sparse signal can be replaced by an offline critic-learned that predicts step-wise success probabilities, enabling finer credit assignment without changing the architecture.

Hyperparameter Sensitivity and Robustness. Figure 4 illustrates key trade-offs. For noise level σ, excessive noise impedes convergence by overly expanding the search space, while insufficient noise limits exploration. For trust region size β, results indicate that values around [1.0,2.0] are optimal; larger β violate local linearity, whereas smaller steps induce gradient instability. Regarding the decay α, a dynamic strategy proves most effective. High decay (slow updates) causes significant off-policy lag and lowers the performance ceiling, while constant or overly aggressive updates risk collapse. A dynamic schedule that progressively increases decay balances initial acceleration with final stability, which matches the small-step alignment in Theorem 4.4.

### 6. Conclusion

In this paper, we introduced π-StepNFT, a critic-andlikelihood-free framework for flow-based VLAs that structurally eliminates auxiliary value networks and requires only a single forward pass per step. We identified that wider exploration spaces necessitate finer-grained, step-wise guidance for effective alignment. Empirically, π-StepNFT unlocks latent potential on LIBERO in few-shot SFT settings and achieves superior OOD generalization on ManiSkill by preventing multimodal overfitting. These results establish a scalable, robust paradigm for fine-tuning generalist robot policies in complex real-world scenarios.

### Impact Statement

This paper introduces a framework for fine-tuning flowbased vision-language-action (VLA) policies to improve the efficiency and robustness of embodied agents. Beyond algorithmic advances, our work has implications for the accessibility and sustainability of robotic learning.

Democratization of embodied AI research: Training large-scale VLA models often requires substantial compute, in part due to the cost of differentiating through ODE trajectories. By proposing a likelihood-free, critic-free approach that uses a single forward pass per optimization step, π-StepNFT lowers the hardware barrier. This reduced overhead can broaden participation by smaller labs and academic groups, supporting a more diverse research community.

Safety and robustness: By improving out-of-distribution (OOD) generalization, our method can yield agents that behave more reliably in unstructured real-world settings. While increased capability may introduce dual-use concerns, our fine-grained supervision encourages adherence to expert manifolds and may reduce unpredictable behaviors during deployment.

### References

Bai, J., Yu, X., Xu, M., Lu, W., Pan, X., Maeng, K., Kifer, D., Wang, J., and Wang, Y. Towards better optimization for listwise preference in diffusion models. arXiv preprint arXiv:2510.01540, 2025.

Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Bjorck, J., Casta˜neda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S.

Training diffusion models with reinforcement learning. 2023.

Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M. R., Finn, C., Fusai, N., Galliker, M. Y., et al. π0.5: a vision-language-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., LiBell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L. X., Tanner, J., Vuong, Q., Walling, A., Wang, H., and Zhilinsky, U. π0: A vision-language-action flow model for general robot control, 2026. URL https:

//arxiv.org/abs/2410.24164.

Chen, H., Zheng, K., Zhang, Q., Cui, G., Cui, Y., Ye, H., Lin, T.-Y., Liu, M.-Y., Zhu, J., and Wang, H. NFT: Bridging supervised learning and reinforcement learning in math reasoning. In The Fourteenth International Conference on Learning Representations, 2026. URL https:// openreview.net/forum?id=ujBrsQm6Zu.

Chen, K., Liu, Z., Zhang, T., Guo, Z., Xu, S., Lin, H., Zang, H., Li, X., Zhang, Q., Yu, Z., Fan, G., Huang, T., Wang, Y., and Yu, C. πRL: Online rl fine-tuning for flow-based vision-language-action models, 2025. URL https://arxiv.org/abs/2510.25889.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Fan, J., Shen, S., Cheng, C., Chen, Y., Liang, C., and Liu, G. Online reward-weighted fine-tuning of flow matching with wasserstein regularization. In The Thirteenth International Conference on Learning Representations, 2025.

Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. Reinforcement learning for fine-tuning text-to-image diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:

//openreview.net/forum?id=8OTPepXzeh.

Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., Luo, J., Tan, Y. L., Chen, L. Y., Sanketi, P., Vuong, Q., Xiao, T., Sadigh, D., Finn, C., and Levine, S. Octo: An opensource generalist robot policy, 2024. URL https: //arxiv.org/abs/2405.12213.

Han, J., Wang, A., Xu, M., Chu, W., Dang, M., Yue, Y., and Ermon, S. Discrete diffusion trajectory alignment via stepwise decomposition. arXiv preprint arXiv:2507.04832, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Intelligence, P., Amin, A., Aniceto, R., Balakrishna, A., Black, K., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., DiCarlo, J., et al. π0∗.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

Kim, D., Lyu, S., Kim, S. W., and Seo, P. H. Direct diffusion score preference optimization via stepwise contrastive policy-pair supervision. arXiv preprint arXiv:2512.23426, 2025a.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Kim, M. J., Finn, C., and Liang, P. Fine-tuning visionlanguage-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025b.

Li, H., Zuo, Y., Yu, J., Zhang, Y., Yang, Z., Zhang, K., Zhu, X., Zhang, Y., Chen, T., Cui, G., et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025a.

Li, Y., Ma, X., Xu, J., Cui, Y., Cui, Z., Han, Z., Huang, L., Kong, T., Liu, Y., Niu, H., et al. Gr-rl: Going dexterous and precise for long-horizon robotic manipulation. arXiv preprint arXiv:2512.01801, 2025b.

Liao, X., Wei, W., Qu, X., and Cheng, Y. Step-level reward for free in rl-based t2i diffusion model fine-tuning. arXiv preprint arXiv:2505.19196, 2025.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training

flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

Liu, J., Gao, F., Wei, B., Chen, X., Liao, Q., Wu, Y., Yu, C., and Wang, Y. What can rl bring to vla generalization? an empirical study, 2026. URL https://arxiv.org/ abs/2505.19789.

Lu, G., Guo, W., Zhang, C., Zhou, Y., Jiang, H., Gao, Z., Tang, Y., and Wang, Z. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.

McAllister, D., Ge, S., Yi, B., Kim, C. M., Weber, E., Choi, H., Feng, H., and Kanazawa, A. Flow matching policy gradients. arXiv preprint arXiv:2507.21053, 2025.

Mu, T., Ling, Z., Xiang, F., Yang, D., Li, X., Tao, S., Huang, Z., Jia, Z., and Su, H. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. arXiv preprint arXiv:2107.14483, 2021.

NVIDIA. Nvidia h100 tensor core gpu datasheet. Technical report, NVIDIA, 2022. URL https://resources. nvidia.com/en-us-gpu-resources/ h100-datasheet-24306. Technical Report.

NVIDIA. Nvidia ampere gpu architecture whitepaper. Technical report, NVIDIA, 2023. URL https://images. nvidia.com/aem-dam/Solutions/geforce/ ada/nvidia-ada-gpu-architecture.pdf. Technical Report.

Pan, C., Anantharaman, G., Huang, N.-C., Jin, C., Pfrommer, D., Yuan, C., Permenter, F., Qu, G., Boffi, N., Shi, G., et al. Much ado about noising: Dispelling the myths of generative robotic control. arXiv preprint arXiv:2512.01809, 2025.

Pei, M., Li, G., Si, J., Zhu, Z., Mo, Z., Wang, P., Song, Z., Liang, X., and Cheng, J. Gcc: A 3dgs inference architecture with gaussian-wise and cross-stage conditional processing. In Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture, pp. 1824– 1837, 2025.

Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Pfrommer, S., Huang, Y., and Sojoudi, S. Reinforcement learning for flow-matching policies. arXiv preprint arXiv:2507.15073, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model.

Advances in neural information processing systems, 36: 53728–53741, 2023.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al. Smolvla: A vision-languageaction model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

Song, H., Qu, D., Yao, Y., Chen, Q., Lv, Q., Tang, Y., Shi, M., Ren, G., Yao, M., Zhao, B., et al. Hume: Introducing system-2 thinking in visual-language-action model. arXiv preprint arXiv:2505.21432, 2025.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Steiner, A., Pinto, A. S., Tschannen, M., Keysers, D., Wang, X., Bitton, Y., Gritsenko, A., Minderer, M., Sherbondy,

- A., Long, S., et al. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024.

Sun, M., Ding, P., Zhang, W., and Wang, D. Iterative refinement of flow policies in probability space for online reinforcement learning. arXiv preprint arXiv:2510.15388, 2025.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

Wagenmaker, A., Nakamoto, M., Zhang, Y., Park, S., Yagoub, W., Nagabandi, A., Gupta, A., and Levine, S. Steering your diffusion policy with latent space reinforcement learning. arXiv preprint arXiv:2506.15799, 2025.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Wang, F. and Yu, Z. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Wu, Y.-L., Ruan, B.-K., Tseng, C., and Shuai, H.-H. Ranking-based preference optimization for diffusion models from implicit user feedback. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Yang, S., Zhang, Y., He, H., Pan, L., Li, X., Bai, C., and Li, X. Steering vision-language-action models as antiexploration: A test-time scaling approach. arXiv preprint arXiv:2512.02834, 2025.

Ye, A., Wang, B., Ni, C., Huang, G., Zhao, G., Li, H., Li, J., Zhu, J., Feng, L., et al. Gigabrain-0: A world modelpowered vision-language-action model. arXiv preprint arXiv:2510.19430, 2025.

Yu, C., Wang, Y., Guo, Z., Lin, H., Xu, S., Zang, H., Zhang, Q., Wu, Y., Zhu, C., Hu, J., et al. Rlinf: Flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint arXiv:2509.15965, 2025.

Zhang, T., Da, C., Ding, K., Yang, H., Jin, K., Li, Y., Gao, T., Zhang, D., Xiang, S., and Pan, C. Diffusion model as a noise-aware latent reward model for step-level preference optimization. arXiv preprint arXiv:2502.01051, 2025a.

Zhang, T., Yu, C., Su, S., and Wang, Y. Reinflow: Finetuning flow matching policy with online reinforcement learning, 2025b. URL https://arxiv.org/abs/ 2505.22094.

Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., and Liu, M.-Y. DiffusionNFT: Online diffusion reinforcement with forward process. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.

net/forum?id=VJZ477R89F.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023.

- A. Theoretical Analysis and Proofs In this section, we provide detailed proofs for the lemmas, propositions, and theorems presented in the main text.

- A.1. Proof of Equation 3 (Affine Mean Derivation) We derive the explicit forms of Ut and Bt stated in Equation 3.

The flow-SDE solver computes the next mean µt(v) by mixing the rectified flow endpoints xpred0 = xt − v · t and xpred1 = xt + v · (1 − t) with weights derived from the Euler-Maruyama discretization. The weights are given by:

w0 = 1 − t + δt, w1 = (t − δt) −

σt2δt 2t

. (14)

The mean is given by the linear combination:

µt(v) = w0xpred0 + w1xpred1

= w0(xt − vt) + w1(xt + v(1 − t))

= (w0 + w1)xt + − tw0 + (1 − t)w1 v. (15) Directly computing the coefficients for xt and v yields: Coefficient for xt (Ut):

Ut = w0 + w1 = (1 − t + δt) + t − δt −

σt2δt 2t

= 1 −

σt2δt 2t

. (16)

Coefficient for v (Bt):

Bt = −tw0 + (1 − t)w1

= −t(1 − t + δt) + (1 − t) t − δt −

σt2δt 2t

= −t + t2 − tδt + (t − δt −

σt2δt 2t − t2 + tδt + t

σt2δt 2t

)

= −δt − (1 − t)

σt2δt 2t

. (17)

Thus, µt(v) = Ut(xt,t) + Bt(t)v, matching the affine form in Equation (3).

| |
|---|

- A.2. Proof of Lemma 4.2 (Log-Likelihood Ratio)

We prove that the difference in variance-normalized errors equals the log-likelihood ratio. Let q(x) = N(µ,Σ). Its log-density is:

- 1

- 2

- 1

- 2

(x − µ)⊤Σ−1(x − µ) −

log det(2πΣ). (18)

log q(x) = −

For the two branches qt+ = N(µ+t ,Σt) and qt− = N(µ−t ,Σt), they share the same covariance Σt. Subtracting their log-densities cancels the normalization constant:

- 1

- 2

log qt+(xt−) − log qt−(xt−) = −

(xt− − µ+t )⊤Σ−t 1(xt− − µ+t ) − (xt− − µ−t )⊤Σ−t 1(xt− − µ−t )

- 1

- 2 ∥xt− − µ+t ∥2Σ−1

##### − ∥xt− − µ−t ∥2Σ−1

= −

t

t

- 1

- 2

(Eθ+ − Eθ−). (19) This concludes the proof.

= −

| |
|---|

- A.3. Proof of Proposition 4.3 (Bayes Monotonicity) Fix (xt,c) and denote the prior success probability conditioned on (xt,c) by π(xt,c) ≜ P(o = 1 | xt,c) ∈ (0,1). By Bayes’ rule, the posterior success probability given the observed next state xt− is

η(xt−;xt,c) ≜ P(o = 1 | xt−,xt,c) =

p(xt− | xt,c,o = 1)π(xt,c) p(xt− | xt,c,o = 1)π(xt,c) + p(xt− | xt,c,o = 0)(1 − π(xt,c))

. (20) Define the oracle likelihood ratio

Λ(xt−;xt,c) ≜

p(xt− | xt,c,o = 1) p(xt− | xt,c,o = 0)

. Dividing the numerator and denominator by p(xt− | xt,c,o = 0) yields

η(xt−;xt,c) =

Λ(xt−;xt,c)π(xt,c) Λ(xt−;xt,c)π(xt,c) + (1 − π(xt,c))

. (21)

For constants a = π(xt,c) > 0 and b = 1 − π(xt,c) > 0, consider f(λ) = aλaλ+b for λ > 0. Its derivative is

f′(λ) =

ab (aλ + b)2

> 0.

Therefore η(xt−;xt,c) is strictly increasing in the likelihood ratio Λ(xt−;xt,c), completing the proof.

| |
|---|

- A.4. Oracle Splits from Diffusion-NFT

Notation and symbol disambiguation. We use κt(xt | x0) to denote the forward/noising kernel that maps a terminal sample x0 to an intermediate noisy state xt (as in diffusion models).

Setup. Fix a context c. Let x0 denote the terminal sample (e.g., the final solver output used to form an action), with rollout terminal distribution π0old(x0 | c) induced by πθold. Let κt(xt | x0) be the forward kernel and define the induced marginal

πtold(xt | c) = κt(xt | x0)π0old(x0 | c)dx0, and the diffusion posterior

κt(xt | x0)π0old(x0 | c) πtold(xt | c)

π0old|t (x0 | xt,c) =

.

Optimality variable. Introduce a latent optimality variable o ∈ {0,1} and an instance-level score r(x0,c) ∈ [0,1] satisfying r(x0,c) = P(o = 1 | x0,c). This is the same setup as Diffusion-NFT; see their Appendix for proofs.

- Lemma A.1 (Distribution Split (Diffusion-NFT)). Let p(c) ≜ Ex

0∼π0old(·|c)[r(x0,c)]. Define the oracle terminal distributions

π0+(x0 | c) =

r(x0,c)π0old(x0 | c) p(c)

, π0−(x0 | c) =

(1 − r(x0,c))π0old(x0 | c) 1 − p(c)

. Then

π0old(x0 | c) = p(c)π0+(x0 | c) + (1 − p(c))π0−(x0 | c). Reference. Diffusion-NFT Appendix (Distribution Split).

- Lemma A.2 (Posterior Split (Diffusion-NFT)). Let π0±|t(x0 | xt,c) be the posteriors induced by π0±:

κt(xt | x0)π0±(x0 | c) πt±(xt | c)

, πt±(xt | c) = κt(xt | x0)π0±(x0 | c)dx0. Then the rollout posterior satisfies

π0±|t(x0 | xt,c) =

π0old|t (x0 | xt,c) = α(xt,c)π0+|t(x0 | xt,c) + 1 − α(xt,c) π0−|t(x0 | xt,c), where the mixing weight is

πt+(xt | c) πtold(xt | c)

α(xt,c) = P(o = 1 | xt,c) = p(c)

.

Reference. Diffusion-NFT Appendix (Posterior Split). Corollary A.3 (Posterior Expectation Split). Under Lemma A.2, for any integrable function ϕ(x0),

Eπold

0|t(·|xt,c)[ϕ(x0)] = α(xt,c)Eπ+

0|t(·|xt,c)[ϕ(x0)] + (1 − α(xt,c))Eπ−

0|t(·|xt,c)[ϕ(x0)].

Oracle vs. constructed branches. The oracle objects (π0±,π0±|t) are defined by conditioning on the latent outcome o. In contrast, our method constructs mirrored branches vθ± = vold ± β(vθ − vold) and the induced solver transitions qθ,t± (xt− | xt,c) (Section 4.1).

Lemma A.4 (Oracle Velocity/Mean Splits (for alignment)). Assume the (oracle) velocity field at solver time t can be expressed as a posterior expectation under π0|t(· | xt,c), i.e., v(xt,c,t) = Eπ

0|t(·|xt,c)[ψ(x0,xt,c,t)] for some function ψ. Define oracle velocities v± by taking the same expectation under π0±|t. Then

vold(xt,c,t) = α(xt,c)v+(xt,c,t) + (1 − α(xt,c))v−(xt,c,t).

If additionally the one-step solver mean admits the affine form µt(v) = Atxt + Btv (Eq. 3), then

µoldt (xt,c) = α(xt,c)µ+t (xt,c) + (1 − α(xt,c))µ−t (xt,c), ∆µ⋆t(xt,c) ≜ µ+t (xt,c) − µ−t (xt,c) = Bt v+ − v− . Remark. This lemma is a direct consequence of Corollary A.3 and linearity of expectation.

#### A.5. Proof of Theorem 4.4 (Gradient Form and Alignment)

Fix a sampled training tuple (xt,xt−,c) collected under the rollout policy vold = πθold(c,xt,t). Let the rollout one-step mean be µoldt := µt(vold) and define the residual et ≜ xt− − µoldt .

Recall that our constructed mirrored branches are vθ± = vold ± β(vθ − vold), ∆vθ ≜ vθ − vold, and that the one-step mean admits the affine form (Eq. 3) µt(v) = Atxt + Btv, with shared covariance Σt.

- Step 1: Constructed mean shift. Define the constructed branch means µ±θ,t ≜ µt(vθ±). Using the affine mean form,

µ+θ,t − µoldt = Bt (vθ+ − vold) = Bt β(vθ − vold) = βBt∆vθ ≜ dt, (22) µoldt − µ−θ,t = Bt (vold − vθ−) = Bt β(vθ − vold) = dt. (23)

Hence,

µ±θ,t = µoldt ± dt, dt = βBt∆vθ. (24)

- Step 2: Error difference (Theorem 4.4.a). Recall the step errors Eθ± = ∥xt− − µ±θ,t∥2Σ−1

t

.

Substituting µ±θ,t = µoldt ± dt and et = xt− − µoldt gives Eθ+ − Eθ− = ∥et − dt∥2Σ−1

t

− ∥et + dt∥2Σ−1

t

= ∥et∥2Σ−1

t

+ ∥dt∥2Σ−1

t

− 2⟨et,dt⟩Σ−t 1 − ∥et∥2Σ−1

t

+ ∥dt∥2Σ−1

t

+ 2⟨et,dt⟩Σ−t 1

= −4⟨et,dt⟩Σ−t 1 = −4⟨Σ−t 1et,dt⟩, (25)

- which proves part (a).

- Step 3: Gradient form (Theorem 4.4.b). Define the logit zt ≜ 12y(Eθ+ − Eθ−), ℓt(θ) = softplus(zt). Using ∇softplus(z) = σ(z),

∇θℓt(θ) = σ(zt)∇θzt. From Step 2,

Eθ+ − Eθ− = −4e⊤t Σ−t 1dt,

thus

##### ∇θzt = 21y ∇θ(Eθ+ − Eθ−) = 12y ∇θ −4e⊤t Σ−t 1dt .

During optimization we treat the rollout branch vold (hence µoldt and et) as constant with respect to θ, so only dt depends on θ. From Equation (24), dt = βBt(vθ − vold) and therefore

⊤

∂vθ ∂θ

∇θdt = βBt∇θvθ, equivalently (∇θdt)⊤ =

βBt⊤. Absorbing constant scalar factors into ∝, we obtain

⊤

∂vθ ∂θ

BtΣ−t 1et, (26)

−∇θℓt(θ) ∝ σ(zt)y

- which proves part (b).

- Step 4: Small-step alignment (general r ∈ [0,1] and binary case). We relate the conditional expected update direction to an oracle improvement signal. Recall the signed label is defined in the main text as y ≜ 2r − 1, r ∈ [0,1], where r is the observed terminal signal (e.g., success indicator or a normalized score). Conditioned on (xt,c), the rollout residual et ≜ xt− − µoldt (xt,c) is zero-mean since µoldt (xt,c) = E[xt− | xt,c], hence

##### E[et | xt,c] = 0. (27)

Therefore,

E[yet | xt,c] = E[(2r − 1)et | xt,c]

= 2E[ret | xt,c] − E[et | xt,c]

= 2E[ret | xt,c]. (28)

- (i) General case (r ∈ [0,1]). Equation (28) shows that the conditional expected direction is governed by the correlation

between the terminal signal r and the local rollout residual et. In general, this produces a mixture of success- and failureassociated components (and does not reduce to a single oracle mean-gap direction without additional assumptions relating r to the latent optimality variable).

- (ii) Binary case (r ∈ {0,1} with r = o). In sparse-success RL we often take r to be the episode success indicator, i.e., r = o ∈ {0,1} where o is the latent optimality variable. We have α(xt,c) ≜ P(o = 1 | xt,c) in Lemma A.2. Then using the indicator property of o and Equation (27),

E[yet | xt,c] = E[(2o − 1)et | xt,c]

= 2E[oet | xt,c] − E[et | xt,c]

= 2E[o(xt− − µoldt ) | xt,c]

= 2P(o = 1 | xt,c)E[xt− − µoldt | xt,c,o = 1]

= 2α(xt,c) µ+t (xt,c) − µoldt (xt,c) , (29)

where µ+t (xt,c) ≜ E[xt− | xt,c,o = 1] is the oracle success-branch mean. By Lemma A.4, the oracle mean mixture identity holds:

µoldt (xt,c) = α(xt,c)µ+t (xt,c) + (1 − α(xt,c))µ−t (xt,c), hence

µ+t (xt,c) − µoldt (xt,c) = (1 − α(xt,c)) µ+t (xt,c) − µ−t (xt,c) = (1 − α(xt,c))∆µ⋆t(xt,c), where ∆µ⋆t(xt,c) ≜ µ+t (xt,c) − µ−t (xt,c) is the oracle mean gap. Substituting into Equation (29) yields

E[yet | xt,c] = 2α(xt,c)(1 − α(xt,c))∆µ⋆t(xt,c). (30)

Finally, at the start of training (or for sufficiently small updates) where vθ ≈ vold so that σ(zt) ≈ const, taking conditional expectation of the gradient form in Step 3 and using Equation (30) gives

E[−∇θℓt(θ) | xt,c] ∥

∂vθ ∂θ

⊤

BtΣ−t 1∆µ⋆t(xt,c),

where scalar factors such as 2α(1 − α) do not affect alignment. This proves part (c). □

#### A.6. Proof of Theorem 4.5 (Comparison with wMSE)

- A.6.1. DECOMPOSITION OF WMSE

We substitute Eθ± = ∥et ∓ dt∥2Σ−1

t

into the weighted-MSE objective LwMSE = rEθ+ + (1 − r)Eθ−. Expanding the terms immediately yields:

LwMSE = r(∥et∥2 + ∥dt∥2 − 2⟨Σ−t 1et,dt⟩) + (1 − r)(∥et∥2 + ∥dt∥2 + 2⟨Σ−t 1et,dt⟩)

= (r + 1 − r)(∥et∥2 + ∥dt∥2) + 2⟨Σ−t 1et,dt⟩(−r + (1 − r))

= const + ∥dt∥2Σ−1

t

+ 2(1 − 2r)⟨Σ−t 1et,dt⟩. (31) Using y = 2r − 1 (which implies 1 − 2r = −y), we obtain:

LwMSE = const − 2y⟨Σ−t 1et,dt⟩ + ∥dt∥2Σ−1

t

. (32)

- A.6.2. RANKING CALIBRATION Define the ranking error event at a sampled step t:

Et := {y(x0,c) · (Eθ+ − Eθ−) > 0}. (33) π-StepNFT minimizes softplus(y(E+ − E−)), which is a convex upper bound on the indicator function 1E

t

, thus directly minimizing ranking errors. In contrast, wMSE minimizes a regression loss with the penalty ∥dt∥2 (from A.7.1), which restricts the branch separation required to satisfy the ranking condition Et when the margin is small.

- A.6.3. BINARY CASE ANALYSIS When r ∈ {0,1}:

- • wMSE: If r = 1, LwMSE = E+. It pulls µ+ to xt− but provides no signal to µ−.
- • π-StepNFT: Minimizes softplus(E+ − E−). It simultaneously pulls µ+ to xt− and pushes µ− away. This “push-pull” dynamic generates stronger gradients for discrimination.

| |
|---|

### B. Detailed Related Works

#### B.1. Online RL for VLAs

VLA models map multimodal inputs to actions via diverse representations: discretizing actions into tokens (RT series (Zitkovich et al., 2023), OpenVLA (Kim et al., 2024)), mapping to continuous regression features (OpenVLA-OFT (Kim et al., 2025b)), or outputting actions via generative denoising processes (Octo (Ghosh et al., 2024), GR00T (Bjorck et al., 2025), OpenPi (Black et al., 2026; 2025; Intelligence et al., 2025)). While pre-training establishes broad capabilities, the post-training focus is shifting from SFT to online RL to bridge the domain gap. Adapting RL depends on these representations, where discrete approaches (VLA-RL (Lu et al., 2025), RL4VLA (Liu et al., 2026)) leverage accessible token probabilities, while continuous mappings (SimpleVLA-RL (Li et al., 2025a)) treat outputs as Gaussian means. However, flow-based VLAs face the challenge of intractable likelihoods due to multi-step ODE sampling. Some methods bypass likelihood calculation entirely: GR-RL (Li et al., 2025b) distills value functions in the latent space, while π0∗.6 utilizes preference-based feedback. Conversely, πRL (Chen et al., 2025) addresses this by transforming the deterministic ODE into an SDE or adding auxiliary noise networks. Crucially, this noise injection serves a dual purpose: it not only facilitates mathematical likelihood approximation but also significantly enhances exploration. This importance of noise-induced exploration is further echoed by test-time scaling strategies like TACO (Yang et al., 2025) and Hume (Song et al., 2025), as well as DSRL (Wagenmaker et al., 2025), which operates RL directly in the diffusion noise space.

#### B.2. Policy Optimization for Generative Models

Integrating online RL into generative models typically follows three paradigms to handle intractable likelihoods. Explicit Gradient and Advantage Methods. Approaches like DDPO (Black et al., 2023) and DPOK (Fan et al., 2023) treat denoising

- as a sequential decision process. Flow-GRPO (Liu et al., 2025) and ReinFlow (Zhang et al., 2025b) further facilitate this by converting ODEs to SDEs or using Gaussian approximations to enable policy gradient updates. Reward-Weighted Likelihood-Free Methods. To avoid exact likelihood computation, methods such as RWFM (Pfrommer et al., 2025; Fan et al.,

2025) and FPO (McAllister et al., 2025) construct proxy objectives or advantage-weighted ratios, effectively optimizing the flow model via regression targets derived from high-reward samples. However, these paradigms often suffer from high variance in gradient estimation or rely on complex reward proxies to stabilize training. Preference and Contrastive Methods. These approaches align distributions via ranking losses, bypassing explicit advantages. Diffusion-DPO (Wallace et al., 2024) aligns models based on trajectory outcomes, while LPO (Zhang et al., 2025a) ensures fine-grained consistency at the latent noise-step level. Uniquely, Diffusion-NFT (Zheng et al., 2026) proposes a solver-agnostic framework that constructs implicit positive and negative update directions directly within the forward process, offering a computationally efficient paradigm without requiring explicit likelihoods or value networks.

### C. Experiment Details

- C.1. Detailed Introduction of Benchmarks We evaluate on 2 multitask benchmarks.

- • LIBERO (Liu et al., 2023): We follow the standard protocol across four suites (Spatial, Object, Goal, Long), reporting average success rates over 500 episodes (50 states × 10 sub-tasks) per suite. The agent receives dual 224×224 RGB inputs, language instructions, and 7-dimensional proprioceptive states (6-DoF joints + gripper). It outputs continuous end-effector actions. The environment provides a sparse binary reward (1 for success, 0 otherwise).
- • ManiSkill (Mu et al., 2021): We adopt the PutOnPlateInScene multitask setting from RL4VLA (Liu et al., 2026). This benchmark defines 4,352 compositional tasks derived from 16 objects, 17 receptacles, and 16 tabletop scenes. Observations consist of a single 480×640 third-person view, language instructions, and joint poses. Actions are continuous joint-space commands. The environment provides a composite reward to discourage degenerate throwing behaviors.

Table 3. OOD task mapping for ManiSkill PutOnPlateInScene25* across Vision, Semantics, and Execution categories.

Category Sub-category (OOD type) ManiSkill env IDs

Unseen Table (background) PutOnPlateInScene25VisionImage-v1 Dynamic Textures (foreground, weak) PutOnPlateInScene25VisionTexture03-v1 Dynamic Textures (foreground, strong) PutOnPlateInScene25VisionTexture05-v1 Dynamic Noise (image-level, weak) PutOnPlateInScene25VisionWhole03-v1 Dynamic Noise (image-level, strong) PutOnPlateInScene25VisionWhole05-v1

Vision

Unseen Objects PutOnPlateInScene25Carrot-v1 Unseen Receptacles PutOnPlateInScene25Plate-v1 Unseen Instruction Phrasings PutOnPlateInScene25Instruct-v1 Multi-Object PutOnPlateInScene25MultiCarrot-v1 Distractive Receptacle PutOnPlateInScene25MultiPlate-v1

Semantics

Unseen Position PutOnPlateInScene25Position-v1 Unseen Robot Init Pose PutOnPlateInScene25EEPose-v1 Mid-Episode Object Reposition PutOnPlateInScene25PositionChangeTo-v1

Execution

- C.2. Hyperparameters for Training Table 4. Hyperparameter settings for Libero and ManiSkill.

LIBERO ManiSkill

Parameters

π0 π0.5 π0 π0.5

###### Spatial Object Goal Long Spatial Object Goal Long Multitask Multitask

Train epochs 400 400 400 400 400 400 400 400 240 240 Batch size 2048 2048 2048 2048 2048 2048 2048 2048 5120 5120 Update epochs 2 2 4 4 1 1 3 4 5 5 Actor lr 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 8e-6 8e-6

Interaction steps 240 240 320 480 240 240 320 480 60 60 Parallel environments 64 64 64 64 64 64 64 64 64 64 Rollout epochs 8 8 8 8 8 8 8 8 – – Action chunk H 5 5 5 10 5 5 5 10 5 5 Denoise steps 4 4 4 4 4 4 4 4 4 4 Noise level σ 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 Trust Region Size β 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 Initial Decay α0 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 End Decay α−1 0.995 0.995 0.995 0.995 0.995 0.995 0.995 0.995 0.995 0.995

### D. Additional Ablation: Step Selection Strategy

Motivation. Our step-wise supervision is defined on a single solver transition (xt → xt−) sampled from a K-step Flow-SDE rollout. In our default implementation, we uniformly sample the solver step index j ∼ U{0,...,K − 1} at each

training iteration and construct (xt,xt−,t) = (xt

,tj). This stochastic step selection exposes the model to transitions

,xt

j

j+1

- at different noise levels and denoising stages, providing more balanced supervision across the entire solver trajectory.

Ablation setup. We compare the default Random Step strategy against several Fixed Step variants, where the solver index j is held constant throughout training. All other configurations (solver, objective, training budget, and environment settings) remain identical.

Results. As shown in Figure 5, uniformly random step selection achieves more stable optimization and improves the final success rate compared to fixed-step choices. We hypothesize that fixed-step supervision biases learning toward a narrow noise regime, while random step selection provides coverage over multiple denoising stages and thus yields more robust policy learning.

1.0

0.8

###### SuccessRate

0.6

0.4

- Fixed Step=0

- Fixed Step=1

- Fixed Step=2

- Fixed Step=3

0.2

Random Step

0.0

0 25 50 75 100 125 150 175 200

Training Steps

Figure 5. Step selection ablation. Performance comparison between uniform random solver-step sampling and fixed-step selection strategies.

