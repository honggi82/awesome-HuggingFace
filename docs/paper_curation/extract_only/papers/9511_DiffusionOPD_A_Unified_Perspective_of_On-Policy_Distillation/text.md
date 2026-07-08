# arXiv:2605.15055v1[cs.LG]14May2026

## DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models

Quanhao Li1∗ Junqiu Yu1* Kaixun Jiang1 Yujie Wei1 Zhen Xing2‡ Pandeng Li2 Ruihang Chu2 Shiwei Zhang2‡ Yu Liu2 Zuxuan Wu1†

1Fudan University 2Wan Team, Alibaba Group

liqh24@m.fudan.edu.cn zxwu@fudan.edu.cn

Project page: https://quanhaol.github.io/DiffusionOPD-site/

### Abstract

Reinforcement learning has emerged as a powerful tool for improving diffusionbased text-to-image models, but existing methods are largely limited to single-task optimization. Extending RL to multiple tasks is challenging: joint optimization suffers from cross-task interference and imbalance, while cascade RL is cumbersome and prone to catastrophic forgetting. We propose DiffusionOPD, a new multi-task training paradigm for diffusion models based on Online Policy Distillation (OPD). DiffusionOPD first trains task-specific teachers independently, then distills their capabilities into a unified student along the student’s own rollout trajectories. This decouples single-task exploration from multi-task integration and avoids the optimization burden of solving all tasks jointly from scratch. Theoretically, we lift the OPD framework from discrete tokens to continuous-state Markov processes, deriving a closed-form per-step KL objective that unifies both stochastic SDE and deterministic ODE refinement via mean-matching. We formally and empirically demonstrate that this analytic gradient provides lower variance and better generality compared to conventional PPO-style policy gradients. Extensive experiments show that DiffusionOPD consistently surpasses both multi-reward RL and cascade RL baselines in training efficiency and final performance, while achieving state-of-the-art results on all evaluated benchmarks.

(a) (b)

[Figure 1]

[Figure 2]

DiffusionOPD (0.914)

[Figure 3]

Cascade NFT(0.903)

Multi-Task GRPO-Guard (0.884)

Multi-Task NFT (0.861)

[Figure 4]

[Figure 5]

Figure 1: (a) DiffusionOPD exhibits significantly faster convergence and a higher performance ceiling than all multi-task reinforcement learning baselines. (b) DiffusionOPD consistently outperforms all baselines across multiple domains, including GenEval, OCR, and aesthetics.

∗Equal contribution, †Corresponding author, ‡Project leader

Preprint.

### 1 Introduction

Reinforcement learning (RL) [15, 21, 22] has recently emerged as a powerful paradigm for improving diffusion-based text-to-image models [8, 13, 16]. A growing body of work [10, 27, 27, 28, 35, 37, 42] has shown that RL can substantially boost performance when optimizing against a single reward signal. However, these gains are typically task-specific. In practice, users often expect a single model to satisfy multiple objectives simultaneously, for example, generating images that are both aesthetically pleasing and faithful to textual instructions. This mismatch between single-objective optimization and multi-objective user demand naturally motivates the study of multi-task RL.

Multi-task RL aims to equip a single diffusion model with multiple capabilities by optimizing it over several task-specific rewards. Existing approaches mainly follow two paradigms. The first is joint optimization, which trains all tasks simultaneously within a unified framework. Although appealing in principle, this strategy often suffers from two fundamental challenges: objective conflict across tasks and task-difficulty imbalance. Different tasks may induce inconsistent optimization directions, causing cross-task interference during training, while easier tasks tend to dominate the learning dynamics and suppress signals from more challenging ones.

The second paradigm is cascade RL [13, 42], which optimizes the policy on different tasks sequentially rather than simultaneously, avoiding direct gradient conflict within each training stage. However, this strategy is often cumbersome in practice, as it requires multiple training stages, carefully designed schedules, and task-specific hyperparameter. It is also prone to catastrophic forgetting [6], where adaptation to later tasks can degrade performance on those learned earlier.

To address the reward conflict in joint optimization and the cumbersome training procedure of cascade optimization, we argue that multi-task RL should be decoupled into two distinct processes: single-task on-policy exploration and multi-task capability integration. Motivated by the success of On-Policy Distillation (OPD) [26], we propose DiffusionOPD, an on-policy distillation framework for diffusion models. Concretely, we first train a set of task-specific teacher models, each optimized independently for a single task, and then distill their capabilities into a unified student model. This avoids cross-task interference during teacher training and eliminates the student’s exploration burden to solve all tasks from scratch.

To extend OPD from LLMs to diffusion models, we first derive a diffusion-domain OPD objective. Specifically, we lift the original formulation from autoregressive token transitions to continuous-state denoising transitions, and model the diffusion denoising process as a discrete-time Markov chain induced by the reverse-time SDE [10]. Under this view, both the student and the teacher define one-step Gaussian transition kernels at each denoising state. Since these kernels share the same covariance, and their reverse KL admits a closed-form expression, yielding the OPD objective for diffusion.

Given this objective, a straightforward choice is to follow [26] and optimize the student with a PPO-style objective, using the per-step reverse KL as a dense reward and treating the teacher as a process-level reward model [2, 29, 41] along the student trajectory. However, our derivation reveals that this formulation introduces an additional score-function term proportional to Gaussian noise. Although unbiased in expectation, this term increases gradient variance, making PPO [21] an unnecessarily noisy way to optimize a quantity that is already available in closed form.

We therefore directly optimize the closed-form KL objective rather than relying on a PPO-style surrogate. This design reduces gradient variance and yields stronger empirical performance. Moreover, it naturally extends to deterministic ODE samplers, where it recovers direct transition matching, thereby offering a unified view of on-policy distillation across different diffusion samplers.

More importantly, our framework is not limited to the closed-form reverse-KL objective derived above. Once the student generates on-policy rollouts, the teacher can supervise the visited denoising states using a broad family of existing distillation objectives [12, 39, 40]. DiffusionOPD should therefore be viewed not merely as a reverse-KL method, but more generally as a unified framework for on-policy distillation in diffusion models.

We further evaluate DiffusionOPD in the multi-task setting, where it consistently surpasses all multitask RL baselines across diverse benchmarks in both training efficiency and final performance. We also conduct ablations on key design choices, including the distillation objective, loss formulation, and sampler noise level.

Our contributions can be summarized as follows:

- • We propose DiffusionOPD, a new on-policy distillation paradigm for multi-task training of diffusion models, where domain-specific teachers supervise a unified student along its own rollout trajectories.
- • We establish a principled framework for on-policy diffusion distillation by deriving a unified closed-form KL objective for both stochastic and deterministic samplers, enabling lowervariance optimization than PPO-style policy gradients.
- • We validate DiffusionOPD through multi-task experiments and ablations, showing consistent gains over prior baselines in both training efficiency and final performance, with state-ofthe-art results on aesthetics, OCR, and GenEval. Our ablations further highlight the impact of key design choices.

### 2 Related Works

- 2.1 RL for Diffusion.

Reinforcement learning (RL) has recently emerged as an effective paradigm for improving diffusionbased text-to-image models [16]. Building on advances in Reinforcement Learning [15, 21, 22], a growing line of work has adapted RL to diffusion generation and shown that it can substantially improve model behavior under task-specific reward signals, such as aesthetic quality, text rendering accuracy, and compositional alignment [1, 9, 10, 25, 27, 28, 31, 33, 35, 37, 42]. Most existing methods, however, focus on optimizing a single reward at a time, yielding task-specialized improvements rather than a unified model that performs well across multiple objectives. In practice, users often expect a single text-to-image model to satisfy several desiderata simultaneously, such as visual appeal, prompt faithfulness, and OCR correctness. This gap has motivated growing interest in extending RL for diffusion models from single-task optimization to the multi-task setting.

- 2.2 Diffusion Distillation

Diffusion distillation aims to transfer the knowledge of a teacher diffusion model to a student model. Most prior work in this area has focused on step distillation, where a many-step teacher is compressed into a few-step student for more efficient inference. Existing approaches can be broadly grouped into two categories. Trajectory distillation [11, 14, 18, 23, 24] distills the teacher’s denoising process by imitating intermediate transitions or enforcing consistency across timesteps. Distribution matching methods, on the other hand, train student models by aligning their distributions with those of the teacher at selected timesteps, including Diffusion-GAN hybrids [19, 36] and scoredistillation methods [12, 32, 38, 39, 43]. In contrast to this line of work, we do not use distillation for step reduction. Instead, we study how to distill multiple reward-specialized teachers into a single aligned student in the multi-task setting, using task-specific teachers to provide dense supervision for capability integration.

- 3 Method

- 3.1 Preliminary: OPD in the LLM Domain

Let πθ denote the student language model and let π⋆ denote a frozen teacher. For a token sequence x = (x1,...,xT), both policies factorize autoregressively:

πθ(x) =

T

πθ(xt | x<t), π⋆(x) =

t=1

T

π⋆(xt | x<t). (1)

t=1

On-policy distillation [26] lets the student autoregressively generate a full sequence from its own policy, and then trains the student to match the teacher on the prefixes that the student itself visits. A natural sequence-level objective is therefore the reverse-KL under student-generated trajectories:

LLLMOPD(θ) = KL πθ(·)∥π⋆(·) = Ex∼π

θ

πθ(x) π⋆(x)

log

. (2)

where the expectation is taken over full sequences sampled from the student model πθ. Using the autoregressive factorization, the sequence-level KL decomposes exactly into a sum of per-step conditional KLs evaluated along the student’s own trajectory:

T

LLLMOPD(θ) = Ex∼π

KL πθ(· | x<t) π⋆(· | x<t) . (3)

θ

t=1

For LLMs, this inner KL is a discrete distribution over a finite vocabulary V, so it admits a closed form as shown below.

πθ(v | x<t) π⋆(v | x<t)

KL πθ(· | x<t) π⋆(· | x<t) =

πθ(v | x<t)log

.

v∈V

In contrast to standard on-policy reinforcement learning, where the model generates a full response and receives only an outcome-level scalar reward, OPD provides token-level dense supervision. The student receives a full next-token distributional target from the teacher at every decoding step along its own trajectory. This allows the objective to be optimized as an analytic per-step KL via direct backpropagation, avoiding the high-variance policy gradients inherent in sparse reward settings.

#### 3.2 DiffusionOPD

Lifting OPD to a continuous-state Markov chain We reinterpret (3) as a statement about any discrete-time Markov chain in which the student and teacher share the same state space and transition kernel structure. Concretely, let xt

) and pT(· | xt

be a trajectory of states and let pS(· | xt

,xt

,...,xt

0

1

j

N

) denote the student and teacher one-step transition kernels. Replacing “πθ(· | x<t)” by “pS(· | xt

j

)” and analogously for π⋆, the OPD objective becomes

j

 . (4)

 

N−1

LOPD(θ) = Ex

KL pS(· | xt

) pT(· | xt

)

0:N∼pS

j

j

j=0

Two structural properties of (4) survive the lift: (i) the trajectory is sampled from the student (onpolicy), and (ii) the per-step KL must be available in closed form so we never need the REINFORCE trick.

Per-step Gaussian transitions For a flow-matching model on latents x ∈ Rd, we follow FlowGRPO [10] and discretize the reverse-time SDE by Euler–Maruyama on a schedule 1 = t0 > t1 > ··· > tN = 0 with step size ∆tj := tj+1 − tj < 0. Let σt = a t/(1 − t) denote the SDE diffusion coefficient, where a is the global noise level. Writing vjS := vθ(xt

,tj) for the student velocity, the student SDE step is

j

2 tj

+ vjS + σ

j −∆tj εj (5)

+ (1−tj)vjS ∆tj + σt

xt

= xt

2tj xt

j+1

j

j

where εj ∼ N(0,Id) injects stochasticity. Collecting the deterministic part of (5) and abbreviating the per-step variance as σ¯j2 := σt2

(−∆tj), the one-step transition kernel is the Gaussian pS xt

j

), σ¯j2 Id , (6) with student transition mean

= N µS(xt

xt

j+1

j

j

2 tj

2 tj (1−tj)

) = 1 + σ

+ 1 + σ

2tj vjS ∆tj. (7)

µS(xt

2tj ∆tj xt

j

j

We thus construct the teacher kernel pT by the same formulas (6)–(7) on the same scheduler and noise level, with the student velocity replaced by the frozen teacher velocity vjT := vϕ(xt

,tj): pT xt

j

), σ¯j2 Id , (8) µT(xt

= N µT(xt

xt

j+1

j

j

2 tj (1−tj)

2 tj

+ 1 + σ

) = 1 + σ

2tj vjT ∆tj. (9)

2tj ∆tj xt

j

j

Closed-form reverse KL between same-covariance Gaussians. Since the per-step covariance σ¯j2Id depends only on the scheduler (tj,∆tj) and the global noise level a, it is identical for the student

and teacher. Moreover, under on-policy distillation, both transition kernels are evaluated at the same student-rollout state xt

), while sharing the same covariance. For two d-dimensional Gaussians with common covariance Σ,

. Therefore, pS and pT differ only in their means, µS(xt

##### ) and µT(xt

j

j

j

KL N(µ1,Σ)∥N(µ2,Σ) = 12 (µ1 − µ2)⊤Σ−1(µ1 − µ2). Specializing to Σ = σj2Id gives

KL N(µ1,σj2I)∥N(µ2,σj2I) = ∥µ1 − µ2∥22 2σj2

. (10)

This expression is exact and introduces no Monte-Carlo variance, since the sample noise εj cancels analytically.

Plugging (6)–(9) and Eq. (10) into the generic OPD objective (4) yields

 . (11)

 

) 22 2σj2

N−1

##### ;θ) − µT(xt

##### µS(xt

LdiffusionOPD (θ) = Ex

j

j

0:N∼pS,θ

j=0

Deterministic regime: direct L2 matching. In the LLM setting, reverse KL is the natural OPD objective because the model defines a stochastic next-token distribution at each prefix, so matching the teacher necessarily amounts to matching conditional distributions. By contrast, under the deterministic ODE Euler update in diffusion models, the next state is uniquely determined by the current latent xt

, the student and teacher therefore induce two deterministic transition targets, µS(xt

. For a given xt

j

j

), respectively. In this regime, distribution matching reduces to pointwise transition matching, and the reverse-KL objective can be replaced by a direct squared L2 loss:

##### ;θ) and µT(xt

j

j

|Ldiffusion-ODEOPD (θ) = Ex<br><br>0:N∼pS,θ<br><br> <br><br>N−1<br><br>j=0<br><br>1<br><br>2 µS(xt<br><br><br>j<br><br>;θ) − µT(xt<br><br>j<br><br>) 2 2<br><br> .|
|---|

(12)

This yields a deterministic specialization of DiffusionOPD in which the student is trained to match the teacher’s one-step transitions directly along its own rollout trajectory.

- 3.3 Discussion: Closed-form KL vs. PPO-style Policy Gradient Our DiffusionOPD objective in Eq. (11) already provides a closed-form per-step supervision signal:

 

 , (13)

N−1

LdiffusionOPD (θ) = Ex

KL pS(· | xt

##### )∥pT(· | xt

)

0:N∼pS,θ

j

j

j=0

with

)∥22 2¯σj2

##### ) = ∥µS(xt

##### ;θ) − µT(xt

. (14)

j

j

##### KL pS(· | xt

##### )∥pT(· | xt

j

j

Since the student and teacher share the same covariance σ¯j2Id, the KL depends only on the mean mismatch and can be optimized by direct backpropagation.

Direct closed-form KL. Differentiating Eq. (14) gives

 

 . (15)

N−1

##### ;θ) − µT(xt

##### µS(xt

##### ) σ¯j2 · ∇θµS(xt

∇θLdiffusionOPD (θ) = Ex

j

j

;θ)

0:N∼pS,θ

j

j=0

This is a standard pathwise gradient: the loss is an explicit differentiable function of the student transition mean.

PPO-style policy gradient. Alternatively, one may regard the teacher model as a process reward model [2, 29, 41], which provides dense per-step supervision along the student trajectory. In this view, a natural choice of per-step advantage is the negative KL,

##### Aj = −KL pS(· | xt

##### )∥pT(· | xt

##### ) ,

j

j

and one can optimize a PPO-style surrogate [21]:

j∼πθold min ρj(θ)Aj, clip(ρj(θ),1 − ε,1 + ε)Aj , (16) where ρj(θ) = πθ(aj | xt

LPG(θ) = −Ea

).

(aj | xt

)/πθ

j

j

old

Ignoring clipping, the PPO surrogate reduces to

j∼πθold ρj(θ)∆j(θ) . (17)

LPG(θ) = −Ea

Since the model parameters are held fixed over an entire rollout through gradient accumulation (refer to Algorithm 1 for gradient accumulation details), the rollout policy equals the current student policy, i.e., πθ

= πθ. For a sampled transition, the gradient decomposes as ∇θ ρj(θ)∆j(θ) = ρj(θ)∇θ∆j(θ) + ρj(θ)∆j(θ)∇θ log πθ(aj | xt

old

). (18) Under πθ

j

= πθ, we have ρj(θ) = 1, so Eq. (18) becomes

old

∇θ ρj(θ)∆j(θ) = ∇θ∆j(θ) pathwise term

+∆j(θ)∇θ log πθ(aj | xt

)

j

score-function term

. (19)

where ∆j(θ) := KL pS(· | xt

) . Since ∆j(θ) does not depend on the sampled action aj, therefore

)∥pT(· | xt

j

j

Ea

) = ∆j(θ)Ea

j∼πθ ∆j(θ)∇θ log πθ(aj | xt

j∼πθ ∇θ log πθ(aj | xt

)

j

j

= ∆j(θ) · 0 = 0,

Hence the two objectives have the same expected gradient:

(20)

E[∇θLPG(θ)] = ∇θLdiffusionOPD (θ). (21)

Equation (21) shows that direct KL minimization and PPO-style optimization are equivalent in expectation.

Why the closed-form KL is a better solution. The closed-form KL is preferable to a PPO-style surrogate for two reasons.

First, it yields a lower-variance gradient estimator. The direct objective in Eq. (14) is an analytic function of the student transition mean, so its gradient is obtained entirely by pathwise backpropagation. By contrast, the PPO formulation introduces an additional score-function term of the form ∆j(θ)∇θ log πθ(aj | xt

). For a Gaussian transition with aj = µS(xt

j

;θ) + σ¯jϵj, ϵj ∼ N(0,Id). (22) we have

j

ϵj σ¯j · ∇θµS(xt

∇θ log πθ(aj | xt

;θ). (23)

) =

j

j

Thus, the PPO estimator contains an additional stochastic term proportional to Gaussian noise. Although this term is unbiased in expectation, it introduces nonzero gradient variance, which is absent in the closed-form KL objective.

Second, the closed-form KL loss formulation remains valid in both stochastic and deterministic sampling regimes. In the deterministic ODE regime, we can use Eq. (12) to update student policy. A PPO-style objective, however, is inherently tied to a stochastic policy density through log πθ and the importance ratio ρj.

Therefore, for DiffusionOPD, the closed-form KL is not only lower-variance but also applicable to a wider range of samplers, covering both SDE and ODE samplers within a single training principle.

- 3.4 Training Recipe

Our DiffusionOPD follows a two-stage training paradigm, as summarized in Algorithm 1. In the first stage, we decompose the multi-task problem into M individual tasks and train a separate task-specific

Algorithm 1: DiffusionOPD

Input: Tasks M = {1, . . . , M}; prompt datasets {C(m)}Mm=1; pretrained diffusion policy vref; denoising

schedule {tj, σ¯j2}Nj=0−1.

Output: Unified multi-task diffusion student vθ.

- Stage 1: Per-task teacher training; foreach m ∈ M do

Train a task-specific teacher vϕ(m)

m

for task m using an off-the-shelf RL algorithm; end

- Stage 2: Multi-task on-policy distillation;

Initialize vθ ← vref; for each training round do

Initialize total round loss Ltotal ← 0; for m = 1, . . . , M do

Sample prompts c ∼ C(m); Roll out the current student vθ on c to obtain on-policy trajectory {xtj}Nj=0 ; // no_grad Compute task loss Lm via Monte Carlo estimate of Eq. (11) or Eq. (12) using teacher vϕ(m)

; // Eq. (12) by default

m

Ltotal += Lm

end Update θ by performing one backward pass on Ltotal and taking an optimizer step;

end

teacher vϕ(m)

for each task m ∈ M using off-the-shelf diffusion RL algorithms [28, 42]. This stage allows each teacher to specialize in its own reward objective without being affected by inter-task interference.

m

In the second stage, we distill these specialized teachers into a single unified student vθ, initialized from the pretrained diffusion policy vref. Training proceeds in a round-robin on-policy manner over all tasks. For each task m, we first sample prompts from C(m), then roll out the current student to obtain an on-policy denoising trajectory {xt

j}Nj=0. Along this sampled trajectory, we evaluate the corresponding task teacher and compute a Monte Carlo estimate of the OPD objective in Eq. (11), which matches the student and teacher transition means at every denoising step.

To stabilize multi-task optimization, we accumulate losses over a full round-robin cycle before updating the student. Concretely, we set the gradient accumulation factor to G = M, i.e., one accumulation step per task, and average the task losses within each round. A single backward pass and optimizer step are performed only after all M tasks have been visited once. This design makes each parameter update reflect the supervision from the complete task set, reducing update variance and mitigating bias toward any individual task.

### 4 Experiments

In this section, we detail the experimental setup and demonstrate the capabilities of DiffusionOPD from three perspectives: (1) comparison with major multi-task learning baselines, (2) comparison with alternative distillation methods for transferring knowledge from multiple single-task teachers, and (3) ablation studies on key design choices.

#### 4.1 Experimental Setup

Implementation Details. We follow DiffusionNFT [42] for the experimental setup and use SD3.5Medium [3] at 512×512 resolution as the base model. Our reward models include both rule-based and model-based signals. The rule-based rewards are GenEval [4] for compositional generation and OCR for visual text rendering, while the model-based rewards include PickScore [7], ClipScore [5], HPSv2.1 [34], Aesthetics [20], ImageReward [35], and UnifiedReward [30]. For data, we use the FlowGRPO splits for GenEval and OCR, and train on Pick-a-Pic [7] while evaluating on DrawBench [17] for the model-based rewards. We also adopt the same finetuning and evaluation

Table 1: Evaluation Results. Gray-colored: in-domain reward. ‡Evaluated at 1024×1024 resolution. Bold: best; Underline: second best. ∗Approximated training time. Wall-clock time is reported in hours. The Average column denotes the mean of min-max normalized scores over all metrics. †For DiffusionOPD, wall-clock time is reported as the maximum teacher training time plus OPD training time.

Rule-Based Model-Based

Model Wall-clock time

Average GenEval OCR PickScore ClipScore HPSv2.1 Aesthetic ImgRwd UniRwd

SD-XL‡ — 0.55 0.14 22.42 0.287 0.280 5.60 0.76 2.93 0.390 SD3.5-L‡ — 0.71 0.68 22.91 0.289 0.288 5.50 0.96 3.25 0.601 FLUX.1-Dev — 0.66 0.59 22.84 0.295 0.274 5.71 0.96 3.27 0.599

SD3.5-M (w/o CFG) — 0.24 0.12 20.51 0.237 0.204 5.13 -0.58 2.02 0.000 + CFG — 0.63 0.59 22.34 0.285 0.279 5.36 0.85 3.03 0.484 GenEval Teacher 46.92 0.96 0.40 22.04 0.274 0.248 5.24 0.59 2.97 0.473 OCR Teacher 33.17 0.65 0.93 22.27 0.290 0.272 5.26 0.90 3.09 0.550 Aes Teacher 85.75 0.49 0.59 24.02 0.295 0.346 6.22 1.498 3.48 0.698

Multi-Task GRPO-Guard 129.86 0.89 0.94 23.12 0.296 0.307 5.61 1.31 3.33 0.763 Multi-Task NFT 128.42 0.95 0.96 22.59 0.288 0.282 5.41 1.08 3.23 0.715 Cascade NFT 148.49∗ 0.94 0.91 23.80 0.293 0.331 6.01 1.49 3.49 0.851 DiffusionOPD(Ours) 85.75+11.26† 0.96 0.94 23.99 0.297 0.342 6.15 1.50 3.50 0.929

[Figure 6]

[Figure 7]

A storefront with 'Google Research Pizza Cafe'

A pizza on the right of a suitcase.

written on it.

[Figure 8]

[Figure 9]

A sign that says 'Diffusion'.

A black apple and a green backpack.

Figure 2: Qualitative comparisons against multi-task RL methods and single-task teachers. Each case is presented in two rows. The first row shows, from left to right, DiffusionOPD (ours), Multi-Task GRPO-Guard, Multi-Task NFT, and Cascade NFT. The second row shows the input prompt, our Aes Teacher, our GenEval Teacher, and our OCR Teacher.

configuration as DiffusionNFT, using LoRA (α = 64, r = 32) and a 40-step first-order ODE sampler for evaluation.

Single-Task Teachers. We select the training algorithm for each teacher according to the characteristics of its reward task. For OCR and Aesthetics, we train the teachers with GRPO-Guard. In our preliminary experiments, although DiffusionNFT converges rapidly, it is highly susceptible to reward hacking on OCR, often achieving high reward scores at the cost of severe image quality degradation. For the aesthetics teacher, we optimize an equally weighted (1:1:1) mixture of PickScore, Clip-

Score, and HPSv2.1, and find that GRPO-Guard consistently attains a higher performance ceiling than DiffusionNFT on this objective. For GenEval, we instead use DiffusionNFT to train the teacher, as it exhibits faster convergence and a higher performance ceiling on this task.

Baselines. We compare DiffusionOPD against several competitive baselines: (1) Single-task teachers, i.e., the specialized models described above; (2) Multi-Task RL, which uses different RL algorithms to jointly train on multiple tasks by alternating across the corresponding datasets in the same curriculum as DiffusionOPD; and (3) Cascade NFT [42], a sequential training baseline where different tasks are learned stage by stage.

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 3: DiffusionOPD outperforms multi-task RL baselines in both efficiency and performance.

4.2 Comparisons with Multi-Task RL Methods

Table 1 shows that single-task teachers are highly specialized to their own training domains, but generalize poorly across heterogeneous rewards. The GenEval Teacher mainly excels at compositional alignment, the OCR Teacher is strongest on text rendering, and the Aes Teacher performs best on aesthetic-related objectives, while each of them shows limited transferability beyond its own optimization target. Multi-task RL methods improve overall task coverage, but require substantially longer training time and still struggle on more challenging objectives such as aesthetics, indicating slower convergence and stronger optimization interference across domains. Although Cascade NFT achieves relatively competitive performance, it is the slowest and most cumbersome strategy due to sequential multi-stage training, and is also prone to catastrophic forgetting, which limits its final performance.

By contrast, DiffusionOPD achieves the best overall performance, demonstrating the effectiveness of our training paradigm for multi-domain preference optimization. Qualitative comparison in Figure 2 and 7 also demonstrates the superior visual quality of our method.

To further evaluate training efficiency, we plot the convergence curves in Figure 3. As shown, multi-task RL baselines converge more slowly than single-task RL teachers, indicating that jointly optimizing heterogeneous rewards introduces severe optimization interference and hinders learning efficiency. Besides, DiffusionOPD requires much less total training time than the Multi-Task RL baselines to reach the same target score, while also attaining a substantially higher performance ceiling.

4.3 Ablation Studies

[Figure 13]

[Figure 14]

A zebra to the right of a fire hydrant.

[Figure 15]

[Figure 16]

New York Skyline with 'Google Research Pizza Cafe' written with fireworks on the sky.

- Figure 4: Qualitative comparisons with different distillation methods. From left to right: DiffusionOPD (ours), DMD, TDM, and SFT.

Distillation Methods. We further compare DiffusionOPD with several representative distillation baselines that transfer knowledge from single-task teachers, including DMD [39], TDM [12], and supervised fine-tuning (SFT). For SFT, we use the corresponding teacher to generate images online and train the student to imitate these teacher-generated samples, which can also be viewed as a form of teacher knowledge distillation. For DMD and TDM, we perform on-policy sampling using the student model and distill the corresponding teacher through the training gradients defined by each method. To ensure a fair comparison, we implement all baselines under the same setting as DiffusionOPD: each method is distilled from the identical set of specialized teachers, and training is conducted by alternating across datasets. As shown in Figure 5, DiffusionOPD consistently achieves the fastest convergence and highest final performance ceiling among all compared distillation methods. Qualitative results in Fig. 4 and Fig. 8 also demonstrate our superiority.

[Figure 17]

[Figure 18]

[Figure 19]

- Figure 5: Ablation studies on distillation methods. SFT is trained on images generated from teacher rollouts, while all other baselines use student on-policy rollouts and distill from the same set of teacher models using their respective objectives.

Loss formulation. To validate our analysis in Section 3.3, we compare the closed-form KL objective against the PPO-style policy gradient. To ensure a fair comparison, both methods are evaluated in the multi-task setting with an identical sampling noise level of a = 0.7. As shown in Figure 6, under the same noise level, the closed-form KL objective achieves faster reward improvement and a higher performance ceiling than PPO-style policy gradients.

Noise Level. We further conduct an ablation study on the noise level of the SDE sampler used during distillation. As shown in Figure 6, reducing the noise level consistently leads to faster convergence and higher evaluation scores for the student model. In particular, the ODE sampler with is up to five times more efficient than the SDE sampler with noise level=0.7.

[Figure 20]

[Figure 21]

[Figure 22]

- Figure 6: Ablation studies on the loss formulation and sampler noise level. When the noise level is set to 0, the SDE sampler reduces to an ODE sampler, and the student is optimized using Eq. (12). As shown, the PPO-style policy gradient underperforms its closed-form KL counterpart. Moreover, lower noise levels lead to faster convergence and higher performance ceiling.

### 5 Conclusion

We introduced DiffusionOPD, a new on-policy distillation paradigm for multi-task training of diffusion models. By decoupling single-task exploration from multi-task capability integration, DiffusionOPD avoids the optimization conflict of joint multi-task RL and the inefficiency and forgetting of cascade RL. We further developed a principled theoretical framework that extends OPD to diffusion Markov chains, yielding a closed-form per-step reverse-KL objective that unifies stochastic SDE and deterministic ODE refinement. Compared with PPO-style policy-gradient optimization, this objective enables lower-variance training and applies naturally across sampler types. Extensive experiments and ablations show that DiffusionOPD consistently improves both training efficiency and final performance over prior baselines, achieving state-of-the-art results on aesthetics, OCR, and GenEval. We hope DiffusionOPD can serve as a useful foundation for future work on multi-task and preference-aligned diffusion modeling.

### References

- [1] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pages 4965–4987, 2024.
- [2] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- [3] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [4] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [5] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
- [6] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.
- [7] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.
- [8] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.
- [9] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Miles Yang, and Zhao Zhong. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde, 2025. URL https: //arxiv.org/abs/2507.21802.
- [10] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026.
- [11] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [12] Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17719–17728, 2025.
- [13] Chaojie Mao, Chen-Wei Xie, Chongyang Zhong, Haoyou Deng, Jiaxing Zhao, Jie Xiao, Jinbo Xing, Jingfeng Zhang, Jingren Zhou, Jingyi Zhang, et al. Wan-image: Pushing the boundaries of generative visual intelligence. arXiv preprint arXiv:2604.19858, 2026.
- [14] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14297–14306, June 2023.

- [15] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [16] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [17] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022.
- [18] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.
- [19] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation, 2023.
- [20] Christoph Schuhmann. Laion-aesthetics. https://laion.ai/blog/laion-aesthetics/, 2022.
- [21] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [22] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [23] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=WNzy9bRDvG.
- [24] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [25] Bingda Tang, Yuhui Zhang, Xiaohan Wang, Jiayuan Mao, Ludwig Schmidt, and Serena YeungLevy. V-grpo: Online reinforcement learning for denoising generative models is easier than you think. arXiv preprint arXiv:2604.23380, 2026.
- [26] Thinking Machines Lab. On-policy distillation. https://thinkingmachines.ai/blog/ on-policy-distillation/, 2026.
- [27] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.
- [28] Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, et al. Grpo-guard: Mitigating implicit over-optimization in flow matching via regulated clipping. arXiv preprint arXiv:2510.22319, 2025.
- [29] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024.
- [30] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.
- [31] Yiyang Wang, Xi Chen, Xiaogang Xu, Yu Liu, and Hengshuang Zhao. Gdro: Group-level reward post-training suitable for diffusion models. arXiv preprint arXiv:2601.02036, 2026.
- [32] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023.

- [33] Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al. Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025.
- [34] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023.
- [35] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.
- [36] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. ArXiv, abs/2311.09257, 2023. URL https: //api.semanticscholar.org/CorpusID:265221033.
- [37] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [38] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. arXiv preprint arXiv:2311.18828, 2023.
- [39] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.
- [40] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024.
- [41] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 10495–10516, 2025.
- [42] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.
- [43] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In International Conference on Machine Learning, 2024.

[Figure 23]

[Figure 24]

New York Skyline with 'Hello World' written with fireworks on the sky.

A giraffe underneath a microwave.

[Figure 25]

[Figure 26]

An old photograph of a 1920s airship shaped like a pig, floating over a wheat field.

A zebra underneath a broccoli.

[Figure 27]

[Figure 28]

A stack of 3 cubes. A red cube is on the top, sitting on a red cube. The red cube is in the middle, sitting on a

New York Skyline with 'Google Brain Toronto' written with fireworks on the sky.

green cube. The green cube is on the bottom.

[Figure 29]

[Figure 30]

A stop sign on the right of a refrigerator.

A car on the left of a bus.

[Figure 31]

[Figure 32]

Five dogs on the street.

Pafrking metr.

- Figure 7: Qualitative comparisons against multi-task RL methods and single-task teachers. Each case is presented in two rows. The first row shows, from left to right, DiffusionOPD (ours), Multi-Task GRPO-Guard, Multi-Task NFT, and Cascade NFT. The second row shows the input prompt, our Aes Teacher, our GenEval Teacher, and our OCR Teacher.

[Figure 33]

New York Skyline with 'Google Research Pizza Cafe' written with fireworks on the sky.

[Figure 34]

A carrot on the

left of a broccoli.

[Figure 35]

A sign that says 'Hello World'.

[Figure 36]

A car on the left of a bus.

[Figure 37]

A cat on the right of a tennis racket.

[Figure 38]

A red colored car.

[Figure 39]

New York Skyline with 'Diffusion'

written with fireworks on the sky.

[Figure 40]

Artophagous.

- Figure 8: Qualitative comparisons with different distillation methods. From left to right: DiffusionOPD (ours), DMD, TDM, and SFT.

