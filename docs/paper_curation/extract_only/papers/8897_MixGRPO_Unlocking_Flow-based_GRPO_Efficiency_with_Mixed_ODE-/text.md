# arXiv:2507.21802v7[cs.AI]28Jun2026

## MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE

Junzhe Li1,2,3⋆, Yutao Cui1*, Tao Huang1*, Weijie Kong1, Chuxuan Zeng4, Yiming Cheng5, Yinping Ma3, Chun Fan3, Miles Yang1, Zhao Zhong1 , and Liefeng Bo1

1 Hunyuan, Tencent, China 2 School of Computer Science, Peking University, China

- 3 Computer Center, Peking University, China
- 4 China Mobile Communications Group, China

5 Department of Engineering Physics, Tsinghua University, China

Abstract. Although GRPO substantially improves flow-matching models for human preference alignment in vision generation, mainstream methods such as DanceGRPO rely on Global-Stochastic Differential Equations (SDE) sampling across full timesteps in the Markov Decision Process (MDP), which remains computationally inefficient. In this paper, we propose MixGRPO, a novel framework that leverages the flexibility of mixed sampling strategies through the integration of SDE and Ordinary Differential Equations (ODE). This redesign streamlines optimization within the MDP, delivering gains in both generation performance and training efficiency. Specifically, MixGRPO introduces a sliding window mechanism, using SDE sampling and GRPO-guided optimization only within the window, while applying ODE sampling outside. This design confines sampling randomness to the time-steps within the window, thereby reducing the optimization overhead, and allowing for more focused gradient updates to accelerate convergence. Additionally, as time-steps beyond the sliding window are not involved in optimization, higher-order solvers are supported for faster sampling. So we present a faster variant, termed MixGRPO-Flash, which further improves training efficiency while achieving comparable performance. MixGRPO exhibits substantial gains across multiple dimensions of human preference alignment, outperforming DanceGRPO in both effectiveness and efficiency, with nearly 50% lower training time. Notably, MixGRPOFlash further reduces training time by 71%.6

Keywords: Vision Generation · Flow-matching · Online RL · GRPO

### 1 Introduction

Recent advances [20,21,48,49] in vision generation have demonstrated that probability flow models can achieve improved performance by incorporating Rein-

⋆ Equal contribution. (lijunzhe1028@gmail.com) Corresponding authors. 6 Code is available in https://github.com/Tencent-Hunyuan/MixGRPO

[Figure 1]

###### Prompt: Three cows eating in a field with sea in background.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

DanceGRPO DanceGRPO DanceGRPO DanceGRPO MixGRPO (Ours)

- Fig. 1: Comparison of MixGRPO and DanceGRPO with varying denoising timesteps optimized. MixGRPO achieves higher performance with lower overhead.

forcement Learning from Human Feedback (RLHF) [30] strategies during the post-training stage to maximize rewards. Specifically, methods [20,49] based on Group Relative Policy Optimization (GRPO) [36], have recently been studied, achieving optimal alignment with human preferences.

Current GRPO methods in probability flow models, such as Flow-GRPO and DanceGRPO, combine a rollout phase followed by an online optimization phase. During rollouts, they employ Stochastic Differential Equation (SDE) sampling at each denoising step to introduce randomness, thereby enabling the stochastic exploration required by RLHF. This allows the entire denoising process to be framed as a Markov Decision Process (MDP) in a stochastic environment, where GRPO is then applied to optimize the complete state-action sequence in the online optimization phase. However, the requirement to optimize the entire denoising steps introduces two challenges, prohibitive computational overhead and unfocused and inefficient optimization. i) The overhead arises because computing the policy ratio requires separate full-trajectory sampling from both the old (πθ

) and new policies (πθ). Although DanceGRPO proposes a workaround by optimizing a random subset of steps, our findings in Figure 1 reveal a critical trade-off: reducing the subset size to save computation severely compromises model performance. ii) Early-step gradients prioritize global structure while latestep gradients focus on fine details, leading to conflicting update signals. This may result in an inefficient training process.

old

To address these issues, we propose MixGRPO, a method that optimizes a strategic subset of denoising steps to drastically reduce computational overhead while ensuring a more focused and efficient optimization. The policy optimization is confined only to the SDE sub-interval. This interval operates as a sliding window, systematically advancing from low-SNR (signal-to-noise ratio) steps to high-SNR denoising steps as training progresses. This curriculum-based approach is conceptually analogous to temporal discounting in RL [2, 12, 31], as it prioritizes the optimization of initial, high-impact steps that establish global structure from a vast exploration space (Figure 2 Right), before progressively shifting focus to the refinement of local details. This selective optimization not only reduces the computational burden but also fosters a more dedicated exploration. Furthermore, this decoupling allows us to accelerate the deterministic ODE portions with fast solvers (e.g., DPMSolver++ [25]), since their posterior distributions are irrelevant to the optimization, thereby reducing rollout time without compromising final image quality. Finally, we adopt Coefficients-

Preserving Sampling (CPS) [43] as a stable stochastic sampler in our framework, which reduces sampling artifacts and yields more reliable reward feedback.

To evaluate the performance and efficiency of MixGRPO, we first conduct comprehensive experiments on FLUX [16] by directly comparing against DanceGRPO [49], a representative Global-SDE baseline. Results show that our method consistently surpasses DanceGRPO in both effectiveness and efficiency. In particular, MixGRPO improves ImageReward [48] from 1.088 to 1.629, exceeding DanceGRPO’s 1.436, while generating images with better semantic fidelity, aesthetics, and fewer distortions. Moreover, compared with DanceGRPO under its official setting, MixGRPO reduces training time by nearly 50%, and MixGRPOFlash further achieves a 71% training-time reduction. These results highlight the advantage of our Mixed ODE-SDE with sliding-window formulation. We further validate MixGRPO on both in-domain and out-of-domain reward models, and extend evaluation to cross-dataset settings, where MixGRPO demonstrates strong generalization. Beyond this, we verify robustness across different backbone models and scenarios, including SD3.5-M [5] and HunyuanImage-3.0 [4], and further extend the method to video generation on HunyuanVideo-1.5 [41], where consistent gains are observed. Finally, to demonstrate the competitiveness of MixGRPO, we compare with other RL-based alignment methods, including Flow-DPO [21], Flow-GRPO [20], and DiffusionNFT [55]. Across these comparisons, MixGRPO achieves advanced alignment performance while delivering substantial training acceleration.

To summarize, the key contributions are outlined below:

- – We propose a mixed ODE-SDE framework for GRPO training with flowbased models. This approach alleviates computational overhead by confining the stochastic optimization to a specific sub-interval of the MDP.
- – We introduce an SDE sliding window to dynamically schedule the optimized timesteps. By guiding the optimization from broad exploration to fine-grained refinement, this strategy significantly enhances performance.
- – Our hybrid framework enables the use of high-order ODE solvers to accel-

erate the deterministic sampling of πθ

old

during GRPO training. This yields substantial speedups with negligible performance degradation.

- – Comprehensive experiments across multiple frameworks demonstrate the versatility of MixGRPO. Our method consistently achieves substantial performance gains while significantly reducing training overhead, highlighting its broad applicability and effectiveness.

### 2 Related Work

RLHF for vision generation has advanced from differentiable reward-based fine tuning [48] to PPO-style [27] and DPO-style [21], and more recently to GRPObased optimization in flow-based models, where SDE-induced stochasticity enables broader exploration. Methods such as Flow-GRPO and DanceGRPO improve alignment quality, but still suffer from a key bottleneck: optimizing many

[Figure 7]

T 0 T SDE 0 T SDE 0

SDE

ODE ODE ODE ODE ODE ODE

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

tr(Σ) = 5.05 tr(Σ) = 1.51 tr(Σ) = 1.06 larger exploration space

- Fig. 2: (Left) Methods Comparison. By employing the mixed ODE-SDE and sliding window strategy, MixGRPO reduces the number of optimized timesteps and achieves more efficient training than DanceGRPO [49]. (Right) SDE Exploration Analysis. As training progresses, MixGRPO gradually slides the SDE window from a high-noise to a low-noise regime, causing the exploration space to shrink and the distribution of sampled images to transition from dispersed to concentrated.

imbalanced timesteps, which leads to redundant effective MDP horizons under Global-SDE sampling. In parallel, sampling research has progressed from ODE solvers to high-order variants. Motivated by the SDE-ODE equivalence in probability flow, we establish a mixed SDE-ODE MDP formulation and systematically explore different scheduling strategies for optimizing selected SDE timesteps, establishing a more efficient paradigm, MixGRPO. A more comprehensive review of RL for Vision Generation and Sampling Methods is provided in Appendix 7.

- 3 Method

#### 3.1 Mixed ODE-SDE Sampling in GRPO

According to Flow-GRPO [20], the SDE sampling in flow matching can be framed as a Markov Decision Process (MDP) (S,A,ρ0,P,R). The agent produces a trajectory during the discrete sampling process, defined as Γ = {(st,at)}Tt=0, where the reward is provided only at the final step by the reward model, specifically R(si,ai) ≜ R(xT,c) if i = T, and 0 otherwise.

In MixGRPO, we propose a sampling method that combines SDE and ODE. MixGRPO defines a time interval S = [tl,tr) ⊆ [0,1), which corresponds to a subset of denoising timesteps, such that 0 ≤ l < r ≤ T and ti = Ti . We use SDE sampling within the interval S and ODE sampling outside, while S shifts along the denoising direction throughout the training process (See Figure 2 Left). MixGRPO restricts the stochastic exploration to the interval S, shortening the sequence length of the MDP to a subset ΓMixGRPO = {(st,at)}rt=l and requires policy optimization only within S:

EΓMixGRPO∼πθ

max

θ

t∈S

R(st, at) − βDKL π(· | st)∥πref(· | st) , (1)

MixGRPO reduces training overhead while also enabling more concentrated gradient updates. Next, we derive the specific sampling form. For a deterministic reverse probability flow ODE [39], it takes the following form:

dxt dt

- 1

- 2

g2(t)∇xt log qt(xt), (2)

= f(xt, t) −

where qt(xt) represents the evolution process of the reverse probability distribution from 0 to T. ∇xt

log qt(xt) is the score function at time t. According to the Fokker-Planck equation [29,32], [39] has demonstrated that Eq. (2) has the following equivalent probability flow SDE, which maintains the same marginal distribution at each time t:

dxt dt

dw dt

= f(xt, t) − g2(t)∇xt log qt(xt) + g(t)

. (3)

In MixGRPO, we mix ODE and SDE for sampling. Under standard regularity assumptions (continuous g(t), locally Lipschitz drift/score fields, finite second moments, and sufficiently small ∆t), the mixed process preserves the same time-marginal dynamics as the corresponding probability-flow ODE up to discretization/score-approximation error; a full proof is provided in Appendix 8. The specific form is as follows:

f(xt, t)− g2(t)st(xt) dt + g(t)dw, if t ∈ S, f(xt, t)− 21g2(t)st(xt) dt, otherwise,

(4)

dxt=

where we define the score function as st ≜ ∇xt

log qt(xt). In particular, for Flow Matching (FM) [19], especially the Rectified Flow (RF) [22], the sampling process can be viewed as a deterministic ODE:

dxt dt

= vt. (5)

Eq. (5) is actually a special case of the Eq. (2) with vt = f(xt,t)− 12g2(t)st(xt). So we can derive mixed ODE-SDE sampling form for RF as follows:

dxt=

vt− 12g2(t)st(xt) dt + g(t)dw, if t ∈ S, vtdt, otherwise.

(6)

In the RF framework, the model is used to predict the velocity field as vθ(xt,t) = dxt

t −1−t tvθ(xt,t). The g(t) is represented as the standard deviation of the noise g(t) = σt. According to the definition of the standard Wiener process, we use dw =

dt . Following [20], the score function is represented as st(xt) = −x

t

√

dtϵ, where ϵ ∼ N(0,I). Applying Euler-Maruyama discretization for SDE and Euler discretization for ODE, we build the final denoising process in MixGRPO:

√

∆tϵ, if t ∈ S,

xt + µθ(xt, t)∆t + σt

(7)

xt+∆t =

xt + vθ(xt, t)∆t, otherwise,

where the SDE drift term for the sampling process is defined as µθ(xt,t) ≜ vθ(xt,t) + σ

2 t (xt+(1−t)vθ(xt,t))

2t . For timesteps t ∈ S, Eq. (7) induces (via Euler– Maruyama) the conditional policy density

qθ(xt+∆t | xt, c) = N xt + µθ(xt, t)∆t, σt2∆t I , t ∈ S, (8)

. For t ∈/ S, the update is deterministic (Dirac transition), and these steps are excluded from GRPO ratio/KL evaluation. According to Eq. (1), GRPO optimization is only performed on timesteps within the interval S for each group of N samples. This design not only shortens the effective MDP horizon but also concentrates gradient updates, leading to the following training objective:

with the same form for qθ

old

JMixGRPO(θ) = Ec∼C, {xi

T }Ni=1∼πθold(·|c)

N

1 N

1 |S| t∈S

i=1

clip(rti(θ), 1 − ε, 1 + ε)Ai − βJKL ,

min rti(θ)Ai,

(9)

where ε is the clipping threshold, rti(θ) is the policy ratio, and Ai is the advantage score. For optimized timesteps (t ∈ S), we parameterize the policy by the mean of the Gaussian transition above, while the covariance is fixed to σt2∆tI for both θ and θold. Therefore, the ratio and KL are defined as follows according to [20]:

R xiT, c − mean {R xiT, c }Ni=1 std ({R (xiT, c)}Ni=1)

qθ(xt+∆t|xt, c) qθold(xt+∆t|xt, c)

rti(θ) =

, t ∈ S, Ai =

,

JKL=DKL(qθ(·|xt, c) || qθold(·|xt, c))= ||xt+∆t(θ)−xt+∆t(θold)||2 2σt2∆t

, t ∈ S.

(10)

MixGRPO reduces the NFE of πθ and optimized timesteps compared to GlobalSDE sampling. However, the NFE of πθ

is not reduced, as complete inference is required to obtain the final image for reward calculation. In Section 3.3, we will introduce the use of higher-order ODE solvers, which also reduce the NFE of πθ

old

leading to further speedup. In summary, the mixed ODE-SDE sampling significantly streamlines the MDP, enhancing training efficiency by lowering optimization overhead and enabling more focused gradient updates.

old

#### 3.2 Sliding Window as Optimized Timestep Scheduler

In this section, we will introduce the sliding window to describe the movement of S, which leads to a significant improvement in the quality of the generated images. Along the denosing time-steps {0,1,...,T − 1}, MixGRPO defines a SDE sliding window W(l) and optimization is only employed at the timesteps within W(l).

W(l) = {tl,tl+1,...,tl+w−1}, l ≤ T − w, (11)

where l is the left boundary of the sliding window. The movement strategy during training is governed by three hyperparameters: (1) window size w, which determines the number of timesteps included in each window; (2) shift interval τ, which specifies the number of training steps between two consecutive window shifts; and (3) window stride s, which indicates how many denoising timesteps the left boundary l advances at each shift. In this work, we use a fixed progressive-constant scheduler as a simple and stable default across experiments.

Algorithm 1 MixGRPO Training Process

Require: initial policy model πθ; reward models {Rk}Kk=1; prompt dataset C; total

sampling steps T; number of samples per prompt N;

Require: sliding window W(l), window size w, shift interval τ, window stride s

- 1: Init left boundary of W(l): l ← 0
- 2: for training iteration m = 1 to M do
- 3: Sample batch prompts Cb ∼ C
- 4: Update old policy model: πθold ← πθ
- 5: for each prompt c ∈ Cb do
- 6: Init the same noise x0 ∼ N(0, I) ▷ according to DanceGRPO [49]
- 7: for generate i-th image from i = 1 to N do
- 8: for sampling timestep t = 0 to T − 1 do
- 9: if t ∈ W(l) then
- 10: Use SDE Sampling to get xit+1
- 11: else
- 12: Use ODE Sampling to get xit+1
- 13: end if
- 14: end for
- 15: end for
- 16: for i-th image from i = 1 to N do
- 17: Calculate advantages: Ai ← Kk=1

R(xiT ,c)ik−µk σk

- 18: end for
- 19: for optimization timestep t ∈ W(l) do
- 20: Update policy model: θ ← θ + η∇θJ
- 21: end for
- 22: end for
- 23: if m mod τ is 0 then ▷ move sliding window
- 24: l ← min(l + s, T − w)
- 25: end if
- 26: end for

The detailed sliding-window strategy and the MixGRPO algorithm are provided in Algorithm 1.

In MixGRPO, denoising follows a natural stochastic-to-deterministic trajectory along probability flow. We exploit this property with a sliding-window curriculum: optimization starts in low-SNR (high-stochasticity) regions and progressively shifts to high-SNR (more deterministic) regions for refinement, as illustrated in Figure 2 (right). This coarse-to-fine schedule stabilizes early optimization and improves final alignment quality. Table 4 shows that progressive movement consistently outperforms random selection, while even a frozen window on early timesteps remains competitive, indicating that early denoising steps contribute disproportionately to optimization payoff [5,14].

#### 3.3 Trade-off Between Overhead and Performance

Unlike DanceGRPO [49], which relies on Global-SDE sampling, MixGRPO employs a mixed ODE-SDE method, allowing the use of higher-order ODE solvers

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### FLUXMixGRPODanceGRPO

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

A photo of a vase right of a horse.

A photo of four sinks.

A lot of building on each side of the road, with a very curvy road in the middle.

A raccoon riding an oversized fox through a forest in a furry art anime still.

A still of Doraemon from "Shaun the Sheep" by Aardman Animation.

A close-up of a young man with a rugged beard and intense eyes, wearing a leather jacket and standing in front of a motorcycle.

- Fig. 3: Qualitative comparison. MixGRPO achieve superior performance in semantics, aesthetics and text-image alignment.

to accelerate GRPO training-time sampling. However, accelerating the ODE phase before the SDE window amplifies the solver’s numerical errors due to the window’s stochasticity, severely degrading image quality and corrupting the reward signal (See Appendix 13 for details). In contrast, accelerating the ODE phase after the window provides a significant speed-up without compromising the image fidelity required for reliable reward evaluation. Therefore, MixGRPO exclusively accelerates the post-window ODE timesteps.

[9] has demonstrated the equivalence between the ODE sampling of flow matching models (FM) and DDIM, and Section 3.1 has also shown that diffusion probabilistic models (DPM) and FM share the same ODE form during the denoising process. Therefore, the higher-order ODE solvers e.g., DPM-Solver Series [24,25,56], UniPC [54] designed for DPM sampling acceleration are also applicable to FM. We have reformulated DPM-Solver++ [25] to apply it in the FM framework for ODE sampling acceleration and released detailed derivations in Appendix 9.

By applying higher-order solvers, we achieve acceleration in the sampling of πθ

during GRPO training, which is essentially a balance between overhead and performance. Excessive acceleration leads to fewer timesteps, which inevitably results in a decline in image generation quality, thereby accumulating errors in the computation of rewards. We have found in practice that the 2nd-order DPMSolver++ is sufficient to provide significant acceleration while ensuring that the generated images align well with human preferences in Table 8.

old

Ultimately, we integrate DPM-Solver++ with both progressive and frozen sliding-window strategies, leading to MixGRPO-Flash and MixGRPO-Flash*. Notably, MixGRPO-Flash* enables acceleration over a larger number of ODE timesteps. A detailed description of the algorithm is provided in Appendix 10. These faster versions offer more substantial acceleration relative to base version of our MixGRPO, while simultaneously surpassing DanceGRPO in their alignment with human preferences.

- Table 1: Comparison between Mixed ODE-SDE and Global-SDE. MixGRPO achieves the best performance across multiple metrics. MixGRPO-Flash significantly reduces training time while outperforming DanceGRPO. Bold: rank 1. Underline: rank 2. ∗The Frozen strategy means that optimization is only employed at the initial denoising steps.

Human Preference Alignment HPS-v2.1↑ Pick Score↑ ImageReward↑ Unified Reward↑ FLUX / / / 0.313 0.227 1.088 3.370

Model NFEπθ

↓ NFEπθ↓ Iteration Time (s)↓

old

25 14 291.284 0.356 0.233 1.436 3.397 25 4 149.978 0.334 0.225 1.335 3.374 25 4∗ 150.059 0.333 0.229 1.235 3.325

DanceGRPO

MixGRPO 25 4 149.326 0.369 0.238 1.645 3.419 MixGRPO-Flash 16 (Avg) 4 112.372 0.358 0.236 1.528 3.407 MixGRPO-Flash* 8 4∗ 83.278 0.357 0.232 1.624 3.402

#### 3.4 Coefficients-Preserving Sampling for Stable Exploration

In addition to the mixed ODE-SDE design, we adopt Coefficients-Preserving Sampling (CPS) [43] to improve sampling stability in the stochastic segment. Compared with standard SDE discretization, CPS better preserves the interpolation structure of flow matching while injecting controlled stochasticity, reducing high-frequency artifacts that can lead to reward hacking. In our implementation, CPS is used as an alternative stochastic discretization while keeping the same per-step Gaussian covariance scale σt2∆tI for GRPO density-ratio/KL evaluation in Eq. (9); therefore, the policy-ratio and KL parameterization remain unchanged. This yields cleaner rollouts and more reliable reward signals for GRPO optimization. Detailed derivations are provided in Appendix 19.

### 4 Experiments

#### 4.1 Experiment Setup

Dataset We conduct T2I experiments using the prompts provided by the HPDv2 dataset, which is the official dataset for the HPS-v2 benchmark [47]. The training set contains 103,700 prompts; in fact, MixGRPO already achieves strong human preference alignment before completing one full epoch, using only 9,600 prompts. The test set consists of 400 prompts. The prompts are diverse, encompassing four styles: “Animation”, “Concept Art”, “Painting”, and “Photo”. Base Model Following DanceGRPO [49], we use FLUX.1-dev [16] as the primary backbone, a strong flow-matching-based text-to-image model. To further verify the robustness of MixGRPO, we also extend our method to additional backbones, including LoRA fine-tuning on SD3.5-M [5]. More implementation details are in Appendix 11.

Overhead Evaluation For the evaluation of overhead, we use two metrics: the number of function evaluations (NFE) [24] and the time consumption per iteration during training. The NFE is decomposed into NFEπ

. NFEπ

and NFEπ

θold

θ

denotes the number of forward passes of the reference model used for rollout sampling. NFEπ

θold

is the number of forward passes of the policy model solely for the policy ratio. Additionally, the average training time per GRPO iteration provides a more accurate reflection of the acceleration effect.

θ

A photo of a horse and a train

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Freshly baked donuts priced to sell at 60cents each

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

A tall giraffe in a zoo eating branches

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

DanceGRPO MixGRPO MixGRPO-Flash MixGRPO-Flash*

DanceGRPO*

- Fig. 4: Qualitative results of MixGRPO-Flash. As the training overhead is reduced, generated images maintain high quality.

Reward Model We employ four reward models: HPS-v2.1 [47], Pick Score [15],

ImageReward [48], and Unified Reward [44]. Although all four are preferencealigned metrics, they capture complementary aspects of quality. In particular, ImageReward emphasizes image-text alignment and fidelity, while Unified Reward focuses more on semantic consistency. Following the findings of DanceGRPO [49], we adopt multi-reward guidance to obtain stronger and more stable alignment performance. To validate the generalization capability of MixGRPO, we conduct cross-domain evaluations across both reward models and datasets.

#### 4.2 Main Results

Mixed ODE-SDE GRPO vs. Global-SDE GRPO We first evaluated the overhead and performance of MixGRPO against Global-SDE GRPO methods (e.g., DanceGRPO), with results shown in Table 1. In the official DanceGRPO setting, 14 timesteps are randomly selected from 25 for optimization. Our results show that MixGRPO achieves better performance while optimizing only 4 timesteps, reducing per-iteration GRPO training time from 291.284s to 150.839s. Qualitative results in Figure 3 further show that MixGRPO improves both aesthetic quality and semantic alignment. For fairness, we also modify DanceGRPO to use 4 timesteps (either randomly selected or the initial ones), matching the overhead of MixGRPO; however, DanceGRPO remains inferior to MixGRPO in alignment performance under both settings. For MixGRPO-Flash, we evaluate both progressive and frozen strategies. Although MixGRPO-Flash is slightly weaker than MixGRPO, all evaluation metrics are still higher than

those of DanceGRPO (official setting), and the fastest setting reduces time from 291.284s to 83.278s. As shown in Figure 4, MixGRPO-Flash maintains strong image quality even as overhead changes. More visualized results are provided in Appendix 20.

Cross-domain Reward Models To evaluate the generalization ability of MixGRPO, we follow DanceGRPO and use HPS-v2.1 (aesthetics) and CLIP Score [10] (semantic-consistency) as in-domain reward models, under both singlereward and two-reward training settings. We further assess out-of-domain performance using Pick Score, ImageReward, and Unified Reward. As shown in Table 2, MixGRPO consistently outperforms both DanceGRPO and the FLUX baseline on in-domain and out-of-domain metrics. These results indicate that MixGRPO improves overall image-generation alignment rather than relying on reward hacking (i.e., overfitting to in-domain reward models while failing to match human preference).

MixGRPO vs. Other Alignment Methods In addition to DanceGRPO, we compare MixGRPO with other representative alignment methods on SD3.5M, including DiffusionNFT [55], Offline/Online Flow-DPO, and Flow-GRPO, with results shown in Table 3. For fairness, the settings of the other alignment methods follow the best configurations reported in their original papers. We use HPS-v2.1, Pick Score, and ImageReward as multi-guidance and evaluation metrics. The results show that MixGRPO can converge to the best performance during training while optimizing only 4 timesteps. More experimental details are provided in Appendix 11. We also compare with ReNO [6], an alignment method for single-step generators; as shown in Appendix 12, MixGRPO achieves better performance without incurring the overhead of step distillation.

Extension to Larger-Scale Model and Video Generation We further validate the scalability of MixGRPO beyond standard T2I settings. On the industrial-scale HunyuanImage-3.0 [4] backbone (80B parameters) trained on 512 GPUs, MixGRPO maintains stable optimization and substantial efficiency gains, demonstrating that its advantages persist at very large scale. Appendix 14 provides additional experimental details and visualizations of human blind-test preferences. We also extend MixGRPO to HunyuanVideo-1.5 [41] for text-tovideo alignment, where it consistently outperforms Flow-GRPO in training stability and reward improvement. These results confirm that our mixed ODE-SDE design generalizes well across both larger image models and video generation tasks (see Appendix 15 for training stability curves and more details).

#### 4.3 Ablation Experiments

Sliding Window Hyperparamters As introduced in Section 3.2, the moving strategy, shift interval τ, window size w and window stride s are parameters

- Table 2: Comparison under cross-rewards. The results demonstrate that MixGRPO achieves the best performance on both in-domain and out-of-domain rewards.

Reward Model Method

In Domain Out-of-Domain

HPS-v2.1 CLIP Score Pick Score ImageReward Unified Reward / FLUX 0.313 0.388 0.227 1.088 3.370 HPS-v2.1

DanceGRPO 0.367 0.349 0.227 1.141 3.270

MixGRPO 0.373 0.372 0.228 1.396 3.370 HPS-v2.1 & CLIP Score

DanceGRPO 0.346 0.400 0.228 1.314 3.377 MixGRPO 0.349 0.415 0.229 1.416 3.430

- Table 3: Comparison with other alignment methods. The results demonstrate that MixGRPO outperforms others in training efficiency and performance.

###### NFEπθ HPS-v2.1 Pick Score ImageReward

Model NFEπθ

old

SD3.5-M / / 0.307 0.227 1.163 Offline Flow-DPO 40 40 0.304 0.222 1.452 Online Flow-DPO 40 40 0.313 0.221 1.500 DiffusionNFT 10 10 0.313 0.235 1.494 Flow-GRPO 10 10 0.331 0.232 1.457 DanceGRPO 10 10 0.309 0.226 1.433 MixGRPO 10 4 0.342 0.236 1.485

introduced by the sliding window. We conducted ablation experiments on each of them. For moving strategy, we compared three approaches: frozen, where the window remains stationary; random, where a random window position is selected at each iteration; and progressive, where the sliding window moves incrementally with denoising steps. For progressive strategy, we further evaluated different scheduling schemes in which the interval τ starts from 25 and evolves over training. As shown in Table 4, a constant schedule under the progressive strategy yields the strongest overall trade-off. For shift interval τ, we use τ = 25

- as a robust default setting (see Table 5). The number of forward passes for πθ increases with the growth of the window size w, leading to greater time overhead. We compared different settings of w, and results are shown in Table 6. Ultimately, we use w = 4 as a balanced default setting between overhead and performance. For window stride s, we use s = 1 as the default choice, as shown in Table 7.

High Order ODE Solver MixGRPO-Flash, which incorporates a highorder ODE solver to accelerate the sampling process of the reference model, achieves an effective trade-off between speed and performance. For MixGRPOFlash, we first conducted ablation experiments on the order of the solver, using DPM-Solver++ [25] as the high-order solver with the progressive strategy. The results, as shown in Table 8, indicate that the second-order mid-point method is the optimal setting, optimizing the most human preference alignment metrics while simultaneously accelerating the process.

Then we compared two acceleration approaches. One is MixGRPO-Flash, and the other is MixGRPO-Flash*. Both utilize a second-order ODE solver for acceleration, but they differ in their sliding-window moving strategies. The

###### Table 4: Ablation for moving strategies.

###### Table 5: Ablation for shift interval τ.

Interval Schedule

Strategy

HPS-v2.1 Pick Score ImageReward Unified Reward

Frozen / 0.354 0.234 1.580 3.403 Random Constant 0.365 0.237 1.513 3.388

Decay (Linear) 0.365 0.235 1.566 3.382 Decay (Exp) 0.360 0.239 1.632 3.416 Constant 0.367 0.237 1.629 3.418

Progressive

###### Table 6: Ablation for window size w.

###### w NFEπθ HPS-v2.1 Pick Score ImageReward Unified Reward

- 1 1 0.359 0.234 1.629 3.235
- 2 2 0.362 0.235 1.588 3.419

- 4 4 0.367 0.237 1.629 3.418 6 6 0.370 0.238 1.547 3.398

###### τ HPS-v2.1 Pick Score ImageReward Unified Reward

15 0.366 0.237 1.509 3.403 20 0.366 0.238 1.610 3.411 25 0.367 0.237 1.629 3.418 30 0.350 0.229 1.589 3.385

###### Table 7: Ablation for window stride s.

s HPS-v2.1 Pick Score ImageReward Unified Reward

- 1 0.367 0.237 1.629 3.418
- 2 0.357 0.236 1.575 3.391
- 3 0.370 0.236 1.578 3.404
- 4 0.368 0.238 1.575 3.407

quantitative results are presented in Table 9. MixGRPO-Flash requires the window to move throughout the training process, resulting in a smaller portion of the ODE being accelerated compared to MixGRPO-Flash*. Consequently, MixGRPO-Flash* not only achieves a higher degree of acceleration for the reference model but also yields superior results in the ImageReward [48] and Unified Reward [44] metrics.

Cross-Dataset Experiments and Sensitivity Analysis Following the cross-dataset protocol in Appendix 16, we train on HPD-v2 and Pick-a-Pic-v1 datasets reciprocally and evaluate on both ID and OOD splits. The same hyperparameter setting (progressive-constant, τ = 25, w = 4, s = 1) remains consistently strong across datasets, while performance stays stable within a reasonable range of (τ,w,s) (Figure 5). These results indicate that MixGRPO is robust to dataset shifts and insensitive to delicate hyperparameter tuning, achieving reliable gains without additional tuning cost.

[Figure 56]

[Figure 57]

[Figure 58]

- Fig. 5: Sensitivity analysis of hyperparameters (w, τ, s). The results show that MixGRPO is not sensitive to hyperparameters and consistently outperforms Global-SDE.

Coefficients-Preserving Sampling We further ablate the stochastic sampler by comparing standard SDE sampling and CPS [43] under the same training

Table 8: Comparison for different ODE solvers. The second-order Midpoint method achieves the best performance.

Table 9: Comparison for sampling steps of the reference model. MixGRPO-Flash preserves performance while providing acceleration.

Order Solver Type HPS-v2.1 Pick Score ImageReward Unified Reward

- 1 / 0.367 0.236 1.570 3.403

- 2

Midpoint 0.358 0.237 1.578 3.407 Heun 0.362 0.233 1.488 3.399

- 3 / 0.359 0.234 1.512 3.387

Sampling Overhead Human Preference Alignment NFEπθ

Method

###### Time per Image (s) HPS-v2.1 Pick Score ImageReward Unified Reward DanceGRPO 25 9.301 0.334 0.225 1.335 3.374

old

19 (Avg) 7.343 0.357 0.236 1.564 3.394 16 (Avg) 6.426 0.362 0.237 1.578 3.407 13 (Avg) 5.453 0.344 0.229 1.447 3.363

MixGRPO-Flash

12 4.859 0.353 0.230 1.588 3.396 10 4.214 0.359 0.234 1.548 3.430 8 3.789 0.357 0.232 1.624 3.402

MixGRPO-Flash*

- Table 10: Comparison between SDE and CPS sampling, CPS effectively improves alignment performance.

Model HPS-v2.1 Pick Score ImageReward Unified Reward

FLUX 0.313 0.227 1.088 3.370 MixGRPO-SDE 0.367 0.237 1.629 3.418 MixGRPO-CPS 0.369 0.238 1.645 3.419

setup. As shown in Table 10, CPS consistently improves all reward metrics, indicating that reducing sampling artifacts provides more reliable optimization signals for preference alignment.

### 5 Limitation

A key limitation of GRPO is that its performance ceiling is inherently tied to the capability of the reward model. Specifically, when the reward model is not sufficiently powerful to provide a comprehensive assessment of image quality, GRPO is prone to reward hacking, particularly in the later stages of training (see Appendix 18 for concrete examples). As confirmed by several studies [28, 45, 46], reward hacking is fundamentally driven by limitations of the reward model, rather than the RL algorithm itself. In this context, the primary goal of MixGRPO is not to eliminate these reward-model limitations, but to achieve faster convergence and stronger performance under the guidance of existing, imperfect reward models. In addition, our current sliding-window scheduler still relies on predefined hyperparameters (w,τ,s). Although we observe strong robustness across datasets and backbones, an adaptive scheduler based on online signals (e.g., reward convergence or gradient variance) remains an important direction for future work. Looking ahead, while ultimate performance remains contingent on reward-model quality, we plan to develop and train stronger reward models in future work.

### 6 Conclusion

In this work, we have presented MixGRPO, a novel hybrid ODE-SDE framework for improving GRPO training efficiency and performance. We have proposed a strategy to confine optimization to a dynamic stochastic interval managed by a sliding window. This guides the optimization process from broad exploration to

fine-grained refinement, thus enhancing performance. Experiments demonstrate that MixGRPO has achieved superior performance in both single-reward and multi-reward settings while substantially reducing overhead. Furthermore, we have presented MixGRPO-Flash, a variant offering a flexible trade-off between performance and computational cost.

### References

- 1. Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E.: Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 (2023)
- 2. Amit, R., Meir, R., Ciosek, K.: Discount factor as a regularizer in reinforcement learning. In: International conference on machine learning. pp. 269–278. PMLR

(2020)

- 3. Black, K., Janner, M., Du, Y., Kostrikov, I., Levine, S.: Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301 (2023)
- 4. Cao, S., Chen, H., Chen, P., Cheng, Y., Cui, Y., Deng, X., Dong, Y., Gong, K., Gu, T., Gu, X., et al.: Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951 (2025)
- 5. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 6. Eyring, L., Karthik, S., Roth, K., Dosovitskiy, A., Akata, Z.: Reno: Enhancing one-step text-to-image models through reward-based noise optimization. Neural Information Processing Systems (NeurIPS) (2024)
- 7. Fan, Y., Lee, K.: Optimizing ddpm sampling with shortcut fine-tuning. arXiv preprint arXiv:2301.13362 (2023)
- 8. Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., Lee, K.: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems 36, 79858–79885 (2023)
- 9. Gao, R., Hoogeboom, E., Heek, J., Bortoli, V.D., Murphy, K.P., Salimans, T.: Diffusion meets flow matching: Two sides of the same coin (2024), https:// diffusionflow.github.io/, accessed: 2026-03-04
- 10. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 7514–7528 (2021)
- 11. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 12. Hu, H., Yang, Y., Zhao, Q., Zhang, C.: On the role of discount factor in offline reinforcement learning. In: International conference on machine learning. pp. 9072–

9098. PMLR (2022)

- 13. Jordan, K., Jin, Y., Boza, V., Jiacheng, Y., Cesista, F., Newhouse, L., Bernstein, J.: Muon: An optimizer for hidden layers in neural networks (2024), https:// kellerjordan.github.io/posts/muon/, accessed: 2026-03-04
- 14. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusionbased generative models. Advances in neural information processing systems 35, 26565–26577 (2022)

- 15. Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., Levy, O.: Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 36652–36663 (2023)
- 16. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024), accessed: 2026-03-04
- 17. Lee, K., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Gu, S.S.: Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192 (2023)
- 18. Liang, Z., Yuan, Y., Gu, S., Chen, B., Hang, T., Cheng, M., Li, J., Zheng, L.: Aesthetic post-training diffusion models from generic preferences with step-bystep preference optimization. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 13199–13208 (2025)
- 19. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)
- 20. Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., Ouyang, W.: Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470 (2025)
- 21. Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., Zheng, M., Wu, X., Wang, Q., Qin, W., Xia, M., et al.: Improving video generation with human feedback. arXiv preprint arXiv:2501.13918 (2025)
- 22. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)
- 23. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 24. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems 35, 5775–5787 (2022)
- 25. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095

(2022)

- 26. Ma, Y., Wu, X., Sun, K., Li, H.: Hpsv3: Towards wide-spectrum human preference score (2025), https://arxiv.org/abs/2508.03789, accessed: 2026-03-04
- 27. Miao, Z., Wang, J., Wang, Z., Yang, Z., Wang, L., Qiu, Q., Liu, Z.: Training diffusion models towards diverse image generation with reinforcement learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10844–10853 (2024)
- 28. Nishimura-Gasparian, K., Dunn, I., Sleight, H., Turpin, M., Hubinger, E., Denison, C., Perez, E.: Reward hacking behavior can generalize across tasks—ai alignment forum. In: AI Alignment Forum (2024)
- 29. Øksendal, B.: Stochastic differential equations. In: Stochastic differential equations: an introduction with applications, pp. 38–50. Springer (2003)
- 30. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)
- 31. Pitis, S.: Rethinking the discount factor in reinforcement learning: A decision theoretic approach. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 7949–7956 (2019)
- 32. Risken, H., Risken, H.: Fokker-planck equation. Springer (1996)

- 33. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 34. Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512 (2022)
- 35. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)
- 36. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)
- 37. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)
- 38. Song, J., Zhang, Q., Yin, H., Mardani, M., Liu, M.Y., Kautz, J., Chen, Y., Vahdat, A.: Loss-guided diffusion models for plug-and-play controllable generation. In: International Conference on Machine Learning. pp. 32483–32498. PMLR (2023)
- 39. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020)
- 40. Tang, Z., Peng, J., Tang, J., Hong, M., Wang, F., Chang, T.H.: Inferencetime alignment of diffusion models with direct noise optimization. arXiv preprint arXiv:2405.18881 (2024)
- 41. Team, T.H.F.M.: Hunyuanvideo 1.5 technical report (2025), https://arxiv.org/ abs/2511.18870, accessed: 2026-03-04
- 42. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Diffusion model alignment using direct preference optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8228–8238 (2024)
- 43. Wang, F., Yu, Z.: Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952 (2025)
- 44. Wang, Y., Zang, Y., Li, H., Jin, C., Wang, J.: Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236 (2025)
- 45. Weng, L.: Reward hacking in reinforcement learning. lilianweng.github.io (Nov 2024), https://lilianweng.github.io/posts/2024-11-28-reward-hacking/, accessed: 2026-03-04
- 46. Wu, J., Gao, Y., Ye, Z., Li, M., Li, L., Guo, H., Liu, J., Xue, Z., Hou, X., Liu, W., et al.: Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826 (2025)
- 47. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023)
- 48. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)
- 49. Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al.: Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818 (2025)
- 50. Yeh, P.H., Lee, K.H., Chen, J.C.: Training-free diffusion model alignment with sampling demons. arXiv preprint arXiv:2410.05760 (2024)
- 51. yifan123, G.U.: Discussion on flow-grpo issue 7. https://github.com/yifan123/ flow_grpo/issues/7#issuecomment-2870678379 (2025), accessed: 2025-05-12

- 52. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6613– 6623 (2024)
- 53. Yuan, H., Chen, Z., Ji, K., Gu, Q.: Self-play fine-tuning of diffusion models for text-to-image generation. Advances in Neural Information Processing Systems 37, 73366–73398 (2024)
- 54. Zhao, W., Bai, L., Rao, Y., Zhou, J., Lu, J.: Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems 36, 49842–49869 (2023)
- 55. Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., Liu, M.Y.: Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117 (2025)
- 56. Zheng, K., Lu, C., Chen, J., Zhu, J.: Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

## Appendix of MixGRPO

### 7 Related Work

#### 7.1 RL For Vision Generation

Inspired by Proximal Policy Optimization (PPO) [35], early works [3, 7, 8, 17] incorporated reinforcement learning (RL) into diffusion models by optimizing the score function [39] via policy gradient methods, enabling image generation that better aligns with human preferences. Subsequently, [42] introduced offline Direct Preference Optimization (DPO) into text-to-image (T2I) generation, allowing diffusion models to learn directly from human preference pairs without explicit reward modeling, and demonstrating strong scalability on large models. In parallel, training-free methods [6, 38, 40, 50] improve diffusion quality and efficiency by modifying inference-time sampling rather than updating model parameters. However, because offline-DPO may induce distributional shift by gradually moving the model away from its original data distribution through win–lose pair training, subsequent works [18, 53] adopt online alignment with step-aware preference signals to better preserve distributional consistency and improve optimization stability. More recently, GRPO-based approaches, e.g., Flow-GRPO [20], and DanceGRPO [49], have further advanced RL-enhanced image generation. These methods extend Group Relative Policy Optimization (GRPO) to flow matching models by introducing stochasticity into the probability flow dynamics, enabling trajectory-level policy updates. This line of research has also inspired several variants, such as DiffusionNFT [55], which reformulates diffusion reinforcement from a forward-process perspective. While effective, they rely on discretized sampling trajectories and full-step optimization, which introduce substantial computational overhead. Although DanceGRPO attempts to alleviate this issue by reducing or randomly subsampling optimization steps, the fundamental challenge of trajectory-level credit assignment remains. In this work, we revisit the probabilistic structure of diffusion probability flow and investigate the core mechanism of GRPO under mixed sampling and adaptive optimization schemes.

#### 7.2 Sampling Methods for Probability Flow

The development of generative sampling began with DDPM [11], which used a slow, SDE-based probability flow requiring thousands of steps. To accelerate this, DDIM [37] introduced a deterministic ODE-based approach, reducing sampling steps to around 100. This SDE / ODE duality was later unified by score-based models [39], paving the way for higher-order ODE solvers like DPM-Solver [24,25] that reduced steps to 10. Higher-performance solvers [54, 56] continue to be proposed; however, the gains are relatively marginal and have ultimately been replaced by the distillation method [34,52]. Concurrently, flow matching models [5,19] simplified training by directly predicting the vector field, also enabling

fast ODE sampling. Crucially, recent theoretical work [1,9] has proven that flow matching and diffusion models share equivalent SDE and ODE formulations. This unification provides the theoretical foundation for our work, in which we explore an interleaved SDE and ODE sampling strategy within these probability flow models, thereby confining stochasticity to the SDE intervals in RL optimization, which shortens the effective MDP horizon and reduces training overhead.

### 8 Proof of Convergence for Mixed ODE-SDE Sampling

To prove that the mixed ODE-SDE sampling method in Eq. (7) has the same convergence as Eq. (2), which uses only ODE sampling, referencing [39], we approach this from the perspective of distribution evolution, where the distribution

t(x)

- at each time step, e.g., ∂q

∂t must be the same. Let the interval for SDE be defined as S = [tl,tr) ∈ [0,1). Along the denoising direction, when the same initial Gaussian noise distribution q0(x0) is given, the probability distribution evolution in the ODE interval preceding the SDE is completely identical. The key point is whether the distribution evolution of the SDE within the interval

- S is completely equivalent to that of the ODE. If they are equivalent, then the ODE interval following the SDE will naturally be equivalent to using only ODE sampling. Next, we will provide a detailed proof for this key point.

Consider the SDE Eq. (3) in the interval S, which possesses the following form:

dx = [f(x, t) − g2(t)∇x log qt(x)]dt + g(t)dw, t ∈ S. (12)

The marginal probability density qt(xt) evolves according to Kolmogorov’s equation (Fokker-Planck equation) [29]

∂qt(x) ∂t

= − ∇x · f(x, t) − g2(t)∇x log qt(x) qt(x)

(13)

- 1

- 2

g2(t)∇2xqt(x)

+

According to the definition of the Laplace operator∇2h ≡ ∇ · ∇(h) and ∇x log qt(x) = ∇q

t(x)

qt(x) , we can obtain:

∂qt(x) ∂t

= −∇x · f(x, t)qt(x) − g2(t)∇xqt(x)

- 1

- 2

g2(t)∇2xqt(x)

+

(14)

- 1

- 2

g2(t)∇xqt(x)]

= −∇x · [f(x, t)qt(x) −

- 1

- 2

g2(t)∇x log qt(x)

= −∇x· f(x, t)−

qt(x) .

fODE(x,t)

The Eq. (14) is indeed the Fokker-Planck equation of the ODE Eq. (2). Therefore, within the interval S, the distribution evolution of SDE and ODE sampling is consistent.

### 9 DPM-Solver++ for Recitified Flow

For clarity and to avoid ambiguity between continuous time and discrete steps, we adopt the following notation in this section. We denote the discrete time steps by an index i ∈ {0,1,...,T − 1}, where T is the total number of sampling steps. The continuous time corresponding to step i is denoted by ti = Ti ∈ [0,1).

The DPM-Solver++ algorithm [25] is originally designed for the x0-prediction diffusion model [33], where the model outputs the denoised feature x0 based on the noisy feature xt

, the time condition ti and the text condition c. According to the definition of Rectified Flow (RF) [22], there is the following transfer equation:

i

= tix1 + (1 − ti)x0. (15)

#### xt

i

According to the theory of stochastic interpolation [1], RF effectively approximates x1 − x0 by modeling vt

:

i

= x1 − x0. (16) Based on Eq. (15) and Eq. (16), we obtain the following relationship:

#### vt

i

ti. (17)

i − vt

x0 = xt

i

By using a neural network for approximation, we establish the relationship between RF and the x0-prediction model:

xθ(xi,ti,c) = xi − vθ(xi,ti,c) · ti. (18)

Taking the multistep second-order DPMSolver++ as an example (see Algorithm 2 in [25]), we derive the corrected xθ for the RF sampling process as Di:

hi 2hi−1

Di ← 1 +

(xi−1 − vθ(xi−1,ti−1,c) · ti−1)

(19)

hi 2hi−1

−

(xi−2 − vθ(xi−2,ti−2,c) · ti−2),

where hi = λt

. The continuous time ti corresponds to the discrete step i over a total of T sampling steps. The term λt

i − λt

i−1

is the log-signal-to-noise-ratio (log-SNR) and is defined in RF as:

i

λt

i

:= log

1 − ti ti

. (20)

Based on the exact discretization formula for the probability flow ODE proposed in DPM-Solver++ (Eq. (9) in [25]), we can derive the final transfer equation:

ti ti−1

xi−1 − (1 − ti) e−hi − 1 Di, 1 ≤ i < T. (21)

xi ←

Algorithm 2 MixGRPO-Flash Training Process

Require: initial policy model πθ; reward models {Rk}Kk=1; prompt dataset C; total sampling steps T˜; number of samples per prompt N; ODE compression rate r˜ Require: sliding window W(l), window size w, shift interval τ, window stride s

- 1: Init left boundary of W(l): l ← 0
- 2: for training iteration m = 1 to M do
- 3: Sample batch prompts Cb ∼ C
- 4: Update old policy model: πθold ← πθ
- 5: for each prompt c ∈ Cb do
- 6: Init the same noise x0 ∼ N(0, I)
- 7: for generate i-th image from i = 1 to N do
- 8: for sampling timestep t = 0 to T˜ − 1 do
- 9: if t < l then
- 10: Use first-order ODE sampling to get xit+1
- 11: else if l ≤ t < l + w then
- 12: Use SDE sampling to get xit+1
- 13: else ▷ higher-order ODE
- 14: Use DPM-Solver++ sampling to get xit+1
- 15: end if
- 16: end for
- 17: end for
- 18: for i-th image from i = 1 to N do
- 19: Calculate advantage: Ai ← Kk=1

R(xiT˜,c)ik−µk σk

- 20: end for
- 21: for optimization timestep t ∈ W(l) do
- 22: Update policy model: θ ← θ + η∇θJ
- 23: end for
- 24: end for
- 25: if use MixGRPO-Flash* then ▷ move sliding window
- 26: l ← 0
- 27: else
- 28: if m mod τ is 0 then
- 29: l ← min(l + s, T − w)
- 30: end if
- 31: end if
- 32: end for

### 10 MixGRPO-Flash Algorithm

MixGRPO-Flash Algorithm 2 accelerates the ODE sampling that does not contribute to the calculation of the policy ratio after the sliding window by using DPM-Solver++ in the Eq. (21). We introduce a compression rate r˜ such that the ODE sampling after the window only requires (T − l − w)˜r time steps. And the total time-steps is T˜ = l +w +(T −l −w)˜r The final algorithm is as follows:

Note that when using MixGRPO-Flash*, the frozen strategy is applied, with the left boundary of the sliding window l ≡ 0. The theoretical speedup of the

training-time sampling can be described as follows:

T w + (T − w)˜r

. (22)

S =

For MixGRPO-Flash, since the sliding window moves according to the progressive strategy during training, the average speedup can be expressed in the following form:

T El (w + l + ⌈(T − w − l)˜r⌉)

<

S =

T w + (T − w)˜r

. (23)

### 11 Implementation Details

Based on FLUX.1-dev [16] Training is conducted on 32 NVIDIA GPUs with a per-GPU batch size of 1 for up to 300 iterations. For training-time sampling, we first apply the time shift t˜= 1−(˜st−1)t to uniformly sampled timesteps ti = Ti

(i ∈ [0,...,T − 1]), and define σt = η 1−t˜t˜ with s˜ = 3 and η = 0.7. We set

- T = 25 sampling steps. In GRPO training, the model generates 12 images per prompt, clips advantages to [−5,5], and uses 3-step gradient accumulation (i.e.,

- 4 gradient updates per training iteration). For multi-reward training, all rewards are equally weighted. We use AdamW [23] with a learning rate of 1 × 10−5 and weight decay of 1×10−4, train in bf16 mixed precision, and keep master weights in fp32. Based on SD3.5-M [5] Training is performed on 24 NVIDIA H20 GPUs using Adam with lr = 3 × 10−4 and wd = 1 × 10−4. We sample N = 24 images per prompt at 512 × 512 resolution, use T = 10 steps for training and T = 40 for evaluation, and set the classifier-free guidance scale to 4.5. The global batch size is 96 prompts per update (12 prompts per GPU with 6 gradient accumulation steps). We use a timestep fraction of 0.99, KL coefficient β = 0.001, and EMA. For MixGRPO, we set window size w = 4, shift interval τ = 150, and stride s = 1. Rewards are an equally weighted combination of HPSv2, PickScore, and ImageReward to jointly optimize aesthetics and text-image alignment.

### 12 Multi-step MDP GRPO vs. Alignment with Single-step Generators

To evaluate the efficiency of MixGRPO under multi-step MDP rollout against models that leverage single-step generators, we include a comparison with Rewardbased Noise Optimization (ReNO) [6]. ReNO improves one-step text-to-image models by iteratively optimizing the initial latent noise during inference with reward gradients from frozen preference models.

Nevertheless, the application of ReNO is largely restricted to single-step models (e.g., SD3.5-Large-Turbo [5]) due to the cost of backpropagating through the full sampling path. In contrast, our MixGRPO framework is compatible

- Table 11: Comparison with ReNO [6]. MixGRPO enables efficient optimization of the base model, whereas ReNO requires step distillation.

Base Model Method Training Overhead (sec/step)↓ HPS-v2.1↑ Pick Score↑ ImageReward↑ SD3.5-L

ReNO ≫ 94.297† 0.352 0.235 1.725 MixGRPO 94.297 0.366 0.237 1.659

Bold: best results per base model; †ReNO requires one-time distillation overhead to a single-step generator.

- Table 12: Comparison of MixGRPO-Flash variants with and without pre-window ODE acceleration. Accelerating the dual ODE segment causes GRPO optimization collapse.

Base Model Method HPS-v2.1↑ Pick Score↑ ImageReward↑ FLUX

MixGRPO-Flash (Dual) 0.335 0.223 1.235 MixGRPO-Flash (Post) 0.358 0.236 1.528

with standard multi-step diffusion and flow-matching pipelines. As shown in Table 11, MixGRPO provides a better efficiency–alignment trade-off while avoiding the additional step-distillation requirement.

- 13 Degradation caused by accelerating the ODE before the SDE window

In the main paper, we accelerate ODE segments outside the optimization window with MixGRPO-Flash to reduce training-time sampling cost. A key implementation choice is where to apply this acceleration relative to the SDE optimization window: Dual (accelerating both the pre-window and post-window ODE segments) versus Post (accelerating only the post-window ODE segment). We observe that accelerating the ODE segment before the SDE window causes visible degradation. As shown in Fig. 6, MixGRPO-Flash (Post) shows better highfrequency detail preservation and medium colour saturation, while MixGRPOFlash (Dual) exhibits high-frequency information loss and high colour saturation. Moreover, this degradation pattern of MixGRPO-Flash (Dual) becomes increasingly severe as training proceeds.

For a controlled comparison, all settings are identical between the two variants except the acceleration strategy (i.e., whether pre-window ODE steps are accelerated): the same base model initialization, prompts, reward models, optimizer and hyperparameters, rollout length, sliding-window configuration, and training iterations. The visual examples in Fig. 6 are GRPO rollout sampling results. Quantitative results in Table 12 further support the visual findings, where MixGRPO-Flash (Post) consistently outperforms MixGRPO-Flash (Dual) on HPS-v2.1, Pick Score, and ImageReward. This indicates that keeping the prewindow ODE segment uncompressed is important for stable GRPO optimization, because it determines the state distribution entering the RL-optimized SDE window.

[Figure 59]

[Figure 60]

MixGRPO-Flash (Post)

High-frequency detail preservation 😊 Medium colour saturation 😊

[Figure 61]

[Figure 62]

MixGRPO-Flash (Dual)

High-frequency information loss 😥 High colour saturation 😥

- Fig. 6: GRPO rollout visualizations of MixGRPO-Flash (Dual) and MixGRPO-Flash (Post).

### 14 Scaling MixGRPO to Industrial-Scale Models

To further validate the scalability and practical impact of MixGRPO, we extended our experiments to an industrial-scale setting, moving beyond standard academic benchmarks. This large-scale evaluation aims to demonstrate that MixGRPO is not merely a refinement of existing techniques but a fundamental paradigm shift in efficient reinforcement learning for flow-based generative models.

Experimental Setup. We applied MixGRPO to HunyuanImage-3.0 [4], an advanced flow-based large-scale text-to-image model with approximately 80B parameters. The training was conducted on a high-performance computing cluster using 512 NVIDIA GPUs. This unprecedented scale allows us to examine the behavior of our Mixed ODE-SDE paradigm and sliding window strategy when dealing with extremely high-dimensional parameter spaces and massive data throughput.

Efficiency and Performance Gains. As illustrated in Fig. 7, MixGRPO achieves remarkable performance improvements while significantly reducing the computational burden. Specifically, our method provides a consistent speedup (approximately 70%) compared to full-trajectory optimization methods like FlowGRPO or DanceGRPO. More importantly, the convergence of MixGRPO at this scale is notably more stable. By operationalizing RL Temporal Discounting within the flow-matching framework, our sliding window strategy effectively

###### Comparative Experiments on HunyuanImage-3.0 (80B)

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

❌

❌

Flow-GRPOMixGRPO

[Figure 67]

[Figure 68]

✅

✅

Semantic Consistency （Prompt: Three Monkeys...)

Realism, Aesthetics

- Fig. 7: Performance and efficiency gains of MixGRPO when scaled to the 80B HunyuanImage-3.0 model on 512 GPUs. The results demonstrate that our method maintains its significant advantages in both convergence speed and reward optimization even at an industrial scale.

resolves the gradient conflicts that often plague full-trajectory RL in extremely large models.

Methodological Impact. The success of these industrial-scale experiments underscores several key advantages of MixGRPO:

- – Optimal Budget Allocation: MixGRPO offers a transformative perspective on how to allocate the sampling and optimization budget for continuoustime generation. Instead of treating all timesteps equally, it focuses the RL signal where it is most effective.
- – Viability of GRPO: These results transform GRPO from a theoretically interesting but computationally prohibitive technique into a viable, costeffective paradigm for the next generation of 100B+ parameter models.
- – Robustness to Scale: The performance gains do not diminish as the model size or GPU count increases, proving that the theoretical foundation of our Mixed ODE-SDE paradigm (detailed in Sec. 3.1) is rigorously sound and scalable.

In summary, the application of MixGRPO to the 80B HunyuanImage-3.0 model confirms that our approach represents a fundamental breakthrough in the efficiency and effectiveness of Reinforcement Learning from Human Feedback (RLHF) for state-of-the-art generative models.

### 15 Extension to Text-to-Video Generation

To further validate the scalability and robustness of MixGRPO, we conducted comparative experiments against Flow-GRPO using the HunyuanVideo-1.5 [41] backbone.

Implementation Details. For the experimental setup, training is performed on 64 Nvidia H800 GPUs. Consistent with the pre-training strategy, we utilize the Muon [13] optimizer (lr = 1e-5, wd = 0.01). During the rollout phase, we adopt prompts from DanceGRPO [49], generating N = 8 video samples per prompt

HPSv3

###### VideoAlign_MQ

###### VideoAlign_VQ

###### VideoAlign_TA

0.00

0.2

0.0

6

0.25

0.4

0.2

0.50

Reward

0.6

4

0.75

0.4

0.8

1.00

0.6

2

1.0

1.25

0.8

Flow-GRPO

Flow-GRPO

Flow-GRPO

Flow-GRPO

1.2

1.50

0

MixGRPO

MixGRPO

MixGRPO

MixGRPO

0 25 50 75 100 125 150

0 25 50 75 100 125 150

0 25 50 75 100 125 150

0 25 50 75 100 125 150

Training Steps

Training Steps

Training Steps

Training Steps

- Fig. 8: Training Reward Dynamics on HunyuanVideo-1.5. Comparison of reward curves between MixGRPO (ours) and Flow-GRPO during text-to-video alignment training.

at a resolution of 480 × 864 with 121 frames (approx. 5 seconds). The sampling process is configured with T = 25 steps, a time shift of 5, and a stochastic scale η = 0.5. The global batch size is set to 8 prompts (resulting in a total of 64 video samples) for each update. Regarding MixGRPO-specific configurations, we set the sliding window size w = 6, shift interval τ = 100, and stride s = 1. The optimization is guided by an equal-weighted combination of HPSv3 [26] and VideoAlign [21], comprehensively assessing aesthetics, semantics, and motion quality.

Analysis of Training Dynamics. The training reward curves are shown in Figure 8. We observe that standard Flow-GRPO is unstable and inefficient in the high-dimensional latent space of video generation. Specifically, on VideoAlign sub-metrics (Visual Quality, VQ; Text Alignment, TA) and HPSv3, Flow-GRPO yields only marginal gains and even degrades at some stages. In contrast, MixGRPO demonstrates superior stability and convergence efficiency. By confining the stochastic exploration to a sliding window, MixGRPO effectively reduces the optimization search space, allowing for more precise gradient updates. As shown in the figures, MixGRPO achieves consistent monotonic improvements across all three metrics—Aesthetics (HPSv3), Motion Quality (MQ) and Visual Quality (VQ)—significantly outperforming the baseline. This confirms that our mixed ODE-SDE strategy is not only effective for images but crucial for stabilizing RL training in complex video generation tasks.

### 16 Cross-dataset Experiments and Hyperparameter Sensitivity Analysis

To investigate the robustness and parameter sensitivity of the sliding window strategy in MixGRPO, we conducted a series of cross-dataset ablation studies. We established two reciprocal settings to evaluate both in-domain (ID) and outof-domain (OOD) performance. In cross-dataset experiment 1, the model was trained on the HPDv2 [47] dataset and evaluated on the test sets of HPDv2 (ID) and Pick-a-Pic v1 (OOD). In cross-dataset experiment 1, conversely, the model was trained on Pick-a-Pic v1 [15] and evaluated on the test sets of Pick-a-Pic v1 (ID) and HPDv2 (OOD). Within these settings, we ablated the key parameters

of MixGRPO: the moving strategy, shift interval τ, window size w and window stride s. The results are presented in the Tables 13, 14, 15, 16.

As shown in Table 13, Progressive-Decay(Exp) and Progressive-Constant emerge as the top-performing moving strategies. While Progressive-Decay(Exp) consistently achieves the highest ImageReward [48] score in in-domain evaluations, Progressive-Constant demonstrates superior overall performance. Notably, the Progressive-Constant strategy consistently outperforms all others in out-ofdomain tests, demonstrating its strong robustness and generalization capabilities.

The results concerning the shift interval (τ), which dictates the frequency of window movement in training iteration steps, are summarized in Table 14. Overall, the optimal performance is consistently observed at τ = 25. Specifically, as τ is incrementally increased from 15 to 25, performance metrics exhibit a gradual rise across both in-domain and out-of-domain test sets. However, a crucial observation is the sharp decline in performance when the interval is extended to τ = 30, with results even falling below the baseline achieved at τ = 15. This trend suggests that a moderate reduction in the sliding window’s movement speed (from τ = 15 to τ = 25) allows the temporal behavior within the window to be sufficiently optimized. Conversely, overly slow window movement (exceeding τ = 25) is prone to over-optimization at certain timesteps, causing the model’s distribution to diverge significantly from the target preferences of the reward models. The cross-dataset experiments further confirm the universality of τ = 25 as the optimal setting.

The comparative results for the window size (w) are presented in Table 15. It is evident that both w = 4 and w = 6 represent optimal settings. Specifically, w = 6 slightly surpasses w = 4 only on the in-domain test set of Cross-dataset Experiment 1, and in other experiments, it only holds a marginal advantage in the HPS-v2.1 [47] score. It is crucial to note that the window size w corresponds to the number of denoising timesteps that must be optimized during each training iteration, which, in turn, is linearly correlated with the overall optimization overhead of the Reinforcement Learning (RL) process. Consequently, w = 4 is identified as the best compromise, offering an optimal trade-off between performance and computational efficiency. Regarding parameter sensitivity, we observe that the change in performance metrics—across both in-domain and out-of-domain tests—is relatively minimal when w varies between 2 and 6. This stability suggests that the model exhibits robust performance with respect to the window size setting. As long as the window size remains within a reasonable range (e.g., 2 ≤ w ≤ 6), the model is capable of effective learning and maintaining stable performance, thereby reducing necessity for meticulous hyperparameter tuning.

The results concerning the stride (s), which defines the step size for each sliding window movement, are presented in Table 16. Considering all experimental metrics across the different datasets, a stride of s = 1 is identified as the optimal setting overall. It is important to note that the optimal window size of w = 4 was used in this experiment. With s = 1, the last three timesteps within

the window are repeatedly optimized in the subsequent training cycle. Crucially, the experimental results indicate that this repetition leads to performance improvement. This is likely due to the nature of the high-SNR denoising process, where the initial timesteps (e.g., the first timestep in the window) are subject to a greater extent of GRPO optimization, despite the window size being set at w = 4. Conversely, the behavior learned during the subsequent three timesteps remains relatively under-optimized (underfitted) and thus significantly benefits from this repeated optimization. Furthermore, we observe that the highest indomain HPS-v2.1 [47] score is achieved when s = 3. This localized optimal result may be attributed to the HPS-v2.1 [47] reward model’s preference alignment being relatively easier to optimize, suggesting that repeated optimization (i.e., smaller strides) could potentially induce over-optimization in this specific metric.In summary, s = 1 proves to be the most robust and generalizable choice across the varying experimental conditions.

Shift Interval (τ). The results concerning the shift interval (τ), which dictates the frequency of window movement in training iteration steps, are summarized in Table 14. Overall, the optimal performance is consistently observed at τ = 25. Specifically, as τ is incrementally increased from 15 to 25, performance metrics exhibit a gradual rise across both ID and OOD test sets. However, a crucial observation is the sharp decline in performance when the interval is extended to τ = 30, with results even falling below the baseline achieved at τ = 15. This trend suggests that a moderate reduction in the sliding window’s movement speed allows the temporal behavior within the window to be sufficiently optimized. Conversely, overly slow window movement (exceeding τ = 25) is prone to reward overfitting at certain timesteps, causing the model’s distribution to diverge significantly from the target preferences of the reward models. The crossdataset experiments further confirm the universality of τ = 25 as the optimal setting across different training sources.

Window Size (w) and Stride (s). The comparative results for window size (w) and stride (s) are presented in Tables 15 and 16. As visualized in Fig. 5, we observe three critical trends: (1) Consistent Superiority: MixGRPO’s performance (solid lines) consistently stays above the DanceGRPO and FlowGRPO baselines (dashed lines) regardless of the specific values of w or s. (2) Stability: The performance curves are relatively flat within a reasonable range (e.g., 2 ≤ w ≤ 6), indicating that the method is not overly sensitive to precise tuning. (3) Robustness: The same optimal configurations (w = 4,s = 1) yield peak performance across both ID and OOD scenarios, confirming its strong generalization.

Specifically, while window size w = 6 shows marginal gains in ID metrics, w = 4 provides the best trade-off between performance and computational overhead. Similarly, s = 1 demonstrates superior generalizability across OOD conditions, effectively preventing the model from collapsing into narrow reward-specific local optima by repeatedly optimizing critical high-SNR timesteps.

Summary on Generalization. In conclusion, these cross-dataset evaluations and sensitivity analyses confirm that MixGRPO’s performance gains are

- Table 13: Comparison for different moving strategies in cross-dataset experiments. The optimal setting is highlighted in green.

|Strategy<br><br>Interval Schedule<br><br>|Cross-dataset Experiment 1| |Cross-dataset Experiment 2| |
|---|---|---|---|---|
| |In Domain Dataset (HPD v2)<br><br>|Out-of-Domain Dataset (Pick-a-Pi v1)<br><br>|In Domain Dataset (Pick-a-Pi v1)|Out-of-Domain Dataset (HPD v2)|
| |HPS-v2.1 Pick Score ImageReward<br><br>|HPS-v2.1 Pick Score ImageReward|HPS-v2.1 Pick Score ImageReward<br><br>|HPS-v2.1 Pick Score ImageReward|
|Frozen /|0.354 0.234 1.580<br><br>|0.352 0.226 1.539<br><br>|0.346 0.230 1.601|0.351 0.231 1.587|
|Random Constant<br><br>|0.365 0.237 1.513<br><br>|0.361 0.230 1.512<br><br>|0.349 0.227 1.524|0.343 0.225 1.530|
|Progressive<br><br>Decay(Linear) Decay(Exp)|0.365 0.235 1.566 0.360 0.239 1.632<br><br>|0.363 0.229 1.572<br><br>0.364 0.232 1.612<br><br><br>|0.363 0.231 1.614 0.361 0.234 1.628<br><br>|0.355 0.230 1.597 0.358 0.230 1.584|

Constant 0.367 0.237 1.629 0.366 0.234 1.622 0.365 0.238 1.618 0.359 0.232 1.601

- Table 14: Comparison for different shift intervals τ in cross-dataset experiments. The optimal setting is highlighted in green.

|Cross-dataset Experiment 1<br><br>| |Cross-dataset Experiment 2| |
|---|---|---|---|
|In Domain Dataset (HPD v2)<br><br>|Out-of-Domain Dataset (Pick-a-Pi v1)|In Domain Dataset (Pick-a-Pi v1)<br><br>|Out-of-Domain Dataset (HPD v2)|
|HPS-v2.1 Pick Score ImageReward<br><br>|HPS-v2.1 Pick Score ImageReward<br><br>|HPS-v2.1 Pick Score ImageReward|HPS-v2.1 Pick Score ImageReward|
|0.366 0.237 1.509<br><br>0.366 0.238 1.610 0.367 0.237 1.629<br><br><br>|0.359 0.227 1.470<br><br>0.360 0.228 1.619 0.366 0.234 1.623<br><br><br>|0.358 0.234 1.610 0.361 0.237 1.615 0.365 0.238 1.618<br><br>|0.353 0.228 1.542 0.357 0.233 1.588 0.359 0.232 1.601<br><br>|
|0.350 0.229 1.589<br><br>|0.348 0.221 1.585<br><br>|0.355 0.234 1.609|0.351 0.229 1.509|

τ

15 20 25 30

- Table 15: Comparison for different window sizes w in cross-dataset experiments. The optimal setting is highlighted in green.

|Cross-dataset Experiment 1<br><br>| |Cross-dataset Experiment 2| |
|---|---|---|---|
|In Domain Dataset (HPD v2)<br><br>|Out-of-Domain Dataset (Pick-a-Pi v1)<br><br>|In Domain Dataset (Pick-a-Pi v1)|Out-of-Domain Dataset (HPD v2)|
|HPS-v2.1 Pick Score ImageReward|HPS-v2.1 Pick Score ImageReward<br><br>|HPS-v2.1 Pick Score ImageReward<br><br>|HPS-v2.1 Pick Score ImageReward|
|0.359 0.232 1.571<br><br>0.366 0.235 1.618<br>0.367 0.237 1.629<br>|0.349 0.223 1.445 0.363 0.230 1.614 0.366 0.234 1.623<br><br>|0.349 0.229 1.553 0.363 0.234 1.597 0.365 0.238 1.618<br><br>|0.346 0.221 1.565 0.359 0.230 1.585 0.359 0.232 1.601|
|0.370 0.238 1.624|0.364 0.232 1.620|0.366 0.237 1.608|0.362 0.232 1.594|

w

- 1
- 2 4 6

- Table 16: Comparison for different window strides s in cross-dataset experiments. The optimal setting is highlighted in green.

|Cross-dataset Experiment 1<br><br>| |Cross-dataset Experiment 2| |
|---|---|---|---|
|In Domain Dataset (HPD v2) HPS-v2.1 Pick Score ImageReward|Out-of-Domain Dataset (Pick-a-Pi v1)<br><br>HPS-v2.1 Pick Score ImageReward<br><br>|In Domain Dataset (Pick-a-Pi v1) HPS-v2.1 Pick Score ImageReward|Out-of-Domain Dataset (HPD v2) HPS-v2.1 Pick Score ImageReward<br><br>|
|0.367 0.237 1.629 0.357 0.236 1.575 0.370 0.236 1.578<br><br>0.368 0.238 1.575<br><br><br>|0.366 0.234 1.623 0.357 0.233 1.587 0.364 0.230 1.579 0.349 0.224 1.573<br><br>|0.364 0.238 1.618 0.363 0.234 1.574 0.368 0.237 1.586 0.366 0.231 1.566<br><br>|0.359 0.232 1.601 0.354 0.231 1.591 0.358 0.228 1.585 0.357 0.225 1.568<br><br>|

s

- 1
- 2
- 3
- 4

not tied to specific hyperparameter artifacts. The consistent superiority of our method over baselines across various configurations and unseen reward models underscores its methodological robustness and strong generalization capabilities.

- 17 Ablation Study on Intra-group Initial Noise

In this section, we investigate the impact of the initial noise setting within a sampling group during the MixGRPO training process. A potential concern in group-based reinforcement learning is that diverse initial noises across different samples within the same group might introduce unwanted variance, potentially leading to biased reward signals or "reward hacking."

Following the observations in DanceGRPO [49], we adopt a strategy of fixing the intra-group initial noise. This approach ensures that the performance variations within a group primarily stem from the model’s policy updates rather than the stochasticity of the starting latent states. By neutralizing the noise factor, the relative advantage of certain generations becomes a more reliable indicator for policy improvement.

To validate this design choice, we conducted an ablation study comparing MixGRPO with and without fixed intra-group initial noise. As shown in Table 17, fixing the initial noise consistently yields superior results across all preference benchmarks, including HPS-v2.1, Pick Score, and ImageReward. These results demonstrate that stabilizing the initial conditions effectively mitigates reward hacking and leads to more robust policy optimization.

Table 17: Ablation study of intra-group initial noise on MixGRPO performance.

###### Method HPS-v2.1 ↑ Pick Score ↑ ImageReward ↑

MixGRPO (w/o fixed initial noise) 0.342 0.228 1.448 MixGRPO (w/ fixed initial noise) 0.367 0.237 1.629

### 18 Hybrid Inference for Solving Reward Hacking

As discussed in Section 5, reward hacking stems from the limited evaluation capabilities of the reward model. To address reward hacking and improve visualization, we employ the hybrid inference strategy from [51], which uses the post-trained model for low-SNR (signal-to-noise ratio) steps and the original model for high-SNR steps during inference-time sampling. In our experiments, we also applied hybrid inference to the other baseline models to ensure a fair and consistent comparison.

We employ hybrid inference and introduce the hybrid percent pmix. This means that the initial, high-SNR pmixT denoising steps, are handled by the model trained with GRPO, while the remaining denoising process is finished by the original model. Table 18 and Figure 9 respectively illustrate the changes in performance and images as pmix increases under the multi-rewards training scenario. The experimental results demonstrate that pmix = 80% is an optimal empirical value that effectively mitigates hacking while maximizing alignment with human preferences.

Table 18: Comparison with different hybrid inference percentages

##### pmix HPS-v2.1 Pick Score ImageReward Unified Reward

0% 0.313 0.226 1.089 3.369 20% 0.342 0.233 1.372 3.386 40% 0.356 0.235 1.539 3.395 60% 0.362 0.236 1.598 3.407 80% 0.366 0.238 1.610 3.411 100% 0.369 0.238 1.607 3.378

Prompt: A painting depicting a snowy winter scene featuring a river, a small house on a hill, and a dreamy cloudy sky.

HACKING!

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

0% 20% 40% 60% 80% 100%

- Fig. 9: Qualitative comparison with different hybrid inference percentages. We found 80% to be the optimal value, as it maximizes image quality without causing overoptimization.

### 19 Coefficients-Preserving Sampling

In our MixGRPO framework, introducing stochasticity during the inference phase is crucial for effective exploration in reinforcement learning. While a common practice involves the use of Stochastic Differential Equations (SDEs), we adopt Coefficients-Preserving Sampling (CPS) [43] as a more refined alternative to maintain the integrity of the probability path.

The standard SDE-based sampling, often discretized through the Euler-Maruyama method, follows the update rule:

√

2σ2∆tϵi, ϵi ∼ N(0,I) (24)

i − vt

#### xt

= xt

(xt

,ti)∆t +

i−1

i

i

Although this formulation provides the necessary stochasticity, it tends to inject excessive independent noise at each discretization step, resulting in "grainy" artifacts in the generated samples. As observed in our preliminary experiments, such high-frequency noise can lead to reward hacking, where reward models (e.g., Pick Score [15] or HPS-v2.1 [47]) prioritize low-level textures over structural coherence, thereby hindering the convergence of RL optimization.

To address these issues, we utilize CPS [43], which reformulates the transition process by drawing inspiration from DDIM [37] framework. The update rule for CPS [43] is defined as:

1 − ti−1 1 − ti

1 − ti−1 1 − ti

ϵi (25)

+ ti−1 −

#### xt

=

#### xt

ti vt

+ σt

i−1

i

i

i

The core advantage of CPS [43] lies in its strategic construction of coefficients for xt

, which preserves the linear interpolation structure of Flow Matching while introducing controlled stochasticity via σt

and vt

i

i

. By effectively eliminating sampling artifacts inherent in SDE-based methods, CPS [43] yields cleaner images that provide more reliable feedback for MixGRPO. In our implementation, we set NFEπ

i

= 25, window size w = 4, and window stride s = 1 with fixed initial noise, employing HPS-v2.1 [47], Pick Score [15], and ImageReward [48] as multi-reward metrics. Both the original SDE sampling and CPS [43] sampling were trained for 300 steps and evaluated on the HPDv2 dataset [47]. The quantitative comparison is moved to the main ablation section (Table 10), and qualitative comparisons are shown in Figure 13.

θold

### 20 More Visualized Results

PROMPT: An image of an aircraft carrier made of cheese.

[Figure 75]

[Figure 76]

[Figure 77]

FLUX DanceGRPO MixGRPO

PROMPT: 16-year-old teenager wearing a white bear-ear hat with a smirk on their face.

[Figure 78]

[Figure 79]

[Figure 80]

FLUX DanceGRPO MixGRPO

PROMPT: A lemon with a McDonald's hat.

[Figure 81]

[Figure 82]

[Figure 83]

FLUX DanceGRPO MixGRPO

- Fig. 10: Comparison of the visualization results of FLUX, DanceGRPO, and MixGRPO under HPS-v2.1 as the reward model.

PROMPT: A photorealistic image from a furry fandom convention set in a biopunk era after the genetic revolution and quantum singularity.

[Figure 84]

[Figure 85]

[Figure 86]

FLUX DanceGRPO MixGRPO

PROMPT: a castle is in the middle of a eurpean city

[Figure 87]

[Figure 88]

[Figure 89]

FLUX DanceGRPO MixGRPO

PROMPT: A detailed soft painting of a bat with golden rose flowers and amethyst stained glass in the background.

[Figure 90]

[Figure 91]

[Figure 92]

FLUX DanceGRPO MixGRPO

###### Fig. 11: Comparison of the visualization results of FLUX, DanceGRPO, and MixGRPO under HPS-v2.1 and CLIP Score as multi-reward models.

PROMPT: a cute polar bear baby, digital oil painting by paul nicklen and by van gogh and monet

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Flow-GRPO MixGRPO

SD3.5

Flow-DPO(offline)

Flow-DPO(online)

PROMPT: A chocolate cake with the word "SD" written on it, professional photography, food photography

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

SD3.5

Flow-DPO(offline)

Flow-DPO(online)

Flow-GRPO MixGRPO

PROMPT: a photo of a yellow sports ball and a green boat

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

SD3.5 Flow-GRPO MixGRPO

Flow-DPO(offline) Flow-DPO(online)

###### Fig. 12: Comparison of the visualization results of SD3.5-M, offline DPO, online DPO, Flow-GRPO and MixGRPO under HPS-v2.1, Pick Score and ImageReward as multireward models.

PROMPT: A key shot of an Australian Shepherd with a pastel color palette and dramatic lighting.

[Figure 108]

[Figure 109]

[Figure 110]

FLUX MixGRPO-SDE MixGRPO-CPS

PROMPT: Portrait of an anime princess in white and golden clothes.

[Figure 111]

[Figure 112]

[Figure 113]

FLUX MixGRPO-SDE MixGRPO-CPS

PROMPT: A colorful digital painting with a front view and anime-inspired vibes featuring a magical composition.

[Figure 114]

[Figure 115]

[Figure 116]

FLUX MixGRPO-SDE MixGRPO-CPS

###### Fig. 13: Comparison of the visualization results of FLUX, SDE sampling and CPS sampling

