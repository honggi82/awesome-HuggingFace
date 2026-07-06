## EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning

Wujiang Xu1 Wentian Zhao2 Zhenting Wang1 Yu-Jhe Li2 Can Jin1 Mingyu Jin1 Kai Mei1 Kun Wan2 Dimitris N. Metaxas1

# arXiv:2509.22576v2[cs.LG]10Feb2026

### Abstract

Reinforcement learning has enabled LLMs to acquire reasoning through single-turn verified rewards, yet extending this paradigm to multi-turn agents—where tasks span 30+ turns with only terminal rewards—introduces fundamental challenges. We identify a critical failure mode in this setting: the exploration-exploitation cascade failure. Unlike single-turn training, multi-turn agents share policy parameters across all turns, meaning entropy adjustments cannot independently control early-turn exploration and late-turn exploitation. Combined with sparse rewards that provide no intermediate correction, this coupling causes severe entropy oscillations that destabilize training. We propose Entropy-regularized Policy Optimization (EPO), a framework that stabilizes entropy dynamics through three synergistic mechanisms: (1) trajectory-level entropy regularization that captures multi-turn temporal structure, (2) an entropy smoothing regularizer that penalizes deviations from historical averages to dampen oscillations, and (3) adaptive phase-based weighting that transitions from conservative exploration to strong stabilization across training. EPO achieves up to 152% improvement on ScienceWorld and 19.8% on ALFWorld, transforming previously unstable sparse-reward scenarios into smoothly converging optimization. Our work demonstrates that multi-turn settings require fundamentally different entropy control than traditional RL. The code is available at the URL 1.

### 1. Introduction

Reinforcement Learning (RL) (Shao et al., 2024; Schulman et al., 2017) has become an important approach for post-

1Rutgers University, USA 2Adobe Inc., USA. Correspondence to: Wujiang Xu <wujiang.xu@rutgers.edu>. Preprint. February 11, 2026.

1https://github.com/WujiangXu/EPO

training Large Language Models (LLMs), enabling them to acquire reasoning ability through single-turn verified rewards (Guo et al., 2025; Yang et al., 2025). Extending this paradigm to multi-turn LLM agents, which interact with environments across extended episodes, has achieved promising results in coding (Anthropic, 2025; OpenAI, 2025b), computer use (OpenAI, 2025a; Xie et al., 2024), and web search (Team et al., 2025; Zeng et al., 2025), yet introduces fundamental challenges. In multi-turn settings, episodes can span 30+ turns (Shridhar et al., 2021; Wang et al., 2022) yet rewards arrive only at task completion. This means early-turn actions shape entire trajectories without receiving corrective feedback, making exploration-exploitation balance (Sutton, 1988) a critical challenge that single-turn methods handle implicitly through immediate rewards.

Traditional RL approaches (Haarnoja et al., 2018; Mnih et al., 2016; Williams & Peng, 1991) employ entropy regularization to address exploration, adding the policy’s entropy to the objective function to discourage premature convergence to deterministic policies. In LLM training, researchers have adapted these mechanisms to reward high-entropy token generation (Cui et al., 2025; Dong et al., 2025; He et al., 2025; Wang et al., 2025a), operating under the assumption that entropy collapse is the primary failure mode. Through extensive empirical analysis on standard multi-turn agent benchmarks, we uncover a critical limitation in multi-turn agent training settings: the challenge is not entropy collapse but rather uncontrolled entropy dynamics. In multi-turn training, the same policy parameters control all turns. Since all turns share parameters, adjusting entropy weights per turn cannot independently control exploration and exploitation: increasing entropy for early turns inevitably affects late turns, and vice versa. Combined with sparse rewards that provide no intermediate correction signal, this makes training dynamics inherently unstable. We term this the exploration-exploitation cascade failure.

As illustrated in Figure 1, we empirically validate this limitation. In Figure (a) and (b), we attempt a natural solution: applying higher entropy weights to early turns to promote exploration, and lower weights to late turns to encourage exploitation (EPO-Decay). However, due to shared policy

[Figure 1]

[Figure 2]

[Figure 3]

(a) Early Steps (b) Late Steps (c) PPO vs PPO+EPO

- Figure 1. The exploration-exploitation cascade failure in multi-turn training. (a-b) Adjusting per-turn entropy weights fails due to shared parameters: EPO-Decay applies higher weights to early turns and lower weights to late turns, yet entropy curves remain nearly identical for both early steps (a) and late steps (b). (c) Standard PPO exhibits severe entropy fluctuation (purple) with stagnant rewards (yellow), while our proposed EPO maintains stable entropy (green, declining from 1.2 to 0.3) and achieves consistent reward improvement (red).

parameters, early-turn and late-turn entropy curves remain nearly identical to the unweighted baseline (EPO-Base). This confirms that per-turn weight adjustment cannot break the coupling between turns. Figure (c) shows the consequence: PPO exhibits severe entropy fluctuation with stagnant rewards, while our proposed EPO maintains stable declining entropy and consistent reward growth. This cascade failure exposes a mismatch between existing entropy methods and multi-turn structure. Standard entropy regularization operates statelessly, treating each gradient update in isolation. However, in multi-turn training, shared parameters mean that each gradient update affects all turns simultaneously, so entropy at specific step depends on the entire training history, not just the current update. This raises a critical challenge: how can we stabilize entropy dynamics over time when shared parameters prevent independent per-turn control?

To address this cascade issue, we propose Entropyregularized Policy Optimization (EPO), a novel framework that combines entropy regularization with specialized mechanisms for stable on-policy training under sparse reward conditions. Our contributions are fourfold.

❶ We empirically discover and formally characterize the exploration-exploitation cascade failure on standard multiturn benchmarks, demonstrating that this novel failure mode, where entropy oscillations compound across turns under sparse rewards, is the primary cause of poor performance in standard RL baselines.

❷ We extend standard entropy regularization to multiturn settings, where entropy accumulates across sequential turns through shared parameters, requiring trajectory-level aggregation rather than per-step computation.

❸ We introduce an entropy smoothing regularizer that penalizes deviations from historical entropy averages, effectively reducing the oscillations between overconfidence and over-exploration that characterize cascade failure.

❹ We develop an adaptive weighting scheme that dynamically balances exploration and exploitation across training phases, applying stronger smoothing early to establish

controlled exploration, relaxing mid-training as the policy stabilizes, and maintaining sufficient regularization in later phases to ensure convergence.

Together, these components create a theoretically grounded and general framework that prevents the empirically observed cascade failure and ensures optimal explorationexploitation trade-offs while being compatible with any on-policy optimization method. We validate EPO on challenging benchmarks ScienceWorld (Wang et al., 2022) and ALFWorld (Shridhar et al., 2021), achieving up to 152% performance improvement with more stable training dynamics, transforming previously unstable sparse-reward scenarios into smoothly converging optimization problems.

### 2. Related Work

#### 2.1. Reinforcement Learning for LLMs

RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) have become foundational approaches for aligning LLMs with human preferences, with both methods significantly improving model alignment and instruction-following capabilities. Recent RL methods such as GRPO (Shao et al., 2024) and DAPO (Yu et al., 2025) further enhance LLM reasoning abilities during post-training through verified rewards. In contrast to PPO (Schulman et al., 2017), these methods leverage batch-wise advantage computation from identical prompts, obviating the critic model and substantially improving the computational tractability of large-scale RL training for LLMs. However, a primary challenge in applying RL to LLMs is the phenomenon of policy entropy collapse: models rapidly reduce their stochasticity, converging on narrow, over-optimized behaviors (Cui et al., 2025; Dong et al., 2025; Wang et al., 2025a; Deng et al., 2025; He et al., 2025; Cheng et al., 2025a;b). In response, recent work has focused on integrating entropy control mechanisms into the optimization process to preserve policy diversity. For instance, Cui et al. (Cui et al., 2025) regulate the impact of high-covariance tokens by applying a clipping function and a KL penalty. Meanwhile, Cheng et al. (Cheng et al., 2025b)

augment the advantage function with a clipped, gradientdetached entropy term, which encourages deeper reasoning chains changing the original policy optimization direction.

- 2.2. Reinforcement Learning for LLM Agents

To enhance their autonomy, LLM agents are designed to interact with external environments using diverse toolsets (OpenAI, 2025a; Team, 2025). However, training agents to complete multi-step tasks with these tools presents significant challenges for standard reinforcement learning, including sparse rewards and credit assignment problems. To address these issues, seminal works have introduced advanced training paradigms such as hierarchical RL (Zhou et al., 2024), autonomous learning (Bai et al., 2024), and off-policy Q-learning (Bai et al., 2025). Meanwhile, another line of research employs supervised fine-tuning (SFT) to directly enhance the models’ decision-making abilities, training them on vast datasets of high-quality tool-use trajectories to master complex environments and APIs (Xi et al., 2024; Zhang et al., 2024; Qin et al., 2024). Recently, to leverage the training stability of GRPO (Shao

- et al., 2024), researchers have extended single-turn GRPO to multi-turn training settings with various training techniques to improve performance (Jin et al., 2025; Feng et al., 2025; Wang et al., 2025b). To guide learning in long-horizon scenarios, RLVMR (Zhang et al., 2025) introduces verifiable meta-reasoning rewards that provide dense, intermediate feedback on the agent’s reasoning process. However, these methods overlook the entropy dynamics unique to multi-turn settings. Because all turns share policy parameters, entropy adjustments cannot independently control early-turn exploration and late-turn exploitation. This coupling, combined with sparse rewards that provide no intermediate correction, causes the exploration-exploitation cascade failure, where severe entropy oscillations destabilize training and prevent effective long-horizon learning.

- 3. Preliminary

- 3.1. On-policy Optimization

On-policy optimization is a fundamental paradigm in reinforcement learning where the agent learns to improve its policy by directly optimizing the expected return using trajectories sampled from the current policy. Given a parameterized stochastic policy πθ(a|s) with parameters θ ∈ Rd, the objective is to maximize the expected return,given by J(θ) = Eτ∼π

[R(τ)], where τ denotes a trajectory and

θ

R(τ) = Tt=0 γtrt is the discounted return. On-policy optimization methods build on:

T

∇θ log πθ(at|st)Aπ

∇θJ(θ) = Eτ∼π

(st,at)

θ

θ

t=0

(1) where Aπ

(st) is the advantage function. While the policy gradient provides an unbiased estimator of ∇θJ(θ), directly optimizing this objective can lead to instability due to large policy updates. To address this, modern on-policy methods employ surrogate objective functions that approximate the policy gradient while ensuring stable learning. The standard policy gradient can be reformulated as the surrogate objective LPG(θ) = Eτ∼π

(st,at) = Qπ

(st,at) − V π

θ

θ

θ

πθ(at|st) πθold(at|st)

Aˆt . However, this surrogate objective is only valid for infinitesimally small updates. Proximal Policy Optimization (PPO) (Schulman et al., 2017) addresses this limitation by constraining the policy ratio through a clipped surrogate objective: LCLIP(θ) = Eτ∼π

θold

min rt(θ)Aˆt,clip(rt(θ),1 − ϵ,1 + ϵ)Aˆt where rt(θ) = π

θold

θ(at|st)

πθold(at|st) is the importance sampling ratio, Eτ∼π

[·] denotes expectation over trajectories sampled under πθ

θold

, and ϵ defines the trust region bounds. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) extends PPO by modifying the advantage function computation. Instead of using standard advantages Aˆt, GRPO employs group-relative advantages, computed as A˜t = Rσt−µg

old

g+δ . where Rt is the return from timestep t, µg and σg are the mean and standard deviation of returns within group g, and δ is a small constant for numerical stability. This group-based normalization provides more stable gradient estimates.

#### 3.2. Problem formulation

We formalize the multi-turn task as a sequential decisionmaking reinforcement learning problem. A single LLM agent πθ executes a task through T turns over a trajectory τ = (s0,a0,r0,...,sT,aT,rT). The reward is sparse, with rt = 0 for all intermediate turns and only the final turn receiving the task outcome reward, such that the total return is R(τ) = Tt=0 γtrt = rT, where rT ∈ {0,1} represents the binary task outcome. In our experimental settings, specifically ALFWorld (Shridhar et al., 2021) and SciWorld (Wang et al., 2022), we assume an undiscounted formulation where γ = 1. All turns within the same task share the final outcome reward, creating a credit assignment challenge across sequential turns. The multi-turn policy optimization differs from standard RL in that losses accumulate across all T turns before parameter updates: LMT(θ) = Eτ∼π

[Et∼T [min(rt(θ)At,clip(rt(θ),1 − ϵ,1 + ϵ)At)]] where rt(θ) = π

θold

θ(at|st)

πθold(at|st) and At represents the advantage estimate at turn t within the multi-turn trajectory. Et[·]

denotes expectation over timesteps within turn t, while the outer expectation is over trajectories sampled under πθ

.

old

### 4. Methodology

#### 4.1. Entropy Regularization

To address the exploration-exploitation cascade failure, we first adapt entropy regularization to capture the temporal structure of multi-turn interactions. Unlike traditional RL where entropy is computed per-step, we recognize that in multi-turn environments, decisions compound across subsequent turns through shared policy parameters. Therefore, we compute entropy across all turns within each trajectory and average over the batch of trajectories. The entropy-regularized policy loss is formulated as LER(θ) = LMT(θ) − λLH(θ), where λ is the entropy coefficient, and the entropy loss is averaged over the batch of trajectories:

|τj,t|

T−1

B

1 B

1 |τj,t|

1 T

LH(θ) =

Hj,t,i (2)

t=0

j=1

i=1

where B is the batch size (number of trajectories), T is the number of turns per trajectory, Hj,t,i is the entropy at token position i in turn t of trajectory j, and |τj,t| represents the sequence length at turn t of trajectory j. The token-level entropy H is computed from the model’s probability distribution over the vocabulary at each position as H = − v∈V p(v|w<t)log p(v|w<t), where p(v|w<t) is the probability of token v from vocabulary V given the preceding context w<t.

#### 4.2. Entropy Smoothing Regularizer

While trajectory-level entropy regularization captures multiturn structure, it cannot prevent the severe entropy oscillations caused by shared parameters and sparse rewards. To stabilize entropy dynamics, we introduce an entropy smoothing mechanism that anchors current entropy to historical averages. We maintain an entropy history window Wk = {H¯0,...,H¯m,...,H¯k−1} for RL step k, storing the average entropy H¯m across all trajectories at the token level for each previous RL step m. The historical entropy reference is computed as the mean H¯W

= k1 km−=01 H¯m. This historical anchoring dampens the entropy fluctuations that destabilize training under sparse reward conditions. We apply a token-wise penalty based on acceptable entropy ranges relative to this historical average:

k

Pn,t,i = κlH¯W

k

− Hn,t,i + + Hn,t,i − κrH¯W

k

+ (3)

where [x]+ = max(0,x) denotes the ReLU function, and κl, κr define the acceptable entropy corridor. By bounding entropy within historical averages, we prevent the severe

oscillations that arise from shared parameter coupling under sparse rewards. Aggregating these penalties across all tokens, turns, and trajectories yields the smoothing loss:

Lsmooth(θ) =

T−1

B

1 B

1 T

n=1

t=0

|τn,t|

1 |τn,t|

αPn,t,i (4)

i=1

where α provides the penalty weight for tokens with entropy outside the acceptable range. The complete entropysmoothed policy optimization loss is then defined as LEPO(θ) = LMT(θ) − λ[LH(θ) − βkLsmooth(θ)], where the dynamic coefficient βk follows an exponential schedule that adapts smoothing strength across training:

k

βk = 1 + e−γ

kmid (5)

This adaptive schedule stabilizes entropy dynamics throughout training: it begins with stronger smoothing to establish controlled exploration, gradually relaxes as the policy stabilizes around mid-training (kmid = ⌊K/2⌋), and maintains sufficient regularization in later phases to ensure smooth convergence. The parameter γ controls the decay rate of this transition. Overall, the policy is updated according to the following objective:

LEPO(θ) = LMT(θ) − λ[LH(θ) − βkLsmooth(θ)]. (6)

where λ is the entropy coefficient controlling the overall strength of entropy regularization, and βk is a dynamic coefficient that adapts the smoothing strength across training phases. By decoupling the entropy terms from the policy loss, this formulation provides direct gradient signals ∇θLH(θ) to guide exploration while the smoothing regularizer Lsmooth(θ) ensures stability by anchoring entropy within historical bounds. Algorithm 1 presents the full optimization procedure, incorporating these components.

### 5. Experiments

We validate EPO on ScienceWorld and ALFWorld, two sparse-reward benchmarks requiring 30+ turn interactions. Our experiments examine: (1) performance gains over existing RL methods, (2) necessity of each proposed component via ablation, and (3) failure modes of alternative entropy control strategies. Results demonstrate that EPO effectively eliminates cascade failure, transforming unstable training into smooth convergence.

#### 5.1. Experiments Setup

This section outlines our experimental setup, including the benchmarks, evaluation protocol, baselines, and implementation details. Further details can be found in Appendix B.

Table 1. For PPO and GRPO baselines with our EPO method: better performance indicates improvement over baseline, worse performance indicates degradation. Highlighted values represent the best performance among other baseline methods. ∆ shows the relative improvement (%) when applying our method. Results for other baseline methods (ReAct, AgentGym, SFT, GiGPO, RLVMR) are sourced from the RLVMR (Zhang et al., 2025) paper. We ran our own implementations of PPO, GRPO, and EPO, tuning hyperparameters across multiple trials to obtain stable results.

|Method<br><br>|ScienceWorld LLM IID OOD<br><br>Succ.∗ Succ. Succ.∗ Succ.<br><br>|ALFWorld LLM IID OOD<br><br>Succ.∗ Succ. Succ.∗ Succ.<br><br>|
|---|---|---|
|ReAct(Yaoetal.,2023) ReAct(Yaoetal.,2023) ReAct(Yaoetal.,2023)<br><br>|GPT-4o 45.4 - 49.2 DeepSeek-R1 22.2 - 31.4 -<br><br>Qwen2.5-7B 7.8 - 11.3 -|GPT-4o 57.3 - 66.0 DeepSeek-R1 68.8 - 70.2 Qwen2.5-7B 23.1 - 28.5 -|
|AgentGym(Xietal.,2024) SFT|LLaMa2-7B 46.9 - 33.6 Qwen2.5-7B 36.7 - 32.0 -|LLaMa2-7B 76.6 - 63.3 Qwen2.5-7B 63.3 - 57.0 -<br><br>|
|GiGPO(Fengetal.,2025) RLVMR(Zhangetal.,2025)|Qwen2.5-7B 53.4 - 35.2 Qwen2.5-7B 67.2 - 43.0 -<br><br>|Qwen2.5-7B 89.5 - 90.2 Qwen2.5-7B 91.4 - 91.8 -|
|PPO(Schulmanetal.,2017)<br><br>+EPO ∆ GRPO(Shaoetal.,2024)<br><br>+EPO ∆|Qwen2.5-7B 64.6 38.4 58.3 39.1 Qwen2.5-7B 100.0 96.8 100.0 96.2<br><br>54.8% 152.1% 71.5% 146.0%<br><br>Qwen2.5-7B 93.8 81.6 91.7 80.9 Qwen2.5-7B 95.8 83.8 95.8 81.3<br><br>2.1% 2.7% 4.5% 0.5%|Qwen2.5-3B 95.8 72.3 87.5 70.9 Qwen2.5-3B 85.4 73.4 91.7 74.3<br><br>-10.9% 1.5% 4.8% 4.8%<br><br>Qwen2.5-3B 87.5 63.3 83.3 63.5 Qwen2.5-3B 91.7 75.8 89.6 75.4<br><br>4.8% 19.8% 7.6% 18.7%|

Algorithm 1 Entropy-regularized Policy Optimization (EPO)

Require: Policy πθ, entropy coefficient λ, penalty weight

α, corridor bounds κl,κr, decay rate γ, total steps K

- 1: Initialize θ0, entropy history W0 ← ∅
- 2: for k = 0,1,...,K − 1 do
- 3: Collect trajectories Dk = {τj}Bj=1 using πθ

old

- 4: Compute advantages Aˆt and policy loss LMT(θ)
- 5: Compute entropy loss LH(θ) via Equation 2
- 6: if k > 0 then
- 7: H¯W

k

← k1 km−=01

H¯m {Entropy mean}

- 8: Compute loss Lsmooth(θ) via Equation 4
- 9: else
- 10: Lsmooth(θ) ← 0
- 11: end if
- 12: βk ← 1+exp(−γ ·k/kmid) where kmid = ⌊K/2⌋
- 13: Update θ by minimizing Equation 6
- 14: Wk+1 ← Wk ∪ {H¯k} {Append current batch entropy}
- 15: end for Ensure: Optimized policy parameters θK

Benchmark. We evaluate on two challenging benchmarks that require different reasoning capabilities, ScienceWorld (Wang et al., 2022) and ALFWorld (Shridhar et al., 2021). ScienceWorld focuses on text-based scientific experimentation, demanding systematic hypothesis testing and structured exploration. ALFWorld is an embodied environment containing 4,639 household task instances across six categories, requiring multi-step decision-making and spatial reasoning. To improve the generalizability of our approach across multiple scenarios, we finetune the foundation model directly on the environment using RL, rather than employing

trajectory finetuning for initialization.

Evaluation Setting. To evaluate generalization capabilities, we focus on two key evaluation scenarios: IID (indistribution) covers seen task variants and categories, while OOD (out-of-distribution) evaluates on unseen task variants within seen categories. This design allows us to measure both optimization effectiveness and generalization robustness, which are crucial for practical deployment. We employ dual success rate metrics to capture different aspects of performance: Succ.∗ reports the average of maximum success rates across random seeds, while Succ. measures average performance after convergence, reflecting practical reliability. Given the high variance inherent in RL, final performance scores alone can be misleading. We therefore present averaged curves to provide a more robust comparison and illustrate the performance evolution throughout the training process.

Baselines. We conduct comprehensive comparisons across multiple paradigms: (1) Prompting-based approaches such as ReAct (Yao et al., 2023) that utilize in-context learning without parameter optimization; (2) Trajectorybased and platform methods including supervised finetuning (SFT) through expert trajectory imitation and AgentGym (Xi et al., 2024) which provides a unified framework with behavioral cloning and self-evolution mechanisms; (3) General reinforcement learning approaches encompassing standard on-policy methods (PPO (Schulman et al., 2017), GRPO (Shao et al., 2024)); (4) Agent RL approaches including recent methods specifically designed for agent training (GiGPO (Feng et al., 2025), RLVMR (Zhang et al., 2025)). Our proposed EPO method is architected as a general enhancement framework that can be seamlessly integrated with existing RL paradigms, as exemplified through our PPO+EPO and GRPO+EPO implementations.

[Figure 7]

[Figure 8]

[Figure 9]

(d) Training Rewards (ALFWorld) (e) IID Success Rate (ALFWorld) (f) OOD Success Rate (ALFWorld)

- Figure 2. Training dynamics and generalization performance. (a-c) ScienceWorld results comparing PPO and PPO+EPO across training rewards, IID success rate, and OOD success rate. (d-f) ALFWorld results comparing GRPO and GRPO+EPO under identical metrics. EPO eliminates the characteristic entropy oscillations of cascade failure, transforming unstable training into smooth convergence with substantial gains in both IID and OOD scenarios.

Implementation Details. To optimize performance for each benchmark’s complexity, we employ Qwen2.5-3BInstruct for ALFWorld and the larger Qwen2.5-7B-Instruct for ScienceWorld’s more complex scientific reasoning tasks. Constrained by computational resources, we adopt a single foundation model per task whose size is sufficient to ensure proper convergence, as smaller models consistently fail to converge. We conduct our own implementations and experiments for our proposed method, PPO, and GRPO baselines across three random seeds to ensure statistical reliability. Results for other baseline methods (ReAct, AgentGym, SFT, GiGPO, RLVMR) are sourced from the RLVMR’s (Zhang

- et al., 2025) paper. To account for their different convergence characteristics, we trained the model for 120 RL steps on ScienceWorld and 150 steps on ALFWorld.

#### 5.2. Performance Comparison

Quantitative Results Analysis. Table 1 presents comprehensive performance comparisons across both ScienceWorld and ALFWorld environments. Our EPO enhancement demonstrates substantial improvements when integrated with existing RL methods. Notably, PPO+EPO achieves a remarkable 152.1% improvement in averaged success rates (Succ.) on ScienceWorld IID tasks, significantly outperforming agent-specialized methods including GiGPO(53.4%) and RLVMR(67.2%). This dramatic improvement stems from EPO’s ability to address the exploration-exploitation cascade failure: PPO’s aggressive policy updates amplify entropy oscillations under sparse rewards, which EPO’s smoothing regularizer effectively dampens by anchoring entropy within historical bounds. In

contrast, GRPO’s group-relative advantage computation provides inherent stability against such oscillations, yielding more modest but consistent improvements (19.8% on ALFWorld IID) when combined with EPO. We emphasize the Succ. metric as it represents performance averaged across multiple evaluation episodes, providing a more robust comparison by reducing variance from individual runs.

Training Dynamics Analysis. Figure 2 illustrates training dynamics and validation performance. The reward curves demonstrate that EPO-enhanced methods achieve substantially higher reward accumulation with superior stability. In ScienceWorld, PPO+EPO reaches approximately 2× higher training rewards (15 vs. 8) with smooth monotonic trajectories, while GRPO+EPO maintains steady upward trends on ALFWorld. The validation curves reveal rapid convergence: on ScienceWorld, EPO variants achieve high success rates (>0.8 for both IID and OOD) within 40 steps, compared to baselines that struggle to exceed 0.4 after 100 steps. On ALFWorld, EPO-enhanced approaches demonstrate consistent performance with reduced variance, particularly in OOD evaluation where baselines frequently drop below 0.2 while EPO variants maintain performance above 0.4.

The key pattern across both environments is the elimination of severe entropy oscillations, the characteristic signature of cascade failure (Figure 1(c)), that destabilize standard RL methods. By maintaining entropy within historical bounds through our smoothing regularizer, EPO transforms previously unstable sparse-reward scenarios into smoothly converging optimization, validating our framework’s effectiveness in multi-turn LLM agent training.

[Figure 13]

[Figure 14]

[Figure 15]

(d) Training Rewards (ALFWorld) (e) IID Success Rate (ALFWorld) (f) OOD Success Rate (ALFWorld)

- Figure 3. Ablation studies on entropy regularization components. (a-c) ScienceWorld comparison of EPO versus EPO-Base without entropy smoothing, demonstrating that smoothing is essential for stable convergence in sparse reward settings. (d-f) ALFWorld comparison of EPO with dynamic βk versus EPO W/O DW using constant β, showing that adaptive weighting accelerates early training progress.

[Figure 16]

(a) EPO-Base vs EPO-Decay (b) Early vs Late Entropy (c) EPO vs EA Success Rate

[Figure 17]

[Figure 18]

- Figure 4. Model studies on ScienceWorld employing PPO with weighted entropy loss and entropy-based advantage shaping (Cheng et al., 2025b).
- 5.3. Ablation Study

ing the adaptive βk with a constant weight (EPO W/O DW) yields comparable final performance (0.7–0.8 success rate) but results in slower convergence during the initial 40–60 episodes and increased training variance. The adaptive schedule (Equation 5) automatically modulates regularization intensity across training phases, applying stronger smoothing early to establish controlled exploration and relaxing mid-training as the policy stabilizes. This improves convergence efficiency without sacrificing final performance. See Appendix B.3 for detailed analysis.

Figure 3 presents ablation studies on two key components: the entropy smoothing regularizer and its dynamic weighting coefficient βk, evaluated across two base methods (PPO, GRPO) and two environments (ScienceWorld, ALFWorld).

Entropy Smoothing Regularizer. On ScienceWorld, removing the smoothing regularizer (EPO-Base) severely degrades performance. EPO-Base exhibits delayed convergence with minimal rewards until step 40 and plateaus at 0.5–0.6 success rate, while full EPO achieves meaningful learning by step 20 and reaches 0.8–1.0 success rate, demonstrating a 50–60% relative improvement. This gap directly reflects the cascade failure mechanism: ScienceWorld’s sparse rewards induce severe entropy oscillations that compound across turns, which the smoothing regularizer effectively dampens by anchoring entropy within the corridor [κl,κr].

#### 5.4. Model Study

Turn-wise vs. Uniform Entropy Weighting. We compare consistent entropy regularization (PPO+EPO-Base) against a decaying schedule (PPO+EPO-Decay), which applies higher entropy weights to early turns and lower weights to later turns. Counter-intuitively, Figure 4(a) shows that decay consistently underperforms by prematurely suppressing early-turn exploration, locking agents into suboptimal strate-

Dynamic Weighting Coefficient. On ALFWorld, replac-

gies. Figure 4(b) further reveals the mechanism: while both methods exhibit similar early-step entropy, the decay schedule causes late-step entropy to drop too rapidly, reducing the policy’s ability to recover from suboptimal trajectories.

Decoupled Regularization vs. Advantage Reshaping. We compare EPO against the Entropy-based Advantage (EA) method from Cheng et al. (2025b). As shown in Figure 4(c), EPO converges to near-perfect success rates (∼1.0) while EA plateaus at 0.5–0.6. EA incorporates entropy directly into the advantage function, reshaping the policy gradient. In contrast, EPO decouples entropy as a separate regularization term, providing direct gradient signals while preserving the integrity of advantage-based credit assignment.

Cumulative vs. Sliding Entropy Window. Our cumulative entropy window Wk is an intentional design choice for longhorizon training. With K = 120-150 steps, the cumulative average adapts quickly when k is small (weight ∼ 1/k) and stabilizes later to anchor entropy against cascade failure. This design synergizes with the adaptive βk: while βk modulates smoothing strength dynamically, a sliding window would create conflicting adaptation mechanisms.

These studies reveal two key insights: (1) multi-turn sparsereward tasks require sustained exploration rather than conventional exploration-to-exploitation scheduling, and (2) decoupled entropy regularization with temporal smoothing is superior to advantage reshaping for maintaining longhorizon policy stability.

### 6. Theoretical Analysis

We provide theoretical analysis to understand EPO’s design and characterize when and why it outperforms standard entropy regularization. Our analysis reveals that EPO is a principled approach grounded in constrained optimization theory, with particular advantages in multi-turn settings where entropy instability compounds across turns. Full proofs are provided in Appendix A.

Proposition 6.1 (Exact Penalty Interpretation). The EPO objective equals Vλ,βπ = V π + λH(π) − ρk · P(π), where P(π) is the exact penalty for violating the entropy corridor [κlH¯ref,κrH¯ref], and ρk = λβkα is an adaptive penalty weight.

This connects EPO to classical optimization: the smoothing loss is precisely the penalty for constraint violation, and the adaptive schedule implements curriculum-style enforcement—stronger early to guide entropy toward the corridor, relaxing later as stability is learned. While Proposition 6.1 establishes EPO’s principled foundation, it does not explain why EPO’s benefits are disproportionately large in multi-turn settings. The key insight is how entropy errors propagate:

- Proposition 6.2 (Multi-Turn Entropy Stability). Define cu-

mulative entropy deviation ΓT(π) := Tt=1 |H¯t(π)−H¯ref| over T turns.

EPO: Under the corridor constraint, ΓT(πθ) = O(T). Standard entropy regularization: Without corridor enforcement, ΓT(π) = O(T2) in the worst case.

Without the corridor, entropy drift at turn t affects turn t + 1’s context, which affects turn t + 2—errors compound quadratically. EPO’s corridor acts as a guardrail, correcting deviations immediately. The advantage grows with dialogue length: 5.5× for T = 10, 10.5× for T = 20. The O(T2) worst-case arises under conditions common in LLM training: sparse end-of-dialogue rewards, entropy collapse dynamics, and autoregressive context dependence (Remark A.15).

Why Stability Improves Convergence. The O(T) vs O(T2) gap translates to convergence through two mechanisms: (i) gradient variance: entropy collapse causes extreme importance weights, increasing gradient noise; and (ii) trajectory diversity: collapsed entropy reduces the trajectories that reach informative rewards. We validate empirically that EPO achieves lower gradient variance and faster convergence (Appendix 5).

- Proposition 6.3 (Performance Bound). Under mild assumptions (Appendix A), EPO achieves:

V π

∗

−V π

θ

≤

|D|2ϵ2 2λ(1 − αβk)C

Optimization

+λH log |A|

|A∗H|1/H

Entropy Bias

+λβk∆smoothπ∗

Stability Cost

(7)

The bound decomposes into optimization error (decreases with training), entropy bias (standard in entropy-regularized RL), and stability cost (price for corridor enforcement). Crucially, this cost often vanishes:

- Proposition 6.4 (Zero Stability Cost). If π∗ has moderate

entropy (H(π∗|s) ∈ [κlH¯ref,κrH¯ref]), then ∆smoothπ∗ = 0EPO matches standard bounds while providing stability guarantees for free.

In sparse-reward multi-turn tasks, optimal policies typically require moderate entropy to discover good trajectories, naturally satisfying this condition. Thus EPO provides stability "for free" precisely when cascade failure threatens most.

### 7. Conclusion

In this work, we identified and addressed the explorationexploitation cascade failure, a fundamental challenge unique to training LLM agents in multi-turn environments with sparse rewards. Our proposed EPO framework addresses this through three synergistic mechanisms: trajectory-aware entropy computation, entropy smoothing regularization, and

adaptive phase-based weighting. Together, these components prevent the severe entropy oscillations that destabilize training under shared policy parameters. Empirical results demonstrate up to 152% performance improvement on ScienceWorld and 19.8% on ALFWorld, transforming previously unstable sparse-reward scenarios into smoothly converging optimization. This work establishes that multiturn LLM agent training requires fundamentally different entropy control than traditional RL, opening new directions for effective training methods for agentic LLMs.

### Impact Statement

This paper presents work whose goal is to advance the field of machine learning, specifically improving the training stability of LLM agents in multi-turn environments. Our method enables more reliable training of agents for beneficial applications such as scientific experimentation, household task assistance, and interactive problem-solving.

As with any advancement in autonomous agent capabilities, there exists potential for misuse if deployed without appropriate safeguards. However, our contribution is methodological in nature, focusing on training dynamics rather than enabling fundamentally new capabilities. We encourage practitioners to apply standard responsible AI practices when deploying agents trained with our method.

There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here beyond the considerations common to research on LLM agents.

### References

Anthropic. Claude code: Deep coding at terminal velocity, 2025. URL https://www.anthropic.com/ claude-code.

Bai, H., Zhou, Y., Pan, J., Cemri, M., Suhr, A., Levine, S., and Kumar, A. Digirl: Training in-the-wild devicecontrol agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37: 12461–12495, 2024.

Bai, H., Zhou, Y., Li, L. E., Levine, S., and Kumar, A. Digiq: Learning q-value functions for training device-control agents. arXiv preprint arXiv:2502.15760, 2025.

Cheng, D., Huang, S., Zhu, X., Dai, B., Zhao, W. X., Zhang, Z., and Wei, F. Reasoning with exploration: An entropy

- perspective. arXiv preprint arXiv:2506.14758, 2025a.

Cheng, D., Huang, S., Zhu, X., Dai, B., Zhao, W. X., Zhang, Z., and Wei, F. Reasoning with exploration: An entropy

- perspective. arXiv preprint arXiv:2506.14758, 2025b.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Deng, J., Chen, J., Chen, Z., Cheng, D., Bai, F., Zhang, B., Min, Y., Gao, Y., Zhao, W. X., and Wen, J.-R. From trial-and-error to improvement: A systematic analysis of llm exploration mechanisms in rlvr. arXiv preprint arXiv:2508.07534, 2025.

Dong, Y., Jiang, X., Tao, Y., Liu, H., Zhang, K., Mou, L., Cao, R., Ma, Y., Chen, J., Li, B., et al. Rl-plus: Countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization. arXiv preprint arXiv:2508.00222, 2025.

Feng, L., Xue, Z., Liu, T., and An, B. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. Pmlr, 2018.

He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., Zhang, X., Zhang, F., Xu, J., Shen, W., et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.

Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S. O., Wang, D., Zamani, H., and Han, J. Search-r1: Training LLMs to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum? id=Rwhi91ideu.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International conference on machine learning, pp. 1928– 1937. PmLR, 2016.

Nocedal, J. Numerical optimization. Springer Ser. Oper. Res. Financ. Eng./Springer, 2006.

OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world. 2025a. URL https://openai.com/ index/computer-using-agent.

OpenAI. Introducing codex, 2025b. URL https:// openai.com/index/introducing-codex/.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, S., Hong, L.,

Tian, R., Xie, R., Zhou, J., Gerstein, M., dahai li, Liu, Z., and Sun, M. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=dHng2O0Jjr.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shen, H. On entropy control in llm-rl algorithms. arXiv preprint arXiv:2509.03493, 2025.

Shridhar, M., Yuan, X., Côté, M.-A., Bisk, Y., Trischler,

- A., and Hausknecht, M. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2021. URL https://arxiv. org/abs/2010.03768.

Sutton, R. S. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44, 1988.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Team, T. D. Tongyi-deepresearch. https://github. com/Alibaba-NLP/DeepResearch, 2025.

- Wang, R., Jansen, P., Côté, M.-A., and Ammanabrolu, P. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279– 11298, 2022.
- Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025a.

Wang, Z., Wang, K., Wang, Q., Zhang, P., Li, L., Yang, Z., Jin, X., Yu, K., Nguyen, M. N., Liu, L., et al. Ragen: Understanding self-evolution in llm agents via multi-turn

reinforcement learning. arXiv preprint arXiv:2504.20073, 2025b.

Williams, R. J. and Peng, J. Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268, 1991.

Xi, Z., Ding, Y., Chen, W., Hong, B., Guo, H., Wang, J., Yang, D., Liao, C., Guo, X., He, W., et al. Agentgym: Evolving large language model-based agents across diverse environments. arXiv preprint arXiv:2406.04151, 2024.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Xu, W., Mei, K., Gao, H., Tan, J., Liang, Z., and Zhang, Y. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zeng, A., Lv, X., Zheng, Q., Hou, Z., Chen, B., Xie, C., Wang, C., Yin, D., Zeng, H., Zhang, J., et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Zhang, K., Li, J., Li, G., Shi, X., and Jin, Z. Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13643–13658, 2024.

Zhang, Z., Chen, Z., Li, M., Tu, Z., and Li, X. Rlvmr: Reinforcement learning with verifiable meta-reasoning rewards for robust long-horizon agents. arXiv preprint arXiv:2507.22844, 2025.

Zhou, Y., Zanette, A., Pan, J., Levine, S., and Kumar, A. ArCHer: Training language model agents via hierarchical multi-turn RL. In Forty-first International Conference on Machine Learning, 2024. URL https:

//openreview.net/forum?id=b6rA0kAHT1.

### Contents

- 1 Introduction 1
- 2 Related Work 2

- 2.1 Reinforcement Learning for LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- 2.2 Reinforcement Learning for LLM Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- 3 Preliminary 3

- 3.1 On-policy Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3.2 Problem formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- 4 Methodology 4

- 4.1 Entropy Regularization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 4.2 Entropy Smoothing Regularizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 5 Experiments 4

- 5.1 Experiments Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 5.2 Performance Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 5.3 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.4 Model Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 6 Theoretical Analysis 8
- 7 Conclusion 8

- A Theoretical Analysis: Full Proofs 13
- B Experiments 19
- C Core Implementation Pseudocode 27
- D System Prompts 28
- E Limitation and Future Work 30
- F Use of Large Language Models 30

### A. Theoretical Analysis: Full Proofs

#### A.1. Notation and Setup

We consider multi-turn dialogue generation as a finite-horizon MDP. Let s0 denote the initial query, H the maximum horizon (total tokens), and T the number of dialogue turns. Each turn t ∈ [T] generates Lt tokens. The policy πθ(a|s) maps states (conversation history) to distributions over tokens (actions).

Standard Entropy-Regularized Objective:

Vλπ(s0) = Eπ

H−1

rh + λH(π(·|sh)) s0 (8)

h=0

EPO Objective:

Vλ,βπ (s0) = Vλπ(s0) − λβk · LsmoothH¯ref (θ) (9) where the smoothing loss is:

α |B|

LsmoothH¯ref (θ) =

P(H(πθ(·|s));H¯ref) (10)

(s,a)∈B

with corridor penalty:

P(H;H¯ref) = [κlH¯ref − H]+ + [H − κrH¯ref]+ (11)

#### A.2. Assumptions

- Assumption A.1 (Fixed Entropy Reference). The smoothing loss is evaluated against a fixed reference H¯ref > 0, representing the target entropy level.
- Assumption A.2 (Entropy Corridor Constraint). The learned policy πθ maintains entropy within the corridor: κlH¯ref ≤ H(πθ(·|s)) ≤ κrH¯ref

for all reachable states s ∈ S(s0).

- Assumption A.3 (Policy Regularity). The policy gradient satisfies ∥∇V π

θ

λ,β(D)∥ ≤ ϵ and mina,s πθ(a|s) ≥ δπ > 0.

- Assumption A.4 (Bounded Smoothing Strength). The penalty weight satisfies αβmax < 1, where βmax = 1 + e0 = 2. Equivalently, α < 0.5.

Remark A.5 (Justification). Assumption A.2 captures EPO’s design intent—the smoothing penalty enforces corridor compliance. We verify this empirically in Appendix 5. Assumption A.4 ensures the EPO objective remains concave; our experiments use α = 0.1 < 0.5.

#### A.3. Key Definitions

- Definition A.6 (Stability Cost). For any policy π˜:

∆πsmooth˜ (s0) := Eπ˜ LsmoothH¯ref (θ˜) s0 ≥ 0 Under Assumption A.2, ∆smoothπ

θ

(s0) = 0.

- Definition A.7 (Concentrability Coefficient). For initial state s0 and policy πθ:

Cπ

θ

λ,β(s0) = Cd2 · min

h,s∈S(s0)

Pπ

θ

h (s) min

a

πθ(a|s)

2

· min

h,s∈S(s0)

Pπ

θ

h (s) Pπ

∗ λ,β

h (s|s0)

(12)

where Cd = (|A|H)−1/2 and Pπh(s) is the state visitation probability at step h.

- Definition A.8 (EPO-Optimal Policy).

πλ,β∗ := arg max π∈Π

Vλ,βπ (D) Under Assumption A.4, the objective is concave, so this is well-defined.

- Definition A.9 (Cumulative Entropy Deviation). For T-turn dialogue with average entropy H¯t at turn t:

T

|H¯t(π) − H¯ref|

ΓT(π) :=

t=1

#### A.4. Preliminary Lemmas

- Lemma A.10 (Penalty Function Properties). The corridor penalty P(H;H¯ref) = [κlH¯ref − H]+ + [H − κrH¯ref]+ satisfies:

- 1. Non-negativity: P ≥ 0
- 2. Corridor property: P = 0 if and only if H ∈ [κlH¯ref,κrH¯ref]
- 3. Lipschitz continuity: |P(H1) − P(H2)| ≤ |H1 − H2|

Proof. Properties (1) and (2) follow from the definition of [·]+ = max(0,·). For (3), both [κlH¯ref −H]+ and [H−κrH¯ref]+ are Lipschitz with constant 1, and at most one is active at any H.

| |
|---|

- Lemma A.11 (Gradient Towards Corridor). For entropy outside the corridor:

∇θLsmooth =

 



−α∇θH(πθ) if H < κlH¯ref (increases entropy) 0 if H ∈ [κlH¯ref,κrH¯ref]

+α∇θH(πθ) if H > κrH¯ref (decreases entropy)

Proof. Taking gradients of P(H;H¯ref):

∇θP =

 



−∇θH if H < κlH¯ref 0 if H ∈ [κlH¯ref,κrH¯ref]

+∇θH if H > κrH¯ref The result follows from Lsmooth = α · E[P].

| |
|---|

- Lemma A.12 (Performance Difference (Nocedal, 2006)). For any policies π,π′:

Vhπ(s) − V π

′

h (s) = Eπ

H−1

t=h

Aπ

′

t (st,at) sh = s

where Aπ

′

t (s,a) = Qπ

′

t (s,a) − V π

′

t (s) is the advantage function.

- Lemma A.13 (Entropy Bias (Shen, 2025)).

λ (s0) + λH log |A|

∗ λ

∗

(s0) ≤ V π

λ (s0) − V π

V π

(s0) − V π

θ

θ

|A∗H(s0)|1/H where |A∗H(s0)| is the number of optimal action sequences from s0.

#### A.5. Main Results

- A.5.1. EXACT PENALTY INTERPRETATION

- Proposition 6.1 (Exact Penalty Interpretation). The EPO objective equals Vλ,βπ = V π + λH(π) − ρk · P(π), where P(π) is the exact penalty for violating the entropy corridor [κlH¯ref,κrH¯ref], and ρk = λβkα is an adaptive penalty weight.

Proof. Expanding the EPO objective:

Vλ,βπ = V π + λH(π) − λβk · LsmoothH¯ref (13)

α |B|

P(H(π|s);H¯ref) (14)

= V π + λH(π) − λβk ·

(s,a)∈B

·Es∼π[P(H(π|s);H¯ref)]

= V π + λH(π) − λβkα

(15)

ρk

P(π)

This matches the exact penalty formulation for the constraint H(π|s) ∈ [κlH¯ref,κrH¯ref].

| |
|---|

- Remark A.14 (Connection to Classical Optimization). The exact penalty method (Nocedal, 2006) guarantees that for sufficiently large ρ, the unconstrained optimum coincides with the constrained optimum. EPO’s adaptive schedule ρk = λβkα provides curriculum-style enforcement: weaker early (exploration), stronger later (exploitation).

- A.5.2. MULTI-TURN STABILITY

- Proposition 6.2 (Multi-Turn Entropy Stability). Define cumulative entropy deviation ΓT(π) := Tt=1 |H¯t(π)−H¯ref| over T turns. EPO: Under the corridor constraint, ΓT(πθ) = O(T). Standard entropy regularization: Without corridor enforcement, ΓT(π) = O(T2) in the worst case.

Proof. Part (i): EPO bound. Under Assumption A.2, for all turns t and states s:

The average entropy at turn t satisfies:

H(πθ(·|s)) ∈ [κlH¯ref,κrH¯ref]

1 Lt h∈turn tH(πθ(·|sh)) ∈ [κlH¯ref,κrH¯ref]

H¯t(πθ) =

Therefore:

κr − κl

2 · H¯ref := ∆max Summing over T turns:

|H¯t(πθ) − H¯ref| ≤ max{|1 − κl|,|κr − 1|} · H¯ref ≤

T

|H¯t(πθ) − H¯ref| ≤ T · ∆max = O(T)

ΓT(πθ) =

t=1

#### Part (ii): Entropy regularization worst case.

Without corridor enforcement, entropy at turn t depends on context from turns 1,...,t − 1. Consider the cascade failure scenario where each turn introduces drift δ:

t

H¯t − H¯ref =

δs

s=1

If drifts are systematic (e.g., consistently decreasing entropy), |H¯t − H¯ref| ≤ t · δmax. Summing:

T

T(T + 1) 2 · δmax = O(T2)

ΓT(π) ≤

t · δmax =

t=1

| |
|---|

- Remark A.15 (Conditions for Cascade Failure). The O(T2) worst-case arises under specific but common conditions in multi-turn LLM training:

- (i) Sparse rewards: When rewards are only observed at dialogue end (e.g., task success/failure), there is no per-turn gradient signal to correct entropy drift early. Each turn’s drift goes unchecked until final reward, by which point deviations have compounded.
- (ii) Entropy collapse dynamics: Once entropy begins decreasing, the softmax policy assigns increasingly extreme probabilities. This reduces exploration, which reduces trajectory diversity, which reduces the probability of discovering reward signal, which further reduces entropy—a positive feedback loop.
- (iii) Context-dependent amplification: In autoregressive generation, turn t’s output becomes turn t + 1’s context. Lowentropy (confident) outputs at turn t bias the model toward similar low-entropy behavior at turn t + 1, causing drift to accumulate rather than cancel.

These conditions are prevalent in multi-turn LLM training, so the O(T2) bound captures realistic failure modes rather than pathological edge cases.

- Remark A.16 (Practical Implication). The O(T) vs O(T2) gap implies EPO’s advantage grows with dialogue length. For T = 10 turns, the worst-case ratio is T(T+1)T /2 = T+12 = 5.5×. For T = 20, this becomes 10.5×. This explains why EPO’s empirical benefits are most pronounced in long multi-turn tasks (Appendix 5).

A.5.3. CONNECTION BETWEEN ENTROPY STABILITY AND CONVERGENCE

- Remark A.17 (Why Entropy Stability Improves Convergence). While we do not formally prove a bound relating ΓT to suboptimality, we identify two mechanisms through which entropy stability improves convergence:

- Mechanism 1: Importance Weight Variance. Policy gradient estimators use importance weights w = πθ(a|s)/πold(a|s). When entropy collapses (H → 0), policies become near-deterministic, causing w → ∞ for some actions and w → 0 for others. This increases gradient variance, slowing convergence. EPO’s corridor constraint bounds entropy away from zero, limiting weight variance.
- Mechanism 2: Trajectory Diversity. In multi-turn generation with sparse rewards, gradient signal comes from trajectories that reach reward. Collapsed entropy reduces trajectory diversity exponentially with horizon: if per-step entropy drops by factor c < 1, trajectory diversity drops by cT. The corridor constraint maintains per-step entropy, preserving trajectory diversity throughout training.

Empirical Validation. Rather than formalizing these mechanisms into a bound with unverifiable constants, we directly measure the effect in Appendix 5: EPO achieves lower gradient variance and faster convergence compared to standard entropy regularization, consistent with this intuition.

- A.5.4. PERFORMANCE BOUND

- Proposition 6.3 (Performance Bound). Under mild assumptions (Appendix A), EPO achieves:

|D|2ϵ2 2λ(1 − αβk)C

+λH log |A|

∗

+λβk∆smoothπ∗

V π

− V π

≤

θ

|A∗H|1/H

Stability Cost

Optimization

Entropy Bias

(7)

#### Proof. Step 1: Decomposition through EPO-optimal policy.

∗ λ,β

− V π

∗

∗

V π

(s0) − V π

(s0) = [V π

λ,β ] (I)

θ

∗ λ,β

+[V π

λ,β − V π

+[V π

λ,β − V π

λ,β] (II)

]

θ

θ

θ

(III)

(16)

- Step 2: Bounding Term (II). Under Assumption A.4, the EPO objective has effective entropy coefficient:

##### λeff = λ(1 − αβk) > 0

Since αβk < 1, the objective Vλ,β remains concave in policy parameters. The gradient-to-performance bound from Shen

(2025) applies:

|D|2 2λeff · Cπ

ϵ2 = |D|2ϵ2 2λ(1 − αβk)Cπ

(17)

(II) ≤

λ,β(s0)

λ,β(s0)

θ

θ

- Step 3: Bounding Term (III). Under Assumption A.2, Lsmooth(θ) = 0 by Lemma A.10(2). Thus:

(III) = V π

θ

λ,β(s0) − V π

θ

(s0) = λH(πθ|s0) − 0 = λH(πθ|s0) (18)

- Step 4: Bounding Term (I).

By definition of πλ,β∗ as the maximizer:

V π

∗ λ,β

λ,β (s0) ≥ V π

∗

λ,β(s0) Expanding V π

∗

λ,β:

V π

∗

λ,β(s0) = V π

∗

(s0) + λH(π∗|s0) − λβk∆smoothπ∗ (s0) Therefore:

(I) = V π

∗

− V π

∗ λ,β

λ,β ≤ V π

∗

− V π

∗

λ,β = −λH(π∗|s0) + λβk∆smoothπ∗ (s0) (19)

- Step 5: Combining (I) + (III).

(I) + (III) ≤ λH(πθ|s0) − λH(π∗|s0) + λβk∆smoothπ∗ (s0) (20) = λ(H(πθ|s0) − H(π∗|s0)) + λβk∆smoothπ∗ (s0) (21)

Using entropy bounds: H(πθ|s0) ≤ H log |A| and H(π∗|s0) ≥ log |A∗H(s0)|:

(I) + (III) ≤ λH log |A| |A∗H(s0)|1/H

+ λβk∆smoothπ∗ (s0) (22)

- Step 6: Final bound. Combining Steps 2 and 5:

|D|2ϵ2 2λ(1 − αβk)Cπ

+ λH log |A| |A∗H(s0)|1/H

∗

+ λβk∆smoothπ∗ (s0) (23)

V π

##### (s0) − V π

(s0) ≤

θ

λ,β(s0)

θ

| |
|---|

- A.5.5. ZERO STABILITY COST CONDITION

- Proposition 6.4 (Zero Stability Cost). If π∗ has moderate entropy (H(π∗|s) ∈ [κlH¯ref,κrH¯ref]), then ∆smoothπ∗ = 0—EPO matches standard bounds while providing stability guarantees for free.

Proof. Part 1: Zero stability cost. Under the stated condition, H(π∗(·|s)) ∈ [κlH¯ref,κrH¯ref] for all s ∈ S(s0). By Lemma A.10(2), P(H(π∗|s);H¯ref) = 0 for all such states. Therefore:

LsmoothH¯ref (θ∗) = α · Es∼π∗[P(H(π∗|s);H¯ref)] = 0 And:

∆smoothπ∗ (s0) = Eπ∗[Lsmooth|s0] = 0

- Part 2: Matching bound. Substituting ∆smoothπ∗ = 0 into Proposition 6.3:

V π

∗

− V π

θ

≤

|D|2ϵ2 2λ(1 − αβk)C

+ λH log |A| |A∗H|1/H

For small α (e.g., α = 0.1), the factor (1−αβk)−1 ≈ 1.25, so the optimization term is only slightly larger than the standard entropy-regularized bound.

- Part 3: Additional guarantees.

Even with ∆smoothπ∗ = 0, EPO provides stability via Lemma A.11: deviations from the corridor are immediately corrected, preventing cascade failure (Proposition 6.2).

| |
|---|

- Remark A.18 (When Does the Condition Hold?). The condition H(π∗|s) ∈ [κlH¯ref,κrH¯ref] holds when:

- 1. The task admits multiple near-optimal solutions, requiring stochastic π∗
- 2. The corridor [κl,κr] is calibrated to include typical optimal entropy levels
- 3. Multi-turn tasks where exploration remains valuable throughout the dialogue

In sparse-reward multi-turn settings, deterministic policies often fail to discover good trajectories, so optimal policies naturally maintain moderate entropy.

- Remark A.19 (Corridor Width Trade-off). The corridor bounds [κl,κr] control a precision-stability trade-off: Narrow corridor (κr − κl small):

- • Strong entropy enforcement—deviations corrected quickly
- • Higher stability cost if π∗ lies outside corridor
- • Risk: over-constraining may prevent learning optimal behavior

#### Wide corridor (κr − κl large):

- • Weak entropy enforcement—allows more variation
- • Lower stability cost—most policies satisfy constraint
- • Risk: may not prevent cascade failure if corridor is too permissive

Practical guidance: Set [κl,κr] to contain the entropy range of good policies while excluding collapse/explosion regimes. Based on experiments across multiple tasks (Appendix 5.3), we find κl = 0.5,κr = 1.5 (allowing entropy to vary by ±50% from reference) provides a robust default.

Sensitivity: Let w = κr − κl denote corridor width. The stability cost scales as:

∆smoothπ∗ ∝ max 0, |H(π∗) − H¯ref| H¯ref

w 2

−

Widening the corridor by δ reduces stability cost by O(δ) when π∗ is near the boundary.

- A.6. Comparison with Standard Entropy Regularization Corollary A.20 (EPO vs Standard Entropy Regularization). The standard entropy-regularized bound (Shen, 2025):

|D|2ϵ2 2λCπ

+ λH log |A|

∗

V π

− V π

≤

θ

|A∗H|1/H EPO’s bound differs by:

θ

λ

- 1. A factor of (1 − αβk)−1 ∈ [1.11,1.25] in optimization error (for α = 0.1)
- 2. The stability cost λβk∆smoothπ∗ (zero under Proposition 6.4)

Remark A.21 (Quantifying Multi-Turn Advantage). For the multi-turn stability advantage (Proposition 6.2) to outweigh EPO’s per-step overhead, we require:

O(T2) O(T)

1 1 − αβk

= O(T) >

With α = 0.1 and βk ∈ [1,2], the right-hand side is at most 1.25. Thus for T ≥ 2, the multi-turn stability advantage dominates the per-step overhead. For typical dialogue lengths (T ≥ 5), the advantage is substantial: 5× to 10× reduction in cumulative entropy deviation.

### B. Experiments

#### B.1. Detailed Experiment Setup

- B.1.1. BENCHMARK

We evaluate our method on two challenging and complementary benchmarks, ScienceWorld and ALFWorld, which test distinct yet crucial reasoning capabilities.

ScienceWorld (Wang et al., 2022) is a dynamic, text-based environment simulating a grade-school science lab, where the agent must solve open-ended scientific tasks. Success demands systematic hypothesis testing, common-sense reasoning about object properties, and a deep understanding of cause and effect. Its curriculum is divided into over 30 task types sourced from official study guides, primarily spanning: Physics, with tasks such as powering electrical circuits, testing the conductivity of materials, and manipulating states of matter; Chemistry, including identifying acids and bases or observing chemical reactions; and Life Science, which involves classifying organisms based on their traits. These tasks rigorously test an agent’s abstract knowledge and procedural reasoning.

In contrast to the abstract challenges in ScienceWorld, ALFWorld (Shridhar et al., 2021) tests embodied reasoning in a visually-rich environment. It requires an agent to interpret high-level natural language instructions and decompose them into long sequences of low-level actions within simulated household settings. The benchmark is structured around seven main task categories designed to test long-horizon planning and language grounding: (1) Pick & Place, for simple object relocation (e.g., "Put a mug in the coffee maker"); (2) Pick Two & Place, for handling multiple objects; (3) Pick & Cool/Heat, requiring state changes using appliances; (4) Pick & Clean, involving interaction with sinks; and more complex compositional tasks like (5) Look At Obj & Pick and (6) Examine In Light. Success in ALFWorld hinges on multi-step task planning, spatial awareness, and the ability to ground language in a physical context.

Together, these two benchmarks provide a comprehensive testbed for our agent’s capabilities, spanning from abstract knowledge application in ScienceWorld to embodied task execution in ALFWorld.

- B.1.2. EVALUATION SETTING

We employ dual success rate metrics to capture different aspects of performance: Succ.∗ reports the average of maximum success rates across random seeds, while Succ. measures average performance after convergence, reflecting practical reliability. To calculate the average converged success rate (Succ.), we first identify a convergence period where performance stabilizes.

In the ALFWorldenvironment, we observed that all methods exhibit similar convergence trends, with success rates plateauing after step 125. Therefore, we compute Succ. by averaging the success rates from step 125 onward (inclusive) across three random seeds. In contrast, the ScienceWorldenvironment exhibited more varied convergence behaviors across different random seeds, necessitating a per-run analysis. Specifically, the epoch ranges for computing the converged success rate in the ScienceWorldenvironment were determined as follows: In our comparison with GRPO, the evaluation windows for the three random seeds of the GRPO baseline were set to epochs 70-120, 90-120, and 90-120. In stark contrast, for our method (GRPO with EPO), these windows began significantly earlier, spanning epochs 60-80, 80-120, and 25-120, respectively. A similar trend was observed in the PPO comparison. The PPO baseline’s convergence was identified late in training, with windows of 100-120, 105-120, and 105-120. When enhanced with EPO, the model converged much faster, with its evaluation periods set to 70-120, 60-120, and 60-120 for the three seeds. This detailed breakdown confirms that EPO consistently accelerates convergence across different algorithms and seeds.

Given the high variance inherent in RL, final performance scores alone can be misleading. We therefore present averaged curves to provide a more robust comparison and illustrate the performance evolution throughout the training process. We apply wandb’s default running average (window size of 10) to smooth all training curves. This standard practice avoids visualization-specific tuning and ensures a fair comparison of the underlying learning trends. Additionally, we scale the variance by a factor of 0.8 for better visual clarity.

- B.1.3. BASELINES

We conduct comprehensive comparisons across multiple paradigms to evaluate the effectiveness of our proposed EPO methodology. These baselines are grouped into four categories, each representing a distinct approach to training large language model (LLM) agents.

Prompting-based Approaches This paradigm focuses on leveraging the in-context learning capabilities of LLMs without any parameter optimization.

The ReAct framework (Yao et al., 2023) synergizes reasoning and acting in language models. Its core innovation is the interleaving of textual reasoning traces with actions that interact with an external environment. Unlike prior methods that treated reasoning and acting as separate processes, ReAct allows the model to create and adjust high-level plans while grounding them in reality by gathering information from the environment. As a prompting-based method, ReAct’s performance relies on the quality of the in-context examples and the inherent capabilities of the base model. Its operational scope is confined to single-pass inference, without mechanisms for parameter optimization or learning from experiences across multiple episodes to discover novel policies.

Trajectory-based and Platform Methods This category includes methods that rely on imitating expert trajectories and platforms designed for agent development.

SFT is a fundamental approach for adapting pre-trained language models to specific tasks by imitating expert-provided trajectories. The effectiveness of SFT is contingent upon the availability of a comprehensive dataset of high-quality expert demonstrations. The resulting agent’s policy is inherently bounded by the scope of behaviors observed within this dataset, constraining its ability to explore novel strategies beyond the demonstrated examples.

AgentGym (Xi et al., 2024) is a comprehensive framework for building and evaluating generalist LLM-based agents, introducing a self-evolution method, AgentEvol. While AgentGym provides a valuable framework for evaluation, its AgentEvol method operates through a form of behavioral cloning. The evolution of the agent is thus guided by the quality and diversity of the initial trajectory data, which influences its sample efficiency in exploring the environment.

General Reinforcement Learning Approaches This group consists of well-established reinforcement learning algorithms that are not specifically designed for LLM agents but are widely used in the field.

PPO (Schulman et al., 2017) is a state-of-the-art on-policy reinforcement learning algorithm known for its stability and ease of implementation. It uses a clipped surrogate objective function to constrain policy updates. When applied to multi-turn, sparse-reward environments, credit assignment in standard PPO is performed based on the terminal reward signal. This structure can present challenges in distributing credit across a long sequence of actions, potentially leading to instabilities such as the “entropy oscillation” phenomenon.

GRPO (Shao et al., 2024) is a critic-free reinforcement learning algorithm. Instead of relying on a learned value function, it compares the performance of a group of trajectories generated from the same initial state. GRPO performs credit assignment

at the trajectory level, evaluating the collective outcome of an entire episode. This design provides a holistic signal for policy updates, rather than turn-by-turn feedback, which is a consideration for learning complex, multi-step tasks.

Agent RL Approaches This category includes recent reinforcement learning methods that are specifically designed for training LLM agents.

GIGPO (Feng et al., 2025) extends GRPO by introducing a two-level hierarchical structure for advantage estimation. It refines the trajectory-level credit assignment of GRPO by introducing a micro-level analysis based on “anchor states.” The utility of this mechanism is related to the frequency with which these anchor states are revisited, and learning is guided by the single reward signal provided at the conclusion of each episode.

RLVMR (Zhang et al., 2025) is a framework that integrates dense, process-level supervision into the reinforcement learning loop by rewarding verifiable, meta-reasoning behaviors. RLVMR shapes the agent’s behavior by leveraging a “teacher” model (e.g., GPT-4) to annotate expert trajectories with meta-reasoning tags. The learning process is thus guided by the quality and potential biases inherent in these teacher-provided annotations.

- B.1.4. IMPLEMENTATION DETAILS

All of our experiments were conducted on a single server node equipped with eight NVIDIA H100 or A100 GPUs to ensure consistent and reproducible results. The training times varied based on the environment’s complexity, the RL algorithm, and the GPU architecture. On the computationally intensive SciWorld benchmark, a full PPO training run required approximately 23 hours on H100s and 31 hours on A100s. The more efficient GRPO baseline was faster, completing in 16 hours on H100s and 20 hours on A100s. For the ALFWorld environment, the PPO baseline took 23 hours with H100s and 31 hours with A100s, while the GRPO baseline required 18 hours and 25 hours, respectively. Crucially, our proposed EPO method is designed for efficiency and introduces negligible computational overhead. The training times for our EPO-enhanced variants (PPO+EPO and GRPO+EPO) are effectively identical to their corresponding PPO and GRPO baselines under the same hardware configuration. The detailed hyperparameter configurations for each setup are presented in Table 3 and Table 2.

Table 2. Hyperparameter settings for the SciWorld environment. Hyperparameter PPO GRPO PPO+EPO GRPO+EPO General Setup

Foundation Model Qwen2.5-7B-Instruct Total RL Steps (K) 125 Max Prompt Length 2048 Max Response Length 512 Test Frequency (steps) 5

Optimizer & LR Scheduler

Optimizer AdamW LR Scheduler Cosine (num_cycles = 0.5) Learning Rate 3 × 10−6 5 × 10−6 3 × 10−6 5 × 10−6 Warmup / Min LR Ratio 0.1 / 0.2

Batch Sizes & PPO-Specific Setup

Mini-batch Size 64 128 64 128 Micro-batch Size 8 16 8 16 Critic Micro-batch Size 4 — 4 —

EPO

λ — — 0.001 κl, κr — — 0.0, 2.0 γ — — 3.0 λk — — 0.05

#### B.2. Performance Comparison

- Figure 5 demonstrates the training dynamics and generalization performance of our EPO enhancement across two environments and algorithms. Our EPO variants exhibit superior convergence characteristics in both settings. In ScienceWorld,

Table 3. Hyperparameter settings for the ALFWorld environment. Hyperparameter PPO GRPO PPO+EPO GRPO+EPO General Setup

Foundation Model Qwen2.5-3B-Instruct Total RL Steps (K) 150 Max Prompt Length 2048 Max Response Length 512 Test Frequency (steps) 5

Optimizer & LR Scheduler

Optimizer AdamW LR Scheduler Cosine (num_cycles = 0.5) Learning Rate 5 × 10−6 Warmup / Min LR Ratio 0.1 / 0.2

Batch Sizes & PPO-Specific Setup

Mini-batch Size 256 — 256 Micro-batch Size 32 — 32 Critic Micro-batch Size 16 — 16 —

EPO

λ — — 0.001 κl, κr — — 0.0, 2.0 γ — — 3.0 λk — — 0.1

GRPO+EPO achieves early convergence around step 60 with higher peak rewards than baseline GRPO ( Figure 5(a)). Similarly, PPO+EPO in ALFWorld maintains more consistent reward accumulation with reduced oscillations ( Figure 5(b)). This improved stability stems from EPO’s entropy regularization guiding exploration toward productive policy regions.

For ScienceWorld, GRPO+EPO demonstrates clear advantages across both IID and OOD settings, achieving success rates exceeding 0.8 within 40 steps while baseline GRPO struggles to surpass 0.6 ( Figure 5(c),(e)). In ALFWorld, PPO+EPO prioritizes OOD robustness over IID performance. While showing comparable IID results ( Figure 5(d)), PPO+EPO maintains consistent OOD success rates above 0.6 compared to baseline PPO’s frequent drops below 0.4 ( Figure 5(f)).

The key advantage of EPO lies in variance reduction and elimination of training oscillations. Across both environments, EPO variants show tighter confidence intervals and more reliable convergence patterns. This stability particularly benefits OOD scenarios where baseline methods exhibit substantial performance degradation. These results validate our entropy regularization approach for addressing exploration-exploitation challenges in multi-turn LLM agent training, demonstrating simultaneous improvements in generalization capability and convergence reliability.

#### B.3. Ablation Study

- Figure 6 extends our ablation analysis to both GRPO and PPO variants across ScienceWorld and ALFWorld. In ScienceWorld (a,c,e), the entropy smoothing regularizer proves essential: GRPO+EPO-Base exhibits severely delayed learning with rewards remaining near 2 until step 40 and success rates plateauing at 0.6, while GRPO+EPO achieves immediate progress reaching 7-8 rewards and 0.7-0.85 success rates with a 40-50% relative improvement that persists throughout training. This pattern holds across both GRPO and PPO, confirming the mechanism’s algorithm-agnostic benefits. ALFWorld (b,d,f) shows markedly different dynamics: both PPO variants converge to similar final performance ( 12.5 reward, 0.8 success rate), with PPO+EPO primarily demonstrating 20-episode faster convergence. This differential impact validates our theoretical framework—ScienceWorld’s extreme sparsity (30+ actions before feedback) creates pathological exploration-exploitation oscillations that the smoothing regularizer effectively breaks by maintaining entropy within historical bounds. ALFWorld’s structured feedback naturally prevents such oscillations, making smoothing beneficial for speed but not essential for convergence.

The consistent improvements across both GRPO and PPO in sparse settings confirm that entropy smoothing addresses a fundamental challenge in multi-turn optimization rather than algorithm-specific weaknesses. The adaptive βk weighting

[Figure 19]

[Figure 20]

(a) Training Rewards (ScienceWorld) (b) Training Rewards (ALFWorld)

[Figure 21]

[Figure 22]

(c) IID Success Rate (ScienceWorld) (d) IID Success Rate (ALFWorld)

[Figure 23]

[Figure 24]

(e) OOD Success Rate (ScienceWorld) (f) OOD Success Rate (ALFWorld)

- Figure 5. Training dynamics and generalization performance analysis. We present the evolution of training rewards and validation success rates across both in-distribution (IID) and out-of-distribution (OOD) evaluation settings. (a,c,e) ScienceWorld experimental results contrasting GRPO and GRPO+EPO performance across training reward accumulation, IID validation, and OOD validation metrics. (b,d,f) ALFWorld experimental results contrasting PPO and PPO+EPO under identical evaluation criteria. Our EPO enhancement exhibits significantly improved training stability and substantial performance gains across both IID and OOD evaluation scenarios against baseline methods.

enables this by providing strong stabilization when oscillations are detected while relaxing constraints during natural convergence, transforming intractable sparse reward problems into smoothly converging optimization processes.

#### B.4. Model Study

[Figure 25]

[Figure 26]

[Figure 27]

(a) Training Rewards (ScienceWorld) (b) Training Rewards (ALFWorld) (c) IID Success Rate (ScienceWorld)

[Figure 28]

[Figure 29]

[Figure 30]

(d) IID Success Rate (ALFWorld) - PPO (e) OOD Success Rate (ScienceWorld) - PPO (f) OOD Success Rate (ALFWorld) - PPO

[Figure 31]

[Figure 32]

[Figure 33]

(g) Training Rewards (ALFWorld) - GRPO (h) IID Success Rate (ALFWorld) - GRPO (i) OOD Success Rate (ALFWorld) - GRPO

- Figure 6. Impact of the entropy smoothing regularizer on training dynamics and performance. This ablation study contrasts our full method (EPO) with a variant that excludes the entropy smoothing regularizer (EPO-base). The comparison on ScienceWorld (a,c,e) and ALFWorld (b,d,f) demonstrates that the smoothing mechanism is essential for stable RL training progression.

- B.4.1. STUDY OF ENTROPY REGULARIZATION

To analyze the exploration-exploitation trade-off, we compare our standard method, PPO+EPO-Base, which applies a consistent entropy regularization coefficient throughout training, against an experimental variant, PPO+EPO-Decay. This variant was designed to test the hypothesis that a dynamic schedule could improve performance. It employs a formula to modulate the entropy weight over time: assigning a higher weight during initial training phases to promote exploration, and systematically reducing the weight in later phases to encourage exploitation.

Contrary to our hypothesis, the empirical results in Figure 7 show this strategy is counterproductive. The PPO+EPO-Decay variant consistently underperforms the baseline across all metrics, including episodic reward (a), in-distribution success rate (b), and out-of-distribution success rate (c).

Panel (d) provides insight into this failure by analyzing the intra-episode entropy, comparing the average entropy of the first 10 tokens (“Early Steps”) with the last 10 tokens (“Late Steps”). While the decay schedule successfully reduces the policy’s entropy in the later stages of training, it does so at a significant cost. The schedule prematurely suppresses exploration in the crucial initial turns of each episode. This insufficient early exploration locks the agent into suboptimal strategies from which it cannot recover, even as the policy becomes more deterministic. This finding underscores that for complex, multi-turn tasks, maintaining a robust and consistent exploration pressure is more effective than manually scheduling a transition towards exploitation.

- B.4.2. STUDY OF ENTROPY-SHAPED ADVANTAGE

We compare our Entropy-smoothed Policy Optimization (EPO) with the Entropy-based Advantage (EA) shaping method from Cheng et al. (Cheng et al., 2025b). As shown in Figure 8, while PPO+EA improves over the baseline, our PPO+EPO

[Figure 34]

[Figure 35]

(a) Training Rewards (ScienceWorld) (b) IID Success Rate (ScienceWorld)

[Figure 36]

[Figure 37]

(c) OOD Success Rate (ScienceWorld) (d) Entropy Comparison (ScienceWorld)

- Figure 7. Performance comparison of our standard PPO+EPO-Base against PPO+EPO-Decay, which uses a decaying entropy coefficient, on the ScienceWorld benchmark. Panels (a-c) demonstrate that the dynamic decay schedule consistently degrades performance across episodic rewards and success rates. Panel (d) analyzes the intra-episode entropy for early versus late tokens, revealing that the decay schedule prematurely suppresses crucial early-turn exploration, which negatively impacts overall performance.

is substantially superior in both final performance and convergence speed.

The primary difference lies in the gradient signal. The EA method uses a detached entropy term , which acts as an indirect intrinsic reward rather than a direct, optimizable objective. Consequently, the policy receives no gradient signal to explicitly increase its entropy. In contrast, our EPO formulation integrates entropy directly into the policy loss, enabling a direct gradient ∇θLH(θ) to explicitly guide the policy towards more exploratory behavior. Furthermore, EA’s hard clipping on the advantage bonus can induce training instability, and its myopic nature considers only instantaneous entropy. Our EPO method promotes smoother and more consistent updates by using a continuous smoothing regularizer that leverages a historical entropy window. This temporal consistency is critical for long-horizon reasoning tasks.

These theoretical advantages explain the empirical gap: PPO+EPO converges to a near-optimal success rate of almost 1.0, while PPO+EA plateaus far lower at 0.5-0.6. We posit that EA’s direct advantage modification distorts the credit assignment process. In contrast, EPO’s decoupled regularization preserves the integrity of the value signal, leading to more robust and effective learning.

[Figure 38]

[Figure 39]

(a) ScienceWorld (b) ScienceWorld

[Figure 40]

[Figure 41]

(c) ScienceWorld (d) ScienceWorld

[Figure 42]

[Figure 43]

(e) ScienceWorld (f) ScienceWorld

- Figure 8. Performance comparison on ScienceWorld environment: vanilla PPO, PPO with Entropy-based Advantage shaping (PPO+EA) from Cheng et al. (Cheng et al., 2025b), and our PPO with Entropy-smoothed Policy Optimization (PPO+EPO). Results show episodic rewards (a,b), validation IID success rates (c,d), and OOD success rates (e,f). PPO+EPO consistently outperforms both baselines, achieving near-perfect success rates (∼1.0) compared to PPO+EA’s plateau at 0.5-0.6. Curves show mean values with shaded standard error across multiple seeds.

### C. Core Implementation Pseudocode

Algorithm 2 presents a PyTorch-style pseudocode outlining the core computational steps for our proposed Entropy-Smoothed Policy Optimization (EPO) loss. This implementation directly corresponds to the methodology described in Section 3, detailing how the base policy loss, entropy regularization, and the entropy smoothing regularizer are combined to form the final training objective for a single policy update step.

Algorithm 2 PyTorch-style Pseudocode for EPO Loss Calculation

- 1 # Given: policy ‘policy_pi‘, batch ‘data‘, current epoch ‘k‘

- 2 # ‘data‘ contains: old_log_probs, advantages, response_mask, entropy_history

- 3 # Hyperparameters: lambda_, kappa_l, kappa_r, alpha, K

- 4

- 5 # 1. Forward pass for current log probabilities and token-level entropy

- 6 logits = policy_pi(data.input_ids, data.attention_mask)

- 7 log_prob, entropy = get_logprob_and_entropy(logits, data.responses)

- 8

- 9 # 2. Compute the base multi-turn policy loss L^MT (e.g., PPO objective)

- 10 pg_loss = compute_policy_loss(

- 11 log_prob=log_prob,

- 12 old_log_prob=data.old_log_probs,

- 13 advantages=data.advantages,

- 14 response_mask=data.response_mask,

- 15 clip_ratio=0.2

- 16 )

- 17

- 18 # 3. Compute the entropy regularization loss L^H (Eq. 6)

- 19 entropy_loss = agg_loss(entropy, data.response_mask)

- 20

- 21 # 4. Compute the entropy smoothing regularizer

- 22 # 4a. Calculate historical average entropy from window W_k

- 23 historical_avg_entropy = data.entropy_history.mean()

- 24

- 25 # 4b. Generate token-wise penalty mask P based on historical avg (Eq. 8)

- 26 penalty_mask = generate_entropy_penalty(

- 27 current_entropy=entropy,

- 28 historical_avg_entropy=historical_avg_entropy,

- 29 min_ratio=kappa_l, max_ratio=kappa_r, penalty_weight=alpha

- 30 )

- 31

- 32 # 4c. Calculate smoothing loss L^smooth by aggregating penalties (Eq. 9)

- 33 smoothing_loss = agg_loss(penalty_mask, data.response_mask)

- 34

- 35 # 4d. Get dynamic coefficient beta_k for the current step k (Eq. 11)

- 36 beta_k = calculate_dynamic_beta(current_step=k, total_steps=K)

- 37

- 38 # 5. Combine entropy and smoothing terms

- 39 # Corresponds to [L^H(theta) - beta_k * L^smooth(theta)]

- 40 entropy_term = entropy_loss - beta_k * smoothing_loss

- 41

- 42 # 6. Compute the final EPO loss (Eq. 10)

- 43 # L^EPO = L^MT - lambda * [L^H - beta_k * L^smooth]

- 44 final_loss = pg_loss - lambda_ * entropy_term

- 45

### D. System Prompts

This appendix details the system prompts used to guide the language model agents in the ALFWorld and ScienceWorld environments. For each environment, we provide two versions of the prompt: one that includes historical context (previous actions and observations) and one that omits it for the initial turn. The placeholders in curly braces, such as

{current_observation}, are dynamically replaced with environment-specific information at runtime.

- D.1. ALFWorld Prompts Listing 1. ALFWorld prompt (without history)

You are an expert agent operating in the ALFRED Embodied Environment. Your current observation is: {current_observation} Your admissible actions of the current situation are: [{admissible_actions}].

Now it’s your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process

→ MUST be enclosed within <think> </think> tags.

Once you’ve finished your reasoning, you should choose an admissible action for current

→ step and present it within <action> </action> tags.

Listing 2. ALFWorld prompt (with history)

You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {

→ task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most

→ recent {history_length} observations and the corresponding actions you took: {

→ action_history} You are now at step {current_step} and your current observation is: {current_observation} Your admissible actions of the current situation are: [{admissible_actions}].

Now it’s your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process

→ MUST be enclosed within <think> </think> tags.

Once you’ve finished your reasoning, you should choose an admissible action for current

→ step and present it within <action> </action> tags.

- D.2. ScienceWorld Prompts Listing 3. ScienceWorld prompt (without history)

You are an expert agent operating in the ScienceWorld environment, which is a text-based

→ virtual environment centered around accomplishing tasks from the elementary

→ science curriculum. Your current task is: {task_description} Your current observation is: {current_observation} Here are the actions you may take: [ {"action": "open OBJ", "description": "open a container"}, {"action": "close OBJ", "description": "close a container"}, {"action": "activate OBJ", "description": "activate a device"}, {"action": "deactivate OBJ", "description": "deactivate a device"}, {"action": "connect OBJ to OBJ", "description": "connect electrical components"}, {"action": "disconnect OBJ", "description": "disconnect electrical components"}, {"action": "use OBJ [on OBJ]", "description": "use a device/item"}, {"action": "look around", "description": "describe the current room"}, {"action": "look at OBJ", "description": "describe an object in detail"},

{"action": "look in OBJ", "description": "describe a container’s contents"}, {"action": "read OBJ", "description": "read a note or book"}, {"action": "move OBJ to OBJ", "description": "move an object to a container"}, {"action": "pick up OBJ", "description": "move an object to the inventory"}, {"action": "put down OBJ", "description": "drop an inventory item"}, {"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}, {"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}, {"action": "mix OBJ", "description": "chemically mix a container"}, {"action": "go to LOC", "description": "move to a new location"}, {"action": "eat OBJ", "description": "eat a food"}, {"action": "flush OBJ", "description": "flush a toilet"}, {"action": "focus on OBJ", "description": "signal intent on a task object"}, {"action": "wait", "description": "take no action for 10 iterations"}, {"action": "wait1", "description": "take no action for 1 iteration"}, {"action": "task", "description": "describe current task"}, {"action": "inventory", "description": "list your inventory"} ] Current available actions: {available_actions}

Now it’s your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process

→ MUST be enclosed within <think> </think> tags.

Once you’ve finished your reasoning, you should choose an appropriate action for the

→ current step and present it within <action> </action> tags.

Listing 4. ScienceWorld prompt (with history)

You are an expert agent operating in the ScienceWorld environment, which is a text-based

→ virtual environment centered around accomplishing tasks from the elementary

→ science curriculum. Your current task is: {task_description} Prior to this step, you have already taken {step_count} step(s). Below are the most

→ recent {history_length} observations and the corresponding actions you took: {

→ action_history} You are now at step {current_step} and your current observation is: {current_observation} Here are the actions you may take: [ {"action": "open OBJ", "description": "open a container"}, {"action": "close OBJ", "description": "close a container"}, {"action": "activate OBJ", "description": "activate a device"}, {"action": "deactivate OBJ", "description": "deactivate a device"}, {"action": "connect OBJ to OBJ", "description": "connect electrical components"}, {"action": "disconnect OBJ", "description": "disconnect electrical components"}, {"action": "use OBJ [on OBJ]", "description": "use a device/item"}, {"action": "look around", "description": "describe the current room"}, {"action": "look at OBJ", "description": "describe an object in detail"}, {"action": "look in OBJ", "description": "describe a container’s contents"}, {"action": "read OBJ", "description": "read a note or book"}, {"action": "move OBJ to OBJ", "description": "move an object to a container"}, {"action": "pick up OBJ", "description": "move an object to the inventory"}, {"action": "put down OBJ", "description": "drop an inventory item"}, {"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}, {"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}, {"action": "mix OBJ", "description": "chemically mix a container"}, {"action": "go to LOC", "description": "move to a new location"}, {"action": "eat OBJ", "description": "eat a food"}, {"action": "flush OBJ", "description": "flush a toilet"}, {"action": "focus on OBJ", "description": "signal intent on a task object"}, {"action": "wait", "description": "take no action for 10 iterations"}, {"action": "wait1", "description": "take no action for 1 iteration"},

{"action": "task", "description": "describe current task"}, {"action": "inventory", "description": "list your inventory"} ] Current available actions: {available_actions} Now it’s your turn to take an action. You should first reason step-by-step about the

→ current situation. This reasoning process MUST be enclosed within <think> </think>

→ tags.

Once you’ve finished your reasoning, you should choose an appropriate action for the

→ current step and present it within <action> </action> tags.

### E. Limitation and Future Work

While EPO effectively addresses the exploration-exploitation cascade failure in multi-turn sparse-reward environments, our approach does not fully leverage memory systems (Xu et al., 2025) to enhance learning from past trajectories. Currently, EPO uses historical entropy information solely for regularization, but does not incorporate explicit memory mechanisms that could help agents recall and reuse successful behavioral patterns from previous episodes. In multi-turn settings where sparse rewards make successful trajectories particularly valuable, a memory-augmented approach could potentially accelerate learning by allowing agents to explicitly store and retrieve relevant past experiences, especially those leading to rare positive rewards.

Future work could extend EPO to vision-language model (VLM) agents operating in multi-turn visual environments, where the cascade failure may manifest differently due to the multimodal nature of observations and actions. The interplay between visual and textual entropy in VLM agents presents unique challenges—visual observations might require different entropy bounds than textual responses, and the temporal dependencies across modalities could amplify or dampen the cascade failure.

### F. Use of Large Language Models

We utilized Large Language Models (LLMs), such as Claude, exclusively for ancillary support in two main areas: (i) language editing and polishing of the manuscript, and (ii) coding assistance for minor boilerplate tasks, such as generating plotting scripts and small utilities. All model-generated outputs were thoroughly reviewed, modified, and rigorously tested by the authors to ensure their accuracy and appropriateness.

The core intellectual contributions of this work—including all research ideas, algorithmic designs, experimental methodologies, data analysis, and conclusions—were conceived and validated entirely by the authors. Critically, LLMs were not used to generate any experimental results, create annotations or ground truth data, or influence methodological decisions. The authors assume full and sole responsibility for all content presented in this paper.

