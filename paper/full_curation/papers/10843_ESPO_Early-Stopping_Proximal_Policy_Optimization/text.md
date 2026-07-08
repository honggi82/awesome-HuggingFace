# arXiv:2605.29860v1[cs.LG]28May2026

[Figure 1]

2026-05-29

## ESPO: Early-Stopping Proximal Policy Optimization

Zihang Li1,2, Rui Zhou2, Yingcheng Shi1†, Wenhan Yu2, Zhewen Tan2, Zixiang Liu2, Zeming Li2, Binhua Li1, Yongbin Li1, Tong Yang2†, Jieping Ye1

[Figure 2]

1Tongyi Lab , Alibaba Group 2Peking University

### Abstract

When a large language model under reinforcement learning commits a wrong reasoning step early in a trajectory, standard algorithms force it to keep generating until the maximum horizon, spending compute on tokens that never receive positive reward and polluting advantage estimates with post-failure noise. We propose ESPO (Early-Stopping Proximal Policy Optimization), which detects trajectory failure on-the-fly and terminates rollouts early. At each generation step, ESPO computes a surrogate regret using only the logits already computed during sampling, and terminates when the smoothed cumulative regret significantly exceeds its estimated values. Truncated trajectories are treated as absorbing failure states with a terminal reward, concentrating negative temporal-difference (TD) errors near the detected failure step without any additional reward model or human annotation. On DeepSeek-R1-Distill-Qwen7B trained for mathematical reasoning, ESPO surpasses PPO on AIME 2024 (46.28% vs. 45.25%), AMC 2023 (85.83% vs. 82.94%), and MATH-500 (87.42% vs. 85.43%), while saving more than 20% rollout tokens cumulatively.

### 1 Introduction

Reinforcement learning (RL) for large language models (LLMs) has emerged as a dominant paradigm for improving reasoning ability, particularly in multi-step problem solving tasks (Ouyang et al., 2022). Process-supervised approaches further demonstrate that RL can substantially enhance chain-of-thought reasoning performance (Lightman et al., 2023b). And alignment-oriented RLHF pipelines have become central to modern LLM post-training (Casper et al., 2023). In these settings, models must generate trajectories of hundreds to thousands of tokens, each conditioned on the entire preceding context. This long-horizon generation process introduces significant challenges in credit assignment and training efficiency (Shao et al., 2024). When the model takes an inappropriate reasoning step at t∗—for example, misidentifying a mathematical operation, straying from the topic in a processing paper, or branching down an incorrect proof path—the subsequent trajectory cannot recover. The eventual reward is zero or negative, yet standard policy gradient algorithms force the policy to continue generating until the end or the fixed rollout cap Tmax. These post-failure tokens receive no positive reward but are included in the advantage estimates alongside tokens from successful trajectories, introducing noisy gradient directions that misdirect the learning process away from the actual failure mode. This leads to wasted computation on sequences that will never improve the policy. We call this the rollout continuation problem.

Existing works address adjacent problems but leave rollout continuation waste largely unsolved. Process reward models (Lightman et al., 2023b) provide step-level feedback but require extensive human annotation of intermediate reasoning steps. Methods such as GRPO (Shao et al., 2024) and DAPO (Yu et al., 2025) improve credit assignment through group-normalized advantages and clipped importance weights. However, these methods still exhaust the full horizon for every trajectory. Learned termination

†Corresponding author.

[Figure 3]

[Figure 4]

(a) Accuracy (b) Token Saving

- Figure 1: ESPO surpasses PPO on AIME 2024 at lower token cost (DeepSeek-R1-Distill-Qwen-7B). Left: AIME 2024 avg@32 (%) vs. gradient update steps. ESPO surpasses PPO earlier and maintains the lead through training. Right: It records ESPO’s cumulative rollout token saving during training compared with PPO. (tokens per step differ because ESPO truncates failing trajectories). All methods use identical prompts, reward functions, and evaluation protocol.

approaches, such as Option-Critic (Bacon et al., 2017), introduce a dedicated termination module that must be trained alongside the main policy, adding model complexity and a separate optimization objective. None of these methods detects failure on-the-fly using only the signals already produced by the actor and critic during a standard PPO (Schulman et al., 2017) step.

We introduce ESPO (Early-Stopping Proximal Policy Optimization), a lightweight rollout termination mechanism that reuses the policy’s existing logit vector and the critic’s value estimate with negligible additional computation. The core insight is that a policy in a high-regret, low-value state is less likely to recover: the gap between the action the policy would have taken greedily and the action it did take diverges sharply at failure points, while the critic simultaneously assigns low remaining value. ESPO formalizes this intuition through the following components:

- 1. Per-step surrogate regret. A regret signal computed from the logit gap at each timestep, capturing deviation from the greedy action.
- 2. EMA normalization. An exponential moving average (EMA) scheme keeps the regret signal scale-comparable to the critic’s value estimate throughout training.
- 3. Value-gated stopping criterion. A termination rule stops the rollout when the normalized cumulative regret significantly exceeds the estimated value.
- 4. Terminal failure penalty. Truncation is treated as an absorbing failure state, producing a concentrated negative TD-error at the stopping step without introducing the non-stationary bias associated with per-step reward shaping.

Experiments on 1.5B and 7B scales show that ESPO consistently outperforms both PPO and DAPO across almost all benchmarks while consuming much fewer rollout tokens. For example, at the 1.5B scale, ESPO achieves an average accuracy of 59.09% across all three benchmarks, outperforming both PPO (57.03%) and DAPO (58.29%), while consuming 927.96M cumulative tokens—significantly less than DAPO (1223.96M, −24%) and PPO (1069.66M, −13%). The ablation against random truncation (Variant F in Table 2), which matches ESPO’s stopping rate but ignores policy confidence and value estimates, scores only 42.4% on AIME 2024 despite a similar average rollout length—confirming that the improvement stems from where trajectories are truncated, rather than the reduced token budget alone.

### 2 Related Work

PPO-based LLM alignment and reasoning. Proximal Policy Optimization (PPO) (Schulman et al., 2017) is the standard backbone for RLHF in LLM alignment (Ouyang et al., 2022; Bai et al., 2022; Touvron et al., 2023), due to its stable clipped objective and implicit trust-region control. Recent work improves credit assignment and stability for long-horizon reasoning. GRPO (Shao et al., 2024) reduces variance via group-wise advantage normalization, removing the need for a learned baseline. DAPO (Yu et al., 2025) introduces dynamic sampling and advantage clipping. GSPO (Zhang et al., 2025) further stabilizes training through gradient-level regularization and structured policy updates. Despite these refinements, all PPO-style methods share a key inefficiency: they roll out trajectories to a fixed horizon Tmax even after irrecoverable errors. ESPO is orthogonal to these methods—its stopping criterion can be layered on top of any PPO-style advantage estimator.

Process reward models and step-level credit assignment. Sparse outcome rewards are a major bottleneck in multi-step reasoning. Process reward models (PRMs) (Lightman et al., 2023a) provide dense step-level supervision but require costly human annotation. Outcome reward models (ORMs) (Cobbe et al., 2021; Uesato et al., 2022) scale better but still depend on full trajectories. Preference-based methods such as DPO (Rafailov et al., 2023) avoid online RL but forgo on-policy exploration and do not extend naturally to interactive agentic settings. In contrast, ESPO requires no step-level annotation and no separate reward model: it derives its failure signal directly from the actor’s logit vector and the critic’s value estimate.

Learned termination and the options framework. Termination has been studied in RL via the OptionCritic framework (Bacon et al., 2017), which learns both the option policy and a dedicated termination function in a joint end-to-end framework. Pardo et al. (Pardo et al., 2018) show that mismatched time limits introduce bias in value estimation, motivating a distinction between truncation and natural terminal states. ESPO adopts this perspective: it maps forced truncations to absorbing failure states rather than mid-episode transitions. Unlike prior work, ESPO requires no learned termination module—it reuses signals already computed in a standard PPO forward pass.

Inference-time early stopping for reasoning models. A parallel line of work targets the overthinking problem at inference time rather than during training. ESTAR (Wang et al., 2026) introduces a tokenaware early-stopping criterion that monitors the model’s evolving answer to terminate chain-of-thought generation once a stable prediction emerges. TERMINATOR (Nagle et al., 2026) instead learns optimal exit points by training a predictor on the first-arrival positions of the final answer, achieving 14–55% length reduction on reasoning benchmarks. Both approaches operate purely at inference and assume a fixed, already-trained policy. ESPO is complementary: it truncates trajectories during RL training to remove post-failure noise from the policy gradient, yielding a policy whose own rollouts are shorter and more accurate at inference time without requiring any auxiliary inference-time controller.

Efficiency-oriented RL for reasoning. Several recent methods address compute inefficiency directly within the RL post-training stage. DRPO (Li et al., 2025) identifies that GRPO’s group-relative advantage can assign negative advantages to correct-but-long rollouts when length penalties are introduced, and decouples the length-based learning signal of correct and incorrect rollouts to mitigate this misclassification. Latent-GRPO (Deng et al., 2026) takes a more structural approach, performing policy optimization in a continuous latent reasoning space rather than over explicit token chains, thereby compressing reasoning into shorter trajectories. ESPO differs from both in mechanism and scope: rather than reshaping the reward (DRPO) or moving reasoning into latent space (Latent-GRPO), ESPO leaves the reward and the token-level action space unchanged, and instead intervenes at the rollout-collection level by detecting on-the-fly when a trajectory has entered an irrecoverable failure mode. This makes ESPO orthogonal and composable.

### 3 Preliminaries

Token-level RL for LLM generation. We formulate autoregressive decoding as a finite-horizon Markov decision process. Given an input prompt x ∼ D, the state at step t is st = (x, y<t), namely the prompt together with the previously generated tokens. The action at ∈ V is the next token selected from the vocabulary, and the transition is deterministic: st+1 = (x, y<t, at). An episode terminates when the model emits an end-of-sequence token or reaches the horizon Tmax. Following standard outcome-supervised RL for reasoning tasks, we consider sparse rewards,

rt = 0 (t < T), rT = R(x, y1:T), (1) where R(x, y1:T) denotes the final task reward, e.g., a binary correctness score.

Actor-critic PPO. Let πθ(at | st) denote the policy and Vϕ(st) the value of critic. PPO updates πθ by maximizing the clipped surrogate objective

LPPO(θ) = Et min ρtAˆt,clip(ρt,1 − ϵppo,1 + ϵppo)Aˆt , (2) where

πθ(at | st) πold(at | st)

(3) is the importance ratio. Advantages are estimated with generalized advantage estimation (GAE):

ρt =

δt = rt + γVϕ(st+1) − Vϕ(st), (4) Aˆt =

T−t−1

(γλ)lδt+l. (5)

∑

l=0

This actor-critic formulation is important for our method because the critic provides a state-dependent estimate of remaining return, while GAE propagates the effect of early termination backward to preceding tokens.

### 4 ESPO: Early-Stopping Proximal Policy Optimization

Overview. ESPO modifies rollout collection rather than the PPO objective itself. At each decoding step, it computes a cheap token-level deviation signal from the policy logits, smooths this signal over time, and compares it against a value-dependent threshold derived from the critic. Once the threshold is exceeded, the trajectory is terminated and mapped to an absorbing failure transition with a terminal penalty. PPO and GAE are then applied to the truncated trajectory in the usual way. In this sense, ESPO can be viewed as PPO on an augmented episodic MDP whose termination rule is induced online during generation.

#### 4.1 Stepwise deviation signal

At state st, let at ∼ πθ(· | st) be the sampled token. We define the stepwise deviation signal (also called the regret value) as

log πθ(a | st) − log πθ(at | st). (6) By construction, gt ≥ 0. The quantity is small when the sampled token is close to the policy mode, and increases when sampling deviates from the policy’s most preferred action. Importantly, gt is obtained directly from the logits already computed for decoding and therefore introduces negligible additional computation cost.

gt = max

a∈V

#### 4.2 Normalized cumulative stopping statistic

Because the scale of gt changes during training, we first normalize it with running batch statistics. Let µg and σg2 denote an exponential moving average of the batch mean and variance of gt:

µg ← αemaµg + (1 − αema) gB, (7) σg2 ← αemaσg2 + (1 − αema) Var(gB), (8)

where B denotes the current rollout batch. Crucially, to ensure causal correctness and prevent future information within the current rollout from leaking into the termination decision, µg and σg2 are updated only at the boundary of each training batch. Using these strictly frozen batch-level statistics during generation, we define the normalized signal

  , (9)

 

gt − µg σg2 + δ

g˜t = clip

, −c, c

where δ > 0 is a numerical stabilizer and c is a clipping bound. We then accumulate the normalized signal within each trajectory using

zt = αszt−1 + (1 − αs)g˜t, z0 = 0. (10) The scalar zt (also called the cumulative regret value) serves as the stopping statistic used by ESPO.

- 4.3 Value-gated early termination ESPO triggers early termination at the current step t if the cumulative deviation satisfies

zt > β · max Vϕ(st), ε . (11)

ε acts as the threshold. The gating has a simple interpretation: states with high predicted future return are granted larger tolerance, whereas states with low predicted return are terminated after a smaller amount of accumulated deviation. This also to some extent prevent the correct but non-mode token (with high regret) from being wrongly terminated.

To maintain a stable stopping frequency throughout training, the threshold multiplier can be adjusted by a proportional controller,

β ← clip β + ηβ(ρˆstop − τ), βmin, βmax , (12) where ρˆstop is the empirical stop rate and τ is the target rate. In practice, evaluating Equation (11) immediately would induce spurious early terminations due to uncalibrated value baselines from the randomly initialized critic. Therefore, we disable the stopping rule during an adaptive critic warmup period, which dynamically concludes once the critic exhibits stable learning dynamics (e.g., when validation loss improvement drops below a specified tolerance continuously). More details about warmup are stated in Appendix B. Following this burn-in, β is linearly annealed from a conservative upper bound down to its target regime, safely transitioning the data collection pipeline without introducing shock to the policy gradient.

- 4.4 Failure transition and PPO training

Suppose the stopping rule fires at step Tstop. ESPO then converts the current prefix into an absorbing failure transition:

rt = 0 (t < Tstop), rTstop = rfail. (13) No further decoding is performed after Tstop, and no bootstrap term is applied beyond the absorbing state. The resulting trajectory is therefore shorter than the nominal rollout horizon, but its advantages are still computed by the same GAE recursion in Equation (5). In particular, the stopping event induces a negative temporal-difference signal at the termination point, which is propagated backward to earlier steps through PPO training.

The final training objective remains the standard PPO formulation; ESPO changes the sampled trajectories, not the algebraic form of the policy update. Applying an absorbing penalty exactly at the termination step inherently avoids the pathologies associated with explicit per-step penalties. A state-dependent per-step penalty would introduce a non-stationary reward function that biases the critic and inadvertently incentivizes the policy to collapse its logit spread rather than solve the task.

Algorithm 1 ESPO Rollout Collection Require: Policy πθ, critic Vϕ, terminal penalty rfail Require: Parameters αs, β, ε, warmup status

- 1: z ← 0; t ← 0; done ← False
- 2: while t < Tmax and not done do
- 3: Compute logits ℓ = log πθ(· | st); sample at ∼ πθ(· | st)
- 4: gt ← maxa ℓa − ℓat {Single-step surrogate regret}
- 5: g˜t ← clip (gt − µg)/ σg2 + δ, −c, c {Uses frozen batch EMA}

- 6: z ← αsz + (1 − αs)g˜t {Exponential smoothing}
- 7: if warmup complete and z > β · max Vϕ(st), ε then
- 8: rt ← rfail; done ← True {Absorbing failure transition}
- 9: else
- 10: rt ← 0; t ← t + 1 {Continue decoding}
- 11: end if
- 12: end while
- 13: return trajectory τ = (s0, a0,r0, . . . , sTstop, aTstop,rfail)

We acknowledge that formulating ESPO as an augmented MDP introduces an inherent objective bias via false positives, terminating locally uncertain but globally recoverable trajectories. However, the aforementioned critic warmup acts as the primary safeguard against this bias early in training, ensuring that valid exploration is not aggressively truncated before the value function is informative. The complete rollout collection procedure is summarized in Algorithm 1.

### 5 Experiments

#### 5.1 Setup

Models and Benchmarks We evaluate DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-AI, 2025) and DeepSeek-

R1-Distill-Qwen-7B (DeepSeek-AI, 2025) models which are trained on DAPO-Math-17k (Yu et al., 2025). The performance is evaluated across three held-out benchmarks: AIME24 (Jia, 2024), AMC23 (Project Numina, 2024) and MATH500 (Lightman et al., 2023b).

Baselines and Implementation We compare ESPO against (1) Base Model; (2) PPO (Schulman et al., 2017), the standard token-level actor-critic baseline with full-horizon rollout collection; and (3) DAPO (Yu et al., 2025), which builds on GRPO with dynamic sampling and advantage clipping to improve training stability. The number of rollout is set to 8 and the global batch size is set as 64 for all methods. All trained under identical data, reward, sampling, and evaluation settings. All algorithms are implemented via verl (Sheng et al., 2024) using outcome-based rewards. Moreover, all experiments were conducted on 8 × H20 GPUs. All detailed training parameters related are provided in Appendix A. For evaluation, to maintain results stability, we repeat the evaluation 32 times and reporting the Pass@1 (averaged over 32 samples). The hyperparameters of inference are consistently set to a temperature of 1.0, a top-p of 0.7 and a top-k of -1.0.

#### 5.2 Main Results

Overall Performance. Table 1 presents performance across all benchmarks at both model scales. At the 7B scale, ESPO achieves 85.83% on AMC 2023, 46.28% on AIME 2024, and 87.42% on MATH-500, surpassing both PPO and DAPO on every benchmark while consuming only 839.24M cumulative rollout tokens—roughly 22% fewer than PPO and 19% fewer than DAPO. In terms of average accuracy across all three benchmarks, ESPO achieves 73.17%, compared to 71.20% for PPO (+1.97pp) and 71.76% for

- Table 1: Main results on math benchmarks on 1.5B and 7B scales. Cumulative Tokens (M) denotes total rollout tokens consumed over 500 training steps. Avg Tokens denotes mean rollout length per trajectory. Avg acc denotes the average accuracy across the three datasets. Best results in bold.

DeepSeek-R1-Distill-Qwen-1.5B Method AMC23 AIME24 MATH500 Avg Acc Cumulative Tokens(M) Avg Tokens

Base Model 58.28 20.31 74.81 51.13 - 5808.00 PPO 68.43 23.02 79.65 57.03 1069.66 4178.37 DAPO 70.23 24.37 80.28 58.29 1223.96 4781.09 ESPO(Ours) 71.87 23.87 81.53 59.09 927.96 3624.86

DeepSeek-R1-Distill-Qwen-7B Method AMC23 AIME24 MATH500 Avg Acc Cumulative Tokens(M) Avg Tokens

Base Model 78.64 40.13 83.36 62.04 - 5357.00 PPO 82.94 45.25 85.43 71.20 1072.40 4189.06 DAPO 83.76 45.57 85.95 71.76 1035.01 4043.00 ESPO(Ours) 85.83 46.28 87.42 73.17 839.24 3278.30

DAPO (+1.41pp).

1.5B Scale Results. At the 1.5B scale, ESPO likewise demonstrates consistent improvements. It achieves 71.87% on AMC 2023, 81.53% on MATH-500, and an average accuracy of 59.09% across all benchmarks, outperforming both PPO (57.03%) and DAPO (58.29%). Although ESPO’s AIME 2024 score of 23.87% is marginally below DAPO (24.37%), it consumes only 927.96M cumulative tokens—significantly less than DAPO (1223.96M, −24%) and PPO (1069.66M, −13%)—while maintaining competitive accuracy, confirming that early-stopping successfully eliminates post-failure token waste without degrading generation quality. Figure 1 (right panel) further illustrates training efficiency in terms of cumulative rollout tokens consumed.

Comparison with DAPO. DAPO, which introduces dynamic sampling and advantage clipping on top of GRPO, generally outperforms standard PPO at both scales, confirming the importance of improved credit assignment for long-horizon reasoning. ESPO further improves upon DAPO on almost all benchmarks, while consistently using fewer rollout tokens. This demonstrates that ESPO is complementary to advantage-estimation improvements: by eliminating uninformative post-failure tokens from the rollout buffer, ESPO provides a cleaner learning signal regardless of the advantage normalization scheme employed.

6 Ablation Studies

- Table 2 isolates the contribution of each ESPO component by removing or replacing it while keeping all other settings fixed. Experiments are conducted on DeepSeek-R1-Distill-Qwen-7B.

Critic warmup (A vs. B). Variant B removes the adaptive warmup schedule described in Section 4. The gap between A and B (46.3 vs. 44.2) is 2.1 points: the warmup increases performance because the warmup serves as a primary safeguard against objective bias by preventing the algorithm from aggressively truncating valid exploration before the critic becomes informative. This phase ensures that the value-gated threshold β · max Vϕ(st), ε is derived from a stabilized value baseline, thereby avoiding spurious early terminations caused by the high variance of a randomly initialized critic.

Table 2: Ablation on AIME24 evaluated by avg@32, cumulative tokens and average training rollout tokens per trajectory. (A) is the full ESPO; all other rows remove or replace one component. ↑ higher is better, ↓ lower is better.

Variant AIME24 ↑ Cumulative Tokens(M) Avg Tokens ↓

- (A) Full ESPO (ours) 46.3 839.24 3278.30
- (B) w/o warmup 44.2 858.37 3353.03

- (C) w/o terminal failure penalty 43.7 901.65 3522.09
- (D) Value-only stop (Vϕ < τ, no regret) 44.0 1090.05 4258.02
- (E) Regret-only stop (zt > τ, no value gate) 44.8 1086.51 4244.18

- (F) Random stop 42.4 855.59 3342.14

Terminal failure penalty (A vs. C). Removing the terminal failure penalty reduces AIME24 by 2.6 points and adds more than 181 average training rollout tokens per trajectory (from 3278 to 3522). This performance drop occurs because treating the truncation as an absorbing failure state with a specific terminal penalty induces a concentrated negative temporal-difference (TD) error. Through training, this more precise negative-signal efficiently propagates backward to earlier steps, providing precise credit assignment.

Value-only vs. regret-only stopping (A vs. D vs. E). Variant D stops when Vϕ(st) < τ without any regret signal, where τ is a fixed threshold, relying solely on the critic. Variant E stops when the cumulative deviation zt > τ without value gating. Both underperform full ESPO (A), with variant D scoring 44.0 and variant E scoring 44.8. Value-only stopping depends entirely on the Critic’s absolute scale, which varies across tasks and training stages. Regret-only stopping lacks the recovery allowance that the value term provides. The combination in variant A outperforms either component alone, confirming that both signals carry complementary information.

Random stop (A vs. F). Variant F replaces the surrogate regret with a random stop signal: the rollout stops randomly when generating a trajectory and to control for variables, we set the rate of random truncation to be the same as that in our method. This achieves 42.4 on AIME24—below all ESPO variants, indicating that randomly reducing rollout tokens without considering the policy’s internal confidence or value estimates fails to remove the specific post-failure noise precisely that hampers learning. Furthermore, despite having a similar average rollout length (3342 tokens) to the full ESPO, Variant F exhibits a significant performance gap of 3.9 points, confirming that the benefits of ESPO stem from "where" the trajectories are truncated rather than simply training on shorter sequences.

### 7 Analysis

#### 7.1 Response Length

Figure 2a plots the mean response length curves over training. The way we calculate ESPO/Original is that during the rollout process, we compute the cumulative regret value as usual. When the first token position satisfying the truncation condition is identified, the sequence is not actually truncated. Instead, we set all mask entries after the truncation point to zero to simulate the truncation. In the subsequent calculation of losses, tokens generated beyond the truncation point are treated as padding tokens and excluded from gradient backpropagation. This allows us to identify the truncated but correct answer sequences. The false-positive rate measuring the proportion of originally correct sequences that are improperly truncated by ESPO is also calculated in this way.

Trajectories that reach natural completion in ESPO (ESPO/Original) are closely tracking the mean response

[Figure 5]

[Figure 6]

(a) Response length over training steps. (b) Actor entropy over training steps.

[Figure 7]

(c) False Positive Rate over training steps.

- Figure 2: Training dynamics of ESPO vs. PPO on DeepSeek-R1-Distill-Qwen-1.5B. In (a), ESPO/Original records mean length of all trajectories in ESPO that the model should have generated without executing the early-stopping; PPO/Original records mean length of all trajectories during training process in PPO; ESPO/Actual records mean length of all trajectories in ESPO when executing the early-stopping, including the truncated responses and the non-truncated responses. In (b), ESPO maintains higher entropy throughout training, mitigating premature entropy collapse. In (c), the false positive rate measures the proportion of correct-yet-truncated responses at each step.

length in PPO(PPO/Original)—while the actual length of trajectories in ESPO(ESPO/Actual) is significantly lower than them, indicating that our method does not destroy the policy’s distribution of the response lengths. At the same time, the accuracy on the validation set also exceeds the baseline method, indicating the positive effectiveness of removing post-failure noise on model training.

#### 7.2 Policy Entropy and Diversity

A natural concern is that ESPO’s logit-gap signal—which fires when the sampled token deviates from the policy mode—might suppress exploration by penalizing low-probability tokens regardless of whether they reflect irrecoverable errors. To check for this, we track the entropy of πθ(·|st) over training as stated in Figure 2b. ESPO not only does not cause entropy collapse, but also reduces the rate of decrease in the entropy of policy compared to PPO, that is, it opened up the exploration space of the model further. These controls rule out the hypothesis that ESPO’s gains arise from encouraging greedy/mode-seeking behavior at the expense of exploratory tokens. From another perspective, the stopping strategy of ESPO mainly determines whether to stop by calculating the numerical relationship between value and logit-gap. It does not directly penalize low-probability tokens, so it will not directly lead to a decrease in entropy.

Beyond the absence of direct penalty on token probabilities, ESPO actively slows entropy decay by removing a source of spurious gradient signal. In standard PPO, post-failure tokens within a doomed trajectory still receive negative advantages and contribute to the policy gradient, pushing the policy to sharpen its distribution against tokens that were not, in fact, the cause of failure. By truncating these trajectories, ESPO eliminates this misattributed pressure, leaving the policy free to retain probability mass on plausible alternative continuations.

#### 7.3 False-positive truncation

We record how often ESPO truncates trajectories that would have recovered under full rollout in Figure 2c. We found that, on average in each batch of data, 2.7% of the trajectories could have yielded correct answers, but are instead truncated, implying an average false-positive truncation rate of 2.7% in the training process. These false positives incur a small training cost (the policy misses 2.7% of potentially correct trajectories), but this is bounded by the improvement over full-horizon PPO, suggesting the benefit of removing post-failure noise outweighs the cost of occasional early false positives.

### 8 Limitations and Future Work

Our method has limitations when dealing with models that are incorrect but highly confident. In such cases, the surrogate regret approaches zero when the policy is confidently wrong: a policy that assigns high probability to an incorrect reasoning branch produces a small logit-gap, delaying detection. Similarly, as shown in the above error killing rate, ESPO may prematurely trigger the stop condition on a few high-entropy but correct steps, resulting in a false kill. A more refined early stopping strategy can be incorporated into future work to reduce this phenomenon. The truncation rate needs to be manually adjusted to suit different models and tasks. Future work can explore more adaptive mechanisms to reduce the sensitivity to hyperparameters. Furthermore, extending the stopping criterion to tool-use and multi-turn agentic settings, where the error may occur across environment steps rather than individual tokens, is a natural next direction.

### 9 Conclusion

We introduced ESPO, an efficient regret-aware rollout termination method designed to optimize agentic reinforcement learning and LLM reasoning training. By combining a surrogate regret signal derived from the actor’s own logit distribution with a dynamic value-gated threshold, ESPO accurately detects and truncates failing trajectories on-the-fly. Applying an implicit failure penalty at the termination step successfully isolates and removes post-failure noise from the policy gradient, completely eliminating the need for auxiliary reward models or human annotations. Extensive evaluations across rigorous mathematical reasoning benchmarks validate the effectiveness of our method. Notably, on DeepSeekR1-Distill-Qwen-7B, ESPO outperforms the PPO and DAPO baselines on multiple benchmarks, while simultaneously reducing the training rollout tokens cumulatively by more than 20%. Ultimately, these results demonstrate that ESPO provides a scalable, compute-efficient framework for advancing longhorizon reasoning capabilities in large language models.

### References

Pierre-Luc Bacon, Jean Harb, and Doina Precup. The option-critic architecture. In Thirty-First AAAI Conference on Artificial Intelligence, 2017.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, et al. Constitutional ai: Harmlessness

from ai feedback. arXiv preprint arXiv:2212.08073, 2022. URL https://arxiv.org/abs/2212.08073. Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando,

Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.or g/abs/2110.14168.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Jingcheng Deng, Zihao Wei, Liang Pang, Junhong Wu, Shicheng Xu, Zenghao Duan, and Huawei Shen. Latent-GRPO: Group relative policy optimization for latent reasoning. arXiv preprint arXiv:2604.27998, 2026.

Maxwell Jia. AIME 2024 dataset. https://huggingface.co/datasets/Maxwell-Jia/AIME_2024, 2024. Gang Li, Yan Chen, Ming Lin, and Tianbao Yang. DRPO: Efficient reasoning via decoupled reward policy

optimization. arXiv preprint arXiv:2510.04474, 2025.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023a. URL https: //arxiv.org/abs/2305.20050.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023b.

Alliot Nagle, Jakhongir Saydaliev, Dhia Garbaya, Michael Gastpar, Ashok Vardhan Makkuva, and Hyeji Kim. TERMINATOR: Learning optimal exit points for early stopping in chain-of-thought reasoning. arXiv preprint arXiv:2603.12529, 2026.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Fabio Pardo, Arash Tavakoli, Vitaly Levdik, and Petar Kormushev. Time limits in reinforcement learning. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholm, Sweden, Proceedings of Machine Learning Research, pages 4042–4051. PMLR, 2018. URL http://proceedings. mlr.press/v80/pardo18a.html.

Project Numina. AI-MO validation AMC dataset. https://huggingface.co/datasets/AI-MO/aimo-v

alidation-amc, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems 36: NeurIPS 2023, New Orleans, LA, USA, 2023. URL http: //papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstr act-Conference.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. URL https://arxiv.org/abs/2302.13971.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, et al. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022. URL https://arxiv.org/abs/2211.14275.

Junda Wang, Zhichao Yang, Dongxu Zhang, Sanjit Singh Batra, and Robert E. Tillman. ESTAR: Earlystopping token-aware reasoning for efficient inference. arXiv preprint arXiv:2602.10004, 2026.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Tianyi Zhang, Zhihan Liu, Yichao Xu, Jing Zhou, and Yuxin Chen. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. URL https://arxiv.org/abs/2507.18071.

### A Training Details

For the ESPO training hyperparameters, the learning rate is set to 1 × 10−6. The maximum rollout length (Tmax) is configured to 8192 tokens, alongside a global batch size of 64 and the number of rollout is 8. Regarding the algorithmic specifics, the failure reward rfail is set to −1.0, the EMA αema to 0.99, and the normalisation αs to 0.9. The initial value of β is set to 7.0. A β adjustment rate of 0.1 is applied to maintain a target termination rate of 0.25. The baseline methods maintain the same settings, such as the global batch size. Additionally, for DAPO, the clip_ratio_low is set to 0.2 and the clip_ratio_high is set to 0.28 as stated in (Yu et al., 2025).

Table 3: ESPO hyperparameters.

Hyperparameter Value

Failure reward rfail −1.0 EMA αEMA 0.99 Normalization αs 0.9 Initial β 7.0 Minimum βmin 0.0 β adjustment rate 0.1 Target termination rate 0.25 Max rollout length Tmax 8192 Learning rate 1 × 10−6 The Value’s ε 0.2

### B Adaptive Critic Warmup Details

During warmup, the stopping criterion is disabled and the critic is updated using only the base PPO objective. Warmup exits early if the absolute value of critic’s loss is less than 0.5 or the difference between adjacent step’s values is less than 0.1 for three consecutive steps, indicating the critic has converged. If critic’s loss does not meet the convergence criterion after 10% of total training steps, warmup ends unconditionally and the stopping criterion activates to avoid indefinitely deferring the ESPO mechanism.

[Figure 8]

[Figure 9]

(a) Critic loss on DeepSeek-R1-Distill-Qwen-1.5B (b) Critic loss on DeepSeek-R1-Distill-Qwen-7B

- Figure 3: The left figure illustrates the critic loss for DeepSeek-R1-Distill-Qwen-1.5B, while the right figure depicts the critic loss for DeepSeek-R1-Distill-Qwen-7B.

