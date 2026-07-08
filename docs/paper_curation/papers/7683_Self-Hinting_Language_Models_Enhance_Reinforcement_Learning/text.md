## Self-Hinting Language Models Enhance Reinforcement Learning

Baohao Liao*†12 Hanze Dong*1 Xinxing Xu1 Christof Monz2 Jiang Bian1

### Abstract

SAGE

###### Standard GRPO

Group Relative Policy Optimization (GRPO) has recently emerged as a practical recipe for aligning large language models with verifiable objectives. However, under sparse terminal rewards, GRPO often stalls because rollouts within a group frequently receive identical rewards, causing relative advantages to collapse and updates to vanish. We propose self-hint aligned GRPO with privileged supervision (SAGE), an on-policy reinforcement learning framework that injects privileged hints during training to reshape the rollout distribution under the same terminal verifier reward. For each prompt x, the model samples a compact hint h (e.g., a plan or decomposition) and then generates a solution τ conditioned on (x,h). Crucially, the task reward R(x,τ) is unchanged; hints only increase within-group outcome diversity under finite sampling, preventing GRPO advantages from collapsing under sparse rewards. At test time, we set h = ∅ and deploy the no-hint policy without any privileged information. Moreover, sampling diverse selfhints serves as an adaptive curriculum that tracks the learner’s bottlenecks more effectively than fixed hints from an initial policy or a stronger external model. Experiments over 6 benchmarks with 3 LLMs show that SAGE consistently outperforms GRPO, on average +2.0 on Llama-3.2-3BInstruct, +1.2 on Qwen2.5-7B-Instruct and +1.3 on Qwen3-4B-Instruct. The code is available at https://github.com/BaohaoLiao/SAGE.

reference trajectory

hardprompt 𝑙 𝜏∗ 𝑥

hint strength

𝑥

hint

# arXiv:2602.03143v1[cs.LG]3Feb2026

𝑞∅ 𝑥,𝜏∗,𝑙)

policy

𝜋 𝑥,ℎ)

𝜋 𝑥)

ℎ

self-hinting same model for 𝑞∅ and 𝜋

trajectory

𝜏 𝜏 . . . 𝜏

𝜏 . . . 𝜏

𝜏

[Figure 1]

reward = 0 advantage collapse update vanish

adaptive hint strength retain on-policy use all difficulty level of prompts

Figure 1. An overview of our proposed method, SAGE. When an LLM can’t sample any correct trajectory for a prompt, the LLM self-generates hint from the reference solution of the prompt. The hint is then used together with the difficult prompt as input to the LLM, avoiding advantage collapse and ensuring the sampling of correct trajectories to update the policy model.

verifiable objectives such as exact-match correctness, unit tests, or automated checkers (Ouyang et al., 2022; Schulman et al., 2017; DeepSeek-AI et al., 2025). In this setting, the objective is straightforward: maximization the expected reward over prompts, yet optimization can be fragile: with finite sampling, policy-gradient estimators may exhibit high variance and can even become degenerate on hard prompts.

A salient example arises with Group Relative Policy Optimization (GRPO) (Shao et al., 2024) under sparse terminal rewards. GRPO centers (and often standardizes) rewards within each rollout group, relying on within-group outcome differences to produce a nonzero update. With a 0/1 verifier, difficult prompts frequently yield groups where all rollouts receive the same reward (typically all zeros). In that case, the group-centered advantages collapse and the minibatch policy-gradient estimate becomes identically zero. Importantly, this is a finite-sample pathology: the underlying expected objective needs not be flat, but the estimator provides no learning signal for many prompts.

### 1. Introduction

Reinforcement learning (RL) has become a core tool for training and aligning large language models (LLMs), particularly when supervision is most naturally expressed via

*Equal contribution †This work was done during an internship at Microsoft. 1Microsoft Research 2Language Technology Lab, University of Amsterdam. Correspondence to: Hanze Dong <hanzedong@microsoft.com>.

Existing remedies largely modify data collection. A common baseline is to skip uninformative updates (e.g., degenerate groups) and resample prompts, which improves performance but implicitly biases training toward easier prompts (Yu et al., 2025; Xiong et al., 2025a). More systematic

Preprint. February 4, 2026.

approaches include adaptive sampling or curriculum-style scheduling to allocate more rollouts to difficult prompts (Yao et al., 2025; Xiong et al., 2025b; Li et al., 2025; Zhang et al., 2025c), as well as leveraging offline data or externally generated candidates (e.g., from stronger models) to bootstrap learning (Zhang et al., 2025a; Yan et al., 2025; Zhang et al., 2025b). While effective, these strategies can either skew the training distribution or introduce context/distribution mismatch that must be handled carefully.

We propose SAGE (Self-hint Aligned GRPO with Privileged Supervision), a complementary approach based on privileged hinting. During training, we provide an additional hint h, a lossy compression of a reference solution τ⋆, and roll out from the hint-conditioned policy πθ(· | x,h). Hints only reshape the rollout distribution to increase the probability of observing mixed outcomes within a finite group. At test time, we deploy the no-hint policy. We refer to hints generated by the policy itself as self-hints, and to the procedure of generating such hints as self-hinting.

This degeneracy can be made explicit. Let pθ(x) be the no-hint success probability and G the group size. The probability that a rollout group contains mixed outcomes is 1 − (1 − pθ(x))G − pθ(x)G ≈ Gpθ(x), so updates vanish whenever Gpθ(x) ≪ 1. Hinting is useful precisely when it increases the effective success probability so that mixed-outcome groups become common for the same G.

Contributions. (i) We introduce SAGE as shown in Figure 1, an on-policy RL framework that conditions rollouts on privileged hints during training while keeping the task reward unchanged and removing hints at test time. (ii) We develop a policy-dependent hint-strength scheduler, that activates hints only when within-group rewards collapse, yielding an automatic curriculum. (iii) We propose an online self-hinting scheme that periodically refreshes the hint distribution during training to maintain calibration to the learner, avoiding overly weak/overly strong fixed hints. (iv) We provide analysis that characterizes GRPO collapse as a gate-opening probability under Bernoulli rewards and empirically validate that SAGE improves sample efficiency and final accuracy on challenging reasoning benchmarks.

### 2. RL with Privileged Hinting

Vanilla GRPO works well when a prompt x yields occasional positive rollouts. In hard regimes, groups often receive identical rewards, collapsing within-group advantages and stalling learning. We address this failure mode by injecting privileged hints during training that keep the reward unchanged while reshaping the rollout distribution to surface informative trajectories under finite sampling.

Epoch 1 Epoch 2 Epoch 3

100

Llama-3.2-3B-Instruct

Promptsw/osignal(%)

- Qwen2.5-7B-Instruct

- Qwen3-4B-Instruct

80

60

40

20

0 125 250 375

Step

Figure 2. The percentage of prompts whose correct trajectories have NEVER been sampled w.r.t. the training step. Here we train on 64k prompts, and sample 8 traces per prompt per step. A large number of prompts is wasted during RL, especially for a weaker LLM, since they don’t offer any signal for training.

##### 2.1. Setup and the GRPO stall

Let x ∼ D be a prompt. A policy πθ generates a trajectory τ = (y1,...,yT), written as τ ∼ πθ(· | x). We use a binary reward R(x,τ) ∈ {0,1}. Define the success probability

[R(x,τ) = 1]. (1)

pθ(x) = Pr

τ∼πθ(·|x)

GRPO implementations often standardize groupwise advantages by the within-group standard deviation. For a group of G rollouts {τi}Gi=1, let

G

G

(Ri − R¯)2,

Ri = R(x,τi), R¯ = G1

Ri, s2 = G1

i=1

i=1

and define standardized advantages

Ri − R¯ s + ϵ

, (2)

Ai =

where ϵ ≥ 0 is a numerical stabilizer and Ri ∈ {0,1}. When pθ(x) is tiny, a group is often all-zero and Ai = 0 for all i. The chance of a non-degenerate group is

1−(1−pθ(x))G−pθ(x)G ≈ 1−(1−pθ(x))G ≈ Gpθ(x), so training stalls whenever Gpθ(x) ≪ 1 on most prompts. Figure 2 illustrates this phenomenon in practice: for many hard prompts, correct trajectories are never sampled for a long stretch of training, yielding no learning signal.

##### 2.2. Privileged hinting as sampling

When a reference trajectory τ⋆ is available during training, we generate a hint h as a lossy compression of τ⋆. The hint is appended to the prompt as additional context.

Hint strength and the no-hint case. We control hint informativeness with a discrete strength level ℓ ∈ {0,1,...,L},

where larger ℓ indicates more information about the reference trajectory τ⋆. We sample

ℓ ∼ p(ℓ), h ∼ q(h | x,τ⋆,ℓ), (3) where ℓ = 0 corresponds to the no-hint setting and q(h | x,τ⋆,0) = δ∅(h) (i.e., h = ∅ deterministically).

Policy-dependent scheduling of ℓ. Hints should be used only when a prompt provides no learning signal. We let the sampling of ℓ depend on the policy through a simple statistic, such as a collapse indicator c(x) = I Var {R(x,τi)}Gi=1 = 0 , computed from a small probe group under the policy model πθ. A minimal scheduler is

p(ℓ | x) =

δ0, c(x) = 0, p(ℓ), c(x) = 1,

(4)

so ℓ > 0 is activated only when the group collapses. With a hint, we sample from πθ(· | x,h). The success rate increases: p(θℓ)(x) = Prτ∼π

θ(·|x,h)[R(x,τ) = 1],h ∼ q(· | x,τ⋆,ℓ). Hinting is useful when it raises p(θℓ)(x) enough that Gp(θℓ)(x) is no longer tiny, so non-degenerate groups become common and RL receives updates.

- 2.3. GRPO with hints and the final loss Given x, sample ℓ and h, then draw a group of rollouts

τi ∼ πθ(· | x,h), Ri = R(x,τi), i = 1,...,G. Compute Ai with Eq. (2). The loss conditioned on hints is

L(θ) = −E G 1

G

i=1

Ai

Ti

t=1

log πθ(yi,t | x,h,yi,<t)

+β E KL πθ(· | x,h)∥πref(· | x,h) ,

(5)

where the expectation is over x ∼ D, the hint sampling from Eq. (3), and rollouts from πθ(· | x,h). At test time we set ℓ = 0, h = ∅ and run the policy πθ(· | x,∅) ≡ πθ(· | x).

Summary. Sparse rewards can cause GRPO to stall because, for many prompts x, finite groups contain no positive samples and advantages collapse. Privileged hinting fixes this by changing the rollout distribution for such prompts while keeping the reward unchanged. A policy-dependent scheduler activates hints only when groups collapse, yielding an automatic curriculum. Training remains on-policy since rollouts are drawn from πθ(· | x,h). Deployment uses ℓ = 0 and requires no hints or privileged information.

- 3. Analysis

##### 3.1. Standardized GRPO as a gated update objective

Fix a context (x,h) and draw G rollouts with rewards Ri ∈ {0,1}. Let R¯ = G1 i Ri and s2 = G1 i(Ri − R¯)2. Standardized GRPO uses Ai = R

i−R¯ s+ϵ .

Corollary 3.1 (Signal energy equals a gate probability). Define the advantage energy E := G1 Gi=1 A2i. If ϵ > 0,

s2

(s + ϵ)2 ∈ [0,1], (6) which is monotone in s and still collapses to 0 when s = 0. For GRPO, the prompt-level update magnitude is dominated by whether the group is non-degenerate (s > 0). In other words, training behaves like a gated procedure that updates only when the rollout group contains mixed outcomes.

E =

- Proposition 3.2 (Gate opening probability under Bernoulli rewards). Let pθ(x,h) = Pr[R(x,τ) = 1 | x,h]. Then

Pr[s > 0 | x,h] = 1 − (1 − pθ(x,h))G − pθ(x,h)G, (7)

Pr[s > 0 | x,h] is maximized at pθ = 21. In the sparse regime pθ(x,h) ≪ 1, Pr[s > 0 | x,h] ≈ Gpθ(x,h).

Thus, SAGE should choose the hint strength to move hard prompts out of the regime where Gpθ ≪ 1, and avoid overly strong hints that push pθ ≈ 1 to close the gate again.

- Proposition 3.3 (Optimal hint distribution is policy-dependent). Fix a prompt x and group size G ≥ 2. Let

θ(·|x,h)[R(x,τ) = 1] and define

pθ(x,h) = Prτ∼π

u(p) := 1 − (1 − p)G − pG. (8) Under Bernoulli rewards, u(pθ(x,h)) = Pr[s > 0 | x,h]. For any distribution q(· | x), define the expected probability

Jx(θ,q) := Eh∼q(·|x)[u(pθ(x,h))]. (9) Then u is symmetric and strictly concave on [0,1], and is maximized at p = 12. Consequently, any maximizer qθ⋆(· | x) ∈ arg max

Jx(θ,q) (10)

q

must place its mass on calibrating hints that make pθ ≈ 12. In general, the set of calibrating hints depends on θ. Unless pθ(x,h) is invariant in θ for q-almost all h, a fixed q cannot remain (near) optimal for Jx(θ,q) throughout training. Updating q online therefore reduces this gate-mismatch and increases the frequency of non-degenerate GRPO updates. Remark 3.4 (Why not sample many hints per prompt.). Let u(p) = 1 − (1 − p)G − pG denote the gate probability in (7). For G ≥ 2, u is concave on [0,1] since u′′(p) = −G(G − 1) pG−2 + (1 − p)G−2 ≤ 0. Therefore, for any hint distribution h ∼ q,

Eh∼q u(pθ(x,h)) ≤ u(Eh∼q[pθ(x,h)]). (11)

At a fixed mean success rate, additional randomness across hints can only reduce the expected frequency of nondegenerate groups. Motivated by (11), we sample a single hint realization per prompt per epoch and spend compute on G or better strength scheduling.

58

GPT-5 hint

Self-hint online

w/o training

| |
|---|

60

Self-hint offline

No hint, l=0

58.3

Averageaccuracy(%)

56

Averageaccuracy(%)

58

57.1

57.0 56.7

56.5 56.3 55.6 55.4

54

55.9

56

52

54.1

54

53.3

GPT-5 hint

Self-hint offline Self-hint online No hint, l=0

50

52

50

l=1 l=2 l=3 Hint level

0 50 100 150 200 250 300 350 400

Step

- Figure 3. Average accuracy on Qwen3-4B-Instruct over 6 benchmarks. The 4.5k training prompts here are extremely hard, whose correct trajectories have never been sampled during training as Figure 2. The number of rollouts per prompt per step here is set to 32 to encourage exploration. Left: Performance on various hints. Training without hint only slightly improves the performance, since the reward signal from the hard prompts is sparse. However, training with any hint boosts the performance. Among all methods, online self-hinting consistently achieves the best performance across different hint levels. Right: Average accuracy w.r.t. the training steps for hint level l = 2. Training without any hint even degrades the performance as the training goes, since the reward signal is too sparse, making it overfit to a few solvable prompts. However, online self-hinting boosts the performance steadily. Refer to Table C.1 for detailed number.

Overall, standardized GRPO turns sparse-reward learning into maximizing a gate probability. The proposed method should operationalize this by (i) ensuring on-policy conditioning on h, (ii) scheduling ℓ when the gate is closed, (iii) updating the self-hint generator online to keep pθ calibrated.

In SAGE, hint generation is online. We implement qϕ(h | x,τ⋆,ℓ) by prompting the policy πθ to produce a procedureonly plan aligned with the reference trajectory.

We evaluate three variants: (1) Fixed privileged hints: qϕ is derived from πθ

and frozen after initialization. (2) Online privileged hints (SAGE): qϕ is derived from πθ and refreshed during training. (3) External teacher hints: qϕ is produced by a stronger frozen model, when available.

0

### 4. Design of SAGE

##### 4.1. On-policy training

By Figure 3, adding hints improves performance across hint levels compared with no hint, consistent with hints increasing the chance of sampling informative trajectories under sparse rewards. Besides, online self-hinting consistently performs best, indicating that continually refreshed self-hints are better calibrated to the learner than fixed hints.

Why hints must be in the conditioning context. SAGE appends a hint h to the prompt and samples rollouts from the hint-conditioned policy πθ(· | x,h). This is not merely a modeling choice: it is what keeps training on-policy for the augmented context. The loss is t log πθ(yt | x,h,y<t). If one instead samples τ ∼ πθ(· | x,h) but evaluates log πθ(τ | x) (i.e., dropping h inside the log-prob), the update no longer corresponds to the gradient of any onpolicy objective under the sampling process. In practice this mismatch behaves like an off-policy update and is markedly less stable under sparse rewards. We also include a controlled ablation (Sec. 5.2) that keeps the sampling process identical (roll out with hint) but changes the conditioning.

##### 4.3. Policy-dependent scheduling

We control hint informativeness with a discrete strength variable ℓ ∈ {0,1,...,L}, where ℓ = 0 corresponds to the deployable no-hint setting. The scheduler is policy-dependent in the sense that it adapts ℓ using statistics collected from recent rollouts under the policy πθ (stop-gradient). Intuitively, we increase ℓ only when the current policy provides insufficient learning signal on a prompt.

##### 4.2. Online self-hinting

Scheme 1 (SAGE-LIGHT): epoch-level accuracy threshold. Let pˆt−1(x) denote the empirical success rate of prompt x measured in the previous epoch using rollouts from policy model with current hint level. Given a target threshold α ∈ (0,1), we increase hint strength when the prompt is too

In the algorithm, we can produce privileged hints in two ways: (1) Offline hints (fixed). A fixed hint generator (e.g., extracted once from τ⋆) is simple, but it does not adapt to the learner and can become miscalibrated over training. (2) Online hints. We periodically refresh the hint generator using a copy of the current policy πθ.

Algorithm 1 SAGE / SAGE-LIGHT: Self-hint Aligned GRPO with Privileged Supervision

Require: Training set D = {(x, τ⋆)}, policy model πθ, group size G, KL weight β, stabilizer ϵ > 0, max hint level L, reference policy

πref, hint generator qϕ(h | x, τ⋆, ℓ) based on πθ, threshold α (SAGE-LIGHT only).

- 1: Initialize policy parameters θ; initialize per-prompt level map ℓ(x) ← 0 for all x ∈ D.
- 2: Repeat for epochs:
- 3: for each minibatch {(xb, τb⋆)}Bb=1:
- 4: for b = 1, . . . , B do
- 5: if SAGE-LIGHT then
- 6: if epoch > 1 and R¯b < α then ℓ(xb) ← min{ℓ(xb) + 1, L}
- 7: Sample hb ∼ qϕ(h | xb, τb⋆, ℓ(xb))
- 8: else (SAGE)
- 9: for ℓ = 0, . . . , L do
- 10: Sample h˜b ∼ qϕ(h | xb, τb⋆, ℓ)
- 11: Sample τ˜b,i ∼ πθ(· | xb, h˜b) for i = 1, . . . , G and compute R˜b,i ← R(xb, τ˜b,i)
- 12: if Gi=1 R˜b,i > 0 or ℓ = L then
- 13: hb ← h˜b, τb,i ← τ˜b,i, Rb,i ← R˜b,i; break
- 14: (If SAGE-LIGHT) Sample τb,i ∼ πθ(· | xb, hb) for i = 1, . . . , G and compute Rb,i ← R(xb, τb,i).
- 15: Compute R¯b ← G1 Gi=1 Rb,i, sb ← G1 Gi=1(Rb,i − R¯b)2, Ab,i ← Rb,is −R¯b

b+ϵ .

- 16: Lpg ← −BG1 Bb=1

G i=1 Ab,i Tt=1b,i log πθ(yb,i,t | xb, hb, yb,i,<t)

- 17: Lkl ← B1 Bb=1 KL(πθ(· | xb, hb) ∥ πref(· | xb, hb))

- 18: Update θ ← θ − η ∇θ Lpg + β Lkl
- 19: Deployment: set ℓ = 0 so h = ∅, and run πθ(· | x).

hard:

ℓt(x) = min ℓt−1(x) + 1, L if pˆt−1(x) < α, (12)

and otherwise keep ℓt(x) = ℓt−1(x). Thus, hints are activated only when success is consistently rare.

Scheme 2 (SAGE): group-degeneracy trigger. GRPO requires within-group outcome differences to produce a nonzero update. We therefore use a more local trigger based on whether a probe group contains any positive sample. For a small probe group {τi}Gi=1 rolled out from πθ(· | x,h) at the current strength ℓt−1(x), define z(x) = I Gi=1 R(x,τi) = 0 , i.e., z(x) = 1 when the group has no positive rollouts. We then increase hint strength only on such collapsed prompts:

ℓt(x) = min ℓt−1(x) + 1, L if z(x) = 1, (13)

and otherwise keep ℓt(x) = ℓt−1(x). This rule targets the specific finite-sample pathology of sparse rewards: when a group contains no positives, standardized GRPO advantages collapse and the policy-gradient estimator vanishes.

Discussion. SAGE-LIGHT (Scheme 1) is compute-efficient because it updates the hint strength only at the epoch level (no additional computation for rollouts), but it can react slowly to sudden reward collapse. SAGE (Scheme 2) is more reactive and directly targets GRPO’s failure mode via a no-positives trigger, at the cost of additional probe rollouts. We report results for both schemes, and find that the no-positives trigger typically recovers faster from stalled training on hard prompts and yields better performance.

Overall, Algorithm 1 summarizes SAGE implementation. Each prompt draws one hint per epoch (given ℓ), and all G rollouts for that prompt share the sam context. This design reduces unnecessary variance from hint.

### 5. Empirical Results

Models. We use LLMs with varying degrees of math specialization: Llama-3.2-3B-Instruct (Meta, 2024), Qwen2.57B-Instruct (Yang et al., 2024), and Qwen3-4B-Instruct2507 (Yang et al., 2025), representing low, moderate, and high levels of math-focused optimization, respectively, with the latter trained extensively via RL.

Training set. Our training data are drawn from OpenR1Math-220k (Hugging Face, 2025), using prompts from NuminaMath 1.5 (Li et al., 2024) and reasoning traces generated by DeepSeek-R1 (DeepSeek-AI et al., 2025). The initial dataset contains 94k prompts. To ensure answer verifiability, we apply the Math-Verify tool (Kydl´ıˇcek) to remove prompts whose DeepSeek-R1 reasoning traces are incorrectly verified, resulting in 64k prompts. These prompts are used in Figure 2. Due to limited resources, we further subsample 15k prompts from this set, restricting the corresponding DeepSeek-R1 reasoning traces to fewer than 8,192 tokens. This constraint is necessary because one of our baselines, LUFFY (Yan et al., 2025), relies on these reasoning traces, and excessively long traces would significantly increase RL training time. As we do not filter prompts based on pass rate, the resulting 15k prompts span a wide range of difficulty levels, resembling a practical RL training dataset.

- Table 1. Accuracy on in-distribution and out-of-distribution tasks across three LLMs. The best and second-best results are in bold and underlined, respectively. SAGE and SAGE-LIGHT consistently outperform baselines on average across various LLMs.

In-distribution Out-of-distribution

Method

AIME24 / 25 AMC23 MATH-500 Minerva Olympiad Avg. ∆ GPQA MMLU-Pro Avg. ∆ Llama-3.2-3B-Instruct 6.5 / 0.6 22.8 44.7 17.8 14.2 17.8 0 17.9 27.0 22.5 0 SFT 0.4 / 0.6 9.5 26.9 5.1 6.5 8.2 −9.6 11.6 18.8 15.2 −7.3 GRPO 6.7 / 0.8 29.5 52.1 20.5 21.8 21.9 +4.1 26.3 39.8 33.1 +10.6 LUFFY 4.4 / 0.4 18.6 38.9 14.3 11.9 14.7 −3.1 16.0 26.7 21.4 −1.1 Scaf-GRPO 7.7 / 2.3 28.8 51.7 19.4 19.5 21.5 +3.7 24.1 38.0 31.0 +8.5 SAGE-LIGHT 8.8 / 1.9 32.2 54.1 20.8 20.1 23.0 +5.2 26.8 39.6 33.2 +10.7 SAGE 9.2 / 0.8 34.7 56.3 20.1 22.0 23.9 +6.1 27.3 40.7 34.0 +11.5 Qwen2.5-7B-Instruct 13.8 / 6.7 53.4 75.7 38.1 39.2 37.8 0 37.1 56.4 46.7 0 SFT 3.5 / 7.1 30.9 56.2 20.0 21.7 23.2 −14.6 9.5 35.6 22.5 −24.2 GRPO 15.0 / 13.5 55.5 79.2 39.1 44.5 41.1 +3.3 37.2 57.6 47.4 +0.7 LUFFY 17.1 / 13.5 55.2 81.3 39.0 44.2 41.7 +3.9 38.1 59.1 48.6 +1.9 Scaf-GRPO 14.6 / 12.7 58.8 78.0 39.8 42.0 41.0 +2.2 36.6 58.4 47.5 +0.8 SAGE-LIGHT 17.1 / 11.7 58.1 79.9 38.6 46.1 41.9 +4.1 36.6 58.8 47.7 +1.0 SAGE 16.0 / 12.5 60.3 80.0 39.3 45.9 42.3 +4.5 38.0 59.3 48.6 +1.9 Qwen3-4B-Instruct 52.1 / 43.1 92.2 93.6 46.1 67.7 65.8 0 57.6 70.9 64.3 0 SFT 14.4 / 22.1 55.2 78.9 38.1 40.2 41.5 −24.3 29.4 52.6 41.0 −23.3 GRPO 55.8 / 45.0 95.0 96.0 50.1 70.4 68.7 +2.9 57.0 72.0 64.5 +0.2 LUFFY 42.3 / 36.0 86.4 91.1 48.2 59.4 60.6 −5.2 31.9 46.6 39.3 −25.0 Scaf-GRPO 59.8 / 45.2 92.2 95.1 48.9 69.8 68.5 +2.7 54.3 72.1 63.2 −1.1 SAGE-LIGHT 59.2 / 47.1 92.2 95.1 49.5 70.5 68.9 +3.1 57.1 72.0 64.5 +0.2 SAGE 58.1 / 52.1 94.2 95.4 49.2 71.2 70.0 +4.2 57.8 72.5 65.2 +0.9

Evaluation sets. We primarily evaluate our models on six widely used mathematical benchmarks: AIME24 (MAA Committees, 2024), AIME25 (MAA Committees, 2025), AMC23 (Li et al., 2024), MATH-500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). In addition, we include two non-mathematical benchmarks, GPQA-diamond (Rein

- et al., 2024) and MMLU-Pro (Wang et al., 2024), to assess the generalization ability of the trained models. Notably, we only use hint during training. The prompt alone is the input to an LLM for evaluation.

Baselines. We compare SAGE with the following baselines: (1) Supervised Fine-Tuning (SFT), which finetunes the model on reasoning traces from DeepSeek-R1; (2) GRPO (Shao et al., 2024), which learns without any hints; (3) LUFFY (Yan et al., 2025), which replaces one on-policy trajectory with the corresponding correct trajectory from DeepSeek-R1; and (4) Scaf-GRPO (Zhang et al., 2025b), which incorporates hints generated by GPT-5.2 under a lowreasoning-effort setting. Notably, SFT, LUFFY and ScafGRPO all rely on a stronger external LLM, whereas SAGE learns only from self-generated hints. For fair comparison, we reproduce LUFFY and Scaf-GRPO using their opensource implementations on the same 15k sampled prompts, aligning only the batch size and number of training steps.

Implementation details. We run all experiments on 8 A100 GPUs, and use verl (Sheng et al., 2025) for training and vLLM (Kwon et al., 2023) for sampling. Following DAPO (Yu et al., 2025), we disable the KL term by setting β = 0, and apply asymmetric clipping with ϵlow = 0.2 and ϵhigh =

0.28. Unless otherwise specified, the maximum response length is set to 8096 for both training and evaluation,1 with a batch size of 128, 8 trajectories per prompt,2 and 500 training steps in total. We evaluate every 50 steps, and report the best average accuracy over all checkpoints. We set L as 3, and α = 0.35 for SAGE-LIGHT. Details of the prompt used for hint generation and injection are provided in Appendix B. Complete training and evaluation settings for all methods are reported in Appendix C.

##### 5.1. Main results

We report results for all methods and LLMs on eight benchmarks in Table 1, with corresponding training dynamics shown in Figure 4. Across three base models, SAGE consistently achieves the highest average performance among all baselines, yielding improvements of +6.1 (Llama-3.2), +4.5 (Qwen2.5), and +4.2 (Qwen3) on average across the benchmarks. We use a fixed training set for all LLMs, despite their differing degrees of optimization for mathematical tasks. Consequently, the training set is relatively easier for Qwen3 and more challenging for Llama, a discrepancy that is also reflected in Figure 2. Nevertheless, SAGE consistently improves performance across all LLMs, demonstrating robust and effective generalization.

SAGE vs. SFT. SFT yields the worst performance, underperforming even the base LLM, due to its tendency to overfit

- 1We use 8096 for the main results, as required by LUFFY, and

2048 for the remaining experiments due to resource constraints.

- 2We use 4 trajectories for Qwen3-4B-Instruct due to slower

training caused by its long response length.

Llama-3.2-3B-Instruct

Qwen2.5-7B-Instruct

Qwen3-4B-Instruct

0.4

Trainingrewards

0.9

0.3

0.6

GRPO

GRPO

GRPO

LUFFY

LUFFY

LUFFY

0.2

0.8

Scaf-GRPO

Scaf-GRPO

Scaf-GRPO

0.4

SAGE

SAGE

SAGE

0.1

0.7

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Responselength

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

4000

6000

4000

3000

4000

2000

3000

2000

1000

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

6

0.4

0.4

Entropy

4

0.2

0.2

2

0

0.0

0 100 200 300 400 500

0 100 200 300 400 500

0 100 200 300 400 500

Step

- Figure 4. The training dynamics of different methods. For the training rewards, one should focus on the trend instead of the value, since adding hint (SAGE and Scaf-GRPO) modifies the prompt difficulty, and using a correct off-policy trajectory (LUFFY) increases the reward. (1) LUFFY shows the most instability, with a very high entropy for Llama and a very low reward at the beginning of training for Qwen3, because it imitates the off-policy trajectory whose distribution might not be aligned with the policy model. (2) Scaf-GRPO shows the lowest entropy, implying less exploration. (3) SAGE retains the on-policy characteristic, has a mild entropy and shows a stable growth in response length, which normally implies a better reasoning pattern.

- Table 2. Percentage of prompts without any training signal, i.e., not any correct trajectories of these prompts are sampled during the whole training procedure.

Method Llama-3.2-3B Qwen2.5-7B Qwen3-4B Base 56.9% 29.8% 21.8% GRPO 40.2% 10.3% 1.3% SAGE 30.0% 8.2% 1.0%

training data. In contrast, SAGE preserves the RL characteristics, selectively sharpening the model’s distribution to correct trajectories.

SAGE vs. GRPO. Table 2 reports the proportion of prompts that never provide a training signal. Compared to GRPO, SAGE makes substantially more effective use of the prompt set. This effect is particularly pronounced for the weaker LLM, Llama-3.2: by leveraging self-generated hints, SAGE successfully utilizes 10% more prompts, leading to the largest performance improvement over GRPO (+2.0). For the stronger model, Qwen3, SAGE behaves more similarly to GRPO, with nearly identical prompt utilization. Nevertheless, despite using only 0.3% more prompts, SAGE still achieves a +1.3 accuracy gain over GRPO. These hard prompts play a critical role in RL, which aligns with prior work (Xiong et al., 2025b; Yu et al., 2025) that favors RL on prompt sets with lower pass rates. Furthermore, in Figure 4, SAGE exhibits faster response-length growth than GRPO for both Llama-3.2 and Qwen2.5, due to learning from hard prompts that fail to provide any signal under GRPO.

SAGE vs. LUFFY. LUFFY exhibits the second-largest degree of off-policy behavior, following SFT, as one of its trajectories is generated by a different model. In Figure 4, the response length increases dramatically at the early stage of training, reflecting the LLM’s tendency to imitate the stronger model. However, this off-policy setting introduces training instability due to the misalignment between the policy model and the stronger model. Specifically, Llama-3.2 trained with LUFFY displays excessively high entropy and highly oscillatory response lengths, while Qwen3 trained with LUFFY suffers from very low rewards at the beginning of training. As reported in Table 1, LUFFY only outperforms GRPO (while still underperforming SAGE) on Qwen2.5, and performs worse than the base model on both Llama-3.2 and Qwen3.

SAGE vs. Scaf-GRPO. Scaf-GRPO relies on hints generated by a stronger model (e.g., GPT-5.2 in our setting). As shown in Figure 4, it exhibits the lowest entropy among all methods, indicating limited exploration. This behavior may stem from the hints revealing excessive information. In contrast, SAGE maintains an entropy level comparable to GRPO,3 while consistently outperforming Scaf-GRPO in Table 1. Moreover, learning from self-generated hints enables a more end-to-end training procedure and simplifies implementation.

SAGE vs, SAGE-LIGHT. SAGE-LIGHT achieves slightly lower accuracy but improves efficiency, requiring 53% of

3The entropy of GRPO on Llama-3.2 is abnormally high.

Table 3. Training time on Qwen2.5-7B-Instruct.

GRPO LUFFY Scaf-GRPO SAGE SAGE-LIGHT 1.0× (25.3h) 1.2× 1.5× 2.3× 1.2×

No hint, l=0 w/o training SAGE

60

59.2

Averageaccuracy(%)

58.3

58

56.3 56.5

56

55.4

54.1

54

53.3

52

50

ofline l=2 w/ moreSAGEoff-policy hints

onlinel=2 offlinel=2

- Figure 5. Ablation studies on Qwen3-4B-Instruct trained with the same prompt set as Figure 3.

SAGE’s training time.

Out-of-distribution performance. In Table 1, performance on out-of-distribution benchmarks shows a similar pattern to that on in-distribution ones. SAGE and SAGE-LIGHT consistently achieve the best and second-best accuracy, respectively, indicating superior generalization capability.

##### 5.2. Discussion

Latency. A potential limitation of SAGE is its latency, as it must generate and use hints on the fly when a correct trajectory of the prompt can’t be sampled. Table 3 reports the training time of different RL methods. Among them, SAGE incurs the highest training cost, while SAGE-LIGHT requires only slightly more time than GRPO. For highly complex prompts, SAGE may sample hints across multiple levels (from l = 0 to l = 3), which increases computational overhead. In contrast, SAGE-LIGHT leverages the prompt accuracy from the previous epoch to select an appropriate hint level, and thus samples from only a single level. These two SAGE variants provide flexible trade-offs for practitioners with different efficiency requirements, and both consistently outperform the baseline methods.

Offline with more hints. In Figure 3, online self-hinting generates a new hint at each training step, whereas offline self-hinting relies on a fixed hint generated prior to training. One possible explanation for the superior performance of online self-hinting is the increased diversity of hints. To examine this, we have an additional ablation (Figure 5): offline self-hinting with multiple hints. Specifically, before training, we use the base LLM to generate 10 hints with temperature=1.0. During training, a different hint

use hint

40

- level l = 1

- level l = 2

- level l = 3

30

#Prompts

20

10

0

0 100 200 300 400 500

Step

Figure 6. Number of prompts uses hint during training on Llama3.2-3B-Instruct. Batch size is 128. The model use less hint w.r.t. the training step, indicating that the model becomes more powerful.

is used for the same prompt at each step, ensuring that identical prompts are paired with diverse hints over time. We observe that increased hint diversity indeed improves performance, yielding a +0.9 gain over standard offline selfhinting. Nevertheless, online self-hinting still outperforms this variant by a margin of 2.0. We argue that hint generated in an online manner offer more benefit than diversity.

Off-policy. We ablate whether the policy-gradient matches the sampling context: SAGE optimizes log πθ(· | x,h), while an off-policy variant rolls out with h but optimizes log πθ(· | x). Figure 5 shows a clear drop for the off-policy variant (56.5) compared with on-policy SAGE (59.2) and even the single level hint level baseline (58.3).

Same level hint vs. SAGE. SAGE does not rely on a fixed hint level. Instead, it adaptively increases the hint level only when the current level fails to yield a correct response. In Figure 5, this strategy leads to a clear performance gain over using a constant hint level (e.g., online l = 2), achieving an improvement of +0.9. This design enables more effective utilization of the hard prompt, allowing the LLM to progressively learn from weaker hints, down to l = 0.

Less hint w.r.t. step. In Figure 6, we can observe that the LLM use less and less hint during the training. It indicates that self-hinting indeed enhances RL. LLM becomes more and more powerful and can gradually solve difficult problems without the help of a hint.

Case study. An example on how hint helps (Appendix A).

### 6. Related Work

Data resampling and external guidance. Data selection and filtering are widely used in online RL for LLMs (Zhang et al.; Dong et al., 2023; Xiong et al., 2023; Dong et al., 2024; Shi et al., 2024; Liao et al., 2025; Feng et al., 2025),

and become particularly important for GRPO-style methods where groupwise advantages can collapse under sparse rewards. Prior work mitigates this issue mainly by reshaping the training distribution or injecting external guidance. A common workaround is to skip degenerate groups and resample or upweight prompts (Yu et al., 2025; Xiong et al., 2025a; Yao et al., 2025; Li et al., 2025), which improves efficiency but biases training toward prompts with non-trivial success probability. Another direction bootstraps learning by adding positive trajectories from stronger teachers, reference models, or offline buffers (Zhang et al., 2025a; Yan

- et al., 2025), but this can introduce context or distribution mismatch when mixed with on-policy rollouts. In contrast, SAGE preserves a clean on-policy objective by changing the rollout distribution through privileged hinting, without discarding hard prompts or relying on static external buffers.

Privileged hinting and SAGE. While leveraging intermediate guidance, such as plans or gold solutions, has a rich history in RL (Ng et al., 1999; Szepesv´ari, 2022; Ouyang et al., 2022), recent LLM-specific adaptations often implement hinting via heuristic “batch surgery.” For instance, Zhang et al. (2025b) mitigates collapse by augmenting rollout batches with hinted trajectories upon detecting failure. This approach mixes contexts (e.g., x and x,h ) within a single group, which blurs the interpretation of groupwise baselines and advantage normalization. Furthermore, external hint generators may not be calibrated to the learner’s current capabilities. SAGE distinguishes itself through three key design choices: (1) it formalizes hinting as an explicitly augmented on-policy sampling process, ensuring the GRPO loss remains well-defined; (2) it employs a policydependent strength scheduler that activates hints only when the ”learning gate” is closed; and (3) it utilizes online selfhinting via a lagged policy, ensuring the hint distribution tracks the learner’s evolving support rather than relying on a potentially misaligned external teacher.

### Acknowledgements

This research was partly supported by the Netherlands Organization for Scientific Research (NWO) under project number VI.C.192.080.

### Impact Statements

This work can reduce training cost and improve stability of RL for LLMs on verifiable tasks. Risks include misuse to optimize harmful verifiable objectives.

### References

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X.,

- Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen,

- Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen,
- R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye,
- S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen,

- X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang,

- X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang,

- X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang,
- Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y.,

Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang,

- Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

### 7. Conclusion

We identify a finite-sample degeneracy in GRPO under sparse 0/1 rewards. When group rewards are identical, advantage standardization collapse, and the minibatch gradient vanishes on hard prompts. We propose SAGE, a privileged procedural hinting method that injects reference-solutionderived hints during training to shift the rollout distribution while preserving the original reward definition. A policydependent schedule gates hint strength based on detected group collapse, and inference uses the no-hint policy. Extensive experiments validate the improvements across tasks.

Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W., Pan, R., Diao, S., Zhang, J., SHUM, K., and Zhang, T. RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learn-

ing Research, 2023. ISSN 2835-8856. URL https: //openreview.net/forum?id=m7p5O7zblY.

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Feng, Y., Kwiatkowski, A., Zheng, K., Kempe, J., and Duan, Y. Pilaf: Optimal human preference sampling for reward modeling. arXiv preprint arXiv:2502.04270, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hu, J., Wu, X., Zhu, Z., Xianyu, Wang, W., Zhang, D., and Cao, Y. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github. com/huggingface/open-r1.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Kydl´ıˇcek, H. Math-Verify: Math Verification Library. URL https://github.com/huggingface/ math-verify.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

Li, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S., Rasul, K., Yu, L., Jiang, A. Q., Shen, Z., et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

Li, Z., Chen, C., Yang, T., Ding, T., Sun, R., Zhang, G., Huang, W., and Luo, Z.-Q. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation. arXiv preprint arXiv:2509.25849, 2025.

Liao, B., Xu, Y., Dong, H., Li, J., Monz, C., Savarese, S., Sahoo, D., and Xiong, C. Reward-guided speculative decoding for efficient llm reasoning. arXiv preprint arXiv:2501.19324, 2025.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

MAA Committees. AIME Problems and Solutions. https://artofproblemsolving.com/wiki/ index.php/AIME_Problems_and_Solutions,

- 2024.

MAA Committees. AIME Problems and Solutions. https://artofproblemsolving.com/wiki/ index.php/AIME_Problems_and_Solutions,

- 2025.

Meta. Introducing meta llama 3: The most capable openly available llm to date. Meta AI Blog, 2024. https: //ai.meta.com/blog/meta-llama-3/.

Ng, A. Y., Harada, D., and Russell, S. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml, volume 99, pp. 278–287. Citeseer, 1999.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279– 1297, 2025.

Shi, R., Zhou, R., and Du, S. S. The crucial role of samplers in online direct preference optimization. arXiv preprint arXiv:2409.19605, 2024.

Szepesv´ari, C. Algorithms for reinforcement learning. Springer nature, 2022.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. 2023.

Xiong, W., Yao, J., Xu, Y., Pang, B., Wang, L., Sahoo, D., Li, J., Jiang, N., Zhang, T., Xiong, C., et al. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343, 2025a.

Xiong, W., Ye, C., Liao, B., Dong, H., Xu, X., Monz, C., Bian, J., Jiang, N., and Zhang, T. Reinforce-ada: An adaptive sampling framework under non-linear rl objectives. arXiv preprint arXiv:2510.04996, 2025b.

- Yan, J., Li, Y., Hu, Z., Wang, Z., Cui, G., Qu, X., Cheng, Y., and Zhang, Y. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu,

- B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

- Yao, J., Hao, Y., Zhang, H., Dong, H., Xiong, W., Jiang, N., and Zhang, T. Optimizing chain-of-thought reasoners via gradient variance minimization in rejection sampling and rl. arXiv preprint arXiv:2505.02391, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Fan, T., Liu, G., Liu, L., Liu, X., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zhang, C., Shen, W., Zhao, L., Zhang, X., Qi, L., Dou, W., and Bian, J. Policy filtration in rlhf to fine-tune llm for code generation.

Zhang, K., Lv, A., Li, J., Wang, Y., Wang, F., Hu, H., and Yan, R. Stephint: Multi-level stepwise hints enhance reinforcement learning to reason. arXiv preprint arXiv:2507.02841, 2025a.

Zhang, X., Wu, S., Zhu, Y., Tan, H., Yu, S., He, Z., and Jia, J. Scaf-grpo: Scaffolded group relative policy optimization for enhancing llm reasoning. arXiv preprint arXiv:2510.19807, 2025b.

Zhang, Y., Yao, W., Yu, C., Liu, Y., Yin, Q., Yin, B., Yun, H., and Li, L. Improving sampling efficiency in rlvr through adaptive rollout and response reuse. arXiv preprint arXiv:2509.25808, 2025c.

### A. Illustration of Privileged Hinting

Prompt

Vanilla Response ( )

Find the sum of all integer bases for which is a divisor of .

We know that in base and

.

… We can conclude that there are no integer values of

Hint

for which divides .

Model

Model

Solution

Response with hint ( )

Convert and to b-based values (like something of the form

In base , , We need .Note: So iff . Let . Then is a positive divisor of , and since

). Then use the fact that if divides , it also divides any linear combination of these two expressions that cancels the b-term.

To solve the problem, we need to convert the numbers

and from base to base 10, and then use the given divisibility condition to find the sum of all integer bases for which is a divisor of .

|Model<br><br>| |
|---|---|
| | |

Divisors of greater than are and . Thus: , Sum of all such bases: \(21+49 = 70.\) Answer:

… The final answer is:

| |
|---|

.

| |
|---|

.

Figure A.1. Example of privileged hinting for a single prompt. Given a math prompt x, a hint generator qϕ uses the reference solution τ⋆ during training to produce a procedural hint (h) that summarizes intermediate reasoning without revealing the final answer. Rolling out the policy πθ on the original prompt can yield an incorrect solution with zero reward, while conditioning on (x, h) shifts the rollout distribution and enables a correct solution with positive reward. The task return is unchanged, and at deployment the hint is removed so the model runs on the original prompt only.

Case Study: Progressive privileged hints for 17b | 97b

Prompt. Find the sum of all integer bases b > 9 for which 17b is a divisor of 97b. Reference solution (available only during training). In base b,

17b = 1 · b + 7 = b + 7, 97b = 9 · b + 7 = 9b + 7. We need b + 7 | 9b + 7. Note that

9b + 7 = 9(b + 7) − 56,

hence b + 7 | 9b + 7 iff b + 7 | 56. Let d = b + 7. Then d is a positive divisor of 56, and since b > 9 we have d = b + 7 > 16. The divisors of 56 greater than 16 are 28 and 56, so b ∈ {21, 49} and the sum is 70.

###### Privileged hints (used only during training; never shown at test time).

- • Level 1 (minimal). Rewrite the base-b numerals as ordinary integers in terms of b, then turn the divisibility condition into a statement about a simple linear expression.
- • Level 2 (medium). Convert 17b and 97b into the form Ab + C. If (b + 7) divides (9b + 7), it also divides any linear combination of these expressions that cancels the b-term.
- • Level 3 (detailed). Compute 17b = b + 7 and 97b = 9b + 7. Subtract a multiple of (b + 7) from (9b + 7) to eliminate b, e.g., (9b + 7) − 9(b + 7). The condition becomes “(b + 7) divides a constant”. Enumerate divisors and keep only b > 9.

###### Representative model behaviors (illustrative).

- • No hint (h = ∅): the model may mis-expand 97b (e.g., treating it as a multi-digit polynomial), and incorrectly conclude there is no valid b > 9, yielding a wrong terminal decision and thus reward 0.
- • With stronger hints (Level 2/3): the model is steered toward the key cancellation step 9b + 7 − 9(b + 7) = −56, after which it can enumerate divisors and recover the correct bases and final sum (70).

Case Study: Privileged Hinting on a Simple Verifiable Math Task. To make the idea of privileged hinting concrete, we include a small case study on a verifiable divisibility question. The key point is that hints are training-time privileged context: they do not modify the verifier or the terminal reward. Instead, they reshape the rollout distribution so that, under finite sampling, the policy is more likely to generate informative trajectories (e.g., those that perform the correct algebraic reduction). In practice, without hints the model may repeatedly take an incorrect representation path (e.g., mis-expanding base-b numerals) and receive identical zero rewards across a rollout group, causing GRPO advantages to collapse. With

progressive hints, the model is guided toward the correct reduction, increasing the chance that a group contains mixed outcomes and thus yields a non-degenerate update.

This example matches the operational role of SAGE: hints do not change the verifier reward, but they increase the probability that at least one rollout in a finite group follows a useful trajectory (here, the cancellation-and-divisor-enumeration path). This increases within-group outcome diversity, reduces the frequency of degenerate all-zero groups, and therefore prevents standardized GRPO advantages from collapsing on hard prompts under finite sampling.

### B. Prompt for hint generation and the usage of hint

System prompt for hint generation

You are a tutoring assistant that generates progressive hints to help students solve difficult problems without revealing the solution directly.

TASK: Given a question and its solution, generate 3 levels of hints that progressively guide the student toward solving the problem independently.

HINT LEVELS:

- - Level 1: Minimal hint - Points to the key concept or approach without specifics
- - Level 2: Medium hint - Provides more direction on the method or intermediate steps
- - Level 3: Detailed hint - Gives substantial guidance while still requiring the student to complete the solution GUIDELINES:
- - Never reveal the final answer
- - Each level should be built on the previous one
- - Hints should inspire problem-solving, not just provide steps to copy
- - Tailor hint difficulty to bridge the gap between the student’s level and the solution OUTPUT FORMAT:

‘‘‘json {

- "level_1": "minimal hint text",
- "level_2": "medium hint text",
- "level_3": "detailed hint text"

} ‘‘‘

User prompt for hint generation Question: {problem} Solution: {solution}

System prompt for RL Please reason step by step, and put your final answer within \\boxed{}.

User prompt for RL {problem} Here is a hint to help you: {hint}

### C. Detailed implementation settings

##### C.1. Training settings

SFT. We use OpenRLHF (Hu et al., 2024) for SFT, and set the learning rate as 5e-5, the batch size as 64, warmup ratio as 10%, and number of epochs as 3. We evaluate on the final checkpoints.

GRPO. GRPO shares the same training settings as SAGE, as stated in §5.

LUFFY. We use the open-source implementation4 to reproduce LUFFY, and set the batch size as 128, and ppo mini batch size as 64. These two hyper-parameters stay the same for all RL methods.

Scaf-GRPO. We use the open-source implementation5 to reproduce Scaf-GRPO.

C.2. Evaluation settings

For the main results in Table 1, we evaluate all models with a max response length of 8192, temperature=0.6 and top p=0.95. For the rest, we set a max response length of 2048.

Table C.1. Detailed number for Figure 3 (Left). By default, we train for 200 steps, but for 400 steps for methods denoted by ∗. The results for level l = 2 in Figure 3 are from methods denoted by ∗. Different from the default training setting of SAGE, we set ppo mini batch size=32 here.

Method Hint level AIME24 AIME25 AMC23 MATH-500 Minerva Olympiad Avg. Qwen3-4B-Instruct - 30.8 28.5 75.5 87.8 42.8 54.3 53.3 No hint 0 34.0 29.2 78.8 88.6 40.7 53.1 54.1

- GPT-5 hint 1 34.0 29.8 83.0 89.6 42.2 56.9 55.9

- Self-hint offline 1 35.8 28.3 80.2 89.9 42.4 56.9 55.6

- Self-hint online 1 35.0 28.5 82.7 90.0 42.7 61.0 56.7

GPT-5 hint 2 34.8 29.6 81.1 89.6 42.9 60.9 56.5 Self-hint offline 2 33.1 27.3 78.1 89.0 42.5 59.2 54.9

- Self-hint online 2 34.2 27.7 81.7 89.2 43.5 61.8 56.4

GPT-5 hint∗ 2 34.8 29.6 81.1 89.6 42.9 60.9 56.5 Self-hint offline∗ 2 34.0 26.0 79.8 88.5 43.7 60.4 55.4 Self-hint online∗ 2 38.5 30.8 82.7 90.3 44.4 62.9 58.3

GPT-5 hint 3 37.9 26.9 78.3 89.2 43.1 62.1 56.3 Self-hint offline 3 36.7 27.5 83.1 89.6 43.3 61.5 57.0

- Self-hint online 3 36.7 27.2 83.1 89.9 43.2 62.6 57.1

#### D. Proofs This appendix provides detailed proofs for the results in Section 3.1 (and additional analysis).

- 4https://github.com/ElliottYan/LUFFY
- 5https://github.com/JIA-Lab-research/Scaf-GRPO

- D.1. Preliminaries: Bernoulli groups and sample variance Fix a context (x,h) and draw G ≥ 2 i.i.d. rollouts with terminal rewards Ri ∈ {0,1}. Let

R¯ :=

G

1 G

1 G

Ri, s2 :=

i=1

G

√

(Ri − R¯)2, s :=

s2.

i=1

When standardized GRPO is used, advantages are

Ri − R¯ s + ϵ

Ai :=

,

where ϵ > 0 is a numerical stabilizer (as used in the main text). We will repeatedly use the fact that since Ri ∈ {0,1},

G

(Ri − R¯)2 =

i=1

G

Ri2 − GR¯2 =

i=1

G

Ri − GR¯2 = GR¯ − GR¯2 = GR¯(1 − R¯), (14)

i=1

hence

s2 = R¯(1 − R¯). (15) In particular, s2 = 0 iff R¯ ∈ {0,1}, i.e., iff all Ri are identical.

- D.2. Proof of Corollary 3.1 Proof. Recall the advantage energy

E :=

1 G

G

i=1

A2i, Ai =

Ri − R¯ s + ϵ

.

For ϵ > 0, we compute

E =

1 G

G

i=1

(Ri − R¯)2 (s + ϵ)2

=

1 (s + ϵ)2 ·

1 G

G

i=1

(Ri − R¯)2 =

s2 (s + ϵ)2

,

which is exactly Eq. (6). Since s ≥ 0 and ϵ > 0, we have 0 ≤ s+sϵ < 1, hence

0 ≤ E =

s s + ϵ

2

< 1,

so E ∈ [0,1) (and in the limit ϵ → 0+, E → I[s > 0]). Monotonicity in s follows by differentiation: define f(s) := s

2

(s+ϵ)2 for s ≥ 0. Then

f′(s) =

2s(s + ϵ)2 − s2 · 2(s + ϵ) (s + ϵ)4

=

2sϵ (s + ϵ)3 ≥ 0,

so E is non-decreasing in s. Finally, if the group is degenerate, then s = 0 and therefore E = 0. This shows the standardized signal energy collapses to 0 exactly when within-group variance collapses.

| |
|---|

- D.3. Proof of Proposition 3.2 Proof. Write p := pθ(x,h) = Pr[R(x,τ) = 1 | x,h]. Then R1,...,RG are i.i.d. Bernoulli(p).

As noted in Section D.1, s = 0 iff all rewards are identical. There are exactly two degenerate cases: (i) all-zero: R1 = ··· = RG = 0; (ii) all-one: R1 = ··· = RG = 1. Thus

###### Pr[s > 0 | x,h] = 1 − Pr[all-zero] − Pr[all-one] = 1 − (1 − p)G − pG,

which proves Eq. (7). To locate the maximizer, define u(p) := 1 − (1 − p)G − pG on [0,1]. We have symmetry u(p) = u(1 − p). Moreover, for G ≥ 2,

###### u′′(p) = − G(G − 1) pG−2 + (1 − p)G−2 < 0,

so u is strictly concave, hence has a unique maximizer. By symmetry, the unique maximizer must be at p = 12. Finally, in the sparse regime p ≪ 1,

u(p) = 1 − (1 − p)G − pG = 1 − 1 − Gp + O(p2) − O(pG) = Gp + O(p2),

so Pr[s > 0 | x,h] ≈ Gp.

- D.4. Proof of Proposition 3.3 Proof. Fix x and G ≥ 2. Recall u(p) = 1 − (1 − p)G − pG and

| |
|---|

Jx(θ,q) = Eh∼q(·|x) u pθ(x,h) .

We first verify the claims about u. Symmetry: u(1 − p) = 1 − pG − (1 − p)G = u(p). For strict concavity, compute for G ≥ 2:

###### u′′(p) = − G(G − 1) pG−2 + (1 − p)G−2 < 0,

so u is strictly concave on [0,1]. By symmetry and strict concavity, the unique maximizer is p = 12. For a fixed θ, define the measurable function

###### vθ(h) := u(pθ(x,h)) ∈ [0,1].

Then Jx(θ,q) = Eh∼q[vθ(h)]. Over all probability distributions q(· | x) supported on the admissible hint space, the maximizers must concentrate probability mass on (essential) maximizers of vθ: indeed, since the objective is linear in q, any optimizer can be chosen to put all mass on

arg max

vθ(h) = arg max

u(pθ(x,h)).

h

h

Because u is uniquely maximized at p = 12 and is strictly decreasing as p moves away from 12 (due to strict concavity and symmetry), the maximizers of vθ(h) are exactly the calibrating hints that make pθ(x,h) as close as possible to 12 (and equal to 12 when achievable). This proves the statement that an optimal qθ⋆(· | x) places its mass on calibrating hints.

In general, pθ(x,h) changes with θ because the rollout distribution under πθ(· | x,h) changes with θ. Therefore the set of calibrating hints

Hθ⋆(x) := arg max

u(pθ(x,h))

h

typically varies with θ. Unless pθ(x,h) (hence vθ(h)) is invariant in θ for q-almost all h, a fixed distribution q that was optimal (or near-optimal) early in training will drift away from being optimal later, reducing Jx(θ,q) relative to a θ-adapted choice. This formalizes why updating the hint generator online using the policy can reduce gate-mismatch.

| |
|---|

##### D.5. Proof of the Jensen inequality in Remark 1

Proof. Fix G ≥ 2 and define u(p) = 1 − (1 − p)G − pG. We showed above that u is concave on [0,1] because u′′(p) ≤ 0. Let Z := pθ(x,h) ∈ [0,1] be the random success probability induced by sampling h ∼ q. Then Jensen’s inequality for concave u gives

Eh u(Z) ≤ u(Eh[Z]), which is exactly Eq. (11). Equality holds only when Z is almost surely constant (or when u is affine on the support, which does not happen for G ≥ 2 except in degenerate cases). This shows that, at a fixed mean success probability, additional variability across hints can only decrease the expected gate-opening frequency.

| |
|---|

##### D.6. A sharper small-p expansion of the gate probability

For completeness, we also record an exact decomposition that makes the Gp scaling explicit. Let p = Pr[R = 1 | x,h]. Then

Pr[s > 0 | x,h] = 1 − (1 − p)G − pG

G−1

G k

pk(1 − p)G−k. (16)

=

k=1

When p ≪ 1, the dominant term is k = 1:

Pr[s > 0 | x,h] = Gp(1 − p)G−1 + O(p2) ≈ Gp, and the neglected pG term is exponentially smaller in G.

##### D.7. Non-standardized GRPO signal energy

Non-standardized advantage is also widely used in GRPO-like algorithms (Liu et al., 2025), which also provide insights about the behavior.

Setup. Define non-standardized (mean-centered) advantages

G

1 G

A˜i := Ri − R,¯ R¯ =

Ri,

i=1

and define the (non-standardized) energy

G

1 G

A˜2i.

E˜ :=

i=1

Note that E˜ = s2 by definition.

- Proposition D.1 (Expected non-standardized energy under Bernoulli rewards). Conditioned on (x,h) with success probability p = pθ(x,h),

G − 1 G

E E ˜ | x,h =

p(1 − p). (17)

Proof. Let S = Gi=1 Ri so that R¯ = S/G. Using identity (14),

1 G

E˜ =

G

(Ri − R¯)2 = R¯(1 − R¯).

i=1

Taking expectation:

E[E˜] = E[R¯] − E[R¯2]. We have E[R¯] = p. Also,

E[R¯2] = Var(R¯) + (E[R¯])2 =

p(1 − p) G

1 G2

1 G2 · Gp(1 − p) + p2 =

Var(S) + p2 =

+ p2.

Therefore

E[E˜] = p −

p(1 − p) G

G − 1 G

+ p2 =

p(1 − p).

| |
|---|

- Proposition D.2 (Optimal calibrated difficulty for mean-centered updates). For fixed G ≥ 2, the right-hand side of Eq. (17) is uniquely maximized at p = 12.

Proof. Let f(p) = p(1 − p) = p − p2. Then f′(p) = 1 − 2p and f′′(p) = −2 < 0, so f is strictly concave and uniquely maximized at p = 12. The prefactor (G − 1)/G does not affect the maximizer. Remark D.3 (Hint randomness incurs a variance penalty at fixed mean). Let Z = pθ(x,h) be random due to h ∼ q and p¯ = E[Z]. Then

| |
|---|

E Z(1 − Z) = p¯(1 − p¯) − Var(Z). Thus, at fixed mean success rate, additional variability in pθ(x,h) across hints reduces the expected energy. Proof. Compute

###### E[Z(1 − Z)] = E[Z] − E[Z2] = p¯− Var(Z) + p¯2 = p¯(1 − p¯) − Var(Z).

| |
|---|

