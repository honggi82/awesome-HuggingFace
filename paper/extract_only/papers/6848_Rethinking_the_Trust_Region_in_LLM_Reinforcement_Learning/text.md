## Rethinking the Trust Region in LLM Reinforcement Learning

Penghui Qi*12 Xiangxin Zhou*1 Zichen Liu2 Tianyu Pang1 Chao Du1 Min Lin1 Wee Sun Lee2

https://github.com/sail-sg/Stable-RL

[Figure 1]

# arXiv:2602.04879v3[cs.LG]12Jun2026

Figure 1. Comparison of PPO and the proposed DPPO (the Binary-TV variant in Section 4.4). (Left) The surrogate objective and corresponding masks for PPO and DPPO. PPO (and variants like GRPO) employs a heuristic mask based on the probability ratio, which over-penalizes low-probability tokens and under-penalizes high-probability ones (Section 4.2). In contrast, DPPO utilizes a more principled mask based on a direct approximation of policy divergence (e.g., Total Variation), ensuring updates stay within a theoretically grounded trust region (Section 3). (Right) Experimental results on the AIME24 using Qwen3-30B-A3B-Base. DPPO significantly outperforms GRPO baselines, achieving superior training efficiency and stability even without rollout routing replay (R3) (Section 7).

### Abstract

inefficiency and instability. To address this, we propose Divergence Proximal Policy Optimization (DPPO), which substitutes heuristic clipping with a more principled constraint based on a direct estimate of policy divergence (e.g., Total Variation or KL). To avoid huge memory footprint, we introduce the efficient Binary and Top-K approximations to capture the essential divergence with negligible overhead. Extensive empirical evaluations demonstrate that DPPO achieves superior training stability and efficiency compared to existing methods, offering a more robust foundation for RL-based LLM fine-tuning.

Reinforcement learning (RL) has become a cornerstone for fine-tuning Large Language Models (LLMs), with Proximal Policy Optimization (PPO) serving as the de facto standard algorithm. Despite its ubiquity, we argue that the core ratio clipping mechanism in PPO is structurally illsuited for the large vocabularies inherent to LLMs. PPO constrains policy updates based on the probability ratio of sampled tokens, which serves as a noisy single-sample Monte Carlo estimate of the true policy divergence. This creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, while potentially catastrophic shifts in high-probability tokens are under-constrained, leading to training

### 1. Introduction

Reinforcement learning (RL) is a foundational paradigm for fine-tuning Large Language Models (LLMs), enabling alignment with human preferences (Ouyang et al., 2022; Rafailov et al., 2023) and complex reasoning tasks (Guo et al., 2025; Qi et al., 2025a). In practice, LLM RL is often off-policy: an inference engine samples rollouts, while a trainer engine computes gradients. Even when the two engines use the

*Equal contribution 1Sea AI Lab, Singapore 2School of Computing, National University of Singapore. Correspondence to: Wee Sun Lee <leews@comp.nus.edu.sg>, Penghui Qi <penghuiq@comp.nus.edu.sg>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

[Figure 2]

- Figure 2. The plots show numerical differences between training and inference engines for Qwen3-30B-A3B-Base with identical parameters. (Left) The token-level probability ratio (used in PPO) is highly volatile for low-probability tokens. (Right) The tokenlevel TV divergence (used in DPPO) is more stable. This highlights a key flaw of PPO’s clipping: it over-penalizes low-probability tokens, which can slow down learning; and under-penalizes highprobability tokens, which can permit large, destabilizing updates.

same model parameters, training-inference mismatch (Yao et al., 2025; Qi et al., 2025b; Zheng et al., 2025) can make their token distributions differ. Practical systems also use collected rollouts for several minibatch updates to improve throughput (Liu et al., 2025a). These choices make it essential to control how far each update moves the trained policy from the policy that generated the data.

Proximal Policy Optimization (PPO1) (Schulman et al., 2017) is the dominant algorithm in this setting because it is simple and scalable. Its main safeguard is ratio clipping. If the probability ratio between the new policy and the behavior policy becomes too large or too small on a sampled token, PPO clips the objective. This rule is meant to keep updates inside a trust region, where monotonic improvement is theoretically guaranteed (Schulman et al., 2015).

Despite its widespread adoption, we argue that PPO’s core mechanism, ratio clipping, is structurally ill-suited for the expansive, long-tailed vocabularies inherent to LLMs. Although motivated by Trust Region Policy Optimization (TRPO) (Schulman et al., 2015), which constrains KL or Total Variation (TV) divergence of policy distributions, PPO instead clips the probability ratio of the sampled token. This ratio is only a noisy, single-sample estimate of the true policy divergence. While this approximation may suffice for classical RL environments with limited action spaces, it fails in the LLM regime because the ratio depends strongly on the original token probability. For example, increasing a rare token’s probability from 10−5 to 10−3 gives a ratio of 100 and triggers clipping, although the moved probability mass is small. Conversely, decreasing a high-probability token from 0.99 to 0.8 moves much more mass, yet the ratio may remain inside a typical clipping range.

Training-inference mismatch makes this problem worse. As

1We denote PPO by its ratio-clipping loss, regardless of advantage estimation. Under this definition, GRPO is a PPO variant.

illustrated in Figure 2, the probability ratio becomes highly volatile for low-probability tokens, while TV divergence remains stable. Consequently, PPO creates a sub-optimal learning dynamic: updates to low-probability tokens are aggressively over-penalized, slowing learning, while updates to high-probability tokens are under-penalized, risking instability. This implicit bias necessitates a fundamental rethinking of the trust region approach in LLM fine-tuning to ensure both efficiency and stability.

To address this limitation, we propose Divergence Proximal Policy Optimization (DPPO), which replaces ratiobased clipping with a constraint based on policy divergence. Rather than relying on noisy single-sample ratios, DPPO directly estimates policy divergence (e.g., TV or KL divergence). To ensure memory feasibility, we introduce two efficient approximations, Binary and Top-K divergence, which capture essential distributional shifts with negligible overhead. This allows DPPO to rigorously distinguish between safe and unsafe updates, effectively resolving the problems of over- and under-constraining inherent in standard PPO.

In this work, we provide a comprehensive rethinking of the trust region in the context of LLM fine-tuning. Our contributions are threefold. Theoretical Formulation: We derive policy improvement bounds specifically tailored to the finite-horizon, undiscounted setting of LLM generation, establishing a rigorous theoretical foundation for trust-region methods in this domain. Stability and Efficiency Analysis: We isolate the primary sources of training instability to provide practical stabilization guidelines, while further highlighting the significant role that low-probability tokens play in driving exploration. Algorithmic Performance: We demonstrate that DPPO achieves superior stability and final performance compared to existing methods like GRPO, providing a robust new framework for RL-based fine-tuning.

### 2. Background

###### 2.1. Policy Performance Difference

We begin with the standard formulation of a Markov Decision Process (MDP), defined by the tuple M = (S,A,P,r,ρ0,γ), which includes the state space S, action space A, transition dynamics P(s′|s,a), reward function r(s,a), initial state distribution ρ0(s), and a discount factor γ ∈ [0,1]. A stochastic policy π(a|s) generates trajectories τ = (s0,a0,r0,s1,a1,r1,...) by sampling actions at ∼ π(·|st) and transitioning to states st+1 ∼ P(·|st,at). The central goal of RL is to find a policy that maximizes the expected discounted return:

η(π) = Eτ∼π

∞

γtrt .

t=0

To facilitate policy optimization, we define the standard value functions under a policy π: the state-value function

V π(s) = Eτ∼π ∞t=0 γtrt s0 = s , the action-value function Qπ(s,a) = Eτ∼π ∞t=0 γtrt s0 = s,a0 = a , and the advantage function Aπ(s,a) = Qπ(s,a) − V π(s). A key theoretical tool for relating the performance of two distinct policies is the policy performance difference theorem (Kakade & Langford, 2002). It states that for any two policies, a target policy (to be optimized) π and a behavior policy (for rollout) µ, their expected returns are related by:

1 1 − γ

Es∼ρπ, a∼π(·|s) Aµ(s,a) . (1)

η(π) − η(µ) =

Here, ρπ(s) = (1 − γ) ∞t=0 γt Pr(st = s|π) is the normalized discounted state-visitation distribution induced by

the policy π. This identity is fundamental, as it implies that any policy update that results in a non-negative expected advantage guarantees monotonic performance improvement, i.e., η(π) ≥ η(µ).

###### 2.2. Policy Improvement Bound

While Equation 1 provides a direct expression for policy improvement, its dependence on the state-visitation distribution ρπ of the new policy makes it intractable for direct optimization. To overcome this, Schulman et al. (2015) derive a lower bound on performance improvement that can be estimated using samples from the behavior policy µ, with a penalty term that measures the divergence between the old and new policies. This lower bound forms the basis of trust-region methods.

- Theorem 2.1. (Schulman et al., 2015; Achiam et al., 2017) Given any two policies, µ and π, the following bound holds:

π(a|s) µ(a|s)

1 1 − γ

Aµ(s,a)

Es∼ρµ, a∼µ(·|s)

η(π)−η(µ) ≥

2ξγ (1 − γ)2

DTVmax(µ∥π)2,

−

(2)

where ξ = maxs,a Aµ(s,a) and DTVmax(µ∥π) = maxs DTV µ(·|s)∥π(·|s) , which is the maximum Total Variation (TV) divergence among all states.

This bound provides a direct path to guaranteed policy improvement. The right-hand side of the inequality forms a surrogate objective that is a tight lower bound on the true performance improvement, touching the objective when π = µ. Therefore, iteratively maximizing this surrogate guarantees monotonic improvement in η(π), following the principles of the Minorize-Maximization (MM) algorithm (Hunter & Lange, 2004; Schulman et al., 2015).

###### 2.3. Trust Region Policy Optimization

The policy improvement bound in Equation (2) directly justifies a surrogate objective,

π(a|s) µ(a|s)

1 1 − γ

Aµ(s,a) . (3)

Es∼ρµ, a∼µ(a|s)

Lµ(π) =

This objective serves as a first-order approximation of the true performance improvement η(π) − η(µ), as their values and gradients match at the point of expansion π = µ (Kakade & Langford, 2002; Schulman et al., 2015; Zheng et al., 2025). Therefore, maximizing Lµ(π) within a small trust region guarantees stable and meaningful policy improvement. This insight motivates the trust-region optimization approach (Schulman et al., 2015; Xie et al., 2024), which involves maximizing Lµ(π) subject to a constraint that keeps the new policy π within a trust region around the current policy µ, thereby ensuring the validity of the approximation. Formally, this is expressed as the following constrained optimization problem:

Lµ(π), s.t. DTVmax(µ∥π) ≤ δ, (4)

max

π

where the constraint can also be applied on a KL divergence DKL, justified via Pinsker’s inequality:

DTV(µ∥π)2 ≤ 21DKL(µ∥π).

### 3. Trust Region Under LLM Regime

In this section, we adapt the trust region framework to the specific context of LLM fine-tuning. This setting differs from the classical RL paradigm in two crucial ways. First, the learning problem is structured as an undiscounted (γ = 1) episodic task with a finite horizon T, which makes the original bound in Equation (2) ill-defined, as the 1−1γ term diverges to infinity. Second, due to the sparse reward nature, advantages are often estimated at the sequence level (Shao et al., 2024), rather than on a per-token basis.

Formally, given a prompt x, a policy π (the LLM) generates a response y = (y1,...,yT) by sequentially sampling tokens. At each step t, the policy defines a conditional distribution π(yt|st) over the vocabulary A, where the state st = (x,y1,...,yt−1) consists of the prompt and previously generated tokens. The probability of the complete response is the product of these conditional probabilities: π(y|x) = Tt=1 π(yt|st). After the full response is generated, a scalar reward R(y,x) is provided. For brevity, we will omit the dependency on the initial prompt x and write the objective function as:

J (π) = Ey∼π[R(y)].

We now derive performance difference identity and policy improvement bound tailored to this regime.

- Theorem 3.1 (Performance Difference Identity for LLMs). In a finite-horizon setting (T) with no discount (γ = 1), for any two policies π and µ, the performance difference can be decomposed as:

J (π) − J (µ) = L′µ(π) − ∆(µ,π),

where L′µ(π) is a surrogate objective defined as:

 R(y)

|y|

π(yt|st) µ(yt|st) − 1

L′µ(π) = Ey∼µ

t=1

and ∆(µ,π) is an error term given by:

 , (5)

∆(µ,π) =Ey∼µ R(y) (6)

  .

 1−

|y|

T

π(yt|st) µ(yt|st) −1

π(yj|sj) µ(yj|sj)

t=1

j=t+1

This theorem provides an exact expression for the policy improvement. The surrogate L′µ(π) represents a first-order approximation, while the error term ∆ captures the higherorder effects of the policy change. To make this practical for optimization, we bound the error term.

Theorem 3.2 (Policy Improvement Bound for LLMs). In a finite-horizon setting (T) with no discount (γ = 1), let ξ = maxy |R(y)| be the maximum absolute reward, DTVmax(µ∥π) = maxs

DTV µ(·|st)∥π(·|st) be the maximum Total Variation (TV) divergence over all states, and D¯TV(µ,π) = Ey∼µ |ty=1| DTV µ(·|st)∥π(·|st) be the average token-level TV divergence. Then the policy improvement is lower-bounded by both:

t

J (π)−J (µ) ≥ L′µ(π) − 2ξT(T −1) · DTVmax(µ∥π)2, (7) J (π)−J (µ) ≥ L′µ(π) − 4ξD¯TV(µ,π). (8)

This theorem establishes lower bounds on policy improvement. The max-divergence form in Equation (7) is structurally analogous to the bound in Theorem 2.1 (see Appendix A.4), with the horizon T playing a role similar to the effective horizon 1−1γ in the discounted setting. The average-divergence form in Equation (8) is tighter for long LLM responses and more directly matches the per-token divergence control used in PPO and our algorithm. Together, they provide a clear theoretical justification for adapting the trust region approach into LLM regime. Similar to Equation (4), we can solve the following constrained optimization problem to guarantee stable learning:

L′µ(π), s.t. DTVmax(µ∥π) ≤ δ, (9) where the constraint can also be applied on a KL divergence. The proofs for Theorem 3.1 and both parts of Theorem 3.2 are deferred to Appendix A.

max

π

### 4. Methodology

###### 4.1. Proximal Policy Optimization

While theoretically appealing, the constrained optimization in TRPO requires second-order information and is difficult to scale. PPO (Schulman et al., 2017) was introduced as a first-order alternative that retains much of the stability. Owing to its simplicity and strong empirical performance, PPO has become a standard algorithm for fine-tuning LLMs.

Instead of enforcing an explicit trust-region constraint, PPO optimizes a clipped surrogate objective:

 ,

 

|y|

min rtAˆt,clip(rt,1−ϵ,1+ϵ)Aˆt

LPPOµ (π)=Ey∼µ

t=1

π(yt|st) µ(yt|st)

, (10)

rt =

where Aˆt denotes the estimated advantage at timestep t. In LLM fine-tuning, critic training is expensive and often noisy, so critic-free methods such as RLOO and GRPO are commonly used (Ahmadian et al., 2024; Shao et al., 2024; Liu et al., 2025d). A typical advantage estimate compares the reward of a sampled response against the average reward of a group of responses generated for the same prompt:

1 G

Aˆ = R(y) −

G

R(yi),

i=1

where yiGi=1 denotes the response group.

The term rtAˆt in Equation (10) is the commonly used form of the surrogate in Equation (5): replacing rt − 1 with rt leaves the gradient unchanged, while replacing R(y) with Aˆt reduces variance without biasing the expected policy gradient (see Appendix A.4). The combination of the min and clip operations then acts as an implicit trust-region constraint. Once rt is outside the interval [1 − ϵ,1 + ϵ], the clipped branch removes the incentive to move the new policy farther from the old one. The connection between this clipping rule and the formal trust region can be understood by examining the TV divergence:

- 1

- 2

DTV µ(·|st)∥π(·|st) =

t∼µ rt − 1 . (11)

Ey

From this perspective, PPO’s clipping condition, |rt−1| ≤ ϵ, can be interpreted as constraining a single-sample Monte Carlo estimate of the expected value in Equation (11). In essence, PPO enforces its trust region not on the true TV divergence, but on a noisy, single-point estimation. As we will argue next, this crude approximation is the source of significant pathologies when applied to the large, long-tailed vocabulary distributions characteristic of LLMs.

###### 4.2. Limitations of PPO Ratio Clipping

The key limitation of PPO is that whether an update is clipped depends heavily on the sampled token’s probability, rather than the true TV divergence between µ(·|st) and π(·|st). Concretely, consider a fixed state s and two tokens alow and ahigh with

µ(alow|s) = 10−4, π(alow|s) = 10−2, µ(ahigh|s) = 0.99, π(ahigh|s) = 0.80.

The probability ratio for the low-probability token is rlow = 10−2 10−4 = 100, which is far outside a typical clipping range [1−ϵ,1+ϵ] (e.g., ϵ = 0.2). PPO would thus heavily clip the

contribution of this update. In contrast, the actual contribution of this change to the TV divergence can be very small, because the total mass moved at alow is tiny. For the highprobability token, rhigh = 00..8099 ≈ 0.808, which can still lie inside the clipping range for a moderate ϵ. Yet this update removes 0.19 probability mass from the dominant token, and therefore induces a much larger contribution to DTV.

These examples highlight a structural flaw in PPO’s clipping heuristic. For low-probability tokens, an update that produces a large probability ratio is aggressively constrained, even when its impact on the TV divergence is negligible, thereby slowing training efficiency. Conversely, for highprobability tokens, an update producing a ratio close to one may go unpenalized, even when the absolute change in probability mass is large enough to cause a substantial TV divergence, which in turn risks training instability.

Connections to Existing Work The insight that PPO’s ratio clipping disproportionately penalizes low-probability tokens aligns with several prior studies. For instance, methods like Clip-Higher (Yu et al., 2025) and CISPO (Chen et al., 2025) observe that important “exploration” or “reasoning” tokens often have low initial probabilities (see Appendix D). These tokens usually get high importance ratios during policy updates and are consequently clipped, hindering the learning process. However, the solutions proposed remain heuristic and problematic. Clip-Higher suggests manually increasing the upper clipping bound, while CISPO continues to apply the gradient even for large divergence, completely ignoring the trust region. While these methods correctly identify the symptom, they fail to address the root cause: the fundamental mismatch between the single-sample probability ratio and the true distributional divergence.

###### 4.3. Divergence Proximal Policy Optimization

To address the limitations of PPO’s ratio clipping, we introduce Divergence Proximal Policy Optimization (DPPO), which directly uses a divergence-based constraint grounded in trust region theory (see Section 8.1 for a discussion on related work). Similar to PPO, we employ a dynamic mask

to block updates that would push policy outside the trust region. The DPPO objective is:

 

 ,

|y|

MtDPPO · rt · Aˆt

LDPPOµ (π) = Ey∼µ

t=1

 

(12)

- 0, if (Aˆt>0 and rt>1 and D>δ) or (Aˆt<0 and rt<1 and D>δ)
- 1, otherwise,

MtDPPO=



where D ≡ D µ(·|st)∥π(·|st) denotes the divergence (e.g., TV or KL) between the rollout and training policy distributions, and δ is a divergence threshold hyperparameter. As noted by Chen et al. (2025); Zheng et al. (2025), this objective recovers the original PPO algorithm in Equation (10) when the divergence D is replaced by |rt − 1|. Our key innovation lies in the design of this mask: instead of relying on the noisy single-sample ratio, it is conditioned on a direct measure of the distributional policy shift.

This design directly approximates the formal trust region constraint from Theorem 3.2 while preserving the beneficial asymmetric structure of PPO’s clipping. The mask only considers blocking an update if it is already moving away from the trusted region (i.e., rt > 1 for a positive advantage or rt < 1 for a negative advantage). It never blocks updates that move the policy ratio towards one (e.g., when Aˆt > 0 and rt < 1), a desirable property for accelerating learning. Unlike PPO, the final decision to block an update is based on whether the entire policy distribution has shifted too far (D > δ), not on the noisy and often misleading ratio of a single sample. This resolves the over- and under-constraining issues inherent in standard PPO. The primary remaining challenge is the overhead of calculating the full divergence D over a large vocabulary in LLMs, which we address next.

###### 4.4. Approximating Distribution Divergence

Directly computing the policy divergence is memoryprohibitive for LLMs. To make it practical, we introduce two lightweight approximations, which serve as principled lower bounds of the true divergence (see Appendix B).

Binary Approximation The binary approximation collapses the original categorical distribution into a simple Bernoulli distribution, distinguishing only between the sampled token and all other tokens. We define the new distribution as: pπt˜= π ˜(at|st), 1 − π˜(at|st) , where π˜ can be µ or π. The TV and KL divergences are then computed as:

DTVBin(t) = µ(at|st) − π(at|st) , (13) DKLBin(t) =µ(at|st)log

µ(at|st) π(at|st)

(14)

1 − µ(at|st) 1 − π(at|st)

+ (1 − µ(at|st))log

.

###### Mean of | - |

###### Max&Min of | - |

###### Rewards

###### AIME 2024

1.00

0.40

1.00

0.0200

0.95

0.75

0.0175

0.90

0.35

0.50

PG-IS

0.85

0.0150

0.25

MiniRL

0.80

0.0125

0.30

0.00

PG-TIS (CISPO)

0.75

No mismatch ( = )

0.0100

MiniRL-TIS

0.25

0.70

GRPO-ClipHigher

0.25

0.0075

0.50

0.65

DPPO-Binary-TV DPPO-Binary-KL

0.0050

0.75

0.60

0.20

1.00

0.0025

0.55

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

0 1000 2000 3000

Training Step

Training Step

Training Step

Training Step

- Figure 3. DPPO variants achieve stable training while controlling the training-inference mismatch at a low level. In contrast, methods without a trust region (PG-IS, CISPO) or with a misspecified one (MiniRL) suffer from growing mismatch and eventual collapse.

This binary divergence can be computed at negligible overhead. Crucially, it correctly distinguishes between large versus small shifts in absolute probability mass, thereby resolving the primary failure mode of PPO’s clipping.

Top-K Approximation To provide a richer and more faithful approximation of the distributional shift, the top-K variant explicitly tracks the most probable tokens. First, we define a small, representative set of tokens A′t as: A′t = TopK µ(·|st),K ∪ {at}, which includes the K

highest-probability tokens under the behavior policy, augmented with the sampled token at if it is not already present. We then form reduced categorical distributions, pµt and pπt , over the new vocabulary A′′t = A′t ∪{other}. For any token a ∈ A′t, its probability is its original probability, while all other tokens are aggregated into the “other” category:

pπt˜(a) = π˜(a|st) ∀a ∈ A′t, pπt˜(other) = 1 −

π˜(a|st),

a∈A′t

where π˜ can be µ or π. The divergence is then computed over this reduced distribution:

- 1

- 2 a∈A

pµt (a) − pπt (a) , (15)

DTVTopK(t) =

′′ t

pµt (a) pπt (a)

DKLTopK(t) =

pµt (a)log

. (16)

a∈A′′t

This approach better captures changes in the head of the policy distribution, which typically dominates the true divergence value. The overhead is minimal, making it a practical and high-fidelity choice for DPPO.

### 5. Analysis on Training Stability

The RL fine-tuning of LLMs is prone to training instability due to training-inference mismatch (see Section 8.2). In this section, we conduct an empirical study to dissect this issue and verify the stability of our DPPO algorithm. To formalize our analysis, we denote the parameters being optimized as θ and the parameters used for data generation as θ′. We aim to answer three fundamental research questions:

- 1. Given the extremely low learning rates (e.g., 10−6) common in LLM fine-tuning, is a trust region still necessary to ensure training stability?
- 2. Should the trust region be defined with respect to the

original rollout distribution (µθ′) or a recomputed policy distribution (πθ′)?

- 3. What specific types of policy updates are the primary drivers of training instability?

Experimental Setting: Our experimental setup follows the sanity test proposed by Qi et al. (2025b). We finetune DeepSeek-R1-Distill-Qwen-1.5B (Guo et al., 2025) on a curated set of 1,460 problems from the MATH dataset (Hendrycks et al., 2021). In this setting, a stable algorithm should theoretically converge to 100% training accuracy, as all problems are known to be solvable by the initial model.

We evaluate several algorithms, each representing a different approach to managing the policy update. The baselines include: PG-IS and its truncated variant PG-TIS (also known as CISPO (Chen et al., 2025)), which use standard policy gradients with token-level importance sampling; GRPO with Clip-Higher, a PPO-like algorithm where clipping is based on the rollout policy ratio rt = π

(Shao et al., 2024; Liu et al., 2025d); and MiniRL & MiniRL-TIS, a PPO variant where clipping is based on a recomputed policy ratio rt = π

θ µθ′

(Zheng et al., 2025). We compare these against DPPO, our proposed method using either binary KL or TV divergence, with the trust region defined with respect to the rollout distribution µθ′. Detailed configurations for each algorithm are provided in the Appendix C.

θ πθ′

###### 5.1. The Necessity of a Trust Region

Our first question addresses whether a trust region is redundant at low learning rates. Figure 3 provides a clear answer. The unconstrained methods, PG-IS and PG-TIS (CISPO), both suffer from an increasing training-inference mismatch, which culminates in a collapse of performance. In contrast, our DPPO variants, which enforce a principled trust region, maintain a stable, low level of mismatch throughout training and achieve near-perfect final rewards.

- Takeaway 1: A trust region is essential for stable training, even with very small learning rates. Without it, the traininginference mismatch accumulates and leads to collapse.

0 500 1000 1500 2000 2500 3000 3500

Training step

0.55

0.60

0.65

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Rewards

DPPO-KL-Rollout

DPPO-KL-Recompute

0 500 1000 1500 2000 2500 3000 3500

Training step

0.005

0.010

0.015

0.020

0.025

0.030

Mean of | - |

Figure 4. Switching the stable DPPO-KL to a decoupled objective causes the mismatch to grow and performance to collapse, confirming that the trust region must be anchored to the rollout policy.

5.2. The Correct Anchor for the Trust Region

Next, we investigate to which distribution the trust region should be anchored. A common practice in open-source implementations (Sheng et al., 2024; Zhu et al., 2025) is to use a decoupled objective (Hilton et al., 2022), where the trust region is enforced relative to a recomputed policy distribution (πθ′) instead of the original behavior policy (µθ′). The MiniRL algorithm, for example, follows this design (Zheng et al., 2025). Our results show this choice is detrimental. As in Figure 3, MiniRL fails to control the training-inference mismatch and its performance collapses, despite using a trust region. To confirm this, we created a decoupled version of our stable DPPO-KL algorithm. Figure 4 shows that this single change corrupts the stable training process, causing the mismatch to grow and performance to collapse.

- Takeaway 2: The trust region must be defined with respect

to the original behavior policy (µθ′). Using a recomputed on-policy distribution as the anchor leads to instability. This finding aligns with the theoretical bound in Equation (7) and offers a significant practical benefit: by removing the need for recomputation, we can reduce training costs by approximately 25% (Qi et al., 2024).

1.0

0.5

Percentage(dashed)

0.9

0.4

###### Rewards(solid)

0.8

0.3

PG-IS

Mask-0.5-Recompute

0.7

0.2

Mask-0.5-Rollout Mask-0.8-Rollout

0.6

0.1

0 500 1000 1500 2000 2500 3000 3500

Training Step

- Figure 5. Isolating the source of instability. The solid curves are training rewards, while the dashed lines are the percentage of bad updates. Starting with the unstable PG-IS, applying a minimal mask that only blocks large-divergence bad updates on negative samples is sufficient to stabilize training, indicating these bad updates are the primary cause of training instability.

###### 5.3. Identifying the Source of Instability

Finally, we seek to pinpoint which specific policy updates are most responsible for the instability. Our methodology is to start with the unstable PG-IS algorithm, which applies no update masking, and introduce the most minimal mask necessary to restore stability. This allows us to isolate the most detrimental class of updates. Since updates on positively rewarded samples are typically safe, we focus on negative samples where the policy is penalized (Liu et al., 2025a; Ren & Sutherland, 2025). We design a simple mask that only blocks updates on negative samples where the probability of the sampled token is decreased by more than a threshold δ:

Mt = 0 if Aˆt < 0 and µθ′(yt|st) − πθ(yt|st) ≥ δ.

As shown in Figure 5, applying this minimal mask with δ = 0.5 is sufficient to stabilize the training. In contrast, a slightly looser mask (δ = 0.8) or one anchored to the recomputed distribution (“Mask-0.5-Recompute”) both fail to prevent the eventual collapse. We define bad updates as those where this divergence exceeds 0.5 and plot their percentage over time. The plot reveals that only a very small fraction of updates are “bad” (≤ 0.5%) yet they are the primary culprits behind training collapse. Furthermore, the percentage of these bad updates strongly correlates with reward fluctuation; as the fraction of bad updates rises, the reward curve becomes more erratic, reinforcing a causal link.

Takeaway 3: The primary source of instability is a small subset of updates on negative samples that push the policy far outside the trust region. A likely reason is that aggressively penalizing a token the model deems probable can corrupt the LLM’s internal knowledge and destabilize the learning process. This finding confirms the critical need for a trust region, particularly when handling negative feedback.

###### 5.4. The Pitfalls of Truncated Importance Sampling

Our results also reveal an unexpected drawback of Truncated Importance Sampling (TIS), a common technique for reducing policy-gradient variance (Yao et al., 2025; Chen et al., 2025). In our experiments, TIS worsens training stability: as shown in Figure 3, PG-TIS and MiniRL-TIS collapse prematurely and substantially underperform their untruncated counterparts.

We hypothesize that this detrimental effect stems from the same issue as PPO’s ratio clipping: low-probability tokens, which naturally produce high-variance ratios, are the most likely to be truncated by TIS. While this does reduce variance, it systematically down-weights the gradient signal from these tokens, introducing a significant and harmful bias into the policy update. This suggests that naive truncation can be as damaging as naive clipping.

### 6. Analysis on Training Efficiency

Beyond training stability, the design of trust region is also critical for training efficiency. As motivated in Section 4.2, PPO’s ratio-clipping over-constrains the updates to low-probability tokens, which might be permitted by a divergence-based trust region. In this section, we aim to analyze how low-probability tokens affect the training dynamics, thus justifying the adoption of divergence-based trust region in our DPPO algorithm.

Experimental Setting: We fine-tune Qwen3-1.7B-Base (Yang et al., 2025) on the DAPO dataset (Yu et al., 2025). We employ GRPO (Guo et al., 2025; Liu et al., 2025d) with the Clip-Higher trick (Yu et al., 2025) as the baseline algorithm. We then relax trust regions by setting the clipping threshold ϵ in Equation (10) as infinity for tokens with µ(yt|st) < α, thus isolating the effect of low-probability tokens.

###### Rewards

###### Clipped rollout probability

###### Clipped entropy

0.35

4.0

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0

0.7

3.5

0.30

0.1 0.3 0.5 0.7

0.6

3.0

0.25

0.5

2.5

0.20

2.0

0.4

0.15

1.5

0.3

0.10

1.0

0.2

0.05

0.5

0.1

0.0

0.00

0 10 20 30 40 50 60 70

0 10 20 30 40 50 60 70

0 10 20 30 40 50 60 70

Training step

Training step

Training step

- Figure 6. Analysis of relaxing trust regions for low-probability tokens. (Left) Training reward curves. (Middle) Rollout probability of clipped tokens. (Right) Entropy of clipped tokens.

relax either one end (Relax-high or Relax-low) or both ends (Relax-both). For example, Relax-high is implemented by (ϵlow = 0.2,ϵhigh = ∞) for tokens with µ(yt|st) < α.

###### 0.30 Rewards

Entropy

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.25

1.00

0.20

0.10

0.15

Baseline

0.10

Relax-both Relax-high Relax-low

0.01

0.05

0.00

0 100 200 300 400 500

0 100 200 300 400 500

Training step

Training step

Figure 7. Analysis of trust region relaxation direction. (Left) Training reward curves. (Right) Policy entropy.

As illustrated in Figure 7, the direction of clip relaxation plays a critical role in the training efficiency and stability. Relax-high can be viewed as an extreme variant of the Clip-Higher trick (Yu et al., 2025) applied only to lowprobability tokens. While this approach maintains high entropy, it fails to yield significant gains in training efficiency. Conversely, Relax-low exhibits substantially faster initial learning2. However, this strategy eventually drops due to entropy collapse (Cui et al., 2025). Ultimately, we find that Relax-both is the most effective strategy for achieving both efficient and stable training, thereby validating the design of DPPO in relaxing both ends of the trust region.

The learning curves for varying values of α are presented in Figure 6. Notably, relaxing the clipping constraint for tokens with µ(yt|st) < 0.1 yields a substantial improvement in training efficiency compared to the GRPO baseline (α = 0). This observation validates our hypothesis that the ratio-clipping mechanism in PPO over-constrains updates to low-probability tokens, thereby hindering overall learning progress. The middle plot reveals that clipped tokens are predominantly characterized by low probabilities (typically below 0.15 for the baseline in blue). As α increases, the probabilities of clipped tokens also rise, confirming that PPO’s ratio-clipping is structurally biased against low-probability tokens. Furthermore, the right plot demonstrates that clipped tokens frequently exhibit high entropy. Consistent with Wang et al. (2025a), which posits that RL is driven primarily by high-entropy tokens in LLMs, our results suggest that relaxing constraints on these tokens enables more informative policy updates and thus achieves higher training efficiency (see Appendix D for most frequent clipped tokens).

Furthermore, we examine the effect of directional clip relaxation with a fixed α=0.1. We generalize the clip operation with asymmetric thresholds, denoted as clip(rt,1−ϵlow,1+ ϵhigh), where ϵlow = 0.2 and ϵhigh = 0.28 by default. We

### 7. Broader Evaluation

We conduct large-scale experiments to further validate our methods. We train on a filtered subset of DAPO-Math (Li et al., 2026), containing approximately 13k samples. Five model configurations (different base models and training techniques) are evaluated: (1) MoE Base: Qwen3-30BA3B-Base (Yang et al., 2025); (2) MoE Base w/ R3: Qwen330B-A3B-Base with rollout router replay (R3) (Ma et al., 2025); (3) MoE Thinking: Qwen3-30B-A3B; (4) Dense Base: Qwen3-8B-Base; (5) MoE Base w/ LoRA: Qwen330B-A3B-Base with LoRA (Hu et al., 2022). Baseline methods include GRPO-ClipHigher(Shao et al., 2024; Liu et al., 2025d; Yu et al., 2025) and CISPO(Chen et al., 2025; Khatri et al., 2025). All methods use the behavior policy (µθ′) instead of recomputed policy distribution (πθ′) to construct the trust region (i.e., for clipping or masking). We compare our proposed methods, DPPO-Binary-KL and DPPO-Binary-TV, against these baselines. More details are provided in Appendix E.

2In contrast to the Clip-Higher intuition (Yu et al., 2025), we observe that “Clip-Lower” (relaxing ϵlow) for low-probability tokens is more vital for efficiency. This aligns with findings by Tajwar et al. (2024) regarding the role of negative gradients in accelerating preference learning.

- 0.2

- 0.3

- 0.4

- 0.5

- 0.6

AIME24 w/o R3

GRPO-ClipHigher

CISPO

DPPO-Binary-KL DPPO-Binary-TV

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 50 100 150 200

Training Step

0.2

0.3

0.4

0.5

0.6

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

AIME24 w/ R3

0 50 100 150 200

Training Step

0.05

0.10

0.15

0.20

0.25

0.30

0.35

0.40

AIME25 w/o R3

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 50 100 150 200

Training Step

0.05

0.10

0.15

0.20

0.25

0.30

0.35

0.40

AIME25 w/ R3

Figure 8. Evolution of AIME24 and AIME25 Avg@32 scores during RL training using Qwen3-30B-A3B-Base. The first and third panels correspond to the same experiment without rollout router replay (w/o R3), while the second and fourth panels correspond to the same experiment with rollout router replay (w/ R3).

0 50 100 150 200 250

Training Step

0.64

0.66

0.68

0.70

0.72

0.74

0.76

0.78

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

A3B - AIME24

A3B-GRPO-ClipHigher

A3B-CISPO

A3B-DPPO-Binary-KL A3B-DPPO-Binary-TV

0 50 100 150 200 250

Training Step

0.54

0.56

0.58

0.60

0.62

0.64

0.66

0.68

A3B - AIME25

0 200 400 600 800 1000 1200 1400

Training Step

0.10

0.15

0.20

0.25

0.30

0.35

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

8B - AIME24

8B-GRPO-ClipHigher

8B-CISPO

8B-DPPO-Binary-KL 8B-DPPO-Binary-TV

0 200 400 600 800 1000 1200 1400

Training Step

0.10

0.15

0.20

0.25

0.30

8B - AIME25

Figure 9. Evolution of AIME24 and AIME25 scores during RL training using Qwen3-30B-A3B (left) and Qwen3-8B-Base (right).

- 7.1. Main Results

0 50 100 150 200

Training Step

###### 7.2. RLHF-style Alignment Tasks

We present online evaluation results on AIME24 and AIME25 (MAA, 2025) during RL training in the following figures: Figure 8 (MoE Base with and without R3) and Figure 9 (MoE Thinking and Dense Base). Results for MoE Base with LoRA are provided in Appendix F.2.

Our proposed method consistently demonstrates superior stability and efficiency across all five large-scale experiments. Specifically, DPPO optimizes rewards at a significantly faster speed than the GRPO-ClipHigher baseline and achieves better converged performance, providing empirical validation for the motivations discussed in Section 4.2. While all baseline methods frequently exhibit training instability or catastrophic collapse (e.g., CISPO in MoE Base without R3 and GRPO-ClipHigher in MoE Thinking), our approach maintains a remarkably stable training process.

Rollout router replay (R3) is widely considered a necessary technique for stabilizing RL training in MoE models (Ma et al., 2025; Zheng et al., 2025; Liu et al., 2025a). However, as illustrated in Figure 8, our DPPO variants (without R3) even consistently outperform the R3-enhanced baselines, which underscores the superior training efficiency and inherent stability of the DPPO framework. Furthermore, incorporating R3 yields additional gains for DPPO, suggesting that the benefits of R3 and DPPO are largely orthogonal. We provide additional detailed results and extended discussions in Appendix F.2.

###### gemma-2-9b-it + UltraFeedback

Qwen3-4B-Instruct-2507 + hh-rlhf

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

70

80

###### GRPO DPPO

60

70

50

60

Reward

40

50

30

40

20

30

10

20

0 20 40 60 80 100 120

0 25 50 75 100 125 150 175 200

Training step

Training step

Figure 10. RLHF experiments with Skywork-Reward-Llama-3.18B as the reward model. DPPO improves reward faster and reaches higher final reward than GRPO on both Gemma-2-9B-It with UltraFeedback and Qwen3-4B-Instruct-2507 with HH-RLHF.

To evaluate DPPO beyond verifiable rewards, we further conduct RLHF experiments on open-ended alignment data. We compare GRPO with DPPO-Binary-TV in two settings: Gemma-2-9B-It (Team et al., 2024) trained on UltraFeedback (Cui et al., 2023), and Qwen3-4B-Instruct-2507 (Yang et al., 2025) trained on HH-RLHF (Bai et al., 2022). Both settings use Skywork-Reward-Llama-3.1-8B (Liu et al., 2024) as the reward model. As shown in Figure 10, DPPO improves the learned reward substantially faster and reaches higher final rewards in both settings. We further evaluate the fine-tuned Qwen3 models on AlpacaEval 2.0 in Appendix F.1, where DPPO achieves 80.90 length-controlled and 79.93 raw win rates, establishing a new SOTA on the AlpacaEval 2.0 community leaderboard. These results show

that the effectiveness of DPPO also generalizes to noisy learned-reward optimization.

###### 7.3. Ablation on Divergence Approximation

###### AIME24

###### AIME25

0.60

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.40

0.50

0.35

0.30

0.40

0.25

GRPO-ClipHigher

0.20

CISPO

0.30

DPPO-Binary-KL DPPO-Binary-TV

0.15

DPPO-TopK-KL DPPO-TopK-TV

0.10

0.20

0.05

0 50 100 150 200

0 50 100 150 200

Training Step

Training Step

Figure 11. Evolution of AIME24 and AIME25 scores for baselines and DPPO with binary/Top-K (K=20) TV/KL approximation under the same setting as MoE Base w/o R3.

In the above experiments, DPPO is implemented using the binary TV/KL approximation (Equations 13 and 14). To assess the impact of this simplification, we compare it against DPPO with the top-K (K=20) TV/KL (Equations 15 and 16) under the same setting as MoE Base. The results, presented in Figure 11, show that both approximations perform similarly and significantly outperform the baselines. This finding indicates that the easy-to-implement binary approximation is a sufficient and computationally efficient choice for scalable RL. We provide more detailed results in Appendix F.3.

###### 7.4. Generalization to Other Model Families and Tasks

We also conduct experiments on Llama family models (Touvron et al., 2023; Wang et al., 2025b) and on general reasoning tasks (Liu et al., 2025e). The results, which are presented in Appendix F.5, show DPPO outperforms the baseline across most settings, highlighting its broad applicability.

- 7.5. Hyperparameter Sensitivity

We provide the hyperparameter sensitivity experiments in Appendix F.4.

- 8. Related Work

###### 8.1. Extended Connections to Existing Work

In this work, we identify a structural flaw in PPO’s ratioclipping mechanism within the LLM regime: it overpenalizes low-probability tokens and under-penalizes highprobability ones, thereby impairing training efficiency and stability. Our proposed DPPO addresses this issue by directly constraining the policy divergence. This methodology aligns with the insights of Wang et al. (2019; 2020), who observed similar exploration issues and proposed adaptive

clipping based on KL divergence in traditional RL settings. However, in the context of LLMs, computing the exact divergence is prohibitive due to the huge memory footprint. To overcome this, we propose a binary divergence approximation, which empirically captures most of the benefits (see Figure 11). Furthermore, as demonstrated in Section 5 and Section 6, the challenges of training stability and efficiency are exacerbated in LLMs by their expansive vocabularies, because low-probability tokens form a non-trivial portion of the entire distribution due to the long-tailed nature (see Figure 2). Finally, the training-inference mismatch inherent to the LLM era introduces additional algorithmic complexities, as further detailed in Section 5.

###### 8.2. Training-inference Mismatch

Recent work has identified a key culprit for training instability: the training-inference mismatch (πθ ̸= µθ), where the policy distribution used for gradient computation (πθ) diverges from the one used for data generation (µθ), even when using identical model parameters θ (Yao et al., 2025; Qi et al., 2025b; Liu et al., 2025b; Zheng et al., 2025). This discrepancy arises from numerical precision errors (Qi et al., 2025b) and subtle differences in implementation (Team et al., 2025a; He, 2025). As training progresses, this mismatch can be amplified if the RL algorithm cannot manage it appropriately, leading to catastrophic performance degradation (Qi et al., 2025b; Liu et al., 2025b).

Existing efforts to mitigate this issue primarily focus on correcting biased gradients through importance sampling. Building on this principle, techniques such as Truncated Importance Sampling (TIS) (Yao et al., 2025; Zheng et al., 2025) and Masked Importance Sampling (Liu et al., 2025b; Team et al., 2025b) have been introduced at both the token and sequence levels. However, as suggested by Qi et al. (2025b), these methods often fail to achieve a satisfactory balance between training efficiency and stability. In contrast,

- our DPPO algorithm significantly enhances both aspects compared to these existing approaches.

Another line of research attempts to resolve the mismatch issue through higher precision (Qi et al., 2025b) or rigor-

- ous engineering alignment (Team et al., 2025a; He, 2025; Zhang et al., 2025). While promising, these methods face limited applicability. For instance, aligning implementation details often requires specific training engines or model architectures, hindering broad adoption. Furthermore, in low-precision settings optimized for high-speed training, we must tolerate a significant training-inference mismatch. In such scenarios, a robust and fast algorithm like DPPO remains essential. Finally, our algorithmic design is orthogonal to these engineering-level optimizations and can be combined with them to achieve even greater performance gains.

### 9. Conclusion

In this work, we have presented a comprehensive rethinking of the trust region framework within the context of LLM fine-tuning. We derived policy improvement bounds specifically tailored to the finite-horizon, undiscounted setting of LLM generation, establishing a rigorous theoretical foundation for future trust-region research. Furthermore, through extensive empirical analysis, we investigated the trade-offs between training stability and efficiency, providing practical guidelines to optimize both.

Central to our contribution is the introduction of Divergence Proximal Policy Optimization (DPPO). We identified and addressed a critical structural flaw in PPO algorithm: it over-constrains updates to low-probability tokens while under-constraining potentially catastrophic shifts in high-probability tokens. This implicit bias results in a suboptimal training dynamic for the expansive, long-tailed vocabularies inherent to LLMs. By substituting heuristic ratio clipping with a more principled policy divergence, DPPO significantly enhances both efficiency and stability. To avoid computing an exact policy divergence, we introduced Binary and Top-K approximations, which capture essential divergence with negligible overhead. Our evaluations demonstrate that DPPO consistently outperforms existing methods like GRPO in both training efficiency and stability, offering a more robust foundation for the RL-based LLM fine-tuning.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Achiam, J., Held, D., Tamar, A., and Abbeel, P. Constrained policy optimization. In International conference on machine learning, pp. 22–31. PMLR, 2017.

Ahmadian, A., Cremer, C., Gall´e, M., Fadaee, M., Kreutzer, J., Pietquin, O., Ust¨¨ un, A., and Hooker, S. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Chen, A., Li, A., Gong, B., Jiang, B., Fei, B., Yang, B.,

- Shan, B., Yu, C., Wang, C., Zhu, C., et al. Minimaxm1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., et al. Ultrafeedback: Boosting language models with scaled ai feedback. arXiv preprint arXiv:2310.01377, 2023.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, H. Defeating nondeterminism in llm inference. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml. 20250910. https://thinkingmachines.ai/blog/defeatingnondeterminism-in-llm-inference/.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hilton, J., Cobbe, K., and Schulman, J. Batch sizeinvariance for policy optimization. Advances in Neural Information Processing Systems, 35:17086–17098, 2022.

Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=nZeVKeeFYf9.

Hunter, D. R. and Lange, K. A tutorial on mm algorithms. The American Statistician, 58(1):30–37, 2004.

Kakade, S. and Langford, J. Approximately optimal approximate reinforcement learning. In Proceedings of the nineteenth international conference on machine learning, pp. 267–274, 2002.

Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri, S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D., and Agarwal, R. The art of scaling reinforcement learning compute for llms. arXiv preprint arXiv:2510.13786, 2025.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Li, Y., Xu, J., Li, Z., Liu, J., Liu, W., Tong, Y., Zheng, L., Xue, Z., Zhang, Y., Cai, T., et al. The optimal token baseline: Variance reduction for long-horizon llm-rl. arXiv preprint arXiv:2602.07078, 2026.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Liu, C. Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., and Zhou, Y. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024.

Liu, J., Li, Y., Fu, Y., Wang, J., Liu, Q., and Shen, Y. When speed kills stability: Demystifying rl collapse from the inference-training mismatch, 2025b. https://yingru.notion.site/WhenSpeed-Kills-Stability-Demystifying-RLCollapse-from-the-Inference-Training-Mismatch271211a558b7808d8b12d403fd15edda.

Liu, Z., Chen, C., Du, C., Lee, W. S., and Lin, M. Oat: A research-friendly framework for llm online alignment. https://github.com/sail-sg/oat, 2025c.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025d.

Liu, Z., Sims, A., Duan, K., Chen, C., Yu, S., Zhou, X., Xu, H., Xiong, S., Liu, B., Tan, C., et al. Gem: A gym for agentic llms. arXiv preprint arXiv:2510.01051, 2025e.

Ma, W., Zhang, H., Zhao, L., Song, Y., Wang, Y., Sui, Z., and Luo, F. Stabilizing moe reinforcement learning by aligning training and inference routers. arXiv preprint arXiv:2510.11370, 2025.

MAA. American invitational mathematics examination aime. https://maa.org/, 2025.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Qi, P., Wan, X., Huang, G., and Lin, M. Zero bubble (almost) pipeline parallelism. In The Twelfth International Conference on Learning Representations, 2024.

Qi, P., Liu, Z., Pang, T., Du, C., Lee, W. S., and Lin, M. Optimizing anytime reasoning via budget relative policy optimization. arXiv preprint arXiv:2505.13438, 2025a.

Qi, P., Liu, Z., Zhou, X., Pang, T., Du, C., Lee, W. S., and Lin, M. Defeating the training-inference mismatch via fp16. arXiv preprint arXiv:2510.26788, 2025b.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.

Ren, Y. and Sutherland, D. J. Learning dynamics of LLM finetuning. In The Thirteenth International Conference on Learning Representations, 2025.

Schulman, J. and Thinking Machines Lab. Lora without regret. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20250929. https://thinkingmachines.ai/blog/lora/.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International conference on machine learning, pp. 1889–1897. PMLR, 2015.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.

Tajwar, F., Singh, A., Sharma, A., Rafailov, R., Schneider, J., Xie, T., Ermon, S., Finn, C., and Kumar, A. Preference fine-tuning of LLMs should leverage suboptimal, onpolicy data. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 47441–47474. PMLR, 21–27 Jul 2024.

Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Ram´e, A., et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Team, L., Han, B., Tang, C., Liang, C., Zhang, D., Yuan, F., Zhu, F., Gao, J., Hu, J., Li, L., Li, M., Zhang, M., Jiang, P., Jiao, P., Zhao, Q., Yang, Q., Shen, W., Yang, X., Zhang, Y., Ren, Y., Zhao, Y., Cao, Y., Sun, Y., Zhang, Y., Fang, Y., Lin, Z., Cheng, Z., and Zhou, J. Every attention matters: An efficient hybrid architecture for long-context reasoning. arXiv preprint arXiv:2510.19338, 2025a.

Team, L., Shen, A., Li, B., Hu, B., Jing, B., Chen, C., Huang, C., Zhang, C., Yang, C., Lin, C., et al. Every step evolves: Scaling reinforcement learning for trillion-scale thinking model. arXiv preprint arXiv:2510.18855, 2025b.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wan, X., Qi, P., Huang, G., Ruan, C., Lin, M., and Li, J. Revisiting parameter server in llm post-training. arXiv preprint arXiv:2601.19362, 2026.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025a.

Wang, Y., He, H., Tan, X., and Gan, Y. Trust region-guided proximal policy optimization. Advances in Neural Information Processing Systems, 32, 2019.

- Wang, Y., He, H., and Tan, X. Truly proximal policy optimization. In Uncertainty in artificial intelligence, pp. 113–122. PMLR, 2020.
- Wang, Z., Zhou, F., Li, X., and Liu, P. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025b. Preprint.

Xie, Z., Zhang, Q., Yang, F., Hutter, M., and Xu, R. Simple policy optimization. arXiv preprint arXiv:2401.16025, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yao, F., Liu, L., Zhang, D., Dong, C., Shang, J., and Gao, J. Your efficient rl framework secretly brings you off-policy rl training, August 2025. https://fengyao.notion.site/offpolicy-rl.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zhang, Z., Ding, X., Yuan, J., Liu, R., Mao, H., Xing, J., and Liu, Z. Deterministic inference across tensor parallel sizes that eliminates training-inference mismatch. arXiv

- preprint arXiv:2511.17826, 2025.

Zheng, C., Dang, K., Yu, B., Li, M., Jiang, H., Lin, J., Liu, Y., Lin, H., Wu, C., Hu, F., et al. Stabilizing reinforcement learning with llms: Formulation and practices. arXiv

- preprint arXiv:2512.01374, 2025.

Zhu, Z., Xie, C., Lv, X., and slime Contributors. slime: An llm post-training framework for rl scaling. https:// github.com/THUDM/slime, 2025. GitHub repository. Corresponding author: Xin Lv.

### A. Trust Region in LLMs

- A.1. Proof of Performance Difference Identity Proof of Theorem 3.1. We begin by expressing the difference in expected returns by its definition:

J (π) − J (µ) = Ey∼π[R(y)] − Ey∼µ[R(y)]

=

y

π(y|x) − µ(y|x) R(y).

The core of the proof is to establish an identity for the difference in the probabilities of generating a sequence y, π(y|x) − µ(y|x). We use the following telescoping sum identity, which can be verified by expanding the terms:

π(y|x) − µ(y|x) =

T

t=1

t−1

k=1

µ(yk|sk) π(yt|st) − µ(yt|st)

 

T

j=t+1

π(yj|sj)

 .

Substituting this identity into the expression for the performance difference yields:

J (π) − J (µ) =

y

R(y)

T

t=1

t−1

k=1

µ(yk|sk) π(yt|st) − µ(yt|st)

 

T

j=t+1

π(yj|sj)

 

=

y

µ(y|x)R(y)

T

t=1

π(yt|st) µ(yt|st) − 1

 

T

j=t+1

π(yj|sj) µ(yj|sj)

 

= Ey∼µ

 R(y)

T

t=1

π(yt|st) µ(yt|st) − 1

 

T

j=t+1

π(yj|sj) µ(yj|sj)

 

 

= Ey∼µ

 R(y)

|y|

t=1

π(yt|st) µ(yt|st) − 1

 

− Ey∼µ

 R(y)

|y|

t=1

π(yt|st) µ(yt|st) − 1

 1 −

T

j=t+1

π(yj|sj) µ(yj|sj)

 

 .

By identifying the terms with the definitions in the theorem statement, we arrive at:

J (π) − J (µ) = L′µ(π) − ∆(µ,π), where

L′µ(π) = Ey∼µ

 R(y)

|y|

t=1

π(yt|st) µ(yt|st) − 1

 ,

∆(µ,π) = Ey∼µ

 R(y)

|y|

t=1

π(yt|st) µ(yt|st) − 1

 1 −

T

j=t+1

π(yj|sj) µ(yj|sj)

 

 .

This completes the proof.

| |
|---|

- A.2. Proof of Policy Improvement Bound: Max-Divergence Part

Lemma A.1 (Bound on Sequence-Level TV Divergence). Let µ and π be two policies that generate sequences of length N. Let µN(·|s1) and πN(·|s1) denote the distributions over sequences y = (y1,...,yN). The total variation (TV) divergence between these sequence distributions is bounded by the sum of the expected single-step TV divergences:

N

DTV µN(·|s1)∥πN(·|s1) ≤

t=1

Es

t∼µ DTV µ(·|st)∥π(·|st) ,

where the expectation is over the state distribution induced by policy µ.

Proof. Let P(y) = µN(y|s1) and Q(y) = πN(y|s1).

2DTV(P∥Q) =

y

|P(y) − Q(y)| =

y

N

N

µ(yt|st) −

π(yt|st) .

t=1

t=1

We use the algebraic identity a1 ...aN −b1 ...bN = Nt=1 tk−=11 ak (at −bt) Nj=t+1 bj . Applying this to the policy probabilities and then using the triangle inequality, we get:

 

 

t−1

N

N

###### 2DTV(P∥Q) ≤

µ(yk|sk) |µ(yt|st) − π(yt|st)|

π(yj|sj)

y

t=1

j=t+1

k=1

 

 .

t−1

N

N

µ(yk|sk) |µ(yt|st) − π(yt|st)|

π(yj|sj)

=

t=1 y

j=t+1

k=1

For each term in the outer sum over t, we can sum over the variables yj for j > t. Since y

π(yj|sj) = 1 for all sj, the product of terms for j > t sums to 1 when we integrate out yt+1,...,yN. This leaves:

j

t−1

N

2DTV(P∥Q) ≤

µ(yk|sk) |µ(yt|st) − π(yt|st)|

t=1 y1,...,yt

k=1

t−1

N

µ(yk|sk)

|µ(yt|st) − π(yt|st)|.

=

t=1 y1,...,yt−1

yt

k=1

The inner sum is 2DTV(µ(·|st)∥π(·|st)). The outer sum over y1,...,yt−1 defines an expectation over states st under policy µ. Thus, we have:

N

Es

t∼µ 2DTV µ(·|st)∥π(·|st) . Dividing by 2 yields the desired result.

2DTV(P∥Q) ≤

t=1

| |
|---|

Proof of the max-divergence bound in Theorem 3.2. From Lemma 3.1, we start with the exact performance difference identity:

J (π) − J (µ) = L′µ(π) − ∆(µ,π).

For brevity, we define y≤t = {x,y1,...,yt} and y>t = {yt+1,yt+2,...}, then we can rewrite ∆(µ,π) as:

 R(y)

 .

|y|

π(y>t|st+1) µ(y>t|st+1)

π(yt|st) µ(yt|st) − 1 1 −

∆(µ,π) = Ey∼µ

t=1

Our goal is to find an upper bound for the error term ∆(µ,π). We begin by bounding the reward by its maximum absolute value, ξ = maxy |R(y)|.

T

π(y>t|st+1) µ(y>t|st+1)

π(yt|st) µ(yt|st) − 1 · 1 −

∆(µ,π) ≤ ξ · Ey∼µ

t=1

(17)

T

π(yt|st) µ(yt|st) − 1 · Ey

π(y>t|st+1) µ(y>t|st+1)

Ey

= ξ ·

>t∼µ(·|st+1) 1 −

.

≤t∼µ

t=1

The inner expectation is exactly twice the TV divergence between the distributions over future trajectories:

π(y>t|st+1) µ(y>t|st+1)

Ey

>t∼µ(·|st+1) 1 −

= 2DTV µ>t(·|st+1)∥π>t(·|st+1) .

Using Theorem A.1 on this sequence-level TV divergence (for a sequence of length T − t), we get:

T

DTV µ>t(·|st+1)∥π>t(·|st+1) ≤

k=t+1

Es

k∼µ(·|st+1) DTV µ(·|sk)∥π(·|sk) .

We bound each term in the sum by the maximum single-step TV divergence, DTVmax(µ∥π) = maxs DTV(µ(·|s)∥π(·|s)), which gives:

DTV µ>t(·|st+1)∥π>t(·|st+1) ≤

T

DTVmax(µ∥π) = (T − t)DTVmax(µ∥π).

k=t+1

Substituting this back into the bound for ∆(µ,π):

T

π(yt|st) µ(yt|st) − 1 · 2(T − t)DTVmax(µ∥π)

Ey

∆(µ,π) ≤ ξ ·

≤t∼µ

t=1

T

π(yt|st) µ(yt|st) − 1

= 2ξ · DTVmax(µ∥π)

(T − t)Es

µ(yt|st)

t∼µ

yt

t=1

T

= 2ξ · DTVmax(µ∥π)

(T − t)Es

t∼µ [2DTV(µ(·|st)∥π(·|st))]

t=1

T

t∼µ [2DTVmax(µ∥π)]

≤ 2ξ · DTVmax(µ∥π)

(T − t) · Es

t=1

T

= 4ξ · DTVmax(µ∥π)2

(T − t)

t=1

= 2ξT(T − 1) · DTVmax(µ∥π)2.

Substituting this into the performance difference identity gives the max-divergence bound in Equation (7):

(18)

This completes the proof.

###### J (π) − J (µ) ≥ L′µ(π) − 2ξT(T − 1) · DTVmax(µ∥π)2.

| |
|---|

###### A.3. Proof of Policy Improvement Bound: Average-Divergence Part

We now prove the linear average-divergence bound in Equation (8), which is the second part of Theorem 3.2. Compared with the quadratic max-divergence bound in Equation (7), this form avoids a T2 dependence and is therefore more suitable for long LLM responses. The key step is to use the fact that total variation is always bounded by one, i.e., DTV(P∥Q) ≤ 1, instead of upper-bounding the future-trajectory divergence by (T − t)DTVmax(µ∥π).

We begin from the intermediate step in Equation (17):

T

Ey

∆(µ,π) ≤ ξ ·

≤t∼µ

t=1

π(yt|st) µ(yt|st) − 1 · Ey

π(y>t|st+1) µ(y>t|st+1)

>t∼µ(·|st+1) 1 −

.

The inner expectation is exactly twice the TV divergence between the future trajectory distributions, 2DTV µ>t(·|st+1)∥π>t(·|st+1) . Instead of bounding this term with 2(T − t)DTVmax(µ∥π), we now apply the simple

upper bound of 2:

T

π(yt|st) µ(yt|st) − 1 · 2DTV µ>t(·|st+1)∥π>t(·|st+1)

Ey

∆(µ,π) ≤ ξ ·

≤t∼µ

t=1

T

π(yt|st) µ(yt|st) − 1 (19)

Ey

≤ 2ξ ·

≤t∼µ

t=1

T

π(yt|st) µ(yt|st) − 1

Es

t∼ρµt Ey

= 2ξ ·

t∼µ(·|st)

t=1

T

Es

= 2ξ ·

t∼ρµt [2DTV(µ(·|st)∥π(·|st))]

t=1

 

 . (20)

|y|

= 4ξ · Ey∼µ

DTV(µ(·|st)∥π(·|st))

t=1

Substituting Equation (20) into the performance difference identity proves the average-divergence part of Theorem 3.2:

 

 .

|y|

J (π) − J (µ) ≥ L′µ(π) − 4ξ · Ey∼µ

DTV(µ(·|st)∥π(·|st))

t=1

Equivalently, using the notation in Theorem 3.2, this is

J (π) − J (µ) ≥ L′µ(π) − 4ξD¯TV(µ,π).

Since both bounds hold simultaneously, an immediate corollary is the following composite guarantee:

J (π) − J (µ) ≥ L′µ(π) − ∆(µ,π)

 2ξT(T − 1) · DTVmax2,4ξ · Ey∼µ

 

 

 .

|y|

≥ L′µ(π) − min

DTV(µ(·|st)∥π(·|st))

t=1

This composite bound leverages the quadratic bound for infinitesimal updates and the linear bound for larger updates or longer horizons.

###### A.4. Comparing Surrogate Objectives with Classical RL

At first glance, the surrogate objective for the LLM regime in Equation (5) appears distinct from the classical RL surrogate in Equation (3). The former is an expectation over full trajectories y weighted by the reward R(y), while the latter is an expectation over state-action pairs (s,a) weighted by the advantage Aµ(s,a). However, we will now show that their gradients with respect to the policy parameters θ are fundamentally analogous, confirming that our LLM-specific formulation is a valid adaptation of the standard policy gradient theorem.

Let the policy π be parameterized by θ. We will use the identity ∇θπθ(a|s) = πθ(a|s)∇θ log πθ(a|s). Gradient of the Classical Surrogate Objective. We begin with the classical surrogate objective from Equation (3):

πθ(a|s) µ(a|s)

1 1 − γ

Aµ(s,a) .

Es∼ρµ, a∼µ(a|s)

Lµ(πθ) =

Taking the gradient with respect to θ and moving it inside the expectation, we get:

1 1 − γ

Es∼ρµ, a∼µ(a|s)

∇θLµ(πθ) =

1 1 − γ

Es∼ρµ, a∼µ(a|s)

=

∇θ πθ(a|s) µ(a|s)

Aµ(s,a)

πθ(a|s) µ(a|s) ∇θ log πθ(a|s)Aµ(s,a) .

(21)

Gradient of the LLM Surrogate Objective. Next, we consider our LLM-specific surrogate from Equation (5):

 R(y)

 .

|y|

πθ(yt|st) µ(yt|st) − 1

L′µ(πθ) = Ey∼µ

t=1

Taking the gradient with respect to θ and noting that the −1 term has a zero gradient:

 

 R(y)

|y|

∇θ πθ(yt|st) µ(yt|st)

∇θL′µ(πθ) = Ey∼µ

t=1

 

 .

|y|

πθ(yt|st) µ(yt|st) ∇θ log πθ(yt|st)R(y)

= Ey∼µ

t=1

If we define a sequence-level advantage as Aµ(st,yt) = R(y) − V (x), where V (x) is a baseline value function for the prompt, the gradient becomes:

 . (22)

 

|y|

πθ(yt|st) µ(yt|st) ∇θ log πθ(yt|st)Aµ(st,yt)

∇θL′µ(πθ) = Ey∼µ

t=1

This form is directly analogous to the classical policy gradient in Equation (21), where the sum over timesteps in a trajectory replaces the expectation over the state distribution ρµ. Thus, our LLM surrogate objective is a theoretically sound adaptation of the classical trust region framework to the undiscounted, sequence-reward setting.

### B. Approximations as Lower Bounds of True Divergence

In this section, we provide a formal justification for our Binary and Top-K divergence approximations. We demonstrate that both are principled lower bounds on the true divergence and explicitly state the conditions under which these approximations become exact.

Let C = {C1,...,Cm} be any partition of the vocabulary A. Our Binary and Top-K approximations correspond to specific choices of this partition. We will show that the divergence computed on the partitioned space is a lower bound on the true divergence.

###### B.1. Total Variation Divergence

The true TV divergence is DTV(µ∥π) = 12 a∈A |µ(a|st) − π(a|st)|. The divergence on a partitioned space C is DTVC = 12 mj=1 |µ(Cj|st) − π(Cj|st)|.

Proof of Lower Bound. By definition, |µ(Cj|st) − π(Cj|st)| = | a∈Cj(µ(a|st) − π(a|st))|. The triangle inequality states that the absolute value of a sum is less than or equal to the sum of the absolute values. Applying this, we get

| a∈Cj(µ(a|st) − π(a|st))| ≤ a∈Cj |µ(a|st) − π(a|st)|. Summing over all partitions j:

m

- 1

- 2

DTVC =

(µ(a|st) − π(a|st))

j=1 a∈Cj

m

- 1

- 2

|µ(a|st) − π(a|st)|

≤

j=1 a∈Cj

= DTV(µ∥π). Thus, DTV(µ∥π) ≥ DTVC . This holds for both Binary and Top-K partitions. Analysis of the Approximation Gap. The gap between the true and approximated TV divergence is the sum of the gaps within each partition. For any partition Cj, the gap is 12 a∈C

|µ(a|st) − π(a|st)| − a∈Cj(µ(a|st) − π(a|st)) .

j

#### Reward

#### Top-20 Probability

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

##### GRPO DPPO

70

0.999

60

0.998

50

40

0.997

30

20

0.996

10

0.995

0

0 50 100 150 200 250 300 350 400 450

0 50 100 150 200 250 300 350 400 450

Training step

Training step

Figure 12. RLHF training curves and top-20 probability mass over training. The top-20 tokens cover nearly all probability mass throughout training, making the Top-K tail and the resulting TV approximation gap small.

This gap is bounded by the total probability mass of the partition: Gap(Cj) ≤

- 1

- 2 a∈C

- 1

- 2

(µ(a|st) + π(a|st)) =

(µ(Cj|st) + π(Cj|st)).

j

For the Top-K approximation, the only partition with a potential gap is the ”other” category, which contains the tail of the distribution. The total probability mass of this tail, µ(Cother|st), is typically very small. Therefore, the approximation gap is also small, justifying Top-K TV as a high-fidelity approximation.

We further validate that this tail-mass bound remains small across training. In an RLHF run, we track the average probability mass covered by the top-20 tokens, matching the K = 20 Top-K approximation used in our experiments. As shown in

- Figure 12, the top-20 mass is already 99.4% at the beginning of training and increases to 99.9% after 100 training steps. Equivalently, the “other” category contains only 0.6% of the mass initially and about 0.1% after the policy sharpens, so the Top-K TV approximation gap remains tiny and empirically improves during training rather than degrading.

Equality Condition. Equality DTV = DTVC holds if the gap is zero for all partitions. This occurs when µ(a|st) − π(a|st) has the same sign for all tokens a within each partition Cj.

###### B.2. KL Divergence

The true KL divergence is DKL(µ∥π) = a∈A µ(a|st)log µ(a|s

t)

π(a|st). The divergence on the partitioned space is DKLC =

m j=1 µ(Cj|st)log πµ((CCj|st)

j|st). Proof of Lower Bound. The proof relies on the log-sum inequality, which states that for any two sets of non-negative numbers {x1,...,xn} and {y1,...,yn}:

n

n

n i=1 xi n i=1 yi

xi yi ≥

xi log

xi log

.

i=1

i=1

We apply this inequality to each partition Cj in our vocabulary, setting xa = µ(a|st) and ya = π(a|st):

 log a∈Cj

 

µ(a|st) a∈Cj π(a|st)

µ(a|st) π(a|st) ≥

µ(a|st)

µ(a|st)log

a∈Cj

a∈Cj

µ(Cj|st) π(Cj|st)

= µ(Cj|st)log

.

Summing over all partitions j gives the desired result:

m

µ(a|st) π(a|st)

DKL(µ∥π) =

µ(a|st)log

j=1 a∈Cj

m

µ(Cj|st) π(Cj|st)

= DKLC .

≥

µ(Cj|st)log

j=1

Equality Condition. The log-sum inequality holds with equality if and only if the ratio xi

yi is constant for all i. In our context, this means that for each partition Cj, the ratio µ(a|s

t)

π(a|st) must be constant for all tokens a ∈ Cj. For both Binary and Top-K approximations, this implies the policy update must scale the probabilities of all tokens within the ”other” category by a uniform factor.

### C. More Details for Stability Analysis

Our experimental setup strictly follows the sanity test established in Qi et al. (2025b). Each policy iteration begins by sampling a batch of 64 questions. For each question, we generate 8 responses (rollouts) using a maximum context length of 8,000. The collected data is then used to perform 4 gradient steps. All experiments are conducted using the VeRL framework (Sheng et al., 2024) together with the ODC optimization (Wan et al., 2026), and models are trained in BFloat16 precision to better expose potential numerical instabilities between algorithms. For the evaluation on AIME, we sample 32 responses for each test question to ensure a robust assessment.

###### C.1. Algorithmic Details for Stability Analysis

In this section, we provide the specific policy gradient formulations for each algorithm evaluated in our stability analysis (Section 5). To facilitate a direct comparison, we show how each algorithm’s gradient update can be interpreted through the lens of a single, unified framework.

A Unified Policy Gradient Formulation. The policy gradient for the algorithms we tested can be generalized into the following form, where the gradient of the objective L(θ) is expressed as:

 

 . (23)

|y|

πθ(yt|st) µθ′(yt|st)

,C · Aˆt · ∇θ log πθ(yt|st)

∇θL(θ) = Ey∼µ

Mt · min

θ′

t=1

In this formulation, Aˆt is the advantage, estimated following the GRPO method but without standard deviation normalization (Shao et al., 2024; Liu et al., 2025d). The algorithms differ primarily in their definition of the binary mask Mt and the clipping bound C.

- • For PG-IS, we have Mt = 1 and C = ∞.
- • For PG-TIS (CISPO), we have Mt = 1 and C = 3.
- • For GRPO, the mask Mt is the PPO-style clipping mask, and C = ∞.
- • For MiniRL, the mask Mt is also a PPO-style clipping mask but is conditioned on a recomputed policy ratio. For this algorithm, C = ∞.
- • For MiniRL-TIS, the mask Mt is the same as in MiniRL, but with C = 3.
- • For DPPO (Ours), the mask Mt is conditioned on the policy divergence, and C = ∞.

Mask Definitions. The specific forms of the masks are as follows:

θ(yt|st)

- • For GRPO, the mask uses the rollout ratio rt = π

µθ′(yt|st) and experimental hyperparameters ϵhigh = 0.28,ϵlow = 0.2:

- 0, if (Aˆt > 0 and rt > 1 + ϵhigh) or (Aˆt < 0 and rt < 1 − ϵlow)
- 1, otherwise.

Mt =

- • For MiniRL and MiniRL-TIS, the mask is structurally identical to GRPO’s and uses the same hyperparameters, but it

is conditioned on the recomputed ratio rt′ = π

θ(yt|st) πθ′(yt|st):

Mt =

- 0, if (Aˆt > 0 and rt′ > 1 + ϵhigh) or (Aˆt < 0 and rt′ < 1 − ϵlow)
- 1, otherwise.

- • For DPPO, our mask is conditioned on the policy divergence Dt:

Mt =

- 0, if (Aˆt > 0 and rt > 1 and Dt > δ) or (Aˆt < 0 and rt < 1 and Dt > δ)
- 1, otherwise.

In our experiments, we set the divergence threshold δ = 0.15 for TV divergence and δ = 0.05 for KL divergence.

### D. Characterizing Clipped Tokens

To understand the practical consequences of ratio clipping, we analyzed which tokens are most frequently penalized by a PPO-style algorithm. We trained a Qwen3-4B-Base model on the DAPO dataset with GRPO and, at training step 50, collected two sets of tokens:

- • Clipped Positive Tokens: From samples with Aˆt > 0, tokens whose updates were blocked due to a high ratio (rt > 1.28).
- • Clipped Negative Tokens: From samples with Aˆt < 0, tokens whose updates were blocked due to a low ratio (rt < 0.8). The 50 most frequently clipped tokens from positively-rewarded samples.

’ the’, ’ \\(’, ’1’, ’Let’, ’ in’, ’ ’, ’,’, ’We’, ’ +’, ’ \\’, ’ numbers’, ’:\n\n’, ’Wait’, ’4’, ’6’, ’ Identify’, ’(’, ’Next’, ’ from’, ’)’, ’ k’, ’ -’, ’Since’, ’ solve’, ’\\[’, ’ how’, ’ ->’, ’ to’, ’ are’, ’Sub’, ’I’, ’):\n’, ’ \n\n’, ’ spiral’, ’ Instead’, ’ this’, ’If’, ’div’, ’ Conditions’, ’ vector’, ’ have’, ’ =’, ’ feasible’, ’Or’, ’ inconsistency’, ’ express’, ’_{’, ’ increase’, ’ exact’, ’ consider’

The 50 most frequently clipped tokens from negatively-rewarded samples.

’ \\(’, ’ the’, ’,’, ’ a’, ’ \\’, ’ ’, ’2’, ’1’, ’:\n\n’, ’0’, ’3’, ’ and’, ’ (’, ’ that’, ’-’, ’ to’, ’5’, ’ of’, ’However’, ’\\’, ’ is’, ’ =’, ’4’, ’ in’, ’ for’, ’ all’, ’ we’, ’We’, ’)’, ’.\n\n’, ’ our’, ’.’, ’:\n’, ’ but’, ’ with’, ’So’, ’ both’, ’From’, ’ Let’, ’ this’, ’Thus’, ’Wait’, ’ if’, ’ -’, ’ +’, ’ˆ’, ’ only’, ’ at’, ’Since’, ’ integer’

The 50 most frequent tokens in each category reveal a striking pattern. Far from being random noise, the clipped tokens are often critical for task performance. The lists for both positive and negative samples are dominated by two key categories:

- 1. Numerical and Mathematical Tokens: A significant portion of the clipped tokens are numbers (e.g., ‘1’, ‘4’) and mathematical symbols (e.g., ‘+’, ‘=’, ‘div’).
- 2. Reasoning and Structural Words: The list also includes many words essential for logical exposition, such as ‘Wait’, ‘Next’, ‘Thus’, and ‘Since’.

These findings highlight a fundamental flaw in ratio-based clipping. For positive samples, it blocks beneficial updates to tokens that are integral to constructing correct solutions. For negative samples, it blocks the necessary suppression of these same tokens when they are part of an incorrect reasoning path. By systematically interfering with the learning signal for these high-utility tokens, the algorithm inadvertently slows learning, stifles exploration, and hinders the model’s ability to refine its problem-solving capabilities.

- E. More Details for Broader Evaluation In this section, we provide detailed training and evaluation settings of the scaling experiments in Section 7.

Training Settings. We conduct experiments using the VeRL framework (Sheng et al., 2024) on NVIDIA H Series GPUs. All methods follow the hyperparameter configurations detailed in Table 1. Rollout router replay (R3) (Ma et al., 2025) records the routed experts used in the inference engine and replays them in the training engine, which mitigates the training-inference mismatch and stabilizes RL training for MoE models. We only use R3 in the MoE Base w/ R3 experiment and do not use it in all other experiments. For experiments that utilize LoRA, as suggested by Schulman & Thinking Machines Lab (2025), we employ a larger learning rate of 1 × 10−5. For the MoE Base w/ LoRA experiment, we set lora rank=32 and lora alpha=64.

As suggested in Section 5.2, for all methods, we use the behavior policy (µθ′) instead of recomputed policy distribution (πθ′) to construct the trust region (i.e., for clipping or masking). Under the unified policy gradient formulation (Equation 23), the method-specific hyperparameters (C = 5 by default) are configured as follows:

- • For GRPO-ClipHigher, we have

Mt =

- 0, if (Aˆt > 0 and rt > 1 + ϵhigh) or (Aˆt < 0 and rt < 1 − ϵlow)
- 1, otherwise.

where ϵhigh = 0.27 and ϵlow = 0.2, which follows the hyperparameters used in Zheng et al. (2025).

- • For CISPO, we have Mt = 1.
- • For DPPO-Binary-KL and DPPO-Binary-TV, we have

Mt =

- 0, if (Aˆt > 0 and rt > 1 and Dt > δ) or (Aˆt < 0 and rt < 1 and Dt > δ)
- 1, otherwise.

where Dt is binary approximation of KL or TV as defined in Section 4.4. For DPPO-Binary-KL, δ = 0.05 for all scaling experiments. For DPPO-Binary-TV, we use δ = 0.15 for MoE Base w/ LoRA experiment and δ = 0.2 for all other scaling experiments.

Table 1. Detailed RL training hyperparameters of scaling experiments.

Hyperparameters MoE Base MoE Base w/ R3 MoE Thinking Dense Base MoE Base w/ LoRA

max prompt length 1024 1024 1024 1024 1024 max response length 16384 16384 16384 8000 8000 train batch size 256 256 256 128 128 ppo mini batch size 32 32 32 32 16 optim.lr 1e-6 1e-6 1e-6 1e-6 1e-5 rollout.temperature 1.0 1.0 1.0 1.0 1.0 rollout.n 16 16 16 8 8

Detailed Results Figure 13 Figure 14 Figure 15 Figure 16 Figure 17

Evaluation Settings. We perform online evaluation for each method and experimental configuration, monitoring AIME24 and AIME25 scores throughout RL training. Evaluations are conducted every 5 training steps for MoE Base, MoE Base w/ R3, and MoE Thinking, and every 10 steps for Dense Base and MoE Base w/ LoRA.

Across all scaling experiments, we use consistent sampling parameters: temperature=0.7, top p=0.95, and n=32. The n=32 setting indicates that each question from AIME24 and AIME25 is sampled 32 times, and we report the average scores. The max response length remains identical to that used during training rollouts.

|AIME 20| | | | | |25| | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
|Entrop| | | | | |y| | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

###### Rewards

###### AIME 2024

0.60

0.45

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.8

0.55

0.40

0.7

0.50

0.35

0.45

0.6

0.30

0.40

0.5

0.25

0.35

0.20

0.4

0.30

GRPO-ClipHigher

0.15

CISPO

0.25

0.3

DPPO-Binary-KL DPPO-Binary-TV

0.10

0.20

0.2

0.15

0.05

###### Mean of | - |

###### Response Length

0.010

10000

2000

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1750

0.008

8000

1500

1250

0.006

6000

1000

0.004

4000

750

500

0.002

2000

250

0.000

0

0

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training Step

Training Step

Training Step

Figure 13. Evolution of metrics for MoE Base w/o R3 experiment (based on Qwen3-30B-A3B-Base, without rollout router replay).

### F. More Empirical Results

###### F.1. AlpacaEval 2.0 Evaluation for RLHF

To complement the RLHF experiments, we also evaluate the Qwen3-4B-Instruct-2507 model fine-tuned on UltraFeedback with Skywork-Reward-Llama-3.1-8B on AlpacaEval 2.0. In this run, DPPO also obtains higher learned reward than GRPO at step 150 (51.32 vs. 36.70) and step 450 (70.27 vs. 45.24). We report the checkpoint at step 150 for AlpacaEval, since later checkpoints obtain lower AlpacaEval scores despite higher reward, indicating reward hacking.

Table 2. AlpacaEval 2.0 results for RLHF fine-tuning on Qwen3-4B-Instruct-2507 with UltraFeedback. We report length-controlled win rate (LC-WR), raw win rate (WR), and average response length.

Model LC-WR WR Avg. Length

Initial model 59.39 69.38 3147 GRPO fine-tuning 77.05 72.20 1756 DPPO fine-tuning 80.90 79.93 2003

As shown in Table 2, DPPO achieves the highest length-controlled and raw win rates among the compared checkpoints, establishing a new SOTA on the AlpacaEval 2.0 community leaderboard. This indicates that the reward improvement from DPPO transfers to an external open-ended alignment benchmark, rather than only increasing the training reward model score.

###### F.2. Extended Main Results

In addition to the results provided in Section 7, here we provide more detailed results of the five scaling experiments:

- Figure 13 for MoE Base w/o R3, Figure 14 for MoE Base w/ R3, Figure 15 for MoE Thinking, Figure 16 for Dense Base,

- Figure 17 for MoE Base w/ LoRA. We record the following metrics throughout the RL training: training rewards (denoted

as “Rewards”), AIME 2024 Avg@32 scores, AIME 2025 Avg@32 scores, mean of |µθ′ −πθ′ | (denoted as “Mean of |π −µ|”), mean of the response length (denoted as “Response Length”), and mean of token entropy (denoted as “Entropy”). For clearer visualization, all metrics except AIME24 and AIME25 are smoothed using a Gaussian filter with standard deviation σ = 2. The original unsmoothed curves are shown in the background as shaded regions.

Overall, across the five experiments, our method DPPO demonstrates consistent and robust improvements in training rewards,

###### Rewards

###### AIME 2024

###### AIME 2025

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.60

0.8

0.40

0.35

0.7

0.50

0.30

0.6

0.25

0.40

0.5

0.20

0.4

R3-GRPO-ClipHigher

0.30

0.15

R3-CISPO

0.3

R3-DPPO-Binary-KL

0.10

R3-DPPO-Binary-TV 0.20

0.2

0.05

###### Mean of | - |

###### Response Length

Entropy

0.006

10000

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

1000

0.005

8000

800

0.004

6000

600

0.003

4000

400

0.002

2000

200

0.001

0.000

0

0

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training Step

Training Step

Training Step

Figure 14. Evolution of metrics for MoE Base w/ R3 experiment (based on Qwen3-30B-A3B-Base, with rollout router replay).

highlighting its stability and efficiency. On both AIME 24 and AIME 25 benchmarks, DPPO exhibits a clear, stable upward trend during training and maintains superior performance after convergence. The stability of our approach is evidenced by learning curves that generally show less fluctuation compared to baseline methods. Its efficiency is reflected in the rapid increase of training rewards and the strong final performance.

DPPO variants consistently demonstrate healthy training dynamics. The training-inference mismatch (measured by the mean absolute deviation |π − µ|) and policy entropy remain within a stable, proper region throughout RL training. DPPO also effectively increases the generated response length across all scaling experiments, except for MoE Thinking. We note that the model Qwen3-30B-A3B already produces extremely long responses; as our training enforces a maximum length of approximately 16k tokens, RL training naturally shortens responses to fit this constraint.

In contrast, the GRPO-ClipHigher baseline, which relies on the ratio clipping mechanism of PPO, shows lower stability than DPPO and achieves inferior final performance in all five large-scale experiments. For example, in MoE Base w/o R3 (see Figure 13), GRPO-ClipHigher, though more stable than CISPO, improves more slowly and converges to lower training rewards and AIME scores than DPPO. In MoE Thinking (see Figure 15), GRPO-ClipHigher suffers a significant training collapse. Notably, GRPO-ClipHigher consistently leads to excessively high entropy in all large-scale experiments, a phenomenon not observed with other methods.

The CISPO baseline, which retains gradients for all tokens, is generally less stable and prone to collapse in certain settings. For instance, in MoE Base w/o R3 (see Figure 13), CISPO experiences a sudden and severe collapse leading to complete failure. In Dense Base (see Figure 16), CISPO shows a degenerative trend, particularly on AIME25. In MoE Base w/ LoRA (see Figure 17), the AIME24 scores, mean of |π − µ|, and response length exhibit noticeable fluctuations, further indicating instability.

We also analyze the effect of rollout router replay (R3). Remarkably, DPPO variants without R3 already outperform baselines that use R3, underscoring the importance of a proper masking mechanism in RL training (see Figures 13 and 14). Furthermore, incorporating R3 yields additional gains for DPPO, suggesting that the benefits of R3 and DPPO are largely orthogonal. This implies that DPPO provides a robust foundation for LLM RL fine-tuning, capable of further improvement even when training-inference mismatch is mitigated by other techniques.

###### Rewards

###### AIME 2024

###### AIME 2025

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

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.68

0.78

0.925

0.66

0.76

0.900

0.64

0.74

0.875

0.62

0.72

0.850

0.60

0.70

0.825

0.58

A3B-GRPO-ClipHigher

0.68

A3B-CISPO

0.800

0.56

A3B-DPPO-Binary-KL A3B-DPPO-Binary-TV

0.66

0.775

0.54

0.64

###### Mean of | - |

###### Response Length

Entropy

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
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

10000

0.009

2500

0.009

9000

2000

0.008

8000

0.007

1500

0.007

7000

1000

0.007

0.006

6000

500

0.005

0 50 100 150 200 250

0 50 100 150 200 250

0 50 100 150 200 250

Training Step

Training Step

Training Step

Figure 15. Evolution of metrics for MoE Thinking experiment (based on Qwen3-30B-A3B).

###### Rewards

###### AIME 2024

###### AIME 2025

0.30

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.35

0.7

0.25

0.6

0.30

0.5

0.25

0.20

0.4

0.20

0.15

8B-GRPO-ClipHigher

0.3

8B-CISPO

0.15

8B-DPPO-Binary-KL

0.10

8B-DPPO-Binary-TV 0.10

0.2

###### Mean of | - |

###### Response Length

Entropy

6000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

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

1750

0.025

1500

5000

0.020

1250

4000

0.015

1000

3000

750

0.010

500

2000

0.005

250

1000

0

0 200 400 600 800 1000 1200

0 200 400 600 800 1000 1200

0 200 400 600 800 1000 1200

Training Step

Training Step

Training Step

Figure 16. Evolution of metrics for Dense Base experiment (based on Qwen3-8B-Base).

###### Rewards

###### AIME 2024

###### AIME 2025

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.8

0.35

0.50

0.7

0.45

0.30

0.6

0.40

0.25

0.35

0.5

0.20

0.30

0.4

LoRA-GRPO-ClipHigher

0.15

0.25

LoRA-CISPO

0.3

LoRA-DPPO-Binary-KL LoRA-DPPO-Binary-TV

0.10

0.20

0.2

0.05

###### Mean of | - |

###### Response Length

Entropy

0.020

300

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

4500

0.018

250

4000

0.016

3500

0.014

200

3000

0.012

150

0.010

2500

100

0.008

2000

0.006

1500

50

0.004

1000

0

0 100 200 300 400 500 600 700 800 900

0 100 200 300 400 500 600 700 800 900

0 100 200 300 400 500 600 700 800 900

Training Step

Training Step

Training Step

Figure 17. Evolution of metrics for MoE Base w/ LoRA experiment (based on Qwen3-30B-A3B-Base, with LoRA).

|AIME 20| | | | | |25| | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
|Entrop| | | | | |y| | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

###### Rewards

###### AIME 2024

0.60

0.45

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.8

0.55

0.40

0.50

0.7

0.35

0.45

0.6

0.30

0.40

0.25

GRPO-ClipHigher

0.5

0.35

CISPO

0.20

0.30

0.4

DPPO-Binary-KL DPPO-Binary-TV

0.15

0.25

0.3

DPPO-TopK-KL DPPO-TopK-TV

0.10

0.20

0.2

0.15

0.05

###### Mean of | - |

###### Response Length

0.008

10000

2000

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1750

0.007

8000

1500

0.006

1250

6000

0.005

1000

4000

750

0.004

500

2000

0.003

250

0.002

0

0

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training Step

Training Step

Training Step

- Figure 18. Evolution of metrics for baselines, DPPO with binary TV/KL approximation, and DPPO with Top-K (K=20) approximation under the same setting as MoE Base w/o R3.

DPPO-TV-Binary

DPPO-KL-Binary

GRPO

GRPO-low0.2-high0.28

0.50

GRPO-low0.2-high0.2

0.45

###### AIME2024

0.40

0.35

0.30

0.25

DPPO-TV-Binary-0.15 DPPO-TV-Binary-0.10 DPPO-TV-Binary-0.20

DPPO-KL-Binary-0.05 DPPO-KL-Binary-0.10 DPPO-KL-Binary-0.15

0.20

0.15

0.40

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

0.35

###### AIME2025

0.30

0.25

0.20

0.15

0.10

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Training step

Training step

Training step

- Figure 19. Hyperparameter sensitivity of DPPO-Binary-TV, DPPO-Binary-KL, and GRPO on Qwen3-30B-A3B-Base trained on DAPO with 8k context length. DPPO remains robust across a broad range of thresholds, while GRPO is more sensitive to the clipping range.

###### F.3. Ablation on Divergence Approximation

In the scaling experiments, we compared DPPO variants using binary TV/KL approximations (Equations 13 and 14) against several baselines. To further investigate the approximation strategy, we experiment with DPPO variants with top-K TV/KL approximations (Equations 15 and 16), where we set K = 20; these variants are denoted as DPPO-TopK-TV and DPPO-TopK-KL. The choice K = 20 is limited by vLLM (Kwon et al., 2023), which supports returning log probabilities for at most 20 candidate tokens per step. We strictly replicate the experimental setting of MoE Base w/o R3. As in the main scaling experiments, for DPPO-Binary-TV and DPPO-TopK-TV we set the clip threshold δ = 0.2, while for DPPO-Binary-KL and DPPO-TopK-KL we set δ = 0.05.

As presented in Figure 18, introducing the top-K approximation does not yield significant performance gains, indicating that the simpler binary approximation already provides a sufficient and efficient proxy for constructing the trust region. This finding is encouraging, suggesting that DPPO with binary TV/KL remains highly scalable without sacrificing effectiveness.

###### F.4. Hyperparameter Sensitivity

We further examine the sensitivity of the divergence threshold δ in DPPO. We fine-tune Qwen3-30B-A3B-Base on the DAPO dataset with an 8k context length and compare DPPO-Binary-TV with δ ∈ {0.10,0.15,0.20}, DPPO-Binary-KL with δ ∈ {0.05,0.10,0.15}, and GRPO variants using ϵlow = 0.2 with different upper clipping values.

As shown in Figure 19, DPPO is relatively insensitive within the tested ranges. DPPO-Binary-TV performs comparably for δ ∈ [0.10,0.20], and DPPO-Binary-KL remains strong for δ ∈ [0.05,0.15]. In contrast, the GRPO curves vary more noticeably with the upper clipping parameter, and all tested DPPO configurations outperform the GRPO baselines.

###### F.5. Extended Results for Different Model × Task Combinations

Besides experimental results presented in Section 7, we evaluate DPPO on more model × task settings to validate its advantage over the GRPO baseline. The settings we considered include:

- 1. Different model family. Training on a new model different from the Qwen family, OctoThinker-3B-Hybrid-Base (Wang et al., 2025b), on the standard math reasoning dataset (Hendrycks et al., 2021).
- 2. Abstract reasoning and induction. Training the Qwen3-1.7B-Base model on abstract reasoning task (Arc1D) and induction task (Acre) from the Gem library (Liu et al., 2025e).

- 3. Multi-turn reasoning. Training the Qwen3-1.7B-Base model on the multi-turn reasoning environment (Sudoku-v0easy) from Gem the library (Liu et al., 2025e).

The training is conducted using Oat (Liu et al., 2025c) with their example scripts (thereby the standard hyper-parameters) for math RL and Gem RL. For the TV divergence clipping, we use a threshold of δ = 0.2. Figure 20 shows the comparison between the TV variant of DPPO and the vanilla ratio-based PPO, both based on the GRPO algorithmic framework with the only difference being the trust region masking strategy. We can observe DPPO improves the efficiency (and sometimes asymptotic performance) over the baseline across different settings, validating its general effectiveness.

###### OctoThinker

###### Arc1D

###### Acre

0.50 Sudoku

0.70

0.80

0.60

0.40

0.40

0.70

0.50

0.30

0.60

0.30

0.40

0.50

0.20

0.20

0.30

0.40

0.10

0.20

0.30

0.10

DPPO-Binary-TV

0.00

0.10

0.20

PPO-Ratio

0.00

0.00

-0.10

0 50 100 150 200 250 300 350

0 20 40 60 80 100 120

0 20 40 60 80 100 120

0 50 100 150 200 250 300 350 400

Training step

Training step

Training step

Training step

- Figure 20. Learning curve comparison of using ratio (PPO-Ratio) and TV divergence (DPPO-Binary-TV) for the trust region clipping.

