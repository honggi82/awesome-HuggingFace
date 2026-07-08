# arXiv:2512.15687v1[cs.LG]17Dec2025

## Can LLMs Guide Their Own Exploration? Gradient-Guided Reinforcement Learning for LLM Reasoning

Zhenwen Liang1, Sidi Lu1, Wenhao Yu1, Kishan Panaganti1, Yujun Zhou1,2, Haitao Mi1, Dong Yu1

1Tencent AI Lab, 2University of Notre Dame Correspondence to: zhenwzliang@global.tencent.com

##### Abstract

Reinforcement learning has become essential for strengthening the reasoning abilities of large language models, yet current exploration mechanisms remain fundamentally misaligned with how these models actually learn. Entropy bonuses and external semantic comparators encourage surface-level variation but offer no guarantee that sampled trajectories differ in the update directions that shape optimization. We propose G2RL, a gradient-guided reinforcement learning framework in which exploration is driven not by external heuristics but by the model’s own first-order update geometry. For each response, G2RL constructs a sequence-level feature from the model’s final-layer sensitivity—obtainable at negligible cost from a standard forward pass—and measures how each trajectory would reshape the policy by comparing these features within a sampled group. Trajectories that introduce novel gradient directions receive a bounded multiplicative reward scaler, while redundant or off-manifold updates are deemphasized, yielding a self-referential exploration signal that is naturally aligned with PPO-style stability and KL control. Across math and general reasoning benchmarks (MATH500, AMC, AIME24, AIME25, GPQA, MMLUPRO) on Qwen3-base 1.7B/4B models, G2RL consistently improves pass@1, maj@16, and pass@k over entropy-based GRPO and external-embedding methods. Analyzing the induced geometry, we find that G2RL expands exploration into substantially more orthogonal—and often opposing—gradient directions while maintaining semantic coherence, revealing that a policy’s own update space provides a far more faithful and effective basis for guiding exploration in LLM RL.

##### 1 Introduction

Reinforcement learning (RL) has become a central mechanism for improving the reasoning and decision-making abilities of large language models (LLMs), extending their capabilities beyond supervised finetuning (Christiano et al., 2017; Ouyang et al., 2022; Rafailov et al., 2023). Yet, despite this progress, exploration in LLM RL remains fundamentally underdeveloped. Current exploration strategies—entropy bonuses, outcome rarity, or external semantic comparators—are all driven by signals extrinsic to the model. They encourage the policy to sample more widely, but they do so without regard for the model’s own update structure. As a result, exploration often becomes diffuse, misaligned, or fragile under sparse reward signals, especially when supervision is binary or verifiable (Sutton et al., 1998; Auer et al., 2002; Kakade & Langford, 2002).

Entropy increases randomness but is oblivious to whether two responses induce meaningfully different directions. External similarity models based on semantic embeddings provide sequence-level contrast but operate in representation spaces that differ from the model’s internal geometry. A trajectory may appear “novel” semantically yet offer no new gradient information for the policy; conversely, trajectories essential for improving the model’s reasoning may be penalized for superficial semantic similarity. In short, existing methods guide exploration through lenses external to the policy, producing an enduring mismatch between exploration signals and the optimization dynamics that actually govern learning.

#### GRPO EVOL-RL G2RL

[Figure 3]

[Figure 4]

[Figure 5]

Semantic Diversity Gradient Diversity

Semantic Diversity Gradient Diversity

Semantic Diversity Gradient Diversity

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 1: Comparison of the characteristics of GRPO, EVOL-RL, and G2RL. The dispersion of points indicates semantic variety, while arrows represent gradient directions. Only G2RL explicitly encourages exploration aligned with the model’s intrinsic update geometry.

This motivates a more principled question: Can an LLM learn to explore by examining how each trajectory would reshape its own parameters, rather than relying on external proxies? Such an approach would shift exploration from being externally imposed to being self-guided: trajectories are preferred when they meaningfully expand the policy’s own update directions, and discouraged when they contribute redundant or uninformative gradients.

We propose G2RL, a Gradient-guided reinforcement learning framework that realizes this idea. Instead of encouraging diversity in the output space, G2RL encourages exploration directly in the policy’s update geometry. For each generated response, we construct a sequence-level feature derived from the model’s own first-order sensitivity at the final layer—available at negligible cost from a standard forward pass. This feature summarizes how the trajectory would steer the model’s output distribution through its gradients. By comparing these features within a group of candidates, we obtain a policy-intrinsic exploration score: responses that introduce new gradient directions are upweighted through a bounded reward scaler, whereas responses that merely reinforce existing update directions receive less emphasis.

This gradient-guided mechanism offers three conceptual advantages. First, it grounds exploration in the same geometry that governs optimization, eliminating the semantic–optimization mismatch inherent in external comparators. Second, it provides a self-referential criterion: the model explores what it stands to learn from, not what appears diverse to an auxiliary encoder. Third, the construction integrates seamlessly into GRPO (Shao et al., 2024), requiring no extra backward passes.

We evaluate G2RL across math and general reasoning benchmarks and multiple LLM scales, finding consistent improvements in pass@1, maj@16, and pass@k. Beyond raw accuracy, G2RL produces richer reasoning trajectories and more meaningful gradient dispersion, supporting the central hypothesis that exploration should be guided not by external heuristics but by the policy’s own update geometry.

###### Contributions.

- • We introduce G2RL, a gradient-guided RL method that defines exploration through the model’s own first-order update geometry, avoiding reliance on entropy or external semantic encoders.
- • We propose a bounded, groupwise reward-scaling mechanism that emphasizes trajectories offering novel gradient directions while preserving optimization stability.
- • Experiments on math and general reasoning tasks demonstrate that G2RL systematically improves single-sample accuracy and multi-sample coverage, achieving healthier and more structurally meaningful exploration dynamics.

##### 2 Method

We introduce G2RL, a gradient-guided reinforcement learning method for large language models (LLMs) that augments group-relative policy optimization (GRPO) with an exploration signal

computed in the policy’s own update space. Rather than rewarding diversity in output space, G2RL adjusts the contribution of each trajectory according to how it reshapes the model’s gradient directions. This section formalizes the setting, recalls GRPO, derives the gradient features, defines a groupwise gradient-guided exploration score, and integrates it into a stable PPO-style objective with KL control. All symbols are summarized as they appear.

###### 2.1 Preliminaries and Group-Relative Policy Optimization

Let X be a set of prompts and Y the response space. For x ∈ X, a response is the token sequence y = (y1, . . . , yL) ∈ Y. An autoregressive LLM with parameters θ defines

L

∏

pθ(yt | x, y<t) . (1)

pθ(y | x) =

t=1

We generate, for each prompt x, a group of m candidate responses {y(i)}im=1 from a fixed behavior policy πθold (autoregressive or nucleus sampling). Each response receives a base scalar reward r(i) ∈ R, e.g., a verifiable pass/fail or a task-specific score.

GRPO (Shao et al., 2024) dispenses with a learned critic and estimates groupwise advantages by standardizing rewards within the group:

r¯ =

m

1 m

r(i), sr =

∑

i=1

with ε > 0 for numerical stability. Let

m

1 m

r(i) − r¯ 2 + ε, A(i) =

∑

i=1

r(i) − r¯ sr

, (2)

πθ y(i) | x πθold y(i) | x

ρ(i) =

, clip(u;1 − ϵ,1 + ϵ) = min max{u,1 − ϵ}, 1 + ϵ , (3)

and let DKL(πθ(· | x) ∥ πref(· | x)) be a per-prompt KL penalty to a reference policy (e.g., the SFT model). The GRPO objective is

m

1 m

min ρ(i)A(i), clip ρ(i);1 − ϵ,1 + ϵ A(i) − β DKL(πθ∥πref) .

∑

JGRPO(θ) = Ex,{y(i)}∼π

θold

i=1

(4) The clipping term stabilizes policy updates, while the KL regularizer prevents drift from the reference policy. G2RL modifies only the way advantages are constructed, leaving PPO-style clipping and KL control unchanged.

###### 2.2 Gradient Features: Policy-Referential Sensitivity Let ht ∈ Rdh denote the final-layer hidden state at time t, and let the LM head be linear:

zt T

zt = W⊤ht + b ∈ RV, pt = softmax

, (5)

where W ∈ RV×dh, b ∈ RV, vocabulary size V, and temperature T > 0. Writing e(yt) ∈ {0,1}V for the one-hot of the realized token, the standard score-function identity yields the exact token-level gradient with respect to ht:

∂ log pθ(yt | x, y<t) ∂ht

1 T

W e(yt) − pt ∈ Rdh. (6) Define the token residual rt := e(yt) − pt and the token feature

=

ϕt := Wrt ∈ Rdh. (7)

Up to the constant 1/T, ϕt is the exact first-order sensitivity of the log-likelihood to perturbations at ht. Aggregating over the response produces a sequence-level feature

L

αt ∑sL=1 αs + ε

### ∑

α˜t ϕt, α˜t =

Φ(x, y) =

, (8)

t=1

with default uniform weights αt ≡ 1 and masking for non-response tokens if applicable.

Interpretation. By (5)–(6), any infinitesimal perturbation δht induces δ log pθ(yt | ·) ≈ ⟨ϕt/T, δht⟩, so the sequence feature Φ(x, y) summarizes, to first order, how the response (x, y) would steer the model’s output distribution through its final-layer representation. As detailed in Appendix A.1, the full parameter gradient along a trajectory factors through this feature: for every layer k there exists a trajectory-dependent linear operator Lk(x, y) such that

1 T Lk(x, y) Φ(x, y).

∇θkℓ(x, y) =

Thus, all upstream updates lie in linear images of the same sequence feature Φ(x, y), and angular relations between responses in Φ-space are propagated—up to layerwise linear transforms—to the actual optimization directions. We use these features not because the last layer dominates all upstream effects, but because it forms the unique first-order sensitivity bottleneck through which trajectory-specific information must pass, and it is available from the forward pass without any extra backpropagation. This makes cosine geometry on Φ a principled and computationally cheap proxy for comparing how different trajectories guide the policy’s updates.

###### 2.3 Groupwise Gradient-Guided Exploration Score

Given a group {Φ(i)}im=1, we first unit-normalize the sequence features and define pairwise cosine similarities:

Φ(i) ∥Φ(i)∥2 + ε

Φˆ (i) =

, Sij = Φ ˆ (i), Φˆ (j) ∈ [−1,1]. (9)

Next, we construct reward-weighted coefficients

exp r(j) 1{j ̸= i} ∑k̸=i exp r(k) + ε

, (10)

wij =

which form a probability distribution over the other candidates in the group: wij ≥ 0 and ∑j̸=i wij =

- 1. Higher-reward responses therefore act as more important reference directions when we assess the contribution of y(i). Using these ingredients, we define a bounded, scale-invariant gradient-guided exploration score:

ν(i) = max 1− ∑

wij Sij2, 0 ∈ [0,1]. (11)

j̸=i

Intuitively, the term ∑j̸=i wij Sij2 measures how well the direction Φˆ (i) can be “explained” by a weighted combination of the other responses’ gradient directions. If Φˆ (i) is almost parallel to several high-reward peers, the weighted squared cosine similarities Sij2 are large and their sum approaches 1; in that case, 1 − ∑j̸=i wijSij2 is small and ν(i) is close to 0, indicating that y(i) contributes little new information to the update geometry. Conversely, if Φˆ (i) is nearly orthogonal to most high-reward responses, all Sij2 are small, the weighted sum is far below 1, and ν(i) is close to 1. Thus ν(i) can be read as the “remaining” component of the update direction for y(i) that is not captured by the dominant, high-reward directions in the group. Because we work with unit vectors in (9), this score is invariant to any common rescaling of the underlying features Φ(i).

###### 2.4 Gradient-Guided Reward Shaping

We convert the exploration score into a multiplicative reward factor that preserves optimization stability. Let ν¯(i) denote a bounded, monotone transformation of ν(i) (e.g., min–max normalization within the group). We define

f ν(i),r(i) = 1 + λ r(i) ν ¯(i), λ ∈ [0, λmax], (12) and the gradient-guided reward is

r˜(i) = r(i) · f ν(i),r(i) . (13)

This formulation induces an asymmetric effect. For correct responses (r(i) = 1), high exploration score boosts the reward, prioritizing trajectories that follow successful yet geometrically distinct update directions over redundant repetitions. For incorrect responses (r(i) = −1), high exploration score amplifies the penalty (r˜(i) < −1), while low exploration score (high similarity to correct peers) mitigates the penalty (r˜(i) > −1). An incorrect response with low exploration score has its gradient feature Φ(i) aligned with the subspace of correct solutions, suggesting a “near-miss” whose update direction is still informative; by penalizing these less, the policy is encouraged to stay within a plausible reasoning manifold. Conversely, an incorrect response with high exploration score is nearly orthogonal to successful trajectories, indicating off-manifold or hallucinated behavior that should be suppressed.

###### 2.5 Practical Reward Rescaling

In all our experiments the base rewards are binary, r(i) ∈ {−1,1}, indicating incorrect vs. correct responses. The gradient-guided factor in (12) modifies these rewards but we keep the overall scale tightly controlled. Concretely, we instantiate the mapping f(ν(i),r(i)) so that it is bounded and monotone in the normalized exploration score, and we clip the shaped reward in (13) to a fixed interval:

r˜(i) ← clip r ˜(i); −c, c , c = 3. Since r(i) ∈ {−1,1}, this guarantees that the effective reward magnitude seen by the policy always satisfies |r˜(i)| ≤ 3, so the gradient-guided term can at most moderately up- or down-weight individual samples. This keeps the advantage scale stable while still allowing the exploration signal to reshape the relative weighting of candidates within each group.

###### 2.6 Computation and Implementation

Token features without extra backprop. Equation (6) depends only on quantities from the forward pass. Computing ϕt = Wrt can be implemented as

ϕt = W e(yt) − W pt = W(:,yt) − Ev∼pt W(:,v) , (14) i.e., a column gather and a matrix–vector product with pt. Aggregating ϕt over t yields Φ(x, y) with negligible overhead relative to the softmax and log-prob computations already performed. In practice, G2RL is therefore a drop-in modification of GRPO that replaces groupwise advantages by gradient-guided ones while leaving the rest of the RL pipeline unchanged.

##### 3 Experiments

We evaluate the proposed G2RL on two Qwen3 base models (1.7B and 4B). Our gradient-guided exploration term is applied as reward shaping within the groupwise standardization. We use the MATH training set yielding 7.5k training problems (Hendrycks et al., 2021). A rule-based verifier provides both the RL reward signal during training and the correctness oracle at evaluation time.

###### 3.1 Benchmarks and Metrics

We report results on AIME24, AIME25, MATH500, AMC, and GPQA. For each prompt we report pass@1, maj@16 and pass@16. pass@k is the fraction of prompts for which at least one of the k samples is verified correct. maj@16 is the majority-vote accuracy over 16 samples: we select the most frequent final answer string among the 16 and verify that single prediction (ties broken uniformly). All numbers are percentages; bold indicates the best within each block.

###### 3.2 Main Results

- Table 1: Main results on MATH500, AMC, AIME24, AIME25 with Qwen3-1.7B-Base and Qwen3-4BBase backbone.

MATH500 AMC AIME24 AIME25 Model pass@1 maj@16 pass@16 pass@1 maj@16 pass@16 pass@1 maj@16 pass@16 pass@1 maj@16 pass@16 Qwen3-1.7B-Base

GRPO 63.5 73.2 86.6 31.2 42.0 68.1 7.5 14.8 24.4 4.6 6.8 22.2 Entropy Bonus 64.3 74.1 86.7 32.2 42.3 65.7 9.6 17.2 23.3 4.6 6.9 24.5 EVOL-RL 64.3 73.7 86.9 32.2 43.9 66.2 8.4 15.8 27.3 5.3 7.9 21.6 G2RL 66.2 76.8 88.7 33.9 44.8 68.5 10.1 17.4 28.0 7.5 11.4 24.8

Qwen3-4B-Base

GRPO 76.9 81.6 90.8 47.9 56.2 75.1 12.4 18.2 31.1 10.0 16.2 32.5 Entropy Bonus 79.0 87.2 93.2 50.5 63.7 79.5 17.8 25.4 40.0 16.1 24.4 41.5 EVOL-RL 80.0 87.7 93.5 50.9 62.0 81.9 19.4 28.2 42.4 17.5 23.9 39.8 G2RL (Ours) 80.8 87.8 93.6 52.3 63.8 82.0 19.9 28.7 43.8 20.1 29.0 45.0

On 1.7B, the gradient-guided variant G2RL improves both single-try quality and multi-sample coverage across all datasets, indicating that exploration in the model’s own update space leads to more effective use of the sampling budget. The gains are largest on AIME25, where pass@1 rises to 7.5 and maj@16 to 11.4, indicating that G2RL encourages useful optimization-space exploration rather than mere noise. On MATH500, improvements are consistent through pass@16 (88.7 vs. 86.9 for the strongest baseline). The only case where a baseline slightly edges out G2RL is AMCpass@16 (68.1 vs. 68.5), while G2RL leads on the remaining metrics for that dataset.

The 4B results amplify these trends. On AIME25, pass@1 improves from 17.5 (best baseline) to 20.1 and maj@16 from 23.9 to 29.0. On MATH500, gains are smaller but consistent across all metrics, reaching pass@16 = 93.6. On AMC, G2RL maintains the best pass@1, maj@16, and pass@16, indicating stronger sample efficiency and coverage at scale.

Across both model sizes, G2RL improves pass@1 on every dataset, showing that gradient-guided exploration shifts probability mass toward higher-quality solutions. Gains in pass@16 demonstrate better coverage of distinct correct modes, especially on challenging splits like AIME25.

###### 3.3 Analysis on General Reasoning Tasks

We assess generalization on two broad-coverage reasoning benchmarks using the 4B model. On GPQA, which is four-option multiple-choice, sampling benefits are pronounced: pass@k rises with k. Our method improves single-try quality and consensus (pass@1 = 38.7, maj@16 = 44.0) and achieves the best pass@16 = 89.2; at pass@32 the task is near-saturated and EVOL-RL is marginally higher (93.7 vs. 93.5). On the larger and more diverse MMLUpro, we report pass@1 only, our approach attains a higher micro-average (58.47) than EVOL-RL (57.17), entropy bonus (57.14), and GRPO (56.15). These results indicate that the gradient-guided exploration signal improves generalization beyond math-style settings, raising both single-try accuracy and useful coverage without auxiliary encoders.

- Table 2: General reasoning results on the backbone of Qwen3-4B-Base. Left: GPQA (multiple-choice) with full sampling metrics. Right: MMLUpro micro-average pass@1. All numbers are percentages; best in bold.

GPQA (4B) MMLUpro (4B, pass@1)

Method pass@1 maj@16 pass@16 pass@32 GRPO 37.2 39.2 81.2 85.2 Entropy Bonus 37.8 42.8 88.6 92.6 EVOL-RL 37.4 42.1 88.9 93.7 G2RL (Ours) 38.7 44.0 89.2 93.5

Method Micro Avg. GRPO 56.15 Entropy Bonus 57.14 EVOL-RL 57.17 G2RL (Ours) 58.47

- 3.4 Training dynamics

Key Finding 1

G2RL achieves the fastest and highest gains in accuracy and response length while keeping entropy moderate; entropy bonuses mainly inflate entropy and token count, and EVOL-RL shows healthy but ultimately weaker improvements due to its externally defined exploration signal.

AIME25 Accuracy (mean@8)

Average Response Length

Entropy

2750

2.00

G2RL

G2RL

0.20

2500

AverageLength(tokens)

1.75

Entropy Bonus

Entropy Bonus

0.18

EVOL-RL

EVOL-RL

2250

1.50

0.16

2000

1.25

Accuracy

Entropy

0.14

1750

1.00

0.12

1500

0.75

0.10

1250

0.50

0.08

G2RL

1000

Entropy Bonus

0.25

0.06

750

EVOL-RL

0.04

0.00

0 50 100 150 200

0 50 100 150 200

0 50 100 150 200

Training Steps

Training Steps

Training Steps

- Figure 2: Training dynamics on AIME25: mean@8 accuracy, average response length, and entropy for GRPO, Entropy Bonus, EVOL-RL, and G2RL.

To understand how different exploration mechanisms influence learning, we compare training dynamics for three methods: Entropy Bonus, EVOL-RL, and G2RL. We track AIME25 mean@8 accuracy, average response length, and output entropy over training steps (Figure 2).

Accuracy and response length. G2RL shows the steepest and most stable improvement in mean@8: its accuracy curve rises quickly and plateaus at the highest level among all methods. EVOL-RL improves more smoothly than entropy bonus, but still converges below G2RL. In parallel, G2RL drives a rapid early increase in average response length, indicating that the model quickly learns to produce richer, more structured reasoning rather than merely recycling short patterns. For Entropy Bonus, length is more volatile and less predictive of accuracy.

Entropy as a noisy proxy for exploration. Entropy Bonus unsurprisingly produces the largest increase in output entropy, but this growth is only weakly coupled to accuracy: entropy continues to rise even when accuracy saturates, suggesting that many additional tokens are uninformative for solving the task. G2RL, in contrast, yields a moderate but aligned entropy increase: entropy rises together with both accuracy and response length, reflecting exploration that contributes to useful reasoning. EVOL-RL behaves in between these extremes.

Why G2RL outperforms EVOL-RL. Both EVOL-RL and G2RL maintain “healthy” curves where accuracy and length co-evolve without obvious instability. The key difference lies in how exploration signal is defined. EVOL-RL relies on an external encoder whose similarity geometry is only loosely tied to the current policy, and thus cannot perfectly adapt to its evolving representation. G2RL instead measures exploration pressure directly in the policy’s gradient feature space, keeping the exploration signal aligned with the actual update directions and yielding faster learning and higher final accuracy.

- 3.5 Analysis of Exploration Geometry

Key Finding 2

G2RL alters the optimization landscape, increasing the ratio of opposing gradient directions (negative similarity) by nearly 5× compared to GRPO. Crucially, we observe a distinct misalignment between semantic-space and gradient-space geometry: external embeddings fail to capture the structural orthogonality that drives efficient exploration.

###### Gradient Cosine Similarity Distribution (Lower = More Diverse)

###### Semantic Cosine Similarity Distribution (Lower = More Diverse)

###### Negative Similarity Ratio (Higher = More Diverse)

100

Negative (GRPO) Negative (EVOL-RL) Negative (G2RL) Positive Similarity

1.0

1.0

| |
|---|

| |
|---|

0.8

| |
|---|

80

0.9

SemanticCosineSimilarity

GradientCosineSimilarity

0.6

71.9%

0.8

###### Percentage(%)

60

81.1%

0.4

94.1%

0.7

0.2

40

0.0

0.6

0.2

20

0.5

28.1%

18.9%

0.4

5.9%

0.4

0

GRPO EVOL-RL G2RL

GRPO EVOL-RL G2RL

GRPO EVOL-RL G2RL

Model

Model

Model

- Figure 3: Exploration-geometry analysis on AIME25. Left: Distribution of pairwise gradient-space cosine similarity. G2RL significantly shifts the distribution toward zero. Middle: Distribution of semantic-space cosine similarity. G2RL maintains high semantic coherence despite much more diverse gradient geometry. Right: Ratio of negative similarity pairs. G2RL generates nearly 5× more orthogonal or opposing optimization directions than vanilla GRPO.

To investigate the underlying mechanism of our method, we conducted a controlled analysis of the exploration geometry induced by different training strategies in both the policy’s native gradient space and an external semantic space. We sampled 8 responses for each of 30 randomly selected problems from the AIME25 validation set using three models: vanilla GRPO, EVOL-RL, and G2RL. We measured pairwise similarity using two metrics: Gradient Geometry (based on the policy’s own update features Φ) and Semantic Similarity (based on an external embedding model).

Gradient Geometry and Orthogonality. As shown in Figure 3 (Left), G2RL shifts the distribution of pairwise gradient similarities significantly toward zero. While vanilla GRPO responses exhibit high collinearity (mean cosine similarity 0.208), G2RL reduces this to 0.064, indicating a much broader coverage of the optimization space. Most notably, we analyze the Negative Similarity Ratio (Figure 3, Right), which tracks response pairs pointing in opposing directions. Vanilla GRPO produces only 5.91% negative pairs. In contrast, G2RL boosts this to 28.09%. This confirms that our gradient-guided exploration mechanism successfully drives the policy toward structurally distinct reasoning paths that offer complementary gradient information, rather than merely rephrasing similar solutions.

Misalignment in Semantic Space. The results in semantic space (Figure 3, Middle) reveal a critical insight. Vanilla GRPO actually yields lower semantic similarity (0.738) than G2RL (0.769). A naive interpretation might suggest GRPO is "more diverse." However, given GRPO’s inferior performance, this likely reflects incoherent or off-manifold exploration. G2RL maintains higher semantic consistency (staying on-topic and coherent) while simultaneously maximizing gradient orthogonality. This discrepancy shows that external semantic embeddings are an unreliable proxy for RL exploration: they may penalize subtle but update-relevant reasoning variations that are valuable in the policy’s own gradient space.

##### 4 Discussion: The Geometry of Efficient Exploration

Breaking Collinearity via Orthogonal Gradients. Standard GRPO treats all correct responses uniformly, assigning them identical positive advantages. Our analysis of gradient geometry reveals a structural flaw in this approach: under vanilla GRPO, successful trajectories exhibit high gradient collinearity (mean similarity 0.21). This implies that nominally “diverse” samples are often redundant in optimization space, pushing parameters along the same dominant direction and accelerating mode collapse. G2RL fundamentally alters this dynamic. By explicitly guiding exploration in the policy’s own sensitivity space, it does not merely spread samples out in output space; it encourages them to be functionally orthogonal in update space. Our experiments show a

###### 5× increase in negative similarity pairs compared to the baseline, suggesting that G2RL actively selects trajectories that provide complementary gradient information—effectively counterbalancing over-represented directions of the dominant mode and keeping the optimization landscape flat enough to permit continued, stable exploration.

Semantic vs. Optimization Geometry. A critical insight from our comparison with EVOL-RL is the misalignment between human-intelligible semantic variation and optimizer-relevant structural variation. Intuitively, one might expect lower semantic similarity to correlate with better exploration. However, our results show the opposite: G2RL maintains higher semantic consistency (0.77) than vanilla GRPO (0.74) while achieving drastically lower gradient similarity. This indicates that external embedding models are deceptive proxies for RL: they may reward surface-level changes (phrasing, irrelevant tangents) that contribute little to learning, or penalize subtle but high-value reasoning shifts that matter for optimization. G2RL succeeds because it operates directly in the policy’s intrinsic gradient geometry: it encourages variations that maximize the geometric difference in the parameter update step, regardless of whether those variations appear “semantically distinct” to an external encoder. In doing so, it effectively decouples useful exploration from random noise in output space.

Credit Assignment within Correctness Classes. Finally, G2RL addresses a subtle credit assignment ambiguity in sparse-reward settings. In binary-reward math tasks, the optimizer cannot distinguish between a fragile, pattern-matched solution and a robust, principled one if both happen to be correct. By modulating the reward with gradient-guided weights, G2RL implements a dynamic re-weighting scheme: it down-weights the “easy,” repetitive paths whose gradient directions are already crowded, and amplifies the signal from rarer trajectories that open new update directions. This allows the model to continue extracting learning signal from correct answers long after a standard policy gradient would have saturated on a single mode, and it does so by leveraging the model’s own update geometry as the reference for what constitutes informative exploration.

##### 5 Related Work

Exploration and diversity in reinforcement learning. A long line of work encourages exploration by maximizing entropy, which improves robustness and prevents premature convergence by discouraging early overcommitment to narrow modes (Haarnoja et al., 2018). Quality–diversity (QD) methods such as MAP-Elites extend this idea by simultaneously maintaining performance and

behavioral diversity, yielding repertoires of high-quality yet distinct solutions (Mouret & Clune, 2015; Pugh et al., 2016). In unsupervised RL, DIAYN maximizes mutual information between latent skills and states to acquire a set of diverse policies without external rewards (Eysenbach et al., 2018). Taken together, these approaches demonstrate that structured exploration is not merely auxiliary but central to stable learning and broad generalization. However, they typically reason about exploration in terms of entropy or behavior space, rather than in terms of the geometry of parameter updates.

Exploration-aware RL formulations for LLMs. Recent work on LLM reinforcement learning makes exploration explicit in the training objective. Song et al. (2025) diagnose diversity collapse under majority-style training and propose outcome-based exploration, which assigns bonuses to rare outcomes (historically or within-batch) to recover coverage without sacrificing accuracy. DARLING jointly optimizes a learned diversity signal with task reward, showing gains on both non-verifiable and competition-math settings (Li et al., 2025). Label-free formulations also emphasize exploration to avert collapse: EVOL-RL couples majority-based selection for stability with novelty-aware rewards for exploration, embodying a variation–selection principle (Zhou et al., 2025), while RESTRAIN turns spurious majority signals into penalties for overconfident, low-consistency rollouts, enabling self-driven RL that maintains healthy variability without gold labels (Yu et al., 2025).

Our work, G2RL, is aligned with this trajectory in that it modifies the RL objective to reshape exploration, but it differs in how the exploration signal is defined. Instead of relying on entropy or external embedding spaces to measure diversity, G2RL computes policy-referential similarity in the model’s own gradient feature space, avoiding auxiliary encoders and aligning the exploration signal directly with the policy’s update geometry.

##### 6 Conclusion

G2RL offers a principled and practical framework for guiding exploration in large language models by anchoring it to the policy’s own update geometry rather than to entropy or external semantic encoders. By deriving a sequence-level representation from last-layer gradient sensitivity and integrating it as a bounded, reward-coupled weighting within GRPO, the method selectively amplifies correct trajectories that introduce new optimization directions and suppresses incoherent ones, breaking gradient collinearity without compromising training stability. Across math and general reasoning benchmarks and two Qwen3-base model scales, this simple modification yields consistent gains in pass@1, maj@16, and pass@k, produces healthier training dynamics, and increases the prevalence of genuinely orthogonal gradient pairs by five-fold—while maintaining coherent, on-topic outputs. These results support a broader perspective: efficient exploration in LLM reinforcement learning is achieved not by inflating entropy or superficial semantic dispersion, but by shaping the geometry of the optimization landscape through gradient-guided, policy-intrinsic signals.

##### References

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. Pmlr, 2018.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the nineteenth international conference on machine learning, pp. 267–274, 2002.

Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, Jason Weston, Jack Lanchantin, and Tianlu Wang. Jointly reinforcing diversity and quality in language model generations. arXiv preprint arXiv:2509.02534, 2025.

Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites. arXiv preprint arXiv:1504.04909, 2015.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Justin K Pugh, Lisa B Soros, and Kenneth O Stanley. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40, 2016.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yuda Song, Julia Kempe, and Remi Munos. Outcome-based exploration for llm reasoning. arXiv preprint arXiv:2509.06941, 2025.

Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Zhaoning Yu, Will Su, Leitian Tao, Haozhu Wang, Aashu Singh, Hanchao Yu, Jianyu Wang, Hongyang Gao, Weizhe Yuan, Jason Weston, et al. Restrain: From spurious votes to signals– self-driven rl with self-penalization. arXiv preprint arXiv:2510.02172, 2025.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, and Dong Yu. Evolving language models without labels: Majority drives selection, novelty promotes variation. arXiv preprint arXiv:2509.15194, 2025.

##### A Appendix

###### A.1 Theoretical Motivation: Diversity in the Optimization Landscape

- A.1.1 First-Order Sensitivity at the Last Layer

For a generated trajectory (x, y), the contribution of token t to the log-likelihood gradient in last-layer space is

∇htℓt =

1 T

W e(yt) − pt =:

1 T

ϕt, (15)

where pt = softmax(W⊤ht/T). Collecting all token-level gradients yields the sequence-level sensitivity

G(h1:L) := ∇h1ℓ, . . . , ∇hLℓ =

1 T

(ϕ1, . . . , ϕL). (16)

- A.1.2 Layerwise Factorization of Backpropagation The network computation graph is

hk+1 = fk+1(hk; θk), k = 0, . . . , L − 1. (17) During backpropagation, the parameters are fixed. Define the Jacobians

∂hk+1 ∂hk

∂hk+1 ∂θk

Jk(+h)1 :=

, Jk(+θ)1 :=

. (18) Applying the chain rule repeatedly gives the per-token upstream gradient

∇θkℓt = Jk(+θ)1 ⊤ Jk(+h)2

⊤

· · · JL(h)

⊤

∇hLℓt. (19)

Equation (19) can be organized by defining the cumulative transition operator

Tk→L := Jk(+h)2Jk(+h)3 · · · JL(h) (20) (with TL→L = I), so that

∇θkℓt = Jk(+θ)1 ⊤Tk⊤→L∇hLℓt. (21) Substituting (15) yields the explicit upstream factorization:

1 T

Jk(+θ)1 ⊤Tk⊤→L

ϕt. (22)

∇θkℓt =

Lk(x,y)

Here

⊤

Lk(x, y) := Jk(+θ)1

Tk⊤→L (23) is a trajectory-dependent linear operator determined entirely by the forward pass activations. Equation (22) exhibits a complete structural decomposition:

|∇θkℓt ∈ Im Lk(x, y) · ϕt.|
|---|

(24)

Two structural sources of diversity. From (24), diversity in parameter updates across responses can arise only from:

- (i) Variation in ϕt (differences in last-layer sensitivity to tokens), (25)
- (ii) Variation in Lk(x, y) (differences in intermediate activations and Jacobians). (26)

Why shaping Φ is principled. For a response y, define a sequence-level feature such as

Φ(x, y) =

so that the full upstream gradient satisfies

L

### ∑

αtϕt, (27)

t=1

1 T Lk(x, y) Φ(x, y). (28)

∇θkℓ(x, y) =

Thus all optimization signals factor through the same structural pipeline:

###### Φ(x, y) −→ Lk(x, y) −→ ∇θkℓ(x, y).

Shaping the geometry of Φ therefore shapes the geometry of the entire family of upstream gradients in (28). In particular, promoting angular dispersion among Φ(x, y) encourages the resulting updates to explore a broader subspace of the parameter space, without making assumptions on the Jacobiansbecause Φ is the unique quantity through which all upstream gradients must factor linearly.

###### A.2 Training Configuration.

We conduct our experiments on two recent open-source base models: Qwen3-1.7B-Base and Qwen34B-Base. Our training process is implemented using the GRPO algorithm. To ensure that the model has sufficient capacity for complex, multi-step reasoning, we set the maximum response length to 8k and 12k tokens during generation for 1.7B and 4B models, respectively. To guide the model’s reasoning process, we utilize the system prompt from SimpleRL-Zoo (Zeng et al., 2025):

###### System Prompt

###### Please reason step by step, and put your final answer within \boxed{}.

Our training hyper-parameters are: Table 3: General hyper-parameters for RL training.

###### Hyperparameter Value

Train Batch Size 16 PPO Mini-Batch Size 1 (effective size of 32) PPO Micro-Batch Size 2 Rollouts 16 Generation Temperature 1.0 Validation Temperature 0.8 Learning Rate 5e-7 Use KL Loss True KL Loss Coefficient 0.001

